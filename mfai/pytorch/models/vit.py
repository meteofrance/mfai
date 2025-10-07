"""
VIT adapted from Lucidrain's repo https://github.com/lucidrains/vit-pytorch.
Added a multi-token output for multimodal LLMs.
"""

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from dataclasses_json import dataclass_json
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Size, Tensor, nn

from .base import AutoPaddingModel, BaseModel, ModelType

# helpers


def pair(t: torch.Size | tuple[int, int] | int) -> torch.Size | tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers: Iterable = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTCore(nn.Module):
    """
    Core ViT implementation without any classification or specific head.
    """

    def __init__(
        self,
        *,
        image_size: torch.Size | tuple[int, int] | int,
        patch_size: torch.Size | tuple[int, int] | int,
        emb_dim: int,
        n_layers: int,
        n_heads: int,
        mlp_dim: int,
        n_input_channels: int = 3,
        dim_head: int = 64,
        transformer_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        raise_on_size: bool = False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        if raise_on_size:
            assert (
                image_height % patch_height == 0 and image_width % patch_width == 0
            ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = n_input_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            emb_dim, n_layers, n_heads, dim_head, mlp_dim, transformer_dropout
        )

    def forward(self, img: Tensor) -> Tensor:
        # img shape = (B, features, h, w)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape  # (B, n_patches_h * n_patches_w = n, embed_dim)

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)  # (B, 1, embed_dim)
        # Add the "class token" before the sequence of patches:
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n + 1, embed_dim)
        x += self.pos_embedding  # (B, n + 1, embed_dim)
        x = self.dropout(x)

        return self.transformer(x)  # (B, n + 1, embed_dim)


@dataclass_json
@dataclass
class ViTEncoderSettings:
    patch_size: tuple[int, int] | int = 8
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 16  # Number of attention heads
    n_layers: int = 6  # Number of layers
    mlp_dim: int = 2048
    transformer_dropout: float = 0.1  # Dropout rate
    emb_dropout: float = 0.1  # Dropout rate
    autopad_enabled: bool = False  # Enable automatic padding


@dataclass_json
@dataclass
class ViTClassifierSettings(ViTEncoderSettings):
    pool: Literal["cls", "mean"] = "cls"  # Pooling method, either 'cls' or 'mean'


class VitPaddingMixin(AutoPaddingModel):
    """
    Mixin implementing the padding logic for ViT models.
    """

    def validate_input_shape(self, input_shape: Size) -> tuple[bool, Size]:
        """
        Check if the input shape is divisible by the patch size and returns the new shape if padding is required.
        """
        x_size, y_size = input_shape[-2:]
        patch_size = self.settings.patch_size

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        patch_x, patch_y = patch_size

        # checks if x requires padding
        x_padding = (patch_x - (x_size % patch_x)) % patch_x

        # checks if y requires padding
        y_padding = (patch_y - (y_size % patch_y)) % patch_y

        if x_padding == 0 and y_padding == 0:
            return True, input_shape
        else:
            new_shape = (x_size + x_padding, y_size + y_padding)
            return False, torch.Size(new_shape)


class ViTClassifier(BaseModel, VitPaddingMixin):
    """
    Vision Transformer (ViT) classifier model outputing class probabilities per input sample.
    THIS IS NOT A per pixel/grid classifier, but a global image/sample classifier.
    """

    settings_kls = ViTClassifierSettings
    onnx_supported: bool = False
    supported_num_spatial_dims: tuple[int, ...] = (2,)
    features_last: bool = False
    model_type: ModelType = ModelType.VISION_TRANSFORMER
    num_spatial_dims: int = 2
    register: bool = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: tuple[int, int] = (64, 64),
        settings: ViTClassifierSettings = ViTClassifierSettings(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        # we create fake data to get the input shape from our padding mixin
        # this is needed to initialize the ViTCore correctly
        fake_data = torch.zeros((1, in_channels, *input_shape))
        reshaped_data, _ = self._maybe_padding(data_tensor=fake_data)

        self.vit = ViTCore(
            image_size=reshaped_data.shape[-2:],
            patch_size=settings.patch_size,
            emb_dim=settings.emb_dim,
            n_layers=settings.n_layers,
            n_heads=settings.n_heads,
            mlp_dim=settings.mlp_dim,
            n_input_channels=in_channels,
            dim_head=settings.emb_dim // settings.n_heads,
            transformer_dropout=settings.transformer_dropout,
            emb_dropout=settings.emb_dropout,
        )

        self.pool = settings.pool
        self.mlp_head = nn.Linear(settings.emb_dim, out_channels)
        self.check_required_attributes()

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self._maybe_padding(data_tensor=x)
        x = self.vit(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.mlp_head(x)

    @property
    def settings(self) -> ViTClassifierSettings:
        """
        Returns the settings instance used to configure for this model.
        """
        return self._settings


class VitEncoder(BaseModel, VitPaddingMixin):
    """
    ViT vision encoder for multimodal LLMs.
    The number of output tokens is equal to the number of patches + 1.
    """

    settings_kls = ViTEncoderSettings
    onnx_supported: bool = False
    supported_num_spatial_dims: tuple[int, ...] = (2,)
    features_last: bool = False
    model_type: ModelType = ModelType.VISION_TRANSFORMER
    num_spatial_dims: int = 2
    register: bool = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: tuple[int, int] = (64, 64),
        settings: ViTEncoderSettings = ViTEncoderSettings(),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        # we create fake data to get the input shape from our padding mixin
        # this is needed to initialize the ViTCore correctly
        fake_data = torch.zeros((1, in_channels, *input_shape))
        reshaped_data, _ = self._maybe_padding(data_tensor=fake_data)

        self.vit = ViTCore(
            image_size=reshaped_data.shape[-2:],
            patch_size=settings.patch_size,
            emb_dim=settings.emb_dim,
            n_layers=settings.n_layers,
            n_heads=settings.n_heads,
            mlp_dim=settings.mlp_dim,
            n_input_channels=in_channels,
            dim_head=settings.emb_dim // settings.n_heads,
            transformer_dropout=settings.transformer_dropout,
            emb_dropout=settings.emb_dropout,
        )
        self.check_required_attributes()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of the ViT vision encoder.

        Args:
            x (Tensor): tensor of shape (B, features, height, width)

        Returns:
            Tensor: tensor of shape (B, n_patches_h * n_patches_w + 1, embed_dim)
        """
        x, _ = self._maybe_padding(data_tensor=x)  # (B, features, h, w)
        return self.vit(x)

    @property
    def settings(self) -> ViTEncoderSettings:
        """
        Returns the settings instance used to configure for this model.
        """
        return self._settings
