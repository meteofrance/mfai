from dataclasses import asdict, dataclass
from typing import Literal

import torch
from dataclasses_json import dataclass_json
from torch import Tensor, nn

from mfai.pytorch.models.llms.fuyu import FreezeMLMMixin
from mfai.pytorch.models.llms.gpt2 import CrossAttentionGPT2
from mfai.pytorch.models.resnet import (
    ResNet50MLM,
    ResNet50MLMSettings,
)
from mfai.pytorch.models.vit import VitEncoder, ViTEncoderSettings
from mfai.pytorch.models.weather_projector import (
    WeatherProjector,
    WeatherProjectorSettings,
)
from mfai.pytorch.namedtensor import NamedTensor


@dataclass_json
@dataclass
class XAttMultiModalLMSettings:
    """
    Settings for our cross attention multimodal language model.
    """

    emb_dim: int = 768  # Embedding dimension
    context_length: int = 1024  # Context length
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias
    vision_input_shape: tuple[int, int, int, int] = (
        256,
        256,
        10,
        10,
    )
    x_att_ratio: int = 4  # Cross attention layer ratio
    resnet_num_tokens: int = 32  # Number of vision/weather tokens
    # absolute positional embedding for the vision encoder
    resnet_pos_embedding: bool = False

    # mlp output for the vision encoder
    resnet_mlp_output: bool = False

    vision_encoder: Literal["resnet50", "linear", "vit"] = "linear"

    # layer norm visual/weather embeddings
    layer_norm_vis: bool = True

    patch_size: int | tuple[int, int] = 8


class XAttMultiModalLM(FreezeMLMMixin, nn.Module):
    """
    A multimodal LLM with cross attention.
    Can use GPT2 or Llama2 as its LLM backend.
    """

    def __init__(
        self,
        settings: XAttMultiModalLMSettings = XAttMultiModalLMSettings(),
        vocab_size: int = 50257,
    ):
        super().__init__()
        self.settings = settings

        # Initialize our X ATT GPT2 backend
        self.backend = CrossAttentionGPT2(
            CrossAttentionGPT2.settings_kls(
                **{
                    k: v
                    for k, v in asdict(settings).items()
                    if k in CrossAttentionGPT2.settings_kls.__dataclass_fields__
                }
            ),
            vocab_size=vocab_size,
        )

        self.vision_encoder: WeatherProjector | ResNet50MLM | VitEncoder

        if self.settings.vision_encoder == "linear":
            input_dims = (
                settings.vision_input_shape[0],
                settings.vision_input_shape[1],
                settings.vision_input_shape[-1],
            )
            s = WeatherProjectorSettings(
                input_dims=input_dims,
                embedding_dim=self.settings.emb_dim,
                patch_size=self.settings.patch_size,
            )
            self.vision_encoder = WeatherProjector(settings=s)

        elif self.settings.vision_encoder == "resnet50":
            self.vision_encoder = ResNet50MLM(
                num_channels=settings.vision_input_shape[3],
                num_classes=settings.emb_dim,
                settings=ResNet50MLMSettings(
                    num_tokens=settings.resnet_num_tokens,
                    pos_embedding=settings.resnet_pos_embedding,
                    mlp_output=settings.resnet_mlp_output,
                ),
            )
        elif self.settings.vision_encoder == "vit":
            self.vision_encoder = VitEncoder(
                in_channels=settings.vision_input_shape[-1],
                out_channels=settings.emb_dim,
                settings=ViTEncoderSettings(
                    emb_dim=settings.emb_dim,
                    transformer_dropout=settings.drop_rate,
                    emb_dropout=settings.drop_rate,
                    autopad_enabled=True,
                    patch_size=self.settings.patch_size,
                ),
                input_shape=settings.vision_input_shape[:2],
            )

        else:
            raise ValueError(
                f"Unknown vision encoder: {self.settings.vision_encoder}. Use 'linear', 'vit' or 'resnet50'."
            )

        self.norm_or_ident: nn.Identity | nn.LayerNorm
        if settings.layer_norm_vis:
            self.norm_or_ident = nn.LayerNorm(self.settings.emb_dim)
        else:
            self.norm_or_ident = nn.Identity()

    @property
    def context_length(self) -> int:
        return self.backend.context_length

    def forward(self, token_ids: Tensor, vision_input: NamedTensor) -> Tensor:
        vis_timesteps_embeds = []

        for timestep_nt in vision_input.iter_dim("timestep"):
            # batch, lat, lon, features
            # rearrange to batch, features, lat, lon
            timestep_nt.rearrange_("batch lat lon features -> batch features lat lon")

            vis_timesteps_embeds.append(self.vision_encoder(timestep_nt.tensor))
        vis_embeds = torch.cat(vis_timesteps_embeds, dim=1)

        # Normalize the output
        if self.settings.layer_norm_vis:
            vis_embeds = self.norm_or_ident(vis_embeds)

        return self.backend(token_ids, vis_embeds)
