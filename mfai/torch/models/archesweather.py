# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as gradient_checkpoint
from axial_attention import AxialAttention, AxialPositionalEmbedding
from timm.layers import DropPath
from torch import Tensor
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint


from .pangu import (
    MLP,
    CustomPad3d,
    DownSample,
    EarthAttention3D,
    PatchEmbedding,
    PatchRecovery,
    UpSample,
    generate_3d_attention_mask,
)

from .base import BaseModel, ModelType


class EarthSpecificBlock(nn.Module):
    """3D transformer block with Earth-Specific bias and window attention,
    see https://github.com/microsoft/Swin-Transformer for the official implementation of
    2D window attention. The major difference is that we expand the dimensions to 3 and
    replace the relative position bias with Earth-Specific bias.

    Args:
        data_size (torch.Size): data size in terms of plevel, latitude, longitude
        dim (int): token size
        drop_path_ratio (float): ratio to apply to drop path
        heads (int): number of attention heads
        window_size (tuple[int], optional): window size for the sliding window attention.
        Defaults to (2, 6, 12).
        dropout_rate (float, optional): dropout rate in the MLP. Defaults to 0..
        axial_attn (bool, optional): whether to use axial attention. Defaults to False.
        axial_attn_heads (int, optional): number of heads for axial attention. Defaults to 8.
        checkpoint_activation (bool, optional): whether to use checkpoint for activation.
        Defaults to False.
        lam (bool, optional): whether to use limited area setting for shifted-window attention.
        Defaults to False.
    """

    def __init__(
        self,
        data_size: torch.Size,
        dim: int,
        drop_path_ratio: float,
        heads: int,
        window_size: tuple[int, int, int] = (2, 6, 12),
        dropout_rate: float = 0.0,
        axial_attn: bool = False,
        axial_attn_heads: int = 8,
        checkpoint_activation: bool = False,
        lam: bool = False,
    ) -> None:
        super().__init__()

        self.checkpoint_activation = checkpoint_activation
        self.lam = lam
        # Define the window size of the neural network
        self.window_size = window_size
        assert all(
            [w_s == 1 or w_s % 2 == 0 for w_s in window_size]
        ), "Window size must be divisible by 2"
        self.shift_size = tuple(w_size // 2 + w_size % 2 for w_size in window_size)

        # Initialize serveral operations
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = LayerNorm(dim)
        self.pad3D = CustomPad3d(data_size[-3:], self.window_size)
        self.attention = EarthAttention3D(
            self.pad3D.padded_size, dim, heads, dropout_rate, self.window_size
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, dropout_rate=dropout_rate)

        if axial_attn:
            self.axis_pos_embed = AxialPositionalEmbedding(
                dim=dim, shape=(data_size[-3],), emb_dim_index=-1
            )
            self.axial_attn = AxialAttention(
                dim=dim,
                dim_index=-1,
                heads=axial_attn_heads,
                num_dimensions=1,
                sum_axial_out=True,
            )

    def forward(
        self,
        x: Tensor,
        embedding_shape: torch.Size,
        cond_embed: Optional[Tensor] = None,
        roll: bool = False,
    ) -> Tensor:
        # Save the shortcut for skip-connection
        shortcut = x
        x = self.norm1(x)

        # ArchesWeather code
        if cond_embed is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                cond_embed.chunk(6, dim=1)
            )
            x = x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        # End of ArchesWeather code

        # Reshape input to three dimensions to calculate window attention
        x = x.view(embedding_shape)

        # Zero-pad input if needed
        # reshape data for padding, from B, Z, H, W, C to B, C, Z, H, W
        x = x.permute((0, 4, 1, 2, 3))
        x = self.pad3D(x)

        # back to previous shape
        x = x.permute((0, 2, 3, 4, 1))

        batch_size, padded_z, padded_h, padded_w, channels = x.shape

        if roll:
            # Roll x for half of the window for 3 dimensions
            x = x.roll(
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
            # Generate mask of attention masks
            # If two pixels are not adjacent, then mask the attention between them
            # Your can set the matrix element to -1000 when it is not adjacent,
            # then add it to the attention
            assert len(self.shift_size) == 3, "Shift size must be 3D"
            mask = generate_3d_attention_mask(
                x, self.window_size, self.shift_size, self.lam
            )
        else:
            # e.g., zero matrix when you add mask to attention
            mask = None

        # Reorganize data to calculate window attention
        x_window = x.reshape(
            shape=(
                x.shape[0],
                padded_z // self.window_size[0],
                self.window_size[0],
                padded_h // self.window_size[1],
                self.window_size[1],
                padded_w // self.window_size[2],
                self.window_size[2],
                -1,
            )
        )
        x_window = x_window.permute((0, 1, 3, 5, 2, 4, 6, 7))

        # Get data stacked in 3D cubes, which will further be used
        # to calculate attention among each cube
        x_window = x_window.reshape(
            shape=(
                -1,
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                channels,
            )
        )

        # Apply 3D window attention with Earth-Specific bias
        if self.checkpoint_activation:
            x_window = checkpoint(
                self.attention,
                x_window,
                mask,
                batch_size,
                padded_z,
                padded_h,
                use_reentrant=False,
            )
        else:
            x_window = self.attention(x_window, mask, batch_size, padded_z, padded_h)

        # Reorganize data to original shapes
        x = x_window.reshape(
            shape=(
                batch_size,
                padded_z // self.window_size[0],
                padded_h // self.window_size[1],
                padded_w // self.window_size[2],
                self.window_size[0],
                self.window_size[1],
                self.window_size[2],
                -1,
            )
        )
        x = x.permute((0, 1, 4, 2, 5, 3, 6, 7))

        # Reshape the tensor back to its original shape
        x = x.reshape(shape=(batch_size, padded_z, padded_h, padded_w, -1))

        if roll:
            # Roll x back for half of the window
            x = x.roll(
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )

        # Crop the zero-padding
        (
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            padding_front,
            padding_back,
        ) = self.pad3D.padding
        x = x[
            :,
            padding_front : padded_z - padding_back,
            padding_top : padded_h - padding_bottom,
            padding_left : padded_w - padding_right,
            :,
        ]

        # Reshape the tensor back to the input shape
        batch_size, pl, lat, lon, channels = x.shape
        x = x.reshape(shape=(batch_size, -1, channels))

        # ArchesWeather code
        if hasattr(self, "axial_attn"):
            x2 = (
                x.reshape(batch_size, pl, lat * lon, channels)
                .movedim(2, 1)
                .flatten(0, 1)
            )  # B*Lat*Lon, Pl, C
            x2 = self.axis_pos_embed(x2)
            x2 = self.axial_attn(x2)
            x2 = (
                x2.reshape(batch_size, lat * lon, pl, channels)
                .movedim(1, 2)
                .flatten(1, 2)
            )  # B, Pl*Lat*Lon, C

        # Main calculation stages
        if cond_embed is None:
            x = shortcut + self.drop_path(x)
            if hasattr(self, "axial_attn"):
                x = x + self.drop_path(x2)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if hasattr(self, "axial_attn"):
                x = x + self.drop_path(x2)
            x = shortcut + gate_msa[:, None, :] * self.drop_path(x)
            mlp_input = (
                self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            )
            x = x + self.drop_path(gate_mlp[:, None, :] * self.mlp(mlp_input))
        return x  # B, Pl*Lat*Lon, C
        # End of ArchesWeather code


class EarthSpecificLayer(nn.Module):
    """Basic layer of our network, contains 2 or 6 blocks

    Args:
        depth (int): number of blocks
        data_size (torch.Size): see EarthSpecificBlock
        dim (int): see EarthSpecificBlock
        drop_path_ratio_list (list[float]): see EarthSpecificBlock
        num_heads (int): see EarthSpecificBlock
    """

    def __init__(
        self,
        depth: int,
        data_size: torch.Size,
        dim: int,
        drop_path_ratio_list: list[float],
        num_heads: int,
        window_size: tuple[int, int, int],
        dropout_rate: float,
        axial_attn: bool,
        axial_attn_heads: int,
        checkpoint_activation: bool,
        lam: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()

        # Construct basic blocks
        for i in range(depth):
            self.blocks.append(
                EarthSpecificBlock(
                    data_size=data_size,
                    dim=dim,
                    drop_path_ratio=drop_path_ratio_list[i],
                    heads=num_heads,
                    window_size=window_size,
                    dropout_rate=dropout_rate,
                    axial_attn=axial_attn,
                    axial_attn_heads=axial_attn_heads,
                    checkpoint_activation=checkpoint_activation,
                    lam=lam,
                )
            )

    def forward(
        self,
        x: Tensor,
        embedding_shape: torch.Size,
        cond_embed: Optional[Tensor] = None,
    ) -> Tensor:
        for i, block in enumerate(self.blocks):
            # Roll the input every two blocks
            if i % 2 == 0:
                x = block(x, embedding_shape, roll=False, cond_embed=cond_embed)
            else:
                x = block(x, embedding_shape, roll=True, cond_embed=cond_embed)
        return x


class Interpolate(nn.Module):
    """Interpolation module.
    Args:
        scale_factor (float): scaling
        mode (str): interpolation mode
        align_corners (bool): align corners
    """

    def __init__(
        self, scale_factor: float, mode: str, align_corners: bool = False
    ) -> None:
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class PatchRecoveryConv(nn.Module):
    """Upsampling with interpolation + conv to avoid chessboard effect
    Args:
        input_dim (int): input feature size
        downfactor (int): downsampling factor (patch size in latitude and longitude)
        hidden_dim (int): hidden feature size
        plevel_variables (int): number of level variables
        surface_variables (int): number of surface variables
        plevels (int): number of levels
    """

    def __init__(
        self,
        input_dim: int,
        downfactor: int = 4,
        hidden_dim: int = 96,
        plevel_variables: int = 5,
        surface_variables: int = 4,
        plevels: int = 13,
    ) -> None:
        super().__init__()
        assert np.log2(downfactor).is_integer(), "downfactor should be a power of 2"
        self.total_levels = plevels + 1
        self.input_conv = nn.Conv2d(
            input_dim,
            self.total_levels * hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.interp = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

        self.head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.upsampling_heads = nn.ModuleList()
        if downfactor > 2:
            for _ in range(1, int(np.log2(downfactor))):
                self.upsampling_heads.append(
                    nn.Sequential(
                        nn.GroupNorm(
                            num_groups=32,
                            num_channels=hidden_dim,
                            eps=1e-6,
                            affine=True,
                        ),
                        nn.Conv3d(
                            hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1
                        ),
                        nn.GELU(),
                        nn.Conv3d(
                            hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1
                        ),
                        nn.GELU(),
                    )
                )

        self.proj_surface = nn.Conv2d(
            hidden_dim, surface_variables, kernel_size=1, stride=1, padding=0
        )
        self.proj_level = nn.Conv3d(
            hidden_dim, plevel_variables, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        x = x.flatten(1, 2)
        x = self.input_conv(x)
        x = x.reshape(
            (batch_size, self.total_levels, -1, x.shape[-2], x.shape[-1])
        ).flatten(0, 1)
        x = self.interp(x)
        x = x.reshape(
            batch_size, self.total_levels, -1, x.shape[-2], x.shape[-1]
        ).movedim(1, 2)
        x = self.head(x)
        for head in self.upsampling_heads:
            x = x.reshape(
                (batch_size, self.total_levels, -1, x.shape[-2], x.shape[-1])
            ).flatten(0, 1)
            x = self.interp(x)
            x = x.reshape(
                batch_size, self.total_levels, -1, x.shape[-2], x.shape[-1]
            ).movedim(1, 2)
            x = head(x)

        output_surface = self.proj_surface(x[:, :, -1])
        output_level = self.proj_level(x[:, :, :-1])

        return output_level, output_surface.unsqueeze(-3)


class LinVert(nn.Module):
    """Linear layer for the vertical dimension
    Args:
        in_features (int): input feature size
        embedding_size (tuple[int]): embedding size
    """

    def __init__(self, in_features: int, embedding_size: Tuple[int, ...]) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(
            embedding_size[-3] * in_features, embedding_size[-3] * in_features
        )

    def forward(self, x: Tensor) -> Tensor:
        x2 = (
            x.reshape((x.shape[0], self.embedding_size[-3], -1, x.shape[-1]))
            .movedim(1, -2)
            .flatten(-2, -1)
        )  # B, lat*lon, Pl*C
        x2 = self.fc1(x2)
        x2 = (
            x2.reshape((x2.shape[0], -1, self.embedding_size[-3], x.shape[-1]))
            .movedim(-2, 1)
            .flatten(1, 2)
        )  # B, Pl*lat*lon, C

        return x + x2


class CondBasicLayer(EarthSpecificLayer):
    """Wrapper for EarthSpecificLayer with conditional embeddings
    Args:
        dim (int): token size.
        cond_dim (int): size of the conditional embedding.
    """

    def __init__(
        self,
        depth: int,
        data_size: torch.Size,
        dim: int,
        cond_dim: int,
        drop_path_ratio_list: list[float],
        num_heads: int,
        window_size: tuple[int, int, int],
        dropout_rate: float,
        lam: bool,
        axial_attn: bool,
        axial_attn_heads: int,
        checkpoint_activation: bool,
    ):
        super().__init__(
            depth=depth,
            data_size=data_size,
            dim=dim,
            drop_path_ratio_list=drop_path_ratio_list,
            num_heads=num_heads,
            window_size=window_size,
            dropout_rate=dropout_rate,
            axial_attn=axial_attn,
            axial_attn_heads=axial_attn_heads,
            checkpoint_activation=checkpoint_activation,
            lam=lam,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x: Tensor,
        embedding_shape: torch.Size,
        cond_emb: Optional[Tensor] = None,
    ) -> Tensor:
        c = self.adaLN_modulation(cond_emb)
        return super().forward(x, embedding_shape, c)

@dataclass_json
@dataclass
class ArchesWeatherSettings:
    """ArchesWeather configuration class"""

    emb_dim: int = 192
    cond_dim: int = 32
    num_heads: tuple = (6, 12, 12, 6)
    droppath_coeff: float = 0.2
    patch_size: tuple = (2, 2, 2)
    window_size: tuple = (1, 6, 10)
    depth_multiplier: int = 1
    position_embs_dim: int = 0
    use_prev: bool = False
    use_skip: bool = False
    conv_head: bool = False
    dropout: float = 0.0
    first_interaction_layer: bool = False
    gradient_checkpointing: bool = False
    axial_attn: bool = False
    axial_attn_heads: int = 8
    lam: bool = False
    lon_resolution: int = 240
    lat_resolution: int = 120
    surface_variables: int = 4
    static_length: int = 3
    plevel_variables: int = 5
    plevels: int = 13
    spatial_dims: int = 2


class ArchesWeather(BaseModel):
    """ArchesWeather model as described in http://arxiv.org/abs/2405.14527"""

    onnx_supported: bool = False
    supported_num_spatial_dims: Tuple = (2,)
    settings_kls = ArchesWeatherSettings
    model_type = ModelType.PANGU
    features_last: bool = False
    register: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: Tuple[int, ...],
        archesweather_config: ArchesWeatherSettings = ArchesWeatherSettings(),
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels, including constant mask if any.
            out_channels: dimension of output channels.
            input_shape: dimension of input image.
            emb_dim: embedding size
            cond_dim: conditioning embedding size
            num_heads: number of heads per EarthSpecificLayer
            droppath_coeff: drop path coefficient
            patch_size: patch size for input data embedding
            window_size: window size for shifted-window attention of EarthSpecificBlock
            depth_multiplier: depth multiplier for the number of blocks in EarthSpecificLayer
            position_embs_dim: dimension of positional embeddings
            use_prev: whether to use previous state
            use_skip: whether to use skip connections
            conv_head: whether to use a convolutional head for patch recovery
            dropout: dropout rate
            first_interaction_layer: whether to use a linear interaction layer before the first EarthSpecificLayer
            gradient_checkpointing: whether to use gradient checkpointing
            axial_attn: whether to use axial attention
            axial_attn_head: number of heads for axial attention
            lam: whether to use limited area setting in the attention mask
            lon_resolution: longitude resolution
            lat_resolution: latitude resolution
            surface_variables: number of variables in the surface data
            static_length: number of variables in the mask data
            plevel_variables: number of variables in the level data
            plevels: number of atmospheric levels in the level data
            spatial_dims: number of spatial dimensions (2).
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self.archesweather_config = archesweather_config

        if archesweather_config.spatial_dims == 2:
            lat_resolution, lon_resolution = input_shape
        else:
            raise ValueError(f"Unsupported spatial dimension: {archesweather_config.spatial_dims}")

        surface_variables = archesweather_config.surface_variables,
        static_length = archesweather_config.static_length,
        plevel_variables = archesweather_config.plevel_variables,
        plevels = archesweather_config.plevels,

        drop_path = np.linspace(
            0,
            archesweather_config.droppath_coeff / archesweather_config.depth_multiplier,
            8 * archesweather_config.depth_multiplier,
        ).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.layer1_shape = (
            lat_resolution // archesweather_config.patch_size[1],
            lon_resolution // archesweather_config.patch_size[2],
        )

        self.positional_embeddings = nn.Parameter(
            torch.zeros(
                (archesweather_config.position_embs_dim, lat_resolution, lon_resolution)
            )
        )
        torch.nn.init.trunc_normal_(self.positional_embeddings, 0.02)

        # Pangu code
        self.patchembed = PatchEmbedding(
            c_dim=archesweather_config.emb_dim,
            patch_size=archesweather_config.patch_size,
            plevel_size=torch.Size(
                (plevel_variables, plevels, lat_resolution, lon_resolution)
            ),
            surface_size=torch.Size(
                (
                    surface_variables + static_length + archesweather_config.position_embs_dim,
                    lat_resolution,
                    lon_resolution,
                )
            ),
        )
        embedding_size = self.patchembed.embedding_size

        if archesweather_config.first_interaction_layer:
            self.interaction_layer = LinVert(
                in_features=archesweather_config.emb_dim, embedding_size=embedding_size
            )

        self.layer1 = CondBasicLayer(
            depth=2 * archesweather_config.depth_multiplier,
            data_size=embedding_size,
            dim=archesweather_config.emb_dim,
            cond_dim=archesweather_config.cond_dim,
            drop_path_ratio_list=drop_path[: 2 * archesweather_config.depth_multiplier],
            num_heads=archesweather_config.num_heads[0],
            window_size=archesweather_config.window_size,
            dropout_rate=archesweather_config.dropout,
            lam=archesweather_config.lam,
            axial_attn=archesweather_config.axial_attn,
            axial_attn_heads=archesweather_config.axial_attn_heads,
            checkpoint_activation=archesweather_config.gradient_checkpointing,
        )
        # Pangu code
        self.downsample = DownSample(embedding_size, archesweather_config.emb_dim)
        downsampled_size = self.downsample.downsampled_size
        self.layer2 = CondBasicLayer(
            depth=6 * archesweather_config.depth_multiplier,
            data_size=downsampled_size,
            dim=archesweather_config.emb_dim * 2,
            cond_dim=archesweather_config.cond_dim,
            drop_path_ratio_list=drop_path[2 * archesweather_config.depth_multiplier :],
            num_heads=archesweather_config.num_heads[1],
            window_size=archesweather_config.window_size,
            dropout_rate=archesweather_config.dropout,
            lam=archesweather_config.lam,
            axial_attn=archesweather_config.axial_attn,
            axial_attn_heads=archesweather_config.axial_attn_heads,
            checkpoint_activation=archesweather_config.gradient_checkpointing,
        )
        self.layer3 = CondBasicLayer(
            depth=6 * archesweather_config.depth_multiplier,
            data_size=downsampled_size,
            dim=archesweather_config.emb_dim * 2,
            cond_dim=archesweather_config.cond_dim,
            drop_path_ratio_list=drop_path[2 * archesweather_config.depth_multiplier :],
            num_heads=archesweather_config.num_heads[2],
            window_size=archesweather_config.window_size,
            dropout_rate=archesweather_config.dropout,
            lam=archesweather_config.lam,
            axial_attn=archesweather_config.axial_attn,
            axial_attn_heads=archesweather_config.axial_attn_heads,
            checkpoint_activation=archesweather_config.gradient_checkpointing,
        )
        # Pangu code
        self.upsample = UpSample(
            archesweather_config.emb_dim * 2, archesweather_config.emb_dim
        )
        out_dim = (
            archesweather_config.emb_dim
            if not archesweather_config.use_skip
            else 2 * archesweather_config.emb_dim
        )
        self.layer4 = CondBasicLayer(
            depth=2 * archesweather_config.depth_multiplier,
            data_size=embedding_size,
            dim=out_dim,
            cond_dim=archesweather_config.cond_dim,
            drop_path_ratio_list=drop_path[: 2 * archesweather_config.depth_multiplier],
            num_heads=archesweather_config.num_heads[3],
            window_size=archesweather_config.window_size,
            dropout_rate=archesweather_config.dropout,
            lam=archesweather_config.lam,
            axial_attn=archesweather_config.axial_attn,
            axial_attn_heads=archesweather_config.axial_attn_heads,
            checkpoint_activation=archesweather_config.gradient_checkpointing,
        )

        self.patchrecovery: PatchRecovery | PatchRecoveryConv
        if not archesweather_config.conv_head:
            # Pangu code
            self.patchrecovery = PatchRecovery(
                out_dim, archesweather_config.patch_size, plevel_variables, surface_variables
            )
        else:
            self.patchrecovery = PatchRecoveryConv(
                input_dim=embedding_size[-3] * out_dim,
                downfactor=archesweather_config.patch_size[-1],
                plevel_variables=plevel_variables,
                surface_variables=surface_variables,
                plevels=plevels,
            )
    @property
    def settings(self) -> ArchesWeatherSettings:
        return self.archesweather_config

    @property
    def num_spatial_dims(self) -> int:
        return self.archesweather_config.spatial_dims

    def forward(
        self,
        input_surface: Tensor,
        input_level: Tensor,
        cond_emb: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        pos_embs = self.positional_embeddings[None].expand(
            (input_surface.shape[0], *self.positional_embeddings.shape)
        )

        input_surface = torch.concat([input_surface, pos_embs], dim=1)

        x, embedding_shape = self.patchembed(input_level, input_surface)

        if self.archesweather_config.first_interaction_layer:
            x = self.interaction_layer(x)

        x = self.layer1(x, embedding_shape, cond_emb)

        skip = x
        x, downsampled_shape = self.downsample(x, embedding_shape)

        x = self.layer2(x, downsampled_shape, cond_emb)

        if self.archesweather_config.gradient_checkpointing:
            x = gradient_checkpoint.checkpoint(
                self.layer3, x, downsampled_shape, cond_emb, use_reentrant=False
            )
        else:
            x = self.layer3(x, downsampled_shape, cond_emb)

        x = self.upsample(x, embedding_shape)
        if self.archesweather_config.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
            embedding_shape = list(embedding_shape)
            embedding_shape[-1] = 2 * embedding_shape[-1]
        x = self.layer4(x, embedding_shape, cond_emb)  # B, Pl*Lat*Lon, C

        output = x
        output = output.transpose(1, 2).reshape(
            output.shape[0], -1, *self.patchembed.embedding_size
        )

        if not self.archesweather_config.conv_head:
            output_level, output_surface = self.patchrecovery(output, embedding_shape)
            output_surface = output_surface.unsqueeze(-3)
            # Crop the output to remove zero-paddings
            padded_z, padded_h, padded_w = output_level.shape[2:5]
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            ) = self.patchembed.pad_plevel_data.padding
            output_level = output_level[
                :,
                :,
                padding_front : padded_z - padding_back,
                padding_top : padded_h - padding_bottom,
                padding_left : padded_w - padding_right,
            ]
            output_surface = output_surface[
                :,
                :,
                :,
                padding_top : padded_h - padding_bottom,
                padding_left : padded_w - padding_right,
            ]
        else:
            output_level, output_surface = self.patchrecovery(output)
            # Crop the output to remove zero-paddings
            _, padded_h, padded_w = output_level.shape[2:5]
            padding_left, padding_right, padding_top, padding_bottom, _, _ = (
                self.patchembed.pad_plevel_data.padding
            )
            output_level = output_level[
                :,
                :,
                :,
                padding_top : padded_h - padding_bottom,
                padding_left : padded_w - padding_right,
            ]
            output_surface = output_surface[
                :,
                :,
                :,
                padding_top : padded_h - padding_bottom,
                padding_left : padded_w - padding_right,
            ]

        out = dict(next_state_level=output_level, next_state_surface=output_surface)
        return out
