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
from typing import Tuple, Optional

import torch
from torch.nn import (
    Linear,
    Conv3d,
    Conv2d,
    ConvTranspose3d,
    ConvTranspose2d,
    GELU,
    Dropout,
    LayerNorm,
    Softmax,
    ConstantPad3d,
    ConstantPad2d,
)
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from timm.layers import DropPath


from .base import BaseModel, ModelType


def define_3d_earth_position_index(window_size: Tuple[int, int, int]) -> Tensor:
    """Build the index for the Earth specific positional bias of sliding
    attention windows from PanguWeather.
    See http://arxiv.org/abs/2211.02556

    Args:
        window_size (Tuple[int, int, int]): size of the sliding window

    Returns:
        Tensor: index
    """
    assert len(window_size) == 3, (
        "Data must be 3D, but window has {}" "dimension(s)".format(len(window_size))
    )

    # Index in the pressure level of query matrix
    coords_zi = torch.arange(window_size[0])
    # Index in the pressure level of key matrix
    coords_zj = -torch.arange(window_size[0]) * window_size[0]

    # Index in the latitude of query matrix
    coords_hi = torch.arange(window_size[1])
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(window_size[1]) * window_size[1]

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(window_size[2])

    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
    coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += window_size[2] - 1
    coords[:, :, 1] *= 2 * window_size[2] - 1
    coords[:, :, 0] *= (2 * window_size[2] - 1) * window_size[1] * window_size[1]

    # Sum up the indexes in three dimensions
    position_index = coords.sum(dim=-1)

    # Flatten the position index to facilitate further indexing
    position_index = torch.flatten(position_index)

    return position_index


def generate_3d_attention_mask(
    x: Tensor,
    window_size: Tuple[int, int, int],
    shift_size: Tuple[int, ...],
    lam: bool = False,
) -> Tensor:
    """Method to generate attention mask for sliding window attention in the context of 3D data.
    Based on https://pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html#swin_s

    Args:
        x (Tensor): input data, used to generate the mask on the same device
        window_size (Tuple[int, int, int]): size of the sliding window
        shift_size (Tuple[int, ...]): size of the shift for the sliding window

    Returns:
        Tensor: attention mask
    """
    assert x.dim() == 5, "Data must be 3D, but has {} dimension(s)".format(x.dim())
    _, pad_z, pad_h, pad_w, _ = x.shape
    assert (
        pad_z % window_size[0] == 0
        and pad_h % window_size[1] == 0
        and pad_w % window_size[2] == 0
    ), "the data size must divisible by window size"
    assert (
        window_size[0] % shift_size[0] == 0
        and window_size[1] % shift_size[1] == 0
        and window_size[2] % shift_size[2] == 0
    ), "the window size must divisible by shift size"

    # Create the attention mask from the data to have same type and device
    attention_mask = x.new_zeros((pad_z, pad_h, pad_w))
    z_slices = ((0, -shift_size[0]), (-shift_size[0], None))
    h_slices = ((0, -shift_size[1]), (-shift_size[1], None))
    w_slices: Tuple[Tuple[int, Optional[int]], ...]
    if lam:
        w_slices = ((0, -shift_size[2]), (-shift_size[2], None))
    else:
        w_slices = ((0, None),)

    count = 0
    for z in z_slices:
        for h in h_slices:
            for w in w_slices:
                attention_mask[z[0] : z[1], h[0] : h[1], w[0] : w[1]] = count
                count += 1

    attention_mask = attention_mask.reshape(
        pad_z // window_size[0],
        window_size[0],
        pad_h // window_size[1],
        window_size[1],
        pad_w // window_size[2],
        window_size[2],
    )
    num_windows = (
        (pad_z // window_size[0])
        * (pad_h // window_size[1])
        * (pad_w // window_size[2])
    )
    attention_mask = torch.permute(attention_mask, (0, 2, 4, 1, 3, 5)).reshape(
        num_windows, -1
    )
    attention_mask = attention_mask.unsqueeze(1) - attention_mask.unsqueeze(2)
    attention_mask.masked_fill_(attention_mask != 0, -100.0)
    return attention_mask


@dataclass_json
@dataclass
class PanguWeatherSettings:
    plevel_patch_size: Tuple[int, int, int] = (2, 4, 4)
    token_size: int = 192
    layer_depth: Tuple[int, int] = (2, 6)
    num_heads: Tuple[int, int] = (6, 12)
    spatial_dims: int = 2
    surface_variables: int = 4
    plevel_variables: int = 5
    plevels: int = 13
    static_length: int = 3
    window_size: Tuple[int, int, int] = (2, 6, 12)
    dropout_rate: float = 0.0
    checkpoint_activation: bool = False
    lam: bool = False


class PanguWeather(BaseModel):
    """
    PanguWeather network as described in http://arxiv.org/abs/2211.02556 and https://www.nature.com/articles/s41586-023-06185-3.
    This implementation follows the official pseudo code here: https://github.com/198808xc/Pangu-Weather
    """

    onnx_supported: bool = False
    supported_num_spatial_dims: Tuple = (2,)
    settings_kls = PanguWeatherSettings
    model_type = ModelType.PANGU
    features_last: bool = False
    register: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: Tuple[int, ...],
        settings: PanguWeatherSettings = PanguWeatherSettings(),
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels, including constant mask if any.
            out_channels: dimension of output channels.
            input_shape: dimension of input image.
            plevel_patch_size : Patch size for the pressure level data. Default is (2, 4, 4). Setting (2, 8, 8) leads to Pangu Lite.
            token_size : Size of the tokens (equivalent to channel size) of the first layer. Default is 192.
            layer_depth : Number of blocks in layers. Default is (2, 6), meaning that the first and fourth layers contain 2 blocks, and the second and third contain 6.
            num_heads : Number of heads in attention layers. Default is (6, 12), corresponding to respectively first and fourth layers, and second and third.
            spatial_dims: number of spatial dimensions (2 or 3).
            surface_variables: number of surface variables.
            plevel_variables: number of pressure level variables.
            plevels: number of pressure levels.
            static_length: number of static variables (e.g., land sea mask).
            window_size: size of the sliding window.
            dropout_rate: faction of the input units to drop.
            checkpoint_activation: whether to use checkpoint activation.
            lam: whether to use the limited area attention mask.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        if settings.spatial_dims == 2:
            latitude, longitude = input_shape
        else:
            raise ValueError(f"Unsupported spatial dimension: {settings.spatial_dims}")

        self.plevel_patch_size = settings.plevel_patch_size

        # Compute surface size and plevel size. Needed to compute the size of earth specific bias
        self.surface_channels = settings.surface_variables + settings.static_length
        surface_size = torch.Size((self.surface_channels, latitude, longitude))
        plevel_size = torch.Size(
            (settings.plevel_variables, settings.plevels, latitude, longitude)
        )

        # Drop path rate is linearly increased as the depth increases
        drop_path_list = torch.linspace(0, 0.2, 8)

        # Patch embedding
        token_size = settings.token_size
        self.embedding_layer = PatchEmbedding(
            token_size,
            self.plevel_patch_size,
            plevel_size=plevel_size,
            surface_size=surface_size,
        )
        embedding_size = self.embedding_layer.embedding_size

        # Upsample and downsample
        self.downsample = DownSample(embedding_size, token_size)
        downsampled_size = self.downsample.downsampled_size
        self.upsample = UpSample(token_size * 2, token_size)

        # Four basic layers
        layer_depth = settings.layer_depth
        num_heads = settings.num_heads
        self.layer1 = EarthSpecificLayer(
            depth=layer_depth[0],
            data_size=embedding_size,
            dim=token_size,
            drop_path_ratio_list=drop_path_list[:2],
            num_heads=num_heads[0],
            window_size=settings.window_size,
            dropout_rate=settings.dropout_rate,
            checkpoint_activation=settings.checkpoint_activation,
            lam=settings.lam,
        )
        self.layer2 = EarthSpecificLayer(
            depth=layer_depth[1],
            data_size=downsampled_size,
            dim=token_size * 2,
            drop_path_ratio_list=drop_path_list[2:],
            num_heads=num_heads[1],
            window_size=settings.window_size,
            dropout_rate=settings.dropout_rate,
            checkpoint_activation=settings.checkpoint_activation,
            lam=settings.lam,
        )
        self.layer3 = EarthSpecificLayer(
            depth=layer_depth[1],
            data_size=downsampled_size,
            dim=token_size * 2,
            drop_path_ratio_list=drop_path_list[2:],
            num_heads=num_heads[1],
            window_size=settings.window_size,
            dropout_rate=settings.dropout_rate,
            checkpoint_activation=settings.checkpoint_activation,
            lam=settings.lam,
        )
        self.layer4 = EarthSpecificLayer(
            depth=layer_depth[0],
            data_size=embedding_size,
            dim=token_size,
            drop_path_ratio_list=drop_path_list[:2],
            num_heads=num_heads[0],
            window_size=settings.window_size,
            dropout_rate=settings.dropout_rate,
            checkpoint_activation=settings.checkpoint_activation,
            lam=settings.lam,
        )

        # Patch Recovery
        self._output_layer = PatchRecovery(
            dim=token_size * 2,
            patch_size=self.plevel_patch_size,
            plevel_channels=settings.plevel_variables,
            surface_channels=settings.surface_variables,
        )

        self.check_required_attributes()

    @property
    def settings(self) -> PanguWeatherSettings:
        return self._settings

    @property
    def num_spatial_dims(self) -> int:
        return self.settings.spatial_dims

    def forward(
        self, input_plevel: Tensor, input_surface: Tensor, static_data: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the PanguWeather model.
        Args:
            input_plevel (Tensor): Input tensor of shape (N, C, Z, H, W) for pressure level data.
            input_surface (Tensor): Input tensor of shape (N, C, H, W) for surface data.
            static_data (Tensor, optional): Static data tensor, e.g., land sea mask, of shape (N, C, H, W). Defaults to None.
        """
        if static_data is not None:
            surface_data = torch.cat([input_surface, static_data], dim=1)
        else:
            surface_data = input_surface

        # Embed the input fields into patches
        x, embedding_shape = self.embedding_layer(input_plevel, surface_data)

        # Encoder, composed of two layers
        x = self.layer1(x, embedding_shape)

        # Store the tensor for skip-connection
        skip = x

        # Downsample from (8, 181, 360) to (8, 91, 180) in the case of vanilla Pangu
        x, downsampled_shape = self.downsample(x, embedding_shape)

        # Layer 2, shape (8, 91, 180, 2C), C = 192 as in the original paper in the case of vanilla Pangu
        x = self.layer2(x, downsampled_shape)

        # Decoder, composed of two layers
        # Layer 3, shape (8, 91, 180, 2C), C = 192 as in the original paper in the case of vanilla Pangu
        x = self.layer3(x, downsampled_shape)

        # Upsample from (8, 91, 180) to (8, 181, 360)
        x = self.upsample(x, embedding_shape)

        # Layer 4, shape (8, 181, 360, 2C), C = 192 as in the original paper in the case of vanilla Pangu
        x = self.layer4(x, embedding_shape)

        # Skip connect, in last dimension(C from 192 to 384) in the case of vanilla Pangu
        x = torch.cat([x, skip], dim=-1)

        # Recover the output fields from patches
        output_plevel, output_surface = self._output_layer(x, embedding_shape)

        # Crop the output to remove zero-paddings
        padded_z, padded_h, padded_w = output_plevel.shape[2:5]
        (
            padding_left,
            padding_right,
            padding_top,
            padding_bottom,
            padding_front,
            padding_back,
        ) = self.embedding_layer.pad_plevel_data.padding
        output_plevel = output_plevel[
            :,
            :,
            padding_front : padded_z - padding_back,
            padding_top : padded_h - padding_bottom,
            padding_left : padded_w - padding_right,
        ]
        output_surface = output_surface[
            :,
            :,
            padding_top : padded_h - padding_bottom,
            padding_left : padded_w - padding_right,
        ]

        return output_plevel, output_surface


class CustomPad3d(ConstantPad3d):
    """Custom 3d padding based on token embedding patch size. Padding direction is center.

    Args:
        data_size (torch.Size): data size
        patch_size (Tuple[int, int, int]): patch size for the token embedding operation
        value (float, optional): padding value. Defaults to 0.
    """

    def __init__(
        self,
        data_size: torch.Size,
        patch_size: Tuple[int, int, int],
        value: float = 0.0,
    ) -> None:
        # Compute paddings, starts from the last dim and goes backward
        assert (
            len(data_size) == 3
        ), "This padding class is for 3d data, but data has {} dimension(s)".format(
            len(data_size)
        )
        assert (
            len(patch_size) == 3
        ), "Patch should be 3d, but has {} dimension(s)".format(len(patch_size))
        padding_lon = (
            patch_size[-1] - (data_size[-1] % patch_size[-1])
            if (data_size[-1] % patch_size[-1]) > 0
            else 0
        )
        padding_left = padding_lon // 2
        padding_right = padding_lon - padding_left
        padding_lat = (
            patch_size[-2] - (data_size[-2] % patch_size[-2])
            if (data_size[-2] % patch_size[-2]) > 0
            else 0
        )
        padding_top = padding_lat // 2
        padding_bottom = padding_lat - padding_top
        padding_level = (
            patch_size[-3] - (data_size[-3] % patch_size[-3])
            if (data_size[-3] % patch_size[-3]) > 0
            else 0
        )
        padding_front = padding_level // 2
        padding_back = padding_level - padding_front
        super().__init__(
            padding=(
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            ),
            value=value,
        )
        self.padded_size = torch.Size(
            [
                data_size[0] + padding_level,
                data_size[1] + padding_lat,
                data_size[2] + padding_lon,
            ]
        )


class CustomPad2d(ConstantPad2d):
    """Custom 2d padding based on token embedding patch size. Padding direction is center.

    Args:
        data_size (torch.Size): data size
        patch_size (Tuple[int, int]): patch size for the token embedding operation
        value (float, optional): padding value. Defaults to 0.
    """

    def __init__(
        self, data_size: torch.Size, patch_size: Tuple[int, int], value: float = 0.0
    ) -> None:
        # Compute paddings, starts from the last dim and goes backward
        assert (
            len(data_size) == 2
        ), "This padding class is for 2d data, but data has {} dimension(s)".format(
            len(data_size)
        )
        assert (
            len(patch_size) == 2
        ), "Patch should be 2d, but has {} dimension(s)".format(len(patch_size))
        padding_lon = (
            patch_size[-1] - (data_size[-1] % patch_size[-1])
            if (data_size[-1] % patch_size[-1]) > 0
            else 0
        )
        padding_left = padding_lon // 2
        padding_right = padding_lon - padding_left
        padding_lat = (
            patch_size[-2] - (data_size[-2] % patch_size[-2])
            if (data_size[-2] % patch_size[-2]) > 0
            else 0
        )
        padding_top = padding_lat // 2
        padding_bottom = padding_lat - padding_top
        super().__init__(
            padding=(padding_left, padding_right, padding_top, padding_bottom),
            value=value,
        )
        self.padded_size = torch.Size(
            [data_size[0] + padding_lat, data_size[1] + padding_lon]
        )


class PatchEmbedding(nn.Module):
    """Patch embedding operation. Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches,
        patch_size = (2, 4, 4) in the original paper

    Args:
        c_dim (_type_): embeeding channel size
        patch_size (Tuple[int, int, int]): patch size for pressure level data
        plevel_size (torch.Size): pressure level data size
        surface_size (torch.Size): surface data size
    """

    def __init__(
        self,
        c_dim: int,
        patch_size: Tuple[int, int, int],
        plevel_size: torch.Size,
        surface_size: torch.Size,
    ) -> None:
        super().__init__()

        # Here we use convolution to partition data into cubes
        self.conv_surface = Conv2d(
            in_channels=surface_size[0],
            out_channels=c_dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )
        self.conv_plevel = Conv3d(
            in_channels=plevel_size[0],
            out_channels=c_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # init padding
        self.pad_plevel_data = CustomPad3d(plevel_size[-3:], patch_size)
        self.pad_surface_data = CustomPad2d(surface_size[-2:], patch_size[1:])

        # Compute output size
        plevel_padded_size = self.pad_plevel_data.padded_size
        embedding_size = [
            plevel_dim // patch_dim
            for plevel_dim, patch_dim in zip(plevel_padded_size, patch_size)
        ]
        embedding_size[0] += 1
        self.embedding_size = torch.Size(embedding_size)

    def forward(
        self, input_plevel: Tensor, input_surface: Tensor
    ) -> Tuple[Tensor, torch.Size]:
        # Zero-pad the input
        plevel_data = self.pad_plevel_data(input_plevel)
        surface_data = self.pad_surface_data(input_surface)

        # Project to embedding space
        plevel_tokens = self.conv_plevel(plevel_data)
        surface_tokens = self.conv_surface(surface_data)

        # Concatenate the input in the pressure level, i.e., in Z dimension
        x = torch.cat([plevel_tokens, surface_tokens.unsqueeze(2)], dim=2)

        # Reshape x for calculation of linear projections
        x = x.permute((0, 2, 3, 4, 1))
        embedding_shape = x.shape
        x = x.reshape(shape=(x.shape[0], -1, x.shape[-1]))

        return x, embedding_shape


class PatchRecovery(nn.Module):
    """Patch recovery operation. The inverse operation of the patch embedding operation.

    Args:
        dim (int): number of channels
        patch_size (Tuple[int, int, int]): pressure level patch size, e. g., (2, 4, 4) as in the original paper
        plevel_channels (int, optional): pressure level data channel size
        surface_channels (int, optional): surface data channel size
    """

    def __init__(
        self,
        dim: int,
        patch_size: Tuple[int, int, int],
        plevel_channels: int = 5,
        surface_channels: int = 4,
    ) -> None:
        super().__init__()
        # Hear we use two transposed convolutions to recover data
        self.conv_surface = ConvTranspose2d(
            in_channels=dim,
            out_channels=surface_channels,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )
        self.conv = ConvTranspose3d(
            in_channels=dim,
            out_channels=plevel_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor, embedding_shape: torch.Size) -> Tuple[Tensor, Tensor]:
        # Reshape x back to three dimensions
        x = x.reshape(
            x.shape[0], embedding_shape[1], embedding_shape[2], embedding_shape[3], -1
        )
        x = x.permute(0, 4, 1, 2, 3)

        # Call the transposed convolution
        output_plevel = self.conv(x[:, :, :-1, :, :].contiguous())
        output_surface = self.conv_surface(x[:, :, -1, :, :].contiguous())

        return output_plevel, output_surface


class DownSample(nn.Module):
    """Down-sampling operation. The number of tokens is divided by 4 while their size in multiplied by 2.
    E. g., from (8x360x181) tokens of size 192 to (8x180x91) tokens of size 384.

    Args:
        data_size (torch.Size): data size in terms of embeded plevel, latitude, longitude
        dim (int): initial size of the tokens
    """

    def __init__(self, data_size: torch.Size, dim: int) -> None:
        super().__init__()
        # A linear function and a layer normalization
        self.linear = Linear(in_features=4 * dim, out_features=2 * dim, bias=False)
        self.norm = LayerNorm(normalized_shape=4 * dim)
        self.pad = CustomPad3d(data_size[-3:], (1, 2, 2))
        padded_size = self.pad.padded_size
        self.downsampled_size = torch.Size(
            [padded_size[0], padded_size[1] // 2, padded_size[2] // 2]
        )

    def forward(
        self, x: Tensor, embedding_shape: torch.Size
    ) -> Tuple[Tensor, torch.Size]:
        # Reshape x to three dimensions for downsampling
        x = x.reshape(shape=embedding_shape)

        # Padding the input to facilitate downsampling
        x = x.permute((0, 4, 1, 2, 3))
        x = self.pad(x)
        x = x.permute((0, 2, 3, 4, 1))

        # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 182, 360) to (8, 91, 180)
        Z, H, W = x.shape[1:4]
        # Reshape x to facilitate downsampling
        x = x.reshape(shape=(x.shape[0], Z, H // 2, 2, W // 2, 2, x.shape[-1]))
        # Change the order of x
        x = x.permute((0, 1, 2, 4, 3, 5, 6))
        # Reshape to get a tensor of resolution (8, 91, 180) -> 4 times less tokens of 4 times bigger size
        x = x.reshape(shape=(x.shape[0], Z * (H // 2) * (W // 2), -1))

        # Call the layer normalization
        x = self.norm(x)

        # Decrease the size of the tokens to reduce computation cost
        x = self.linear(x)
        return x, torch.Size([x.shape[0], Z, (H // 2), (W // 2), x.shape[-1]])


class UpSample(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Up-sampling operation. The number of tokens is mutiplied by 4.

        Args:
            input_dim (int): input token size
            output_dim (int): output token size
        """
        super().__init__()
        # Linear layers without bias to increase channels of the data
        self.linear1 = Linear(input_dim, output_dim * 4, bias=False)

        # Linear layers without bias to mix the data up
        self.linear2 = Linear(output_dim, output_dim, bias=False)

        # Normalization
        self.norm = LayerNorm(output_dim)

    def forward(self, x: Tensor, embedding_shape: torch.Size) -> Tensor:
        assert (
            x.shape[-1] % 4 == 0
        ), "The token size must be divisible by 4, but is {}".format(x.shape[-1])
        # Z, H, W represent the desired output shape
        h_d = embedding_shape[2] // 2 + embedding_shape[2] % 2
        w_d = embedding_shape[3] // 2 + embedding_shape[3] % 2

        # Call the linear functions to increase channels of the data
        x = self.linear1(x)

        # Reorganize x to increase the resolution: simply change the order and upsample from (8, 91, 180) to (8, 182, 360)
        # Reshape x to facilitate upsampling.
        x = x.reshape(shape=(x.shape[0], embedding_shape[1], h_d, w_d, 2, 2, -1))
        # Change the order of x
        x = x.permute((0, 1, 2, 4, 3, 5, 6))
        # Reshape to get Tensor with a resolution of (8, 182, 360)
        x = x.reshape(shape=(x.shape[0], embedding_shape[1], h_d * 2, w_d * 2, -1))

        # Crop the output to the input shape of the network
        x = x[:, :, : embedding_shape[2], : embedding_shape[3], :]

        # Reshape x back
        x = x.reshape(shape=(x.shape[0], -1, x.shape[-1]))

        # Call the layer normalization
        x = self.norm(x)

        # Mixup normalized tensors
        x = self.linear2(x)
        return x


class EarthSpecificLayer(nn.Module):
    """Basic layer of our network, contains 2 or 6 blocks

    Args:
        depth (int): number of blocks
        data_size (torch.Size): see EarthSpecificBlock
        dim (int): see EarthSpecificBlock
        drop_path_ratio_list (Tensor]): see EarthSpecificBlock
        num_heads (int): see EarthSpecificBlock
        window_size (Tuple[int, int, int], optional): see EarthSpecificBlock
        dropout_rate (float, optional): see EarthSpecificBlock
        checkpoint_activation (bool, optional): see EarthSpecificBlock
        lam (bool, optional): see EarthSpecificBlock
    """

    def __init__(
        self,
        depth: int,
        data_size: torch.Size,
        dim: int,
        drop_path_ratio_list: Tensor,
        num_heads: int,
        window_size: Tuple[int, int, int],
        dropout_rate: float,
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
                    drop_path_ratio=drop_path_ratio_list[i].item(),
                    num_heads=num_heads,
                    window_size=window_size,
                    dropout_rate=dropout_rate,
                    checkpoint_activation=checkpoint_activation,
                    lam=lam,
                )
            )

    def forward(self, x: Tensor, embedding_shape: torch.Size) -> Tensor:
        for i, block in enumerate(self.blocks):
            # Roll the input every two blocks
            if i % 2 == 0:
                x = block(x, embedding_shape, roll=False)
            else:
                x = block(x, embedding_shape, roll=True)
        return x


class EarthSpecificBlock(nn.Module):
    """3D transformer block with Earth-Specific bias and window attention,
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
        The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.

    Args:
        data_size (torch.Size): data size in terms of plevel, latitude, longitude
        dim (int): token size
        drop_path_ratio (float): ratio to apply to drop path
        num_heads (int): number of attention heads
        window_size (Tuple[int, int, int], optional): window size for the sliding window attention. Defaults to (2, 6, 12).
        dropout_rate (float, optional): dropout rate in the MLP. Defaults to 0..
        checkpoint_activation (bool, optional): whether to use checkpoint activation. Defaults to False.
        lam (bool, optional): whether to use the limited area attention mask. Defaults to False.
    """

    def __init__(
        self,
        data_size: torch.Size,
        dim: int,
        drop_path_ratio: float,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 6, 12),
        dropout_rate: float = 0.0,
        checkpoint_activation: bool = False,
        lam: bool = False,
    ) -> None:
        super().__init__()

        self.checkpoint_activation = checkpoint_activation
        self.lam = lam
        # Define the window size of the neural network
        self.window_size = window_size
        assert all(
            [w_s % 2 == 0 for w_s in window_size]
        ), "Window size must be divisible by 2"
        self.shift_size = tuple(w_size // 2 for w_size in window_size)

        # Initialize serveral operations
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.norm1 = LayerNorm(dim)
        self.pad3D = CustomPad3d(data_size[-3:], self.window_size)
        self.attention = EarthAttention3D(
            self.pad3D.padded_size, dim, num_heads, dropout_rate, self.window_size
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, dropout_rate=dropout_rate)

    def forward(self, x: Tensor, embedding_shape: torch.Size, roll: bool) -> Tensor:
        # Save the shortcut for skip-connection
        shortcut = x

        # Reshape input to three dimensions to calculate window attention
        # x = x.view((x.shape[0], Z, H, W, -1))
        x = x.view(embedding_shape)

        # Zero-pad input if needed
        # reshape data for padding, from B, Z, H, W, C to B, C, Z, H, W
        x = x.permute((0, 4, 1, 2, 3))
        x = self.pad3D(x)

        # back to previous shape
        x = x.permute((0, 2, 3, 4, 1))

        batch_size, padded_z, padded_h, padded_w, C = x.shape

        if roll:
            # Roll x for half of the window for 3 dimensions
            x = x.roll(
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
            # Generate mask of attention masks
            # If two pixels are not adjacent, then mask the attention between them
            # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
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

        # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube
        x_window = x_window.reshape(
            shape=(
                -1,
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                C,
            )
        )

        # Apply 3D window attention with Earth-Specific bias
        if self.checkpoint_activation:
            x_window = checkpoint(
                self.attention, x_window, mask, batch_size, use_reentrant=False
            )
        else:
            x_window = self.attention(x_window, mask, batch_size)

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
        x = x.reshape(shape=(batch_size, -1, C))

        # Main calculation stages
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class EarthAttention3D(nn.Module):
    """3D sliding window attention with the Earth-Specific bias,
        see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D sliding window attention.

    Args:
        data_size (torch.Size): data size in terms of plevel, latitude, longitude
        dim (int): token size
        num_heads (int): number of heads
        dropout_rate (float): dropout rate
        window_size (Tuple[int, int, int]): window size (z, h ,w)
    """

    def __init__(
        self,
        data_size: torch.Size,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        window_size: Tuple[int, int, int],
    ) -> None:
        super().__init__()

        # Store several attributes
        self.head_number = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        self.window_size = window_size

        # Construct position index to reuse self.earth_specific_bias
        self.register_buffer(
            "position_index", define_3d_earth_position_index(window_size)
        )

        # Init earth specific bias
        # only pressure level and latitude have absolute bias, longitude is cyclic
        # data_size is plevel, latitude, longitude
        self.num_windows = (data_size[0] // self.window_size[0]) * (
            data_size[1] // self.window_size[1]
        )

        # For each window, we will construct a set of parameters according to the paper
        # Inside a window, plevel and latitude positions are absolute while longitude are relative
        self.earth_specific_bias = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[2] - 1)
                * self.window_size[1] ** 2
                * self.window_size[0] ** 2,
                self.num_windows,
                self.head_number,
            )
        )

        # Initialize several operations
        self.linear1 = Linear(dim, dim * 3, bias=True)
        self.linear2 = Linear(dim, dim)
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(p=dropout_rate)

        # Initialize the tensors using Truncated normal distribution
        torch.nn.init.trunc_normal_(self.earth_specific_bias, mean=0.0, std=0.02)

    def forward(self, x: Tensor, mask: Tensor, batch_size: int) -> Tensor:
        # Record the original shape of the input (B*num_windows, window_size, dim)
        original_shape = x.shape

        # Linear layer to create query, key and value
        x = self.linear1(x)

        # reshape the data to calculate multi-head attention
        qkv = x.reshape(
            shape=(
                x.shape[0],
                x.shape[1],
                3,
                self.head_number,
                self.dim // self.head_number,
            )
        )
        query, key, value = qkv.permute(
            (2, 0, 3, 1, 4)
        )  # 3, B*num_windows, head_number, window_size, dim_head

        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        self.attention = (
            query @ key.mT
        )  # @ denotes matrix multiplication ; B*num_windows_lon*num_windows, head_number, window_size, window_size

        # self.earth_specific_bias is a set of neural network parameters to optimize.
        assert isinstance(self.position_index, Tensor)
        earth_specific_bias = self.earth_specific_bias[self.position_index]

        # Reshape the learnable bias to the same shape as the attention matrix
        earth_specific_bias = earth_specific_bias.reshape(
            shape=(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.num_windows,
                self.head_number,
            )
        )
        earth_specific_bias = earth_specific_bias.permute((2, 3, 0, 1))
        earth_specific_bias = earth_specific_bias.unsqueeze(
            0
        )  # 1, num_windows, head_number, window_size, window_size

        # Add the Earth-Specific bias to the attention matrix
        attention_shape = self.attention.shape
        # Reshape and permute the lon dim to match the shape of earth_specific_bias
        attention = self.attention.reshape(
            batch_size,
            self.num_windows,
            -1,
            self.head_number,
            attention_shape[-2],
            attention_shape[-1],
        )
        attention = attention.permute(0, 2, 1, 3, 4, 5)
        attention = attention.reshape(
            -1,
            self.num_windows,
            self.head_number,
            attention_shape[-2],
            attention_shape[-1],
        )
        # add bias
        attention = attention + earth_specific_bias
        # Reshape the attention matrix back
        attention = attention.reshape(
            batch_size,
            -1,
            self.num_windows,
            self.head_number,
            attention_shape[-2],
            attention_shape[-1],
        )
        attention = attention.permute(0, 2, 1, 3, 4, 5)
        self.attention2 = attention.reshape(attention_shape)

        # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
        if mask is not None:
            attention = self.attention2.view(
                batch_size, -1, self.head_number, original_shape[1], original_shape[1]
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            self.attention2 = attention.view(
                -1, self.head_number, original_shape[1], original_shape[1]
            )
        attention = self.softmax(self.attention2)
        attention = self.dropout(attention)

        # Calculated the tensor after spatial mixing.
        x = (
            attention @ value
        )  # @ denote matrix multiplication ; B*num_windows, head_number, window_size, dim_head

        # Reshape tensor to the original shape
        x = x.permute((0, 2, 1, 3))  # B*num_windows, window_size, head_number, dim_head
        x = x.reshape(shape=original_shape)  # B*num_windows, window_size, dim

        # Linear layer to post-process operated tensor
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """MLP layers, same as most vision transformer architectures.

    Args:
        dim (int): input and output token size
        dropout_rate (float): dropout rate applied after each linear layer
    """

    def __init__(self, dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear1 = Linear(dim, dim * 4)
        self.linear2 = Linear(dim * 4, dim)
        self.activation = GELU()
        self.drop = Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x
