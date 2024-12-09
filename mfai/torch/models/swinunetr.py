from dataclasses import dataclass
from typing import Tuple, Union

import torch
from dataclasses_json import dataclass_json
from monai.networks.blocks.dynunet_block import UnetResBlock
from monai.networks.nets.swin_unetr import SwinUNETR as MonaiSwinUNETR
from torch import nn

from .base import ModelABC, ModelType


@dataclass_json
@dataclass(slots=True)
class SwinUNETRSettings:
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    feature_size: int = 24
    norm_name: tuple | str = "instance"
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    normalize: bool = True
    use_checkpoint: bool = False
    downsample = "merging"
    use_v2 = False


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_name: tuple | str,
    ):
        super().__init__()
        self.upsampler = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_block = UnetResBlock(
            2,
            out_channels * 2,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.upsampler(inp)
        # concat along the channels/features dimension
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class SwinUNETR(ModelABC, MonaiSwinUNETR):
    """
    Wrapper around the SwinUNETR from MONAI.
    Instanciated in 2D for now, with a custom decoder.
    """

    onnx_supported = False
    settings_kls = SwinUNETRSettings
    supported_num_spatial_dims = (2,)
    features_last = False
    model_type = ModelType.VISION_TRANSFORMER
    num_spatial_dims: int = 2
    register: bool = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: Union[None, Tuple[int, int]] = None,
        settings: SwinUNETRSettings = SwinUNETRSettings(),
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=2,
            img_size=(128, 128),  # TODO: fix this and pass the grid shape
            **settings.to_dict(),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        # We replace the decoders by UpsamplingBilinear2d + Conv2d
        # because ConvTranspose2d introduced checkerboard artifacts

        feature_size = settings.feature_size
        self.decoder5 = UpsampleBlock(
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            norm_name=settings.norm_name,
        )

        self.decoder4 = UpsampleBlock(
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            norm_name=settings.norm_name,
        )

        self.decoder3 = UpsampleBlock(
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            norm_name=settings.norm_name,
        )
        self.decoder2 = UpsampleBlock(
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            norm_name=settings.norm_name,
        )

        self.decoder1 = UpsampleBlock(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            norm_name=settings.norm_name,
        )

        self.check_required_attributes()

    @property
    def settings(self) -> SwinUNETRSettings:
        return self._settings
