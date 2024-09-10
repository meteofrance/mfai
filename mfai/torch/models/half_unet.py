from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from typing import Tuple, Union

import torch
from dataclasses_json import dataclass_json
from torch import nn

from mfai.torch.models.base import ModelABC
from mfai.torch.models.utils import AbsolutePosEmdebding


@dataclass_json
@dataclass(slots=True)
class HalfUNetSettings:
    num_filters: int = 64
    dilation: int = 1
    bias: bool = False
    use_ghost: bool = False
    last_activation: str = "Identity"
    absolute_pos_embed: bool = False


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        bias: bool = False,
        kernel_size=3,
        padding="same",
        dilation=1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.sepconv = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=out_channels // 2,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x2 = self.sepconv(x)
        x = torch.cat([x, x2], dim=1)
        x = self.bn(x)
        return self.relu(x)


class HalfUNet(ModelABC, nn.Module):
    settings_kls = HalfUNetSettings
    onnx_supported = True
    input_spatial_dims = (2,)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: Union[None, Tuple[int, int]] = None,
        settings: HalfUNetSettings = HalfUNetSettings(),
        *args,
        **kwargs,
    ):
        if settings.absolute_pos_embed and input_shape is None:
            raise ValueError(
                "You must provide a grid_shape to use absolute_pos_embed in HalfUnet"
            )

        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape

        if settings.absolute_pos_embed:
            if input_shape is None:
                raise ValueError(
                    "You must provide an input_shape to use absolute_pos_embed in HalfUnet"
                )

        self.encoder1 = self._block(
            in_channels,
            settings.num_filters,
            name="enc1",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
            absolute_pos_embed=settings.absolute_pos_embed,
            grid_shape=input_shape,
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc2",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
            absolute_pos_embed=settings.absolute_pos_embed,
            grid_shape=[x // 2 for x in input_shape] if input_shape else None,
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc3",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
            absolute_pos_embed=settings.absolute_pos_embed,
            grid_shape=[x // 4 for x in input_shape] if input_shape else None,
        )

        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc4",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
            absolute_pos_embed=settings.absolute_pos_embed,
            grid_shape=[x // 8 for x in input_shape] if input_shape else None,
        )
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = self._block(
            settings.num_filters,
            settings.num_filters,
            name="enc5",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
            absolute_pos_embed=settings.absolute_pos_embed,
            grid_shape=[x // 16 for x in input_shape] if input_shape else None,
        )
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=16)

        self.decoder = self._block(
            settings.num_filters,
            settings.num_filters,
            name="decoder",
            bias=settings.bias,
            use_ghost=settings.use_ghost,
            dilation=settings.dilation,
            absolute_pos_embed=settings.absolute_pos_embed,
            grid_shape=input_shape,
        )

        self.outconv = nn.Conv2d(
            in_channels=settings.num_filters,
            out_channels=out_channels,
            kernel_size=1,
            bias=settings.bias,
        )

        self.activation = getattr(nn, settings.last_activation)()

        self.check_required_attributes()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        summed = reduce(
            torch.Tensor.add_,
            [enc1, self.up2(enc2), self.up3(enc3), self.up4(enc4), self.up5(enc5)],
            torch.zeros_like(enc1),
        )
        dec = self.decoder(summed)
        return self.activation(self.outconv(dec))

    @staticmethod
    def _block(
        in_channels,
        features,
        name,
        bias=False,
        use_ghost: bool = False,
        dilation: int = 1,
        padding="same",
        absolute_pos_embed: bool = False,
        grid_shape: Tuple[int, int] = None,
    ):
        if use_ghost:
            layers = nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "ghost1",
                            GhostModule(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                        (
                            name + "ghost2",
                            GhostModule(
                                in_channels=features,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                    ]
                )
            )
        else:
            layers = nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "conv1",
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                        (name + "norm1", nn.BatchNorm2d(num_features=features)),
                        (name + "relu1", nn.ReLU(inplace=True)),
                        (
                            name + "conv2",
                            nn.Conv2d(
                                in_channels=features,
                                out_channels=features,
                                kernel_size=3,
                                padding=padding,
                                bias=bias,
                                dilation=dilation,
                            ),
                        ),
                        (name + "norm2", nn.BatchNorm2d(num_features=features)),
                        (name + "relu2", nn.ReLU(inplace=True)),
                    ]
                )
            )
        if absolute_pos_embed:
            layers = nn.Sequential(
                AbsolutePosEmdebding(
                    input_shape=(grid_shape[0], grid_shape[1]),
                    num_features=in_channels,
                ),
                *layers,
            )
        return layers
