"""
pytorch models wrapped
for DSM/LabIA projects.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union

import torch
from dataclasses_json import dataclass_json
from torch import nn

from .base import ModelABC
from mfai.torch.models.encoders import get_encoder


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, name: str):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        return self.double_conv(x)


@dataclass_json
@dataclass(slots=True)
class UnetSettings:
    init_features: int = 64


class UNet(ModelABC, nn.Module):
    """
    Returns a u_net architecture, with uninitialised weights, matching desired numbers of input and output channels.

    Implementation from the original paper: https://arxiv.org/pdf/1505.04597.pdf.
    """

    settings_kls = UnetSettings
    onnx_supported = True
    input_spatial_dims = (2,)

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        input_shape: Union[None, Tuple[int, int]] = None,
        settings: UnetSettings = UnetSettings(),
    ):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape

        features = settings.init_features

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        self.check_required_attributes()

    def forward(self, x):
        """
        Description of the architecture from the original paper (https://arxiv.org/pdf/1505.04597.pdf):
        The network architecture is illustrated in Figure 1. It consists of a contracting
        path (left side) and an expansive path (right side). The contracting path follows
        the typical architecture of a convolutional network. It consists of the repeated
        application of two 3x3 convolutions (unpadded convolutions), each followed by
        a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
        for downsampling. At each downsampling step we double the number of feature
        channels. Every step in the expansive path consists of an upsampling of the
        feature map followed by a 2x2 convolution (“up-convolution”) that halves the
        number of feature channels, a concatenation with the correspondingly cropped
        feature map from the contracting path, and two 3x3 convolutions, each fol-
        lowed by a ReLU. The cropping is necessary due to the loss of border pixels in
        every convolution. At the final layer a 1x1 convolution is used to map each 64-
        component feature vector to the desired number of classes. In total the network
        has 23 convolutional layers.
        To allow a seamless tiling of the output segmentation map (see Figure 2), it
        is important to select the input tile size such that all 2x2 max-pooling operations
        are applied to a layer with an even x- and y-size.
        """

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.max_pool(enc1))
        enc3 = self.encoder3(self.max_pool(enc2))
        enc4 = self.encoder4(self.max_pool(enc3))

        bottleneck = self.bottleneck(self.max_pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
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
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


@dataclass_json
@dataclass(slots=True)
class CustomUnetSettings:
    encoder_name: str = "resnet18"
    encoder_depth: int = 5
    encoder_weights: bool = True


class CustomUnet(ModelABC, nn.Module):
    settings_kls = CustomUnetSettings
    onnx_supported = True
    input_spatial_dims = (2,)

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        input_shape: Union[None, Tuple[int, int]] = None,
        settings: CustomUnetSettings = CustomUnetSettings(),
    ):
        super(CustomUnet, self).__init__()

        self.encoder = get_encoder(
            settings.encoder_name,
            in_channels=in_channels,
            depth=settings.encoder_depth,
            weights=settings.encoder_weights,
        )

        decoder_channels = self.encoder.out_channels[
            ::-1
        ]  # Reverse the order to be the same index of the decoder

        # Decoder layers
        self.upconvs, self.decoders = nn.ModuleList(), nn.ModuleList()
        for i, (decoder_in_channel, decoder_out_channel) in enumerate(
            zip(decoder_channels[:-1], decoder_channels[1:])
        ):
            self.upconvs.append(
                nn.ConvTranspose2d(
                    decoder_in_channel, decoder_out_channel, kernel_size=2, stride=2
                )
            )
            self.decoders.append(
                DoubleConv(decoder_out_channel * 2, decoder_out_channel, f"dec{i}")
            )

        # Final convolutional layer for segmentation map
        self.final_conv = nn.Conv2d(decoder_out_channel, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder part
        encoder_outputs = self.encoder(x)
        encoder_outputs = encoder_outputs[
            ::-1
        ]  # Reverse the order to be the same index of the decoder
        x = encoder_outputs[0]

        # Decoder part
        for skip, upconv, decoder in zip(
            encoder_outputs[1:], self.upconvs, self.decoders
        ):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final_conv(x)
