from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from dataclasses_json import dataclass_json
from torch.nn import functional as F

from .base import ModelABC, ModelType
from .resnet import ResNetEncoder, get_resnet_encoder


class Activation(nn.Module):
    def __init__(
        self,
        name: Literal[
            "identity", "sigmoid", "softmax2d", "softmax", "logsoftmax", "tanh"
        ]
        | Callable = "identity",
        **params: dict[str, Any],
    ) -> None:
        super().__init__()

        self.activation: (
            nn.Identity | nn.Sigmoid | nn.Softmax | nn.LogSoftmax | nn.Tanh | Any
        )
        if name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1)
        elif name == "softmax":
            self.activation = nn.Softmax(dim=None)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(dim=None)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class DeepLabV3Decoder(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: tuple[int, int, int] = (12, 24, 36),
    ) -> None:
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features: tuple[torch.Tensor]) -> torch.Tensor:
        return super().forward(features[-1])


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        out_channels: int = 256,
        atrous_rates: tuple[int, int, int] = (12, 24, 36),
        output_stride: int = 16,
    ) -> None:
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError(
                "Output stride should be 8 or 16, got {}.".format(output_stride)
            )

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(
                out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(
                highres_in_channels, highres_out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features: tuple[torch.Tensor]) -> torch.Tensor:
        aspp_features: torch.Tensor = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features: torch.Tensor = self.block1(features[-4])
        concat_features: torch.Tensor = torch.cat(
            [aspp_features, high_res_features], dim=1
        )
        fused_features: torch.Tensor = self.block2(concat_features)
        return fused_features


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> None:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: tuple[int, int, int],
        separable: bool = False,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: list[torch.Tensor] = []
        for conv in self.convs:
            res.append(conv(x))
        concat: torch.Tensor = torch.cat(res, dim=1)
        return self.project(concat)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


@dataclass_json
@dataclass(slots=True)
class DeepLabV3Settings:
    """
    encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
        to extract features of different spatial resolution
    encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
        two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
        with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
        Default is 5
    encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
        other pretrained weights (see table with available weights for each encoder_name)
    decoder_channels: A number of convolution filters in ASPP module. Default is 256
    activation: An activation function to apply after the final convolution layer.
        Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**.
        Default is **None**
    upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
    aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
        on top of encoder if **aux_params** is not **None** (default). Supported params:
            - classes (int): A number of classes
            - pooling (str): One of "max", "avg". Default is "avg"
            - dropout (float): Dropout factor in [0, 1)
            - activation (str): An activation function to apply "sigmoid"/"softmax"
                (could be **None** to return logits)
    """

    encoder_name: Literal["resnet18", "resnet34", "resnet50"] = "resnet18"
    encoder_depth: int = 5
    encoder_weights: bool = True
    decoder_channels: int = 256
    activation: (
        Literal["identity", "sigmoid", "softmax2d", "softmax", "logsoftmax", "tanh"]
        | Callable
    ) = "identity"
    upsampling: int = 8
    aux_params: Optional[dict] = None


class DeepLabV3(ModelABC, torch.nn.Module):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        out_channels: A number of channels for output mask (or you can think as a number of classes of output mask)
        settings: DeepLabV3Settings

    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587
    """

    onnx_supported: bool = True
    settings_kls = DeepLabV3Settings
    supported_num_spatial_dims = (2,)
    features_last: bool = False
    model_type: ModelType = ModelType.CONVOLUTIONAL
    num_spatial_dims: int = 2
    register: bool = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        input_shape: tuple[int, ...] = (-1,),
        settings: DeepLabV3Settings = DeepLabV3Settings(),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        self.encoder: ResNetEncoder = get_resnet_encoder(
            settings.encoder_name,
            in_channels=in_channels,
            depth=settings.encoder_depth,
            weights=settings.encoder_weights,
            output_stride=8,
        )

        self.decoder: DeepLabV3Decoder = self.get_decoder(
            in_channels=self.encoder.out_channels[0],
            out_channels=settings.decoder_channels,
        )

        self.segmentation_head: nn.Sequential = self.get_segmentation_head(
            in_channels=self.decoder.out_channels,
            out_channels=out_channels,
            activation=settings.activation,
            kernel_size=1,
            upsampling=settings.upsampling,
        )

        self.classification_head: nn.Sequential | None
        if settings.aux_params is not None:
            self.classification_head = self.get_classification_head(
                in_channels=self.encoder.out_channels[-1], **settings.aux_params
            )
        else:
            self.classification_head = None

        self.check_required_attributes()

    @property
    def settings(self) -> DeepLabV3Settings:
        """
        Returns the settings instance used to configure the model.
        """
        return self._settings

    def initialize(self) -> None:
        self.initialize_decoder(self.decoder)
        self.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            self.initialize_head(self.classification_head)

    def initialize_decoder(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize_head(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def check_input_shape(self, x: torch.Tensor) -> None:
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, in_channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, out_channels, height, width)

        """
        if self.training:
            self.eval()

        prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = self.forward(x)

        return prediction

    def get_segmentation_head(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Literal[
            "identity", "sigmoid", "softmax2d", "softmax", "logsoftmax", "tanh"
        ]
        | Callable = "identity",
        upsampling: int = 1,
    ) -> nn.Sequential:
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling_module: nn.Module = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        return nn.Sequential(conv2d, upsampling_module, activation)

    def get_classification_head(
        self,
        in_channels: int,
        out_channels: int,
        pooling: str = "avg",
        dropout: float = 0.2,
        activation: Literal[
            "identity", "sigmoid", "softmax2d", "softmax", "logsoftmax", "tanh"
        ]
        | Callable = "identity",
    ) -> nn.Sequential:
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout_module: nn.Module = (
            nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        )
        linear = nn.Linear(in_channels, out_channels, bias=True)
        activation_module: Activation = Activation(activation)
        return nn.Sequential(pool, flatten, dropout_module, linear, activation_module)

    def get_decoder(self, in_channels: int, out_channels: int) -> DeepLabV3Decoder:
        return DeepLabV3Decoder(in_channels=in_channels, out_channels=out_channels)


@dataclass_json
@dataclass(slots=True)
class DeepLabV3PlusSettings(DeepLabV3Settings):
    """
    encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
    decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
    upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
    """

    encoder_output_stride: int = 16
    decoder_atrous_rates: tuple = (12, 24, 36)
    upsampling: int = 4


class DeepLabV3Plus(DeepLabV3):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        settings: DeepLabV3Settings

    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    settings_kls = DeepLabV3PlusSettings

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        input_shape: Tuple[int, int] = (-1, -1),
        settings: DeepLabV3PlusSettings = DeepLabV3PlusSettings(),
    ):
        self._settings: DeepLabV3PlusSettings
        super().__init__(in_channels, out_channels, input_shape, settings)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape

        self.encoder: ResNetEncoder = get_resnet_encoder(
            settings.encoder_name,
            in_channels=in_channels,
            depth=settings.encoder_depth,
            weights=settings.encoder_weights,
            output_stride=settings.encoder_output_stride,
        )

        # Violates liskov substitution principle
        self.decoder = DeepLabV3PlusDecoder(  # type: ignore[assignment]
            encoder_channels=self.encoder.out_channels,
            out_channels=settings.decoder_channels,
            atrous_rates=settings.decoder_atrous_rates,
            output_stride=settings.encoder_output_stride,
        )

        self.check_required_attributes()

    @property
    def settings(self) -> DeepLabV3PlusSettings:
        """
        Returns the settings instance used to configure the model.
        """
        return self._settings
