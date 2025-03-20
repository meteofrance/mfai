from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from dataclasses_json import dataclass_json
from torchvision.models.resnet import ResNet

from mfai.torch.models import get_vision_encoder, utils


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self) -> int:
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels: int, pretrained: bool = True) -> None:
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(
            model=self, new_in_channels=in_channels, pretrained=pretrained
        )

    def get_stages(self):
        """Override it in your implementation"""
        raise NotImplementedError

    def make_dilated(self, output_stride: int) -> None:
        if output_stride == 16:
            stage_list = [5]
            dilation_list = [2]

        elif output_stride == 8:
            stage_list = [4, 5]
            dilation_list = [2, 4]

        else:
            raise ValueError(
                "Output stride should be 16 or 8, got {}.".format(output_stride)
            )

        self._output_stride = output_stride

        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )


class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels: int, depth: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self) -> List[nn.Module]:
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x) -> List[torch.Tensor]:
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs) -> None:
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


@dataclass_json
@dataclass(slots=True)
class ResNet50Settings:
    encoder_depth: int = 5
    encoder_weights: bool = False
    encoder_stride: int = 32


class ResNet50(torch.nn.Module):
    settings_kls = ResNet50Settings

    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 1000,
        input_shape: Union[None, Tuple[int, int]] = None,
        settings: ResNet50Settings = ResNet50Settings(),
    ):
        super().__init__()
        self.encoder = get_vision_encoder(
            name="resnet50",
            in_channels=num_channels,
            depth=settings.encoder_depth,
            weights=settings.encoder_weights,
            output_stride=settings.encoder_stride,
        )
        # For details, see:
        # https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * 4, num_classes)

    def forward(self, x: torch.Tensor):
        y_hat = self.encoder(x)[-1]
        y_hat = self.avgpool(y_hat)
        y_hat = y_hat.reshape(y_hat.shape[0], -1)
        y_hat = self.fc(y_hat)
        return y_hat
