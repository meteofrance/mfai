import math
from typing import Tuple

import einops
import torch
import torch.nn as nn


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size,
            )
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(
            module.out_channels, new_in_channels // module.groups, *module.kernel_size
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()


class AbsolutePosEmdebding(nn.Module):
    """
    Absolute pos embedding.
    Learns a position dependent bias for each pixel/node of each feature map.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_features: int,
        feature_last: bool = False,
    ):
        super().__init__()
        if feature_last:
            self.pos_embedding = nn.Parameter(
                init_(torch.zeros(1, *input_shape, num_features)), requires_grad=True
            )
        else:
            self.pos_embedding = nn.Parameter(
                init_(torch.zeros(1, num_features, *input_shape), dim_idx=1),
                requires_grad=True,
            )

    def forward(self, x):
        return x + self.pos_embedding


def init_(tensor: torch.Tensor, dim_idx=-1):
    dim = tensor.shape[dim_idx]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def features_last_to_second(x: torch.Tensor) -> torch.Tensor:
    """
    Moves features from the last dimension to the second dimension.
    """
    return einops.rearrange(x, "b x y n -> b n x y").contiguous()


def features_second_to_last(y: torch.Tensor) -> torch.Tensor:
    """
    Moves features from the second dimension to the last dimension.
    """
    return einops.rearrange(y, "b n x y -> b x y n").contiguous()


def expand_to_batch(x: torch.Tensor, batch_size: int):
    """
    Expand tensor with initial batch dimension
    """
    # In order to be generic (for 1D or 2D grid)
    sizes = [batch_size] + [-1 for _ in x.shape]
    return x.unsqueeze(0).expand(*sizes)
