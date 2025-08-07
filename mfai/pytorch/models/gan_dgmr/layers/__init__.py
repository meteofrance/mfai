"""Submodules for the layers."""

from typing import Literal

import torch

from .attention import AttentionLayer  # noqa: F401
from .conv_gru import ConvGRU  # noqa: F401
from .coord_conv import CoordConv


def get_conv_layer(
    conv_type: Literal["standard", "coord", "3d"] = "standard",
) -> torch.nn.Module:
    """Return a conv layer based on the passed in string name."""
    if conv_type == "standard":
        return torch.nn.Conv2d
    elif conv_type == "coord":
        return CoordConv
    elif conv_type == "3d":
        return torch.nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
