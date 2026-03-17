from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json
from torch import Tensor

from .base import BaseModel, ModelType


@dataclass_json
@dataclass(slots=True)
class IdentityModelSettings:
    """Empty dataclass because IdentityModel has no parameters."""

    pass


class IdentityModel(BaseModel):
    """Implementation of an identity model. Its forward method returns
    the input tensor.
    """

    settings = None
    settings_kls = IdentityModelSettings
    onnx_supported = False
    supported_num_spatial_dims = (2, 3)
    features_last = False
    model_type = ModelType.IDENTITY
    num_spatial_dims: int = 2
    register: bool = True

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: unused arguments.
            **kwargs: unused keywords arguments.

        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Return the input torch.Tensor without any changes."""
        return x
