from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json
from torch import Tensor
from typing_extensions import override

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

    @property
    @override
    def settings(self) -> Any:
        return None

    settings_kls = IdentityModelSettings
    onnx_supported = False
    supported_num_spatial_dims = (2, 3)
    features_last = False
    model_type = ModelType.IDENTITY

    @property
    @override
    def num_spatial_dims(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: unused arguments.
            **kwargs: unused keywords arguments.

        """
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Return the input torch.Tensor without any changes."""
        return x
