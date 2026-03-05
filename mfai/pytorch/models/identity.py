from torch import Tensor

from .base import BaseModel, ModelType


class IdentityModel(BaseModel):
    """Implementation of an identity model. It means that it return
    the input tensor.
    """

    settings = None
    settings_kls = None
    onnx_supported = False
    supported_num_spatial_dims = (2, 3)
    features_last = False
    model_type = ModelType.IDENTITY
    num_spatial_dims: int = 2
    register: bool = True

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: unused arguments.
            *kwargs: unused keywords arguments.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Return the input torch.Tensor without any changes."""
        return x
