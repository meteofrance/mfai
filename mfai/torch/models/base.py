"""
Interface contract for our models.
"""

from abc import ABC, abstractproperty
from typing import Tuple


class ModelABC(ABC):
    @abstractproperty
    def onnx_supported(self) -> bool:
        """
        Indicates if our model supports onnx export.
        """

    @abstractproperty
    def settings_kls(self):
        """
        Returns the settings class for this model.
        """

    @abstractproperty
    def input_spatial_dims(self) -> Tuple[int, ...]:
        """
        Returns the input spatial dimensions supported by the model.
        A model supporting 2d tensors should return (2,).
        A model supporting 2d and 3d tensors should return (2, 3).
        """

    def check_required_attributes(self):
        required_attrs = ["in_channels", "out_channels", "input_shape"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute : {attr}")
