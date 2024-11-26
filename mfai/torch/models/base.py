"""
Interface contract for our models.
"""

from abc import ABC, abstractproperty, abstractmethod
from typing import Tuple
from torch import Size


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
        
    @property
    def auto_padding_supported(self) -> bool:
        """
        Indicates whether the model supports automatic padding.
        """
        return isinstance(self, AutoPaddingModel)
    
    def check_required_attributes(self):
        # we check that the model has defined the following attributes.
        # this must be called at the end of the __init__ of each subclass.
        required_attrs = ["in_channels", "out_channels", "input_shape"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute : {attr}")


class AutoPaddingModel(ABC):
    @abstractmethod
    def validate_input_shape(self, input_shape: Size) -> Tuple[bool | Size]:
        """ Given an input shape, verifies whether the inputs fit with the 
            calling model's specifications. 

        Args:
            input_shape (Size): The shape of the input data, excluding any batch dimension and channel dimension.  
                                For example, for a batch of 2D tensors of shape [B,C,W,H], [W,H] should be passed.
                                For 3D data instead of shape [B,C,W,H,D], instead, [W,H,D] should be passed. 

        Returns:
            Tuple[bool, Size]: Returns a tuple where the first element is a boolean signaling whether the given input shape 
                                already fits the model's requirements. If that value is False, the second element contains the closest 
                                shape that fits the model, otherwise it will be None.
        """