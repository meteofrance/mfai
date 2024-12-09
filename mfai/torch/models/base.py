"""
Interface contract for our models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Tuple


class ModelType(Enum):
    """
    Enum to classify the models depending on their architecture family.
    Having the model expose this as an attributee facilitates top level code:
    reshaping input tensors, iterating only on a subset of models, etc.
    """

    GRAPH = 1
    CONVOLUTIONAL = 2
    VISION_TRANSFORMER = 3
    LLM = 4
    MULTIMODAL_LLM = 5


class ModelABC(ABC):
    # concrete subclasses shoudl set register to True
    # to be included in the registry of available models.
    register: bool = False

    @property
    @abstractmethod
    def onnx_supported(self) -> bool:
        """
        Indicates if our model supports onnx export.
        """

    @property
    @abstractmethod
    def settings_kls(self):
        """
        Returns the settings class for this model.
        """

    @property
    @abstractmethod
    def supported_num_spatial_dims(self) -> Tuple[int, ...]:
        """
        Returns the number of input spatial dimensions supported by the model.
        A 2d vision model supporting (H, W) should return (2,).
        A model supporting both 2d and 3d inputs (by settings) should return (2, 3).
        Once instanciated the model will be in 2d OR 3d mode.
        """

    @property
    @abstractmethod
    def settings(self) -> Any:
        """
        Returns the settings instance used to configure for this model.
        """

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """
        Returns the model type.
        """

    @property
    @abstractmethod
    def num_spatial_dims(self) -> int:
        """
        Returns the number of spatial dimensions of the instanciated model.
        """

    @property
    @abstractmethod
    def features_last(self) -> bool:
        """
        Indicates if the features are the last dimension in the input/output tensors.
        Conv and ViT typically have features as the second dimension (Batch, Features, ...)
        versus GNNs for which features are the last dimension (Batch, ..., Features)
        """

    @property
    def features_second(self) -> bool:
        return not self.features_last

    def check_required_attributes(self):
        # we check that the model has defined the following attributes.
        # this must be called at the end of the __init__ of each subclass.
        required_attrs = ["in_channels", "out_channels", "input_shape"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute : {attr}")
