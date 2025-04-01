"""
Interface contract for our models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
from torch import Size
import torch

from mfai.torch.padding import pad_batch, undo_padding


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


class ModelABC(ABC, torch.nn.Module):
    # concrete subclasses should set register to True
    # to be included in the registry of available models.
    register: bool = False

    in_channels: int
    out_channels: int
    input_shape: tuple[int, int]

    @property
    @abstractmethod
    def onnx_supported(self) -> bool:
        """
        Indicates if our model supports onnx export.
        """

    @property
    @abstractmethod
    def settings_kls(self) -> Any:
        """
        Returns the settings class for this model.
        """

    @property
    @abstractmethod
    def supported_num_spatial_dims(self) -> tuple[int, ...]:
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

    def check_required_attributes(self) -> None:
        # we check that the model has defined the following attributes.
        # this must be called at the end of the __init__ of each subclass.
        required_attrs = ["in_channels", "out_channels", "input_shape"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing required attribute : {attr}")


class AutoPaddingModel(ABC):
    @property
    @abstractmethod
    def settings(self) -> Any:
        """
        Returns the settings instance used to configure for this model.
        """

    @property
    @abstractmethod
    def input_shape(self) -> Any:
        """
        Returns the settings instance used to configure for this model.
        """

    @abstractmethod
    def validate_input_shape(self, input_shape: Size) -> tuple[bool, Size]:
        """Given an input shape, verifies whether the inputs fit with the
            calling model's specifications.

        Args:
            input_shape (Size): The shape of the input data, excluding any batch dimension and channel dimension.
                                For example, for a batch of 2D tensors of shape [B,C,W,H], [W,H] should be passed.
                                For 3D data instead of shape [B,C,W,H,D], instead, [W,H,D] should be passed.

        Returns:
            tuple[bool, Size]: Returns a tuple where the first element is a boolean signaling whether the given input shape
                                already fits the model's requirements. If that value is False, the second element contains the closest
                                shape that fits the model, otherwise it will be None.
        """

    def _maybe_padding(
        self, data_tensor: torch.Tensor
    ) -> tuple[torch.Tensor | ValueError, Optional[torch.Size]]:
        """Performs an optional padding to ensure that the data tensor can be fed
            to the underlying model. Padding will happen if if
            autopadding was enabled via the settings.

        Args:
            data_tensor (torch.Tensor): the input data to be potentially padded.

        Returns:
            tuple[torch.Tensor, Optional[torch.Size]]: the padded tensor, where the original data is found in the center,
            and the old size if padding was possible. If not possible or the shape is already fine,
            the data is returned untouched and the second return value will be none.
        """
        if not self.settings.autopad_enabled:
            return data_tensor, None

        old_shape = data_tensor.shape[-len(self.input_shape) :]
        valid_shape, new_shape = self.validate_input_shape(
            data_tensor.shape[-len(self.input_shape) :]
        )
        if not valid_shape:
            return pad_batch(
                batch=data_tensor, new_shape=new_shape, pad_value=0
            ), old_shape
        return data_tensor, None

    def _maybe_unpadding(
        self, data_tensor: torch.Tensor, old_shape: torch.Size
    ) -> torch.Tensor | ValueError:
        """Potentially removes the padding previously added to the given tensor. This action
           is only carried out if autopadding was enabled via the settings.

        Args:
            data_tensor (torch.Tensor): The data tensor from which padding is to be removed.
            old_shape (torch.Size): The previous shape of the data tensor. It can either be
            [W,H] or [W,H,D] for 2D and 3D data respectively. old_shape is returned by self._maybe_padding.

        Returns:
            torch.Tensor: The data tensor with the padding removed, if possible.
        """
        if self.settings.autopad_enabled and old_shape is not None:
            return undo_padding(data_tensor, old_shape=old_shape)
        return data_tensor
