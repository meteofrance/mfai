"""
Test our pure PyTorch models to make sure they can be :
1. Instanciated
2. Trained
3. onnx exported
4. onnx loaded and used for inference
"""

import tempfile
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
import torch
from marshmallow.exceptions import ValidationError
from torch import Tensor

from mfai.pytorch import export_to_onnx, onnx_load_and_infer, padding
from mfai.pytorch.models import (
    autopad_nn_architectures,
    load_from_settings_file,
    nn_architectures,
)
from mfai.pytorch.models.base import ModelType
from mfai.pytorch.models.deeplabv3 import DeepLabV3Plus
from mfai.pytorch.models.half_unet import HalfUNet


def to_numpy(tensor: Tensor) -> Any:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class FakeSumDataset(torch.utils.data.Dataset):
    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = input_shape
        super().__init__()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = torch.rand(*self.input_shape)
        y = torch.sum(x, 0).unsqueeze(0)
        return x, y


class FakePanguDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        surface_variables: int,
        plevel_variables: int,
        plevels: int,
        static_length: int,
    ) -> None:
        self.surface_shape = (surface_variables, *input_shape)
        self.plevel_shape = (plevel_variables, plevels, *input_shape)
        self.static_shape = (static_length, *input_shape)
        super().__init__()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        input_surface = torch.rand(*self.surface_shape)
        input_plevel = torch.rand(*self.plevel_shape)
        input_static = torch.rand(*self.static_shape)
        target_surface = torch.rand(*self.surface_shape)
        target_plevel = torch.rand(*self.plevel_shape)
        return {
            "input_surface": input_surface,
            "input_plevel": input_plevel,
            "input_static": input_static,
            "target_surface": target_surface,
            "target_plevel": target_plevel,
        }


def train_model(
    model: torch.nn.Module, input_shape: Tuple[int, ...]
) -> torch.nn.Module:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.MSELoss()

    ds = FakeSumDataset(input_shape)

    training_loader = torch.utils.data.DataLoader(ds, batch_size=2)

    # Simulate 2 EPOCHS of training
    for _ in range(2):
        for _, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, targets = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, targets)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

    # Make a prediction in eval mode
    model.eval()
    sample = ds[0][0].unsqueeze(0)
    model(sample)
    return model


def meshgrid(grid_width: int, grid_height: int) -> Tensor:
    x = np.arange(0, grid_width, 1)
    y = np.arange(0, grid_height, 1)
    xx, yy = np.meshgrid(x, y)
    return torch.from_numpy(np.asarray([xx, yy]))


@pytest.mark.parametrize("model_kls", nn_architectures[ModelType.GRAPH])
def test_torch_graph_training_loop(model_kls: Any) -> None:
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1

    settings = model_kls.settings_kls()

    # for GNN models we test them with a fake 2d regular grid
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(settings, meshgrid(64, 64))

    if model_kls.features_last:
        input_shape = (64 * 64, NUM_INPUTS)
    else:
        input_shape = (NUM_INPUTS, 64 * 64)

    model = model_kls(
        in_channels=NUM_INPUTS,
        out_channels=NUM_OUTPUTS,
        input_shape=input_shape,
        settings=settings,
    )

    model = train_model(model, input_shape)


@pytest.mark.parametrize(
    "model_kls",
    nn_architectures[ModelType.CONVOLUTIONAL]
    + nn_architectures[ModelType.VISION_TRANSFORMER],
)
def test_torch_convolutional_and_vision_transformer_training_loop(
    model_kls: Any,
) -> None:
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    INPUT_SHAPE = (64, 64, 64)
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1

    settings = model_kls.settings_kls()

    # We test the model for all supported input spatial dimensions
    for spatial_dims in model_kls.supported_num_spatial_dims:
        if hasattr(settings, "spatial_dims"):
            settings.spatial_dims = spatial_dims

        model = model_kls(
            in_channels=NUM_INPUTS,
            out_channels=NUM_OUTPUTS,
            input_shape=INPUT_SHAPE[:spatial_dims],
            settings=settings,
        )
        model = train_model(model, (NUM_INPUTS, *INPUT_SHAPE[:spatial_dims]))

        # We test if models claiming to be onnx exportable really are post training.
        # See https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
        if model.onnx_supported:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".onnx") as dst:
                sample = torch.rand(1, NUM_INPUTS, *INPUT_SHAPE[:spatial_dims])
                export_to_onnx(model, sample, dst.name)
                onnx_load_and_infer(dst.name, sample)


@pytest.mark.parametrize("model_kls", nn_architectures[ModelType.PANGU])
def test_torch_pangu_training_loop(model_kls: Any) -> None:
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    INPUT_SHAPE = (64, 64, 64)
    NUM_INPUTS = 7
    NUM_OUTPUTS = 6
    SURFACE_VARIABLES = 2
    PLEVEL_VARIABLES = 2
    PLEVELS = 2
    STATIC_LENGTH = 1

    settings = model_kls.settings_kls(
        surface_variables=SURFACE_VARIABLES,
        plevel_variables=PLEVEL_VARIABLES,
        plevels=PLEVELS,
        static_length=STATIC_LENGTH,
    )

    # We test the model for all supported input spatial dimensions
    for spatial_dims in model_kls.supported_num_spatial_dims:
        if hasattr(settings, "spatial_dims"):
            settings.spatial_dims = spatial_dims

        model = model_kls(
            in_channels=NUM_INPUTS,
            out_channels=NUM_OUTPUTS,
            input_shape=INPUT_SHAPE[:spatial_dims],
            settings=settings,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.MSELoss()

        ds = FakePanguDataset(
            input_shape=INPUT_SHAPE[:spatial_dims],
            surface_variables=SURFACE_VARIABLES,
            plevel_variables=PLEVEL_VARIABLES,
            plevels=PLEVELS,
            static_length=STATIC_LENGTH,
        )

        training_loader = torch.utils.data.DataLoader(ds, batch_size=2)

        # Simulate 2 EPOCHS of training
        for _ in range(2):
            for data in training_loader:
                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                output_plevel, output_surface = model(
                    data["input_plevel"], data["input_surface"], data["input_static"]
                )

                # Compute the loss and its gradients
                loss = loss_fn(output_plevel, data["target_plevel"]) + loss_fn(
                    output_surface, data["target_surface"]
                )
                loss.backward()

                # Adjust learning weights
                optimizer.step()

        # Make a prediction in eval mode
        model.eval()
        sample = ds[0]
        model(
            sample["input_plevel"].unsqueeze(0),
            sample["input_surface"].unsqueeze(0),
            sample["input_static"].unsqueeze(0),
        )

        # We test if models claiming to be onnx exportable really are post training.
        # See https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
        if model.onnx_supported:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".onnx") as dst:
                sample_surface = torch.rand(
                    1, SURFACE_VARIABLES, *INPUT_SHAPE[:spatial_dims]
                )
                sample_plevel = torch.rand(
                    1, PLEVEL_VARIABLES, PLEVELS, *INPUT_SHAPE[:spatial_dims]
                )
                sample_static = torch.rand(
                    1, STATIC_LENGTH, *INPUT_SHAPE[:spatial_dims]
                )
                samples = (sample_plevel, sample_surface, sample_static)
                export_to_onnx(model, samples, dst.name)
                onnx_load_and_infer(dst.name, samples)


@pytest.mark.parametrize(
    "model_and_settings",
    [
        (HalfUNet, HalfUNet.settings_kls(use_ghost=True, absolute_pos_embed=True)),
        (HalfUNet, HalfUNet.settings_kls(use_ghost=False, absolute_pos_embed=True)),
        (DeepLabV3Plus, DeepLabV3Plus.settings_kls(activation="sigmoid")),
        (DeepLabV3Plus, DeepLabV3Plus.settings_kls(activation="softmax")),
        (DeepLabV3Plus, DeepLabV3Plus.settings_kls(activation="tanh")),
        (DeepLabV3Plus, DeepLabV3Plus.settings_kls(activation="logsoftmax")),
    ],
)
def test_extra_models(model_and_settings: Any) -> None:
    """
    Tests some extra models and settings.
    """
    INPUT_SHAPE = (64, 64, 64)
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    model_kls, settings = model_and_settings
    for spatial_dims in model_kls.supported_num_spatial_dims:
        model = model_kls(
            in_channels=NUM_INPUTS,
            out_channels=NUM_OUTPUTS,
            input_shape=INPUT_SHAPE[:2],
            settings=settings,
        )
        train_model(model, (NUM_INPUTS, *INPUT_SHAPE[:spatial_dims]))


def test_load_model_by_name() -> None:
    with pytest.raises(ValueError):
        load_from_settings_file("NotAValidModel", 2, 2, Path(""), (1, 1))

    # Should work: valid settings file for this model
    load_from_settings_file(
        model_name="HalfUNet",
        in_channels=2,
        out_channels=2,
        settings_path=Path(__file__).parents[1]
        / "mfai"
        / "config"
        / "models"
        / "halfunet128.json",
        input_shape=(10, 10, 10),
    )

    # Should raise: invalid settings file for this model
    with pytest.raises(ValidationError):
        load_from_settings_file(
            model_name="UNetRPP",
            in_channels=2,
            out_channels=2,
            settings_path=Path(__file__).parents[1]
            / "mfai"
            / "config"
            / "models"
            / "halfunet128.json",
            input_shape=(10, 10, 10),
        )


@pytest.mark.parametrize("model_class", autopad_nn_architectures)
def test_input_shape_validation(model_class: Any) -> None:
    B, C, W, H = 8, 3, 61, 65

    input_data = torch.randn(B, C, W, H)
    net = model_class(in_channels=C, out_channels=1, input_shape=input_data.shape)

    # assert it fails before padding
    with pytest.raises((RuntimeError, ValueError)):
        net(input_data)

    valid_shape, new_shape = net.validate_input_shape(input_data.shape[-2:])

    assert not valid_shape


@pytest.mark.parametrize("model_class", autopad_nn_architectures)
def test_autopad_models(model_class: Any) -> None:
    B, C, W, H = 32, 3, 64, 65  # invalid [W,H]

    input_data = torch.randn(B, C, W, H)
    settings = model_class.settings_kls()
    settings.autopad_enabled = True  # enable autopad
    net = model_class(
        in_channels=C, out_channels=1, input_shape=(64, 65), settings=settings
    )

    net(input_data)  # assert it does not fail
