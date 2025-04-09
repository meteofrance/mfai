"""
Test our pure PyTorch models to make sure they can be :
1. Instanciated
2. Trained
3. onnx exported
4. onnx loaded and used for inference
"""

import numpy as np
import tempfile
from pathlib import Path
from typing import Tuple

from mfai.torch.models.base import ModelType
import pytest
import torch
from marshmallow.exceptions import ValidationError

from mfai.torch import export_to_onnx, onnx_load_and_infer
from mfai.torch.models import (
    all_nn_architectures,
    autopad_nn_architectures,
    load_from_settings_file,
)

from mfai.torch import padding
from mfai.torch.models.deeplabv3 import DeepLabV3Plus
from mfai.torch.models.half_unet import HalfUNet


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class FakeSumDataset(torch.utils.data.Dataset):
    def __init__(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape
        super().__init__()

    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        x = torch.rand(*self.input_shape)
        y = torch.sum(x, 0).unsqueeze(0)
        return x, y


def train_model(model: torch.nn.Module, input_shape: Tuple[int, ...]):
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


def meshgrid(grid_width: int, grid_height: int):
    x = np.arange(0, grid_width, 1)
    y = np.arange(0, grid_height, 1)
    xx, yy = np.meshgrid(x, y)
    return torch.from_numpy(np.asarray([xx, yy]))


@pytest.mark.parametrize("model_kls", all_nn_architectures)
def test_torch_training_loop(model_kls):
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    INPUT_SHAPE = (64, 64, 64)
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1

    settings = model_kls.settings_kls()

    # for GNN models we test them with a fake 2d regular grid
    if model_kls.model_type == ModelType.GRAPH:
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

    else:
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
def test_extra_models(model_and_settings):
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


def test_load_model_by_name():
    with pytest.raises(ValueError):
        load_from_settings_file("NotAValidModel", 2, 2, None, (1, 1))

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
            model_name="UNETRPP",
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
def test_input_shape_validation(model_class):
    B, C, W, H = 32, 3, 64, 65

    input_data = torch.randn(B, C, W, H)
    net = model_class(in_channels=C, out_channels=1, input_shape=input_data.shape)

    # assert it fails before padding
    with pytest.raises(RuntimeError):
        net(input_data)

    valid_shape, new_shape = net.validate_input_shape(input_data.shape[-2:])

    assert not valid_shape

    # assert it does not fail after padding
    input_data_pad = padding.pad_batch(
        batch=input_data, new_shape=new_shape, pad_value=0
    )
    net(input_data_pad)


@pytest.mark.parametrize("model_class", autopad_nn_architectures)
def test_autopad_models(model_class):
    B, C, W, H = 32, 3, 64, 65  # invalid [W,H]

    input_data = torch.randn(B, C, W, H)
    settings = model_class.settings_kls()
    settings.autopad_enabled = True  # enable autopad
    net = model_class(
        in_channels=C, out_channels=1, input_shape=(64, 65), settings=settings
    )

    net(input_data)  # assert it does not fail
