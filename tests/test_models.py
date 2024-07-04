"""
Test our pure PyTorch models to make sure they can be :
1. Instanciated
2. Trained
3. onnx exported
4. onnx loaded and used for inference
"""

from pathlib import Path
import tempfile

from marshmallow.exceptions import ValidationError
import torch
import pytest
from mfai.torch import export_to_onnx, onnx_load_and_infer
from mfai.torch.models import all_nn_architectures, load_from_settings_file


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class FakeSumDataset(torch.utils.data.Dataset):
    def __init__(self, grid_height: int, grid_width: int, num_inputs: int):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_inputs = num_inputs
        super().__init__()

    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        x = torch.rand(self.num_inputs, self.grid_height, self.grid_width)
        y = torch.sum(x, 0).unsqueeze(0)
        return x, y


@pytest.mark.parametrize("model_kls", all_nn_architectures)
def test_torch_training_loop(model_kls):
    """
    Checks that our models are trainable on a toy problem (sum).
    """
    GRID_WIDTH = 64
    GRID_HEIGHT = 64
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1

    model = model_kls(
        in_channels=NUM_INPUTS,
        out_channels=NUM_OUTPUTS,
        input_shape=(GRID_HEIGHT, GRID_WIDTH),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    ds = FakeSumDataset(GRID_HEIGHT, GRID_WIDTH, NUM_INPUTS)
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

    # We test if models claiming to be onnx exportable really are post training.
    # See https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
    if model.onnx_supported:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".onnx") as dst:
            sample = torch.rand(1, NUM_INPUTS, GRID_HEIGHT, GRID_WIDTH)
            export_to_onnx(model, sample, dst.name)
            onnx_load_and_infer(dst.name, sample)


def test_load_model_by_name():
    with pytest.raises(ValueError):
        load_from_settings_file("NotAValidModel", 2, 2, None)

    # Should work: valid settings file for this model
    load_from_settings_file(
        "HalfUNet",
        2,
        2,
        Path(__file__).parents[1] / "mfai" / "config" / "models" / "halfunet128.json",
    )

    # Should raise: invalid settings file for this model
    with pytest.raises(ValidationError):
        load_from_settings_file(
            "UNETRPP",
            2,
            2,
            Path(__file__).parents[1]
            / "mfai"
            / "config"
            / "models"
            / "halfunet128.json",
        )
