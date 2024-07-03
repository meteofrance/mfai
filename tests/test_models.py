"""
Test our pure PyTorch models to make sure they can be :
1. Instanciated
2. Trained
3. onnx exported
4. onnx loaded and used for inference
"""
import tempfile

import onnx
import onnxruntime
import torch
import pytest
from mfai.torch.models import all_nn_architectures


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


def onnx_export_load_infer(model, filepath, sample):

    torch.onnx.export(
    model,  # model being run
    sample,  # model input (or a tuple for multiple inputs)
    filepath,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
    )
    
    # Check the model with onnx
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)

    # Perform an inference with onnx

    ort_session = onnxruntime.InferenceSession(
        filepath, providers=["CPUExecutionProvider"]
    )

    ort_session.run(None, {"input": to_numpy(sample)})

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
            onnx_export_load_infer(model, dst.name, sample)
