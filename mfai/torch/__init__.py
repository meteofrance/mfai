from pathlib import Path

import numpy
import onnx
import onnxruntime
import torch
from torch import nn


def to_numpy(tensor: torch.Tensor) -> numpy.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_to_onnx(model: nn.Module, sample: torch.Tensor, filepath: Path) -> None:
    """
    Exports a model to ONNX format.
    """
    torch.onnx.export(
        model,  # model being run
        sample,  # model input (or a tuple for multiple inputs)
        filepath.as_posix(),  # where to save the model (can be a file or file-like object)
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


def onnx_load_and_infer(filepath: Path, sample: torch.Tensor) -> numpy.ndarray:
    """
    Loads a model using onnx, checks it, and performs an inference.
    """
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)

    # Perform an inference with onnx

    ort_session = onnxruntime.InferenceSession(
        filepath, providers=["CPUExecutionProvider"]
    )

    return ort_session.run(None, {"input": to_numpy(sample)})
