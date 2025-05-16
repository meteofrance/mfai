from pathlib import Path
from typing import Any, Iterable

import numpy
import onnx
import onnxruntime
import torch
from torch import Tensor, nn


def to_numpy(
    input: Tensor | tuple[Tensor, ...],
) -> numpy.ndarray | tuple[numpy.ndarray, ...]:
    if isinstance(input, tuple):
        l = []
        for tensor in input:
            l.append(
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )
        return tuple(l)
    return input.detach().cpu().numpy() if input.requires_grad else input.cpu().numpy()


def export_to_onnx(
    model: nn.Module, sample: Tensor | tuple[Any, ...], filepath: Path | str
) -> None:
    """
    Exports a model to ONNX format.
    """
    if isinstance(sample, Tensor):
        sample = (sample,)

    if isinstance(filepath, Path):
        filepath = filepath.as_posix()

    torch.onnx.export(
        model=model,  # model being run
        args=sample,  # model input (or a tuple for multiple inputs)
        f=filepath,  # where to save the model (can be a file or file-like object)
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


def onnx_load_and_infer(
    filepath: Path | str, input: Tensor | tuple[Tensor, ...]
) -> numpy.ndarray:
    """
    Loads a model using onnx, checks it, and performs an inference.
    """
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)

    # Perform an inference with onnx

    ort_session = onnxruntime.InferenceSession(
        filepath, providers=["CPUExecutionProvider"]
    )

    return ort_session.run(None, {"input": to_numpy(input)})
