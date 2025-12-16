import typing
from pathlib import Path
from typing import Any

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
    model: nn.Module,
    sample: Tensor | tuple[Any, ...],
    filepath: Path | str,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Exports a model to ONNX format.
    """
    if isinstance(sample, Tensor):
        sample = (sample,)

    if isinstance(filepath, Path):
        filepath = filepath.as_posix()

    # Allow dynamic batch size by default
    dynamic_batch_size: dict[str, dict[int, str]] = {
        "input": {0: "batch"},  # variable batch size
        "output": {0: "batch"},  # variable batch size
    }
    if kwargs is None:
        kwargs = {"dynamic_axes": dynamic_batch_size}
    elif "dynamic_axes" not in kwargs.keys():
        kwargs["dynamic_axes"] = dynamic_batch_size

    torch.onnx.export(
        model=model,  # model being run
        args=sample,  # model input (or a tuple for multiple inputs)
        f=filepath,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        **kwargs,
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


@typing.no_type_check
def assign(left: Tensor, right: numpy.ndarray) -> torch.nn.Parameter:
    """
    Used when loading weights coming from another training
    framework in to pytorch models.
    Checks the shapes matches and creates the learnable parameters from the
    supplied weights (rights).
    Copied from the llm from scratch repo "as-is"
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
