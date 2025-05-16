from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def pad_batch(
    batch: Tensor,
    new_shape: torch.Size,
    mode: str = "constant",
    pad_value: Optional[float] = 0,
) -> Tensor:
    """Given batch of 2D or 3D data and a shape new_shape,
        pads the tensor with the given pad_value.

    Args:
        batch (Tensor): the batch of values of shape (B, C, D, H, W) or (B, C, H, W).
        new_shape (torch.Size): Target shape (D, H, W) for 3D tensors or (H, W) for 2D tensors to be given.
        pad_value (float, optional): the padding value to be used. Defaults to 0.

    Returns:
        Tensor: The padded tensor.
    """

    fits = batch.shape[-len(new_shape) :] == new_shape

    if mode != "constant":
        pad_value = None

    if fits:
        return batch

    if len(new_shape) == 2:
        left, right, top, bottom = _get_2D_padding(
            new_shape=new_shape, old_shape=batch.shape[-len(new_shape) :]
        )

        return F.pad(batch, (left, right, top, bottom), mode=mode, value=pad_value)
    elif len(new_shape) == 3:
        left, right, top, bottom, front, back = _get_3D_padding(
            new_shape=new_shape, old_shape=batch.shape[-len(new_shape) :]
        )

        return F.pad(
            batch, (left, right, top, bottom, front, back), mode=mode, value=pad_value
        )

    raise ValueError("new_shape must be a torch.Size of length 2 or 3.")


def _get_2D_padding(
    new_shape: torch.Size, old_shape: torch.Size
) -> Tuple[int, int, int, int]:
    """Returns the left, right, top, bottom paddings that needs to be added to a
        2D tensor of shape old_shape to get new_shape as a new shape.

    Args:
        new_shape (torch.Size): the new shape
        old_shape (torch.Size): the old shape

    Returns:
        Tuple[int]: left,right,top and bottom sizes to be added.
    """

    diff_x = new_shape[1] - old_shape[-1]
    diff_y = new_shape[0] - old_shape[-2]

    left, right = diff_x // 2, diff_x - (diff_x // 2)
    top, bottom = diff_y // 2, diff_y - (diff_y // 2)

    return left, right, top, bottom


def _get_3D_padding(
    new_shape: torch.Size, old_shape: torch.Size
) -> Tuple[int, int, int, int, int, int]:
    diff_z = new_shape[0] - old_shape[-3]
    diff_y = new_shape[1] - old_shape[-2]
    diff_x = new_shape[2] - old_shape[-1]

    left, right = diff_x // 2, diff_x - (diff_x // 2)
    top, bottom = diff_y // 2, diff_y - (diff_y // 2)
    front, back = diff_z // 2, diff_z - (diff_z // 2)

    return left, right, top, bottom, front, back


def undo_padding(batch: Tensor, old_shape: torch.Size, inplace: bool = False) -> Tensor:
    """Removes the padding added by pad_batch

    Args:
        batch (Tensor): The padded batch of data
        old_shape (torch.Size): The original shape of the data
        inplace (bool): Whether the returned tensor is just a sliced view of the
                        given tensor or a new copy.
    """

    new_shape = batch.shape[-len(old_shape) :]
    assert all(o <= n for o, n in zip(old_shape, new_shape))

    if len(old_shape) == 2:
        left, right, top, bottom = _get_2D_padding(
            new_shape=new_shape, old_shape=old_shape
        )
        if inplace:
            return batch[
                ..., top : batch.shape[-2] - bottom, left : batch.shape[-1] - right
            ]
        return batch[
            ..., top : batch.shape[-2] - bottom, left : batch.shape[-1] - right
        ].clone()
    elif len(old_shape) == 3:
        left, right, top, bottom, front, back = _get_3D_padding(
            new_shape=new_shape, old_shape=old_shape
        )
        if inplace:
            return batch[
                ...,
                front : batch.shape[-3] - back,
                top : batch.shape[-2] - bottom,
                left : batch.shape[-1] - right,
            ]
        return batch[
            ...,
            front : batch.shape[-3] - back,
            top : batch.shape[-2] - bottom,
            left : batch.shape[-1] - right,
        ].clone()
    else:
        raise ValueError("old_shape must be a torch.Size of length 2 or 3.")
