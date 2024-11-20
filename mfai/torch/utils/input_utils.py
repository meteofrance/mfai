import torch.nn.functional as F
import torch
from typing import Tuple 

def pad_batch(batch: torch.Tensor, new_shape: torch.Size, pad_value: float=0) -> torch.Tensor:
    """ Given batch of 2D or 3D data and a shape new_shape, 
        pads the tensor with the given pad_value.  

    Args:
        batch (torch.Tensor): the batch of values of shape (B, C, D, H, W) or (B, C, H, W).
        new_shape (torch.Size): Target shape (D, H, W) for 3D tensors or (H, W) for 2D tensors to be given. 
        pad_value (float, optional): the padding value to be used. Defaults to 0.

    Returns:
        torch.Tensor: The padded tensor.
    """

    fits = batch.shape[-len(new_shape):] == new_shape

    if fits:
        return batch

    if len(new_shape) == 2:        
        left, right, top, bottom = _get_2D_padding(new_shape=new_shape, old_shape=batch.shape[-len(new_shape):])

        return F.pad(batch, (left, right, top, bottom), mode='constant', value=pad_value)
    elif len(new_shape) == 3:
        left, right, top, bottom, front, back = _get_3D_padding(new_shape=new_shape, old_shape=batch.shape[-len(new_shape):])

        return F.pad(batch, (left, right, top, bottom, front, back), mode='constant', value=pad_value)
    
    return ValueError("new_shape must be a torch.Size of length 2 or 3.")


def _get_2D_padding(new_shape: torch.Size, old_shape: torch.Size) -> Tuple[int]:
    """ Returns the left, right, top, bottom paddings that needs to be added to a 
        2D tensor of shape old_shape to get new_shape as a new shape. 

    Args:
        new_shape (torch.Size): the new shape
        old_shape (torch.Size): the old shape

    Returns:
        Tuple[int]: left,right,top and bottom sizes to be added. 
    """
    
    diff_x = new_shape[1] - old_shape[-1]
    diff_y = new_shape[0] - old_shape[-2]


    left, right = diff_x//2, diff_x - (diff_x//2)
    top, bottom = diff_y//2, diff_y - (diff_y//2)
    
    return left, right, top, bottom

def _get_3D_padding(new_shape: torch.Size, old_shape: torch.Size) -> Tuple[int]:
    diff_z = new_shape[0] - old_shape[-3]
    diff_y = new_shape[1] - old_shape[-2]
    diff_x = new_shape[2] - old_shape[-1]

    left, right = diff_x // 2, diff_x - (diff_x // 2)
    top, bottom = diff_y // 2, diff_y - (diff_y // 2)
    front, back = diff_z // 2, diff_z - (diff_z // 2)
    
    return left, right, top, bottom, front, back


def undo_padding(batch: torch.Tensor, old_shape: torch.Size, inplace=True):
    """ Removes the padding added by pad_batch

    Args:
        batch (torch.Tensor): The padded batch of data
        old_shape (torch.Size): The original shape of the data
        inplace (bool): Whether the returned tensor is just a sliced view of the 
                        given tensor or a new copy. 
    """
    
    new_shape = batch.shape[-len(old_shape):]
    assert all(o <= n for o, n in zip(old_shape, new_shape))
    
    if len(old_shape) == 2: 
        l,r,t,b = _get_2D_padding(new_shape=new_shape, old_shape=old_shape)
        if inplace:
            return batch[..., t:batch.shape[-2]-b, l:batch.shape[-1]-r]
        return batch[..., t:batch.shape[-2]-b, l:batch.shape[-1]-r].copy()
    elif len(old_shape) == 3: 
        left, right, top, bottom, front, back = _get_3D_padding(new_shape=new_shape,
                                                                old_shape=old_shape)
        if inplace:
            return batch[..., front:batch.shape[-3]-back, top:batch.shape[-2]-bottom, left:batch.shape[-1]-right]
        return batch[..., front:batch.shape[-3]-back, top:batch.shape[-2]-bottom, left:batch.shape[-1]-right].copy()
    else:
        return ValueError("old_shape must be a torch.Size of length 2 or 3.")
    
