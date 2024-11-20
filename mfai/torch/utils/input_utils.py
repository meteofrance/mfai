import torch.nn.functional as F
import torch

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
        diff_x = new_shape[1] - batch.shape[-1]
        diff_y = new_shape[0] - batch.shape[-2]


        left, right = diff_x//2, diff_x - (diff_x//2)
        top, bottom = diff_y//2, diff_y - (diff_y//2)

        return F.pad(batch, (left, right, top, bottom), mode='constant', value=pad_value)
    elif len(new_shape) == 3:
        diff_z = new_shape[0] - batch.shape[-3]
        diff_y = new_shape[1] - batch.shape[-2]
        diff_x = new_shape[2] - batch.shape[-1]

        left, right = diff_x // 2, diff_x - (diff_x // 2)
        top, bottom = diff_y // 2, diff_y - (diff_y // 2)
        front, back = diff_z // 2, diff_z - (diff_z // 2)

        return F.pad(batch, (left, right, top, bottom, front, back), mode='constant', value=pad_value)
    
    return ValueError("new_shape must be a torch.Size of length 2 or 3.")
