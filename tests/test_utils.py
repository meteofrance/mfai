import pytest
import torch
import random 

from mfai.torch.utils import input_utils

@pytest.mark.parametrize("dims", [2, 3])   
def test_pad(dims):
    # initial data with 16 batch elements, 3 channels and random input shape
    tensor_shape = torch.Size([16,3]+[random.randint(60, 100) for _ in range(dims)])
    data = torch.randn(*tensor_shape)
    
    # generate a new random input shape, larger than the original one
    new_delta = [random.randint(5, 25) for _ in range(dims)]
    new_shape = torch.Size([a + b for a, b in zip(tensor_shape[-dims:], new_delta)])
    # pad the data 
    padded_data = input_utils.pad_batch(batch=data, new_shape=new_shape, pad_value=0)

    assert padded_data.shape[-len(new_shape):] == new_shape
    
    # test undo padding 
    
    padded_data_undone = input_utils.undo_padding(padded_data, old_shape=tensor_shape[-dims:])
    if dims == 3:
        print('ciao')
    assert (padded_data_undone == data).all()
    
