import pytest
import torch
from torch import Tensor

from mfai.torch.padding import pad_batch, undo_padding


@pytest.mark.parametrize("dims", [2, 3])
def test_pad(dims):
    # get all the True/False combinations
    combinations = torch.cartesian_prod(*[torch.tensor([0, 1]) for _ in range(dims)])
    even = 8
    odd = 9
    mapped_combinations = torch.where(
        combinations == 1, torch.tensor(even), torch.tensor(odd)
    )

    for comb in mapped_combinations:
        # initial data with 8 batch elements, 3 channels and a combination of even and odd data dimensions
        tensor_shape = torch.Size([8, 3] + [d for d in comb])
        data = torch.randn(*tensor_shape)

        # generate a new input shape, larger than the original one
        for even_delta in {True, False}:
            new_delta = [even if even_delta else odd for _ in range(dims)]
            new_shape = torch.Size(
                [a + b for a, b in zip(tensor_shape[-dims:], new_delta)]
            )
            # pad the data
            padded_data = pad_batch(batch=data, new_shape=new_shape, pad_value=0)

            assert padded_data.shape[-len(new_shape) :] == new_shape

            # test undo padding

            padded_data_undone = undo_padding(
                padded_data, old_shape=tensor_shape[-dims:]
            )

            assert (padded_data_undone == data).all()
