from collections.abc import Sequence

import pytest
import torch

from mfai.pytorch.namedtensor import NamedTensor
from mfai.pytorch.transforms import (
    DimensionSubSampler,
    MeanDimensionSubsampler,
    RandomCropWithMinPositivePixels,
)


def test_RandomCropWithMinPositivePixels() -> None:
    t = RandomCropWithMinPositivePixels()
    y = torch.randint(0, 2, (1, 2048, 2048))
    x = torch.randn(8, 2048, 2048)
    cropped_x, cropped_y, _, _ = t((x, y))
    assert cropped_x.shape[1] == cropped_x.shape[2] == 512
    assert cropped_x.shape[0] == 8
    assert cropped_y.shape[1] == cropped_y.shape[2] == 512


@pytest.mark.parametrize("idx_to_keep", [(0, 1, 2, 3, 4), (0, 2, 4), (0, 4)])
def test_DimensionSubSampler(idx_to_keep: Sequence[int]) -> None:
    # Instantiate data
    data = NamedTensor(
        tensor=torch.rand(5, 512, 512, 3),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=["a", "b", "c"],
    )

    # Instantiate sub sampler
    sub_sampler = DimensionSubSampler(
        dim_name="timesteps",
        idx_to_keep=idx_to_keep,
    )

    # Transform data
    transformed_data: NamedTensor = sub_sampler(data)

    # Test result
    for i, index in enumerate(idx_to_keep):
        assert torch.equal(
            transformed_data.tensor[i, :, :, :], data.tensor[index, :, :, :]
        )


@pytest.mark.parametrize(
    "idx_to_be_meaned",
    [
        [(0, 1, 2, 3, 4), (0, 2, 4), (0, 4)],
        [(0,), (4,)],
    ],
)
def test_MeanDimensionSubSampler(idx_to_be_meaned: Sequence[Sequence[int]]) -> None:
    # Instantiate data
    data = NamedTensor(
        tensor=torch.rand(5, 512, 512, 3),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=["a", "b", "c"],
    )

    # Instantiate sub sampler
    sub_sampler = MeanDimensionSubsampler(
        dim_name="timesteps",
        idx_to_be_meaned=idx_to_be_meaned,
    )

    # Transform data
    transformed_data = sub_sampler(data)

    # Test result
    for i, idx_to_mean in enumerate(idx_to_be_meaned):
        assert torch.equal(
            transformed_data.tensor[i, :, :, :],
            torch.mean(
                data.tensor[idx_to_mean, :, :, :],
                dim=0,
            ),
        )

