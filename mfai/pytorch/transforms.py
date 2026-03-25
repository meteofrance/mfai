import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from torch import Tensor, nn

from mfai.pytorch.namedtensor import NamedTensor


class RandomCropWithMinPositivePixels(object):
    """Randomly crops an input image to a 512x512 image, with min 15% of positive pixels.
    The random crop is performed 5 times at the most, or until an image with 15% of
    positive pixels is found. If such an image is not found, it returns the image with
    the most positive pixels from the 5 images cropped.

    input is a tuple sample with x and y
    x shape (C, H, W)
    y shape (1, H, W)
    """

    def __init__(
        self,
        crop_size: tuple[int, int] = (512, 512),
        min_positive_percentage: float = 15.0,
        tries: int = 5,
    ):
        self.min_positive_percentage = min_positive_percentage
        self.crop_size = crop_size
        self.tries = tries

    def get_one_crop(self, y: Tensor) -> tuple[int, int, float]:
        nb_pixels_crop = self.crop_size[0] * self.crop_size[1]

        # Randomly choose the top-left corner of the crop
        left = random.randint(0, y.shape[2] - self.crop_size[1])
        top = random.randint(0, y.shape[1] - self.crop_size[0])

        # Crop the image
        cropped_y = TF.crop(y, top, left, self.crop_size[0], self.crop_size[1])

        # Calculate the percentage of positive pixels in the cropped image
        positive_pixels = torch.sum(cropped_y > 0).item()
        perc = positive_pixels / nb_pixels_crop * 100
        return top, left, perc

    def get_crop(self, y: Tensor) -> tuple[int, int]:
        best_top, best_left, best_perc = self.get_one_crop(y)
        for _ in range(self.tries - 1):
            top, left, perc = self.get_one_crop(y)
            # Check if the positive percentage satisfies the condition
            if perc >= self.min_positive_percentage:
                return left, top
            # else, save values and try again
            if perc > best_perc:
                best_left, best_top, best_perc = left, top, perc
        return best_left, best_top

    def __call__(
        self, sample: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, int, int]:
        x, y = sample
        left, top = self.get_crop(y)
        cropped_x = TF.crop(x, top, left, self.crop_size[0], self.crop_size[1])
        cropped_y = TF.crop(y, top, left, self.crop_size[0], self.crop_size[1])
        return cropped_x, cropped_y, left, top


class DimensionSubSampler(nn.Module):
    """Subsamples a NamedTensor dimension.

    Exemple:
    ```
    import torch
    from mfai.pytorch.namedtensor import NamedTensor


    data = NamedTensor(
        tensor=torch.rand(10, 512, 512, 3),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=["u", "v", "t2m"],
    )

    sub_sampler = DimensionSubSampler(
        dim_name="timesteps",
        idx_to_keep=[0, 3, 6, 9],
    )

    transformed_data = sub_sampler(data)

    print("data:")
    print(data)
    print("transformed_data:")
    print(transformed_data)
    ```
    =>
    ```
    data:
    --- NamedTensor ---
    Names: ['timesteps', 'lat', 'lon', 'features']
    Tensor Shape: torch.Size([10, 512, 512, 3]))
    transformed_data:
    --- NamedTensor ---
    Names: ['timesteps', 'lat', 'lon', 'features']
    Tensor Shape: torch.Size([4, 512, 512, 3]))
    ```
    """

    def __init__(self, dim_name: str, idx_to_keep: Sequence[int]) -> None:
        """
        Args:
            dim_name: name of the name tensor dimension to subsample
            leadtimes_to_keep: list of indexes to be kept in the given dimension.
        """
        super().__init__()
        self.lead_times_indices = torch.tensor(idx_to_keep)
        self.dim_name = dim_name
        self.idx_to_keep = idx_to_keep

    def forward(self, named_tensor: NamedTensor) -> NamedTensor:
        return named_tensor.index_select_dim(self.dim_name, self.idx_to_keep)


class MeanDimensionSubsampler(nn.Module):
    """Subsamples a NamedTensor dimension by averaging the specified indexes.

    Exemple:
    ```
    import torch
    from mfai.pytorch.namedtensor import NamedTensor


    data = NamedTensor(
        tensor=torch.rand(10, 512, 512, 3),
        names=["timesteps", "lat", "lon", "features"],
        feature_names=["u", "v", "t2m"],
    )

    sub_sampler = DimensionSubSampler(
        dim_name="timesteps",
        idx_to_be_meaned=[
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [6, 7, 8],
        ],
    )

    transformed_data = sub_sampler(data)

    print("data:")
    print(data)
    print("transformed_data:")
    print(transformed_data)
    ```
    =>
    ```
    data:
    --- NamedTensor ---
    Names: ['timesteps', 'lat', 'lon', 'features']
    Tensor Shape: torch.Size([10, 512, 512, 3]))
    transformed_data:
    --- NamedTensor ---
    Names: ['timesteps', 'lat', 'lon', 'features']
    Tensor Shape: torch.Size([4, 512, 512, 3]))
    ```
    """

    def __init__(
        self, dim_name: str, idx_to_be_meaned: Sequence[Sequence[int]]
    ) -> None:
        """
        Args:
            dim_name: Name of the dimension to subsample.
            leadtimes_to_keep: List of list of indexes to meaned in the given dimension.
        """
        super().__init__()
        self.dim_name = dim_name
        self.idx_to_be_meaned = idx_to_be_meaned

    def forward(self, named_tensor: NamedTensor) -> NamedTensor:
        # The named tensor's dimension names without the target dimension
        meaned_tensor_dimension_names = (
            named_tensor.names[: named_tensor.dim_index(self.dim_name)]
            + named_tensor.names[named_tensor.dim_index(self.dim_name) + 1 :]
        )

        # List of meaned tensors
        meaned_tensors = [
            NamedTensor(
                tensor=torch.mean(
                    named_tensor.index_select_tensor_dim(self.dim_name, idxs),
                    dim=named_tensor.dim_index(self.dim_name),
                ),
                names=meaned_tensor_dimension_names,
                feature_names=named_tensor.feature_names,
            )
            for idxs in self.idx_to_be_meaned
        ]

        # Stack allong the target dimension the list of meaned tensors
        vision_out = NamedTensor.stack(
            meaned_tensors,
            self.dim_name,
            named_tensor.dim_index(self.dim_name),
        )
        return vision_out
