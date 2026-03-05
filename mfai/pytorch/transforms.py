import random

import torch
import torchvision.transforms.functional as TF
from torch import Tensor


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

    def _get_one_crop(self, y: Tensor) -> tuple[int, int, float]:
        """Returns the top-left corner of a random crop and the percentage
        of positive pixels in the crop.

        Args:
            y: The input image to crop, of shape (1, H, W)

        Returns:
            int: The left coordinate of the crop.
            int: The top coordinate of the crop.
            float: The percentage of positive pixels in the crop.
        """

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

    def _get_crop(self, y: Tensor) -> tuple[int, int]:
        """Returns the top-left corner of a random crop that satisfies
        the condition of having at least min_positive_percentage of positive
        pixels. If such a crop is not found after tries attempts, it returns
        the crop with the most positive pixels.

        Args:
            y: The input image to crop, of shape (1, H, W)

        Returns:
            int: The left coordinate of the crop.
            int: The top coordinate of the crop.
        """

        best_top, best_left, best_perc = self._get_one_crop(y)
        for _ in range(self.tries - 1):
            top, left, perc = self._get_one_crop(y)
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
        """Crops the input image and mask.

        Args:
            sample: A tuple containing the input image and mask, of shape
                (C, H, W) and (1, H, W) respectively.

        Returns:
            Tensor: The cropped input image, of shape (C, crop_size[0], crop_size[1]).
            Tensor: The cropped mask, of shape (1, crop_size[0], crop_size[1]).
            int: The left coordinate of the crop.
            int: The top coordinate of the crop.
        """

        x, y = sample
        left, top = self._get_crop(y)
        cropped_x = TF.crop(x, top, left, self.crop_size[0], self.crop_size[1])
        cropped_y = TF.crop(y, top, left, self.crop_size[0], self.crop_size[1])
        return cropped_x, cropped_y, left, top
