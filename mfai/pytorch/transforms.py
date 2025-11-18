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
