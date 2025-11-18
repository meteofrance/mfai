import torch

from mfai.pytorch.transforms import RandomCropWithMinPositivePixels


def test_random_crop() -> None:
    t = RandomCropWithMinPositivePixels()
    y = torch.randint(0, 2, (1, 2048, 2048))
    x = torch.randn(8, 2048, 2048)
    cropped_x, cropped_y, _, _ = t((x, y))
    assert cropped_x.shape[1] == cropped_x.shape[2] == 512
    assert cropped_x.shape[0] == 8
    assert cropped_y.shape[1] == cropped_y.shape[2] == 512
