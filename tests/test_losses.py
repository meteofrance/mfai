"""Test DGMR GAN loss functions"""

import torch

from mfai.pytorch.losses.gan_dgmr import (
    GridCellLoss,
    loss_hinge_disc,
    loss_hinge_gen,
)


def test_grid_cell_loss() -> None:
    """Test the grid cell loss function."""
    generated_images = torch.rand(2, 3, 1, 10, 10)
    targets = torch.rand(2, 3, 1, 10, 10)
    loss_fn = GridCellLoss()
    loss = loss_fn(generated_images, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_loss_hinge_disc() -> None:
    """Test the discriminator hinge loss function."""
    score_generated = torch.tensor([[0.5], [0.3]], dtype=torch.float32)
    score_real = torch.tensor([[0.2], [0.1]], dtype=torch.float32)
    loss = loss_hinge_disc(score_generated, score_real)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss.item() == 2.25


def test_loss_hinge_gen() -> None:
    """Test the generator hinge loss function."""
    score_generated = torch.tensor([[1], [3]], dtype=torch.float32)
    loss = loss_hinge_gen(score_generated)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss.item() == -2
