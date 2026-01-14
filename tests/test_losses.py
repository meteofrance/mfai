"""Test DGMR GAN loss functions"""

import torch

from mfai.pytorch.losses.gan_dgmr import (
    GridCellLoss,
    loss_hinge_disc,
    loss_hinge_gen,
)
from mfai.pytorch.losses.toolbelt import (
    DiceLoss,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss,
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


def test_dice_loss() -> None:
    dice_loss_binary = DiceLoss("binary")
    y_pred = torch.rand(2, 1, 16, 16)
    y_true = torch.randint(0, 1, (2, 1, 16, 16)).float()
    loss = dice_loss_binary(y_true, y_pred)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    dice_loss_multiclass = DiceLoss("multilabel")

    y_pred = torch.rand(2, 4, 16, 16)
    y_true = torch.randint(0, 1, (2, 4, 16, 16)).float()
    loss = dice_loss_multiclass(y_true, y_pred)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_soft_cross_entropy() -> None:
    # target contains class indices between 0 and 3 included
    y_true = torch.randint(0, 4, (2, 16, 16))

    # prediction contains class probabilities
    y_pred = torch.nn.functional.softmax(torch.rand(2, 4, 16, 16), dim=1)

    sce = SoftCrossEntropyLoss()
    loss = sce(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_soft_bce() -> None:
    y_true = torch.randn(2, 4, 16, 16)
    y_pred = torch.randn(2, 4, 16, 16)

    sbce = SoftBCEWithLogitsLoss()
    loss = sbce(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
