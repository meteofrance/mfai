"""Module for various loss functions used with DGMR GAN."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GridCellLoss(nn.Module):
    """Grid Cell Regularizer loss from Skillful Nowcasting, see equation (3) of
    the original paper (https://arxiv.org/pdf/2104.00954.pdf).
    """

    def __init__(
        self,
        precip_weight_cap: float = 24.0,
    ) -> None:
        """
        Initialize GridCellLoss.

        Args:
            weight_fn: A function to compute weights for the loss.
            precip_weight_cap: Custom ceiling value for the weight function.
        """
        super().__init__()
        self.precip_weight_cap = precip_weight_cap

    def forward(self, generated_images: Tensor, targets: Tensor) -> Tensor:
        """
        Forward function.

        Calculates the grid cell regularizer value, assumes generated images are the mean
        predictions from 6 calls to the generator (Monte Carlo estimation of the
        expectations for the latent variable).

        .. math:: L_R(\Theta) = \frac{1}{HWN} \| (\mathbb{E}_Z [G_|theta(Z; X_{1:M})] - X_{M+1:M+N}) \circ w(X_{M+1:M+N}) \|_1

        where H, W and N represent height, width and leadtimes.

        Args:
            generated_images: Mean generated images from the generator
            targets: Ground truth future frames

        Returns:
            Grid Cell Regularizer term
        """
        difference = generated_images - targets
        if self.weight_fn is not None:
            weights = self.weight_fn(targets, self.precip_weight_cap)
            difference *= weights
        difference = difference.norm(p=1)
        return difference / (targets.size(1) * targets.size(3) * targets.size(4))

    def weight_fn(self, y: Tensor, precip_weight_cap: float = 24.0) -> Tensor:
        """
        Weight function for the grid cell loss.

        w(y) = max(y + 1, precip_weight_cap)

        Args:
            y: Tensor of rainfall intensities.
            precip_weight_cap: Custom ceiling for the weight function.

        Returns:
            Weights for each grid cell.
        """
        return torch.max(y + 1, torch.tensor(precip_weight_cap, device=y.device))


def loss_hinge_disc(score_generated: Tensor, score_real: Tensor) -> Tensor:
    """Discriminator Hinge loss."""
    relu_score_real = F.relu(1.0 - score_real)
    loss = torch.mean(relu_score_real)
    relu_score_generated = F.relu(1.0 + score_generated)
    loss += torch.mean(relu_score_generated)
    return loss


def loss_hinge_gen(score_generated: Tensor) -> Tensor:
    """Generator Hinge loss."""
    return -torch.mean(score_generated)
