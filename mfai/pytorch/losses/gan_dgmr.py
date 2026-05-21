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
        precip_weight_cap: float,
    ) -> None:
        """
        Initialize GridCellLoss.

        Args:
            precip_weight_cap: Custom ceiling value for the weight function.

        """
        super().__init__()
        self.precip_weight_cap = precip_weight_cap

    def forward(self, generated_images: Tensor, targets: Tensor) -> Tensor:
        r"""
        Forward function.

        Calculates the grid cell regularizer value, assumes generated images are the mean predictions from 6 calls
        to the generator (Monte Carlo estimation of the expectations for the latent variable).

        .. math::

            L_R(\Theta) = \frac{1}{HWN} \\| (\mathbb{E}_Z [G_|theta(Z; X_{1:M})] - X_{M+1:M+T}) \circ w(X_{M+1:M+T}) \\|_1

        where H, W and T represent height, width and leadtimes.

        Note:
            Instead of apply the formula of the weights describe in the original article (:math:`w(y) = max(y+1, precip\_weight\_cap)`),
            we implement a formula closer to the pseudocode released by Google Deepmind. So our formula is : :math:`w(y) = clip(y, 1, precip\_weight\_cap)`,
            which mean that weights are between 1 and `precip_weight_cap`.

        Args:
            generated_images: generated images from the generator. Tensor of shape (N B T C H W), where N is the number of generated images.
            targets: Ground truth future frames. Tensor of shape (B T C H W).

        Returns:
            Grid Cell Regularizer term

        """
        assert len(generated_images.shape) == 6
        gen_mean: Tensor = generated_images.mean(dim=0)  # (B T C H W)
        weights = torch.clip(targets, 1, self.precip_weight_cap)  # (B T C H W)
        loss = ((gen_mean - targets) * weights).norm(p=1)
        return loss


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
