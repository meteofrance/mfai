"""
This module defines the DGMRLightningModule class, which is a PyTorch Lightning module
for training a Deep Generative Model of Radar (DGMR). The model is designed for
forecasting future radar images using a Generative Adversarial Network (GAN) architecture.

The DGMRLightningModule includes:
- Initialization of model parameters and components, including the generator and discriminator.
- Forward pass method to generate predictions from input radar data.
- Discriminator and generator training steps, including loss calculations.
- Configuration of optimizers for training the generator and discriminator.

The implementation is inspired by the Skillful Nowcasting GAN from OpenClimateFix and
is modified for multiple satellite channels.
"""

from typing import Any, Literal

import torch
from lightning import LightningModule
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mfai.pytorch.losses.gan_dgmr import (
    GridCellLoss,
    loss_hinge_disc,
    loss_hinge_gen,
)
from mfai.pytorch.models.gan_dgmr import (
    ContextConditioningStack,
    Discriminator,
    Generator,
    LatentConditioningStack,
    Sampler,
)
from mfai.pytorch.namedtensor import NamedTensor


class DGMRLightningModule(LightningModule):
    """Pytorch Lightning Module to train the GAN."""

    def __init__(
        self,
        forecast_steps: int = 18,
        input_channels: int = 1,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        conv_type: Literal["standard", "coord", "3d"] = "standard",
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 768,
        context_channels: int = 384,
        generation_steps: int = 6,
        precip_weight_cap: float = 24.0,
        use_attention: bool = True,
        temporal_num_layers: int = 3,
        spatial_num_layers: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        From OpenClimateFix, Skillfull Nowcasting:
        https://github.com/openclimatefix/skillful_nowcasting/blob/main/dgmr/dgmr.py

        Initialize the Deep Generative Model of Radar model.

        This is a recreation of DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels.

        Args:
            forecast_steps: Number of steps to predict in the future.
            input_channels: Number of input channels per image.
            gen_lr: Learning rate for the generator.
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial
            discriminator.
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options.
            beta1: Beta1 for Adam optimizer.
            beta2: Beta2 for Adam optimizer.
            grid_lambda: Lambda for the grid regularization loss.
            context_channels: Number of context channels (int)
            generation_steps: Number of generation steps to use in forward pass, in paper is 6
            and the best is chosen for the loss this results in huge amounts of GPU memory though,
            so less might work better for training.
            latent_channels: Number of channels that the latent space should be reshaped to, input
            dimension into ConvGRU, also affects the number of channels for other linked
            inputs/outputs.
            precip_weight_cap: Custom ceiling for the weight function to compute the grid cell loss.
            **kwargs: Allow initialization of the parameters above through key pairs
        """
        super().__init__(**kwargs)

        self.forecast_steps = forecast_steps
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.grid_lambda = grid_lambda
        self.generation_steps = generation_steps

        # Definition of Loss
        self.grid_regularizer = GridCellLoss(precip_weight_cap=precip_weight_cap)

        # Definition of GAN's modules
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            input_channels=8 * input_channels,
            output_channels=latent_channels,
            use_attention=use_attention,
        )
        self.sampler = Sampler(
            forecast_steps=self.forecast_steps,
            latent_channels=latent_channels,
            context_channels=context_channels,
        )
        self.generator = Generator(
            self.conditioning_stack, self.latent_stack, self.sampler
        )
        self.discriminator = Discriminator(
            input_channels=input_channels,
            temporal_num_layers=temporal_num_layers,
            spatial_num_layers=spatial_num_layers,
        )

        self.save_hyperparameters()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, x: NamedTensor) -> NamedTensor:
        """Apply the generator to the tensor."""
        x_copy = x.clone()  # to avoid modifying the original tensor
        x_copy.rearrange_(
            "batch time height width features -> batch time features height width"
        )
        x_rain = x_copy["rain"].float()

        y_hat_rain = self.generator(x_rain)  # Apply model

        # Create future radar mask by repeating last input radar mask:
        mask = torch.cat(
            [x_copy["mask"][:, -1:] for _ in range(y_hat_rain.shape[1])], dim=1
        )
        y_hat = NamedTensor.new_like(torch.cat([y_hat_rain, mask], dim=2), x_copy)
        y_hat.rearrange_(
            "batch time features height width -> batch time height width features"
        )
        return y_hat

    def discriminator_step(
        self,
        predictions: Tensor,
        images: Tensor,
        real_sequence: Tensor,
    ) -> Tensor:
        # Cat along time dimension [B, T, C, H, W]
        generated_sequence = torch.cat([images, predictions], dim=1)

        # Cat along batch for the real+generated
        concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

        concatenated_outputs = self.discriminator(concatenated_inputs)

        score_real, score_generated = torch.split(
            concatenated_outputs,
            [real_sequence.shape[0], generated_sequence.shape[0]],
            dim=0,
        )
        score_real_spatial, score_real_temporal = torch.split(score_real, 1, dim=1)
        score_generated_spatial, score_generated_temporal = torch.split(
            score_generated, 1, dim=1
        )
        discriminator_loss = loss_hinge_disc(
            score_generated_spatial, score_real_spatial
        ) + loss_hinge_disc(score_generated_temporal, score_real_temporal)

        return discriminator_loss

    def generator_step(
        self,
        predictions: list[Tensor],
        images: Tensor,
        future_images: Tensor,
        real_sequence: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gen_mean = torch.stack(predictions, dim=0).mean(dim=0)
        grid_cell_reg = self.grid_regularizer(gen_mean, future_images)

        # Concat along time dimension
        generated_sequences = [torch.cat([images, x], dim=1) for x in predictions]
        # Cat along batch for the real+generated, for each example in the range
        # For each of the 6 examples
        generated_scores = []
        for g_seq in generated_sequences:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            # Split along the concatenated dimension, as discrimnator concatenates along dim=1
            _, score_generated = torch.split(
                concatenated_outputs, [real_sequence.shape[0], g_seq.shape[0]], dim=0
            )
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        return generator_loss, grid_cell_reg

    def training_step(self, batch: tuple[NamedTensor, NamedTensor]) -> tuple[float]:
        """Performs the training step for the batch."""
        images, future_images = batch
        images.rearrange_(
            "batch time height width features -> batch time features height width"
        )
        future_images.rearrange_(
            "batch time height width features -> batch time features height width"
        )
        images = images["rain"].float()
        future_images = future_images["rain"].float()

        real_sequence = torch.cat([images, future_images], dim=1)

        g_opt, d_opt = self.optimizers()

        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = checkpoint(
                self.generator.forward, images, use_reentrant=False
            )  # Use gradient checkpointing to reduce RAM usage during backward pass
            discriminator_loss = self.discriminator_step(
                predictions, images, real_sequence
            )
            # Backward
            d_opt.zero_grad()
            self.manual_backward(discriminator_loss)
            d_opt.step()

        predictions = [
            checkpoint(self.generator.forward, images, use_reentrant=False)
            for _ in range(self.generation_steps)
        ]  # Use gradient checkpointing to reduce RAM usage during backward pass
        generator_loss, grid_cell_reg = self.generator_step(
            predictions, images, future_images, real_sequence
        )
        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()

        return discriminator_loss, generator_loss, grid_cell_reg

    def configure_optimizers(self) -> tuple[list[torch.optim.Adam], list]:
        """Return the Adam optimizers."""
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.gen_lr, betas=(self.beta1, self.beta2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=(self.beta1, self.beta2),
        )
        return [opt_g, opt_d], []
