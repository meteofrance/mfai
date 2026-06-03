"""Discriminators."""

from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.transforms import RandomCrop

from .blocks import DBlock


class Discriminator(torch.nn.Module):
    """Discriminators class."""

    def __init__(
        self,
        input_channels: int = 12,
        num_spatial_frames: int = 8,
        conv_type: Literal["standard", "coord", "3d"] = "standard",
        temporal_num_layers: int = 3,
        spatial_num_layers: int = 4,
    ) -> None:
        """
        Initialize the discriminator.

        Args:
            input_channels: Number of input channels.
            num_spatial_frames: Number of spatial frames.
            conv_type: the specified convolution type.
            temporal_num_layers: Number of intermediate DBlock layers to use in the temporal discriminator.
            spatial_num_layers: Number of intermediate DBlock layers to use in the spatial discriminator.

        """
        super().__init__()

        self.spatial_discriminator = SpatialDiscriminator(
            input_channels=input_channels,
            num_timesteps=num_spatial_frames,
            conv_type=conv_type,
            num_layers=spatial_num_layers,
        )
        self.temporal_discriminator = TemporalDiscriminator(
            input_channels=input_channels,
            conv_type=conv_type,
            num_layers=temporal_num_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Mixes the spatial loss and temporal loss of the tensor prior to returning it.

        Args:
            x: a tensor with a complete observation (b, t, c, h, w).
        """
        spatial_loss = self.spatial_discriminator(x)
        temporal_loss = self.temporal_discriminator(x)

        return torch.cat([spatial_loss, temporal_loss], dim=1)


class TemporalDiscriminator(torch.nn.Module):
    """Temporal Discriminator class."""

    def __init__(
        self,
        input_channels: int = 12,
        num_layers: int = 3,
        conv_type: Literal["standard", "coord", "3d"] = "standard",
    ) -> None:
        """
        Temporal Discriminator from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf.

        Args:
            input_channels: Number of channels per timestep
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options

        """
        super().__init__()

        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 48
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=internal_chn * input_channels,
            conv_type="3d",
            first_relu=False,
        )
        self.d2 = DBlock(
            input_channels=internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            conv_type="3d",
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )

        self.d_last = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tensor of predictions or observations of shape (b, t, c, h, w).
        """
        batch_size, timesteps, _, height, width = x.size()

        # Choose the offset of a random crop of size 128x128 out of 256x256 and
        # pick full sequence samples.
        random_crop = RandomCrop(size=(height // 2, width // 2))
        x = random_crop(x)

        # Process each of the timesteps inputs independently.
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # Space-to-depth stacking from (1, 128, 128) to (4, 64, 64).
        x = self.space2depth(x)

        # Stack back to sequences of length timesteps.
        x = rearrange(x, "(b t) c h w -> b c t h w", b=batch_size, t=timesteps)

        # Two residual 3D Blocks to halve the resolution of the image, double
        # the number of channels, and reduce the number of time steps.
        x = self.d1(x)
        x = self.d2(x)

        # Process each of the t images independently.
        y = rearrange(x, "b c t h w -> (b t) c h w")

        # Three residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        for dblocks in self.intermediate_dblocks:
            y = dblocks(y)

        # One more D Block without downsampling or increase number of channels
        y = self.d_last(y)

        # Sum-pool the representations and feed to spectrally normalized lin. layer.
        y = torch.sum(F.relu(y), dim=[2, 3])
        y = self.bn(y)
        y = self.fc(y)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        out = rearrange(y, "(b t) c -> b t c", b=batch_size)
        out = torch.sum(out, keepdim=True, dim=1)
        return out


class SpatialDiscriminator(torch.nn.Module):
    """Spatial Discriminator class."""

    def __init__(
        self,
        input_channels: int = 12,
        num_timesteps: int = 8,
        num_layers: int = 4,
        conv_type: Literal["standard", "coord", "3d"] = "standard",
    ) -> None:
        """
        Spatial discriminator from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf.

        Args:
            input_channels: Number of input channels per timestep
            num_timesteps: Number of timesteps to use, in the paper 8/18 timesteps were chosen
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options

        """
        super().__init__()

        # Randomly, uniformly, select 8 timesteps to do this on from the input
        self.num_timesteps = num_timesteps
        # First step is mean pooling 2x2 to reduce input by half
        self.mean_pool = torch.nn.AvgPool2d(2)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 24
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=2 * internal_chn * input_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        # Spectrally normalized linear layer for binary classification
        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(2 * internal_chn * input_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tensor of predictions or observations of shape (b, t, c, h, w).
        """
        batch_size, timesteps, _, _, _ = x.size()
        # x should be the chosen 8 or so
        random_indices = torch.randint(
            low=0, high=timesteps, size=(self.num_timesteps,), device=x.device
        )
        x = x.index_select(1, random_indices)

        # Process each of the n inputs independently.
        frames = rearrange(x, "b t c h w -> (b t) c h w")

        # Space-to-depth stacking from (1, 128, 128) to (4, 64, 64).
        y = self.mean_pool(frames)
        y = self.space2depth(y)

        # Five residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        y = self.d1(y)  # (48, 32, 32)
        for d in self.intermediate_dblocks:  # Intermediate DBlocks
            y = d(y)
        # One more D Block without downsampling or increase in number of channels.
        y = self.d6(y)  # (768, 2, 2)

        # Sum-pool the representations and feed to spectrally normalized linear layer.
        y = torch.sum(F.relu(y), dim=[2, 3])
        y = self.bn(y)
        y = self.fc(y)  # (1,)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        out = rearrange(y, "(b t) c -> b t c", b=batch_size, t=self.num_timesteps, c=1)
        out = torch.sum(out, keepdim=True, dim=1)
        return out
