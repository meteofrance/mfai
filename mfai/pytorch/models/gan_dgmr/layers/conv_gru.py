"""Implementation of Conv GRU and cell module."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm


class ConvGRUCell(torch.nn.Module):
    """A ConvGRU implementation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        sn_eps: float = 0.0001,
    ) -> None:
        """Conv GRU class.

        Args:
          input_channels: number of input channels (int)
          output_channels: number of output channels (int)
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.

        """
        super().__init__()
        self._kernel_size: int = kernel_size
        self._sn_eps: float = sn_eps
        self.read_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.update_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.output_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )

    def forward(self, x: Tensor, prev_state: Tensor) -> Tensor:
        """
        Conv GRU forward, returning the current + new state.

        Args:
            x: input tensor of shape (B, input_channels, H, W).
            prev_state: Previous state

        Returns:
            output of the convGRU, tensor of shape (B, output_channels, H, W).

        """
        # Concatenate the inputs and previous state along the channel axis.
        xh = torch.cat([x, prev_state], dim=1)

        # Read gate of the GRU.
        read_gate = F.sigmoid(self.read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate = F.sigmoid(self.update_gate_conv(xh))

        # Gate the inputs.
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)

        # Gate the cell and state / outputs.
        c = F.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        return out


class ConvGRU(torch.nn.Module):
    """ConvGRU Cell wrapper to replace tf.static_rnn in TF implementation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        sn_eps: float = 0.0001,
    ) -> None:
        """Initialize the convolution layer."""
        super().__init__()
        self.output_channels = output_channels
        self.cell = ConvGRUCell(input_channels, output_channels, kernel_size, sn_eps)

    def forward(self, x: Tensor, hidden_state: Tensor) -> Tensor:
        """Apply the forward function on each cell prior to returning it as a stack.

        Args:
            x: tensor from the conditionning stack or from the previous ConvGRU. (timesteps, B, input_channels, H, W)
            hidden_state: tensor from the latent conditionning stack. (B, latent_channels, H, W)
        """
        timesteps, batch_size, _, height, width = x.shape
        outputs = torch.empty(
            timesteps,
            batch_size,
            self.output_channels,
            height,
            width,
            device=x.device,
            dtype=x.dtype,
            layout=x.layout,
        )
        for i, step in enumerate(x):  # Iterate over the timestep dimension
            # Compute current timestep
            hidden_state = self.cell(step, hidden_state)
            outputs[i] = hidden_state
        return outputs
