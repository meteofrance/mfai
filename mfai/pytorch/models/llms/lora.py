"""
Lora layer and utility code adapted from https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb
"""

import math

import torch


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(
            self.A, a=math.sqrt(5)
        )  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: The original chapter didn't include the scaling by self.rank
        # This scaling is not necessary, but it's more canonical and convenient
        # as this lets us compare runs across different ranks without retuning learning rates
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model: torch.nn.Module, rank: int, alpha: float) -> None:
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)


def setup_model_for_lora(model: torch.nn.Module, rank: int, alpha: float) -> None:
    """
    0. Print total number of trainable params before lora replacement
    1. Freeze all model parameters
    2. replace Linear Layers by LinearWithLora layers (not frozen)
    3. Print total number of trainable params after lora replacement
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters before LORA: {total_params:,}")
    
    for param in model.parameters():
            param.requires_grad = False

    replace_linear_with_lora(model, rank, alpha)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters after LORA: {total_params:,}")
