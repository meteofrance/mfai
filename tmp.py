from pathlib import Path

import torch
from mfai.pytorch.models.llms.gpt2 import GPT2, GPT2Settings

output_dir = Path("/scratch/shared/gpt2_weights")
size = "124M"

gpt2 = GPT2(GPT2Settings())
gpt2.load_state_dict(torch.load(output_dir / f"gpt2_{size}.pkl", weights_only=True))

breakpoint()