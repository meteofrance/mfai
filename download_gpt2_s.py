

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from mfai.pytorch.models.llms.gpt2 import GPT2, GPT2Settings
from mfai.tensorflow import download_and_load_gpt2


path = Path("downloads")
allowed_sizes = ("124M", "355M", "774M", "1558M")
for size in tqdm(allowed_sizes, desc="downloading"):
    settings = GPT2Settings(
        model_size=size,
        attn_tf_compat=True
    )
    gpt2 = GPT2(settings)
    gpt2.download_weights_from_tf_ckpt(path)
    torch.save(gpt2.state_dict(), path / f"gpt2_{size}.pkl")