"""Tests for the standalone gpt2 weights download script.

We only exercise the pure, tensorflow-free parts of the script: the
:func:`assign` helper and the TF-shaped-weights to pytorch mapping done by
:func:`load_gpt2_from_dict`. Downloading real checkpoints and converting
tensorflow files is covered by the download script's integration usage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from gpt2_weights_download import assign, load_gpt2_from_dict

from mfai.pytorch.models.llms.gpt2 import GPT2, GPT2Settings


def _small_settings() -> GPT2Settings:
    """Return a small GPT2 configuration fast to instantiate in tests."""
    return GPT2Settings(
        model_size="custom",
        emb_dim=16,
        n_layers=1,
        n_heads=1,
        context_length=16,
        attn_tf_compat=True,
    )


def _make_params(
    *,
    emb_dim: int = 16,
    n_layers: int = 1,
    context_length: int = 16,
    vocab_size: int = 32,
) -> dict[str, Any]:
    """Build a synthetic TF-shaped params dict matching the small GPT2 cfg."""
    # Real tensorflow checkpoints load their weights as float32, matching the
    # model's parameters, so the synthetic weights are cast accordingly.
    rng = np.random.default_rng(0)

    def rand(*shape: int) -> np.ndarray:
        return rng.normal(size=shape).astype(np.float32)

    blocks: list[dict[str, Any]] = []
    for _ in range(n_layers):
        blocks.append(
            {
                "attn": {
                    "c_attn": {
                        "w": rand(emb_dim, 3 * emb_dim),
                        "b": rand(3 * emb_dim),
                    },
                    "c_proj": {"w": rand(emb_dim, emb_dim), "b": rand(emb_dim)},
                },
                "mlp": {
                    "c_fc": {"w": rand(emb_dim, 4 * emb_dim), "b": rand(4 * emb_dim)},
                    "c_proj": {"w": rand(4 * emb_dim, emb_dim), "b": rand(emb_dim)},
                },
                "ln_1": {"g": rand(emb_dim), "b": rand(emb_dim)},
                "ln_2": {"g": rand(emb_dim), "b": rand(emb_dim)},
            }
        )
    params: dict[str, Any] = {
        "wpe": rand(context_length, emb_dim),
        "wte": rand(vocab_size, emb_dim),
        "blocks": blocks,
        "g": rand(emb_dim),
        "b": rand(emb_dim),
    }
    return params


def test_assign_creates_parameter_with_matching_values() -> None:
    """Assign returns a Parameter containing the supplied values."""
    left = torch.zeros(2, 3)
    right = np.arange(6).reshape(2, 3).astype(np.float32)
    param = assign(left, right)
    assert isinstance(param, torch.nn.Parameter)
    assert torch.equal(param, torch.tensor(right))


def test_assign_raises_on_shape_mismatch() -> None:
    """Assign raises a ValueError when the shapes do not match."""
    left = torch.zeros(2, 3)
    right = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        assign(left, right)


def test_load_gpt2_from_dict_with_matching_shapes() -> None:
    """Weights are copied into the model when all shapes match exactly."""
    settings = _small_settings()
    params = _make_params(vocab_size=32)
    gpt2 = load_gpt2_from_dict(GPT2(settings, vocab_size=32), params)

    assert torch.equal(gpt2.tok_emb.weight.detach(), torch.tensor(params["wte"]))
    assert torch.equal(gpt2.pos_emb.weight.detach(), torch.tensor(params["wpe"]))
    assert torch.equal(gpt2.final_norm.scale.detach(), torch.tensor(params["g"]))

    # The model still runs a forward pass with the loaded weights.
    token_ids = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        logits = gpt2(token_ids)
    assert logits.shape == (1, token_ids.shape[1], 32)


def test_load_gpt2_from_dict_with_extra_tokens_and_context() -> None:
    """Extra tokens and longer context are partially loaded from the weights."""
    model = GPT2(
        GPT2Settings(
            model_size="custom",
            emb_dim=16,
            n_layers=1,
            n_heads=1,
            context_length=24,
            attn_tf_compat=True,
        ),
        vocab_size=64,
    )
    params = _make_params(context_length=16, vocab_size=32)
    gpt2 = load_gpt2_from_dict(model, params)

    # The official tokens/positions are copied, the extra rows are kept.
    assert gpt2.tok_emb.weight.shape == (64, 16)
    assert torch.equal(gpt2.tok_emb.weight.detach()[:32], torch.tensor(params["wte"]))
    assert gpt2.pos_emb.weight.shape == (24, 16)
    assert torch.equal(gpt2.pos_emb.weight.detach()[:16], torch.tensor(params["wpe"]))
