from functools import partial
from typing import Any

import pytest
import torch
from test_multimodal_lm import generate_text_simple

from mfai.tokenizers import GPT2Tokenizer, LlamaTokenizer, Tokenizer
from mfai.torch.models.llms import GPT2, GPT2Settings, Llama2, Llama2Settings, CrossAttentionGPT2, CrossAttGPT2Settings


@pytest.mark.parametrize(
    "model_target_tokenizer",
    [
        (
            partial(GPT2, GPT2Settings()),
            "Hello, I amisi invincible collided 1500 tenomenotinables thinks republic",
            GPT2Tokenizer(),
        ),
        (
            partial(Llama2, Llama2Settings()),
            "Hello, I am LCCN entertain fielGB surface деревняA proposeDid嘉",
            LlamaTokenizer(),
        ),
    ],
)
def test_llms(model_target_tokenizer: tuple[Any, str, Tokenizer]) -> None:
    torch.manual_seed(999)
    model, target, tokenizer = model_target_tokenizer
    model = model()
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=model.context_length,
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    assert decoded_text == target


def test_cross_attention_gpt2() -> None:
    """
    Here we only test that the model is mathematically correct (matmul compat, shapes, attention, ...)
    """
    torch.manual_seed(999)
    settings = CrossAttGPT2Settings(
        context_length=32,
        n_heads=1,
        n_layers=1,
        emb_dim=32,
    )
    model = CrossAttentionGPT2(settings)
    token_ids = torch.rand(1, 16).long()

    generate_text_simple(
        model=model,
        idx=token_ids,
        max_new_tokens=10,
        context_size=model.context_length,
        vision_input=torch.randn(1, 8, settings.emb_dim),
    )
