import time
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
from test_multimodal_lm import generate_text_simple

from mfai.pytorch.models.llms.gpt2 import (
    GPT2,
    CrossAttentionGPT2,
    CrossAttentionGPT2Settings,
    GPT2Settings,
)
from mfai.pytorch.models.llms.llama2 import Llama2, Llama2Settings
from mfai.pytorch.models.llms.llama3 import Llama3, Llama3Settings
from mfai.pytorch.models.llms.qwen3_5 import Qwen3_5, Qwen3_5Settings
from mfai.tokenizers import GPT2Tokenizer, LlamaTokenizer, Qwen3_5Tokenizer, Tokenizer


@pytest.mark.parametrize(
    "model_target_tokenizer",
    [
        (
            partial(GPT2, GPT2Settings()),
            "Hello, I am CH commemor scholarly fingerprint oppositethrenClean Bridgewatericidal scalp",
            GPT2Tokenizer(),
        ),
        (
            partial(Llama2, Llama2Settings()),
            "Hello, I am LCCN entertain fielGB surface деревняA proposeDid嘉",
            LlamaTokenizer(),
        ),
        (
            partial(Llama3, Llama3Settings()),
            "Hello, I am voicessource Cris fjär pltheimer spectral проис mentreinental",
            LlamaTokenizer(),
        ),
        (
            partial(Qwen3_5, Qwen3_5Settings()),
            "Hello, I am键千千万 хоче ని металлуasted项手感预告ত্ত",
            Qwen3_5Tokenizer(apply_chat_template=False),
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


@pytest.mark.parametrize(
    "model_tokenizer",
    [
        (GPT2(GPT2Settings(attn_tf_compat=True)), GPT2Tokenizer()),
        (Llama3(Llama3Settings()), LlamaTokenizer()),
    ],
)
def test_kv_cache(model_tokenizer: tuple[GPT2, GPT2Settings]) -> None:
    """
    We check that KV cache implementation is working and a speed-up text generation.
    """
    model, tokenizer = model_tokenizer
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    start = time.perf_counter()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=400,
        context_size=model.context_length,
        use_cache=False,
    )
    end = time.perf_counter()
    exec_time = end - start
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    start = time.perf_counter()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=400,
        context_size=model.context_length,
        use_cache=True,
    )
    end = time.perf_counter()
    exec_time_with_cache = end - start
    decoded_text_with_cache = tokenizer.decode(out.squeeze(0).tolist())

    assert decoded_text == decoded_text_with_cache
    assert exec_time_with_cache < exec_time


def test_cross_attention_gpt2() -> None:
    """
    Here we only test that the model is mathematically correct (matmul compat, shapes, attention, ...).
    """
    torch.manual_seed(999)
    settings = CrossAttentionGPT2Settings(
        context_length=32,
        n_heads=1,
        n_layers=4,
        emb_dim=32,
        x_att_ratio=2,
    )
    model = CrossAttentionGPT2(settings)
    token_ids = torch.rand(1, 16).long()

    generate_text_simple(
        model=model,
        idx=token_ids,
        max_new_tokens=10,
        context_size=model.context_length,
        vision_inputs=torch.randn(1, 8, settings.emb_dim),
    )


def test_download_gpt2_weights(tmp_path: Path) -> None:
    model = GPT2(GPT2Settings(attn_tf_compat=True))
    model.dowload_weights_from_tf_ckpt(tmp_path)

    # test with extra tokens
    model = GPT2(GPT2Settings(attn_tf_compat=True), vocab_size=50400)
    model.dowload_weights_from_tf_ckpt(tmp_path)

    # test with longer context len - default is 1024 for gpt2 small
    model = GPT2(GPT2Settings(attn_tf_compat=True, context_length=1032))
    model.dowload_weights_from_tf_ckpt(tmp_path)
