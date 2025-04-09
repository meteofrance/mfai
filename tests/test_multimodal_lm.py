from typing import Literal, Tuple, Union

import pytest
import torch
from torch import Tensor, nn

from mfai.tokenizers import GPT2Tokenizer, LlamaTokenizer
from mfai.torch.models.llms.multimodal import MultiModalLM, MultiModalLMSettings
from mfai.torch.namedtensor import NamedTensor


def generate_text_simple(
    model: nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    context_size: int,
    vision_input: Union[None, NamedTensor] = None,
) -> Tensor:
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            if vision_input:
                logits = model(idx_cond, vision_input)
            else:
                logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


@pytest.mark.parametrize(
    "llm_backend, tokenizer, expected_text",
    [
        (
            "llama2",
            LlamaTokenizer(),
            (
                "Sustine et abstineAlignment Геrace sqlwesten Loggerлага Bushに同",
                "Sustine et abstine makulsion flag重глеägerhand Av Lincoln mul",
            ),
        ),
        (
            "gpt2",
            GPT2Tokenizer(),
            (
                "Sustine et abstinegreg LXamm Local addition Immun GlassrikeFal Resurrection",
                "Sustine et abstineohoorphLE updates� Oaks Coconut VC Privacy backward",
            ),
        ),
        (
            "gpt2",
            LlamaTokenizer(),
            (
                "Sustine et abstine współ terrestführtrange지edتズ ownershipantal",
                "Sustine et abstine detected *rit україн dernièreistoryikalcorüssknow",
            ),
        ),
        (
            "gpt2",
            GPT2Tokenizer(),
            (
                "Sustine et abstinegreg LXamm Local addition Immun GlassrikeFal Resurrection",
                "Sustine et abstineohoorphLE updates� Oaks Coconut VC Privacy backward",
            ),
        ),
    ],
)
def test_multimodal_llm(
    llm_backend: Literal["llama2", "gpt2"],
    tokenizer: Union[GPT2Tokenizer, LlamaTokenizer],
    expected_text: Tuple[str, str],
):
    torch.manual_seed(999)
    for force_vision in (False, True):
        model = MultiModalLM(
            settings=MultiModalLMSettings(
                vision_input_shape=(3, 3, 2, 1),
                backend=llm_backend,
                n_heads=1,
                n_layers=1,
                emb_dim=32,
                hidden_dim=32,
                context_length=32,
                inject_vision_each_stage=force_vision,
            ),
            vocab_size=tokenizer.vocab_size,
        )
        vision_input = NamedTensor(
            torch.randn(1, 3, 3, 2, 1),
            names=("batch", "lat", "lon", "timestep", "features"),
            feature_names=("u",),
        )
        encoded = tokenizer.encode("Sustine et abstine")
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)

        out = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=10,
            context_size=model.context_length,
            vision_input=vision_input,
        )
        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        print(llm_backend, tokenizer.name(), decoded_text)
        assert decoded_text == expected_text[0 if not force_vision else 1]
