import pytest
import torch
from torch import Tensor, nn
from mfai.torch.models.llms import Llama2Settings, Llama2, GPT2, GPT2Settings
from mfai.tokenizers import LlamaTokenizer, GPT2Tokenizer

torch.manual_seed(999)


def generate_text_simple(
    model: nn.Module, idx: Tensor, max_new_tokens: int, context_size: int
) -> Tensor:
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
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
    "model_target_tokenizer",
    [
        (
            GPT2(GPT2Settings()),
            "Hello, I am CH commemor talent Container GPL dab OpenGL unsuccessful formallymits",
            GPT2Tokenizer(),
        ),
        (
            Llama2(Llama2Settings()),
            "Hello, I am Federation ontobatonce Gr usefulfecategories representations alten",
            LlamaTokenizer(),
        ),
    ],
)
def test_llms(model_target_tokenizer):
    model, target, tokenizer = model_target_tokenizer
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
