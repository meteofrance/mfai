from pathlib import Path
from typing import Literal, Tuple, Union

import pytest
import torch
from torch import Tensor, nn

from mfai.tokenizers import GPT2Tokenizer, LlamaTokenizer
from mfai.torch.models.clip import Clip, ClipSettings
from mfai.torch.models.llms import GPT2, GPT2Settings
from mfai.torch.models.llms.multimodal import MultiModalLM, MultiModalLMSettings
from mfai.torch.models.resnet import ResNet50, ResNet50Settings
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
                "Sustine et abstine Patron nationalist grease Carly Detectiveuceditta Mysteryolationitivity",
                "Sustine et abstine grinned Supporters strife dissemination crewsrush error paternalirementsuania",
            ),
        ),
        (
            "gpt2",
            LlamaTokenizer(),
            (
                "Sustine et abstine współ terrestführt substantial arrow atoms introduction mil стар sze",
                "Sustine et abstine logging extremdan={\glyское elabor commissionategymapping",
            ),
        ),
    ],
)
def test_multimodal_llm(
    llm_backend: Literal["llama2", "gpt2"],
    tokenizer: Union[GPT2Tokenizer, LlamaTokenizer],
    expected_text: Tuple[str, str],
) -> None:
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
            names=["batch", "lat", "lon", "timestep", "features"],
            feature_names=[
                "u",
            ],
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


def test_multimodal_with_pretrained_clip() -> None:
    torch.manual_seed(666)
    embed_dim = 32
    vision_input_shape = (128, 128, 2, 1)
    num_channels: int = vision_input_shape[2] * vision_input_shape[3]
    path_checkpoint = Path("checkpoint.tar")

    # Setup the CLIP model
    resnet_clip = ResNet50(
        num_channels=num_channels,
        num_classes=embed_dim,
        # Optional : encoder pretrained with imagenet
        settings=ResNet50Settings(encoder_weights=True),
    )
    llm_clip = GPT2(
        settings=GPT2Settings(
            n_heads=2,
            n_layers=4,
            context_length=64,
        )
    )
    clip = Clip(
        settings=ClipSettings(
            emb_dim=embed_dim,
            image_encoder=resnet_clip,
            text_encoder=llm_clip,
            init_temperature=666,
        )
    )

    # Save the weights and parameters of the image encoder ResNet50
    clip.save_vision_encoder(path_checkpoint)

    tokenizer = GPT2Tokenizer()
    model = MultiModalLM(
        settings=MultiModalLMSettings(
            vision_input_shape=vision_input_shape,
            backend="gpt2",
            n_heads=1,
            n_layers=1,
            emb_dim=embed_dim,
            hidden_dim=32,
            context_length=32,
            inject_vision_each_stage=False,
            vision_encoder="resnet50",
            resnet_checkpoint=path_checkpoint,
        ),
        vocab_size=tokenizer.vocab_size,
    )
    vision_input = NamedTensor(
        torch.randn(1, 128, 128, 2, 1),
        names=["batch", "lat", "lon", "timestep", "features"],
        feature_names=[
            "u",
        ],
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
    tokenizer.decode(out.squeeze(0).tolist())
