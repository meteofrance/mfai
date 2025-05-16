"""
Implementation of CLIP (Contrastive Langage-Image Pre-training) model. Based on the original https://arxiv.org/abs/2103.00020
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from dataclasses_json import dataclass_json

from mfai.torch.models.llms import GPT2, LayerNorm, Llama2
from mfai.torch.models.resnet import ResNet50
from mfai.torch.namedtensor import NamedTensor


@dataclass_json
@dataclass(slots=True)
class ClipSettings:
    # Image settings
    image_encoder: ResNet50

    # Text settings
    text_encoder: Union[GPT2, Llama2]

    emb_dim: int = 1024
    init_temperature: float = 1 / 0.07  # Value from CLIP paper


class Clip(nn.Module):
    """
    Implementation of CLIP (Contrastive Langage-Image Pre-training) model.
    - Based on the original article from OpenAI:
        https://arxiv.org/abs/2103.00020
    """

    def __init__(self, settings: ClipSettings):
        super().__init__()
        self.image_encoder = settings.image_encoder
        self.text_encoder = settings.text_encoder

        self.emb_dim = settings.emb_dim
        self.text_emb_dim = self.text_encoder.emb_dim

        self.text_norm = LayerNorm(self.text_emb_dim)
        self.text_projection = nn.Parameter(
            torch.empty(self.text_emb_dim, self.emb_dim)
        )
        nn.init.normal_(self.text_projection, std=self.text_encoder.emb_dim**-0.5)
        self.temperature = nn.Parameter(
            torch.ones([]) * torch.log(Tensor([settings.init_temperature]))
        )

    def encode_text(self, text_tokens: Tensor) -> Tensor:
        # Keep only the last context_length tokens:
        text_tokens = text_tokens[:, -self.text_encoder.context_length :]
        x = self.text_encoder.embed_tokens(
            text_tokens
        )  # [batch_size, seq_len, emb_dim]
        x = self.text_encoder.drop_emb(x)  # type: ignore[operator]
        x = self.text_encoder.trf_blocks(x)  # Apply transformer model
        x = self.text_norm(x)  # [batch_size, seq_len, emb_dim]

        # We consider that EOT token (in practice the highest number in each sequence) represents
        # the sentence, as it said in the original article (p.5):
        # "The [EOS] token are treated as the feature representation of the text which is
        # layer normalized and then linearly projected into the multi-modal embedding space."
        x = (
            x[torch.arange(len(text_tokens)), text_tokens.argmax(dim=-1)]
            @ self.text_projection
        )
        return x  # [batch_size, emb_dim]

    def forward(
        self, text_tokens: Tensor, image_input: NamedTensor
    ) -> Tuple[Tensor, Tensor]:
        image_features = self.image_encoder(image_input.tensor)
        text_features = self.encode_text(text_tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        image_logits = image_features @ text_features.T * torch.exp(self.temperature)
        text_logits = image_logits.T

        return text_logits, image_logits

    def save_vision_encoder(self, path: Path) -> None:
        """
        Save the weights and parameters of the image encoder ResNet50
        """
        ckpt = {
            "model_state_dict": self.image_encoder.state_dict(),
            "num_channels": self.image_encoder.num_channels,
            "num_classes": self.image_encoder.num_classes,
            "settings": asdict(self.image_encoder.settings),
        }
        torch.save(ckpt, path)
