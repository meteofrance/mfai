"""
Implementation of CLIP (Contrastive Langage-Image Pre-training) model. Based on the original https://arxiv.org/abs/2103.00020
"""

from dataclasses import dataclass
from typing import Tuple, Union

import torch

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
            torch.ones([]) * torch.log(torch.Tensor([settings.init_temperature]))
        )

    def forward(
        self, text_tokens: torch.Tensor, image_input: NamedTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens = text_tokens[
            :, -self.text_encoder.context_length :
        ]  # Keep only the last context_length tokens

        image_features = self.image_encoder(image_input.tensor)
        text_features = self.text_encoder.embed_tokens(text_tokens)

        # We consider that EOT token (in practice the highest number in each sequence) represents
        # the sentence, as it said in the original article (p.5):
        # "The [EOS] token are treated as the feature representation of the text which is
        # layer normalized and then linearly projected into the multi-modal embedding space."
        text_features = self.text_norm(text_features)
        text_features = (
            text_features[
                torch.arange(len(text_tokens), device=text_tokens.device),
                text_tokens.argmax(dim=-1),
            ]
            @ self.text_projection
        )

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        image_logits = image_features @ text_features.T * torch.exp(self.temperature)
        text_logits = image_logits.T

        return text_logits, image_logits
