from dataclasses import asdict, dataclass

import einops
from dataclasses_json import dataclass_json
from torch import Tensor, nn

from mfai.pytorch.models.llms.fuyu import FreezeMLMMixin
from mfai.pytorch.models.llms.gpt2 import CrossAttentionGPT2
from mfai.pytorch.models.resnet import (
    ResNet50MLM,
    ResNet50MLMSettings,
)
from mfai.pytorch.namedtensor import NamedTensor


@dataclass_json
@dataclass
class XAttMultiModalLMSettings:
    """
    Settings for our cross attention multimodal language model.
    """

    emb_dim: int = 768  # Embedding dimension
    context_length: int = 1024  # Context length
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias
    vision_input_shape: tuple[int, int, int, int] = (
        256,
        256,
        10,
        10,
    )
    x_att_ratio: int = 4  # Cross attention layer ratio
    num_tokens_vision: int = 32  # Number of vision/weather tokens


class XAttMultiModalLM(FreezeMLMMixin, nn.Module):
    """
    A multimodal LLM with cross attention.
    Can use GPT2 or Llama2 as its LLM backend.
    """

    def __init__(
        self,
        settings: XAttMultiModalLMSettings = XAttMultiModalLMSettings(),
        vocab_size: int = 50257,
    ):
        super().__init__()
        self.settings = settings

        # Initialize our X ATT GPT2 backend
        self.backend = CrossAttentionGPT2(
            CrossAttentionGPT2.settings_kls(
                **{
                    k: v
                    for k, v in asdict(settings).items()
                    if k in CrossAttentionGPT2.settings_kls.__dataclass_fields__
                }
            ),
            vocab_size=vocab_size,
        )

        # Initialize our resnet50 vision encoder
        num_channels_viz = (
            settings.vision_input_shape[2] * settings.vision_input_shape[3]
        )
        self.vision_encoder = ResNet50MLM(
            num_channels=num_channels_viz,
            num_classes=settings.emb_dim,
            settings=ResNet50MLMSettings(num_tokens=settings.num_tokens_vision),
        )

    @property
    def context_length(self) -> int:
        return self.backend.context_length

    def forward(self, token_ids: Tensor, vision_input: NamedTensor) -> Tensor:
        # Reshape the vision input
        new_tensor = einops.rearrange(
            vision_input.tensor,
            "batch lat lon timestep features -> batch (timestep features) lat lon",
        )

        vis_embeds = self.vision_encoder(new_tensor)

        # Normalize the output
        vis_embeds = vis_embeds / vis_embeds.norm(dim=2, keepdim=True)
        return self.backend(token_ids, vis_embeds)
