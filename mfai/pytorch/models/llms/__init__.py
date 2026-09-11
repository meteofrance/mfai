from abc import ABC
from importlib.util import find_spec

# Check for optional dependency
if any(
    (
        find_spec("huggingface_hub") is None,
        find_spec("sentencepiece") is None,
        find_spec("tiktoken") is None,
        find_spec("tokenizers") is None,
    )
):
    raise ImportError(
        "To use mfai's llm models, install mfai's optional dependency\n\tmfai[llm]"
    )

from mfai.pytorch.models.llms.gpt2 import GPT2, CrossAttentionGPT2
from mfai.pytorch.models.llms.llama2 import Llama2
from mfai.pytorch.models.llms.llama3 import Llama3
from mfai.pytorch.models.resnet import ResNet50MLM
from mfai.pytorch.models.vit import VitEncoder
from mfai.pytorch.models.weather_projector import WeatherProjector


class FreezeMLMMixin(ABC):
    """
    A Mixin for (un)freezing llm and vision stages
    of a multimodal model.
    """

    backend: GPT2 | Llama2 | CrossAttentionGPT2 | Llama3
    vision_encoder: WeatherProjector | ResNet50MLM | VitEncoder

    def freeze_llm(self) -> None:
        """
        Freeze the LLM layers (not the vision layers).
        """
        for param in self.backend.parameters():
            param.requires_grad = False

    def unfreeze_llm(self) -> None:
        """
        Unfreeze the LLM layers.
        """
        for param in self.backend.parameters():
            param.requires_grad = True

    def freeze_vision(self) -> None:
        """
        Freeze the vision encoder layers.
        """
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def unfreeze_vision(self) -> None:
        """
        Unfreeze the vision encoder layers.
        """
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
