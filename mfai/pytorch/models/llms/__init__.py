from torch import nn

from mfai.pytorch.models.llms.gpt2 import GPT2, CrossAttentionGPT2
from mfai.pytorch.models.llms.llama2 import Llama2


class FreezeMLMMixin:
    """
    A Mixin for (un)freezing llm and vision stages
    of a multimodal model
    """

    backend: GPT2 | Llama2 | CrossAttentionGPT2
    vision_encoder: nn.Module

    def freeze_llm(self) -> None:
        """
        Freeze the LLM layers (not the vision layers)
        """
        for param in self.backend.parameters():
            param.requires_grad = False

    def unfreeze_llm(self) -> None:
        """
        Unfreeze the LLM layers
        """
        for param in self.backend.parameters():
            param.requires_grad = True

    def freeze_vision(self) -> None:
        """
        Freeze the vision encoder layers
        """
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def unfreeze_vision(self) -> None:
        """
        Unfreeze the vision encoder layers
        """
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
