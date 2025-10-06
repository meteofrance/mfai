from dataclasses import asdict, dataclass
from typing import Literal

import torch
from dataclasses_json import dataclass_json
from torch import Tensor, nn

from mfai.pytorch.models.base import ModelType
from mfai.pytorch.models.llms import FreezeMLMMixin
from mfai.pytorch.models.llms.gpt2 import GPT2
from mfai.pytorch.models.llms.llama2 import Llama2
from mfai.pytorch.models.resnet import (
    ResNet50MLM,
    ResNet50MLMSettings,
)
from mfai.pytorch.models.vit import VitEncoder, ViTEncoderSettings
from mfai.pytorch.models.weather_projector import (
    WeatherProjector,
    WeatherProjectorSettings,
)
from mfai.pytorch.namedtensor import NamedTensor


@dataclass_json
@dataclass
class FuyuSettings:
    """
    Settings for a multimodal language model.
    """

    backend: str = "gpt2"
    emb_dim: int = 768  # Embedding dimension
    context_length: int = 1024  # Context length
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias
    hidden_dim: int = 768  # Size of the intermediate dimension in FeedForward - Llama2

    vision_input_shape: tuple[int, int, int, int] = (
        256,
        256,
        10,
        10,
    )  # lat_dim, lon_dim, timestep_dim, features_dim

    # Inject vision tokens at each stage ?
    inject_vision_each_stage: bool = False

    # choice of vision encoder
    # "resnet50", "linear"
    vision_encoder: Literal["resnet50", "linear", "vit"] = "linear"

    # number of tokens for ResNet50 encoder
    resnet_num_tokens: int = 32

    # absolute positional embedding for the vision encoder
    resnet_pos_embedding: bool = False

    # mlp output for the vision encoder
    resnet_mlp_output: bool = False

    # layer norm vis + txt tokens
    layer_norm_vis_txt: bool = True

    patch_size: int | tuple[int, int] = 8


class Fuyu(FreezeMLMMixin, nn.Module):
    """
    A multimodal LLM : vision/weather and txt inspired by Fuyu.
    Can use GPT2 or Llama2 as its LLM backend.
    """

    settings_kls = FuyuSettings
    model_type: Literal[ModelType.MULTIMODAL_LLM]

    def __init__(
        self,
        settings: FuyuSettings = FuyuSettings(),
        vocab_size: int = 50257,
    ):
        super().__init__()

        self.settings = settings
        # Init the backend model
        # Here we only pass the settings that are relevant to the backend model
        # by iterating over the fields of the settings object and filtering out
        self.backend: GPT2 | Llama2
        if settings.backend == "gpt2":
            self.backend = GPT2(
                GPT2.settings_kls(
                    **{
                        k: v
                        for k, v in asdict(settings).items()
                        if k in GPT2.settings_kls.__dataclass_fields__
                    }
                ),
                vocab_size=vocab_size,
            )
        elif settings.backend == "llama2":
            self.backend = Llama2(
                Llama2.settings_kls(
                    **{
                        k: v
                        for k, v in asdict(settings).items()
                        if k in Llama2.settings_kls.__dataclass_fields__
                    }
                ),
                vocab_size=vocab_size,
            )
        else:
            raise ValueError(f"Unknown backend: {settings.backend}")

        # Builds linear projection layer for weather input data (same for each time step)
        # lat_dim, lon_dim, timestep_dim, features_dim
        if settings.layer_norm_vis_txt:
            self.norm_or_ident: nn.Identity | nn.LayerNorm = nn.LayerNorm(
                self.settings.emb_dim
            )
        else:
            self.norm_or_ident = nn.Identity()

        self.vision_encoder: WeatherProjector | ResNet50MLM | VitEncoder

        if self.settings.vision_encoder == "linear":
            input_dims = (
                settings.vision_input_shape[0],
                settings.vision_input_shape[1],
                settings.vision_input_shape[-1],
            )
            s = WeatherProjectorSettings(
                input_dims=input_dims,
                embedding_dim=self.settings.emb_dim,
                patch_size=self.settings.patch_size,
            )
            self.vision_encoder = WeatherProjector(settings=s)

        elif self.settings.vision_encoder == "resnet50":
            self.vision_encoder = ResNet50MLM(
                num_channels=settings.vision_input_shape[3],
                num_classes=settings.emb_dim,
                settings=ResNet50MLMSettings(
                    num_tokens=settings.resnet_num_tokens,
                    pos_embedding=settings.resnet_pos_embedding,
                    mlp_output=settings.resnet_mlp_output,
                ),
            )
        elif self.settings.vision_encoder == "vit":
            # Initialize the ViT encoder, we have one input channel per feature per timestep
            self.vision_encoder = VitEncoder(
                in_channels=settings.vision_input_shape[-1],
                out_channels=settings.emb_dim,
                settings=ViTEncoderSettings(
                    emb_dim=settings.emb_dim,
                    transformer_dropout=settings.drop_rate,
                    emb_dropout=settings.drop_rate,
                    autopad_enabled=True,
                    patch_size=self.settings.patch_size,
                ),
                input_shape=settings.vision_input_shape[:2],
            )

        else:
            raise ValueError(
                f"Unknown vision encoder: {self.settings.vision_encoder}. Use 'linear', 'vit' or 'resnet50'."
            )

    @property
    def context_length(self) -> int:
        return self.backend.context_length

    def forward(self, token_ids: Tensor, vision_input: NamedTensor) -> Tensor:
        # token_ids shape=(B, n_tok), vision_input shape=(B, lat, lon, features, time)

        # Projection of weather input data into LLM token space
        vis_timesteps_embeds = []
        for timestep_nt in vision_input.iter_dim("timestep"):
            timestep_nt.rearrange_("batch lat lon features -> batch features lat lon")
            vis_timesteps_embeds.append(self.vision_encoder(timestep_nt.tensor))
            # shape = (B, n'_tok, embed_dim)
        vis_embeds = torch.cat(vis_timesteps_embeds, dim=1)
        # shape = (B, n'_tok * time, embed_dim)

        text_embeds = self.backend.tok_emb(token_ids) # (B, n_tok, embed_dim)

        vis_txt_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
        # shape = (B, n'_tok * time + n_tok, embed_dim)

        vis_txt_embeds = self.norm_or_ident(vis_txt_embeds)

        if vis_txt_embeds.shape[1] > self.context_length:
            print(
                f"Warning: Input sequence length {vis_txt_embeds.shape[1]} is longer than the model's context length {self.context_length}. Truncating input."
            )
            # Keep only the last context_length tokens:
            # shape = (batch_size,max(n'_tok * time + n_tok, context_len), embed_dim)
            vis_txt_embeds = vis_txt_embeds[:, -self.context_length :]

        embeds_idx = torch.arange(vis_txt_embeds.shape[1], device=token_ids.device)
        if hasattr(self.backend, "pos_emb") and isinstance(
            self.backend.pos_emb, nn.Embedding
        ):
            pos_embeds = self.backend.pos_emb(embeds_idx) # (max(...), embed_dim)
            x = vis_txt_embeds + pos_embeds.unsqueeze(0)
        else:
            x = vis_txt_embeds
        # x shape = (B, max(n'_tok * time + n_tok, context_len), embed_dim)

        if self.settings.inject_vision_each_stage:
            # Inject vision tokens at each stage
            logits = self.backend.forward_vectors(
                x, vis_txt_embeds[:, : vis_embeds.shape[1]]
            )
        else:
            logits = self.backend.forward_vectors(x)
        # logits shape = (B, max(n'_tok * time + n_tok, context_len), vocab_size)

        # removes the vision part of the logits
        return logits[:, vis_embeds.shape[1] :] # (B, n_tok, vocab_size)
