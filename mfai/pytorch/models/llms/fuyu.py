import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import einops
import torch
from dataclasses_json import dataclass_json
from torch import Tensor, nn

from mfai.pytorch.models.base import ModelType
from mfai.pytorch.models.llms import FreezeMLMMixin
from mfai.pytorch.models.llms.gpt2 import GPT2
from mfai.pytorch.models.llms.llama2 import Llama2
from mfai.pytorch.models.resnet import (
    ResNet50,
    ResNet50MLM,
    ResNet50MLMSettings,
    ResNet50Settings,
)
from mfai.pytorch.models.vit import VitEncoder, ViTEncoderSettings
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
    downsampling_rate: int = 2  # Downsampling rate for the vision input

    vision_input_shape: tuple[int, int, int, int] = (
        256,
        256,
        10,
        10,
    )  # lat_dim, lon_dim, timestep_dim, features_dim
    layer_norm_vis: bool = True

    # Inject vision tokens at each stage ?
    inject_vision_each_stage: bool = False

    # choice of vision encoder
    # "resnet50", "linear"
    vision_encoder: Literal["resnet50", "linear", "vit", "vit_by_timestep", "vit_by_feature"] = "linear"
    # Optional checkpoint path for the resnet encoder
    resnet_checkpoint: None | Path = None

    # number of tokens for ResNet50 encoder
    resnet_num_tokens: int = 32

    # absolute positional embedding for the vision encoder
    resnet_pos_embedding: bool = False

    # mlp output for the vision encoder
    resnet_mlp_output: bool = False


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

        downsampled_lat = math.floor(
            settings.vision_input_shape[0] / settings.downsampling_rate
        )
        downsampled_lon = math.floor(
            settings.vision_input_shape[1] / settings.downsampling_rate
        )
        spatial_dims = downsampled_lat * downsampled_lon

        # downsampled spatial dims
        self.downsampler = nn.MaxPool2d(settings.downsampling_rate)

        if self.settings.vision_encoder == "linear":
            # One linear projection per feature/weather field
            self.vision_encoder: nn.ModuleList | ResNet50 | ResNet50MLM | VitEncoder = (
                nn.ModuleList(
                    [
                        nn.Linear(spatial_dims, settings.emb_dim)
                        for _ in range(settings.vision_input_shape[3])
                    ]
                )
            )

            self.layer_norm_vis = settings.layer_norm_vis
            if settings.layer_norm_vis:
                # One Layer Norm per feature
                self.vis_layer_norms = nn.ModuleList(
                    [
                        torch.nn.LayerNorm(spatial_dims)
                        for _ in range(settings.vision_input_shape[3])
                    ]
                )
        elif self.settings.vision_encoder == "resnet50":
            num_classes = settings.emb_dim
            num_channels = (
                settings.vision_input_shape[2] * settings.vision_input_shape[3]
            )

            if settings.resnet_checkpoint:
                # Load the checkpoint of the pretrained resnet encoder if a path is provided
                checkpoint = torch.load(settings.resnet_checkpoint, weights_only=True)
                if checkpoint["num_channels"] != num_channels:
                    raise ValueError(
                        f"Checkpoint num_channels {checkpoint['num_channels']} does not match the model num_channels {num_channels}."
                    )
                if checkpoint["num_classes"] != num_classes:
                    raise ValueError(
                        f"Checkpoint num_classes {checkpoint['num_classes']} does not match the model num_classes {num_classes}."
                    )
                # Instantiate the ResNet50 encoder with the same parameters as the checkpoint
                self.vision_encoder = ResNet50(
                    num_channels=checkpoint["num_channels"],
                    num_classes=checkpoint["num_classes"],
                    settings=ResNet50Settings(**checkpoint["settings"]),
                )
                # Load the pretrained weights
                self.vision_encoder.load_state_dict(checkpoint["model_state_dict"])

            else:
                self.vision_encoder = ResNet50MLM(
                    num_channels=num_channels,
                    num_classes=num_classes,
                    settings=ResNet50MLMSettings(
                        num_tokens=settings.resnet_num_tokens,
                        pos_embedding=settings.resnet_pos_embedding,
                        mlp_output=settings.resnet_mlp_output,
                    ),
                )
        elif self.settings.vision_encoder == "vit":
            # Initialize the ViT encoder, we have one input channel per feature per timestep
            self.vision_encoder = VitEncoder(
                in_channels=settings.vision_input_shape[2]
                * settings.vision_input_shape[3],
                out_channels=settings.emb_dim,
                settings=ViTEncoderSettings(
                    emb_dim=settings.emb_dim,
                    transformer_dropout=settings.drop_rate,
                    emb_dropout=settings.drop_rate,
                    autopad_enabled=True,
                ),
                input_shape=settings.vision_input_shape[:2],
            )

        elif self.settings.vision_encoder == "vit_by_timestep":
            # Initialize the ViT encoder, we have one input channel per feature per timestep
            self.vision_encoder = VitEncoder(
                in_channels=settings.vision_input_shape[3],
                out_channels=settings.emb_dim,
                settings=ViTEncoderSettings(
                    emb_dim=settings.emb_dim,
                    transformer_dropout=settings.drop_rate,
                    emb_dropout=settings.drop_rate,
                    autopad_enabled=True,
                ),
                input_shape=settings.vision_input_shape[:2],
            )
            self.vit_layer_norm = torch.nn.LayerNorm(settings.emb_dim)

        elif self.settings.vision_encoder == "vit_by_feature":
            # Initialize the ViT encoder, we have one input channel per feature per timestep
            self.vision_encoder = VitEncoder(
                in_channels=settings.vision_input_shape[2],
                out_channels=settings.emb_dim,
                settings=ViTEncoderSettings(
                    emb_dim=settings.emb_dim,
                    transformer_dropout=settings.drop_rate,
                    emb_dropout=settings.drop_rate,
                    autopad_enabled=True,
                ),
                input_shape=settings.vision_input_shape[:2],
            )
            self.vit_layer_norm = torch.nn.LayerNorm(settings.emb_dim)

        else:
            raise ValueError(
                f"Unknown vision encoder: {self.settings.vision_encoder}. Use 'linear' or 'resnet50'."
            )

    @property
    def context_length(self) -> int:
        return self.backend.context_length

    def forward(self, token_ids: Tensor, vision_input: NamedTensor) -> Tensor:
        # Linear projection of weather input data
        vis_timesteps_embeds = []
        if self.settings.vision_encoder == "linear":
            for timestep_nt in vision_input.iter_dim("timestep"):
                timestep_embed = []
                # batch, lat, lon, features
                # rearrange to batch, features, lat, lon
                timestep_nt.rearrange_(
                    "batch lat lon features -> batch features lat lon"
                )

                for i in range(timestep_nt.dim_size("features")):
                    timestep_tensor = timestep_nt.select_tensor_dim("features", i)
                    timestep_tensor = self.downsampler(timestep_tensor)
                    timestep_tensor = timestep_tensor.flatten(1, 2)
                    if self.layer_norm_vis:
                        timestep_tensor = self.vis_layer_norms[i](timestep_tensor)
                    timestep_embed.append(
                        self.vision_encoder[i](timestep_tensor)  # type: ignore[index]
                    )

                timestep_embed_tensor = torch.stack(timestep_embed, dim=1)
                vis_timesteps_embeds.append(timestep_embed_tensor)
            vis_embeds = torch.cat(vis_timesteps_embeds, dim=1)

        elif self.settings.vision_encoder == "resnet50":
            new_tensor = einops.rearrange(
                vision_input.tensor,
                "batch lat lon timestep features -> batch (timestep features) lat lon",
            )

            # Resnet50 encoder
            vis_embeds = self.vision_encoder(new_tensor)

            # resnet50mlm already outputs an extra token dim
            if isinstance(self.vision_encoder, ResNet50):
                vis_embeds = vis_embeds.unsqueeze(1)

            # Normalize the output along embedding dimension
            vis_embeds = vis_embeds / vis_embeds.norm(dim=2, keepdim=True)

        elif self.settings.vision_encoder == "vit":
            # Reshape the vision input for ViT
            new_tensor = einops.rearrange(
                vision_input.tensor,
                "batch lat lon timestep features -> batch (timestep features) lat lon",
            )
            # ViT encoder
            vis_embeds = self.vision_encoder(new_tensor)

            # Normalize the output along embedding dimension
            vis_embeds = vis_embeds / vis_embeds.norm(dim=2, keepdim=True)

        elif self.settings.vision_encoder == "vit_by_timestep":
            for timestep_nt in vision_input.iter_dim("timestep"):
                timestep_embed = []
                # batch, lat, lon, features
                # rearrange to batch, features, lat, lon
                new_tensor = einops.rearrange(
                timestep_nt.tensor,
                "batch lat lon features -> batch features lat lon",
                )

                # ViT encoder
                vis_embeds_nt = self.vision_encoder(new_tensor)

                # Normalize the output along embedding dimension
                vis_embeds_nt = self.vit_layer_norm(vis_embeds_nt)
                vis_timesteps_embeds.append(vis_embeds_nt)
            vis_embeds = torch.cat(vis_timesteps_embeds, dim=1)

        elif self.settings.vision_encoder == "vit_by_feature":
            for feature_nt in vision_input.iter_dim("features"):
                feature_embed = []
                # batch, lat, lon, timesteps
                # rearrange to batch, timesteps, lat, lon
                new_tensor = einops.rearrange(
                feature_nt.tensor,
                "batch lat lon timestep -> batch timestep lat lon",
                )

                # ViT encoder
                vis_embeds_nt = self.vision_encoder(new_tensor)

                # Normalize the output along embedding dimension
                vis_embeds_nt = self.vit_layer_norm(vis_embeds_nt)
                vis_timesteps_embeds.append(vis_embeds_nt)
            vis_embeds = torch.cat(vis_timesteps_embeds, dim=1)

        else:
            raise ValueError(
                f"Unknown vision encoder: {self.settings.vision_encoder}. Use 'linear' or 'resnet50'."
            )

        text_embeds = self.backend.tok_emb(token_ids)

        vis_txt_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
        if vis_txt_embeds.shape[1] > self.context_length:
            print(
                f"Warning: Input sequence length {vis_txt_embeds.shape[1]} is longer than the model's context length {self.context_length}. Truncating input."
            )
            # Keep only the last context_length tokens, (batch_size, context_length)
            vis_txt_embeds = vis_txt_embeds[:, -self.context_length :]
        embeds_idx = torch.arange(vis_txt_embeds.shape[1], device=token_ids.device)

        if hasattr(self.backend, "pos_emb") and isinstance(
            self.backend.pos_emb, nn.Embedding
        ):
            pos_embeds = self.backend.pos_emb(embeds_idx)
            x = vis_txt_embeds + pos_embeds.unsqueeze(0)
        else:
            x = vis_txt_embeds

        if self.settings.inject_vision_each_stage:
            # Inject vision tokens at each stage
            logits = self.backend.forward_vectors(
                x, vis_txt_embeds[:, : vis_embeds.shape[1]]
            )
        else:
            logits = self.backend.forward_vectors(x)

        # removes the vision part of the logits
        return logits[:, vis_embeds.shape[1] :]
