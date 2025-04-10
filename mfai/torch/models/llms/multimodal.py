import torch
from torch import nn
from torch import Tensor
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
import math
import einops
from mfai.torch.models.llms import GPT2, Llama2
from mfai.torch.models.base import ModelType
from mfai.torch.namedtensor import NamedTensor
from mfai.torch.models.resnet import ResNet50


@dataclass_json
@dataclass
class MultiModalLMSettings:
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
    vision_encoder: str = "linear"


class MultiModalLM(nn.Module):
    """
    A multimodal LLM : vision/weather and txt façon Fuyu.
    Can use GPT2 or Llama2 as its LLM backend.
    """

    settings_kls = MultiModalLMSettings
    model_type: ModelType.MULTIMODAL_LLM

    def __init__(
        self,
        settings: MultiModalLMSettings = MultiModalLMSettings(),
        vocab_size: int = 50257,
    ):
        super().__init__()

        self.settings = settings
        # Init the backend model
        # Here we only pass the settings that are relevant to the backend model
        # by iterating over the fields of the settings object and filtering out
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
            self.linear_vis_projs = nn.ModuleList(
                [
                    nn.Linear(spatial_dims, settings.emb_dim)
                    for _ in range(settings.vision_input_shape[3])
                ]
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
            self.resnet50 = ResNet50(
                num_channels=settings.vision_input_shape[3]
                * settings.vision_input_shape[2],
                num_classes=settings.emb_dim,
            )
        else:
            raise ValueError(
                f"Unknown vision encoder: {self.settings.vision_encoder}. Use 'linear' or 'resnet50'."
            )

    @property
    def context_length(self):
        return self.backend.context_length

    def freeze_llm(self):
        """
        Freeze the LLM layers (not the vision layers)
        """
        for param in self.backend.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        """
        Unfreeze the LLM layers
        """
        for param in self.backend.parameters():
            param.requires_grad = True

    def forward(self, text_tokens: Tensor, vision_input: NamedTensor) -> Tensor:
        # Linear projection of weather input data
        vis_timesteps_embeds = []
        if self.settings.vision_encoder == "linear":
            for timestep_nt in vision_input.iter_dim("timestep", bare_tensor=False):
                timestep_embed = []
                # batch, lat, lon, features
                # rearrange to batch, features, lat, lon
                timestep_nt.rearrange_(
                    "batch lat lon features -> batch features lat lon"
                )

                for i in range(timestep_nt.dim_size("features")):
                    timestep_tensor = timestep_nt.select_dim("features", i)
                    timestep_tensor = self.downsampler(timestep_tensor)
                    timestep_tensor = timestep_tensor.flatten(1, 2)
                    if self.layer_norm_vis:
                        timestep_tensor = self.vis_layer_norms[i](timestep_tensor)
                    timestep_embed.append(self.linear_vis_projs[i](timestep_tensor))

                timestep_embed = torch.stack(timestep_embed, dim=1)
                vis_timesteps_embeds.append(timestep_embed)
                vis_embeds = torch.cat(vis_timesteps_embeds, dim=1)

        elif self.settings.vision_encoder == "resnet50":
            new_tensor = einops.rearrange(
                vision_input.tensor,
                "batch lat lon timestep features -> batch (timestep features) lat lon",
            )

            # Resnet50 encoder
            vis_embeds = self.resnet50(new_tensor)

            # Normalize the output
            vis_embeds = vis_embeds / vis_embeds.norm(dim=1, keepdim=True)

            vis_embeds = vis_embeds.unsqueeze(1)

        else:
            raise ValueError(
                f"Unknown vision encoder: {self.settings.vision_encoder}. Use 'linear' or 'resnet50'."
            )

        text_embeds = self.backend.tok_emb(text_tokens)

        vis_txt_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
        if vis_txt_embeds.shape[1] > self.context_length:
            print(
                f"Warning: Input sequence length {vis_txt_embeds.shape[1]} is longer than the model's context length {self.context_length}. Truncating input."
            )
            # Keep only the last context_length tokens, (batch_size, context_length)
            vis_txt_embeds = vis_txt_embeds[:, -self.context_length :]
        embeds_idx = torch.arange(vis_txt_embeds.shape[1], device=text_tokens.device)

        if hasattr(self.backend, "pos_emb"):
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
