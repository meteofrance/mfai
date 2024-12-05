"""
SegFormer adapted from https://github.com/lucidrains/segformer-pytorch
"""

from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Tuple

import torch
from dataclasses_json import dataclass_json
from einops import rearrange
from torch import einsum, nn

from .base import ModelABC, ModelType


def exists(val):
    return val is not None


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


@dataclass_json
@dataclass(slots=True)
class SegformerSettings:
    dims: Tuple[int, ...] = (32, 64, 160, 256)
    heads: Tuple[int, ...] = (1, 2, 5, 8)
    ff_expansion: Tuple[int, ...] = (8, 8, 4, 4)
    reduction_ratio: Tuple[int, ...] = (8, 4, 2, 1)
    num_layers: int = 2
    decoder_dim: int = 256

    # Number of channels after downsampling
    # injected in the mit
    num_downsampling_chans: int = 32


class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class EfficientSelfAttention(nn.Module):
    def __init__(self, *, dim, heads, reduction_ratio):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(
            dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False
        )
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) (x y) c", h=heads), (q, k, v)
        )

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) (x y) c -> b (h c) x y", h=heads, x=h, y=w)
        return self.to_out(out)


class MixFeedForward(nn.Module):
    def __init__(self, *, dim, expansion_factor):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class MiT(nn.Module):
    def __init__(
        self, *, channels, dims, heads, ff_expansion, reduction_ratio, num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (
            (dim_in, dim_out),
            (kernel, stride, padding),
            num_layers,
            ff_expansion,
            heads,
            reduction_ratio,
        ) in zip(
            dim_pairs,
            stage_kernel_stride_pad,
            num_layers,
            ff_expansion,
            heads,
            reduction_ratio,
        ):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel**2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim_out,
                                EfficientSelfAttention(
                                    dim=dim_out,
                                    heads=heads,
                                    reduction_ratio=reduction_ratio,
                                ),
                            ),
                            PreNorm(
                                dim_out,
                                MixFeedForward(
                                    dim=dim_out, expansion_factor=ff_expansion
                                ),
                            ),
                        ]
                    )
                )

            self.stages.append(
                nn.ModuleList([get_overlap_patches, overlap_patch_embed, layers])
            )

    def forward(self, x, return_layer_outputs=False):
        h, w = x.shape[-2:]

        layer_outputs = []
        i = 0
        for get_overlap_patches, overlap_embed, layers in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, "b c (h w) -> b c h w", h=h // ratio)

            x = overlap_embed(x)
            for attn, ff in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)
            i += 1

        ret = x if not return_layer_outputs else layer_outputs
        return ret


class Segformer(ModelABC, nn.Module):
    """
    Segformer architecture with extra
    upsampling in the decoder to match
    the input image size.
    """

    onnx_supported = True
    settings_kls = SegformerSettings
    supported_num_spatial_dims = (2,)
    features_last = False
    model_type = ModelType.VISION_TRANSFORMER
    num_spatial_dims: int = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: Tuple[int, int],
        settings: SegformerSettings = SegformerSettings(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        dims, heads, ff_expansion, reduction_ratio, num_layers = map(
            partial(cast_tuple, depth=4),
            (
                settings.dims,
                settings.heads,
                settings.ff_expansion,
                settings.reduction_ratio,
                settings.num_layers,
            ),
        )
        assert all(
            map(
                lambda t: len(t) == 4,
                (dims, heads, ff_expansion, reduction_ratio, num_layers),
            )
        ), "only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values"

        # reduce image size by a factor 2
        # and spread over num_downsampling_chans channels
        num_chans = settings.num_downsampling_chans
        self.downsampler = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, num_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_chans, num_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_chans),
            nn.ReLU(inplace=True),
        )

        self.mit = MiT(
            channels=num_chans,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers,
        )

        self.to_fused = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, settings.decoder_dim, 1),
                    nn.Upsample(scale_factor=2**i),
                )
                for i, dim in enumerate(dims)
            ]
        )

        # Step by step upsampling
        # to match the input width and height dimensions
        dim_out = settings.decoder_dim
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim_out * 4, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=dim_out),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim_out, dim_out // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=dim_out // 2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim_out // 2, dim_out // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=dim_out // 4),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dim_out // 4, out_channels, kernel_size=3, padding=1),
        )
        self.check_required_attributes()

    @property
    def settings(self) -> SegformerSettings:
        return self._settings

    def forward(self, x):
        x = self.downsampler(x)
        layer_outputs = self.mit(x, return_layer_outputs=True)

        fused = [
            to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)
        ]
        fused = torch.cat(fused, dim=1)
        return self.upsampler(fused)
