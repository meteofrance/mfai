"""
Magnitude-Preserving Unet based on: "Karras et al.,
https://arxiv.org/abs/2312.02696
Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py
Used as the default model for diffusion
"""

import math
from functools import partial, wraps
from packaging import version #type: ignore
import torch
from torch import nn
import torch.nn.functional as F #type: ignore
from typing import Tuple, Union
from dataclasses import dataclass
from dataclasses_json import dataclass_json # type: ignore
from mfai.torch.models.base import ModelABC, AutoPaddingModel, ModelType

from einops import rearrange, repeat, pack, unpack # type: ignore
from collections import namedtuple

#from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/attend.py
#for Attend

# constants

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        if device_version > version.parse('8.0'):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out

#building blocks

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def xnor(x, y):
    return not (x ^ y)

def append(arr, el):
    arr.append(el)

def prepend(arr, el):
    arr.insert(0, el)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# in paper, they use eps 1e-4 for pixelnorm

def l2norm(t, dim = -1, eps = 1e-12):
    return F.normalize(t, dim = dim, eps = eps)

def normalize_weight(weight, eps = 1e-4):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * math.sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')

class Conv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):

        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / math.sqrt(self.fan_in)

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value = 1.)

        return F.conv2d(x, weight, padding='same')

class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain

class MPFourierEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * math.sqrt(2)

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / math.sqrt(self.fan_in)
        return F.linear(x, weight)

class MPAdd(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = math.sqrt((1 - t) ** 2 + t ** 2)
        return num / den

class MPSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x) / 0.596


class MPCat(nn.Module):
    def __init__(self, t = 0.5, dim = -1):
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        C = math.sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

        a = a * (1. - t) / math.sqrt(Na)
        b = b * t / math.sqrt(Nb)

        return C * torch.cat((a, b), dim = dim)

class PixelNorm(nn.Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        # high epsilon for the pixel norm in the paper
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return l2norm(x, dim = dim, eps = self.eps) * math.sqrt(x.shape[dim])

# forced weight normed conv2d and linear
# algorithm 1 in paper

def normalize_weight(weight, eps = 1e-4):
    weight, ps = pack_one(weight, 'o *')
    normed_weight = l2norm(weight, eps = eps)
    normed_weight = normed_weight * math.sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, 'o *')


class Encoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        downsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv2d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        self.pixel_norm = PixelNorm(dim = 1)

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(curr_dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(math.ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    def forward(
        self,
        x,
        emb = None
    ):
        if self.downsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h // 2, w // 2), mode = 'bilinear')
            x = self.downsample_conv(x)

        x = self.pixel_norm(x)

        res = x.clone()

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

class Decoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        attn_flash = False,
        upsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.upsample = upsample
        self.needs_skip = not upsample

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.res_mp_add = MPAdd(t = mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim = dim_out,
                heads = max(math.ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
                flash = attn_flash
            )

    def forward(
        self,
        x,
        emb = None
    ):
        if self.upsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h * 2, w * 2), mode = 'bilinear')

        res = self.res_conv(x)

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        flash = False,
        mp_add_t = 0.3
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim = -1)

        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1)
        self.to_out = Conv2d(hidden_dim, dim, 1)

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        res, b, c, h, w = x, *x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        return self.mp_add(out, res)

#mfai implementation
@dataclass_json
@dataclass(slots=True)
class UNetKarrasSettings:
    #default settings from implementation
    #image_size = ?
    dim = 192
    dim_max = 768            # channels will double every downsample and cap out to this value
    num_classes = None       # in paper, they do 1000 classes for a popular benchmark
    channels = 4             # 4 channels in paper for some reason, must be alpha channel?
    num_downsamples = 3
    num_blocks_per_stage = 4
    attn_res = (16, 8)
    fourier_dim = 16
    attn_dim_head = 64
    attn_flash = False
    mp_cat_t = 0.5
    mp_add_emb_t = 0.5
    attn_res_mp_add_t = 0.3
    resnet_mp_add_t = 0.3
    dropout = 0.1
    self_condition = False

class UNetKarras(ModelABC, nn.Module):
    settings_kls = UNetKarrasSettings
    onnx_supported: bool = True
    supported_num_spatial_dims = (2,)
    num_spatial_dims: int = 2
    features_last: bool = False
    model_type: int = ModelType.DIFFUSION
    register: bool = True
    #forced signature
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        settings: UNetKarrasSettings,
        input_shape: Union[None, Tuple[int, int]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        super().__init__()

        self.self_condition = settings.self_condition

        # determine dimensions

        self.channels = settings.channels
        self.image_size = settings.image_size
        input_channels = settings.channels * (2 if settings.self_condition else 1)

        # input and output blocks

        self.input_block = Conv2d(input_channels, settings.dim, 3, concat_ones_to_input = True)

        self.output_block = nn.Sequential(
            Conv2d(settings.dim, settings.channels, 3),
            Gain()
        )

        # time embedding

        emb_dim = settings.dim * 4

        self.to_time_emb = nn.Sequential(
            MPFourierEmbedding(settings.fourier_dim),
            Linear(settings.fourier_dim, emb_dim)
        )

        # class embedding

        self.needs_class_labels = exists(settings.num_classes)
        self.num_classes = settings.num_classes

        if self.needs_class_labels:
            self.to_class_emb = Linear(settings.num_classes, 4 * settings.dim)
            self.add_class_emb = MPAdd(t = settings.mp_add_emb_t)

        # final embedding activations

        self.emb_activation = MPSiLU()

        # number of downsamples

        self.num_downsamples = settings.num_downsamples

        # attention

        attn_res = set(cast_tuple(attn_res))

        # resnet block

        block_kwargs = dict(
            dropout = settings.dropout,
            emb_dim = emb_dim,
            attn_dim_head = settings.attn_dim_head,
            attn_res_mp_add_t = settings.attn_res_mp_add_t,
            attn_flash = settings.attn_flash
        )

        # unet encoder and decoders

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        curr_dim = settings.dim
        curr_res = settings.image_size

        self.skip_mp_cat = MPCat(t = settings.mp_cat_t, dim = 1)

        # take care of skip connection for initial input block and first three encoder blocks

        prepend(self.ups, Decoder(settings.dim * 2, settings.dim, **block_kwargs))

        assert settings.num_blocks_per_stage >= 1

        for _ in range(settings.num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)

            append(self.downs, enc)
            prepend(self.ups, dec)

        # stages

        for _ in range(self.num_downsamples):
            dim_out = min(settings.dim_max, curr_dim * 2)
            upsample = Decoder(dim_out, curr_dim, has_attn = curr_res in attn_res, upsample = True, **block_kwargs)

            curr_res //= 2
            has_attn = curr_res in attn_res

            downsample = Encoder(curr_dim, dim_out, downsample = True, has_attn = has_attn, **block_kwargs)

            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(self.ups, Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs))

            for _ in range(settings.num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn = has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs)

                settings.append(self.downs, enc)
                settings.prepend(self.ups, dec)

            curr_dim = dim_out

        # take care of the two middle decoders

        mid_has_attn = curr_res in attn_res

        self.mids = nn.ModuleList([
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
            Decoder(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
        ])

        self.out_dim = settings.channels

    @property
    def downsample_factor(self):
        return 2 ** self.num_downsamples

    def forward(
        self,
        x,
        time,
        self_cond = None,
        class_labels = None
    ):
        # validate image shape

        assert x.shape[1:] == (self.channels, self.image_size, self.image_size)

        # self conditioning

        if self.self_condition:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim = 1)
        else:
            assert not exists(self_cond)

        # time condition

        time_emb = self.to_time_emb(time)

        # class condition

        assert xnor(exists(class_labels), self.needs_class_labels)

        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)

            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * math.sqrt(self.num_classes)

            class_emb = self.to_class_emb(class_labels)

            time_emb = self.add_class_emb(time_emb, class_emb)

        # final mp-silu for embedding

        emb = self.emb_activation(time_emb)

        # skip connections

        skips = []

        # input block

        x = self.input_block(x)

        skips.append(x)

        # down

        for encoder in self.downs:
            x = encoder(x, emb = emb)
            skips.append(x)

        # mid

        for decoder in self.mids:
            x = decoder(x, emb = emb)

        # up

        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)

            x = decoder(x, emb = emb)

        # output block

        return self.output_block(x)