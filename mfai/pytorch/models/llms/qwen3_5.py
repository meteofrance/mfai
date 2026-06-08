"""Pytorch implementation of Qwen 3.5.
It is widely inspired by Sebastian Raschka's book and work
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/16_qwen3.5.

Qwen3.5 helper blocks copied from Hugging Face Transformers
Source file:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
"""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses_json import dataclass_json
from torch import Tensor

from mfai.pytorch.models.base import ModelType
from mfai.tokenizers import Qwen3_5Tokenizer

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    use_fast_implem = True
except ImportError:
    warnings.warn(
        "The fast implementation is not available because one of the required library "
        "is not installed. Falling back to torch implementation. "
        "To install follow https://github.com/fla-org/flash-linear-attention#installation and"
        " https://github.com/Dao-AILab/causal-conv1d"
    )
    use_fast_implem = False


@dataclass_json
@dataclass(slots=True)
class Qwen3_5Settings:
    """Qwen3.5-0.8B text configuration"""

    vocab_size: int = 248_320
    context_length: int = 262_144
    emb_dim: int = 1_024
    n_heads: int = 8
    n_layers: int = 24
    hidden_dim: int = 3_584
    head_dim: int = 256
    qk_norm: bool = True
    n_kv_groups: int = 2
    rope_base: int = 10_000_000
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    dtype: torch.dtype = torch.bfloat16
    layer_types: tuple[str, ...] = tuple(
        (["linear_attention"] * 3 + ["full_attention"]) * 6
    )
    hidden_activation: str = "silu"


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor, gate: Tensor | None = None) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(
    hidden_states: Tensor, attention_mask: Tensor | None
) -> Tensor:
    """
    Tunes out the hidden states for padding tokens,
    see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if (
        attention_mask is not None
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def torch_causal_conv1d_update(
    hidden_states: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size
    )
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, gate = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, gate)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    gate = F.pad(gate, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    gate = gate.reshape(gate.shape[0], gate.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    gate = gate.cumsum(dim=-1)
    decay_mask = ((gate.unsqueeze(-1) - gate.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * gate.exp().unsqueeze(-1))
    last_recurrent_state: Tensor = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * gate[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * gate[:, :, i, -1, None, None].exp()
            + (
                k_i * (gate[:, :, i, -1, None] - gate[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    last_recurrent_state_out = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state_out


def torch_recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    output_final_state: Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, gate = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, gate)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(
        value
    )
    last_recurrent_state: Tensor = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = gate[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    last_recurrent_state_out = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state_out


# Minimal change: enforce config dtype at the end to avoid bf16/fp32 matmul mismatch
# in a mixed notebook implementation
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, settings: Qwen3_5Settings, layer_idx: int):
        super().__init__()
        self.hidden_size = settings.emb_dim
        self.num_v_heads = settings.linear_num_value_heads
        self.num_k_heads = settings.linear_num_key_heads
        self.head_k_dim = settings.linear_key_head_dim
        self.head_v_dim = settings.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = settings.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = settings.hidden_activation
        self.layer_norm_epsilon = settings.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
            FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=settings.dtype
                if settings.dtype is not None
                else torch.get_default_dtype(),
            )
            if use_fast_implem
            else Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # self.causal_conv1d_fn = causal_conv1d_fn if use_fast_implem else None
        self.causal_conv1d_update = (
            causal_conv1d_update if use_fast_implem else torch_causal_conv1d_update
        )
        self.chunk_gated_delta_rule = (
            chunk_gated_delta_rule if use_fast_implem else torch_chunk_gated_delta_rule
        )
        self.recurrent_gated_delta_rule = (
            fused_recurrent_gated_delta_rule
            if use_fast_implem
            else torch_recurrent_gated_delta_rule
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Notebook adaptation for dtype consistency.
        if settings.dtype is not None:
            self.to(dtype=settings.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        cache_params: None = None,  # TODO : used for kv cache ?
        cache_position: None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # TODO :  WTF ?
        # use_precomputed_states = (
        #     cache_params is not None
        #     and cache_params.has_previous_state
        #     and seq_len == 1
        #     and cache_position is not None
        # )

        # getting projected states from cache if it exists
        # if cache_params is not None:
        #     conv_state = cache_params.conv_states[self.layer_idx]
        #     recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # if use_precomputed_states:
        #     # 2. Convolution sequence transformation
        #     # NOTE: the conv state is updated in `causal_conv1d_update`
        #     mixed_qkv = self.causal_conv1d_update(
        #         mixed_qkv,
        #         conv_state,
        #         self.conv1d.weight.squeeze(1),
        #         self.conv1d.bias,
        #         self.activation,
        #     )
        # else:
        # if cache_params is not None:
        #     conv_state = F.pad(
        #         mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
        #     )
        #     cache_params.conv_states[self.layer_idx] = conv_state
        if use_fast_implem:
            mixed_qkv = causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            gate=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

        # else:
        #     core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
        #         query,
        #         key,
        #         value,
        #         gate=g,
        #         beta=beta,
        #         initial_state=recurrent_state,
        #         output_final_state=cache_params is not None,
        #         use_qk_l2norm_in_kernel=True,
        #     )

        # Update cache
        # if cache_params is not None:
        #     cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Qwen3.5 uses (1 + weight) scaling with zero init
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(dtype=x.dtype)


def compute_rope_params(
    head_dim: int,
    theta_base: int = 10_000,
    context_length: int = 4096,
    partial_rotary_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {head_dim}.")

    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))

    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=dtype)[: (rotary_dim // 2)].float()
            / rotary_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    rot_dim = cos.shape[-1]
    if rot_dim > head_dim:
        raise ValueError(f"RoPE dim {rot_dim} cannot exceed head_dim {head_dim}.")

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    x1 = x_rot[..., : rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x_rot * cos) + (rotated * sin)

    x_out = torch.cat([x_rotated, x_pass], dim=-1)
    return x_out.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError(
                "`num_heads` must be divisible by `num_kv_groups`, got"
                f"num_heads={num_heads} and num_kv_groups={num_kv_groups}."
            )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            if d_in % num_heads != 0:
                raise ValueError(
                    "`d_in` must be divisible by `num_heads` if `head_dim` is not set, "
                    f"got num_heads={num_heads} and d_in={d_in}."
                )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # Qwen3.5 full-attention uses a gated Q projection (2x output dim)
        self.W_query = nn.Linear(d_in, self.d_out * 2, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        self.q_norm: RMSNorm | None
        self.k_norm: RMSNorm | None
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x: Tensor, mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, num_tokens, _ = x.shape

        q_and_gate = self.W_query(x)
        q_and_gate = q_and_gate.view(b, num_tokens, self.num_heads, self.head_dim * 2)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, num_tokens, self.d_out)

        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores * (self.head_dim**-0.5),
            dim=-1,
            dtype=torch.float32,
        ).to(queries.dtype)

        context = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )

        # Qwen3.5 full-attention uses a gated Q projection
        context = context * torch.sigmoid(gate)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, settings: Qwen3_5Settings) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            settings.emb_dim, settings.hidden_dim, dtype=settings.dtype, bias=False
        )
        self.fc2 = nn.Linear(
            settings.emb_dim, settings.hidden_dim, dtype=settings.dtype, bias=False
        )
        self.fc3 = nn.Linear(
            settings.hidden_dim, settings.emb_dim, dtype=settings.dtype, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, settings: Qwen3_5Settings, layer_type: str, layer_idx: int
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.token_mixer: GroupedQueryAttention | Qwen3_5GatedDeltaNet
        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(
                d_in=settings.emb_dim,
                num_heads=settings.n_heads,
                head_dim=settings.head_dim,
                num_kv_groups=settings.n_kv_groups,
                qk_norm=settings.qk_norm,
                dtype=settings.dtype,
            )
        elif layer_type == "linear_attention":
            self.token_mixer = Qwen3_5GatedDeltaNet(Qwen3_5Settings(), layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.ff = FeedForward(settings)
        self.norm1 = RMSNorm(settings.emb_dim, eps=settings.rms_norm_eps)
        self.norm2 = RMSNorm(settings.emb_dim, eps=settings.rms_norm_eps)

    def forward(self, x: Tensor, mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.layer_type == "full_attention":
            x = self.token_mixer(x, mask, cos, sin)
        else:
            x = self.token_mixer(x)

        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


class Qwen3_5(nn.Module):
    settings_kls = Qwen3_5Settings
    model_type = ModelType.LLM

    def __init__(self, settings: Qwen3_5Settings) -> None:
        super().__init__()
        self.context_length = settings.context_length
        self.tok_emb = nn.Embedding(
            settings.vocab_size, settings.emb_dim, dtype=settings.dtype
        )

        layer_types = settings.layer_types
        if len(layer_types) != settings.n_layers:
            raise ValueError(
                f"len(layer_types) must be equal to n_layers, got "
                f"len(layer_types)={len(layer_types)} and n_layers={settings.n_layers}"
            )

        self.trf_blocks = nn.ModuleList(
            [
                TransformerBlock(settings, layer_type, idx)
                for idx, layer_type in enumerate(layer_types)
            ]
        )

        self.final_norm = RMSNorm(settings.emb_dim, eps=settings.rms_norm_eps)
        self.out_head = nn.Linear(
            settings.emb_dim, settings.vocab_size, bias=False, dtype=settings.dtype
        )

        head_dim = (
            settings.emb_dim // settings.n_heads
            if settings.head_dim is None
            else settings.head_dim
        )
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=settings.rope_base,
            context_length=self.context_length,
            partial_rotary_factor=settings.partial_rotary_factor,
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.settings = settings

    def forward(self, in_idx: Tensor) -> Tensor:
        x = self.tok_emb(in_idx)

        num_tokens = x.shape[1]
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.settings.dtype))
        return logits

    def compute_memory_size(self, input_dtype: torch.dtype = torch.float32) -> float:
        total_params = 0
        total_grads = 0
        for param in self.parameters():
            # Calculate total number of elements per parameter
            param_size = param.numel()
            total_params += param_size
            # Check if gradients are stored for this parameter
            if param.requires_grad:
                total_grads += param_size

        # Calculate buffer size (non-parameters that require memory)
        total_buffers = sum(buf.numel() for buf in self.buffers())

        # Size in bytes = (Number of elements) * (Size of each element in bytes)
        # We assume parameters and gradients are stored in the same type as input dtype
        element_size = torch.tensor(0, dtype=input_dtype).element_size()
        total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

        # Convert bytes to gigabytes
        total_memory_gb = total_memory_bytes / (1024**3)

        return total_memory_gb

    def load_weights_from_dict(self, params: dict) -> None:
        def assign(
            left: Tensor, right: Tensor, tensor_name: str = "unknown"
        ) -> nn.Parameter:
            if left.shape != right.shape:
                raise ValueError(
                    f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
                )
            with torch.no_grad():
                left.copy_(right)
            return nn.Parameter(left)

        if "model.embed_tokens.weight" in params:
            model_prefix = "model"
        elif "model.language_model.embed_tokens.weight" in params:
            model_prefix = "model.language_model"
        else:
            raise KeyError("Could not find embed token weights in checkpoint.")

        def pkey(suffix: str) -> str:
            return f"{model_prefix}.{suffix}"

        self.tok_emb.weight = assign(
            self.tok_emb.weight,
            params[pkey("embed_tokens.weight")],
            pkey("embed_tokens.weight"),
        )

        n_layers = self.settings.n_layers
        layer_types = self.settings.layer_types

        for id_layer in range(n_layers):
            block = cast(TransformerBlock, self.trf_blocks[id_layer])
            token_mixer: GroupedQueryAttention | Qwen3_5GatedDeltaNet = (
                block.token_mixer
            )
            layer_type = layer_types[id_layer]

            if layer_type == "full_attention":
                att = cast(GroupedQueryAttention, token_mixer)
                att.W_query.weight = assign(
                    att.W_query.weight,
                    params[pkey(f"layers.{id_layer}.self_attn.q_proj.weight")],
                    pkey(f"layers.{id_layer}.self_attn.q_proj.weight"),
                )
                att.W_key.weight = assign(
                    att.W_key.weight,
                    params[pkey(f"layers.{id_layer}.self_attn.k_proj.weight")],
                    pkey(f"layers.{id_layer}.self_attn.k_proj.weight"),
                )
                att.W_value.weight = assign(
                    att.W_value.weight,
                    params[pkey(f"layers.{id_layer}.self_attn.v_proj.weight")],
                    pkey(f"layers.{id_layer}.self_attn.v_proj.weight"),
                )
                att.out_proj.weight = assign(
                    att.out_proj.weight,
                    params[pkey(f"layers.{id_layer}.self_attn.o_proj.weight")],
                    pkey(f"layers.{id_layer}.self_attn.o_proj.weight"),
                )
                if hasattr(att, "q_norm") and att.q_norm is not None:
                    att.q_norm.weight = assign(
                        att.q_norm.weight,
                        params[pkey(f"layers.{id_layer}.self_attn.q_norm.weight")],
                        pkey(f"layers.{id_layer}.self_attn.q_norm.weight"),
                    )
                if hasattr(att, "k_norm") and att.k_norm is not None:
                    att.k_norm.weight = assign(
                        att.k_norm.weight,
                        params[pkey(f"layers.{id_layer}.self_attn.k_norm.weight")],
                        pkey(f"layers.{id_layer}.self_attn.k_norm.weight"),
                    )

            elif layer_type == "linear_attention":
                lat = cast(Qwen3_5GatedDeltaNet, token_mixer)
                lat.dt_bias = assign(
                    lat.dt_bias,
                    params[pkey(f"layers.{id_layer}.linear_attn.dt_bias")],
                    pkey(f"layers.{id_layer}.linear_attn.dt_bias"),
                )
                lat.A_log = assign(
                    lat.A_log,
                    params[pkey(f"layers.{id_layer}.linear_attn.A_log")],
                    pkey(f"layers.{id_layer}.linear_attn.A_log"),
                )
                lat.conv1d.weight = assign(
                    lat.conv1d.weight,
                    params[pkey(f"layers.{id_layer}.linear_attn.conv1d.weight")],
                    pkey(f"layers.{id_layer}.linear_attn.conv1d.weight"),
                )
                lat.norm.weight = assign(
                    lat.norm.weight,
                    params[pkey(f"layers.{id_layer}.linear_attn.norm.weight")],
                    pkey(f"layers.{id_layer}.linear_attn.norm.weight"),
                )
                lat.out_proj.weight = assign(
                    lat.out_proj.weight,
                    params[pkey(f"layers.{id_layer}.linear_attn.out_proj.weight")],
                    pkey(f"layers.{id_layer}.linear_attn.out_proj.weight"),
                )
                lat.in_proj_qkv.weight = assign(
                    lat.in_proj_qkv.weight,
                    params[pkey(f"layers.{id_layer}.linear_attn.in_proj_qkv.weight")],
                    pkey(f"layers.{id_layer}.linear_attn.in_proj_qkv.weight"),
                )
                lat.in_proj_z.weight = assign(
                    lat.in_proj_z.weight,
                    params[pkey(f"layers.{id_layer}.linear_attn.in_proj_z.weight")],
                    pkey(f"layers.{id_layer}.linear_attn.in_proj_z.weight"),
                )
                lat.in_proj_b.weight = assign(
                    lat.in_proj_b.weight,
                    params[pkey(f"layers.{id_layer}.linear_attn.in_proj_b.weight")],
                    pkey(f"layers.{id_layer}.linear_attn.in_proj_b.weight"),
                )
                lat.in_proj_a.weight = assign(
                    lat.in_proj_a.weight,
                    params[pkey(f"layers.{id_layer}.linear_attn.in_proj_a.weight")],
                    pkey(f"layers.{id_layer}.linear_attn.in_proj_a.weight"),
                )

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            block.norm1.weight = assign(
                block.norm1.weight,
                params[pkey(f"layers.{id_layer}.input_layernorm.weight")],
                pkey(f"layers.{id_layer}.input_layernorm.weight"),
            )

            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[pkey(f"layers.{id_layer}.mlp.gate_proj.weight")],
                pkey(f"layers.{id_layer}.mlp.gate_proj.weight"),
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[pkey(f"layers.{id_layer}.mlp.up_proj.weight")],
                pkey(f"layers.{id_layer}.mlp.up_proj.weight"),
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[pkey(f"layers.{id_layer}.mlp.down_proj.weight")],
                pkey(f"layers.{id_layer}.mlp.down_proj.weight"),
            )
            block.norm2.weight = assign(
                block.norm2.weight,
                params[pkey(f"layers.{id_layer}.post_attention_layernorm.weight")],
                pkey(f"layers.{id_layer}.post_attention_layernorm.weight"),
            )

        self.final_norm.weight = assign(
            self.final_norm.weight,
            params[pkey("norm.weight")],
            pkey("norm.weight"),
        )

        if "lm_head.weight" in params:
            self.out_head.weight = assign(
                self.out_head.weight, params["lm_head.weight"], "lm_head.weight"
            )
        elif pkey("lm_head.weight") in params:
            self.out_head.weight = assign(
                self.out_head.weight,
                params[pkey("lm_head.weight")],
                pkey("lm_head.weight"),
            )
        else:
            self.out_head.weight = self.tok_emb.weight

    def download_weights_from_hf(self, model_dir: Path) -> None:
        repo_id = "Qwen/Qwen3.5-0.8B"
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=model_dir)
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)

        self.load_weights_from_dict(weights_dict)

    def generate_output_stream(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> Iterator[Tensor]:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self(token_ids)[:, -1]
                next_token = torch.argmax(out, dim=-1, keepdim=True)

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break

                yield next_token

                token_ids = torch.cat([token_ids, next_token], dim=1)

    def generate_text(
        self,
        prompt: str,
        tokenizer: Qwen3_5Tokenizer,
        max_new_tokens: int,
    ) -> str:
        input_token_ids = tokenizer.encode(prompt)
        device = next(self.parameters()).device
        input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(
            0
        )

        text = prompt
        for token in self.generate_output_stream(
            token_ids=input_token_ids_tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eot_token,
        ):
            token_id = token.squeeze(0).tolist()
            text += tokenizer.decode(token_id)
        return text


if __name__ == "__main__":
    import json
    import os
    import time
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    torch.manual_seed(123)
    model = Qwen3_5(Qwen3_5Settings())
    print(model)
    print(model(torch.tensor([1, 2, 3]).unsqueeze(0)))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

    print(
        f"float32 (PyTorch default): {model.compute_memory_size(torch.float32):.2f} GB"
    )
    print(f"bfloat16: {model.compute_memory_size(torch.bfloat16):.2f} GB")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("--> device : ", device)
    model.to(device)

    local_dir = Path("/scratch/shared/qwen3.5/")

    model.download_weights_from_hf(local_dir)
    model.to(device)

    tokenizer = Qwen3_5Tokenizer(
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)
    print(text)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    generated_tokens = 0

    for token in model.generate_output_stream(
        token_ids=input_token_ids_tensor,
        max_new_tokens=500,
        eos_token_id=tokenizer.eot_token,
    ):
        generated_tokens += 1
        token_id = token.squeeze(0).tolist()
        print(tokenizer.decode(token_id), end="", flush=True)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n\nGeneration speed: {tokens_per_sec:.2f} tokens/sec")

    if torch.cuda.is_available():

        def calc_gpu_gb(x: int) -> str:
            return f"{x / 1024 / 1024 / 1024:.2f} GB"

        print(f"GPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")

# TODO :
# - check that it works on GPU
# - check that it works with fast attention
# - kv cache ? see TODOs ?
