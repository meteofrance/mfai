"""Pytorch implementation of Qwen 3.5.
It is widely inspired by Sebastian Raschka's book and work
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/16_qwen3.5.

Qwen3.5 helper blocks copied from Hugging Face Transformers
Source file:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


import re
from tokenizers import Tokenizer


# Placeholder types for copied annotations
class Qwen3_5Config:
    pass


class Qwen3_5DynamicCache:
    pass


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x, dim=-1, eps=1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# Minimal change: enforce config dtype at the end to avoid bf16/fp32 matmul mismatch
# in a mixed notebook implementation
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

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
            Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            print(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Notebook adaptation for dtype consistency.
        if config.dtype is not None:
            self.to(dtype=config.dtype)

    def forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
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

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Qwen3.5 uses (1 + weight) scaling with zero init
        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(dtype=x.dtype)

def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    partial_rotary_factor=1.0,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))

    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, rotary_dim, 2, dtype=dtype)[: (rotary_dim // 2)].float() / rotary_dim
        )
    )

    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
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
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        # Qwen3.5 full-attention uses a gated Q projection (2x output dim)
        self.W_query = nn.Linear(d_in, self.d_out * 2, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        q_and_gate = self.W_query(x)
        q_and_gate = q_and_gate.view(b, num_tokens, self.num_heads, self.head_dim * 2)
        queries, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(b, num_tokens, self.d_out)

        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

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
            attn_scores * (self.head_dim ** -0.5),
            dim=-1,
            dtype=torch.float32,
        ).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)

        # Qwen3.5 full-attention uses a gated Q projection
        context = context * torch.sigmoid(gate)
        return self.out_proj(context)


# Just a mapping for the different naming convention in Hugging Face transformers
class _Qwen3_5ConfigAdapter:
    def __init__(self, cfg):
        self.hidden_size = cfg["emb_dim"]
        self.linear_num_value_heads = cfg["linear_num_value_heads"]
        self.linear_num_key_heads = cfg["linear_num_key_heads"]
        self.linear_key_head_dim = cfg["linear_key_head_dim"]
        self.linear_value_head_dim = cfg["linear_value_head_dim"]
        self.linear_conv_kernel_dim = cfg["linear_conv_kernel_dim"]
        self.hidden_act = "silu"
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        self.dtype = cfg.get("dtype", None)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_type, layer_idx):
        super().__init__()
        self.layer_type = layer_type

        if layer_type == "full_attention":
            self.token_mixer = GroupedQueryAttention(
                d_in=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                head_dim=cfg["head_dim"],
                num_kv_groups=cfg["n_kv_groups"],
                qk_norm=cfg["qk_norm"],
                dtype=cfg["dtype"],
            )
        elif layer_type == "linear_attention":
            self.token_mixer = Qwen3_5GatedDeltaNet(_Qwen3_5ConfigAdapter(cfg), layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))

    def forward(self, x, mask, cos, sin):
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


class Qwen3_5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        layer_types = cfg.get("layer_types", ["full_attention"] * cfg["n_layers"])
        if len(layer_types) != cfg["n_layers"]:
            raise ValueError("len(layer_types) must equal n_layers")

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_type, idx) for idx, layer_type in enumerate(layer_types)]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["emb_dim"] // cfg["n_heads"] if cfg["head_dim"] is None else cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)

        num_tokens = x.shape[1]
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

# Qwen3.5-0.8B text configuration
QWEN3_5_CONFIG = {
    "vocab_size": 248_320,
    "context_length": 262_144,
    "emb_dim": 1_024,
    "n_heads": 8,
    "n_layers": 24,
    "hidden_dim": 3_584,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "dtype": torch.bfloat16,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ],
}

# Notebook shims for optional fast kernels in transformers
causal_conv1d_fn = None
causal_conv1d_update = None
chunk_gated_delta_rule = None
fused_recurrent_gated_delta_rule = None
is_fast_path_available = False
FusedRMSNormGated = None
ACT2FN = {"silu": F.silu}

def calc_model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb

def load_weights_into_qwen3_5(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    if "model.embed_tokens.weight" in params:
        model_prefix = "model"
    elif "model.language_model.embed_tokens.weight" in params:
        model_prefix = "model.language_model"
    else:
        raise KeyError("Could not find embed token weights in checkpoint.")

    def pkey(suffix):
        return f"{model_prefix}.{suffix}"

    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params[pkey("embed_tokens.weight")],
        pkey("embed_tokens.weight"),
    )

    n_layers = param_config["n_layers"]
    layer_types = param_config.get("layer_types", ["full_attention"] * n_layers)

    for l in range(n_layers):
        block = model.trf_blocks[l]
        layer_type = layer_types[l]

        if layer_type == "full_attention":
            att = block.token_mixer
            att.W_query.weight = assign(
                att.W_query.weight,
                params[pkey(f"layers.{l}.self_attn.q_proj.weight")],
                pkey(f"layers.{l}.self_attn.q_proj.weight"),
            )
            att.W_key.weight = assign(
                att.W_key.weight,
                params[pkey(f"layers.{l}.self_attn.k_proj.weight")],
                pkey(f"layers.{l}.self_attn.k_proj.weight"),
            )
            att.W_value.weight = assign(
                att.W_value.weight,
                params[pkey(f"layers.{l}.self_attn.v_proj.weight")],
                pkey(f"layers.{l}.self_attn.v_proj.weight"),
            )
            att.out_proj.weight = assign(
                att.out_proj.weight,
                params[pkey(f"layers.{l}.self_attn.o_proj.weight")],
                pkey(f"layers.{l}.self_attn.o_proj.weight"),
            )
            if hasattr(att, "q_norm") and att.q_norm is not None:
                att.q_norm.weight = assign(
                    att.q_norm.weight,
                    params[pkey(f"layers.{l}.self_attn.q_norm.weight")],
                    pkey(f"layers.{l}.self_attn.q_norm.weight"),
                )
            if hasattr(att, "k_norm") and att.k_norm is not None:
                att.k_norm.weight = assign(
                    att.k_norm.weight,
                    params[pkey(f"layers.{l}.self_attn.k_norm.weight")],
                    pkey(f"layers.{l}.self_attn.k_norm.weight"),
                )

        elif layer_type == "linear_attention":
            lat = block.token_mixer
            lat.dt_bias = assign(
                lat.dt_bias,
                params[pkey(f"layers.{l}.linear_attn.dt_bias")],
                pkey(f"layers.{l}.linear_attn.dt_bias"),
            )
            lat.A_log = assign(
                lat.A_log,
                params[pkey(f"layers.{l}.linear_attn.A_log")],
                pkey(f"layers.{l}.linear_attn.A_log"),
            )
            lat.conv1d.weight = assign(
                lat.conv1d.weight,
                params[pkey(f"layers.{l}.linear_attn.conv1d.weight")],
                pkey(f"layers.{l}.linear_attn.conv1d.weight"),
            )
            lat.norm.weight = assign(
                lat.norm.weight,
                params[pkey(f"layers.{l}.linear_attn.norm.weight")],
                pkey(f"layers.{l}.linear_attn.norm.weight"),
            )
            lat.out_proj.weight = assign(
                lat.out_proj.weight,
                params[pkey(f"layers.{l}.linear_attn.out_proj.weight")],
                pkey(f"layers.{l}.linear_attn.out_proj.weight"),
            )
            lat.in_proj_qkv.weight = assign(
                lat.in_proj_qkv.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight"),
            )
            lat.in_proj_z.weight = assign(
                lat.in_proj_z.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_z.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_z.weight"),
            )
            lat.in_proj_b.weight = assign(
                lat.in_proj_b.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_b.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_b.weight"),
            )
            lat.in_proj_a.weight = assign(
                lat.in_proj_a.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_a.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_a.weight"),
            )

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        block.norm1.weight = assign(
            block.norm1.weight,
            params[pkey(f"layers.{l}.input_layernorm.weight")],
            pkey(f"layers.{l}.input_layernorm.weight"),
        )

        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[pkey(f"layers.{l}.mlp.gate_proj.weight")],
            pkey(f"layers.{l}.mlp.gate_proj.weight"),
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[pkey(f"layers.{l}.mlp.up_proj.weight")],
            pkey(f"layers.{l}.mlp.up_proj.weight"),
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[pkey(f"layers.{l}.mlp.down_proj.weight")],
            pkey(f"layers.{l}.mlp.down_proj.weight"),
        )
        block.norm2.weight = assign(
            block.norm2.weight,
            params[pkey(f"layers.{l}.post_attention_layernorm.weight")],
            pkey(f"layers.{l}.post_attention_layernorm.weight"),
        )

    model.final_norm.weight = assign(
        model.final_norm.weight,
        params[pkey("norm.weight")],
        pkey("norm.weight"),
    )

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    elif pkey("lm_head.weight") in params:
        model.out_head.weight = assign(model.out_head.weight, params[pkey("lm_head.weight")], pkey("lm_head.weight"))
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")

class Qwen3_5Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(
        self,
        tokenizer_file_path="tokenizer.json",
        repo_id=None,
        apply_chat_template=True,
        add_generation_prompt=False,
        add_thinking=False,
    ):
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                self._special_to_id[t] = tid

        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        # Mirrors Qwen3.5 chat_template behavior:
        # add_generation_prompt + thinking => "<think>\n"
        # add_generation_prompt + no thinking => empty think scaffold
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant\n"
            if self.add_thinking:
                s += "<think>\n"
            else:
                s += "<think>\n\n</think>\n\n"
        return s

def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                   and torch.all(next_token == eos_token_id)):
               break

            yield next_token
            
            token_ids = torch.cat([token_ids, next_token], dim=1)


if __name__=="__main__":

    import json
    import os


    import time

    from pathlib import Path
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, snapshot_download

    torch.manual_seed(123)
    model = Qwen3_5Model(QWEN3_5_CONFIG)
    print(model)
    print(model(torch.tensor([1, 2, 3]).unsqueeze(0)))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

    print(f"float32 (PyTorch default): {calc_model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {calc_model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    repo_id = "Qwen/Qwen3.5-0.8B"
    local_dir = Path("/scratch/shared/qwen3.5/")

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in sorted(set(index["weight_map"].values())):
        shard_path = os.path.join(repo_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)

    load_weights_into_qwen3_5(model, QWEN3_5_CONFIG, weights_dict)
    model.to(device)
    del weights_dict


    tokenizer_file_path = "Qwen3.5-0.8B/tokenizer.json"

    hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=local_dir,
    )

    tokenizer = Qwen3_5Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )



    prompt = "Give me a short introduction to large language models."

    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)
    print(text)

    prompt = "Give me a short introduction to large language models."

    input_token_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    generated_tokens = 0

    for token in generate_text_basic_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=500,
        eos_token_id=tokenizer.eos_token_id
    ):
        generated_tokens += 1
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n\nGeneration speed: {tokens_per_sec:.2f} tokens/sec")

    if torch.cuda.is_available():
        def calc_gpu_gb(x):
            return f"{x / 1024 / 1024 / 1024:.2f} GB"

        print(f"GPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")

