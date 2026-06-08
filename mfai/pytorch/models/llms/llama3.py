"""
Llama3 standalone implementation inspired from
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
Explanations on grouped query attention: https://www.ibm.com/think/topics/grouped-query-attention.
"""

from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
from torch import Tensor, nn

from mfai.pytorch.models.base import ModelType
from mfai.pytorch.models.llms.llama2 import FeedForwardLlama2


def compute_rope_params(
    head_dim: int, theta_base: float = 10_000.0, context_length: int = 4096
) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor:
    # x: (batch_size, num_heads, seq_len, head_dim)
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = (
        cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    )  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int, num_kv_groups: int):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        cos: Tensor,
        sin: Tensor,
        start_pos: int = 0,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Performs the forward pass of grouped query attention.

        This implements multi-head attention with grouped query heads, where multiple
        query heads share the same key-value pairs. This reduces memory usage and
        computation compared to standard multi-head attention while maintaining
        performance.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in)
            mask (Tensor): Attention mask of shape (batch_size, 1, num_tokens, num_tokens)
                Used to prevent attention to future tokens in causal masking
            cos (Tensor): Cosine values for RoPE positional encoding
            sin (Tensor): Sine values for RoPE positional encoding
            start_pos (int, optional): Starting position for RoPE positional encoding. Defaults to 0.
            cache (tuple[Tensor, Tensor], optional): Cached key-value tensors from previous iterations. Defaults to None.

        Returns:
            tuple[Tensor, tuple[Tensor, Tensor]]: A tuple containing:
                - context_vec (Tensor): Output tensor of shape (batch_size, num_tokens, d_out)
                - next_cache (tuple[Tensor, Tensor]): Updated cache tensors for next iteration (keys_cache, values_cache).
        """
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_kv_groups, num_tokens, head_dim)
        values = values.transpose(
            1, 2
        )  # Shape: (b, num_kv_groups, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)

        # Apply RoPE
        keys = apply_rope(keys, cos, sin, offset=start_pos)
        queries = apply_rope(queries, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys], dim=2)
            values = torch.cat([prev_v, values], dim=2)
        next_cache: tuple[Tensor, Tensor] = (keys, values)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(
            self.group_size, dim=1
        )  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(
            self.group_size, dim=1
        )  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Compute attention scores
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec, next_cache


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_kv_groups: int,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
        )
        self.ff = FeedForwardLlama2(emb_dim, hidden_dim, dtype)
        self.norm1 = nn.RMSNorm(emb_dim, eps=1e-5, dtype=dtype)
        self.norm2 = nn.RMSNorm(emb_dim, eps=1e-5, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        cos: Tensor,
        sin: Tensor,
        start_pos: int,
        cache: tuple[Tensor | Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(
            x, mask, cos, sin, start_pos=start_pos, cache=cache
        )  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache


@dataclass_json
@dataclass(slots=True)
class Llama3Settings:
    emb_dim: int = 256  # Embedding dimension
    context_length: int = 512  # Context length
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 8  # Number of layers
    hidden_dim: int = 768  # Size of the intermediate dimension in FeedForward
    num_kv_groups: int = 2  # number of kv groups in grouped-query attention
    rope_base: float = 500_000.0


class Llama3(nn.Module):
    settings_kls = Llama3Settings
    model_type = ModelType.LLM

    def __init__(self, settings: Llama3Settings, vocab_size: int = 32000) -> None:
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(vocab_size, settings.emb_dim)

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [
                TransformerBlock(
                    emb_dim=settings.emb_dim,
                    hidden_dim=settings.hidden_dim,
                    num_heads=settings.n_heads,
                    num_kv_groups=settings.num_kv_groups,
                )
                for _ in range(settings.n_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(settings.emb_dim, eps=1e-5)
        self.out_head = nn.Linear(settings.emb_dim, vocab_size, bias=False)

        # Reusable utilities
        cos, sin = compute_rope_params(
            head_dim=settings.emb_dim // settings.n_heads,
            theta_base=settings.rope_base,
            context_length=settings.context_length,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.settings = settings
        self.context_length = settings.context_length

        # To use KV cache
        self.current_pos = 0  # Track current position in KV cache
        self.cache: list[None | tuple[Tensor, Tensor]] = [None] * settings.n_layers

    def embed_tokens(self, tok_ids: Tensor) -> Tensor:
        return self.tok_emb(tok_ids)

    def forward_vectors(
        self,
        embeddings: Tensor,
        first_embedding: None | Tensor = None,
        use_cache: bool = False,
    ) -> Tensor:
        """
        Process a batch of embeddings through the model.
        If first_embedding is supplied the first tokens of each blocks are replaced
        by the corresponding embeddings. Useful for multimodal models with injection of vision data
        embeddings at each stage.
        """

        _, num_tokens, _ = embeddings.shape

        if use_cache:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            mask = torch.triu(
                torch.ones(
                    pos_end, pos_end, device=embeddings.device, dtype=torch.bool
                ),
                diagonal=1,
            )[pos_start:pos_end, :pos_end]
            self.current_pos = pos_end
        else:
            pos_start = 0
            mask = torch.triu(
                torch.ones(
                    num_tokens, num_tokens, device=embeddings.device, dtype=torch.bool
                ),
                diagonal=1,
            )

        for i, block in enumerate(self.trf_blocks):
            if first_embedding is not None:
                # replace the first token of x by the corresponding first_embedding
                embeddings = torch.cat(
                    [first_embedding, embeddings[:, first_embedding.shape[1] :, :]],
                    dim=1,
                )
            blk_cache: None | tuple[Tensor, Tensor] = self.cache[i]
            x, new_blk_cache = block(
                embeddings,
                mask,
                self.cos,
                self.sin,
                start_pos=pos_start,
                cache=blk_cache,
            )
            if use_cache:
                self.cache[i] = new_blk_cache

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def forward(self, tok_ids: Tensor, use_cache: bool = False) -> Tensor:
        """Performs the forward pass of the LLama3 model.

        Args:
            tok_ids (Tensor): Tensor of token IDs input with shape (batch_size, sequence_length).
            use_cache (bool, optional): If True, uses attention caching for computational
                optimization. Defaults to False.

        Returns:
            Tensor: Model output tensor with shape (batch_size, sequence_length, hidden_size) or
                (batch_size, context_length, hidden_size) depending on model dimensions.

        Note:
            The KV cache implementation is largely inspired by S. Rashka. See
            https://github.com/rasbt/LLMs-from-scratch/blob/main/pkg/llms_from_scratch/llama3.py for more details.

        """
        return self.forward_vectors(self.embed_tokens(tok_ids), use_cache=use_cache)

    def reset_kv_cache(self) -> None:
        """Clear the Key-Value cache used for incremental decoding.

        This method must be called between processing independent sequences to
        prevent cross-sequence contamination in autoregressive generation. After
        calling this method, the cache will be reset to None and ready for a new
        sequence.
        """
        self.current_pos = 0
        self.cache = [None] * self.settings.n_layers
