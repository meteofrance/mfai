"""
This a pytorch implementation of GPT-2 and Llama2 models.
It is widely inspired by Sebastian Raschka's book and work
https://github.com/rasbt/LLMs-from-scratch/
"""

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor
from dataclasses_json import dataclass_json
from torch import nn

from mfai.torch.models.base import ModelType

##########################################################################################################
#######################################         GPT2           ###########################################
##########################################################################################################


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MultiHeadAttentionPySDPA(nn.Module):
    """
    Mutli Head Attention using Pytorch's scaled_dot_product_attention
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if d_out % num_heads != 0:
            raise Exception("ERROR: embed_dim is indivisible by num_heads")

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0.0 if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )

        context_vec = self.proj(context_vec)

        return context_vec


@dataclass_json
@dataclass(slots=True)
class GPT2Settings:
    """default settings correspond to a GPT2 small"""

    emb_dim: int = 768  # Embedding dimension
    context_length: int = 1024  # Context length
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias


class TransformerBlock(nn.Module):
    """A transformer block
    - Based on Sebastian Raschka's book and github repo : https://github.com/rasbt/LLMs-from-scratch/

    - Attention used is based on pytorch's scaled_dot_product_attention
    ( Most efficient MultiHeadAttention module accodring S.Raschka's benchmark
    https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention
    )

    """

    def __init__(self, settings: GPT2Settings) -> None:
        super().__init__()
        self.att = MultiHeadAttentionPySDPA(
            d_in=settings.emb_dim,
            d_out=settings.emb_dim,
            context_length=settings.context_length,
            num_heads=settings.n_heads,
            dropout=settings.drop_rate,
            qkv_bias=settings.qkv_bias,
        )
        self.ff = FeedForward(settings.emb_dim)
        self.norm1 = LayerNorm(settings.emb_dim)
        self.norm2 = LayerNorm(settings.emb_dim)
        self.drop_shortcut = nn.Dropout(settings.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPT2(nn.Module):
    """GPT implementation
    - Based on Sebastian Raschka's book and github repo :
        https://github.com/rasbt/LLMs-from-scratch/

    """

    settings_kls = GPT2Settings
    model_type = ModelType.LLM

    def __init__(self, settings: GPT2Settings, vocab_size: int = 50257) -> None:
        super().__init__()
        self.context_length = settings.context_length
        self.emb_dim = settings.emb_dim
        self.tok_emb = nn.Embedding(vocab_size, settings.emb_dim)
        self.pos_emb = nn.Embedding(settings.context_length, settings.emb_dim)
        self.drop_emb = nn.Dropout(settings.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(settings) for _ in range(settings.n_layers)]
        )

        self.final_norm = LayerNorm(settings.emb_dim)
        self.out_head = nn.Linear(settings.emb_dim, vocab_size, bias=False)

    def forward_vectors(
        self, embeddings: Tensor, first_embedding: Union[None, Tensor] = None
    ) -> Tensor:
        """
        Process a batch of embeddings through the model.
        If first_embedding is supplied the first tokens of each blocks are replaced
        by the corresponding embeddings. Useful for multimodal models with injection of vision data
        at each stage.
        """

        x = self.drop_emb(embeddings)

        if first_embedding is not None:
            for block in self.trf_blocks:
                # replace the first token of x by the corresponding first_embedding
                x = torch.cat(
                    [first_embedding, x[:, first_embedding.shape[1] :, :]], dim=1
                )
                x = block(x)
        else:
            x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def embed_tokens(self, tok_ids: Tensor) -> Tensor:
        """
        Embeds and pos encodes tokens.
        """
        if tok_ids.shape[1] > self.context_length:
            raise ValueError(
                f"The tokens shape ({tok_ids.shape[1]}) should be less than or equal to 'context_length' ({self.context_length})."
            )
        _, seq_len = tok_ids.shape
        tok_embeds = self.tok_emb(tok_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=tok_ids.device))
        return tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]

    def forward(self, tok_ids: Tensor) -> Tensor:
        tok_ids = tok_ids[
            :, -self.context_length :
        ]  # Keep only the last context_length tokens
        x = self.embed_tokens(tok_ids)
        return self.forward_vectors(x)


##########################################################################################################
#######################################         llama2           #########################################
##########################################################################################################


class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x: Tensor) -> Tensor:
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class FeedForwardLlama2(nn.Module):
    def __init__(
        self, emb_dim: int, hidden_dim: int, dtype: torch.dtype = None
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)


def precompute_rope_params(
    head_dim: int, theta_base: int = 10_000, context_length: int = 4096
) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (batch_size, num_heads, seq_len, head_dim)
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class MultiHeadAttentionPySDPALlama2(nn.Module):
    """
    Mutli Head Attention using Pytorch's scaled_dot_product_attention
    """

    cos: Tensor
    sin: Tensor

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        context_length: int,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=False, dtype=dtype)
        self.proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        cos, sin = precompute_rope_params(
            head_dim=self.head_dim, context_length=context_length
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )

        context_vec = self.proj(context_vec)

        return context_vec


@dataclass_json
@dataclass(slots=True)
class Llama2Settings:
    emb_dim: int = 256  # Embedding dimension
    context_length: int = 512  # Context length
    n_heads: int = 4  # Number of attention heads
    n_layers: int = 4  # Number of layers
    hidden_dim: int = 768  # Size of the intermediate dimension in FeedForward


class TransformerBlockLlama2(nn.Module):
    """A transformer block
    - Based on Sebastian Raschka's book and github repo : https://github.com/rasbt/LLMs-from-scratch/

    - Attention used is based on pytorch's scaled_dot_product_attention
    ( Most efficient MultiHeadAttention module accodring S.Raschka's benchmark
    https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/02_bonus_efficient-multihead-attention
    )

    """

    def __init__(self, settings: Llama2Settings) -> None:
        super().__init__()
        self.att = MultiHeadAttentionPySDPALlama2(
            d_in=settings.emb_dim,
            d_out=settings.emb_dim,
            context_length=settings.context_length,
            num_heads=settings.n_heads,
        )
        self.ff = FeedForwardLlama2(settings.emb_dim, settings.hidden_dim)
        self.norm1 = RMSNorm(settings.emb_dim)
        self.norm2 = RMSNorm(settings.emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class Llama2(nn.Module):
    """Llama2 implementation
    - Based on Sebastian Raschka's book and github repo :
        https://github.com/rasbt/LLMs-from-scratch/

    """

    settings_kls = Llama2Settings
    model_type = ModelType.LLM

    def __init__(self, settings: Llama2Settings, vocab_size: int = 32000) -> None:
        super().__init__()
        self.emb_dim = settings.emb_dim
        self.tok_emb = nn.Embedding(vocab_size, self.emb_dim)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlockLlama2(settings) for _ in range(settings.n_layers)]
        )

        self.final_norm = RMSNorm(self.emb_dim)
        self.out_head = nn.Linear(self.emb_dim, vocab_size, bias=False)
        self.context_length = settings.context_length

    def embed_tokens(self, tok_ids: Tensor) -> Tensor:
        return self.tok_emb(tok_ids)

    def forward_vectors(
        self, embeddings: Tensor, first_embedding: Union[None, Tensor] = None
    ) -> Tensor:
        """
        Process a batch of embeddings through the model.
        If first_embedding is supplied the first tokens of each blocks are replaced
        by the corresponding embeddings. Useful for multimodal models with injection of vision data
        at each stage.
        """

        x = embeddings
        if first_embedding is not None:
            for block in self.trf_blocks:
                # replace the first token of x by the corresponding first_embedding
                embeddings = torch.cat(
                    [first_embedding, x[:, first_embedding.shape[1] :, :]], dim=1
                )
                x = block(x)
        else:
            x = self.trf_blocks(embeddings)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def forward(self, tok_ids: Tensor) -> Tensor:
        return self.forward_vectors(self.embed_tokens(tok_ids))
