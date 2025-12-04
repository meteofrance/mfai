"""Pytorch implementation of GPT-2.
It is widely inspired by Sebastian Raschka's book and work
https://github.com/rasbt/LLMs-from-scratch/
"""

import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from dataclasses_json import dataclass_json
from torch import Tensor, nn

from mfai.pytorch import assign
from mfai.pytorch.models.base import ModelType
from mfai.tensorflow import download_and_load_gpt2


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


class MultiHeadAttention(nn.Module):
    """
    MultiHead Attention compatible with tensorflow original implementation and weigths.
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
        """
        Constructor of the multihead attention model with the following parameters:
        d_in: dimension of the input tensor (Batch, num_tokens, d_in)
        d_out: dimension of the output tensor (Batch, num_tokens, d_out)
        qkv_bias: If True, adds a bias learnable parameters to the query, key and value Linear Layers.

        See book and repo for further explanation: https://github.com/rasbt/LLMs-from-scratch
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore[operator]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


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


class MultiHeadCrossAttentionPySDPA(nn.Module):
    """
    Mutli Head Cross Attention using Pytorch's scaled_dot_product_attention
    The query and key/values are from different sources.
    """

    def __init__(
        self,
        d_in_q: int,
        d_in_kv: int,
        d_out: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError("embed_dim is indivisible by num_heads")

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.q = nn.Linear(d_in_q, d_out, bias=qkv_bias)
        self.kv = nn.Linear(d_in_kv, 2 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        batch_size, num_tokens_q, _ = x_q.shape
        batch_size, num_tokens_kv, _ = x_kv.shape

        # (b, num_tokens_q, embed_dim) --> (b, num_tokens, d_out)
        q = self.q(x_q)
        q = q.view(batch_size, num_tokens_q, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (b, num_heads, num_tokens_q, head_dim)

        # (b, num_tokens_kv, embed_dim) --> (b, num_tokens, 2 * d_out)
        kv = self.kv(x_kv)

        # (b, num_tokens_kv, 2 * d_out) --> (b, num_tokens_kv, 2, num_heads, head_dim)
        kv = kv.view(batch_size, num_tokens_kv, 2, self.num_heads, self.head_dim)

        # (b, num_tokens_kv, 2, num_heads, head_dim) --> (2, b, num_heads, num_tokens_kv, head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)

        # (2, b, num_heads, num_tokens_kv, head_dim) -> 2 times (b, num_heads, num_tokens_kv, head_dim)
        keys, values = kv

        use_dropout = 0.0 if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=None,
            dropout_p=use_dropout,
        )

        # Combine heads
        context_vec = (
            context_vec.transpose(1, -1)
            .contiguous()
            .view(batch_size, num_tokens_q, self.d_out)
        )

        context_vec = self.proj(context_vec)

        return context_vec


@dataclass_json
@dataclass(slots=True)
class GPT2Settings:
    """default settings correspond to a GPT2 small '124M'"""

    emb_dim: int = 768  # Embedding dimension
    context_length: int = 1024  # Context length
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias
    model_size: Literal["124M", "355M", "774M", "1558M"] = "124M"  # Alias used to download official weights
    attn_tf_compat: bool = False  # If true, uses a less GPU efficient implementation of attn compatible with official weights


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
        if settings.attn_tf_compat:
            self.att: MultiHeadAttention | MultiHeadAttentionPySDPA = (
                MultiHeadAttention(
                    d_in=settings.emb_dim,
                    d_out=settings.emb_dim,
                    context_length=settings.context_length,
                    num_heads=settings.n_heads,
                    dropout=settings.drop_rate,
                    qkv_bias=True,
                )
            )
        else:
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
        self.model_size = settings.model_size

    @typing.no_type_check
    def load_weights_from_dict(self, params: dict):
        """
        Loads weights into self using a dict
        likely coming from a tensorflow or other framework
        training. Use this to finetune from the official weights.
        """
        self.pos_emb.weight = assign(self.pos_emb.weight, params["wpe"])

        # we allow for adding special tokens
        if self.tok_emb.weight.shape[0] > len(params["wte"]):
            self.tok_emb.weight = torch.nn.Parameter(
                self.tok_emb.weight.index_put(
                    (torch.LongTensor(range(len(params["wte"]))),),
                    torch.tensor(params["wte"]),
                )
            )
        else:
            self.tok_emb.weight = assign(self.tok_emb.weight, params["wte"])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
            )
            self.trf_blocks[b].att.W_query.weight = assign(
                self.trf_blocks[b].att.W_query.weight, q_w.T
            )
            self.trf_blocks[b].att.W_key.weight = assign(
                self.trf_blocks[b].att.W_key.weight, k_w.T
            )
            self.trf_blocks[b].att.W_value.weight = assign(
                self.trf_blocks[b].att.W_value.weight, v_w.T
            )

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
            )
            self.trf_blocks[b].att.W_query.bias = assign(
                self.trf_blocks[b].att.W_query.bias, q_b
            )
            self.trf_blocks[b].att.W_key.bias = assign(
                self.trf_blocks[b].att.W_key.bias, k_b
            )
            self.trf_blocks[b].att.W_value.bias = assign(
                self.trf_blocks[b].att.W_value.bias, v_b
            )

            self.trf_blocks[b].att.out_proj.weight = assign(
                self.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T,
            )
            self.trf_blocks[b].att.out_proj.bias = assign(
                self.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"],
            )

            self.trf_blocks[b].ff.layers[0].weight = assign(
                self.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T,
            )
            self.trf_blocks[b].ff.layers[0].bias = assign(
                self.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"],
            )
            self.trf_blocks[b].ff.layers[2].weight = assign(
                self.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T,
            )
            self.trf_blocks[b].ff.layers[2].bias = assign(
                self.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"],
            )

            self.trf_blocks[b].norm1.scale = assign(
                self.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
            )
            self.trf_blocks[b].norm1.shift = assign(
                self.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
            )
            self.trf_blocks[b].norm2.scale = assign(
                self.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
            )
            self.trf_blocks[b].norm2.shift = assign(
                self.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
            )

        self.final_norm.scale = assign(self.final_norm.scale, params["g"])
        self.final_norm.shift = assign(self.final_norm.shift, params["b"])

        # same here we allow for extra tokens
        if self.out_head.weight.shape[0] > len(params["wte"]):
            self.out_head.weight = torch.nn.Parameter(
                self.out_head.weight.index_put(
                    (torch.LongTensor(range(len(params["wte"]))),),
                    torch.tensor(params["wte"]),
                )
            )
        else:
            self.out_head.weight = assign(self.out_head.weight, params["wte"])

    def dowload_weights_from_tf_ckpt(self, model_dir: str | Path) -> None:
        """
        Downloads a tensorflow checkpoint into model_dir and sets the weights of self.
        """
        _, params = download_and_load_gpt2(self.model_size, model_dir)
        self.load_weights_from_dict(params)

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


@dataclass_json
@dataclass(slots=True)
class CrossAttentionGPT2Settings(GPT2Settings):
    x_att_ratio: int = 4  # Ratio of cross attention blocks, default one out of 4


class CrossAttentionTransformerBlock(nn.Module):
    """
    A cross attention transformer block
    """

    def __init__(self, settings: CrossAttentionGPT2Settings) -> None:
        super().__init__()
        self.x_att = MultiHeadCrossAttentionPySDPA(
            d_in_q=settings.emb_dim,
            d_in_kv=settings.emb_dim,
            d_out=settings.emb_dim,
            context_length=settings.context_length,
            num_heads=settings.n_heads,
            dropout=settings.drop_rate,
            qkv_bias=settings.qkv_bias,
        )
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

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        # Shortcut connection for attention block
        shortcut = x_q
        x_q = self.norm1(x_q)

        x_q = self.att(x_q)
        x_q += shortcut
        shortcut = x_q

        x = self.x_att(x_q, x_kv)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class CrossAttentionGPT2(nn.Module):
    """
    A GPT2 with cross attention to allow vision/weather data injection as key/values into some of the transformer block.
    Freely inspired by Llama3.2 as described here : https://magazine.sebastianraschka.com/i/151078631/the-llama-herd-of-models
    """

    settings_kls = CrossAttentionGPT2Settings
    model_type = ModelType.LLM

    def __init__(
        self, settings: CrossAttentionGPT2Settings, vocab_size: int = 50257
    ) -> None:
        super().__init__()
        self.context_length = settings.context_length
        self.emb_dim = settings.emb_dim
        self.tok_emb = nn.Embedding(vocab_size, settings.emb_dim)
        self.pos_emb = nn.Embedding(settings.context_length, settings.emb_dim)
        self.drop_emb = nn.Dropout(settings.drop_rate)

        # Build the transformer blocks, every nth block includes a cross attention block
        trf_blocks: list[TransformerBlock | CrossAttentionTransformerBlock] = []
        for i in range(settings.n_layers):
            if i % settings.x_att_ratio == 0:
                trf_blocks.append(CrossAttentionTransformerBlock(settings))
            else:
                trf_blocks.append(TransformerBlock(settings))
        self.trf_blocks = nn.Sequential(*trf_blocks)
        self.final_norm = LayerNorm(settings.emb_dim)
        self.out_head = nn.Linear(settings.emb_dim, vocab_size, bias=False)

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

    def forward(self, token_ids: Tensor, vision_inputs: Tensor) -> Tensor:
        # token_ids shape=(B, n_tok), vision_input shape=(B, n'_tok * time, embed_dim)
        token_ids = token_ids[
            :, -self.context_length :
        ]  # Keep only the last context_length tokens
        x = self.embed_tokens(token_ids)  # (B, n_tok, embed_dim)
        x = self.drop_emb(x)  # (B, n_tok, embed_dim)

        for b in self.trf_blocks:
            if isinstance(b, CrossAttentionTransformerBlock):
                # If the block is a cross attention block, we pass the vision inputs
                x = b(x, vision_inputs)
            else:
                x = b(x)
        x = self.final_norm(x)  # (B, n_tok, embed_dim)
        logits = self.out_head(x)  # (B, n_tok, vocab_size)
        return logits
