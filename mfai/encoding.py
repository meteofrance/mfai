"""Wrapper of tiktoken.get_encoding to retrieve tokenizer files from disk.
Usefull for machines that don't have access to the internet."""

import os
from pathlib import Path

tiktoken_cache_dir = Path(__file__).parent / "tokenizer" / "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

import tiktoken_ext.openai_public  # noqa
import inspect  # noqa
import hashlib  # noqa
import tiktoken  # noqa

available_tokenizers = {"gpt2": tiktoken_ext.openai_public.gpt2}


def get_tiktoken_encoding(tokenizer_name: str) -> tiktoken.Encoding:
    """Wrapper of tiktoken.get_encoding to retrieve tokenizer files from disk."""
    if tokenizer_name not in available_tokenizers:
        raise ValueError(
            f"Tokenizer {tokenizer_name} is not available. {available_tokenizers} are available."
        )
    blobpath = inspect.getsource(available_tokenizers[tokenizer_name])

    vocab_bpe_url = blobpath.split('"')[1]
    cache_key_vocab = hashlib.sha1(vocab_bpe_url.encode()).hexdigest()
    vocab_path = tiktoken_cache_dir / cache_key_vocab

    encoder_json_url = blobpath.split('"')[3]
    cache_key_encoder = hashlib.sha1(encoder_json_url.encode()).hexdigest()
    encoder_path = tiktoken_cache_dir / cache_key_encoder

    if not vocab_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"Missing some files for the {tokenizer_name} tokenizer: {vocab_path} and {encoder_path}"
        )

    return tiktoken.get_encoding(tokenizer_name)
