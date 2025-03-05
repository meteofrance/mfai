"""Wrapper of tiktoken.get_encoding to retrieve tokenizer files from disk.
Usefull for machines which are behind a proxy."""

import os
from pathlib import Path

# Define the cache dir for tiktoken lib
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

    # First we want to check that the encoding files are in the cache dir
    # but the desired files have specific hashed names that we need to retrieve

    # 1. Inspect the request that is made to openai through tiktoken
    blobpath = inspect.getsource(available_tokenizers[tokenizer_name])

    # 2. Extract the url of the desired vocab file
    vocab_bpe_url = blobpath.split('"')[1]

    # 3. Compute the hash name of this file
    cache_key_vocab = hashlib.sha1(vocab_bpe_url.encode()).hexdigest()

    # 4. Define the path where the hashed file should be stored
    vocab_path = tiktoken_cache_dir / cache_key_vocab

    # 5. Same steps for the encoder file
    encoder_json_url = blobpath.split('"')[3]
    cache_key_encoder = hashlib.sha1(encoder_json_url.encode()).hexdigest()
    encoder_path = tiktoken_cache_dir / cache_key_encoder

    # 6. Check that we have all the needed files
    if not vocab_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"Missing some files for the {tokenizer_name} tokenizer: {vocab_path} and {encoder_path}"
        )

    # 7. Load the encoding through the cache dir
    return tiktoken.get_encoding(tokenizer_name)
