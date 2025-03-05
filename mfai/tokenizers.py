"""
Module with various LLM tokenizers wrapped in a common interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import sentencepiece as spm
import tiktoken
import torch
from huggingface_hub import hf_hub_download, login

from mfai.encoding import get_tiktoken_encoding


class Tokenizer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def encode(self, text: str, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, tokens: list, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def eot_token(self) -> int:
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def post_init(self, tokens: set):
        """
        Do any post init using the full set of tokens
        available in the dataset.
        """


class GPT2Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = get_tiktoken_encoding("gpt2")

    def name(self) -> str:
        return "gpt2"

    def encode(self, text: str, *args, **kwargs) -> torch.Tensor:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list, *args, **kwargs) -> str:
        return self.tokenizer.decode(tokens, *args, **kwargs)

    @property
    def eot_token(self) -> int:
        return self.tokenizer.eot_token

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def post_init(self, tokens: set):
        pass


class LlamaTokenizer(Tokenizer):
    def __init__(self):
        sp = spm.SentencePieceProcessor()

        folderpath = Path(__file__).parent / "tokenizer" / "Llama-2-7B"

        if not folderpath.exists():
            login()
            tokenizer_file = hf_hub_download(
                repo_id="meta-llama/Llama-2-7b",
                filename="tokenizer.model",
                local_dir=folderpath,
            )
        else:
            tokenizer_file = str(folderpath / "tokenizer.model")

        sp.load(tokenizer_file)
        self.tokenizer = sp

    def name(self) -> str:
        return "llama"
    
    def encode(self, text: str, *args, **kwargs) -> torch.Tensor:
        return self.tokenizer.encode_as_ids(text)

    def decode(self, tokens: list, *args, **kwargs) -> str:
        return self.tokenizer.decode_pieces(tokens)

    @property
    def eot_token(self) -> int:
        return self.tokenizer.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

    def post_init(self, tokens: set):
        pass


class MiniTokenizer(Tokenizer):
    """
    A Tokenizer using a reduced set of tokens
    from a base/parent Tokenizer. Typical use case is for
    narrow vocab problems with only 1000 tokens out of a vocab of 50000.
    """

    def __init__(self, base_tokenizer: Tokenizer):
        self.base_tokenizer = base_tokenizer

        # at this stage the lookup table are not initialised
        self.token_to_id = None
        self.id_to_token = None

    def post_init(self, tokens: set):
        """
        Constructs the forward and backward lookup tables between base tokenizer tokens
        and reduced set of ids.
        """
        self.token_to_id = dict()
        self.id_to_token = dict()

        for idx, token_id in enumerate(tokens):
            self.token_to_id[token_id] = idx
            self.id_to_token[idx] = token_id

        if self.base_tokenizer.eot_token not in self.token_to_id:
            new_id = len(self.token_to_id)
            self.token_to_id[self.base_tokenizer.eot_token] = new_id
            self.id_to_token[new_id] = self.base_tokenizer.eot_token

    def encode(self, text: str, *args, **kwargs) -> torch.Tensor:
        base_token_ids = self.base_tokenizer.encode(text)
        if self.token_to_id is not None:
            return [self.token_to_id[x] for x in base_token_ids]
        else:
            return base_token_ids

    def decode(self, tokens: list, *args, **kwargs) -> str:
        if self.id_to_token is not None:
            base_tokens = [self.id_to_token[x] for x in tokens]
            return self.base_tokenizer.decode(base_tokens)
        else:
            return self.base_tokenizer.decode(tokens)

    @property
    def eot_token(self) -> int:
        if self.token_to_id is not None:
            return self.token_to_id[self.base_tokenizer.eot_token]
        else:
            return self.base_tokenizer.eot_token

    @property
    def vocab_size(self) -> int:
        if self.token_to_id is not None:
            return len(self.token_to_id)
        else:
            return self.base_tokenizer.vocab_size
