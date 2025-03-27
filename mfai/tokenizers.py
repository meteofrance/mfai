"""
Module with various LLM tokenizers wrapped in a common interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import sentencepiece as spm
from huggingface_hub import hf_hub_download, login

from mfai.encoding import get_tiktoken_encoding


class Tokenizer(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list, *args: Any, **kwargs: Any) -> str:
        pass

    @property
    @abstractmethod
    def eot_token(self) -> int:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def post_init(self, tokens: set) -> None:
        """
        Do any post init using the full set of tokens
        available in the dataset.
        """


class GPT2Tokenizer(Tokenizer):
    def __init__(self) -> None:
        self.tokenizer = get_tiktoken_encoding("gpt2")

    def name(self) -> str:
        return "gpt2"

    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[int]:
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode(tokens, *args, **kwargs)

    @property
    def eot_token(self) -> int:
        return self.tokenizer.eot_token

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def post_init(self, tokens: set) -> None:
        pass


class LlamaTokenizer(Tokenizer):
    def __init__(self) -> None:
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

    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[int]:
        return self.tokenizer.encode_as_ids(text)

    def decode(self, tokens: list, *args: Any, **kwargs: Any) -> str:
        return self.tokenizer.decode_pieces(tokens)

    @property
    def eot_token(self) -> int:
        return self.tokenizer.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

    def post_init(self, tokens: set) -> None:
        pass


class MiniTokenizer(Tokenizer, ABC):
    """
    A Tokenizer using a reduced set of tokens from a base/parent Tokenizer.
    Typical use case is for narrow vocab problems with only 1000 tokens out
    of a vocab of 50000.
    To use these class, you only have to implement the method 'get_set_tokens'.
    """

    def __init__(self, base_tokenizer: Tokenizer):
        self.base_tokenizer = base_tokenizer
        tokens = self.get_set_tokens()

        self.token_to_id: dict[int, int] = dict()
        self.id_to_token: dict[int, int] = dict()

        self.post_init(tokens)

    @abstractmethod
    def get_set_tokens(self) -> set:
        """
        Method that return a set of tokenized words.

        Example:
            def get_set_tokens(self) -> set:
                unique_tokens = set()
                texts: list[str] = ...  # Load all texts you want to encode
                for text in texts:
                    tokens = self.base_tokenizer.encode(text)
                    unique_tokens.update(tokens)
                return unique_tokens
        """

    def post_init(self, tokens: set) -> None:
        """
        Constructs the forward and backward lookup tables between base tokenizer tokens
        and reduced set of ids.
        """
        for idx, token_id in enumerate(tokens):
            self.token_to_id[token_id] = idx
            self.id_to_token[idx] = token_id

        if self.base_tokenizer.eot_token not in self.token_to_id:
            new_id = len(self.token_to_id)
            self.token_to_id[self.base_tokenizer.eot_token] = new_id
            self.id_to_token[new_id] = self.base_tokenizer.eot_token

    def name(self) -> str:
        return "mini_" + self.base_tokenizer.name()

    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[int]:
        base_token_ids = self.base_tokenizer.encode(text)
        return base_token_ids

    def decode(self, tokens: list, *args: Any, **kwargs: Any) -> str:
        return self.base_tokenizer.decode(tokens)

    @property
    def eot_token(self) -> int:
        return self.base_tokenizer.eot_token

    @property
    def vocab_size(self) -> int:
        return self.base_tokenizer.vocab_size
