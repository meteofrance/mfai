"""
Module with various LLM tokenizers wrapped in a common interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import sentencepiece as spm
import tiktoken  # noqa
from huggingface_hub import hf_hub_download, login

from mfai.encoding import get_tiktoken_encoding


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the tokenizer."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        """Encodes a string into a list of token ids."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: list[int], *args: Any, **kwargs: Any) -> str:
        """Decodes a list of token ids back into a string."""
        raise NotImplementedError

    @property
    @abstractmethod
    def eot_token(self) -> int:
        """Returns the token id for the end-of-text token."""
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""
        raise NotImplementedError


class GPT2Tokenizer(Tokenizer):
    """Tokenizer based on the GPT-2 tokenizer from the tiktoken library."""

    def __init__(self) -> None:
        self.base_tokenizer = get_tiktoken_encoding("gpt2")
        self.special_tokens: list[str] = []
        self.tokenizer = self.base_tokenizer

    def add_special_tokens(self, new_special_tokens: list[str]) -> None:
        """Method to add some special tokens to the tokenizer.

        For more details about extending a tiktoken.Encoding
        https://github.com/openai/tiktoken/tree/main?tab=readme-ov-file#extending-tiktoken
        """
        for tok in new_special_tokens:
            if (
                tok not in self.special_tokens
                and tok not in self.base_tokenizer._special_tokens
            ):
                self.special_tokens.append(tok)

        special_tokens: dict[str, int] = {
            tok: self.base_tokenizer.n_vocab + i
            for i, tok in enumerate(self.special_tokens)
        }

        self.tokenizer = tiktoken.Encoding(
            name=f"custom_{self.name()}",
            pat_str=self.base_tokenizer._pat_str,
            mergeable_ranks=self.base_tokenizer._mergeable_ranks,
            special_tokens={**self.base_tokenizer._special_tokens} | special_tokens,
        )

    def name(self) -> str:
        """Returns the name of the tokenizer."""
        return "gpt2"

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        """Encodes a string into a list of token ids.

        Args:
            text: The input string to encode.
            *args: Additional positional arguments to pass to the tokenizer.
            **kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            list[int]: A list of token ids representing the encoded input string.
        """

        return self.tokenizer.encode(text, allowed_special="all", *args, **kwargs)

    def decode(self, tokens: list[int], *args: Any, **kwargs: Any) -> str:
        """Decodes a list of token ids back into a string.

        Args:
            tokens: A list of token ids to decode.
            *args: Additional positional arguments to pass to the tokenizer.
            **kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            str: The decoded string corresponding to the input list of token ids.
        """

        return self.tokenizer.decode(tokens, *args, **kwargs)

    @property
    def eot_token(self) -> int:
        """Returns the token id for the end-of-text token."""

        return self.tokenizer.eot_token

    @property
    def vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""

        return self.tokenizer.n_vocab


class LlamaTokenizer(Tokenizer):
    """Tokenizer based on the LLaMA tokenizer from the sentencepiece library."""

    def __init__(self) -> None:
        sp = spm.SentencePieceProcessor()

        folderpath = Path(__file__).parent / "Llama-2-7B"
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
        """Returns the name of the tokenizer."""

        return "llama"

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        """Encodes a string into a list of token ids.

        Args:
            text: The input string to encode.
            *args: Additional positional arguments to pass to the tokenizer.
            **kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            list[int]: A list of token ids representing the encoded input string.
        """

        return self.tokenizer.encode_as_ids(text)

    def decode(self, tokens: list, *args: Any, **kwargs: Any) -> str:
        """Decodes a list of token ids back into a string.

        Args:
            tokens: A list of token ids to decode.
            *args: Additional positional arguments to pass to the tokenizer.
            **kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            str: The decoded string corresponding to the input list of token ids.
        """

        return self.tokenizer.decode_pieces(tokens)

    @property
    def eot_token(self) -> int:
        """Returns the token id for the end-of-text token."""

        return self.tokenizer.eos_id()

    @property
    def vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""

        return self.tokenizer.vocab_size()


class MiniGPT2Tokenizer(Tokenizer, ABC):
    """
    A Tokenizer using a reduced set of tokens from a base GPT2Tokenizer.
    Typical use case is for narrow vocab problems with only 1000 tokens out
    of a vocab of 50000.
    To use these class, you only have to implement the method 'get_set_tokens'.
    """

    def __init__(self) -> None:
        self.gpt2_tokenizer = GPT2Tokenizer()

        self.token_to_id: dict[int, int] = dict()
        self.id_to_token: dict[int, int] = dict()

        for idx, token_id in enumerate(self.tokens()):
            self.token_to_id[token_id] = idx
            self.id_to_token[idx] = token_id

        # Add manually the EOT token if needed
        mini_eot_id = self.vocab_size
        base_eot_id = self.gpt2_tokenizer.encode("<|endoftext|>")[0]
        if base_eot_id not in self.token_to_id.keys():
            self.token_to_id[base_eot_id] = mini_eot_id
            self.id_to_token[mini_eot_id] = base_eot_id

    @abstractmethod
    def tokens(self) -> set[int]:
        """
        Method that return a set of tokenized words.

        Example:
            def tokens(self) -> set:
                unique_tokens = set()
                texts: list[str] = ...  # Load all texts you want to encode
                for text in texts:
                    tokens = self.gpt2_tokenizer.encode(text)
                    unique_tokens.update(tokens)
                return unique_tokens
        """

    def add_special_tokens(self, special_tokens: list[str]) -> None:
        """
        Method to add some special tokens to the tokenizer.

        For more details about extending a tiktoken.Encoding
        https://github.com/openai/tiktoken/tree/main?tab=readme-ov-file#extending-tiktoken
        """
        self.gpt2_tokenizer.add_special_tokens(special_tokens)

        vocab_size = self.vocab_size
        for i, special_token in enumerate(self.gpt2_tokenizer.special_tokens):
            mini_tok_id = vocab_size + i
            base_tok_id = self.gpt2_tokenizer.encode(special_token)[0]
            if base_tok_id in self.token_to_id.keys():
                vocab_size -= 1
            else:
                self.token_to_id[base_tok_id] = mini_tok_id
                self.id_to_token[mini_tok_id] = base_tok_id

    def name(self) -> str:
        """Returns the name of the tokenizer."""
        return "mini_" + self.gpt2_tokenizer.name()

    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        """Encodes a string into a list of token ids.

        Args:
            text: The input string to encode.
            *args: Additional positional arguments to pass to the tokenizer.
            **kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            list[int]: A list of token ids representing the encoded input string.
        """

        base_token_ids = self.gpt2_tokenizer.encode(text)
        return [self.token_to_id[x] for x in base_token_ids]

    def decode(self, tokens: list, *args: Any, **kwargs: Any) -> str:
        """Decodes a list of token ids back into a string.

        Args:
            tokens: A list of token ids to decode.
            *args: Additional positional arguments to pass to the tokenizer.
            **kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            str: The decoded string corresponding to the input list of token ids.
        """

        base_tokens = [self.id_to_token[x] for x in tokens]
        return self.gpt2_tokenizer.decode(base_tokens)

    @property
    def eot_token(self) -> int:
        """Returns the token id for the end-of-text token."""
        return self.token_to_id[self.gpt2_tokenizer.eot_token]

    @property
    def vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""
        return len(self.token_to_id)
