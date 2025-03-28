import pytest

from mfai.tokenizers import MiniTokenizer, Tokenizer, GPT2Tokenizer, LlamaTokenizer

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."


class LoremMiniTokenizer(MiniTokenizer):
    def get_set_tokens(self) -> set:
        unique_tokens = set()
        tokens = self.base_tokenizer.encode(LOREM_IPSUM)
        unique_tokens.update(tokens)
        return unique_tokens


@pytest.mark.parametrize(
    "base_tokenizer, expected_num_tokens", [(GPT2Tokenizer(), 40), (LlamaTokenizer(), 38)],
)
def test_mini_tokenizer(base_tokenizer: Tokenizer, expected_num_tokens: int):
    tokenizer = LoremMiniTokenizer(base_tokenizer)
    assert tokenizer.vocab_size == expected_num_tokens
