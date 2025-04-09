import pytest

from mfai.tokenizers import MiniTokenizer, Tokenizer, GPT2Tokenizer, LlamaTokenizer

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
LOREM_IPSUM_SPECIAL_TOKENS = "<|Lorem ipsum|> dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."


##########################################################################################################
###################################         Mini Tokenizer           #####################################
##########################################################################################################
class LoremMiniTokenizer(MiniTokenizer):
    def tokens(self) -> set:
        unique_tokens = set()
        tokens = self.base_tokenizer.encode(LOREM_IPSUM)
        unique_tokens.update(tokens)
        return unique_tokens


@pytest.mark.parametrize(
    "base_tokenizer, expected_vocab_size",
    [(GPT2Tokenizer(), 40), (LlamaTokenizer(), 38)],
)
def test_mini_tokenizer(base_tokenizer: Tokenizer, expected_vocab_size: int):
    tokenizer = LoremMiniTokenizer(base_tokenizer)
    assert tokenizer.vocab_size == expected_vocab_size


##########################################################################################################
###################################         Special Tokens           #####################################
##########################################################################################################
@pytest.mark.parametrize("base_tokenizer", [GPT2Tokenizer])
def test_add_special_tokens(base_tokenizer: Tokenizer):
    tokenizer = base_tokenizer(special_tokens=set(["<|Lorem ipsum|>"]))
    assert tokenizer.vocab_size == base_tokenizer().vocab_size + 1

    text_encoded = tokenizer.encode(LOREM_IPSUM_SPECIAL_TOKENS)
    assert text_encoded[0] == base_tokenizer().vocab_size
