import copy

import pytest

from mfai.tokenizers import GPT2Tokenizer, MiniGPT2Tokenizer, Tokenizer

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
LOREM_IPSUM_SPECIAL_TOKENS = "<|Lorem ipsum|> dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.<|endoftext|>"


##########################################################################################################
#################################         MiniGPT2Tokenizer           ####################################
##########################################################################################################
class LoremMiniTokenizer(MiniGPT2Tokenizer):
    def tokens(self) -> set:
        unique_tokens = set()
        tokens = self.gpt2_tokenizer.encode(LOREM_IPSUM)
        unique_tokens.update(tokens)
        return unique_tokens


def test_mini_tokenizer():
    tokenizer = LoremMiniTokenizer()
    assert tokenizer.vocab_size == 40


##########################################################################################################
###################################         Special Tokens           #####################################
##########################################################################################################
@pytest.mark.parametrize("tokenizer", [GPT2Tokenizer(), LoremMiniTokenizer()])
def test_add_special_tokens(tokenizer: Tokenizer):
    base_tokenizer = copy.deepcopy(tokenizer)

    # Check that the vocab_size is increased by 1 because we add 1 new special token
    tokenizer.add_special_tokens(["<|Lorem ipsum|>"])
    assert tokenizer.vocab_size == base_tokenizer.vocab_size + 1
    # The <|endoftext|> should not be consider as new token, so the vocab_size
    # should stay the same
    tokenizer.add_special_tokens(["<|endoftext|>"])
    assert tokenizer.vocab_size == base_tokenizer.vocab_size + 1

    text_encoded: list[int] = tokenizer.encode(LOREM_IPSUM_SPECIAL_TOKENS)
    # Check that the encoding of <|Lorem ipsum|> is last token id because it is the
    # last token added to the tokenizer
    assert text_encoded[0] == tokenizer.vocab_size - 1
    # Check that the encoding of <|endoftext|> is unchanged after adding some new token
    assert text_encoded[-1] == base_tokenizer.eot_token
