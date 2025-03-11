from mfai.encoding import get_tiktoken_encoding


def test_encoding():
    encoding = get_tiktoken_encoding("gpt2")
    code = encoding.encode("Hello, world")
    assert code == [15496, 11, 995]
