import pytest


def test_char_level_tokenizer_round_trip():
    """CharLevelTokenizer must round-trip all BMP codepoints losslessly."""
    from praxis.tokenizers import CharLevelTokenizer

    t = CharLevelTokenizer()
    # 4 named specials + 65_536 BMP + 4 tool-control specials.
    assert t.vocab_size == 65544

    cases = [
        "Hello, world!",
        "Unicode: café naïve résumé",
        "CJK: 你好世界",
        "RTL: مرحبا",
        "digits 0123456789 symbols +-*/=",
        "\nmultiline\ttext\n",
    ]
    for text in cases:
        ids = t.encode(text, add_special_tokens=False)
        assert t.decode(ids, skip_special_tokens=False) == text


def test_char_level_tokenizer_special_tokens():
    from praxis.tokenizers import CharLevelTokenizer

    t = CharLevelTokenizer()
    text = f"{t.bos_token}Hello{t.eos_token}"
    ids = t.encode(text, add_special_tokens=False)
    assert ids[0] == t.BOS_ID
    assert ids[-1] == t.EOS_ID
    assert t.decode(ids, skip_special_tokens=False) == text


def test_char_level_tokenizer_non_bmp_round_trips_via_surrogates():
    """Non-BMP chars are encoded as UTF-16 surrogate pairs and round trip."""
    from praxis.tokenizers import CharLevelTokenizer

    t = CharLevelTokenizer()
    text = "A👋B"
    ids = t.encode(text, add_special_tokens=False)
    assert len(ids) == 4  # A + high + low + B
    decoded = t.decode(ids, skip_special_tokens=False)
    assert decoded == text
    assert decoded.encode("utf-8")  # valid UTF-8


def test_char_level_tokenizer_lone_surrogate_output_is_safe():
    """A lone surrogate id produced by the model must decode safely."""
    from praxis.tokenizers import CharLevelTokenizer

    t = CharLevelTokenizer()
    lone_high = 0xD801 + t.offset
    ids = [ord("h") + t.offset, lone_high, ord("i") + t.offset]
    decoded = t.decode(ids, skip_special_tokens=False)
    # Lone surrogate should not survive to the output string.
    assert decoded.encode("utf-8")
    assert "h" in decoded and "i" in decoded


def test_char_level_tokenizer_persistence(tmp_path):
    from praxis.tokenizers import CharLevelTokenizer

    t = CharLevelTokenizer()
    t.encode("héllo", add_special_tokens=False)
    t.save_vocabulary(str(tmp_path))
    loaded = CharLevelTokenizer.from_pretrained(str(tmp_path))
    assert ord("h") in loaded._observed
    assert ord("é") in loaded._observed
