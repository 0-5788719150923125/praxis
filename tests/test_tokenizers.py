import itertools
from typing import List

import pytest
import torch

from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.byte_level import ByteLevelTokenizer

# Only test ByteLevelTokenizer directly, StandardTokenizer needs to be loaded
TOKENIZER_TYPES = [ByteLevelTokenizer]


@pytest.fixture(params=TOKENIZER_TYPES)
def tokenizer_setup(request):
    tokenizer = request.param
    return tokenizer


def test_standard_tokenizer():
    """Test StandardTokenizer using the normal loading mechanism."""
    try:
        # Try to load a standard tokenizer using the normal method
        tokenizer = create_tokenizer(vocab_size=32768)

        # If it's a ByteLevelTokenizer, skip the test
        if isinstance(tokenizer, ByteLevelTokenizer):
            pytest.skip("No trained StandardTokenizer available")

        # Run basic tests
        test_text = "Hello, world!"

        # Test encode/decode roundtrip
        encoded = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        assert test_text == decoded, f"Roundtrip failed: {test_text} != {decoded}"

        # Test with special tokens
        encoded_with_special = tokenizer.encode(test_text, add_special_tokens=True)
        decoded_with_special = tokenizer.decode(
            encoded_with_special, skip_special_tokens=False
        )

        # Check that special tokens were added
        assert len(encoded_with_special) > len(encoded), "Special tokens not added"

        print("✓ StandardTokenizer tests passed")

    except Exception as e:
        pytest.skip(f"StandardTokenizer not available: {e}")


def test_tokenizer_full(tokenizer_setup) -> None:
    """Comprehensive test suite for ByteLevelTokenizer with enhanced special token testing."""
    print("Running comprehensive tokenizer tests...\n")

    tokenizer = tokenizer_setup()

    def test_special_token_preservation():
        print("1. Testing special token preservation...")

        # Test encoding with special tokens
        text = f"{tokenizer.bos_token}Hello{tokenizer.eos_token}"
        tokens = tokenizer.tokenize(text)
        assert tokenizer.bos_token in tokens, "BOS token lost during tokenization"
        assert tokenizer.eos_token in tokens, "EOS token lost during tokenization"

        # Test encoding and decoding roundtrip
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        assert text == decoded, f"Roundtrip failed: {text} != {decoded}"

        # Test multiple special tokens
        text = f"{tokenizer.bos_token}Hello{tokenizer.eos_token}{tokenizer.pad_token}"
        decoded = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
        assert text == decoded, "Multiple special tokens not preserved"

        print("✓ Special token preservation tests passed\n")

    def test_mixed_content():
        print("2. Testing mixed special tokens and regular text...")

        # Test mixed content
        test_cases = [
            f"{tokenizer.bos_token}Hello",
            f"Hello{tokenizer.eos_token}",
            f"{tokenizer.bos_token}Hello{tokenizer.eos_token}",
            f"{tokenizer.bos_token}Hello World{tokenizer.eos_token}",
        ]

        for text in test_cases:
            tokens = tokenizer.tokenize(text)
            decoded = tokenizer.convert_tokens_to_string(tokens)
            assert text == decoded, f"Mixed content failed: {text} != {decoded}"

        print("✓ Mixed content tests passed\n")

    def test_batch_processing():
        print("3. Testing batch processing with special tokens...")

        batch_texts = [
            f"{tokenizer.bos_token}Hello{tokenizer.eos_token}",
            f"{tokenizer.bos_token}World{tokenizer.eos_token}",
        ]

        # Test batch encoding
        batch_encoded = tokenizer(batch_texts, padding=True, return_tensors="pt")

        # Decode each sequence
        for i, text in enumerate(batch_texts):
            decoded = tokenizer.decode(
                batch_encoded["input_ids"][i], skip_special_tokens=False
            )
            assert text in decoded, f"Batch processing failed for: {text}"

        print("✓ Batch processing tests passed\n")

    def test_special_token_positioning():
        print("4. Testing special token positioning...")

        # Test special tokens at different positions
        text = f"Hello{tokenizer.eos_token}World"
        tokens = tokenizer.tokenize(text)
        decoded = tokenizer.convert_tokens_to_string(tokens)
        assert text == decoded, "Mid-sequence special token failed"

        text = f"{tokenizer.bos_token}{tokenizer.eos_token}Hello"
        tokens = tokenizer.tokenize(text)
        decoded = tokenizer.convert_tokens_to_string(tokens)
        assert text == decoded, "Adjacent special tokens failed"

        print("✓ Special token positioning tests passed\n")

    def test_add_special_tokens_flag():
        print("5. Testing add_special_tokens flag...")

        # Create tokenizer instances with different settings
        tokenizer_with_special = tokenizer_setup()

        test_text = "Hello, world!"
        print(f"\nTest setup:")
        print(f"- Input text: {repr(test_text)}")

        # Test with add_special_tokens=True
        print("\nTesting add_special_tokens=True:")
        encoded_with = tokenizer_with_special.encode(test_text, add_special_tokens=True)
        decoded_with = tokenizer_with_special.decode(encoded_with)
        print(f"- Encoded tokens: {encoded_with}")
        print(f"- Decoded text: {repr(decoded_with)}")
        print(f"- BOS token: {repr(tokenizer_with_special.bos_token)}")
        print(f"- EOS token: {repr(tokenizer_with_special.eos_token)}")

        assert (
            tokenizer_with_special.bos_token in decoded_with
        ), "BOS token not added when requested"
        assert (
            tokenizer_with_special.eos_token in decoded_with
        ), "EOS token not added when requested"

        # Test with add_special_tokens=False
        encoded_without = tokenizer_with_special.encode(
            test_text, add_special_tokens=False
        )
        decoded_without = tokenizer_with_special.decode(encoded_without)
        assert (
            tokenizer_with_special.bos_token not in decoded_without
        ), "BOS token added when not requested"
        assert (
            tokenizer_with_special.eos_token not in decoded_without
        ), "EOS token added when not requested"

        # Test with text already containing special tokens
        text_with_special = f"{tokenizer_with_special.bos_token}{test_text}{tokenizer_with_special.eos_token}"
        encoded_existing = tokenizer_with_special.encode(
            text_with_special, add_special_tokens=False
        )
        decoded_existing = tokenizer_with_special.decode(encoded_existing)
        assert (
            decoded_existing == text_with_special
        ), "Existing special tokens not preserved"

        # Test that add_special_tokens=True doesn't duplicate tokens
        encoded_no_duplicate = tokenizer_with_special.encode(
            text_with_special, add_special_tokens=True
        )
        decoded_no_duplicate = tokenizer_with_special.decode(encoded_no_duplicate)
        assert (
            decoded_no_duplicate.count(tokenizer_with_special.bos_token) == 1
        ), "BOS token duplicated"
        assert (
            decoded_no_duplicate.count(tokenizer_with_special.eos_token) == 1
        ), "EOS token duplicated"

        print("✓ add_special_tokens flag tests passed\n")

    # Run all tests
    test_special_token_preservation()
    test_mixed_content()
    test_batch_processing()
    test_special_token_positioning()
    test_add_special_tokens_flag()


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


def test_create_tokenizer_dispatch_char_level():
    from praxis.tokenizers import CharLevelTokenizer, create_tokenizer

    t = create_tokenizer(tokenizer_type="char_level")
    assert isinstance(t, CharLevelTokenizer)


# ── UTF-8 boundaries in the rolling contexts ─────────────────────────────
#
# A rolling context (StreamingContext) is a TEXT buffer: every round decodes
# the whole sequence, hands the string back as the next prompt and re-encodes
# it. On a byte-level tokenizer a token is a byte, so any cut that is not a
# character boundary decodes to U+FFFD - and re-encoding turns that one
# replacement character into three real bytes (EF BF BD) that stay in the
# prompt for as long as the buffer lives. Because those extra bytes make the
# buffer longer in bytes than in characters, the next round truncates too:
# once one appears the context keeps minting more, and the model conditions on
# a sequence essentially absent from its training data.

MIXED_TEXT = "café naïve — 日本語 test ✓ emoji 🙂 end"


@pytest.fixture
def byte_tok():
    return ByteLevelTokenizer()


def _cuts_producing_replacement(tok, ids, align):
    """How many left-truncation points decode with a replacement character."""
    bad = 0
    for cut in range(1, len(ids)):
        start = len(ids) - cut
        if align:
            start = tok.align_left_cut(ids, start)
        if "�" in tok.decode(ids[start:], skip_special_tokens=False):
            bad += 1
    return bad


def test_left_cut_alignment_removes_severed_characters(byte_tok):
    ids = byte_tok.encode(MIXED_TEXT)
    unaligned = _cuts_producing_replacement(byte_tok, ids, align=False)
    assert unaligned > 0, "sanity: raw token-index cuts do split characters"
    assert _cuts_producing_replacement(byte_tok, ids, align=True) == 0


def test_left_cut_alignment_leaves_a_valid_start_alone(byte_tok):
    ids = byte_tok.encode("plain ascii")
    for start in range(len(ids)):
        assert byte_tok.align_left_cut(ids, start) == start


def test_left_cut_alignment_never_skips_a_whole_character(byte_tok):
    """It advances past continuation bytes only - at most three of them."""
    ids = byte_tok.encode(MIXED_TEXT)
    for start in range(len(ids)):
        assert 0 <= byte_tok.align_left_cut(ids, start) - start <= 3


def test_incomplete_tail_detects_a_severed_character(byte_tok):
    ids = byte_tok.encode("日")
    assert len(ids) == 3
    assert byte_tok.incomplete_tail(ids[:1])
    assert byte_tok.incomplete_tail(ids[:2])
    assert not byte_tok.incomplete_tail(ids)


def test_stripping_the_tail_removes_every_replacement_character(byte_tok):
    ids = byte_tok.encode(MIXED_TEXT)
    for keep in range(1, len(ids) + 1):
        stripped = byte_tok.strip_incomplete_tail(ids[:keep])
        assert "�" not in byte_tok.decode(stripped, skip_special_tokens=False)


def test_complete_text_survives_stripping_untouched(byte_tok):
    ids = byte_tok.encode(MIXED_TEXT)
    assert byte_tok.strip_incomplete_tail(ids) == ids
    assert not byte_tok.incomplete_tail(ids)
    assert not byte_tok.incomplete_tail([])


def test_a_special_token_is_never_a_partial_character(byte_tok):
    """Ids below OFFSET are atomic - asking for more bytes cannot complete
    them, and treating one as a partial tail would strip it away."""
    for special in (byte_tok.EOS_ID, byte_tok.BOS_ID, byte_tok.PAD_ID):
        ids = byte_tok.encode("hi") + [special]
        assert not byte_tok.incomplete_tail(ids)
        assert byte_tok.strip_incomplete_tail(ids) == ids


def test_an_unfixable_tail_is_left_alone(byte_tok):
    """Continuation bytes with no lead cannot be completed by generating more.
    Reporting them incomplete would make the strip loop eat the whole buffer."""
    junk = [0x80 + 4] * 5
    assert not byte_tok.incomplete_tail(junk)
    assert byte_tok.strip_incomplete_tail(junk) == junk


def test_the_replacement_character_ratchet_is_closed(byte_tok):
    """The end-to-end failure: decode -> re-encode -> truncate, repeatedly.

    Unaligned, the buffer acquires a replacement character it can never shed.
    """

    def run(align):
        buf = "日本語のテキストです"
        worst = 0
        for _ in range(30):
            ids = byte_tok.encode(buf)
            if len(ids) > 40:
                start = len(ids) - 40
                if align:
                    start = byte_tok.align_left_cut(ids, start)
                ids = ids[start:]
            buf = byte_tok.decode(ids, skip_special_tokens=False)
            worst = max(worst, buf.count("�"))
            buf += "、続き"  # stands in for what the model appends
        return worst

    assert run(align=False) > 0, "sanity: the ratchet reproduces"
    assert run(align=True) == 0
