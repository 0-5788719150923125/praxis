import pytest

from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.chat_templates import chat_format_of

# ------------------------------------------------------------------------------
# tokenizers
# ------------------------------------------------------------------------------


# Only test ByteLevelTokenizer directly, StandardTokenizer needs to be loaded
TOKENIZER_TYPES = [ByteLevelTokenizer]


@pytest.fixture(params=TOKENIZER_TYPES)
def tokenizer_setup(request):
    tokenizer = request.param
    return tokenizer


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


# ------------------------------------------------------------------------------
# chat_formats
# ------------------------------------------------------------------------------
# Tests for the ``chat_formats`` registry and the text-boundary (prose) format.
#
# The invariants worth pinning are the ones that silently produce a broken run rather
# than an exception:
#
# - the `default` profile must stay byte-identical, since every existing checkpoint's
# data pipeline depends on it, - the boundary that ENDS a generated turn must be a
# trained target (the defect `prose` exists to remove), - a stop-string halt must not
# re-fire on the boundary it resumed from, or the tool loop returns zero new tokens
# forever, - the tool flow's three boundaries must classify unambiguously.


def test_prose_has_no_separator_to_decode(prose_tokenizer):
    """Prose defines no control token, so none can appear in a decode.

    The reply extractor therefore only ever has to cut on role boundaries -
    there is no literal `[EOS]` string for it to trip over, because the id does
    not exist in this tokenizer at all.
    """
    from praxis.generation.reply import (
        EMPTY_REPLY_PLACEHOLDER,
        extract_assistant_reply,
    )

    assert prose_tokenizer.eos_token_id is None
    assert chat_format_of(prose_tokenizer).document_separator is None

    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert extract_assistant_reply(
        f"{prompt}Hello there.\n\nuser\n\n", prose_tokenizer
    ) == ("Hello there.")
    assert (
        extract_assistant_reply(f"{prompt}\n\nuser\n\n", prose_tokenizer)
        == EMPTY_REPLY_PLACEHOLDER
    )


# ------------------------------------------------- control-token inventory
#
# The head is exactly as wide as the tokenizer's alphabet, so every id the
# format cannot make a target is an id sampling can still pick. These pin the
# inventory in both directions: what must not exist, and what must.


def test_prose_does_not_register_the_tool_tokens(prose_tokenizer, default_tokenizer):
    """No [TOOL_CALL] id under prose - it lays tool calls out as ordinary turns.

    Registering them anyway is not cosmetic: byte_alphabet_size counts them and
    the model's output head is sized from it, so four logits would exist that no
    training example can ever make a target.
    """
    assert not prose_tokenizer.tool_tokens_registered
    assert "[TOOL_CALL]" not in prose_tokenizer.get_vocab()
    assert prose_tokenizer.tool_call_token_id is None
    # The string is now ordinary text, so it encodes to its bytes.
    assert len(prose_tokenizer.encode("[TOOL_CALL]")) == len("[TOOL_CALL]")

    # default keeps them: its template renders them as atomic markers.
    assert default_tokenizer.tool_tokens_registered
    assert default_tokenizer.encode("[TOOL_CALL]") == [
        default_tokenizer.tool_call_token_id
    ]


def test_alphabet_and_head_shrink_together(prose_tokenizer, default_tokenizer):
    """The tokenizer's alphabet IS the byte-latent head width.

    A tokenizer that drops tokens without the encoder following would leave the
    head sized for ids the tokenizer can no longer produce, which is the same
    defect with the sign flipped.
    """
    from types import SimpleNamespace

    from praxis.encoders.byte_latent.config import create_base_config

    def head_width(tok):
        cfg = SimpleNamespace(
            byte_vocab_size=tok.byte_alphabet_size,
            hidden_size=64,
            embed_size=32,
            dropout=0.0,
            epsilon=1e-5,
            max_position_embeddings=1024,
            meta=[],
        )
        return create_base_config(cfg).local_vocab_size

    assert default_tokenizer.byte_alphabet_size == 264  # 256 + 4 named + 4 tool
    assert prose_tokenizer.byte_alphabet_size == 256  # pure bytes, nothing else
    assert head_width(default_tokenizer) == 264
    assert head_width(prose_tokenizer) == 256
    # The offset moves with the alphabet, or byte arithmetic downstream breaks.
    assert default_tokenizer.byte_offset == 4
    assert prose_tokenizer.byte_offset == 0
