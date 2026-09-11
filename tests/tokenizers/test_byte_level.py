"""``ByteLevelTokenizer``: specials, UTF-8 boundaries, and the per-format alphabet.

Special-token round trips are shared with ``CharLevelTokenizer``, which makes the
same promises.
"""

import pytest

from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.char_level import CharLevelTokenizer
from praxis.tokenizers.chat_templates import chat_format_of

TOKENIZERS = [ByteLevelTokenizer, CharLevelTokenizer]


def _specials(tok):
    return tok.bos_token, tok.eos_token, tok.pad_token


@pytest.mark.parametrize("cls", TOKENIZERS)
@pytest.mark.parametrize(
    "layout",
    [
        "{bos}Hello{eos}",
        "{bos}Hello{eos}{pad}",
        "Hello{eos}World",
        "{bos}{eos}Hello",
        "{bos}Hello World{eos}",
    ],
)
def test_specials_round_trip(cls, layout):
    """Named specials inline in text survive both the id path and the token
    path, wherever they sit."""
    tok = cls()
    bos, eos, pad = _specials(tok)
    text = layout.format(bos=bos, eos=eos, pad=pad)

    ids = tok.encode(text, add_special_tokens=False)
    assert tok.decode(ids, skip_special_tokens=False) == text
    if layout.startswith("{bos}"):
        assert ids[0] == tok.BOS_ID
    tokens = tok.tokenize(text)
    assert all(special in tokens for special in (bos, eos, pad) if special in text)
    assert tok.convert_tokens_to_string(tokens) == text


@pytest.mark.parametrize("cls", TOKENIZERS)
def test_batch_padding_keeps_specials(cls):
    tok = cls()
    bos, eos, _ = _specials(tok)
    texts = [f"{bos}Hello{eos}", f"{bos}Hi{eos}"]
    batch = tok(texts, padding=True, return_tensors="pt")
    assert batch["input_ids"].shape[0] == 2
    for i, text in enumerate(texts):
        assert text in tok.decode(batch["input_ids"][i], skip_special_tokens=False)


def test_add_special_tokens_flag():
    """Specials are added when asked, absent otherwise, and never doubled onto
    text that already carries them. (CharLevelTokenizer adds none either way.)"""
    tok = ByteLevelTokenizer()
    bos, eos, _ = _specials(tok)
    text = "Hello, world!"

    added = tok.decode(tok.encode(text, add_special_tokens=True))
    assert bos in added and eos in added
    plain = tok.decode(tok.encode(text, add_special_tokens=False))
    assert bos not in plain and eos not in plain

    wrapped = f"{bos}{text}{eos}"
    assert tok.decode(tok.encode(wrapped, add_special_tokens=False)) == wrapped
    again = tok.decode(tok.encode(wrapped, add_special_tokens=True))
    assert again.count(bos) == 1 and again.count(eos) == 1


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

MIXED_TEXT = "café naïve \u2014 日本語 test ✓ emoji 🙂 end"


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


# ------------------------------------------------- control-token inventory
#
# The head is exactly as wide as the tokenizer's alphabet, so every id the
# format cannot make a target is an id sampling can still pick. These pin the
# inventory in both directions: what must not exist, and what must.


def test_prose_registers_no_control_token(prose_tokenizer):
    """No named special and no [TOOL_CALL] id under prose - it lays turns and
    tool calls out as ordinary text, so none can appear in a decode.

    Registering them anyway is not cosmetic: byte_alphabet_size counts them and
    the model's output head is sized from it, so four logits would exist that no
    training example can ever make a target.
    """
    assert prose_tokenizer.eos_token_id is None
    assert chat_format_of(prose_tokenizer).document_separator is None
    assert not prose_tokenizer.tool_tokens_registered
    assert "[TOOL_CALL]" not in prose_tokenizer.get_vocab()
    assert prose_tokenizer.tool_call_token_id is None
    # The string is ordinary text, so it encodes to its bytes.
    assert len(prose_tokenizer.encode("[TOOL_CALL]")) == len("[TOOL_CALL]")


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
