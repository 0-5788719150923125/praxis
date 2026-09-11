"""Sweeps over every entry of the ``tokenizers`` and ``chat_formats`` registries.

Both lists are taken at COLLECTION time: importing the tokenmonster integration
registers more tokenizers at run time, and a sweep that iterated the registry then
would depend on test order.
"""

import pytest

from praxis import registry
from praxis.tokenizers.base import PraxisToolTokensMixin
from praxis.tokenizers.chat_templates import chat_format_of, tokenize_with_mask
from tests.stubs import tokenizer_for
from tests.tokenizers.conversations import CONVERSATION, MULTIBYTE

TOKENIZERS = sorted(registry.namespace("tokenizers"))
TEMPLATED_FORMATS = sorted(
    name for name, fmt in registry.namespace("chat_formats").items() if fmt.template
)


# ---------------------------------------------------------------------------
# tokenizers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", TOKENIZERS)
def test_every_tokenizer_builds_and_encodes(key):
    """Built from the registry factory directly, so the trained (bpe/unigram)
    entries construct untrained and never reach the hub. Those know no text
    yet (bpe encodes it to nothing); the merge-free ones must round-trip it."""
    tok = registry.lookup("tokenizers", key)(vocab_size=1024)
    assert isinstance(tok, PraxisToolTokensMixin)
    text = "Hello, world!"
    ids = tok.encode(text, add_special_tokens=False)
    assert all(isinstance(i, int) and 0 <= i < len(tok) for i in ids)
    if getattr(tok, "context_free_tokenization", False):
        assert tok.decode(ids, skip_special_tokens=True) == text


# ---------------------------------------------------------------------------
# chat formats with a template
# ---------------------------------------------------------------------------


def _trained_text(tok, messages):
    ids, mask = tokenize_with_mask(tok, messages)
    return tok.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)


@pytest.mark.parametrize("fmt_name", TEMPLATED_FORMATS)
def test_mask_is_exact_on_multibyte_text(fmt_name):
    """HuggingFace's return_assistant_tokens_mask maps CHARACTER offsets to token
    spans, which slips wherever one character is several tokens: every multi-byte
    character before a span shifted the prose mask two tokens (cumulatively, so
    'The answer' became 'r is'). That is a silently corrupted SFT objective on any
    text with a curly quote, accent, dash or emoji - which is most real text."""
    trained = _trained_text(tokenizer_for(fmt_name), MULTIBYTE)
    # Every assistant turn, whole and unshifted.
    assert MULTIBYTE[2]["content"] in trained
    assert "Summer." in trained
    # And nothing from a prompt turn.
    assert "quoted" not in trained
    assert "1156" not in trained
    assert "été" not in trained


@pytest.mark.parametrize("fmt_name", TEMPLATED_FORMATS)
def test_segment_join_is_byte_identical_to_the_template(fmt_name):
    """The segment split only stays safe while it renders exactly what Jinja
    does - otherwise it silently changes the training data."""
    tok = tokenizer_for(fmt_name)
    fmt = chat_format_of(tok)
    cases = [CONVERSATION, MULTIBYTE, [{"role": "user", "content": "solo"}]]
    for messages in cases:
        for add_gen in (False, True):
            for omit in (False, True):
                kwargs = {"add_generation_prompt": add_gen}
                if omit:
                    kwargs["omit_leading_bos"] = True
                jinja = tok.apply_chat_template(messages, tokenize=False, **kwargs)
                segments = fmt.render_segments(
                    messages,
                    tok,
                    add_generation_prompt=add_gen,
                    omit_leading_bos=omit,
                )
                assert "".join(text for text, _ in segments) == jinja


@pytest.mark.parametrize("fmt_name", TEMPLATED_FORMATS)
def test_segment_ids_match_whole_string_encoding(fmt_name):
    """Piece-wise encoding is only equivalent for merge-free tokenizers; this is
    the property that licenses the whole approach."""
    tok = tokenizer_for(fmt_name)
    ids, mask = tokenize_with_mask(tok, MULTIBYTE)
    whole = tok.encode(
        tok.apply_chat_template(MULTIBYTE, tokenize=False), add_special_tokens=False
    )
    assert ids == whole
    assert len(mask) == len(ids)
