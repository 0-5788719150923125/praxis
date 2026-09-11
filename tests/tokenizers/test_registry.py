import pytest

from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.chat_templates import chat_format_of, tokenize_with_mask

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


CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "And of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
]


def tokenizer_for(chat_format):
    return create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format=chat_format
    )


# ------------------------------------------- assistant mask on non-ASCII text


MULTIBYTE = [
    {"role": "system", "content": "“quoted”"},
    {"role": "user", "content": "Calculate √1156 \U0001f600"},
    {"role": "assistant", "content": "— the answer is 34."},
    {"role": "user", "content": "and été?"},
    {"role": "assistant", "content": "Summer."},
]


def _trained_text(tok, messages):
    from praxis.tokenizers.chat_templates import tokenize_with_mask

    ids, mask = tokenize_with_mask(tok, messages)
    return tok.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)


@pytest.mark.parametrize("fmt_name", ["default", "prose"])
def test_mask_is_exact_on_multibyte_text(fmt_name):
    """HuggingFace's return_assistant_tokens_mask maps CHARACTER offsets to token
    spans, which slips wherever one character is several tokens. Measured on the
    byte tokenizer before the fix: every multi-byte character before a span shifted
    the prose mask two tokens (cumulatively, so 'The answer' became 'r is'), and a
    multi-byte character starting a span shifted the default mask by its byte
    length. That is a silently corrupted SFT objective on any text with a curly
    quote, accent, em dash or emoji - which is most real text."""
    tok = tokenizer_for(fmt_name)
    trained = _trained_text(tok, MULTIBYTE)
    # Every assistant turn, whole and unshifted.
    assert "— the answer is 34." in trained
    assert "Summer." in trained
    # And nothing from a prompt turn.
    assert "quoted" not in trained
    assert "1156" not in trained
    assert "été" not in trained


@pytest.mark.parametrize("fmt_name", ["default", "prose"])
def test_segment_join_is_byte_identical_to_the_template(fmt_name):
    """The segment split only stays safe while it renders exactly what Jinja
    does - otherwise the fix silently changes the training data."""
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


@pytest.mark.parametrize("fmt_name", ["default", "prose"])
def test_segment_ids_match_whole_string_encoding(fmt_name):
    """Piece-wise encoding is only equivalent for merge-free tokenizers; this is
    the property that licenses the whole approach."""
    from praxis.tokenizers.chat_templates import tokenize_with_mask

    tok = tokenizer_for(fmt_name)
    ids, mask = tokenize_with_mask(tok, MULTIBYTE)
    whole = tok.encode(
        tok.apply_chat_template(MULTIBYTE, tokenize=False), add_special_tokens=False
    )
    assert ids == whole
    assert len(mask) == len(ids)


# ------------------------------------------------------------------------------
# tokenizers
# ------------------------------------------------------------------------------


def test_create_tokenizer_dispatch_char_level():
    from praxis.tokenizers import CharLevelTokenizer, create_tokenizer

    t = create_tokenizer(tokenizer_type="char_level")
    assert isinstance(t, CharLevelTokenizer)
