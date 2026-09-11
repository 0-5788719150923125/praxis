import pytest

from praxis import registry
from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.chat_templates import (
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_FORMAT,
    HF_NATIVE_FORMAT,
    apply_chat_format,
    chat_format_of,
    get_chat_template,
    resolve_chat_format,
    tokenize_with_mask,
)

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


def test_unknown_format_is_a_hard_error():
    with pytest.raises(ValueError, match="Unknown chat_format"):
        resolve_chat_format("chatml-ish")
    with pytest.raises(ValueError, match="Unknown chat_format"):
        create_tokenizer(
            tokenizer_type="byte_level", vocab_size=1024, chat_format="chatml-ish"
        )


def test_default_template_unchanged(default_tokenizer):
    """Existing runs depend on this string; the registry must not perturb it."""
    assert default_tokenizer.chat_template == DEFAULT_CHAT_TEMPLATE
    assert get_chat_template("default") == DEFAULT_CHAT_TEMPLATE
    # Pre-registry call sites pass a tokenizer TYPE, which is not a format name.
    assert get_chat_template("byte_level") == DEFAULT_CHAT_TEMPLATE


def test_format_recovered_from_template_when_attribute_is_lost(prose_tokenizer):
    """`chat_format` is a plain attribute and does not survive
    save_pretrained; `chat_template` does. Losing the pairing would leave a
    prose template with the default halting contract, which never terminates."""
    tok = tokenizer_for("prose")
    del tok.chat_format
    assert chat_format_of(tok).name == "prose"


def test_apply_chat_format_sets_both_halves(default_tokenizer):
    tok = tokenizer_for("default")
    apply_chat_format(tok, "prose")
    assert chat_format_of(tok).name == "prose"
    assert tok.chat_template == registry.lookup("chat_formats", "prose").template


# ------------------------------------------------------------- rendering


def test_prose_render_has_no_control_tokens(prose_tokenizer):
    text = prose_tokenizer.apply_chat_template(CONVERSATION, tokenize=False)
    for token in ("[BOS]", "[EOS]", "[SEP]", "[PAD]"):
        assert token not in text
    assert text.startswith("system\n\nYou are a helpful assistant.\n\nuser\n\n")
    assert text.endswith("Tokyo.\n\n")

    ids = prose_tokenizer.encode(text)
    assert all(i >= 4 for i in ids), "no id below OFFSET may survive"


def test_prose_trains_the_boundary_that_ends_the_turn(prose_tokenizer):
    """The whole point: an assistant turn's mask must cover the boundary
    naming the next speaker, so the halt signal is a trained target."""
    enc = prose_tokenizer.apply_chat_template(
        CONVERSATION,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = enc["input_ids"], enc["assistant_masks"]
    assert len(mask) == len(ids)

    text = prose_tokenizer.decode(ids, skip_special_tokens=False)
    trained = "".join(prose_tokenizer.decode([t]) for t, m in zip(ids, mask) if m)
    # First assistant turn ends by naming the next speaker.
    assert "Paris is the capital of France.\n\nuser\n\n" in trained
    # The user's own words are never a target.
    assert "And of Japan?" not in trained
    assert "And of Japan?" in text


def test_default_leaves_its_turn_opener_untrained(default_tokenizer):
    """The measured defect, pinned so a template edit cannot reintroduce it
    silently: under `default` the BOS opening a turn has zero gradient."""
    enc = default_tokenizer.apply_chat_template(
        CONVERSATION,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = enc["input_ids"], enc["assistant_masks"]
    bos = default_tokenizer.bos_token_id
    bos_positions = [i for i, t in enumerate(ids) if t == bos]
    assert bos_positions, "sanity: the default format uses BOS"
    assert not any(mask[i] for i in bos_positions)


# --------------------------------------------------------------- halting


def test_prose_halts_on_strings_plus_the_one_retained_id(
    prose_tokenizer, default_tokenizer
):
    """Turn boundaries are strings, and there is no halt ID at all.

    prose keeps no control token: the template emits none and the packer
    appends none, so an id-based halt would be a logit the data never makes a
    target. Halting is entirely by stop string, which is a trained target
    because the boundary sits inside the generated turn's span.
    """
    prose = chat_format_of(prose_tokenizer)
    assert prose.stop_token_ids(prose_tokenizer) == []
    assert "\n\nuser\n\n" in prose.stop_strings()

    default = chat_format_of(default_tokenizer)
    assert default.stop_token_ids(default_tokenizer) == [
        default_tokenizer.eos_token_id,
        default_tokenizer.sep_token_id,
    ]
    assert default.stop_strings() == ()


def test_reply_boundary_is_not_a_stop_string(prose_tokenizer):
    """Both the generation prompt and the post-tool splice END with the reply
    boundary; treating it as a stop string would halt every resumed step
    before it produced a token."""
    fmt = chat_format_of(prose_tokenizer)
    assert fmt.boundary(fmt.reply_role) not in fmt.stop_strings()


def test_bpe_keeps_the_offset_mask():
    """A merge can straddle a segment boundary, so piece-wise encoding would
    change BPE's tokenization. Those tokenizers must decline the segment path -
    and they do not need it: their characters map to tokens cleanly."""
    from praxis.tokenizers.chat_templates import tokenize_with_mask
    from praxis.tokenizers.standard import StandardTokenizer

    bpe = StandardTokenizer(tokenizer_type="bpe", vocab_size=1024)
    assert not getattr(bpe, "context_free_tokenization", False)
    assert tokenize_with_mask(bpe, CONVERSATION) is None


def test_unproducible_control_ids_are_suppressed(prose_tokenizer, default_tokenizer):
    """Suppression covers ids a format cannot train but could still sample.

    In the BLT layout ids 0-3 exist whether or not a format uses them, so the
    unused ones are kept out of samples. In the pure-byte layout there is
    nothing to suppress, because there is no id that is not a byte - which is
    the stronger version of the same guarantee.
    """
    prose = chat_format_of(prose_tokenizer)
    assert prose.suppressed_token_ids(prose_tokenizer) == []
    assert prose_tokenizer.byte_alphabet_size == 256

    # default renders BOS and SEP, so only PAD is unreachable there.
    default = chat_format_of(default_tokenizer)
    assert default.suppressed_token_ids(default_tokenizer) == [
        default_tokenizer.pad_token_id
    ]


# ------------------------------------------------------------------------------
# hf_native_format
# ------------------------------------------------------------------------------
# Running a model Praxis did not train.
#
# Everything else in the generation stack is now model-agnostic - the decode loops are
# transformers decoding methods, halting is transformers' own criteria, and the backend
# wraps ``model.generate``. What remained Praxis-specific was the TOKENIZER side:
# ``get_chat_format`` answered ``default`` for anything it did not recognise, so a
# foreign model's prompt was rendered by its own template while halting and reply
# extraction were measured against ``[BOS]role`` boundaries its output never contains.
# The turn ran to ``max_new_tokens`` and came back as the raw transcript, with no error
# anywhere.


class _ForeignTokenizer:
    """The shape of an off-the-hub tokenizer, minus everything irrelevant."""

    chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
        "{{ m['content'] }}<|im_end|>{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    eos_token = "<|im_end|>"
    eos_token_id = 7
    bos_token = "<|begin|>"
    bos_token_id = 1
    sep_token = None
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=False):
        return "".join("<|im_end|>" if i == 7 else "?" for i in ids)


@pytest.fixture(scope="module")
def praxis_tokenizer():
    return create_tokenizer(tokenizer_type="byte_level", vocab_size=1024)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_an_unrecognised_template_resolves_to_the_foreign_contract():
    assert chat_format_of(_ForeignTokenizer()) is HF_NATIVE_FORMAT


def test_a_customised_praxis_tokenizer_is_not_treated_as_foreign(praxis_tokenizer):
    """A Praxis tokenizer someone gave a custom template is still ours, and
    keeps the contract it has always had."""
    praxis_tokenizer.chat_template = "{{ 'something bespoke' }}"
    try:
        assert chat_format_of(praxis_tokenizer) is DEFAULT_FORMAT
    finally:
        del praxis_tokenizer.chat_template


def test_a_tokenizer_with_no_template_is_unchanged():
    assert chat_format_of(object()) is DEFAULT_FORMAT
    assert chat_format_of(None) is DEFAULT_FORMAT


def test_the_foreign_contract_is_selectable_by_name():
    assert "hf_native" in registry.namespace("chat_formats")
    assert resolve_chat_format("hf_native") is HF_NATIVE_FORMAT


# ---------------------------------------------------------------------------
# what the contract actually asserts
# ---------------------------------------------------------------------------


def test_the_turn_ends_at_the_models_own_eos():
    tok = _ForeignTokenizer()
    assert HF_NATIVE_FORMAT.stop_token_ids(tok) == [tok.eos_token_id]
    # No stop STRINGS: this format halts on ids, and inventing text boundaries
    # for a model that was never trained on them is how the old fallback broke.
    assert HF_NATIVE_FORMAT.stop_strings() == ()


def test_tool_calling_turns_itself_off_for_a_foreign_model():
    """A model never trained on our tool layout cannot participate in it. The
    format names tool markers, the foreign tokenizer registers none, and the
    Generator's `boundaries_detectable` check is what notices."""
    from praxis.tools import tool_token_ids

    tok = _ForeignTokenizer()
    assert HF_NATIVE_FORMAT.tool_style == "tokens"
    assert tool_token_ids(tok).get("call_close") is None


def test_padding_is_never_sampled_but_the_models_own_tokens_are():
    """The suppression list exists to keep ids the model cannot have learned
    out of samples. For a foreign model we know only one such id."""
    tok = _ForeignTokenizer()
    suppressed = HF_NATIVE_FORMAT.suppressed_token_ids(tok)
    assert tok.pad_token_id in suppressed
    assert tok.eos_token_id not in suppressed


def test_the_foreign_contract_claims_no_template():
    """Leaving it empty is deliberate: nothing here is a trained target, and a
    populated template would make it look like a format to train against.
    Rendering always goes through the tokenizer's own `apply_chat_template`."""
    assert HF_NATIVE_FORMAT.template == ""
    # ...and an empty template must never match during template recovery, or
    # every tokenizer without one would resolve here.
    assert chat_format_of(object()) is DEFAULT_FORMAT
