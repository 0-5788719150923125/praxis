"""Running a model Praxis did not train.

Everything else in the generation stack is now model-agnostic - the decode
loops are transformers decoding methods, halting is transformers' own criteria,
and the backend wraps ``model.generate``. What remained Praxis-specific was the
TOKENIZER side: ``get_chat_format`` answered ``default`` for anything it did not
recognise, so a foreign model's prompt was rendered by its own template while
halting and reply extraction were measured against ``[BOS]role`` boundaries its
output never contains. The turn ran to ``max_new_tokens`` and came back as the
raw transcript, with no error anywhere.
"""

import pytest

from praxis import registry
from praxis.generation.reply import extract_assistant_reply
from praxis.generation.request import GenerationResult
from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.chat_templates import (
    DEFAULT_FORMAT,
    HF_NATIVE_FORMAT,
    PROSE_FORMAT,
    chat_format_of,
    resolve_chat_format,
)


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


def test_a_praxis_tokenizer_keeps_its_own_format(praxis_tokenizer):
    """The declared `chat_format` attribute wins, and always did."""
    assert chat_format_of(praxis_tokenizer) is DEFAULT_FORMAT
    prose = create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format="prose"
    )
    assert chat_format_of(prose) is PROSE_FORMAT


def test_a_praxis_template_still_round_trips_through_save_and_load():
    """`chat_format` is a plain attribute and does not survive
    save_pretrained/from_pretrained, but `chat_template` does - so a Praxis
    template must still be recognised on a tokenizer carrying nothing else."""

    class _Reloaded:
        chat_template = PROSE_FORMAT.template

    assert chat_format_of(_Reloaded()) is PROSE_FORMAT


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


def test_the_reply_is_cut_at_the_models_own_end_token():
    tok = _ForeignTokenizer()
    prompt = "<|im_start|>user\nhi<|im_end|><|im_start|>assistant\n"
    full = prompt + "Here is the answer.<|im_end|>whatever came after"
    reply = extract_assistant_reply(GenerationResult(full, len(prompt)), tok)
    assert reply == "Here is the answer."


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
