import pytest

from praxis.generation.reply import extract_assistant_reply
from praxis.generation.request import GenerationResult
from praxis.tokenizers.chat_templates import chat_format_of

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


# ------------------------------------------------------------ web layer


def test_reply_extraction_under_prose(prose_tokenizer):
    from praxis.generation.reply import extract_assistant_reply

    text = (
        "user\n\nWhat is 750 times 485?\n\nassistant\n\nlet me check.\n\n"
        'call\n\n{"name": "calc", "arguments": {}}\n\ntool\n\n363750.0\n\n'
        "assistant\n\n750 times 485 equals 363750.\n\nuser\n\n"
    )
    assert extract_assistant_reply(text, prose_tokenizer) == (
        "750 times 485 equals 363750."
    )


@pytest.mark.parametrize("role", ["user", "system", "developer", "tool", "call"])
def test_empty_prose_turn_does_not_leak_the_next_speaker(prose_tokenizer, role):
    """The seam case: the prompt already supplied the boundary's blank line.

    A generation prompt ends `...assistant\\n\\n`, so a model that writes
    nothing and goes straight to naming the next speaker emits only
    `<role>\\n\\n`. The full sequence ends in a real stop string and generation
    halts correctly - but the slice handed to the extractor is missing the
    leading `\\n\\n`, so a cut that only looks for the full `\\n\\n<role>\\n\\n`
    form returns the bare role word as if it were the model's answer.
    """
    from praxis.generation.reply import (
        EMPTY_REPLY_PLACEHOLDER,
        extract_assistant_reply,
    )

    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = f"{prompt}{role}\n\n"
    assert extract_assistant_reply(text, prose_tokenizer) == EMPTY_REPLY_PLACEHOLDER


def test_prose_reply_keeps_a_role_word_that_is_only_prose(prose_tokenizer):
    """The cut needs the boundary's trailing blank line, not just the word."""
    from praxis.generation.reply import extract_assistant_reply

    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = f"{prompt}call me back later.\n\nuser\n\n"
    assert extract_assistant_reply(text, prose_tokenizer) == "call me back later."


def test_reply_extraction_strips_tool_plumbing_under_default(default_tokenizer):
    from praxis.generation.reply import extract_assistant_reply

    text = (
        "[BOS]assistant\n[TOOL_CALL]\n"
        '{"name": "calc", "arguments": {}}\n[/TOOL_CALL]\n'
        "The answer is 4.\n[SEP]\n"
    )
    assert extract_assistant_reply(text, default_tokenizer) == "The answer is 4."


def test_seam_cut_needs_the_boundary_to_be_the_whole_turn(prose_tokenizer):
    """A role word merely OPENING ordinary prose is not an empty turn.

    Unreachable from the decode loop - every role here is a stop string, so
    generation halts at the seam and the slice really is just the boundary -
    but `extract_assistant_reply` is public and also takes bare strings.
    """
    from praxis.generation.reply import (
        EMPTY_REPLY_PLACEHOLDER,
        _extract_reply_text_boundaries,
    )

    fmt = chat_format_of(prose_tokenizer)
    empty = _extract_reply_text_boundaries("user\n\n", fmt, prose_tokenizer, 0)
    assert empty == ""
    kept = _extract_reply_text_boundaries(
        "call\n\nme back later.", fmt, prose_tokenizer, 0
    )
    assert kept == "call\n\nme back later."
    assert EMPTY_REPLY_PLACEHOLDER  # the caller substitutes it for `empty`


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


def test_the_reply_is_cut_at_the_models_own_end_token():
    tok = _ForeignTokenizer()
    prompt = "<|im_start|>user\nhi<|im_end|><|im_start|>assistant\n"
    full = prompt + "Here is the answer.<|im_end|>whatever came after"
    reply = extract_assistant_reply(GenerationResult(full, len(prompt)), tok)
    assert reply == "Here is the answer."
