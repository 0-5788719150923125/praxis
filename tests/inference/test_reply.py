"""Reply extraction: cutting the model's answer out of a finished turn."""

import pytest

from praxis.inference.reply import (
    EMPTY_REPLY_PLACEHOLDER,
    _extract_reply_text_boundaries,
    extract_assistant_reply,
)
from praxis.inference.request import GenerationResult
from praxis.tokenizers.chat_templates import chat_format_of
from tests.stubs import _ForeignTokenizer


def test_reply_extraction_under_prose(prose_tokenizer):
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
    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = f"{prompt}{role}\n\n"
    assert extract_assistant_reply(text, prose_tokenizer) == EMPTY_REPLY_PLACEHOLDER


def test_prose_reply_keeps_a_role_word_that_is_only_prose(prose_tokenizer):
    """The cut needs the boundary's trailing blank line, not just the word."""
    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = f"{prompt}call me back later.\n\nuser\n\n"
    assert extract_assistant_reply(text, prose_tokenizer) == "call me back later."


def test_reply_extraction_strips_tool_plumbing_under_default(default_tokenizer):
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
    fmt = chat_format_of(prose_tokenizer)
    empty = _extract_reply_text_boundaries("user\n\n", fmt, prose_tokenizer, 0)
    assert empty == ""
    kept = _extract_reply_text_boundaries(
        "call\n\nme back later.", fmt, prose_tokenizer, 0
    )
    assert kept == "call\n\nme back later."


def test_a_foreign_reply_is_cut_at_the_models_own_end_token():
    """A model Praxis did not train ends its turn on its own EOS, not on our
    boundaries; cutting anywhere else returns the raw transcript."""
    tok = _ForeignTokenizer()
    prompt = "<|im_start|>user\nhi<|im_end|><|im_start|>assistant\n"
    full = prompt + "Here is the answer.<|im_end|>whatever came after"
    reply = extract_assistant_reply(GenerationResult(full, len(prompt)), tok)
    assert reply == "Here is the answer."
