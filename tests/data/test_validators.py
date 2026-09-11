"""Tests for praxis/data/validators: the chat-template validator's structural
BOS check (default format) and its boundary check (prose format)."""

import pytest
import torch

from praxis.data.validators import ChatTemplateValidator

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "And of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
]


@pytest.fixture
def validator(default_tokenizer):
    return ChatTemplateValidator(default_tokenizer, strict_mode=False)


def _render(tokenizer, messages):
    enc = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
    return torch.as_tensor(enc["input_ids"], dtype=torch.long)


def _encode(tokenizer, text):
    return torch.as_tensor(
        tokenizer.encode(text, add_special_tokens=False), dtype=torch.long
    )


# ------------------------------------------------------------------------------
# default format: every structural [BOS] opens a role
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "messages,min_bos",
    [
        *[
            pytest.param([{"role": role, "content": "Test content"}], 1, id=role)
            for role in ("system", "developer", "user", "assistant", "tool")
        ],
        pytest.param(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "developer", "content": "Continue this text:"},
                {"role": "assistant", "content": "Here is the continuation..."},
            ],
            3,
            id="developer turn",
        ),
        pytest.param(CONVERSATION, 5, id="multi-turn"),
    ],
)
def test_rendered_documents_validate(default_tokenizer, validator, messages, min_bos):
    ids = _render(default_tokenizer, messages)
    assert (ids == validator.bos_token_id).sum().item() >= min_bos
    is_valid, violations = validator.validate_token_sequence(ids, messages)
    assert is_valid, violations


def test_bos_followed_by_non_role_is_invalid(default_tokenizer, validator):
    bad = torch.cat(
        [
            torch.tensor([validator.bos_token_id]),
            _encode(default_tokenizer, "random text here"),
        ]
    )
    is_valid, violations = validator.validate_token_sequence(bad)
    assert not is_valid
    assert violations[0]["position"] == 0

    report = validator.format_violation_report(
        violations,
        messages=[{"role": "user", "content": "test"}],
        formatted_text="[BOS]random text here",
        token_ids=bad,
    )
    assert "Position" in report

    # The same check on a hand-written document, [BOS] encoded from text.
    bad = _encode(default_tokenizer, "[BOS]not-a-role\nbody\n[SEP]\n")
    assert not validator.validate_token_sequence(bad)[0]


def test_trailing_bos_is_skipped(default_tokenizer, validator):
    """A [BOS] at the very end has no next token to check, so it is not a
    violation."""
    messages = [{"role": "user", "content": "Hi"}]
    ids = torch.cat(
        [_render(default_tokenizer, messages), torch.tensor([validator.bos_token_id])]
    )
    is_valid, violations = validator.validate_token_sequence(ids, messages)
    assert is_valid, violations


def test_structural_vs_content_bos(default_tokenizer, validator):
    """Only a [BOS] at the start or after [SEP] is structural; one inside
    content is not validated."""
    sequence = torch.cat(
        [
            torch.tensor([validator.bos_token_id]),  # structural: position 0
            _encode(default_tokenizer, "assistant"),
            _encode(default_tokenizer, "Hello"),
            torch.tensor([validator.bos_token_id]),  # content: not after [SEP]
            _encode(default_tokenizer, "World"),
            torch.tensor([validator.sep_token_id]),
        ]
    )
    is_valid, violations = validator.validate_token_sequence(sequence)
    assert is_valid, violations


# ------------------------------------------------------------------------------
# prose format: every message's boundary appears, in order
# ------------------------------------------------------------------------------


def test_validator_accepts_prose_documents(prose_tokenizer):
    """The BOS-role check would reject every prose doc, silently draining the
    training stream, so the validator has to switch modes with the format."""
    validator = ChatTemplateValidator(tokenizer=prose_tokenizer)
    ids = _render(prose_tokenizer, CONVERSATION)
    is_valid, report = validator.validate_and_report(ids, messages=CONVERSATION)
    assert is_valid, report


def test_validator_flags_a_missing_prose_boundary(prose_tokenizer):
    validator = ChatTemplateValidator(tokenizer=prose_tokenizer)
    ids = _encode(prose_tokenizer, "system\n\nhi\n\nuser\n\nthere\n\n")
    # Claim an assistant turn the render does not contain.
    is_valid, report = validator.validate_and_report(ids, messages=CONVERSATION[:3])
    assert not is_valid
    assert "Missing boundary" in report
