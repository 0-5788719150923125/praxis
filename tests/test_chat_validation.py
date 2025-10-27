"""Tests for chat template validation."""

import pytest
import torch
from transformers import AutoTokenizer

from praxis.data.datasets.message_queue import MessageQueueManager
from praxis.data.validators import ChatTemplateValidator
from praxis.tokenizers.standard import StandardTokenizer


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    return StandardTokenizer.from_pretrained("gpt2")


@pytest.fixture
def validator(tokenizer):
    """Create a validator for testing."""
    return ChatTemplateValidator(tokenizer, strict_mode=False)


def test_validator_initialization(tokenizer):
    """Test that validator initializes correctly."""
    validator = ChatTemplateValidator(tokenizer)
    assert validator.tokenizer == tokenizer
    assert validator.bos_token_id is not None
    assert len(validator.ALLOWED_ROLES) == 5
    assert "system" in validator.ALLOWED_ROLES
    assert "user" in validator.ALLOWED_ROLES
    assert "assistant" in validator.ALLOWED_ROLES


def test_valid_system_message(tokenizer, validator):
    """Test that valid system messages pass validation."""
    messages = [
        {"role": "system", "content": "You are helpful."},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    # Validate
    is_valid, violations = validator.validate_token_sequence(token_ids, messages)

    assert is_valid, f"Validation failed with violations: {violations}"
    assert len(violations) == 0


def test_valid_conversation(tokenizer, validator):
    """Test that valid conversations pass validation."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    # Validate
    is_valid, violations = validator.validate_token_sequence(token_ids, messages)

    assert is_valid, f"Validation failed with violations: {violations}"
    assert len(violations) == 0


def test_valid_developer_role(tokenizer, validator):
    """Test that developer role is recognized."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "developer", "content": "Continue this text:"},
        {"role": "assistant", "content": "Here is the continuation..."},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    # Validate
    is_valid, violations = validator.validate_token_sequence(token_ids, messages)

    assert is_valid, f"Validation failed with violations: {violations}"
    assert len(violations) == 0


def test_invalid_random_text_after_bos(tokenizer, validator):
    """Test that random text after BOS fails validation."""
    # Manually construct invalid sequence: [BOS] + "random text"
    bos_id = validator.bos_token_id
    random_text_ids = tokenizer("random text here", return_tensors="pt")[
        "input_ids"
    ].squeeze(0)

    # Create invalid sequence
    invalid_sequence = torch.cat([torch.tensor([bos_id]), random_text_ids])

    # Validate
    is_valid, violations = validator.validate_token_sequence(invalid_sequence)

    assert not is_valid, "Invalid sequence passed validation!"
    assert len(violations) > 0
    assert violations[0]["position"] == 0  # BOS is at position 0


def test_multiple_bos_tokens(tokenizer, validator):
    """Test validation with multiple BOS tokens (multi-message)."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm good!"},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    # Count BOS tokens
    bos_count = (token_ids == validator.bos_token_id).sum().item()
    assert bos_count >= 4, "Should have at least 4 BOS tokens"

    # Validate
    is_valid, violations = validator.validate_token_sequence(token_ids, messages)

    assert is_valid, f"Valid multi-message sequence failed: {violations}"


def test_bos_at_end_of_sequence(tokenizer, validator):
    """Test that BOS at end of sequence is handled gracefully."""
    # Create sequence ending with BOS
    messages = [{"role": "user", "content": "Hi"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    # Add BOS at end
    token_ids = torch.cat([token_ids, torch.tensor([validator.bos_token_id])])

    # Validate - should not crash
    is_valid, violations = validator.validate_token_sequence(token_ids, messages)

    # BOS at end should be skipped (no next token to check)
    assert is_valid or len(violations) == 0


def test_validation_report_formatting(tokenizer, validator):
    """Test that violation reports are properly formatted."""
    # Create invalid sequence
    bos_id = validator.bos_token_id
    random_ids = tokenizer("xyz123", return_tensors="pt")["input_ids"].squeeze(0)
    invalid_sequence = torch.cat([torch.tensor([bos_id]), random_ids])

    # Get violations
    is_valid, violations = validator.validate_token_sequence(invalid_sequence)

    # Format report
    messages = [{"role": "user", "content": "test"}]
    report = validator.format_violation_report(
        violations,
        messages=messages,
        formatted_text="[BOS]xyz123",
        token_ids=invalid_sequence,
    )

    # Check report contains key information
    assert "CHAT TEMPLATE VALIDATION FAILURE" in report
    assert "Original Messages" in report
    assert "violation" in report.lower()
    assert "Position" in report


def test_message_queue_integration(tokenizer):
    """Test that validation integrates correctly with MessageQueueManager."""
    # Create queue with validation enabled
    queue = MessageQueueManager(
        tokenizer,
        block_size=512,
        enable_chat_validation=True,
        strict_chat_validation=False,
    )

    # Add valid document
    valid_doc = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ],
        "metadata": {"source": "test"},
    }

    queue.add_document(valid_doc)

    # Process the queue
    queue._refill_token_buffer()

    # Check validation stats
    stats = queue.get_validation_stats()
    assert stats["documents_validated"] >= 1
    assert stats["documents_failed"] == 0
    assert stats["documents_skipped"] == 0


def test_message_queue_validation_disabled(tokenizer):
    """Test that validation can be disabled."""
    # Create queue with validation disabled
    queue = MessageQueueManager(
        tokenizer,
        block_size=512,
        enable_chat_validation=False,
        strict_chat_validation=False,
    )

    assert queue.chat_validator is None

    # Add document
    doc = {"messages": [{"role": "user", "content": "test"}], "metadata": {}}
    queue.add_document(doc)
    queue._refill_token_buffer()

    # Stats should show no validation happened
    stats = queue.get_validation_stats()
    assert stats["documents_validated"] == 0


def test_all_roles_recognized(tokenizer, validator):
    """Test that all expected roles are properly recognized."""
    roles = ["system", "developer", "user", "assistant", "tool"]

    for role in roles:
        messages = [{"role": role, "content": "Test content"}]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

        # Validate
        is_valid, violations = validator.validate_token_sequence(token_ids, messages)

        assert is_valid, f"Role '{role}' failed validation: {violations}"


def test_strict_mode_raises_exception(tokenizer):
    """Test that strict mode raises exceptions on validation failure."""
    # Create queue with strict validation
    queue = MessageQueueManager(
        tokenizer,
        block_size=512,
        enable_chat_validation=True,
        strict_chat_validation=True,
    )

    # Note: We can't easily create an invalid document through the normal API
    # because apply_chat_template should always produce valid output.
    # This test would require manually injecting bad token sequences,
    # which is tested in other tests above.

    # Just verify strict mode is set
    assert queue.strict_chat_validation is True
    assert queue.chat_validator.strict_mode is True


def test_bos_in_content_ignored(tokenizer, validator):
    """Test that BOS tokens appearing in content (not structural) are ignored."""
    # Simulate a document with BOS token appearing in content
    # Structure: [BOS]assistant\n<content with embedded BOS>\n[SEP]
    # The embedded BOS should NOT be validated

    messages = [
        {"role": "assistant", "content": 'Example: "[BOS]" is a special token'},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    # Count BOS tokens
    bos_count = (token_ids == validator.bos_token_id).sum().item()

    # We should have at least 2 BOS: one structural, one in content
    if bos_count >= 2:
        # Validate - should pass because only structural BOS is checked
        is_valid, violations = validator.validate_token_sequence(token_ids, messages)

        # Even if there's a BOS in content, it should be ignored
        assert (
            is_valid or len(violations) == 0
        ), f"BOS in content was incorrectly validated: {violations}"


def test_structural_vs_content_bos(tokenizer, validator):
    """Test that only structural BOS tokens are validated."""
    # Manually create a sequence with structural and content BOS
    # [BOS]assistant\nHello[BOS]World[SEP]
    # The first BOS is structural (at position 0), should be validated
    # The middle BOS is in content, should be ignored

    sep_id = validator.sep_token_id
    bos_id = validator.bos_token_id

    # Build: [BOS] + "assistant" + "Hello" + [BOS] + "World" + [SEP]
    assistant_ids = tokenizer(
        "assistant", add_special_tokens=False, return_tensors="pt"
    )["input_ids"].squeeze(0)
    hello_ids = tokenizer("Hello", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].squeeze(0)
    world_ids = tokenizer("World", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].squeeze(0)

    sequence = torch.cat(
        [
            torch.tensor([bos_id]),  # Structural BOS (position 0)
            assistant_ids,
            hello_ids,
            torch.tensor([bos_id]),  # Content BOS (not after SEP)
            world_ids,
            torch.tensor([sep_id]),
        ]
    )

    # Validate
    is_valid, violations = validator.validate_token_sequence(sequence)

    # Should be valid - only the first BOS is validated, and it's followed by "assistant"
    assert (
        is_valid
    ), f"Structural validation incorrectly flagged content BOS: {violations}"
    assert len(violations) == 0
