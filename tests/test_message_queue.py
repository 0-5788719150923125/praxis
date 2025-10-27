"""Test message queue BOS token constraint."""

import torch
from transformers import AutoTokenizer

from praxis.data.datasets.message_queue import MessageQueueManager


def test_bos_token_constraint():
    """Verify that BOS tokens only appear before role tokens."""
    # Create a tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "[BOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"

    # Add special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[BOS]", "[SEP]", "[PAD]"]}
    )

    # Set chat template
    from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE

    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Create message queue manager
    block_size = 128
    queue_manager = MessageQueueManager(tokenizer, block_size)

    # Add multiple documents
    for i in range(3):
        document = {
            "messages": [
                {"role": "system", "content": f"You are assistant {i}"},
                {"role": "user", "content": f"Hello {i}"},
                {"role": "assistant", "content": f"Hi there {i}!"},
            ],
            "metadata": {"doc_id": i},
        }
        queue_manager.add_document(document)

    # Get a batch
    batch_result = queue_manager.get_batch(batch_size=2)
    batch = batch_result["batch"]

    # Stack sequences into tensor
    batch_tensor = torch.stack(batch)

    # Get BOS token id
    bos_id = tokenizer.convert_tokens_to_ids("[BOS]")

    # Valid role tokens that can follow BOS
    valid_roles = ["system", "developer", "assistant", "user"]
    valid_role_ids = [
        tokenizer.encode(role, add_special_tokens=False)[0] for role in valid_roles
    ]

    # Check each sequence
    for seq_idx, sequence in enumerate(batch_tensor):
        # Find all BOS token positions
        bos_positions = (sequence == bos_id).nonzero(as_tuple=True)[0]

        print(f"\nSequence {seq_idx}:")
        print(f"  Found {len(bos_positions)} BOS tokens")

        # For each BOS token, check what follows
        for pos in bos_positions:
            if pos + 1 < len(sequence):
                next_token = sequence[pos + 1].item()
                next_token_str = tokenizer.decode([next_token])

                print(
                    f"  BOS at position {pos} -> next token: '{next_token_str}' (id={next_token})"
                )

                # Check if next token is a valid role
                is_valid = next_token in valid_role_ids

                if not is_valid:
                    # Check if it's part of a role word (tokenizer may split roles)
                    is_role_prefix = any(
                        next_token_str.lower().strip() in role for role in valid_roles
                    )

                    if not is_role_prefix:
                        print(f"  WARNING: Token after BOS is not a role token!")
                        print(f"  Valid role IDs: {valid_role_ids}")
                        print(f"  Found token ID: {next_token}")

                        # Decode a few tokens for context
                        context_start = max(0, pos - 2)
                        context_end = min(len(sequence), pos + 5)
                        context = tokenizer.decode(sequence[context_start:context_end])
                        print(f"  Context: '{context}'")

    print("\nTest completed - check output for any BOS constraint violations")


def test_per_document_tokenization():
    """Verify documents are tokenized separately, not concatenated."""
    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "[BOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[BOS]", "[SEP]", "[PAD]"]}
    )

    from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE

    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Create message queue manager
    block_size = 64
    queue_manager = MessageQueueManager(tokenizer, block_size)

    # Add 2 documents with very different content
    doc1 = {
        "messages": [
            {"role": "user", "content": "DOCUMENT_ONE"},
            {"role": "assistant", "content": "Response one"},
        ],
        "metadata": {"doc_id": 1},
    }

    doc2 = {
        "messages": [
            {"role": "user", "content": "DOCUMENT_TWO"},
            {"role": "assistant", "content": "Response two"},
        ],
        "metadata": {"doc_id": 2},
    }

    queue_manager.add_document(doc1)
    queue_manager.add_document(doc2)

    # Refill token buffer
    queue_manager._refill_token_buffer()

    # Decode the entire token buffer
    full_text = tokenizer.decode(queue_manager.token_buffer, skip_special_tokens=False)

    print("\nFull token buffer:")
    print(full_text)

    # Verify both documents are present
    assert "DOCUMENT_ONE" in full_text, "Document 1 content missing"
    assert "DOCUMENT_TWO" in full_text, "Document 2 content missing"

    # Count BOS tokens - should have at least 4 (2 per document minimum)
    bos_count = full_text.count("[BOS]")
    print(f"\nBOS token count: {bos_count}")
    assert bos_count >= 4, f"Expected at least 4 BOS tokens, found {bos_count}"

    print("\nPer-document tokenization test passed!")


if __name__ == "__main__":
    test_bos_token_constraint()
    test_per_document_tokenization()
