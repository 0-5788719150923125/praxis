import itertools
from typing import List

import pytest
import torch

from praxis.tokenizer_praxis import ByteLevelTokenizer

TOKENIZER_TYPES = [ByteLevelTokenizer]


@pytest.fixture(params=TOKENIZER_TYPES)
def tokenizer_setup(request):
    tokenizer = request.param
    return tokenizer


def test_tokenizer_full(tokenizer_setup) -> None:
    """Comprehensive test suite for ByteLevelTokenizer with enhanced special token testing."""
    print("Running comprehensive tokenizer tests...\n")

    tokenizer = tokenizer_setup()

    def test_special_token_preservation():
        print("1. Testing special token preservation...")

        # Test encoding with special tokens
        text = f"{tokenizer.bos_token}Hello{tokenizer.eos_token}"
        tokens = tokenizer.tokenize(text)
        assert tokenizer.bos_token in tokens, "BOS token lost during tokenization"
        assert tokenizer.eos_token in tokens, "EOS token lost during tokenization"

        # Test encoding and decoding roundtrip
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        assert text == decoded, f"Roundtrip failed: {text} != {decoded}"

        # Test multiple special tokens
        text = f"{tokenizer.bos_token}Hello{tokenizer.eos_token}{tokenizer.pad_token}"
        decoded = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
        assert text == decoded, "Multiple special tokens not preserved"

        print("✓ Special token preservation tests passed\n")

    def test_mixed_content():
        print("2. Testing mixed special tokens and regular text...")

        # Test mixed content
        test_cases = [
            f"{tokenizer.bos_token}Hello",
            f"Hello{tokenizer.eos_token}",
            f"{tokenizer.bos_token}Hello{tokenizer.eos_token}",
            f"{tokenizer.bos_token}Hello World{tokenizer.eos_token}",
        ]

        for text in test_cases:
            tokens = tokenizer.tokenize(text)
            decoded = tokenizer.convert_tokens_to_string(tokens)
            assert text == decoded, f"Mixed content failed: {text} != {decoded}"

        print("✓ Mixed content tests passed\n")

    def test_batch_processing():
        print("3. Testing batch processing with special tokens...")

        batch_texts = [
            f"{tokenizer.bos_token}Hello{tokenizer.eos_token}",
            f"{tokenizer.bos_token}World{tokenizer.eos_token}",
        ]

        # Test batch encoding
        batch_encoded = tokenizer(batch_texts, padding=True, return_tensors="pt")

        # Decode each sequence
        for i, text in enumerate(batch_texts):
            decoded = tokenizer.decode(batch_encoded["input_ids"][i])
            assert text in decoded, f"Batch processing failed for: {text}"

        print("✓ Batch processing tests passed\n")

    def test_special_token_positioning():
        print("4. Testing special token positioning...")

        # Test special tokens at different positions
        text = f"Hello{tokenizer.eos_token}World"
        tokens = tokenizer.tokenize(text)
        decoded = tokenizer.convert_tokens_to_string(tokens)
        assert text == decoded, "Mid-sequence special token failed"

        text = f"{tokenizer.bos_token}{tokenizer.eos_token}Hello"
        tokens = tokenizer.tokenize(text)
        decoded = tokenizer.convert_tokens_to_string(tokens)
        assert text == decoded, "Adjacent special tokens failed"

        print("✓ Special token positioning tests passed\n")

    def test_add_special_tokens_flag():
        print("5. Testing add_special_tokens flag...")

        # Create tokenizer instances with different settings
        tokenizer_with_special = tokenizer_setup()

        test_text = "Hello, world!"
        print(f"\nTest setup:")
        print(f"- Input text: {repr(test_text)}")

        # Test with add_special_tokens=True
        print("\nTesting add_special_tokens=True:")
        encoded_with = tokenizer_with_special.encode(test_text, add_special_tokens=True)
        decoded_with = tokenizer_with_special.decode(encoded_with)
        print(f"- Encoded tokens: {encoded_with}")
        print(f"- Decoded text: {repr(decoded_with)}")
        print(f"- BOS token: {repr(tokenizer_with_special.bos_token)}")
        print(f"- EOS token: {repr(tokenizer_with_special.eos_token)}")

        assert (
            tokenizer_with_special.bos_token in decoded_with
        ), "BOS token not added when requested"
        assert (
            tokenizer_with_special.eos_token in decoded_with
        ), "EOS token not added when requested"

        # Test with add_special_tokens=False
        encoded_without = tokenizer_with_special.encode(
            test_text, add_special_tokens=False
        )
        decoded_without = tokenizer_with_special.decode(encoded_without)
        assert (
            tokenizer_with_special.bos_token not in decoded_without
        ), "BOS token added when not requested"
        assert (
            tokenizer_with_special.eos_token not in decoded_without
        ), "EOS token added when not requested"

        # Test with text already containing special tokens
        text_with_special = f"{tokenizer_with_special.bos_token}{test_text}{tokenizer_with_special.eos_token}"
        encoded_existing = tokenizer_with_special.encode(
            text_with_special, add_special_tokens=False
        )
        decoded_existing = tokenizer_with_special.decode(encoded_existing)
        assert (
            decoded_existing == text_with_special
        ), "Existing special tokens not preserved"

        # Test that add_special_tokens=True doesn't duplicate tokens
        encoded_no_duplicate = tokenizer_with_special.encode(
            text_with_special, add_special_tokens=True
        )
        decoded_no_duplicate = tokenizer_with_special.decode(encoded_no_duplicate)
        assert (
            decoded_no_duplicate.count(tokenizer_with_special.bos_token) == 1
        ), "BOS token duplicated"
        assert (
            decoded_no_duplicate.count(tokenizer_with_special.eos_token) == 1
        ), "EOS token duplicated"

        print("✓ add_special_tokens flag tests passed\n")

    # Run all tests
    test_special_token_preservation()
    test_mixed_content()
    test_batch_processing()
    test_special_token_positioning()
    # test_add_special_tokens_flag()
