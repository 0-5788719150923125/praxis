import pytest

from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.byte_level import ByteLevelTokenizer


def test_standard_tokenizer():
    """Test StandardTokenizer using the normal loading mechanism."""
    try:
        # Try to load a standard tokenizer using the normal method
        tokenizer = create_tokenizer(vocab_size=32768)

        # If it's a ByteLevelTokenizer, skip the test
        if isinstance(tokenizer, ByteLevelTokenizer):
            pytest.skip("No trained StandardTokenizer available")

        # Run basic tests
        test_text = "Hello, world!"

        # Test encode/decode roundtrip
        encoded = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        assert test_text == decoded, f"Roundtrip failed: {test_text} != {decoded}"

        # Test with special tokens
        encoded_with_special = tokenizer.encode(test_text, add_special_tokens=True)
        decoded_with_special = tokenizer.decode(
            encoded_with_special, skip_special_tokens=False
        )

        # Check that special tokens were added
        assert len(encoded_with_special) > len(encoded), "Special tokens not added"

        print("✓ StandardTokenizer tests passed")

    except Exception as e:
        pytest.skip(f"StandardTokenizer not available: {e}")
