from typing import Dict, List, Optional, Tuple

import torch
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from transformers import PreTrainedTokenizer


class ByteLevelTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer class that wraps the BltTokenizer while conforming to
    the HuggingFace PreTrainedTokenizer interface.
    """

    # Single source of truth for special tokens
    SPECIAL_TOKENS = {
        "boe_token": ("<|boe|>", 0),
        "pad_token": ("<|endoftext|>", 1),
        "bos_token": ("<|im_start|>", 2),
        "eos_token": ("<|im_end|>", 3),
        "bpe_token": ("<|bpe|>", 4),
    }

    def __init__(self, add_bos: bool = False, add_eos: bool = False, **kwargs):
        # Initialize special token attributes from SPECIAL_TOKENS
        for token_name, (token_value, _) in self.SPECIAL_TOKENS.items():
            if token_name not in kwargs:
                kwargs[token_name] = token_value

        # Store initialization parameters
        self.vocab_size_unit_1 = 256

        super().__init__(**kwargs)

        # Initialize the byte tokenizer
        self._tokenizer = BltTokenizer(
            vocab_size_unit_1=self.vocab_size_unit_1,
            bpe_delim=False,
            add_bos=False,  # We handle special tokens ourselves
            add_eos=False,
        )

        # Set special token IDs in the underlying tokenizer
        for token_name, (_, token_id) in self.SPECIAL_TOKENS.items():
            setattr(self._tokenizer, f"{token_name[:-6]}_id", token_id)

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary of token to index."""
        vocab = {
            token_value: token_id
            for _, (token_value, token_id) in self.SPECIAL_TOKENS.items()
        }

        # Add byte tokens after special tokens
        for i in range(self.vocab_size_unit_1):
            vocab[str(i)] = i + self._token_id_offset

        return vocab

    @property
    def vocab_size(self) -> int:
        """Returns the size of vocabulary."""
        return self.vocab_size_unit_1 + len(self.SPECIAL_TOKENS)

    @property
    def _token_id_offset(self) -> int:
        """Return the offset for regular token IDs after special tokens."""
        return max(id for _, id in self.SPECIAL_TOKENS.values()) + 1

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Converts a string into a sequence of tokens."""
        return self._tokenizer.encode(text, add_bos=False, add_eos=False)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token string to its ID."""
        # Check special tokens first
        for _, (token_value, token_id) in self.SPECIAL_TOKENS.items():
            if token == token_value:
                return token_id

        # Regular tokens are offset by special token count
        return int(token) + self._token_id_offset

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an ID to its string token representation."""
        # Check special token IDs
        for _, (token_value, token_id) in self.SPECIAL_TOKENS.items():
            if index == token_id:
                return token_value

        # Regular tokens
        return str(index - self._token_id_offset)

    def _is_special_token(self, token: str) -> bool:
        """Helper method to check if a token is special."""
        return any(
            token == special_token for special_token, _ in self.SPECIAL_TOKENS.values()
        )

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens to a single string."""
        byte_tokens = []
        for token in tokens:
            if not self._is_special_token(token):
                try:
                    byte_tokens.append(int(token))
                except ValueError:
                    continue

        decoded = self._tokenizer.decode(byte_tokens)
        return decoded if decoded else " "


def run_comprehensive_tests():
    """
    Comprehensive test suite for ByteLevelTokenizer covering common HuggingFace usage patterns.
    """
    print("Running comprehensive tokenizer tests...\n")

    # Initialize tokenizer with custom tokens
    tokenizer = ByteLevelTokenizer(
        bpe_delim=False,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )

    def test_special_tokens():
        print("1. Testing special token properties...")
        # Test direct property access
        assert tokenizer.bos_token == "<|im_start|>", "BOS token mismatch"
        assert tokenizer.eos_token == "<|im_end|>", "EOS token mismatch"
        assert tokenizer.pad_token == "<|endoftext|>", "PAD token mismatch"

        # Test special token IDs
        assert tokenizer.bos_token_id is not None, "BOS token ID is None"
        assert tokenizer.eos_token_id is not None, "EOS token ID is None"
        assert tokenizer.pad_token_id is not None, "PAD token ID is None"

        print("✓ Special token tests passed\n")

    def test_encode_decode():
        print("2. Testing encode/decode functionality...")

        test_text = "Hello, world!"

        # Test basic encode
        tokens = tokenizer.encode(test_text)
        assert isinstance(tokens, list), "Encode should return a list"

        # Test encode with tensor output
        tensor_tokens = tokenizer.encode(test_text, return_tensors="pt")
        assert isinstance(tensor_tokens, torch.Tensor), "Should return PyTorch tensor"

        # Test decode with and without special tokens
        decoded = tokenizer.decode(tokens)
        assert test_text in decoded, "Basic decode failed"

        decoded_no_special = tokenizer.decode(tokens, skip_special_tokens=True)
        assert test_text in decoded_no_special, "Decode without special tokens failed"

        print("✓ Encode/decode tests passed\n")

    def test_direct_call():
        print("3. Testing direct tokenizer call...")

        # Test single text
        output = tokenizer(
            text="Hello, world!",
            return_tensors="pt",
        )
        assert "input_ids" in output, "Missing input_ids in output"
        assert isinstance(
            output["input_ids"], torch.Tensor
        ), "input_ids should be tensor"

        # Test batch processing
        batch_output = tokenizer(
            text=["Hello, world!", "Another test"],
            padding=True,
            return_tensors="pt",
        )
        assert batch_output["input_ids"].shape[0] == 2, "Batch processing failed"

        print("✓ Direct call tests passed\n")

    def test_padding_truncation():
        print("4. Testing padding and truncation...")

        long_text = "This is a very long text that should be truncated." * 10

        # Test truncation
        output = tokenizer(
            text=long_text,
            max_length=20,
            truncation=True,
            return_tensors="pt",
        )
        assert output["input_ids"].shape[1] <= 20, "Truncation failed"

        # Test padding
        batch_output = tokenizer(
            text=["Short text", long_text],
            padding=True,
            max_length=20,
            truncation=True,
            return_tensors="pt",
        )
        assert batch_output["input_ids"].shape[1] == 20, "Padding failed"

        print("✓ Padding/truncation tests passed\n")

    def test_overflow_tokens():
        print("5. Testing overflow tokens...")

        # Test with return_overflowing_tokens
        output = tokenizer(
            text="Testing overflow tokens",
            max_length=5,
            stride=2,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        assert len(output["input_ids"].shape) == 2, "Overflow tokens handling failed"

        print("✓ Overflow tokens tests passed\n")

    def test_common_usage_patterns():
        print("6. Testing common usage patterns...")

        # Test pattern from example 2
        special_tokens = [
            tokenizer.bos_token,
            tokenizer.eos_token,
            "<|im_start|> user",
            "<|im_start|> assistant",
            "<|im_end|>",
        ]
        assert all(
            isinstance(t, str) for t in special_tokens
        ), "Special token access failed"

        # Test pattern from example 5
        interleaved = "Some interleaved text"
        tokens = tokenizer(
            text=interleaved,
            padding=False,
            return_tensors="pt",
        )[
            "input_ids"
        ].squeeze(0)
        assert isinstance(tokens, torch.Tensor), "Token tensor conversion failed"

        print("✓ Common usage pattern tests passed\n")

    # Run all tests
    try:
        test_special_tokens()
        test_encode_decode()
        test_direct_call()
        test_padding_truncation()
        test_overflow_tokens()
        test_common_usage_patterns()
        print("All tests passed successfully! 🎉")
    except AssertionError as e:
        print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    run_comprehensive_tests()
