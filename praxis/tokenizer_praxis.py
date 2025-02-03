from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.constants import (
    BOE_ID,
    BOS_ID,
    BPE_ID,
    BYTE_UNITS,
    EOS_ID,
    OFFSET,
    PAD_ID,
)
from transformers import PreTrainedTokenizer


class ByteLevelTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer class that wraps the BltTokenizer while conforming to
    the HuggingFace PreTrainedTokenizer interface.
    """

    SPECIAL_TOKENS = {
        "boe_token": ("<|boe|>", BOE_ID),
        # "pad_token": ("<|endoftext|>", PAD_ID),
        "pad_token": ("<|endoftext|>", 0),
        "bos_token": ("<|im_start|>", BOS_ID),
        "eos_token": ("<|im_end|>", EOS_ID),
        "bpe_token": ("<|bpe|>", BPE_ID),
    }

    def __init__(self, **kwargs: Any) -> None:
        for token_name, (token_value, _) in self.SPECIAL_TOKENS.items():
            if token_name not in kwargs:
                kwargs[token_name] = token_value

        self.vocab_size_unit_1 = 256
        super().__init__(**kwargs)

        self._tokenizer = BltTokenizer(
            vocab_size_unit_1=self.vocab_size_unit_1,
            bpe_delim=False,
            add_bos=False,  # We handle special tokens ourselves
            add_eos=False,
        )

        for token_name, (_, token_id) in self.SPECIAL_TOKENS.items():
            setattr(self._tokenizer, f"{token_name[:-6]}_id", token_id)

        # Create reverse mapping for special tokens
        self._id_to_special_token = {
            token_id: token_value
            for _, (token_value, token_id) in self.SPECIAL_TOKENS.items()
        }

    def _special_tokens_present(self, text: str) -> bool:
        """Check if special tokens already exist in the text."""
        return any(
            token_value in text for _, (token_value, _) in self.SPECIAL_TOKENS.items()
        )

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> List[int]:
        """Override encode to properly handle add_special_tokens flag."""

        # Check if special tokens already exist
        # has_special = self._special_tokens_present(text)

        # Basic tokenization
        tokens = self._tokenize(text, **kwargs)

        token_ids = self.convert_tokens_to_ids(tokens)

        # Return with special tokens if requested AND not already present
        # if add_special_tokens and not has_special:
        #     result = [self.bos_token_id] + token_ids + [self.eos_token_id]
        #     return result

        return token_ids

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Minimal implementation that just preserves existing special tokens."""
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Converts a string into a sequence of tokens."""
        # First check if the text is a special token
        if text in [token for _, (token, _) in self.SPECIAL_TOKENS.items()]:
            return [text]

        # Otherwise, use byte tokenization
        byte_tokens = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        return [
            str(token - self._tokenizer.offsetting_special_char)
            for token in byte_tokens
        ]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens to a single string."""
        byte_tokens = []
        result = ""

        for token in tokens:
            if self._is_special_token(token):
                # If we have accumulated byte tokens, decode them first
                if byte_tokens:
                    decoded = self._tokenizer.decode(
                        [
                            int(t) + self._tokenizer.offsetting_special_char
                            for t in byte_tokens
                        ]
                    )
                    if decoded:
                        result += decoded
                    byte_tokens = []
                # Add the special token directly - no spaces
                result += token
            else:
                byte_tokens.append(token)

        # Handle any remaining byte tokens
        if byte_tokens:
            decoded = self._tokenizer.decode(
                [int(t) + self._tokenizer.offsetting_special_char for t in byte_tokens]
            )
            if decoded:
                result += decoded

        return result if result else " "

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary of token to index."""
        vocab = {
            token_value: token_id
            for _, (token_value, token_id) in self.SPECIAL_TOKENS.items()
        }

        for i in range(self.vocab_size_unit_1):
            vocab[str(i)] = i + self._token_id_offset

        return vocab

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_unit_1 + len(self.SPECIAL_TOKENS)

    @property
    def _token_id_offset(self) -> int:
        return max(id for _, id in self.SPECIAL_TOKENS.values()) + 1

    def _convert_token_to_id(self, token: str) -> int:
        # Check special tokens first
        for _, (token_value, token_id) in self.SPECIAL_TOKENS.items():
            if token == token_value:
                return token_id
        # Regular tokens are offset by special token count
        return int(token) + self._token_id_offset

    def _convert_id_to_token(self, index: int) -> str:
        # Check if it's a special token ID
        if index in self._id_to_special_token:
            return self._id_to_special_token[index]
        # Regular tokens
        return str(index - self._token_id_offset)

    def _is_special_token(self, token: str) -> bool:
        return any(
            token == special_token for special_token, _ in self.SPECIAL_TOKENS.values()
        )


def run_comprehensive_tests() -> None:
    """Comprehensive test suite for ByteLevelTokenizer with enhanced special token testing."""
    print("Running comprehensive tokenizer tests...\n")

    tokenizer = ByteLevelTokenizer(
        bpe_delim=False,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )

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
        tokenizer_with_special = ByteLevelTokenizer()

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
    try:
        test_special_token_preservation()
        test_mixed_content()
        test_batch_processing()
        test_special_token_positioning()
        test_add_special_tokens_flag()
        print("All tests passed successfully! 🎉")
    except AssertionError as e:
        print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    run_comprehensive_tests()
