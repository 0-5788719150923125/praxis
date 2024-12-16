from typing import Dict, List, Optional, Tuple

import torch
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.constants import (
    BOE_ID,
    BOS_ID,
    BPE_ID,
    BYTE_UNITS,
    EOS_ID,
    PAD_ID,
)
from transformers import PreTrainedTokenizer


class ByteLevelTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer class that wraps the BltTokenizer while conforming to
    the HuggingFace PreTrainedTokenizer interface.
    """

    model_max_length = 2048  # Default max length for the model

    def __init__(
        self,
        bpe_tokenizer_path: str = None,
        vocab_size_unit_1: int = 256,  # Basic byte vocabulary
        bpe_delim: bool = False,
        add_bos: bool = True,
        add_eos: bool = True,
        **kwargs,
    ):
        # Store initialization parameters for later
        self._init_params = {
            "bpe_tokenizer_path": bpe_tokenizer_path,
            "vocab_size_unit_1": vocab_size_unit_1,
            "bpe_delim": bpe_delim,
            "add_bos": add_bos,
            "add_eos": add_eos,
        }

        # Flag to track initialization state
        self._is_fully_initialized = False

        # Define our default special tokens
        special_tokens = {
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
        }

        # Update with any provided tokens from kwargs
        for key, default_value in special_tokens.items():
            if key not in kwargs:
                kwargs[key] = default_value

        # Initialize parent class first
        super().__init__(**kwargs)

        # Now we can safely initialize the byte tokenizer
        self._tokenizer = BltTokenizer(
            vocab_size_unit_1=self._init_params["vocab_size_unit_1"],
            bpe_delim=self._init_params["bpe_delim"],
            bpe_tokenizer_path=self._init_params["bpe_tokenizer_path"],
            add_bos=False,  # We'll handle special tokens ourselves
            add_eos=False,
        )

        self._is_fully_initialized = True

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.
        """
        vocab = {}

        # Add special tokens first - they always exist
        for i, token in enumerate(
            [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
        ):
            vocab[token] = i

        # Add byte tokens only if fully initialized
        if self._is_fully_initialized:
            offset = len(vocab)
            for i in range(self._init_params["vocab_size_unit_1"]):
                vocab[str(i)] = i + offset

        return vocab

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of vocabulary.
        """
        special_tokens_count = len(
            [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
        )
        if self._is_fully_initialized:
            return self._init_params["vocab_size_unit_1"] + special_tokens_count
        return special_tokens_count

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string into a sequence of tokens.
        """
        byte_ids = self._tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )
        return [str(id) for id in byte_ids]

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token string to its ID.
        """
        vocab = self.get_vocab()
        return vocab.get(token, vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an ID to its string token representation.
        """
        vocab = self.get_vocab()
        for token, idx in vocab.items():
            if idx == index:
                return token
        return self.unk_token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens to a single string.
        """
        # Filter out special tokens
        byte_tokens = []
        for token in tokens:
            if token not in [
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.unk_token,
            ]:
                try:
                    byte_tokens.append(int(token))
                except ValueError:
                    continue

        return self._tokenizer.decode(byte_tokens)

    @property
    def initial_text(self) -> str:
        """
        Property to support common usage patterns.
        """
        return self.bos_token


def run_comprehensive_tests():
    """
    Comprehensive test suite for ByteLevelTokenizer covering common HuggingFace usage patterns.
    """
    print("Running comprehensive tokenizer tests...\n")

    # Initialize tokenizer with custom tokens
    tokenizer = ByteLevelTokenizer(
        bpe_delim=False,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
    )

    def test_special_tokens():
        print("1. Testing special token properties...")
        # Test direct property access
        assert tokenizer.bos_token == "<|bos|>", "BOS token mismatch"
        assert tokenizer.eos_token == "<|eos|>", "EOS token mismatch"
        assert tokenizer.pad_token == "<|pad|>", "PAD token mismatch"

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

        # Test pattern from example 1
        assert isinstance(tokenizer.initial_text, str), "Initial text property failed"

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
