from typing import Dict, List, Optional, Tuple

import torch
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from transformers import PreTrainedTokenizer


class ByteLevelTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer class that wraps the BltTokenizer while conforming to
    the HuggingFace PreTrainedTokenizer interface.
    """

    def __init__(
        self,
        bpe_tokenizer_path: str = None,
        vocab_size_unit_1: int = 256,  # Basic byte vocabulary
        bpe_delim: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
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

        # Define our special tokens with specific IDs
        # IDs 0-3 reserved for special tokens
        self._special_token_ids = {
            "boe_token": 0,
            "pad_token": 1,
            "bos_token": 2,
            "eos_token": 3,
            "bpe_token": 4,
        }

        # Define our default special tokens
        special_tokens = {
            "boe_token": "<|boe|>",
            "pad_token": "<|pad|>",
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "bpe_token": "<|bpe|>",
        }

        # Update with any provided tokens from kwargs
        for key, default_value in special_tokens.items():
            if key not in kwargs:
                kwargs[key] = default_value

        self.boe_token = special_tokens["boe_token"]
        self.bpe_token = special_tokens["bpe_token"]

        # Initialize parent class first
        super().__init__(**kwargs)

        # Now we can safely initialize the byte tokenizer with our special token IDs
        self._tokenizer = BltTokenizer(
            vocab_size_unit_1=self._init_params["vocab_size_unit_1"],
            bpe_delim=self._init_params["bpe_delim"],
            bpe_tokenizer_path=self._init_params["bpe_tokenizer_path"],
            add_bos=False,  # We'll handle special tokens ourselves
            add_eos=False,
        )

        # Override BltTokenizer's special token IDs with our own
        self._tokenizer.bos_id = self._special_token_ids["bos_token"]
        self._tokenizer.eos_id = self._special_token_ids["eos_token"]
        self._tokenizer.pad_id = self._special_token_ids["pad_token"]
        self._tokenizer.boe_id = self._special_token_ids["boe_token"]
        self._tokenizer.bpe_id = self._special_token_ids["bpe_token"]
        # Note: BOE and BPE IDs aren't used in our implementation

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.
        """
        vocab = {}

        # Add special tokens first with their predefined IDs
        vocab[self.bos_token] = self._special_token_ids["bos_token"]
        vocab[self.eos_token] = self._special_token_ids["eos_token"]
        vocab[self.pad_token] = self._special_token_ids["pad_token"]
        vocab[self.boe_token] = self._special_token_ids["boe_token"]
        vocab[self.bpe_token] = self._special_token_ids["bpe_token"]

        # Add byte tokens, starting byte token IDs after special tokens
        offset = max(self._special_token_ids.values()) + 1
        for i in range(self._init_params["vocab_size_unit_1"]):
            vocab[str(i)] = i + offset

        return vocab

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of vocabulary.
        """
        special_tokens_count = len(
            [
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.boe_token,
                self.bpe_token,
            ]
        )
        return self._init_params["vocab_size_unit_1"] + special_tokens_count

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string into a sequence of tokens.
        """
        byte_ids = self._tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )
        return byte_ids

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token string to its ID.
        """
        # Handle special tokens using our predefined IDs
        special_tokens_map = {
            self.bos_token: self._special_token_ids["bos_token"],
            self.eos_token: self._special_token_ids["eos_token"],
            self.pad_token: self._special_token_ids["pad_token"],
            self.boe_token: self._special_token_ids["boe_token"],
            self.bpe_token: self._special_token_ids["bpe_token"],
        }

        if token in special_tokens_map:
            return special_tokens_map[token]

        # For regular tokens, they are already string representations of IDs
        return int(token) + max(self._special_token_ids.values()) + 1

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an ID to its string token representation.
        """
        # Handle special token IDs
        id_to_special_token = {v: k for k, v in self._special_token_ids.items()}
        if index in id_to_special_token:
            return getattr(self, id_to_special_token[index])

        # For regular tokens, adjust the index back to byte value
        adjusted_index = index - max(self._special_token_ids.values()) - 1
        return str(adjusted_index)

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
                self.boe_token,
                self.bpe_token,
            ]:
                try:
                    byte_tokens.append(int(token))
                except ValueError:
                    continue

        decoded = self._tokenizer.decode(byte_tokens)
        if len(decoded) == 0:
            decoded = " "
        return decoded


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
