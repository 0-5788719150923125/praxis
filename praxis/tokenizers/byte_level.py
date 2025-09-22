"""ByteLevel tokenizer implementation for Praxis with HuggingFace compatibility."""

from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from .base import PraxisTokenizerBase
from .chat_templates import get_chat_template

# Try to import BltTokenizer, but make it optional
try:
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    HAS_BLT = True
except ImportError:
    HAS_BLT = False
    BltTokenizer = None

# Import byte constants from our local module
try:
    from praxis.encoders.byte_latent.constants import OFFSET
except ImportError:
    # Fallback if the encoder module isn't available
    OFFSET = 6


class ByteLevelTokenizer(PreTrainedTokenizerFast):
    """
    ByteLevel tokenizer that operates on raw bytes.

    This tokenizer provides byte-level tokenization with full HuggingFace
    compatibility including chat template support.
    """

    # Only 4 unique special tokens matching BLT IDs exactly
    # PAD_ID/BOE_ID = 0, BOS_ID = 1, EOS_ID = 2, SEP_ID = 3
    SPECIAL_TOKENS = {
        "pad_token": "[PAD]",  # ID: 0
        "bos_token": "[BOS]",  # ID: 1
        "eos_token": "[EOS]",  # ID: 2
        "sep_token": "[SEP]",  # ID: 3
        # UNK is not used in ByteLatent
    }

    # Special token IDs matching BLT constants
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    SEP_ID = 3

    def __init__(
        self,
        tokenizer_object: Optional[Tokenizer] = None,
        vocab_size_unit_1: int = 256,
        bpe_delim: bool = False,
        add_bos: bool = True,  # Changed default to True
        add_eos: bool = True,  # Changed default to True
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the ByteLevel tokenizer.

        Args:
            tokenizer_object: Pre-built tokenizer object
            vocab_size_unit_1: Size of the byte vocabulary (default 256)
            bpe_delim: Whether to use BPE delimiter (requires BLT)
            add_bos: Whether to add BOS token automatically
            add_eos: Whether to add EOS token automatically
            chat_template: Chat template for conversation formatting
            **kwargs: Additional arguments passed to parent class
        """
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.bpe_delim = bpe_delim
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Initialize special tokens from kwargs or use defaults
        for token_name, token_value in self.SPECIAL_TOKENS.items():
            if token_name not in kwargs:
                kwargs[token_name] = token_value

        # HuggingFace expects unk_token - use PAD as fallback
        if "unk_token" not in kwargs:
            kwargs["unk_token"] = "[PAD]"  # Use PAD as unknown token

        # Create or use provided tokenizer object
        if tokenizer_object is None:
            tokenizer_object = self._create_tokenizer(vocab_size_unit_1)

        # Initialize PreTrainedTokenizerFast (this gives us apply_chat_template!)
        PreTrainedTokenizerFast.__init__(
            self, tokenizer_object=tokenizer_object, **kwargs
        )

        # Set chat template
        if chat_template is None:
            self.chat_template = get_chat_template("byte_level")
        else:
            self.chat_template = chat_template

        # Legacy BLT tokenizer for compatibility (if needed)
        self._blt_tokenizer = None
        if bpe_delim and HAS_BLT:
            self._blt_tokenizer = BltTokenizer(
                vocab_size_unit_1=vocab_size_unit_1,
                bpe_delim=bpe_delim,
                add_bos=add_bos,
                add_eos=add_eos,
            )

    def _create_tokenizer(self, vocab_size: int) -> Tokenizer:
        """
        Create a tokenizer object for BLT-compatible byte-level encoding.

        Args:
            vocab_size: Size of the byte vocabulary

        Returns:
            Tokenizer object configured for byte-level encoding
        """
        # Build vocab with BLT-compatible IDs
        vocab = {}
        merges = []

        # Special tokens use IDs 0-3
        vocab["[PAD]"] = self.PAD_ID
        vocab["[BOS]"] = self.BOS_ID
        vocab["[EOS]"] = self.EOS_ID
        vocab["[SEP]"] = self.SEP_ID

        # Byte tokens start at OFFSET (4)
        for i in range(vocab_size):
            byte_char = chr(i) if i < 256 else f"<unk>"
            vocab[byte_char] = i + OFFSET

        # Create BPE model with vocab and empty merges
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=False))

        # Use simple pre-tokenizer that preserves everything
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
        tokenizer.decoder = decoders.Sequence([])

        return tokenizer

    def train(self, texts, vocab_size: int = 32768, **kwargs) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Note: ByteLevel tokenizer doesn't need training as it operates
        on raw bytes. This method is a no-op but provided for interface
        compatibility.
        """
        pass  # ByteLevel tokenizer doesn't need training

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Encode text to token IDs using BLT-compatible offset system.

        Bytes are offset by OFFSET (4), special tokens use IDs 0-3.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            **kwargs: Additional arguments

        Returns:
            List of token IDs
        """
        # For testing: handle special token strings in text
        # This allows tests to pass "[BOS]Hello[EOS]" and have it work
        special_token_map = {
            "[PAD]": self.PAD_ID,
            "[BOS]": self.BOS_ID,
            "[EOS]": self.EOS_ID,
            "[SEP]": self.SEP_ID,
        }

        tokens = []
        i = 0
        while i < len(text):
            # Check if we're at a special token string
            found_special = False
            for token_str, token_id in special_token_map.items():
                if text[i:i+len(token_str)] == token_str:
                    tokens.append(token_id)
                    i += len(token_str)
                    found_special = True
                    break

            if not found_special:
                # Regular character - convert to byte and add offset
                byte_val = ord(text[i]) if ord(text[i]) < 256 else ord('?')
                tokens.append(byte_val + OFFSET)
                i += 1

        # Add BOS/EOS if requested (and not already present)
        if add_special_tokens:
            if self.add_bos and (not tokens or tokens[0] != self.BOS_ID):
                tokens.insert(0, self.BOS_ID)
            if self.add_eos and (not tokens or tokens[-1] != self.EOS_ID):
                tokens.append(self.EOS_ID)

        return tokens

    def decode(self, token_ids: Union[List[int], int], skip_special_tokens: bool = False, **kwargs) -> str:
        """
        Decode token IDs back to text using BLT-compatible offset system.

        For testing compatibility, special tokens are decoded to their string form.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments

        Returns:
            Decoded text
        """
        # Handle tensors
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]

        # Special token ID to string mapping (for testing)
        special_id_map = {
            self.PAD_ID: "[PAD]",
            self.BOS_ID: "[BOS]",
            self.EOS_ID: "[EOS]",
            self.SEP_ID: "[SEP]",
        }

        result = []
        for tok in token_ids:
            # Check if it's a special token (0-3)
            if tok in special_id_map:
                if not skip_special_tokens:
                    result.append(special_id_map[tok])
            else:
                # Regular byte token - reverse the offset
                byte_val = tok - OFFSET
                if 0 <= byte_val < 256:
                    result.append(chr(byte_val))

        return ''.join(result)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text into a list of token strings.

        Args:
            text: Text to tokenize
            **kwargs: Additional arguments

        Returns:
            List of token strings
        """
        token_ids = self.encode(text, add_special_tokens=False)

        # Map special token IDs to strings
        special_id_map = {
            self.PAD_ID: "[PAD]",
            self.BOS_ID: "[BOS]",
            self.EOS_ID: "[EOS]",
            self.SEP_ID: "[SEP]",
        }

        tokens = []
        for token_id in token_ids:
            if token_id in special_id_map:
                tokens.append(special_id_map[token_id])
            else:
                # Regular byte token - reverse the offset to get char
                byte_val = token_id - OFFSET
                if 0 <= byte_val < 256:
                    tokens.append(chr(byte_val))

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens back to a string.

        Args:
            tokens: List of token strings

        Returns:
            Reconstructed string
        """
        return ''.join(tokens)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size (matching BLT: 256 + offset)."""
        return self.vocab_size_unit_1 + OFFSET  # 256 + 4 = 260