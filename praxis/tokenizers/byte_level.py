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

    # Only 4 unique special tokens (PAD/BOE share ID 0)
    # Mapping to match ByteLatent constants:
    # PAD_ID/BOE_ID = 0, BOS_ID = 1, EOS_ID = 2, SEP_ID = 3
    SPECIAL_TOKENS = {
        "pad_token": "[PAD]",  # ID: 256 (0 after byte offset)
        "bos_token": "[BOS]",  # ID: 257 (1 after byte offset)
        "eos_token": "[EOS]",  # ID: 258 (2 after byte offset)
        "sep_token": "[SEP]",  # ID: 259 (3 after byte offset)
        # UNK is not used in ByteLatent - removed to keep vocab at 260
    }

    def __init__(
        self,
        tokenizer_object: Optional[Tokenizer] = None,
        vocab_size_unit_1: int = 256,
        bpe_delim: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
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
        Create a tokenizer object for byte-level encoding using HuggingFace components.

        Args:
            vocab_size: Size of the byte vocabulary

        Returns:
            Tokenizer object configured for byte-level encoding
        """
        # Use BPE model with empty merges for byte-level tokenization
        vocab = {}
        merges = []

        # Add byte tokens (0-255)
        for i in range(vocab_size):
            # Use the same format as HuggingFace ByteLevel
            byte_char = chr(i) if i < 256 else f"<unk>"
            vocab[byte_char] = i

        # Add special tokens right after byte tokens
        for i, (token_name, token_value) in enumerate(self.SPECIAL_TOKENS.items()):
            vocab[token_value] = vocab_size + i

        # Create BPE model with vocab and empty merges
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=True))

        # Set up byte-level components
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        return tokenizer

    def train(self, texts, vocab_size: int = 32768, **kwargs) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Note: ByteLevel tokenizer doesn't need training as it operates
        on raw bytes. This method is a no-op but provided for interface
        compatibility.
        """
        pass  # ByteLevel tokenizer doesn't need training

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size_unit_1 + len(self.SPECIAL_TOKENS)  # 256 + 4 = 260