"""ByteLevel tokenizer implementation for Praxis."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from jinja2 import Template

from .base import PraxisTokenizerBase
from .chat_templates import get_chat_template

# Try to import BltTokenizer, but make it optional
try:
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    HAS_BLT = True
except ImportError:
    HAS_BLT = False
    BltTokenizer = None


class ByteLevelTokenizer(PraxisTokenizerBase):
    """
    ByteLevel tokenizer that operates on raw bytes.
    
    This tokenizer uses the BltTokenizer when available, providing
    byte-level tokenization with special token support.
    """
    
    SPECIAL_TOKENS = {
        "boe_token": ("[BOE]", 0),
        "pad_token": ("[PAD]", 0),  # Shares ID with BOE
        "bos_token": ("[BOS]", 1),
        "eos_token": ("[EOS]", 2),
        "sep_token": ("[SEP]", 3),
        "bpe_token": ("[BPE]", 4),
        "unk_token": ("[UNK]", 5),
    }
    
    def __init__(
        self,
        vocab_size_unit_1: int = 256,
        bpe_delim: bool = False,
        add_bos: bool = False,
        add_eos: bool = False,
        chat_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the ByteLevel tokenizer.
        
        Args:
            vocab_size_unit_1: Size of the byte vocabulary (default 256)
            bpe_delim: Whether to use BPE delimiter
            add_bos: Whether to add BOS token automatically
            add_eos: Whether to add EOS token automatically
            chat_template: Chat template for conversation formatting
            **kwargs: Additional arguments passed to parent class
        """
        if not HAS_BLT:
            raise ImportError(
                "ByteLevelTokenizer requires bytelatent package. "
                "Please install it with: pip install bytelatent"
            )
        
        # Initialize special tokens from kwargs or use defaults
        for token_name, (token_value, _) in self.SPECIAL_TOKENS.items():
            if token_name not in kwargs:
                kwargs[token_name] = token_value
        
        self.vocab_size_unit_1 = vocab_size_unit_1
        
        # Calculate total vocab size
        vocab_size = vocab_size_unit_1 + len(self.SPECIAL_TOKENS)
        
        # Initialize parent
        super().__init__(
            vocab_size=vocab_size,
            chat_template=chat_template,
            **kwargs
        )
        
        # Initialize the underlying BltTokenizer
        self._tokenizer = BltTokenizer(
            vocab_size_unit_1=self.vocab_size_unit_1,
            bpe_delim=bpe_delim,
            add_bos=add_bos,
            add_eos=add_eos,
        )
        
        # Set special token IDs on the underlying tokenizer
        for token_name, (_, token_id) in self.SPECIAL_TOKENS.items():
            setattr(self._tokenizer, f"{token_name[:-6]}_id", token_id)
        
        # Create reverse mapping for special tokens
        self._id_to_special_token = {
            token_id: token_value
            for _, (token_value, token_id) in self.SPECIAL_TOKENS.items()
        }
        
        # Set default chat template if not provided
        if self.chat_template is None:
            self.chat_template = get_chat_template("byte_level")
    
    def train(
        self,
        texts: Union[List[str], Any],
        vocab_size: int = 32768,
        **kwargs
    ) -> None:
        """
        Train the tokenizer on a corpus of texts.
        
        Note: ByteLevel tokenizer doesn't need training as it operates
        on raw bytes. This method is a no-op but provided for interface
        compatibility.
        
        Args:
            texts: Training texts (ignored)
            vocab_size: Target vocabulary size (ignored)
            **kwargs: Additional parameters (ignored)
        """
        pass  # ByteLevel tokenizer doesn't need training
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a string into a list of tokens.
        
        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters
            
        Returns:
            List of token strings
        """
        # Check if the text is a special token
        if text in [token for _, (token, _) in self.SPECIAL_TOKENS.items()]:
            return [text]
        
        # Use byte tokenization
        byte_tokens = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        return [
            str(token - self._tokenizer.offsetting_special_char)
            for token in byte_tokens
        ]
    
    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert a token string to its ID.
        
        Args:
            token: Token string
            
        Returns:
            Token ID
        """
        # Check special tokens first
        for _, (token_value, token_id) in self.SPECIAL_TOKENS.items():
            if token == token_value:
                return token_id
        
        # Regular tokens are offset by special token count
        return int(token) + self._token_id_offset
    
    def _convert_id_to_token(self, index: int) -> str:
        """
        Convert a token ID to its string representation.
        
        Args:
            index: Token ID
            
        Returns:
            Token string
        """
        # Check if it's a special token ID
        if index in self._id_to_special_token:
            return self._id_to_special_token[index]
        
        # Regular tokens
        return str(index - self._token_id_offset)
    
    @property
    def _token_id_offset(self) -> int:
        """Get the offset for regular token IDs."""
        return max(id for _, id in self.SPECIAL_TOKENS.values()) + 1
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Return the vocabulary as a dictionary.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        vocab = {
            token_value: token_id
            for _, (token_value, token_id) in self.SPECIAL_TOKENS.items()
        }
        
        for i in range(self.vocab_size_unit_1):
            vocab[str(i)] = i + self._token_id_offset
        
        return vocab
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save the tokenizer vocabulary to a directory.
        
        Args:
            save_directory: Directory to save vocabulary files
            filename_prefix: Optional prefix for filenames
            
        Returns:
            Tuple of saved file paths
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer config
        config_file = save_directory / "tokenizer_config.json"
        config = {
            "tokenizer_class": self.__class__.__name__,
            "vocab_size_unit_1": self.vocab_size_unit_1,
            "special_tokens": self.SPECIAL_TOKENS,
            "chat_template": self.chat_template,
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        return (str(config_file),)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        **kwargs
    ) -> "ByteLevelTokenizer":
        """
        Load a pretrained tokenizer.
        
        Args:
            pretrained_model_name_or_path: Path to tokenizer directory
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded tokenizer instance
        """
        path = Path(pretrained_model_name_or_path)
        config_file = path / "tokenizer_config.json"
        
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            
            # Override with any provided kwargs
            config.update(kwargs)
            
            return cls(**config)
        else:
            # Default initialization
            return cls(**kwargs)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a sequence of tokens to a single string.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Decoded string
        """
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
                # Add the special token directly
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
    
    def _is_special_token(self, token: str) -> bool:
        """Check if a token is a special token."""
        return any(
            token == special_token for special_token, _ in self.SPECIAL_TOKENS.values()
        )