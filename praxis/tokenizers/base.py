"""Base tokenizer class and utilities for Praxis."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class PraxisTokenizerBase(PreTrainedTokenizer, ABC):
    """
    Abstract base class for all Praxis tokenizers.

    This class provides a unified interface for different tokenizer types
    while maintaining compatibility with HuggingFace transformers.
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        model_max_length: int = 2048,
        padding_side: str = "left",
        truncation_side: str = "right",
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the base tokenizer.

        Args:
            vocab_size: Size of the vocabulary
            model_max_length: Maximum sequence length
            padding_side: Side to pad sequences ("left" or "right")
            truncation_side: Side to truncate sequences ("left" or "right")
            chat_template: Jinja2 template for chat formatting
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            model_max_length=model_max_length,
            padding_side=padding_side,
            truncation_side=truncation_side,
            **kwargs,
        )

        self._vocab_size = vocab_size
        self.chat_template = chat_template

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        if self._vocab_size is not None:
            return self._vocab_size
        return len(self.get_vocab())

    @abstractmethod
    def train(
        self, texts: Union[List[str], Any], vocab_size: int = 32768, **kwargs
    ) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Args:
            texts: Training texts or iterator
            vocab_size: Target vocabulary size
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a string into a list of tokens.

        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters

        Returns:
            List of token strings
        """
        pass

    @abstractmethod
    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert a token string to its ID.

        Args:
            token: Token string

        Returns:
            Token ID
        """
        pass

    @abstractmethod
    def _convert_id_to_token(self, index: int) -> str:
        """
        Convert a token ID to its string representation.

        Args:
            index: Token ID

        Returns:
            Token string
        """
        pass

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """
        Return the vocabulary as a dictionary.

        Returns:
            Dictionary mapping tokens to IDs
        """
        pass

    @abstractmethod
    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save the tokenizer vocabulary to a directory.

        Args:
            save_directory: Directory to save vocabulary files
            filename_prefix: Optional prefix for filenames

        Returns:
            Tuple of saved file paths
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, Path], **kwargs
    ) -> "PraxisTokenizerBase":
        """
        Load a pretrained tokenizer.

        Args:
            pretrained_model_name_or_path: Path or model identifier
            **kwargs: Additional loading parameters

        Returns:
            Loaded tokenizer instance
        """
        pass

    def prepare_for_training(self, use_special_tokens: bool = True) -> None:
        """
        Prepare the tokenizer for training a model.

        Args:
            use_special_tokens: Whether to use special tokens
        """
        if use_special_tokens:
            special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]
            existing_special_tokens = self.special_tokens_map.values()
            tokens_to_add = [
                t for t in special_tokens if t not in existing_special_tokens
            ]
            if tokens_to_add:
                self.add_special_tokens({"additional_special_tokens": tokens_to_add})

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Get a mask indicating special tokens.

        Args:
            token_ids_0: First sequence token IDs
            token_ids_1: Optional second sequence token IDs
            already_has_special_tokens: Whether special tokens are already added

        Returns:
            List of 0s and 1s indicating special tokens
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        # Default implementation
        if token_ids_1 is None:
            return [1] + ([0] * (len(token_ids_0) - 2)) + [1]
        return (
            [1]
            + ([0] * (len(token_ids_0) - 2))
            + [1, 1]
            + ([0] * (len(token_ids_1) - 1))
            + [1]
        )

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        **kwargs
    ) -> str:
        """
        Apply chat template to format messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            add_generation_prompt: Whether to add generation prompt
            **kwargs: Additional template variables

        Returns:
            Formatted chat string

        Raises:
            ValueError: If no chat template is available
        """
        if self.chat_template is None:
            raise ValueError("No chat template available for this tokenizer")

        from jinja2 import Template

        # Create template
        template = Template(self.chat_template)

        # Prepare template variables
        template_vars = {
            'messages': messages,
            'add_generation_prompt': add_generation_prompt,
            'bos_token': getattr(self, 'bos_token', '[BOS]'),
            'eos_token': getattr(self, 'eos_token', '[EOS]'),
            'sep_token': getattr(self, 'sep_token', '[SEP]'),
            'pad_token': getattr(self, 'pad_token', '[PAD]'),
            **kwargs
        }

        # Render template
        try:
            return template.render(**template_vars)
        except Exception as e:
            raise ValueError(f"Failed to render chat template: {e}") from e
