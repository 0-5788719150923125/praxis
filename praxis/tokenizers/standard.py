"""Standard tokenizer with BPE/Unigram support for Praxis."""

import json
import os
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tokenizers import (Tokenizer, decoders, models, normalizers,
                        pre_tokenizers, processors, trainers)
from transformers import PreTrainedTokenizerFast

from .base import PraxisTokenizerBase
from .chat_templates import get_chat_template


class StandardTokenizer(PreTrainedTokenizerFast, PraxisTokenizerBase):
    """
    Standard tokenizer supporting BPE and Unigram models.
    
    This tokenizer can be trained on text corpora and supports
    the full HuggingFace tokenizer interface.
    """
    
    SPECIAL_TOKENS = {
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "sep_token": "[SEP]",
        # "mask_token": "[MASK]",  # Not used in Praxis
    }
    
    def __init__(
        self,
        tokenizer_object: Optional[Tokenizer] = None,
        tokenizer_type: str = "unigram",
        vocab_size: int = 32768,
        dropout: float = 0.1,
        chat_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the standard tokenizer.
        
        Args:
            tokenizer_object: Pre-built tokenizer object
            tokenizer_type: Type of tokenizer ("bpe" or "unigram")
            vocab_size: Size of the vocabulary
            dropout: Dropout rate for BPE
            chat_template: Chat template for conversation formatting
            **kwargs: Additional arguments passed to parent class
        """
        self.tokenizer_type = tokenizer_type
        self._vocab_size = vocab_size
        self.dropout = dropout
        
        # Initialize special tokens
        for token_name, token_value in self.SPECIAL_TOKENS.items():
            if token_name not in kwargs:
                kwargs[token_name] = token_value
        
        # Create or use provided tokenizer object
        if tokenizer_object is None:
            tokenizer_object = self._create_tokenizer(tokenizer_type, dropout)
        
        # Initialize PreTrainedTokenizerFast
        PreTrainedTokenizerFast.__init__(
            self,
            tokenizer_object=tokenizer_object,
            **kwargs
        )
        
        # Set chat template
        if chat_template is None:
            self.chat_template = get_chat_template(tokenizer_type)
        else:
            self.chat_template = chat_template
    
    def _create_tokenizer(self, tokenizer_type: str, dropout: float) -> Tokenizer:
        """
        Create a new tokenizer object.
        
        Args:
            tokenizer_type: Type of tokenizer ("bpe" or "unigram")
            dropout: Dropout rate for BPE
            
        Returns:
            Tokenizer object
        """
        if tokenizer_type == "bpe":
            tokenizer = Tokenizer(models.BPE(dropout=dropout, byte_fallback=True))
        elif tokenizer_type == "unigram":
            tokenizer = Tokenizer(models.Unigram(byte_fallback=True))
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        # Set up tokenizer components
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel()
        
        # Add special tokens
        all_special_tokens = list(self.SPECIAL_TOKENS.values())
        tokenizer.add_special_tokens(all_special_tokens)
        
        return tokenizer
    
    def train(
        self,
        texts: Union[List[str], Iterator[str], Any],
        vocab_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs
    ) -> None:
        """
        Train the tokenizer on a corpus of texts.
        
        Args:
            texts: Training texts or iterator
            vocab_size: Target vocabulary size (uses instance default if None)
            show_progress: Whether to show training progress
            **kwargs: Additional training parameters
        """
        if vocab_size is None:
            vocab_size = self._vocab_size
        
        # Get all special tokens
        all_special_tokens = list(self.SPECIAL_TOKENS.values())
        additional_special_tokens = kwargs.get("additional_special_tokens", [])
        all_special_tokens = list(set(all_special_tokens + additional_special_tokens))
        
        # Create trainer based on tokenizer type
        trainer_args = dict(
            vocab_size=vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=show_progress,
            special_tokens=all_special_tokens,
        )
        
        if self.tokenizer_type == "bpe":
            trainer = trainers.BpeTrainer(**trainer_args)
        elif self.tokenizer_type == "unigram":
            trainer = trainers.UnigramTrainer(
                shrinking_factor=0.75,
                max_piece_length=16,
                n_sub_iterations=8,
                **trainer_args
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")
        
        # Train the tokenizer
        if isinstance(texts, list):
            self._tokenizer.train(texts, trainer)
        else:
            # For iterators, we need to know the length
            length = kwargs.get("length", None)
            self._tokenizer.train_from_iterator(
                iterator=texts,
                trainer=trainer,
                length=length
            )
        
        # Update vocab size
        self._vocab_size = vocab_size
        
        # Add special tokens to the trained tokenizer
        self.add_special_tokens({
            **self.SPECIAL_TOKENS,
            "additional_special_tokens": additional_special_tokens
        })
    
    @classmethod
    def train_from_dataset(
        cls,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_config: str = "sample-350BT",
        num_examples: int = 5_000_000,
        vocab_size: int = 32768,
        tokenizer_type: str = "unigram",
        dropout: float = 0.1,
        cache_dir: str = "build/datasets",
        **kwargs
    ) -> "StandardTokenizer":
        """
        Train a new tokenizer from a dataset.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            dataset_config: Dataset configuration name
            num_examples: Number of examples to use for training
            vocab_size: Target vocabulary size
            tokenizer_type: Type of tokenizer ("bpe" or "unigram")
            dropout: Dropout rate for BPE
            cache_dir: Cache directory for dataset
            **kwargs: Additional arguments
            
        Returns:
            Trained tokenizer instance
        """
        from datasets import load_dataset

        # Load dataset
        dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        ).shuffle(seed=42, buffer_size=10_000)
        
        # Create iterator
        key = kwargs.get("text_key", "text")
        iterator = islice((item[key] for item in dataset), num_examples)
        
        # Create tokenizer
        tokenizer = cls(
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            dropout=dropout,
            **kwargs
        )
        
        # Train tokenizer
        tokenizer.train(iterator, vocab_size=vocab_size, length=num_examples)
        
        return tokenizer
    
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
        # Use parent class method for saving
        return super().save_vocabulary(save_directory, filename_prefix)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        **kwargs
    ) -> "StandardTokenizer":
        """
        Load a pretrained tokenizer.
        
        Args:
            pretrained_model_name_or_path: Path or model identifier
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded tokenizer instance
        """
        # Try to load as a PreTrainedTokenizerFast first
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )
            
            # Wrap it in our StandardTokenizer class
            return cls(
                tokenizer_object=tokenizer._tokenizer,
                **kwargs
            )
        except Exception:
            # Fall back to default initialization
            return cls(**kwargs)
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a string into a list of tokens.
        
        This method is provided for PraxisTokenizerBase compatibility,
        but PreTrainedTokenizerFast handles tokenization internally.
        
        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters
            
        Returns:
            List of token strings
        """
        encoding = self._tokenizer.encode(text)
        return [self._tokenizer.id_to_token(id) for id in encoding.ids]
    
    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert a token string to its ID.
        
        Args:
            token: Token string
            
        Returns:
            Token ID
        """
        return self._tokenizer.token_to_id(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """
        Convert a token ID to its string representation.
        
        Args:
            index: Token ID
            
        Returns:
            Token string
        """
        return self._tokenizer.id_to_token(index)
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Return the vocabulary as a dictionary.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        return self._tokenizer.get_vocab()
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._tokenizer.get_vocab_size()