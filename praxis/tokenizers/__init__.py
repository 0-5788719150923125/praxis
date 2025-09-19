"""Praxis tokenizer implementations and registry."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from .base import PraxisTokenizerBase
from .standard import StandardTokenizer

# Check if ByteLevelTokenizer is available
try:
    from .byte_level import ByteLevelTokenizer

    HAS_BYTE_LEVEL = True
except ImportError:
    HAS_BYTE_LEVEL = False
    ByteLevelTokenizer = None


def create_tokenizer(
    vocab_size: int = 32768,
    encoder_type: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> PreTrainedTokenizer:
    """
    Create a tokenizer instance based on vocab_size.

    Simple logic:
    1. If encoder_type is "byte_latent", use ByteLevelTokenizer
    2. Try to load existing tokenizer for vocab_size
    3. Create new StandardTokenizer if needed

    Args:
        vocab_size: Vocabulary size for tokenizer
        encoder_type: Encoder type (if "byte_latent", uses ByteLevel tokenizer)
        cache_dir: Cache directory for downloading tokenizers
        **kwargs: Additional arguments passed to tokenizer constructor

    Returns:
        Tokenizer instance
    """
    # 1. Special case: byte_latent encoders use ByteLevelTokenizer
    if encoder_type and encoder_type.startswith("byte_latent"):
        if HAS_BYTE_LEVEL:
            return ByteLevelTokenizer(**kwargs)
        else:
            raise ImportError(
                "ByteLevelTokenizer requires bytelatent package. "
                "Please install it with: pip install bytelatent"
            )

    # 2. Try to load existing tokenizer for this vocab_size
    # First check local path
    local_path = Path(f"build/tokenizers/praxis-{vocab_size}-unigram")
    if local_path.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_path, cache_dir=cache_dir, **kwargs
            )
            # Override with our updated chat template
            from .chat_templates import get_chat_template

            tokenizer.chat_template = get_chat_template("default")
            return tokenizer
        except Exception:
            pass

    # Try HuggingFace repo
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            f"UNSAFE/praxis-{vocab_size}", cache_dir=cache_dir, **kwargs
        )
        # Override with our updated chat template
        from .chat_templates import get_chat_template

        tokenizer.chat_template = get_chat_template("default")
        return tokenizer
    except Exception:
        pass

    # 3. Create new StandardTokenizer if nothing found
    # print(
    #     f"No tokenizer found for vocab_size={vocab_size}, creating new unigram tokenizer"
    # )
    # return StandardTokenizer(
    #     tokenizer_type="unigram",
    #     vocab_size=vocab_size,
    #     model_max_length=kwargs.get("model_max_length", 2048),
    #     **kwargs,
    # )


def train_tokenizer(
    tokenizer_type: str = "unigram",
    vocab_size: int = 32768,
    num_examples: int = 5_000_000,
    save: bool = True,
    **kwargs,
) -> StandardTokenizer:
    """
    Train a new tokenizer from a dataset.

    This function provides the functionality of train_tokenizer.py
    as a callable function.

    Args:
        tokenizer_type: Type of tokenizer ("bpe" or "unigram")
        vocab_size: Target vocabulary size
        num_examples: Number of examples to use for training
        save: Whether to save the tokenizer to disk
        **kwargs: Additional arguments passed to train_from_dataset

    Returns:
        Trained tokenizer instance
    """
    # Train the tokenizer
    tokenizer = StandardTokenizer.train_from_dataset(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        num_examples=num_examples,
        **kwargs,
    )

    # Save to deterministic locations
    if save:
        base_path = Path("build/tokenizers")

        # Main save path: build/tokenizers/praxis-{vocab_size}-{type}
        save_path = base_path / f"praxis-{vocab_size}-{tokenizer_type}"

        # Also save to a generic "model" folder for backward compatibility
        generic_path = base_path / "model"

        save_path.mkdir(parents=True, exist_ok=True)
        generic_path.mkdir(parents=True, exist_ok=True)

        tokenizer.save_pretrained(save_path)
        tokenizer.save_pretrained(generic_path)

        print(f"Tokenizer saved to {save_path} and {generic_path}")

    return tokenizer


__all__ = [
    # Base classes
    "PraxisTokenizerBase",
    "ByteLevelTokenizer",
    "StandardTokenizer",
    # Factory functions
    "create_tokenizer",
    "train_tokenizer",
]
