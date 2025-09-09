"""Praxis tokenizer implementations and registry."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from .base import PraxisTokenizerBase
from .byte_level import ByteLevelTokenizer
from .standard import StandardTokenizer

# Check if ByteLevelTokenizer is available
try:
    from .byte_level import ByteLevelTokenizer
    HAS_BYTE_LEVEL = True
except ImportError:
    HAS_BYTE_LEVEL = False
    ByteLevelTokenizer = None


# Tokenizer registry
TOKENIZER_REGISTRY: Dict[str, Type[PraxisTokenizerBase]] = {
    "standard": StandardTokenizer,
    "bpe": StandardTokenizer,
    "unigram": StandardTokenizer,
}

if HAS_BYTE_LEVEL:
    TOKENIZER_REGISTRY["byte_level"] = ByteLevelTokenizer
    TOKENIZER_REGISTRY["byte"] = ByteLevelTokenizer
    TOKENIZER_REGISTRY["bytelevel"] = ByteLevelTokenizer


# Tokenizer profiles with default configurations
TOKENIZER_PROFILES = {
    "default": {
        "tokenizer_class": "standard",
        "tokenizer_type": "unigram",
        "vocab_size": 32768,
        "model_max_length": 2048,
    },
    "bpe": {
        "tokenizer_class": "standard",
        "tokenizer_type": "bpe",
        "vocab_size": 32768,
        "dropout": 0.1,
        "model_max_length": 2048,
    },
    "unigram": {
        "tokenizer_class": "standard",
        "tokenizer_type": "unigram",
        "vocab_size": 32768,
        "model_max_length": 2048,
    },
    "byte": {
        "tokenizer_class": "byte_level",
        "vocab_size_unit_1": 256,
        "model_max_length": 2048,
    },
    "small": {
        "tokenizer_class": "standard",
        "tokenizer_type": "unigram",
        "vocab_size": 8192,
        "model_max_length": 1024,
    },
    "large": {
        "tokenizer_class": "standard",
        "tokenizer_type": "unigram",
        "vocab_size": 65536,
        "model_max_length": 4096,
    },
}


def create_tokenizer(
    tokenizer_name: Optional[str] = None,
    tokenizer_profile: Optional[str] = None,
    tokenizer_path: Optional[Union[str, Path]] = None,
    encoder_type: Optional[str] = None,
    vocab_size: int = 32768,
    cache_dir: Optional[str] = None,
    **kwargs
) -> PreTrainedTokenizer:
    """
    Create a tokenizer instance with unified loading logic.
    
    This function provides a single interface for all tokenizer creation,
    handling encoder type detection, local paths, and remote loading.
    
    Args:
        tokenizer_name: Name from the tokenizer registry
        tokenizer_profile: Profile name with preset configuration
        tokenizer_path: Path to load a pretrained tokenizer
        encoder_type: Encoder type (if "byte_latent", uses ByteLevel tokenizer)
        vocab_size: Vocabulary size for tokenizer
        cache_dir: Cache directory for downloading tokenizers
        **kwargs: Additional arguments passed to tokenizer constructor
    
    Returns:
        Tokenizer instance
    
    Priority:
        1. Explicit tokenizer_path if provided
        2. Explicit tokenizer_profile or tokenizer_name if provided
        3. Encoder type detection (byte_latent -> ByteLevel)
        4. Local paths (build/tokenizers/praxis-{vocab_size}-unigram)
        5. Remote HuggingFace repos (UNSAFE/praxis-{vocab_size})
        6. Default profile
    """
    # 1. Try to load from explicit path if provided
    if tokenizer_path is not None:
        path = Path(tokenizer_path)
        if path.exists():
            try:
                return AutoTokenizer.from_pretrained(
                    path,
                    cache_dir=cache_dir,
                    **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer from {path}: {e}")
    
    # 2. Use explicit profile or name if provided
    if tokenizer_profile is not None:
        if tokenizer_profile not in TOKENIZER_PROFILES:
            raise ValueError(
                f"Unknown tokenizer profile: {tokenizer_profile}. "
                f"Available profiles: {list(TOKENIZER_PROFILES.keys())}"
            )
        
        profile = TOKENIZER_PROFILES[tokenizer_profile].copy()
        tokenizer_class_name = profile.pop("tokenizer_class")
        
        # Merge profile settings with kwargs (kwargs take precedence)
        profile.update(kwargs)
        
        if tokenizer_class_name not in TOKENIZER_REGISTRY:
            raise ValueError(f"Tokenizer class {tokenizer_class_name} not in registry")
        
        tokenizer_class = TOKENIZER_REGISTRY[tokenizer_class_name]
        return tokenizer_class(**profile)
    
    if tokenizer_name is not None:
        if tokenizer_name not in TOKENIZER_REGISTRY:
            raise ValueError(
                f"Unknown tokenizer: {tokenizer_name}. "
                f"Available tokenizers: {list(TOKENIZER_REGISTRY.keys())}"
            )
        
        tokenizer_class = TOKENIZER_REGISTRY[tokenizer_name]
        return tokenizer_class(**kwargs)
    
    # 3. Determine tokenizer based on encoder type
    if encoder_type == "byte_latent":
        if HAS_BYTE_LEVEL:
            return ByteLevelTokenizer(**kwargs)
        else:
            raise ImportError(
                "ByteLevelTokenizer requires bytelatent package. "
                "Please install it with: pip install bytelatent"
            )
    
    # 4. Try to load from local paths (single unified path)
    local_path = Path(f"build/tokenizers/praxis-{vocab_size}-unigram")
    if local_path.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                cache_dir=cache_dir,
                **kwargs
            )
            # Override with our updated chat template
            from .chat_templates import get_chat_template
            tokenizer.chat_template = get_chat_template("default")
            return tokenizer
        except Exception:
            pass
    
    # 5. Try to load from HuggingFace repo
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            f"UNSAFE/praxis-{vocab_size}",
            cache_dir=cache_dir,
            **kwargs
        )
        # Override with our updated chat template
        from .chat_templates import get_chat_template
        tokenizer.chat_template = get_chat_template("default")
        return tokenizer
    except Exception:
        pass
    
    # 6. Fall back to default profile
    print(f"No tokenizer found, creating default unigram tokenizer")
    return create_tokenizer(tokenizer_profile="default", vocab_size=vocab_size, **kwargs)


def get_tokenizer(
    byte_latent: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs
) -> PreTrainedTokenizer:
    """
    Get a tokenizer with backward compatibility.
    
    This function provides backward compatibility with the old
    byte_latent flag while using the new tokenizer system.
    
    Args:
        byte_latent: Whether to use byte-level tokenizer
        cache_dir: Cache directory for downloading tokenizers
        **kwargs: Additional arguments
    
    Returns:
        Tokenizer instance
    """
    if byte_latent:
        if not HAS_BYTE_LEVEL:
            raise ImportError(
                "ByteLevelTokenizer requires bytelatent package. "
                "Please install it with: pip install bytelatent"
            )
        return create_tokenizer(tokenizer_name="byte_level", **kwargs)
    else:
        # Try to load from standard paths first
        return create_tokenizer(cache_dir=cache_dir, **kwargs)


def train_tokenizer(
    tokenizer_type: str = "unigram",
    vocab_size: int = 32768,
    num_examples: int = 5_000_000,
    save: bool = True,
    **kwargs
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
        **kwargs
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
    # Registry and profiles
    "TOKENIZER_REGISTRY",
    "TOKENIZER_PROFILES",
    # Factory functions
    "create_tokenizer",
    "get_tokenizer",
    "train_tokenizer",
]