"""Praxis tokenizer implementations and registry."""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer

from .base import PraxisTokenizerBase
from .char_level import CharLevelTokenizer
from .chat_templates import get_chat_template
from .standard import StandardTokenizer

# ByteLevel depends on the byte-latent stack; tolerate its absence.
try:
    from .byte_level import ByteLevelTokenizer

    HAS_BYTE_LEVEL = True
except ImportError:
    HAS_BYTE_LEVEL = False
    ByteLevelTokenizer = None


# Registry of named tokenizer implementations. Each entry is a callable
# that accepts ``vocab_size=...`` and ``**kwargs``. BPE / unigram are
# partials over StandardTokenizer so both names dispatch to the same
# class with different training models.
TOKENIZER_REGISTRY: Dict[str, Any] = {
    "char_level": CharLevelTokenizer,
    "bpe": partial(StandardTokenizer, tokenizer_type="bpe"),
    "unigram": partial(StandardTokenizer, tokenizer_type="unigram"),
}
if HAS_BYTE_LEVEL:
    TOKENIZER_REGISTRY["byte_level"] = ByteLevelTokenizer

# Valid --vocab-size values. Mutable on purpose: integrations that ship
# pretrained vocabs (e.g. tokenmonster) extend this at import time, before
# the CLI argument groups are built.
VOCAB_SIZE_CHOICES: list = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

# Default tokenizer when ``tokenizer_type`` is unset. Stays at unigram
# because that's the only flavor we actually publish pre-trained
# checkpoints for; switching the default to "bpe" before the BPE
# tokenizers are trained and uploaded produces a fresh, untrained
# StandardTokenizer (vocab_size = number of special tokens), which then
# tokenizes every input into noise and collapses the SFT loss to zero.
DEFAULT_TOKENIZER: str = "unigram"


def _needs_byte_level_tokenizer(encoder_type: str) -> bool:
    """Whether ``encoder_type`` requires the byte-level tokenizer.

    Used only to emit a warning when the user's encoder/tokenizer pair
    is incompatible; selection itself is now explicit.
    """
    try:
        from praxis.encoders import ENCODER_REGISTRY
        from praxis.encoders.byte_latent import ByteLatentEncoder

        encoder_cls = ENCODER_REGISTRY.get(encoder_type)
        if encoder_cls is None:
            return False
        actual_cls = getattr(encoder_cls, "func", encoder_cls)
        return issubclass(actual_cls, ByteLatentEncoder)
    except ImportError:
        return encoder_type.startswith("byte_latent")


def _try_load_trained_tokenizer(
    vocab_size: int,
    tokenizer_type: str,
    cache_dir: Optional[str],
    **kwargs,
) -> Optional[PreTrainedTokenizer]:
    """Look for a pre-trained tokenizer on disk first, then on the Hub.

    Tries the type-suffixed location first (``praxis-{vocab_size}-{type}``)
    so a future BPE checkpoint will resolve cleanly, then falls back to
    the legacy location (``praxis-{vocab_size}``, no suffix) which is
    where the unigram tokenizers actually live. Returns ``None`` if no
    candidate resolves.
    """
    candidates_local = [
        Path(f"build/tokenizers/praxis-{vocab_size}-{tokenizer_type}"),
        Path(f"build/tokenizers/praxis-{vocab_size}"),
    ]
    for local_path in candidates_local:
        if not local_path.exists():
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_path, cache_dir=cache_dir, **kwargs
            )
            tokenizer.chat_template = get_chat_template("default")
            return tokenizer
        except Exception:
            continue

    candidates_hub = [
        f"UNSAFE/praxis-{vocab_size}-{tokenizer_type}",
        f"UNSAFE/praxis-{vocab_size}",
    ]
    for repo in candidates_hub:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo, cache_dir=cache_dir, **kwargs
            )
            tokenizer.chat_template = get_chat_template("default")
            return tokenizer
        except Exception:
            continue

    return None


def create_tokenizer(
    vocab_size: int = 32768,
    encoder_type: Optional[str] = None,
    tokenizer_type: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> PreTrainedTokenizer:
    """Create a tokenizer instance from :data:`TOKENIZER_REGISTRY`.

    Dispatch is explicit: an unrecognized ``tokenizer_type`` is a hard
    error. When ``tokenizer_type`` is unset, defaults to
    :data:`DEFAULT_TOKENIZER` (BPE). ``encoder_type`` is only inspected
    to emit a compatibility warning when the chosen tokenizer can't
    produce the byte ids the encoder expects - there is no implicit
    override.
    """
    if tokenizer_type is None:
        tokenizer_type = DEFAULT_TOKENIZER

    if tokenizer_type not in TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unknown tokenizer_type={tokenizer_type!r}. "
            f"Valid choices: {sorted(TOKENIZER_REGISTRY)}"
        )

    if (
        encoder_type
        and _needs_byte_level_tokenizer(encoder_type)
        and tokenizer_type != "byte_level"
    ):
        # Byte-latent encoders only operate on byte ids; any other tokenizer
        # produces incompatible inputs. Override rather than warn so the run
        # stays internally consistent.
        print(
            f"[INFO] encoder_type={encoder_type!r} requires byte-level "
            f"tokenization; overriding tokenizer_type={tokenizer_type!r} "
            f"-> 'byte_level'."
        )
        tokenizer_type = "byte_level"

    # Trained tokenizers: try loading a pre-trained one for the requested
    # vocab_size before instantiating a fresh untrained instance.
    if tokenizer_type in {"bpe", "unigram"}:
        loaded = _try_load_trained_tokenizer(
            vocab_size=vocab_size,
            tokenizer_type=tokenizer_type,
            cache_dir=cache_dir,
            **kwargs,
        )
        if loaded is not None:
            return loaded

    factory = TOKENIZER_REGISTRY[tokenizer_type]
    return factory(vocab_size=vocab_size, **kwargs)


def train_tokenizer(
    tokenizer_type: str = "unigram",
    vocab_size: int = 32768,
    num_examples: int = 5_000_000,
    save: bool = True,
    **kwargs,
) -> StandardTokenizer:
    """Train a new tokenizer from a dataset.

    Args:
        tokenizer_type: ``"bpe"`` or ``"unigram"``.
        vocab_size: Target vocabulary size.
        num_examples: Number of examples to use for training.
        save: Whether to save the tokenizer to disk.
        **kwargs: Additional arguments passed to ``train_from_dataset``.

    Returns:
        Trained tokenizer instance.
    """
    tokenizer = StandardTokenizer.train_from_dataset(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        num_examples=num_examples,
        **kwargs,
    )

    if save:
        base_path = Path("build/tokenizers")
        save_path = base_path / f"praxis-{vocab_size}-{tokenizer_type}"
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
    "CharLevelTokenizer",
    "StandardTokenizer",
    # Registry
    "TOKENIZER_REGISTRY",
    "DEFAULT_TOKENIZER",
    # Factory functions
    "create_tokenizer",
    "train_tokenizer",
]
