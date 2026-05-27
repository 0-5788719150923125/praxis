"""Configuration utilities for byte-latent encoding."""

from dataclasses import dataclass
from typing import Optional

from .constants import BYTE_UNITS, NUM_TOOL_TOKENS, OFFSET


@dataclass
class ByteLatentConfig:
    """Simplified configuration for byte-latent models."""

    # Local byte-level table size (256 bytes + named specials + tool specials).
    # Distinct from the model's external vocab_size.
    local_vocab_size: int = BYTE_UNITS + OFFSET + NUM_TOOL_TOKENS
    dim: int = 512
    dim_token_emb: int = 128
    dim_global: int = 512

    # Encoder/decoder layers
    n_layers_local_encoder: int = 1
    n_layers_local_decoder: int = 1

    # Patching
    patch_size: int = 6
    patching_mode: str = "entropy"
    patching_threshold: float = 3.14159
    monotonicity: bool = False

    # Downsampling
    downsampling_by_pooling: Optional[str] = "max"

    # Cross-attention settings
    cross_attn_encoder: bool = False
    cross_attn_decoder: bool = False
    cross_attn_k: int = 1
    cross_attn_window_encoder: int = 512
    cross_attn_window_decoder: int = 512

    # Other settings
    dropout: float = 0.0
    norm_eps: float = 1e-5
    max_seqlen: int = 8192
    eos_id: int = 2


def create_base_config(praxis_config) -> ByteLatentConfig:
    """
    Create ByteLatentConfig from a Praxis config object.

    Args:
        praxis_config: Praxis model configuration

    Returns:
        ByteLatentConfig with mapped settings
    """
    # Determine downsampling method from meta flags
    downsampling_method = "max"
    if hasattr(praxis_config, "meta"):
        if "min" in praxis_config.meta:
            downsampling_method = "min"
        elif "mean" in praxis_config.meta:
            downsampling_method = "mean"
        elif any(flag.startswith("topk:") for flag in praxis_config.meta):
            downsampling_method = next(
                x for x in praxis_config.meta if x.startswith("topk:")
            )

    return ByteLatentConfig(
        local_vocab_size=BYTE_UNITS + OFFSET + NUM_TOOL_TOKENS,
        # 256 bytes + 4 named specials + 4 tool-control specials = 264
        dim=praxis_config.hidden_size,
        dim_token_emb=praxis_config.embed_size,
        dim_global=praxis_config.hidden_size,
        n_layers_local_encoder=1,
        n_layers_local_decoder=1,
        dropout=praxis_config.dropout,
        norm_eps=praxis_config.epsilon,
        downsampling_by_pooling=downsampling_method,
        max_seqlen=praxis_config.max_position_embeddings,
    )


# Dimension calculation helpers
def get_encoder_dim_token_emb(config: ByteLatentConfig) -> int:
    """Get token embedding dimension for encoder."""
    return config.dim_token_emb


def get_encoder_dim_patch_emb(config: ByteLatentConfig) -> Optional[int]:
    """Get patch embedding dimension for encoder."""
    if config.cross_attn_encoder:
        return config.dim_global
    return None


def get_decoder_dim_token_emb(config: ByteLatentConfig) -> int:
    """Get token embedding dimension for decoder."""
    return config.dim_token_emb
