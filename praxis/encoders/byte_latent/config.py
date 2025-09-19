"""Configuration utilities for byte-latent encoding."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import BYTE_UNITS, OFFSET


class EmbeddingType(Enum):
    """Types of embeddings for byte-latent models."""

    HASH_TOK = "hash_tok"
    STANDARD = "standard"


@dataclass
class ByteLatentConfig:
    """Simplified configuration for byte-latent models."""

    # Core dimensions
    vocab_size: int = BYTE_UNITS + OFFSET
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

    # Hash embeddings
    encoder_hash_byte_group_size: List[int] = None
    encoder_hash_byte_group_nb_functions: int = 1
    encoder_hash_byte_group_vocab: int = 32768

    # Other settings
    dropout: float = 0.0
    norm_eps: float = 1e-5
    max_seqlen: int = 8192
    eos_id: int = 2

    def __post_init__(self):
        """Initialize default values."""
        if self.encoder_hash_byte_group_size is None:
            self.encoder_hash_byte_group_size = [3, 4, 5]


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
        vocab_size=BYTE_UNITS + OFFSET,  # 256 bytes + 4 special tokens = 260 (internal)
        dim=praxis_config.hidden_size,
        dim_token_emb=praxis_config.embed_size,
        dim_global=praxis_config.hidden_size,
        n_layers_local_encoder=1,
        n_layers_local_decoder=1,
        dropout=praxis_config.dropout,
        norm_eps=praxis_config.epsilon,
        downsampling_by_pooling=downsampling_method,
        encoder_hash_byte_group_vocab=praxis_config.vocab_size,  # Use external vocab_size for hash space
        max_seqlen=praxis_config.max_length,
    )


def init_embeddings(
    config: ByteLatentConfig,
    embedding_type: EmbeddingType,
    local_encoder_dim: Optional[int] = None,
    encoder_hash_byte_group_size: Optional[List[int]] = None,
) -> Optional[nn.Module]:
    """
    Initialize embeddings for byte-latent models.

    Args:
        config: Byte-latent configuration
        embedding_type: Type of embedding to create
        local_encoder_dim: Dimension for local encoder
        encoder_hash_byte_group_size: Hash byte group sizes

    Returns:
        Embedding module or None
    """
    if embedding_type == EmbeddingType.HASH_TOK:
        return HashEmbedding(
            config=config,
            local_encoder_dim=local_encoder_dim or config.dim_token_emb,
            encoder_hash_byte_group_size=encoder_hash_byte_group_size
            or config.encoder_hash_byte_group_size,
        )
    return None


# Define primes for polynomial hashing (same as original BLT)
PRIMES = [
    1000000007,
    5915587277,
    1500450271,
    3267000013,
    5754853343,
    4093082899,
    9576890767,
    3628273133,
    2860486313,
    5463458053,
]


def rolling_polynomial_hash(t: torch.Tensor, hash_func_nb: int = 0) -> torch.Tensor:
    """Compute rolling polynomial hash using vectorized operations."""
    prime = torch.tensor(PRIMES[hash_func_nb], dtype=torch.int64, device=t.device)
    prime_powers = torch.stack([prime**i for i in range(t.shape[-1])])
    return torch.sum(t * prime_powers, dim=-1)


def byte_group_hash_function(
    x: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000
) -> torch.Tensor:
    """
    Vectorized implementation of byte group hash function.

    Args:
        x: Input tensor of shape [batch_size, seq_len]
        group_size: Size of the byte group
        hash_func_nb: Which hash function to use
        max_hash: Maximum hash value

    Returns:
        Hash values of shape [batch_size, seq_len]
    """
    with torch.no_grad():
        bs, seq_len = x.shape

        # Add prefix padding to handle edge cases
        prefix = torch.zeros(bs, group_size - 1, dtype=torch.int64, device=x.device)
        x_padded = torch.cat([prefix, x], dim=1)

        # Create sliding windows using unfold
        windows = x_padded.unfold(1, group_size, 1)

        # Compute hash using polynomial rolling hash
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values_range = hashes % max_hash

    hash_values_range.requires_grad = False
    return hash_values_range


class HashEmbedding(nn.Module):
    """
    Hash-based embeddings for byte sequences using vectorized operations.

    Creates embeddings based on n-gram hashes to provide additional context
    for byte-level tokens. Uses the same hash functions as the original BLT.
    """

    def __init__(
        self,
        config: ByteLatentConfig,
        local_encoder_dim: int,
        encoder_hash_byte_group_size: List[int],
    ):
        """
        Initialize hash embeddings.

        Args:
            config: Byte-latent configuration
            local_encoder_dim: Local encoder dimension
            encoder_hash_byte_group_size: List of n-gram sizes to use
        """
        super().__init__()
        self.config = config
        self.local_encoder_dim = local_encoder_dim
        self.encoder_hash_byte_group_size = encoder_hash_byte_group_size
        self.encoder_hash_byte_group_nb_functions = (
            config.encoder_hash_byte_group_nb_functions
        )

        # Create embeddings for each combination of hash function and group size
        self.embeddings = nn.ModuleList()
        for func_nb in range(self.encoder_hash_byte_group_nb_functions):
            for group_size in encoder_hash_byte_group_size:
                embedding = nn.Embedding(
                    config.encoder_hash_byte_group_vocab, local_encoder_dim
                )
                self.embeddings.append(embedding)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute hash embeddings for input tokens using vectorized operations.

        Args:
            tokens: Input tokens of shape [batch_size, seq_len]

        Returns:
            Hash embeddings of shape [batch_size, seq_len, local_encoder_dim]
        """
        if not self.embeddings:
            return torch.zeros(
                tokens.shape[0],
                tokens.shape[1],
                self.local_encoder_dim,
                device=tokens.device,
                dtype=torch.float32,
            )

        # Start with zeros for accumulation
        result = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            self.local_encoder_dim,
            device=tokens.device,
            dtype=torch.float32,
        )

        embedding_idx = 0
        for func_nb in range(self.encoder_hash_byte_group_nb_functions):
            for group_size in self.encoder_hash_byte_group_size:
                # Compute hash IDs using vectorized function
                hash_ids = byte_group_hash_function(
                    tokens,
                    group_size=group_size,
                    hash_func_nb=func_nb,
                    max_hash=self.config.encoder_hash_byte_group_vocab,
                )

                # Get embeddings and accumulate
                hash_embedding = self.embeddings[embedding_idx]
                result = result + hash_embedding(hash_ids)
                embedding_idx += 1

        return result


def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder: Optional[nn.Module],
    encoder_hash_tok_embedding: Optional[nn.Module],
) -> Optional[torch.Tensor]:
    """
    Compute embeddings using hash token embeddings.

    The HashEmbedding module handles all the complexity internally, including:
    - Iterating through different hash functions and byte group sizes
    - Computing hash IDs for each combination
    - Accumulating embeddings from each hash function/group size pair

    Args:
        local_encoder_tokens: Input token IDs tensor
        local_encoder: Encoder object with tok_emb method
        encoder_hash_tok_embedding: HashEmbedding module that handles hash embeddings

    Returns:
        torch.Tensor: Combined base embeddings + hash embeddings, or None if no hash embeddings
    """
    if encoder_hash_tok_embedding is None:
        return None

    # Get base token embeddings from local encoder
    local_encoder_embeds = local_encoder.tok_emb(local_encoder_tokens)

    # Get hash embeddings from HashEmbedding module (it handles all the looping internally)
    hash_embeds = encoder_hash_tok_embedding(local_encoder_tokens)

    return local_encoder_embeds + hash_embeds


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
