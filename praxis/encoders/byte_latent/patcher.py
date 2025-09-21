"""Byte-level patching utilities for sequence compression."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .constants import BPE_ID, EOS_ID, OFFSET


class PatchingMode(str, Enum):
    """Supported patching modes."""

    entropy = "entropy"
    space = "space"
    static = "static"
    byte = "byte"


@dataclass
class PatcherConfig:
    """Configuration for the Patcher."""

    patching_mode: PatchingMode = PatchingMode.entropy
    patch_size: int = 6
    threshold: float = 3.14159  # Default to pi
    monotonicity: bool = False
    device: str = "cuda"
    realtime_patching: bool = False


class Patcher:
    """
    Patcher for creating variable-length patches from sequences.

    This is a simplified version of the BLT Patcher, focusing on
    the core functionality needed for byte-latent encoding.
    """

    def __init__(self, config: PatcherConfig):
        """
        Initialize the Patcher.

        Args:
            config: Patcher configuration
        """
        self.config = config
        self.entropy_model: Optional[nn.Module] = None

    def patch(
        self,
        tokens: torch.Tensor,
        include_next_token: bool = True,
        threshold: Optional[float] = None,
        entropies: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create patches from input tokens.

        Args:
            tokens: Input token tensor of shape [batch_size, seq_len]
            include_next_token: Whether to include the next token in patch boundaries
            threshold: Optional entropy threshold override
            entropies: Pre-computed entropy scores (for entropy mode)

        Returns:
            Tuple of:
                - patch_lengths: Tensor of patch lengths [batch_size, num_patches]
                - patch_boundaries: Binary tensor marking patch boundaries
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        if self.config.patching_mode == PatchingMode.entropy:
            if entropies is None:
                raise ValueError("Entropy scores required for entropy patching mode")
            return self._entropy_patching(
                tokens, entropies, threshold or self.config.threshold, include_next_token
            )
        elif self.config.patching_mode == PatchingMode.space:
            return self._space_patching(tokens, include_next_token)
        elif self.config.patching_mode == PatchingMode.static:
            return self._static_patching(tokens, self.config.patch_size)
        else:
            # Default to static patching
            return self._static_patching(tokens, self.config.patch_size)

    def _entropy_patching(
        self,
        tokens: torch.Tensor,
        entropies: torch.Tensor,
        threshold: float,
        include_next_token: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create patches based on entropy scores.

        Args:
            tokens: Input tokens [batch_size, seq_len]
            entropies: Entropy scores [batch_size, seq_len]
            threshold: Entropy threshold for creating boundaries
            include_next_token: Whether to include next token

        Returns:
            Tuple of patch lengths and boundaries
        """
        batch_size, seq_len = tokens.shape

        # Create boundaries where entropy exceeds threshold
        boundaries = entropies > threshold

        # Ensure monotonicity if configured
        if self.config.monotonicity:
            # Once a boundary is set, all following positions are boundaries
            boundaries = boundaries.cummax(dim=1)[0]

        # Calculate patch lengths from boundaries
        patch_lengths = self._boundaries_to_lengths(boundaries, include_next_token)

        return patch_lengths, boundaries

    def _space_patching(
        self, tokens: torch.Tensor, include_next_token: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create patches at space boundaries using BLT reference logic.

        This implementation matches the BLT reference which creates patch boundaries
        at "space-like" bytes defined as any byte that is not a latin character,
        digit, or UTF-8 continuation byte.

        Args:
            tokens: Input tokens [batch_size, seq_len]
            include_next_token: Whether to include next token

        Returns:
            Tuple of patch lengths and boundaries
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Convert tokens to raw bytes by removing offset
        tokens_no_offset = tokens - OFFSET

        # Create patch end mask for "space-like" bytes (non-alphanumeric characters)
        # This matches the BLT reference implementation logic
        patch_end_mask = (
            (tokens_no_offset < ord("0"))  # Before digits
            | ((ord("9") < tokens_no_offset) & (tokens_no_offset < ord("A")))  # Between digits and uppercase
            | ((ord("Z") < tokens_no_offset) & (tokens_no_offset < ord("a")))  # Between upper and lowercase
            | ((ord("z") < tokens_no_offset) & (tokens_no_offset < 0b1000_0000))  # After lowercase, before high bit
            | (0b1100_0000 <= tokens_no_offset)  # UTF-8 start bytes
        )

        # Ensure we don't have consecutive patch ends (only first in sequence)
        patch_end_mask[:, 1:] &= patch_end_mask[:, :-1].bitwise_not()

        # Handle special tokens (those below OFFSET) as patch boundaries
        patch_end_mask |= tokens < OFFSET

        # Create patch start mask following BLT logic
        # Start with first position and position after each patch end
        patch_start_mask = torch.zeros_like(tokens, dtype=torch.bool)
        patch_start_mask[:, 0] = True  # Always start at position 0

        # Start new patches after space-like characters
        if seq_len > 1:
            patch_start_mask[:, 1:] = patch_end_mask[:, :-1]

        # Convert to patch start IDs and then to lengths using BLT method
        patch_start_ids = self._patch_start_mask_to_ids(patch_start_mask)
        seq_len_with_next = seq_len + (1 if include_next_token else 0)
        patch_lengths = self._patch_lengths_from_start_ids(patch_start_ids, seq_len_with_next)

        return patch_lengths, patch_start_mask

    def _patch_start_mask_to_ids(self, patch_start_mask: torch.Tensor) -> torch.Tensor:
        """Convert patch start mask to start IDs following BLT logic."""
        batch_size, seq_len = patch_start_mask.shape
        device = patch_start_mask.device

        # Find patch start positions for each batch
        patch_start_ids_list = []
        max_patches = 0

        for b in range(batch_size):
            start_positions = torch.where(patch_start_mask[b])[0]
            if len(start_positions) == 0:
                start_positions = torch.tensor([0], device=device)
            patch_start_ids_list.append(start_positions)
            max_patches = max(max_patches, len(start_positions))

        # Pad to same length (fill with seq_len for unused positions)
        patch_start_ids = torch.full(
            (batch_size, max_patches), seq_len, device=device, dtype=torch.long
        )
        for b, positions in enumerate(patch_start_ids_list):
            patch_start_ids[b, :len(positions)] = positions

        return patch_start_ids

    def _patch_lengths_from_start_ids(self, patch_start_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Calculate patch lengths from start IDs (BLT reference method)."""
        batch_size, max_patches = patch_start_ids.shape
        device = patch_start_ids.device

        # Calculate end positions: next start position - 1, or seq_len - 1 for last patch
        patch_end_ids = torch.full_like(patch_start_ids, seq_len - 1)
        patch_end_ids[:, :-1] = patch_start_ids[:, 1:] - 1

        # Calculate lengths
        patch_lengths = patch_end_ids - patch_start_ids + 1

        # Zero out invalid patches (where start_id >= seq_len)
        valid_mask = patch_start_ids < seq_len
        patch_lengths = patch_lengths * valid_mask.long()

        return patch_lengths

    def _static_patching(
        self, tokens: torch.Tensor, patch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create fixed-size patches.

        Args:
            tokens: Input tokens [batch_size, seq_len]
            patch_size: Size of each patch

        Returns:
            Tuple of patch lengths and boundaries
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Calculate number of patches
        num_patches = (seq_len + patch_size - 1) // patch_size

        # Create uniform patch lengths
        patch_lengths = torch.full(
            (batch_size, num_patches), patch_size, device=device, dtype=torch.long
        )

        # Handle last patch if sequence doesn't divide evenly
        last_patch_size = seq_len % patch_size
        if last_patch_size > 0:
            patch_lengths[:, -1] = last_patch_size

        # Create boundaries at patch edges
        boundaries = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        for i in range(patch_size - 1, seq_len, patch_size):
            boundaries[:, i] = True

        return patch_lengths, boundaries

    def _boundaries_to_lengths(
        self, boundaries: torch.Tensor, include_next_token: bool
    ) -> torch.Tensor:
        """
        Convert boundary markers to patch lengths.

        Args:
            boundaries: Binary tensor marking patch boundaries [batch_size, seq_len]
            include_next_token: Whether to include next token in patches

        Returns:
            Tensor of patch lengths [batch_size, num_patches]
        """
        batch_size, seq_len = boundaries.shape
        device = boundaries.device

        # Add implicit boundary at the end
        boundaries = torch.cat(
            [boundaries, torch.ones(batch_size, 1, device=device, dtype=torch.bool)],
            dim=1,
        )

        # Find positions of boundaries
        boundary_positions = []
        for b in range(batch_size):
            positions = torch.where(boundaries[b])[0]
            if len(positions) == 0:
                # No boundaries, entire sequence is one patch
                positions = torch.tensor([seq_len - 1], device=device)
            boundary_positions.append(positions)

        # Calculate patch lengths
        max_patches = max(len(pos) for pos in boundary_positions)
        patch_lengths = torch.zeros(batch_size, max_patches, device=device, dtype=torch.long)

        for b in range(batch_size):
            positions = boundary_positions[b]
            prev_pos = -1
            for i, pos in enumerate(positions):
                if i < max_patches:
                    length = pos - prev_pos
                    if include_next_token and pos < seq_len:
                        length += 1
                    patch_lengths[b, i] = min(length, seq_len - prev_pos - 1)
                prev_pos = int(pos)

        return patch_lengths


def calculate_entropies(
    tokens: torch.Tensor,
    entropy_model: nn.Module,
    patching_batch_size: int = 1,
    device: str = "cuda",
    enable_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate entropy scores for tokens using an entropy model.

    Args:
        tokens: Input tokens [batch_size, seq_len]
        entropy_model: Model for predicting next-token distributions
        patching_batch_size: Batch size for entropy calculation
        device: Device to use
        enable_grad: Whether to enable gradients

    Returns:
        Tuple of:
            - entropy_scores: Entropy at each position [batch_size, seq_len]
            - entropy_preds: Raw predictions [batch_size, seq_len * vocab_size]
    """
    batch_size, seq_len = tokens.shape

    context = torch.enable_grad() if enable_grad else torch.no_grad()

    with context:
        # Get predictions from entropy model
        logits = entropy_model(tokens)  # [batch_size, seq_len, vocab_size]

        # Calculate entropy from logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_scores = -(probs * log_probs).sum(dim=-1)  # [batch_size, seq_len]

        # Flatten predictions for loss calculation
        vocab_size = logits.shape[-1]
        entropy_preds = logits.reshape(batch_size, seq_len * vocab_size)

    return entropy_scores, entropy_preds


def patch_ids_from_lengths(
    patch_lengths: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """
    Convert patch lengths to patch IDs for each token.

    Args:
        patch_lengths: Tensor of patch lengths [batch_size, num_patches]
        seq_len: Original sequence length

    Returns:
        Tensor mapping each token to its patch ID [batch_size, seq_len]
    """
    batch_size, num_patches = patch_lengths.shape
    device = patch_lengths.device

    # Vectorized approach to avoid loops and .item() calls
    # Create cumulative sum of patch lengths
    cumsum = torch.cumsum(patch_lengths, dim=1)  # [batch_size, num_patches]

    # Create position indices
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]

    # For each position, find which patch it belongs to by comparing with cumsum
    # We add a leading zero to cumsum for the comparison
    cumsum_with_zero = torch.cat([
        torch.zeros(batch_size, 1, device=device, dtype=cumsum.dtype),
        cumsum
    ], dim=1)  # [batch_size, num_patches + 1]

    # Expand dimensions for broadcasting
    positions_expanded = positions.unsqueeze(2)  # [batch_size, seq_len, 1]
    cumsum_expanded = cumsum_with_zero.unsqueeze(1)  # [batch_size, 1, num_patches + 1]

    # Find which patch each position belongs to
    # A position belongs to patch p if cumsum[p-1] <= position < cumsum[p]
    patch_ids = (positions_expanded >= cumsum_expanded[..., :-1]) & (positions_expanded < cumsum_expanded[..., 1:])

    # Convert boolean mask to patch indices
    patch_indices = torch.arange(num_patches, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches]
    patch_ids = (patch_ids * patch_indices).sum(dim=2)  # [batch_size, seq_len]

    return patch_ids


def decoder_patch_ids_from_lengths(
    patch_lengths: torch.Tensor, nb_boe: int, seq_len: int
) -> torch.Tensor:
    """
    Create patch IDs for decoder with optional beginning-of-entity tokens.

    Args:
        patch_lengths: Tensor of patch lengths [batch_size, num_patches]
        nb_boe: Number of BOE tokens to prepend
        seq_len: Original sequence length

    Returns:
        Tensor of decoder patch IDs [batch_size, seq_len + nb_boe]
    """
    batch_size = patch_lengths.shape[0]
    device = patch_lengths.device

    # Get regular patch IDs
    patch_ids = patch_ids_from_lengths(patch_lengths, seq_len)

    # Add BOE tokens if needed
    if nb_boe > 0:
        boe_ids = torch.zeros(batch_size, nb_boe, device=device, dtype=torch.long)
        patch_ids = torch.cat([boe_ids, patch_ids], dim=1)

    return patch_ids


def cross_attn_mask(
    patch_ids: torch.Tensor,
    patch_lengths: torch.Tensor,
    seq_len: int,
    patches_as_queries: bool = True,
    cross_attn_k: int = 1,
    window: int = 512,
    block_mask: bool = False,
) -> Optional[torch.Tensor]:
    """
    Create cross-attention mask between patches and tokens.

    Args:
        patch_ids: Patch IDs for each token [batch_size, seq_len]
        patch_lengths: Length of each patch [batch_size, num_patches]
        seq_len: Original sequence length
        patches_as_queries: Whether patches are queries (vs keys)
        cross_attn_k: Cross-attention multiplier
        window: Attention window size
        block_mask: Whether to use block masking

    Returns:
        Cross-attention mask tensor or None
    """
    # Simplified implementation - return None for now
    # Full implementation would create proper attention masks
    return None


def concat_downsample(h: torch.Tensor, patch_lengths: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Downsample by concatenating embeddings within patches.

    Args:
        h: Hidden states [batch_size, seq_len, dim]
        patch_lengths: Length of each patch [batch_size, num_patches]
        patch_size: Size of each patch

    Returns:
        Downsampled tensor [batch_size, num_patches, dim * patch_size]
    """
    bs, seq_len, emb_dim = h.shape
    patch_end_ids = torch.cumsum(patch_lengths, dim=1)
    patch_ids = patch_end_ids.unsqueeze(-1) - torch.arange(patch_size, 0, -1).to(
        patch_end_ids.device
    )
    # Clamp to valid indices
    patch_ids = patch_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])
    patch_ids = patch_ids.view(bs, -1, emb_dim)
    # Gather tokens for each patch
    h = torch.gather(h, 1, patch_ids)
    h = h.reshape(bs, patch_lengths.shape[1], patch_size * h.size(-1))
    return h


def patch_reduce(
    h: torch.Tensor, max_num_patches: int, reduction: str, patch_ids: torch.Tensor
) -> torch.Tensor:
    """
    Reduce variable length patches to single embedding per patch.

    Args:
        h: Hidden states [batch_size, seq_len, dim]
        max_num_patches: Maximum number of patches
        reduction: Reduction method ('mean', 'amax', 'amin')
        patch_ids: Patch IDs for each token [batch_size, seq_len]

    Returns:
        Reduced embeddings [batch_size, max_num_patches, dim]
    """
    bs, seq_len, emb_dim = h.shape

    patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])

    reduced_embs = torch.zeros(
        (bs, max_num_patches, emb_dim), dtype=h.dtype, device=h.device
    )
    reduced_embs = reduced_embs.scatter_reduce(
        src=h,
        dim=1,
        index=patch_ids,
        reduce=reduction,
        include_self=False,
    )
    reduced_embs = reduced_embs[:, :max_num_patches, :]

    return reduced_embs