"""Byte-level patching utilities for sequence compression."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

from .constants import BOE_ID, BPE_ID, EOS_ID, OFFSET, PAD_ID


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
    threshold_add: Optional[float] = None
    max_patch_length: Optional[int] = None
    monotonicity: bool = False
    device: str = "cuda"
    realtime_patching: bool = False
    log_time: bool = False


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
                tokens,
                entropies,
                threshold or self.config.threshold,
                include_next_token,
            )
        elif self.config.patching_mode == PatchingMode.space:
            return self._space_patching(tokens, include_next_token)
        elif self.config.patching_mode == PatchingMode.static:
            return self._static_patching(tokens, self.config.patch_size, include_next_token)
        else:
            # Default to static patching
            return self._static_patching(tokens, self.config.patch_size, include_next_token)

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

        # Ensure monotonicity if configured (BLT reference logic)
        if self.config.monotonicity:
            # Boundaries only at positions where entropy *increases* by more than threshold
            differences = entropies[:, 1:] - entropies[:, :-1]
            monotonic_boundaries = differences > threshold
            # Prepend the first position's boundary (unchanged)
            boundaries = torch.cat(
                [boundaries[:, :1], monotonic_boundaries | boundaries[:, 1:]], dim=1
            )

        # Calculate patch lengths from boundaries
        patch_lengths = self._boundaries_to_lengths(boundaries, include_next_token)

        return self._postprocess_patch_lengths(
            patch_lengths, tokens, include_next_token, scores=entropies
        )

    def _space_patching(
        self, tokens: torch.Tensor, include_next_token: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create patches at space boundaries using exact BLT reference logic.

        Args:
            tokens: Input tokens [batch_size, seq_len]
            include_next_token: Whether to include next token

        Returns:
            Tuple of patch lengths and boundaries
        """
        # Use exact BLT reference implementation
        patch_start_ids = find_space_patch_start_ids(tokens)
        seq_len_with_next = tokens.shape[1] + (1 if include_next_token else 0)
        patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_with_next)

        # Create boundaries for compatibility (though not used in BLT)
        boundaries = torch.zeros_like(tokens, dtype=torch.bool)
        # Mark first positions as boundaries for visualization
        boundaries[:, 0] = True
        if patch_start_ids.shape[1] > 1 and patch_start_ids[0, 1] < tokens.shape[1]:
            for b in range(tokens.shape[0]):
                for p in range(1, patch_start_ids.shape[1]):
                    start_pos = patch_start_ids[b, p].item()
                    if start_pos < tokens.shape[1]:
                        boundaries[b, start_pos] = True

        return self._postprocess_patch_lengths(
            patch_lengths, tokens, include_next_token, scores=None
        )

    def _postprocess_patch_lengths(
        self,
        patch_lengths: torch.Tensor,
        tokens: torch.Tensor,
        include_next_token: bool,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Post-process patch lengths following BLT reference logic.

        Args:
            patch_lengths: Raw patch lengths
            tokens: Original tokens
            include_next_token: Whether next token was included
            scores: Optional scores (entropy, etc.)

        Returns:
            Tuple of (processed_patch_lengths, scores)
        """
        # Apply max_patch_length processing if needed
        if self.config.max_patch_length is not None:
            # Convert to list for processing (following BLT TODO note)
            patch_lengths_list = [
                self._split_large_patches(pl.tolist(), self.config.max_patch_length)
                for pl in patch_lengths
            ]
            max_len = max(len(pl) for pl in patch_lengths_list)
            patch_lengths_list = [
                pl + [0] * (max_len - len(pl))
                for pl in patch_lengths_list  # Right pad with zeros
            ]
            patch_lengths = torch.tensor(
                patch_lengths_list, dtype=tokens.dtype, device=tokens.device
            )

        # Ensure no non-zero values after zero values (BLT validation)
        assert not self._check_non_zero_after_zero(
            patch_lengths
        ), "Non-zero values found after zero values"

        # Trim trailing zero columns (BLT reference logic)
        if patch_lengths.numel() > 0:
            last_non_zero_col_reversed = (
                (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
            )
            patch_lengths = patch_lengths[
                :, : patch_lengths.shape[1] - last_non_zero_col_reversed
            ]

        # Critical validation (BLT reference)
        expected_total = tokens.numel() + include_next_token * tokens.shape[0]
        actual_total = torch.sum(patch_lengths)
        assert (
            actual_total == expected_total
        ), f"Patch length sum mismatch: {actual_total} != {expected_total}"

        return patch_lengths, scores

    def _split_large_patches(self, patch_lengths: list, max_length: int) -> list:
        """Split patches that exceed max_length."""
        result = []
        for length in patch_lengths:
            if length <= max_length:
                result.append(length)
            else:
                # Split into chunks of max_length
                while length > max_length:
                    result.append(max_length)
                    length -= max_length
                if length > 0:
                    result.append(length)
        return result

    def _check_non_zero_after_zero(self, patch_lengths: torch.Tensor) -> bool:
        """Check if there are non-zero values after zero values (should not happen)."""
        for row in patch_lengths:
            found_zero = False
            for val in row:
                if val == 0:
                    found_zero = True
                elif found_zero and val != 0:
                    return True
        return False

    def _static_patching(
        self, tokens: torch.Tensor, patch_size: int, include_next_token: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create fixed-size patches.

        Args:
            tokens: Input tokens [batch_size, seq_len]
            patch_size: Size of each patch
            include_next_token: Whether to include next token in patches

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

        return self._postprocess_patch_lengths(
            patch_lengths, tokens, include_next_token, scores=None
        )

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
        patch_lengths = torch.zeros(
            batch_size, max_patches, device=device, dtype=torch.long
        )

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


def patch_ids_from_lengths(patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
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
    positions = (
        torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )  # [batch_size, seq_len]

    # For each position, find which patch it belongs to by comparing with cumsum
    # We add a leading zero to cumsum for the comparison
    cumsum_with_zero = torch.cat(
        [torch.zeros(batch_size, 1, device=device, dtype=cumsum.dtype), cumsum], dim=1
    )  # [batch_size, num_patches + 1]

    # Expand dimensions for broadcasting
    positions_expanded = positions.unsqueeze(2)  # [batch_size, seq_len, 1]
    cumsum_expanded = cumsum_with_zero.unsqueeze(1)  # [batch_size, 1, num_patches + 1]

    # Find which patch each position belongs to
    # A position belongs to patch p if cumsum[p-1] <= position < cumsum[p]
    patch_ids = (positions_expanded >= cumsum_expanded[..., :-1]) & (
        positions_expanded < cumsum_expanded[..., 1:]
    )

    # Convert boolean mask to patch indices
    patch_indices = (
        torch.arange(num_patches, device=device).unsqueeze(0).unsqueeze(0)
    )  # [1, 1, num_patches]
    patch_ids = (patch_ids * patch_indices).sum(dim=2)  # [batch_size, seq_len]

    return patch_ids


def decoder_patch_ids_from_lengths(
    patch_lengths: torch.Tensor, nb_boe: int, seq_len: int
) -> torch.Tensor:
    """
    Create patch IDs for decoder using exact BLT reference logic.

    BLT removes the first patch for decoder inputs and adjusts subsequent patches.
    This is critical for proper alignment between encoder and decoder.

    Args:
        patch_lengths: Tensor of patch lengths [batch_size, num_patches]
        nb_boe: Number of BOE tokens (used for validation)
        seq_len: Original sequence length

    Returns:
        Tensor of decoder patch IDs [batch_size, seq_len]
    """
    # BLT reference validation
    first_patch_length = patch_lengths[0, 0]
    assert torch.all(
        first_patch_length == patch_lengths[:, 0]
    ), "first patch should always be the same size (1 for dynamic, patch_size for static)."

    assert (
        first_patch_length - nb_boe == 1
    ), f"First patch (patch length: {first_patch_length}) should have one non-boe token (boe toks: {nb_boe})"

    # Remove first patch from patch_ids for local decoder inputs (BLT reference logic)
    decoder_patch_lengths = patch_lengths[:, 1:]

    # Validation that sums work out correctly
    assert (
        decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]
        == patch_lengths.sum()
    ), f"{decoder_patch_lengths.sum() + (nb_boe + 1) * patch_lengths.shape[0]} != {patch_lengths.sum()}"

    assert torch.all(decoder_patch_lengths >= 0), f"{decoder_patch_lengths}"

    # Generate patch IDs from reduced patch lengths
    decoder_patch_ids = patch_ids_from_lengths(
        patch_lengths=decoder_patch_lengths, seq_len=seq_len
    )

    return decoder_patch_ids


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
    # Not yet implemented - cross-attention mode is non-functional
    logger.warning(
        "cross_attn_mask() is a stub and returns None. "
        "Cross-attention encoder/decoder mode is not yet implemented."
    )
    return None


def concat_downsample(
    h: torch.Tensor, patch_lengths: torch.Tensor, patch_size: int
) -> torch.Tensor:
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


def get_blt_input(
    tokens: torch.Tensor,
    enforce_patch_size_multiple: bool,
    nb_boe: int,
    patch_size: int,
    boe_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create proper token streams for BLT encoder, global, and decoder.

    This function replicates the BLT reference logic for creating separate
    token streams with proper BOE token handling and lag compensation.

    Args:
        tokens: Input tokens [batch_size, seq_len]
        enforce_patch_size_multiple: Whether to enforce patch size multiple
        nb_boe: Number of BOE tokens to prepend
        patch_size: Size of patches (for static patching)
        boe_id: BOE token ID

    Returns:
        Tuple of (local_encoder_tokens, global_tokens, local_decoder_tokens)
    """
    bs, N = tokens.shape
    device = tokens.device

    if nb_boe > 0:
        # Create BOE tokens to prepend
        boe_tokens = torch.full((bs, nb_boe), boe_id, dtype=tokens.dtype, device=device)

        # Local encoder tokens: BOE + original tokens
        local_encoder_tokens = torch.cat([boe_tokens, tokens], dim=1)

        # Global tokens: same as encoder tokens initially
        global_tokens = local_encoder_tokens.clone()

        # Local decoder tokens: original tokens (no BOE)
        local_decoder_tokens = tokens
    else:
        # No BOE tokens needed
        local_encoder_tokens = tokens
        global_tokens = tokens
        local_decoder_tokens = tokens

    # Handle padding if needed for patch size multiple
    if enforce_patch_size_multiple and patch_size > 1:
        encoder_len = local_encoder_tokens.shape[1]
        padding_needed = (patch_size - (encoder_len % patch_size)) % patch_size

        if padding_needed > 0:
            pad_tokens = torch.full(
                (bs, padding_needed), PAD_ID, dtype=tokens.dtype, device=device
            )
            local_encoder_tokens = torch.cat([local_encoder_tokens, pad_tokens], dim=1)
            global_tokens = torch.cat([global_tokens, pad_tokens], dim=1)

    return local_encoder_tokens, global_tokens, local_decoder_tokens


def find_space_patch_start_ids(tokens: torch.Tensor) -> torch.Tensor:
    """
    Find space patch start IDs using exact BLT reference logic.

    Args:
        tokens: Input tokens [batch_size, seq_len]

    Returns:
        Patch start IDs [batch_size, max_patches]
    """
    bs, seq_len = tokens.shape
    tokens_no_offset = tokens - OFFSET
    patch_end_mask = (
        (tokens_no_offset < ord("0"))
        | ((ord("9") < tokens_no_offset) & (tokens_no_offset < ord("A")))
        | ((ord("Z") < tokens_no_offset) & (tokens_no_offset < ord("a")))
        | ((ord("z") < tokens_no_offset) & (tokens_no_offset < 0b1000_0000))
        | (0b1100_0000 <= tokens_no_offset)
    )
    patch_end_mask[:, 1:] &= patch_end_mask[:, :-1].bitwise_not()
    patch_end_mask |= tokens < OFFSET

    patch_start_mask = torch.cat(
        [
            torch.tensor([1, 1], device=tokens.device, dtype=torch.bool)
            .unsqueeze(0)
            .repeat(bs, 1),
            patch_end_mask[:, 1:],
        ],
        dim=1,
    )
    max_patches = patch_start_mask.sum(dim=1).max()

    patch_ids = (
        torch.arange(seq_len + 1, device=tokens.device).unsqueeze(0).repeat(bs, 1)
    )
    extra_patch_ids = torch.full(
        (bs, seq_len + 1), seq_len + 1, dtype=torch.long, device=tokens.device
    )
    all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
    patch_start_mask_padded = torch.cat((patch_start_mask, ~patch_start_mask), dim=1)

    patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(bs, -1)[
        :, :max_patches
    ]
    return patch_start_ids


def patch_lengths_from_start_ids(
    patch_start_ids: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """
    Calculate patch lengths from start IDs using exact BLT reference logic.

    Args:
        patch_start_ids: Patch start IDs [batch_size, max_patches]
        seq_len: Sequence length

    Returns:
        Patch lengths [batch_size, max_patches]
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    return patch_lengths
