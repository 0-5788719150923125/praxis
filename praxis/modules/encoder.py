import math
import os
import random
from typing import List, Optional, Tuple, TypeVar, Union

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"
os.environ["BLT_ALLOW_MISSING_FLEX_ATTENTION"] = "1"

import bytelatent
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from bytelatent.data.ngram_processor import NgramProcessor
from bytelatent.data.patcher import Patcher, PatcherArgs, calculate_entropies
from bytelatent.model.blt import (
    ByteLatentTransformerArgs,
    EmbeddingType,
    compute_hash_embeddings,
    cross_attn_mask,
    decoder_patch_ids_from_lengths,
    get_decoder_dim_token_emb,
    get_encoder_dim_patch_emb,
    get_encoder_dim_token_emb,
    init_embeddings,
    parse_ngram_to_size,
    patch_ids_from_lengths,
)
from bytelatent.model.local_models import LocalDecoder, LocalEncoder, LocalModelArgs
from bytelatent.model.utils import concat_downsample, patch_reduce
from bytelatent.tokenizers.constants import BYTE_UNITS, EOS_ID, OFFSET
from bytelatent.transformer import LMTransformer, LMTransformerArgs
from torch import nn

from praxis.modules.recurrent import minGRU

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class PraxisEncoder(nn.Module):
    """
    An implementation of the Byte Latent Encoder/Decoder, from:
    https://arxiv.org/abs/2412.09871

    TODO: This code is an absolute mess. Both this repo and BLT are in active development, so
    it has been difficult to standardize. This could be a lot cleaner.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize byte latent encoder/decoder.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.debug: bool = config.debug
        self.log_rate: float = 0.005
        self.device_map = config.device_map
        self.args = create_base_args(config)
        self.args.dim_token_emb = config.embed_size

        self.entropy_model: Optional[nn.Module] = None
        self.args.patching_mode = "entropy" if "entropy" in config.meta else "space"
        self.args.patching_threshold = math.pi
        self.args.patch_size = 6
        if self.args.patching_mode == "entropy":
            self.args.monotonicity = True
            if "original" in config.meta:
                self.entropy_model = LMTransformer(create_entropy_model_args(self.args))
            elif "conv" in config.meta:
                self.entropy_model = ConvEntropyModel(
                    self.args.vocab_size, config.hidden_size, config.dropout, n_layers=2
                )
            else:
                self.entropy_model = RecurrentEntropyModel(
                    self.args.vocab_size, config.hidden_size, config.dropout, n_layers=1
                )
            # Threshold optimization parameters
            self.loss_scale: float = 1.0
            self.target_ratio: float = 0.125  # reduction of original length
            # Register buffers for both current and EMA thresholds
            self.register_buffer(
                "optimal_threshold",
                torch.tensor(self.args.patching_threshold, dtype=torch.float32),
            )

        self.patcher: Patcher = Patcher(
            PatcherArgs(
                realtime_patching=False,
                device=self.device_map,
                patch_size=self.args.patch_size,
                patching_mode=self.args.patching_mode,
                threshold=self.args.patching_threshold,
                threshold_add=None,
                monotonicity=self.args.monotonicity,
            )
        )
        self.patcher.entropy_model = self.entropy_model
        self.hash_embeds: Optional[nn.Module] = None
        if "no_hash" in config.meta:
            self.hash_embeds = None
        else:
            self.hash_embeds = init_embeddings(
                self.args,
                EmbeddingType.HASH_TOK,
                local_encoder_dim=self.args.dim_token_emb,
                encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
            )
        self.token_proj: Optional[nn.Linear] = None
        if self.args.dim_token_emb != self.args.dim:
            self.token_proj = nn.Linear(
                self.args.dim_token_emb,
                config.hidden_size,
                bias=False,
            )
        if "original" in config.meta:
            self.encoder = LocalEncoder(create_local_encoder_args(self.args))
            self.decoder = LocalDecoder(create_local_decoder_args(self.args))
        elif "conv" in config.meta:
            self.args.n_layers_local_encoder = 3
            self.args.n_layers_local_decoder = 3
            self.encoder = ConvEncoder(create_local_encoder_args(self.args))
            self.decoder = ConvDecoder(create_local_decoder_args(self.args))
        else:
            self.encoder = RecurrentEncoder(create_local_encoder_args(self.args))
            self.decoder = RecurrentDecoder(create_local_decoder_args(self.args))

    def __repr__(self) -> str:
        """
        String representation of the encoder module.

        Returns:
            String representation
        """
        return (
            f"{self.__class__.__name__}("
            + f"type='byte_latent', "
            + f"n_encoders={len(self.encoder.layers)}, "
            + f"n_decoders={len(self.decoder.layers)})"
        )

    def encode(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Encode input tokens into latent representation.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]

        Returns:
            Tuple containing:
                - Encoded hidden states
                - Encoder output hidden states
                - Patch lengths tensor
                - Block IDs tensor
                - Auxiliary loss value
        """
        aux_loss: float = 0
        best_threshold = None
        if self.entropy_model is None:
            # Space patching mode
            patch_lengths, _ = self.patcher.patch(input_ids, include_next_token=True)
        else:
            # Entropy patching mode
            entropy_scores, entropy_preds = calculate_entropies(
                tokens=input_ids,
                entropy_model=self.entropy_model,
                patching_batch_size=input_ids.size(0),
                device=self.device_map,
                enable_grad=True,
            )
            modified_entropy_scores = patch_entropies_for_special_tokens(
                input_ids, entropy_scores
            )
            if self.training:
                # First, find a safe threshold that guarantees we're under target length
                safe_threshold = self._find_safe_threshold(
                    input_ids, modified_entropy_scores
                )
                # Now sample thresholds that are guaranteed to be safe
                n_samples = 30
                min_candidates = 15
                # Sample only multipliers >= 1.0 to stay above safe_threshold
                multipliers = 1.0 + torch.abs(
                    torch.normal(
                        mean=0.0,
                        std=1.0,
                        size=(n_samples - 1,),
                        device=input_ids.device,
                    )
                )

                # Always include the safe threshold itself
                multipliers = torch.cat(
                    [multipliers, torch.tensor([1.0], device=input_ids.device)]
                )

                candidates = safe_threshold * multipliers

                # Rest proceeds as before, but now we're guaranteed safe
                best_threshold = safe_threshold
                best_ratio = float("inf")
                best_patch_lengths = None
                target_ratio = self.target_ratio

                # Try each candidate
                for i, threshold in enumerate(candidates):

                    # early exit
                    if i > min_candidates and best_patch_lengths is not None:
                        break

                    # create the patches
                    patch_lengths, _ = self.patcher.patch(
                        input_ids,
                        include_next_token=False,
                        threshold=abs(threshold),
                        entropies=modified_entropy_scores,
                    )

                    actual_ratio = patch_lengths.shape[1] / input_ids.shape[1]
                    distance = abs(actual_ratio - target_ratio)

                    if distance < best_ratio and actual_ratio <= target_ratio:
                        best_ratio = distance
                        best_threshold = abs(threshold)
                        best_patch_lengths = patch_lengths

                # Use best values
                self.optimal_threshold.fill_(best_threshold)
                patch_lengths = best_patch_lengths

                # Compute cross entropy loss
                modified_entropy_preds = mask_entropy_preds_at_special_tokens(
                    input_ids, entropy_preds
                )

                # Calculate vocab_size
                batch_size, seq_len = input_ids.shape
                _, total_size = modified_entropy_preds.shape
                vocab_size = total_size // seq_len

                # Reshape entropy_preds to [batch_size, seq_len, vocab_size] for easier handling
                reshaped_preds = modified_entropy_preds.view(
                    batch_size, seq_len, vocab_size
                )

                # Reshape to [batch_size * (seq_len-1), vocab_size]
                flattened_preds = reshaped_preds[:, :-1].reshape(-1, vocab_size)

                # Flatten the target tokens (positions 1 to seq_len-1)
                flattened_targets = input_ids[:, 1:].reshape(-1)

                # Calculate the loss
                aux_loss = (
                    F.cross_entropy(flattened_preds, flattened_targets)
                    * self.loss_scale
                )

            else:
                # During inference, use stored optimal threshold
                patch_lengths, _ = self.patcher.patch(
                    input_ids,
                    include_next_token=True,
                    threshold=float(self.optimal_threshold.item()),
                    entropies=modified_entropy_scores,
                )

        if self.training:
            if self.debug and random.random() < self.log_rate:
                avg_patch_size = patch_lengths.float().mean().item()
                print(
                    f"DEBUG: length={input_ids.size(1)}, reduced length={patch_lengths.shape[1]}, "
                    f"avg patch size={avg_patch_size:.2f}, "
                    f"patching threshold={best_threshold or self.args.patching_threshold:.4f}"
                )

        patch_ids = patch_ids_from_lengths(patch_lengths, input_ids.shape[-1])

        # Create cross attention mask if needed
        cross_attn_mask_enc = None
        if self.args.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                input_ids.shape[-1],  # N
                patches_as_queries=True,
                cross_attn_k=self.args.cross_attn_k,
                window=self.args.cross_attn_window_encoder,
                block_mask=False,
            )

        # Compute embeddings
        hash_embeds = compute_hash_embeddings(
            local_encoder_tokens=input_ids,
            local_encoder=self.encoder,
            encoder_hash_tok_embedding=self.hash_embeds,
            encoder_hash_byte_group_nb_functions=self.args.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.args.encoder_hash_byte_group_vocab,
        )

        # Local encoder with cross attention
        (h_encoder, h_cross), _ = self.encoder(
            tokens=input_ids,
            embeds=hash_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
            mask=None,
        )

        # Downsampling
        if self.args.cross_attn_encoder:
            h = h_cross.view(h_encoder.shape[0], -1, h_encoder.shape[-1])
        else:
            h = downsample(
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths,
                patch_ids,
                downsampling_by_pooling=self.args.downsampling_by_pooling,
                patch_size=self.args.patch_size,
            )

        block_ids = create_patch_block_ids(input_ids, patch_lengths, patch_ids)

        if self.token_proj is not None:
            h = self.token_proj(h)

        return h, h_encoder, patch_lengths, block_ids, aux_loss

    def decode(
        self,
        h: torch.Tensor,
        h_encoder: torch.Tensor,
        input_ids: torch.Tensor,
        patch_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent representation back to token space.

        Args:
            h: Hidden states to decode
            h_encoder: Encoder output hidden states
            input_ids: Original input token IDs
            patch_lengths: Lengths of patches

        Returns:
            Decoded output tensor
        """
        nb_boe: int = 0
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, input_ids.shape[-1]
        )

        # Upsampling
        cross_mask = None
        if self.args.cross_attn_decoder:
            h = (
                h
                if self.args.cross_attn_encoder
                else h.repeat_interleave(self.args.cross_attn_k, dim=1)
            )
            cross_mask = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                input_ids.shape[-1],
                patches_as_queries=False,
                cross_attn_k=self.args.cross_attn_k,
                window=self.args.cross_attn_window_decoder,
                block_mask=False,
            )
        else:
            h = torch.gather(
                h, 1, decoder_patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])
            )

        output, _ = self.decoder(
            tokens=input_ids,
            patch_embeds=h,
            embeds=h_encoder,
            cross_mask=cross_mask,
            mask=None,
        )

        return output

    def _find_safe_threshold(
        self, input_ids: torch.Tensor, entropy_scores: torch.Tensor
    ) -> float:
        """
        Find a safe threshold for entropy-based patching.

        Args:
            input_ids: Input token IDs
            entropy_scores: Entropy scores for tokens

        Returns:
            Safe threshold value

        Raises:
            ValueError: If a working threshold cannot be found
        """
        # Start with current threshold
        threshold = float(self.optimal_threshold.item())
        target_len = int(input_ids.shape[1] * self.target_ratio)

        # First, find any working threshold by doubling until we succeed
        while True:
            patch_lengths, _ = self.patcher.patch(
                input_ids,
                include_next_token=False,
                threshold=threshold,
                entropies=entropy_scores,
            )

            if patch_lengths.shape[1] <= target_len:
                # Found a working threshold
                safe_threshold = threshold
                break

            # If still too long, double the threshold
            threshold *= 2.0

            # Safety check - if we've increased too much, something is wrong
            if threshold > 1000.0:  # Arbitrary large number
                raise ValueError("Could not find a working threshold")

        # Now we can do binary search to optimize it
        left = threshold / 4.0  # Go back a bit to find better threshold
        right = threshold

        for _ in range(10):  # Usually converges in < 10 steps
            mid = (left + right) / 2
            patch_lengths, _ = self.patcher.patch(
                input_ids,
                include_next_token=False,
                threshold=mid,
                entropies=entropy_scores,
            )

            current_len = patch_lengths.shape[1]
            if current_len > target_len:
                # Sequence too long, need higher threshold
                left = mid
            else:
                # Sequence acceptable, but might be able to go lower
                right = mid
                safe_threshold = mid  # Keep track of last working threshold

            if abs(right - left) < 1e-4:
                break

        return safe_threshold


def downsample(
    h: torch.Tensor,
    num_patches: int,
    patch_lengths: Optional[torch.Tensor] = None,
    patch_ids: Optional[torch.Tensor] = None,
    downsampling_by_pooling: Optional[str] = None,
    patch_size: int = 4,
) -> torch.Tensor:
    """
    Downsample hidden representations through pooling or concatenation.

    Two options are available:
        a. concatenating embeddings in the patch
            Note: with dynamic patching, patch the last patch_size tokens.
        b. pooling embeddings in the patch

    Args:
        h: Hidden states tensor of shape [batch_size, seq_len, dim]
        num_patches: Number of patches to create
        patch_lengths: Optional patch lengths tensor
        patch_ids: Optional patch IDs tensor
        downsampling_by_pooling: Pooling method to use (if any)
        patch_size: Size of patches for fixed-length patching

    Returns:
        Downsampled tensor
    """
    # input: h.shape = [batch_size, seq_len, dim]
    # input: pool h.shape = [batch_size, seq_len / patch_size, dim]
    # if we don't use the cross_attn, we pool so that we convert bytes rep to patch rep
    if downsampling_by_pooling is not None and len(downsampling_by_pooling) > 0:
        # By pooling
        max_num_patches = num_patches
        assert (
            patch_ids is not None
        ), "Patch IDs must be provided for pooling-based downsampling"
        h = pooling_downsample(h, max_num_patches, downsampling_by_pooling, patch_ids)
    else:
        # TODO: remove this condition
        # By concatenating (fixed lengths patching)
        assert (
            patch_lengths is not None
        ), "Patch lengths must be provided for concatenation-based downsampling"
        h = concat_downsample(h, patch_lengths, patch_size)
    return h


def pooling_downsample(
    h: torch.Tensor, max_num_patches: int, pooling_mode: str, patch_ids: torch.Tensor
) -> torch.Tensor:
    """
    Downsample hidden representations using various pooling methods.

    Args:
        h: Hidden states tensor of shape [batch_size, seq_len, dim]
        max_num_patches: Maximum number of patches
        pooling_mode: Pooling method to use ("avg", "min", "max", or "topk:N")
        patch_ids: Patch IDs tensor of shape [batch_size, seq_len]

    Returns:
        Pooled tensor of shape [batch_size, max_num_patches, dim*n_modes]
        where n_modes is the number of pooling methods used

    Raises:
        AssertionError: If no pooling method was applied
    """
    cat: List[torch.Tensor] = []
    if "avg" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "mean", patch_ids))
    if "min" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amin", patch_ids))
    if "max" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amax", patch_ids))
    if pooling_mode.startswith("topk:"):
        k = int(pooling_mode.split(":")[1])
        cat.append(topk_mean_pooling(h, max_num_patches, patch_ids, k))

    assert len(cat) > 0, f"No pooling method was applied for mode: {pooling_mode}"
    h = torch.cat(cat, dim=-1)
    return h


def topk_mean_pooling(
    h: torch.Tensor, max_num_patches: int, patch_ids: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Calculate top-k mean pooling for each patch.

    The function groups input embeddings (h) into patches using patch_ids, selects
    the top k elements per patch (ignoring padded values), and returns their mean.
    It effectively summarizes each patch by averaging its most significant k embeddings.

    Args:
        h: Hidden states tensor of shape [batch_size, seq_len, dim]
        max_num_patches: Maximum number of patches
        patch_ids: Patch IDs tensor of shape [batch_size, seq_len]
        k: Number of top elements to select from each patch

    Returns:
        Pooled tensor of shape [batch_size, max_num_patches, dim]
    """
    bs, seq_len, emb_dim = h.shape
    device = h.device

    # Compute how many tokens fall into each patch.
    # one_hot: [bs, seq_len, max_num_patches], then sum along seq_len → [bs, max_num_patches]
    one_hot = F.one_hot(patch_ids, num_classes=max_num_patches).to(h.dtype)
    counts = one_hot.sum(dim=1)  # shape: [bs, max_num_patches]

    # Expand patch_ids so it can be used as an index for scatter_reduce.
    # We want to operate on h which is [bs, seq_len, emb_dim], so expand patch_ids to that shape.
    patch_ids_exp = patch_ids.unsqueeze(-1).expand(bs, seq_len, emb_dim)
    # Work on a clone of h so that we can mask out used tokens.
    h_work = h.clone()

    # We'll use NEG_INF as the value to mark masked-out tokens.
    NEG_INF = -1e9

    # We'll collect the k max values for each patch.
    collected = []
    for i in range(k):
        # Perform a scatter_reduce (max) to compute the current max for each patch and embedding.
        # We create a temporary tensor (current_max) of shape [bs, max_num_patches, emb_dim],
        # and scatter_reduce from h_work along the sequence dimension (dim=1) based on patch_ids_exp.
        current_max = torch.zeros(
            bs, max_num_patches, emb_dim, device=device, dtype=h.dtype
        )
        current_max = current_max.scatter_reduce(
            dim=1,
            index=patch_ids_exp,
            src=h_work,
            reduce="amax",
            include_self=False,
        )
        collected.append(current_max)

        # To mask out the positions that produced the current max, we broadcast current_max back
        # to token-level via gather. (For each token, pick the max of its corresponding patch.)
        current_max_expanded = current_max.gather(1, patch_ids_exp)
        # Create a mask for positions that equal the current max.
        mask = h_work == current_max_expanded
        # Mask out these positions in h_work so they won't be selected in subsequent iterations.
        h_work = h_work.masked_fill(mask, NEG_INF)

    # Stack the k rounds: shape [k, bs, max_num_patches, emb_dim]
    topk_vals = torch.stack(collected, dim=0)
    # Rearrange to shape [bs, max_num_patches, k, emb_dim]
    topk_vals = topk_vals.permute(1, 2, 0, 3)

    # Build an iteration index vector of shape [1, 1, k] to compare against counts.
    iter_range = torch.arange(k, device=device).view(1, 1, k)  # shape: [1, 1, k]
    # Expand counts to shape [bs, max_num_patches, 1]
    counts_exp = counts.unsqueeze(-1)
    valid_mask = iter_range < counts_exp  # shape: [bs, max_num_patches, k]
    # Convert mask to float so we can multiply.
    valid_mask = valid_mask.to(h.dtype)

    # Sum the valid top-k values and divide by the number of valid iterations.
    sum_valid = (topk_vals * valid_mask.unsqueeze(-1)).sum(
        dim=2
    )  # shape: [bs, max_num_patches, emb_dim]
    num_valid = valid_mask.sum(dim=2).clamp(min=1)  # shape: [bs, max_num_patches]
    pooled = sum_valid / num_valid.unsqueeze(-1)
    return pooled


def patch_entropies_for_special_tokens(
    input_ids: torch.LongTensor,
    entropy_scores: torch.Tensor,
    special_tokens: List[int] = [0],
    high_entropy_value: float = 1e9,
) -> torch.Tensor:
    """
    Forces patch boundaries at special tokens by setting their entropy scores high.

    Args:
        input_ids: Token IDs of shape [batch_size, seq_len]
        entropy_scores: Original entropy values of shape [batch_size, seq_len]
        special_tokens: List of special token IDs to mark boundaries
        high_entropy_value: Value to assign at special token positions

    Returns:
        Modified entropy scores tensor of shape [batch_size, seq_len]
    """
    # Convert special_tokens to tensor for isin operation
    special_tokens_tensor = torch.tensor(
        special_tokens, device=input_ids.device, dtype=input_ids.dtype
    )

    # Create special token mask using isin
    token_mask = torch.isin(input_ids, special_tokens_tensor)

    # Apply high entropy value where special tokens exist
    modified_entropy_scores = torch.where(
        token_mask,
        torch.tensor(
            high_entropy_value,
            device=entropy_scores.device,
            dtype=entropy_scores.dtype,
        ),
        entropy_scores,
    )

    return modified_entropy_scores


def mask_entropy_preds_at_special_tokens(
    input_ids: torch.LongTensor,
    entropy_preds: torch.Tensor,
    special_tokens: List[int] = [0],
) -> torch.Tensor:
    """
    Masks entropy predictions at special token positions by setting them to zero.
    This prevents the model from learning to predict across sequence boundaries.

    Args:
        input_ids: Token IDs of shape [batch_size, seq_len]
        entropy_preds: Original entropy predictions of shape [batch_size, seq_len * vocab_size]
        special_tokens: List of special token IDs to mask

    Returns:
        Masked entropy predictions tensor of shape [batch_size, seq_len * vocab_size]
    """
    # Get shapes
    batch_size, seq_len = input_ids.shape

    # Calculate vocab_size
    _, total_size = entropy_preds.shape
    vocab_size = total_size // seq_len

    # Create special token positions mask
    special_tokens_tensor = torch.tensor(
        special_tokens, device=input_ids.device, dtype=input_ids.dtype
    )
    special_token_mask = torch.isin(input_ids, special_tokens_tensor)

    # Create a mask for all vocab positions corresponding to special tokens
    # Initialize with zeros (will be set to 1 where special tokens are)
    full_mask = torch.zeros_like(entropy_preds, dtype=torch.bool)

    # For each position in the sequence that contains a special token,
    # we need to zero out all corresponding vocab entries
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            if special_token_mask[batch_idx, seq_idx]:
                # Calculate the starting and ending indices for this position in the flattened tensor
                start_idx = seq_idx * vocab_size
                end_idx = start_idx + vocab_size

                # Set the mask to True for all vocab entries at this position
                full_mask[batch_idx, start_idx:end_idx] = True

    # Apply the mask - set to zero where special tokens exist
    masked_entropy_preds = torch.where(
        full_mask, torch.zeros_like(entropy_preds), entropy_preds
    )

    # Return the masked tensor
    return masked_entropy_preds


def create_patch_block_ids(
    input_ids: torch.LongTensor,
    patch_lengths: torch.LongTensor,
    patch_ids: torch.LongTensor,
    special_tokens: List[int] = [0],
) -> torch.LongTensor:
    """
    Creates block IDs for patches, with boundaries after patches containing special tokens.
    Special token patches are part of the previous block.

    Args:
        input_ids: Token IDs of shape [batch_size, seq_len]
        patch_lengths: Length of each patch of shape [batch_size, num_patches]
        patch_ids: Mapping of tokens to patches of shape [batch_size, seq_len]
        special_tokens: List of special token IDs to mark boundaries

    Returns:
        Block IDs tensor of shape [batch_size, num_patches]
    """
    batch_size, seq_len = input_ids.shape
    _, num_patches = patch_lengths.shape

    # Create special token mask using isin
    special_tokens_tensor = torch.tensor(
        special_tokens, device=input_ids.device, dtype=input_ids.dtype
    )
    special_token_mask = torch.isin(input_ids, special_tokens_tensor).float()

    # Initialize tensor for accumulating special tokens per patch
    patch_special_counts = torch.zeros(
        (batch_size, num_patches), device=input_ids.device, dtype=torch.float
    )

    # Add up special tokens for each patch
    patch_special_counts.scatter_add_(
        1,  # scatter along patch dimension
        patch_ids,  # mapping from tokens to patches
        special_token_mask,  # which tokens are special
    )

    # Convert counts to boolean - any non-zero count means patch has special token
    patch_has_special = patch_special_counts > 0

    # Create boundaries only AFTER patches with special tokens
    patch_boundaries = torch.cat(
        [
            torch.ones(
                (batch_size, 1), device=patch_has_special.device, dtype=torch.bool
            ),  # First patch always starts a block
            patch_has_special[
                :, :-1
            ],  # Only create boundary after special token patches
        ],
        dim=1,
    )

    # Generate block ids
    patch_block_ids = patch_boundaries.cumsum(dim=1)

    return patch_block_ids


def packed_rnn_block(
    rnn: nn.Module, x: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int = 0
) -> torch.Tensor:
    """
    Efficiently use packed sequences within transformer architecture.

    This function identifies sequence lengths based on EOS tokens, creates packed
    sequences for efficient RNN processing, and then unpacks back to regular tensors.

    Args:
        rnn: nn.RNN module (or compatible RNN type like GRU, LSTM)
        x: Feature tensor from transformer of shape [batch_size, seq_len, features]
        input_ids: Token IDs to identify EOS positions of shape [batch_size, seq_len]
        eos_token_id: ID of EOS token to split on

    Returns:
        Processed tensor of shape [batch_size, seq_len, hidden_size] with same sequence length as input
    """
    batch_size, seq_len = input_ids.size()
    device = x.device

    # Find lengths based on EOS tokens
    lengths = torch.full((batch_size,), seq_len, device=device)

    # Find first EOS in each sequence
    for i in range(batch_size):
        eos_positions = (input_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # +1 to include the EOS token in the sequence
            lengths[i] = min(eos_positions[0].item() + 1, seq_len)

    # Create packed sequence
    packed_x = nn.utils.rnn.pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
    )

    # Process with RNN
    packed_output, _ = rnn(packed_x)

    # Unpack back to regular tensor
    output, _ = rnn_utils.pad_packed_sequence(
        packed_output, batch_first=True, total_length=seq_len
    )

    return output


class RecurrentBlock(nn.Module):
    """
    RecurrentBlock using GRU for efficient sequence processing.

    This block applies layer normalization followed by GRU processing, with a residual
    connection from input to output.
    """

    def __init__(self, dim: int, dim_out: Optional[int] = None, norm_eps: float = 1e-5):
        """
        Initialize RecurrentBlock.

        Args:
            dim: Input dimension
            dim_out: Output dimension (unused, kept for API compatibility)
            norm_eps: Epsilon for RMSNorm layer
        """
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Process input through RMSNorm, GRU and residual connection.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            input_ids: Token IDs for sequence delimiting of shape [batch_size, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        out = packed_rnn_block(
            self.gru, self.norm(x), input_ids=input_ids, eos_token_id=0
        )
        return out + x


# class RecurrentBlock(minGRU):
#     """
#     We replace transformer blocks in the encoder/decoder with something
#     that is more memory-efficient, and faster to compute.
#     """

#     def __init__(self, dim, dim_out=None, norm_eps=1e-5):
#         super().__init__(dim=dim, dim_out=dim_out, proj_out=True)
#         self.norm = nn.RMSNorm(dim, eps=norm_eps)

#     def forward(self, x: torch.Tensor, input_ids: torch.Tensor = None, *args, **kwargs):
#         out, _ = super().forward(self.norm(x), input_ids=input_ids)
#         return out + x


class RecurrentEncoder(LocalEncoder):
    """
    Recurrent encoder implementation using RecurrentBlock layers.

    This encoder replaces transformer blocks with recurrent blocks
    for more efficient processing of long sequences.
    """

    def __init__(self, args: LocalModelArgs):
        """
        Initialize RecurrentEncoder.

        Args:
            args: Local model arguments containing configuration
        """
        super().__init__(args)
        self.layers = nn.ModuleList(
            [RecurrentBlock(dim=args.dim) for _ in range(args.n_layers)]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[Tuple[torch.Tensor, None], None]:
        """
        Forward pass through the recurrent encoder.

        Args:
            tokens: Input token IDs of shape [batch_size, seq_len]
            embeds: Optional pre-computed embeddings

        Returns:
            Tuple containing:
                - Tuple of (hidden states, None)
                - None (for API compatibility)
        """
        h = self.apply_embedding(tokens, embeds)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            h = layer(h, input_ids=tokens)

        return (h, None), None


class RecurrentDecoder(LocalDecoder):
    """
    Recurrent decoder implementation using RecurrentBlock layers.

    This decoder replaces transformer blocks with recurrent blocks
    for more efficient processing of long sequences.
    """

    def __init__(self, args: LocalModelArgs):
        """
        Initialize RecurrentDecoder.

        Args:
            args: Local model arguments containing configuration
        """
        super().__init__(args)
        self.layers = nn.ModuleList(
            [RecurrentBlock(dim=args.dim_token_emb) for _ in range(args.n_layers)]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: torch.Tensor,
        patch_embeds: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through the recurrent decoder.

        Args:
            tokens: Input token IDs of shape [batch_size, seq_len]
            embeds: Pre-computed embeddings from encoder
            patch_embeds: Optional patch embeddings

        Returns:
            Tuple containing:
                - Output predictions of shape [batch_size, seq_len, vocab_size]
                - None (for API compatibility)

        Raises:
            AssertionError: If required embeddings are not provided
        """
        bs, seqlen = tokens.shape
        assert embeds is not None, "Embeddings must be provided"

        h = embeds

        if self.patch_embedding_projection is not None:
            assert patch_embeds is not None, "Patch embeddings must be passed."
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            if self.cross_attn_k is not None:
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        if patch_embeds is not None and not self.cross_attn_decoder:
            h = h + patch_embeds

        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            h = layer(h, input_ids=tokens)

        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.dropout, training=self.training)
        h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        return h_preds, None


class RecurrentEntropyModel(nn.Module):
    """
    Recurrent model for entropy prediction.

    Used for entropy-based patching to determine where to create patch boundaries
    based on prediction difficulty.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 256,
        dropout: float = 0,
        n_layers: int = 1,
    ):
        """
        Initialize RecurrentEntropyModel.

        Args:
            vocab_size: Size of the vocabulary
            dim: Hidden dimension size
            dropout: Dropout probability (unused, kept for API compatibility)
            n_layers: Number of recurrent layers
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim)  # byte embedding

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(RecurrentBlock(dim=dim, norm_eps=1e-5))

        # Project to byte probabilities
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the entropy model.

        Args:
            x: Input token IDs of shape [batch_size, seq_len]

        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # x: [batch, seq_len]
        input_ids = x
        x = self.embedding(x)  # [batch, seq_len, dim]

        for block in self.blocks:
            x = block(x, input_ids=input_ids)

        return self.output(self.norm(x))


class ConvBlock(nn.Module):
    """
    Convolutional block with same interface as RecurrentBlock.
    Uses dilated causal convolution with residual connection.
    """

    def __init__(
        self,
        dim: int,
        norm_eps: float = 1e-5,
        dilation: int = 0,
        padding: int = 0,
        kernel_size: int = 3,
    ):
        """
        Initialize ConvBlock.

        Args:
            dim: Input dimension
            norm_eps: Epsilon for RMSNorm layer
            dilation: Dilation factor for the convolutional layer
            padding: Padding amount for causal convolution
            kernel_size: Size of the convolutional kernel
        """
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=norm_eps)

        # Causal padding to match sequence length
        self.padding = padding

        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, dilation=dilation, padding=padding
        )

        self.activation = nn.ReLU()

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> torch.Tensor:
        """
        Process input through normalization, causal convolution, and residual connection.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            input_ids: Token IDs (unused, kept for API compatibility)

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # x: [batch, seq_len, dim]
        normed = self.norm(x)

        # Reshape for convolution
        conv_in = normed.transpose(1, 2)  # [batch, dim, seq_len]

        # Apply convolution
        conv_out = self.conv(conv_in)

        # Remove extra padding to maintain causality
        conv_out = conv_out[..., : -self.padding]

        # Reshape back
        out = conv_out.transpose(1, 2)  # [batch, seq_len, dim]

        return self.activation(out) + x


class ConvEncoder(LocalEncoder):
    """
    Convolutional encoder implementation using ConvBlock layers with dilated convolutions.

    This encoder uses a stack of dilated convolutional layers where each layer has
    exponentially increasing dilation factor to capture long-range dependencies
    while maintaining efficiency.
    """

    def __init__(self, args: LocalModelArgs):
        """
        Initialize ConvEncoder.

        Args:
            args: Local model arguments containing configuration
        """
        super().__init__(args)
        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            kernel_size = 3
            dilation = 2**i
            padding = (kernel_size - 1) * dilation  # Causal padding
            self.layers.append(
                ConvBlock(
                    args.dim,
                    args.norm_eps,
                    dilation,
                    padding,
                    kernel_size,
                ),
            )
        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[Tuple[torch.Tensor, None], None]:
        """
        Forward pass through the convolutional encoder.

        Args:
            tokens: Input token IDs of shape [batch_size, seq_len]
            embeds: Optional pre-computed embeddings

        Returns:
            Tuple containing:
                - Tuple of (hidden states, None)
                - None (for API compatibility)
        """
        h = self.apply_embedding(tokens, embeds)

        for layer in self.layers:
            h = self.dropout(h)
            h = layer(h, input_ids=tokens)

        return (h, None), None


class ConvDecoder(LocalDecoder):
    """
    Convolutional decoder implementation using ConvBlock layers.

    This decoder uses dilated causal convolutions to process sequences efficiently
    while maintaining autoregressive properties.
    """

    def __init__(self, args: LocalModelArgs):
        """
        Initialize ConvDecoder.

        Args:
            args: Local model arguments containing configuration
        """
        super().__init__(args)
        self.layers = nn.ModuleList()
        for i in range(args.n_layers):
            kernel_size = 3
            dilation = 2**i
            padding = (kernel_size - 1) * dilation  # Causal padding
            self.layers.append(
                ConvBlock(
                    args.dim,
                    args.norm_eps,
                    dilation,
                    padding,
                    kernel_size,
                ),
            )
        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: torch.Tensor,
        patch_embeds: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through the convolutional decoder.

        Args:
            tokens: Input token IDs of shape [batch_size, seq_len]
            embeds: Pre-computed embeddings from encoder
            patch_embeds: Optional patch embeddings

        Returns:
            Tuple containing:
                - Output predictions of shape [batch_size, seq_len, vocab_size]
                - None (for API compatibility)

        Raises:
            AssertionError: If required embeddings are not provided
        """
        bs, seqlen = tokens.shape
        assert embeds is not None, "Embeddings must be provided"

        h = embeds

        if self.patch_embedding_projection is not None:
            assert patch_embeds is not None, "Patch embeddings must be passed."
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            if self.cross_attn_k is not None:
                patch_embeds = patch_embeds.reshape(
                    bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
                )

        if patch_embeds is not None and not self.cross_attn_decoder:
            h = h + patch_embeds

        for layer in self.layers:
            h = self.dropout(h)
            h = layer(h, input_ids=tokens)

        h_preds = self.norm(h)
        h_preds = self.dropout(h_preds)
        h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        return h_preds, None


class ConvEntropyModel(nn.Module):
    """
    Convolutional model for entropy prediction.

    Uses a stack of dilated convolutions to efficiently model token sequences
    for entropy-based patching.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        channels: int = 256,
        dropout: float = 0,
        n_layers: int = 1,
        kernel_size: int = 3,
    ):
        """
        Initialize ConvEntropyModel.

        Args:
            vocab_size: Size of the vocabulary
            channels: Number of channels in the hidden layers
            dropout: Dropout probability
            n_layers: Number of convolutional layers
            kernel_size: Size of the convolutional kernel
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, channels)  # byte embedding

        # Stack of dilated convolutions
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation  # Causal padding
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.Dropout(dropout),
                )
            )

        self.activation = nn.ReLU()

        # Project to byte probabilities
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, vocab_size)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the convolutional entropy model.

        Args:
            x: Input token IDs of shape [batch_size, seq_len]

        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # x: [batch, seq_len]
        x = self.embedding(x).transpose(1, 2)  # [batch, channels, seq_len]

        # Causal convolution stack
        for conv in self.convs:
            out = conv(x)
            out = out[..., : x.size(-1)]
            x = self.activation(out) + x

        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        return self.output(self.norm(x))


def create_base_args(config: ConfigType) -> ByteLatentTransformerArgs:
    """
    Create base arguments for the ByteLatent model.

    Uses defaults from the original Facebook code:
    https://github.com/facebookresearch/blt/blob/main/bytelatent/test_blt.py

    Args:
        config: Model configuration object

    Returns:
        ByteLatentTransformerArgs with the configured parameters
    """
    downsampling_method = "max"
    if "min" in config.meta:
        downsampling_method = "min"
    elif "mean" in config.meta:
        downsampling_method = "mean"
    elif any(flag.startswith("topk:") for flag in config.meta):
        downsampling_method = next(x for x in config.meta if x.startswith("topk:"))

    return ByteLatentTransformerArgs(
        # stuff to care about
        vocab_size=BYTE_UNITS + OFFSET,  # 256 bytes + 4 special tokens
        norm_eps=config.epsilon,
        dim=config.hidden_size,
        dim_token_emb=config.embed_size,
        dim_global=config.hidden_size,
        encoder_hash_byte_group_nb_functions=1,
        encoder_hash_byte_group_size=[3, 4, 5],
        encoder_hash_byte_group_vocab=config.vocab_size,
        n_layers_local_encoder=1,
        n_layers_local_decoder=1,
        # stuff to probably ignore
        dim_local_encoder=config.hidden_size,
        dim_local_decoder=config.hidden_size,
        cross_attn_encoder=False,  # the authors found that using cross-attention in the decoder is most effective.
        cross_attn_decoder=False,
        cross_attn_window_encoder=512,
        cross_attn_window_decoder=512,
        cross_attn_k=1,  # is a multiplier for token and patch embedding
        cross_attn_nheads=1,  # num heads used in cross attn
        cross_attn_all_layers_encoder=False,
        cross_attn_all_layers_decoder=False,
        cross_attn_use_flex_attention=False,  # not supported on CPU and older GPUs
        cross_attn_init_by_pooling=True,
        use_rope=True,
        downsampling_by_pooling=downsampling_method,
        share_encoder_decoder_emb=False,
        dropout=config.dropout,
        entropy_model_checkpoint_dir=None,
        attn_impl="xformers",
        local_attention_window_len=512,  # sliding window
        n_heads=4,
        n_heads_local_encoder=4,
        n_heads_local_decoder=4,
        max_seqlen=config.max_length,
        eos_id=0,  # currently we split packed sequences with the pad_token ID
        # max_encoder_seq_length=8192,
    )


def create_local_encoder_args(args: ByteLatentTransformerArgs) -> LocalModelArgs:
    """
    Create arguments for the local encoder model.

    Args:
        args: Base ByteLatent transformer arguments

    Returns:
        LocalModelArgs configured for the encoder
    """
    return LocalModelArgs(
        # Updated args
        dim=args.dim_token_emb,
        n_layers=args.n_layers_local_encoder,
        n_heads=args.n_heads_local_encoder,
        dim_token_emb=get_encoder_dim_token_emb(args),
        dim_patch_emb=get_encoder_dim_patch_emb(args),
        cross_attn_encoder=args.cross_attn_encoder,
        cross_attn_decoder=False,
        cross_attn_k=args.cross_attn_k if args.cross_attn_encoder else None,
        cross_attn_init_by_pooling=args.cross_attn_init_by_pooling,
        # Defaults
        dropout=args.dropout,
        vocab_size=args.vocab_size + args.pm_size,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
        cross_attn_nheads=args.cross_attn_nheads,
        max_seqlen=args.max_seqlen,
        attn_impl=args.attn_impl,
        eos_id=args.eos_id,
    )


def create_local_decoder_args(args: ByteLatentTransformerArgs) -> LocalModelArgs:
    """
    Create arguments for the local decoder model.

    Args:
        args: Base ByteLatent transformer arguments

    Returns:
        LocalModelArgs configured for the decoder
    """
    return LocalModelArgs(
        # Updated args
        dim=args.dim_token_emb,
        n_layers=args.n_layers_local_decoder,
        n_heads=args.n_heads_local_decoder,
        dim_token_emb=args.dim_token_emb,
        dim_patch_emb=args.dim,
        cross_attn_encoder=False,
        cross_attn_decoder=args.cross_attn_decoder,
        cross_attn_init_by_pooling=False,
        cross_attn_k=args.cross_attn_k if args.cross_attn_decoder else None,
        # Defaults
        dropout=args.dropout,
        vocab_size=args.vocab_size + args.pm_size,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        cross_attn_nheads=args.cross_attn_nheads,
        max_seqlen=args.max_seqlen,
        attn_impl=args.attn_impl,
        eos_id=args.eos_id,
    )


def create_entropy_model_args(args: ByteLatentTransformerArgs) -> LMTransformerArgs:
    """
    Create arguments for the entropy prediction model.

    Args:
        args: Base ByteLatent transformer arguments

    Returns:
        LMTransformerArgs configured for the entropy model
    """
    return LMTransformerArgs(
        # Updated args
        dim=args.dim,
        n_layers=args.n_layers_local_decoder,
        n_heads=args.n_heads_local_decoder,
        # Defaults
        vocab_size=args.vocab_size + args.pm_size,
        sliding_window=args.local_attention_window_len,
        max_seqlen=args.max_seqlen,
        attn_impl=args.attn_impl,
        eos_id=args.eos_id,
    )
