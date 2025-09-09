"""Data interleaving and management for multi-dataset training."""

import hashlib
import random
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import PreTrainedTokenizer

from praxis.data.formatters import COT_TAGS, _rl_logger


class InterleaveDataManager:
    """Manages interleaving of multiple datasets with dynamic weighting."""
    
    # Dynamic weighting control (hardcoded switch)
    use_dynamic_weights = True  # Set to False to use static weights
    ema_alpha = 0.1  # EMA smoothing factor (lower for more conservative updates)

    # Class variable to store shared weights across all instances
    # This is needed because DataLoader workers create separate instances
    shared_weights = None
    shared_weights_initialized = False

    def __init__(
        self,
        samplers,
        weights,
        tokenizer,
        block_size,
        text_cache_size=100_000,
        rl_type=None,
    ):
        self.samplers = samplers
        self.static_weights = weights.copy()  # Store original weights
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_cache_size = text_cache_size
        self.rl_type = rl_type
        self.token_stream = torch.tensor(
            [], dtype=torch.long
        )  # Single continuous stream
        # Track sequences and their boundaries
        self.sequence_boundaries = (
            []
        )  # List of (start_idx, end_idx, reward, metadata) tuples
        self.current_stream_offset = 0  # Track position in token stream

        # Dynamic weighting metrics
        if self.use_dynamic_weights:
            self.sampling_count = 0  # Total number of samplings
            self.sampler_metrics = {}
            for i, sampler in enumerate(self.samplers):
                dataset_name = getattr(sampler, "dataset_path", f"sampler_{i}")
                self.sampler_metrics[i] = {
                    "name": dataset_name,
                    "avg_doc_length": None,  # Will be initialized on first sample
                    "total_samples": 0,  # Total times sampled
                    "total_tokens": 0,  # Total tokens consumed
                }
            # Initialize dynamic weights for this instance
            # Each dataset (train/val) maintains its own weights based on its sampler count
            self.dynamic_weights = self.static_weights.copy()

            # Only share weights between workers of the same dataset type (train OR val)
            # Check if this is training by looking at the number of samplers
            num_samplers = len(self.samplers)
            if (
                InterleaveDataManager.shared_weights_initialized
                and InterleaveDataManager.shared_weights is not None
                and len(InterleaveDataManager.shared_weights) == num_samplers
            ):
                # Use shared weights only if they match our sampler count
                self.dynamic_weights = InterleaveDataManager.shared_weights.copy()
            elif not InterleaveDataManager.shared_weights_initialized:
                # First instance - initialize shared weights for training
                InterleaveDataManager.shared_weights = self.dynamic_weights.copy()
                InterleaveDataManager.shared_weights_initialized = True

            # Always use dynamic weights when enabled
            self.weights = self.dynamic_weights
        else:
            # Static weights mode
            self.weights = weights

    def get_batch(
        self,
        batch_size: int,
        oversample: bool = False,
        supersample: bool = False,
        hypersample: bool = False,
    ) -> Dict[str, Any]:
        sequence_length = self.block_size
        current_batch_size = batch_size

        # Update weights if using dynamic weighting and they match our sampler count
        if self.use_dynamic_weights:
            # Only use shared weights if they match our sampler count
            if InterleaveDataManager.shared_weights is not None and len(
                InterleaveDataManager.shared_weights
            ) == len(self.samplers):
                self.weights = InterleaveDataManager.shared_weights
            else:
                # Use our own dynamic weights (validation or mismatched sampler count)
                self.weights = self.dynamic_weights
        # Check if batch size supports the requested sampling mode
        if hypersample and batch_size >= 64:
            current_batch_size = batch_size // 64
            sequence_length = self.block_size * 8
        elif supersample and batch_size >= 16:
            current_batch_size = batch_size // 16
            sequence_length = self.block_size * 4
        elif oversample and batch_size >= 4:
            current_batch_size = batch_size // 4
            sequence_length = self.block_size * 2

        # Calculate how many total tokens we need
        tokens_needed = current_batch_size * sequence_length

        # Make sure we have enough tokens
        while len(self.token_stream) < tokens_needed:
            self._extend_token_stream()

        # Extract batch
        batch = []
        rewards = [] if self.rl_type else None
        metadata = [] if self.rl_type else None
        token_weights = [] if self.rl_type and self.rl_type == "cot" else None

        for i in range(current_batch_size):
            start = i * sequence_length
            end = start + sequence_length
            batch.append(self.token_stream[start:end])

            # Find the reward and metadata for this sequence chunk if RL is enabled
            if self.rl_type:
                sequence_reward, sequence_metadata = (
                    self._get_reward_and_metadata_for_range(start, end)
                )
                rewards.append(sequence_reward)
                if metadata is not None:
                    metadata.append(sequence_metadata)

                # Extract token weights for CoT
                if (
                    token_weights is not None
                    and sequence_metadata.get("token_weights") is not None
                ):
                    weights = sequence_metadata["token_weights"]
                    # Ensure it's a tensor and matches the sequence length
                    if isinstance(weights, torch.Tensor):
                        original_length = weights.shape[0]
                        if weights.shape[0] > sequence_length:
                            weights = weights[:sequence_length]
                            print(
                                f"[Builder] Truncating token weights from {original_length} to {sequence_length}"
                            )
                        elif weights.shape[0] < sequence_length:
                            padding = torch.ones(sequence_length - weights.shape[0])
                            weights = torch.cat([weights, padding])
                            print(
                                f"[Builder] Padding token weights from {original_length} to {sequence_length}"
                            )

                        # Log when we're adding CoT weights to a batch
                        non_default = (weights != 1.0).sum().item()
                        if non_default > 0:
                            print(
                                f"[Builder] Adding CoT weights to batch: {non_default}/{len(weights)} non-default tokens"
                            )

                        token_weights.append(weights)
                    else:
                        # Default weights if no token weights available
                        token_weights.append(torch.ones(sequence_length))
                elif token_weights is not None:
                    # Default weights for non-CoT sequences
                    token_weights.append(torch.ones(sequence_length))

        # Remove used tokens from the stream and update boundaries
        self.token_stream = self.token_stream[tokens_needed:]
        self._update_boundaries_after_removal(tokens_needed)

        # Include current weights in the return value for logging
        result = {
            "batch": batch,
            "rewards": rewards,
            "metadata": metadata,
            "token_weights": token_weights,
        }

        # Add the current sampler weights if dynamic weighting is enabled
        if self.use_dynamic_weights:
            result["sampler_weights"] = (
                self.weights.copy() if hasattr(self, "weights") else None
            )

        return result

    def _get_reward_and_metadata_for_range(
        self, start: int, end: int
    ) -> Tuple[float, Dict[str, Any]]:
        """Get the reward and metadata for a token range, handling sequence boundaries properly.

        This implements sequence-level reward assignment where:
        - If a chunk is entirely within one sequence, it gets that sequence's full reward and metadata
        - If a chunk spans multiple sequences, it gets a weighted average reward and metadata from the dominant sequence
        - This ensures that long sequences split across batches are rewarded consistently

        Args:
            start: Start index in current token stream
            end: End index in current token stream

        Returns:
            Tuple of (reward value, metadata dict) for this range
        """
        # Adjust indices to account for stream offset
        abs_start = self.current_stream_offset + start
        abs_end = self.current_stream_offset + end

        # Find all sequences that overlap with this range
        overlapping_data = []  # List of (reward, metadata, weight) tuples

        for seq_start, seq_end, reward, metadata in self.sequence_boundaries:
            # Calculate overlap
            overlap_start = max(abs_start, seq_start)
            overlap_end = min(abs_end, seq_end)

            if overlap_start < overlap_end:
                # This sequence overlaps with our range
                overlap_size = overlap_end - overlap_start
                overlapping_data.append((reward, metadata, overlap_size))

        # If we have overlapping sequences, return weighted average reward and dominant metadata
        if overlapping_data:
            # COMMON CASE: If a single sequence fully contains this chunk,
            # give it the full reward and metadata (most chunks will be fully within one sequence)
            for seq_start, seq_end, reward, metadata in self.sequence_boundaries:
                if seq_start <= abs_start and seq_end >= abs_end:
                    return reward, metadata

            # EDGE CASE: Chunk spans multiple sequences (rare)
            # Use weighted average for reward, metadata from dominant sequence
            total_weight = sum(weight for _, _, weight in overlapping_data)
            if total_weight > 0:
                # Weighted average reward
                weighted_reward = (
                    sum(r * w for r, _, w in overlapping_data) / total_weight
                )
                # Metadata from the sequence with most overlap
                dominant_metadata = max(overlapping_data, key=lambda x: x[2])[1]
                return weighted_reward, dominant_metadata

        # No overlap found, return defaults
        return 0.0, {}

    def _update_dynamic_weights_after_sample(self, sampler_idx: int, doc_length: int):
        """Update metrics and weights with EMA after each sample."""
        if not self.use_dynamic_weights:
            return

        metrics = self.sampler_metrics[sampler_idx]

        # Update total counts
        metrics["total_samples"] += 1
        metrics["total_tokens"] += doc_length

        # Update average document length with EMA
        if metrics["avg_doc_length"] is None:
            metrics["avg_doc_length"] = float(doc_length)
        else:
            metrics["avg_doc_length"] = (
                self.ema_alpha * doc_length
                + (1 - self.ema_alpha) * metrics["avg_doc_length"]
            )

        # Calculate target weights based on current metrics
        target_weights = self._calculate_target_weights()

        # Update dynamic weights with EMA towards target
        old_weights = self.dynamic_weights.copy()
        for i in range(len(self.dynamic_weights)):
            self.dynamic_weights[i] = (
                self.ema_alpha * target_weights[i]
                + (1 - self.ema_alpha) * self.dynamic_weights[i]
            )

        # Normalize to ensure weights sum to 1
        total = sum(self.dynamic_weights)
        if total > 0:
            self.dynamic_weights = [w / total for w in self.dynamic_weights]

        # Update the shared class variable only if we're the training dataset
        # (validation datasets maintain their own weights)
        if (
            len(self.samplers) == len(InterleaveDataManager.shared_weights)
            if InterleaveDataManager.shared_weights
            else True
        ):
            InterleaveDataManager.shared_weights = self.dynamic_weights.copy()

    def _calculate_target_weights(self):
        """Calculate target weights based on current metrics."""
        if not self.sampler_metrics:
            return self.static_weights

        # Skip if we don't have enough data yet
        if all(m["avg_doc_length"] is None for m in self.sampler_metrics.values()):
            return self.static_weights

        target_weights = []

        # Calculate average document length across all samplers
        avg_length = sum(
            m["avg_doc_length"]
            for m in self.sampler_metrics.values()
            if m["avg_doc_length"] is not None
        ) / len(self.sampler_metrics)

        # Calculate target based on balancing token consumption
        total_tokens = sum(m["total_tokens"] for m in self.sampler_metrics.values())
        avg_tokens_per_sampler = (
            total_tokens / len(self.sampler_metrics) if total_tokens > 0 else 1
        )

        for i in range(len(self.samplers)):
            metrics = self.sampler_metrics[i]

            # Start with static weight
            weight = self.static_weights[i]

            if metrics["avg_doc_length"] is not None and metrics["total_samples"] > 0:
                # Factor 1: Inverse document length (shorter docs get higher weight)
                length_factor = avg_length / max(metrics["avg_doc_length"], 1.0)

                # Factor 2: Balance token consumption (underrepresented gets boost)
                if metrics["total_tokens"] > 0:
                    token_balance_factor = avg_tokens_per_sampler / max(
                        metrics["total_tokens"], 1.0
                    )
                else:
                    token_balance_factor = 2.0  # Strong boost for never sampled

                # Combine factors - geometric mean for balance
                weight = weight * (length_factor * token_balance_factor) ** 0.5

            target_weights.append(weight)

        # Normalize weights to sum to 1
        total = sum(target_weights)
        if total > 0:
            return [w / total for w in target_weights]
        else:
            return self.static_weights

    def _update_boundaries_after_removal(self, tokens_removed: int):
        """Update sequence boundaries after removing tokens from the stream."""
        self.current_stream_offset += tokens_removed

        # Remove boundaries that are now completely before the current stream
        self.sequence_boundaries = [
            (start, end, reward, metadata)
            for start, end, reward, metadata in self.sequence_boundaries
            if end > self.current_stream_offset
        ]

    def _extend_token_stream(self):
        """Add more tokens to our stream when needed, tracking sequence boundaries."""
        sequences_to_add = []
        total_text = ""

        # Collect sequences until we have enough text
        while len(total_text) < self.text_cache_size:
            # Pick a sampler based on current weights
            sampler_idx = random.choices(
                range(len(self.samplers)), weights=self.weights, k=1
            )[0]
            sampler = self.samplers[sampler_idx]
            # Get a sequence from that sampler
            new_sequences = sampler.get_sequences(1)
            text = new_sequences[0]

            # Update dynamic weights after each sample
            if self.use_dynamic_weights:
                self.sampling_count += 1
                doc_length = len(text)
                self._update_dynamic_weights_after_sample(sampler_idx, doc_length)

            # Track dataset sampling
            dataset_name = getattr(sampler, "dataset_path", "unknown")

            # Get reward and metadata for this sequence if applicable
            reward = 0.0
            metadata = {}
            has_reward = False

            if self.rl_type and hasattr(sampler, "reward_cache"):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                cache_data = sampler.reward_cache.get(text_hash, None)

                if cache_data is None:
                    reward = 0.0
                    metadata = {}
                elif isinstance(cache_data, dict):
                    reward = cache_data.get("reward", 0.0)
                    metadata = cache_data  # Store the full metadata
                else:
                    # Legacy format
                    reward = cache_data if isinstance(cache_data, (int, float)) else 0.0
                    metadata = {"reward": reward}

                # Only log interesting reward events
                if reward == -1:
                    print(f"[RL] Found generation sequence from {dataset_name}")
                elif reward > 0:
                    print(f"[RL] Found static reward {reward} from {dataset_name}")

                has_reward = (
                    reward != 0
                )  # Any non-zero reward (including -1 generation flag)

                if has_reward and reward != -1:
                    _rl_logger.log_reward_found(
                        reward, dataset_name
                    )  # Only log static rewards

            _rl_logger.log_dataset_sample(dataset_name, has_reward)

            # Add separator
            text_with_sep = text.rstrip() + "\n"
            sequences_to_add.append(
                (
                    len(total_text),
                    len(total_text) + len(text_with_sep),
                    reward,
                    metadata,
                )
            )
            total_text += text_with_sep

        # Tokenize the entire text at once
        tokens = self.tokenizer(
            text=total_text,
            padding=False,
            return_tensors="pt",
        )[
            "input_ids"
        ].squeeze(0)

        # Convert character positions to token positions
        # This is approximate but should work well enough
        chars_per_token = len(total_text) / len(tokens) if len(tokens) > 0 else 1.0
        current_pos = self.current_stream_offset + len(self.token_stream)

        for char_start, char_end, reward, metadata in sequences_to_add:
            # Estimate token positions based on character positions
            token_start = current_pos + int(char_start / chars_per_token)
            token_end = current_pos + int(char_end / chars_per_token)

            # Ensure we have at least one token per sequence
            if token_end <= token_start:
                token_end = token_start + 1

            # For CoT sequences, compute token-level rewards/weights
            if metadata.get("cot_metadata") is not None:
                # Extract the sequence text
                sequence_text = total_text[char_start:char_end]
                # Compute token-level weights for this sequence
                token_weights = self._compute_cot_token_weights(
                    sequence_text,
                    token_start - current_pos,  # Local start position
                    token_end - current_pos,  # Local end position
                    tokens,
                    metadata["cot_metadata"],
                )
                metadata["token_weights"] = token_weights

                # Validation logging
                if metadata["cot_metadata"].get("has_cot", False):
                    non_default = (token_weights != 1.0).sum().item()
                    print(f"[Builder] CoT sequence detected:")
                    print(f"  Text length: {len(sequence_text)} chars")
                    print(
                        f"  Token range: [{token_start - current_pos}, {token_end - current_pos})"
                    )
                    print(f"  Tags present: {metadata['cot_metadata']['tags_present']}")
                    print(f"  Non-default weights: {non_default}/{len(token_weights)}")
                    print(
                        f"  Weight range: [{token_weights.min():.3f}, {token_weights.max():.3f}]"
                    )

                    # Check for potential splitting issues
                    thinking_start = sequence_text.find("<thinking>")
                    thinking_end = sequence_text.find("</thinking>")
                    if thinking_start != -1 and thinking_end == -1:
                        print(
                            f"  ⚠️  WARNING: <thinking> tag opened but not closed - sequence may be split"
                        )
                    elif thinking_start == -1 and thinking_end != -1:
                        print(
                            f"  ⚠️  WARNING: </thinking> tag found but no opening - sequence may be split"
                        )

                    # Show first few chars for context
                    preview = sequence_text[:100].replace("\n", "\\n")
                    print(f"  Preview: {preview}...")

            # Store the boundary information with metadata
            self.sequence_boundaries.append((token_start, token_end, reward, metadata))

        # Add tokens to stream
        self.token_stream = torch.cat([self.token_stream, tokens])

    def _compute_cot_token_weights(
        self, text, local_start, local_end, tokens, cot_metadata
    ):
        """
        Compute token-level weights for CoT sequences based on tag positions.

        Returns a tensor of weights for each token in the sequence.
        """
        seq_length = local_end - local_start
        token_weights = torch.ones(seq_length)

        # If no CoT tags present, return default weights
        if not cot_metadata.get("has_cot", False):
            return token_weights

        # Extract just the tokens for this sequence
        seq_tokens = tokens[local_start:local_end]

        # Decode tokens to get exact character positions
        # We need to map tag regions to token positions
        char_to_token = {}
        current_char = 0

        for i, token_id in enumerate(seq_tokens):
            # Decode single token to get its text
            token_text = self.tokenizer.decode(
                [token_id.item()], skip_special_tokens=False
            )
            char_to_token[current_char] = i
            current_char += len(token_text)

        # Find tag regions in the text and map to tokens
        regions_found = []
        incomplete_tags = []

        for tag_type in ["wrapper", "thinking_components"]:
            for tag_name, (open_tag, close_tag) in COT_TAGS[tag_type].items():
                if tag_name in cot_metadata.get("tags_present", []):
                    weight = COT_TAGS["tag_weights"].get(tag_name, 1.0)

                    # Find all occurrences of this tag pair
                    start_pos = 0
                    while True:
                        open_pos = text.find(open_tag, start_pos)
                        if open_pos == -1:
                            break
                        close_pos = text.find(close_tag, open_pos + len(open_tag))

                        if close_pos == -1:
                            # Handle incomplete tag (split sequence) - apply weight from open tag to end
                            print(
                                f"[Builder] Incomplete {tag_name} tag detected - applying weight to end of sequence"
                            )
                            token_start = 0
                            for char_pos, token_pos in sorted(char_to_token.items()):
                                if char_pos <= open_pos:
                                    token_start = token_pos
                            token_weights[token_start:] = weight
                            incomplete_tags.append(tag_name)
                            break
                        else:
                            # Complete tag pair found
                            # Map character positions to token positions
                            token_start = 0
                            token_end = seq_length - 1

                            for char_pos, token_pos in sorted(char_to_token.items()):
                                if char_pos <= open_pos:
                                    token_start = token_pos
                                if char_pos <= close_pos + len(close_tag):
                                    token_end = min(token_pos + 1, seq_length)

                            # Apply weight to tokens in this region
                            token_weights[token_start:token_end] = weight
                            regions_found.append((tag_name, token_start, token_end))

                        start_pos = (
                            close_pos + len(close_tag) if close_pos != -1 else len(text)
                        )

                # Also check for orphaned closing tags (from split sequences)
                close_pos = text.find(close_tag)
                if close_pos != -1 and text.find(open_tag) == -1:
                    print(
                        f"[Builder] Orphaned closing {tag_name} tag detected - applying weight from start"
                    )
                    token_end = seq_length - 1
                    for char_pos, token_pos in sorted(char_to_token.items()):
                        if char_pos <= close_pos + len(close_tag):
                            token_end = min(token_pos + 1, seq_length)
                    token_weights[:token_end] = weight
                    incomplete_tags.append(f"{tag_name}_orphaned")

        # Log findings
        if regions_found or incomplete_tags:
            print(f"[Builder] Token weight mapping for sequence:")
            for tag_name, start, end in regions_found:
                print(
                    f"  Complete {tag_name}: tokens [{start}:{end}] = {COT_TAGS['tag_weights'][tag_name]}"
                )
            for tag in incomplete_tags:
                print(f"  Incomplete/orphaned: {tag}")

        return token_weights