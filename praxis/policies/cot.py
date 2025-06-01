"""
Chain of Thought (CoT) policy for training models with step-by-step reasoning.

This module implements a simple CoT training approach that:
1. Encourages generation of reasoning steps before final answers
2. Rewards correct reasoning patterns
3. Uses supervised learning with structured prompts
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.containers import LossContainer

# Import COT tags from builders
try:
    from builders import COT_TAGS
except ImportError:
    # Fallback for when builders is not in path (e.g., during testing)
    COT_TAGS = None


class ChainOfThought(nn.Module):
    """
    Chain of Thought training policy.

    This is a simplified approach that:
    - Uses supervised learning on CoT examples
    - Rewards structured reasoning (presence of thinking tags)
    - Applies higher weight to reasoning steps vs final answer

    Key features:
    - No complex RL needed initially - supervised learning works well for CoT
    - Gradually introduces rewards for reasoning structure
    - Can be extended with REINFORCE for more complex scenarios
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # Optional: Simple MLP to predict reasoning quality
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Logging
        self.step_count = 0
        self.log_interval = 100

        # CoT tracking statistics
        self.cot_stats = {
            "total_batches": 0,
            "batches_with_cot": 0,
            "total_cot_tokens": 0,
            "total_tokens": 0,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[LossContainer]]:
        """
        Forward pass for CoT training.

        For now, we primarily return the hidden states unchanged and compute
        a weighted loss that emphasizes reasoning steps. This can be extended
        with REINFORCE later if needed.

        Args:
            hidden_states: Hidden states from LM [batch_size, seq_len, hidden_size]
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Token labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_ids: Original token IDs for pattern detection [batch_size, seq_len]

        Returns:
            hidden_states: Unchanged hidden states
            losses: LossContainer containing cot_loss and metrics
        """
        # Get actual logits shape since it might be different due to shifting
        _, logits_seq_len, _ = logits.shape
        device = hidden_states.device

        # Initialize an empty loss container
        losses = LossContainer(cot_usage_loss=0, quality_loss=0)

        # Early return if no CoT examples in this batch
        if not self.training or token_weights is None:
            return hidden_states, losses

        weights = token_weights

        # Ensure weights match the logits sequence length (accounting for shift)
        if weights.shape[-1] > logits_seq_len:
            weights = weights[..., :logits_seq_len]
        elif weights.shape[-1] < logits_seq_len:
            # Pad with default weight if needed
            padding = torch.ones(
                *weights.shape[:-1],
                logits_seq_len - weights.shape[-1],
                device=device,
            )
            weights = torch.cat([weights.to(device), padding], dim=-1)

        token_weights = weights

        # Compute reasoning quality score on full sequence
        # Apply quality head to each position, then pool the quality scores
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Trim hidden states to match logits sequence length if needed
        if seq_len > logits_seq_len:
            hidden_states = hidden_states[:, :logits_seq_len, :]
            seq_len = logits_seq_len

        # Reshape to apply quality head to all positions
        flat_hidden = hidden_states.reshape(-1, hidden_size)  # [batch*seq, hidden]
        position_qualities = self.quality_head(flat_hidden)  # [batch*seq, 1]
        position_qualities = position_qualities.view(
            batch_size, seq_len
        )  # [batch, seq]

        # Pool quality scores with attention mask (for logging only)
        if attention_mask is not None:
            # Trim attention mask to match sequence length
            if attention_mask.shape[1] > seq_len:
                attention_mask = attention_mask[:, :seq_len]
            masked_qualities = position_qualities * attention_mask
            quality_scores = masked_qualities.sum(dim=1) / attention_mask.sum(
                dim=1
            ).clamp(min=1)
        else:
            quality_scores = position_qualities.mean(dim=1)

        # Update statistics
        self.cot_stats["total_batches"] += 1
        self.cot_stats["total_tokens"] += token_weights.numel()

        cot_tokens_in_batch = (token_weights != 1.0).sum().item()
        if cot_tokens_in_batch > 0:
            self.cot_stats["batches_with_cot"] += 1
            self.cot_stats["total_cot_tokens"] += cot_tokens_in_batch

        # CoT usage loss: minimize when CoT is used more (encourage CoT usage)
        total_tokens = token_weights.numel()
        cot_usage_ratio = (
            cot_tokens_in_batch / total_tokens if total_tokens > 0 else 0.0
        )
        # Convert to positive loss: high usage = low loss, low usage = high loss
        cot_usage_loss = torch.tensor(
            1.0 - cot_usage_ratio, device=device
        )  # Positive loss that decreases with more CoT usage

        # Self-modeling: quality head predicts logits confidence (inverse entropy)
        # High quality reasoning should correlate with confident predictions

        # Compute normalized entropy of logits at each position as target
        # Lower entropy = higher confidence = higher quality
        # CRITICAL: Detach to prevent gradients from flowing back to main model logits
        logits_detached = logits.detach()
        logits_flat = logits_detached.view(
            -1, logits_detached.shape[-1]
        )  # [batch*seq, vocab]
        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = F.softmax(logits_flat, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch*seq]

        # Normalize entropy to [0,1] range (0=very confident, 1=very uncertain)
        max_entropy = torch.log(
            torch.tensor(logits.shape[-1], dtype=torch.float, device=device)
        )
        normalized_entropy = entropy / max_entropy

        # Convert to confidence: 1 - normalized_entropy (1=confident, 0=uncertain)
        confidence_targets = 1.0 - normalized_entropy
        confidence_targets = confidence_targets.view(batch_size, logits_seq_len)

        # Apply attention mask to targets
        if attention_mask is not None:
            confidence_targets = confidence_targets * attention_mask

        # MSE loss between position_qualities and confidence_targets
        if attention_mask is not None:
            # Apply mask and compute MSE only on valid positions
            mask_flat = attention_mask.view(-1)
            valid_positions = mask_flat > 0
            quality_loss = F.mse_loss(
                position_qualities.view(-1)[valid_positions],
                confidence_targets.view(-1)[valid_positions],
            )
        else:
            quality_loss = F.mse_loss(position_qualities, confidence_targets)

        # Store token weight stats for consolidated logging
        self._last_token_stats = {
            "non_default_tokens": (token_weights != 1.0).sum().item(),
            "total_tokens": token_weights.numel(),
            "weight_range": (
                token_weights.min().item(),
                token_weights.max().item(),
            ),
            "mean_weight": token_weights.mean().item(),
        }

        # Store current step data for consolidated logging
        self.step_count += 1
        self._last_step_data = {
            "step": self.step_count,
            "quality_mean": quality_scores.mean().item(),
            "quality_loss": quality_loss.item(),
            "cot_usage_loss": cot_usage_loss.item(),
            "cot_usage_ratio": cot_usage_ratio,
        }

        # Consolidated logging - only log here
        if self.step_count % self.log_interval == 0:
            self._log_consolidated_stats()

        # Add both losses
        losses.add_loss(
            "cot_usage_loss", cot_usage_loss
        )  # Loss for CoT usage (positive loss, decreases with more usage)
        losses.add_loss(
            "quality_loss", quality_loss
        )  # Loss for quality improvement (positive = error)

        return hidden_states, losses

    def _log_consolidated_stats(self):
        """Consolidated, clean logging for CoT policy."""
        step_data = self._last_step_data

        # Calculate CoT usage statistics
        total_batches = self.cot_stats["total_batches"]
        cot_batches = self.cot_stats["batches_with_cot"]
        total_tokens = self.cot_stats["total_tokens"]
        cot_tokens = self.cot_stats["total_cot_tokens"]

        cot_batch_pct = (cot_batches / total_batches * 100) if total_batches > 0 else 0
        cot_token_pct = (cot_tokens / total_tokens * 100) if total_tokens > 0 else 0

        # Get token weight stats if available
        token_stats = getattr(self, "_last_token_stats", None)

        print(f"\n[CoT Policy] Step {step_data['step']}:")

        # Show both loss components
        cot_usage_loss = step_data["cot_usage_loss"]
        quality_loss = step_data["quality_loss"]
        quality = step_data["quality_mean"]
        usage_ratio = step_data["cot_usage_ratio"]

        print(
            f"  CoT usage loss: {cot_usage_loss:.4f} (usage-based, usage={usage_ratio:.3f})"
        )
        print(f"  Quality loss: {quality_loss:.4f} (quality={quality:.3f})")
        print(f"  CoT usage: {cot_batch_pct:.1f}% batches, {cot_token_pct:.1f}% tokens")

        if token_stats:
            non_default = token_stats["non_default_tokens"]
            total = token_stats["total_tokens"]
            min_w, max_w = token_stats["weight_range"]
            mean_w = token_stats["mean_weight"]
            print(
                f"  Token weights: {non_default}/{total} CoT tokens, range=[{min_w:.2f}, {max_w:.2f}], mean={mean_w:.2f}"
            )
