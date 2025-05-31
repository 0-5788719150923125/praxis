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

        # CoT-specific parameters (keeping them local as requested)
        self.reasoning_weight = 1.5  # Higher weight for reasoning tokens
        self.structure_bonus = 0.2  # Bonus for proper tag structure
        self.min_reasoning_length = 50  # Minimum tokens for reasoning

        # CoT policy is now simple - just applies pre-computed token weights from builder

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
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
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
            losses: Dict containing cot_loss and metrics
        """
        # Metadata contains pre-computed token weights from the data pipeline
        if not self.training:
            return hidden_states, None

        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Get actual logits shape since it might be different due to shifting
        logits_batch_size, logits_seq_len, vocab_size = logits.shape

        # Create token weights - use pre-computed weights if available, otherwise default
        if token_weights is not None:
            # Use pre-computed token weights from builder
            # Ensure shape matches [batch_size, seq_len] for logits
            if token_weights.dim() == 1:
                # Single sequence, broadcast to batch
                weights = token_weights.unsqueeze(0).expand(logits_batch_size, -1)
            else:
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

            # Store token weight stats for consolidated logging
            if token_weights is not None:
                self._last_token_stats = {
                    "non_default_tokens": (token_weights != 1.0).sum().item(),
                    "total_tokens": token_weights.numel(),
                    "weight_range": (
                        token_weights.min().item(),
                        token_weights.max().item(),
                    ),
                    "mean_weight": token_weights.mean().item(),
                }
        else:
            # Default to uniform weights with slight emphasis on early tokens (reasoning)
            token_weights = torch.ones(logits_batch_size, logits_seq_len, device=device)
            reasoning_end = int(logits_seq_len * 0.7)
            token_weights[:, :reasoning_end] = self.reasoning_weight

        # Compute reasoning quality score
        if attention_mask is not None:
            # Pool hidden states for quality estimation
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled_hidden = sum_hidden / count
        else:
            pooled_hidden = hidden_states.mean(dim=1)

        quality_scores = self.quality_head(pooled_hidden).squeeze(-1)

        # Compute weighted cross-entropy loss
        # The logits passed here are already shifted by the model (:-1)
        # So labels should already match the logits shape

        if attention_mask is not None:
            # Also need to trim attention mask to match shifted logits
            attention_mask_shifted = (
                attention_mask[..., :-1]
                if attention_mask.shape[-1] > logits_seq_len
                else attention_mask
            )
            token_weights = token_weights * attention_mask_shifted

        # Flatten for loss computation
        batch_size_actual, seq_len_actual, vocab_size = logits.shape

        # Debug shapes
        # print(f"[CoT Debug] logits shape: {logits.shape}")
        # print(f"[CoT Debug] labels shape: {labels.shape}")
        # print(f"[CoT Debug] token_weights shape: {token_weights.shape}")

        # Ensure labels match the shifted logits shape
        if labels.shape[-1] != seq_len_actual:
            # This shouldn't happen if everything is set up correctly
            print(
                f"[CoT Warning] Label shape mismatch: labels {labels.shape} vs logits {logits.shape}"
            )
            # Trim labels to match
            labels = labels[..., :seq_len_actual].contiguous()

        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        weights_flat = token_weights.contiguous().view(-1)

        # Filter out padding tokens (label == -100)
        valid_mask = labels_flat != -100

        if valid_mask.sum() == 0:
            # No valid tokens to compute loss on
            return hidden_states, None

        # Compute per-token loss only on valid tokens
        per_token_loss = F.cross_entropy(
            logits_flat, labels_flat, reduction="none", ignore_index=-100
        )

        # Apply weights and average
        weighted_loss = (per_token_loss * weights_flat).sum() / weights_flat[
            valid_mask
        ].sum().clamp(min=1)

        # Add structure bonus based on quality scores
        structure_loss = -self.structure_bonus * quality_scores.mean()

        # Total CoT loss
        cot_loss = weighted_loss + structure_loss

        # Update statistics
        self.cot_stats["total_batches"] += 1
        self.cot_stats["total_tokens"] += token_weights.numel()

        if token_weights is not None:
            cot_tokens_in_batch = (token_weights != 1.0).sum().item()
            if cot_tokens_in_batch > 0:
                self.cot_stats["batches_with_cot"] += 1
                self.cot_stats["total_cot_tokens"] += cot_tokens_in_batch

        # Store current step data for consolidated logging
        self.step_count += 1
        self._last_step_data = {
            "step": self.step_count,
            "weighted_loss": weighted_loss.item(),
            "structure_bonus": -structure_loss.item(),
            "quality_mean": quality_scores.mean().item(),
            "cot_loss": cot_loss.item(),
        }

        # Consolidated logging - only log here
        if self.step_count % self.log_interval == 0:
            self._log_consolidated_stats()

        losses = {
            "cot_loss": cot_loss,
            "reasoning_quality": quality_scores.mean(),
            "weighted_loss": weighted_loss,
        }

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
        
        # Be clear about what each value represents  
        cot_total = step_data['cot_loss']
        structure_bonus = step_data['structure_bonus']
        quality = step_data['quality_mean']
        
        if cot_total >= 0:
            print(f"  Total: {cot_total:.4f} (loss) | Structure reward: +{structure_bonus:.4f} | Quality: {quality:.3f}")
        else:
            print(f"  Total: {cot_total:.4f} (net reward) | Structure reward: +{structure_bonus:.4f} | Quality: {quality:.3f}")
        print(f"  CoT usage: {cot_batch_pct:.1f}% batches, {cot_token_pct:.1f}% tokens")

        if token_stats:
            non_default = token_stats["non_default_tokens"]
            total = token_stats["total_tokens"]
            min_w, max_w = token_stats["weight_range"]
            mean_w = token_stats["mean_weight"]
            print(
                f"  Token weights: {non_default}/{total} CoT tokens, range=[{min_w:.2f}, {max_w:.2f}], mean={mean_w:.2f}"
            )

        # Store reward stats for reward exploitation monitoring
        if hasattr(self, "_last_rewards"):
            rewards = self._last_rewards
            # Show actual reward values (positive = good performance)
            print(f"  RL Rewards (positive=good): mean={rewards.mean():.3f}, range=[{rewards.min():.3f}, {rewards.max():.3f}]")
