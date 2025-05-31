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

        # CoT-specific parameters (keeping them local as requested)
        self.structure_bonus = 0.2  # Bonus for proper tag structure

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
        # Initialize an empty loss container
        losses = LossContainer(
            cot_reward=0, reasoning_quality=0, structure_reward=0, weighting_reward=0
        )

        # Early return if no CoT examples in this batch
        if not self.training or token_weights is None:
            return hidden_states, losses

        device = hidden_states.device

        # Get actual logits shape since it might be different due to shifting
        _, logits_seq_len, _ = logits.shape

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
        self._last_token_stats = {
            "non_default_tokens": (token_weights != 1.0).sum().item(),
            "total_tokens": token_weights.numel(),
            "weight_range": (
                token_weights.min().item(),
                token_weights.max().item(),
            ),
            "mean_weight": token_weights.mean().item(),
        }

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

        # Structure reward based on quality scores (confidence in reasoning)
        structure_reward = (
            torch.tensor(self.structure_bonus, device=device) * quality_scores.mean()
        )

        # Token weighting reward - incentivize using reasoning tokens appropriately
        # Reward based on how much the model uses CoT structure
        cot_token_ratio = (token_weights != 1.0).float().mean()
        weighting_reward = torch.tensor(0.1, device=device) * cot_token_ratio

        # Total CoT reward (negative because we want to maximize rewards)
        cot_reward = -(structure_reward + weighting_reward)

        # Update statistics
        self.cot_stats["total_batches"] += 1
        self.cot_stats["total_tokens"] += token_weights.numel()

        cot_tokens_in_batch = (token_weights != 1.0).sum().item()
        if cot_tokens_in_batch > 0:
            self.cot_stats["batches_with_cot"] += 1
            self.cot_stats["total_cot_tokens"] += cot_tokens_in_batch

        # Store current step data for consolidated logging
        self.step_count += 1
        self._last_step_data = {
            "step": self.step_count,
            "structure_reward": structure_reward.item(),
            "weighting_reward": weighting_reward,
            "quality_mean": quality_scores.mean().item(),
            "cot_reward": cot_reward.item(),
        }

        # Consolidated logging - only log here
        if self.step_count % self.log_interval == 0:
            self._log_consolidated_stats()

        # Update all losses
        losses.add_loss("cot_reward", cot_reward)  # Negative value (reward)
        losses.add_loss("reasoning_quality", quality_scores.mean())
        losses.add_loss("structure_reward", structure_reward)
        losses.add_loss("weighting_reward", weighting_reward)

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

        # Show reward components clearly
        cot_reward = step_data["cot_reward"]
        structure_reward = step_data["structure_reward"]
        weighting_reward = step_data["weighting_reward"]
        quality = step_data["quality_mean"]

        print(
            f"  CoT reward: {cot_reward:.4f} (structure: +{structure_reward:.4f}, weighting: +{weighting_reward:.4f})"
        )
        print(f"  Quality: {quality:.3f}")
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
            print(
                f"  RL Rewards (positive=good): mean={rewards.mean():.3f}, range=[{rewards.min():.3f}, {rewards.max():.3f}]"
            )
