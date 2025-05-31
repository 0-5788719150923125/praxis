"""
Reinforcement Learning components for chain-of-thought operations.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class REINFORCE(nn.Module):
    """
    Reinforcement Learning policy module for chain-of-thought reasoning.

    This module learns to generate better reasoning traces by using rewards
    from the INTELLECT-2-RL dataset's solve_rate scores.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.rl_weight = getattr(config, "rl_weight", 0.1)

        # Value head for estimating expected rewards
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

        # Policy improvement network
        self.policy_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Baseline for variance reduction
        self.baseline = nn.Parameter(torch.tensor(0.5))

        # Logging
        self.step_count = 0
        self.log_interval = 100

    def forward(
        self,
        hidden_states: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the RL policy.

        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            rewards: Reward signals (solve_rate scores) [batch_size]
            mask: Attention mask [batch_size, seq_len]

        Returns:
            policy_hidden_states: Modified hidden states with policy improvements
            rl_loss: Reinforcement learning loss (if rewards provided)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Apply policy improvement
        policy_hidden_states = hidden_states + self.policy_mlp(hidden_states)

        rl_loss = None
        if rewards is not None and self.training:
            # Pool hidden states for value estimation
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled_hidden = sum_hidden / count
            else:
                # Simple mean pooling
                pooled_hidden = hidden_states.mean(dim=1)

            # Estimate values
            values = self.value_head(pooled_hidden).squeeze(-1)

            # Compute advantages
            advantages = rewards - self.baseline

            # The key insight: we need to compute a pseudo log-probability from the hidden states
            # This represents how much the policy network is "committing" to its modifications
            policy_logits = (
                (policy_hidden_states - hidden_states).pow(2).sum(dim=-1).mean(dim=1)
            )
            policy_log_probs = (
                -policy_logits
            )  # Negative squared distance as log prob proxy

            # Policy gradient loss (REINFORCE): maximize log_prob * advantage
            # We minimize the negative to effectively maximize
            policy_loss = -(policy_log_probs * advantages.detach()).mean()

            # Value loss (MSE between predicted and actual rewards)
            value_loss = F.mse_loss(values, rewards)

            # Combined RL loss with weight applied
            rl_loss = self.rl_weight * (policy_loss + 0.5 * value_loss)

            # Update baseline with exponential moving average
            with torch.no_grad():
                self.baseline.data = 0.99 * self.baseline.data + 0.01 * rewards.mean()

            # Store rewards for consolidated logging (done by CoT policy)
            # No separate logging here - too verbose
            self.step_count += 1

        return policy_hidden_states, rl_loss
