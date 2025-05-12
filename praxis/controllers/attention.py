from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class AttentionChanneler(BaseController):
    """
    An attention-based controller that makes routing decisions by attending to the
    history of previously-selected layers while minimally modifying hidden states.
    """

    def __init__(self, config: ConfigType) -> None:
        super().__init__(config, allow_visualizer=True)

        # Configuration
        hidden_size = config.hidden_size

        # Number of features to modify (small fraction of hidden size)
        self.channel_size = min(16, hidden_size // 16)  # Just a few features

        # Layer embeddings directly in hidden_size dimension
        self.expert_embeddings = nn.ModuleList(
            [nn.Embedding(self.num_experts, hidden_size) for _ in range(config.depth)]
        )

        # Attention mechanism operating in hidden_size space
        self.attention_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(config.depth - 1)]
        )
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=config.num_heads,
                    batch_first=True,
                    dropout=config.dropout,
                )
                for _ in range(config.depth - 1)
            ]
        )

        # Router bottlenecks with residual connections
        self.bottleneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 4),
                    ACT2FN["gelu"],
                    nn.Dropout(config.dropout),
                    nn.Linear(hidden_size // 4, hidden_size),
                )
                for _ in range(config.depth)
            ]
        )

        # Final output projection for routing decisions
        self.router = nn.Linear(hidden_size, self.num_experts)

        # Logits residual connection
        self.logits_projection = nn.ModuleList(
            [
                nn.Linear(self.num_experts, self.channel_size)
                for _ in range(config.depth)
            ]
        )

        # Project expert probs to just the subset of features we'll modify
        bottleneck_size = self.channel_size // 2
        self.embedders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.num_experts),
                    nn.Linear(self.num_experts, bottleneck_size),
                    nn.Tanh(),
                    nn.Linear(bottleneck_size, self.channel_size),
                )
                for _ in range(config.depth)
            ]
        )

        # Initialize expert usage tracking per depth
        self.register_buffer(
            "expert_utilization", torch.ones(config.depth, self.num_experts) * 1e-5
        )

        # Hyperparameters for balancing
        self.balance_coefficient = 0.01
        self.ema_factor = 0.99  # For exponential moving average of usage statistics

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Optional[Tensor],
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[int]]:
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Get the last token state directly
        context_token = hidden_states[:, -1]

        # Handle the case with no route history
        if current_route:
            # Create history tensor from route
            route_tensor = torch.tensor(current_route, device=device).long()

            # Get layer embeddings for previous route
            history_embeds = (
                self.expert_embeddings[current_depth](route_tensor)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

            # Create attention mask (causal)
            attn_mask = (
                torch.triu(torch.ones(1, len(current_route)), diagonal=1)
                .bool()
                .to(device)
            )

            # Reshape current state for attention
            query = context_token.unsqueeze(1)  # [batch_size, 1, hidden_size]
            query_norm = self.attention_norm[current_depth - 1](query)

            # Apply attention with causality
            output, _ = self.attention[current_depth - 1](
                query=query_norm,
                key=history_embeds,
                value=history_embeds,
                attn_mask=attn_mask,
                is_causal=True,
            )

            # Apply residual and reshape
            context_token = (output + query).squeeze(1)  # [batch_size, hidden_size]

        # Predict an expert layer
        logits = self.router(context_token)  # [batch_size, num_experts]

        if self.training:
            # Sample from Gumbel distribution for each batch element
            probs = F.gumbel_softmax(logits, tau=0.75, hard=True)
        else:
            # During inference, use standard softmax for determinism
            probs = F.softmax(logits, dim=-1)

        # Calculate batch consensus
        mean_probs = probs.mean(dim=0)  # [num_experts]

        # Scale by batch consensus
        scaled_probs = probs * mean_probs.unsqueeze(0)  # Element-wise multiplication

        # Renormalize to maintain probability distribution
        scaled_probs = F.normalize(scaled_probs, p=1, dim=1)  # Row-wise normalization

        # Project logits for residual connection
        logits_residual = self.logits_projection[current_depth](logits)

        # Use for feature updates
        feature_updates = self.embedders[current_depth](scaled_probs) + logits_residual

        # Create a broadcast-compatible version of the feature updates
        global_update = feature_updates.unsqueeze(1)  # [batch_size, 1, channel_size]

        # Create a new hidden states tensor
        new_states = hidden_states.clone()

        # Update a subset of features in all tokens (create a side channel)
        new_states[:, :, -self.channel_size :] += global_update

        # Select the top expert index
        next_expert_idx = torch.argmax(mean_probs).item()

        # Auxiliary loss
        aux_loss = self.load_balancing_loss(mean_probs, current_depth, next_expert_idx)

        return new_states, controller_state, aux_loss, next_expert_idx

    def load_balancing_loss(self, mean_probs, current_depth, next_expert_idx):
        device = mean_probs.device

        # Track expert usage
        expert_one_hot = F.one_hot(
            torch.tensor(next_expert_idx, device=device),
            num_classes=self.num_experts,
        ).float()

        # Update usage statistics without affecting gradients
        self.expert_utilization[current_depth] = (
            self.ema_factor * self.expert_utilization[current_depth]
            + (1 - self.ema_factor) * expert_one_hot
        )

        # Only calculate loss during training
        if not self.training:
            return torch.tensor(0.0, device=device)

        # Get current usage distribution
        usage_distribution = F.normalize(
            self.expert_utilization[current_depth], p=1, dim=0
        )

        # Create target that's between uniform and current distribution
        # α controls specialization (higher == exploration, lower == exploitation)
        alpha = 0.35
        uniform_target = torch.ones_like(mean_probs) / self.num_experts
        biased_target = alpha * uniform_target + (1 - alpha) * usage_distribution

        # Apply minimum usage threshold to prevent complete abandonment
        min_usage = 0.01 / self.num_experts
        biased_target = torch.max(biased_target, torch.tensor(min_usage, device=device))
        biased_target = F.normalize(biased_target, p=1, dim=0)  # Renormalize

        # KL divergence to this biased target instead of uniform
        balance_loss = F.kl_div(
            (mean_probs + 1e-10).log(),
            biased_target,
            reduction="sum",
        )

        return balance_loss * self.balance_coefficient
