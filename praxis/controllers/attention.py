import math
from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.containers.loss import LossContainer
from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class AttentionChanneler(BaseController):
    """
    An attention-based controller that makes routing decisions by attending to the
    history of previously-selected layers while minimally modifying hidden states.
    """

    def __init__(
        self, config: ConfigType, max_tokens: int = 1, initial_queries: int = 1
    ) -> None:
        super().__init__(config, allow_visualizer=True)

        # Configuration
        hidden_size = config.hidden_size

        # Number of features to modify (small fraction of hidden size)
        self.channel_size = min(16, hidden_size // 16)  # Just a few features

        # The maximum number of recent tokens to consider during attention
        self.max_tokens = max_tokens

        # Add learnable embeddings for initial state (when no route exists)
        self.initial_queries = nn.Parameter(
            torch.randn(1, initial_queries, hidden_size)
        )

        # Layer embeddings directly in hidden_size dimension
        self.expert_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.num_experts, hidden_size)
                for _ in range(self.num_experts)
            ]
        )

        # Attention mechanism operating in hidden_size space
        self.attention_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(self.num_experts)]
        )
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=config.num_heads,
                    batch_first=True,
                    dropout=config.dropout,
                    add_zero_attn=True,
                )
                for _ in range(self.num_experts)
            ]
        )

        # Weight expert attention importance
        self.reducer = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                ACT2FN["tanh"],
                nn.Linear(hidden_size // 2, 1),
                nn.Softmax(dim=1),
            )
            for _ in range(self.num_experts)
        )

        # Final output projection for routing decisions
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            ACT2FN[config.activation],
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size * 2, self.num_experts),
        )

        # Residual projections
        self.router_projection = nn.Linear(hidden_size, self.num_experts)
        self.logits_projection = nn.ModuleList(
            nn.Linear(self.num_experts, self.channel_size)
            for _ in range(self.num_experts)
        )

        # Project expert probs through a bottleneck
        self.embedders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.num_experts),
                    nn.Linear(self.num_experts, self.channel_size // 2),
                    ACT2FN["relu"],
                    nn.Linear(self.channel_size // 2, self.channel_size),
                )
                for _ in range(self.num_experts)
            ]
        )

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Optional[Tensor],
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, Optional[Tensor], LossContainer, Optional[int]]:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        allowed_length = min(self.max_tokens, seq_len)
        context_tokens = hidden_states[
            :, -allowed_length:
        ]  # [batch_size, seq_len, hidden_size]

        # Always start with initial queries
        initial_queries = self.initial_queries.expand(
            batch_size, -1, -1
        )  # [batch_size, num_initial_queries, hidden_size]

        # Create route tensor and get embeddings
        route_tensor = torch.tensor(current_route, device=device).long()
        history_embeds = (
            self.expert_embeddings[current_depth](route_tensor)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # If we have route history, append those embeddings to our query set
        queries = torch.cat(
            [initial_queries, history_embeds], dim=1
        )  # [batch_size, num_initial_queries + len(route), hidden_size]

        # Apply attention
        kv_norm = self.attention_norm[current_depth](context_tokens)
        output = (
            self.attention[current_depth](query=queries, key=kv_norm, value=kv_norm)[0]
            + queries  # Residual connection
        )

        # Dynamic importance weighting
        weights = self.reducer[current_depth](output)  # [batch_size, context_len, 1]
        scaled_output = output * (1.0 / math.sqrt(output.size(1)))
        reduced_output = torch.bmm(weights.transpose(1, 2), scaled_output).squeeze(
            1
        )  # [batch_size, hidden_size]

        # Predict an expert layer
        router_residual = self.router_projection(reduced_output)
        logits = (
            self.router(reduced_output) + router_residual
        )  # [batch_size, num_experts]

        # Use for feature updates
        logits_residual = self.logits_projection[current_depth](logits)
        feature_updates = self.embedders[current_depth](logits) + logits_residual

        # Create a broadcast-compatible version of the feature updates
        global_update = feature_updates.unsqueeze(1)  # [batch_size, 1, channel_size]
        global_update = F.layer_norm(
            global_update, normalized_shape=[self.channel_size]
        )

        # Update a subset of features in all tokens (create a side channel)
        new_states = hidden_states.clone()
        new_states[:, :, -self.channel_size :] *= global_update

        # Convert to probability distribution
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # Calculate batch consensus
        mean_probs = probs.mean(dim=0)  # [num_experts]

        # During inference, use the top expert
        next_expert_idx = torch.argmax(mean_probs).item()

        # Compute expert selection (one-hot)
        expert_mask = torch.zeros_like(probs)
        expert_indices = torch.argmax(probs, dim=-1)  # [batch_size]
        expert_mask.scatter_(
            -1, expert_indices.unsqueeze(-1), 1.0
        )  # [batch_size, num_experts]

        # Compute actual utilization
        expert_density = expert_mask.mean(dim=0)  # [num_experts]

        # Compute balance loss (correlation between utilization and probability)
        balance_loss = 0.01 * (mean_probs * expert_density).mean() * self.num_experts**2

        # Z-loss penalizes overconfidence
        z_loss = 0.001 * torch.logsumexp(logits, dim=-1).square().mean()

        # Create loss container with initial losses
        loss_container = LossContainer(balance=balance_loss, z_loss=z_loss)

        return new_states, controller_state, loss_container, next_expert_idx
