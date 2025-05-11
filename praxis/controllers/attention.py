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
        self.channel_size = min(32, hidden_size // 16)  # Just a few features

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

        # Router MLPs with residual connections
        self.feedforward = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size * 2),
                    ACT2FN["leaky_relu"],
                    nn.Dropout(config.dropout),
                    nn.Linear(hidden_size * 2, hidden_size),
                )
                for _ in range(config.depth)
            ]
        )

        # Final output projection for routing decisions
        self.router = nn.Linear(hidden_size, self.num_experts)

        # Project expert probs to just the subset of features we'll modify
        self.channelers = nn.ModuleList(
            [
                nn.Linear(self.num_experts, self.channel_size)
                for _ in range(config.depth)
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
            history_embeddings = self.expert_embeddings[current_depth](route_tensor)
            history_embeddings = history_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
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
                key=history_embeddings,
                value=history_embeddings,
                attn_mask=attn_mask,
                is_causal=True,
            )

            # Apply residual and reshape
            context_token = (output + query).squeeze(1)  # [batch_size, hidden_size]

        # Apply router with residual
        router_hidden = self.feedforward[current_depth](context_token)
        router_hidden += context_token

        # Predict an expert layer
        logits = self.router(router_hidden)  # [batch_size, num_experts]

        # Project to just the feature subset we'll modify
        feature_updates = self.channelers[current_depth](logits)

        # # Create a new hidden states tensor
        # new_states = hidden_states.clone()

        # # Update a subset of features in the final token (create a side channel)
        # new_states[:, -1, -self.channel_size :] += feature_updates

        # Create a broadcast-compatible version of the feature updates
        global_update = feature_updates.unsqueeze(1)  # [batch_size, 1, channel_size]

        # Create a new hidden states tensor
        new_states = hidden_states.clone()

        # Update a subset of features in all tokens (create a side channel)
        new_states[:, :, -self.channel_size :] += global_update

        # Get each example's vote for which expert to use next
        batch_votes = torch.argmax(logits, dim=1)  # [batch_size]

        # Find the most common vote (mode) across the batch
        vote_counts = torch.bincount(batch_votes)
        next_expert_idx = torch.argmax(vote_counts).item()

        # No auxiliary loss
        aux_loss = torch.tensor(0.0, device=device)

        return new_states, controller_state, aux_loss, next_expert_idx
