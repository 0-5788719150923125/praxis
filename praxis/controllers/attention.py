from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class AttentionController(BaseController):
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

        # Routers for final decision
        self.routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    ACT2FN["relu"],
                    nn.Linear(hidden_size * 4, self.num_experts),
                )
                for _ in range(config.depth)
            ]
        )

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
        current_state = hidden_states[:, -1]

        # Handle the case with no route history
        if not current_route:
            # Simple prediction with no history
            logits = self.routers[current_depth](current_state)
        else:
            # Create history tensor from route
            route_tensor = torch.tensor(current_route, device=device).long()

            # Get layer embeddings for previous route
            history_embeddings = self.expert_embeddings[current_depth](route_tensor)
            history_embeddings = history_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            # Reshape current state for attention
            query = current_state.unsqueeze(1)  # [batch_size, 1, hidden_size]

            # Create attention mask (causal)
            attn_mask = (
                torch.triu(torch.ones(1, len(current_route)), diagonal=1)
                .bool()
                .to(device)
            )

            # Apply attention with causality
            attn_output, _ = self.attention[current_depth - 1](
                query=query,
                key=history_embeddings,
                value=history_embeddings,
                attn_mask=attn_mask,
                is_causal=True,
            )

            # Route based on attention output
            logits = self.routers[current_depth](attn_output.squeeze(1))

        # Project to just the feature subset we'll modify
        feature_updates = self.channelers[current_depth](logits)

        # Create a broadcast-compatible version of the feature updates
        global_update = feature_updates.unsqueeze(1)  # [batch_size, 1, channel_size]

        # Create a new hidden states tensor
        new_states = hidden_states.clone()

        # Update only the subset of features in the last token (create a side channel)
        new_states[:, :, -self.channel_size :] += global_update

        # Get each example's vote for which expert to use next
        batch_votes = torch.argmax(logits, dim=1)  # [batch_size]

        # Find the most common vote (mode) across the batch
        vote_counts = torch.bincount(batch_votes, minlength=self.num_experts)
        next_expert_idx = torch.argmax(vote_counts).item()

        # No auxiliary loss
        aux_loss = torch.tensor(0.0, device=device)

        return new_states, controller_state, aux_loss, next_expert_idx
