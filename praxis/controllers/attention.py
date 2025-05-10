from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class AttentionController(BaseController):
    """
    A simplified attention-based controller that makes routing decisions by attending to
    the history of previously selected layers, operating directly in hidden_size space.
    """

    def __init__(self, config: ConfigType) -> None:
        super().__init__(config, allow_visualizer=True)

        # Configuration
        hidden_size = config.hidden_size

        # Layer embeddings directly in hidden_size dimension
        self.expert_embeddings = nn.ModuleList(
            [nn.Embedding(self.num_experts, hidden_size) for _ in range(config.depth)]
        )

        # Attention mechanism operating in hidden_size space
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
            dropout=config.dropout,
        )

        # Router for final decision
        self.router = nn.Linear(hidden_size, self.num_experts)

        # Project expert preds back to original hidden size
        self.projector = nn.Linear(self.num_experts, hidden_size)

        # Layer that combines original hidden state with routing information
        self.blenders = nn.ModuleList(
            [nn.Linear(hidden_size * 2, hidden_size) for _ in range(config.depth)]
        )

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Optional[Tensor],
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, List[int], Optional[int]]:
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Get the last token state directly
        current_state = hidden_states[:, -1]

        # Handle the case with no route history
        if not current_route:
            # Simple prediction with no history
            logits = self.router(current_state)
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
            attn_output, _ = self.attention(
                query=query,
                key=history_embeddings,
                value=history_embeddings,
                attn_mask=attn_mask,
                is_causal=True,
            )

            # Route based on attention output
            logits = self.router(attn_output.squeeze(1))

        # Get state-mixing weights
        probs = F.softmax(logits, dim=-1)
        state_update = self.projector(probs)

        # Update only the last token with routing information
        # Use a gated combination to control information flow
        combined = torch.cat([hidden_states[:, -1], state_update], dim=-1)
        updated_last_token = self.blenders[current_depth](combined)

        # Create a copy of hidden_states to modify
        new_states = hidden_states.clone()
        new_states[:, -1] = updated_last_token

        # Get each example's vote for which expert to use next
        batch_votes = torch.argmax(logits, dim=1)  # [batch_size]

        # Find the most common vote (mode) across the batch
        vote_counts = torch.bincount(batch_votes, minlength=self.num_experts)
        next_expert_idx = torch.argmax(vote_counts).item()

        return new_states, controller_state, 0, next_expert_idx
