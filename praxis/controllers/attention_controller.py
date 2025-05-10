from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class AttentionController(BaseController):
    """
    An attention-based controller that makes routing decisions by attending to
    the history of previously selected layers.
    """

    def __init__(self, config: ConfigType, allow_early_exits: bool = False) -> None:
        super().__init__(config, allow_visualizer=True)

        # Configuration
        self.hidden_size = config.hidden_size
        embed_dim = min(256, self.hidden_size // 2)

        # Projection from hidden states to embedding dimension
        self.projector = nn.Linear(self.hidden_size, embed_dim)

        # Layer embeddings - unique representation for each layer
        self.embedder = nn.Parameter(torch.randn(self.num_experts, embed_dim))

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=config.dropout,
        )

        # Router for final decision
        self.router = nn.Linear(embed_dim, self.num_experts)

        # Linear transformation to map attention output back to hidden dimension
        self.updater = nn.Linear(embed_dim, self.hidden_size)

        # Layer that combines original hidden state with routing information
        self.combiner = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Optional[Tensor],
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, List[int], Optional[int]]:
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Project current hidden state
        current_state = self.projector(hidden_states[:, -1])

        # Handle the case with no route history
        if not current_route:
            # Simple prediction with no history
            attn_output = current_state
            logits = self.router(attn_output)
        else:
            # Create history tensor from route
            history_layers = torch.tensor(current_route, device=device).long()
            history_embeddings = F.embedding(history_layers, self.embedder)
            history_embeddings = history_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            # Reshape current state for attention
            query = current_state.unsqueeze(1)  # [batch_size, 1, emb_dim]

            # Create attention mask (causal)
            mask_size = len(current_route)
            attn_mask = (
                torch.triu(torch.ones(1, mask_size), diagonal=1).bool().to(device)
            )

            # Apply attention with causality
            attn_output, _ = self.attention(
                query=query,
                key=history_embeddings,
                value=history_embeddings,
                attn_mask=attn_mask,
                is_causal=True,
            )
            attn_output = attn_output.squeeze(1)

            # Route based on attention output
            logits = self.router(attn_output)

        # Create differentiable connection by updating hidden states
        # This ensures gradient flow from routing decisions back to model parameters
        routing_hidden = self.updater(attn_output)

        # Create a copy of hidden_states to modify
        new_states = hidden_states.clone()

        # Update only the last token with routing information
        # Use a gated combination to control information flow
        combined = torch.cat([hidden_states[:, -1], routing_hidden], dim=-1)
        updated_last_token = self.combiner(combined)
        new_states[:, -1] = updated_last_token

        # Get each example's vote for which expert to use next
        batch_votes = torch.argmax(logits, dim=1)  # [batch_size]

        # Find the most common vote (mode) across the batch
        vote_counts = torch.bincount(batch_votes, minlength=self.num_experts)
        next_expert_idx = torch.argmax(vote_counts).item()

        # Update route
        current_route = self._update_route(
            hidden_states, current_route, current_depth, next_expert_idx
        )

        return new_states, controller_state, 0, current_route, next_expert_idx
