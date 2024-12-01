import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PraxisGraph(nn.Module):
    """
    Graph-based expert routing inspired by Graphformer
    https://arxiv.org/abs/2209.10655
    """

    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_experts
        self.hidden_dim = config.num_dims
        self.num_heads = 3
        self.num_context_tokens = 3
        self.used_experts = set()
        self.routing_scale = 0.01

        # Layer embeddings (nodes)
        self.layer_embeddings = nn.Parameter(
            torch.randn(self.num_layers, self.hidden_dim)
        )

        # Centrality encoding (based on layer depth)
        self.centrality_embeddings = nn.Parameter(
            torch.randn(self.num_layers, self.hidden_dim)
        )

        # Spatial encoding (transition distances)
        self.max_distance = self.num_layers
        self.spatial_embeddings = nn.Parameter(torch.randn(self.max_distance + 1, 1))

        edge_dim = self.hidden_dim // 4
        self.edge_embeddings = nn.Parameter(torch.randn(self.num_layers, edge_dim))

        # Normalize the hidden states
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Graph attention components
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.num_layers)

        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.layer_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.centrality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.spatial_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.edge_embeddings, mean=0.0, std=0.02)

    def add_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Add layer-specific context
        context = (
            self.layer_embeddings[position].unsqueeze(0).unsqueeze(1)
        )  # [1, 1, hidden_dim]
        context = context.expand(
            hidden_states.shape[0], self.num_context_tokens, -1
        )  # [batch_size, num_context, hidden_dim]

        extended_states = torch.cat([context, hidden_states], dim=1)

        # Extend attention mask
        context_mask = attention_mask.new_ones(
            attention_mask.shape[0], self.num_context_tokens
        )
        extended_mask = torch.cat([context_mask, attention_mask], dim=1)

        return extended_states, extended_mask

    def remove_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Only remove the exact number of context tokens we added
        trimmed_states = hidden_states[:, self.num_context_tokens :, :]
        trimmed_mask = attention_mask[:, self.num_context_tokens :]

        return trimmed_states, trimmed_mask

    def _compute_attention_scores(
        self,
        hidden_states: torch.Tensor,
        current_layer: int,
        available_indices: List[int],
    ) -> torch.Tensor:

        # Get current layer representation
        current_embed = self.layer_embeddings[current_layer]
        normalized_states = self.norm(hidden_states)
        hidden_mean = normalized_states.mean(dim=1)
        query_input = (current_embed + hidden_mean).unsqueeze(1)

        # Add centrality encoding
        layer_features = self.layer_embeddings + self.centrality_embeddings

        # Project for attention
        q = self.query(query_input)
        k = self.key(layer_features)
        v = self.value(layer_features)

        # Force ensembling
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        # Compute attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attention = attention.squeeze(1)

        # Add spatial bias
        distances = torch.abs(
            torch.arange(self.num_layers, device=q.device) - current_layer
        )
        spatial_bias = self.spatial_embeddings[distances].transpose(0, 1)
        spatial_bias = spatial_bias.expand(attention.shape[0], -1)

        scores = attention + spatial_bias

        # Add edge encoding
        edge_bias = torch.zeros_like(scores)

        for i in range(scores.shape[1]):
            if i in available_indices and i != current_layer:
                # Debug edge computation
                distance = abs(current_layer - i)
                edge_feat = self.edge_embeddings[i]  # [edge_dim]

                # Correct the projection
                edge_weight = torch.matmul(
                    edge_feat.unsqueeze(0),  # [1, edge_dim]
                    self.edge_embeddings.T,  # [edge_dim, num_layers]
                )  # Result: [1, num_layers]

                # Expand to match batch size
                edge_weight = edge_weight.expand(scores.shape[0], -1)
                edge_weight = edge_weight / max(distance, 1)

                edge_bias = edge_bias + edge_weight

        scores = scores + edge_bias

        # Create mask for both unavailable and used experts
        mask = torch.ones_like(scores, dtype=torch.bool)
        for idx in available_indices:
            if idx not in self.used_experts:
                mask[:, idx] = False

        return scores.masked_fill(mask, -1e9)

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_depth: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        # Reset used experts at the start of new sequence
        current_idx = original_experts.index(current_expert)
        if current_depth == 0:
            self.used_experts = {
                current_idx
            }  # The expert at depth 0 is always first and should never be reused

        # Get available expert indices (excluding used ones)
        available_indices = [
            i
            for i, expert in enumerate(original_experts)
            if expert in current_experts and i not in self.used_experts
        ]

        # Reset if no unused experts available
        if not available_indices:
            return 0, None

        # Compute attention scores and probabilities
        scores = self._compute_attention_scores(
            hidden_states, current_depth, available_indices
        )

        probs = F.softmax(scores, dim=-1)

        # Compute routing loss with safe computation
        num_experts = len(available_indices)
        uniform_target = torch.ones_like(probs) / num_experts
        probs_safe = torch.clamp(probs, min=1e-10)
        routing_loss = (
            F.kl_div(probs_safe.log(), uniform_target, reduction="batchmean")
            * self.routing_scale
        )

        # Select next expert
        if self.training:
            probs = F.gumbel_softmax(scores, tau=1.0, hard=True)

        next_idx = torch.argmax(probs[0], dim=-1).item()

        # Add current expert to used set
        self.used_experts.add(next_idx)

        # Inference-only logging at the end
        # if not self.training:
        #     print("\nDEBUG Information:")
        #     print(f"Current expert index: {current_idx}")
        #     print(f"Available indices: {available_indices}")
        #     print(f"Attention shape: {scores.shape}")
        #     print(f"Raw attention: {scores[0].detach().cpu().numpy()}")
        #     print(f"Softmax probs: {softmax_probs[0].detach().cpu().numpy()}")
        #     print(f"Selected expert: {next_idx}")
        #     print(f"Used experts: {self.used_experts}")

        return routing_loss, next_idx
