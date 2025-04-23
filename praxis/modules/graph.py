import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D

from praxis.modules.dense import PraxisGLU
from praxis.modules.visualization import RouteVisualizer


class PraxisGraph(nn.Module):
    """
    Graph-based expert routing, inspired by Graphformer.
    https://arxiv.org/abs/2209.10655
    """

    def __init__(self, config):
        super().__init__()
        self.debug = config.debug
        self.causal = config.causal
        self.num_layers = config.num_experts
        self.hidden_dim = config.hidden_size
        self.num_context_tokens = 0
        self.routing_scale = 0.01
        self.variance_scale = 0.01

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

        # Transform the hidden states
        self.mix_norm = nn.LayerNorm(self.hidden_dim)
        self.mixer = PraxisGLU(config)

        # Graph attention components
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        self.attn = GraphAttention(
            embed_dim=self.hidden_dim,
            output_dim=self.num_layers,
            dropout=config.dropout,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

        # Extra functionality
        self.current_route = []
        self.visualizer = (
            RouteVisualizer(
                num_experts=config.num_experts,
                max_history=10000,
                save_rate=100 * config.depth,
            )
            if self.debug
            else False
        )

    def reset_parameters(self):
        # Base layer features - keep subtle
        nn.init.normal_(self.layer_embeddings, mean=0.0, std=0.1)

        # Importance should be distinct
        nn.init.orthogonal_(self.centrality_embeddings, gain=0.1)

        # Spatial embeddings should be based on distance
        dist_init = torch.arange(self.max_distance + 1).float()
        dist_init = -0.1 * dist_init  # Negative correlation with distance
        self.spatial_embeddings.data = dist_init.unsqueeze(-1)

        # Edge features should be differentiable
        nn.init.normal_(self.edge_embeddings, mean=0.0, std=0.05)

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
        extended_mask = None
        if attention_mask is not None:
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

        trimmed_mask = None
        if attention_mask is not None:
            trimmed_mask = attention_mask[:, self.num_context_tokens :]

        return trimmed_states, trimmed_mask

    def _compute_attention_scores(
        self,
        hidden_states: torch.Tensor,
        current_layer: int,
        available_indices: List[int],
    ) -> torch.Tensor:

        # Projection stage with residual
        residual = hidden_states
        projected_states = self.mixer(self.mix_norm(hidden_states))
        hidden_states = projected_states + residual  # [B, S, H]

        # Attention stage with residual
        residual = hidden_states
        normalized_states = self.attn_norm(hidden_states)  # [B, S, H]

        # Get current layer representation
        current_embed = self.layer_embeddings[current_layer]  # [H]

        # Combine current embed with states
        query_input = current_embed.view(1, 1, -1) + normalized_states  # [B, S, H]

        # Add centrality encoding
        layer_features = (
            self.layer_embeddings + self.centrality_embeddings
        )  # [num_layers, H]

        # Compute attention scores
        attention, v = self.attn.compute_scores(
            query_input, layer_features, layer_features
        )

        # Add spatial bias before softmax
        distances = torch.abs(
            torch.arange(self.num_layers, device=attention.device) - current_layer
        )
        spatial_bias = self.spatial_embeddings[distances].transpose(
            0, 1
        )  # [1, num_layers]
        spatial_bias = spatial_bias.unsqueeze(1)  # [1, 1, num_layers]
        spatial_bias = spatial_bias.expand(
            attention.shape[0], attention.shape[1], -1
        )  # [B, S, num_layers]

        attention = attention + spatial_bias

        # Compute edge weights between current layer and available experts
        edge_feat_current = self.edge_embeddings[current_layer]
        edge_feats_available = self.edge_embeddings[
            available_indices
        ]  # [num_available, edge_dim]

        edge_weight = torch.matmul(
            edge_feat_current.unsqueeze(0),
            edge_feats_available.T,
        )  # [1, num_available]

        # Apply distance factors and expand
        distances = torch.abs(
            torch.tensor(available_indices, device=attention.device) - current_layer
        ).float()
        distance_factors = 1.0 / (distances.clamp(min=1) + 1e-6)
        edge_weight = (
            (edge_weight / distance_factors)
            .unsqueeze(1)
            .expand(attention.shape[0], attention.shape[1], -1)
        )  # [B, S, num_available]

        # Map edge_weight to full attention size
        full_edge_bias = torch.zeros_like(attention)
        full_edge_bias[:, :, available_indices] = edge_weight
        attention = attention + full_edge_bias

        if self.causal:
            # Create causal mask
            attention = self.attn.apply_causal_mask(attention)

        # Compute weighted values
        weighted_values = self.attn.compute_weights(attention, v)  # [B, S, H]

        # Project to get scores for each layer
        scores = self.attn.compute_gated_output(
            query_input, weighted_values, residual
        )  # [B, num_layers]

        return scores

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
        self.current_route.append(current_idx)

        # Get available expert indices
        available_indices = [
            i for i, expert in enumerate(original_experts) if expert in current_experts
        ]

        # Reset if no unused experts available
        if not available_indices:
            return 0, None

        # Compute attention scores and probabilities
        scores = self._compute_attention_scores(
            hidden_states, current_depth, available_indices
        )

        # Return per-example consensus scores
        seq_averaged_scores = scores.mean(dim=1)  # [B, num_layers]

        # Compute batch consensus first
        batch_averaged_scores = seq_averaged_scores.mean(dim=0)  # [num_experts]

        if self.training:
            # Apply temperature to averaged scores
            temperature = 0.1
            probs = F.gumbel_softmax(
                batch_averaged_scores.unsqueeze(0),
                tau=temperature,
                hard=False,
            )
            next_idx = torch.argmax(probs[0], dim=-1).item()

            # Compute importance losses
            importance = probs.sum(dim=0)
            importance = importance / importance.sum()
            importance_loss = (importance * probs).sum() * self.routing_scale

            # Add variance penalty with safety checks
            individual_probs = F.softmax(
                torch.clamp(seq_averaged_scores, min=-100, max=100), dim=-1
            )
            variance_loss = 0
            if individual_probs.size(0) > 1:
                route_variance = individual_probs.var(dim=0, unbiased=False).mean()
                variance_loss = route_variance * self.variance_scale

            # Sum the losses
            routing_loss = importance_loss + variance_loss
        else:
            # Use similar logic for inference
            temperature = 0.5
            scaled_scores = batch_averaged_scores / temperature
            probs = F.softmax(scaled_scores, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            routing_loss = 0

        return routing_loss, next_idx

        # Update route
        if not self.training and self.visualizer and hidden_states.size(0) == 1:
            # Just send the immediate transition
            self.visualizer.add_transition(current_idx, next_idx)

        return routing_loss, next_idx

    def reset_route(self):
        if self.debug:
            route = [str(r) for r in self.current_route]
            if not self.training:
                print(f"DEBUG: inferencing through: {' -> '.join(route)}")
            elif random.random() < 0.005:
                print(f"DEBUG: training through: {' -> '.join(route)}")
        self.current_route = []


class GraphAttention(nn.Module):
    """
    According to MEGA, "Single-head gated attention has been empirically
    shown [to be] as performant as vanilla multi-head attention."
    https://arxiv.org/abs/2209.10655
    """

    def __init__(self, embed_dim, output_dim=None, gate_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        # Q, K, V projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        if output_dim is None:
            output_dim = embed_dim
        self.output = nn.Linear(embed_dim, output_dim)

        # Gate generator G(X)
        if gate_hidden_dim is None:
            gate_hidden_dim = embed_dim * 4

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, query_input, key_input, value_input, mask=None):
        scores, v = self.compute_scores(query_input, key_input, value_input)
        out = self.compute_weights(scores, v)
        return self.compute_gated_output(query_input, out)

    def compute_scores(self, query_input, key_input, value_input):
        B, S, E = query_input.shape

        # Project inputs
        q = self.query(query_input)  # [B, S, E]
        k = self.key(key_input)  # [num_layers, E]
        v = self.value(value_input)  # [num_layers, E]

        # Expand k and v to match batch size
        k = k.unsqueeze(0).expand(B, -1, E)  # [B, num_layers, E]
        v = v.unsqueeze(0).expand(B, -1, E)  # [B, num_layers, E]

        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * (
            1.0 / math.sqrt(E)
        )  # [B, S, num_layers]

        return scores, v

    def apply_causal_mask(self, inputs):
        _, seq_len, num_experts = inputs.shape
        # Create causal mask
        seq_mask = torch.triu(
            torch.ones((seq_len, num_experts), device=inputs.device), diagonal=1
        ).bool()
        # Apply to attention scores
        inputs = inputs.masked_fill(seq_mask.unsqueeze(0), -1e9)
        return inputs

    def compute_weights(self, scores, v):
        attn = F.softmax(scores, dim=-1)  # [B, S, num_layers]
        attn = self.dropout(attn)
        # Apply attention to values
        out = torch.bmm(attn, v)  # [B, S, E]
        return out

    def compute_gated_output(self, query_input, weights, residual):
        # Generate and apply gates
        gates = self.gate_net(query_input)  # [B, S, E]
        gated_weights = weights * gates
        return self.output(gated_weights + residual)
