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
        self.hidden_dim = config.num_dims
        self.num_context_tokens = 3
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


class RouteVisualizer:
    def __init__(
        self,
        num_experts: int,
        save_dir: str = "data",
        max_history: int = 10000,
        save_rate: int = 1000,
    ):
        self.num_experts = num_experts
        self.save_dir = save_dir
        self.max_history = max_history
        self.save_rate = save_rate

        # Core data structures
        self.transition_history = deque(
            maxlen=None
        )  # Store (from_expert, to_expert) tuples
        self.route_counts = defaultdict(int)
        self.recurrent_counts = defaultdict(int)
        self.total_events = 0

        self.inference_count = 0
        self.node_radius = 0.15

    def add_transition(self, from_expert: int, to_expert: int):
        # Add new transition
        self.transition_history.append((from_expert, to_expert))
        self.route_counts[(from_expert, to_expert)] += 1
        if from_expert == to_expert:
            self.recurrent_counts[from_expert] += 1
        self.total_events += 1

        # Prune if needed
        self._prune_history()

        # Handle visualization saving
        self.inference_count += 1
        if self.inference_count % self.save_rate == 0:
            self._save_visualization()

    def _prune_history(self):
        """Remove oldest transitions to maintain max_history limit"""
        while self.total_events > self.max_history and self.transition_history:
            # Remove oldest transition
            from_expert, to_expert = self.transition_history.popleft()

            # Update counts
            self.route_counts[(from_expert, to_expert)] -= 1
            if self.route_counts[(from_expert, to_expert)] == 0:
                del self.route_counts[(from_expert, to_expert)]

            # Update recurrence counts if applicable
            if from_expert == to_expert:
                self.recurrent_counts[from_expert] -= 1
                if self.recurrent_counts[from_expert] == 0:
                    del self.recurrent_counts[from_expert]

            self.total_events -= 1

    def _get_current_counts(self) -> Tuple[Dict, Dict, List]:
        """Get current statistics - returns pre-computed counts"""
        return self.route_counts, self.recurrent_counts

    def _get_loop_parameters(self, pos, node, count, total_edge_weight):
        """Generate parameters for varied self-loops using transformed circles"""
        center_x, center_y = pos[node]
        num_loops = count

        loops = []
        base_radius = 0.05

        for i in range(num_loops):
            # Calculate base position around node
            angle = (i * 2 * np.pi / num_loops) + np.random.uniform(-0.1, 0.1)

            # Size variation (keep modest to maintain consistent look)
            size_factor = 1.0 + np.random.uniform(-0.1, 0.1)
            radius = base_radius * size_factor

            # Position circle so inner edge touches node center
            # Distance from node center to circle center should equal circle radius
            circle_center_x = center_x + radius * np.cos(angle)
            circle_center_y = center_y + radius * np.sin(angle)

            # Generate transform variations
            scale_x = np.random.normal(0.2, 0.75)
            scale_y = np.random.normal(0.2, 0.75)
            rotation = np.random.uniform(0, 2 * np.pi)
            skew = np.random.uniform(-0.05, 0.05)  # Reduced skew range for cleaner look

            # Create transform matrix
            transform = (
                Affine2D()
                .scale(scale_x, scale_y)
                .rotate(rotation)
                .skew(skew, 0)
                .translate(circle_center_x, circle_center_y)
            )

            # Alpha based on count
            base_alpha = 0.8 / np.sqrt(num_loops)
            alpha = max(0.1, min(0.8, base_alpha))

            loops.append(
                {
                    "center": (0, 0),  # Will be transformed
                    "radius": radius,
                    "transform": transform,
                    "alpha": alpha,
                }
            )

        return loops

    def _create_feathered_node(self, ax, pos, color, alpha=1.0, zorder=1000):
        """Create a feathered node using stacked rings with fixed alpha"""
        import numpy as np
        from matplotlib.collections import PathCollection
        from matplotlib.path import Path

        center_x, center_y = pos
        base_radius = self.node_radius * 0.05

        # More rings for smoother gradient
        n_rings = 30
        n_points = 50  # Points per ring for smooth circle

        # Convert color to RGBA
        if isinstance(color, np.ndarray):
            if len(color) == 4:
                rgba = color
            else:
                rgba = np.append(color, alpha)
        else:
            rgba = plt.cm.colors.to_rgba(color, alpha)

        # Create points for all rings
        theta = np.linspace(0, 2 * np.pi, n_points)
        paths = []

        # Fixed alpha for all rings
        ring_alpha = 4.0 / n_rings  # Fixed transparency
        ring_scale = 7.0

        # Create rings from largest to smallest for proper stacking
        for i in range(n_rings - 1, -1, -1):
            # Linear spacing for more uniform gradient
            progress = i / (n_rings - 1)
            ring_radius = base_radius * (1 + progress * ring_scale)

            # Create circle points
            x = center_x + ring_radius * np.cos(theta)
            y = center_y + ring_radius * np.sin(theta)

            # Create path for this ring
            vertices = np.column_stack((x, y))
            vertices = np.vstack((vertices, vertices[0]))
            codes = [Path.MOVETO] + [Path.LINETO] * (n_points - 1) + [Path.CLOSEPOLY]
            paths.append(Path(vertices, codes))

        # Create uniform colors array with fixed alpha
        colors = [(rgba[0], rgba[1], rgba[2], ring_alpha)] * n_rings

        # Create path collection
        collection = PathCollection(
            paths, facecolors=colors, edgecolors="none", zorder=zorder
        )

        ax.add_collection(collection)

    def _get_text_color(self, background_color):
        """
        Determine appropriate text color (black or white) based on background color.
        Uses relative luminance formula to determine brightness.
        """
        if isinstance(background_color, np.ndarray):
            r, g, b = background_color[:3]
        else:
            rgb = plt.cm.colors.to_rgb(background_color)
            r, g, b = rgb

        # Calculate relative luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # Return white for dark backgrounds, black for light backgrounds
        return "white" if luminance < 0.5 else "black"

    def _get_curved_path_points(
        self, pos, src, dst, rad, num_points=20
    ):  # Reduced from 100
        """Generate points along a curved path between two nodes"""

        # Ensure positions are numpy arrays
        src_pos = np.array(pos[src], dtype=float)
        dst_pos = np.array(pos[dst], dtype=float)

        # Calculate midpoint
        mid_pos = (src_pos + dst_pos) / 2

        # Calculate perpendicular offset for control point
        diff = dst_pos - src_pos
        norm = np.array([-diff[1], diff[0]])  # Perpendicular vector

        # Normalize and scale
        length = np.linalg.norm(norm)
        if length > 0:
            norm = norm / length * rad

        # Control point
        ctrl_pos = mid_pos + norm

        # Generate curve points
        t = np.linspace(0, 1, num_points)
        t = t.reshape(-1, 1)

        # Quadratic Bezier curve
        curve = (1 - t) ** 2 * src_pos + 2 * (1 - t) * t * ctrl_pos + t**2 * dst_pos

        return curve

    def _draw_gradient_edge(self, ax, pos, src, dst, alpha):
        """Draw a single edge with a blue-to-red gradient"""

        # Get curved path points
        rad = 0.2 + np.random.uniform(-0.1, 0.1)
        points = self._get_curved_path_points(pos, src, dst, rad)

        # Create segments for LineCollection
        segments = np.concatenate(
            [points[:-1, np.newaxis], points[1:, np.newaxis]], axis=1
        )

        # Create blue-to-red gradient colors
        t = np.linspace(0, 1, len(segments))
        colors = np.zeros((len(segments), 4))
        colors[:, 0] = t  # Red increases
        colors[:, 2] = 1 - t  # Blue decreases
        colors[:, 3] = alpha  # Constant alpha

        # Create and add LineCollection with solid lines
        lc = LineCollection(
            segments,
            colors=colors,
            linewidth=1.5,
            zorder=0,
            capstyle=None,
            joinstyle=None,
            linestyle="solid",
        )
        ax.add_collection(lc)

    def _save_visualization(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.suptitle("Expert-Routing Graph", fontsize=16, y=0.93)
        plt.subplots_adjust(top=0.90)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_on()
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Keep spines (border) visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

        # Initialize graph and get counts
        G = nx.DiGraph()
        for i in range(self.num_experts):
            G.add_node(i)

        route_counts, recurrent_counts = self._get_current_counts()
        total_edge_weight = sum(route_counts.values())
        if total_edge_weight == 0:
            print("Warning: No edges found!")
            return

        # Calculate node usage
        node_usage = defaultdict(int)
        for (src, dst), count in route_counts.items():
            node_usage[src] += count
            node_usage[dst] += count
        total_usage = sum(node_usage.values())

        # Generate layout with increased scale
        pos = nx.spring_layout(
            G, k=2.0, iterations=50, scale=2.0, center=(0, 0)  # Increased scale
        )

        # Scale positions while maintaining proportions
        positions = np.array(list(pos.values()))
        max_range = max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1]))

        if max_range > 0:
            for key in pos:
                pos[key] = pos[key] / max_range

        # Set bounds with reduced margins
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)

        # Define color maps
        blue_to_red = LinearSegmentedColormap.from_list("", ["blue", "red"])
        max_edge_count = max(route_counts.values()) if route_counts else 1

        # Draw edges with gradients
        for (src, dst), count in route_counts.items():
            color_val = count / max_edge_count
            edge_color = blue_to_red(color_val)
            if src == dst:
                # Self-loops
                loops = self._get_loop_parameters(pos, src, count, total_edge_weight)
                for loop in loops:
                    circle = Circle(
                        loop["center"],
                        loop["radius"],
                        facecolor="none",
                        edgecolor=edge_color,
                        alpha=loop["alpha"],
                        zorder=0,
                        transform=loop["transform"] + ax.transData,
                    )
                    ax.add_patch(circle)
            else:
                # Regular edges with gradient
                num_curves = min(int(np.sqrt(count)), 100)
                for _ in range(num_curves):
                    alpha = 0.15 + np.random.uniform(0, 0.1)
                    self._draw_gradient_edge(ax, pos, src, dst, alpha)

        # Draw nodes
        max_usage = max(node_usage.values()) if node_usage else 1
        node_colors = {}

        for node in G.nodes():
            color = plt.cm.YlOrRd(node_usage[node] / max_usage)
            node_colors[node] = color
            self._create_feathered_node(ax, pos[node], color)

        # Add node labels
        labels = {node: str(node) for node in G.nodes()}
        for node, (x, y) in pos.items():
            text_color = self._get_text_color(node_colors[node])
            ax.text(
                x,
                y,
                labels[node],
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontweight="bold",
                zorder=2000,
            )

        # Create legend content
        legend_lines = []
        legend_labels = []

        # Expert usage section
        legend_labels.append("Expert Usage")
        legend_lines.append(plt.Line2D([0], [0], color="none"))

        for node in sorted(node_usage.keys()):
            count = node_usage[node]
            percentage = (count / total_usage) * 100 if total_usage > 0 else 0
            color = plt.cm.YlOrRd(node_usage[node] / max_usage)
            legend_lines.append(
                plt.Line2D(
                    [0], [0], color=color, marker="o", linestyle="", markersize=10
                )
            )
            legend_labels.append(f"E{node}: {count} ({percentage:.1f}%)")

        # Spacing between sections
        legend_lines.append(plt.Line2D([0], [0], color="none"))
        legend_labels.append("")

        # Transitions section
        legend_labels.append("Top Transitions")
        legend_lines.append(plt.Line2D([0], [0], color="none"))

        # Sort and select transitions to show
        num_transitions_to_show = 10
        sorted_edges = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)
        edges_to_show = sorted_edges[:num_transitions_to_show]

        # Add transitions to legend
        for (src, dst), count in edges_to_show:
            color_val = count / max_edge_count
            edge_color = blue_to_red(color_val)
            legend_lines.append(
                plt.Line2D([0], [0], color=edge_color, marker=">", markersize=8)
            )
            percentage = (count / total_edge_weight) * 100
            legend_labels.append(f"{src}→{dst}: {count} ({percentage:.1f}%)")

        legend = ax.legend(
            legend_lines,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            borderaxespad=0,
            frameon=True,
            fontsize=9,
            title="Statistics",
            title_fontsize=10,
            handletextpad=1,
            labelspacing=0.2,
        )

        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_facecolor("white")
        legend.set_zorder(2000)

        # Save with tight layout
        plt.savefig(
            os.path.join(self.save_dir, f"route_viz.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close()


if __name__ == "__main__":
    import random

    num_experts = 5
    num_transitions = 1000000
    recurrent_loop_probability = 0.3

    visualizer = RouteVisualizer(
        num_experts=num_experts, save_dir="data", save_rate=1000, max_history=10000
    )

    current_expert = random.randint(0, num_experts - 1)

    for _ in range(num_transitions):
        # Determine next expert
        if random.random() < recurrent_loop_probability:
            next_expert = current_expert  # Generate a recurrent loop
        else:
            next_expert = random.randint(
                0, num_experts - 1
            )  # Generate a regular transition
            while next_expert == current_expert:  # Avoid accidental loops
                next_expert = random.randint(0, num_experts - 1)

        # Add transition to visualizer
        visualizer.add_transition(current_expert, next_expert)

        # Update current expert for next iteration
        current_expert = next_expert

    print("Visualization saved to data/route_viz.png")
