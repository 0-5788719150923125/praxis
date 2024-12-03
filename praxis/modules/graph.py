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
        self.super_debug = False
        self.causal = config.causal
        self.num_layers = config.num_experts
        self.hidden_dim = config.num_dims
        self.num_context_tokens = 3
        self.routing_scale = 0.01
        self.step = 0

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
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.num_layers)

        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

        # Extra functionality
        self.visualizer = (
            RouteVisualizer(num_experts=config.num_experts) if self.debug else False
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

        # Project for attention
        q = self.query(query_input)  # [B, S, H]
        k = self.key(layer_features)  # [num_layers, H]
        v = self.value(layer_features)  # [num_layers, H]

        # Force ensembling
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        # Compute raw attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.hidden_dim
        )  # [B, S, num_layers]

        # Force sparsity
        attention = self.dropout(attention)

        if self.causal:
            _, seq_len, num_experts = attention.shape
            # Create causal mask
            seq_mask = torch.triu(
                torch.ones((seq_len, num_experts), device=q.device), diagonal=1
            ).bool()
            # Apply to attention scores
            attention = attention.masked_fill(seq_mask.unsqueeze(0), -1e9)

        # Add spatial bias before softmax
        distances = torch.abs(
            torch.arange(self.num_layers, device=q.device) - current_layer
        )
        spatial_bias = self.spatial_embeddings[distances].transpose(
            0, 1
        )  # [1, num_layers]
        spatial_bias = spatial_bias.unsqueeze(1)  # [1, 1, num_layers]
        spatial_bias = spatial_bias.expand(
            attention.shape[0], attention.shape[1], -1
        )  # [B, S, num_layers]

        attention = attention + spatial_bias

        # Add edge bias before softmax
        edge_bias = torch.zeros_like(attention)
        for i in range(attention.shape[2]):  # iterate over num_layers
            if i in available_indices and i != current_layer:
                distance = abs(current_layer - i)
                edge_feat = self.edge_embeddings[i]
                edge_weight = torch.matmul(
                    edge_feat.unsqueeze(0),
                    self.edge_embeddings.T,
                )  # [1, num_layers]
                edge_weight = edge_weight.unsqueeze(1)  # [1, 1, num_layers]
                edge_weight = edge_weight.expand(
                    attention.shape[0], attention.shape[1], -1
                )  # [B, S, num_layers]
                edge_weight = edge_weight / max(distance, 1)
                edge_bias = edge_bias + edge_weight

        attention = attention + edge_bias

        # Convert to probabilities
        attention_probs = F.softmax(attention, dim=-1)  # [B, S, num_layers]

        # Apply to values
        weighted_values = torch.matmul(attention_probs, v)  # [B, S, H]

        # Final residual connection for attention
        hidden_states = weighted_values + residual

        # Project to get scores for each layer
        scores = self.output(hidden_states)  # [B, num_layers]

        # Return per-example consensus scores
        return scores.mean(dim=1)  # [B, num_layers]

    def _get_temperature(self, step: int, tau_min=0.1, tau_max=1.0, period=10000):
        """
        Calculates the temperature for a given step using a cyclical schedule.

        The temperature oscillates between tau_min and tau_max following a cosine wave,
        starting at tau_max at step 0.
        """
        if tau_min < 0 or tau_max <= tau_min:
            raise ValueError("Ensure that 0 <= tau_min < tau_max.")

        base = (tau_max + tau_min) / 2
        amplitude = (tau_max - tau_min) / 2
        tau = base + amplitude * math.cos(2 * math.pi * step / period)
        return tau

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
            tau = self._get_temperature(self.step)
            self.step += 1
            probs = F.gumbel_softmax(scores, tau=tau, hard=True)

        # Mean across batch before selection
        batch_averaged_probs = probs.mean(dim=0)  # [num_experts]
        next_idx = torch.argmax(batch_averaged_probs, dim=-1).item()

        # Inference-only logging at the end
        if not self.training and self.super_debug:
            print("\nDEBUG Information:")
            print(f"Current expert index: {current_idx}")
            print(f"Available indices: {available_indices}")
            print(f"Attention shape: {scores.shape}")
            print(f"Raw attention: {scores[0].detach().cpu().numpy()}")
            print(f"Softmax probs: {softmax_probs[0].detach().cpu().numpy()}")
            print(f"Selected expert: {next_idx}")
            print(f"Used experts: {self.used_experts}")

        # Update route
        if not self.training and self.visualizer and hidden_states.size(0) == 1:
            # Just send the immediate transition
            self.visualizer.add_transition(current_idx, next_idx)

        return routing_loss, next_idx


class RouteVisualizer:
    def __init__(
        self,
        num_experts: int,
        save_dir: str = "data",
        max_history: int = 10000,
        save_rate: int = 100,
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

    def prune_history(self):
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

    def add_transition(self, from_expert: int, to_expert: int):
        # Add new transition
        self.transition_history.append((from_expert, to_expert))
        self.route_counts[(from_expert, to_expert)] += 1
        if from_expert == to_expert:
            self.recurrent_counts[from_expert] += 1
        self.total_events += 1

        # Prune if needed
        self.prune_history()

        # Handle visualization saving
        self.inference_count += 1
        if self.inference_count % self.save_rate == 0:
            self.save_visualization()

    def get_current_counts(self) -> Tuple[Dict, Dict, List]:
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

            # Calculate arrow parameters (scaled down with circle size)
            arrow_angle = angle + np.pi / 4
            arrow_length = radius * 0.4  # Shorter arrows for smaller circles
            arrow_start_x = circle_center_x + radius * 0.6 * np.cos(arrow_angle)
            arrow_start_y = circle_center_y + radius * 0.6 * np.sin(arrow_angle)
            arrow_end_x = arrow_start_x + arrow_length * np.cos(arrow_angle)
            arrow_end_y = arrow_start_y + arrow_length * np.sin(arrow_angle)

            # Alpha based on count
            base_alpha = 0.8 / np.sqrt(num_loops)
            alpha = max(0.1, min(0.8, base_alpha))

            loops.append(
                {
                    "center": (0, 0),  # Will be transformed
                    "radius": radius,
                    "transform": transform,
                    "arrow_start": (arrow_start_x, arrow_start_y),
                    "arrow_end": (arrow_end_x, arrow_end_y),
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

    def save_visualization(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.suptitle("Expert-Routing Graph", fontsize=16, y=0.98)
        ax.set_aspect("equal", adjustable="datalim")

        G = nx.DiGraph()
        for i in range(self.num_experts):
            G.add_node(i)

        route_counts, recurrent_counts = self.get_current_counts()

        total_edge_weight = sum(route_counts.values())
        if total_edge_weight == 0:
            print("Warning: No edges found!")
            return

        # Calculate node usage from current transitions
        node_usage = defaultdict(int)
        for (src, dst), count in route_counts.items():
            node_usage[src] += count
            node_usage[dst] += count

        total_usage = sum(node_usage.values())

        # Constrain spring_layout to keep nodes fully visible
        x_scale = 0.8  # Shrink layout area to ensure nodes stay in bounds
        y_scale = 0.7  # Adjusted for aspect ratio

        pos = nx.spring_layout(
            G,
            k=2.0,
            iterations=50,
            scale=min(x_scale, y_scale),
            center=(0, 0),
        )

        blue_to_red = LinearSegmentedColormap.from_list("", ["blue", "red"])
        max_edge_count = max(route_counts.values()) if route_counts else 1

        # Draw edges
        for (src, dst), count in route_counts.items():
            color_val = count / max_edge_count
            edge_color = blue_to_red(color_val)

            if src == dst:
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
                num_curves = min(int(np.sqrt(count)), 10)
                for _ in range(num_curves):
                    rad = 0.2 + np.random.uniform(-0.1, 0.1)
                    alpha = 0.15 + np.random.uniform(0, 0.1)
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(src, dst)],
                        edge_color=[edge_color],
                        width=1.5,
                        alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}",
                        arrowsize=1,
                    )

        # Draw nodes with feathering effect
        max_usage = max(node_usage.values()) if node_usage else 1
        node_colors = {}

        # Draw nodes with feathering effect
        for node in G.nodes():
            color = plt.cm.YlOrRd(node_usage[node] / max_usage)
            node_colors[node] = color
            self._create_feathered_node(ax, pos[node], color)

        # Draw labels with adaptive colors
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

        # Create legend
        legend_lines = []
        legend_labels = []

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

        legend_lines.append(plt.Line2D([0], [0], color="none"))
        legend_labels.append("")

        sorted_edges = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)

        # Use log scale for color mapping to better distribute colors
        counts = np.array(list(route_counts.values()))
        min_count = min(counts)
        max_count = max(counts)

        def get_color_val(count):
            # Use log scale to better distribute colors
            if min_count == max_count:
                return 1.0
            log_min = np.log1p(min_count)
            log_max = np.log1p(max_count)
            log_count = np.log1p(count)
            return (log_count - log_min) / (log_max - log_min)

        # Show more transitions in legend with better distribution
        legend_labels.append("Top Transitions")
        legend_lines.append(plt.Line2D([0], [0], color="none"))

        # Show more transitions (e.g., top 10 instead of 5)
        num_transitions_to_show = 10
        sorted_edges = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)

        # Also show some edges from different percentiles for better distribution
        percentile_indices = [
            int(len(sorted_edges) * p) for p in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        ]
        edges_to_show = set()

        # Add top N transitions
        edges_to_show.update(sorted_edges[:num_transitions_to_show])

        # Add percentile-based transitions
        edges_to_show.update(
            sorted_edges[i] for i in percentile_indices if i < len(sorted_edges)
        )

        # Sort final selection by count
        edges_to_show = sorted(edges_to_show, key=lambda x: x[1], reverse=True)

        for (src, dst), count in edges_to_show:
            color_val = get_color_val(count)
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

        plt.tight_layout(rect=[0, 0, 1, 1.0])

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
    num_transitions = 10000
    recurrent_loop_probability = 0.3

    visualizer = RouteVisualizer(
        num_experts=num_experts, save_dir="data", save_rate=100, max_history=1000
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
