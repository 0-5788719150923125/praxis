import math
import os
from collections import defaultdict, deque
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, FancyArrowPatch


class PraxisGraph(nn.Module):
    """
    Graph-based expert routing inspired by Graphformer
    https://arxiv.org/abs/2209.10655
    """

    def __init__(self, config):
        super().__init__()
        self.debug = config.debug
        self.num_layers = config.num_experts
        self.hidden_dim = config.num_dims
        self.num_heads = 3
        self.num_context_tokens = 3
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

        visualize_routes = self.debug
        self.route_visualizer = (
            RouteVisualizer(num_experts=config.num_experts)
            if visualize_routes
            else None
        )
        self.current_route = []

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

        return scores + edge_bias

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
            probs = F.gumbel_softmax(scores, tau=1.0, hard=True)

        # Mean across batch before selection
        batch_averaged_probs = probs.mean(dim=0)  # [num_experts]
        next_idx = torch.argmax(batch_averaged_probs, dim=-1).item()

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

        # Initialize or update route
        if current_depth == 0:
            self.current_route = [current_idx]
        elif not self.training and self.route_visualizer is not None:
            self.current_route.append(next_idx)
            if current_depth == self.num_layers - 1:  # At end of sequence
                self.route_visualizer.add_route(self.current_route)
                self.current_route = []

        return routing_loss, next_idx


class RouteVisualizer:
    def __init__(
        self,
        num_experts: int,
        save_dir: str = "data",
        max_history: int = 10000,
        save_rate: int = 10,
    ):
        self.num_experts = num_experts
        self.save_dir = save_dir
        self.max_history = max_history
        self.save_rate = save_rate
        self.route_history = deque(maxlen=max_history)
        self.route_counts = defaultdict(int)
        self.recurrent_counts = defaultdict(int)
        self.inference_count = 0

    def add_route(self, route: List[int]):
        self.route_history.append(route)

        for i in range(len(route) - 1):
            edge = (route[i], route[i + 1])
            self.route_counts[edge] += 1

        for i, expert in enumerate(route):
            for j in range(i + 1, len(route)):
                if route[j] == expert:
                    self.recurrent_counts[expert] += 1
                    break

        self.inference_count += 1
        if self.inference_count % self.save_rate == 0:
            self.save_visualization()
            if self.inference_count == self.save_rate:
                print(f"Saving graph visualization to: {self.save_dir}/route_viz.png")

    def save_visualization(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.suptitle("Graph Routing", fontsize=16, y=0.98)

        G = nx.DiGraph()
        for i in range(self.num_experts):
            G.add_node(i)

        total_edge_weight = sum(self.route_counts.values()) + sum(
            self.recurrent_counts.values()
        )
        if total_edge_weight == 0:
            print("Warning: No edges found!")
            return

        node_usage = defaultdict(int)
        for (src, dst), count in self.route_counts.items():
            node_usage[src] += count
            node_usage[dst] += count
        for node, count in self.recurrent_counts.items():
            node_usage[node] += count

        total_usage = sum(node_usage.values())
        pos = nx.spring_layout(G, k=2.0, iterations=50)

        # Create color map for edges (reversed order for intuitive high=red)
        blue_to_red = LinearSegmentedColormap.from_list("", ["blue", "red"])
        max_edge_count = max(max(self.route_counts.values()), 1)  # Avoid div by zero

        # Draw edges for all transitions (including self-loops)
        for (src, dst), count in self.route_counts.items():
            num_curves = min(int(np.sqrt(count)), 10)

            # Calculate color intensity based on relative count
            color_val = count / max_edge_count
            edge_color = blue_to_red(color_val)

            for _ in range(num_curves):
                rad = 0.2 + np.random.uniform(-0.1, 0.1)
                alpha = 0.15 + np.random.uniform(0, 0.1)

                # Special handling for self-loops, but maintain color consistency
                if src == dst:
                    rad = 0.3 + np.random.uniform(0, 0.2) * (count / total_edge_weight)
                    alpha = 0.2

                # Draw the edge with consistent coloring
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(src, dst)],
                    edge_color=[edge_color],
                    width=1.5,
                    alpha=alpha,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowsize=20,
                )

        # Draw nodes
        node_colors = [
            plt.cm.YlOrRd(node_usage[node] / max(node_usage.values()))
            for node in G.nodes()
        ]
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500)
        nodes.set_zorder(1000)

        # Draw labels
        labels = nx.draw_networkx_labels(G, pos)
        for text in labels.values():
            text.set_zorder(1001)

        # Create legend
        legend_lines = []
        legend_labels = []

        legend_labels.append("Expert Usage")
        legend_lines.append(plt.Line2D([0], [0], color="none"))

        for node in sorted(node_usage.keys()):
            count = node_usage[node]
            percentage = (count / total_usage) * 100
            color = plt.cm.YlOrRd(node_usage[node] / max(node_usage.values()))
            legend_lines.append(
                plt.Line2D(
                    [0], [0], color=color, marker="o", linestyle="", markersize=10
                )
            )
            legend_labels.append(f"E{node}: {count} ({percentage:.1f}%)")

        legend_lines.append(plt.Line2D([0], [0], color="none"))
        legend_labels.append("")

        legend_labels.append("Top Transitions")
        legend_lines.append(plt.Line2D([0], [0], color="none"))

        sorted_edges = sorted(
            self.route_counts.items(), key=lambda x: x[1], reverse=True
        )
        for (src, dst), count in sorted_edges[:5]:
            color_val = count / max_edge_count
            edge_color = blue_to_red(color_val)
            legend_lines.append(
                plt.Line2D([0], [0], color=edge_color, marker=">", markersize=8)
            )
            legend_labels.append(f"{src}→{dst}: {count}")

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
    num_routes = 10000
    max_route_length = 10
    recurrent_loop_probability = 0.3

    visualizer = RouteVisualizer(
        num_experts=num_experts, save_dir="data", save_rate=100
    )

    for _ in range(num_routes):
        route_length = random.randint(1, max_route_length)
        route = [random.randint(0, num_experts - 1)]

        for _ in range(route_length - 1):
            if random.random() < recurrent_loop_probability:
                route.append(route[-1])  # Generate a recurrent loop
            else:
                route.append(
                    random.randint(0, num_experts - 1)
                )  # Generate a regular transition

        visualizer.add_route(route)

    print("Visualization saved to data/route_viz.png")
