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
        plt.suptitle("Pathfinder Graph", fontsize=16, y=0.93)
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
            # color = plt.cm.YlOrRd(node_usage[node] / max_usage)
            color = blue_to_red(node_usage[node] / max_usage)
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
