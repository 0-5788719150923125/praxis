import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


class TransitionVisualizer:
    """
    Visualizes transition patterns between experts in a mixture-of-experts model,
    focusing on each depth transition independently rather than on full paths.
    """

    def __init__(
        self,
        num_experts: int,
        save_dir: str = "data",
        max_depth: int = 4,
        window_size: int = 1000,
        use_time_weighting: bool = True,
        save_every: int = 100,  # Save visualizations every N routes
        fig_width: int = 20,  # Adjustable figure width
        fig_height: int = 10,  # Adjustable figure height
    ) -> None:
        self.num_experts = num_experts
        self.save_dir = save_dir
        self.max_depth = max_depth - 1  # Fix off-by-one issue
        self.window_size = window_size
        self.use_time_weighting = use_time_weighting
        self.save_every = save_every
        self.fig_width = fig_width
        self.fig_height = fig_height

        os.makedirs(save_dir, exist_ok=True)

        # Tracking transitions between adjacent depth levels
        # Format: transitions[from_depth][(from_expert, to_expert)] = count
        self.transitions: List[DefaultDict[Tuple[int, int], float]] = [
            defaultdict(float) for _ in range(max_depth)
        ]

        # Time-weighted transition tracking
        self.recent_routes: List[Tuple[Sequence[int], int]] = []  # (route, timestamp)
        self.total_routes = 0

    def add_full_route(self, route: Sequence[int]) -> None:
        """Add a complete route and update transition statistics between adjacent experts."""
        if len(route) > self.max_depth + 1:
            route = route[: self.max_depth + 1]

        # Record the route for time-weighted analysis
        if self.use_time_weighting:
            self.recent_routes.append((route, self.total_routes))
            if len(self.recent_routes) > self.window_size:
                self.recent_routes.pop(0)  # Remove oldest

        # Update transition counts
        for i in range(len(route) - 1):
            if i < self.max_depth:
                from_expert = route[i]
                to_expert = route[i + 1]
                self.transitions[i][(from_expert, to_expert)] += 1

        self.total_routes += 1

        # Automatically save visualizations at the specified interval
        if self.save_every > 0 and self.total_routes % self.save_every == 0:
            self.visualize_transitions(time_weighted=False)  # All-time transitions
            if self.use_time_weighting:
                self.visualize_transitions(time_weighted=True)  # Recent transitions

            # Also save expert usage visualization on the same schedule - both all-time and recent
            self.visualize_expert_usage(time_weighted=False)  # All-time usage
            if self.use_time_weighting:
                self.visualize_expert_usage(time_weighted=True)  # Recent usage

    def reset_transitions(self) -> None:
        """Reset all transition statistics."""
        self.transitions = [defaultdict(float) for _ in range(self.max_depth)]

    def _calculate_recent_transitions(
        self,
    ) -> List[DefaultDict[Tuple[int, int], float]]:
        """Calculate transition frequencies based on only recent routes."""
        if not self.recent_routes:
            return [defaultdict(float) for _ in range(self.max_depth)]

        recent_transitions = [defaultdict(float) for _ in range(self.max_depth)]

        for route, _ in self.recent_routes:
            for i in range(len(route) - 1):
                if i < self.max_depth:
                    from_expert = route[i]
                    to_expert = route[i + 1]
                    recent_transitions[i][(from_expert, to_expert)] += 1

        return recent_transitions

    def visualize_transitions(self, time_weighted: bool = False) -> None:
        """
        Create an edge-weighted network graph showing transitions between experts at adjacent depths.

        Args:
            time_weighted: If True, visualize only recent transitions
        """
        transitions_data = (
            self._calculate_recent_transitions() if time_weighted else self.transitions
        )

        # Create figure with adjustable size
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Add more horizontal space between depth levels
        horizontal_spacing = 1.0  # Increase for more space between columns

        # Title and labels
        title_prefix = "Recent" if time_weighted else "All-time"
        ax.set_title(f"{title_prefix} Expert Transition Patterns", fontsize=16, pad=20)
        ax.set_xlabel("Depth", fontsize=14, labelpad=15)
        ax.set_ylabel("Expert", fontsize=14, labelpad=15)

        # Setup grid with extra spacing
        x_min = -0.5
        x_max = self.max_depth * horizontal_spacing + 0.5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, self.num_experts - 0.5)

        # Add grid lines with adjusted spacing
        for i in range(self.max_depth + 1):
            ax.axvline(i * horizontal_spacing, color="gray", linestyle="-", alpha=0.15)
        for i in range(self.num_experts):
            ax.axhline(i, color="gray", linestyle="-", alpha=0.15)

        # Add labels with adjusted spacing
        ax.set_xticks([i * horizontal_spacing for i in range(self.max_depth + 1)])
        ax.set_xticklabels([f"{i}" for i in range(self.max_depth + 1)])

        ax.set_yticks(range(self.num_experts))
        ax.set_yticklabels([f"{i}" for i in range(self.num_experts)])

        # Node positions with adjusted spacing
        node_positions = {}
        for depth in range(self.max_depth + 1):
            for expert in range(self.num_experts):
                node_positions[(depth, expert)] = (depth * horizontal_spacing, expert)

        # Draw edges (transitions)
        max_weight = 0
        for depth, depth_transitions in enumerate(transitions_data):
            if not depth_transitions:
                continue

            for (from_expert, to_expert), weight in depth_transitions.items():
                max_weight = max(max_weight, weight)

        # Global normalization for edge weights
        norm = Normalize(vmin=0, vmax=max_weight)

        # Create a custom colormap
        edge_cmap = plt.cm.viridis

        # Draw edges with variable width and color based on weight
        for depth, depth_transitions in enumerate(transitions_data):
            if not depth_transitions:
                continue

            for (from_expert, to_expert), weight in depth_transitions.items():
                # Skip only extremely low weight edges - show many more transitions
                if (
                    weight < max_weight * 0.0001
                ):  # Showing transitions down to 0.01% of max weight
                    continue

                # Get normalized weight for visual mapping
                norm_weight = norm(weight)

                # Edge width based on weight
                edge_width = 0.5 + 6 * norm_weight

                # Edge color based on weight
                edge_color = edge_cmap(norm_weight)

                # Edge transparency based on weight to emphasize strong connections
                edge_alpha = 0.4 + 0.5 * norm_weight

                # Get positions with adjusted spacing
                start = node_positions[(depth, from_expert)]
                end = node_positions[(depth + 1, to_expert)]

                # Calculate control points for a curved edge
                # Curve more if the from_expert and to_expert are far apart
                distance = abs(from_expert - to_expert)
                curvature = 0.2 + 0.1 * distance

                # Adjust curvature based on horizontal spacing
                curvature = curvature * horizontal_spacing

                # Direction of curve based on relative position
                curve_direction = 1 if from_expert <= to_expert else -1

                # Calculate midpoint with offset for curve
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2 + curve_direction * curvature

                # Create curved path
                curve = plt.matplotlib.path.Path(
                    [start, (mid_x, mid_y), end],
                    [
                        plt.matplotlib.path.Path.MOVETO,
                        plt.matplotlib.path.Path.CURVE3,
                        plt.matplotlib.path.Path.CURVE3,
                    ],
                )

                # Draw the edge
                patch = plt.matplotlib.patches.PathPatch(
                    curve,
                    facecolor="none",
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    alpha=edge_alpha,
                    zorder=5,
                )
                ax.add_patch(patch)

                # Only label the most significant transition at each depth
                if depth_transitions:
                    # Find the most significant transition at this depth
                    max_transition = max(depth_transitions.items(), key=lambda x: x[1])
                    max_transition_key, max_transition_weight = max_transition

                    # Only add a label if this is the current transition
                    if (from_expert, to_expert) == max_transition_key:
                        # Percentage of all transitions at this depth
                        total_depth_weight = sum(transitions_data[depth].values())
                        percentage = (weight / total_depth_weight) * 100

                        # Position the label at the midpoint of the curve
                        ax.annotate(
                            f"{percentage:.1f}%",
                            (mid_x, mid_y),
                            fontsize=10,  # Slightly larger font
                            color="black",
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.9,
                                edgecolor="none",
                            ),
                            zorder=10,
                        )

        # Draw nodes
        for depth in range(self.max_depth + 1):
            # Calculate node sizes based on usage at this depth
            node_usage = [0] * self.num_experts

            # For first depth, count outgoing transitions
            if depth == 0:
                if transitions_data[0]:  # If we have data for depth 0->1
                    for (from_expert, _), weight in transitions_data[0].items():
                        node_usage[from_expert] += weight
            # For middle depths, average incoming and outgoing
            elif depth < self.max_depth:
                # Incoming
                for (from_expert, to_expert), weight in transitions_data[
                    depth - 1
                ].items():
                    node_usage[to_expert] += weight
                # Outgoing
                for (from_expert, to_expert), weight in transitions_data[depth].items():
                    node_usage[from_expert] += weight
            # For last depth, count incoming transitions
            else:
                for (_, to_expert), weight in transitions_data[depth - 1].items():
                    node_usage[to_expert] += weight

            # Normalize node sizes
            max_usage = max(node_usage) if max(node_usage) > 0 else 1
            for expert in range(self.num_experts):
                # Node size based on usage
                norm_usage = node_usage[expert] / max_usage
                node_size = 120 + 400 * norm_usage  # Slightly larger nodes

                # Color based on depth
                node_color = plt.cm.plasma(depth / self.max_depth)

                # First draw black outline
                ax.scatter(
                    depth * horizontal_spacing,  # Adjusted spacing
                    expert,
                    s=node_size + 20,
                    color="black",
                    alpha=0.8,
                    zorder=15,
                )

                # Then draw the colored node
                ax.scatter(
                    depth * horizontal_spacing,  # Adjusted spacing
                    expert,
                    s=node_size,
                    color=node_color,
                    alpha=0.9,
                    zorder=20,
                )

        # Add a colorbar to show the transition weight scale
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Transition Frequency", fontsize=12)

        # Add additional information with simple approach
        info_text = (
            f"{'Recent' if time_weighted else 'All-time'} transitions between experts"
        )
        if time_weighted:
            info_text += (
                f" (last {min(self.window_size, len(self.recent_routes)):,} routes)"
            )
        info_text += f" | Total routes: {self.total_routes:,}"

        # Use simple text annotation without custom axes
        plt.figtext(
            0.5,
            0.01,
            info_text,
            ha="center",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.9,
                edgecolor="lightgray",
            ),
        )

        # Save the visualization with more space for the info text
        filename = (
            "transition_viz_recent.png" if time_weighted else "transition_viz.png"
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.savefig(
            os.path.join(self.save_dir, filename),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close()

    def _calculate_expert_usage(self, time_weighted: bool = False) -> List[List[float]]:
        """
        Calculate expert usage distribution at each depth, using either all-time or recent data.

        Args:
            time_weighted: If True, use only recent transitions

        Returns:
            List of usage counts for each depth
        """
        # Get the appropriate transition data
        transitions_data = (
            self._calculate_recent_transitions() if time_weighted else self.transitions
        )

        # Calculate expert usage at each depth
        expert_usage = []

        # First depth usage (from outgoing connections)
        depth0_usage = [0] * self.num_experts
        for (from_expert, _), weight in transitions_data[0].items():
            depth0_usage[from_expert] += weight
        expert_usage.append(depth0_usage)

        # Middle depths (from incoming connections)
        for depth in range(1, self.max_depth):
            depth_usage = [0] * self.num_experts
            for (_, to_expert), weight in transitions_data[depth - 1].items():
                depth_usage[to_expert] += weight
            expert_usage.append(depth_usage)

        # Last depth (from incoming connections to final depth)
        final_depth_usage = [0] * self.num_experts
        for (_, to_expert), weight in transitions_data[self.max_depth - 1].items():
            final_depth_usage[to_expert] += weight
        expert_usage.append(final_depth_usage)

        return expert_usage

    def visualize_expert_usage(self, time_weighted: bool = False) -> None:
        """
        Create a visualization showing the usage distribution of experts at each depth.

        Args:
            time_weighted: If True, visualize only recent expert usage
        """
        # Calculate expert usage using appropriate data
        expert_usage = self._calculate_expert_usage(time_weighted)

        # Create a grid of subplots, one for each depth - wider figure
        fig, axs = plt.subplots(
            1, self.max_depth + 1, figsize=(self.fig_width + 2, 8), sharey=True
        )  # Increased height from 6 to 8

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.05)  # Reduce space between plots

        # Plot each depth as a bar chart
        for depth, usage in enumerate(expert_usage):
            ax = axs[depth]

            # Calculate percentages
            total = sum(usage) if sum(usage) > 0 else 1
            percentages = [(u / total) * 100 for u in usage]

            # Color bars by their percentage
            colors = [plt.cm.viridis(p / 100) for p in percentages]

            # Create bars
            bars = ax.bar(
                range(self.num_experts),
                percentages,
                color=colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                width=0.7,  # Slightly wider bars
            )

            # Add percentage labels on top of bars with increased vertical margin
            max_percentage = max(percentages) if percentages else 0

            # Set reasonable y-limit with much more headroom for labels
            # Find global maximum before plotting
            global_max_percentage = max(
                [max(usage_list) if usage_list else 0 for usage_list in expert_usage]
            )
            global_max_percentage = (global_max_percentage / total) * 100

            # Then in the plotting loop:
            ax.set_ylim(0, max(global_max_percentage * 1.4, 10))

            # Only label the highest percentage bar in each depth
            if percentages:
                max_value_index = percentages.index(max(percentages))
                bar = bars[max_value_index]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max_percentage * 0.05),
                    f"{percentages[max_value_index]:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,  # Slightly larger
                    weight="bold",  # Make it stand out
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="none",
                    ),
                )

            # Set title and labels
            ax.set_title(f"Depth {depth}", fontsize=12)
            ax.set_xticks(range(self.num_experts))
            ax.set_xticklabels([str(i) for i in range(self.num_experts)])

            # Only set y-label on leftmost subplot
            if depth == 0:
                ax.set_ylabel("Expert Usage (%)", fontsize=12)

            # Set x-label on all subplots
            ax.set_xlabel("Expert", fontsize=10)

            # Add grid lines
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Overall title with time prefix
        title_prefix = "Recent" if time_weighted else "All-time"
        fig.suptitle(f"{title_prefix} Expert Usage Distribution by Depth", fontsize=16)

        # Add information about total routes or recent routes
        info_text = f"Based on "
        if time_weighted:
            info_text += (
                f"last {min(self.window_size, len(self.recent_routes)):,} routes"
            )
        else:
            info_text += f"{self.total_routes:,} total routes"

        plt.figtext(
            0.5,
            0.01,
            info_text,
            ha="center",
            fontsize=10,
        )

        # Save visualization with consistent naming
        filename = (
            "transition_expert_usage_recent.png"
            if time_weighted
            else "transition_expert_usage.png"
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(
            os.path.join(self.save_dir, filename),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close()


# Example usage with the same data as RouteVisualizer
if __name__ == "__main__":
    import random

    max_depth = 5
    num_experts = 5

    visualizer = TransitionVisualizer(
        num_experts=num_experts,
        save_dir="data",
        max_depth=max_depth,  # No need to adjust here since it's handled in __init__
        window_size=5000,
        use_time_weighting=True,
        save_every=1000,  # Save visualizations every 1000 routes
        fig_width=20,  # Wider figure (default was 14)
        fig_height=10,  # Same height as before
    )

    # Demo with changing patterns over time
    print("Generating routes with evolving patterns...")

    # Phase 1: Initial pattern
    print("Phase 1: Initial routing pattern")
    initial_path = [0, 2, 1, 3, 4][: visualizer.max_depth + 1]
    for i in range(3000):
        # Sometimes add noise
        if random.random() < 0.2:
            idx = random.randint(0, len(initial_path) - 1)
            new_path = initial_path.copy()
            new_path[idx] = random.randint(0, num_experts - 1)
            visualizer.add_full_route(new_path)
        else:
            visualizer.add_full_route(initial_path)

    # Phase 2: Transition to new pattern
    print("Phase 2: Transition to new pattern")
    new_path = [1, 3, 0, 4, 2][: visualizer.max_depth + 1]
    for i in range(3000):
        # Gradually increase probability of new path
        p_new = min(0.1 + i / 3000, 0.9)

        if random.random() < p_new:
            # Use new path, sometimes with noise
            if random.random() < 0.2:
                idx = random.randint(0, len(new_path) - 1)
                path = new_path.copy()
                path[idx] = random.randint(0, num_experts - 1)
                visualizer.add_full_route(path)
            else:
                visualizer.add_full_route(new_path)
        else:
            # Use old path, sometimes with noise
            if random.random() < 0.2:
                idx = random.randint(0, len(initial_path) - 1)
                path = initial_path.copy()
                path[idx] = random.randint(0, num_experts - 1)
                visualizer.add_full_route(path)
            else:
                visualizer.add_full_route(initial_path)

    # Phase 3: New dominant pattern plus intentional expert biases at certain depths
    print("Phase 3: New pattern with expert biases")

    # Create depth-specific expert biases (e.g., depth 1 heavily uses expert 2)
    depth_biases = {
        1: 3,
        3: 1,
    }  # At depth 1, bias toward expert 3; at depth 3, bias toward expert 1

    for i in range(3000):
        if random.random() < 0.8:
            # Use new path with occasional expert biases
            path = new_path.copy()

            # Apply the biases with high probability
            for depth, expert in depth_biases.items():
                if depth < len(path) and random.random() < 0.7:
                    path[depth] = expert

            # Sometimes add noise
            if random.random() < 0.1:
                idx = random.randint(0, len(path) - 1)
                path[idx] = random.randint(0, num_experts - 1)

            visualizer.add_full_route(path)
        else:
            # Random exploration of other paths
            path_len = random.randint(3, visualizer.max_depth + 1)
            random_path = [random.randint(0, num_experts - 1) for _ in range(path_len)]
            visualizer.add_full_route(random_path)

    print(f"Generated {visualizer.total_routes} routes")

    # Generate final visualizations (in case the total routes isn't divisible by save_every)
    visualizer.visualize_transitions(time_weighted=False)  # All-time transitions
    visualizer.visualize_transitions(time_weighted=True)  # Recent transitions
    visualizer.visualize_expert_usage(time_weighted=False)  # All-time expert usage
    visualizer.visualize_expert_usage(time_weighted=True)  # Recent expert usage

    print("Visualizations saved to data/")
    print("- transition_viz.png: All-time transition patterns")
    print("- transition_viz_recent.png: Recent transition patterns")
    print("- transition_expert_usage.png: All-time expert usage distribution by depth")
    print(
        "- transition_expert_usage_recent.png: Recent expert usage distribution by depth"
    )
