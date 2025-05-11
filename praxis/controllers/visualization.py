import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


class RouteVisualizer:
    """
    Visualizes routing patterns between experts in a mixture-of-experts model,
    with support for time-weighted route tracking to identify evolving patterns.
    """

    def __init__(
        self,
        num_experts: int,
        save_dir: str = "data",
        save_rate: int = 100,
        max_routes: int = 1000,
        max_depth: int = 4,
        window_size: int = 10000,  # Size of rolling window
        decay_factor: float = 0.99,  # Optional decay for aging routes
        use_time_weighting: bool = True,  # Whether to use time weighting
    ) -> None:
        self.num_experts = num_experts
        self.save_dir = save_dir
        self.save_rate = save_rate
        self.max_routes = max_routes
        self.max_depth = max_depth - 1
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.use_time_weighting = use_time_weighting

        os.makedirs(save_dir, exist_ok=True)

        # Traditional count-based tracking
        self.routes: Dict[Tuple[int, ...], int] = defaultdict(int)
        self.total_routes = 0

        # Time-weighted tracking with rolling window
        self.recent_routes: List[Tuple[Tuple[int, ...], int]] = []  # (route, timestamp)
        self.last_decay_time = 0  # For periodic decay
        self.decay_interval = 1000  # Apply decay every N routes

    def add_full_route(self, route: Sequence[int]) -> None:
        """Add a complete route through the network with time-based tracking."""
        if len(route) > self.max_depth + 1:
            route = route[: self.max_depth + 1]

        route_tuple = tuple(route)
        timestamp = self.total_routes  # Using route count as timestamp

        # Add to traditional counter
        self.routes[route_tuple] += 1

        # Add to time-weighted rolling window
        if self.use_time_weighting:
            self.recent_routes.append((route_tuple, timestamp))

            # Maintain window size
            if len(self.recent_routes) > self.window_size:
                self.recent_routes.pop(0)  # Remove oldest

        # Apply periodic decay to all routes
        if self.use_time_weighting and (
            timestamp - self.last_decay_time >= self.decay_interval
        ):
            self._apply_decay()
            self.last_decay_time = timestamp

        self.total_routes += 1

        # Maintain max_routes limit for traditional tracking
        if len(self.routes) > self.max_routes:
            least_common = min(self.routes.items(), key=lambda x: x[1])
            del self.routes[least_common[0]]

        if self.total_routes % self.save_rate == 0:
            self._save_visualization()
            # Also save time-weighted version
            if self.use_time_weighting:
                self._save_visualization(time_weighted=True)

    def _apply_decay(self) -> None:
        """Apply decay factor to route counts to gradually age out older patterns."""
        for route in list(self.routes.keys()):
            self.routes[route] *= self.decay_factor
            # Remove routes that have decayed below threshold
            if self.routes[route] < 0.5:
                del self.routes[route]

    def get_recent_routes(self) -> List[Tuple[Tuple[int, ...], float]]:
        """Calculate route frequencies based on the recent time window."""
        if not self.recent_routes:
            return []

        # Count frequencies in recent window
        recent_counts = defaultdict(int)
        for route, _ in self.recent_routes:
            recent_counts[route] += 1

        # Normalize by window size
        window_size = len(self.recent_routes)
        return [(route, count) for route, count in recent_counts.items()]

    # Backward compatibility method
    def add_transition(self, from_expert: int, to_expert: int) -> None:
        """Legacy method for backward compatibility - does nothing."""
        pass

    def _calculate_path_points(self, start, end, rad, num_points=100):
        """Calculate points along a curved path to visually extend to node centers."""
        # Convert to numpy arrays
        start = np.array(start)
        end = np.array(end)

        # Calculate midpoint
        mid = (start + end) / 2

        # Calculate perpendicular vector
        diff = end - start
        perp = np.array([-diff[1], diff[0]])

        # Normalize and scale
        norm = np.linalg.norm(perp)
        if norm > 0:
            perp = perp / norm * rad

        # Control point for quadratic Bezier curve
        ctrl = mid + perp

        # Generate curve points
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))

        # Quadratic Bezier curve formula
        for i in range(num_points):
            curve[i] = (
                (1 - t[i]) ** 2 * start + 2 * (1 - t[i]) * t[i] * ctrl + t[i] ** 2 * end
            )

        return curve

    def _create_curved_connection(
        self, ax, start, end, color, width, path_index, num_paths
    ):
        """Create a curved connection between points that visually connects to nodes."""
        # Use a different approach to calculate curvature
        # Higher index paths get more curvature to avoid overlap
        rad_base = 0.12 + (path_index % 5) * 0.04

        # Alternate direction based on path index
        rad = rad_base * (1 if path_index % 2 == 0 else -1)

        # For longer horizontal distances, reduce the curvature
        if abs(end[0] - start[0]) > 1:
            rad *= 0.7

        # Calculate the path points
        curve = self._calculate_path_points(start, end, rad)

        # Create the line collection
        line = ax.plot(
            curve[:, 0],
            curve[:, 1],
            color=color,
            linewidth=width,
            alpha=0.7,
            solid_capstyle="round",
            zorder=5,
        )[0]

        return line

    def _save_visualization(self, time_weighted: bool = False) -> None:
        """Create and save a grid-based visualization showing routing paths."""
        if time_weighted and not self.recent_routes:
            return
        elif not time_weighted and not self.routes:
            return

        # Get appropriate path data based on visualization type
        if time_weighted:
            paths = sorted(self.get_recent_routes(), key=lambda x: x[1], reverse=True)
            viz_type = "recent"
            filename = "route_viz_recent.png"
        else:
            paths = sorted(self.routes.items(), key=lambda x: x[1], reverse=True)
            viz_type = "all-time"
            filename = "route_viz.png"

        if not paths:
            return

        # Fixed to match actual depth range (0-4 = 5 positions)
        display_depth = self.max_depth + 1

        # Create figure
        fig_width = 14
        fig_height = 8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Set up grid
        ax.set_xlim(-0.8, display_depth - 0.2)
        ax.set_ylim(-0.8, self.num_experts - 0.2)

        # Add grid lines
        for i in range(display_depth):
            ax.axvline(i, color="gray", linestyle="-", alpha=0.15)
        for i in range(self.num_experts):
            ax.axhline(i, color="gray", linestyle="-", alpha=0.15)

        # Label axes
        ax.set_xticks(range(display_depth))
        ax.set_xticklabels([f"{i}" for i in range(display_depth)])
        ax.set_xlabel("Depth", fontsize=12, labelpad=10)

        ax.set_yticks(range(self.num_experts))
        ax.set_yticklabels([f"{i}" for i in range(self.num_experts)])
        ax.set_ylabel("Expert", fontsize=12, labelpad=10)

        title_prefix = "Recent" if time_weighted else "All-time"
        ax.set_title(f"{title_prefix} Expert Routing Patterns", fontsize=14, pad=15)

        # Draw paths
        top_paths = paths[:15]
        total_count = sum(count for _, count in top_paths)
        cmap = plt.cm.viridis
        max_count = max(count for _, count in paths)

        # Add legend entries
        legend_lines = []
        legend_labels = []

        # Draw connections FIRST (so nodes will be on top)
        for i, (path, count) in enumerate(top_paths):
            if len(path) > display_depth:
                path = path[:display_depth]

            # Calculate line width based on frequency
            line_width = 1 + 3 * (count / max_count)
            color = cmap(i / max(1, len(top_paths) - 1))

            # Add to legend
            legend_lines.append(plt.Line2D([0], [0], color=color, linewidth=line_width))
            percentage = (count / total_count) * 100
            legend_labels.append(f"Path {i+1}: {percentage:.1f}%")

            # Draw connections FIRST
            for j in range(len(path) - 1):
                start = (j, path[j])
                end = (j + 1, path[j + 1])
                self._create_curved_connection(
                    ax, start, end, color, line_width, i, len(top_paths)
                )

        # Now draw ALL nodes AFTER connections to ensure they're on top
        for i, (path, count) in enumerate(top_paths):
            if len(path) > display_depth:
                path = path[:display_depth]

            # Calculate line width and color again
            line_width = 1 + 3 * (count / max_count)
            color = cmap(i / max(1, len(top_paths) - 1))

            # Draw nodes with slightly larger size to cover connection ends
            node_size = line_width * 35  # Slightly larger than before

            # Draw outline nodes first (black border, slightly larger)
            for j, expert in enumerate(path):
                ax.scatter(
                    j,
                    expert,
                    s=node_size + 10,  # Larger for outline
                    color="black",
                    zorder=18,
                )

            # Then draw actual colored nodes on top
            for j, expert in enumerate(path):
                ax.scatter(
                    j,
                    expert,
                    s=node_size,
                    color=color,
                    zorder=20,
                )

        # Add info text
        if time_weighted:
            window_size = min(len(self.recent_routes), self.window_size)
            info_text = (
                f"Recent routes (last {window_size:,})   "
                f"Unique paths: {len(paths):,}   "
                f"Total routes tracked: {self.total_routes:,}"
            )
        else:
            info_text = (
                f"All-time data   "
                f"Unique paths: {len(self.routes):,}   "
                f"Total routes tracked: {self.total_routes:,}"
            )

        # Create stats box
        text_box = plt.annotate(
            info_text,
            xy=(0.5, 0.01),
            xycoords="figure fraction",
            ha="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.8,
                edgecolor="lightgray",
            ),
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.15, 1, 0.98])

        # Add legend
        if legend_lines:
            legend = ax.legend(
                legend_lines,
                legend_labels,
                title="Path Distribution",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=min(5, len(legend_lines)),
                fontsize=9,
                frameon=True,
                framealpha=0.8,
            )
            legend.get_frame().set_facecolor("#f8f8f8")

        # Save visualization
        plt.savefig(
            os.path.join(self.save_dir, filename),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close()

    def save_comparison_visualization(self) -> None:
        """Create a visualization comparing recent vs all-time popular routes."""
        if not self.routes or not self.recent_routes:
            return

        # Get data for both metrics
        all_time_routes = dict(
            sorted(self.routes.items(), key=lambda x: x[1], reverse=True)[:15]
        )
        recent_routes = dict(self.get_recent_routes())

        # Calculate trend scores (positive = trending up, negative = trending down)
        comparison_data = []

        # Convert to list before slicing (fix TypeError: 'set' object is not subscriptable)
        combined_routes = list(
            set(list(all_time_routes.keys()) + list(recent_routes.keys()))
        )
        for route in combined_routes[:20]:
            all_time_count = all_time_routes.get(route, 0)
            recent_count = recent_routes.get(route, 0)

            # Skip routes with very low counts
            if all_time_count < 0.5 and recent_count < 0.5:
                continue

            # Calculate a normalized trend score
            if all_time_count > 0:
                # Normalize recent count based on window size
                window_ratio = min(self.window_size, len(self.recent_routes)) / max(
                    1, self.total_routes
                )
                expected_count = all_time_count * window_ratio
                trend_score = (recent_count - expected_count) / (
                    expected_count + 1
                )  # +1 to avoid division by zero
            else:
                # New route not in all-time top routes
                trend_score = 1.0  # Maximum trend score

            comparison_data.append((route, all_time_count, recent_count, trend_score))

        # Sort by trend score
        comparison_data.sort(key=lambda x: x[3], reverse=True)

        # Create visualization
        display_depth = self.max_depth + 1
        fig, ax = plt.subplots(figsize=(16, 10))

        # Set up grid similar to previous visualizations
        # Note: display_depth is max_depth + 1, which is the correct number of positions (0 to max_depth)
        ax.set_xlim(-0.8, display_depth - 0.2)
        ax.set_ylim(-0.8, self.num_experts - 0.2)

        # Grid lines and labels
        for i in range(display_depth):
            ax.axvline(i, color="gray", linestyle="-", alpha=0.15)
        for i in range(self.num_experts):
            ax.axhline(i, color="gray", linestyle="-", alpha=0.15)

        ax.set_xticks(range(display_depth))
        ax.set_xticklabels([f"{i}" for i in range(display_depth)])
        ax.set_xlabel("Depth", fontsize=12, labelpad=10)

        ax.set_yticks(range(self.num_experts))
        ax.set_yticklabels([f"{i}" for i in range(self.num_experts)])
        ax.set_ylabel("Expert", fontsize=12, labelpad=10)

        ax.set_title("Trending Expert Routing Patterns", fontsize=16, pad=15)

        # Draw paths
        legend_lines = []
        legend_labels = []

        # Use different color for trending up vs down
        trend_cmap_up = plt.cm.Greens
        trend_cmap_down = plt.cm.Reds

        # Draw connections
        for i, (path, all_time, recent, trend) in enumerate(comparison_data[:10]):
            if len(path) > display_depth:
                path = path[:display_depth]

            # Width based on recent popularity
            width_factor = 1 + 2 * (
                recent / (max(r for _, _, r, _ in comparison_data[:10]) + 0.1)
            )

            # Color based on trend (green = up, red = down)
            if trend >= 0:
                # Trending up or new
                color = trend_cmap_up(min(0.3 + 0.7 * trend, 0.9))
                trend_desc = f"↑ {trend:.1f}x"
            else:
                # Trending down
                color = trend_cmap_down(min(0.3 + 0.7 * abs(trend), 0.9))
                trend_desc = f"↓ {abs(trend):.1f}x"

            legend_lines.append(
                plt.Line2D([0], [0], color=color, linewidth=width_factor * 2)
            )
            legend_labels.append(f"Path {i+1}: {trend_desc}")

            # Draw connections
            for j in range(len(path) - 1):
                start = (j, path[j])
                end = (j + 1, path[j + 1])
                self._create_curved_connection(
                    ax, start, end, color, width_factor, i, 10
                )

            # Draw nodes
            node_size = width_factor * 30
            for j, expert in enumerate(path):
                # Outline
                ax.scatter(j, expert, s=node_size + 10, color="black", zorder=18)
                # Fill
                ax.scatter(j, expert, s=node_size, color=color, zorder=20)

        # Info text
        window_size = min(len(self.recent_routes), self.window_size)
        info_text = f"Comparison: Recent ({window_size:,} routes) vs All-time ({self.total_routes:,} routes) patterns"

        plt.annotate(
            info_text,
            xy=(0.5, 0.01),
            xycoords="figure fraction",
            ha="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.8,
                edgecolor="lightgray",
            ),
        )

        # Legend
        if legend_lines:
            legend = ax.legend(
                legend_lines,
                legend_labels,
                title="Trending Paths (↑ gaining, ↓ declining)",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=min(5, len(legend_lines)),
                fontsize=9,
                frameon=True,
                framealpha=0.8,
            )
            legend.get_frame().set_facecolor("#f8f8f8")

        # Save visualization
        plt.tight_layout(rect=[0, 0.15, 1, 0.98])
        plt.savefig(
            os.path.join(self.save_dir, "route_viz_trend.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close()


if __name__ == "__main__":
    import random

    max_depth = 5
    num_experts = 5

    visualizer = RouteVisualizer(
        num_experts=num_experts,
        save_dir="data",
        save_rate=1000,
        window_size=5000,  # Keep last 5000 routes for time-weighted analysis
        use_time_weighting=True,
        max_depth=max_depth,  # Ensure x-axis is 0-4 (5 positions total)
    )

    # Demo with changing patterns over time
    print("Generating routes with evolving patterns...")

    # Phase 1: Initial pattern
    print("Phase 1: Initial routing pattern")
    initial_path = [0, 2, 1, 3, 4][
        : visualizer.max_depth + 1
    ]  # Ensure path matches max_depth
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
    print("Phase 2: Transition to new routing pattern")
    new_path = [1, 3, 0, 4, 2][
        : visualizer.max_depth + 1
    ]  # Ensure path matches max_depth
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

    # Phase 3: New dominant pattern
    print("Phase 3: New dominant routing pattern")
    for i in range(3000):
        # High probability of new path
        if random.random() < 0.8:
            # Use new path, sometimes with noise
            if random.random() < 0.1:
                idx = random.randint(0, len(new_path) - 1)
                path = new_path.copy()
                path[idx] = random.randint(0, num_experts - 1)
                visualizer.add_full_route(path)
            else:
                visualizer.add_full_route(new_path)
        else:
            # Random exploration of other paths
            path_len = random.randint(3, 6)
            random_path = [random.randint(0, num_experts - 1) for _ in range(path_len)]
            visualizer.add_full_route(random_path)

    print(f"Generated {visualizer.total_routes} routes")

    # Force final visualizations
    visualizer._save_visualization(time_weighted=False)  # All-time popular routes
    visualizer._save_visualization(time_weighted=True)  # Recent popular routes
    visualizer.save_comparison_visualization()  # Trending analysis

    print("Visualizations saved to data/")
    print("- route_viz.png: All-time popular routes")
    print("- route_viz_recent.png: Recently popular routes")
    print("- route_viz_trend.png: Trending analysis (rising/falling patterns)")
