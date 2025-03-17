import matplotlib.pyplot as plt
import numpy as np


def generate_u_shape_values(
    depth: int,
    decay_point: float = 0.3,
    ramp_point: float = 0.7,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    steepness: float = 5.0,
) -> list:
    """
    Generate a list of U-shaped values that start at upper_bound, decay to lower_bound,
    and then ramp back up to upper_bound.

    Args:
        depth (int): Number of values to generate
        decay_point (float): Position where the initial decay occurs (0.0-1.0)
        ramp_point (float): Position where the final ramp-up occurs (0.0-1.0)
        lower_bound (float): Minimum value in the U-shape (default: 0.0)
        upper_bound (float): Maximum value in the U-shape (default: 1.0)
        steepness (float): Controls how steep the decay and ramp are (default: 5.0)

    Returns:
        list: List of float values showing U-shaped pattern
    """
    # Create normalized positions from 0 to 1
    positions = np.linspace(0, 1, depth)
    values = np.zeros(depth)

    # We'll use a modified sigmoid approach that ensures values start and end at 1.0
    for i, pos in enumerate(positions):
        # For decay: Use a modified function that equals 1.0 at position 0
        # Adjusted decay function that starts at 1.0 when pos = 0
        if pos < decay_point:
            # Normalize position to 0-1 range within the decay region
            norm_pos = pos / decay_point if decay_point > 0 else 0
            # Use a function that starts at 1 and approaches 0
            decay_factor = (1 - norm_pos) ** steepness
        else:
            decay_factor = 0

        # For ramp: Use a modified function that equals 1.0 at position 1
        if pos > ramp_point:
            # Normalize position to 0-1 range within the ramp region
            norm_pos = (pos - ramp_point) / (1 - ramp_point) if ramp_point < 1 else 0
            # Use a function that ends at 1
            ramp_factor = norm_pos**steepness
        else:
            ramp_factor = 0

        # Combine factors and scale to bounds
        combined_factor = max(decay_factor, ramp_factor)
        values[i] = lower_bound + (upper_bound - lower_bound) * combined_factor

    return values.tolist()


if __name__ == "__main__":
    # Test parameters
    depth = 100

    # Create a figure with 2x2 subplots to show parameter interactions
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("U-Shape Parameter Interactions", fontsize=16)

    # Plot 1: Decay-Ramp pairs with different steepness values
    ax = axs[0, 0]
    ax.set_title("Effect of Steepness on Different Decay-Ramp Pairs", fontsize=12)

    decay_ramp_pairs = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
    steepness_values = [3.0, 5.0, 8.0]

    colors = ["b", "g", "r"]
    line_styles = ["-", "--", "-."]

    for i, (decay, ramp) in enumerate(decay_ramp_pairs):
        for j, steep in enumerate(steepness_values):
            values = generate_u_shape_values(
                depth, decay_point=decay, ramp_point=ramp, steepness=steep
            )
            ax.plot(
                values,
                color=colors[i],
                linestyle=line_styles[j],
                label=f"Decay={decay}, Ramp={ramp}, Steep={steep}",
            )

    ax.set_ylabel("Value")
    ax.set_xlabel("Index")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Plot 2: Bounds variations with different decay-ramp pairs
    ax = axs[0, 1]
    ax.set_title("Effect of Decay-Ramp Pairs on Different Bounds", fontsize=12)

    bound_pairs = [(0.0, 1.0), (0.2, 0.8), (-0.5, 1.5)]
    decay_ramp_pairs = [(0.2, 0.8), (0.3, 0.7)]

    for i, (lower, upper) in enumerate(bound_pairs):
        for j, (decay, ramp) in enumerate(decay_ramp_pairs):
            values = generate_u_shape_values(
                depth,
                decay_point=decay,
                ramp_point=ramp,
                lower_bound=lower,
                upper_bound=upper,
            )
            ax.plot(
                values,
                color=colors[i],
                linestyle=line_styles[j],
                label=f"Bounds=({lower},{upper}), Decay={decay}, Ramp={ramp}",
            )

    ax.set_ylabel("Value")
    ax.set_xlabel("Index")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Plot 3: Steepness variations with different bounds
    ax = axs[1, 0]
    ax.set_title("Effect of Bounds on Different Steepness Values", fontsize=12)

    steepness_values = [3.0, 5.0, 8.0]
    bound_pairs = [(0.0, 1.0), (0.2, 0.8)]

    for i, steep in enumerate(steepness_values):
        for j, (lower, upper) in enumerate(bound_pairs):
            values = generate_u_shape_values(
                depth, steepness=steep, lower_bound=lower, upper_bound=upper
            )
            ax.plot(
                values,
                color=colors[i],
                linestyle=line_styles[j],
                label=f"Steep={steep}, Bounds=({lower},{upper})",
            )

    ax.set_ylabel("Value")
    ax.set_xlabel("Index")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Plot 4: Fixed decay but variable ramp with different steepness
    ax = axs[1, 1]
    ax.set_title("Fixed Decay (0.3) with Variable Ramp and Steepness", fontsize=12)

    ramp_points = [0.6, 0.7, 0.8]
    steepness_values = [3.0, 8.0]

    for i, ramp in enumerate(ramp_points):
        for j, steep in enumerate(steepness_values):
            values = generate_u_shape_values(
                depth, decay_point=0.3, ramp_point=ramp, steepness=steep
            )
            ax.plot(
                values,
                color=colors[i],
                linestyle=line_styles[j],
                label=f"Ramp={ramp}, Steep={steep}",
            )

    ax.set_ylabel("Value")
    ax.set_xlabel("Index")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Add a small plot to verify first and last points are at upper_bound
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.set_title(
        "Verification that First and Last Points Equal Upper Bound", fontsize=12
    )

    test_cases = [
        {"decay": 0.2, "ramp": 0.8, "lower": 0.0, "upper": 1.0, "steep": 5.0},
        {"decay": 0.3, "ramp": 0.7, "lower": 0.2, "upper": 0.8, "steep": 3.0},
        {"decay": 0.4, "ramp": 0.6, "lower": -0.5, "upper": 1.5, "steep": 8.0},
    ]

    for i, params in enumerate(test_cases):
        values = generate_u_shape_values(
            depth,
            decay_point=params["decay"],
            ramp_point=params["ramp"],
            lower_bound=params["lower"],
            upper_bound=params["upper"],
            steepness=params["steep"],
        )

        # Plot the full curve
        ax2.plot(values, label=f"Case {i+1}: Upper bound = {params['upper']}")

        # Mark the first and last points
        ax2.plot(0, values[0], "o", markersize=8, color=f"C{i}")
        ax2.plot(len(values) - 1, values[-1], "o", markersize=8, color=f"C{i}")

        # Print verification
        print(
            f"Case {i+1}: Upper bound = {params['upper']}, First point = {values[0]}, Last point = {values[-1]}"
        )

    ax2.set_ylabel("Value")
    ax2.set_xlabel("Index")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
