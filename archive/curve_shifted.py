import matplotlib.pyplot as plt
import numpy as np


def generate_decay_values(
    depth: int,
    reverse: bool = False,
    center: float = 0.5,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> list:
    """
    Generate a list of S-shaped decaying values with adjustable center point and bounds

    Args:
        depth (int): Number of values to generate
        reverse (bool): If True, reverse the order of values
        center (float): Position of the center point (0.5 is middle, <0.5 shifts left, >0.5 shifts right)
                        Value should be between 0 and 1
        lower_bound (float): Minimum value for the decay (default: 0.0)
        upper_bound (float): Maximum value for the decay (default: 1.0)

    Returns:
        list: List of float values showing S-shaped decay
    """
    # Generate evenly spaced x values (adjusted range for S-shape)
    x = np.linspace(-6, 6, depth)

    # Calculate the shift needed based on the center parameter
    # When center = 0.5, shift = 0 (no shift)
    # When center < 0.5, shift is positive (shifts curve left)
    # When center > 0.5, shift is negative (shifts curve right)
    shift = (0.5 - center) * 12  # Scale by the range of x (-6 to 6 = 12)

    # Apply the shift to x values
    x = x + shift

    # Calculate S-shaped values using sigmoid function (0 to 1 range)
    base_values = 1 - (1 / (1 + np.exp(x)))

    # Scale values to the desired range [lower_bound, upper_bound]
    values = lower_bound + (upper_bound - lower_bound) * base_values

    # Convert to list and optionally reverse
    result = values.tolist()
    if reverse:
        result.reverse()

    return result


if __name__ == "__main__":
    # Test the function with different depths, centers, and bounds
    depths = [20]
    centers = [0.2, 0.5, 0.8]  # Left-shifted, middle, right-shifted

    # Create a figure with subplots for different center values
    fig1, axs1 = plt.subplots(2, len(centers), figsize=(15, 8))
    fig1.suptitle("S-Shaped Decay Values with Different Center Points")

    # Test both normal and reversed orders with different centers
    for i, center in enumerate(centers):
        # Normal order
        values = generate_decay_values(depths[0], reverse=False, center=center)
        axs1[0, i].plot(values, "b-o")
        axs1[0, i].set_title(f"Center={center}")
        axs1[0, i].set_ylabel("Value")
        axs1[0, i].set_xlabel("Index")
        axs1[0, i].grid(True)

        # Reversed order
        values_rev = generate_decay_values(depths[0], reverse=True, center=center)
        axs1[1, i].plot(values_rev, "r-o")
        axs1[1, i].set_title(f"Center={center} (Reversed)")
        axs1[1, i].set_ylabel("Value")
        axs1[1, i].set_xlabel("Index")
        axs1[1, i].grid(True)

    # Create a second figure to test different bounds
    bounds = [(0.0, 1.0), (0.2, 0.8), (-0.5, 1.5)]
    fig2, axs2 = plt.subplots(len(bounds), 1, figsize=(10, 10))
    fig2.suptitle("S-Shaped Decay Values with Different Bounds")

    for i, (lower, upper) in enumerate(bounds):
        values = generate_decay_values(depths[0], lower_bound=lower, upper_bound=upper)
        axs2[i].plot(values, "g-o")
        axs2[i].set_title(f"Bounds: [{lower}, {upper}]")
        axs2[i].set_ylabel("Value")
        axs2[i].set_xlabel("Index")
        axs2[i].grid(True)

    plt.tight_layout()
    plt.show()

    # Print some example values with different parameters
    print("\nExample values (depth=20):")
    print("Default bounds (0.0 to 1.0):", generate_decay_values(20))
    print(
        "Custom bounds (0.2 to 0.8):",
        generate_decay_values(20, lower_bound=0.2, upper_bound=0.8),
    )
    print(
        "Negative lower bound (-0.5 to 1.5):",
        generate_decay_values(20, lower_bound=-0.5, upper_bound=1.5),
    )
