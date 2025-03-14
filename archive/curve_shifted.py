import matplotlib.pyplot as plt
import numpy as np


def generate_decay_values(
    depth: int, reverse: bool = False, center: float = 0.5
) -> list:
    """
    Generate a list of S-shaped decaying values from 1.0 to near 0.0 with adjustable center point

    Args:
        depth (int): Number of values to generate
        reverse (bool): If True, reverse the order of values
        center (float): Position of the center point (0.5 is middle, <0.5 shifts left, >0.5 shifts right)
                        Value should be between 0 and 1

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

    # Calculate S-shaped values using sigmoid function
    values = 1 - (1 / (1 + np.exp(x)))

    # Convert to list and optionally reverse
    result = values.tolist()
    if reverse:
        result.reverse()

    return result


if __name__ == "__main__":
    # Test the function with different depths and center values
    depths = [20]
    centers = [0.2, 0.5, 0.8]  # Left-shifted, middle, right-shifted

    # Create a figure with subplots
    fig, axs = plt.subplots(2, len(centers), figsize=(15, 8))
    fig.suptitle("S-Shaped Decay Values with Different Center Points")

    # Test both normal and reversed orders with different centers
    for i, center in enumerate(centers):
        # Normal order
        values = generate_decay_values(depths[0], reverse=False, center=center)
        axs[0, i].plot(values, "b-o")
        axs[0, i].set_title(f"Center={center}")
        axs[0, i].set_ylabel("Value")
        axs[0, i].set_xlabel("Index")
        axs[0, i].grid(True)

        # Reversed order
        values_rev = generate_decay_values(depths[0], reverse=True, center=center)
        axs[1, i].plot(values_rev, "r-o")
        axs[1, i].set_title(f"Center={center} (Reversed)")
        axs[1, i].set_ylabel("Value")
        axs[1, i].set_xlabel("Index")
        axs[1, i].grid(True)

    plt.tight_layout()
    plt.show()

    # Print some example values with different centers
    print("\nExample values (depth=20):")
    print("Left-shifted (center=0.2):", generate_decay_values(20, center=0.2))
    print("Middle (center=0.5):", generate_decay_values(20, center=0.5))
    print("Right-shifted (center=0.8):", generate_decay_values(20, center=0.8))
