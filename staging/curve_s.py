import matplotlib.pyplot as plt
import numpy as np


def generate_decay_values(depth: int, reverse: bool = False) -> list:
    """
    Generate a list of S-shaped decaying values from 1.0 to near 0.0

    Args:
        depth (int): Number of values to generate
        reverse (bool): If True, reverse the order of values

    Returns:
        list: List of float values showing S-shaped decay
    """
    # Generate evenly spaced x values (adjusted range for S-shape)
    x = np.linspace(-6, 6, depth)  # Using broader range for sigmoid

    # Calculate S-shaped values using sigmoid function
    # 1) Calculate basic sigmoid: 1 / (1 + e^x)
    # 2) Flip it to get decay: 1 - sigmoid
    values = 1 - (1 / (1 + np.exp(x)))

    # Convert to list and optionally reverse
    result = values.tolist()
    if reverse:
        result.reverse()

    return result


if __name__ == "__main__":
    # Test the function with different depths
    depths = [10, 20, 50]

    # Create a figure with subplots
    fig, axs = plt.subplots(2, len(depths), figsize=(15, 8))
    fig.suptitle("S-Shaped Decay Values")

    # Test both normal and reversed orders
    for i, depth in enumerate(depths):
        # Normal order
        values = generate_decay_values(depth, reverse=False)
        axs[0, i].plot(values, "b-o")
        axs[0, i].set_title(f"Depth={depth}")
        axs[0, i].set_ylabel("Value")
        axs[0, i].set_xlabel("Index")
        axs[0, i].grid(True)

        # Reversed order
        values_rev = generate_decay_values(depth, reverse=True)
        axs[1, i].plot(values_rev, "r-o")
        axs[1, i].set_title(f"Depth={depth} (Reversed)")
        axs[1, i].set_ylabel("Value")
        axs[1, i].set_xlabel("Index")
        axs[1, i].grid(True)

    plt.tight_layout()
    plt.show()

    # Print some example values
    print("\nExample values (depth=10):")
    print("Normal order:", generate_decay_values(10))
    print("Reversed:", generate_decay_values(10, reverse=True))
