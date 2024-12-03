import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class HopfieldNetwork(nn.Module):
    def __init__(self, num_neurons):
        """
        Initialize a Hopfield Network

        Args:
            num_neurons (int): Number of neurons in the network
        """
        super(HopfieldNetwork, self).__init__()
        self.num_neurons = num_neurons
        # Initialize weights matrix with zeros
        self.weights = nn.Parameter(
            torch.zeros(num_neurons, num_neurons), requires_grad=False
        )
        # Set diagonal elements to zero (no self-connections)
        self.weights.data.fill_diagonal_(0)

    def hebbian_learning(self, patterns):
        """
        Train the network using Hebbian learning rule

        Args:
            patterns (torch.Tensor): Training patterns of shape (num_patterns, num_neurons)
        """
        # Reset weights
        self.weights.data.zero_()

        # Number of patterns
        num_patterns = patterns.shape[0]

        # Apply Hebbian learning rule
        for i in range(num_patterns):
            pattern = patterns[i]
            # Outer product of pattern with itself
            self.weights.data += torch.outer(pattern, pattern)

        # Normalize weights by number of patterns
        self.weights.data /= num_patterns

        # Set diagonal elements to zero
        self.weights.data.fill_diagonal_(0)

    def energy(self, state):
        """
        Calculate the energy of a given state

        Args:
            state (torch.Tensor): Current state of the network

        Returns:
            float: Energy value
        """
        return -0.5 * torch.sum(state @ self.weights @ state)

    def forward(self, input_pattern, max_iterations=100, threshold=0):
        """
        Recall a pattern using asynchronous updates

        Args:
            input_pattern (torch.Tensor): Initial state
            max_iterations (int): Maximum number of iterations
            threshold (float): Threshold for activation function

        Returns:
            torch.Tensor: Recalled pattern
        """
        current_state = input_pattern.clone()

        for _ in range(max_iterations):
            old_state = current_state.clone()

            # Update neurons asynchronously
            for i in range(self.num_neurons):
                activation = torch.dot(self.weights[i], current_state)
                current_state[i] = torch.sign(activation - threshold)

            # Check if network has converged
            if torch.all(old_state == current_state):
                break

        return current_state


# Test code
def create_test_patterns():
    """Create simple test patterns"""
    # Define simple 5x5 patterns
    pattern1 = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.float32,
    )

    pattern2 = torch.tensor(
        [
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
        ],
        dtype=torch.float32,
    )

    return torch.stack([pattern1, pattern2])


def add_noise(pattern, noise_level=0.2):
    """Add random noise to a pattern"""
    noise_mask = torch.rand_like(pattern) < noise_level
    noisy_pattern = pattern.clone()
    noisy_pattern[noise_mask] *= -1
    return noisy_pattern


def visualize_pattern(pattern):
    """Visualize a 5x5 pattern"""
    plt.imshow(pattern.view(5, 5), cmap="binary")
    plt.axis("off")


def test_hopfield_network():
    # Create network
    net = HopfieldNetwork(25)  # 5x5 = 25 neurons

    # Create and store patterns
    patterns = create_test_patterns()
    net.hebbian_learning(patterns)

    # Test with noisy patterns
    for i in range(2):
        original_pattern = patterns[i]
        noisy_pattern = add_noise(original_pattern)

        # Recall pattern
        recalled_pattern = net(noisy_pattern)

        # Plot results
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        visualize_pattern(original_pattern)
        plt.title("Original Pattern")

        plt.subplot(132)
        visualize_pattern(noisy_pattern)
        plt.title("Noisy Pattern")

        plt.subplot(133)
        visualize_pattern(recalled_pattern)
        plt.title("Recalled Pattern")

        plt.show()


if __name__ == "__main__":
    test_hopfield_network()
