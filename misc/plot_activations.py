import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from praxis.activations.nmda import NMDA

# Define activation functions
activation_functions = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Leaky ReLU": nn.LeakyReLU(),
    "ELU": nn.ELU(),
    "NMDA": NMDA(),
}

# Generate input data for curves
x = torch.linspace(-10, 10, 1000).unsqueeze(1)  # Shape: [1000, 1]

# Plot activation function curves
plt.figure(figsize=(10, 6))
for name, activation in activation_functions.items():
    y = activation(x)
    plt.plot(x.numpy(), y.numpy(), label=name)
plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# Generate random input data for distribution
num_samples = 10000
input_data = torch.randn(num_samples)

# Plot distributions after activation
num_activations = len(activation_functions)
cols = 3
rows = (num_activations + cols - 1) // cols

plt.figure(figsize=(15, 5 * rows))
for idx, (name, activation) in enumerate(activation_functions.items(), 1):
    activated_data = activation(input_data)
    plt.subplot(rows, cols, idx)
    plt.hist(
        activated_data.numpy(), bins=100, alpha=0.7, color="skyblue", edgecolor="black"
    )
    plt.title(f"{name} Activation")
    plt.xlabel("Activated Output")
    plt.ylabel("Frequency")
    plt.grid(True)
plt.tight_layout()
plt.show()
