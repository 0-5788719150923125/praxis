import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from praxis import activations
import random

# Define activation functions
activation_functions = {
    # "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Leaky ReLU": nn.LeakyReLU(),
    # "ELU": nn.ELU(),
    # "JaggedSine": activations.JaggedSine(
    #     frequencies=[1.0, 2.3, 5.9], amplitudes=[1.0, 0.1, 0.23]
    # ),
    "JaggedSine": activations.JaggedSine(
        frequencies=[random.uniform(-2, 2) for _ in range(9)],
        amplitudes=[random.uniform(-2, 2) for _ in range(9)],
    ),
    # "NMDA": NMDA(),
    # "SERF": SERF(),
    # "SinLU": SinLU(),
    "Sine": activations.Sine(),
    # "SineCosine": activations.SineCosine(),
    "PeriodicReLU": activations.PeriodicReLU(),
}

# Generate input data for activation curves
x = torch.linspace(-10, 10, 1000).unsqueeze(1)  # Shape: [1000, 1]

# ===============================
# 1. Plot Activation Function Curves
# ===============================
plt.figure(figsize=(14, 8), constrained_layout=True)  # Increased figure width
for name, activation in activation_functions.items():
    y = activation(x)
    plt.plot(x.numpy(), y.detach().numpy(), label=name)
plt.title("Activation Functions", fontsize=16)
plt.xlabel("Input", fontsize=14)
plt.ylabel("Output", fontsize=14)
plt.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 1))  # Legend outside
plt.grid(True)
plt.show()

# ===============================
# 2. Plot Distributions After Activation
# ===============================
num_samples = 10000
input_data = torch.randn(num_samples)

num_activations = len(activation_functions)
cols = 3
rows = (num_activations + cols - 1) // cols

plt.figure(figsize=(18, 5 * rows), constrained_layout=True)
for idx, (name, activation) in enumerate(activation_functions.items(), 1):
    activated_data = activation(input_data)
    plt.subplot(rows, cols, idx)
    plt.hist(
        activated_data.detach().numpy(),
        bins=100,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.title(f"{name} Activation", fontsize=14)
    plt.xlabel("Activated Output", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 3. Plot Activation Functions and Their Derivatives Using Autograd
# ===============================
# Enable gradient tracking and retain gradients for non-leaf tensors
x_deriv = torch.linspace(-10, 10, 1000, requires_grad=True).unsqueeze(1)
x_deriv.retain_grad()  # Retain gradients for non-leaf tensor

# Create subplots: each row has Activation and Derivative plots
fig, axes = plt.subplots(
    len(activation_functions),
    2,
    figsize=(16, 4 * len(activation_functions)),
    constrained_layout=True,
)

for idx, (name, activation) in enumerate(activation_functions.items()):
    # Forward pass
    y = activation(x_deriv)

    # Compute gradients
    y_sum = y.sum()
    y_sum.backward()
    dy = x_deriv.grad.clone()
    x_deriv.grad.zero_()  # Reset gradients for next iteration

    # Plot Activation Function
    axes[idx, 0].plot(x_deriv.detach().numpy(), y.detach().numpy(), color="blue")
    axes[idx, 0].set_title(f"{name} Activation Function", fontsize=14)
    axes[idx, 0].set_xlabel("Input", fontsize=12)
    axes[idx, 0].set_ylabel("Output", fontsize=12)
    axes[idx, 0].grid(True)
    # Optional: Add a single legend outside if needed
    # axes[idx, 0].legend([name], fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

    # Plot Derivative (No Legend)
    axes[idx, 1].plot(x_deriv.detach().numpy(), dy.numpy(), color="red")
    axes[idx, 1].set_title(f"{name} Derivative", fontsize=14)
    axes[idx, 1].set_xlabel("Input", fontsize=12)
    axes[idx, 1].set_ylabel("Derivative", fontsize=12)
    axes[idx, 1].grid(True)
    # Optional: Add a label inside the plot if needed
    # axes[idx, 1].text(0.05, 0.95, f"{name} Derivative", transform=axes[idx, 1].transAxes, fontsize=12, verticalalignment='top')

plt.show()

# ===============================
# 4. Plot Distributions of Derivatives
# ===============================
plt.figure(figsize=(18, 5 * rows), constrained_layout=True)
for idx, (name, activation) in enumerate(activation_functions.items(), 1):
    # Prepare input data with gradient tracking
    input_deriv = input_data.clone().detach().requires_grad_(True)
    activated = activation(input_deriv)

    # Compute gradients
    activated_sum = activated.sum()
    activated_sum.backward()
    derivatives = input_deriv.grad.detach().clone()

    # Plot derivative distribution
    plt.subplot(rows, cols, idx)
    plt.hist(
        derivatives.numpy(),
        bins=100,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    plt.title(f"{name} Derivative Distribution", fontsize=14)
    plt.xlabel("Derivative Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True)
plt.tight_layout()
plt.show()
