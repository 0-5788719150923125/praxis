from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_optimizer import CosineAnnealingWarmupRestarts, create_optimizer


class ComplexModel(nn.Module):
    def __init__(self, input_dim: int = 100, hidden_dim: int = 256):
        super().__init__()

        # Complex architecture with potential for instability
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

        self.activation = nn.GELU()  # Non-linear activation
        self.dropout = nn.Dropout(0.1)

        # Initialize with slightly larger weights to stress test optimizer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.5)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Complex forward pass with skip connections
        h1 = self.activation(self.layer1(x))
        h2 = self.activation(self.layer2(h1)) + h1
        h3 = self.activation(self.layer3(h2)) + h2
        h4 = self.activation(self.layer4(h3)) + h3
        return self.output(h4)


def generate_data(
    batch_size: int = 32, input_dim: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data with complex patterns."""
    # Create input data
    X = torch.randn(batch_size, input_dim)

    # Create target with non-linear relationship
    y = (
        torch.sin(X.sum(dim=1) * 0.1)
        + torch.square(X[:, 0]) * 0.1
        + torch.exp(-torch.abs(X[:, 1])) * 0.5
    )

    # Add some noise
    y += torch.randn_like(y) * 0.1

    # Reshape y to match model output
    y = y.view(-1, 1)

    return X, y


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_steps: int = 10000,
    batch_size: int = 32,
    input_dim: int = 100,
    log_interval: int = 100,
) -> list:
    """Training loop that tracks losses and can detect instability."""
    model.train()
    criterion = nn.MSELoss()
    losses = []

    try:
        for step in range(n_steps):
            # Generate fresh data each step
            X, y = generate_data(batch_size, input_dim)

            # Forward pass
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)

            # Backward pass
            loss.backward()

            # Check for exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Gradient explosion detected at step {step}!")
                print(f"Gradient norm: {grad_norm}")
                break

            optimizer.step()

            # Log progress
            if step % log_interval == 0:
                print(
                    f"Step {step}, Loss: {loss.item():.6f}, Grad norm: {grad_norm:.6f}"
                )

            losses.append(loss.item())

            # Early stopping for extreme instability
            if loss.item() > 1e6:
                print(f"Loss explosion detected at step {step}!")
                break

    except RuntimeError as e:
        print(f"Runtime error at step {step}: {str(e)}")

    return losses


# Usage example:
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    model = ComplexModel()

    # Create optimizer (to be replaced with AdamG)
    optimizer = create_optimizer(
        model,
        optimizer_name="AdamG",
        lr=1e-3,
        weight_decay=0.001,
        weight_decouple=True,
        p=0.5,
        q=0.24,
        eps=1e-8,
    )

    # Train
    losses = train_loop(model, optimizer)

    print(f"Final loss: {losses[-1]:.6f}")
