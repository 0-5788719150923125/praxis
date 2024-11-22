from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PraxisVAE(nn.Module):
    """
    A flexible Variational Autoencoder implementation with β-VAE support.
    Handles 3D inputs of shape [batch_size, seq_len, num_features].

    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        beta (float): Weight of the KL divergence term (β-VAE)
    """

    def __init__(self, input_dim: int, output_dim: int, beta: float = 1.0):
        super().__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.beta = beta

        # Calculate bottleneck dimension
        self.bottleneck_dim = max(input_dim, output_dim) // 2
        self.latent_dim = self.bottleneck_dim // 2

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(self.bottleneck_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.bottleneck_dim, self.latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, output_dim),
            nn.Sigmoid(),  # Assuming output should be in [0,1]
        )

        self.direct_projection = nn.Sequential(
            nn.Linear(output_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters mu and log_var."""
        # Reshape input: [batch_size, seq_len, features] -> [batch_size * seq_len, features]
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, self.input_dim)

        # Encode
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        # Reshape back: [batch_size * seq_len, latent_dim] -> [batch_size, seq_len, latent_dim]
        mu = mu.reshape(batch_size, seq_len, -1)
        log_var = log_var.reshape(batch_size, seq_len, -1)

        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, project_to_input: bool = False) -> torch.Tensor:
        """Decode latent vector with optional projection to input dimension."""
        batch_size, seq_len, _ = z.shape
        z = z.reshape(-1, self.latent_dim)

        # Decode to compressed dimension
        x = self.decoder(z)

        # Optionally project back to input dimension
        if project_to_input:
            x = self.direct_projection(x)

        # Reshape back
        x = x.reshape(batch_size, seq_len, -1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            tuple: (reconstruction, kl_loss * beta)
            - reconstruction has shape [batch_size, seq_len, output_dim]
            - kl_loss is the KL divergence loss term weighted by beta
        """
        # Encode
        mu, log_var = self.encode(x)

        # Sample from latent space
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decode(z)

        # Calculate KL divergence
        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=2).mean()
        )

        return reconstruction, self.beta * kl_loss


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    def run_smoke_test(description: str, test_fn) -> bool:
        """Helper function to run and report smoke tests."""
        try:
            test_fn()
            print(f"✓ {description}")
            return True
        except Exception as e:
            print(f"✗ {description}")
            print(f"  Error: {str(e)}")
            return False

    # Test 1: Model instantiation
    def test_model_instantiation():
        model = PraxisVAE(input_dim=10, output_dim=5, beta=2.0)
        assert isinstance(model, nn.Module), "Model should be a nn.Module"
        assert model.beta == 2.0, "Beta parameter not set correctly"

    # Test 2: Basic forward pass
    def test_forward_pass():
        model = PraxisVAE(input_dim=10, output_dim=5)
        x = torch.randn(32, 8, 10)  # [batch_size, seq_len, input_dim]
        reconstruction, kl_loss = model(x)
        assert reconstruction.shape == (
            32,
            8,
            5,
        ), f"Wrong output shape: {reconstruction.shape}"
        assert kl_loss.dim() == 0, "KL loss should be a scalar"

    # Test 3: Output range
    def test_output_range():
        model = PraxisVAE(input_dim=10, output_dim=5)
        x = torch.randn(16, 4, 10)
        reconstruction, _ = model(x)
        assert torch.all(reconstruction >= 0) and torch.all(
            reconstruction <= 1
        ), "Output should be in range [0, 1] due to sigmoid"

    # Test 4: Beta effect on KL loss
    def test_beta_effect():
        x = torch.randn(16, 4, 10)
        model1 = PraxisVAE(input_dim=10, output_dim=5, beta=1.0)
        model2 = PraxisVAE(input_dim=10, output_dim=5, beta=2.0)

        # Use the same weights for both models
        model2.load_state_dict(model1.state_dict())

        _, kl_loss1 = model1(x)
        _, kl_loss2 = model2(x)
        assert abs(kl_loss2 - 2 * kl_loss1) < 1e-5, "Beta should scale KL loss linearly"

    # Test 5: Shape preservation with equal dimensions
    def test_equal_dimensions():
        model = PraxisVAE(input_dim=10, output_dim=10)
        x = torch.randn(8, 6, 10)
        reconstruction, _ = model(x)
        assert (
            reconstruction.shape == x.shape
        ), "Shape should be preserved when input_dim == output_dim"

    # Test 6: Different sequence lengths
    def test_different_seq_lengths():
        model = PraxisVAE(input_dim=10, output_dim=5)
        x1 = torch.randn(8, 6, 10)
        x2 = torch.randn(8, 12, 10)
        rec1, _ = model(x1)
        rec2, _ = model(x2)
        assert rec1.shape == (8, 6, 5) and rec2.shape == (
            8,
            12,
            5,
        ), "Model should handle different sequence lengths"

    # Run all smoke tests
    tests = [
        ("Model instantiation", test_model_instantiation),
        ("Basic forward pass", test_forward_pass),
        ("Output range", test_output_range),
        ("Beta effect on KL loss", test_beta_effect),
        ("Equal dimensions", test_equal_dimensions),
        ("Different sequence lengths", test_different_seq_lengths),
    ]

    print("Running smoke tests...")
    all_passed = all(run_smoke_test(desc, test_fn) for desc, test_fn in tests)
    print(
        f"\nSmoke test summary: {'All tests passed!' if all_passed else 'Some tests failed.'}"
    )
