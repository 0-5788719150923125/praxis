from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalVAEOutput:
    """Output container for CausalVAE"""

    reconstructed: torch.Tensor  # Reconstructed input
    loss: torch.Tensor  # Total loss
    kl_loss: torch.Tensor  # KL divergence loss
    recon_loss: torch.Tensor  # Reconstruction loss
    dag_loss: torch.Tensor  # DAG structure loss


class CausalVAELayer(nn.Module):
    """
    A Causal VAE layer for sequence modeling tasks.

    Implements structured causal disentanglement following the CausalVAE paper,
    adapted for sequence data.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_concepts: int,
        concept_dim: int,
        alpha: float = 0.3,
        beta: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Validate dimensions
        if latent_dim != num_concepts * concept_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must equal num_concepts * concept_dim "
                f"({num_concepts * concept_dim})"
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # Encoder network (µ, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2), nn.LayerNorm(latent_dim * 2)
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim),
        )

        # DAG structure matrix (learned)
        self.dag_weights = nn.Parameter(torch.zeros(num_concepts, num_concepts))

        # Concept-specific transformation networks
        self.concept_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(concept_dim, concept_dim),
                    nn.LayerNorm(concept_dim),
                    nn.ReLU(),
                    nn.Linear(concept_dim, concept_dim),
                )
                for _ in range(num_concepts)
            ]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def get_dag_matrix(self) -> torch.Tensor:
        """Get the DAG adjacency matrix (lower triangular)"""
        return torch.tril(F.softplus(self.dag_weights), diagonal=-1)

    def causal_transform(
        self, z: torch.Tensor, intervention: Optional[Tuple[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Apply causal transformation with optional intervention"""
        # Reshape to [batch, seq, num_concepts, concept_dim]
        batch_size, seq_len = z.shape[:2]
        z = z.view(batch_size, seq_len, self.num_concepts, self.concept_dim)

        # Get DAG structure
        dag_adj = self.get_dag_matrix()

        # Causal transformation
        z_flat = z.view(-1, self.num_concepts, self.concept_dim)
        z_causal = z_flat + torch.matmul(dag_adj, z_flat)
        z_causal = z_causal.view(
            batch_size, seq_len, self.num_concepts, self.concept_dim
        )

        # Handle intervention if specified
        if intervention is not None:
            idx, value = intervention
            z_causal = z_causal.clone()
            z_causal[:, :, idx] = value.view(batch_size, seq_len, self.concept_dim)

        # Apply concept-specific transformations
        z_transformed = []
        for i in range(self.num_concepts):
            transformed = self.concept_transforms[i](z_causal[:, :, i])
            z_transformed.append(transformed)

        # Stack and reshape back
        z_transformed = torch.stack(z_transformed, dim=2)
        return z_transformed.view(batch_size, seq_len, self.latent_dim)

    def compute_losses(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute all loss components"""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # DAG loss (sparsity + acyclicity)
        dag_adj = self.get_dag_matrix()
        dag_loss = torch.sum(torch.abs(dag_adj)) + torch.trace(
            torch.matrix_exp(dag_adj)
        )

        # Total loss
        total_loss = recon_loss + self.alpha * kl_loss + self.beta * dag_loss

        return total_loss, kl_loss, recon_loss, dag_loss

    def forward(
        self, x: torch.Tensor, intervention: Optional[Tuple[int, torch.Tensor]] = None
    ) -> CausalVAEOutput:
        """
        Forward pass with optional intervention

        Args:
            x: Input tensor [batch, seq, input_dim]
            intervention: Optional (concept_idx, value) for intervention

        Returns:
            CausalVAEOutput containing reconstruction and losses
        """
        # Encode
        mu, logvar = self.encode(x)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Apply causal transformation
        z_transformed = self.causal_transform(z, intervention)

        # Decode
        x_recon = self.decoder(z_transformed)

        # Compute losses
        total_loss, kl_loss, recon_loss, dag_loss = self.compute_losses(
            x, x_recon, mu, logvar
        )

        return CausalVAEOutput(
            reconstructed=x_recon,
            loss=total_loss,
            kl_loss=kl_loss,
            recon_loss=recon_loss,
            dag_loss=dag_loss,
        )

    def intervene(
        self, x: torch.Tensor, concept_idx: int, concept_value: torch.Tensor
    ) -> torch.Tensor:
        """Perform intervention on a specific concept"""
        if not 0 <= concept_idx < self.num_concepts:
            raise ValueError(f"concept_idx must be between 0 and {self.num_concepts-1}")

        batch_size, seq_len = x.shape[:2]
        expected_shape = (batch_size, seq_len, self.concept_dim)

        if concept_value.shape != expected_shape:
            raise ValueError(
                f"concept_value shape should be {expected_shape}, "
                f"got {concept_value.shape}"
            )

        return self.forward(x, intervention=(concept_idx, concept_value)).reconstructed


import pytest
import torch
import torch.nn.functional as F


def test_causal_vae_initialization():
    """Test that the model initializes correctly"""
    model = CausalVAELayer(input_dim=64, latent_dim=32, num_concepts=4, concept_dim=8)

    assert model.input_dim == 64
    assert model.latent_dim == 32
    assert model.num_concepts == 4
    assert model.concept_dim == 8

    # Should raise error when dimensions don't match
    with pytest.raises(ValueError):
        CausalVAELayer(
            input_dim=64,
            latent_dim=30,  # Doesn't match num_concepts * concept_dim
            num_concepts=4,
            concept_dim=8,
        )


def test_forward_pass_shapes():
    """Test that the model produces correct output shapes"""
    batch_size = 32
    seq_len = 16
    input_dim = 64
    num_concepts = 4
    concept_dim = 8
    latent_dim = num_concepts * concept_dim

    model = CausalVAELayer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_concepts=num_concepts,
        concept_dim=concept_dim,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    # Check output shapes
    assert output.reconstructed.shape == x.shape
    assert output.loss.ndim == 0  # scalar
    assert output.kl_loss.ndim == 0
    assert output.recon_loss.ndim == 0
    assert output.dag_loss.ndim == 0


def test_intervention():
    """Test intervention functionality"""
    batch_size = 8
    seq_len = 10
    input_dim = 64
    num_concepts = 4
    concept_dim = 8
    latent_dim = num_concepts * concept_dim

    model = CausalVAELayer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_concepts=num_concepts,
        concept_dim=concept_dim,
    )

    x = torch.randn(batch_size, seq_len, input_dim)

    # Test valid intervention
    concept_idx = 1
    concept_value = torch.randn(batch_size, seq_len, concept_dim)
    output = model.intervene(x, concept_idx, concept_value)
    assert output.shape == x.shape

    # Test invalid concept index
    with pytest.raises(ValueError):
        model.intervene(x, -1, concept_value)
    with pytest.raises(ValueError):
        model.intervene(x, num_concepts, concept_value)

    # Test invalid concept value shape
    bad_value = torch.randn(batch_size, seq_len, concept_dim + 1)
    with pytest.raises(ValueError):
        model.intervene(x, concept_idx, bad_value)


def test_dag_properties():
    """Test DAG matrix properties"""
    model = CausalVAELayer(input_dim=64, latent_dim=32, num_concepts=4, concept_dim=8)

    dag_adj = model.get_dag_matrix()

    # Check that matrix is lower triangular
    assert torch.all(torch.triu(dag_adj, diagonal=0) == 0)

    # Check that diagonal is zero
    assert torch.all(torch.diagonal(dag_adj) == 0)

    # Check that values are non-negative
    assert torch.all(dag_adj >= 0)


def test_training_mode():
    """Test that the model behaves differently in training vs eval modes"""
    model = CausalVAELayer(input_dim=64, latent_dim=32, num_concepts=4, concept_dim=8)

    x = torch.randn(4, 5, 64)

    # Test training mode
    model.train()
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1.reconstructed, out2.reconstructed)

    # Test eval mode
    model.eval()
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1.reconstructed, out2.reconstructed)


if __name__ == "__main__":
    # Run all tests
    test_causal_vae_initialization()
    test_forward_pass_shapes()
    test_intervention()
    test_dag_properties()
    test_training_mode()
    print("All tests passed!")
