import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class ScaledTanh(nn.Module):
    """Scaled tanh activation for better gradient flow."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        return torch.tanh(x / self.scale) * self.scale


class DAGLayer(nn.Module):
    """Layer for learning and applying DAG structure."""

    def __init__(self, dim: int, inference: bool = False):
        super().__init__()
        self.dim = dim
        self.inference = inference
        # Initialize adjacency matrix with correct dimensions
        self.A = nn.Parameter(torch.zeros(dim, dim))
        if not inference:
            nn.init.uniform_(self.A, -0.1, 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Transform independent factors to causally related ones."""
        batch_size, seq_len, feat_dim = z.shape
        assert (
            feat_dim == self.dim
        ), f"Expected feature dimension {self.dim}, got {feat_dim}"

        # Ensure matrix is strictly upper triangular for DAG constraint
        A_mask = torch.triu(torch.ones_like(self.A), diagonal=1)
        A = self.A * A_mask

        # Reshape maintaining batch structure
        z_flat = z.reshape(-1, feat_dim)  # [batch_size * seq_len, feat_dim]

        # Create identity matrix matching feature dimension
        I = torch.eye(feat_dim, device=z.device)

        # Solve the system
        z_causal = torch.linalg.solve(I - A.T, z_flat.T).T

        # Reshape back to original dimensions
        return z_causal.reshape(batch_size, seq_len, feat_dim)

    def dag_constraint(self) -> torch.Tensor:
        """Calculate DAG constraint tr(e^(A ⊙ A)) - dim."""
        A_mask = torch.triu(torch.ones_like(self.A), diagonal=1)
        A = self.A * A_mask
        M = torch.matrix_exp(A * A)
        h = torch.trace(M) - self.dim
        return h


class MaskLayer(nn.Module):
    """Layer for enabling interventions in the causal structure."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mask = nn.Parameter(torch.ones(dim))

    def forward(
        self, z: torch.Tensor, intervention: Optional[dict] = None
    ) -> torch.Tensor:
        """Apply masking, optionally with interventions."""
        if intervention is not None:
            # Apply intervention by replacing values
            mask = self.mask.clone()
            for idx, value in intervention.items():
                z[:, :, idx] = value
            return z * mask.view(1, 1, -1)
        return z * self.mask.view(1, 1, -1)


class PraxisCausalVAE(nn.Module):
    """
    Causal Variational Autoencoder for sequence data.
    Handles 3D inputs of shape [batch_size, seq_len, num_features].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_concepts: int = 4,
        beta: float = 1.0,
        requires_projection: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.beta = beta
        self.num_concepts = num_concepts

        # Calculate dimensions with minimum sizes
        MIN_DIM = 32  # Ensure reasonable minimum dimension
        self.bottleneck_dim = max(MIN_DIM, min(input_dim, output_dim))
        self.latent_dim = self.bottleneck_dim

        # Ensure latent_dim is divisible by num_heads
        num_heads = 4
        self.latent_dim = ((self.latent_dim + num_heads - 1) // num_heads) * num_heads

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.bottleneck_dim),
            ScaledTanh(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            ScaledTanh(),
        )

        # Latent space parameters
        self.mu = nn.Linear(self.bottleneck_dim, self.latent_dim)
        self.var = nn.Linear(self.bottleneck_dim, self.latent_dim)

        # Causal structure
        self.dag_layer = DAGLayer(self.latent_dim)
        self.mask_layer = MaskLayer(self.latent_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim, num_heads=num_heads, batch_first=True
        )

        # Initialize conditional prior
        self.conditional_prior = nn.Linear(num_concepts, self.latent_dim * 2)

        # Calculate concept output dimensions
        if output_dim % num_concepts != 0:
            # If not evenly divisible, adjust concept dimensions
            base_concept_dim = output_dim // num_concepts
            extra_dims = output_dim % num_concepts
            concept_dims = [
                base_concept_dim + (1 if i < extra_dims else 0)
                for i in range(num_concepts)
            ]
        else:
            concept_dims = [output_dim // num_concepts] * num_concepts

        # Separable concept decoders
        self.concept_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.bottleneck_dim),
                    ScaledTanh(),
                    nn.Linear(self.bottleneck_dim, dim),
                )
                for dim in concept_dims
            ]
        )

        # Optional projection layer
        self.projection = False
        if requires_projection:
            self.projection = nn.Sequential(
                nn.Linear(output_dim, self.bottleneck_dim),
                ScaledTanh(),
                nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
                ScaledTanh(),
                nn.Linear(self.bottleneck_dim, input_dim),
            )

    def compute_conditional_prior(
        self, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional prior parameters p(z|u).
        Args:
            u: supervision tensor of shape [batch_size, seq_len, num_concepts]
        Returns:
            tuple of (mu, log_var) each of shape [batch_size, seq_len, latent_dim]
        """
        batch_size, seq_len, _ = u.shape

        # Flatten batch and sequence dimensions
        flat_u = u.reshape(-1, self.num_concepts)

        # Get parameters from linear layer
        params = self.conditional_prior(flat_u)  # [batch*seq, latent_dim*2]

        # Split into mu and log_var
        mu, log_var = params.chunk(2, dim=-1)

        # Reshape back to 3D
        mu = mu.reshape(batch_size, seq_len, self.latent_dim)
        log_var = log_var.reshape(batch_size, seq_len, self.latent_dim)

        return mu, log_var

    def decode_concepts(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Decode each concept separately."""
        concepts = []
        for decoder in self.concept_decoders:
            concept = decoder(z)
            concepts.append(concept)
        return concepts

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        batch_size, seq_len, feat_dim = x.shape

        # Flatten batch and sequence dimensions
        flat_x = x.reshape(-1, self.input_dim)

        # Encode
        encoded = self.encoder(flat_x)
        mu = self.mu(encoded)
        log_var = self.var(encoded)

        # Reshape back to 3D
        mu = mu.reshape(batch_size, seq_len, self.latent_dim)
        log_var = log_var.reshape(batch_size, seq_len, self.latent_dim)

        return mu, log_var

    def decode(
        self,
        z: torch.Tensor,
        intervention: Optional[dict] = None,
        compressed_input: bool = False,
        project_to_input: bool = False,
    ) -> torch.Tensor:
        """Decode latent representations, optionally with interventions."""
        batch_size, seq_len, feat_dim = z.shape

        # Apply causal structure and masking
        z = self.dag_layer(z)
        z = self.mask_layer(z, intervention)

        # Flatten batch and sequence dimensions
        flat_z = z.reshape(-1, feat_dim)

        if compressed_input:
            decoded = flat_z
        else:
            decoded = self.decoder(flat_z)

        if project_to_input:
            decoded = self.projection(decoded)

        # Reshape back to 3D
        out_dim = self.input_dim if project_to_input else self.output_dim
        output = decoded.reshape(batch_size, seq_len, out_dim)

        return output

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        intervention: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        mu, log_var = self.encode(x)

        # Sample latent
        z = self.reparameterize(mu, log_var)

        # Apply causal structure
        z = self.dag_layer(z)

        # Apply attention
        attn_output, _ = self.attention(z, z, z)
        z = z + attn_output

        # Apply mask/intervention
        z = self.mask_layer(z, intervention)

        # Decode concepts separately
        concepts = self.decode_concepts(z)
        compressed = torch.cat(concepts, dim=-1)

        # Calculate losses
        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=2).mean()
        )
        dag_loss = self.dag_layer.dag_constraint()

        # Add conditional prior loss if supervision provided
        prior_loss = 0
        if u is not None:
            prior_mu, prior_log_var = self.compute_conditional_prior(u)
            prior_loss = (
                -0.5
                * torch.sum(
                    1 + prior_log_var - prior_mu.pow(2) - prior_log_var.exp(), dim=2
                ).mean()
            )
        total_loss = self.beta * kl_loss + 0.1 * dag_loss + prior_loss

        return compressed, total_loss

    def compute_reconstruction_quality(
        self, x: torch.Tensor, projected: torch.Tensor
    ) -> dict:
        """Compute reconstruction quality metrics."""
        x_flat = x.reshape(-1)
        proj_flat = projected.reshape(-1)

        mse = F.mse_loss(projected, x).item()
        mae = F.l1_loss(projected, x).item()
        corr = torch.corrcoef(torch.stack([x_flat, proj_flat]))[0, 1].item()
        cos_sim = F.cosine_similarity(
            x_flat.unsqueeze(0), proj_flat.unsqueeze(0)
        ).item()

        return {
            "mse": f"{mse:.2f}",
            "mae": f"{mae:.2f}",
            "corr": f"{corr:.2f}",
            "cosine": f"{cos_sim:.2f}",
        }


# Tests
if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)

    def run_smoke_test(description: str, test_fn) -> bool:
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
        model = PraxisCausalVAE(
            input_dim=10, output_dim=8, num_concepts=4, beta=2.0, debug=True
        )
        assert isinstance(model, nn.Module)
        assert model.beta == 2.0
        assert model.num_concepts == 4
        assert model.latent_dim >= 32
        assert model.latent_dim % 4 == 0

    # Test 2: Basic forward pass
    def test_forward_pass():
        model = PraxisCausalVAE(input_dim=10, output_dim=8, num_concepts=4)
        x = torch.randn(32, 8, 10)
        reconstruction, loss = model(x)
        assert reconstruction.shape == (32, 8, 8)
        assert loss.dim() == 0

    # Test 3: Intervention
    def test_intervention():
        model = PraxisCausalVAE(input_dim=10, output_dim=8, num_concepts=4)
        x = torch.randn(16, 4, 10)  # [batch, seq, input_dim]
        u = torch.randn(16, 4, 4)  # [batch, seq, num_concepts]
        intervention = {0: torch.ones(16, 4)}
        reconstruction, _ = model(x, u, intervention)
        assert reconstruction.shape == (16, 4, 8)

    # Test 4: DAG constraint
    def test_dag_constraint():
        model = PraxisCausalVAE(input_dim=10, output_dim=8, num_concepts=4)
        x = torch.randn(16, 4, 10)
        _, loss = model(x)
        assert loss > 0  # DAG constraint should contribute to loss

    # Test 5: Shape preservation
    def test_shape_preservation():
        # Initialize model with same input/output dimensions
        model = PraxisCausalVAE(input_dim=10, output_dim=10, num_concepts=4)

        # Create random input
        x = torch.randn(8, 6, 10)

        # Get model output
        reconstruction, _ = model(x)

        # Shape tests
        assert (
            reconstruction.shape == x.shape
        ), f"Expected shape {x.shape}, got {reconstruction.shape}"

        # Verify decoder dimensions
        total_decoder_dims = sum(
            list(decoder.children())[-1].out_features
            for decoder in model.concept_decoders
        )
        assert (
            reconstruction.shape[-1] == total_decoder_dims
        ), f"Reconstruction dimension {reconstruction.shape[-1]} doesn't match total decoder dimensions {total_decoder_dims}"
        assert (
            total_decoder_dims == x.shape[-1]
        ), f"Total decoder dimensions {total_decoder_dims} doesn't match input dimension {x.shape[-1]}"

    # Test 6: Different sequence lengths
    def test_different_seq_lengths():
        model = PraxisCausalVAE(input_dim=10, output_dim=8, num_concepts=4)
        x1 = torch.randn(8, 6, 10)
        x2 = torch.randn(8, 12, 10)
        rec1, _ = model(x1)
        rec2, _ = model(x2)
        assert rec1.shape == (8, 6, 8) and rec2.shape == (8, 12, 8)

    # Run all tests
    tests = [
        ("Model instantiation", test_model_instantiation),
        ("Basic forward pass", test_forward_pass),
        ("Intervention test", test_intervention),
        ("DAG constraint test", test_dag_constraint),
        ("Shape preservation", test_shape_preservation),
        ("Different sequence lengths", test_different_seq_lengths),
    ]

    print("Running smoke tests...")
    all_passed = all(run_smoke_test(desc, test_fn) for desc, test_fn in tests)
    print(
        f"\nSmoke test summary: {'All tests passed!' if all_passed else 'Some tests failed.'}"
    )
