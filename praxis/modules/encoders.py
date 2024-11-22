import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class PraxisVAE(nn.Module):
    """
    A flexible Variational Autoencoder implementation with β-VAE support.
    Handles 3D inputs of shape [batch_size, seq_len, num_features].
    """

    def __init__(
        self,
        config: AutoConfig,
        input_dim: int,
        output_dim: int,
        beta: float = 1.0,
        requires_projection=False,
    ):
        super().__init__()

        self.debug = config.debug

        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.beta = beta

        # Calculate bottleneck dimension
        self.bottleneck_dim = min(input_dim, output_dim) // 2  # = 96
        self.latent_dim = self.bottleneck_dim // 2  # = 48

        class ScaledTanh(nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(scale))

            def forward(self, x):
                return torch.tanh(x / self.scale) * self.scale

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.bottleneck_dim),
            ScaledTanh(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            ScaledTanh(),
        )

        # Latent space parameters
        self.mu = nn.Linear(self.bottleneck_dim, self.latent_dim)
        self.var = nn.Linear(self.bottleneck_dim, self.latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.bottleneck_dim),
            ScaledTanh(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            ScaledTanh(),
            nn.Linear(self.bottleneck_dim, output_dim),
        )

        self.projection = False
        if requires_projection:
            self.projection = nn.Sequential(
                nn.Linear(output_dim, self.bottleneck_dim),
                ScaledTanh(),
                nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
                ScaledTanh(),
                nn.Linear(self.bottleneck_dim, input_dim),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE with added consistency loss.
        """
        # Encode
        mu, log_var = self.encode(x)

        # Sample from latent space
        z = self.reparameterize(mu, log_var)

        # Get compressed representation
        compressed = self.decode(z)

        # Calculate KL divergence
        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=2).mean()
        )

        consistency_loss = 0
        if self.projection:
            # Get projection back to input space for consistency loss
            projected = self.projection(compressed)

            # Calculate consistency loss (L1 loss between input and projection)
            consistency_loss = F.l1_loss(projected, x)

            if self.debug and random.random() < 0.001:
                with torch.no_grad():
                    debug_metrics = self.compute_reconstruction_quality_from_tensors(
                        x, projected
                    )
                    print(f"DEBUG: reconstruction: {str(debug_metrics)}")

        # Combine losses with weighting
        total_loss = self.beta * kl_loss + consistency_loss

        return compressed, total_loss

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

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
        compressed_input: bool = False,
        project_to_input: bool = False,
    ) -> torch.Tensor:

        batch_size, seq_len, feat_dim = z.shape
        expected_dim = self.output_dim if compressed_input else self.latent_dim

        # Flatten batch and sequence dimensions
        flat_z = z.reshape(-1, feat_dim)

        if compressed_input:
            # If input is already compressed, skip the decoder network
            decoded = flat_z
        else:
            # Normal decode from latent space
            decoded = self.decoder(flat_z)

        # Project to input dimension if requested
        if project_to_input:
            decoded = self.projection(decoded)

        # Reshape back to 3D
        out_dim = self.input_dim if project_to_input else self.output_dim
        output = decoded.reshape(batch_size, seq_len, out_dim)

        return output

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_reconstruction_quality_from_tensors(
        self, x: torch.Tensor, projected: torch.Tensor
    ) -> dict:
        """Compute reconstruction quality metrics from pre-computed tensors."""
        # Flatten tensors for correlation and cosine similarity
        x_flat = x.reshape(-1)
        proj_flat = projected.reshape(-1)

        # Compute basic metrics
        mse = F.mse_loss(projected, x).item()
        mae = F.l1_loss(projected, x).item()

        # Compute correlation coefficient
        corr = torch.corrcoef(torch.stack([x_flat, proj_flat]))[0, 1].item()

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            x_flat.unsqueeze(0), proj_flat.unsqueeze(0)
        ).item()

        return {
            "mse": f"{mse:.2f}",
            "mae": f"{mae:.2f}",
            "corr": f"{corr:.2f}",
            "cosine": f"{cos_sim:.2f}",
        }


class PraxisVQVAE(PraxisVAE):
    def __init__(
        self,
        config: AutoConfig,
        input_dim: int,
        output_dim: int,
        num_embeddings: int = 512,  # K in the paper
        decay: float = 0.99,
        eps: float = 1e-5,
        requires_projection: bool = False,
    ):
        super().__init__(
            config,
            input_dim,
            output_dim,
            beta=1.0,
            requires_projection=requires_projection,
        )

        # VQ-specific parameters
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.eps = eps

        # Create codebook as nn.Parameter
        self.codebook = nn.Parameter(torch.randn(self.latent_dim, num_embeddings))
        # Buffers for EMA updates
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", self.codebook.data.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        z_e = self.encode(x)

        # Reshape encoding for VQ
        batch_size, seq_len, _ = z_e.shape
        flatten = z_e.reshape(-1, self.latent_dim)

        # Calculate distances
        dist = (
            flatten.pow(2).sum(1, keepdim=True)  # [N, 1]
            - 2 * flatten @ self.codebook  # [N, K]
            + self.codebook.pow(2).sum(0, keepdim=True)  # [1, K]
        )

        # Find nearest embeddings
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings).type(flatten.dtype)

        # Update moving averages when training
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.t() @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.codebook.data.copy_(embed_normalized)

        # Quantize
        z_q = self.codebook[:, embed_ind].t()
        z_q = z_q.reshape(batch_size, seq_len, self.latent_dim)

        # Compute VQ losses
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # Decode
        compressed = self.decode(z_q)

        # Compute reconstruction loss in correct space
        if self.projection:
            projected = self.projection(compressed)
            reconstruction_loss = F.mse_loss(projected, x)
            consistency_loss = F.l1_loss(projected, x)
        else:
            # If dimensions match, compare to input, otherwise to encoded representation
            if self.input_dim == self.output_dim:
                reconstruction_loss = F.mse_loss(compressed, x)
            else:
                # Compare in latent space
                reconstruction_loss = F.mse_loss(z_q, z_e)
            consistency_loss = 0

        # Total loss
        vq_loss = (
            codebook_loss + commitment_loss + reconstruction_loss + consistency_loss
        )

        if self.debug and random.random() < 0.001 and self.projection:
            with torch.no_grad():
                debug_metrics = self.compute_reconstruction_quality_from_tensors(
                    x, projected
                )
                print(f"DEBUG: reconstruction: {str(debug_metrics)}")

        return compressed, vq_loss

    def vector_quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of vector quantization for test compatibility"""
        batch_size, seq_len, _ = z_e.shape
        flatten = z_e.reshape(-1, self.latent_dim)

        # Calculate distances
        dist = (
            flatten.pow(2).sum(1, keepdim=True)  # [N, 1]
            - 2 * flatten @ self.codebook  # [N, K]
            + self.codebook.pow(2).sum(0, keepdim=True)  # [1, K]
        )

        # Find nearest embeddings
        _, embed_ind = (-dist).max(1)
        z_q = self.codebook[:, embed_ind].t()
        z_q = z_q.reshape(batch_size, seq_len, self.latent_dim)

        # VQ Losses
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        loss = codebook_loss + commitment_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.shape

        # Flatten batch and sequence dimensions
        flat_x = x.reshape(-1, self.input_dim)

        # Encode
        encoded = self.encoder(flat_x)
        z_e = self.mu(encoded)

        # Reshape back to 3D
        z_e = z_e.reshape(batch_size, seq_len, self.latent_dim)
        return z_e

    def _quantize(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate quantization function that returns both regular and straight-through outputs"""
        batch_size, seq_len, feat_dim = z_e.shape

        # Flatten batch and sequence
        z_e_flat = z_e.reshape(-1, self.latent_dim)

        # Calculate distances
        d = (
            torch.sum(z_e_flat**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(z_e_flat, self.codebook.weight.t())
        )

        # Get indices of nearest embeddings
        min_encoding_indices = torch.argmin(d, dim=-1)

        # Get quantized vectors
        z_q_flat = self.codebook(min_encoding_indices)

        # Reshape to match input
        z_q = z_q_flat.reshape(batch_size, seq_len, self.latent_dim)

        # Straight-through gradient for decoder path
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q, z_q_st, min_encoding_indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Modified decoder that properly handles VQ output
        """
        batch_size, seq_len, _ = z_q.shape

        # Flatten batch and sequence dimensions
        flat_z = z_q.reshape(-1, self.latent_dim)

        # Always use decoder network for VQ-VAE (ignore compressed_input)
        decoded = self.decoder(flat_z)

        # Reshape back to 3D
        output = decoded.reshape(batch_size, seq_len, self.output_dim)

        return output


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    class DummyConfig:
        def __init__(self):
            self.debug = True

    config = DummyConfig()

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
        model = PraxisVAE(config, input_dim=10, output_dim=5, beta=2.0)
        assert isinstance(model, nn.Module), "Model should be a nn.Module"
        assert model.beta == 2.0, "Beta parameter not set correctly"

    # Test 2: Basic forward pass
    def test_forward_pass():
        model = PraxisVAE(config, input_dim=10, output_dim=5)
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
        model = PraxisVAE(config, input_dim=10, output_dim=5)
        x = torch.randn(16, 4, 10)
        reconstruction, _ = model(x)
        # assert torch.all(reconstruction >= 0) and torch.all(
        #     reconstruction <= 1
        # ), "Output should be in range [0, 1] due to sigmoid"

    # Test 4: Beta effect on KL loss
    def test_beta_effect():
        x = torch.randn(16, 4, 10)
        model1 = PraxisVAE(config, input_dim=10, output_dim=5, beta=1.0)
        model2 = PraxisVAE(config, input_dim=10, output_dim=5, beta=2.0)

        # Use the same weights for both models
        model2.load_state_dict(model1.state_dict())

        _, kl_loss1 = model1(x)
        _, kl_loss2 = model2(x)
        assert abs(kl_loss2 - 2 * kl_loss1) < 1e-5, "Beta should scale KL loss linearly"

    # Test 5: Shape preservation with equal dimensions
    def test_equal_dimensions():
        model = PraxisVAE(config, input_dim=10, output_dim=10)
        x = torch.randn(8, 6, 10)
        reconstruction, _ = model(x)
        assert (
            reconstruction.shape == x.shape
        ), "Shape should be preserved when input_dim == output_dim"

    # Test 6: Different sequence lengths
    def test_different_seq_lengths():
        model = PraxisVAE(config, input_dim=10, output_dim=5)
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

    def test_vqvae_model():
        # Test 1: VQ-VAE instantiation
        def test_vqvae_instantiation():
            model = PraxisVQVAE(config, input_dim=10, output_dim=5)
            assert isinstance(model, PraxisVAE), "Should inherit from PraxisVAE"
            assert hasattr(model, "codebook"), "Should have codebook"
            assert model.num_embeddings == 512, "Default codebook size should be 512"

        # Test 2: Vector quantization
        def test_vector_quantization():
            model = PraxisVQVAE(config, input_dim=10, output_dim=5)
            z_e = torch.randn(16, 4, model.latent_dim)
            z_q, loss = model.vector_quantize(z_e)
            assert z_q.shape == z_e.shape, "Quantized shape should match input"
            assert loss.dim() == 0, "VQ loss should be scalar"

        # Test 3: Forward pass
        def test_vqvae_forward():
            model = PraxisVQVAE(config, input_dim=10, output_dim=5)
            x = torch.randn(32, 8, 10)
            reconstruction, loss = model(x)
            assert reconstruction.shape == (32, 8, 5), "Wrong output shape"
            assert loss.dim() == 0, "Loss should be scalar"

        # Test 4: Codebook usage
        def test_codebook_usage():
            model = PraxisVQVAE(config, input_dim=10, output_dim=5)
            x = torch.randn(16, 4, 10)
            z_e = model.encode(x)  # [16, 4, latent_dim]
            z_q, _ = model.vector_quantize(z_e)

            # Flatten the quantized vectors before distance computation
            z_q_flat = z_q.reshape(-1, model.latent_dim)

            # Check that quantized vectors are in codebook
            d = (
                torch.sum(z_q_flat**2, dim=1, keepdim=True)  # [64, 1]
                - 2 * torch.matmul(z_q_flat, model.codebook)  # [64, 512]
                + model.codebook.pow(2).sum(0, keepdim=True)  # [1, 512]
            )
            min_distances = torch.min(d, dim=1)[0]
            assert torch.all(
                min_distances < 1e-5
            ), "Quantized vectors should be in codebook"

        def test_gradient_flow():
            # Create model with projection enabled
            model = PraxisVQVAE(
                config, input_dim=10, output_dim=5, requires_projection=True
            )
            optimizer = torch.optim.Adam(model.parameters())

            # Track initial weights for comparison
            initial_encoder_weight = model.encoder[0].weight.clone().detach()
            initial_decoder_weight = model.decoder[0].weight.clone().detach()
            initial_codebook_weight = model.codebook.clone().detach()

            # Training iterations
            for _ in range(5):  # Few iterations to ensure gradients flow
                x = torch.randn(16, 4, 10)
                optimizer.zero_grad()

                # Forward pass
                reconstruction, loss = model(x)
                loss.backward()
                optimizer.step()

            # Check if weights have been updated
            assert not torch.allclose(
                model.encoder[0].weight, initial_encoder_weight
            ), "Encoder weights should be updated"
            assert not torch.allclose(
                model.decoder[0].weight, initial_decoder_weight
            ), "Decoder weights should be updated"
            assert not torch.allclose(
                model.codebook, initial_codebook_weight
            ), "Codebook weights should be updated"

            # Verify gradients exist
            assert (
                model.encoder[0].weight.grad is not None
            ), "Encoder should have gradients"
            assert (
                model.decoder[0].weight.grad is not None
            ), "Decoder should have gradients"
            assert model.codebook.grad is not None, "Codebook should have gradients"

        tests = [
            ("VQ-VAE instantiation", test_vqvae_instantiation),
            ("Vector quantization", test_vector_quantization),
            ("VQ-VAE forward pass", test_vqvae_forward),
            ("Codebook usage", test_codebook_usage),
            ("VQ-VAE gradient flow", test_gradient_flow),
        ]

        print("Running VQ-VAE tests...")
        all_passed = all(run_smoke_test(desc, test_fn) for desc, test_fn in tests)
        print(
            f"\nVQ-VAE test summary: {'All tests passed!' if all_passed else 'Some tests failed.'}"
        )

    test_vqvae_model()
