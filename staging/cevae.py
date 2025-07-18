from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class PraxisCEVAE(nn.Module):
    """
    A Causal Effect Variational Autoencoder implementation based on PraxisVAE.
    Handles 3D inputs of shape [batch_size, seq_len, num_features].
    """

    def __init__(
        self,
        config: AutoConfig,
        input_dim: int,
        outcome_dim: int,
        beta: float = 1.0,
        hidden_dim: int = 20,
    ):
        super().__init__()

        self.debug = config.debug
        self.input_dim = input_dim
        self.outcome_dim = outcome_dim
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.latent_dim = hidden_dim  # Dimension of z

        class ScaledTanh(nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(scale))

            def forward(self, x):
                return torch.tanh(x / self.scale) * self.scale

        # Feature reconstruction network p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            ScaledTanh(),
            nn.Linear(hidden_dim, hidden_dim),
            ScaledTanh(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Treatment prediction network p(t|z)
        self.treatment_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            ScaledTanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Outcome prediction network p(y|z,t) - separate paths for t=0,1
        self.outcome_predictor_t0 = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            ScaledTanh(),
            nn.Linear(hidden_dim, outcome_dim),
        )
        self.outcome_predictor_t1 = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            ScaledTanh(),
            nn.Linear(hidden_dim, outcome_dim),
        )

        # Inference networks
        self.treatment_inference = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ScaledTanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Encoder q(z|x,t,y)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1 + outcome_dim, hidden_dim),
            ScaledTanh(),
            nn.Linear(hidden_dim, hidden_dim),
            ScaledTanh(),
        )

        self.z_mean = nn.Linear(hidden_dim, self.latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, self.latent_dim)

    def encode(
        self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode inputs to latent distribution parameters.
        """
        batch_size, seq_len, _ = x.shape

        # Flatten batch and sequence dimensions
        x_flat = x.reshape(-1, self.input_dim)
        t_flat = t.reshape(-1, 1)

        if y is None:
            y_flat = torch.zeros(x_flat.shape[0], self.outcome_dim, device=x.device)
        else:
            y_flat = y.reshape(-1, self.outcome_dim)

        # Concatenate inputs
        encoder_input = torch.cat([x_flat, t_flat, y_flat], dim=1)

        # Encode
        encoded = self.encoder(encoder_input)
        mu = self.z_mean(encoded)
        log_var = self.z_logvar(encoded)

        # Reshape back to 3D
        mu = mu.reshape(batch_size, seq_len, self.latent_dim)
        log_var = log_var.reshape(batch_size, seq_len, self.latent_dim)

        return mu, log_var

    def decode(
        self, z: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent samples to reconstructions and predictions.
        """
        batch_size, seq_len, _ = z.shape

        # Flatten batch and sequence dimensions
        z_flat = z.reshape(-1, self.latent_dim)

        # Reconstruct features
        x_recon = self.decoder(z_flat)

        # Predict treatment if not provided
        if t is None:
            t_flat = self.treatment_predictor(z_flat)
        else:
            t_flat = t.reshape(-1, 1)

        # Predict outcomes for both potential treatment values
        y_pred_t0 = self.outcome_predictor_t0(z_flat)
        y_pred_t1 = self.outcome_predictor_t1(z_flat)

        # Return factual outcome based on treatment
        y_pred = t_flat * y_pred_t1 + (1 - t_flat) * y_pred_t0

        # Reshape back to 3D
        x_recon = x_recon.reshape(batch_size, seq_len, self.input_dim)
        t_pred = t_flat.reshape(batch_size, seq_len, 1)
        y_pred = y_pred.reshape(batch_size, seq_len, self.outcome_dim)

        return x_recon, t_pred, y_pred

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var)."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_reconstruction_quality(
        self, x: torch.Tensor, x_recon: torch.Tensor
    ) -> dict:
        """Compute reconstruction quality metrics."""
        # Flatten tensors for correlation and cosine similarity
        x_flat = x.reshape(-1)
        recon_flat = x_recon.reshape(-1)

        # Compute basic metrics
        mse = F.mse_loss(x_recon, x).item()
        mae = F.l1_loss(x_recon, x).item()

        # Compute correlation coefficient
        corr = torch.corrcoef(torch.stack([x_flat, recon_flat]))[0, 1].item()

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            x_flat.unsqueeze(0), recon_flat.unsqueeze(0)
        ).item()

        return {
            "mse": f"{mse:.2f}",
            "mae": f"{mae:.2f}",
            "corr": f"{corr:.2f}",
            "cosine": f"{cos_sim:.2f}",
        }

    def estimate_ate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate Average Treatment Effect (ATE) for given features.
        """
        # Encode features (without treatment/outcome)
        mu, log_var = self.encode(
            x, torch.zeros(x.shape[0], x.shape[1], 1, device=x.device), None
        )

        # Sample latent variables
        z = self.reparameterize(mu, log_var)

        # Get outcomes for both treatment values
        _, _, y_pred_t0 = self.decode(z, torch.zeros_like(mu[:, :, 0:1]))
        _, _, y_pred_t1 = self.decode(z, torch.ones_like(mu[:, :, 0:1]))

        # ATE is average difference between outcomes
        ate = (y_pred_t1 - y_pred_t0).mean()

        return ate

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing reconstructions and ELBO components.
        """
        # Encode
        mu, log_var = self.encode(x, t, y)

        # Sample latent variable
        z = self.reparameterize(mu, log_var)

        # Decode
        x_recon, t_pred, y_pred = self.decode(z, t)

        # Compute losses
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=2).mean()
        )
        treatment_loss = F.binary_cross_entropy(t_pred, t)

        if y is not None:
            outcome_loss = F.mse_loss(y_pred, y)
        else:
            outcome_loss = torch.tensor(0.0, device=x.device)

        total_loss = recon_loss + self.beta * kl_loss + treatment_loss + outcome_loss

        if self.debug and torch.rand(1).item() < 0.001:
            metrics = self.compute_reconstruction_quality(x, x_recon)
            print(f"DEBUG: reconstruction: {str(metrics)}")

        return x_recon, y_pred, t_pred, total_loss


def main():
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
        model = PraxisCEVAE(config, input_dim=10, outcome_dim=1, beta=2.0)
        assert isinstance(model, nn.Module), "Model should be a nn.Module"
        assert model.beta == 2.0, "Beta parameter not set correctly"

    # Test 2: Basic forward pass
    def test_forward_pass():
        model = PraxisCEVAE(config, input_dim=10, outcome_dim=1)
        x = torch.randn(32, 8, 10)  # [batch_size, seq_len, input_dim]
        t = torch.randint(0, 2, (32, 8, 1)).float()  # Binary treatment
        y = torch.randn(32, 8, 1)  # Continuous outcome

        x_recon, y_pred, t_pred, loss = model(x, t, y)

        assert x_recon.shape == x.shape, f"Wrong x_recon shape: {x_recon.shape}"
        assert y_pred.shape == y.shape, f"Wrong y_pred shape: {y_pred.shape}"
        assert t_pred.shape == t.shape, f"Wrong t_pred shape: {t_pred.shape}"
        assert loss.dim() == 0, "Loss should be a scalar"

    # Test 3: Counterfactual prediction
    def test_counterfactual():
        model = PraxisCEVAE(config, input_dim=10, outcome_dim=1)
        x = torch.randn(16, 4, 10)
        t = torch.randint(0, 2, (16, 4, 1)).float()

        # Forward pass without outcomes
        x_recon, y_pred, t_pred, loss = model(x, t)

        assert x_recon.shape == x.shape, "Wrong reconstruction shape"
        assert y_pred.shape == (16, 4, 1), "Wrong outcome prediction shape"
        assert loss.dim() == 0, "Loss should be a scalar"

    # Test 4: Treatment effect estimation
    def test_treatment_effect():
        model = PraxisCEVAE(config, input_dim=10, outcome_dim=1)
        x = torch.randn(16, 4, 10)

        ate = model.estimate_ate(x)
        assert ate.dim() == 0, "ATE should be a scalar"
        assert not torch.isnan(ate), "ATE should not be NaN"

    # Test 5: Shape preservation with different sequence lengths
    def test_different_seq_lengths():
        model = PraxisCEVAE(config, input_dim=10, outcome_dim=1)
        x1 = torch.randn(8, 6, 10)
        t1 = torch.randint(0, 2, (8, 6, 1)).float()
        x2 = torch.randn(8, 12, 10)
        t2 = torch.randint(0, 2, (8, 12, 1)).float()

        x_recon1, y_pred1, t_pred1, _ = model(x1, t1)
        x_recon2, y_pred2, t_pred2, _ = model(x2, t2)

        assert (
            x_recon1.shape == x1.shape and x_recon2.shape == x2.shape
        ), "Model should handle different sequence lengths"

    # Test 6: Loss components
    def test_loss_components():
        model = PraxisCEVAE(config, input_dim=10, outcome_dim=1)
        x = torch.randn(16, 4, 10)
        t = torch.randint(0, 2, (16, 4, 1)).float()
        y = torch.randn(16, 4, 1)

        # Get encodings
        mu, log_var = model.encode(x, t, y)
        z = model.reparameterize(mu, log_var)
        x_recon, t_pred, y_pred = model.decode(z, t)

        # Verify loss components are computable
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=2).mean()
        )
        treatment_loss = F.binary_cross_entropy(t_pred, t)
        outcome_loss = F.mse_loss(y_pred, y)

        assert not torch.isnan(recon_loss), "Reconstruction loss is NaN"
        assert not torch.isnan(kl_loss), "KL loss is NaN"
        assert not torch.isnan(treatment_loss), "Treatment loss is NaN"
        assert not torch.isnan(outcome_loss), "Outcome loss is NaN"

        # Verify total loss computation
        total_loss = recon_loss + model.beta * kl_loss + treatment_loss + outcome_loss
        assert not torch.isnan(total_loss), "Total loss is NaN"

    # Run all tests
    tests = [
        ("Model instantiation", test_model_instantiation),
        ("Basic forward pass", test_forward_pass),
        ("Counterfactual prediction", test_counterfactual),
        ("Treatment effect estimation", test_treatment_effect),
        ("Different sequence lengths", test_different_seq_lengths),
        ("Loss components", test_loss_components),
    ]

    print("Running smoke tests...")
    all_passed = all(run_smoke_test(desc, test_fn) for desc, test_fn in tests)
    print(
        f"\nSmoke test summary: {'All tests passed!' if all_passed else 'Some tests failed.'}"
    )


if __name__ == "__main__":
    main()
