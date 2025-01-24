from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLayer(nn.Module):
    def __init__(self, z_dim: int, concept: int = 4, z2_dim: int = 4):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept

        def create_network():
            return nn.Sequential(
                nn.Linear(z2_dim, 32),
                nn.ELU(),
                nn.Linear(32, z2_dim),
            )

        # Create individual networks for each concept
        self.net1 = create_network()
        self.net2 = create_network()
        self.net3 = create_network()
        self.net4 = create_network() if concept == 4 else None

    def mix(self, z: torch.Tensor) -> torch.Tensor:
        zy = z.view(-1, self.concept * self.z2_dim)

        if self.z2_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
            else:
                zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
        else:
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
            else:
                zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)

        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)

        if self.concept == 4:
            rx4 = self.net4(zy4)
            return torch.cat((rx1, rx2, rx3, rx4), dim=1)
        else:
            return torch.cat((rx1, rx2, rx3), dim=1)


def kl_normal(
    qm: torch.Tensor, qv: torch.Tensor, pm: torch.Tensor, pv: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two normal distributions

    Args:
        qm, qv: Parameters of q distribution (mean, variance)
        pm, pv: Parameters of p distribution (mean, variance)
    """
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    return element_wise.sum(-1)


def log_bernoulli_with_logits(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Compute log bernoulli probability."""
    return -F.binary_cross_entropy_with_logits(logits, x, reduction="none")


def condition_prior(
    scale: np.ndarray, label: torch.Tensor, z2_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate conditional prior based on labels

    Args:
        scale: Scale factors for each concept
        label: Label tensor [batch_size, seq_len, num_concepts]
        z2_dim: Dimension of each concept
    """
    batch_size, seq_len = label.shape[:2]
    num_concepts = label.shape[-1]

    # Reshape label to combine batch and sequence dims
    label_flat = label.view(-1, num_concepts)

    # Create conditional means based on labels
    means = []
    for i in range(num_concepts):
        # Calculate concept mean without extra dimensions
        concept_mean = (
            label_flat[:, i : i + 1] * scale[i, 0]
            + (1 - label_flat[:, i : i + 1]) * scale[i, 1]
        )
        # Expand to concept dimension
        concept_mean = concept_mean.unsqueeze(-1).expand(-1, 1, z2_dim)
        means.append(concept_mean)

    # Concatenate along concept dimension
    cp_m = torch.cat(means, dim=1)

    # Reshape back to include sequence dimension
    cp_m = cp_m.view(batch_size, seq_len, num_concepts, z2_dim)

    # Create matching variance tensor
    cp_v = torch.ones_like(cp_m)

    return cp_m, cp_v


@dataclass
class CausalVAEOutput:
    """Extended output container for CausalVAE showing ELBO components"""

    reconstructed: torch.Tensor
    elbo_loss: torch.Tensor
    reconstruction_term: torch.Tensor
    kl_divergence: torch.Tensor
    mask_loss: torch.Tensor
    z_masked: torch.Tensor


class Attention(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.M = nn.Parameter(
            torch.nn.init.normal_(torch.zeros(in_features, in_features), mean=0, std=1)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        a = self.sigmoid(a)
        A = torch.softmax(a, dim=1)
        e = torch.matmul(A, e)
        return e, A


class DecoderDAG(nn.Module):
    def __init__(self, z_dim, concept, z2_dim, output_dim, channel=4):
        super().__init__()
        self.z_dim = z_dim
        self.z2_dim = z2_dim
        self.concept = concept
        self.output_dim = output_dim

        # Create networks per concept - output logits
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(z2_dim, 300),
                    nn.ELU(),
                    nn.Linear(300, 300),
                    nn.ELU(),
                    nn.Linear(300, output_dim),
                    # No final activation - output logits for BCE loss
                )
                for _ in range(concept)
            ]
        )

    def decode_sep(self, z, u=None):
        z = z.view(-1, self.concept * self.z2_dim)
        if self.z2_dim == 1:
            z = z.reshape(z.size()[0], z.size()[1], 1)
            zs = [z[:, i] for i in range(self.concept)]
        else:
            zs = torch.split(z, self.z_dim // self.concept, dim=1)

        rxs = [net(zi) for net, zi in zip(self.nets, zs)]
        h = sum(rxs) / self.concept

        # Original returns 5 copies - maintaining interface
        return h, h, h, h, h


class DagLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, i=False, initial=True):
        super().__init__()
        self.input_dim = input_dim
        if initial:
            # Initialize as upper triangular matrix
            A = torch.zeros(input_dim, input_dim)
            self.A = nn.Parameter(A)

        # Create upper triangular mask
        self.mask = torch.triu(torch.ones(input_dim, input_dim), diagonal=1).bool()
        self.I = torch.eye(input_dim)

    def mask_z(self, z):
        return z

    def mask_u(self, u):
        return u

    def calculate_dag(self, z, v):
        """
        Calculate DAG transformation
        Args:
            z: input tensor [batch_size, num_concepts, concept_dim]
            v: variance tensor of same shape as z
        """
        # Apply mask to keep only upper triangular elements
        A_masked = self.A * self.mask.to(self.A.device)

        # Reshape z if needed
        batch_size = z.size(0)
        orig_shape = z.shape
        z = z.view(batch_size, self.input_dim, -1)

        # Calculate (I - A)^(-1)
        I_minus_A = self.I.to(self.A.device) - A_masked
        I_minus_A_inv = torch.linalg.inv(I_minus_A)

        # Apply transformation
        z_hat = torch.matmul(I_minus_A_inv, z)

        # Restore original shape
        z_hat = z_hat.view(orig_shape)

        return z_hat, v


class CausalVAELayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_concepts: int,
        concept_dim: int,
        alpha: float = 0.3,
        beta: float = 1.0,
    ):
        super().__init__()

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
        self.channel = 4  # Matching original code

        # Define scale matrix for conditional prior
        self.scale = np.array([[1, 0]] * num_concepts)

        # Core components matching original architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, latent_dim * 2),
        )

        self.decoder = DecoderDAG(
            latent_dim,
            num_concepts,
            concept_dim,
            input_dim,  # Pass input_dim as output_dim
            self.channel,
        )
        self.attention = Attention(concept_dim)
        self.dag = DagLayer(num_concepts, num_concepts)  # Add DAG layer
        self.mask_z = MaskLayer(latent_dim, num_concepts, concept_dim)
        self.mask_u = MaskLayer(num_concepts, num_concepts, 1)

    def compute_elbo(
        self,
        x: torch.Tensor,
        x_recon_logits: torch.Tensor,  # Now explicitly logits
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z_masked: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Evidence Lower BOund (ELBO) and components"""
        batch_size, seq_len = x.size(0), x.size(1)

        # Use logits-based BCE loss
        recon_term = -F.binary_cross_entropy_with_logits(
            x_recon_logits, x, reduction="mean"
        )

        # KL Divergence term
        kl_div = self.alpha * torch.mean(
            kl_normal(
                mu, torch.exp(logvar), torch.zeros_like(mu), torch.ones_like(logvar)
            )
        )

        # Additional terms for conditional prior
        if labels is not None:
            cp_m, cp_v = condition_prior(self.scale, labels, self.concept_dim)
            cp_m, cp_v = cp_m.to(x.device), cp_v.to(x.device)

            # Reshape z_masked to match conditional prior
            z_masked_reshaped = z_masked.view(
                batch_size, seq_len, self.num_concepts, self.concept_dim
            )
            z_masked_var = torch.ones_like(z_masked_reshaped)

            for i in range(self.num_concepts):
                kl_div = kl_div + self.beta * torch.mean(
                    kl_normal(
                        z_masked_reshaped[..., i, :],
                        z_masked_var[..., i, :],
                        cp_m[..., i, :],
                        cp_v[..., i, :],
                    )
                )

            mask_loss = F.mse_loss(
                self.mask_u.mix(labels.view(-1, self.num_concepts)),
                labels.view(-1, self.num_concepts).float(),
            )
        else:
            mask_loss = torch.tensor(0.0, device=x.device)

        elbo_loss = -recon_term + kl_div + mask_loss
        return elbo_loss, recon_term, kl_div, mask_loss

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> CausalVAEOutput:
        batch_size, seq_len = x.shape[:2]
        x_flat = x.view(-1, self.input_dim)

        # Encode and shape
        encoded = self.encoder(x_flat)
        mu, logvar = encoded.chunk(2, dim=-1)
        mu = mu.view(batch_size * seq_len, self.num_concepts, -1)
        logvar = logvar.view(batch_size * seq_len, self.num_concepts, -1)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Apply DAG transformation
        z_dag, _ = self.dag.calculate_dag(z, torch.ones_like(z))

        # Apply attention before mask
        z_attended, _ = self.attention.attention(z_dag, z_dag)

        # Apply mask (using only mix operation)
        z_masked = self.mask_z.mix(z_attended.reshape(-1, self.latent_dim))

        # Decode
        x_recon_logits, _, _, _, _ = self.decoder.decode_sep(z_masked)
        x_recon = x_recon_logits.view(batch_size, seq_len, -1)

        # Get probabilities
        reconstructed = torch.sigmoid(x_recon)

        # Reshape for ELBO
        mu = mu.view(-1, self.latent_dim)
        logvar = logvar.view(-1, self.latent_dim)

        elbo_loss, recon_term, kl_div, mask_loss = self.compute_elbo(
            x=x,
            x_recon_logits=x_recon,
            mu=mu,
            logvar=logvar,
            z_masked=z_masked,
            labels=labels,
        )

        return CausalVAEOutput(
            reconstructed=reconstructed,
            elbo_loss=elbo_loss,
            reconstruction_term=recon_term,
            kl_divergence=kl_div,
            mask_loss=mask_loss,
            z_masked=z_masked.view(batch_size, seq_len, -1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters (μ, log σ²)"""
        h = self.encoder(x)
        return h.chunk(2, dim=-1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstructed input with sigmoid"""
        return self.decoder(z)


def test_causal_vae():
    batch_size, seq_len = 32, 16
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

    # Create random binary input AND labels
    x = torch.randint(0, 2, (batch_size, seq_len, input_dim)).float()
    labels = torch.randint(0, 2, (batch_size, seq_len, num_concepts)).float()

    # Run with labels
    output = model(x, labels)

    # Basic shape checks
    assert output.reconstructed.shape == x.shape
    assert output.reconstructed.min() >= 0 and output.reconstructed.max() <= 1
    assert output.elbo_loss.ndim == 0  # Scalar loss

    # Compute gradients
    output.elbo_loss.backward()

    # Check gradients only for tensor parameters
    has_grads = []
    for name, param in model.named_parameters():
        if isinstance(param, torch.Tensor):
            has_grad = param.grad is not None and param.grad.abs().sum() > 0
            has_grads.append(has_grad)
            if not has_grad:
                print(f"No gradient for parameter: {name}")

    assert all(has_grads), "Some tensor parameters missing gradients"
    print("All tests passed!")


if __name__ == "__main__":
    test_causal_vae()
