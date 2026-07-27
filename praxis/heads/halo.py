"""HALO head: hyperspherical distance classifier (the HALO scoring function).

The official HALO reference (https://github.com/4rtemi5/halo) is not just a
loss - it is a model contract. ``HALOModel`` owns the class centroids,
mean-centers them every forward, and the model's *predictions are the
distance logits themselves*. Grafting the loss onto a model whose emitted
logits come from an unrelated scoring function trains geometry that
inference never reads.

This module restores that contract inside the standardized head stack:
``HaloClassifier`` owns the centroids, the learnable gamma temperature and
the abstain calibration, and emits the true distance logits
``-gamma * ||x - c||^2 / D`` (top-normalized to 0, matching the crystal
head's logit-scale contract for inference processors). ``HALOLoss``
delegates to this module when it finds it on the classifier path, so
training and inference score with the SAME function on the SAME features.

Placement rule: the HALO arm must read the trunk features directly (a
direct ParallelHead branch, or the model's head). Behind a feature
transform (e.g. Sequential(HarmonicField, HaloHead)) the loss - which
scores the trunk embeddings - would train a different feature space than
the arm scores at inference: the exact mismatch that scrambled the
borrowed-crystal wiring.

Calibration note: centers are initialized ``randn`` (unit per-coordinate
std) and inputs are RMS-normalized, so the official calibration assumption
``r_sq_init = 2.0`` holds *by construction* - gamma and the abstain bias
are exact at init instead of describing a state the model never occupies.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.heads.base import BaseHead
from praxis.losses.halo import HALO_LABEL_SMOOTHING


class HaloClassifier(nn.Module):
    """Owns the HALO geometry: centroids, gamma, abstain calibration.

    ``HALOLoss`` detects this module via ``is_halo`` and sources its
    centered centroids, softplus gamma and abstain bias from here, so
    there is exactly one scoring function shared by training and
    inference.
    """

    is_halo = True

    metric_descriptions = {
        "halo_centers_norm_mean": {
            "description": (
                "Mean L2 norm of the HALO centroids (before centering). "
                "Under HALO the frequency prior lives in centroid norms "
                "(prior ~ -gamma*||c||^2/D), so drift here reads how hard "
                "Zipf is bending the geometry."
            ),
            "chart": {
                "title": "HALO Center Norm (Mean)",
                "y_label": "Mean ||c_v||",
                "y_scale": "linear",
                "group": "halo",
                "order": 5,
            },
        },
        "halo_centers_norm_std": {
            "description": (
                "Std of per-centroid L2 norms. Rising = a few tokens "
                "stretching far from the cloud (usually rare tokens buying "
                "a very negative prior)."
            ),
            "chart": {
                "title": "HALO Center Norm (Std)",
                "y_label": "Std ||c_v||",
                "y_scale": "linear",
                "group": "halo",
                "order": 6,
            },
        },
    }

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        D = float(hidden_size)
        K = float(vocab_size)

        # Unit-std init matters: with RMS-normalized inputs it realizes the
        # official r_sq_init = 2.0 (E||x - c||^2/D = 1 + 1 at init), making
        # the gamma/abstain calibration below exact rather than aspirational.
        self.centers = nn.Parameter(torch.randn(vocab_size, hidden_size))

        # Official calibration (verbatim math from the reference loss).
        r_sq_target = 1.0 - (2.0 / D)
        r_sq_init = 2.0
        init_gamma = 20.0 / (r_sq_init - r_sq_target)

        ls = HALO_LABEL_SMOOTHING
        max_prob = 1.0 - ls + (ls / K)
        min_prob = ls / K
        margin_ce = math.log(max_prob / min_prob)
        t_ideal = init_gamma * (1.0 - r_sq_target)
        self.abstain_bias = float(t_ideal - margin_ce)

        if init_gamma > 20.0:
            gamma_start = init_gamma
        else:
            gamma_start = math.log(math.expm1(init_gamma))  # inverse softplus
        self.gamma = nn.Parameter(torch.tensor([gamma_start], dtype=torch.float32))

    def gamma_value(self) -> Tensor:
        return F.softplus(self.gamma)

    def centroids(self) -> Tensor:
        """Mean-centered centroids (official HALOModel behavior). Gradient
        flows through the centering, projecting the common-drift direction
        out of every update - the guard against the Zipfian frequency
        direction walking the cloud off the abstain origin."""
        c = self.centers.float()
        return c - c.mean(dim=0, keepdim=True)

    @staticmethod
    def normalize(x: Tensor) -> Tensor:
        """RMS-normalize to unit per-coordinate scale. The one normalization
        both the loss and inference apply - keep them identical."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6))

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        out_dtype = x.dtype
        # fp32 like crystal: the per-class spread rides on a common baseline
        # that low precision would quantize away.
        x_flat = self.normalize(x.reshape(-1, orig_shape[-1]).float())
        cen = self.centroids()
        D = float(self.hidden_size)

        gamma = self.gamma_value()
        y_sq = cen.pow(2).mean(dim=-1)
        dot = (x_flat @ cen.T) / D
        shifted = gamma * (2.0 * dot - y_sq.unsqueeze(0))
        # True distance logits: -gamma * ||x - c||^2 / D (x_sq == 1 after RMS
        # norm). Clamp against float error, exactly like the reference.
        true_logits = torch.clamp(shifted - gamma, max=0.0)
        true_logits = torch.nan_to_num(true_logits, nan=-1e9, neginf=-1e9)
        # Pin the top logit at 0 (shift-invariant for softmax/CE). Same
        # contract as crystal: sign-sensitive inference processors like
        # repetition_penalty break on a large negative offset.
        true_logits = true_logits - true_logits.amax(dim=-1, keepdim=True)
        return true_logits.view(*orig_shape[:-1], self.vocab_size).to(out_dtype)

    @torch.no_grad()
    def centers_norm_mean(self) -> Tensor:
        return self.centers.norm(dim=-1).mean()

    @torch.no_grad()
    def centers_norm_std(self) -> Tensor:
        return self.centers.norm(dim=-1).std()


class HaloHead(BaseHead):
    """LM head emitting HALO distance logits; the honest HALO arm.

    As a ParallelHead branch its logits are DETACHED in the gate blend
    (``detach_in_blend``): the mixture CE trains the gate's opinion of the
    arm (and the other arms), while the arm's own parameters train purely
    under HALOLoss's geometric objective. The gate share is then an
    uncontaminated verdict on whether HALO's scoring earns its keep.
    """

    # Centers must keep their unit-std init (the calibration ground truth);
    # tying them to the token embedding would re-anchor both geometries.
    self_ties = False

    # See class docstring; ParallelHead reads this in its terminal blend.
    detach_in_blend = True

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)
        if config.loss_func == "cut_cross_entropy":
            raise ValueError(
                "HaloHead is incompatible with loss_func='cut_cross_entropy' "
                "(cut-CE assumes a dot-product classifier)"
            )
        dims = self.output_dims()
        if dims is None:
            raise ValueError(
                "HaloHead needs an output layout; it can't pair with a "
                "loss-owning encoder (handles_loss)."
            )
        feature_dim, vocab_size = dims
        self.lm_head = HaloClassifier(hidden_size=feature_dim, vocab_size=vocab_size)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.lm_head(hidden_states)

    @property
    def classifier(self) -> nn.Module:
        return self.lm_head

    def compose_repr(self) -> str:
        return "HaloClassifier"

    def training_metrics(self) -> dict:
        c = self.lm_head
        return {
            "halo_centers_norm_mean": float(c.centers_norm_mean().item()),
            "halo_centers_norm_std": float(c.centers_norm_std().item()),
        }
