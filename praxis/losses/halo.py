import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.reduction import weighted_reduce


class HALOLoss(nn.Module):
    """
    Hyperspherical Active Learning Objective (HALO) loss adapted for language modeling.
    https://github.com/4rtemi5/halo

    Instead of standard cross-entropy over logits, HALO operates in embedding space
    using distance-based scoring against centroid vectors. The classifier's weight
    matrix serves as the centroids - one per vocab token.

    Key components:
    - Gamma-scaled distance metric with learnable temperature
    - Abstain class acting as an origin sink at the theoretically ideal equilibrium
    - Geometric regularizer encouraging embeddings toward the hyperspherical shell
    - Distillation-based label smoothing using margin-aware soft targets
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        learn_gamma: bool = True,
        distill: bool = True,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.learn_gamma = learn_gamma
        self.distill = distill
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # D (embedding dim) is unknown until the first forward pass
        self._initialized = False
        self.gamma = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32),
            requires_grad=learn_gamma,
        )
        self.abstain_bias = 0.0
        self._last_stats = None

    def _lazy_init(self, emb_dims: int) -> None:
        """Initialize gamma and abstain bias once we know the embedding dimension."""
        D = float(emb_dims)
        K = float(self.vocab_size)
        r_sq_target = 1.0 - (2.0 / D)

        r_sq_init = 2.0
        init_gamma = 20.0 / (r_sq_init - r_sq_target)

        if self.label_smoothing > 0:
            max_prob = 1.0 - self.label_smoothing + (self.label_smoothing / K)
            min_prob = self.label_smoothing / K
        else:
            max_prob = 0.99
            min_prob = 0.01 / K

        margin_ce = math.log(max_prob / min_prob)
        t_ideal = init_gamma * (1.0 - r_sq_target)
        self.abstain_bias = t_ideal - margin_ce

        # Inverse softplus for gamma initialization
        if init_gamma > 20.0:
            gamma_start = init_gamma
        else:
            gamma_start = math.log(math.expm1(init_gamma))

        with torch.no_grad():
            self.gamma.fill_(gamma_start)

        self._D = D
        self._initialized = True

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        embeddings: Tensor = None,
        classifier: nn.Module = None,
        loss_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if embeddings is None or classifier is None:
            raise RuntimeError(
                "HALOLoss requires both `embeddings` and `classifier` kwargs. "
                "It cannot operate on logits alone."
            )

        # Caller already shifts logits/embeddings to [:-1] and labels to [1:],
        # so everything arrives aligned - just flatten.
        flat_labels = labels.view(-1)
        emb_dims = embeddings.shape[-1]

        if not self._initialized:
            self._lazy_init(emb_dims)

        flat_emb = embeddings.contiguous().view(-1, emb_dims)
        flat_weights = loss_weights.reshape(-1) if loss_weights is not None else None
        # HALO is a distance-to-centroid objective: a Linear head exposes its
        # centroids as ``weight``, the crystal head as ``centers``.
        centroids = getattr(classifier, "weight", None)
        if centroids is None:
            centroids = getattr(classifier, "centers", None)
        if centroids is None:
            raise RuntimeError(
                "HALOLoss needs a centroid matrix on the classifier "
                "(`weight` or `centers`)."
            )
        return self._halo_forward(flat_emb, flat_labels, centroids, flat_weights)

    def _halo_forward(
        self,
        pos: Tensor,
        target: Tensor,
        centroids: Tensor,
        loss_weights: Optional[Tensor] = None,
    ) -> Tensor:
        D = self._D
        pos = pos.to(torch.float32)
        cen = centroids.to(torch.float32)

        # Mask out padding tokens before computing HALO
        valid_mask = target != -100
        if not valid_mask.any():
            return pos.new_zeros((), requires_grad=True)

        pos = pos[valid_mask]
        target = target[valid_mask]
        # Filter weights to match the post-mask token set so per-token
        # alignment is preserved end-to-end.
        if loss_weights is not None:
            loss_weights = loss_weights[valid_mask]

        x_sq = pos.pow(2).mean(dim=-1, keepdim=True)
        y_sq = cen.pow(2).mean(dim=-1, keepdim=True)
        dot_product = (pos @ cen.T) / D

        gamma = F.softplus(self.gamma)

        # Softmax is shift-invariant, so we factor out -(x_sq * gamma).
        # This leaves standard dot-product similarity with an L2 penalty on keys.
        logits_k_shifted = gamma * (2.0 * dot_product - y_sq.T)

        # The abstain class acts as an origin sink
        logit_abstain_shifted = torch.full(
            (pos.size(0), 1), self.abstain_bias, dtype=pos.dtype, device=pos.device
        )

        # Shape: N x (K+1)
        logits_k_plus_1 = torch.cat([logits_k_shifted, logit_abstain_shifted], dim=-1)

        # True absolute distances for distillation and regularizer
        logits_k_true = torch.clamp(logits_k_shifted - (gamma * x_sq), max=0.0)

        # Cross-entropy on K+1 classes - keep per-token so we can weight.
        if self.distill:
            centroid_targets = torch.arange(
                cen.size(0), device=pos.device, dtype=target.dtype
            )
            mask = target.unsqueeze(1) == centroid_targets.unsqueeze(0)
            with torch.no_grad():
                margin = logits_k_true / self.label_smoothing
                target_logits = torch.where(mask, 0.0, margin)
                target_probs_k = F.softmax(target_logits, dim=-1)

                zeros = torch.zeros(
                    (pos.size(0), 1), device=pos.device, dtype=pos.dtype
                )
                target_probs = torch.cat([target_probs_k, zeros], dim=-1)

            loss_ce_per_token = F.cross_entropy(
                logits_k_plus_1, target_probs, reduction="none"
            )
        else:
            with torch.no_grad():
                K = logits_k_shifted.size(1)
                target_probs = torch.full_like(
                    logits_k_plus_1,
                    self.label_smoothing / K,
                    dtype=pos.dtype,
                    device=pos.device,
                )
                target_probs.scatter_(
                    1,
                    target.unsqueeze(1),
                    1.0 - self.label_smoothing + (self.label_smoothing / K),
                )
                target_probs[:, -1] = 0.0

            loss_ce_per_token = F.cross_entropy(
                logits_k_plus_1, target_probs, reduction="none"
            )

        # Geometric regularizer (per-token)
        diff_true = pos - cen[target]
        r_sq_true = diff_true.pow(2).mean(dim=-1).to(pos.dtype)

        volume_coeff = 0.5 - 1.0 / D
        volume_term = volume_coeff * torch.log(r_sq_true)
        gaussian_term = -0.5 * r_sq_true
        radial_nll = -(volume_term + gaussian_term)

        per_token = loss_ce_per_token + radial_nll

        self._stash_geometry(x_sq, logits_k_plus_1, gamma)

        # target was already filtered by valid_mask; pass labels=None so
        # weighted_reduce doesn't try to filter again.
        return weighted_reduce(
            per_token, labels=None, loss_weights=loss_weights, reduction=self.reduction
        )

    # ── Dashboard geometry ────────────────────────────────────────────────
    RING_BINS = 96

    @torch.no_grad()
    def _stash_geometry(self, x_sq: Tensor, logits_kp1: Tensor, gamma: Tensor) -> None:
        """Snapshot the radial energy of the batch for the HALO ring viz.

        Embeddings ideally settle on a shell of mean-square radius
        ``r_sq_target = 1 - 2/D``; the abstain class sinks the rest to the
        origin. We histogram per-token radii so the renderer can paint the
        bright ring of consensus and the dark interior/exterior of variance.
        """
        if torch.compiler.is_compiling():
            return
        radius = x_sq.detach().clamp_min(0).sqrt().view(-1).float()
        if radius.numel() == 0:
            return
        shell_r = math.sqrt(max(1.0 - 2.0 / self._D, 1e-6))
        r_max = max(float(radius.max().item()), shell_r * 1.6)
        hist = torch.histc(radius, bins=self.RING_BINS, min=0.0, max=r_max)
        peak = hist.max().clamp_min(1.0)
        abstain = F.softmax(logits_kp1.detach().float(), dim=-1)[:, -1].mean()
        self._last_stats = {
            "radii": (hist / peak).cpu().tolist(),
            "r_max": r_max,
            "shell_r": shell_r,
            "mean_radius": float(radius.mean().item()),
            "radius_spread": float(radius.std().item()) if radius.numel() > 1 else 0.0,
            "abstain_rate": float(abstain.item()),
            "gamma": float(gamma.detach().item()),
            "n": int(radius.numel()),
        }

    def training_metrics(self) -> dict:
        s = self._last_stats
        if not s:
            return {}
        return {
            "halo_gamma": s["gamma"],
            "halo_shell_radius": s["shell_r"],
            "halo_mean_radius": s["mean_radius"],
            "halo_radius_spread": s["radius_spread"],
            "halo_abstain_rate": s["abstain_rate"],
        }

    def dashboard_snapshots(self) -> dict:
        s = self._last_stats
        if not s:
            return {}
        return {
            "halo_ring": {
                "radii": s["radii"],
                "r_max": s["r_max"],
                "shell_r": s["shell_r"],
                "n": s["n"],
            }
        }

    metric_descriptions = {
        "halo_gamma": {
            "description": "Learnable inverse-temperature scaling the distance metric. Higher gamma = sharper centroid assignment.",
            "chart": {"title": "HALO Gamma", "group": "halo", "order": 0},
        },
        "halo_mean_radius": {
            "description": "Mean embedding radius (mean-square scale). HALO pulls this toward the shell radius sqrt(1 - 2/D).",
            "chart": {"title": "HALO Radius", "group": "halo", "order": 1, "series_group": "halo_radius", "series_label": "mean"},
        },
        "halo_shell_radius": {
            "description": "Target shell radius sqrt(1 - 2/D): the ring of consensus the objective drives embeddings onto.",
            "chart": {"title": "HALO Radius", "group": "halo", "order": 2, "series_group": "halo_radius", "series_label": "shell"},
        },
        "halo_radius_spread": {
            "description": "Std of embedding radii. Collapsing toward the shell tightens this; high spread is residual variance.",
            "chart": {"title": "HALO Radius Spread", "group": "halo", "order": 3},
        },
        "halo_abstain_rate": {
            "description": "Mean probability mass on the abstain class (the origin sink). High abstain = low-confidence tokens parked at the center.",
            "chart": {"title": "HALO Abstain Rate", "group": "halo", "order": 4},
        },
        "halo_ring": {
            "description": "Radial energy map: the bright ring marks consensus embeddings settled on the hyperspherical shell, with the dark interior (abstain sink) and exterior carrying no structure - a geometry of bias and variance.",
            "snapshot": {"title": "HALO Energy Ring", "renderer": "halo_ring", "order": 60},
        },
    }
