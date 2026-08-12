import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.reduction import weighted_reduce

# The one label-smoothing constant shared by the loss's distillation targets
# and the HaloClassifier's abstain calibration (praxis/heads/halo.py imports
# it). The two derivations must agree or the abstain equilibrium is computed
# for a target distribution the loss never produces.
HALO_LABEL_SMOOTHING: float = 0.1


class HALOLoss(nn.Module):
    """
    Hyperspherical Active Learning Objective (HALO) loss adapted for language modeling.
    https://github.com/4rtemi5/halo

    Instead of standard cross-entropy over logits, HALO operates in embedding
    space using distance-based scoring against centroid vectors.

    Key components:
    - Gamma-scaled distance metric with learnable temperature
    - Abstain class acting as an origin sink at the theoretically ideal equilibrium
    - Geometric regularizer encouraging embeddings toward the hyperspherical shell
    - Distillation-based label smoothing using margin-aware soft targets

    Two operating modes, keyed off the classifier:

    1. **Honest mode** (classifier exposes ``is_halo``, i.e. a
       :class:`~praxis.heads.halo.HaloClassifier` - the prismatic5 arm).
       The classifier owns centroids/gamma/abstain; the loss delegates to it
       so training and inference share ONE scoring function. The total loss
       is composite: standard CE on the model's emitted logits (training the
       gate and the non-HALO arms) PLUS the HALO geometric objective on the
       trunk embeddings. No relative weight knob: 1:1. Whether the CE term
       also reaches the HALO arm is set by ``HaloHead.detach_in_blend`` -
       detached, the geometric term is that arm's sole teacher and its gate
       share is an uncontaminated verdict on HALO's scoring; attached, the
       arm trains under both and the verdict is traded for the chance that
       the arm becomes useful.

    2. **Legacy side-loss mode** (any other centroid matrix: a Linear head's
       ``weight``, a crystal head's ``centers`` - e.g. CALM's geometric
       mode). Pure HALO on (embeddings, centroids), as before, with the
       official mean-centering applied when the centroids are trainable
       (frozen instruments like CALM's codec are measured as-is), and gamma
       calibrated from the MEASURED first-batch geometry instead of the
       official ``r_sq_init = 2.0`` guess (which assumes randn centroids -
       false for LM head inits like std 1/sqrt(D)).
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        learn_gamma: bool = True,
        distill: bool = True,
        label_smoothing: float = HALO_LABEL_SMOOTHING,
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

        self.gamma = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32),
            requires_grad=learn_gamma,
        )
        # Legacy-mode calibration state. Persistent buffers: on resume the
        # loss must NOT recalibrate - refilling gamma would clobber the
        # learned temperature, and a fresh abstain_bias would shift the
        # equilibrium mid-run.
        self.register_buffer(
            "_calibrated", torch.zeros((), dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "abstain_bias", torch.zeros((), dtype=torch.float32), persistent=True
        )
        self._D: Optional[float] = None
        self._last_stats = None

    def _calibrate(self, D: float, r_sq_init: float) -> None:
        """Initialize gamma and the abstain bias from the measured initial
        geometry (legacy mode only; the HaloClassifier calibrates itself).

        The official formula assumes ``r_sq_init = 2.0`` (randn centroids,
        unit-RMS embeddings). Here ``r_sq_init`` is the measured first-batch
        mean of ``||x - c_t||^2 / D``. The denominator is floored at the
        official value's (``2.0 - r_sq_target``), so gamma starts at most as
        sharp as the reference default and softer when the cloud starts
        farther out - it is learnable and can sharpen from there.
        """
        K = float(self.vocab_size)
        r_sq_target = 1.0 - (2.0 / D)

        denom = max(r_sq_init - r_sq_target, 2.0 - r_sq_target)
        init_gamma = 20.0 / denom

        if self.label_smoothing > 0:
            max_prob = 1.0 - self.label_smoothing + (self.label_smoothing / K)
            min_prob = self.label_smoothing / K
        else:
            max_prob = 0.99
            min_prob = 0.01 / K

        margin_ce = math.log(max_prob / min_prob)
        t_ideal = init_gamma * (1.0 - r_sq_target)

        # Inverse softplus for gamma initialization
        if init_gamma > 20.0:
            gamma_start = init_gamma
        else:
            gamma_start = math.log(math.expm1(init_gamma))

        with torch.no_grad():
            self.gamma.fill_(gamma_start)
            self.abstain_bias.fill_(t_ideal - margin_ce)
            self._calibrated.fill_(True)

    def forward(
        self,
        logits: Tensor = None,
        labels: Tensor = None,
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
        self._D = float(emb_dims)

        flat_emb = embeddings.contiguous().view(-1, emb_dims)
        flat_weights = loss_weights.reshape(-1) if loss_weights is not None else None

        is_halo_head = bool(getattr(classifier, "is_halo", False))

        # Normalize inputs to unit per-coordinate scale (RMS), matching the
        # shell target 1 - 2/D. HALO is hyperspherical: without this its r_sq,
        # dot products and loss scale with the upstream activation norm, so a
        # layer emitting large or drifting activations makes the loss magnitude
        # (and its gradient) track that norm - destabilizing training and any
        # loss-delta RL reward downstream. Normalizing decouples HALO from
        # upstream scale and is the geometry the objective assumes. The
        # HaloClassifier applies the IDENTICAL normalization at inference.
        pos = flat_emb.to(torch.float32)
        pos = pos * torch.rsqrt(pos.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-6))

        # Mask out padding tokens before computing HALO
        valid_mask = flat_labels != -100
        if not valid_mask.any():
            return pos.new_zeros((), requires_grad=True)

        pos = pos[valid_mask]
        target = flat_labels[valid_mask]
        # Filter weights to match the post-mask token set so per-token
        # alignment is preserved end-to-end.
        if flat_weights is not None:
            flat_weights = flat_weights[valid_mask]

        if is_halo_head:
            # Honest mode: the classifier owns the geometry. Its centroids()
            # are already mean-centered (official HALOModel behavior) and its
            # gamma/abstain were calibrated exactly at construction.
            cen = classifier.centroids()
            gamma = classifier.gamma_value()
            abstain_bias = float(classifier.abstain_bias)
        else:
            # Legacy mode: borrowed centroid matrix. A Linear head exposes its
            # centroids as ``weight``, the crystal head as ``centers``.
            centroids = getattr(classifier, "weight", None)
            if centroids is None:
                centroids = getattr(classifier, "centers", None)
            if centroids is None:
                raise RuntimeError(
                    "HALOLoss needs a centroid matrix on the classifier "
                    "(`weight` or `centers`)."
                )
            cen = centroids.to(torch.float32)
            # Official HALOModel mean-centers the centroids every forward:
            # under Zipfian targets the CE gradient pushes all centroids in a
            # correlated frequency direction, walking the cloud off the
            # abstain origin. Center only TRAINABLE centroids - a frozen
            # matrix (CALM's codec instrument) can't drift and must be
            # measured in its own geometry.
            if centroids.requires_grad:
                cen = cen - cen.mean(dim=0, keepdim=True)
            if not bool(self._calibrated.item()):
                with torch.no_grad():
                    r_sq_init = (pos - cen[target]).pow(2).mean(dim=-1).mean().item()
                self._calibrate(self._D, r_sq_init)
            gamma = F.softplus(self.gamma)
            abstain_bias = float(self.abstain_bias.item())

        halo_loss = self._halo_terms(
            pos, target, cen, gamma, abstain_bias, flat_weights
        )
        if not is_halo_head:
            return halo_loss

        # Honest-mode composite: standard CE on the model's emitted logits.
        # This is the gate's and the non-HALO arms' training signal. Whether it
        # ALSO reaches the HALO arm is the arm's own choice, not this loss's:
        # HaloHead.detach_in_blend decides, and the two settings ask different
        # questions (see that class). Detached (prismatic5) the geometric term
        # above is the arm's only teacher and its gate share is a clean verdict
        # on HALO's scoring; attached (prismatic6) the arm trains under both.
        # Either way the geometric term runs. Excludes -100 positions from the
        # denominator; per-token task weights pass through.
        flat_logits = logits.contiguous().view(-1, logits.shape[-1]).float()
        ce_per_token = F.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        ce_weights = (
            loss_weights.reshape(-1)
            if loss_weights is not None
            else torch.ones_like(ce_per_token)
        )
        ce_loss = weighted_reduce(
            ce_per_token, labels=flat_labels, loss_weights=ce_weights
        )
        return ce_loss + halo_loss

    def _halo_terms(
        self,
        pos: Tensor,
        target: Tensor,
        cen: Tensor,
        gamma: Tensor,
        abstain_bias: float,
        loss_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """The HALO objective proper, on pre-masked, RMS-normalized embeddings
        and mean-centered centroids, with gamma/abstain supplied by whichever
        module owns them (see ``forward``)."""
        D = self._D

        x_sq = pos.pow(2).mean(dim=-1, keepdim=True)
        y_sq = cen.pow(2).mean(dim=-1, keepdim=True)
        dot_product = (pos @ cen.T) / D

        # Softmax is shift-invariant, so we factor out -(x_sq * gamma).
        # This leaves standard dot-product similarity with an L2 penalty on keys.
        logits_k_shifted = gamma * (2.0 * dot_product - y_sq.T)

        # The abstain class acts as an origin sink
        logit_abstain_shifted = torch.full(
            (pos.size(0), 1), abstain_bias, dtype=pos.dtype, device=pos.device
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
        # Floor r_sq under the log barrier only. As r_sq -> 0 (an embedding
        # landing on its centroid, or a near-zero upstream activation) the
        # volume term's gradient -vc/r_sq diverges and detonates training -
        # the instability HALO is notorious for. Cap well below the shell
        # target (1 - 2/D) so the barrier is untouched in the healthy regime;
        # the gaussian term keeps the true r_sq (it has no singularity).
        r_sq_floor = 1e-2 * max(1.0 - 2.0 / D, 1e-6)
        volume_term = volume_coeff * torch.log(r_sq_true.clamp_min(r_sq_floor))
        gaussian_term = -0.5 * r_sq_true
        radial_nll = -(volume_term + gaussian_term)

        per_token = loss_ce_per_token + radial_nll

        self._stash_geometry(r_sq_true, logits_k_plus_1, gamma)

        # target was already filtered by valid_mask; pass labels=None so
        # weighted_reduce doesn't try to filter again.
        return weighted_reduce(
            per_token, labels=None, loss_weights=loss_weights, reduction=self.reduction
        )

    # ── Dashboard geometry ────────────────────────────────────────────────
    RING_BINS = 96

    @torch.no_grad()
    def _stash_geometry(
        self, r_sq_true: Tensor, logits_kp1: Tensor, gamma: Tensor
    ) -> None:
        """Snapshot the consensus geometry for the HALO ring viz.

        Inputs are RMS-normalized, so radius-from-origin is constant; the
        meaningful geometry is each token's distance to its centroid. The loss
        pulls that onto a shell of radius ``sqrt(1 - 2/D)``. We histogram those
        distances so the renderer paints the bright ring of consensus (tokens
        settled on the shell) with the dark interior (collapse) and exterior
        (variance) carrying no structure.
        """
        if torch.compiler.is_compiling():
            return
        radius = r_sq_true.detach().clamp_min(0).sqrt().view(-1).float()
        if radius.numel() == 0:
            return
        shell_r = math.sqrt(max(1.0 - 2.0 / self._D, 1e-6))
        r_max = max(float(radius.max().item()), shell_r * 2.0)
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
            "chart": {
                "title": "HALO Gamma",
                "group": "halo",
                "group_order": 70,
                "order": 0,
            },
        },
        "halo_mean_radius": {
            "description": "Mean distance from each token to its centroid (inputs are RMS-normalized). HALO pulls this onto the shell radius sqrt(1 - 2/D).",
            "chart": {
                "title": "HALO Radius",
                "group": "halo",
                "order": 1,
                "series_group": "halo_radius",
                "series_label": "mean",
            },
        },
        "halo_shell_radius": {
            "description": "Target shell radius sqrt(1 - 2/D): the ring of consensus the objective drives token-to-centroid distances onto.",
            "chart": {
                "title": "HALO Radius",
                "group": "halo",
                "order": 2,
                "series_group": "halo_radius",
                "series_label": "shell",
            },
        },
        "halo_radius_spread": {
            "description": "Std of token-to-centroid distances. Settling onto the shell tightens this; high spread is residual variance.",
            "chart": {"title": "HALO Radius Spread", "group": "halo", "order": 3},
        },
        "halo_abstain_rate": {
            "description": "Mean probability mass on the abstain class (the origin sink). High abstain = low-confidence tokens parked at the center.",
            "chart": {"title": "HALO Abstain Rate", "group": "halo", "order": 4},
        },
        "halo_ring": {
            "description": "Radial energy map of token-to-centroid distance: the bright ring marks consensus tokens settled on the hyperspherical shell, with the dark interior (collapse) and exterior (variance) carrying no structure - a geometry of bias and variance.",
            "snapshot": {
                "title": "HALO Energy Ring",
                "renderer": "halo_ring",
                "group": "halo",
                "order": 60,
            },
        },
    }
