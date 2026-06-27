"""Crystal head: distance-based classifier (harmonic loss).

Replaces the standard ``W @ x`` logits with Euclidean-distance logits to
class centers. Probabilities follow ``p_i ∝ 1 / d_i^(2n)`` (i.e.
``(d_i²)^(-n)``) where ``d_i = ||c_i - x||_2`` and ``n`` is the harmonic
exponent - matching the grow-crystals ``DistLayer``, which raises ``d²``
to ``-n`` directly. The head returns ``pseudo_logits = -n * log(d²)``,
offset so the nearest center's logit is 0, then label-smoothed
(``prob + alpha/V``); ``softmax`` over them reproduces those probabilities,
so the standard CE pipeline consumes the output unchanged.

The output-layer weights become *class centers* (convex combinations of
training examples) rather than arbitrary direction vectors, giving the
"crystal" geometry the paper is named for. Weights stay bounded under
training because the minimum of ``-log p_target`` lives at finite norm.

Reference: Baek et al., "Harmonic Loss Trains Interpretable AI Models"
(arXiv:2502.01628). Naming follows the authors' "grow-crystals" repo.
"""

import math
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.heads.base import BaseHead

# Number of CrystalClassifiers in the VEAR-merged bank for the prismatic4 head.
# Baked (fixed, model-agnostic) per the tuning-free stance; override with a
# partial(CrystalVearHead, n_experts=...) in a head profile.
CRYSTAL_BANK_SIZE: int = 4

EPS: float = 1e-4

# Paper default for the mean-column-RMS centers regularizer (Baek et al.).
# The crystal head reads this when computing its centers_rms aux loss.
# Override per-experiment via YAML by setting ``embedding_rms_lambda``.
DEFAULT_EMBEDDING_RMS_LAMBDA: float = 0.01

# Label-smoothing weight: mixes the harmonic distribution with uniform
# (``prob + alpha/V``), matching the grow-crystals ``model_l2loss`` LM head.
# Caps the loss and curbs overconfidence. Override via ``crystal_label_smoothing``.
DEFAULT_LABEL_SMOOTHING: float = 0.01

# Resolution of the PCA density heatmap. 64 keeps the payload small
# (~16KB of ints) while resolving enough structure to read.
PCA_GRID_SIZE: int = 64


@torch.no_grad()
def _pca_density_grid(weights: list, grid_size: int = PCA_GRID_SIZE) -> dict:
    """Project stacked row vectors to 2D PCA, bin into a density grid.

    Rows are stacked so the PCA sees every input table at once (the
    crystal head passes its ``[V, D]`` centers). Uses ``svd_lowrank`` to
    grab only the top-2 PCs - cheap regardless of vocab size.
    """
    if not weights:
        return {}
    stacked = torch.cat([W.detach().to(torch.float32) for W in weights], dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    # svd_lowrank wants q <= min(rows, cols); 2 PCs is always safe.
    U, S, V = torch.svd_lowrank(centered, q=2)
    proj = centered @ V  # [N, 2]

    x_min, x_max = float(proj[:, 0].min()), float(proj[:, 0].max())
    y_min, y_max = float(proj[:, 1].min()), float(proj[:, 1].max())
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    xb = (
        ((proj[:, 0] - x_min) / x_span * (grid_size - 1))
        .long()
        .clamp_(0, grid_size - 1)
    )
    yb = (
        ((proj[:, 1] - y_min) / y_span * (grid_size - 1))
        .long()
        .clamp_(0, grid_size - 1)
    )
    flat = yb * grid_size + xb
    counts = torch.bincount(flat, minlength=grid_size * grid_size)
    grid = counts.view(grid_size, grid_size)

    n_rows = max(centered.shape[0] - 1, 1)
    total_var = float((centered.pow(2).sum() / n_rows).item())
    pc_vars = (S.pow(2) / n_rows).tolist() if total_var > 0 else [0.0, 0.0]
    var_explained = [v / total_var for v in pc_vars] if total_var > 0 else [0.0, 0.0]

    return {
        "grid": grid.cpu().tolist(),
        "grid_size": grid_size,
        "x_range": [x_min, x_max],
        "y_range": [y_min, y_max],
        "variance_explained": var_explained,
        "max_count": int(grid.max().item()),
        "n_points": int(stacked.shape[0]),
    }


class CrystalClassifier(nn.Module):
    """Distance-based classifier emitting pseudo-logits for CE."""

    metric_descriptions = {
        "crystal_centers_norm_mean": {
            "description": (
                "Mean L2 norm of vocabulary centers. Should plateau under "
                "harmonic loss (the paper's headline claim) rather than grow "
                "unboundedly the way standard CE weights do."
            ),
            "chart": {
                "title": "Center Norm (Mean)",
                "y_label": "Mean ||c_v||",
                "y_scale": "linear",
                "group": "crystal_head",
                "group_order": 50,
                "order": 10,
            },
        },
        "crystal_centers_norm_std": {
            "description": (
                "Std of per-center L2 norms. Falling = centers settling to a "
                "common scale; rising = a few centers stretching far from the "
                "rest, often a sign of over-confident outliers."
            ),
            "chart": {
                "title": "Center Norm (Std)",
                "y_label": "Std ||c_v||",
                "y_scale": "linear",
                "group": "crystal_head",
                "order": 20,
            },
        },
        "crystal_centers_grad_norm": {
            "description": (
                "L2 norm of the gradient on centers. Reads directly whether "
                "the model is still moving centers (learning) or has stalled."
            ),
            "chart": {
                "title": "Center Gradient Norm",
                "y_label": "||grad(centers)||",
                "y_scale": "logarithmic",
                "group": "crystal_head",
                "order": 30,
            },
        },
        "crystal_effective_dim": {
            "description": (
                "Number of PCA components needed to explain 90% of center "
                "variance. Low = centers form compact, low-dimensional "
                "geometry (the 'crystal' the paper looks for). Approaching "
                "feature_dim = no structure being learned."
            ),
            "chart": {
                "title": "Center Effective Dimension",
                "y_label": "# PCs for 90% variance",
                "y_scale": "linear",
                "group": "crystal_head",
                "order": 40,
            },
        },
        "crystal_centers_pca": {
            "description": (
                "Top-2 PCA projection of the vocabulary centers, binned to "
                "a density grid. The paper's 'crystal' view: as harmonic "
                "loss pulls centers into class prototypes this should "
                "develop structure (clusters, bands) rather than staying an "
                "isotropic blob."
            ),
            "snapshot": {
                "title": "Center PCA Density",
                "renderer": "heatmap_2d",
                "color_scale": "log",
                "group": "crystal_head",
                "order": 100,
            },
        },
    }

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        n: float,
        eps: float = EPS,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n = float(n)
        self.eps = float(eps)
        self.label_smoothing = float(label_smoothing)
        self.centers = nn.Parameter(torch.empty(vocab_size, hidden_size))
        # std = 1/sqrt(D), matching the grow-crystals tied-embedding init
        # (`std=1/np.sqrt(embd_dim)`). Centers inflate toward the feature
        # scale through the harmonic gradient during training.
        nn.init.normal_(self.centers, mean=0.0, std=1.0 / math.sqrt(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        out_dtype = x.dtype
        # Distance math in fp32: the per-class spread rides on a large
        # ~||x||^2 baseline, which low precision would quantize away.
        x_flat = x.reshape(-1, orig_shape[-1]).float()
        centers = self.centers.float()
        cc = (centers * centers).sum(-1)
        xx = (x_flat * x_flat).sum(-1, keepdim=True)
        cx = x_flat @ centers.T
        dist_sq = (cc.unsqueeze(0) + xx - 2.0 * cx).clamp_min(self.eps)
        # A non-finite distance (upstream NaN/inf in x) would poison the
        # softmax; treat it as "far" so that class collapses to ~0 prob.
        dist_sq = torch.nan_to_num(dist_sq, nan=1e9, posinf=1e9)
        # Normalize by the nearest center: scale-invariance makes this a
        # no-op for softmax/CE, but it pins the top logit at 0 instead of a
        # large negative offset. That offset is invisible to training yet
        # breaks sign-sensitive inference processors like repetition_penalty
        # (it multiplies negatives, suppressing correct recurring tokens as
        # context grows). Matches the grow-crystals DistLayer.
        dist_sq = dist_sq / dist_sq.amin(dim=-1, keepdim=True)
        # p_i ∝ (d²)^(-n) = d^(-2n), matching the grow-crystals DistLayer
        # (`(dist_sq)**(-n)`). n applies to d², not d: this sharp exponent is
        # what drives the centers to organize - halving it stalls them.
        pseudo_logits = -self.n * torch.log(dist_sq)
        # Label smoothing (grow-crystals model_l2loss): mix the harmonic
        # distribution with uniform via ``prob + alpha/V``, then re-log.
        # log_softmax/softmax downstream renormalizes, so this stays a valid
        # logit tensor (top still ~0) for both CE and inference sampling.
        if self.label_smoothing > 0.0:
            prob = torch.softmax(pseudo_logits, dim=-1)
            prob = prob + self.label_smoothing / self.vocab_size
            pseudo_logits = torch.log(prob)
        return pseudo_logits.view(*orig_shape[:-1], self.vocab_size).to(out_dtype)

    @torch.no_grad()
    def centers_norm_mean(self) -> Tensor:
        return self.centers.norm(dim=-1).mean()

    @torch.no_grad()
    def centers_norm_std(self) -> Tensor:
        return self.centers.norm(dim=-1).std()

    @torch.no_grad()
    def effective_dim(self, threshold: float = 0.9) -> int:
        """PCs needed to capture `threshold` of center variance.

        We work via the D x D covariance matrix's eigenvalues rather
        than a full SVD of the V x D center matrix - same answer, much
        cheaper when V >> D (typical for LM vocabularies).
        """
        c = self.centers.detach()
        centered = c - c.mean(dim=0, keepdim=True)
        denom = max(centered.shape[0] - 1, 1)
        cov = (centered.t() @ centered) / denom
        eigvals = torch.linalg.eigvalsh(cov.float()).flip(0).clamp_min(0.0)
        total = eigvals.sum()
        if float(total) <= 0:
            return int(eigvals.numel())
        cumvar = torch.cumsum(eigvals, dim=0) / total
        hits = (cumvar >= threshold).nonzero(as_tuple=False)
        if hits.numel() == 0:
            return int(eigvals.numel())
        return int(hits[0].item()) + 1


class CrystalHead(BaseHead):
    """LM head with a distance-based classifier (harmonic loss).

    In encoder-attached mode the head sizes its centers to match the
    encoder's classifier (so the distance computation lives in the
    encoder's feature space) and replaces the encoder's dot-product
    projection at the loss boundary.
    """

    # Crystal shares its centers with the input embedding in tie_weights().
    self_ties = True

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)
        if config.loss_func == "cut_cross_entropy":
            raise ValueError(
                "head_type='crystal' is incompatible with "
                "loss_func='cut_cross_entropy' (cut-CE assumes a "
                "dot-product classifier)"
            )

        n_cfg = getattr(config, "crystal_n", None)
        # The reference lists pow_n in {1, sqrt(D), D} as a hyperparameter.
        # We default to sqrt(D): n=D collapsed the center PCA (too sharp,
        # winner-take-all), while sqrt(D) gives the spread "crystal"
        # structure. Override via crystal_n.
        smoothing = float(
            getattr(config, "crystal_label_smoothing", DEFAULT_LABEL_SMOOTHING) or 0.0
        )
        # Projects hidden states down to the centers' space before the
        # distance. Only needed for standard-mode tying, where the centers
        # live in embed_size (to share the token embedding) but hidden
        # states are hidden_size. Mirrors the TiedWeights head.
        self.pre_projection: Optional[nn.Module] = None
        dims = self.output_dims()
        if dims is None:
            raise ValueError(
                "head_type='crystal' needs an encoder that declares an output "
                "layout; it can't pair with a loss-owning encoder (handles_loss)."
            )
        feature_dim, vocab_size = dims
        if self.has_encoder:
            # Encoder emits features at feature_dim (== embed_size), so the
            # centers match and distances need no projection. Tying to the
            # local tok_emb is likewise projection-free.
            center_dim = feature_dim
        else:
            # Standard mode: tie -> centers in embed_size (share the token
            # embedding [V, embed_size]) with a hidden->embed projection;
            # else centers in hidden_size (== feature_dim here).
            tie = bool(getattr(config, "tie_word_embeddings", False))
            embed_size = getattr(config, "embed_size", self.hidden_size)
            center_dim = embed_size if tie else feature_dim
            if tie and embed_size != self.hidden_size:
                self.pre_projection = nn.Linear(
                    self.hidden_size, embed_size, bias=False
                )
        n = float(n_cfg) if n_cfg is not None else math.sqrt(center_dim)
        self.lm_head = CrystalClassifier(
            hidden_size=center_dim,
            vocab_size=vocab_size,
            n=n,
            label_smoothing=smoothing,
        )

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        if self.pre_projection is not None:
            hidden_states = self.pre_projection(hidden_states)
        return self.lm_head(hidden_states)

    @property
    def classifier(self) -> nn.Module:
        return self.lm_head

    def compose_repr(self) -> str:
        return "CrystalClassifier"

    def training_metrics(self) -> dict:
        c = self.lm_head
        out = {
            "crystal_centers_norm_mean": float(c.centers_norm_mean().item()),
            "crystal_centers_norm_std": float(c.centers_norm_std().item()),
            "crystal_effective_dim": int(c.effective_dim()),
        }
        grad = c.centers.grad
        if grad is not None:
            out["crystal_centers_grad_norm"] = float(grad.detach().norm().item())
        return out

    def dashboard_snapshots(self) -> dict:
        """Top-2 PCA density grid of the vocabulary centers.

        The paper's 'crystal' view: as harmonic loss pulls centers into
        class prototypes this heatmap should develop structure (peaks,
        bands, clusters) rather than staying an isotropic blob.
        """
        grid = _pca_density_grid([self.lm_head.centers])
        return {"crystal_centers_pca": grid} if grid else {}

    def aux_losses(self) -> dict:
        """Mean-column-RMS regularizer on the centers (Baek et al.).

        Penalizes ``mean(sqrt(mean(c**2, dim=0)))`` - per-column RMS
        averaged across columns - matching the exact formula in the
        ``grow-crystals`` reference (``src/utils/model.py``). There the
        regularized embedding *is* the unembedding (weight-tied), so the
        penalty lands on the centers. Our centers are untied, so we
        regularize them directly; the input embeddings play no role in
        the distance geometry.
        """
        lam = float(
            getattr(self.config, "embedding_rms_lambda", DEFAULT_EMBEDDING_RMS_LAMBDA)
            or 0.0
        )
        if lam <= 0.0:
            return {}
        c = self.lm_head.centers
        rms = c.pow(2).mean(dim=0).clamp_min(1e-12).sqrt().mean()
        return {"centers_rms": lam * rms}


class CrystalVearHead(BaseHead):
    """A bank of ``CrystalClassifier``s merged by a VEAR router.

    Where ``CrystalHead`` learns one center geometry, this learns ``n_experts``
    and lets VEAR pick a discrete, per-context blend: sharpened routing selects a
    near-single crystal per batch (not the smeared convex-hull average SMEAR would
    give), and VEAR's inter-expert repulsion keeps the geometries distinct - a
    "population" of output geometries. Drop-in for ``CrystalHead`` inside a
    prismatic arm (``prismatic4``). Reuses VEAR's merge machinery
    (``praxis/routers/vear.py``); see ``next/roadmap.md`` (geometry banks + voting).

    Honest limit inherited from SMEAR/VEAR: the merge reduces to ONE crystal per
    batch (``routing_probs.mean(dim=0)``), so every token in the batch shares the
    selected geometry. Per-token crystal selection is a future refinement.
    """

    self_ties = False  # a bank has no single tie target; keep it untied

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None,
                 n_experts: int = CRYSTAL_BANK_SIZE) -> None:
        super().__init__(config, encoder)
        if config.loss_func == "cut_cross_entropy":
            raise ValueError(
                "CrystalVearHead (prismatic4) is incompatible with "
                "loss_func='cut_cross_entropy' (cut-CE assumes a dot-product head)"
            )
        # Deferred import: routers/ -> heads/ would otherwise risk an import cycle.
        from praxis.routers.vear import VEAR, VEAR_REPULSION, VEAR_SHARPEN

        self._rep_scale = float(VEAR_REPULSION)
        self._sharpen = float(VEAR_SHARPEN)
        n_cfg = getattr(config, "crystal_n", None)
        smoothing = float(
            getattr(config, "crystal_label_smoothing", DEFAULT_LABEL_SMOOTHING) or 0.0
        )
        self.pre_projection: Optional[nn.Module] = None
        dims = self.output_dims()
        if dims is None:
            raise ValueError(
                "head_type='prismatic4' needs an encoder that declares an output "
                "layout; it can't pair with a loss-owning encoder (handles_loss)."
            )
        feature_dim, vocab_size = dims
        if self.has_encoder:
            center_dim = feature_dim
        else:
            tie = bool(getattr(config, "tie_word_embeddings", False))
            embed_size = getattr(config, "embed_size", self.hidden_size)
            center_dim = embed_size if tie else feature_dim
            if tie and embed_size != self.hidden_size:
                self.pre_projection = nn.Linear(self.hidden_size, embed_size, bias=False)
        n = float(n_cfg) if n_cfg is not None else math.sqrt(center_dim)
        self.n_experts = int(n_experts)
        experts = [
            CrystalClassifier(
                hidden_size=center_dim, vocab_size=vocab_size, n=n,
                label_smoothing=smoothing,
            )
            for _ in range(self.n_experts)
        ]
        # VEAR owns the router (Linear(center_dim, N) + norm) and the N experts.
        vcfg = SimpleNamespace(
            num_experts=self.n_experts, hidden_size=center_dim, expert_dropout=0.1
        )
        self.bank = VEAR(vcfg, experts=experts)

    def _route(self, hidden_states: Tensor) -> Tensor:
        """Per-sequence routing probs ``[B, N]`` (mirrors SMEAR's routing)."""
        v = self.bank
        if hidden_states.dim() >= 3:
            router_input = hidden_states.mean(dim=1)
        else:
            router_input = hidden_states.reshape(
                -1, hidden_states.shape[-1]
            ).mean(dim=0, keepdim=True)
        router_input = v.router_norm(router_input)
        weight = F.normalize(v.router.weight, dim=1)
        logits = F.linear(router_input, weight, v.router.bias)
        probs = torch.softmax(logits, dim=-1)
        if v.training and v.dropout_rate > 0:
            mask = torch.bernoulli(torch.ones_like(probs) * (1 - v.dropout_rate))
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return probs

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        if self.pre_projection is not None:
            hidden_states = self.pre_projection(hidden_states)
        probs = self._route(hidden_states)  # [B, N] (post-dropout)
        # VEAR discrete: sharpen, then batch-mean -> one merged center-set. Done
        # with plain tensor ops (no functional_call), which is faster and keeps
        # the head out of any functional_call/autograd edge cases.
        sharp = probs.pow(self._sharpen)
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ew = sharp.mean(dim=0)  # [N]
        experts = self.bank.experts
        merged_centers = sum(ew[i] * experts[i].centers for i in range(len(experts)))
        # One merged crystal applied to every token (CrystalClassifier distance
        # math inlined; identical to functional_call'ing experts[0] with merged).
        return self._crystal_logits(hidden_states, merged_centers, experts[0])

    def _crystal_logits(self, x: Tensor, centers: Tensor, ref: nn.Module) -> Tensor:
        """CrystalClassifier.forward, but with externally-merged ``centers`` (the
        only param) instead of ``ref.centers`` - avoids functional_call."""
        orig_shape = x.shape
        out_dtype = x.dtype
        x_flat = x.reshape(-1, orig_shape[-1]).float()
        c = centers.float()
        cc = (c * c).sum(-1)
        xx = (x_flat * x_flat).sum(-1, keepdim=True)
        cx = x_flat @ c.T
        dist_sq = (cc.unsqueeze(0) + xx - 2.0 * cx).clamp_min(ref.eps)
        dist_sq = torch.nan_to_num(dist_sq, nan=1e9, posinf=1e9)
        dist_sq = dist_sq / dist_sq.amin(dim=-1, keepdim=True)
        pseudo_logits = -ref.n * torch.log(dist_sq)
        if ref.label_smoothing > 0.0:
            prob = torch.softmax(pseudo_logits, dim=-1) + ref.label_smoothing / ref.vocab_size
            pseudo_logits = torch.log(prob)
        return pseudo_logits.view(*orig_shape[:-1], ref.vocab_size).to(out_dtype)

    @property
    def classifier(self) -> nn.Module:
        return self.bank.experts[0]

    def compose_repr(self) -> str:
        return f"CrystalVearBank({self.n_experts})"

    def aux_losses(self) -> dict:
        out: dict = {}
        # Repulsion computed fresh here (parameter-only, collected post-forward
        # like centers_rms) - no stash, so nothing escapes the forward graph.
        if self.training and self.n_experts >= 2:
            out["crystal_bank_repulsion"] = (
                self._rep_scale * self.bank._inter_expert_repulsion()
            )
        lam = float(
            getattr(self.config, "embedding_rms_lambda", DEFAULT_EMBEDDING_RMS_LAMBDA)
            or 0.0
        )
        if lam > 0.0:
            rms = torch.stack(
                [
                    e.centers.pow(2).mean(dim=0).clamp_min(1e-12).sqrt().mean()
                    for e in self.bank.experts
                ]
            ).mean()
            out["centers_rms"] = lam * rms
        return out

    @torch.no_grad()
    def _bank_distinctness(self) -> float:
        """Mean pairwise L2 distance between the experts' center-sets - rises as
        VEAR's repulsion drives the geometries apart; ~0 = collapsed/redundant."""
        flat = torch.stack([e.centers.reshape(-1) for e in self.bank.experts], dim=0)
        n = flat.shape[0]
        if n < 2:
            return 0.0
        d = torch.cdist(flat, flat)  # [N, N], diagonal 0
        return float(d.sum().item() / (n * (n - 1)))

    def training_metrics(self) -> dict:
        experts = self.bank.experts
        return {
            "crystal_centers_norm_mean": float(
                torch.stack([e.centers_norm_mean() for e in experts]).mean().item()
            ),
            "crystal_centers_norm_std": float(
                torch.stack([e.centers_norm_std() for e in experts]).mean().item()
            ),
            "crystal_effective_dim": int(
                round(sum(e.effective_dim() for e in experts) / len(experts))
            ),
            # The direct readout of VEAR's goal: are the geometries actually unique?
            "crystal_bank_distinctness": self._bank_distinctness(),
        }

    def dashboard_snapshots(self) -> dict:
        # One PCA density per expert: distinct structure across experts = VEAR
        # producing unique geometries; identical clouds = the bank collapsed.
        out: dict = {}
        for k, e in enumerate(self.bank.experts):
            grid = _pca_density_grid([e.centers])
            if grid:
                out[f"crystal_centers_pca_{k}"] = grid
        return out

    def all_metric_descriptions(self) -> dict:
        # Start from the module-walk (the per-expert CrystalClassifiers contribute
        # the scalar descriptions), then swap the single-centers PCA for one card
        # per expert and drop the grad-norm (not tracked for the merged bank).
        out = dict(super().all_metric_descriptions())
        out.pop("crystal_centers_pca", None)
        out.pop("crystal_centers_grad_norm", None)
        for k in range(self.n_experts):
            out[f"crystal_centers_pca_{k}"] = {
                "description": (
                    f"Top-2 PCA density of crystal expert {k}'s vocabulary centers. "
                    "Distinct structure across experts means VEAR is producing unique "
                    "geometries; identical clouds mean the bank collapsed."
                ),
                "snapshot": {
                    "title": f"Center PCA Density (expert {k})",
                    "renderer": "heatmap_2d",
                    "color_scale": "log",
                    "group": "crystal_head",
                    "order": 100 + k,
                },
                "caller": "CrystalClassifier",
            }
        out["crystal_bank_distinctness"] = {
            "description": (
                "Mean pairwise L2 distance between the bank's expert center-sets. "
                "Rises as VEAR's repulsion drives the geometries apart; near 0 means "
                "collapsed / redundant experts."
            ),
            "chart": {
                "title": "Crystal Bank Distinctness",
                "y_label": "mean pairwise center dist",
                "y_scale": "linear",
                "group": "crystal_head",
                "order": 45,
            },
            "caller": "CrystalVearHead",
        }
        return out
