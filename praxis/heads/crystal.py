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

from praxis.heads.base import BaseHead, decode_context

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
def _pca_frame(weights: list) -> tuple:
    """Shared ``(mean, V, ranges)`` projection frame for a set of ``[V, D]``
    tables: the top-2 PCs over every row of every table, plus the extent of the
    projection, so panels drawn in this frame are directly comparable.

    The SVD is the deterministic full one, matching the paper figure's copy of
    this view (``praxis.pillars.geometries.pca_density_grid``, which moved off
    the randomized path first). ``torch.svd_lowrank`` draws from the global RNG:
    the same centers binned differently on every dashboard refresh, and the
    draws perturbed the training RNG stream of the run being watched. Economy
    SVD on a ``[V, D]`` table is cheap either way.
    """
    stacked = torch.cat([W.detach().to(torch.float32) for W in weights], dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    centered = stacked - mean
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    basis = Vh[:2].transpose(-2, -1)  # [D, 2]
    proj = centered @ basis
    ranges = (
        [float(proj[:, 0].min()), float(proj[:, 0].max())],
        [float(proj[:, 1].min()), float(proj[:, 1].max())],
    )
    return mean, basis, ranges


@torch.no_grad()
def _pca_density_grid(
    weights: list, grid_size: int = PCA_GRID_SIZE, frame: Optional[tuple] = None
) -> dict:
    """Project stacked row vectors to 2D PCA, bin into a density grid.

    Rows are stacked so the PCA sees every input table at once (the
    crystal head passes its ``[V, D]`` centers).

    ``frame`` is an optional :func:`_pca_frame` to draw in. Pass one when
    several panels are meant to be COMPARED - the bank's per-expert cards -
    so each lands in the same basis and the same axes: identical geometries
    then render identically, and a deviation reads as displacement instead of
    as a re-fit of the projection. Omitted, the table fits its own frame,
    which is the right thing for a lone card.
    """
    if not weights:
        return {}
    stacked = torch.cat([W.detach().to(torch.float32) for W in weights], dim=0)
    mean, basis, ranges = frame if frame is not None else _pca_frame(weights)
    centered = stacked - mean
    proj = centered @ basis  # [N, 2]

    (x_min, x_max), (y_min, y_max) = ranges
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
    # Equal to the singular values when this table defines the frame; in a
    # shared frame it is this panel's own spread along the shared axes.
    pc_vars = (
        (proj.pow(2).sum(dim=0) / n_rows).tolist() if total_var > 0 else [0.0, 0.0]
    )
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

    # Inference routes per position on the PREFIX mean; under cached decode the
    # running sum is carried across chunks (praxis.heads.base.decode_context).
    accepts_decode_cache: bool = True
    decode_cache: Any = None

    def __init__(
        self,
        config: Any,
        encoder: Optional[nn.Module] = None,
        n_experts: int = CRYSTAL_BANK_SIZE,
        sharpen: Optional[float] = None,
    ) -> None:
        """``sharpen`` is the routing exponent, and it is the ONLY thing that
        separates a VEAR bank from a SMEAR one here.

        ``probs.pow(s)`` renormalized: ``s = VEAR_SHARPEN`` (4.0, the default)
        drives routing toward a near-discrete pick, so one crystal's geometry
        dominates the merge. ``s = 1.0`` leaves the softmax untouched, which is
        exactly SMEAR - every expert contributes in proportion to its routing
        probability and every expert therefore receives gradient on every step.

        The trade is real in both directions. Sharpening keeps the merged
        centers close to ONE trained geometry; a soft blend is a convex
        combination of distinct center sets, and averaging points that live on a
        shell pulls the result toward the origin, which is a geometry no expert
        was trained to be. Softening pays for that with gradient to every
        expert instead of near-winner-take-all, which is what a bank needs if
        the experts are meant to specialize rather than compete. Repulsion is
        orthogonal and stays on in both modes.
        """
        super().__init__(config, encoder)
        if config.loss_func == "cut_cross_entropy":
            raise ValueError(
                "CrystalVearHead (prismatic4) is incompatible with "
                "loss_func='cut_cross_entropy' (cut-CE assumes a dot-product head)"
            )
        # Deferred import: routers/ -> heads/ would otherwise risk an import cycle.
        from praxis.routers.bank import (
            VEAR_REPULSION,
            VEAR_SHARPEN,
            SharpenedExpertBank,
        )

        self._rep_scale = float(VEAR_REPULSION)
        self._sharpen = float(VEAR_SHARPEN if sharpen is None else sharpen)
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
                self.pre_projection = nn.Linear(
                    self.hidden_size, embed_size, bias=False
                )
        n = float(n_cfg) if n_cfg is not None else math.sqrt(center_dim)
        self.n_experts = int(n_experts)
        experts = [
            CrystalClassifier(
                hidden_size=center_dim,
                vocab_size=vocab_size,
                n=n,
                label_smoothing=smoothing,
            )
            for _ in range(self.n_experts)
        ]
        # VEAR owns the router (Linear(center_dim, N) + norm) and the N experts.
        vcfg = SimpleNamespace(
            num_experts=self.n_experts, hidden_size=center_dim, expert_dropout=0.1
        )
        self.bank = SharpenedExpertBank(vcfg, experts=experts)

    def _route(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Per-sequence routing probs ``[B, N]`` (mirrors SMEAR's routing).

        ``mask`` (``[B, T]``, 1 = real) excludes padding from the sequence
        pooling: without it, padding shifts the mean and can flip the discrete
        crystal selection, so a padded batch routes differently from the same
        sequence unpadded - which breaks batched multi-token inference. Masked,
        the routing is padding-invariant.
        """
        v = self.bank
        if hidden_states.dim() >= 3:
            if mask is not None:
                m = mask.to(hidden_states.dtype).unsqueeze(-1)  # [B, T, 1]
                router_input = (hidden_states * m).sum(1) / m.sum(1).clamp_min(1.0)
            else:
                router_input = hidden_states.mean(dim=1)
        else:
            router_input = hidden_states.reshape(-1, hidden_states.shape[-1]).mean(
                dim=0, keepdim=True
            )
        router_input = v.router_norm(router_input)
        weight = F.normalize(v.router.weight, dim=1)
        logits = F.linear(router_input, weight, v.router.bias)
        probs = torch.softmax(logits, dim=-1)
        if v.training and v.dropout_rate > 0:
            dmask = torch.bernoulli(torch.ones_like(probs) * (1 - v.dropout_rate))
            probs = probs * dmask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return probs

    def _route_causal(
        self, hidden_states: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Per-POSITION routing probs ``[B, T, N]`` from the PREFIX mean.

        ``_route`` pools the whole sequence, so the crystal chosen for position
        ``t`` depends on bytes after ``t``. That is a future read in a causal LM,
        and it is the single thing that made speculative verification expensive:
        appending draft bytes shifted the pooled mean and moved every earlier
        logit, so ``_speculative_generate`` could not verify a block in one row
        and fell back to one full re-encode per candidate.

        The cumulative mean removes it. At position ``t`` this is the mean over
        ``0..t``, which is exactly the sequence mean ``_route`` would compute for
        a prefix ending at ``t`` - so reading position ``t`` of a long row equals
        running that prefix alone. At the LAST real position the two agree
        identically, so single-step greedy decoding is bit-for-bit unchanged;
        only the earlier positions of a multi-position read move, and those were
        the contaminated ones.
        """
        v = self.bank
        x = hidden_states
        if mask is not None:
            m = mask.to(x.dtype).unsqueeze(-1)  # [B, T, 1]
            total = (x * m).cumsum(1)
            count = m.cumsum(1)
        else:
            total = x.cumsum(1)
            count = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype)
            count = count.view(1, -1, 1).expand(x.size(0), -1, 1)
        # Cached decode: the chunk is a suffix, so the prefix mean continues
        # from the carried running sum - otherwise a one-token step routes on
        # itself alone and disagrees with the full-sequence read.
        _, state, commit = decode_context(self, hidden_states)
        if state is not None and "prefix_sum" in state:
            total = total + state["prefix_sum"].to(x.device, x.dtype).unsqueeze(1)
            count = count + state["prefix_count"].to(x.device, x.dtype).unsqueeze(1)
        if commit is not None:
            commit(
                {
                    "prefix_sum": total[:, -1].detach(),
                    "prefix_count": count[:, -1].detach(),
                }
            )
        router_input = total / count.clamp_min(1.0)
        router_input = v.router_norm(router_input)
        weight = F.normalize(v.router.weight, dim=1)
        logits = F.linear(router_input, weight, v.router.bias)
        return torch.softmax(logits, dim=-1)

    def _expert_centers(self) -> Tensor:
        """The bank's center sets, ``[N, V, D]``.

        Every read of the bank's geometry goes through here so a subclass can
        change HOW the N sets are stored without touching the routing, the
        logits, or the diagnostics. ``CrystalSmearHead`` uses it to hold one
        base set plus low-rank deviations instead of N independent sets.
        """
        return torch.stack([e.centers for e in self.bank.experts], dim=0)

    def _training_logits(
        self, hidden_states: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        """Training merge: ONE crystal for the whole batch.

        ``sharp.mean(dim=0)`` is the honest limit named in the class docstring,
        and it is also the reason the routing cannot learn to be
        input-conditional: the loss reaches the coefficients only through the
        batch mean, so every example contributes the identical routing gradient
        and a constant router is the fixed point. ``CrystalSmearHead`` overrides
        exactly this method and nothing else.
        """
        probs = self._route(hidden_states, mask)  # [B, N] (post-dropout)
        sharp = probs.pow(self._sharpen)
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ew = sharp.mean(dim=0)
        merged = torch.einsum("n,nvd->vd", ew.to(self._expert_centers().dtype),
                              self._expert_centers())
        return self._crystal_logits(hidden_states, merged, self.bank.experts[0])

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        if self.pre_projection is not None:
            hidden_states = self.pre_projection(hidden_states)
        mask = kwargs.get("attention_mask", None)
        # Cached decode: generate() passes the mask for the WHOLE sequence while
        # the hidden states are only the new suffix. The suffix's own mask is
        # its trailing columns; a full-length mask would broadcast the routing
        # pool back out to the full length.
        if (
            mask is not None
            and hidden_states.dim() >= 3
            and mask.dim() == 2
            and mask.shape[1] != hidden_states.shape[1]
        ):
            mask = mask[:, -hidden_states.shape[1] :]
        experts = self.bank.experts
        if self.training:
            return self._training_logits(hidden_states, mask)
        # Inference routes PER POSITION on the prefix mean, so every position is
        # a causal read: a batched forward equals each sequence run alone (the
        # padding invariance batched decode needs) AND a long row equals each of
        # its prefixes run alone (the property single-row speculative
        # verification needs).
        if hidden_states.dim() < 3:
            probs = self._route(hidden_states, mask)
            sharp = probs.pow(self._sharpen)
            sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            stacked = self._expert_centers()  # [N, V, D]
            merged = torch.einsum("bn,nvd->bvd", sharp, stacked)  # [B, V, D]
            return self._crystal_logits_perseq(hidden_states, merged, experts[0])
        sharp = self._route_causal(hidden_states, mask).pow(self._sharpen)
        sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, T, N]
        return self._crystal_logits_perpos(hidden_states, sharp, experts[0])

    def _crystal_logits_perpos(
        self, x: Tensor, sharp: Tensor, ref: nn.Module
    ) -> Tensor:
        """``_crystal_logits`` with a per-POSITION center set.

        ``x`` ``[B, T, D]``, ``sharp`` ``[B, T, N]`` -> logits ``[B, T, V]``,
        without ever materializing the ``[B, T, V, D]`` merged centers. Both
        terms of ``||x - c||^2`` that depend on ``c`` are cheap in the mixing
        weights: the cross term is linear (accumulate one expert at a time), and
        the center norm is the quadratic form ``s^T G s`` over the precomputed
        Gram ``G[n, m, v] = C_n[v] . C_m[v]``, which is only ``[N, N, V]``.
        """
        out_dtype = x.dtype
        xf = x.float()  # [B, T, D]
        stack0 = self._expert_centers().float()  # [N, V, D]
        centers = list(stack0.unbind(0))
        s = sharp.float()  # [B, T, N]
        xx = (xf * xf).sum(-1, keepdim=True)  # [B, T, 1]
        cx = None  # sum_n s_n (x . C_n)
        for n, c in enumerate(centers):
            term = s[..., n : n + 1] * (xf @ c.T)  # [B, T, V]
            cx = term if cx is None else cx + term
        gram = torch.einsum("nvd,mvd->nmv", stack0, stack0)  # [N, N, V]
        ss = (s.unsqueeze(-1) * s.unsqueeze(-2)).flatten(-2)  # [B, T, N*N]
        cc = ss @ gram.flatten(0, 1)  # [B, T, V]
        dist_sq = (cc + xx - 2.0 * cx).clamp_min(ref.eps)
        dist_sq = torch.nan_to_num(dist_sq, nan=1e9, posinf=1e9)
        dist_sq = dist_sq / dist_sq.amin(dim=-1, keepdim=True)
        pseudo_logits = -ref.n * torch.log(dist_sq)
        if ref.label_smoothing > 0.0:
            prob = (
                torch.softmax(pseudo_logits, dim=-1)
                + ref.label_smoothing / ref.vocab_size
            )
            pseudo_logits = torch.log(prob)
        return pseudo_logits.to(out_dtype)

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
            prob = (
                torch.softmax(pseudo_logits, dim=-1)
                + ref.label_smoothing / ref.vocab_size
            )
            pseudo_logits = torch.log(prob)
        return pseudo_logits.view(*orig_shape[:-1], ref.vocab_size).to(out_dtype)

    def _crystal_logits_perseq(
        self, x: Tensor, centers: Tensor, ref: nn.Module
    ) -> Tensor:
        """Like ``_crystal_logits`` but with a per-sequence center set:
        ``x`` ``[B, T, D]``, ``centers`` ``[B, V, D]`` -> logits ``[B, T, V]``."""
        out_dtype = x.dtype
        xf = x.float()  # [B, T, D]
        c = centers.float()  # [B, V, D]
        xx = (xf * xf).sum(-1, keepdim=True)  # [B, T, 1]
        cc = (c * c).sum(-1).unsqueeze(1)  # [B, 1, V]
        cx = torch.einsum("btd,bvd->btv", xf, c)  # [B, T, V]
        dist_sq = (cc + xx - 2.0 * cx).clamp_min(ref.eps)
        dist_sq = torch.nan_to_num(dist_sq, nan=1e9, posinf=1e9)
        dist_sq = dist_sq / dist_sq.amin(dim=-1, keepdim=True)
        pseudo_logits = -ref.n * torch.log(dist_sq)
        if ref.label_smoothing > 0.0:
            prob = (
                torch.softmax(pseudo_logits, dim=-1)
                + ref.label_smoothing / ref.vocab_size
            )
            pseudo_logits = torch.log(prob)
        return pseudo_logits.to(out_dtype)

    @property
    def classifier(self) -> nn.Module:
        return self.bank.experts[0]

    def compose_repr(self) -> str:
        mode = "Smear" if self._sharpen == 1.0 else "Vear"
        return f"Crystal{mode}Bank({self.n_experts})"

    def aux_losses(self) -> dict:
        out: dict = {}
        # Repulsion computed fresh here (parameter-only, collected post-forward
        # like centers_rms) - no stash, so nothing escapes the forward graph.
        if self.training and self.n_experts >= 2 and self._rep_scale > 0:
            out["crystal_bank_repulsion"] = (
                self._rep_scale * self.bank._inter_expert_repulsion()
            )
        lam = float(
            getattr(self.config, "embedding_rms_lambda", DEFAULT_EMBEDDING_RMS_LAMBDA)
            or 0.0
        )
        if lam > 0.0:
            rms = (
                self._expert_centers()
                .pow(2)
                .mean(dim=1)
                .clamp_min(1e-12)
                .sqrt()
                .mean()
            )
            out["centers_rms"] = lam * rms
        return out




    @torch.no_grad()
    def _bank_distinctness(self) -> float:
        """Mean pairwise L2 distance between the experts' center-sets - rises as
        VEAR's repulsion drives the geometries apart; ~0 = collapsed/redundant."""
        flat = self._expert_centers().reshape(self.n_experts, -1)
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

    @torch.no_grad()
    def dashboard_snapshots(self) -> dict:
        """One PCA density per expert: distinct structure across experts = the
        bank producing unique geometries; identical clouds = it collapsed.

        Reads ``_expert_centers()`` rather than looping over ``bank.experts``,
        so a subclass that stores the bank differently still gets one card per
        EXPERT. ``CrystalSmearHead`` keeps a single shared trunk module plus
        rank-r deviations, and the old loop emitted card 0 and left the other
        ``n_experts - 1`` - which ``all_metric_descriptions`` declares - blank.

        Every panel is drawn in one shared frame, since the whole point of the
        set is comparing them against each other. The frame spans every
        expert's rows, so one expert moving redraws all four panels together -
        the price of a common axis, and far less drift than the per-panel
        re-fit this replaced.
        """
        tables = list(self._expert_centers())
        if not tables:
            return {}
        frame = _pca_frame(tables)
        out: dict = {}
        for k, centers in enumerate(tables):
            grid = _pca_density_grid([centers], frame=frame)
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


class CrystalSmearHead(CrystalVearHead):
    """prismatic7's bank: SMEAR's mechanism where prismatic6 has the batch mean.

    ``CrystalVearHead`` at ``sharpen=1.0`` already blends rather than votes, so
    prismatic6 is SMEAR in that one respect and nothing else. Two things it does
    not do, and both are the paper's (arxiv 2306.03745), not new here:

    ROUTING IS PER EXAMPLE. The parent merges on ``sharp.mean(dim=0)`` - one
    crystal for the whole batch. That is not merely coarse: the loss reaches the
    coefficients only through the batch mean, so every example contributes the
    identical routing gradient ``dL/dw / B`` and a CONSTANT router is the
    design's fixed point rather than a training failure. It is the same defect
    the decoder's router had, where ``smear_input_dependence`` sat at ~0 through
    abstractinator-m/n/p. Here each example merges its own center set, which
    needs no approximation: ``_crystal_logits_perseq`` already consumes a
    ``[B, V, D]`` center stack, because inference has always routed per position.
    Training was the odd one out, and this also closes that train/inference gap.

    THE BANK IS BASE PLUS DEVIATIONS. The parent holds ``N`` independent
    ``[V, D]`` center sets. Here there is ONE set plus ``N`` rank-r deviations,
    which is the same merge written in a different basis
    (``base + sum_e w_e delta_e``) with two properties the parent lacks: it is
    EXACTLY the base at initialization (LoRA init, ``b`` zero), so swapping
    prismatic6 -> prismatic7 is a clean A/B rather than a reroll; and the shared
    trunk receives full gradient whatever the routing does, since
    ``d(merged)/d(base) = sum_e w_e = 1``. A starved deviation costs its rank
    instead of a whole geometry.

    Repulsion is OFF (``_rep_scale = 0``). It is VEAR's, not the paper's, and it
    exists to keep independent geometries apart; deviations off a shared base
    are not competing for the same role. Expert dropout stays at the bank's 0.1,
    which IS the paper's balancing mechanism.

    Averaging distinct center sets pulls the result toward the origin - the
    parent's argument for sharpening - but that argument applies to a convex hull
    of independently trained shells. A base plus zero-mean deviations has no such
    interior: the merge stays on the base's shell and the deviations perturb it.
    """

    def __init__(
        self,
        config: Any,
        encoder: Optional[nn.Module] = None,
        n_experts: int = CRYSTAL_BANK_SIZE,
    ) -> None:
        # sharpen=1.0: prismatic7 is SMEAR by construction, not by configuration.
        super().__init__(config, encoder, n_experts=n_experts, sharpen=1.0)
        from praxis.routers.smear import MIN_RANK, RANK_DIVISOR

        base = self.bank.experts[0]
        # The parent built N full center sets; keep one as the shared trunk (and
        # as the reference for n / eps / label_smoothing / vocab_size) and carry
        # the rest as deviations off it.
        self.bank.experts = nn.ModuleList([base])
        vocab_size, center_dim = base.centers.shape
        self.rank = max(MIN_RANK, min(vocab_size, center_dim) // RANK_DIVISOR)
        self.lora_a = nn.Parameter(torch.empty(n_experts, self.rank, center_dim))
        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
        self.lora_b = nn.Parameter(torch.zeros(n_experts, vocab_size, self.rank))
        self._rep_scale = 0.0

    def _expert_centers(self) -> Tensor:
        """``[N, V, D]``: the shared geometry plus each deviation.

        Exactly ``base`` for every expert at step 0, since ``lora_b`` is zero.
        """
        base = self.bank.experts[0].centers
        delta = torch.einsum("evr,erd->evd", self.lora_b, self.lora_a)
        return base.unsqueeze(0) + delta

    def _training_logits(
        self, hidden_states: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        """Per-EXAMPLE merge, which is the whole point of this subclass."""
        if hidden_states.dim() < 3:
            # No batch axis to route over; the parent's path is already correct.
            return super()._training_logits(hidden_states, mask)
        probs = self._route(hidden_states, mask)  # [B, N], padded-masked
        stack = self._expert_centers()  # [N, V, D]
        merged = torch.einsum("bn,nvd->bvd", probs.to(stack.dtype), stack)
        return self._crystal_logits_perseq(
            hidden_states, merged, self.bank.experts[0]
        )

    def all_metric_descriptions(self) -> dict:
        # The parent's per-expert wording ("identical clouds mean the bank
        # collapsed") reads the opposite way here: these are deviations off a
        # shared trunk with lora_b zeroed at init, so identical panels are the
        # designed starting point and separation is the thing to wait for.
        out = dict(super().all_metric_descriptions())
        for k in range(self.n_experts):
            key = f"crystal_centers_pca_{k}"
            if key in out:
                out[key] = {
                    **out[key],
                    "description": (
                        f"Top-2 PCA density of crystal expert {k}: the shared "
                        "center geometry plus this expert's rank-r deviation. "
                        "All panels are IDENTICAL at step 0 by construction "
                        "(LoRA init, lora_b zero), which is what makes "
                        "prismatic6 -> prismatic7 a clean A/B - identical "
                        "clouds are not collapse here. They separate only as "
                        "the deviations earn their rank; a panel still "
                        "indistinguishable from the others late in training is "
                        "a deviation the router never paid for."
                    ),
                }
        return out

    def compose_repr(self) -> str:
        return f"CrystalSmearBank({self.n_experts}, rank={self.rank})"
