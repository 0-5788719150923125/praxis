"""Crystal head: distance-based classifier (harmonic loss).

Replaces the standard ``W @ x`` logits with Euclidean-distance logits to
class centers. Probabilities follow ``p_i ∝ 1 / d_i^(2n)`` where
``d_i = ||c_i - x||_2`` and ``n`` is the harmonic exponent. The head
returns ``pseudo_logits = -n * log(d²)`` so that ``softmax`` over them
reproduces those probabilities - meaning the standard CE pipeline can
consume the output unchanged.

The output-layer weights become *class centers* (convex combinations of
training examples) rather than arbitrary direction vectors, giving the
"crystal" geometry the paper is named for. Weights stay bounded under
training because the minimum of ``-log p_target`` lives at finite norm.

Cut-cross-entropy is incompatible with this head - cut-CE assumes the
classifier is a linear projection, but the centers are L2 anchors, not
dot-product weights. Use ``loss_func: cross_entropy``.

Reference: Baek et al., "Harmonic Loss Trains Interpretable AI Models"
(arXiv:2502.01628). Naming follows the authors' "grow-crystals" repo.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead

EPS: float = 1e-4

# Paper default for the mean-column-RMS embedding regularizer (Baek et al.).
# The crystal head reads this when computing its embedding_rms aux loss.
# Override per-experiment via YAML by setting ``embedding_rms_lambda``.
DEFAULT_EMBEDDING_RMS_LAMBDA: float = 0.01

# Resolution of the PCA density heatmap. 64 keeps the payload small
# (~16KB of ints) while resolving enough structure to read.
PCA_GRID_SIZE: int = 64


@torch.no_grad()
def _pca_density_grid(weights: list, grid_size: int = PCA_GRID_SIZE) -> dict:
    """Project stacked embedding weights to 2D PCA, bin into a density grid.

    Weights are stacked row-wise so the PCA sees the union of all
    embedding tables (e.g., byte-latent's 3 hash tables become one
    ``[3*V, D]`` matrix). Uses ``svd_lowrank`` to grab only the top-2
    PCs - cheap regardless of vocab size.
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

    xb = ((proj[:, 0] - x_min) / x_span * (grid_size - 1)).long().clamp_(0, grid_size - 1)
    yb = ((proj[:, 1] - y_min) / y_span * (grid_size - 1)).long().clamp_(0, grid_size - 1)
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
        "crystal_embedding_pca": {
            "description": (
                "Top-2 PCA projection of the embedding tables the crystal "
                "head regularizes (token embedding or byte-latent hash "
                "tables, stacked), binned to a density grid. Visual "
                "companion to ``embedding_rms``: structure here means the "
                "regularizer has shaped the embedding geometry into "
                "interpretable patterns."
            ),
            "snapshot": {
                "title": "Embedding PCA Density",
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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n = float(n)
        self.eps = float(eps)
        self.centers = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.centers, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        cc = (self.centers * self.centers).sum(-1)
        xx = (x_flat * x_flat).sum(-1, keepdim=True)
        cx = x_flat @ self.centers.T
        dist_sq = (cc.unsqueeze(0) + xx - 2.0 * cx).clamp_min(self.eps)
        pseudo_logits = -self.n * torch.log(dist_sq)
        return pseudo_logits.view(*orig_shape[:-1], self.vocab_size)

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

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)
        if config.loss_func == "cut_cross_entropy":
            raise ValueError(
                "head_type='crystal' is incompatible with "
                "loss_func='cut_cross_entropy' (cut-CE assumes a "
                "dot-product classifier)"
            )

        n_cfg = getattr(config, "crystal_n", None)
        if self.has_encoder:
            classifier = getattr(encoder, "classifier", None)
            if classifier is None or not hasattr(classifier, "weight"):
                raise ValueError(
                    "head_type='crystal' with an encoder requires the encoder "
                    "to expose a `.classifier` with a `.weight` attribute"
                )
            v_dim, f_dim = classifier.weight.shape
            n = float(n_cfg) if n_cfg is not None else math.sqrt(f_dim)
            self.lm_head = CrystalClassifier(
                hidden_size=f_dim, vocab_size=v_dim, n=n
            )
        else:
            n = float(n_cfg) if n_cfg is not None else math.sqrt(self.hidden_size)
            self.lm_head = CrystalClassifier(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                n=n,
            )

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.lm_head(hidden_states)

    @property
    def classifier(self) -> nn.Module:
        return self.lm_head

    def process_encoder_output(
        self,
        decoder_embeds: Tensor,
        encoder_logits: Tensor,
        encoder_classifier: nn.Module,
    ) -> Tuple[Tensor, Tensor, nn.Module]:
        logits = self.lm_head(decoder_embeds).to(encoder_logits.dtype)
        return logits, decoder_embeds, self.lm_head

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

    def dashboard_snapshots(
        self, embedding_weights: Optional[list] = None
    ) -> dict:
        """Top-2 PCA density grid of the regularized embeddings.

        Visual companion to the ``embedding_rms`` aux loss: if the
        regularizer is shaping geometry, this heatmap should develop
        structure (peaks, bands, clusters) rather than staying uniform.
        """
        grid = _pca_density_grid(embedding_weights or [])
        return {"crystal_embedding_pca": grid} if grid else {}

    def aux_losses(self, embedding_weights: Optional[list] = None) -> dict:
        """Mean-column-RMS embedding regularizer (Baek et al.).

        Penalizes ``mean(sqrt(mean(W**2, dim=0)))`` averaged across
        whatever embedding tables the model exposes (token embedding or
        byte-latent hash tables), so the geometry stays bounded the way
        the paper's harmonic-loss training expects. Specific to the
        crystal head: standard CE heads benefit from unbounded weight
        growth and we don't want to suppress it for them.
        """
        if not embedding_weights:
            return {}
        lam = float(
            getattr(self.config, "embedding_rms_lambda", DEFAULT_EMBEDDING_RMS_LAMBDA)
            or 0.0
        )
        if lam <= 0.0:
            return {}
        per_table = [
            W.pow(2).mean(dim=0).clamp_min(1e-12).sqrt().mean()
            for W in embedding_weights
        ]
        return {"embedding_rms": lam * torch.stack(per_table).mean()}
