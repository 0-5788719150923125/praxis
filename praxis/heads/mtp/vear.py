"""VEAR-harmonic MTP: sliding-window merged light harmonic depth-transforms.

Replaces the K independent full transformer blocks of the transformer/conv MTP
with a shared pool of N *light* harmonic experts, each a
``norm -> concat(h_prev, position_embed) -> project -> harmonic activation``
transform (no attention, memory, or PEER - a fraction of a full block). Each MTP
depth is a fixed cyclic sliding window over the pool: depth k merges experts
``[k, k+1, k+2]`` (uniform average of their parameters, one ``functional_call`` -
SMEAR's cheap-merge mechanic). Adjacent depths predict adjacent future positions
and share two of three experts, so the layout bakes in the correlation the task
already has.

This unifies the run's two VEAR uses: here (depth-transform merge) and the
prismatic4 terminal head's crystal bank. The experts stay distinct via a VEAR-
style repulsion aux loss on their projection directions - "variance-driven,
unique geometries" applied to the harmonic depth-transforms rather than a
router's affinity rows (there is no router: the window IS the routing). The
harmonic activation matches the codec/memory/head; the heavy HarmonicField grid
stays in the prismatic4 terminal every draft classifies through.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from praxis.activations import ACT2CLS
from praxis.normalization import NORMALIZATION_REGISTRY

# Cyclic window width over the expert pool. 3 = your "N-1 over 4" layout; fixed
# and model-agnostic (adjacent depths overlap by window-1 experts).
_WINDOW: int = 3
# Weight on the inter-expert repulsion, matching VEAR's constant.
_REPULSION: float = 0.01


def _hoyer(v: Tensor) -> Tensor:
    """Hoyer sparsity in [0, 1]: ``(sqrt(N) - ||v||_1/||v||_2) / (sqrt(N) - 1)``.
    1 = all energy on one feature, 0 = perfectly uniform. Scale-invariant. Same
    definition as ``HarmonicField.concentration`` so the two are comparable."""
    a = v.abs()
    n = a.numel()
    if n < 2:
        return a.new_zeros(())
    l1 = a.sum()
    l2 = torch.sqrt((a * a).sum() + 1e-12)
    return (math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1)


class _HarmonicExpert(nn.Module):
    """One light depth-transform: norm both inputs, concat, project, harmonic
    activation. Works in the config's ``hidden_size`` (set to ``embed_size`` for
    byte-level MTP so it matches the shared byte head's input width)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm_hidden = NORMALIZATION_REGISTRY[config.norm_type](
            config.hidden_size, eps=config.epsilon
        )
        self.norm_embed = NORMALIZATION_REGISTRY[config.norm_type](
            config.embed_size, eps=config.epsilon
        )
        self.projection = nn.Linear(
            config.hidden_size + config.embed_size, config.hidden_size, bias=False
        )
        # The harmonic nonlinearity (serpent for abstractinator-b) - the same
        # activation the codec, memory, and head use.
        self.act = ACT2CLS[config.activation]()

    def forward(self, hidden_states: Tensor, token_embeds: Tensor, mask=None) -> Tensor:
        h = self.norm_hidden(hidden_states, mode="direct")
        e = self.norm_embed(token_embeds, mode="direct")
        return self.act(self.projection(torch.cat([h, e], dim=-1)))


class VearHarmonicMTPBank(nn.Module):
    """A pool of ``num_experts`` light harmonic experts; each depth is a fixed
    cyclic sliding-window average of the pool."""

    # Field diagnostics, read off the experts' Serpent spectrum (per-feature
    # primary frequency ``a`` and secondary amplitude ``g``). The Serpent-based
    # analogue of HarmonicField's amplitude-grid metrics, so the harmonic-field
    # story survives the light-expert swap (same Hoyer definition, comparable
    # concentration reading). Collected via the MTP dynamics path; the paper's
    # harmonic section reads ``mtp_field_concentration`` through an inline.
    metric_descriptions = {
        "mtp_field_freq_norm": {
            "description": (
                "Mean L2 norm of each expert's per-feature primary frequency "
                "(Serpent alpha) - the magnitude of the learned harmonic "
                "spectrum in the MTP depth transforms. Stable near init = no "
                "structure learned; growing = the field is shaping itself."
            ),
            "chart": {
                "title": "MTP Field Frequency Norm",
                "y_label": "||alpha|| (mean over experts)",
                "y_scale": "logarithmic",
                "group": "mtp_field",
                "group_order": 45,
                "order": 10,
            },
        },
        "mtp_field_concentration": {
            "description": (
                "Mean Hoyer sparsity of the experts' frequency spectrum in "
                "[0, 1] (1 = all energy on one feature, 0 = uniform). The "
                "Serpent analogue of the harmonic field's spectral "
                "concentration - evidence the depth transforms commit to "
                "specific harmonics. Same Hoyer definition as HarmonicField."
            ),
            "chart": {
                "title": "MTP Field Concentration",
                "y_label": "Hoyer Sparsity",
                "y_scale": "linear",
                "group": "mtp_field",
                "order": 20,
            },
        },
        "mtp_field_amp_depth": {
            "description": (
                "Mean peak-to-trough of each expert's secondary amplitude "
                "(Serpent gamma) - how much harmonic modulation the transform "
                "carries. 0 = a flat (near-linear) transform; >0 = an "
                "oscillatory field."
            ),
            "chart": {
                "title": "MTP Field Amplitude Depth",
                "y_label": "gamma peak-to-trough",
                "y_scale": "linear",
                "group": "mtp_field",
                "order": 30,
            },
        },
        "mtp_field_distinctness": {
            "description": (
                "1 - mean pairwise |cosine| of the experts' frequency spectra: "
                "the readout of VEAR's repulsion goal. Near 0 = the pool "
                "collapsed to one geometry (redundant experts); rising toward 1 "
                "= the sliding-window experts specialized into distinct harmonic "
                "geometries."
            ),
            "chart": {
                "title": "MTP Field Distinctness",
                "y_label": "1 - mean |cosine|",
                "y_scale": "linear",
                "group": "mtp_field",
                "order": 40,
            },
        },
    }

    def __init__(self, config, num_experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_HarmonicExpert(config) for _ in range(num_experts)]
        )
        self.window = min(_WINDOW, num_experts)
        self._materialize(config.hidden_size, config.embed_size)

    @torch.no_grad()
    def _materialize(self, hidden_size: int, embed_size: int) -> None:
        """Run each expert once so lazy activation params (serpent) become
        concrete - the merge averages parameters via ``functional_call`` without
        running each expert, so it can't hit an ``UninitializedParameter``."""
        if not any(
            isinstance(p, UninitializedParameter)
            for e in self.experts
            for p in e.parameters()
        ):
            return
        h = torch.zeros(1, 1, hidden_size)
        e = torch.zeros(1, 1, embed_size)
        for expert in self.experts:
            expert(h, e)

    def _window(self, depth: int) -> list:
        n = len(self.experts)
        return [(depth + j) % n for j in range(self.window)]

    def forward(self, hidden_states: Tensor, token_embeds: Tensor, mask, depth: int):
        """Merge this depth's expert window (uniform parameter average) and run
        the merged transform. Grad flows to every expert in the window."""
        idx = self._window(depth)
        names = dict(self.experts[idx[0]].named_parameters()).keys()
        merged = {
            name: torch.stack(
                [dict(self.experts[i].named_parameters())[name] for i in idx]
            ).mean(dim=0)
            for name in names
        }
        return torch.func.functional_call(
            self.experts[idx[0]], merged, (hidden_states, token_embeds, mask)
        )

    def repulsion_loss(self) -> Tensor:
        """VEAR-style: drive the experts to distinct harmonic geometries by
        penalizing the mean off-diagonal |cosine| between their projection
        directions. 0 = orthogonal (unique); high = redundant."""
        n = len(self.experts)
        if n < 2:
            return self.experts[0].projection.weight.new_zeros(())
        rows = F.normalize(
            torch.stack([e.projection.weight.reshape(-1) for e in self.experts]), dim=1
        )
        sim = rows @ rows.t()
        off = sim - torch.eye(n, device=rows.device, dtype=rows.dtype)
        return _REPULSION * (off.abs().sum() / (n * (n - 1)))

    # ── Harmonic-field diagnostics (Serpent spectrum) ───────────────────────

    def _spectrum(self) -> Optional[tuple]:
        """Per-expert Serpent parameters ``(alpha, gamma)`` as ``[N, D]``, or
        ``None`` while any expert's activation is still lazy."""
        acts = [e.act for e in self.experts]
        if any(
            isinstance(p, UninitializedParameter) for a in acts for p in a.parameters()
        ):
            return None
        alpha = torch.stack([a.a.detach() for a in acts])  # [N, D]
        gamma = torch.stack([a.g.detach() for a in acts])  # [N, D]
        return alpha, gamma

    @torch.no_grad()
    def training_metrics(self) -> dict:
        spec = self._spectrum()
        if spec is None:
            return {}
        alpha, gamma = spec
        alpha = alpha.float()
        n = alpha.shape[0]
        conc = torch.stack([_hoyer(alpha[i]) for i in range(n)]).mean()
        rows = F.normalize(alpha, dim=1)
        sim = rows @ rows.t()
        off = (sim - torch.eye(n, device=rows.device, dtype=rows.dtype)).abs()
        distinct = 1.0 - (off.sum() / max(n * (n - 1), 1))
        depth = (gamma.float().amax(dim=1) - gamma.float().amin(dim=1)).mean()
        return {
            "mtp_field_freq_norm": float(alpha.norm(dim=1).mean().item()),
            "mtp_field_concentration": float(conc.item()),
            "mtp_field_amp_depth": float(depth.item()),
            "mtp_field_distinctness": float(distinct.item()),
        }

    @torch.no_grad()
    def dashboard_snapshots(self) -> dict:
        """The per-expert frequency spectrum as a heatmap ``[N experts x D]`` -
        the Serpent analogue of HarmonicField's amplitude-grid view. Rows are
        experts (the sliding-window pool), columns features; distinct rows are
        the repulsion working."""
        spec = self._spectrum()
        if spec is None:
            return {}
        alpha, _ = spec
        return {
            "mtp_field_spectrum": {
                "status": "ok",
                "grid": alpha.abs().float().cpu().tolist(),
                "rows": int(alpha.shape[0]),
                "cols": int(alpha.shape[1]),
            }
        }
