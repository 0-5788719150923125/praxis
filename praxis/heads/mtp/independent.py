"""Per-depth MTP: K independent light harmonic transforms, chained by hidden.

This is the DeepSeek-V3 MTP shape - one module per draft depth, each taking the
previous depth's hidden and the next position's ground-truth embedding - with
the depth transform kept POINTWISE:

    h_k = act(W_k [norm(h_{k-1}) ; norm(e_k)])

Nothing is shared between depths and nothing forces ``h_k`` back toward
``h_{k-1}``. That is the whole difference from ``serpent_rnn``, whose single
gated cell both reuses one set of weights at every unroll step and blends
convexly with its own previous state - a constraint depth ``k`` cannot opt out
of. Here depth ``k`` is free to map wherever its own target offset wants.

WHY NOT THE TRANSFORMER BANK, which is the literal DeepSeek module. Its depth
transform contains attention, so its output at a position depends on the
positions before it. Training runs it over the whole sequence; drafting runs it
over a single position with no cache, which is a different function - measured
on abstractinator-u's width, the last-position output moves by 2.02 on a scale
of 2.96 between the two. ``conv`` fails the same way through its zero-padded
history. Both predate the byte-latent draft path and were written for the
token-path auxiliary loss, where nothing ever drafts. A pointwise transform has
no history to lose, so the drafted form here is exactly the trained form - the
same property that makes ``vear`` and ``serpent_rnn`` safe to draft with.

Cost is ``K`` light experts (norm, norm, one projection, one harmonic
activation) - no attention, memory, or PEER. Against the transformer bank's
full block per depth that is roughly an eighth of the parameters at equal
depth, and it is the cheapest structure that still gives each depth its own.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from praxis.heads.mtp.vear import _HarmonicExpert, _hoyer


class PerDepthMTPBank(nn.Module):
    """One independent light harmonic transform per MTP depth."""

    def __init__(self, config, num_depths: int) -> None:
        super().__init__()
        self.num_depths = num_depths
        self.depths = nn.ModuleList(
            [_HarmonicExpert(config) for _ in range(num_depths)]
        )
        self._materialize(config.hidden_size, config.embed_size)

    @torch.no_grad()
    def _materialize(self, hidden_size: int, embed_size: int) -> None:
        """Run each transform once so lazy activation params (serpent) become
        concrete before the optimizer walks the parameter list - an
        ``UninitializedParameter`` surviving into ``parameters()`` raises the
        moment anything reads ``.numel()``."""
        if not any(
            isinstance(p, UninitializedParameter)
            for d in self.depths
            for p in d.parameters()
        ):
            return
        h = torch.zeros(1, 1, hidden_size)
        e = torch.zeros(1, 1, embed_size)
        for depth in self.depths:
            depth(h, e)

    def forward(
        self, hidden_states: Tensor, token_embeds: Tensor, mask=None, depth: int = 0
    ) -> Tensor:
        return self.depths[depth](hidden_states, token_embeds, mask)

    # ── Harmonic-field diagnostics (Serpent spectrum) ───────────────────────

    def _spectrum(self) -> Optional[tuple]:
        """Per-depth Serpent parameters ``(alpha, gamma)`` as ``[K, D]``, or
        ``None`` while any depth's activation is still lazy."""
        acts = [d.act for d in self.depths]
        if any(
            isinstance(p, UninitializedParameter) for a in acts for p in a.parameters()
        ):
            return None
        return (
            torch.stack([a.a.detach() for a in acts]),
            torch.stack([a.g.detach() for a in acts]),
        )

    @torch.no_grad()
    def training_metrics(self) -> dict:
        spec = self._spectrum()
        if spec is None:
            return {}
        alpha, gamma = spec
        alpha = alpha.float()
        gamma = gamma.float()
        k = alpha.shape[0]
        rows = F.normalize(alpha, dim=1)
        off = rows @ rows.t() - torch.eye(k, device=rows.device, dtype=rows.dtype)
        out = {
            "mtp_field_freq_norm": float(alpha.norm(dim=1).mean().item()),
            "mtp_field_concentration": float(
                torch.stack([_hoyer(alpha[i]) for i in range(k)]).mean().item()
            ),
            "mtp_field_amp_depth": float(
                (gamma.amax(dim=1) - gamma.amin(dim=1)).mean().item()
            ),
            "mtp_field_distinctness": float(
                1.0 - (off.abs().sum() / max(k * (k - 1), 1)).item()
            ),
        }
        # Per-depth projection norms: the direct readout of whether the depths
        # actually diverged. This bank's premise is that they are free to; the
        # shared-cell alternative can only report how far ONE function drifted
        # per offset, so a flat profile here would say the freedom went unused.
        for i, d in enumerate(self.depths):
            out[f"mtp_depth_weight_d{i}"] = float(d.projection.weight.norm().item())
        return out

    @torch.no_grad()
    def dashboard_snapshots(self) -> dict:
        """Per-depth frequency spectrum as a ``[K x D]`` heatmap. Rows are draft
        depths (not vear's expert pool), so a row that matches its neighbour is
        two offsets that settled on the same harmonic geometry."""
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

    def metric_descriptions(self) -> dict:
        """Chart hints. Per-depth keys depend on ``num_depths``, so this is an
        instance method rather than vear's static class dict."""
        out: dict = {
            "mtp_field_freq_norm": {
                "description": (
                    "Mean L2 norm of each depth transform's per-feature primary "
                    "frequency (Serpent alpha) - the magnitude of the learned "
                    "harmonic spectrum in the MTP depths. Stable near init = no "
                    "structure learned; growing = the field is shaping itself."
                ),
                "chart": {
                    "title": "MTP Field Frequency Norm",
                    "y_label": "||alpha|| (mean over depths)",
                    "y_scale": "logarithmic",
                    "group": "mtp_field",
                    "group_order": 45,
                    "order": 10,
                },
            },
            "mtp_field_concentration": {
                "description": (
                    "Mean Hoyer sparsity of the depth transforms' frequency "
                    "spectrum in [0, 1] (1 = all energy on one feature, 0 = "
                    "uniform) - evidence the transforms commit to specific "
                    "harmonics. Same Hoyer definition as HarmonicField, "
                    "comparable with vear and serpent_rnn runs."
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
                    "Mean peak-to-trough of each depth transform's secondary "
                    "amplitude (Serpent gamma) - how much harmonic modulation "
                    "the transform carries. 0 = a flat (near-linear) transform; "
                    ">0 = an oscillatory field."
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
                    "1 - mean pairwise |cosine| of the depths' frequency "
                    "spectra. Near 0 = every offset settled on the same harmonic "
                    "geometry, which is the case where independent per-depth "
                    "parameters bought nothing over a shared cell; rising toward "
                    "1 = the depths specialized. Unlike vear there is no "
                    "repulsion penalty holding this up - it is a measurement, "
                    "not a target."
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
        out.update(
            {
                f"mtp_depth_weight_d{k}": {
                    "description": (
                        f"L2 norm of draft depth {k}'s projection weight. The "
                        "per-depth counterpart of serpent_rnn's depth-signature "
                        "norm: these are independent parameters, so a profile "
                        "that stays flat across depths says the offsets are "
                        "converging on one transform anyway."
                    ),
                    "chart": {
                        "title": "MTP Depth Transform Scale",
                        "y_label": "||W_k||",
                        "y_scale": "linear",
                        "group": "mtp_field",
                        "group_order": 45,
                        "order": 70,
                        "series_group": "mtp_depth_weight",
                        "series_label": f"depth {k}",
                    },
                }
                for k in range(self.num_depths)
            }
        )
        return out
