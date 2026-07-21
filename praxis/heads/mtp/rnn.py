"""Serpent-RNN MTP: one shared gated recurrent cell unrolled across depths.

The chained MTP computation (previous depth's hidden + next position's embed
-> this depth's hidden) IS a recurrence, so model it as one. vear already
reduced each depth to a pointwise ``norm -> concat -> project -> serpent``
transform - an ungated Elman cell whose per-step weights are synthesized by a
sliding-window parameter merge. This bank ties those steps into a SINGLE cell
with a minGRU-style update gate:

    x_k  = concat(norm(h_{k-1}), norm(e_k)) + d_k
    z_k  = sigmoid(W_z x_k + b_z)
    c_k  = postnorm(serpent(W_c x_k + b_c))
    h_k  = (1 - z_k) * h_{k-1} + z_k * c_k

where ``d_k`` is a zero-initialized per-depth embedding: the depths start as
the same function and differentiate only as far as the task pulls them apart,
replacing vear's window merge as the depth-specialization mechanism - no K
modules, no per-call parameter stack, no ``functional_call``.

Parameters are O(1) in ``mtp_depth`` (plus the K x (H+E) depth table). The
drafted form is exactly the trained form: the cell is pointwise, so nothing it
learns depends on cross-position context that would vanish at the one-position
draft step. Stability across the unroll follows the recurrent-depth rule
(additive signals compound across depth passes unless post-normalized): the
candidate is post-normalized - the sandwich placement - and the state update is
a convex blend, which is what makes the biases and the additive depth
embedding safe to carry inside the loop.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

from praxis.activations import ACT2CLS
from praxis.heads.mtp.vear import _hoyer
from praxis.normalization import NORMALIZATION_REGISTRY


class SerpentRNNMTPBank(nn.Module):
    """One shared gated serpent cell; each MTP depth is one unroll step."""

    def __init__(self, config, num_depths: int) -> None:
        super().__init__()
        self.num_depths = num_depths
        hidden, embed = config.hidden_size, config.embed_size
        self.norm_hidden = NORMALIZATION_REGISTRY[config.norm_type](
            hidden, eps=config.epsilon
        )
        self.norm_embed = NORMALIZATION_REGISTRY[config.norm_type](
            embed, eps=config.epsilon
        )
        # Zero-init: every depth starts as the same function; specialization is
        # learned, not imposed.
        self.depth_embed = nn.Parameter(torch.zeros(num_depths, hidden + embed))
        self.gate = nn.Linear(hidden + embed, hidden)
        self.candidate = nn.Linear(hidden + embed, hidden)
        # The harmonic nonlinearity - the same activation the codec, memory,
        # and head use.
        self.act = ACT2CLS[config.activation]()
        self.norm_out = NORMALIZATION_REGISTRY[config.norm_type](
            hidden, eps=config.epsilon
        )
        # Rolling per-depth gate means (buffer, not python state: indexed
        # in-place writes stay inside a compiled graph; a python-side .item()
        # here would force a device sync every depth).
        self.register_buffer("gate_mean", torch.zeros(num_depths), persistent=False)
        self._materialize(hidden, embed)

    @torch.no_grad()
    def _materialize(self, hidden: int, embed: int) -> None:
        """Run the cell once so lazy activation params (serpent) become
        concrete before the optimizer walks the parameter list."""
        if not any(
            isinstance(p, UninitializedParameter) for p in self.act.parameters()
        ):
            return
        self(torch.zeros(1, 1, hidden), torch.zeros(1, 1, embed), None)
        self.gate_mean.zero_()

    def forward(
        self, hidden_states: Tensor, token_embeds: Tensor, mask=None, depth: int = 0
    ) -> Tensor:
        h = self.norm_hidden(hidden_states, mode="direct")
        e = self.norm_embed(token_embeds, mode="direct")
        x = torch.cat([h, e], dim=-1) + self.depth_embed[depth]
        z = torch.sigmoid(self.gate(x))
        cand = self.norm_out(self.act(self.candidate(x)), mode="direct")
        with torch.no_grad():
            self.gate_mean[depth] = z.mean()
        return (1.0 - z) * hidden_states + z * cand

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def _spectrum(self) -> Optional[tuple]:
        """The cell's Serpent parameters ``(alpha, gamma)`` as ``[D]``, or
        ``None`` while the activation is still lazy."""
        if any(isinstance(p, UninitializedParameter) for p in self.act.parameters()):
            return None
        return self.act.a.detach(), self.act.g.detach()

    @torch.no_grad()
    def training_metrics(self) -> dict:
        out: dict = {}
        for k, g in enumerate(self.gate_mean.tolist()):
            out[f"mtp_rnn_gate_d{k}"] = float(g)
        for k, n in enumerate(self.depth_embed.detach().norm(dim=1).tolist()):
            out[f"mtp_rnn_depth_embed_d{k}"] = float(n)
        spec = self._spectrum()
        if spec is not None:
            alpha, gamma = spec
            alpha = alpha.float()
            gamma = gamma.float()
            out["mtp_field_freq_norm"] = float(alpha.norm().item())
            out["mtp_field_concentration"] = float(_hoyer(alpha).item())
            out["mtp_field_amp_depth"] = float((gamma.amax() - gamma.amin()).item())
        return out

    @torch.no_grad()
    def dashboard_snapshots(self) -> dict:
        """The cell's harmonic signature as a 2-row grid: primary frequency
        (Serpent alpha) and secondary amplitude (gamma) per feature."""
        spec = self._spectrum()
        if spec is None:
            return {}
        alpha, gamma = spec
        grid = torch.stack([alpha.abs(), gamma.abs()]).float().cpu().tolist()
        return {
            "mtp_field_spectrum": {
                "status": "ok",
                "grid": grid,
                "rows": 2,
                "cols": int(alpha.shape[0]),
            }
        }

    def metric_descriptions(self) -> dict:
        """Chart hints. Per-depth keys depend on ``num_depths``, so this is an
        instance method rather than vear's static class dict."""
        out: dict = {
            f"mtp_rnn_gate_d{k}": {
                "description": (
                    f"Mean update-gate value at unroll step {k} of the shared "
                    "serpent MTP cell. The recurrence health readout: near 0 = "
                    "the cell ignores its candidate and passes the state "
                    "through (identity collapse); near 1 = the state is "
                    "discarded every step (the chain carries nothing)."
                ),
                "chart": {
                    "title": "MTP RNN Gate (per depth)",
                    "y_label": "mean sigmoid gate",
                    "y_scale": "linear",
                    "group": "mtp_field",
                    "group_order": 45,
                    "order": 60,
                    "series_group": "mtp_rnn_gate",
                    "series_label": f"depth {k}",
                },
            }
            for k in range(self.num_depths)
        }
        out.update(
            {
                f"mtp_rnn_depth_embed_d{k}": {
                    "description": (
                        f"L2 norm of unroll step {k}'s depth embedding. Zero-"
                        "initialized, so this measures how far the shared cell "
                        "has differentiated per depth - the learned replacement "
                        "for vear's per-depth window merge. Flat at 0 = one "
                        "function serves every offset."
                    ),
                    "chart": {
                        "title": "MTP RNN Depth Specialization",
                        "y_label": "||depth embed||",
                        "y_scale": "linear",
                        "group": "mtp_field",
                        "group_order": 45,
                        "order": 70,
                        "series_group": "mtp_rnn_depth_embed",
                        "series_label": f"depth {k}",
                    },
                }
                for k in range(self.num_depths)
            }
        )
        out["mtp_field_freq_norm"] = {
            "description": (
                "L2 norm of the shared MTP cell's per-feature primary frequency "
                "(Serpent alpha) - the magnitude of the learned harmonic "
                "spectrum in the depth transform. Stable near init = no "
                "structure learned; growing = the field is shaping itself."
            ),
            "chart": {
                "title": "MTP Field Frequency Norm",
                "y_label": "||alpha||",
                "y_scale": "logarithmic",
                "group": "mtp_field",
                "group_order": 45,
                "order": 10,
            },
        }
        out["mtp_field_concentration"] = {
            "description": (
                "Hoyer sparsity of the shared MTP cell's frequency spectrum in "
                "[0, 1] (1 = all energy on one feature, 0 = uniform) - evidence "
                "the depth transform commits to specific harmonics. Same Hoyer "
                "definition as HarmonicField, comparable with vear runs."
            ),
            "chart": {
                "title": "MTP Field Concentration",
                "y_label": "Hoyer Sparsity",
                "y_scale": "linear",
                "group": "mtp_field",
                "order": 20,
            },
        }
        out["mtp_field_amp_depth"] = {
            "description": (
                "Peak-to-trough of the shared MTP cell's secondary amplitude "
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
        }
        return out
