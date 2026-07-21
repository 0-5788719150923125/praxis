"""Mono-Forward as a decoder feature: break the graph, score locally.

The simplest form of the idea (Mono-Forward, https://arxiv.org/abs/2501.09238):
between decoder layers, a module DETACHES the hidden states and computes a
local goodness score at the cut, so no gradient ever crosses a cut - each
segment trains from its own local loss, and one ordinary backward pass over
the (disconnected) graph trains every segment simultaneously. No actors, no
pipelines, no parallelism machinery: autograd already decomposes a
disconnected graph into independent local updates. The older trainers in
``praxis/trainers/mono_forward/`` exist for the distributed story; this
module is the sequential one, integrated with the decoder like any other
component (halting, width, compression).

Cut schedules (``--mono-type``), chosen to plan ahead rather than hard-code
the flat layout:

- ``layer``: detach after EVERY expert call - the totally flat regime; the
  main head loss trains only the head, every layer lives on its goodness.
- ``cycle``: detach after each full pass through the ``num_layers`` physical
  experts - one cut per recurrent depth step, so backprop spans a single
  recurrence cycle and no further.
- ``final``: one cut after the whole stack - the decoder trains from a single
  goodness score at the top, and the head trains alone beyond it.

Goodness is path-aware:

- Token path: the paper's form - a per-cut projection to vocab logits, CE
  against the (shifted) labels.
- Encoder path (byte-latent / CALM): the decoder runs in PATCH space, where
  byte labels do not align per position, so the categorical target is
  replaced by its patch-space analogue: predict the NEXT position's input
  patch embedding (Huber). The target is external to the layer (produced by
  the encoder, detached here), which matters: with the graph cut, a layer's
  ONLY training signal is its local score, and a self-referential target
  (its own shifted output) would reward collapsing to a constant. Anchoring
  on the encoder's stream keeps the local objective non-degenerate.

Each cut owns its projection (the paper's per-layer M_i); projections index
by cut slot, so halting/early exit simply uses fewer of them.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.containers import LossContainer

# Named cut schedules. Values are spec dicts (docs render the descriptions);
# selection flows through build_mono, never a per-knob CLI flag.
MONO_REGISTRY: Dict[str, Optional[dict]] = {
    "layer": dict(
        description="Detach after every expert call (totally flat; default "
        "mono-forward regime).",
    ),
    "cycle": dict(
        description="Detach after each full pass through the num_layers "
        "physical experts (one cut per recurrent depth step).",
    ),
    "final": dict(
        description="Detach once after the whole stack (single goodness at "
        "the top; the head trains alone beyond it).",
    ),
}


# Rendered by the auto-docs generator (registry values are spec dicts).
MONO_DESCRIPTIONS: Dict[str, str] = {
    name: spec["description"] for name, spec in MONO_REGISTRY.items() if spec
}


class MonoBase(nn.Module):
    """No-op mono-forward. Decoders hold a real object and never branch."""

    enabled = False

    def __init__(self, config=None) -> None:
        super().__init__()

    def begin(self, hidden_states: Tensor, labels: Optional[Tensor]) -> None:
        pass

    def cut(
        self, hidden_states: Tensor, losses: LossContainer, current_depth: int
    ) -> Tensor:
        return hidden_states

    def finalize(self, hidden_states: Tensor, losses: LossContainer) -> Tensor:
        return hidden_states

    def metrics(self) -> dict:
        return {}


class MonoForward(MonoBase):
    """Graph-cutting goodness module for the sequential decoder."""

    enabled = True

    def __init__(self, config, mode: str) -> None:
        super().__init__()
        if mode not in MONO_REGISTRY:
            raise ValueError(
                f"Unknown mono type '{mode}'. Choices: {sorted(MONO_REGISTRY)}"
            )
        self.mode = mode
        self.num_layers = getattr(config, "num_layers", 1) or 1
        self.latent = config.encoder_type is not None
        hidden = config.hidden_size
        out_dim = hidden if self.latent else config.vocab_size
        if mode == "layer":
            n_cuts = config.depth
        elif mode == "cycle":
            n_cuts = max(1, math.ceil(config.depth / self.num_layers))
        else:  # final
            n_cuts = 1
        # The paper's per-layer projection M_i, one per cut slot.
        self.proj = nn.ModuleList(
            nn.Linear(hidden, out_dim, bias=False) for _ in range(n_cuts)
        )
        self._cut_idx = 0
        self._scores: list = []  # graph-connected local losses, this forward
        self._score_values: list = []  # detached per-cut values, for metrics
        self._labels: Optional[Tensor] = None
        self._target: Optional[Tensor] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mode='{self.mode}', cuts={len(self.proj)}, "
            f"goodness='{'latent' if self.latent else 'vocab'}')"
        )

    # ── forward-pass protocol (called by SequentialDecoder) ────────────────

    def begin(self, hidden_states: Tensor, labels: Optional[Tensor]) -> None:
        """Reset per-forward state and stash the goodness target."""
        self._cut_idx = 0
        self._scores = []
        self._score_values = []
        self._labels = labels
        # Latent goodness target: the decoder's INPUT stream, shifted at score
        # time. Detached - the target is the encoder's, not the layer's.
        self._target = hidden_states.detach() if self.latent else None

    def _due(self, current_depth: int) -> bool:
        if self.mode == "layer":
            return True
        if self.mode == "cycle":
            return (current_depth + 1) % self.num_layers == 0
        return False  # final: handled in finalize

    def cut(
        self, hidden_states: Tensor, losses: LossContainer, current_depth: int
    ) -> Tensor:
        if not self.training or not self._due(current_depth):
            return hidden_states
        self._score(hidden_states)
        return hidden_states.detach()

    def finalize(self, hidden_states: Tensor, losses: LossContainer) -> Tensor:
        """Close the forward: the ``final`` schedule cuts here, and the mean
        over this forward's local scores lands in the container as "mono"
        (mean, not sum, so the aux weight doesn't scale with depth)."""
        if not self.training:
            return hidden_states
        if self.mode == "final":
            self._score(hidden_states)
            hidden_states = hidden_states.detach()
        if self._scores:
            losses.add_loss("mono", sum(self._scores) / len(self._scores))
            self._scores = []
        return hidden_states

    # ── goodness ────────────────────────────────────────────────────────────

    def _score(self, hidden_states: Tensor) -> None:
        idx = min(self._cut_idx, len(self.proj) - 1)
        self._cut_idx += 1
        if self.latent:
            target = self._target
            # A compressor may have changed the sequence length mid-loop; the
            # stashed input stream no longer aligns, so this cut scores nothing
            # (the detach still happens - the graph break is the contract).
            if target is None or target.size(1) != hidden_states.size(1):
                return
            if hidden_states.size(1) < 2:
                return
            pred = self.proj[idx](hidden_states[:, :-1])
            loss = F.smooth_l1_loss(pred, target[:, 1:])
        else:
            labels = self._labels
            if labels is None or labels.size(1) != hidden_states.size(1):
                return
            if hidden_states.size(1) < 2:
                return
            logits = self.proj[idx](hidden_states[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        self._scores.append(loss)
        with torch.no_grad():
            self._score_values.append(loss.detach())

    def metrics(self) -> dict:
        """Per-cut goodness values from the last training forward (merged into
        the decoder's get_metrics extras, alongside the depth/ keys). Values
        stay on-device until read."""
        out = {}
        for k, v in enumerate(self._score_values):
            out[f"mono/goodness_d{k}"] = float(v)
        if self._score_values:
            out["mono/goodness"] = float(
                sum(self._score_values) / len(self._score_values)
            )
            out["mono/cuts"] = float(len(self._score_values))
        return out


def build_mono(config) -> MonoBase:
    """Instantiate the mono-forward feature for a decoder, or the no-op.

    Sequential-only: the cut protocol is a property of walking layers one at
    a time; a parallel decoder has no between-layer seam to cut.
    """
    mode = getattr(config, "mono_type", None)
    if not mode:
        return MonoBase(config)
    if getattr(config, "decoder_type", "sequential") != "sequential":
        raise ValueError(
            "mono_type requires decoder_type='sequential' - the graph cuts "
            "live between sequentially-walked layers."
        )
    return MonoForward(config, mode)
