"""Titans memory surfacings: how a learned long-term memory branch is
combined with the residual stream.

``MemoryBase`` is a concrete no-op (identity forward, no metrics) and the
parent of the real surfacings, so a memory-free block carries a real object
instead of ``None`` - the block and decoder never branch on whether memory is
present. Each real surfacing wraps the shared ``NeuralMemory`` core:
- MAL applies memory as its own residual sub-layer.
- MAG blends a parallel memory branch with attention through a learned gate.
"""

from collections import deque
from typing import Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from praxis.memory.models import build_memory_model
from praxis.memory.neural_memory import NeuralMemory, NeuralMemState

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class MemoryBase(nn.Module):
    """No-op memory and base for the real surfacings. Passes the stream
    through unchanged and reports no metrics."""

    def __init__(self, config: ConfigType) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

    def forward(
        self,
        stream: Tensor,
        attn_output: Tensor,
        state: Optional[NeuralMemState] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[NeuralMemState]]:
        return stream, state

    def training_metrics(self) -> dict:
        """Diagnostic scalars surfaced each logging step (no-op by default)."""
        return {}

    @staticmethod
    def collect_training_metrics(root: nn.Module) -> dict:
        """Average each memory metric across the active memory modules under
        ``root`` (empty when none are active)."""
        sums: dict = {}
        counts: dict = {}
        for module in root.modules():
            if isinstance(module, MemoryBase):
                for key, value in module.training_metrics().items():
                    if value is not None:
                        sums[key] = sums.get(key, 0.0) + value
                        counts[key] = counts.get(key, 0) + 1
        return {key: sums[key] / counts[key] for key in sums}

    @staticmethod
    def collect_metric_descriptions(root: nn.Module) -> dict:
        """Gather ``metric_descriptions`` from memory modules under ``root``."""
        out: dict = {}
        for module in root.modules():
            if isinstance(module, MemoryBase):
                descs = getattr(type(module), "metric_descriptions", None)
                if isinstance(descs, dict):
                    out.update(descs)
        return out


class MemorySurfacing(MemoryBase):
    """Base for surfacings that own a ``NeuralMemory`` core."""

    # Declared here so the metric's definition lives with the component; the
    # dynamics callback and metric-description walker discover it.
    metric_descriptions = {
        "memory_surprise": {
            "description": (
                "Mean RAW reconstruction loss at the cold init weights, averaged "
                "across memory layers. Pre-update novelty to the cold memory and "
                "scale-sensitive (it can be dominated by the memory net's free "
                "output scale); read Memory Surprise (norm) for the scale-free "
                "quantity the update optimizes."
            ),
            "chart": {
                "title": "Memory Surprise (raw)",
                "y_label": "Surprise",
                "y_scale": "logarithmic",
                "group": "memory",
                "group_order": 20,
                "order": 10,
            },
        },
        "memory_surprise_norm": {
            "description": (
                "Surprise in RMS-normalized (directional) space, which is what "
                "the readout's out_norm consumes - the scale-free quantity the "
                "energy update actually optimizes. In the predictive arm this is "
                "the next-latent (Huber) prediction error; falling = the memory "
                "is learning to forecast the stream, i.e. storing belief-state "
                "structure rather than echoing the current token."
            ),
            "chart": {
                "title": "Memory Surprise (norm)",
                "y_label": "Surprise (normalized)",
                "y_scale": "linear",
                "group": "memory",
                "order": 11,
            },
        },
        "memory_gain": {
            "description": (
                "Memory output magnitude relative to the residual stream "
                "(||retrieved|| / ||stream||), averaged across memory layers. "
                "Decaying toward 0 means the model is routing around the memory."
            ),
            "chart": {
                "title": "Memory Gain",
                "y_label": "retrieved / stream",
                "y_scale": "linear",
                "group": "memory",
                "order": 12,
            },
        },
        "memory_write": {
            "description": (
                "Relative size of the per-sequence test-time weight update "
                "(||W_T - W0|| / ||W0||), averaged across memory layers. Near 0 "
                "means the update is inert (the memory is not memorizing)."
            ),
            "chart": {
                "title": "Memory Write",
                "y_label": "delta-W / W0",
                "y_scale": "linear",
                "group": "memory",
                "order": 13,
            },
        },
        # Event sizes share one chart (mean/min/max are the same scale) via a
        # series_group; the lowest-order member supplies the title/axis/subtitle.
        "memory_event_size": {
            "description": (
                "Event lengths (tokens) from surprise-based segmentation (energy "
                "mode), averaged across memory layers: mean, min and max across "
                "the events in a store pass. Events split at surprise spikes and "
                "are capped at chunk_size; mean below the cap means the memory is "
                "finding boundaries."
            ),
            "chart": {
                "title": "Memory Event Size",
                "y_label": "tokens / event",
                "y_scale": "linear",
                "group": "memory",
                "order": 14,
                "series_group": "memory_event",
                "series_label": "mean",
            },
        },
        "memory_event_min": {
            "description": "Smallest event length (tokens) in the store pass.",
            "chart": {
                "title": "Memory Event Size",
                "y_label": "tokens / event",
                "y_scale": "linear",
                "group": "memory",
                "order": 15,
                "series_group": "memory_event",
                "series_label": "min",
            },
        },
        "memory_event_max": {
            "description": "Largest event length (tokens) in the store pass (caps at chunk_size).",
            "chart": {
                "title": "Memory Event Size",
                "y_label": "tokens / event",
                "y_scale": "linear",
                "group": "memory",
                "order": 16,
                "series_group": "memory_event",
                "series_label": "max",
            },
        },
    }

    def __init__(self, config: ConfigType, spec: dict) -> None:
        super().__init__(config)
        self.mem = NeuralMemory(
            dim=self.hidden_size,
            model=build_memory_model(config, spec),
            chunk_size=spec.get("chunk_size", 64),
            momentum=spec.get("momentum", True),
            use_energy=spec.get("use_energy", False),
            segment=spec.get("segment", False),
            segment_block=spec.get("segment_block", 16),
            parallel_scan=spec.get("parallel_scan", True),
            write_objective=spec.get("write_objective", "recon"),
        )

    def forward(self, stream, attn_output, state=None, current_depth: int = 0):
        raise NotImplementedError

    def training_metrics(self) -> dict:
        m = self.mem
        out = {}
        if m.last_surprise is not None:
            out["memory_surprise"] = float(m.last_surprise)
        if m.last_surprise_norm is not None:
            out["memory_surprise_norm"] = float(m.last_surprise_norm)
        if m.last_gain is not None:
            out["memory_gain"] = float(m.last_gain)
        if m.last_write is not None:
            out["memory_write"] = float(m.last_write)
        if m.last_event_mean is not None:
            out["memory_event_size"] = float(m.last_event_mean)
            out["memory_event_min"] = float(m.last_event_min)
            out["memory_event_max"] = float(m.last_event_max)
        return out


class MemoryAsLayer(MemorySurfacing):
    """MAL: memory as its own residual sub-layer within the block."""

    def forward(self, stream, attn_output, state=None, current_depth: int = 0):
        retrieved, state = self.mem(stream, state)
        return stream + retrieved, state


class MemoryAsGate(MemorySurfacing):
    """MAG: a memory branch blended with the attention-carrying stream through
    a learned per-channel gate. The gate starts near the stream so memory eases
    in during training."""

    def __init__(self, config, spec):
        super().__init__(config, spec)
        self.gate = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, stream, attn_output, state=None, current_depth: int = 0):
        retrieved, state = self.mem(stream, state)
        g = self.gate(stream).sigmoid()
        return g * retrieved + (1 - g) * stream, state


# EML (core B) can never be weighted below this - the exploration floor that
# keeps the granular regime alive long enough to earn its keep, instead of a
# loss-optimized router starving it before it matures. Symmetric (core A floored
# too), so neither regime can fully collapse: the two are held on a stable axis.
_BLEND_FLOOR: float = 0.1
# EMA momentum on each core's earned value - deliberately slow, so the incentive
# to lean on a regime is latent (built over many steps), not reactive to one
# noisy batch. Fixed, model-agnostic (no per-experiment knob).
_VALUE_EMA: float = 0.99
# The competition is visualized over exactly the EMA's effective horizon
# (1 / (1 - momentum)) - the number of steps the running average actually keeps,
# so the river card shows precisely what the bandit is "remembering".
_RIVER_HORIZON: int = round(1.0 / (1.0 - _VALUE_EMA))


# Short display names for the regime bands (river headers + blend charts). Keyed
# by dense variant; unknown variants fall back to the raw key.
_REGIME_NAMES = {
    "mlp": "energy",
    "eml_tree": "EML",
    "kan": "fractal-KAN",
}


class MemoryBandSmear(MemoryBase):
    """A bank of N test-time memory cores, each a DIFFERENT function-class regime,
    combined by a REWARD-protected blend rather than a loss-trained router.

    Arm 0 is the profile's own memory net (``spec['dense']``, the exponential
    energy regime); each further arm swaps the memory net's function class to
    ``spec['dense_b']``, ``spec['dense_c']``, ... (e.g. the EML tree's
    ``e^x - Log(y)`` log-minus-exponent regime, or a geometric-grid KAN's
    multi-scale radial cascade). A router trained on the LM loss would collapse
    this: early on a granular core predicts worse, the gradient downweights it,
    and it is starved before it can mature (rich-get-richer). So the blend is NOT
    a learned router - it is a self-contained bandit with a floor:

      * Each core's reward is how well it forecasts the *same* NextLat target
        (its scale-free surprise; lower = better), so the arms are directly
        comparable. Each arm's share tracks a slow EMA of that reward.
      * The weights are DETACHED from the LM gradient (read off buffers), so the
        greedy loss can't collapse the mix; the cores' readouts still train
        through the blend, only the balance is reward-driven.
      * A floor on every arm means none can fully win or vanish - the "pull to
        center" is structural, so the opposed regimes loop on a stable simplex.

    N=2 reproduces the original dual EXACTLY: the inverse-surprise share
    ``(1/s_i) / Σ(1/s_j)`` is ``sa/(sa+sb)`` for two arms, and the floored weight
    is the same affine map. Each core keeps its own test-time state; the state is
    the tuple of per-core states.
    """

    metric_descriptions = {
        "memory_blend_b": {
            "description": (
                "Reward-driven weight on the second core (B) vs the exponential "
                "core A - a bandit over each core's forecast quality (surprise), "
                "floored so neither collapses. 0.5 = the regimes balance (the "
                "center); a slow rise means B is earning its granular keep; a fall "
                "toward the floor (0.1) means it is not. Not gradient-trained - it "
                "can't be starved by the loss."
            ),
            "chart": {
                "title": "Memory Blend (core B earned share)",
                "y_label": "weight on core B",
                "y_scale": "linear",
                "group": "memory",
                "group_order": 20,
                "order": 13,
            },
        },
        "memory_blend_c": {
            "description": (
                "Reward-driven weight on the third core (C, e.g. the geometric-KAN "
                "multi-scale radial regime) - same floored surprise bandit as core "
                "B. A rise means the third regime is winning forecast share; a fall "
                "to the floor means the other two carry it."
            ),
            "chart": {
                "title": "Memory Blend (core C earned share)",
                "y_label": "weight on core C",
                "y_scale": "linear",
                "group": "memory",
                "group_order": 20,
                "order": 15,
            },
        },
        "memory_regime_river": {
            "description": (
                "The memory regimes as a species-over-time river (after NEAT, "
                "Figure 7): time runs down the EMA horizon, each row split by the "
                "blend weights (band width = a regime's share), brightness = that "
                "regime's forecast fitness. The floor shows as a width no band "
                "falls below - protection made visible, extinction ruled out."
            ),
            "snapshot": {
                "title": "Memory Regime River",
                "renderer": "regime_river",
                "group": "memory",
                "group_order": 20,
                "order": 14,
            },
        },
    }

    def __init__(self, config, spec):
        super().__init__(config)
        # Arm 0 = spec['dense']; further arms = dense_b, dense_c, dense_d ...
        denses = [spec[k] for k in ("dense", "dense_b", "dense_c", "dense_d") if spec.get(k)]
        if len(denses) < 2:
            denses = (denses + ["eml_tree"])[:2]  # never fewer than two arms
        self._denses = denses

        # Sparse KAN: the geometric-grid KAN core is by far the most expensive to
        # run (spline matrix replicated per chunk as a fast weight, then a
        # test-time double-backward), and it runs at EVERY recurrent step. A
        # ``kan_sparse={period, phase}`` spec fires it only when
        # ``current_depth % period == phase`` - e.g. period=4, phase=3 runs it at
        # the 4th recurrent step and every 4th after (5 of 21 depths here), so
        # the other steps blend just the two cheap cores. It's a sparse
        # specialist: a few well-placed modules, not one per step. (With the vear
        # router the experts are parameter-merged, so structure must be identical
        # across them - the gate is a runtime skip, not a per-layer structural
        # change; recurrent step is the only stable, deterministic axis here.)
        rule = spec.get("kan_sparse")
        self._active_rule = []
        for d in denses:
            if d == "kan" and rule:
                self._active_rule.append((int(rule["period"]), int(rule["phase"])))
            else:
                self._active_rule.append(None)  # always on

        def _core(dense_name):
            s = {**spec, "dense": dense_name}
            return NeuralMemory(
                dim=self.hidden_size,
                model=build_memory_model(config, s),
                chunk_size=s.get("chunk_size", 64),
                momentum=s.get("momentum", True),
                use_energy=s.get("use_energy", False),
                segment=s.get("segment", False),
                segment_block=s.get("segment_block", 16),
                parallel_scan=s.get("parallel_scan", True),
                write_objective=s.get("write_objective", "recon"),
            )

        self.mems = nn.ModuleList([_core(d) for d in denses])
        # Slow EMAs of each core's surprise (forecast error on the shared NextLat
        # target). A buffer, so it carries no gradient and resumes cleanly. Init
        # equal -> the blend starts at the center (1/N each).
        self.register_buffer("values", torch.ones(len(denses)))
        self._labels = [
            f"{_REGIME_NAMES.get(d, d)} ({chr(65 + i)})" for i, d in enumerate(denses)
        ]
        self._last_weights: Optional[list] = None
        # Each arm's earned share the last time it was ACTIVE (a sparse arm skips
        # most steps, so its running metric would otherwise read 0 at the last
        # depth). None until the arm first fires.
        self._recent_weight: list = [None] * len(denses)
        # Rolling (weights, values) over exactly the EMA horizon, for the
        # regime-river card. Not a buffer (viz only, need not resume).
        self._history: deque = deque(maxlen=_RIVER_HORIZON)

    def _is_active(self, i: int, current_depth: int) -> bool:
        """Whether arm ``i`` runs at this recurrent step. Always-on unless it has
        a sparse rule (period, phase): active iff current_depth % period == phase."""
        rule = self._active_rule[i]
        return rule is None or (current_depth % rule[0]) == rule[1]

    def _blend_weights(self, active: list) -> list:
        """Per-arm weight from the inverse-surprise share (lower surprise = more
        weight), read off the detached value EMAs, over the ACTIVE arms only.
        Inactive (sparse-skipped) arms get weight 0; the active arms share the
        full mass, each floored. Scale-free; equal surprises -> 1/k each."""
        idx = [i for i, a in enumerate(active) if a]
        inv = torch.stack([1.0 / self.values[i].clamp_min(1e-8) for i in idx])
        share = inv / inv.sum()
        k = len(idx)
        w_active = _BLEND_FLOOR + (1.0 - k * _BLEND_FLOOR) * share
        w = [0.0] * len(active)
        for j, i in enumerate(idx):
            w[i] = float(w_active[j])
        return w

    def forward(self, stream, attn_output, state=None, current_depth: int = 0):
        states = list(state) if state is not None else [None] * len(self.mems)
        active = [self._is_active(i, current_depth) for i in range(len(self.mems))]
        # Act on the running estimate, then update it (standard bandit order).
        w = self._blend_weights(active)
        retrieved, new_states = None, []
        for i, mem in enumerate(self.mems):
            if not active[i]:
                new_states.append(states[i])  # skipped: no forward, state passes through
                continue
            r, si = mem(stream, states[i])
            new_states.append(si)
            contrib = w[i] * r
            retrieved = contrib if retrieved is None else retrieved + contrib
        self._last_weights = w
        with torch.no_grad():
            for i, mem in enumerate(self.mems):
                if not active[i]:
                    continue
                self._recent_weight[i] = w[i]
                s = mem.last_surprise_norm
                if s is not None:
                    self.values[i].mul_(_VALUE_EMA).add_((1.0 - _VALUE_EMA) * float(s))
            self._history.append((list(w), [float(v) for v in self.values]))
        if retrieved is None:  # no arm active (never, with A/B always on)
            return stream, tuple(new_states)
        return stream + retrieved, tuple(new_states)

    def dashboard_snapshots(self) -> dict:
        """The regime river: per-step (band widths, band fitnesses) over the EMA
        horizon. Fitness = surprise min-maxed across the window and inverted
        (lowest surprise = brightest), so brightness tracks forecast quality the
        way NEAT's brightness tracks species fitness. Row layout is
        ``[w_0..w_{N-1}, fit_0..fit_{N-1}]`` (N=2 -> [wa, wb, fa, fb])."""
        if not self._history:
            return {}
        weights = [h[0] for h in self._history]
        vals = [h[1] for h in self._history]
        flat = [v for row in vals for v in row]
        lo, hi = min(flat), max(flat)
        rng = (hi - lo) or 1.0
        fit = lambda v: 1.0 - (v - lo) / rng  # lower surprise -> brighter
        river = [
            weights[i] + [fit(v) for v in vals[i]] for i in range(len(weights))
        ]
        return {
            "memory_regime_river": {
                "status": "ok",
                "river": river,
                "labels": self._labels,
                "horizon": _RIVER_HORIZON,
            }
        }

    def _core_metrics(self, mem, prefix: str) -> dict:
        out = {}
        for attr, key in (
            ("last_surprise", "memory_surprise"),
            ("last_surprise_norm", "memory_surprise_norm"),
            ("last_gain", "memory_gain"),
            ("last_write", "memory_write"),
        ):
            v = getattr(mem, attr, None)
            if v is not None:
                out[f"{prefix}_{key}"] = float(v)
        return out

    def training_metrics(self) -> dict:
        out = {}
        for i, mem in enumerate(self.mems):
            out.update(self._core_metrics(mem, chr(97 + i)))  # a, b, c, ...
        # Arm 0 is the reference; report each further arm's earned share when it
        # last ran as memory_blend_b, memory_blend_c, ... A sparse arm reports
        # its most-recent active share (not 0 from a step it sat out).
        for i in range(1, len(self._recent_weight)):
            if self._recent_weight[i] is not None:
                out[f"memory_blend_{chr(ord('a') + i)}"] = self._recent_weight[i]
        return out


# Back-compat alias: the surfacing registry and older references use the "dual"
# name; the class is now N-arm (N=2 is byte-identical to the old dual).
MemoryDualSmear = MemoryBandSmear
