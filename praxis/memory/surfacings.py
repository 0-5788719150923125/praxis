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

    def forward(self, stream, attn_output, state=None):
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

    def forward(self, stream, attn_output, state=None):
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

    def forward(self, stream, attn_output, state=None):
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


class MemoryDualSmear(MemoryBase):
    """Two test-time memory cores of DIFFERENT function-class regimes, combined
    by a REWARD-protected blend rather than a loss-trained router.

    Core A is the profile's memory (the exponential energy regime); core B swaps
    the memory net's function class to ``spec['dense_b']`` (e.g. the EML tree's
    ``e^x - Log(y)``, the log-minus-exponent regime). A router trained on the LM
    loss would collapse this: early on the granular EML core predicts worse, the
    gradient downweights it, and it is starved before it can mature (rich-get-
    richer). So the blend weight is NOT a learned router - it is a self-contained
    bandit with a floor:

      * The reward each core earns is how well it forecasts the *same* NextLat
        target (its scale-free surprise; lower = better), so the two are directly
        comparable. Core B's share tracks a slow EMA of that reward.
      * The weight is DETACHED from the LM gradient (read off buffers), so the
        greedy loss can't collapse the mix; the cores' readouts still train
        through the blend, only the balance is reward-driven.
      * A floor on both cores means neither can fully win - the "pull to center"
        is structural, not a hope, so the opposed regimes loop on a stable axis
        instead of one warping outward.

    Each core keeps its own test-time state; the state is the pair.
    """

    metric_descriptions = {
        "memory_blend_b": {
            "description": (
                "Reward-driven weight on the EML (log-minus-exponent) core B vs "
                "the exponential core A - a bandit over each core's forecast "
                "quality (surprise), floored so neither collapses. 0.5 = the two "
                "regimes balance (the center); a slow rise above 0.5 means EML is "
                "earning its granular keep; a fall toward the floor (0.1) means it "
                "is not. Not gradient-trained - it can't be starved by the loss."
            ),
            "chart": {
                "title": "Dual-Memory Blend (EML earned share)",
                "y_label": "weight on core B (EML)",
                "y_scale": "linear",
                "group": "memory",
                "group_order": 20,
                "order": 13,
            },
        },
        "memory_regime_river": {
            "description": (
                "The two memory regimes as a species-over-time river (after NEAT, "
                "Figure 7): time runs down the EMA horizon, each row split by the "
                "blend weight (band width = a regime's share), brightness = that "
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
        spec_b = {**spec, "dense": spec.get("dense_b", "eml_tree")}

        def _core(s):
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

        self.mem_a = _core(spec)
        self.mem_b = _core(spec_b)
        # Slow EMAs of each core's surprise (its forecast error on the shared
        # NextLat target). Buffers, so they carry no gradient and resume cleanly.
        # Init equal -> the blend starts at the center (0.5).
        self.register_buffer("value_a", torch.ones(()))
        self.register_buffer("value_b", torch.ones(()))
        self._last_blend_b: Optional[float] = None
        # Rolling (weight_b, value_a, value_b) over exactly the EMA horizon, for
        # the regime-river card. Not a buffer (viz only, need not resume).
        self._history: deque = deque(maxlen=_RIVER_HORIZON)

    def _blend_weight_b(self) -> float:
        """Core B's weight in [floor, 1-floor] from the inverse-surprise share
        (lower surprise = more weight), read off the detached value EMAs."""
        sa = float(self.value_a)
        sb = float(self.value_b)
        # Inverse-surprise share: b's fraction of forecast quality. Scale-free,
        # so no gain/temperature to tune. Equal surprises -> 0.5.
        share_b = sa / (sa + sb + 1e-8)
        return _BLEND_FLOOR + (1.0 - 2.0 * _BLEND_FLOOR) * share_b

    def forward(self, stream, attn_output, state=None):
        sa, sb = state if state is not None else (None, None)
        ret_a, sa = self.mem_a(stream, sa)
        ret_b, sb = self.mem_b(stream, sb)
        # Act on the running estimate, then update it (standard bandit order).
        w_b = self._blend_weight_b()
        retrieved = (1.0 - w_b) * ret_a + w_b * ret_b
        self._last_blend_b = w_b
        with torch.no_grad():
            for buf, mem in ((self.value_a, self.mem_a), (self.value_b, self.mem_b)):
                s = mem.last_surprise_norm
                if s is not None:
                    buf.mul_(_VALUE_EMA).add_((1.0 - _VALUE_EMA) * float(s))
            self._history.append((w_b, float(self.value_a), float(self.value_b)))
        return stream + retrieved, (sa, sb)

    def dashboard_snapshots(self) -> dict:
        """The regime river: per-step (band widths, band fitnesses) over the EMA
        horizon. Fitness = surprise min-maxed across the window and inverted
        (lowest surprise = brightest), so brightness tracks forecast quality the
        way NEAT's brightness tracks species fitness."""
        if not self._history:
            return {}
        wb = [h[0] for h in self._history]
        va = [h[1] for h in self._history]
        vb = [h[2] for h in self._history]
        allv = va + vb
        lo, hi = min(allv), max(allv)
        rng = (hi - lo) or 1.0
        fit = lambda v: 1.0 - (v - lo) / rng  # lower surprise -> brighter
        # Each row: [width_a, width_b, fitness_a, fitness_b].
        river = [[1.0 - wb[i], wb[i], fit(va[i]), fit(vb[i])] for i in range(len(wb))]
        return {
            "memory_regime_river": {
                "status": "ok",
                "river": river,
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
        out.update(self._core_metrics(self.mem_a, "a"))
        out.update(self._core_metrics(self.mem_b, "b"))
        if self._last_blend_b is not None:
            out["memory_blend_b"] = self._last_blend_b
        return out
