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
        "memory_adapt": {
            "description": (
                "What the test-time update changed in the READOUT: "
                "||read(W_T) - read(W0)|| / ||read(W0)||, the same write Memory "
                "Write measures, in function space instead of weight space. Read "
                "the two together. Memory Write divides by a ||W0|| that grows "
                "as the trunk trains, so a step of unchanged size reads as a "
                "falling ratio - scaling the memory net by k divides it by "
                "exactly k, news about the weights and not about the memory. "
                "This denominator instead sits behind out_norm's RMS "
                "normalization and is pinned regardless of that scale, so what "
                "moves here is the numerator: how far the trunk's view of the "
                "memory actually shifted between the pre- and post-write "
                "weights. 0 means the writes changed nothing the trunk can see."
            ),
            "chart": {
                "title": "Memory Adaptation",
                "y_label": "delta-read / read",
                "y_scale": "linear",
                "group": "memory",
                "order": 14,
            },
        },
        "memory_gate": {
            "description": (
                "Mean opening of the MAG gate: how much of the stream the model "
                "replaces with the memory's readout, averaged over channels and "
                "positions. This is the verdict line - the model's own answer to "
                "whether it wants the memory, in one number. It starts near 0.05 "
                "(bias -3) so the module eases in from a near-identity, and a "
                "decay back toward 0 is the model routing around it, stated "
                "plainly instead of hidden in a residual it has to cancel "
                "downstream. Read against Memory Adaptation: a rising gate with "
                "adaptation near 0 means the model wants the memory net as a "
                "static function, not as a memory."
            ),
            "chart": {
                "title": "Memory Gate",
                "y_label": "gate opening",
                "y_scale": "linear",
                "group": "memory",
                "order": 9,
            },
        },
        "memory_chunks": {
            "description": (
                "Chunks the store pass resolved the sequence into. This is the "
                "ceiling on every other line in this group: retrieval reads "
                "PRE-write weights, so a chunk's update is only ever visible to "
                "the chunks after it and the number of writes the model can "
                "actually feel is chunks - 1. At 1 the memory is a static "
                "readout at the cold weights, the update is computed and "
                "discarded, and Memory Adaptation is exactly 0 while Gain and "
                "Write still look healthy - read this line before concluding "
                "anything from those. Set by the sequence length against the "
                "update grid (segment_block, or chunk_size when unsegmented), "
                "so it moves with the sequence-length curriculum."
            ),
            "chart": {
                "title": "Memory Chunks",
                "y_label": "chunks / store pass",
                "y_scale": "linear",
                "group": "memory",
                "order": 10,
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
                "order": 15,
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
                "order": 16,
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
                "order": 17,
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
        # Which recurrent passes run the memory. None = every pass, the old
        # behaviour. A list keys the memory to the PASS index
        # (``current_depth // num_layers``), the same unit MemoryDepthBank uses
        # and the only unit halting can cut at: KL checks fire at loop
        # boundaries and training samples a loop count up front, so pass 0 is
        # the only station every input reaches and every gradient step trains.
        # Anything deeper is seen with the halting distribution's own frequency,
        # which is how the depth bank's late cores starved.
        passes = spec.get("passes")
        self.passes = None if passes is None else frozenset(int(p) for p in passes)
        self.num_layers = max(1, int(getattr(config, "num_layers", 1) or 1))

    def _runs_at(self, current_depth: int) -> bool:
        """Whether the memory fires at this recurrent step."""
        if self.passes is None:
            return True
        return (int(current_depth) // self.num_layers) in self.passes

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
        if m.last_adapt is not None:
            out["memory_adapt"] = float(m.last_adapt)
        if m.last_num_chunks is not None:
            out["memory_chunks"] = float(m.last_num_chunks)
        if m.last_event_mean is not None:
            out["memory_event_size"] = float(m.last_event_mean)
            out["memory_event_min"] = float(m.last_event_min)
            out["memory_event_max"] = float(m.last_event_max)
        return out


class MemoryAsLayer(MemorySurfacing):
    """MAL: memory as its own residual sub-layer within the block."""

    def forward(self, stream, attn_output, state=None, current_depth: int = 0):
        if not self._runs_at(current_depth):
            return stream, state
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
        # Mean gate opening from the last firing. Unlike MAL's full-weight
        # residual add - which the trunk can only neutralize by cancelling it
        # downstream, invisibly - this is the model's own answer to "do I want
        # the memory", as one number.
        self.last_gate: Optional[Tensor] = None

    def forward(self, stream, attn_output, state=None, current_depth: int = 0):
        if not self._runs_at(current_depth):
            return stream, state
        retrieved, state = self.mem(stream, state)
        g = self.gate(stream).sigmoid()
        with torch.no_grad():
            self.last_gate = g.mean()
        return g * retrieved + (1 - g) * stream, state

    def training_metrics(self) -> dict:
        out = super().training_metrics()
        if self.last_gate is not None:
            out["memory_gate"] = float(self.last_gate)
        return out


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
    "spline": "knot-spline",
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
        "memory_blend_d": {
            "description": (
                "Reward-driven weight on the fourth core (D, e.g. the learned-"
                "knot spline's adaptive-resolution regime) - same floored "
                "surprise bandit as cores B/C. Read against memory_blend_c: "
                "D earning share the fixed geometric grid loses is the direct "
                "fixed-vs-learned-placement verdict."
            ),
            "chart": {
                "title": "Memory Blend (core D earned share)",
                "y_label": "weight on core D",
                "y_scale": "linear",
                "group": "memory",
                "group_order": 20,
                "order": 16,
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
        denses = [
            spec[k] for k in ("dense", "dense_b", "dense_c", "dense_d") if spec.get(k)
        ]
        if len(denses) < 2:
            denses = (denses + ["eml_tree"])[:2]  # never fewer than two arms
        self._denses = denses

        # Sparse arms: grid-basis cores (KAN, spline) are by far the most
        # expensive to run (basis matrix replicated per chunk as a fast weight,
        # then a test-time double-backward), and by default every arm runs at
        # EVERY recurrent step. A ``sparse={dense_name: {period, phase}}`` spec
        # fires such an arm only when ``current_depth % period == phase`` - e.g.
        # period=4, phase=3 runs it at the 4th recurrent step and every 4th
        # after (5 of 21 depths); staggered phases keep at most one expensive
        # core per step. On skipped steps the blend renormalizes over the active
        # arms. It's a sparse specialist: a few well-placed modules, not one per
        # step. (With the vear router the experts are parameter-merged, so
        # structure must be identical across them - the gate is a runtime skip,
        # not a per-layer structural change; recurrent step is the only stable,
        # deterministic axis here.) ``kan_sparse`` is the back-compat spelling
        # of ``sparse={"kan": ...}``.
        rules = dict(spec.get("sparse") or {})
        if spec.get("kan_sparse"):
            rules.setdefault("kan", spec["kan_sparse"])
        self._active_rule = []
        for d in denses:
            rule = rules.get(d)
            self._active_rule.append(
                (int(rule["period"]), int(rule["phase"])) if rule else None
            )

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
                new_states.append(
                    states[i]
                )  # skipped: no forward, state passes through
                continue
            r, si = mem(stream, states[i])
            new_states.append(si)
            contrib = w[i] * r
            retrieved = contrib if retrieved is None else retrieved + contrib
        self._last_weights = w
        # Training-only: the bandit's reward EMA is model STATE, and _blend_weights
        # reads it back at :446, so advancing it on an inference forward changes
        # what the next inference forward computes. Two identical forwards then
        # disagree (measured 3.5e-04, and 7.2e-03 through prismatic5's crystal
        # route, against 0.0e+00 with memory_type: none), which is a real problem
        # for the byte-latent speculative decoder: _spec_logits_and_hidden and
        # _verify_prefixes_batched are SEPARATE forwards, so the verify scores the
        # drafts under a blend the drafting pass already moved. That turns the
        # greedy-lossless guarantee into lossless-up-to-drift, and it is drift the
        # accept test (an exact id comparison, modeling.py:1244) reads as a
        # divergence, so it silently costs accepted bytes. The arms' forecast
        # quality is a property of the training distribution anyway; generation
        # should read the mix, not rewrite it.
        if self.training:
            with torch.no_grad():
                for i, mem in enumerate(self.mems):
                    if not active[i]:
                        continue
                    self._recent_weight[i] = w[i]
                    s = mem.last_surprise_norm
                    if s is not None:
                        self.values[i].mul_(_VALUE_EMA).add_(
                            (1.0 - _VALUE_EMA) * float(s)
                        )
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
        river = [weights[i] + [fit(v) for v in vals[i]] for i in range(len(weights))]
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
            ("last_adapt", "memory_adapt"),
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


# Bank positions are POSITIONAL: core i is whatever ``dense``/``dense_b``/... the
# profile names at that slot. The chart series are labelled by letter for that
# reason - a profile that reorders its bank would make a name-based legend lie,
# while the letter stays true. The regime NAMES ride the river card instead,
# where they are read off the live spec (_REGIME_NAMES).
_BANK_LETTERS = "abcdefgh"


def _per_core_charts(
    suffix: str,
    title: str,
    y_label: str,
    description: str,
    base_order: int,
    y_scale: str = "linear",
    arms: int = 4,
) -> dict:
    """One chart per metric family, one line per bank position.

    The class declares five families over a four-core bank; twenty separate
    cards would bury the thing worth seeing (the regimes side by side), so each
    family shares a ``series_group`` and renders as one multi-line chart. Keys
    the run never emits are pruned by the frontend, so declaring the full bank
    here costs nothing to a profile that holds fewer arms.
    """
    return {
        f"{letter}_{suffix}": {
            "description": description,
            "chart": {
                "title": title,
                "y_label": y_label,
                "y_scale": y_scale,
                "group": "memory",
                "group_order": 20,
                "order": base_order + i,
                "series_group": f"depth_bank_{suffix}",
                "series_label": letter.upper(),
            },
        }
        for i, letter in enumerate(_BANK_LETTERS[:arms])
    }


class MemoryDepthBank(MemoryBase):
    """ONE test-time memory core per recurrent pass, drawn from a bank of N
    function-class regimes - the depth axis IS the router.

    ``MemoryBandSmear`` stacks its bank at every step: each arm runs and the
    outputs are blended, so the cheap arms cost N memory forwards and N
    test-time updates per recurrent step (the sparse rules trim the grid arms,
    but the two cheap ones are always on - never fewer than 2 cores per step).
    Here the bank is spread ALONG the recurrence instead. Pass p runs core
    ``p % N`` and nothing else, so step cost is exactly one core no matter how
    many regimes the bank holds, and each regime specializes to its own station
    in the recurrence rather than competing for the same one.

    The assignment is keyed to the PASS index (``current_depth // num_layers``)
    because that is the unit halting can actually cut at: the KL check only
    fires at loop boundaries (praxis/halting/kl.py:154), and training does not
    check at all - it samples a loop COUNT up front (kl.py:122-133). So the bank
    is declared cheapest-first and the pass a core sits at is the price of
    reaching it:

      * pass 0's core runs on every forward - the memory the model always has.
      * later cores are reached only when the pass budget goes that deep. In
        training that is the log-normal Poisson's tail; at inference it is
        inputs whose latent has not converged by then. Either way an easy input
        never pays for the expensive regimes at all, which is the saving - and
        the same fact means a late core sees proportionally fewer gradient
        steps, which is the cost. Read ``*_memory_core_use`` for the actual
        split; it is the experiment, not a diagnostic.

    There is no bandit and no blend here, deliberately. The band smear's arms
    are comparable because they forecast the SAME NextLat target from the same
    stream; these read a different depth's stream each, so an inverse-surprise
    share between them would be measuring depth, not forecast quality. Routing
    is therefore a pure function of ``current_depth`` - bit-identical on every
    forward, which the byte-latent speculative decoder needs (the note at
    :472 records what a state-mutating blend cost it). Nothing this class
    tracks feeds the output; it is all diagnostic.
    """

    metric_descriptions = {
        "memory_depth_river": {
            "description": (
                "The bank as a species-over-time river (after NEAT, Figure 7): "
                "time runs down the recent-forwards horizon, each row split by "
                "how many of that forward's passes each core actually ran (band "
                "width = occupancy, NOT a blend weight - only one core runs per "
                "pass). Brightness is that core's forecast quality against its "
                "OWN recent range, since the cores read different depths and do "
                "not share a scale. A late band narrowing to nothing is the "
                "model exiting before it ever reaches that regime."
            ),
            "snapshot": {
                "title": "Memory Depth Bank",
                "renderer": "regime_river",
                "group": "memory",
                "group_order": 20,
                "order": 5,
            },
        },
        **_per_core_charts(
            "memory_core_use",
            "Memory Core Use",
            "passes run / passes taken",
            (
                "Share of the forward's recurrent passes that ran this core - "
                "fixed by the assignment (pass p runs core p % N) and by how "
                "deep the pass budget went, so the four lines are the halting "
                "distribution read through the bank. A shallow forward pushes "
                "core A's share toward 1 and pins the late cores at 0; a "
                "forward that completes the cycle settles every share at 1/N. "
                "A late core sitting near 0 is a regime the model never pays "
                "for - the compute saving, and equally the reason it may never "
                "mature."
            ),
            base_order=10,
        ),
        **_per_core_charts(
            "memory_surprise_norm",
            "Memory Core Surprise (norm)",
            "surprise (normalized)",
            (
                "Per-core surprise in RMS-normalized space - the scale-free "
                "next-latent (Huber) forecast error the energy update actually "
                "optimizes, reported for the core that ran. NOT comparable "
                "across cores the way the band smear's arms are: each sits at a "
                "different recurrent pass and reads a different stream. Read "
                "each line against its own history."
            ),
            base_order=20,
        ),
        **_per_core_charts(
            "memory_surprise",
            "Memory Core Surprise (raw)",
            "surprise",
            (
                "Per-core RAW reconstruction loss at the cold init weights. "
                "Scale-sensitive (a core's free output scale can dominate it); "
                "the normalized chart is the quantity the update optimizes."
            ),
            base_order=30,
            y_scale="logarithmic",
        ),
        **_per_core_charts(
            "memory_gain",
            "Memory Core Gain",
            "retrieved / stream",
            (
                "Per-core output magnitude relative to the residual stream at "
                "its own pass. Unlike the band smear there is no blend to "
                "divide the contribution, so a core writes at full weight when "
                "it runs; decay toward 0 means the model is routing around that "
                "regime rather than around the memory as a whole."
            ),
            base_order=40,
        ),
        **_per_core_charts(
            "memory_write",
            "Memory Core Write",
            "delta-W / W0",
            (
                "Per-core relative size of the test-time weight update "
                "(||W_T - W0|| / ||W0||). A small value is NOT by itself an "
                "inert update: the denominator is the core's meta-learned "
                "weights, which grow as the trunk trains, so a step of "
                "unchanged size reads as a falling ratio. Memory Core "
                "Adaptation is the same write without that confound."
            ),
            base_order=50,
        ),
        **_per_core_charts(
            "memory_adapt",
            "Memory Core Adaptation",
            "delta-read / read",
            (
                "Per-core effect of the test-time update on the READOUT: "
                "||read(W_T) - read(W0)|| / ||read(W0)||, the write measured in "
                "function space. This is the line that separates a core doing "
                "gentle in-context work from one whose update the readout cannot "
                "feel - Memory Core Write cannot, since it falls by 1/k whenever "
                "the core's weights grow k-fold against a fixed update step. The "
                "denominator here is RMS-normalized by out_norm, so it holds "
                "still under that growth and a fall is a real one. Sampled on a "
                "cadence, so it moves in steps rather than per-batch."
            ),
            base_order=60,
        ),
    }

    def __init__(self, config, spec) -> None:
        super().__init__(config)
        denses = [
            spec[k] for k in ("dense", "dense_b", "dense_c", "dense_d") if spec.get(k)
        ]
        if len(denses) < 2:
            raise ValueError(
                "depth_bank needs at least two cores (dense + dense_b); a "
                "one-core bank is just the mal/mal_energy surfacing."
            )
        self._denses = denses

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
        # The pass index, not the raw depth: halting can only cut on a whole
        # num_layers cycle, so keying the bank to the cycle makes every possible
        # exit land on a core boundary. With num_layers=1 the two coincide.
        self.num_layers = max(1, int(getattr(config, "num_layers", 1) or 1))
        max_loops = max(1, int(getattr(config, "depth", 1) or 1)) // self.num_layers
        if max_loops < len(denses):
            unreachable = ", ".join(denses[max_loops:])
            print(
                f"[MEMORY] depth_bank: {len(denses)} cores but only {max_loops} "
                f"recurrent pass(es) - {unreachable} can never run. Drop the "
                "unreachable arms from the profile or raise depth."
            )
        self._labels = [
            f"{_REGIME_NAMES.get(d, d)} ({chr(65 + i)})" for i, d in enumerate(denses)
        ]
        # Deepest pass index reached in the forward being assembled, the raw
        # depth of the previous call (how _track finds a forward boundary), and
        # the per-core diagnostics captured while each core ran.
        self._passes_seen = 0
        self._last_depth: Optional[int] = None
        self._core_stats: list = [None] * len(denses)
        # Last surprise each core reported while ACTIVE - the river's brightness
        # source, kept across forwards so a core that sat this one out still
        # paints its band rather than dropping to a fake zero.
        self._core_surprise: list = [None] * len(denses)
        self._history: deque = deque(maxlen=_RIVER_HORIZON)

    def _core_index(self, current_depth: int) -> int:
        """Which core runs at this recurrent step. Wraps, so a controller that
        overruns config.depth still lands on a real core."""
        return (int(current_depth) // self.num_layers) % len(self.mems)

    def _use_fractions(self, passes: int) -> list:
        """Share of the forward's passes that ran each core. Exactly one core
        runs per pass, so these sum to 1 - which is what the river renderer
        assumes of its band widths."""
        counts = [0] * len(self.mems)
        for p in range(passes):
            counts[p % len(self.mems)] += 1
        return [c / passes for c in counts]

    def _settle_forward(self) -> None:
        """Fold the finished forward into the river. There is no pass-end
        callback, so the next forward's first pass settles the previous one -
        the same idiom KL halting uses for its peak EMA
        (praxis/halting/kl.py:105). Held back until every core has reported
        once, so the card starts with real fitnesses rather than filler."""
        if self._passes_seen <= 0 or any(s is None for s in self._core_surprise):
            return
        self._history.append(
            (self._use_fractions(self._passes_seen), list(self._core_surprise))
        )

    def _track(self, core: int, depth: int) -> None:
        """Per-forward accounting for the cards, and the ONLY place this class
        writes state.

        Training-only, deliberately. Generation runs INSIDE the training loop
        (praxis/callbacks/lightning/generation_queue.py drains the queue from
        on_train_batch_end, and the decode backend flips the model to eval), at
        whatever depth the KL early exit picks - so an ungated counter would let
        a decode overwrite the training forward's occupancy and the river would
        paint the decode instead of the run.

        A forward starts at pass 0 that did not ADVANCE from the previous call.
        That boundary holds however the decoder hands out depths: a shared
        expert-bank block sees every depth (so pass 0 spans num_layers
        consecutive, increasing depths - not a new forward), while distinct
        LocalLayers see only their own residue class (so block j never sees
        depth 0 at all and its first pass starts at depth j). Under gradient
        checkpointing the recompute replays depths in reverse; with num_layers
        1 it cannot reach pass 0, since depth 0 is the one depth
        ``should_checkpoint`` always declines. Everything else here is
        idempotent under replay anyway - _passes_seen only takes a max, and the
        stats are rewritten with the values they already hold.
        """
        pass_index = depth // self.num_layers
        if pass_index == 0 and (self._last_depth is None or depth <= self._last_depth):
            self._settle_forward()
            self._passes_seen = 0
            self._core_stats = [None] * len(self.mems)
        self._last_depth = depth
        self._passes_seen = max(self._passes_seen, pass_index + 1)
        # Snapshot the core's diagnostics WHILE they are fresh. NeuralMemory's
        # last_* attributes are overwritten by every call and reset nowhere, so
        # reading them after the forward would report a core that sat this one
        # out as if it had just run.
        self._core_stats[core] = self._core_metrics(self.mems[core])
        surprise = self._core_stats[core].get("memory_surprise_norm")
        if surprise is not None:
            self._core_surprise[core] = surprise

    def forward(self, stream, attn_output, state=None, current_depth: int = 0):
        states = list(state) if state is not None else [None] * len(self.mems)
        i = self._core_index(current_depth)
        retrieved, states[i] = self.mems[i](stream, states[i])
        if self.training:
            self._track(i, int(current_depth))
        return stream + retrieved, tuple(states)

    def dashboard_snapshots(self) -> dict:
        """The depth bank as a river: per-forward (band widths, band fitness).
        Widths are occupancy - the share of passes that reached each core - so a
        halted forward simply has no late bands. Fitness is min-maxed WITHIN a
        band and inverted, not across bands: the cores read different depths, so
        a shared normalization would paint depth rather than forecast quality.
        Row layout is ``[use_0..use_{N-1}, fit_0..fit_{N-1}]``, the layout the
        regime_river renderer reads."""
        if not self._history:
            return {}
        uses = [h[0] for h in self._history]
        vals = [h[1] for h in self._history]
        n = len(self.mems)
        fits = [[0.5] * n for _ in vals]
        for i in range(n):
            col = [row[i] for row in vals]
            lo, hi = min(col), max(col)
            rng = hi - lo
            if rng <= 0:
                continue  # flat band: leave it mid-bright rather than all-or-nothing
            for r, v in enumerate(col):
                fits[r][i] = 1.0 - (v - lo) / rng  # lower surprise -> brighter
        return {
            "memory_depth_river": {
                "status": "ok",
                "river": [uses[r] + fits[r] for r in range(len(uses))],
                "labels": self._labels,
                "horizon": _RIVER_HORIZON,
            }
        }

    def _core_metrics(self, mem) -> dict:
        out = {}
        for attr, key in (
            ("last_surprise", "memory_surprise"),
            ("last_surprise_norm", "memory_surprise_norm"),
            ("last_gain", "memory_gain"),
            ("last_write", "memory_write"),
            ("last_adapt", "memory_adapt"),
        ):
            v = getattr(mem, attr, None)
            if v is not None:
                out[key] = float(v)
        return out

    def training_metrics(self) -> dict:
        """Per-core diagnostics for the cores that ran in the last TRAINING
        forward, plus every core's share of that forward's passes.

        The two families are deliberately asymmetric. Occupancy is reported for
        every core, including the unreached ones - a hard 0 there is the signal
        the experiment is read on. The surprise/gain/write families are dropped
        for a core the forward never reached, since NeuralMemory's last_*
        attributes are never cleared and would otherwise repeat a value from an
        earlier step; the collector skips absent keys, so those lines just go
        sparse where the model exited above the core."""
        out = {}
        for i, stats in enumerate(self._core_stats):
            if not stats:
                continue
            letter = _BANK_LETTERS[i]
            out.update({f"{letter}_{key}": value for key, value in stats.items()})
        if self._passes_seen > 0:
            for i, use in enumerate(self._use_fractions(self._passes_seen)):
                out[f"{_BANK_LETTERS[i]}_memory_core_use"] = use
        return out
