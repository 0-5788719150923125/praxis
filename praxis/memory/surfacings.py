"""Titans memory surfacings: how a learned long-term memory branch is
combined with the residual stream.

``MemoryBase`` is a concrete no-op (identity forward, no metrics) and the
parent of the real surfacings, so a memory-free block carries a real object
instead of ``None`` - the block and decoder never branch on whether memory is
present. Each real surfacing wraps the shared ``NeuralMemory`` core:
- MAL applies memory as its own residual sub-layer.
- MAG blends a parallel memory branch with attention through a learned gate.
"""

from typing import Optional, Tuple, TypeVar

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
                "Reconstruction loss in RMS-normalized (directional) space, "
                "which is what the readout's out_norm consumes - the scale-free "
                "quantity the energy update actually optimizes. Unlike the raw "
                "surprise it isn't dominated by output-scale drift; falling = "
                "the memory is learning directional associations."
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
