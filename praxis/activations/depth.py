"""Per-depth activation specialization: one activation instance per recurrent pass.

A recurrent stack reuses the same block at every depth, so a single activation
module serves every pass and learns one compromise shape for all of them. This
wrapper gives each pass its own instance - the same trick ``ArcGLU`` already
plays with its ``act`` ModuleList, lifted out so any dense module can hold it.

Cost is one parameter vector per pass per activation (Serpent: 3 x features), so
a depth-6 stack of 272 features pays ~5k parameters for six independently shaped
nonlinearities. Parameter-free activations (SiLU, GELU, ...) cannot specialize,
so they collapse back to a single shared instance and cost nothing.

Indexing matches ArcGLU: a block sees ``current_depth`` values
``{i, i + num_layers, ...}``, so the pass index is ``current_depth //
num_layers``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import UninitializedParameter

# `build_activation` lives in praxis.activations.__init__, which imports this
# module to register its profile - so every use of it here is a lazy import,
# matching praxis/activations/mixture.py.


def depth_passes(config) -> int:
    """How many times one block is revisited: ``ceil(depth / num_layers)``."""
    num_layers = max(1, int(getattr(config, "num_layers", 1) or 1))
    depth = max(1, int(getattr(config, "depth", 1) or 1))
    return max(1, math.ceil(depth / num_layers))


class DepthActivation(nn.Module):
    """An activation bank indexed by recurrent pass.

    Args:
        spec: activation spec (registry key, or a mixture declaration) - the
            same value ``build_activation`` takes.
        num_passes: how many instances to hold. 1 makes this a transparent
            wrapper around a single activation.
        num_layers: physical layers per pass, used to turn ``current_depth``
            into a pass index.
    """

    metric_descriptions = {
        "depth_act_specialization": {
            "description": (
                "Depth-specific fraction of the per-pass activation parameters. 0 = "
                "every pass learned identical params; rising = passes specializing."
            ),
            "chart": {
                "title": "Activation Depth Specialization",
                "y_label": "Specialized fraction",
                "y_scale": "linear",
                "group": "activation_depth",
                "group_order": 92,
                "order": 10,
            },
        },
        "depth_act_similarity": {
            "description": (
                "Mean pairwise cosine between the per-pass activation parameter "
                "vectors. ~1 = all passes converged to one shape; falling = diverging."
            ),
            "chart": {
                "title": "Activation Depth Similarity",
                "y_label": "Mean pairwise cosine",
                "y_scale": "linear",
                "group": "activation_depth",
                "order": 20,
            },
        },
    }

    def __init__(
        self,
        spec: Any,
        num_passes: int = 1,
        num_layers: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from praxis.activations import build_activation

        self.num_layers = max(1, int(num_layers))
        first = build_activation(spec, **kwargs)
        # A parameter-free activation is the same function at every depth, so N
        # copies would be N identical modules and N curves saying one thing.
        self.shared = not any(True for _ in first.parameters())
        count = 1 if self.shared else max(1, int(num_passes))
        self.passes = nn.ModuleList(
            [first] + [build_activation(spec, **kwargs) for _ in range(count - 1)]
        )
        # PEER asks whether the activation wants the expert index; the answer is
        # a property of the wrapped function, not of this wrapper.
        self.wants_keys: bool = getattr(first, "wants_keys", False)

    def pass_index(self, current_depth: int) -> int:
        return (int(current_depth) // self.num_layers) % len(self.passes)

    def _pending(self) -> bool:
        return any(isinstance(p, UninitializedParameter) for p in self.parameters())

    def forward(self, x: Tensor, current_depth: int = 0, **kwargs: Any) -> Tensor:
        # Serpent and friends are LAZY: they size themselves on their first
        # forward. Only the pass being run would materialize, so a depth that
        # does not execute on step 0 would still hold UninitializedParameters
        # when the optimizer collects parameters - and would then materialize
        # later, outside any param group, silently untrained. Run every pass
        # once instead, which is what PerDepthMTPBank does for the same reason.
        if self._pending():
            with torch.no_grad():
                for act in self.passes:
                    act(x, **kwargs)
        return self.passes[self.pass_index(current_depth)](x, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, *args: Any, **kwargs: Any):
        """Accept checkpoints written before this wrapper existed.

        Those stored the activation's parameters directly at this module's own
        prefix (``...act.a``); here they live one level down, per pass
        (``...act.passes.0.a``). Every pass starts from the single shape the old
        run had learned, which is the state the old model was actually in.
        """
        legacy = [
            key
            for key in list(state_dict)
            if key.startswith(prefix) and not key[len(prefix) :].startswith("passes.")
        ]
        if legacy and not any(k.startswith(f"{prefix}passes.") for k in state_dict):
            for key in legacy:
                suffix = key[len(prefix) :]
                value = state_dict.pop(key)
                for index in range(len(self.passes)):
                    state_dict[f"{prefix}passes.{index}.{suffix}"] = value.clone()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def training_metrics(self) -> dict:
        """Whether the passes are specializing or collapsing onto one shape.

        Empty when there is nothing to measure: a shared (parameter-free)
        activation, a single pass, or parameters not yet materialized - Serpent
        and friends are lazy, so this stays silent until the first forward.
        """
        from praxis.metrics.specialization import depth_dispersion

        if self.shared or len(self.passes) < 2:
            return {}
        rows = []
        for act in self.passes:
            params = list(act.parameters())
            if not params or any(isinstance(p, UninitializedParameter) for p in params):
                return {}
            rows.append(torch.cat([p.detach().reshape(-1) for p in params]))

        disp = depth_dispersion(torch.stack(rows, dim=0))
        if disp is None:
            return {}
        return {
            "depth_act_specialization": disp["specialization"],
            "depth_act_similarity": disp["similarity"],
        }

    def extra_repr(self) -> str:
        if self.shared:
            return "passes=1, shared=True"
        return f"passes={len(self.passes)}, num_layers={self.num_layers}"


def build_depth_activation(
    spec: Any,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> nn.Module:
    """Build an activation that specializes over depth when the config says so.

    Returns a plain activation when depth specialization is off or the stack is
    not recurrent, so a non-recurrent model carries no wrapper at all.
    """
    from praxis.activations import build_activation

    enabled = bool(getattr(config, "depth_activations", True)) if config else False
    passes = depth_passes(config) if config else 1
    if not enabled or passes < 2:
        return build_activation(spec, **kwargs)
    return DepthActivation(
        spec,
        num_passes=passes,
        num_layers=max(1, int(getattr(config, "num_layers", 1) or 1)),
        **kwargs,
    )


def base_activation(act: nn.Module) -> nn.Module:
    """The underlying activation, unwrapping :class:`DepthActivation`.

    Every pass is built from the same spec, so pass 0 answers any question about
    the function CLASS (its type, whether it wants keys, its harmonic spectrum).
    Use this for introspection; use the module itself to actually evaluate.
    """
    return act.passes[0] if isinstance(act, DepthActivation) else act
