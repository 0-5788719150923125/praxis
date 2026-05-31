"""Shared worker helpers for Mono-Forward backends.

Both the Ray-backed :class:`LayerActor` and the in-process
:class:`LocalLayerWorker` use identical optimizer / scheduler / param
plumbing. Keeping the helpers here (rather than in ``actor.py``, which
imports ``ray`` at module scope) lets the in-process backend reuse them
without forcing Ray into the import graph of a pure-PyTorch run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ActorParamShim(nn.Module):
    """Minimal ``nn.Module`` wrapper over one worker's (layer, projection) pair.

    :func:`praxis.optimizers.get_optimizer` calls
    :func:`pytorch_optimizer.create_optimizer`, which takes an
    ``nn.Module`` and walks ``named_parameters`` / ``named_modules``
    to apply the weight-decay ban list (LayerNorm, bias, etc.). Under
    MF each worker owns its layer and its per-layer projection matrix
    ``M_i``, so we need an intermediate container that exposes both
    as registered children.
    """

    def __init__(self, layer: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.projection = projection


def build_optimizer(
    shim: ActorParamShim,
    optimizer_config: Optional[Dict[str, Any]],
    wrappers: list,
    fallback_lr: float,
    criterion: nn.Module,
    strategy: Optional[Any],
) -> torch.optim.Optimizer:
    """Construct an optimizer for one worker's (layer, projection) params.

    When an ``optimizer_config`` dict is provided (i.e. the trainer
    was constructed via main.py), this routes through
    :func:`praxis.optimizers.get_optimizer` with the same wrapper keys the
    backprop path would use (``--optimizer`` / ``--optimizer-wrappers``), so
    they all honor their CLI selection under MF.

    When no config is provided (direct-construction unit tests),
    falls back to ``torch.optim.Adam(params, lr=fallback_lr)``.
    """
    if optimizer_config is None:
        params = list(shim.parameters())
        params += list(criterion.parameters())
        if strategy is not None and isinstance(strategy, nn.Module):
            params += list(strategy.parameters())
        return torch.optim.Adam(params, lr=fallback_lr)

    from praxis.optimizers import get_optimizer

    optimizer = get_optimizer(shim, wrappers=list(wrappers or []), **optimizer_config)
    extras: list = []
    extras += [p for p in criterion.parameters() if p.requires_grad]
    if strategy is not None and isinstance(strategy, nn.Module):
        extras += [p for p in strategy.parameters() if p.requires_grad]
    if extras:
        try:
            base_lr = optimizer.param_groups[0]["lr"]
            optimizer.add_param_group({"params": extras, "lr": base_lr})
        except Exception:
            pass
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    optimizer_config: Optional[Dict[str, Any]],
    warmup_steps: int,
    disable_schedule: bool,
) -> Optional[Any]:
    """Construct an LR scheduler for this worker's optimizer.

    Returns ``None`` when there's nothing to schedule (no config, or
    construction fails); callers should no-op the ``.step()`` in that
    case.
    """
    if optimizer_config is None:
        return None
    try:
        from praxis.schedulers import get_scheduler_func

        scheduler_func = get_scheduler_func(
            optimizer_config=optimizer_config,
            disable_schedule=disable_schedule,
            warmup_steps=max(int(warmup_steps), 1),
        )
        return scheduler_func(optimizer)
    except Exception:
        return None
