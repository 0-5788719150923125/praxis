"""Registry of optimizer wrappers, composed in sequence.

Each entry is a factory ``(optimizer) -> wrapped optimizer``; wrappers compose
by nesting (each one *is* an optimizer exposing ``.optimizer``), so
``SequentialWrapper`` just folds them on in order - the optimizer-side analog of
how ``SequentialHead`` composes heads. A factory may carry a
``disables_schedule`` attribute (the schedule-free family runs without an LR
schedule and boosts lr / moves weight decay to ``z``); ``SequentialWrapper``
surfaces that so the scheduler can be built accordingly.

Replaces the old ``--trac/--ortho/--lookahead/--schedule-free`` boolean flags.
"""

from typing import Iterable, List

from pytorch_optimizer.optimizer import TRAC, Lookahead, OrthoGrad, ScheduleFreeWrapper

from praxis.optimizers.gated_schedule_free import GatedScheduleFree
from praxis.optimizers.half_lion import HalfLion
from praxis.optimizers.low_rank_moment import LowRankSecondMoment
from praxis.optimizers.wave_schedule_free import WaveScheduleFree

SCHEDULE_FREE_MOMENTUM = 0.98


def _trac(optimizer):
    return TRAC(optimizer, num_coefs=128)


def _ortho(optimizer):
    return OrthoGrad(optimizer)


def _lookahead(optimizer):
    return Lookahead(optimizer, k=5, alpha=0.5, pullback_momentum="none")


def _schedule_free_prep(optimizer) -> float:
    """Schedule-free wants a boosted lr and weight decay applied at ``z`` (not
    in the base optimizer). Boost every group's lr 2x and capture+zero the base
    weight decay; return the captured value for the wrapper. Equivalent to the
    old pre-creation handling (lr*2 commutes with tasker-LR promotion)."""
    wd = 0.0
    for group in optimizer.param_groups:
        group["lr"] *= 2.0
        wd = max(wd, float(group.get("weight_decay", 0.0) or 0.0))
        group["weight_decay"] = 0.0
    return wd


def _schedule_free(optimizer):
    wd = _schedule_free_prep(optimizer)
    return ScheduleFreeWrapper(
        optimizer, momentum=SCHEDULE_FREE_MOMENTUM, r=0, weight_decay=wd
    )


_schedule_free.disables_schedule = True


def _gated_schedule_free(optimizer):
    wd = _schedule_free_prep(optimizer)
    return GatedScheduleFree(
        optimizer, momentum=SCHEDULE_FREE_MOMENTUM, weight_decay=wd
    )


_gated_schedule_free.disables_schedule = True


def _wave_schedule_free(optimizer):
    wd = _schedule_free_prep(optimizer)
    return WaveScheduleFree(optimizer, momentum=SCHEDULE_FREE_MOMENTUM, weight_decay=wd)


_wave_schedule_free.disables_schedule = True


def _half_lion(optimizer):
    return HalfLion(optimizer)


def _low_rank_moment(optimizer):
    return LowRankSecondMoment(optimizer)


WRAPPER_REGISTRY = {
    "trac": _trac,  # mitigate plasticity loss over time
    "ortho": _ortho,  # gradients orthogonal to params
    "lookahead": _lookahead,  # slow/fast weight interpolation
    "schedule_free": _schedule_free,  # Polyak averaging, no LR schedule
    "gated_schedule_free": _gated_schedule_free,  # per-coordinate SNR-gated averaging
    "wave_schedule_free": _wave_schedule_free,  # standing-wave gate over param index
    "half_lion": _half_lion,  # blend live weights with frozen init
    "low_rank_moment": _low_rank_moment,  # factored second-moment telemetry
}


class SequentialWrapper:
    """Fold a sequence of registry wrappers onto a base optimizer, in order
    (innermost first). The result is the nested optimizer (wrappers self-nest)."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys: List[str] = list(keys or [])
        unknown = [k for k in self.keys if k not in WRAPPER_REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown optimizer wrapper(s) {unknown}; "
                f"choices: {sorted(WRAPPER_REGISTRY)}"
            )

    @property
    def disables_schedule(self) -> bool:
        return any(
            getattr(WRAPPER_REGISTRY[k], "disables_schedule", False) for k in self.keys
        )

    def __call__(self, optimizer):
        for key in self.keys:
            optimizer = WRAPPER_REGISTRY[key](optimizer)
        return optimizer


def wrappers_disable_schedule(keys: Iterable[str]) -> bool:
    """True if any selected wrapper runs without an LR schedule."""
    return SequentialWrapper(keys).disables_schedule
