"""Registry of optimizer wrappers, composed in sequence.

Each entry is a factory ``(optimizer) -> wrapped optimizer``; wrappers compose
by nesting (each one *is* an optimizer exposing ``.optimizer``), so
``SequentialWrapper`` just folds them on in order - the optimizer-side analog of
how ``SequentialClassifier`` composes classifiers. A factory may carry a
``disables_schedule`` attribute (the schedule-free family runs without an LR
schedule and boosts lr / moves weight decay to ``z``); ``SequentialWrapper``
surfaces that so the scheduler can be built accordingly.

Replaces the old ``--trac/--ortho/--lookahead/--schedule-free`` boolean flags.
"""

from typing import Iterable, List

from pytorch_optimizer.optimizer import TRAC, Lookahead, OrthoGrad, ScheduleFreeWrapper

from praxis import registry
from praxis.optimization.gated_schedule_free import GatedScheduleFree
from praxis.optimization.half_lion import HalfLion
from praxis.optimization.low_rank_moment import LowRankSecondMoment
from praxis.optimization.wave_schedule_free import WaveScheduleFree
from praxis.registry import Entry

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


registry.declare(
    "wrappers",
    title="Optimizer wrappers",
    doc=(
        (
            "Composable wrappers layered onto the base optimizer, listed in order and "
            "applied innermost-first. Each entry is a factory ``(optimizer) -> wrapped "
            "optimizer``, and wrappers nest, so any stack is itself an optimizer. The "
            "schedule-free family runs without an LR schedule; the others keep it."
        )
    ),
    entries={
        "trac": Entry(
            _trac,
            (
                "TRAC: tunes a per-parameter learning-rate scale online to mitigate "
                "loss of plasticity over long training runs."
            ),
        ),
        "ortho": Entry(
            _ortho,
            (
                "OrthoGrad: projects each gradient orthogonal to the current weights "
                "before the base step, a grokking and regularization aid."
            ),
        ),
        "lookahead": Entry(
            _lookahead,
            (
                "Lookahead: keeps slow weights and pulls the fast iterate toward them "
                "every k steps (k=5, alpha=0.5)."
            ),
        ),
        "schedule_free": Entry(
            _schedule_free,
            (
                "Schedule-Free: Polyak-style averaging in place of an LR schedule. The "
                "base iterate z is averaged into a running average x, which is what "
                "eval deploys. The base lr is doubled and weight decay moves from the "
                "base optimizer to z."
            ),
        ),
        "gated_schedule_free": Entry(
            _gated_schedule_free,
            (
                "Schedule-Free with a per-coordinate gradient-SNR gate on the "
                "averaging weight, so each coordinate picks its own bias-variance "
                "point from its own gradient statistics, with no knob. Pinned to a "
                "gate of 1 it is exactly schedule_free."
            ),
        ),
        "wave_schedule_free": Entry(
            _wave_schedule_free,
            (
                "Schedule-Free whose averaging weight is a standing wave over the "
                "flattened parameter index (a frozen ~pi cycles per tensor): "
                "content-free structure, a baseline against the SNR gate of "
                "gated_schedule_free. The harmonic-weight RL controller can drive the "
                "wave per episode (``harmonic_weight_wave``)."
            ),
        ),
        "half_lion": Entry(
            _half_lion,
            (
                "Blends the live weights with a frozen copy of their init through a "
                "traveling standing wave over the parameter index. The gradient is "
                "taken at the blend and applied to the current weights, and eval "
                "deploys 100% current weights. Keeps the LR schedule. Cannot stack "
                "with wave_schedule_free, since both own what the forward sees."
            ),
        ),
        "low_rank_moment": Entry(
            _low_rank_moment,
            (
                "Passthrough telemetry: tracks an Adafactor-style factored second "
                "moment of the gradient (O(out+in) per matrix), so the second-moment "
                "dashboard cards populate even under Lion. Does not change the update."
            ),
        ),
    },
)


class SequentialWrapper:
    """Fold a sequence of registry wrappers onto a base optimizer, in order
    (innermost first). The result is the nested optimizer (wrappers self-nest)."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys: List[str] = list(keys or [])
        unknown = [k for k in self.keys if k not in registry.namespace("wrappers")]
        if unknown:
            raise ValueError(
                f"unknown optimizer wrapper(s) {unknown}; "
                f"choices: {sorted(registry.namespace("wrappers"))}"
            )

    @property
    def disables_schedule(self) -> bool:
        return any(
            getattr(registry.lookup("wrappers", k), "disables_schedule", False)
            for k in self.keys
        )

    def __call__(self, optimizer):
        for key in self.keys:
            optimizer = registry.lookup("wrappers", key)(optimizer)
        return optimizer


def wrappers_disable_schedule(keys: Iterable[str]) -> bool:
    """True if any selected wrapper runs without an LR schedule."""
    return SequentialWrapper(keys).disables_schedule
