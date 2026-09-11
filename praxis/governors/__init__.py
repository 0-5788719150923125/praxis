"""Training-loop governors: feedback controllers over loop-level knobs.

A governor regulates a training-loop hyperparameter from an endogenous
signal measured during training - the mechanical-governor sense: a feedback
device, not a search. This is the loop-level sibling of the patterns already
in the stack (LionGeo's hypergradient geometry blend inside the optimizer,
the learning-progress bandit inside the data pipeline): one controller per
knob, each driven by the signal native to that knob, registered here.

Entries are selected by the ``governor`` experiment key. Each registry value
is a builder ``(batch_size, target_batch_size, val_every,
sequence_multiplier_tiers) -> Callback`` invoked by the training callback
assembly; builders import their Lightning callback lazily so this package stays
importable without Lightning.

``gns_batch`` - governs the effective batch (rows per optimizer step) by
tracking the measured gradient noise scale. ``batch_size`` and
``target_batch_size`` are both ceilings (one microbatch, one step);
``praxis/data/batch_schedule.py`` factorizes the governed row count against
them. See ``praxis/governors/gns.py`` for the estimator and rationale.
"""

from praxis import registry
from praxis.governors.gns import (
    GOVERNOR_METRIC_DESCRIPTIONS,
    BatchTierController,
    GradientNoiseEstimator,
)
from praxis.registry import Entry


def _build_gns_batch(
    batch_size: int,
    target_batch_size: int,
    val_every=None,
    sequence_multiplier_tiers=(),
):
    from praxis.callbacks.lightning.governor import GNSBatchGovernor

    return GNSBatchGovernor(
        batch_size=batch_size,
        target_batch_size=target_batch_size,
        val_every=val_every,
        sequence_multiplier_tiers=sequence_multiplier_tiers,
    )


registry.declare(
    "governors",
    title="Training-loop governors",
    doc=(
        (
            "Feedback controllers over loop-level knobs, each driven by an endogenous "
            "signal native to its knob - a feedback device, not a search. Each entry is a "
            "builder ``(batch_size, target_batch_size, val_every, "
            "sequence_multiplier_tiers) -> Callback``. Unset keeps the static accumulation "
            "factor."
        )
    ),
    entries={
        "gns_batch": Entry(
            _build_gns_batch,
            (
                "Governs the effective batch (rows per optimizer step) by tracking the "
                "measured gradient noise scale. ``batch_size`` and "
                "``target_batch_size`` both become ceilings (one microbatch, one "
                "step), and ``praxis/data/batch_schedule.py`` factorizes the governed "
                "row count against them."
            ),
        ),
    },
)

__all__ = [
    "GOVERNOR_METRIC_DESCRIPTIONS",
    "GradientNoiseEstimator",
    "BatchTierController",
]
