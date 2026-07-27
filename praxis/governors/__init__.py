"""Training-loop governors: feedback controllers over loop-level knobs.

A governor regulates a training-loop hyperparameter from an endogenous
signal measured during training - the mechanical-governor sense: a feedback
device, not a search. This is the loop-level sibling of the patterns already
in the stack (LionGeo's hypergradient geometry blend inside the optimizer,
the learning-progress bandit inside the data pipeline): one controller per
knob, each driven by the signal native to that knob, registered here.

Entries are selected by the ``governor`` experiment key. Each registry value
is a builder ``(batch_size, target_batch_size) -> Callback`` invoked by the
training callback assembly; builders import their Lightning callback lazily
so this package stays importable without Lightning.

``gns_batch`` - governs the gradient-accumulation factor (the effective
batch) by tracking the measured gradient noise scale. See
``praxis/governors/gns.py`` for the estimator and rationale.
"""

from praxis.governors.gns import (
    GOVERNOR_METRIC_DESCRIPTIONS,
    BatchTierController,
    GradientNoiseEstimator,
)


def _build_gns_batch(batch_size: int, target_batch_size: int):
    from praxis.callbacks.lightning.governor import GNSBatchGovernor

    return GNSBatchGovernor(
        batch_size=batch_size, target_batch_size=target_batch_size
    )


GOVERNOR_REGISTRY = {
    "gns_batch": _build_gns_batch,
}

__all__ = [
    "GOVERNOR_REGISTRY",
    "GOVERNOR_METRIC_DESCRIPTIONS",
    "GradientNoiseEstimator",
    "BatchTierController",
]
