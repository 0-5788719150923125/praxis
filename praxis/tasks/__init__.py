"""Task taxonomy and per-task loss weighting.

A "task" tags each training token with its source category so the loss
can be reweighted per category (e.g. damp instruction tokens, leave
pretraining tokens at 1.0). The taxonomy is intentionally small: too
many categories defeats the point.

Add a new weighting strategy by defining (or reusing) a weighter class
in :mod:`praxis.tasks.weighter` and dropping a factory entry in
``TASK_WEIGHTER_REGISTRY`` below.
"""

from functools import partial
from typing import Dict

from praxis.tasks.types import (
    DEFAULT_TASK,
    TASK_NAME_TO_ID,
    TASK_NAMES,
    TaskType,
    coerce_task,
    task_id,
    task_name,
)
from praxis.tasks.weighter import (
    DifficultyTaskLossWeighter,
    FixedTaskLossWeighter,
    LearnableTaskLossWeighter,
    TaskLossWeighter,
)

# Starting-value dict reused by fixed and learnable bias_pretrain variants.
#
# EVERY TaskType must appear here. `_targets_to_tensor` seeds from
# `torch.ones(len(TaskType))` (weighter.py) and only overwrites listed keys, so
# a task added to the enum after this dict was written silently inherits 1.0 -
# i.e. 3.3x `conversation` - and nothing reports it. That is exactly what
# happened to `joke`: paired with `rated-jokes` at PRINT_WEIGHT=5.0 (forced into
# the mix by JokePolicy.dataset_collections), it ran at 5x sampling and 3.3x
# per-token weight for the whole abstractinator-d..g line. See next/rl.md.
#
# The dynamic weighter does NOT absorb a wrong value here: DifficultyTaskLoss-
# Weighter._effective() returns `targets * clamp(ratio**gamma, 0.1, 4.0)`, so
# these are a multiplicative BASE the curriculum moves around, not a prior it
# can overrule.
BIAS_PRETRAIN_TARGETS: Dict[str, float] = {
    "pretrain": 1.0,
    "instruction": 0.3,
    "conversation": 0.3,
    "tool_call": 0.2,
    "reasoning": 0.5,
    "rl": 0.5,
    # Joke text is conversation content; it gets conversation's weight. Its
    # 5x sampling weight is a separate lever (praxis/data/config.py::joke).
    "joke": 0.3,
    # Preference sides. Chosen text trains as ordinary conversation data (the
    # SFT anchor of the ORPO-shaped objective), so it matches `conversation`.
    "pref_chosen": 0.3,
    # Rejected text is contrast-only and is hard-excluded from the main CE in
    # _build_loss_weights; 0.0 states that here rather than leaving it implicit
    # in one call site. PreferencePolicy selects its own tokens by task tag and
    # is unaffected by this value.
    "pref_rejected": 0.0,
}


# Factories match the LOSS_REGISTRY pattern: each entry is callable with
# no arguments and returns a configured TaskLossWeighter.
TASK_WEIGHTER_REGISTRY: Dict[str, callable] = {
    # Identity: every task weighted equally. Backward-compatible default.
    "flat": FixedTaskLossWeighter,
    # Fixed bias toward unstructured pretraining content.
    "bias_pretrain": partial(FixedTaskLossWeighter, targets=BIAS_PRETRAIN_TARGETS),
    # Same starting values, but the per-task scalars are learnable:
    # ``weight = 2 * target * sigmoid(raw)`` with an L2 anchor on ``raw``.
    "learnable_bias_pretrain": partial(
        LearnableTaskLossWeighter,
        targets=BIAS_PRETRAIN_TARGETS,
        anchor_weight=0.01,
    ),
    # Stop-gradient EMA-driven curriculum: upweights hard tasks,
    # downweights easy ones, capped by floor/ceiling. No anchor needed
    # because no gradient flows through the multiplier.
    "difficulty_bias_pretrain": partial(
        DifficultyTaskLossWeighter,
        targets=BIAS_PRETRAIN_TARGETS,
    ),
}


def resolve_task_weighter(name) -> TaskLossWeighter:
    """Look up a named strategy in ``TASK_WEIGHTER_REGISTRY`` and instantiate it.

    ``None`` or an empty string returns the ``flat`` (identity) weighter,
    so the default path stays a no-op. Passing an already-constructed
    weighter is a pass-through (useful for tests).
    """
    if not name:
        return TASK_WEIGHTER_REGISTRY["flat"]()
    if isinstance(name, TaskLossWeighter):
        return name
    if name not in TASK_WEIGHTER_REGISTRY:
        raise KeyError(
            f"Unknown task-weighter {name!r}. "
            f"Known: {sorted(TASK_WEIGHTER_REGISTRY)}"
        )
    return TASK_WEIGHTER_REGISTRY[name]()


__all__ = [
    "BIAS_PRETRAIN_TARGETS",
    "DEFAULT_TASK",
    "TASK_NAMES",
    "TASK_NAME_TO_ID",
    "TASK_WEIGHTER_REGISTRY",
    "TaskType",
    "TaskLossWeighter",
    "FixedTaskLossWeighter",
    "LearnableTaskLossWeighter",
    "DifficultyTaskLossWeighter",
    "coerce_task",
    "resolve_task_weighter",
    "task_id",
    "task_name",
]
