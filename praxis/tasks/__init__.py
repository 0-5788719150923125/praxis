"""Task taxonomy and per-task loss weighting.

A "task" tags each training token with its source category so the loss
can be reweighted per category (e.g. damp instruction tokens, leave
pretraining tokens at 1.0). The taxonomy is intentionally small: too
many categories defeats the point.

Add a new weighting strategy by defining (or reusing) a weighter class
in :mod:`praxis.tasks.weighter` and dropping a factory entry in
the ``task_weights`` registry below.
"""

from functools import partial
from typing import Dict

from praxis import registry
from praxis.registry import Entry
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
    # Local files (--data-path dirs + the repo-root `praxis` dataset) are a few
    # MB seen over and over, against effectively infinite web pretraining data,
    # and they are oversampled on top of that - one sampler per directory. 0.1
    # is deliberately aggressive: under the difficulty weighter's [0.1, 4.0]
    # clamp the effective weight lands in [0.01, 0.4], so local can never reach
    # even half of `pretrain` no matter how hard it looks. Note this is doing
    # double duty under `sampler_mode: tasker`, where a dataset's pull is
    # `static_weight x task_weight` (manager.py::_compute_weights) - so this
    # cuts the sampling share too, which is the other half of the problem. The
    # 0.2 `loss_uniform_mix` floor keeps it from starving outright.
    "local": 0.1,
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


registry.declare(
    "task_weights",
    title="Per-task loss weighting",
    doc=(
        (
            "How each token's loss is reweighted by its task tag, the source category it "
            "came from (pretraining, instruction, conversation, ...). Each entry is a "
            "factory taking no arguments that returns a configured TaskLossWeighter. Unset "
            "weights every task equally, as ``flat`` does."
        )
    ),
    entries={
        "flat": Entry(
            FixedTaskLossWeighter,
            "Identity: every task weighted 1.0. The default.",
        ),
        "bias_pretrain": Entry(
            partial(FixedTaskLossWeighter, targets=BIAS_PRETRAIN_TARGETS),
            (
                "Fixed per-task weights (``BIAS_PRETRAIN_TARGETS``) biased toward "
                "unstructured pretraining content: instruction, conversation and "
                "tool-call tokens are damped, and local-file tokens most of all."
            ),
        ),
        "learnable_bias_pretrain": Entry(
            partial(
                LearnableTaskLossWeighter,
                targets=BIAS_PRETRAIN_TARGETS,
                anchor_weight=0.01,
            ),
            (
                "bias_pretrain's starting values as learnable per-task scalars: "
                "``weight = 2 * target * sigmoid(raw)``, with an L2 anchor on ``raw``."
            ),
        ),
        "difficulty_bias_pretrain": Entry(
            partial(
                DifficultyTaskLossWeighter,
                targets=BIAS_PRETRAIN_TARGETS,
            ),
            (
                "A stop-gradient curriculum over bias_pretrain's targets: an EMA of "
                "each task's loss upweights hard tasks and downweights easy ones, "
                "within a floor and ceiling. No gradient flows through the multiplier, "
                "so it needs no anchor."
            ),
        ),
    },
)


def resolve_task_weighter(name) -> TaskLossWeighter:
    """Look up a named strategy in the ``task_weights`` registry and instantiate it.

    ``None`` or an empty string returns the ``flat`` (identity) weighter,
    so the default path stays a no-op. Passing an already-constructed
    weighter is a pass-through (useful for tests).
    """
    if not name:
        return registry.lookup("task_weights", "flat")()
    if isinstance(name, TaskLossWeighter):
        return name
    if name not in registry.namespace("task_weights"):
        raise KeyError(
            f"Unknown task-weighter {name!r}. "
            f"Known: {sorted(registry.namespace("task_weights"))}"
        )
    return registry.lookup("task_weights", name)()


__all__ = [
    "BIAS_PRETRAIN_TARGETS",
    "DEFAULT_TASK",
    "TASK_NAMES",
    "TASK_NAME_TO_ID",
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
