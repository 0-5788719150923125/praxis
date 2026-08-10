"""Task type taxonomy.

Tasks are stored as small ints in tensors that ride alongside ``input_ids``,
so the registry must be stable across processes. Add new tasks at the end
to keep existing checkpoints valid.
"""

from enum import IntEnum
from typing import Optional, Union


class TaskType(IntEnum):
    PRETRAIN = 0
    INSTRUCTION = 1
    CONVERSATION = 2
    TOOL_CALL = 3
    REASONING = 4
    RL = 5
    JOKE = 6
    # Preference-pair material (e.g. Anthropic/hh-rlhf): the two sides of a
    # chosen/rejected pair. CHOSEN trains as conversation data AND anchors the
    # preference margin; REJECTED is contrast-only - it is structurally
    # excluded from the main CE (see _build_loss_weights) and only ever
    # trained through the preference policy's margin loss.
    PREF_CHOSEN = 7
    PREF_REJECTED = 8
    # Files read off local disk: --data-path directories and the repo-root
    # `praxis` dataset. Split out of PRETRAIN because the two have opposite
    # economics. Web-scale pretraining corpora are effectively infinite and
    # never repeat; a handful of local repos is a few MB that the model sees
    # again and again within a single epoch. Folded together they shared one
    # loss-weight line AND one difficulty EMA, so the curriculum could not
    # tell an unrepeatable token from an overfit one. They are also heavily
    # oversampled: each --data-path entry is its own sampler (see
    # praxis/data/utils.py), so 14 directories outnumber the web corpora in
    # the mix. See BIAS_PRETRAIN_TARGETS for the weight this carries.
    LOCAL = 9


DEFAULT_TASK = TaskType.PRETRAIN

TASK_NAMES = tuple(t.name.lower() for t in TaskType)
TASK_NAME_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}


def task_name(task_id: int) -> str:
    return TaskType(task_id).name.lower()


def task_id(name: str) -> int:
    key = name.lower().strip()
    if key not in TASK_NAME_TO_ID:
        raise KeyError(f"Unknown task type: {name!r}. Known: {TASK_NAMES}")
    return TASK_NAME_TO_ID[key]


def coerce_task(value: Optional[Union[str, int, "TaskType"]]) -> int:
    """Normalize a task spec to its int ID. None falls back to DEFAULT_TASK."""
    if value is None:
        return int(DEFAULT_TASK)
    if isinstance(value, TaskType):
        return int(value)
    if isinstance(value, int):
        TaskType(value)  # validates
        return value
    return task_id(value)
