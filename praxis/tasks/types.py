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
