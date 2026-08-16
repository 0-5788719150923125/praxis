"""Task-type resolution for DATASETS entries.

``resolve_task_type`` reads an explicit ``task_type`` first and otherwise maps
the entry's ``format`` through ``FORMAT_TO_TASK``. Type-only entries - the
synthetic and locally-sourced datasets - set no ``format``, so they hit
``DEFAULT_TASK`` (PRETRAIN) unless they say otherwise. That fallback is silent:
nothing warns, the dataset simply trains on fineweb's loss-weight line and
shares fineweb's difficulty EMA.

``git-history`` sat there. It is one repo's commit log, a few MB re-read every
epoch, and it carried ``pretrain``'s 1.0 target instead of ``local``'s 0.1.
"""

from praxis.data.config import DATASETS, FORMAT_TO_TASK, resolve_task_type
from praxis.tasks import DEFAULT_TASK, TaskType

# Type-only entries knowingly left on the PRETRAIN fallback. Anything NOT here
# that lands on it is an oversight, which is the whole point of the list.
#
# Empty, and worth keeping that way: every type-only dataset so far turned out
# to be locally sourced. An entry here needs a reason next to it.
ACKNOWLEDGED_PRETRAIN_FALLBACKS: set = set()

# Everything sourced from this machine rather than from a web-scale corpus.
LOCAL_SOURCED = ("git-history", "praxis", "kb")


def _falls_back(cfg):
    """True when neither an explicit task_type nor a format decides the task."""
    return "task_type" not in cfg and cfg.get("format") is None


def test_git_history_is_local_not_pretrain():
    """The repo's own git log is local material, not pretraining corpus.

    Left on PRETRAIN it drew ``BIAS_PRETRAIN_TARGETS["pretrain"]`` = 1.0
    against ``local``'s 0.1 - and under ``sampler_mode: tasker`` a dataset's
    pull is ``static_weight x task_weight``, so the mistake inflated its
    sampling share as well as its loss weight.
    """
    resolved = resolve_task_type(DATASETS["git-history"])
    assert TaskType(resolved) is TaskType.LOCAL, (
        f"git-history resolved to {TaskType(resolved).name}; a few MB of commit "
        f"history re-read every epoch must not share fineweb's weight line"
    )


def test_local_sourced_datasets_agree():
    """Everything read off this machine carries the same task type.

    They share one economics: a bounded corpus the model revisits, against web
    pretraining data it never sees twice. Splitting them across task types
    would give the same material two different loss weights and two different
    difficulty EMAs.
    """
    for name in LOCAL_SOURCED:
        resolved = TaskType(resolve_task_type(DATASETS[name]))
        assert resolved is TaskType.LOCAL, f"{name} is {resolved.name}, not LOCAL"


def test_no_dataset_lands_on_the_pretrain_fallback_by_accident():
    """A type-only entry that forgets ``task_type`` is silently PRETRAIN.

    This is the trap that caught git-history. New synthetic or local datasets
    are declared with ``type`` and no ``format``, so they inherit DEFAULT_TASK
    without any signal that a choice was skipped.
    """
    offenders = sorted(
        name
        for name, cfg in DATASETS.items()
        if _falls_back(cfg) and name not in ACKNOWLEDGED_PRETRAIN_FALLBACKS
    )
    assert not offenders, (
        f"{offenders} declare neither `task_type` nor `format`, so they train "
        f"as PRETRAIN by accident. Declare `task_type` on the DATASETS entry, "
        f"or add the name to ACKNOWLEDGED_PRETRAIN_FALLBACKS with a reason."
    )


def test_resolution_precedence_is_explicit_over_format():
    """An explicit task_type must beat whatever the format would imply."""
    assert TaskType(resolve_task_type({"task_type": "local"})) is TaskType.LOCAL
    # A format-only config follows the map...
    for fmt, expected in FORMAT_TO_TASK.items():
        assert resolve_task_type({"format": fmt}) == int(expected)
    # ...and an override wins over it.
    fmt = next(f for f, t in FORMAT_TO_TASK.items() if t is not TaskType.LOCAL)
    assert (
        TaskType(resolve_task_type({"format": fmt, "task_type": "local"}))
        is TaskType.LOCAL
    )
    # Nothing declared at all is the silent fallback this suite guards.
    assert resolve_task_type({}) == int(DEFAULT_TASK)
