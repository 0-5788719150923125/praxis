"""The `tasker` sampling mode: dataset sampling driven by the model's learned
per-task loss weights.

Closes the loop between the loss weighter (praxis/tasks/weighter.py) and the
data sampler (InterleaveDataManager): a task the weighter deems hard gets both
upweighted in the loss and upsampled in the data.
"""

import torch

from praxis.data.datasets.manager import InterleaveDataManager
from praxis.tasks.weighter import DifficultyTaskLossWeighter


def _reset():
    InterleaveDataManager.shared_task_weights = None


def _tasker_manager(sampler_task_ids, static_weights, task_weights):
    """A minimal manager wired only for the tasker weight calculation - avoids
    the heavyweight __init__ (tokenizer, message queue, dataset fetches)."""
    m = object.__new__(InterleaveDataManager)
    m.weighting_mode = "tasker"
    m.samplers = [None] * len(sampler_task_ids)
    m.sampler_task_ids = list(sampler_task_ids)
    m.static_weights = list(static_weights)
    InterleaveDataManager.shared_task_weights = (
        None if task_weights is None else list(task_weights)
    )
    return m


# --------------------------------------------------------------------------
# update_task_weights classmethod (the trainer -> sampler push)
# --------------------------------------------------------------------------


def test_update_task_weights_noops_when_not_armed():
    _reset()
    # Not in tasker mode: the push must be a silent no-op.
    InterleaveDataManager.update_task_weights([1.0, 2.0, 3.0])
    assert InterleaveDataManager.shared_task_weights is None


def test_update_task_weights_accepts_tensor_and_list():
    _reset()
    InterleaveDataManager.shared_task_weights = [1.0, 1.0]
    InterleaveDataManager.update_task_weights(torch.tensor([1.0, 4.0]))
    assert InterleaveDataManager.shared_task_weights == [1.0, 4.0]
    InterleaveDataManager.update_task_weights([2.0, 0.5])
    assert InterleaveDataManager.shared_task_weights == [2.0, 0.5]
    _reset()


# --------------------------------------------------------------------------
# _calculate_target_weights: task weight -> sampling weight
# --------------------------------------------------------------------------


def test_warmup_is_uniform_until_tasker_reports():
    _reset()
    m = _tasker_manager([0, 0, 1], [1.0, 1.0, 1.0], task_weights=None)
    w = m._calculate_target_weights()
    assert w == [1 / 3, 1 / 3, 1 / 3]


def test_hard_task_is_upsampled():
    _reset()
    # task 1 is "hard" (weight 3x); the dataset on it should be sampled most.
    m = _tasker_manager(
        sampler_task_ids=[0, 0, 1],
        static_weights=[1.0, 1.0, 1.0],
        task_weights=[1.0, 3.0],
    )
    w = m._calculate_target_weights()
    assert abs(sum(w) - 1.0) < 1e-9  # uniform-floor mix preserves normalization
    assert w[2] > w[0]  # hard-task dataset dominates
    assert abs(w[0] - w[1]) < 1e-9  # equal datasets within the same task stay equal


def test_static_weights_scale_within_a_task():
    _reset()
    # Two datasets on the same (equal-weight) task keep their configured ratio.
    m = _tasker_manager(
        sampler_task_ids=[0, 0],
        static_weights=[3.0, 1.0],
        task_weights=[1.0],
    )
    w = m._calculate_target_weights()
    assert w[0] > w[1]


def test_uniform_floor_keeps_easy_task_from_starving():
    _reset()
    # Even with a near-zero task weight, the floor keeps a positive share.
    m = _tasker_manager(
        sampler_task_ids=[0, 1],
        static_weights=[1.0, 1.0],
        task_weights=[0.0, 5.0],
    )
    w = m._calculate_target_weights()
    assert w[0] > 0.0
    assert w[1] > w[0]
    _reset()


# --------------------------------------------------------------------------
# End-to-end signal: a DifficultyTaskLossWeighter's output drives sampling
# --------------------------------------------------------------------------


class _StubSampler:
    """Minimal sampler for the refill loop: fixed task, canned document."""

    def __init__(self, name, task_type):
        self.dataset_path = name
        self.task_type = task_type
        self.weight = 1.0

    def get_document(self):
        return {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "metadata": {},
        }


def test_refill_in_tasker_mode_adapts_weights_and_logs_metrics(tmp_path):
    """Regression: the refill dispatch skipped tasker mode entirely, so tasker
    runs never adapted sampling weights toward the learned task targets and
    logged an EMPTY data_metrics.db (no sampling-weights card - every
    tasker-mode run in build/runs had 0 rows)."""
    import sqlite3

    _reset()
    InterleaveDataManager.shared_weights = None
    InterleaveDataManager.shared_weights_initialized = False
    samplers = [_StubSampler("easy_ds", 0), _StubSampler("hard_ds", 1)]
    m = InterleaveDataManager(
        samplers,
        [0.5, 0.5],
        tokenizer=None,
        block_size=64,
        weighting_mode="tasker",
        run_dir=str(tmp_path),
        data_metrics_log_interval=10,
        enable_chat_validation=False,
    )
    # The trainer reports: task 1 is hard (3x weight).
    InterleaveDataManager.update_task_weights([1.0, 3.0])

    for _ in range(3):
        m._refill_message_queue(min_documents=64)
        m.message_queue.message_queue.clear()  # force the next refill

    # Sampling weights moved toward the hard task...
    assert m.dynamic_weights[1] > m.dynamic_weights[0]
    assert m.sampling_count > 0
    # ...and the sampling-weights card has rows to render.
    m.data_metrics_logger.close()
    con = sqlite3.connect(tmp_path / "data_metrics.db")
    rows = con.execute(
        "select count(*), max(sampling_weights) from data_metrics"
    ).fetchone()
    con.close()
    assert rows[0] > 0
    assert "hard_ds" in rows[1]
    _reset()
    InterleaveDataManager.shared_weights = None
    InterleaveDataManager.shared_weights_initialized = False


def test_difficulty_weighter_output_upsamples_its_hard_task():
    _reset()
    # Feed the weighter a high loss on task 1 and a low loss on task 0, then
    # route its effective_weights through the sampler.
    weighter = DifficultyTaskLossWeighter(gamma=1.0)
    task_ids = torch.tensor([[0, 0, 1, 1]])
    losses = torch.tensor([[0.1, 0.1, 5.0, 5.0]])
    for _ in range(20):  # let the EMA settle
        weighter.observe(task_ids, losses)

    eff = weighter.effective_weights()
    assert eff[1] > eff[0]  # difficulty weighter upweighted the hard task

    InterleaveDataManager.shared_task_weights = [1.0, 1.0]
    InterleaveDataManager.update_task_weights(eff)
    m = _tasker_manager([0, 1], [1.0, 1.0], task_weights=None)
    InterleaveDataManager.shared_task_weights = [float(x) for x in eff]
    w = m._calculate_target_weights()
    assert w[1] > w[0]  # the hard task is sampled more
    _reset()
