"""Terminal inference must not re-enter warmup during validation."""

from types import SimpleNamespace

from praxis.callbacks.lightning.terminal import TerminalInterface


class _NoGen(TerminalInterface):
    def __init__(self):  # bypass full init; only warmup math is exercised
        self.generator = object()
        self.interval = 10
        self.last_time = None
        self.captured = None

    def _is_trigger_passed(self, last_time, interval):
        self.captured = interval
        return False  # stop after recording the interval


def test_validation_batches_keep_trained_interval():
    cb = _NoGen()
    lm = SimpleNamespace(
        trainer=SimpleNamespace(accumulate_grad_batches=1, global_step=6127)
    )
    cb._generate_text(lm, batch_idx=0, interval=10)  # validation: batch_idx resets
    assert cb.captured == cb.interval  # past warmup; no 120s stall


def test_fresh_run_still_warms_up():
    cb = _NoGen()
    lm = SimpleNamespace(
        trainer=SimpleNamespace(accumulate_grad_batches=1, global_step=0)
    )
    cb._generate_text(lm, batch_idx=0, interval=10)
    assert cb.captured > cb.interval  # step 0 starts at the slow end
