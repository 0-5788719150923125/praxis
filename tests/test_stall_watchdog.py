"""The wedge detector has to work when nothing else does, so test that it
actually writes stacks rather than just that it constructs."""

import time
from types import SimpleNamespace

from praxis.callbacks.lightning import StallWatchdogCallback


class _Trainer(SimpleNamespace):
    is_global_zero = True
    global_step = 7


def test_dumps_stacks_when_a_step_overruns(tmp_path):
    wd = StallWatchdogCallback(run_dir=tmp_path, timeout_s=0.2)
    trainer = _Trainer()
    wd.on_fit_start(trainer, None)
    wd.on_train_batch_start(trainer, None, None, 0)
    time.sleep(0.6)  # overrun: the C timer thread fires while we sit here
    wd.on_train_batch_end(trainer, None, None, None, 0)
    wd.on_train_end(trainer, None)

    log = (tmp_path / "stalls.log").read_text()
    assert "watchdog armed" in log
    assert "Thread" in log or "File " in log, log  # a real traceback landed
    assert "step 7 took" in log


def test_priority_dump_lands_without_the_training_thread(tmp_path):
    """The case that matters: the step never ends, so no Lightning hook runs
    again. The watch thread has to notice and dump the MAIN thread by itself -
    faulthandler alone caps at 100 threads and drops main off the end."""
    wd = StallWatchdogCallback(run_dir=tmp_path, timeout_s=0.2)
    wd.POLL_S = 0.05
    trainer = _Trainer()
    wd.on_fit_start(trainer, None)
    wd.on_train_batch_start(trainer, None, None, 0)
    time.sleep(1.0)  # never call another hook: this is the deadlock shape
    log = (tmp_path / "stalls.log").read_text()
    wd.on_train_end(trainer, None)

    assert "priority dump" in log, log
    assert "[MAIN]" in log, log
    assert "has been running" in log
    # And it dumps once per stuck step, not once per poll.
    assert log.count("priority dump (") == 1


def test_a_normal_step_dumps_nothing(tmp_path):
    wd = StallWatchdogCallback(run_dir=tmp_path, timeout_s=30.0)
    trainer = _Trainer()
    wd.on_fit_start(trainer, None)
    for i in range(3):
        wd.on_train_batch_start(trainer, None, None, i)
        wd.on_train_batch_end(trainer, None, None, None, i)
    wd.on_train_end(trainer, None)

    log = (tmp_path / "stalls.log").read_text()
    assert "watchdog armed" in log
    assert "Traceback" not in log and "took" not in log
