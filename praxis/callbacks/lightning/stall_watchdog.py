"""Dump every thread's stack when the training loop stops making progress.

A wedged run is the worst kind of bug to chase, because the evidence dies with
the process: training stops, the inline generation queue stops, the terminal's
rolling contexts stop, and the API server (which serves several endpoints from
the training thread's own data) stops answering. From outside all you can say
is "it froze", and ``py-spy dump`` needs ptrace privileges the process may not
have.

``faulthandler.dump_traceback_later`` solves it from inside and for free. It
arms a timer in a dedicated C thread; if the timer expires before it is
re-armed, that thread writes every Python thread's stack to a file and lets the
process carry on. Critically it does NOT need the GIL to fire, so it still
reports when the interpreter is wedged - which is exactly when nothing else
can.

This callback re-arms the timer at each batch start. A step that overruns
``timeout_s`` therefore dumps the stack of whatever it is stuck in, including
anything running from the batch-end hooks (the generation queue drains there,
so a hung generation is covered). ``repeat`` keeps it dumping, which is what
separates a HANG (the same stack over and over) from something merely SLOW (a
stack that moves).

The timeout is deliberately generous. A step here is milliseconds, but the
first compiled step can take minutes and an inference tick is not free, so this
is a wedge detector, not a performance monitor.
"""

import faulthandler
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from lightning.pytorch.callbacks import Callback

# Long enough that a slow first step, a torch.compile, or an inference tick
# never trips it; short enough that a wedged run is caught within the hour.
DEFAULT_TIMEOUT_S: float = 600.0


class StallWatchdogCallback(Callback):
    """Write all thread stacks to ``stalls.log`` when a step overruns.

    Args:
        run_dir: directory to write ``stalls.log`` into.
        timeout_s: a step (plus its end-hooks) may take this long before the
            watchdog treats it as wedged.
    """

    def __init__(self, run_dir, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__()
        self.path = Path(run_dir) / "stalls.log"
        self.timeout_s = float(timeout_s)
        self._file = None
        self._armed_at = None
        self._step = -1
        self._dumped_for = None
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle ---------------------------------------------------------

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        try:
            # Unbuffered: a wedged process never gets to flush, and a dump
            # nobody can read is the same as no dump at all.
            self._file = open(self.path, "a", buffering=1)
        except OSError as e:
            print(f"[StallWatchdog] Cannot open {self.path}: {e}")
            return
        # The dump thread writes to a raw fd, so the handle has to outlive
        # this call and stay open for as long as the timer is armed.
        faulthandler.enable(file=self._file, all_threads=True)
        self._file.write(
            f"=== watchdog armed pid={os.getpid()} timeout={self.timeout_s:.0f}s "
            f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._watch, name="stall-watchdog", daemon=True
        )
        self._thread.start()
        print(
            f"[StallWatchdog] Dumping all stacks to {self.path} if a step "
            f"exceeds {self.timeout_s:.0f}s"
        )

    def _dump_priority_threads(self) -> None:
        """Write the MAIN thread's stack, plus any thread in project code.

        faulthandler caps its dump at 100 threads and then prints ``...``. This
        process runs ~160 (idle pool workers dominate), so the main thread -
        the one actually running training, and the one worth seeing - fell off
        the end of every dump and a background thread got blamed for a wedge it
        was only a victim of. This pass is not capped and puts main first.

        It needs the GIL, so it cannot report a GIL-starved interpreter; that
        is exactly the case faulthandler still covers. The two are complements,
        which is why both run.
        """
        try:
            frames = sys._current_frames()
        except Exception:  # pragma: no cover - never worth failing a run over
            return
        main_id = threading.main_thread().ident
        by_id = {t.ident: t for t in threading.enumerate()}
        order = [main_id] + [i for i in frames if i != main_id]
        self._file.write(f"--- priority dump ({len(frames)} threads live) ---\n")
        for ident in order:
            frame = frames.get(ident)
            if frame is None:
                continue
            stack = traceback.extract_stack(frame)
            in_project = any("/praxis/" in f.filename for f in stack)
            if ident != main_id and not in_project:
                continue  # library idle threads say nothing; skip the noise
            name = getattr(by_id.get(ident), "name", "?")
            label = "MAIN" if ident == main_id else "project"
            self._file.write(f"\nThread {ident} ({name}) [{label}]:\n")
            for line in traceback.format_list(stack):
                self._file.write(line)
        self._file.write("--- end priority dump ---\n")
        self._file.flush()

    def _watch(self) -> None:
        """Poll for an overrun from OUR OWN thread and dump when one lands.

        The priority dump cannot be driven from the training thread: a
        deadlocked run never reaches another hook, which is precisely when the
        dump is wanted. So this runs on a daemon thread of its own. It needs
        only the GIL, and in a futex deadlock every thread is parked and the
        GIL is free - the case we actually hit.
        """
        while not self._stop.wait(self.POLL_S):
            armed_at = self._armed_at
            if armed_at is None or self._file is None:
                continue
            if time.monotonic() - armed_at <= self.timeout_s:
                continue
            if self._dumped_for == self._step:
                continue  # one uncapped dump per stuck step is enough
            self._dumped_for = self._step
            try:
                self._file.write(
                    f"=== step {self._step} has been running "
                    f"{time.monotonic() - armed_at:.0f}s "
                    f"(watchdog timeout {self.timeout_s:.0f}s) ===\n"
                )
                self._dump_priority_threads()
            except Exception:  # pragma: no cover
                pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._file is None:
            return
        self._step = trainer.global_step
        self._armed_at = time.monotonic()
        # Re-arming cancels the previous timer, so the countdown restarts every
        # step and only an overrun ever fires. repeat=True keeps dumping while
        # the wedge lasts, which is what tells a hang apart from a slow step.
        faulthandler.dump_traceback_later(
            self.timeout_s, repeat=True, file=self._file, exit=False
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Deliberately NOT cancelled here. The generation queue drains from a
        # batch-end hook, and callback order is not ours to rely on, so leaving
        # the timer armed until the next batch starts is what keeps a hung
        # generation inside the watched window.
        if self._file is not None and self._armed_at is not None:
            elapsed = time.monotonic() - self._armed_at
            if elapsed > self.timeout_s:
                # The dump already landed; label it so the log says which step
                # and how long, which the raw traceback does not carry.
                self._file.write(
                    f"=== step {self._step} took {elapsed:.0f}s "
                    f"(watchdog timeout {self.timeout_s:.0f}s) ===\n"
                )

    POLL_S: float = 30.0  # how often the watch thread checks for an overrun

    def _disarm(self):
        self._stop.set()
        self._armed_at = None
        faulthandler.cancel_dump_traceback_later()
        if self._file is not None:
            self._file.close()
            self._file = None

    def on_train_end(self, trainer, pl_module):
        self._disarm()

    def on_exception(self, trainer, pl_module, exception):
        if self._file is not None:
            self._file.write(
                f"=== exception at step {self._step}: "
                f"{type(exception).__name__}: {exception} ===\n"
            )
        self._disarm()
