"""Lightning callback that logs host-RAM growth to a file to localize leaks.

Born from the calm-d-1 crashes: the process grew ~2-3GB/h of anonymous memory
until RAM and all 36GB of swap were exhausted and the kernel OOM killer fired.
The dashboard swallows stdout into a capped deque and `docker compose run --rm`
discards the container, so nothing survived a crash to say where the memory
went. This callback appends one line every ``SAMPLE_EVERY_S`` seconds to
``{run_dir}/host_memory.log`` - process RSS/swap, direct-descendant RSS
(dataloader workers, orchestration sidecar), system MemAvailable/SwapFree, and
growth rates since start and over the last hour - flushed per write, so the
tail of the file after a SIGKILL is the growth curve. Cost is a few /proc
reads per minute.

A tracemalloc tier once lived here for allocation-site attribution, but
per-allocation stack capture under the GIL multiplied this model's
Python-heavy step time ~70x and was removed as unusable. For attribution,
prefer out-of-process tools (e.g. ``memray attach <pid>``) or the
discriminating experiments in the run notes (dataset subsets,
MALLOC_ARENA_MAX=2, --no-compile) guided by this file's growth rates.
"""

import os
import time
from collections import deque

from lightning.pytorch.callbacks import Callback

SAMPLE_EVERY_S = 60


def _read_kb(path, fields):
    """Return {field: kB} parsed from a /proc status-style file; {} on failure."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                key = line.split(":", 1)[0]
                if key in fields:
                    out[key] = int(line.split()[1])
    except OSError:
        pass
    return out


def _descendant_pids(root_pid, max_depth=3):
    """Collect descendant PIDs via /proc/<pid>/task/*/children, depth-limited."""
    pids, frontier = [], [(root_pid, 0)]
    while frontier:
        pid, depth = frontier.pop()
        if depth >= max_depth:
            continue
        task_dir = f"/proc/{pid}/task"
        try:
            tids = os.listdir(task_dir)
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"{task_dir}/{tid}/children") as f:
                    kids = f.read().split()
            except OSError:
                continue
            for kid in kids:
                pids.append(int(kid))
                frontier.append((int(kid), depth + 1))
    return pids


class HostMemoryCallback(Callback):
    """Append host-memory samples to ``{run_dir}/host_memory.log``.

    Args:
        run_dir: Directory for the current run; ``host_memory.log`` lands here.
    """

    def __init__(self, run_dir: str):
        super().__init__()
        self.path = os.path.join(run_dir, "host_memory.log")
        self._file = None
        self._t0 = None
        self._base_mb = None
        self._next_sample = 0.0
        # (monotonic, footprint_mb) samples covering a bit over the last hour
        # at the 60s cadence, for the trailing growth rate.
        self._window = deque(maxlen=(3600 // SAMPLE_EVERY_S) + 4)

    # -- plumbing ----------------------------------------------------------

    def _log(self, text):
        if self._file is None:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for line in text.splitlines():
            self._file.write(f"[{stamp}] {line}\n")
        self._file.flush()

    def _footprint_mb(self):
        """(rss_mb, swap_mb) of this process; (0, 0) if /proc is unreadable."""
        status = _read_kb("/proc/self/status", ("VmRSS", "VmSwap"))
        return status.get("VmRSS", 0) / 1024, status.get("VmSwap", 0) / 1024

    # -- lifecycle ---------------------------------------------------------

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        try:
            self._file = open(self.path, "a", buffering=1)
        except OSError as e:
            print(f"[HostMemory] Cannot open {self.path}: {e}")
            return
        self._t0 = time.monotonic()
        rss, swap = self._footprint_mb()
        self._base_mb = rss + swap
        self._log(f"=== run start pid={os.getpid()} rss={rss:.0f}MB swap={swap:.0f}MB ===")
        print(f"[HostMemory] Logging host-RAM samples to {self.path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._file is None:
            return
        now = time.monotonic()
        if now >= self._next_sample:
            self._next_sample = now + SAMPLE_EVERY_S
            self._sample(trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        if self._file is None:
            return
        self._sample(trainer.global_step)
        self._log("=== run end ===")
        self._file.close()
        self._file = None

    def on_exception(self, trainer, pl_module, exception):
        if self._file is None:
            return
        self._log(f"=== exception: {type(exception).__name__}: {exception} ===")
        self._sample(trainer.global_step)

    # -- sampling ----------------------------------------------------------

    def _sample(self, step):
        rss, swap = self._footprint_mb()
        total = rss + swap
        now = time.monotonic()
        self._window.append((now, total))

        kids_mb = 0.0
        for pid in _descendant_pids(os.getpid()):
            status = _read_kb(f"/proc/{pid}/status", ("VmRSS", "VmSwap"))
            kids_mb += (status.get("VmRSS", 0) + status.get("VmSwap", 0)) / 1024

        meminfo = _read_kb("/proc/meminfo", ("MemAvailable", "SwapFree"))

        hours = (now - self._t0) / 3600
        rate_total = (total - self._base_mb) / hours if hours > 0.01 else 0.0
        t_old, mb_old = self._window[0]
        span_h = (now - t_old) / 3600
        rate_hour = (total - mb_old) / span_h if span_h > 0.01 else 0.0

        self._log(
            f"step={step} rss={rss:.0f}MB swap={swap:.0f}MB children={kids_mb:.0f}MB "
            f"sys_avail={meminfo.get('MemAvailable', 0) / 1024:.0f}MB "
            f"sys_swap_free={meminfo.get('SwapFree', 0) / 1024:.0f}MB "
            f"growth={rate_total:+.0f}MB/h trailing={rate_hour:+.0f}MB/h"
        )
