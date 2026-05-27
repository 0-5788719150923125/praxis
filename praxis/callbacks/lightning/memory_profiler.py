"""Lightning callback that records a CUDA memory snapshot for a step window.

Wraps ``torch.cuda.memory._record_memory_history`` / ``_dump_snapshot`` so a run
can capture every allocation (with call stacks) over a window of training steps,
then write a ``.pickle`` to load at https://pytorch.org/memory_viz. The snapshot
attributes peak VRAM to whatever produced it - encoder, attention, Titans memory,
halting loops - without per-component instrumentation. Diagnostic only; gated on
``--profile-memory`` because recording carries overhead and the file grows with
allocation traffic.
"""

import os

import torch
from lightning.pytorch.callbacks import Callback


class MemoryProfilerCallback(Callback):
    """Record a CUDA allocation history over a step window and dump a snapshot.

    Args:
        run_dir: Directory for the current run; the snapshot lands here.
        start_step: Begin recording once ``global_step`` reaches this (skip warmup).
        num_steps: Stop and dump after this many recorded training batches.
        max_entries: Cap on retained allocation events (bounds memory + file size).
    """

    def __init__(
        self,
        run_dir: str,
        start_step: int = 0,
        num_steps: int = 50,
        max_entries: int = 100_000,
    ):
        super().__init__()
        self.path = os.path.join(run_dir, "memory_snapshot.pickle")
        self.start_step = start_step
        self.num_steps = num_steps
        self.max_entries = max_entries
        self._recording = False
        self._done = False
        self._count = 0

    def _active(self, trainer) -> bool:
        return torch.cuda.is_available() and trainer.is_global_zero

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._done or self._recording or not self._active(trainer):
            return
        if trainer.global_step < self.start_step:
            return
        # stacks="python" so allocations attribute to praxis source, not C++
        # allocator unwind frames.
        torch.cuda.memory._record_memory_history(
            max_entries=self.max_entries, stacks="python"
        )
        self._recording = True
        print(
            f"[MemoryProfiler] Recording allocations for {self.num_steps} steps "
            f"from step {trainer.global_step} -> {self.path}"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._recording or self._done:
            return
        self._count += 1
        if self._count >= self.num_steps:
            self._dump()

    def on_train_end(self, trainer, pl_module):
        # Catch runs that stop before the window fills.
        if self._recording and not self._done:
            self._dump()

    def _dump(self):
        try:
            torch.cuda.memory._dump_snapshot(self.path)
            print(
                f"[MemoryProfiler] Wrote snapshot ({self._count} steps) to {self.path}. "
                "Load it at https://pytorch.org/memory_viz"
            )
        except Exception as e:
            print(f"[MemoryProfiler] Failed to dump snapshot: {e}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
            self._recording = False
            self._done = True
