"""Periodically rebuild the living research paper during training.

The paper in ``research/`` draws its figures and gated prose from the current
run (checkpoints for the geometry figure, metrics.db for halting, the resolved
config for framings). This callback regenerates those inputs and recompiles
``research/main.pdf`` on the ``--save-every`` cadence - right after a checkpoint
lands, so the geometry figure sees fresh weights.

It is deliberately unobtrusive: the build runs in a daemon thread (training never
waits on latexmk), only one build runs at a time, output is redirected to a log
in the run directory, latexmk-absent or any build error is swallowed with a
single warning, and only rank 0 builds. Disable with ``--no-paper``.
"""

import contextlib
import os
import shutil
import subprocess
import threading

from lightning.pytorch.callbacks import Callback

# praxis/callbacks/lightning/paper.py -> repo root is parents[3].
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RESEARCH_DIR = os.path.join(REPO_ROOT, "research")


class PaperBuildCallback(Callback):
    """Rebuild research/main.pdf every ``every`` steps, in the background."""

    def __init__(self, every: int, log_dir: str):
        self.every = max(int(every), 0)
        self.log_path = os.path.join(log_dir, "paper_build.log")
        self._lock = threading.Lock()
        self._latexmk = shutil.which("latexmk")
        self._warned = False
        if self.every and not self._latexmk:
            print("[Paper] latexmk not found; paper rebuild disabled for this run.")

    def on_train_batch_end(self, trainer, *args, **kwargs):
        if not self.every or not self._latexmk:
            return
        if trainer.global_step == 0 or trainer.global_step % self.every != 0:
            return
        if trainer.global_rank != 0 or self._lock.locked():
            return  # another build still running, or not the lead rank
        threading.Thread(
            target=self._build, args=(trainer.global_step,), daemon=True
        ).start()

    def _build(self, step: int) -> None:
        with self._lock:
            try:
                with open(self.log_path, "w") as log:
                    log.write(f"# paper rebuild at step {step}\n")
                    log.flush()
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(
                        log
                    ):
                        from praxis.pillars.build import build_all

                        build_all()
                    subprocess.run(
                        [self._latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"],
                        cwd=RESEARCH_DIR,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=300,
                        check=False,
                    )
            except Exception as exc:
                if not self._warned:  # warn once; a flaky build must not spam
                    self._warned = True
                    print(f"[Paper] rebuild failed (see {self.log_path}): {exc}")
