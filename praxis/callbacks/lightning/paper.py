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

    def __init__(self, every: int, log_dir: str, authors=None):
        self.every = max(int(every), 0)
        self.log_path = os.path.join(log_dir, "paper_build.log")
        self._authors = authors  # already-ordered (see builder._resolve_authors)
        self._lock = threading.Lock()
        self._latexmk = shutil.which("latexmk")
        self._warned = False
        if self.every and not self._latexmk:
            # The generated inputs (figures, framing, the strand snapshot) still
            # regenerate from the live model each cadence; only the final PDF
            # compile is skipped, so the .tex/figures stay fresh for a host-side
            # or later compile.
            print(
                "[Paper] latexmk not found; regenerating paper inputs only "
                "(PDF compile skipped this run)."
            )

    def _snapshot_model(self, pl_module):
        """The model whose live snapshots the figures should match.

        Prefer the web dashboard's model - the generator's inference copy in
        ``app.config["generator"]`` - so the strand figure reads the SAME
        transient state (e.g. ``_last_input_coeffs``) the live cards show. The
        training model (pl_module.model) is a distinct instance with its own
        non-persistent buffers, so reading it would diverge from the dashboard
        (the 0%-vs-14%-variance mismatch). Falls back to the training model when
        no generator is wired (e.g. Mono-Forward, or before the server is up)."""
        try:
            from praxis.web.app import app

            gen = app.config.get("generator")
            m = getattr(gen, "model", None)
            if m is not None:
                return m
        except Exception:
            pass
        return getattr(pl_module, "model", pl_module)

    def on_train_start(self, trainer, pl_module, *args, **kwargs):
        """Rebuild once at launch/resume so the paper reflects the loaded
        checkpoint immediately, rather than waiting for the first save-every
        step. (The strand figure needs populated forward state; the renderer
        keeps the prior figure rather than downgrading to all-blue until a
        forward has run - see strands.export_strands.)"""
        if not self.every or trainer.global_rank != 0 or self._lock.locked():
            return
        threading.Thread(
            target=self._build,
            args=(trainer.global_step, self._snapshot_model(pl_module)),
            daemon=True,
        ).start()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if not self.every:
            return
        if trainer.global_step == 0 or trainer.global_step % self.every != 0:
            return
        if trainer.global_rank != 0 or self._lock.locked():
            return  # another build still running, or not the lead rank
        # Render figures from the same model the dashboard shows (see
        # _snapshot_model), read-only like the live snapshots.
        threading.Thread(
            target=self._build,
            args=(trainer.global_step, self._snapshot_model(pl_module)),
            daemon=True,
        ).start()

    def _build(self, step: int, model=None) -> None:
        with self._lock:
            try:
                with open(self.log_path, "w") as log:
                    log.write(f"# paper rebuild at step {step}\n")
                    log.flush()
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(
                        log
                    ):
                        from praxis.pillars.build import build_all

                        build_all(model=model, authors=self._authors)
                    # PDF compile is the only latexmk-dependent step; the inputs
                    # above already refreshed from the live model.
                    if self._latexmk:
                        subprocess.run(
                            [
                                self._latexmk,
                                "-pdf",
                                "-interaction=nonstopmode",
                                "main.tex",
                            ],
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
