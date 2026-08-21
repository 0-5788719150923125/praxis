"""Runs the model-probing dashboard snapshots from inside the training loop."""

import logging

from lightning.pytorch.callbacks import Callback

_log = logging.getLogger("praxis.web")


class SnapshotPumpCallback(Callback):
    """Gives the snapshot producer the training loop's turn.

    The producer used to probe the model from its own daemon thread. That is
    not merely a race: a torch op on a tensor the training step is also using
    deadlocks the process on (GIL, AutogradMeta.mutex_), and it wedged
    abstractinator-s twice - the second time so completely that the stall
    watchdog's own dump could not run, because the dump needs the GIL and the
    GIL is what is stuck. ``praxis/web/snapshots.py`` has the full ABBA.

    So the producer hands those recipes here at ``on_fit_start`` and this
    callback runs them at a batch boundary, where the training thread owns the
    model outright - the same arrangement ``GenerationQueueCallback`` uses for
    inference requests, and for the same reason.

    Cost is a dict scan per step; a recipe only actually runs once its own
    interval (seconds, not steps) has elapsed.
    """

    def __init__(self, producer):
        super().__init__()
        self.producer = producer

    def on_fit_start(self, trainer, pl_module):
        if self.producer is None:
            return
        # Fences a recipe already in flight on the producer thread, so the
        # first training batch does not race the handover itself.
        self.producer.attach_pump()

    def _pump(self):
        try:
            self.producer.pump()
        except Exception:
            # pump() already isolates per-recipe failures; this is the backstop
            # for anything structural. A dashboard card is never worth ending a
            # training run over.
            _log.exception("Failed to pump the snapshot producer")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.producer is not None:
            self._pump()

    def on_validation_end(self, trainer, pl_module):
        # Validation can run for many batches with no train hook firing. Without
        # this the model cards freeze for its duration and any probe queued by a
        # request thread sits until it expires.
        if self.producer is not None:
            self._pump()

    def _release(self):
        if self.producer is not None:
            self.producer.detach_pump()

    def on_fit_end(self, trainer, pl_module):
        self._release()

    def on_exception(self, trainer, pl_module, exception):
        self._release()
