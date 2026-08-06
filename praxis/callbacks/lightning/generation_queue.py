"""Drains the inference request queue from inside the training loop."""

import logging

from lightning.pytorch.callbacks import Callback

_log = logging.getLogger("praxis.generation")


class GenerationQueueCallback(Callback):
    """Serves queued API generations at step rate.

    The ``Generator`` is asynchronous by design: an HTTP handler cannot run the
    model, because the training loop owns it and a concurrent forward would
    race the optimizer. ``request_generation`` therefore enqueues, and the work
    happens wherever ``fulfill_requests`` is called from - which has to be a
    place that already holds the training loop's turn.

    That place used to be ``TerminalInterface``'s inference tick, and coupling
    the two made every API request wait on the terminal's DISPLAY cadence.
    That cadence starts at 120s and only reaches ``infer_every`` after 250
    optimizer steps, while the client gives up at 60s
    (``generate_from_messages``), so for roughly the first 143 steps of a fresh
    run every chat request timed out and came back empty. Under ``--quiet`` the
    tick never runs at all and nothing was ever served.

    Draining here decouples them. The terminal keeps its own cadence for the
    rolling contexts it paints; API requests are served every step regardless.
    """

    def __init__(self, generator, max_requests: int = 1):
        super().__init__()
        self.generator = generator
        # Bounded per step: each request is a full generate() inline in the
        # training loop, so an unbounded drain would let a burst of chat
        # traffic stall the run. Requests past the cap wait for the next step,
        # which is milliseconds away rather than minutes.
        self.max_requests = max_requests

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        if self.generator is None:
            return
        try:
            self.generator.fulfill_requests(max_requests=self.max_requests)
        except Exception:
            # fulfill_requests already isolates per-request failures; this is
            # the backstop for anything structural. Serving chat is never worth
            # ending a training run over.
            _log.exception("Failed to drain the generation queue")
