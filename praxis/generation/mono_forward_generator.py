"""Mono-Forward inference as a thin configuration of :class:`Generator`.

Under Mono-Forward the model is sharded across Ray actors, so inference
hops activations through the live actor chain rather than calling
``model.generate()``. That single difference lives in
:class:`MonoForwardBackend`; everything else - the request queue, the
tool-call loop, prompt handling, sampling defaults - is the shared
Generator.

Ray serializes method calls per actor, so an inference request submitted
during training queues behind any in-flight ``train_batch`` and sees a
consistent weight snapshot. The web API path has no training loop to
drain the queue, so this runs ``synchronous=True``: requests are
fulfilled in-place in ``request_generation``.
"""

from __future__ import annotations

from typing import Any

from praxis.generation.decode_backend import MonoForwardBackend
from praxis.generation.generator import Generator


def MonoForwardGenerator(trainer: Any, tokenizer: Any) -> Generator:
    """Build a :class:`Generator` that decodes through the MF actor chain."""
    return Generator(
        tokenizer=tokenizer,
        device="cpu",
        backend=MonoForwardBackend(trainer, tokenizer),
        synchronous=True,
    )
