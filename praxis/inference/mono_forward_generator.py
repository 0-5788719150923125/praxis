"""Mono-Forward inference as a thin configuration of :class:`Generator`.

Under Mono-Forward the model is sharded across Ray actors, so inference hops
activations through the live actor chain rather than calling a single module's
forward. That difference is confined to :class:`MonoForwardLM`, a
``PreTrainedModel`` face over the actor chain; from there up everything is the
ordinary path - the same :class:`ModelBackend`, the same ``model.generate``,
and so the same prepared logits processors, stop strings, deadline and
streamer that every other run gets.

This used to be a second decode backend with its own halt set, stop-string
scan, deadline check and sampler. It was the path that quietly missed features
(the deterministic tool-call region switch among them), because every one of
them had to be implemented twice.

Ray serializes method calls per actor, so an inference request submitted during
training queues behind any in-flight ``train_batch`` and sees a consistent
weight snapshot. The web API path has no training loop to drain the queue, so
this runs ``synchronous=True``: requests are fulfilled in-place in
``request_generation``.
"""

from __future__ import annotations

from typing import Any

from praxis.inference.decode_backend import ModelBackend
from praxis.inference.generator import Generator


def MonoForwardGenerator(trainer: Any, tokenizer: Any) -> Generator:
    """Build a :class:`Generator` that decodes through the MF actor chain."""
    from praxis.trainers.mono_forward.hf_model import MonoForwardLM

    return Generator(
        tokenizer=tokenizer,
        device="cpu",  # actors run CPU-only; prompts must live on CPU
        backend=ModelBackend(MonoForwardLM(trainer), tokenizer),
        synchronous=True,
    )
