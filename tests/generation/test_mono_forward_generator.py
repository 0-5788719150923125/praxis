"""Mono-Forward (in-process + Ray) correctness tests.

Consolidated test suite for everything MF-related: the in-process
math harness, the Ray-based pipelined trainer, checkpoint roundtrip,
and the Phase 5 live-inference-during-training hook. See
``PHASE_5.md`` for the project plan and decision rationale (D1..D8).

Tests that don't need Ray (the in-process math + detach checks) run
anywhere. Everything else is gated behind ``requires_ray`` so the host
venv (Python >= 3.14, no Ray wheels) skips them cleanly; run under
Docker compose to exercise the full suite:

    docker compose -f compose.yml run --rm --no-deps agent \\
        /workspace/.venv/bin/python -m pytest tests/test_mono_forward.py -v
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import IterableDataset

from praxis import PraxisConfig
from praxis.modeling import PraxisForCausalLM

try:
    import ray  # noqa: F401

    HAS_RAY = True
except ImportError:
    HAS_RAY = False

requires_ray = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")

# Import the Ray trainer lazily so module import on non-Ray hosts still
# runs (the in-process math tests below don't need it).
if HAS_RAY:
    from praxis.trainers.mono_forward import MonoForwardTrainer
else:
    MonoForwardTrainer = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _mf_config(num_layers: int = 4) -> PraxisConfig:
    """Tiny CPU-only config that mirrors ``experiments/mike.yml`` in shape.

    ``num_layers`` is parameterised because different tests want
    different depths - the pipeline-overlap assertion needs at least 4
    layers for ``pipeline_in_flight_max >= num_layers - 1`` to be a
    meaningful "actually pipelined" proof, but the math and
    single-fit tests run faster at depth 2.
    """
    return PraxisConfig(
        vocab_size=256,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=num_layers,
        num_layers=num_layers,
        max_length=64,
        decoder_type="sequential",
        attention_type="modular",
        encoder_type=None,
        tie_weights=False,
    )


class _FixedBatchDataset(IterableDataset):
    """Deterministic batch stream - yields the same sample forever.

    Memorisation workload: if the MF plumbing is sound the loss must
    trend downward. Used by every Ray test in this file.
    """

    def __init__(self, vocab_size: int, batch_size: int, seq_len: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.batch = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)

    def __iter__(self):
        while True:
            yield {"input_ids": self.batch}


class _SyntheticDataModule:
    """Minimal stand-in for a Lightning DataModule.

    ``MonoForwardTrainer.fit`` only calls ``train_dataloader()`` and
    iterates what it gets, so the real Praxis datamodule is overkill
    for a correctness smoke.
    """

    def __init__(self, dataset: _FixedBatchDataset) -> None:
        self._dataset = dataset

    def train_dataloader(self):
        return iter(self._dataset)


# ---------------------------------------------------------------------------
# Phase 6: Flask/API-facing generator adapter
# ---------------------------------------------------------------------------


class _ToyTokenizer:
    """Minimal tokenizer stub that satisfies MonoForwardGenerator's needs.

    The real Praxis tokenizer (``StandardTokenizer``) is heavy to build
    and reads dataset metadata. A character-level stub covers the surface
    ``Generator`` touches - ``encode``, ``decode``, ``apply_chat_template``,
    and the id lookup ``ChatFormat.suppressed_token_ids`` needs - without
    pulling in any of the training-time tokenization machinery.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.sep_token_id = 3
        self.bos_token = "<s>"
        self.eos_token = "</s>"

    def convert_tokens_to_ids(self, token):
        """Named control tokens only; anything else is unknown (None).

        ``ChatFormat.suppressed_token_ids`` asks for the ids a format's data
        never makes a target, so the Generator can keep them out of the
        sampler.
        """
        named = {
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
        }
        if isinstance(token, (list, tuple)):
            return [named.get(t) for t in token]
        return named.get(token)

    def encode(self, text: str) -> list:
        # Byte-level: map each character to ``ord(c) % vocab_size``,
        # which is deterministic and guaranteed to fall within the
        # toy vocab range the tests use.
        if not text:
            return [self.bos_token_id]
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list, skip_special_tokens: bool = False) -> str:
        special = {0, 1, 2, 3}
        chars = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(chr(int(i) % 128))
        return "".join(chars)

    def apply_chat_template(
        self, messages: list, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        return "\n".join(parts)


class _GeneratorBridgeTrainer(MonoForwardTrainer if HAS_RAY else object):
    """Runs a :class:`MonoForwardGenerator` request during checkpoint save.

    Same trick as :class:`_IdleGenerateTrainer`: override
    ``_save_checkpoint`` to run the adapter while the actor set is
    still alive. This lets us exercise the full
    tokenize → trainer.generate → decode pipeline without spinning
    up a real API server thread.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.string_prompt_result = None
        self.message_prompt_result = None

    def _save_checkpoint(self, model_host, actors):  # type: ignore[override]
        from praxis.generation import MonoForwardGenerator

        tokenizer = _ToyTokenizer()
        gen = MonoForwardGenerator(trainer=self, tokenizer=tokenizer)

        # String prompt path - the ``/input`` API route shape.
        request_id = gen.request_generation(
            "hello", {"max_new_tokens": 4, "do_sample": False}
        )
        self.string_prompt_result = gen.get_result(request_id)

        # Message-list prompt path - the ``/messages`` API route shape.
        request_id = gen.request_generation(
            [{"role": "user", "content": "hi"}],
            {"max_new_tokens": 3, "do_sample": False},
        )
        self.message_prompt_result = gen.get_result(request_id)

        super()._save_checkpoint(model_host, actors)


@requires_ray
def test_mono_forward_generator_api_bridge(tmp_path):
    """The API-facing :class:`MonoForwardGenerator` decodes to strings.

    End-to-end coverage of the Phase 6 inference-routing path: submit
    both a string prompt and a messages-list prompt, verify that
    ``get_result`` returns a non-empty decoded string for each. The
    actual training weights are undertrained, so we don't assert on
    the content of the reply - only that the plumbing is sound.
    """
    torch.manual_seed(0)
    config = _mf_config(num_layers=3)
    model = PraxisForCausalLM(config)
    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=10, seed=3
    )

    trainer = _GeneratorBridgeTrainer(
        max_steps=6, log_every_n_steps=6, cache_dir=str(tmp_path)
    )
    trainer.fit(model, _SyntheticDataModule(dataset))

    assert isinstance(trainer.string_prompt_result, str)
    assert len(trainer.string_prompt_result) > 0
    assert trainer.string_prompt_result.startswith("hello"), (
        f"expected decoded text to preserve prompt prefix, got "
        f"{trainer.string_prompt_result!r}"
    )

    assert isinstance(trainer.message_prompt_result, str)
    assert len(trainer.message_prompt_result) > 0
    # apply_chat_template flattened the message into "user: hi", so
    # the decoded echo should at least contain the user role tag.
    assert "user" in trainer.message_prompt_result, (
        f"messages-list path lost the chat template: "
        f"{trainer.message_prompt_result!r}"
    )


class _StubTrainer:
    """The whole surface :class:`MonoForwardLM` needs: a config and one forward.

    That is the point of the face - Mono-Forward's only real difference from
    in-process decoding is where the forward runs, so everything above it
    (sampling, halting, the request queue) is the ordinary path and needs no
    stub at all.
    """

    def __init__(self, token: int = 65, num_layers: int = 2):
        self._config = _mf_config(num_layers=num_layers)
        self.token = token
        self.calls = 0

    def infer_logits(self, input_ids):
        self.calls += 1
        b, t = input_ids.shape
        logits = torch.full((b, t, self._config.vocab_size), -10.0)
        logits[:, :, self.token] = 10.0
        return logits


def _stub_generator(trainer=None):
    from praxis.generation import MonoForwardGenerator

    return MonoForwardGenerator(
        trainer=trainer or _StubTrainer(), tokenizer=_ToyTokenizer()
    )


def test_mono_forward_decodes_through_the_standard_backend():
    """Mono-Forward is no longer a second decode backend.

    Its weights live on Ray actors, so the forward differs - and nothing else
    does. `MonoForwardLM` is a PreTrainedModel over the actor chain handed to
    the same `ModelBackend` every other run uses, which is what gets this path
    the prepared logits processors, the stop strings, the deadline and the
    streamer instead of a hand-rolled copy of each.
    """
    from praxis.generation.decode_backend import ModelBackend
    from praxis.trainers.mono_forward.hf_model import MonoForwardLM

    gen = _stub_generator()
    assert isinstance(gen.backend, ModelBackend)
    assert isinstance(gen.backend.model, MonoForwardLM)


def test_mono_forward_generator_result_is_popped_once():
    """``get_result`` is destructive: a second lookup returns None.

    No Ray required - this test only exercises the adapter's bookkeeping, with
    a mock trainer whose forward returns a fixed distribution.
    """
    gen = _stub_generator()
    request_id = gen.request_generation("x", {"max_new_tokens": 2})

    first = gen.get_result(request_id)
    assert isinstance(first, str) and len(first) > 0

    # Second lookup of the same id must be None - the backprop
    # Generator's ``get_result`` is destructive and integrations
    # depend on that.
    assert gen.get_result(request_id) is None


def test_mono_forward_generator_fulfill_requests_is_noop():
    """fulfill_requests exists for interface parity but does no work.

    The backprop Generator defers the heavy lifting to
    ``fulfill_requests`` (called from a background thread). The MF
    adapter runs synchronously in ``request_generation``, so
    ``fulfill_requests`` should always find the queue empty and
    return 0.
    """
    gen = _stub_generator()
    gen.request_generation("x", {"max_new_tokens": 1})
    assert gen.fulfill_requests() == 0
    assert gen.fulfill_requests(max_requests=10) == 0


# ---------------------------------------------------------------------------
# Unproducible control ids must not be sampleable on this decode path either.
# A chat format declares ids its data never makes a target
# (ChatFormat.suppressed_token_ids); under prose [BOS]/[SEP] hold ids 1 and 3
# of a 260-wide byte head, so leaving them reachable puts a bracketed token the
# model was structurally forbidden from learning into generations. This used to
# need a hand-rolled mask because the MF loop sampled for itself.
# ---------------------------------------------------------------------------


def test_suppressed_tokens_are_unreachable_on_the_served_path():
    """The Generator puts the format's suppression list in step_kwargs and the
    backend hands it to transformers, which builds a
    SuppressTokensLogitsProcessor - no Mono-Forward-specific mask involved."""
    trainer = _StubTrainer(token=3)  # the model would emit id 3 every step
    gen = _stub_generator(trainer)

    rid = gen.request_generation(
        "x", {"max_new_tokens": 6, "do_sample": False, "suppress_tokens": [1, 3]}
    )
    out = gen.get_result(rid)
    assert out is not None
    # Decoded through the toy tokenizer, so assert on the ids the face saw.
    assert trainer.calls > 0
    ids = _ToyTokenizer().encode(out)
    assert 3 not in ids[1:], f"a suppressed id was sampled anyway: {ids}"
