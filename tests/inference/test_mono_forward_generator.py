"""``MonoForwardGenerator``: serving a Mono-Forward run through the standard backend.

Mono-Forward's weights live on per-layer workers, so its forward differs - and
nothing else does. ``MonoForwardLM`` is a ``PreTrainedModel`` over the worker chain,
handed to the same ``ModelBackend`` every other run uses. The Ray test is skipped on
hosts without Ray; run it under Docker compose:

    docker compose -f compose.yml run --rm --no-deps agent \\
        .venv/bin/python -m pytest tests/inference/test_mono_forward_generator.py
"""

from __future__ import annotations

import pytest
import torch

from praxis.inference import MonoForwardGenerator
from praxis.inference.decode_backend import ModelBackend
from praxis.modeling import PraxisForCausalLM
from praxis.trainers.mono_forward import MonoForwardTrainer
from praxis.trainers.mono_forward.hf_model import MonoForwardLM
from praxis.trainers.mono_forward.inprocess_trainer import InProcessMonoForwardTrainer
from tests.stubs import (
    _FixedBatchDataset,
    _mf_config,
    _stub_generator,
    _StubTrainer,
    _SyntheticDataModule,
    _ToyTokenizer,
    requires_ray,
)


def _serve(trainer):
    """The worker chain's own forward, then both API prompt shapes through a
    generator over ``trainer``. The forward is recorded separately because the
    Generator turns a failed decode into a reply that is just the prompt."""
    trainer.logits_shape = tuple(trainer.infer_logits(torch.tensor([[104, 105]])).shape)
    gen = MonoForwardGenerator(trainer=trainer, tokenizer=_ToyTokenizer())
    # String prompt - the ``/input`` route shape.
    rid = gen.request_generation("hello", {"max_new_tokens": 4, "do_sample": False})
    string_result = gen.get_result(rid)
    # Message-list prompt - the ``/messages`` route shape.
    rid = gen.request_generation(
        [{"role": "user", "content": "hi"}],
        {"max_new_tokens": 3, "do_sample": False},
    )
    return string_result, gen.get_result(rid)


class _GeneratorBridgeTrainer(MonoForwardTrainer):
    """Serves a request during checkpoint save, while the actor set is alive."""

    results = (None, None)
    logits_shape = None

    def _save_checkpoint(self, model_host, actors):  # type: ignore[override]
        self.results = _serve(self)
        super()._save_checkpoint(model_host, actors)


class _InProcessBridgeTrainer(InProcessMonoForwardTrainer):
    """The same bridge over the in-process workers."""

    results = (None, None)
    logits_shape = None

    def _save_checkpoint_inprocess(self, **kwargs):  # type: ignore[override]
        self.results = _serve(self)
        super()._save_checkpoint_inprocess(**kwargs)


def _fit_and_serve(trainer_cls, tmp_path):
    torch.manual_seed(0)
    config = _mf_config(num_layers=3)
    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=10, seed=3
    )
    trainer = trainer_cls(max_steps=6, log_every_n_steps=6, cache_dir=str(tmp_path))
    trainer.fit(PraxisForCausalLM(config), _SyntheticDataModule(dataset))
    assert trainer.logits_shape is not None and trainer.logits_shape[:2] == (1, 2)
    string_result, message_result = trainer.results

    # Undertrained weights, so only the plumbing is asserted: the prompt
    # survives the round trip, and the chat template flattened the message
    # into "user: hi".
    assert isinstance(string_result, str) and string_result.startswith("hello")
    assert isinstance(message_result, str) and "user" in message_result


@requires_ray
def test_mono_forward_generator_api_bridge(tmp_path):
    _fit_and_serve(_GeneratorBridgeTrainer, tmp_path)


@pytest.mark.xfail(
    strict=True,
    reason="InProcessMonoForwardTrainer inherits MonoForwardTrainer.infer_logits, "
    "which imports ray and calls .remote() on plain LocalLayerWorker objects",
)
def test_mono_forward_generator_serves_an_in_process_run(tmp_path):
    """``swap_inference_generator`` routes every MonoForwardTrainer subclass
    through this generator, the in-process one included."""
    _fit_and_serve(_InProcessBridgeTrainer, tmp_path)


def test_generator_is_a_synchronous_standard_backend():
    """The ordinary ``ModelBackend`` over ``MonoForwardLM``, which is what gives
    this path the prepared processors, stop strings, deadline and streamer. It
    runs each request in ``request_generation``, so ``fulfill_requests`` has
    nothing left to do, and ``get_result`` is destructive like the backprop
    Generator's, which integrations depend on."""
    gen = _stub_generator()
    assert isinstance(gen.backend, ModelBackend)
    assert isinstance(gen.backend.model, MonoForwardLM)

    rid = gen.request_generation("x", {"max_new_tokens": 2})
    assert gen.fulfill_requests() == 0
    first = gen.get_result(rid)
    assert isinstance(first, str) and len(first) > 0
    assert gen.get_result(rid) is None


class _RunnerUpTrainer(_StubTrainer):
    """Prefers ``token`` every step, with ``runner_up`` a clear second."""

    def __init__(self, token, runner_up):
        super().__init__(token=token)
        self.runner_up = runner_up

    def infer_logits(self, input_ids):
        logits = super().infer_logits(input_ids)
        logits[:, :, self.runner_up] = 5.0
        return logits


def test_suppressed_tokens_are_unreachable_on_the_served_path():
    """Unproducible control ids must not be sampleable here either. The
    Generator puts the format's suppression list in step_kwargs and the backend
    hands it to transformers - no Mono-Forward-specific mask involved."""
    trainer = _RunnerUpTrainer(token=3, runner_up=ord("A"))
    gen = _stub_generator(trainer)

    rid = gen.request_generation(
        "x",
        {
            "max_new_tokens": 6,
            "do_sample": False,
            "suppress_tokens": [1, 3],
            # Keep id 3 visible in the decode if it were ever sampled.
            "skip_special_tokens": False,
        },
    )
    assert gen.get_result(rid) == "x" + "A" * 6
