"""The request deadline, which is what bounds a chat request's cost to the run.

Queued generations are served by ``GenerationQueueCallback`` from inside
``on_train_batch_end``, so they hold the training loop's turn. The wait in
``generate_from_messages`` is client-side only: before the deadline existed,
giving up after 60s stopped us listening but did not stop the loop from
decoding the whole turn. Measured on ``abstractinator-r``, where the encoder
stack cannot cache and every decode step is a full forward: one 512-byte
Discord turn stalled training for 208 seconds, ~148 of them after the client
had already timed out and thrown the eventual reply away.

Two things have to hold, and neither is visible from the caller's side:

- a request that expires BEFORE it is served must never run,
- a request served just under the wire must stop decoding when it expires,
  rather than running to ``max_new_tokens``.
"""

import contextlib
import time

import pytest
import torch

from praxis.generation.generator import Generator
from praxis.tokenizers import create_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format="prose"
    )


class _SlowBackend:
    """Spends ``delay`` seconds per token and never halts on a boundary.

    Not halting is the point: it isolates the deadline as the only thing that
    can end the decode, so a passing test cannot be passing because the model
    happened to stop. The per-token deadline check mirrors what a real backend
    does - ``ModelBackend`` hands a ``DeadlineCriteria`` to the transformers
    loop, ``MonoForwardBackend`` checks it in its yield loop - because the
    caller only calls this ONCE for a turn with no tool in it.
    """

    model = None
    default_sampling_temperature = None

    def __init__(self, delay=0.01):
        self.delay = delay
        self.device = "cpu"
        self.max_positions = None
        self.calls = 0
        self.tokens_emitted = 0

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def generate_until_halt(self, tokens, step_kwargs, deadline=None):
        self.calls += 1
        budget = int(step_kwargs.get("max_new_tokens", 100))
        for _ in range(budget):
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(self.delay)
            # One ordinary byte ('a'), which is not a boundary under prose.
            nxt = torch.tensor([[ord("a")]], dtype=torch.long)
            tokens = torch.cat([tokens, nxt], dim=-1)
            self.tokens_emitted += 1
        return tokens


def _generator(tokenizer, backend):
    gen = Generator(backend=backend, tokenizer=tokenizer)
    gen.tools = {}
    return gen


def test_expired_request_is_never_served(tokenizer):
    """The stall the deadline exists to prevent: nobody is listening, so the
    training loop must not spend a single forward on it."""
    backend = _SlowBackend()
    gen = _generator(tokenizer, backend)

    rid = gen.request_generation(
        "user\n\nhi\n\nassistant\n\n",
        {"max_new_tokens": 5000},
        deadline=time.time() - 1.0,
    )
    served = gen.fulfill_requests(max_requests=1)

    assert backend.calls == 0, "an abandoned request still ran the model"
    # Dropped, not silently forgotten: a late poller gets a falsy answer rather
    # than waiting out its own timeout on a request that will never run.
    assert gen.get_result(rid) == ""
    assert served == 0, "a drop must not spend the per-step generation budget"


def test_drops_do_not_consume_the_request_budget(tokenizer):
    """``max_requests`` bounds GENERATION per step. Expired requests run none,
    so a burst of them must clear in one drain rather than one step each."""
    backend = _SlowBackend()
    gen = _generator(tokenizer, backend)

    expired = time.time() - 1
    stale = [
        gen.request_generation("user\n\nhi\n\nassistant\n\n", {}, deadline=expired)
        for _ in range(5)
    ]
    live = gen.request_generation(
        "user\n\nhi\n\nassistant\n\n",
        {"max_new_tokens": 2},
        deadline=time.time() + 30,
    )

    assert gen.fulfill_requests(max_requests=1) == 1
    assert all(gen.get_result(r) == "" for r in stale)
    assert gen.get_result(live) is not None


def test_decode_stops_at_the_deadline(tokenizer):
    """Served under the wire, then expires mid-decode.

    This is the case the between-steps check alone did NOT cover: a turn with
    no tool call enters ``generate_until_halt`` exactly once, so the deadline
    has to reach inside the decode or the request keeps the training loop for
    the full ``max_new_tokens`` regardless.
    """
    backend = _SlowBackend(delay=0.01)
    gen = _generator(tokenizer, backend)

    rid = gen.request_generation(
        "user\n\nhi\n\nassistant\n\n",
        {"max_new_tokens": 5000},
        deadline=time.time() + 0.3,
    )
    started = time.time()
    gen.fulfill_requests(max_requests=1)
    elapsed = time.time() - started

    assert elapsed < 5.0, f"decode ran {elapsed:.1f}s past a 0.3s deadline"
    assert backend.calls == 1, "the turn should be one decode call"
    assert backend.tokens_emitted < 5000, "decode spent the whole budget anyway"
    # The partial turn still comes back rather than being discarded: reply_start
    # is the runtime's own offset and does not depend on halting cleanly.
    result = gen.get_result(rid)
    assert result is not None and result.reply_start is not None


def test_no_deadline_means_no_limit(tokenizer):
    """``/input`` polls forever and passes no deadline; that must keep working
    exactly as before."""
    backend = _SlowBackend(delay=0.0)
    gen = _generator(tokenizer, backend)

    rid = gen.request_generation("user\n\nhi\n\nassistant\n\n", {"max_new_tokens": 8})
    gen.fulfill_requests(max_requests=1)

    assert backend.tokens_emitted == 8
    assert gen.get_result(rid) is not None
