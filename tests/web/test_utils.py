import contextlib
import time

import pytest
import torch

from praxis.generation.generator import Generator
from praxis.tokenizers import create_tokenizer

# ------------------------------------------------------------------------------
# generation_deadline
# ------------------------------------------------------------------------------
# The request deadline, which is what bounds a chat request's cost to the run.
#
# Queued generations are served by ``GenerationQueueCallback`` from inside
# ``on_train_batch_end``, so they hold the training loop's turn. The wait in
# ``generate_from_messages`` is client-side only: before the deadline existed, giving up
# after 60s stopped us listening but did not stop the loop from decoding the whole turn.
# Measured on ``abstractinator-r``, where the encoder stack cannot cache and every
# decode step is a full forward: one 512-byte Discord turn stalled training for 208
# seconds, ~148 of them after the client had already timed out and thrown the eventual
# reply away.
#
# Two things have to hold, and neither is visible from the caller's side:
#
# - a request that expires BEFORE it is served must never run, - a request served just
# under the wire must stop decoding when it expires, rather than running to
# ``max_new_tokens``.


@pytest.fixture(scope="module")
def tokenizer():
    return create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format="prose"
    )


class _SlowBackend:
    """Spends ``delay`` seconds per token and never halts on a boundary.

    Not halting is the point: it isolates the deadline as the only thing that
    can end the decode, so a passing test cannot be passing because the model
    happened to stop. The per-token check mirrors what the real backend gets
    from transformers - ``ModelBackend`` turns the deadline into
    ``GenerationConfig.max_time``, which ``_get_stopping_criteria`` builds into
    a ``MaxTimeCriteria`` evaluated after every token - because the caller only
    calls this ONCE for a turn with no tool in it.
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

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
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


# ---------------------------------------------------------------------------
# a turn the deadline cut short is still a turn
# ---------------------------------------------------------------------------


def test_a_deadline_cut_turn_is_collected_rather_than_discarded(tokenizer):
    """The deadline stops the DECODE, not the reply.

    `_process_single_request` returns the partial turn it was part-way through.
    The caller used to stop listening at exactly the deadline, so nobody ever
    collected it - the route answered `""`, and a client that had been watching
    that very text stream in replaced it with "Error: No response" at the end.
    """
    from praxis.web.utils.formatters import generate_from_messages

    backend = _SlowBackend(delay=0.01)
    gen = _generator(tokenizer, backend)

    # Served from a background "training loop", the way the real one does.
    import threading

    stop = threading.Event()

    def drain():
        while not stop.is_set():
            gen.fulfill_requests(max_requests=1)
            time.sleep(0.02)

    worker = threading.Thread(target=drain, daemon=True)
    worker.start()
    try:
        reply = generate_from_messages(
            messages=[{"role": "user", "content": "hi"}],
            generator=gen,
            tokenizer=tokenizer,
            max_new_tokens=5000,
            timeout=0.5,
        )
    finally:
        stop.set()
        worker.join(timeout=5)

    assert backend.tokens_emitted > 0, "the model never got to write anything"
    assert reply, "the partial turn was thrown away"


# ------------------------------------------------------------------------------
# chat_formats
# ------------------------------------------------------------------------------
# Tests for the ``chat_formats`` registry and the text-boundary (prose) format.
#
# The invariants worth pinning are the ones that silently produce a broken run rather
# than an exception:
#
# - the `default` profile must stay byte-identical, since every existing checkpoint's
# data pipeline depends on it, - the boundary that ENDS a generated turn must be a
# trained target (the defect `prose` exists to remove), - a stop-string halt must not
# re-fire on the boundary it resumed from, or the tool loop returns zero new tokens
# forever, - the tool flow's three boundaries must classify unambiguously.


def test_api_roles_exclude_runtime_injected_turns(prose_tokenizer):
    """A client must not be able to fabricate a tool result."""
    from praxis.web.utils.formatters import format_messages_to_chatml

    with pytest.raises(ValueError, match="Invalid role"):
        format_messages_to_chatml([{"role": "tool", "content": "999"}], prose_tokenizer)
    with pytest.raises(ValueError, match="Invalid role"):
        format_messages_to_chatml([{"role": "call", "content": "{}"}], prose_tokenizer)
    # Ordinary roles still render.
    assert format_messages_to_chatml(
        [{"role": "user", "content": "hi"}], prose_tokenizer
    ).endswith("assistant\n\n")
