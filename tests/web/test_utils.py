"""Chat helpers (praxis/web/utils): prompt formatting and the request wait.

The wait in ``generate_from_messages`` is client-side: queued generations are
served from inside ``on_train_batch_end``, so the deadline is what bounds a chat
request's cost to the run (one 512-byte turn once stalled training for 208s).
"""

import threading
import time

import pytest

from praxis.inference.generator import Generator
from tests.stubs import _SlowBackend


def test_a_deadline_cut_turn_is_collected_rather_than_discarded(prose_tokenizer):
    """The deadline stops the DECODE, not the reply.

    `_process_single_request` returns the partial turn it was part-way through.
    The caller used to stop listening at exactly the deadline, so nobody ever
    collected it - the route answered `""`, and a client that had been watching
    that very text stream in replaced it with "Error: No response" at the end.
    """
    from praxis.web.utils.formatters import generate_from_messages

    backend = _SlowBackend(delay=0.01)
    gen = Generator(backend=backend, tokenizer=prose_tokenizer)
    gen.tools = {}

    # Served from a background "training loop", the way the real one does.
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
            tokenizer=prose_tokenizer,
            max_new_tokens=5000,
            timeout=0.5,
        )
    finally:
        stop.set()
        worker.join(timeout=5)

    assert backend.tokens_emitted > 0, "the model never got to write anything"
    assert reply, "the partial turn was thrown away"


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
