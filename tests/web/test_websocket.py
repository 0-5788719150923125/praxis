"""Generation streaming (praxis/web/websocket): the side channel that carries a reply to the browser as it is written.

``POST /messages/`` is unchanged - one request, one final JSON reply, still
authoritative. The deltas ride the ``/realtime`` socket the client already has
open, keyed by an id the client mints. The properties worth pinning are the
ones whose failure is silent: opting out has to change nothing, and a broken
socket has to cost the preview and nothing else.
"""

import sys

from praxis.web.websocket.generation_stream import stream_callbacks
from praxis.web.websocket.realtime import NAMESPACE


def test_the_channel_is_not_named_for_any_one_passenger():
    """It carries metrics, cache invalidations and generation deltas, so it is
    named for what it is: the server's live push channel."""
    from praxis.web.app import socketio
    from praxis.web.websocket import (
        generation_stream,
        realtime,
        setup_realtime_namespace,
    )

    assert NAMESPACE == "/realtime"
    # One place the name lives: a publisher that spelled its own literal could
    # drift from the namespace the server actually serves.
    assert generation_stream.NAMESPACE is realtime.NAMESPACE

    # ...and that is the namespace the server registers.
    setup_realtime_namespace(socketio)
    assert NAMESPACE in socketio.server.namespace_handlers


def test_no_stream_id_means_no_streaming():
    """How a caller opts out. `Generator` then builds no streamer at all, so a
    client that does not want deltas pays nothing for them."""
    assert stream_callbacks(None) == (None, None, None)
    assert stream_callbacks("") == (None, None, None)


def test_deltas_carry_the_clients_own_id(emitted):
    on_text, _, _ = stream_callbacks("abc123")
    on_text("Hello")
    on_text(" there")

    assert [f[0] for f in emitted] == ["gen_delta", "gen_delta"]
    assert all(f[1]["id"] == "abc123" for f in emitted)
    assert "".join(f[1]["text"] for f in emitted) == "Hello there"
    # The shared live push channel, not a name of this feature's own.
    assert all(f[2] == NAMESPACE for f in emitted)


def test_frames_are_sequenced(emitted):
    """Deltas are produced on the training thread and the final reply comes
    back over HTTP, so ARRIVAL order is not guaranteed even though production
    order is. The client uses the sequence to drop a straggler."""
    on_text, on_reset, _ = stream_callbacks("s1")
    on_text("a")
    on_reset()
    on_text("b")

    seqs = [f[1]["seq"] for f in emitted]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert [f[0] for f in emitted] == ["gen_delta", "gen_reset", "gen_delta"]
    # A reset means "drop what you have", not "replace it with this".
    assert "text" not in emitted[1][1]


def test_a_broken_socket_never_reaches_the_training_loop(monkeypatch):
    """These callbacks fire from inside `on_train_batch_end` for a queued
    request. A dropped frame costs a chunk of preview text; an exception would
    cost the training step."""

    class _Broken:
        def emit(self, *args, **kwargs):
            raise RuntimeError("socket is gone")

    import praxis.web.app  # noqa: F401

    monkeypatch.setattr(sys.modules["praxis.web.app"], "socketio", _Broken())

    on_text, on_reset, on_tool = stream_callbacks("s1")
    on_text("this must not raise")
    on_reset()
    on_tool("read_file")
