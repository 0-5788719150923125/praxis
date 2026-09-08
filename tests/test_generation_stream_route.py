"""The side channel that carries a reply to the browser as it is written.

``POST /messages/`` is unchanged - one request, one final JSON reply, still
authoritative. The deltas ride the ``/realtime`` socket the client already has
open, keyed by an id the client mints. The properties worth pinning are the
ones whose failure is silent: opting out has to change nothing, and a broken
socket has to cost the preview and nothing else.
"""

import sys

import pytest

from praxis.web.websocket.generation_stream import stream_callbacks
from praxis.web.websocket.realtime import NAMESPACE


@pytest.fixture
def emitted(monkeypatch):
    """Capture what would go out on the socket."""
    frames = []

    class _Socket:
        def emit(self, event, payload, namespace=None):
            frames.append((event, payload, namespace))

    # Reached through sys.modules on purpose: `praxis/web/__init__.py` does
    # `from .app import app`, which rebinds the attribute `praxis.web.app` to
    # the FLASK OBJECT - so `import praxis.web.app as m` binds the Flask app,
    # not the module, and patching it would silently do nothing.
    import praxis.web.app  # noqa: F401  (ensure it is in sys.modules)

    monkeypatch.setattr(sys.modules["praxis.web.app"], "socketio", _Socket())
    return frames


@pytest.fixture
def client():
    """A throwaway Flask app carrying only the generation blueprint.

    Deliberately NOT `praxis.web.app.app`: that is a module-level singleton the
    rest of the web tests share, and both mutating its config and registering
    blueprints on it leak - the second registration makes `APIServer` fail to
    start in whatever test runs next. A blueprint can be registered on any
    number of apps, and this route only reads `current_app.config`.
    """
    from flask import Flask

    from praxis.web.routes.generation import generation_bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(generation_bp)
    return app


def test_the_channel_is_not_named_for_any_one_passenger():
    """It carries metrics, cache invalidations and generation deltas, so it is
    named for what it is - the server's live push channel. It was
    ``/metrics-live`` while metrics were all it carried, which stopped being
    true the moment inference started riding it."""
    from praxis.web.app import socketio
    from praxis.web.websocket import setup_realtime_namespace
    from praxis.web.websocket import generation_stream, realtime

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
    assert stream_callbacks(None) == (None, None)
    assert stream_callbacks("") == (None, None)


def test_deltas_carry_the_clients_own_id(emitted):
    on_text, on_reset = stream_callbacks("abc123")
    on_text("Hello")
    on_text(" there")

    assert [f[0] for f in emitted] == ["gen_delta", "gen_delta"]
    assert all(f[1]["id"] == "abc123" for f in emitted)
    assert "".join(f[1]["text"] for f in emitted) == "Hello there"
    # The shared live push channel, not a name of this feature's own.
    assert all(f[2] == NAMESPACE for f in emitted)


def test_frames_are_sequenced(emitted):
    """The deltas are produced on the training thread and the final reply comes
    back over HTTP, so their ARRIVAL order is not guaranteed even though their
    production order is. The client uses this to drop a straggler."""
    on_text, on_reset = stream_callbacks("s1")
    on_text("a")
    on_reset()
    on_text("b")

    seqs = [f[1]["seq"] for f in emitted]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert [f[0] for f in emitted] == ["gen_delta", "gen_reset", "gen_delta"]


def test_a_reset_carries_no_text(emitted):
    """It means "drop what you have", not "replace it with this"."""
    _, on_reset = stream_callbacks("s1")
    on_reset()
    assert "text" not in emitted[0][1]


def test_a_broken_socket_never_reaches_the_training_loop(monkeypatch):
    """These callbacks fire from inside `on_train_batch_end` for a queued
    request. A dropped frame costs a chunk of preview text; an exception would
    cost the training step."""

    class _Broken:
        def emit(self, *args, **kwargs):
            raise RuntimeError("socket is gone")

    import praxis.web.app  # noqa: F401

    monkeypatch.setattr(sys.modules["praxis.web.app"], "socketio", _Broken())

    on_text, on_reset = stream_callbacks("s1")
    on_text("this must not raise")
    on_reset()


def test_the_route_streams_and_still_returns_the_whole_reply(emitted, client):
    """End to end through the Flask route: the deltas go out AND the response
    body is the same complete reply it always was."""

    class _StreamingGenerator:
        """Publishes a reply in pieces, then returns it whole - the shape a
        real `Generator` with a `ReplyStreamer` attached produces."""

        def __init__(self):
            self.pending = None

        def request_generation(
            self, prompt, kwargs, deadline=None, on_text=None, on_reset=None
        ):
            for chunk in ("It ", "is ", "noon."):
                if on_text:
                    on_text(chunk)
            self.pending = "It is noon."
            return "rid"

        def get_result(self, request_id):
            result, self.pending = self.pending, None
            return result

    class _Tokenizer:
        bos_token = "[BOS]"
        eos_token = "[EOS]"
        sep_token = "[SEP]"

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False
        ):
            return "[BOS]user\nhi[SEP]\n[BOS]assistant\n"

        def convert_tokens_to_ids(self, token):
            return None

    client.config["generator"] = _StreamingGenerator()
    client.config["tokenizer"] = _Tokenizer()
    client.config.pop("api_server", None)

    with client.test_client() as http:
        response = http.post(
            "/messages/",
            json={
                "messages": [{"role": "user", "content": "what time is it?"}],
                "stream_id": "tab-1",
            },
        )

    assert response.status_code == 200
    assert "It is noon." in response.get_json()["response"]

    deltas = [f[1]["text"] for f in emitted if f[0] == "gen_delta"]
    assert "".join(deltas) == "It is noon."
    assert all(f[1]["id"] == "tab-1" for f in emitted)


def test_the_route_without_a_stream_id_emits_nothing(emitted, client):
    """The unchanged path. A client that never learned about streaming - or one
    whose socket is down - gets exactly the behavior it had before."""

    class _Generator:
        def request_generation(
            self, prompt, kwargs, deadline=None, on_text=None, on_reset=None
        ):
            assert on_text is None and on_reset is None
            return "rid"

        def get_result(self, request_id):
            return "[BOS]assistant\nquiet reply[SEP]"

    class _Tokenizer:
        bos_token = "[BOS]"
        eos_token = "[EOS]"
        sep_token = "[SEP]"

        def apply_chat_template(self, messages, **kwargs):
            return "[BOS]user\nhi[SEP]\n[BOS]assistant\n"

        def convert_tokens_to_ids(self, token):
            return None

    client.config["generator"] = _Generator()
    client.config["tokenizer"] = _Tokenizer()
    client.config.pop("api_server", None)

    with client.test_client() as http:
        response = http.post(
            "/messages/", json={"messages": [{"role": "user", "content": "hi"}]}
        )

    assert response.status_code == 200
    assert response.get_json()["response"] == "quiet reply"
    assert emitted == []
