"""Fixtures shared across the praxis.web tests.

``praxis.web.app.app`` is a module-level singleton: ``APIServer.start()``
writes its config and registers every blueprint on it. So the session runs
exactly ONE live server, and tests that only need a route build a throwaway
Flask app around that route's blueprint instead.
"""

import socket
import sys

import pytest

from praxis.web import APIServer


class MockGenerator:
    """Answers every request at once with a numbered reply."""

    def __init__(self):
        self.model = None
        self.request_counter = 0
        self.last_deadline = None

    def request_generation(self, prompt: str, kwargs: dict, deadline=None, **_) -> str:
        """``**_`` absorbs the streaming callbacks the real `Generator` takes,
        so an addition to that interface is not an unrelated test failure."""
        self.request_counter += 1
        self.last_deadline = deadline
        return f"request_{self.request_counter}"

    def get_result(self, request_id: str) -> str:
        if "request_" in request_id:
            return f"Generated response for {request_id}"
        return None


class FakeTokenizer:
    """Renders messages in the default `[BOS]role\\ncontent[SEP]` envelope."""

    bos_token = "[BOS]"
    eos_token = "[EOS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **_
    ):
        result = "".join(
            f"{self.bos_token}{m.get('role', 'user')}\n{m.get('content', '')}"
            f"{self.sep_token}\n"
            for m in messages
        )
        if add_generation_prompt:
            result += f"{self.bos_token}assistant\n"
        return result

    def convert_tokens_to_ids(self, token):
        return None


def _free_port() -> int:
    """A port nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_generator():
    return MockGenerator()


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def free_port():
    return _free_port()


@pytest.fixture(scope="session")
def api_server(tmp_path_factory):
    """The session's one live APIServer, serving a frontend built into tmp.

    The build goes to a temporary static folder rather than praxis/web/static,
    which is what a live dashboard on this machine is serving.
    """
    import praxis.web.src.build as build
    from praxis.web.app import app

    static = tmp_path_factory.mktemp("static")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build, "STATIC_DIR", static)
        build.build_dev()
        mp.setattr(app, "static_folder", str(static))

        server = APIServer(
            generator=MockGenerator(),
            host="localhost",
            port=_free_port(),
            tokenizer=FakeTokenizer(),
            param_stats={"total": 1000000, "trainable": 900000},
            seed=42,
            truncated_hash="test12345",
            full_hash="test1234567890abcdef",
            dev_mode=True,
            launch_command="python test.py",
        )
        server.start()  # returns once the listener is bound
        yield server
        server.stop()


@pytest.fixture(scope="session")
def api_url(api_server):
    return f"http://localhost:{api_server.port}"


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


class _BareModel:
    """Stands in for the live model the generator holds: no classifier,
    criterion or encoder, so only what a test stashes on it shows up in a
    snapshot."""

    classifier = None
    criterion = None
    encoder = None


@pytest.fixture
def bare_model():
    return _BareModel()


@pytest.fixture
def compute_profile():
    """A `ComputeProfiler` stash as `/api/classifier_snapshots` serves it."""
    return {
        "compute_profile": {
            "total_ms": 100.0,
            "coverage": 0.71,
            "samples": 3,
            "interval": 100,
            "ema_alpha": 0.2,
            "groups": [
                {
                    "name": "ArcAttention",
                    "ms": 50.0,
                    "share": 0.5,
                    "calls": 2.0,
                    "outside": False,
                    "residual": False,
                    "children": [
                        {
                            "name": "decoder.locals.0.block.attn",
                            "ms": 50.0,
                            "share": 0.5,
                            "fwd_ms": 30.0,
                            "bwd_ms": 20.0,
                            "calls": 2.0,
                        }
                    ],
                },
                {
                    "name": "(outside model)",
                    "ms": 50.0,
                    "share": 0.5,
                    "calls": 0.0,
                    "outside": True,
                    "residual": False,
                    "children": [],
                },
            ],
        }
    }
