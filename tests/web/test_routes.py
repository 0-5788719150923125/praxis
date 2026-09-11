"""HTTP routes (praxis/web/routes): one section per route module.

Routes that only read ``current_app.config`` are exercised on a throwaway Flask
app carrying their blueprint; the few that need the full app (static files, git,
agents, the home page) go through the session's live server.
"""

import io
import json
import sqlite3
import zipfile
from unittest.mock import Mock, patch

import flask
import pytest
import requests
import torch
import torch.nn as nn
import yaml
from flask import Flask

from praxis.metrics.training_metrics import TRAINING_METRIC_REGISTRY, X_AXIS_REGISTRY
from praxis.web.routes import print as print_route
from praxis.web.routes import register_routes
from praxis.web.routes.cards import cards_bp
from praxis.web.routes.dynamics import dynamics_bp
from praxis.web.websocket.realtime import NAMESPACE

# ------------------------------------------------------------------------------
# generation: /input and /messages
# ------------------------------------------------------------------------------


@pytest.fixture
def generation_app(fake_tokenizer):
    """A throwaway app carrying only the generation blueprint.

    Deliberately NOT `praxis.web.app.app`: mutating that singleton's config, or
    registering blueprints on it a second time, leaks into every later test.
    """
    from praxis.web.routes.generation import generation_bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["tokenizer"] = fake_tokenizer
    app.register_blueprint(generation_bp)
    return app


def _post(app, path, generator, **body):
    app.config["generator"] = generator
    with app.test_client() as http:
        return http.post(path, json=body)


def test_input_generation(generation_app, mock_generator):
    response = _post(
        generation_app, "/input", mock_generator, prompt="Hello, world!", max_new_tokens=50
    )
    assert response.status_code == 200
    assert "Generated response" in response.get_json()["response"]


def test_messages_generation(generation_app, mock_generator):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    response = _post(generation_app, "/messages", mock_generator, messages=messages)
    assert response.status_code == 200
    assert response.get_json()["response"]


@pytest.mark.parametrize(
    "path,body,complaint",
    [
        ("/input", {"max_new_tokens": 50}, "prompt"),
        (
            "/input",
            {"prompt": "test", "messages": [{"role": "user", "content": "x"}]},
            "/messages endpoint",
        ),
        ("/messages", {"max_new_tokens": 50}, "messages"),
    ],
)
def test_malformed_requests_are_rejected(
    generation_app, mock_generator, path, body, complaint
):
    response = _post(generation_app, path, mock_generator, **body)
    assert response.status_code == 400
    assert complaint in response.get_json()["error"].lower()


# The reply also streams to the browser as it is written. `POST /messages/`
# stays one request with one authoritative JSON reply; the deltas ride the
# `/realtime` socket, keyed by an id the client mints.


class _StreamingGenerator:
    """Publishes a reply in pieces, then returns it whole - the shape a real
    `Generator` with a `ReplyStreamer` attached produces."""

    def __init__(self):
        self.pending = None

    def request_generation(
        self, prompt, kwargs, deadline=None, on_text=None, on_reset=None, **_
    ):
        for chunk in ("It ", "is ", "noon."):
            if on_text:
                on_text(chunk)
        self.pending = "It is noon."
        return "rid"

    def get_result(self, request_id):
        result, self.pending = self.pending, None
        return result


def test_the_route_streams_and_still_returns_the_whole_reply(emitted, generation_app):
    """The deltas go out AND the response body is the same complete reply."""
    response = _post(
        generation_app,
        "/messages/",
        _StreamingGenerator(),
        messages=[{"role": "user", "content": "what time is it?"}],
        stream_id="tab-1",
    )

    assert response.status_code == 200
    assert "It is noon." in response.get_json()["response"]
    deltas = [f[1]["text"] for f in emitted if f[0] == "gen_delta"]
    assert "".join(deltas) == "It is noon."
    assert all(f[1]["id"] == "tab-1" for f in emitted)


def test_the_route_without_a_stream_id_emits_nothing(emitted, generation_app):
    """A client that never learned about streaming - or whose socket is down -
    gets exactly the behavior it had before."""

    class _Generator:
        def request_generation(
            self, prompt, kwargs, deadline=None, on_text=None, on_reset=None, **kw
        ):
            assert on_text is None and on_reset is None
            # The tool tally is installed unconditionally: its counts ride the
            # RESPONSE, not the socket.
            assert kw.get("on_tool") is not None
            return "rid"

        def get_result(self, request_id):
            return "[BOS]assistant\nquiet reply[SEP]"

    response = _post(
        generation_app,
        "/messages/",
        _Generator(),
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.status_code == 200
    assert response.get_json()["response"] == "quiet reply"
    assert emitted == []


class _ToolUsingGenerator:
    """Runs two tools (one of them twice) and then answers, which is the shape
    `Generator._process_single_request` produces around a spliced result."""

    def __init__(self):
        self.pending = None

    def request_generation(
        self, prompt, kwargs, deadline=None, on_text=None, on_reset=None, on_tool=None
    ):
        for name in ("read_file", "search", "read_file"):
            if on_tool:
                on_tool(name)
            if on_reset:
                # Every tool call moves the turn anchor, so a reset follows it.
                on_reset()
        if on_text:
            on_text("done")
        self.pending = "done"
        return "rid"

    def get_result(self, request_id):
        result, self.pending = self.pending, None
        return result


TOOL_TALLY = [{"name": "read_file", "count": 2}, {"name": "search", "count": 1}]


def _post_tools(app, **body):
    return _post(
        app,
        "/messages/",
        _ToolUsingGenerator(),
        messages=[{"role": "user", "content": "hi"}],
        **body,
    )


def test_tool_use_is_counted_in_the_response_without_a_socket(emitted, generation_app):
    """The reply extractor strips the whole call/result exchange, so this tally
    is the only record a tool ran. Counted per name in first-use order (the
    order the chips render in), and server-side: a client whose socket is down
    loses the live chips, not the record."""
    response = _post_tools(generation_app)

    assert response.status_code == 200
    assert response.get_json()["tools"] == TOOL_TALLY
    assert emitted == []  # no stream id, so nothing went out on the wire


def test_a_tool_frame_names_the_tool_ahead_of_its_reset(emitted, generation_app):
    """One frame per execution, carrying the client's own id, and each lands
    before the reset it causes - a chip arriving after the reset would look
    like something the reset should have taken back."""
    response = _post_tools(generation_app, stream_id="tab-9")

    tools = [f[1] for f in emitted if f[0] == "gen_tool"]
    assert [f["name"] for f in tools] == ["read_file", "search", "read_file"]
    assert all(f["id"] == "tab-9" for f in tools)
    assert all(f[2] == NAMESPACE for f in emitted)
    kinds = [f[0] for f in emitted if f[0] in ("gen_tool", "gen_reset")]
    assert kinds == ["gen_tool", "gen_reset"] * 3
    # ...and the response still carries the authoritative tally to settle on.
    assert response.get_json()["tools"] == TOOL_TALLY


# ------------------------------------------------------------------------------
# core: ping, spec, home page, config download
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["get", "post"])
def test_ping_endpoint(api_url, method):
    response = getattr(requests, method)(f"{api_url}/api/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Praxis API server is running" in data["message"]


def test_spec_endpoint(api_url):
    response = requests.get(f"{api_url}/api/spec")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["truncated_hash"] == "test12345"
    assert data["full_hash"] == "test1234567890abcdef"
    assert "args" in data
    assert data["param_stats"]["total"] == 1000000
    assert data["seed"] == 42


def test_home_page(api_url):
    response = requests.get(f"{api_url}/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("Content-Type", "")
    assert "Content-Security-Policy" in response.headers
    assert "<!DOCTYPE html>" in response.text
    assert "<title>Praxis</title>" in response.text


def test_route_serves_the_annotated_file(tmp_path, monkeypatch):
    """The config behind the web app's Download button, annotated with defaults."""
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "base.yml").write_text("block_size: 128\n")
    (experiments / "child.yml").write_text("# a comment\nextends: base\ndepth: 3\n")
    monkeypatch.chdir(tmp_path)

    app = flask.Flask(__name__)
    app.config.update(config_file="experiments/child.yml")
    register_routes(app)
    response = app.test_client().get("/api/config")

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert yaml.safe_load(text) == {"block_size": 128, "depth": 3}
    assert "block_size: 128  # default: 512" in text


# ------------------------------------------------------------------------------
# agents, static, git
# ------------------------------------------------------------------------------


@patch("subprocess.run")
def test_agents_endpoint_lists_this_instance(mock_run, api_url):
    mock_run.return_value = Mock(
        returncode=0,
        stdout="origin\thttps://github.com/test/repo.git\t(fetch)\n",
        stderr="",
    )
    response = requests.get(f"{api_url}/api/agents")
    assert response.status_code == 200
    names = [agent["name"] for agent in response.json()["agents"]]
    assert any(name.startswith("self-") for name in names)


def test_favicon(api_url):
    # 204 when no favicon was built.
    assert requests.get(f"{api_url}/favicon.ico").status_code in (200, 204)


def test_missing_static_file_is_a_404(api_url):
    assert requests.get(f"{api_url}/static/nonexistent.js").status_code == 404


@patch("subprocess.run")
def test_git_info_refs(mock_run, api_url):
    mock_run.return_value = Mock(returncode=0, stdout=b"abc123\tHEAD\n", stderr=b"")
    response = requests.get(f"{api_url}/praxis/info/refs?service=git-upload-pack")
    assert response.status_code == 200
    assert "application/x-git-upload-pack-advertisement" in response.headers.get(
        "Content-Type", ""
    )


@pytest.mark.parametrize(
    "service,status", [("invalid", 400), ("git-receive-pack", 403)]
)
def test_git_refuses_unknown_services_and_writes(api_url, service, status):
    response = requests.get(f"{api_url}/praxis/info/refs?service={service}")
    assert response.status_code == status


# ------------------------------------------------------------------------------
# dynamics: activation probes and the compute treemap
# ------------------------------------------------------------------------------
# Probes read the live model, whose module tree is shared with the training
# thread: ``torch.func.functional_call`` swaps entries in a module's
# ``_parameters`` dict (not thread-safe, and the activations it was used on live
# inside the memory's vmap), and a graph through live parameters races the
# optimizer's version counter. So a probe reads and does nothing else: no
# parameter swap, no graph, no grads. A torn read costs one wrong sample.


class Activation(nn.Module):
    """Feature-dim parameters, like Serpent/Servant."""

    def __init__(self, dim=8):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.sin(self.a * x) + self.b


def _probe(module, points=33):
    from praxis.web.routes.dynamics import _sample_activation

    return _sample_activation(
        module, -3.0, 3.0, points, torch.device("cpu"), torch.float32
    )


def test_probe_is_read_only(monkeypatch):
    """functional_call is never reached, and every parameter comes back the
    same object, untouched, with no grad and still trainable."""
    import torch.func

    def boom(*a, **k):
        raise AssertionError("probe reparametrized a live module")

    monkeypatch.setattr(torch.func, "functional_call", boom)
    module = Activation()
    before = {n: p.detach().clone() for n, p in module.named_parameters()}

    assert _probe(module) is not None
    for name, p in module.named_parameters():
        assert isinstance(p, nn.Parameter), f"{name} was swapped for a plain tensor"
        assert torch.equal(p, before[name]), f"{name} was mutated by the probe"
        assert p.grad is None, f"{name} got a gradient"
        assert p.requires_grad


def test_probe_derivative_is_right():
    """Read-only means a numeric derivative; it still has to be correct."""
    module = Activation()
    sample = _probe(module, points=201)
    x = torch.tensor(sample["x"])
    got = torch.tensor(sample["backward"])
    want = module.a[0] * torch.cos(module.a[0] * x)  # d/dx sin(a x)
    interior = slice(2, -2)
    torch.testing.assert_close(
        got[interior], want[interior].to(got.dtype), rtol=0.02, atol=0.02
    )


def test_probe_skips_a_module_that_is_mid_transform():
    """Sampling a module the Titans memory is driving under vmap raises "tensor
    escaped from inside a function being vmapped" - true, transient and not
    worth a traceback every poll. The probe checks first and skips, and the skip
    is not logged as a failure: that would suppress the genuine warning for the
    class forever after."""
    import torch.func as F
    from torch.func import vmap

    from praxis.web.routes import dynamics
    from praxis.web.routes.dynamics import _inside_a_transform

    module = Activation()
    assert not _inside_a_transform(module)
    failures_before = set(dynamics._SAMPLE_FAILURES)

    # Observed from INSIDE the reparametrized scope, where the real collision
    # happens: functional_call swaps the module's _parameters process-wide for
    # the duration of the call. A pre-hook is the faithful vantage point.
    seen = {}
    handle = module.register_forward_pre_hook(
        lambda m, inp: seen.update(
            mid=_inside_a_transform(m), skipped=_probe(m) is None
        )
    )
    try:
        params = {
            n: p.unsqueeze(0).expand(3, *p.shape) for n, p in module.named_parameters()
        }
        vmap(lambda w, row: F.functional_call(module, w, (row,)).sum())(
            params, torch.randn(3, 8)
        )
    finally:
        handle.remove()

    assert seen["mid"] is True, "mid-transform state was not detected"
    assert seen["skipped"] is True, "probe sampled a module that was mid-transform"
    assert set(dynamics._SAMPLE_FAILURES) == failures_before
    # ...and once the transform is done, sampling works again by itself.
    assert not _inside_a_transform(module)
    assert _probe(module) is not None


def test_activation_curves_route_preserves_training_mode():
    """The route samples the live training model. Flipping it to eval races the
    training forward - a CALM stage-2 forward seen in eval mode detaches its
    whole loss and crashes backward()."""
    from praxis.web.routes.dynamics import _compute_activation_curves

    model = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    model.train()
    seen_training = []
    model[1].register_forward_hook(lambda m, i, o: seen_training.append(model.training))

    curves, _ = _compute_activation_curves(model, -6.0, 6.0, 64)

    assert curves, "expected at least one activation curve (GELU)"
    assert model.training is True, "route left the model out of train mode"
    assert seen_training and all(seen_training), "model left train mode mid-sample"


def _head_snapshots(model):
    """GET /api/head_snapshots through the live fallback (no snapshot store)."""
    app = Flask(__name__)
    app.register_blueprint(dynamics_bp)
    app.config["snapshot_store"] = None
    app.config["generator"] = type("G", (), {"model": model})()
    with app.test_client() as c:
        return json.loads(c.get("/api/head_snapshots").data)


def test_route_serves_the_stashed_profile(bare_model, compute_profile):
    bare_model._compute_profile = compute_profile
    body = _head_snapshots(bare_model)

    assert body["status"] == "ok"
    profile = body["snapshots"]["compute_profile"]
    assert profile["samples"] == 3
    assert profile["groups"][0]["name"] == "ArcAttention"
    assert sum(g["share"] for g in profile["groups"]) == pytest.approx(1.0)


@pytest.mark.parametrize("stash", [None, "not a dict"], ids=["absent", "non-dict"])
def test_route_is_quiet_without_a_profile(bare_model, stash):
    """A run that never profiled (e.g. torch.compile) grows no compute card."""
    if stash is not None:
        bare_model._compute_profile = stash
    assert "compute_profile" not in _head_snapshots(bare_model).get("snapshots", {})


def test_recipe_and_route_agree_on_the_compute_key(bare_model, compute_profile):
    """Two implementations of the same payload; keep them from drifting."""
    from praxis.web.snapshots import _recipe_head_snapshots

    bare_model._compute_profile = compute_profile
    recipe_out = _recipe_head_snapshots(bare_model)["snapshots"]
    route_out = _head_snapshots(bare_model)["snapshots"]
    assert recipe_out["compute_profile"] == route_out["compute_profile"]


# ------------------------------------------------------------------------------
# metrics: the per-run payload behind the Research tab
# ------------------------------------------------------------------------------


def _payload(db):
    """The per-run payload /api/metrics builds, read from one metrics.db."""
    from praxis.web.routes.metrics import _read_metrics_file, _transform_metrics

    return _transform_metrics(_read_metrics_file(db, 0))


def _has(payload, key):
    values = payload.get(key)
    return bool(values) and any(v is not None for v in values)


@pytest.fixture
def old_run(tmp_path):
    """A metrics.db written before most of today's registry columns existed,
    carrying one column the registry has since dropped and no extra_metrics."""
    db = tmp_path / "metrics.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE metrics (step INTEGER PRIMARY KEY, ts REAL NOT NULL, "
        "loss REAL, val_loss REAL, num_tokens REAL, retired_metric REAL)"
    )
    for step in range(10):
        conn.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?)",
            (step, 1000.0 + step, 3.0 - 0.1 * step, 2.9 if step == 9 else None,
             1000.0 * step, 0.5),
        )
    conn.commit()
    conn.close()
    return db


def test_an_old_schema_run_keeps_every_series_it_recorded(old_run):
    """Naming every registry column in one SELECT raised ``no such column`` on
    any run older than the newest metric, and the caller dropped the run - its
    loss curve with it. What the database holds is what the payload serves.

    Not "every run draws every card": a column a run never had reads as NULL."""
    payload = _payload(old_run)

    assert payload and "steps" in payload
    conn = sqlite3.connect(old_run)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(metrics)")}
    recorded = {
        key
        for key, entry in TRAINING_METRIC_REGISTRY.items()
        if entry.get("chart")
        and key in columns
        and conn.execute(f"SELECT COUNT({key}) FROM metrics").fetchone()[0]
    }
    conn.close()
    assert {"loss", "val_loss"} <= recorded
    for key in recorded:
        assert _has(payload, key), f"{key!r} was recorded but dropped from the payload"


def test_validation_points_carry_every_x_coordinate(tmp_path):
    """A val point must know where it sits on EVERY axis.

    Validation drains at ``trainer.global_step`` and ``MetricsLogger.log``
    upserts on the step key, so the val write MERGES into the row the training
    step already wrote, carrying that step's num_tokens and ts. If validation
    ever lands on a row of its own, its coordinates go null and a val curve on
    the token axis is silently misplaced."""
    from praxis.logging.metrics_logger import MetricsLogger

    logger = MetricsLogger(run_dir=tmp_path, csv_mirror=False)
    for step in range(10):
        logger.log(step=step, loss=3.0 - 0.1 * step, num_tokens=1000 * step)
        if step % 4 == 3:
            logger.log(step=step, val_loss=2.9)
    logger.close()

    payload = _payload(tmp_path / "metrics.db")
    val_rows = [i for i, v in enumerate(payload["val_loss"]) if v is not None]
    assert val_rows == [3, 7]
    for axis in X_AXIS_REGISTRY:
        column = payload.get(axis["source"])
        assert column is not None, f"no {axis['source']} column"
        assert len(column) == len(payload["val_loss"])
        missing = [i for i in val_rows if column[i] is None]
        assert not missing, f"val points {missing} have no {axis['source']}"


def test_elapsed_discounts_a_suspension():
    """A stopped-and-resumed run must not be billed for the hours it sat idle."""
    from praxis.web.routes.metrics import _TS_RAW, _elapsed_seconds

    rows = lambda stamps: [{_TS_RAW: t} for t in stamps]

    # Uninterrupted: elapsed is exactly the span.
    assert _elapsed_seconds(rows([0, 10, 20, 30]))[-1] == 30

    # An 8-hour pause between 10-second rows contributes one typical interval.
    paused = _elapsed_seconds(rows([0, 10, 20, 28800, 28810]))
    assert paused[-1] < 100, f"pause was billed as training time: {paused[-1]}s"

    # Never runs backwards, whatever the clock did.
    wobbly = _elapsed_seconds(rows([0, 10, 5, 15]))
    assert all(b >= a for a, b in zip(wobbly, wobbly[1:]))

    # Degenerate inputs must not raise - they reach here on brand-new runs.
    assert _elapsed_seconds([]) == []
    assert _elapsed_seconds(rows([1234.5])) == [0.0]
    assert _elapsed_seconds([{}, {}]) == [0.0, 0.0]


def test_transform_strips_the_raw_timestamp():
    """``_ts_epoch`` is scratch for the wall-clock maths, never payload."""
    from praxis.web.routes.metrics import _TS_RAW, _transform_metrics

    out = _transform_metrics([{"step": 0, "ts": "x", _TS_RAW: 100.0, "loss": 1.0}])
    assert _TS_RAW not in out and "ts" not in out
    assert out["elapsed_s"] == [0.0]


# ------------------------------------------------------------------------------
# print: model-led question -> user answer -> reward
# ------------------------------------------------------------------------------


class _QuestionGenerator:
    """Returns a model-led 'question\\nanswer' in the chat-template envelope."""

    def __init__(self, reply):
        self._reply = reply

    def request_generation(self, prompt, kwargs, deadline=None, **_):
        return "rid"

    def get_result(self, rid):
        return self._reply


@pytest.fixture
def print_client(fake_tokenizer):
    """Client factory; the pending slots are process-global, so clear them."""
    with print_route._lock:
        print_route._pending.clear()
        print_route._loop_pending.clear()

    def make(reply="[BOS]assistant\nWhat is the capital of France?\nParis[SEP]"):
        app = flask.Flask(__name__)
        app.config["tokenizer"] = fake_tokenizer
        if reply is not None:
            app.config["generator"] = _QuestionGenerator(reply)
        register_routes(app)
        return app.test_client()

    return make


def test_button_is_conditional_until_asked(print_client):
    c = print_client()
    assert c.get("/api/print/pending").get_json() == {"available": False}
    ask = c.post("/api/print/ask", json={}).get_json()
    assert ask["available"] is True
    assert ask["question"] == "What is the capital of France?"
    assert c.get("/api/print/pending").get_json()["available"] is True


def test_ask_is_idempotent_while_pending(print_client):
    c = print_client()
    a1 = c.post("/api/print/ask", json={}).get_json()
    a2 = c.post("/api/print/ask", json={}).get_json()
    assert a1["id"] == a2["id"]


def test_respond_scores_and_clears(print_client):
    c = print_client()
    ask = c.post("/api/print/ask", json={}).get_json()
    r = c.post(
        "/api/print/respond", json={"id": ask["id"], "response": "Is it Paris?"}
    ).get_json()
    assert r["status"] == "ok"
    assert r["activation"] == 1.0  # 'Paris?' matches predicted 'Paris'
    assert r["recall"] == 1.0
    assert r["predicted_answer"] == "Paris"
    assert c.get("/api/print/pending").get_json() == {"available": False}


def test_stale_id_is_rejected(print_client):
    c = print_client()
    c.post("/api/print/ask", json={})
    resp = c.post("/api/print/respond", json={"id": "nope", "response": "x"})
    assert resp.status_code == 409


def test_unavailable_when_generator_missing(print_client):
    out = print_client(reply=None).post("/api/print/ask", json={}).get_json()
    assert out["available"] is False


def test_loop_approve_records_joke_reward(print_client):
    from praxis.policies.engagement_channel import LIVE_JOKES

    LIVE_JOKES.drain()
    c = print_client()
    approve = c.post("/api/loop/approve", json={"score": 1.0}).get_json()
    assert approve["status"] == "ok"
    assert approve["activation"] == 1.0
    # A rejection still sustains energy (engagement alone counts), but its
    # valence lives in the signed reward, not the activation.
    reject = c.post("/api/loop/approve", json={"approve": False}).get_json()
    assert reject["activation"] == pytest.approx(0.8)
    assert reject["score"] == -1.0 and reject["reward"] == -1.0
    assert c.get("/api/loop/energy").get_json()["count"] >= 2
    # Both events buffered for the joke drain callback.
    assert len(LIVE_JOKES.drain()) >= 2


def test_loop_generate_then_calibrated_approve(print_client):
    from praxis.policies.engagement_channel import LIVE_JOKES

    LIVE_JOKES.drain()
    c = print_client(reply="[BOS]assistant\nA pun!\n+0.6[SEP]")
    gen = c.post("/api/loop/generate", json={"task": "joke"}).get_json()
    assert gen["available"] is True
    assert gen["mode"] == "calibration"
    assert gen["text"] == "A pun!"  # prediction parsed off the display text
    assert gen["predicted"] == pytest.approx(0.6)

    # Confirming the model's guess = zero correction = full activation.
    ok = c.post("/api/loop/approve", json={"id": gen["id"], "score": 0.6}).get_json()
    assert ok["correction"] == pytest.approx(0.0)
    assert ok["activation"] == 1.0

    # A large correction shrinks the activation; valence keeps the user's sign.
    bad = c.post("/api/loop/approve", json={"id": gen["id"], "score": -0.4}).get_json()
    assert bad["correction"] == pytest.approx(1.0)
    assert bad["reward"] == pytest.approx(-0.4)
    assert bad["activation"] < ok["activation"]
    assert len(LIVE_JOKES.drain()) == 2


# ------------------------------------------------------------------------------
# cards: preview SVG and PDF downloads
# ------------------------------------------------------------------------------


@pytest.fixture
def cards_client():
    app = Flask(__name__)
    app.register_blueprint(cards_bp)
    app.config["author"] = ["Ryan J. Brooks"]
    app.config["donations"] = "https://example.com/donate"
    app.config["truncated_hash"] = "abc123"
    return app.test_client()


@pytest.fixture
def rendered_sides(monkeypatch):
    """Stub the PDF renderers (the routes import them lazily): these tests are
    about packaging, and tests/pillars renders a real sheet."""
    import praxis.pillars.projections as projections

    sides = []

    def stub(side, *args, **kwargs):
        sides.append(side)
        return b"%PDF-stub"

    monkeypatch.setattr(projections, "render_single_pdf", stub)
    monkeypatch.setattr(projections, "render_sheet_pdf", stub)
    return sides


@pytest.mark.parametrize("query", ["?seed=5&side=front&theme=dark&hue=161", ""])
def test_preview_route(cards_client, query):
    resp = cards_client.get(f"/api/card/preview.svg{query}")
    assert resp.status_code == 200
    assert resp.mimetype == "image/svg+xml"
    seed = int(resp.headers["X-Card-Seed"])
    assert seed == 5 if query else seed >= 0


@pytest.mark.parametrize(
    "path,names",
    [
        ("/api/card/cards.zip", {"praxis-card-front.pdf", "praxis-card-back.pdf"}),
        (
            "/api/card/sheets.zip",
            {"praxis-cards-10up-front.pdf", "praxis-cards-10up-back.pdf"},
        ),
    ],
)
def test_zip_routes_carry_both_sides(cards_client, rendered_sides, path, names):
    resp = cards_client.get(f"{path}?seed=5")
    assert resp.status_code == 200
    assert resp.mimetype == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(resp.data))
    assert set(zf.namelist()) == names
    assert all(zf.read(n) == b"%PDF-stub" for n in names)
    assert sorted(rendered_sides) == ["back", "front"]


@pytest.mark.parametrize(
    "path,name",
    [
        ("/api/card/card.pdf", "praxis-card-back.pdf"),
        ("/api/card/sheet.pdf", "praxis-cards-10up-back.pdf"),
    ],
)
def test_pdf_routes_render_the_requested_side(cards_client, rendered_sides, path, name):
    resp = cards_client.get(f"{path}?seed=5&side=back")
    assert resp.status_code == 200
    assert resp.mimetype == "application/pdf"
    assert resp.data == b"%PDF-stub"
    assert name in resp.headers["Content-Disposition"]
    assert rendered_sides == ["back"]
