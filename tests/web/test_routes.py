import json
import sys
import time
from typing import Generator
from unittest.mock import Mock, patch

import flask
import pytest
import requests
import torch
import torch.nn as nn
import yaml
from flask import Flask

from praxis import registry
from praxis.metrics.training_metrics import TRAINING_METRIC_REGISTRY, X_AXIS_REGISTRY
from praxis.web import APIServer, app
from praxis.web.routes import print as print_route
from praxis.web.routes import register_routes
from praxis.web.routes.cards import cards_bp
from praxis.web.routes.dynamics import dynamics_bp
from praxis.web.websocket.realtime import NAMESPACE

# ------------------------------------------------------------------------------
# generation_stream_route
# ------------------------------------------------------------------------------
# The side channel that carries a reply to the browser as it is written.
#
# ``POST /messages/`` is unchanged - one request, one final JSON reply, still
# authoritative. The deltas ride the ``/realtime`` socket the client already has open,
# keyed by an id the client mints. The properties worth pinning are the ones whose
# failure is silent: opting out has to change nothing, and a broken socket has to cost
# the preview and nothing else.


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


def test_the_route_streams_and_still_returns_the_whole_reply(emitted, client):
    """End to end through the Flask route: the deltas go out AND the response
    body is the same complete reply it always was."""

    class _StreamingGenerator:
        """Publishes a reply in pieces, then returns it whole - the shape a
        real `Generator` with a `ReplyStreamer` attached produces."""

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
            self, prompt, kwargs, deadline=None, on_text=None, on_reset=None, **kw
        ):
            # `on_tool` is the exception: the route installs a tally for it
            # unconditionally, because the counts ride the RESPONSE and not
            # the socket. Nothing is emitted without a stream id, which is
            # what this test goes on to assert.
            assert on_text is None and on_reset is None
            assert kw.get("on_tool") is not None
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


# ---------------------------------------------------------------------------
# tool use: the one thing the reply itself can never show
# ---------------------------------------------------------------------------


class _ToolTokenizer:
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    sep_token = "[SEP]"

    def apply_chat_template(self, messages, **kwargs):
        return "[BOS]user\nhi[SEP]\n[BOS]assistant\n"

    def convert_tokens_to_ids(self, token):
        return None


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


def _post_tools(client, **body):
    client.config["generator"] = _ToolUsingGenerator()
    client.config["tokenizer"] = _ToolTokenizer()
    client.config.pop("api_server", None)
    with client.test_client() as http:
        return http.post(
            "/messages/",
            json={"messages": [{"role": "user", "content": "hi"}], **body},
        )


def test_tool_use_is_counted_in_the_response(emitted, client):
    """The reply extractor strips the whole call/result exchange, so without
    this the response is the same whether a tool ran or not. Counted per name,
    in first-use order - which is the order the chips render in."""
    response = _post_tools(client)

    assert response.status_code == 200
    assert response.get_json()["tools"] == [
        {"name": "read_file", "count": 2},
        {"name": "search", "count": 1},
    ]


def test_tool_counts_do_not_need_a_socket(emitted, client):
    """The tally is server-side and unconditional. A client whose socket is
    down loses the live chips, not the record - the same bargain the reply
    text already makes."""
    response = _post_tools(client)

    assert emitted == []  # no stream id, so nothing went out on the wire
    assert [t["name"] for t in response.get_json()["tools"]] == ["read_file", "search"]


def test_a_tool_frame_names_the_tool(emitted, client):
    """Live half. One frame per execution, carrying the client's own id."""
    response = _post_tools(client, stream_id="tab-9")

    tools = [f[1] for f in emitted if f[0] == "gen_tool"]
    assert [f["name"] for f in tools] == ["read_file", "search", "read_file"]
    assert all(f["id"] == "tab-9" for f in tools)
    assert all(f[2] == NAMESPACE for f in emitted)
    # ...and the response still carries the authoritative tally to settle on.
    assert response.get_json()["tools"] == [
        {"name": "read_file", "count": 2},
        {"name": "search", "count": 1},
    ]


def test_a_tool_frame_precedes_the_reset_it_causes(emitted, client):
    """Ordering the client depends on: the reset retracts the model's pre-call
    chatter, and a chip that arrived after it would look like something the
    reset should have taken back. It arrives first, and stands."""
    _post_tools(client, stream_id="tab-9")

    kinds = [f[0] for f in emitted if f[0] in ("gen_tool", "gen_reset")]
    assert kinds == ["gen_tool", "gen_reset"] * 3


# ------------------------------------------------------------------------------
# print_route
# ------------------------------------------------------------------------------
# Tests for the Print mechanism: model-led question -> user answer -> reward.


@pytest.fixture(autouse=True)
def _reset_pending():
    """The pending slots are process-global; clear them between tests."""
    with print_route._lock:
        print_route._pending.clear()
        print_route._loop_pending.clear()
    yield


class _FakeGen:
    """Returns a model-led 'question\\nanswer' in the chat-template envelope."""

    def __init__(self, reply):
        self._reply = reply

    def request_generation(self, prompt, kwargs, deadline=None, **_):
        return "rid"

    def get_result(self, rid):
        return self._reply


class _FakeTok:
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    sep_token = "[SEP]"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "[BOS]system\n..."


def _client(reply="[BOS]assistant\nWhat is the capital of France?\nParis[SEP]"):
    app = flask.Flask(__name__)
    app.config.update(generator=_FakeGen(reply), tokenizer=_FakeTok())
    register_routes(app)
    return app.test_client()


def test_button_is_conditional_until_asked():
    c = _client()
    assert c.get("/api/print/pending").get_json() == {"available": False}
    ask = c.post("/api/print/ask", json={}).get_json()
    assert ask["available"] is True
    assert ask["question"] == "What is the capital of France?"
    assert c.get("/api/print/pending").get_json()["available"] is True


def test_ask_is_idempotent_while_pending():
    c = _client()
    a1 = c.post("/api/print/ask", json={}).get_json()
    a2 = c.post("/api/print/ask", json={}).get_json()
    assert a1["id"] == a2["id"]


def test_respond_scores_and_clears():
    c = _client()
    ask = c.post("/api/print/ask", json={}).get_json()
    r = c.post(
        "/api/print/respond", json={"id": ask["id"], "response": "Is it Paris?"}
    ).get_json()
    assert r["status"] == "ok"
    assert r["activation"] == 1.0  # 'Paris?' matches predicted 'Paris'
    assert r["recall"] == 1.0
    assert r["predicted_answer"] == "Paris"
    # Slot cleared after answering.
    assert c.get("/api/print/pending").get_json() == {"available": False}


def test_stale_id_is_rejected():
    c = _client()
    c.post("/api/print/ask", json={})
    resp = c.post("/api/print/respond", json={"id": "nope", "response": "x"})
    assert resp.status_code == 409


def test_unavailable_when_generator_missing():
    app = flask.Flask(__name__)
    app.config["tokenizer"] = _FakeTok()  # no generator
    register_routes(app)
    out = app.test_client().post("/api/print/ask", json={}).get_json()
    assert out["available"] is False


def test_loop_approve_records_joke_reward():
    from praxis.policies.engagement_channel import LIVE_JOKES

    LIVE_JOKES.drain()
    c = _client()
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


def test_loop_generate_then_calibrated_approve():
    from praxis.policies.engagement_channel import LIVE_JOKES

    LIVE_JOKES.drain()
    c = _client(reply="[BOS]assistant\nA pun!\n+0.6[SEP]")
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
# web_probe_isolation
# ------------------------------------------------------------------------------
# Web probes must not touch training state.
#
# Probes no longer run on the API thread at all - ``SnapshotPumpCallback`` runs them
# from the training loop, because "read-only" was never enough: a pure ``tensor[...,
# :k]`` still locks that tensor's ``AutogradMeta``, and holding the GIL while waiting
# for a lock the training thread holds deadlocks the process outright
# (praxis/web/snapshots.py has the ABBA). These rules still stand on top of that, both
# because they are what a probe means and because an inference-only server has no
# training loop to pump them.
#
# The dashboard samples the live model, and the module tree is shared state:
#
# * ``torch.func.functional_call`` swaps entries in a module's ``_parameters`` dict. It
# is not thread-safe, and the activations it was briefly used on here live INSIDE
# ``memory_model`` (``NeuralMemory(model=..., activation=serpent)``). The training
# thread read those swapped, detached tensors and died with "One of the differentiated
# Tensors does not require grad" - and, when the swap landed during the memory's own
# vmap, "tensor escaped from inside a function being vmapped". Hundreds of steps in,
# only under the full launcher, which is why no single-process test run ever caught it.
# * Building an autograd graph through live parameters lets the optimizer bump a version
# counter between the probe's forward and its backward.
#
# So a probe reads, and does nothing else: no parameter swap, no graph, no grads. A torn
# read costs one wrong sample on one poll; the next poll fixes it.


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


def test_probe_does_not_mutate_parameters():
    module = Activation()
    before = {n: p.detach().clone() for n, p in module.named_parameters()}
    assert _probe(module) is not None
    for name, p in module.named_parameters():
        assert torch.equal(p, before[name]), f"{name} was mutated by the probe"
        assert isinstance(p, nn.Parameter), f"{name} was swapped out for a plain tensor"


def test_probe_creates_no_gradients():
    """No graph through live parameters - that is the version-counter race."""
    module = Activation()
    assert _probe(module) is not None
    assert all(p.grad is None for p in module.parameters())
    assert all(p.requires_grad for p in module.parameters())


def test_probe_leaves_parameters_usable_by_functorch():
    """The exact operations the memory performs on its own parameters after a
    probe has run: a batched tensor left installed breaks both."""
    module = Activation()
    assert _probe(module) is not None
    for p in module.parameters():
        p.unsqueeze(0).expand(4, *p.shape)  # _init_weights
        p.detach().cpu()  # Lightning teardown


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


def test_probe_does_not_reparametrize(monkeypatch):
    """A hard guard: functional_call must never be reached from this path."""
    import torch.func

    def boom(*a, **k):
        raise AssertionError("probe reparametrized a live module")

    monkeypatch.setattr(torch.func, "functional_call", boom)
    assert _probe(Activation()) is not None


def test_probe_skips_a_module_that_is_mid_transform():
    """Several activations live inside memory_model, which the Titans memory
    drives under vmap. Sampling one then raises "tensor escaped from inside a
    function being vmapped" - true, transient, and not worth a traceback every
    poll. The probe checks first and skips on purpose."""
    import torch.func as F
    from torch.func import vmap

    from praxis.web.routes.dynamics import _inside_a_transform

    module = Activation()
    assert not _inside_a_transform(module)

    # Observed from INSIDE the reparametrized scope, which is where the real
    # collision happens: functional_call swaps the module's _parameters
    # process-wide for the duration of the call, so the API thread sees batched
    # tensors exactly here. A pre-hook is the faithful vantage point.
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

    # ...and once the transform is done, sampling works again by itself.
    assert not _inside_a_transform(module)
    assert _probe(module) is not None


def test_mid_transform_skip_is_not_logged_as_a_failure():
    """The skip must not poison _SAMPLE_FAILURES, or a transient collision would
    suppress the genuine warning for that class forever after."""
    from praxis.web.routes import dynamics

    before = set(dynamics._SAMPLE_FAILURES)
    module = Activation()
    assert _probe(module) is not None
    assert set(dynamics._SAMPLE_FAILURES) == before


# ------------------------------------------------------------------------------
# compute_profiler_route
# ------------------------------------------------------------------------------
# /api/head_snapshots must serve the compute treemap stashed on the model.


class FakeModel:
    """Stands in for the live model the generator holds."""

    head = None
    criterion = None
    encoder = None


@pytest.fixture
def dynamics_client():
    app = Flask(__name__)
    app.register_blueprint(dynamics_bp)
    app.config["snapshot_store"] = None  # force the live fallback path
    return app


def _payload():
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


def test_route_serves_the_stashed_profile(dynamics_client):
    model = FakeModel()
    model._compute_profile = _payload()
    dynamics_client.config["generator"] = type("G", (), {"model": model})()

    with dynamics_client.test_client() as c:
        body = json.loads(c.get("/api/head_snapshots").data)

    assert body["status"] == "ok"
    profile = body["snapshots"]["compute_profile"]
    assert profile["samples"] == 3
    assert profile["groups"][0]["name"] == "ArcAttention"
    assert sum(g["share"] for g in profile["groups"]) == pytest.approx(1.0)


def test_route_is_quiet_without_the_profiler(dynamics_client):
    """A run that never profiled (e.g. torch.compile) grows no compute card."""
    dynamics_client.config["generator"] = type("G", (), {"model": FakeModel()})()

    with dynamics_client.test_client() as c:
        body = json.loads(c.get("/api/head_snapshots").data)

    assert "compute_profile" not in body.get("snapshots", {})


def test_route_ignores_a_non_dict_stash(dynamics_client):
    model = FakeModel()
    model._compute_profile = "not a dict"
    dynamics_client.config["generator"] = type("G", (), {"model": model})()

    with dynamics_client.test_client() as c:
        body = json.loads(c.get("/api/head_snapshots").data)

    assert "compute_profile" not in body.get("snapshots", {})


def test_recipe_and_route_agree_on_the_compute_key():
    """Two implementations of the same payload; keep them from drifting again."""
    from praxis.web.snapshots import _recipe_head_snapshots

    model = FakeModel()
    model._compute_profile = _payload()
    recipe_out = _recipe_head_snapshots(model)["snapshots"]

    app = Flask(__name__)
    app.register_blueprint(dynamics_bp)
    app.config["snapshot_store"] = None
    app.config["generator"] = type("G", (), {"model": model})()
    with app.test_client() as c:
        route_out = json.loads(c.get("/api/head_snapshots").data)["snapshots"]

    assert recipe_out["compute_profile"] == route_out["compute_profile"]


# ------------------------------------------------------------------------------
# metric_cards
# ------------------------------------------------------------------------------
# Dashboard card invariants for the Research-tab metric registries.
#
# The Research tab builds its deck with ``buildScalarConfigsFromRegistry`` and
# ``buildCompositeConfigsFromRegistry`` (praxis/web/src/js/charts.js), which concatenate
# ALL scalars ahead of ALL composites and sort each half flat by ``order``. Neither
# honours ``group``, ``group_order`` or ``series_group`` - those belong to the Dynamics
# tab's manifest builder. Two things went wrong because of that and are pinned here:
#
# * four (since removed) density entries carried ``series_group`` expecting to merge
# into two cards, so the deck rendered four; * they also carried ``order: 10``, tying
# with ``loss``, and a stable sort puts the earlier-declared entry first - which put a
# research probe at deck position 1, ahead of training loss.
#
# The information-density probe now emits only ``readout_*`` keys into extra_metrics (no
# schema columns), claimed by the composite cards pinned below.


# --- run comparison ----------------------------------------------------------


def _payload_metric_cards(run_hash, limit=1000):
    """The exact per-run payload /api/metrics builds for the Research tab."""
    import pathlib

    from praxis.web.routes.metrics import (
        _downsample_metrics,
        _read_metrics_file,
        _transform_metrics,
    )

    db = pathlib.Path("build/runs") / run_hash / "metrics.db"
    if not db.exists():
        return None
    rows = _read_metrics_file(db, 0, max_rows=limit * 3)
    if not rows:
        return None
    if len(rows) > limit:
        rows = _downsample_metrics(rows, limit, "lttb")
    return _transform_metrics(rows)


def _has(payload, key):
    values = payload.get(key)
    return bool(values) and any(v is not None for v in values)


def test_no_run_loses_a_series_it_actually_recorded():
    """A run must surface every charted metric it genuinely has data for.

    Older runs used to vanish from the Research tab entirely: the SELECT named
    every registry column, so any run predating one raised ``no such column``,
    ``_read_metrics_file`` swallowed it, and the caller dropped the run - taking
    its loss curve with it. The projection is per-database now
    (``_projection_for``), and this pins the property that guarantees: what the
    database holds is what the payload serves.

    Deliberately NOT "every run draws every card". A run that stopped before its
    first validation step has no val_loss, and that is correct, not a defect.
    """
    import pathlib
    import sqlite3

    checked = 0
    for path in sorted(pathlib.Path("build/runs").iterdir())[:6]:
        db = path / "metrics.db"
        if not db.exists():
            continue
        payload = _payload_metric_cards(path.name, limit=200)
        if payload is None:
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(metrics)")}
            for key, entry in TRAINING_METRIC_REGISTRY.items():
                if not entry.get("chart") or key not in columns:
                    continue
                recorded = conn.execute(
                    f"SELECT COUNT(*) FROM metrics WHERE {key} IS NOT NULL"
                ).fetchone()[0]
                if recorded:
                    assert _has(payload, key), (
                        f"{path.name}: {key!r} has {recorded} recorded values "
                        f"but was dropped from the payload"
                    )
        finally:
            conn.close()
        checked += 1
    if checked == 0:
        pytest.skip("no runs on disk")


def test_core_metrics_survive_in_every_run_on_disk():
    """loss / val_loss are the comparison baseline - they must never drop out."""
    import pathlib

    checked = 0
    for path in sorted(pathlib.Path("build/runs").iterdir()):
        if not (path / "metrics.db").exists():
            continue
        payload = _payload_metric_cards(path.name, limit=200)
        if payload is None:
            continue
        checked += 1
        assert "steps" in payload, f"{path.name}: no step axis"
        assert _has(payload, "loss"), f"{path.name}: lost its loss series"
    if checked == 0:
        pytest.skip("no runs on disk")


def test_validation_points_carry_every_x_coordinate():
    """A val point must know exactly where it sits on EVERY axis.

    This is the property the whole feature rests on. Validation is computed
    every ``val_check_interval`` steps, and the worry was that plotting it
    against tokens would silently lag by up to one interval. It does not:
    ``MetricsLoggerCallback.on_validation_end`` drains callback_metrics at
    ``trainer.global_step``, and ``MetricsLogger.log`` upserts on the step
    primary key, so the val write MERGES into the row the training step already
    wrote - carrying that step's exact num_tokens and ts.

    If a refactor ever splits validation onto its own row, the coordinates go
    null and this fails. That failure is the point: a val curve plotted against
    a carried-forward token count is wrong in a way nobody would see.
    """
    import pathlib

    sources = [axis["source"] for axis in X_AXIS_REGISTRY]

    checked = 0
    for path in sorted(pathlib.Path("build/runs").iterdir()):
        if not (path / "metrics.db").exists():
            continue
        payload = _payload_metric_cards(path.name, limit=200)
        if payload is None or not _has(payload, "val_loss"):
            continue
        checked += 1
        val_rows = [i for i, v in enumerate(payload["val_loss"]) if v is not None]
        for source in sources:
            column = payload.get(source)
            assert column is not None, f"{path.name}: no {source} column"
            assert len(column) == len(
                payload["val_loss"]
            ), f"{path.name}: {source} is not index-aligned with val_loss"
            missing = [i for i in val_rows if column[i] is None]
            assert not missing, (
                f"{path.name}: {len(missing)} validation points have no {source} "
                f"coordinate - they cannot be plotted on that axis"
            )
    if checked == 0:
        pytest.skip("no run on disk has validation data")


def test_elapsed_discounts_a_suspension():
    """A stopped-and-resumed run must not be billed for the hours it sat idle."""
    from praxis.web.routes.metrics import _TS_RAW, _elapsed_seconds

    rows = lambda stamps: [{_TS_RAW: t} for t in stamps]

    # Uninterrupted: elapsed is exactly the span.
    assert _elapsed_seconds(rows([0, 10, 20, 30]))[-1] == 30

    # An 8-hour pause between 10-second rows contributes one typical interval,
    # not 8 hours.
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
# cards
# ------------------------------------------------------------------------------


AUTHORS = ["Ryan J. Brooks"]
DONATE = "https://example.com/donate"


@pytest.fixture
def cards_client():
    app = Flask(__name__)
    app.register_blueprint(cards_bp)
    app.config["author"] = AUTHORS
    app.config["donations"] = DONATE
    app.config["truncated_hash"] = "abc123"
    return app.test_client()


def test_preview_route(cards_client):
    resp = cards_client.get(
        "/api/card/preview.svg?seed=5&side=front&theme=dark&hue=161"
    )
    assert resp.status_code == 200
    assert resp.mimetype == "image/svg+xml"
    assert resp.headers["X-Card-Seed"] == "5"


def test_preview_route_random_seed(cards_client):
    resp = cards_client.get("/api/card/preview.svg")
    assert resp.status_code == 200
    assert int(resp.headers["X-Card-Seed"]) >= 0


def test_zip_routes(cards_client):
    import io
    import zipfile

    for path, names in [
        ("/api/card/cards.zip", {"praxis-card-front.pdf", "praxis-card-back.pdf"}),
        (
            "/api/card/sheets.zip",
            {"praxis-cards-10up-front.pdf", "praxis-cards-10up-back.pdf"},
        ),
    ]:
        resp = cards_client.get(f"{path}?seed=5")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        assert set(zf.namelist()) == names
        for n in names:
            assert zf.read(n)[:4] == b"%PDF"


def test_pdf_routes(cards_client):
    for path, name in [
        ("/api/card/card.pdf", "praxis-card-back.pdf"),
        ("/api/card/sheet.pdf", "praxis-cards-10up-back.pdf"),
    ]:
        resp = cards_client.get(f"{path}?seed=5&side=back")
        assert resp.status_code == 200
        assert resp.data[:4] == b"%PDF"
        assert name in resp.headers["Content-Disposition"]


# ------------------------------------------------------------------------------
# api
# ------------------------------------------------------------------------------
# Comprehensive test suite for the Praxis API server.


class TestCoreRoutes:
    """Test core API routes."""

    def test_ping_endpoint(self, api_url):
        """Test /api/ping endpoint."""
        response = requests.get(f"{api_url}/api/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "Praxis API server is running" in data["message"]

    def test_ping_post(self, api_url):
        """Test /api/ping with POST method."""
        response = requests.post(f"{api_url}/api/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_ping_options(self, api_url):
        """Test /api/ping with OPTIONS method."""
        response = requests.options(f"{api_url}/api/ping")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_spec_endpoint(self, api_url):
        """Test /api/spec endpoint."""
        response = requests.get(f"{api_url}/api/spec")
        assert response.status_code == 200, response.text
        data = response.json()

        # Check required fields
        assert "truncated_hash" in data
        assert data["truncated_hash"] == "test12345"
        assert "full_hash" in data
        assert data["full_hash"] == "test1234567890abcdef"
        assert "args" in data
        assert "param_stats" in data
        assert data["param_stats"]["total"] == 1000000
        assert "seed" in data
        assert data["seed"] == 42

    def test_home_page(self, api_url):
        """Test home page returns HTML."""
        response = requests.get(f"{api_url}/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("Content-Type", "")
        # Check for CSP header
        assert "Content-Security-Policy" in response.headers

        # Check that it's actually HTML content
        assert "<!DOCTYPE html>" in response.text
        assert "<title>Praxis</title>" in response.text


class TestGenerationRoutes:
    """Test generation API routes."""

    def test_input_generation(self, api_url):
        """Test /input endpoint for string-based generation."""
        payload = {"prompt": "Hello, world!", "max_new_tokens": 50, "temperature": 0.7}
        response = requests.post(f"{api_url}/input", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "Generated response" in data["response"]

    def test_input_missing_prompt(self, api_url):
        """Test /input endpoint with missing prompt."""
        payload = {"max_new_tokens": 50}
        response = requests.post(f"{api_url}/input", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "prompt" in data["error"].lower()

    def test_input_with_messages_error(self, api_url):
        """Test /input endpoint rejects messages."""
        payload = {"prompt": "test", "messages": [{"role": "user", "content": "test"}]}
        response = requests.post(f"{api_url}/input", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "/messages endpoint" in data["error"]

    def test_messages_generation(self, api_url):
        """Test /messages endpoint for chat-based generation."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "max_new_tokens": 50,
        }
        response = requests.post(f"{api_url}/messages", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        # Response should contain generated text
        assert len(data["response"]) > 0

    def test_messages_missing_messages(self, api_url):
        """Test /messages endpoint with missing messages."""
        payload = {"max_new_tokens": 50}
        response = requests.post(f"{api_url}/messages", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "messages" in data["error"].lower()

    def test_generation_options(self, api_url):
        """Test generation endpoints with OPTIONS method."""
        response = requests.options(f"{api_url}/input")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

        response = requests.options(f"{api_url}/messages")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers


class TestAgentsRoute:
    """Test agent discovery route."""

    @patch("subprocess.run")
    def test_agents_endpoint(self, mock_run, api_url):
        """Test /api/agents endpoint."""
        # Mock git commands
        mock_run.return_value = Mock(
            returncode=0,
            stdout="origin\thttps://github.com/test/repo.git\t(fetch)\n",
            stderr="",
        )

        response = requests.get(f"{api_url}/api/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)

        # Should at least have a "self-*" agent
        agent_names = [agent["name"] for agent in data["agents"]]
        assert any(name.startswith("self-") for name in agent_names)

    def test_agents_options(self, api_url):
        """Test /api/agents with OPTIONS method."""
        response = requests.options(f"{api_url}/api/agents")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers


class TestStaticRoutes:
    """Test static file serving routes."""

    def test_favicon(self, api_url):
        """Test favicon.ico endpoint."""
        response = requests.get(f"{api_url}/favicon.ico")
        # Should return 204 No Content if favicon doesn't exist
        assert response.status_code in [200, 204]

    def test_static_files(self, api_url):
        """Test static file serving."""
        # Try to get a non-existent static file
        response = requests.get(f"{api_url}/static/nonexistent.js")
        # Should return 404
        assert response.status_code == 404


class TestGitRoutes:
    """Test Git HTTP backend routes."""

    @patch("subprocess.run")
    def test_git_info_refs(self, mock_run, api_url):
        """Test git info/refs endpoint."""
        mock_run.return_value = Mock(returncode=0, stdout=b"abc123\tHEAD\n", stderr=b"")

        response = requests.get(f"{api_url}/praxis/info/refs?service=git-upload-pack")
        assert response.status_code == 200
        assert "application/x-git-upload-pack-advertisement" in response.headers.get(
            "Content-Type", ""
        )

    def test_git_invalid_service(self, api_url):
        """Test git endpoint with invalid service."""
        response = requests.get(f"{api_url}/praxis/info/refs?service=invalid")
        assert response.status_code == 400

    def test_git_write_denied(self, api_url):
        """Test git write access is denied."""
        response = requests.get(f"{api_url}/praxis/info/refs?service=git-receive-pack")
        assert response.status_code == 403


def test_activation_curves_route_preserves_training_mode():
    """The /api/activation_curves route runs on the API server thread against
    the live training model. It must never flip the shared model's train/eval
    mode: doing so races the training forward, and a CALM stage-2 forward
    observed in eval mode detaches its entire loss (every grad-bearing term is
    gated on self.training), crashing backward().
    """
    import torch.nn as nn

    from praxis.web.routes.dynamics import _compute_activation_curves

    model = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    model.train()

    seen_training = []
    model[1].register_forward_hook(lambda m, i, o: seen_training.append(model.training))

    curves, _ = _compute_activation_curves(model, -6.0, 6.0, 64)

    assert curves, "expected at least one activation curve (GELU)"
    assert model.training is True, "route left the model out of train mode"
    assert seen_training and all(
        seen_training
    ), "model dropped out of train mode during sampling"


# ------------------------------------------------------------------------------
# annotated_config
# ------------------------------------------------------------------------------
# The annotated config behind the web app's Download button.


def test_route_serves_the_annotated_file(tmp_path, monkeypatch):
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
