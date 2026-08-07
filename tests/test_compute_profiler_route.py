"""/api/head_snapshots must serve the compute treemap stashed on the model."""

import json

import pytest
from flask import Flask

from praxis.web.routes.dynamics import dynamics_bp


class FakeModel:
    """Stands in for the live model the generator holds."""

    head = None
    criterion = None
    encoder = None


@pytest.fixture
def client():
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


def test_route_serves_the_stashed_profile(client):
    model = FakeModel()
    model._compute_profile = _payload()
    client.config["generator"] = type("G", (), {"model": model})()

    with client.test_client() as c:
        body = json.loads(c.get("/api/head_snapshots").data)

    assert body["status"] == "ok"
    profile = body["snapshots"]["compute_profile"]
    assert profile["samples"] == 3
    assert profile["groups"][0]["name"] == "ArcAttention"
    assert sum(g["share"] for g in profile["groups"]) == pytest.approx(1.0)


def test_route_is_quiet_without_the_profiler(client):
    """A run that never profiled (e.g. torch.compile) grows no compute card."""
    client.config["generator"] = type("G", (), {"model": FakeModel()})()

    with client.test_client() as c:
        body = json.loads(c.get("/api/head_snapshots").data)

    assert "compute_profile" not in body.get("snapshots", {})


def test_route_ignores_a_non_dict_stash(client):
    model = FakeModel()
    model._compute_profile = "not a dict"
    client.config["generator"] = type("G", (), {"model": model})()

    with client.test_client() as c:
        body = json.loads(c.get("/api/head_snapshots").data)

    assert "compute_profile" not in body.get("snapshots", {})


def test_snapshot_producer_recipe_includes_the_profile():
    """The recipe - NOT the route's fallback - is what serves a live run.

    ``serve_snapshot`` only calls the route's ``_live()`` before the producer
    thread has filled the store, i.e. cold start. Once running, this recipe is
    the only path, so it must carry everything the fallback does.
    """
    from praxis.web.snapshots import _recipe_head_snapshots

    model = FakeModel()
    model._compute_profile = _payload()

    out = _recipe_head_snapshots(model)

    assert out["status"] == "ok"
    assert "compute_profile" in out["snapshots"]
    assert out["snapshots"]["compute_profile"]["samples"] == 3


def test_snapshot_producer_recipe_is_quiet_without_the_profiler():
    from praxis.web.snapshots import _recipe_head_snapshots

    out = _recipe_head_snapshots(FakeModel())
    assert "compute_profile" not in out.get("snapshots", {})


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


def test_payload_is_json_serialisable_end_to_end():
    """The renderer receives exactly what ComputeProfiler.snapshot() produced."""
    from praxis.metrics.compute import ComputeProfiler

    prof = ComputeProfiler()
    prof._calls = {"a.b|Alpha": 2}
    prof._fold({"a.b|Alpha": 6.0}, {"a.b|Alpha": 4.0}, 20.0, 10.0)
    snap = prof.snapshot()

    round_tripped = json.loads(json.dumps(snap))
    assert round_tripped == snap
    groups = round_tripped["compute_profile"]["groups"]
    # every field the JS renderer reads must survive the trip
    for g in groups:
        assert {"name", "ms", "share", "calls", "outside", "residual", "children"} <= set(g)
        for child in g["children"]:
            assert {"name", "ms", "share", "fwd_ms", "bwd_ms", "calls"} <= set(child)
