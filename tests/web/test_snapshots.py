"""Snapshot producer (praxis/web/snapshots.py): model-touching recipes run on
the training thread, never beside it.

The producer's own thread running a torch op on the live model is what wedged
abstractinator-s: an ABBA deadlock on (GIL, AutogradMeta.mutex_) that not even
the stall watchdog could report. These pin the structural rule that replaced
it, since the deadlock itself is timing-dependent.
"""

import threading
import time

import pytest

from praxis.web.snapshots import (
    DEFAULT_RECIPES,
    Recipe,
    SnapshotProducer,
    SnapshotStore,
    _as_recipe,
    _recipe_classifier_snapshots,
)


@pytest.fixture
def make_producer():
    """Producer factory; every producer's thread is stopped on teardown, so a
    failed assertion cannot leave one running."""
    made = []

    def make(recipes, tick=0.01):
        p = SnapshotProducer(
            store=SnapshotStore(),
            model_fn=lambda: "MODEL",
            shutdown_event=threading.Event(),
            recipes=recipes,
            tick=tick,
        )
        made.append(p)
        return p

    yield make
    for p in made:
        p.shutdown_event.set()


def _recorder():
    """A recipe that records the thread that ran it."""
    seen = []

    def recipe(model):
        seen.append(threading.current_thread().name)
        return {"n": len(seen)}

    return seen, recipe


def _wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)
    return cond()


def _raise():
    raise RuntimeError("probe failed")


def test_bare_tuple_recipes_default_to_the_safe_side():
    """An unannotated recipe is assumed to touch the model, not assumed safe."""
    assert _as_recipe((print, 1.0)).on_trainer is True
    assert _as_recipe(Recipe(print, 1.0, on_trainer=False)).on_trainer is False


def test_model_probing_defaults_are_marked_on_trainer():
    assert DEFAULT_RECIPES["activation_curves"].on_trainer is True
    assert DEFAULT_RECIPES["classifier_snapshots"].on_trainer is True
    # git-derived, no tensors: keeps running off the training loop
    assert DEFAULT_RECIPES["evolution"].on_trainer is False


def test_producer_thread_stops_touching_the_model_once_pumped(make_producer):
    seen, recipe = _recorder()
    p = make_producer({"probe": Recipe(recipe, 0.0)})
    p.attach_pump()
    p.start()
    time.sleep(0.1)
    assert seen == [], f"producer thread ran a model recipe: {seen}"

    p.pump()
    assert seen == [threading.current_thread().name]


def test_producer_thread_runs_everything_while_nothing_pumps(make_producer):
    """An inference-only server has no training loop to race."""
    seen, recipe = _recorder()
    p = make_producer({"probe": Recipe(recipe, 0.0)})
    p.start()
    assert _wait_for(lambda: seen)
    assert seen and seen[0] == "snapshot-producer"


def test_sqlite_recipes_stay_off_the_training_loop(make_producer):
    seen, recipe = _recorder()
    p = make_producer({"scan": Recipe(recipe, 0.0, on_trainer=False)})
    p.attach_pump()
    p.pump()
    assert seen == [], "a non-model recipe should not cost the training loop a step"

    p.start()
    assert _wait_for(lambda: seen)
    assert seen and seen[0] == "snapshot-producer"


def test_intervals_still_gate_pumped_recipes(make_producer):
    seen, recipe = _recorder()
    p = make_producer({"probe": Recipe(recipe, 60.0)})
    p.attach_pump()
    p.pump()
    p.pump()
    assert len(seen) == 1, "a pump per step must not mean a recompute per step"


def test_a_failing_recipe_does_not_escape_the_pump(make_producer):
    def boom(model):
        raise RuntimeError("nope")

    p = make_producer({"probe": Recipe(boom, 0.0)})
    p.attach_pump()
    p.pump()  # must not raise


def test_submit_runs_inline_when_no_training_loop_owns_the_model(make_producer):
    p = make_producer({})
    assert p.submit(lambda: "inline") == "inline"


def test_submit_defers_to_the_training_loop_once_pumped(make_producer):
    p = make_producer({})
    p.attach_pump()
    ran = []

    result = {}

    def waiter():
        result["value"] = p.submit(lambda: ran.append("x") or "probed", timeout=5)

    t = threading.Thread(target=waiter)
    t.start()
    assert _wait_for(lambda: not p._jobs.empty())
    assert ran == [], "the probe must wait for the training thread"

    p.pump()
    t.join(timeout=5)
    assert result["value"] == "probed"


def test_submit_propagates_the_probe_error_to_its_caller(make_producer):
    p = make_producer({})
    p.attach_pump()
    box = {}

    def waiter():
        try:
            p.submit(_raise, timeout=5)
        except RuntimeError as exc:
            box["exc"] = str(exc)

    t = threading.Thread(target=waiter)
    t.start()
    assert _wait_for(lambda: not p._jobs.empty())
    p.pump()
    t.join(timeout=5)
    assert box["exc"] == "probe failed"


def test_expired_probes_are_dropped_rather_than_costing_a_step(make_producer):
    p = make_producer({})
    p.attach_pump()
    ran = []
    done = threading.Event()
    p._jobs.put((lambda: ran.append("x"), {}, done, time.monotonic() - 1.0))

    p.pump()

    assert ran == []
    assert done.is_set(), "the waiter must be woken, not left hanging"


def test_attach_pump_waits_out_a_recipe_already_in_flight(make_producer):
    """The handover cannot just flip a flag: the first training batch would
    race whatever the producer thread is already inside."""
    entered = threading.Event()
    release = threading.Event()

    def slow(model):
        entered.set()
        release.wait(5)
        return {}

    p = make_producer({"probe": Recipe(slow, 0.0)})
    p.start()
    assert entered.wait(5)

    attached = threading.Event()
    threading.Thread(target=lambda: (p.attach_pump(), attached.set())).start()
    assert not attached.wait(0.2), "attach_pump returned while a recipe was running"

    release.set()
    assert attached.wait(5)


def test_classifier_snapshots_recipe_is_quiet_without_the_profiler(bare_model):
    assert "compute_profile" not in _recipe_classifier_snapshots(bare_model).get(
        "snapshots", {}
    )


def test_classifier_snapshots_recipe_merges_a_stashed_landscape(bare_model):
    """No classifier, criterion or encoder, and the stashed RLCT grid still shows."""
    bare_model._rlct_landscape = {"rlct_landscape": {"grid": [[0.0]], "status": "ok"}}
    out = _recipe_classifier_snapshots(bare_model)
    assert out["status"] == "ok"
    assert "rlct_landscape" in out["snapshots"]


# ---------------------------------------------------------------------------
# ETag validity across runs
# ---------------------------------------------------------------------------


def _snapshot_app(store):
    """A throwaway app whose one route is served from ``store``."""
    from flask import Flask

    from praxis.web.snapshots import serve_snapshot

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["snapshot_store"] = store

    @app.route("/api/thing")
    def thing():
        return serve_snapshot("thing", lambda: {"status": "no_data"})

    return app


def test_two_runs_never_share_an_etag():
    """The version counter restarts at 1 in every process, so it cannot be the
    whole validator: two runs' FIRST snapshots collided, and a browser
    revalidating at the same origin (localhost:2100 for every run) got a 304
    and kept rendering the previous run's data. That is how one model's
    activation curves appeared on a different model's dashboard."""
    from praxis.web.snapshots import SnapshotStore

    first, second = SnapshotStore(), SnapshotStore()
    first.set("thing", {"model": "abstractinator"})
    second.set("thing", {"model": "smollm"})

    # Same version - which is exactly why the version alone was not enough.
    assert first.get("thing")["version"] == second.get("thing")["version"]

    old_etag = _snapshot_app(first).test_client().get("/api/thing").headers["ETag"]
    response = (
        _snapshot_app(second)
        .test_client()
        .get("/api/thing", headers={"If-None-Match": old_etag})
    )

    assert response.status_code == 200, "a new run must not answer 304 to an old ETag"
    assert response.get_json() == {"model": "smollm"}


def test_the_same_run_still_revalidates_cheaply():
    """The 304 path is the point of the ETag; only cross-run reuse is wrong."""
    from praxis.web.snapshots import SnapshotStore

    store = SnapshotStore()
    store.set("thing", {"model": "smollm"})
    client = _snapshot_app(store).test_client()

    etag = client.get("/api/thing").headers["ETag"]
    assert client.get("/api/thing", headers={"If-None-Match": etag}).status_code == 304


def test_a_changed_payload_breaks_the_etag_within_a_run():
    from praxis.web.snapshots import SnapshotStore

    store = SnapshotStore()
    store.set("thing", {"step": 1})
    client = _snapshot_app(store).test_client()
    etag = client.get("/api/thing").headers["ETag"]

    store.set("thing", {"step": 2})
    assert client.get("/api/thing", headers={"If-None-Match": etag}).status_code == 200
