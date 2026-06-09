"""Precomputed API snapshots: one producer, many cheap readers.

The expensive dashboard endpoints (activation curves, head snapshots, evolution)
probe the live model. Computing them per-request means every client - and there
can be many on a public server - stampedes the model and races its train/eval
mode (the calm-c stage-2 crash). Instead a single background thread computes each
snapshot on a fixed cadence and stashes it here; the routes just read the latest.

Readers never touch the model, so concurrent requests are pure dict lookups and
the model is only ever read from one thread, at a known-safe point.
"""

import logging
import threading
import time

api_logger = logging.getLogger("praxis.web")

# Cadence for the producer loop. A touch under the routes' old max-age=5 so a
# stored snapshot is usually fresher than a client's poll interval.
DEFAULT_INTERVAL = 4.0


class SnapshotStore:
    """Thread-safe latest-value store. Each entry is replaced wholesale (never
    mutated in place), so a reader holding a payload reference is always safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}
        self._version = 0

    def set(self, name, payload):
        with self._lock:
            self._version += 1
            self._data[name] = {
                "version": self._version,
                "computed_at": time.time(),
                "payload": payload,
            }

    def get(self, name):
        with self._lock:
            return self._data.get(name)


def serve_snapshot(name, fallback, cache_seconds=5):
    """Serve snapshot ``name`` as JSON with ETag revalidation. Falls back to a
    live ``fallback()`` compute only before the producer has filled the slot
    (cold start, or snapshots disabled)."""
    from flask import current_app, jsonify, request

    store = current_app.config.get("snapshot_store")
    entry = store.get(name) if store else None

    if entry is None:
        resp = jsonify(fallback())
        resp.headers["Cache-Control"] = f"max-age={cache_seconds}"
        return resp

    etag = f'W/"{name}.{entry["version"]}"'
    if request.headers.get("If-None-Match") == etag:
        resp = current_app.response_class(status=304)
        resp.headers["ETag"] = etag
        return resp

    resp = jsonify(entry["payload"])
    resp.headers["ETag"] = etag
    resp.headers["Cache-Control"] = "no-cache"  # revalidate, but 304s are cheap
    return resp


# --- Recipes: (name -> fn(model) -> payload). Each mirrors what its route used
# to return, must tolerate model=None, and must never flip the model's mode. ---


def _recipe_activation_curves(model):
    from .routes.dynamics import _compute_activation_curves

    if model is None:
        return {"status": "no_data", "curves": []}
    x_min, x_max, points = -6.0, 6.0, 256
    curves, activation_type = _compute_activation_curves(model, x_min, x_max, points)
    return {
        "status": "ok" if curves else "no_data",
        "activation_type": activation_type,
        "x_range": [x_min, x_max],
        "curves": curves,
    }


def _recipe_head_snapshots(model):
    if model is None:
        return {"status": "no_data", "snapshots": {}}
    head = getattr(model, "head", None)
    criterion = getattr(model, "criterion", None)
    encoder = getattr(model, "encoder", None)

    snapshots = {}
    if head is not None:
        snapshots.update(head.dashboard_snapshots() or {})
    if criterion is not None and hasattr(criterion, "dashboard_snapshots"):
        snapshots.update(criterion.dashboard_snapshots() or {})
    if encoder and hasattr(encoder, "dashboard_snapshots"):
        snapshots.update(encoder.dashboard_snapshots() or {})
    return {
        "status": "ok" if snapshots else "no_data",
        "snapshots": snapshots,
    }


def _recipe_evolution(model):
    from praxis.pillars.evolution import evolution_data

    data = evolution_data()
    return {"status": "ok" if data else "no_data", "data": data or None}


# name -> (recipe, interval_seconds). Model probes track the model every few
# seconds; evolution is git-derived and only changes on commit, so it idles slow.
DEFAULT_RECIPES = {
    "activation_curves": (_recipe_activation_curves, DEFAULT_INTERVAL),
    "head_snapshots": (_recipe_head_snapshots, DEFAULT_INTERVAL),
    "evolution": (_recipe_evolution, 60.0),
}


class SnapshotProducer:
    """Daemon thread that keeps each recipe's snapshot fresh on its own cadence.

    Wakes every ``tick`` seconds and recomputes only the recipes whose interval
    has elapsed, so a slow recipe (evolution) never holds up a fast one and a
    fast one never drags the slow one along.
    """

    def __init__(self, store, model_fn, shutdown_event, recipes=None, tick=1.0):
        self.store = store
        self.model_fn = model_fn  # called each tick for the current model
        self.shutdown_event = shutdown_event
        self.recipes = recipes if recipes is not None else DEFAULT_RECIPES
        self.tick = tick
        self._due = {name: 0.0 for name in self.recipes}  # monotonic next-run time
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="snapshot-producer")
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        while not self.shutdown_event.is_set():
            now = time.monotonic()
            due = [n for n, t in self._due.items() if now >= t]
            if due:
                model = None
                try:
                    model = self.model_fn()
                except Exception as exc:
                    api_logger.debug(f"[snapshots] model unavailable: {exc}")
                for name in due:
                    if self.shutdown_event.is_set():
                        break
                    recipe, interval = self.recipes[name]
                    try:
                        self.store.set(name, recipe(model))
                    except Exception as exc:
                        # One bad recipe never stalls the loop or the others.
                        api_logger.warning(f"[snapshots] {name} compute failed: {exc}")
                    self._due[name] = time.monotonic() + interval

            self.shutdown_event.wait(self.tick)
