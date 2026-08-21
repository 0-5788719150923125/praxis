"""Precomputed API snapshots: one producer, many cheap readers.

The expensive dashboard endpoints (activation curves, head snapshots, evolution)
probe the live model. Computing them per-request means every client - and there
can be many on a public server - stampedes the model and races its train/eval
mode (the calm-c stage-2 crash). Instead each snapshot is computed on a fixed
cadence and stashed here; the routes just read the latest.

Readers never touch the model, so concurrent requests are pure dict lookups.

WHERE a recipe runs matters as much as how often. A recipe that runs a torch op
on the live model may only run on the TRAINING thread. Not "should" - may not.
Two threads running ops on the same tensor deadlock the process outright:

    training thread   add(amp_coeffs, ...) -> collect_next_edges
                      -> grad_accumulator LOCKS the tensor's AutogradMeta
                      -> incref_pyobject WANTS the GIL
    producer thread   amp_coeffs[..., :F_t] HOLDS the GIL (getitem never
                      drops it) -> slice_Tensor -> fw_grad
                      WANTS the same AutogradMeta lock

That is an ABBA deadlock on (GIL, AutogradMeta.mutex_), and it wedged
abstractinator-s hard enough that even the stall watchdog could not report it:
the priority dump needs the GIL, and the GIL is exactly what is stuck. It is
not a rare-tensor problem either - any op on any tensor both threads touch will
do it, so the rule has to be structural.

So recipes declare where they run. ``Recipe.on_trainer`` (the DEFAULT, because
the unsafe direction must be the one you opt out of) means the training loop
runs it, pumped from a batch-end hook by ``SnapshotPumpCallback`` - the same
shape as ``GenerationQueueCallback``. Recipes that only read SQLite or git set
``on_trainer=False`` and keep running on the background thread, where a slow
whole-table scan costs the run nothing. Until a pump attaches (no trainer at
all, e.g. an inference-only server) there is nothing to race and the thread
runs everything.
"""

import logging
import queue
import threading
import time
from typing import Callable, NamedTuple

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
        # Optional fn(name, version), called after a slot actually changes.
        # The server wires this to a websocket "invalidate" broadcast so
        # clients refresh on change instead of polling on a timer.
        self.notify = None

    def set(self, name, payload):
        with self._lock:
            prev = self._data.get(name)
            # Identical recompute: keep the version (and thus the ETag) stable
            # so clients keep getting 304s and never re-render unchanged data.
            if prev is not None and prev["payload"] == payload:
                prev["computed_at"] = time.time()
                return
            self._version += 1
            version = self._version
            self._data[name] = {
                "version": version,
                "computed_at": time.time(),
                "payload": payload,
            }
        if self.notify is not None:
            try:
                self.notify(name, version)
            except Exception as exc:
                api_logger.debug(f"[snapshots] notify failed: {exc}")

    def get(self, name):
        with self._lock:
            return self._data.get(name)


# How long a request thread will wait for the training loop to run its probe.
# Generous because it is bounded by one training step, and a step is
# milliseconds until something (swap thrash, a compile) makes it not.
PROBE_TIMEOUT_S = 30.0


def probe_model(fn, timeout=PROBE_TIMEOUT_S):
    """Run a live model probe ``fn()`` from a request thread, safely.

    Routes must never run a torch op on the live model themselves - see the
    module docstring for the deadlock. This hands the work to the producer,
    which either runs it inline (no training loop to race) or has the training
    loop run it at its next batch boundary. Without a producer in the app
    config (tests, a bare app) there is nothing to race either.
    """
    from flask import current_app

    producer = current_app.config.get("snapshot_producer")
    return fn() if producer is None else producer.submit(fn, timeout=timeout)


def serve_snapshot(name, fallback, cache_seconds=5, touches_model=False):
    """Serve snapshot ``name`` as JSON with ETag revalidation. Falls back to a
    live ``fallback()`` compute only before the producer has filled the slot
    (cold start, or snapshots disabled).

    ``touches_model`` routes that fallback through :func:`probe_model` instead
    of running it on the request thread.
    """
    from flask import current_app, jsonify, request

    store = current_app.config.get("snapshot_store")
    entry = store.get(name) if store else None

    if entry is None:
        payload = probe_model(fallback) if touches_model else fallback()
        resp = jsonify(payload)
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


class Recipe(NamedTuple):
    """A snapshot recipe and where it is allowed to run.

    Args:
        fn: ``fn(model) -> payload``. Mirrors what its route used to return,
            tolerates ``model=None``, and never flips the model's mode.
        interval: seconds between recomputes.
        on_trainer: True (the default) when ``fn`` runs a torch op on the live
            model, and so may only run on the training thread. See the module
            docstring for what happens when it does not. Set False ONLY for a
            recipe that reads no tensors at all - SQLite, git, plain attribute
            walks - which then keeps its slow scan off the training loop.
    """

    fn: Callable
    interval: float
    on_trainer: bool = True


def _as_recipe(value) -> Recipe:
    """Accept a ``Recipe`` or a bare ``(fn, interval)`` pair.

    The pair form predates the ``on_trainer`` split, so it lands on the safe
    default: an unannotated recipe is assumed to touch the model.
    """
    return value if isinstance(value, Recipe) else Recipe(*value)


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
    # Memory surfacings live inside the decoder blocks; collect the first one
    # that offers a snapshot (e.g. the dual-regime river). One card, not one per
    # block - they share the same competition dynamics.
    if hasattr(model, "modules"):
        for mod in model.modules():
            fn = getattr(mod, "dashboard_snapshots", None)
            if callable(fn) and type(mod).__name__.startswith("Memory"):
                snaps = fn() or {}
                if snaps:
                    snapshots.update(snaps)
                    break
    # Attention mechanisms with their own geometry to draw (SSOG's field).
    # A walk, because attention is not an attribute of the model the way the
    # head and the encoder are, and unlike the memory surfacings above we want
    # every one of them, not the first.
    if hasattr(model, "modules"):
        from praxis.metrics.specialization import collect_attention_snapshots

        snapshots.update(collect_attention_snapshots(model) or {})
    # RLCT loss-landscape grid, stashed on the model by RLCTLandscapeCallback.
    rlct = getattr(model, "_rlct_landscape", None)
    if isinstance(rlct, dict):
        snapshots.update(rlct)
    # Per-module compute-time treemap, stashed by ComputeProfilerCallback.
    compute = getattr(model, "_compute_profile", None)
    if isinstance(compute, dict):
        snapshots.update(compute)
    return {
        "status": "ok" if snapshots else "no_data",
        "snapshots": snapshots,
    }


def _recipe_evolution(model):
    from praxis.pillars.evolution import evolution_data

    data = evolution_data()
    return {"status": "ok" if data else "no_data", "data": data or None}


# name -> Recipe. Model probes track the model every few seconds; evolution is
# git-derived and only changes on commit, so it idles slow.
DEFAULT_RECIPES = {
    "activation_curves": Recipe(_recipe_activation_curves, DEFAULT_INTERVAL),
    "head_snapshots": Recipe(_recipe_head_snapshots, DEFAULT_INTERVAL),
    # Reads git history, never the model.
    "evolution": Recipe(_recipe_evolution, 60.0, on_trainer=False),
}


# --- Recipes needing the run hash, not the model. Not part of DEFAULT_RECIPES
# (the hash is per-APIServer-instance); server.py wraps these in closures over
# self.truncated_hash when it builds the producer's recipes dict. ---


def _recipe_dynamics(model, run_hash):
    from pathlib import Path

    from .routes.dynamics import _fetch_dynamics_payload

    if not run_hash:
        return {"status": "no_data", "runs": []}
    dynamics_file = Path("build/runs") / run_hash / "dynamics.db"
    return _fetch_dynamics_payload(dynamics_file, run_hash, model)


def _recipe_data_metrics(model, run_hash):
    from .routes.data_metrics import _current_run_data_metrics

    if not run_hash:
        return {"status": "no_data", "message": "No current run"}
    return _current_run_data_metrics(run_hash)


class SnapshotProducer:
    """Keeps each recipe's snapshot fresh on its own cadence.

    Two engines, one schedule. The background thread runs the recipes that
    touch nothing but SQLite and git (``on_trainer=False``), so a slow
    whole-table scan never costs the run a step. Everything else runs on the
    training thread, pumped from a batch-end hook, because a torch op on the
    live model from any other thread can deadlock the process (module
    docstring). Before a pump attaches there is no training loop to race, so
    the thread runs both kinds.

    Either way a recipe is only recomputed once its own interval has elapsed,
    so a slow recipe never holds up a fast one.
    """

    # A queued job whose waiter has already given up is not worth a step: the
    # answer would be thrown away and the training loop paid for it. Mirrors
    # the generation queue's deadline handling.
    MAX_JOBS_PER_PUMP = 4

    def __init__(self, store, model_fn, shutdown_event, recipes=None, tick=1.0):
        self.store = store
        self.model_fn = model_fn  # called each tick for the current model
        self.shutdown_event = shutdown_event
        source = recipes if recipes is not None else DEFAULT_RECIPES
        self.recipes = {name: _as_recipe(v) for name, v in source.items()}
        self.tick = tick
        self._due = {name: 0.0 for name in self.recipes}  # monotonic next-run time
        self._thread = None
        # Held whenever a NON-training thread runs model-touching work. Its
        # only job is to fence `attach_pump`: the pump cannot simply flip a
        # flag and return, because the producer thread may already be inside a
        # recipe, and the first training batch would then race it.
        self._model_lock = threading.RLock()
        self._pumped = False
        self._jobs = queue.Queue()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="snapshot-producer")
        self._thread.daemon = True
        self._thread.start()

    # -- handing the model over to the training loop -----------------------

    def attach_pump(self):
        """Stop touching the model from this thread; ``pump`` takes over.

        Taking ``_model_lock`` once is the fence: it returns only when no
        other thread is inside a model-touching recipe, so the caller can
        start stepping the model knowing it has the field to itself.
        """
        with self._model_lock:
            self._pumped = True

    def detach_pump(self):
        """Training is over - the thread may touch the model again."""
        self._pumped = False

    def pump(self):
        """Run everything that must run on the training thread.

        Called from a training-loop hook. Cheap when nothing is due: a
        dict scan and a monotonic clock read.
        """
        self._drain_jobs()
        due = self._due_recipes(on_trainer=True)
        if due:
            self._compute(due)

    def submit(self, fn, timeout=None):
        """Run ``fn()`` somewhere it is safe to touch the model, return its result.

        With no pump attached there is no training loop to race, so it runs
        inline (under the lock, so an attach in flight waits it out). Otherwise
        the training loop runs it at its next batch boundary. This is the way
        in for the routes' live cold-start and custom-range probes, which would
        otherwise run the model straight off a request thread.
        """
        with self._model_lock:
            if not self._pumped:
                return fn()
            box = {}
            done = threading.Event()
            deadline = None if timeout is None else time.monotonic() + timeout
            self._jobs.put((fn, box, done, deadline))
        if not done.wait(timeout):
            raise TimeoutError("Timed out waiting for the training loop to run a probe")
        if "exc" in box:
            raise box["exc"]
        return box["result"]

    # -- internals ---------------------------------------------------------

    def _due_recipes(self, on_trainer):
        now = time.monotonic()
        return [
            name
            for name, when in self._due.items()
            if now >= when and self.recipes[name].on_trainer == on_trainer
        ]

    def _compute(self, due):
        model = None
        try:
            model = self.model_fn()
        except Exception as exc:
            api_logger.debug(f"[snapshots] model unavailable: {exc}")
        for name in due:
            if self.shutdown_event.is_set():
                break
            recipe = self.recipes[name]
            try:
                self.store.set(name, recipe.fn(model))
            except Exception as exc:
                # One bad recipe never stalls the loop or the others.
                api_logger.warning(f"[snapshots] {name} compute failed: {exc}")
            self._due[name] = time.monotonic() + recipe.interval

    def _drain_jobs(self):
        served = 0
        while served < self.MAX_JOBS_PER_PUMP:
            try:
                fn, box, done, deadline = self._jobs.get_nowait()
            except queue.Empty:
                return
            if deadline is not None and time.monotonic() >= deadline:
                # Nobody is listening any more. Waking the waiter costs
                # nothing and running the probe would cost a step, so drop it
                # without counting it against the budget.
                done.set()
                continue
            try:
                box["result"] = fn()
            except Exception as exc:
                box["exc"] = exc
            finally:
                done.set()
            served += 1

    def _run(self):
        while not self.shutdown_event.is_set():
            with self._model_lock:
                # Model recipes are this thread's only while nothing pumps.
                # Under the lock so `attach_pump` can fence a recipe already
                # in flight rather than racing the first training batch.
                if not self._pumped:
                    due = self._due_recipes(on_trainer=True)
                    if due:
                        self._compute(due)
            due = self._due_recipes(on_trainer=False)
            if due:
                self._compute(due)
            self.shutdown_event.wait(self.tick)
