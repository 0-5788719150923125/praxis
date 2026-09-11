import threading
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.web.snapshots import (
    DEFAULT_RECIPES,
    Recipe,
    SnapshotProducer,
    SnapshotStore,
    _as_recipe,
)

# ------------------------------------------------------------------------------
# snapshot_pump
# ------------------------------------------------------------------------------
# Model-touching snapshots must run on the training thread, never beside it.
#
# The producer's own thread running a torch op on the live model is what wedged
# abstractinator-s: an ABBA deadlock on (GIL, AutogradMeta.mutex_) that not even the
# stall watchdog could report. These tests pin the structural rule that replaced it,
# since the deadlock itself is timing-dependent and cannot be reproduced reliably in a
# unit test.


def _producer(recipes, tick=0.01):
    return SnapshotProducer(
        store=SnapshotStore(),
        model_fn=lambda: "MODEL",
        shutdown_event=threading.Event(),
        recipes=recipes,
        tick=tick,
    )


def _recorder():
    """A recipe that records the thread that ran it."""
    seen = []
    return (
        seen,
        lambda model: (seen.append(threading.current_thread().name), {"n": len(seen)})[
            1
        ],
    )


def test_bare_tuple_recipes_default_to_the_safe_side():
    """An unannotated recipe is assumed to touch the model, not assumed safe."""
    assert _as_recipe((print, 1.0)).on_trainer is True
    assert _as_recipe(Recipe(print, 1.0, on_trainer=False)).on_trainer is False


def test_model_probing_defaults_are_marked_on_trainer():
    assert DEFAULT_RECIPES["activation_curves"].on_trainer is True
    assert DEFAULT_RECIPES["head_snapshots"].on_trainer is True
    # git-derived, no tensors: keeps running off the training loop
    assert DEFAULT_RECIPES["evolution"].on_trainer is False


def test_producer_thread_stops_touching_the_model_once_pumped():
    seen, recipe = _recorder()
    p = _producer({"probe": Recipe(recipe, 0.0)})
    p.attach_pump()
    p.start()
    time.sleep(0.1)
    assert seen == [], f"producer thread ran a model recipe: {seen}"

    p.pump()
    assert seen == [threading.current_thread().name]
    p.shutdown_event.set()


def test_producer_thread_runs_everything_while_nothing_pumps():
    """An inference-only server has no training loop to race."""
    seen, recipe = _recorder()
    p = _producer({"probe": Recipe(recipe, 0.0)})
    p.start()
    for _ in range(200):
        if seen:
            break
        time.sleep(0.01)
    p.shutdown_event.set()
    assert seen and seen[0] == "snapshot-producer"


def test_sqlite_recipes_stay_off_the_training_loop():
    seen, recipe = _recorder()
    p = _producer({"scan": Recipe(recipe, 0.0, on_trainer=False)})
    p.attach_pump()
    p.pump()
    assert seen == [], "a non-model recipe should not cost the training loop a step"

    p.start()
    for _ in range(200):
        if seen:
            break
        time.sleep(0.01)
    p.shutdown_event.set()
    assert seen and seen[0] == "snapshot-producer"


def test_intervals_still_gate_pumped_recipes():
    seen, recipe = _recorder()
    p = _producer({"probe": Recipe(recipe, 60.0)})
    p.attach_pump()
    p.pump()
    p.pump()
    assert len(seen) == 1, "a pump per step must not mean a recompute per step"


def test_a_failing_recipe_does_not_escape_the_pump():
    def boom(model):
        raise RuntimeError("nope")

    p = _producer({"probe": Recipe(boom, 0.0)})
    p.attach_pump()
    p.pump()  # must not raise


def test_submit_runs_inline_when_no_training_loop_owns_the_model():
    p = _producer({})
    assert p.submit(lambda: "inline") == "inline"


def test_submit_defers_to_the_training_loop_once_pumped():
    p = _producer({})
    p.attach_pump()
    ran = []

    result = {}

    def waiter():
        result["value"] = p.submit(lambda: ran.append("x") or "probed", timeout=5)

    t = threading.Thread(target=waiter)
    t.start()
    for _ in range(200):
        if not p._jobs.empty():
            break
        time.sleep(0.01)
    assert ran == [], "the probe must wait for the training thread"

    p.pump()
    t.join(timeout=5)
    assert result["value"] == "probed"


def test_submit_propagates_the_probe_error_to_its_caller():
    p = _producer({})
    p.attach_pump()
    box = {}

    def waiter():
        try:
            p.submit(_raise, timeout=5)
        except RuntimeError as exc:
            box["exc"] = str(exc)

    t = threading.Thread(target=waiter)
    t.start()
    for _ in range(200):
        if not p._jobs.empty():
            break
        time.sleep(0.01)
    p.pump()
    t.join(timeout=5)
    assert box["exc"] == "probe failed"


def _raise():
    raise RuntimeError("probe failed")


def test_expired_probes_are_dropped_rather_than_costing_a_step():
    p = _producer({})
    p.attach_pump()
    ran = []
    done = threading.Event()
    p._jobs.put((lambda: ran.append("x"), {}, done, time.monotonic() - 1.0))

    p.pump()

    assert ran == []
    assert done.is_set(), "the waiter must be woken, not left hanging"


def test_attach_pump_waits_out_a_recipe_already_in_flight():
    """The handover cannot just flip a flag: the first training batch would
    race whatever the producer thread is already inside."""
    entered = threading.Event()
    release = threading.Event()

    def slow(model):
        entered.set()
        release.wait(5)
        return {}

    p = _producer({"probe": Recipe(slow, 0.0)})
    p.start()
    assert entered.wait(5)

    attached = threading.Event()
    threading.Thread(target=lambda: (p.attach_pump(), attached.set())).start()
    assert not attached.wait(0.2), "attach_pump returned while a recipe was running"

    release.set()
    assert attached.wait(5)
    p.shutdown_event.set()


# ------------------------------------------------------------------------------
# compute_profiler_route
# ------------------------------------------------------------------------------
# /api/head_snapshots must serve the compute treemap stashed on the model.


class FakeModel:
    """Stands in for the live model the generator holds."""

    head = None
    criterion = None
    encoder = None


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


def test_snapshot_producer_recipe_is_quiet_without_the_profiler():
    from praxis.web.snapshots import _recipe_head_snapshots

    out = _recipe_head_snapshots(FakeModel())
    assert "compute_profile" not in out.get("snapshots", {})


# ------------------------------------------------------------------------------
# rlct_landscape
# ------------------------------------------------------------------------------
# RLCT loss-landscape probe (praxis.metrics.rlct + snapshot wiring).


class _Tiny(nn.Module):
    """Minimal causal-LM stand-in with a mutable buffer (to test restore)."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, dim)
        self.lin = nn.Linear(dim, vocab)
        self.register_buffer("ticks", torch.zeros(1))

    def forward(self, input_ids=None, labels=None, **kw):
        self.ticks += 1.0  # a train-mode side effect the probe must undo
        logits = self.lin(self.emb(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, : labels.size(1)].reshape(-1, self.vocab),
                labels.reshape(-1),
            )
        return SimpleNamespace(loss=loss, logits=logits)


def test_snapshot_recipe_merges_stashed_landscape():
    from praxis.web.snapshots import _recipe_head_snapshots

    model = _Tiny()
    # No head/criterion/encoder -> recipe must still surface the stashed grid.
    model.head = None
    model.criterion = None
    model.encoder = None
    model._rlct_landscape = {"rlct_landscape": {"grid": [[0.0]], "status": "ok"}}

    out = _recipe_head_snapshots(model)
    assert out["status"] == "ok"
    assert "rlct_landscape" in out["snapshots"]
