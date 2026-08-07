"""Per-module compute-time attribution: scoping, EMA smoothing, payload shape."""

import pytest
import torch
import torch.nn as nn

from praxis.metrics.compute import (
    COMPUTE_DEFAULTS,
    COMPUTE_METRIC_DESCRIPTIONS,
    OUTSIDE_MODEL,
    ComputeProfiler,
)


class Tiny(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.up = nn.Linear(d, d * 2)
        self.down = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.down(torch.relu(self.up(x))))


class Stack(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.blocks = nn.ModuleList([Tiny(d) for _ in range(n)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ── install / teardown ──────────────────────────────────────────────────────


def test_install_attaches_hooks_and_scopes():
    model = Stack()
    prof = ComputeProfiler()
    assert prof.install(model, device="cpu")
    assert model.blocks[0].up._praxis_scope == "blocks.0.up|Linear"
    prof.remove()


def test_install_accepts_a_compiled_model_at_coarse_depth():
    """Compiled models are instrumented coarsely, not refused."""
    inner = Stack()
    wrapper = torch.nn.Module()
    wrapper._orig_mod = inner  # what torch.compile's wrapper looks like
    prof = ComputeProfiler()
    assert prof.install(wrapper, device="cpu", max_depth=0) is True
    hooked = {
        m._praxis_scope.split("|")[0]
        for _, m in inner.named_modules()
        if hasattr(m, "_praxis_scope")
    }
    assert hooked == {"blocks"}
    prof.remove()


def test_hooks_are_inert_until_armed():
    model = Stack()
    prof = ComputeProfiler()
    prof.install(model, device="cpu")
    model(torch.randn(2, 16))
    assert prof.hooks_fired() == 0
    prof.remove()


def test_remove_detaches_every_hook():
    model = Stack()
    prof = ComputeProfiler()
    prof.install(model, device="cpu")
    prof.remove()
    assert not prof._handles
    for _, mod in model.named_modules():
        assert not getattr(mod, "_forward_pre_hooks", {})
        assert not getattr(mod, "_forward_hooks", {})


# ── attribution ─────────────────────────────────────────────────────────────


class FakeEvent:
    """Minimal stand-in for a profiler FunctionEvent."""

    def __init__(self, name, self_us=0, parent=None, seq=-1, fwd_thread=0):
        self.name = name
        self.self_cpu_time_total = self_us
        self.self_device_time_total = self_us
        self.cpu_parent = parent
        self.sequence_nr = seq
        self.fwd_thread = fwd_thread


def test_attribute_credits_forward_to_innermost_scope():
    prof = ComputeProfiler()
    prof._device = "cpu"
    outer = FakeEvent("@blocks.0|Tiny")
    inner = FakeEvent("@blocks.0.up|Linear", parent=outer)
    kernel = FakeEvent("aten::addmm", self_us=3000, parent=inner)
    loose = FakeEvent("aten::gelu", self_us=1000, parent=outer)

    fwd, bwd, total, attributed = prof.attribute([outer, inner, kernel, loose])

    assert fwd["blocks.0.up|Linear"] == pytest.approx(3.0)
    assert fwd["blocks.0|Tiny"] == pytest.approx(1.0)
    assert bwd == {}
    assert total == pytest.approx(4.0)
    assert attributed == pytest.approx(4.0)


def test_attribute_links_backward_subtree_by_sequence_nr():
    """Kernels hang off the autograd node, so the whole subtree is credited."""
    prof = ComputeProfiler()
    prof._device = "cpu"
    scope = FakeEvent("@blocks.0.up|Linear")
    fwd_op = FakeEvent("aten::addmm", self_us=1000, parent=scope, seq=7)
    node = FakeEvent("AddmmBackward0", self_us=10, seq=7, fwd_thread=1)
    bwd_kernel = FakeEvent("aten::mm", self_us=2000, parent=node)

    fwd, bwd, total, attributed = prof.attribute([scope, fwd_op, node, bwd_kernel])

    assert fwd["blocks.0.up|Linear"] == pytest.approx(1.0)
    # 10us node + 2000us kernel, both credited to the forward module
    assert bwd["blocks.0.up|Linear"] == pytest.approx(2.01)
    assert attributed == pytest.approx(total)


def test_attribute_leaves_unscoped_time_unattributed():
    prof = ComputeProfiler()
    prof._device = "cpu"
    scope = FakeEvent("@head|Linear")
    inside = FakeEvent("aten::addmm", self_us=1000, parent=scope)
    outside = FakeEvent("aten::nll_loss", self_us=500)

    _, _, total, attributed = prof.attribute([scope, inside, outside])

    assert total == pytest.approx(1.5)
    assert attributed == pytest.approx(1.0)


def test_attribute_handles_no_events():
    prof = ComputeProfiler()
    assert prof.attribute([]) == ({}, {}, 0.0, 0.0)


def test_attribute_survives_a_deep_event_chain():
    """The real tree runs to thousands of nodes; scoping must not recurse away."""
    prof = ComputeProfiler()
    prof._device = "cpu"
    root = FakeEvent("@deep|Linear")
    events = [root]
    cur = root
    for i in range(3000):
        cur = FakeEvent(f"aten::op{i}", self_us=1, parent=cur)
        events.append(cur)

    fwd, _, total, attributed = prof.attribute(events)

    assert fwd["deep|Linear"] == pytest.approx(3.0)
    assert attributed == pytest.approx(total)


# ── EMA smoothing ───────────────────────────────────────────────────────────


def test_ema_smooths_toward_a_shifted_value():
    prof = ComputeProfiler({"ema_alpha": 0.5})
    prof._fold({"a|A": 10.0}, {}, 10.0, 10.0)
    assert prof._ema_fwd["a|A"] == pytest.approx(10.0)  # first sample, unbiased
    prof._fold({"a|A": 20.0}, {}, 20.0, 20.0)
    assert prof._ema_fwd["a|A"] == pytest.approx(15.0)
    prof._fold({"a|A": 20.0}, {}, 20.0, 20.0)
    assert prof._ema_fwd["a|A"] == pytest.approx(17.5)


def test_ema_alpha_one_disables_smoothing():
    prof = ComputeProfiler({"ema_alpha": 1.0})
    prof._fold({"a|A": 10.0}, {}, 10.0, 10.0)
    prof._fold({"a|A": 3.0}, {}, 3.0, 3.0)
    assert prof._ema_fwd["a|A"] == pytest.approx(3.0)


def test_absent_scope_decays_instead_of_freezing():
    """An early-halted block genuinely cost nothing that pass."""
    prof = ComputeProfiler({"ema_alpha": 0.5})
    prof._fold({"a|A": 8.0, "b|B": 8.0}, {}, 16.0, 16.0)
    prof._fold({"a|A": 8.0}, {}, 8.0, 8.0)  # b absent
    assert prof._ema_fwd["b|B"] == pytest.approx(4.0)
    prof._fold({"a|A": 8.0}, {}, 8.0, 8.0)
    assert prof._ema_fwd["b|B"] == pytest.approx(2.0)


def test_ema_damps_alternating_samples():
    """The point of the smoothing: a flip-flopping input must not flip the card."""
    prof = ComputeProfiler({"ema_alpha": 0.2})
    seen = []
    for i in range(12):
        hot = 100.0 if i % 2 == 0 else 10.0
        prof._fold({"a|A": hot}, {}, hot, hot)
        seen.append(prof._ema_fwd["a|A"])
    swings = [abs(b - a) for a, b in zip(seen[4:], seen[5:])]
    assert max(swings) < 25.0  # raw input swings by 90 every sample


def test_samples_counter_tracks_folds():
    prof = ComputeProfiler()
    assert prof.samples == 0
    prof._fold({"a|A": 1.0}, {}, 1.0, 1.0)
    prof._fold({"a|A": 1.0}, {}, 1.0, 1.0)
    assert prof.samples == 2


# ── rollup / payload ────────────────────────────────────────────────────────


def _seeded(**cfg):
    prof = ComputeProfiler(cfg)
    prof._calls = {"x.0|Alpha": 4, "x.1|Alpha": 2, "y|Beta": 1}
    prof._fold(
        {"x.0|Alpha": 6.0, "x.1|Alpha": 2.0, "y|Beta": 2.0},
        {"x.0|Alpha": 4.0, "y|Beta": 1.0},
        20.0,
        15.0,
    )
    return prof


def test_snapshot_shares_sum_to_one():
    payload = _seeded().snapshot()["compute_profile"]
    assert sum(g["share"] for g in payload["groups"]) == pytest.approx(1.0)


def test_snapshot_groups_by_class_sorted_desc():
    payload = _seeded().snapshot()["compute_profile"]
    names = [g["name"] for g in payload["groups"]]
    assert names[0] == "Alpha"  # 6+2+4 = 12 ms beats Beta's 3
    ms = [g["ms"] for g in payload["groups"]]
    assert ms == sorted(ms, reverse=True)


def test_snapshot_exposes_unattributed_as_its_own_bucket():
    payload = _seeded().snapshot()["compute_profile"]
    outside = [g for g in payload["groups"] if g["outside"]]
    assert len(outside) == 1
    assert outside[0]["name"] == OUTSIDE_MODEL
    # 20 total, 15 attributed -> 5 ms outside the module tree
    assert outside[0]["ms"] == pytest.approx(5.0)


def test_snapshot_children_carry_the_fwd_bwd_split():
    payload = _seeded().snapshot()["compute_profile"]
    alpha = next(g for g in payload["groups"] if g["name"] == "Alpha")
    hot = next(c for c in alpha["children"] if c["name"] == "x.0")
    assert hot["fwd_ms"] == pytest.approx(6.0)
    assert hot["bwd_ms"] == pytest.approx(4.0)


def test_snapshot_is_empty_before_any_sample():
    assert ComputeProfiler().snapshot() == {}
    assert ComputeProfiler().metrics() == {}


def test_max_classes_folds_the_tail():
    prof = ComputeProfiler({"max_classes": 2})
    prof._fold({f"m{i}|C{i}": float(10 - i) for i in range(5)}, {}, 40.0, 40.0)
    payload = prof.snapshot()["compute_profile"]
    names = [g["name"] for g in payload["groups"] if not g["outside"]]
    assert len(names) == 3  # 2 real + 1 residual
    assert names[-1].startswith("other (")
    assert sum(g["share"] for g in payload["groups"]) == pytest.approx(1.0)


def test_top_k_folds_children_within_a_class():
    prof = ComputeProfiler({"top_k": 2})
    prof._fold({f"m{i}|Same": float(10 - i) for i in range(5)}, {}, 40.0, 40.0)
    payload = prof.snapshot()["compute_profile"]
    same = next(g for g in payload["groups"] if g["name"] == "Same")
    assert len(same["children"]) == 3  # 2 real + 1 residual
    assert same["children"][-1]["name"] == "+3 more"
    kids = sum(c["ms"] for c in same["children"])
    assert kids == pytest.approx(same["ms"])


def test_metrics_reports_coverage_and_dominant_share():
    metrics = _seeded().metrics()
    assert metrics["compute_coverage"] == pytest.approx(0.75)
    assert metrics["compute_samples"] == 1.0
    assert 0.0 < metrics["compute_top_share"] <= 1.0


def test_metric_descriptions_declare_a_renderer():
    entry = COMPUTE_METRIC_DESCRIPTIONS["compute_profile"]
    assert entry["snapshot"]["renderer"] == "compute_treemap"
    for key, spec in COMPUTE_METRIC_DESCRIPTIONS.items():
        assert spec.get("description"), f"{key} needs a description"


def test_defaults_are_sane():
    assert COMPUTE_DEFAULTS["interval"] >= 1
    assert 0.0 < COMPUTE_DEFAULTS["ema_alpha"] <= 1.0
    assert COMPUTE_DEFAULTS["warmup_steps"] >= 0


# ── live round trip (CPU) ───────────────────────────────────────────────────


def test_end_to_end_on_cpu_attributes_the_hot_module():
    """The wide Linear must outrank the norm; no negative entries anywhere."""
    model = Stack(d=64, n=2)
    model.blocks[0].up = nn.Linear(64, 2048)
    model.blocks[0].down = nn.Linear(2048, 64)
    prof = ComputeProfiler({"ema_alpha": 0.5})
    assert prof.install(model, device="cpu")

    x = torch.randn(8, 64)
    for _ in range(3):
        ctx = prof.start()
        model(x).sum().backward()
        model.zero_grad(set_to_none=True)
        prof.stop(ctx)
    prof.remove()

    assert prof.samples >= 1, "no sample landed"
    payload = prof.snapshot()["compute_profile"]
    assert sum(g["share"] for g in payload["groups"]) == pytest.approx(1.0)
    assert all(g["ms"] >= 0 for g in payload["groups"])
    assert all(c["ms"] >= 0 for g in payload["groups"] for c in g["children"])

    linear = next((g for g in payload["groups"] if g["name"] == "Linear"), None)
    norm = next((g for g in payload["groups"] if g["name"] == "LayerNorm"), None)
    assert linear is not None
    if norm is not None:
        assert linear["ms"] > norm["ms"]


def test_start_stop_is_safe_when_nothing_is_installed():
    prof = ComputeProfiler()
    ctx = prof.start()
    assert prof.stop(ctx) is False  # no hooks fired -> no sample
    assert prof.samples == 0


# ── thread isolation ────────────────────────────────────────────────────────


def test_a_foreign_thread_forward_is_ignored():
    """The dashboard generates from this same model on a request thread."""
    import threading

    model = Stack(d=32, n=2)
    prof = ComputeProfiler()
    prof.install(model, device="cpu")

    ctx = prof.start()
    model(torch.randn(4, 32)).sum().backward()  # owner thread: counted
    owner_fired = prof.hooks_fired()

    done = threading.Event()

    def intruder():
        with torch.no_grad():
            model(torch.randn(4, 32))
        done.set()

    t = threading.Thread(target=intruder)
    t.start()
    done.wait(timeout=10)
    t.join(timeout=10)

    assert prof.hooks_fired() == owner_fired, "foreign thread was counted"
    # and nothing of the intruder's was left dangling on the scope stack
    assert all(not stack for stack in prof._scopes.values())

    prof.stop(ctx)
    prof.remove()
    model.zero_grad(set_to_none=True)


def test_owner_is_reset_each_window():
    model = Stack(d=16, n=1)
    prof = ComputeProfiler()
    prof.install(model, device="cpu")
    ctx = prof.start()
    first = prof._owner
    prof.stop(ctx)
    ctx = prof.start()
    assert prof._owner == first  # same thread here, but re-stamped each start
    prof.stop(ctx)
    prof.remove()


# ── compiled mode: coarse scopes, forward only ──────────────────────────────


def test_max_depth_limits_which_modules_are_hooked():
    model = Stack(d=16, n=3)
    prof = ComputeProfiler()
    prof.install(model, device="cpu", max_depth=0)
    hooked = {
        m._praxis_scope.split("|")[0]
        for _, m in model.named_modules()
        if hasattr(m, "_praxis_scope")
    }
    assert hooked == {"blocks"}  # only the model's direct children
    prof.remove()


def test_max_depth_none_hooks_everything():
    model = Stack(d=16, n=2)
    prof = ComputeProfiler()
    prof.install(model, device="cpu", max_depth=None)
    n = sum(1 for _, m in model.named_modules() if hasattr(m, "_praxis_scope"))
    assert n == sum(1 for name, _ in model.named_modules() if name)
    prof.remove()


def test_install_unwraps_a_compiled_model():
    """Hooks must land on the inner module so scope names stay clean."""
    inner = Stack(d=16, n=1)
    wrapper = torch.nn.Module()
    wrapper._orig_mod = inner
    prof = ComputeProfiler()
    assert prof.install(wrapper, device="cpu")
    assert inner.blocks[0]._praxis_scope.startswith("blocks.0|")
    prof.remove()


def test_forward_only_excludes_backward_from_the_rollup():
    prof = ComputeProfiler()
    prof.forward_only = True
    prof._fold({"a|A": 6.0}, {"a|A": 40.0}, 50.0, 46.0)
    payload = prof.snapshot()["compute_profile"]
    assert payload["mode"] == "forward"
    alpha = next(g for g in payload["groups"] if g["name"] == "A")
    assert alpha["ms"] == pytest.approx(6.0)  # backward's 40 ms excluded


def test_forward_only_drops_the_unattributed_tile():
    """It would be mostly the backward we deliberately excluded, not a layer."""
    prof = ComputeProfiler()
    prof.forward_only = True
    prof._fold({"a|A": 6.0}, {"a|A": 40.0}, 50.0, 10.0)
    payload = prof.snapshot()["compute_profile"]
    assert not any(g["outside"] for g in payload["groups"])
    assert sum(g["share"] for g in payload["groups"]) == pytest.approx(1.0)
    # coverage is still reported so the header can be honest about it
    assert payload["coverage"] == pytest.approx(0.2)


def test_full_mode_keeps_the_unattributed_tile():
    prof = ComputeProfiler()
    prof._fold({"a|A": 6.0}, {"a|A": 4.0}, 20.0, 10.0)
    payload = prof.snapshot()["compute_profile"]
    assert payload["mode"] == "full"
    assert any(g["outside"] for g in payload["groups"])


def test_coarse_depth_on_a_container_only_model_fires_nothing():
    """A depth-0 child that is a bare ModuleList never executes.

    Not a failure mode to crash on: stop() must simply report no sample rather
    than fold an empty window into the EMA.
    """
    model = Stack(d=16, n=2)  # its only direct child is a ModuleList
    prof = ComputeProfiler()
    prof.install(model, device="cpu", max_depth=0)
    ctx = prof.start()
    model(torch.randn(2, 16)).sum().backward()
    model.zero_grad(set_to_none=True)
    assert prof.stop(ctx) is False
    assert prof.samples == 0
    assert prof.snapshot() == {}
    prof.remove()


def test_profiler_does_not_write_to_the_raw_file_descriptors():
    """libkineto logs from C++, below anything the terminal dashboard captures.

    A leaked `profiler_start` line lands on top of the rendered TUI and corrupts
    the frame, once per profiled step.
    """
    import os
    import sys

    model = Stack(d=32, n=2)
    prof = ComputeProfiler()
    prof.install(model, device="cpu")

    read_fd, write_fd = os.pipe()
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    try:
        for _ in range(2):
            ctx = prof.start()
            model(torch.randn(4, 32)).sum().backward()
            model.zero_grad(set_to_none=True)
            prof.stop(ctx)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(write_fd)
        os.close(saved_out)
        os.close(saved_err)
    leaked = os.read(read_fd, 200000).decode(errors="replace")
    os.close(read_fd)
    prof.remove()

    assert leaked.strip() == "", f"native output leaked: {leaked[:300]!r}"


def test_quiet_native_logs_restores_descriptors_on_error():
    from praxis.metrics.compute import _quiet_native_logs

    import os

    before = os.fstat(1).st_ino, os.fstat(2).st_ino
    with pytest.raises(RuntimeError):
        with _quiet_native_logs():
            raise RuntimeError("boom")
    after = os.fstat(1).st_ino, os.fstat(2).st_ino
    assert before == after, "stdout/stderr not restored after an exception"
