import pytest
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.memory import NeuralMemory, mem_state_detach
from praxis.memory.neural_memory import NeuralMemory, _affine_scan, decode_compiled
from praxis.modeling import PraxisForCausalLM

# ------------------------------------------------------------------------------
# memory
# ------------------------------------------------------------------------------
# Tests for the Titans NeuralMemory core and surfacings (praxis.memory).


def test_affine_scan_matches_naive():
    """The parallel scan equals a naive x_t = a_t*x_{t-1} + b_t recurrence."""
    torch.manual_seed(0)
    bsz, nc, p = 3, 7, (4, 5)
    a, b = torch.rand(bsz, nc), torch.randn(bsz, nc, *p)
    prev = torch.randn(bsz, *p)

    out = _affine_scan(a, b, prev)

    ref = torch.empty_like(b)
    x = prev
    for t in range(nc):
        x = a[:, t].reshape(bsz, 1, 1) * x + b[:, t]
        ref[:, t] = x
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.fixture
def mem():
    torch.manual_seed(0)
    # NeuralMemory takes any dim -> dim module; a plain MLP keeps the unit
    # tests decoupled from the dense registry.
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    return NeuralMemory(dim=64, model=model, chunk_size=8)


def test_shape_preserved(mem):
    """Retrieval returns the input shape, including a non-chunk-aligned length."""
    seq = torch.randn(2, 30, 64)  # 30 is not a multiple of chunk_size (8)
    out, state = mem(seq)
    assert out.shape == seq.shape
    assert state.seq_index == 30


def test_memorizes_at_test_time(mem):
    """The defining Titans property: storing a sequence lowers the memory's
    reconstruction loss on that sequence relative to the cold init weights."""
    seq = torch.randn(2, 64, 64)
    cold = mem.init_state(batch=2)
    _, warm = mem(seq)

    loss_cold = mem.memory_loss(seq, cold.weights)
    loss_warm = mem.memory_loss(seq, warm.weights)
    assert loss_warm < loss_cold


def test_state_threads_across_segments(mem):
    """State carries across segments. Gradients freeze at each segment's start
    weights (Titans semantics), so the first segment reproduces the matching
    prefix of a single pass exactly; later segments freeze at the carried
    weights and legitimately diverge."""
    seq = torch.randn(1, 32, 64)  # 4 chunks of size 8
    out_whole, _ = mem(seq)

    split = 16  # chunk boundary
    out_a, state_a = mem(seq[:, :split])
    out_b, _ = mem(seq[:, split:], state=state_a)

    # First segment matches the whole-run prefix (both freeze at W0).
    assert torch.allclose(out_a, out_whole[:, :split], atol=1e-4)

    out_split = torch.cat([out_a, out_b], dim=1)
    assert out_split.shape == out_whole.shape
    assert torch.isfinite(out_split).all()


def test_detach_breaks_graph(mem):
    """mem_state_detach yields state with no grad history (for truncated BPTT)."""
    seq = torch.randn(1, 16, 64)
    _, state = mem(seq)
    assert any(w.requires_grad for w in state.weights.values())
    detached = mem_state_detach(state)
    assert all(not w.requires_grad for w in detached.weights.values())


def test_meta_params_receive_gradient(mem):
    """An outer loss backpropagates into the memory's meta-learned params,
    confirming the test-time update is differentiable end-to-end."""
    seq = torch.randn(1, 16, 64)
    out, _ = mem(seq)
    out.sum().backward()
    grads = [p.grad for p in mem.memory_model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_standard_mode_trains_store_projections(mem):
    """In the default mode the differentiable update gives the store-side
    projections a gradient (contrast with energy mode below)."""
    out, _ = mem(torch.randn(2, 16, 64))
    out.sum().backward()
    assert mem.to_keys.weight.grad is not None


# --- energy (detached) mode -------------------------------------------------


def _energy_mem():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    return NeuralMemory(dim=64, model=model, chunk_size=8, use_energy=True)


def test_energy_mode_ties_keys_and_fixes_values():
    """Energy mode ties the key projection to the query projection (and the
    store norm to the retrieve norm) and fixes the value side to identity, so
    addressing learns on the task while the encoder can't collapse the energy."""
    mem = _energy_mem()
    assert mem.to_keys is mem.to_queries
    assert mem.store_norm is mem.retrieve_norm
    assert isinstance(mem.to_values, nn.Identity)


def test_energy_mode_has_no_frozen_params():
    """Every parameter in energy mode receives a gradient (nothing is left
    untrained), so the module is DDP-clean."""
    mem = _energy_mem()
    out, _ = mem(torch.randn(2, 16, 64))
    out.sum().backward()
    missing = [n for n, p in mem.named_parameters() if p.grad is None]
    assert not missing, missing


def test_energy_mode_has_no_learned_gate_heads():
    """Energy mode replaces the learned lr/momentum/decay gates with the
    Adam-style rule, so it carries no untrained gate heads."""
    mem = _energy_mem()
    assert not hasattr(mem, "to_lr")
    assert not hasattr(mem, "to_momentum")
    assert not hasattr(mem, "to_decay")


def test_energy_mode_still_memorizes():
    """The detached update still adapts the fast weights at test time."""
    mem = _energy_mem()
    seq = torch.randn(2, 64, 64)
    cold = mem.init_state(batch=2)
    _, warm = mem(seq)
    assert mem.memory_loss(seq, warm.weights) < mem.memory_loss(seq, cold.weights)


def test_reports_gain_and_write():
    """A store pass records the gain (output vs stream) and write (relative
    weight update) diagnostics, with a positive write (the update did work)."""
    mem = _energy_mem()
    mem(torch.randn(2, 32, 64))
    assert mem.last_gain is not None and torch.isfinite(mem.last_gain)
    assert mem.last_write is not None and torch.isfinite(mem.last_write)
    assert mem.last_write > 0


def test_reports_readout_delta():
    """The readout probe records what the write changed in FUNCTION space:
    finite, positive, and independent of the weight-space ratio."""
    mem = _energy_mem()
    mem(torch.randn(2, 32, 64))
    assert mem.last_adapt is not None and torch.isfinite(mem.last_adapt)
    assert mem.last_adapt > 0


@pytest.mark.parametrize("scale", [3.0, 32.0, 185.0])
def test_write_strength_is_invariant_to_the_weight_scale(scale):
    """The test-time step is RELATIVE to ||W0||, so growing the meta-learned
    weights by k leaves the write ratio where it was.

    Before this, the step was a fixed absolute max_lr while W0 was a trained
    parameter free to grow - and it does grow, because the readout sits behind
    out_norm (exactly scale-invariant), so nothing constrains the memory net's
    output magnitude. abstractinator-x drifted to ~185x over 19k steps, and the
    write ratio fell 14x with memory_adapt following it 78x down to 0.010: the
    module ended up a static nonlinearity the gate still wanted but that no
    longer learned in context. This is the invariance that failure needed."""
    seq = torch.randn(2, 32, 64)
    small = _energy_mem()
    small(seq)
    big = _energy_mem()
    with torch.no_grad():
        for param in big.memory_model.parameters():
            param.mul_(scale)
    big(seq)

    assert float(big.last_write) == pytest.approx(float(small.last_write), rel=0.15)
    # And the mechanism the write drives keeps working, rather than decaying
    # like 1/k - the old rule reached adapt 0.0087 at this scale.
    assert float(big.last_adapt) > 0.5 * float(small.last_adapt)


@pytest.mark.parametrize("segment_block,chunks", [(32, 2), (16, 4), (8, 8), (4, 16)])
def test_update_grid_is_not_a_hidden_learning_rate(segment_block, chunks):
    """Write strength must not depend on how finely the pass is chunked.

    ``u`` is sign-like and roughly decorrelated across chunks, so a pass's total
    write used to accumulate as ``max_lr * sqrt(nc)`` - 0.0148 at 2 chunks
    rising to 0.0649 at 32, a 4.4x swing from a knob nobody declared as a
    learning rate. Taking abstractinator-x's segment_block from 16 to 4 silently
    multiplied its effective step by 1.41 on exactly that mechanism. ``max_lr``
    now means the total relative write per pass, and the grid sets granularity
    only."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
    mem = NeuralMemory(
        dim=64,
        model=model,
        chunk_size=64,
        segment_block=segment_block,
        max_lr=0.01,
        use_energy=True,
        segment=True,
        write_objective="predictive",
    )
    mem.train()
    mem(torch.randn(4, 64, 64))
    assert mem.last_num_chunks == chunks
    assert float(mem.last_write) == pytest.approx(0.01, rel=0.25)


def test_readout_delta_is_not_a_restatement_of_the_write_ratio():
    """The two metrics still measure different things, and the grid separates
    them cleanly now that write strength is grid-invariant. Write is a
    weight-space number pinned to ``max_lr`` per pass however the pass is
    chunked; the readout delta is a function-space one that still rises with the
    number of VISIBLE writes, because retrieval reads pre-write weights and only
    ``chunks - 1`` of them ever reach the trunk."""
    seq = torch.randn(2, 64, 64)
    few = _energy_mem()
    few(seq[:, :16])  # 2 chunks -> 1 visible write
    many = _energy_mem()
    many(seq)  # 8 chunks -> 7 visible writes

    # Write is held flat by the grid normalization...
    assert float(many.last_write) == pytest.approx(float(few.last_write), rel=0.3)
    # ...while the readout still feels the extra visible writes.
    assert float(many.last_adapt) > 1.3 * float(few.last_adapt)


def test_readout_delta_matches_sequential_path():
    """The probe reports the same value from the sequential loop as from the
    parallel scan (it rides both paths, not just the fast one)."""
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32))
    mem = NeuralMemory(dim=32, model=model, chunk_size=32, use_energy=True)
    seq = torch.randn(2, 96, 32)

    mem.parallel_scan, mem._probe_tick = True, -1
    mem(seq, mem.init_state(2))
    parallel = float(mem.last_adapt)
    mem.parallel_scan, mem._probe_tick = False, -1
    mem(seq, mem.init_state(2))
    assert float(mem.last_adapt) == pytest.approx(parallel, rel=1e-4)


def test_readout_delta_runs_on_a_cadence():
    """The extra forward is gated: one call in PROBE_EVERY while training, and
    never in eval, so the cost rides the logging cadence rather than every
    forward through the module."""
    mem = _energy_mem()
    mem.PROBE_EVERY = 2
    seq = torch.randn(2, 32, 64)

    mem(seq)  # tick 0 -> probes
    assert mem.last_adapt is not None
    mem.last_adapt = None
    mem(seq)  # tick 1 -> skipped, value goes stale rather than wrong
    assert mem.last_adapt is None
    mem(seq)  # tick 2 -> probes
    assert mem.last_adapt is not None

    mem.eval()
    mem.last_adapt = None
    for _ in range(4):
        mem(seq)
    assert mem.last_adapt is None


def test_energy_surprise_is_scale_free():
    """The normalized surprise is bounded/O(1) even when the memory net's
    output scale is large, where the raw surprise blows up. This is the fix for
    the runaway raw surprise: the update optimizes the scale-free quantity."""
    mem = _energy_mem()
    # Blow up the memory net's output scale, mimicking trained scale drift.
    with torch.no_grad():
        for p in mem.memory_model.parameters():
            p.mul_(50.0)
    mem(torch.randn(2, 32, 64))
    assert mem.last_surprise_norm is not None
    # Normalized surprise stays small; raw is dominated by the inflated scale.
    assert mem.last_surprise_norm < 10.0
    assert mem.last_surprise > 100.0 * mem.last_surprise_norm


# --- surprise-based segmentation (EM-LLM) -----------------------------------


def _segment_mem():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    return NeuralMemory(
        dim=64,
        model=model,
        chunk_size=64,
        use_energy=True,
        segment=True,
        segment_block=16,
    )


def test_segment_requires_energy():
    """Segmentation only takes effect in energy mode; off otherwise."""
    model = nn.Sequential(nn.Linear(64, 64))
    mem = NeuralMemory(dim=64, model=model, segment=True, use_energy=False)
    assert mem.segment is False


def test_segment_cap_without_spikes():
    """A uniform stream has no surprise spikes, so events are forced only at the
    chunk_size cap: every event is exactly chunk_size tokens."""
    mem = _segment_mem()
    pattern = torch.randn(2, 1, 64)
    seq = pattern.repeat(1, 128, 1)  # 128 = 2 * chunk_size, no variation
    mem(seq)
    assert float(mem.last_event_max) == 64.0
    assert float(mem.last_event_mean) == 64.0


def test_segment_helper_boundaries():
    """A surprise spike forces an event boundary; the cap forces one regardless;
    the per-event position resets at each boundary."""
    mem = _segment_mem()  # cap = 64 / 16 = 4 blocks
    s = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0]])
    reset, t_event = mem._segment(s)
    # Boundaries: block 0 (forced start), block 4 (cap), block 5 (spike).
    assert reset[0].tolist() == [1, 0, 0, 0, 1, 1, 0, 0]
    assert t_event[0].tolist() == [1, 2, 3, 4, 1, 1, 2, 3]


def test_segment_events_bounded_and_surfaced():
    """Event sizes are reported at grid granularity, so they stay bounded by
    [segment_block, chunk_size] even when the sequence is not block-aligned
    (200 % 16 != 0): the padded trailing block never reports below one block."""
    torch.manual_seed(1)
    mem = _segment_mem()
    seq = torch.randn(2, 200, 64)  # not a multiple of segment_block (16)
    seq[:, 100:] += 8.0  # context shift -> surprise spike
    mem(seq)
    assert mem.last_event_mean is not None
    assert float(mem.last_event_max) <= 64.0
    assert float(mem.last_event_min) >= 16.0


def test_segment_still_memorizes():
    """Segmented updates still adapt the fast weights at test time."""
    mem = _segment_mem()
    seq = torch.randn(2, 128, 64)
    cold = mem.init_state(batch=2)
    _, warm = mem(seq)
    assert mem.memory_loss(seq, warm.weights) < mem.memory_loss(seq, cold.weights)


# --- pad handling and the chunk-count floor ---------------------------------


def _pad_mem(**kw):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    base = dict(
        dim=64,
        model=model,
        chunk_size=64,
        use_energy=True,
        segment=True,
        segment_block=16,
    )
    base.update(kw)
    return NeuralMemory(**base)


@pytest.mark.parametrize("objective", ["recon", "predictive"])
@pytest.mark.parametrize("n", [17, 33, 47])
def test_pad_does_not_enter_the_write(monkeypatch, objective, n):
    """A tail pad must be inert, and the sharpest statement of that is that its
    CONTENTS cannot matter. A zero pad is not self-evidently a no-op: the store
    path RMS-normalizes, which maps the zero vector to itself, so the surprise
    reads a full-magnitude "predict nothing from nothing" error and the update
    chases it. Filling the pad with a large constant instead must therefore leave
    the fast weights and the reported surprise bit-identical."""
    import praxis.memory.neural_memory as nm

    seq = torch.randn(2, n, 64)

    def run(fill):
        real_pad = torch.nn.functional.pad
        if fill is not None:
            monkeypatch.setattr(
                nm.F, "pad", lambda t, p, **kw: real_pad(t, p, value=fill)
            )
        else:
            monkeypatch.setattr(nm.F, "pad", real_pad)
        mem = _pad_mem(write_objective=objective)
        _, st = mem(seq)
        return st, float(mem.last_surprise_norm), float(mem.last_surprise)

    zero_pad, s_norm_z, s_raw_z = run(None)
    junk_pad, s_norm_j, s_raw_j = run(7.5)

    for k in zero_pad.weights:
        assert torch.equal(
            zero_pad.weights[k], junk_pad.weights[k]
        ), f"pad contents changed the fast weights ({k})"
    assert s_norm_z == s_norm_j, "pad contents changed the reported surprise"
    assert s_raw_z == s_raw_j


def test_predictive_target_does_not_shift_off_the_end():
    """The last REAL token has no successor and must target itself. Shifting the
    padded tensor would hand it a zero pad, training the memory to forecast
    nothing at every sequence end."""
    mem = _pad_mem(write_objective="predictive")
    stored = torch.randn(2, 48, 64)
    n = 33  # 15 pad positions follow
    tgt = mem._shift_targets(stored, n)
    assert tgt.shape == stored.shape
    assert torch.equal(tgt[:, : n - 1], stored[:, 1:n])  # interior: next latent
    assert torch.equal(tgt[:, n - 1], stored[:, n - 1])  # last real: itself
    # With no pad the behaviour is unchanged (last token still targets itself).
    full = mem._shift_targets(stored, stored.shape[1])
    assert torch.equal(full[:, -1], stored[:, -1])


@pytest.mark.parametrize("n,expected", [(8, 1), (16, 1), (32, 2), (33, 3), (128, 8)])
def test_chunk_count_is_reported(n, expected):
    """``memory_chunks`` is the ceiling on adaptation, so it is surfaced rather
    than left to be inferred from the sequence length."""
    mem = _pad_mem()
    mem(torch.randn(2, n, 64))
    assert mem.last_num_chunks == expected


def test_single_chunk_cannot_adapt():
    """Retrieval reads PRE-write weights, so at one chunk the readout is the
    cold one and the update is discarded - adapt is exactly 0 while gain and
    write still look healthy. This is the failure mode that made a memory read
    as a static MLP; it must stay visible."""
    mem = _pad_mem()
    mem.train()
    mem(torch.randn(2, 16, 64))  # 16 tokens on a 16-token grid -> 1 chunk
    assert mem.last_num_chunks == 1
    assert float(mem.last_adapt) == 0.0
    assert float(mem.last_write) > 0.0  # the update happened, it is just unread

    mem2 = _pad_mem()
    mem2.train()
    mem2(torch.randn(2, 64, 64))  # 4 chunks -> 3 visible writes
    assert float(mem2.last_adapt) > 0.0


# --- write gating -----------------------------------------------------------


def _gate_mem(gate, **kw):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32))
    base = dict(
        dim=32,
        model=model,
        chunk_size=16,
        segment_block=16,
        use_energy=True,
        write_objective="predictive",
        write_gate=gate,
    )
    base.update(kw)
    return NeuralMemory(**base)


def test_write_gate_is_off_by_default():
    """The gate is opt-in: an unconfigured memory writes every real token, and
    emits none of the gate metrics."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32))
    mem = NeuralMemory(dim=32, model=model, chunk_size=16, use_energy=True)
    assert mem.write_gate == "none"
    mem(torch.randn(2, 64, 32))
    assert mem.last_write_share is None
    assert mem.last_write_selectivity is None
    assert mem.last_write_tilt is None


@pytest.mark.parametrize("capacity,expected_k", [(0.125, 2), (0.25, 4), (0.5, 8)])
def test_topk_gate_writes_exactly_its_capacity(capacity, expected_k):
    """Expert-choice capacity is exact, so the sparse arm is never degenerate
    and its write share is a constant rather than a measurement."""
    mem = _gate_mem("topk", write_capacity=capacity)
    mem(torch.randn(2, 128, 32))  # 8 chunks of 16, no pad
    assert float(mem.last_write_share) == pytest.approx(expected_k / 16)


def test_topk_gate_keeps_the_highest_scoring_tokens():
    """The mask IS the top-k of the score, per chunk - the router is the
    surprise, not a learned head."""
    scores = torch.tensor([[[0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]]])
    ones = torch.ones_like(scores)
    # 8 positions at capacity 0.25 -> k=2: the 0.9 and the 0.8.
    mask, _ = _gate_mem("topk", write_capacity=0.25)._write_gate_mask(scores, ones)
    assert mask.flatten().tolist() == [0, 1, 0, 0, 0, 1, 0, 0]
    # Capacity rounds UP, so a fraction that lands between token counts still
    # writes rather than dropping the chunk: 8 * 0.3 -> k=3.
    mask, _ = _gate_mem("topk", write_capacity=0.3)._write_gate_mask(scores, ones)
    assert mask.flatten().tolist() == [0, 1, 0, 1, 0, 1, 0, 0]
    # The pad is never picked, even when it outranks every real token.
    valid = torch.tensor([[[1.0, 1, 1, 1, 0, 0, 0, 0]]])
    high = torch.tensor([[[0.1, 0.9, 0.3, 0.2, 9.0, 9.0, 9.0, 9.0]]])
    mask, _ = _gate_mem("topk", write_capacity=0.25)._write_gate_mask(high, valid)
    assert mask.flatten().tolist() == [0, 1, 1, 0, 0, 0, 0, 0]


def _repeat_shares(tilted, passes=30):
    """Write shares from re-presenting one stream to a memory that keeps its
    state, with the cross-pass tilt either live or pinned at 1.0."""
    mem = _gate_mem("threshold")
    if not tilted:
        mem._gate_tilt = lambda: torch.ones(1)
    torch.manual_seed(100)
    seq = torch.randn(1, 128, 32)
    state = mem.init_state(1)
    out = []
    for _ in range(passes):
        _, state = mem(seq, state)  # same stream, threaded state
        out.append(float(mem.last_write_share))
    return out


def test_threshold_gate_closes_as_the_memory_learns():
    """The property the gate exists for: re-presenting a stream the memory
    already forecasts must write LESS of it each time. Nothing in the dense
    write can express this - it writes every token at a fixed step forever.

    Not asserted monotone. The bar rides a causal running mean that moves with
    the data, so the share wanders within a pass-to-pass band; the claim is the
    trend, and a monotonicity assertion here would be pinning noise.
    """
    shares = _repeat_shares(tilted=True)
    head = sum(shares[:10]) / 10
    tail = sum(shares[-10:]) / 10
    assert tail < head * 0.95, shares


def test_only_the_cross_pass_tilt_can_close_the_gate():
    """Why the bar carries state across passes at all. A purely within-sequence
    bar is scale-invariant - halve every surprise and the running mean halves
    with it, so the same fraction still clears - which means it can say which
    tokens here are worth writing but never that none of them are. Pinning the
    tilt at 1.0 reduces the gate to exactly that, and the decline disappears.
    """

    def tail(x):
        return sum(x[-10:]) / 10

    tilted, flat = _repeat_shares(True), _repeat_shares(False)
    assert tail(tilted) < tail(flat), (tilted, flat)
    assert tail(flat) >= sum(flat[:10]) / 10 * 0.98, flat


def test_the_tilt_reads_above_one_while_the_memory_improves():
    """The tilt is the gate's motive, reported apart from its share: a slow
    surprise EMA over a fast one, so it lifts off 1.0 exactly when the fast EMA
    is leading the slow one down."""
    mem = _gate_mem("threshold")
    torch.manual_seed(100)
    seq = torch.randn(1, 128, 32)
    state = mem.init_state(1)
    _, state = mem(seq, state)
    assert float(mem.last_write_tilt) == pytest.approx(1.0)  # both EMAs seeded
    for _ in range(7):
        _, state = mem(seq, state)
    assert float(mem.last_write_tilt) > 1.0


def test_eval_does_not_move_the_gate_reference():
    """The bar in force at eval is the one training left, not one eval moved for
    itself - otherwise validation would re-centre the gate on its own data."""
    mem = _gate_mem("threshold")
    seq = torch.randn(2, 64, 32)
    mem(seq)
    before = mem._gate_ref.clone()
    mem.eval()
    mem(torch.randn(2, 64, 32) * 10.0)
    assert torch.equal(before, mem._gate_ref)


def test_the_gate_reference_survives_a_checkpoint():
    """It is state the run accumulates, so it has to round-trip: a reloaded
    model that forgot its bar would re-open the gate and write everything."""
    mem = _gate_mem("threshold")
    mem(torch.randn(2, 64, 32))
    fresh = _gate_mem("threshold")
    assert not torch.equal(fresh._gate_ref, mem._gate_ref)
    fresh.load_state_dict(mem.state_dict())
    assert torch.equal(fresh._gate_ref, mem._gate_ref)
    assert torch.equal(fresh._gate_ref_ready, mem._gate_ref_ready)


@pytest.mark.parametrize("parallel", [True, False])
def test_a_fully_gated_pass_holds_the_weights_exactly(parallel):
    """The opt-out has to be a real zero, not a small step. With no token
    cleared to write, the surprise gradient is exactly zero, weight_decay is 0
    and the Adam moments only decay - so the state must come back bit-identical
    on BOTH paths."""
    mem = _gate_mem("threshold", parallel_scan=parallel)
    mem._write_gate_mask = lambda sc, v, prior=None: (torch.zeros_like(sc), prior)
    seq = torch.randn(2, 64, 32)
    cold = mem.init_state(2)
    _, warm = mem(seq, cold)
    for k in cold.weights:
        assert torch.equal(warm.weights[k], cold.weights[k]), k
        assert torch.equal(warm.momentum[k], cold.momentum[k]), k
    assert float(mem.last_write) == 0.0


@pytest.mark.parametrize("gate", ["topk", "threshold"])
def test_the_gate_never_lets_the_pad_write(monkeypatch, gate):
    """Same statement as the ungated pad test, on the gate: the tail pad's
    CONTENTS cannot reach the fast weights. A gate scored on surprise is the way
    a junk pad would get in - it is the most surprising thing in the sequence.
    """
    import praxis.memory.neural_memory as nm

    seq = torch.randn(2, 47, 64)

    def run(fill):
        real_pad = torch.nn.functional.pad
        monkeypatch.setattr(
            nm.F,
            "pad",
            (
                real_pad
                if fill is None
                else (lambda t, p, **kw: real_pad(t, p, value=fill))
            ),
        )
        mem = _pad_mem(write_objective="predictive", write_gate=gate)
        _, st = mem(seq)
        return st, float(mem.last_write_share)

    zero_pad, share_z = run(None)
    junk_pad, share_j = run(7.5)
    for k in zero_pad.weights:
        assert torch.equal(zero_pad.weights[k], junk_pad.weights[k]), k
    assert share_z == share_j


@pytest.mark.parametrize("gate", ["topk", "threshold"])
def test_the_gate_selects_on_content(gate):
    """The null this whole arm has to beat: a gate that kept tokens at random
    would report a selectivity of 1.0. Kept tokens must be measurably more
    surprising than the average one."""
    mem = _gate_mem(gate)
    mem(torch.randn(2, 128, 32))
    assert float(mem.last_write_selectivity) > 1.0


@pytest.mark.parametrize("gate", ["topk", "threshold"])
def test_gating_still_memorizes_at_test_time(gate):
    """The defining Titans property has to survive the gate: storing a sequence
    still has to lower the memory's loss on it, from a fraction of the writes.
    """
    mem = _gate_mem(gate)
    seq = torch.randn(2, 128, 32)
    cold = mem.init_state(2)
    _, warm = mem(seq, cold)

    def loss(weights):
        stored = mem.store_norm(seq)
        keys = mem.to_keys(stored)
        target = torch.cat([stored[:, 1:], stored[:, -1:]], dim=1)
        pred = vmap(lambda w, k: functional_call(mem.memory_model, w, (k,)))(
            weights, keys
        )
        return float(mem._recon_per_token(pred, target, True).mean().detach())

    assert loss(warm.weights) < loss(cold.weights)
    assert float(mem.last_write_share) < 1.0


def _converge(mem, passes=12, b=2, n=128, seed=7):
    """Run the gate on fresh data and return the trailing share. A causal rank
    has no convergence period, so a dozen passes carry the claim."""
    torch.manual_seed(seed)
    shares = []
    for _ in range(passes):
        mem(torch.randn(b, n, 32), mem.init_state(b))
        shares.append(float(mem.last_write_share))
    return sum(shares[-10:]) / 10


@pytest.mark.parametrize("capacity", [0.125, 0.25, 0.5])
def test_adaptive_share_lands_on_its_target(capacity):
    """A causal rank writes the top ``target`` fraction of the prefix, so the
    realized share is the target with no convergence period to wait through."""
    mem = _gate_mem("adaptive", write_capacity=capacity)
    assert _converge(mem) == pytest.approx(capacity, abs=0.03)
    assert float(mem.last_write_target) == pytest.approx(capacity, abs=0.01)


@pytest.mark.parametrize("transform", ["scale", "shift", "square"])
def test_adaptive_gate_ignores_the_score_distribution(transform):
    """Why a rank and not a level. Every level-based bar depends on where the
    distribution sits and how wide it is, and both move pass to pass by about as
    much as each other. A rank depends on the ORDER only, so any monotone
    transform of the score must leave the mask bit-identical."""
    fns = {
        "scale": lambda x: x * 100.0,
        "shift": lambda x: x + 5.0,
        "square": lambda x: x**2,
    }
    out = {}
    for label, f in (("raw", lambda x: x), (transform, fns[transform])):
        mem = _gate_mem("adaptive")
        base = mem._write_scores
        mem._write_scores = lambda w, k, v, g=f: g(base(w, k, v))
        out[label] = _converge(mem)
    assert out["raw"] == pytest.approx(out[transform], abs=1e-9)


def test_adaptive_share_is_quiet_under_a_drifting_score():
    """The failure both level-based shapes had. When the score's location moves
    pass to pass by about as much as the distribution's own width - the regime
    measured on the real model - a carried bar sits somewhere different inside
    the distribution each time and the share swings end to end. A rank cannot.
    """
    mem = _gate_mem("adaptive")
    torch.manual_seed(5)
    shares = []
    for _ in range(40):
        loc = 1.0 + 0.10 * torch.randn(1).item()
        base = mem._write_scores
        mem._write_scores = lambda w, k, v, s=loc: base(w, k, v) * s
        mem(torch.randn(4, 128, 32), mem.init_state(4))
        shares.append(float(mem.last_write_share))
        mem._write_scores = base
    tail = shares[-20:]
    target = float(mem.last_write_target)
    assert max(tail) - min(tail) < 0.15, (target, tail)
    assert sum(tail) / len(tail) == pytest.approx(target, abs=0.05), (target, tail)


def test_adaptive_target_closes_at_the_tilt_ceiling():
    """The opt-out has to stay reachable. ``2 - tilt`` puts the closed point at
    the tilt clamp, so a memory whose fast surprise EMA has halved against its
    slow one targets no writes at all."""
    mem = _gate_mem("adaptive")
    mem._gate_ref_ready.fill_(1.0)
    mem._gate_ref.copy_(torch.tensor([1.0, 1.0]))
    assert float(mem._gate_target()) == pytest.approx(mem.write_capacity)
    mem._gate_ref.copy_(torch.tensor([1.0, 0.4]))  # fast well below slow
    assert float(mem._gate_tilt()) == pytest.approx(2.0)  # at the clamp
    assert float(mem._gate_target()) == 0.0
    mem._gate_ref.copy_(torch.tensor([1.0, 2.0]))  # losing ground
    assert float(mem._gate_target()) > mem.write_capacity


def test_a_causal_prefix_rank_writes_early_under_a_falling_score():
    """The one property a causal-prefix rank has that a level bar does not,
    stated so it is not a surprise later.

    Ranking token t against tokens up to t is what keeps the gate legal - a
    chunk's write is only read by later chunks - but it also means that if the
    score TRENDS DOWN along a sequence, later tokens can never reach the top
    fraction of their own prefix, and the realized share falls below the target.

    That is by construction and not a defect: a falling surprise IS the memory
    learning as it goes. It matters only if the trend is real, and on the
    trained abstractinator-v checkpoint it is not - the per-quarter score means
    are 0.6115 / 0.6131 / 0.6104 / 0.6147, flat to 0.5%. If a future change
    introduces a trend, the share will read below the target and this is why.
    """
    mem = _gate_mem("adaptive", write_capacity=0.25)
    n = 128
    # A score that falls monotonically across the sequence, nothing else.
    ramp = torch.linspace(2.0, 1.0, n).reshape(1, 1, n)
    mask, _ = mem._write_gate_mask(ramp, torch.ones_like(ramp))
    kept = mask.flatten().nonzero().flatten()
    assert float(mask.mean()) < 0.25, "a falling score must undershoot the target"
    assert kept.max() < n // 2, "everything written must come from the prefix"

    # Flat-in-expectation scores hit the target instead.
    torch.manual_seed(0)
    flat = torch.rand(1, 1, n) + 1.0
    mask, _ = mem._write_gate_mask(flat, torch.ones_like(flat))
    assert float(mask.mean()) == pytest.approx(0.25, abs=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(use_energy=True, segment=True),  # mal_energy (the default profile)
        dict(use_energy=True, segment=False),
        dict(use_energy=False, momentum=True),  # standard, differentiable update
        dict(use_energy=False, momentum=False),
        dict(use_energy=True, segment=True, write_gate="topk"),
        dict(use_energy=True, segment=True, write_gate="threshold"),
        dict(use_energy=True, segment=True, write_gate="adaptive"),
    ],
)
def test_sequential_matches_parallel_scan(kwargs):
    """The sequential loop must reproduce the parallel-scan path exactly (same
    output and carried state), so ``parallel_scan`` is purely a perf knob."""
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32))
    mem = NeuralMemory(dim=32, model=model, chunk_size=32, segment_block=8, **kwargs)
    # 100 is NOT block-aligned (100 % 8 != 0), so this also pins the two paths
    # to the same pad masking and the same event-boundary threshold - the
    # sequential path carried a relative tolerance the parallel one lacked.
    seq = torch.randn(2, 100, 32)

    # The threshold gate's bar is an EMA the store pass folds into, so the two
    # runs have to START from the same reference or the second one is gated
    # against a bar the first one moved. Restoring it is what keeps this a test
    # of the two paths rather than of the reference.
    ref = (mem._gate_ref.clone(), mem._gate_ref_ready.clone())

    def run(parallel):
        mem._gate_ref.copy_(ref[0])
        mem._gate_ref_ready.copy_(ref[1])
        mem.parallel_scan = parallel
        return mem(seq, mem.init_state(2))

    (out_p, st_p), (out_s, st_s) = run(True), run(False)
    assert torch.allclose(out_p, out_s, atol=1e-4)
    for field in ("weights", "momentum", "second_moment"):
        for k in getattr(st_p, field):
            assert torch.allclose(
                getattr(st_p, field)[k], getattr(st_s, field)[k], atol=1e-4
            )


# --- standard (backprop) mode -----------------------------------------------


def _standard_mem(chunk_size=4):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
    return NeuralMemory(dim=64, model=model, chunk_size=chunk_size, use_energy=False)


@pytest.mark.parametrize("n,chunks", [(8, 2), (16, 4), (32, 8), (64, 16)])
def test_forgetting_gate_starts_at_retain(n, chunks):
    """alpha_t (paper Eq. 13) compounds ONCE PER CHUNK, so its init decides
    whether a store pass writes to the meta-learned memory or erases it. A
    default Linear init sits at sigmoid(0) = 0.5 - halfway to the paper's "clear
    the entire memory" - which left 0.06% of W0 alive after 16 chunks, and got
    worse the more chunks the memory was given."""
    mem = _standard_mem()
    _, st = mem(torch.randn(2, n, 64))
    assert mem.last_num_chunks == chunks
    w0 = {k: v.detach() for k, v in mem._init_weights(2).items()}
    retained = min(float(st.weights[k].detach().norm() / w0[k].norm()) for k in w0)
    assert (
        retained > 0.5
    ), f"{chunks} chunks erased the memory: ||W_T||/||W0||={retained}"


def test_forgetting_gate_can_still_forget():
    """Retention at init must be a starting point, not a wall: the gate is
    learnable and driving it positive still clears the memory."""
    mem = _standard_mem()
    with torch.no_grad():
        mem.to_decay.bias.fill_(5.0)  # alpha -> 1, "clear the entire memory"
    _, st = mem(torch.randn(2, 32, 64))
    w0 = {k: v.detach() for k, v in mem._init_weights(2).items()}
    retained = max(float(st.weights[k].detach().norm() / w0[k].norm()) for k in w0)
    assert retained < 0.1


def test_standard_mode_reports_the_scale_free_surprise():
    """Standard mode optimizes the paper's raw MSE, but its readout is behind
    out_norm just like energy mode's - so the output magnitude is a free mode and
    the scale-free line has to be readable, or a drifting surprise cannot be told
    apart from a memory that stopped learning."""
    mem = _standard_mem()
    seq = torch.randn(2, 32, 64)
    mem(seq)
    raw0, norm0 = float(mem.last_surprise), float(mem.last_surprise_norm)
    with torch.no_grad():  # simulate the memory net's weights growing
        for p in mem.memory_model.parameters():
            p.mul_(50.0)
    mem(seq)
    raw1, norm1 = float(mem.last_surprise), float(mem.last_surprise_norm)
    assert raw1 / raw0 > 100.0  # the raw line is scale-sensitive, as documented
    assert norm1 == pytest.approx(norm0, rel=0.05)  # the scale-free one is not


# ------------------------------------------------------------------------------
# decode_compile
# ------------------------------------------------------------------------------
# Decode-time compilation of NeuralMemory: plumbing, scoping, and fallback.
#
# The measured payoff (1.7x on a 128-byte generation, byte-identical output) needs a GPU
# and several minutes of Inductor, so it is not asserted here. What IS asserted is
# everything that could silently break it or, worse, leak a compiled body into training:
# installation, dispatch, restoration, the ``no_compile`` gate, and degrading to eager
# when compilation raises.
#
# ``torch.compile`` is stubbed throughout - compiling for real would make this test
# minutes long and would test Inductor rather than this wiring.


def build_memory_model(**overrides):
    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=200,
        hidden_size=64,
        embed_size=64,
        depth=2,
        num_layers=2,
        num_heads=4,
        device="cpu",
        block_type="transformer",
        max_position_embeddings=256,
        attention_type="causal",
        encoding="rope",
        memory_type="mal_energy",
        **overrides,
    )
    return PraxisForCausalLM(cfg).eval()


def neural_memories(model):
    return [m for m in model.modules() if isinstance(m, NeuralMemory)]


@pytest.fixture
def stub_compile(monkeypatch):
    """Replace torch.compile with a counting pass-through."""
    calls = {"n": 0}

    def fake(fn, **kwargs):
        calls["n"] += 1

        def wrapper(*args, **kw):
            calls.setdefault("invoked", 0)
            calls["invoked"] += 1
            return fn(*args, **kw)

        return wrapper

    monkeypatch.setattr(torch, "compile", fake)
    return calls


def test_model_has_neural_memory():
    """Guard the fixture itself: a config that quietly built the no-op memory
    would make every other test in this file vacuously pass."""
    assert neural_memories(build_memory_model())


def test_installs_inside_and_restores_outside(stub_compile):
    model = build_memory_model()
    mems = neural_memories(model)
    assert all(m._decode_forward is None for m in mems)

    with decode_compiled(model):
        assert all(m._decode_forward is not None for m in mems)

    assert all(m._decode_forward is None for m in mems)
    assert stub_compile["n"] == len(mems)


def test_compiled_body_is_cached_across_windows(stub_compile):
    """Only the first generation of a run may pay for Inductor."""
    model = build_memory_model()
    for _ in range(3):
        with decode_compiled(model):
            pass
    assert stub_compile["n"] == len(neural_memories(model))


def test_restores_when_the_body_raises(stub_compile):
    model = build_memory_model()
    with pytest.raises(RuntimeError):
        with decode_compiled(model):
            raise RuntimeError("boom")
    assert all(m._decode_forward is None for m in neural_memories(model))


def test_disabled_is_a_no_op(stub_compile):
    model = build_memory_model()
    with decode_compiled(model, enabled=False):
        assert all(m._decode_forward is None for m in neural_memories(model))
    assert stub_compile["n"] == 0


def test_compile_failure_degrades_to_eager(monkeypatch):
    def explode(fn, **kwargs):
        raise RuntimeError("inductor said no")

    monkeypatch.setattr(torch, "compile", explode)
    model = build_memory_model()
    ids = torch.randint(0, 200, (1, 8))
    with decode_compiled(model):
        assert all(m._decode_forward is None for m in neural_memories(model))
        with torch.no_grad():
            model(input_ids=ids)  # still runs, on the eager body


def test_forward_dispatches_to_the_installed_body(stub_compile):
    model = build_memory_model()
    ids = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        with decode_compiled(model):
            model(input_ids=ids)
    assert stub_compile.get("invoked", 0) > 0


def test_output_is_unchanged_by_the_dispatch_hop(stub_compile):
    """The dispatcher must be a pure hop. A real compile is byte-identical too
    (verified on abstractinator-t); this pins the plumbing that surrounds it."""
    model = build_memory_model()
    ids = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        eager = model(input_ids=ids).logits
        with decode_compiled(model):
            hopped = model(input_ids=ids).logits
    assert torch.equal(eager, hopped)
