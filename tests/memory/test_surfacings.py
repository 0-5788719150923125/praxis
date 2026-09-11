"""Tests for the Titans NeuralMemory core and surfacings (praxis.memory)."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.blocks.transformer import TransformerBlock
from praxis.memory import MemoryBase, NeuralMemState
from praxis.memory.surfacings import MemorySurfacing
from praxis.modeling import PraxisForCausalLM


def _block_config(memory_type, depth=2):
    return PraxisConfig(
        vocab_size=256,
        hidden_size=64,
        embed_size=64,
        num_heads=4,
        num_queries=1,
        depth=depth,
        num_layers=2,
        memory_type=memory_type,
    )


# --- stitched writes across linked batch rows -------------------------------


def _links(pattern):
    return torch.tensor(pattern, dtype=torch.bool)


def test_stitching_a_detached_update_starves_the_memory_net():
    """Stitching only pays with a DIFFERENTIABLE update. In energy mode the
    state handed between rows of a run carries no graph, so a row past the first
    reads detached weights at every chunk - only run-START rows train the memory
    net, and the signal falls with run length. abstractinator-e ran this pairing
    at run_length 1.94 and gave up its unstitched twin's advantage."""
    from praxis.memory import build_memory

    def grad_at(memory_type, links):
        mem = build_memory(_mag_block_config(memory_type))
        mem.train()
        mem.stitch = True  # force the pairing, for energy mode
        torch.manual_seed(1)
        x = torch.randn(8, 32, 64)
        MemoryBase.set_row_links(mem, links)
        out, _ = mem(x, x, None, current_depth=0)
        mem.zero_grad()
        out.pow(2).mean().backward()
        return float(
            sum(p.grad.pow(2).sum() for p in mem.mem.memory_model.parameters()).sqrt()
        )

    one_run = _links([0, 1, 1, 1, 1, 1, 1, 1])
    e_flat, e_run = grad_at("mag_energy_stitch", None), grad_at(
        "mag_energy_stitch", one_run
    )
    s_flat, s_run = grad_at("mag_standard_stitch", None), grad_at(
        "mag_standard_stitch", one_run
    )
    # Detached: most of the gradient is gone at a run of 8.
    assert e_run / e_flat < 0.6
    # Differentiable: nearly all of it survives.
    assert s_run / s_flat > 0.85


def test_stitch_is_opt_in():
    """Profiles written before row linkage existed must be untouched by it, so
    a stitched run differs from its twin by one declared key."""
    from praxis.memory import build_memory

    plain = build_memory(_mag_block_config("mag_energy"))
    stitched = build_memory(_mag_block_config("mag_energy_stitch"))
    plain.train()
    stitched.train()
    x = torch.randn(8, 32, 64)
    links = _links([0, 1, 1, 1, 0, 1, 0, 0])

    MemoryBase.set_row_links(plain, links)
    MemoryBase.set_row_links(stitched, links)
    a, _ = plain(x, x, None, current_depth=0)
    b, _ = stitched(x, x, None, current_depth=0)
    assert plain.last_run_length is None  # never stitched
    assert stitched.last_run_length == pytest.approx(2.0)
    assert not torch.allclose(a, b, atol=1e-6)


def test_stitch_threads_state_along_a_run_only():
    """A run's FIRST row has nothing to inherit and must be bit-identical to the
    unstitched result; a row deeper in the run reads the earlier rows' writes."""
    from praxis.memory import build_memory

    mem = build_memory(_mag_block_config("mag_energy_stitch"))
    mem.train()
    x = torch.randn(8, 32, 64)

    MemoryBase.set_row_links(mem, None)
    base, _ = mem(x, x, None, current_depth=0)
    MemoryBase.set_row_links(mem, _links([0, 1, 1, 1, 0, 1, 0, 0]))
    out, state = mem(x, x, None, current_depth=0)

    assert torch.allclose(base[0], out[0], atol=1e-6)  # run start
    assert torch.allclose(base[6], out[6], atol=1e-6)  # singleton run
    assert torch.allclose(base[7], out[7], atol=1e-6)  # singleton run
    assert not torch.allclose(base[3], out[3], atol=1e-6)  # deep in a run
    assert not torch.allclose(base[5], out[5], atol=1e-6)
    # The returned state keeps the per-ROW contract the block expects.
    assert next(iter(state.weights.values())).shape[0] == 8


def test_stitch_is_training_only():
    """Generation is a single continuous stream with no batch to stitch, and a
    stale flag there would group unrelated rows."""
    from praxis.memory import build_memory

    mem = build_memory(_mag_block_config("mag_energy_stitch"))
    x = torch.randn(8, 32, 64)
    links = _links([0, 1, 1, 1, 0, 1, 0, 0])

    mem.train()
    MemoryBase.set_row_links(mem, None)
    unstitched, _ = mem(x, x, None, current_depth=0)
    mem.eval()
    MemoryBase.set_row_links(mem, links)
    evaled, _ = mem(x, x, None, current_depth=0)
    assert torch.allclose(unstitched, evaled, atol=1e-6)


def test_stitch_lengthens_the_write_span():
    """The point of the feature: the span the memory writes over is
    run_length x chunks, while the trunk still only sees one row."""
    from praxis.memory import build_memory

    mem = build_memory(_mag_block_config("mag_energy_stitch"))
    mem.train()
    x = torch.randn(8, 32, 64)
    MemoryBase.set_row_links(mem, _links([0, 1, 1, 1, 1, 1, 1, 1]))  # one run of 8
    mem(x, x, None, current_depth=0)
    assert mem.last_run_length == pytest.approx(8.0)
    assert mem.last_run_length * mem.mem.last_num_chunks >= 8 * 2


def test_row_links_reach_the_memory_from_the_model():
    """End-to-end: the model publishes the batch's linkage before the decoder
    runs, so nothing has to thread it through the block's positional chain."""
    model = PraxisForCausalLM(_block_config("mag_energy_stitch"))
    model.train()
    ids = torch.randint(0, 256, (4, 32))
    model(input_ids=ids, row_continues=_links([0, 1, 0, 1]))
    mems = [m for m in model.modules() if isinstance(m, MemorySurfacing)]
    assert mems and all(m.last_run_length == pytest.approx(2.0) for m in mems)
    # And a forward without linkage cannot inherit the previous one's grouping.
    model(input_ids=ids)
    assert all(m._row_links is None for m in mems)


def test_standard_mode_keeps_the_outer_loss_connected_to_the_memory_net():
    """THE reason mag_standard exists. Energy mode detaches the update, so
    retrieval reads W0 only at chunk 0 and a detached constant at every later
    chunk - the gradient reaching the memory net decays as 1/nc, and the meta
    weights get trained as a cold readout rather than as an initialization for
    the update. Differentiating through the update keeps every chunk connected."""
    from praxis.memory import build_memory

    def grad_at(memory_type, n):
        cfg = _mag_block_config(memory_type)
        mem = build_memory(cfg)
        mem.train()
        torch.manual_seed(1)
        x = torch.randn(4, n, 64)
        out, _ = mem(x, x, None, current_depth=0)
        mem.zero_grad()
        out.pow(2).mean().backward()
        g = sum(p.grad.pow(2).sum() for p in mem.mem.memory_model.parameters()).sqrt()
        return float(g), mem.mem.last_num_chunks

    e_lo, nc_lo = grad_at("mag_energy", 8)
    e_hi, nc_hi = grad_at("mag_energy", 64)
    s_lo, _ = grad_at("mag_standard", 8)
    s_hi, _ = grad_at("mag_standard", 64)
    assert nc_hi > nc_lo
    # Energy loses most of the signal as the grid gets finer; standard keeps
    # substantially more of it.
    assert (s_hi / s_lo) > 2.0 * (e_hi / e_lo)


def test_standard_mode_supports_the_predictive_target_and_stop_grads_it():
    """The predictive (NextLat) objective is no longer gated on energy mode, but
    its target MUST stay stop-gradded: a differentiable next-latent target lets
    the encoder minimize surprise by collapsing the stream rather than by
    memorizing it."""
    from praxis.memory import build_memory

    mem = build_memory(_mag_block_config("mag_standard"))
    mem.train()
    assert mem.mem.predictive and not mem.mem.use_energy
    x = torch.randn(2, 32, 64)
    out, _ = mem(x, x, None, current_depth=0)
    out.pow(2).mean().backward()
    # It still trains (standard mode's store projections receive gradient)...
    assert mem.mem.to_keys.weight.grad is not None
    # ...and the target carries no graph.
    tgt = mem.mem._shift_targets(mem.mem.store_norm(x), 32).detach()
    assert not tgt.requires_grad


def test_static_control_matches_its_live_twin_except_for_the_write():
    """``mag_energy_static`` must differ from ``mag_energy`` in exactly one
    thing: whether the write lands. Same parameters, same chunk count, same
    surprise - otherwise it is not a control, it is a second variable."""
    from praxis.memory import build_memory

    live = build_memory(_mag_block_config("mag_energy"))
    static = build_memory(_mag_block_config("mag_energy_static"))
    live.train()
    static.train()
    x = torch.randn(2, 32, 64)
    live(x, x, None, current_depth=0)
    static(x, x, None, current_depth=0)

    assert sum(p.numel() for p in live.parameters()) == sum(
        p.numel() for p in static.parameters()
    )
    assert live.mem.last_num_chunks == static.mem.last_num_chunks
    assert static.mem.max_lr == 0.0
    # The write is frozen, and so is everything downstream of it...
    # abs tolerance, not exact: retrieval and the readout probe are two
    # separate vmapped forwards, so they differ at float noise even when the
    # weights are identical.
    assert float(static.mem.last_write) == pytest.approx(0.0, abs=1e-6)
    assert float(static.mem.last_adapt) == pytest.approx(0.0, abs=1e-6)
    assert float(live.mem.last_write) > 0.0
    # ...but the surprise is still computed, so step cost and the governor's
    # view of the run are unchanged.
    assert static.mem.last_surprise_norm is not None
    assert float(static.mem.last_surprise_norm) > 0.0


# --- pass gating and the MAG verdict line -----------------------------------


def _mag_block_config(memory_type):
    return PraxisConfig(
        depth=6,
        num_layers=1,
        num_experts=1,
        hidden_size=64,
        embed_size=64,
        num_heads=1,
        head_size=32,
        memory_type=memory_type,
    )


def test_passes_gates_the_memory_to_one_recurrent_step():
    """``passes`` keys the memory to the PASS index. Pass 0 is the only station
    every input reaches (training samples a loop count up front, eval exits at
    loop boundaries), which is why the depth bank's late cores starved."""
    from praxis.memory import build_memory

    cfg = _mag_block_config("mag_energy")
    mem = build_memory(cfg)
    mem.train()
    assert mem.passes == frozenset({0})

    x = torch.randn(2, 32, 64)
    fired = []
    for depth in range(cfg.depth):
        out, _ = mem(x, x, None, current_depth=depth)
        fired.append(not torch.equal(out, x))
    assert fired == [True, False, False, False, False, False]


def test_passes_none_runs_every_step():
    """The default is unchanged: no ``passes`` key means every pass fires."""
    from praxis.memory import build_memory

    mem = build_memory(_mag_block_config("mal_energy"))
    mem.train()
    x = torch.randn(2, 32, 64)
    assert mem.passes is None
    assert all(
        not torch.equal(mem(x, x, None, current_depth=d)[0], x) for d in range(6)
    )


def test_pass_gate_is_a_true_identity():
    """A skipped pass must return the stream and the state untouched - not a
    zero-gain write, an actual no-op, so a gated pass costs nothing."""
    from praxis.memory import build_memory

    mem = build_memory(_mag_block_config("mag_energy"))
    mem.train()
    x = torch.randn(2, 32, 64)
    sentinel = object()
    out, state = mem(x, x, sentinel, current_depth=3)
    assert out is x and state is sentinel


def test_mag_reports_the_gate():
    """The gate is the verdict line: the model's own answer to whether it wants
    the memory. It must start near-identity (bias -3) and be surfaced."""
    from praxis.memory import build_memory

    mem = build_memory(_mag_block_config("mag_energy"))
    mem.train()
    mem(torch.randn(2, 32, 64), torch.randn(2, 32, 64), None, current_depth=0)
    metrics = mem.training_metrics()
    assert metrics["memory_gate"] == pytest.approx(
        torch.sigmoid(torch.tensor(-3.0)).item(), abs=1e-3
    )
    assert "memory_gate" in type(mem).metric_descriptions


def test_fine_grid_gives_the_update_room_to_be_seen():
    """The point of the 4-token grid: retrieval reads pre-write weights, so the
    writes the model can feel is chunks - 1. The 16-token grid this repo ran
    resolved typical latent lengths to a single chunk, where adapt is exactly 0."""
    from praxis.memory import build_memory

    fine = build_memory(_mag_block_config("mag_energy"))
    fine.train()
    fine(torch.randn(2, 32, 64), torch.randn(2, 32, 64), None, current_depth=0)
    assert fine.mem.last_num_chunks == 8
    assert float(fine.mem.last_adapt) > 0.0


# --- N-arm reward-bandit memory bank (dual / triple smear) ------------------

BAND_PROFILES = {"mal_energy_dual": 2, "mal_energy_triple": 3}


@pytest.mark.parametrize("memory_type,n_arms", list(BAND_PROFILES.items()))
def test_band_smear_arms_state_and_output(memory_type, n_arms):
    """A band-smear block runs N cores, changes activations vs no memory, and
    returns a tuple of N per-core NeuralMemStates."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)
    torch.manual_seed(1)
    plain = TransformerBlock(_block_config("none", depth=8))
    out_plain, _, _, _ = plain(x, attention_mask=None)
    torch.manual_seed(1)
    block = TransformerBlock(_block_config(memory_type, depth=8))
    out_mem, _, state, _ = block(x, attention_mask=None, current_depth=3)
    assert len(block.memory.mems) == n_arms
    assert isinstance(state, tuple) and len(state) == n_arms
    assert all(isinstance(s, NeuralMemState) for s in state)
    assert not torch.allclose(out_plain, out_mem)


@pytest.mark.parametrize("memory_type,n_arms", list(BAND_PROFILES.items()))
def test_band_smear_backprops_all_cores(memory_type, n_arms):
    """Backward reaches every core's meta-learned params (no arm is detached)."""
    block = TransformerBlock(_block_config(memory_type, depth=8))
    x = torch.randn(2, 16, 64)
    out, _, _, _ = block(x, attention_mask=None, current_depth=3)
    out.sum().backward()
    assert len(block.memory.mems) == n_arms
    for mem in block.memory.mems:
        grads = [p.grad for p in mem.memory_model.parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("memory_type,n_arms", list(BAND_PROFILES.items()))
def test_band_smear_blend_weights_and_river(memory_type, n_arms):
    """Blend weights form a floored simplex (sum to 1, each >= floor); the river
    snapshot carries 2N columns + N labels; equal surprises at init -> 1/N each
    (so N=2 reproduces the old dual's 0.5 center)."""
    from praxis.memory.surfacings import _BLEND_FLOOR

    block = TransformerBlock(_block_config(memory_type, depth=8))
    x = torch.randn(2, 16, 64)
    block(x, attention_mask=None, current_depth=3)  # firing depth: all arms active
    bank = block.memory
    w = bank._last_weights
    assert len(w) == n_arms
    assert abs(sum(w) - 1.0) < 1e-5
    assert min(w) >= _BLEND_FLOOR - 1e-6
    assert all(abs(wi - 1.0 / n_arms) < 1e-6 for wi in w)  # equal at init
    snap = bank.dashboard_snapshots()["memory_regime_river"]
    assert len(snap["river"][0]) == 2 * n_arms
    assert len(snap["labels"]) == n_arms


def test_band_smear_quad_spline_stagger():
    """mal_energy_quad runs four regimes with the two grid cores STAGGERED
    (spline fires at depth%4==1, KAN at depth%4==3), so no single step pays
    both. The spline's knots/widths are Parameters (fast weights - the
    adaptive-resolution thesis), unlike the KAN grid's frozen buffers; when the
    spline fires it receives gradient and reports its earned share as
    memory_blend_d."""
    from praxis.dense.spline import SplineNetwork

    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)
    block = TransformerBlock(_block_config("mal_energy_quad", depth=8))
    bank = block.memory
    assert len(bank.mems) == 4
    assert bank._active_rule[2] == (4, 3)  # KAN
    assert bank._active_rule[3] == (4, 1)  # spline

    # The spline arm's basis placement is fast weights, not frozen buffers.
    spline_net = bank.mems[3].memory_model
    assert isinstance(spline_net, SplineNetwork)
    param_names = {n for n, _ in spline_net.named_parameters()}
    assert {"knots", "log_widths"} <= param_names

    # Depth 1: spline fires, KAN sits out; active arms share a floored simplex.
    out, _, state, _ = block(x, attention_mask=None, current_depth=1)
    w = bank._last_weights
    assert w[2] == 0.0 and w[3] > 0.0
    assert abs(sum(w) - 1.0) < 1e-5
    assert state[2] is None and isinstance(state[3], NeuralMemState)
    out.sum().backward()
    grads = [p.grad for p in spline_net.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert "memory_blend_d" in bank.training_metrics()

    # Depth 3 on a fresh block: the mirror phase - KAN fires, spline sits out.
    fresh = TransformerBlock(_block_config("mal_energy_quad", depth=8))
    _, _, s3, _ = fresh(x, attention_mask=None, current_depth=3)
    w3 = fresh.memory._last_weights
    assert w3[3] == 0.0 and w3[2] > 0.0
    assert s3[3] is None and isinstance(s3[2], NeuralMemState)


def test_band_smear_sparse_kan_gate():
    """mal_energy_triple gates its KAN core by recurrent step (period 4, phase 3):
    non-firing steps skip its forward (weight 0, 2-arm renorm, no grad); firing
    steps run all three and the KAN receives gradient. Fresh block per check so a
    prior call's surprise EMA doesn't perturb the at-init 1/3 shares."""
    x = torch.randn(2, 16, 64)

    # Non-firing depth -> KAN (arm 2) sits out; A/B renormalize to 0.5 each.
    off = TransformerBlock(_block_config("mal_energy_triple", depth=8))
    assert off.memory._active_rule[2] == (4, 3)  # KAN is the sparse arm
    off(x, attention_mask=None, current_depth=0)
    assert off.memory._last_weights[2] == 0.0
    assert abs(off.memory._last_weights[0] - 0.5) < 1e-6
    assert abs(sum(off.memory._last_weights) - 1.0) < 1e-6

    # Firing depth on a fresh block -> all three active, equal at init, and the
    # KAN core receives gradient through the blend.
    on = TransformerBlock(_block_config("mal_energy_triple", depth=8))
    out, _, _, _ = on(x, attention_mask=None, current_depth=3)
    assert out.requires_grad
    assert all(abs(w - 1.0 / 3) < 1e-6 for w in on.memory._last_weights)
    out.sum().backward()
    kan_grads = [p.grad for p in on.memory.mems[2].memory_model.parameters()]
    assert kan_grads and all(
        g is not None and torch.isfinite(g).all() for g in kan_grads
    )


# --- one-core-per-pass memory bank (depth bank) -----------------------------
#
# Deliberately NOT in BAND_PROFILES or SURFACINGS: those suites assert contracts
# this surfacing breaks by construction (every arm active in one forward, a
# floored simplex of blend weights, a bare `.mem`). The precedent is
# test_band_smear_quad_spline_stagger - a dedicated test for a profile where
# arms sit out. `_block_config` builds num_layers=2, so the pass index is
# `current_depth // 2` and depth=8 gives exactly four passes for the four cores.


def _sweep(block, x, depths):
    """Run a block over consecutive depths the way SequentialDecoder does,
    threading the memory state and returning the final hidden states."""
    state, hidden = None, x
    for depth in depths:
        hidden, _, state, _ = block(
            hidden, attention_mask=None, current_state=state, current_depth=depth
        )
    return hidden, state


def test_depth_bank_runs_exactly_one_core_per_pass():
    """Pass p runs core p % N and nothing else: one state slot advances per
    call, the assignment is keyed to the PASS (current_depth // num_layers, so
    both depths of a two-layer pass share a core), and it wraps past the bank
    instead of running off its end."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)
    torch.manual_seed(1)
    plain = TransformerBlock(_block_config("none", depth=8))
    out_plain, _, _, _ = plain(x, attention_mask=None)
    torch.manual_seed(1)
    block = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    bank = block.memory
    assert len(bank.mems) == 4

    # num_layers=2 -> two depths per pass, and the bank wraps at pass 4.
    assert [bank._core_index(d) for d in range(10)] == [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]

    out_mem, _, state, _ = block(x, attention_mask=None, current_depth=4)
    assert not torch.allclose(out_plain, out_mem)
    assert isinstance(state, tuple) and len(state) == 4
    # Only the pass's own core wrote a state; the rest stay untouched.
    assert isinstance(state[2], NeuralMemState)
    assert [state[i] for i in (0, 1, 3)] == [None, None, None]


def test_depth_bank_backprop_follows_the_assignment():
    """A single pass gives gradient to its core alone (that is the compute
    saving made visible); a forward deep enough to cycle the whole bank gives
    gradient to every core."""
    x = torch.randn(2, 16, 64)

    one = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    out, _, _, _ = one(x, attention_mask=None, current_depth=0)  # pass 0 -> core A
    out.sum().backward()
    assert all(p.grad is not None for p in one.memory.mems[0].memory_model.parameters())
    for mem in list(one.memory.mems)[1:]:
        assert all(p.grad is None for p in mem.memory_model.parameters())

    full = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    out, _ = _sweep(full, x, range(8))  # four passes -> the whole cycle
    out.sum().backward()
    for mem in full.memory.mems:
        grads = [p.grad for p in mem.memory_model.parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_depth_bank_use_tracks_the_pass_budget():
    """*_memory_core_use is the halting distribution read through the bank: a
    forward cut short leaves the late cores at 0 occupancy and emits no
    diagnostics for them at all (rather than a stale repeat of an earlier
    step), while a full cycle splits evenly."""
    x = torch.randn(2, 16, 64)

    halted = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    _sweep(halted, x, range(4))  # two of four passes, as an early exit would
    metrics = halted.memory.training_metrics()
    assert [metrics[f"{c}_memory_core_use"] for c in "abcd"] == [0.5, 0.5, 0.0, 0.0]
    assert "a_memory_surprise_norm" in metrics and "b_memory_surprise_norm" in metrics
    for letter in ("c", "d"):  # never reached -> nothing to report
        assert not [k for k in metrics if k.startswith(f"{letter}_memory_s")]
        assert f"{letter}_memory_gain" not in metrics

    full = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    _sweep(full, x, range(8))
    metrics = full.memory.training_metrics()
    assert [metrics[f"{c}_memory_core_use"] for c in "abcd"] == [0.25] * 4
    for letter in "abcd":
        assert metrics[f"{letter}_memory_gain"] > 0.0


def test_depth_bank_river_widths_are_occupancy():
    """The river carries 2N columns + N labels like the regime river, but the
    widths are occupancy rather than blend weights - so they still sum to 1
    (exactly one core per pass), and the card stays empty until every core has
    reported a surprise, rather than painting filler."""
    x = torch.randn(2, 16, 64)
    block = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    bank = block.memory

    _sweep(block, x, range(8))
    assert bank.dashboard_snapshots() == {}  # settled only by the NEXT forward
    _sweep(block, x, range(8))

    snap = bank.dashboard_snapshots()["memory_depth_river"]
    assert len(snap["labels"]) == 4
    row = snap["river"][0]
    assert len(row) == 8
    assert abs(sum(row[:4]) - 1.0) < 1e-6
    assert all(0.0 <= f <= 1.0 for f in row[4:])


def test_depth_bank_river_brightness_is_per_band_and_inverted():
    """Brightness is min-maxed WITHIN a band and inverted, so the two axes stay
    independent: the lowest surprise a core has recently shown is its brightest
    row, and a band whose own range is wide cannot dim a band whose range is
    narrow (they sit at different depths and share no scale)."""
    block = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    bank = block.memory
    bank._passes_seen = 4

    # Band A swings over a wide range, band B over a narrow one at a much
    # higher level; C and D are flat. Rows are (oldest -> newest).
    for surprises in ([1.0, 90.0, 5.0, 5.0], [3.0, 91.0, 5.0, 5.0]):
        bank._core_surprise = list(surprises)
        bank._settle_forward()

    rows = bank.dashboard_snapshots()["memory_depth_river"]["river"]
    assert len(rows) == 2
    fits = [row[4:] for row in rows]
    assert fits[0][0] == 1.0 and fits[1][0] == 0.0  # A: 1.0 < 3.0 -> brighter
    assert fits[0][1] == 1.0 and fits[1][1] == 0.0  # B: its own range, not A's
    assert fits[0][2] == fits[1][2] == 0.5  # flat band stays mid-bright
    assert fits[0][3] == fits[1][3] == 0.5


def test_depth_bank_river_waits_for_every_core_to_report():
    """A row is held back until every core has reported a surprise, so the card
    opens on real fitnesses instead of filler for the deep cores the model has
    not reached yet."""
    block = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    bank = block.memory
    bank._passes_seen = 4

    bank._core_surprise = [0.5, 0.5, 0.5, None]  # spline never reached
    bank._settle_forward()
    assert bank.dashboard_snapshots() == {}

    bank._core_surprise[3] = 0.5
    bank._settle_forward()
    assert len(bank.dashboard_snapshots()["memory_depth_river"]["river"]) == 1


def test_depth_bank_ignores_eval_forwards():
    """Generation runs inside the training loop in eval mode, at whatever depth
    the KL exit picks. Its occupancy must not land in the cards: an eval
    forward leaves the accounting exactly as the last training forward left
    it."""
    x = torch.randn(2, 16, 64)
    block = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    _sweep(block, x, range(8))  # training: the full four-pass cycle
    bank = block.memory
    trained = bank.training_metrics()

    block.eval()
    with torch.no_grad():
        _sweep(block, x, range(2))  # a shallow decode: one pass, core A only
    assert bank.training_metrics() == trained
    assert bank._passes_seen == 4


def test_depth_bank_settles_per_forward_for_every_layer_position():
    """The forward boundary is the first pass, not depth 0. With distinct
    physical layers the decoder gives block j only depths congruent to j, so a
    depth-0 key would never reset block 1 and its occupancy would ratchet to an
    all-time maximum instead of reporting this forward."""
    x = torch.randn(2, 16, 64)
    block = TransformerBlock(_block_config("mal_energy_bank", depth=8))
    bank = block.memory  # num_layers=2: block 1 sees depths 1, 3, 5, 7

    _sweep(block, x, [1, 3, 5, 7])  # a deep forward - all four passes
    assert [bank.training_metrics()[f"{c}_memory_core_use"] for c in "abcd"] == [
        0.25
    ] * 4

    _sweep(block, x, [1, 3])  # a shallow one - the counter must fall back
    assert [bank.training_metrics()[f"{c}_memory_core_use"] for c in "abcd"] == [
        0.5,
        0.5,
        0.0,
        0.0,
    ]


def test_depth_bank_routing_is_a_pure_function_of_depth():
    """Nothing this surfacing tracks feeds the output, so two identical eval
    forwards agree exactly - what the byte-latent speculative decoder needs
    from anything on this path (draft and verify are separate forwards)."""
    x = torch.randn(2, 16, 64)
    block = TransformerBlock(_block_config("mal_energy_bank", depth=8)).eval()
    with torch.no_grad():
        first, _ = _sweep(block, x, range(8))
        second, _ = _sweep(block, x, range(8))
    assert torch.equal(first, second)


def test_depth_bank_warns_when_the_recurrence_cannot_reach_every_core(capsys):
    """A bank deeper than the pass budget carries cores that can never run;
    that is a config error worth saying out loud rather than silently paying
    for dead parameters."""
    TransformerBlock(_block_config("mal_energy_bank", depth=2))  # 1 pass, 4 cores
    out = capsys.readouterr().out
    assert "depth_bank" in out and "can never run" in out
    assert "kan" in out and "spline" in out
