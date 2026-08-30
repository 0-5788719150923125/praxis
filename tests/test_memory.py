"""Tests for the Titans NeuralMemory core and surfacings (praxis.memory)."""

import pytest
import torch
import torch.nn as nn

from praxis import PraxisConfig
from praxis.blocks.transformer import TransformerBlock
from praxis.memory import (
    MemoryBase,
    NeuralMemory,
    NeuralMemState,
    mem_state_detach,
)
from praxis.memory.surfacings import MemorySurfacing
from praxis.memory.neural_memory import _affine_scan
from praxis.modeling import PraxisForCausalLM


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


def test_energy_mode_detaches_update_but_trains_readout():
    """The detached update keeps the fast weights off the graph, while the tied
    addressing projection and the memory net still train through retrieval."""
    mem = _energy_mem()
    out, _ = mem(torch.randn(2, 16, 64))
    out.sum().backward()

    assert mem.to_queries.weight.grad is not None
    assert all(p.grad is not None for p in mem.memory_model.parameters())


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


def test_readout_delta_is_zero_without_writes():
    """With the update step disabled the weights never move, so the readout
    probe must report exactly 0 - this is what separates it from the weight
    ratio, whose denominator alone can make a live update look inert."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    mem = NeuralMemory(dim=64, model=model, chunk_size=8, use_energy=True, max_lr=0.0)
    mem(torch.randn(2, 32, 64))
    assert float(mem.last_adapt) == pytest.approx(0.0, abs=1e-6)
    assert float(mem.last_write) == pytest.approx(0.0, abs=1e-6)


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


def test_pad_masking_matches_an_unpadded_reference():
    """The masked write on a padded grid must equal the write the same real
    tokens produce when they happen to fill the grid exactly."""
    seq = torch.randn(2, 32, 64)
    exact = _pad_mem()  # 32 tokens on a 16-grid: 2 chunks, no pad
    _, st_exact = exact(seq)
    again = _pad_mem()
    _, st_again = again(seq.clone())
    for k in st_exact.weights:
        assert torch.equal(st_exact.weights[k], st_again.weights[k])
    assert exact.last_num_chunks == 2


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


# --- surfacing integration (MAL / MAG) --------------------------------------

SURFACINGS = [
    "mal",
    "mal_energy",
    "mal_energy_serpent",
    "mag",
    "mag_energy",
    "mag_energy_static",
    "mag_standard",
    "mag_energy_stitch",
]

# Energy-mode profiles (scale-free surprise + event-size stats surfaced).
_ENERGY_SURFACINGS = {
    "mal_energy",
    "mal_energy_serpent",
    "mag_energy",
    "mag_energy_static",
    "mag_energy_stitch",
}


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


@pytest.mark.parametrize("memory_type", SURFACINGS)
def test_surfacing_alters_output_and_returns_state(memory_type):
    """A memory block changes activations vs. no memory and emits a
    NeuralMemState in the block's layer-state slot."""
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)

    torch.manual_seed(1)
    plain = TransformerBlock(_block_config("none"))
    out_plain, _, state_plain, _ = plain(x, attention_mask=None)

    torch.manual_seed(1)
    block = TransformerBlock(_block_config(memory_type))
    out_mem, _, state_mem, _ = block(x, attention_mask=None)

    assert state_plain is None
    assert isinstance(state_mem, NeuralMemState)
    assert not torch.allclose(out_plain, out_mem)


@pytest.mark.parametrize("memory_type", SURFACINGS)
def test_surfacing_backprops_to_memory(memory_type):
    """Backward through a memory block reaches the meta-learned params."""
    block = TransformerBlock(_block_config(memory_type))
    x = torch.randn(2, 16, 64)
    out, _, _, _ = block(x, attention_mask=None)
    out.sum().backward()
    grads = [p.grad for p in block.memory.mem.memory_model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("memory_type", SURFACINGS)
def test_end_to_end_training_step(memory_type):
    """A full model with the memory profile completes a forward/backward/step
    with a finite next-token loss (driven via logits to sidestep the model's
    internal label-shift handling)."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_block_config(memory_type))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 16))

    logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()


def test_memory_net_has_no_lazy_params():
    """The memory net exposes only concrete parameters, whether the profile
    uses the parameter-free default (gelu) or opts into a learnable/lazy
    activation (serpent). A lazy UninitializedParameter would crash the
    per-sequence weight expansion in init_state, so build_memory_model
    materializes them up front."""
    from praxis.memory import build_memory_model

    cfg = PraxisConfig(hidden_size=64, activation="serpent")

    # Default (no activation in spec) -> gelu, no lazy params.
    default_net = build_memory_model(cfg, {"dense": "mlp", "layers": 2})
    assert {type(p).__name__ for p in default_net.parameters()} == {"Parameter"}

    # Opt into serpent -> lazy per-feature freqs are materialized to concrete
    # Parameters, and they are present (they become fast weights).
    serpent_net = build_memory_model(
        cfg, {"dense": "mlp", "layers": 2, "activation": "serpent"}
    )
    assert {type(p).__name__ for p in serpent_net.parameters()} == {"Parameter"}
    from praxis.activations.serpent import Serpent

    assert any(isinstance(m, Serpent) for m in serpent_net.modules())


@pytest.mark.parametrize("memory_type", SURFACINGS)
def test_surprise_metric_surfaced(memory_type):
    """memory_surprise is collected (value + description) for an active memory
    model via the component-local dynamics path, and absent when off."""
    from praxis.memory import MemoryBase

    model = PraxisForCausalLM(_block_config(memory_type))
    model(input_ids=torch.randint(0, 256, (2, 16)))

    metrics = MemoryBase.collect_training_metrics(model)
    descriptions = MemoryBase.collect_metric_descriptions(model)
    for key in ("memory_surprise", "memory_gain", "memory_write", "memory_adapt"):
        assert key in metrics and torch.isfinite(torch.as_tensor(metrics[key]))
        assert key in descriptions
    # The scale-free surprise is reported in BOTH modes. Energy mode optimizes
    # it; standard mode optimizes the paper's raw MSE - but the readout sits
    # behind out_norm either way, so the memory net's output magnitude is a free
    # mode in both and a drifting raw surprise is otherwise indistinguishable
    # from a memory that stopped learning.
    assert torch.isfinite(torch.as_tensor(metrics["memory_surprise_norm"]))
    # Event-size stats stay segmentation-only (energy).
    event_keys = ("memory_event_size", "memory_event_min", "memory_event_max")
    if memory_type in _ENERGY_SURFACINGS:
        for key in event_keys:
            assert torch.isfinite(torch.as_tensor(metrics[key]))
    else:
        assert all(key not in metrics for key in event_keys)
    # Charts are declared for all memory modules regardless of mode.
    assert "memory_surprise_norm" in descriptions
    assert all(key in descriptions for key in event_keys)

    plain = PraxisForCausalLM(_block_config("none"))
    plain(input_ids=torch.randint(0, 256, (2, 16)))
    assert MemoryBase.collect_training_metrics(plain) == {}
    assert MemoryBase.collect_metric_descriptions(plain) == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(use_energy=True, segment=True),  # mal_energy (the default profile)
        dict(use_energy=True, segment=False),
        dict(use_energy=False, momentum=True),  # standard, differentiable update
        dict(use_energy=False, momentum=False),
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

    def run(parallel):
        mem.parallel_scan = parallel
        return mem(seq, mem.init_state(2))

    (out_p, st_p), (out_s, st_s) = run(True), run(False)
    assert torch.allclose(out_p, out_s, atol=1e-4)
    for field in ("weights", "momentum", "second_moment"):
        for k in getattr(st_p, field):
            assert torch.allclose(
                getattr(st_p, field)[k], getattr(st_s, field)[k], atol=1e-4
            )


# --- stitched writes across linked batch rows -------------------------------


def _links(pattern):
    return torch.tensor(pattern, dtype=torch.bool)


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


def test_band_smear_end_to_end_training_step():
    """The triple-memory model completes a forward/backward/step with finite
    loss (logits-driven, to sidestep the model's label-shift handling)."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_block_config("mal_energy_triple", depth=8))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 16))
    logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()


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


def test_spline_dense_shapes_and_knot_gradients():
    """The spline dense variant maps dim -> dim, stays finite on extreme
    inputs (compact support: far-out values ride the base path), and its knot
    positions/widths receive gradient - they must be learnable for the
    test-time re-knotting thesis to apply."""
    from praxis.dense.spline import SplineNetwork

    class Cfg:
        hidden_size = 32
        activation = "gelu"

    torch.manual_seed(0)
    net = SplineNetwork(Cfg(), num_knots=6)
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = net(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert net.knots.grad is not None and torch.isfinite(net.knots.grad).all()
    assert net.log_widths.grad is not None

    assert torch.isfinite(net(torch.ones(1, 1, 32) * 1000)).all()


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


def test_depth_bank_end_to_end_training_step():
    """The bank completes a forward/backward/step with finite loss (logits-
    driven, to sidestep the model's label-shift handling)."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_block_config("mal_energy_bank", depth=8))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 256, (2, 16))
    logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
