"""Tests for the Titans NeuralMemory core and surfacings (praxis.memory)."""

import pytest
import torch
import torch.nn as nn

from praxis import PraxisConfig
from praxis.blocks.transformer import TransformerBlock
from praxis.memory import NeuralMemory, NeuralMemState, mem_state_detach
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


# --- surfacing integration (MAL / MAG) --------------------------------------

SURFACINGS = ["mal", "mal_energy", "mal_energy_serpent", "mag"]

# Energy-mode profiles (scale-free surprise + event-size stats surfaced).
_ENERGY_SURFACINGS = {"mal_energy", "mal_energy_serpent"}


def _block_config(memory_type):
    return PraxisConfig(
        vocab_size=256,
        hidden_size=64,
        embed_size=64,
        num_heads=4,
        num_queries=1,
        depth=2,
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
    for key in ("memory_surprise", "memory_gain", "memory_write"):
        assert key in metrics and torch.isfinite(torch.as_tensor(metrics[key]))
        assert key in descriptions
    # The scale-free surprise and event-size stats are energy/segment only.
    event_keys = ("memory_event_size", "memory_event_min", "memory_event_max")
    if memory_type in _ENERGY_SURFACINGS:
        assert torch.isfinite(torch.as_tensor(metrics["memory_surprise_norm"]))
        for key in event_keys:
            assert torch.isfinite(torch.as_tensor(metrics[key]))
    else:
        assert "memory_surprise_norm" not in metrics
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
    seq = torch.randn(2, 96, 32)

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
