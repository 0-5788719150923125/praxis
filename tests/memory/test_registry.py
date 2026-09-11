"""Tests for the Titans NeuralMemory core and surfacings (praxis.memory)."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.blocks.transformer import TransformerBlock
from praxis.memory import MemoryBase, NeuralMemState
from praxis.modeling import PraxisForCausalLM

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
    "mag_energy_stitch_gated",
    "mag_energy_stitch_adaptive",
    "mag_standard_stitch",
]

# Energy-mode profiles (scale-free surprise + event-size stats surfaced).
_ENERGY_SURFACINGS = {
    "mal_energy",
    "mal_energy_serpent",
    "mag_energy",
    "mag_energy_static",
    "mag_energy_stitch",
    "mag_energy_stitch_gated",
    "mag_energy_stitch_adaptive",
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

    # Gate metrics are gate-only: an ungated profile must not put an empty
    # series on the card.
    gate_keys = ("memory_write_share", "memory_write_selectivity")
    if memory_type in ("mag_energy_stitch_gated", "mag_energy_stitch_adaptive"):
        for key in gate_keys + ("memory_write_tilt",):
            assert torch.isfinite(torch.as_tensor(metrics[key])), key
        assert 0.0 < metrics["memory_write_share"] <= 1.0
        # The target is adaptive-only; nothing else has one to report.
        has_target = "memory_write_target" in metrics
        assert has_target == (memory_type == "mag_energy_stitch_adaptive")
    else:
        assert all(key not in metrics for key in gate_keys)
    assert all(key in descriptions for key in gate_keys)

    plain = PraxisForCausalLM(_block_config("none"))
    plain(input_ids=torch.randint(0, 256, (2, 16)))
    assert MemoryBase.collect_training_metrics(plain) == {}
    assert MemoryBase.collect_metric_descriptions(plain) == {}


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
