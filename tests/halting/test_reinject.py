from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from praxis import registry
from praxis.containers import LossContainer

# ------------------------------------------------------------------------------
# halting
# ------------------------------------------------------------------------------
# The training-time depth prior, which is the shape the halting signal learns.
#
# `KLDivergenceHalting` samples a loop count per forward so the model never knows how
# much compute it will get. The distribution those samples come from is the experiment:
# it decides how much of the budget the model learns to treat as routine, and the
# inference-time KL rule can only ever exit somewhere the prior taught it to be useful.
#
# What is pinned here is the SHAPE, not the sampler's internals - the ramp toward
# multiple steps, and how fast the tail dies as the depth budget grows.


# --- kl_log_reinject: the recurrence re-reads its input, positions exit alone --


def _reinject(depth=6, hidden=16):
    return registry.lookup("halting", "kl_log_reinject")(
        SimpleNamespace(depth=depth, num_layers=1, hidden_size=hidden)
    )


def test_reinject_starts_from_noise_and_rereads_the_input():
    h = _reinject()
    x = torch.randn(2, 5, 16)
    h.seed(x)
    s0 = h.initial_state(x)
    assert s0.shape == x.shape and not torch.allclose(s0, x)
    assert s0.abs().max() <= 3.0
    # Two different inputs must give two different block inputs from one state.
    a = h.inject(s0, 0)
    h.seed(torch.randn(2, 5, 16))
    b = h.inject(s0, 0)
    assert a.shape == x.shape and not torch.allclose(a, b)


def test_reinject_noise_is_deterministic_at_inference_only():
    h = _reinject()
    x = torch.randn(2, 5, 16)
    h.eval()
    torch.testing.assert_close(h.initial_state(x), h.initial_state(x))
    h.train()
    assert not torch.allclose(h.initial_state(x), h.initial_state(x))


def test_reinject_adapter_trains():
    h = _reinject()
    x = torch.randn(2, 5, 16, requires_grad=True)
    h.seed(x)
    h.inject(h.initial_state(x), 0).sum().backward()
    assert h.adapter.weight.grad is not None and x.grad is not None


def test_reinject_holds_no_graph_after_the_loop():
    """The re-read input carries the encoder's graph; a forward that ends with it
    still held keeps that graph alive until the next forward."""
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    torch.manual_seed(0)
    config = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=3,
        decoder_type="sequential",
        halting_type="kl_log_reinject",
    )
    model = PraxisForCausalLM(config).train()
    ids = torch.randint(4, 900, (2, 12))
    model(input_ids=ids, labels=ids[:, 1:].contiguous()).loss.backward()
    assert model.decoder.halting._input is None


def test_reinject_rejects_a_length_change_inside_the_loop():
    h = _reinject()
    h.seed(torch.randn(2, 5, 16))
    with pytest.raises(ValueError, match="cannot change length"):
        h.inject(torch.randn(2, 4, 16), 0)


def _run_loop(h, states):
    """Drive the decoder's per-step calls over a scripted state sequence."""
    h.eval()
    h.get_depth()
    h.seed(states[0])
    out = None
    for depth, state in enumerate(states):
        out = h.settle(state)
        if h.check(out, depth):
            break
    return out, depth


def test_positions_exit_independently_and_stay_frozen():
    """Position 0 is unchanged from step 2 on, so step 3 is its first zero KL;
    position 1 keeps moving. The first exits and holds its state, the second
    runs to the budget."""
    torch.manual_seed(0)
    depth = 6
    h = _reinject(depth=depth)
    base = torch.randn(1, 2, 16)
    states = []
    for r in range(depth):
        s = base.clone()
        s[0, 1] = torch.randn(16) * 3  # position 1 never settles
        if r == 0:
            s[0, 0] = torch.randn(16) * 3  # step 1 is the move out of noise
        states.append(s)
    out, stopped_at = _run_loop(h, states)
    hist = {r: c for r, c in h._eval_hist.items() if c}
    assert hist.get(3) == 1, f"position 0 should exit once it stops moving: {hist}"
    assert hist.get(depth) == 1, f"position 1 should run to the budget: {hist}"
    assert stopped_at == depth - 1
    torch.testing.assert_close(out[0, 0], states[2][0, 0])  # frozen at its exit


def test_the_scale_skips_the_step_out_of_noise():
    """The first check stores a baseline and never exits anyone: a floor anchored
    on the jump out of noise would clear at the first refinement everywhere."""
    h = _reinject(depth=4)
    x = torch.randn(1, 3, 16)
    h.eval()
    h.get_depth()
    h.seed(x)
    assert h.check(h.settle(x), 0) is False
    assert sum(h._eval_hist.values()) == 0
    assert h._pass_peak == 0.0


def test_the_whole_pass_ends_when_every_position_has_exited():
    h = _reinject(depth=8)
    still = torch.randn(1, 4, 16)
    moving = torch.randn(1, 4, 16) * 3
    out, stopped_at = _run_loop(
        h, [moving, still, still, still, still, still, still, still]
    )
    assert stopped_at < 7, "a fully converged batch ran to the budget"
    assert sum(h._eval_hist.values()) == 4


# ------------------------------------------------------------------------------
# smear_integration
# ------------------------------------------------------------------------------
# Test SMEAR integration with sequential decoder and multiple experts.


@dataclass
class MockConfig:
    """Mock configuration for testing SMEAR integration."""

    # Core configuration
    hidden_size: int = 256
    depth: int = 6
    num_experts: int = 3  # Number of experts for SMEAR to manage
    num_layers: int = 3  # Number of layer components for controllers
    epsilon: float = 1e-6
    dropout: float = 0.1

    # Decoder configuration
    decoder_type: str = "sequential"
    block_type: str = "recurrent"
    router_type: str = "smear"
    controller_type: str = "base"
    compression_type: str = "none"
    sorting_type: str = "none"
    halting_type: str = "none"

    # Additional required fields
    checkpoint_every: int = 0
    debug: bool = False
    evolve: bool = False
    hivemind: bool = False
    expert: str = "default"
    meta: dict = None

    # For blocks that need these
    num_heads: int = 8
    activation: str = "swish"
    causal: bool = True

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


def test_reinject_halting_runs_through_the_decoder():
    """kl_log_reinject inside the real sequential loop: training re-reads the
    input and trains the adapter; inference records one exit per position."""
    from dataclasses import replace

    config = replace(
        MockConfig(num_experts=3, num_layers=1, depth=4), halting_type="kl_log_reinject"
    )
    decoder = registry.lookup("decoders", "sequential")(config)
    x = torch.randn(2, 7, config.hidden_size, requires_grad=True)

    decoder.train()
    out, _, _, _ = decoder(x, losses=LossContainer())
    out.mean().backward()
    assert decoder.halting.adapter.weight.grad is not None
    assert x.grad is not None and x.grad.abs().sum() > 0

    decoder.eval()
    with torch.no_grad():
        out, _, _, _ = decoder(x.detach(), losses=LossContainer())
    assert out.shape == x.shape
    assert sum(decoder.halting._eval_hist.values()) == 2 * 7
