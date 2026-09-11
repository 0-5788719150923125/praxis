"""The training-time depth prior, which is the shape the halting signal learns.

`KLDivergenceHalting` samples a loop count per forward so the model never knows
how much compute it will get. The distribution those samples come from is the
experiment: it decides how much of the budget the model learns to treat as
routine, and the inference-time KL rule can only ever exit somewhere the prior
taught it to be useful.

What is pinned here is the SHAPE, not the sampler's internals - the ramp toward
multiple steps, and how fast the tail dies as the depth budget grows.
"""

import collections
import functools
import math

import pytest
import torch
from types import SimpleNamespace

from praxis.halting import HALTING_REGISTRY
from praxis.halting.kl import LOOP_PRIORS, KLDivergenceHalting

SAMPLES = 20_000


def _halting(key, depth, num_layers=1):
    return HALTING_REGISTRY[key](
        SimpleNamespace(depth=depth, num_layers=num_layers, hidden_size=16)
    )


@functools.lru_cache(maxsize=None)
def _pmf(key, depth, seed=0):
    """Monte-Carlo the real sampler. Deliberately not an analytic reimplementation
    - the point is to measure what training will actually see. Cached because the
    sampler builds two distribution objects per draw and several tests read the
    same curve."""
    torch.manual_seed(seed)
    module = _halting(key, depth)
    counts = collections.Counter(module._sample_loop_count() for _ in range(SAMPLES))
    return tuple(counts[r] / SAMPLES for r in range(1, module.max_loops + 1))


def test_the_linear_prior_is_untouched():
    """Adding a prior must not move `kl`, or every arm on it changes shape."""
    assert _halting("kl", 6).r_bar == pytest.approx(2.5)
    assert _halting("kl", 18).r_bar == pytest.approx(8.5)


@pytest.mark.parametrize("max_loops", [4, 6, 12, 18, 32])
def test_the_log_prior_tracks_the_logarithm_of_the_budget(max_loops):
    """Derived from the depth budget, not swept. Doubling the budget adds a
    constant to the expected loop count instead of doubling it, which is the
    whole difference between the two curves."""
    assert _halting("kl_log", max_loops).r_bar == pytest.approx(
        max(1.0, math.log(max_loops))
    )
    doubled = _halting("kl_log", 2 * max_loops).r_bar
    assert doubled - math.log(max_loops) == pytest.approx(math.log(2), abs=1e-6)


@pytest.mark.parametrize("key", sorted(HALTING_REGISTRY.keys() - {"none"}))
@pytest.mark.parametrize("depth", [6, 18])
def test_the_ramp_toward_multiple_steps_survives(key, depth):
    """The property that keeps the model from exiting at one loop constantly:
    halting at 1 is LESS likely than halting at 2. True of the paper's curve and
    it has to stay true of any replacement, or the prior stops teaching depth."""
    p = _pmf(key, depth)
    assert p[0] < p[1]


def test_the_log_prior_puts_its_mass_on_the_first_few_loops():
    """At a budget of 18 the linear rule slides its whole curve right - mode 7,
    and better than a third of forwards running 10+ loops, which is close to
    uniform over the range. The log rule keeps the mode where a 6-deep model had
    it and spends the extra budget on a thin tail instead."""
    linear, log = _pmf("kl", 18), _pmf("kl_log", 18)

    assert linear.index(max(linear)) + 1 >= 6
    assert log.index(max(log)) + 1 <= 3

    assert sum(linear[9:]) > 0.25
    assert sum(log[9:]) < 0.05


def test_full_depth_is_rare_rather_than_routine():
    """ "Essentially unreachable" is the requirement, and 1.6% of forwards is not
    that. Measured as a ratio because the absolute count at 20k samples is small
    enough to be noisy."""
    linear, log = _pmf("kl", 18), _pmf("kl_log", 18)
    assert log[-1] < linear[-1] / 20


def test_the_descent_is_monotone_past_the_mode():
    """A deeper descent, not a bumpy one: once past the peak every further loop is
    strictly less likely than the one before it."""
    p = _pmf("kl_log", 18)
    peak = p.index(max(p))
    tail = p[peak:]
    assert all(b <= a for a, b in zip(tail, tail[1:]))


def test_an_explicit_r_bar_still_wins():
    """The escape hatch tests use to pin a specific curve. It has to beat the
    prior, not be averaged with it."""
    config = SimpleNamespace(depth=18, num_layers=1)
    assert KLDivergenceHalting(config, r_bar=5.0, prior="log").r_bar == 5.0


def test_an_unknown_prior_is_refused():
    """A typo in a registry profile should not silently fall back to the paper's
    curve - that failure would look like a successful run of the wrong experiment."""
    with pytest.raises(ValueError, match="Unknown loop prior"):
        KLDivergenceHalting(SimpleNamespace(depth=6, num_layers=1), prior="logarithmic")


@pytest.mark.parametrize("name", sorted(LOOP_PRIORS))
def test_every_prior_is_usable_at_every_reachable_budget(name):
    """`r_bar` feeds `log(r_bar)` inside the sampler, so a rule dipping to zero at
    a shallow budget would raise there rather than at construction. The floor in
    the constructor is what stops that, and it applies to every rule."""
    for max_loops in range(1, 33):
        module = KLDivergenceHalting(
            SimpleNamespace(depth=max_loops, num_layers=1), prior=name
        )
        assert module.r_bar >= 1.0
        assert 1 <= module._sample_loop_count() <= max_loops


# --- kl_log_reinject: the recurrence re-reads its input, positions exit alone --


def _reinject(depth=6, hidden=16):
    return HALTING_REGISTRY["kl_log_reinject"](
        SimpleNamespace(depth=depth, num_layers=1, hidden_size=hidden)
    )


def test_other_profiles_leave_the_loop_untouched():
    """The three loop hooks are identities everywhere but kl_log_reinject."""
    x = torch.randn(2, 5, 16)
    for key in HALTING_REGISTRY.keys() - {"kl_log_reinject"}:
        h = _halting(key, 6)
        assert h.initial_state(x) is x
        assert h.inject(x, 0) is x
        assert h.settle(x) is x


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
