import math

import torch

from praxis.activations.serpent import Serpent
from praxis.activations.servant import MOD_MAX, Servant
from praxis.metrics.specialization import (
    collect_activation_descriptions,
    collect_activation_metrics,
)


def _built(cls, x, **kwargs):
    """Lazy modules materialize on first forward."""
    module = cls(**kwargs)
    module(x)
    return module


def test_identity_to_serpent_at_init_in_value_and_gradient():
    """`v` is zero-init, so a_eff == a and -m starts as -k exactly. Unlike
    Ouroboros this is a plain differentiable forward, so the GRADIENT is exact
    too, not just the value."""
    torch.manual_seed(0)
    x0 = torch.randn(2, 5, 16)

    servant = _built(Servant, x0, a=1.0, b=1.0, g=0.1)
    serpent = _built(Serpent, x0, a=1.0, b=1.0, g=0.1)

    xa = x0.clone().requires_grad_(True)
    xb = x0.clone().requires_grad_(True)
    ya, yb = servant(xa), serpent(xb)
    ya.sum().backward()
    yb.sum().backward()

    assert (ya - yb).abs().max().item() == 0.0
    assert (xa.grad - xb.grad).abs().max().item() == 0.0


def test_coupling_moves_the_frequency():
    """With v away from zero the frequency must actually track token energy,
    and the swing must stay inside MOD_MAX."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    servant = _built(Servant, x, a=1.0, b=1.0, g=0.1)
    with torch.no_grad():
        servant.v.fill_(2.0)

    # Two batches at deliberately different energies must chirp differently.
    quiet = servant(x * 0.1)
    loud = servant(x * 10.0)
    assert not torch.allclose(quiet, loud * 0.0)  # sanity: outputs are live

    metrics = servant.training_metrics()
    assert metrics["servant_coupling"] > 0.9  # tanh(2.0) ~ 0.96
    assert 0.0 < metrics["servant_chirp"] <= MOD_MAX


def test_metrics_are_zero_at_init_and_reachable_by_the_walk():
    """servant_coupling is the run's gate metric, so it has to start at exactly
    zero and has to survive the module walk the dynamics logger uses."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    servant = _built(Servant, x)
    servant(x)  # populate the realized-swing stash

    metrics = servant.training_metrics()
    assert metrics["servant_coupling"] == 0.0
    assert metrics["servant_coupling_std"] == 0.0
    assert metrics["servant_chirp"] == 0.0

    root = torch.nn.Sequential(torch.nn.Linear(16, 16), servant)
    collected = collect_activation_metrics(root)
    assert collected["servant_coupling"] == 0.0
    assert "servant_coupling" in collect_activation_descriptions(root)


def test_plain_serpent_publishes_nothing():
    """The walk must not invent metrics for activations that never opted in."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    root = torch.nn.Sequential(_built(Serpent, x))
    assert collect_activation_metrics(root) == {}
    assert collect_activation_descriptions(root) == {}


def test_uninitialized_module_reports_nothing():
    """training_metrics runs on the logging cadence and may fire before the
    first forward has materialized the lazy parameters."""
    assert Servant().training_metrics() == {}


def test_the_swing_stash_ignores_a_no_grad_probe():
    """The dashboard samples every activation on a `linspace(-6, 6)` under
    `torch.no_grad()`, without changing train/eval mode (it races the trainer).

    While `self.training` was the only guard, that read-only probe overwrote the
    realized-swing stash with the swing of a synthetic ramp, and `servant_chirp`
    went bimodal - roughly 6% of logged points carried the probe's number.
    """
    act = Servant()
    act(torch.randn(4, 16, 32))  # materialize
    with torch.no_grad():
        act.v.fill_(1.0)  # something to chirp with
    real = torch.randn(4, 16, 32) * 3.0
    act(real)
    stashed = float(act._swing)

    probe = torch.linspace(-6.0, 6.0, 256).unsqueeze(-1).expand(256, 32).contiguous()
    with torch.no_grad():
        act(probe)
    assert float(act._swing) == stashed, "a no_grad probe wrote the training stash"

    # ...and a real forward still updates it.
    act(torch.randn(4, 16, 32) * 0.01)
    assert float(act._swing) != stashed


def test_the_energy_signal_survives_a_shift_in_activation_scale():
    """The defect that killed the original: `tanh(log s - log_s_ref)` centred on
    a scalar frozen at init.

    A network's activation scale drifts several nats over training, and a tanh
    is one nat wide, so the signal saturates and stops being a signal. Measured
    on abstractinator-s, mean |m| was 0.999 from step 4000 to step 24000. The
    standardized form has to stay graded through the same drift.
    """
    act = Servant()
    x = torch.randn(8, 64, 32)
    act(x)  # materialize + seed the running stats

    # Scale up by e^4 - a bigger drift than the real run showed.
    for _ in range(400):
        act(torch.randn(8, 64, 32) * math.exp(4.0))

    signal = float(act._signal)
    assert 0.1 < signal < 0.9, f"signal saturated or dead at {signal}"


def test_chirp_measures_dispersion_not_magnitude():
    """A constant swing is a rescaling of `a`, not a chirp, and must read ~0."""
    act = Servant()
    act(torch.randn(8, 64, 32))
    with torch.no_grad():
        act.v.fill_(2.0)  # strongly coupled

    # Every token carries the SAME energy -> m is constant -> no chirp, however
    # large the coupling. The magnitude-based metric read MOD_MAX*coupling here.
    flat = torch.ones(8, 64, 32)
    act(flat)
    assert float(act._swing) < 1e-4, float(act._swing)

    # A spread of token energies IS a chirp, and registers.
    varied = torch.randn(8, 64, 32) * torch.rand(8, 64, 1).mul(4.0).add(0.1)
    act(varied)
    assert float(act._swing) > 1e-3, float(act._swing)


def test_the_signal_is_centered_so_v_is_not_a_second_copy_of_a():
    """With E[m] ~ 0 the swing averages out, so `a_eff` averages to `a` and the
    coupling is not a redundant reparametrisation the optimizer can decay away
    along a degenerate direction - which is what the falling-chirp arc was."""
    act = Servant()
    x = torch.randn(16, 64, 32)
    act(x)
    for _ in range(200):
        act(torch.randn(16, 64, 32))
    with torch.no_grad():
        act.v.fill_(1.5)
        m = act._energy_signal(torch.randn(16, 64, 32), live=False)
    assert abs(float(m.mean())) < 0.15, float(m.mean())


def test_running_stats_are_buffers_and_stay_out_of_the_optimizer():
    act = Servant()
    act(torch.randn(4, 16, 32))
    names = {n for n, _ in act.named_parameters()}
    assert "log_s_mean" not in names and "log_s_var" not in names
    assert not hasattr(act, "log_s_ref")
    buffers = dict(act.named_buffers())
    assert buffers["log_s_mean"].shape == (1,)
    assert buffers["log_s_var"].shape == (1,)


def test_a_no_grad_probe_does_not_advance_the_running_stats():
    act = Servant()
    act(torch.randn(4, 16, 32))
    before = act.log_s_mean.clone()
    probe = torch.linspace(-6.0, 6.0, 256).unsqueeze(-1).expand(256, 32).contiguous()
    with torch.no_grad():
        act(probe)
    assert torch.equal(act.log_s_mean, before)
