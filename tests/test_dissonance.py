"""The dissonance term, and the instrument it was built beside.

`harmonic_kl` is named for a claim about the harmonic basis but EMAs whatever
the head hands over as its readout - HALO's centroids on the prismatic line -
so a near-zero drift there says nothing about the field. These tests pin what
each term actually watches, pin the roughness kernel to the published
Plomp-Levelt curve and its scale to the true ceiling, and pin the dual to its
constraint: the roughness of the signal the field multiplies.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from praxis import registry
from praxis.losses.dissonance import (
    CRITICAL_BAND,
    LAMBDA_MAX,
    PL_B1,
    PL_B2,
    RHO_INIT,
    RHO_MAX,
    Dissonance,
    _find_field,
    roughness_ceiling,
    roughness_kernel,
)
from praxis.losses.harmonic_kl import HarmonicKLRegularizer


def _model(**overrides):
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    cfg = dict(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=1,
        depth=2,
        tokenizer_type="byte_level",
        decoder_type="sequential",
        head_type="prismatic5",
        residual_type="smear",
        loss_func="halo",
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    return PraxisForCausalLM(PraxisConfig(**cfg)).train()


def _step(model, opt=None):
    ids = torch.randint(4, 900, (2, 24))
    out = model(input_ids=ids, labels=ids[..., 1:].contiguous())
    if opt is not None:
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    return out


# ── what each term watches ─────────────────────────────────────────────────


def test_harmonic_kl_names_the_readout_it_actually_watches():
    """The misreading this exists to stop: on a multi-arm head the target is
    HALO's centroids, not the harmonic field."""
    m = _model(regularizers=["harmonic_kl"])
    assert repr(m.criterion.harmonic_kl) == "HarmonicKLRegularizer(target=readout)"
    _step(m)
    target = m.criterion.harmonic_kl._target
    assert target.startswith("HaloClassifier(")
    assert "centers" in target and "gamma" in target


def test_the_dissonance_term_watches_the_field_wherever_it_is_mounted():
    """Located by the method it needs, not by a fixed path: the field sits at
    the stem on some profiles and inside a sequential arm on others."""
    m = _model()
    field = _find_field(m.head)
    assert field is not None
    assert any(mod is field for mod in m.head.modules())
    # Not the readout - which is the whole distinction harmonic_kl blurs.
    assert field is not m.head.classifier
    assert _find_field(None) is None
    assert _find_field(torch.nn.Linear(4, 4)) is None


# ── the kernel ─────────────────────────────────────────────────────────────


def test_the_kernel_is_the_published_curve():
    """Plomp-Levelt in Sethares' two-exponential form: zero at unison, zero for
    wide intervals, peak at 0.22 critical bands."""
    kernel = roughness_kernel(64)
    assert torch.equal(kernel, kernel.T)
    assert float(kernel.diagonal().abs().max()) == 0.0
    # Normalised by the CONTINUOUS peak, which integer mode pairs only
    # approach - so the discrete maximum sits just under 1.
    assert 0.99 < float(kernel.max()) <= 1.0
    assert float(kernel.min()) >= 0.0

    peak_x = math.log(PL_B2 / PL_B1) / (PL_B2 - PL_B1)
    i, j = divmod(int(kernel.argmax()), kernel.shape[0])
    f_lo, f_hi = min(i, j) + 1, max(i, j) + 1
    x = (f_hi - f_lo) / (CRITICAL_BAND * f_lo)
    assert x == pytest.approx(peak_x, abs=0.03)


def test_the_peak_is_out_of_reach_on_a_small_grid():
    """Stated in the module docstring and worth pinning: below F_t ~ 24 the
    roughest reachable pair is the top one, and the term degenerates into
    'push mass upward'."""
    small = roughness_kernel(8)
    i, j = divmod(int(small.argmax()), small.shape[0])
    assert {i, j} == {6, 7}  # the top pair, because nothing closer is reachable
    assert float(small.max()) < 0.5
    big = roughness_kernel(48)
    assert float(big.max()) > 0.99


# ── the quantity ───────────────────────────────────────────────────────────


def test_roughness_is_scale_free_in_the_amplitudes():
    """Shrinking the field must not satisfy the term - only redistributing it
    can. Otherwise the cheapest response is to turn the field off."""
    m = _model(regularizers=["dissonance"])
    field = _find_field(m.head)
    term = m.criterion.dissonance
    _step(m)
    before = term.training_metrics()["dissonance"]
    with torch.no_grad():
        field.amplitudes.mul_(0.1)
    _step(m)
    assert term.training_metrics()["dissonance"] == pytest.approx(before, rel=1e-4)


def test_one_mode_alone_cannot_beat():
    m = _model(regularizers=["dissonance"])
    field = _find_field(m.head)
    term = m.criterion.dissonance
    with torch.no_grad():
        field.amplitudes.zero_()
        field.amplitudes[5, :] = 1.0
    _step(m)
    assert term.training_metrics()["dissonance"] == pytest.approx(0.0, abs=1e-5)


def test_the_scale_tops_out_at_one():
    """1 is the roughest spectrum the kernel allows - a band, not the roughest
    pair, which reads well below it - and nothing reads above it."""
    term = Dissonance()
    n = 64
    kernel = roughness_kernel(n)
    i, j = divmod(int(kernel.argmax()), n)
    pair = torch.zeros(n)
    pair[i] = pair[j] = 0.5
    assert float(term._roughness(pair)) < 0.75
    assert roughness_ceiling(n) > 1.4
    rng = torch.Generator().manual_seed(0)
    for _ in range(200):
        p = torch.rand(n, generator=rng) ** 6
        assert float(term._roughness(p / p.sum())) <= 1.0 + 1e-6


def test_a_flat_spectrum_is_not_maximally_rough():
    """Roughness is not spread: a sawtooth carries energy in every harmonic
    and is a consonant tone."""
    term = Dissonance()
    n = 64
    flat = torch.full((n,), 1.0 / n)
    kernel = roughness_kernel(n)
    i, j = divmod(int(kernel.argmax()), n)
    paired = torch.zeros(n)
    paired[i] = paired[j] = 0.5
    assert float(term._roughness(flat)) < float(term._roughness(paired))
    # And a spread spectrum is not automatically a rough one - the two
    # quantities are free to move apart, which is the point.
    assert float(term._roughness(flat)) < 0.25


def test_the_term_reaches_the_amplitudes():
    m = _model(regularizers=["dissonance"])
    field = _find_field(m.head)
    with torch.no_grad():  # give the dual something to work with
        m.criterion.dissonance.rho.fill_(0.0)
    out = _step(m)
    out.loss.backward()
    assert field.amplitudes.grad is not None
    assert float(field.amplitudes.grad.abs().sum()) > 0


# ── the dual ───────────────────────────────────────────────────────────────


def _steps(term, roughness, target, n):
    term.train()
    for _ in range(n):
        term._step_dual(target, roughness)


def test_the_multiplier_starts_effectively_off():
    """It has to earn its strength; softplus(0) would hand it most of the cap
    before a single step of evidence."""
    term = Dissonance()
    assert term._lambda() < 0.01


def test_the_multiplier_climbs_while_the_field_is_smoother_than_its_signal():
    term = Dissonance()
    _steps(term, roughness=0.1, target=0.4, n=500)
    assert float(term.rho) > RHO_INIT


def test_the_multiplier_relaxes_once_the_field_is_rougher():
    term = Dissonance()
    term.rho.fill_(0.0)
    _steps(term, roughness=0.6, target=0.2, n=500)
    assert float(term.rho) < 0.0


def test_the_cap_is_left_the_moment_the_constraint_stops_binding():
    """Anti-windup: however long the field sat below its target, rho never
    climbs past the cap, so one step on the other side is enough to leave it."""
    term = Dissonance()
    _steps(term, roughness=0.0, target=1.0, n=50_000)
    assert float(term.rho) == pytest.approx(RHO_MAX)
    assert term._lambda() == pytest.approx(LAMBDA_MAX)
    _steps(term, roughness=1.0, target=0.0, n=1)
    assert term._lambda() < LAMBDA_MAX


def test_no_step_before_a_target_exists():
    term = Dissonance()
    _steps(term, roughness=0.0, target=None, n=100)
    assert float(term.rho) == RHO_INIT
    assert float(term.target) < 0.0


def test_the_floor_is_recoverable():
    """A relaxed multiplier lands back where it started, not somewhere it needs
    a run's worth of steps to climb out of."""
    term = Dissonance()
    _steps(term, roughness=1.0, target=0.0, n=5_000)
    assert float(term.rho) == pytest.approx(RHO_INIT)


def test_the_multiplier_is_capped():
    term = Dissonance()
    with torch.no_grad():
        term.rho.fill_(50.0)
    assert term._lambda() == pytest.approx(LAMBDA_MAX)


def test_non_finite_readings_are_ignored():
    """A non-finite target leaves the EMA alone; a non-finite roughness skips
    the step."""
    term = Dissonance()
    _steps(term, roughness=0.2, target=0.3, n=1)
    before = (float(term.rho), float(term.target))
    _steps(term, roughness=0.2, target=float("nan"), n=1)  # target ignored
    _steps(term, roughness=float("nan"), target=0.3, n=1)  # no step
    assert float(term.rho) == pytest.approx(before[0] + 0.003 * 0.1, rel=1e-3)


def test_inference_does_not_step_the_dual():
    term = Dissonance()
    term.eval()
    term._step_dual(0.9, 0.0)
    assert float(term.rho) == RHO_INIT and float(term.target) < 0.0


def test_the_target_is_the_signals_roughness_on_the_fields_modes():
    """Measured on one field period, so FFT bin f is the field's mode f: a
    signal with two partials reads exactly that pair's roughness."""
    period, n = 64, 32
    t = torch.arange(period, dtype=torch.float32)
    wave = torch.cos(2 * math.pi * 20 * t / period) + torch.cos(
        2 * math.pi * 22 * t / period
    )
    h = wave.view(1, -1, 1).expand(2, -1, 8).contiguous()
    term = Dissonance()
    measured = term._measure_target(h, SimpleNamespace(T=period), n)
    pair = torch.zeros(n)
    pair[19] = pair[21] = 0.5
    assert measured == pytest.approx(float(term._roughness(pair)), rel=1e-4)


def test_rows_shorter_than_a_period_leave_the_target_unmeasured():
    term = Dissonance()
    short = torch.randn(2, 16, 8)
    assert term._measure_target(short, SimpleNamespace(T=64), 32) is None


def test_the_penalty_is_never_negative():
    m = _model(regularizers=["dissonance"])
    term = m.criterion.dissonance
    with torch.no_grad():
        term.rho.fill_(RHO_MAX)
    _step(m)
    assert 0.0 <= term.training_metrics()["dissonance_loss"] <= LAMBDA_MAX


def test_the_dual_survives_a_resume():
    term = Dissonance()
    _steps(term, roughness=0.1, target=0.4, n=50)
    fresh = Dissonance()
    fresh.load_state_dict(term.state_dict())
    assert float(fresh.rho) == float(term.rho)
    assert float(fresh.target) == float(term.target)


def test_an_older_checkpoint_still_loads():
    """Stale buffers are dropped, the target starts unobserved, and an
    out-of-range rho is clamped to the cap."""
    old = {
        "rho": torch.full((1,), 10.0),
        "seen": torch.ones(1),
        "ce_fast": torch.full((1,), 2.0),
        "ce_slow": torch.full((1,), 2.1),
        "gap_mean": torch.zeros(1),
        "gap_var": torch.ones(1),
    }
    term = Dissonance()
    term.load_state_dict(old)
    assert float(term.rho) == pytest.approx(RHO_MAX)
    assert float(term.target) < 0.0


# ── the probe ──────────────────────────────────────────────────────────────


def test_the_probe_measures_without_pushing():
    m = _model(regularizers=["dissonance_probe"])
    term = m.criterion.dissonance
    assert term.observe_only is True
    with torch.no_grad():
        term.rho.fill_(10.0)  # even wound up, it must contribute nothing
    _step(m)
    metrics = term.training_metrics()
    assert metrics["dissonance_modes"] > 0  # the readings are still there
    assert metrics["dissonance_loss"] == 0.0
    # The field still receives the main loss's gradient - it is on the forward
    # path - so the check that matters is that THIS term contributes no graph.
    penalty = term(torch.randn(2, 5, 32), None, head=m.head)
    assert float(penalty) == 0.0 and penalty.grad_fn is None


def test_both_profiles_are_registered():
    assert registry.lookup("regularizers", "dissonance") is Dissonance
    assert registry.lookup("regularizers", "dissonance_probe")().observe_only is True


# ── a head with no field ───────────────────────────────────────────────────


def test_the_term_is_inert_without_a_harmonic_field(capsys):
    m = _model(head_type="forward", regularizers=["dissonance"])
    out = _step(m)
    assert torch.isfinite(out.loss)
    assert m.criterion.dissonance.training_metrics() == {}
    assert "no harmonic field" in capsys.readouterr().out
