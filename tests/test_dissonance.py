"""The dissonance term, and the instrument it was built beside.

`harmonic_kl` is named for a claim about the harmonic basis but EMAs whatever
the head hands over as its readout - HALO's centroids on the prismatic line -
so a near-zero drift there says nothing about the field. These tests pin what
each term actually watches, pin the roughness kernel to the published
Plomp-Levelt curve, and pin the dual to the one thing that is supposed to bound
it: whether the task is still improving.

The dual tests below carry the -v postmortem. A controller stepping on
`sign(ce_fast < ce_slow)` is stationary only where that sign is right 80% of the
time, which is a property of the loss's noise and not of the constraint, so it
fell to its clamp and stayed. The replacements pin the two properties that
failure needed: no drift under a trendless loss, and a floor that a run can
climb back out of.
"""

import math

import pytest
import torch

from praxis.losses import REGULARIZER_REGISTRY
from praxis.losses.harmonic_kl import HarmonicKLRegularizer
from praxis.losses.dissonance import (
    CRITICAL_BAND,
    LAMBDA_MAX,
    PL_B1,
    PL_B2,
    RHO_INIT,
    Dissonance,
    _find_field,
    roughness_kernel,
)


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


def test_all_mass_on_the_roughest_pair_reads_one():
    term = Dissonance()
    n = 64
    kernel = roughness_kernel(n)
    i, j = divmod(int(kernel.argmax()), n)
    p = torch.zeros(n)
    p[i] = p[j] = 0.5
    assert float(term._roughness(p)) == pytest.approx(1.0, abs=1e-3)


def test_a_flat_spectrum_is_not_maximally_rough():
    """The whole reason this is not the spread term it replaced: a sawtooth
    carries energy in every harmonic and is a consonant tone."""
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
    assert float(term._roughness(flat)) < 0.5


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


def test_the_multiplier_starts_effectively_off():
    """It has to earn its strength; softplus(0) would hand it most of the cap
    before a single step of evidence."""
    term = Dissonance()
    assert term._lambda() < 0.01


def test_the_multiplier_climbs_while_the_task_improves():
    term = Dissonance()
    start = float(term.rho)
    for value in (5.0, 4.0, 3.0, 2.0, 1.0):
        term._step_dual(torch.tensor(value))
    assert float(term.rho) > start


def test_the_multiplier_retreats_faster_than_it_climbs():
    """A run that starts to break has to pull the term off itself faster than
    it put it on, or the retreat arrives after the damage.

    Mirror-image traces, so the t-statistic has the same magnitude in both
    directions and only the eta differs.
    """
    falling = [10.0 - 0.02 * i for i in range(400)]
    up = Dissonance()
    for value in falling:
        up._step_dual(torch.tensor(value))
    gained = float(up.rho) - RHO_INIT

    down = Dissonance()
    down.rho.fill_(10.0)  # off the floor, so the retreat has room to show
    for value in reversed(falling):
        down._step_dual(torch.tensor(value))
    lost = 10.0 - float(down.rho)

    assert gained > 0 and lost > 0
    assert lost == pytest.approx(4.0 * gained, rel=0.05)


def test_a_trendless_loss_does_not_drive_the_multiplier_to_its_floor():
    """The -v failure. Noise with no trend must leave the dual where it is:
    a controller whose resting point is set by the up/down ratio rather than by
    the constraint collapses on every real run, because the sign of a fast-vs-
    slow EMA comparison is noise long before the loss stops improving.
    """
    rng = torch.Generator().manual_seed(0)
    term = Dissonance()
    term.rho.fill_(0.0)
    for _ in range(3000):
        value = 10.0 * torch.exp(0.25 * torch.randn((), generator=rng))
        term._step_dual(value)
    # Free to wander; what it must not do is walk to the floor and stay.
    assert float(term.rho) > RHO_INIT + 1.0


def test_a_genuinely_improving_run_engages_the_term():
    """The other half of the -v postmortem. Backing off under noise is only
    correct if a real trend still gets through: a loss that halves under 25%
    per-step noise has to move the multiplier well off its floor, or the term
    is just off with extra steps.
    """
    rng = torch.Generator().manual_seed(0)
    term = Dissonance()
    for i in range(20000):
        clean = 12.0 * math.exp(-i / 15000)
        noise = torch.exp(0.25 * torch.randn((), generator=rng))
        term._step_dual(clean * noise)
    assert term._lambda() > 0.5


def test_the_controller_is_indifferent_to_the_loss_scale():
    """Same trend, two scales. Dividing by the gap's own spread is what makes
    the signal dimensionless; without it the step size would track how large
    the loss happens to be."""

    def final_rho(scale):
        term = Dissonance()
        for i in range(400):
            term._step_dual(torch.tensor(scale * (10.0 - 0.02 * i)))
        return float(term.rho)

    assert final_rho(1.0) == pytest.approx(final_rho(100.0), rel=1e-3)


def test_the_floor_is_recoverable():
    """A collapsed multiplier lands back where it started, not somewhere it
    needs a run's worth of steps to climb out of."""
    term = Dissonance()
    for i in range(2000):  # a loss that only rises
        term._step_dual(torch.tensor(1.0 + 0.01 * i))
    assert float(term.rho) == pytest.approx(RHO_INIT)


def test_the_multiplier_is_capped():
    term = Dissonance()
    with torch.no_grad():
        term.rho.fill_(50.0)
    assert term._lambda() == pytest.approx(LAMBDA_MAX)


def test_a_non_finite_loss_does_not_move_the_dual():
    term = Dissonance()
    term._step_dual(torch.tensor(3.0))
    before = float(term.rho)
    term._step_dual(torch.tensor(float("nan")))
    assert float(term.rho) == before


def test_the_dual_survives_a_resume():
    """A multiplier that reset to zero on resume would restart the ratchet
    every time the run is picked up."""
    term = Dissonance()
    for value in (5.0, 4.0, 3.0):
        term._step_dual(torch.tensor(value))
    fresh = Dissonance()
    fresh.load_state_dict(term.state_dict())
    assert float(fresh.rho) == float(term.rho)
    assert float(fresh.ce_slow) == float(term.ce_slow)


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
    assert REGULARIZER_REGISTRY["dissonance"] is Dissonance
    assert REGULARIZER_REGISTRY["dissonance_probe"]().observe_only is True


# ── a head with no field ───────────────────────────────────────────────────


def test_the_term_is_inert_without_a_harmonic_field(capsys):
    m = _model(head_type="forward", regularizers=["dissonance"])
    out = _step(m)
    assert torch.isfinite(out.loss)
    assert m.criterion.dissonance.training_metrics() == {}
    assert "no harmonic field" in capsys.readouterr().out
