"""LinearPrior post-freeze re-solve: milestone-gated, damped, reject-if-worse."""

import torch

from praxis.heads.energy import (
    PRIOR_RESOLVE_GAP_DELTA,
    PRIOR_RESOLVE_VERIFY_STEPS,
    LinearPrior,
)


def _trained_prior(d=8, l=4, seed=0):
    torch.manual_seed(seed)
    p = LinearPrior(feature_dim=d, latent_dim=l)
    for _ in range(10):
        h = torch.randn(16, d)
        z = h @ torch.randn(d, l) * 0.1 + 0.01 * torch.randn(16, l)
        p.observe(h, z)
        p.solve()
    p.freeze()
    return p


def _baseline(p, gap=0.0, loss=1.0):
    # First call seeds loss EMA; second seeds the gap anchor.
    p.update_resolve(cond_gap=gap, energy_loss=loss, opt_step=0)
    p.update_resolve(cond_gap=gap, energy_loss=loss, opt_step=0)


def test_no_resolve_below_gap_milestone():
    p = _trained_prior()
    _baseline(p)
    w0 = p.W.clone()
    p.update_resolve(
        cond_gap=PRIOR_RESOLVE_GAP_DELTA * 0.9, energy_loss=1.0, opt_step=10
    )
    assert not bool(p.pending)
    torch.testing.assert_close(p.W, w0)


def test_resolve_applies_damped_blend_and_pends():
    p = _trained_prior()
    _baseline(p)
    # Shift the statistics so the fresh ridge differs from W.
    for _ in range(5):
        h = torch.randn(16, 8)
        p.observe(h, h @ torch.randn(8, 4))
    w0 = p.W.clone()
    p.update_resolve(cond_gap=PRIOR_RESOLVE_GAP_DELTA + 0.1, energy_loss=1.0, opt_step=10)
    assert bool(p.pending)
    assert not torch.allclose(p.W, w0)
    torch.testing.assert_close(p.W_prev, w0)  # restore point saved


def test_worse_loss_restores_old_w():
    p = _trained_prior()
    _baseline(p)
    for _ in range(5):
        p.observe(torch.randn(16, 8), torch.randn(16, 4))
    w0 = p.W.clone()
    p.update_resolve(cond_gap=1.0, energy_loss=1.0, opt_step=10)
    assert bool(p.pending)
    # Loss EMA degrades hard during the verify window.
    for s in range(PRIOR_RESOLVE_VERIFY_STEPS + 1):
        p.update_resolve(cond_gap=1.0, energy_loss=5.0, opt_step=11 + s)
    assert not bool(p.pending)
    torch.testing.assert_close(p.W, w0)  # rolled back
    assert float(p.resolves_rejected) == 1.0


def test_not_worse_loss_keeps_new_w():
    p = _trained_prior()
    _baseline(p)
    for _ in range(5):
        p.observe(torch.randn(16, 8), torch.randn(16, 4))
    w0 = p.W.clone()
    p.update_resolve(cond_gap=1.0, energy_loss=1.0, opt_step=10)
    w_new = p.W.clone()
    for s in range(PRIOR_RESOLVE_VERIFY_STEPS + 1):
        p.update_resolve(cond_gap=1.0, energy_loss=0.9, opt_step=11 + s)
    assert not bool(p.pending)
    torch.testing.assert_close(p.W, w_new)
    assert not torch.allclose(p.W, w0)
    assert float(p.resolves_kept) == 1.0


def test_old_checkpoint_loads_without_resolve_buffers():
    p = _trained_prior()
    sd = {k: v for k, v in p.state_dict().items() if "resolve" not in k
          and k not in ("W_prev", "pending", "pending_step", "gap_anchor",
                        "loss_ema", "ema_at_apply", "resolves_kept", "resolves_rejected")}
    p2 = LinearPrior(feature_dim=8, latent_dim=4)
    p2.load_state_dict(sd)
    torch.testing.assert_close(p2.W, p.W)
    assert not bool(p2.pending)
