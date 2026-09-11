"""Per-objective gradients at the trunk output, the anchor they compare to, and when they can be taken."""

import torch

from praxis.losses.trunk_grads import resolve_anchor, trunk_gradients, usable


def test_resolve_anchor_preference():
    """main whenever it is live; the surgical head's arm_surgery row when main
    cannot reach the trunk; nothing when neither can."""
    h = torch.randn(4, requires_grad=True)
    both = {"main": (h * 2).sum(), "arm_surgery": (h * 3).sum()}
    assert resolve_anchor(trunk_gradients(both, h)) == "main"

    surgical = {"arm_surgery": (h * 3).sum(), "harmonic_kl": (h * 0.5).sum()}
    assert resolve_anchor(trunk_gradients(surgical, h)) == "arm_surgery"

    assert resolve_anchor(trunk_gradients({"harmonic_kl": h.sum()}, h)) is None


def test_a_term_with_no_path_to_the_trunk_gets_no_row():
    """Parameter-only terms do not compete for the shared representation, so
    absence is the answer rather than a zero. Terms that cannot be
    differentiated at all are dropped the same way instead of raising."""
    h = torch.randn(4, requires_grad=True)
    p = torch.randn(4, requires_grad=True)
    consumed = h.pow(2).sum()
    consumed.backward()  # frees its graph: autograd.grad would raise

    live = trunk_gradients(
        {
            "main": (h * 2).sum(),
            "centers_rms": (p**2).sum(),
            "constant": torch.tensor(1.0),
            "number": 0.5,
            "flat": (h * 0.0).sum(),
            "consumed": consumed,
        },
        h,
    )

    assert set(live) == {"main"}
    assert torch.allclose(live["main"], torch.full((4,), 2.0))


def test_usable_only_where_a_gradient_can_be_taken():
    h = torch.randn(4, requires_grad=True)
    assert usable(h)
    assert not usable(torch.randn(4))
    assert not usable(None)
    with torch.no_grad():
        assert not usable(h)
