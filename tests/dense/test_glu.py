import pytest
import torch

from praxis import registry


def test_value_slot_activates_the_glus_linear_half():
    """Filling the `value` slot makes both halves nonlinear, so they multiply.

    It is a config SLOT rather than a class or a constructor flag, so this pins
    what makes that legitimate: an unfilled slot must leave `glu` byte-for-byte
    unchanged, and the two must match on parameter count so a swap between them
    is a clean one-variable change.
    """
    from types import SimpleNamespace

    import torch

    cfg = SimpleNamespace(
        hidden_size=64, activation="serpent", epsilon=1e-5, dropout=0.0
    )
    dual_cfg = SimpleNamespace(
        hidden_size=64,
        activation={"type": "single", "values": ["serpent"], "linear": "gelu"},
        epsilon=1e-5,
        dropout=0.0,
    )
    glu = registry.lookup("dense", "glu")(cfg)
    dual = registry.lookup("dense", "glu")(dual_cfg)
    x = torch.randn(2, 16, 64)
    with torch.no_grad():  # serpent carries lazy params until first forward
        glu(x)
        dual(x)
    assert sum(p.numel() for p in glu.parameters()) == sum(
        p.numel() for p in dual.parameters()
    )
    y = dual(x)
    y.pow(2).mean().backward()
    assert y.shape == x.shape
    assert all(p.grad is not None for p in dual.parameters())
    # The value half is genuinely activated, unlike a GLU's linear branch.
    assert glu.act_value is None
    assert dual.act_value is not None
    assert type(dual.act_value) is not type(dual.act)
