from itertools import product

import pytest
import torch

from praxis.dense import DENSE_REGISTRY

# Define test parameters
MODULE_CLASSES = list(DENSE_REGISTRY.values())
HIDDEN_SIZES = [64, 256]

# Create parameter combinations
MODULE_PARAMS = list(product(MODULE_CLASSES, HIDDEN_SIZES))


@pytest.fixture(params=MODULE_PARAMS)
def module_setup(request, config):
    """
    Parametrized fixture that provides both module and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (module instance, hidden_size)
    """
    module_class, hidden_size = request.param
    # Use the update method from our existing config
    setattr(config, "hidden_size", hidden_size)
    module = module_class(config)
    return module, hidden_size


def test_forward_pass(module_setup):
    """Test using parametrized module and dimensions."""
    module, hidden_size = module_setup
    batch_size = 32
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = module(x)
    assert output.shape == (batch_size, seq_len, hidden_size)


def test_dual_act_multiplies_two_activated_halves():
    """Not a GLU: both halves are nonlinear and they multiply. Parameter count
    must match GatedLinearMLP so a swap is a clean one-variable change."""
    import torch
    from types import SimpleNamespace
    from praxis.dense import DENSE_REGISTRY

    cfg = SimpleNamespace(
        hidden_size=64, activation="serpent", epsilon=1e-5, dropout=0.0
    )
    glu = DENSE_REGISTRY["glu"](cfg)
    dual = DENSE_REGISTRY["dual_act"](cfg)
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
    # The gate half is genuinely activated, unlike a GLU's linear branch.
    assert not isinstance(dual.act_gate, torch.nn.Identity)
    assert type(dual.act_gate) is not type(dual.act)


def test_peer_glu_value_branch_defaults_to_identity():
    """`act_value` is opt-in: without it, peer_glu is byte-for-byte the old
    behaviour, so every config written before it is unaffected."""
    import torch
    from types import SimpleNamespace
    from praxis.dense import DENSE_REGISTRY

    cfg = SimpleNamespace(
        hidden_size=64,
        activation="serpent",
        epsilon=1e-5,
        dropout=0.0,
        num_experts=4,
        num_heads=1,
        k=8,
        num_queries=1,
        head_size=32,
        block_size=64,
        depth=6,
        num_layers=1,
    )
    torch.manual_seed(0)
    plain = DENSE_REGISTRY["peer_glu"](cfg)
    with torch.no_grad():
        plain(torch.zeros(1, 4, 64))
    assert isinstance(plain.act_value, torch.nn.Identity)
    torch.manual_seed(0)
    dual = DENSE_REGISTRY["peer_glu"](cfg, act_value="gelu")
    with torch.no_grad():
        dual(torch.zeros(1, 4, 64))
    x = torch.randn(2, 16, 64)
    assert not torch.allclose(plain(x), dual(x), atol=1e-6)
