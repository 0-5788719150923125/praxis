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


def test_peer_split_activates_bank_halves_differently():
    """`act_alt` splits the expert BANK, not the head or rank axis, and does so
    without changing capacity.

    Four things are asserted together because each alone would pass for a wrong
    reason: parameter parity alone would pass if `act_alt` were ignored, and a
    plain "output differs" alone would pass if it had swapped the activation for
    every expert. The last two are the ones that pin the split - forcing both
    slots to the same function must reproduce `peer_glu` exactly, and the split
    must survive `num_heads: 1`, which is what this line's configs actually run
    and what rules the head axis out.

    The base activation is gelu rather than silu on purpose: `swish` and `silu`
    are the SAME function under two registry keys, so a silu base would make the
    split a genuine no-op and the difference assertion would fail for a reason
    that says nothing about the code.
    """
    from types import SimpleNamespace

    from praxis.dense import DENSE_REGISTRY

    def cfg(num_heads):
        return SimpleNamespace(
            hidden_size=64,
            activation="gelu",
            epsilon=1e-5,
            dropout=0.0,
            num_experts=4,
            num_heads=num_heads,
            k=8,
            num_queries=1,
            head_size=32,
            block_size=64,
            depth=6,
            num_layers=1,
        )

    def build(name, num_heads=4, **kw):
        torch.manual_seed(0)
        return DENSE_REGISTRY[name](cfg(num_heads), **kw)

    plain = build("peer_glu")
    split = build("peer_split")

    # A bank-half partition, and the GLU's linear value branch is untouched - so
    # this is not `peer_dual` wearing a different name.
    assert split.act_split == split.num_experts // 2
    assert isinstance(split.act_value, torch.nn.Identity)

    def count(m):
        return sum(p.numel() for p in m.parameters())

    assert count(plain) == count(split), "swish carries no parameters"

    x = torch.randn(2, 16, 64)
    plain.eval()
    split.eval()
    with torch.no_grad():
        assert not torch.allclose(plain(x), split(x), atol=1e-6)

    # Same function in both slots == no split at all. If `_activate` masked on
    # the wrong axis or misaligned the mask against `projected`, this is where
    # it shows.
    identical = build("peer_glu", act_alt="gelu")
    identical.eval()
    with torch.no_grad():
        assert torch.allclose(plain(x), identical(x), atol=1e-6)

    # The configs in this line run `num_heads: 1`. A head-axis split would be
    # impossible there; an expert-index split is not.
    single = build("peer_split", num_heads=1)
    single.eval()
    with torch.no_grad():
        assert single(x).shape == (2, 16, 64)


def test_peer_split_keys_the_activation_to_the_expert_not_the_rank():
    """The function class has to be a property of the bank row.

    Retrieval order is score order, so a k-axis split would hand one activation
    the high-scoring experts systematically; and with `offset_heads` False every
    head shares one bank, so a head-axis split would train the same row under
    two different functions. This asserts the mask is built from the expert
    index and therefore agrees with itself across heads and ranks.
    """
    from types import SimpleNamespace

    from praxis.dense import DENSE_REGISTRY

    cfg = SimpleNamespace(
        hidden_size=64,
        activation="gelu",
        epsilon=1e-5,
        dropout=0.0,
        num_experts=4,
        num_heads=4,
        k=8,
        num_queries=1,
        head_size=32,
        block_size=64,
        depth=6,
        num_layers=1,
    )
    torch.manual_seed(0)
    m = DENSE_REGISTRY["peer_split"](cfg)

    # Same expert, reached from two different (head, rank) slots, must take the
    # same branch. Constructing indices directly is the point: it isolates
    # `_activate` from retrieval.
    e_front, e_back = 0, m.num_experts - 1
    indices = torch.tensor([[[[e_front, e_back] * 4] * 4]])  # [1, 1, 4, 8]
    projected = torch.full_like(indices, 1, dtype=torch.float32)
    out = m._activate(projected, indices)

    front = out[..., 0::2]
    back = out[..., 1::2]
    assert torch.allclose(front, front[..., :1].expand_as(front))
    assert torch.allclose(back, back[..., :1].expand_as(back))
    assert not torch.allclose(front[..., 0], back[..., 0])
