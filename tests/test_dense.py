from itertools import product

import pytest
import torch

from praxis import registry

# Define test parameters
MODULE_CLASSES = list(registry.namespace("dense").values())
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


def test_peer_glu_value_branch_defaults_to_identity():
    """The `value` slot is opt-in: unfilled, peer_glu is byte-for-byte the old
    behaviour, so every config written before it is unaffected."""
    from types import SimpleNamespace

    import torch

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
    plain = registry.lookup("dense", "peer_glu")(cfg)
    with torch.no_grad():
        plain(torch.zeros(1, 4, 64))
    assert plain.act_value is None
    torch.manual_seed(0)
    dual = registry.lookup("dense", "peer_glu")(
        cfg, activation={"type": "single", "values": [cfg.activation], "linear": "gelu"}
    )
    with torch.no_grad():
        dual(torch.zeros(1, 4, 64))
    x = torch.randn(2, 16, 64)
    assert not torch.allclose(plain(x), dual(x), atol=1e-6)


SPLIT = {"type": "mix_split", "values": ["servant", "swish"]}
MIX = {"type": "mix_gated", "values": ["serpent", "swish", "linear"]}
LINEAR = {"type": "single", "values": ["gelu"], "linear": "gelu"}


def _peer_cfg(num_heads=4, activation="gelu"):
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_size=64,
        activation=activation,
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
        transform_type="none",
    )


def test_peer_split_partitions_the_bank_by_expert():
    """`peer_split` still splits the expert BANK, now through a `keyed`
    mixture rather than PEER-local logic.

    Three things together, because each alone passes for a wrong reason: the
    activation slot has to hold a keyed mixture (not a silently-ignored kwarg),
    the output has to differ from `peer_glu`, and the GLU's linear value branch
    has to survive - otherwise this is `peer_dual` wearing a different name.
    """
    from praxis.activations.mixture import ActivationMixture

    torch.manual_seed(0)
    plain = registry.lookup("dense", "peer_glu")(_peer_cfg())
    torch.manual_seed(0)
    split = registry.lookup("dense", "peer_glu")(_peer_cfg(activation=SPLIT))

    assert isinstance(split.act, ActivationMixture)
    assert split.act.type_name == "mix_split" and split.act.wants_keys
    assert split.act.names == ("servant", "swish")
    assert split.act_value is None

    x = torch.randn(2, 16, 64)
    plain.eval()
    split.eval()
    with torch.no_grad():
        assert not torch.allclose(plain(x), split(x), atol=1e-6)

    # The configs in this line run `num_heads: 1`. A head-axis split would be
    # impossible there; an expert-index split is not.
    torch.manual_seed(0)
    single = registry.lookup("dense", "peer_glu")(
        _peer_cfg(num_heads=1, activation=SPLIT)
    )
    single.eval()
    with torch.no_grad():
        assert single(x).shape == (2, 16, 64)


def test_peer_split_keys_the_activation_to_the_expert_not_the_rank():
    """The function class has to be a property of the bank row.

    Retrieval order is score order, so a k-axis split would hand one activation
    the high-scoring experts systematically; and with `offset_heads` False every
    head shares one bank, so a head-axis split would train the same row under
    two different functions. This asserts the partition is built from the expert
    index and therefore agrees with itself across heads and ranks.

    Asserted on the one-hot partition rather than on activation OUTPUTS,
    because `mix_split`'s periodic branch (Servant) carries per-feature
    parameters and legitimately produces a different value per slot - an output
    comparison would fail for a reason that says nothing about the split.
    """

    torch.manual_seed(0)
    m = registry.lookup("dense", "peer_glu")(_peer_cfg(activation=SPLIT))

    # The same expert, reached from two different (head, rank) slots, must take
    # the same branch. Constructing indices directly isolates the partition
    # from retrieval.
    front, back = 0, m.num_experts - 1
    indices = torch.tensor([[[[front, back] * 4] * 4]])  # [1, 1, 4, 8]
    weights = m.act._partition((indices % m.num_experts) / m.num_experts)

    took_front = weights[..., 0::2, :]
    took_back = weights[..., 1::2, :]
    assert (took_front == took_front[..., :1, :]).all()
    assert (took_back == took_back[..., :1, :]).all()
    assert not torch.equal(took_front[..., 0, :], took_back[..., 0, :])

    # And the shape is untouched, so nothing downstream is aware of the split.
    projected = torch.ones_like(indices, dtype=torch.float32)
    assert m._activate(projected, indices).shape == indices.shape


def test_peer_mix_routes_through_an_activation_bank():
    """`peer_mix` is `peer_glu` with a MIXTURE in the activation slot.

    Four things together, because each alone passes for a wrong reason: the
    slot has to actually hold the wrapper (not a silently-ignored kwarg), the
    output has to differ from `peer_glu`, the GLU's linear value branch has to
    survive (or this is `peer_dual` wearing a different name), and it has to
    work at `num_heads: 1`, which is what the configs in this line run.

    The base activation is gelu rather than silu on purpose: `swish` and `silu`
    are the SAME function under two registry keys, so a silu base would make one
    bank entry a duplicate and weaken the difference assertion.
    """
    from types import SimpleNamespace

    from praxis.activations.mixture import ActivationMixture

    def build(name, num_heads=4, **kw):
        torch.manual_seed(0)
        return registry.lookup("dense", name)(_peer_cfg(num_heads), **kw)

    plain = build("peer_glu")
    mixed = build("peer_glu", activation=MIX)

    assert isinstance(mixed.act, ActivationMixture)
    # Continuous, not the `keyed` partition `peer_split` runs - that is the one
    # variable between the two arms.
    assert mixed.act.type_name == "mix_gated" and not mixed.act.wants_keys
    # The GLU's linear value branch is untouched: nonlinear DEPTH is unchanged,
    # only the function class in the existing slot.
    assert mixed.act_value is None

    x = torch.randn(2, 16, 64)
    plain.eval()
    mixed.eval()
    with torch.no_grad():
        assert not torch.allclose(plain(x), mixed(x), atol=1e-6)

    single = build("peer_glu", num_heads=1, activation=MIX)
    single.eval()
    with torch.no_grad():
        assert single(x).shape == (2, 16, 64)


def test_peer_activation_override_defaults_to_config():
    """The `activation` override is opt-in. Without it PEER reads
    `config.activation`, so every config written before the override is
    unaffected."""
    from types import SimpleNamespace

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
    default = registry.lookup("dense", "peer_glu")(cfg)
    torch.manual_seed(0)
    explicit = registry.lookup("dense", "peer_glu")(cfg, activation="gelu")

    x = torch.randn(2, 16, 64)
    default.eval()
    explicit.eval()
    with torch.no_grad():
        assert torch.allclose(default(x), explicit(x), atol=1e-6)
