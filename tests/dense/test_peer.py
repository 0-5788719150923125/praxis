from itertools import product

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.activations.depth import base_activation
from praxis.dense.glu import GatedLinearMLP
from praxis.dense.peer import (
    BANK_WIDTH_MULTIPLE,
    ROWS_PER_EXPERT,
    ROWS_PER_GLU_EXPERT,
    ParameterEfficientExpertRetrieval,
)

# ------------------------------------------------------------------------------
# peer
# ------------------------------------------------------------------------------
# PEER gated experts (praxis/dense/peer.py, the "dense" registry entry "peer_glu").
#
# ``test_dense.py`` already covers forward-pass shape for every registry entry. What
# needs its own pins here are the invariants that make ``peer_glu`` a fair comparison
# against ``peer`` rather than a disguised size increase.


WIDTHS = [64, 128, 256, 512]


def make_config(hidden_size=256, num_heads=16, activation="gelu", dropout=0.0):
    config = PraxisConfig()
    config.hidden_size = hidden_size
    config.num_heads = num_heads
    config.activation = activation
    config.dropout = dropout
    return config


def params(module):
    return sum(p.numel() for p in module.parameters())


def test_registry_exposes_the_gated_variant():
    assert "peer_glu" in registry.namespace("dense")
    module = registry.lookup("dense", "peer_glu")(make_config())
    assert isinstance(module, ParameterEfficientExpertRetrieval)
    assert module.glu is True
    assert module.gate is not None
    # And the ungated entry is untouched.
    assert registry.lookup("dense", "peer")(make_config()).gate is None


@pytest.mark.parametrize("hidden_size", WIDTHS)
def test_gated_experts_are_capacity_matched(hidden_size):
    """The third bank row per expert comes out of expert COUNT, not the
    parameter budget - otherwise `peer_glu` vs `peer` would measure 1.5x more
    parameters rather than the architecture change."""
    config = make_config(hidden_size=hidden_size)
    plain = ParameterEfficientExpertRetrieval(config, glu=False)
    gated = ParameterEfficientExpertRetrieval(config, glu=True)

    assert gated.num_experts < plain.num_experts  # breadth traded away
    assert gated.rows_per_expert == ROWS_PER_GLU_EXPERT
    assert plain.rows_per_expert == ROWS_PER_EXPERT
    # Rounding the bank to a perfect square is the only slack.
    ratio = params(gated) / params(plain)
    assert 0.9 < ratio < 1.1, (hidden_size, ratio, params(gated), params(plain))


@pytest.mark.parametrize("hidden_size", WIDTHS)
def test_bank_holds_its_ratio_to_the_dense_ffn(hidden_size):
    """The budgeting invariant the module documents, now under either expert
    form: the bank tracks the dense FFN it replaces at every width."""
    config = make_config(hidden_size=hidden_size)
    dense = params(GatedLinearMLP(config))
    for glu in (False, True):
        peer = ParameterEfficientExpertRetrieval(config, glu=glu)
        assert peer.num_experts == peer.num_keys**2  # product-key square
        budget = BANK_WIDTH_MULTIPLE * hidden_size * ROWS_PER_EXPERT
        assert abs(peer.num_experts * peer.rows_per_expert - budget) / budget < 0.25
        assert params(peer) > dense  # a retrieval bank is the bigger object


def test_gated_forward_differs_from_ungated():
    """Sanity that the gate actually participates: zeroing it must change the
    output, and the gated path must not silently reduce to the plain one."""
    torch.manual_seed(0)
    config = make_config(hidden_size=64, num_heads=4)
    module = ParameterEfficientExpertRetrieval(config, glu=True).eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        before = module(x, current_depth=0).clone()
        module.gate.weight.zero_()
        after = module(x, current_depth=0)
    assert not torch.allclose(before, after)


def test_gradients_reach_every_bank():
    config = make_config(hidden_size=64, num_heads=4, dropout=0.1)
    module = ParameterEfficientExpertRetrieval(config, glu=True)
    x = torch.randn(3, 12, 64, requires_grad=True)
    module(x, current_depth=0).sum().backward()
    for name in ("down", "gate", "up", "keys"):
        tensor = getattr(module, name)
        weight = tensor if name == "keys" else tensor.weight
        assert weight.grad is not None, name
        assert float(weight.grad.abs().sum()) > 0.0, name
    assert float(x.grad.abs().sum()) > 0.0


def test_repr_names_the_expert_form():
    """The blueprint tab renders __repr__, so the variant has to be visible
    there - two runs whose configs differ only in ffn_type would otherwise be
    indistinguishable in the architecture view."""
    plain = ParameterEfficientExpertRetrieval(make_config(), glu=False)
    gated = ParameterEfficientExpertRetrieval(make_config(), glu=True)
    assert "expert=rank1" in plain.extra_repr()
    assert "expert=glu" in gated.extra_repr()


def test_repr_is_a_field_list():
    """``print(model)`` is a field listing: the product-key grid is a field of
    its own, not an annotation glued to ``num_experts``."""
    peer = ParameterEfficientExpertRetrieval(make_config())
    fields = [part.strip() for part in peer.extra_repr().split(",")]
    for field in fields:
        key, sep, value = field.partition("=")
        assert sep, f"not a key=value field: {field!r}"
        assert key.isidentifier(), f"not an identifier: {key!r}"
        assert value.strip(), f"no value for {key!r}"

    assert f"num_experts={peer.num_experts}" in fields
    assert f"num_keys={peer.num_keys}" in fields
    # The invariant the parenthetical was trying to convey, still visible.
    assert peer.num_keys**2 == peer.num_experts


def test_explicit_expert_count_overrides_the_budget():
    """Registry profiles may pin the bank; the rows factor must not fight it."""
    module = ParameterEfficientExpertRetrieval(make_config(), num_experts=256, glu=True)
    assert module.num_experts == 256
    assert module.num_keys == 16


def test_sparse_gated_banks_stay_sparse():
    """Sparse gradients come from the embedding lookup, so the gate bank has to
    carry the same flag or the sparse path densifies through it."""
    module = ParameterEfficientExpertRetrieval(
        make_config(hidden_size=64, num_heads=4), glu=True, sparse=True
    )
    assert module.down.sparse and module.up.sparse and module.gate.sparse
    assert module._gathers()  # sparse always gathers
    out = module(torch.randn(2, 8, 64), current_depth=0)
    assert out.shape == (2, 8, 64)


# ── odd hidden_size ─────────────────────────────────────────────────────────
# The query net emits `key_dims * num_heads * 2` rather than halving the model
# width, so any hidden_size works. These pin that.

ODD_WIDTHS = [33, 65, 111, 257]


@pytest.mark.parametrize("hidden_size", ODD_WIDTHS)
@pytest.mark.parametrize("expert", ["peer", "peer_glu"])
def test_odd_hidden_size_builds_and_runs(hidden_size, expert):
    config = make_config(hidden_size=hidden_size, num_heads=4)
    module = registry.lookup("dense", expert)(config)

    x = torch.randn(2, 8, hidden_size, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("expert", ["peer", "peer_glu"])
def test_odd_width_matches_its_even_neighbour(expert):
    """Parity is not a cliff: 111 and 112 size the bank identically."""
    odd = registry.lookup("dense", expert)(make_config(hidden_size=111, num_heads=4))
    even = registry.lookup("dense", expert)(make_config(hidden_size=112, num_heads=4))
    assert odd.num_experts == even.num_experts
    assert odd.key_dims == even.key_dims
    assert odd.num_keys == even.num_keys


def test_key_dims_floor_holds_at_a_tiny_odd_width():
    """hidden_size // (2 * num_heads) floors to 0 here; MIN_KEY_DIMS catches it."""
    from praxis.dense.peer import MIN_KEY_DIMS

    module = registry.lookup("dense", "peer")(make_config(hidden_size=3, num_heads=4))
    assert module.key_dims == MIN_KEY_DIMS
    x = torch.randn(2, 4, 3)
    assert module(x).shape == x.shape


# --------------------------------------------------------------- initialization


@pytest.mark.parametrize("glu", [True, False])
def test_expert_banks_are_initialized_by_true_fan_in(glu):
    """Bank size is a LOOKUP dimension, not a fan, and must not set the scale.

    Xavier reads ``fan_out`` off the tensor shape, which for these banks is
    ``num_experts`` - so the init scale fell as the bank grew, coupling two
    things that have no business being coupled. See ``init_weights``.
    """
    import math

    module = ParameterEfficientExpertRetrieval(make_config(), glu=glu)
    d = module.hidden_size

    # down/gate project the input onto one expert vector: fan-in is hidden_size.
    assert module.down.weight.std().item() == pytest.approx(d**-0.5, rel=0.15)
    if module.gate is not None:
        assert module.gate.weight.std().item() == pytest.approx(d**-0.5, rel=0.15)

    # up is summed over the retrieved experts: fan-in is that fan-out.
    expected_up = (module.num_heads * module.k) ** -0.5
    assert module.up.weight.std().item() == pytest.approx(expected_up, rel=0.15)

    # Product keys match the reference implementation.
    assert module.keys.std().item() == pytest.approx(0.02, rel=0.2)

    # And the scale must NOT track the bank size, which is what Xavier did.
    xavier_std = math.sqrt(2.0 / (d + module.num_experts * module.num_sets))
    assert module.down.weight.std().item() > 1.2 * xavier_std


def test_output_scale_is_stable_across_widths():
    """The property the fan-in derivation buys: PEER contributes at the same
    relative scale whatever the bank size, instead of going quiet as it grows.
    """
    torch.manual_seed(0)
    ratios = []
    for hidden_size in (64, 128, 256, 512):
        module = ParameterEfficientExpertRetrieval(
            make_config(hidden_size=hidden_size), glu=True
        ).eval()
        x = torch.randn(4, 32, hidden_size)
        with torch.no_grad():
            y = module(x)
        ratios.append(float(y.std()) / float(x.std()))

    # Every width lands in the same band, and none is near-silent.
    assert min(ratios) > 0.2, f"PEER is attenuating its input: {ratios}"
    assert max(ratios) < 3.0, f"PEER is amplifying its input: {ratios}"
    assert max(ratios) / min(ratios) < 2.5, f"scale tracks bank size: {ratios}"


@pytest.mark.parametrize("expert", ["peer", "peer_glu"])
def test_training_forward_is_causal_and_row_independent(expert):
    """Retrieval is per token: in TRAINING mode, changing one position must
    leave every earlier position and every other row untouched. A batch
    statistic anywhere on the query path breaks both, and top-k retrieval
    turns even a small shift into different experts."""
    import copy

    torch.manual_seed(0)
    module = registry.lookup("dense", expert)(
        make_config(hidden_size=64, num_heads=4)
    ).train()
    # Earlier batches may move the query norm's running statistics; give it some
    # history, then compare two forwards from the same state.
    with torch.no_grad():
        for _ in range(3):
            module(torch.randn(3, 12, 64) + 2.0, 0)
    x = torch.randn(3, 12, 64)
    p = 7
    xp = x.clone()
    xp[0, p] += torch.randn(64)
    with torch.no_grad():
        a = copy.deepcopy(module)(x, 0)
        b = copy.deepcopy(module)(xp, 0)
    torch.testing.assert_close(a[0, :p], b[0, :p], rtol=0.0, atol=0.0)
    torch.testing.assert_close(a[1:], b[1:], rtol=0.0, atol=0.0)
    assert not torch.allclose(a[0, p], b[0, p])


def _retrieved(module, x):
    """Distinct experts the batch retrieves, by the module's own query path."""
    queries = module.queries(x)
    sim = torch.einsum("p b n h d, h k p d -> p b n h k", queries, module.keys)
    scores, idx = sim.topk(module.k, dim=-1)
    pairs = (scores[0].unsqueeze(-1) + scores[1].unsqueeze(-2)).flatten(-2)
    grid = (idx[0].unsqueeze(-1) * module.num_keys + idx[1].unsqueeze(-2)).flatten(-2)
    return grid.gather(-1, pairs.topk(module.k, dim=-1)[1]).unique().numel()


def test_query_norm_keeps_a_shared_offset_from_collapsing_retrieval():
    """A direction every token shares hands every token the same experts; the
    query norm's running statistics remove it."""
    torch.manual_seed(0)
    config = make_config(hidden_size=64, num_heads=1)
    module = registry.lookup("dense", "peer_glu")(config)
    offset = 20.0 * torch.randn(64)

    def batch():
        return offset + torch.randn(4, 32, 64)

    with torch.no_grad():
        fresh = _retrieved(module.eval(), batch())
        module.train()
        for _ in range(60):
            module(batch(), 0)
        fitted = _retrieved(module.eval(), batch())
    assert fresh <= 2 * module.k
    assert fitted > 4 * fresh


def test_query_norm_commits_its_statistics_once_per_forward():
    """Every recurrent pass contributes, and none of it lands until the next
    forward begins - a pass never reads statistics that include later tokens."""
    module = registry.lookup("dense", "peer")(make_config(hidden_size=64, num_heads=4))
    norm = module.queries[0]
    module.train()
    with torch.no_grad():
        for depth in range(3):
            module(torch.randn(2, 8, 64) + 5.0, depth)
            assert float(norm.running_mean.abs().max()) == 0.0
            assert int(norm.num_batches_tracked) == 0
        module(torch.randn(2, 8, 64), 0)
    assert int(norm.num_batches_tracked) == 1
    assert float(norm.running_mean.mean()) > 0.3


def test_query_norm_changes_nothing_in_eval():
    module = registry.lookup("dense", "peer")(make_config(hidden_size=64, num_heads=4))
    module.eval()
    with torch.no_grad():
        module(torch.randn(2, 8, 64) + 5.0, 0)
        module(torch.randn(2, 8, 64), 0)
    assert int(module.queries[0].num_batches_tracked) == 0


def test_checkpoints_without_the_query_norm_still_load():
    """Checkpoints written without the norm hold the query Linear at
    ``queries.0``; it loads into ``queries.1`` and the norm starts fresh."""
    torch.manual_seed(0)
    source = registry.lookup("dense", "peer")(make_config(hidden_size=64, num_heads=4))
    state = {
        key.replace("queries.1.", "queries.0."): value
        for key, value in source.state_dict().items()
        if not key.startswith("queries.0.")
    }
    target = registry.lookup("dense", "peer")(make_config(hidden_size=64, num_heads=4))
    result = target.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.testing.assert_close(target.queries[1].weight, source.queries[1].weight)
    assert int(target.queries[0].num_batches_tracked) == 0


# ------------------------------------------------------------------------------
# dense
# ------------------------------------------------------------------------------


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

    assert isinstance(base_activation(split.act), ActivationMixture)
    assert base_activation(split.act).type_name == "mix_split"
    assert split.act.wants_keys
    assert base_activation(split.act).names == ("servant", "swish")
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
    weights = base_activation(m.act)._partition(
        (indices % m.num_experts) / m.num_experts
    )

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

    assert isinstance(base_activation(mixed.act), ActivationMixture)
    # Continuous, not the `keyed` partition `peer_split` runs - that is the one
    # variable between the two arms.
    assert base_activation(mixed.act).type_name == "mix_gated"
    assert not mixed.act.wants_keys
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
