"""Modular SMEAR: the paper's granularity, over a shared block plus deviations.

Not a new method - SMEAR (arxiv 2306.03745) applied the way the paper applies
it, which praxis/routers/smear.py does not. What is pinned here is exactly that:

  * targets are discovered per MODULE, not per block, and each gets its own
    coefficient row (the paper puts a router on each inserted adapter);
  * Linear targets route PER EXAMPLE (the paper's routing granularity), while
    elementwise targets stay on the batch mean (the paper does not treat
    layernorm parameters as experts either);
  * expert dropout is present, because that is the paper's load-balancing
    mechanism and omitting it collapsed every target to one-hot;
  * ``MERGE_OPAQUE`` subtrees and reference-tied parameters are never merged -
    the two structural exclusions PEER and the Titans memory rely on;
  * the router is EXACTLY identity at init, so a config swap is a clean A/B;
  * the merge really is the paper's merge in a base-plus-deviation basis, i.e.
    it equals the convex combination of the implied experts;
  * the shared trunk receives full gradient whatever the routing does, which is
    the property VEAR's dead experts did not have.
"""

import pytest
import torch
import torch.nn as nn

from praxis.routers import ROUTER_REGISTRY
import torch.nn.utils.parametrize as parametrize

from praxis.routers.smear import SMEAR, Deviation, MergedLinear, _get_param
from praxis.routers.vear import VEAR
from praxis.routers.targeting import TARGET_PROFILES, discover_targets


class Opaque(nn.Module):
    """Stands in for PEER: routes itself, so it opts out of merging."""

    MERGE_OPAQUE = True

    def __init__(self, d=32):
        super().__init__()
        self.big = nn.Linear(d, d)

    def forward(self, x):
        return self.big(x)


class Block(nn.Module):
    """A miniature of the real decoder block's shape."""

    def __init__(self, d=32):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(d, d)
        self.attn.output = nn.Linear(d, d)
        self.attn.kappa = nn.Parameter(torch.randn(2, d) * 0.02)
        self.attn_norm = nn.LayerNorm(d)
        self.ffn = Opaque(d)
        self.ffn_norm = nn.LayerNorm(d)

    def forward(
        self,
        inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        current_depth=0,
        block_ids=None,
        router_weights=None,
        positions=None,
    ):
        h = self.attn_norm(inputs)
        h = self.attn.output(self.attn.qkv(h) * self.attn.kappa[0])
        h = self.ffn(self.ffn_norm(inputs + h))
        return h, past_key_values, current_state, 0.0


class Cfg:
    hidden_size = 32
    depth = 6
    num_experts = 1


def make(cls=SMEAR, n=4, profile="all", depth=6, dropout=0.0):
    cfg = Cfg()
    cfg.depth = depth
    block = Block(cfg.hidden_size)
    router = cls(cfg, block=block, num_experts=n, target_profile=profile, verbose=False)
    # Dropout is stochastic; most assertions below want the router's own
    # coefficients rather than a particular draw, so it is off unless asked for.
    router.EXPERT_DROPOUT = dropout
    return router, block


def router_args(block, x, depth=0):
    return (block, x, None, None, None, depth, None)


# --- targeting ---------------------------------------------------------------


def test_opaque_subtree_is_never_targeted():
    _, block = make()
    groups, skipped = discover_targets(block, TARGET_PROFILES["all"])
    names = {g.name for g in groups}
    assert not any(n.startswith("ffn.") or n == "ffn" for n in names)
    assert skipped["opaque"] == 2  # Opaque.big weight + bias


def test_tied_parameters_are_merged_at_most_once():
    _, block = make()
    block.attn.output.weight = block.attn.qkv.weight  # tie by reference
    groups, skipped = discover_targets(block, TARGET_PROFILES["all"])
    flat = [p for g in groups for p in g.params]
    assert flat.count("attn.qkv.weight") + flat.count("attn.output.weight") == 1
    assert skipped["shared"] == 1


def test_frozen_parameters_are_skipped():
    _, block = make()
    block.attn_norm.weight.requires_grad_(False)
    _, skipped = discover_targets(block, TARGET_PROFILES["all"])
    assert skipped["frozen"] == 1


def test_granularity_is_per_module_not_per_block():
    router, _ = make()
    # attn (bare kappa), attn.qkv, attn.output, attn_norm, ffn_norm
    assert len(router.targets) == 5
    assert router.router.out_features == 5 * 4


# --- identity at init --------------------------------------------------------


def test_merge_is_exactly_identity_at_init():
    """Both mechanisms must be exact no-ops at step 0."""
    router, block = make()
    x = torch.randn(3, 7, Cfg.hidden_size)
    merge, _ = router._coefficients(x, 0)

    base = {n: p.detach().clone() for n, p in block.named_parameters()}
    with router._coefficient_scope(merge):
        # Parametrized reads return the base exactly (banks are zero).
        for dev in router.deviations:
            assert torch.all(dev.bank == 0)
        assert torch.equal(block.attn_norm.weight, base["attn_norm.parametrizations.weight.original"])
        assert torch.equal(block.attn.kappa, base["attn.parametrizations.kappa.original"])
        # Wrapped Linears reduce to the Linear they replaced.
        for label, wrapper in router.wrappers.items():
            probe = torch.randn(3, 7, wrapper.in_features)
            torch.testing.assert_close(
                wrapper(probe),
                torch.nn.functional.linear(probe, wrapper.weight, wrapper.bias),
                msg=f"{label} moved at init",
            )


def test_forward_matches_bare_block_at_init():
    router, block = make()
    x = torch.randn(2, 5, Cfg.hidden_size)
    with torch.no_grad():
        want = block(x)[0]
        got = router(*router_args(block, x))[0]
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("cls", [SMEAR])
def test_depth_bias_is_identity_at_init(cls):
    router, block = make(cls)
    assert torch.all(router.depth_bias.weight == 0)
    x = torch.randn(2, 5, Cfg.hidden_size)
    w0, _ = router._coefficients(x, 0)
    w5, _ = router._coefficients(x, 5)
    torch.testing.assert_close(w0, w5)


def test_depth_bias_wraps_past_the_table():
    router, block = make(SMEAR, depth=6)
    x = torch.randn(2, 5, Cfg.hidden_size)
    router(*router_args(block, x, depth=9))  # must not raise


# --- the merge is SMEAR in a different basis ---------------------------------


def test_merge_equals_convex_combination_of_implied_experts():
    """base + sum_e w_e delta_e == sum_e w_e (base + delta_e), on a Deviation."""
    router, block = make()
    dev = router.deviations[0]
    nn.init.normal_(dev.bank, std=0.05)
    base = torch.randn_like(dev.bank[0])
    w = torch.softmax(torch.randn(router.num_experts), dim=-1)

    dev._coeff = w
    got = dev(base)
    dev._coeff = None

    want = sum(w[e] * (base + dev.bank[e]) for e in range(router.num_experts))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)


def test_coefficients_sum_to_one_per_target():
    router, _ = make()
    x = torch.randn(4, 6, Cfg.hidden_size)
    merge, _ = router._coefficients(x, 0)
    assert merge.shape == (4, len(router.targets), 4), "coefficients must stay per-example"
    torch.testing.assert_close(merge.sum(dim=-1), torch.ones(4, len(router.targets)))


def test_sharpening_concentrates_without_changing_shape():
    sharp, _ = make(VEAR)
    plain, _ = make(SMEAR)
    sharp.router.load_state_dict(plain.router.state_dict())
    sharp.router_norm.load_state_dict(plain.router_norm.state_dict())
    x = torch.randn(8, 6, Cfg.hidden_size)
    ws, _ = sharp._coefficients(x, 0)
    wp, _ = plain._coefficients(x, 0)
    assert ws.shape == wp.shape
    assert ws.max(dim=-1).values.mean() >= wp.max(dim=-1).values.mean()


# --- gradient behaviour -------------------------------------------------------


def test_shared_trunk_gets_full_gradient_under_collapsed_routing():
    """Whatever the routing does, the shared base learns."""
    router, block = make()
    with torch.no_grad():  # force near one-hot onto expert 0
        router.router.bias.zero_()
        router.router.bias.view(len(router.targets), 4)[:, 0] = 50.0
    x = torch.randn(2, 5, Cfg.hidden_size)
    router(*router_args(block, x))[0].sum().backward()

    assert block.attn.qkv.weight.grad.abs().sum() > 0        # wrapped base
    lora_b = router.wrappers["attn_qkv"].lora_b
    assert lora_b.grad[0].abs().sum() > 0                     # selected deviation
    assert lora_b.grad[3].abs().sum() < lora_b.grad[0].abs().sum()

    # ...and the parametrized path behaves the same way.
    orig = block.attn_norm.parametrizations.weight.original
    assert orig.grad is not None and orig.grad.abs().sum() > 0


def test_deviations_receive_gradient():
    router, block = make()
    x = torch.randn(2, 5, Cfg.hidden_size)
    router(*router_args(block, x))[0].sum().backward()
    banks = [d.bank for d in router.deviations if d.bank.grad is not None]
    loras = [w.lora_b for w in router.wrappers.values() if w.lora_b.grad is not None]
    assert banks and any(b.grad.abs().sum() > 0 for b in banks)
    assert loras and any(l.grad.abs().sum() > 0 for l in loras)


def test_repulsion_is_a_scalar_and_training_only():
    """Repulsion is VEAR's, not SMEAR's - the paper's SMEAR has neither it nor
    sharpening."""
    assert not hasattr(SMEAR, "router_aux_loss")
    router, _ = make(VEAR)
    router.train()
    aux = router.router_aux_loss()
    assert "vear_repulsion" in aux and aux["vear_repulsion"].dim() == 0
    router.eval()
    assert router.router_aux_loss() == {}


# --- cost ---------------------------------------------------------------------


def test_factored_deviations_are_cheaper_than_whole_copies():
    """The point of the redesign: N deviations must cost less than N blocks."""
    router, block = make(n=4)
    block_params = sum(p.numel() for p in block.parameters())
    # What the old design would have paid for the same expert count.
    smear_cost = 3 * block_params
    assert router.delta_numel < smear_cost, (router.delta_numel, smear_cost)


def test_linear_targets_are_wrapped_and_the_rest_are_not():
    """Linear targets route per example via MergedLinear; everything else is
    reached by a parametrization - so nothing is dropped for being awkward."""
    router, block = make()
    assert set(router.wrappers) == {"attn_qkv", "attn_output"}
    assert isinstance(block.attn.qkv, MergedLinear)
    # Raw parameters on a module nobody edited are still covered.
    assert parametrize.is_parametrized(block.attn, "kappa")
    assert parametrize.is_parametrized(block.attn_norm, "weight")
    assert len(router.deviations) == len(router._deviation_row)
    assert all(isinstance(d, Deviation) for d in router.deviations)


def test_every_targeted_parameter_is_covered():
    """Full coverage: each target is either wrapped or parametrized. A target
    that is silently uncovered is the failure mode this replaced."""
    router, block = make()
    wrapped = set(router.wrappers)
    for group in router.targets:
        if group.label in wrapped:
            continue
        for pname in group.params:
            owner_name, _, leaf = pname.rpartition(".")
            owner = block.get_submodule(owner_name) if owner_name else block
            assert parametrize.is_parametrized(owner, leaf), f"{pname} uncovered"


def test_wrapper_preserves_parameter_identity_and_names():
    """The base Parameter objects are held directly, so qualified names survive
    and an older checkpoint still resolves."""
    router, block = make()
    base_weight = block.attn.qkv.weight
    names = dict(block.named_parameters())
    assert "attn.qkv.weight" in names and names["attn.qkv.weight"] is base_weight
    assert "attn.qkv.lora_a" in names and "attn.qkv.lora_b" in names
    assert block.attn.qkv.in_features == Cfg.hidden_size


def test_router_never_reparametrizes_the_block():
    """THE structural guarantee. functional_call swaps entries in _parameters,
    and the block contains the Titans memory, which runs vmap + grad + its own
    nested functional_call. Nesting those produced three separate failures on
    abstractinator-m. The router must call the block plainly."""
    from praxis.routers import smear as smear_mod

    # Names actually referenced by the compiled forward - comments and the
    # docstring (which explain why it must not) do not count.
    names = smear_mod.SMEAR.forward.__code__.co_names
    consts = smear_mod.SMEAR.forward.__code__.co_consts
    assert "functional_call" not in names, "SMEAR.forward reparametrizes again"
    assert not any(isinstance(c, str) and c == "functional_call" for c in consts)

    router, block = make()
    x = torch.randn(2, 5, Cfg.hidden_size)
    before = {id(p) for p in block.parameters()}
    router(*router_args(block, x))
    assert {id(p) for p in block.parameters()} == before, (
        "block parameter objects were swapped during the forward"
    )


def test_merged_linear_equals_the_explicit_merged_weight():
    """The associativity trick must agree with materializing the merged weight."""
    router, block = make()
    w = router.wrappers["attn_qkv"]
    nn.init.normal_(w.lora_b, std=0.05)
    coeff = torch.softmax(torch.randn(3, w.num_experts), dim=-1)
    x = torch.randn(3, 5, w.in_features)

    w._coeff = coeff
    got = w(x)
    w._coeff = None

    want = torch.stack([
        torch.nn.functional.linear(
            x[b],
            w.weight + sum(coeff[b, e] * (w.lora_b[e] @ w.lora_a[e])
                           for e in range(w.num_experts)),
            w.bias,
        )
        for b in range(3)
    ])
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_per_example_routing_gives_examples_different_geometries():
    """The whole point: two examples in one batch can be transformed by
    different merged weights. Under a batch mean this is impossible."""
    router, block = make()
    w = router.wrappers["attn_qkv"]
    nn.init.normal_(w.lora_b, std=0.2)
    x = torch.randn(1, 4, w.in_features).expand(2, 4, w.in_features).contiguous()

    coeff = torch.zeros(2, w.num_experts)
    coeff[0, 0] = 1.0
    coeff[1, 3] = 1.0  # same input, different routing
    w._coeff = coeff
    out = w(x)
    w._coeff = None
    assert not torch.allclose(out[0], out[1]), "per-example routing had no effect"


def test_wrapper_falls_back_to_base_when_no_scope_is_open():
    router, block = make()
    w = router.wrappers["attn_qkv"]
    nn.init.normal_(w.lora_b, std=0.2)
    x = torch.randn(2, 4, w.in_features)
    assert w._coeff is None
    torch.testing.assert_close(
        w(x), torch.nn.functional.linear(x, w.weight, w.bias)
    )


def test_coefficient_scope_is_released():
    """A stale coefficient would silently route the next forward with the last
    one's routing."""
    router, block = make()
    x = torch.randn(2, 5, Cfg.hidden_size)
    router(*router_args(block, x))
    assert all(w._coeff is None for w in router.wrappers.values())


def test_untargeted_parameters_are_absent_from_the_merge():
    """MERGE_OPAQUE subtrees carry no parametrization at all."""
    router, block = make()
    for name, module in block.ffn.named_modules():
        for pname, _ in module.named_parameters(recurse=False):
            assert not parametrize.is_parametrized(module, pname)


def test_metrics_carry_no_depth_prefix():
    """SMEAR's nine chart families x one series per recurrent pass is the thing
    this design is replacing; nothing here may reintroduce a layer_{d}_ key."""
    router, block = make()
    x = torch.randn(4, 5, Cfg.hidden_size)
    for d in range(6):
        router(*router_args(block, x, depth=d))
    m = router.get_metrics()
    assert m, "no metrics emitted"
    assert not any(k.startswith("layer_") for k in m)
    assert "smear_target_dispersion" in m
    assert "smear_input_dependence" in m
    assert sum(k.startswith("smear_coeff_") for k in m) == len(router.targets) * 4


def test_dispersion_is_zero_when_targets_agree():
    router, block = make()
    with torch.no_grad():
        router.router.weight.zero_()
        router.router.bias.zero_()
    x = torch.randn(4, 5, Cfg.hidden_size)
    router(*router_args(block, x))
    assert router.get_metrics()["smear_target_dispersion"] == pytest.approx(0.0, abs=1e-6)


# --- registry -----------------------------------------------------------------


@pytest.mark.parametrize("key", ["smear", "vear", "smear_batch", "smear_token"])
def test_registry_entries_build(key):
    cfg = Cfg()
    block = Block(cfg.hidden_size)
    cfg.num_experts = 4
    router = ROUTER_REGISTRY[key](cfg, block=block, verbose=False)
    x = torch.randn(2, 5, cfg.hidden_size)
    out = router(*router_args(block, x))
    assert out[0].shape == x.shape


def test_state_dict_round_trips():
    router, block = make(SMEAR)
    for dev in router.deviations:
        nn.init.normal_(dev.bank, std=0.05)
    nn.init.normal_(router.depth_bias.weight, std=0.05)

    # The router and its block are one unit now: the deviations are registered
    # ON the block by parametrize, and the wrapped Linears ARE block modules. So
    # a round trip has to carry both and be replayed against the fresh block.
    fresh, fresh_block = make(SMEAR)
    fresh.load_state_dict(router.state_dict())
    fresh_block.load_state_dict(block.state_dict())

    x = torch.randn(2, 5, Cfg.hidden_size)
    with torch.no_grad():
        torch.testing.assert_close(
            fresh(*router_args(fresh_block, x))[0], router(*router_args(block, x))[0]
        )


def test_layout_is_shared_so_the_decoder_builds_one_block():
    from praxis.decoders.base import _router_layout, _wants_expert_bank

    assert _router_layout("smear") == "shared"
    assert _router_layout("vear") == "shared"
    assert not _wants_expert_bank("smear")
    # Distance is the only bank router left (praxis/routers/bank.py).
    assert _router_layout("distance") == "bank"


# --- expert dropout (the paper's load-balancing mechanism) --------------------


def test_expert_dropout_defaults_to_the_paper_rate():
    assert SMEAR.EXPERT_DROPOUT == pytest.approx(0.1)


def test_dropout_perturbs_coefficients_only_while_training():
    router, _ = make()
    router.EXPERT_DROPOUT = 0.5
    x = torch.randn(64, 5, Cfg.hidden_size)

    router.eval()
    a, _ = router._coefficients(x, 0)
    b, _ = router._coefficients(x, 0)
    torch.testing.assert_close(a, b)  # deterministic with dropout disabled

    router.train()
    c, _ = router._coefficients(x, 0)
    d, _ = router._coefficients(x, 0)
    assert not torch.allclose(c, d), "dropout did not perturb the coefficients"


def test_all_dropped_falls_back_to_base_rather_than_zeroing_the_block():
    """An all-dropped draw yields zero coefficients, and zero coefficients mean
    the base runs unchanged. SMEAR's sum_e w_e P_e would give an all-zero block."""
    router, block = make()
    router.EXPERT_DROPOUT = 1.0
    router.train()
    x = torch.randn(8, 5, Cfg.hidden_size)
    merge, _ = router._coefficients(x, 0)
    assert torch.all(merge == 0)

    for dev in router.deviations:
        nn.init.normal_(dev.bank, std=0.2)
    base = block.attn_norm.parametrizations.weight.original.detach().clone()
    with router._coefficient_scope(merge):
        torch.testing.assert_close(block.attn_norm.weight, base)

    w = router.wrappers["attn_qkv"]
    nn.init.normal_(w.lora_b, std=0.2)
    probe = torch.randn(8, 5, w.in_features)
    with router._coefficient_scope(merge):
        torch.testing.assert_close(
            w(probe), torch.nn.functional.linear(probe, w.weight, w.bias)
        )


def test_diagnostics_are_computed_before_dropout():
    """Metrics describe the ROUTER, so a dropped draw must not be what they
    report - the same distinction SMEAR draws for its own diagnostics."""
    router, block = make()
    router.EXPERT_DROPOUT = 0.9
    router.train()
    x = torch.randn(64, 5, Cfg.hidden_size)
    _, probs = router._coefficients(x, 0)
    torch.testing.assert_close(
        probs.sum(dim=-1), torch.ones(64, len(router.targets))
    )
    assert (probs > 0).all(), "diagnostics saw a post-dropout distribution"


def test_utilization_metric_spans_collapse_to_balance():
    router, block = make()
    x = torch.randn(16, 5, Cfg.hidden_size)

    with torch.no_grad():  # uniform routing -> every deviation in use
        router.router.weight.zero_()
        router.router.bias.zero_()
    router(*router_args(block, x))
    assert router.get_metrics()["smear_expert_utilization"] == pytest.approx(1.0)

    router._tick = 0
    with torch.no_grad():  # collapsed routing -> one of four in use
        router.router.bias.view(len(router.targets), 4)[:, 0] = 50.0
    router(*router_args(block, x))
    assert router.get_metrics()["smear_expert_utilization"] == pytest.approx(0.25)


# --- reduction: how far a routing decision is shared -------------------------


def test_reduction_shapes():
    """Each reduction changes what the coefficients are indexed by."""
    x = torch.randn(4, 6, Cfg.hidden_size)
    for reduction, want in (
        ("example", (4, None, 4)),
        ("token", (4, 6, None, 4)),
        ("batch", (4, None, 4)),
    ):
        cfg = Cfg()
        block = Block(cfg.hidden_size)
        r = SMEAR(cfg, block=block, num_experts=4, reduction=reduction, verbose=False)
        r.EXPERT_DROPOUT = 0.0
        merge, _ = r._coefficients(x, 0)
        expected = tuple(len(r.targets) if d is None else d for d in want)
        assert merge.shape == expected, f"{reduction}: {merge.shape} != {expected}"
        # Elementwise targets always collapse to one geometry per forward.
        assert r._flatten(merge).shape == (len(r.targets), 4)


def test_batch_reduction_gives_every_example_the_same_routing():
    """The legacy behaviour, and the reason a constant router was its fixed
    point: no gradient path can distinguish two examples."""
    cfg = Cfg()
    block = Block(cfg.hidden_size)
    r = SMEAR(cfg, block=block, num_experts=4, reduction="batch", verbose=False)
    r.EXPERT_DROPOUT = 0.0
    merge, _ = r._coefficients(torch.randn(8, 5, Cfg.hidden_size), 0)
    for b in range(1, 8):
        torch.testing.assert_close(merge[b], merge[0])


def test_example_reduction_does_not(): 
    cfg = Cfg()
    block = Block(cfg.hidden_size)
    r = SMEAR(cfg, block=block, num_experts=4, reduction="example", verbose=False)
    r.EXPERT_DROPOUT = 0.0
    with torch.no_grad():  # make the router actually read its input
        nn.init.normal_(r.router.weight, std=1.0)
    merge, _ = r._coefficients(torch.randn(8, 5, Cfg.hidden_size), 0)
    assert not torch.allclose(merge[0], merge[1])


def test_token_reduction_routes_positions_independently():
    """Beyond the paper, and only possible because MergedLinear never
    materializes the merged weight."""
    cfg = Cfg()
    block = Block(cfg.hidden_size)
    r = SMEAR(cfg, block=block, num_experts=4, reduction="token", verbose=False)
    r.EXPERT_DROPOUT = 0.0
    with torch.no_grad():
        nn.init.normal_(r.router.weight, std=1.0)
    merge, _ = r._coefficients(torch.randn(2, 6, Cfg.hidden_size), 0)
    assert not torch.allclose(merge[0, 0], merge[0, 1]), "positions routed alike"


@pytest.mark.parametrize("reduction", ["token", "example", "batch"])
def test_every_reduction_runs_forward_and_backward(reduction):
    cfg = Cfg()
    block = Block(cfg.hidden_size)
    r = SMEAR(cfg, block=block, num_experts=4, reduction=reduction, verbose=False)
    x = torch.randn(3, 5, Cfg.hidden_size)
    r(*router_args(block, x))[0].sum().backward()
    assert block.attn.qkv.weight.grad.abs().sum() > 0
    assert r.wrappers["attn_qkv"].lora_b.grad is not None


def test_unknown_reduction_is_rejected():
    cfg = Cfg()
    with pytest.raises(ValueError, match="Unknown reduction"):
        SMEAR(cfg, block=Block(cfg.hidden_size), reduction="sequence", verbose=False)


def test_num_experts_comes_from_the_config():
    """One registry entry per router; the count is config, not a key suffix."""
    cfg = Cfg()
    cfg.num_experts = 7
    r = SMEAR(cfg, block=Block(cfg.hidden_size), verbose=False)
    assert r.num_experts == 7
    assert r.router.out_features == len(r.targets) * 7


# --- the failure this design exists to prevent -------------------------------


def test_routes_a_block_containing_functorch_transforms():
    """A module that runs vmap + grad + its own functional_call must survive
    being routed, across several recurrent depths.

    This is the Titans memory's shape (praxis/memory/neural_memory.py), and
    routing it under the old ``functional_call`` design failed three different
    ways on abstractinator-m - a lost requires_grad, and a BatchedTensor that
    escaped its vmap and was still installed on a parameter at teardown.
    """
    from torch.func import functional_call, grad, vmap

    inner = nn.Linear(Cfg.hidden_size, Cfg.hidden_size)

    class Memoryish(nn.Module):
        """Stands in for NeuralMemory: reparametrizes ITSELF inside a vmap."""

        MERGE_OPAQUE = True

        def forward(s, x):
            w = {
                n: p.unsqueeze(0).expand(x.shape[0], *p.shape)
                for n, p in inner.named_parameters()
            }

            def loss(wi, xi):
                return functional_call(inner, wi, (xi,)).sum()

            vmap(grad(loss))(w, x.mean(dim=1))
            return x

    class Blk(Block):
        def __init__(s, d):
            super().__init__(d)
            s.memory = Memoryish()

        def forward(s, inputs, *a, **kw):
            out = super().forward(inputs, *a, **kw)
            return (s.memory(out[0]),) + out[1:]

    cfg = Cfg()
    block = Blk(cfg.hidden_size)
    router = SMEAR(cfg, block=block, num_experts=4, verbose=False)
    x = torch.randn(2, 8, Cfg.hidden_size)

    for depth in range(cfg.depth):
        out = router(*router_args(block, x, depth=depth))
        assert out[0].shape == x.shape

    # The parameters the transform touched must still be ordinary tensors -
    # a BatchedTensor left installed is what took the run down at teardown.
    for p in inner.parameters():
        p.unsqueeze(0)          # raises if it escaped a vmap
        p.detach().cpu()        # what Lightning's teardown does

    out = router(*router_args(block, x))
    out[0].sum().backward()
    assert block.attn.qkv.weight.grad is not None
