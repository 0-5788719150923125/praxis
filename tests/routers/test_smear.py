"""Modular SMEAR (praxis/routers/smear.py): the paper's granularity, over a
shared block plus deviations.

Not a new method - SMEAR (arxiv 2306.03745) applied the way the paper applies
it. Pinned here:

  * targets are discovered per MODULE, not per block, and each gets its own
    coefficient row (the paper puts a router on each inserted adapter);
  * Linear targets route per position on the running mean of the prefix (the
    paper's per-example pooling, made causal), while elementwise targets take
    the input-free depth prior, so no position reads a later one and no row
    reads another;
  * expert dropout is present, because that is the paper's load-balancing
    mechanism and without it every target collapses to one-hot;
  * the router is EXACTLY identity at init, so a config swap is a clean A/B;
  * the merge equals the convex combination of the implied experts;
  * the shared trunk receives full gradient whatever the routing does.

Target discovery itself (opaque subtrees, tied and frozen parameters) is
tests/transforms/test_targeting.py.
"""

from itertools import product

import pytest
import torch
import torch.nn as nn

from praxis import PraxisConfig, registry
from praxis.containers import LossContainer
from praxis.routers.smear import SMEAR, MergedLinear, _get_param
from praxis.routers.vear import VEAR
from tests.routers.toy_block import Block, Cfg, make, router_args

# --- targets ------------------------------------------------------------------


def test_targets_are_per_module_and_only_linears_are_wrapped():
    """One coefficient row per module, sized by the configured expert count.
    Linear targets route per example via MergedLinear; the rest merge on the
    batch mean. That split is the paper's: it routes adapters, not layernorms."""
    router, block = make()
    # attn (bare kappa), attn.qkv, attn.output, attn_norm, ffn_norm
    assert len(router.targets) == 5
    assert router.router.out_features == 5 * 4
    assert set(router.wrappers) == {"attn_qkv", "attn_output"}
    assert isinstance(block.attn.qkv, MergedLinear)
    assert isinstance(block.attn.output, MergedLinear)
    # Wrapped Linears leave the batch-mean bookkeeping entirely.
    assert not any(n.startswith("attn.qkv.") for n in router._param_row)
    assert "attn_norm.weight" in router._param_row

    # One registry entry per router; the count is config, not a key suffix.
    cfg = Cfg()
    cfg.num_experts = 7
    r = SMEAR(cfg, block=Block(cfg.hidden_size), verbose=False)
    assert r.num_experts == 7
    assert r.router.out_features == len(r.targets) * 7


# --- identity at init --------------------------------------------------------


def test_merge_is_exactly_identity_at_init():
    """Both merge paths, the depth bias and the whole forward are exact no-ops
    at step 0."""
    router, block = make()
    x = torch.randn(3, 7, Cfg.hidden_size)
    merge, _ = router._coefficients(x, 0)

    # Depth-prior path: every merged tensor is bit-identical to its base, and
    # nothing outside the targets (the opaque ffn) is in the merge at all.
    merged = router._merged_state_dict(block, router._depth_prior(0))
    assert not any(k.startswith("ffn.") for k in merged)
    for name, tensor in merged.items():
        assert torch.equal(tensor, _get_param(block, name)), f"{name} moved at init"

    # Routed path: each wrapper reduces to the Linear it replaced.
    with router._coefficient_scope(merge):
        for label, wrapper in router.wrappers.items():
            probe = torch.randn(3, 7, wrapper.in_features)
            got = wrapper(probe)
            want = torch.nn.functional.linear(probe, wrapper.weight, wrapper.bias)
            torch.testing.assert_close(got, want, msg=f"{label} moved at init")

    # The per-recurrent-pass bias starts at zero, so every depth routes alike.
    assert torch.all(router.depth_bias.weight == 0)
    torch.testing.assert_close(merge, router._coefficients(x, 5)[0])

    with torch.no_grad():
        want = block(x)[0]
        got = router(*router_args(block, x))[0]
    torch.testing.assert_close(got, want)


def test_depth_bias_wraps_past_the_table():
    router, block = make(SMEAR, depth=6)
    x = torch.randn(2, 5, Cfg.hidden_size)
    router(*router_args(block, x, depth=9))  # must not raise


# --- the merge is SMEAR in a different basis ---------------------------------


def test_merge_equals_convex_combination_of_implied_experts():
    """base + sum_e w_e delta_e == sum_e w_e (base + delta_e), on the depth-prior path."""
    router, block = make()
    for param in router.deltas.values():
        nn.init.normal_(param, std=0.05)
    with torch.no_grad():  # a non-uniform prior, so the check has teeth
        nn.init.normal_(router.router.bias, std=1.0)
    w = router._depth_prior(0)
    merged = router._merged_state_dict(block, w)

    for name, row in router._param_row.items():
        base = _get_param(block, name)
        coeffs = w[row]
        experts = []
        for e in range(router.num_experts):
            onehot = torch.zeros_like(coeffs)
            onehot[e] = 1.0
            experts.append(base + router._delta_for(name, onehot, base.dtype))
        want = sum(c * p for c, p in zip(coeffs, experts))
        torch.testing.assert_close(merged[name], want, rtol=1e-4, atol=1e-6)


# --- gradient behaviour -------------------------------------------------------


def test_shared_trunk_gets_full_gradient_under_collapsed_routing():
    """The property SMEAR lacked: whatever the routing does, the base learns.

    Under SMEAR an expert at weight ~0 received ~0 gradient AND held the only
    copy of its geometry. Here a collapsed coefficient starves one deviation
    while the trunk keeps its full gradient path.
    """
    router, block = make()
    with torch.no_grad():  # force a near one-hot routing onto expert 0
        router.router.bias.zero_()
        router.router.bias.view(len(router.targets), 4)[:, 0] = 50.0
    x = torch.randn(2, 5, Cfg.hidden_size)
    router(*router_args(block, x))[0].sum().backward()

    assert block.attn.qkv.weight.grad is not None
    assert block.attn.qkv.weight.grad.abs().sum() > 0

    lora_b = router.wrappers["attn_qkv"].lora_b
    assert lora_b.grad[0].abs().sum() > 0  # the selected deviation learns
    assert (
        lora_b.grad[3].abs().sum() < lora_b.grad[0].abs().sum()
    )  # a starved one does not

    # ...and a depth-prior target behaves the same way.
    bank = router.deltas[router._key("attn_norm.weight")]
    assert bank.grad[0].abs().sum() > 0
    assert block.attn_norm.weight.grad.abs().sum() > 0


# --- cost ---------------------------------------------------------------------


def test_factored_deviations_are_cheaper_than_whole_copies():
    """N deviations must cost less than N blocks."""
    router, block = make(n=4)
    block_params = sum(p.numel() for p in block.parameters())
    # What whole-block copies would cost for the same expert count.
    smear_cost = 3 * block_params
    assert router.delta_numel < smear_cost, (router.delta_numel, smear_cost)


def test_wrapper_preserves_parameter_identity_and_names():
    """The base Parameter objects are held directly, so qualified names survive
    and an older checkpoint still resolves."""
    router, block = make()
    base_weight = block.attn.qkv.weight
    names = dict(block.named_parameters())
    assert "attn.qkv.weight" in names and names["attn.qkv.weight"] is base_weight
    assert "attn.qkv.lora_a" in names and "attn.qkv.lora_b" in names
    assert block.attn.qkv.in_features == Cfg.hidden_size


def test_batch_mean_parametrization_is_chosen_by_shape():
    """On the batch-mean path, big 2-D tensors factor and small ones stay dense.
    Shape-derived, so there is nothing to configure per experiment."""
    from praxis.transforms.targeting import DENSE_DELTA_MAX_NUMEL

    router, block = make()
    assert router._factored["attn_norm.weight"] is False  # 1-D
    assert router._factored["attn.kappa"] is False
    assert all(
        _get_param(block, n).numel() <= DENSE_DELTA_MAX_NUMEL or router._factored[n]
        for n in router._param_row
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

    want = torch.stack(
        [
            torch.nn.functional.linear(
                x[b],
                w.weight
                + sum(
                    coeff[b, e] * (w.lora_b[e] @ w.lora_a[e])
                    for e in range(w.num_experts)
                ),
                w.bias,
            )
            for b in range(3)
        ]
    )
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_merged_linear_equals_the_explicit_merged_weight_per_position():
    """Causal routing hands each position its own coefficients, ``[B, T, N]``;
    each position must see exactly its own merged weight."""
    router, block = make()
    w = router.wrappers["attn_qkv"]
    nn.init.normal_(w.lora_b, std=0.05)
    coeff = torch.softmax(torch.randn(2, 4, w.num_experts), dim=-1)
    x = torch.randn(2, 4, w.in_features)

    w._coeff = coeff
    got = w(x)
    w._coeff = None

    for b, t in product(range(2), range(4)):
        merged = w.weight + sum(
            coeff[b, t, e] * (w.lora_b[e] @ w.lora_a[e]) for e in range(w.num_experts)
        )
        want = torch.nn.functional.linear(x[b, t], merged, w.bias)
        torch.testing.assert_close(got[b, t], want, rtol=1e-4, atol=1e-5)


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


def test_wrappers_fall_back_to_base_outside_a_released_scope():
    """A stale coefficient would silently route the next forward with the last
    one's routing, so every forward must release its scope."""
    router, block = make()
    w = router.wrappers["attn_qkv"]
    nn.init.normal_(w.lora_b, std=0.2)
    x = torch.randn(2, 4, w.in_features)
    assert w._coeff is None
    torch.testing.assert_close(w(x), torch.nn.functional.linear(x, w.weight, w.bias))

    router(*router_args(block, torch.randn(2, 5, Cfg.hidden_size)))
    assert all(w._coeff is None for w in router.wrappers.values())


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
    assert router.get_metrics()["smear_target_dispersion"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_state_dict_round_trips():
    router, block = make(SMEAR)
    for p in router.deltas.values():
        nn.init.normal_(p, std=0.05)
    nn.init.normal_(router.depth_bias.weight, std=0.05)

    fresh, _ = make(SMEAR)
    fresh.load_state_dict(router.state_dict())

    x = torch.randn(2, 5, Cfg.hidden_size)
    with torch.no_grad():
        torch.testing.assert_close(
            fresh(*router_args(block, x))[0], router(*router_args(block, x))[0]
        )


# --- expert dropout (the paper's load-balancing mechanism) --------------------


def test_dropout_perturbs_coefficients_only_while_training():
    assert SMEAR.EXPERT_DROPOUT == pytest.approx(0.1)  # the paper's rate
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
    """The safety property base-plus-deviation has and SMEAR does not: an
    all-dropped draw yields zero coefficients, and zero coefficients mean the
    base runs unchanged. SMEAR's sum_e w_e P_e would give an all-zero block."""
    router, block = make()
    router.EXPERT_DROPOUT = 1.0  # drop everything, every time
    router.train()
    x = torch.randn(8, 5, Cfg.hidden_size)
    merge, _ = router._coefficients(x, 0)
    assert torch.all(merge == 0)

    prior = router._depth_prior(0)
    assert torch.all(prior == 0)
    merged = router._merged_state_dict(block, prior)
    for name, tensor in merged.items():
        assert torch.equal(tensor, _get_param(block, name))

    w = router.wrappers["attn_qkv"]
    nn.init.normal_(w.lora_b, std=0.2)
    probe = torch.randn(8, 5, w.in_features)
    with router._coefficient_scope(merge):
        torch.testing.assert_close(
            w(probe), torch.nn.functional.linear(probe, w.weight, w.bias)
        )


def test_utilization_metric_spans_collapse_to_balance():
    router, block = make()
    x = torch.randn(16, 5, Cfg.hidden_size)

    with torch.no_grad():  # uniform routing -> every deviation in use
        router.router.weight.zero_()
        router.router.bias.zero_()
    router(*router_args(block, x))
    assert router.get_metrics()["smear_expert_utilization"] == pytest.approx(1.0)

    # Diagnostics average over a window of `depth` passes, so clear what was
    # already reported before measuring the second regime.
    router._tick, router._metrics = 0, {}
    with torch.no_grad():  # collapsed routing -> one of four in use
        router.router.bias.view(len(router.targets), 4)[:, 0] = 50.0
    router(*router_args(block, x))
    assert router.get_metrics()["smear_expert_utilization"] == pytest.approx(0.25)


# --- reduction: how far a routing decision is shared -------------------------


def test_reduction_shapes():
    """Each reduction changes what the coefficients are indexed by."""
    x = torch.randn(4, 6, Cfg.hidden_size)
    for reduction, want in (
        ("example", (4, 6, None, 4)),
        ("token", (4, 6, None, 4)),
        ("batch", (4, None, 4)),
    ):
        r, _ = make(reduction=reduction)
        merge, _ = r._coefficients(x, 0)
        expected = tuple(len(r.targets) if d is None else d for d in want)
        assert merge.shape == expected, f"{reduction}: {merge.shape} != {expected}"
        torch.testing.assert_close(merge.sum(dim=-1), torch.ones(expected[:-1]))
        # Elementwise targets always hold one geometry per forward.
        assert r._depth_prior(0).shape == (len(r.targets), 4)


def test_batch_reduction_gives_every_example_the_same_routing():
    """The control arm: one input-free geometry for every example."""
    r, _ = make(reduction="batch")
    merge, _ = r._coefficients(torch.randn(8, 5, Cfg.hidden_size), 0)
    for b in range(1, 8):
        torch.testing.assert_close(merge[b], merge[0])


def test_example_reduction_does_not():
    r, _ = make(reduction="example")
    with torch.no_grad():  # make the router actually read its input
        nn.init.normal_(r.router.weight, std=1.0)
    merge, _ = r._coefficients(torch.randn(8, 5, Cfg.hidden_size), 0)
    assert not torch.allclose(merge[0], merge[1])


def test_token_reduction_routes_positions_independently():
    """Beyond the paper, and only possible because MergedLinear never
    materializes the merged weight."""
    r, _ = make(reduction="token")
    with torch.no_grad():
        nn.init.normal_(r.router.weight, std=1.0)
    merge, _ = r._coefficients(torch.randn(2, 6, Cfg.hidden_size), 0)
    assert not torch.allclose(merge[0, 0], merge[0, 1]), "positions routed alike"


def test_unknown_reduction_is_rejected():
    cfg = Cfg()
    with pytest.raises(ValueError, match="Unknown reduction"):
        SMEAR(cfg, block=Block(cfg.hidden_size), reduction="sequence", verbose=False)


# --- causality: no position reads a later one, no row reads another ----------


def _reading(reduction="example"):
    """A router whose routing and deviations are far from identity, so a leak
    would show up rather than hide behind a zero-init merge."""
    r, block = make(reduction=reduction)
    with torch.no_grad():
        nn.init.normal_(r.router.weight, std=1.0)
        nn.init.normal_(r.router.bias, std=1.0)
        for w in r.wrappers.values():
            nn.init.normal_(w.lora_b, std=0.2)
        for p in r.deltas.values():
            nn.init.normal_(p, std=0.05)
    return r, block


@pytest.mark.parametrize("reduction", ["token", "example", "batch"])
def test_forward_reads_no_later_position_and_no_other_row(reduction):
    """Perturb row 0 at position ``p``: row 0 before ``p`` and every other row
    must be untouched. The Block mixes nothing across positions or rows, so any
    change there could only come from the routing."""
    r, block = _reading(reduction)
    x = torch.randn(3, 9, Cfg.hidden_size)
    p = 5
    xp = x.clone()
    xp[0, p] += torch.randn_like(xp[0, p])  # a direction, not a shift LayerNorm erases
    with torch.no_grad():
        a = r(*router_args(block, x))[0]
        b = r(*router_args(block, xp))[0]
    torch.testing.assert_close(a[0, :p], b[0, :p], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(a[1:], b[1:], rtol=0.0, atol=1e-6)
    assert not torch.allclose(a[0, p], b[0, p])


def test_last_position_routes_on_the_papers_pooled_input():
    """At the last position the running mean IS the sequence mean SMEAR pools."""
    r, _ = _reading()
    x = torch.randn(3, 7, Cfg.hidden_size)
    merge, _ = r._coefficients(x, 0)
    pooled = torch.softmax(r._route_logits(r.router_norm(x.mean(dim=1)), 0), dim=-1)
    torch.testing.assert_close(merge[:, -1], pooled)


class _Cache:
    """The two lookups SMEAR makes on a PraxisCache, with a settable length."""

    def __init__(self):
        self.head_states, self.length = {}, 0

    def get_head_state(self, key):
        return self.head_states.get(key)

    def set_head_state(self, key, state):
        self.head_states[key] = state

    def get_seq_length(self, layer_idx=0):
        return self.length


def test_cached_decode_continues_the_prefix():
    """A suffix routed on the carried running sum matches the same positions
    routed inside the full sequence."""
    r, _ = _reading()
    x = torch.randn(2, 8, Cfg.hidden_size)
    full, _ = r._coefficients(x, 0)
    cache = _Cache()
    r._coefficients(x[:, :5], 0, cache)  # prefill
    cache.length = 5  # the trunk has written the prefill at this depth
    step, _ = r._coefficients(x[:, 5:], 0, cache)
    torch.testing.assert_close(step, full[:, 5:])


def test_stale_decode_state_falls_back_to_the_suffix():
    r, _ = _reading()
    x = torch.randn(2, 8, Cfg.hidden_size)
    cache = _Cache()
    r._coefficients(x[:, :5], 0, cache)
    cache.length = 3  # rolled back: the carried sum no longer lines up
    step, _ = r._coefficients(x[:, 5:], 0, cache)
    alone, _ = r._coefficients(x[:, 5:], 0)
    torch.testing.assert_close(step, alone)


# --- functorch compatibility --------------------------------------------------


def test_routes_a_block_containing_functorch_transforms():
    """A module that runs vmap + grad + its own functional_call must survive
    being routed, across several recurrent depths.

    This is the Titans memory's shape (praxis/memory/neural_memory.py), nested
    inside the router's own ``functional_call``.
    """
    from torch.func import functional_call, grad, vmap

    inner = nn.Linear(Cfg.hidden_size, Cfg.hidden_size)

    class Memoryish(nn.Module):
        """Reparametrizes ITSELF inside a vmap, like NeuralMemory does."""

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
        assert router(*router_args(block, x, depth=depth))[0].shape == x.shape

    # The parameters the transform touched must still be ordinary tensors - a
    # BatchedTensor left installed is what took the run down at teardown.
    for p in inner.parameters():
        p.unsqueeze(0)
        p.detach().cpu()

    router(*router_args(block, x))[0].sum().backward()
    assert block.attn.qkv.weight.grad is not None


# --- selection diagnostics ---------------------------------------------------
#
# `smear_expert_utilization` averages the coefficients over the batch BEFORE
# scoring them, so on its own it cannot say why a target is concentrated. VEAR
# exists to make routing discrete, so "one deviation dominates" is its INTENDED
# outcome when different examples pick different deviations, and a failure
# (dead experts) when they do not. Sharpness and
# diversity separate those two, which is what makes a vear-vs-smear run legible.


class _Probe(SMEAR):
    """Just the diagnostics, with no block to build against."""

    def __init__(self, num_experts=4, targets=3, sharpen=1.0):
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.targets = [type("G", (), {"label": f"t{i}"})() for i in range(targets)]
        self.SHARPEN = sharpen
        # Every row routes on the input here; the selection metrics read only
        # the rows that do.
        self._wrapper_row = {f"t{i}": i for i in range(targets)}
        self._metrics = {}
        self._accum = {}
        self._passes = 0

    def score(self, probs):
        self._log_metrics(probs, probs)
        self._flush_metrics()  # accumulate-then-flush; one pass averages to itself
        return self._metrics


def _one_hot(idx, n):
    """Near-one-hot rows, kept off exactly 1.0 so entropy stays well defined."""
    return torch.nn.functional.one_hot(idx, n).float() * 0.97 + 0.01


def test_sharpness_separates_a_blend_from_a_choice():
    n = 4
    blend = _Probe(n).score(torch.full((256, 3, n), 1.0 / n))[
        "smear_selection_sharpness"
    ]
    choice = _Probe(n).score(_one_hot(torch.randint(0, n, (256, 3)), n))[
        "smear_selection_sharpness"
    ]
    assert blend == pytest.approx(0.0, abs=1e-5)
    assert choice > 0.8


def test_diversity_separates_specialization_from_collapse():
    """The distinction utilization alone cannot draw: both of these are equally
    sharp, and only one of them is using the bank."""
    n = 4
    spread = _Probe(n).score(_one_hot(torch.randint(0, n, (256, 3)), n))
    same = _Probe(n).score(_one_hot(torch.ones(256, 3, dtype=torch.long), n))

    assert spread["smear_selection_sharpness"] == pytest.approx(
        same["smear_selection_sharpness"], abs=1e-5
    ), "the two cases must be indistinguishable by sharpness, or the test is vacuous"
    assert spread["smear_selection_diversity"] > 0.9
    assert same["smear_selection_diversity"] == pytest.approx(0.0, abs=1e-5)


def test_sharpening_raises_sharpness_without_touching_diversity():
    """What VEAR's p**4 is supposed to do, and the reason -n is worth running:
    it should peak each decision, not narrow which deviations get chosen."""
    torch.manual_seed(0)
    n = 4
    probs = torch.softmax(torch.randn(256, 3, n) * 0.8, dim=-1)
    soft = _Probe(n, sharpen=1.0).score(probs)
    sharp = _Probe(n, sharpen=4.0).score(probs)

    assert sharp["smear_selection_sharpness"] > soft["smear_selection_sharpness"] + 0.3
    assert sharp["smear_selection_diversity"] == pytest.approx(
        soft["smear_selection_diversity"], abs=0.05
    )


def test_selection_metrics_are_measured_before_expert_dropout():
    """Dropout zeroes a tenth of the coefficients and renormalizes, which reads
    as peakedness the router did not choose. If it leaked into these numbers, a
    smear run would look more discrete than it is and the vear comparison would
    be biased before it started."""
    torch.manual_seed(0)
    router, block = make(dropout=0.5)  # exaggerated, so a leak is unmistakable
    assert router.EXPERT_DROPOUT > 0 and router.training

    x = torch.randn(8, 6, router.hidden_size)
    inputs = router.router_norm(router._prefix_mean(x, None, 0))
    probs = torch.softmax(router._route_logits(inputs, 0), dim=-1)

    merge, reported = router._coefficients(x, 0)
    torch.testing.assert_close(reported, probs)
    assert not torch.allclose(merge, probs), "dropout did not fire; test is vacuous"

    router._accum, router._passes, router._metrics = {}, 0, {}
    router._log_metrics(merge, reported)
    router._flush_metrics()
    from_probs = dict(router._metrics)
    router._accum, router._passes, router._metrics = {}, 0, {}
    router._log_metrics(merge, merge)  # what leaking dropout in would produce
    router._flush_metrics()
    from_merge = dict(router._metrics)

    assert (
        from_probs["smear_selection_sharpness"]
        != from_merge["smear_selection_sharpness"]
    ), "the dropout-free path is not actually distinguishable here"


@pytest.mark.parametrize("cls", [SMEAR, VEAR])
def test_a_real_forward_emits_the_selection_metrics(cls):
    """The -n card has to have data in it. get_metrics() is what the dashboard
    drains, so pin the whole path rather than _log_metrics in isolation."""
    router, block = make(cls)
    router(*router_args(block, torch.randn(8, 6, router.hidden_size)))
    metrics = router.get_metrics()
    for key in ("smear_selection_sharpness", "smear_selection_diversity"):
        assert key in metrics, f"{cls.__name__} never emitted {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of range"


def test_diagnostics_cover_every_recurrent_depth():
    """`_tick` advances once per recurrent PASS, so a sampler keyed on it must
    still report every depth, not a fixed subset of them."""
    router, block = make(depth=6)
    x = torch.randn(4, 6, router.hidden_size)

    seen = []
    original = router._log_metrics
    router._log_metrics = lambda m, p: (
        seen.append(router._probe_depth),
        original(m, p),
    )

    for _ in range(20):
        for depth in range(6):
            router._probe_depth = depth
            router(*router_args(block, x, depth=depth))

    assert set(seen) == set(
        range(6)
    ), f"depths missed by the diagnostics: {sorted(set(range(6)) - set(seen))}"
    assert seen.count(1) == seen.count(0), "passes are not weighted equally"


def test_flush_averages_rather_than_overwrites():
    """Two passes with known, different coefficients must report their mean."""
    router, _ = make()
    router._metrics, router._accum, router._passes = {}, {}, 0
    t, n = len(router.targets), router.num_experts

    for value in (0.0, 1.0):
        router._add("coeff", torch.full((t, n), value))
        router._add("smear_input_dependence", torch.tensor(value))
        router._passes += 1
    router._flush_metrics()

    assert router._metrics["smear_input_dependence"] == pytest.approx(0.5)
    label = router.targets[0].label
    assert router._metrics[f"smear_coeff_{label}_0"] == pytest.approx(0.5)
    assert router._accum == {} and router._passes == 0, "accumulator not reset"


def test_accumulator_holds_no_autograd_graph():
    """A retained graph across passes would be a slow leak, not a wrong number."""
    router, block = make()
    router(*router_args(block, torch.randn(4, 6, router.hidden_size)))
    for key, value in router._accum.items():
        assert not value.requires_grad, f"{key} carries grad"
        assert value.grad_fn is None, f"{key} retains a graph"


# --- deviation magnitude -----------------------------------------------------
#
# The coefficients say how the deviations are MIXED; this says whether the
# mixture moves the geometry at all. A rich mixture over numerically tiny
# deviations is a router arguing about nothing, and reads identically to a real
# one on the coefficient heatmap.


def _delta_scales(router, block):
    router(*router_args(block, torch.randn(4, 6, router.hidden_size)))
    router._flush_metrics()
    return {
        g.label: router._metrics[f"smear_delta_scale_{g.label}"] for g in router.targets
    }


def test_delta_scale_is_exactly_zero_at_init_on_both_merge_paths():
    """LoRA init makes every deviation exactly zero, so the reported movement
    must be exactly 0 - not merely small - and keyed for every target. The
    Linear targets never materialize a merged weight, so their deviation has to
    be formed explicitly; skipping it would silently drop them from the metric."""
    router, block = make()
    scales = _delta_scales(router, block)
    assert set(scales) == {g.label for g in router.targets}
    assert set(router.wrappers).issubset(scales), "MergedLinear targets are missing"
    assert set(router._param_row), "test needs both paths populated"
    for label, value in scales.items():
        assert value == pytest.approx(0.0, abs=1e-9), f"{label} moved at init: {value}"
    assert router._metrics["smear_delta_scale_mean"] == pytest.approx(0.0, abs=1e-9)


def test_delta_scale_tracks_the_true_norm_ratio():
    """Against the ratio computed directly from the parameters."""
    torch.manual_seed(0)
    router, block = make()
    for param in router.deltas.values():
        nn.init.normal_(param, std=0.05)
    for wrapper in router.wrappers.values():
        nn.init.normal_(wrapper.lora_b, std=0.05)

    scales = _delta_scales(router, block)
    x = torch.randn(4, 6, router.hidden_size)
    w = router._flatten(router._coefficients(x, 0)[0])

    for label, row in router._wrapper_row.items():
        wrap = router.wrappers[label]
        delta = torch.einsum("e,eor,eri->oi", w[row], wrap.lora_b, wrap.lora_a)
        want = (delta.norm() / wrap.weight.norm()).item()
        assert scales[label] == pytest.approx(want, rel=0.3), label
        assert scales[label] > 0, f"{label} reported no movement despite real deltas"


def test_diagnostics_survive_the_model_moving_device():
    """initialize_lazy_modules runs a dummy forward on CPU before the model is
    moved to its device, so a window can be left holding tensors on another
    device that the next pass cannot be added to. That is a RuntimeError inside
    forward(), not a wrong metric. Simulated without a GPU: the partial window
    is parked on the meta device and the next passes run on CPU."""
    router, block = make()
    x = torch.randn(4, 6, router.hidden_size)
    router(*router_args(block, x))
    assert router._accum, "nothing accumulated; test is vacuous"
    router._accum = {k: v.to("meta") for k, v in router._accum.items()}

    for _ in range(router._window + 1):
        router(*router_args(block, x))  # must not raise

    assert router.get_metrics()["smear_delta_scale_mean"] is not None
    for value in router._accum.values():
        assert value.device.type == "cpu"


def test_delta_scale_never_raises_into_forward():
    """Whatever it hits, the step continues."""
    router, block = make()
    router._delta_scale_inner = lambda *a, **k: 1 / 0
    router(*router_args(block, torch.randn(4, 6, router.hidden_size)))


# --- inside the sequential decoder --------------------------------------------


def _decoder(**overrides):
    config = PraxisConfig(
        **{
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 6,
            "num_experts": 3,
            "num_layers": 3,
            "decoder_type": "sequential",
            "block_type": "recurrent",
            "router_type": "smear",
            **overrides,
        }
    )
    return registry.lookup("decoders", config.decoder_type)(config), config


@pytest.mark.parametrize("block_type", ["recurrent", "gru", "min"])
def test_routes_every_recurrent_block_type(block_type):
    decoder, config = _decoder(block_type=block_type)
    hidden_states = torch.randn(2, 10, config.hidden_size)
    output, _, _, _ = decoder(hidden_states, losses=LossContainer())
    assert output.shape == hidden_states.shape


def test_gradient_reaches_the_input_and_the_router_through_the_decoder():
    # The min block holds Linear targets, so the input-routed path is live too.
    decoder, config = _decoder(block_type="min")
    hidden_states = torch.randn(2, 10, config.hidden_size, requires_grad=True)
    output, _, _, _ = decoder(hidden_states, losses=LossContainer())
    output.mean().backward()

    assert hidden_states.grad is not None
    assert hidden_states.grad.abs().sum() > 0

    # Both routing paths are in the graph: the depth prior always, the input
    # projection because some targets are Linears, which route on the input.
    # (Their gradients are exactly zero at init, where every deviation is.)
    router = decoder.locals[0].router
    assert router.wrappers, "no Linear targets; the input projection is untested"
    assert router.router.bias.grad is not None
    assert router.depth_bias.weight.grad is not None
    assert router.router.weight.grad is not None
