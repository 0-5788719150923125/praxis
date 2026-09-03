"""Kaleidoscope attention: frozen mirrors, input-conditional turn, per-depth facets."""

import math
from types import SimpleNamespace

import pytest
import torch

from praxis.attention.kaleidoscope import (
    MIRROR_RES,
    TURN_MOD,
    FACET_SCALE,
    KaleidoscopeAttention,
)


def _config(**over):
    cfg = SimpleNamespace(
        hidden_size=32,
        num_heads=1,  # patch_config forces this; set >1 only to test the patch
        head_size=16,
        num_queries=1,
        causal=True,
        dropout=0.0,
        depth=4,
        window_size=None,
        max_position_embeddings=64,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _attn(**over):
    torch.manual_seed(0)
    return KaleidoscopeAttention(_config(**over))


# ------------------------------------------------------------------- the mirrors
def test_mirrors_are_canonical_resolution_not_sequence_length():
    """A mirror is a function on the unit square, not a `[T, T]` table."""
    a = _attn()
    assert a.mirrors.shape == (a.num_mirrors, MIRROR_RES, MIRROR_RES)
    assert a.facet_u.shape == (a.depths, a.num_mirrors, MIRROR_RES)
    assert not hasattr(a, "span")


def test_mirrors_are_frozen_and_not_parameters():
    """The dictionary is architecture, not learned state."""
    a = _attn()
    names = {n for n, _ in a.named_parameters()}
    assert not any("mirror" in n for n in names)
    assert not a.mirrors.requires_grad
    # Non-persistent: N * span^2 floats stay out of every checkpoint.
    assert "mirrors" not in a.state_dict()


def test_mirrors_are_deterministic_across_constructions():
    """Non-persistent buffers only work if construction reproduces them."""
    assert torch.equal(_attn().mirrors, _attn().mirrors)


def test_no_query_or_key_projections_exist():
    """The matrix IS the parameter, so there is nothing to project from."""
    names = {n for n, _ in _attn().named_parameters()}
    assert not any(k in n.lower() for n in names for k in ("query", "key", "qkv"))
    assert {
        "turn.weight",
        "facet_u",
        "facet_v",
        "value.weight",
        "gate.weight",
        "output.weight",
    } <= names


# --------------------------------------------------------------------- the turn



def test_different_sequences_get_different_geometry():
    a = _attn()
    with torch.no_grad():
        a.turn.weight.normal_(std=1.0)
    x = torch.randn(2, 12, 32)
    w = torch.softmax(a.turn(x).view(2, 12, a.num_heads, a.num_mirrors).float(), -1)
    s = a._scores(w, a._faceted(0, 12))
    assert not torch.allclose(s[0], s[1])


# ------------------------------------------------------------------- the facets
def test_facets_are_identity_at_init():
    a = _attn()
    assert torch.equal(a._canonical(0), a.mirrors)
    assert a.training_metrics()["kaleido_facet_strength"] == pytest.approx(0.0)


def test_facet_deformation_does_not_factor_out_of_the_mixture():
    """The reason the bias is PER MIRROR rather than one shared matrix.

    Turn weights sum to 1, so a deformation added to every mirror alike would
    factor straight back out and reduce to a per-depth score bias. Per-mirror
    facets must survive the mixture as something the blend cannot undo.
    """
    a = _attn()
    with torch.no_grad():
        a.facet_u.normal_(std=0.5)
        a.facet_v.normal_(std=0.5)
        a.turn.weight.normal_(std=1.0)
    x = torch.randn(2, 12, 32)
    w = torch.softmax(a.turn(x).view(2, 12, a.num_heads, a.num_mirrors).float(), -1)

    faceted = a._scores(w, a._faceted(0, 12))
    frozen = a._scores(w, a._faceted_frozen(12))
    delta = faceted - frozen

    # If it factored out, delta would be one [T, T] matrix broadcast over every
    # batch element. It must not be.
    assert not torch.allclose(delta[0, 0], delta[1, 0], atol=1e-5)


def test_facets_specialize_by_depth_and_the_metric_reads_it():
    a = _attn()
    with torch.no_grad():  # every depth ground identically -> no specialization
        a.facet_u.copy_(a.facet_u[0].unsqueeze(0).expand_as(a.facet_u).clone())
        a.facet_u += 0.5
        a.facet_v.copy_(a.facet_v[0].unsqueeze(0).expand_as(a.facet_v).clone())
    shared = a.training_metrics()["kaleido_facet_depth_specialization"]
    with torch.no_grad():
        a.facet_u.normal_(std=0.5)
    varied = a.training_metrics()["kaleido_facet_depth_specialization"]
    assert shared == pytest.approx(0.0, abs=1e-4)
    assert varied > shared


def test_facet_strength_is_bounded_by_its_cap():
    a = _attn()
    with torch.no_grad():
        a.facet_u.normal_(std=50.0)
        a.facet_v.normal_(std=50.0)
    assert a.training_metrics()["kaleido_facet_strength"] <= 1.0 + 1e-6


def test_depth_changes_the_geometry_but_not_the_frozen_core():
    a = _attn()
    with torch.no_grad():
        a.facet_u.normal_(std=0.5)
    assert not torch.allclose(a._canonical(0), a._canonical(1))
    for d in range(a.depths):  # never further than the cap from the mirrors
        assert (a._canonical(d) - a.mirrors).abs().max() <= FACET_SCALE


def test_depth_index_saturates_past_the_configured_depth():
    a = _attn(depth=2)
    assert torch.equal(a._canonical(99), a._canonical(1))


# ------------------------------------------------------------------- the forward
def test_forward_shape_and_gradients_reach_the_blend():
    a = _attn()
    x = torch.randn(2, 16, 32, requires_grad=True)
    out, _, aux = a(x, current_depth=1)
    assert out.shape == (2, 16, 32)
    assert aux == 0.0
    out.sum().backward()
    # Both halves of the blend move from step 0: dS/dbeta_k and dS/dw_k are the
    # mirrors themselves, which are non-zero whatever the weights are.
    assert a.turn_static.weight.grad[1].abs().sum() > 0
    assert a.turn.weight.grad is not None and a.turn.weight.grad.abs().sum() > 0


def test_attention_is_causal():
    a = _attn()
    with torch.no_grad():
        a.turn.weight.normal_(std=1.0)
    T = 12
    x = torch.randn(1, T, 32)
    w = torch.softmax(a.turn(x).view(1, T, a.num_heads, a.num_mirrors).float(), -1)
    s = a._scores(w, a._faceted(0, T))
    pos = torch.arange(T)
    s = s.masked_fill(~((pos[:, None] - pos[None, :]) >= 0), float("-inf"))
    probs = torch.softmax(s, dim=-1)
    assert torch.allclose(probs.triu(diagonal=1), torch.zeros_like(probs), atol=1e-7)
    assert torch.allclose(probs.sum(-1), torch.ones(1, a.num_heads, T), atol=1e-5)


def test_window_size_bounds_the_reach():
    a = _attn(window_size=3)
    out, _, _ = a(torch.randn(1, 16, 32))
    assert out.shape == (1, 16, 32)



def test_patch_config_forces_a_single_head():
    """The count is corrected so config.json reports the head actually built."""
    cfg = _config(num_heads=8, num_queries=4)
    a = KaleidoscopeAttention(cfg)
    assert cfg.num_heads == 1 and cfg.num_queries == 1
    assert a.num_heads == 1
    # head_size is a WIDTH and is left alone.
    assert a.head_dim == cfg.head_size


def test_patch_config_is_idempotent():
    cfg = _config(num_heads=8)
    KaleidoscopeAttention.patch_config(cfg)
    KaleidoscopeAttention.patch_config(cfg)
    assert cfg.num_heads == 1


def test_unset_head_size_gives_one_head_spanning_the_hidden_size():
    cfg = _config(num_heads=4)
    cfg.head_size = None
    assert KaleidoscopeAttention(cfg).head_dim == cfg.hidden_size


# --------------------------------------------------------------------- the gate
def test_gate_can_go_negative_which_is_the_point_of_silu():
    """Mega Theorem 1 needs a gate that can amplify and flip sign; a sigmoid
    gate lands in (0, 1) and cannot, which is why this one is SiLU."""
    a = _attn()
    a.train()
    a(torch.randn(4, 16, 32))
    m = a.training_metrics()
    assert 0.0 < m["kaleido_gate_negative"] < 1.0
    assert m["kaleido_gate_magnitude"] > 0.0


def test_gate_multiplies_the_attention_output():
    a = _attn()
    a.train()
    x = torch.randn(2, 12, 32)
    baseline = a(x)[0]
    with torch.no_grad():  # a zero gate is a zero branch, not an identity
        a.gate.weight.zero_()
        a.gate.bias.zero_()
    assert torch.allclose(a(x)[0], torch.zeros_like(baseline), atol=1e-6)


def test_gate_receives_gradient():
    a = _attn()
    out, _, _ = a(torch.randn(2, 16, 32))
    out.sum().backward()
    assert a.gate.weight.grad is not None and a.gate.weight.grad.abs().sum() > 0


def test_registered_in_the_attention_registry():
    from praxis.attention import ATTENTION_REGISTRY

    assert ATTENTION_REGISTRY["kaleido"] is KaleidoscopeAttention


def test_specialization_is_absent_rather_than_1_when_there_is_no_deformation():
    """A ratio against ~0 energy reads 1.0 - "fully specialized" - which is the
    opposite of the truth, and is what every step before the facets move would
    have reported."""
    a = _attn()
    assert a.training_metrics()["kaleido_facet_strength"] == pytest.approx(0.0)
    assert "kaleido_facet_depth_specialization" not in a.training_metrics()
    with torch.no_grad():
        a.facet_u.normal_(std=0.5)
    assert "kaleido_facet_depth_specialization" in a.training_metrics()


def test_facets_unlock_in_stages_and_that_is_intended():
    """A gradient audit will flag the facets as dead at step 0. They are, twice
    over, and both are structural rather than bugs.

    `dS/d(facet_k) = w_k`, and the blend is zero-init, so NO facet moves until
    the blend does. Then within a facet, `d/dv (u (x) v) = u` and `u` is
    zero-init, so `v` waits on `u`. Blend -> u -> v, and the blend has gradient
    from step 0 so the chain unlocks immediately.
    """
    a = _attn()
    a(torch.randn(2, 12, 32), current_depth=0)[0].sum().backward()
    assert a.turn_static.weight.grad.abs().sum() > 0  # the blend moves first
    assert a.facet_u.grad.abs().sum() == 0  # gated on w != 0
    assert a.facet_v.grad.abs().sum() == 0

    a.zero_grad(set_to_none=True)
    with torch.no_grad():
        a.turn_static.weight.normal_(std=0.5)
    a(torch.randn(2, 12, 32), current_depth=0)[0].sum().backward()
    assert a.facet_u.grad.abs().sum() > 0  # u unlocked
    assert a.facet_v.grad.abs().sum() == 0  # v still waits on u

    a.zero_grad(set_to_none=True)
    with torch.no_grad():
        a.facet_u.normal_(std=0.3)
    a(torch.randn(2, 12, 32), current_depth=0)[0].sum().backward()
    assert a.facet_v.grad.abs().sum() > 0


# ---------------------------------------------------------------- ghostmax
def test_ghostmax_matches_an_explicit_zero_logit_column():
    """softmax1 = softmax * sigmoid(logsumexp), computed without the column.

    Asserted against the literal construction - append a zero logit whose value
    is zero, softmax over the wider row, drop the ghost - because the identity
    is the whole reason no extra column is materialized.
    """
    a = _attn()
    with torch.no_grad():
        a.turn.weight.normal_(std=1.0)
    B, T = 2, 12
    x = torch.randn(B, T, 32)
    w = torch.softmax(a.turn(x).view(B, T, a.num_heads, a.num_mirrors).float(), -1)
    s = a._scores(w, a._faceted(0, T))
    pos = torch.arange(T)
    s = s.masked_fill(~((pos[:, None] - pos[None, :]) >= 0), float("-inf"))
    v = torch.randn(B, a.num_heads, T, a.head_dim)

    ours = (torch.softmax(s, -1) @ v) * torch.sigmoid(torch.logsumexp(s, -1)).unsqueeze(-1)

    ghost_logit = torch.zeros(B, a.num_heads, T, 1)
    wide = torch.cat([ghost_logit, s], dim=-1)
    v_ghost = torch.cat([torch.zeros(B, a.num_heads, 1, a.head_dim), v], dim=-2)
    literal = torch.softmax(wide, -1) @ v_ghost

    assert torch.allclose(ours, literal, atol=1e-5)


def test_ghost_share_is_small_at_init_and_falls_with_length():
    """ssog.py declined the ghost because a Gaussian field's log-density logits
    hand it ~half the mass at EVERY position. Unit-scale mirrors do not - but
    the mean is still length-dependent, because position 0 has one key and gives
    the ghost ~0.5 whatever the logits do. That is the ghost doing the job SSOG
    needed a learned null atom for, and it is why the metric must be compared
    across like lengths."""
    a = _attn(max_position_embeddings=256)
    a.train()
    a(torch.randn(4, 64, 32))
    short = a.training_metrics()["kaleido_ghost_share"]
    a(torch.randn(4, 256, 32))
    long = a.training_metrics()["kaleido_ghost_share"]
    assert 0.0 < long < short < 0.10


def test_ghost_gives_queries_a_way_to_decline():
    """Uniformly tiny scores must route mass to the ghost, shrinking the output -
    which plain softmax, being scale-free in its normalizer, cannot do."""
    a = _attn()
    a.train()
    with torch.no_grad():
        a.turn.weight.zero_()
        a.turn_static.weight.fill_(1.0)  # a blend must exist for scores to exist
        a.mirrors.mul_(0.0).add_(-10.0)  # every real key is deeply unattractive
    a(torch.randn(2, 16, 32))
    assert a.training_metrics()["kaleido_ghost_share"] > 0.5


# ----------------------------------------------------------------- dropoff
def test_dropoff_is_off_by_default():
    a = _attn()
    assert a.dropoff_step is None
    v = torch.randn(1, 1, 8, 4)
    for d in range(a.depths):
        assert torch.equal(a._maybe_dropoff(v, d), v)


def test_dropoff_fires_only_on_the_last_pass_and_sinks_the_tip():
    a = _attn(depth=4, num_layers=1)
    a.train()
    a.dropoff_mode, a.dropoff_step = "warp", 3
    v = torch.ones(1, 1, 8, 4)
    for d in (0, 1, 2):
        assert torch.equal(a._maybe_dropoff(v, d), v)
    warped = a._maybe_dropoff(v, 3)
    assert not torch.equal(warped, v)
    assert torch.allclose(warped[..., -1, :], torch.zeros(4))  # tip sunk
    assert warped[..., 0, :].abs().sum() > 0  # start intact


def test_dropoff_never_fires_at_inference():
    """Sinking the tip during decode throws away the token being conditioned
    on, and whether it fired used to depend on where KL halting stopped."""
    a = _attn(depth=4, num_layers=1)
    a.dropoff_mode, a.dropoff_step, a.dropoff_every = "warp", 3, True
    a.eval()
    v = torch.ones(1, 1, 8, 4)
    for d in range(a.depths):
        assert torch.equal(a._maybe_dropoff(v, d), v)


def test_dropoff_every_fires_on_every_pass():
    a = _attn(depth=4, num_layers=1)
    a.train()
    a.dropoff_mode, a.dropoff_step, a.dropoff_every = "warp", 3, True
    v = torch.ones(1, 1, 8, 4)
    for d in range(a.depths):
        warped = a._maybe_dropoff(v, d)
        assert torch.allclose(warped[..., -1, :], torch.zeros(4))
        assert warped[..., 0, :].abs().sum() > 0


def test_always_profile_is_registered_and_differs_from_the_one_beat_profile():
    from praxis.attention import ATTENTION_REGISTRY

    once = ATTENTION_REGISTRY["kaleido_dropoff"](_config(depth=6, num_layers=1))
    always = ATTENTION_REGISTRY["kaleido_dropoff_always"](_config(depth=6, num_layers=1))
    once.train()
    always.train()
    assert once.dropoff_every is False and always.dropoff_every is True
    v = torch.ones(1, 1, 8, 4)
    assert torch.equal(once._maybe_dropoff(v, 0), v)  # not its step
    assert not torch.equal(always._maybe_dropoff(v, 0), v)


def test_dropoff_profile_is_registered_and_sets_the_step():
    from praxis.attention import ATTENTION_REGISTRY

    a = ATTENTION_REGISTRY["kaleido_dropoff"](_config(depth=6, num_layers=1))
    assert a.dropoff_mode == "warp" and a.dropoff_step == 5 and not a.dropoff_every
    assert a(torch.randn(1, 12, 32), current_depth=5)[0].shape == (1, 12, 32)


def test_dropoff_envelope_is_the_shared_one_not_a_copy():
    """A second implementation of the ablation would drift from the arc configs."""
    from praxis.attention.causal import CausalAttention

    a = _attn(depth=2, num_layers=1)
    a.train()
    a.dropoff_mode, a.dropoff_step = "warp", 1
    v = torch.randn(1, 1, 8, 4)
    assert torch.equal(a._maybe_dropoff(v, 1), CausalAttention._dropoff_warp_value(v))


def test_arc_inherits_the_training_gate_and_the_always_schedule():
    """The fix lives in CausalAttention so every dropoff user gets it at once."""
    from praxis.attention import ATTENTION_REGISTRY

    cfg = _config(depth=6, num_layers=1)
    cfg.encoding, cfg.vocab_size, cfg.dropout = "nope", 256, 0.0
    a = ATTENTION_REGISTRY["arc_single_dropoff_always_nomem"](cfg)
    assert a.dropoff_every is True and a.dropoff_step == 5
    k = v = torch.ones(1, 1, 8, 4)
    a.eval()
    assert torch.equal(a._maybe_dropoff(k, v, 5)[1], v)  # inference: no-op
    a.train()
    assert not torch.equal(a._maybe_dropoff(k, v, 0)[1], v)  # every pass


# ------------------------------------------------------- static blend (base)



def test_turn_depth_specialization_reads_collapse():
    a = _attn(depth=4)
    assert a.training_metrics()["kaleido_turn_depth_specialization"] == pytest.approx(0.0)
    with torch.no_grad():  # identical across depths -> still collapsed
        a.turn_static.weight.copy_(torch.ones_like(a.turn_static.weight))
    assert a.training_metrics()["kaleido_turn_depth_specialization"] == pytest.approx(
        0.0, abs=1e-5
    )
    with torch.no_grad():
        a.turn_static.weight.normal_(std=1.0)
    assert a.training_metrics()["kaleido_turn_depth_specialization"] > 0.1


def test_static_blend_receives_gradient():
    a = _attn()
    a(torch.randn(2, 16, 32), current_depth=2)[0].sum().backward()
    assert a.turn_static.weight.grad[2].abs().sum() > 0


# ------------------------------------------------- SMEAR targeting interaction
def test_block_is_merge_opaque_to_the_smear_target_walker():
    """The block routes its own parameters per token, so a per-example SMEAR
    merge wrapped around it is the case MERGE_OPAQUE exists to exclude.

    The decisive reason is measurement: SMEAR would wrap `attn.turn.weight` in
    a MergedLinear routed per EXAMPLE, and `kaleido_turn_dependence` would then
    read variation caused by SMEAR's router rather than by this one.
    """
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM
    from praxis.routers.targeting import TARGET_PROFILES, discover_targets

    assert KaleidoscopeAttention.MERGE_OPAQUE is True
    cfg = PraxisConfig(
        depth=4, num_layers=1, hidden_size=64, embed_size=64, vocab_size=256,
        num_heads=4, num_queries=2, head_size=16, block_size=64,
        max_position_embeddings=128, attention_type="kaleido",
        router_type="smear", device_map="cpu",
    )
    groups, skipped = discover_targets(PraxisForCausalLM(cfg), TARGET_PROFILES["all"])
    names = [getattr(g, "name", "") for g in groups]
    for banned in ("turn", "facet", "value", "gate", "output"):
        assert not any(banned in n for n in names), f"SMEAR still targets {banned}"
    assert skipped["opaque"] > 0


def test_frozen_mirrors_are_invisible_to_the_walker_regardless():
    """Buffers are not parameters, so the dictionary could never be merged."""
    a = _attn()
    assert not any("mirror" in n for n, _ in a.named_parameters(recurse=False))


# ------------------------------------------------------ mirror utilization


# ------------------------------------------------- the blend is a span, not a hull
def test_score_matrix_is_exactly_zero_at_init():
    """Both halves zero-init, so attention opens UNIFORM over the causal prefix.

    A cleaner identity start than the softmax version gave, where a uniform
    blend still produced the dictionary mean - an arbitrary random matrix the
    model had to unlearn.
    """
    a = _attn()
    x = torch.randn(2, 12, 32)
    w = (a.turn_static.weight[0].view(1, 1, a.num_heads, a.num_mirrors)
         + TURN_MOD * torch.tanh(a.turn(x).view(2, 12, a.num_heads, a.num_mirrors)))
    assert torch.equal(w, torch.zeros_like(w))
    assert torch.equal(a._scores(w, a._faceted(0, 12)), torch.zeros(2, a.num_heads, 12, 12))


def test_weights_are_not_a_simplex():
    """Free and signed: they need not sum to one, and they may go negative."""
    a = _attn()
    with torch.no_grad():
        a.turn_static.weight.normal_(std=1.0)
    w = a.turn_static.weight[0]
    assert not torch.allclose(w.sum(), torch.tensor(1.0), atol=0.1)
    assert (w < 0).any()


def test_a_negative_weight_reaches_geometry_no_mixture_can():
    """Subtracting a mirror is the point of leaving the simplex.

    Any convex blend of the mirrors is bounded below by their pointwise minimum;
    a signed blend is not, so it reaches score patterns no softmax router of any
    temperature could produce.
    """
    a = _attn()
    M = a._faceted(0, 12)
    hull_low = M.min(dim=0).values
    signed = torch.tensor([1.0, -1.0, 0.0, 0.0])
    out = (signed[:, None, None] * M).sum(0)
    assert (out < hull_low - 1e-4).any()


def test_pre_softmax_blend_leaves_the_convex_hull_of_the_mirrors():
    """The mechanism's whole claim: mixing logits is a product of experts, so a
    spread blend synthesizes attention rows outside the hull of the dictionary -
    and at one mirror it degenerates back onto that mirror exactly."""
    a = _attn()
    # Two mirrors that mildly agree on key 3 and disagree elsewhere. A mixture
    # can only interpolate their two mild peaks; a product compounds them.
    row = torch.zeros(a.num_mirrors, 8)
    row[0, 3], row[0, 1] = 1.0, 1.0
    row[1, 3], row[1, 6] = 1.0, 1.0
    per_mirror = torch.softmax(row, dim=-1)

    spread = torch.softmax(row.sum(0), -1)
    assert spread[3] > per_mirror[:, 3].max() + 1e-6  # outside the hull

    onehot = torch.zeros(a.num_mirrors, 1)
    onehot[0] = 1.0
    assert torch.allclose(torch.softmax((onehot * row).sum(0), -1), per_mirror[0], atol=1e-6)


def test_turn_scale_is_the_effective_temperature():
    """||w|| sets the score variance, so the model owns its attention sharpness -
    a degree of freedom a simplex does not have."""
    a = _attn()
    a.train()
    for std in (0.2, 2.0):
        with torch.no_grad():
            a.turn_static.weight.normal_(std=std)
        a(torch.randn(4, 16, 32), current_depth=0)
        scale = a.training_metrics()["kaleido_turn_scale"]
        if std == 0.2:
            small = scale
    assert scale > small


def test_input_deviation_is_bounded_but_the_static_half_is_not():
    """The static blend is free (like HarmonicField.amplitudes); only the
    per-token half is capped, so it cannot run away with the logit scale."""
    a = _attn()
    with torch.no_grad():
        a.turn.weight.normal_(std=1e3)
    cond = TURN_MOD * torch.tanh(a.turn(torch.randn(4, 16, 32)))
    assert cond.abs().max() <= TURN_MOD + 1e-6


# ------------------------------------------------------------- mirror dropout
def test_mirror_dropout_is_training_only():
    a = _attn()
    w = torch.ones(4, 16, a.num_heads, a.num_mirrors)
    a.eval()
    assert torch.equal(a._mirror_dropout(w), w)
    a.train()
    torch.manual_seed(0)
    assert not torch.equal(a._mirror_dropout(w), w)


def test_mirror_dropout_zeroes_whole_mirrors_and_never_rescales():
    """A survivor keeps its exact coefficient: these are weights on frozen
    matrices, so inverted-dropout rescaling would change the softmax
    temperature rather than preserve an expectation."""
    a = _attn()
    a.train()
    torch.manual_seed(0)
    w = torch.full((8, 16, a.num_heads, a.num_mirrors), 0.7)
    out = a._mirror_dropout(w)
    vals = set(round(float(v), 6) for v in out.unique())
    assert vals <= {0.0, 0.7}
    assert 0.0 in vals and 0.7 in vals


def test_dropping_every_mirror_falls_back_to_uniform_attention():
    """SMEAR's safety property, inherited: an all-dropped blend is w = 0, which
    is this module's identity state, not a degenerate one."""
    a = _attn()
    scores = a._scores(torch.zeros(2, 12, a.num_heads, a.num_mirrors), a._faceted(0, 12))
    assert torch.equal(scores, torch.zeros_like(scores))
    probs = torch.softmax(scores.masked_fill(
        ~((torch.arange(12)[:, None] - torch.arange(12)[None, :]) >= 0), float("-inf")), -1)
    assert torch.allclose(probs[0, 0, -1], torch.full((12,), 1 / 12), atol=1e-6)


def test_turn_metrics_are_absent_at_init_rather_than_reporting_collapse():
    """Every turn ratio is 0/0 while the blend is zero. Reporting them would say
    'less than one effective mirror' and 'no mirror used' - which read as
    collapse, the opposite of an untouched identity start."""
    a = _attn()
    a.train()
    a(torch.randn(4, 16, 32))
    m = a.training_metrics()
    for k in ("kaleido_turn_modes", "kaleido_turn_negative", "kaleido_turn_scale",
              "kaleido_mirror_utilization", "kaleido_turn_static_share"):
        assert k not in m
    with torch.no_grad():
        a.turn_static.weight.normal_(std=0.5)
    a(torch.randn(4, 16, 32))
    assert set(a.training_metrics()) >= {"kaleido_turn_modes", "kaleido_turn_negative"}


def test_turn_modes_reads_collapse_and_spread():
    a = _attn()
    a.train()
    with torch.no_grad():  # everything on one mirror
        a.turn_static.weight[:] = torch.tensor([3.0, 0.0, 0.0, 0.0] * a.num_heads)
    a(torch.randn(4, 16, 32), current_depth=0)
    collapsed = a.training_metrics()["kaleido_turn_modes"]
    with torch.no_grad():  # evenly spread
        a.turn_static.weight[:] = torch.tensor([1.0, 1.0, 1.0, 1.0] * a.num_heads)
    a(torch.randn(4, 16, 32), current_depth=0)
    spread = a.training_metrics()["kaleido_turn_modes"]
    assert collapsed == pytest.approx(1.0, abs=0.15)
    assert spread > collapsed * 2


# --------------------------------------------- length invariance by resampling
def test_any_sequence_length_works_including_a_cached_decode_step():
    """No span, nothing to slice, no length that raises. T=1 is the decode case."""
    a = _attn(max_position_embeddings=64)
    for T in (1, 2, 7, 32, MIRROR_RES, 200):
        assert a(torch.randn(1, T, 32))[0].shape == (1, T, 32)


def test_geometry_is_the_same_at_every_length_in_relative_position():
    """The point of resampling: under a sequence curriculum T changes every
    batch, and the model should see ONE geometry stretched to fit rather than a
    different corner of a big matrix at each length."""
    a = _attn()
    vals = [
        float(a._faceted(0, T).detach()[0, round(0.5 * (T - 1)), round(0.25 * (T - 1))])
        for T in (128, 256, 512, 1024)
    ]
    # Converges as the resample gets finer; every length reads the same point of
    # the same underlying function.
    assert max(vals) - min(vals) < 0.15
    assert abs(vals[-1] - vals[-2]) < abs(vals[1] - vals[0])


def test_corners_are_pinned_so_the_distribution_stretches_rather_than_crops():
    """align_corners=True: the canonical grid's corners land on the sequence's."""
    a = _attn()
    for T in (16, 128):
        g = a._faceted(0, T).detach()
        assert torch.allclose(g[:, 0, 0], a.mirrors[:, 0, 0], atol=1e-5)
        assert torch.allclose(g[:, -1, -1], a.mirrors[:, -1, -1], atol=1e-5)


def test_facets_live_in_canonical_space_so_they_are_length_free_too():
    a = _attn()
    with torch.no_grad():
        a.facet_u.normal_(std=0.5)
    # Deform then resample must equal what _faceted does, at any T.
    import torch.nn.functional as F

    want = F.interpolate(
        a._canonical(1).unsqueeze(0), size=(48, 48), mode="bilinear", align_corners=True
    ).squeeze(0)
    assert torch.allclose(a._faceted(1, 48), want, atol=1e-6)


def test_gradients_flow_through_the_resample():
    a = _attn()
    with torch.no_grad():
        a.turn_static.weight.normal_(std=0.5)
    a(torch.randn(2, 100, 32), current_depth=1)[0].sum().backward()  # T != R
    assert a.facet_u.grad.abs().sum() > 0


def test_ratio_structure_survives_but_fixed_lag_smears():
    """The honest cost of relative indexing, asserted rather than described.

    A canonical previous-token band is lag 1 at R and smears as T grows, so a
    dictionary of ratio mirrors cannot express "the token immediately before" at
    long lengths. Ratio structure - the diagonal, "attend to the start" - is
    exact at every length.
    """
    import torch.nn.functional as F

    band = torch.zeros(1, 1, MIRROR_RES, MIRROR_RES)
    for i in range(1, MIRROR_RES):
        band[0, 0, i, i - 1] = 1.0
    widths = []
    for T in (MIRROR_RES, 4 * MIRROR_RES):
        up = F.interpolate(band, size=(T, T), mode="bilinear", align_corners=True)[0, 0]
        widths.append(int((up[T // 2] > 0.1).sum()))
    assert widths[0] == 1 and widths[1] > 5  # smears with length

    # The diagonal is preserved: (p, p) maps to (p, p) under any resample, up to
    # the discretization of a spike that lands between canonical cells.
    diag = torch.zeros(1, 1, MIRROR_RES, MIRROR_RES)
    diag[0, 0].fill_diagonal_(1.0)
    up = F.interpolate(diag, size=(256, 256), mode="bilinear", align_corners=True)[0, 0]
    for row in (64, 128, 192):
        assert abs(int(up[row].argmax()) - row) <= 2
