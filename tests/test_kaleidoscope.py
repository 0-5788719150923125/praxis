"""Kaleidoscope attention: frozen mirrors, input-conditional turn, per-depth facets."""

import math
from types import SimpleNamespace

import pytest
import torch

from praxis.attention.kaleidoscope import (
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
def test_turn_starts_uniform_so_the_field_is_the_dictionary_mean():
    a = _attn()
    a.train()
    a(torch.randn(3, 16, 32))
    assert a.training_metrics()["kaleido_turn_entropy"] == pytest.approx(1.0, abs=1e-5)


def test_turn_is_input_conditional_once_the_router_moves():
    """The whole claim: the blend must be a function of x, not a parameter.

    Zero dependence with a trained router means it collapsed to a constant and
    the block degraded into a single fixed matrix.
    """
    a = _attn()
    with torch.no_grad():
        a.turn.weight.normal_(std=1.0)
    a.train()
    a(torch.randn(8, 16, 32))
    m = a.training_metrics()
    assert m["kaleido_turn_dependence"] > 0.0
    assert m["kaleido_turn_entropy"] < 1.0


def test_turn_dependence_is_zero_for_a_constant_router():
    """The failure mode the metric exists to catch, forced deliberately."""
    a = _attn()
    with torch.no_grad():
        a.turn.weight.zero_()
    a.train()
    a(torch.randn(8, 16, 32))
    assert a.training_metrics()["kaleido_turn_dependence"] == pytest.approx(0.0, abs=1e-4)


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
    assert torch.equal(a._faceted(0, 16), a.mirrors[:, :16, :16])
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
    frozen = a._scores(w, a.mirrors[:, :12, :12])
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
    assert not torch.allclose(a._faceted(0, 12), a._faceted(1, 12))
    for d in range(a.depths):  # never further than the cap from the mirrors
        assert (a._faceted(d, 12) - a.mirrors[:, :12, :12]).abs().max() <= FACET_SCALE


def test_depth_index_saturates_past_the_configured_depth():
    a = _attn(depth=2)
    assert torch.equal(a._faceted(99, 8), a._faceted(1, 8))


# ------------------------------------------------------------------- the forward
def test_forward_shape_and_gradients_reach_turn_and_facets():
    a = _attn()
    x = torch.randn(2, 16, 32, requires_grad=True)
    out, _, aux = a(x, current_depth=1)
    assert out.shape == (2, 16, 32)
    assert aux == 0.0
    out.sum().backward()
    assert a.turn.weight.grad is not None and a.turn.weight.grad.abs().sum() > 0
    # facet_u is zero-init, so its gradient rides on facet_v being non-zero.
    assert a.facet_u.grad is not None and a.facet_u.grad[1].abs().sum() > 0


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


def test_sequence_past_the_span_raises_rather_than_truncating():
    a = _attn(max_position_embeddings=16)
    assert a.span == 16
    with pytest.raises(ValueError, match="span"):
        a(torch.randn(1, 17, 32))


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


def test_facet_v_gradient_unlocks_only_once_u_has_moved():
    """`d/dv (u (x) v) = u`, and u is zero-init, so v is gradient-dead at step 0.

    That reads like a bug in a full-model gradient audit, so it is asserted
    here as the intended asymmetry: u moves first, then v joins. Same shape as
    HarmonicField's fast_u / fast_v.
    """
    a = _attn()
    a(torch.randn(2, 12, 32), current_depth=0)[0].sum().backward()
    assert a.facet_u.grad.abs().sum() > 0
    assert a.facet_v.grad.abs().sum() == 0

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
def test_blend_is_base_plus_deviation_and_starts_uniform():
    """SMEAR's own form: a learned static base plus an input-conditional
    deviation. Both zero-init, so the opening blend is the dictionary mean."""
    a = _attn()
    assert torch.equal(a.turn_static.weight, torch.zeros_like(a.turn_static.weight))
    assert a.turn.bias is None  # the static term IS the bias, per depth
    a.train()
    a(torch.randn(3, 16, 32))
    assert a.training_metrics()["kaleido_turn_entropy"] == pytest.approx(1.0, abs=1e-5)
    # Undefined while both terms are zero rather than reported as a bogus split.
    assert "kaleido_turn_static_share" not in a.training_metrics()


def test_static_blend_alone_shifts_the_mixture_with_no_input_dependence():
    """A learned constant preference must be expressible WITHOUT pretending to
    be input-driven - that separation is what makes turn_dependence honest."""
    a = _attn()
    a.train()
    with torch.no_grad():
        a.turn_static.weight[0, :] = torch.tensor([3.0, 0.0, 0.0, 0.0] * a.num_heads)
    a(torch.randn(8, 16, 32), current_depth=0)
    m = a.training_metrics()
    assert m["kaleido_turn_entropy"] < 0.9  # committed to a mirror
    assert m["kaleido_turn_dependence"] == pytest.approx(0.0, abs=1e-4)
    assert m["kaleido_turn_static_share"] == pytest.approx(1.0, abs=1e-4)


def test_static_share_falls_as_the_input_term_takes_over():
    a = _attn()
    a.train()
    with torch.no_grad():
        a.turn_static.weight.normal_(std=0.5)
    a(torch.randn(8, 16, 32), current_depth=0)
    static_only = a.training_metrics()["kaleido_turn_static_share"]
    with torch.no_grad():
        a.turn.weight.normal_(std=2.0)
    a(torch.randn(8, 16, 32), current_depth=0)
    assert a.training_metrics()["kaleido_turn_static_share"] < static_only


def test_static_blend_is_per_depth():
    a = _attn(depth=4)
    a.train()
    with torch.no_grad():
        a.turn_static.weight[0].fill_(0.0)
        a.turn_static.weight[1, 0] = 5.0
    x = torch.randn(2, 12, 32)
    a(x, current_depth=0)
    e0 = a.training_metrics()["kaleido_turn_entropy"]
    a(x, current_depth=1)
    assert a.training_metrics()["kaleido_turn_entropy"] < e0


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
def test_mirror_utilization_is_full_when_the_blend_is_uniform():
    a = _attn()
    a.train()
    a(torch.randn(4, 16, 32))
    assert a.training_metrics()["kaleido_mirror_utilization"] == pytest.approx(1.0)


def test_mirror_utilization_falls_when_the_dictionary_collapses():
    """1/N is total collapse: one mirror does everything, the rest are dead."""
    a = _attn()
    a.train()
    with torch.no_grad():  # drive the static blend hard onto mirror 0
        a.turn_static.weight[:] = torch.tensor([20.0, -20.0, -20.0, -20.0] * a.num_heads)
    a(torch.randn(4, 16, 32))
    u = a.training_metrics()["kaleido_mirror_utilization"]
    assert u == pytest.approx(1.0 / a.num_mirrors, abs=1e-6)
