"""KaleidoscopeAttention: frozen mirrors, an input-conditional signed blend (the
turn), per-depth facets, ghostmax, dropoff, the 1/k^alpha envelope, the ratio/lag
coordinate split, and zoom."""

import inspect
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from praxis import registry
from praxis.attention.causal import CausalAttention
from praxis.attention.kaleidoscope import (
    FACET_SCALE,
    MIRROR_RES,
    TURN_MOD,
    KaleidoscopeAttention,
    zoom_ladder,
)

# Keywords the constructor takes (profile arguments), as opposed to config fields.
MODULE_KWARGS = set(inspect.signature(KaleidoscopeAttention).parameters) - {"config"}


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


def _attn(seed=0, **over):
    """Build under a given global seed; profile arguments go to the constructor,
    the rest onto the config."""
    kwargs = {k: over.pop(k) for k in list(over) if k in MODULE_KWARGS}
    torch.manual_seed(seed)
    return KaleidoscopeAttention(_config(**over), **kwargs)


def _blend(a, x, depth=0):
    """The blend ``w`` exactly as the forward builds it (outside training)."""
    B, T, _ = x.shape
    cond = TURN_MOD * torch.tanh(a.turn(x).view(B, T, a.num_heads, a.num_mirrors))
    return cond + a.turn_static.weight[depth].view(1, 1, a.num_heads, a.num_mirrors)


def _causal(scores):
    T = scores.shape[-1]
    pos = torch.arange(T)
    return scores.masked_fill(pos[None, :] > pos[:, None], float("-inf"))


# ------------------------------------------------------------------- the mirrors
def test_mirrors_are_canonical_resolution_not_sequence_length():
    """A mirror is a function on the unit square, not a `[T, T]` table."""
    a = _attn()
    assert a.mirrors.shape == (a.num_mirrors, MIRROR_RES, MIRROR_RES)
    assert a.facet_u.shape == (a.depths, a.num_mirrors, MIRROR_RES)


def test_mirrors_are_frozen_and_not_parameters():
    """The dictionary is architecture, not learned state."""
    a = _attn()
    names = {n for n, _ in a.named_parameters()}
    assert not any("mirror" in n for n in names)
    assert not a.mirrors.requires_grad
    # Non-persistent: N * span^2 floats stay out of every checkpoint.
    assert "mirrors" not in a.state_dict()


def test_mirrors_are_deterministic_across_constructions():
    """Non-persistent buffers only work if construction reproduces them, whatever
    the global RNG was doing."""
    assert torch.equal(_attn(seed=0).mirrors, _attn(seed=1).mirrors)


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


# ------------------------------------------------------------------- the facets
def test_blend_and_facets_are_identity_at_init():
    """Both halves of the blend zero-init, so the score matrix is exactly zero and
    attention opens UNIFORM over the causal prefix - a cleaner start than a
    softmax blend, whose uniform mix is the dictionary mean, an arbitrary random
    matrix to unlearn. It is also where an all-dropped blend (w = 0) falls back
    to, so SMEAR's safety property is inherited. The facets start undeformed."""
    a = _attn()
    assert torch.equal(a._canonical(0), a.mirrors)
    assert a.training_metrics()["kaleido_facet_strength"] == pytest.approx(0.0)
    assert not a.turn.weight.any() and not a.turn_static.weight.any()

    x = torch.randn(2, 12, 32)
    w = _blend(a, x)
    assert not w.any()
    scores = a._scores(w, a._faceted(0, 12))
    assert not scores.any()
    probs = torch.softmax(_causal(scores), -1)
    assert torch.allclose(probs[0, 0, -1], torch.full((12,), 1 / 12), atol=1e-6)


def test_facet_deformation_does_not_factor_out_of_the_mixture():
    """The reason the bias is PER MIRROR rather than one shared matrix.

    A deformation added to every mirror alike would factor straight back out of
    the blend and reduce to a per-depth score bias. Per-mirror facets must
    survive the mixture as something the blend cannot undo.
    """
    a = _attn()
    with torch.no_grad():
        a.facet_u.normal_(std=0.5)
        a.facet_v.normal_(std=0.5)
        a.turn.weight.normal_(std=1.0)
    x = torch.randn(2, 12, 32)
    w = _blend(a, x)

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
@pytest.mark.parametrize(
    "variant", [{}, {"alpha": 1.0}, {"coords": "split"}], ids=["flat", "pink", "split"]
)
def test_forward_shape_and_gradients_reach_the_blend(variant):
    a = _attn(**variant).train()
    x = torch.randn(2, 16, 32, requires_grad=True)
    out, _, aux = a(x, current_depth=1)
    assert out.shape == (2, 16, 32)
    assert aux == 0.0
    out.sum().backward()
    # Both halves of the blend move from step 0: dS/dbeta_k and dS/dw_k are the
    # mirrors themselves, which are non-zero whatever the weights are.
    assert a.turn_static.weight.grad[1].abs().sum() > 0
    assert a.turn.weight.grad.abs().sum() > 0
    assert a.gate.weight.grad.abs().sum() > 0
    assert not a.envelope.requires_grad


def test_window_size_bounds_the_reach():
    """A token more than ``window_size`` back is invisible."""
    a = _attn(window_size=3).eval()
    x = torch.randn(1, 16, 32)
    edited = x.clone()
    edited[:, 0] += 1.0
    with torch.no_grad():
        base, moved = a(x)[0], a(edited)[0]
    assert torch.allclose(moved[:, 4:], base[:, 4:], atol=1e-6)
    assert not torch.allclose(moved[:, 1:4], base[:, 1:4])


def test_patch_config_forces_a_single_head():
    """The count is corrected so config.json reports the head actually built, and
    the correction is idempotent (it runs again from ``__init__``)."""
    cfg = _config(num_heads=8, num_queries=4)
    a = KaleidoscopeAttention(cfg)
    assert cfg.num_heads == 1 and cfg.num_queries == 1
    assert a.num_heads == 1
    # head_size is a WIDTH and is left alone.
    assert a.head_dim == cfg.head_size
    KaleidoscopeAttention.patch_config(cfg)
    assert cfg.num_heads == 1 and cfg.head_size == 16
    # Unset, the one head spans the hidden size.
    cfg = _config(num_heads=4, head_size=None)
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

    The forward is checked against the literal construction - append a zero
    logit whose value is zero, softmax over the wider row, drop the ghost -
    because the identity is the whole reason no extra column is materialized.
    """
    a = _attn().eval()
    with torch.no_grad():
        a.turn.weight.normal_(std=1.0)
    B, T, H = 2, 12, a.num_heads
    x = torch.randn(B, T, 32)
    with torch.no_grad():
        ours = a(x)[0]
        s = _causal(a._scores(_blend(a, x), a._faceted(0, T)))
        v = a.value(x).view(B, T, H, a.head_dim).transpose(1, 2)
        wide = torch.cat([torch.zeros(B, H, T, 1), s], dim=-1)
        v_ghost = torch.cat([torch.zeros(B, H, 1, a.head_dim), v], dim=-2)
        attended = (torch.softmax(wide, -1) @ v_ghost).transpose(1, 2).reshape(B, T, -1)
        literal = a.output(attended * F.silu(a.gate(x)))
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


def test_dropoff_profiles_set_the_step_and_the_schedule():
    """``kaleido_dropoff`` sinks only at ``depth - num_layers``; the ``_always``
    profile at every pass."""
    cfg = dict(depth=6, num_layers=1)
    once = registry.lookup("attention", "kaleido_dropoff")(_config(**cfg)).train()
    always = registry.lookup("attention", "kaleido_dropoff_always")(_config(**cfg))
    always.train()
    assert once.dropoff_mode == "warp" and once.dropoff_step == 5
    assert once.dropoff_every is False and always.dropoff_every is True
    v = torch.ones(1, 1, 8, 4)
    assert torch.equal(once._maybe_dropoff(v, 0), v)  # not its step
    assert not torch.equal(always._maybe_dropoff(v, 0), v)
    assert once(torch.randn(1, 12, 32), current_depth=5)[0].shape == (1, 12, 32)


def test_dropoff_envelope_is_the_shared_one_not_a_copy():
    """A second implementation of the ablation would drift from the arc configs."""
    a = _attn(depth=2, num_layers=1)
    a.train()
    a.dropoff_mode, a.dropoff_step = "warp", 1
    v = torch.randn(1, 1, 8, 4)
    assert torch.equal(a._maybe_dropoff(v, 1), CausalAttention._dropoff_warp_value(v))


# ------------------------------------------------------- static blend (base)


def test_turn_depth_specialization_reads_collapse():
    a = _attn(depth=4)
    assert a.training_metrics()["kaleido_turn_depth_specialization"] == pytest.approx(
        0.0
    )
    with torch.no_grad():  # identical across depths -> still collapsed
        a.turn_static.weight.copy_(torch.ones_like(a.turn_static.weight))
    assert a.training_metrics()["kaleido_turn_depth_specialization"] == pytest.approx(
        0.0, abs=1e-5
    )
    with torch.no_grad():
        a.turn_static.weight.normal_(std=1.0)
    assert a.training_metrics()["kaleido_turn_depth_specialization"] > 0.1


# ------------------------------------------------- SMEAR targeting interaction
def test_block_is_merge_opaque_to_the_smear_target_walker():
    """The block routes its own parameters per token, so a per-example SMEAR
    merge wrapped around it is the case MERGE_OPAQUE exists to exclude: SMEAR
    would wrap ``turn.weight`` in a MergedLinear routed per EXAMPLE, and
    ``kaleido_turn_dependence`` would read SMEAR's routing rather than this
    block's. The walker honouring the flag is tests/routers/test_smear.py's."""
    assert KaleidoscopeAttention.MERGE_OPAQUE is True


# ------------------------------------------------- the blend is a span, not a hull
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


# ------------------------------------------------------------- mirror dropout
def test_mirror_dropout_zeroes_whole_mirrors_in_training_only():
    """A survivor keeps its exact coefficient: these are weights on frozen
    matrices, so inverted-dropout rescaling would change the softmax
    temperature rather than preserve an expectation."""
    a = _attn()
    w = torch.full((8, 16, a.num_heads, a.num_mirrors), 0.7)
    a.eval()
    assert torch.equal(a._mirror_dropout(w), w)
    a.train()
    torch.manual_seed(0)
    vals = set(round(float(v), 6) for v in a._mirror_dropout(w).unique())
    assert vals == {0.0, 0.7}


def test_turn_metrics_are_absent_at_init_rather_than_reporting_collapse():
    """Every turn ratio is 0/0 while the blend is zero. Reporting them would say
    'less than one effective mirror' and 'no mirror used' - which read as
    collapse, the opposite of an untouched identity start."""
    a = _attn()
    a.train()
    a(torch.randn(4, 16, 32))
    m = a.training_metrics()
    for k in (
        "kaleido_turn_modes",
        "kaleido_turn_negative",
        "kaleido_turn_scale",
        "kaleido_mirror_utilization",
        "kaleido_turn_static_share",
    ):
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
@pytest.mark.parametrize("coords", ["ratio", "split"])
def test_any_sequence_length_works_including_a_cached_decode_step(coords):
    """No span, nothing to slice, no length that raises. T=1 is the decode case."""
    a = _attn(coords=coords, max_position_embeddings=64)
    for T in (1, 2, 7, 32, MIRROR_RES, 129, 200):
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
    want = F.interpolate(
        a._canonical(1).unsqueeze(0), size=(48, 48), mode="bilinear", align_corners=True
    ).squeeze(0)
    assert torch.allclose(a._faceted(1, 48), want, atol=1e-6)


@pytest.mark.parametrize("coords", ["ratio", "split"])
def test_gradients_flow_through_the_resample(coords):
    """``interpolate`` and ``grid_sample`` must stay differentiable, or a half of
    the dictionary is frozen dead."""
    a = _attn(coords=coords).train()
    with torch.no_grad():
        # The score is einsum(w, mirrors), so a zero blend gives the WHOLE
        # dictionary zero gradient. Open the turn first, or this passes
        # vacuously for the wrong reason.
        a.turn_static.weight.normal_(std=0.5)
        a.facet_u.normal_(std=0.1)
    a(torch.randn(2, 100, 32), current_depth=1)[0].sum().backward()  # T != R
    assert a.facet_u.grad[:, : a.n_ratio].abs().sum() > 0
    if a.n_lag:
        assert a.facet_u.grad[:, a.n_ratio :].abs().sum() > 0


# ------------------------------------------------- the 1/k^alpha envelope
def test_base_dictionary_is_flat_and_pink_is_not():
    """alpha=0 is the paper's own alpha=0 corner, not an oversight. Without the
    split, the pink envelope ranks the whole dictionary 1..N."""
    a, p = _attn(), _attn(alpha=1.0)
    assert a.alpha == 0.0 and p.alpha == 1.0
    assert torch.equal(a.envelope, torch.ones(a.num_mirrors))
    assert p.env_rank.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert torch.allclose(p.envelope, torch.tensor([1.0, 0.5, 1 / 3, 0.25]))


def test_flat_and_pink_share_the_same_draw_so_the_ab_isolates_the_envelope():
    a, p = _attn(seed=0), _attn(seed=1, alpha=1.0)
    assert torch.allclose(a.mirrors[0], p.mirrors[0])
    assert torch.allclose(a.mirrors[2] / 3.0, p.mirrors[2], atol=1e-6)


def test_envelope_fight_is_calibrated_against_proposition_iii():
    """0 = accepting the prior (so it has only cost capacity), 1 = compensating
    exactly, >1 = the suppressed mirrors dominating. The paper's claim is that
    capacity exists only above 0."""
    p = _attn(alpha=1.0).train()
    N = p.num_mirrors
    for weights, want in [
        (torch.ones(N), 0.0),  # flat |w| -> accepting
        (torch.arange(1.0, N + 1), 1.0),  # |w| ~ k^alpha -> exact
        (torch.arange(1.0, N + 1) ** 2, 2.0),  # over-compensating
    ]:
        with torch.no_grad():
            p.turn_static.weight[:] = weights.repeat(p.num_heads)
        p(torch.randn(4, 16, 32), current_depth=0)
        assert p.training_metrics()["kaleido_envelope_fight"] == pytest.approx(
            want, abs=1e-3
        )


def test_envelope_fight_is_absent_without_an_envelope():
    a = _attn()
    a.train()
    with torch.no_grad():
        a.turn_static.weight.normal_(std=0.5)
    a(torch.randn(4, 16, 32))
    assert "kaleido_envelope_fight" not in a.training_metrics()


# -------------------------------------------------- split coordinate systems
def test_default_coords_are_ratio_only_and_split_halves_the_dictionary():
    """The warp is opt-in, so -i and -j stay exactly the block they trained."""
    a = _attn()
    assert a.coords == "ratio"
    assert (a.n_ratio, a.n_lag) == (a.num_mirrors, 0)
    assert (_attn(coords="split").n_ratio, _attn(coords="split").n_lag) == (2, 2)


def test_unknown_coords_rejected():
    with pytest.raises(ValueError, match="coordinates"):
        KaleidoscopeAttention(_config(), coords="polar")


def test_split_shares_the_dictionary_draw_with_ratio():
    """Same mirrors whatever the global seed - so a ratio/split A/B isolates the
    coordinates."""
    assert torch.equal(_attn(seed=0).mirrors, _attn(seed=1, coords="split").mirrors)


def test_lag_coordinate_resolves_a_single_token_at_length():
    """The whole point of the warp: lag 1 stays one token wide at any T.

    A one-cell canonical feature at the column the log warp assigns to lag 1
    must resample to a band of width 1, where the ratio half smears the same
    feature across T/R positions.
    """
    a = _attn(coords="split")
    R = a.resolution
    for T in (256, 512):
        col = round(math.log1p(1.0) / math.log1p(T - 1) * (R - 1))
        m = torch.zeros(a.num_mirrors, R, R)
        m[a.n_ratio, :, col] = 1.0
        row = a._resample(m, T)[a.n_ratio][T // 2]
        nz = (row > 1e-3).nonzero().flatten()
        assert len(nz) == 1, f"T={T}: lag band spans {len(nz)} positions"
        assert T // 2 - int(nz[0]) == 1


def test_ratio_coordinate_still_smears_fixed_lag():
    """The control for the test above - and why both halves exist."""
    a = _attn(coords="split")
    R, T = a.resolution, 512
    m = torch.zeros(a.num_mirrors, R, R)
    m[0, :, 1] = 1.0
    row = a._resample(m, T)[0][T // 2]
    nz = (row > 1e-3).nonzero().flatten()
    assert len(nz) > 8


def test_lag_coordinate_preserves_causality():
    """Nothing in the warp may sample across the diagonal."""
    a = _attn(coords="split")
    T = 32
    g = a._lag_grid(T, torch.device("cpu"), torch.float32)[0]
    # Lag is clamped at 0, so future keys all collapse onto the diagonal cell
    # rather than reaching around to a different part of the mirror.
    assert torch.allclose(g[0, 1:, 0], g[0, 0, 0])


def test_envelope_ranks_within_each_coordinate_group():
    """Global ranking would suppress the lag half for being stored second."""
    a = _attn(coords="split", alpha=1.0)
    assert a.env_rank.tolist() == [1.0, 2.0, 1.0, 2.0]
    assert torch.allclose(a.envelope, torch.tensor([1.0, 0.5, 1.0, 0.5]))
    # Both coordinate systems get the same capacity ladder.
    assert a.envelope[: a.n_ratio].tolist() == a.envelope[a.n_ratio :].tolist()


def test_lag_share_reported_only_when_split():
    a = _attn(coords="split")
    a.train()
    a.turn_static.weight.data.normal_(0, 0.5)
    a(torch.randn(2, 16, 32), current_depth=0)
    share = a.training_metrics()["kaleido_lag_share"]
    assert 0.0 <= share <= 1.0

    b = _attn()
    b.train()
    b.turn_static.weight.data.normal_(0, 0.5)
    b(torch.randn(2, 16, 32), current_depth=0)
    assert "kaleido_lag_share" not in b.training_metrics()


def test_lag_share_reads_one_when_only_lag_mirrors_are_used():
    a = _attn(coords="split")
    a.train()
    w = torch.zeros(2, 16, a.num_heads, a.num_mirrors)
    w[..., a.n_ratio :] = 1.0
    a._note_turn(w, torch.zeros_like(w), torch.zeros(1, 1, a.num_heads, a.num_mirrors))
    assert a._metrics["kaleido_lag_share"] == pytest.approx(1.0)


# ------------------------------------------------------------------- zoom
def test_zoom_ladder_steps_outward_from_the_identity():
    assert zoom_ladder(1) == (1.0,)
    assert zoom_ladder(5) == (1 / 3, 1 / 2, 1.0, 2.0, 3.0)
    assert all(z > 0 for z in zoom_ladder(12)) and len(zoom_ladder(12)) == 12


@pytest.mark.parametrize("zoom", [(1.0, 0.0), (-2.0,)])
def test_zoom_factors_must_be_positive(zoom):
    """The fold is odd, so a negative factor is the same mirror reversed, and
    zero is constant across the row and invisible to softmax."""
    with pytest.raises(ValueError, match="positive"):
        _attn(zoom=zoom)


def test_unit_zoom_is_the_plain_ratio_mirror():
    """z = 1 folds nothing: the folded grid reads the same geometry the plain
    resample does, at any length (up to float error in the fold coordinates)."""
    plain, unit = _attn(), _attn(zoom=(1.0,))
    assert unit.zoom_factor.tolist() == [1.0] * unit.n_ratio
    for T in (16, 100):
        torch.testing.assert_close(
            unit._faceted(0, T), plain._faceted(0, T), rtol=0, atol=1e-4
        )
    zoomed = _attn(zoom=True)
    assert zoomed.zoom == zoom_ladder(zoomed.n_ratio)
    assert not torch.allclose(zoomed._faceted(0, 100), plain._faceted(0, 100))


def test_mix_norm_scales_scores_by_inverse_root_n():
    plain, normed = _attn(), _attn(mix_norm=True)
    w = torch.randn(2, 12, plain.num_heads, plain.num_mirrors)
    mirrors = plain._faceted(0, 12)
    torch.testing.assert_close(
        normed._scores(w, mirrors), plain._scores(w, mirrors) * plain.num_mirrors**-0.5
    )
