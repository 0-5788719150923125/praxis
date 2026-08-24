"""Regularizer registry: default selection, build, and the activation option."""

import pytest
import torch

from praxis.losses.regularizers import (
    REGULARIZER_REGISTRY,
    build_regularizers,
)


def test_default_is_contrastive_isotropy():
    reg = build_regularizers(None)
    assert len(reg) == 1
    assert reg[0].name == "contrastive"


def test_empty_list_disables_all():
    assert len(build_regularizers([])) == 0


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        build_regularizers(["does_not_exist"])


def test_multiple_regularizers_compose():
    reg = build_regularizers(list(REGULARIZER_REGISTRY.keys()))
    assert len(reg) == len(REGULARIZER_REGISTRY)
    names = {m.name for m in reg}
    assert "contrastive" in names and "activation_reg" in names


def test_activation_regularizer_forward_and_metrics():
    reg = build_regularizers(["activation"])[0]
    h = torch.randn(2, 8, 16)
    ids = torch.randint(0, 32, (2, 8))
    loss = reg(h, ids)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss >= 0
    m = reg.training_metrics()
    assert set(m) == {"activation_ar", "activation_tar"}


def test_activation_regularizer_single_token_no_tar():
    reg = build_regularizers(["activation"])[0]
    h = torch.randn(2, 1, 16)
    loss = reg(h, torch.zeros(2, 1, dtype=torch.long))
    assert torch.isfinite(loss)
    assert reg.training_metrics()["activation_tar"] == 0.0


def test_contrastive_isotropy_is_batch_size_invariant():
    """Loss and metric are means over pairs, so replicating one sequence across
    the batch must not change either. The pair count has to follow the batch
    whether or not the padding mask broadcasts it - encoder models take the
    unmasked path (token-level ids do not align to patch reps), token models
    take the masked one, and a batch-dependent scale would make the two
    disagree and would drift as the batch governor re-tiers."""
    reg = build_regularizers(["contrastive_isotropy"])[0]
    torch.manual_seed(0)
    seq_len, dim = 32, 16
    # Anisotropic on purpose: a shared component pushes cosines past the margin
    # so the loss is nonzero and the comparison is not trivially 0 == 0.
    one = torch.randn(1, seq_len, dim) * 0.3 + 2.0 * torch.randn(1, 1, dim)

    ref_loss = ref_metric = None
    for batch in (1, 2, 5):
        h = one.repeat(batch, 1, 1)
        # Aligned ids (mask applies) and mismatched ids (mask skipped) must agree.
        aligned = reg(h, torch.randint(1, 99, (batch, seq_len)))
        aligned_metric = reg.training_metrics()["repr_anisotropy"]
        skipped = reg(h, torch.randint(1, 99, (batch, seq_len * 4)))
        skipped_metric = reg.training_metrics()["repr_anisotropy"]

        assert aligned > 0
        assert torch.allclose(aligned, skipped)
        assert aligned_metric == pytest.approx(skipped_metric)
        if ref_loss is None:
            ref_loss, ref_metric = aligned, aligned_metric
        assert torch.allclose(aligned, ref_loss)
        assert aligned_metric == pytest.approx(ref_metric)


def test_repr_anisotropy_equals_squared_magnetization():
    """The logged anisotropy is the token ensemble's magnetization: for unit
    vectors u_i with m = mean(u_i), mean off-diagonal cosine is
    (T * ||m||^2 - 1) / (T - 1). Guards the O(T^2) metric against an O(T * D)
    reformulation, and pins the identity a per-depth order parameter would use."""
    reg = build_regularizers(["contrastive_isotropy"])[0]
    torch.manual_seed(1)
    batch, seq_len, dim = 3, 64, 32
    h = torch.randn(batch, seq_len, dim) + 1.5 * torch.randn(1, 1, dim)

    reg(h, torch.randint(1, 99, (batch, seq_len)))
    measured = reg.training_metrics()["repr_anisotropy"]

    u = torch.nn.functional.normalize(h, dim=-1)
    mag_sq = u.mean(dim=1).pow(2).sum(dim=-1)  # ||m||^2 per sequence
    closed_form = ((seq_len * mag_sq - 1.0) / (seq_len - 1)).mean().item()
    assert measured == pytest.approx(closed_form, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize(
    "case,builder,anisotropy,dimensions,nematic",
    [
        # The two readings the mean cosine gets BACKWARDS, which is why the
        # spectral pair exists. Expected values are (low, high) bounds.
        (
            "isotropic",
            lambda t, d: torch.randn(t, d),
            (None, 0.1),
            (0.9, None),
            (None, 0.1),
        ),
        # Only translated: mean cosine screams collapse, the space is untouched.
        (
            "translated",
            lambda t, d: torch.randn(t, d) * 0.1 + torch.randn(1, d),
            (0.9, None),
            (0.9, None),
            (None, 0.1),
        ),
        # Genuinely degenerate: mean cosine reports a healthy isotropic space.
        (
            "rank_3",
            lambda t, d: torch.randn(t, 3) @ torch.randn(3, d),
            (None, 0.2),
            (None, 0.15),
            (0.3, None),
        ),
    ],
)
def test_spectral_metrics_catch_what_mean_cosine_inverts(
    case, builder, anisotropy, dimensions, nematic
):
    reg = build_regularizers(["contrastive_isotropy"])[0]
    torch.manual_seed(0)
    seq_len, dim = 512, 111
    h = builder(seq_len, dim).unsqueeze(0)
    reg(h, None)
    m = reg.training_metrics()

    for name, value, (low, high) in (
        ("repr_anisotropy", m["repr_anisotropy"], anisotropy),
        ("repr_dimensions", m["repr_dimensions"], dimensions),
        ("repr_nematic", m["repr_nematic"], nematic),
    ):
        if low is not None:
            assert value > low, f"{case}: {name} = {value:.4f}, expected > {low}"
        if high is not None:
            assert value < high, f"{case}: {name} = {value:.4f}, expected < {high}"


def test_spectral_metrics_normalized_against_their_isotropic_null():
    """Both readings divide out their finite-sample null, so an isotropic cloud
    reads 1.0 and 0.0 at ANY T and D. Without that the sequence-length
    curriculum would move the metrics on its own and the chart would show a
    geometry change where there is only a shape change."""
    reg = build_regularizers(["contrastive_isotropy"])[0]
    torch.manual_seed(0)
    for seq_len in (256, 512, 2048):
        for dim in (64, 111, 384):
            reg(torch.randn(1, seq_len, dim), None)
            m = reg.training_metrics()
            assert m["repr_dimensions"] == pytest.approx(
                1.0, abs=0.05
            ), f"T={seq_len} D={dim}: repr_dimensions = {m['repr_dimensions']:.4f}"
            assert (
                m["repr_nematic"] < 0.05
            ), f"T={seq_len} D={dim}: repr_nematic = {m['repr_nematic']:.4f}"


def test_nematic_sees_antipodal_domains_that_mean_cosine_cancels():
    """Two clusters facing opposite directions - the multi-domain structure the
    magnet picture is about. Their mean directions cancel, so mean cosine reads
    like a healthy isotropic space; the axis order parameter reads near 1."""
    reg = build_regularizers(["contrastive_isotropy"])[0]
    torch.manual_seed(0)
    seq_len, dim = 512, 111
    axis = torch.randn(1, dim)
    half = seq_len // 2
    h = torch.cat(
        [axis + 0.1 * torch.randn(half, dim), -axis + 0.1 * torch.randn(half, dim)]
    ).unsqueeze(0)

    reg(h, None)
    m = reg.training_metrics()
    assert abs(m["repr_anisotropy"]) < 0.1  # blind
    assert m["repr_nematic"] > 0.9  # sees it
    assert m["repr_dimensions"] < 0.1  # and it is genuinely one-dimensional


def test_contrastive_active_frac_reports_hinge_saturation():
    """A hinge that is never a hinge is a linear penalty at constant maximum
    gradient. The fraction of pairs above the margin is how that shows up."""
    reg = build_regularizers(["contrastive_isotropy"])[0]
    torch.manual_seed(0)
    seq_len, dim = 64, 32

    # Every pair above the margin: one direction plus a whisper of noise.
    saturated = torch.randn(1, dim) + 0.01 * torch.randn(1, seq_len, dim)
    loss = reg(saturated, None)
    m = reg.training_metrics()
    assert m["contrastive_active_frac"] == pytest.approx(1.0)
    # Saturated hinge => loss is exactly the mean cosine minus the margin.
    assert loss.item() == pytest.approx(m["repr_anisotropy"] - 0.5, abs=1e-5)

    # No pair above the margin: the hinge is inactive and the term is dead.
    reg(torch.randn(1, seq_len, dim), None)
    assert reg.training_metrics()["contrastive_active_frac"] < 0.05
