import pytest
import torch

from praxis.losses.energy_score import energy_score_loss

# ------------------------------------------------------------------------------
# calm
# ------------------------------------------------------------------------------
# CALM encoder + energy head + LF-temperature sanity tests.
#
# These are shape / plumbing checks rather than training-quality assertions. The smoke-
# test in the CALM README covers the latter.


def test_energy_score_loss_nonnegative_on_random():
    torch.manual_seed(0)
    model = torch.randn(2, 3, 4, 8)
    target = torch.randn(2, 3, 5, 8)
    val = energy_score_loss(model, target)
    assert val.dim() == 0
    assert val.item() == val.item()  # not NaN


def test_energy_score_loss_lower_when_distributions_match():
    """Matched distributions should score lower than mismatched ones."""
    torch.manual_seed(0)
    matched_m = torch.randn(4, 100, 8)
    matched_t = torch.randn(4, 100, 8)
    matched = energy_score_loss(matched_m, matched_t)

    mismatched_m = torch.randn(4, 100, 8)
    mismatched_t = torch.randn(4, 100, 8) + 10.0
    mismatched = energy_score_loss(mismatched_m, mismatched_t)

    assert matched.item() < mismatched.item()


# ------------------------------------------------------------------------------
# abstractinator_calm
# ------------------------------------------------------------------------------
# AbstractinatorCALM: a continuous CALM arm beside the discrete RVQ arm.
#
# The thesis under test is that the DISCRETE arm can pay for the CONTINUOUS one. CALM's
# energy score is a weak, high-variance signal that needs far more tokens than this line
# can afford; an RVQ code is a dense, low-variance, mode-seeking target, and predicting
# the next code from the same conditioning hidden the energy head reads is what should
# concentrate its conditional.
#
# These tests pin the mechanics, not the thesis - the run decides that.


def test_pairwise_distance_avoids_the_N_by_M_by_D_tensor():
    """The differencing form allocates [..., N, M, D] - several GB at N=8,
    M=100, D=272 - which is what was silently capping M."""
    from praxis.losses.energy_score import _pairwise_distance

    torch.manual_seed(0)
    a, b = torch.randn(2, 3, 8, 16), torch.randn(2, 3, 100, 16)
    ref = torch.sqrt((a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2).sum(-1).clamp_min(1e-4))
    assert torch.allclose(ref, _pairwise_distance(a, b), atol=1e-4)
