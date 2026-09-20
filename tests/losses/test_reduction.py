import pytest
import torch
import torch._dynamo as dynamo

from praxis.losses.reduction import weighted_reduce

# ------------------------------------------------------------------------------
# reduction
# ------------------------------------------------------------------------------
# weighted_reduce is on the per-step hot path inside the compiled model, so the
# degenerate denominator has to be handled without a data-dependent branch.


DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _weights(kind, n=64):
    torch.manual_seed(0)
    if kind == "all_zero":
        return torch.zeros(n)
    if kind == "tiny":
        return torch.full((n,), 1e-9)
    if kind == "sparse":
        return torch.rand(n) * (torch.rand(n) > 0.9)
    return torch.rand(n)


# ── unweighted paths keep plain reduction semantics ───────────────────────


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_no_weights_matches_plain_reduction(reduction):
    loss = torch.rand(8)
    expected = {"mean": loss.mean(), "sum": loss.sum(), "none": loss}[reduction]
    assert torch.equal(weighted_reduce(loss, reduction=reduction), expected)


# ── weighted mean ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", DTYPES)
def test_weighted_mean_is_sum_over_weight_sum(dtype):
    loss = torch.rand(64, dtype=dtype)
    w = _weights("normal").to(dtype)
    expected = (loss * w).sum() / w.sum()
    assert torch.equal(weighted_reduce(loss, loss_weights=w), expected)


def test_masked_labels_leave_the_denominator():
    loss = torch.rand(64)
    w = torch.ones(64)
    labels = torch.zeros(64, dtype=torch.long)
    labels[:16] = -100
    # Only the 48 unmasked positions count, so this is their plain mean.
    assert torch.allclose(
        weighted_reduce(loss, labels=labels, loss_weights=w), loss[16:].mean()
    )


# ── degenerate denominator ────────────────────────────────────────────────
# Weights are non-negative, so a zero denominator implies a zero numerator.
# The result must be 0.0 rather than NaN, and must stay differentiable.


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["all_zero", "tiny", "sparse", "normal"])
def test_zero_denominator_yields_zero_not_nan(dtype, kind):
    loss = torch.rand(64, dtype=dtype, requires_grad=True)
    w = _weights(kind).to(dtype)
    out = weighted_reduce(loss, loss_weights=w)
    assert torch.isfinite(out), f"{kind}/{dtype} produced {out}"
    if kind == "all_zero":
        assert out.item() == 0.0


def test_fully_masked_batch_is_zero_and_differentiable():
    loss = torch.rand(64, requires_grad=True)
    labels = torch.full((64,), -100)
    out = weighted_reduce(loss, labels=labels, loss_weights=torch.ones(64))
    assert out.item() == 0.0
    # The backward pass still has to run - a fully masked microbatch must not
    # break the step for the other losses sharing the graph.
    (grad,) = torch.autograd.grad(out, loss)
    assert torch.isfinite(grad).all()
    assert torch.equal(grad, torch.zeros_like(grad))


# ── compiled hot path ─────────────────────────────────────────────────────


def test_no_graph_break_under_dynamo():
    """A Python branch on the denominator syncs the device every step."""
    dynamo.reset()
    loss = torch.rand(64)
    labels = torch.randint(0, 5, (64,))
    labels[::7] = -100
    explanation = dynamo.explain(weighted_reduce)(
        loss, labels=labels, loss_weights=torch.rand(64)
    )
    assert explanation.graph_break_count == 0, explanation.break_reasons
