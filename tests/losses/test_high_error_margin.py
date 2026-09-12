"""HEM's margin behavior, its masking, and the HEM+ unigram margin."""

import pytest
import torch
import torch.nn.functional as F

from praxis.losses.high_error_margin import HighErrorMarginLoss

VOCAB, TOKENS = 64, 32


def logits_and_labels(seed=0, scale=3.0):
    torch.manual_seed(seed)
    return torch.randn(TOKENS, VOCAB) * scale, torch.randint(0, VOCAB, (TOKENS,))


def test_matches_the_reference_formulation():
    """Reproduces codeberg.org/mwspratling/HEMLoss ``hem_loss`` exactly: a
    per-token mean over the above-average margin violations, then a mean over
    the tokens that still carry error."""
    logits, labels = logits_and_labels()
    margin = 1.0

    targets = F.one_hot(labels, VOCAB).float()
    error = (1 - targets) * F.relu(
        logits - (logits.gather(1, labels[:, None]) - margin)
    )
    threshold = error.mean(dim=1, keepdim=True)
    above = torch.where(error >= threshold, error, torch.nan).nanmean(dim=1)
    expected = above.sum() / (above > 0).sum()

    loss = HighErrorMarginLoss(margin=margin, vocab_size=VOCAB)(
        logits=logits, labels=labels
    )
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_separated_tokens_stop_producing_gradient():
    """The continual-learning claim: once the target leads by the margin the
    token is done, and a wider lead does not lower the loss further."""
    loss_function = HighErrorMarginLoss(margin=1.0, vocab_size=VOCAB)
    logits = torch.zeros(2, VOCAB)
    labels = torch.tensor([0, 1])
    logits[0, 0] = 5.0  # separated by well over the margin
    logits[1, 1] = 0.5  # inside the margin

    per_token_grad = torch.autograd.grad(
        loss_function(logits=logits.requires_grad_(), labels=labels), logits
    )[0]

    assert per_token_grad[0].abs().sum() == 0
    assert per_token_grad[1].abs().sum() > 0
    assert loss_function.training_metrics()["hem_separated"] == 0.5


def test_masked_positions_are_excluded():
    logits, labels = logits_and_labels()
    masked = labels.clone()
    masked[:8] = -100

    full = HighErrorMarginLoss(vocab_size=VOCAB)(logits=logits, labels=labels)
    kept = HighErrorMarginLoss(vocab_size=VOCAB)(logits=logits[8:], labels=labels[8:])
    partial = HighErrorMarginLoss(vocab_size=VOCAB)(logits=logits, labels=masked)

    assert partial.item() == pytest.approx(kept.item(), rel=1e-5)
    assert partial.item() != pytest.approx(full.item(), rel=1e-5)


def test_loss_weights_reweight_the_surviving_tokens():
    logits, labels = logits_and_labels()
    weights = torch.zeros(TOKENS)
    weights[:8] = 1.0

    weighted = HighErrorMarginLoss(vocab_size=VOCAB)(
        logits=logits, labels=labels, loss_weights=weights
    )
    subset = HighErrorMarginLoss(vocab_size=VOCAB)(logits=logits[:8], labels=labels[:8])
    assert weighted.item() == pytest.approx(subset.item(), rel=1e-5)


def test_adaptive_margin_widens_for_rare_tokens():
    """HEM+ starts uniform, then tracks the unigram distribution it sees."""
    loss_function = HighErrorMarginLoss(adaptive_margin=True, vocab_size=VOCAB)
    logits, _ = logits_and_labels()

    common, rare = torch.zeros(TOKENS, dtype=torch.long), torch.full((TOKENS,), 7)
    loss_function(logits=logits, labels=common)
    start = loss_function.training_metrics()["hem_margin"]

    for _ in range(50):  # pile counts onto token 0 only
        loss_function(logits=logits, labels=common)

    loss_function(logits=logits, labels=common)
    frequent = loss_function.training_metrics()["hem_margin"]
    loss_function(logits=logits, labels=rare)
    infrequent = loss_function.training_metrics()["hem_margin"]

    assert start == pytest.approx(1.0, rel=1e-3)
    assert frequent < start < infrequent


def test_adaptive_margin_is_capped():
    """A token the stream has never produced must not get an unbounded margin."""
    loss_function = HighErrorMarginLoss(
        adaptive_margin=True, vocab_size=VOCAB, margin_ratio_cap=4.0
    )
    loss_function.token_counts[0] = 1e9
    logits, _ = logits_and_labels()

    loss_function(logits=logits, labels=torch.full((TOKENS,), 9))
    assert loss_function.training_metrics()["hem_margin"] == pytest.approx(4.0)


def test_metrics_are_finite_scalars():
    logits, labels = logits_and_labels()
    loss_function = HighErrorMarginLoss(vocab_size=VOCAB)
    loss_function(logits=logits, labels=labels)

    metrics = loss_function.training_metrics()
    assert set(metrics) == set(HighErrorMarginLoss.metric_descriptions)
    assert all(isinstance(value, float) for value in metrics.values())
