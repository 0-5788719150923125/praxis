"""Tests for praxis/policies/preference.py: the reference-free preference
margin over paired chosen/rejected responses."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.policies.preference import PreferencePolicy
from praxis.tasks import TaskType

CHOSEN = int(TaskType.PREF_CHOSEN)
REJECTED = int(TaskType.PREF_REJECTED)


def _config(**kwargs):
    return PraxisConfig(
        vocab_size=64,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=2,
        decoder_type="sequential",
        **kwargs,
    )


def _batch(vocab=64, seq=48, pairs=(1, 1)):
    """One row per side, tagged and paired. Labels follow the production
    convention: PRE-SHIFTED (input_ids[..., 1:]), while the task, assistant and
    pair channels stay full-length.

    ``pairs`` sets the pair id on each row, so a test can put the two sides in
    the same comparison (the default) or in different ones.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, seq, vocab, requires_grad=True)
    labels = torch.randint(0, vocab, (2, seq - 1))
    task = torch.full((2, seq), CHOSEN, dtype=torch.long)
    task[1] = REJECTED
    mask = torch.ones(2, seq, dtype=torch.uint8)
    pair_ids = torch.zeros(2, seq, dtype=torch.long)
    pair_ids[0] = pairs[0]
    pair_ids[1] = pairs[1]
    return logits, labels, task, mask, pair_ids


def test_margin_loss_and_gradient_direction():
    """The loss is -logsigmoid(beta * margin - gamma), and its gradient pushes
    chosen token logprobs UP and rejected token logprobs DOWN."""
    policy = PreferencePolicy(_config()).train()
    logits, labels, task, mask, pair_ids = _batch()
    loss, metrics = policy(
        logits, labels, assistant_mask=mask, task_type_ids=task, pair_ids=pair_ids
    )
    assert loss is not None and torch.isfinite(loss)
    assert metrics["preference_pairs"] == 1.0
    # The term is scaled by the preference share of the batch so that rl_weight
    # is the honest per-token ratio against the main CE, which normalises over
    # every supervised position rather than over this policy's own tokens.
    expected = (
        policy.rl_weight
        * metrics["preference_share"]
        * -torch.nn.functional.logsigmoid(
            torch.tensor(policy.BETA * metrics["preference_margin"] - policy.GAMMA)
        )
    )
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    loss.backward()
    # Gradient at the label logit: negative = raising that logit lowers the
    # loss. Chosen row targets should be pushed up, rejected pushed down.
    # Labels are pre-shifted, so logit position t pairs with labels[t].
    g = logits.grad
    L = labels.size(1)
    chosen_grads = g[0, :L].gather(-1, labels[0].unsqueeze(-1))
    rejected_grads = g[1, :L].gather(-1, labels[1].unsqueeze(-1))
    assert chosen_grads.sum() < 0  # increase chosen likelihood
    assert rejected_grads.sum() > 0  # decrease rejected likelihood


def test_only_marked_positions_are_scored():
    """pair_ids, not the task tag, decides what the margin sees. Everything the
    two sides share carries no id, and must not reach the comparison - it is
    identical text, so scoring it would push a response down for the prompt it
    was answering."""
    policy = PreferencePolicy(_config()).train()
    logits, labels, task, mask, pair_ids = _batch()
    full, _ = policy(
        logits, labels, assistant_mask=mask, task_type_ids=task, pair_ids=pair_ids
    )

    # Same batch, but only the trailing half of each row is in the comparison.
    half = pair_ids.clone()
    half[:, : half.size(1) // 2] = 0
    trimmed, metrics = policy(
        logits, labels, assistant_mask=mask, task_type_ids=task, pair_ids=half
    )
    assert trimmed is not None
    assert not torch.isclose(full, trimmed)
    assert metrics["preference_chosen_tokens"] < labels.size(1)


def test_pairs_are_scored_independently():
    """Two comparisons in one batch are averaged as comparisons, not pooled
    into one contrast between token populations - so a long response cannot
    outvote a short one."""
    policy = PreferencePolicy(_config()).train()
    torch.manual_seed(1)
    seq = 48
    logits = torch.randn(4, seq, 64, requires_grad=True)
    labels = torch.randint(0, 64, (4, seq - 1))
    task = torch.tensor([CHOSEN, REJECTED, CHOSEN, REJECTED]).view(4, 1).repeat(1, seq)
    mask = torch.ones(4, seq, dtype=torch.uint8)
    pair_ids = torch.tensor([1, 1, 2, 2]).view(4, 1).repeat(1, seq)

    loss, metrics = policy(
        logits, labels, assistant_mask=mask, task_type_ids=task, pair_ids=pair_ids
    )
    assert metrics["preference_pairs"] == 2.0
    assert 0.0 <= metrics["preference_accuracy"] <= 1.0

    # Scoring each pair alone and averaging gives the same margin.
    margins = []
    for pid in (1, 2):
        one = torch.where(pair_ids == pid, pair_ids, torch.zeros_like(pair_ids))
        _, m = policy(
            logits, labels, assistant_mask=mask, task_type_ids=task, pair_ids=one
        )
        margins.append(m["preference_margin"])
    assert metrics["preference_margin"] == pytest.approx(sum(margins) / 2, abs=1e-5)


def test_noop_without_a_scorable_pair():
    policy = PreferencePolicy(_config()).train()
    logits, labels, task, mask, pair_ids = _batch()

    only_chosen = torch.full_like(task, CHOSEN)
    loss, _ = policy(
        logits,
        labels,
        assistant_mask=mask,
        task_type_ids=only_chosen,
        pair_ids=pair_ids,
    )
    assert loss is None

    # Both sides present, but in DIFFERENT comparisons: the old pooled contrast
    # scored this, and what it measured was the gap between two unrelated
    # documents.
    split = _batch(pairs=(1, 2))
    loss, _ = policy(
        split[0],
        split[1],
        assistant_mask=split[3],
        task_type_ids=split[2],
        pair_ids=split[4],
    )
    assert loss is None

    # Below the per-side floor -> no-op, not noise.
    tiny = _batch(seq=4)
    loss, _ = policy(
        tiny[0],
        tiny[1],
        assistant_mask=tiny[3],
        task_type_ids=tiny[2],
        pair_ids=tiny[4],
    )
    assert loss is None

    # Unpaired data reaches this policy on every run that mixes datasets; it is
    # a no-op there rather than a pooled guess.
    loss, _ = policy(
        logits,
        labels,
        assistant_mask=mask,
        task_type_ids=task,
        pair_ids=torch.zeros_like(pair_ids),
    )
    assert loss is None

    loss, _ = policy(
        logits, labels, assistant_mask=mask, task_type_ids=None, pair_ids=pair_ids
    )
    assert loss is None

    loss, _ = policy(
        logits, labels, assistant_mask=mask, task_type_ids=task, pair_ids=None
    )
    assert loss is None

    policy.eval()
    loss, _ = policy(
        logits, labels, assistant_mask=mask, task_type_ids=task, pair_ids=pair_ids
    )
    assert loss is None
