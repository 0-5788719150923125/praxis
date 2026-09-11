"""Tests for the forward-path preference policy (praxis/policies/preference.py)
and the paired hh-rlhf data path."""

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


def _batch(vocab=64, seq=48):
    """Two rows: row 0 chosen-tagged, row 1 rejected-tagged, all assistant.
    Labels follow the production convention: PRE-SHIFTED (input_ids[..., 1:]),
    while task/assistant masks stay full-length.

    ``seq`` must leave each side above ``PreferencePolicy.MIN_SIDE_TOKENS`` or
    the policy correctly no-ops: a mean per-token log-prob over a handful of
    bytes is not a statistic, and the live run reached populations of one."""
    torch.manual_seed(0)
    logits = torch.randn(2, seq, vocab, requires_grad=True)
    labels = torch.randint(0, vocab, (2, seq - 1))
    task = torch.full((2, seq), CHOSEN, dtype=torch.long)
    task[1] = REJECTED
    mask = torch.ones(2, seq, dtype=torch.uint8)
    return logits, labels, task, mask


def test_margin_loss_and_gradient_direction():
    """The loss is -logsigmoid(beta * margin), and its gradient pushes chosen
    token logprobs UP and rejected token logprobs DOWN."""
    policy = PreferencePolicy(_config()).train()
    logits, labels, task, mask = _batch()
    loss, metrics = policy(logits, labels, assistant_mask=mask, task_type_ids=task)
    assert loss is not None and torch.isfinite(loss)
    # The term is scaled by the preference share of the batch so that rl_weight
    # is the honest per-token ratio against the main CE, which normalises over
    # every supervised position rather than over this policy's own tokens.
    expected = (
        policy.rl_weight
        * metrics["preference_share"]
        * -torch.nn.functional.logsigmoid(
            torch.tensor(policy.BETA * metrics["preference_margin"])
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


def test_noop_without_both_sides_or_in_eval():
    policy = PreferencePolicy(_config()).train()
    logits, labels, task, mask = _batch()

    only_chosen = torch.full_like(task, CHOSEN)
    loss, _ = policy(logits, labels, assistant_mask=mask, task_type_ids=only_chosen)
    assert loss is None

    # Both sides present but too small to mean anything -> no-op, not noise.
    tiny_logits, tiny_labels, tiny_task, tiny_mask = _batch(seq=8)
    loss, _ = policy(
        tiny_logits, tiny_labels, assistant_mask=tiny_mask, task_type_ids=tiny_task
    )
    assert loss is None

    loss, _ = policy(logits, labels, assistant_mask=mask, task_type_ids=None)
    assert loss is None

    policy.eval()
    loss, _ = policy(logits, labels, assistant_mask=mask, task_type_ids=task)
    assert loss is None
