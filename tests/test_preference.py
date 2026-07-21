"""Tests for the forward-path preference policy (praxis/policies/preference.py)
and the paired hh-rlhf data path."""

import random

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


def _batch(vocab=64, seq=12):
    """Two rows: row 0 chosen-tagged, row 1 rejected-tagged, all assistant.
    Labels follow the production convention: PRE-SHIFTED (input_ids[..., 1:]),
    while task/assistant masks stay full-length."""
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
    expected = policy.rl_weight * -torch.nn.functional.logsigmoid(
        torch.tensor(policy.BETA * metrics["preference_margin"])
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

    loss, _ = policy(logits, labels, assistant_mask=mask, task_type_ids=None)
    assert loss is None

    policy.eval()
    loss, _ = policy(logits, labels, assistant_mask=mask, task_type_ids=task)
    assert loss is None


def test_rejected_tokens_excluded_from_main_ce():
    """_build_loss_weights zeroes PREF_REJECTED positions regardless of the
    weighter profile - the card's no-SFT contract for the rejected side."""
    from praxis.modeling import PraxisForCausalLM

    model = PraxisForCausalLM(_config())
    labels = torch.randint(0, 64, (1, 8))
    task = torch.full((1, 8), CHOSEN, dtype=torch.long)
    task[0, 4:] = REJECTED
    weights = model._build_loss_weights(
        labels=labels, task_type_ids=task, assistant_mask=None
    )
    assert (weights[0, 4:] == 0).all()
    assert (weights[0, :4] > 0).all()


def test_formatter_emits_both_sides_with_tags():
    from praxis.data.formatters import format_preference_pair

    doc = {
        "chosen": "\n\nHuman: Hi there\n\nAssistant: Good answer",
        "rejected": "\n\nHuman: Hi there\n\nAssistant: Bad answer",
    }
    random.seed(0)
    seen = set()
    for _ in range(40):
        out = format_preference_pair(doc, ["chosen", "rejected"], tokenizer=None)
        assert out["messages"], "pair side must parse to messages"
        tag = out["metadata"]["task_type"]
        seen.add(tag)
        text = out["messages"][-1]["content"]
        if tag == CHOSEN:
            assert "Good" in text
        else:
            assert tag == REJECTED and "Bad" in text
    assert seen == {CHOSEN, REJECTED}  # both sides drawn over repeated calls


def test_build_rl_policies_recall_family():
    from praxis.modeling import build_rl_policies

    cfg = _config(rl_type=["engagement", "joke", "preference"])
    policy, policy_type, recall = build_rl_policies(cfg)
    assert policy is None and policy_type is None
    assert set(recall) == {"engagement", "joke", "preference"}
    assert isinstance(recall["preference"], PreferencePolicy)


def test_byte_latent_forward_with_preference():
    """The full -d-shaped stack trains a step with the preference loss landing
    in the container and finite gradients."""
    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_harmonic_serpent",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        byte_level=True,
        head_type="prismatic4",
        residual_type="smear",
        rl_type=["preference"],
    )
    from praxis.modeling import PraxisForCausalLM

    model = PraxisForCausalLM(cfg).train()
    ids = torch.randint(4, 260, (2, 24))
    task = torch.full((2, 24), CHOSEN, dtype=torch.long)
    task[1] = REJECTED
    mask = torch.ones(2, 24, dtype=torch.uint8)
    out = model(
        input_ids=ids,
        labels=ids[..., 1:].contiguous(),  # production convention: pre-shifted
        task_type_ids=task,
        assistant_mask=mask,
    )
    assert torch.isfinite(out.loss)
    metrics = model.policies["preference"].get_metrics()
    assert "preference_margin" in metrics
    out.loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
