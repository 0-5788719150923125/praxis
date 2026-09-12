"""`ForeignCausalLM`: a published model trained under Praxis objectives.

The point of every test here is that a Praxis mechanism actually REACHES a model
Praxis did not build. transformers accepts unknown forward kwargs and drops
them, so "the run completed" proves nothing on its own - each test checks that
the mechanism changed the loss, or that it failed loudly when it could not.
"""

import copy

import pytest
import torch

from praxis import PraxisConfig
from praxis.models import ForeignCausalLM, reject_unsupported
from praxis.tasks import TaskType

from .conftest import HIDDEN, VOCAB


def test_forward_returns_logits_and_loss(foreign, batch):
    ids, labels = batch
    out = foreign(input_ids=ids, labels=labels)
    assert out.logits.shape == (*ids.shape, VOCAB)
    assert out.loss.requires_grad


def test_labels_are_never_handed_to_the_hosted_model(foreign, batch, monkeypatch):
    """The hosted model shifts internally; Praxis pre-shifts. Passing labels
    down would shift twice, so the wrapper must keep them."""
    seen = {}
    inner = foreign.model.forward

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return inner(*args, **kwargs)

    monkeypatch.setattr(foreign.model, "forward", spy)
    ids, labels = batch
    foreign(input_ids=ids, labels=labels)
    assert "labels" not in seen or seen["labels"] is None
    assert seen["output_hidden_states"] is True


def test_praxis_side_channels_are_not_silently_dropped(foreign, batch):
    """The whole failure mode this wrapper exists to prevent: an assistant mask
    that reaches nothing and leaves the loss unchanged."""
    ids, labels = batch
    plain = foreign(input_ids=ids, labels=labels).loss
    mask = torch.zeros_like(ids)
    mask[:, -4:] = 1
    masked = foreign(input_ids=ids, labels=labels, assistant_mask=mask).loss
    assert not torch.allclose(plain, masked)


def test_loss_function_is_the_praxis_one(hosted, praxis_config, batch):
    """`--loss-func` selects the criterion for a foreign model too."""
    ids, labels = batch
    losses = {}
    for name in ("cross_entropy", "focal"):
        config = copy.deepcopy(praxis_config)
        config.loss_func = name
        model = ForeignCausalLM(copy.deepcopy(hosted), config)
        assert type(model.criterion.main).__name__.lower().startswith(
            name.split("_")[0]
        )
        losses[name] = float(model(input_ids=ids, labels=labels).loss)
    assert losses["cross_entropy"] != losses["focal"]


def test_preference_policy_runs_on_a_foreign_model(hosted, praxis_config):
    """The proof-of-concept objective: a forward-path RL policy that needs
    per-token task tags, on a model that knows nothing about them."""
    config = copy.deepcopy(praxis_config)
    config.rl_type = "preference"
    model = ForeignCausalLM(copy.deepcopy(hosted), config).train()

    rows, length = 4, 96
    ids = torch.randint(0, VOCAB, (rows, length))
    tags = torch.full((rows, length), int(TaskType.PREF_CHOSEN))
    tags[rows // 2 :] = int(TaskType.PREF_REJECTED)
    # The margin contrasts WITHIN a pair id, so rows 0/2 and 1/3 are the two
    # halves of two pairs. Without this channel the policy is a deliberate
    # no-op - which is the behaviour the next test pins.
    pairs = torch.tensor([1, 2, 1, 2]).unsqueeze(1).expand(rows, length)

    out = model(
        input_ids=ids,
        labels=ids[..., 1:].contiguous(),
        task_type_ids=tags,
        assistant_mask=torch.ones_like(ids),
        pair_ids=pairs,
    )
    out.loss.backward()
    metrics = model.policies["preference"].get_metrics()
    assert metrics["preference_pairs"] == 2
    assert metrics["preference_chosen_tokens"] > 0
    assert metrics["preference_rejected_tokens"] > 0
    assert "preference_margin" in metrics


def test_preference_is_a_no_op_without_the_pairing(hosted, praxis_config):
    """A batch carrying no pair ids cannot be contrasted, and the policy says so
    by scoring nothing rather than inventing a pairing."""
    config = copy.deepcopy(praxis_config)
    config.rl_type = "preference"
    model = ForeignCausalLM(copy.deepcopy(hosted), config).train()

    ids = torch.randint(0, VOCAB, (4, 96))
    model(
        input_ids=ids,
        labels=ids[..., 1:].contiguous(),
        task_type_ids=torch.full_like(ids, int(TaskType.PREF_CHOSEN)),
        assistant_mask=torch.ones_like(ids),
    )
    assert model.policies["preference"].get_metrics() == {}


def test_rejected_tokens_leave_the_main_objective(hosted, praxis_config):
    """The hh-rlhf card's contract: rejected text is contrast material only."""
    config = copy.deepcopy(praxis_config)
    config.rl_type = "preference"
    model = ForeignCausalLM(copy.deepcopy(hosted), config).train()

    ids = torch.randint(0, VOCAB, (2, 32))
    labels = ids[..., 1:].contiguous()
    chosen = torch.full_like(ids, int(TaskType.PREF_CHOSEN))
    rejected = torch.full_like(ids, int(TaskType.PREF_REJECTED))

    weights = model._build_loss_weights(labels, rejected, None)
    assert float(weights.sum()) == 0.0
    weights = model._build_loss_weights(labels, chosen, None)
    assert float(weights.sum()) > 0.0


def test_unsupported_mechanisms_are_a_hard_error(praxis_config):
    praxis_config.mtp_type = "vanilla"
    with pytest.raises(ValueError, match="mtp_type"):
        reject_unsupported(ForeignCausalLM, praxis_config)


def test_supported_run_passes_the_capability_check(praxis_config):
    reject_unsupported(ForeignCausalLM, praxis_config)  # does not raise


def test_praxis_sentinels_answer_like_a_praxis_model(foreign):
    """Every consumer probes these with getattr; a foreign model has to give
    the same answers a standalone Praxis model does."""
    assert foreign.encoder is False
    assert foreign.decoder is None
    assert foreign.classifier is None
    assert foreign.stage_warmup_anchor() == -1
    assert foreign.get_metrics() == {}
    assert foreign.scorer is foreign.model.get_output_embeddings()


def test_objective_config_is_separate_from_the_model_config(foreign, praxis_config):
    """The hub config must never carry a Praxis knob."""
    assert foreign.config is foreign.model.config
    assert foreign.objective_config is praxis_config
    assert not hasattr(foreign.config, "loss_func")


def test_generate_delegates_to_the_hosted_model(foreign):
    ids = torch.randint(0, VOCAB, (1, 4))
    out = foreign.generate(ids, max_new_tokens=3, do_sample=False)
    assert out.shape[1] == 7


def test_full_checkpoint_without_an_adapter(foreign):
    assert foreign.partial_checkpoint is False
    assert any("lora" not in key for key in foreign.state_dict())
