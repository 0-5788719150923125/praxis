"""Unit tests for ``praxis.losses.compute_layer_wise_loss``.

The helper is framework-agnostic so the tests are framework-agnostic:
no Ray, no actors, no multiprocessing. We construct a real
``ForwardHead`` + the real ``CrossEntropyLoss`` criterion and feed the
helper synthetic hidden states, then verify:

1. It produces a scalar loss that actually backprops (the trainable
   tensors must receive gradient signal).
2. The cut-CE fast path is selected for a ``CutCrossEntropyLoss``-shaped
   criterion (detected by class name so we don't need the real
   integration package to be installed). We verify that the classifier
   module receives ``embeddings`` instead of pre-shifted logits.
3. Aux losses from a stub router (a scalar tensor) observably fold
   into the reported local loss via ``strategy``.
4. A ``LossContainer`` with multiple entries is unpacked correctly.
"""

from __future__ import annotations

from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis import PraxisConfig
from praxis.containers.loss import LossContainer
from praxis.heads.forward import ForwardHead
from praxis.losses import LOSS_REGISTRY, compute_layer_wise_loss
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.strategies.naive import NaiveSummation


def _toy_config(vocab_size: int = 32, hidden_size: int = 16) -> PraxisConfig:
    return PraxisConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        embed_size=hidden_size,
        num_heads=2,
        depth=1,
        num_layers=1,
        max_length=32,
        decoder_type="sequential",
        attention_type="standard",
        encoder_type=None,
        tie_weights=False,
    )


def test_compute_layer_wise_loss_plain_cross_entropy_backprops():
    """Happy path: plain cross entropy folds no aux losses and backprops."""
    torch.manual_seed(0)
    config = _toy_config()

    head = ForwardHead(config)
    criterion = LOSS_REGISTRY["cross_entropy"](vocab_size=config.vocab_size)

    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids[..., 1:].contiguous()

    loss = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        strategy=None,
        aux_losses=None,
        input_ids=input_ids,
    )

    assert loss.dim() == 0, "local loss must be a scalar"
    assert torch.isfinite(loss), f"loss was {loss}"

    # Gradients must reach the hidden states (the MF boundary) and the
    # head weights (each actor owns its own head copy and must train it).
    loss.backward()
    assert hidden_states.grad is not None
    assert head.lm_head.weight.grad is not None
    # And they should be non-zero - a sanity check that the criterion
    # actually saw the hidden states.
    assert hidden_states.grad.abs().sum().item() > 0
    assert head.lm_head.weight.grad.abs().sum().item() > 0


class _StubCutCrossEntropyLoss(nn.Module):
    """Shape-compatible stand-in for the real ``CutCrossEntropyLoss``.

    We can't import the real one without the optional ``cut_cross_entropy``
    dependency. The layer-wise helper selects the cut-CE branch via
    class-name match, so any subclass with the matching name triggers the
    same code path. This stub records what it was called with so the test
    can assert the helper handed it unshifted inputs (the whole point of
    the cut-CE branch).
    """

    # Name must match exactly: this is how the helper picks the branch.
    def __init__(self) -> None:
        super().__init__()
        self.calls: List[dict] = []

    def forward(self, logits, embeddings, classifier, labels, input_ids, **kwargs):
        self.calls.append(
            dict(
                logits_shape=tuple(logits.shape),
                embeddings_shape=tuple(embeddings.shape),
                labels_shape=tuple(labels.shape),
                input_ids_shape=tuple(input_ids.shape),
                classifier=classifier,
            )
        )
        # Return a scalar that depends on embeddings and classifier.weight
        # so gradients flow back through both.
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        logits_out = flat @ classifier.weight.t()
        return logits_out.pow(2).mean()


# Alias class with the expected name so the helper's class-name check
# fires. (``_is_cut_cross_entropy`` does ``criterion.__class__.__name__``.)
_StubCutCrossEntropyLoss.__name__ = "CutCrossEntropyLoss"


def test_compute_layer_wise_loss_cut_ce_fast_path_fires():
    """Cut-CE branch: the criterion sees full unshifted embeddings."""
    torch.manual_seed(1)
    config = _toy_config()

    head = ForwardHead(config)
    criterion = _StubCutCrossEntropyLoss()

    batch, seq_len = 2, 6
    hidden_states = torch.randn(batch, seq_len, config.hidden_size, requires_grad=True)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids[..., 1:].contiguous()

    loss = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        input_ids=input_ids,
    )

    assert len(criterion.calls) == 1
    call = criterion.calls[0]
    # The key assertion: the criterion was handed the *full unshifted*
    # hidden-state tensor (shape [batch, seq_len, hidden]), NOT
    # [batch, seq_len-1, hidden] that the plain-CE branch would produce.
    assert call["embeddings_shape"] == (batch, seq_len, config.hidden_size), (
        f"cut-CE branch should receive full unshifted embeddings, "
        f"got {call['embeddings_shape']}"
    )
    # input_ids arrives unshifted too.
    assert call["input_ids_shape"] == (batch, seq_len)
    # The classifier handed in must be the head's ``classifier`` property,
    # because cut-CE uses it as its linear layer.
    assert call["classifier"] is head.classifier

    assert torch.isfinite(loss)
    loss.backward()
    assert hidden_states.grad is not None
    assert head.lm_head.weight.grad is not None


def test_aux_losses_fold_via_strategy():
    """Aux losses from a router must change the reported loss value."""
    torch.manual_seed(2)
    config = _toy_config()

    head = ForwardHead(config)
    criterion = CrossEntropyLoss()
    strategy = NaiveSummation()

    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids[..., 1:].contiguous()

    # Baseline: no aux losses at all.
    base_loss = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        strategy=strategy,
        aux_losses=None,
        input_ids=input_ids,
    )

    aux_value = 0.75
    aux = torch.tensor(aux_value, dtype=torch.float32)
    folded = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        strategy=strategy,
        aux_losses=[aux],
        input_ids=input_ids,
    )

    # NaiveSummation = sum. So the folded loss should be baseline + aux.
    assert folded.item() == pytest.approx(base_loss.item() + aux_value, rel=1e-5)


def test_loss_container_entries_are_unpacked():
    """A LossContainer with multiple losses should fold each entry in."""
    torch.manual_seed(3)
    config = _toy_config()

    head = ForwardHead(config)
    criterion = CrossEntropyLoss()
    strategy = NaiveSummation()

    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids[..., 1:].contiguous()

    base_loss = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        strategy=strategy,
        aux_losses=None,
        input_ids=input_ids,
    )

    # A freshly-constructed LossContainer carries a zero "main" entry
    # already; add two more tagged losses with known scalar values.
    container = LossContainer()
    container.add_loss("router_aux", torch.tensor(0.25))
    container.add_loss("controller_aux", torch.tensor(0.5))

    folded = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        strategy=strategy,
        aux_losses=[container],
        input_ids=input_ids,
    )

    expected = base_loss.item() + 0.0 + 0.25 + 0.5
    assert folded.item() == pytest.approx(expected, rel=1e-5)


def test_zero_aux_tensor_is_treated_as_no_aux():
    """The LocalLayer happy path returns ``0.0`` for aux - no fold should happen."""
    torch.manual_seed(4)
    config = _toy_config()

    head = ForwardHead(config)
    criterion = CrossEntropyLoss()
    strategy = NaiveSummation()

    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids[..., 1:].contiguous()

    baseline = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        strategy=strategy,
        aux_losses=None,
        input_ids=input_ids,
    )

    zero_aux = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        strategy=strategy,
        aux_losses=[torch.tensor(0.0), 0.0, None],
        input_ids=input_ids,
    )
    assert zero_aux.item() == pytest.approx(baseline.item(), rel=1e-6)


def test_shift_convention_matches_manual_cross_entropy():
    """The helper's shift convention must match a hand-written MF step.

    This is the identical math the Phase 1 harness in
    ``tests/test_mono_forward_math.py`` runs, so by construction a
    helper-produced loss must be equal to a manual-shift-and-apply-CE
    loss on the same inputs.
    """
    torch.manual_seed(5)
    config = _toy_config()

    head = ForwardHead(config)
    criterion = CrossEntropyLoss()

    batch, seq_len = 3, 10
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = input_ids[..., 1:].contiguous()

    helper_loss = compute_layer_wise_loss(
        hidden_states=hidden_states,
        labels=labels,
        head=head,
        criterion=criterion,
        input_ids=input_ids,
    )

    manual_logits = head(hidden_states)
    manual_shift_logits = manual_logits[..., :-1, :].contiguous()
    manual_loss = F.cross_entropy(
        manual_shift_logits.reshape(-1, manual_shift_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    assert helper_loss.item() == pytest.approx(manual_loss.item(), rel=1e-5)
