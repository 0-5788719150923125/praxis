"""compute_layer_wise_loss: one layer's local loss for a layer-wise trainer.

Built on a real ForwardHead and CrossEntropyLoss with synthetic hidden states:
the shift convention matches a hand-written step, the cut-CE branch receives
unshifted inputs, and aux losses fold in through the strategy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis import PraxisConfig
from praxis.containers.loss import LossContainer
from praxis.heads.forward import ForwardHead
from praxis.losses import compute_layer_wise_loss
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.strategies.naive import NaiveSummation

VOCAB, HIDDEN, BATCH, SEQ = 32, 16, 2, 8


@pytest.fixture
def layer():
    """A ForwardHead plus a batch of hidden states and next-token labels."""
    torch.manual_seed(0)
    config = PraxisConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        embed_size=HIDDEN,
        num_heads=2,
        depth=1,
        num_layers=1,
        max_length=32,
        decoder_type="sequential",
        attention_type="modular",
        encoder_type=None,
        tie_weights=False,
    )
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    return SimpleNamespace(
        head=ForwardHead(config),
        hidden_states=torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True),
        input_ids=input_ids,
        labels=input_ids[..., 1:].contiguous(),
    )


def _loss(layer, criterion, **kwargs):
    return compute_layer_wise_loss(
        hidden_states=layer.hidden_states,
        labels=layer.labels,
        head=layer.head,
        criterion=criterion,
        input_ids=layer.input_ids,
        **kwargs,
    )


def test_shift_convention_matches_manual_cross_entropy(layer):
    """The helper's loss equals shift-then-CE by hand, and it backprops into
    both the hidden states (the layer boundary) and the head each layer owns."""
    loss = _loss(layer, CrossEntropyLoss())

    logits = layer.head(layer.hidden_states)[..., :-1, :]
    manual = F.cross_entropy(logits.reshape(-1, VOCAB), layer.labels.reshape(-1))
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)

    loss.backward()
    assert layer.hidden_states.grad.abs().sum().item() > 0
    assert layer.head.lm_head.weight.grad.abs().sum().item() > 0


class _StubCutCrossEntropyLoss(nn.Module):
    """Stand-in for the optional ``CutCrossEntropyLoss``, which the helper
    detects by class name. Records the shapes it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[dict] = []

    def forward(self, logits, embeddings, classifier, labels, input_ids, **kwargs):
        self.calls.append(
            dict(
                embeddings_shape=tuple(embeddings.shape),
                input_ids_shape=tuple(input_ids.shape),
                classifier=classifier,
            )
        )
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        return (flat @ classifier.weight.t()).pow(2).mean()


# The helper's ``_is_cut_cross_entropy`` matches ``criterion.__class__.__name__``.
_StubCutCrossEntropyLoss.__name__ = "CutCrossEntropyLoss"


def test_compute_layer_wise_loss_cut_ce_fast_path_fires(layer):
    """Cut-CE shifts internally, so it gets the full unshifted hidden states
    and input_ids, and the head's ``classifier`` as its linear layer."""
    criterion = _StubCutCrossEntropyLoss()

    loss = _loss(layer, criterion)

    (call,) = criterion.calls
    assert call["embeddings_shape"] == (BATCH, SEQ, HIDDEN)
    assert call["input_ids_shape"] == (BATCH, SEQ)
    assert call["classifier"] is layer.head.classifier
    loss.backward()
    assert layer.hidden_states.grad is not None
    assert layer.head.lm_head.weight.grad is not None


class _RecordingSum(NaiveSummation):
    def __init__(self):
        super().__init__()
        self.sizes: List[int] = []

    def forward(self, losses, names=None, trunk=None):
        self.sizes.append(len(losses))
        return super().forward(losses)


@pytest.mark.parametrize(
    "aux_losses, use_strategy, added, folded_terms",
    [
        ([torch.tensor(0.75)], True, 0.75, 2),
        # Every container entry folds, its zero "main" included.
        ([LossContainer(router_aux=0.25, controller_aux=0.5)], True, 0.75, 4),
        ([0.3], True, 0.3, 2),
        # LocalLayer's default aux is 0.0: no fold at all.
        ([torch.tensor(0.0), 0.0, None], True, 0.0, None),
        # No strategy supplied: the built-in sum.
        ([torch.tensor(0.75)], False, 0.75, None),
    ],
    ids=["tensor", "container", "float", "zeros", "no_strategy"],
)
def test_aux_losses_fold(layer, aux_losses, use_strategy, added, folded_terms):
    criterion = CrossEntropyLoss()
    strategy = _RecordingSum() if use_strategy else None
    base = _loss(layer, criterion).item()

    folded = _loss(layer, criterion, strategy=strategy, aux_losses=aux_losses)

    assert folded.item() == pytest.approx(base + added, rel=1e-5)
    if strategy is not None:
        assert strategy.sizes == ([folded_terms] if folded_terms else [])
