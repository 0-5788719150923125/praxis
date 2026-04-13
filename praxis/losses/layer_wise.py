"""Framework-agnostic layer-wise loss helper.

This module provides :func:`compute_layer_wise_loss`, a generic helper that
any distributed Mono-Forward style trainer can call to compute one layer's
local loss. It deliberately has no knowledge of Ray, Hivemind,
``torch.distributed.rpc``, Flower, monarch, or any other framework - it
takes plain torch tensors and modules, runs the head projection and
criterion, and folds aux losses via a strategy.

The helper has two goals:

1. **Route through the model's real criterion.** Before this helper
   existed, ``LayerActor.train_batch`` hardcoded ``F.cross_entropy``. That
   bypassed ``--loss-func cut_cross_entropy`` (and its memory win), plus
   any other configurable criterion on Praxis's loss side. This helper
   calls ``criterion(logits=..., embeddings=..., classifier=..., labels=...)``
   with the same argument shape that :meth:`PraxisForCausalLM._compute_loss`
   uses, so cut-CE and plain CE both Just Work.

2. **Fold auxiliary losses per decision D5.** Routers and controllers on a
   ``LocalLayer`` can emit scalar aux losses (or a ``LossContainer``). When
   a layer-wise trainer is running one actor per layer, those aux losses
   must fold into the owning layer's *local* CE loss - otherwise they'd
   be discarded silently. The ``strategy`` callable (typically a
   ``praxis.strategies`` module, defaulting to ``NaiveSummation``) performs
   the fold.

Shift convention matches :meth:`PraxisForCausalLM._compute_loss` exactly:
non-cut-CE criteria see ``logits[..., :-1, :]`` shifted against the
already-shifted labels the caller supplies (``input_ids[..., 1:]``); cut-CE
criteria see the *full unshifted* hidden states and do the shift
internally via ``shift=1``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from praxis.containers.loss import LossContainer


def _is_cut_cross_entropy(criterion: nn.Module) -> bool:
    """Detect CutCrossEntropyLoss by class name.

    The cut-CE integration is an optional dependency (it pulls in the
    ``cut_cross_entropy`` package), so we can't do an ``isinstance`` check
    without importing and maybe-failing. Match on class name instead, same
    as :meth:`PraxisForCausalLM._compute_loss`.
    """
    return criterion.__class__.__name__ == "CutCrossEntropyLoss"


def _extract_aux_scalars(aux_losses: List[Any]) -> List[Tensor]:
    """Flatten a list of aux-loss entries to a list of scalar tensors.

    Each entry can be any of:
    - ``None`` / ``0`` / ``0.0`` (no aux loss at this layer) -> skipped.
    - a plain float / int -> wrapped in a zero-dim tensor.
    - a torch ``Tensor`` (any shape, typically scalar) -> used as-is.
    - a :class:`LossContainer` -> every contained loss value is added
      as a separate scalar (``get_loss_values()`` preserves the order
      the container was populated in).
    """
    scalars: List[Tensor] = []
    for entry in aux_losses:
        if entry is None:
            continue
        if isinstance(entry, LossContainer):
            for value in entry.get_loss_values():
                if isinstance(value, Tensor):
                    scalars.append(value)
                else:
                    scalars.append(torch.tensor(float(value), dtype=torch.float32))
            continue
        if isinstance(entry, Tensor):
            # Treat an exact numeric-zero tensor as "no aux loss" so the
            # happy path (LocalLayer's default 0.0 return) doesn't produce
            # a spurious strategy fold.
            if entry.dim() == 0 and float(entry.detach()) == 0.0:
                continue
            scalars.append(entry)
            continue
        # Python scalar fall-through.
        value = float(entry)
        if value == 0.0:
            continue
        scalars.append(torch.tensor(value, dtype=torch.float32))
    return scalars


def compute_layer_wise_loss(
    hidden_states: Tensor,
    labels: Tensor,
    head: nn.Module,
    criterion: nn.Module,
    strategy: Optional[Callable[[List[Tensor]], Tensor]] = None,
    aux_losses: Optional[List[Any]] = None,
    input_ids: Optional[Tensor] = None,
) -> Tensor:
    """Compute one layer's local loss for a layer-wise trainer.

    Args:
        hidden_states: Post-layer activations, shape
            ``[batch, seq_len, hidden_size]``. These are the "embeddings"
            that :meth:`PraxisForCausalLM._compute_loss` passes to the
            criterion.
        labels: Next-token labels, **already shifted**
            (``input_ids[..., 1:]``). Same convention as
            ``_compute_loss``.
        head: The per-layer replicated head module. Must expose both
            ``head(hidden_states)`` (to produce logits) and
            ``head.classifier`` (for the cut-CE fast path).
        criterion: A Praxis loss module with the signature
            ``forward(logits, embeddings, classifier, labels, input_ids, ...)``.
            ``CutCrossEntropyLoss`` is detected by class name and given
            the full unshifted ``hidden_states`` plus the full
            unshifted ``input_ids``; everything else receives
            pre-shifted ``logits`` and ``embeddings``.
        strategy: Optional callable (typically a
            ``praxis.strategies.*`` module) that folds a list of losses
            into a single scalar. Used when ``aux_losses`` has entries.
            Defaults to :class:`NaiveSummation` semantics
            (``sum(losses)``) when aux losses are present and no
            ``strategy`` was supplied.
        aux_losses: Optional list of per-layer auxiliary losses emitted
            by routers / controllers (plain floats, scalar tensors, or
            ``LossContainer`` instances). Folded into the per-layer CE
            via ``strategy`` when non-empty. Per decision D5 in
            ``PLAN.md`` / ``PHASE_5.md``.
        input_ids: Optional full unshifted input_ids. Only required when
            the criterion is ``CutCrossEntropyLoss`` - cut-CE uses them
            as the unshifted targets its ``shift=1`` internal path wants.
            For non-cut-CE criteria this is passed through for
            completeness (some criteria, e.g. ``dedup`` cross-entropy,
            read ``input_ids`` to penalise duplicated tokens).

    Returns:
        A scalar loss tensor suitable for ``loss.backward()`` on the
        caller's per-layer optimizer.
    """
    is_cut_ce = _is_cut_cross_entropy(criterion)

    if is_cut_ce:
        # Cut-CE wants the full unshifted embeddings + the classifier
        # weight; it handles shifting internally via ``shift=1``. We pass
        # a dummy logits tensor (it's ignored on the cut-CE code path).
        # Matching _compute_loss exactly, including "input_ids" keyword.
        classifier = head.classifier
        local_input_ids = input_ids if input_ids is not None else labels
        local_loss = criterion(
            logits=hidden_states,  # ignored by cut-CE
            embeddings=hidden_states,
            classifier=classifier,
            labels=labels,
            input_ids=local_input_ids,
        )
    else:
        logits = head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_embeddings = hidden_states[..., :-1, :].contiguous()
        classifier = head.classifier if hasattr(head, "classifier") else None
        # ``input_ids`` passthrough mirrors _compute_loss: if the caller
        # didn't supply it, fall back to the shifted labels which at
        # least lets the dedup criterion see *some* token stream.
        local_input_ids = input_ids if input_ids is not None else labels
        local_loss = criterion(
            logits=shift_logits,
            embeddings=shift_embeddings,
            classifier=classifier,
            labels=labels,
            input_ids=local_input_ids,
        )

    scalars = _extract_aux_scalars(list(aux_losses) if aux_losses else [])
    if not scalars:
        return local_loss

    # D5: fold aux losses into the owning layer's local CE via strategy.
    # Default to naive summation when caller didn't provide one - this
    # matches praxis.strategies.NaiveSummation semantics without the
    # import cost.
    all_losses: List[Tensor] = [local_loss, *scalars]
    if strategy is None:
        folded = all_losses[0]
        for extra in all_losses[1:]:
            folded = folded + extra
        return folded
    return strategy(all_losses)
