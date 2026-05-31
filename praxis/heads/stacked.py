"""SequentialHead: compose standardized Praxis heads like ``nn.Sequential``.

Each non-terminal head contributes its ``transform`` (a feature -> feature map,
identity by default); the terminal head classifies. So a sequence of
``[HarmonicHead(transform-only), CrystalHead]`` runs the harmonic field's
multiplicative modulation, then the crystal distance classifier - one coherent
logit stream, with crystal's sign structure (which inference processors like
``repetition_penalty`` depend on) preserved.

The sub-heads are passed as *builders* - callables ``(config, encoder) -> head``
(a head class, or a ``functools.partial`` over one) - so a registry entry can
compose a stack dynamically without a bespoke subclass, e.g.::

    prismatic = partial(SequentialHead, heads=[
        partial(HarmonicHead, amp_modulation="learned", build_classifier=False),
        CrystalHead,
    ])

Auxiliary losses, training metrics, and dashboard snapshots merge across the
sub-heads, and per-head charts surface automatically because the sub-heads are
submodules (see ``BaseHead.all_metric_descriptions``).
"""

from typing import Any, Callable, List, Optional, Union

import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead

HeadSpec = Union[BaseHead, Callable[..., BaseHead]]


class SequentialHead(BaseHead):
    """Chain of heads: transforms compose, the last one classifies.

    ``heads`` is a list of builders (head class / ``partial`` over one) that are
    instantiated with ``(config, encoder)``; already-built heads are accepted
    too (for direct use).
    """

    # A composed head manages its own output via its terminal head (crystal
    # self-ties), so the model keeps it under tie_word_embeddings rather than
    # swapping in the generic TiedWeights head.
    self_ties = True

    def __init__(
        self,
        config: Any,
        encoder: Optional[nn.Module] = None,
        *,
        heads: List[HeadSpec],
    ) -> None:
        super().__init__(config, encoder)
        if not heads:
            raise ValueError("SequentialHead needs at least one head.")
        built = [
            h if isinstance(h, BaseHead) else h(config, encoder=encoder)
            for h in heads
        ]
        self.heads = nn.ModuleList(built)
        # Point each transform stage's grad-ratio at the terminal classifier it
        # actually feeds (heads that don't track it just ignore the call).
        terminal = self.heads[-1].classifier
        for head in self.heads[:-1]:
            if hasattr(head, "set_downstream"):
                head.set_downstream(terminal)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        h = hidden_states
        for head in self.heads[:-1]:
            h = head.transform(h)
        return self.heads[-1](h, **kwargs)

    def transform(self, hidden_states: Tensor) -> Tensor:
        # Composable itself: every stage's transform, terminal included.
        h = hidden_states
        for head in self.heads:
            h = head.transform(h)
        return h

    @property
    def classifier(self) -> Optional[nn.Module]:
        return self.heads[-1].classifier

    def aux_losses(self) -> dict:
        out: dict = {}
        for head in self.heads:
            out.update(head.aux_losses())
        return out

    def training_metrics(self) -> dict:
        out: dict = {}
        for head in self.heads:
            out.update(head.training_metrics())
        return out

    def dashboard_snapshots(self) -> dict:
        out: dict = {}
        for head in self.heads:
            out.update(head.dashboard_snapshots())
        return out
