"""SequentialClassifier: compose standardized Praxis classifiers like ``nn.Sequential``.

Each non-terminal stage contributes its ``transform`` (a feature -> feature map,
identity by default); the terminal stage classifies. So a sequence of
``[HarmonicClassifier(transform-only), CrystalClassifier]`` runs the harmonic field's
multiplicative modulation, then the crystal distance classifier - one coherent
logit stream, with crystal's sign structure (which inference processors like
``repetition_penalty`` depend on) preserved.

The stages are passed as *builders* - callables ``(config, encoder) -> classifier``
(a classifier class, or a ``functools.partial`` over one) - so a registry entry
can compose a stack dynamically without a bespoke subclass, e.g.::

    prismatic = partial(SequentialClassifier, stages=[
        partial(HarmonicClassifier, amp_modulation="learned", build_scorer=False),
        CrystalClassifier,
    ])

Auxiliary losses, training metrics, and dashboard snapshots merge across the
stages, and per-stage charts surface automatically because the stages are
submodules (see ``BaseClassifier.all_metric_descriptions``).
"""

from typing import Any, Callable, List, Optional, Union

import torch.nn as nn
from torch import Tensor

from praxis.classifiers.base import BaseClassifier

ClassifierSpec = Union[BaseClassifier, Callable[..., BaseClassifier]]


class SequentialClassifier(BaseClassifier):
    """Chain of classifiers: transforms compose, the last one classifies.

    ``stages`` is a list of builders (classifier class / ``partial`` over one)
    that are instantiated with ``(config, encoder)``; already-built classifiers
    are accepted too (for direct use).
    """

    # A composed classifier manages its own output via its terminal stage
    # (crystal self-ties), so the model keeps it under tie_word_embeddings
    # rather than swapping in the generic TiedClassifier.
    self_ties = True

    def __init__(
        self,
        config: Any,
        encoder: Optional[nn.Module] = None,
        *,
        stages: List[ClassifierSpec],
    ) -> None:
        super().__init__(config, encoder)
        if not stages:
            raise ValueError("SequentialClassifier needs at least one stage.")
        built = [
            s if isinstance(s, BaseClassifier) else s(config, encoder=encoder)
            for s in stages
        ]
        self.stages = nn.ModuleList(built)
        # Mirror the terminal stage's resolved layout (it sized itself to the
        # encoder, e.g. CALM's 264 byte vocab) so the wrapper reports the same.
        terminal = self.stages[-1]
        self.hidden_size = terminal.hidden_size
        self.vocab_size = terminal.vocab_size
        # Point each transform stage's grad-ratio at the terminal scorer it
        # actually feeds (stages that don't track it just ignore the call).
        for stage in self.stages[:-1]:
            if hasattr(stage, "set_downstream"):
                stage.set_downstream(terminal.scorer)

    def compose_repr(self) -> str:
        return "Sequential(" + ", ".join(s.compose_repr() for s in self.stages) + ")"

    def __repr__(self) -> str:
        return self.compose_repr()

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        h = hidden_states
        for stage in self.stages[:-1]:
            h = stage.transform(h)
        return self.stages[-1](h, **kwargs)

    def transform(self, hidden_states: Tensor) -> Tensor:
        # Composable itself: every stage's transform, terminal included.
        h = hidden_states
        for stage in self.stages:
            h = stage.transform(h)
        return h

    @property
    def scorer(self) -> Optional[nn.Module]:
        return self.stages[-1].scorer

    def aux_losses(self) -> dict:
        out: dict = {}
        for stage in self.stages:
            out.update(stage.aux_losses())
        return out

    def training_metrics(self) -> dict:
        out: dict = {}
        for stage in self.stages:
            out.update(stage.training_metrics())
        return out

    def dashboard_snapshots(self) -> dict:
        out: dict = {}
        for stage in self.stages:
            out.update(stage.dashboard_snapshots())
        return out

    def all_metric_descriptions(self) -> dict:
        # Delegate to each stage rather than the default flat module walk, so
        # a nested ParallelClassifier's namespaced (``p{i}_``) per-branch
        # descriptions propagate instead of colliding on shared class-level keys.
        out: dict = {}
        for stage in self.stages:
            out.update(stage.all_metric_descriptions())
        return out
