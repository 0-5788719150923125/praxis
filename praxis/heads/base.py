"""Base class for language modeling heads."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseHead(nn.Module, ABC):
    """Abstract base class for language modeling heads.

    Heads have two execution modes:

    Standalone (no encoder): ``forward(hidden_states)`` produces logits
    and ``classifier`` exposes the projection module for cut-CE.

    Encoder-attached: the encoder owns the byte/patch decode and produces
    its own ``(logits, decoder_embeds)``. The head's
    :meth:`process_encoder_output` hook can inspect, modulate, or replace
    those outputs - that's how harmonic / crystal heads participate in
    byte-latent runs without ``modeling.py`` knowing their internals.
    """

    def __init__(self, config: ConfigType, encoder: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self._encoder = encoder

    @property
    def has_encoder(self) -> bool:
        return self._encoder is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_size={self.hidden_size}, vocab_size={self.vocab_size})"

    @abstractmethod
    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        """Standalone forward pass (no encoder).

        Args:
            hidden_states: Hidden states from the model [batch, seq_len, hidden_size]

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        pass

    @property
    @abstractmethod
    def classifier(self) -> Optional[nn.Module]:
        """The classifier module used downstream (e.g., by cut-CE)."""
        pass

    def process_encoder_output(
        self,
        decoder_embeds: Tensor,
        encoder_logits: Tensor,
        encoder_classifier: nn.Module,
    ) -> Tuple[Tensor, Tensor, nn.Module]:
        """Encoder-attached hook. Default: pass through unchanged.

        Override to participate in the encoder path (modulate embeds,
        replace the classifier, etc.).

        Returns:
            (logits, decoder_embeds, classifier) - whatever the model
            should bind as ``logits``, ``hidden_states``, and the
            ``classifier`` passed to the loss criterion.
        """
        return encoder_logits, decoder_embeds, encoder_classifier

    def aux_losses(
        self, embedding_weights: Optional[list] = None
    ) -> Dict[str, Tensor]:
        """Named auxiliary losses to fold into the main objective.

        Each entry's key becomes the loss name in the model's loss
        container (and surfaces as a dashboard metric), so use stable,
        descriptive keys.

        Args:
            embedding_weights: Input-embedding weight tensors the model
                exposes for heads that want to regularize them (e.g.,
                the crystal head's column-RMS penalty). May be ``None``
                or empty if the model has no exposed embeddings. Heads
                that don't use embedding weights can ignore this arg.

        Default: no aux losses.
        """
        return {}

    def training_metrics(self) -> Dict[str, float]:
        """Diagnostic scalars to surface each logging step.

        Called from the dynamics callback before the optimizer step, so
        parameter gradients are still available - heads that report
        grad-derived metrics (e.g., grad-ratio against a downstream
        classifier) can read ``param.grad`` directly. Default: no
        metrics.
        """
        return {}

    def dashboard_snapshots(
        self, embedding_weights: Optional[list] = None
    ) -> Dict[str, Any]:
        """Non-scalar live snapshots for dashboard visualizations
        (heatmaps, scatters, etc.).

        Served by ``/api/head_snapshots``, keyed by a stable snapshot
        name so the frontend can dispatch on the key. Scalars belong in
        ``training_metrics()``; this is the place for matrices, PCA
        projections, and similar non-scalar shapes.

        Args:
            embedding_weights: Input-embedding weight tensors the model
                exposes for heads that want to visualize them (e.g.,
                crystal's PCA density grid of regularized embeddings).
                ``None`` or empty if unavailable. Heads that don't
                consume embeddings can ignore this arg.

        Default: no snapshots.
        """
        return {}

    def all_metric_descriptions(self) -> Dict[str, Any]:
        """Collect ``metric_descriptions`` from the head and its submodules.

        Each module class may declare a ``metric_descriptions`` dict whose
        values are either plain strings or rich descriptors with chart
        rendering hints (see :mod:`praxis.metrics.descriptions`). The
        walk picks up the head's own dict plus any contribution from
        children like the harmonic field or the crystal classifier, so
        the frontend can render head-specific charts without
        ``modeling.py`` knowing about them.
        """
        out: Dict[str, Any] = {}
        for mod in self.modules():
            descs = getattr(type(mod), "metric_descriptions", None)
            if isinstance(descs, dict):
                out.update(descs)
        return out
