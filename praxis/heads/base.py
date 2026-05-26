"""Base class for language modeling heads."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseHead(nn.Module, ABC):
    """Abstract base class for language modeling heads.

    A head always owns its classifier and produces logits from features
    via ``forward(hidden_states)``; ``classifier`` exposes the projection
    module for cut-CE. The only difference between standalone and
    encoder-attached modes is the classifier's size: standalone uses
    ``(hidden_size, vocab_size)``; with an encoder the head sizes itself
    to the encoder's declared output layout (see :meth:`output_dims`),
    so byte-latent decode produces *features* and the head classifies
    them - same as the standalone path.
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

    def output_dims(self) -> Optional[Tuple[int, int]]:
        """Resolve ``(feature_dim, vocab_size)`` for this head's classifier.

        Standalone: ``(hidden_size, vocab_size)`` from config. Encoder
        mode: the encoder declares its output layout via ``output_dim`` /
        ``output_vocab_size``. If the encoder owns its full output pipeline
        (``handles_loss``, e.g. CALM), returns ``None`` - the head builds
        no classifier.
        """
        enc = self._encoder
        if enc is None:
            return (self.hidden_size, self.vocab_size)
        if getattr(enc, "handles_loss", False):
            return None
        return (int(enc.output_dim), int(enc.output_vocab_size))

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


    def aux_losses(self) -> Dict[str, Tensor]:
        """Named auxiliary losses to fold into the main objective.

        Each entry's key becomes the loss name in the model's loss
        container (and surfaces as a dashboard metric), so use stable,
        descriptive keys.

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

    def dashboard_snapshots(self) -> Dict[str, Any]:
        """Non-scalar live snapshots for dashboard visualizations
        (heatmaps, scatters, etc.).

        Served by ``/api/head_snapshots``, keyed by a stable snapshot
        name so the frontend can dispatch on the key. Scalars belong in
        ``training_metrics()``; this is the place for matrices, PCA
        projections, and similar non-scalar shapes.

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
