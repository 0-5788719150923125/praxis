"""Standard linear classifier: one projection from features to logits."""

from typing import Any, Optional

import torch.nn as nn
from torch import Tensor

from praxis.classifiers.base import BaseClassifier


class LinearClassifier(BaseClassifier):
    """Standard next-token prediction classifier.

    Its scorer is a single linear projection sized to :meth:`output_dims` -
    ``(hidden_size, vocab_size)`` standalone, or the encoder's declared
    byte-output layout in encoder mode. Builds nothing only when the
    encoder owns its full output pipeline (``handles_loss``, e.g. CALM).
    """

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)
        dims = self.output_dims()
        if dims is None:
            self.scorer = None
        else:
            feature_dim, vocab_size = dims
            self.scorer = nn.Linear(feature_dim, vocab_size, bias=False)
            self.scorer.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.scorer(hidden_states)
