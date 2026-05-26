"""Standard forward (causal) language modeling head."""

from typing import Any, Optional

import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead


class ForwardHead(BaseHead):
    """Standard next-token prediction head.

    Owns a single linear classifier sized to :meth:`output_dims` -
    ``(hidden_size, vocab_size)`` standalone, or the encoder's declared
    byte-output layout in encoder mode. Builds nothing only when the
    encoder owns its full output pipeline (``handles_loss``, e.g. CALM).
    """

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)
        dims = self.output_dims()
        if dims is None:
            self.lm_head = None
        else:
            feature_dim, vocab_size = dims
            self.lm_head = nn.Linear(feature_dim, vocab_size, bias=False)
            self.lm_head.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.lm_head(hidden_states)

    @property
    def classifier(self) -> Optional[nn.Module]:
        return self.lm_head
