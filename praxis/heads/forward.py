"""Standard forward (causal) language modeling head."""

from typing import Any, Optional

import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead


class ForwardHead(BaseHead):
    """Standard next-token prediction head.

    In encoder-attached mode the encoder owns the classifier, so this
    head allocates nothing and stays out of the way - the default
    pass-through ``process_encoder_output`` lets the encoder's logits
    flow through unchanged.
    """

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)
        if self.has_encoder:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.lm_head.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.lm_head(hidden_states)

    @property
    def classifier(self) -> Optional[nn.Module]:
        return self.lm_head
