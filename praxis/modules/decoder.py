import random

import torch
import torch.nn as nn
from torch import Tensor

from praxis.processors import PROCESSOR_REGISTRY


class PraxisDecoder(nn.Module):
    """
    A module that wraps core decoder operations.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.processor = PROCESSOR_REGISTRY.get(config.processor)(config)

    def forward(
        self,
        inputs: Tensor,
        current_state: Tensor = None,
        attention_mask: Tensor = None,
        past_key_values=None,
        block_ids=None,
    ):
        return self.processor(
            inputs,
            attention_mask,
            past_key_values,
            block_ids,
            current_state,
        )

    def get_metrics(self):
        """Return current prediction accuracies"""
        return self.processor.stack.get_metrics()
