import random

import torch
import torch.nn as nn
from torch import Tensor

from praxis.processors import PROCESSOR_REGISTRY
from praxis.stacks import PraxisStack


class PraxisDecoder(nn.Module):
    """
    A module that wraps core decoder operations.
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.stack = PraxisStack(config)
        self.processor = PROCESSOR_REGISTRY.get(config.processor)(config)

    def forward(
        self,
        inputs: Tensor,
        current_state: Tensor = None,
        attention_mask: Tensor = None,
        past_key_values=None,
        block_ids=None,
    ):

        experts = list(self.stack.locals) + list(self.stack.remotes)
        original_order = experts.copy()
        if hasattr(self.stack.behavior, "shuffle_experts"):
            experts = self.stack.behavior.shuffle_experts(experts)

        return self.processor(
            experts,
            self.stack,
            inputs,
            attention_mask,
            past_key_values,
            block_ids,
            current_state,
            original_order,
            self.training,
        )

    def get_metrics(self):
        """Return current prediction accuracies"""
        return self.stack.get_metrics()
