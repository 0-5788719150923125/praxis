from typing import Dict, Optional, Type

import torch
import torch.nn as nn


class NoSort(nn.Module):
    """
    Base class for sorting mechanisms. This implementation does not perform any sorting.
    """

    # Whether this module reorders what it is given. The registry slot also
    # hosts differentiable positional-bias fields (decay_bias, amplitude_field)
    # that reuse the hook without permuting anything, so "is registered here"
    # does not imply "sorts" - ask this instead of matching on the key name.
    permutes: bool = False

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass without any sorting operation.

        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size)

        Returns:
            The unchanged input tensor
        """
        return hidden_states


# Registry for sorting mechanisms
SORTING_REGISTRY: Dict[str, Type[NoSort]] = {}


def register_sorting(name: str):
    """
    Decorator to register a sorting mechanism.

    Args:
        name: The name to register the sorting mechanism under
    """

    def register_sorting_cls(cls):
        SORTING_REGISTRY[name] = cls
        return cls

    return register_sorting_cls


# Register the base NoSort class
register_sorting("none")(NoSort)
