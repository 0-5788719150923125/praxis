import torch
import torch.nn as nn
from typing import Dict, Type, Optional


class NoSort(nn.Module):
    """
    Base class for sorting mechanisms. This implementation does not perform any sorting.
    """
    
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