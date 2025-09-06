import torch
from .base import NoSort, register_sorting


@register_sorting("native")
class NativeSort(NoSort):
    """
    Sorting mechanism that uses torch.sort() to sort the feature dimension of the input.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ascending = getattr(config, "sort_ascending", False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sort the feature dimension of the input tensor.

        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size)

        Returns:
            Tensor with sorted feature dimensions
        """
        # Sort along the feature dimension (dim=-1)
        sorted_hidden_states, _ = torch.sort(
            hidden_states, dim=-1, descending=not self.ascending
        )
        return sorted_hidden_states
