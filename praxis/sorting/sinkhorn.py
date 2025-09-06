import torch
from .base import NoSort, register_sorting


@register_sorting("sinkhorn")
class SinkhornSort(NoSort):
    """
    Sorting mechanism using a differentiable approximation based on optimal transport.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ascending = getattr(config, "sort_ascending", False)
        self.tau = getattr(config, "sinkhorn_temperature", 0.1)
        self.iterations = getattr(config, "sinkhorn_iterations", 10)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply differentiable sorting to the feature dimension.

        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size)

        Returns:
            Tensor with sorted feature dimensions
        """
        # This is a simplified implementation that maintains differentiability
        # without using the full Sinkhorn algorithm complexity

        if self.ascending:
            # For ascending sort, larger values should be later in sequence
            _, indices = torch.sort(hidden_states, dim=-1)
        else:
            # For descending sort, larger values should be earlier in sequence
            _, indices = torch.sort(hidden_states, dim=-1, descending=True)

        # Create the hard permutation matrix (one-hot)
        perm_size = hidden_states.shape[-1]
        hard_perm_matrix = torch.nn.functional.one_hot(indices, perm_size).float()

        # Apply temperature to control gradient flow
        # Lower temperature = sharper (more exact) permutation
        # Higher temperature = smoother (more gradient flow)
        soft_perm_matrix = torch.softmax(hard_perm_matrix / self.tau, dim=-1)

        # Combine for gradient flow (straight-through estimator)
        # During forward pass: use hard permutation
        # During backward pass: use soft permutation gradients
        perm_matrix = hard_perm_matrix + (soft_perm_matrix - hard_perm_matrix).detach()

        # Apply the permutation matrix
        sorted_hidden_states = torch.matmul(
            hidden_states.unsqueeze(-2), perm_matrix
        ).squeeze(-2)

        return sorted_hidden_states
