import torch

from .base import NoSort, register_sorting


@register_sorting("sinkhorn")
class SinkhornSort(NoSort):
    """
    Sorting mechanism using a differentiable approximation based on optimal transport.
    """

    permutes = True

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
            values, indices = torch.sort(hidden_states, dim=-1)
        else:
            # For descending sort, larger values should be earlier in sequence
            values, indices = torch.sort(hidden_states, dim=-1, descending=True)

        # Create the hard permutation matrix (one-hot): row j selects the
        # feature that lands at sorted position j
        perm_size = hidden_states.shape[-1]
        hard_perm_matrix = torch.nn.functional.one_hot(indices, perm_size).to(
            hidden_states.dtype
        )

        # Apply temperature to control gradient flow
        # Lower temperature = sharper (more exact) permutation
        # Higher temperature = smoother (more gradient flow)
        soft_perm_matrix = torch.softmax(hard_perm_matrix / self.tau, dim=-1)

        # Gather through the soft permutation: y[j] = sum_i P[j, i] * x[i]
        soft_sorted = torch.matmul(
            hidden_states.unsqueeze(-2), soft_perm_matrix.transpose(-1, -2)
        ).squeeze(-2)

        # Straight-through estimator: the forward pass is the exact sort, the
        # backward pass follows the soft permutation
        return values.detach() + (soft_sorted - soft_sorted.detach())
