import torch
from torch import Tensor


def ghostmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Implementation of softmax1, which adds 1 to denominator
    to allow for "no-op" attention.
    https://www.evanmiller.org/attention-is-off-by-one.html

    Args:
        x: Input tensor
        dim: Dimension along which to apply ghostmax

    Returns:
        Tensor with ghostmax applied along the specified dimension
    """
    # Get max value for numerical stability (like in standard softmax)
    max_score, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_score

    # Calculate exponentials
    exp_x = torch.exp(x)

    # Sum exponentials and add 1
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True) + 1.0

    # Divide by sum plus 1
    return exp_x / sum_exp
