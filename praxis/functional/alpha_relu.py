from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function


class AlphaReLUFunction(Function):
    """
    An optimized version of α-ReLU, which introduces sparsity into the
    softmax operation.
    https://arxiv.org/abs/2111.06832
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor, dim: int, alpha: float, tau: float) -> Tensor:
        ctx.dim = dim
        ctx.alpha = alpha

        # Apply the formula: [(α-1)z_i - τ]_+^(1/(α-1))
        scaled = (alpha - 1) * input - tau
        relu_output = F.relu(scaled)
        output = torch.pow(relu_output, 1 / (alpha - 1))

        ctx.save_for_backward(output, relu_output)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[Optional[Tensor], None, None, None]:
        output, relu_output = ctx.saved_tensors

        # Compute gradient
        power = (2 - ctx.alpha) / (ctx.alpha - 1)
        grad_input = grad_output * torch.pow(output, power)

        # Return gradients for all inputs (None for dim, alpha, and tau as they're not parameters)
        return grad_input, None, None, None


def get_tau(X: Tensor, dim: int = -1, k: Optional[int] = None) -> Tensor:
    """Core computation for 1.5-entmax: optimal threshold (tau).
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star


def _make_ix_like(X: Tensor, dim: int) -> Tensor:
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X: Tensor, dim: int) -> Tensor:
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def alpha_relu(
    input: Tensor, dim: int = -1, alpha: float = 1.5, tau: Optional[float] = None
) -> Tensor:
    if tau is None:
        # Default tau based on paper's recommendations
        tau = 0.5  # This should be properly initialized in practice
    return AlphaReLUFunction.apply(input, dim, alpha, tau)


class AlphaReLU(nn.Module):
    def __init__(self, alpha: float = 1.5, tau: Optional[float] = None, dim: int = -1):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return alpha_relu(x, self.dim, self.alpha, self.tau)


# Modify your test section
if __name__ == "__main__":
    # Add tests for AlphaReLU alongside existing Entmax15 tests
    print("\nTesting AlphaReLU:")
    x = torch.randn(3, 5, requires_grad=True)

    # Test AlphaReLU
    alpha_relu_layer = AlphaReLU(dim=-1)
    out_relu = alpha_relu_layer(x)
    print("\nAlphaReLU output:")
    print(out_relu)

    # Test gradients
    loss_relu = out_relu.sum()
    loss_relu.backward()
    print("\nAlphaReLU gradients exist:", x.grad is not None)

    # Shape tests
    batch_size, seq_len, dim = 2, 3, 4
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    out_relu = alpha_relu_layer(x)
    print("\nAlphaReLU shape preservation:", x.shape == out_relu.shape)

    # Numerical checks
    assert (out_relu >= 0).all(), "Output should be non-negative"
    print("\nAlphaReLU tests passed!")
