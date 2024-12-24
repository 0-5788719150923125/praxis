import torch
import torch.nn as nn
from torch.autograd import Function


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class Entmax15Function(Function):
    @staticmethod
    def forward(ctx, input, dim):  # Made dim a required positional argument
        ctx.dim = dim

        # Normalize and scale
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = (
            input - max_val
        ) / 2  # For numerical stability and solving actual Entmax

        # Compute threshold and support
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)
        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)

        # Compute output
        output = torch.clamp(input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # derivative of gradient
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None


# Create a wrapper function that handles the dim argument properly
def alpha_entmax(input, dim=-1):
    return Entmax15Function.apply(input, dim)


class Entmax15(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return alpha_entmax(x, self.dim)


if __name__ == "__main__":
    # Test 1: Basic functionality
    x = torch.randn(3, 5, requires_grad=True)
    print("\nTest input:")
    print(x)

    # Forward pass
    out = alpha_entmax(x, dim=-1)
    print("\nOutput (should be sparse and sum to 1 along dim=-1):")
    print(out)
    print("\nOutput sums (should be close to 1):")
    print(out.sum(dim=-1))
    print("\nNumber of zeros (should be > 0):")
    print((out == 0).sum())

    # Test 2: Gradient computation
    loss = out.sum()
    loss.backward()
    print("\nGradients exist:", x.grad is not None)

    # Test 3: Shape preservation
    batch_size, seq_len, dim = 2, 3, 4
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    out = alpha_entmax(x, dim=-1)
    print("\nShape preservation:", x.shape == out.shape)

    # Test 4: As a layer
    layer = Entmax15(dim=-1)
    out = layer(x)
    print("\nLayer output shape:", out.shape)

    # Numerical checks
    assert torch.allclose(
        out.sum(dim=-1), torch.ones_like(out.sum(dim=-1))
    ), "Output should sum to 1"
    assert (out >= 0).all(), "Output should be non-negative"
    assert (out <= 1).all(), "Output should be <= 1"
    print("\nAll tests passed!")
