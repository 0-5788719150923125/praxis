from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor, addcdiv, sin, square
from torch.autograd import Function
from torch.distributions.exponential import Exponential
from torch.nn import Module
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


class SnakeFunction(Function):
    """
    A periodic activation function with learnable parameters.
    https://arxiv.org/abs/2006.08195
    """

    @staticmethod
    def forward(ctx, x: Tensor, a: Tensor) -> Tensor:
        ctx.save_for_backward(x, a)
        # We need to ensure 'a' is properly broadcast
        if a.dim() < x.dim():
            # Add missing dimensions to properly broadcast
            a = a.view(*([1] * (x.dim() - a.dim())), *a.shape)
        return torch.where(a == 0, x, addcdiv(x, square(sin(a * x)), a))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        x, a = ctx.saved_tensors
        sin2ax = sin(2 * a * x) if any(ctx.needs_input_grad) else None
        grad_x = grad_output * (1 + sin2ax) if ctx.needs_input_grad[0] else None
        grad_a = (
            grad_output
            * torch.where(a == 0, square(x), sin2ax * x / a - square(sin(a * x) / a))
            if ctx.needs_input_grad[1]
            else None
        )
        return grad_x, grad_a


class Snake(LazyModuleMixin, Module):
    def __init__(
        self,
        a: Optional[float] = None,
        trainable: bool = True,
        exp_rate: float = 1.0,
    ):
        super().__init__()
        self.trainable = trainable
        self.a_value = a
        self.exp_rate = exp_rate

        if trainable:
            self.a = UninitializedParameter()
        else:
            self.register_buffer("a", None)

    def initialize_parameters(self, x: Tensor, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameter 'a' based on input tensor.
        For sequence models, we only use the hidden dimension (last dimension)."""
        # Get feature dimension (always last dimension)
        feature_shape = x.shape[-1:]

        # Create initial tensor
        if self.a_value is not None:
            initial_a = torch.full(
                size=feature_shape,
                fill_value=self.a_value,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            dist = Exponential(torch.tensor(self.exp_rate, device=x.device))
            initial_a = dist.sample(feature_shape)

        if self.trainable:
            self.a.materialize(initial_a.shape, device=x.device, dtype=x.dtype)
            with torch.no_grad():
                self.a.copy_(initial_a)
        else:
            self.register_buffer("a", initial_a)

    def forward(self, x: Tensor) -> Tensor:
        return SnakeFunction.apply(x, self.a)
