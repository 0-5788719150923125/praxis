from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SUGAR(nn.Module):
    """
    Surrogate Gradient Learning for ReLU (SUGAR) activation function.

    This activation preserves ReLU in the forward pass but uses a smooth
    surrogate gradient in the backward pass to avoid the dying ReLU problem.

    Based on "The Resurrection of the ReLU" paper.
    https://arxiv.org/abs/2505.22074
    """

    def __init__(self, surrogate_type: str = "bsilu", alpha: float = 1.67):
        """
        Initialize SUGAR activation.

        Args:
            surrogate_type: Type of surrogate gradient ("bsilu", "nelu", "gelu", "silu", "elu")
            alpha: Parameter for B-SiLU or NeLU (default: 1.67)
        """
        super().__init__()
        self.surrogate_type = surrogate_type.lower()
        self.alpha = alpha

        if self.surrogate_type not in [
            "bsilu",
            "nelu",
            "gelu",
            "silu",
            "elu",
            "leaky_relu",
        ]:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the multiplication trick for gradient injection.

        Args:
            x: Input tensor

        Returns:
            Output with ReLU forward pass but surrogate gradient backward pass
        """
        # Compute surrogate gradient
        surrogate_grad = self._compute_surrogate_gradient(x)

        # Multiplication trick (Eq. 11-12 in the paper)
        m = x * surrogate_grad.detach()
        y = m - m.detach() + F.relu(x).detach()

        return y

    def _compute_surrogate_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the surrogate function.

        Args:
            x: Input tensor

        Returns:
            Gradient of the surrogate function
        """
        if self.surrogate_type == "bsilu":
            return self._bsilu_gradient(x)
        elif self.surrogate_type == "nelu":
            return self._nelu_gradient(x)
        elif self.surrogate_type == "gelu":
            return self._gelu_gradient(x)
        elif self.surrogate_type == "silu":
            return self._silu_gradient(x)
        elif self.surrogate_type == "elu":
            return self._elu_gradient(x)
        elif self.surrogate_type == "leaky_relu":
            return self._leaky_relu_gradient(x)

    def _bsilu_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of Bounded SiLU (B-SiLU).

        d/dx B-SiLU(x) = sigmoid(x) + (x + alpha) * sigmoid(x) * (1 - sigmoid(x))
        """
        sigmoid_x = torch.sigmoid(x)
        return sigmoid_x + (x + self.alpha) * sigmoid_x * (1 - sigmoid_x)

    def _nelu_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of Negative slope Linear Unit (NeLU).

        d/dx NeLU(x) = 1 if x > 0, else alpha * 2x / (1 + x^2)^2
        """
        positive_mask = x > 0
        negative_grad = self.alpha * 2 * x / (1 + x**2) ** 2
        return torch.where(positive_mask, torch.ones_like(x), negative_grad)

    def _gelu_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of GELU approximation.

        Using tanh approximation: d/dx GELU(x) ≈ 0.5 + 0.5 * tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
                                                + x * sech^2(...) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        """
        # Constants for GELU approximation
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        a = 0.044715

        # Compute tanh argument
        tanh_arg = sqrt_2_over_pi * (x + a * x**3)
        tanh_val = torch.tanh(tanh_arg)

        # sech^2 = 1 - tanh^2
        sech2_val = 1 - tanh_val**2

        # Full gradient
        return (
            0.5 + 0.5 * tanh_val + x * sech2_val * sqrt_2_over_pi * (1 + 3 * a * x**2)
        )

    def _silu_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of SiLU (Swish).

        d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        """
        sigmoid_x = torch.sigmoid(x)
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)

    def _elu_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of ELU.

        d/dx ELU(x) = 1 if x > 0, else exp(x)
        """
        return torch.where(x > 0, torch.ones_like(x), torch.exp(x))

    def _leaky_relu_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of Leaky ReLU.

        d/dx LeakyReLU(x) = 1 if x > 0, else 0.01
        """
        return torch.where(x > 0, torch.ones_like(x), 0.01 * torch.ones_like(x))


class BSiLU(nn.Module):
    """
    Bounded Sigmoid Linear Unit (B-SiLU) activation function.

    B-SiLU(x) = (x + alpha) * sigmoid(x) - alpha/2

    This can be used as a standalone activation or as a surrogate for SUGAR.
    """

    def __init__(self, alpha: float = 1.67):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.alpha) * torch.sigmoid(x) - self.alpha / 2


class NeLU(nn.Module):
    """
    Negative slope Linear Unit (NeLU) activation function.

    This activation has ReLU-like behavior for x > 0 but smooth negative gradients.
    Note: This is defined by its gradient behavior in the paper, so we integrate
    to get the forward function.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For x > 0: integral of 1 = x
        # For x <= 0: integral of alpha * 2x / (1 + x^2)^2
        # The integral for x <= 0 is: -alpha / (1 + x^2)
        # We need to ensure continuity at x = 0
        positive_part = x
        negative_part = -self.alpha / (1 + x**2) + self.alpha  # +alpha for continuity

        return torch.where(x > 0, positive_part, negative_part)
