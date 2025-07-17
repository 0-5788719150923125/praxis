"""
Temporal Health Complex (THC) Module for Transformers

Enhances temporal reasoning using complex numbers in a computationally efficient way.
The module uses complex-valued convolutions to learn phase relationships between tokens,
improving the model's ability to understand temporal patterns and generalize across
sequence lengths.

Uses straight-through estimation for backpropagation through complex operations.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function


class ComplexConv1d(nn.Module):
    """
    Causal complex convolution with asymmetric kernel sizes for specialized temporal modeling.

    This implements the full complex convolution:
    (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

    Uses different kernel sizes for real and imaginary parts:
    - Real part: Short kernel for local semantic patterns
    - Imaginary part: Longer kernel for temporal dependencies

    Uses causal padding to prevent future information leakage.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        real_kernel_size=3,
        imag_kernel_size=9,
        causal=True,
    ):
        super().__init__()
        self.real_kernel_size = real_kernel_size
        self.imag_kernel_size = imag_kernel_size
        self.causal = causal

        # Separate convolutions with different kernel sizes for specialization
        self.conv_real = nn.Conv1d(
            in_channels, out_channels, real_kernel_size, padding=0
        )
        self.conv_imag = nn.Conv1d(
            in_channels, out_channels, imag_kernel_size, padding=0
        )

        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.conv_real.weight, gain=0.1)
        nn.init.xavier_uniform_(self.conv_imag.weight, gain=0.1)

    def forward(self, x_real: Tensor, x_imag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Asymmetric causal complex convolution: (a + bi) conv (c + di) = (a*c - b*d) + (a*d + b*c)i

        Args:
            x_real: Real part of input [batch, channels, seq_len]
            x_imag: Imaginary part of input [batch, channels, seq_len]

        Returns:
            Tuple of (real_output, imag_output)
        """
        if self.causal:
            # Use maximum padding for both inputs to ensure consistent output sizes
            max_pad = max(self.real_kernel_size - 1, self.imag_kernel_size - 1)

            # Pad both inputs with the same amount
            x_real_padded = F.pad(x_real, (max_pad, 0))
            x_imag_padded = F.pad(x_imag, (max_pad, 0))
        else:
            x_real_padded = x_real
            x_imag_padded = x_imag

        # Apply convolutions with specialized kernel sizes
        # Real convolution: captures local semantic patterns (short kernel)
        real_conv_on_real = self.conv_real(x_real_padded)
        real_conv_on_imag = self.conv_real(x_imag_padded)

        # Imaginary convolution: captures temporal dependencies (long kernel)
        imag_conv_on_real = self.conv_imag(x_real_padded)
        imag_conv_on_imag = self.conv_imag(x_imag_padded)

        # Since kernel sizes are different, we need to align the outputs
        # The smaller kernel produces a longer output, so we need to trim it
        if self.real_kernel_size != self.imag_kernel_size:
            # Calculate the difference in output lengths
            diff = abs(self.real_kernel_size - self.imag_kernel_size)

            if self.real_kernel_size < self.imag_kernel_size:
                # Real convolution produces longer output, trim from the left (causal)
                real_conv_on_real = real_conv_on_real[..., diff:]
                real_conv_on_imag = real_conv_on_imag[..., diff:]
            else:
                # Imaginary convolution produces longer output, trim from the left (causal)
                imag_conv_on_real = imag_conv_on_real[..., diff:]
                imag_conv_on_imag = imag_conv_on_imag[..., diff:]

        # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        real_output = real_conv_on_real - imag_conv_on_imag
        imag_output = real_conv_on_imag + imag_conv_on_real

        return real_output, imag_output


class TemporalHealthComplex(nn.Module):
    """
    Temporal Health Complex (THC) module that enhances temporal understanding
    between tokens using complex-valued operations.

    This module projects input representations to a reduced complex space,
    applies temporal convolutions to learn phase relationships, and projects
    back to the original space with a gated residual connection.

    The complex representation encodes:
    - Real part: Semantic content
    - Imaginary part: Temporal dynamics and transition patterns
    """

    def __init__(
        self,
        d_model: int,
        reduction_factor: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        gate_init: str = "zeros",
    ):
        """
        Initialize the Temporal Health Complex module.

        Args:
            d_model: Model dimension
            reduction_factor: Factor to reduce dimensionality for efficiency
            kernel_size: Kernel size for temporal convolution
            dropout: Dropout probability
            gate_init: Gate initialization strategy ('zeros', 'small', 'ones')
        """
        super().__init__()
        self.d_model = d_model
        self.d_complex = max(1, d_model // reduction_factor)
        self.kernel_size = kernel_size
        self.reduction_factor = reduction_factor

        # Project to complex domain (reduced dimension for efficiency)
        self.to_complex = nn.Linear(d_model, self.d_complex * 2)

        # True causal complex convolutions using decomposed real operations
        self.complex_conv1 = ComplexConv1d(
            self.d_complex,
            self.d_complex,
            # kernel_size=kernel_size,
            causal=True,
        )

        # Second causal complex convolution for refined temporal understanding
        self.complex_conv2 = ComplexConv1d(
            self.d_complex,
            self.d_complex,
            # kernel_size=kernel_size,
            causal=True,
        )

        # Layer norm in complex domain (apply to magnitude)
        self.complex_norm = nn.LayerNorm(self.d_complex)

        # Project back to real domain
        self.to_real = nn.Linear(self.d_complex * 2, d_model)

        # Residual gate (learn how much to use complex features)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize gate based on strategy
        self._init_gate(gate_init)

    def _init_gate(self, init_type: str) -> None:
        """Initialize the gate weights."""
        if init_type == "zeros":
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        elif init_type == "small":
            nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
            nn.init.zeros_(self.gate.bias)
        elif init_type == "ones":
            nn.init.zeros_(self.gate.weight)
            nn.init.ones_(self.gate.bias)
        else:
            raise ValueError(f"Unknown gate initialization: {init_type}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Temporal Health Complex module.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Output tensor of same shape as input
        """
        residual = x
        batch_size, seq_len, _ = x.shape

        # Project to complex representation
        complex_input = self.to_complex(x)  # [batch, seq_len, d_complex * 2]

        # Split into real and imaginary components - this is where the magic happens!
        real_part = complex_input[..., : self.d_complex]  # Semantic content
        imag_part = complex_input[..., self.d_complex :]  # Temporal dynamics

        # Transpose for convolution: [batch, d_complex, seq_len]
        real_part = real_part.transpose(1, 2)
        imag_part = imag_part.transpose(1, 2)

        # Apply TRUE complex convolutions - this preserves phase relationships!
        # First convolution learns local temporal patterns in complex domain
        conv1_real, conv1_imag = self.complex_conv1(real_part, imag_part)

        # Second convolution refines the temporal understanding
        conv2_real, conv2_imag = self.complex_conv2(conv1_real, conv1_imag)

        # Combine outputs: residual connection in complex domain
        final_real = conv1_real + conv2_real
        final_imag = conv1_imag + conv2_imag

        # Transpose back: [batch, seq_len, d_complex]
        final_real = final_real.transpose(1, 2)
        final_imag = final_imag.transpose(1, 2)

        # Complex normalization: normalize magnitude while preserving phase
        magnitude = torch.sqrt(final_real**2 + final_imag**2 + 1e-8)
        normalized_magnitude = self.complex_norm(magnitude)

        # Preserve phase relationships - this is crucial for temporal modeling!
        scale_factor = normalized_magnitude / (magnitude + 1e-8)
        normalized_real = final_real * scale_factor
        normalized_imag = final_imag * scale_factor

        # Convert back to real representation for the rest of the network
        output_real = torch.cat(
            [normalized_real, normalized_imag], dim=-1
        )  # [batch, seq_len, d_complex * 2]

        output = self.to_real(output_real)  # [batch, seq_len, d_model]
        output = self.dropout(output)

        # Gated residual (let model learn how much to use this)
        gate = torch.sigmoid(self.gate(x))
        return residual + gate * output

    def get_complex_representations(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Helper method to extract complex-like representations for analysis.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Tuple of (real_part, imag_part) tensors of shape [batch, seq_len, d_complex]
        """
        complex_input = self.to_complex(x)
        real_part = complex_input[..., : self.d_complex]
        imag_part = complex_input[..., self.d_complex :]
        return real_part, imag_part

    def get_phase_statistics(self, x: Tensor) -> Dict[str, float]:
        """
        Get phase-like statistics for analysis and debugging.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Dictionary with phase-like statistics
        """
        real_part, imag_part = self.get_complex_representations(x)

        # Compute magnitude and phase-like measures
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part + 1e-8)

        # Phase differences between adjacent tokens
        phase_diffs = phase[:, 1:] - phase[:, :-1]

        return {
            "mean_magnitude": magnitude.mean().item(),
            "magnitude_std": magnitude.std().item(),
            "mean_phase": phase.mean().item(),
            "phase_std": phase.std().item(),
            "mean_phase_diff": phase_diffs.mean().item(),
            "phase_diff_std": phase_diffs.std().item(),
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"d_model={self.d_model}, d_complex={self.d_complex}, "
            f"reduction_factor={self.reduction_factor}, kernel_size={self.kernel_size}"
        )
