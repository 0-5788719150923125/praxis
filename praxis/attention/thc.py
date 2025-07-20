"""
Temporal Health Complex (THC) Module for Transformers

Enhances temporal reasoning using complex numbers in a computationally efficient way.
The module uses complex-valued convolutions to learn phase relationships between tokens,
improving the model's ability to understand temporal patterns and generalize across
sequence lengths.

Uses straight-through estimation for backpropagation through complex operations.
"""

from typing import Dict, Optional, Tuple

import math
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
    Temporal Health Complex (THC) module that captures phase relationships between tokens
    using a causal approach with complex-valued representations.
    
    This implementation measures relative phase shifts between adjacent tokens using
    a combination of:
    1. Causal complex convolutions (ComplexConv1d) for local phase relationships
    2. Learned phase evolution that accumulates causally through the sequence
    3. Phase-preserving normalization to maintain stability
    
    The key insight is that phase relationships should be measured between tokens,
    not from absolute positions, to maintain causality in autoregressive models.
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
            kernel_size: Kernel size for complex convolutions
            dropout: Dropout probability
            gate_init: Gate initialization strategy ('zeros', 'small', 'ones')
        """
        super().__init__()
        self.d_model = d_model
        self.d_complex = max(1, d_model // reduction_factor)
        self.kernel_size = kernel_size
        self.reduction_factor = reduction_factor
        
        # Project to complex space (real and imaginary parts)
        self.to_complex = nn.Linear(d_model, self.d_complex * 2)
        
        # Complex convolution layers with different kernel sizes
        # First layer: captures local phase relationships (small kernel)
        self.complex_conv1 = ComplexConv1d(
            self.d_complex, self.d_complex, 
            real_kernel_size=3, 
            imag_kernel_size=5,
            causal=True
        )
        
        # Second layer: captures longer-range phase evolution (larger kernel)
        self.complex_conv2 = ComplexConv1d(
            self.d_complex, self.d_complex,
            real_kernel_size=5,
            imag_kernel_size=9,
            causal=True
        )
        
        # Phase evolution network - learns how phase changes propagate causally
        self.phase_evolution = nn.Sequential(
            nn.Linear(self.d_complex * 2, self.d_complex * 4),
            nn.GELU(),
            nn.Linear(self.d_complex * 4, self.d_complex * 2)
        )
        
        # Complex-aware normalization
        self.complex_norm = nn.LayerNorm(self.d_complex * 2)
        
        # Project back to model space
        self.to_model_space = nn.Linear(self.d_complex * 2, d_model)
        
        # Residual gate
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize gate
        self._init_gate(gate_init)
        
        # Initialize weights for stability
        nn.init.xavier_uniform_(self.to_complex.weight, gain=0.1)
        nn.init.zeros_(self.to_complex.bias)
        nn.init.xavier_uniform_(self.to_model_space.weight, gain=0.1)
        nn.init.zeros_(self.to_model_space.bias)
        
        # Initialize phase evolution to be near-identity initially
        with torch.no_grad():
            # Initialize first layer with small random weights
            if hasattr(self.phase_evolution[0], 'weight'):
                nn.init.xavier_uniform_(self.phase_evolution[0].weight, gain=0.1)
                self.phase_evolution[0].bias.data.zero_()
            # Initialize second layer to approximate identity mapping
            if hasattr(self.phase_evolution[2], 'weight'):
                # Create identity-like initialization
                nn.init.xavier_uniform_(self.phase_evolution[2].weight, gain=0.1)
                # Add identity to the center part
                d = self.d_complex * 2
                if self.phase_evolution[2].weight.shape[0] == d and self.phase_evolution[2].weight.shape[1] >= d:
                    self.phase_evolution[2].weight.data[:d, :d] += 0.1 * torch.eye(d)
                self.phase_evolution[2].bias.data.zero_()

    def __repr__(self) -> str:
        """String representation of the module."""
        return f"{self.__class__.__name__}()"

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
        
        This implementation uses causal complex convolutions to capture phase
        relationships between tokens without violating autoregressive constraints.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Output tensor of same shape as input
        """
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # Project to complex representation (real and imaginary parts)
        complex_features = self.to_complex(x)  # [batch, seq_len, d_complex * 2]
        
        # Split into real and imaginary parts for complex processing
        real_part = complex_features[..., :self.d_complex]  # [batch, seq_len, d_complex]
        imag_part = complex_features[..., self.d_complex:]  # [batch, seq_len, d_complex]
        
        # Reshape for convolution [batch, channels, seq_len]
        real_part = real_part.transpose(1, 2)
        imag_part = imag_part.transpose(1, 2)
        
        # First complex convolution - captures local phase relationships
        real_part, imag_part = self.complex_conv1(real_part, imag_part)
        
        # Second complex convolution - captures longer-range phase evolution
        real_part, imag_part = self.complex_conv2(real_part, imag_part)
        
        # Reshape back to [batch, seq_len, channels]
        real_part = real_part.transpose(1, 2)
        imag_part = imag_part.transpose(1, 2)
        
        # Combine real and imaginary parts
        complex_output = torch.cat([real_part, imag_part], dim=-1)  # [batch, seq_len, d_complex * 2]
        
        # Apply phase evolution network
        # This learns how phase relationships evolve through the sequence
        phase_evolved = self.phase_evolution(complex_output)
        complex_output = complex_output + phase_evolved  # Residual connection
        
        # Complex-aware normalization
        # This preserves phase information while normalizing magnitude
        complex_output = self.complex_norm(complex_output)
        
        # Project back to model dimension
        output = self.to_model_space(complex_output)  # [batch, seq_len, d_model]
        output = self.dropout(output)
        
        # Gated residual connection
        gate = torch.sigmoid(self.gate(x))
        return residual + gate * output

    def get_complex_representations(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Helper method to extract complex representations for analysis.
        Returns real and imaginary parts of the complex features.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Tuple of (real_part, imag_part) tensors of shape [batch, seq_len, d_complex]
        """
        complex_features = self.to_complex(x)
        real_part = complex_features[..., :self.d_complex]
        imag_part = complex_features[..., self.d_complex:]
        return real_part, imag_part

    def get_phase_statistics(self, x: Tensor) -> Dict[str, float]:
        """
        Get phase statistics for analysis and debugging.
        
        This computes statistics about the phase relationships learned by the module,
        including how phases evolve between adjacent tokens (causal).

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Dictionary with phase statistics
        """
        real_part, imag_part = self.get_complex_representations(x)
        
        # Compute magnitude and phase
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        phase = torch.atan2(imag_part, real_part)
        
        # Phase differences between adjacent tokens (causal relationship)
        phase_diffs = phase[:, 1:] - phase[:, :-1]
        
        # Normalize phase differences to [-pi, pi]
        phase_diffs = torch.atan2(torch.sin(phase_diffs), torch.cos(phase_diffs))
        
        return {
            "mean_magnitude": magnitude.mean().item(),
            "magnitude_std": magnitude.std().item(),
            "mean_phase": phase.mean().item(),
            "phase_std": phase.std().item(),
            "mean_phase_diff": phase_diffs.mean().item(),
            "phase_diff_std": phase_diffs.std().item(),
            "phase_coherence": torch.abs(torch.exp(1j * phase_diffs).mean()).item(),  # Measure of phase consistency
        }

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"d_model={self.d_model}, d_complex={self.d_complex}, "
            f"reduction_factor={self.reduction_factor}, kernel_size={self.kernel_size}"
        )
