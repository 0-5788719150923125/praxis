import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class LMul(nn.Module):
    """Linear-complexity multiplication (L-Mul) implementation from the paper
    'Addition is All You Need for Energy-efficient Language Models'.
    https://arxiv.org/abs/2410.00907
    """

    def __init__(self, mantissa_bits=3):
        super().__init__()
        self.mantissa_bits = mantissa_bits

    def get_exponent_mantissa(self, x):
        """Extract exponent and mantissa from floating point number."""
        # Get absolute value to handle signs separately
        abs_x = torch.abs(x)

        # Get exponent using log2
        exponent = torch.floor(torch.log2(abs_x + 1e-10))

        # Calculate mantissa
        mantissa = abs_x / (2.0**exponent) - 1.0

        # Quantize mantissa to specified bits
        scale = 2**self.mantissa_bits
        mantissa = torch.floor(mantissa * scale) / scale

        return exponent, mantissa

    def l_function(self, m):
        """Implement the l(m) function from the paper."""
        if m <= 3:
            return m
        elif m == 4:
            return 3
        else:
            return 4

    def forward(self, x, y):
        """Forward pass implementing L-Mul algorithm.

        Args:
            x (Tensor): First input tensor
            y (Tensor): Second input tensor

        Returns:
            Tensor: Result of L-Mul operation
        """
        # Handle zero values
        zero_mask = (x == 0) | (y == 0)

        # Get exponents and mantissas
        x_exp, x_man = self.get_exponent_mantissa(x)
        y_exp, y_man = self.get_exponent_mantissa(y)

        # Calculate result using L-Mul formula
        l_m = self.l_function(self.mantissa_bits)
        result = (1 + x_man + y_man + 2 ** (-l_m)) * 2 ** (x_exp + y_exp)

        # Handle signs
        signs = torch.sign(x) * torch.sign(y)
        result = result * signs

        # Zero out results where inputs were zero
        result = torch.where(zero_mask, torch.zeros_like(result), result)

        return result


def lmul(x, y, mantissa_bits=3):
    """Functional interface for L-Mul operation.

    Args:
        x (Tensor): First input tensor
        y (Tensor): Second input tensor
        mantissa_bits (int): Number of mantissa bits to use

    Returns:
        Tensor: Result of L-Mul operation
    """
    module = LMul(mantissa_bits)
    return module(x, y)


def profile_memory(func) -> Tuple[float, float]:
    """Profile peak memory usage of a function."""
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()

    start_time = time.time()
    result = func()
    end_time = time.time()

    peak_mem = torch.cuda.max_memory_allocated()
    return result, peak_mem - start_mem, end_time - start_time


if __name__ == "__main__":
    print("Running L-Mul tests and comparisons...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test 1: Basic functionality with small tensors
    print("\n1. Basic Functionality Test:")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(device)
    y = torch.tensor([2.0, 3.0, 4.0, 5.0]).to(device)

    standard_result = x * y
    lmul_result = lmul(x, y)

    print(f"Standard multiplication: {standard_result}")
    print(f"L-Mul result:           {lmul_result}")
    print(
        f"Relative error: {torch.mean(torch.abs(standard_result - lmul_result) / standard_result):.6f}"
    )

    # Test 2: Memory usage comparison with large tensors
    print("\n2. Memory Usage Test:")
    size = 1000

    def standard_mul():
        a = torch.randn(size, size).to(device)
        b = torch.randn(size, size).to(device)
        return torch.matmul(a, b)

    def lmul_operation():
        a = torch.randn(size, size).to(device)
        b = torch.randn(size, size).to(device)
        return lmul(a, b)

    # Profile standard multiplication
    _, std_memory, std_time = profile_memory(standard_mul)
    print(f"Standard multiplication memory: {std_memory/1024**2:.2f} MB")
    print(f"Standard multiplication time: {std_time:.2f} seconds")

    # Profile L-Mul
    _, lmul_memory, lmul_time = profile_memory(lmul_operation)
    print(f"L-Mul memory: {lmul_memory/1024**2:.2f} MB")
    print(f"L-Mul time: {lmul_time:.2f} seconds")
    print(f"Memory reduction: {(1 - lmul_memory/std_memory)*100:.1f}%")

    # Test 3: Accuracy comparison with different mantissa bits
    print("\n3. Accuracy vs Mantissa Bits Test:")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    standard_result = x * y

    for bits in [2, 3, 4, 5]:
        lmul_result = lmul(x, y, mantissa_bits=bits)
        relative_error = torch.mean(
            torch.abs(standard_result - lmul_result) / torch.abs(standard_result)
        )
        print(f"Mantissa bits: {bits}, Average relative error: {relative_error:.6f}")

    # Test 4: Matrix multiplication comparison
    print("\n4. Matrix Multiplication Test:")
    batch_size = 32
    seq_len = 512
    hidden_dim = 768  # Typical transformer dimensions

    def standard_matmul():
        q = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        k = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        return torch.matmul(q, k.transpose(-2, -1))

    def lmul_matmul():
        q = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        k = torch.randn(batch_size, seq_len, hidden_dim).to(device)
        # Implement matrix multiplication using L-Mul
        result = torch.zeros(batch_size, seq_len, seq_len).to(device)
        for i in range(hidden_dim):
            result += lmul(q[..., i : i + 1], k[..., i : i + 1].transpose(-2, -1))
        return result

    # Profile matrix multiplication
    std_result, std_mem, std_time = profile_memory(standard_matmul)
    lmul_result, lmul_mem, lmul_time = profile_memory(lmul_matmul)

    print(f"Standard matmul memory: {std_mem/1024**2:.2f} MB")
    print(f"L-Mul matmul memory: {lmul_mem/1024**2:.2f} MB")
    print(f"Memory reduction: {(1 - lmul_mem/std_mem)*100:.1f}%")
    print(
        f"Relative error: {torch.mean(torch.abs(std_result - lmul_result) / torch.abs(std_result)):.6f}"
    )
