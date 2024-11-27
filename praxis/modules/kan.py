import math
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from praxis.activations import ACT2FN


class PraxisKAN(nn.Module):
    """
    An implementation of an approximate Kolmogorov-Arnold Network, of which is
    a theoretical alternative to the traditional Multi-Layer Perceptron. We use
    an approximate here, because it is much more efficient than the standard KAN.
    Code was borrowed from here:
    https://github.com/ZiyaoLi/fast-kan
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: AutoConfig,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = config.num_dims
        output_dim = config.num_dims
        base_activation = ACT2FN[config.activation]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2,
    ):
        """this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        """
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]  # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts,
        )  # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class SplineLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


if __name__ == "__main__":
    import torch

    # Test configurations
    configs = [
        {"batch_size": 2, "seq_len": 3, "dim": 5},
        {"batch_size": 1, "seq_len": 1, "dim": 2},
        {"batch_size": 32, "seq_len": 10, "dim": 64},
    ]

    def run_test(config):
        print(f"\nTesting configuration: {config}")

        class DummyConfig:
            num_dims = config["dim"]
            activation = "silu"

        # Create model
        model = PraxisKAN(config=DummyConfig())

        # Create input tensor
        x = torch.randn(config["batch_size"], config["seq_len"], config["dim"])

        try:
            # Forward pass
            y = model(x)

            # Check output shape
            expected_shape = (
                config["batch_size"],
                config["seq_len"],
                config["dim"],
            )
            assert (
                y.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {y.shape}"

            # Check if output contains valid values
            assert not torch.isnan(y).any(), "Output contains NaN values"
            assert not torch.isinf(y).any(), "Output contains infinite values"

            print(f"✓ Test passed! Output shape: {y.shape}")

            # Optional: Print sample of input and output
            print(f"Sample input[0,0,:]: {x[0,0,:]}")
            print(f"Sample output[0,0,:]: {y[0,0,:]}")

        except Exception as e:
            print(f"✗ Test failed: {str(e)}")

    # Run all tests
    for config in configs:
        run_test(config)

    print("\nAdditional tests for edge cases:")

    class DummyConfig:
        num_dims = 2
        activation = "silu"

    # Test with zero inputs
    try:
        model = PraxisKAN(DummyConfig())
        x_zero = torch.zeros(1, 1, 2)
        y_zero = model(x_zero)
        print("✓ Zero input test passed")
    except Exception as e:
        print(f"✗ Zero input test failed: {str(e)}")

    # Test with very large values
    try:
        model = PraxisKAN(DummyConfig())
        x_large = torch.ones(1, 1, 2) * 1000
        y_large = model(x_large)
        print("✓ Large value test passed")
    except Exception as e:
        print(f"✗ Large value test failed: {str(e)}")
