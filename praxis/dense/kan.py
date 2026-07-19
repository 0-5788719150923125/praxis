import math
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.dense.base import BaseDense

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class KolmogorovArnoldNetwork(BaseDense):
    """
    An implementation of an approximate Kolmogorov-Arnold Network, of which is
    a theoretical alternative to the traditional Multi-Layer Perceptron. We use
    an approximate here, because it is much more efficient than the standard KAN.
    Code was borrowed from here:
    https://github.com/ZiyaoLi/fast-kan
    """

    def __init__(
        self,
        config: ConfigType,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        grid_spacing: str = "linear",
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        """
        Initialize KAN module.

        Args:
            config: Configuration object with model parameters
            grid_min: Minimum grid value for RBF
            grid_max: Maximum grid value for RBF
            num_grids: Number of grid points
            grid_spacing: ``"linear"`` (uniform centers, stock FastKAN) or
                ``"geometric"`` (log-magnitude centers with per-center widths - a
                coarse-to-fine radial cascade, dense near 0 and sparse far out).
                The geometric grid resolves multiple scales with fewer centers,
                so it keeps ``num_grids`` (and the fast-weight cost when this is a
                test-time memory net) low while covering a wide dynamic range.
            use_base_update: Whether to use base update
            spline_weight_init_scale: Scale for weight initialization
        """
        super().__init__()
        input_dim = config.hidden_size
        output_dim = config.hidden_size
        base_activation = ACT2FN[config.activation]
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.rbf: RadialBasisFunction = RadialBasisFunction(
            grid_min, grid_max, num_grids, spacing=grid_spacing
        )
        self.spline_linear: SplineLinear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale
        )
        self.use_base_update: bool = use_base_update
        if use_base_update:
            self.base_activation: Callable[[Tensor], Tensor] = base_activation
            self.base_linear: nn.Linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass through KAN module.

        Args:
            inputs: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after KAN processing
        """
        spline_basis = self.rbf(inputs)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(inputs))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return the learned curves in a FastKANLayer.

        Args:
            input_index: Selected index of the input, in [0, input_dim)
            output_index: Selected index of the output, in [0, output_dim)
            num_pts: Number of points sampled for the curve
            num_extrapolate_bins: Number of bins extrapolating from the given grids
                The curve will be calculated in [grid_min - h * N_e, grid_max + h * N_e]

        Returns:
            Tuple containing:
                - x-values for the curve
                - y-values for the curve
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
    """
    Linear layer with special initialization for spline weights.
    """

    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw: Any
    ) -> None:
        """
        Initialize spline linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            init_scale: Scale for weight initialization
            **kw: Additional keyword arguments for nn.Linear
        """
        self.init_scale: float = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        """Initialize weights using truncated normal distribution"""
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    """
    Radial basis function module for KAN.
    """

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: Optional[
            float
        ] = None,  # larger denominators lead to smoother basis
        spacing: str = "linear",
    ) -> None:
        """
        Initialize radial basis function module.

        Args:
            grid_min: Minimum grid value
            grid_max: Maximum grid value
            num_grids: Number of grid points
            denominator: Denominator for RBF calculation (larger values = smoother basis)
            spacing: ``"linear"`` for uniform centers (single width) or
                ``"geometric"`` for log-magnitude centers, each with its own
                width set to its neighbour gap - a coarse-to-fine cascade.

        The grid and per-center widths are fixed BUFFERS, not parameters, so a
        test-time memory net that replicates its fast weights per chunk never
        copies or surprise-updates them: the multi-scale basis stays put.
        """
        super().__init__()
        self.grid_min: float = grid_min
        self.grid_max: float = grid_max
        self.num_grids: int = num_grids
        self.spacing: str = spacing
        if spacing == "geometric":
            grid, denom = self._geometric_grid(grid_min, grid_max, num_grids)
        else:
            grid = torch.linspace(grid_min, grid_max, num_grids)
            step = (grid_max - grid_min) / (num_grids - 1)
            denom = torch.full((num_grids,), float(denominator or step))
        self.register_buffer("grid", grid)
        self.register_buffer("denom", denom)
        # Back-compat scalar (plot_curve / callers read `.denominator`); the
        # forward path uses the per-center `denom` buffer.
        self.denominator: float = float(denom.mean())

    @staticmethod
    def _geometric_grid(
        grid_min: float, grid_max: float, num_grids: int
    ) -> Tuple[Tensor, Tensor]:
        """Symmetric log-magnitude centers (factor-2 geometric), each width set
        to its nearest-neighbour gap so the bumps tile at every scale."""
        m = max(abs(grid_min), abs(grid_max))
        half = num_grids // 2
        exps = torch.arange(half - 1, -1, -1, dtype=torch.float32)
        mags = m * (0.5**exps)  # smallest .. m, geometric
        if num_grids % 2 == 1:
            grid = torch.cat([-mags.flip(0), torch.zeros(1), mags])
        else:
            grid = torch.cat([-mags.flip(0), mags])
        grid, _ = torch.sort(grid)
        gaps = torch.full_like(grid, float("inf"))
        d = grid[1:] - grid[:-1]
        gaps[:-1] = torch.minimum(gaps[:-1], d)
        gaps[1:] = torch.minimum(gaps[1:], d)
        denom = gaps.clamp_min(1e-3)
        return grid, denom

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through RBF.

        Args:
            x: Input tensor

        Returns:
            RBF values for the input
        """
        return torch.exp(-(((x[..., None] - self.grid) / self.denom) ** 2))


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
            hidden_size = config["dim"]
            activation = "silu"

        # Create model
        model = KolmogorovArnoldNetwork(config=DummyConfig())

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
        hidden_size = 2
        activation = "silu"

    # Test with zero inputs
    try:
        model = KolmogorovArnoldNetwork(DummyConfig())
        x_zero = torch.zeros(1, 1, 2)
        y_zero = model(x_zero)
        print("✓ Zero input test passed")
    except Exception as e:
        print(f"✗ Zero input test failed: {str(e)}")

    # Test with very large values
    try:
        model = KolmogorovArnoldNetwork(DummyConfig())
        x_large = torch.ones(1, 1, 2) * 1000
        y_large = model(x_large)
        print("✓ Large value test passed")
    except Exception as e:
        print(f"✗ Large value test failed: {str(e)}")
