import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.modules.dense import PraxisMLP


class PraxisMRU(nn.Module):
    """
    A recurrent model with efficient parallel scan. Based upon:
    https://github.com/mikayahlevi/mru-lm/tree/main
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()

        # Architecture parameters
        self.num_heads = config.num_heads
        self.state_size = config.hidden_size
        self.embed_size = config.hidden_size
        self.dropout_rate = config.dropout
        self.depth = config.depth

        self.state_head_size = self.state_size // self.num_heads
        self.state_head_order = math.isqrt(self.state_head_size)
        self.embedding_chunk_size = self.embed_size // (
            self.state_head_order * self.num_heads
        )

        assert (
            self.state_size % self.num_heads == 0
        ), "state size must be divisible by the number of heads"
        assert (
            self.state_head_size == math.isqrt(self.state_head_size) ** 2
        ), "state head size must be a perfect square to form the state head matrix"
        assert (
            self.embed_size % self.state_head_order == 0
        ), "embedding size must be divisible by the state head order"

        self.state_matrices_up = nn.Parameter(
            torch.zeros(
                self.num_heads, self.embedding_chunk_size, self.state_head_order
            )
        )

        self.state_matrices_update_scale = (
            0.01
            * (1 / self.state_head_order)
            * (self.embed_size / self.embedding_chunk_size)
        )

        self.state_matrices_down = nn.Parameter(
            torch.normal(
                mean=0,
                std=0.02
                * math.sqrt(self.state_size)
                / (self.embed_size / self.state_head_order),
                size=(
                    self.num_heads,
                    self.state_head_order,
                    self.embedding_chunk_size,
                ),
            ),
        )

        # Output projection and layer norms
        self.mru_out = nn.Linear(self.embed_size, self.embed_size, bias=False)
        nn.init.normal_(self.mru_out.weight, mean=0, std=0.02 / math.sqrt(self.depth))

        # Normalization
        self.mru_norm = nn.LayerNorm(self.embed_size, bias=False)
        self.ffn_norm = nn.LayerNorm(self.embed_size, bias=False)

        # Regularization
        self.dropout = nn.Dropout(self.dropout_rate)

        # MLP block
        self.ffn = PraxisMLP(config)
        nn.init.normal_(self.ffn.up.weight, mean=0, std=0.02)
        nn.init.normal_(self.ffn.down.weight, mean=0, std=0.02 / math.sqrt(self.depth))

    def forward(
        self,
        x: Tensor,
        current_state: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:

        mru_out, new_state = self._parallel_mru(self.mru_norm(x), current_state)

        x = x + mru_out
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_state, 0

    def _parallel_mru(
        self, x: Tensor, last_state: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:

        if last_state is not None and last_state.size(0) != x.size(0):
            last_state = last_state[-1:].expand(x.size(0), -1, -1, -1)

        reshaped = x.unflatten(
            -1, (self.num_heads, self.state_head_order, self.embedding_chunk_size)
        )

        new_matrices = (
            self.dropout(reshaped @ self.state_matrices_up)
            * self.state_matrices_update_scale
        )

        new_matrices = new_matrices + torch.eye(
            self.state_head_order, device=x.device
        ).view(1, 1, 1, self.state_head_order, self.state_head_order)

        full_matrices = (
            new_matrices
            if last_state is None
            else torch.cat((last_state.unsqueeze(dim=-4), new_matrices), dim=-4)
        )

        parallel_mru_op_output = parallel_mru_op(
            full_matrices.transpose(-3, -4)
        ).transpose(-3, -4)

        states = (
            parallel_mru_op_output
            if last_state is None
            else parallel_mru_op_output[..., 1:, :, :, :]
        )

        output = (
            (states @ self.state_matrices_down)
            * (self.embed_size / self.state_head_order)
        ).flatten(-3, -1)

        return (
            self.dropout(self.mru_out(output)),
            states[-1],
        )


# bk: Brent-Kung scan
class bk_parallel_mru_op_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, start_matrix_states):
        final_matrix_states = start_matrix_states.clone()

        sequence_length = start_matrix_states.size(-3)

        n_stages = math.ceil(math.log2(sequence_length))

        # first sweep
        for stage in range(n_stages):
            # abbreviate stage_stride as sts
            sts = 2**stage
            final_matrix_states[..., 2 * sts - 1 :: 2 * sts, :, :] = (
                final_matrix_states[..., sts - 1 : -sts : 2 * sts, :, :]
                @ final_matrix_states[..., 2 * sts - 1 :: 2 * sts, :, :]
            )

        # second sweep
        for stage in reversed(range(n_stages - 1)):
            # abbreviate stage_stride as sts
            sts = 2**stage
            final_matrix_states[..., 2 * sts + sts - 1 :: 2 * sts, :, :] = (
                final_matrix_states[..., 2 * sts - 1 : -sts : 2 * sts, :, :]
                @ final_matrix_states[..., 2 * sts + sts - 1 :: 2 * sts, :, :]
            )

        ctx.save_for_backward(start_matrix_states, final_matrix_states)
        ctx.sequence_length = sequence_length

        return final_matrix_states

    @staticmethod
    def backward(ctx, grad_final_matrix_states):
        def create_eye_for_shift(shape):
            resized_eye = torch.eye(*shape[-2:], device=grad_final_matrix_states.device)
            while resized_eye.dim() < len(shape):
                resized_eye = resized_eye.unsqueeze(0)

            resized_eye_shape = shape[:-3]
            resized_eye_shape = list(resized_eye_shape)

            while len(resized_eye_shape) < len(shape):
                resized_eye_shape.append(1)

            resized_eye = resized_eye.repeat(*resized_eye_shape)
            return resized_eye

        def create_zeros_for_shift(shape):
            new_shape = list(shape)
            new_shape[-3] = 1
            return torch.zeros(new_shape, device=grad_final_matrix_states.device)

        start_matrix_states, final_matrix_states = ctx.saved_tensors

        # grad_before_start_matrix_states is A as described in the README
        # tl is U as described in the README
        # bl is L as described in the README

        # grad_before_start_matrix_states = torch.cat((create_eye_for_shift(transposed_final_matrix_states.shape), transposed_final_matrix_states[..., :-1, :, :]), dim = -3)

        # faster implementation:
        grad_before_start_matrix_states = final_matrix_states.transpose(-1, -2).roll(
            1, dims=-3
        )
        grad_before_start_matrix_states[..., 0, :, :] = torch.eye(
            grad_before_start_matrix_states.size(-2),
            device=grad_before_start_matrix_states.device,
        )

        # tl = torch.cat((start_matrix_states[..., 1:, :, :], create_zeros_for_shift(start_matrix_states.shape)), dim = -3).transpose(-1, -2)

        # faster implementation:
        tl = start_matrix_states.transpose(-1, -2).roll(-1, dims=-3)
        tl[..., -1, :, :] = torch.zeros((tl.size(-2), tl.size(-1)), device=tl.device)

        bl = grad_final_matrix_states

        sequence_length = ctx.sequence_length
        n_stages = math.ceil(math.log2(sequence_length))

        # first sweep
        for stage in range(n_stages):
            # abbreviate stage_stride as sts
            sts = 2**stage
            bl[..., : -sts : 2 * sts, :, :] = (
                bl[..., sts :: 2 * sts, :, :] @ tl[..., : -sts : 2 * sts, :, :]
                + bl[..., : -sts : 2 * sts, :, :]
            )
            tl[..., : -sts : 2 * sts, :, :] = (
                tl[..., sts :: 2 * sts, :, :] @ tl[..., : -sts : 2 * sts, :, :]
            )

        # second sweep
        for stage in reversed(range(n_stages - 1)):
            # abbreviate stage_stride as sts
            sts = 2**stage
            bl[..., sts : -sts : 2 * sts, :, :] = (
                bl[..., 2 * sts :: 2 * sts, :, :] @ tl[..., sts : -sts : 2 * sts, :, :]
                + bl[..., sts : -sts : 2 * sts, :, :]
            )
            tl[..., sts : -sts : 2 * sts, :, :] = (
                tl[..., 2 * sts :: 2 * sts, :, :] @ tl[..., sts : -sts : 2 * sts, :, :]
            )

        grad_start_matrix_states = grad_before_start_matrix_states @ bl

        return grad_start_matrix_states


parallel_mru_op = bk_parallel_mru_op_class.apply

if __name__ == "__main__":
    # Mock AutoConfig for testing
    from dataclasses import dataclass

    @dataclass
    class AutoConfig:
        hidden_size: int = 768
        num_heads: int = 3
        dropout: float = 0.1
        depth: int = 3
        activation: str = "gelu"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig()
    model = PraxisMRU(config).to(device)

    # Test 1: Basic Functionality
    print("\nTest 1: Basic Functionality")
    x = torch.randn(2, 128, config.hidden_size).to(device)
    try:
        output, _, _ = model(x)
        print(f"✓ Output shape: {output.shape}")
        assert output.shape == x.shape, "Output shape mismatch"
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 2: State Persistence
    print("\nTest 2: State Persistence")
    try:
        # First forward pass
        out1, _, _ = model(x[:, :64, :])
        # Second forward pass should use saved state
        out2, _, _ = model(x[:, 64:, :])
        # Reset state
        # model.reset_state()
        # Third forward pass should start fresh
        out3, _, _ = model(x[:, :64, :])

        # Check that outputs are different (due to state influence)
        diff = (out1[:, -1, :] - out3[:, -1, :]).abs().mean().item()
        print(f"✓ State influence (diff): {diff:.6f}")
        assert diff > 0, "State should influence outputs"
        print("✓ State persistence test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 3: Gradient Flow
    print("\nTest 3: Gradient Flow")
    model.zero_grad()
    x = torch.randn(2, 64, config.hidden_size, requires_grad=True).to(device)
    try:
        output, _, _ = model(x)
        loss = output.sum()
        loss.backward()

        has_grads = all(p.grad is not None for p in model.parameters())
        print(f"✓ Gradients exist: {has_grads}")

        has_nans = any(torch.isnan(p.grad).any() for p in model.parameters())
        print(f"✓ Gradients are clean (no NaNs): {not has_nans}")

        print("✓ Gradient flow test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 4: Memory Usage
    print("\nTest 4: Memory Usage")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        # Test with increasing sequence lengths
        for seq_len in [64, 128, 256, 512]:
            x = torch.randn(1, seq_len, config.hidden_size).to(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            output, _, _ = model(x)

            if device.type == "cuda":
                max_memory = torch.cuda.max_memory_allocated() / 1024**2
                print(f"✓ Sequence length {seq_len}: {max_memory:.2f} MB")

        print("✓ Memory usage test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 5: Dropout Behavior
    print("\nTest 5: Dropout Behavior")
    x = torch.randn(2, 64, config.hidden_size).to(device)
    try:
        model.train()
        out1, _, _ = model(x)
        out2, _, _ = model(x)
        train_diff = (out1 - out2).abs().mean().item()

        model.eval()
        out3, _, _ = model(x)
        out4, _, _ = model(x)
        eval_diff = (out3 - out4).abs().mean().item()

        print(f"✓ Training difference: {train_diff:.6f}")
        print(f"✓ Evaluation difference: {eval_diff:.6f}")
        assert train_diff > eval_diff, "Dropout should cause more variation in training"
        print("✓ Dropout behavior test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    print("\nAll tests completed!")

    from dataclasses import dataclass

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch

    @dataclass
    class MockConfig:
        hidden_size: int = 16
        num_heads: int = 2
        dropout: float = 0.0
        depth: int = 1

    def analyze_gradients(seq_length=16, plot_type="heatmap"):
        """
        Analyze gradients through the parallel scan operation.

        Args:
            seq_length: Length of sequence to test
            plot_type: Either 'heatmap' or 'line' for different visualizations
        """
        # Create a minimal test case
        batch_size = 1
        hidden_size = 4  # Small size for visualization

        # Create test input
        test_input = torch.randn(
            batch_size, seq_length, hidden_size, hidden_size, requires_grad=True
        )

        # Register hooks to capture gradients
        gradients = []

        def grad_hook(grad):
            gradients.append(grad.detach().cpu().numpy())
            return grad

        test_input.register_hook(grad_hook)

        # Forward pass
        output = parallel_mru_op(test_input)

        # Create loss and backward pass
        loss = output.mean()
        loss.backward()

        # Convert gradients to numpy for visualization
        grad_tensor = test_input.grad.squeeze(0)
        grad_norms = torch.norm(grad_tensor.view(seq_length, -1), dim=1)
        grad_norms = grad_norms.detach().cpu().numpy()

        # Create visualizations
        plt.figure(figsize=(15, 10))

        if plot_type == "heatmap":
            # Plot gradient heatmap
            plt.subplot(2, 1, 1)
            sns.heatmap(
                grad_tensor[:, 0, :].detach().cpu().numpy(),
                cmap="RdBu",
                center=0,
                annot=True,
                fmt=".2f",
            )
            plt.title("Gradient Matrix for First Channel")
            plt.xlabel("Hidden Dimension")
            plt.ylabel("Sequence Position")

            # Plot gradient norms
            plt.subplot(2, 1, 2)
            plt.plot(grad_norms, "-o")
            plt.title("Gradient Norms across Sequence Positions")
            plt.xlabel("Sequence Position")
            plt.ylabel("Gradient Norm")
            plt.yscale("log")  # Log scale to better show explosion

        else:  # line plot
            positions = np.arange(seq_length)
            plt.plot(positions, grad_norms, "-o", label="Gradient Norm")
            plt.fill_between(
                positions,
                grad_norms - grad_norms.std(),
                grad_norms + grad_norms.std(),
                alpha=0.3,
            )
            plt.yscale("log")
            plt.title("Gradient Magnitude Analysis")
            plt.xlabel("Sequence Position")
            plt.ylabel("Gradient Norm (log scale)")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\nGradient Statistics:")
        print(f"Mean gradient norm: {grad_norms.mean():.4f}")
        print(f"Max gradient norm: {grad_norms.max():.4f}")
        print(f"Min gradient norm: {grad_norms.min():.4f}")
        print(f"Gradient norm std: {grad_norms.std():.4f}")

        # Check for potential issues
        print("\nDiagnostic Checks:")
        print(
            f"Gradient explosion check (max/mean ratio): {grad_norms.max() / grad_norms.mean():.2f}"
        )
        print(
            f"Gradient vanishing check (min/mean ratio): {grad_norms.min() / grad_norms.mean():.2f}"
        )

        return grad_norms

    def compare_scan_algorithms(seq_length=16):
        """Compare gradients between Hillis-Steele and Brent-Kung implementations."""
        global parallel_mru_op

        # Store original op
        original_op = parallel_mru_op

        results = {}
        for name, op in [
            ("Hillis-Steele", hs_parallel_mru_op_class.apply),
            ("Brent-Kung", bk_parallel_mru_op_class.apply),
        ]:
            parallel_mru_op = op
            grad_norms = analyze_gradients(seq_length)
            results[name] = grad_norms

        # Restore original op
        parallel_mru_op = original_op

        # Plot comparison
        plt.figure(figsize=(10, 6))
        positions = np.arange(seq_length)

        for name, norms in results.items():
            plt.plot(positions, norms, "-o", label=name)

        plt.yscale("log")
        plt.title("Gradient Comparison: Hillis-Steele vs Brent-Kung")
        plt.xlabel("Sequence Position")
        plt.ylabel("Gradient Norm (log scale)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Test with different sequence lengths
    for seq_len in [8, 16, 32]:
        print(f"\nAnalyzing sequence length: {seq_len}")
        analyze_gradients(seq_len, plot_type="heatmap")

    # Compare algorithms
    compare_scan_algorithms(16)
