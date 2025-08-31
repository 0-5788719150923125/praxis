"""
Test suite for Temporal Health Complex (THC) module.

Tests cover:
- Basic functionality and shape preservation
- Complex number operations and phase relationships
- Gradient flow and training compatibility
- Integration with transformer blocks
- Performance characteristics
- Sequence generalization capabilities
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from praxis import PraxisConfig
from praxis.attention.thc import TemporalHealthComplex
from praxis.blocks.transformer import TransformerBlock


class TestTemporalHealthComplex:
    """Test suite for the TemporalHealthComplex module."""

    @pytest.fixture
    def thc_module(self):
        """Create a standard THC module for testing."""
        return TemporalHealthComplex(d_model=256, reduction_factor=8, kernel_size=3)

    @pytest.fixture
    def test_input(self):
        """Create test input tensor."""
        return torch.randn(4, 32, 256)

    def test_module_initialization(self):
        """Test that THC module initializes correctly with various parameters."""
        # Test default initialization
        thc = TemporalHealthComplex(d_model=128)
        assert thc.d_model == 128
        assert thc.d_complex == 16  # 128 // 8
        assert thc.kernel_size == 3

        # Test custom parameters
        thc = TemporalHealthComplex(
            d_model=256, reduction_factor=4, kernel_size=5, dropout=0.2
        )
        assert thc.d_model == 256
        assert thc.d_complex == 64  # 256 // 4
        assert thc.kernel_size == 5

        # Test edge case with small d_model
        thc = TemporalHealthComplex(d_model=4, reduction_factor=8)
        assert thc.d_complex == 1  # max(1, 4 // 8)

    def test_gate_initialization_strategies(self):
        """Test different gate initialization strategies."""
        # Test zeros initialization
        thc_zeros = TemporalHealthComplex(d_model=128, gate_init="zeros")
        assert torch.allclose(
            thc_zeros.gate.weight, torch.zeros_like(thc_zeros.gate.weight)
        )
        assert torch.allclose(
            thc_zeros.gate.bias, torch.zeros_like(thc_zeros.gate.bias)
        )

        # Test small initialization
        thc_small = TemporalHealthComplex(d_model=128, gate_init="small")
        assert not torch.allclose(
            thc_small.gate.weight, torch.zeros_like(thc_small.gate.weight)
        )
        assert torch.allclose(
            thc_small.gate.bias, torch.zeros_like(thc_small.gate.bias)
        )

        # Test ones initialization
        thc_ones = TemporalHealthComplex(d_model=128, gate_init="ones")
        assert torch.allclose(
            thc_ones.gate.weight, torch.zeros_like(thc_ones.gate.weight)
        )
        assert torch.allclose(thc_ones.gate.bias, torch.ones_like(thc_ones.gate.bias))

        # Test invalid initialization
        with pytest.raises(ValueError, match="Unknown gate initialization"):
            TemporalHealthComplex(d_model=128, gate_init="invalid")

    def test_forward_shape_preservation(self, thc_module, test_input):
        """Test that forward pass preserves input shape."""
        output = thc_module(test_input)
        assert output.shape == test_input.shape
        assert output.dtype == test_input.dtype

    def test_forward_different_shapes(self, thc_module):
        """Test forward pass with different input shapes."""
        shapes = [
            (1, 10, 256),  # Small batch, short sequence
            (8, 64, 256),  # Medium batch, medium sequence
            (2, 128, 256),  # Small batch, long sequence
        ]

        for batch_size, seq_len, d_model in shapes:
            x = torch.randn(batch_size, seq_len, d_model)
            output = thc_module(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_gradient_flow(self, thc_module, test_input):
        """Test that gradients flow correctly through the module."""
        test_input.requires_grad_(True)
        thc_module.train()

        output = thc_module(test_input)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert test_input.grad is not None
        assert not torch.allclose(test_input.grad, torch.zeros_like(test_input.grad))

        # Check that all module parameters have gradients
        for name, param in thc_module.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            if "complex_conv" in name:
                # Complex parameters should have non-zero gradients
                assert not torch.allclose(
                    param.grad, torch.zeros_like(param.grad)
                ), f"Zero gradient for complex parameter: {name}"

    def test_complex_representations(self, thc_module, test_input):
        """Test complex-like representation extraction and properties."""
        real_part, imag_part = thc_module.get_complex_representations(test_input)

        # Check shape and dtype
        expected_shape = (
            test_input.shape[0],
            test_input.shape[1],
            thc_module.d_complex,
        )
        assert real_part.shape == expected_shape
        assert imag_part.shape == expected_shape
        assert real_part.dtype == torch.float32
        assert imag_part.dtype == torch.float32

        # Check that both real and imaginary parts have meaningful values
        assert not torch.allclose(real_part, torch.zeros_like(real_part))
        assert not torch.allclose(imag_part, torch.zeros_like(imag_part))

    def test_phase_statistics(self, thc_module, test_input):
        """Test phase statistics computation."""
        stats = thc_module.get_phase_statistics(test_input)

        # Check that all expected keys are present
        expected_keys = {
            "mean_magnitude",
            "magnitude_std",
            "mean_phase",
            "phase_std",
            "mean_phase_diff",
            "phase_diff_std",
            "phase_coherence",
        }
        assert set(stats.keys()) == expected_keys

        # Check that all values are finite
        for key, value in stats.items():
            assert np.isfinite(value), f"Non-finite value for {key}: {value}"

        # Check reasonable ranges
        assert stats["magnitude_std"] >= 0
        assert stats["phase_diff_std"] >= 0
        assert 0 <= stats["phase_coherence"] <= 1, f"Phase coherence out of range: {stats['phase_coherence']}"

    def test_training_mode_effects(self, thc_module, test_input):
        """Test that training/eval modes affect the module appropriately."""
        thc_module.train()
        output_train = thc_module(test_input)

        thc_module.eval()
        with torch.no_grad():
            output_eval = thc_module(test_input)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval)

    def test_deterministic_behavior(self, thc_module, test_input):
        """Test that the module behaves deterministically in eval mode."""
        thc_module.eval()

        with torch.no_grad():
            output1 = thc_module(test_input)
            output2 = thc_module(test_input)

        assert torch.allclose(output1, output2)

    def test_residual_connection_strength(self):
        """Test that residual connection starts weak and can grow."""
        # Test with zero gate initialization
        thc_zeros = TemporalHealthComplex(d_model=128, gate_init="zeros")
        x = torch.randn(2, 16, 128)

        thc_zeros.eval()
        with torch.no_grad():
            output = thc_zeros(x)
            # With zero gate, output should be close to input (but not exact due to other transformations)
            difference = torch.norm(output - x) / torch.norm(x)
            assert (
                difference < 0.3
            ), f"Output too different from input with zero gate: {difference}"

        # Test with ones gate initialization
        thc_ones = TemporalHealthComplex(d_model=128, gate_init="ones")
        thc_ones.eval()
        with torch.no_grad():
            output = thc_ones(x)
            # With ones gate, output should differ more from input
            assert not torch.allclose(output, x, atol=1e-3)

    def test_parameter_count(self):
        """Test parameter count scaling with different configurations."""
        base_thc = TemporalHealthComplex(d_model=256, reduction_factor=8)
        base_params = sum(p.numel() for p in base_thc.parameters())

        # Test with different reduction factors
        thc_less_reduction = TemporalHealthComplex(d_model=256, reduction_factor=4)
        less_reduction_params = sum(p.numel() for p in thc_less_reduction.parameters())
        assert less_reduction_params > base_params

        # Test with different model dimensions
        thc_larger = TemporalHealthComplex(d_model=512, reduction_factor=8)
        larger_params = sum(p.numel() for p in thc_larger.parameters())
        assert larger_params > base_params

    def test_memory_efficiency(self):
        """Test memory usage with large inputs."""
        thc = TemporalHealthComplex(d_model=1024, reduction_factor=16)  # More efficient

        # Test with large input
        large_input = torch.randn(4, 256, 1024)

        # Should not raise memory errors
        output = thc(large_input)
        assert output.shape == large_input.shape

    @pytest.mark.parametrize("reduction_factor", [2, 4, 8, 16])
    def test_reduction_factor_effects(self, reduction_factor):
        """Test effects of different reduction factors."""
        d_model = 256
        thc = TemporalHealthComplex(d_model=d_model, reduction_factor=reduction_factor)

        expected_d_complex = max(1, d_model // reduction_factor)
        assert thc.d_complex == expected_d_complex

        x = torch.randn(2, 32, d_model)
        output = thc(x)
        assert output.shape == x.shape

    @pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
    def test_kernel_size_effects(self, kernel_size):
        """Test effects of different kernel sizes."""
        thc = TemporalHealthComplex(d_model=128, kernel_size=kernel_size)
        assert thc.kernel_size == kernel_size

        x = torch.randn(2, 32, 128)
        output = thc(x)
        assert output.shape == x.shape


class TestTHCIntegration:
    """Test THC integration with transformer blocks."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return PraxisConfig(
            hidden_size=256,
            num_heads=8,
            depth=2,
            dropout=0.1,
            vocab_size=1000,
            max_length=512,
        )

    def _check_thc_integration(self, block):
        """Helper to check if THC is integrated in the block."""
        # Check various possible integration points
        has_thc = (
            hasattr(block, "use_thc")
            or hasattr(block, "thc")
            or hasattr(block.attn, "thc")
            if hasattr(block, "attn")
            else False
        )
        return has_thc

    @pytest.mark.skipif(
        True, reason="THC integration not yet implemented in TransformerBlock"
    )
    def test_transformer_block_with_thc(self, config):
        """Test that transformer block works with THC enabled."""
        block = TransformerBlock(config)

        # Check that THC is enabled and initialized
        assert hasattr(block, "use_thc")
        assert block.use_thc is True
        assert hasattr(block, "thc")
        assert isinstance(block.thc, TemporalHealthComplex)

        # Test forward pass
        batch_size, seq_len = 4, 32
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)
        # Use proper attention mask with batch dimension
        attention_mask = torch.ones(batch_size, seq_len)

        output, past_kv, state, aux_loss = block(inputs, attention_mask)

        assert output.shape == inputs.shape
        assert isinstance(aux_loss, (int, float, torch.Tensor))

    def test_transformer_block_basic_functionality(self, config):
        """Test basic functionality of transformer block (without assuming THC)."""
        block = TransformerBlock(config)

        # Test forward pass
        batch_size, seq_len = 4, 32
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)
        # Create proper attention mask with batch dimension
        # For causal attention, we typically use a 1D mask per sequence
        attention_mask = torch.ones(batch_size, seq_len)

        output, past_kv, state, aux_loss = block(inputs, attention_mask)

        assert output.shape == inputs.shape
        assert isinstance(aux_loss, (int, float, torch.Tensor))

    @pytest.mark.skipif(
        True, reason="THC integration not yet implemented in TransformerBlock"
    )
    def test_transformer_block_gradient_flow(self, config):
        """Test gradient flow through transformer block with THC."""
        block = TransformerBlock(config)
        block.train()

        batch_size, seq_len = 2, 16
        inputs = torch.randn(
            batch_size, seq_len, config.hidden_size, requires_grad=True
        )
        attention_mask = torch.ones(batch_size, seq_len)

        output, _, _, aux_loss = block(inputs, attention_mask)
        loss = output.sum() + aux_loss
        loss.backward()

        # Check gradients exist
        assert inputs.grad is not None

        # Check THC parameters have gradients
        for name, param in block.thc.named_parameters():
            assert param.grad is not None, f"No gradient for THC parameter: {name}"

    @pytest.mark.skipif(
        True, reason="THC integration not yet implemented in TransformerBlock"
    )
    def test_parameter_count_increase(self, config):
        """Test parameter count increase with THC."""
        # Temporarily disable THC to get baseline
        block_without_thc = TransformerBlock(config)
        block_without_thc.use_thc = False
        params_without = sum(
            p.numel()
            for p in block_without_thc.parameters()
            if "thc" not in [name for name, _ in block_without_thc.named_parameters()]
        )

        block_with_thc = TransformerBlock(config)
        params_with = sum(p.numel() for p in block_with_thc.parameters())

        # THC should add parameters but not too many
        param_increase = params_with - params_without
        assert param_increase > 0

        # Should be reasonable increase (less than 20% typically)
        increase_ratio = param_increase / params_without
        assert (
            increase_ratio < 0.3
        ), f"Parameter increase too large: {increase_ratio:.2%}"


class TestTHCGeneralization:
    """Test THC's sequence generalization capabilities."""

    def test_thc_standalone_phase_learning(self):
        """Test THC module's ability to process phase patterns standalone."""
        config = PraxisConfig(
            hidden_size=128, num_heads=4, depth=2, dropout=0.1, vocab_size=50
        )

        # Create standalone THC module
        thc = TemporalHealthComplex(d_model=config.hidden_size)

        def generate_phase_pattern(batch_size, seq_len):
            """Generate sequences with complex phase relationships."""
            sequences = torch.zeros(batch_size, seq_len, config.hidden_size)

            for b in range(batch_size):
                phase = (b * 0.7) % (2 * np.pi)
                for i in range(seq_len):
                    t = i / seq_len * 4 * np.pi
                    # Complex wave with phase relationships
                    wave = np.sin(t + phase) + 0.5 * np.cos(1.5 * t + phase * 2)
                    sequences[b, i, :] = wave

            return sequences

        # Test THC can process phase patterns
        batch_size, seq_len = 4, 16
        test_input = generate_phase_pattern(batch_size, seq_len)

        output = thc(test_input)
        assert output.shape == test_input.shape

        # Check phase statistics show meaningful patterns
        stats = thc.get_phase_statistics(test_input)
        assert stats["phase_diff_std"] > 0  # Should have phase variations

    @pytest.mark.skipif(
        True, reason="THC integration not yet implemented in TransformerBlock"
    )
    def test_phase_pattern_learning(self):
        """Test THC's ability to learn complex phase patterns."""
        # Create models with and without THC
        config = PraxisConfig(
            hidden_size=128, num_heads=4, depth=2, dropout=0.1, vocab_size=50
        )

        model_with_thc = TransformerBlock(config)

        # Create a model without THC for comparison
        model_without_thc = TransformerBlock(config)
        model_without_thc.use_thc = False

        def generate_phase_pattern(batch_size, seq_len):
            """Generate sequences with complex phase relationships."""
            sequences = torch.zeros(batch_size, seq_len, config.hidden_size)

            for b in range(batch_size):
                phase = (b * 0.7) % (2 * np.pi)
                for i in range(seq_len):
                    t = i / seq_len * 4 * np.pi
                    # Complex wave with phase relationships
                    wave = np.sin(t + phase) + 0.5 * np.cos(1.5 * t + phase * 2)
                    sequences[b, i, :] = wave

            return sequences

        # Test both models can process the input
        batch_size, seq_len = 4, 16
        test_input = generate_phase_pattern(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        # Both should work without errors
        output_with, _, _, _ = model_with_thc(test_input, attention_mask)
        output_without, _, _, _ = model_without_thc(test_input, attention_mask)

        assert output_with.shape == test_input.shape
        assert output_without.shape == test_input.shape

        # Outputs should be different
        assert not torch.allclose(output_with, output_without, atol=1e-5)

    def test_thc_sequence_length_flexibility(self):
        """Test that THC handles different sequence lengths."""
        config = PraxisConfig(
            hidden_size=64, num_heads=2, depth=1, dropout=0.0, vocab_size=30
        )

        thc = TemporalHealthComplex(d_model=config.hidden_size)
        thc.eval()

        # Test with different sequence lengths
        for seq_len in [8, 16, 32, 64]:
            batch_size = 2
            inputs = torch.randn(batch_size, seq_len, config.hidden_size)

            with torch.no_grad():
                output = thc(inputs)

            assert output.shape == inputs.shape
            assert torch.isfinite(output).all()

    @pytest.mark.skipif(
        True, reason="THC integration not yet implemented in TransformerBlock"
    )
    def test_sequence_length_generalization(self):
        """Test that THC helps with sequence length generalization."""
        config = PraxisConfig(
            hidden_size=64, num_heads=2, depth=1, dropout=0.0, vocab_size=30
        )

        thc_model = TransformerBlock(config)
        thc_model.eval()

        # Test with different sequence lengths
        for seq_len in [8, 16, 32, 64]:
            batch_size = 2
            inputs = torch.randn(batch_size, seq_len, config.hidden_size)
            attention_mask = torch.ones(batch_size, seq_len)

            with torch.no_grad():
                output, _, _, _ = thc_model(inputs, attention_mask)

            assert output.shape == inputs.shape
            assert torch.isfinite(output).all()

    def test_thc_complex_statistics_evolution(self):
        """Test that THC's complex statistics evolve during training."""
        thc = TemporalHealthComplex(d_model=128)

        # Get initial statistics
        test_input = torch.randn(4, 16, 128)
        initial_stats = thc.get_phase_statistics(test_input)

        # Do some training steps
        optimizer = torch.optim.Adam(thc.parameters(), lr=1e-3)
        thc.train()

        for _ in range(10):
            inputs = torch.randn(4, 16, 128)
            output = thc(inputs)
            loss = output.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Get final statistics
        thc.eval()
        with torch.no_grad():
            final_stats = thc.get_phase_statistics(test_input)

        # Statistics should have evolved (not necessarily better, just different)
        stats_changed = any(
            abs(initial_stats[key] - final_stats[key]) > 1e-6
            for key in initial_stats.keys()
        )
        assert stats_changed, "THC statistics did not evolve during training"


class TestTHCEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_model(self):
        """Test THC with very small model dimensions."""
        thc = TemporalHealthComplex(d_model=8, reduction_factor=16)
        assert thc.d_complex == 1  # Should not be zero

        x = torch.randn(2, 4, 8)
        output = thc(x)
        assert output.shape == x.shape

    def test_very_large_reduction_factor(self):
        """Test THC with very large reduction factor."""
        thc = TemporalHealthComplex(d_model=128, reduction_factor=256)
        assert thc.d_complex == 1

        x = torch.randn(1, 8, 128)
        output = thc(x)
        assert output.shape == x.shape

    def test_single_token_sequence(self):
        """Test THC with single token sequences."""
        thc = TemporalHealthComplex(d_model=64)
        x = torch.randn(3, 1, 64)  # Single token sequences

        output = thc(x)
        assert output.shape == x.shape

    def test_large_kernel_size(self):
        """Test THC with kernel size larger than sequence."""
        thc = TemporalHealthComplex(d_model=64, kernel_size=15)
        x = torch.randn(2, 8, 64)  # Sequence shorter than kernel

        output = thc(x)
        assert output.shape == x.shape

    def test_complex_overflow_protection(self):
        """Test that complex operations don't overflow."""
        thc = TemporalHealthComplex(d_model=128)

        # Very large input values
        x = torch.randn(2, 16, 128) * 100
        output = thc(x)

        assert torch.isfinite(output).all()
        assert output.shape == x.shape
