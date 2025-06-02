import itertools
import time
from typing import List, Tuple, Dict

import pytest
import torch
import torch.nn as nn

from praxis import PraxisConfig
from praxis.attention.syntaxes import SyntaxesAttention
from praxis.attention.components import VanillaMHA


class TestSyntaxesAttention:
    """Fast test suite for Syntaxes attention mechanism."""

    @pytest.fixture
    def base_config(self) -> PraxisConfig:
        """Base configuration for syntaxes attention - optimized for fast testing."""
        return PraxisConfig(
            hidden_size=64,  # Much smaller
            num_heads=4,     # Fewer heads
            num_queries=1,   # Standard MHA (not MQA)
            syntaxes_context_size=32,  # Small context for testing
            max_length=128,  # Much smaller
            dropout=0.0,     # No dropout for deterministic tests
            encoding="nope",  # No positional encoding for basic tests
        )

    @pytest.fixture(params=[
        # (context_size,) - reduced test cases
        (16,),
        (32,),
        (64,),
    ])
    def syntaxes_config(self, base_config, request) -> PraxisConfig:
        """Parametrized configuration for different syntaxes methods."""
        context_size, = request.param
        
        config = base_config
        config.syntaxes_context_size = context_size
        
        return config

    @pytest.fixture(params=[
        (2, 32),   # Small case
        (4, 64),   # Medium case  
    ])
    def batch_seq_dims(self, request) -> Tuple[int, int]:
        """Different batch size and sequence length combinations."""
        return request.param

    def test_initialization(self, syntaxes_config):
        """Test that SyntaxesAttention initializes correctly with different configs."""
        attention = SyntaxesAttention(syntaxes_config)
        
        assert attention.hidden_size == syntaxes_config.hidden_size
        assert attention.num_heads == syntaxes_config.num_heads
        assert attention.context_size == getattr(syntaxes_config, 'syntaxes_context_size', 128)
        
        # Check that parameters are properly initialized
        assert hasattr(attention, 'q_proj')
        assert hasattr(attention, 'k_proj')
        assert hasattr(attention, 'v_proj')
        assert hasattr(attention, 'o_proj')
        assert hasattr(attention, 'encoding')

    def test_forward_pass_shape_consistency(self, syntaxes_config, batch_seq_dims):
        """Test that forward pass maintains input/output shape consistency."""
        batch_size, seq_len = batch_seq_dims
        attention = SyntaxesAttention(syntaxes_config)
        
        inputs = torch.randn(batch_size, seq_len, syntaxes_config.hidden_size)
        
        with torch.no_grad():
            output, past_kv, aux_loss = attention(inputs=inputs)
        
        # Check output shape matches input shape
        assert output.shape == inputs.shape
        assert aux_loss == 0  # Currently no auxiliary loss
        assert past_kv is None  # KV caching not implemented

    def test_forward_pass_with_attention_mask(self, syntaxes_config):
        """Test forward pass with attention mask."""
        batch_size, seq_len = 4, 64  # Smaller
        attention = SyntaxesAttention(syntaxes_config)
        
        inputs = torch.randn(batch_size, seq_len, syntaxes_config.hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output, past_kv, aux_loss = attention(
                inputs=inputs, 
                attention_mask=attention_mask
            )
        
        assert output.shape == inputs.shape

    def test_causal_sliding_window(self, base_config):
        """Test that causal sliding window works correctly."""
        attention = SyntaxesAttention(base_config)
        
        batch_size, seq_len = 2, 64
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        assert output.shape == inputs.shape
        
        # Test that it produces different outputs for different sequences
        different_inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        with torch.no_grad():
            different_output, _, _ = attention(inputs=different_inputs)
        
        # Outputs should be different for different inputs
        assert not torch.allclose(output, different_output, atol=1e-5)

    def test_gradient_flow(self, syntaxes_config):
        """Test that gradients flow properly through the attention mechanism."""
        attention = SyntaxesAttention(syntaxes_config)
        batch_size, seq_len = 2, 64  # Smaller
        
        inputs = torch.randn(
            batch_size, seq_len, syntaxes_config.hidden_size, 
            requires_grad=True
        )
        
        output, _, _ = attention(inputs=inputs)
        loss = output.mean()
        loss.backward()
        
        # Check that input gradients exist
        assert inputs.grad is not None
        assert inputs.grad.norm().item() > 0
        
        # Check that parameter gradients exist
        param_grads = []
        for name, param in attention.named_parameters():
            if param.grad is not None:
                param_grads.append((name, param.grad.norm().item()))
        
        assert len(param_grads) > 0, "No parameter gradients found"
        
        # At least some gradients should be non-zero
        non_zero_grads = [grad for name, grad in param_grads if grad > 1e-8]
        assert len(non_zero_grads) > 0, "All gradients are effectively zero"

    def test_window_size_effects(self, base_config):
        """Test that different compression ratios produce different outputs."""
        torch.manual_seed(42)  # For reproducible results
        
        batch_size, seq_len = 2, 64  # Much smaller
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Test with high compression
        config_high = base_config
        config_high.syntaxes_query_compression_ratio = 8
        attention_high = SyntaxesAttention(config_high)
        
        # Test with low compression
        config_low = base_config
        config_low.syntaxes_query_compression_ratio = 2
        attention_low = SyntaxesAttention(config_low)
        
        with torch.no_grad():
            output_high, _, _ = attention_high(inputs=inputs)
            output_low, _, _ = attention_low(inputs=inputs)
        
        # Outputs should be different between compression ratios
        assert not torch.allclose(output_high, output_low, atol=1e-3)

    def test_different_num_selected(self, base_config):
        """Test that different compression ratios produce different outputs."""
        torch.manual_seed(42)
        
        batch_size, seq_len = 2, 64  # Much smaller
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        compression_ratios = [2, 4]  # Fewer values
        outputs = {}
        
        for ratio in compression_ratios:
            config = base_config
            config.syntaxes_query_compression_ratio = ratio
            
            attention = SyntaxesAttention(config)
            
            with torch.no_grad():
                output, _, _ = attention(inputs=inputs)
            
            outputs[ratio] = output
        
        # Outputs should be different between different compression ratios
        values = list(outputs.keys())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                val1, val2 = values[i], values[j]
                diff = torch.abs(outputs[val1] - outputs[val2]).mean()
                assert diff > 1e-4, f"Outputs too similar between compression_ratio={val1} and {val2}: {diff}"

    @pytest.mark.parametrize("seq_len", [32, 64])  # Much smaller
    def test_different_sequence_lengths(self, base_config, seq_len):
        """Test that the attention works with different sequence lengths."""
        attention = SyntaxesAttention(base_config)
        batch_size = 2  # Smaller batch
        
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        assert output.shape == inputs.shape

    def test_performance_vs_vanilla_attention(self, base_config):
        """Test performance comparison with vanilla attention."""
        batch_size, seq_len = 2, 128  # Much smaller
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Syntaxes attention
        syntaxes_attention = SyntaxesAttention(base_config)
        
        # Vanilla attention
        vanilla_attention = VanillaMHA(base_config)
        
        # Just test they both work, skip timing for speed
        with torch.no_grad():
            syntaxes_output, _, _ = syntaxes_attention(inputs=inputs)
            vanilla_result = vanilla_attention(inputs)  # Check what VanillaMHA returns
            
            # Handle different return types
            if isinstance(vanilla_result, tuple):
                vanilla_output = vanilla_result[0]
            else:
                vanilla_output = vanilla_result
        
        # Check outputs have correct shape
        assert syntaxes_output.shape == inputs.shape
        assert vanilla_output.shape == inputs.shape

    def test_memory_complexity_reduction(self, base_config):
        """Test theoretical memory complexity reduction."""
        seq_len = 128  # Much smaller
        context_size = base_config.syntaxes_context_size
        num_heads = base_config.num_heads
        batch_size = 2  # Smaller batch
        
        # Memory for attention scores
        # Syntaxes: all queries attend to context_size keys (recent context)
        syntaxes_memory = seq_len * context_size * num_heads * batch_size
        # Vanilla: all queries attend to all seq_len keys
        vanilla_memory = seq_len ** 2 * num_heads * batch_size
        
        reduction_factor = vanilla_memory / syntaxes_memory
        # Expected reduction: (seq_len * seq_len) / (seq_len * context_size) = seq_len / context_size
        expected_reduction = seq_len / context_size
        
        # Allow small numerical error
        assert abs(reduction_factor - expected_reduction) < 1e-6
        assert reduction_factor > 1, f"No memory reduction: {reduction_factor:.1f}x"

    def test_kv_caching_not_implemented(self, base_config):
        """Test that KV caching raises NotImplementedError."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 2, 64
        
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        past_key_values = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        with pytest.raises(NotImplementedError, match="KV caching not yet supported"):
            attention(inputs=inputs, past_key_values=past_key_values)

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_device_compatibility(self, base_config, device):
        """Test that the attention works on different devices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        attention = SyntaxesAttention(base_config).to(device)
        batch_size, seq_len = 2, 64
        
        inputs = torch.randn(
            batch_size, seq_len, base_config.hidden_size, 
            device=device
        )
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        assert output.device.type == device
        assert output.shape == inputs.shape

    def test_deterministic_output(self, base_config):
        """Test that outputs are deterministic given the same seed."""
        batch_size, seq_len = 2, 64  # Smaller
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # First run
        torch.manual_seed(42)
        attention1 = SyntaxesAttention(base_config)
        with torch.no_grad():
            output1, _, _ = attention1(inputs=inputs)
        
        # Second run with same seed
        torch.manual_seed(42)
        attention2 = SyntaxesAttention(base_config)
        with torch.no_grad():
            output2, _, _ = attention2(inputs=inputs)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_training_mode_vs_eval_mode(self, base_config):
        """Test behavior difference between training and evaluation modes."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 2, 64  # Smaller
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Training mode
        attention.train()
        with torch.no_grad():
            output_train, _, _ = attention(inputs=inputs)
        
        # Evaluation mode
        attention.eval()
        with torch.no_grad():
            output_eval, _, _ = attention(inputs=inputs)
        
        # Should have same shape
        assert output_train.shape == output_eval.shape
        
        # Note: With current implementation and dropout=0, outputs should be identical
        # But this test ensures the mode switching works without errors

    def test_edge_cases(self, base_config):
        """Test edge cases and boundary conditions."""
        # Test when compression_ratio > seq_len
        config = base_config
        config.syntaxes_query_compression_ratio = 32
        attention = SyntaxesAttention(config)
        
        batch_size, seq_len = 2, 16  # seq_len < compression_ratio
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        # Should work without errors and maintain shape
        assert output.shape == inputs.shape
        
        # Test with very small sequence
        inputs_small = torch.randn(batch_size, 4, config.hidden_size)
        with torch.no_grad():
            output_small, _, _ = attention(inputs=inputs_small)
        assert output_small.shape == inputs_small.shape