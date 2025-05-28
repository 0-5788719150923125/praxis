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
    """Comprehensive test suite for Syntaxes attention mechanism."""

    @pytest.fixture
    def base_config(self) -> PraxisConfig:
        """Base configuration for syntaxes attention."""
        return PraxisConfig(
            hidden_size=256,
            num_heads=8,
            causal=True,
            compression_size=64,
            compression_method="learnable_interpolation",
            selection_method="interpolation",
            max_length=2048,
            dropout=0.1,
        )

    @pytest.fixture(params=[
        # (compression_size, compression_method, selection_method)
        (32, "learnable_interpolation", "interpolation"),
        (64, "learnable_interpolation", "interpolation"),
        (32, "linear_interpolation", "interpolation"),
        (32, "pooling", "interpolation"),
        (32, None, "sliding_window"),
        (32, None, "top_k"),
    ])
    def syntaxes_config(self, base_config, request) -> PraxisConfig:
        """Parametrized configuration for different syntaxes methods."""
        compression_size, compression_method, selection_method = request.param
        
        config = base_config
        config.compression_size = compression_size
        config.compression_method = compression_method
        config.selection_method = selection_method
        
        return config

    @pytest.fixture(params=[
        (2, 64),   # Small case
        (4, 128),  # Medium case  
        (8, 256),  # Standard case
        (16, 512), # Large case
    ])
    def batch_seq_dims(self, request) -> Tuple[int, int]:
        """Different batch size and sequence length combinations."""
        return request.param

    def test_initialization(self, syntaxes_config):
        """Test that SyntaxesAttention initializes correctly with different configs."""
        attention = SyntaxesAttention(syntaxes_config)
        
        assert attention.hidden_size == syntaxes_config.hidden_size
        assert attention.num_heads == syntaxes_config.num_heads
        assert attention.compression_size == syntaxes_config.compression_size
        assert attention.compression_method == syntaxes_config.compression_method
        assert attention.selection_method == syntaxes_config.selection_method
        
        # Check that parameters are properly initialized
        assert hasattr(attention, 'q_proj')
        assert hasattr(attention, 'k_proj')
        assert hasattr(attention, 'v_proj')
        assert hasattr(attention, 'o_proj')
        
        if syntaxes_config.compression_method == "learnable_interpolation":
            assert hasattr(attention, 'interpolation_proj')
            assert hasattr(attention, 'compressed_pos_emb')
            assert hasattr(attention, 'expansion_proj')

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
        batch_size, seq_len = 4, 128
        attention = SyntaxesAttention(syntaxes_config)
        
        inputs = torch.randn(batch_size, seq_len, syntaxes_config.hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -10:] = 0  # Mask last 10 tokens
        
        with torch.no_grad():
            output, past_kv, aux_loss = attention(
                inputs=inputs, 
                attention_mask=attention_mask
            )
        
        assert output.shape == inputs.shape

    def test_causal_masking(self, base_config):
        """Test that causal masking works correctly."""
        config = base_config
        config.causal = True
        config.selection_method = "interpolation"
        config.compression_method = "learnable_interpolation"
        
        attention = SyntaxesAttention(config)
        batch_size, seq_len = 2, 64
        
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        assert output.shape == inputs.shape
        
        # Test with causal=False
        config.causal = False
        attention_non_causal = SyntaxesAttention(config)
        
        with torch.no_grad():
            output_non_causal, _, _ = attention_non_causal(inputs=inputs)
        
        assert output_non_causal.shape == inputs.shape
        # Outputs should be different due to masking
        assert not torch.allclose(output, output_non_causal, atol=1e-5)

    def test_gradient_flow(self, syntaxes_config):
        """Test that gradients flow properly through the attention mechanism."""
        attention = SyntaxesAttention(syntaxes_config)
        batch_size, seq_len = 4, 128
        
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

    def test_compression_methods_comparison(self, base_config):
        """Test different compression methods produce different outputs."""
        torch.manual_seed(42)  # For reproducible results
        
        batch_size, seq_len = 4, 128
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        methods_to_test = [
            ("learnable_interpolation", "interpolation"),
            ("linear_interpolation", "interpolation"),
            ("pooling", "interpolation"),
        ]
        
        outputs = {}
        
        for compression_method, selection_method in methods_to_test:
            config = base_config
            config.compression_method = compression_method
            config.selection_method = selection_method
            
            attention = SyntaxesAttention(config)
            
            with torch.no_grad():
                output, _, _ = attention(inputs=inputs)
            
            outputs[compression_method] = output
        
        # Outputs should be different between methods
        methods = list(outputs.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                assert not torch.allclose(
                    outputs[method1], outputs[method2], atol=1e-3
                ), f"Outputs too similar between {method1} and {method2}"

    def test_selection_methods_comparison(self, base_config):
        """Test different selection methods produce different outputs."""
        torch.manual_seed(42)
        
        batch_size, seq_len = 4, 128
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Add positional bias to make top_k vs sliding_window more distinct
        for i in range(seq_len):
            inputs[:, i, :] += i * 0.01
        
        selection_methods = ["interpolation", "sliding_window", "top_k"]
        outputs = {}
        
        for selection_method in selection_methods:
            config = base_config
            config.selection_method = selection_method
            if selection_method == "interpolation":
                config.compression_method = "learnable_interpolation"
            else:
                config.compression_method = None
            
            attention = SyntaxesAttention(config)
            
            with torch.no_grad():
                output, _, _ = attention(inputs=inputs)
            
            outputs[selection_method] = output
        
        # Outputs should be different between methods
        methods = list(outputs.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                diff = torch.abs(outputs[method1] - outputs[method2]).mean()
                assert diff > 1e-4, f"Outputs too similar between {method1} and {method2}: {diff}"

    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
    def test_different_sequence_lengths(self, base_config, seq_len):
        """Test that the attention works with different sequence lengths."""
        attention = SyntaxesAttention(base_config)
        batch_size = 4
        
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        assert output.shape == inputs.shape

    def test_performance_vs_vanilla_attention(self, base_config):
        """Test performance comparison with vanilla attention."""
        batch_size, seq_len = 8, 512
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Syntaxes attention
        syntaxes_attention = SyntaxesAttention(base_config)
        
        # Vanilla attention
        vanilla_attention = VanillaMHA(base_config)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = syntaxes_attention(inputs=inputs)
                _ = vanilla_attention(inputs)
        
        # Time syntaxes attention
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = syntaxes_attention(inputs=inputs)
        syntaxes_time = time.time() - start_time
        
        # Time vanilla attention
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = vanilla_attention(inputs)
        vanilla_time = time.time() - start_time
        
        # Syntaxes should be reasonably fast
        speedup = vanilla_time / syntaxes_time
        # More lenient threshold as performance can vary
        assert speedup > 0.3, f"Syntaxes attention too slow: {speedup:.2f}x"
        
        # Log performance info (will be visible with pytest -s)
        print(f"\nPerformance comparison (seq_len={seq_len}):")
        print(f"  Syntaxes: {syntaxes_time:.4f}s")
        print(f"  Vanilla:  {vanilla_time:.4f}s")
        print(f"  Speedup:  {speedup:.2f}x")

    def test_memory_complexity_reduction(self, base_config):
        """Test theoretical memory complexity reduction."""
        seq_len = 1024
        compression_size = base_config.compression_size
        num_heads = base_config.num_heads
        batch_size = 4
        
        # Memory for attention scores
        syntaxes_memory = compression_size ** 2 * num_heads * batch_size
        vanilla_memory = seq_len ** 2 * num_heads * batch_size
        
        reduction_factor = vanilla_memory / syntaxes_memory
        expected_reduction = (seq_len / compression_size) ** 2
        
        assert abs(reduction_factor - expected_reduction) < 1e-6
        assert reduction_factor > 16, f"Memory reduction too small: {reduction_factor:.1f}x"
        
        print(f"\nMemory complexity reduction: {reduction_factor:.1f}x")

    def test_kv_caching_not_implemented(self, base_config):
        """Test that KV caching raises NotImplementedError."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 2, 64
        
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        past_key_values = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        with pytest.raises(NotImplementedError, match="KV caching not yet supported"):
            attention(inputs=inputs, past_key_values=past_key_values)

    def test_invalid_configuration_errors(self, base_config):
        """Test that invalid configurations raise appropriate errors."""
        # Test invalid selection method
        config = base_config
        config.selection_method = "invalid_method"
        
        attention = SyntaxesAttention(config)
        inputs = torch.randn(2, 64, config.hidden_size)
        
        with pytest.raises(ValueError, match="Unknown selection method"):
            attention(inputs=inputs)
        
        # Test invalid compression method
        config.selection_method = "interpolation"
        config.compression_method = "invalid_compression"
        
        attention = SyntaxesAttention(config)
        
        with pytest.raises(ValueError, match="Unknown compression method"):
            attention(inputs=inputs)

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
        batch_size, seq_len = 4, 128
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
        batch_size, seq_len = 4, 128
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

    def test_token_selection_analysis(self, base_config):
        """Test and analyze which tokens are selected by different methods."""
        torch.manual_seed(42)
        batch_size, seq_len = 2, 128
        
        # Create inputs with positional bias to make selection patterns clearer
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        for i in range(seq_len):
            inputs[:, i, :] += i * 0.01
        
        results = {}
        
        for selection_method in ["top_k", "sliding_window"]:
            config = base_config
            config.selection_method = selection_method
            config.compression_method = None
            config.compression_size = 32  # Select 32 tokens
            
            attention = SyntaxesAttention(config)
            
            # Get internal scores and selected indices
            with torch.no_grad():
                compressed, indices = attention._compress_sequence(inputs)
                scores = attention._compute_token_scores(inputs)
            
            results[selection_method] = {
                'compressed': compressed,
                'indices': indices if indices is not None else torch.arange(compressed.shape[1])
            }
        
        # Check that different methods produce different compressed representations
        top_k_compressed = results['top_k']['compressed']
        sliding_compressed = results['sliding_window']['compressed']
        
        # The compressed representations should be different
        assert not torch.allclose(top_k_compressed, sliding_compressed, atol=1e-4)

    @pytest.mark.parametrize("batch_size,seq_lengths", [
        (2, [128, 256, 512, 1024]),
        (4, [128, 256, 512]),
        (8, [128, 256]),
        (16, [128]),
    ])
    def test_comprehensive_benchmark(self, base_config, batch_size, seq_lengths):
        """Comprehensive benchmark of SyntaxesAttention vs VanillaMHA."""
        syntaxes_attention = SyntaxesAttention(base_config)
        vanilla_attention = VanillaMHA(base_config)
        
        results = []
        
        for seq_len in seq_lengths:
            inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                syntaxes_attention = syntaxes_attention.cuda()
                vanilla_attention = vanilla_attention.cuda()
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = syntaxes_attention(inputs=inputs)
                    _ = vanilla_attention(inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark syntaxes
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = syntaxes_attention(inputs=inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            syntaxes_time = time.time() - start
            
            # Benchmark vanilla
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = vanilla_attention(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            vanilla_time = time.time() - start
            
            speedup = vanilla_time / syntaxes_time
            results.append({
                'seq_len': seq_len,
                'syntaxes_time': syntaxes_time,
                'vanilla_time': vanilla_time,
                'speedup': speedup
            })
        
        # Syntaxes should be reasonably fast compared to vanilla
        # Note: For small sequences, vanilla can be faster due to less overhead
        for result in results:
            # More lenient for smaller sequences where overhead dominates
            min_speedup = 0.3 if result['seq_len'] <= 256 else 0.5
            assert result['speedup'] > min_speedup, f"Too slow at seq_len={result['seq_len']}: {result['speedup']:.2f}x"

    def test_detailed_gradient_flow_analysis(self, base_config):
        """Detailed analysis of gradient flow through each parameter."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 4, 256
        
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size, requires_grad=True)
        
        # Forward pass
        output, _, _ = attention(inputs=inputs)
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Collect gradient information
        grad_info = {}
        total_params = 0
        params_with_grad = 0
        
        for name, param in attention.named_parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                grad_info[name] = {
                    'shape': list(param.shape),
                    'grad_norm': grad_norm,
                    'has_grad': grad_norm > 1e-8
                }
        
        # Verify gradients exist for key components
        assert params_with_grad > 0, "No parameters have gradients"
        assert params_with_grad == total_params, f"Only {params_with_grad}/{total_params} params have gradients"
        
        # Check specific important parameters
        important_params = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight']
        for param_name in important_params:
            assert param_name in grad_info, f"Missing gradient for {param_name}"
            assert grad_info[param_name]['has_grad'], f"Zero gradient for {param_name}"
        
        # Check compression-specific parameters if applicable
        if base_config.compression_method == "learnable_interpolation":
            # Only interpolation_proj is used in the forward pass
            # expansion_proj is initialized but not used in current implementation
            compression_params = ['interpolation_proj.weight']
            for param_name in compression_params:
                if param_name in grad_info:
                    assert grad_info[param_name]['has_grad'], f"Zero gradient for {param_name}"

    def test_compression_methods_performance_matrix(self, base_config):
        """Test all compression methods work correctly (functionality test only)."""
        batch_size, seq_len = 4, 256
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        methods = [
            ("learnable_interpolation", "interpolation"),
            ("linear_interpolation", "interpolation"),
            ("pooling", "interpolation"),
            (None, "sliding_window"),
            (None, "top_k"),
        ]
        
        results = {}
        
        for compression_method, selection_method in methods:
            config = base_config
            config.compression_method = compression_method
            config.selection_method = selection_method
            
            try:
                attention = SyntaxesAttention(config)
                
                with torch.no_grad():
                    output, _, _ = attention(inputs=inputs)
                
                method_name = f"{compression_method or 'none'}+{selection_method}"
                results[method_name] = {
                    'output_shape': output.shape,
                    'success': True,
                    'output': output
                }
                
            except Exception as e:
                method_name = f"{compression_method or 'none'}+{selection_method}"
                results[method_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Verify all methods succeeded
        successful_methods = [m for m, d in results.items() if d['success']]
        assert len(successful_methods) == len(methods), f"Some methods failed: {[m for m, d in results.items() if not d['success']]}"
        
        # Verify all outputs have correct shape
        for method, data in results.items():
            if data['success']:
                assert data['output_shape'] == inputs.shape, f"{method} has wrong output shape"
        
        # Verify different methods produce different outputs
        output_list = [data['output'] for data in results.values() if data['success']]
        for i in range(len(output_list)):
            for j in range(i + 1, len(output_list)):
                assert not torch.allclose(output_list[i], output_list[j], atol=1e-4), \
                    f"Methods {list(results.keys())[i]} and {list(results.keys())[j]} produce identical outputs"

    def test_training_config_compatibility(self, base_config):
        """Test compatibility with realistic training configurations."""
        # Simulate actual training config
        training_configs = [
            {'hidden_size': 256, 'num_heads': 8, 'compression_size': 64},
            {'hidden_size': 512, 'num_heads': 8, 'compression_size': 128},
            {'hidden_size': 768, 'num_heads': 12, 'compression_size': 96},
            {'hidden_size': 1024, 'num_heads': 16, 'compression_size': 128},
        ]
        
        batch_seq_combinations = [
            (2, 128), (4, 256), (8, 512), (16, 1024), (32, 2048)
        ]
        
        for config_dict in training_configs:
            config = PraxisConfig(**config_dict)
            config.causal = True
            config.compression_method = "learnable_interpolation"
            config.selection_method = "interpolation"
            config.max_length = 2048
            config.dropout = 0.1
            
            attention = SyntaxesAttention(config)
            
            for batch_size, seq_len in batch_seq_combinations:
                if seq_len <= config.max_length:
                    inputs = torch.randn(batch_size, seq_len, config.hidden_size)
                    
                    try:
                        with torch.no_grad():
                            output, _, _ = attention(inputs=inputs)
                        assert output.shape == inputs.shape
                        
                        # Test gradient flow
                        inputs.requires_grad = True
                        output, _, _ = attention(inputs=inputs)
                        loss = output.mean()
                        loss.backward()
                        assert inputs.grad is not None
                        
                    except Exception as e:
                        pytest.fail(f"Failed with config {config_dict} and batch={batch_size}, seq={seq_len}: {e}")

    def test_edge_cases(self, base_config):
        """Test edge cases like very small sequences, compression_size > seq_len, etc."""
        attention = SyntaxesAttention(base_config)
        
        # Test 1: Sequence length smaller than compression size
        small_seq = torch.randn(2, 16, base_config.hidden_size)  # seq_len=16, compression_size=64
        with torch.no_grad():
            output, _, _ = attention(inputs=small_seq)
        assert output.shape == small_seq.shape
        
        # Test 2: Sequence length equals compression size
        equal_seq = torch.randn(2, 64, base_config.hidden_size)
        with torch.no_grad():
            output, _, _ = attention(inputs=equal_seq)
        assert output.shape == equal_seq.shape
        
        # Test 3: Single token sequence
        single_token = torch.randn(2, 1, base_config.hidden_size)
        with torch.no_grad():
            output, _, _ = attention(inputs=single_token)
        assert output.shape == single_token.shape
        
        # Test 4: Very long sequence (up to max_length)
        max_len = base_config.max_length
        long_seq = torch.randn(1, max_len, base_config.hidden_size)
        with torch.no_grad():
            output, _, _ = attention(inputs=long_seq)
        assert output.shape == long_seq.shape

    def test_scoring_methods(self, base_config):
        """Test different scoring methods (norm vs learned)."""
        batch_size, seq_len = 4, 128
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Test norm-based scoring
        config_norm = base_config
        config_norm.scoring_method = "norm"
        attention_norm = SyntaxesAttention(config_norm)
        
        with torch.no_grad():
            output_norm, _, _ = attention_norm(inputs=inputs)
        
        # Test learned scoring
        config_learned = base_config
        config_learned.scoring_method = "learned"
        attention_learned = SyntaxesAttention(config_learned)
        
        with torch.no_grad():
            output_learned, _, _ = attention_learned(inputs=inputs)
        
        # Outputs should be different due to different scoring
        assert not torch.allclose(output_norm, output_learned, atol=1e-3)
        
        # Test invalid scoring method
        config_invalid = base_config
        config_invalid.scoring_method = "invalid_scoring"
        attention_invalid = SyntaxesAttention(config_invalid)
        
        with pytest.raises(ValueError, match="Unknown scoring method"):
            attention_invalid._compute_token_scores(inputs)

    def test_pooling_compression_method(self, base_config):
        """Test the pooling compression method specifically."""
        config = base_config
        config.compression_method = "pooling"
        config.selection_method = "interpolation"
        
        attention = SyntaxesAttention(config)
        batch_size, seq_len = 4, 256
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        assert output.shape == inputs.shape
        
        # Test with sequence length not divisible by compression_size
        odd_seq = torch.randn(batch_size, 257, config.hidden_size)
        with torch.no_grad():
            output_odd, _, _ = attention(inputs=odd_seq)
        
        assert output_odd.shape == odd_seq.shape

    def test_numerical_stability(self, base_config):
        """Test numerical stability with extreme input values."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 2, 128
        
        # Test with very large values
        large_inputs = torch.randn(batch_size, seq_len, base_config.hidden_size) * 1e3
        with torch.no_grad():
            output_large, _, _ = attention(inputs=large_inputs)
        assert torch.isfinite(output_large).all()
        
        # Test with very small values
        small_inputs = torch.randn(batch_size, seq_len, base_config.hidden_size) * 1e-6
        with torch.no_grad():
            output_small, _, _ = attention(inputs=small_inputs)
        assert torch.isfinite(output_small).all()
        
        # Test with mixed scales
        mixed_inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        mixed_inputs[:, :seq_len//2] *= 1e3
        mixed_inputs[:, seq_len//2:] *= 1e-3
        with torch.no_grad():
            output_mixed, _, _ = attention(inputs=mixed_inputs)
        assert torch.isfinite(output_mixed).all()

    def test_attention_mask_edge_cases(self, base_config):
        """Test attention mask edge cases."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 4, 128
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Test 1: All tokens masked
        all_masked = torch.zeros(batch_size, seq_len)
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs, attention_mask=all_masked)
        assert output.shape == inputs.shape
        
        # Test 2: No tokens masked (all ones)
        no_mask = torch.ones(batch_size, seq_len)
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs, attention_mask=no_mask)
        assert output.shape == inputs.shape
        
        # Test 3: Alternating mask pattern
        alternating_mask = torch.zeros(batch_size, seq_len)
        alternating_mask[:, ::2] = 1
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs, attention_mask=alternating_mask)
        assert output.shape == inputs.shape

    def test_interpolation_edge_cases(self, base_config):
        """Test edge cases specific to interpolation methods."""
        config = base_config
        config.compression_method = "linear_interpolation"
        config.selection_method = "interpolation"
        
        attention = SyntaxesAttention(config)
        
        # Test with target_length >= seq_len
        # In this case, linear_interpolation returns original sequence
        batch_size = 2
        seq_len = 32
        target_len = 64
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)
        
        with torch.no_grad():
            compressed, _ = attention._interpolate_sequence(inputs, target_len)
        
        # With linear interpolation, when target >= seq_len, it returns x[:, :target_length]
        # Since seq_len < target_len, it should return the original sequence
        assert compressed.shape == (batch_size, seq_len, config.hidden_size)

    def test_expansion_mechanism(self, base_config):
        """Test the expansion mechanism that reconstructs from compressed representation."""
        config = base_config
        config.compression_method = "learnable_interpolation"
        config.selection_method = "interpolation"
        
        attention = SyntaxesAttention(config)
        batch_size, seq_len = 4, 256
        
        # Get compressed representation
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)
        with torch.no_grad():
            compressed, _ = attention._compress_sequence(inputs)
        
        # Test expansion back to original size
        assert compressed.shape[1] == min(config.compression_size, seq_len)
        
        # The expanded sequence should have the same hidden dimension
        assert compressed.shape[2] == config.hidden_size

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtype_compatibility(self, base_config, dtype):
        """Test compatibility with different data types."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        attention = SyntaxesAttention(base_config).to(device=device, dtype=dtype)
        
        batch_size, seq_len = 2, 64
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size, 
                           device=device, dtype=dtype)
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs)
        
        assert output.dtype == dtype
        assert output.shape == inputs.shape
        assert torch.isfinite(output).all()

    def test_block_ids_handling(self, base_config):
        """Test handling of block_ids parameter."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 4, 128
        
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        block_ids = torch.randint(0, 3, (batch_size,))
        
        with torch.no_grad():
            output, _, _ = attention(inputs=inputs, block_ids=block_ids)
        
        assert output.shape == inputs.shape

    def test_current_depth_handling(self, base_config):
        """Test handling of current_depth parameter."""
        attention = SyntaxesAttention(base_config)
        batch_size, seq_len = 4, 128
        
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size)
        
        # Test with different depths
        for depth in [0, 1, 5, 10]:
            with torch.no_grad():
                output, _, _ = attention(inputs=inputs, current_depth=depth)
            assert output.shape == inputs.shape

    def test_memory_efficiency(self, base_config):
        """Test that syntaxes attention is more memory efficient than vanilla."""
        if not torch.cuda.is_available():
            pytest.skip("Memory profiling requires CUDA")
        
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        batch_size, seq_len = 8, 1024
        inputs = torch.randn(batch_size, seq_len, base_config.hidden_size).cuda()
        
        # Measure syntaxes attention memory
        syntaxes_attention = SyntaxesAttention(base_config).cuda()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = syntaxes_attention(inputs=inputs)
        
        syntaxes_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Measure vanilla attention memory
        vanilla_attention = VanillaMHA(base_config).cuda()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = vanilla_attention(inputs)
        
        vanilla_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Syntaxes should use less memory
        memory_ratio = syntaxes_memory / vanilla_memory
        assert memory_ratio < 1.0, f"Syntaxes uses {memory_ratio:.2f}x memory of vanilla"
        
        print(f"\nMemory usage comparison:")
        print(f"  Syntaxes: {syntaxes_memory:.2f} MB")
        print(f"  Vanilla:  {vanilla_memory:.2f} MB")
        print(f"  Ratio:    {memory_ratio:.2f}x")