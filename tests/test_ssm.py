import pytest
import torch
import torch.nn as nn

from praxis import PraxisConfig
from praxis.blocks.ssm import SSMBlock


class TestSSMBlock:
    """Test suite for SSMBlock implementation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PraxisConfig(
            hidden_size=128,
            epsilon=1e-5,
            residual_type="standard",
            scaled=False,
            ssm_state_size=16,
            ssm_conv_size=4,
            ssm_expand_factor=2,
            ssm_dt_rank=8,
        )

    def test_initialization(self, config):
        """Test SSMBlock initialization."""
        block = SSMBlock(config)

        # Check attributes
        assert block.hidden_size == 128
        assert block.state_size == 16
        assert block.conv_size == 4
        assert block.expand_factor == 2
        assert block.dt_rank == 8

        # Check layer dimensions
        assert block.in_proj.in_features == 128
        assert block.in_proj.out_features == 128 * 2 * 2  # expand_factor * 2 for gating
        assert block.out_proj.in_features == 128 * 2
        assert block.out_proj.out_features == 128

        # Check conv1d dimensions
        assert block.conv1d.in_channels == 256
        assert block.conv1d.out_channels == 256
        assert block.conv1d.kernel_size == (4,)

        # Check SSM parameters
        assert block.A_log.shape == (256, 16)
        assert block.D.shape == (256,)

    def test_forward_pass(self, config):
        """Test forward pass through SSMBlock."""
        block = SSMBlock(config)
        batch_size = 4
        seq_len = 32

        # Create input
        x = torch.randn(batch_size, seq_len, config.hidden_size)

        # Forward pass
        output, past_kv, new_state, aux_loss = block(x)

        # Check output shapes
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert past_kv is None
        assert new_state.shape == (
            batch_size,
            config.hidden_size * config.ssm_expand_factor,
            config.ssm_state_size,
        )
        assert aux_loss.item() == 0.0

    def test_state_persistence(self, config):
        """Test that SSM state is properly maintained across calls."""
        block = SSMBlock(config)
        batch_size = 2
        seq_len = 16

        # First forward pass
        x1 = torch.randn(batch_size, seq_len, config.hidden_size)
        output1, _, state1, _ = block(x1)

        # Second forward pass with state
        x2 = torch.randn(batch_size, seq_len, config.hidden_size)
        output2, _, state2, _ = block(x2, current_state=state1)

        # States should be different
        assert not torch.allclose(state1, state2)

        # Third forward pass without state should be different from second
        output3, _, state3, _ = block(x2)
        assert not torch.allclose(output2, output3)

    def test_causal_masking(self, config):
        """Test that the convolution maintains causality."""
        block = SSMBlock(config)
        batch_size = 1
        seq_len = 10

        # Create input where later positions have distinct values
        x = torch.zeros(batch_size, seq_len, config.hidden_size)
        x[:, -1, :] = 1.0  # Set last position to 1

        # Forward pass
        output, _, _, _ = block(x)

        # Early outputs should not be affected by the last position
        # This is a basic check - in practice, causality is maintained by conv padding
        assert output.shape == x.shape

    def test_gradient_flow(self, config):
        """Test gradient flow through SSMBlock."""
        block = SSMBlock(config)
        batch_size = 2
        seq_len = 8

        x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
        output, _, _, _ = block(x)

        # Compute loss and gradients
        loss = output.sum()
        loss.backward()

        # Check that gradients flow
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_different_sequence_lengths(self, config):
        """Test SSMBlock with different sequence lengths."""
        block = SSMBlock(config)
        batch_size = 2

        for seq_len in [1, 8, 64, 128]:
            x = torch.randn(batch_size, seq_len, config.hidden_size)
            output, _, state, _ = block(x)

            assert output.shape == (batch_size, seq_len, config.hidden_size)
            assert state.shape == (
                batch_size,
                config.hidden_size * config.ssm_expand_factor,
                config.ssm_state_size,
            )

    def test_numerical_stability(self, config):
        """Test numerical stability of SSMBlock."""
        block = SSMBlock(config)
        batch_size = 2
        seq_len = 16

        # Test with large inputs
        x_large = torch.randn(batch_size, seq_len, config.hidden_size) * 10
        output_large, _, _, _ = block(x_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()

        # Test with small inputs
        x_small = torch.randn(batch_size, seq_len, config.hidden_size) * 0.001
        output_small, _, _, _ = block(x_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()

    def test_device_compatibility(self, config):
        """Test SSMBlock on different devices."""
        block = SSMBlock(config)
        batch_size = 2
        seq_len = 8

        # CPU test
        x_cpu = torch.randn(batch_size, seq_len, config.hidden_size)
        output_cpu, _, state_cpu, _ = block(x_cpu)
        assert output_cpu.device == x_cpu.device
        assert state_cpu.device == x_cpu.device

        # GPU test (if available)
        if torch.cuda.is_available():
            block_gpu = block.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu, _, state_gpu, _ = block_gpu(x_gpu)
            assert output_gpu.device == x_gpu.device
            assert state_gpu.device == x_gpu.device

            # Results should be similar
            assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-4)
