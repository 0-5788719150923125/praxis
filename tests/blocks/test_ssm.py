"""SSMBlock (praxis/blocks/ssm.py). Its gradient path is covered by the blocks
registry sweep and its causality by tests/test_modeling.py."""

import pytest
import torch

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

    def test_different_sequence_lengths(self, config):
        """Test SSMBlock with different sequence lengths."""
        block = SSMBlock(config)
        batch_size = 2

        for seq_len in [1, 64]:  # 1 is the decode case
            x = torch.randn(batch_size, seq_len, config.hidden_size)
            output, _, state, _ = block(x)

            assert output.shape == (batch_size, seq_len, config.hidden_size)
            assert state.shape == (
                batch_size,
                config.hidden_size * config.ssm_expand_factor,
                config.ssm_state_size,
            )
