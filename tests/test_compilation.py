"""
Test suite for torch.compile functionality.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from praxis.trainers.compile import try_compile
from praxis import PraxisConfig, PraxisForCausalLM


class TestCompilation:
    """Test suite for torch.compile functionality."""

    @pytest.fixture
    def base_config(self):
        """Create a minimal base configuration."""
        return PraxisConfig(
            vocab_size=1000,
            hidden_size=64,
            embed_size=64,
            depth=2,
            num_heads=4,
            device="cpu",
        )

    def test_try_compile_with_model(self, base_config):
        """Test that try_compile works with a model."""
        model = PraxisForCausalLM(base_config)

        # Since we're on CPU and have a simple finally clause that returns model,
        # the function should return the original model
        compiled = try_compile(model, base_config)
        assert compiled is not None

    def test_try_compile_with_optimizer(self, base_config):
        """Test that try_compile works with an optimizer."""
        model = PraxisForCausalLM(base_config)
        optimizer = torch.optim.Adam(model.parameters())

        # The function should handle optimizers as well
        compiled = try_compile(optimizer, base_config)
        assert compiled is not None

    def test_try_compile_handles_exceptions(self, base_config):
        """Test that try_compile gracefully handles compilation failures."""
        model = PraxisForCausalLM(base_config)

        # Mock torch.compile to raise an exception
        with patch("torch.compile", side_effect=Exception("Compilation failed")):
            compiled = try_compile(model, base_config)
            # Should return the original model when compilation fails
            assert compiled is model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
