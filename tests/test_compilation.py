"""
Test suite for components that cannot be compiled with torch.compile.
This suite verifies that components marked with can_compile=False are properly
detected and skipped during compilation.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from praxis.trainers.compile import try_compile_model, _check_module_compilability
from praxis import PraxisConfig, PraxisForCausalLM


class TestNonCompilableComponents:
    """Test suite for components that cannot use torch.compile."""

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

    def test_flex_attention_not_compilable(self, base_config):
        """Test that FlexAttention is marked as non-compilable."""
        config = base_config
        config.attention_type = "flex_attention"

        model = PraxisForCausalLM(config)

        # Check that FlexAttention modules have can_compile=False
        for name, module in model.named_modules():
            if module.__class__.__name__ == "FlexAttention":
                assert hasattr(
                    module.__class__, "can_compile"
                ), f"FlexAttention at {name} missing can_compile attribute"
                assert (
                    not module.__class__.can_compile
                ), f"FlexAttention at {name} should have can_compile=False"
                break
        else:
            pytest.fail("No FlexAttention module found in model")

    def test_smear_router_not_compilable(self, base_config):
        """Test that SmearRouter is marked as non-compilable."""
        config = base_config
        config.router_type = "smear"
        config.num_experts = 4

        model = PraxisForCausalLM(config)

        # Check that SmearRouter modules have can_compile=False
        for name, module in model.named_modules():
            if module.__class__.__name__ == "SmearRouter":
                assert hasattr(
                    module.__class__, "can_compile"
                ), f"SmearRouter at {name} missing can_compile attribute"
                assert (
                    not module.__class__.can_compile
                ), f"SmearRouter at {name} should have can_compile=False"
                break
        else:
            # SmearRouter might not be present if not using expert-based decoder
            pytest.skip(
                "No SmearRouter module found in model (expected if not using expert-based decoder)"
            )

    def test_compilability_check_detects_non_compilable(self, base_config):
        """Test that _check_module_compilability correctly identifies non-compilable modules."""
        config = base_config
        config.attention_type = "flex_attention"

        model = PraxisForCausalLM(config)

        can_compile, module_path, module_type = _check_module_compilability(model)

        assert (
            not can_compile
        ), "Model with FlexAttention should be detected as non-compilable"
        assert (
            module_type == "FlexAttention"
        ), f"Expected FlexAttention, got {module_type}"
        assert (
            module_path is not None
        ), "Should have a path to the non-compilable module"

    def test_compile_model_skips_non_compilable(self, base_config):
        """Test that try_compile_model skips compilation for non-compilable components."""
        config = base_config
        config.attention_type = "flex_attention"

        model = PraxisForCausalLM(config)

        # Mock torch.compile to track if it was called
        with patch("torch.compile") as mock_compile:
            compiled_model = try_compile_model(model, config)

            # torch.compile should NOT be called for non-compilable models
            mock_compile.assert_not_called()

            # The returned model should be the same as the input
            assert (
                compiled_model is model
            ), "Non-compilable model should be returned unchanged"

    def test_compile_model_compiles_normal_models(self, base_config):
        """Test that try_compile_model does compile models without non-compilable components."""
        config = base_config
        config.attention_type = "standard"  # Use a compilable attention type
        config.device = "cuda"  # Force cuda device to enable compilation

        model = PraxisForCausalLM(config)

        # Check that the model is compilable
        can_compile, _, _ = _check_module_compilability(model)
        assert can_compile, "Model with standard attention should be compilable"

        # Mock torch.compile to track if it was called
        with patch("torch.compile") as mock_compile:
            mock_compiled = MagicMock()
            mock_compile.return_value = mock_compiled

            compiled_model = try_compile_model(model, config)

            # torch.compile should be called for compilable models
            mock_compile.assert_called_once()

            # The returned model should be the compiled version
            assert (
                compiled_model is mock_compiled
            ), "Compilable model should return compiled version"

    @pytest.mark.parametrize(
        "attention_type,should_compile",
        [
            ("standard", True),
            ("vanilla", True),
            ("syntaxes", True),
            ("flex_attention", False),
        ],
    )
    def test_attention_types_compilability(
        self, base_config, attention_type, should_compile
    ):
        """Test compilability of different attention types."""
        config = base_config
        config.attention_type = attention_type

        try:
            model = PraxisForCausalLM(config)
        except Exception as e:
            pytest.skip(f"Could not create model with {attention_type}: {e}")

        can_compile, module_path, module_type = _check_module_compilability(model)

        if should_compile:
            assert can_compile, f"{attention_type} should be compilable"
            assert (
                module_path is None
            ), f"No non-compilable module should be found for {attention_type}"
        else:
            assert not can_compile, f"{attention_type} should not be compilable"
            assert (
                module_path is not None
            ), f"Should find non-compilable module for {attention_type}"

    def test_custom_non_compilable_module(self):
        """Test that custom modules with can_compile=False are detected."""

        class CustomNonCompilableModule(nn.Module):
            can_compile = False

            def forward(self, x):
                return x

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.custom = CustomNonCompilableModule()
                self.layer2 = nn.Linear(10, 10)

            def forward(self, x):
                x = self.layer1(x)
                x = self.custom(x)
                x = self.layer2(x)
                return x

        model = TestModel()

        can_compile, module_path, module_type = _check_module_compilability(model)

        assert (
            not can_compile
        ), "Model with custom non-compilable module should be detected"
        assert module_type == "CustomNonCompilableModule"
        assert "custom" in module_path

    def test_nested_non_compilable_detection(self):
        """Test that deeply nested non-compilable modules are detected."""

        class NonCompilableLayer(nn.Module):
            can_compile = False

            def forward(self, x):
                return x

        class NestedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.deep = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.Sequential(
                        nn.Linear(10, 10), NonCompilableLayer(), nn.Linear(10, 10)
                    ),
                )

            def forward(self, x):
                return self.deep(x)

        model = NestedModule()

        can_compile, module_path, module_type = _check_module_compilability(model)

        assert not can_compile, "Deeply nested non-compilable module should be detected"
        assert module_type == "NonCompilableLayer"
        assert "deep" in module_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
