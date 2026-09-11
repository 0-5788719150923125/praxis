"""Tests for the trainers module."""

import pytest

from praxis import PraxisConfig, PraxisForCausalLM, registry
from praxis.trainers import BackpropagationTrainer, create_trainer_with_module


class TestTrainerFactory:
    """Test the trainer factory function."""

    def test_create_trainer_with_module(self, tmpdir):
        """Test creating a trainer with module."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=100,
        )
        model = PraxisForCausalLM(config)

        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2
            vocab_size = 100

        tokenizer = MockTokenizer()

        # Create a temporary checkpoint directory
        checkpoint_dir = str(tmpdir.mkdir("checkpoints"))

        trainer, trainer_module = create_trainer_with_module(
            trainer_type="backpropagation",
            model=model,
            tokenizer=tokenizer,
            hparams={
                "batch_size": 4,
                "device": "cpu",
                "learning_rate": 1e-3,
                "max_epochs": 1,
                "accumulate_grad_batches": 1,
                "gradient_clip_val": 1.0,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_every_n_steps": 100,
            },
            experiment_name="test",
            run_name="test_run",
        )

        assert trainer is not None
        assert trainer_module is not None
        assert isinstance(trainer_module, BackpropagationTrainer)

    def test_trainer_registry(self):
        """Test that trainer registry contains expected trainers."""
        assert "backpropagation" in registry.namespace("trainers")
        assert "mono_forward" in registry.namespace("trainers")
        assert "mono_forward_ray" in registry.namespace("trainers")

        # Test that backpropagation trainer is directly accessible
        assert registry.lookup("trainers", "backpropagation") == BackpropagationTrainer

        # Both Mono-Forward profiles are lazy loaders so the package
        # (and Ray) only get imported when the profile is actually
        # selected.
        assert callable(registry.lookup("trainers", "mono_forward"))
        assert callable(registry.lookup("trainers", "mono_forward_ray"))
