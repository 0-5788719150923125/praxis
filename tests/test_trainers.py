"""Tests for the trainers module."""

import pytest
import torch
from transformers import AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.optimizers import get_optimizer, get_optimizer_profile
from praxis.schedulers import get_scheduler_func
from praxis.trainers import (
    TRAINER_REGISTRY,
    BackpropagationTrainer,
    create_trainer_with_module,
    try_compile,
)


class TestBackpropagationTrainer:
    """Test cases for BackpropagationTrainer."""

    @pytest.fixture
    def setup_model(self):
        """Create a small model for testing."""
        config = PraxisConfig(
            depth=2,
            hidden_size=64,
            embed_size=32,
            vocab_size=100,
            num_heads=2,
            num_queries=2,
            device_map="cpu",
        )
        model = PraxisForCausalLM(config)
        return model, config

    @pytest.fixture
    def setup_tokenizer(self):
        """Create a simple tokenizer for testing."""

        # Create a mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2
            sep_token_id = 3

            def encode(self, text, return_tensors=None):
                # Simple mock encoding
                if return_tensors == "pt":
                    return torch.tensor([[1, 2, 3, 4, 5]])
                return [1, 2, 3, 4, 5]

            def decode(self, ids, skip_special_tokens=False):
                return "test output"

        return MockTokenizer()

    def test_trainer_initialization(self, setup_model, setup_tokenizer):
        """Test basic BackpropagationTrainer initialization."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        # Setup optimizer and scheduler
        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        # Create trainer
        trainer = BackpropagationTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            byte_latent=False,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.tokenizer is tokenizer
        assert (
            trainer.outputs_are_aligned is False
        )  # byte_latent=False means outputs_are_aligned=False

    def test_trainer_forward_with_kwargs(self, setup_model, setup_tokenizer):
        """Test that BackpropagationTrainer forward accepts keyword arguments."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        trainer = BackpropagationTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            byte_latent=False,
        )

        batch_size = 2
        seq_len = 10

        # Test with keyword arguments (matching training_step usage)
        outputs = trainer.forward(
            input_ids=torch.randint(0, 100, (batch_size, seq_len)),
            labels=torch.randint(0, 100, (batch_size, seq_len - 1)),
        )
        assert outputs is not None
        assert hasattr(outputs, "loss")

    def test_trainer_forward_pass(self, setup_model, setup_tokenizer):
        """Test forward pass through BackpropagationTrainer."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        trainer = BackpropagationTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            byte_latent=False,
        )

        batch_size = 2
        seq_len = 10
        inputs = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "labels": torch.randint(0, 100, (batch_size, seq_len - 1)),
        }

        # Forward pass with dict unpacking
        outputs = trainer.forward(**inputs)
        assert outputs is not None
        assert hasattr(outputs, "loss")


class TestCompilationUtils:
    """Test compilation utilities."""

    def test_try_compile_with_model(self):
        """Test that try_compile works with models."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=10,
        )
        model = PraxisForCausalLM(config)
        hparams = {"device": "cpu"}

        compiled = try_compile(model, hparams)
        # Should return a model (original or compiled)
        assert compiled is not None

    def test_try_compile_with_optimizer(self):
        """Test that try_compile works with optimizers."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=10,
        )
        model = PraxisForCausalLM(config)
        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        hparams = {"device": "cpu"}

        compiled = try_compile(optimizer, hparams)
        # Should return an optimizer (original or compiled)
        assert compiled is not None


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
        assert "backpropagation" in TRAINER_REGISTRY
        assert "mono_forward" in TRAINER_REGISTRY

        # Test that backpropagation trainer is directly accessible
        assert TRAINER_REGISTRY["backpropagation"] == BackpropagationTrainer

        # Test that mono_forward is a callable (lazy loader)
        assert callable(TRAINER_REGISTRY["mono_forward"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
