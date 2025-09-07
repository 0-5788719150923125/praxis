"""Tests for the trainers module."""

import pytest
import torch
from transformers import AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.optimizers import get_optimizer, get_optimizer_profile
from praxis.trainers import (
    PraxisTrainer, 
    try_compile_model, 
    try_compile_optimizer,
    create_trainer_with_module,
    TRAINER_REGISTRY
)
from praxis.schedulers import get_scheduler_func


class TestPraxisTrainer:
    """Test cases for PraxisTrainer."""

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
        """Test basic PraxisTrainer initialization."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        # Setup optimizer and scheduler
        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        # Create trainer
        trainer = PraxisTrainer(
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
        assert trainer.byte_latent is False

    def test_trainer_forward_with_kwargs(self, setup_model, setup_tokenizer):
        """Test that PraxisTrainer forward accepts keyword arguments."""
        model, config = setup_model
        tokenizer = setup_tokenizer
        
        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)
        
        trainer = PraxisTrainer(
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
        """Test forward pass through PraxisTrainer."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        trainer = PraxisTrainer(
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

    def test_try_compile_model_cpu(self):
        """Test that compilation is skipped on CPU."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=10,
        )
        model = PraxisForCausalLM(config)
        hparams = {"device": "cpu"}

        compiled = try_compile_model(model, hparams)
        # Should return original model on CPU
        assert compiled is model

    def test_try_compile_model_dev_mode(self):
        """Test that compilation is skipped in dev mode."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=10,
        )
        model = PraxisForCausalLM(config)
        hparams = {"device": "cuda", "dev": True}

        compiled = try_compile_model(model, hparams)
        # Should return original model in dev mode
        assert compiled is model

    def test_try_compile_optimizer_cpu(self):
        """Test that optimizer compilation is skipped on CPU."""
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

        compiled = try_compile_optimizer(optimizer, hparams)
        # Should return original optimizer on CPU
        assert compiled is optimizer


class TestTrainerFactory:
    """Test the trainer factory function."""
    
    @pytest.fixture
    def setup_components(self):
        """Create model and optimizer components for testing."""
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
        
        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)
        
        class MockTokenizer:
            pad_token_id = 0
            
        return model, optimizer, scheduler, MockTokenizer()
    
    def test_create_praxis_trainer(self, setup_components):
        """Test creating PraxisTrainer through factory."""
        model, optimizer, scheduler, tokenizer = setup_components
        
        trainer, training_module = create_trainer_with_module(
            trainer_type="praxis",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            trainer_params={"max_steps": 10}
        )
        
        # Should return Lightning Trainer and PraxisTrainer module
        from praxis.trainers import Trainer
        assert isinstance(trainer, Trainer)
        assert isinstance(training_module, PraxisTrainer)
        assert training_module.model is not None
        
    def test_create_default_trainer(self, setup_components):
        """Test that 'default' maps to PraxisTrainer."""
        model, optimizer, scheduler, tokenizer = setup_components
        
        trainer, training_module = create_trainer_with_module(
            trainer_type="default",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            trainer_params={"max_steps": 10}
        )
        
        from praxis.trainers import Trainer
        assert isinstance(trainer, Trainer)
        assert isinstance(training_module, PraxisTrainer)
        
    def test_create_unknown_trainer(self, setup_components):
        """Test fallback for unknown trainer type."""
        model, optimizer, scheduler, tokenizer = setup_components
        
        trainer, training_module = create_trainer_with_module(
            trainer_type="unknown_type",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            trainer_params={"max_steps": 10}
        )
        
        from praxis.trainers import Trainer
        assert isinstance(trainer, Trainer)
        # Should return original model for unknown types
        assert training_module is model
        
    def test_trainer_registry_exists(self):
        """Test that trainer registry contains expected entries."""
        assert "praxis" in TRAINER_REGISTRY
        assert "default" in TRAINER_REGISTRY
        assert "mono_forward" in TRAINER_REGISTRY
        assert "mono-forward" in TRAINER_REGISTRY
        
    def test_encoder_type_detection(self, setup_components):
        """Test that byte_latent is set based on encoder_type."""
        model, optimizer, scheduler, tokenizer = setup_components
        
        trainer, training_module = create_trainer_with_module(
            trainer_type="praxis",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"encoder_type": "byte_latent"},
            tokenizer=tokenizer,
            trainer_params={"max_steps": 10}
        )
        
        assert isinstance(training_module, PraxisTrainer)
        assert training_module.byte_latent is True