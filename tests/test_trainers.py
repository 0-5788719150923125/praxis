"""Tests for the trainers module."""

import pytest
import torch
from transformers import AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.optimizers import get_optimizer, get_optimizer_profile
from praxis.trainers import PraxisTrainer, try_compile_model, try_compile_optimizer
from praxis.utils import get_scheduler


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
        """Test that PraxisTrainer can be initialized."""
        model, config = setup_model
        tokenizer = setup_tokenizer
        
        # Create optimizer and scheduler
        optimizer_config, disable_schedule = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler = get_scheduler(optimizer, optimizer_config, disable_schedule)(optimizer)
        
        # Create trainer
        hparams = {"device": "cpu", "dev": True}
        trainer = PraxisTrainer(
            model, 
            optimizer, 
            scheduler, 
            hparams,
            tokenizer=tokenizer,
            byte_latent=False
        )
        
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_forward_pass(self, setup_model, setup_tokenizer):
        """Test forward pass through trainer."""
        model, config = setup_model
        tokenizer = setup_tokenizer
        
        optimizer_config, disable_schedule = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler = get_scheduler(optimizer, optimizer_config, disable_schedule)(optimizer)
        
        hparams = {"device": "cpu", "dev": True}
        trainer = PraxisTrainer(
            model, 
            optimizer, 
            scheduler, 
            hparams,
            tokenizer=tokenizer,
            byte_latent=False
        )
        
        # Create dummy inputs
        batch_size, seq_len = 2, 10
        inputs = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "labels": torch.randint(0, 100, (batch_size, seq_len - 1)),
        }
        
        # Forward pass
        outputs = trainer.forward(inputs)
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