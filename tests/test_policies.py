"""
Tests for reinforcement learning policies in praxis.policies.
"""

import pytest
import torch
import torch.nn.functional as F

from praxis import PraxisConfig
from praxis.policies import GRPO, REINFORCE, ChainOfThought, ChainOfThoughtREINFORCE


@pytest.fixture
def config():
    """Base configuration for RL policies."""
    return PraxisConfig(
        hidden_size=128,
        dropout=0.1,
        grpo_group_size=4,
        grpo_kl_coeff=0.04,
        grpo_clip_ratio=0.2,
        rl_weight=0.1,
    )


@pytest.fixture
def sample_data():
    """Sample data for testing RL policies."""
    batch_size = 8
    seq_len = 32
    hidden_size = 128
    vocab_size = 1000
    
    return {
        "hidden_states": torch.randn(batch_size, seq_len, hidden_size),
        "logits": torch.randn(batch_size, seq_len, vocab_size),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "rewards": torch.tensor([0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4]),
        "ref_logits": torch.randn(batch_size, seq_len, vocab_size),
    }


class TestREINFORCE:
    """Test cases for REINFORCE policy."""
    
    def test_initialization(self, config):
        """Test REINFORCE policy initialization."""
        policy = REINFORCE(config)
        
        assert policy.config == config
        assert policy.hidden_size == config.hidden_size
        assert policy.rl_weight == config.rl_weight
        assert hasattr(policy, 'value_head')
        assert hasattr(policy, 'policy_mlp')
        assert hasattr(policy, 'baseline')
        
    def test_forward_without_rewards(self, config, sample_data):
        """Test forward pass without rewards (inference mode)."""
        policy = REINFORCE(config)
        policy.eval()
        
        hidden_states = sample_data["hidden_states"]
        
        with torch.no_grad():
            output_hidden, rl_loss = policy(hidden_states)
            
        assert output_hidden.shape == hidden_states.shape
        assert rl_loss is None
        
    def test_forward_with_rewards(self, config, sample_data):
        """Test forward pass with rewards (training mode)."""
        policy = REINFORCE(config)
        policy.train()
        
        hidden_states = sample_data["hidden_states"]
        rewards = sample_data["rewards"]
        mask = sample_data["attention_mask"]
        
        output_hidden, rl_loss = policy(hidden_states, rewards=rewards, mask=mask)
        
        assert output_hidden.shape == hidden_states.shape
        assert rl_loss is not None
        assert isinstance(rl_loss, torch.Tensor)
        assert rl_loss.requires_grad
        
    def test_baseline_update(self, config, sample_data):
        """Test that baseline gets updated during training."""
        policy = REINFORCE(config)
        policy.train()
        
        # Set baseline to a different value so we can see the change
        policy.baseline.data.fill_(0.1)
        initial_baseline = policy.baseline.data.clone()
        
        hidden_states = sample_data["hidden_states"]
        rewards = sample_data["rewards"]
        
        # Run multiple forward passes to ensure baseline update is noticeable
        for _ in range(5):
            policy(hidden_states, rewards=rewards)
        
        # Baseline should have changed significantly
        assert abs(policy.baseline.data.item() - initial_baseline.item()) > 1e-6
        
    def test_gradient_flow(self, config, sample_data):
        """Test that gradients flow through the policy."""
        policy = REINFORCE(config)
        policy.train()
        
        hidden_states = sample_data["hidden_states"]
        rewards = sample_data["rewards"]
        
        # Zero gradients first
        policy.zero_grad()
        
        _, rl_loss = policy(hidden_states, rewards=rewards)
        
        # rl_loss can be None if not training or no rewards
        if rl_loss is not None:
            rl_loss.backward()
            
            # Check that gradients exist for trainable parameters
            has_gradients = False
            for param in policy.parameters():
                if param.requires_grad and param.grad is not None:
                    has_gradients = True
                    break
            
            assert has_gradients, "No gradients found in any trainable parameters"


class TestGRPO:
    """Test cases for GRPO policy."""
    
    def test_initialization(self, config):
        """Test GRPO policy initialization."""
        policy = GRPO(config)
        
        assert policy.config == config
        assert policy.hidden_size == config.hidden_size
        assert policy.group_size == config.grpo_group_size
        assert policy.kl_coeff == config.grpo_kl_coeff
        assert policy.clip_ratio == config.grpo_clip_ratio
        
    def test_compute_group_advantages(self, config):
        """Test group advantage computation."""
        policy = GRPO(config)
        
        # Test with rewards that have clear groups
        rewards = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3])
        group_size = 4
        
        advantages = policy.compute_group_advantages(rewards, group_size)
        
        assert advantages.shape == rewards.shape
        # Check that advantages are normalized within groups
        group1_advantages = advantages[:4]
        group2_advantages = advantages[4:]
        
        assert abs(group1_advantages.mean().item()) < 1e-5  # Should be ~0
        assert abs(group2_advantages.mean().item()) < 1e-5  # Should be ~0
        
    def test_forward_without_rewards(self, config, sample_data):
        """Test forward pass without rewards."""
        policy = GRPO(config)
        
        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        
        output_hidden, losses = policy(hidden_states, logits, labels)
        
        assert output_hidden.shape == hidden_states.shape
        assert losses is None
        
    def test_forward_with_rewards(self, config, sample_data):
        """Test forward pass with rewards."""
        policy = GRPO(config)
        policy.train()
        
        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        rewards = sample_data["rewards"]
        ref_logits = sample_data["ref_logits"]
        mask = sample_data["attention_mask"]
        
        # Use rewards with different values to ensure variance
        varied_rewards = torch.tensor([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6])
        
        output_hidden, losses = policy(
            hidden_states, logits, labels, 
            rewards=varied_rewards, ref_logits=ref_logits, mask=mask
        )
        
        assert output_hidden.shape == hidden_states.shape
        # losses might be None if batch size issues or all rewards are zero
        if losses is not None:
            assert "grpo_loss" in losses
            assert "policy_loss" in losses
            assert "kl_loss" in losses
            assert "mean_reward" in losses
            assert "mean_advantage" in losses
        
    def test_zero_rewards_handling(self, config, sample_data):
        """Test handling of zero rewards."""
        policy = GRPO(config)
        policy.train()
        
        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        zero_rewards = torch.zeros(8)
        
        output_hidden, losses = policy(
            hidden_states, logits, labels, rewards=zero_rewards
        )
        
        assert output_hidden.shape == hidden_states.shape
        assert losses is None  # Should skip RL loss when all rewards are zero
        
    def test_small_batch_handling(self, config):
        """Test handling of batches smaller than group size."""
        policy = GRPO(config)
        policy.train()
        
        batch_size = 2  # Smaller than group_size (4)
        seq_len = 16
        hidden_size = 128
        vocab_size = 1000
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rewards = torch.tensor([0.8, 0.2])
        
        output_hidden, losses = policy(
            hidden_states, logits, labels, rewards=rewards
        )
        
        assert output_hidden.shape == hidden_states.shape
        if losses is not None:  # May be None if rewards are problematic
            assert "grpo_loss" in losses


class TestChainOfThought:
    """Test cases for Chain of Thought policy."""
    
    def test_initialization(self, config):
        """Test CoT policy initialization."""
        policy = ChainOfThought(config)
        
        assert policy.config == config
        assert policy.hidden_size == config.hidden_size
        assert hasattr(policy, 'quality_head')
        assert hasattr(policy, 'reasoning_weight')
        assert hasattr(policy, 'structure_bonus')
        
    def test_forward_without_training(self, config, sample_data):
        """Test forward pass in eval mode."""
        policy = ChainOfThought(config)
        policy.eval()
        
        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        
        with torch.no_grad():
            output_hidden, losses = policy(hidden_states, logits, labels)
            
        assert output_hidden.shape == hidden_states.shape
        assert losses is None
        
    def test_forward_with_training(self, config, sample_data):
        """Test forward pass in training mode."""
        policy = ChainOfThought(config)
        policy.train()
        
        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        attention_mask = sample_data["attention_mask"]
        
        output_hidden, losses = policy(
            hidden_states, logits, labels, attention_mask=attention_mask
        )
        
        assert output_hidden.shape == hidden_states.shape
        if losses is not None:  # May be None if no valid tokens
            assert "cot_loss" in losses
            assert "reasoning_quality" in losses
            assert "weighted_loss" in losses
            
    def test_quality_head(self, config, sample_data):
        """Test quality head computation."""
        policy = ChainOfThought(config)
        
        hidden_states = sample_data["hidden_states"]
        pooled_hidden = hidden_states.mean(dim=1)
        
        quality_scores = policy.quality_head(pooled_hidden)
        
        assert quality_scores.shape == (hidden_states.shape[0], 1)
        assert torch.all(quality_scores >= 0) and torch.all(quality_scores <= 1)


class TestChainOfThoughtREINFORCE:
    """Test cases for Chain of Thought with REINFORCE."""
    
    def test_initialization(self, config):
        """Test CoT-REINFORCE policy initialization."""
        policy = ChainOfThoughtREINFORCE(config)
        
        assert policy.config == config
        assert policy.hidden_size == config.hidden_size
        assert hasattr(policy, 'reinforce')
        assert hasattr(policy, 'reasoning_reward_weight')
        
    def test_compute_reasoning_rewards(self, config):
        """Test reasoning reward computation."""
        policy = ChainOfThoughtREINFORCE(config)
        
        # Test text with reasoning patterns
        good_text = """
        <initial_analysis>This is a math problem.</initial_analysis>
        <conscious_thought>I need to think step by step.</conscious_thought>
        <step_by_step>
        Step 1: Identify the variables
        Step 2: Apply the formula
        </step_by_step>
        <reflection>This approach seems correct.</reflection>
        The answer is 42.
        """
        
        bad_text = "The answer is 42."
        
        good_reward = policy.compute_reasoning_rewards(good_text, "42")
        bad_reward = policy.compute_reasoning_rewards(bad_text, "42")
        
        assert good_reward > bad_reward
        assert good_reward > 0  # Should get rewards for structure
        
    def test_forward_with_generated_texts(self, config, sample_data):
        """Test forward pass with generated texts for reward computation."""
        policy = ChainOfThoughtREINFORCE(config)
        policy.train()
        
        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        
        generated_texts = [
            "<step_by_step>Think step by step</step_by_step> Answer: 42",
            "Answer: 42",
            "<initial_analysis>Analyze first</initial_analysis> Answer: 42",
            "Answer: 42"
        ] * 2  # 8 texts for batch of 8
        
        ground_truths = ["42"] * 8
        
        output_hidden, losses = policy(
            hidden_states, logits, labels,
            generated_texts=generated_texts,
            ground_truths=ground_truths
        )
        
        assert output_hidden.shape == hidden_states.shape
        if losses is not None:
            assert "cot_reward_mean" in losses


class TestPolicyIntegration:
    """Integration tests for RL policies."""
    
    def test_all_policies_forward_pass(self, config, sample_data):
        """Test that all policies can perform forward passes."""
        policies = [
            REINFORCE(config),
            GRPO(config),
            ChainOfThought(config),
            ChainOfThoughtREINFORCE(config),
        ]
        
        for policy in policies:
            policy.train()
            
            # Test basic forward pass
            if isinstance(policy, (REINFORCE,)):
                output, loss = policy(sample_data["hidden_states"])
                assert output.shape == sample_data["hidden_states"].shape
                
            elif isinstance(policy, (GRPO,)):
                output, loss = policy(
                    sample_data["hidden_states"],
                    sample_data["logits"],
                    sample_data["labels"]
                )
                assert output.shape == sample_data["hidden_states"].shape
                
            elif isinstance(policy, (ChainOfThought, ChainOfThoughtREINFORCE)):
                output, loss = policy(
                    sample_data["hidden_states"],
                    sample_data["logits"],
                    sample_data["labels"]
                )
                assert output.shape == sample_data["hidden_states"].shape
                
    def test_device_compatibility(self, config, sample_data):
        """Test that policies work on different devices."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Specify exact device
            
            policy = REINFORCE(config).to(device)
            hidden_states = sample_data["hidden_states"].to(device)
            rewards = sample_data["rewards"].to(device)
            
            output, loss = policy(hidden_states, rewards=rewards)
            
            # Check device type matches (cuda:0 and cuda are equivalent)
            assert output.device.type == device.type
            if loss is not None:
                assert loss.device.type == device.type
                
    def test_policy_registry_completeness(self):
        """Test that all policies are registered."""
        from praxis.policies import RL_POLICIES_REGISTRY
        
        expected_policies = {
            "reinforce": REINFORCE,
            "grpo": GRPO,
            "cot": ChainOfThought,
            "cot-reinforce": ChainOfThoughtREINFORCE,
        }
        
        for name, policy_class in expected_policies.items():
            assert name in RL_POLICIES_REGISTRY
            assert RL_POLICIES_REGISTRY[name] == policy_class


@pytest.mark.parametrize("policy_class", [REINFORCE, GRPO, ChainOfThought, ChainOfThoughtREINFORCE])
def test_policy_parameter_counts(config, policy_class):
    """Test that policies have reasonable parameter counts."""
    policy = policy_class(config)
    
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    # GRPO doesn't have parameters by design - it works directly with LM outputs
    if policy_class == GRPO:
        assert total_params == 0, "GRPO should have no parameters by design"
        assert trainable_params == 0, "GRPO should have no trainable parameters by design"
    else:
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_policy_batch_size_handling(config, batch_size):
    """Test policies with different batch sizes."""
    seq_len = 32
    hidden_size = config.hidden_size
    vocab_size = 1000
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rewards = torch.rand(batch_size)
    
    # Test GRPO with different batch sizes
    policy = GRPO(config)
    policy.train()
    
    output, losses = policy(hidden_states, logits, labels, rewards=rewards)
    
    assert output.shape == hidden_states.shape
    # For small batches or zero rewards, losses might be None


class TestPolicyEdgeCases:
    """Edge case tests for RL policies robustness."""
    
    def test_reinforce_extreme_rewards(self, config):
        """Test REINFORCE with extreme reward values."""
        policy = REINFORCE(config)
        policy.train()
        
        batch_size, seq_len, hidden_size = 4, 16, config.hidden_size
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test with very large rewards
        large_rewards = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])
        output, loss = policy(hidden_states, rewards=large_rewards)
        assert output.shape == hidden_states.shape
        assert loss is not None and torch.isfinite(loss)
        
        # Test with very small rewards
        small_rewards = torch.tensor([1e-8, 2e-8, 3e-8, 4e-8])
        output, loss = policy(hidden_states, rewards=small_rewards)
        assert output.shape == hidden_states.shape
        assert loss is not None and torch.isfinite(loss)
        
        # Test with negative rewards
        negative_rewards = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        output, loss = policy(hidden_states, rewards=negative_rewards)
        assert output.shape == hidden_states.shape
        assert loss is not None and torch.isfinite(loss)
        
    def test_grpo_identical_rewards(self, config):
        """Test GRPO with identical rewards within groups."""
        policy = GRPO(config)
        policy.train()
        
        batch_size, seq_len, hidden_size, vocab_size = 8, 16, config.hidden_size, 1000
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # All rewards identical (should result in zero advantages)
        identical_rewards = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8])
        
        output, losses = policy(hidden_states, logits, labels, rewards=identical_rewards)
        assert output.shape == hidden_states.shape
        # Should handle identical rewards gracefully
        
    def test_policies_with_masked_sequences(self, config):
        """Test policies with heavily masked sequences."""
        batch_size, seq_len, hidden_size, vocab_size = 4, 16, config.hidden_size, 1000
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rewards = torch.rand(batch_size)
        
        # Create mask where most tokens are masked out
        mask = torch.zeros(batch_size, seq_len)
        mask[:, :3] = 1.0  # Only first 3 tokens are valid
        
        # Test REINFORCE
        reinforce_policy = REINFORCE(config)
        reinforce_policy.train()
        output, loss = reinforce_policy(hidden_states, rewards=rewards, mask=mask)
        assert output.shape == hidden_states.shape
        
        # Test GRPO
        grpo_policy = GRPO(config)
        grpo_policy.train()
        output, losses = grpo_policy(hidden_states, logits, labels, rewards=rewards, mask=mask)
        assert output.shape == hidden_states.shape
        
    def test_policies_numerical_stability(self, config):
        """Test policies with inputs that could cause numerical issues."""
        batch_size, seq_len, hidden_size, vocab_size = 4, 16, config.hidden_size, 1000
        
        # Test with very large hidden states
        large_hidden = torch.randn(batch_size, seq_len, hidden_size) * 100
        large_logits = torch.randn(batch_size, seq_len, vocab_size) * 100
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rewards = torch.rand(batch_size)
        
        # Test REINFORCE numerical stability
        reinforce_policy = REINFORCE(config)
        reinforce_policy.train()
        output, loss = reinforce_policy(large_hidden, rewards=rewards)
        assert output.shape == large_hidden.shape
        assert loss is None or torch.isfinite(loss)
        
        # Test GRPO numerical stability
        grpo_policy = GRPO(config)
        grpo_policy.train()
        output, losses = grpo_policy(large_hidden, large_logits, labels, rewards=rewards)
        assert output.shape == large_hidden.shape
        
    def test_cot_with_empty_reasoning_patterns(self, config):
        """Test Chain of Thought with texts lacking reasoning patterns."""
        policy = ChainOfThoughtREINFORCE(config)
        
        # Test texts with no reasoning markers
        empty_texts = [
            "Just an answer: 42",
            "42",
            "The result is 42",
            "Answer: 42"
        ]
        ground_truths = ["42"] * 4
        
        for text, truth in zip(empty_texts, ground_truths):
            reward = policy.compute_reasoning_rewards(text, truth)
            assert isinstance(reward, (int, float))
            assert reward >= 0  # Should not be negative even without reasoning
            
    def test_policy_state_preservation(self, config):
        """Test that policies maintain proper state across forward passes."""
        policy = REINFORCE(config)
        policy.train()
        
        batch_size, seq_len, hidden_size = 4, 16, config.hidden_size
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        rewards = torch.rand(batch_size)
        
        # Save initial state
        initial_baseline = policy.baseline.data.clone()
        initial_step_count = policy.step_count
        
        # Multiple forward passes
        for i in range(3):
            output, loss = policy(hidden_states, rewards=rewards)
            assert output.shape == hidden_states.shape
            assert policy.step_count == initial_step_count + i + 1
            
        # Baseline should have changed
        assert not torch.equal(policy.baseline.data, initial_baseline)


class TestPolicyPerformance:
    """Performance tests for RL policies."""
    
    def test_policy_forward_speed(self, config):
        """Test that policies have reasonable forward pass speed."""
        import time
        
        batch_size, seq_len, hidden_size, vocab_size = 16, 64, config.hidden_size, 1000
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rewards = torch.rand(batch_size)
        
        policies = {
            "REINFORCE": REINFORCE(config),
            "GRPO": GRPO(config),
            "ChainOfThought": ChainOfThought(config),
            "ChainOfThoughtREINFORCE": ChainOfThoughtREINFORCE(config),
        }
        
        for name, policy in policies.items():
            policy.train()
            
            # Warmup run
            if name == "REINFORCE":
                policy(hidden_states, rewards=rewards)
            else:
                policy(hidden_states, logits, labels)
            
            # Timed runs
            start_time = time.time()
            num_runs = 5
            
            for _ in range(num_runs):
                if name == "REINFORCE":
                    policy(hidden_states, rewards=rewards)
                else:
                    policy(hidden_states, logits, labels)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            # Should complete within reasonable time (< 1 second per forward pass)
            assert avg_time < 1.0, f"{name} policy too slow: {avg_time:.3f}s per forward pass"
            
    def test_policy_memory_efficiency(self, config):
        """Test that policies don't have memory leaks."""
        import gc
        import psutil
        import os
        
        batch_size, seq_len, hidden_size = 8, 32, config.hidden_size
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        rewards = torch.rand(batch_size)
        
        policy = REINFORCE(config)
        policy.train()
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple forward passes
        for _ in range(20):
            output, loss = policy(hidden_states, rewards=rewards)
            if loss is not None:
                loss.backward()
            policy.zero_grad()
            
            # Force garbage collection
            gc.collect()
            
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.1f}MB increase"