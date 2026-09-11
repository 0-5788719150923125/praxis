import pytest
import torch

from praxis import registry
from praxis.policies import GRPO, REINFORCE, ChainOfThought
from praxis.policies.harmonic_weight_rl import HarmonicWeightPolicy

# ------------------------------------------------------------------------------
# policies
# ------------------------------------------------------------------------------
# Tests for reinforcement learning policies in praxis.policies.


class TestPolicyIntegration:
    """Integration tests for RL policies."""

    def test_all_policies_forward_pass(self, config, sample_data):
        """Test that all policies can perform forward passes."""
        policies = [
            REINFORCE(config),
            GRPO(config),
            ChainOfThought(config),
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
                    sample_data["labels"],
                )
                assert output.shape == sample_data["hidden_states"].shape

            elif isinstance(policy, (ChainOfThought)):
                output, loss = policy(
                    sample_data["hidden_states"],
                    sample_data["logits"],
                    sample_data["labels"],
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

        expected_policies = {
            "reinforce": REINFORCE,
            "grpo": GRPO,
            "cot": ChainOfThought,
        }

        for name, policy_class in expected_policies.items():
            assert name in registry.namespace("rl_policies")
            assert registry.lookup("rl_policies", name) == policy_class


# ------------------------------------------------------------------------------
# harmonic_weight_rl
# ------------------------------------------------------------------------------
# Harmonic-weight RL controller: policy-gradient mechanics + callback loop.


def test_registered_in_rl_registry():
    assert registry.lookup("rl_policies", "harmonic_weight") is HarmonicWeightPolicy
    assert HarmonicWeightPolicy.is_weight_controller is True
