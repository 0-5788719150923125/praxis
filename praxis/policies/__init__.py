"""
Reinforcement Learning policies for training language models.
"""

from praxis.policies.cot import ChainOfThought
from praxis.policies.grpo import GRPO
from praxis.policies.harmonic_weight_rl import HarmonicWeightPolicy
from praxis.policies.reinforce import REINFORCE

# Registry for RL algorithms
RL_POLICIES_REGISTRY = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
    "cot": ChainOfThought,  # Basic supervised CoT
    # Weight-editing controller (driven by a callback, not the forward pass).
    "harmonic_weight": HarmonicWeightPolicy,
    # "ppo": PPO,    # TODO: Implement PPO
}

__all__ = [
    "REINFORCE",
    "GRPO",
    "ChainOfThought",
    "HarmonicWeightPolicy",
    "RL_POLICIES_REGISTRY",
]
