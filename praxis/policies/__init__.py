"""
Reinforcement Learning policies for training language models.
"""

from praxis.policies.grpo import GRPO
from praxis.policies.reinforce import REINFORCE
from praxis.policies.cot import ChainOfThought, ChainOfThoughtREINFORCE

# Registry for RL algorithms
RL_POLICIES_REGISTRY = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
    "cot": ChainOfThought,  # Basic supervised CoT
    "cot-reinforce": ChainOfThoughtREINFORCE,  # CoT with REINFORCE
    # "ppo": PPO,    # TODO: Implement PPO
}

__all__ = ["REINFORCE", "GRPO", "ChainOfThought", "ChainOfThoughtREINFORCE", "RL_POLICIES_REGISTRY"]