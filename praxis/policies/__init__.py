"""
Reinforcement Learning policies for training language models.
"""

from praxis.policies.cot import ChainOfThought
from praxis.policies.grpo import GRPO
from praxis.policies.reinforce import REINFORCE

# Registry for RL algorithms
RL_POLICIES_REGISTRY = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
    "cot": ChainOfThought,  # Basic supervised CoT
    # "ppo": PPO,    # TODO: Implement PPO
}

__all__ = ["REINFORCE", "GRPO", "ChainOfThought", "RL_POLICIES_REGISTRY"]
