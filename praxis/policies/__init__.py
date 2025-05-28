"""
Reinforcement Learning policies for training language models.
"""

from praxis.policies.grpo import GRPO
from praxis.policies.reinforce import REINFORCE

# Registry for RL algorithms
RL_POLICIES_REGISTRY = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
    # "ppo": PPO,    # TODO: Implement PPO
}

__all__ = ["REINFORCE", "GRPO", "RL_POLICIES_REGISTRY"]