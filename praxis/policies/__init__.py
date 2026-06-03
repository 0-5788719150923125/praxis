"""
Reinforcement Learning policies for training language models.
"""

from praxis.policies.cot import ChainOfThought
from praxis.policies.engagement import EngagementPolicy, JokePolicy
from praxis.policies.grpo import GRPO
from praxis.policies.harmonic_weight_rl import HarmonicWeightPolicy
from praxis.policies.reinforce import REINFORCE

# Registry for RL algorithms
RL_POLICIES_REGISTRY = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
    "cot": ChainOfThought,  # Basic supervised CoT
    # Forward-path engagement-prediction reward (computes its own reward from
    # labels, so it needs no RL dataset). See PLAN.md / next/homeostatic_engagement.md.
    "engagement": EngagementPolicy,
    # Forward-path joke reward (same machinery; dense grounding from well-rated
    # jokes, live signal from human approval via the Loop UI). PLAN.md section 7b.
    "joke": JokePolicy,
    # Weight-editing controller (driven by a callback, not the forward pass).
    "harmonic_weight": HarmonicWeightPolicy,
    # "ppo": PPO,    # TODO: Implement PPO
}

# Profiles for the weight-editing controller. ``--rl-type`` selects one, and the
# profile bundles everything that defines the variant - the underlying policy,
# the controller behavior (edit_mode, selector), and the credit-assignment knobs
# (period, horizon, warmup, reward_decay) - so an experiment sets a single key
# instead of a soup of rl_* flags. Mirrors the memory/orchestration registries.
# These are profile defaults; an experiment may still override any one via the
# matching rl_* config key (the builder falls back to the profile value).
RL_PROFILES = {
    "harmonic_weight": dict(
        policy="harmonic_weight",
        edit_mode="harmonic",
        selector="sinusoidal",
        period=50,
        horizon=20,
        warmup_steps=200,
        reward_decay=0.9,
    ),
    # Drives HalfLion's wave gate (amp, cycles, phase) per episode instead of
    # editing weight rows (calm-c). A non-helpful change restores the three
    # scalars - no weight surgery. Small localized harmonic edits manifest
    # slowly, so it credits them over a long window: a 100-step horizon with a
    # matched ~100-step EMA (1/(1-0.99)=100) accumulates the delayed effect
    # rather than snapshotting a noisy endpoint.
    "harmonic_weight_wave": dict(
        policy="harmonic_weight",
        edit_mode="wave",
        selector="sinusoidal",
        period=50,
        horizon=100,
        warmup_steps=200,
        reward_decay=0.99,
    ),
    # Hash-gated frozen-anchor weight replacement (see next/hash_gated_anchor.md).
    "harmonic_weight_anchor": dict(
        policy="harmonic_weight",
        edit_mode="anchor_gate",
        selector="sinusoidal",
        period=50,
        horizon=20,
        warmup_steps=200,
        reward_decay=0.9,
    ),
}


def get_rl_profile(name):
    """Resolve an ``rl_type`` to its profile dict, or None if it isn't a
    weight-editing profile (e.g. reinforce/grpo/cot run on the forward path)."""
    return RL_PROFILES.get(name)


def normalize_rl_types(rl_type):
    """Coerce an ``rl_type`` config value to a list of policy/profile names.

    Accepts None, a single name, a comma-separated string, or a list. Multiple
    entries declare multiple discrete RL tasks that coexist - e.g. a forward-path
    reward policy alongside a weight-editing controller.
    """
    if rl_type is None:
        return []
    if isinstance(rl_type, str):
        return [s.strip() for s in rl_type.split(",") if s.strip()]
    return [str(s).strip() for s in rl_type if str(s).strip()]


def _policy_for(name):
    """Resolve an ``rl_type`` name (policy or profile key) to its policy class."""
    profile = get_rl_profile(name)
    policy_key = profile["policy"] if profile else name
    return RL_POLICIES_REGISTRY.get(policy_key)


def resolves_to_weight_controller(name):
    """Whether an ``rl_type`` name maps to a weight-editing controller (driven by
    a callback, not the forward pass)."""
    return bool(getattr(_policy_for(name), "is_weight_controller", False))


def needs_rl_datasets(name):
    """Whether an ``rl_type`` name requires the RL/cot data collection. Weight
    controllers reward from a callback and some forward policies (e.g.
    engagement) compute their own reward from labels, so neither needs RL data.
    Unknown names fail open (assume they do)."""
    cls = _policy_for(name)
    if cls is None:
        return True
    if getattr(cls, "is_weight_controller", False):
        return False
    return bool(getattr(cls, "needs_rl_datasets", True))


__all__ = [
    "REINFORCE",
    "GRPO",
    "ChainOfThought",
    "HarmonicWeightPolicy",
    "RL_POLICIES_REGISTRY",
    "RL_PROFILES",
    "get_rl_profile",
    "normalize_rl_types",
    "resolves_to_weight_controller",
    "needs_rl_datasets",
    "EngagementPolicy",
    "JokePolicy",
]
