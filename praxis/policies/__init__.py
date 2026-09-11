"""
Reinforcement Learning policies for training language models.
"""

from praxis import registry
from praxis.policies.cot import ChainOfThought
from praxis.policies.engagement import EngagementPolicy, JokePolicy
from praxis.policies.grpo import GRPO
from praxis.policies.harmonic_weight_rl import HarmonicWeightPolicy
from praxis.policies.preference import PreferencePolicy
from praxis.policies.reinforce import REINFORCE
from praxis.registry import Entry

registry.declare(
    "rl_policies",
    title="RL policies",
    doc=(
        (
            "Reinforcement-learning policy losses for post-training: forward-path policies "
            "that reweight the LM's own log-probabilities, and weight-editing controllers "
            "driven by a callback. The same flag also takes the controller profiles in "
            "``rl_profiles``, and several names may be combined with commas."
        )
    ),
    entries={
        "reinforce": Entry(
            REINFORCE,
            (
                "REINFORCE on reasoning traces, rewarded by the RL dataset's "
                "solve_rate scores."
            ),
        ),
        "grpo": Entry(
            GRPO,
            (
                "Group Relative Policy Optimization (DeepSeekMath): outcome-level "
                "rewards with group-normalized advantages in place of a value network, "
                "plus a KL penalty against reward hacking. Rewards are the dataset's "
                "static solve_rate scores."
            ),
        ),
        "cot": Entry(
            ChainOfThought,
            (
                "Supervised chain-of-thought training: rewards structured reasoning "
                "(thinking tags) and weights reasoning steps above the final answer."
            ),
        ),
        "engagement": Entry(
            EngagementPolicy,
            (
                "Forward-path engagement-prediction reward: the model's recall of its "
                "own assistant-region answer tokens, plus the homeostatic energy, "
                "reweights the LM's log-probs over those tokens. The reward comes from "
                "labels, so it needs no RL dataset."
            ),
        ),
        "joke": Entry(
            JokePolicy,
            (
                "Forward-path reward for the joke task, on engagement's machinery: "
                "dense grounding from well-rated jokes in the data mix, and a live "
                "signal from human approval through the Loop UI."
            ),
        ),
        "preference": Entry(
            PreferencePolicy,
            (
                "Forward-path reference-free preference margin over chosen/rejected "
                "task tags: pushes the mean per-token log-probability of chosen text "
                "above that of rejected text. It is the preference-modeling use that "
                "the hh-rlhf dataset card permits."
            ),
        ),
        "harmonic_weight": Entry(
            HarmonicWeightPolicy,
            (
                "Weight-editing controller: a small Gaussian policy that proposes "
                "harmonic edits (alpha, omega, phi) to one weight row at a time and is "
                "rewarded by the loss improvement that follows. Driven by a callback, "
                "not the forward pass; its ``rl_profiles`` entries set the edit mode "
                "and credit-assignment knobs."
            ),
        ),
    },
)

registry.declare(
    "rl_profiles",
    title="RL controller profiles",
    doc=(
        (
            "Profiles for the weight-editing RL controller, selected alongside the RL "
            "policies (``rl_policies``). Each bundles everything that defines a variant - "
            "the underlying policy, the controller behavior (edit_mode, selector) and the "
            "credit-assignment knobs (period, horizon, warmup, reward_decay) - so an "
            "experiment sets one key instead of a set of rl_* flags. The values are "
            "defaults: an experiment may override any one through the matching rl_* config "
            "key."
        )
    ),
    entries={
        "harmonic_weight": Entry(
            dict(
                policy="harmonic_weight",
                edit_mode="harmonic",
                selector="sinusoidal",
                period=50,
                horizon=20,
                warmup_steps=200,
                reward_decay=0.9,
            ),
            (
                "The harmonic_weight controller at its defaults: every 50 steps after "
                "a 200-step warmup it modulates one weight row with a sinusoid, "
                "credits the edit over a 20-step horizon, and keeps or rolls it back."
            ),
        ),
        "harmonic_weight_wave": Entry(
            dict(
                policy="harmonic_weight",
                edit_mode="wave",
                selector="sinusoidal",
                period=50,
                horizon=100,
                warmup_steps=200,
                reward_decay=0.99,
            ),
            (
                "Drives the wave gate (amp, cycles, phase) of a wave-bearing optimizer "
                "wrapper (``half_lion`` or ``wave_schedule_free``) per episode instead "
                "of editing weight rows; an unhelpful change restores the three "
                "scalars, with no weight surgery. Small localized harmonic edits "
                "manifest slowly, so it credits them over a long window: a 100-step "
                "horizon with a matched ~100-step EMA (1/(1-0.99)) accumulates the "
                "delayed effect rather than snapshotting a noisy endpoint."
            ),
        ),
        "harmonic_weight_anchor": Entry(
            dict(
                policy="harmonic_weight",
                edit_mode="anchor_gate",
                selector="sinusoidal",
                period=50,
                horizon=20,
                warmup_steps=200,
                reward_decay=0.9,
            ),
            (
                "Hash-gated frozen-anchor weight replacement: instead of modulating a "
                "row, each edit resets a gated subset of it to a frozen snapshot of "
                "the weights taken at the end of warmup."
            ),
        ),
    },
)


def get_rl_profile(name):
    """Resolve an ``rl_type`` to its profile dict, or None if it isn't a
    weight-editing profile (e.g. reinforce/grpo/cot run on the forward path)."""
    return registry.namespace("rl_profiles").get(name)


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
    return registry.namespace("rl_policies").get(policy_key)


def resolves_to_weight_controller(name):
    """Whether an ``rl_type`` name maps to a weight-editing controller (driven by
    a callback, not the forward pass)."""
    return bool(getattr(_policy_for(name), "is_weight_controller", False))


def rl_dataset_collections(name):
    """Named collections an ``rl_type`` entry needs in ``train_datasets``.

    Every RL policy here is bound to particular data, and the binding used to
    be invisible: ``rl_type`` and ``train_datasets`` are separate config keys,
    so an experiment could name a policy and simply not get it. The failure is
    silent rather than loud - a policy filters by task tag (``task_types``,
    ``PREF_CHOSEN``/``PREF_REJECTED``) or by a reward field, and when nothing in
    the mix carries that tag it scores zero positions, emits no loss, and logs
    nothing to distinguish "off" from "on but starved".

    Declaring the binding on the policy class lets ``get_dataset_configs`` pull
    the collection in automatically, so an experiment sets ``rl_type`` alone and
    the pairing cannot drift. Weight-editing controllers reward from a callback
    and declare none.

    Unregistered names (the legacy ``cot-reinforce``) keep the historical
    fail-open mapping rather than resolving to nothing, since loading no data
    is the failure this function exists to prevent.

    A policy that owns its data outright declares ``dataset_weights`` instead;
    see :func:`rl_dataset_weights`.
    """
    cls = _policy_for(name)
    if cls is None:
        return ("cot",) if "cot" in name else ("rl",)
    if getattr(cls, "is_weight_controller", False):
        return ()
    return tuple(getattr(cls, "dataset_collections", ()) or ())


def rl_dataset_weights(name):
    """Individual datasets an ``rl_type`` entry injects, as ``{id: weight}``.

    The dataset-level half of the same binding: a policy names DATASETS keys
    directly rather than a shared collection, which is what a dataset with
    only one legitimate consumer needs. ``Anthropic/hh-rlhf`` is the case -
    its card permits preference modeling only, so it cannot sit in a general
    collection where any experiment picks it up as conversation data. Its
    entry declares ``requires_rl_type`` and ``add_collection`` refuses it, so
    the policy's declaration here is the only door in.

    Weight controllers reward from a callback and inject nothing.
    """
    cls = _policy_for(name)
    if cls is None or getattr(cls, "is_weight_controller", False):
        return {}
    return dict(getattr(cls, "dataset_weights", {}) or {})


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
    "get_rl_profile",
    "normalize_rl_types",
    "resolves_to_weight_controller",
    "needs_rl_datasets",
    "rl_dataset_collections",
    "rl_dataset_weights",
    "EngagementPolicy",
    "JokePolicy",
]
