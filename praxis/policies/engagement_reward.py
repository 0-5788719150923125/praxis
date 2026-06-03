"""Engagement-prediction reward: pure math for the `Print` policy (PLAN.md).

The model leads with a question and predicts the user's answer; the reward is how
well the prediction matches the actual response (recall, "mentioned at all"), fed
to a slow homeostatic "energy" variable that the policy is rewarded for keeping
near a setpoint. THE RULE: this rewards prediction/correctness, never raw
engagement. All constants are fixed and model-agnostic (no per-experiment tuning).

Kept dependency-free and side-effect-free so it unit-tests in isolation.
"""

from typing import Iterable

# Homeostatic energy constants. decay sets a long ~1000-step horizon (the
# "multi-hour" feel); gain is large so a single activation jumps the energy fast;
# (1 - E/E_MAX) is the satiating, diminishing-returns term that bounds spam.
ENERGY_DECAY = 0.999
ENERGY_GAIN = 0.5
ENERGY_MAX = 1.0


def activation(predicted: Iterable[int], response: Iterable[int]) -> float:
    """1.0 if any predicted answer token is mentioned in the response, else 0.0.

    The headline signal: "mentioned AT ALL". Order- and count-insensitive.
    """
    pred, resp = set(predicted), set(response)
    return 1.0 if pred & resp else 0.0


def recall(predicted: Iterable[int], response: Iterable[int]) -> float:
    """Graded overlap |pred & resp| / max(1, |pred|) for a smoother gradient.

    Fraction of the predicted answer tokens that the response mentions.
    """
    pred, resp = set(predicted), set(response)
    if not pred:
        return 0.0
    return len(pred & resp) / len(pred)


class HomeostaticEnergy:
    """Slow EMA "energy" with fast accumulation and satiation.

    ``E <- decay * E + gain * a * (1 - E/E_max)`` - a single activation jumps E
    fast, repeated activations saturate toward the setpoint, and E decays over a
    long horizon so the policy is drawn back to seeking the next genuine
    prediction. Used as the REINFORCE baseline.
    """

    def __init__(
        self,
        decay: float = ENERGY_DECAY,
        gain: float = ENERGY_GAIN,
        e_max: float = ENERGY_MAX,
        init: float = 0.0,
    ):
        self.decay = float(decay)
        self.gain = float(gain)
        self.e_max = float(e_max)
        self.value = float(init)

    def update(self, activation_rate: float) -> float:
        """Fold one activation (or a batch's mean activation in [0, 1]) in;
        returns the new energy."""
        a = max(0.0, min(1.0, float(activation_rate)))
        satiation = 1.0 - self.value / self.e_max
        self.value = self.decay * self.value + self.gain * a * satiation
        return self.value
