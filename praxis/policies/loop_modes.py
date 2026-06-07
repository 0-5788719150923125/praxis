"""Loop query modes: how a looped task is asked, parsed, and scored.

The Loop UI is a generic live-RL interface; this registry defines the looping
patterns it can host. Each mode owns the full contract between the model's
generation and the human signal: the task prompt, how the raw generation is
parsed (e.g. stripping a self-predicted score), and how (user_score, predicted)
becomes the channel's (activation, reward) pair. The route resolves the mode
from the active RL policy's ``loop_mode`` attribute, defaulting to
``calibration``. All constants are fixed and model-agnostic.
"""

import re
from typing import Optional, Tuple

# A trailing line is only a self-predicted score if it's a bare number in (a
# tolerance of) the slider range - a year or a phone number in joke content
# must read as content, not a wildly-clamped prediction.
_SCORE_LINE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_SCORE_TOLERANCE = 1.5


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


class ApprovalLoopMode:
    """The original behavior: the human's signed want->need score is taken at
    face value. Activation = (score+1)/2 (any engagement sustains), reward = the
    signed score (raw approval valence)."""

    name = "approval"

    def task_prompt(self, task: str) -> str:
        # "joke" is the headline task keyword; anything else is a literal prompt.
        return "Tell me a joke." if task == "joke" else task

    def parse(self, reply: str) -> Tuple[str, Optional[float]]:
        """(display_text, predicted score or None)."""
        return reply.strip(), None

    def score(self, user_score: float, predicted: Optional[float]) -> dict:
        s = _clamp(user_score)
        return {"activation": (s + 1.0) / 2.0, "reward": s, "extra": {}}


class CalibrationLoopMode(ApprovalLoopMode):
    """The model appends a self-predicted want->need score to its output; the
    human corrects it, and the CORRECTION MAGNITUDE is the signal: activation =
    1 - |user - predicted| / 2, so less correction = more energy = more reward
    (the Print rule - reward prediction/correctness, never raw engagement). The
    signed user score is kept as the logged valence so "accurately predicted
    awful" sustains energy without reinforcing awful. While the model can't yet
    emit a parseable score, scoring degrades to approval semantics - early
    reward is plain approval and anneals into calibration as the format lands.
    """

    name = "calibration"

    def parse(self, reply: str) -> Tuple[str, Optional[float]]:
        text = reply.strip()
        lines = text.splitlines()
        if len(lines) >= 2 and _SCORE_LINE.match(lines[-1].strip()):
            try:
                raw = float(lines[-1])
            except ValueError:
                return text, None
            if abs(raw) <= _SCORE_TOLERANCE:
                return "\n".join(lines[:-1]).rstrip(), _clamp(raw)
        return text, None

    def score(self, user_score: float, predicted: Optional[float]) -> dict:
        s = _clamp(user_score)
        if predicted is None:
            return super().score(s, None)
        correction = abs(s - predicted)
        return {
            "activation": 1.0 - correction / 2.0,
            "reward": s,
            "extra": {"predicted": predicted, "correction": correction},
        }


LOOP_MODE_REGISTRY = {
    mode.name: mode for mode in (CalibrationLoopMode(), ApprovalLoopMode())
}
DEFAULT_LOOP_MODE = "calibration"


def get_loop_mode(name: Optional[str] = None):
    return LOOP_MODE_REGISTRY.get(
        name or DEFAULT_LOOP_MODE, LOOP_MODE_REGISTRY[DEFAULT_LOOP_MODE]
    )
