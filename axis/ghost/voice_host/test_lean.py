"""The LEAN: a marked stretch's small move in register and effort, on top of the discourse plan.

    <venv>/bin/python voice_host/test_lean.py

Two-sided, on the plan alone (no checkpoint needed): a lean moves every sentence's semitones and
effort by exactly what was asked and nothing else - the rate and the timing variety are
untouched - and an absurd lean is clamped to a lean.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from backends.piper import _discourse_plan  # noqa: E402

groups = [[{"text": "one", "punct": "."}], [{"text": "two", "punct": "."}]]
base = {
    "dynamics": 0.6,
    "prosody_arc": 1.2,
    "effort": 0.4,
    "plan_u": 0.5,
    "plan_v": 0.2,
}
fails = []


def check(cond, what):
    if not cond:
        fails.append(what)


plain = _discourse_plan(groups, base)
lean = _discourse_plan(groups, dict(base, lean_semis=0.7, lean_effort=0.1))
for a, b in zip(plain, lean):
    check(
        abs((b["semis"] - a["semis"]) - 0.7) < 1e-9,
        "the lean did not move the register by 0.7 st",
    )
    check(
        abs(b["tilt"] - a["tilt"] - 0.45 * 0.1) < 1e-9
        and abs(b["gain_db"] - a["gain_db"] - 2.6 * 0.1) < 1e-9,
        "the lean did not move the effort by exactly what was asked",
    )
    check(
        b["rate"] == a["rate"] and b["noise_w_mul"] == a["noise_w_mul"],
        "the lean touched the timing",
    )
wild = _discourse_plan(groups, dict(base, lean_semis=9.0, lean_effort=-4.0))
check(
    abs((wild[0]["semis"] - plain[0]["semis"]) - 1.5) < 1e-9,
    "a wild lean was not clamped to 1.5 st",
)
print("test_lean: %s" % ("ALL OK" if not fails else "FAILED - " + "; ".join(fails)))
sys.exit(1 if fails else 0)
