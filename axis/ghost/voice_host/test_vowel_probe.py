#!/usr/bin/env python3
"""Tests for the vowel probe (vowel_probe.py).

Run it directly with the voice venv's python - no pytest needed:

    ~/.local/share/godot/app_userdata/ghost/voice_venv/bin/python \
        axis/ghost/voice_host/test_vowel_probe.py

Two halves, and the second is the one that matters.

  THE REWRITE - what `repair` does to a phoneme string, which is pure text and
  can be checked exactly. A stressed /iː/ before a consonant gets the offglide
  English already says there; a word-final one, an unstressed one and every other
  vowel are left alone, and running it twice changes nothing.

  THE VERDICT - whether the probe correctly separates a checkpoint that needs the
  repair from four that do not. This is the half that would make the feature worth
  having or not, and it is measured against the real models, because a fake one
  would be answering a different question: the whole claim is about what a
  particular set of weights does with a particular phoneme string.

The verdict half needs the voices installed. If they are not, it says so and skips
that half rather than passing vacuously.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import vowel_probe  # noqa: E402

_checks = 0
_failed = 0


def check(name: str, got, want) -> None:
    global _checks, _failed
    _checks += 1
    ok = got == want
    if not ok:
        _failed += 1
    print(f"    {'ok ' if ok else 'FAIL'} {name} == {got!r}" + ("" if ok else f" (want {want!r})"))


# -- the rewrite -----------------------------------------------------------


def test_repair_writes_the_offglide() -> None:
    print("\ntest_repair_writes_the_offglide")
    # the string the whole module exists for
    check("read", vowel_probe.repair("ɹˈiːd"), "ɹˈiːjd")
    # and it is not keyed to that word - any stressed iː before a consonant
    check("healing", vowel_probe.repair("hˈiːlɪŋ"), "hˈiːjlɪŋ")
    check("receive", vowel_probe.repair("ɹᵻsˈiːv"), "ɹᵻsˈiːjv")
    check("beat", vowel_probe.repair("bˈiːt"), "bˈiːjt")


def test_repair_leaves_everything_else() -> None:
    print("\ntest_repair_leaves_everything_else")
    # WORD-FINAL. Nothing follows the vowel, so there is no consonant to glide
    # into and the lookahead simply does not match.
    check("free", vowel_probe.repair("fɹˈiː"), "fɹˈiː")
    check("he", vowel_probe.repair("hiː"), "hiː")
    # UNSTRESSED. A reduced vowel is supposed to be short; the measurement was
    # taken on the stressed one and the rewrite claims nothing about this.
    check("unstressed", vowel_probe.repair("ɹiːd"), "ɹiːd")
    check("secondary stress", vowel_probe.repair("ɹˌiːd"), "ɹˌiːd")
    # ANOTHER VOWEL, however similar it looks.
    check("moving", vowel_probe.repair("mˈuːvɪŋ"), "mˈuːvɪŋ")
    check("rid", vowel_probe.repair("ɹˈɪd"), "ɹˈɪd")
    # THE STRESS MARK HAS TO BE IN THIS WORD. A mark on the previous word does
    # not reach across the space that separates them.
    check("across a space", vowel_probe.repair("hˈaʊs iːd"), "hˈaʊs iːd")
    # ...and the offglide is not written twice.
    check("idempotent", vowel_probe.repair(vowel_probe.repair("ɹˈiːd")), "ɹˈiːjd")


def test_repair_is_gated_on_the_verdict() -> None:
    """`repair_for` is what the backend calls, and it does nothing until measured."""
    print("\ntest_repair_is_gated_on_the_verdict")
    p = vowel_probe.VowelProbe(None, None, HERE)
    check("unmeasured voice", p.repair_for("whatever", "ɹˈiːd"), "ɹˈiːd")
    p._verdict["healthy"] = False
    check("measured healthy", p.repair_for("healthy", "ɹˈiːd"), "ɹˈiːd")
    p._verdict["needs it"] = True
    check("measured broken", p.repair_for("needs it", "ɹˈiːd"), "ɹˈiːjd")


# -- the verdict -----------------------------------------------------------


def test_the_probe_picks_the_broken_checkpoint() -> None:
    """The claim, against the real weights.

    en_US-libritts-high renders a word-initial `ɹˈiːd` with the vowel of `rid` and the
    other four installed voices do not. On kristin the rewrite would actively hurt (F2
    2730 -> 2070), so a wrong verdict here is not a missed fix, it is damage.

    The probe renders deterministically (see PROBE_PARAMS), so this is not a statistical
    claim and is not allowed to be flaky - it returned the same five verdicts, from the
    same F1 measurements to four decimal places, over ten consecutive cold starts. If it
    ever wavers, the instrument has drifted, and that is the finding rather than noise.
    Measured ratios, |suspect - /iː/| over that reader's own /iː/-to-/ɪ/ distance:
    libritts 1.46, john 0.51, ljspeech 0.23, kristin 0.10, norman 0.02, against a
    threshold of 1.0.
    """
    print("\ntest_the_probe_picks_the_broken_checkpoint")
    from backends.piper import PiperBackend

    be = PiperBackend()
    want = {
        "en_US-libritts-high": True,
        "en_US-kristin-medium": False,
        "en_US-ljspeech-medium": False,
        "en_US-john-medium": False,
        "en_US-norman-medium": False,
    }
    installed = {
        v["id"] for v in be.voices() if v["installed"] and v["id"] in want
    }
    if not installed:
        print("    SKIP no piper voices installed; nothing to measure")
        return
    for voice, expect in want.items():
        if voice not in installed:
            print(f"    skip {voice} (not installed)")
            continue
        sess, cfg = be._load(voice)
        lang = str(cfg.get("espeak", {}).get("voice", "en-us"))
        # Straight to `_run`, past both caches: `measure` would hand back whatever
        # `_load` already decided and the test would be checking a dict lookup.
        check(voice, be._vowels._run(voice, cfg, sess, lang), expect)


def test_a_probe_fault_is_not_fatal() -> None:
    """Anything going wrong means no rewrite, never a lost sentence."""
    print("\ntest_a_probe_fault_is_not_fatal")

    import tempfile

    def explode(*_a, **_k):
        raise RuntimeError("no phonemizer today")

    # A real directory that is not the source tree: `measure` caches the verdict
    # even when the probe failed - a checkpoint that cannot be measured should not
    # be re-probed on every load - and that write has to land somewhere disposable.
    with tempfile.TemporaryDirectory() as tmp:
        p = vowel_probe.VowelProbe(explode, explode, tmp)
        check("verdict on failure", p.measure("broken", {}, None), False)
        check("and it rewrites nothing", p.repair_for("broken", "ɹˈiːd"), "ɹˈiːd")


if __name__ == "__main__":
    test_repair_writes_the_offglide()
    test_repair_leaves_everything_else()
    test_repair_is_gated_on_the_verdict()
    test_a_probe_fault_is_not_fatal()
    test_the_probe_picks_the_broken_checkpoint()
    print(f"\n{_checks - _failed}/{_checks} checks passed")
    raise SystemExit(1 if _failed else 0)
