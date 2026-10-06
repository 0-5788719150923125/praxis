#!/usr/bin/env python3
"""Tests for punctuation pauses in the Piper backend.

Run it directly with the voice venv's python - no pytest needed:

    ~/.local/share/godot/app_userdata/ghost/voice_venv/bin/python \
        axis/ghost/voice_host/test_pauses.py

Three things are being checked, and the third is the one that matters most:

  1. the splice maths - inserted length, placement, and the shifted timings
     still landing on the silence they describe;
  2. that the cut does not click, by construction rather than by ear;
  3. that `pause_scale` = 0 reproduces the PREVIOUS code byte for byte. That
     last one is run against the actual HEAD revision of piper.py in a
     subprocess, not against a remembered number, so it cannot rot.

Synthesis runs against a fake ONNX session: deterministic audio, deterministic
durations, no model download, no eSpeak. The real graph is stochastic, so it
could not answer a byte-identity question at all.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SR = 22050


# -- fake voice ------------------------------------------------------------


class _FakeMap:
    """A phoneme_id_map that knows every symbol, so nothing is ever dropped."""

    def __init__(self) -> None:
        self._ids: dict = {}

    def get(self, sym, default=None):
        if sym not in self._ids:
            self._ids[sym] = [len(self._ids) + 3]
        return self._ids[sym]


class _FakeSession:
    """Stands in for the ONNX graph. One duration frame per id, and a waveform
    with a 64-sample period so zero crossings are easy to reason about."""

    def run(self, _outputs, feeds):
        import numpy as np

        ids = list(feeds["input"][0])
        frames = np.array([2 + (int(i) % 3) for i in ids], dtype=np.float32)
        from backends.piper import HOP_LENGTH

        n = int(frames.sum()) * HOP_LENGTH
        t = np.arange(n, dtype=np.float64)
        audio = (0.8 * np.sin(2.0 * np.pi * t / 64.0)).astype(np.float32)
        return [audio.reshape(1, 1, -1), frames.reshape(1, -1)]


class _FloorSession(_FakeSession):
    """The fake voice behind a patched graph: takes the optional rest floor, as a Max on
    the duration plan, and records what was fed."""

    def __init__(self) -> None:
        self.fed: list = []

    def get_overridable_initializers(self):
        import types
        from backends.piper import REST_FLOOR_INPUT

        return [types.SimpleNamespace(name=REST_FLOOR_INPUT)]

    def run(self, _outputs, feeds):
        import numpy as np
        from backends.piper import HOP_LENGTH, REST_FLOOR_INPUT

        ids = list(feeds["input"][0])
        floor = feeds.get(REST_FLOOR_INPUT)
        self.fed.append((ids, None if floor is None else np.array(floor)))
        frames = np.array([2 + (int(i) % 3) for i in ids], dtype=np.float32)
        if floor is not None:
            frames = np.maximum(frames, floor)
        n = int(frames.sum()) * HOP_LENGTH
        t = np.arange(n, dtype=np.float64)
        audio = (0.8 * np.sin(2.0 * np.pi * t / 64.0)).astype(np.float32)
        return [audio.reshape(1, 1, -1), frames.reshape(1, -1)]


def _cfg() -> dict:
    return {
        "audio": {"sample_rate": SR},
        "phoneme_id_map": _FakeMap(),
        "num_speakers": 1,
        "inference": {},
        "espeak": {"voice": "en-us"},
    }


def _tok(text, punct, arpa):
    return {"text": text, "punct": punct, "fallback": arpa}


TOKENS_ONE = [_tok("one", "", ["W", "AH1", "N"]), _tok("word", ".", ["W", "ER1", "D"])]

TOKENS_TWO = [
    _tok("one", "", ["W", "AH1", "N"]),
    _tok("word", ".", ["W", "ER1", "D"]),
    _tok("then", "", ["DH", "EH1", "N"]),
    _tok("more", ".", ["M", "AO1", "R"]),
]

TOKENS_MARKS = [
    _tok("one", ",", ["W", "AH1", "N"]),
    _tok("two", ":", ["T", "UW1"]),
    _tok("three", ".", ["TH", "R", "IY1"]),
    _tok("four", "", ["F", "AO1", "R"]),
    _tok("five", "?", ["F", "AY1", "V"]),
]

# The OTHER front end: bare phones, marks inline (arpabet.PUNCT_PASSTHROUGH).
PHONES_ONE = ["W", "AH1", "N", "W", "ER1", "D", "."]
PHONES_MARKS = [
    "W",
    "AH1",
    "N",
    ",",
    "T",
    "UW1",
    ":",
    "TH",
    "R",
    "IY1",
    ".",
    "F",
    "AO1",
    "R",
    ".",
]

# name -> (kind, items, params). Params are chosen so everything but the two
# *_marks cases MUST reproduce the previous implementation byte for byte.
CASES = {
    "tok_one_sentence_scale0": ("tokens", TOKENS_ONE, {"pause_scale": 0.0}),
    "tok_two_sentences_default": ("tokens", TOKENS_TWO, {}),
    "tok_marks_scale1": ("tokens", TOKENS_MARKS, {"pause_scale": 1.0}),
    "ph_one_sentence_scale0": ("phones", PHONES_ONE, {"pause_scale": 0.0}),
    "ph_marks_scale1": ("phones", PHONES_MARKS, {"pause_scale": 1.0}),
}


def _result(data: bytes, res: dict) -> dict:
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "duration": res["duration"],
        "tokens": res.get("tokens", []),
        "phones": res.get("phones", []),
        "sentences": res.get("sentences", 1),
    }


def synth(tokens: list, params: dict) -> dict:
    """Run the real _synth_tokens against the fake voice."""
    from backends.piper import PiperBackend

    be = PiperBackend()
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "take.wav")
        res = be._synth_tokens(
            list(tokens),
            "fake",
            out,
            {"phonemizer": "ghost", **params},
            _cfg(),
            _FakeSession(),
        )
        return _result(Path(out).read_bytes(), res)


def synth_phones(phones: list, params: dict) -> dict:
    """Run the real synthesize() phones path against the fake voice.

    Pre-seeding the caches is what keeps _load() from wanting a real model.
    """
    from backends.piper import PiperBackend

    be = PiperBackend()
    be._sessions["fake"] = _FakeSession()
    be._configs["fake"] = _cfg()
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "take.wav")
        res = be.synthesize(
            "", "fake", out, {"phonemizer": "ghost", **params}, list(phones)
        )
        return _result(Path(out).read_bytes(), res)


def run_case(kind: str, items: list, params: dict) -> dict:
    return synth(items, params) if kind == "tokens" else synth_phones(items, params)


# -- helpers ---------------------------------------------------------------


def _sine(seconds: float, period: int = 64, amp: float = 0.8):
    import numpy as np

    t = np.arange(int(seconds * SR), dtype=np.float64)
    return (amp * np.sin(2.0 * np.pi * t / period)).astype(np.float32)


def _zero_runs(a, minlen: int = 8):
    """[(start, length)] of every run of exact digital zero."""
    import numpy as np

    z = np.concatenate(([0], (np.asarray(a) == 0.0).astype(np.int8), [0]))
    d = np.diff(z)
    return [
        (int(s), int(e - s))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
        if e - s >= minlen
    ]


CHECKS: list = []


def near(got, want, what, tol=1e-6):
    """`eq` for a value that comes off a curve rather than out of a table."""
    ok(abs(got - want) < tol, "%s == %.6f (got %.6f)" % (what, want, got))


def check(fn):
    CHECKS.append(fn)
    return fn


def eq(got, want, what: str):
    assert got == want, f"{what}: got {got!r}, want {want!r}"
    print(f"    ok  {what} == {want!r}")


def ok(cond, what: str):
    assert cond, f"FAILED: {what}"
    print(f"    ok  {what}")


def within(got, want, what: str, slop: int = 2):
    """`eq` for a SAMPLE COUNT that went through the waveform on its way here.

    The splicer measures the silence already at a mark rather than looking it up, and
    even this fake voice - a continuous sine - dips under the quiet threshold for a
    sample here and there. So a prediction made with `have = 0` lands a sample or two off
    the real insertion, every time, and always by the same tiny amount. That is the
    measurement working; demanding exactness here would only ever gate the sine's zero
    crossings.
    """
    ok(abs(got - want) <= slop, "%s == %r (got %r)" % (what, want, got))


# -- 1. the table and the scale -------------------------------------------


@check
def test_table_and_scale():
    from backends.piper import PAUSE_AFTER, _gap_for, _pause_for

    # AT 1.0 THE TABLE IS THE ANSWER, exactly - the dial's whole job is to leave the
    # calibrated default alone and this is where that is pinned down.
    eq(_pause_for(",", {}), PAUSE_AFTER[","], "comma at default scale")
    eq(_pause_for(";", {"pause_scale": 0.0}), 0.0, "semicolon at scale 0")
    eq(_pause_for("x", {}), 0.0, "a non-mark gets nothing")
    # the unification: no explicit sentence_gap means the table, and the table's
    # sentence-final value IS the old 0.32 default
    eq(_gap_for(".", {}), 0.32, "sentence gap default")
    eq(_gap_for("?", {}), 0.32, "question mark uses the same gap")
    eq(_gap_for(".", {"sentence_gap": 0.5}), 0.5, "explicit sentence_gap still wins")
    eq(_gap_for(".", {"pause_scale": "nonsense"}), 0.32, "a bad scale falls back")
    # ...AND AWAY FROM 1.0 IT IS NOT A MULTIPLICATION. These were `table x scale` and are
    # not any more, deliberately: the dial scales the whole rest (the model's own dwell
    # included, see DWELL) along a saturating curve, so our share of it is what is left
    # after the model's is subtracted back off. Doubling the dial does NOT double the
    # splice - it doubles neither, it moves the REST toward its ceiling, which is the
    # only reading a listener can hear as "the same pacing, slower".
    near(
        _pause_for(":", {"pause_scale": 2.0}),
        0.446739,
        "colon at 2.0 is the curve, not the table doubled",
    )
    near(
        _gap_for(".", {"pause_scale": 0.5}),
        0.128006,
        "under 1.0 the model is already resting most of it",
    )
    near(
        _gap_for(".", {"sentence_gap": 0.5, "pause_scale": 2.0}),
        0.923875,
        "an explicit sentence_gap is a top-up at 1.0 like any other, and rides the same curve",
    )
    eq(
        _gap_for(".", {"sentence_gap": 0.5, "pause_scale": 1.0}),
        0.5,
        "...and is itself, exactly, at 1.0",
    )


# -- 2. splice maths -------------------------------------------------------


@check
def test_splice_length_and_placement():
    from backends.piper import _splice_pauses

    audio = _sine(1.0)
    out, inserted = _splice_pauses(audio, [(0.25, 0.12), (0.60, 0.26)], SR)
    pad_a, pad_b = round(0.12 * SR), round(0.26 * SR)
    eq(int(out.size), int(audio.size) + pad_a + pad_b, "total length")
    eq(
        [round(d, 6) for _, d in inserted],
        [round(pad_a / SR, 6), round(pad_b / SR, 6)],
        "reported durations",
    )
    # The reported time is where the silence ACTUALLY went, not the nominal mark: the cut
    # slides forward to the quiet place between the words, and a timing shifted from the
    # nominal point would drift off the waveform by exactly that much.
    slack = round(3.0 * SR / 1000.0) / SR + 1e-9
    for got, want in zip([t for t, _ in inserted], [0.25, 0.60]):
        ok(
            want <= got <= want + slack,
            f"silence reported at/after its mark and within {slack * 1000:.0f} ms: {got:.4f}",
        )

    runs = _zero_runs(out)
    eq([n for _, n in runs], [pad_a, pad_b], "two silences, exact lengths")
    search = round(3.0 * SR / 1000.0)
    ok(
        abs(runs[0][0] - round(0.25 * SR)) <= search,
        f"first silence within {search} samples of its nominal point",
    )
    ok(
        abs(runs[1][0] - (round(0.60 * SR) + pad_a)) <= search,
        "second silence lands after the first shift",
    )


@check
def test_timings_still_land_on_the_silence():
    from backends.piper import _shift, _splice_pauses

    audio = _sine(1.0)
    # five contiguous 0.2 s tokens; the mark sits on token 1
    spans = {i: [round(i * 0.2, 4), round((i + 1) * 0.2, 4)] for i in range(5)}
    out, inserted = _splice_pauses(audio, [(spans[1][1], 0.26)], SR)
    pad = round(0.26 * SR)
    moved = {
        i: (_shift(s[0], inserted, True), _shift(s[1], inserted, False))
        for i, s in spans.items()
    }
    eq(moved[1], (0.2, 0.4), "the mark's own token does not move")
    eq(moved[2], (0.4 + pad / SR, 0.6 + pad / SR), "the next token moves by the pad")
    eq(
        round(moved[4][1] - spans[4][1], 6),
        round(pad / SR, 6),
        "the last token moves by exactly one pad",
    )

    # and the gap those timings now describe really is silent in the audio
    import numpy as np

    guard = round(6.0 * SR / 1000.0)  # search window + ramp
    a = int(moved[1][1] * SR) + guard
    b = int(moved[2][0] * SR) - guard
    ok(b > a, "the described gap is wider than the guard band")
    ok(
        float(np.abs(out[a:b]).max()) == 0.0,
        "the audio between the two shifted timings is digital silence",
    )
    ok(
        float(np.abs(out[: int(moved[1][1] * SR) - guard]).max()) > 0.5,
        "the speech before it is untouched",
    )


@check
def test_nothing_to_insert_is_a_no_op():
    from backends.piper import _splice_pauses

    audio = _sine(0.2)
    out, inserted = _splice_pauses(audio, [(0.1, 0.0), (0.15, 0.0)], SR)
    ok(out is audio, "scale 0 returns the SAME array, not a copy")
    eq(inserted, [], "nothing reported as inserted")
    out2, _ = _splice_pauses(audio, [], SR)
    ok(out2 is audio, "an empty point list is a no-op too")


# -- 3. clicks -------------------------------------------------------------


def _splatter(out, at: int, sr: int = SR) -> float:
    """Share of power above 1.5 kHz in the 80 ms around sample `at`, Hann-windowed.

    The test signals are a 345 Hz sine, so everything up there is what an edit made.
    """
    import numpy as np

    half = int(0.04 * sr)
    seg = np.asarray(out[max(0, at - half) : at + half], dtype=np.float64)
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    return float(spec[freqs > 1500.0].sum() / max(spec.sum(), 1e-30))


def _old_ramps():
    """A context in which every edge gets the 2 ms ramp, whatever is sounding there."""
    import contextlib
    import backends.piper as P

    @contextlib.contextmanager
    def ctx():
        keep = (P.SPLICE_RELEASE_MS, P.SPLICE_ATTACK_MS)
        P.SPLICE_RELEASE_MS = P.SPLICE_ATTACK_MS = P.SPLICE_FADE_MS
        try:
            yield
        finally:
            P.SPLICE_RELEASE_MS, P.SPLICE_ATTACK_MS = keep

    return ctx()


@check
def test_a_cut_in_silence_stays_short():
    import numpy as np
    from backends.piper import _splice_pauses

    # speech, 120 ms of silence, speech: the window spans the silence and some speech
    audio = _sine(0.6)
    a, b = round(0.25 * SR), round(0.37 * SR)
    audio[a:b] = 0.0
    out, inserted = _splice_pauses(audio, [(0.25, 0.12, 0.40, None, 0.22, 0.40)], SR)
    start, length = _zero_runs(out)[0]
    ok(length > b - a, "silence was added (%d samples)" % (length - (b - a)))
    ok(
        abs(inserted[0][0] - 0.5 * (a + b) / SR) < 0.005,
        "the cut is in the middle of the silence the voice left (%.4f)" % inserted[0][0],
    )
    ok(
        np.array_equal(out[:a], audio[:a]),
        "speech before the silence is bit-identical: the ramp only touched silence",
    )
    ok(np.array_equal(out[start + length :], audio[b:]), "speech after it is bit-identical")


@check
def test_a_cut_in_sound_is_released():
    """The voice running straight through the mark: the edges ramp for what is sounding.

    Two-sided on the same signal and the same cut: 2 ms ramps on every edge must splatter
    (the click), and the level-sized release and onset must not.
    """
    from backends.piper import SPLICE_ATTACK_MS, SPLICE_RELEASE_MS, _splice_pauses

    audio = _sine(0.6, amp=0.5)
    point = [(0.30, 0.12, 0.30, None, 0.28, 0.32)]
    new, _ = _splice_pauses(audio, point, SR)
    with _old_ramps():
        old, _ = _splice_pauses(audio, point, SR)
    s_new, n_new = _zero_runs(new)[0]
    s_old, n_old = _zero_runs(old)[0]
    out_new, out_old = _splatter(new, s_new), _splatter(old, s_old)
    in_new, in_old = _splatter(new, s_new + n_new), _splatter(old, s_old + n_old)
    base = _splatter(audio, s_old)
    print(
        "      splatter out %.1e -> %.1e, in %.1e -> %.1e (unedited %.1e)"
        % (out_old, out_new, in_old, in_new, base)
    )
    ok(min(out_old, in_old) > 100.0 * base, "the 2 ms ramps splatter (control)")
    ok(out_new < out_old / 100.0, "the release splatters >20 dB less than a 2 ms ramp")
    ok(in_new < in_old / 30.0, "the onset splatters >15 dB less than a 2 ms ramp")
    # ...and the ramps are as long as the level asks for: this sine is past SPLICE_LOUD
    import numpy as np

    tail = np.abs(new[s_new - round(0.5 * SPLICE_RELEASE_MS * SR / 1000.0) : s_new])
    ok(float(tail.max()) < 0.5 * 0.55, "the last half of the release is under half level")
    head = np.abs(new[s_new + n_new : s_new + n_new + round(0.25 * SPLICE_ATTACK_MS * SR / 1000.0)])
    ok(float(head.max()) < 0.5 * 0.2, "the first quarter of the onset stays low")


@check
def test_the_cut_finds_the_dip():
    """Inside its window the cut goes where the voice is quietest, never outside it."""
    import numpy as np
    from backends.piper import _splice_pauses

    audio = _sine(0.6, amp=0.5)
    t = np.arange(audio.size) / SR
    dip = 0.335  # between two words, 25 ms after the nominal mark
    audio *= (1.0 - 0.9 * np.exp(-(((t - dip) / 0.006) ** 2))).astype(np.float32)
    out, inserted = _splice_pauses(audio, [(0.31, 0.12, 0.31, None, 0.29, 0.36)], SR)
    start, _ = _zero_runs(out)[0]
    ok(abs(start / SR - dip) < 0.004, "cut at %.4f s, the dip is at %.4f" % (start / SR, dip))
    eq([round(a, 4) for a, _ in inserted], [0.31], "reported at the mark, for the timings")
    # a dip outside the window is not reachable
    out2, _ = _splice_pauses(audio, [(0.31, 0.12, 0.31, None, 0.29, 0.32)], SR)
    start2, _ = _zero_runs(out2)[0]
    ok(0.29 <= start2 / SR <= 0.32, "cut stays in its window (%.4f)" % (start2 / SR))


@check
def test_a_long_rest_is_trimmed_to_the_target():
    """A rest longer than its target gives the excess back, from the middle of the silence.

    Two-sided on one signal: the same 0.40 s rest grows to a 0.60 s target and shrinks to
    a 0.20 s one, the speech either side bit-identical both ways.
    """
    import numpy as np
    from backends.piper import _shift, _splice_pauses

    audio = _sine(0.9)
    a, b = round(0.25 * SR), round(0.65 * SR)
    audio[a:b] = 0.0

    def rest(dwell, top_up):  # target = (dwell + top_up) * mult, mult 1
        return _splice_pauses(audio, [(0.65, top_up, 0.65, dwell, 0.24, 0.66)], SR)

    grown, _ = rest(0.30, 0.30)
    shrunk, cut = rest(0.10, 0.10)
    for out, want, what in ((grown, 0.60, "grows"), (shrunk, 0.20, "shrinks")):
        runs = _zero_runs(out)
        eq(len(runs), 1, "one rest after it %s" % what)
        start, length = runs[0]
        within(length, round(want * SR), "the rest %s to its target" % what, slop=3)
        ok(np.array_equal(out[:a], audio[:a]), "speech before it is bit-identical (%s)" % what)
        ok(np.array_equal(out[start + length :], audio[b:]), "speech after it is bit-identical (%s)" % what)
    eq(len(cut), 1, "one removal reported")
    at, dur = cut[0]
    ok(dur < 0.0 and 0.25 < at < 0.65, "it is a removal inside the silence (%.4f, %.4f)" % (at, dur))
    near(_shift(0.80, cut, True), 0.80 + dur, "a time after the removal moves back by all of it")
    near(_shift(at + 0.5 * -dur, cut, False), at, "a time inside it lands where it starts")
    near(_shift(at, cut, True), at, "a time exactly at its start stays put")
    near(_shift(0.65, cut, False), _shift(0.65, cut, True), "a mark's end and the next start agree")


@check
def test_the_floor_goes_on_the_space_after_a_mark():
    """The model is asked to rest on the word-space after a mark that gets a pause.

    Never on the lead-in spaces, an unmarked space or after a sentence end; nothing at
    Pause 0; and nothing fed to a graph that cannot take it (two-sided: the same text).
    """
    from backends.piper import HOP_LENGTH, PiperBackend, _rest_floor

    def fed(sess, params):
        cfg = _cfg()
        with tempfile.TemporaryDirectory() as td:
            PiperBackend()._synth_tokens(
                list(TOKENS_MARKS), "fake", str(Path(td) / "a.wav"),
                {"phonemizer": "ghost", **params}, cfg, sess,
            )
        return cfg["phoneme_id_map"], sess.fed

    pmap, runs = fed(_FloorSession(), {"pause_scale": 1.0})
    ids, floor = runs[0]
    comma, colon, space = pmap.get(",")[0], pmap.get(":")[0], pmap.get(" ")[0]
    want = {i + 2: m for i, x in enumerate(ids) for m, c in ((",", comma), (":", colon)) if x == c}
    ok(all(ids[i] == space for i in want), "two ids past each mark is its word-space")
    got = {i: float(f) for i, f in enumerate(floor) if f > 0.0}
    frame = HOP_LENGTH / float(SR)
    eq(got, {i: float(round(_rest_floor(m, {"pause_scale": 1.0}) / frame)) for i, m in want.items()},
       "frames on the space after , and : only")
    eq(runs[1][1], None, "the sentence after the full stop has no paused mark: nothing fed")
    _, runs = fed(_FloorSession(), {"pause_scale": 0.0})
    ok(all(f is None for _, f in runs), "nothing at Pause 0")

    class _Plain(_FakeSession):
        fed: list = []

        def run(self, outputs, feeds):
            self.fed.append((list(feeds["input"][0]), feeds.get("rest_floor")))
            return super().run(outputs, feeds)

    _, runs = fed(_Plain(), {"pause_scale": 1.0})
    ok(all(f is None for _, f in runs), "a graph without the input is never fed it")


@check
def test_a_floored_id_is_fixed_in_the_nominal_length():
    """An id held at its floor does not scale with the length scale; one past it does."""
    from backends.piper import HOP_LENGTH, _nominal_seconds

    frames = [1.0, 5.0, 13.0, 20.0]
    floor = [0.0, 0.0, 13.0, 13.0]
    with_floor = _nominal_seconds(frames, 1.25, SR, floor)
    without = _nominal_seconds(frames, 1.25, SR)
    ok(with_floor > without, "the floored id keeps its frames at another scale")
    gap = (with_floor - without) * SR / HOP_LENGTH
    ok(2.0 < gap < 3.0, "by what 13 frames lose at 1.25x, and only that id (%.2f)" % gap)
    near(_nominal_seconds(frames, 1.0, SR, floor), _nominal_seconds(frames, 1.0, SR),
         "no difference at ratio 1")


@check
def test_the_patch_on_a_small_graph():
    """`_ensure_patched` on a graph shaped like Piper's around its ceiling.

    Exposes the plan, adds the optional floor that every reader of the plan sees, changes
    nothing when the floor is not fed, and leaves a patched file byte-identical.
    """
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError as exc:
        print("    -- onnx/onnxruntime missing (%s); skipping" % exc)
        return
    from backends.piper import REST_FLOOR_INPUT, PiperBackend

    graph = helper.make_graph(
        [
            helper.make_node("Ceil", ["x"], ["w"]),
            helper.make_node("ReduceSum", ["w"], ["total"], keepdims=0),
        ],
        "plan",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, "n"])],
        [helper.make_tensor_value_info("total", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # what Piper exports; a newer default can outrun onnxruntime
    x = np.array([[[0.2, 1.5, 2.0]]], np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "v.onnx"
        onnx.save(model, str(path))
        PiperBackend._ensure_patched(path)
        once = path.read_bytes()
        PiperBackend._ensure_patched(path)
        eq(path.read_bytes() == once, True, "a patched graph is left alone")
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    eq([o.name for o in s.get_outputs()], ["total", "w"], "the plan is an output")
    eq([i.name for i in s.get_overridable_initializers()], [REST_FLOOR_INPUT], "the floor is optional")
    total, w = s.run(None, {"x": x})
    eq(np.asarray(w).ravel().tolist(), [1.0, 2.0, 2.0], "unfed, the plan is the ceiling")
    total, w = s.run(None, {"x": x, REST_FLOOR_INPUT: np.array([0.0, 5.0, 0.0], np.float32)})
    eq(np.asarray(w).ravel().tolist(), [1.0, 5.0, 2.0], "fed, the plan is floored")
    eq(float(total), 8.0, "and the graph's own reader of the plan sees the floor")


@check
def test_rest_points_for_holds():
    """Every word but a chunk's last says where a rest after it goes - in the gap.

    GenerativeEditor._splice_holds puts hesitations and action holds there. After a mark
    it is inside the silence already spliced in; after a sentence, in the gap before the
    next; between two words, between them.
    """
    import numpy as np
    from backends.piper import PiperBackend

    be = PiperBackend()
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "take.wav")
        res = be._synth_tokens(
            list(TOKENS_MARKS),
            "fake",
            out,
            {"phonemizer": "ghost", "pause_scale": 1.0},
            _cfg(),
            _FakeSession(),
        )
        raw = Path(out).read_bytes()
    audio = np.frombuffer(raw[44:], "<i2").astype(np.float32) / 32768.0
    rows = {t["index"]: t for t in res["tokens"]}
    eq(sorted(i for i, t in rows.items() if "rest" in t), [0, 1, 2, 3], "who carries a rest")
    for i, why in ((0, "after the comma"), (1, "after the colon"), (2, "between sentences")):
        at = int(round(rows[i]["rest"] * SR))
        ok(float(np.abs(audio[at - 5 : at + 5]).max()) == 0.0, "the rest %s is in silence" % why)
    ok(
        rows[3]["t0"] < rows[3]["rest"] <= rows[4]["t0"] + 0.02,
        "an unmarked boundary's rest is between its words (%.4f)" % rows[3]["rest"],
    )


# -- 4. the real synthesis path -------------------------------------------


@check
def test_synth_tokens_inserts_the_right_totals():
    from backends.piper import PAUSE_AFTER

    zero = synth(TOKENS_MARKS, {"pause_scale": 0.0})
    one = synth(TOKENS_MARKS, {"pause_scale": 1.0})
    two = synth(TOKENS_MARKS, {"pause_scale": 2.0})
    eq(zero["sentences"], 2, "two sentences (the ? is recognized as one end)")

    def frames(r):
        return (r["bytes"] - 44) // 2

    # ASK THE RULE, don't restate it. What this gates is that `_synth_tokens` inserts and
    # accounts for what the pause rule says - the rule itself is gated in
    # check_the_punctuation_hierarchy_survives_the_slider, and a second copy of it here
    # only ever meant this test failed whenever that one changed. At scale 1.0 the rule
    # returns the table exactly (asserted in test_table_and_scale), so the default case is
    # pinned to the same numbers it always was.
    # The fake voice's audio never goes quiet, so there is no dwell at these marks to
    # MEASURE - and the splicer therefore has to supply the whole of the target, which is
    # the property being gated here. The target comes from the table (`_dwell_for`) and
    # what is already in the waveform is subtracted from it; those are two different
    # numbers and conflating them is the bug `_rest_from` documents. On a real voice the
    # measured dwell is not zero and the same call adds less, which is measured end to
    # end rather than here.
    from backends.piper import _dwell_for, _gap_for, _pause_multiplier, _rest_from

    for scale, res in ((1.0, one), (2.0, two)):
        mult = _pause_multiplier({"pause_scale": scale})
        rule = {
            m: _rest_from(_dwell_for(m), top, mult, 0.0)
            for m, top in ((",", 0.10), (":", 0.16))
        }
        rule["."] = _gap_for(".", {"pause_scale": scale})
        want = sum(round(rule[m] * SR) for m in (",", ":", "."))
        # Within a sample or two, not exact: the dwell is measured off the waveform, and
        # even this fake voice dips under the quiet threshold for a sample here and there,
        # which is the measurement working rather than an accounting error.
        got = frames(res) - frames(zero)
        ok(
            abs(got - want) <= 2,
            f"samples added at scale {scale} (comma + colon + sentence gap): {got} ~ {want}",
        )

    # the token AFTER the comma must move by the comma's pause, not by more. ASK THE RULE
    # here too: the fake voice rests nothing at the mark, so the splicer supplies the whole
    # target rather than only the top-up, and restating the top-up would be gating what
    # this file used to do instead of what it does.
    t_zero = {t["index"]: t for t in zero["tokens"]}
    t_one = {t["index"]: t for t in one["tokens"]}
    comma_at_1 = _rest_from(_dwell_for(","), PAUSE_AFTER[","], 1.0, 0.0)
    eq(
        round(t_one[1]["t0"] - t_zero[1]["t0"], 4),
        round(round(comma_at_1 * SR) / SR, 4),
        "token 1 shifted by the comma",
    )
    eq(
        round(t_one[0]["t1"] - t_zero[0]["t1"], 4),
        0.0,
        "the comma's OWN token is not shifted",
    )
    within(
        round((t_one[2]["t0"] - t_zero[2]["t0"]) * SR),
        round(comma_at_1 * SR)
        + round(_rest_from(_dwell_for(":"), PAUSE_AFTER[":"], 1.0, 0.0) * SR),
        "token 2 shifted by comma + colon (samples)",
    )
    ok(
        t_one[4]["t1"] <= one["duration"] + 1e-6,
        "the last timing still lies inside the audio",
    )


@check
def test_semicolon_and_bang():
    from backends.piper import PAUSE_AFTER, _dwell_for, _gap_for, _rest_from

    toks = [
        _tok("one", ";", ["W", "AH1", "N"]),
        _tok("two", "!", ["T", "UW1"]),
        _tok("three", "", ["TH", "R", "IY1"]),
    ]
    zero = synth(toks, {"pause_scale": 0.0})
    one = synth(toks, {"pause_scale": 1.0})
    eq(one["sentences"], 2, "! ends a sentence")
    # The `;` is spliced INTO the sentence, so it is topped up to its target against a
    # fake voice that rests nothing; the `!` ends one, so it goes through `_gap_for` and
    # is the top-up exactly. Two different paths through the same rule, which is the point
    # of checking them in one sentence.
    want = round(_rest_from(_dwell_for(";"), PAUSE_AFTER[";"], 1.0, 0.0) * SR) + round(
        _gap_for("!", {"pause_scale": 1.0}) * SR
    )
    within((one["bytes"] - zero["bytes"]) // 2, want, "semicolon + ! gap")


@check
def test_phones_path_inserts_the_right_totals():
    from backends.piper import PAUSE_AFTER, _dwell_for, _gap_for, _rest_from

    zero = synth_phones(PHONES_MARKS, {"pause_scale": 0.0})
    one = synth_phones(PHONES_MARKS, {"pause_scale": 1.0})
    eq(one["sentences"], 2, "the phones front end still splits on the full stop")
    # Same rule as the token path, asked the same way - the two front ends splicing
    # different amounts at the same mark is a thing this file exists to catch.
    want = sum(
        round(_rest_from(_dwell_for(m), PAUSE_AFTER[m], 1.0, 0.0) * SR)
        for m in (",", ":")
    ) + round(_gap_for(".", {"pause_scale": 1.0}) * SR)
    within(
        (one["bytes"] - zero["bytes"]) // 2,
        want,
        "samples added on the phones path (comma + colon + sentence gap)",
    )

    # phone timings shift with the audio, same rule as tokens
    pz = {i: p for i, p in enumerate(zero["phones"])}
    po = {i: p for i, p in enumerate(one["phones"])}
    # Ask the rule, with the fake voice's zero dwell, exactly as the totals above do.
    comma = round(_rest_from(_dwell_for(","), PAUSE_AFTER[","], 1.0, 0.0) * SR) / SR
    colon = round(_rest_from(_dwell_for(":"), PAUSE_AFTER[":"], 1.0, 0.0) * SR) / SR
    eq(round(po[3]["t1"] - pz[3]["t1"], 4), 0.0, "the comma phone itself does not move")
    eq(
        round(po[4]["t0"] - pz[4]["t0"], 4),
        round(comma, 4),
        "the phone after the comma moves by the comma",
    )
    within(
        round((po[7]["t0"] - pz[7]["t0"]) * SR),
        round((comma + colon) * SR),
        "the phone after the colon moves by comma + colon (samples)",
    )
    ok(
        po[-1 + len(po)]["t1"] <= one["duration"] + 1e-6,
        "the last phone timing still lies inside the audio",
    )


@check
def test_unaligned_voice_degrades_loudly_not_silently():
    """A voice with no duration output has nowhere to splice. It must still
    synthesize, still get its sentence gaps, and say why the rest is missing."""
    import io
    import contextlib
    import backends.piper as P

    class _NoAlign(_FakeSession):
        def run(self, outputs, feeds):
            return [super().run(outputs, feeds)[0]]

    from backends.piper import PiperBackend

    be = PiperBackend()
    P._warned_unaligned = False
    err = io.StringIO()
    with tempfile.TemporaryDirectory() as td, contextlib.redirect_stderr(err):
        res = be._synth_tokens(
            list(TOKENS_MARKS),
            "fake",
            str(Path(td) / "a.wav"),
            {"phonemizer": "ghost", "pause_scale": 1.0},
            _cfg(),
            _NoAlign(),
        )
        size = Path(str(Path(td) / "a.wav")).stat().st_size
    ok(size > 44, "audio was still produced")
    eq(res["tokens"], [], "no timings, as before")
    ok(
        "cannot be placed" in err.getvalue(),
        f"warned on stderr: {err.getvalue().strip()}",
    )
    ok("," in err.getvalue() and ":" in err.getvalue(), "named the marks it dropped")
    P._warned_unaligned = False


# -- 5. byte-identity against the previous implementation ------------------


def _reference_results():
    """Run the same fake synthesis against HEAD's piper.py, in a subprocess."""
    root = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    rel = str(Path(__file__).resolve().relative_to(root).parent / "backends/piper.py")
    head = subprocess.run(
        ["git", "-C", root, "show", f"HEAD:{rel}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp(prefix="ghost_piper_ref_"))
    dest = tmp / "voice_host"
    shutil.copytree(HERE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "backends" / "piper.py").write_text(head)
    proc = subprocess.run(
        [sys.executable, str(dest / "test_pauses.py"), "--reference"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError("reference run failed:\n" + proc.stderr[-2000:])
    shutil.rmtree(tmp, ignore_errors=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


@check
def test_byte_identical_to_head():
    ref = _reference_results()
    print("    (reference = HEAD:voice_host/backends/piper.py, same fake voice)")
    print(
        "    NOTE: this compares against the last COMMIT. If you changed synthesis on"
    )
    print(
        "    purpose - a lead-in, a splice, a phoneme rewrite - it is supposed to fail"
    )
    print(
        "    here, and it goes green again once the change is committed. Read the diff"
    )
    print("    before assuming a regression.")
    for case, (kind, items, params) in CASES.items():
        got = run_case(kind, items, params)
        r = ref[case]
        # A ONE-TIME MIGRATION ASSERTION, now retired. The two *_marks cases used to
        # assert they DIFFERED from HEAD, because the punctuation splice was the change
        # being landed and its whole point was to make those cases longer. That
        # expectation is self-invalidating: once the change was committed, HEAD contained
        # it, current matched HEAD, and "must differ" could never pass again - which is
        # exactly how it was found, failing identically with and without the edit under
        # review. Every case now asserts byte-identity, which is the property actually
        # worth guarding from here on: nothing changes the audio unless it means to.
        eq(got["sha256"], r["sha256"], f"{case}: byte-identical to HEAD")


# -- runner ----------------------------------------------------------------


@check
def check_syllabic_fold():
    """A syllabic consonant a voice cannot spell becomes schwa + consonant.

    eSpeak marks the American glottalized -ten/-tain family with U+0329: "written" comes
    back as /r\u026a\u0294n\u0329/, "certain" as /s\u025c\u0294n\u0329/, "gotten" as /g\u0251\u0294n\u0329/. Four of the five
    installed voices carry that symbol in their 157-entry maps; en_US-libritts-high
    carries 130 and does not, and there the mark was simply dropped - leaving a glottal
    stop and a bare consonant with nothing to stand as the second syllable. Across
    north-star that is 146 tokens over 19 word types, led by "written" (51), "certain"
    (34) and "gotten" (15), so it is not an edge case.

    A syllabic consonant IS a schwa plus that consonant, and eSpeak spells the
    un-glottalized members of the same family that way itself ("sudden" -> /s\u028cd\u0259n/),
    so this is a faithful rewrite rather than a patch over a gap.
    """
    from backends.piper import PiperBackend, SYLLABIC, SCHWA

    fold = PiperBackend._fold_syllabic
    poor = {
        c: [i] for i, c in enumerate("\u0279\u026a\u0294n" + SCHWA + "\u02c8")
    }  # no U+0329, like libritts-high
    rich = dict(poor)
    rich[SYLLABIC] = [99]

    stream = [("\u0279", 0), ("\u026a", 0), ("\u0294", 1), ("n", 1), (SYLLABIC, 1)]

    out, folded = fold(stream, poor)
    ok(folded, "a voice without U+0329 folds it")
    eq(
        "".join(c for c, _ in out),
        "\u0279\u026a\u0294" + SCHWA + "n",
        "written becomes schwa + n",
    )
    ok(all(c in poor for c, _ in out), "nothing survives that the voice cannot spell")
    eq(
        [i for _, i in out],
        [0, 0, 1, 1, 1],
        "the inserted schwa inherits the base's source index (karaoke alignment)",
    )

    out2, folded2 = fold(stream, rich)
    ok(not folded2, "a voice that CAN spell U+0329 is left alone")
    eq(out2, stream, "the rich-map stream is unchanged")

    out3, _ = fold([(SYLLABIC, 0)], poor)
    eq(
        out3,
        [],
        "a stray mark with no base is discarded, not turned into a leading schwa",
    )


@check
@check
def check_the_punctuation_hierarchy_survives_the_slider():
    """A comma may never out-rest a full stop, and their RATIO may not move with the dial.

    Three bugs live in this one function's history, all reported by ear, and the third is
    only visible if you measure what a listener hears rather than what we splice.

      1. An inversion. Mid-sentence marks were uncapped while the sentence gap was pinned,
         so past 1.0 the commas overtook the full stops.
      2. A dead dial. Capping each mark at its share of one absolute ceiling meant they all
         hit it at 3.75x, and the top 62% of the slider did nothing.
      3. Drifting proportions. Scaling only OUR silence linearly does not scale the REST
         linearly, because the model is resting too - so the dial stretched the gaps between
         the marks rather than the marks themselves, and no setting suited both ends of the
         punctuation ladder at once.

    So the assertions below are on the TOTAL rest, which is the only thing anyone hears.
    """
    from backends import piper

    def total(mark, scale):
        p = {"pause_scale": scale}
        add = piper._gap_for(mark, p) if mark in ".!?" else piper._pause_for(mark, p)
        return piper.DWELL[mark] + add

    for scale in (1.0, 2.0, 3.0, 3.75, 5.0, 6.0, 7.5, 10.0):
        c, sm, cl, g = (total(m, scale) for m in (",", ";", ":", "."))
        ok(
            c <= sm <= cl <= g,
            "at %gx the order , <= ; <= : <= . holds (%.3f %.3f %.3f %.3f)"
            % (scale, c, sm, cl, g),
        )
    # BELOW 1.0 THE ORDER IS THE MODEL'S, not ours, and that is not a failure to assert
    # around: we can only ADD silence. This voice dwells 0.30 s at a colon and 0.18 after a
    # full stop, so under about 0.6 on the dial - where the target rest falls below what the
    # model is already doing - the colon genuinely out-rests the sentence and there is
    # nothing to subtract. What must hold is that we are not making it worse: nothing is
    # spliced anywhere down there.
    for scale in (0.0, 0.25):
        for mark in (",", ";", ":"):
            ok(
                piper._pause_for(mark, {"pause_scale": scale}) == 0.0,
                "at %gx `%s` splices nothing - the model is already resting longer"
                % (scale, mark),
            )
    # ...and just under 1.0 it is down to milliseconds rather than exactly nothing, which is
    # the floor arriving smoothly instead of as a cliff.
    for mark in (",", ";", ":"):
        v = piper._pause_for(mark, {"pause_scale": 0.5})
        ok(v < 0.02, "at 0.5x `%s` splices next to nothing (%.4fs)" % (mark, v))

    # THE RATIO IS THE FIX. A full stop rests twice as long as a comma at 1.0; it has to
    # rest twice as long as a comma everywhere, or one setting cannot suit both.
    # From 1.0 up, which is the dial's whole working range: below it the floor above is
    # what decides, and a ratio nobody can influence is not a claim about this rule.
    at_one = total(".", 1.0) / total(",", 1.0)
    for scale in (1.0, 2.0, 3.0, 3.75, 5.0, 6.0, 7.5, 10.0):
        r = total(".", scale) / total(",", scale)
        ok(
            abs(r - at_one) < 0.02,
            "at %gx a full stop is still %.2fx a comma (%.2f)" % (scale, at_one, r),
        )

    # The default reading is the one it always was, to the millisecond.
    for mark, base in ((",", 0.10), (";", 0.13), (":", 0.16)):
        ok(
            abs(piper._pause_for(mark, {"pause_scale": 1.0}) - base) < 1e-6,
            "at 1x `%s` splices exactly its table entry (%.4f)"
            % (mark, piper._pause_for(mark, {"pause_scale": 1.0})),
        )
    ok(
        abs(piper._gap_for(".", {"pause_scale": 1.0}) - 0.32) < 1e-6,
        "...and so does a full stop",
    )

    # Still live to the end of its travel - bug 2 must not come back.
    ok(
        total(".", 10.0) > total(".", 3.75) * 1.9,
        "the top of the slider rests far past its middle (%.2fs vs %.2fs)"
        % (total(".", 10.0), total(".", 3.75)),
    )
    # AND THE TOP IS REACHABLE AT ALL. The curve this replaced was a saturating exponential
    # that reached 3.2x at the dial's top, 3.28 at 20 and 3.29 at 100 - so "the pause effect
    # barely seems to work at 10x" could not have been answered by allowing a bigger number.
    # A power law keeps climbing, and this is the assertion that it does.
    from backends.piper import _pause_multiplier

    ok(
        abs(_pause_multiplier({"pause_scale": 10.0}) - 5.0) < 0.01,
        "the top of the dial is 5x the natural rest (%.2f)"
        % _pause_multiplier({"pause_scale": 10.0}),
    )

    # ...and NOT finicky: every step up the dial adds less than the step before it, and the
    # one the report named (5 to 6) is a nudge rather than a lurch.
    steps = [total(".", s + 1.0) - total(".", s) for s in range(1, 10)]
    ok(
        all(b <= a + 1e-9 for a, b in zip(steps, steps[1:])),
        "from 1.0 up, each unit of dial adds less than the one before (%s)"
        % " ".join("%.2f" % x for x in steps),
    )
    # Reach costs step size - that is the trade, made deliberately and bounded here. 13% per
    # whole unit of dial is a nudge; the linear law that prompted the complaint moved 18%.
    jump = (total(".", 6.0) - total(".", 5.0)) / total(".", 5.0)
    ok(
        jump < 0.15,
        "5.0 -> 6.0 moves a full stop by %.0f%%, not a lurch" % (jump * 100.0),
    )


@check
def check_real_voice_comma():
    """On the real checkpoint, a comma the voice runs straight through.

    "Hello, my loves." with no rest floor (a graph that cannot take one) - this voice
    lengthens the vowel into the comma rather than pausing, so there is no silence to cut
    in. Old and new splice the SAME render: the old cut at the token boundary with 2 ms
    ramps must leave a loud edge (the click), the new must not. Skipped when the voice is
    not installed.
    """
    import numpy as np
    import backends.piper as P
    from backends.piper import PiperBackend

    voice = "en_US-libritts-high"
    be = PiperBackend()
    try:
        sess, cfg = be._load(voice)
    except Exception as exc:  # noqa: BLE001
        print("    -- %s is not installed here (%s); skipping" % (voice, exc))
        return
    sr = int(cfg["audio"]["sample_rate"])
    params = {
        "speaker": 13, "length_scale": 1.08, "noise_scale": 0.78, "noise_w": 0.52,
        "pause_scale": 6.5,
    }
    toks = [_tok("Hello", ",", []), _tok("my", "", []), _tok("loves", ".", [])]
    seen: list = []
    real = P._splice_pauses

    def spy(audio, points, sr_, mult=1.0):
        seen.append((audio, list(points), mult))
        return real(audio, points, sr_, mult)

    def edges(a, peak):
        n3 = int(0.003 * sr)
        out = []
        for start, length in _zero_runs(a, int(0.05 * sr)):
            if start + length < a.size:
                for seg in (a[start - n3 : start], a[start + length : start + length + n3]):
                    out.append(float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) / peak)
        return out

    P._splice_pauses = spy
    floor = P._rest_floor
    P._rest_floor = lambda mark, params: 0.0
    try:
        for _ in range(5):
            seen.clear()
            with tempfile.TemporaryDirectory() as td:
                be._synth_tokens(list(toks), voice, str(Path(td) / "a.wav"), params, cfg, sess)
            audio, points, mult = seen[0]
            peak = float(np.max(np.abs(audio)))
            with _old_ramps():
                old, _ = real(audio, [pt[:4] for pt in points], sr, mult)
            if max(edges(old, peak), default=0.0) > 0.03:
                break
    finally:
        P._splice_pauses = real
        P._rest_floor = floor
    new, _ = real(audio, points, sr, mult)
    e_old, e_new = edges(old, peak), edges(new, peak)
    print("      edges (share of peak, 3 ms): old %s new %s"
          % (" ".join("%.3f" % x for x in e_old), " ".join("%.3f" % x for x in e_new)))
    ok(max(e_old) > 0.03, "the old cut leaves a loud edge (control)")
    ok(max(e_new) < 0.02, "the new cut leaves none")


@check
def check_real_voice_rests_at_a_mark():
    """On the real checkpoint, the word before a comma is ended by the model, not the splice.

    With no rest floor this voice runs through the comma and the spliced silence starts
    its fade inside the vowel - the word heard cut off. With the floor the model rests
    there itself and the splice lands in that rest. Two-sided on the same sentences and
    settings (the tarot narrator's): without the floor most cuts have voice in the 30 ms
    before them, with it none do. Skipped when the voice is not installed.
    """
    import numpy as np
    import backends.piper as P
    from backends.piper import PiperBackend

    voice = "en_US-libritts-high"
    be = PiperBackend()
    try:
        sess, cfg = be._load(voice)
    except Exception as exc:  # noqa: BLE001
        print("    -- %s is not installed here (%s); skipping" % (voice, exc))
        return
    sr = int(cfg["audio"]["sample_rate"])
    params = {
        "speaker": 13, "length_scale": 1.08, "noise_scale": 0.78, "noise_w": 0.52,
        "pause_scale": 6.5,
    }
    sentences = [
        [_tok("On", "", []), _tok("you", ",", []), _tok("my", "", []), _tok("loves", ".", [])],
        [_tok("Hello", ",", []), _tok("my", "", []), _tok("loves", ".", [])],
    ]

    def before_cuts() -> list:
        """Voice in the 30 ms before each spliced rest, as a share of the take's peak."""
        out = []
        for toks in sentences:
            for _ in range(4):
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "a.wav"
                    be._synth_tokens(list(toks), voice, str(path), params, cfg, sess)
                    raw = path.read_bytes()
                a = np.frombuffer(raw[44:], "<i2").astype(np.float32) / 32768.0
                peak = float(np.max(np.abs(a)))
                n = int(0.030 * sr)
                for start, length in _zero_runs(a, int(0.05 * sr)):
                    if n <= start and start + length < a.size:
                        seg = a[start - n : start].astype(np.float64)
                        out.append(float(np.sqrt(np.mean(seg**2))) / peak)
        return out

    floor = P._rest_floor
    P._rest_floor = lambda mark, params: 0.0
    try:
        bare = before_cuts()
    finally:
        P._rest_floor = floor
    floored = before_cuts()
    print("      voice before the cut, no floor: %s" % " ".join("%.3f" % x for x in bare))
    print("      voice before the cut, floored:  %s" % " ".join("%.3f" % x for x in floored))
    ok(len(bare) >= 6 and len(floored) >= 6, "every take has its comma rest")
    ok(sum(x > 0.05 for x in bare) * 2 > len(bare), "without the floor most cuts are in voice (control)")
    ok(max(floored) < 0.04, "with it, every cut is in the model's own rest")


def check_hyphen_is_a_word_boundary():
    """A hyphen INSIDE a word reaches the phonemizer as a word space.

    The reported symptom was a rest in the middle of "ten-forty" and
    "eleven-thirty". eSpeak returns the same phones for the hyphenated spelling
    and the spaced one and differs only in the word space, so the boundary was
    being dropped on the way in - and en_US-libritts-high answers two primary
    stresses welded together with a hole where the boundary should be (measured
    on "forty-second": 0.40 s of near-silence inside one word, 0.07 s once the
    space is sent). See _espeak_word.

    The two HOLDS matter as much as the switch. The hyphen must survive in the
    token's own text, because the karaoke draws the source spelling, and a token
    must never be handed to the phonemizer as an empty string - phonemizer drops
    an empty input instead of returning "" for it, which pairs every later word
    with its neighbor's phonemes.
    """
    from backends.piper import LEAD_IN_SPACES, PiperBackend, _espeak_word

    eq(_espeak_word("ten-forty"), "ten forty", "an internal hyphen becomes a space")
    eq(_espeak_word("mother-in-law"), "mother in law", "and every hyphen does")
    eq(_espeak_word("ordinary"), "ordinary", "a word without one is untouched")
    eq(_espeak_word("ten-"), "ten", "a trailing dash contributes no empty word")
    eq(_espeak_word("-"), "-", "a token that is ONLY a dash is left alone")
    eq(_espeak_word("  spaced  "), "spaced", "the strip the old code did still happens")

    seen: list = []
    orig = PiperBackend._espeak

    def fake(cls, words, voice="en-us"):
        seen.extend(words)
        return ["ab" if " " not in w else "a b" for w in words]

    PiperBackend._espeak = classmethod(fake)
    try:
        be = PiperBackend()
        tokens = [
            {"text": "ten-forty", "punct": ",", "fallback": []},
            {"text": "now", "punct": ".", "fallback": []},
        ]
        out = be._symbols(tokens, "espeak", "en-us")
    finally:
        PiperBackend._espeak = orig

    eq(seen, ["ten forty", "now"], "the phonemizer is asked for the spaced spelling")
    eq(tokens[0]["text"], "ten-forty", "the token keeps its spelling for the karaoke")
    body = out[LEAD_IN_SPACES:]
    first = [c for c, i in body if i == 0]
    eq(first, ["a", " ", "b", ",", " "], "the boundary reaches the model as a space")
    ok(
        all(i in (0, 1) for _, i in body),
        "every symbol still belongs to a real token (alignment intact)",
    )


def main() -> int:
    if "--reference" in sys.argv:
        print(
            json.dumps(
                {
                    k: {
                        kk: vv
                        for kk, vv in run_case(*c).items()
                        if kk not in ("tokens", "phones")
                    }
                    for k, c in CASES.items()
                }
            )
        )
        return 0
    failed = 0
    for fn in CHECKS:
        print(f"\n{fn.__name__}")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"    FAIL {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
