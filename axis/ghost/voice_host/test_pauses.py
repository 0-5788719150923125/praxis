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


def check(fn):
    CHECKS.append(fn)
    return fn


def eq(got, want, what: str):
    assert got == want, f"{what}: got {got!r}, want {want!r}"
    print(f"    ok  {what} == {want!r}")


def ok(cond, what: str):
    assert cond, f"FAILED: {what}"
    print(f"    ok  {what}")


# -- 1. the table and the scale -------------------------------------------


@check
def test_table_and_scale():
    from backends.piper import PAUSE_AFTER, _gap_for, _pause_for

    eq(_pause_for(",", {}), PAUSE_AFTER[","], "comma at default scale")
    eq(_pause_for(":", {"pause_scale": 2.0}), PAUSE_AFTER[":"] * 2.0, "colon doubled")
    eq(_pause_for(";", {"pause_scale": 0.0}), 0.0, "semicolon at scale 0")
    eq(_pause_for("x", {}), 0.0, "a non-mark gets nothing")
    # the unification: no explicit sentence_gap means the table, and the table's
    # sentence-final value IS the old 0.32 default
    eq(_gap_for(".", {}), 0.32, "sentence gap default")
    eq(_gap_for("?", {}), 0.32, "question mark uses the same gap")
    eq(_gap_for(".", {"pause_scale": 0.5}), 0.16, "sentence gap halved")
    eq(_gap_for(".", {"sentence_gap": 0.5}), 0.5, "explicit sentence_gap still wins")
    eq(
        _gap_for(".", {"sentence_gap": 0.5, "pause_scale": 2.0}),
        1.0,
        "explicit sentence_gap is scaled too",
    )
    eq(_gap_for(".", {"pause_scale": "nonsense"}), 0.32, "a bad scale falls back")


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


@check
def test_the_cut_does_not_click():
    import numpy as np
    from backends.piper import _splice_pauses

    audio = _sine(0.5)
    nominal = round(0.25 * SR)
    ok(
        abs(float(audio[nominal])) > 0.5,
        f"the nominal cut sample is loud ({audio[nominal]:.3f}) - a hard cut "
        "there would step straight to zero",
    )
    out, _ = _splice_pauses(audio, [(0.25, 0.12)], SR)
    start, length = _zero_runs(out)[0]
    before, after = abs(float(out[start - 1])), abs(float(out[start + length]))
    ok(before < 1e-3, f"sample entering the silence is {before:.2e}, not a step")
    ok(after < 1e-3, f"sample leaving the silence is {after:.2e}, not a step")
    ok(
        before < abs(float(audio[nominal])) / 100.0,
        "zero-crossing search + ramp is >100x quieter at the seam than a naive cut",
    )
    # the ramp only touches near-silent samples: everything more than the ramp
    # away from the seam is bit-identical to the input
    fade = round(2.0 * SR / 1000.0)
    ok(
        np.array_equal(out[: start - fade], audio[: start - fade]),
        "speech before the ramp is bit-identical",
    )


# -- 4. the real synthesis path -------------------------------------------


@check
def test_synth_tokens_inserts_the_right_totals():
    from backends.piper import PAUSE_AFTER

    zero = synth(TOKENS_MARKS, {"pause_scale": 0.0})
    one = synth(TOKENS_MARKS, {"pause_scale": 1.0})
    two = synth(TOKENS_MARKS, {"pause_scale": 2.0})
    eq(zero["sentences"], 2, "two sentences (the ? is recognised as one end)")

    def frames(r):
        return (r["bytes"] - 44) // 2

    for scale, res in ((1.0, one), (2.0, two)):
        want = sum(round(PAUSE_AFTER[m] * scale * SR) for m in (",", ":", "."))
        eq(
            frames(res) - frames(zero),
            want,
            f"samples added at scale {scale} (comma + colon + sentence gap)",
        )

    # the token AFTER the comma must move by the comma's pause, not by more
    t_zero = {t["index"]: t for t in zero["tokens"]}
    t_one = {t["index"]: t for t in one["tokens"]}
    eq(
        round(t_one[1]["t0"] - t_zero[1]["t0"], 4),
        round(round(PAUSE_AFTER[","] * SR) / SR, 4),
        "token 1 shifted by the comma",
    )
    eq(
        round(t_one[0]["t1"] - t_zero[0]["t1"], 4),
        0.0,
        "the comma's OWN token is not shifted",
    )
    eq(
        round(t_one[2]["t0"] - t_zero[2]["t0"], 4),
        round(round(PAUSE_AFTER[","] * SR) / SR + round(PAUSE_AFTER[":"] * SR) / SR, 4),
        "token 2 shifted by comma + colon",
    )
    ok(
        t_one[4]["t1"] <= one["duration"] + 1e-6,
        "the last timing still lies inside the audio",
    )


@check
def test_semicolon_and_bang():
    from backends.piper import PAUSE_AFTER

    toks = [
        _tok("one", ";", ["W", "AH1", "N"]),
        _tok("two", "!", ["T", "UW1"]),
        _tok("three", "", ["TH", "R", "IY1"]),
    ]
    zero = synth(toks, {"pause_scale": 0.0})
    one = synth(toks, {"pause_scale": 1.0})
    eq(one["sentences"], 2, "! ends a sentence")
    want = round(PAUSE_AFTER[";"] * SR) + round(PAUSE_AFTER["!"] * SR)
    eq((one["bytes"] - zero["bytes"]) // 2, want, "semicolon + ! gap")


@check
def test_phones_path_inserts_the_right_totals():
    from backends.piper import PAUSE_AFTER

    zero = synth_phones(PHONES_MARKS, {"pause_scale": 0.0})
    one = synth_phones(PHONES_MARKS, {"pause_scale": 1.0})
    eq(one["sentences"], 2, "the phones front end still splits on the full stop")
    want = sum(round(PAUSE_AFTER[m] * SR) for m in (",", ":", "."))
    eq(
        (one["bytes"] - zero["bytes"]) // 2,
        want,
        "samples added on the phones path (comma + colon + sentence gap)",
    )

    # phone timings shift with the audio, same rule as tokens
    pz = {i: p for i, p in enumerate(zero["phones"])}
    po = {i: p for i, p in enumerate(one["phones"])}
    comma, colon = round(PAUSE_AFTER[","] * SR) / SR, round(PAUSE_AFTER[":"] * SR) / SR
    eq(round(po[3]["t1"] - pz[3]["t1"], 4), 0.0, "the comma phone itself does not move")
    eq(
        round(po[4]["t0"] - pz[4]["t0"], 4),
        round(comma, 4),
        "the phone after the comma moves by the comma",
    )
    eq(
        round(po[7]["t0"] - pz[7]["t0"], 4),
        round(comma + colon, 4),
        "the phone after the colon moves by comma + colon",
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
    """A comma may never out-rest a full stop, at any Pause setting.

    The sentence gap is capped (it compounds over a chapter); mid-sentence marks were not, so
    at the slider's top a comma took 1.0 s of spliced silence on top of the ~0.3 s the model
    already dwells there while a full stop stayed pinned at 1.2 s. Reported as a large breath
    after every comma and none anywhere else, which is exactly what an inverted hierarchy
    sounds like. The ordering has to hold at EVERY scale, not just at 1.
    """
    from backends import piper

    for scale in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0):
        p = {"pause_scale": scale}
        comma = piper._pause_for(",", p)
        semi = piper._pause_for(";", p)
        colon = piper._pause_for(":", p)
        gap = piper._gap_for(".", p)
        ok(
            comma <= semi <= colon < gap or scale == 0.0,
            "at %gx the order , <= ; <= : < . holds (%.3f %.3f %.3f %.3f)"
            % (scale, comma, semi, colon, gap))
    # ...and the default render is untouched: below the ceiling the cap cannot bite, so
    # everything anyone has already heard sounds the same.
    for scale in (0.5, 1.0, 2.0, 3.0):
        p = {"pause_scale": scale}
        ok(
            abs(piper._pause_for(",", p) - 0.10 * scale) < 1e-9,
            "at %gx a comma is still exactly linear (%.4f)" % (scale, piper._pause_for(",", p)),
        )


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
    with its neighbour's phonemes.
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
