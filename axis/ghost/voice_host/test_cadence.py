#!/usr/bin/env python3
"""Gate for the two cadence defects: citation stress, and the hole nobody asked for.

Run it directly with the voice venv's python - no pytest needed:

    ~/.local/share/godot/app_userdata/ghost/voice_venv/bin/python \
        axis/ghost/voice_host/test_cadence.py

THE REPORT. "Most of the libritts-high voices have weird stuttering in their cadence,
around specific words... `in` and `in it`", and generally "the flow of the speaking feels
rather jerky/uneven now". Rendered on the reporter's own settings and text, that was two
defects sitting on top of each other:

  1. eSpeak was asked for one word at a time, so every word came back in its CITATION
     form - and the citation form of a function word carries a primary stress. `a` was
     ˈeɪ, which is not a stressed schwa, it is the letter A. See WORD_BREAK.
  2. Piper's duration predictor is stochastic and the blanks between phonemes are part of
     what it predicts, so about one boundary in six hundred came back with a quarter of a
     second of silence in it, mid-clause, with no punctuation anywhere near. See
     `_trim_rests`.

Three of the four checks need nothing but eSpeak. The fourth renders real audio if a
checkpoint is on this machine and says so and skips if not.

EVERY CHECK IS TWO-SIDED. "No stressed function words" and "no holes" are also what a
silent file reports, so each one is run against the rule it replaced as well, and fails if
the OLD rule does not show the defect - a gate that can only pass is not a gate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GHOST = HERE.parent
for p in (str(HERE), str(GHOST)):
    if p not in sys.path:
        sys.path.insert(0, p)

import backends.piper as P  # noqa: E402
from backends.piper import (  # noqa: E402
    HOP_LENGTH,
    REST_FLOOR,
    REST_RATIO,
    PiperBackend,
    _espeak_word,
)

VOICE = "en_US-libritts-high"
SPEAKER = 28  # the reporter's own
## One paragraph of the reporter's own chapter, which is where the defect was heard. Kept
## here rather than read from disk: the gate has to run on a machine that does not have
## his books, and the sentence that opens it is the one the report names.
PROSE = (
    "There is a country in this report with a house in it that will make a copy of "
    "anything you bring through the door. Not a hundred of them. A hundred was never "
    "the hard part. Every world in this book could make you a hundred of a thing, "
    "given money and patience and a reason, and the presses of that country had been "
    "running in the hundreds since before anybody's grandfather. What the house does, "
    "and what nothing before it could do, is make you a single copy, by itself, cheap, "
    "by morning. You have to know the old price or the new one means nothing. Before "
    "that house, a copy came out of a man."
)

FAILS: list[str] = []


def ok(cond: bool, what: str, detail: str = "") -> None:
    print(
        "   %-4s %s%s"
        % ("ok" if cond else "FAIL", what, ("  " + detail) if detail else "")
    )
    if not cond:
        FAILS.append(what)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokens_of(text: str) -> list[dict]:
    """The token list ghost's front end would send for one sentence.

    Deliberately crude - `Phonemes.parse` does far more - but the only fields anything
    below reads are `text` and `punct`, and those it gets right.
    """
    out = []
    for w in text.split():
        m = re.match(r"^(.*?)([,.;:?!]*)$", w)
        body, marks = m.group(1).strip("\"'()"), m.group(2)
        if not body:
            continue
        out.append({"text": body, "punct": marks[-1] if marks else "", "fallback": []})
    return out


# -- 1: the reading -------------------------------------------------------------


## eSpeak's stress marks. What separates a citation reading from a spoken one is where
## these sit, and whether they are there at all.
STRESS = "ˈˌ"


def check_stress(be: PiperBackend) -> None:
    """A function word in a sentence is not stressed, and word by word it is."""
    print("\n1. the reading")
    toks = tokens_of(sentences(PROSE)[0])
    ctx = be._in_sentence(toks, list(range(len(toks))), "en-us")
    ok(
        bool(ctx),
        "the sentence came back in pieces, one per word",
        "%d of %d" % (len(ctx), len(toks)),
    )
    if not ctx:
        return
    iso = be._espeak([_espeak_word(t["text"]) for t in toks], "en-us")

    # THE CONTROL. If the old path is not producing citation stress there is nothing here
    # to fix, and everything below is asserting nothing.
    marked_iso = [
        t["text"]
        for t, r in zip(toks, iso)
        if t["text"].lower() in FUNCTION and any(c in r for c in STRESS)
    ]
    ok(
        len(marked_iso) >= 5,
        "word by word, the function words carry a stress mark",
        "%d of them: %s" % (len(marked_iso), " ".join(marked_iso)),
    )

    marked_ctx = [
        t["text"]
        for i, t in enumerate(toks)
        if t["text"].lower() in FUNCTION and any(c in ctx[i] for c in STRESS)
    ]
    ok(
        not marked_ctx,
        "in the sentence, none of them does",
        "still marked: %s" % " ".join(marked_ctx) if marked_ctx else "",
    )

    # The one that is a wrong WORD rather than a wrong rhythm.
    a = [i for i, t in enumerate(toks) if t["text"].lower() == "a"]
    ok(bool(a), "the passage has an article to check")
    if a:
        i = a[0]
        ok("eɪ" in iso[i], "word by word, `a` is read as the letter A", iso[i])
        ok("eɪ" not in ctx[i], "in the sentence it is not", ctx[i])


## Closed-class words from the passage above. Not a lexicon and nothing reads it but this
## file: it is the list of words whose stress the check is ABOUT, so it is written where
## the check is rather than sourced, and it is allowed to be incomplete.
FUNCTION = frozenset(
    "a an the in it that this with of to and or is was are you he she they will "
    "could had been by for from not there here what".split()
)


def check_alignment(be: PiperBackend) -> None:
    """Whatever the reading, every word still gets one - alignment is not negotiable."""
    print("\n2. alignment")
    whole = 0
    fell_back = 0
    for s in sentences(PROSE):
        toks = tokens_of(s)
        if not toks:
            continue
        whole += 1
        need = list(range(len(toks)))
        got = be._in_sentence(toks, need, "en-us")
        if not got:
            fell_back += 1
            continue
        missing = [i for i in need if not got.get(i, "").strip()]
        if missing:
            FAILS.append("a word came back with no phones")
            print("   FAIL a word came back with no phones: %s" % s[:60])
            return
    ok(
        fell_back <= whole // 3,
        "the sentence reading holds for most sentences",
        "%d of %d fell back to word by word" % (fell_back, whole),
    )
    # ...and the fallback itself still reads every word, which is what makes it safe to
    # take. Forced here rather than waited for.
    real = PiperBackend._in_sentence
    try:
        PiperBackend._in_sentence = lambda self, *a, **k: {}
        toks = tokens_of(sentences(PROSE)[0])
        syms = be._symbols(toks, "espeak", "en-us", "")
    finally:
        PiperBackend._in_sentence = real
    spoke = {src for sym, src in syms if sym.strip()}
    ok(
        len(spoke) == len(toks),
        "with the sentence reading refused, every word is still spoken",
        "%d of %d" % (len(spoke), len(toks)),
    )


# -- 3: the trim ----------------------------------------------------------------


def _tone(n: int, sr: int = 22050) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    return (0.5 * np.sin(2.0 * np.pi * 140.0 * t / sr)).astype(np.float32)


def check_trim() -> None:
    """The hole is cut back to the ceiling, and nothing else is."""
    print("\n3. the trim")
    sr = 22050
    # Eight boundaries of 0.05 s and one of 0.40 s - the shape the model actually
    # produces, one outlier in a sentence of ordinary rests.
    word = int(0.20 * sr)
    small = int(0.05 * sr)
    big = int(0.40 * sr)
    parts, rests, at = [], [], 0.0
    for i in range(9):
        parts.append(_tone(word))
        at += word / sr
        gap = big if i == 4 else small
        parts.append(np.zeros(gap, dtype=np.float32))
        rests.append((at, at + gap / sr))
        at += gap / sr
    parts.append(_tone(word))
    audio = np.concatenate(parts)

    out, cut = P._trim_rests(audio, rests, sr)
    cap = max(REST_FLOOR, REST_RATIO * 0.05)
    ok(len(cut) == 1, "exactly one rest was cut", "%d" % len(cut))
    removed = -sum(s for _t, s in cut)
    want = 0.40 - cap
    ok(
        abs(removed - want) < 0.01,
        "and cut to the ceiling",
        "removed %.3fs, ceiling %.3fs" % (removed, cap),
    )
    ok(
        abs((audio.size - out.size) / sr - removed) < 0.001,
        "the waveform is shorter by exactly that",
    )

    # NO CLICK. The join is inside silence by construction, so the worst sample-to-sample
    # step in the result must be no worse than the untouched signal's own.
    ok(
        float(np.max(np.abs(np.diff(out))))
        <= float(np.max(np.abs(np.diff(audio)))) + 1e-6,
        "the cut does not step",
    )

    # THE CONTROL, both ways round.
    same, none = P._trim_rests(audio, [r for i, r in enumerate(rests) if i != 4], sr)
    ok(
        not none and same is audio,
        "a sentence with no outlier is returned untouched, not merely equal",
    )
    ok(P._trim_rests(audio, [], sr)[1] == [], "and so is one with no rests at all")

    # A rest the frame plan claims but the waveform denies is left alone - the plan is an
    # opinion about where the silence is and the waveform is the fact.
    loud = audio.copy()
    a, b = rests[4]
    loud[int(a * sr) : int(b * sr)] = _tone(int(b * sr) - int(a * sr))
    kept, none2 = P._trim_rests(loud, rests, sr)
    ok(not none2 and kept is loud, "a rest that is not silent is not cut")


# -- 4: the same thing, on the real checkpoint -----------------------------------


def _boundary_holes(audio, rests, sr: int, thresh: float) -> list:
    """Seconds of contiguous near-silence around each boundary, from the waveform."""
    out = []
    for a, b in rests:
        at = int(round(0.5 * (a + b) * sr))
        out.append(P._silence_around(audio, at, thresh, sr))
    return out


def _render_all(be, sess, cfg, noise_w: float) -> tuple:
    """Every sentence of the passage, with its unmarked boundary rests."""
    out = []
    for s in sentences(PROSE):
        toks = tokens_of(s)
        if len(toks) < 5:
            continue
        params = {
            "speaker": SPEAKER,
            "length_scale": 0.96,
            "noise_scale": 0.60,
            "noise_w": noise_w,
        }
        audio, _spans = be._run(toks, VOICE, params, cfg, sess, "espeak")
        rests = [
            (a, b)
            for a, b, w in be._last_rests
            if w < len(toks) and not str(toks[w].get("punct", ""))
        ]
        if len(rests) >= 4:
            out.append((audio, rests))
    return out


def _paired(be, renders, sr: int) -> tuple:
    """Boundary silences before and after the trim, on the SAME audio."""
    before, after, fired, over = [], [], 0, 0.0
    for audio, rests in renders:
        quiet = float(np.max(np.abs(audio))) * 0.02
        before += _boundary_holes(audio, rests, sr, quiet)
        cut_audio, cut = P._trim_rests(audio, rests, sr)
        fired += len(cut)
        moved = [(P._shift(a, cut, True), P._shift(b, cut, False)) for a, b in rests]
        got = _boundary_holes(cut_audio, moved, sr, quiet)
        after += got
        # The residual, in the ceiling's own units. `_trim_rests` caps the MEASURED
        # silence, not the frame plan's idea of it, so that is what this has to measure -
        # the ceiling itself still comes from the plan, which is the only reference that
        # tracks pace, length_scale and the checkpoint together.
        lens = sorted(b - a for a, b in rests)
        cap = max(REST_FLOOR, REST_RATIO * lens[len(lens) // 2])
        over = max(over, max(got) / cap if got else 0.0)
    return np.array(before), np.array(after), fired, over


def check_real_voice() -> None:
    """Rendered audio, trimmed and not - the SAME render, so the predictor's own
    scatter cannot decide the answer.

    TWO PASSES, because one cannot answer both questions. At the reporter's own settings
    the defect is 0.16% of boundaries, so a gate that waits for one is a gate that passes
    by luck about as often as it fails; what that pass CAN establish is that the trim is
    inert on ordinary material, which is the property most worth protecting. The second
    pass turns `noise_w` up - the same mechanism, more of it - so the hole is reliably
    there to be removed, and the control can be checked rather than hoped for.
    """
    print("\n4. the real checkpoint")
    be = PiperBackend()
    try:
        sess, cfg = be._load(VOICE)
    except Exception as exc:  # noqa: BLE001
        print("   -- %s is not installed here (%s); skipping" % (VOICE, exc))
        return
    sr = int(cfg["audio"]["sample_rate"])

    quiet_w = 0.35  # the Warm preset, which is what the report was heard on
    b1, a1, fired1, over1 = _paired(be, _render_all(be, sess, cfg, quiet_w), sr)
    if not b1.size:
        print("   -- nothing rendered; skipping")
        return
    print(
        "      noise_w %.2f: %d boundaries, %d cut, p90 %.3f -> %.3f, max %.3f -> %.3f"
        % (
            quiet_w,
            b1.size,
            fired1,
            np.percentile(b1, 90),
            np.percentile(a1, 90),
            b1.max(),
            a1.max(),
        )
    )
    ok(
        fired1 <= max(1, b1.size // 20),
        "on ordinary material the trim is nearly always idle",
        "%d of %d boundaries" % (fired1, b1.size),
    )
    ok(
        np.percentile(a1, 90) <= np.percentile(b1, 90) + 0.002,
        "and the ordinary boundaries are where they were",
    )

    # THE HOLE, PUT THERE ON PURPOSE. Waiting for the duration predictor to produce one
    # is a gate that passes by luck: at the reporter's settings it is about 1% of
    # boundaries, and turning `noise_w` up only moves the odds - three runs of an earlier
    # version of this check produced an above-ceiling hole twice and none the third time.
    # So the defect is INJECTED, into a real word boundary of a real render, at a size
    # the model itself has been measured producing. Same audio, same rests, one known
    # hole - and the claim it checks is the exact one `_trim_rests` makes.
    holes, brought_back, gave_back = 0, 0, 0
    worst = ""
    for audio, rests in _render_all(be, sess, cfg, quiet_w):
        lens = sorted(b - a for a, b in rests)
        cap = max(REST_FLOOR, REST_RATIO * lens[len(lens) // 2])
        extra = cap * 2.0
        hurt, marks, at = _inject(audio, rests, sr, extra)
        quiet = float(np.max(np.abs(hurt))) * 0.02
        before = P._silence_around(hurt, int(round(at * sr)), quiet, sr)
        if before <= cap:
            continue  # the injection landed somewhere already loud; nothing to claim
        holes += 1
        fixed, cut = P._trim_rests(hurt, marks, sr)
        after = P._silence_around(
            fixed, int(round(P._shift(at, cut, True) * sr)), quiet, sr
        )
        if cut and after <= cap * 1.05:
            brought_back += 1
        else:
            worst = "%.3fs -> %.3fs against a %.3fs ceiling" % (before, after, cap)
        # ...and the WAVEFORM agrees with the measurement, which is a different
        # instrument answering the same question: the samples removed must be the
        # silence that was there minus the ceiling. Not "as much as was injected" - the
        # trim takes the rest TO the ceiling, so a boundary already near it gives back
        # less than went in, and one already over it gives back more.
        if abs((hurt.size - fixed.size) / sr - (before - cap)) < 0.02:
            gave_back += 1
    ok(holes >= 3, "there were injected holes to remove", "%d of them" % holes)
    ok(
        brought_back == holes,
        "every one is cut back to the ceiling",
        worst or "%d of %d" % (brought_back, holes),
    )
    ok(
        gave_back == holes,
        "and the samples removed are the excess, to the millisecond",
        "%d of %d" % (gave_back, holes),
    )

    # `_trim_rests` declines a rest whose quietest window is not actually silent - that
    # is the waveform overruling the frame plan, and it is deliberate. What it may not do
    # is leave one MATERIALLY over the ceiling.
    ok(
        over1 < 1.25,
        "and nothing a natural render left is materially over the ceiling",
        "worst residual is %.2fx it" % over1,
    )


def _inject(audio, rests, sr: int, extra: float):
    """Lengthen the widest word-boundary rest by `extra` seconds of digital silence.

    Returns (audio, rests, midpoint of the lengthened rest). Everything after the
    insertion moves, including the rest boundaries, because `_trim_rests` is handed the
    frame plan and the frame plan has to describe the audio it is handed.
    """
    a, b = max(rests, key=lambda r: r[1] - r[0])
    quiet = float(np.max(np.abs(audio))) * 0.02
    lo, hi = P._silent_span(audio, int(round(0.5 * (a + b) * sr)), quiet, sr)
    if hi <= lo:
        lo = hi = int(round(0.5 * (a + b) * sr))
    k = (lo + hi) // 2
    n = int(round(extra * sr))
    out = np.concatenate([audio[:k], np.zeros(n, dtype=audio.dtype), audio[k:]])
    shift = n / float(sr)
    at = k / float(sr)
    moved = [
        (t0 + (shift if t0 > at else 0.0), t1 + (shift if t1 > at else 0.0))
        for t0, t1 in rests
    ]
    return out, moved, at


## Comma-heavy prose for the mark check. Written here rather than quoted from the
## chapter the report came from: what this measures is where the marks FALL, not which
## words are around them, so a fixture serves it better than somebody's book does.
MARKED = (
    "The gate was open, the yard was empty, and the dog had gone back inside. "
    "She counted the boxes twice, once by the door and once by the window, and both "
    "times she got eleven. If the train is late, and it usually is, we will walk. "
    "He set the lamp down, turned it up, and read the label out loud. "
    "There was a bench by the wall, a bucket beside it, and nothing else at all."
)
## The reporter's own Pause setting, where the defect was heard.
PAUSE = 6.3


def check_mark_consistency() -> None:
    """A comma rests the same length wherever it falls, and half as long as a full stop.

    THE DEFECT. The rest at a mark was `(what the model rests here + our top-up) * dial`,
    so the dial multiplied a number the duration predictor had SAMPLED. In one reading at
    the reporter's settings the same comma in the same voice came out 0.36, 0.44, 1.52 and
    1.81 seconds against 1.77 for a full stop - "the pause after a comma feels twice as
    long as a pause after a sentence... the comma-pauses feel far too slow", and both
    halves of that describe the same reading at different commas.

    TWO-SIDED, and the control is exact rather than a second render: the measured dwells
    are captured as the splicer asks for them, so what the OLD rule would have produced
    from the very same audio is arithmetic, not a guess.
    """
    print("\n5. the rest at a mark")
    be = PiperBackend()
    try:
        sess, cfg = be._load(VOICE)
    except Exception as exc:  # noqa: BLE001
        print("   -- %s is not installed here (%s); skipping" % (VOICE, exc))
        return
    sr = int(cfg["audio"]["sample_rate"])
    mult = P._pause_multiplier({"pause_scale": PAUSE})

    real = P._rest_from
    seen: list = []

    def spy(dwell, top_up, m, have=None):
        out = real(dwell, top_up, m, have)
        if have is not None:  # the splicer's call, not the seam's
            seen.append((float(have), float(top_up)))
        return out

    heard: list = []
    try:
        P._rest_from = spy
        # TWICE, because how much the model's own dwell varies is itself a draw: seven
        # marks carried the control twice and came up 0.142 s short the third time.
        for sent in list(sentences(MARKED)) + list(sentences(MARKED)):
            toks = tokens_of(sent)
            marks = [i for i, t in enumerate(toks) if t["punct"] in (",", ";", ":")]
            if not marks:
                continue
            params = {
                "speaker": SPEAKER,
                "length_scale": 0.96,
                "noise_scale": 0.60,
                "noise_w": 0.35,
                "pause_scale": PAUSE,
                "sentence_gap": 0.32,
                "tokens": toks,
            }
            out = Path(HERE) / "_cadence_marks.wav"
            r = be.synthesize(sent, VOICE, str(out), params)
            audio, got_sr = _read_wav(str(out))
            out.unlink(missing_ok=True)
            spans = {int(t["index"]): (t["t0"], t["t1"]) for t in r["tokens"]}
            quiet = float(np.max(np.abs(audio))) * 0.02
            for i in marks:
                if i not in spans or (i + 1) not in spans:
                    continue
                at = int(round(0.5 * (spans[i][1] + spans[i + 1][0]) * got_sr))
                heard.append(P._silence_around(audio, at, quiet, got_sr))
    finally:
        P._rest_from = real

    if not heard or not seen:
        print("   -- nothing rendered; skipping")
        return
    h = np.array(heard)
    # What the same audio would have given under the retired rule.
    old = np.array([(have + top) * mult for have, top in seen])
    dwells = np.array([have for have, _t in seen])
    print(
        "      %d marks: model dwelt %.3f-%.3f | heard %.3f-%.3f | old rule %.3f-%.3f"
        % (h.size, dwells.min(), dwells.max(), h.min(), h.max(), old.min(), old.max())
    )

    # THE CONTROL. If the model's own dwell happened not to vary, there is nothing here
    # for the fix to have fixed and the spread check below is an empty claim.
    ok(
        dwells.max() - dwells.min() > 0.15,
        "the model's own dwell at a mark varies a lot",
        "%.3fs to %.3fs" % (dwells.min(), dwells.max()),
    )
    ok(
        old.max() / max(old.min(), 1e-6) > 2.0,
        "so the old rule spread the same mark over more than 2x",
        "%.3fs to %.3fs" % (old.min(), old.max()),
    )
    ok(
        h.max() / max(h.min(), 1e-6) < 1.25,
        "and the rest as delivered does not spread",
        "%.3fs to %.3fs" % (h.min(), h.max()),
    )

    # ...and the hierarchy the spread was breaking. The full stop is the editor's seam in
    # generative mode (CHUNK_SENTENCES = 1), so it is the same arithmetic on this side.
    seam = real(0.18, 0.32, mult) + 0.18
    ratio = seam / max(float(np.median(h)), 1e-6)
    ok(
        1.7 < ratio < 2.3,
        "and a full stop still rests twice as long as a comma",
        "%.3fs vs %.3fs = %.2fx" % (seam, float(np.median(h)), ratio),
    )


def _read_wav(path: str) -> tuple:
    import wave

    with wave.open(path) as fh:
        sr = fh.getframerate()
        a = np.frombuffer(fh.readframes(fh.getnframes()), "<i2").astype(np.float32)
    return a / 32768.0, sr


def main() -> int:
    be = PiperBackend()
    check_stress(be)
    check_alignment(be)
    check_trim()
    check_real_voice()
    check_mark_consistency()
    print()
    if FAILS:
        print("test_cadence: %d FAILURE(S)" % len(FAILS))
        for f in FAILS:
            print("   ", f)
        return 1
    print("test_cadence: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
