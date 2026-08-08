#!/usr/bin/env python3
"""Objective intelligibility metrics for ghost's voice.

WHY THIS EXISTS
---------------
Every past voice investigation in this project was driven by ear, one artifact
at a time, and next/voice_rca.md records the cost: frication was repeatedly
turned DOWN because it "offended", when the real defect was that non-phonemic
hiss was masking it. Pleasantness and intelligibility were traded against each
other for months without anyone measuring the thing that mattered.

So: this is the gate. It measures what a listener needs in order to identify
words, not what sounds nice. The thresholds below are acceptance numbers from
the phonetics literature. They do not move to make a run pass.

Godot renders; Python measures. `tests/render_fixtures.gd` writes WAV plus an
exact alignment (we synthesized it, so phone boundaries are ground truth rather
than a forced alignment), and everything here reads that pair.

The in-engine gate `tests/pure_say.gd` already covers fricative contrast and
spectral distinctness. This covers what GDScript cannot do cheaply: LPC formant
tracking, vowel-space area, word-boundary trough depth, and inventory-wide
level balance.

USAGE
    python axis/ghost/measure_voice.py                 # render + measure
    python axis/ghost/measure_voice.py --no-render     # measure existing WAVs
    python axis/ghost/measure_voice.py --wer           # add ASR word error rate
    python axis/ghost/measure_voice.py --baseline out.json   # save a baseline
    python axis/ghost/measure_voice.py --against out.json    # diff vs baseline

Only numpy is required. --wer additionally needs torch + transformers and a
one-time model download into the standard Hugging Face cache (outside the repo;
nothing is written to git).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

GHOST = Path(__file__).resolve().parent
OUT = GHOST / "build" / "voice"

# ---------------------------------------------------------------------------
# Phoneme classes. Grouped by what the LISTENER has to tell apart, which is not
# always how the synthesizer groups them.

VOWELS = ["IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER",
          "AY", "EY", "OY", "AW", "OW"]
SIBILANTS = ["S", "SH", "Z", "ZH"]
WEAK_FRIC = ["F", "TH", "V", "DH"]
STOPS_VL = ["P", "T", "K"]
STOPS_VD = ["B", "D", "G"]
NASALS = ["M", "N", "NG"]
LIQUIDS = ["L", "R"]
GLIDES = ["W", "Y"]

CLASSES = [
    ("vowel", VOWELS), ("sibilant", SIBILANTS), ("weak_fric", WEAK_FRIC),
    ("stop_voiceless", STOPS_VL), ("stop_voiced", STOPS_VD),
    ("nasal", NASALS), ("liquid", LIQUIDS), ("glide", GLIDES),
]

# Acceptance bands, dB relative to the mean vowel level. From the connected
# speech literature (Klatt 1980; Stevens, Acoustic Phonetics). A class outside
# its band is not a matter of taste - it is a class the listener will lose.
LEVEL_TARGETS = {
    "sibilant":       (-8.0,   2.0),   # /s/ and /sh/ sit near vowel level
    "weak_fric":      (-25.0, -10.0),  # quiet, but audible by band contrast
    "stop_voiceless": (-15.0,  -5.0),  # measured over the RELEASE, not closure
    "stop_voiced":    (-20.0,  -6.0),
    "nasal":          (-12.0,  -2.0),
    "liquid":         (-8.0,    1.0),
    "glide":          (-8.0,    1.0),
}

# A word boundary has to produce an actual trough or the listener cannot find
# where words begin. Measured as the floor inside the gap, dB below speech RMS.
BOUNDARY_TROUGH_MAX_DB = -25.0

# Unstressed vowels centralize; they must not COLLAPSE. Ratio of unstressed to
# stressed vowel-space area. Natural connected speech runs roughly 0.45-0.7.
VOWEL_HULL_RATIO_MIN = 0.40

# Artifact ceilings. The RCA history (next/voice_rca.md sections 7-16) settled
# at 2-8 legitimate plosive transients per take, so anything approaching one
# per second is the failure the user hears as "static".
CLICKS_PER_SEC_MAX = 0.35
# Spectral slope, dB per octave from 500 Hz to 8 kHz. Steeper than TILT_MIN is
# the "muffled" complaint; flatter than TILT_MAX is thin and sibilant.
TILT_MIN, TILT_MAX = -11.0, -4.0
# Frication should be approximately FLAT at the lips (Klatt 1980 p.977, after
# Stevens 1971). Ghost measured +5.2 to +10.7 dB/oct before the radiation fix.
FRIC_TILT_MIN, FRIC_TILT_MAX = -4.0, 4.0

# Formant tracking works in a decimated band; 11025 Hz is plenty for F1/F2.
LPC_SR = 11025
LPC_ORDER = 12


# ---------------------------------------------------------------------------
# Audio helpers (stdlib + numpy only - scipy and librosa are not installed and
# adding them for four functions is not worth the dependency).

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return x, sr


def db(x: float, floor: float = 1e-12) -> float:
    return 20.0 * math.log10(max(abs(x), floor))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def decimate(x: np.ndarray, sr: int, target: int) -> tuple[np.ndarray, int]:
    """Anti-aliased decimation by an integer factor, via a windowed-sinc FIR."""
    factor = max(1, int(round(sr / target)))
    if factor == 1:
        return x, sr
    cutoff = 0.9 / (2.0 * factor)                       # cycles/sample
    n = 64 * factor + 1
    t = np.arange(n) - (n - 1) / 2.0
    h = np.sinc(2 * cutoff * t) * np.hamming(n)
    h /= h.sum()
    return np.convolve(x, h, mode="same")[::factor], sr // factor


def lpc(x: np.ndarray, order: int) -> np.ndarray | None:
    """Levinson-Durbin on the autocorrelation. Returns [1, a1..ap]."""
    if x.size <= order:
        return None
    r = np.correlate(x, x, mode="full")[x.size - 1: x.size + order]
    if r[0] <= 0 or not np.all(np.isfinite(r)):
        return None
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else r[i]
        k = -acc / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= (1.0 - k * k)
        if e <= 0:
            return None
    return a


def formants(seg: np.ndarray, sr: int) -> list[float]:
    """F1..Fn of one steady segment, by LPC root-solving."""
    y, fs = decimate(seg, sr, LPC_SR)
    if y.size < 128:
        return []
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])          # pre-emphasis
    y = y * np.hamming(y.size)
    a = lpc(y, LPC_ORDER)
    if a is None:
        return []
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0.01]
    if roots.size == 0:
        return []
    freqs = np.angle(roots) * fs / (2 * math.pi)
    bws = -0.5 * (fs / math.pi) * np.log(np.abs(roots) + 1e-12)
    keep = (freqs > 90) & (freqs < fs / 2 - 200) & (bws < 400)
    return sorted(float(f) for f in freqs[keep])


def bark(f: float) -> float:
    return 13.0 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500.0) ** 2)


def hull_area(points: list[tuple[float, float]]) -> float:
    """Convex hull area by monotone chain plus the shoelace formula."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (x1, y1), (x2, y2) = out[-2], out[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    h = half(pts) + half(reversed(pts))
    area = 0.0
    for i in range(len(h)):
        x1, y1 = h[i]
        x2, y2 = h[(i + 1) % len(h)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


# ---------------------------------------------------------------------------
# Metrics

def measure_levels(x: np.ndarray, sr: int, phones: list[dict]) -> dict:
    """Delivered level per phoneme class, dB relative to the mean vowel.

    Stops are measured over their RELEASE, not the whole segment: a stop is
    mostly closure silence, so segment RMS would report the silence and call
    the burst quiet no matter how loud the burst was.
    """
    by_class: dict[str, list[float]] = {}
    per_phone: dict[str, list[float]] = {}
    for ph in phones:
        p = ph["p"]
        i0, i1 = int(ph["t0"] * sr), int(ph["t1"] * sr)
        if i1 <= i0 or i1 > x.size:
            continue
        seg = x[i0:i1]
        if p in STOPS_VL or p in STOPS_VD:
            seg = seg[-int(0.030 * sr):] if seg.size > int(0.030 * sr) else seg
        v = rms(seg)
        if v <= 0:
            continue
        per_phone.setdefault(p, []).append(v)
        for name, members in CLASSES:
            if p in members:
                by_class.setdefault(name, []).append(v)
                break

    if not by_class.get("vowel"):
        return {"error": "no vowels found"}
    ref = float(np.mean(by_class["vowel"]))

    out = {"reference_vowel_rms": ref, "classes": {}, "phones": {}}
    for name, _ in CLASSES:
        vals = by_class.get(name)
        if not vals:
            continue
        rel = db(float(np.mean(vals)) / ref)
        entry = {"n": len(vals), "db_re_vowel": round(rel, 2)}
        if name in LEVEL_TARGETS:
            lo, hi = LEVEL_TARGETS[name]
            entry["target"] = [lo, hi]
            entry["pass"] = bool(lo <= rel <= hi)
        out["classes"][name] = entry
    for p, vals in sorted(per_phone.items()):
        out["phones"][p] = {"n": len(vals),
                            "db_re_vowel": round(db(float(np.mean(vals)) / ref), 2)}
    return out


def measure_vowel_space(x: np.ndarray, sr: int, phones: list[dict]) -> dict:
    """Vowel-space area in Bark^2, split by whether the planner reduced it.

    Sampled at the segment midpoint over a 40% window, which is where a vowel
    is closest to its target and furthest from its neighbours' coarticulation.
    """
    groups: dict[str, list[tuple[float, float]]] = {"stressed": [], "reduced": []}
    tokens: dict[str, list[tuple[float, float]]] = {}
    for ph in phones:
        if ph["p"] not in VOWELS:
            continue
        i0, i1 = int(ph["t0"] * sr), int(ph["t1"] * sr)
        if i1 - i0 < int(0.020 * sr) or i1 > x.size:
            continue
        mid, span = (i0 + i1) // 2, int((i1 - i0) * 0.2)
        f = formants(x[mid - span:mid + span], sr)
        if len(f) < 2:
            continue
        pt = (bark(f[0]), bark(f[1]))
        groups["reduced" if ph.get("reduce", 0.0) > 0.05 else "stressed"].append(pt)
        tokens.setdefault(ph["p"], []).append((f[0], f[1]))

    res: dict = {}
    for name, pts in groups.items():
        res[name] = {"n": len(pts), "hull_bark2": round(hull_area(pts), 2)}
    s, r = res["stressed"]["hull_bark2"], res["reduced"]["hull_bark2"]
    res["ratio"] = round(r / s, 3) if s > 0 else 0.0
    res["ratio_min"] = VOWEL_HULL_RATIO_MIN
    # A convex hull needs points. Below ~12 tokens the area is dominated by
    # which vowels happened to occur, not by how far apart the space is, so the
    # ratio is reported but not gated - a fixture that simply lacks reduced
    # vowels must not read as a vowel-space collapse.
    res["n_sufficient"] = bool(min(res["stressed"]["n"], res["reduced"]["n"]) >= 12)
    res["pass"] = bool(not res["n_sufficient"] or res["ratio"] >= VOWEL_HULL_RATIO_MIN)
    res["per_vowel_hz"] = {
        p: [round(float(np.median([t[0] for t in v]))),
            round(float(np.median([t[1] for t in v]))), len(v)]
        for p, v in sorted(tokens.items())
    }
    return res


def measure_boundaries(x: np.ndarray, sr: int, words: list[dict]) -> dict:
    """How deep the signal actually falls between words.

    A gap the planner inserted is not a gap the listener hears if something
    else - a drone, a reverb tail, a noise floor - holds level through it.
    """
    speech = rms(x)
    troughs: list[tuple[float, float]] = []
    for a, b in zip(words, words[1:]):
        gap = b["t0"] - a["t1"]
        if gap < 0.005:
            continue
        i0, i1 = int(a["t1"] * sr), int(b["t0"] * sr)
        if i1 <= i0 or i1 > x.size:
            continue
        win = max(1, int(0.005 * sr))
        seg = x[i0:i1]
        n = seg.size // win
        floor = min((rms(seg[k * win:(k + 1) * win]) for k in range(max(n, 1))),
                    default=rms(seg))
        troughs.append((gap, db(floor / speech) if speech > 0 else 0.0))

    if not troughs:
        return {"error": "no word gaps found"}
    buckets: dict[str, list[float]] = {}
    for gap, d in troughs:
        key = "<15ms" if gap < 0.015 else "15-40ms" if gap < 0.040 else ">=40ms"
        buckets.setdefault(key, []).append(d)
    med_long = float(np.median(buckets.get(">=40ms") or buckets.get("15-40ms") or [0.0]))
    return {
        "n_gaps": len(troughs),
        "median_trough_db": {k: round(float(np.median(v)), 2) for k, v in buckets.items()},
        "counts": {k: len(v) for k, v in buckets.items()},
        "threshold_db": BOUNDARY_TROUGH_MAX_DB,
        "pass": bool(med_long <= BOUNDARY_TROUGH_MAX_DB),
    }


def measure_bands(x: np.ndarray, sr: int) -> dict:
    """Long-term average spectrum in octave bands. Consonant cues live above
    2 kHz; a take with nothing up there cannot carry them however well the
    segments are balanced."""
    n = 1 << 15
    acc = np.zeros(n // 2 + 1)
    frames = 0
    for i in range(0, max(x.size - n, 1), n // 2):
        acc += np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n), n)) ** 2
        frames += 1
    if frames:
        acc /= frames
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    edges = [(0, 300), (300, 800), (800, 1500), (1500, 2500),
             (2500, 4000), (4000, 6000), (6000, 8000), (8000, 14000)]
    total = acc.sum() or 1.0
    return {f"{lo}-{hi}": round(10 * math.log10(max(acc[(freqs >= lo) & (freqs < hi)].sum() / total, 1e-12)), 2)
            for lo, hi in edges}


def measure_artifacts(x: np.ndarray, sr: int) -> dict:
    """Clicks, saturation and spectral tilt - the three ways a take gets worse
    while its phonetic metrics get better.

    This exists because the first pass of the intelligibility program raised
    consonant energy without restaging the output, and the reported result was
    "more intelligible but full of static". Every level change from here on has
    to answer to these numbers too, or the gate is only measuring half of what
    a listener hears.

    A click is a first-difference outlier against the LOCAL texture, not a
    fixed threshold: real plosive releases are legitimately steep, so the test
    is whether a step stands out from its own neighbourhood.
    """
    peak = float(np.max(np.abs(x))) or 1.0
    d = np.abs(np.diff(x))

    win = int(0.020 * sr)
    pad = (-d.size) % win
    blocks = np.append(d, np.zeros(pad)).reshape(-1, win)
    local = np.repeat(np.median(blocks, axis=1), win)[: d.size]
    thresh = np.maximum(8.0 * local, 0.02 * peak)
    hits = np.flatnonzero(d > thresh)

    events = 0
    last = -(10 ** 9)
    guard = int(0.005 * sr)
    for i in hits:
        if i - last > guard:
            events += 1
            last = int(i)

    dur = x.size / sr
    # KNEE and CLIP_CEIL from voice.gd's broadcast chain: anything above the
    # knee is being shaped by the soft clip rather than passed through.
    return {
        "clicks_per_sec": round(events / max(dur, 0.001), 2),
        "click_events": events,
        "max_step": round(float(np.max(d)), 4),
        "pct_above_knee_0p7": round(100.0 * float(np.mean(np.abs(x) > 0.70)), 3),
        "pct_above_limit_0p85": round(100.0 * float(np.mean(np.abs(x) > 0.85)), 3),
        "peak": round(peak, 4),
        "clicks_per_sec_max": CLICKS_PER_SEC_MAX,
        "pass": bool(events / max(dur, 0.001) <= CLICKS_PER_SEC_MAX
                     and float(np.mean(np.abs(x) > 0.85)) < 0.0005),
    }


def measure_tilt(x: np.ndarray, sr: int) -> dict:
    """Spectral slope from 500 Hz to 8 kHz, dB per octave.

    "Muffled" is not a level problem, it is a tilt problem: a take can sit at
    the right RMS with all its energy under 1 kHz. Natural speech falls roughly
    6-9 dB per octave across this range; steeper than about -11 reads as dull
    however loud the consonants measure.
    """
    n = 1 << 15
    acc = np.zeros(n // 2 + 1)
    frames = 0
    for i in range(0, max(x.size - n, 1), n // 2):
        acc += np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n), n)) ** 2
        frames += 1
    if frames:
        acc /= frames
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    sel = (freqs >= 500) & (freqs <= 8000) & (acc > 0)
    if sel.sum() < 8:
        return {"error": "insufficient spectrum"}
    lf = np.log2(freqs[sel])
    ld = 10.0 * np.log10(acc[sel])
    slope = float(np.polyfit(lf, ld, 1)[0])
    return {
        "db_per_octave": round(slope, 2),
        "target": [TILT_MIN, TILT_MAX],
        "pass": bool(TILT_MIN <= slope <= TILT_MAX),
    }


def measure_tilt_by_class(x, sr, phones):
    """Spectral tilt measured SEPARATELY over vowels and over obstruents.

    The whole-take tilt is worse than useless here: this engine's vowels are
    far too dark and its fricatives far too bright, so the two errors cancel
    and the average passes while both halves fail. Measured 2026-08-08:
    whole-take -4.5 to -5.8 dB/oct, vowels-only -11.2 to -11.5, which is past
    this file's own muffle threshold. Never gate on the mixed number again.
    """
    out = {}
    for name, members in (("vowel", VOWELS),
                          ("obstruent", SIBILANTS + WEAK_FRIC + STOPS_VL + STOPS_VD)):
        chunks = []
        for ph in phones:
            if ph["p"] not in members:
                continue
            i0, i1 = int(ph["t0"] * sr), int(ph["t1"] * sr)
            if i1 - i0 >= int(0.030 * sr) and i1 <= x.size:
                chunks.append(x[i0:i1])
        if not chunks:
            out[name] = {"error": "no tokens"}
            continue
        n = 2048
        acc = np.zeros(n // 2 + 1)
        frames = 0
        for seg in chunks:
            for i in range(0, max(seg.size - n, 1), n // 2):
                w = seg[i:i + n]
                if w.size < n:
                    break
                acc += np.abs(np.fft.rfft(w * np.hanning(n), n)) ** 2
                frames += 1
        if not frames:
            out[name] = {"error": "no frames"}
            continue
        acc /= frames
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        # Vowels are fitted only to 4 kHz. Klatt Table I places F4/F5 as a
        # narrow cluster specifically to give "an energy concentration around
        # 3 to 3.5 kHz and a rapid falloff above about 4 kHz" (1980, p.980), so
        # a single line fit through 8 kHz penalizes the engine for having the
        # intended shape. Fit where the speech information is; the falloff
        # above it is a design feature, not darkness. Obstruents keep the full
        # range because their target really is flat to the top.
        hi = 4000.0 if name == "vowel" else 8000.0
        sel = (freqs >= 500) & (freqs <= hi) & (acc > 0)
        slope = float(np.polyfit(np.log2(freqs[sel]), 10 * np.log10(acc[sel]), 1)[0])
        entry = {"db_per_octave": round(slope, 2), "n_tokens": len(chunks)}
        if name == "vowel":
            entry["target"] = [TILT_MIN, TILT_MAX]
            entry["pass"] = bool(TILT_MIN <= slope <= TILT_MAX)
        else:
            entry["target"] = [FRIC_TILT_MIN, FRIC_TILT_MAX]
            entry["pass"] = bool(FRIC_TILT_MIN <= slope <= FRIC_TILT_MAX)
        out[name] = entry
    return out


def measure_wer(wavs: list[tuple[str, Path, str]]) -> dict:
    """ASR word error rate - the primary intelligibility metric.

    The model is a TEST ORACLE. It never touches the render path and ships
    nothing, so it does not bear on the project's no-generative-AI constraint;
    it is here for the same reason a reference decoder is here, to tell us
    whether a human could recover the words. Cached in the standard Hugging
    Face location outside the repo.
    """
    try:
        import torch  # noqa: F401
        from transformers import pipeline
    except ImportError as exc:
        return {"error": f"torch/transformers unavailable: {exc}"}

    model = "openai/whisper-small.en"
    try:
        asr = pipeline("automatic-speech-recognition", model=model, device=-1)
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"could not load {model}: {exc}"}

    def norm(s: str) -> list[str]:
        keep = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in s)
        return keep.split()

    def edit(a: list[str], b: list[str]) -> int:
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, y in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
            prev = cur
        return prev[-1]

    # WER is only meaningful on connected speech. An isolated minimal-pair list
    # is out of distribution for an ASR language model: the first run of this
    # harness scored `pairs` at 1348% because whisper fell into a repetition
    # loop and emitted "pin" forty times. That number measures the decoder, not
    # the synthesizer. Minimal pairs need a forced-choice listener - the human
    # rhyme test in next/voice_intelligibility.md section 3 - so they are
    # excluded here rather than silently poisoning the mean.
    NOT_CONNECTED = {"pairs"}

    out: dict = {"model": model, "fixtures": {}, "excluded": sorted(NOT_CONNECTED)}
    for name, path, text in wavs:
        if name in NOT_CONNECTED:
            continue
        hyp = asr(str(path), chunk_length_s=30)["text"]
        ref_w, hyp_w = norm(text), norm(hyp)
        d = edit(ref_w, hyp_w)
        out["fixtures"][name] = {
            "wer": round(d / max(len(ref_w), 1), 4),
            "ref_words": len(ref_w),
            "hypothesis": hyp.strip()[:600],
        }
    vals = [f["wer"] for f in out["fixtures"].values()]
    out["mean_wer"] = round(float(np.mean(vals)), 4) if vals else None
    return out


# ---------------------------------------------------------------------------

def render(out_dir: Path) -> None:
    cmd = ["godot", "--headless", "--path", str(GHOST),
           "--script", "tests/render_fixtures.gd", "--", str(out_dir)]
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in r.stdout.splitlines():
        if "render_fixtures" in line:
            print("  " + line)
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"render failed ({r.returncode})")


def analyze(out_dir: Path, want_wer: bool) -> dict:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    report: dict = {"engine": {k: v for k, v in manifest.items() if k != "fixtures"},
                    "fixtures": {}}
    wav_refs: list[tuple[str, Path, str]] = []

    for f in manifest["fixtures"]:
        name = f["name"]
        side = json.loads((out_dir / f"{name}.json").read_text())
        x, sr = read_wav(out_dir / f"{name}.wav")
        wav_refs.append((name, out_dir / f"{name}.wav", side["text"]))
        report["fixtures"][name] = {
            "dur": round(side["dur"], 2),
            "peak_db": round(db(float(np.max(np.abs(x)))), 2),
            "rms_db": round(db(rms(x)), 2),
            "levels": measure_levels(x, sr, side["phones"]),
            "vowel_space": measure_vowel_space(x, sr, side["phones"]),
            "boundaries": measure_boundaries(x, sr, side["words"]),
            "bands_db": measure_bands(x, sr),
            "artifacts": measure_artifacts(x, sr),
            "tilt": measure_tilt(x, sr),
            "tilt_by_class": measure_tilt_by_class(x, sr, side["phones"]),
        }

    if want_wer:
        report["wer"] = measure_wer(wav_refs)
    return report


def summarize(report: dict, baseline: dict | None) -> bool:
    def prior(path: list[str]):
        node = baseline
        for k in path:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node

    def delta(val, path) -> str:
        old = prior(path)
        if old is None or not isinstance(old, (int, float)):
            return ""
        d = val - old
        return f"  ({d:+.2f} vs baseline)" if abs(d) >= 0.01 else "  (unchanged)"

    print("\nengine: " + ", ".join(f"{k}={v}" for k, v in report["engine"].items()))
    ok = True
    for name, fx in report["fixtures"].items():
        print(f"\n=== {name}  {fx['dur']}s  peak {fx['peak_db']} dB  rms {fx['rms_db']} dB")

        print("  level by class (dB re mean vowel)")
        for cls, e in fx["levels"].get("classes", {}).items():
            if "target" not in e:
                print(f"    {cls:<15} {e['db_re_vowel']:>7.2f}   n={e['n']}")
                continue
            mark = "ok  " if e["pass"] else "FAIL"
            ok &= e["pass"]
            lo, hi = e["target"]
            print(f"    {cls:<15} {e['db_re_vowel']:>7.2f}   target [{lo:g}, {hi:g}]  {mark}"
                  + delta(e["db_re_vowel"], ["fixtures", name, "levels", "classes", cls, "db_re_vowel"]))

        vs = fx["vowel_space"]
        mark = "ok" if vs.get("pass") else "FAIL"
        ok &= bool(vs.get("pass"))
        print(f"  vowel space  stressed {vs['stressed']['hull_bark2']} Bark^2 (n={vs['stressed']['n']})"
              f"  reduced {vs['reduced']['hull_bark2']} (n={vs['reduced']['n']})"
              f"  ratio {vs['ratio']} >= {vs['ratio_min']}  {mark}"
              + delta(vs["ratio"], ["fixtures", name, "vowel_space", "ratio"]))

        b = fx["boundaries"]
        if "error" not in b:
            mark = "ok" if b["pass"] else "FAIL"
            ok &= b["pass"]
            print(f"  boundaries   {b['median_trough_db']}  counts {b['counts']}"
                  f"  threshold {b['threshold_db']}  {mark}")

        a = fx["artifacts"]
        ok &= a["pass"]
        print(f"  artifacts    {a['clicks_per_sec']}/s (max {a['clicks_per_sec_max']})  "
              f"max step {a['max_step']}  peak {a['peak']}  "
              f">knee {a['pct_above_knee_0p7']}%  >limit {a['pct_above_limit_0p85']}%  "
              + ("ok" if a["pass"] else "FAIL"))
        t = fx["tilt"]
        if "error" not in t:
            ok &= t["pass"]
            print(f"  tilt         {t['db_per_octave']} dB/oct  target {t['target']}  "
                  + ("ok" if t["pass"] else "FAIL")
                  + delta(t["db_per_octave"], ["fixtures", name, "tilt", "db_per_octave"]))
        for cls, e in fx["tilt_by_class"].items():
            if "error" in e:
                continue
            ok &= e["pass"]
            print(f"  tilt/{cls:<9} {e['db_per_octave']:>7.2f} dB/oct  target {e['target']}  "
                  + ("ok" if e["pass"] else "FAIL")
                  + delta(e["db_per_octave"], ["fixtures", name, "tilt_by_class", cls, "db_per_octave"]))
        print("  bands (dB re total): " + "  ".join(f"{k}:{v}" for k, v in fx["bands_db"].items()))

    if "wer" in report:
        w = report["wer"]
        if "error" in w:
            print(f"\nWER: unavailable - {w['error']}")
        else:
            print(f"\nWER ({w['model']}), mean {w['mean_wer']:.1%}")
            for name, f in w["fixtures"].items():
                print(f"  {name:<11} {f['wer']:.1%}  ({f['ref_words']} words)"
                      + delta(f["wer"], ["wer", "fixtures", name, "wer"]))
                print(f"      heard: {f['hypothesis'][:160]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-render", action="store_true", help="measure existing WAVs")
    ap.add_argument("--wer", action="store_true", help="add ASR word error rate")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--baseline", type=Path, help="write this run as a baseline")
    ap.add_argument("--against", type=Path, help="compare against a saved baseline")
    args = ap.parse_args()

    if not args.no_render:
        render(args.out)
    report = analyze(args.out, args.wer)

    baseline = json.loads(args.against.read_text()) if args.against and args.against.exists() else None
    ok = summarize(report, baseline)

    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nreport -> {args.out / 'report.json'}")
    if args.baseline:
        args.baseline.write_text(json.dumps(report, indent=1))
        print(f"baseline -> {args.baseline}")
    print("\nGATE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
