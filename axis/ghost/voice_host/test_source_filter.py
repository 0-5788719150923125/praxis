#!/usr/bin/env python3
"""Gate for the two source-filter effects: the whisper, and the formant lock.

Run it directly with the voice venv's python - no pytest needed:

    ~/.local/share/godot/app_userdata/ghost/voice_venv/bin/python \
        axis/ghost/voice_host/test_source_filter.py

WHY IT IS BUILT ON A SYNTHETIC VOWEL. Both effects make a claim about what they
leave ALONE, and a claim like that cannot be checked against a recording whose
true formants nobody knows. So the signal under test is built here from a pulse
train driven through three resonators at frequencies this file chose: the answer
is known before the measurement, and a measurement that disagrees with it is the
effect being wrong rather than the estimator being unlucky.

The three claims:

  1. A resample moves the formants. This is the BUG, and it is checked first,
     because a gate that only shows the fix working cannot tell you the fix was
     needed - and this one is also the A/B the fix is measured against.
  2. `_restore_formants` puts them back while leaving the pitch where the
     resample moved it. Both halves matter: formants restored AND pitch still
     shifted, or it has simply undone the arc.
  3. `_whisper` takes the voice out and leaves the words in - harmonicity to the
     floor, formants where they were, level unchanged.

Real synthesis is then run through the same measurements if a checkpoint happens
to be on this machine, and skipped with a note if not. The synthetic case is the
gate; the rendered one is the sanity check that the gate is about this voice.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GHOST = HERE.parent
for p in (str(HERE), str(GHOST)):
    if p not in sys.path:
        sys.path.insert(0, p)

import measure_voice as MV  # noqa: E402
from backends.piper import (  # noqa: E402
    _muffle,
    _resample,
    _restore_formants,
    _whisper,
)

SR = 22050
## generative_editor._open_up at the top of its travel - the largest `dynamics` the backend
## can be handed. Mirrors piper.DEPTH_TOP; the point of the check below is that the two
## agree, so it is written out here rather than imported from the thing under test.
DEPTH_TOP_FOR_TEST = 2.5
F0 = 118.0
FORMANTS = (700.0, 1220.0, 2600.0)  # a synthetic open vowel
BANDWIDTHS = (80.0, 90.0, 140.0)

_fails: list[str] = []


def ok(cond: bool, what: str, detail: str = "") -> None:
    print(
        ("  ok   " if cond else "  FAIL ") + what + (("  " + detail) if detail else "")
    )
    if not cond:
        _fails.append(what)


# --- the signal under test --------------------------------------------------


def vowel(seconds: float = 1.2, f0: float = F0) -> np.ndarray:
    """A pulse train through three resonators - a vowel whose formants we know.

    Filtered in the frequency domain rather than sample by sample: the transfer
    function of a two-pole resonator is written out directly, which needs no IIR
    routine and so needs no scipy (the backend does not have one either).
    """
    n = int(seconds * SR)
    x = np.zeros(n)
    period = SR / f0
    x[(np.arange(int(n / period)) * period).astype(int)] = 1.0
    spec = np.fft.rfft(x)
    w = 2.0 * math.pi * np.fft.rfftfreq(n)
    for f, bw in zip(FORMANTS, BANDWIDTHS):
        r = math.exp(-math.pi * bw / SR)
        theta = 2.0 * math.pi * f / SR
        z = np.exp(-1j * w)
        denom = 1.0 - 2.0 * r * math.cos(theta) * z + (r * r) * z * z
        spec = spec / denom
    y = np.fft.irfft(spec, n)
    y *= 0.5 / max(1e-9, np.max(np.abs(y)))
    # a gentle fade so the ends are not a click the estimators have to survive
    ramp = int(0.02 * SR)
    y[:ramp] *= np.linspace(0.0, 1.0, ramp)
    y[-ramp:] *= np.linspace(1.0, 0.0, ramp)
    return y.astype(np.float32)


# --- measurement ------------------------------------------------------------


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Autocorrelation by FFT, normalised to 1 at lag zero.

    Not np.correlate: that is O(n^2) and takes half a minute on a second of audio
    here, which is a gate nobody runs. This is the same Wiener-Khinchin route the
    effect's own LPC uses.
    """
    seg = x[x.size // 4 : 3 * x.size // 4].astype(np.float64)
    seg = seg - seg.mean()
    size = 1
    while size < 2 * seg.size:
        size *= 2
    spec = np.fft.rfft(seg, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[: seg.size]
    return ac / max(ac[0], 1e-12)


def measure_f0(x: np.ndarray, lo: float = 60.0, hi: float = 400.0) -> float:
    """Fundamental by autocorrelation, over the steady middle of the signal."""
    ac = _autocorr(x)
    lo_lag, hi_lag = int(SR / hi), int(SR / lo)
    band = ac[lo_lag:hi_lag]
    if band.size == 0:
        return 0.0
    return float(SR / (lo_lag + int(np.argmax(band))))


def harmonicity(x: np.ndarray, lo: float = 60.0, hi: float = 400.0) -> float:
    """Peak normalised autocorrelation in the pitch band: 1 buzzes, 0 breathes."""
    ac = _autocorr(x)
    band = ac[int(SR / hi) : int(SR / lo)]
    return float(np.max(band)) if band.size else 0.0


def measure_formants(x: np.ndarray) -> list[float]:
    """Formants by LPC root-solving, at the full rate and a generous order.

    measure_voice.formants is the project's tracker and is used for the real
    voice, but it decimates to 11 kHz at order 12, and on a synthetic vowel - three
    resonances and a pulse train, with none of the spectral clutter of a real one -
    that fit spends a pole pair on a harmonic and reports it between F2 and F3.
    A higher order at the full rate resolves the three cleanly. Same Levinson, same
    root-solve, same bandwidth filter; only the resolution differs.
    """
    mid = x[x.size // 2 - 700 : x.size // 2 + 700].astype(np.float64)
    y = np.append(mid[0], mid[1:] - 0.97 * mid[:-1]) * np.hamming(mid.size)
    a = MV.lpc(y, 20)
    if a is None:
        return []
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0.01]
    freqs = np.angle(roots) * SR / (2.0 * math.pi)
    bws = -0.5 * (SR / math.pi) * np.log(np.abs(roots) + 1e-12)
    keep = (freqs > 200.0) & (freqs < SR / 2 - 500.0) & (bws < 500.0)
    return sorted(float(f) for f in freqs[keep])


def nearest(found: list[float], wanted: tuple) -> list[float]:
    """Each expected formant paired with the pole that landed closest to it.

    By position rather than by rank: a fit that finds a spurious pole would
    otherwise shift every formant after it and report a failure that is the
    ESTIMATOR's, not the effect's. A formant that genuinely moved is still caught,
    because the nearest pole to where it used to be is then far away.
    """
    return [min(found, key=lambda f: abs(f - w)) if found else 0.0 for w in wanted]


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else 0.0


def semis(ratio: float) -> float:
    return 12.0 * math.log2(ratio) if ratio > 0 else 0.0


# --- the claims -------------------------------------------------------------


def check_resample_moves_formants(v: np.ndarray, ratio: float) -> None:
    """1. The bug, stated as a measurement rather than as a belief."""
    print("\nA resample alone (the old arc), %+.1f semitones:" % semis(ratio))
    shifted = _resample(v, ratio)
    f0 = measure_f0(shifted)
    # matched against where the resample MOVED them to, so the pairing is right
    # at the far end of the dial, and then reported against where they started
    fs = nearest(measure_formants(shifted), tuple(f * ratio for f in FORMANTS))
    ok(
        abs(pct(f0, F0 * ratio)) < 6.0,
        "pitch moves with the resample",
        "f0 %.0f Hz, wanted %.0f" % (f0, F0 * ratio),
    )
    drift = [pct(f, w) for f, w in zip(fs, FORMANTS)]
    ok(
        len(fs) == 3 and min(abs(d) for d in drift) > 8.0,
        "...and so do the formants, which is the bug",
        "F1..F3 " + " ".join("%+.0f%%" % d for d in drift),
    )


def check_formant_lock(v: np.ndarray, ratio: float) -> None:
    """2. Pitch moved, speaker held."""
    print("\nResample plus formant lock, %+.1f semitones:" % semis(ratio))
    shifted = _resample(v, ratio)
    fixed = _restore_formants(shifted, v, ratio, SR)
    ok(fixed.size == shifted.size, "length is unchanged", "%d samples" % fixed.size)
    f0 = measure_f0(fixed)
    ok(
        abs(pct(f0, F0 * ratio)) < 6.0,
        "the pitch move SURVIVES the lock",
        "f0 %.0f Hz, wanted %.0f" % (f0, F0 * ratio),
    )
    fs = nearest(measure_formants(fixed), FORMANTS)
    drift = [abs(pct(f, w)) for f, w in zip(fs, FORMANTS)]
    ok(
        len(fs) == 3 and max(drift) < 6.0,
        "the formants are back where the speaker had them",
        "F1..F3 " + " ".join("%.1f%%" % d for d in drift),
    )


def check_whisper(v: np.ndarray) -> None:
    """3. Voice out, words in."""
    print("\nWhisper:")
    before = harmonicity(v)
    w = _whisper(v, SR, 1.0)
    after = harmonicity(w)
    ok(w.size == v.size, "length is unchanged", "%d samples" % w.size)
    ok(before > 0.5, "the input is voiced to begin with", "harmonicity %.2f" % before)
    ok(
        after < 0.15,
        "the fundamental is gone",
        "harmonicity %.2f -> %.2f" % (before, after),
    )
    fs = nearest(measure_formants(w), FORMANTS)
    drift = [abs(pct(f, wanted)) for f, wanted in zip(fs, FORMANTS)]
    # F1 is allowed further than the other two: the envelope is deliberately
    # smoothed (see _WHISPER_GAMMA in the backend) and that reads F1 high, which
    # is the direction whispered vowels move anyway. F2 and F3 are where a vowel
    # is identified, and the resample above moves ALL THREE by 12-23%, so this
    # still tells the two apart by a wide margin.
    ok(
        len(fs) == 3 and drift[0] < 15.0 and max(drift[1:]) < 8.0,
        "the vocal tract is still there - the words survive",
        "F1..F3 " + " ".join("%.1f%%" % d for d in drift),
    )
    lvl = 20.0 * math.log10(
        max(1e-9, float(np.sqrt(np.mean(w**2))))
        / max(1e-9, float(np.sqrt(np.mean(v.astype(np.float64) ** 2))))
    )
    ok(abs(lvl) < 2.0, "the level is held", "%+.2f dB" % lvl)
    half = _whisper(v, SR, 0.5)
    mid = harmonicity(half)
    ok(
        after < mid < before,
        "half way is half way - a stage whisper, not a switch",
        "harmonicity %.2f, between %.2f and %.2f" % (mid, after, before),
    )
    # ...AND IT DOES NOT DUCK ON THE WAY THROUGH. The two halves of this blend are
    # uncorrelated - noise through the tract against the periodic voice under it - so they
    # add in power, and a plain `a*wet + (1-a)*dry` sits at sqrt(a^2+(1-a)^2) of the level:
    # nothing at either end and -3 dB in the middle, which is exactly where Hushed lives.
    # Reported as "with hushed specifically, it is too quiet", and invisible to every check
    # here because both ends measured fine.
    ref = float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))
    worst = 0.0
    for a in (0.1, 0.25, 0.45, 0.5, 0.6, 0.75, 0.9):
        blend = _whisper(v, SR, a)
        db = 20.0 * math.log10(
            max(1e-9, float(np.sqrt(np.mean(blend.astype(np.float64) ** 2))))
            / max(ref, 1e-9)
        )
        worst = max(worst, abs(db))
    ok(
        worst < 0.5,
        "no blend amount ducks the level",
        "worst %.2f dB over the range" % worst,
    )
    ok(
        np.array_equal(_whisper(v, SR, 1.0), w),
        "the same input renders the same whisper twice",
    )


def check_muffle(v: np.ndarray) -> None:
    """4. A voice through something: the top taken off, the words left in place."""
    print("\nMuffle:")
    # Thresholds measured on THIS fixture, which is three resonators and therefore has less
    # top than speech does: the same 0.55 reads 4.5 dB on a real rendered sentence.
    for amount, want_tilt in ((0.55, 2.5), (1.0, 6.0)):
        m = _muffle(v, SR, amount)
        ok(
            m.size == v.size,
            "length is unchanged at %.2f" % amount,
            "%d samples" % m.size,
        )
        spec_v, spec_m = np.abs(np.fft.rfft(v.astype(np.float64))), np.abs(
            np.fft.rfft(m.astype(np.float64))
        )
        freq = np.fft.rfftfreq(v.size, 1.0 / SR)
        low = (freq > 100.0) & (freq < 1000.0)
        high = (freq > 3000.0) & (freq < 8000.0)
        tilt = 20.0 * math.log10(
            (spec_m[high].mean() / spec_m[low].mean())
            / (spec_v[high].mean() / spec_v[low].mean())
        )
        ok(
            tilt < -want_tilt,
            "at %.2f the top is off by %.1f dB (wanted %.0f+)"
            % (amount, -tilt, want_tilt),
        )
        lvl = 20.0 * math.log10(
            max(1e-9, float(np.sqrt(np.mean(m.astype(np.float64) ** 2))))
            / max(1e-9, float(np.sqrt(np.mean(v.astype(np.float64) ** 2))))
        )
        # Quieter, but only a little: a covered voice is not a distant one, and this must
        # never become the Presence dial by another name.
        ok(
            -2.5 < lvl < 0.0,
            "at %.2f it is a touch quieter, not far away" % amount,
            "%+.2f dB" % lvl,
        )
    ok(np.array_equal(_muffle(v, SR, 0.0), v), "at 0 it is the input, untouched")


def check_real_voice() -> None:
    """The same two questions asked of the actual checkpoint, if there is one."""
    root = Path.home() / ".local" / "share" / "ghost" / "voices" / "piper"
    models = sorted(root.glob("*.onnx")) if root.is_dir() else []
    if not models:
        print("\nReal voice: no checkpoint on this machine, skipped.")
        return
    voice = models[0].stem
    print("\nReal voice (%s):" % voice)
    try:
        import backends.piper as P

        be = P.PiperBackend()
        be._load(voice)
        tokens = [
            {"text": w, "punct": "." if w == "shadows" else ""}
            for w in "the harbour lights are drowning in the shadows".split()
        ]
        out = Path("/tmp/ghost_source_filter_probe.wav")
        # ONE render, and the transform applied to THAT. Two renders cannot be
        # compared at all: VITS samples its own durations, so the same sentence
        # comes back a different length every time - which is what the first cut
        # of this check measured when it thought the whisper changed the timing.
        # WITH THE NOISE OFF. VITS samples its durations and its flow, so the same
        # sentence is a different performance every time and the loudest 300 ms lands on a
        # different phone - measured 0.86 harmonicity one run and 0.77 the next, against a
        # threshold in between. That is the project's own rule for any measurement (see
        # voice_host/vowel_probe.py, which had to learn it the same way) and it makes this
        # check bit-exact rather than a sample of one.
        be.synthesize(
            "",
            voice,
            str(out),
            {
                "tokens": tokens,
                "phonemizer": "espeak",
                "noise_scale": 0.0,
                "noise_w": 0.0,
            },
        )
        dry_raw, sr = MV.read_wav(out)
        dry = np.asarray(dry_raw, dtype=np.float64)
        wet = np.asarray(P._whisper(dry.astype(np.float32), sr, 1.0), dtype=np.float64)
    except Exception as exc:  # a missing espeak or a cold model is not a failure
        print("  ..skipped: %s" % exc)
        return

    def loudest(x, seconds=0.3):
        """The most energetic window: a whole sentence is mostly not a vowel."""
        n = int(seconds * sr)
        if x.size <= n:
            return x
        power = np.convolve(x * x, np.ones(n) / n, mode="valid")
        return x[int(np.argmax(power)) :][:n]

    hb = harmonicity(loudest(dry), 60.0, 400.0)
    ha = harmonicity(loudest(wet), 60.0, 400.0)
    ok(hb > 0.4, "the rendered voice is voiced to begin with", "harmonicity %.2f" % hb)
    ok(ha < hb * 0.35, "a real sentence loses its voice too", "%.2f -> %.2f" % (hb, ha))
    ok(
        wet.size == dry.size,
        "and keeps its timing, so the subtitles still fit",
        "%d samples" % wet.size,
    )
    lvl = 20.0 * math.log10(
        max(1e-9, float(np.sqrt(np.mean(wet**2))))
        / max(1e-9, float(np.sqrt(np.mean(dry**2))))
    )
    ok(abs(lvl) < 2.0, "at the level it was rendered at", "%+.2f dB" % lvl)

    # THE PARAGRAPH ARC MUST NOT TILT THE PACE. The arc buys its pitch move by rendering at
    # a different length scale and playing back to compensate, and the two do not cancel by
    # themselves - part of every sentence is frame-quantised and does not scale - so the
    # reading used to accelerate into each paragraph and drag out of it, by 5.1% and 2.0% at
    # the top of the dial. "The pace grows slower and slower over time, decreasing in speed
    # with the increase in the Arc value." Measured here on the real voice, both ends of the
    # arc against no arc at all, with the model's noise off so a render is repeatable.
    def render(extra):
        be.synthesize(
            "",
            voice,
            str(out),
            {
                "tokens": tokens,
                "phonemizer": "espeak",
                "noise_scale": 0.0,
                "noise_w": 0.0,
                **extra,
            },
        )
        a, rate = MV.read_wav(out)
        return float(len(a)) / rate, np.asarray(a, dtype=np.float64), rate

    flat, _, _ = render({})
    for u, tag in ((0.0, "opening"), (1.0, "closing")):
        got, wave, rate = render({"prosody_arc": 4.0, "plan_u": u, "plan_v": 0.0})
        drift = 100.0 * (got - flat) / flat
        ok(
            abs(drift) < 3.5,
            "the arc leaves the %s sentence's pace alone" % tag,
            "%+.1f%%" % drift,
        )
    # AND NEITHER MAY DYNAMICS RUN AWAY WITH IT. The timing coefficients in
    # `_discourse_plan` are written as the figures they should deliver at the TOP of the
    # dial, and for a while the editor's own curve multiplied every one of them by 2.5
    # without this file knowing - a sentence at the end of a paragraph AND a section ran at
    # rate 0.375, two and a half times its own length. Reported as "the cadence/pace/speed
    # of the voice becomes slower and slower and slower", and blamed on the Arc, which at
    # the reported settings was contributing 0.0% of it.
    #
    # The claim is about the RATIO between a unit's first sentence and its last, at the top
    # of the dial, which is where any future rescaling of that curve would show up first.
    top = DEPTH_TOP_FOR_TEST
    opening, _, _ = render({"dynamics": top, "plan_u": 0.0, "plan_v": 0.0})
    closing, _, _ = render({"dynamics": top, "plan_u": 1.0, "plan_v": 1.0})
    slowdown = 100.0 * (closing - opening) / opening
    # The design figure is 18% off the rate for the paragraph and 7% for the section, which
    # is 33% of duration if a sentence sits at the end of both - a lot, deliberately, at the
    # very top of the dial. The bound is there to catch the coefficients being multiplied
    # again behind this file's back, which is exactly what happened last time.
    ok(
        5.0 < slowdown < 35.0,
        "at the top of Dynamics a unit closes slower than it opens, and by a bounded amount",
        "%+.0f%%" % slowdown,
    )
    flat_dyn, _, _ = render({"dynamics": 0.0, "plan_u": 1.0, "plan_v": 1.0})
    ok(
        abs(flat_dyn - flat) / flat < 0.02,
        "...and with Dynamics at zero a sentence is the same length wherever it sits",
        "%+.1f%%" % (100.0 * (flat_dyn - flat) / flat),
    )

    # ...while still moving the pitch, or the pace would be flat for the wrong reason.
    _, high, rate = render({"prosody_arc": 4.0, "plan_u": 0.0, "plan_v": 0.0})
    _, low, _ = render({"prosody_arc": 4.0, "plan_u": 1.0, "plan_v": 0.0})

    def f0(x):
        # On the loudest window, not the middle half: a whole sentence is mostly not a
        # vowel, and an autocorrelation over one lands wherever the energy happens to be -
        # measured -3.9 semitones across an arc that plainly rises, because the two renders
        # were being measured on different phones.
        seg = loudest(x) - loudest(x).mean()
        n = 1
        while n < 2 * seg.size:
            n *= 2
        sp = np.fft.rfft(seg, n)
        ac = np.fft.irfft(sp * np.conj(sp), n)[: seg.size]
        lo_l, hi_l = int(rate / 400.0), int(rate / 60.0)
        return float(rate / (lo_l + int(np.argmax(ac[lo_l:hi_l]))))

    span = 12.0 * math.log2(f0(high) / f0(low))
    ok(
        span > 1.5,
        "and the arc is still an arc",
        "%.2f semitones across the unit" % span,
    )


def main() -> int:
    v = vowel()
    print("Synthetic vowel: f0 %.0f Hz, F1..F3 %s" % (F0, FORMANTS))
    f0 = measure_f0(v)
    fs = nearest(measure_formants(v), FORMANTS)
    ok(abs(pct(f0, F0)) < 4.0, "the fixture measures as built (pitch)", "%.0f Hz" % f0)
    ok(
        len(fs) == 3 and max(abs(pct(f, w)) for f, w in zip(fs, FORMANTS)) < 6.0,
        "the fixture measures as built (formants)",
        " ".join("%.0f" % f for f in fs),
    )
    for st in (3.0, -3.0):
        ratio = 2.0 ** (st / 12.0)
        check_resample_moves_formants(v, ratio)
        check_formant_lock(v, ratio)
    check_whisper(v)
    check_muffle(v)
    check_real_voice()
    print()
    if _fails:
        for f in _fails:
            print("test_source_filter: FAIL - " + f)
        print("test_source_filter: %d FAILED" % len(_fails))
        return 1
    print("test_source_filter: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
