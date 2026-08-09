"""Piper voices, run as raw ONNX. No GPL code is imported, linked or shipped.

THE LICENSING SITUATION, because it drives every design choice here
-------------------------------------------------------------------
`rhasspy/piper` was archived 2025-10-06. Development moved to
`OHF-Voice/piper1-gpl`, which relicensed to **GPL-3.0-or-later** in v1.3.0
because the wheel vendors eSpeak-NG. So the obvious route - `pip install
piper-tts` - would make a shipped game GPL.

It is also unnecessary. A Piper voice is two files: an `.onnx` graph and an
`.onnx.json` config, distributed from an MIT-tagged Hugging Face repository as
DATA. The config carries a `phoneme_id_map`. ghost already has its own G2P, so
we can map our phonemes to ids ourselves and call onnxruntime (MIT) directly.
Nothing GPL is involved at inference or distribution time.

THE PER-VOICE TRAP, which is worse than it looks
------------------------------------------------
Voice checkpoints are licensed individually AND the license is transitive
through fine-tuning, which is not obvious and is not flagged anywhere machine
readable. Piper's own default demo voice, `lessac`, is Blizzard 2013, which
restricts use to research and explicitly excludes "the development, marketing,
commercialisation, sale or licencing of voice synthesis products". Voices
fine-tuned FROM lessac inherit that: amy, joe, hfc_female, hfc_male, libritts_r.
Separately ryan and the hfc pair are CC BY-NC-SA, and kathleen is CC0 data
fine-tuned from ryan.

So VOICES below is an allowlist, not a catalogue, and every entry records its
dataset AND its derivation chain. Adding a voice means reading its MODEL_CARD.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from . import Backend, BackendError, register

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# VITS hop length: one duration-predictor frame is this many samples.
HOP_LENGTH = 256

# Sentence-final marks. ghost's front end already emits these as phones, so the
# split costs nothing and needs no text.
SENTENCE_END = {".", "!", "?"}


def _split_sentences(phones: list) -> list[list]:
    """Break a phone stream at sentence-final punctuation, keeping the mark."""
    out, cur = [], []
    for p in phones:
        cur.append(p)
        if str(p) in SENTENCE_END:
            out.append(cur)
            cur = []
    if cur and any(str(x).strip() for x in cur):
        out.append(cur)
    return out or [list(phones)]

# Phoneme id stream shape, from piper1-gpl docs/ALIGNMENTS.md:
#   [BOS, PAD, id, PAD, id, PAD, ..., EOS]
BOS, EOS, PAD = 1, 2, 0

# Commercially clean English voices only. Verified MODEL_CARD by MODEL_CARD,
# 2026-08-09. Anything not on this list needs its card read before it is added.
VOICES: dict[str, dict] = {
    "en_US-ljspeech-medium": {
        "path": "en/en_US/ljspeech/medium/en_US-ljspeech-medium",
        "license": "Public domain (LJ Speech)",
        "chain": "trained from scratch",
        "speakers": 1,
        "notes": "Single speaker, small, the quickest thing to prove the pipeline.",
    },
    "en_US-libritts-high": {
        "path": "en/en_US/libritts/high/en_US-libritts-high",
        "license": "CC BY 4.0 (LibriTTS, openslr.org/60) - attribution required",
        "chain": "trained from scratch",
        "speakers": 904,
        "notes": "904 speakers under one checkpoint. The natural analogue of the "
                 "fishing game's genome: identity becomes a speaker id you can "
                 "throw, catch and interpolate.",
    },
    "en_US-kristin-medium": {
        "path": "en/en_US/kristin/medium/en_US-kristin-medium",
        "license": "Public domain (LibriVox, ~11.5h)",
        "chain": "trained from scratch",
        "speakers": 1,
    },
    "en_US-norman-medium": {
        "path": "en/en_US/norman/medium/en_US-norman-medium",
        "license": "Public domain (LibriVox, ~15.5h)",
        "chain": "trained from scratch",
        "speakers": 1,
    },
    "en_US-john-medium": {
        "path": "en/en_US/john/medium/en_US-john-medium",
        "license": "Public domain (LibriVox, ~12.5h)",
        "chain": "fine-tuned from kristin (clean chain)",
        "speakers": 1,
    },
}


@register
class PiperBackend(Backend):
    name = "piper"

    @classmethod
    def describe(cls) -> dict:
        d = super().describe()
        d.update({
            "streaming": False,        # per-sentence today; sub-utterance is possible
            "phoneme_input": True,     # the point of this backend
            "duration_control": True,  # length_scale
            "pitch_control": False,    # VITS has no direct f0 handle
            "timings": True,           # exact, from the duration predictor
            "reference_audio": False,
            "singing": False,
        })
        return d

    @classmethod
    def owns(cls, voice: str) -> bool:
        return voice in VOICES

    def __init__(self) -> None:
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise BackendError(
                "onnxruntime/numpy are not installed in the voice environment"
            ) from exc
        self._sessions: dict[str, Any] = {}
        self._configs: dict[str, dict] = {}

    # -- storage -----------------------------------------------------------

    @staticmethod
    def _root() -> Path:
        # Weights never live in the repo. This mirrors user:// on the Godot
        # side and is created on demand.
        root = Path.home() / ".local" / "share" / "ghost" / "voices" / "piper"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _files(self, voice: str) -> tuple[Path, Path]:
        root = self._root()
        return root / f"{voice}.onnx", root / f"{voice}.onnx.json"

    def voices(self) -> list[dict]:
        out = []
        for vid, meta in VOICES.items():
            onnx, cfg = self._files(vid)
            out.append({
                "id": vid,
                "name": vid.replace("en_US-", "").replace("-", " "),
                "backend": self.name,
                "license": meta["license"],
                "derivation": meta["chain"],
                "speakers": meta["speakers"],
                "installed": onnx.exists() and cfg.exists(),
                "notes": meta.get("notes", ""),
            })
        return out

    def ensure(self, voice: str) -> dict:
        if voice not in VOICES:
            raise BackendError(f"unknown or non-allowlisted voice '{voice}'")
        onnx, cfg = self._files(voice)
        if onnx.exists() and cfg.exists():
            return {"voice": voice, "installed": True, "downloaded": False}
        base = f"{HF_BASE}/{VOICES[voice]['path']}"
        for url, dest in ((f"{base}.onnx", onnx), (f"{base}.onnx.json", cfg)):
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, tmp)
            except Exception as exc:                             # noqa: BLE001
                tmp.unlink(missing_ok=True)
                raise BackendError(f"could not fetch {url}: {exc}") from exc
            tmp.replace(dest)     # atomic: a reader sees whole file or none
        return {"voice": voice, "installed": True, "downloaded": True,
                "bytes": onnx.stat().st_size + cfg.stat().st_size}

    # -- model -------------------------------------------------------------

    def _load(self, voice: str):
        if voice in self._sessions:
            return self._sessions[voice], self._configs[voice]
        self.ensure(voice)
        import onnxruntime as ort
        onnx, cfgp = self._files(voice)
        self._ensure_aligned(onnx)
        cfg = json.loads(cfgp.read_text())
        opts = ort.SessionOptions()
        opts.log_severity_level = 3               # the protocol owns stdout
        sess = ort.InferenceSession(str(onnx), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        self._sessions[voice] = sess
        self._configs[voice] = cfg
        return sess, cfg

    @staticmethod
    def _ensure_aligned(onnx: Path) -> None:
        """Expose the duration predictor, once, on first load of a voice.

        Stock Piper graphs return audio only. Without the duration tensor there
        are no phoneme times, no word times, and therefore NO SUBTITLES - which
        is how a voice that had been hand-patched during development worked
        while every other voice silently had no overlay at all. Patching belongs
        here, where a voice is first used, not in a script someone has to
        remember to run.

        Idempotent and cheap: the check is one session open, and a patched graph
        is left alone.
        """
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        probe = ort.InferenceSession(str(onnx), sess_options=opts,
                                     providers=["CPUExecutionProvider"])
        if len(probe.get_outputs()) > 1:
            return
        del probe
        try:
            import onnx as onnx_mod
        except ImportError:
            # no `onnx` package: synthesis still works, subtitles do not
            print(f"ghost/voice: {onnx.name} has no alignment output and the "
                  "`onnx` package is missing - subtitles unavailable",
                  file=sys.stderr)
            return
        model = onnx_mod.load(str(onnx))
        ceil_nodes = [n for n in model.graph.node if n.op_type == "Ceil"]
        if len(ceil_nodes) != 1:
            print(f"ghost/voice: {onnx.name} has {len(ceil_nodes)} Ceil nodes, "
                  "expected 1 - not patching", file=sys.stderr)
            return
        tensor = ceil_nodes[0].output[0]
        model.graph.output.append(onnx_mod.helper.make_tensor_value_info(
            tensor, onnx_mod.TensorProto.FLOAT, None))
        tmp = onnx.with_suffix(onnx.suffix + ".part")
        onnx_mod.save(model, str(tmp))
        tmp.replace(onnx)      # atomic: a reader sees whole graph or old graph
        print(f"ghost/voice: patched {onnx.name} for alignment", file=sys.stderr)

    # -- synthesis ---------------------------------------------------------

    # -- phonemization --------------------------------------------------

    _espeak_ready = False
    _warned_symbols: set = set()

    @classmethod
    def _espeak(cls, words: list[str], voice: str = "en-us") -> list[str]:
        """eSpeak's IPA for a list of words, in the voice's OWN eSpeak language.

        Word by word rather than whole-sentence, because ghost needs to know
        which phones belong to which word to place the karaoke subtitles, and
        eSpeak gives no word boundaries in a sentence-level transcription.

        THE LANGUAGE MUST COME FROM THE VOICE CONFIG, not from the voice's name.
        en_US-ljspeech-medium declares `espeak.voice = "en"`, which eSpeak
        resolves to BRITISH English - so the model was trained on RP symbols
        despite the American name and audio. Hardcoding "en-us" fed it æ, ʌ, ɑː
        and oʊ where it expected a, ɒ, ɒ and əʊ, and it rendered them as the
        nearest thing it knew: "lamp" came out "lump", "not" came out "nart",
        "was" came out "wars", "balanced" came out "bullenced". Every one of
        those was this single line.
        """
        if not cls._espeak_ready:
            try:
                import espeakng_loader
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
                EspeakWrapper.set_library(espeakng_loader.get_library_path())
                EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            except ImportError as exc:
                raise BackendError(
                    "eSpeak phonemizer unavailable; set phonemizer=\"ghost\" to "
                    "use ghost's own ARPAbet front end instead"
                ) from exc
            cls._espeak_ready = True
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        # phonemizer validates against its own language list, which does not
        # include eSpeak's bare "en" - and "en" IS British in eSpeak, which is
        # the whole point of reading it from the config.
        lang = {"en": "en-gb", "en-uk": "en-gb"}.get(voice, voice)
        # SENTENCE CONTEXT, word boundaries preserved by the separator.
        # Phonemizing word by word loses homograph disambiguation: eSpeak reads
        # "live" as laɪv in isolation and lɪv in "where they live", and ghost was
        # getting the isolated reading for every occurrence. Passing whole
        # sentences and splitting on the word separator gets both.
        out = phonemize(words, language=lang, backend="espeak", strip=True,
                        with_stress=True, njobs=1,
                        separator=Separator(word=" ", phone=""))
        return [o.strip() for o in out]

    def _symbols(self, tokens: list, phonemizer: str, espeak_voice: str = "en-us") -> list:
        """(codepoint, token index) pairs for a chunk.

        A token carries its source text AND, optionally, ghost's own ARPAbet.
        Inline [K AE T] overrides arrive with `arpa` set and ALWAYS win - that
        is the whole point of the override, and it is how invented proper nouns
        stay pronounceable whichever phonemizer is in use.
        """
        import arpabet
        need = [i for i, t in enumerate(tokens)
                if not t.get("arpa") and t.get("text")]
        espoke: dict[int, str] = {}
        if need and phonemizer == "espeak":
            got = self._espeak([str(tokens[i]["text"]) for i in need], espeak_voice)
            espoke = dict(zip(need, got))

        out: list = []
        for i, t in enumerate(tokens):
            if t.get("arpa"):
                for ch, _ in arpabet.to_symbols([str(x) for x in t["arpa"]]):
                    out.append((ch, i))
            elif i in espoke:
                for ch in espoke[i]:
                    out.append((ch, i))
            elif t.get("text"):
                # no eSpeak: fall back to ghost's ARPAbet if it sent any
                for ch, _ in arpabet.to_symbols([str(x) for x in t.get("fallback", [])]):
                    out.append((ch, i))
            for ch in str(t.get("punct", "")):
                out.append((ch, i))
            out.append((" ", i))
        return out

    def synthesize(self, text: str, voice: str, out_path: str,
                   params: dict[str, Any], phonemes: Any = None) -> dict:
        """Synthesize SENTENCE BY SENTENCE, joined with real silence.

        A whole paragraph handed over as one utterance comes back rushed: the
        model has no reason to breathe at a full stop it is in the middle of,
        and the reported symptom was exactly that - "the sentences feel rushed,
        like the model should have hesitated before starting a new one".
        Splitting also keeps each inference short, which is what makes a
        streaming path possible later.
        """
        import numpy as np
        sess, cfg = self._load(voice)
        pmap: dict = cfg["phoneme_id_map"]
        sample_rate = int(cfg["audio"]["sample_rate"])

        tokens = params.get("tokens")
        if tokens:
            return self._synth_tokens(list(tokens), voice, out_path, params, cfg, sess)
        sentences = _split_sentences(list(phonemes))
        if len(sentences) > 1:
            gap = float(params.get("sentence_gap", 0.32))
            chunks, phones_all, cursor = [], [], 0.0
            for si, sent in enumerate(sentences):
                sub = self.synthesize(text, voice, out_path + f".s{si}",
                                      {**params, "_no_split": True}, sent)
                import wave as _w
                with _w.open(sub["wav"]) as fh:
                    import numpy as _np
                    a = _np.frombuffer(fh.readframes(fh.getnframes()), "<i2").astype(_np.float32) / 32768.0
                Path(sub["wav"]).unlink(missing_ok=True)
                for ph in sub.get("phones", []):
                    phones_all.append({"p": ph["p"], "t0": round(ph["t0"] + cursor, 4),
                                       "t1": round(ph["t1"] + cursor, 4)})
                chunks.append(a)
                cursor += len(a) / sub["sample_rate"]
                if si < len(sentences) - 1:
                    import numpy as _np
                    sil = _np.zeros(int(gap * sub["sample_rate"]), dtype=_np.float32)
                    chunks.append(sil)
                    cursor += gap
            import numpy as _np
            joined = _np.concatenate(chunks) if chunks else _np.zeros(1, dtype=_np.float32)
            sr = int(cfg["audio"]["sample_rate"])
            self._write_wav(out_path, joined, sr)
            return {"wav": out_path, "sample_rate": sr,
                    "duration": float(joined.size) / sr,
                    "phones": phones_all, "aligned": bool(phones_all),
                    "sentences": len(sentences)}

        if not phonemes:
            raise BackendError(
                "this backend takes phonemes, not text: ghost's own G2P supplies "
                "them, which is what keeps eSpeak-NG (GPL) out of the build"
            )

        # [BOS, PAD, id, PAD, ..., EOS]. Track which ids came from which input
        # phoneme so the durations can be folded back into words later.
        # ghost sends its own ARPAbet. Translating here rather than in Godot
        # keeps the [K AE T] override working on this path and keeps eSpeak-NG
        # out of the build entirely - see arpabet.py.
        import arpabet
        symbols = arpabet.to_symbols(list(phonemes))

        # [BOS, PAD, id, PAD, id, PAD, ..., EOS] - the pad comes BEFORE each id
        # and after the last one, per piper1-gpl docs/ALIGNMENTS.md. Emitting it
        # AFTER instead drops the pad that follows BOS, which shifts the whole
        # stream by one relative to what the model was trained on.
        ids: list[int] = [BOS]
        owner: list[int] = [-1]
        missing: list[str] = []
        for sym, src in symbols:
            mapped = pmap.get(sym)
            if mapped is None:
                missing.append(sym)
                continue
            ids.append(PAD)
            owner.append(src)
            for m in mapped:
                ids.append(int(m))
                owner.append(src)
        ids.append(PAD)
        owner.append(-1)
        ids.append(EOS)
        owner.append(-1)

        if missing:
            # Loud, not silent. A symbol absent from phoneme_id_map is dropped
            # by the encoder without complaint, and that exact failure mode
            # deleted every numeral from ghost's own front end for months.
            raise BackendError(
                "phonemes not in this voice's phoneme_id_map: "
                + " ".join(sorted(set(missing))[:12])
                + "  (regenerate the ARPAbet mapping against this voice)"
            )

        inf = cfg.get("inference", {})
        feeds = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([
                float(params.get("noise_scale", inf.get("noise_scale", 0.667))),
                float(params.get("length_scale", inf.get("length_scale", 1.0))),
                float(params.get("noise_w", inf.get("noise_w", 0.333))),
            ], dtype=np.float32),
        }
        if int(cfg.get("num_speakers", 1)) > 1:
            feeds["sid"] = np.array([int(params.get("speaker", 0))], dtype=np.int64)

        outputs = sess.run(None, feeds)
        audio = np.asarray(outputs[0]).squeeze().astype(np.float32)

        phone_times = []
        if len(outputs) > 1:
            phone_times = self._durations(outputs[1], owner, phonemes, sample_rate)

        self._write_wav(out_path, audio, sample_rate)
        return {
            "wav": out_path,
            "sample_rate": sample_rate,
            "duration": float(audio.size) / sample_rate,
            "phones": phone_times,
            "aligned": bool(phone_times),
        }

    def _synth_tokens(self, tokens: list, voice: str, out_path: str,
                      params: dict, cfg: dict, sess) -> dict:
        """Token path: sentence-split, phonemize, synthesize, join.

        Tokens carry their own text, so the sentence split happens HERE on the
        punctuation ghost already parsed, and the phonemizer sees whole words
        with their boundaries intact.
        """
        import numpy as np
        phonemizer = str(params.get("phonemizer", "espeak"))
        groups: list[list] = []
        cur: list = []
        for t in tokens:
            cur.append(t)
            if str(t.get("punct", "")) in SENTENCE_END:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)

        sr = int(cfg["audio"]["sample_rate"])
        gap = float(params.get("sentence_gap", 0.32))
        chunks: list = []
        times: list = []
        cursor = 0.0
        base = 0
        for gi, group in enumerate(groups):
            audio, per_token = self._run(group, voice, params, cfg, sess, phonemizer)
            for ti, span in per_token.items():
                # float(): numpy scalars are not JSON-serializable and the
                # protocol is JSON
                times.append({"index": base + ti,
                              "t0": round(float(span[0]) + cursor, 4),
                              "t1": round(float(span[1]) + cursor, 4)})
            chunks.append(audio)
            cursor += len(audio) / sr
            base += len(group)
            if gi < len(groups) - 1:
                chunks.append(np.zeros(int(gap * sr), dtype=np.float32))
                cursor += gap
        joined = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        self._write_wav(out_path, joined, sr)
        return {"wav": out_path, "sample_rate": sr,
                "duration": float(joined.size) / sr,
                "tokens": times, "aligned": bool(times),
                "sentences": len(groups)}


    def _run(self, group: list, voice: str, params: dict, cfg: dict, sess,
             phonemizer: str):
        """One sentence: symbols -> ids -> audio, plus per-token time spans."""
        import numpy as np
        pmap: dict = cfg["phoneme_id_map"]
        symbols = self._symbols(group, phonemizer,
                                str(cfg.get("espeak", {}).get("voice", "en-us")))
        ids: list[int] = [BOS]
        owner: list[int] = [-1]
        missing: list[str] = []
        for sym, src in symbols:
            mapped = pmap.get(sym)
            if mapped is None:
                missing.append(sym)
                continue
            ids.append(PAD)
            owner.append(src)
            for m in mapped:
                ids.append(int(m))
                owner.append(src)
        ids.append(PAD)
        owner.append(-1)
        ids.append(EOS)
        owner.append(-1)
        if missing:
            # DROP, do not fail. eSpeak emits marks that a given voice's map may
            # not carry - U+0329 (syllabic) is the common one, on words like
            # "prison" where it marks a syllabic consonant. The base phoneme is
            # already in the stream, so dropping the mark costs a nuance;
            # raising costs the whole request, which is how a cosmetic gap
            # became "request failed" and stopped playback dead.
            #
            # Reported once per symbol per process, to stderr, which ghost now
            # echoes to the terminal - visible without being fatal.
            for sym in sorted(set(missing)):
                if sym in self._warned_symbols:
                    continue
                self._warned_symbols.add(sym)
                print("ghost/voice: dropping U+%04X (%r), absent from this "
                      "voice's phoneme_id_map" % (ord(sym), sym), file=sys.stderr)

        inf = cfg.get("inference", {})
        feeds = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([
                float(params.get("noise_scale", inf.get("noise_scale", 0.667))),
                float(params.get("length_scale", inf.get("length_scale", 1.0))),
                float(params.get("noise_w", inf.get("noise_w", 0.333))),
            ], dtype=np.float32),
        }
        if int(cfg.get("num_speakers", 1)) > 1:
            feeds["sid"] = np.array([int(params.get("speaker", 0))], dtype=np.int64)
        out = sess.run(None, feeds)
        audio = np.asarray(out[0]).squeeze().astype(np.float32)

        spans: dict = {}
        if len(out) > 1:
            frames = np.asarray(out[1]).squeeze().astype(np.float64)
            if frames.ndim == 1 and frames.size == len(owner):
                t = 0.0
                for dur, who in zip(frames * HOP_LENGTH / float(cfg["audio"]["sample_rate"]),
                                    owner):
                    if who >= 0:
                        if who not in spans:
                            spans[who] = [t, t + dur]
                        else:
                            spans[who][1] = t + dur
                    t += dur
        return audio, spans


    @staticmethod
    def _durations(w_ceil, owner: list[int], phonemes: list, sample_rate: int) -> list[dict]:
        """Per-phoneme boundaries from the VITS duration predictor.

        `w_ceil` is the ceiling of the stochastic duration predictor, one entry
        per phoneme id including pads and BOS/EOS, in frames. Multiply by the
        hop length for samples. This is the plan the synthesizer actually used,
        so it is exact - strictly better than running a second forced-alignment
        model, and free.

        Only present when the voice's ONNX has been patched to expose it (see
        patch_alignment.py). Absent is fine; timings are then unavailable.
        """
        import numpy as np
        frames = np.asarray(w_ceil).squeeze().astype(np.float64)
        if frames.ndim != 1 or frames.size != len(owner):
            return []
        samples = frames * HOP_LENGTH
        # fold every id's duration back onto the input phoneme that produced it,
        # pads included - a pad's time belongs to the phoneme it followed
        spans: dict[int, float] = {}
        for dur, who in zip(samples, owner):
            if who >= 0:
                spans[who] = spans.get(who, 0.0) + float(dur)
        out, cursor = [], 0.0
        for i, sym in enumerate(phonemes):
            dur = spans.get(i, 0.0) / sample_rate
            out.append({"p": str(sym), "t0": round(cursor, 4),
                        "t1": round(cursor + dur, 4)})
            cursor += dur
        return out

    @staticmethod
    def _write_wav(path: str, audio, sample_rate: int) -> None:
        import wave
        import numpy as np
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        tmp = path + ".part"
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        Path(tmp).replace(path)   # atomic, same discipline as Voice.write_wav
