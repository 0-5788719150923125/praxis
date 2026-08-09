"""ARPAbet to IPA, so ghost's own G2P can drive a neural backend.

WHY THIS EXISTS
---------------
Most small neural TTS models are trained on IPA phonemes produced by eSpeak-NG,
which is GPLv3 and therefore viral for a shipped game. That is the single most
common reason projects in this class cannot be used commercially.

ghost does not have that problem, because ghost already has its own grapheme-to-
phoneme front end: `scripts/phonemes.gd` plus a CMUdict-derived lexicon, in
ARPAbet. Translating ARPAbet to the model's symbol set lets us feed the model
directly and never link, ship or invoke eSpeak at all.

It also preserves something worth keeping. ghost's `[K AE T]` inline phonetics
let a human override the pronunciation of any word - which matters for a novel
full of invented proper nouns - and that override is expressed in ARPAbet. Route
through this table and the same override works on the neural path.

MAPPING NOTES
-------------
ARPAbet vowels carry a stress digit (0/1/2) which is stripped here; stress is
handled separately, as prosody, not as phoneme identity.

The mapping targets the eSpeak-NG en-us phoneme inventory as consumed by
VITS-family models. Several correspondences are genuinely approximate and are
marked: ARPAbet was designed for American English and IPA transcription
conventions vary between corpora, so a model trained on one convention may want
a slightly different table. VERIFY against the target model's own phoneme_id_map
before trusting this in production - a symbol absent from that map is silently
dropped by most inference code, which is exactly the class of failure that
deleted numerals from ghost's own front end for months.
"""

from __future__ import annotations

# ARPAbet -> eSpeak-NG en-us IPA. VERIFIED EMPIRICALLY, not copied from a
# textbook table - see build_arpabet_map.py for why and how to regenerate.
#
# A hand-written textbook mapping measured 22.4% word error against 6.0% for
# ghost's own formant synthesizer, which is not a result a neural model can
# produce unless its input is wrong. eSpeak writes length marks and its own
# rhotics: "room" is ɹuːm, not ɹum, and "thirty" is θɜːɾi, not θɚti. Five
# symbols were wrong: AA, AO, ER, IY, UW.
ARPA_TO_IPA: dict[str, str] = {
    # monophthongs
    "AA": "ɑː",    # odd, father
    "AE": "æ",    # at, bat
    "AH": "ʌ",    # hut  (unstressed AH0 is schwa - see to_ipa)
    "AO": "ɔː",    # ought, story
    "EH": "ɛ",    # Ed, bed
    "ER": "ɜː",    # hurt  (r-coloured; some corpora use "ɜː" + "ɹ")
    "IH": "ɪ",    # it, bit
    "IY": "iː",    # eat, bee
    "UH": "ʊ",    # hood
    "UW": "uː",    # two, boot
    # diphthongs
    "AW": "aʊ",   # cow
    "AY": "aɪ",   # hide
    "EY": "eɪ",   # ate
    "OW": "oʊ",   # oat
    "OY": "ɔɪ",   # toy
    # stops
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "ɡ",
    # affricates
    "CH": "tʃ", "JH": "dʒ",
    # fricatives
    "F": "f", "V": "v", "TH": "θ", "DH": "ð",
    "S": "s", "Z": "z", "SH": "ʃ", "ZH": "ʒ", "HH": "h",
    # nasals
    "M": "m", "N": "n", "NG": "ŋ",
    # liquids and glides
    "L": "l", "R": "ɹ", "W": "w", "Y": "j",
}

# Unstressed AH is schwa in every American corpus, and conflating the two is
# audible: "the" would come out as "thuh" with a full vowel.
SCHWA = "ə"
# ...and unstressed ER is the r-coloured schwa, not the stressed "ɜː" of "hurt".
# eSpeak writes "seizure" as siːʒɚ. Same stress-conditioned split as AH/schwa.
SCHWA_R = "ɚ"
# ARPAbet folds the r-coloured vowel and a following /r/ into one symbol; eSpeak
# does not. "surrendered" is CMUdict S ER0 EH1 N D ER0 D but eSpeak sɚɹˈɛndɚd -
# the ɹ is spelled out when another vowel follows. Without it the word came back
# as "suh-endered", which is the mispronunciation that was reported.
RHOTIC_LINK = "ɹ"
_VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}

# Punctuation ghost's front end preserves, and which the models use as prosodic
# boundaries rather than as sounds.
# The space is a real phoneme id in these models (" " -> [3] in every voice
# config checked) and it is what marks a word boundary. Leaving it out of this
# set made to_symbols drop it silently, so the model received one continuous
# run-on utterance - measured as several points of word error on its own.
PUNCT_PASSTHROUGH = set(".,!?;: ")


def strip_stress(phone: str) -> tuple[str, int]:
    """`AE1` -> `("AE", 1)`. Consonants have no digit and return -1."""
    if phone and phone[-1].isdigit():
        return phone[:-1], int(phone[-1])
    return phone, -1


def to_symbols(phones: list[str], stress: bool = True) -> list[tuple[str, int]]:
    """Translate ARPAbet into (codepoint, source-index) pairs.

    Two things make this the real entry point rather than `to_ipa`:

    CODEPOINTS, NOT SYMBOLS. A Piper voice's `phoneme_id_map` is keyed by single
    Unicode codepoints - verified against en_US-ljspeech-medium, where "tʃ",
    "dʒ", "oʊ" and "aɪ" are all absent as units while their components are all
    present. An affricate or diphthong must therefore be emitted as its parts.

    PROVENANCE. Each emitted codepoint carries the index of the ARPAbet phone it
    came from, so the model's per-phoneme durations can be folded back onto the
    phones ghost's own G2P produced, and from there onto words for the karaoke
    subtitles.

    STRESS. ARPAbet marks stress on the vowel with a digit; eSpeak's IPA marks
    it with a modifier BEFORE the syllable. Emitting the modifier immediately
    before the vowel is an approximation - properly it belongs at the syllable
    onset - but it is the difference between a reading with accent and a flat
    one, and the alternative needs syllabification the front end does not carry.
    """
    out: list[tuple[str, int]] = []
    for i, raw in enumerate(phones):
        p = raw if raw == " " else raw.strip()
        if not p:
            continue
        if p in PUNCT_PASSTHROUGH:
            out.append((p, i))
            continue
        base, lex = strip_stress(p.upper())
        if stress and lex == 1:
            out.append(("ˈ", i))
        elif stress and lex == 2:
            out.append(("ˌ", i))
        # eSpeak drops the length mark on unstressed IY: "slowly" is slˈoʊli,
        # not slˈoʊliː. Same shape as the AH/schwa and ER/ɚ splits.
        if base == "IY" and lex == 0:
            ipa = "i"
        elif base == "AH" and lex == 0:
            ipa = SCHWA
        elif base == "ER" and lex == 0:
            ipa = SCHWA_R
        else:
            ipa = ARPA_TO_IPA.get(base)
        if ipa is None:
            continue
        for ch in ipa:
            out.append((ch, i))
        # spell the linking /r/ that ARPAbet hides inside ER when a vowel follows
        if base == "ER":
            nxt, _ = strip_stress(str(phones[i + 1]).strip().upper()) if i + 1 < len(phones) else ("", -1)
            if nxt in _VOWELS:
                out.append((RHOTIC_LINK, i))
    return out


def to_ipa(phones: list[str], keep_punct: bool = True) -> str:
    """Translate a phone sequence into an IPA string.

    Unknown symbols are DROPPED, but the caller can see that happened by
    comparing lengths via `unmapped()`. Silent dropping is how ghost's own front
    end lost every numeral for months, so anything using this should check.
    """
    out: list[str] = []
    for raw in phones:
        p = raw.strip()
        if not p:
            continue
        if p in PUNCT_PASSTHROUGH:
            if keep_punct:
                out.append(p)
            continue
        base, stress = strip_stress(p.upper())
        if base == "AH" and stress == 0:
            out.append(SCHWA)
            continue
        if base in ARPA_TO_IPA:
            out.append(ARPA_TO_IPA[base])
    return "".join(out)


def unmapped(phones: list[str]) -> list[str]:
    """Which symbols this table cannot translate. Call it in tests and on first
    use of a new model; do not let it fail quietly."""
    missing: list[str] = []
    for raw in phones:
        p = raw.strip()
        if not p or p in PUNCT_PASSTHROUGH:
            continue
        base, _ = strip_stress(p.upper())
        if base not in ARPA_TO_IPA:
            missing.append(p)
    return missing


def check_against(phoneme_id_map: dict) -> dict:
    """Audit this table against a model's own symbol inventory.

    A VITS voice config carries a `phoneme_id_map`. Any IPA symbol we emit that
    is absent from it will be dropped by the model's own encoder without
    complaint, so this is the check that turns a silent mispronunciation into a
    startup error. Returns the symbols we would emit that the model does not
    know, and the ARPAbet phones that produce them.
    """
    known = set(phoneme_id_map.keys())
    bad: dict[str, list[str]] = {}
    for arpa, ipa in list(ARPA_TO_IPA.items()) + [("AH0", SCHWA)]:
        missing = [ch for ch in ipa if ch not in known]
        if missing:
            bad.setdefault("".join(missing), []).append(arpa)
    return bad
