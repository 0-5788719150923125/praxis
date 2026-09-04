extends RefCounted
class_name Phonemes

## Phonemes - the phoneme inventory and the text-to-phoneme expansion.
##
## The speech front end for [Voice] (see next/voice.md at the repo root): a table
## of English phonemes with formant targets (Hz), durations, and source types, plus
## a rule-based letter-to-sound expansion (greedy digraph matching, a magic-e rule,
## and a small exceptions dictionary for the most common irregular words). This is
## deliberately 1980s technology - deterministic, inspectable, no model - and it
## will mispronounce rare words; the author co-owns pronunciation: any word can be
## written phonetically inline as `[K AE T]` (uppercase ARPABET-ish keys from the
## table below) and passes through literally.
##
## Types: `vowel` (periodic source through the formant cascade), `glide` (a weak
## vowel), `nasal` (murmur), `fric` (noise through its own resonator; voiced frics
## mix both sources), `stop` (closure then burst), `asp` (aspiration noise through
## the NEXT phone's formants), `sil` (pause). Diphthongs carry a second formant
## target `f2` and sweep from `f` to `f2` across the segment.

# Formants are neutral adult targets; [Voice] scales them per-voice (vocal tract
# length) and EMA-smooths across segments (coarticulation), so the numbers here
# are centres, not absolutes.
const TABLE := {
	# vowels: f = [F1, F2, F3], dur in ms
	"IY": {"type": "vowel", "f": [270.0, 2290.0, 3010.0], "dur": 110.0},
	"IH": {"type": "vowel", "f": [390.0, 1990.0, 2550.0], "dur": 85.0},
	"EH": {"type": "vowel", "f": [530.0, 1840.0, 2480.0], "dur": 100.0},
	"AE": {"type": "vowel", "f": [660.0, 1720.0, 2410.0], "dur": 130.0},
	"AA": {"type": "vowel", "f": [730.0, 1090.0, 2440.0], "dur": 130.0},
	"AO": {"type": "vowel", "f": [570.0, 840.0, 2410.0], "dur": 125.0},
	"UH": {"type": "vowel", "f": [440.0, 1020.0, 2240.0], "dur": 85.0},
	"UW": {"type": "vowel", "f": [300.0, 870.0, 2240.0], "dur": 115.0},
	"AH": {"type": "vowel", "f": [640.0, 1190.0, 2390.0], "dur": 85.0},
	"ER": {"type": "vowel", "f": [490.0, 1350.0, 1690.0], "dur": 110.0},
	# diphthongs: sweep f -> f2
	"AY": {"type": "vowel", "f": [730.0, 1090.0, 2440.0], "f2": [300.0, 2200.0, 2900.0], "dur": 170.0},
	"EY": {"type": "vowel", "f": [530.0, 1840.0, 2480.0], "f2": [300.0, 2200.0, 2900.0], "dur": 150.0},
	"OY": {"type": "vowel", "f": [570.0, 840.0, 2410.0], "f2": [330.0, 2100.0, 2800.0], "dur": 180.0},
	"AW": {"type": "vowel", "f": [730.0, 1090.0, 2440.0], "f2": [430.0, 1000.0, 2240.0], "dur": 175.0},
	"OW": {"type": "vowel", "f": [570.0, 900.0, 2410.0], "f2": [330.0, 880.0, 2240.0], "dur": 145.0},
	# glides / liquids
	"W": {"type": "glide", "f": [300.0, 610.0, 2200.0], "dur": 65.0},
	"Y": {"type": "glide", "f": [280.0, 2250.0, 2950.0], "dur": 60.0},
	"R": {"type": "glide", "f": [310.0, 1060.0, 1380.0], "dur": 75.0},
	"L": {"type": "glide", "f": [360.0, 1300.0, 2700.0], "dur": 70.0},
	# nasals: a murmur through low poles MINUS a side-cavity zero. The zero is
	# per-place - it is the whole difference between sum, sun and sung, and one
	# shared 1000 Hz notch collapsed all three onto each other.
	"M": {"type": "nasal", "f": [280.0, 900.0, 2200.0], "zero": 950.0, "dur": 70.0},
	"N": {"type": "nasal", "f": [280.0, 1700.0, 2600.0], "zero": 1800.0, "dur": 65.0},
	"NG": {"type": "nasal", "f": [280.0, 2300.0, 2750.0], "zero": 3000.0, "dur": 75.0},
	# Obstruents. Two independent things, and conflating them is what made every
	# fricative sound the same:
	#   `f`   - the TRACT POSTURE (the locus). Where the articulators are, which
	#           is what bends the neighbouring vowel's formants. This is the
	#           primary place cue for /f/ and /th/, whose own noise is weak.
	#   `par` - the PARALLEL BRANCH: [centre Hz, bandwidth Hz, amplitude] triples
	#           describing the front-cavity resonances the turbulence excites.
	#           These are summed AFTER the cascade, never through it: the cascade
	#           is an all-pole lowpass whose top pole sits near 4.7 kHz, so
	#           routing frication through it removed 42 dB at /f/, 69 dB at /th/
	#           and 85 dB at /s/ and left all four sibilants correlated at 0.99.
	#           (Klatt 1980 splits cascade and parallel for exactly this reason.)
	#   `namp`- the branch's overall level; FRIC_LEVEL in [Voice] seats the set.
	# Bandwidths here are ABSOLUTE and are NOT scaled by vocal tract length; only
	# the centres are (a short tract raises the resonance, it does not sharpen it).
	"S": {"type": "fric", "voiced": false, "f": [320.0, 1750.0, 2600.0],
		"namp": 1.0,
		"pa": [0, 0, 0, 0, 0, 0, 0.7943], "ab": 0, "dur": 105.0},
	"Z": {"type": "fric", "voiced": true, "f": [320.0, 1750.0, 2600.0],
		"namp": 0.5,
		"pa": [0, 0, 0, 0, 0, 0, 0.7943], "ab": 0, "dur": 90.0},
	# /sh/ carries the highest amplitude of any English fricative - and it needs
	# it here, because its band overlaps the vowel's own F4/F5 where /s/'s does
	# not, so equal amplitude left it 20 dB short of /s/ in contrast.
	"SH": {"type": "fric", "voiced": false, "f": [320.0, 2000.0, 2500.0],
		"namp": 3.35,
		"pa": [0, 0, 0.7079, 0.2512, 0.2512, 0.1995, 0], "ab": 0, "dur": 110.0},
	"ZH": {"type": "fric", "voiced": true, "f": [320.0, 2000.0, 2500.0],
		"namp": 1.675,
		"pa": [0, 0, 0.7079, 0.2512, 0.2512, 0.1995, 0], "ab": 0, "dur": 95.0},
	# /f/ and /th/ are labiodental and dental: almost no front cavity, so their
	# spectra are FLAT and DIFFUSE rather than peaked, and quiet in absolute
	# terms. They are audible because they sit in 4-10 kHz where vowels have
	# nothing - the cue is spectral CONTRAST against the neighbour, not level.
	"F": {"type": "fric", "voiced": false, "f": [320.0, 900.0, 2200.0],
		"namp": 0.226,
		"pa": [0, 0, 0, 0, 0, 0, 0], "ab": 0.7079, "dur": 95.0},
	"V": {"type": "fric", "voiced": true, "f": [320.0, 900.0, 2200.0],
		"namp": 0.131,
		"pa": [0, 0, 0, 0, 0, 0, 0], "ab": 0.7079, "dur": 70.0},
	"TH": {"type": "fric", "voiced": false, "f": [320.0, 1400.0, 2400.0],
		"namp": 0.20,
		"pa": [0, 0, 0, 0, 0, 0.0251, 0], "ab": 0.2512, "dur": 90.0},
	"DH": {"type": "fric", "voiced": true, "f": [320.0, 1400.0, 2400.0],
		"namp": 0.106,
		"pa": [0, 0, 0, 0, 0, 0.0251, 0], "ab": 0.2512, "dur": 60.0},
	# stops: closure at the locus, then a burst through the parallel branch.
	# The three classic burst shapes: labial diffuse-falling, alveolar
	# diffuse-rising, velar compact (a single mid peak - the "velar pinch").
	"P": {"type": "stop", "voiced": false, "f": [320.0, 900.0, 2200.0],
		"namp": 0.9,
		"pa": [0, 0, 0, 0, 0, 0, 0], "ab": 1.4125, "dur": 90.0},
	"B": {"type": "stop", "voiced": true, "f": [320.0, 900.0, 2200.0],
		"namp": 0.6,
		"pa": [0, 0, 0, 0, 0, 0, 0], "ab": 1.4125, "dur": 75.0},
	"T": {"type": "stop", "voiced": false, "f": [320.0, 1750.0, 2600.0],
		"namp": 1.0,
		"pa": [0, 0, 0.0316, 0.1778, 0.7079, 1.4125, 0], "ab": 0, "dur": 90.0},
	"D": {"type": "stop", "voiced": true, "f": [320.0, 1750.0, 2600.0],
		"namp": 0.65,
		"pa": [0, 0, 0.2239, 1, 1.2589, 1, 0], "ab": 0, "dur": 75.0},
	"K": {"type": "stop", "voiced": false, "f": [320.0, 1900.0, 2250.0],
		"namp": 0.95,
		"pa": [0, 0, 0.4467, 0.1413, 0.1778, 0.1778, 0], "ab": 0, "dur": 95.0},
	"G": {"type": "stop", "voiced": true, "f": [320.0, 1900.0, 2250.0],
		"namp": 0.6,
		"pa": [0, 0, 0.4467, 0.1413, 0.1778, 0.1778, 0], "ab": 0, "dur": 80.0},
	# affricates are expanded at parse time: CH -> T SH, JH -> D ZH
	# /h/ is glottal: its noise DOES belong in the cascade (the whole tract is
	# the filter), which is why it has no `par` and no posture of its own.
	"HH": {"type": "asp", "dur": 65.0},
	"SIL": {"type": "sil", "dur": 1.0},
}

# Greedy longest-match spelling rules, tried before single letters. Order within
# a length class does not matter; longer keys always win.
const DIGRAPHS := {
	"tch": ["T", "SH"], "igh": ["AY"], "eigh": ["EY"], "ough": ["OW"],
	"ch": ["T", "SH"], "sh": ["SH"], "th": ["TH"], "ph": ["F"], "wh": ["W"],
	"ck": ["K"], "ng": ["NG"], "qu": ["K", "W"], "gh": [],
	"ee": ["IY"], "ea": ["IY"], "oo": ["UW"], "ou": ["AW"], "ow": ["OW"],
	"ai": ["EY"], "ay": ["EY"], "oa": ["OW"], "oi": ["OY"], "oy": ["OY"],
	"ew": ["UW"], "ue": ["UW"], "au": ["AO"], "aw": ["AO"],
	"ar": ["AA", "R"], "or": ["AO", "R"], "er": ["ER"], "ir": ["ER"], "ur": ["ER"],
	"ll": ["L"], "ss": ["S"], "tt": ["T"], "pp": ["P"], "bb": ["B"], "dd": ["D"],
	"mm": ["M"], "nn": ["N"], "rr": ["R"], "gg": ["G"], "ff": ["F"], "zz": ["Z"], "cc": ["K"],
}

const SINGLES := {
	"a": ["AE"], "e": ["EH"], "i": ["IH"], "o": ["AA"], "u": ["AH"],
	"b": ["B"], "c": ["K"], "d": ["D"], "f": ["F"], "g": ["G"], "h": ["HH"],
	"j": ["D", "ZH"], "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"], "p": ["P"],
	"q": ["K"], "r": ["R"], "s": ["S"], "t": ["T"], "v": ["V"], "w": ["W"],
	"x": ["K", "S"], "y": ["IH"], "z": ["Z"],
}

# The magic-e long vowels (`make`, `time`, `hope`): V C e$ -> long V, e dropped.
const LONG := {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}

const VOWEL_LETTERS := "aeiouy"

# `th` is VOICED (DH) in three environments and voiceless everywhere else.
# It was unconditionally voiceless, which is wrong on some of the highest
# frequency words in English - `father` and `mother` came out with the `th` of
# `thin`. These are the function words that carry a voiced initial `th`; the
# intervocalic and `-the$` cases are handled by rule in word_to_phones.
const TH_VOICED := ["the", "this", "that", "these", "those", "them", "then",
	"than", "there", "their", "they", "though", "thus", "thence", "thee",
	"thy", "thine", "themselves", "therefore"]

# `g` before e/i/y is SOFT (as in `age`, `magic`) often enough to be the
# default, but hard in a set of very common Germanic words. Listing the hard
# ones is the smaller and safer change: flipping the default would break the
# Latinate majority (`large`, `change`, `region`, `energy`, `message`).
# Suffixed forms reduce to these stems through the morphology pass below.
const HARD_G := ["get", "give", "given", "giving", "getting", "girl", "gift",
	"begin", "began", "begun", "beginning", "forget", "forgive", "together",
	"target", "tiger", "anger", "angry", "eager", "finger", "hunger", "gear",
	"geese", "gate", "gone", "girth", "gild"]

# Phones after which a plural / third-person `-s` stays voiceless.
const VOICELESS_PHONES := ["P", "T", "K", "F", "TH", "S", "SH", "HH"]
# ... and after which it becomes a whole syllable (`buses`, `wishes`).
const SIBILANT_PHONES := ["S", "Z", "SH", "ZH"]

# Common irregular words the rules would butcher. Small on purpose - the author
# writes `[.]` phonetics for anything else.
const EXCEPTIONS := {
	"the": ["DH", "AH"], "of": ["AH", "V"], "to": ["T", "UW"], "you": ["Y", "UW"],
	"was": ["W", "AH", "Z"], "is": ["IH", "Z"], "as": ["AE", "Z"], "his": ["HH", "IH", "Z"],
	"are": ["AA", "R"], "were": ["W", "ER"], "one": ["W", "AH", "N"], "once": ["W", "AH", "N", "S"],
	"two": ["T", "UW"], "do": ["D", "UW"], "does": ["D", "AH", "Z"], "done": ["D", "AH", "N"],
	"who": ["HH", "UW"], "what": ["W", "AH", "T"], "where": ["W", "EH", "R"],
	"there": ["DH", "EH", "R"], "their": ["DH", "EH", "R"], "they": ["DH", "EY"],
	"said": ["S", "EH", "D"], "says": ["S", "EH", "Z"], "have": ["HH", "AE", "V"],
	"has": ["HH", "AE", "Z"], "give": ["G", "IH", "V"], "live": ["L", "IH", "V"],
	"love": ["L", "AH", "V"], "some": ["S", "AH", "M"], "come": ["K", "AH", "M"],
	"gone": ["G", "AO", "N"], "been": ["B", "IH", "N"], "your": ["Y", "AO", "R"],
	"our": ["AW", "R"], "my": ["M", "AY"], "i": ["AY"], "eye": ["AY"], "by": ["B", "AY"],
	"why": ["W", "AY"], "would": ["W", "UH", "D"], "could": ["K", "UH", "D"],
	"should": ["SH", "UH", "D"], "through": ["TH", "R", "UW"], "though": ["DH", "OW"],
	"thought": ["TH", "AO", "T"], "into": ["IH", "N", "T", "UW"], "over": ["OW", "V", "ER"],
	"only": ["OW", "N", "L", "IY"], "very": ["V", "EH", "R", "IY"],
	"any": ["EH", "N", "IY"], "many": ["M", "EH", "N", "IY"], "again": ["AH", "G", "EH", "N"],
	"water": ["W", "AO", "T", "ER"], "world": ["W", "ER", "L", "D"],
	"move": ["M", "UW", "V"], "prove": ["P", "R", "UW", "V"], "own": ["OW", "N"],
	"body": ["B", "AA", "D", "IY"], "eyes": ["AY", "Z"], "says'": ["S", "EH", "Z"],
	"a": ["AH"], "or": ["AO", "R"], "for": ["F", "AO", "R"], "from": ["F", "R", "AH", "M"],
}

# Function words never take an accent (the stress heuristic in parse()).
const FUNCTION_WORDS := [
	"the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "and", "or",
	"but", "is", "are", "was", "were", "be", "been", "it", "its", "as", "that",
	"this", "with", "from", "into", "than", "then", "so", "if", "not", "no",
]


## Expand a paragraph into sentences of timed-ready words. Returns an Array of
## sentences; each sentence is an Array of word Dictionaries:
## `{text, phones: [String], stressed: bool, pause_after: "none"|"comma"|"stop"}`.
## `[K AE T]` bracket groups pass through as literal phonemes (shown as the
## bracketed text in subtitles).
static func parse(text: String) -> Array:
	var sentences: Array = []
	var words: Array = []
	# Normalize FIRST: fold typographic punctuation to ASCII and turn numerals,
	# ordinals, currency, times and dotted abbreviations into words. Before this
	# stage existed, a numeral produced an empty phone array and vanished from
	# the utterance without a trace, and a curly apostrophe defeated both the
	# contraction split and the dictionary key. See TextNorm.
	# ...and keep WHAT THE PAGE SAID alongside what the mouth will say. `2009` is
	# spoken "two thousand nine" and must still be SHOWN as `2009`; see
	# TextNorm.normalize_marked for why the two spellings have to travel together.
	var marked := TextNorm.normalize_marked(text)
	text = String(marked["text"])
	var spans: Array = marked["spans"]
	var span_at := 0                 # spans are ascending; walk them with the tokens
	var span_open := -1              # the span the previous word belonged to, if any
	var tok_at := PackedInt32Array()
	var toks := _tokenize(text, tok_at)
	# The emphasis level currently in force. The sentinels TextNorm leaves behind are
	# toggles welded to the word at each end of a run, so a run of any length needs only
	# its two ends marked and this carries the level across everything between.
	var emph_state := 0
	for ti in toks.size():
		var token: String = toks[ti]
		# PEEL THE SENTINELS FIRST, before anything reads the token. They are typography:
		# the dictionary must not see them, and `display` must not print them - what they
		# leave behind is `emph`, which the subtitle draws as a slanted or bold face.
		var emph := emph_state
		if token.contains(TextNorm.EMPH_ITALIC) or token.contains(TextNorm.EMPH_BOLD):
			var clean := ""
			var seen := 0
			for ci in token.length():
				var ech := token[ci]
				if ech == TextNorm.EMPH_ITALIC:
					emph_state ^= TextNorm.EMPH_I
				elif ech == TextNorm.EMPH_BOLD:
					emph_state ^= TextNorm.EMPH_B
				else:
					seen |= emph_state
					clean += ech
			token = clean
			emph = seen
		# WHICH SOURCE RUN IS THIS TOKEN PART OF. A rewritten run covers one or
		# more tokens ("two thousand nine"), and the reader is shown its source
		# spelling ONCE, over the whole run - so the first token of a run carries
		# the source and the rest carry nothing to draw. `src_span` groups them
		# so a subtitle can span the run's whole duration rather than flashing
		# the numeral over its first syllable.
		var start: int = tok_at[ti] if ti < tok_at.size() else -1
		var in_span := -1
		var span_src := ""
		while span_at < spans.size():
			var sp: Dictionary = spans[span_at]
			if start >= 0 and start >= int(sp["at"]) + int(sp["len"]):
				span_at += 1        # the run ended before this token began
				continue
			if start >= 0 and start >= int(sp["at"]):
				in_span = span_at
				span_src = String(sp["src"])
			break
		if token.begins_with("["):
			var lit := _literal_word(token)
			lit["emph"] = emph
			words.append(lit)
			continue
		var pause := "none"
		var punct := ""
		var bare := token
		# ORDER MATTERS (2026-08-09). The wrapper strip used to run AFTER this
		# loop, so a token like `early,"` failed the loop's very first test on
		# the closing quote and exited immediately: pause stayed "none", punct
		# stayed empty, and every line of dialogue lost its terminal contour and
		# its pause. Strip wrappers, then terminal punctuation, then any wrapper
		# the punctuation was hiding.
		# ... and it has to REPEAT, which the two-pass version above did not. A token can
		# interleave wrappers and marks - `know."` followed by a line break arrives as
		# `know."\n` - and one pass of each peels the newline, stops at the quote, then
		# peels the quote and leaves the FULL STOP welded to the word. Measured on the
		# manuscript: `you.`, `week.`, `dog.`, `free.`, `window.` and a dozen more reached
		# the dictionary with a period attached, which means they missed it - and any such
		# word would also have missed the homograph table, so `record."` at the end of a
		# line of dialogue would have been read as the noun whatever the sentence said.
		# Peel to a fixpoint instead.
		while true:
			var before := bare
			bare = bare.lstrip("\"'(").rstrip("\"')")
			while bare.length() > 0 and bare[bare.length() - 1] in ".,!?;:\n":
				var c := bare[bare.length() - 1]
				pause = "stop" if c in ".!?\n" else "comma"
				if punct.is_empty() and c != "\n":
					punct = c            # the terminal mark drives the contour (?, !, .)
				bare = bare.substr(0, bare.length() - 1)
			bare = bare.strip_edges()
			if bare == before:
				break
		bare = bare.to_lower()
		# %HESITATION (the ASR transcript token for a filled pause): an authored
		# "um" - low, flat, reduced. Shows as an ellipsis in the karaoke.
		if bare.begins_with("%"):
			words.append({"text": "…", "display": "…", "phones": ["AH", "M"],
				"stressed": false, "pause_after": pause, "punct": punct, "hesit": true,
				"emph": emph})
			if pause == "stop" and words.size() > 0 and _ends_sentence(toks, ti):
				sentences.append(words)
				words = []
			continue
		# A DASH standing alone is a rest, not a word. ghost's own prose style writes an
		# aside as " - " rather than with an em dash, so this is the common case in the
		# book text this engine exists to read - and the mark has no phoneme of its own,
		# so it is voiced as the rest a reader would take there.
		if bare.length() > 0 and _is_dash(bare):
			pause = "comma"
			punct = ","          # a hyphen means nothing to the model; a comma is the rest
			bare = ""
		# A MARK WITH NO WORD OF ITS OWN used to vanish outright: nothing was appended, so
		# its pause and its mark went with it. A spaced dash produced no rest at all, and
		# neither did a comma left adrift by a stray space ("one , two"). Hand it to the
		# word BEFORE, which is where the reader actually breathes.
		if bare.is_empty() and pause != "none" and words.size() > 0:
			var prev: Dictionary = words[words.size() - 1]
			# Never downgrade: a stop already claimed is stronger than this comma.
			if String(prev.get("pause_after", "none")) != "stop":
				prev["pause_after"] = pause
				if String(prev.get("punct", "")).is_empty():
					prev["punct"] = punct
			# AND KEEP IT VISIBLE. `display` is what the reader sees in the subtitles, and a
			# mark that lost its own word would vanish from the page as well as the mouth -
			# "a black breath - the glass going dark" was being SHOWN without its dash.
			# Normalization is for the phoneme lookup, never for the reader's eyes.
			var mark := token.strip_edges()
			if not mark.is_empty() and mark != "\n":
				prev["display"] = String(prev.get("display", "")) + " " + mark
		if bare.length() > 0:
			var got := lookup(bare)
			var phones: Array = got.phones
			if phones.is_empty():
				# NOTHING MAY LEAVE THE PIPELINE SILENTLY. This `if phones.size() > 0` was the
				# second half of a reported loss: `2009` at the end of a paragraph escaped
				# TextNorm's numeral pass (see TextNorm._expand_numbers, now fixed), reached here
				# as digits, and word_to_phones - which walks letters - returned nothing, so the
				# word was deleted from the utterance with no trace in the audio, the subtitles or
				# the log. Two things now stand between a token and that fate: a rescue that reads
				# any digits it contains, and a warning for whatever is left.
				var saved := _rescue_phones(bare)
				phones = saved["phones"] as Array
				if phones.is_empty():
					push_warning("ghost/voice: no pronunciation for %s - dropping it. "
						% [token] + "If it should be spoken, add it under `names:` in "
						+ "data/english.yml or inline as [P AH0 N S]")
				else:
					push_warning("ghost/voice: %s reached the phonemizer unnormalized; "
						% [token] + "read as '%s'" % [String(saved["said"])])
					got = {"phones": phones, "stress": saved["stress"]}
			if phones.size() > 0:
				# THE SOURCE SPELLING, caps and punctuation intact - normalization
				# is for the phoneme lookup, never for the reader's eyes. Inside a
				# rewritten run that is what the page said (`2009`), shown once on
				# the run's first word; elsewhere the token already IS the source.
				var shown := token.trim_suffix("\n")
				if in_span >= 0:
					shown = span_src.trim_suffix("\n") if in_span != span_open else ""
					span_open = in_span
				words.append({
					"text": bare,
					"display": shown,
					"src_span": in_span,
					"phones": phones,
					"stress": got.stress,
					"stressed": not is_function_word(bare),
					"pause_after": pause,
					"punct": punct,
					"emph": emph,
				})
		if pause == "stop" and words.size() > 0 and _ends_sentence(toks, ti):
			sentences.append(words)
			words = []
	if words.size() > 0:
		sentences.append(words)
	_resolve_homographs(sentences)
	return sentences


## `at` collects each token's start offset in `text`, for callers that have to
## map a token back onto the source it was normalized from (see [method parse]).
## Left empty by every other caller; the tokenization is unchanged.
static func _tokenize(text: String, at: PackedInt32Array = PackedInt32Array()) -> PackedStringArray:
	var out := PackedStringArray()
	var i := 0
	while i < text.length():
		var c := text[i]
		if c == "[":
			var close := text.find("]", i)
			if close < 0:
				close = text.length() - 1
			at.append(i)
			out.append(text.substr(i, close - i + 1))
			i = close + 1
		elif c == " " or c == "\t":
			i += 1
		elif c == "\n":
			# a bare newline acts as a sentence break on the previous word
			if out.size() > 0 and not out[out.size() - 1].ends_with("\n"):
				out[out.size() - 1] = out[out.size() - 1] + "\n"
			i += 1
		else:
			var j := i
			while j < text.length() and not (text[j] in " \t\n["):
				j += 1
			at.append(i)
			out.append(text.substr(i, j - i))
			i = j
	return out


static func _literal_word(token: String) -> Dictionary:
	var inner := token.trim_prefix("[").trim_suffix("]").strip_edges()
	var phones: Array = []
	for p in inner.split(" ", false):
		var key := String(p).to_upper()
		if TABLE.has(key):
			phones.append(key)
		elif key == "CH":
			phones.append_array(["T", "SH"])
		elif key == "JH":
			phones.append_array(["D", "ZH"])
	var st := _with_default_stress(phones)
	# `literal` is what makes ghost's own ARPAbet BEAT eSpeak downstream (see
	# generative_editor._build_chunks). Without it an inline [K AE T] was quietly
	# re-phonemized by eSpeak like any other word - the documented escape hatch for a
	# mispronounced name did nothing at all on the generative path.
	return {"text": inner.to_lower(), "display": inner, "phones": phones,
		"stress": st.stress, "stressed": true, "pause_after": "none", "literal": true}


# ---- the external language data ---------------------------------------------
#
# `data/english.yml` holds the pronunciation LEXICON and the SUFFIX table. It is
# data, not configuration: the language itself, external so it can be read and
# corrected without touching the synthesizer, and so a wrong word is a one-line
# diff instead of a code change. The built-in EXCEPTIONS above remain the
# fallback, so a missing or malformed file degrades to the old behaviour rather
# than breaking speech.
const DATA_PATH := "res://data/english.yml"
# CMUdict: 126k words with LEXICAL STRESS, vendored (BSD-2-Clause, see
# data/cmudict.LICENSE). Hand-building a lexicon was the wrong instinct - this
# is a dictionary, not a model, exactly the kind of 1980s-technology data table
# the phoneme inventory already is, and it carries the one thing a hand list
# could never supply cheaply: which syllable of every word is stressed. That is
# what drives vowel reduction and duration, so the dictionary is not a shortcut
# past the interesting work, it is the input the interesting work needed.
# Measured: 3.3 MB, 117 ms to read and index once, at first speech.
const CMUDICT_PATH := "res://data/cmudict.dict"

static var _lexicon := {}                 # our overrides (data/english.yml)
static var _names := {}                   # invented words / proper nouns (data/english.yml `names:`)
static var _cmu := {}                     # word -> raw "F AA1 DH ER0"
static var _clitics := {}
static var _function_words := {}
static var _suffixes: Array = []
static var _loaded := false


static func _load_data() -> void:
	if _loaded:
		return
	_loaded = true
	if FileAccess.file_exists(CMUDICT_PATH):
		var cf := FileAccess.open(CMUDICT_PATH, FileAccess.READ)
		var txt := cf.get_as_text()
		cf.close()
		for line in txt.split("\n", false):
			var sp := line.find(" ")
			if sp > 0:
				_cmu[line.substr(0, sp)] = line.substr(sp + 1)
	else:
		push_warning("phonemes: %s missing - falling back to letter rules" % CMUDICT_PATH)
	if not FileAccess.file_exists(DATA_PATH):
		push_warning("phonemes: %s missing - falling back to built-in EXCEPTIONS" % DATA_PATH)
		return
	var f := FileAccess.open(DATA_PATH, FileAccess.READ)
	var res := MiniYaml.parse(f.get_as_text())
	f.close()
	if not res.ok:
		push_warning("phonemes: %s - %s" % [DATA_PATH, res.error])
		return
	var data: Dictionary = res.data if res.data is Dictionary else {}
	for w in (data.get("lexicon", {}) as Dictionary):
		# stored RAW so stress digits survive: an override without them silently
		# fell back to "primary stress on the first vowel", which mis-stressed 56
		# high-frequency words (about, because, before, between, again ...) the
		# moment CMUdict arrived. An override must be able to carry stress.
		_lexicon[String(w)] = String(data.lexicon[w])
	# NAMES - words no dictionary has, and the neural backend's own guess is wrong for.
	# Kept apart from `lexicon` because they are resolved differently: a lexicon entry
	# only reaches the PROCEDURAL voice, since the neural path sends text to eSpeak and
	# uses ghost's own phones for a word only when it is marked `literal`. An invented
	# proper noun has to beat eSpeak, so these are loaded into the same table the
	# homograph resolver writes through, which sets that flag.
	for nm in (data.get("names", {}) as Dictionary):
		_names[String(nm).to_lower()] = String(data.names[nm])
	for fw in (data.get("function_words", []) as Array):
		_function_words[String(fw)] = true
	for c in (data.get("clitics", {}) as Dictionary):
		_clitics[String(c)] = _split_phones(String(data.clitics[c]))
	for entry in (data.get("suffixes", []) as Array):
		if entry is Dictionary and entry.has("spelling"):
			_suffixes.append(entry)


## "K AE T" (or a list) -> ["K", "AE", "T"], expanding the affricates.
static func _split_phones(spec: String) -> Array:
	var out: Array = []
	for p in spec.split(" ", false):
		var key := String(p).to_upper()
		if key == "CH":
			out.append_array(["T", "SH"])
		elif key == "JH":
			out.append_array(["D", "ZH"])
		elif TABLE.has(key):
			out.append(key)
	return out


static func _phone_list(v: Variant) -> Array:
	if v is Array:
		var out: Array = []
		for p in v:
			out.append_array(_split_phones(String(p)))
		return out
	return _split_phones(String(v))


## One lowercase word -> `{phones: [String], stress: [int]}`, the arrays
## parallel. Stress is 0 (unstressed / consonant), 1 (primary) or 2 (secondary),
## and is the signal the whole rhythm of the reading hangs off: which vowels
## reduce to schwa, which get length, and which carries the pitch accent.
## A LAST RESORT for a token the dictionary and the letter rules both have nothing for, so that
## it is never simply deleted. Digits are the case that matters and the case that was reported:
## anything numeric is normalized here and read as the words it stands for, with the phones of
## those words concatenated into this one token (its DISPLAY stays the source spelling, so the
## karaoke line still shows `2009`). A token with no digits and no letters is a symbol - there is
## nothing to say for it, and the caller warns and drops it.
##
## This is deliberately defence in depth rather than the fix: numerals are expanded before the
## tokenizer ever sees them (see TextNorm), and if one reaches here at all that is a bug in the
## normalizer. It just must not be an INAUDIBLE bug.
static func _rescue_phones(bare: String) -> Dictionary:
	var digits := false
	for i in bare.length():
		if bare[i] >= "0" and bare[i] <= "9":
			digits = true
			break
	if not digits:
		return {"phones": [], "stress": [], "said": ""}
	var said := TextNorm.normalize(bare)
	var phones: Array = []
	var stress: Array = []
	for part in said.split(" ", false):
		var w := String(part).to_lower().strip_edges()
		while w.length() > 0 and w[w.length() - 1] in ".,!?;:\"')":
			w = w.substr(0, w.length() - 1)
		if w.is_empty() or w == bare:
			continue                       # w == bare: the normalizer had nothing either
		var g := lookup(w)
		phones.append_array(g["phones"] as Array)
		stress.append_array(g["stress"] as Array)
	return {"phones": phones, "stress": stress, "said": said}


static func lookup(word: String) -> Dictionary:
	_load_data()
	# our overrides win: data/english.yml is where we disagree with the
	# dictionary on purpose (reduced narrator forms, house pronunciations)
	var got: Dictionary
	if _lexicon.has(word):
		got = _parse_cmu(String(_lexicon[word]))
	elif _cmu.has(word):
		got = _parse_cmu(String(_cmu[word]))
	else:
		got = _with_default_stress(word_to_phones(word))
	# A FUNCTION WORD IS UNSTRESSED, whatever the dictionary says. CMUdict lists
	# citation forms (`has` as HH AE1 Z), which is right for a word said alone
	# and wrong for one said in a sentence - and a stress-1 mark stops the
	# reduction stage from touching it, so every `has`, `for` and `your` came
	# out at full length and full vowel quality. Stripping the stress here lets
	# the existing machinery reduce them, at whatever depth the speaker and the
	# tempo call for, instead of freezing one reduced spelling into the lexicon.
	# ... but only a MONOSYLLABIC one. English reduces `the`, `of`, `at`, `for`;
	# it does not flatten `about`, `between`, `under` or `after`, which keep
	# their internal stress pattern even when the sentence gives them no accent.
	# Flattening those made `about` come back with no stressed vowel at all.
	# Not being accented is Phrasing's job, and it already handles both cases.
	if is_function_word(word) and _syllables(got.phones as Array) <= 1:
		var flat: Array = []
		for _i in (got.stress as Array).size():
			flat.append(0)
		got.stress = flat
	return got


static func _syllables(phones: Array) -> int:
	var n := 0
	for p in phones:
		if TABLE.get(p, {}).get("type", "") == "vowel":
			n += 1
	return n


## "F AA1 DH ER0" -> phones + stress, expanding the affricates.
static func _parse_cmu(spec: String) -> Dictionary:
	var phones: Array = []
	var stress: Array = []
	# an entry with no digits at all gets the default marking instead, so a
	# hand-written override may omit them for a monosyllable
	var marked := false
	for t in spec.split(" ", false):
		var c := String(t)[String(t).length() - 1]
		if c >= "0" and c <= "9":
			marked = true
			break
	for tok in spec.split(" ", false):
		var t := String(tok)
		var st := 0
		var last := t[t.length() - 1]
		if last >= "0" and last <= "9":
			st = int(last)
			t = t.substr(0, t.length() - 1)
		if t == "CH":
			phones.append_array(["T", "SH"])
			stress.append_array([0, 0])
		elif t == "JH":
			phones.append_array(["D", "ZH"])
			stress.append_array([0, 0])
		elif TABLE.has(t):
			phones.append(t)
			stress.append(st)
	if not marked:
		return _with_default_stress(phones)
	return {"phones": phones, "stress": stress}


## Phones with no marks of their own: primary stress on the first vowel, which
## is right for the short function words data/english.yml holds and a harmless
## default elsewhere (the dictionary answers almost everything real).
static func _with_default_stress(phones: Array) -> Dictionary:
	var stress: Array = []
	var seen := false
	for p in phones:
		var is_v: bool = TABLE.get(p, {}).get("type", "") == "vowel"
		stress.append(1 if (is_v and not seen) else 0)
		if is_v:
			seen = true
	return {"phones": phones.duplicate(), "stress": stress}


## Is this word ANSWERED, or is the front end guessing at it?
##
## The distinction the audit tool turns on. Everything below is a real answer from a real
## source; anything else falls through to the letter-to-sound rules, which are
## deliberately 1980s technology and will mispronounce an invented word or an unusual
## proper noun without any indication that they have. A hyphenated compound counts as
## known when every part of it is, since the tokenizer splits it before it ever reaches
## the rules, and a plural or possessive of a known stem is handled by the morphology
## step rather than by guessing.
static func is_known(word: String) -> bool:
	_load_data()
	var w := word.to_lower().strip_edges()
	if w.is_empty():
		return true
	if _names.has(w) or _lexicon.has(w) or _cmu.has(w) or EXCEPTIONS.has(w):
		return true
	if w.contains("-"):
		for part in w.split("-", false):
			if not is_known(String(part)):
				return false
		return true
	# A PRODUCTIVE AFFIX IS NOT AN UNKNOWN WORD. `unmaking`, `watchmaker's` and
	# `reproducible` are not gaps in the dictionary, they are regular English built out
	# of parts the dictionary has - and both this front end and eSpeak's letter-to-sound
	# rules handle them correctly. Reporting them as risks buries the two or three words
	# that genuinely are risks under a hundred that are not, which is how an audit gets
	# ignored. One level of stripping at each end is enough for ordinary prose.
	for suf in ["'s", "s'", "es", "ed", "ing", "er", "est", "ly", "ness", "ment", "ful",
			"able", "ible", "ist", "ism", "s", "'"]:
		if w.length() > suf.length() + 2 and w.ends_with(suf):
			var stem := w.substr(0, w.length() - suf.length())
			if _known_stem(stem) or _known_stem(stem + "e"):
				return true
	for pre in ["un", "re", "non", "pre", "mid", "over", "under", "anti", "de", "dis",
			"mis", "sub", "inter", "counter", "self"]:
		if w.length() > pre.length() + 2 and w.begins_with(pre):
			if is_known(w.substr(pre.length())):
				return true
	return false


## A stem answered outright, without recursing back through the affix stripping (which
## would let a word validate itself through an unbounded chain of prefixes).
static func _known_stem(stem: String) -> bool:
	return _cmu.has(stem) or _lexicon.has(stem) or _names.has(stem) or EXCEPTIONS.has(stem)


## One lowercase word -> phoneme keys: lexicon, then suffixes, then letter rules.
static func word_to_phones(word: String) -> Array:
	_load_data()
	if _lexicon.has(word):
		return (_parse_cmu(String(_lexicon[word])).phones as Array)
	# A HYPHENATED COMPOUND IS ITS PARTS. TextNorm used to split these before the
	# tokenizer ever saw them, which pronounced them correctly but deleted the hyphen
	# from the text - and the karaoke line shows the source spelling, so "twenty-five"
	# was subtitled as "twenty five". The hyphen stays now, so the decomposition has to
	# happen here instead: each part goes through this same pipeline and gets its own
	# dictionary entry. Without it the letter rules read the whole thing as one run of
	# characters and lose real sounds - "off-by-one" came out with no /w/ in "one".
	if word.contains("-") and not _cmu.has(word):
		var joined: Array = []
		var any_part := false
		for part in word.split("-", false):
			var p := String(part)
			if p.is_empty():
				continue
			any_part = true
			joined.append_array(word_to_phones(p))
		if any_part and not joined.is_empty():
			return joined
	if _cmu.has(word):
		return (_parse_cmu(String(_cmu[word])).phones as Array)
	if EXCEPTIONS.has(word):
		return (EXCEPTIONS[word] as Array).duplicate()
	var cl := _try_clitic(word)
	if not cl.is_empty():
		return cl
	var suf := _try_suffix(word)
	if not suf.is_empty():
		return suf
	# MORPHOLOGY, one level: strip a final plural / third-person `-s`, pronounce
	# the STEM, then reattach the suffix with voicing assimilation. Letter rules
	# alone cannot do this - they produced `comes` as K AA M EH S (a phantom
	# syllable), `gives` as D ZH IH V EH S, and `yours` as Y AW R S instead of
	# Y AO R Z. Going through the stem also lets the exceptions dictionary and
	# the magic-e rule do their work, which is why this one pass fixes the
	# vowel, the consonant and the spurious syllable at the same time.
	var stem := _s_stem(word)
	if not stem.is_empty():
		var sp := word_to_phones(stem)          # stem never ends in a strippable -s
		if sp.size() > 0:
			return _attach_s(sp, stem)
	var w := word
	var long_vowel_at := -1
	# magic-e: ...V C e$ (consonant not r/w) -> long vowel, silent e
	if w.length() >= 3 and w[w.length() - 1] == "e":
		var cons := w[w.length() - 2]
		var vow := w[w.length() - 3]
		if LONG.has(vow) and not (cons in "aeiourw"):
			long_vowel_at = w.length() - 3
			w = w.substr(0, w.length() - 1)
	# ... and the GENERAL silent final e, which magic-e cannot reach because it
	# only inspects single letters: `bathe` and `breathe` have a digraph in the
	# slot it checks, so they kept a whole spurious EH syllable. A final `e` is
	# silent unless the word is tiny (`be`, `he`, `the`) or the `e` follows
	# another vowel (`see`, `free`, `toe`), which is what those tests protect.
	if w.length() >= 4 and w[w.length() - 1] == "e" \
			and not VOWEL_LETTERS.contains(w[w.length() - 2]):
		w = w.substr(0, w.length() - 1)
	var phones: Array = []
	var i := 0
	while i < w.length():
		if i == long_vowel_at:
			phones.append(LONG[w[i]])
			i += 1
			continue
		var matched := false
		for span in [4, 3, 2]:
			if i + span <= w.length():
				var chunk := w.substr(i, span)
				# `th` is the one digraph whose value is contextual, so it is
				# decided here rather than in the table
				if chunk == "th":
					phones.append("DH" if _th_voiced(word, w, i) else "TH")
					i += 2
					matched = true
					break
				if DIGRAPHS.has(chunk):
					phones.append_array(DIGRAPHS[chunk])
					i += span
					matched = true
					break
		if matched:
			continue
		var c := w[i]
		var last: bool = i == w.length() - 1
		if SINGLES.has(c):
			# word-initial y is the glide, not the vowel
			if c == "y" and i == 0:
				phones.append("Y")
			# word-final y: AY in a monosyllable (`try`, `sky`, `fly`), IY
			# otherwise (`every`, `really`, `happy`) - it was IH for all of them
			elif c == "y" and last:
				phones.append("AY" if not _has_vowel_before(w, i) else "IY")
			# word-final o is the long one: `go`, `no`, `so` (and `goes` reaches
			# it through the -es stem)
			elif c == "o" and last and w.length() <= 3:
				phones.append("OW")
			# soft c / g before e, i, y - tested against the ORIGINAL spelling,
			# because a dropped silent e is exactly the letter that softens them
			# (`large`, `change`, `village` lose their trigger otherwise)
			elif c == "c" and i + 1 < word.length() and word[i + 1] in "eiy":
				phones.append("S")
			elif c == "g" and i + 1 < word.length() and word[i + 1] in "eiy" \
					and not HARD_G.has(word):
				phones.append_array(["D", "ZH"])
			else:
				phones.append_array(SINGLES[c])
		i += 1
	return phones


static func _has_vowel_before(w: String, i: int) -> bool:
	for k in i:
		if VOWEL_LETTERS.contains(w[k]):
			return true
	return false


## Is this `th` voiced? Three environments, checked against the ORIGINAL word
## (so a dropped magic-e still counts):
##   a function word starting in th-  (the, this, them, those, then, than)
##   between two vowels              (father, mother, other, together, weather)
##   the `-the$` spelling            (bathe, breathe, clothe, soothe)
static func _th_voiced(word: String, w: String, i: int) -> bool:
	if i == 0 and TH_VOICED.has(word):
		return true
	# `aeiou`, NOT VOWEL_LETTERS: a preceding `y` is a word-joint far more often
	# than a real vowel here, and it voiced the `th` in `everything`.
	if i > 0 and i + 2 < w.length():
		if "aeiou".contains(w[i - 1]) and "aeiou".contains(w[i + 2]):
			return true
	return word.ends_with("the") and i + 3 == word.length()


## Split a contraction at its apostrophe and pronounce the head through this
## same pipeline. `'s` reuses the plural's voicing rule, since it is the same
## clitic. Contractions whose HEAD changes shape (don't, won't, can't) are
## whole-word lexicon entries instead - `do` + `n't` would give D UW AH N T.
static func _try_clitic(word: String) -> Array:
	var at := word.rfind("'")
	if at <= 0 or at >= word.length() - 1:
		return []
	var head := word.substr(0, at)
	var tail := word.substr(at)
	var hp := word_to_phones(head)
	if hp.is_empty():
		return []
	if tail == "'s":
		return _attach_s(hp, head)
	# `n't` carries the n from the head's spelling, so match on it too
	if _clitics.has(tail):
		return hp + (_clitics[tail] as Array)
	if tail == "'t" and head.ends_with("n"):
		var stem := word_to_phones(head.substr(0, head.length() - 1))
		if not stem.is_empty() and _clitics.has("n't"):
			return stem + (_clitics["n't"] as Array)
	return []


## Try each suffix in `data/english.yml`: strip the spelling, rebuild a stem
## (undoubling `stopped -> stop`, restoring a dropped `e` in `hoping -> hope`),
## pronounce that stem through this same pipeline, then reattach.
##
## The stem has to be pronounced BEFORE the suffix can be chosen: whether the
## `-ed` of `watched` is a syllable, a /t/ or a /d/ depends on the last PHONE of
## the stem, which the letter rules cannot know from spelling. That dependency
## is the whole reason a flat rule table produced `watched` as W AE T SH EH D.
static func _try_suffix(word: String) -> Array:
	for s in _suffixes:
		var sp := String(s.spelling)
		if not word.ends_with(sp) or word.length() < sp.length() + 2:
			continue
		var base := word.substr(0, word.length() - sp.length())
		for stem in _stem_candidates(base, s):
			var sph := word_to_phones(stem)
			if sph.is_empty():
				continue
			return _attach_suffix(sph, s)
	return []


## The spellings to try for a stem, best first.
static func _stem_candidates(base: String, s: Dictionary) -> Array:
	var out: Array = []
	var n := base.length()
	out.append(base)
	# `hoping -> hope`: the stem's silent e was dropped before the suffix, and
	# restoring it lets magic-e fire again. Tried SECOND, because trying it
	# first turned `labeled` into `labele` -> L AE B IY L D; a stem that the
	# lexicon or the plain rules already know is always the better answer.
	if bool(s.get("restore_e", false)) and n >= 2 and not VOWEL_LETTERS.contains(base[n - 1]):
		out.append(base + "e")
	# `stopped -> stop`, `running -> run`: a final consonant was doubled to keep
	# the preceding vowel short
	if bool(s.get("undouble", false)) and n >= 3 and base[n - 1] == base[n - 2] \
			and not VOWEL_LETTERS.contains(base[n - 1]):
		out.append(base.substr(0, n - 1))
	return out


static func _attach_suffix(phones: Array, s: Dictionary) -> Array:
	var out: Array = phones.duplicate()
	if s.has("add"):
		out.append_array(_phone_list(s.add))
		return out
	var last := String(out[out.size() - 1])
	if _phone_list(s.get("syllabic_after", [])).has(last):
		out.append_array(_phone_list(s.get("syllabic", [])))
	elif _phone_list(s.get("voiceless_after", [])).has(last):
		out.append_array(_phone_list(s.get("voiceless", [])))
	else:
		out.append_array(_phone_list(s.get("otherwise", [])))
	return out


## The stem of a plural / third-person `-s`, or "" when the final `s` belongs to
## the word itself. No lexicon here, so the test is orthographic and deliberately
## conservative - a wrong strip invents a word, which is worse than missing one:
##   `ss$`             never (glass, class, less)
##   vowel + `s$`      never (bus, gas, yes, plus) - real suffixed forms after a
##                     vowel are spelled `-es` and caught below
##   vowel + `es$`     strip both (goes, toes, shoes)
##   sibilant + `es$`  strip both, syllabic (buses, boxes, wishes, churches)
##   otherwise `s$`    strip the s only, keeping any `e` so magic-e still fires
##                     (comes -> come, gives -> give, makes -> make)
static func _s_stem(word: String) -> String:
	var n := word.length()
	if n < 4 or not word.ends_with("s") or word.ends_with("ss"):
		return ""
	var prev := word[n - 2]
	if prev != "e":
		if VOWEL_LETTERS.contains(prev):
			return ""                       # bus, gas, yes, plus, focus
		return word.substr(0, n - 1)        # dogs, yours, cats, lands
	var prev2 := word[n - 3]
	if VOWEL_LETTERS.contains(prev2) or prev2 in "sxzhc":
		return word.substr(0, n - 2)        # goes, toes / buses, boxes, wishes
	return word.substr(0, n - 1)            # comes, gives, makes, hopes


## Reattach the `-s`: syllabic after a sibilant, voiceless after a voiceless
## phone, voiced otherwise. `stem` only decides the syllabic case for spellings
## whose `e` we already dropped.
static func _attach_s(phones: Array, stem: String) -> Array:
	var out: Array = phones.duplicate()
	var last := String(out[out.size() - 1])
	if SIBILANT_PHONES.has(last):
		out.append_array(["IH", "Z"])
	elif VOICELESS_PHONES.has(last):
		out.append("S")
	else:
		out.append("Z")
	return out


## Does this word carry no sentence accent? Data-driven (data/english.yml
## `function_words`), falling back to the built-in list.
static func is_function_word(word: String) -> bool:
	_load_data()
	var table: Variant = _function_words if not _function_words.is_empty() else null
	if table == null:
		return FUNCTION_WORDS.has(word)
	if _function_words.has(word):
		return true
	# a contraction of function words is still a function word: `we're`, `it's`
	# and `you've` were being read as content and taking accents off the verb
	var at := word.rfind("'")
	if at > 0:
		return _function_words.has(word.substr(0, at))
	return false


## Index (into phones) of the vowel that carries the word's accent: the first
## vowel of a stressed word. -1 if the word has no vowel.
static func stress_vowel(phones: Array, stress: Array = []) -> int:
	if not stress.is_empty():
		for want in [1, 2]:
			for i in mini(phones.size(), stress.size()):
				if int(stress[i]) == want \
						and TABLE.get(phones[i], {}).get("type", "") == "vowel":
					return i
	for i in phones.size():
		if TABLE.get(phones[i], {}).get("type", "") == "vowel":
			return i
	return -1

## Is this token nothing but dash characters? A spaced dash is an aside, not a word.
static func _is_dash(t: String) -> bool:
	if t.is_empty():
		return false
	for i in t.length():
		if not (t[i] == "-" or t[i] == "\u2013" or t[i] == "\u2014"):
			return false
	return true


## Is the terminal mark on token [param at] really the end of a sentence?
##
## A DIALOGUE TAG IS NOT A NEW SENTENCE. Prose attributes speech by closing the quotation
## and continuing the same sentence in lower case:
##
##     "Do you think Jesus would do this?" you asked.
##     "Hmm," you said, the remote still raised.
##
## The question mark is inside the quotation because it belongs to what was SAID, not to
## the sentence containing it - so splitting there put "you asked." on a subtitle card of
## its own, orphaned from the line it belongs to. Written English has a simple, reliable
## signal for this and it is not the quotation: a genuine new sentence begins with a
## CAPITAL. A lower-case word after a full stop is a continuation, whether it follows a
## dialogue tag or an abbreviation this front end failed to expand.
##
## THE PAUSE IS UNAFFECTED, deliberately. A reader does rest at the close of a quotation
## before the attribution, so `pause_after` and the terminal contour are left exactly as
## they were - this changes only how the words are GROUPED, which is what the karaoke
## draws. The voice is untouched.
static func _ends_sentence(toks: PackedStringArray, at: int) -> bool:
	if at + 1 >= toks.size():
		return true
	var nxt := String(toks[at + 1])
	# Skip anything that carries no case of its own, so an opening quote or bracket before
	# the next word does not hide it.
	var i := 0
	while i < nxt.length() and nxt[i] in "\"\'([" :
		i += 1
	if i >= nxt.length():
		return true
	var c := nxt[i]
	# Only a lower-case LETTER continues. A digit, a symbol or a capital all start
	# something new - "He counted. 5 was missing." is two sentences.
	return not (c >= "a" and c <= "z")


## Words ghost pronounces ITSELF, because eSpeak reads them wrong.
##
## eSpeak picks these from its own part-of-speech guess, and on some very common constructions it
## is confidently wrong: "the way a prisoner lives in his cell" comes back with the NOUN reading
## (lˈaɪvz), because it parses "prisoner lives" as a compound noun. Sentence context does not
## rescue it - measured, the whole-sentence transcription returns the same wrong answer as the word
## in isolation, so this is not something a phonemizer setting can fix.
##
## Each entry gives the reading to use by DEFAULT and the reading to switch to when the word before
## or after says otherwise. The resolved word is marked `literal`, which makes ghost's own ARPAbet
## win for that word only - the rest of the sentence still gets eSpeak's transcription, so nothing
## else about the reading changes.
##
## Add sparingly and only where eSpeak is measurably wrong: a bad rule here mispronounces a word
## that was previously fine, which is worse than the occasional homograph.
# Shared by `record` / `records`, which take the same contexts. Named constants rather than
# two inline copies, so the pair can never drift apart.
#
#   VERB_PREV - a subject pronoun, modal, or infinitive marker: only a verb can follow.
#   VERB_NEXT - a determiner or quantifier opening an OBJECT, so the word is transitive.
#   NOUN_PREV - a determiner or preposition directly before it, which always makes it a noun
#               ("for the record", "on record"). Vetoes VERB_NEXT, so "the record a clerk
#               kept" does not switch on that "a".
const _RECORD_VERB_PREV := ["i", "we", "they", "you", "he", "she", "who", "to", "will",
	"would", "shall", "should", "can", "could", "cannot", "must", "may", "might",
	"do", "does", "did", "don't", "didn't", "never", "not", "also", "please", "let",
	"help", "helps", "helped", "carefully", "dutifully", "faithfully"]
const _RECORD_VERB_NEXT := ["a", "an", "the", "no", "every", "each", "all", "any", "some",
	"this", "that", "these", "those", "his", "her", "its", "their", "our", "my", "your",
	"it", "them", "him", "everything", "nothing", "what", "how", "only", "both",
	"more", "less", "fewer", "several", "many", "much",
	"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
const _RECORD_NOUN_PREV := ["the", "a", "an", "this", "that", "these", "those", "his", "her",
	"its", "their", "my", "your", "our", "every", "each", "one", "another", "same",
	"new", "old", "world", "official", "public", "written", "permanent", "track",
	"on", "off", "for", "of", "in", "into", "by", "with", "at", "from", "without"]

const SPEAK_AS := {
	"lives": {
		"base": "L IH1 V Z",       # the VERB - "a prisoner lives here"
		"alt": "L AY1 V Z",        # the NOUN  - "their lives changed"
		"alt_prev": ["the", "their", "our", "your", "my", "his", "her", "its", "these",
			"those", "many", "all", "both", "few", "several", "countless", "other",
			"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
			"young", "past"],
		"alt_next": ["were", "are", "of", "matter", "lost", "hang", "depend"],
	},
	# RECORD. The same failure as `lives`, found the same way: eSpeak reads the VERB as the NOUN
	# whenever the subject is a plural noun rather than a pronoun, because it parses "<noun>
	# record" as a compound. Measured -
	#   "They record no discrepancy."         -> ɹᵻkˈɔːɹd   correct (verb)
	#   "He records everything."              -> ɹᵻkˈɔːɹdz  correct (verb)
	#   "The ledgers record no discrepancy."  -> ɹˈɛkɚd     WRONG  (noun)
	#   "The ledger records no discrepancy."  -> ɹˈɛkɚdz    WRONG  (noun)
	#   "Historians record the year."         -> ɹˈɛkɚd     WRONG  (noun)
	#
	# Every noun context it already gets right ("the record shows", "a record of", "record
	# player", "she broke the record") returns ɹˈɛkɚd, and `base` here maps through
	# arpabet.to_symbols to exactly ɹˈɛkɚd - so a rule that does NOT fire changes nothing.
	# That is the safety property: this can only ever alter the reading it fires on.
	#
	# The switch is keyed to what FOLLOWS, because a transitive verb takes an object - a
	# determiner or quantifier after the word means it is being DONE, not held. `keep_prev`
	# vetoes the one construction that test would otherwise fool ("the record a clerk kept"),
	# since a determiner or preposition directly before the word always makes it the noun.
	"record": {
		"base": "R EH1 K ER0 D",      # the NOUN - "the record shows", "a record of"
		"alt": "R IH0 K AO1 R D",     # the VERB - "the ledgers record no discrepancy"
		"alt_prev": _RECORD_VERB_PREV,
		"alt_next": _RECORD_VERB_NEXT,
		"keep_prev": _RECORD_NOUN_PREV,
	},
	"records": {
		"base": "R EH1 K ER0 D Z",
		"alt": "R IH0 K AO1 R D Z",
		"alt_prev": _RECORD_VERB_PREV,
		"alt_next": _RECORD_VERB_NEXT,
		"keep_prev": _RECORD_NOUN_PREV,
	},
	# HUMS. eSpeak spells these out instead of humming them - measured, it returns
	#   "hmm" -> həm ("hem"),  "hm" -> ˌeɪtʃˈɛm ("aitch em"),  "mmm" -> ˌɛmˌɛmˈɛm ("em em em").
	#
	# A hum is NASAL MURMUR: nearly all of its energy sits below 500 Hz, with no vowel
	# formants above it. Measured on this voice, share of spectral energy under 500 Hz -
	#   "M M"    80.6%  (0.155 s)   <- a hum
	#   "M"      90.7%  (0.064 s)   right shape, too short to register
	#   "HH M"   41.5%              the HH puts 57% into 0.5-2 kHz: heard as "em"
	#   "HH AH M" 32.2%             indistinguishable from the real word "hum" (28.0%)
	# So no H and no vowel, and doubled because the voice has no syllabic m - a single
	# one is over before it reads as anything. This is the second attempt: "HH M" was
	# chosen for having no vowel, which was right in principle, but the H alone carries
	# enough mid-band energy to be heard as one.
	"hm": {"base": "M M"},
	"hmm": {"base": "M M"},
	"hmmm": {"base": "M M M"},
	"mm": {"base": "M M"},
	"mmm": {"base": "M M M"},
}


## Pick each homograph's reading from its neighbours, once the whole sentence is known.
static func _resolve_homographs(sentences: Array) -> void:
	for s in sentences:
		var sent: Array = s
		for i in sent.size():
			var w: Dictionary = sent[i]
			var key := String(w.get("text", "")).to_lower()
			if bool(w.get("literal", false)):
				continue                      # an authored [L IH V Z] outranks every table
			# A NAME beats everything else here. These are words no dictionary holds -
			# invented ones, coined ones, proper nouns - so there is no reading to weigh
			# them against; whatever the file says is simply what they are called.
			if _names.has(key):
				var nm := _parse_cmu(String(_names[key]))
				w["phones"] = nm.phones
				w["stress"] = nm.stress
				w["literal"] = true
				continue
			if not SPEAK_AS.has(key):
				continue
			var rule: Dictionary = SPEAK_AS[key]
			var prev := String(sent[i - 1].get("text", "")).to_lower() if i > 0 else ""
			var nxt := String(sent[i + 1].get("text", "")).to_lower() if i + 1 < sent.size() else ""
			# An entry with no `alt` is a plain override: one reading, always.
			#
			# `keep_prev` is a VETO, checked first and beating both alt lists. Without it a
			# rule keyed to the following word cannot tell "the ledgers record a loss" (verb)
			# from "the record a clerk kept" (noun) - the word after is "a" in both. A
			# determiner or preposition immediately before the word settles it, so that case
			# is answered by the left neighbour and never reaches the right one.
			var vetoed: bool = (rule.get("keep_prev", []) as Array).has(prev)
			var alt: bool = rule.has("alt") and not vetoed \
				and ((rule.get("alt_prev", []) as Array).has(prev) \
					or (rule.get("alt_next", []) as Array).has(nxt))
			var got := _parse_cmu(String(rule["alt"] if alt else rule["base"]))
			w["phones"] = got.phones
			w["stress"] = got.stress
			w["literal"] = true
