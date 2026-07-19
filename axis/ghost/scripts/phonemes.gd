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
	# nasals (weak murmur through low formants)
	"M": {"type": "nasal", "f": [280.0, 900.0, 2200.0], "dur": 70.0},
	"N": {"type": "nasal", "f": [280.0, 1700.0, 2600.0], "dur": 65.0},
	"NG": {"type": "nasal", "f": [280.0, 2300.0, 2750.0], "dur": 75.0},
	# fricatives: noise_f / noise_bw shape the frication; voiced ones add the buzz
	"S": {"type": "fric", "voiced": false, "noise_f": 6200.0, "noise_bw": 2600.0, "namp": 0.6, "dur": 105.0},
	"Z": {"type": "fric", "voiced": true, "noise_f": 6200.0, "noise_bw": 2600.0, "namp": 0.35, "dur": 90.0},
	"SH": {"type": "fric", "voiced": false, "noise_f": 2700.0, "noise_bw": 1600.0, "namp": 0.6, "dur": 110.0},
	"ZH": {"type": "fric", "voiced": true, "noise_f": 2700.0, "noise_bw": 1600.0, "namp": 0.35, "dur": 95.0},
	"F": {"type": "fric", "voiced": false, "noise_f": 4500.0, "noise_bw": 4000.0, "namp": 0.25, "dur": 95.0},
	"V": {"type": "fric", "voiced": true, "noise_f": 4500.0, "noise_bw": 4000.0, "namp": 0.15, "dur": 70.0},
	"TH": {"type": "fric", "voiced": false, "noise_f": 5500.0, "noise_bw": 4500.0, "namp": 0.2, "dur": 90.0},
	"DH": {"type": "fric", "voiced": true, "noise_f": 5500.0, "noise_bw": 4500.0, "namp": 0.12, "dur": 60.0},
	# stops: closure then a short burst; burst_f centres the burst noise
	"P": {"type": "stop", "voiced": false, "burst_f": 900.0, "burst_bw": 1600.0, "dur": 90.0},
	"B": {"type": "stop", "voiced": true, "burst_f": 900.0, "burst_bw": 1600.0, "dur": 75.0},
	"T": {"type": "stop", "voiced": false, "burst_f": 4200.0, "burst_bw": 2600.0, "dur": 90.0},
	"D": {"type": "stop", "voiced": true, "burst_f": 4200.0, "burst_bw": 2600.0, "dur": 75.0},
	"K": {"type": "stop", "voiced": false, "burst_f": 1900.0, "burst_bw": 1400.0, "dur": 95.0},
	"G": {"type": "stop", "voiced": true, "burst_f": 1900.0, "burst_bw": 1400.0, "dur": 80.0},
	# affricates are expanded at parse time: CH -> T SH, JH -> D ZH
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
	for token in _tokenize(text):
		if token.begins_with("["):
			words.append(_literal_word(token))
			continue
		var pause := "none"
		var punct := ""
		var bare := token
		while bare.length() > 0 and bare[bare.length() - 1] in ".,!?;:\n":
			var c := bare[bare.length() - 1]
			pause = "stop" if c in ".!?\n" else "comma"
			if punct.is_empty() and c != "\n":
				punct = c            # the terminal mark drives the contour (?, !, .)
			bare = bare.substr(0, bare.length() - 1)
		bare = bare.to_lower().strip_edges()
		bare = bare.lstrip("\"'(").rstrip("\"')")
		if bare.length() > 0:
			var phones := word_to_phones(bare)
			if phones.size() > 0:
				words.append({
					"text": bare,
					"phones": phones,
					"stressed": not FUNCTION_WORDS.has(bare),
					"pause_after": pause,
					"punct": punct,
				})
		if pause == "stop" and words.size() > 0:
			sentences.append(words)
			words = []
	if words.size() > 0:
		sentences.append(words)
	return sentences


static func _tokenize(text: String) -> PackedStringArray:
	var out := PackedStringArray()
	var i := 0
	while i < text.length():
		var c := text[i]
		if c == "[":
			var close := text.find("]", i)
			if close < 0:
				close = text.length() - 1
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
	return {"text": inner.to_lower(), "phones": phones, "stressed": true, "pause_after": "none"}


## One lowercase word -> phoneme keys, via exceptions, magic-e, digraphs, singles.
static func word_to_phones(word: String) -> Array:
	if EXCEPTIONS.has(word):
		return (EXCEPTIONS[word] as Array).duplicate()
	var w := word
	var long_vowel_at := -1
	# magic-e: ...V C e$ (consonant not r/w) -> long vowel, silent e
	if w.length() >= 3 and w[w.length() - 1] == "e":
		var cons := w[w.length() - 2]
		var vow := w[w.length() - 3]
		if LONG.has(vow) and not (cons in "aeiourw"):
			long_vowel_at = w.length() - 3
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
				if DIGRAPHS.has(chunk):
					phones.append_array(DIGRAPHS[chunk])
					i += span
					matched = true
					break
		if matched:
			continue
		var c := w[i]
		if SINGLES.has(c):
			# word-initial y is the glide, not the vowel
			if c == "y" and i == 0:
				phones.append("Y")
			# soft c / g before e, i, y
			elif c == "c" and i + 1 < w.length() and w[i + 1] in "eiy":
				phones.append("S")
			elif c == "g" and i + 1 < w.length() and w[i + 1] in "eiy" and word != "give" and word != "get":
				phones.append_array(["D", "ZH"])
			else:
				phones.append_array(SINGLES[c])
		i += 1
	return phones


## Index (into phones) of the vowel that carries the word's accent: the first
## vowel of a stressed word. -1 if the word has no vowel.
static func stress_vowel(phones: Array) -> int:
	for i in phones.size():
		var entry: Dictionary = TABLE.get(phones[i], {})
		if entry.get("type", "") == "vowel":
			return i
	return -1
