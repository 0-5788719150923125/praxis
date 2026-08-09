extends RefCounted
class_name TextNorm

## TextNorm - turn written English into speakable words, before anything else.
##
## This stage did not exist, and its absence was silent rather than noisy, which
## is why it survived so long. Two independent defects, measured on manuscript
## prose (see next/voice_intelligibility.md):
##
##   NUMERALS VANISHED. [Phonemes.word_to_phones] walks a word character by
##   character looking each up in DIGRAPHS/SINGLES, which contain only letters,
##   and the loop has no else branch - an unmatched character is skipped. A
##   purely numeric token therefore returns an EMPTY phone array and the word is
##   dropped from the utterance entirely. Not mispronounced: gone, with nothing
##   in the audio or the subtitles to show it. 2.2% of tokens on ordinary prose.
##
##   TYPOGRAPHIC PUNCTUATION BROKE LOOKUP. The punctuation set was the ASCII
##   literal ".,!?;:" and the wrapper sets were ASCII, so U+2018/2019 (curly
##   quotes), U+201C/201D, U+2013/2014 (dashes) and U+2026 (ellipsis) matched
##   nothing, survived into the dictionary key, and turned every quoted or
##   contracted word into a miss that fell through to letter-to-sound rules.
##   Dictionary coverage measured 88.3% on typographic text against 92.7% for
##   the same text in ASCII, with 10.4% of tokens damaged in total.
##
## The fix is ordered: fold characters first, then expand numbers, then hand
## clean ASCII to the tokenizer. Everything here is deterministic and has no
## opinion about voice - it is a text transform, and it is equally required by
## a procedural synthesizer and by any neural backend (see VOICE_PLAN.md P0).

# --- character folding ------------------------------------------------------
# Typographic punctuation folded to the ASCII the rest of the front end already
# understands. The dashes become a comma because that is what an em dash does
# prosodically: it is a break, not a word.
const FOLD := {
	"‘": "'", "’": "'", "‚": ",", "‛": "'",   # single quotes
	"“": "\"", "”": "\"", "„": "\"", "‟": "\"", # double quotes
	"′": "'", "″": "\"",                                 # primes
	"‐": "-", "‑": "-", "‒": "-", "–": "-",     # hyphens/en dash
	"—": ",", "―": ",",                                  # em dash = a break
	"…": ",",                                                 # ellipsis = a break
	" ": " ", " ": " ", " ": " ", " ": " ",     # exotic spaces
	"­": "",                                                  # soft hyphen
	"×": " times ", "÷": " divided by ",
	"½": " one half ", "¼": " one quarter ", "¾": " three quarters ",
	"°": " degrees ", "™": "", "®": "", "©": "",
}

# --- number words -----------------------------------------------------------
const ONES := ["zero", "one", "two", "three", "four", "five", "six", "seven",
	"eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
	"fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
const TENS := ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
	"eighty", "ninety"]
const SCALES := [[1000000000, "billion"], [1000000, "million"], [1000, "thousand"]]
# Ordinals that are not just "<cardinal>th".
const ORDINAL := {
	"one": "first", "two": "second", "three": "third", "five": "fifth",
	"eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}

# Abbreviations whose trailing period is NOT a sentence boundary. Expanded to
# their spoken form, which also removes the period and the false full stop with
# it - a chapter of dialogue was measuring 7 false sentence stops in 42.
const ABBREV := {
	"mr": "mister", "mrs": "missus", "ms": "miz", "dr": "doctor",
	"st": "saint", "mt": "mount", "prof": "professor", "rev": "reverend",
	"jr": "junior", "sr": "senior", "vs": "versus", "etc": "et cetera",
	"approx": "approximately", "dept": "department", "est": "established",
	"fig": "figure", "no": "number", "vol": "volume", "ch": "chapter",
}

const CURRENCY := {"$": "dollars", "£": "pounds", "€": "euros", "¥": "yen"}


## The entry point. Fold, expand, and hand back plain ASCII words.
static func normalize(text: String) -> String:
	var s := _fold(text)
	s = _expand_numbers(s)
	s = _expand_abbrev(s)
	return s


static func _fold(text: String) -> String:
	var out := ""
	for i in text.length():
		var c := text[i]
		out += String(FOLD[c]) if FOLD.has(c) else c
	return out


## Cardinals under a thousand. The recursive core everything else calls.
static func _under_thousand(n: int) -> String:
	if n < 20:
		return String(ONES[n])
	if n < 100:
		var t: String = TENS[n / 10]
		return t if n % 10 == 0 else t + " " + String(ONES[n % 10])
	var h: String = String(ONES[n / 100]) + " hundred"
	return h if n % 100 == 0 else h + " " + _under_thousand(n % 100)


## Any non-negative integer, spoken the way a reader would say it.
static func cardinal(n: int) -> String:
	if n == 0:
		return "zero"
	var parts: Array = []
	var rest := n
	for pair in SCALES:
		var value: int = pair[0]
		if rest >= value:
			parts.append(_under_thousand(rest / value) + " " + String(pair[1]))
			rest = rest % value
	if rest > 0:
		parts.append(_under_thousand(rest))
	return " ".join(parts)


static func ordinal(n: int) -> String:
	var words := cardinal(n)
	var idx := words.rfind(" ")
	var head := words.substr(0, idx + 1) if idx >= 0 else ""
	var tail := words.substr(idx + 1)
	if ORDINAL.has(tail):
		return head + String(ORDINAL[tail])
	if tail.ends_with("y"):                      # twenty -> twentieth
		return head + tail.substr(0, tail.length() - 1) + "ieth"
	return head + tail + "th"


## A four-digit year reads as two pairs - "nineteen eighty five", not "one
## thousand nine hundred and eighty five". 2000-2009 are the exception that
## every rule-based normalizer gets wrong; they read as full cardinals.
static func year(n: int) -> String:
	if n < 1100 or n > 2999:
		return cardinal(n)
	var hi := n / 100
	var lo := n % 100
	if lo == 0:
		return cardinal(hi) + " hundred"
	if n >= 2000 and n <= 2009:
		return cardinal(n)
	if lo < 10:
		return cardinal(hi) + " oh " + cardinal(lo)
	return cardinal(hi) + " " + cardinal(lo)


static func _is_digits(s: String) -> bool:
	if s.is_empty():
		return false
	for i in s.length():
		if s[i] < "0" or s[i] > "9":
			return false
	return true


## Walk the text token by token, replacing anything numeric with words. Runs on
## FOLDED text, so it only ever sees ASCII.
static func _expand_numbers(text: String) -> String:
	var out: Array = []
	for raw in text.split(" ", false):
		out.append(_expand_token(String(raw)))
	return " ".join(out)


static func _expand_token(tok: String) -> String:
	# hold trailing punctuation aside so it still reaches the tokenizer
	var tail := ""
	while tok.length() > 0 and tok[tok.length() - 1] in ".,!?;:\"')":
		tail = tok[tok.length() - 1] + tail
		tok = tok.substr(0, tok.length() - 1)
	var head := ""
	while tok.length() > 0 and tok[0] in "\"'(":
		head += tok[0]
		tok = tok.substr(1)
	if tok.is_empty():
		return head + tail

	var body := _expand_core(tok)
	return head + body + tail


static func _expand_core(tok: String) -> String:
	# currency: $5, £1.50
	var first := tok.substr(0, 1)
	if CURRENCY.has(first) and tok.length() > 1:
		var amount := _expand_core(tok.substr(1))
		return amount + " " + String(CURRENCY[first])
	# percentages
	if tok.ends_with("%"):
		return _expand_core(tok.substr(0, tok.length() - 1)) + " percent"
	# ordinals written with a suffix: 1st, 22nd, 3rd, 14th
	for suf in ["st", "nd", "rd", "th"]:
		if tok.length() > suf.length() and tok.to_lower().ends_with(suf):
			var digits := tok.substr(0, tok.length() - suf.length())
			if _is_digits(digits):
				return ordinal(int(digits))
	# times: 9:30
	if tok.contains(":"):
		var bits := tok.split(":")
		if bits.size() == 2 and _is_digits(bits[0]) and _is_digits(bits[1]):
			var m := int(bits[1])
			if m == 0:
				return cardinal(int(bits[0])) + " o'clock"
			return cardinal(int(bits[0])) + (" oh " if m < 10 else " ") + cardinal(m)
	# decimals: 3.14 reads digit by digit after the point
	if tok.contains("."):
		var bits2 := tok.split(".")
		if bits2.size() == 2 and _is_digits(bits2[0]) and _is_digits(bits2[1]):
			var frac := ""
			for i in String(bits2[1]).length():
				frac += " " + String(ONES[int(String(bits2[1])[i])])
			return cardinal(int(bits2[0])) + " point" + frac
	# thousands separators: 1,200
	if tok.contains(",") and _is_digits(tok.replace(",", "")):
		return cardinal(int(tok.replace(",", "")))
	# hyphenated ranges and compounds: 1939-1945, twenty-one
	if tok.contains("-") and tok.length() > 1:
		var parts: Array = []
		for piece in tok.split("-", false):
			parts.append(_expand_core(String(piece)))
		return " ".join(parts)
	# bare digits: a plausible year reads as pairs, everything else as a cardinal
	if _is_digits(tok):
		var n := int(tok)
		if tok.length() == 4 and n >= 1100 and n <= 2999:
			return year(n)
		return cardinal(n)
	return tok


## Dotted abbreviations, expanded so their period stops being read as the end of
## a sentence. Case-insensitive on the abbreviation, and only when the period is
## actually present - "no" the word must not become "number".
static func _expand_abbrev(text: String) -> String:
	var out: Array = []
	for raw in text.split(" ", false):
		var tok := String(raw)
		var lower := tok.to_lower()
		var matched := false
		for key in ABBREV:
			if lower == key + ".":
				out.append(String(ABBREV[key]))
				matched = true
				break
		if not matched:
			out.append(tok)
	return " ".join(out)
