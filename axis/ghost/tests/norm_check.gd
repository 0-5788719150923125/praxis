extends SceneTree

## norm_check - the gate for [TextNorm], and for the punctuation ordering fix
## in [Phonemes.parse].
##
## Every defect this covers is SILENT: a numeral produced an empty phone array
## and vanished from the utterance with nothing in the audio or the subtitles to
## show it, and a curly apostrophe turned a dictionary hit into a miss that fell
## through to letter rules and merely sounded a bit wrong. Neither would ever
## have failed a test that only checked "does it render".
##
## THE THIRD DEFECT, reported from a chapter render, is why the WHITESPACE cases below exist:
## `2009` vanished from "an opponent who left the building in 2009." - out of the audio and out of
## the subtitles both, with the sentence ending at "in". It was the last token of a paragraph, and
## `_expand_numbers` split its input on the SPACE CHARACTER, so the token arrived as `"2009.\n"`
## with the newline attached; the trailing-punctuation strip does not know `\n`, so the digits
## never reached the numeral rules and survived as digits into a phonemizer that walks letters.
## Every numeral case here is therefore run TWICE - as given, and again with a newline welded to
## its end - because the pass had a hole that only opened at the end of a line.
##
## AND NOTHING MAY BE DROPPED IN SILENCE, which is the general form of all three: the no-drop
## block below requires every token of a paragraph to reach the parse output, and
## Phonemes._rescue_phones plus GenerativeEditor._bridge_words make the two places that used to
## delete a word warn and keep it instead.
##
## Run: godot --headless --path axis/ghost --script tests/norm_check.gd

const TextNorm_ := preload("res://scripts/text_norm.gd")
const Phonemes_ := preload("res://scripts/phonemes.gd")

# expected substrings, because the surrounding punctuation is preserved
const CASES := [
	# numerals - these used to disappear entirely
	["I counted 42 of them.", "forty two"],
	["Chapter 7 begins.", "seven"],
	["He was 1 of 3.", "one"],
	# Capitalised because it OPENS the sentence. A spelled-out numeral is a word like any
	# other and a sentence starts with a capital - and the subtitle shows this string. It
	# also matters mechanically: the sentence splitter reads a lower-case word after a full
	# stop as a continuation (a dialogue tag), so an uncapitalised expansion would weld
	# "1985 was a long time ago." onto the previous sentence's subtitle card.
	["1985 was a long time ago.", "Nineteen eighty five"],
	["It happened in 2003.", "two thousand three"],
	["The year 1900 passed.", "nineteen hundred"],
	["She paid $5.", "five dollars"],
	["Only 20% remained.", "twenty percent"],
	["The 1st of May.", "first"],
	["His 22nd birthday.", "twenty second"],
	["It cost 1,200 pounds.", "one thousand two hundred"],
	["Pi is 3.14 roughly.", "three point one four"],
	["We met at 9:30.", "nine thirty"],
	["Meet me at 6:00.", "o'clock"],
	["The 1939-1945 war.", "nineteen thirty nine"],
	# typographic punctuation - these used to break dictionary lookup
	["“Go away,” she said.", "\"Go away,\" she said."],
	["It’s his.", "It's his."],
	["A pause — then nothing.", ","],
	["Wait… no.", ","],
	# abbreviations - the period used to read as a full stop
	# alphanumerics - "H8" is a board square, "v2" a version. These were not dropped, they were
	# MANGLED: word_to_phones walks letters, so the whole token came back as one phone.
	["He played H8.", "aitch eight"],
	["the v2 build", "vee two"],
	["a 3D print", "three dee"],
	["page A4.", "ay four"],
	["COVID19 rules", "COVID nineteen"],
	["Mr. Blake arrived.", "mister"],
	["Dr. Vane left.", "doctor"],
	["St. Anne's church.", "saint"],
]

# words whose terminal punctuation must survive a wrapping quote, which is the
# ordering bug in parse(): the strip loop met the quote first and gave up, so
# every line of dialogue lost its contour and its pause
## TEMPLATE MACROS: [input, must be read, must NOT be read].
##
## The second column is the whole point and is why these are not ordinary CASES.
## A manuscript carries facts only its build can resolve - how many chapters
## precede this one - and writes `${CHAPTERS_BEFORE_IN_WORDS:twenty-one}` so the
## sentence maintains itself. Ghost is handed one chapter with no book around it
## and cannot resolve any of them, so it reads the DEFAULT; what it must never do
## is read the placeholder, which is one whitespace-delimited token and was
## therefore spoken dollar sign, underscores and all.
const MACRO_CASES := [
	# the form the manuscript actually uses
	["a book for ${CHAPTERS_BEFORE_IN_WORDS:twenty-one} chapters.",
		"for twenty-one chapters", "CHAPTERS"],
	# a numeral default still travels the rest of the pipeline
	["a book for ${CHAPTERS_BEFORE_IN_WORDS:25} chapters.", "twenty five", "25"],
	# defaults may be several words, and may carry an apostrophe
	["It is ${TITLE:Charlotte's Web of Lies} today.", "Charlotte's Web of Lies", "TITLE"],
	# NO DEFAULT: the placeholder goes rather than being read. It warns; see
	# TextNorm._expand_macros for why it may not simply stay.
	["a book for ${WORD_COUNT_IN_WORDS} chapters.", "for chapters", "WORD_COUNT"],
	# an explicitly EMPTY default is a deliberate deletion, not an omission
	["a ${GONE:}placeholder.", "a placeholder", "GONE"],
	# two in one sentence, and a bare $ that is NOT a macro must still be money
	["cost $5 and ${A:1} plus ${B:2}.", "five dollars and one plus two", "${"],
	# A DEFAULT MAY BE ANY LENGTH. There is nothing of this size in the book
	# today and the author has said there will be, so it is held now rather than
	# discovered in a render: several sentences, a line break inside the value,
	# and a blank line - a whole paragraph break - inside it.
	["He said ${Q:It was a long day. It ended badly.} and left.",
		"It was a long day. It ended badly. and left", "${"],
	["He said ${Q:a long day,\nand a bad end.} then.", "and a bad end. then", "${"],
	["Said ${Q:One para.\n\nAnd a second.} then.", "One para.\n\nAnd a second. then", "${"],
	# ...and a value's CONTENT is not the parser's business: a brace inside it
	# nests rather than ending it. `[^{}]*` refused to match this at all, which
	# left the entire macro to be read aloud - silently, and only in the one
	# sentence that happened to contain a brace.
	["A ${Q:set {a, b} of things} here.", "A set {a, b} of things here", "${"],
	# A MACRO THAT NEVER CLOSES IS LEFT VERBATIM, not swallowed. Reading `${Q:`
	# aloud is loud and reported; taking the value to end of file would delete
	# the rest of the chapter, which is the failure this project keeps un-fixing.
	["A ${Q:runaway and the rest of it.", "and the rest of it", "!!never!!"],
]

const PAUSE_CASES := [
	["“Go away,” he said.", "away", "comma"],
	["“Who is there?” she asked.", "there", "stop"],
	["He said \"stop.\"", "stop", "stop"],
]


func _init() -> void:
	var fails := 0

	for c in CASES:
		var got: String = TextNorm_.normalize(String(c[0]))
		if not got.contains(String(c[1])):
			print("norm_check: FAIL  %-34s -> %s   (wanted %s)" % [c[0], got, c[1]])
			fails += 1
		# ...AND AT THE END OF A LINE, which is where the pass used to give up. Run every case
		# again with a newline attached: the expansion must be identical and the newline must
		# survive, because a paragraph break is a full stop to the sentence splitter.
		var nl: String = TextNorm_.normalize(String(c[0]) + "\n")
		if not nl.contains(String(c[1])):
			print("norm_check: FAIL  %-34s + newline -> %s   (wanted %s)" % [c[0], nl, c[1]])
			fails += 1
		if not nl.ends_with("\n"):
			print("norm_check: FAIL  the newline was eaten: %s" % nl)
			fails += 1

	# A numeral in every awkward POSITION, since the hole was positional.
	var pos := [
		["in 2009.", "two thousand nine"],
		["in 2009.\n", "two thousand nine"],
		["in 2009", "two thousand nine"],
		["in 2009\n", "two thousand nine"],
		["a room 12 by 40.\n", "twelve"],
		["a room 12 by 40.\n", "forty"],
		["line one\n2009 opened it.\n", "Two thousand nine"],
		["tabbed\t2009.\n", "two thousand nine"],
	]
	for c in pos:
		var got2: String = TextNorm_.normalize(String(c[0]))
		if not got2.contains(String(c[1])):
			print("norm_check: FAIL  %-24s -> %s   (wanted %s)"
				% [String(c[0]).replace("\n", "\\n"), got2.replace("\n", "\\n"), c[1]])
			fails += 1

	# NOTHING VANISHES. Every whitespace-separated token of these paragraphs carries a letter or a
	# digit, so every one of them has something to say and must appear in the parse output. This is
	# the claim the reported bug broke: the count came back one word short and nothing said so.
	var paras := PackedStringArray([
		"He delivered the closing argument of the decade to an opponent who left the "
			+ "building in 2009.\nIt was over.",
		"A room 12 by 40, in 1985, for $5 a night.\nThe 1st of May.\n",
		"Call 555 before 9:30.\n",
	])
	for para in paras:
		var seen := 0
		for sentence in Phonemes_.parse(para):
			seen += (sentence as Array).size()
		# What the parse SHOULD contain: one word per source token, except that a numeral
		# expands into several - so the count may only ever be GREATER than the token count.
		var toks := 0
		for raw in para.replace("\n", " ").split(" ", false):
			var t := String(raw).strip_edges()
			var has := false
			for i in t.length():
				if (t[i] >= "a" and t[i] <= "z") or (t[i] >= "A" and t[i] <= "Z") \
						or (t[i] >= "0" and t[i] <= "9"):
					has = true
					break
			if has:
				toks += 1
		if seen < toks:
			print("norm_check: FAIL  %d words parsed from %d speakable tokens - something was "
				% [seen, toks] + "dropped: %s" % para.replace("\n", "\\n"))
			fails += 1

	# numerals must actually reach the synthesizer as phonemes, not just as text
	for sentence in Phonemes_.parse("I counted 42 of them."):
		for w in sentence:
			if (w.phones as Array).is_empty():
				print("norm_check: FAIL  empty phones for word '%s'" % w.text)
				fails += 1

	for c in MACRO_CASES:
		var got_m: String = TextNorm_.normalize(String(c[0]))
		if not got_m.contains(String(c[1])):
			print("norm_check: FAIL  %-52s -> %s   (wanted %s)" % [c[0], got_m, c[1]])
			fails += 1
		# THE HALF THAT MATTERS. Reading the macro's own text aloud is the defect;
		# a pass that only checked the default arrived would miss a substitution
		# that appended rather than replaced.
		if got_m.contains(String(c[2])):
			print("norm_check: FAIL  the macro text was left to be read: %s -> %s"
				% [c[0], got_m])
			fails += 1
		# ...and nothing may reach the voice as an empty phone array either
		for sentence in Phonemes_.parse(String(c[0])):
			for w in sentence:
				if (w.phones as Array).is_empty():
					print("norm_check: FAIL  empty phones for '%s' in %s" % [w.text, c[0]])
					fails += 1

	# unresolved_macros is what the panel asks before it offers to narrate, and it
	# must name exactly the macros _expand_macros could not use - no default, or
	# malformed. A panel reporting a different set from the one that was dropped
	# is worse than a panel reporting nothing.
	var bare := TextNorm_.unresolved_macros(
		"${A:1} and ${B} and ${C:} and ${D} again.")
	if Array(bare) != ["B", "D"]:
		print("norm_check: FAIL  unresolved_macros returned %s (wanted [B, D])" % [bare])
		fails += 1
	var broke := TextNorm_.unresolved_macros("ok ${A:{nested} fine} then ${B:runaway")
	if broke.size() != 1 or not String(broke[0]).contains("never closed"):
		print("norm_check: FAIL  an unclosed macro was not reported: %s" % [broke])
		fails += 1
	# a long, multi-paragraph default is USABLE and must not be reported at all
	if not TextNorm_.unresolved_macros("${BODY:One.\n\nTwo, at length.}").is_empty():
		print("norm_check: FAIL  a paragraph-length default was reported as unusable")
		fails += 1

	# THE PAGE AND THE MOUTH SAY DIFFERENT THINGS, and both must survive.
	#
	# Reported: a sentence ending "...who left the building in 2009." was
	# SUBTITLED "...in two thousand nine." Normalization is for the phoneme
	# lookup and never for the reader's eyes - Phonemes.parse has said so beside
	# `display` since it was written - but `display` was read off the text AFTER
	# this file had rewritten it, by which point the numeral existed nowhere.
	# Every case here is a rewrite that must be HEARD one way and SHOWN another.
	# THE ASSERTION IS EQUALITY, not "contains the numeral". A weaker test passes
	# while a span measured on the wrong word leaves `cetera` behind it, which is
	# how this very check first went green on a broken abbreviation span. What
	# the reader sees must be the sentence they wrote, word for word.
	for c in [["He left in 2009.", "two thousand nine"],
			["She paid $5.", "five dollars"],
			["The 1st of May.", "first"],
			["Dr. Smith left.", "doctor"],
			["It cost 1,200 pounds.", "one thousand two hundred"],
			# an abbreviation that expands to TWO words: the span has to cover
			# both, or `cetera` is left on screen reading as itself
			["Bring rope, etc.", "et cetera"]]:
		var shown := ""
		var spoken := ""
		for sentence in Phonemes_.parse(String(c[0])):
			for w in sentence:
				spoken += String(w.text) + " "
				var d := String(w.get("display", ""))
				if not d.is_empty():
					shown += d + " "
		if shown.strip_edges() != String(c[0]):
			print("norm_check: FAIL  the page says %s and the subtitle says '%s'"
				% [c[0], shown.strip_edges()])
			fails += 1
		if not spoken.contains(String(c[1])):
			print("norm_check: FAIL  %-24s spoken as '%s' (wanted %s)" % [c[0], spoken, c[1]])
			fails += 1

	# THE RUN IS GROUPED, so a subtitle can hold the numeral up for as long as it
	# takes to say - the words after the first carry the group and nothing to draw.
	var grouped := 0
	var blanks := 0
	for sentence in Phonemes_.parse("in 2009."):
		for w in sentence:
			if int(w.get("src_span", -1)) >= 0:
				grouped += 1
				if String(w.get("display", "")).is_empty():
					blanks += 1
	if grouped != 3 or blanks != 2:
		print("norm_check: FAIL  '2009' grouped %d word(s), %d of them blank (wanted 3 and 2)"
			% [grouped, blanks])
		fails += 1
	# a word nobody rewrote is its own source and belongs to no group
	for sentence in Phonemes_.parse("plain words only."):
		for w in sentence:
			if int(w.get("src_span", -1)) != -1:
				print("norm_check: FAIL  '%s' was grouped and should not be" % w.text)
				fails += 1

	for c in PAUSE_CASES:
		var found := false
		for sentence in Phonemes_.parse(String(c[0])):
			for w in sentence:
				if String(w.text).to_lower().begins_with(String(c[1])):
					found = true
					if String(w.pause_after) != String(c[2]):
						print("norm_check: FAIL  '%s' in %s -> pause '%s' (wanted '%s')"
							% [c[1], c[0], w.pause_after, c[2]])
						fails += 1
		if not found:
			print("norm_check: FAIL  word '%s' missing entirely from %s" % [c[1], c[0]])
			fails += 1

	# round numbers, spot-checked against how a reader says them
	var nums := {0: "zero", 13: "thirteen", 21: "twenty one", 100: "one hundred",
		115: "one hundred fifteen", 1000: "one thousand",
		1000000: "one million", 2500: "two thousand five hundred"}
	# Years, including the one every rule-based normalizer gets wrong. 2000 read as "twenty
	# hundred" here until the 2000s case was moved above the round-century case.
	var years := {1985: "nineteen eighty five", 1900: "nineteen hundred",
		1905: "nineteen oh five", 2000: "two thousand", 2009: "two thousand nine",
		2019: "twenty nineteen", 2100: "twenty one hundred"}
	for y in years:
		if TextNorm_.year(y) != String(years[y]):
			print("norm_check: FAIL  year(%d) = '%s' (wanted '%s')"
				% [y, TextNorm_.year(y), years[y]])
			fails += 1
	for n in nums:
		if TextNorm_.cardinal(n) != String(nums[n]):
			print("norm_check: FAIL  cardinal(%d) = '%s' (wanted '%s')"
				% [n, TextNorm_.cardinal(n), nums[n]])
			fails += 1

	if fails == 0:
		print("norm_check: ALL OK (%d text cases x2 for the newline, %d macro cases, %d pause "
			% [CASES.size(), MACRO_CASES.size(), PAUSE_CASES.size()]
			+ "cases, %d numbers, %d years, %d paragraphs checked for drops)"
			% [nums.size(), years.size(), paras.size()])
	else:
		print("norm_check: %d FAILURES" % fails)
	quit(1 if fails > 0 else 0)
