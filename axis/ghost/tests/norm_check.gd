extends SceneTree

## norm_check - the gate for [TextNorm], and for the punctuation ordering fix
## in [Phonemes.parse].
##
## Both defects this covers were SILENT: a numeral produced an empty phone array
## and vanished from the utterance with nothing in the audio or the subtitles to
## show it, and a curly apostrophe turned a dictionary hit into a miss that fell
## through to letter rules and merely sounded a bit wrong. Neither would ever
## have failed a test that only checked "does it render".
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
	["Mr. Blake arrived.", "mister"],
	["Dr. Vane left.", "doctor"],
	["St. Anne's church.", "saint"],
]

# words whose terminal punctuation must survive a wrapping quote, which is the
# ordering bug in parse(): the strip loop met the quote first and gave up, so
# every line of dialogue lost its contour and its pause
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

	# numerals must actually reach the synthesizer as phonemes, not just as text
	for sentence in Phonemes_.parse("I counted 42 of them."):
		for w in sentence:
			if (w.phones as Array).is_empty():
				print("norm_check: FAIL  empty phones for word '%s'" % w.text)
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
	for n in nums:
		if TextNorm_.cardinal(n) != String(nums[n]):
			print("norm_check: FAIL  cardinal(%d) = '%s' (wanted '%s')"
				% [n, TextNorm_.cardinal(n), nums[n]])
			fails += 1

	if fails == 0:
		print("norm_check: ALL OK (%d text cases, %d pause cases, %d numbers)"
			% [CASES.size(), PAUSE_CASES.size(), nums.size()])
	else:
		print("norm_check: %d FAILURES" % fails)
	quit(1 if fails > 0 else 0)
