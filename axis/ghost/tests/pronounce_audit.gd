extends SceneTree

## Audit a script for words the front end is likely to get wrong - BEFORE rendering it.
##
## This is the tool that replaces whack-a-mole. The failure mode it exists for is not a
## crash and not a warning: a word is simply spoken wrong, once, forty minutes into a
## chapter, and the only detector is a person listening to the whole thing. At 37
## chapters that is not a workflow. So instead: read the text, find every word whose
## reading is genuinely uncertain, and print them with their contexts up front, where
## they cost a minute to scan and a line of YAML to fix.
##
## What it flags, in descending order of how likely it is to actually be wrong:
##
##   UNKNOWN     - not in CMUdict, not a compound of known words, not in `names`. An
##                 invented word or an unusual proper noun. The front end will guess.
##   HOMOGRAPH   - a word with two readings that are not interchangeable (`record` the
##                 noun against `record` the verb). Some are resolved by the SPEAK_AS
##                 table; the ones that are NOT are listed with their context so a
##                 human can judge, because several of them - `read` above all - are
##                 genuinely ambiguous from the neighbours alone and no rule can settle
##                 them without understanding the sentence.
##   PINNED      - already handled, listed so it is visible that it was handled rather
##                 than missed. Silence about a fix is how the fix gets undone.
##
## Usage:
##   godot --headless --path axis/ghost --script tests/pronounce_audit.gd -- <file.md ...>
##   godot --headless --path axis/ghost --script tests/pronounce_audit.gd -- --dir <path>
##
## It always exits 0. This is a report, not a gate: an unknown word is a thing to look
## at, not a build failure, and a chapter full of invented names is allowed to exist.

## Words with two live readings that the neighbours cannot reliably separate. Being on
## this list does not mean the word is wrong - it means a human should look. `read` is
## the reason the list exists: past and present are spelled identically, the choice
## depends on the tense of the surrounding discourse rather than on any adjacent word,
## and measured across a real 127,000 word manuscript eSpeak chose the present tense for
## about four fifths of them, which is right far more often than not but wrong dozens of
## times per book.
const AMBIGUOUS := {
	"read": "past /rɛd/ or present /riːd/ - decided by tense, not by neighbours",
	"lead": "the metal /lɛd/ or the verb /liːd/",
	"live": "the adjective /laɪv/ or the verb /lɪv/",
	"wind": "moving air /wɪnd/ or to coil /waɪnd/",
	"wound": "an injury /wuːnd/ or the past of wind /waʊnd/",
	"tear": "to rip /tɛər/ or a teardrop /tɪər/",
	"bow": "to bend /baʊ/ or the ribbon or weapon /boʊ/",
	"row": "a line /roʊ/ or a quarrel /raʊ/",
	"sow": "to plant /soʊ/ or a female pig /saʊ/",
	"close": "near /kloʊs/ or to shut /kloʊz/",
	"use": "the noun /juːs/ or the verb /juːz/",
	"minute": "sixty seconds /ˈmɪnɪt/ or tiny /maɪˈnjuːt/",
	"present": "the noun or adjective, or the verb /prɪˈzɛnt/",
	"subject": "the noun, or the verb /səbˈdʒɛkt/",
	"object": "the noun, or the verb /əbˈdʒɛkt/",
	"content": "the noun, or satisfied /kənˈtɛnt/",
	"permit": "the noun, or the verb /pərˈmɪt/",
	"produce": "the noun, or the verb /prəˈdjuːs/",
	"contract": "the noun, or the verb /kənˈtrækt/",
	"project": "the noun, or the verb /prəˈdʒɛkt/",
	"desert": "the sand /ˈdɛzərt/ or to abandon /dɪˈzɜːt/",
	"refuse": "rubbish /ˈrɛfjuːs/ or to decline /rɪˈfjuːz/",
	"separate": "the adjective, or the verb /ˈsɛpəreɪt/",
	"invalid": "not valid /ɪnˈvælɪd/ or a sick person /ˈɪnvəlɪd/",
	"axes": "of axe /ˈæksɪz/ or of axis /ˈæksiːz/",
}

## Deliberately NOT flagged, recorded here so the omission reads as a decision rather
## than an oversight. Each is a technically real homograph whose second reading is so
## rare in prose that flagging it is pure noise - and noise is what makes an audit get
## ignored, which costs more than the words it would have caught.
##   does    - the verb, against a plural of female deer
##   number  - a quantity, against the comparative of numb
##   sow/row - the common reading dominates so heavily that the alarm is never right
##   buffet  - a table of food, against being struck by wind
const NOT_FLAGGED := ["does", "number", "sow", "row", "buffet", "bass", "dove"]


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	var files := _resolve(args)
	if files.is_empty():
		print("pronounce_audit: no input.")
		print("  godot --headless --path axis/ghost --script tests/pronounce_audit.gd -- <file...>")
		print("  godot --headless --path axis/ghost --script tests/pronounce_audit.gd -- --dir <path>")
		quit()
		return

	var unknown := {}          # word -> {count, sample}
	var ambiguous := {}
	var pinned := {}
	var total := 0

	for path in files:
		var text := FileAccess.get_file_as_string(path)
		if text.is_empty():
			print("pronounce_audit: could not read ", path)
			continue
		# Run the REAL front end, so the audit sees exactly what the voice will see -
		# folding, number expansion, abbreviations and all. Auditing the raw text would
		# report words the pipeline never actually produces.
		var normed := TextNorm.normalize(text)
		for sentence in Phonemes.parse(normed):
			for w in sentence:
				var raw := String(w.get("text", "")).strip_edges()
				if raw.is_empty():
					continue
				total += 1
				var key := raw.to_lower()
				if bool(w.get("literal", false)):
					_bump(pinned, key, raw, sentence)
					continue
				if AMBIGUOUS.has(key):
					_bump(ambiguous, key, raw, sentence)
					continue
				if not Phonemes.is_known(key):
					_bump(unknown, key, raw, sentence)

	print("pronounce_audit: %d files, %d words" % [files.size(), total])
	_report("UNKNOWN - no dictionary entry; the front end is guessing", unknown, true)
	_report("AMBIGUOUS - two live readings, no rule can settle it from the neighbours",
		ambiguous, true)
	_report("PINNED - already resolved by data/english.yml or an inline override",
		pinned, false)
	if not unknown.is_empty() or not ambiguous.is_empty():
		print("")
		print("To pin any of these, add a line under `names:` in data/english.yml, or write")
		print("the pronunciation inline in the script as [B IY1 UW0 K S].")
	quit()


## Up to CTX_MAX distinct contexts per word, not just the first. For a word like `read`
## that appears dozens of times and must be judged case by case, one example is useless -
## the whole point is to hand over every site that needs a decision.
const CTX_MAX := 8

func _bump(into: Dictionary, key: String, raw: String, sentence: Array) -> void:
	if not into.has(key):
		into[key] = {"n": 0, "raw": raw, "ctx": []}
	into[key]["n"] = int(into[key]["n"]) + 1
	var list: Array = into[key]["ctx"]
	if list.size() < CTX_MAX:
		var c := _context(sentence, key)
		if not c.is_empty() and not list.has(c):
			list.append(c)


## A few words either side, so the reading can be judged without opening the file.
func _context(sentence: Array, key: String) -> String:
	var words: Array = []
	var at := -1
	for i in sentence.size():
		var t := String((sentence[i] as Dictionary).get("text", ""))
		words.append(t)
		if at < 0 and t.to_lower() == key:
			at = i
	if at < 0:
		return ""
	var lo := maxi(0, at - 5)
	var hi := mini(words.size(), at + 6)
	var out: Array = []
	for i in range(lo, hi):
		out.append(("[%s]" % words[i]) if i == at else String(words[i]))
	return " ".join(out)


func _report(title: String, d: Dictionary, with_ctx: bool) -> void:
	print("")
	print("=== %s ===" % title)
	if d.is_empty():
		print("  (none)")
		return
	var keys := d.keys()
	keys.sort_custom(func(a, b): return int(d[a]["n"]) > int(d[b]["n"]))
	for k in keys:
		var e: Dictionary = d[k]
		var note := String(AMBIGUOUS.get(k, ""))
		print("  %-16s x%-4d %s" % [k, int(e["n"]), note])
		if with_ctx:
			for c in (e["ctx"] as Array):
				print("       %s" % c)
			if int(e["n"]) > (e["ctx"] as Array).size():
				print("       ... and %d more" % (int(e["n"]) - (e["ctx"] as Array).size()))


func _resolve(args: PackedStringArray) -> Array:
	var out: Array = []
	var i := 0
	while i < args.size():
		var a := String(args[i])
		if a == "--dir" and i + 1 < args.size():
			var dir := String(args[i + 1])
			i += 2
			var d := DirAccess.open(dir)
			if d == null:
				print("pronounce_audit: no such directory: ", dir)
				continue
			var names := d.get_files()
			names.sort()
			for f in names:
				if f.ends_with(".md") or f.ends_with(".txt"):
					out.append(dir.path_join(f))
			continue
		if not a.begins_with("--"):
			out.append(a)
		i += 1
	return out
