extends SceneTree

## Show the sentence stress [Phrasing] assigns: every word with its prominence,
## nuclei marked. This is the inspection surface for the stage - if a reading
## leans on the wrong word, this says so in one line instead of by ear.
##
## Run: godot --headless --path axis/ghost --script tests/phrasing_dump.gd -- "text"

const Phonemes_ := preload("res://scripts/phonemes.gd")
const Phrasing_ := preload("res://scripts/phrasing.gd")


func _init() -> void:
	var text := "In the beginning was the couch, and the couch was with Father. The far end is yours. It has been yours for so long that the cushion has surrendered."
	var args := OS.get_cmdline_user_args()
	if args.size() > 0:
		text = " ".join(args)
	var sentences := Phonemes_.parse(text)
	Phrasing_.annotate(sentences)
	for words in sentences:
		var line := ""
		for w in words:
			var p := float(w.get("prominence", 0.0))
			var t := String(w.get("display", w.get("text", "")))
			if bool(w.get("nuclear", false)):
				line += "[%s]%.2f " % [t.to_upper(), p]
			elif p >= 0.5:
				line += "%s(%.2f) " % [t.to_upper(), p]
			else:
				line += "%s(%.2f) " % [t.to_lower(), p]
		print(line)
	print("\n  UPPER = accented, [BRACKETS] = nuclear accent of its phrase, lower = reduced")
	quit()
