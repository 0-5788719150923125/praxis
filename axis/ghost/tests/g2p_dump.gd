extends SceneTree

## Letter-to-sound diagnostic: every distinct word of a text with the phonemes
## the front end produces for it, ordered by corpus frequency so the errors that
## matter most are at the top. No synthesis, so it runs on a whole book in
## under a second.
##
## Run: godot --headless --path axis/ghost --script tests/g2p_dump.gd -- <file> [limit]

const Phonemes_ := preload("res://scripts/phonemes.gd")


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	if args.is_empty():
		print("usage: g2p_dump.gd -- <textfile> [limit]")
		quit(1)
		return
	var f := FileAccess.open(args[0], FileAccess.READ)
	if f == null:
		print("cannot open ", args[0])
		quit(1)
		return
	var limit := int(args[1]) if args.size() > 1 else 400
	var text := f.get_as_text()
	f.close()
	var freq := {}
	for sentence in Phonemes_.parse(text):
		for w in sentence:
			var key := String(w.text)
			if key.is_empty():
				continue
			if not freq.has(key):
				freq[key] = {"n": 0, "phones": w.phones}
			freq[key].n += 1
	var words: Array = freq.keys()
	words.sort_custom(func(a, b): return freq[a].n > freq[b].n)
	var total := 0
	for k in words:
		total += int(freq[k].n)
	print("%d tokens, %d types" % [total, words.size()])
	var shown := 0
	for k in words:
		if shown >= limit:
			break
		print("%5d  %-16s %s" % [freq[k].n, k, " ".join(PackedStringArray(freq[k].phones))])
		shown += 1
	quit()
