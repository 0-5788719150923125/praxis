extends SceneTree

## Gate for the letter-to-sound front end, graded against `data/reference.yml`.
##
## Two halves, because they fail differently:
##   PHONES  - four groups (dictionary hits, suffix/clitic derivations, the
##             out-of-dictionary fallback, and our deliberate overrides). Since
##             CMUdict answers 126k words, the interesting groups are the ones
##             it does NOT answer; a hand list of dictionary words would only
##             prove the file loaded.
##   STRESS  - the index of the primary-stressed vowel. Nothing else in the
##             suite tests this, and it is now what the whole rhythm hangs off:
##             a wrong mark is a word leaned on in the wrong place, which is
##             audible in a way a wrong schwa is not. -1 asserts an unstressed
##             function word.
##
## Run: godot --headless --path axis/ghost --script tests/g2p_check.gd

const Phonemes_ := preload("res://scripts/phonemes.gd")

const REF_PATH := "res://data/reference.yml"
const GROUPS := ["words", "derived", "fallback"]


func _init() -> void:
	var f := FileAccess.open(REF_PATH, FileAccess.READ)
	if f == null:
		print("g2p_check: cannot open ", REF_PATH)
		quit(1)
		return
	var r := MiniYaml.parse(f.get_as_text())
	f.close()
	if not r.ok:
		print("g2p_check: reference parse error - ", r.error)
		quit(1)
		return
	var ref: Dictionary = r.data
	var failures := 0
	for g in GROUPS:
		var group: Dictionary = ref.get(g, {})
		var bad := 0
		for word in group:
			var want: Array = _split(String(group[word]))
			var got: Array = Phonemes_.word_to_phones(String(word))
			if want != got:
				print("  %-9s %-14s want: %-24s got: %s" % [
					g, word, " ".join(PackedStringArray(want)), " ".join(PackedStringArray(got))])
				bad += 1
		print("g2p_check: %-9s %d/%d" % [g, group.size() - bad, group.size()])
		failures += bad
	var st: Dictionary = ref.get("stress", {})
	var sbad := 0
	for word in st:
		var want := int(st[word])
		var got := _primary(String(word))
		if want != got:
			print("  %-9s %-14s want primary-stressed vowel %d, got %d" % ["stress", word, want, got])
			sbad += 1
	print("g2p_check: %-9s %d/%d" % ["stress", st.size() - sbad, st.size()])
	failures += sbad
	if failures > 0:
		print("g2p_check: %d FAILURE(S)" % failures)
		quit(1)
		return
	print("g2p_check: ALL OK")
	quit()


## Index of the primary-stressed vowel, counting vowels only. -1 if none.
func _primary(word: String) -> int:
	var got := Phonemes_.lookup(word)
	var vi := 0
	for i in (got.phones as Array).size():
		if Phonemes_.TABLE.get(got.phones[i], {}).get("type", "") != "vowel":
			continue
		if int(got.stress[i]) == 1:
			return vi
		vi += 1
	return -1


func _split(spec: String) -> Array:
	var out: Array = []
	for p in spec.split(" ", false):
		out.append(String(p).to_upper())
	return out
