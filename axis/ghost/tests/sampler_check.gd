extends SceneTree

## Headless check for the voice sampler's MEASUREMENT half (the mic half
## needs a real device): the round trip. Synthesize takes with KNOWN traits,
## run VoiceSampler.analyze on the raw PCM, and assert the recovered traits
## point the right way with meaningful magnitude - if analysis cannot
## recover the synthesizer's own voices, it cannot echo a living one.
##
## Run: godot --headless --path axis/ghost --script tests/sampler_check.gd

const Voice_ := preload("res://scripts/voice.gd")
const Sampler_ := preload("res://scripts/voice_sampler.gd")

const TEXT := ("Once upon a time, the river spoke softly to the stones. "
	+ "Did the water remember every voice it carried? "
	+ "I think it kept them all somewhere below the surface waiting "
	+ "for someone patient enough to listen. "
	+ "Stones and static, signal and silence - the keeper plays them back.")


func _init() -> void:
	var failures := 0
	# the resonance drone rings through pauses by design - silence it here so
	# the round-trip measures the SPEECH (the mic path measures humans, who
	# bring no drone)
	Voice_.DRONE = false
	var hi := Voice_.render(TEXT, Voice_.Spec.from_traits({"pitch": 0.6, "pace": 0.45}))
	var lo := Voice_.render(TEXT, Voice_.Spec.from_traits({"pitch": -0.6, "pace": -0.4}))
	var a: Dictionary = Sampler_.analyze(hi.pcm, float(Voice_.SR))
	var b: Dictionary = Sampler_.analyze(lo.pcm, float(Voice_.SR))
	if a.is_empty() or b.is_empty():
		print("sampler_check: FAIL - analysis returned empty on a clean synthesized take")
		quit(1)
		return
	print("sampler_check: hi voice -> ", a.report)
	print("sampler_check:   traits ", a.traits)
	print("sampler_check: lo voice -> ", b.report)
	print("sampler_check:   traits ", b.traits)
	# pitch: +-0.6 apart at synthesis must recover clearly separated, same order
	var dp: float = float(a.traits.pitch) - float(b.traits.pitch)
	if dp < 0.5:
		print("sampler_check: FAIL - pitch separation %.2f (want > 0.5)" % dp)
		failures += 1
	if absf(float(a.traits.pitch) - 0.6) > 0.35:
		print("sampler_check: FAIL - hi pitch recovered %.2f, expected near 0.6" % float(a.traits.pitch))
		failures += 1
	# pace: direction must hold (magnitude is looser - the walk modulates tempo)
	if float(a.traits.pace) <= float(b.traits.pace):
		print("sampler_check: FAIL - pace direction not recovered (%.2f vs %.2f)" % [
			float(a.traits.pace), float(b.traits.pace)])
		failures += 1
	# the genome must land inside the walk's own gene bounds ("anchors" is
	# the reserved melodic-modes key, not a gene)
	for key in a.genome:
		if key == "anchors":
			continue
		if not Voice_.ProsodyWalk.G_BOUNDS.has(key):
			print("sampler_check: FAIL - measured genome key '%s' is not a walk gene" % key)
			failures += 1
			continue
		var lohi: Array = Voice_.ProsodyWalk.G_BOUNDS[key]
		var v := float(a.genome[key])
		if v < float(lohi[0]) - 0.0001 or v > float(lohi[1]) + 0.0001:
			print("sampler_check: FAIL - genome %s=%.2f outside G_BOUNDS %s" % [key, v, str(lohi)])
			failures += 1
	# the melodic modes: present, sane, and honored by the walk
	var anchors: Array = a.genome.get("anchors", [])
	if anchors.is_empty() or anchors.size() > 5:
		print("sampler_check: FAIL - expected 1-5 measured anchors, got %d" % anchors.size())
		failures += 1
	for an in anchors:
		if absf(float(an)) > 14.0:
			print("sampler_check: FAIL - anchor %.1f st out of range" % float(an))
			failures += 1
	var walk := Voice_.ProsodyWalk.new([[7]], a.genome)
	if walk._anchors.size() != anchors.size() + 1:
		print("sampler_check: FAIL - the walk did not adopt the measured anchors (%d vs %d+1)" % [
			walk._anchors.size(), anchors.size()])
		failures += 1
	# formant tracking: voices synthesized with tract apart must recover
	# tract apart, same order (the vocal-tract length inversion)
	var wide := Voice_.render(TEXT, Voice_.Spec.from_traits({"tract": 0.7}))
	var narrow := Voice_.render(TEXT, Voice_.Spec.from_traits({"tract": -0.7}))
	var aw: Dictionary = Sampler_.analyze(wide.pcm, float(Voice_.SR))
	var an2: Dictionary = Sampler_.analyze(narrow.pcm, float(Voice_.SR))
	print("sampler_check: tract +0.7 -> %.2f (%s) | tract -0.7 -> %.2f (%s)" % [
		float(aw.traits.tract), aw.report, float(an2.traits.tract), an2.report])
	if float(aw.traits.tract) <= float(an2.traits.tract):
		print("sampler_check: FAIL - tract direction not recovered")
		failures += 1
	elif float(aw.traits.tract) - float(an2.traits.tract) < 0.8:
		print("sampler_check: FAIL - tract separation %.2f too weak (want > 0.8 for a 1.4 truth)"
			% (float(aw.traits.tract) - float(an2.traits.tract)))
		failures += 1
	# determinism: the same recording measures the same voice
	var a2: Dictionary = Sampler_.analyze(hi.pcm, float(Voice_.SR))
	if str(a2.traits) != str(a.traits) or str(a2.genome) != str(a.genome):
		print("sampler_check: FAIL - analysis is not deterministic")
		failures += 1
	# and the minted spec must be renderable: a smoke synthesis from the echo
	var spec := Voice_.Spec.from_traits(a.traits, 424242, [424242])
	spec.adrenochrome = a.genome.duplicate()
	var take := Voice_.render("the echo speaks with a borrowed shape", spec)
	if take.pcm.size() < Voice_.SR:
		print("sampler_check: FAIL - the echoed spec rendered almost nothing")
		failures += 1
	if failures == 0:
		print("sampler_check: ALL OK")
	else:
		print("sampler_check: %d FAILURE(S)" % failures)
	quit(failures)
