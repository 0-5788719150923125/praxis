extends SceneTree

## Render the intelligibility fixture set: text in, WAV + timing JSON out.
##
## This is deliberately dumb. It owns no metrics and makes no judgements - it
## exists so the Python analyzer (axis/ghost/measure_voice.py) has real audio
## with a real alignment to measure. Every acoustic threshold lives there,
## where numpy can do FFTs and LPC; GDScript's job is to be the synthesizer.
##
## The alignment is exact by construction: we synthesized it, so `phones` and
## `words` are ground truth, not a forced alignment. That is what makes the
## vowel-space and boundary measurements possible at all.
##
## Run:  godot --headless --path axis/ghost --script tests/render_fixtures.gd
##       [-- <out_dir>]
## Default out_dir is build/voice (gitignored).

const Voice_ := preload("res://scripts/voice.gd")

const DEFAULT_OUT := "res://build/voice"

# The fixtures. Fixed text and a fixed lineage, so a run is comparable to the
# run before it - that is the entire point of a gate.
#
# `narration` is the primary fixture: ordinary novel prose, the actual target
# use. `consonants` loads the inventory deliberately - every English fricative,
# stop and nasal in both onset and coda, which is what makes the per-class
# level table meaningful. `pairs` is the minimal-pair battery: the contrasts
# that carry the diagnosis (stop place, stop voicing, /f/-/th/, sibilants,
# nasal place, and unstressed /r/, which the F3 reduction was destroying).
const FIXTURES := {
	"narration": "The lamp had gone out sometime in the night, and the room was cold. She sat up slowly, listening for the sound that had woken her, but the house gave back nothing except the settling of old timber. Somewhere below, a door closed. She counted to thirty, then reached for the candle on the table beside the bed, and found that her hands were not quite steady.",
	"consonants": "Six thin fish swam past the sunken ship. The father gathered leather and feathers. Peter picked a basket of ripe berries. Take the cake, bake the dough, and go. Some men sing songs in the morning. A vision of pleasure, measured and treasured. Which choice did the judge chase?",
	"pairs": "Bat pat mat cat. Din tin sin thin. Pin bin. Tin din. Kin gin. Fin thin sin shin. Sum sun sung. Rip lip. Bitter better butter. The water in the winter is bitter.",
}

# Zero trait vector = the hand-curated default speaker (Spec.from_traits centres
# everything). Lineage [1] is the root seed. Both fixed: a gate that moves with
# the seed measures the seed, not the engine.
const LINEAGE := [1]


func _init() -> void:
	var args := OS.get_cmdline_user_args()
	var out_dir: String = args[0] if args.size() > 0 else DEFAULT_OUT
	DirAccess.make_dir_recursive_absolute(out_dir)

	var manifest: Array = []
	for name in FIXTURES:
		var text: String = FIXTURES[name]
		var spec: Variant = Voice_.Spec.from_traits({}, int(LINEAGE[0]), LINEAGE)
		var t0 := Time.get_ticks_msec()
		var result: Dictionary = Voice_.render(text, spec)
		var ms := Time.get_ticks_msec() - t0

		var wav_path: String = out_dir + "/" + name + ".wav"
		var abs_wav: String = Voice_.write_wav(wav_path, result.pcm)
		var side := FileAccess.open(out_dir + "/" + name + ".json", FileAccess.WRITE)
		side.store_string(JSON.stringify({
			"name": name,
			"text": text,
			"sr": result.sr,
			"dur": result.dur,
			"words": result.words,
			"phones": result.phones,
		}))
		side.close()

		var rt: float = (float(ms) / 1000.0) / maxf(result.dur, 0.001)
		print("render_fixtures: %-11s %6.2f s  synth %5.2f s (%.1fx RT)  -> %s"
			% [name, result.dur, float(ms) / 1000.0, 1.0 / maxf(rt, 0.0001), abs_wav])
		manifest.append({"name": name, "wav": abs_wav, "dur": result.dur})

	var mf := FileAccess.open(out_dir + "/manifest.json", FileAccess.WRITE)
	mf.store_string(JSON.stringify({
		"fixtures": manifest,
		"noise_fx": Voice_.NOISE_FX,
		"drone": Voice_.DRONE,
		"raw_mode": Voice_.RAW_MODE,
		"fric_level": Voice_.FRIC_LEVEL,
		"out_gain": Voice_.OUT_GAIN,
	}))
	mf.close()
	print("render_fixtures: done")
	quit(0)
