extends Node

## NOT a gate. Drives the tarot table over an episode on disk with a SYNTHETIC voice - every
## spoken word at a steady pace, and the rest each action asks the voice for - and writes PNGs
## to look at.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/tarot_look_probe.gd 300 \
##       --out /tmp/tarot/t --show truthful-tarot --seed 1 --times 2,9,30 [--every S] [--camera 1]
##
## `--marks 1` photographs each action instead: the moment it starts, a beat in, and the card
## held up after it. Asserts only that a frame is not uniform.

const W := 1280
const H := 720
const DT := 1.0 / 30.0

var _out := "user://tarot_look"
var _show := "truthful-tarot"
var _seed := 1
var _times: Array = [2.0, 9.0]
var _word := 0.33
var _every := 0.0
var _marks := false
var _flat := 0
var _glow := -1.0       # --glow T: the bloom's HDR threshold, to see whether it fires at all
var _from := -1         # --from N: start the reading at spoken word N, as a scrub does
var _foil := -1.0       # --foil F: force the deck's foil amount


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if i + 1 >= args.size():
			break
		match args[i]:
			"--out": _out = args[i + 1]
			"--show": _show = args[i + 1]
			"--seed": _seed = int(args[i + 1])
			"--word": _word = float(args[i + 1])
			"--every": _every = float(args[i + 1])
			"--marks": _marks = args[i + 1] == "1"
			"--camera": Director.camera = float(args[i + 1])
			"--glow": _glow = float(args[i + 1])
			"--from": _from = int(args[i + 1])
			"--foil": _foil = float(args[i + 1])
			"--times":
				_times = []
				for s in String(args[i + 1]).split(","):
					_times.append(float(s))
	var ep := TarotEpisode.open(_show, _seed)
	var script := ep.script()
	if script.is_empty():
		print("tarot_look_probe: episode %s #%d has no script yet (%s)" % [_show, _seed, ep.dir])
		get_tree().quit(2)
		return
	var stage := SubViewport.new()
	stage.size = Vector2i(W, H)
	stage.own_world_3d = true
	stage.transparent_bg = false
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)
	Director.detach()
	var medium: Medium = Medium.make("tarot")
	medium.mount(stage)
	Director.attach(stage, medium)
	Director.hold(true)
	var subs: Subtitles = preload("res://scripts/subtitles.gd").new()
	var parse := TarotScript.parse(script)
	subs.words = timeline(parse, _word, Director.intro_hold)
	subs.document = {"source": script, "title": "Truthful Tarot", "tarot": ep.document()}
	if _from > 0:
		# A SCRUB: the voice starts at word N with no intro, and the medium is handed its first words
		var later: Array = []
		var t0 := float((subs.words[_from] as Dictionary)["t0"]) - 0.5
		for i in range(_from, subs.words.size()):
			var w: Dictionary = (subs.words[i] as Dictionary).duplicate()
			w["t0"] = float(w["t0"]) - t0
			w["t1"] = float(w["t1"]) - t0
			later.append(w)
		subs.words = later
		var sw := PackedStringArray()
		for i in range(_from, mini(_from + 6, (parse["spoken"] as PackedStringArray).size())):
			sw.append((parse["spoken"] as PackedStringArray)[i])
		subs.document["start_words"] = sw
		subs.document["start_index"] = _from
		print("tarot_look_probe: a scrub to word %d, '%s'" % [_from, " ".join(sw)])
	add_child(subs)
	medium.bind_captions(subs)
	if _glow >= 0.0:
		(medium as TarotMedium)._env.glow_hdr_threshold = _glow
	if _foil >= 0.0:
		(medium as TarotMedium)._foil = _foil
	var end := float((subs.words.back() as Dictionary)["t1"]) + 4.0
	print("tarot_look_probe: %d synthetic words, %.0fs, %d actions" % [subs.words.size(), end,
		(parse["actions"] as Array).size()])
	if _every > 0.0:
		_times = []
		var tt := _every
		while tt < end:
			_times.append(tt)
			tt += _every
	if _marks:
		_times = [1.5]
		var at := _action_times(parse, subs.words, Director.intro_hold)
		for a in at:
			for d in [0.2, 1.4, 2.4, 4.5]:
				_times.append(float(a) + d)
		_times.append(end - 2.0)
	_times.sort()
	var t := 0.0
	Spectrum.virtual_clock = 0.0
	var step := 0
	for want in _times:
		while t < float(want):
			t += DT
			Spectrum.virtual_clock = t
			Spectrum.current.time = t
			subs._process(DT)
			medium.advance(Spectrum.current, DT, 1.0)
			# a real frame only now and then, and for the last few before a photograph - the
			# cards' faces are drawn by viewports that need frames of their own to draw in
			step += 1
			if step % 20 == 0 or float(want) - t < 0.2:
				await get_tree().process_frame
		for _i in 3:
			await get_tree().process_frame
		var img := stage.get_texture().get_image()
		if want == _times[0]:
			var tmed := medium as TarotMedium
			var cur := stage.get_camera_3d()
			print("tarot_look_probe: camera is the table's: %s; env glow %s threshold %.2f; foil %.2f" % [
				str(cur == tmed._cam), str(tmed._cam.environment.glow_enabled if tmed._cam.environment else false),
				tmed._env.glow_hdr_threshold, tmed._foil])
		var path := "%s_t%05.1f.png" % [_out, float(want)]
		img.save_png(path)
		var st := _spread(img)
		if st < 0.02:
			_flat += 1
		print("tarot_look_probe: t=%.1f -> %s (spread %.3f)" % [want, path, st])
	print("tarot_look_probe: done, %d flat" % _flat)
	Spectrum.virtual_clock = -1.0
	Director.hold(false)
	Director.detach()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(1 if _flat > 0 else 0)


## Words at a steady pace from the intro on, a rest at each sentence end, and each action's own
## rest (plus a beat either side) where the voice would take it.
static func timeline(parse: Dictionary, word: float, intro: float) -> Array:
	var out: Array = []
	var rests := {}
	for a in parse["actions"]:
		var k := int((a as Dictionary)["after"])
		rests[k] = float(rests.get(k, 0.0)) + float((a as Dictionary)["dur"]) + 0.5
	var t := intro
	var sentence := 0
	var spoken: PackedStringArray = parse["spoken"]
	for i in spoken.size():
		if rests.has(i):
			t += float(rests[i])
			sentence += 1
		out.append({"text": spoken[i], "t0": t, "t1": t + word * 0.85, "sentence": sentence})
		t += word
	return out


## When each action begins in that timeline (roughly - the medium places them itself).
static func _action_times(parse: Dictionary, words: Array, intro: float) -> Array:
	var out: Array = []
	for a in parse["actions"]:
		var k := int((a as Dictionary)["after"])
		var at := intro if k <= 0 else float((words[k - 1] as Dictionary)["t1"])
		out.append(at + 0.2)
	return out


func _spread(img: Image) -> float:
	var lo := 1.0
	var hi := 0.0
	for y in range(0, img.get_height(), 24):
		for x in range(0, img.get_width(), 24):
			var l := img.get_pixel(x, y).get_luminance()
			lo = minf(lo, l)
			hi = maxf(hi, l)
	return hi - lo
