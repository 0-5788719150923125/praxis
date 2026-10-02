extends Node

## NOT a gate. Drives a real tablet-medium session over a chapter with a SYNTHETIC voice - every
## spoken word at a steady pace, a rest at each sentence end, and the action rests
## [method TabletScript.speakable] would ask the voice for - and writes PNGs to look at.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/tablet_look_probe.gd 400 \
##       --out /tmp/tab/t --doc <chapter.md> --times 3,8,20 [--screen 1] [--image <png>]
##
## `--screen 1` also writes the flat screen texture beside each frame. `--every S` photographs
## every S seconds to the end of the reading instead of `--times`. It asserts only that a
## frame is not uniform.

const W := 1280
const H := 720
const DT := 1.0 / 30.0

var _out := "user://tablet"
var _doc := "/home/crow/repos/rift/books/north-star/chapters/42-the-rabbit-hole.md"
var _times: Array = [3.0, 10.0]
var _word := 0.3
var _screen := false
var _every := 0.0
var _image := ""
var _flat := 0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if i + 1 >= args.size():
			break
		match args[i]:
			"--out": _out = args[i + 1]
			"--doc": _doc = args[i + 1]
			"--word": _word = float(args[i + 1])
			"--screen": _screen = args[i + 1] == "1"
			"--every": _every = float(args[i + 1])
			"--image": _image = args[i + 1]
			"--camera": Director.camera = float(args[i + 1])
			"--times":
				_times = []
				for s in String(args[i + 1]).split(","):
					_times.append(float(s))
	var body := FileAccess.get_file_as_string(_doc)
	# QUIT, never hang: a script error leaves a boot probe running until its timeout, holding
	# tests/boot_probe.gd - and any probe started meanwhile clobbers it
	if (TabletScript.parse(body)["spoken"] as PackedInt32Array).is_empty():
		print("tablet_look_probe: no spoken words in '%s' - nothing to drive" % _doc)
		get_tree().quit(2)
		return
	var title := BookLayout.field_of(body, "title")
	if not _image.is_empty():
		var idx := {}
		for im in Manuscript.images(body):
			idx[String(im["key"])] = {"versions": [{"file": _image, "sig": ""}], "current": 0}
		Illustrations.use_for_test({"index": idx}, true)
	var stage := SubViewport.new()
	stage.size = Vector2i(W, H)
	stage.own_world_3d = true
	stage.transparent_bg = false
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)
	Director.detach()
	var medium: Medium = Medium.make("tablet")
	medium.mount(stage)
	Director.attach(stage, medium)
	Director.hold(true)
	var subs: Subtitles = preload("res://scripts/subtitles.gd").new()
	subs.words = _timeline(body, title)
	subs.document = {"source": body, "title": title}
	add_child(subs)
	if medium.bind_captions(subs):
		subs.overlay_hidden = true
	var end := float((subs.words.back() as Dictionary)["t1"]) + 3.0
	print("tablet_look_probe: %d synthetic words, %.0fs" % [subs.words.size(), end])
	if _every > 0.0:
		_times = []
		var tt := _every
		while tt < end:
			_times.append(tt)
			tt += _every
	var tab := medium as TabletMedium
	var t := 0.0
	Spectrum.virtual_clock = 0.0
	_times.sort()
	for want in _times:
		while t < float(want):
			t += DT
			Spectrum.virtual_clock = t
			Spectrum.current.time = t
			medium.advance(Spectrum.current, DT, 1.0)
			await get_tree().process_frame
		await get_tree().process_frame
		var img := stage.get_texture().get_image()
		var path := "%s_t%05.1f.png" % [_out, float(want)]
		img.save_png(path)
		if _screen:
			tab._vp.get_texture().get_image().save_png("%s_t%05.1f_screen.png" % [_out, float(want)])
		var st := _spread(img)
		if st < 0.02:
			_flat += 1
		print("tablet_look_probe: t=%.1f %s (spread %.3f)" % [want, tab.debug_line(), st])
	print("tablet_look_probe: done, %d flat" % _flat)
	Spectrum.virtual_clock = -1.0
	Director.hold(false)
	Director.detach()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(1 if _flat > 0 else 0)


## Every spoken word at a steady pace, a rest at each sentence end, and each run of actions'
## rest after the word before it - the shape a real take's sidecar has.
func _timeline(body: String, title: String) -> Array:
	var d := TabletScript.parse(body)
	var holds := {}
	for a in d["actions"]:
		var n := int(a["after"])
		holds[n] = float(holds.get(n, 0.0)) + float(a["dur"])
	var out: Array = []
	var t := 2.0
	var si := 0
	var spoken: PackedInt32Array = d["spoken"]
	for k in spoken.size():
		t += float(holds.get(k, 0.0))
		var text := String((d["words"][spoken[k]] as Dictionary)["text"])
		out.append({"text": text, "t0": t, "t1": t + _word * 0.85, "sentence": si, "emph": 0})
		t += _word
		var tail := text.rstrip("\"')*_")
		if tail.ends_with(".") or tail.ends_with("?") or tail.ends_with("!"):
			si += 1
			t += 0.45
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
