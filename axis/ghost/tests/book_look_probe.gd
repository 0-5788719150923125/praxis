extends Node

## NOT a gate. Drives a real book-medium session over a chapter with a SYNTHETIC word
## timeline and writes PNGs, because "does this read as a novel" is answered by looking.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/book_look_probe.gd 400 \
##       --out /tmp/book/b --doc <chapter.md> --times 2,20,60 --pages 0,1,2,3
##
## `--times` are show seconds to photograph (the session is stepped there continuously, so the
## camera and the leaf got there the way they would live); `--pages` also writes those flat
## page textures, which is the typesetting on its own; `--wpm`-like pacing is `--word S`
## seconds per word. `--camera X` sets Director.camera for the run (not saved). `--medium
## notebook` drives the notebook instead (same machinery); `--sketch <png>` stands in for every
## `<!-- sketch: -->` the way `--image` does for pictures.
##
## It asserts only that a frame is not uniform - a book that failed to project comes out flat.

const W := 1280
const H := 720
const DT := 1.0 / 30.0

var _out := "user://book"
var _doc := "/home/crow/repos/rift/books/north-star/chapters/40-pauses-and-clauses.md"
var _times: Array = [3.0, 30.0]
var _pages: Array = []
var _word := 0.26
var _flat := 0
var _image := ""        # a PNG to stand in for every illustration (via Illustrations' test seam)
var _turns := 0         # photograph the middle of this many leaf turns
var _medium := "book"
var _sketch := ""       # a PNG to stand in for every sketch


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
			"--camera": Director.camera = float(args[i + 1])
			"--image": _image = args[i + 1]
			"--sketch": _sketch = args[i + 1]
			"--medium": _medium = args[i + 1]
			"--turns": _turns = int(args[i + 1])
			"--settle": _settle = int(args[i + 1])
			"--times":
				_times = []
				for s in String(args[i + 1]).split(","):
					_times.append(float(s))
			"--pages":
				_pages = []
				for s in String(args[i + 1]).split(","):
					_pages.append(int(s))
	var body := FileAccess.get_file_as_string(_doc)
	if not _image.is_empty() or not _sketch.is_empty():
		var idx := {}
		for im in Manuscript.images(body):
			var f := _sketch if bool(im.get("sketch", false)) else _image
			if not f.is_empty():
				idx[String(im["key"])] = {"versions": [{"file": f, "sig": ""}], "current": 0}
		Illustrations.use_for_test({"index": idx}, true)
	var stage := SubViewport.new()
	stage.size = Vector2i(W, H)
	stage.own_world_3d = true
	stage.transparent_bg = false
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)
	Director.detach()
	var medium: Medium = Medium.make(_medium)
	medium.mount(stage)
	Director.attach(stage, medium)
	Director.hold(true)

	var subs: Subtitles = preload("res://scripts/subtitles.gd").new()
	subs.words = _timeline(body)
	subs.document = {"source": body}
	add_child(subs)
	if medium.bind_captions(subs):
		subs.overlay_hidden = true
	print("book_look_probe: %d synthetic words, %.0fs" % [subs.words.size(),
		float((subs.words.back() as Dictionary)["t1"]) if not subs.words.is_empty() else 0.0])

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
			var bk := medium as BookMedium
			# THE END OF A TURN, frame by frame: the last frames of the leaf and the first
			# frames after it lies down - where a stale page texture would show as a flash.
			if _settle > 0 and bk._turn_t >= BookMedium.TURN_TIME - 3.0 * DT:
				await get_tree().process_frame
				stage.get_texture().get_image().save_png("%s_end_%02d.png" % [_out, _settle_i])
				_settle_i += 1
			elif _settle > 0 and _was_turning and bk._turn_t < 0.0:
				for f in 4:
					if f > 0:
						t += DT
						Spectrum.virtual_clock = t
						Spectrum.current.time = t
						medium.advance(Spectrum.current, DT, 1.0)
					await get_tree().process_frame
					stage.get_texture().get_image().save_png("%s_end_%02d.png" % [_out, _settle_i])
					_settle_i += 1
				print("book_look_probe: turn ended at t=%.2f, %d frames written" % [t, _settle_i])
				_settle -= 1
			_was_turning = bk._turn_t >= 0.0
			if _turns > 0 and bk._turn_t >= 0.8 and bk._turn_t < 0.8 + DT:
				await get_tree().process_frame
				var ti := stage.get_texture().get_image()
				ti.save_png("%s_turn_t%04d.png" % [_out, int(t)])
				print("book_look_probe: mid-turn at t=%.1f %s" % [t, bk.debug_line()])
				_turns -= 1
		await get_tree().process_frame
		var img := stage.get_texture().get_image()
		var path := "%s_t%04d.png" % [_out, int(want)]
		img.save_png(path)
		var st := _spread(img)
		if st < 0.02:
			_flat += 1
		print("book_look_probe: t=%.1f %s -> %s (spread %.3f)" % [want,
			(medium as BookMedium).debug_line(), path, st])
	for p in _pages:
		var bv := medium as BookMedium
		var c: BookMedium.PageCanvas = bv._canvases[0]
		bv._slot_page[0] = int(p)
		c.page = int(p)
		c.hl = {}
		c.queue_redraw()
		(bv._vps[0] as SubViewport).render_target_update_mode = SubViewport.UPDATE_ONCE
		for _i in 3:
			await get_tree().process_frame
		var pimg := (bv._vps[0] as SubViewport).get_texture().get_image()
		pimg.save_png("%s_page%02d.png" % [_out, int(p)])
		bv._slot_state[0] = ""
	print("book_look_probe: done, %d flat" % _flat)
	Spectrum.virtual_clock = -1.0
	Director.hold(false)
	Director.detach()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(1 if _flat > 0 else 0)


## Every printed word, spoken at a steady pace with a rest at each sentence end - the shape of
## a real take's sidecar, without a voice host.
var _settle := 0
var _settle_i := 0
var _was_turning := false


func _timeline(body: String) -> Array:
	var lay := BookLayout.new()
	lay.build(body, func(_k: String) -> Vector2: return Vector2.ZERO)
	var out: Array = []
	var t := 2.0
	var si := 0
	for w in lay.words:
		var text := String((w as Dictionary)["text"])
		out.append({"text": text, "t0": t, "t1": t + _word * 0.85, "sentence": si, "emph": 0})
		t += _word
		var tail := text.rstrip("\"'”’)*_")
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
