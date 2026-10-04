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
##
## `--audit 1` writes no pictures: it replays the whole reading and checks, every frame, where
## the word being spoken is - on the browser's viewport, and in the camera's frame - and how
## fast the camera's aim moves, printing each lapse and a summary. `--focus "<phrase>"` adds a
## trace of the seconds around the first place the phrase is read, and `--until S`
## stops at show second S. `--from "<phrase>"` starts the reading there, the way a scrub does:
## the voice from that sentence, no intro, the medium handed its start words.

const W := 1280
const H := 720
const Audit := preload("res://tests/tablet_audit.gd")
const DT := 1.0 / 30.0

var _out := "user://tablet"
var _doc := "/home/crow/repos/rift/books/north-star/chapters/42-what-is-the-7th-realm.md"
var _times: Array = [3.0, 10.0]
var _word := 0.3
var _screen := false
var _every := 0.0
var _image := ""
var _flat := 0
var _intro := -1.0      # --intro: the Director reloads its own at attach, so this is applied after
var _audit := false
var _focus := ""
var _until := INF
var _from := ""


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
			"--intro": _intro = float(args[i + 1])
			"--audit": _audit = args[i + 1] == "1"
			"--focus": _focus = args[i + 1]
			"--until": _until = float(args[i + 1])
			"--from": _from = args[i + 1]
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
	# stand-ins: --image for every picture, and any capture already taken for a real page
	var idx := {}
	if not _image.is_empty():
		for im in Manuscript.images(body):
			idx[String(im["key"])] = {"versions": [{"file": _image, "sig": ""}], "current": 0}
	for sn in TabletScript.snapshots(body):
		var cap := ProjectSettings.globalize_path(PageCapture.DIR.path_join("%s.png" % String(sn["key"])))
		if FileAccess.file_exists(cap):
			idx[String(sn["key"])] = {"versions": [{"file": cap, "sig": ""}], "current": 0}
	if not idx.is_empty():
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
	if _intro >= 0.0:
		Director.intro_hold = _intro
	var subs: Subtitles = preload("res://scripts/subtitles.gd").new()
	var scrub: Dictionary = Audit.scrub_to(body, _from) if not _from.is_empty() else {"si": 0}
	if int(scrub["si"]) < 0:
		print("tablet_look_probe: '%s' is not read in this chapter" % _from)
		get_tree().quit(2)
		return
	subs.words = Audit.timeline(body, _word, Director.intro_hold, int(scrub["si"]))
	subs.document = {"source": body, "title": title}
	if int(scrub["si"]) > 0:
		subs.document["start_words"] = scrub["words"]
		print("tablet_look_probe: a scrub to spoken word %d, '%s'" % [int(scrub["si"]), " ".join(scrub["words"])])
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
	if _audit:
		await _run_audit(tab, medium, end, subs)
		_times = []
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


# --- the audit ---------------------------------------------------------------------------

func _run_audit(tab: TabletMedium, medium: Medium, end: float, subs: Subtitles) -> void:
	var audit = Audit.new(Vector2(W, H))
	# stepped by hand, awaiting a real frame only now and then: a whole chapter otherwise takes
	# most of an hour
	subs.process_mode = Node.PROCESS_MODE_DISABLED
	var step := 0
	var t := 0.0
	var focus_t := -1.0
	var focus_wi := -1
	if not _focus.is_empty():
		var fs := int(Audit.scrub_to(subs.document["source"], _focus)["si"])
		focus_wi = int((tab._doc.get("spoken", PackedInt32Array()) as PackedInt32Array)[fs]) if fs >= 0 \
			and not tab._doc.is_empty() else -1
	var recent: Array = []          # the last seconds of trace, printed when the focus arrives
	var n := 0
	while t < minf(end, _until):
		t += DT
		Spectrum.virtual_clock = t
		Spectrum.current.time = t
		subs._process(DT)
		medium.advance(Spectrum.current, DT, 1.0)
		step += 1
		if step % 30 == 0:
			await get_tree().process_frame
		var row: Dictionary = audit.frame(tab)
		audit.count(row, tab, DT)
		if row.is_empty() or _focus.is_empty():
			continue
		n += 1
		var word := String((tab._doc["words"][int(row["wi"])] as Dictionary)["text"])
		var line := "  t=%7.2f %-14s vp_y %.2f frame_y %.2f aim_y %4.0f scroll %5.0f d %.2f%s" % [t,
			word.left(14), float(row["vp_y"]), float(row["fy"]), float(row["aim_y"]),
			float(row["scroll"]), tab._c_dist, "  busy" if bool(row["busy"]) else ""]
		if focus_t < 0.0:
			if n % 3 == 0:
				recent.append(line)
				if recent.size() > 100:
					recent.pop_front()
			if focus_wi < 0 and not tab._doc.is_empty():
				var fs := int(Audit.scrub_to(subs.document["source"], _focus)["si"])
				focus_wi = int((tab._doc["spoken"] as PackedInt32Array)[fs]) if fs >= 0 else -2
			if int(row["wi"]) == focus_wi:
				focus_t = t
				print("tablet_look_probe: focus '%s' first spoken at t=%.2f" % [_focus, t])
				for l in recent:
					print(l)
		elif t - focus_t < 8.0 and n % 3 == 0:
			print(line)
	audit.finish()
	print("tablet_look_probe: audit - " + audit.summary())


func _spread(img: Image) -> float:
	var lo := 1.0
	var hi := 0.0
	for y in range(0, img.get_height(), 24):
		for x in range(0, img.get_width(), 24):
			var l := img.get_pixel(x, y).get_luminance()
			lo = minf(lo, l)
			hi = maxf(hi, l)
	return hi - lo
