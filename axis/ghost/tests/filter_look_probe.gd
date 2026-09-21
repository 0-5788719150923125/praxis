extends SceneTree

## filter_look_probe - NOT a gate. It asserts almost nothing, renders the Look filters and
## writes PNGs to LOOK at.
##
##   tests/run_quiet.sh -- res://tests/filter_look_probe.gd --out /tmp/look
##   ... --filter grain --sweep 0.2,0.45,0.7,1.0        one filter across its range
##   ... --filter grain --frames 6                      consecutive FRAMES of one setting
##   ... --set monochrome=1,grain=0.5                   a combination, one image
##
## WHY IT EXISTS. [code]stage_filter_check.gd[/code] measures each filter against the property
## that names it, and every one of those measurements passed on a grain that was reported as
## "too uniform and consistent... it doesn't look like any version of film grain I've ever
## seen" - because "is there noise, is it in the midtones, is it absent from the blacks" are
## all true of television static as well. Some questions about a picture can only be answered
## by rendering it and looking, and the standing rule in this project is to do exactly that
## rather than to reason about the shader.
##
## --frames is for the parts that only exist over TIME: grain re-rolls at a film's rate and the
## gate slip displaces a band for a single frame, so one still cannot show either. Successive
## PNGs are successive film frames.
##
## The test picture is photographic on purpose - a broad tonal ramp with flat patches in it -
## because grain is judged on midtones and a vignette on edges, and a synthetic chart with hard
## borders hides both.

const Filters_ := preload("res://scripts/filters.gd")

const W := 640
const H := 360

var _out := "/tmp/look"
var _tex: ImageTexture


func _init() -> void:
	_run()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	_out = _arg(args, "--out", _out)
	DirAccess.make_dir_recursive_absolute(_out)
	_tex = _make_picture()

	var which := _arg(args, "--filter", "")
	var frames := int(_arg(args, "--frames", "1"))
	var wrote: Array = []

	if not _arg(args, "--set", "").is_empty():
		var amounts := _parse(_arg(args, "--set", ""))
		for f in frames:
			wrote.append(await _shot(amounts, "set_%02d" % f))
	elif not which.is_empty():
		for step in _arg(args, "--sweep", "0.25,0.5,0.75,1.0").split(",", false):
			var a := float(step)
			for f in frames:
				wrote.append(await _shot({which: a},
					"%s_%03d%s" % [which, int(a * 100.0), "_f%d" % f if frames > 1 else ""]))
	else:
		# The default contact sheet: every filter at its registry default, plus the
		# unfiltered picture to compare against.
		wrote.append(await _shot({}, "none"))
		for key in Filters_.REGISTRY:
			wrote.append(await _shot({String(key): float(Filters_.DEFAULTS[key])},
				"%s_default" % key))

	print("filter_look_probe: wrote %d image(s) to %s" % [wrote.size(), _out])
	for w in wrote:
		print("  " + String(w))
	quit()


## Render one setting and write it. Several engine frames per shot, so anything keyed to TIME
## has actually moved between consecutive images.
func _shot(amounts: Dictionary, name: String) -> String:
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var tr := TextureRect.new()
	tr.texture = _tex
	tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	tr.size = Vector2(W, H)
	Filters_.apply(tr, amounts, Vector2(W, H))
	vp.add_child(tr)
	root.add_child(vp)
	for i in 6:
		await process_frame
	var img := vp.get_texture().get_image()
	vp.queue_free()
	var path := "%s/%s.png" % [_out, name]
	img.save_png(path)
	return path


## A photographic picture rather than a chart: a sky-to-ground tonal ramp with a bright source
## in it, a mid-grey card to judge grain on, and soft shapes so an edge effect has something to
## fall across.
func _make_picture() -> ImageTexture:
	var img := Image.create(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		var v := float(y) / float(H)
		for x in W:
			var u := float(x) / float(W)
			# A broad vertical ramp, warm below and cool above, with a slow horizontal tilt -
			# the tonal range a vignette falls across and grain lives in the middle of.
			var base := Color(0.62 - 0.34 * v, 0.66 - 0.30 * v, 0.74 - 0.22 * v)
			base = base.lerp(Color(0.42, 0.33, 0.24), clampf((v - 0.55) * 2.2, 0.0, 1.0))
			base.r *= 0.88 + 0.24 * u
			base.g *= 0.90 + 0.18 * u
			base.b *= 0.94 + 0.10 * u
			# A soft bright source, for bloom and for a highlight grain must stay out of.
			var d := Vector2(u - 0.26, v - 0.28).length()
			base = base.lerp(Color(1.0, 0.96, 0.88), clampf(1.0 - d * 6.0, 0.0, 1.0))
			# ...and a dark mass, for the other end.
			var d2 := Vector2(u - 0.76, v - 0.74).length()
			base = base.lerp(Color(0.04, 0.05, 0.07), clampf(1.0 - d2 * 4.2, 0.0, 1.0))
			img.set_pixel(x, y, base)
	# A mid-grey card. Grain is judged here: it is flat in the source, so everything visible on
	# it is the filter's.
	for y in range(H / 2 - 26, H / 2 + 26):
		for x in range(W / 2 - 60, W / 2 + 60):
			img.set_pixel(x, y, Color(0.5, 0.5, 0.5))
	return ImageTexture.create_from_image(img)


func _arg(args: PackedStringArray, name: String, dflt: String) -> String:
	var i := args.find(name)
	return String(args[i + 1]) if i >= 0 and i + 1 < args.size() else dflt


func _parse(spec: String) -> Dictionary:
	var out := {}
	for part in spec.split(",", false):
		var bits := String(part).split("=")
		if bits.size() > 1:
			out[String(bits[0]).strip_edges()] = clampf(float(bits[1]), 0.0, 1.0)
	return out
