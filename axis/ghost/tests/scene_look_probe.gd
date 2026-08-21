extends Node

## NOT a gate - it asserts nothing. It renders ANY scene in the catalogue by name and writes
## PNGs, so a look can be judged by looking at it. (clown_look_probe.gd, vapor_look_probe.gd
## and city_look_probe.gd are the same idea aimed at one scene each; this is the general one,
## for building a new scene where the whole question is "what does it actually draw".)
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/scene_look_probe.gd 300 \
##       --out /tmp/look --scene fractal_zoom --seeds 1,2,3 --times 4,20
##
## GHOST_PROBE_GPU is not optional: a scene reaches the Spectrum and Director autoloads (so it
## needs a real boot) and this reads pixels back (so it needs a real renderer, never --headless,
## whose dummy driver returns nothing from a viewport readback).
##
## THE SETTLE TICKS THE SCENE. A scene whose geometry is built by [FrameForge] relaunches its
## worker from kick(), which update() calls - so running the sim in a tight loop and THEN
## waiting for frames stops the kicks and photographs whichever packet happened to finish
## first. That is a picture of t=0 captioned t=20; it was diagnosed the hard way in
## city_look_probe.gd. Frames and sim steps advance together here.

const DT := 1.0 / 30.0

var _out := "user://look"
var _scene := "fractal_zoom"
var _behavior := "drift"
var _seeds: Array = [1, 2, 3]
var _times: Array = [6.0]
var _w := 1280
var _h := 720
var _wrote := 0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_parse_args()
	for sv in _seeds:
		await _shoot(int(sv))
	print("scene_look_probe: wrote %d frame(s) under %s" % [_wrote, _out])
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(0)


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if i + 1 >= args.size():
			break
		match args[i]:
			"--out": _out = args[i + 1]
			"--scene": _scene = args[i + 1]
			"--behavior": _behavior = args[i + 1]
			"--size":
				var wh := String(args[i + 1]).split("x")
				if wh.size() == 2:
					_w = int(wh[0])
					_h = int(wh[1])
			"--seeds":
				_seeds = []
				for s in String(args[i + 1]).split(","):
					_seeds.append(int(s))
			"--times":
				_times = []
				for s in String(args[i + 1]).split(","):
					_times.append(float(s))


func _shoot(sv: int) -> void:
	var path := "res://scripts/scenes/%s.gd" % _scene
	if not ResourceLoader.exists(path):
		print("scene_look_probe: no such scene '%s'" % _scene)
		return
	var vp := SubViewport.new()
	vp.size = Vector2i(_w, _h)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var t0 := Time.get_ticks_usec()
	var sc = load(path).new()
	vp.add_child(sc)                       # parented BEFORE init: it sizes off the viewport
	sc.init_with_seed(sv, _behavior)
	print("--- %s seed %d (built in %d ms): %s ---"
		% [_scene, sv, (Time.get_ticks_usec() - t0) / 1000, sc.params])
	var t := 0.0
	for target: float in _times:
		while t < target:
			sc.update(_features(t), DT)
			t += DT
		for _i in 20:
			sc.update(_features(t), DT)
			t += DT
			await get_tree().process_frame
		# What a frame of this actually costs, measured over the settle rather than guessed. A
		# fragment-heavy scene can look right and still be unaffordable, and a render pays the
		# same bill; ten frames is enough to see an order of magnitude.
		var t0f := Time.get_ticks_usec()
		for _i in 10:
			sc.update(_features(t), DT)
			t += DT
			await get_tree().process_frame
		var ms := float(Time.get_ticks_usec() - t0f) / 10000.0
		var img := vp.get_texture().get_image()
		var p := "%s_%s_%d_t%02d.png" % [_out, _scene, sv, int(round(target))]
		img.save_png(p)
		_wrote += 1
		print("    %s  (%.1f ms/frame, mean luma %.3f, %s)" % [p, ms, _luma(img), _spread(img)])
	vp.queue_free()
	await get_tree().process_frame


# A synthetic track with structure: a swelling loudness, a spectrum whose peak travels, and
# beats on a half-second grid - enough for every audio path in a scene to be exercised.
func _features(t: float) -> AudioFeatures:
	var f := AudioFeatures.new()
	f.time = t
	var loud := clampf(0.30 + 0.45 * pow(sin(t * 0.55) * 0.5 + 0.5, 1.5), 0.0, 1.0)
	var bands := PackedFloat32Array()
	bands.resize(Spectrum.BAND_COUNT)
	for b in Spectrum.BAND_COUNT:
		var x := float(b) / float(maxi(1, Spectrum.BAND_COUNT - 1))
		var peak := 0.5 + 0.45 * sin(t * 0.37)
		bands[b] = clampf(loud * (1.0 - 0.45 * x) * exp(-8.0 * pow(x - peak, 2.0))
			+ 0.12 * loud * (0.6 + 0.4 * sin(x * 21.0 + t * 2.0)), 0.0, 1.0)
	f.bands = bands
	f.energy = loud
	f.bass = loud * 0.9
	f.low_mid = loud * 0.75
	f.mid = loud * 0.6
	f.high = loud * 0.45
	f.flux = 0.02 + 0.03 * absf(sin(t * 1.7))
	f.movement = 0.3 + 0.3 * sin(t * 0.23)
	f.beat = 1.0 if fposmod(t, 0.5) < DT else 0.0
	f.beat_period = 0.5
	return f


func _luma(img: Image) -> float:
	var acc := 0.0
	var n := 0
	for y in range(0, img.get_height(), 4):
		for x in range(0, img.get_width(), 4):
			var c := img.get_pixel(x, y)
			acc += 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
			n += 1
	return acc / float(maxi(1, n))


## A one-line read on whether the frame has anything IN it: the share of sampled pixels that
## differ from the frame's mean by more than a little. A flat wash and a picture can have the
## same mean luma, and a scene that failed to draw is exactly a flat wash.
func _spread(img: Image) -> String:
	var vals := PackedFloat32Array()
	for y in range(0, img.get_height(), 4):
		for x in range(0, img.get_width(), 4):
			var c := img.get_pixel(x, y)
			vals.append(0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b)
	if vals.is_empty():
		return "empty"
	var m := 0.0
	for v in vals:
		m += v
	m /= float(vals.size())
	var busy := 0
	for v in vals:
		if absf(v - m) > 0.04:
			busy += 1
	return "%.0f%% of the frame is not the mean" % (100.0 * float(busy) / float(vals.size()))
