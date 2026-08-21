extends Node

## Gate for the one thing scenes/fractal_zoom.gd cannot be written without: that PERTURBATION
## actually buys the depth it claims to.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/fractal_depth_check.gd 240
##
## GHOST_PROBE_GPU is not optional here - the whole claim lives in a fragment program, so the
## only way to ask it is to render and read the pixels back.
##
## WHY IT NEEDS A GATE AT ALL. The failure is silent and it looks like a design choice. A fractal
## zoom that has run out of float32 does not error, warn, or glitch: every pixel of the frame
## computes the same complex number, so the frame comes out as ONE FLAT COLOUR. On a scene whose
## palette is sampled per session, a flat frame reads as "this seed rolled a boring one" - and
## the scene would keep shipping that way. It already did once for a different reason: a stray
## `set_shader_parameter("u_ref_len", 0)` written after the reference orbit had been uploaded
## turned the perturbed path off completely, and three seeds in a row rendered flat washes that
## looked exactly like an unlucky palette.
##
## THE CONTROL IS THE POINT. Structure in the perturbed render proves nothing on its own - it
## would also appear at a scale float32 handles fine. So the same scene, at the same anchor, at
## the same depth, is rendered TWICE with only `u_perturb` changed. Past float32's limit the
## direct path must collapse to a flat frame and the perturbed one must not. If they ever both
## pass, this gate is measuring nothing.
##
## It also holds two cheap properties of the CPU half, both of which decide whether there is
## anything at the anchor to look at: that the anchor is ON the boundary (a long but FINITE
## escape - the interior is a flat hole and the far outside is a flat wash), and that the zoom
## stays inside the range its own path can resolve however long it runs.

const SEEDS := [11, 23, 47]
const W := 640
const H := 360

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	for sv in SEEDS:
		await _run_seed(int(sv))
	print("")
	if _fails.is_empty():
		print("fractal_depth_check: ALL OK - perturbation resolves what float32 cannot.")
	else:
		print("fractal_depth_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(1 if not _fails.is_empty() else 0)


func _run_seed(sv: int) -> void:
	# The scene is rolled until it lands on the quadratic Mandelbrot, because that is the only
	# family perturbation is implemented for - and rolling for it is honest, where forcing the
	# fields by hand would test a configuration the Director can never produce.
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var sc = null
	var used := -1
	for k in 40:
		var candidate := sv * 101 + k * 7
		var probe = load("res://scripts/scenes/fractal_zoom.gd").new()
		vp.add_child(probe)
		probe.init_with_seed(candidate, "drift")
		if bool(probe.params.get("perturbed", false)):
			sc = probe
			used = candidate
			break
		vp.remove_child(probe)
		probe.free()
	if sc == null:
		_fails.append("seed %d: no perturbed instance in 40 rolls - the family bag is wrong" % sv)
		vp.queue_free()
		await get_tree().process_frame
		return
	print("")
	print("--- seed %d -> %d: %s ---" % [sv, used, sc.params])

	# --- the anchor is on the boundary: a long escape, but one that finishes ---
	var esc: int = sc._escape(sc._cx, sc._cy, 3000)
	print("  anchor escapes at iteration %d of 3000" % esc)
	_ok(esc > 60, "seed %d: the anchor escapes after only %d iterations - it is out in the flat "
		% [used, esc] + "exterior, and there is nothing at it to zoom into")
	_ok(esc < 3000, "seed %d: the anchor never escapes - it is inside the set, whose interior is "
		% used + "a flat hole, and whose orbit is useless as a reference for the surface")

	# --- the zoom stays inside the range its own path can resolve ---
	var lo: float = sc._zl
	var hi: float = sc._zl
	for _i in 5400:                       # three minutes at the fixed step, well past any hold
		sc._step_zoom(1.0 / 30.0)
		lo = minf(lo, sc._zl)
		hi = maxf(hi, sc._zl)
	print("  over 3 minutes the scale stays in [%s, %s]; its floor is %s"
		% [_sci(exp(lo)), _sci(exp(hi)), _sci(exp(sc._zl_min))])
	_ok(lo >= sc._zl_min - 1e-6, "seed %d: the zoom reached %s, past its own floor of %s - "
		% [used, _sci(exp(lo)), _sci(exp(sc._zl_min))] + "below that the picture is blocks")
	_ok(hi <= sc._zl_max + 1e-6, "seed %d: the zoom reached %s, past its ceiling"
		% [used, _sci(exp(hi))])

	# --- THE CLAIM: at a depth float32 cannot hold, perturbation resolves and direct does not ---
	sc._zl = log(exp(sc._zl_min) * 6.0)   # a little above the floor, comfortably past float32
	sc.update(_features(), 1.0 / 30.0)
	for _i in 6:
		await get_tree().process_frame
	var img_on := vp.get_texture().get_image()
	var on := _structure(img_on)
	var b_on := _blocky(img_on)
	if OS.get_cmdline_user_args().has("--dump"):
		img_on.save_png("/tmp/claude-1000/-home-crow-repos-praxis/6dd88a0f-d6d1-4d60-b874-3b0f3e0e6291/scratchpad/depth_on_%d.png" % used)
	sc._mat.set_shader_parameter("u_perturb", false)
	for _i in 6:
		await get_tree().process_frame
	var img_off := vp.get_texture().get_image()
	var off := _structure(img_off)
	var b_off := _blocky(img_off)
	if OS.get_cmdline_user_args().has("--dump"):
		img_off.save_png("/tmp/claude-1000/-home-crow-repos-praxis/6dd88a0f-d6d1-4d60-b874-3b0f3e0e6291/scratchpad/depth_off_%d.png" % used)
	sc._mat.set_shader_parameter("u_perturb", true)
	print("  at %s: perturbed %.1f%% structure / %.1f%% blocky;  direct (the control) %.1f%% / %.1f%%"
		% [_sci(exp(sc._zl)), on * 100.0, b_on * 100.0, off * 100.0, b_off * 100.0])
	_ok(on > 0.15, "seed %d: the PERTURBED render carries only %.1f%% local contrast at %s - a "
		% [used, on * 100.0, _sci(exp(sc._zl))] + "dead frame, so perturbation is not working "
		+ "and the scene is empty at every depth it was written for")
	_ok(b_off > 0.70, "seed %d: the DIRECT render is only %.1f%% blocky at %s, where float32 "
		% [used, b_off * 100.0, _sci(exp(sc._zl))] + "cannot resolve a pixel - it should be "
		+ "almost entirely quantised, so this gate is not standing where it thinks it is")
	_ok(b_on < b_off - 0.20, "seed %d: perturbed %.1f%% blocky against the direct path's %.1f%% "
		% [used, b_on * 100.0, b_off * 100.0] + "- the two paths are drawing the same quantised "
		+ "picture, which is what perturbation being broken looks like")

	sc.queue_free()
	vp.queue_free()
	await get_tree().process_frame


## THE MEASURE THAT ACTUALLY SEPARATES THEM, and it took a render to find. The obvious one -
## how much of the frame differs from its neighbourhood - does not work: a float32 render past
## its limit is not FLAT, it is BLOCKY. The complex coordinate quantises to a few dozen distinct
## values across the frame, so the picture becomes big hard-edged rectangles of uniform colour,
## and every one of those edges reads as "structure". Measured 24-32% against the perturbed
## render's 44-74% - a difference, but nowhere near a verdict.
##
## Blockiness is the verdict. Inside a quantised block EVERY adjacent pair of pixels is byte
## identical; a resolved fractal changes from pixel to pixel almost everywhere it has detail.
## So this counts the share of horizontally adjacent pairs that are exactly equal.
func _blocky(img: Image) -> float:
	var n := 0
	var same := 0
	for y in range(0, img.get_height(), 2):
		for x in range(0, img.get_width() - 1):
			n += 1
			if img.get_pixel(x, y) == img.get_pixel(x + 1, y):
				same += 1
	return float(same) / float(maxi(1, n))


## How much of a frame carries local contrast at all - the cheap "did anything render" read,
## kept for the perturbed side where the question is only whether the frame is dead.
func _structure(img: Image) -> float:
	var n := 0
	var busy := 0
	for y in range(0, img.get_height() - 4, 3):
		for x in range(0, img.get_width() - 4, 3):
			var a := img.get_pixel(x, y)
			var b := img.get_pixel(x + 4, y + 4)
			n += 1
			if absf(a.r - b.r) + absf(a.g - b.g) + absf(a.b - b.b) > 0.06:
				busy += 1
	return float(busy) / float(maxi(1, n))


## GDScript's % formatting has no %e, and these numbers are all exponent.
func _sci(x: float) -> String:
	return String.num_scientific(x)


func _features() -> AudioFeatures:
	var f := AudioFeatures.new()
	var bands := PackedFloat32Array()
	bands.resize(Spectrum.BAND_COUNT)
	bands.fill(0.35)
	f.bands = bands
	f.energy = 0.4
	f.beat_period = 0.5
	return f


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)
