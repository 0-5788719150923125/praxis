extends Node

## NOT a gate - it asserts nothing. It renders the two CITY scenes and writes PNGs, because a
## question like "are the buildings standing up straight" is answered by looking at a frame and
## not by reading the code that placed them. (vapor_look_probe.gd and clown_look_probe.gd exist
## for the same reason, and the same reason again: visual fixes argued from the source have been
## wrong about things a rendered frame showed in a second.)
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/city_look_probe.gd 300 \
##       --out /tmp/city --scenes terrain_city,spires --seeds 3,11,404 --times 16
##
## GHOST_PROBE_GPU is not optional: these scenes reach the Spectrum and Director autoloads (so
## they need a real boot) and this reads pixels back (so it needs a real renderer, never
## --headless, whose dummy driver returns nothing from a viewport readback).
##
## It also prints the LEAN ON OFFER: how far off vertical the terrain normal at each built plot
## would tip a building, under the rule that used to place them. That is the size of the defect
## the picture is being checked for - the picture itself is what says whether it is still there.

const W := 1280
const H := 720
const DT := 1.0 / 30.0

var _out := "user://city"
var _scenes: Array = ["terrain_city", "spires"]
var _seeds: Array = [3, 11, 404]
var _times: Array = [16.0]
var _wrote := 0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_parse_args()
	for name in _scenes:
		for sv in _seeds:
			await _shoot(String(name), int(sv))
	print("city_look_probe: wrote %d frame(s) under %s" % [_wrote, _out])
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
			"--scenes": _scenes = String(args[i + 1]).split(",")
			"--seeds":
				_seeds = []
				for s in String(args[i + 1]).split(","):
					_seeds.append(int(s))
			"--times":
				_times = []
				for s in String(args[i + 1]).split(","):
					_times.append(float(s))


func _shoot(name: String, sv: int) -> void:
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var sc = load("res://scripts/scenes/%s.gd" % name).new()
	vp.add_child(sc)                       # parented BEFORE init: it sizes off the viewport
	sc.init_with_seed(sv, "drift")
	print("--- %s seed %d: %s ---" % [name, sv, sc.params])
	var t := 0.0
	for target: float in _times:
		while t < target:
			sc.update(_features(t), DT)
			t += DT
		# THE SETTLE MUST KEEP TICKING THE SCENE, and this is the whole probe. [FrameForge] relaunches
		# its worker from kick(), which is called by update() - so a probe that runs the sim in a
		# tight loop and THEN waits for frames stops kicking, and the forge sits on whichever packet
		# it happened to finish first. That is a photograph of the scene at t=0 with the caption t=16.
		# It is not hypothetical: the same seed at the same moment came out as a dense city in one
		# run and as bare terrain in the next, purely on worker timing.
		for _i in 24:
			sc.update(_features(t), DT)
			t += DT
			await get_tree().process_frame
		var img := vp.get_texture().get_image()
		var path := "%s_%s_%d_t%02d.png" % [_out, name, sv, int(round(target))]
		img.save_png(path)
		_wrote += 1
		print("    %s  (mean luma %.3f)" % [path, _luma(img)])
		# AFTER the capture, and after the sim: the city grows in, so measured at t=0 no plot has
		# built and it reported "0 built plots" for a full city.
		print("    lean on offer: %s" % _lean_available(sc, name))
	vp.queue_free()
	await get_tree().process_frame


## What the LEAN would be if the terrain normal were followed - which is the rule that used to
## place these buildings, sampled at the plots that actually built. This reports the terrain, not
## the scene: whether the scene still follows it is what the PICTURE shows.
##
## It is written this way on purpose. The first cut printed "0.00 deg off vertical by
## construction", which is not a measurement - it printed the same line for the control render
## with the lean restored, and a check that cannot fail is worse than no check.
func _lean_available(sc, name: String) -> String:
	var worst := 0.0
	var acc := 0.0
	var n := 0
	if name == "spires":
		for sp in sc._spires:
			var nrm: Vector3 = sc._terrain.normal_world(float(sp.base.x), float(sp.base.z))
			var a := rad_to_deg(nrm.lerp(Vector3.UP, 0.96).normalized().angle_to(Vector3.UP))
			worst = maxf(worst, a)
			acc += a
			n += 1
		return "%d spires; following the terrain normal would lean them %.2f deg mean, %.2f worst"\
			% [n, acc / float(maxi(1, n)), worst]
	for cy in sc.C:
		for cx in sc.C:
			if float(sc._grown[cy * sc.C + cx]) < 0.02:
				continue
			var nr: Vector3 = sc._terrain.normal_world(sc._plot_wx(cx), sc._plot_wz(cy))
			var b := rad_to_deg(nr.lerp(Vector3.UP, 0.92).normalized().angle_to(Vector3.UP))
			worst = maxf(worst, b)
			acc += b
			n += 1
	var lo := 1e9
	var hi := -1e9
	for v in sc._heading:
		lo = minf(lo, float(v))
		hi = maxf(hi, float(v))
	return ("%d built plots; following the terrain normal would lean them %.2f deg mean, %.2f "
		% [n, acc / float(maxi(1, n)), worst]
		+ "worst. Street-grid headings span %.0f deg" % rad_to_deg(hi - lo))


# A synthetic track with structure: a swelling loudness, a spectrum whose peak travels, and
# beats on a half-second grid - enough for every audio path in the scene to be exercised.
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
