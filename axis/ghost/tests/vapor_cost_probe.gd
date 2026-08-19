extends Node

## Throwaway probe: what does one `vapors` frame cost at export resolution?
##
## The field is a full-frame fragment program with ~20 value-noise lookups per pixel, so its
## cost is 2 million pixels' worth of that and nothing else in the scene comes close. This
## measures it the only honest way - render the real scene at 1920x1080 with the field's quad
## visible, then again with it HIDDEN, and difference the frame times. The delta is the field;
## the baseline is the bed, stars and motes it sits on.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/vapor_cost_probe.gd 240
##
## Needs the real renderer (a dummy driver renders no fragments and would report the field as
## free). Vsync is switched off first, or every frame reports as the refresh interval.

const W := 1920
const H := 1080
const WARM := 12
const REPS := 60


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var sc = load("res://scripts/scenes/vapors.gd").new()
	vp.add_child(sc)
	sc.init_with_seed(36, "drift")
	print("vapor_cost_probe: %dx%d, character %s, %d plumes"
		% [W, H, sc.params.get("character", "?"), sc._vapor._plumes.size()])
	var f := AudioFeatures.new()
	f.bands.resize(Spectrum.BAND_COUNT)
	for b in Spectrum.BAND_COUNT:
		f.bands[b] = 0.6
	f.energy = 0.6
	var on := await _time(sc, f, vp, true)
	var off := await _time(sc, f, vp, false)
	print("vapor_cost_probe: field ON %.2f ms/frame, field HIDDEN %.2f ms/frame -> the field costs %.2f ms"
		% [on, off, on - off])
	print("vapor_cost_probe: at 60 fps a frame's whole budget is 16.67 ms")
	for _i in 4:
		await get_tree().process_frame
	get_tree().quit(0)


func _time(sc, f: AudioFeatures, _vp: SubViewport, field_on: bool) -> float:
	# The field lives on a child quad of the scene, so hiding it removes exactly the
	# fragment program and leaves every CPU path (the plume sim, the layers) running.
	for _i in WARM:
		sc.update(f, 1.0 / 60.0)
		await get_tree().process_frame
	if sc._vapor._quad != null:
		sc._vapor._quad.visible = field_on
	for _i in 4:
		await get_tree().process_frame
	var t0 := Time.get_ticks_usec()
	for _i in REPS:
		sc.update(f, 1.0 / 60.0)
		await get_tree().process_frame
	var us := Time.get_ticks_usec() - t0
	return float(us) / float(REPS) / 1000.0
