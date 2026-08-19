extends Node

## NOT a gate - it asserts nothing. It renders the `vapors` scene and writes PNGs, because
## this look cannot be reasoned about: three earlier visual fixes in this project were
## argued from the shader source and were wrong about something a rendered frame would have
## shown in a second (see clown_look_probe.gd, which exists for the same reason).
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/vapor_look_probe.gd 300 \
##       --out /tmp/vapor --chars ink,nebula,gossamer,plasma --times 1,5,11
##
## GHOST_PROBE_GPU is not optional: the scene reaches the Spectrum and Director autoloads
## (so it needs a real boot) and this reads pixels back (so it needs a real renderer).
## One PNG per character per time, plus the seed / mood / layout each one rolled.

const W := 960
const H := 540
const DT := 1.0 / 30.0

var _out := "user://vapor"
var _chars: Array = ["ink", "nebula", "gossamer", "plasma"]
var _times: Array = [1.0, 5.0, 11.0]
var _behavior := "static"
## Shader-uniform overrides applied AFTER the layer seeded itself, so a look can be pushed
## around without editing the scene's tables: --knob u_stretch=3.0,u_gain=1.6
var _knobs: Dictionary = {}
## One uniform swept over several values, each rendered from the same seed at the same
## moment - the contact sheet, in ONE process (one boot, not five): --sweep u_stretch=2,6,14
var _sweep_key := ""
var _sweep_vals: Array = []
var _wrote := 0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_parse_args()
	for want in _chars:
		var sv := _find_seed(String(want))
		if sv < 0:
			print("vapor_look_probe: no seed in the search range rolled character '%s'" % want)
			continue
		await _shoot(String(want), sv)
	print("vapor_look_probe: wrote %d frame(s) under %s" % [_wrote, _out])
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
			"--chars": _chars = String(args[i + 1]).split(",")
			"--behavior": _behavior = args[i + 1]
			"--knob":
				for pair in String(args[i + 1]).split(","):
					var kv := pair.split("=")
					if kv.size() == 2:
						_knobs[String(kv[0])] = float(kv[1])
			"--sweep":
				var kv2 := String(args[i + 1]).split("=")
				if kv2.size() == 2:
					_sweep_key = String(kv2[0])
					_sweep_vals = []
					for v in String(kv2[1]).split(","):
						_sweep_vals.append(float(v))
			"--times":
				_times = []
				for s in String(args[i + 1]).split(","):
					_times.append(float(s))


# The character is rolled from the seed inside build_params, so finding one means hunting
# for a seed that rolls it. Cheap: a vapors instance is a handful of dictionaries.
func _find_seed(want: String) -> int:
	for k in 400:
		var sv := 1 + k * 7
		var sc = load("res://scripts/scenes/vapors.gd").new()
		add_child(sc)
		sc.init_with_seed(sv, _behavior)
		var got := String(sc.params.get("character", ""))
		remove_child(sc)
		sc.free()
		if got == want:
			return sv
	return -1


func _shoot(want: String, sv: int) -> void:
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var sc = load("res://scripts/scenes/vapors.gd").new()
	vp.add_child(sc)                       # parented BEFORE init: it sizes off the viewport
	sc.init_with_seed(sv, _behavior)
	print("--- %s: seed %d, mood %s, layout %s ---"
		% [want, sv, sc.params.get("mood", "?"), sc.params.get("layout", "?")])
	var hues := PackedStringArray()
	for pl in sc._vapor._plumes:
		hues.append("%.2f" % float(pl.hue))
	print("    plumes: %s (hues %s)" % [sc._vapor._plumes.size(), ", ".join(hues)])
	for k in _knobs:
		sc._vapor._mat.set_shader_parameter(String(k), float(_knobs[k]))
	var t := 0.0
	for target: float in _times:
		# Advance the SIM without waiting on the engine (the field is a function of the
		# uniforms the update pushes, so nothing needs a frame between steps) ...
		while t < target:
			sc.update(_features(t), DT)
			t += DT
		# ... then give it frames to actually draw. The shader quad is created inside the
		# first _draw (deferred, see Layer.FieldQuad), so the first capture needs a few.
		for _i in 4:
			await get_tree().process_frame
		# The sweep renders its variants at THIS moment, from this one instance - the field
		# is a pure function of its uniforms, so nothing needs re-seeding between them.
		var vals: Array = _sweep_vals if not _sweep_key.is_empty() else [0.0]
		for vi in vals.size():
			var tag := "t%02d" % int(round(target))
			if not _sweep_key.is_empty():
				sc._vapor._mat.set_shader_parameter(_sweep_key, float(vals[vi]))
				tag = "%s_%s%.2f" % [tag, _sweep_key.trim_prefix("u_"), float(vals[vi])]
				for _j in 2:
					await get_tree().process_frame
			var img := vp.get_texture().get_image()
			var path := "%s_%s_%s.png" % [_out, want, tag]
			img.save_png(path)
			_wrote += 1
			print("    %s  (mean luma %.3f, lit %.1f%%)" % [path, _luma(img), 100.0 * _lit(img)])
	vp.queue_free()
	await get_tree().process_frame


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
	f.treble = loud * 0.3
	var phase := fposmod(t, 0.5)
	f.beat = clampf(1.0 - phase * 6.0, 0.0, 1.0)
	f.flux = 0.35 if phase < DT else 0.03
	f.movement = 0.2
	f.beat_period = 0.5
	return f


func _luma(img: Image) -> float:
	var acc := 0.0
	var n := 0
	for y in range(0, img.get_height(), 4):
		for x in range(0, img.get_width(), 4):
			var c := img.get_pixel(x, y)
			acc += 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
			n += 1
	return acc / float(maxi(1, n))


func _lit(img: Image) -> float:
	var hit := 0
	var n := 0
	for y in range(0, img.get_height(), 4):
		for x in range(0, img.get_width(), 4):
			var c := img.get_pixel(x, y)
			if 0.299 * c.r + 0.587 * c.g + 0.114 * c.b > 0.06:
				hit += 1
			n += 1
	return float(hit) / float(maxi(1, n))
