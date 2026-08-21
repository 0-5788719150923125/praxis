extends Node

## Gate for the two structural claims scenes/tunnel_run.gd is built on. No pixels needed - both
## are properties of the generated track, so this runs headless:
##
##   tests/run_boot_probe.sh tests/tunnel_track_check.gd 120
##
## THE TRACK TURNS OVER, which is the whole reason it is integrated rather than splined - and
## the CONTROL is the other construction, built here and measured beside it.
##
## A path written the obvious way, `(x(s), y(s), s)`, has the tangent `(x', y', 1)`. That third
## component is a constant +1, so however wildly x and y wander the tangent can never point back
## along the advance axis and its vertical component can never change sign. Such a track winds;
## it cannot loop, dive, or invert. Carrying an orthonormal frame and turning it by curvature
## about its OWN axes has no such limit. So the gate builds both from the same curvature numbers
## and asserts the difference: the integrated tangent reverses, the height-mapped one cannot.
##
## THE CONTROL HAD TO BE A CONSTRUCTION, NOT A CHARACTER, and finding that out is the useful part
## of this file. Two earlier cuts used `glide` - the gentle track - as the thing that must NOT
## loop, and tightened its curvature table twice trying to make that true. It never can be, for a
## reason that is the design working: ROLL turns the frame, and a YAW about the frame's own
## now-tilted up axis IS world pitch. Any character with a barrel roll can therefore end up in
## any orientation whatever its pitch table says, and glide's roll is what makes its tunnel
## readable. 192 of 240 glide seeds reached vertical on a pitch amplitude of 0.003 - a twentieth
## of what the scene ships. The characters differ in how HARD they turn, not in which axes they
## can reach.
##
## THE FRAME STAYS ORTHONORMAL. The track is three successive rotations of a stored basis per
## station, and a song's worth of it is tens of thousands of stations. Rotation matrices drift:
## round-off accumulates, the axes stop being perpendicular, and the tube shears into an ellipse
## that slowly flattens. It fails gradually and looks like a design choice, which is exactly the
## kind of thing that ships. The scene re-orthonormalises every step; this measures that it works
## over far more track than any session will ever generate.
##
## And that the buffer stays BOUNDED - the track is endless, so the one thing it must not do is
## accumulate.

## How far out of the horizontal counts as having turned over.
const LOOPED := 0.90

const SEEDS := 120
const STATIONS := 24000

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_construction()
	_loops()
	_orthonormal()
	_bounded()
	print("")
	if _fails.is_empty():
		print("tunnel_track_check: ALL OK - the track loops, holds its frame, and stays bounded.")
	else:
		print("tunnel_track_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	for _i in 4:
		await get_tree().process_frame
	get_tree().quit(1 if not _fails.is_empty() else 0)


## THE CONSTRUCTION CLAIM. The integrated track's tangent must at some point point BACKWARDS
## along the direction it set off in; the height-mapped alternative, driven by the very same
## curvature functions, must never manage it.
func _construction() -> void:
	var tried := 0
	var worst_int := 1.0
	var worst_map := 1.0
	var bad: Array = []
	for k in SEEDS:
		var sc = _make(k * 13 + 5, "")
		if sc == null:
			continue
		tried += 1
		var fwd0 := -(sc._bas[0] as Basis).z
		while sc._pos.size() < 900:
			sc._extend()
		var w := 1.0
		for b in sc._bas:
			w = minf(w, (-(b as Basis).z).dot(fwd0))
		worst_int = minf(worst_int, w)
		if w >= 0.0:
			bad.append("%d (%.2f)" % [k * 13 + 5, w])
		# THE CONTROL, from the same numbers: `(x(s), y(s), s)`, where x and y are driven by this
		# instance's own yaw and pitch curves. Its advance component is a constant one.
		var stp: float = sc.STEP
		var prev := Vector3.ZERO
		for i in 900:
			var si := float(i) * stp
			var pt := Vector3(sc._curve(sc._k_yaw, si) * 30.0,
				sc._curve(sc._k_pitch, si) * 30.0, si)
			if i > 0:
				worst_map = minf(worst_map, (pt - prev).normalized().z)
			prev = pt
		sc.free()
	print("  integrated: the tangent's dot with its start reaches %.2f over %d seeds"
		% [worst_int, tried])
	print("  height-mapped control, same curvature: its advance component never drops below %.2f"
		% worst_map)
	_ok(tried >= 20, "only %d tunnels built" % tried)
	# A SHARE, not every seed, and the distinction is real: whether a GENTLE track has come round
	# within five hundred units is a fact about its curvature amplitudes, not about the
	# construction - two of the softest glides had simply not got there yet. The per-seed
	# guarantee lives on `coaster` in _loops(), where reaching vertical is a strictly stronger
	# claim and the height-mapped control cannot manage it at any length whatsoever.
	var share := float(tried - bad.size()) / float(maxi(1, tried))
	_ok(share >= 0.90, "only %.0f%% of integrated tracks turned past a right angle inside 500 "
		% (share * 100.0) + "units (%s) - the curvature across the character table is too weak "
		% ", ".join(bad.slice(0, 6)) + "for the construction to be worth anything")
	_ok(worst_map > 0.0, "the height-mapped control turned past a right angle (%.2f), which its "
		% worst_map + "own algebra forbids - so this gate is not comparing what it claims to")


## THE LOOP CLAIM, on the character that promises it: a `coaster` tangent must reach the whole
## way to vertical.
func _loops() -> void:
	var tried := 0
	var least := 1.0
	var bad: Array = []
	for k in SEEDS:
		var sc = _make(k * 13 + 5, "coaster")
		if sc == null:
			continue
		tried += 1
		var up0 := (sc._bas[0] as Basis).y
		while sc._pos.size() < 900:
			sc._extend()
		var peak := 0.0
		for b in sc._bas:
			peak = maxf(peak, absf((-(b as Basis).z).dot(up0)))
		least = minf(least, peak)
		if peak <= LOOPED:
			bad.append("%d (%.2f)" % [k * 13 + 5, peak])
		sc.free()
	print("  coaster over %d seeds: the tangent tips to %.2f of vertical even at its least"
		% [tried, least])
	_ok(tried >= 8, "only %d coaster tracks in %d seeds - the character bag is broken"
		% [tried, SEEDS])
	_ok(bad.is_empty(), "%d of %d coasters never reached %.2f of vertical (%s) - the character "
		% [bad.size(), tried, LOOPED, ", ".join(bad.slice(0, 6))] + "that exists to turn right "
		+ "over does not")


## Carry the frame over far more track than a session will ever generate and measure how far it
## has drifted from orthonormal: the axes' mutual dot products (0 when perpendicular) and their
## lengths (1 when normalised).
func _orthonormal() -> void:
	var sc = _make(7717, "")
	if sc == null:
		_fails.append("could not build a tunnel at all")
		return
	while sc._pos.size() < STATIONS:
		sc._extend()
	var worst_dot := 0.0
	var worst_len := 0.0
	for b in sc._bas:
		var bb: Basis = b
		worst_dot = maxf(worst_dot, absf(bb.x.dot(bb.y)))
		worst_dot = maxf(worst_dot, absf(bb.x.dot(bb.z)))
		worst_dot = maxf(worst_dot, absf(bb.y.dot(bb.z)))
		worst_len = maxf(worst_len, absf(bb.x.length() - 1.0))
		worst_len = maxf(worst_len, absf(bb.y.length() - 1.0))
		worst_len = maxf(worst_len, absf(bb.z.length() - 1.0))
	# GDScript's % formatting has no %e, and these are all exponent.
	print("  frame over %d stations: axes off perpendicular by %s, off unit length by %s"
		% [STATIONS, String.num_scientific(worst_dot), String.num_scientific(worst_len)])
	_ok(worst_dot < 1e-4, "the carried frame's axes drifted %s from perpendicular over %d "
		% [String.num_scientific(worst_dot), STATIONS] + "stations - the tube shears into a "
		+ "flattening ellipse")
	_ok(worst_len < 1e-4, "the carried frame's axes drifted %s from unit length over %d "
		% [String.num_scientific(worst_len), STATIONS] + "stations - the tunnel's radius creeps")
	sc.free()


## The track is endless, so the buffer must not be. Fly a long way and check the station count
## has not grown.
func _bounded() -> void:
	var sc = _make(4242, "")
	if sc == null:
		return
	# Explicitly typed: `sc` is an untyped local, so nothing it returns carries a type for
	# inference to work from.
	var start: int = sc._pos.size()
	var f := AudioFeatures.new()
	var bands := PackedFloat32Array()
	bands.resize(Spectrum.BAND_COUNT)
	bands.fill(0.5)
	f.bands = bands
	f.energy = 0.7
	f.beat_period = 0.5
	# The scene's own advance, not _extend directly: the trim is what is under test and it lives
	# in _advance_track, which only runs when the camera has actually moved.
	for _i in 9000:
		sc._s += sc._speed / 60.0
		sc._advance_track()
	print("  after %.0f units of track the buffer holds %d stations (started at %d)"
		% [sc._s, sc._pos.size(), start])
	_ok(sc._pos.size() <= start + 2, "the station buffer grew from %d to %d over %.0f units - "
		% [start, sc._pos.size(), sc._s] + "an endless track that accumulates is a leak")
	sc.free()


## Build a tunnel, rolling seeds until one lands on the wanted track character ("" = any).
func _make(base: int, want: String):
	for k in 24:
		var sc = load("res://scripts/scenes/tunnel_run.gd").new()
		sc.size = Vector2(1920, 1080)
		sc.init_with_seed(base + k * 31, "drift")
		if want == "" or String(sc.params.get("track", "")) == want:
			return sc
		sc.free()
	return null


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)
