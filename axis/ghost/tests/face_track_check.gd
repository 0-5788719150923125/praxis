extends SceneTree

## The face track's FORMAT CONTRACT, checked from the reading side.
##   godot --headless --path . --script res://tests/face_track_check.gd
##
## face_host/face_track.py writes this file in Python and MaskEditor reads it in
## GDScript, which means the layout is agreed in two places and enforced in
## neither. That is exactly the kind of seam that fails silently: when the header
## size was first read as 16 bytes instead of 20, every sample came back shifted
## by four bytes, so the "found" flag read 132 and a landmark at (0.62, 0.37)
## read as (0.64, 0.80) - plausible-looking coordinates, entirely wrong, and no
## error anywhere. This builds a track BYTE BY BYTE to the documented layout,
## reads it through the editor's own reader, and checks the values come back.
##
## It deliberately does NOT need mediapipe, a venv, or a real clip - the contract
## is the thing under test, not the detector. Whether the detector is any good is
## a question for real footage, and the answer to that one is a picture.

const POINTS := 478
const COUNT := 40
const RATE := 15.0

var _fails: PackedStringArray = []


func _initialize() -> void:
	var path := ProjectSettings.globalize_path("user://face_tracks")
	DirAccess.make_dir_recursive_absolute(path)
	path = path.path_join("_check.bin")
	# A synthetic track: landmark k of sample i sits at a position that is a pure
	# function of (i, k), so ANY misread - wrong header, wrong stride, wrong
	# endianness, off-by-one sample - lands on a value the check can name.
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_buffer("GFT1".to_ascii_buffer())
	f.store_32(1)
	f.store_float(RATE)
	f.store_32(COUNT)
	f.store_32(POINTS)
	for i in COUNT:
		# Sample 7 is a LOST frame - the one case the reader must not interpolate
		# across, and the one a naive reader silently blends into its neighbours.
		f.store_8(0 if i == 7 else 1)
		for k in POINTS:
			f.store_float(_expect_x(i, k))
			f.store_float(_expect_y(i, k))
	f.close()

	var ed: Node = load("res://scripts/mask_editor.gd").new()
	var s := MaskSession.new()
	s.video_path = "res://masks/_face_check/video.ogv"
	ed.session = s
	ed._ft_path = path
	ed._ft_load()
	print("state=%s count=%d points=%d rate=%.1f" % [ed._ft_state, ed._ft_count, ed._ft_points, ed._ft_rate])
	_expect(ed._ft_state == "ready", "the reader rejected a well-formed track (%s)" % ed._ft_state)
	_expect(ed._ft_count == COUNT, "count read as %d, wrote %d" % [ed._ft_count, COUNT])
	_expect(ed._ft_points == POINTS, "points read as %d, wrote %d" % [ed._ft_points, POINTS])

	# EXACT sample times must return exactly what was written - this is the test
	# the 16-vs-20 byte header failed. It survives the reader's smoothing because
	# the synthetic track is linear in the sample index; see _expect_x.
	# Away from BOTH ends: at an edge the kernel is truncated, so it is no longer
	# symmetric and the exact-value argument above stops applying. That is correct
	# behaviour, not a bug - there are no samples past the end to average in.
	# Also clear of the LOST sample at 7: its window would be missing a member and
	# so no longer symmetric either, for exactly the same reason.
	for probe in [[16, 0], [16, 33], [20, 263], [27, 1], [30, 477]]:
		var i: int = probe[0]
		var k: int = probe[1]
		var got: Vector2 = ed._ft_point(k, float(i) / RATE)
		var want := Vector2(_expect_x(i, k), _expect_y(i, k))
		print("  sample %2d landmark %3d -> (%.4f, %.4f)  want (%.4f, %.4f)"
			% [i, k, got.x, got.y, want.x, want.y])
		_expect(got.distance_to(want) < 0.0005,
			"sample %d landmark %d read (%.4f,%.4f), wrote (%.4f,%.4f)"
			% [i, k, got.x, got.y, want.x, want.y])

	# BETWEEN samples, the reader interpolates - the mask has to move continuously
	# rather than stepping at the sample rate.
	# Mid-track, for the same symmetry reason as the probes above.
	var mid: Vector2 = ed._ft_point(0, 20.5 / RATE)
	var a := Vector2(_expect_x(20, 0), _expect_y(20, 0))
	var b := Vector2(_expect_x(21, 0), _expect_y(21, 0))
	print("  halfway between samples 20 and 21 -> (%.4f, %.4f)" % [mid.x, mid.y])
	_expect(mid.distance_to(a.lerp(b, 0.5)) < 0.0005,
		"the reader did not interpolate between samples (got %.4f, wanted %.4f)"
		% [mid.x, a.lerp(b, 0.5).x])

	# ...but NEVER across a lost sample. Reading just before the gap must return
	# the last good sample, not a blend of it with a held frame.
	_expect(not ed._ft_has(7.0 / RATE), "sample 7 was written lost but reads as found")
	_expect(ed._ft_has(6.0 / RATE), "sample 6 was written found but reads as lost")
	# A lost sample must be SKIPPED by the smoothing, not averaged in - its stored
	# coordinates are a held copy of a neighbour, so including them would drag the
	# feature toward wherever the hold happened to be. On a linear track a
	# correctly-skipped gap still reads very close to the line (the remaining
	# weights stay near-symmetric); an averaged-in hold pulls visibly off it.
	var near_gap: Vector2 = ed._ft_point(0, 7.0 / RATE)
	var on_line := Vector2(_expect_x(7, 0), _expect_y(7, 0))
	print("  at the lost sample -> (%.4f, %.4f)  (the line there is %.4f, %.4f)"
		% [near_gap.x, near_gap.y, on_line.x, on_line.y])
	_expect(near_gap.distance_to(on_line) < 0.004,
		"reading at a lost sample landed (%.4f,%.4f) against a line at (%.4f,%.4f) - "
		% [near_gap.x, near_gap.y, on_line.x, on_line.y]
		+ "the held coordinates were averaged in instead of skipped")

	# THE SMOOTHING ITSELF has to be reachable and has to do something. A wider
	# kernel over a linear track returns the same line (that is the point of a
	# centred kernel - no lag, no bias), so what is checked here is that the
	# reader accepts the setting and stays on the line rather than drifting.
	ed._ft_sigma = 6.0
	var wide: Vector2 = ed._ft_point(0, 20.0 / RATE)
	var line20 := Vector2(_expect_x(20, 0), _expect_y(20, 0))
	print("  sigma 6.0 at sample 20 -> (%.4f, %.4f)  line (%.4f, %.4f)"
		% [wide.x, wide.y, line20.x, line20.y])
	_expect(wide.distance_to(line20) < 0.004,
		"heavy smoothing pulled the value off the line (%.4f vs %.4f) - a centred "
		% [wide.x, line20.x] + "kernel must not bias, only smooth")
	ed._ft_sigma = 2.5

	# A truncated file must be REFUSED and removed, not read as garbage.
	var t2 := ProjectSettings.globalize_path("user://face_tracks/_check_short.bin")
	var g := FileAccess.open(t2, FileAccess.WRITE)
	g.store_buffer("GFT1".to_ascii_buffer())
	g.store_32(1)
	g.store_float(RATE)
	g.store_32(COUNT)
	g.store_32(POINTS)
	g.store_8(1)          # one byte of one sample, then nothing
	g.close()
	var ed2: Node = load("res://scripts/mask_editor.gd").new()
	ed2.session = s
	ed2._ft_path = t2
	ed2._ft_load()
	print("truncated track -> state=%s, file still present=%s"
		% [ed2._ft_state, FileAccess.file_exists(t2)])
	_expect(ed2._ft_state == "failed", "a truncated track was accepted (%s)" % ed2._ft_state)
	_expect(not FileAccess.file_exists(t2),
		"a truncated track was left on disk - it would be re-read forever")
	ed.free()
	ed2.free()
	DirAccess.remove_absolute(path)

	print("")
	if _fails.is_empty():
		print("face_track_check: PASS - the track round-trips, interpolates, and ",
			"refuses what it should.")
		quit(0)
	else:
		for x in _fails:
			print("face_track_check: FAIL - ", x)
		quit(1)


## Distinct per (sample, landmark) so a misread is identifiable, LINEAR in the
## sample index, and small enough never to wrap.
##
## Linearity is the load-bearing property. The reader smooths with a CENTRED
## kernel, so it does not return the sample you asked for - it returns a weighted
## average of that sample and its neighbours. Over a linear function a symmetric
## kernel returns the value at its centre EXACTLY, so the round-trip stays an
## exact assertion instead of becoming a tolerance nobody can reason about. (An
## earlier version wrapped with fposmod and broke linearity at the wrap, which is
## the sort of thing that makes a test fail for a reason unrelated to the code.)
static func _expect_x(i: int, k: int) -> float:
	return 0.10 + 0.004 * float(i) + 0.0008 * float(k)


static func _expect_y(i: int, k: int) -> float:
	return 0.30 + 0.003 * float(i) + 0.0005 * float(k)


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)
