extends SceneTree

## Is the rain actually rain?
##   godot --path axis/ghost --position -5000,-5000 --script res://tests/rain_check.gd
## Needs a real renderer (it reads pixels back).
##
## Three claims, each from something the first cuts got wrong:
##
##   INDEPENDENCE  Every drop must go its own way. Column-based shader rain hashes
##                 each LANE once, so every drop down a lane repeats the identical
##                 position, calibre and angle forever - reported, exactly, as
##                 "painting thirty train-track paths across the screen, then
##                 sending trains down the exact same track every single time".
##                 Hashing on the cell index as well as the lane gives each PASS
##                 its own everything. Measured as the overlap between the set of
##                 x positions carrying rain at one moment and at another: shared
##                 tracks means near-total overlap, independent drops means little.
##
##   AMOUNT        Turning it down must EMPTY the sky, not merely thin each drop.
##                 Coverage used to move the lane count, so the sky stayed full and
##                 a drizzle was just a finer downpour.
##
##   DEPTH         At Depth 0 nothing falls in front of the subject. The far sheet
##                 is confined to background, which is the only depth cue available
##                 without a depth buffer.

## Wide on purpose. A near lane is a 26th of the frame and a drop is a pixel or
## two across, so at 400px a lane offers only about six distinguishable positions
## and the independence measure hits that ceiling rather than the rain's. At 1200
## a lane is ~46px and the question can actually be asked.
const W := 1200
const H := 600

var _fails: PackedStringArray = []


func _initialize() -> void:
	# A frame split into background (dark, flat) and subject (lit, structured), so
	# the depth heuristic has both to work with.
	var src := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var lit := x > W / 2
			var n := 0.06 * float((x * 7 + y * 13) % 5)   # a little structure
			src.set_pixel(x, y, Color(0.05, 0.05, 0.06) if not lit
				else Color(0.62 + n, 0.48 + n, 0.42 + n))
	var tex := ImageTexture.create_from_image(src)

	# INDEPENDENCE. Three earlier measures failed here and each failure says
	# something, so all three are recorded rather than replaced quietly:
	#
	#   "does this column ever have a drop", over a whole column - SATURATES.
	#   Nearly every lane holds a drop somewhere down the frame however independent
	#   they are; it reported 88% overlap for rain that was in fact scattered.
	#
	#   "do the same crossings recur at two moments" - whether a drop crosses a
	#   given row depends on its PHASE, so crossings differ between moments even
	#   when every drop reuses one fixed path. It passed the bug it was written for.
	#
	#   "are positions quantised to lanes", from one instant - there are ~200 lanes
	#   across the frame, so a snapshot of fixed offsets and a snapshot of random
	#   ones look alike: 0.70 against 0.76, no separation.
	#
	# What actually distinguishes them is TEMPORAL, which is also how it was
	# reported - "sending trains down the exact same track every single time". Watch
	# ONE row over many moments. A lane hashed once puts every one of its drops
	# across that row at the SAME x, so however long you watch, the crossings pile
	# onto as many positions as there are lanes. Hashed per drop, each crossing
	# brings its own x and the count keeps growing. So: distinct positions divided
	# by total crossings, accumulated over time.
	# ...and the fourth attempt, which does discriminate: ISOLATE ONE LANE and
	# watch it. Accumulating over the whole frame saturates on the pixel grid (at a
	# 1.5px merge there are only ~130 distinguishable positions across 400px, and
	# 300 crossings fill nearly all of them whatever the drops are doing). Inside a
	# single lane's width the question is clean and unsaturated: a lane hashed once
	# puts every drop it ever produces at ONE x, so a window that wide collects
	# crossings at a single position however long you watch. Hashed per drop, the
	# crossings scatter across the window.
	var win_lo := int(W * 0.30)
	var win_hi := win_lo + int(float(W) / 26.0)   # exactly one near lane
	var lane_x := []
	var lane_n := 0
	for step in 30:
		var img := await _render(tex, 1.0, 1.0, 0.0, 2.0 + float(step) * 0.211)
		for c in _centres(img, src):
			if c < win_lo or c > win_hi:
				continue
			lane_n += 1
			var dup := false
			for d in lane_x:
				if absf(float(c) - float(d)) <= 1.0:
					dup = true
					break
			if not dup:
				lane_x.append(c)
	var ratio := float(lane_x.size()) / maxf(float(lane_n), 1.0)
	print("independence: within ONE lane, %d crossings over 30 moments landed on %d "
		% [lane_n, lane_x.size()] + "distinct x -> %.2f" % ratio)
	_expect(lane_n > 20, "too few crossings in the sampled lane (%d)" % lane_n)
	# Measured: 0.13 with the shipped per-lane hash (every drop on one track),
	# 0.60+ with the per-drop hash. 0.35 sits between with room on both sides.
	_expect(ratio > 0.35,
		"within one lane, %d crossings landed on only %d distinct positions (%.2f) - "
		% [lane_n, lane_x.size(), ratio]
		+ "every drop is running down the same track")

	# AMOUNT: the sky empties.
	var drizzle := _wetness(await _render(tex, 0.12, 1.0, 0.3, 5.0), src)
	var downpour := _wetness(await _render(tex, 1.0, 1.0, 0.3, 5.0), src)
	print("amount: drizzle %.5f, downpour %.5f  (%.1fx)"
		% [drizzle, downpour, downpour / maxf(drizzle, 1e-6)])
	_expect(downpour > drizzle * 3.0,
		"Amount barely changes how much falls (%.5f vs %.5f) - it is thinning the "
		% [drizzle, downpour] + "drops rather than emptying the sky")
	_expect(drizzle > 0.0,
		"a drizzle produced no rain at all - the existence gate is too tight")

	# DEPTH: at 0, nothing in front of the subject.
	var behind := await _render(tex, 0.8, 0.0, 0.3, 7.0)
	var on_subj := _wetness_where(behind, src, true)
	var on_bg := _wetness_where(behind, src, false)
	print("depth 0: rain on subject %.5f, on background %.5f" % [on_subj, on_bg])
	_expect(on_bg > 0.0005, "Depth 0 produced no rain on the background either")
	_expect(on_subj < on_bg * 0.06,
		"at Depth 0 the rain still falls in front of the subject (%.5f vs %.5f on "
		% [on_subj, on_bg] + "background) - the far sheet is not confined to background")

	print("")
	if _fails.is_empty():
		print("rain_check: PASS - drops go their own way, Amount empties the sky, ",
			"and Depth keeps the far sheet behind the subject.")
		quit(0)
	else:
		for f in _fails:
			print("rain_check: FAIL - ", f)
		quit(1)


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


## Which x positions carry rain ON ONE ROW, over the background half.
##
## One row, not the whole column. Scanning a column's full height asks "does this
## lane ever have a drop in it", and the answer is yes for nearly every lane at
## nearly every moment however independent the drops are - the measurement
## saturates and reports 88% overlap for rain that is in fact scattered. A single
## row asks the question that was actually reported: is a drop crossing HERE, now,
## and is it crossing the same HERE a moment later.
## The CENTRE x of every streak crossing a set of rows, over the background half.
## Centres rather than wet pixels: a streak is several pixels wide, and counting
## each of them would report a wide drop as several distinct positions.
func _centres(got: Image, src: Image) -> Array:
	var out := []
	for row in [int(H * 0.30), int(H * 0.50), int(H * 0.70)]:
		var run_start := -1
		for x in range(2, W - 2):
			var wet := _diff(got, src, x, row) > 0.06
			if wet and run_start < 0:
				run_start = x
			elif not wet and run_start >= 0:
				out.append((run_start + x - 1) / 2)
				run_start = -1
		if run_start >= 0:
			out.append((run_start + W - 3) / 2)
	return out


func _wetness(got: Image, src: Image) -> float:
	var acc := 0.0
	var n := 0
	for y in range(10, H - 10, 2):
		for x in range(2, W / 2 - 2, 2):
			acc += _diff(got, src, x, y)
			n += 1
	return acc / maxf(float(n), 1.0)


func _wetness_where(got: Image, src: Image, subject: bool) -> float:
	var acc := 0.0
	var n := 0
	var lo := W / 2 + 6 if subject else 2
	var hi := W - 6 if subject else W / 2 - 6
	for y in range(10, H - 10, 2):
		for x in range(lo, hi, 2):
			acc += _diff(got, src, x, y)
			n += 1
	return acc / maxf(float(n), 1.0)


static func _diff(got: Image, src: Image, x: int, y: int) -> float:
	var a := got.get_pixel(x, y)
	var b := src.get_pixel(x, y)
	return Vector3(a.r - b.r, a.g - b.g, a.b - b.b).length()


func _render(frame: Texture2D, amount: float, depth: float, squall: float,
		t: float) -> Image:
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/mask_split.gdshader")
	mat.set_shader_parameter("u_threshold", 0.24)
	mat.set_shader_parameter("u_feather", 0.12)
	mat.set_shader_parameter("u_sat_floor", 0.18)
	mat.set_shader_parameter("u_fade", 1.0)
	mat.set_shader_parameter("u_time", t)
	mat.set_shader_parameter("u_aspect", 1.0)
	mat.set_shader_parameter("u_texel", Vector2(1.0 / W, 1.0 / H))
	mat.set_shader_parameter("u_l_count", 1)
	mat.set_shader_parameter("u_l_effect", PackedInt32Array([19, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_hue", PackedFloat32Array([0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_w", PackedFloat32Array([1.0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_off", PackedVector2Array([
		Vector2(0.15, 0), Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO]))
	mat.set_shader_parameter("u_l_scale", _f(1.0))
	mat.set_shader_parameter("u_l_speed", _f(1.0))
	mat.set_shader_parameter("u_l_glow", _f(1.0))
	mat.set_shader_parameter("u_l_dens", _f(amount))
	mat.set_shader_parameter("u_l_con", _f(depth))
	mat.set_shader_parameter("u_l_smooth", _f(squall))
	mat.set_shader_parameter("u_l_lagf", _f(0.0))
	mat.set_shader_parameter("u_l_stick", _f(0.0))
	mat.set_shader_parameter("u_l_tint", _f(0.0))
	mat.set_shader_parameter("u_l_accent", _f(0.0))
	mat.set_shader_parameter("u_l_elag", PackedInt32Array([0, 0, 0, 0, 0, 0]))
	var regs := PackedVector4Array()
	for i in 6:
		regs.append(Vector4(0, 0, 1, 1))
	mat.set_shader_parameter("u_l_region", regs)
	mat.set_shader_parameter("u_l_regsoft", _f(0.0))
	var ew := PackedFloat32Array()
	ew.resize(48)
	for l in 6:
		ew[l * 8] = 1.0
	mat.set_shader_parameter("u_l_ew", ew)
	var tds := PackedVector3Array()
	for i in 6:
		tds.append(Vector3(0.86, -0.37, -0.37))
	mat.set_shader_parameter("u_l_tdir", tds)
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var tr := TextureRect.new()
	tr.texture = frame
	tr.material = mat
	tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	tr.size = Vector2(W, H)
	vp.add_child(tr)
	root.add_child(vp)
	for i in 4:
		await process_frame
	var img: Image = vp.get_texture().get_image()
	vp.queue_free()
	await process_frame
	return img


static func _f(v: float) -> PackedFloat32Array:
	var a := PackedFloat32Array()
	for i in 6:
		a.append(v)
	return a
