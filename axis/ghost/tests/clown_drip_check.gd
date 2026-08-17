extends SceneTree

## The clown's eye-drip: how FAR it runs, and whether it follows the face.
##   godot --path axis/ghost --position -5000,-5000 --script res://tests/clown_drip_check.gd
## Needs a real renderer (the paint field is a ping-pong SubViewport pair).
##
## Two claims, both from the same report - "even when completely maxxed-out the
## streaks barely cross the upper cheek", and "they don't track the facial
## contours at all, they float around on top of the face".
##
##   REACH    at the top of the slider a run gets from the eye to the CHIN. Both
##            halves of the old version worked against that: the distance was a
##            fraction of FRAME height (which says nothing about the face it is
##            running down) and the taper was pow(1 - up, 1.6), already down to a
##            third by the halfway point - so however far the reach was set, the
##            run had faded across the upper cheek and the slider looked inert
##            past its first quarter.
##
##   CONTOUR  the run bends along the picture's own structure. Marching straight
##            up the screen to look for eye paint lays a streak ACROSS a cheekbone
##            rather than down it, which is what reads as floating. Measured on a
##            frame with a strong diagonal ridge: the drip must lean toward the
##            ridge rather than falling vertically.
##
## ...and then two more reports, both on the SHAPE of what was reaching the chin:
## "four or five very thin, very long, very straight streaks per eye, running down
## the middle of the face", and after that "a gigantic upside-down teardrop that
## STARTS at the upper cheekbone and does not blend into the eye shadow". The first
## is what a per-column lane noise draws; the second is what happens when the run's
## size and origin are guessed from a radius rather than measured off the paint
## that is actually there. Three properties are asserted on a scanline through the
## left run, at 10% and 90% of ITS OWN extent (fixed rows measure an unknown
## fraction along a run whose start is measured, not fixed):
##
##   TAPER    wide where it leaves the lash line, a hairline at the tail. A lane
##            bundle is the same thickness end to end and scores about 1x.
##   SINGLE   a pool drains at ONE point - exactly one span. A finer comb cannot be
##            used to check this: the run is about six texels wide at its widest
##            here, and four streaks do not fit in six texels at any spacing. The
##            fault injected instead is a SECOND rivulet a socket-width inboard,
##            which is the shape the lane noise actually drew.
##   OUTBOARD it CURVES outward as it descends, down the outer cheek rather than
##            straight past the nose and the lips. Measured as the shift between
##            the two rows, never against the eye: the drain point already sits
##            outside the eye's centre, so a straight-down run passes a test
##            written that way (verified - it did).
##
## All three verified two-sided by breaking the shader three ways (constant width,
## zero outward bias, a second rivulet) - each fails exactly its own assertion.
##
##   EYEBALL  the run does not paint the ring's OPENING. Once it leaves at the
##            socket's own width its head reaches back up into the hole, over the
##            eyeball - the one boundary the rest of the eye mask respects and the
##            run was ignoring. The synthetic eyes here are ANNULI for that reason;
##            a solid patch cannot test it.
##
## Drip width and Drip curve are pushed WOUND UP here. What is under test is the
## shape law, and at 320px the default lean streak is a two-pixel thread whose
## width cannot be measured to better than 50%.

const W := 320
const H := 320
const CX := 0.5
const CY := 0.40
const RX := 0.26
const RY := 0.34
const EYE_Y := 0.34
## The ring's opening, as a fraction of THE DRAWN PATCH. The editor's numbers are
## an opening of 0.9x the measured eye against a patch grown to Eye size (2.2x at
## its stored default), so the annulus runs 0.42..1.0 of what it draws - a thick
## ring, not a hairline. Getting this wrong is not a detail: a first cut used 0.9
## of the patch, a ring one pixel thick at this resolution, and the shader's walk
## for the patch's lower edge stepped straight over it and found no eye at all.
const HOLE := 0.42

var _fails: PackedStringArray = []


func _initialize() -> void:
	# REACH, on a flat face so nothing bends the run.
	var chin := CY + RY
	var span := chin - EYE_Y
	print("eye at y=%.2f, chin at y=%.2f -> %.2f of the frame between them" % [EYE_Y, chin, span])
	var reached := {}
	for drip in [0.0, 0.35, 1.0]:
		var f := await _run(drip, false)
		var low := -1.0
		for y in range(int(EYE_Y * H) + 6, H):
			var hit := false
			# NEARLY THE WHOLE WIDTH. A window of 0.2..0.8 was fine while the run fell
			# straight down; once it curves outboard the tail leaves that window and
			# the reach reads short for a reason that has nothing to do with reach.
			for x in range(int(W * 0.05), int(W * 0.95)):
				if f.get_pixel(x, y).r > 0.25:
					hit = true
					break
			if hit:
				low = (float(y) + 0.5) / float(H)
		reached[drip] = low
		var frac := (low - EYE_Y) / span if low > 0.0 else 0.0
		print("  Drip %.2f -> paint reaches y=%.3f, which is %.0f%% of the way from "
			% [drip, low, frac * 100.0] + "the eye to the chin")
	var full: float = (float(reached[1.0]) - EYE_Y) / span
	_expect(full > 0.80,
		"at full Drip the run only gets %.0f%% of the way to the chin - the reach "
		% (full * 100.0) + "or the taper is cutting it short")
	_expect(float(reached[0.35]) < float(reached[1.0]) - 0.02,
		"Drip barely changes how far the run gets (%.3f at 0.35 vs %.3f at 1.0)"
		% [reached[0.35], reached[1.0]])

	# CONTOUR: the same face over a strong diagonal ridge. A run that ignores the
	# picture falls straight down from the eye; one that follows it leans.
	var straight := await _run(1.0, false)
	var ridged := await _run(1.0, true)
	var lean_flat := _lean(straight)
	var lean_ridge := _lean(ridged)
	print("")
	print("mean horizontal lean of the run: flat frame %+.4f, over a diagonal ridge %+.4f"
		% [lean_flat, lean_ridge])
	_expect(absf(lean_ridge - lean_flat) > 0.008,
		"the run lies in the same place over a flat frame and over a strong ridge "
		+ "(%.4f vs %.4f) - it is ignoring the picture's structure and just falling "
		% [lean_flat, lean_ridge] + "straight down the screen")

	# SHAPE. The three properties a run of paint leaving a wet eye has, and that a
	# per-column lane noise - which is what this used to be - has none of.
	# Measured on the LEFT half only, so the two eyes' runs are never mixed, and at
	# rows found from THE RUN'S OWN EXTENT rather than at fixed heights: where the
	# run starts is measured off the painted eye patch now, so a fixed row is a row
	# at an unknown fraction along it and the taper it reports means nothing.
	var y0 := 1.0
	var y1 := 0.0
	for yy in range(int((EYE_Y + 0.02) * H), H):
		if not _spans(straight, (float(yy) + 0.5) / float(H)).is_empty():
			y0 = minf(y0, (float(yy) + 0.5) / float(H))
			y1 = maxf(y1, (float(yy) + 0.5) / float(H))
	print("")
	print("the left run occupies y %.3f .. %.3f" % [y0, y1])
	var near := _spans(straight, lerpf(y0, y1, 0.10))
	var far := _spans(straight, lerpf(y0, y1, 0.90))
	print("left run: %d span(s) at 10%% down it totalling %.3f wide, "
		% [near.size(), _width(near)] + "%d span(s) at 90%% down it totalling %.3f"
		% [far.size(), _width(far)])
	# TAPER - wide at the source, one thin streak at the tail. A lane bundle is the
	# same hairline all the way down, so its ratio is about 1.
	var taper := _width(near) / maxf(_width(far), 1e-4)
	print("  taper (near width / far width): %.2fx" % taper)
	_expect(_width(far) > 0.001, "the left run has no material 90% of the way down it")
	_expect(taper > 2.0,
		"the run is %.2fx as wide at the eye as it is down the cheek - it does not "
		% taper + "taper, it is a streak of constant thickness")
	# SINGLE - a pool drains at one point. Counted at the WIDE end, which is where
	# the reported four-or-five parallel threads lived; down at the tail the run is
	# thinner than the comb's own spacing and a count there cannot see the fault.
	# ONE, exactly. Verified by injecting a second rivulet a socket-width inboard,
	# which is the shape the old lane noise drew; a comb finer than that cannot be
	# used to check it, because the run is only about six texels wide at its widest
	# here and four streaks do not fit inside six texels at any spacing.
	_expect(near.size() <= 1,
		"the left run leaves the socket as %d separate streaks - a pool drains at "
		% near.size() + "its lowest point, it does not fan into a comb")
	# OUTBOARD - it must CURVE, drifting further outboard as it descends, rather
	# than falling straight down past the nose and the lips. Measured as the shift
	# between the two rows, not against the eye: the drain point is already outside
	# the eye's centre, so a straight-down run scores 0.028 against the eye and
	# would pass a test written that way (verified - it did).
	if not far.is_empty() and not near.is_empty():
		var cx_far := _centre(far)
		var cx_near := _centre(near)
		print("  the run leaves the socket at x=%.3f and is at x=%.3f down the cheek: "
			% [cx_near, cx_far] + "%+.3f OUTBOARD over that descent" % (cx_near - cx_far))
		_expect(cx_far < cx_near - 0.020,
			"the run is at x=%.3f where it leaves the socket and x=%.3f down the cheek "
			% [cx_near, cx_far] + "- it is falling straight toward the nose and the lips "
			+ "instead of curving out across the cheek")

	# EYEBALL - nothing of the run may land inside the ring's opening.
	var worst := 0.0
	var worst_at := Vector2.ZERO
	for y in range(int((EYE_Y - 0.045) * H), int((EYE_Y + 0.045) * H)):
		for x in range(W):
			var u := (float(x) + 0.5) / float(W)
			var v := (float(y) + 0.5) / float(H)
			for ex in [CX - 0.10, CX + 0.10]:
				# THE CENTRAL HALF of the opening. The synthetic hole is about four
				# texels tall, so a sample taken 15% inside its rim is under a pixel
				# from the ring and reads the ring through bilinear filtering - which
				# it did, at 0.25, and that is not paint on the eyeball. The middle of
				# the opening is the eyeball, and is what the claim is about.
				if Vector2((u - ex) / 0.055, (v - EYE_Y) / 0.030).length() < HOLE * 0.55:
					if straight.get_pixel(x, y).r > worst:
						worst = straight.get_pixel(x, y).r
						worst_at = Vector2(u, v)
	print("")
	print("strongest paint inside the eye openings: %.3f (at %.3f, %.3f)"
		% [worst, worst_at.x, worst_at.y])
	_expect(worst < 0.20,
		"the run puts %.2f of paint inside the ring's opening - it is spilling over "
		% worst + "the eyeball, which is the one boundary the eye mask exists to keep")

	print("")
	if _fails.is_empty():
		print("clown_drip_check: PASS - the run reaches the chin, tapers, runs outboard, "
			+ "bends with the face and keeps off the eyeball.")
		quit(0)
	else:
		for f in _fails:
			print("clown_drip_check: FAIL - ", f)
		quit(1)


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


## The painted spans across one scanline of the LEFT half, as (x0, x1) in uv.
## Gaps of a couple of pixels are bridged - the raggedness in the run's own width
## can pinch it to nothing for a texel, and that is not a second streak.
func _spans(f: Image, uy: float) -> Array:
	var y := clampi(int(uy * float(H)), 0, H - 1)
	var out: Array = []
	var run_x0 := -1
	var gap := 0
	for x in range(int(W * 0.10), int(W * (CX - 0.005))):
		if f.get_pixel(x, y).r > 0.25:
			if run_x0 < 0:
				run_x0 = x
			gap = 0
		elif run_x0 >= 0:
			gap += 1
			if gap > 3:
				out.append(Vector2(float(run_x0) / float(W), float(x - gap) / float(W)))
				run_x0 = -1
	if run_x0 >= 0:
		out.append(Vector2(float(run_x0) / float(W), CX - 0.005))
	return out


## The midpoint of the painted material on a scanline, across all its spans.
func _centre(spans: Array) -> float:
	var acc := 0.0
	var w := 0.0
	for sp in spans:
		var s2: Vector2 = sp
		acc += (s2.x + s2.y) * 0.5 * (s2.y - s2.x)
		w += s2.y - s2.x
	return acc / maxf(w, 1e-4)


func _width(spans: Array) -> float:
	var w := 0.0
	for s in spans:
		w += s.y - s.x
	return w


## How far the painted run sits from the face's centre line, averaged down the
## cheek - the quantity a contour-following march changes and a vertical one does
## not.
func _lean(f: Image) -> float:
	var acc := 0.0
	var n := 0
	for y in range(int((EYE_Y + 0.06) * H), int((CY + RY) * H)):
		for x in range(int(W * 0.15), int(W * 0.85)):
			var v := f.get_pixel(x, y).r
			if v > 0.25:
				acc += ((float(x) + 0.5) / float(W)) - CX
				n += 1
	return acc / maxf(float(n), 1.0)


func _run(drip: float, ridge: bool) -> Image:
	var sten := _stencil()
	var frame := _frame(ridge)
	var vps: Array = []
	var rects: Array = []
	for i in 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(W, H)
		vp.disable_3d = true
		vp.use_hdr_2d = true
		vp.transparent_bg = true
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
		var r := ColorRect.new()
		r.size = Vector2(W, H)
		var mm := ShaderMaterial.new()
		mm.shader = load("res://shaders/clown_paint.gdshader")
		r.material = mm
		vp.add_child(r)
		root.add_child(vp)
		vps.append(vp)
		rects.append(r)
	var ping := 0
	for step in 40:
		var mm: ShaderMaterial = rects[ping].material
		mm.set_shader_parameter("u_prev", vps[1 - ping].get_texture())
		mm.set_shader_parameter("u_frame", frame)
		mm.set_shader_parameter("u_stencil", sten)
		mm.set_shader_parameter("u_has_stencil", 1.0)
		mm.set_shader_parameter("u_coat_feather", 0.012)
		mm.set_shader_parameter("u_drip", drip)
		# WOUND UP, both of them. What is under test is the shape LAW - does the run
		# taper, stay single, curve out - and at 320px the default lean streak is a
		# two-pixel thread whose width cannot be measured to better than 50%.
		mm.set_shader_parameter("u_drip_w", 1.15)
		mm.set_shader_parameter("u_drip_curve", 1.4)
		mm.set_shader_parameter("u_dt", 0.033)
		mm.set_shader_parameter("u_reset", 1)   # the deposit is what is under test
		mm.set_shader_parameter("u_time", 3.0)
		mm.set_shader_parameter("u_aspect", 1.0)
		mm.set_shader_parameter("u_face_lum", 0.6)
		mm.set_shader_parameter("u_face_red", 0.12)
		mm.set_shader_parameter("u_face_c", Vector2(CX, CY))
		mm.set_shader_parameter("u_face_r", Vector2(RX, RY))
		mm.set_shader_parameter("u_eye_l", Vector2(CX - 0.10, EYE_Y))
		mm.set_shader_parameter("u_eye_r", Vector2(CX + 0.10, EYE_Y))
		mm.set_shader_parameter("u_mouth", Vector2(CX, 0.58))
		mm.set_shader_parameter("u_nose", Vector2(CX, 0.47))
		mm.set_shader_parameter("u_eye_lr", 0.035)
		mm.set_shader_parameter("u_eye_rr", 0.035)
		mm.set_shader_parameter("u_mouth_r", Vector2(0.05, 0.03))
		mm.set_shader_parameter("u_scale", 1.0)
		mm.set_shader_parameter("u_evidence", 0.0)
		mm.set_shader_parameter("u_settle", 0.2)
		mm.set_shader_parameter("u_bleed", 0.0)
		mm.set_shader_parameter("u_hollow", 0.0)
		vps[ping].render_target_update_mode = SubViewport.UPDATE_ONCE
		await process_frame
		ping = 1 - ping
	var img: Image = vps[1 - ping].get_texture().get_image()
	for vp in vps:
		vp.queue_free()
	await process_frame
	return img


func _stencil() -> Texture2D:
	var img := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var u := (float(x) + 0.5) / float(W)
			var v := (float(y) + 0.5) / float(H)
			var a := 0.0
			var e := 0.0
			if Vector2((u - CX) / RX, (v - CY) / RY).length() < 1.0:
				a = 1.0
			# A RING, like the real stencil: the editor draws the eye patch as an
			# annulus whose opening already clears the eyeball at the stored Hollow.
			# A solid patch here would let a run paint the eyeball unnoticed.
			for ex in [CX - 0.10, CX + 0.10]:
				var d := Vector2((u - ex) / 0.055, (v - EYE_Y) / 0.030).length()
				if d < 1.0 and d > HOLE:
					e = 1.0
			img.set_pixel(x, y, Color(e, 0, 0, a))
	return ImageTexture.create_from_image(img)


## Flat skin, or skin with a strong diagonal ridge running across it - a stand-in
## for a cheekbone, and the only thing that differs between the two runs.
func _frame(ridge: bool) -> Texture2D:
	var img := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var u := (float(x) + 0.5) / float(W)
			var v := (float(y) + 0.5) / float(H)
			var l := 0.0
			if ridge:
				# A diagonal SAWTOOTH, steep and repeating. Two earlier versions
				# failed to test anything and both for measurable reasons: a narrow
				# ridge only deflects the hairline of the run that lies on it, and
				# a gentle full-frame ramp has a gradient of 0.0018 across the
				# march's 0.0072 tap spacing - below the deflection's own knee, so
				# nothing fired. A real cheekbone moves luminance about 0.2 over
				# 0.03 of the frame, which is 0.05 across that spacing. The
				# sawtooth reaches that magnitude everywhere AND keeps its tangent
				# pointing one way, so the deflection accumulates instead of
				# cancelling.
				l = 0.45 * fposmod((u - v) * 10.0, 1.0)
			img.set_pixel(x, y, Color(0.55 + l, 0.42 + l * 0.8, 0.36 + l * 0.7))
	return ImageTexture.create_from_image(img)
