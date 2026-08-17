extends SceneTree

## Does the clown's paint TRAVEL WITH THE FACE, or sit on the screen?
##   godot --path axis/ghost --position -5000,-5000 --script res://tests/clown_anchor_check.gd
## Needs a real renderer (the paint field is a ping-pong SubViewport pair).
##
## Two things here are patterns rather than placed features - the eye-drip's
## rivulets and (in mask_split) the craquelure - and both were first written in
## SCREEN space. A pattern keyed on UV does not move when the head does, so the
## head slides through it: reported for the cracks as "squiggly smears", and for
## the drip as "the streaks remain fixed, rooted in place, and more streaks pop in
## from nowhere". Both are the same bug and the same fix - evaluate the pattern in
## the FACE's own frame (origin at its centre, x along the eye line, scaled by the
## eye separation) so it travels, rotates and resizes with the head.
##
## The test: render the SAME face at two positions and ask whether the paint moved
## with it. A face-anchored pattern shifts by exactly the amount the face shifted;
## a screen-anchored one stays where it was. Comparing the second render against
## the first SHIFTED BY THE SAME OFFSET is the whole measurement - it should match
## if the paint is anchored, and mismatch if it is not.

const W := 320
const H := 320
const SHIFT := 0.10          # how far the face moves between the two renders
const CX := 0.42
const CY := 0.46
const RX := 0.26
const RY := 0.34

var _fails: PackedStringArray = []


func _initialize() -> void:
	var a := await _run(0.0)
	var b := await _run(SHIFT)
	# Compare b against a shifted by SHIFT. Only the drip band is examined - below
	# the eyes, where the rivulets live - and only inside the face, so the
	# comparison is about the pattern rather than about the silhouette.
	var dx := int(round(SHIFT * float(W)))
	var matched := 0.0
	var moved := 0.0
	var n := 0
	for y in range(int(H * 0.50), int(H * 0.80)):
		for x in range(int(W * 0.12), int(W * 0.62)):
			var va := a.get_pixel(x, y).r
			var vb_anchored := b.get_pixel(mini(x + dx, W - 1), y).r
			var vb_static := b.get_pixel(x, y).r
			if maxf(va, maxf(vb_anchored, vb_static)) < 0.02:
				continue
			matched += absf(va - vb_anchored)
			moved += absf(va - vb_static)
			n += 1
	if n == 0:
		_expect(false, "no drip was produced at all - nothing to judge")
	else:
		matched /= float(n)
		moved /= float(n)
		print("drip over %d pixels: mismatch when the comparison FOLLOWS the face %.4f, "
			% [n, matched] + "when it stays on screen %.4f" % moved)
		# If the pattern is anchored to the face, following the face is the better
		# match by a clear margin. If it is anchored to the screen, the two are
		# comparable or reversed.
		_expect(matched < moved * 0.75,
			"the drip matches the screen (%.4f) about as well as it matches the FACE "
			% moved + "(%.4f) - the rivulets are keyed on frame position, so the head "
			% matched + "slides through them instead of carrying them")

	print("")
	if _fails.is_empty():
		print("clown_anchor_check: PASS - the drip travels with the face.")
		quit(0)
	else:
		for f in _fails:
			print("clown_anchor_check: FAIL - ", f)
		quit(1)


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


## The same face, offset by `dx` in UV. Everything else is identical, so any
## difference in the paint is the anchoring and nothing else.
func _run(dx: float) -> Image:
	var cx := CX + dx
	var sten := _stencil(cx)
	var frame := _frame()
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
		mm.set_shader_parameter("u_drip", 0.9)
		mm.set_shader_parameter("u_drip_w", 1.15)   # enough pixels to compare
		mm.set_shader_parameter("u_drip_curve", 1.4)
		mm.set_shader_parameter("u_dt", 0.033)
		# RESET every step. The sim's own advection and decay would otherwise carry
		# paint from before the move and blur the question; what is under test is
		# where the deposit PUTS the rivulets, not how the liquid then flows.
		mm.set_shader_parameter("u_reset", 1)
		mm.set_shader_parameter("u_time", 3.0)
		mm.set_shader_parameter("u_aspect", 1.0)
		mm.set_shader_parameter("u_face_lum", 0.6)
		mm.set_shader_parameter("u_face_red", 0.12)
		mm.set_shader_parameter("u_face_c", Vector2(cx, CY))
		mm.set_shader_parameter("u_face_r", Vector2(RX, RY))
		mm.set_shader_parameter("u_eye_l", Vector2(cx - 0.10, 0.40))
		mm.set_shader_parameter("u_eye_r", Vector2(cx + 0.10, 0.40))
		mm.set_shader_parameter("u_mouth", Vector2(cx, 0.62))
		mm.set_shader_parameter("u_nose", Vector2(cx, 0.51))
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


## Two eye patches and a face oval, at `cx`. The drip reads the eye channel and
## runs downward from it, so the eyes are what it needs.
func _stencil(cx: float) -> Texture2D:
	var img := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var u := (float(x) + 0.5) / float(W)
			var v := (float(y) + 0.5) / float(H)
			var a := 0.0
			var e := 0.0
			if Vector2((u - cx) / RX, (v - CY) / RY).length() < 1.0:
				a = 1.0
			for ex in [cx - 0.10, cx + 0.10]:
				if Vector2((u - ex) / 0.055, (v - 0.40) / 0.030).length() < 1.0:
					e = 1.0
			img.set_pixel(x, y, Color(e, 0, 0, a))
	return ImageTexture.create_from_image(img)


func _frame() -> Texture2D:
	var img := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	img.fill(Color(0.78, 0.60, 0.50))
	return ImageTexture.create_from_image(img)
