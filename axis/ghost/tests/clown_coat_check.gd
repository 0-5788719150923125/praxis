extends SceneTree

## Does the clown's white coat actually reach the edge of the face it is given?
##   godot --path axis/ghost --position -5000,-5000 --script res://tests/clown_coat_check.gd
## Needs a real renderer (the paint field is a ping-pong SubViewport pair).
##
## Reported from a real session: "the mask no longer tracks to the very edge of my
## face - there is always a region where the paint will not draw", and the only
## control that touched it was Settle, which made the band WIDER. Both halves of
## that are one mechanism. The paint sim advects each channel toward its own
## attractor, and advection samples BACKWARD along the flow, so at the coat's own
## outer boundary it reads from just outside itself - where there is no paint. The
## deposit refills that from zero every step and never catches up, so a permanent
## unpainted rim sits at the edge, as deep as the refill is slow. Settle IS the
## refill rate. Measured against the jawline before the fix: 3.7% of frame width
## at the default Settle, 13% at the top of the slider.
##
## The fix is that the coat takes its deposit as a FLOOR - it is a mask (a
## statement of where the face is), not paint that may drift off the edge of it.
## The features stay erodable, because Hollow and the evidence terms need them to
## be. So this checks the invariant that fix establishes:
##
##   COVERAGE     the coat fills the whole shape it was given, to the boundary.
##   INDIFFERENCE and it does so at EVERY Settle - a coat whose coverage moves
##                with a timing knob is the bug, whatever the coverage is.
##
## Driven by a synthetic stencil rather than a real face, so it needs no landmark
## track, no venv and no clip: the claim is about the sim, not the detector.

const W := 256
const H := 256
const CX := 0.5
const CY := 0.5
const RX := 0.30
const RY := 0.38

var _fails: PackedStringArray = []


func _initialize() -> void:
	var sten := _stencil()
	var frame := _frame()
	var seen := {}
	for settle in [0.0, 0.35, 1.0]:
		for bleed in [0.0, 1.0]:
			var field := await _sim(frame, sten, settle, bleed)
			var res := _measure(field)
			seen["%.2f/%.1f" % [settle, bleed]] = res.cover
			print("Settle %.2f Bleed %.1f -> coat covers %.1f%% of the shape, "
				% [settle, bleed, res.cover * 100.0]
				+ "reaches %+.3f of the way past its edge" % res.reach)
			# The shape it was handed must be filled. Not "mostly" - a ring of
			# unpainted face is exactly what this exists to catch.
			_expect(res.cover > 0.985,
				"Settle %.2f Bleed %.1f: the coat covers only %.1f%% of the shape "
				% [settle, bleed, res.cover * 100.0]
				+ "it was given - there is an unpainted band inside the boundary")

	# INDIFFERENCE TO SETTLE. This is the sharp assertion: before the fix the
	# coverage fell away as Settle rose, and any future change that reintroduces
	# transport at the boundary will show up here first, even if every individual
	# coverage number still scrapes past the bar above.
	var lo := 1.0
	var hi := 0.0
	for k in seen:
		lo = minf(lo, seen[k])
		hi = maxf(hi, seen[k])
	print("")
	print("coverage across every Settle/Bleed: %.1f%% .. %.1f%%" % [lo * 100.0, hi * 100.0])
	_expect(hi - lo < 0.01,
		"coverage moves with the timing knobs (%.1f%% to %.1f%%) - the coat is "
		% [lo * 100.0, hi * 100.0] + "still being eroded at its boundary and refilled at a rate Settle sets")

	print("")
	if _fails.is_empty():
		print("clown_coat_check: PASS - the coat fills its shape to the edge, at every ",
			"Settle and Bleed.")
		quit(0)
	else:
		for f in _fails:
			print("clown_coat_check: FAIL - ", f)
		quit(1)


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


## How much of the given shape the coat channel actually filled, and how far past
## the shape's own edge it reached along the widest row.
func _measure(field: Image) -> Dictionary:
	var inside := 0
	var filled := 0
	for y in field.get_height():
		for x in field.get_width():
			var u := (float(x) + 0.5) / float(field.get_width())
			var v := (float(y) + 0.5) / float(field.get_height())
			if Vector2((u - CX) / RX, (v - CY) / RY).length() > 1.0:
				continue
			inside += 1
			if field.get_pixel(x, y).a > 0.5:
				filled += 1
	# The widest row: where the boundary is most nearly vertical and a rim shows
	# most clearly.
	var mid := int(CY * float(field.get_height()))
	var edge := -1
	for x in range(field.get_width() - 1, 0, -1):
		if field.get_pixel(x, mid).a > 0.5:
			edge = x
			break
	var reach := 0.0
	if edge >= 0:
		reach = ((float(edge) + 0.5) / float(field.get_width()) - (CX + RX)) / RX
	return {"cover": float(filled) / maxf(float(inside), 1.0), "reach": reach}


## The face shape, in the coat's channel, exactly as _update_stencil writes it.
func _stencil() -> Texture2D:
	var img := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var u := (float(x) + 0.5) / float(W)
			var v := (float(y) + 0.5) / float(H)
			var d := Vector2((u - CX) / RX, (v - CY) / RY).length()
			# Antialiased like a rasterized polygon, so the check faces the same
			# soft boundary the real stencil has.
			var a: float = clampf((1.02 - d) / 0.02, 0.0, 1.0)
			img.set_pixel(x, y, Color(0, 0, 0, a))
	return ImageTexture.create_from_image(img)


## A plain lit surface: the coat's deposit is gated on nothing but the stencil, so
## the frame only has to be present and face-coloured.
func _frame() -> Texture2D:
	var img := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	img.fill(Color(0.78, 0.60, 0.50))
	return ImageTexture.create_from_image(img)


func _sim(frame: Texture2D, sten: Texture2D, settle: float, bleed: float) -> Image:
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
	for step in 60:
		var mm: ShaderMaterial = rects[ping].material
		mm.set_shader_parameter("u_prev", vps[1 - ping].get_texture())
		mm.set_shader_parameter("u_frame", frame)
		mm.set_shader_parameter("u_stencil", sten)
		mm.set_shader_parameter("u_has_stencil", 1.0)
		mm.set_shader_parameter("u_dt", 0.033)
		mm.set_shader_parameter("u_reset", 1 if step == 0 else 0)
		mm.set_shader_parameter("u_time", float(step) * 0.033)
		mm.set_shader_parameter("u_aspect", 1.0)
		mm.set_shader_parameter("u_face_lum", 0.6)
		mm.set_shader_parameter("u_face_red", 0.12)
		mm.set_shader_parameter("u_face_c", Vector2(CX, CY))
		mm.set_shader_parameter("u_face_r", Vector2(RX, RY))
		mm.set_shader_parameter("u_eye_l", Vector2(0.40, 0.42))
		mm.set_shader_parameter("u_eye_r", Vector2(0.60, 0.42))
		mm.set_shader_parameter("u_mouth", Vector2(0.50, 0.64))
		mm.set_shader_parameter("u_nose", Vector2(0.50, 0.53))
		mm.set_shader_parameter("u_eye_lr", 0.035)
		mm.set_shader_parameter("u_eye_rr", 0.035)
		mm.set_shader_parameter("u_mouth_r", Vector2(0.05, 0.03))
		mm.set_shader_parameter("u_scale", 1.0)
		mm.set_shader_parameter("u_evidence", 0.0)
		mm.set_shader_parameter("u_settle", settle)
		mm.set_shader_parameter("u_bleed", bleed)
		mm.set_shader_parameter("u_hollow", 0.0)
		mm.set_shader_parameter("u_drip", 0.0)
		vps[ping].render_target_update_mode = SubViewport.UPDATE_ONCE
		await process_frame
		ping = 1 - ping
	var img: Image = vps[1 - ping].get_texture().get_image()
	for vp in vps:
		vp.queue_free()
	await process_frame
	return img
