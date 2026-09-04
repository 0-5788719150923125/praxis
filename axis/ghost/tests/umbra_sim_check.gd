extends SceneTree

## Isolated probe for shaders/umbra_field.gdshader - runs the ping-pong field
## by hand against a SYNTHETIC silhouette (a head on a pair of shoulders) thrown
## by a real cast transform, and dumps what each channel actually holds.
##   godot --path axis/ghost --headless --script res://tests/umbra_sim_check.gd
##
## This exists because the clown's equivalent probe (tests/paint_sim_check.gd)
## found two bugs that were completely invisible from the rendered picture:
## a SubViewport forcing alpha to 1 (so the guard permitted everything) and a
## missing blend_disabled. Look here BEFORE trying to debug the visual.
## Writes /tmp/ghost_scratch/umbra_field.png - the field itself, as a picture.

const W := 384
const H := 216

func _initialize() -> void:
	# HER silhouette, at the grid the pose track writes: a head on a pair of
	# shoulders, left of centre. R is the body the throw magnifies, B is the same
	# mask dilated - the guard, read at plain screen uv (see _pt_upload_mask).
	var region := Image.create_empty(96, 54, false, Image.FORMAT_RGBA8)
	for y in 54:
		for x in 96:
			var px := (float(x) + 0.5) / 96.0
			var py := (float(y) + 0.5) / 54.0
			var head := Vector2((px - 0.40) * 1.7778, py - 0.26).length() < 0.13
			var body: bool = py > 0.36 and absf(px - 0.40) < 0.16
			var m := 1.0 if (head or body) else 0.0
			region.set_pixel(x, y, Color(m, 0.0, m, 1.0))
	var rtex := ImageTexture.create_from_image(region)

	# THE CAST TRANSFORM, at a real magnification and pointed back TOWARD her -
	# the guard has to hold with the ghost scaled up and shoved onto the subject,
	# which is exactly when a stencil-shaped bug would show. Built the same way
	# MaskEditor._umb_solve_cast builds it: her shoulder line (0.40, 0.50) and eye
	# line (0.40, 0.24) in aspect-corrected space, thrown to a head at (1.10, 0.16)
	# with scale 2.0 and no rotation, so the inverse is a plain 1/2.
	var asp := 1.7778
	var her_sh := Vector2(0.40 * asp, 0.50)
	var her_eye := Vector2(0.40 * asp, 0.24)
	var unit := (her_eye - her_sh).length()
	var scale := 2.0
	var ghost_head := Vector2(1.10, 0.16)
	var anchor := ghost_head + Vector2(0.0, unit * scale)

	var shader: Shader = load("res://shaders/umbra_field.gdshader")
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
		var rect := ColorRect.new()
		rect.size = Vector2(W, H)
		var m := ShaderMaterial.new()
		m.shader = shader
		rect.material = m
		vp.add_child(rect)
		root.add_child(vp)
		vps.append(vp)
		rects.append(rect)

	var ping := 0
	var t := 0.0
	for step in 45:
		var mat: ShaderMaterial = rects[ping].material
		mat.set_shader_parameter("u_prev", vps[1 - ping].get_texture())
		mat.set_shader_parameter("u_region", rtex)
		mat.set_shader_parameter("u_dt", 0.033)
		mat.set_shader_parameter("u_reset", 1 if step == 0 else 0)
		mat.set_shader_parameter("u_time", t)
		mat.set_shader_parameter("u_aspect", 1.7778)
		mat.set_shader_parameter("u_dir", Vector2(0.99, -0.12))
		mat.set_shader_parameter("u_loom", 0.55)
		mat.set_shader_parameter("u_rise", 1.2)
		mat.set_shader_parameter("u_roil", 0.65)
		mat.set_shader_parameter("u_cling", 0.35)
		mat.set_shader_parameter("u_wisp", 0.5)
		mat.set_shader_parameter("u_inv0", Vector2(1.0 / scale, 0.0))
		mat.set_shader_parameter("u_inv1", Vector2(0.0, 1.0 / scale))
		mat.set_shader_parameter("u_anchor", anchor)
		mat.set_shader_parameter("u_src", her_sh)
		mat.set_shader_parameter("u_eye_src", her_eye)
		mat.set_shader_parameter("u_unit", unit)
		mat.set_shader_parameter("u_bust0", 1.6)
		mat.set_shader_parameter("u_bust1", 3.2)
		# Eyes ON, and the LEFT one deliberately planted right on the subject:
		# the sockets must never become a way for the effect to reach her.
		mat.set_shader_parameter("u_eye_l", Vector2(0.40, 0.26))
		mat.set_shader_parameter("u_eye_r", Vector2(0.66, 0.16))
		mat.set_shader_parameter("u_eye_rad", 0.06)
		mat.set_shader_parameter("u_eye_amt", 0.9)
		vps[ping].render_target_update_mode = SubViewport.UPDATE_ONCE
		await process_frame
		await process_frame
		ping = 1 - ping
		t += 0.033

	var img: Image = vps[1 - ping].get_texture().get_image()
	print("field %dx%d fmt=%d" % [img.get_width(), img.get_height(), img.get_format()])
	var probes := {
		"ghost head    ": Vector2(1.10 / 1.7778, 0.16),
		"ghost neck    ": Vector2(1.10 / 1.7778, 0.36),
		"ghost chest   ": Vector2(1.10 / 1.7778, 0.62),
		"ON SUBJECT    ": Vector2(0.40, 0.26),
		"subject body  ": Vector2(0.40, 0.60),
		"far background": Vector2(0.08, 0.85),
	}
	var worst := 0.0
	var under := 0.0
	for k in probes:
		var p: Vector2 = probes[k]
		var c: Color = img.get_pixel(int(p.x * float(W)), int(p.y * float(H)))
		print("  %s mass=%.3f essence=%.3f guard=%.3f" % [k, c.r, c.g, c.a])
		if k.begins_with("ON SUBJECT"):
			worst = maxf(c.r * c.a, c.g * c.a)
			under = c.r
	# Peak, so "is there any mass at all" is answerable without hunting probes.
	# Scanned on a downsampled COPY - 83k get_pixel calls in GDScript is slower
	# than the entire simulation it is measuring.
	var small: Image = img.duplicate()
	small.resize(96, 54, Image.INTERPOLATE_BILINEAR)
	var pk := 0.0
	var pkw := 0.0
	for y in 54:
		for x in 96:
			var c2: Color = small.get_pixel(x, y)
			pk = maxf(pk, c2.r)
			pkw = maxf(pkw, c2.g)
	print("  peak mass=%.3f peak essence=%.3f" % [pk, pkw])
	print("  SUBJECT LEAK (must be ~0): %.4f" % worst)
	# ...AND THE MASS IS STILL THERE UNDERNEATH HER, which is the other half of
	# the same change and cannot be inferred from the leak being zero - a field
	# that deposited nothing at all would also read 0.0000. The mass grows under
	# her so that the instant she moves off a pixel it is already full, instead of
	# refilling at Cling's rise time and dragging a hole behind every gesture; the
	# guard is what stops it being DRAWN, and that is what the leak measures.
	print("  MASS UNDER HER (must be > 0, gated only at render): %.4f" % under)
	if under <= 0.05:
		print("  WARNING: nothing is growing under the subject - a gesture will "
			+ "leave a lagging hole in the mass again")
	DirAccess.make_dir_recursive_absolute("/tmp/ghost_scratch")
	img.convert(Image.FORMAT_RGBA8)
	img.save_png("/tmp/ghost_scratch/umbra_field.png")
	print("UMBRA_SIM_CHECK_DONE")
	quit()
