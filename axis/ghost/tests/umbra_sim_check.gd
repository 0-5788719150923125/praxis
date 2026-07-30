extends SceneTree

## Isolated probe for shaders/umbra_field.gdshader - runs the ping-pong field
## by hand against a SYNTHETIC region (a soft blob of "linked shadow" to the
## right of a "subject" rectangle) and dumps what each channel actually holds.
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
	var region := Image.create_empty(96, 54, false, Image.FORMAT_RGBA8)
	for y in 54:
		for x in 96:
			var px := (float(x) + 0.5) / 96.0
			var py := (float(y) + 0.5) / 54.0
			# subject: a slab down the middle-left, like her head and shoulders
			var subj := 1.0 if (px > 0.30 and px < 0.58 and py > 0.15) else 0.0
			# linked shadow: a blob just to its right, like the cast shadow
			var d := Vector2((px - 0.70) * 1.2, py - 0.40).length()
			var shad := 1.0 if (d < 0.16 and subj < 0.5) else 0.0
			region.set_pixel(x, y, Color(shad, shad * 0.85, subj, 1.0))
	var rtex := ImageTexture.create_from_image(region)

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
		mat.set_shader_parameter("u_scale", 1.2)
		vps[ping].render_target_update_mode = SubViewport.UPDATE_ONCE
		await process_frame
		await process_frame
		ping = 1 - ping
		t += 0.033

	var img: Image = vps[1 - ping].get_texture().get_image()
	print("field %dx%d fmt=%d" % [img.get_width(), img.get_height(), img.get_format()])
	var probes := {
		"shadow core   ": Vector2(0.70, 0.40),
		"shadow above  ": Vector2(0.70, 0.24),
		"loom outward  ": Vector2(0.86, 0.38),
		"ON SUBJECT    ": Vector2(0.45, 0.40),
		"subject edge  ": Vector2(0.57, 0.40),
		"far background": Vector2(0.10, 0.80),
		"above shadow  ": Vector2(0.70, 0.10),
	}
	var worst := 0.0
	for k in probes:
		var p: Vector2 = probes[k]
		var c: Color = img.get_pixel(int(p.x * float(W)), int(p.y * float(H)))
		print("  %s mass=%.3f essence=%.3f guard=%.3f" % [k, c.r, c.g, c.a])
		if k.begins_with("ON SUBJECT"):
			worst = maxf(c.r * c.a, c.g * c.a)
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
	DirAccess.make_dir_recursive_absolute("/tmp/ghost_scratch")
	img.convert(Image.FORMAT_RGBA8)
	img.save_png("/tmp/ghost_scratch/umbra_field.png")
	print("UMBRA_SIM_CHECK_DONE")
	quit()
