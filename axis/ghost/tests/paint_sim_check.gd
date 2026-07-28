extends SceneTree

## One-shot probe of the clown liquid-paint sim (clown_paint.gdshader) in
## isolation: builds the same ping-pong SubViewport pair MaskEditor does,
## steps it ~40 frames with a fixed face model, then dumps the field and a
## few landmark samples. Needs a real renderer - run WINDOWED offscreen:
##   godot --path axis/ghost --position -5000,-5000 --script res://tests/paint_sim_check.gd

func _initialize() -> void:
	var vps: Array = []
	var rects: Array = []
	for i in 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(256, 144)
		vp.disable_3d = true
		vp.use_hdr_2d = true
		vp.transparent_bg = true   # the coat channel IS alpha - see MaskEditor._ensure_paint_sim
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
		var r := ColorRect.new()
		r.size = Vector2(256, 144)
		var m := ShaderMaterial.new()
		m.shader = load("res://shaders/clown_paint.gdshader")
		r.material = m
		vp.add_child(r)
		root.add_child(vp)
		vps.append(vp)
		rects.append(r)
	var ping := 0
	for step in 40:
		var m: ShaderMaterial = rects[ping].material
		m.set_shader_parameter("u_prev", vps[1 - ping].get_texture())
		m.set_shader_parameter("u_dt", 0.033)
		m.set_shader_parameter("u_reset", 1 if step == 0 else 0)
		m.set_shader_parameter("u_time", float(step) * 0.033)
		m.set_shader_parameter("u_aspect", 1.7778)
		m.set_shader_parameter("u_eye_l", Vector2(0.43, 0.41))
		m.set_shader_parameter("u_eye_r", Vector2(0.57, 0.39))
		m.set_shader_parameter("u_mouth", Vector2(0.51, 0.60))
		m.set_shader_parameter("u_nose", Vector2(0.5, 0.5))
		m.set_shader_parameter("u_face_c", Vector2(0.5, 0.45))
		m.set_shader_parameter("u_face_r", Vector2(0.14, 0.20))
		m.set_shader_parameter("u_eye_lr", 0.04)
		m.set_shader_parameter("u_eye_rr", 0.033)
		m.set_shader_parameter("u_mouth_r", Vector2(0.05, 0.03))
		m.set_shader_parameter("u_scale", 1.0)
		vps[ping].render_target_update_mode = SubViewport.UPDATE_ONCE
		await process_frame
		ping = 1 - ping
	var img: Image = vps[1 - ping].get_texture().get_image()
	img.save_png("/tmp/ghost_scratch/paint_field.png")
	var corner := img.get_pixel(5, 5)
	var wallL := img.get_pixel(20, 70)
	var center := img.get_pixel(128, 65)
	var eye := img.get_pixel(int(0.43 * 256.0), int(0.41 * 144.0))
	var mouth := img.get_pixel(int(0.51 * 256.0), int(0.60 * 144.0))
	print("PAINT corner=", corner, " wallL=", wallL, " center=", center,
		" eyeL=", eye, " mouth=", mouth)
	quit()
