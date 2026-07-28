extends SceneTree

## One-shot GPU compile check for mask_split.gdshader's clown branch (effect 16):
##   godot --path axis/ghost --headless --script res://tests/clown_shader_check.gd
## `--editor --quit` only validates GDScript - a .gdshader edit needs an actual
## compile with realistic values on every uniform the branch reads (CLAUDE.md's
## validation discipline). The caller judges the output by grepping for
## SHADER ERROR / Invalid; the one known-harmless line is crystal's
## custom_samplers complaint (a function-parameter sampler - not a bug).

func _initialize() -> void:
	var shader: Shader = load("res://shaders/mask_split.gdshader")
	if shader == null:
		push_error("could not load mask_split.gdshader")
		quit(1)
		return
	var mat := ShaderMaterial.new()
	mat.shader = shader
	# Layer 0 = clown, layer 1 = crystal (regression: the neighbour branch still
	# compiles beside it). Arrays at FULL declared length - a short uniform array
	# is silently dropped.
	mat.set_shader_parameter("u_l_count", 2)
	mat.set_shader_parameter("u_l_hue", PackedFloat32Array([0.02, 0.47, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_effect", PackedInt32Array([16, 6, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_w", PackedFloat32Array([0.9, 0.6, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_off", PackedVector2Array([
		Vector2(0.05, -0.1), Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO]))
	mat.set_shader_parameter("u_l_scale", PackedFloat32Array([1.0, 0.8, 1, 1, 1, 1]))
	mat.set_shader_parameter("u_l_dens", PackedFloat32Array([0.45, 0.2, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_con", PackedFloat32Array([0.6, 0.8, 0.5, 0.5, 0.5, 0.5]))
	mat.set_shader_parameter("u_l_glow", PackedFloat32Array([1, 1, 1, 1, 1, 1]))
	mat.set_shader_parameter("u_l_speed", PackedFloat32Array([1, 1, 1, 1, 1, 1]))
	mat.set_shader_parameter("u_l_smooth", PackedFloat32Array([0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_lagf", PackedFloat32Array([0.35, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_stick", PackedFloat32Array([0, 0, 0, 0, 0, 0]))
	var tdirs := PackedVector3Array()
	for i in 6:
		tdirs.append(Vector3(0.7, -0.5, -0.2).normalized())
	mat.set_shader_parameter("u_l_tdir", tdirs)
	var ew := PackedFloat32Array()
	ew.resize(48)
	for l in 6:
		ew[l * 8] = 1.0
	mat.set_shader_parameter("u_l_ew", ew)
	mat.set_shader_parameter("u_l_elag", PackedInt32Array([0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_tint", PackedFloat32Array([0.3, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_anchor", Vector2(0.52, 0.44))
	mat.set_shader_parameter("u_anchor_scale", 0.24)
	mat.set_shader_parameter("u_clown_eye_l", Vector2(0.43, 0.41))
	mat.set_shader_parameter("u_clown_eye_r", Vector2(0.57, 0.39))
	mat.set_shader_parameter("u_clown_mouth", Vector2(0.51, 0.60))
	mat.set_shader_parameter("u_clown_face_r", Vector2(0.13, 0.19))
	mat.set_shader_parameter("u_clown_tint", Vector3(0.5, 0.2, -0.45))
	mat.set_shader_parameter("u_clown_lum", 0.42)
	mat.set_shader_parameter("u_clown_eye_lr", 0.04)
	mat.set_shader_parameter("u_clown_eye_rr", 0.033)
	mat.set_shader_parameter("u_clown_mouth_r", Vector2(0.05, 0.03))
	mat.set_shader_parameter("u_clown_face_c", Vector2(0.5, 0.44))
	# u_clown_paint stays at its default-black fallback here - this check is
	# about compilation, and the branch must survive an empty field anyway.
	mat.set_shader_parameter("u_time", 3.7)
	mat.set_shader_parameter("u_aspect", 1.7778)

	var rect := ColorRect.new()
	rect.material = mat
	rect.size = Vector2(320, 180)
	root.add_child(rect)

	# The liquid-paint sim shader compiles under realistic uniforms too.
	var sim: Shader = load("res://shaders/clown_paint.gdshader")
	var smat := ShaderMaterial.new()
	smat.shader = sim
	smat.set_shader_parameter("u_dt", 0.033)
	smat.set_shader_parameter("u_reset", 0)
	smat.set_shader_parameter("u_time", 2.4)
	smat.set_shader_parameter("u_aspect", 1.7778)
	smat.set_shader_parameter("u_eye_l", Vector2(0.43, 0.41))
	smat.set_shader_parameter("u_eye_r", Vector2(0.57, 0.39))
	smat.set_shader_parameter("u_mouth", Vector2(0.51, 0.60))
	smat.set_shader_parameter("u_eye_lr", 0.04)
	smat.set_shader_parameter("u_eye_rr", 0.033)
	smat.set_shader_parameter("u_mouth_r", Vector2(0.05, 0.03))
	smat.set_shader_parameter("u_nose", Vector2(0.505, 0.50))
	smat.set_shader_parameter("u_face_c", Vector2(0.5, 0.44))
	smat.set_shader_parameter("u_face_r", Vector2(0.13, 0.19))
	smat.set_shader_parameter("u_scale", 1.0)
	var srect := ColorRect.new()
	srect.material = smat
	srect.size = Vector2(256, 144)
	root.add_child(srect)

	for i in 6:
		await process_frame
	print("CLOWN_SHADER_CHECK_DONE")
	quit()
