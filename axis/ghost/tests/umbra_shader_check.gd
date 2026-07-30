extends SceneTree

## One-shot GPU compile check for the umbra effect - mask_split.gdshader's
## effect-17 branch and shaders/umbra_field.gdshader:
##   godot --path axis/ghost --headless --script res://tests/umbra_shader_check.gd
## `--editor --quit` only validates GDScript; a .gdshader edit needs a real
## compile with realistic values on every uniform the branch reads (CLAUDE.md's
## validation discipline). The caller judges by grepping for SHADER ERROR /
## Invalid; the one known-harmless line is crystal's custom_samplers complaint.
##
## Layer 1 is deliberately the CLOWN branch: both effects sample their own
## field texture, and this is the check that would catch the mask shader
## running past a per-GPU sampler limit once a second field was added.

func _initialize() -> void:
	var shader: Shader = load("res://shaders/mask_split.gdshader")
	if shader == null:
		push_error("could not load mask_split.gdshader")
		quit(1)
		return
	var mat := ShaderMaterial.new()
	mat.shader = shader
	mat.set_shader_parameter("u_l_count", 2)
	mat.set_shader_parameter("u_l_hue", PackedFloat32Array([0.50, 0.02, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_effect", PackedInt32Array([17, 16, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_w", PackedFloat32Array([0.85, 0.5, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_off", PackedVector2Array([
		Vector2(0.04, -0.02), Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO]))
	mat.set_shader_parameter("u_l_scale", PackedFloat32Array([1.2, 1.0, 1, 1, 1, 1]))
	mat.set_shader_parameter("u_l_dens", PackedFloat32Array([0.55, 0.2, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_con", PackedFloat32Array([0.65, 0.5, 0.5, 0.5, 0.5, 0.5]))
	mat.set_shader_parameter("u_l_glow", PackedFloat32Array([1, 1, 1, 1, 1, 1]))
	mat.set_shader_parameter("u_l_speed", PackedFloat32Array([1.2, 1, 1, 1, 1, 1]))
	mat.set_shader_parameter("u_l_smooth", PackedFloat32Array([0.4, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_lagf", PackedFloat32Array([0.35, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_stick", PackedFloat32Array([0.7, 0, 0, 0, 0, 0]))
	var tdirs := PackedVector3Array()
	for i in 6:
		tdirs.append(Vector3(-0.82, 0.33, 0.47).normalized())
	mat.set_shader_parameter("u_l_tdir", tdirs)
	var ew := PackedFloat32Array()
	ew.resize(48)
	for l in 6:
		ew[l * 8] = 1.0
	mat.set_shader_parameter("u_l_ew", ew)
	mat.set_shader_parameter("u_l_elag", PackedInt32Array([0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_tint", PackedFloat32Array([0.15, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_anchor", Vector2(0.5, 0.45))
	mat.set_shader_parameter("u_anchor_scale", 0.24)
	mat.set_shader_parameter("u_clown_face_c", Vector2(0.5, 0.44))
	mat.set_shader_parameter("u_clown_tint", Vector3(0.5, 0.2, -0.45))
	mat.set_shader_parameter("u_time", 3.7)
	mat.set_shader_parameter("u_aspect", 1.7778)
	var rect := ColorRect.new()
	rect.material = mat
	rect.size = Vector2(320, 180)
	root.add_child(rect)

	# The field simulation itself, under realistic uniforms.
	var sim: Shader = load("res://shaders/umbra_field.gdshader")
	if sim == null:
		push_error("could not load umbra_field.gdshader")
		quit(1)
		return
	var smat := ShaderMaterial.new()
	smat.shader = sim
	smat.set_shader_parameter("u_dt", 0.033)
	smat.set_shader_parameter("u_reset", 0)
	smat.set_shader_parameter("u_time", 2.4)
	smat.set_shader_parameter("u_aspect", 1.7778)
	smat.set_shader_parameter("u_dir", Vector2(0.99, -0.12))
	smat.set_shader_parameter("u_loom", 0.55)
	smat.set_shader_parameter("u_rise", 1.2)
	smat.set_shader_parameter("u_roil", 0.65)
	smat.set_shader_parameter("u_cling", 0.35)
	smat.set_shader_parameter("u_wisp", 0.5)
	smat.set_shader_parameter("u_scale", 1.2)
	var srect := ColorRect.new()
	srect.material = smat
	srect.size = Vector2(384, 216)
	root.add_child(srect)

	for i in 6:
		await process_frame
	print("UMBRA_SHADER_CHECK_DONE")
	quit()
