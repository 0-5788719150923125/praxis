extends SceneTree

## NOT a gate. Builds every thing in a table description (see [Props]) side by side on a dark
## cloth under candlelight and writes a PNG to look at - for judging shapes and materials without
## the tarot table around them.
##
##   tests/run_quiet.sh -- res://tests/props_look_probe.gd --spec <table.json> --out /tmp/props.png
##
## `--turn D` turns every thing D degrees; `--height H` raises the camera (meters). `--lamp 1` adds
## a lamp as the tarot table hangs one - a spot high to the left - and `--key 0` puts out the
## candle-like key, to see one light's shadows alone; `--lamp-shadow B,N,S` sets the lamp's shadow
## bias, normal bias and size (Godot's own defaults when not given), to try a setting before the
## table takes it. `--key-at X,Y,Z` moves the key; `--key-down 1` turns its shadow's paraboloids to
## face up and down.

const W := 1600
const H := 900


func _init() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	var spec_path := _arg(args, "--spec", "")
	var out := _arg(args, "--out", "/tmp/props.png")
	var turn := float(_arg(args, "--turn", "0"))
	var cam_h := float(_arg(args, "--height", "0.32"))
	var lamp_on := _arg(args, "--lamp", "0") == "1"
	var key_on := _arg(args, "--key", "1") == "1"
	var lamp_shadow := _arg(args, "--lamp-shadow", "")
	var key_at := _arg(args, "--key-at", "")
	var key_down := _arg(args, "--key-down", "0") == "1"
	var j := JSON.new()
	if j.parse(FileAccess.get_file_as_string(spec_path)) != OK or not (j.data is Dictionary):
		print("props_look_probe: %s is not a table description" % spec_path)
		quit(2)
		return
	var spec := Props.sanitize(j.data as Dictionary, ["#b48a43", "#3f6b4a", "#d9d6c7"])
	print("props_look_probe: %d things, %d materials" % [(spec["things"] as Array).size(), (spec["materials"] as Dictionary).size()])
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.own_world_3d = true
	vp.msaa_3d = Viewport.MSAA_4X
	vp.positional_shadow_atlas_size = 4096
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(vp)
	var world := Node3D.new()
	vp.add_child(world)
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.02, 0.02, 0.025)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.5, 0.47, 0.45)
	env.ambient_light_energy = 0.2
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.tonemap_exposure = 0.85
	env.glow_enabled = true
	env.glow_hdr_threshold = 1.25
	var cloth := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(2.0, 1.2)
	cloth.mesh = pm
	var cm := StandardMaterial3D.new()
	cm.albedo_color = Color(0.13, 0.15, 0.2)
	cm.roughness = 0.9
	cloth.material_override = cm
	world.add_child(cloth)
	# the things in a row, widest apart as they need
	var x := 0.0
	var built: Array = []
	var i := 0
	for t in spec["things"]:
		var b := Props.build(t as Dictionary, spec["materials"], 1000 + i)
		i += 1
		var size: AABB = b["size"]
		var w := maxf(size.size.x, size.size.z) + 0.04
		var node: Node3D = b["node"]
		node.rotation_degrees.y = turn + float((t as Dictionary).get("turn", 0.0))
		world.add_child(node)
		built.append([node, w, b])
		x += w
		print("  %-46s %5.1f x %5.1f x %5.1f cm, %d meshes, foot %d pts, %d wicks" % [String((t as Dictionary)["name"]).substr(0, 46),
			size.size.x * 100.0, size.size.y * 100.0, size.size.z * 100.0, (b["meshes"] as Array).size(),
			(b["foot"] as PackedVector2Array).size(), (b["wicks"] as Array).size()])
	var at := -x * 0.5
	var lights := 0
	for e in built:
		var node: Node3D = e[0]
		var w := float(e[1])
		node.position = Vector3(at + w * 0.5, 0.0, 0.0)
		for wk in ((e[2] as Dictionary)["wicks"] as Array):
			var p: Vector3 = node.transform * (wk as Vector3)
			var flame := MeshInstance3D.new()
			var q := QuadMesh.new()
			q.size = Vector2(0.012, 0.03)
			flame.mesh = q
			var fm := StandardMaterial3D.new()
			fm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			fm.albedo_color = Color(1.0, 0.75, 0.4)
			fm.emission_enabled = true
			fm.emission = Color(1.0, 0.7, 0.35)
			fm.emission_energy_multiplier = 4.0
			flame.material_override = fm
			flame.position = p + Vector3(0, 0.014, 0)
			world.add_child(flame)
			var l := OmniLight3D.new()
			l.light_color = Color(1.0, 0.7, 0.4)
			l.omni_range = 1.2
			l.light_energy = 0.5
			l.position = p + Vector3(0, 0.03, 0)
			world.add_child(l)
			lights += 1
		at += w
	# what the things reflect, caught once, as the tarot table's probe does: metal and the play of
	# light read as flat paint with nothing to reflect
	var probe := ReflectionProbe.new()
	probe.update_mode = ReflectionProbe.UPDATE_ONCE
	probe.size = Vector3(maxf(x * 1.3, 1.9), 1.0, 1.4)
	probe.position = Vector3(0.0, 0.2, -0.05)
	probe.box_projection = true
	probe.max_distance = 5.0
	world.add_child(probe)
	if lamp_on:
		var lamp := SpotLight3D.new()
		lamp.light_color = Color(1.0, 0.86, 0.72)
		lamp.light_energy = 1.6
		lamp.spot_range = 4.0
		lamp.spot_angle = 38.0
		lamp.spot_angle_attenuation = 2.2
		lamp.shadow_enabled = true
		lamp.shadow_blur = 1.6
		if not lamp_shadow.is_empty():
			var v := lamp_shadow.split(",")
			lamp.shadow_bias = float(v[0])
			lamp.shadow_normal_bias = float(v[1])
			if v.size() > 2:
				lamp.light_size = float(v[2])
		world.add_child(lamp)
		lamp.position = Vector3(-maxf(x * 0.5, 0.5), 0.85, -0.2)
		lamp.look_at(Vector3(0.0, 0.0, 0.0), Vector3.UP)
		print("props_look_probe: lamp bias %.4f normal %.3f size %.3f" % [lamp.shadow_bias, lamp.shadow_normal_bias, lamp.light_size])
	var key := OmniLight3D.new()
	key.visible = key_on
	key.light_color = Color(1.0, 0.72, 0.45)
	key.light_energy = 1.6
	key.omni_range = 3.0
	key.shadow_enabled = true
	key.light_size = 0.02
	key.shadow_bias = 0.02
	key.shadow_normal_bias = 0.4
	key.position = Vector3(-x * 0.2, 0.22, -0.18)
	if not key_at.is_empty():
		var v := key_at.split(",")
		key.position = Vector3(float(v[0]), float(v[1]), float(v[2]))
	if key_down:
		key.rotation_degrees = Vector3(-90.0, 0.0, 0.0)
	world.add_child(key)
	var fill := DirectionalLight3D.new()
	fill.light_energy = 0.25
	fill.light_color = Color(0.75, 0.8, 1.0)
	fill.rotation_degrees = Vector3(-50, 30, 0)
	world.add_child(fill)
	var cam := Camera3D.new()
	cam.environment = env
	cam.fov = 30.0
	var dist := maxf(x * 0.95, 0.4)
	cam.position = Vector3(0.0, cam_h * dist / 0.5, dist)
	world.add_child(cam)
	cam.look_at(Vector3(0, 0.04, 0), Vector3.UP)
	for _f in 12:
		await process_frame
	var img := vp.get_texture().get_image()
	img.save_png(out)
	print("props_look_probe: %d flames -> %s" % [lights, out])
	quit(0)


func _arg(args: PackedStringArray, key: String, fallback: String) -> String:
	var k := args.find(key)
	return args[k + 1] if k >= 0 and k + 1 < args.size() else fallback
