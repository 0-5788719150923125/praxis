extends SceneTree

## LOOK at the umbra on a real frame, through the real pose track.
##   tests/run_quiet.sh -- res://tests/umbra_look_probe.gd \
##       --frame /abs/frame.png --track /abs/track.bin --time 10.0 --out /abs/out.png
## Needs a real renderer (the field is a ping-pong SubViewport pair).
##
## Not a gate - it asserts nothing and always exits 0. The gates measure claims;
## this exists for the same reason the clown's does: several rounds of "it looks
## terrible" were answered by reasoning about the shader instead of rendering the
## author's own clip and looking at it, and every time the reasoning was wrong
## about something the frame would have shown in a second.
##
## It also PRINTS the throw it solved - her eye line, the ghost's head, the
## anchor, the scale, both eyes - because a ghost that reads wrong is nearly
## always one of those being somewhere other than the code assumed.
##
## `--knob field=value` (repeatable) sets any MaskSession field; `--sweep
## field=a,b,c` renders a contact sheet IN ONE PROCESS. Every run also writes
## `<out>_field.png`, the field's own channels as a picture (R = mass,
## G = essence, B = eyes), which separates "the throw is wrong" from "the render
## is wrong".

const SIM_W := 384
const SIM_H := 216

var _ed: Node
var _frame: Texture2D
var _w := 640
var _h := 360
var _args := {}
var _knobs := {}


func _initialize() -> void:
	var argv := OS.get_cmdline_user_args()
	var i := 0
	while i < argv.size():
		var a := String(argv[i])
		if a == "--knob" and i + 1 < argv.size():
			var kv := String(argv[i + 1]).split("=")
			if kv.size() == 2:
				_knobs[kv[0]] = float(kv[1])
			i += 2
		elif a.begins_with("--") and i + 1 < argv.size():
			_args[a.substr(2)] = String(argv[i + 1])
			i += 2
		else:
			i += 1
	var frame_path := String(_args.get("frame", ""))
	var out_path := String(_args.get("out", "/tmp/ghost_scratch/umbra_look.png"))
	var t := float(_args.get("time", "10.0"))
	var img := Image.new()
	if img.load(frame_path) != OK:
		print("umbra_look_probe: cannot read frame ", frame_path)
		quit(2)
		return
	_w = img.get_width()
	_h = img.get_height()
	_frame = ImageTexture.create_from_image(img)

	_ed = load("res://scripts/mask_editor.gd").new()
	var s := MaskSession.new()
	s.video_path = "res://masks/_look_probe/video.ogv"
	var m := {}
	for k in MaskSession.VECTOR_FIELDS:
		m[k] = MaskSession.DEFAULTS.get(k, 0.0)
	m["effect_a"] = float(MaskSession.EFFECT_UMBRA)
	s.markers.append(m)
	_ed.session = s
	_ed._src_size = Vector2i(_w, _h)
	root.add_child(_ed)
	# `--track` names ONE WINDOW file, and `--chunk` says which window of the clip
	# it is (default 0, which is what `--start 0` produces). The editor reads the
	# clip a window at a time now, so a probe that loaded a whole-clip track would
	# be exercising a path that no longer exists.
	_ed._pt_state = "ready"
	_ed._pt_load_chunk(int(_args.get("chunk", "0")), String(_args.get("track", "")))
	if _ed._pt_chunks.is_empty():
		print("umbra_look_probe: the pose window did not load")
		quit(2)
		return

	var cases: Array = []
	var sweep := String(_args.get("sweep", ""))
	if sweep.contains("="):
		var sf := sweep.split("=")[0]
		for sv in sweep.split("=")[1].split(","):
			var c := _knobs.duplicate()
			c[sf] = float(sv)
			cases.append(c)
	else:
		cases.append(_knobs)
	for ci in cases.size():
		await _one(cases[ci], t,
			out_path if cases.size() == 1
			else out_path.get_basename() + "_%d.png" % ci)
	_ed.free()
	quit(0)


## Resolve one knob set into the editor's live umbra state, exactly as
## _apply_frame_state does from a layer - kept in step with it by hand, which is
## the probe's one maintenance cost and the price of not booting a whole session.
func _one(knobs: Dictionary, t: float, out_path: String) -> void:
	var l := {}
	for k in MaskSession.VECTOR_FIELDS:
		l[k] = MaskSession.DEFAULTS.get(k, 0.0)
	# `--session <session.json>` starts from the author's OWN umbra marker rather
	# than the stored defaults: a fault that is barely visible at the defaults can
	# be glaring at the settings they actually work at.
	if _args.has("session"):
		var f := FileAccess.open(String(_args["session"]), FileAccess.READ)
		if f != null:
			var doc = JSON.parse_string(f.get_as_text())
			f.close()
			if doc is Dictionary:
				for m2 in doc.get("markers", []):
					if int(float(m2.get("effect_a", 0))) == MaskSession.EFFECT_UMBRA:
						for k in m2:
							if l.has(k):
								l[k] = float(m2[k])
	for k in knobs:
		l[k] = knobs[k]
	_ed._umbra_active = true
	# THE SHIPPED RESOLVE, not a copy of it. This probe used to restate every knob
	# mapping and drifted from the editor the first time one of them changed.
	_ed._umb_read_layer(l, 0.35)
	# _umb_solve_cast reads the playhead off the player, which the probe has not
	# got - so the sample is chosen here and _umb_cast_from (the same function,
	# minus the clock) does the rest. Everything past this point is shipped code.
	var sample: int = _ed._pt_slot_at(t + _ed._umb_lead)
	if sample < 0:
		print("umbra_look_probe: no pose at t=%.2f (+lead %.2f)" % [t, _ed._umb_lead])
		return
	_solve_at(sample)

	var asp: float = float(_w) / float(_h)
	print("frame %dx%d  aspect %.3f   t=%.2f  lead %.2f -> sample %d"
		% [_w, _h, asp, t, _ed._umb_lead, sample])
	print("  her eye line (%.3f, %.3f)   unit %.4f   scale %.2f  narrow %.2f  lean %.2f"
		% [_ed._umb_eye_src.x / asp, _ed._umb_eye_src.y, _ed._umb_unit,
			_ed._umb_scale, _ed._umb_narrow, _ed._umb_lean])
	print("  ghost anchor (%.3f, %.3f)   throw (%.2f, %.2f)   bust %.2f..%.2f"
		% [_ed._umb_anchor.x / asp, _ed._umb_anchor.y, _ed._umb_dir.x, _ed._umb_dir.y,
			_ed._umb_bust0, _ed._umb_bust1])
	print("  ghost eyes L (%.3f, %.3f)  R (%.3f, %.3f)  radius %.4f  ok %s"
		% [_ed._umb_eye_l.x, _ed._umb_eye_l.y, _ed._umb_eye_r.x, _ed._umb_eye_r.y,
			_ed._umb_eye_rad, str(_ed._umb_eyes_ok)])

	var field := await _sim()
	var fi: Image = field.get_image()
	fi.convert(Image.FORMAT_RGBA8)
	fi.save_png(out_path.get_basename() + "_field.png")
	var out := await _composite(field, l)
	out.save_png(out_path)
	print("umbra_look_probe: wrote ", out_path)


## The cast solve, with the sample index supplied rather than read from a
## playhead - MaskEditor._umb_cast_from is exactly that split, so the probe runs
## the SHIPPED geometry rather than a copy of it that can drift away from it.
func _solve_at(i: int) -> void:
	_ed._pt_slot = -1
	var ok: bool = _ed._umb_cast_from(i, float(_w) / float(_h))
	if not ok:
		print("umbra_look_probe: the cast solve refused sample %d" % i)
		return
	# Body from this sample, guard from the playhead's own - the probe has no
	# playhead, so `--guard <sample>` overrides it and it defaults to the same one.
	_ed._pt_upload_mask(i, int(_args.get("guard", str(i))))


func _sim() -> Texture2D:
	var vps: Array = []
	var rects: Array = []
	for i in 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(SIM_W, SIM_H)
		vp.disable_3d = true
		vp.use_hdr_2d = true
		vp.transparent_bg = true
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
		var r := ColorRect.new()
		r.size = Vector2(SIM_W, SIM_H)
		var mm := ShaderMaterial.new()
		mm.shader = load("res://shaders/umbra_field.gdshader")
		r.material = mm
		vp.add_child(r)
		root.add_child(vp)
		vps.append(vp)
		rects.append(r)
	var ping := 0
	for step in 40:
		var mm: ShaderMaterial = rects[ping].material
		mm.set_shader_parameter("u_prev", vps[1 - ping].get_texture())
		mm.set_shader_parameter("u_region", _ed._pt_tex)
		mm.set_shader_parameter("u_dt", 0.033)
		mm.set_shader_parameter("u_reset", 1 if step == 0 else 0)
		mm.set_shader_parameter("u_time", float(step) * 0.033)
		mm.set_shader_parameter("u_aspect", float(_w) / float(_h))
		mm.set_shader_parameter("u_dir", _ed._umb_dir)
		mm.set_shader_parameter("u_loom", _ed._umb_loom)
		mm.set_shader_parameter("u_rise", _ed._umb_rise)
		mm.set_shader_parameter("u_roil", _ed._umb_roil)
		mm.set_shader_parameter("u_cling", _ed._umb_cling)
		mm.set_shader_parameter("u_wisp", _ed._umb_wisp)
		mm.set_shader_parameter("u_inv0", _ed._umb_inv0)
		mm.set_shader_parameter("u_inv1", _ed._umb_inv1)
		mm.set_shader_parameter("u_anchor", _ed._umb_anchor)
		mm.set_shader_parameter("u_src", _ed._umb_src)
		mm.set_shader_parameter("u_eye_src", _ed._umb_eye_src)
		mm.set_shader_parameter("u_unit", _ed._umb_unit)
		mm.set_shader_parameter("u_bust0", _ed._umb_bust0)
		mm.set_shader_parameter("u_bust1", _ed._umb_bust1)
		mm.set_shader_parameter("u_eye_l", _ed._umb_eye_l)
		mm.set_shader_parameter("u_eye_r", _ed._umb_eye_r)
		mm.set_shader_parameter("u_eye_rad", _ed._umb_eye_rad)
		mm.set_shader_parameter("u_eye_amt", _ed._umb_eye_amt)
		vps[ping].render_target_update_mode = SubViewport.UPDATE_ONCE
		await process_frame
		ping = 1 - ping
	return vps[1 - ping].get_texture()


func _f(v: float) -> PackedFloat32Array:
	return PackedFloat32Array([v, v, v, v, v, v])


func _composite(field: Texture2D, l: Dictionary) -> Image:
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/mask_split.gdshader")
	mat.set_shader_parameter("u_threshold", 0.24)
	mat.set_shader_parameter("u_feather", 0.12)
	mat.set_shader_parameter("u_sat_floor", 0.18)
	mat.set_shader_parameter("u_fade", 1.0)
	mat.set_shader_parameter("u_time", 2.0)
	mat.set_shader_parameter("u_aspect", float(_w) / float(_h))
	mat.set_shader_parameter("u_texel", Vector2(1.0 / float(_w), 1.0 / float(_h)))
	mat.set_shader_parameter("u_l_count", 1)
	mat.set_shader_parameter("u_l_effect",
		PackedInt32Array([MaskSession.EFFECT_UMBRA, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_hue", PackedFloat32Array([0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_w", PackedFloat32Array([1.0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_off", PackedVector2Array([
		Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO]))
	mat.set_shader_parameter("u_l_scale", _f(float(l["fx_scale"])))
	mat.set_shader_parameter("u_l_speed", _f(float(l["fx_speed"])))
	mat.set_shader_parameter("u_l_glow", _f(1.0))
	mat.set_shader_parameter("u_l_dens", _f(float(l["fx_density"])))
	mat.set_shader_parameter("u_l_con", _f(float(l["fx_contrast"])))
	mat.set_shader_parameter("u_l_smooth", _f(float(l["fx_smooth"])))
	mat.set_shader_parameter("u_l_lagf", _f(float(l["fx_lag"])))
	mat.set_shader_parameter("u_l_stick", _f(float(l["fx_stick"])))
	mat.set_shader_parameter("u_l_tint", _f(float(l["fx_tint"])))
	mat.set_shader_parameter("u_l_accent", _f(float(l["hue_b"])))
	mat.set_shader_parameter("u_l_elag", PackedInt32Array([0, 0, 0, 0, 0, 0]))
	var regs := PackedVector4Array()
	for i in 6:
		regs.append(Vector4(0, 0, 1, 1))
	mat.set_shader_parameter("u_l_region", regs)
	mat.set_shader_parameter("u_l_regsoft", _f(0.0))
	var ew := PackedFloat32Array()
	ew.resize(48)
	for i in 6:
		ew[i * 8] = 1.0
	mat.set_shader_parameter("u_l_ew", ew)
	var tds := PackedVector3Array()
	for i in 6:
		tds.append(Vector3(0.86, -0.37, -0.37))
	mat.set_shader_parameter("u_l_tdir", tds)
	mat.set_shader_parameter("u_umbra_field", field)
	var vp := SubViewport.new()
	vp.size = Vector2i(_w, _h)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var tr := TextureRect.new()
	tr.texture = _frame
	tr.material = mat
	tr.size = Vector2(_w, _h)
	tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	vp.add_child(tr)
	root.add_child(vp)
	for i in 3:
		await process_frame
	var out: Image = vp.get_texture().get_image()
	vp.queue_free()
	await process_frame
	return out
