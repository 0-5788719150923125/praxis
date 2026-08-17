extends SceneTree

## LOOK at the clown on a real face, on a real frame, through the real track.
##   godot --path axis/ghost --position -5000,-5000 --script res://tests/clown_look_probe.gd -- \
##       --frame /abs/frame.png --track /abs/track.bin --time 10.0 --out /abs/out.png
## Needs a real renderer (the paint field is a ping-pong SubViewport pair).
##
## Not a gate - it asserts nothing and always exits 0. The gates measure claims;
## this exists because several rounds of "it looks terrible" were answered by
## reasoning about the shader instead of rendering the author's own clip and
## looking at it, and each time the reasoning was wrong about something the frame
## would have shown immediately. Run it, open the PNG.
##
## It also PRINTS the model it resolved - eye centres, measured radii, the derived
## drip source and width - because those numbers are what the geometry is actually
## built from, and a shape that is wrong on a real face is usually one of them
## being much larger or smaller than the guess in the code assumed.
##
## Optional `--knob field=value` (repeatable) sets any MaskSession field, so the
## same probe renders a sweep of one control.

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
	var track_path := String(_args.get("track", ""))
	var out_path := String(_args.get("out", "/tmp/clown_look.png"))
	var t := float(_args.get("time", "10.0"))
	var img := Image.new()
	if img.load(frame_path) != OK:
		print("clown_look_probe: cannot read frame ", frame_path)
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
	m["effect_a"] = float(MaskSession.EFFECT_CLOWN)
	s.markers.append(m)
	_ed.session = s
	_ed._src_size = Vector2i(_w, _h)
	root.add_child(_ed)
	_ed._ft_path = track_path
	_ed._ft_load()
	if _ed._ft_state != "ready":
		print("clown_look_probe: track did not load (%s)" % _ed._ft_state)
		quit(2)
		return

	# The blob fitter runs once, only for its COLOUR statistics (mean tint and
	# luminance) - landmarks cannot give those and the coat's match test needs
	# them. Every geometric term it sets is overwritten by _ft_apply_model.
	_ed._update_face_model(img)
	# A SWEEP RENDERS IN ONE PROCESS. Godot has no offscreen GPU path on this
	# machine (--headless forces the dummy renderer, which returns no image on
	# readback), so every render is a window that steals focus - nine of them for a
	# nine-cell contact sheet. One process, nine PNGs.
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


func _one(knobs: Dictionary, t: float, out_path: String) -> void:
	var l := {}
	for k in MaskSession.VECTOR_FIELDS:
		l[k] = MaskSession.DEFAULTS.get(k, 0.0)
	# `--session <session.json>` starts from the author's OWN clown marker instead
	# of the stored defaults. It matters more than it sounds: their Eye size is 3.6
	# against a default of 2.2 and their Hollow is wound to the top, so a fault that
	# is barely visible at the defaults is glaring in their editor - which is most
	# of why several rounds of "can you not see what I'm seeing" were fair.
	if _args.has("session"):
		var f := FileAccess.open(String(_args["session"]), FileAccess.READ)
		if f != null:
			var doc = JSON.parse_string(f.get_as_text())
			f.close()
			if doc is Dictionary:
				for m2 in doc.get("markers", []):
					if int(float(m2.get("effect_a", 0))) == MaskSession.EFFECT_CLOWN:
						for k in m2:
							if l.has(k):
								l[k] = float(m2[k])
	for k in knobs:
		l[k] = knobs[k]
	_ed._clown_fs = clampf(float(l["fx_scale"]), 0.3, 2.5)
	_ed._clown_bleed = clampf(float(l["fx_smooth"]), 0.0, 1.0)
	_ed._clown_settle = clampf(float(l["fx_lag"]), 0.0, 1.0)
	_ed._clown_hollow = clampf(float(l["fx_stick"]), 0.0, 1.0)
	_ed._clown_evidence = clampf(float(l["resonance"]), 0.0, 1.0)
	_ed._clown_eye_size = 1.2 + clampf(float(l["threshold"]), 0.0, 1.0) * 4.0
	_ed._clown_drip = clampf(float(l["sat_floor"]), 0.0, 1.0) - 0.18
	_ed._clown_smudge = 0.45 + clampf(float(l["swap"]), 0.0, 1.0) * 0.55
	_ed._clown_drip_w = 0.35 + clampf(float(l["fx_y"]), 0.0, 1.0) * 0.80
	_ed._clown_drip_curve = 0.45 + clampf(float(l["intensity_b"]), 0.0, 1.0) * 1.55
	_ed._clown_smile_w = 1.0 + (clampf(float(l["feather"]), 0.0, 0.5) - 0.12) * 8.0
	_ed._clown_smile_curve = (clampf(float(l["fx_speed"]), 0.0, 2.0) - 1.0) * 0.12
	_ed._clown_feather = 0.012 + clampf(float(l["fx_x"]), 0.0, 1.0) * 0.055
	# The blob fitter is run only for its COLOUR statistics (mean tint/luminance),
	# which landmarks cannot give and the coat's match test needs; the landmark
	# model is applied after it and overwrites every geometric term.
	_ed._ft_apply_model(t)
	_ed._update_stencil(t)
	for j in 3:
		await process_frame

	var asp := Vector2(float(_w) / float(_h), 1.0)
	var el: Vector2 = _ed._face_eye_l_ema * asp
	var er: Vector2 = _ed._face_eye_r_ema * asp
	print("frame %dx%d  aspect %.3f   t=%.2f" % [_w, _h, asp.x, t])
	print("  eye L uv %s  R uv %s   separation (height units) %.4f"
		% [_ed._face_eye_l_ema, _ed._face_eye_r_ema, el.distance_to(er)])
	print("  measured eye radii  L %.4f  R %.4f   (uv height units)"
		% [_ed._face_eye_lr_ema, _ed._face_eye_rr_ema])
	print("  face centre %s  semi-axes %s" % [_ed._face_c_ema, _ed._face_r_ema])
	print("  nose %s   mouth %s" % [_ed._face_nose_ema, _ed._face_mouth_ema])
	print("  chin (face_c.y + face_r.y) %.4f -> eye-to-chin %.4f"
		% [_ed._face_c_ema.y + _ed._face_r_ema.y,
			(_ed._face_c_ema.y + _ed._face_r_ema.y) - _ed._face_eye_l_ema.y])
	print("  Drip %.3f  width %.3f  curve %.3f  eye size %.2fx"
		% [_ed._clown_drip, _ed._clown_drip_w, _ed._clown_drip_curve, _ed._clown_eye_size])

	var paint := await _sim()
	# The eye channel on its own, as grey. mask_split's own gating, the coat and
	# the craquelure all sit on top of this in the composite, so a run that looks
	# wrong there might be wrong here or might be wrong there.
	var pi: Image = paint.get_image()
	var grey := Image.create_empty(pi.get_width(), pi.get_height(), false, Image.FORMAT_RGB8)
	for gy in pi.get_height():
		for gx in pi.get_width():
			var v := clampf(pi.get_pixel(gx, gy).r, 0.0, 1.0)
			grey.set_pixel(gx, gy, Color(v, v, v))
	grey.save_png(out_path.get_basename() + "_paint.png")
	var out := await _composite(paint)
	out.save_png(out_path)
	print("clown_look_probe: wrote ", out_path)


func _sim() -> Texture2D:
	var vps: Array = []
	var rects: Array = []
	for i in 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(_w, _h)
		vp.disable_3d = true
		vp.use_hdr_2d = true
		vp.transparent_bg = true
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
		var r := ColorRect.new()
		r.size = Vector2(_w, _h)
		var mm := ShaderMaterial.new()
		mm.shader = load("res://shaders/clown_paint.gdshader")
		r.material = mm
		vp.add_child(r)
		root.add_child(vp)
		vps.append(vp)
		rects.append(r)
	var ping := 0
	for step in 30:
		var mm: ShaderMaterial = rects[ping].material
		mm.set_shader_parameter("u_prev", vps[1 - ping].get_texture())
		mm.set_shader_parameter("u_frame", _frame)
		mm.set_shader_parameter("u_stencil", _ed._stencil_vp.get_texture())
		mm.set_shader_parameter("u_has_stencil", 1.0)
		mm.set_shader_parameter("u_coat_feather", _ed._clown_feather)
		mm.set_shader_parameter("u_drip", _ed._clown_drip)
		mm.set_shader_parameter("u_drip_w", _ed._clown_drip_w)
		mm.set_shader_parameter("u_eye_smudge", _ed._clown_smudge)
		mm.set_shader_parameter("u_drip_curve", _ed._clown_drip_curve)
		mm.set_shader_parameter("u_dt", 0.033)
		# `--reset 1` deposits every step with no advection or decay, which is how
		# you look at the DEPOSIT's own geometry rather than at what the liquid did
		# to it afterwards. Several rounds of guessing why a run "died early" were
		# actually the sim thinning a tail the deposit had drawn correctly.
		mm.set_shader_parameter("u_reset",
			1 if (step == 0 or _args.has("reset")) else 0)
		mm.set_shader_parameter("u_time", float(step) * 0.033)
		mm.set_shader_parameter("u_aspect", float(_w) / float(_h))
		mm.set_shader_parameter("u_face_lum", _ed._face_lum_ema)
		mm.set_shader_parameter("u_face_red", _ed._face_red_ema)
		mm.set_shader_parameter("u_eye_l", _ed._face_eye_l_ema)
		mm.set_shader_parameter("u_eye_r", _ed._face_eye_r_ema)
		mm.set_shader_parameter("u_mouth", _ed._face_mouth_ema)
		mm.set_shader_parameter("u_nose", _ed._face_nose_ema)
		mm.set_shader_parameter("u_face_c", _ed._face_c_ema)
		mm.set_shader_parameter("u_face_r", _ed._face_r_ema)
		mm.set_shader_parameter("u_eye_lr", _ed._face_eye_lr_ema)
		mm.set_shader_parameter("u_eye_rr", _ed._face_eye_rr_ema)
		mm.set_shader_parameter("u_mouth_r", _ed._face_mouth_r_ema)
		mm.set_shader_parameter("u_scale", _ed._clown_fs)
		mm.set_shader_parameter("u_evidence", _ed._clown_evidence)
		mm.set_shader_parameter("u_settle", _ed._clown_settle)
		mm.set_shader_parameter("u_bleed", _ed._clown_bleed)
		mm.set_shader_parameter("u_hollow", _ed._clown_hollow)
		vps[ping].render_target_update_mode = SubViewport.UPDATE_ONCE
		await process_frame
		ping = 1 - ping
	return vps[1 - ping].get_texture()


func _f(v: float) -> PackedFloat32Array:
	return PackedFloat32Array([v, v, v, v, v, v])


func _composite(paint: Texture2D) -> Image:
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
	mat.set_shader_parameter("u_l_effect", PackedInt32Array([16, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_hue", PackedFloat32Array([0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_w", PackedFloat32Array([1.0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_off", PackedVector2Array([
		Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO]))
	mat.set_shader_parameter("u_l_scale", _f(1.0))
	mat.set_shader_parameter("u_l_speed", _f(1.0))
	mat.set_shader_parameter("u_l_glow", _f(1.0))
	mat.set_shader_parameter("u_l_dens", _f(0.0))
	mat.set_shader_parameter("u_l_con", _f(0.0))
	mat.set_shader_parameter("u_l_smooth", _f(0.0))
	mat.set_shader_parameter("u_l_lagf", _f(0.0))
	mat.set_shader_parameter("u_l_stick", _f(0.0))
	mat.set_shader_parameter("u_l_tint", _f(0.0))
	mat.set_shader_parameter("u_l_accent", _f(0.0))
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
	mat.set_shader_parameter("u_clown_paint", paint)
	mat.set_shader_parameter("u_clown_eye_l", _ed._face_eye_l_ema)
	mat.set_shader_parameter("u_clown_eye_r", _ed._face_eye_r_ema)
	mat.set_shader_parameter("u_clown_mouth", _ed._face_mouth_ema)
	mat.set_shader_parameter("u_clown_face_r", _ed._face_r_ema)
	mat.set_shader_parameter("u_clown_face_c", _ed._face_c_ema)
	mat.set_shader_parameter("u_clown_eye_lr", _ed._face_eye_lr_ema)
	mat.set_shader_parameter("u_clown_eye_rr", _ed._face_eye_rr_ema)
	mat.set_shader_parameter("u_clown_mouth_r", _ed._face_mouth_r_ema)
	mat.set_shader_parameter("u_clown_tint", Vector3(0.16, -0.02, -0.14))
	mat.set_shader_parameter("u_clown_lum", _ed._face_lum_ema)
	var ev: Vector2 = (_ed._face_eye_r_ema - _ed._face_eye_l_ema) \
		* Vector2(float(_w) / float(_h), 1.0)
	var elen: float = maxf(ev.length(), 1e-4)
	mat.set_shader_parameter("u_clown_frame", ev / (elen * elen))
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
