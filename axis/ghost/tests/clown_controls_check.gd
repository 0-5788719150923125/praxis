extends SceneTree

## Does EVERY control on the clown actually do something?
##   godot --path axis/ghost --position -5000,-5000 --script res://tests/clown_controls_check.gd
## Needs a real renderer (the paint field is a ping-pong SubViewport pair).
##
## Written because two controls were reported as inert - "what does Smear and
## Evidence do? Neither seems to have any effect... fix them or remove them" - and
## the honest answer needed a measurement rather than a reading of the code. It
## turned out one was weak and one was genuinely dead:
##
##   Smear     0.010 - present, but far below its neighbours, so it read as inert
##                     beside them. Its wobble range was widened.
##   Evidence  0.0004 - nothing at all. It was a FLOOR under the evidence terms,
##                     and the contour work had quietly made it redundant: a floor
##                     only matters where the evidence is WEAK, and once the window
##                     IS the feature the evidence is strong throughout it. Worse,
##                     the evidence terms are each a dilated MAX over nine taps,
##                     which saturates at 1.0 - so even reshaping them did nothing.
##                     It now blends toward the CENTRE tap, the one that still
##                     varies pixel to pixel.
##
## So this sweeps every control the clown exposes, renders the whole pipeline -
## stencil, paint sim, and the mask_split branch - at each control's extremes, and
## measures how much the picture actually changed. A control that moves nothing is
## a lie on the panel, whatever the code looks like.
##
## The face comes from a SYNTHETIC track written here, so this needs no mediapipe,
## no venv and no clip - same trick face_track_check uses.

const W := 288
const H := 512
const T := 1.0
const POINTS := 478

## A control is judged on how many pixels it MOVES, not on the average movement
## over the frame. Averaging punishes anything that acts on a thin region however
## hard it acts there - Edge feather touches a band around the jaw and scores
## 0.003 by mean while being plainly visible - and it flatters anything that
## tints broadly and weakly. The fraction of pixels that changed by a visible
## amount says the thing that was actually asked: does moving this slider change
## what I see, anywhere.
##
## ...and that fraction is OF WHAT THE EFFECT DREW, not of the frame. Every control
## here acts on the painted mask and nowhere else, so a fraction of the FRAME also
## measures how much of the shot the face happens to fill - a property of the clip,
## not of the code. It bit exactly once: the eye-drip was rebuilt as a tear of
## liner, a streak about a fiftieth of the mask wide, and all three of its controls
## scored 0.2% of the frame and read as dead while being, on the author's own
## footage, a black line down a white cheek. The denominator is therefore the
## pixels the clown CHANGED AT ALL relative to the untouched frame, which is
## bounded by construction and says the thing meant: of what this effect draws, how
## much did this slider move.
const VISIBLE := 0.05     # a per-pixel change this size is one you can see
const DEAD := 0.004       # ...on fewer than 0.4% of the drawn mask is a control doing nothing

var _ed: Node
var _frame: Texture2D
var _fails: PackedStringArray = []


func _initialize() -> void:
	_ed = load("res://scripts/mask_editor.gd").new()
	var s := MaskSession.new()
	s.video_path = "res://masks/_controls_check/video.ogv"
	var m := {}
	for k in MaskSession.VECTOR_FIELDS:
		m[k] = MaskSession.DEFAULTS.get(k, 0.0)
	m["effect_a"] = float(MaskSession.EFFECT_CLOWN)
	s.markers.append(m)
	_ed.session = s
	_ed._src_size = Vector2i(W, H)
	root.add_child(_ed)
	_ed._ft_path = _write_track()
	_ed._ft_load()
	if _ed._ft_state != "ready":
		print("clown_controls_check: could not load the synthetic track (%s)" % _ed._ft_state)
		quit(2)
		return
	_frame = _make_frame()

	# name, marker field, low, high
	var knobs := [
		["Scale", "fx_scale", 0.35, 2.4],
		["Wear", "fx_density", 0.0, 1.0],
		["Smear", "fx_contrast", 0.0, 1.0],
		["Morph", "fx_tint", 0.0, 0.9],
		["Bleed", "fx_smooth", 0.0, 1.0],
		["Settle", "fx_lag", 0.0, 1.0],
		["Hollow", "fx_stick", 0.0, 1.0],
		["Eye size", "threshold", 0.0, 1.0],
		["Drip", "sat_floor", 0.0, 1.0],
		["Smudge", "swap", 0.0, 1.0],
		["Drip width", "fx_y", 0.0, 1.0],
		["Drip curve", "intensity_b", 0.0, 1.0],
		["Smile width", "feather", 0.0, 0.5],
		["Smile curve", "fx_speed", 0.0, 2.0],
		["Evidence", "resonance", 0.0, 1.0],
		["Edge feather", "fx_x", 0.0, 1.0],
	]
	print("%-14s %-12s %s" % ["control", "field", "mean pixel change between its extremes"])
	for k in knobs:
		var a := await _render(String(k[1]), float(k[2]))
		var b := await _render(String(k[1]), float(k[3]))
		var d := 0.0
		var moved := 0
		var drawn := 0
		var n := 0
		var raw := _frame.get_image()
		for y in range(0, H, 2):
			for x in range(0, W, 2):
				var pa := a.get_pixel(x, y)
				var pb := b.get_pixel(x, y)
				var va := Vector3(pa.r, pa.g, pa.b)
				var vb := Vector3(pb.r, pb.g, pb.b)
				var pr := raw.get_pixel(x, y)
				var vr := Vector3(pr.r, pr.g, pr.b)
				var delta := (va - vb).length()
				d += delta
				if delta > VISIBLE:
					moved += 1
				if (va - vr).length() > VISIBLE or (vb - vr).length() > VISIBLE:
					drawn += 1
				n += 1
		var mean := d / maxf(float(n), 1.0)
		var frac := float(moved) / maxf(float(drawn), 1.0)
		print("%-14s %-12s mean %.5f   visibly changed %6.2f%% of the drawn mask%s"
			% [k[0], k[1], mean, frac * 100.0, "   <-- DEAD" if frac < DEAD else ""])
		_expect(frac >= DEAD,
			"\"%s\" (%s) visibly changes %.2f%% of what the clown draws, between its "
			% [k[0], k[1], frac * 100.0] + "extremes - it does nothing. Fix it or take "
			+ "it off the panel")
	_ed.free()

	print("")
	if _fails.is_empty():
		print("clown_controls_check: PASS - every control on the panel moves the picture.")
		quit(0)
	else:
		for f in _fails:
			print("clown_controls_check: FAIL - ", f)
		quit(1)


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


## One render of the whole pipeline with `field` set to `v`, resolved through the
## editor's own mappings so the panel-to-shader path is what is under test rather
## than the shader alone.
func _render(field: String, v: float) -> Image:
	var l := {}
	for k in MaskSession.VECTOR_FIELDS:
		l[k] = MaskSession.DEFAULTS.get(k, 0.0)
	l[field] = v
	# Drip width and Drip curve shape a run that only exists once Drip is up, so
	# with sat_floor at its stored default they would both measure zero and read as
	# dead controls. They are swept ON a run, which is the only state in which the
	# panel offers them anything to do.
	if field == "fx_y" or field == "intensity_b":
		l["sat_floor"] = 1.0
	_ed._clown_fs = clampf(float(l["fx_scale"]), 0.3, 2.5)
	_ed._clown_bleed = clampf(float(l["fx_smooth"]), 0.0, 1.0)
	_ed._clown_settle = clampf(float(l["fx_lag"]), 0.0, 1.0)
	_ed._clown_hollow = clampf(float(l["fx_stick"]), 0.0, 1.0)
	_ed._clown_evidence = clampf(float(l["resonance"]), 0.0, 1.0)
	_ed._clown_eye_size = 1.2 + clampf(float(l["threshold"]), 0.0, 1.0) * 4.0
	_ed._clown_drip = clampf(float(l["sat_floor"]), 0.0, 1.0)
	_ed._clown_smudge = 0.45 + clampf(float(l["swap"]), 0.0, 1.0) * 0.55
	_ed._clown_drip_w = 0.35 + clampf(float(l["fx_y"]), 0.0, 1.0) * 0.80
	_ed._clown_drip_curve = 0.45 + clampf(float(l["intensity_b"]), 0.0, 1.0) * 1.55
	_ed._clown_smile_w = 1.0 + (clampf(float(l["feather"]), 0.0, 0.5) - 0.12) * 8.0
	_ed._clown_smile_curve = (clampf(float(l["fx_speed"]), 0.0, 2.0) - 1.0) * 0.12
	_ed._clown_feather = 0.012 + clampf(float(l["fx_x"]), 0.0, 1.0) * 0.055
	_ed._ft_apply_model(T)
	_ed._update_stencil(T)
	for i in 3:
		await process_frame
	return await _composite(await _sim(), l)


func _sim() -> Texture2D:
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
		mm.set_shader_parameter("u_reset", 1 if step == 0 else 0)
		mm.set_shader_parameter("u_time", float(step) * 0.033)
		mm.set_shader_parameter("u_aspect", float(W) / float(H))
		mm.set_shader_parameter("u_face_lum", 0.55)
		mm.set_shader_parameter("u_face_red", 0.10)
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


func _composite(paint: Texture2D, l: Dictionary) -> Image:
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/mask_split.gdshader")
	mat.set_shader_parameter("u_threshold", 0.24)
	mat.set_shader_parameter("u_feather", 0.12)
	mat.set_shader_parameter("u_sat_floor", 0.18)
	mat.set_shader_parameter("u_fade", 1.0)
	mat.set_shader_parameter("u_time", 2.0)
	mat.set_shader_parameter("u_aspect", float(W) / float(H))
	mat.set_shader_parameter("u_texel", Vector2(1.0 / float(W), 1.0 / float(H)))
	mat.set_shader_parameter("u_l_count", 1)
	mat.set_shader_parameter("u_l_effect", PackedInt32Array([16, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_hue", PackedFloat32Array([0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_w", PackedFloat32Array([1.0, 0, 0, 0, 0, 0]))
	mat.set_shader_parameter("u_l_off", PackedVector2Array([
		Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO, Vector2.ZERO]))
	mat.set_shader_parameter("u_l_scale", _f(float(l["fx_scale"])))
	mat.set_shader_parameter("u_l_speed", _f(1.0))
	mat.set_shader_parameter("u_l_glow", _f(1.0))
	mat.set_shader_parameter("u_l_dens", _f(float(l["fx_density"])))
	mat.set_shader_parameter("u_l_con", _f(float(l["fx_contrast"])))
	mat.set_shader_parameter("u_l_smooth", _f(float(l["fx_smooth"])))
	mat.set_shader_parameter("u_l_lagf", _f(float(l["fx_lag"])))
	mat.set_shader_parameter("u_l_stick", _f(float(l["fx_stick"])))
	mat.set_shader_parameter("u_l_tint", _f(float(l["fx_tint"])))
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
	mat.set_shader_parameter("u_clown_lum", 0.55)
	var ev: Vector2 = (_ed._face_eye_r_ema - _ed._face_eye_l_ema) \
		* Vector2(float(W) / float(H), 1.0)
	var el: float = maxf(ev.length(), 1e-4)
	mat.set_shader_parameter("u_clown_frame", ev / (el * el))
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var tr := TextureRect.new()
	tr.texture = _frame
	tr.material = mat
	tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	tr.size = Vector2(W, H)
	vp.add_child(tr)
	root.add_child(vp)
	for i in 4:
		await process_frame
	var img: Image = vp.get_texture().get_image()
	vp.queue_free()
	return img


## A face with real STRUCTURE in it - dark sockets, a red mouth, a lit nose ridge
## and shading across the cheeks. The evidence terms read the picture, so a flat
## fill would leave half these controls with nothing to act on.
func _make_frame() -> Texture2D:
	var img := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var u := (float(x) + 0.5) / float(W)
			var v := (float(y) + 0.5) / float(H)
			var c := Color(0.10, 0.10, 0.12)
			if Vector2((u - 0.5) / 0.30, (v - 0.42) / 0.30).length() < 1.0:
				var shade: float = 0.10 * sin(u * 34.0) + 0.08 * cos(v * 26.0)
				c = Color(0.74 + shade, 0.57 + shade, 0.48 + shade)
				for ex in [0.38, 0.62]:
					if Vector2((u - ex) / 0.075, (v - 0.34) / 0.035).length() < 1.0:
						c = Color(0.16, 0.14, 0.14)
				if Vector2((u - 0.5) / 0.12, (v - 0.56) / 0.045).length() < 1.0:
					c = Color(0.62, 0.22, 0.24)
				if Vector2((u - 0.5) / 0.05, (v - 0.45) / 0.07).length() < 1.0:
					c = Color(0.88, 0.72, 0.62)
			img.set_pixel(x, y, c)
	return ImageTexture.create_from_image(img)


## A one-sample track holding a plausible face, written in the same format
## face_track.py produces (see face_track_check for the layout).
func _write_track() -> String:
	var dir := ProjectSettings.globalize_path("user://face_tracks")
	DirAccess.make_dir_recursive_absolute(dir)
	var path := dir.path_join("_controls_check.bin")
	var pts := PackedVector2Array()
	pts.resize(POINTS)
	for i in POINTS:
		pts[i] = Vector2(0.5, 0.42)
	_ellipse(pts, _ed.FT_OVAL, Vector2(0.5, 0.42), Vector2(0.30, 0.30))
	_ellipse(pts, _ed.FT_EYE_L, Vector2(0.38, 0.34), Vector2(0.075, 0.035))
	_ellipse(pts, _ed.FT_EYE_R, Vector2(0.62, 0.34), Vector2(0.075, 0.035))
	_ellipse(pts, _ed.FT_LIPS, Vector2(0.50, 0.56), Vector2(0.12, 0.045))
	_ellipse(pts, _ed.FT_NOSE, Vector2(0.50, 0.45), Vector2(0.05, 0.07))
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_buffer("GFT1".to_ascii_buffer())
	f.store_32(1)
	f.store_float(15.0)
	f.store_32(40)
	f.store_32(POINTS)
	for s in 40:
		f.store_8(1)
		for i in POINTS:
			f.store_float(pts[i].x)
			f.store_float(pts[i].y)
	f.close()
	return path


static func _ellipse(pts: PackedVector2Array, idx: Array, c: Vector2, r: Vector2) -> void:
	for k in idx.size():
		var a := TAU * float(k) / float(idx.size())
		pts[int(idx[k])] = c + Vector2(cos(a) * r.x, sin(a) * r.y)


static func _f(v: float) -> PackedFloat32Array:
	var a := PackedFloat32Array()
	for i in 6:
		a.append(v)
	return a
