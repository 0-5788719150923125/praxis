extends Node

## Is it VAPOUR? The gate for `vapors` and shaders/vapor_field.gdshader.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/vapor_check.gd 480
##
## Needs BOTH a real boot (the scene reaches Spectrum and Director) and a real renderer (every
## claim below is measured off pixels), which is what GHOST_PROBE_GPU is for.
##
## This scene is a LOOK, and a look is exactly what gets shipped broken on a careful read: the
## first three cuts of the field all compiled, all ran, all reported plausible coverage, and all
## three drew engraved satin - long parallel contour lines over the whole frame - instead of
## vapour. Then the fourth moved beautifully and pulsed like a bellows. So each claim here is one
## of the properties that separates the look from the ways it has ACTUALLY failed, and each is
## measured against a control that breaks that property, because a threshold nothing can fail is
## not a gate.
##
##   PRESENCE   Masses AND darkness. A field that fills the frame is a colour wash and one that
##              fills none of it is a bug; the negative space is half of what makes masses read.
##   FRONT      A visible boundary. The SHARE of lit edges that jump more than STEEP in one
##              pixel, against the same field with `hard` at 0 - the soft-edged haze this look
##              must not be, and which every existing fog/cloud layer here already is.
##   FILAMENT   Elongated structure. Structure-tensor COHERENCE per block (how aligned the
##              gradients inside it are), against the same field sampled isotropically
##              (`stretch` 1). Cloud is incoherent; drawn-out vapour is not.
##   STEADINESS The masses do not INFLATE. Reported from the field and the worst thing this
##              scene did: "it expands and contracts rapidly with the harmonics... it should be
##              continuous movement forward, not an expand and contract". A plume's amplitude IS
##              the size of its mass, so anything fast reaching it pulses the whole frame.
##              Measured as the largest change in lit AREA over a quarter second across a
##              passage with beats, against a control that drives the amplitudes the way the
##              first version's beat kick did.
##   COLOUR     More than one hue in the frame, area-weighted, against a control with every
##              plume on the SAME colour. Two masses of different colours blending where they
##              meet is the reference image's whole subject.
##   MOTION     The field moves on its own, under the STATIC behavior (no camera drift at all),
##              against the same frames with the phase rolled back - so what is measured is the
##              vapour flowing and not the camera.
##
## FRONT and FILAMENT are averaged over SEVERAL MOMENTS, each against its control at the same
## instant. Single-moment readings of both swing by a third depending on what happens to be in
## frame (front measured x2.01, x1.67 and x1.51 on three states of the same seed), which is wide
## enough to put a fixed threshold on either side of the truth. The controls are paired inside
## the moment, so only the state varies between samples and averaging removes it.

const W := 640
const H := 360
const DT := 1.0 / 30.0
## An edge is STEEP if adjacent pixels differ by more than this in luminance. A front is a TAIL
## property of that distribution: the MEAN step was the first version of this measure and could
## not see a front at all (x1.20 between a cut front and pure haze), because the mean is
## dominated by the filament texture inside the mass, which is there either way.
const STEEP := 0.045
const MOMENTS := 3

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	# One instance, driven to a settled loud passage, then re-rendered under each control. The
	# field is a pure function of its uniforms, so a control is one set_shader_parameter and
	# needs no re-seeding - which is what keeps this gate to a single boot.
	var sc = load("res://scripts/scenes/vapors.gd").new()
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	vp.add_child(sc)
	sc.init_with_seed(8, "static")          # the `ink` character: hard-fronted, multi-hue
	print("vapor_check: character %s, mood %s, %d plumes"
		% [sc.params.get("character", "?"), sc.params.get("mood", "?"), sc._vapor._plumes.size()])
	var mat: ShaderMaterial = sc._vapor._mat
	var hard0: float = mat.get_shader_parameter("u_hard")
	var stretch0: float = mat.get_shader_parameter("u_stretch")
	var t := 0.0
	t = await _advance(sc, t, 5.0)
	var base := await _grab(vp)
	var base_l := _luma(base)

	# --- PRESENCE ---
	var lit := _fraction(base_l, 0.05, true)
	var dark := _fraction(base_l, 0.02, false)
	print("  presence: %.1f%% of the frame lit, %.1f%% near-black" % [100.0 * lit, 100.0 * dark])
	_want(lit > 0.10 and lit < 0.92, "presence: %.2f lit is not a field with masses in it" % lit)
	_want(dark > 0.04, "presence: only %.2f near-black - no negative space left" % dark)

	# --- FRONT and FILAMENT, paired against their controls, over several moments ---
	var f_hard := 0.0
	var f_soft := 0.0
	var c_str := 0.0
	var c_iso := 0.0
	for k in MOMENTS:
		if k > 0:
			t = await _advance(sc, t, t + 1.3)
		var here := _luma(await _grab(vp))
		f_hard += _edge_step(here)
		c_str += _coherence(here)
		mat.set_shader_parameter("u_hard", 0.0)
		f_soft += _edge_step(_luma(await _grab(vp)))
		mat.set_shader_parameter("u_hard", hard0)
		mat.set_shader_parameter("u_stretch", 1.0)
		c_iso += _coherence(_luma(await _grab(vp)))
		mat.set_shader_parameter("u_stretch", stretch0)
	f_hard /= float(MOMENTS)
	f_soft /= float(MOMENTS)
	c_str /= float(MOMENTS)
	c_iso /= float(MOMENTS)
	print("  front: %.2f%% of lit edges steep at hard=%.2f vs %.2f%% at hard=0 (x%.2f, %d moments)"
		% [100.0 * f_hard, hard0, 100.0 * f_soft, f_hard / maxf(f_soft, 1e-5), MOMENTS])
	_want(f_hard > f_soft * 1.35,
		"front: %.4f steep share against %.4f soft - no front" % [f_hard, f_soft])
	print("  filament: coherence %.4f at stretch=%.2f vs %.4f isotropic (+%.4f)"
		% [c_str, stretch0, c_iso, c_str - c_iso])
	_want(c_str > c_iso + 0.01,
		"filament: %.4f stretched against %.4f isotropic - nothing is drawn out" % [c_str, c_iso])

	# --- STEADINESS, against an inflating control ---
	var jump := 0.0
	var prev := _lit_fast(await _grab(vp))
	for _k in 16:                          # 16 samples x 0.25 s = four seconds of music
		t = await _advance(sc, t, t + 0.25)
		var area := _lit_fast(await _grab(vp))
		jump = maxf(jump, absf(area - prev) / maxf(prev, 0.02))
		prev = area
	# The control drives the amplitudes the way the beat kick used to (x1.7 on every plume, on
	# and off), pushed straight into the uniform so it needs none of the old code back. A
	# measure that could not see THAT would not have caught the bug it exists for.
	var amps: PackedFloat32Array = mat.get_shader_parameter("u_amp")
	var jump_pulsed := 0.0
	prev = _lit_fast(await _grab(vp))
	for k in 8:
		var pushed := PackedFloat32Array()
		for i in amps.size():
			pushed.append(amps[i] * (1.7 if k % 2 == 0 else 1.0))
		mat.set_shader_parameter("u_amp", pushed)
		var area2 := _lit_fast(await _grab(vp))
		jump_pulsed = maxf(jump_pulsed, absf(area2 - prev) / maxf(prev, 0.02))
		prev = area2
	mat.set_shader_parameter("u_amp", amps)
	print("  steadiness: worst area change %.2f%% per quarter second (old beat kick: %.2f%%)"
		% [100.0 * jump, 100.0 * jump_pulsed])
	_want(jump < 0.05, "steadiness: the lit area moves %.1f%% in a quarter second - the masses pulse"
		% (100.0 * jump))
	_want(jump_pulsed > 0.10,
		"steadiness: the control only moved %.1f%% - this measure cannot see an inflating mass"
			% (100.0 * jump_pulsed))

	# --- COLOUR, against a one-colour control ---
	var here_img := await _grab(vp)
	var spread_multi := _hue_spread(here_img)
	var cols: PackedVector3Array = mat.get_shader_parameter("u_col")
	var flat := PackedVector3Array()
	for i in cols.size():
		flat.append(cols[0])
	mat.set_shader_parameter("u_col", flat)
	var spread_mono := _hue_spread(await _grab(vp))
	mat.set_shader_parameter("u_col", cols)
	print("  colour: hue spread %.3f of a turn, one-colour control %.3f" % [spread_multi, spread_mono])
	_want(spread_multi > 0.06,
		"colour: %.3f of a turn between the frame's hues - it is one colour" % spread_multi)
	_want(spread_multi > spread_mono + 0.02,
		"colour: %.3f multi against %.3f mono - the plume colours are not reaching the frame"
			% [spread_multi, spread_mono])

	# --- MOTION, against a rolled-back phase ---
	var before := _luma(await _grab(vp))
	t = await _advance(sc, t, t + 0.4)
	var moved := _luma(await _grab(vp))
	var d_live := _mean_abs_diff(before, moved)
	# The control: put the PHASE back where it was 0.4 s ago and leave the plume positions and
	# amplitudes where they are now. What is left is the field's own flow, alone.
	var now: float = mat.get_shader_parameter("u_time")
	mat.set_shader_parameter("u_time", now - 0.4)
	var d_frozen := _mean_abs_diff(moved, _luma(await _grab(vp)))
	mat.set_shader_parameter("u_time", now)
	print("  motion: %.4f mean change over 0.4 s (phase rolled back: %.4f)" % [d_live, d_frozen])
	_want(d_live > 0.006, "motion: %.4f over 0.4 s - the field is a still" % d_live)
	_want(d_frozen > 0.002,
		"motion: rolling the phase back changed %.4f - the field is not flowing" % d_frozen)

	vp.queue_free()
	print("")
	if _fails.is_empty():
		print("vapor_check: ALL OK - it is vapour.")
	else:
		print("vapor_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(1 if not _fails.is_empty() else 0)


func _want(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


# Run the SIM (not the renderer) from `t` up to `until`, and hand back the new clock.
func _advance(sc, t: float, until: float) -> float:
	var now := t
	while now < until:
		sc.update(_features(now), DT)
		now += DT
	return now


func _grab(vp: SubViewport) -> Image:
	for _i in 3:
		await get_tree().process_frame
	return vp.get_texture().get_image()


# Luminance, once per image, as a flat array - get_pixel is far too slow to call from every
# measure that wants the same frame.
func _luma(img: Image) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(W * H)
	for y in H:
		for x in W:
			var c := img.get_pixel(x, y)
			out[y * W + x] = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b
	return out


func _fraction(l: PackedFloat32Array, level: float, above: bool) -> float:
	var hit := 0
	for i in l.size():
		if (l[i] > level) == above:
			hit += 1
	return float(hit) / float(l.size())


# Lit area on a coarse grid: this one is called two dozen times in the steadiness loop, where a
# full-resolution pass costs about a second each.
func _lit_fast(img: Image) -> float:
	var hit := 0
	var n := 0
	for y in range(0, H, 3):
		for x in range(0, W, 3):
			var c := img.get_pixel(x, y)
			if 0.299 * c.r + 0.587 * c.g + 0.114 * c.b > 0.05:
				hit += 1
			n += 1
	return float(hit) / float(maxi(1, n))


# The share of lit edges that are STEEP - see the STEEP constant for why it is a share and not
# a mean.
func _edge_step(l: PackedFloat32Array) -> float:
	var steep := 0
	var n := 0
	for y in H:
		var row := y * W
		for x in W - 1:
			var a := l[row + x]
			var b := l[row + x + 1]
			if maxf(a, b) > 0.03:
				n += 1
				if absf(a - b) > STEEP:
					steep += 1
	return float(steep) / float(maxi(1, n))


# Structure-tensor coherence per block: (l1 - l2) / (l1 + l2) of the local gradient tensor, which
# is 1 where every gradient in the block points the same way (a drawn-out filament) and near 0
# where they scatter (cloud). Averaged over blocks carrying enough gradient energy.
#
# HIGH-PASSED FIRST, and that is not optional. On raw luminance every block is dominated by the
# mass's own broad ramp, which is coherent whatever the fine structure does: the measure read
# 0.78 both stretched and isotropic - saturated, discriminating nothing. Removing the local mean
# leaves the structure at filament scale, which is what the claim is about. Blocks are wide
# enough to hold several filaments, or one filament fills the block and saturates it again.
func _coherence(l0: PackedFloat32Array) -> float:
	var l := _highpass(l0, 5)
	var block := 32
	var acc := 0.0
	var n := 0
	for by in range(1, H - block, block):
		for bx in range(1, W - block, block):
			var sxx := 0.0
			var syy := 0.0
			var sxy := 0.0
			for y in range(by, by + block):
				for x in range(bx, bx + block):
					var gx := l[y * W + x + 1] - l[y * W + x - 1]
					var gy := l[(y + 1) * W + x] - l[(y - 1) * W + x]
					sxx += gx * gx
					syy += gy * gy
					sxy += gx * gy
			var tr := sxx + syy
			if tr < 0.004:           # a block of empty sky says nothing about alignment
				continue
			var det := sqrt((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy)
			acc += det / tr
			n += 1
	return acc / float(maxi(1, n))


# Separable box blur, subtracted: leaves the detail and drops the broad gradients.
func _highpass(l: PackedFloat32Array, r: int) -> PackedFloat32Array:
	var tmp := PackedFloat32Array()
	tmp.resize(W * H)
	var n := float(2 * r + 1)
	for y in H:
		var row := y * W
		for x in W:
			var acc := 0.0
			for k in range(-r, r + 1):
				acc += l[row + clampi(x + k, 0, W - 1)]
			tmp[row + x] = acc / n
	var out := PackedFloat32Array()
	out.resize(W * H)
	for y in H:
		for x in W:
			var acc := 0.0
			for k in range(-r, r + 1):
				acc += tmp[clampi(y + k, 0, H - 1) * W + x]
			out[y * W + x] = l[y * W + x] - acc / n
	return out


# How far apart the frame's hues are, over the area that actually carries colour: the widest
# circular gap between hue bins holding at least 2% of the coloured pixels.
func _hue_spread(img: Image) -> float:
	var bins := PackedFloat32Array()
	bins.resize(36)
	var total := 0.0
	for y in range(0, H, 2):
		for x in range(0, W, 2):
			var c := img.get_pixel(x, y)
			if c.v < 0.10 or c.s < 0.25:
				continue
			bins[clampi(int(c.h * 36.0), 0, 35)] += 1.0
			total += 1.0
	if total < 100.0:
		return 0.0
	var present: Array = []
	for b in 36:
		if bins[b] / total >= 0.02:
			present.append(float(b) / 36.0)
	var worst := 0.0
	for i in present.size():
		for j in range(i + 1, present.size()):
			var d: float = absf(float(present[i]) - float(present[j]))
			worst = maxf(worst, minf(d, 1.0 - d))     # round the wheel, not along it
	return worst


func _mean_abs_diff(la: PackedFloat32Array, lb: PackedFloat32Array) -> float:
	var acc := 0.0
	for i in la.size():
		acc += absf(la[i] - lb[i])
	return acc / float(la.size())


# Synthetic audio: a settled, moderately loud passage with a travelling spectral peak and beats
# on a half-second grid - the state the look is judged in.
func _features(t: float) -> AudioFeatures:
	var f := AudioFeatures.new()
	f.time = t
	var loud := clampf(0.45 + 0.25 * sin(t * 0.6), 0.0, 1.0)
	var bands := PackedFloat32Array()
	bands.resize(Spectrum.BAND_COUNT)
	for b in Spectrum.BAND_COUNT:
		var x := float(b) / float(maxi(1, Spectrum.BAND_COUNT - 1))
		var peak := 0.5 + 0.4 * sin(t * 0.37)
		bands[b] = clampf(loud * (1.0 - 0.4 * x) * exp(-7.0 * pow(x - peak, 2.0))
			+ 0.15 * loud, 0.0, 1.0)
	f.bands = bands
	f.energy = loud
	f.bass = loud * 0.9
	f.low_mid = loud * 0.75
	f.mid = loud * 0.6
	f.high = loud * 0.45
	f.treble = loud * 0.3
	var phase := fposmod(t, 0.5)
	f.beat = clampf(1.0 - phase * 6.0, 0.0, 1.0)
	f.flux = 0.3 if phase < DT else 0.03
	f.movement = 0.2
	f.beat_period = 0.5
	return f
