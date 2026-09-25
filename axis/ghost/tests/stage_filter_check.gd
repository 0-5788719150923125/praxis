extends SceneTree

## stage_filter_check - that the LOOK filters actually do what their names claim, in pixels.
##
##   tests/run_quiet.sh stage_filter_check
##
## NOT `--headless`: every claim here is a measurement of a rendered frame, and the dummy
## renderer returns nothing from a viewport readback. The wrapper gives it a real GPU on a
## virtual display, so no window appears.
##
## WHY EACH CLAIM IS WORTH ASSERTING - every one of these fails in a way that looks like a
## design decision rather than a bug:
##
##   THE UNIFORM CONTRACT. `set_shader_parameter` on a name the shader does not declare is a
##   SILENT no-op. A filter whose registry entry and uniform disagree therefore ships as a
##   checkbox that moves, saves, reloads and does nothing - the exact failure mode ghost's
##   own dependency table exists to prevent, one level down. This gate reads the shader
##   SOURCE and holds the registry to it.
##
##   OFF IS OFF. The claim is not "looks the same" but that the default show runs the code it
##   has always run: no material on the stage view at all, and a readback BIT-IDENTICAL to
##   the unfiltered picture. A shader that multiplies everything by zero would pass a
##   tolerance test and fail this.
##
##   EACH FILTER'S OWN CLAIM, not merely "the picture changed". "The picture changed" passes
##   on a filter wired to the wrong uniform, on a monochrome that only darkens, on a vignette
##   that darkens the sides instead of the corners, and on grain that fogs the blacks. So each
##   is measured against the property that names it, and the two with a specific WRONG
##   implementation are measured against THAT: a vignette that is circular in the image plane
##   (so on a 16:9 frame the sides fall off far harder than the top) rather than the
##   `length(uv - 0.5)` ellipse everyone writes first, which falls off equally; and grain
##   weighted to the midtones rather than flat, which puts noise in a black that exposed
##   nothing.
##
##   THEY COMBINE, which is the whole point of a set of dials rather than a picker: two
##   filters on together must differ from either one alone AND keep each other's property.
##
## THE TEST IMAGE is built here rather than rendered from a scene, because a measurement needs
## a picture whose answer is known before it is taken: a colour ramp to desaturate, a black, a
## mid-grey and a white patch for the grain weighting, and one small very bright spot on a dark
## field for the bloom threshold.

const Filters_ := preload("res://scripts/filters.gd")

const W := 320
const H := 180
## Where the patches are, in pixels: {name: Rect2i}. The grain weighting is read off the first
## three and the bloom off the last two.
const BLACK := Rect2i(12, 120, 40, 40)
const MID := Rect2i(64, 120, 40, 40)
const WHITE := Rect2i(116, 120, 40, 40)
const SPOT := Rect2i(232, 128, 8, 8)      # the bright spot bloom must spread
const NEAR_SPOT := Rect2i(218, 114, 36, 36)   # ...and the dark ring around it
const FAR := Rect2i(170, 20, 24, 24)      # dark field a long way from the spot

var fails := 0
var checks := 0
var _tex: ImageTexture
var _plain: Image


func _init() -> void:
	_run()


func _run() -> void:
	_tex = _make_test_image()
	_check_uniform_contract()
	await _check_off_is_off()
	_plain = await _render({})
	await _check_each_filter_acts()
	await _check_monochrome()
	await _check_vignette()
	await _check_grain()
	await _check_dust()
	await _check_bloom()
	await _check_pointillism()
	await _check_noir()
	await _check_combining()
	await _check_slip()
	if fails == 0:
		print("stage_filter_check: ALL OK (%d checks, %d filters)"
			% [checks, Filters_.REGISTRY.size()])
	else:
		print("stage_filter_check: %d FAILURE(S) of %d checks" % [fails, checks])
	quit(1 if fails > 0 else 0)


func _ok(cond: bool, msg: String) -> void:
	checks += 1
	if not cond:
		fails += 1
		print("stage_filter_check: FAIL  " + msg)


# --- the contract between the registry and the shader -------------------------


## EVERY REGISTRY KEY HAS A UNIFORM, read off the shader's own source.
##
## The control is deliberately included: a search that cannot fail to find something proves
## nothing about the searches that did find something.
func _check_uniform_contract() -> void:
	var src := FileAccess.get_file_as_string(Filters_.SHADER)
	_ok(not src.is_empty(), "the stage filter shader could not be read at all")
	for key in Filters_.REGISTRY:
		var name := String(Filters_.REGISTRY[key])
		_ok(src.contains("uniform float %s" % name),
			"filter '%s' drives `%s`, which the shader does not declare - "
			% [key, name] + "setting it is a silent no-op")
		_ok(Filters_.LABELS.has(key) and Filters_.BLURBS.has(key)
				and Filters_.DEFAULTS.has(key),
			"filter '%s' is missing a label, a blurb or a default" % key)
		var d := float(Filters_.DEFAULTS[key])
		_ok(d > 0.0 and d <= 1.0,
			"filter '%s' has a default of %f - switching it on would do nothing" % [key, d])
	_ok(not src.contains("uniform float u_not_a_filter"),
		"the control is wrong - the shader must not declare a filter nobody registered")
	# The size uniform is not a filter and so is not in the registry, but everything that is
	# sized in pixels depends on it arriving.
	_ok(src.contains("uniform vec2 u_size"),
		"the shader has no u_size - the dots and the grain cannot be sized in pixels")


# --- off is off ---------------------------------------------------------------


func _check_off_is_off() -> void:
	var rect := TextureRect.new()
	Filters_.apply(rect, {}, Vector2(W, H))
	_ok(rect.material == null,
		"an empty filter set left a material on the stage view")
	# ...and a filter at zero is the same as no filter, because `sanitize` drops it.
	Filters_.apply(rect, {"monochrome": 0.0}, Vector2(W, H))
	_ok(rect.material == null, "a filter dialled to zero still hung a material on the view")
	Filters_.apply(rect, {"monochrome": 1.0}, Vector2(W, H))
	_ok(rect.material != null, "the control is wrong - a live filter hung no material")
	# AND IT COMES BACK OFF. Writing only the live uniforms would leave the last value of a
	# filter that has just been switched off still burning.
	Filters_.apply(rect, {}, Vector2(W, H))
	_ok(rect.material == null, "switching the last filter off did not remove the material")
	rect.free()

	var bare := await _render({})
	var src := _tex.get_image()
	src.convert(bare.get_format())
	_ok(_max_diff(bare, src) == 0.0,
		"an unfiltered render is not bit-identical to its source (max channel diff %f)"
		% _max_diff(bare, src))


# --- every filter acts, and acts alone ----------------------------------------


## HOW MANY PIXELS MOVED, not by how much on average.
##
## The mean absolute difference was the first metric here and it is the wrong instrument for
## half this registry. A MEAN cannot see a sparse effect: grain is deliberately heavy-tailed -
## most of the frame carries nearly nothing so that the grains which do land read as grains -
## and rebuilding it that way took the mean from 0.03 to 0.005 while the picture visibly
## gained grain. A mean also flatters the opposite mistake, a filter that tints everything
## faintly and does nothing worth seeing. The share of pixels actually moved answers "is this
## doing something" for both shapes.
func _check_each_filter_acts() -> void:
	for key in Filters_.REGISTRY:
		if key == "slip" or key == "dust":
			continue          # sparse and per-frame - see _check_slip / _check_dust
		var got := await _render({String(key): 1.0})
		var f := _changed_fraction(got, _plain)
		print("stage_filter_check: '%s' moves %.1f%% of the frame at 1.0" % [key, f * 100.0])
		_ok(f > 0.02, "filter '%s' at full strength moved %.2f%% of the pixels - "
			% [key, f * 100.0] + "it is doing nothing")
		# ...and its DEFAULT does something too, which is what a freshly ticked box applies.
		var soft := await _render({String(key): float(Filters_.DEFAULTS[key])})
		_ok(_changed_fraction(soft, _plain) > 0.005,
			"filter '%s' at its default of %.2f moved %.2f%% of the pixels - ticking it "
			% [key, float(Filters_.DEFAULTS[key]), _changed_fraction(soft, _plain) * 100.0]
			+ "reads as broken")


# --- each filter's own claim --------------------------------------------------


## COLOUR IS GONE, measured as chroma (max channel minus min), not as "the picture changed".
func _check_monochrome() -> void:
	var got := await _render({"monochrome": 1.0})
	var before := _mean_chroma(_plain)
	var after := _mean_chroma(got)
	_ok(before > 0.15, "the control is wrong - the test image is not colourful (%.3f)" % before)
	_ok(after < 0.02, "monochrome left %.3f of chroma behind (was %.3f)" % [after, before])
	# ...and it must not just be darkening everything: brightness is roughly preserved.
	_ok(absf(_mean_luma(got) - _mean_luma(_plain)) < 0.06,
		"monochrome moved the picture's brightness by %.3f - it is grading, not desaturating"
		% absf(_mean_luma(got) - _mean_luma(_plain)))


## THE VIGNETTE IS TWO SHAPES ALONG ONE DIAL, and each half is asserted where it applies.
##
## IN THE MIDDLE OF THE RANGE IT IS A LENS, and the discriminating measurement is not "did the
## corners go dark" - an ellipse stretched to the frame does that too. It is the LEFT edge
## against the TOP edge: on this 16:9 frame the mid-left is 0.87 of the way to a corner and the
## mid-top only 0.49, so a circular falloff darkens the sides far harder than the top, where
## the `length(uv - 0.5)` version everyone writes first darkens them exactly the same.
##
## AT THE TOP OF THE RANGE IT CLOSES ON EVERY EDGE, which is the reported defect: "even at 1.0
## it doesn't cover ANY of the edges - it only covers the corners". Obeying the circle is what
## made that true, so past CLOSE_ONSET the falloff crosses to an edge distance. All four edge
## midpoints have to go dark, and the centre still must not.
func _check_vignette() -> void:
	var lens := await _render({"vignette": 0.5})
	var left := _fall(lens, Rect2i(0, H / 2 - 12, 24, 24))
	var top := _fall(lens, Rect2i(W / 2 - 12, 0, 24, 24))
	_ok(left > 0.05, "at half strength the vignette did not darken the sides at all (%.3f)" % left)
	_ok(left > top * 2.0,
		"at half strength the mid-LEFT fell by %.3f and the mid-TOP by %.3f - on a 16:9 frame "
		% [left, top] + "a circular falloff separates those by a long way, an aspect-stretched "
		+ "one not at all")

	var shut := await _render({"vignette": 1.0})
	var corner := 0.5 * (_fall(shut, Rect2i(0, 0, 24, 24))
		+ _fall(shut, Rect2i(W - 24, H - 24, 24, 24)))
	var middle := _fall(shut, Rect2i(W / 2 - 16, H / 2 - 16, 32, 32))
	_ok(corner > 0.3, "at full strength the vignette barely darkened the corners (%.3f)" % corner)
	_ok(middle < 0.02, "the vignette darkened the CENTRE by %.3f" % middle)
	var edges := {
		"left": Rect2i(0, H / 2 - 12, 24, 24),
		"right": Rect2i(W - 24, H / 2 - 12, 24, 24),
		"top": Rect2i(W / 2 - 12, 0, 24, 24),
		"bottom": Rect2i(W / 2 - 12, H - 24, 24, 24),
	}
	for name in edges:
		var f := _fall(shut, edges[name] as Rect2i)
		_ok(f > 0.45,
			"at full strength the %s EDGE fell by only %.3f - the vignette is still only "
			% [name, f] + "covering the corners")


## THE GATE SLIP IS ITS OWN FILTER. It was the top of the Grain dial, so grain past 0.55 tore
## the picture whether that was wanted or not. Held: the slip fires on some film frames at full
## strength and at its default, and on a frame where it fires, grain alone leaves that band where
## it was. Frames are asked for by number (`u_frame`), so this is the same answer every run.
func _check_slip() -> void:
	var fired := -1
	for f in 60:
		if _changed_fraction(await _render({"slip": 1.0}, f), _plain) > 0.02:
			fired = f
			break
	_ok(fired >= 0, "gate slip at 1.0 never moved the picture in 60 film frames")
	var soft := false
	for f in 120:
		if _changed_fraction(await _render({"slip": float(Filters_.DEFAULTS["slip"])}, f), _plain) > 0.005:
			soft = true
			break
	_ok(soft, "gate slip at its default never moved the picture in 120 film frames")
	if fired < 0:
		return
	var quiet := 0
	for f in 60:
		if _changed_fraction(await _render({"slip": 1.0}, f), _plain) <= 0.02:
			quiet += 1
	_ok(quiet > 30, "gate slip at 1.0 fired on %d of 60 frames - it is a tear, not a wobble" % (60 - quiet))
	# THE BAND: rows of the hue ramp (the top half) the slip moved - a slip shifts nearly every
	# pixel of the ramp in its band, where grain changes luminance and leaves hue alone. The
	# first frame to fire may have slipped the bottom half, so find one that crossed the ramp.
	var rows: Array = []
	for f in range(fired, fired + 120):
		var slipped := await _render({"slip": 1.0}, f)
		rows.clear()
		for y in range(0, H / 2, 2):
			var moved := 0
			for x in range(0, W, 2):
				var p := slipped.get_pixel(x, y)
				var q := _plain.get_pixel(x, y)
				if absf(p.h - q.h) > 0.005 and absf(p.h - q.h) < 0.995:
					moved += 1
			if moved > W / 4:
				rows.append(y)
		if rows.size() >= 3:
			fired = f
			break
	_ok(rows.size() >= 3, "no slip crossed the hue ramp in 120 film frames")
	var grain := await _render({"static": 1.0}, fired)
	var in_band := 0
	var n := 0
	for y in rows:
		for x in range(0, W, 2):
			var p := grain.get_pixel(x, y)
			var q := _plain.get_pixel(x, y)
			if absf(p.h - q.h) > 0.005 and absf(p.h - q.h) < 0.995:
				in_band += 1
			n += 1
	_ok(n > 0 and float(in_band) / float(n) < 0.1,
		"grain alone shifted the slip band's hue on %.0f%% of it - grain is still tearing"
		% (100.0 * float(in_band) / maxf(float(n), 1.0)))


## GRAIN LIVES IN THE MIDTONES. Flat noise fogs a black that exposed nothing, which is the one
## place it must not, and it is exactly what an unweighted version does.
func _check_grain() -> void:
	var got := await _render({"static": 1.0})
	var mid := _added_variance(got, MID)
	var black := _added_variance(got, BLACK)
	var white := _added_variance(got, WHITE)
	_ok(mid > 0.0005, "grain added no noise to the midtones (%.6f)" % mid)
	_ok(mid > black * 6.0,
		"grain put %.6f of noise in the blacks against %.6f in the midtones - it is flat, "
		% [black, mid] + "not weighted")
	_ok(mid > white * 6.0,
		"grain put %.6f of noise in the blown highlights against %.6f in the midtones"
		% [white, mid])


## BLOOM SPREADS FROM THE BRIGHT PART AND LEAVES THE DARK ALONE. A bloom with no threshold is
## a fogged lens and lifts the far field just as much.
func _check_bloom() -> void:
	var got := await _render({"bloom": 1.0})
	var near := _mean_luma_of(got, NEAR_SPOT) - _mean_luma_of(_plain, NEAR_SPOT)
	var far := _mean_luma_of(got, FAR) - _mean_luma_of(_plain, FAR)
	_ok(near > 0.01, "the bright spot did not bleed into the dark around it (%.4f)" % near)
	_ok(far < near * 0.35,
		"bloom lifted the far dark field by %.4f against %.4f beside the spot - "
		% [far, near] + "it is fogging the frame rather than blooming the highlights")


## DOTS HAVE EDGES, and re-laying a picture as dots is not the same as washing it out. Two
## claims together: local contrast goes UP (the rim of every dab), and the picture's overall
## colour stays where it was (a dab takes its colour from where it sits).
func _check_pointillism() -> void:
	var got := await _render({"pointillism": 0.6})
	var steps_before := _step_fraction(_plain)
	var steps_after := _step_fraction(got)
	_ok(steps_after > steps_before * 2.0,
		"pointillism raised the fraction of hard pixel steps from %.4f to only %.4f - "
		% [steps_before, steps_after] + "there are no dot rims in it")
	_ok(absf(_mean_luma(got) - _mean_luma(_plain)) < 0.14,
		"pointillism moved the picture's brightness by %.3f - the ground is swallowing it"
		% absf(_mean_luma(got) - _mean_luma(_plain)))


## CONTRAST UP, AND STILL IN COLOUR. Noir grades whatever it is handed; a version that
## desaturated on its own would make Monochrome redundant and Technicolor impossible.
func _check_noir() -> void:
	var got := await _render({"noir": 1.0})
	# Measured on the RAMP, which is where the tones are. Over the whole frame the dark field
	# and the black/white patches are already at the ends of the scale and cannot be expanded,
	# so they dilute exactly the thing under test.
	var ramp := Rect2i(0, 0, W, H / 2)
	_ok(_luma_spread_of(got, ramp) > _luma_spread_of(_plain, ramp) * 1.25,
		"noir raised the tonal spread from %.4f to only %.4f"
		% [_luma_spread_of(_plain, ramp), _luma_spread_of(got, ramp)])
	_ok(_mean_chroma(got) > 0.04,
		"noir took the colour out by itself (%.3f chroma left) - that is Monochrome's job"
		% _mean_chroma(got))


## TWO AT ONCE, which is the reason these are dials and not a picker.
func _check_combining() -> void:
	var mono := await _render({"monochrome": 1.0})
	var grain := await _render({"static": 1.0})
	var both := await _render({"monochrome": 1.0, "static": 1.0})
	_ok(_mean_diff(both, mono) > 0.002,
		"monochrome + grain is indistinguishable from monochrome alone")
	_ok(_mean_diff(both, grain) > 0.01,
		"monochrome + grain is indistinguishable from grain alone")
	# ...and each keeps its own property in the pair.
	_ok(_mean_chroma(both) < 0.02,
		"grain put the colour back into a monochrome picture (%.3f)" % _mean_chroma(both))
	# MOTTLE leaves whole regions nearly clean for a while, so one frame can catch the mid patch
	# in a quiet spell: the best of a few pinned frames, never the clock (which made this pass
	# or fail on how long the checks before it took).
	var mid_var := 0.0
	for f in 4:
		mid_var = maxf(mid_var, _added_variance(await _render({"monochrome": 1.0, "static": 1.0}, f * 7), MID))
	_ok(mid_var > 0.0005, "monochrome flattened the grain out of the midtones")
	# THE WHOLE STACK, which is the setting most likely to be used and the one most likely to
	# hit a clamp: it must still be a picture rather than black or white.
	var all := {}
	for k in Filters_.REGISTRY:
		all[String(k)] = float(Filters_.DEFAULTS[k])
	var everything := await _render(all)
	var l := _mean_luma(everything)
	_ok(l > 0.03 and l < 0.9,
		"every filter on at its default renders at mean luma %.3f - that is not a picture" % l)


# --- rendering and measuring --------------------------------------------------


## The stage view's arrangement in miniature: the picture in a TextureRect, the filter material
## on the rect, read back out of a SubViewport.
func _render(amounts: Dictionary, frame := -1) -> Image:
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var tr := TextureRect.new()
	tr.texture = _tex
	tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	tr.size = Vector2(W, H)
	Filters_.apply(tr, amounts, Vector2(W, H))
	if frame >= 0 and tr.material != null:
		(tr.material as ShaderMaterial).set_shader_parameter("u_frame", frame)
	vp.add_child(tr)
	root.add_child(vp)
	for i in 4:
		await process_frame
	var got := vp.get_texture().get_image()
	vp.queue_free()
	return got


## A picture whose answer is known before it is measured: a saturated hue ramp over the top
## half, three exposure patches along the bottom, and one small very bright spot on a dark
## field for the bloom threshold to find.
func _make_test_image() -> ImageTexture:
	var img := Image.create(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var c: Color
			if y < H / 2:
				# A full hue sweep at a brightness ramp - colour to take out, tones to curve.
				c = Color.from_hsv(float(x) / float(W), 0.85,
					0.25 + 0.55 * (float(y) / float(H / 2)))
			else:
				c = Color(0.06, 0.06, 0.07)      # the dark field the patches sit on
			img.set_pixel(x, y, c)
	_fill_rect(img, BLACK, Color(0.0, 0.0, 0.0))
	_fill_rect(img, MID, Color(0.5, 0.5, 0.5))
	_fill_rect(img, WHITE, Color(1.0, 1.0, 1.0))
	_fill_rect(img, SPOT, Color(1.0, 0.97, 0.9))
	return ImageTexture.create_from_image(img)


func _fill_rect(img: Image, r: Rect2i, c: Color) -> void:
	for y in range(r.position.y, r.position.y + r.size.y):
		for x in range(r.position.x, r.position.x + r.size.x):
			img.set_pixel(x, y, c)


func _mean_diff(a: Image, b: Image) -> float:
	var sum := 0.0
	for y in range(0, H, 2):
		for x in range(0, W, 2):
			var p := a.get_pixel(x, y)
			var q := b.get_pixel(x, y)
			sum += absf(p.r - q.r) + absf(p.g - q.g) + absf(p.b - q.b)
	return sum / (float(W * H) / 4.0 * 3.0)


## The share of sampled pixels whose colour moved at all - a channel step of more than 1/255
## twice over, so filtering noise and dithering do not count as an effect.
func _changed_fraction(a: Image, b: Image, eps := 0.008) -> float:
	var hits := 0
	var n := 0
	for y in range(0, H, 2):
		for x in range(0, W, 2):
			var p := a.get_pixel(x, y)
			var q := b.get_pixel(x, y)
			if absf(p.r - q.r) > eps or absf(p.g - q.g) > eps or absf(p.b - q.b) > eps:
				hits += 1
			n += 1
	return float(hits) / maxf(float(n), 1.0)


func _max_diff(a: Image, b: Image) -> float:
	var worst := 0.0
	for y in H:
		for x in W:
			var p := a.get_pixel(x, y)
			var q := b.get_pixel(x, y)
			worst = maxf(worst, maxf(absf(p.r - q.r), maxf(absf(p.g - q.g), absf(p.b - q.b))))
	return worst


func _mean_chroma(img: Image) -> float:
	var sum := 0.0
	var n := 0
	# The colour ramp only - the grey patches have no chroma to lose and would dilute it.
	for y in range(0, H / 2, 2):
		for x in range(0, W, 2):
			var c := img.get_pixel(x, y)
			sum += maxf(c.r, maxf(c.g, c.b)) - minf(c.r, minf(c.g, c.b))
			n += 1
	return sum / maxf(float(n), 1.0)


func _mean_luma(img: Image) -> float:
	var sum := 0.0
	var n := 0
	for y in range(0, H, 2):
		for x in range(0, W, 2):
			sum += _luma(img.get_pixel(x, y))
			n += 1
	return sum / maxf(float(n), 1.0)


func _mean_luma_of(img: Image, r: Rect2i) -> float:
	var sum := 0.0
	var n := 0
	for y in range(r.position.y, r.position.y + r.size.y):
		for x in range(r.position.x, r.position.x + r.size.x):
			sum += _luma(img.get_pixel(x, y))
			n += 1
	return sum / maxf(float(n), 1.0)


## How much darker a region got, as a fraction of what it was.
func _fall(img: Image, r: Rect2i) -> float:
	var was := _mean_luma_of(_plain, r)
	if was <= 0.001:
		return 0.0
	return clampf(1.0 - _mean_luma_of(img, r) / was, 0.0, 1.0)


## The variance a filter ADDED to a flat patch. The patch is flat in the source, so anything
## here is the filter's own.
func _added_variance(img: Image, r: Rect2i) -> float:
	return maxf(_variance(img, r) - _variance(_plain, r), 0.0)


func _variance(img: Image, r: Rect2i) -> float:
	var vals: Array = []
	for y in range(r.position.y, r.position.y + r.size.y):
		for x in range(r.position.x, r.position.x + r.size.x):
			vals.append(_luma(img.get_pixel(x, y)))
	var mean := 0.0
	for v in vals:
		mean += float(v)
	mean /= maxf(float(vals.size()), 1.0)
	var acc := 0.0
	for v in vals:
		acc += pow(float(v) - mean, 2.0)
	return acc / maxf(float(vals.size()), 1.0)


## The share of horizontally adjacent pixel pairs that differ by more than a tenth - a dot's
## rim is such a pair and a smooth gradient has none.
func _step_fraction(img: Image) -> float:
	var hits := 0
	var n := 0
	for y in range(0, H, 2):
		for x in range(0, W - 1):
			if absf(_luma(img.get_pixel(x, y)) - _luma(img.get_pixel(x + 1, y))) > 0.1:
				hits += 1
			n += 1
	return float(hits) / maxf(float(n), 1.0)


## Standard deviation of luma over a region - "contrast" as a number.
func _luma_spread_of(img: Image, r: Rect2i) -> float:
	var vals: Array = []
	for y in range(r.position.y, r.position.y + r.size.y, 2):
		for x in range(r.position.x, r.position.x + r.size.x, 2):
			vals.append(_luma(img.get_pixel(x, y)))
	var mean := 0.0
	for v in vals:
		mean += float(v)
	mean /= maxf(float(vals.size()), 1.0)
	var acc := 0.0
	for v in vals:
		acc += pow(float(v) - mean, 2.0)
	return sqrt(acc / maxf(float(vals.size()), 1.0))


func _luma(c: Color) -> float:
	return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b


## DUST IS DIRT ON THE PRINT: sparse marks that change every film frame - some of the picture
## moved on every frame, never most of it, a different place each frame, and none at all on a
## frame when the dial is off. Frames are asked for by number (`u_frame`).
func _check_dust() -> void:
	var moved: Array = []
	var images: Array = []
	for f in 6:
		var got := await _render({"dust": 1.0}, f)
		images.append(got)
		moved.append(_changed_fraction(got, _plain))
	var lo := 1.0
	var hi := 0.0
	for m in moved:
		lo = minf(lo, float(m))
		hi = maxf(hi, float(m))
	_ok(lo > 0.0005, "dust at 1.0 left a frame untouched (%.4f moved)" % lo)
	_ok(hi < 0.15, "dust at 1.0 covered %.1f%% of a frame - it is a fog, not dirt" % (hi * 100.0))
	_ok(_mean_diff(images[0], images[1]) > 0.0002, "dust did not change from one film frame to the next")
	# Presence is by chance per frame, so the default is judged over a half second of frames.
	var soft_hit := 0
	for f in 12:
		var soft := await _render({"dust": float(Filters_.DEFAULTS["dust"])}, f)
		if _changed_fraction(soft, _plain) > 0.0:
			soft_hit += 1
	_ok(soft_hit >= 6, "dust at its default touched only %d of 12 frames" % soft_hit)
	# A LOW DIAL IS RARE, not a constant trickle: a fixed per-frame count made 0.05 a speck on
	# every single frame, 24 a second.
	var rare_hit := 0
	for f in 48:
		var faint := await _render({"dust": 0.05}, f)
		if _changed_fraction(faint, _plain) > 0.0:
			rare_hit += 1
	_ok(rare_hit <= 4, "dust at 0.05 marked %d of 48 frames - a low dial should be a rare event" % rare_hit)
