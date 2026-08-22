extends GhostScene

## Fractal zoom - falling into (or back out of) an escape-time set, forever.
##
## The frame is one fragment program over the complex plane (shaders/fractal_field.gdshader);
## this half chooses WHICH set, WHERE in it, HOW it is coloured, and how the camera moves
## through it. Every one of those is sampled per session, because the whole point of the scene
## is the spectrum: the same three lines of arithmetic are a spiral, a seahorse valley, a
## dendrite, a filigree web or a burning coastline depending on six or seven rolls.
##
## THE ZOOM IS THE SUBJECT, IT GOES ONE WAY, AND IT NEVER TURNS ROUND. A fractal zoom is
## normally a fall inward; the pull - the same journey run backwards, structure assembling out
## of the middle of the frame and receding - is a different and rarer picture, so it is a
## first-class mode here rather than a negative sign somebody could set. `direction` is one of:
##   PUSH - inward. The classic, and two rolls in three.
##   PULL - outward, opening from a deep point back toward the whole set.
##
## THERE USED TO BE A THIRD, `breathe`, which turned round on its own clock, and it is gone.
## Reported: "it rolled forward for a bit, before slowing and rolling backwards again... the
## most interesting part of a fractal zoom is the continuous zoom-in, or continuous zoom-out.
## Reversing direction just reveals the same patterns we've already seen." That is right, and
## it is right about the mode as well as about the bug - a reversal is the one camera move
## that is guaranteed to show you nothing new, because it is the shot you have just watched,
## played backwards.
##
## AND IT WAS A BUG BEFORE IT WAS A MODE. The breathe clock was armed in `build_params` before
## the direction was even chosen, so every push and every pull turned round after 11-26
## seconds too. Measured over 240 seeds: ONE HUNDRED PERCENT of scenes reversed inside thirty
## seconds, the first turn landing at 18 s on average, and the net travel over a whole scene
## was under two e-folds because the picture kept coming back to where it started.
##
## SPEED IS A JOURNEY TIME, NOT A RATE, and that is what keeps the fall continuous now. What a
## family can resolve varies by twenty-fold (see the floors below), so a sampled rate makes a
## shallow family run out in twenty seconds and a deep one crawl. Sampling how long the whole
## descent should TAKE and dividing by the depth available inverts that: every instance falls
## for minutes, the deep families plunge and the shallow ones drift, and neither reaches a
## bound while anybody is watching. The music scales the rate around that - loud passages fall
## faster - which is the only thing audio is allowed to touch here. The SET itself never
## deforms to the music, because a fractal that wobbles on the beat stops being a fractal and
## becomes a plasma effect.
##
## THE ZOOM FLOOR, and it is the one hard constraint in the file. Zooming is a scale going to
## zero, and float32 runs out at about 3e-4 of the starting view - twelve seconds, and then
## the picture visibly turns to blocks. The quadratic Mandelbrot escapes that through
## PERTURBATION (see the shader's header): the reference orbit is iterated here in GDScript,
## whose floats are 64-bit, and the shader only ever handles the small difference from it.
## Every other family iterates directly and is CLAMPED to the depth float32 can hold. So the
## floor is a property of the family, `_zl_min` is set from it, and the zoom EASES TO REST
## against it rather than grinding into it or bouncing off it. Arriving is not a failure - the
## journey is sized so it takes minutes - but if a scene is held long enough to get there, the
## camera settling is the graceful end of the shot and a rewind is not.
##
## FLOAT64 IS NOT OPTIONAL ON THIS SIDE. The anchor is held as two loose `float`s and never as
## a Vector2: Godot's vectors are 32-bit, so a Vector2 anchor would quietly round away the
## last six digits of the coordinate - which at depth is the entire location.
##
## WHAT THE SEED DECIDES: the family (mandelbrot / julia / burning ship / tricorn / celtic /
## buffalo) and its integer power; for a Julia set, where its parameter sits on the cardioid
## and how it drifts; the anchor, found by a local search that walks toward the boundary; the
## colouring (smooth escape count, stripe average, three orbit traps, exterior distance) and
## its parameters; the palette, built as a seamless cyclic ramp from a [Scheme]; how the
## interior is treated; the iteration budget; the zoom's direction, speed and turning points;
## the frame's slow rotation; and the vignette.

const SHADER := preload("res://shaders/fractal_field.gdshader")

## Iteration ceiling. The escape loop runs per pixel per frame, so this is a real frame cost
## and not a free knob: most pixels leave in a handful of steps, but the boundary - which is
## most of a deep frame - runs the whole way.
##
## IT IS ALSO WHAT DECIDES HOW DEEP THE SCENE MAY GO, which was not obvious and cost a render
## to see. Too small a budget does not merely lose fine detail: points that WOULD escape are
## classified interior, so the set grows a flat black skirt, and the pixels just outside it
## escape at wildly different counts and come out as confetti. The first cut ran 335 iterations
## at a depth needing about 1400 and most of the frame was exactly that - a black blob with a
## band of grey mush around it. FLOOR_PERTURB is set from this, not the other way round.
const ITER_MAX := 3000

## How long a reference orbit may be. The shader rebases whenever it runs off the end, so a
## short one costs accuracy rather than correctness - but the anchor search deliberately hunts
## for points that take a long time to escape, so this is sized for them.
const REF_MAX := 4096

## The palette ramp's resolution. It is sampled with fract() and interpolated, so this is
## about banding rather than about how many colours there are.
const RAMP := 256

## Families, by shader index. Each is the same iteration under a different fold of the plane,
## which is why they can share one program - and why the catalogue is this cheap to widen.
const FAMILIES := ["mandelbrot", "julia", "burning_ship", "tricorn", "celtic", "buffalo"]

## Colourings, by shader index.
const COLOURS := ["smooth", "stripe", "trap_point", "trap_cross", "trap_circle", "distance"]

## The zoom's directions, as a bag rather than a list: the fall inward is the picture people
## come to a fractal for, and the climb out is the one worth being rarer. See the class doc for
## why there is no third entry any more.
const DIRECTION_BAG := ["push", "push", "pull"]

## How long the WHOLE descent takes at nominal loudness, in seconds - sampled per instance, and
## then divided by the depth available to get a rate. See the class doc.
##
## The floor of the range is what makes the "never turns round" claim true in practice rather
## than in principle: a scene holds for at most `Director.max_hold` x `pace_calm_scale` x
## `pacing`, which is about 60 s at the pacing this was reported on and 134 s at the top of
## that dial. At 95 s nominal - 65 s if the passage is loud the whole way - a fast instance can
## still reach its bound in a long-held scene, and that is what the ease-to-rest below is for.
const JOURNEY_MIN := 95.0
const JOURNEY_MAX := 260.0

## How many e-folds out from a bound the rate starts easing off. Wide enough that the slowing
## reads as the camera settling rather than as the zoom stalling.
const ARRIVE_SPAN := 1.6

## The family roll, weighted rather than uniform. The quadratic Mandelbrot is the one that can
## be flown into indefinitely (it is the only one perturbation is implemented for), so it earns
## the largest share; the folded families are striking but they are also the ones whose
## straight-edged, repetitive structure wears out fastest on screen.
const FAMILY_BAG := [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5]

## Scale (half the frame's height, in complex units) the widest view sits at. About the whole
## Mandelbrot set, and a comfortable window on every other family too.
const SCALE_WIDE := 1.45

## The deepest scale each path may reach. The direct one is float32's limit with a margin
## under it; the perturbed one is set by how far the anchor search can resolve in float64,
## and 1e-9 is about twenty e-folds of continuous fall - longer than any scene's hold.
## The deepest scale each path may reach. The direct one is float32's limit with a margin under
## it; the perturbed one is set by ITER_MAX - about sixteen e-folds is as far as the escape
## count can be resolved inside the budget, which at the speeds sampled here is one to five
## minutes of continuous fall and longer than any scene holds.
const FLOOR_DIRECT := 3.0e-4
const FLOOR_PERTURB := 1.0e-7

var _mat := ShaderMaterial.new()
var _quad: Layer.FieldQuad = null
var _sch: Scheme

# The definition.
var _family := 0
var _power := 2
var _colour := 0
var _perturb := false
var _iter_base := 320
var _iter_gain := 90.0
var _cycle := 0.06
var _stripe := 5.0
var _trap_p := Vector2.ZERO
var _trap_r := 0.6
var _interior := 0
var _in_col := Color(0.02, 0.02, 0.05)
var _gain := 1.0
var _vig := 0.25
var _bail := 256.0

## THE ANCHOR, in float64 and never in a Vector2 - see the class doc.
var _cx := -0.75
var _cy := 0.0

# The Julia parameter and its drift. Held apart from the anchor because for a Julia set the
# plane being drawn is the z-plane and the parameter is what makes it that particular set.
var _jr := 0.0                # angle on the cardioid
var _jpush := 0.0             # how far off the cardioid boundary (0 = on it)
var _jx := -0.8
var _jy := 0.156

# The zoom.
var _zl := 0.0                # ln of the current scale
var _zl_min := 0.0
var _zl_max := 0.0
var _speed := 0.2             # e-folds per second at nominal loudness
var _dir := -1.0              # -1 inward (scale shrinking), +1 outward. NEVER REASSIGNED.

# Live state.
var _rot := 0.0
var _rot_rate := 0.0
var _phase := 0.0
var _phase_rate := 0.05
var _glow := 0.0
var _energy := 0.0
var _ch := Vector2.ZERO
var _ch_seeded := false


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# FIRST, before anything below writes a uniform. The reference orbit is uploaded from inside
	# _find_anchor, and a parameter written against a material with no shader on it yet is a
	# parameter that may not survive the shader being attached.
	_mat.shader = SHADER
	_sch = Scheme.pick(rng)

	_family = int(FAMILY_BAG[rng.randi() % FAMILY_BAG.size()])
	# The power. Two is the Mandelbrot everybody knows; three and up are the multibrots, which
	# are rounder, more symmetric and much less familiar - worth having, worth being rarer. The
	# Mandelbrot leans harder on two because PERTURBATION IS THE QUADRATIC MANDELBROT'S ALONE
	# (see the shader header), and that is what makes a long fall possible at all.
	var keep_two := 0.80 if _family == 0 else 0.55
	_power = 2 if rng.randf() < keep_two else rng.randi_range(3, 6)
	_perturb = (_family == 0 and _power == 2)

	if _family == 1:
		# A Julia set is chosen by its parameter, and the interesting ones live within a hair of
		# the main cardioid's boundary: `c = e^(it)/2 - e^(2it)/4`. Inside it the set is a
		# deformed disc, well outside it is dust, and ON it are the dendrites and rabbits worth
		# looking at. So the parameter is placed on the boundary and pushed a sampled hair off it.
		#
		# AND THEN IT HOLDS STILL. Drifting it was the first cut and it is a lovely effect for
		# about ten seconds: the whole set morphs while the camera falls through it. It is also
		# incompatible with zooming, and that was measured rather than reasoned - the anchor is
		# a point on THIS set's boundary, a different parameter is a different set, and by
		# twenty-five seconds the boundary was nowhere near the frame and two seeds in three
		# rendered a flat wash. A set that morphs and a set that can be flown into are two
		# different scenes; this one is the second.
		_jr = rng.randf() * TAU
		_jpush = rng.randf_range(-0.010, 0.014)
		_julia_at(_jr)

	# THE ANCHOR. A random point is almost always either far outside the set (a flat wash) or
	# deep inside it (a flat hole); the only place worth flying into is the boundary, so it is
	# searched for rather than sampled. See _find_anchor.
	var floor_scale := FLOOR_PERTURB if _perturb else FLOOR_DIRECT
	_find_anchor(rng, floor_scale)

	# The colouring, which is where most of the visual range lives. The distance estimate needs
	# a derivative the perturbed path does not carry, so it is offered only on the direct one -
	# chosen here rather than in the shader so the two can never disagree.
	var pool: Array = [0, 0, 1, 1, 2, 3, 4]
	if not _perturb and (_family == 0 or _family == 1):
		pool.append(5)
	_colour = int(pool[rng.randi() % pool.size()])
	_stripe = float(rng.randi_range(2, 9))
	var ta := rng.randf() * TAU
	var td := rng.randf_range(0.0, 0.9)
	_trap_p = Vector2(cos(ta) * td, sin(ta) * td)
	_trap_r = rng.randf_range(0.25, 1.10)
	# A bigger escape radius makes the smooth iteration count smoother; a small one makes the
	# bands crisp. Trap colourings want a large one so the orbit gets a chance to pass the trap.
	_bail = rng.randf_range(64.0, 400.0) if _colour < 2 else rng.randf_range(400.0, 4000.0)

	# The budget, and its RAMP WITH DEPTH is the part that matters - see ITER_MAX. Escape takes
	# longer the nearer the boundary a point is, so a fixed budget that looks generous on the
	# whole set draws a deep view as a black skirt with confetti round it.
	_iter_base = rng.randi_range(420, 700)
	_iter_gain = rng.randf_range(115.0, 200.0)
	# One palette turn per 30 to 140 iterations. Faster than this and the bands are finer than
	# the screen can resolve, which the mip sampling then averages back to a flat wash - the
	# first cut ran to 0.075 (a turn every 13) and most of the frame came out as grey confetti.
	_cycle = rng.randf_range(0.007, 0.033)
	_phase_rate = rng.randf_range(-0.045, 0.045)
	_gain = rng.randf_range(0.92, 1.18)
	_vig = rng.randf_range(0.10, 0.42)

	# The interior. Flat is right on a wide view where the set is a silhouette; on a deep one
	# the frame can be most interior, and then how close the orbit came is the only structure
	# there is - so the deep paths lean toward the treatments that show it.
	_interior = [0, 0, 1, 2][rng.randi() % 4]
	var iv := rng.randf_range(0.02, 0.16)
	_in_col = Color.from_hsv(fposmod(_sch.hue + rng.randf_range(-0.1, 0.1), 1.0),
		clampf(_sch.sat * rng.randf_range(0.3, 0.9), 0.0, 1.0), iv)

	# The zoom.
	var dir_name := String(DIRECTION_BAG[rng.randi() % DIRECTION_BAG.size()])
	_zl_max = log(SCALE_WIDE)
	_zl_min = log(floor_scale)
	if dir_name == "pull":
		_dir = 1.0
		# A pull begins DEEP - the whole picture of it is structure receding out of the
		# middle of the frame, and that only happens if there is depth to climb out of.
		_zl = _zl_min + rng.randf_range(0.4, 2.0)
	else:
		_dir = -1.0
		# Start wide and fall in. A push begins at the top of its range or there is
		# nowhere to go.
		_zl = _zl_max - rng.randf_range(0.0, 1.2)
	# THE RATE COMES OUT OF THE DISTANCE, not out of a hat - see JOURNEY_MIN. `_zl` is already
	# placed, so this is the depth THIS instance actually has in front of it rather than the
	# family's nominal range, which matters most for a pull: it starts a hair off the floor and
	# has one to two e-folds less to climb than a push has to fall.
	var journey := rng.randf_range(JOURNEY_MIN, JOURNEY_MAX)
	_speed = absf(_zl_bound() - _zl) / journey
	_rot_rate = rng.randf_range(-0.055, 0.055)
	_rot = rng.randf() * TAU

	_build_ramp(rng)
	_push_static()

	return {
		"family": FAMILIES[_family],
		"power": _power,
		"perturbed": _perturb,
		"colour": COLOURS[_colour],
		"direction": dir_name,
		"speed": _speed,
		"journey": "%.0fs" % journey,
		"mood": _sch.name,
		"interior": _interior,
		"iter_base": _iter_base,
		"cycle": _cycle,
		"anchor": "%.12f%+.12fi" % [_cx, _cy],
		"depth": String.num_scientific(exp(_zl_min)),
	}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	# The camera barely breathes: the ZOOM is the motion here, and a drifting frame on top of
	# it reads as a wobble rather than as depth.
	drift_view(f, 0.012, 0.012)
	update_layers(f, delta)
	if not _ch_seeded:
		_ch_seeded = true
		_ch = chroma_hue()
	# Loudness, smoothed. It only ever scales RATES (see the class doc), so a fast attack here
	# would read as the zoom stuttering rather than as the music pushing it.
	var e := clampf(f.energy, 0.0, 1.0)
	_energy = lerpf(_energy, e, 1.0 - exp(-1.6 * delta))
	_glow = maxf(_glow * exp(-3.2 * delta), clampf(f.beat, 0.0, 1.0) * 0.22)

	_step_zoom(delta)
	_rot += _rot_rate * delta * (0.7 + 0.6 * _energy)
	_phase += _phase_rate * delta * (0.6 + 0.9 * _energy)
	_push_live()
	queue_redraw()


## Where this instance is headed, and the only bound it will ever meet: the floor for a push,
## the whole set for a pull.
func _zl_bound() -> float:
	return _zl_max if _dir > 0.0 else _zl_min


## THE ZOOM, on a logarithmic scale because that is what a zoom is: a constant e-folds per
## second, so the picture opens at the same visual rate whatever depth it is at. Anything
## linear in the scale itself crawls when deep and tears when wide.
##
## ONE DIRECTION, chosen once. `_dir` is not written anywhere below and nothing else writes it
## either - see the class doc for the report that made that a rule rather than a default.
func _step_zoom(dt: float) -> void:
	# ARRIVE, never bounce and never grind. Past the floor the picture pixelates and past the
	# ceiling there is nothing left to see, so the rate is taken off over the last ARRIVE_SPAN
	# e-folds instead. Smoothstep, so the slowing has no corner in it at either end: the camera
	# eases to rest against the bound and asymptotes rather than touching it.
	#
	# The frame does not die there - `_rot` and the palette phase keep running - and the
	# journey is sized (see JOURNEY_MIN) so that only a long-held scene gets this far at all.
	var left := clampf(absf(_zl_bound() - _zl) / ARRIVE_SPAN, 0.0, 1.0)
	var arrive := left * left * (3.0 - 2.0 * left)
	_zl += _dir * _speed * (0.55 + 0.9 * _energy) * arrive * dt
	# Belt and braces: the ease above cannot overshoot, but a caller stepping a huge dt could.
	_zl = clampf(_zl, _zl_min, _zl_max)


## Place the Julia parameter at angle [param t] on the main cardioid, pushed off it by
## `_jpush`. The cardioid is the boundary between the parameters whose sets are connected
## blobs and those whose sets are dust, and everything worth looking at is within a hair of it.
func _julia_at(t: float) -> void:
	var c := Vector2(cos(t), sin(t)) * 0.5 - Vector2(cos(2.0 * t), sin(2.0 * t)) * 0.25
	var n := c.normalized() if c.length() > 1e-6 else Vector2.RIGHT
	_jx = c.x + n.x * _jpush
	_jy = c.y + n.y * _jpush


# ---------------------------------------------------------------------------------
# The anchor, and the reference orbit that makes a deep one drawable
# ---------------------------------------------------------------------------------

## Find a point worth falling into, by local search toward the boundary.
##
## The escape time is a fine landscape to hill-climb on: it is small far from the set, infinite
## inside it, and enormous just outside - and "just outside" is exactly where the structure is.
## So this starts wide, samples a ring of candidates, keeps the one that took LONGEST to escape
## (a point that never escapes is rejected: the inside of the set is a flat hole, and its orbit
## is also useless as a reference for anything at the surface), then halves the radius and goes
## again. Twenty-odd halvings walk the anchor from the whole plane down to the depth the zoom
## will reach, and every level is guaranteed still to be on the boundary because that is the
## only place a long escape time can be.
##
## The iteration cap grows with depth for the same reason the scene's does: points nearer the
## boundary take longer to leave, and a cap that stayed put would score every candidate the
## same and the search would turn into a random walk.
func _find_anchor(rng: RandomNumberGenerator, floor_scale: float) -> void:
	var cx := -0.6
	var cy := 0.0
	if _family == 1:
		cx = 0.0
		cy = 0.0
	var radius := 1.35
	var best := -1
	# Seed the walk from the best of a wide scatter, so the search does not start inside a
	# lobe it can never climb out of.
	for _i in 40:
		var px := cx + rng.randf_range(-radius, radius)
		var py := cy + rng.randf_range(-radius, radius)
		var n := _escape(px, py, 220)
		if n > best and n < 220:
			best = n
			_cx = px
			_cy = py
	if best < 0:
		_cx = cx
		_cy = cy
	# ... then descend, halving the search radius each level until it is under the deepest
	# scale the zoom can reach. Six candidates a level is enough: the landscape is smooth at
	# the scale being searched, so this is refinement rather than exploration.
	var cap := 260
	while radius > floor_scale * 0.5:
		radius *= 0.5
		cap = mini(ITER_MAX, cap + 60)
		var bx := _cx
		var by := _cy
		var bn := _escape(_cx, _cy, cap)
		if bn >= cap:
			bn = -1              # the current point is interior at this cap: any escaper beats it
		for _i in 6:
			var a := rng.randf() * TAU
			var r := radius * sqrt(rng.randf())
			var px := _cx + cos(a) * r
			var py := _cy + sin(a) * r
			var n := _escape(px, py, cap)
			if n < cap and n > bn:
				bn = n
				bx = px
				by = py
		_cx = bx
		_cy = by
	# NO COMPOSITION PASS, and that was tried and removed rather than never considered. The
	# obvious next step is to judge the FRAME each candidate would give - how much of it is
	# interior, how wide a spread of escape times it holds - and keep the best. It cannot work
	# here: composing a frame means moving the centre by a fraction of THAT frame's width, and
	# the frame being composed is a mid-depth one whose width is thousands of times the final
	# anchor's refinement. Every such nudge threw away the entire descent. Measured: three of
	# four seeds went from a full frame of structure to a flat wash, including the deep
	# perturbed one. Composition would have to be scored DURING the descent, at every level,
	# which is a coarse grid per candidate per level and seconds of build time.
	#
	# What is here instead is the descent's own guarantee: the anchor is on the boundary at
	# every scale it passed through, so the boundary crosses the frame at every depth the zoom
	# visits. A frame can still come out half flat colour; a fractal frame legitimately can.
	if _perturb:
		_upload_reference()


## Iterate one point in FLOAT64 and return where it escaped, or [param cap] if it never did.
## This mirrors the shader's families exactly - if the two ever disagree the search would be
## hunting on a different landscape from the one being drawn.
func _escape(px: float, py: float, cap: int) -> int:
	var zx := 0.0
	var zy := 0.0
	var cx := px
	var cy := py
	if _family == 1:
		zx = px
		zy = py
		cx = _jx
		cy = _jy
	for i in cap:
		if _family == 2 or _family == 5:
			zx = absf(zx)
			zy = absf(zy)
		elif _family == 3:
			zy = -zy
		# z^p by repeated multiplication, the same way the shader does it.
		var rx := zx
		var ry := zy
		for _k in _power - 1:
			var nx := rx * zx - ry * zy
			ry = rx * zy + ry * zx
			rx = nx
		if _family == 4 or _family == 5:
			rx = absf(rx)
		zx = rx + cx
		zy = ry + cy
		if zx * zx + zy * zy > 4.0:
			return i
	return cap


## Iterate the anchor's own orbit in float64 and hand it to the shader as a texture, one texel
## per iteration. This is the whole of what perturbation needs from this side: the shader never
## sees the anchor's coordinates at all, only where its orbit went.
##
## Stored as float32, which is correct and not a compromise: an error in the reference perturbs
## the per-pixel delta by a RELATIVE amount (the delta is multiplied by Z, never compared to
## it), so seven digits on Z costs seven digits on a number that only has to be accurate to a
## pixel. What must be float64 is the ITERATION, and it is - GDScript floats are 64-bit.
func _upload_reference() -> void:
	var zx := 0.0
	var zy := 0.0
	var data := PackedFloat32Array()
	data.append(0.0)                 # Z(0) = 0, which is what the shader's rebase assumes
	data.append(0.0)
	var n := 1
	for _i in REF_MAX - 1:
		var nx := zx * zx - zy * zy + _cx
		zy = 2.0 * zx * zy + _cy
		zx = nx
		data.append(zx)
		data.append(zy)
		n += 1
		if zx * zx + zy * zy > 4.0:
			break
	var img := Image.create_from_data(n, 1, false, Image.FORMAT_RGF, data.to_byte_array())
	_mat.set_shader_parameter("u_orbit", ImageTexture.create_from_image(img))
	_mat.set_shader_parameter("u_ref_len", n)


# ---------------------------------------------------------------------------------
# The palette
# ---------------------------------------------------------------------------------

## Build the colour ramp as a CLOSED LOOP of control colours, which is the only way it can be
## right: the shader samples it with fract(), so a ramp whose two ends differ would put a hard
## seam through the picture at every palette repeat - and on a smooth iteration count those
## repeats are the contour lines of the whole image.
##
## The hues walk out from the scheme's base toward its accent and back over the cycle, so a
## palette is always two-coloured rather than a rainbow; the values do the same, so there is a
## dark band and a bright band in every one and the bands read as bands.
func _build_ramp(rng: RandomNumberGenerator) -> void:
	var n := rng.randi_range(3, 6)
	# The sweep stays INSIDE the scheme's two hues, give or take a third. It was allowed to run
	# to 2.2x and that is a third of the colour wheel past the accent: the ramp crossed three
	# more hues on its way, and once the mip filtering averaged the fine detail the whole frame
	# came out olive whatever mood had been picked. A palette here is two colours and the walk
	# between them, which is what [Scheme] is for.
	var arc := fposmod(_sch.accent - _sch.hue + 0.5, 1.0) - 0.5
	arc *= rng.randf_range(0.7, 1.35)
	var v_lo := rng.randf_range(0.02, 0.16)
	var v_hi := rng.randf_range(0.72, 1.0)
	var s_lo := clampf(_sch.sat * rng.randf_range(0.35, 0.8), 0.0, 1.0)
	var s_hi := clampf(_sch.sat * rng.randf_range(1.0, 1.6), 0.15, 1.0)
	var v_ph := rng.randf() * TAU
	var stops: Array = []
	for i in n:
		var u := float(i) / float(n)
		var w := 0.5 - 0.5 * cos(TAU * u)              # 0 at the ends of the loop, 1 in the middle
		var vw := 0.5 - 0.5 * cos(TAU * u + v_ph)
		stops.append(Color.from_hsv(fposmod(_sch.hue + arc * w, 1.0),
			lerpf(s_hi, s_lo, vw), lerpf(v_lo, v_hi, vw)))
	var img := Image.create(RAMP, 1, false, Image.FORMAT_RGBA8)
	for x in RAMP:
		var t := float(x) / float(RAMP) * float(n)
		var i0 := int(floor(t)) % n
		var i1 := (i0 + 1) % n
		var fr: float = t - floor(t)
		img.set_pixel(x, 0, (stops[i0] as Color).lerp(stops[i1] as Color,
			fr * fr * (3.0 - 2.0 * fr)))
	# MIPMAPPED, and the picture depends on it. Near the set the palette repeats many times
	# across a single pixel, and point-sampling that is confetti; the shader picks a mip level
	# from the screen-space rate of change instead, which averages exactly the colours the pixel
	# spans. See the ANTIALIASING note in the shader.
	img.generate_mipmaps()
	_mat.set_shader_parameter("u_ramp", ImageTexture.create_from_image(img))


# ---------------------------------------------------------------------------------
# The uniforms
# ---------------------------------------------------------------------------------

# Everything that cannot change after the roll. Split from the per-frame push so a frame is a
# handful of parameter writes rather than thirty.
func _push_static() -> void:
	_mat.set_shader_parameter("u_family", _family)
	_mat.set_shader_parameter("u_power", _power)
	_mat.set_shader_parameter("u_perturb", _perturb)
	_mat.set_shader_parameter("u_colour", _colour)
	_mat.set_shader_parameter("u_trap_p", _trap_p)
	_mat.set_shader_parameter("u_trap_r", _trap_r)
	_mat.set_shader_parameter("u_stripe", _stripe)
	_mat.set_shader_parameter("u_cycle", _cycle)
	_mat.set_shader_parameter("u_interior", _interior)
	_mat.set_shader_parameter("u_in_col", Vector3(_in_col.r, _in_col.g, _in_col.b))
	_mat.set_shader_parameter("u_vig", _vig)
	_mat.set_shader_parameter("u_bail", _bail)
	# The anchor reaches the shader ONLY on the direct path, where the depth is shallow enough
	# for 32 bits to hold it. The perturbed path never sees a coordinate at all.
	_mat.set_shader_parameter("u_center", Vector2(_cx, _cy))


func _push_live() -> void:
	var scale := exp(_zl)
	_mat.set_shader_parameter("u_scale", scale)
	_mat.set_shader_parameter("u_rot", _rot)
	_mat.set_shader_parameter("u_phase", _phase)
	_mat.set_shader_parameter("u_julia", Vector2(_jx, _jy))
	_mat.set_shader_parameter("u_glow", _glow)
	_mat.set_shader_parameter("u_gain", _gain * (0.94 + 0.12 * _energy))
	# THE ITERATION BUDGET RIDES THE DEPTH. Points near the boundary take longer to escape the
	# deeper the view goes, so a fixed budget draws a deep frame as one flat interior-coloured
	# sheet - the detail is all in orbits that had not finished yet. It scales with the number
	# of e-folds travelled, which is the honest measure of how much magnification is in force.
	var efolds := maxf(0.0, log(SCALE_WIDE) - _zl)
	_mat.set_shader_parameter("u_iter",
		clampi(_iter_base + int(_iter_gain * efolds), 64, ITER_MAX))


func _draw() -> void:
	begin_draw()
	# A flat dark ground under the quad. The field is opaque and oversized so this is never
	# seen in a settled frame - it is there for the fade, where the quad's alpha drops and
	# whatever is behind the scene would otherwise show through.
	var x := size.x * 0.5 * 1.25 / maxf(0.001, view.zoom_actual())
	var y := size.y * 0.5 * 1.25 / maxf(0.001, view.zoom_actual())
	draw_colored_polygon(PackedVector2Array([
		Vector2(-x, -y), Vector2(x, -y), Vector2(x, y), Vector2(-x, y)]),
		Color(_in_col.r * 0.5, _in_col.g * 0.5, _in_col.b * 0.5))
	if _quad == null:
		_quad = Layer.FieldQuad.new()
		_quad.material = _mat
		# Deferred: this runs inside _draw, and the tree may not be changed mid-pass.
		call_deferred("add_child", _quad)
		return
	# Oversized for the same reason every layer is: the 2D view drifts and zooms on top of the
	# field, and a quad sized to the frame would slide its own edge into shot.
	var half := Vector2(size.x, size.y) * 0.5 * 1.15
	_quad.px_half = half
	_mat.set_shader_parameter("u_half", half / maxf(1.0, unit()))
	_quad.queue_redraw()
