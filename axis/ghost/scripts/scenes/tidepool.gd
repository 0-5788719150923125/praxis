extends GhostScene

## Tidepool - sunlight through a hand's depth of moving water.
##
## Looking STRAIGHT DOWN into a clear pool over rounded stones, sand and weed, in full
## daylight. Two firsts sit in that sentence. It is the first overhead shot in the
## catalogue - everything else is seen from the side, from below, or from a camera flying
## through a world - and it is one of the first bright grounds, where the pale bed IS the
## subject rather than a luminous thing floating in a dark frame. The surface is a real
## wave field: a swell arriving from one edge, finer chop crossing it, and ring waves that
## spread and die where drips land. What you actually watch is the CAUSTIC NET on the bed,
## a mesh of bright ribs that slides and pinches as the surface curves, with the stones
## swimming slightly out from under the crests because the image of the bed is refracted
## too. Now and again the sun catches a steep enough face that glare closes over a patch of
## bed and then opens again.
##
## THE CAUSTICS ARE COMPUTED, NOT DRAWN. This is the distinction worth the whole scene.
## `Layer.Surface` already has "caustics" and they are hand-placed wavy bright bands; the
## terrain water is a flat tinted plane with an opacity ramp. Here the light is actually
## traced: at every node of a grid the local surface normal comes from [WaveField]'s
## closed-form gradient, the sun vector is refracted through it by Snell's law into the
## water, the refracted ray is walked down to the bed plane, and its energy is splatted
## bilinearly into an accumulation buffer. Bright ribs are where many rays landed in the
## same place - the definition of a caustic - and they pinch, fork and close exactly where
## the surface curvature says they should, because nothing else decided where they go. A
## short separable blur afterwards is the only cosmetic step, and it stands in for the
## sun's half-degree of angular width, which is why a real caustic net has soft edges
## instead of infinitely thin ones.
##
## The bed's own IMAGE is refracted separately, through the same normal but along the
## viewing ray, which points straight down because the camera does. That second refraction
## is small (the view is near-normal, so the ray bends by only a few degrees) and it is
## what makes the stones appear to drift out from under a passing crest rather than
## sitting still under a moving light show. Doing only the caustics and not this is the
## usual shortcut and it reads instantly as a texture with a projector aimed at it.
##
## WHAT THE SEED DECIDES. The pool: its span in world units, its base depth (0.05 to 0.35,
## a hand's depth), whether one corner rises into a dry shelf and along which diagonal, the
## turbidity, and the index of refraction (1.30 to 1.36 - real seawater is 1.34, and the
## range is wide enough that a caustic net visibly tightens across it). The sun: elevation
## from 35 to 88 degrees and a free azimuth, which together decide how far the ribs are
## thrown sideways from the crest that made them. The surface: 6 to 14 Gerstner components
## with their directions clustered around one swell heading, their wavelengths, their
## steepnesses, and one tempo scaling the whole dispersion relation. The bed: a composition
## over cobble, sand, weed and shell, 20 to 90 stones from an ocean or earth [Palette],
## the sand's grain, the weed's patchiness. And the light: the ambient floor, the caustic
## contrast, the specular exponent, the glare threshold and the ground value.
##
## AUDIO. Every component owns a FIXED band, assigned by wavelength rank - the longest
## wave reads the lowest band, the shortest the highest - and its amplitude is a
## slow-attack envelope on that band. A DRIP RING reads the water it is spreading across
## rather than ignoring it: the swell carries its front forward over a crest and holds it
## back in a trough, and chop damps it out, so the two systems read as one body of water
## instead of a circle stamped over a surface (see [method WaveField.sweep_rings]). So a bass-heavy passage grows long swell while a
## bright passage grows fine chop, and the two move against each other instead of throbbing
## together. Nothing anywhere scales with loudness: the wavelengths are fixed at build, and
## a louder song makes a rougher surface, not a bigger one. `f.movement` raises the global
## Gerstner steepness, which is physically the thing that pinches caustic ribs into bright
## knots, so a section change tightens the whole net without moving anything. Rising edges
## of `f.beat` drop drips at pre-rolled Halton positions - a table walked by an index, never
## an rng draw, because the live analyzer and the offline export bake do not produce
## identical feature streams and a draw on an audio-conditioned event would desynchronise
## an export from its preview for the rest of the song. `chroma_hue()` sets the water's
## tint and the sky reflection's hue. `f.energy` moves only the glare threshold.
##
## MEASURED CONSTANTS. `f.energy` is the mean over 64 bands and rarely passes 0.5, and a
## single band read through `f.sample()` spends most of its time under 0.3, so a component's
## drive is written `0.22 + 1.7 * level` rather than as something assuming the band reaches
## 1. The amplitude-to-wavelength ratio is sampled in 0.03 to 0.075 because that is where
## caustics actually appear: the focusing strength of a wave on a bed goes as
## depth * amplitude * wavenumber^2 * (1 - 1/ior), and below roughly a fortieth of a
## wavelength that product stays under one and the bed just gets a gentle mottle instead
## of a net. `jz` is clamped off zero at 0.15 because a Gerstner surface whose Jacobian
## goes negative has folded over itself - a breaking wave, on which a normal is undefined.
##
## COST, HONESTLY, AND MEASURED. Everything - the sweep, the light cast, the blur, the
## shading and the batching - runs inside a [FrameForge] job; the scene itself only ships a
## snapshot and submits a finished packet. A build costs 29 to 36 ms at 1920x1080 on an idle
## machine and half again that under load, which is a frame or two at 60 fps, so the water is
## rebuilt at 20-30 Hz and the forge hands the drawn frame whatever is ready. That number is
## flat in the component count now, which is what the budget is for, and it is below the 51 ms
## the old model's worst case cost. It is affordable only because the surface is SLOW: the
## fastest component's period is 1.5-2.7 s, so a 25 Hz rebuild is nowhere near its Nyquist
## limit. It was not always - the field used to run three times faster and strobed against
## its own build rate.
##
## The grid is sized against a budget in [constant SWEEP_BUDGET], and that budget counts the
## O(nodes) work as well as the sweep, because measurement said the sweep is not the biggest
## term (see the constant). The mesh is one vertex per NODE plus one per cell centre, with
## shared indices, so a frame rewrites only the colours - positions and the index buffer are
## built once. Cells are CROSS-SPLIT into four triangles about that centre rather than two
## about a diagonal, without which a 25-pixel cell reads as a herringbone of creases rather
## than as water. Arrays handed to the main thread are never written again; a change
## allocates a new one, which is the only discipline that makes a pipelined worker safe.
##
## The wave itself is analytic and needs no integration, but the drips and the amplitude
## envelopes are STATE, so they advance on a [SimClock]. That is not optional here: the
## Director sub-steps update() up to fifteen times in one frame, pre-warms a new scene
## twelve times before it is ever drawn, and an Echo re-localize can fast-forward hundreds
## of calls - any of which would have every ring expired before the first frame anyone sees.

## How far the node grid runs past the frame. Refraction throws light sideways by up to
## about 1.1 times the depth, so a pool with a hard edge exactly at the frame would show a
## dim band there; the overdraw plus the wrapped accumulation buffer between them hide it.
const OVER := 1.18

## Node-work per frame the build is allowed, in units of one Gerstner component evaluated at
## one node. The grid is derived from this and the component count, so the two cannot multiply
## into an unaffordable frame.
##
## THE BUDGET USED TO BE WRONG, and measurably so: it counted only `nodes * components`, as if
## the sweep were the whole frame. It is not. Casting the light, blurring the accumulator and
## shading each node are all O(nodes) with large constants, so a SIX-component surface - which
## the old budget rewarded with the finest grid - came out the MOST expensive of all. Measured
## per build at 1920x1080 (tests/tidepool_cost_probe.gd): 7 components over 6588 nodes took
## 47-51 ms, while 13 components over 3555 nodes took 34 ms. The budget was equalising the one
## term that was not dominant.
##
## PER_NODE is the rest of the frame expressed in the same unit, measured from those pairs: the
## fixed work is worth about fourteen components. Budgeting `nodes * (components + PER_NODE)`
## makes the cost flat in the component count, which is what the budget was always for.
const SWEEP_BUDGET := 100000
const PER_NODE := 14.0

## Node-count bounds the budget is clamped into, so neither a very simple nor a very busy
## surface leaves the range where a caustic rib is a few cells wide.
const NODES_MIN := 3400
const NODES_MAX := 6800

## Schlick's F0 for an air/water interface. (1-1.33)/(1+1.33) squared is about 0.02: only
## two percent of light reflects off flat water, which is why a pool seen from directly
## above is transparent and only the steep faces flash.
const F0 := 0.02

## Gerstner surfaces fold over past this Jacobian; clamping there caps how far a crest may
## tilt its own normal instead of letting it invert.
const JZ_FLOOR := 0.15

## At most this many drip rings at once. A fourth ring is already past what a hand-sized
## pool would carry, and each one costs work proportional to its circumference.
const MAX_RINGS := 4

## Pre-rolled drip slots. Walked by an index on a beat, never re-drawn (see the doc).
const DRIP_SLOTS := 24

var _sim: SimClock
var _forge := FrameForge.new()
var _job: PoolJob
var _wf: WaveField

var _gw := 96
var _gh := 54
var _step := 0.01
var _org := Vector2.ZERO
var _span := 1.0

var _bands := PackedFloat32Array()      # per component: its fixed spectral position
var _drive := PackedFloat32Array()      # per component: gain on that band
var _floor := PackedFloat32Array()      # per component: its amplitude at silence
var _base := PackedFloat32Array()       # per component: amplitude at full drive
var _amp := PackedFloat32Array()        # the live, enveloped amplitudes
var _atk := 0.6
var _rel := 0.35

var _steep := 1.0
var _move_gain := 1.4

var _drips: Array = []                  # the pre-rolled slot table
var _drip_i := 0
var _rings: Array = []
var _beat_prev := 0.0
var _beat_arm := 0.0
var _drip_gap := 0.5
## How far a drip ring's front is carried by the swell it is crossing, as a multiple of the
## swell's live amplitude, and how hard the chop damps it. See WaveField.sweep_rings.
var _ring_warp := 1.6
var _ring_damp := 0.6

var _t_sim := 0.0
var _kicked := false
var _ch := Vector2.ZERO           # the SMOOTHED tonal centre, as (hue, strength)
var _ch_vec := Vector2.ZERO       # ... carried in CARTESIAN form, which is what makes it smooth
var _glare_e := 0.0               # enveloped energy behind the glare threshold

var _hue := 0.55
var _sky_hue := 0.58
var _ground_val := 0.72
var _ground_sat := 0.07
var _water_sat := 0.42
var _water_val := 0.86
var _sky_sat := 0.10
var _sun_col := Color(1.0, 0.98, 0.94)
var _amb := 0.4
var _contrast := 1.6
var _sharp := 0.35
var _exposure := 1.0
var _glare_hi := 0.72
var _glare_lo := 0.42
var _glare_gain := 0.9


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"

	# Daylight over water: the cold, pale end of the mood table plus one warm dawn option,
	# because a tidepool at the golden hour is the same scene with a different sun.
	var sch := Scheme.among(["glacier", "teal", "bone", "dawn"], rng)
	_hue = sch.hue
	_sky_hue = sch.accent
	_ground_val = rng.randf_range(0.55, 0.85)
	_ground_sat = rng.randf_range(0.03, 0.11)
	_water_sat = rng.randf_range(0.26, 0.58)
	_water_val = rng.randf_range(0.72, 0.94)
	_sky_sat = rng.randf_range(0.05, 0.18)

	# --- the surface ------------------------------------------------------------------
	var comps := rng.randi_range(6, 14)
	var nodes := clampi(int(float(SWEEP_BUDGET) / (float(comps) + PER_NODE)), NODES_MIN, NODES_MAX)
	# Nominal 16:9, because `size` is not known during build - the grid is mapped onto
	# whatever frame it lands in with a uniform scale, so the waves never stretch.
	_gw = maxi(24, int(round(sqrt(float(nodes) * 16.0 / 9.0))))
	_gh = maxi(16, int(round(float(nodes) / float(_gw))))
	_span = rng.randf_range(0.7, 1.6)
	_step = _span / float(_gw - 1)
	_org = Vector2(-float(_gw - 1) * _step * 0.5, -float(_gh - 1) * _step * 0.5)

	# THE WHOLE FIELD'S SPEED, and it was about three times too high. `tempo` scales the
	# dispersion-derived frequency of every component at once (see WaveField.add), and measured
	# at 0.24-0.52 the SHORTEST component came out with a period of 0.44 to 0.86 seconds - a one
	# to two hertz wobble running for the whole hold, which is what "a constant wave effect,
	# which is too fast" was. It also fought the build rate: a frame costs tens of milliseconds
	# (tests/tidepool_cost_probe.gd), so the surface is rebuilt at 20-30 Hz, and a two hertz
	# ripple sampled at 25 Hz strobes.
	#
	# 0.08-0.17 puts the fastest chop at 1.3-2.6 s and the longest swell at 3.5-7 s, which is
	# both watchable and roughly what water of this apparent size does. The dispersion RELATION
	# is untouched - long waves still outrun short ones by sqrt(g k) - because that relation is
	# what makes a swell and a chop read as one body of water.
	var tempo := rng.randf_range(0.08, 0.17)
	# A few percent of detune per component, so the field does not return to the same
	# configuration on a fixed cycle. With one shared tempo every frequency is an exact function
	# of its wavenumber, and the surface has an audible period; a small spread breaks that
	# without touching how the components rank against each other.
	var detune := rng.randf_range(0.02, 0.09)
	var swell := rng.randf() * TAU               # the heading the whole sea arrives on
	var fan := rng.randf_range(0.35, 1.25)       # how far components fan off that heading
	var ratio := rng.randf_range(0.030, 0.075)   # amplitude as a fraction of wavelength
	var q_lo := rng.randf_range(0.25, 0.45)
	var q_hi := rng.randf_range(0.55, 0.95)
	_move_gain = rng.randf_range(0.8, 2.2)
	_atk = rng.randf_range(0.35, 0.8)            # slow ATTACK: a swell takes seconds to build
	_rel = rng.randf_range(0.18, 0.45)

	_wf = WaveField.new()
	var lam: Array = []
	for i in comps:
		# Wavelengths spread geometrically from a long swell down to a fine chop, as a
		# fraction of the pool: one number, sampled per component, spanning a factor of ten.
		var u := (float(i) + rng.randf_range(0.05, 0.95)) / float(comps)
		lam.append(_span * (0.62 * pow(0.11, u)))
	lam.sort()
	lam.reverse()                                # longest first, so index IS wavelength rank
	for i in comps:
		var wl: float = lam[i]
		var ang := swell + rng.randf_range(-fan, fan) * 0.5
		var q := rng.randf_range(q_lo, q_hi)
		_wf.add(ang, wl, 0.0, q, rng.randf() * TAU, tempo * (1.0 + rng.randf_range(-detune, detune)))
		_base.append(wl * ratio * rng.randf_range(0.7, 1.3))
		# The band assignment IS the audio design: rank 0 (longest) reads the bottom of the
		# spectrum, the last rank the top. Spread evenly and jittered so two components can
		# never sit on the same band and pump as one.
		var bt := (float(i) + 0.5) / float(comps)
		_bands.append(clampf(bt + rng.randf_range(-0.05, 0.05), 0.02, 0.97))
		_drive.append(rng.randf_range(1.2, 2.2))
		_floor.append(rng.randf_range(0.10, 0.30))
		_amp.append(0.0)

	# --- drips ---------------------------------------------------------------------------
	# Halton (2, 3) rather than an rng: a low-discrepancy pair spreads successive drips
	# across the pool instead of clustering them the way uniform draws do, and the sequence
	# is an index, so a beat never has to touch a random source.
	var half_w := float(_gw - 1) * _step * 0.5
	var half_h := float(_gh - 1) * _step * 0.5
	var ring_speed_lo := rng.randf_range(0.05, 0.11)
	var ring_speed_hi := rng.randf_range(0.14, 0.26)
	for i in DRIP_SLOTS:
		var hx := _halton(i + 3, 2)
		var hy := _halton(i + 3, 3)
		var wl := _span * rng.randf_range(0.035, 0.085)
		_drips.append({
			"x": lerpf(-half_w * 0.92, half_w * 0.92, hx),
			"y": lerpf(-half_h * 0.92, half_h * 0.92, hy),
			"k": TAU / wl,
			"w": _span * rng.randf_range(0.055, 0.13),
			"speed": rng.randf_range(ring_speed_lo, ring_speed_hi),
			"a0": wl * rng.randf_range(0.05, 0.11),
			"q": rng.randf_range(0.3, 0.7),
			"phi_rate": rng.randf_range(1.3, 2.6),   # carrier outruns the front (see WaveField)
			"ph": rng.randf() * TAU,
			"life": rng.randf_range(2.2, 5.5),
		})
	_drip_gap = rng.randf_range(0.25, 0.9)
	# How strongly a ring is carried and eaten by the water it crosses. Sampled rather than
	# baked: some pools take a drip cleanly and some tear it apart, and that is a property of
	# the pool, not a constant of the renderer.
	_ring_warp = rng.randf_range(0.9, 2.6)
	_ring_damp = rng.randf_range(0.25, 1.1)

	# --- the pool floor ------------------------------------------------------------------
	var depth := rng.randf_range(0.05, 0.35) * _span
	var turbid := rng.randf_range(0.15, 0.85)
	var ior := rng.randf_range(1.30, 1.36)
	var shelf := rng.randf() < 0.45
	var shelf_ang := rng.randf() * TAU
	var shelf_len := _span * rng.randf_range(0.45, 1.05)
	var shelf_wob := _span * rng.randf_range(0.02, 0.09)

	var sun_el := deg_to_rad(rng.randf_range(35.0, 88.0))
	var sun_az := rng.randf() * TAU
	var shine_sq := rng.randi_range(4, 6)          # specular exponent = 2^shine_sq
	_glare_hi = rng.randf_range(0.62, 0.86)
	_glare_lo = rng.randf_range(0.30, 0.52)
	_glare_gain = rng.randf_range(0.55, 1.25)
	_amb = clampf(rng.randf_range(0.24, 0.48) + turbid * 0.18, 0.0, 0.8)
	_contrast = rng.randf_range(1.15, 2.3) * (1.0 - 0.35 * turbid)
	# SHARPENING IS THE WRONG KNOB HERE, and most of the way down. It squares the caustic's
	# contrast about its own mean, which narrows every rib - and a rib is already only one or two
	# NODES wide, i.e. 21-29 screen pixels at 1920x1080 (measured). Narrowing a feature that is
	# already at the reconstruction's Nyquist limit does not sharpen it, it aliases it, which is
	# the "super jagged... blocky, and angular" the light was reported to have. The net is now
	# resolved rather than pinched, and what it loses in bite it gains in being a smooth surface
	# of light instead of a lattice of wedges.
	_sharp = rng.randf_range(0.0, 0.18)
	_exposure = rng.randf_range(0.94, 1.18)
	_sun_col = Color(1.0, rng.randf_range(0.94, 1.0), rng.randf_range(0.86, 0.99))

	# 20-30 sim ticks a second. The wave is analytic, so this clock exists for the ring
	# lifecycles and the amplitude envelopes; those need consistency, not resolution.
	_sim = SimClock.new(rng.randf_range(22.0, 32.0), 2)

	# THE BED, at more than twice the resolution. 96-128 texels spanned the whole pool, so at
	# 1920x1080 one texel covered 15 to 20 screen pixels - and the bed is not background, it is
	# what the scene is looking AT through the water, and the dry shelf shows it with no water
	# over it at all. That is the "sand rendered to the side or below the water has low fidelity"
	# report. 200-272 puts a texel at 7 to 10 px, which is at the limit of what the refracted
	# bilinear lookup can carry. The cost is build-time only - build_bed runs once per scene, not
	# per frame - and is O(res^2), so this is about 4x of a number that was not in the frame.
	var bed_res := rng.randi_range(200, 272)
	var stones := rng.randi_range(20, 90)
	var ramp_name := "ocean" if rng.randf() < 0.55 else "earth"
	var mix := {
		"cobble": rng.randf_range(0.2, 1.0),
		"sand": rng.randf_range(0.4, 1.0),
		"weed": rng.randf_range(0.0, 0.9),
		"shell": rng.randf_range(0.0, 0.6),
	}

	_job = PoolJob.new()
	_job.wf = _wf
	_job.gw = _gw
	_job.gh = _gh
	_job.org = _org
	_job.step = _step
	_job.eta = 1.0 / ior
	# Two to four [1 2 1] passes, up from one to two. The accumulation buffer is at NODE
	# resolution, so a caustic rib is one or two cells wide - 18 to 28 screen pixels - and a
	# single blur pass leaves its shoulder inside one cell, which is a hard-edged wedge however
	# smoothly the surface underneath is moving. Widening the kernel spreads the shoulder across
	# two or three cells, which is both what the report asked for and what the sun's half-degree
	# of angular width actually does. Each pass is separable and O(nodes), the cheapest term in
	# the frame by some way.
	# Passes of the five-tap binomial in _blur, each worth two of the old three-tap ones. Two to
	# three therefore smooths as four to six used to, at roughly half the cost.
	_job.blur_n = rng.randi_range(2, 3)
	_job.shine_sq = shine_sq
	_job.wet_band = depth * rng.randf_range(0.10, 0.30)
	_job.dry_lift = rng.randf_range(1.02, 1.14)
	_job.sun = Vector3(cos(sun_el) * cos(sun_az), cos(sun_el) * sin(sun_az), sin(sun_el))
	# Blinn half vector against a camera looking straight down, so the glint is a real
	# specular lobe rather than a brightness added wherever the surface happens to tilt.
	_job.half_v = (_job.sun + Vector3(0.0, 0.0, 1.0)).normalized()
	_job.alloc()
	_job.build_depth(depth, shelf, shelf_ang, shelf_len, shelf_wob, turbid, rng.randi())
	_job.build_bed(rng, bed_res, _span, ramp_name, stones, mix, sun_az, sun_el)

	if rng.randf() < 0.5:
		# Grit and pollen riding the film, seen from above: the shaft is off because there
		# is no sideways light in an overhead shot to make one.
		add_layer("dust", rng, {"count": rng.randi_range(30, 80), "shaft": false,
			"hue": sch.accent, "speed": 0.012, "jitter": 0.04})

	return {
		"mood": sch.name, "components": comps, "grid_w": _gw, "grid_h": _gh,
		"span": _span, "depth": depth, "turbidity": turbid, "ior": ior,
		"tempo": tempo, "swell_deg": rad_to_deg(swell), "fan": fan,
		"amp_ratio": ratio, "steep_range": [q_lo, q_hi], "move_gain": _move_gain,
		"attack": _atk, "release": _rel,
		"sun_elev_deg": rad_to_deg(sun_el), "sun_azim_deg": rad_to_deg(sun_az),
		"shelf": "dry" if shelf else "submerged", "shelf_len": shelf_len,
		"bed_res": bed_res, "stones": stones, "ramp": ramp_name, "mix": mix,
		"ambient": _amb, "contrast": _contrast, "sharp": _sharp,
		"shine": 1 << shine_sq, "glare": [_glare_lo, _glare_hi], "glare_gain": _glare_gain,
		"ground_val": _ground_val, "drip_gap": _drip_gap, "rate": _sim.rate,
		"blur": _job.blur_n,
	}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	# A pool does not pan. The drift is barely there - just enough that the frame is not
	# nailed down - because an overhead shot with a moving camera reads as a drone, not as
	# someone standing over a rock.
	drift_view(f, 0.012, 0.018)
	update_layers(f, delta)

	var steps := _sim.ticks(delta)
	for _i in steps:
		_advance(_sim.dt, f)
	if steps <= 0 and _kicked:
		queue_redraw()
		return
	_kicked = true

	# The water's own colour, dragged a little toward the music's tonal centre - a wash,
	# not a repaint, since a bright scene reacts by shifting temperature rather than by
	# changing what it is.
	# Blended on the wheel rather than along the shortest arc (GhostScene.blend_hue): the arc form
	# flips sign at the antipode, which put a jump of 0.30 of a turn in the water and 0.45 in the
	# sky between two consecutive frames whenever the tonal centre drifted opposite this pool's
	# own hue - the other half of the reported colour flicker, and the half that easing alone
	# cannot reach, since the jump is the same size however slowly the input crosses the boundary.
	var wh := blend_hue(_hue, _ch.x, 0.3 * _ch.y)
	var sh := blend_hue(_sky_hue, _ch.x, 0.45 * _ch.y)
	# Glare threshold is the ONLY thing energy touches: a loud passage lowers the bar the
	# specular lobe has to clear, so highlights bloom shut over the bed and open again. It reads
	# the ENVELOPED energy (see _advance) rather than the raw frame value - the glare it gates
	# lays the sky's whole colour over the pool, and on a jumpy source that strobed.
	var thr := lerpf(_glare_hi, _glare_lo, _glare_e)

	var rings: Array = []
	for entry in _rings:
		var r: Dictionary = entry
		rings.append({"x": r["x"], "y": r["y"], "r": r["r"], "w": r["w"],
			"a": r["a"], "k": r["k"], "phi": r["phi"], "ph": r["ph"], "q": r["q"],
			# `warp` multiplies the LOCAL swell height, so it is already scaled by the water
			# actually running right now - on a glassy passage there is nothing to ride and the
			# ring stays round. `warp_max` is the largest displacement that can produce, which
			# is what the row spans have to be widened by; it needs the live amplitude SUM, and
			# folding that into `warp` itself would make the displacement quadratic in it.
			"warp": _ring_warp, "warp_max": _ring_warp * _swell_amp(), "damp": _ring_damp})

	var cell := maxf(maxf(size.x, 1.0) * OVER / float(_gw - 1),
		maxf(size.y, 1.0) * OVER / float(_gh - 1))
	_forge.kick(_job.run, {
		"t": _t_sim,
		"amps": _amp.duplicate(),
		"steep": _steep,
		"rings": rings,
		"cell": cell,
		"water": Color.from_hsv(wh, _water_sat, _water_val),
		"sky": Color.from_hsv(sh, _sky_sat, 1.0),
		"sun": _sun_col,
		"amb": _amb,
		"contrast": _contrast,
		"sharp": _sharp,
		"exposure": _exposure,
		"glare_thr": thr,
		"glare_gain": _glare_gain,
	}, self, _job)
	queue_redraw()


## One fixed simulation step: the amplitude envelopes, the chop, and the drip lifecycles.
func _advance(dt: float, f: AudioFeatures) -> void:
	_t_sim += dt
	for i in _amp.size():
		var lvl := f.sample(_bands[i])
		var target: float = _base[i] * clampf(_floor[i] + _drive[i] * lvl, 0.0, 1.6)
		_amp[i] = Nonlinear.flare(_amp[i], target, dt, _atk, _rel)

	# Movement is a section change, and a section change roughens water. Steepness pinches
	# the ribs; it moves nothing and resizes nothing.
	_steep = Nonlinear.flare(_steep, 1.0 + _move_gain * f.movement, dt, 1.6, 0.7)

	# THE TONAL CENTRE, EASED IN CARTESIAN FORM. chroma_hue() hands back an ANGLE, and an angle is
	# exactly the wrong quantity to use raw: when two tonal centres a tritone apart trade dominance
	# the resultant passes near the origin and its angle flips half a turn between one frame and
	# the next, so the water's tint and the ground's wash snapped between two hues and then settled
	# on whichever won - "the color is unstable, and will often flicker between two colors at
	# random, then stabilize upon ONE of those colors". Easing the VECTOR routes that flip through
	# the middle instead: an ambiguous tonality reads as low STRENGTH and the tint simply fades
	# out rather than choosing a side, and a genuine key change arrives as a gentle sweep. A 1.2 s
	# constant, matched to the 2.5 s window the harmonic signature already integrates.
	var raw := chroma_hue()
	var want := Vector2(cos(raw.x * TAU), sin(raw.x * TAU)) * raw.y
	_ch_vec = _ch_vec.lerp(want, 1.0 - exp(-0.85 * dt))
	_ch = Vector2(fposmod(_ch_vec.angle() / TAU, 1.0), clampf(_ch_vec.length(), 0.0, 1.0))

	# Same discipline for the glare drive - fast to bloom, slow to fall, like the amplitudes.
	_glare_e = Nonlinear.flare(_glare_e, clampf(f.energy * 2.0, 0.0, 1.0), dt, 3.5, 1.2)

	_beat_arm += dt
	var rising := f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	if rising and _beat_arm >= _drip_gap:
		_beat_arm = 0.0
		_spawn_drip()

	var live: Array = []
	for entry in _rings:
		var r: Dictionary = entry
		var age: float = float(r["age"]) + dt
		var rad: float = float(r["r"]) + float(r["speed"]) * dt
		if age >= float(r["life"]) or rad > _span * 1.2:
			continue
		r["age"] = age
		r["r"] = rad
		r["phi"] = float(r["phi"]) + float(r["phi_rate"]) * dt
		# Two decays, both physical: the packet loses energy with time, and a ring spreads
		# its fixed energy round a growing circumference, so its height falls as 1/sqrt(r).
		var fade := clampf(1.0 - age / maxf(0.01, float(r["life"])), 0.0, 1.0)
		var r0: float = maxf(0.01, float(r["w"]))
		r["a"] = float(r["a0"]) * fade * fade * sqrt(r0 / maxf(rad, r0))
		live.append(r)
	_rings = live


## The swell's live amplitude, as the sum of what the components are actually carrying this
## moment. The ring couplings scale by it so they answer the water in front of them rather than
## a build-time guess at how rough the pool would get.
func _swell_amp() -> float:
	var a := 0.0
	for v in _amp:
		a += v
	return a


func _spawn_drip() -> void:
	var slot: Dictionary = _drips[_drip_i % _drips.size()]
	_drip_i += 1
	if _rings.size() >= MAX_RINGS:
		_rings.remove_at(0)                  # the oldest yields, deterministically
	_rings.append({
		"x": slot["x"], "y": slot["y"], "r": 0.0, "w": slot["w"],
		"a": 0.0, "a0": slot["a0"], "k": slot["k"], "q": slot["q"],
		"phi": 0.0, "phi_rate": float(slot["phi_rate"]) * float(slot["speed"]) * float(slot["k"]),
		"ph": slot["ph"], "speed": slot["speed"], "age": 0.0, "life": slot["life"],
	})


func _draw() -> void:
	begin_draw()
	# Flat, bright and full bleed, first thing. The bed mesh covers the frame, so this is
	# the safety net under a drifting camera - but it has to be the HIGH-KEY ground, since
	# Layer's bed vignettes and tops out at a mid tone, which would put a dark corner on a
	# scene whose whole point is daylight.
	paint_ground(_hue, _ground_sat, _ground_val, _ch.y * 0.4, _ch.x)
	draw_layers("back")
	_forge.submit(self)
	draw_layers("front")


## The radical-inverse of [param i] in [param b] - the Halton sequence, used here so
## successive drips land spread across the pool instead of clumping.
static func _halton(i: int, b: int) -> float:
	var f := 1.0
	var r := 0.0
	var n := i
	while n > 0:
		f /= float(b)
		r += f * float(n % b)
		n /= b
	return r


## The off-thread build (the [FrameForge] job-object form). It owns the wave field, every
## scratch buffer, the static depth and absorption maps and the baked bed texture; the
## scene never reads any of them, which is what makes a Director cut mid-build harmless.
##
## The mesh is one vertex per NODE with a shared index buffer, so a frame writes colours
## and nothing else. `pts` and `idx` are rebuilt (as NEW arrays, never in place) only when
## the frame size changes, because an array already handed to the main thread must never
## be written again.
class PoolJob:
	extends RefCounted

	var wf: WaveField
	var gw := 96
	var gh := 54
	var org := Vector2.ZERO
	var step := 0.01

	var hh := PackedFloat32Array()
	var gxx := PackedFloat32Array()
	var gyy := PackedFloat32Array()
	var jzz := PackedFloat32Array()
	var acc := PackedFloat32Array()
	var tmp := PackedFloat32Array()
	var dep := PackedFloat32Array()          # water depth at each node - static
	var att := PackedFloat32Array()          # absorption 0..1 at each node - static

	var bed_r := PackedFloat32Array()
	var bed_g := PackedFloat32Array()
	var bed_b := PackedFloat32Array()
	var bw := 128
	var bh := 128
	var bed_org := Vector2.ZERO
	var bed_step := 0.01

	var pts := PackedVector2Array()
	var idx := PackedInt32Array()
	var pts_cell := -1.0

	var sun := Vector3(0.0, 0.0, 1.0)
	var half_v := Vector3(0.0, 0.0, 1.0)
	var eta := 0.75
	var blur_n := 1
	var shine_sq := 5
	var wet_band := 0.02
	var dry_lift := 1.06
	var wet_nodes := 1


	func alloc() -> void:
		var n := gw * gh
		hh.resize(n)
		gxx.resize(n)
		gyy.resize(n)
		jzz.resize(n)
		acc.resize(n)
		tmp.resize(n)
		dep.resize(n)
		att.resize(n)


	## The pool's floor: a constant depth, optionally rising into a dry shelf along one
	## diagonal with a wandering shoreline. Static for the life of the scene, so the
	## absorption it implies is baked here too rather than recomputed 5000 times a frame.
	func build_depth(base: float, shelf: bool, ang: float, length: float, wob: float,
			turbid: float, seed_value: int) -> void:
		var edge := Field.make("fbm", seed_value, 2.6 / maxf(0.01, length), 3)
		var dir := Vector2(cos(ang), sin(ang))
		# A depth of a hand over pale sand still reads as clear water: the extinction
		# coefficient is set so the deepest sampled pool loses about two thirds of its bed.
		var sigma := (1.6 + 5.0 * turbid) / maxf(0.02, base)
		var i := 0
		wet_nodes = 0
		for iy in gh:
			for ix in gw:
				var p := org + Vector2(float(ix), float(iy)) * step
				var d := base
				if shelf:
					# Distance along the diagonal from the shallow corner, wobbled by a
					# field so the waterline is a shoreline rather than a ruled edge.
					var u := (p.dot(dir) + length * 0.5 + (edge.at(p) - 0.5) * 2.0 * wob) / length
					d = base * smoothstep(0.0, 1.0, clampf(u, 0.0, 1.0))
				dep[i] = d
				att[i] = 1.0 - exp(-sigma * d)
				if d > 0.0005:
					wet_nodes += 1
				i += 1
		wet_nodes = maxi(1, wet_nodes)


	## Bake the bed: sand, stones, weed and shell, into a colour texture the render pass
	## samples bilinearly through the refracted view ray.
	##
	## The texture spans half again the pool so the refracted lookup never runs off it, and
	## it is baked at roughly the node grid's own resolution - finer would be detail the
	## grid cannot carry, coarser would show as texels through the water.
	func build_bed(rng: RandomNumberGenerator, res: int, span: float, ramp_name: String,
			stones: int, mix: Dictionary, sun_az: float, sun_el: float) -> void:
		bw = res
		bh = res
		var side := span * 1.5
		bed_step = side / float(bw - 1)
		bed_org = Vector2(-side * 0.5, -side * 0.5)
		var n := bw * bh
		bed_r.resize(n)
		bed_g.resize(n)
		bed_b.resize(n)

		var w_sand: float = mix["sand"]
		var w_weed: float = mix["weed"]
		var w_cobble: float = mix["cobble"]
		var w_shell: float = mix["shell"]

		var ramp := Palette.named(ramp_name, rng)
		var relief := Field.make("fbm", rng.randi(), rng.randf_range(2.0, 4.5), 3)
		var mottle := Field.make("billow", rng.randi(), rng.randf_range(5.0, 11.0), 3)
		var sand_h := rng.randf_range(0.06, 0.13)
		var sand_s := rng.randf_range(0.10, 0.26)
		var sand_v := rng.randf_range(0.62, 0.86)
		var grain := rng.randf_range(0.04, 0.13) * (0.4 + 0.6 * w_sand)
		var weed_thr := lerpf(1.05, 0.42, clampf(w_weed, 0.0, 1.0))
		var weed_h := rng.randf_range(0.22, 0.36)
		var weed_s := rng.randf_range(0.35, 0.7)
		var weed_v := rng.randf_range(0.16, 0.34)
		var hseed := rng.randf() * 100.0

		var i := 0
		for iy in bh:
			for ix in bw:
				var p := bed_org + Vector2(float(ix), float(iy)) * bed_step
				var rl := relief.at(p)
				var mo := mottle.at(p)
				# One cheap hash for the sand grain: a third noise field would double the
				# bake for detail that only has to be uncorrelated, not smooth.
				var hsh := sin(float(ix) * 127.1 + float(iy) * 311.7 + hseed) * 43758.5453
				hsh -= floor(hsh)
				var v := clampf(sand_v * (0.82 + 0.30 * rl) + (hsh - 0.5) * grain, 0.0, 1.0)
				var col := Color.from_hsv(fposmod(sand_h + (mo - 0.5) * 0.03, 1.0),
					clampf(sand_s * (0.75 + 0.5 * mo), 0.0, 1.0), v)
				if mo > weed_thr:
					var wk := clampf((mo - weed_thr) / maxf(0.02, 1.0 - weed_thr), 0.0, 1.0)
					col = col.lerp(Color.from_hsv(fposmod(weed_h + (rl - 0.5) * 0.05, 1.0),
						weed_s, weed_v * (0.7 + 0.5 * rl)), wk * 0.85)
				bed_r[i] = col.r
				bed_g[i] = col.g
				bed_b[i] = col.b
				i += 1

		# Stones, rasterized over their own bounding boxes only. Each gets a dome normal so
		# it is lit round rather than being a flat disc, and a contact shadow on the side
		# away from the sun - the only reason a top-down cobble reads as a solid at all.
		var sx := cos(sun_az)
		var sy := sin(sun_az)
		var lam := cos(sun_el)
		var count := int(round(float(stones) * clampf(0.3 + w_cobble, 0.3, 1.6)))
		for k in count + int(round(float(stones) * w_shell * 0.5)):
			var shell := k >= count
			var c := bed_org + Vector2(rng.randf(), rng.randf()) * (side)
			var rx: float = span * (rng.randf_range(0.012, 0.030) if shell
				else rng.randf_range(0.022, 0.075))
			var ry := rx * rng.randf_range(0.62, 1.0)
			var rot := rng.randf() * TAU
			var cr := cos(rot)
			var sr := sin(rot)
			var t := rng.randf()
			var sc: Color = ramp.at(0.55 + 0.4 * t)
			if shell:
				sc = Color.from_hsv(fposmod(sand_h + rng.randf_range(-0.04, 0.06), 1.0),
					rng.randf_range(0.04, 0.16), rng.randf_range(0.86, 0.99))
			else:
				sc = Color.from_hsv(sc.h, clampf(sc.s * rng.randf_range(0.5, 1.0), 0.0, 1.0),
					clampf(sc.v * rng.randf_range(0.6, 1.05), 0.0, 1.0))
			var rmax := maxf(rx, ry) + bed_step
			var ix0 := maxi(0, int(floor((c.x - rmax - bed_org.x) / bed_step)))
			var ix1 := mini(bw - 1, int(ceil((c.x + rmax - bed_org.x) / bed_step)))
			var iy0 := maxi(0, int(floor((c.y - rmax - bed_org.y) / bed_step)))
			var iy1 := mini(bh - 1, int(ceil((c.y + rmax - bed_org.y) / bed_step)))
			for iy in range(iy0, iy1 + 1):
				var py := bed_org.y + float(iy) * bed_step - c.y
				for ix in range(ix0, ix1 + 1):
					var px := bed_org.x + float(ix) * bed_step - c.x
					var lx := (px * cr + py * sr) / rx
					var ly := (-px * sr + py * cr) / ry
					var d2 := lx * lx + ly * ly
					if d2 >= 1.0:
						continue
					# A hemisphere's normal, so the shading is the stone's shape and not a
					# radial gradient painted onto it.
					var nz := sqrt(1.0 - d2)
					var lit := clampf(0.42 + 0.58 * (lx * sx * lam + ly * sy * lam
						+ nz * sin(sun_el)), 0.0, 1.35)
					var rim := smoothstep(1.0, 0.86, d2)      # darkened contact edge
					var j := iy * bw + ix
					bed_r[j] = clampf(sc.r * lit * rim, 0.0, 1.0)
					bed_g[j] = clampf(sc.g * lit * rim, 0.0, 1.0)
					bed_b[j] = clampf(sc.b * lit * rim, 0.0, 1.0)


	func run(s: Dictionary) -> Array:
		var n := gw * gh
		var amps: PackedFloat32Array = s["amps"]
		wf.steep = float(s["steep"])
		wf.set_amps(amps)

		hh.fill(0.0)
		gxx.fill(0.0)
		gyy.fill(0.0)
		jzz.fill(1.0)
		wf.sweep(float(s["t"]), org, step, gw, gh, hh, gxx, gyy, jzz)
		var rings: Array = s["rings"]
		WaveField.sweep_rings(rings, org, step, gw, gh, hh, gxx, gyy, jzz)

		acc.fill(0.0)
		var total := _cast()
		_blur()
		# Normalize so the average WET node receives 1.0. The net is then a contrast about
		# that mean rather than an absolute brightness, which is what keeps a shallow pool
		# and a deep one equally readable.
		var norm := float(wet_nodes) / maxf(0.0001, total)

		var cell: float = s["cell"]
		if cell != pts_cell:
			_build_mesh(cell)
		var cols := PackedColorArray()
		cols.resize(n + (gw - 1) * (gh - 1))     # nodes, then the cross-split cell centres
		_shade(s, norm, cols)
		_centres(cols)
		return [{"pts": pts, "cols": cols, "idx": idx}]


	## Refract the sun through every node's surface normal and splat where it lands on the
	## bed. Returns the total energy launched, for the normalization.
	func _cast() -> float:
		var total := 0.0
		var sxv := sun.x
		var syv := sun.y
		var szv := sun.z
		var i := 0
		for iy in gh:
			for ix in gw:
				var d := dep[i]
				if d <= 0.0005:
					i += 1
					continue
				var nx := -gxx[i]
				var ny := -gyy[i]
				var nz := maxf(jzz[i], JZ_FLOOR)
				var inv := 1.0 / sqrt(nx * nx + ny * ny + nz * nz)
				nx *= inv
				ny *= inv
				nz *= inv
				var cosi := nx * sxv + ny * syv + nz * szv
				if cosi <= 0.02:
					i += 1
					continue
				# Snell in vector form, air into water. k2 cannot go negative going into
				# the denser medium, so there is no total internal reflection to handle.
				var k2 := 1.0 - eta * eta * (1.0 - cosi * cosi)
				var fr := eta * cosi - sqrt(maxf(k2, 0.0))
				var tx := -eta * sxv + fr * nx
				var ty := -eta * syv + fr * ny
				var tz := -eta * szv + fr * nz
				if tz >= -0.02:
					i += 1
					continue
				var u := 1.0 - cosi
				var u2 := u * u
				var fres := F0 + (1.0 - F0) * u2 * u2 * u
				var wgt := cosi * (1.0 - fres)
				var travel := (d + hh[i]) / -tz
				# Where the refracted ray lands on the bed, in grid CELLS. The origin
				# cancels against itself, so this is the launch cell plus the lateral
				# travel measured in cells.
				var fx := float(ix) + tx * travel / step
				var fy := float(iy) + ty * travel / step
				# WRAPPED, not clamped or dropped. Energy leaving one edge has to go
				# somewhere: dropping it puts a dim band round the frame, piling it at the
				# border puts a bright one, and wrapping lands it in a net that already
				# looks like every other part of the net.
				var x0 := int(floor(fx))
				var y0 := int(floor(fy))
				var ax := fx - float(x0)
				var ay := fy - float(y0)
				x0 = x0 % gw
				if x0 < 0:
					x0 += gw
				y0 = y0 % gh
				if y0 < 0:
					y0 += gh
				var x1 := x0 + 1
				if x1 >= gw:
					x1 = 0
				var y1 := y0 + 1
				if y1 >= gh:
					y1 = 0
				var r0 := y0 * gw
				var r1 := y1 * gw
				var w11 := ax * ay
				var w01 := ay - w11
				var w10 := ax - w11
				acc[r0 + x0] += wgt * (1.0 - ax - ay + w11)
				acc[r0 + x1] += wgt * w10
				acc[r1 + x0] += wgt * w01
				acc[r1 + x1] += wgt * w11
				total += wgt
				i += 1
		return total


	## A separable [1 2 1] pass, wrapped to match the splat. This is not decoration: the
	## sun has about half a degree of angular width, so a real caustic rib has a soft
	## shoulder, and an unblurred accumulation buffer reads as speckle rather than as light.
	## A separable binomial [1 4 6 4 1]/16 pass, wrapped to match the splat.
	##
	## FIVE TAPS, NOT THREE, and it is a straight saving. Smoothing is what makes the net read as
	## light rather than as speckle, and the amount of it is the accumulated variance: a [1 2 1]
	## pass carries 0.5 cells^2, this one carries 1.0, so one pass here is worth two of those.
	## Measured before the change, a five-pass [1 2 1] blur cost 9-13 ms of a 50-76 ms build - the
	## same smoothing now costs two or three passes, and the per-pass loop overhead is paid half
	## as often. Wrapping rather than clamping is not decoration: _cast wraps its splat, so a rib
	## leaving one edge is already accounted for on the other, and clamping here would pile a
	## bright fringe against the border that the splat deliberately avoids making.
	func _blur() -> void:
		for _p in blur_n:
			for iy in gh:
				var row := iy * gw
				for ix in gw:
					var x1 := ix - 1
					var x2 := ix - 2
					var x3 := ix + 1
					var x4 := ix + 2
					if x1 < 0:
						x1 += gw
					if x2 < 0:
						x2 += gw
					if x3 >= gw:
						x3 -= gw
					if x4 >= gw:
						x4 -= gw
					tmp[row + ix] = 0.0625 * (acc[row + x2] + acc[row + x4]) \
						+ 0.25 * (acc[row + x1] + acc[row + x3]) + 0.375 * acc[row + ix]
			for iy in gh:
				var y1 := iy - 1
				var y2 := iy - 2
				var y3 := iy + 1
				var y4 := iy + 2
				if y1 < 0:
					y1 += gh
				if y2 < 0:
					y2 += gh
				if y3 >= gh:
					y3 -= gh
				if y4 >= gh:
					y4 -= gh
				var row := iy * gw
				var r1 := y1 * gw
				var r2 := y2 * gw
				var r3 := y3 * gw
				var r4 := y4 * gw
				for ix in gw:
					acc[row + ix] = 0.0625 * (tmp[r2 + ix] + tmp[r4 + ix]) \
						+ 0.25 * (tmp[r1 + ix] + tmp[r3 + ix]) + 0.375 * tmp[row + ix]


	## One colour per node: the refracted image of the bed, lit by the caustic net,
	## absorbed by the water above it, with the sky and its glare laid over the top.
	func _shade(s: Dictionary, norm: float, cols: PackedColorArray) -> void:
		var water: Color = s["water"]
		var sky: Color = s["sky"]
		var sun_c: Color = s["sun"]
		var amb: float = s["amb"]
		var contrast: float = s["contrast"]
		var sharp: float = s["sharp"]
		var expo: float = s["exposure"]
		var thr: float = s["glare_thr"]
		var gg: float = s["glare_gain"]
		var hx := half_v.x
		var hy := half_v.y
		var hz := half_v.z
		var inv_thr := 1.0 / maxf(0.02, 1.0 - thr)
		var i := 0
		for iy in gh:
			var py := org.y + float(iy) * step
			for ix in gw:
				var px := org.x + float(ix) * step
				var d := dep[i]
				if d <= 0.0005:
					# The dry shelf: bare bed, no tint, no net, lifted a little because dry
					# stone is paler than wet stone.
					var dc := _bed_rgb(px, py)
					cols[i] = Color(clampf(dc.r * dry_lift, 0.0, 1.0),
						clampf(dc.g * dry_lift, 0.0, 1.0),
						clampf(dc.b * dry_lift, 0.0, 1.0))
					i += 1
					continue
				var nx := -gxx[i]
				var ny := -gyy[i]
				var nz := maxf(jzz[i], JZ_FLOOR)
				var inv := 1.0 / sqrt(nx * nx + ny * ny + nz * nz)
				nx *= inv
				ny *= inv
				nz *= inv

				# The VIEW ray, straight down, refracted through the same normal. This is
				# what makes the stones swim: the bed we see is not the bed under us.
				var k2 := 1.0 - eta * eta * (1.0 - nz * nz)
				var fr := eta * nz - sqrt(maxf(k2, 0.0))
				var vz := -eta + fr * nz
				var travel := (d + hh[i]) / maxf(0.05, -vz)
				var bc := _bed_rgb(px + fr * nx * travel, py + fr * ny * travel)

				# The caustic, as a contrast about its own mean, optionally sharpened. A
				# power would be the honest curve; squaring and blending is the same shape
				# for a fifth of the cost, and this runs once per node per frame.
				var c := 1.0 + contrast * (acc[i] * norm - 1.0)
				if c < 0.0:
					c = 0.0
				c = c + sharp * (c * c - c)
				var lit := (amb + (1.0 - amb) * c) * expo
				var aa := att[i]
				var tr := 1.0 - aa
				var wl := 0.55 + 0.45 * clampf(lit, 0.0, 2.0)

				# Specular: Blinn against a straight-down camera, exponent by repeated
				# squaring so no pow() runs in this loop.
				var sp := nx * hx + ny * hy + nz * hz
				if sp < 0.0:
					sp = 0.0
				for _q in shine_sq:
					sp = sp * sp
				var glare := clampf((sp - thr) * inv_thr, 0.0, 1.0) * gg
				# Fresnel on the view ray: near-normal it is two percent, and only a steep
				# face lifts it, which is exactly when water flashes.
				var uf := 1.0 - nz
				var uf2 := uf * uf
				var refl := F0 + (1.0 - F0) * uf2 * uf2 * uf + glare

				# The shoreline: sand just under the water is darker than sand out of it.
				var wet := 1.0
				if d < wet_band:
					wet = lerpf(0.74, 1.0, clampf(d / maxf(0.0001, wet_band), 0.0, 1.0))

				var rr := bc.r * lit * sun_c.r * tr * wet + water.r * aa * wl + sky.r * refl
				var gg2 := bc.g * lit * sun_c.g * tr * wet + water.g * aa * wl + sky.g * refl
				var bb := bc.b * lit * sun_c.b * tr * wet + water.b * aa * wl + sky.b * refl
				cols[i] = Color(clampf(rr, 0.0, 1.0), clampf(gg2, 0.0, 1.0),
					clampf(bb, 0.0, 1.0))
				i += 1


	## Bilinear read of the baked bed. Bilinear and not nearest because the refracted
	## lookup moves by a fraction of a texel per frame, and nearest turns that into every
	## stone twitching between texels.
	func _bed_rgb(x: float, y: float) -> Color:
		var fx := clampf((x - bed_org.x) / bed_step, 0.0, float(bw - 1) - 0.001)
		var fy := clampf((y - bed_org.y) / bed_step, 0.0, float(bh - 1) - 0.001)
		var x0 := int(fx)
		var y0 := int(fy)
		var ax := fx - float(x0)
		var ay := fy - float(y0)
		var a := y0 * bw + x0
		var b := a + bw
		var r0 := bed_r[a] + (bed_r[a + 1] - bed_r[a]) * ax
		var r1 := bed_r[b] + (bed_r[b + 1] - bed_r[b]) * ax
		var g0 := bed_g[a] + (bed_g[a + 1] - bed_g[a]) * ax
		var g1 := bed_g[b] + (bed_g[b + 1] - bed_g[b]) * ax
		var b0 := bed_b[a] + (bed_b[a + 1] - bed_b[a]) * ax
		var b1 := bed_b[b] + (bed_b[b + 1] - bed_b[b]) * ax
		return Color(r0 + (r1 - r0) * ay, g0 + (g1 - g0) * ay, b0 + (b1 - b0) * ay)


	## Vertex positions and the index buffer for the node mesh. Built once per frame SIZE,
	## never per frame, and always as new arrays - the previous ones may still be on the
	## main thread inside a packet being drawn.
	## CROSS-SPLIT, four triangles about a centre vertex, not two about a diagonal.
	##
	## This is the fix for "the waves are blocky and jagged... even though their behavior is
	## smooth". A cell is 18 to 28 screen pixels across at 1920x1080 (the grid is
	## SWEEP_BUDGET / components nodes, so 76x43 at fourteen components and 117x65 at six), and
	## splitting each one along a single diagonal makes the Gouraud interpolation directional:
	## every cell shades along the SAME 45-degree line, so a smooth caustic rib comes out as a
	## herringbone of straight creases all leaning one way. That reads as jagged even where
	## nothing about the light is.
	##
	## Fanning from the centre is symmetric, so the artifact has no direction left to have, and
	## the centre's colour is the mean of the four corners - which is exactly the value bilinear
	## interpolation gives there, i.e. the one point the two old triangles disagreed about most.
	## Cost: one extra vertex and two extra triangles per cell, 12.6k triangles against 6.3k on
	## the coarse grid, still one draw call - and the positions and indices are built once per
	## frame SIZE, so a frame only pays for the extra colours (about 3.2k cheap averages).
	##
	## What it does NOT do is raise the sampling rate: the caustic buffer is still accumulated at
	## node resolution, so a rib is still one to two cells wide. That is why the blur passes were
	## widened alongside this - see blur_n.
	func _build_mesh(cell: float) -> void:
		var cells := (gw - 1) * (gh - 1)
		var p := PackedVector2Array()
		p.resize(gw * gh + cells)
		var ox := -float(gw - 1) * cell * 0.5
		var oy := -float(gh - 1) * cell * 0.5
		var i := 0
		for iy in gh:
			var y := oy + float(iy) * cell
			for ix in gw:
				p[i] = Vector2(ox + float(ix) * cell, y)
				i += 1
		for iy in gh - 1:                      # the cell centres, appended after the nodes
			var y := oy + (float(iy) + 0.5) * cell
			for ix in gw - 1:
				p[i] = Vector2(ox + (float(ix) + 0.5) * cell, y)
				i += 1
		if idx.is_empty():
			var q := PackedInt32Array()
			q.resize(cells * 12)
			var j := 0
			var m := gw * gh
			for iy in gh - 1:
				var row := iy * gw
				var crow := m + iy * (gw - 1)
				for ix in gw - 1:
					var a := row + ix
					var b := a + 1
					var c := a + gw
					var d := c + 1
					var e := crow + ix
					q[j] = a
					q[j + 1] = b
					q[j + 2] = e
					q[j + 3] = b
					q[j + 4] = d
					q[j + 5] = e
					q[j + 6] = d
					q[j + 7] = c
					q[j + 8] = e
					q[j + 9] = c
					q[j + 10] = a
					q[j + 11] = e
					j += 12
			idx = q
		pts = p
		pts_cell = cell


	## The cell-centre colours, filled after _shade has written the node colours.
	func _centres(cols: PackedColorArray) -> void:
		var j := gw * gh
		for iy in gh - 1:
			var row := iy * gw
			for ix in gw - 1:
				var a := row + ix
				var b := a + 1
				var c := a + gw
				var d := c + 1
				cols[j] = Color(
					(cols[a].r + cols[b].r + cols[c].r + cols[d].r) * 0.25,
					(cols[a].g + cols[b].g + cols[c].g + cols[d].g) * 0.25,
					(cols[a].b + cols[b].b + cols[c].b + cols[d].b) * 0.25)
				j += 1
