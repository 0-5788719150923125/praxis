extends GhostScene

## Cloth - a sheet on a line, and the wind deciding how long it stays whole.
##
## A rectangle of fabric hangs from pins along the top of frame and fills most of it. It
## breathes and luffs, catching light in its folds, its lower edge lifting and snapping. As
## the wind builds the fabric stretches, a seam gives way with a visible run down the weave,
## and a strip peels off and streams away still rippling. By the end the sheet is a set of
## ragged ribbons on the same line, still moving.
##
## THIS IS THE FIRST DEFORMABLE SUBJECT IN THE CATALOGUE. Everything else here is rigid, a
## particle, or a field: [Primitives] is eight field-like forces with no contact and no
## constraint of any kind. Cloth is a real solver - positions and previous positions in packed
## arrays, distance constraints relaxed Gauss-Seidel for N iterations, pins as zero-inverse-mass
## nodes - and it is also the only subject that DESTROYS ITSELF, so a long scene has an arc
## with no keyframes, no lifecycle scripting and no authored tear time. Whether a seam gives
## depends only on accumulated stress in that seam.
##
## THE SHEET IS SIMULATED IN 3D, drawn in 2D. Two extra floats per node buy the whole look: a
## fold has a genuine surface normal, so Lambert shading, a translucent glow where light comes
## through the back, and a specular sheen on the crest of a fold are all real rather than
## faked from projected area - and a gust can push the cloth TOWARD the viewer and be seen as
## a bulge instead of a sideways smear. The projection is deliberately weak (a linear
## `1 + z * persp` scale about frame centre, not a perspective divide) because a hanging sheet
## is square-on - it wants dimension, not a tumbling-card read. Framing is `plane` for the same
## reason, which puts the camera in `PLANE_BAG`: centred, occasionally a slow push.
##
## THE TEAR IS CONSTRAINT REMOVAL, AND IT PROPAGATES ALONG THE WEAVE. Every structural
## constraint carries a yield strain sampled at build (the weave irregularity - where the
## fabric is weakest is decided by the seed, so a given instance always gives at the same
## place) and accumulates DAMAGE at a rate proportional to how far past yield it is, so a long
## quiet passage and a short violent one can both eventually break the same seam. When damage
## reaches 1 the constraint is removed, and with it the shear diagonals of the two cells it
## bordered and the single bend constraint that spanned it - otherwise a diagonal would bridge
## the rip and it would never open.
##
## The propagation is the part worth explaining. A break raises a WEAKENING field on the
## neighbouring edges of the weave grid: strongly on the two edges continuing the run (the same
## warp/weft line, one step along), weakly on the edges to either side, and weaker still on the
## crossing edges at the tip - which is stress concentration at a crack tip, written as data.
## An edge's effective yield is its sampled strength times `1 - weak`. Because the boost along
## the run is several times the sideways boost, the next thing to give is nearly always the
## next edge in line, so the tear reads as a RUN rather than as the fabric shredding everywhere
## at once. The weakening decays a little every tick, so a run that stops finding stress
## arrests instead of inevitably crossing the whole sheet.
##
## Which way the runs go is a sampled property of the FABRIC, not a scripted outcome.
## `weave_bias` is the ratio of weft (vertical thread) yield to warp (horizontal thread) yield:
## above 1 the warp gives first, so the runs are vertical and the sheet ends as ribbons still
## hanging from the line; below 1 the weft gives first, so a horizontal seam parts and the whole
## lower strip peels off and streams away. Both endings live in the same scene.
##
## WHAT THE SEED DECIDES. The mood ([Scheme]) and the whole palette from it. Node grid and
## therefore mesh resolution; solver iterations (see [constant SOLVE_BUDGET]); structural,
## shear and bend stiffness; slack (rest lengths shorter than the grid, so it drapes from the
## first frame); GATHER, how far the pins are pulled in from the fabric's natural width, which
## is what makes a curtain buckle into standing folds rather than hanging flat; the initial
## fold ripple that gives the buckling a direction; the pin pattern (a line of pegs, two
## corners, a diagonal, or a single point at the centre); mass per node, which sets drape by
## dividing the wind but not gravity; damping; gravity; the base wind and how much movement
## adds to it; the wind's direction and turbulence gain; the gust vocabulary; how far the
## threads give before they simply cannot lengthen (`max_stretch`); tearing on or off and its
## yield, irregularity, damage rate, crack-tip gains, weakening decay and breaks per tick;
## `weave_bias`; opacity, translucency, sheen and shininess; the light direction; the
## perspective strength; whether the frame is high-key ([method GhostScene.paint_ground]) or a
## dark [Layer] bed; and whether shed strips settle on a floor at the bottom of frame or simply
## leave.
##
## AUDIO, AND WHY NOTHING HERE MULTIPLIES A SIZE BY LOUDNESS. The wind is a [Flow2D] curl field
## and its overall speed follows a slow [method Nonlinear.flare] envelope of `f.movement`; its
## turbulence scale follows `f.flux` (typically 0.01-0.05, so the gain is large and the result
## is still a modest 1.0-1.7x change in spatial frequency). Loudness never sets an amplitude:
## the fabric's RESPONSE to the wind is the amplitude, which is exactly why a loud smooth
## passage and a quiet busy one look different here - something an amplitude mapping cannot do
## at all. A `f.beat` rising edge releases a GUST, a localized pressure blob that crosses the
## sheet at its own speed, so the beat is seen as a wave travelling through the fabric and
## arriving LATE at the far edge; that delay is physical, not scheduled. The gusts come from a
## vocabulary pre-rolled at build and cycled by a counter, never from a live draw, because the
## live analyzer and the offline export bake do not produce identical beat streams and the
## render would diverge from the preview. `f.treble` lifts the specular sheen on folds whose
## normal faces the light, [Lighting] sweeps hotspots across the sheet, and `chroma_hue()`
## pulls the fabric's colour toward the music's tonal centre.
##
## COST, AND THE TWO NUMBERS THAT SHAPED IT. Everything is budgeted, because a Gauss-Seidel
## relaxation is `constraints x iterations` inner steps per tick and that product is the entire
## frame cost of this scene. [constant SOLVE_BUDGET] caps it, and iterations are clamped DOWN
## to fit whatever grid the seed rolled - so resolution and stiffness trade against each other
## instead of multiplying, and a fine mesh is automatically a softer one. The wind field is
## sampled on a coarse lattice (a handful of [Flow2D] probes per tick) and bilinearly
## interpolated per node, because wind has a far coarser spatial scale than the mesh and a
## curl-noise probe is four 3D noise lookups - one per node would cost more than the solver.
##
## The whole simulation runs on a [SimClock] at 60 Hz with a two-tick ceiling per call. That is
## not decoration: the [Director] sub-steps `update()` up to 15 times in one frame, pre-warms
## every scene with 12 calls before its first frame, and an [Echo] re-localize can fast-forward
## hundreds of calls. A cloth that advanced once per call would arrive on screen already torn.
## The pre-warm produces 24 ticks, 0.4 s of simulated time, which is why `tear_delay` is
## sampled no lower than 1.2 s - the opening frame is a settled, whole sheet, always.

## Ceiling on constraint solves per tick (constraints x iterations). The solver is the frame
## cost, so it is capped rather than hoped for: iterations are clamped down until the product
## fits. A coarse 18x12 grid can afford eight or nine relaxation passes and reads as stiff
## canvas; a 30x20 grid gets two or three and reads as silk. That is a feature - the seed
## cannot roll a sheet that is both very fine and very stiff, and neither can it roll one that
## blocks the main thread.
const SOLVE_BUDGET := 7000

## Draw-time subdivision per simulated cell. 2 quadruples the triangle count and removes
## the faceting; 3 is smoother on a very coarse sheet and rarely worth it. 1 disables it.
const SUBDIV := 2

## Yield strain stored on constraints that must NEVER tear - shear diagonals and bend
## constraints. They are removed only as a consequence of the structural edge they cross
## giving way, never on their own, which is what keeps a tear on the weave.
const NO_YIELD := 1000.0

## Hard ceiling on the yield strain any instance may sample. A relaxation solver run for a
## handful of iterations does not reach equilibrium, so a hanging sheet carries a REAL residual
## strain - the load a constraint transports per tick against the correction it can transport
## back - and that residual is what the tear threshold has to sit above to be a threshold at
## all. It also has to sit BELOW the over-stretch clamp (`max_stretch`, sampled no lower than
## 1.25) or nothing could ever reach yield and the fabric would simply be indestructible.
const YIELD_MAX := 0.18

## Structural candidates queued for breaking in one tick. Bounded so a catastrophic frame
## (the whole sheet past yield at once) cannot turn the selection scan into a real cost.
const MAX_CAND := 48

## Gusts alive at once. Each one is an exp() per node per tick, so this is a real budget;
## three overlapping pressure blobs is already more weather than the eye can follow.
const MAX_GUSTS := 3

## The [Lighting] hotspot field is evaluated on a lattice this size and interpolated per node,
## for the same reason the wind is: `Lighting.at` re-derives each hotspot's drifting centre
## through [ModBank], which is five sines and a dictionary lookup per channel, so calling it
## once per node would be tens of thousands of sines a frame. Hotspots are half-screen-wide
## gaussians - there is nothing at mesh resolution for a finer lattice to resolve.
const HOT_W := 6
const HOT_H := 5

## Pin patterns. `line` is a row of pegs along the top (a washing line), `corners` hangs it
## from its two top corners so it swags, `diagonal` pins a run down one side so it hangs like
## a flag half-struck, `point` hangs the whole sheet from a single peg at the centre of the top
## edge - which drapes into a cone and tears from the pin outward.
const PIN_PATTERNS := ["line", "line", "corners", "diagonal", "point"]

# ---------------------------------------------------------------------------------
# Mesh + solver state. All of it lives in packed arrays: a tear must never allocate
# mid-solve, so constraints are marked dead by SWAP-REMOVE into a live prefix and the
# backing arrays keep their capacity for the life of the scene.
# ---------------------------------------------------------------------------------
var _sim: SimClock = SimClock.new(60.0, 2)

var _cols := 24
var _rows := 16
var _n := 0

var _px := PackedFloat32Array()
var _py := PackedFloat32Array()
var _pz := PackedFloat32Array()
var _qx := PackedFloat32Array()
var _qy := PackedFloat32Array()
var _qz := PackedFloat32Array()
var _w := PackedFloat32Array()          # inverse weight: 0 = pinned, 1 = free

var _pin := PackedInt32Array()          # pinned node indices
var _pax := PackedFloat32Array()        # and their anchors
var _pay := PackedFloat32Array()
var _paz := PackedFloat32Array()

var _ca := PackedInt32Array()
var _cb := PackedInt32Array()
var _crest := PackedFloat32Array()
var _cstiff := PackedFloat32Array()
var _cyield := PackedFloat32Array()
var _cdmg := PackedFloat32Array()
var _cslot := PackedInt32Array()        # weave-grid slot, or -1
var _cn := 0                            # live constraint count (the prefix length)

var _cof := PackedInt32Array()          # slot -> live constraint index, or -1 if gone
var _weak := PackedFloat32Array()       # crack-tip weakening, one per structural slot
var _cand := PackedInt32Array()         # slots past damage 1.0 this tick
var _cells := PackedInt32Array()        # drawable cells, rebuilt when the weave changes
var _cells_dirty := true

# Slot layout: [warp | weft | shear | bend-h | bend-v]
var _n_warp := 0
var _n_weft := 0
var _s_weft := 0
var _s_shear := 0
var _s_bh := 0
var _s_bv := 0
var _nbh := 0
var _nbv := 0

# ---------------------------------------------------------------------------------
# Wind, gusts, light.
# ---------------------------------------------------------------------------------
var _flow: Flow2D
var _light: Lighting
var _gw := 8
var _gh := 6
var _gx0 := -0.8
var _gx1 := 0.8
var _gy0 := -0.5
var _gy1 := 0.6
var _wxa := PackedFloat32Array()
var _wya := PackedFloat32Array()
var _wza := PackedFloat32Array()
var _hota := PackedFloat32Array()

var _gusts: Array = []
var _gust_specs: Array = []
var _gust_n := 0
var _prev_beat := 0.0

var _wind := 0.0                 # live wind speed (flared envelope of movement)
var _wind_target := 0.0
var _turb := 1.0
var _turb_target := 1.0

# ---------------------------------------------------------------------------------
# Sampled definition, cached as typed members (the hot loops never touch `params`).
# ---------------------------------------------------------------------------------
var _iters := 4
var _grav := 0.8
var _damp := 0.985
var _mass := 1.0
var _wind_base := 0.10
var _wind_gain := 0.55
var _wind_dx := 1.0
var _wind_dy := 0.0
var _turb_gain := 14.0
var _sway := 0.012

var _tearing := true
var _tear_delay := 2.0
var _dmg_rate := 0.9
var _tip := 0.55
var _cross := 0.14
var _weak_decay := 0.994
var _break_budget := 2
var _max_stretch := 1.35

var _floor_on := false
var _floor_y := 0.46
var _friction := 0.6

var _persp := 0.30
var _opacity := 0.5
var _trans := 0.45
var _amb := 0.30
var _diff := 0.72
var _sheen := 0.35
var _sheen_gain := 0.9
var _shine := 24.0
var _hot := 0.30
var _hue := 0.1
var _hue_back := 0.5
var _hue_fold := 0.05
var _sat := 0.5
var _val := 0.85
var _lx := 0.3
var _ly := -0.5
var _lz := 0.8
var _hx := 0.0
var _hy := 0.0
var _hz := 1.0

var _high_key := false
var _ground_hue := 0.55
var _ground_sat := 0.08
var _ground_val := 0.92
var _line_y := -0.4
var _line_on := true
var _line_val := 0.25

# Per-frame scratch (allocated once, refilled every draw).
var _spos := PackedVector2Array()
var _ncol := PackedColorArray()
var _keys := PackedInt64Array()

var _f: AudioFeatures = AudioFeatures.new()
var _ch := Vector2.ZERO


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "plane"
	var sch := Scheme.pick(rng)

	_cols = rng.randi_range(18, 30)
	_rows = rng.randi_range(12, 20)
	_n = _cols * _rows

	# The sheet's extent in ghost's centred unit-fraction space. Wide enough to fill a
	# 16:9 frame edge to edge at the top of the range, tall enough that the free lower
	# edge is well inside frame so its snap is visible.
	var wide := rng.randf_range(1.00, 1.55)
	var tall := rng.randf_range(0.62, 0.92)
	_line_y = rng.randf_range(-0.44, -0.34)
	var dx := wide / float(_cols - 1)
	var dy := tall / float(_rows - 1)

	var slack := rng.randf_range(0.0, 0.09)
	var gather := rng.randf_range(0.02, 0.20)
	var folds := float(rng.randi_range(2, 6))
	var fold_amp := rng.randf_range(0.012, 0.045)
	var fold_phase := rng.randf() * TAU

	_mass = rng.randf_range(0.6, 1.8)
	_grav = rng.randf_range(0.50, 1.10)
	_damp = rng.randf_range(0.975, 0.994)

	# --- nodes -------------------------------------------------------------------
	_px.resize(_n); _py.resize(_n); _pz.resize(_n)
	_qx.resize(_n); _qy.resize(_n); _qz.resize(_n)
	_w.resize(_n)
	for r in _rows:
		var rt := float(r) / float(_rows - 1)
		for c in _cols:
			var i := r * _cols + c
			var x := -wide * 0.5 + dx * float(c)
			var y := _line_y + dy * float(r)
			# A seeded ripple across the columns, growing downward. The gathered pins put
			# the top row into compression and it MUST buckle somewhere; without a preferred
			# direction it buckles into numerical noise, which reads as a crumpled bag
			# rather than as a curtain. This is the direction.
			var z := fold_amp * (0.35 + 0.65 * rt) * sin(float(c) / float(_cols - 1) * TAU * folds + fold_phase)
			_px[i] = x; _py[i] = y; _pz[i] = z
			_qx[i] = x; _qy[i] = y; _qz[i] = z
			_w[i] = 1.0

	# --- pins --------------------------------------------------------------------
	var pattern := String(PIN_PATTERNS[rng.randi() % PIN_PATTERNS.size()])
	var step := rng.randi_range(2, 6)
	_pin.clear(); _pax.clear(); _pay.clear(); _paz.clear()
	_line_on = pattern == "line" or pattern == "corners"
	match pattern:
		"corners":
			_add_pin(0, gather, wide)
			_add_pin(_cols - 1, gather, wide)
		"diagonal":
			var m := mini(_rows, _cols)
			var k := 0
			while k < m:
				_add_pin(k * _cols + k, gather, wide)
				k += 2
		"point":
			_add_pin(int(_cols / 2.0), gather, wide)
		_:
			var c := 0
			while c < _cols:
				_add_pin(c, gather, wide)
				c += step
			if _pin.size() < 2 or _pin[_pin.size() - 1] != _cols - 1:
				_add_pin(_cols - 1, gather, wide)

	# --- the weave ---------------------------------------------------------------
	_n_warp = _rows * (_cols - 1)
	_n_weft = (_rows - 1) * _cols
	var n_shear := (_rows - 1) * (_cols - 1)
	_nbh = maxi(0, int((_cols - 3) / 2.0) + 1) if _cols >= 3 else 0
	_nbv = maxi(0, int((_rows - 3) / 2.0) + 1) if _rows >= 3 else 0
	_s_weft = _n_warp
	_s_shear = _n_warp + _n_weft
	_s_bh = _s_shear + n_shear
	_s_bv = _s_bh + _rows * _nbh
	var n_slots := _s_bv + _nbv * _cols

	var k_struct := rng.randf_range(0.72, 1.0)
	var k_shear := rng.randf_range(0.22, 0.55)
	var k_bend := rng.randf_range(0.04, 0.20)

	# Tearing. `weave_bias` is the weft/warp yield ratio and it decides the ENDING:
	# above 1 the warp gives first and the sheet becomes ribbons on the line, below 1
	# the weft parts and the lower strip peels away.
	_tearing = rng.randf() < 0.7
	var y_warp := rng.randf_range(0.07, YIELD_MAX)
	var weave_bias := rng.randf_range(0.55, 1.75)
	var y_weft := minf(YIELD_MAX, y_warp * weave_bias)
	var irr := rng.randf_range(0.06, 0.36)
	_max_stretch = rng.randf_range(1.25, 1.5)
	_tear_delay = rng.randf_range(1.2, 3.6)
	# Damage rate, and why it is this large. A seam sitting a little past yield should creep
	# for many seconds; a crack TIP, which both carries the load its dead neighbour dropped and
	# has had its own yield cut by the weakening field, sits several times further past and so
	# gives in well under a second. One rate produces both because the excess strain, not the
	# rate, is what differs between them - which is the whole reason the run reads as a run.
	_dmg_rate = rng.randf_range(3.0, 12.0)
	_tip = rng.randf_range(0.35, 0.85)
	_cross = rng.randf_range(0.05, 0.24)
	_weak_decay = rng.randf_range(0.985, 0.9985)
	_break_budget = rng.randi_range(1, 3)

	var cap := n_slots
	_ca.resize(cap); _cb.resize(cap); _crest.resize(cap)
	_cstiff.resize(cap); _cyield.resize(cap); _cdmg.resize(cap); _cslot.resize(cap)
	_cof.resize(n_slots); _cof.fill(-1)
	_weak.resize(_s_shear); _weak.fill(0.0)
	_cn = 0

	var rest_x := dx * (1.0 - slack)
	var rest_y := dy * (1.0 - slack)
	var rest_d := sqrt(dx * dx + dy * dy) * (1.0 - slack)
	for r in _rows:
		for c in _cols - 1:
			_add_con(r * _cols + c, r * _cols + c + 1, rest_x, k_struct,
				y_warp * rng.randf_range(1.0 - irr, 1.0 + irr), r * (_cols - 1) + c)
	for r in _rows - 1:
		for c in _cols:
			_add_con(r * _cols + c, (r + 1) * _cols + c, rest_y, k_struct,
				y_weft * rng.randf_range(1.0 - irr, 1.0 + irr), _s_weft + r * _cols + c)
	# One diagonal per cell on a checkerboard rather than both: half the shear cost for
	# a sheet that still refuses to collapse in-plane, and the alternating direction
	# keeps the bias from reading as a consistent lean across the whole fabric.
	for r in _rows - 1:
		for c in _cols - 1:
			var a := r * _cols + c
			var b := (r + 1) * _cols + c + 1
			if (r + c) % 2 == 1:
				a = r * _cols + c + 1
				b = (r + 1) * _cols + c
			_add_con(a, b, rest_d, k_shear, NO_YIELD, _s_shear + r * (_cols - 1) + c)
	for r in _rows:
		var c2 := 0
		while c2 <= _cols - 3:
			_add_con(r * _cols + c2, r * _cols + c2 + 2, rest_x * 2.0, k_bend, NO_YIELD,
				_s_bh + r * _nbh + int(c2 / 2.0))
			c2 += 2
	var r2 := 0
	while r2 <= _rows - 3:
		for c in _cols:
			_add_con(r2 * _cols + c, (r2 + 2) * _cols + c, rest_y * 2.0, k_bend, NO_YIELD,
				_s_bv + int(r2 / 2.0) * _cols + c)
		r2 += 2

	# Settle the gather into folds BEFORE the first frame. Pure relaxation, no gravity,
	# no velocity: the compression from the pins has to go somewhere and the ripple tells
	# it where. Previous positions are re-synced afterward so this costs no impulse.
	for _i in 12:
		_relax(0.0, false)
	for i in _n:
		_qx[i] = _px[i]; _qy[i] = _py[i]; _qz[i] = _pz[i]

	var total := _cn
	_iters = clampi(rng.randi_range(3, 9), 2, maxi(2, int(float(SOLVE_BUDGET) / float(maxi(1, total)))))

	# --- wind --------------------------------------------------------------------
	_flow = Flow2D.new(rng.randi(), rng.randf_range(1.4, 3.2), rng.randf_range(0.04, 0.14))
	_light = Lighting.new(rng, 2)
	_gw = rng.randi_range(7, 10)
	_gh = rng.randi_range(5, 8)
	_gx0 = -wide * 0.75; _gx1 = wide * 0.75
	_gy0 = _line_y - 0.15; _gy1 = _line_y + tall + 0.45
	_wxa.resize(_gw * _gh); _wya.resize(_gw * _gh); _wza.resize(_gw * _gh)
	_hota.resize(HOT_W * HOT_H)
	var wdir := rng.randf_range(-0.5, 0.5) + (0.0 if rng.randf() < 0.5 else PI)
	_wind_dx = cos(wdir)
	_wind_dy = sin(wdir) * 0.35
	_wind_base = rng.randf_range(0.05, 0.16)
	_wind_gain = rng.randf_range(0.30, 0.85)
	_turb_gain = rng.randf_range(8.0, 20.0)
	_sway = rng.randf_range(0.004, 0.018)

	# The gust vocabulary, rolled once. A beat picks the next entry by counter, never by
	# a live draw - the export bake's beat stream is not the live analyzer's.
	var gust_n := rng.randi_range(4, 9)
	for i in gust_n:
		_gust_specs.append({
			"y": rng.randf_range(_line_y + tall * 0.15, _line_y + tall * 1.05),
			"r": rng.randf_range(0.10, 0.30),
			"s": rng.randf_range(0.6, 2.4),
			"v": rng.randf_range(0.7, 2.2),
			"dir": 1.0 if rng.randf() < 0.72 else -1.0,
		})

	# --- look --------------------------------------------------------------------
	_persp = rng.randf_range(0.16, 0.48)
	_opacity = rng.randf_range(0.25, 0.80)
	_trans = rng.randf_range(0.15, 0.60)
	_amb = rng.randf_range(0.18, 0.38)
	_diff = rng.randf_range(0.55, 0.95)
	_sheen = rng.randf_range(0.10, 0.45)
	_sheen_gain = rng.randf_range(0.5, 1.4)
	_shine = rng.randf_range(10.0, 40.0)
	_hot = rng.randf_range(0.15, 0.45)
	_hue = sch.hue
	_hue_back = sch.opposed(sch.hue, 0.14)
	_hue_fold = rng.randf_range(0.01, 0.07)
	_sat = sch.sat
	_val = 0.70 + 0.35 * sch.val
	var laz := rng.randf_range(-1.0, 1.0)
	var lel := rng.randf_range(0.15, 0.75)
	var lv := Vector3(sin(laz) * cos(lel), -sin(lel), cos(laz) * cos(lel)).normalized()
	_lx = lv.x; _ly = lv.y; _lz = lv.z
	var hv := (lv + Vector3(0.0, 0.0, 1.0)).normalized()
	_hx = hv.x; _hy = hv.y; _hz = hv.z

	_high_key = rng.randf() < 0.45
	_ground_hue = sch.accent
	_ground_sat = rng.randf_range(0.04, 0.14)
	_ground_val = rng.randf_range(0.86, 0.96)
	_line_val = rng.randf_range(0.10, 0.30)
	_floor_on = rng.randf() < 0.5
	_floor_y = rng.randf_range(0.40, 0.52)
	_friction = rng.randf_range(0.35, 0.8)

	if not _high_key:
		add_layer("bed", rng, {"hue": sch.hue, "sat": sch.sat * 0.7,
			"val": 0.18 + 0.14 * sch.val, "pools": 3, "z": "back"})
	if rng.randf() < 0.55:
		add_layer("dust", rng, {"count": rng.randi_range(40, 90)})

	_spos.resize(_n)
	_ncol.resize(_n)
	_rebuild_cells()

	return {
		"cols": _cols, "rows": _rows, "nodes": _n, "constraints": total,
		"iters": _iters, "mood": sch.name, "pins": pattern, "pin_step": step,
		"pin_count": _pin.size(), "width": wide, "height": tall,
		"slack": slack, "gather": gather, "folds": folds, "fold_amp": fold_amp,
		"mass": _mass, "gravity": _grav, "damping": _damp,
		"stiff_struct": k_struct, "stiff_shear": k_shear, "stiff_bend": k_bend,
		"tearing": _tearing, "yield_warp": y_warp, "weave_bias": weave_bias,
		"weave_irregularity": irr, "tear_delay": _tear_delay, "damage_rate": _dmg_rate,
		"max_stretch": _max_stretch,
		"tip_gain": _tip, "cross_gain": _cross, "weak_decay": _weak_decay,
		"breaks_per_tick": _break_budget,
		"wind_base": _wind_base, "wind_gain": _wind_gain, "wind_dir": wdir,
		"turbulence_gain": _turb_gain, "gusts": gust_n, "sway": _sway,
		"opacity": _opacity, "translucency": _trans, "sheen": _sheen,
		"shininess": _shine, "perspective": _persp,
		"high_key": _high_key, "floor": _floor_on,
		"hue": _hue, "hue_back": _hue_back, "sat": _sat, "val": _val,
	}


func _add_pin(node: int, gather: float, wide: float) -> void:
	if node < 0 or node >= _n:
		return
	# Pull the anchor in toward the centre. The fabric is then wider than the span it
	# hangs across, which is the whole reason a curtain has standing folds.
	var ax: float = _px[node] * (1.0 - gather)
	_pin.append(node)
	_pax.append(ax)
	_pay.append(_py[node])
	_paz.append(_pz[node])
	_px[node] = ax
	_qx[node] = ax
	_w[node] = 0.0


func _add_con(a: int, b: int, rest: float, stiff: float, yield_strain: float, slot: int) -> void:
	var k := _cn
	_ca[k] = a; _cb[k] = b
	_crest[k] = rest
	_cstiff[k] = stiff
	_cyield[k] = yield_strain
	_cdmg[k] = 0.0
	_cslot[k] = slot
	if slot >= 0:
		_cof[slot] = k
	_cn = k + 1


# ---------------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------------
func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	# A hanging sheet is square-on; the camera stays gentle so the plane never reads as
	# a tumbling card (and PLANE_BAG already keeps the shot centred).
	drift_view(f, 0.02, 0.03)
	_light.update(f, delta)
	update_layers(f, delta)
	_ch = chroma_hue()

	# Wind speed follows MOVEMENT, not loudness. f.movement is a section-change score and
	# sits low through a steady passage, so a small energy term keeps a breeze alive; the
	# envelope itself is flared (builds quickly, dies slowly) inside the sim step.
	_wind_target = _wind_base + _wind_gain * clampf(f.movement * 2.2 + f.energy * 0.5, 0.0, 1.4)
	# f.flux is typically 0.01-0.05, hence the large gain for a modest 1.0-1.7x result.
	_turb_target = 1.0 + _turb_gain * f.flux

	# Beat rising edge releases a gust. Pre-rolled vocabulary, counter-indexed: no rng
	# on an audio-conditioned event, ever.
	if f.beat > 0.5 and _prev_beat <= 0.5 and _gusts.size() < MAX_GUSTS and not _gust_specs.is_empty():
		var sp: Dictionary = _gust_specs[_gust_n % _gust_specs.size()]
		_gust_n += 1
		var dir: float = sp["dir"]
		var entry := _gx0 - 0.12
		if dir < 0.0:
			entry = _gx1 + 0.12
		_gusts.append({
			"x": entry,
			"y": float(sp["y"]),
			"r": float(sp["r"]),
			"s": float(sp["s"]) * (0.5 + 0.9 * f.beat),
			"v": float(sp["v"]),
			"dir": dir,
		})
	_prev_beat = f.beat

	for _i in _sim.ticks(delta):
		_step(_sim.dt)
	queue_redraw()


func _step(dt: float) -> void:
	_flow.advance(dt)
	_wind = Nonlinear.flare(_wind, _wind_target, dt, 1.6, 0.5)
	_turb = lerpf(_turb, _turb_target, 1.0 - exp(-1.2 * dt))
	_sample_wind()
	_move_gusts(dt)
	_integrate(dt)
	var measure := _tearing and _sim.elapsed() > _tear_delay
	if measure:
		_cand.clear()
	for it in _iters:
		_relax(dt, measure and it == 0)
	_clamp_stretch()
	_sway_pins()
	if measure:
		_break_seams()
		for i in _weak.size():
			_weak[i] *= _weak_decay
	if _cells_dirty:
		_rebuild_cells()


## Probe the curl field on a coarse lattice once per tick. One probe per NODE would be four
## 3D noise lookups per node - more than the whole solver costs - and wind has a far coarser
## spatial scale than the mesh anyway, so nothing is lost by interpolating.
func _sample_wind() -> void:
	var t := _turb
	var sx := (_gx1 - _gx0) / float(maxi(1, _gw - 1))
	var sy := (_gy1 - _gy0) / float(maxi(1, _gh - 1))
	for j in _gh:
		var y := _gy0 + sy * float(j)
		for i in _gw:
			var x := _gx0 + sx * float(i)
			var v := _flow.at(Vector2(x, y) * t)
			var k := j * _gw + i
			_wxa[k] = clampf(v.x, -3.0, 3.0) + _wind_dx
			_wya[k] = clampf(v.y, -3.0, 3.0) + _wind_dy
			# A second probe, offset and at a different scale, gives the out-of-plane
			# push its own life instead of making z a linear function of the xy swirl.
			var v2 := _flow.at(Vector2(x * 0.7 + 3.7, y * 0.7 - 2.9) * t)
			_wza[k] = clampf(v2.x, -3.0, 3.0)


func _move_gusts(dt: float) -> void:
	var i := _gusts.size() - 1
	while i >= 0:
		var g: Dictionary = _gusts[i]
		var dir: float = g["dir"]
		var vx: float = g["v"]
		g["x"] = float(g["x"]) + dir * vx * _wind * 6.0 * dt + dir * vx * 0.25 * dt
		var gx: float = g["x"]
		if gx < _gx0 - 0.4 or gx > _gx1 + 0.4:
			_gusts.remove_at(i)
		i -= 1


func _integrate(dt: float) -> void:
	var dt2 := dt * dt
	var gy := _grav
	var wind := _wind / _mass
	var inv_w := float(maxi(1, _gw - 1)) / maxf(0.0001, _gx1 - _gx0)
	var inv_h := float(maxi(1, _gh - 1)) / maxf(0.0001, _gy1 - _gy0)
	# Gusts are flattened out of their dictionaries BEFORE the node loop. Read in place
	# they would be five dictionary lookups per node per gust - thousands of them a tick,
	# which measurably outweighs the arithmetic they feed.
	var ng := _gusts.size()
	var gx_a := PackedFloat32Array()
	var gy_a := PackedFloat32Array()
	var gk_a := PackedFloat32Array()     # 1 / (2 r^2)
	var gs_a := PackedFloat32Array()
	var gd_a := PackedFloat32Array()
	for gi in ng:
		var g: Dictionary = _gusts[gi]
		var gr: float = g["r"]
		gx_a.append(float(g["x"]))
		gy_a.append(float(g["y"]))
		gk_a.append(1.0 / maxf(0.0001, 2.0 * gr * gr))
		gs_a.append(float(g["s"]) / _mass)
		gd_a.append(float(g["dir"]))
	for i in _n:
		if _w[i] <= 0.0:
			continue
		var x := _px[i]
		var y := _py[i]
		var z := _pz[i]
		# Bilinear wind lookup on the coarse lattice.
		var fx := clampf((x - _gx0) * inv_w, 0.0, float(_gw - 1))
		var fy := clampf((y - _gy0) * inv_h, 0.0, float(_gh - 1))
		var i0 := int(fx)
		var j0 := int(fy)
		var i1 := mini(i0 + 1, _gw - 1)
		var j1 := mini(j0 + 1, _gh - 1)
		var tx := fx - float(i0)
		var ty := fy - float(j0)
		var k00 := j0 * _gw + i0
		var k10 := j0 * _gw + i1
		var k01 := j1 * _gw + i0
		var k11 := j1 * _gw + i1
		var m00 := (1.0 - tx) * (1.0 - ty)
		var m10 := tx * (1.0 - ty)
		var m01 := (1.0 - tx) * ty
		var m11 := tx * ty
		var ax := (_wxa[k00] * m00 + _wxa[k10] * m10 + _wxa[k01] * m01 + _wxa[k11] * m11) * wind
		var ay := (_wya[k00] * m00 + _wya[k10] * m10 + _wya[k01] * m01 + _wya[k11] * m11) * wind + gy
		var az := (_wza[k00] * m00 + _wza[k10] * m10 + _wza[k01] * m01 + _wza[k11] * m11) * wind
		# Gusts: a travelling pressure blob, mostly pushing out of the plane so the beat
		# is SEEN as a bulge crossing the fabric rather than as a sideways shove.
		for gi in ng:
			var ddx := x - gx_a[gi]
			var ddy := y - gy_a[gi]
			var q := (ddx * ddx + ddy * ddy) * gk_a[gi]
			if q > 6.0:
				continue
			var gw2 := exp(-q) * gs_a[gi]
			az += gw2
			ax += gw2 * 0.35 * gd_a[gi]
		# Verlet.
		var nx := x + (x - _qx[i]) * _damp + ax * dt2
		var ny := y + (y - _qy[i]) * _damp + ay * dt2
		var nz := z + (z - _qz[i]) * _damp + az * dt2
		_qx[i] = x; _qy[i] = y; _qz[i] = z
		# A shed strip is under no constraint at all and would otherwise integrate toward
		# infinity; the clamp is a sanity rail, not a boundary anything reaches in frame.
		_px[i] = clampf(nx, -8.0, 8.0)
		_py[i] = clampf(ny, -8.0, 8.0)
		_pz[i] = clampf(nz, -8.0, 8.0)
		if _floor_on and _py[i] > _floor_y:
			_py[i] = _floor_y
			_qx[i] = lerpf(_qx[i], _px[i], _friction)
			_qy[i] = _floor_y


## One Gauss-Seidel relaxation pass. When [param measure] is true it also accumulates DAMAGE,
## which is why the strain is read here: the distance is already computed, and the first pass
## of a tick sees the raw stretch left by integration - the honest stress signal.
func _relax(dt: float, measure: bool) -> void:
	for k in _cn:
		var a := _ca[k]
		var b := _cb[k]
		var wa := _w[a]
		var wb := _w[b]
		var ws := wa + wb
		if ws <= 0.0:
			continue
		var dx := _px[b] - _px[a]
		var dy := _py[b] - _py[a]
		var dz := _pz[b] - _pz[a]
		var d2 := dx * dx + dy * dy + dz * dz
		if d2 < 1e-12:
			continue
		var d := sqrt(d2)
		var rest := _crest[k]
		if measure:
			var ys := _cyield[k]
			if ys < 900.0:
				var slot := _cslot[k]
				var y_eff := ys * (1.0 - clampf(_weak[slot], 0.0, 0.92))
				var strain := (d - rest) / rest
				if strain > y_eff:
					_cdmg[k] += (strain - y_eff) * dt * _dmg_rate
					if _cdmg[k] >= 1.0 and _cand.size() < MAX_CAND:
						_cand.append(slot)
		var s := (d - rest) / d * _cstiff[k] / ws
		var mx := dx * s
		var my := dy * s
		var mz := dz * s
		_px[a] += mx * wa; _py[a] += my * wa; _pz[a] += mz * wa
		_px[b] -= mx * wb; _py[b] -= my * wb; _pz[b] -= mz * wb


## The over-stretch projection, run once after the relaxation passes. A few Gauss-Seidel
## iterations cannot equilibrate a whole hanging sheet, so without this a corner or a single
## peg carrying several hundred nodes stretches its neighbourhood like taffy and the fabric
## reads as rubber. This is the standard cloth answer: leave the soft constraint soft, but hard
## project anything past `max_stretch` back to that limit.
##
## Cheap, despite being a whole extra pass: the squared-length test rejects the overwhelming
## majority of constraints before the sqrt, and writes nothing for them.
func _clamp_stretch() -> void:
	for k in _cn:
		var a := _ca[k]
		var b := _cb[k]
		var wa := _w[a]
		var wb := _w[b]
		var ws := wa + wb
		if ws <= 0.0:
			continue
		var dx := _px[b] - _px[a]
		var dy := _py[b] - _py[a]
		var dz := _pz[b] - _pz[a]
		var d2 := dx * dx + dy * dy + dz * dz
		var lim := _crest[k] * _max_stretch
		if d2 <= lim * lim:
			continue
		var d := sqrt(d2)
		var s := (d - lim) / (d * ws)
		var mx := dx * s
		var my := dy * s
		var mz := dz * s
		_px[a] += mx * wa; _py[a] += my * wa; _pz[a] += mz * wa
		_px[b] -= mx * wb; _py[b] -= my * wb; _pz[b] -= mz * wb


## Pins hold, but a peg on a line has a little give: the anchors breathe with the wind so the
## whole sheet lifts as a body rather than pivoting around perfectly rigid points.
func _sway_pins() -> void:
	var t := _sim.elapsed()
	var s := _sway * (0.3 + 1.6 * _wind)
	for i in _pin.size():
		var node := _pin[i]
		var ph := float(i) * 0.7
		_px[node] = _pax[i] + s * sin(t * 1.3 + ph) * _wind_dx
		_py[node] = _pay[i] + s * 0.4 * sin(t * 1.9 + ph)
		_pz[node] = _paz[i] + s * 0.8 * sin(t * 1.1 + ph)
		_qx[node] = _px[node]; _qy[node] = _py[node]; _qz[node] = _pz[node]


# ---------------------------------------------------------------------------------
# The tear
# ---------------------------------------------------------------------------------
func _break_seams() -> void:
	var budget := _break_budget
	while budget > 0 and not _cand.is_empty():
		var best := -1
		var best_d := 1.0
		for ci in _cand.size():
			var slot := _cand[ci]
			var k := _cof[slot]
			if k < 0:
				continue
			if _cdmg[k] >= best_d:
				best_d = _cdmg[k]
				best = slot
		if best < 0:
			break
		_tear_slot(best)
		budget -= 1
	_cand.clear()


## Remove one structural edge, everything that crossed it, and raise the crack-tip weakening
## on its neighbours in the weave. This is the whole propagation rule.
func _tear_slot(slot: int) -> void:
	_remove_slot(slot)
	_cells_dirty = true
	var tip := _tip
	var cross := _tip * _cross
	if slot < _s_weft:
		# A warp (horizontal) edge (r, c) joining node (r,c) to (r,c+1).
		var r := int(float(slot) / float(_cols - 1))
		var c := slot - r * (_cols - 1)
		# The run continues UP and DOWN the same column of the weave.
		if r > 0:
			_bump((r - 1) * (_cols - 1) + c, tip)
		if r < _rows - 1:
			_bump((r + 1) * (_cols - 1) + c, tip)
		if c > 0:
			_bump(r * (_cols - 1) + c - 1, cross)
		if c < _cols - 2:
			_bump(r * (_cols - 1) + c + 1, cross)
		# The crossing threads at the tip, so a crack can turn instead of only running straight.
		var half := cross * 0.5
		if r > 0:
			_bump(_s_weft + (r - 1) * _cols + c, half)
			_bump(_s_weft + (r - 1) * _cols + c + 1, half)
		if r < _rows - 1:
			_bump(_s_weft + r * _cols + c, half)
			_bump(_s_weft + r * _cols + c + 1, half)
		# The diagonals of the cells this edge bordered, and the bend that spanned it,
		# would otherwise bridge the rip and it would never open.
		if r < _rows - 1:
			_remove_slot(_s_shear + r * (_cols - 1) + c)
		if r > 0:
			_remove_slot(_s_shear + (r - 1) * (_cols - 1) + c)
		var cc := c if c % 2 == 0 else c - 1
		if cc >= 0 and cc <= _cols - 3:
			_remove_slot(_s_bh + r * _nbh + int(cc / 2.0))
	else:
		# A weft (vertical) edge (r, c) joining node (r,c) to (r+1,c).
		var e := slot - _s_weft
		var r := int(float(e) / float(_cols))
		var c := e - r * _cols
		# The run continues LEFT and RIGHT along the same row of the weave.
		if c > 0:
			_bump(_s_weft + r * _cols + c - 1, tip)
		if c < _cols - 1:
			_bump(_s_weft + r * _cols + c + 1, tip)
		if r > 0:
			_bump(_s_weft + (r - 1) * _cols + c, cross)
		if r < _rows - 2:
			_bump(_s_weft + (r + 1) * _cols + c, cross)
		var half2 := cross * 0.5
		if c > 0:
			_bump(r * (_cols - 1) + c - 1, half2)
			_bump((r + 1) * (_cols - 1) + c - 1, half2)
		if c < _cols - 1:
			_bump(r * (_cols - 1) + c, half2)
			_bump((r + 1) * (_cols - 1) + c, half2)
		if c < _cols - 1:
			_remove_slot(_s_shear + r * (_cols - 1) + c)
		if c > 0:
			_remove_slot(_s_shear + r * (_cols - 1) + c - 1)
		var rr := r if r % 2 == 0 else r - 1
		if rr >= 0 and rr <= _rows - 3:
			_remove_slot(_s_bv + int(rr / 2.0) * _cols + c)


func _bump(slot: int, amt: float) -> void:
	if slot < 0 or slot >= _s_shear:
		return
	_weak[slot] = minf(1.0, _weak[slot] + amt)


func _remove_slot(slot: int) -> void:
	if slot < 0 or slot >= _cof.size():
		return
	var k := _cof[slot]
	if k < 0:
		return
	# Swap-remove into the live prefix: O(1), no allocation, and the slot map is repaired
	# for the one constraint that moved. Solve order shifts slightly and deterministically.
	_cof[slot] = -1
	var last := _cn - 1
	if k != last:
		_ca[k] = _ca[last]; _cb[k] = _cb[last]
		_crest[k] = _crest[last]; _cstiff[k] = _cstiff[last]
		_cyield[k] = _cyield[last]; _cdmg[k] = _cdmg[last]
		_cslot[k] = _cslot[last]
		var s2 := _cslot[k]
		if s2 >= 0:
			_cof[s2] = k
	_cn = last


## The drawable quad list, rebuilt from the surviving constraints. A cell is painted only
## while ALL FOUR of its structural edges live, so a run down the weave opens as a real slit
## instead of the fabric staying visually welded across a seam that is physically gone.
func _rebuild_cells() -> void:
	_cells_dirty = false
	_cells.clear()
	for r in _rows - 1:
		for c in _cols - 1:
			if _cof[r * (_cols - 1) + c] < 0:
				continue
			if _cof[(r + 1) * (_cols - 1) + c] < 0:
				continue
			if _cof[_s_weft + r * _cols + c] < 0:
				continue
			if _cof[_s_weft + r * _cols + c + 1] < 0:
				continue
			_cells.append(r * (_cols - 1) + c)


# ---------------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------------
func _draw() -> void:
	begin_draw()
	if _high_key:
		paint_ground(_ground_hue, _ground_sat, _ground_val, 0.35 * _ch.y, _ch.x)
	draw_layers("back")

	var u := unit()
	var glow := _light.glow()
	var sheen := _sheen + _sheen_gain * _f.treble
	var hue_shift := _light.hue_shift() * 0.5

	# The hotspot field, once per frame on a coarse lattice (see HOT_W / HOT_H).
	var hsx := (_gx1 - _gx0) / float(HOT_W - 1)
	var hsy := (_gy1 - _gy0) / float(HOT_H - 1)
	for j in HOT_H:
		for i in HOT_W:
			_hota[j * HOT_W + i] = _light.at(Vector2(_gx0 + hsx * float(i), _gy0 + hsy * float(j)))
	var hinv_w := float(HOT_W - 1) / maxf(0.0001, _gx1 - _gx0)
	var hinv_h := float(HOT_H - 1) / maxf(0.0001, _gy1 - _gy0)

	# Per-NODE normals and colours: shading a fold flat-faceted per cell shows the mesh,
	# and the mesh is not the subject. One cross product per node buys smooth folds.
	for r in _rows:
		var rm := maxi(r - 1, 0) * _cols
		var rp := mini(r + 1, _rows - 1) * _cols
		var row := r * _cols
		for c in _cols:
			var i := row + c
			var ia := row + mini(c + 1, _cols - 1)
			var ib := row + maxi(c - 1, 0)
			var ja := rp + c
			var jb := rm + c
			var ux := _px[ia] - _px[ib]
			var uy := _py[ia] - _py[ib]
			var uz := _pz[ia] - _pz[ib]
			var vx := _px[ja] - _px[jb]
			var vy := _py[ja] - _py[jb]
			var vz := _pz[ja] - _pz[jb]
			var nx := uy * vz - uz * vy
			var ny := uz * vx - ux * vz
			var nz := ux * vy - uy * vx
			var nl := sqrt(nx * nx + ny * ny + nz * nz)
			if nl < 1e-9:
				nx = 0.0; ny = 0.0; nz = 1.0
			else:
				nx /= nl; ny /= nl; nz /= nl
			# Face the normal toward the viewer: a thin sheet has no meaningful outside,
			# but WHICH side we are looking at still changes its colour.
			var back := nz < 0.0
			if back:
				nx = -nx; ny = -ny; nz = -nz
			var lam := nx * _lx + ny * _ly + nz * _lz
			# Light arriving through the BACK of the sheet is what makes fabric glow: the
			# transmitted term peaks exactly where the diffuse term is zero.
			var tr := maxf(0.0, -lam) * _trans
			lam = maxf(0.0, lam)
			var sp := nx * _hx + ny * _hy + nz * _hz
			var spec := 0.0
			if sp > 0.0:
				spec = pow(sp, _shine) * sheen
			var hfx := clampf((_px[i] - _gx0) * hinv_w, 0.0, float(HOT_W - 1))
			var hfy := clampf((_py[i] - _gy0) * hinv_h, 0.0, float(HOT_H - 1))
			var hi0 := int(hfx)
			var hj0 := int(hfy)
			var hi1 := mini(hi0 + 1, HOT_W - 1)
			var hj1 := mini(hj0 + 1, HOT_H - 1)
			var htx := hfx - float(hi0)
			var hty := hfy - float(hj0)
			var hot := (_hota[hj0 * HOT_W + hi0] * (1.0 - htx) * (1.0 - hty)
				+ _hota[hj0 * HOT_W + hi1] * htx * (1.0 - hty)
				+ _hota[hj1 * HOT_W + hi0] * (1.0 - htx) * hty
				+ _hota[hj1 * HOT_W + hi1] * htx * hty) * _hot
			var facing := absf(nz)
			var val := _val * (_amb + _diff * lam + tr) + spec + hot * 0.5 + glow * 0.10
			var h := _hue + _hue_fold * (1.0 - facing) + hue_shift
			if back:
				h = _hue_back
				val *= 0.78
			var dh: float = _ch.x - h
			h = fposmod(h + (dh - round(dh)) * 0.35 * _ch.y, 1.0)
			var sat := clampf(_sat * (1.0 - 0.45 * spec) - 0.10 * hot, 0.0, 1.0)
			# Grazing fabric is optically thicker - a fold seen edge-on hides more of what
			# is behind it. Cheap linear stand-in for the real 1/cos thickness law.
			var alpha := clampf(_opacity * (0.72 + 0.55 * (1.0 - facing)), 0.04, 0.96)
			_ncol[i] = Color.from_hsv(h, sat, clampf(val, 0.0, 1.0), alpha)
			var sc := (1.0 + _pz[i] * _persp) * u
			_spos[i] = Vector2(_px[i] * sc, _py[i] * sc)

	var tb := TriBatch.new()
	if _line_on:
		var ly := _line_y * u
		var lw := size.x * 0.75 / maxf(0.001, view.zoom_actual())
		tb.line(Vector2(-lw, ly), Vector2(lw, ly),
			Color.from_hsv(fposmod(_hue + 0.5, 1.0), 0.25, _line_val, 0.85), maxf(1.0, u * 0.004))
	for i in _pin.size():
		var node := _pin[i]
		var p := _spos[node]
		var rr := maxf(1.5, u * 0.006)
		tb.quad(p + Vector2(-rr, -rr), p + Vector2(rr, -rr), p + Vector2(rr, rr), p + Vector2(-rr, rr),
			Color.from_hsv(fposmod(_hue + 0.5, 1.0), 0.3, _line_val * 1.6, 0.9))

	# Painter's order over the cells. TriBatch.painter_sort wants an Array of Dictionary,
	# and building several hundred dictionaries a frame costs more than the sort saves, so
	# the same packed-key trick is applied directly to the cell list: quantized depth in the
	# high bits, cell index in the low 20, sorted natively. Far (small z) first.
	var m := _cells.size()
	if m > 0:
		if _keys.size() != m:
			_keys.resize(m)
		var w1 := _cols - 1
		for ci in m:
			var cell := _cells[ci]
			var r := int(float(cell) / float(w1))
			var c := cell - r * w1
			var i0 := r * _cols + c
			var i1 := i0 + 1
			var i2 := i0 + _cols
			var i3 := i2 + 1
			var zq := int(clampf((_pz[i0] + _pz[i1] + _pz[i2] + _pz[i3]) * 0.25, -60.0, 60.0) * 8192.0)
			_keys[ci] = (zq << 20) | ci
		_keys.sort()
		for ki in m:
			var cell2 := _cells[int(_keys[ki] & 0xFFFFF)]
			var r2 := int(float(cell2) / float(w1))
			var c2 := cell2 - r2 * w1
			var a := r2 * _cols + c2
			var b := a + 1
			var d := a + _cols
			var e := d + 1
			if SUBDIV <= 1:
				tb.quad_colored(
					PackedVector2Array([_spos[a], _spos[b], _spos[e], _spos[d]]),
					PackedColorArray([_ncol[a], _ncol[b], _ncol[e], _ncol[d]]))
			else:
				_emit_smooth(tb, r2, c2)
	tb.flush(self)
	draw_layers("front")


## One cell, emitted as a SUBDIV x SUBDIV patch on a smooth interpolant.
##
## The sheet was drawn as one flat quad per simulated cell, which is why it read as
## "block-like shapes in the waves": the shading was already smooth (per-node normals, a
## cross product each), but the GEOMETRY was piecewise flat, so every fold had visible
## facets and the silhouette was a polygon with one corner per node.
##
## Raising the mesh instead is not available. The solver is `constraints x iterations` per
## tick and measured 5.4-7.9 ms per update on the main thread at the CURRENT size, so
## doubling the nodes would double that and blow the frame - and [constant SOLVE_BUDGET]
## would claw the iterations back down, leaving the cloth softer than before.
##
## So the smoothing goes where it is free: Catmull-Rom across the existing grid at draw
## time. It costs triangles, which are batched into one call anyway, and no simulation at
## all. Catmull-Rom rather than bilinear because bilinear subdivision of a flat quad is
## still that flat quad - it adds vertices without adding curvature. The spline reads the
## neighbouring row and column, so a fold genuinely rounds.
func _emit_smooth(tb: TriBatch, r: int, c: int) -> void:
	var n := SUBDIV
	for sj in n:
		var v0 := float(sj) / float(n)
		var v1 := float(sj + 1) / float(n)
		for si in n:
			var u0 := float(si) / float(n)
			var u1 := float(si + 1) / float(n)
			tb.quad_colored(
				PackedVector2Array([
					_surf(r, c, u0, v0), _surf(r, c, u1, v0),
					_surf(r, c, u1, v1), _surf(r, c, u0, v1)]),
				PackedColorArray([
					_shade(r, c, u0, v0), _shade(r, c, u1, v0),
					_shade(r, c, u1, v1), _shade(r, c, u0, v1)]))


## The screen position at (u, v) inside cell (r, c), Catmull-Rom in both axes. Indices are
## CLAMPED at the sheet's border rather than wrapped: a wrap would join the two edges of a
## hanging sheet and drag its corners toward each other.
func _surf(r: int, c: int, u: float, v: float) -> Vector2:
	var rows := PackedVector2Array()
	rows.resize(4)
	for k in 4:
		var rr := clampi(r - 1 + k, 0, _rows - 1) * _cols
		rows[k] = _cr(
			_spos[rr + clampi(c - 1, 0, _cols - 1)], _spos[rr + clampi(c, 0, _cols - 1)],
			_spos[rr + clampi(c + 1, 0, _cols - 1)], _spos[rr + clampi(c + 2, 0, _cols - 1)], u)
	return _cr(rows[0], rows[1], rows[2], rows[3], v)


## Colour is only bilinear. The nodes already carry smoothly-varying normals, so the
## shading has no facets to remove - and a spline through colours can overshoot outside
## [0,1] and produce a bright rim on a fold, which is worse than the faceting it fixes.
func _shade(r: int, c: int, u: float, v: float) -> Color:
	var i0 := r * _cols + c
	var i1 := i0 + 1
	var i2 := i0 + _cols
	var i3 := i2 + 1
	return _ncol[i0].lerp(_ncol[i1], u).lerp(_ncol[i2].lerp(_ncol[i3], u), v)


static func _cr(p0: Vector2, p1: Vector2, p2: Vector2, p3: Vector2, t: float) -> Vector2:
	var t2 := t * t
	var t3 := t2 * t
	return 0.5 * ((2.0 * p1) + (-p0 + p2) * t
		+ (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
		+ (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
