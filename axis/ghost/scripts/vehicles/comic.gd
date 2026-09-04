extends Vehicle
class_name ComicVehicle

## ComicVehicle - the show as an open comic book, flown over by a real perspective camera.
##
## The same scenes, the same Director, the same cutting. What changes is where a scene
## LANDS: instead of replacing the picture, each cut fills the next PANEL of the spread, and
## when the spread is full the next cut turns the leaf. So `Scene hold` and `Flourishes`
## keep their exact meanings - a burst of quick cuts becomes a run of small panels filling in
## fast, which is what a comic does with a fight.
##
## TWO PAGES AT ONCE, because that is what a comic is. The first cut of this drew ONE
## portrait page and it cost three things at the same time. A portrait sheet inside a
## landscape frame cannot cover it, so there was always a wedge of desk in shot unless the
## camera pushed in until the page stopped reading as a page. A pan had one page's width of
## content to travel before it ran onto the trim edge. And a page turn had no hinge that
## meant anything, because a leaf hinges on a SPINE and one page does not have one. See
## [ComicSpread]: the spread is 2 pages wide by ~1.5 tall, which is 1.33 against the frame's
## 1.78, so the sheet can cover the whole frame at a shot that still shows several panels.
##
## ONE LIVE PANEL, THE REST HELD. The Director runs exactly one scene (two across a
## transition) and this keeps that: the newest panel is live in its own SubViewport, and
## every panel behind it is a STOPPED render target - update mode DISABLED, its scene
## freed, its last frame still sitting in VRAM. Measured in tests/vehicle_probe.gd: the
## texture survives both the stop and the free, bit-exact, so a held panel costs nothing
## at all. That is the only reason this can ship. Six live scenes is not a thing that
## runs - the stage governor already spends its budget on one - and the alternative,
## capturing each panel with get_image(), is the synchronous readback stall that cost
## Masking its frame rate.
##
## It is also simply correct. A comic panel IS a held moment; the page is a sequence of
## them. The frame is never static anyway, because the camera and the page are always
## moving.
##
## THE SPREAD IS REALLY IN 3D. Not a sheared 2D plane faking depth - two quads placed in a
## world and projected through a [Lens3D], free to rotate on X, Y and Z at once. Panels
## are drawn as SUBDIVIDED grids of textured triangles whose vertices are each projected
## individually, because a two-triangle quad is affinely textured and warps: measured at
## a hard yaw, the seam of a two-tone test texture landed 28.5 px away from where
## perspective puts it, on a 512 px frame.

## How many panel slots exist. Two POOLS of [constant ComicSpread.MAX_PANELS], alternating
## per spread, because a page turn shows both spreads at once - reusing one pool would pull
## the outgoing spread's pictures out from under it mid-turn. All FOUR half-pages carry
## textures during a turn (the old left page under the leaf, the old right page on the
## leaf's front, the new left page on its back, the new right page revealed beneath), which
## is why a pool is a whole spread and not a page.
const POOL := ComicSpread.MAX_PANELS

## Panel render-target size, as a fraction of the stage's shorter side. A panel is sampled
## at close to 1:1 in the tight reading shot (which frames roughly one panel), so this is
## about right and not a guess; it is clamped below so a small window still gets a legible
## panel and above so a 4K export does not allocate enormous targets.
const PANEL_SCALE := 1.0
const PANEL_MIN := 224
const PANEL_MAX := 1400
## What an unopened slot is sized to. There are 24 of them now and a spread opens at most
## half; allocating every one at PANEL_SCALE up front - which is what the first cut did -
## reserved a full-size render target for a panel no page had. Sixty-four squared is
## 16 KiB, and _open_slot grows the slot to its real size before anything draws it.
const SLOT_IDLE := 64

## Subdivision of a panel's textured grid, chosen so each cell is about this many pixels on
## screen, between these bounds.
##
## ADAPTIVE, because a fixed number cannot be right. Inside one cell the texture mapping is
## affine while the true mapping is projective, so the error scales with how big the cell is
## ON SCREEN and with how hard the panel is raked. A fixed 8 was measured sufficient at one
## yaw on a 512 px frame, and this vehicle now puts a single panel across a whole 1280 px
## frame at a much harder rake. Sizing the CELL rather than the count bounds the error
## directly, at any framing.
##
## NOT the fix for the banding reported on fractal_zoom - that was checked, and going from 8
## to an adaptive 6-40 changed those frames not at all. Those bands are the scene's own
## float32 precision blocks, which the close framing here simply magnifies. This is a
## correctness change made because the reasoning behind the old constant no longer held, not
## a repair of anything observed.
const CELL_PX := 56.0
const GRID_MIN := 6
const GRID_MAX := 40
## Subdivision of the paper quad. It carries only a slow gradient, so it needs far less.
const PAPER_GRID := 4
## Segments per rounded corner.
const CORNER_SEGS := 5

## ONE PAGE is this many units wide in the world, so the spread is twice it and is centred
## on the origin with the spine on the world Y axis. Deliberately unchanged from the
## single-page vehicle: a panel therefore has exactly the world size it always had, every
## framing constant below keeps the meaning it was tuned with, and the only thing that grew
## is the sheet.
const PAGE_W := 2.0

## HOW MANY PANELS MAY RUN AT ONCE. See [method _update_liveness]: everything on screen
## moves, and the budget is what keeps that affordable. Three covers the panel being read
## plus the neighbours crowding the frame edges at the reading distance. NOT raised for the
## spread even though a wider shot puts more panels in frame - the shot is wider, so each
## panel is smaller, and the budget is about scene cost rather than about screen area.
const LIVE_MAX := 3

## How often a live panel that is NOT the one being read repaints, in frames.
##
## Everything visible moves - but the panel being READ is the one anyone is looking at, and
## a background moment does not need the same temporal resolution as the foreground. Every
## other frame is invisible on a panel at the edge of shot, because those panels are ticked
## with a matching multiple of the step, so they run at the RIGHT SPEED with half the
## samples rather than in slow motion.
##
## Measured on one seed against the same scenes: it and the height-based target sizing took
## the comic's active frame from 84 ms to 73 against the full frame's 41. So three live
## panels cost about 1.8x one, not 3x - the rest of the gap is the stage governor's to
## absorb, which is what it is for.
const OFF_FOCAL_PERIOD := 2

## HOW MANY FRAMES A PANEL MUST ACTUALLY RENDER before it is allowed to be frozen.
##
## THIS IS WHAT "4 OF THE 6 FRAMES ARE PURE BLACK, AND EMPTY" WAS, and the freeze mechanism
## was working perfectly the whole time. A held panel keeps the last picture it DREW - that is
## the measured invariant the whole vehicle rests on (tests/vehicle_probe.gd) - and a panel
## that was cast, given two frames, and frozen has a last picture that is two frames of an
## empty scene. Which is black. Nothing culled it; it was preserved exactly as found.
##
## It only became visible with the spread. A page of three to six panels, read close, kept
## nearly all of them inside the liveness budget long enough to fill in. A spread of nine to
## twelve with the same budget of three, and a reading plan that never visits most of them,
## means a panel can be cast, glimpsed and frozen having drawn nothing at all - including the
## film panel, whose player is still opening its window in those first frames. Reported as
## "the video with the girl WAS playing, but now it's black. It's like the frame got culled."
##
## So liveness is now ordered by NEED first and distance second: a panel below its warm-up
## takes priority over one that has already drawn its moment, and only once every panel on the
## sheet has a picture does the budget go back to following the camera. Twenty-four frames is
## about four tenths of a second - enough for a scene to lay down its first pass and for a
## video window to hand over its first decoded frame.
const WARM_FRAMES := 24

## HOW MANY PANELS OF A SPREAD THE CAMERA ACTUALLY STOPS ON, before the leaf turns.
##
## NOT ALL OF THEM, and that is the correction. The reading used to walk every panel in order
## and turn only when it ran out - so a nine-panel spread was nine cuts, two and a half
## minutes of one sheet, and by the end the camera was hunting for somewhere it had not
## already been. Reported as "it feels like the camera tries to focus on EVERY frame before it
## transitions to the next spread. You wouldn't really expect that, since a lot of those
## frames are already shown during transitions, or during zoom-out."
##
## That is exactly right, and it is how a page is read: you take in the whole spread at a
## glance, rest on a few moments, and turn. The panels that are not rested on are not missing -
## they are drawn the entire time, and every pan across the sheet passes over them. So the
## spread now samples a PLAN of this many panels to settle on, in reading order, and turns
## when the plan is done. A denser spread gets a slightly longer plan, but never one per panel.
const FOCUS := Vector2i(3, 5)

## SPREAD ATTITUDE, in radians - the rest pose the book is rolled into, on all three axes.
## Generous on purpose: the point of putting the paper in a real 3D world is that it can be
## TURNED, and at the reading distance a raked page is what reveals it has depth at all. A
## few degrees reads as a crooked scan rather than as a camera angle.
##
## YAW IS TIGHTER THAN IT WAS (it went to 0.85). A spread is twice as wide as the page these
## were tuned on, so the same yaw that raked one page pleasantly foreshortens the far page of
## a spread into a sliver and throws its far corners behind the eye. Pitch and roll do not
## have that problem - the sheet did not get taller - so only the axis that grew is pulled in.
const PITCH := 0.55
const YAW := 0.44
const ROLL := 0.40
## ...and it never stops moving. Radians per second of slow continuous drift, so the sheet
## turns under the camera through the whole spread rather than settling into a pose.
const DRIFT := 0.035

## CAMERA ELEVATION off the paper, in degrees. Never square-on (at 90 the perspective
## camera and the page agree and the panel is a flat rectangle again) and never grazing
## (the panel foreshortens into a line).
const EL_MIN_DEG := 32.0
const EL_MAX_DEG := 72.0
## Camera roll about the view axis - the Dutch angle, on top of the page's own roll.
const CAM_ROLL := 0.22

## HOW FAR ONE SHOT MAY DEPART FROM THE SPREAD'S SET-UP. This is the fix for "it just bounces
## all over the place... a camera ping-ponging around at random angles".
##
## Every shot used to re-roll its azimuth over the whole circle, its elevation over the
## whole range, its roll and its focal length - independently, at every reading advance. So
## consecutive shots were unrelated positions, and the interpolation between them was a
## camera swinging across the page for no reason. Smoothing it does not help: the problem
## is not the transition, it is that there is nothing in common between where it was and
## where it is going.
##
## So a spread picks ONE set-up - a side, a height, a tilt, a lens - and a shot varies inside
## it. That is coverage rather than chaos, and staying on one side of the subject is the
## oldest rule in the grammar (the 180-degree line). Changing sides is now a PAGE turn,
## which is the one moment where a new set-up reads as a new scene rather than as a mistake.
const AZ_ARC := 0.42              # +/- radians of orbit around the spread's side (~24 deg)
const EL_VARY_DEG := 7.0          # +/- degrees of height off the spread's own
const ROLL_VARY := 0.05           # +/- radians of Dutch on top of the spread's own

## HOW MUCH OF ITS ARC ONE SHOT MAY SPEND, as a fraction. This is the fix for "the camera can
## rotate/turn A LOT every time it shifts from one frame to another".
##
## The angles used to be an INDEPENDENT DRAW per cut: every reading advance re-rolled the
## azimuth anywhere in +/-AZ_ARC, so two consecutive shots could differ by the whole 48-degree
## span, and at a cut every few seconds that is a camera swinging back and forth about a mean.
## Staying inside the arc was never the problem - the problem was arriving anywhere in it in
## one step.
##
## So the angles are a bounded RANDOM WALK instead. A shot moves them by at most this fraction
## of the arc and the walk is clamped to it, so the same total excursion is still reachable -
## it is just spread over several shots, which is what "rotates more gently, over multiple
## per-frame pans" asks for. At a third of the arc it takes three or four cuts to cross, and
## the walk keeps its direction long enough to read as one continuous move rather than as
## jitter.
const ANG_STEP := 0.34
## The lens, chosen once per spread. Narrower than it was (it went to 64): a wide lens this
## close to a panel bends its edges, which is the "unfocused, distorted" half of the report.
const FOV := Vector2(38.0, 54.0)

## HOW TALL THE PANEL BEING READ IS, in frames. 1.0 spans the frame exactly top to bottom;
## above that it overflows and the screen crops into it.
##
## THE FLOOR CAME DOWN, from 1.00 to 0.78, and that is the spread paying for itself. The old
## floor existed because a shot looser than "the panel IS the picture" put a portrait page
## inside a landscape frame and filled the rest with desk - so a wide shot was not a shot,
## it was a framing failure. A spread covers the frame at a much looser shot, so a panel at
## 0.78 frames tall now sits inside a picture that is entirely paper and neighbouring
## panels. That is a comic being READ rather than a video with a border, and it is the shot
## the vocabulary was missing.
const FILL := Vector2(0.78, 1.30)
## The hard near limit, in frames of panel height - however much the spread wants covering,
## the camera stops here. Past about this the panel stops reading as a panel.
##
## SET GENEROUSLY, and the first value was not. At 2.2 this floor came out FURTHER from the
## page than the covering distance did (measured 1.66 against 1.24) and so it, not the
## framing, decided every shot - vetoing exactly the push-in that fills the frame and
## leaving the strip of desk down the side that this whole pass exists to remove. A floor
## that binds in the ordinary case is not a floor, it is the rule.
const CROP_MAX := 4.5
## How far the aim is pulled from the panel's centre toward the middle of ITS OWN PAGE. A
## panel at the outer trim, framed dead on, puts the trim edge - and the void behind the
## paper - into shot; pulling the aim in keeps the sheet under the panel at every panel.
##
## TOWARD THE PAGE, NOT THE SPREAD, and that distinction only exists because of the spread.
## The middle of a spread IS THE SPINE - the widest gutter on the sheet - so pulling every
## aim toward it biases every shot toward the one place with no picture in it, which is the
## defect this pass exists to remove arriving by the other door. Pulling toward the page's
## own centre keeps the original purpose exactly: the risk was always the OUTER trim, and
## the inner edge has a whole second page beyond it.
const AIM_PULL := 0.12

## Pan rates (ease per second) for moving between panels: a slow drift, an ordinary walk,
## and a whip. A comic read at one speed is a slideshow on rails.
const PAN_SLOW := 0.9
const PAN_WALK := 2.2
const PAN_WHIP := 7.5
## The spread's own attitude eases at its own, much slower rate - it is scenery, not a move.
const PAGE_EASE := 0.7
## A page turn has somewhere to be.
const TURN_EASE := 3.0
## How long a page turn takes, in seconds.
const TURN_TIME := 1.15
## How far the leaf swings. A HALF TURN exactly, because at pi the leaf lies flat on the
## left page and its back face - the incoming left page, see [method ComicSpread.mirror] -
## lands precisely where that page is about to be drawn flat. Anything short of pi would
## have to cross-dissolve the difference.
const TURN_ARC := PI

## THE SPREAD LIES ON SOMETHING. Without it the frame past the trim edge is pure black,
## which at a hard rake is a wedge of nothing across the shot - and black is not a
## background, it is the absence of one.
##
## PAINTED IN SCREEN SPACE, not as a world quad. A surface big enough to never run out
## under a close raked camera has corners far outside the view frustum, and the projection
## has to drop any quad with a corner behind the eye (it inverts through infinity) - so the
## world-quad version simply never drew, at any of the sizes worth having. A defocused
## surface has no texture to rake anyway; what it needs to be is present and not black.
## How dark the wash goes at the corners of the frame, as a fraction of the desk colour.
const DESK_VIGNETTE := 0.55
## The sheet's shadow on it: offset in page widths, and how far off the paper it is cast.
const SHADOW_OFF := Vector2(0.045, 0.055)
const SHADOW_DEPTH := 0.06

# --- state -------------------------------------------------------------------
var _slots: Array = []            # POOL * 2 SubViewports; spread s uses pool (s % 2)
var _pool := 0                    # which pool the CURRENT spread is drawing from
var _spread: ComicSpread = null
var _spread_i := -1
var _cast: Array = []             # per panel of the current spread: its GhostScene, or null
var _read := 0                    # panel the camera is reading, and the Director's current
## The panels this spread will settle on, in reading order - see FOCUS. `_read` is one of
## these, and `_step` is how far through the plan the reading has got.
var _plan: Array = []
var _step := 0
var _to_cast: Array = []          # panels still waiting to be cast, one per frame
var _prev: ComicSpread = null     # the outgoing spread, alive only through a turn
var _turn_t := -1.0               # seconds into a page turn, < 0 when none

## WHICH PANEL OF THIS SPREAD HOLDS FOOTAGE, or -1 for none, and the clip it holds.
##
## AT MOST ONE PER SPREAD, which is the whole answer to a question that would otherwise need
## a mechanism. A clip's position is a pure function of the show clock (see
## [method Films.position_at]), so two panels sampling the same clip at the same instant
## would show the same picture twice - and the spread is what you see at once, so one per
## spread IS one at a time. During a turn the outgoing spread's panels are already stopped,
## so even then only one is playing.
var _film_at := -1
var _film_clip: Dictionary = {}
var _film_prev := ""              # last spread's clip path, so two running do not repeat

var _lens := Lens3D.new()
## The two solver lenses, allocated ONCE. They used to be built inside _fit and _cover,
## which is three RefCounted allocations per frame inside a 22-iteration bisection - see
## AUDIT.md. Neither survives the call, so neither needed to be new.
var _fit_lens := Lens3D.new()
var _cover_lens := Lens3D.new()
var _mod: ModBank = null
var _bookend := 1.0
var _panel_px := 512
var _stage_size := Vector2(1280, 720)
var _live: Array = []             # panel indices whose viewport is running this frame
## Frames each panel of the current spread has actually rendered. See WARM_FRAMES.
var _warm: Array = []
var _frame := 0                   # for the off-focal repaint phase, see OFF_FOCAL_PERIOD

# THE SPREAD'S ATTITUDE, eased toward a target that itself DRIFTS. Sampled on all three axes
# with real magnitude: a book that only pitches a few degrees reads as a slightly crooked
# scan, and the whole reason to put the sheet in a real 3D world is that it can be turned.
var _att := Vector3.ZERO
var _att_target := Vector3.ZERO
var _att_rate := Vector3.ZERO     # slow continuous drift, per second
## The attitude as a BASIS, rebuilt once per advance() and read by everything that places a
## point on the paper. It used to be built inside _world (then _page_point) - six
## trigonometric evaluations and a 3x3 construction, per vertex, roughly 2500 times a frame
## with the identical argument, and AUDIT.md attributes most of the page's ~18 ms of _draw
## to it. Hoisting it is exact: the value was always the same.
var _att_basis := Basis.IDENTITY

# THE CAMERA STATE, expressed in SPREAD-LOCAL SPHERICAL rather than as a world position: an
# azimuth around the aim and an elevation off the paper. That is what makes the rake real -
# an eye on the sheet's normal sees a flat rectangle however the sheet is rotated - and it is
# also what makes travelling between panels a move PARALLEL TO THE PAPER, because the
# direction is held and only the point it looks at changes.
# Keys: aim (spread coords), az, el, roll, fill, fov. See _station.
## The spread's SET-UP: the side the camera is on, its height, its tilt and its lens. Fixed
## for the spread; see AZ_ARC.
var _az_base := 0.0
var _el_base := 1.0
var _roll_base := 0.0
var _lens_fov := 46.0
## The bounded walk each angle is currently at, as an offset from its base. See ANG_STEP.
var _az_walk := 0.0
var _el_walk := 0.0
var _roll_walk := 0.0

var _cam := {"aim": Vector2(1.0, 0.7), "az": 0.0, "el": 1.0, "roll": 0.0,
	"fill": 1.1, "fov": 52.0}
## The move being travelled: {kind, a, b, dur, ease}. See MOVES.
var _mv := {}
var _mv_t := 0.0
## How many moves have been CHAINED since the last cut - see [method _chain_move]. It is
## part of the seed, so a chain is reproducible without being a repeat of the move it
## followed.
var _chain := 0
var _dip_t := -1.0                # seconds into a dip to black, < 0 when none
var _roll := 0.0                  # the roll actually in force (field-flattened)
var _eye := Vector3(0, 0, 3.0)
var _look := Vector3.ZERO
var _fov := 52.0

# paper, sampled per session
var _ink_weight := 0.0038
var _desk := Color(0.06, 0.055, 0.07)
var _paper := Color(0.92, 0.90, 0.855)
var _ink := Color(0.07, 0.065, 0.075)
var _rng := RandomNumberGenerator.new()


func mount(st: SubViewport) -> void:
	super.mount(st)
	# ...and belt to the half-texel inset's braces: never let a UV wrap even if one
	# escapes the inset. The default is already this; saying so is cheap and the failure
	# it prevents (the far edge of a panel bleeding in along the near one) is not obvious
	# to diagnose.
	texture_repeat = CanvasItem.TEXTURE_REPEAT_DISABLED
	_size_targets(Vector2(st.size))
	_build_slots()


## Everything sampled, rolled here rather than in mount() because the session seed is not
## resolved until the Director attaches (see Vehicle.begin_session).
func begin_session() -> void:
	_rng.seed = Director.session_seed() ^ 0x0C031C
	# Paper is never pure white - a white page reads as a blank canvas rather than as
	# printed stock - and it is sampled, like everything else here.
	_paper = Color.from_hsv(_rng.randf_range(0.055, 0.11),
		_rng.randf_range(0.05, 0.13), _rng.randf_range(0.88, 0.95))
	# Ink is not black either. A cool near-black prints like ink; pure #000 prints like a
	# UI border.
	_ink = Color.from_hsv(_rng.randf_range(0.58, 0.72), _rng.randf_range(0.05, 0.20),
		_rng.randf_range(0.05, 0.13))
	# The surface the sheet lies on: dark enough that the paper is still the brightest thing
	# in the frame by a long way, but never black.
	_desk = Color.from_hsv(_rng.randf_range(0.02, 0.12), _rng.randf_range(0.10, 0.32),
		_rng.randf_range(0.055, 0.115))
	_ink_weight = _rng.randf_range(0.0030, 0.0052)
	_mod = ModBank.new(Director.session_seed() ^ 0x9A6E)
	_reset_book()


func release() -> void:
	# The vehicle OUTLIVES a session (main owns it; the synthesis modes re-attach per
	# take), so release resets the BOOK rather than tearing the pool down - rebuilding
	# twenty-four render targets on every settings change is exactly the reallocation churn
	# main's governor warns about.
	_reset_book()


func _reset_book() -> void:
	for i in _slots.size():
		_blank(_slots[i])
	_cast = []
	_warm = []
	_to_cast = []
	_plan = []
	_step = 0
	_spread_i = -1
	_read = 0
	_prev = null
	_turn_t = -1.0
	_mv = {}
	_mv_t = 0.0
	_chain = 0
	_dip_t = -1.0
	_film_at = -1
	_film_clip = {}
	_film_prev = ""
	_turn_spread(0)


## A window resize changes what a panel is worth in pixels, but NOTHING is resized here.
##
## A render target reallocated while it is being sampled is the black-triangle and
## colour-noise corruption main's stage governor documents at length - and worse, a resize
## also throws away the held panels' pictures. So the new figure is remembered and applied
## at the next spread, where every target is repainted from scratch anyway.
func on_stage_resized(size: Vector2) -> void:
	_size_targets(size)
	queue_redraw()


func _size_targets(size: Vector2) -> void:
	_stage_size = size
	# A panel fills the frame at the reading distance, so it is worth close to the whole
	# short axis. Clamped below so a small window still gets a legible panel, and above so
	# a 4K export does not allocate twenty-four enormous targets.
	_panel_px = clampi(int(minf(size.x, size.y) * PANEL_SCALE), PANEL_MIN, PANEL_MAX)


func _build_slots() -> void:
	for i in POOL * 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(SLOT_IDLE, SLOT_IDLE)
		vp.transparent_bg = false
		vp.disable_3d = true          # every scene rasterizes to the 2D canvas
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.process_mode = Node.PROCESS_MODE_DISABLED
		add_child(vp)
		_slots.append(vp)


# --- the Vehicle contract ----------------------------------------------------

## The spread owns its cast. See [method Vehicle.owns_cast] - this is the whole difference
## between a page and a slideshow.
func owns_cast() -> bool:
	return true


## A Director cut MOVES THE READING to the next panel; it does not build anything. The
## panel already holds a live scene, cast when the spread turned.
func take_over(_outgoing: GhostScene) -> GhostScene:
	if _spread == null:
		return null
	if _step + 1 >= _plan.size():
		_turn_spread(_spread_i + 1)
	elif _spread_i >= 0:
		_step += 1
		_read = int(_plan[_step])
		_choose_move()
	# CAST ON DEMAND. The rest of the spread is cast one panel per frame to keep the turn
	# from hitching, and a fast cut can reach a panel before its turn in that queue comes
	# up - which handed the Director a null and left the reading stuck on the panel behind.
	# _cast_panel is idempotent, so this only ever builds the one that was still owed.
	_cast_panel(_read)
	return _focal()


func owns_bookend() -> bool:
	return true


func advance(features, delta: float, bookend: float) -> void:
	_bookend = bookend
	if _mod != null:
		_mod.advance(delta, _energy_of(features))
	_cast_one()
	if _turn_t >= 0.0:
		_turn_t += delta
		if _turn_t >= TURN_TIME:
			_turn_t = -1.0
			_prev = null
			# THE OUTGOING SPREAD'S TARGETS ARE NOW DEAD. Nothing samples that pool again
			# until the next turn opens it, and _open_slot sizes a slot before anything
			# draws it - so this is the one safe moment to give the memory back. Skipping
			# it kept a full spread of full-size render targets alive for the whole of the
			# next spread, which is half the vehicle's VRAM doing nothing.
			for i in POOL:
				var vp: SubViewport = _slots[(1 - _pool) * POOL + i]
				if vp.size.x > SLOT_IDLE or vp.size.y > SLOT_IDLE:
					vp.size = Vector2i(SLOT_IDLE, SLOT_IDLE)
	_ease(delta)
	_prepare_lens()
	_update_liveness()
	_tick_cast(features, delta)
	queue_redraw()


## AudioFeatures is not a type this file should have to import to read one number off, and
## a vehicle must keep working when there is no audio at all (the splash, a songless boot),
## so this asks softly rather than declaring a dependency.
func _energy_of(features) -> float:
	if features == null:
		return 0.0
	return clampf(float(features.energy), 0.0, 1.0)


# --- the book ----------------------------------------------------------------

func _focal() -> GhostScene:
	if _read < 0 or _read >= _cast.size():
		return null
	var sc = _cast[_read]
	return sc if sc != null and is_instance_valid(sc) else null


func _turn_spread(idx: int) -> void:
	if idx > 0 and _spread != null:
		_prev = _spread
		_turn_t = 0.0
	# Let the outgoing spread's cast go, but NOT its pictures: _blank stops each viewport
	# before freeing what is inside it, so the spread turning away is still a drawn spread.
	for i in POOL:
		_blank(_slots[_pool * POOL + i])
	_spread_i = idx
	_pool = idx % 2
	_spread = ComicSpread.new(hash([Director.session_seed(), "comic-spread", idx]))
	_cast = []
	_cast.resize(_spread.panels.size())
	_warm = []
	_warm.resize(_spread.panels.size())
	_warm.fill(0)
	_read = 0
	# The panel being read is cast NOW - the show cannot open on an empty frame - and the
	# rest are queued one per frame. Building a dozen scenes in one frame is a visible
	# hitch, and there is nowhere to hide it; spread over the next few frames it lands under
	# the page turn, and any panel the camera reaches before its turn is cast on demand.

	_choose_film()               # BEFORE the plan: _roll_plan forces the film panel into it
	_roll_plan()
	_read = int(_plan[0]) if not _plan.is_empty() else 0
	_step = 0
	# EVERY PANEL EXCEPT THE ONE CAST BELOW. It used to be `range(1, n)`, which was correct
	# only while the reading always opened on panel 0. It does not any more - a spread picks a
	# PLAN and opens on `_plan[0]` (see _roll_plan) - so whenever that is not panel 0, panel 0
	# was queued by nobody and cast by nobody, and stayed an empty ruled frame for the whole
	# spread. Caught by the probe's per-panel line as `1[- - 0]` on a spread whose plan began
	# at panel 2, which is exactly the "empty frame" this pass exists to remove.
	_to_cast = []
	for i in _spread.panels.size():
		if i != _read:
			_to_cast.append(i)
	_cast_panel(_read)
	_choose_spread_look()
	_choose_move()


## WHICH PANELS THIS SPREAD WILL SETTLE ON, in reading order. See FOCUS.
##
## Sampled, spread out, and it always includes the FILM PANEL when there is one - a piece of
## footage on the sheet is the one moment that is certainly worth stopping on, and leaving it
## to chance means importing a video and then watching the camera ignore it.
##
## Spaced by picking one panel from each of `want` equal SLICES of the reading order rather
## than by drawing indices at random. A random draw of four from nine clusters: it will
## happily pick panels 5, 6, 7 and 8 and leave the whole left page unvisited, which reads as
## the camera getting stuck in a corner of the sheet. One per slice guarantees the plan walks
## the spread, while the choice inside each slice keeps it from being the same four panels
## every time.
func _roll_plan() -> void:
	_plan = []
	if _spread == null:
		return
	var n := _spread.panels.size()
	if n <= 0:
		return
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "comic-plan", _spread_i])
	var want := clampi(r.randi_range(FOCUS.x, FOCUS.y), 1, n)
	var picked := {}
	for k in want:
		var lo := int(floor(float(k) * float(n) / float(want)))
		var hi := int(floor(float(k + 1) * float(n) / float(want))) - 1
		var at := clampi(r.randi_range(lo, maxi(lo, hi)), 0, n - 1)
		picked[at] = true
	if _film_at >= 0:
		picked[_film_at] = true
	for i in n:                       # reading order, and each panel at most once
		if picked.has(i):
			_plan.append(i)


## DOES THIS SPREAD GET A PIECE OF FOOTAGE, and if so which panel and which clip.
##
## Seeded off the spread, like everything else here, so a session replays the same footage
## in the same panels - the show is reproducible from one seed and this is not the place
## to make it stop being. A frequency of 0, or a library with nothing in it, leaves
## `_film_at` at -1 and the comic behaves exactly as it did before films existed.
func _choose_film() -> void:
	_film_at = -1
	_film_clip = {}
	if _spread == null or Films.frequency() <= 0.0:
		return
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "film-spread", _spread_i])
	if r.randf() > Films.frequency():
		return
	var list := Films.clips()
	if list.is_empty():
		return
	var at := r.randi() % list.size()
	# NOT THE SAME CLIP TWICE RUNNING. With one film panel per spread, a small library and
	# an independent draw per spread, the same clip lands on consecutive spreads often
	# enough to read as "that video again" rather than as a library.
	if list.size() > 1 and String((list[at] as Dictionary).get("source", "")) == _film_prev:
		at = (at + 1) % list.size()
	_film_clip = list[at]
	_film_at = _best_panel_for(_film_clip, r)
	# IS THERE ANYTHING TO PLAY YET. A clip is prepared a window at a time, cut from the
	# original when something wants it (see Films.WINDOW), so the window covering this
	# moment may still be encoding. Asking starts that cut; not getting it back means this
	# spread simply goes without footage, and the panel is an ordinary scene rather than a
	# blank rectangle waiting for a file. The whole feature self-throttles on this line -
	# film appears as often as the machine can prepare it.
	if not Films.warm(_film_clip, maxf(Spectrum.current.time, 0.0)):
		_film_clip = {}
		_film_at = -1
		return
	_film_prev = String(_film_clip.get("source", ""))


## WHICH PANEL SUITS THIS CLIP'S SHAPE. Footage covers its panel, so whichever axis is
## spare gets cropped off - a 16:9 clip in a 0.45-aspect panel loses three quarters of its
## width, and there is no framing clever enough to make that not matter. A spread offers
## twice as many shapes as a page did, so the answer is simply to put the film in the one
## closest to the clip's own, where the crop is a trim rather than a demolition.
##
## Compared in LOG space, because aspect is a ratio: 2.0 and 0.5 are equally wrong for a
## square clip, and subtracting them would call one of them nearly right.
##
## Falls back to a plain random panel when the clip's shape is unknown, which is the honest
## answer rather than a guess - see Films.aspect_of.
func _best_panel_for(clip: Dictionary, r: RandomNumberGenerator) -> int:
	var n := _spread.panels.size()
	var want := Films.aspect_of(clip)
	if want <= 0.0 or n <= 1:
		return r.randi() % maxi(n, 1)
	var best := 0
	var best_err := INF
	for i in n:
		var err := absf(log(maxf(_spread.panel_aspect(i), 0.01) / want))
		if err < best_err:
			best_err = err
			best = i
	return best


## One queued panel per frame. See _turn_spread.
func _cast_one() -> void:
	if _to_cast.is_empty():
		return
	_cast_panel(int(_to_cast.pop_front()))


func _cast_panel(i: int) -> void:
	if _spread == null or i < 0 or i >= _cast.size():
		return
	if _cast[i] != null and is_instance_valid(_cast[i]):
		return
	var vp := _open_slot(i)
	# FOOTAGE, where the spread called for it. Cast directly rather than through the
	# Director: a film is not in the catalogue (see FilmScene), because a scene that only
	# exists when the viewer has imported something would make the running order depend on
	# the library.
	if i == _film_at and not _film_clip.is_empty():
		var film := FilmScene.new()
		film.set_clip(_film_clip, maxf(Spectrum.current.time, 0.0))
		film.init_with_seed(hash([_spread_i, i, "film"]), "static")
		film.scene_name = "film"
		vp.add_child(film)
		_cast[i] = film
		return
	# The salt is the panel index, so the spread's panels are separate draws off the novelty
	# scheduler rather than the same one repeated - the Director's clock has not moved
	# between them (see Director.mint_scene). Quiet for every panel but the one being read:
	# casting a spread is one change of scene, not a dozen.
	var sc := Director.mint_scene(hash([_spread_i, i]) & 0xFFFF, i != _read)
	vp.add_child(sc)
	_cast[i] = sc


## Size a panel's render target to ITS aspect and start it running.
##
## SIZED BY HEIGHT, because the camera is fitted by height (see _fit): a panel spans the
## frame vertically, so its target wants about the stage's height and its width follows
## from its aspect. The obvious alternative - constant AREA - overshoots badly at the tall
## end, where it makes a 0.44-aspect panel half again taller than the screen it will be
## drawn on, for no visible gain and 2.3x the pixels.
func _open_slot(i: int) -> SubViewport:
	var vp: SubViewport = _slots[_pool * POOL + i]
	var a := _spread.panel_aspect(i)
	var h := _panel_px
	var w := int(round(float(_panel_px) * a))
	vp.size = Vector2i(maxi(64, w), maxi(64, h))
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	vp.process_mode = Node.PROCESS_MODE_INHERIT
	return vp


## Stop a viewport WITHOUT clearing it: the render target keeps its last frame. The ORDER
## matters - stop first, free second - or the target repaints itself empty on the frame
## between the two.
func _freeze(vp: SubViewport) -> void:
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	vp.process_mode = Node.PROCESS_MODE_DISABLED


func _resume(vp: SubViewport) -> void:
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	vp.process_mode = Node.PROCESS_MODE_INHERIT


## Stop a viewport and release the scene inside it, keeping the picture it last drew.
func _blank(vp: SubViewport) -> void:
	_freeze(vp)
	for c in vp.get_children():
		c.queue_free()


# --- liveness ----------------------------------------------------------------

## EVERY PANEL YOU CAN SEE IS MOVING, and no panel you cannot see costs anything.
##
## A spread whose panels arrive one at a time is a slideshow with gutters; a comic page has
## all of it drawn at once. But a dozen live scenes is not a thing that runs - the stage
## governor spends its whole budget on one - so liveness follows the CAMERA instead of the
## clock. At the reading distance one panel fills the frame and its neighbours crowd the
## edges, so what is on screen is typically one to three panels; the rest are stopped
## render targets holding their last frame, exactly as before, and they resume when the
## camera comes back to them.
##
## LIVE_MAX is the hard bound. It is ordered by distance from the panel being read, so if a
## rake ever puts more of the spread in shot than the budget allows, the ones that keep
## moving are the ones the reading is about.
func _update_liveness() -> void:
	if _spread == null:
		return
	var frame := Rect2(Vector2.ZERO, _stage_size).grow(_stage_size.x * 0.06)
	var want: Array = []
	for i in _cast.size():
		if _cast[i] == null or not is_instance_valid(_cast[i]):
			continue
		# A panel still below its warm-up is wanted WHEREVER IT IS, on screen or not. It has
		# no picture yet, so leaving it out is what makes it black - and the camera reaches
		# most of the sheet eventually, so "off screen right now" is not "never seen".
		var cold: bool = int(_warm[i]) < WARM_FRAMES
		if i != _read and not cold and not _panel_rect(i).intersects(frame):
			continue
		want.append({"i": i, "d": absi(i - _read), "cold": 0 if cold else 1})
	# NEED FIRST, DISTANCE SECOND. Until every panel has drawn its moment the budget goes to
	# the ones that have not; after that it follows the reading exactly as it did.
	want.sort_custom(func(a, b):
		return int(a.cold) < int(b.cold) if int(a.cold) != int(b.cold) else int(a.d) < int(b.d))
	var keep: Array = []
	for e in want:
		if keep.size() >= LIVE_MAX:
			break
		keep.append(int(e.i))
	_frame += 1
	for i in _cast.size():
		if _cast[i] == null or not is_instance_valid(_cast[i]):
			continue
		var vp: SubViewport = _slots[_pool * POOL + i]
		# On this frame at all, and if so, is it this one's turn? The read panel's turn is
		# every frame; the others are phased against each other by index so two of them
		# never land on the same frame.
		# The film panel is exempt from the off-focal phasing: half the samples looks
		# like a scene running at the right speed with a coarser clock, and looks like
		# BROKEN PLAYBACK on footage, which the eye reads at a much finer grain.
		# A COLD PANEL RUNS EVERY FRAME IT IS KEPT, phasing included: half the samples is
		# fine for a moment that is already on the paper and wrong for one still drawing it.
		var on: bool = keep.has(i) and (i == _read or i == _film_at
			or int(_warm[i]) < WARM_FRAMES or (_frame + i) % OFF_FOCAL_PERIOD == 0)
		if on:
			_warm[i] = int(_warm[i]) + 1
			_resume(vp)
		else:
			_freeze(vp)
	_live = keep


## Drive every live panel EXCEPT the one being read - the Director drives that one, as its
## current scene, and ticking it twice would run it at double speed. This is the same pair
## of calls Director._tick_animation makes, because it is the same job.
func _tick_cast(features, delta: float) -> void:
	var focal := _focal()
	var dt := minf(delta, 1.0 / 30.0)
	for i in _live:
		var sc = _cast[i]
		if sc == null or not is_instance_valid(sc) or sc == focal:
			continue
		# The film panel runs every frame (period 1) - see _update_liveness.
		var period := 1 if (i == _film_at or int(_warm[i]) < WARM_FRAMES) else OFF_FOCAL_PERIOD
		if (_frame + i) % period != 0:
			continue                      # not this panel's frame - see OFF_FOCAL_PERIOD
		# ...and when it IS its frame, it is handed the whole period's worth of time, so it
		# runs at the right speed rather than at 1/period of it.
		sc.update(features, dt * period)
		sc.view.commit(dt * period)


# --- camera ------------------------------------------------------------------

## THE MOVE VOCABULARY. A cut picks one of these, and the camera then TRAVELS IT.
##
## The first cut of this had no vocabulary: every cut set a new target and the camera eased
## toward it, which is a spring settling, not a camera. It gave one behaviour - bounce to
## the next panel, sit on it, bounce again - and settling on a single panel is a shot that
## should be an EVENT, not the rule. So a move is now a PATH: a start state, an end state, a
## duration and a curve, travelled at its own speed. Most of them never settle at all.
##
##   w      relative weight in the bag
##   dur    seconds, sampled in this range
##   ease   linear (a dolly, constant speed) / smooth / out (fast then settle) / snap (a cut)
##   hard   true = the move BEGINS somewhere else entirely, so it starts on a jump cut
##          rather than continuing from where the camera was
##   open   true = the move deliberately ENDS OFF A PANEL (it runs past its subject, or
##          across a row). Such a move is exempt from the settle guard in _choose_move,
##          because pulling its end back onto a panel would delete the gesture - and it is
##          safe only because a finished move now chains (see _chain_move) instead of
##          stopping wherever it ran out.
##   chain  true = eligible as a CHAINED successor, i.e. gentle and continuous. A jump is a
##          thing a CUT does; a camera that jumps on its own, with no cut behind it, reads
##          as a glitch.
##
## THE BAG IS MOSTLY GENTLE, and that is a correction. It was first weighted the other way,
## in answer to "a constant shot of a single frame should be an event, not the rule" - and
## it overshot: 41% of the weight began somewhere else entirely, so two shots in five opened
## on a jump. Read back as "it just bounces all over the place... not cinematic at all".
##
## Punctuation is now about one shot in six. The rule the two reports agree on, once both
## are taken seriously, is that a camera should be MOVING most of the time and JUMPING
## rarely - which is neither "sit on each panel" nor "cut constantly".
const MOVES := {
	# --- travelling: the camera is moving for the whole shot ---------------------
	"drift":   {"w": 4.0, "dur": [8.0, 15.0], "ease": "smooth", "hard": false, "chain": true},
	"push":    {"w": 3.0, "dur": [7.0, 13.0], "ease": "smooth", "hard": false, "chain": true},
	"track":   {"w": 2.0, "dur": [9.0, 16.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"orbit":   {"w": 2.0, "dur": [9.0, 15.0], "ease": "smooth", "hard": false, "chain": true},
	# THE SPREAD'S OWN MOVE, and the reason the spread earns its complexity: a lateral pan
	# from a panel on one page, across the spine, onto a panel on the other. It is the
	# longest continuous run of content the vehicle has ever had - a single page had no
	# path longer than one page's width, and half of that was margin.
	"spine":   {"w": 2.0, "dur": [12.0, 20.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"sweep_h": {"w": 1.5, "dur": [9.0, 16.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"sweep_v": {"w": 1.0, "dur": [9.0, 16.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"pull":    {"w": 1.5, "dur": [7.0, 13.0], "ease": "smooth", "hard": false, "chain": true},
	# --- and the one that just looks at a panel ----------------------------------
	"hold":    {"w": 1.5, "dur": [5.0, 10.0], "ease": "smooth", "hard": false, "chain": true},
	# --- arrivals: a gesture that lands, then holds ------------------------------
	"swoop":   {"w": 0.6, "dur": [2.6, 5.0],  "ease": "out",    "hard": true},
	"whip":    {"w": 0.4, "dur": [0.45, 0.9], "ease": "out",    "hard": false},
	# --- discontinuities: punctuation, not grammar -------------------------------
	"cut":     {"w": 0.8, "dur": [4.0, 9.0],  "ease": "snap",   "hard": true},
	"dip":     {"w": 0.4, "dur": [4.0, 9.0],  "ease": "snap",   "hard": true},
}

## How far past the subject a `track` keeps going, in page widths. The camera does not stop
## when it arrives - it passes through, which is what a tracking shot is.
const TRACK_OVERRUN := 0.45
## THE AIM NEVER LEAVES THE SHEET. Spread coordinates, as a margin inside the trim.
##
## A move that aims past the outer edge is aiming at the desk, and no framing distance can
## then put paper across the whole frame - the covering solve has nothing to solve. Sweeps
## and tracks deliberately run to and past the edges, so they are clamped here rather than
## being written not to.
const AIM_MARGIN := 0.06
## How deep a `push` gets, in frames of panel height.
const PUSH_FILL := Vector2(2.2, 3.8)
## The far station a `swoop` starts from: elevation in degrees, and how far off.
const SWOOP_EL_DEG := Vector2(14.0, 26.0)
const SWOOP_FILL := 0.42
## The arc an `orbit` sweeps, in radians, at severity 1.
##
## PULLED IN FROM 1.1-2.6 (63 to 149 degrees), which was most of a lap of the sheet in one
## shot. It survived the first pass because the move is only about one shot in ten, so it
## never dominated a still frame - but it dominated the MEAN: with the per-shot walk down to
## about eight degrees, the measured shot-to-shot swing at the default was still 24.6, and two
## thirds of that was orbit and swoop. A rare enormous move is exactly what "the camera can
## rotate/turn A LOT every time it shifts" describes, because rare at one shot in ten is every
## half minute. At 26 to 66 degrees it is still plainly an orbit and no longer a lap.
const ORBIT_ARC := Vector2(0.45, 1.15)
## A `dip` goes out over the first of these and back over the second, in seconds.
const DIP_OUT := 0.16
const DIP_IN := 0.34

## THE CAMERA SEVERITY KNOB, applied. See [member Director.camera] for what it is.
##
## Every constant below that describes how MUCH the camera does is read through one of these,
## so the slider moves the whole behaviour along a single axis instead of asking anyone to
## balance six numbers. Written as multipliers around 1.0 at the default setting, so a session
## that never touches the slider gets exactly the tuned camera.
##
## What it scales, and why each one belongs to "severity" rather than to taste:
##   the angular arc a shot may reach          - how far the camera can swing at all
##   how much of that arc one shot spends      - how gently it gets there
##   how far a page turn moves the set-up      - the one big change in the grammar
##   move duration                             - a short move is a fast move
##   the weight of the discontinuous moves     - at 0 the camera never jump-cuts
##   how deep a push goes, and the orbit arc   - the two most violent gestures
##   the sheet's own drift rate                - the scenery, which reads as camera motion
## Duration is EXPONENTIAL in the setting (pow, not lerp) so that 1.0 is exactly the tuned
## value and the two halves of the range are symmetric: 0 makes a move 1.8x longer by the same
## factor 2 makes it 0.55x shorter. A lerp between the endpoints does not pass through 1.
const DUR_SEVERITY := 0.55
## How far a page turn may move the camera's side of the sheet, in radians at severity 1. It
## used to re-roll over the whole circle; that is the single largest move the camera makes,
## and on a spread it now happens only every six to twelve cuts, so it is worth having but
## not worth having unconditionally.
const TURN_AZ := 1.2
## The discontinuous moves scale by severity SQUARED - they vanish faster than everything else
## going down and arrive faster going up, because "does it ever jump cut" is the single thing
## that most decides whether a camera reads as gentle or as chaotic.
const HARD_SEVERITY := 2.0


## The severity in force, 0..2. Asked of the Director rather than held here so the slider
## takes effect on the next planned move without anything having to be pushed at the vehicle.
func _sev() -> float:
	return clampf(Director.camera, Director.CAMERA_MIN, Director.CAMERA_MAX)


## HOW MUCH GENTLER A FIELD SCENE IS RAKED, 0..1 toward square-on.
##
## ghost already types every scene as `subject` (a discrete object) or `field` (fills the
## frame), and [Shots] already gives field scenes the gentle moves only. The same rule
## belongs here for the same reason, and the reason is visible: a hard rake reads as
## FORESHORTENING when there is a recognisable object in the panel to be foreshortened, and
## as WARPING when there is not. A fractal field at 30 degrees off the paper does not look
## like a picture seen at an angle, it looks like a broken picture - which is exactly how it
## was reported.
const FIELD_FLATTEN := 0.62

## HOW MUCH PANEL IS UNDER THE AIM: the shape of the falloff, in units of the panel's own
## half-size, measured Chebyshev (so it is the panel's rectangle, not a disc inside it).
##
## THIS IS THE FIX FOR THE WHOLE REPORT, and it is one number turned into a constraint:
## "focusing a zoom-in camera at the corner of a frame is terrible". Everything before this
## tried to make the camera AIM well; nothing stopped it from being TIGHT while aiming
## badly, and tight-while-badly-aimed is the only combination that actually looks broken.
## Wide over a gutter shows four panels and reads as a page. Tight over a gutter shows
## paper.
##
## So content is scored, and the framing is a function of it (see _place_eye). Inside CORE -
## the middle 55% of a panel - the score is 1 and the shot is exactly as tight as the move
## asked for. By EDGE, just inside the panel's border, it is 0 and the shot has opened to
## OFF_FILL. Between them it is smooth, so a pan across a gutter WIDENS as it crosses and
## closes again on the far panel, which is what a camera operator does without being told.
const CONTENT_CORE := 0.55
const CONTENT_EDGE := 0.98
## The framing a shot falls back to when there is no panel under the aim, in frames of panel
## height. Well under 1: at 0.62 the panel the camera is nearest is a bit over half the
## frame, so the gutter it is actually looking at is surrounded by the four panels that
## gutter separates. That is a picture of a comic page. Note the covering solve can still
## refuse to go this wide - see _place_eye - and on a spread it usually does not have to.
const OFF_FILL := 0.62
## How much content a move's RESTING aim must have. Not 1.0: snapping every terminal aim to
## a panel's dead centre would make every settled shot the same shot, and the whole point of
## sampling a station is that it is not. This is "far enough inside the panel that the shot
## is of the panel", and _settle_aim walks the aim in until it clears the bar.
const SETTLE_CONTENT := 0.72


## The spread's attitude for this spread: a real rake on all three axes, plus a slow
## continuous drift so the sheet is never still.
func _choose_spread_look() -> void:
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "comic-spread-look", _spread_i])
	_att_target = Vector3(
		r.randf_range(-PITCH, PITCH),
		r.randf_range(-YAW, YAW),
		r.randf_range(-ROLL, ROLL))
	_att_rate = Vector3(
		r.randf_range(-DRIFT, DRIFT),
		r.randf_range(-DRIFT, DRIFT),
		r.randf_range(-DRIFT, DRIFT) * 0.5)
	# THE SET-UP FOR THIS SPREAD - decided ONCE, and then everything below only varies within
	# it. See the note above AZ_ARC: re-rolling these per shot is what made the camera
	# ping-pong, and no amount of smooth interpolation between two unrelated positions
	# rescues it, because the problem is that they are unrelated.
	var sev := _sev()
	if _spread_i <= 0:
		_az_base = r.randf_range(0.0, TAU)      # the first spread has nowhere to come from
	else:
		# A BOUNDED CHANGE OF SIDE, not a fresh draw. Re-rolling over the whole circle is the
		# largest single move the camera makes; on a spread it happens only every six to
		# twelve cuts, so it is worth having - but at a gentle setting it should be a lean
		# rather than a walk round the table.
		_az_base += r.randf_range(-TURN_AZ, TURN_AZ) * sev
	_el_base = deg_to_rad(r.randf_range(EL_MIN_DEG, EL_MAX_DEG))
	_roll_base = r.randf_range(-CAM_ROLL, CAM_ROLL) * sev
	_lens_fov = r.randf_range(FOV.x, FOV.y)
	# The walk starts each spread where the last one left it, so a page turn is the only
	# discontinuity in the angles and everything between turns is continuous.
	# The sheet's own drift is part of how busy the picture is, so it scales too.
	_att_rate *= sev


## A camera station: where it aims on the spread, and from what angle and how close.
func _station(r: RandomNumberGenerator, panel: int) -> Dictionary:
	var i := clampi(panel, 0, maxi(0, _spread.panels.size() - 1))
	var sev := _sev()
	# WITHIN THE SPREAD'S SET-UP, and reached a STEP AT A TIME. See AZ_ARC for the arc and
	# ANG_STEP for why a shot may only spend a fraction of it.
	_az_walk = _walk(_az_walk, AZ_ARC * sev, r)
	_el_walk = _walk(_el_walk, deg_to_rad(EL_VARY_DEG) * sev, r)
	_roll_walk = _walk(_roll_walk, ROLL_VARY * sev, r)
	# A gentler camera also frames more evenly - the widest and tightest shots in the range
	# are part of how restless it reads, not a separate taste.
	var mid := (FILL.x + FILL.y) * 0.5
	var half := (FILL.y - FILL.x) * 0.5 * sev
	return {
		"aim": _spread.panel_center(i).lerp(_page_center(i), AIM_PULL),
		"az": _az_base + _az_walk,
		# Elevation off the PAPER, never square-on: at 90 degrees the perspective camera and
		# the page agree exactly and the panel is a flat rectangle again.
		"el": clampf(_el_base + _el_walk,
			deg_to_rad(EL_MIN_DEG), deg_to_rad(EL_MAX_DEG)),
		"roll": _roll_base + _roll_walk,
		"fill": r.randf_range(mid - half, mid + half),
		# ONE LENS PER SPREAD. A production picks a lens and covers the scene with it;
		# changing focal length every shot is a thing that reads as a mistake even to
		# someone who could not name what changed.
		"fov": _lens_fov,
	}


## The middle of the page panel [param i] is printed on - what an aim is pulled toward, and
## what a swoop is pushed away from. See AIM_PULL for why this is the page and not the sheet.
func _page_center(i: int) -> Vector2:
	return Vector2(float(_spread.side_of(i)) + 0.5, _spread.aspect * 0.5)


## One step of a bounded random walk: move [param at] by at most ANG_STEP of [param arc] and
## keep it inside +/-arc. A zero arc pins it at zero, which is what severity 0 asks for.
##
## Clamped rather than reflected, deliberately. A walk that bounces off its bound reverses
## direction on contact, which reads as the camera being repelled by an invisible wall; one
## that saturates simply dwells at the edge of the arc until the next step happens to come
## back, which reads as a camera that has settled on one side.
func _walk(at: float, arc: float, r: RandomNumberGenerator) -> float:
	if arc <= 0.0:
		return 0.0
	return clampf(at + r.randf_range(-arc, arc) * ANG_STEP, -arc, arc)


## HOW MUCH PANEL IS UNDER [param aim], 0 to 1. See CONTENT_CORE.
##
## The MAXIMUM over panels rather than a sum: two panels never overlap, so at most one can
## contain the aim, and taking the max means a point just outside a big panel scores by its
## distance from THAT panel rather than being diluted by the eleven it is nowhere near.
func _content_at(aim: Vector2) -> float:
	if _spread == null:
		return 1.0
	var best := 0.0
	for i in _spread.panels.size():
		var q := _spread.panel_uv(i, aim)
		var d := maxf(absf(q.x - 0.5), absf(q.y - 0.5)) * 2.0
		var s := 1.0 - smoothstep(CONTENT_CORE, CONTENT_EDGE, d)
		if s > best:
			best = s
			if best > 0.999:
				break
	return best


## WHICH PANEL SETS THE FRAMING SCALE: the one the aim is most inside, and only failing that
## the one whose centre is nearest.
##
## _nearest_panel alone was wrong and measurably so. It compares CENTRES, so an aim sitting
## squarely inside a wide panel can be nearer to the centre of the small panel beside it -
## and then _fit sizes the shot to the small panel, which is a framing computed for
## something that is not in the middle of the picture.
func _framing_panel(aim: Vector2) -> int:
	var best := -1
	var best_s := 0.0
	for i in _spread.panels.size():
		var q := _spread.panel_uv(i, aim)
		var d := maxf(absf(q.x - 0.5), absf(q.y - 0.5)) * 2.0
		var s := 1.0 - smoothstep(CONTENT_CORE, CONTENT_EDGE, d)
		if s > best_s:
			best_s = s
			best = i
	return best if best >= 0 else _nearest_panel(aim)


## Walk an aim in toward its panel until it is far enough inside to be worth stopping on.
##
## PULLED, NOT SNAPPED, and only as far as it has to go - see SETTLE_CONTENT. A move that
## already ends well inside a panel is returned untouched, so this costs the vocabulary
## nothing; a move that ends on a border is walked in until the shot would be of the panel
## rather than of the rule around it.
func _settle_aim(aim: Vector2) -> Vector2:
	if _spread == null:
		return aim
	var i := _framing_panel(aim)
	if i < 0:
		return aim
	var c: Vector2 = _spread.panel_center(i)
	for _step in 8:
		if _content_at(aim) >= SETTLE_CONTENT:
			break
		aim = aim.lerp(c, 0.35)
	return aim


## THE PANELS A SWEEP MAY TRAVEL BETWEEN: the leftmost and rightmost panel centres that
## share a row with `panel`, so a horizontal pan starts on content and ends on content.
## Falls back to the panel itself when it is alone on its row - a sweep with nowhere to go
## becomes a shot that sits, which is better than one that pans across the margin.
##
## ON A SPREAD THIS CROSSES THE SPINE WHENEVER THE ROWS LINE UP, for free: spread
## coordinates run 0..2 and this scans every panel of both pages, so a row band that catches
## panels on each side returns a span that is most of the sheet wide. That is the whole
## "greater spans of content to pan the camera across".
func _row_span(panel: int) -> Vector2:
	var c := _spread.panel_center(panel)
	var lo := c.x
	var hi := c.x
	for i in _spread.panels.size():
		var o := _spread.panel_center(i)
		if absf(o.y - c.y) < _spread.aspect * 0.12:
			lo = minf(lo, o.x)
			hi = maxf(hi, o.x)
	return Vector2(lo, hi) if hi - lo > 0.02 else Vector2(c.x, c.x)


## The same down a column. The tolerance is in one PAGE's width (the spread is two), so a
## column is a column of one page and not a diagonal across the spine.
func _col_span(panel: int) -> Vector2:
	var c := _spread.panel_center(panel)
	var lo := c.y
	var hi := c.y
	for i in _spread.panels.size():
		var o := _spread.panel_center(i)
		if absf(o.x - c.x) < 0.12:
			lo = minf(lo, o.y)
			hi = maxf(hi, o.y)
	return Vector2(lo, hi) if hi - lo > 0.02 else Vector2(c.y, c.y)


## The panel of page [param side] whose centre is nearest the row [param y] - the two ends
## of a `spine` traverse. Returns -1 when that page has nothing (which cannot happen, but a
## caller that indexes -1 is a crash and a caller that checks is not).
func _panel_on_side(side: int, y: float) -> int:
	var best := -1
	var best_d := INF
	for i in _spread.panels.size():
		if _spread.side_of(i) != side:
			continue
		var d := absf(_spread.panel_center(i).y - y)
		if d < best_d:
			best_d = d
			best = i
	return best


## Pick and plan the next move. Called on every reading advance and on every page turn.
func _choose_move() -> void:
	_chain = 0
	_plan_move(hash([Director.session_seed(), "comic-move", _spread_i, _read]), _read, "")


## THE MOVE RAN OUT AND THE DIRECTOR HAS NOT CUT YET.
##
## This is the other half of the report - "then just freeze there for a long time" - and it
## was not a tuning problem, it was a missing state. _ease clamps its progress at 1, so once
## a move's duration elapsed the camera sat at exactly the pose the move ended in, and it
## sat there until the next cut. The arithmetic is stark: the Director holds a scene between
## its minimum and its maximum (7 s and 28 s at the calm reference, scaled by the pacing),
## and the longest move in the bag is 20 s while the most common is 8 to 15. A drift that
## rolled 8 s against a 28 s hold is TWENTY SECONDS of a completely motionless picture.
##
## So a finished move now plans its successor from wherever it actually ended, and the
## camera never arrives at all. The successor is always a GENTLE one (see the `chain` flag):
## a jump is something a cut does, and a camera that jumps with no cut behind it reads as a
## glitch rather than as an edit.
##
## WHERE IT GOES is the reading's business, not the roll's. If the camera has drifted off
## the panels it goes back to the one being read - that is a recovery and it is the common
## case after an `open` move. Otherwise it moves to a NEIGHBOUR in reading order, preferring
## the one ahead, so the camera is already travelling toward the panel the next cut will
## land on and the cut arrives as a correction rather than as a leap.
func _chain_move() -> void:
	if _spread == null:
		return
	_chain += 1
	var key := hash([Director.session_seed(), "comic-chain", _spread_i, _read, _chain])
	var r := RandomNumberGenerator.new()
	r.seed = key
	var to := _read
	if _content_at(_clamp_aim(_cam.aim)) >= 0.35:
		# On content already, so this is not a recovery. Look AHEAD - to the panel the next
		# cut will land on - and never back.
		#
		# IT USED TO GO BACKWARD 30% OF THE TIME, and that is the whole of "it seems like the
		# camera is able to REVISIT the same frame multiple times - panning away, then
		# returning back to a frame it has already focused on". A comic is read once through;
		# a camera that returns to a panel it has already rested on is not re-reading it, it
		# has simply run out of ideas. Passing OVER an old panel on the way somewhere is fine
		# and happens constantly - that is a pan, not a visit.
		if _step + 1 < _plan.size():
			to = int(_plan[_step + 1])
	_plan_move(key, to, "chain")


## Plan a move to [param panel], seeded from [param key]. [param tag] is empty for a cut's
## move and "chain" for a continuation, which is both the log line and the filter on the
## bag - see the `chain` flag in MOVES.
func _plan_move(key: int, panel: int, tag: String) -> void:
	if _spread == null:
		return
	var r := RandomNumberGenerator.new()
	r.seed = key
	var kind := _pick_move(r, tag == "chain")
	var spec: Dictionary = MOVES[kind]
	var b := _station(r, panel)
	# CONTINUITY BY DEFAULT: a move starts from wherever the camera actually is, so the
	# picture never jumps unless the move is one whose whole point is that it jumps.
	#
	# AND NOTHING BELOW MAY OVERWRITE a["aim"] EXCEPT A DECLARED-HARD MOVE. Four of them did:
	# sweep_h and sweep_v snapped the aim to the end of a row or column, and orbit and pull
	# snapped it onto the target panel, all while declaring hard: false. Since _mv_t restarts
	# at zero, the first frame of such a move teleports the picture - measured at a mean of
	# 0.48 to 0.62 page widths, on 100% of orbits and pulls, taking the real jump-cut rate to
	# 38.7% of shots against the "about one shot in six" this table claims. They now aim
	# their DESTINATION instead, which keeps every gesture and starts all of them from the
	# picture that is already on screen.
	var a: Dictionary = b.duplicate() if bool(spec.get("hard", false)) else _cam.duplicate()
	var row: float = _spread.panel_center(panel).y
	var col: float = _spread.panel_center(panel).x

	match kind:
		"track":
			# A constant station passing THROUGH the subject and out the other side - the
			# shot that follows a car down a road. The direction is where the camera has
			# been travelling from, so the reading keeps its momentum.
			var away: Vector2 = (b.aim - a.aim)
			if away.length() < 0.05:
				away = Vector2(1.0, 0.0).rotated(r.randf_range(-0.5, 0.5))
			b["aim"] = b.aim + away.normalized() * TRACK_OVERRUN
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"spine":
			# ACROSS THE SPINE. A panel on one page to a panel on the other at the same
			# height, station held, so it is a pure lateral pan over the longest run of
			# uninterrupted content the sheet has. Which way it goes follows which page the
			# aim is on, so the pan moves AWAY from where the camera already is.
			var far_side := 1 if float(a.aim.x) < ComicSpread.SPINE else 0
			var to_i := _panel_on_side(far_side, row)
			if to_i >= 0:
				b["aim"] = _spread.panel_center(to_i)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"sweep_h":
			# ACROSS THE ROW, PANEL TO PANEL - not across the SHEET. It used to run from
			# -0.10 to 1.10 in page coordinates, which is a pan that begins and ends on the
			# desk and spends much of its length over margin and gutter: "tracking along
			# unfocused edges". A sweep is a pan over CONTENT or it is nothing.
			var span_h := _row_span(panel)
			b["aim"] = Vector2(span_h.y if absf(span_h.y - float(a.aim.x))
				>= absf(span_h.x - float(a.aim.x)) else span_h.x, row)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"sweep_v":
			var span_v := _col_span(panel)
			b["aim"] = Vector2(col, span_v.y if absf(span_v.y - float(a.aim.y))
				>= absf(span_v.x - float(a.aim.y)) else span_v.x)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"push":
			# IT ENDS ON THE PANEL. It used to end at a.aim.lerp(b.aim, 0.5) - the MIDPOINT
			# between wherever the camera happened to be and the panel it was going to -
			# while pushing in to between 2.2 and 3.8 frames of panel height. A midpoint
			# between two panels is a gutter, and a gutter at 3.8 fill is the reported
			# defect exactly: "in the corner of a frame and fully zoomed-in... shows almost
			# nothing". A push-in ends ON its subject. That is what a push-in is.
			# ...and the other one. A push to 3.8 frames of panel height is the tightest the
			# camera ever gets; at a gentle setting it barely pushes past the panel at all.
			b["fill"] = lerpf(1.0, r.randf_range(PUSH_FILL.x, PUSH_FILL.y), _sev() * 0.5)
			for k in ["az", "el", "roll"]:
				b[k] = a[k]
		"pull":
			a["fill"] = lerpf(1.0, r.randf_range(PUSH_FILL.x, PUSH_FILL.y), _sev() * 0.5)
		"orbit":
			# The widest gesture in the bag - up to 149 degrees at the default - so it is one
			# of the two the severity knob pulls in hardest.
			b["az"] = a.az + r.randf_range(ORBIT_ARC.x, ORBIT_ARC.y) * _sev() \
				* (1.0 if r.randf() < 0.5 else -1.0)
			for k in ["el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"swoop":
			# In off the FAR SIDE of the spread, low and wide, arriving on the subject.
			a["az"] = b.az + PI
			a["el"] = deg_to_rad(r.randf_range(SWOOP_EL_DEG.x, SWOOP_EL_DEG.y))
			a["fill"] = SWOOP_FILL
			var mid := _page_center(panel)
			a["aim"] = b.aim + (b.aim - mid).normalized() * 0.5 \
				if b.aim.distance_to(mid) > 0.02 else b.aim

	# THE SETTLE GUARD, and it is a guard rather than a rule so that it survives whatever
	# gets added to the bag next. Any move that is allowed to STOP must stop somewhere worth
	# looking at; the ones marked `open` are exempt because running past the subject is
	# their whole gesture, and _chain_move is what makes that safe.
	if not bool(spec.get("open", false)):
		b["aim"] = _settle_aim(_clamp_aim(b.aim))

	_mv = {
		"kind": kind,
		"a": a,
		"b": b,
		# SEVERITY SETS THE SPEED. A short move is a fast move, so scaling the duration is
		# most of what "gentle" and "cinematic" mean. Exponential about 1.0 - see DUR_SEVERITY.
		"dur": maxf(0.05, r.randf_range(float(spec["dur"][0]), float(spec["dur"][1]))
			* pow(DUR_SEVERITY, _sev() - 1.0)),
		"ease": String(spec["ease"]),
	}
	_mv_t = 0.0
	if bool(spec.get("hard", false)):
		_cam = a.duplicate()          # the jump itself
	_dip_t = 0.0 if kind == "dip" else -1.0
	if _spread_i >= 0:
		print("ghost: comic %s%s -> panel %d of spread %d" % [
			kind, (" (chained)" if tag == "chain" else ""), panel + 1, _spread_i])


func _pick_move(r: RandomNumberGenerator, chain_only: bool) -> String:
	var total := 0.0
	for k in MOVES:
		if chain_only and not bool(MOVES[k].get("chain", false)):
			continue
		total += _weight_of(String(k))
	if total <= 0.0:
		return "drift"
	var pick := r.randf() * total
	for k in MOVES:
		if chain_only and not bool(MOVES[k].get("chain", false)):
			continue
		pick -= _weight_of(String(k))
		if pick <= 0.0:
			return String(k)
	return "drift"


## A move's weight in the bag AT THE CURRENT SEVERITY. The gentle ones keep the weight the
## table gives them; the ones that begin somewhere else entirely scale by severity squared,
## so at 0 the camera never jump-cuts at all and at the top of the range punctuation is four
## times as common. See HARD_SEVERITY.
func _weight_of(kind: String) -> float:
	var spec: Dictionary = MOVES[kind]
	var w := float(spec["w"])
	if bool(spec.get("hard", false)) or kind == "whip":
		return w * pow(_sev(), HARD_SEVERITY)
	return w


func _ease(delta: float) -> void:
	# The spread turns under the camera, slowly and forever - independent of the move,
	# because the sheet is scenery and the camera is the performance.
	_att_target += _att_rate * delta
	var sway := Vector3(
		_mod.value("tilt") * 0.09,
		_mod.value("sway") * 0.12,
		_mod.value("roll") * 0.05) if _mod != null else Vector3.ZERO
	_att = _att.lerp(_att_target + sway, 1.0 - exp(-PAGE_EASE * delta))
	# ONCE PER FRAME, for every point that will be placed on the paper. See _att_basis.
	_att_basis = Basis.from_euler(_att)
	if _dip_t >= 0.0:
		_dip_t += delta
	if _mv.is_empty():
		return
	_mv_t += delta
	# THE MOVE IS OVER AND NOTHING HAS CUT. Chain, do not stop - see _chain_move. The new
	# move starts at _mv_t 0, so this is a continuation and not a reset of the picture.
	if _mv_t >= float(_mv.dur):
		_chain_move()
	var k := clampf(_mv_t / float(_mv.dur), 0.0, 1.0)
	_cam = _lerp_state(_mv.a, _mv.b, _curve(String(_mv.ease), k))
	_place_eye()


func _curve(name: String, k: float) -> float:
	match name:
		"linear": return k
		"out": return 1.0 - pow(1.0 - k, 3.0)
		"snap": return 1.0
	return smoothstep(0.0, 1.0, k)


func _lerp_state(a: Dictionary, b: Dictionary, k: float) -> Dictionary:
	return {
		"aim": (a.aim as Vector2).lerp(b.aim, k),
		# the short way round, or a wrap past TAU sends the camera the long way about
		"az": float(a.az) + wrapf(float(b.az) - float(a.az), -PI, PI) * k,
		"el": lerpf(a.el, b.el, k),
		"roll": lerpf(a.roll, b.roll, k),
		"fill": lerpf(a.fill, b.fill, k),
		"fov": lerpf(a.fov, b.fov, k),
	}


## The whole-frame alpha a `dip` is currently at: out through black, then back.
func _dip_alpha() -> float:
	if _dip_t < 0.0:
		return 1.0
	if _dip_t < DIP_OUT:
		return 1.0 - _dip_t / DIP_OUT
	if _dip_t < DIP_OUT + DIP_IN:
		return (_dip_t - DIP_OUT) / DIP_IN
	return 1.0


## Put the eye on its spread-local spherical offset from where the camera is aiming.
##
## The offset direction is built in SPREAD space and then rotated by the sheet's attitude,
## so it is a fixed station relative to the paper: travelling from panel to panel slides the
## camera ACROSS the spread at a constant angle, which is the move a copy-stand shot makes
## and the one the eye reads as flying over a comic. Building it in world space instead
## would swing the angle around every time the sheet moved.
func _place_eye() -> void:
	if _spread == null:
		return
	var aim: Vector2 = _clamp_aim(_cam.aim)
	var c := _world(aim, _att_basis)
	# A FIELD scene is raked gently, a subject hard - see FIELD_FLATTEN.
	var el: float = lerpf(float(_cam.el), PI * 0.5, _flatten())
	var roll: float = float(_cam.roll) * (1.0 - _flatten() * 0.7)
	_roll = roll
	_fov = float(_cam.fov)
	var dir := _att_basis * Vector3(cos(_cam.az) * cos(el), sin(_cam.az) * cos(el), sin(el))
	# SCALE IS SET BY WHICHEVER PANEL THE AIM IS INSIDE, not by the nearest centre and not by
	# the one being read: a travelling move spends most of its time between panels, and
	# framing those seconds against a panel that is off screen puts the scale somewhere
	# arbitrary. See _framing_panel.
	var pw := _panel_world(_framing_panel(aim))
	_cam["aim"] = aim
	# ZOOM IS COUPLED TO CONTENT - see CONTENT_CORE. Off a panel the requested fill is
	# ignored and the shot opens to OFF_FILL, so the camera physically cannot be tight on a
	# gutter or a corner however a move was planned or wherever a chain left it.
	var eff := lerpf(OFF_FILL, float(_cam.fill), _content_at(aim))
	# TWO CONSTRAINTS, and the nearer wins.
	#
	# The first frames the PANEL: near enough that it spans `eff` frames top to bottom. The
	# second frames the SPREAD: near enough that the sheet covers the frame on BOTH axES. A
	# single portrait page could never satisfy the second without pushing in until the page
	# stopped reading as a page, which is why the first cut of this only ever checked the
	# width; a spread is 1.33 wide against a 1.78 frame, so covering the width covers the
	# height with room to spare and the check can finally be honest.
	var d := minf(_fit(pw, c, eff, dir), _cover(_spread_world(), c, dir))
	# ...and never nearer than this, whatever the two constraints ask for. Pushing in until
	# the paper covers the frame is right; pushing in until the camera is INSIDE one panel's
	# artwork is a shot of a texture, with no page, no gutter and no comic in it.
	d = maxf(d, _fit(pw, c, CROP_MAX, dir))
	_eye = c + dir * d
	_look = c


## How far toward square-on the rake is pulled for the panel currently being read. Zero for
## a subject; FIELD_FLATTEN for a field. See that constant.
func _flatten() -> float:
	var sc := _focal()
	return FIELD_FLATTEN if (sc != null and sc.framing == "field") else 0.0


## Keep an aim on the paper - see AIM_MARGIN. Spread coordinates, so x runs to 2.
func _clamp_aim(p: Vector2) -> Vector2:
	return Vector2(
		clampf(p.x, AIM_MARGIN, 2.0 - AIM_MARGIN),
		clampf(p.y, AIM_MARGIN, _spread.aspect - AIM_MARGIN))


## HOW MUCH OF THE FRAME THE SHEET COVERS. 1.0 when there is paper under all four corners
## of the picture; below that, the number is how far the WORST corner falls outside the
## sheet as a fraction of half the frame height - so 0.92 is a sliver of desk in one corner
## and 0.5 is a wedge across the shot. Read by tests/comic_look_probe.gd, which is the only
## way "is there dead space in the frame" gets checked by something other than an eye.
##
## FOUR CORNERS, not two edges of a bounding box. The old measure took the projected sheet's
## bounding box and asked whether it reached past the left and right frame edges. That is
## wrong twice over. It never looked UP or DOWN, so it reported a clean 1.00 on a frame with
## a band of black desk across the top - the exact frame the report was about. And a
## bounding box is not the sheet: a raked, rolled quad has a box that reaches every edge
## while the quad itself cuts a corner off. A number that cannot see the defect it exists to
## catch is worse than no number.
func page_coverage() -> float:
	if _spread == null:
		return 1.0
	var u := minf(_stage_size.x, _stage_size.y)
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0)
	var hy := (_stage_size.y * 0.5) / maxf(u, 1.0)
	var q := _sheet_poly(_spread_world(), _lens)
	if q.size() < 3:
		return 1.0                     # every corner behind the eye: the sheet is all around
	return clampf(1.0 + _quad_clearance(q, hx, hy) / maxf(hy, 1e-4), 0.0, 1.0)


## The sheet's outline in screen space, CLIPPED TO THE NEAR PLANE.
##
## A perspective projection cannot express a point behind the eye - it inverts through
## infinity - so every draw here bails out when one appears, and the two coverage tests used
## to bail out by declaring themselves covered. On a single page that was nearly honest: a
## page is 2 world units across, and a camera near enough to put a corner behind itself
## really was over the paper. A spread is 4 across. At the reading distance its far corners
## go behind the eye while there is still a wedge of desk in the opposite corner of the
## frame, and the escape hatch then reports a clean 1.00 on exactly the picture the coverage
## number exists to catch.
##
## So the outline is clipped instead of abandoned - Sutherland-Hodgman against the one
## plane, which turns the quad into at most a pentagon and costs one lerp per crossing. The
## result is the sheet's true silhouette, and the corner test below is then exact at any
## framing.
func _sheet_poly(pts: Array, lens: Lens3D) -> PackedVector2Array:
	var out := PackedVector2Array()
	var n := pts.size()
	for i in n:
		var a: Vector3 = pts[i]
		var b: Vector3 = pts[(i + 1) % n]
		var da := lens.depth(a) - lens.near
		var db := lens.depth(b) - lens.near
		if da >= 0.0:
			var pa := lens.project(a)
			out.append(Vector2(pa.x, pa.y))
		if (da >= 0.0) != (db >= 0.0):
			# where the edge crosses the plane, in world space, so the projection of the
			# crossing point is the projection of a point that is genuinely in front
			var t := da / (da - db)
			var pc := lens.project(a.lerp(b, t))
			out.append(Vector2(pc.x, pc.y))
	return out


## The signed distance from the WORST frame corner to the edges of the projected quad
## [param q], in unit-fraction units. Positive means every corner of the frame is inside the
## quad, i.e. the sheet covers the picture; negative is how far the worst one sticks out.
##
## The quad is convex (it is a planar rectangle under a perspective projection), so a point
## is inside exactly when it is on the same side of all four edges - and the least of those
## four signed distances is how much room there is to spare. Taking the least again over the
## four frame corners gives one number for "is there desk in shot, and how much".
##
## Wound from the quad's own signed area rather than assumed, because the page's roll and
## the camera's can put the sheet either way round on screen.
func _quad_clearance(q: PackedVector2Array, hx: float, hy: float) -> float:
	var n := q.size()
	if n < 3:
		return 0.0
	var area := 0.0
	for e in n:
		var a: Vector2 = q[e]
		var b: Vector2 = q[(e + 1) % n]
		area += a.x * b.y - b.x * a.y
	var wind := 1.0 if area >= 0.0 else -1.0
	var worst := INF
	for cx: float in [-hx, hx]:
		for cy: float in [-hy, hy]:
			for e in n:
				var a: Vector2 = q[e]
				var b: Vector2 = q[(e + 1) % n]
				var d := b - a
				var edge := d.length()
				if edge < 1e-9:
					continue
				# signed distance to the edge line, positive on the quad's inside
				var side := wind * (d.x * (cy - a.y) - d.y * (cx - a.x)) / edge
				worst = minf(worst, side)
	return 0.0 if worst == INF else worst


## Which panel the camera is over. Nearest centre in spread coordinates - the fallback for
## [method _framing_panel] when the aim is inside nothing at all.
func _nearest_panel(aim: Vector2) -> int:
	var best := 0
	var best_d := 1e9
	for i in _spread.panels.size():
		var d := aim.distance_squared_to(_spread.panel_center(i))
		if d < best_d:
			best_d = d
			best = i
	return best


func _prepare_lens() -> void:
	_lens.eye = _eye
	_lens.look = _look
	_lens.fov = _fov
	# CAMERA UP, and it must not be world up: at a hard rake the view axis can come within
	# a few degrees of it and the basis collapses (Lens3D falls back and the picture rolls
	# through 90 degrees in one frame). The sheet's own up is always perpendicular to the
	# paper's normal, so it is a stable reference - and taking it from the sheet is also what
	# makes its roll show as a tilted horizon rather than as nothing at all.
	var fwd := (_look - _eye)
	fwd = fwd.normalized() if fwd.length() > 1e-6 else Vector3.FORWARD
	var up := _att_basis * Vector3.UP
	if absf(up.dot(fwd)) > 0.9:
		up = _att_basis * Vector3.RIGHT     # grazing along the sheet's up axis
	_lens.up = Basis(fwd, _roll) * up
	_lens.prepare()


## THE FRAMING DISTANCE: how far along [param dir] the eye must sit for [param pts] to span
## [param fill] frames VERTICALLY.
##
## VERTICAL ONLY, and that is the whole difference between a comic and a slideshow of
## pages. Constraining both axes means fitting the panel ENTIRELY inside the frame, and a
## panel narrower than 16:9 - which most comic panels are - then leaves the rest of the
## picture to the page, the trim edge, and the desk beyond it. Constrained on height, the
## panel always spans the frame, and what fills the sides is the gutter and the panels next
## to it. That is what reading a comic close up actually looks like.
##
## Solved by bisection rather than in closed form, because unlike a straight-down-z camera
## the view basis turns as the distance changes, so there is no single inequality to
## rearrange. A handful of halvings on four points is nothing, and it is exact at any rake.
##
## THE HALF-EXTENT IS 0.5, NOT 1. Lens3D projects into ghost's unit-fraction space, which
## the caller multiplies by unit() = the SHORTER SCREEN AXIS - so the visible vertical range
## is +/- (H/2)/min(W,H) = +/- 0.5. Reading it as 1 puts the camera at half the distance it
## needs, which is what the first cut of this did.
func _fit(pts: Array, c: Vector3, fill: float, dir: Vector3) -> float:
	var u := minf(_stage_size.x, _stage_size.y)
	# fill is how many FRAMES tall the panel should be, so a bigger fill is a nearer
	# camera: the allowed half-extent grows and the bisection settles closer in.
	var hy := (_stage_size.y * 0.5) / maxf(u, 1.0) * fill
	var probe := _fit_lens
	probe.fov = _fov
	probe.look = c
	probe.up = _lens.up
	var lo := 0.15
	var hi := 24.0
	for _iter in SOLVE_STEPS:
		var mid := (lo + hi) * 0.5
		probe.eye = c + dir * mid
		probe.prepare()
		var fits := true
		for pt in pts:
			var pr := probe.project(pt)
			if pr.z <= probe.near or absf(pr.y) > hy:
				fits = false
				break
		if fits:
			hi = mid
		else:
			lo = mid
	return hi


## How many halvings the two solvers take. Sixteen resolves the 0.15-24 bracket to about
## 0.0004 world units, which is a third of a thousandth of a page width - far past anything
## a camera distance can show, and six fewer projections per point than the 22 it was.
const SOLVE_STEPS := 16


## THE COVERING DISTANCE: the FARTHEST the eye may sit along [param dir] while [param pts]
## still span the whole frame, on BOTH axes. Beyond it you can see past them.
##
## The mirror of [method _fit], and it has to bisect the other way: a projection shrinks
## with distance, so "fits inside" is true beyond some distance and "covers" is true within
## one. Same halvings, same exactness at any rake.
func _cover(pts: Array, c: Vector3, dir: Vector3) -> float:
	var u := minf(_stage_size.x, _stage_size.y)
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0)
	var hy := (_stage_size.y * 0.5) / maxf(u, 1.0)
	var probe := _cover_lens
	probe.fov = _fov
	probe.look = c
	probe.up = _lens.up
	var lo := 0.15
	var hi := 24.0
	for _iter in SOLVE_STEPS:
		var mid := (lo + hi) * 0.5
		probe.eye = c + dir * mid
		probe.prepare()
		# CLIPPED, not abandoned, when a corner goes behind the eye - see _sheet_poly. An
		# empty polygon means every corner is behind, which is the most covered case there
		# is; reading THAT as "does not cover" inverts the whole search and drives the
		# camera inside a panel.
		var q := _sheet_poly(pts, probe)
		# Covers when every CORNER OF THE FRAME has paper under it - the same test
		# page_coverage() reports, so the solve and the measurement can never disagree about
		# what "covered" means. See _quad_clearance for why a bounding box is not enough.
		if q.size() < 3 or _quad_clearance(q, hx, hy) >= 0.0:
			lo = mid
		else:
			hi = mid
	return lo


## The four world corners of one panel, cant and spread attitude applied.
func _panel_world(i: int) -> Array:
	var out: Array = []
	for uv in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		out.append(_world(_spread.panel_point(i, uv), _att_basis))
	return out


## The four world corners of the whole spread.
func _spread_world() -> Array:
	var out: Array = []
	for p in _spread.spread_corners():
		out.append(_world(p, _att_basis))
	return out


## Panel [param i]'s bounding box on screen, for the visibility test.
func _panel_rect(i: int) -> Rect2:
	var u := minf(_stage_size.x, _stage_size.y)
	var origin := _stage_size * 0.5
	var r := Rect2()
	var first := true
	for w in _panel_world(i):
		var pr := _lens.project(w)
		if pr.z <= _lens.near:
			return Rect2(Vector2.ZERO, _stage_size)    # partly behind the eye: assume on screen
		var p := Vector2(pr.x, pr.y) * u + origin
		if first:
			r = Rect2(p, Vector2.ZERO)
			first = false
		else:
			r = r.expand(p)
	return r


## A point in SPREAD space (x 0..2 with the spine at 1, y 0..aspect and DOWN) -> world
## space. Everything drawn on the sheet goes through this one call, so the paper, the
## panels, the gutters and the ink can never disagree about where the spread is.
##
## [param m] IS THE WHOLE TRANSFORM, handed in already built: the spread's attitude for a
## flat page, or the attitude times the leaf's hinge rotation for a turning one. It used to
## build Basis.from_euler(att) here, per vertex - six trig calls and a matrix, roughly 2500
## times a frame with an argument that had not changed - and AUDIT.md attributes most of the
## page's cost to exactly that. Hoisting it is not an approximation; the value was always
## identical.
##
## [param mirror] reflects the point about the spine before placing it, which is what draws
## the BACK of a turning leaf. See [method ComicSpread.mirror].
func _world(p: Vector2, m: Basis, aspect := -1.0, mirror := false, depth := 0.0) -> Vector3:
	var a := aspect if aspect > 0.0 else _spread.aspect
	var x := (2.0 * ComicSpread.SPINE - p.x) if mirror else p.x
	# THE TURN HINGES ON THE SPINE, and on a spread the spine is the origin - so unlike the
	# single page, which had to rotate about an off-centre pivot to avoid reading as a card
	# spinning in place, there is no pivot term here at all. The leaf's rotation is simply
	# part of `m`.
	return m * Vector3(
		(x - ComicSpread.SPINE) * PAGE_W,
		-(p.y - a * 0.5) * PAGE_W,        # spread y is DOWN, world y is up
		-depth * PAGE_W)                  # behind the sheet, along its own normal


# --- drawing -----------------------------------------------------------------

func _draw() -> void:
	if _spread == null:
		return
	_lens.eye = _eye
	_lens.look = _look
	_lens.fov = _fov
	_lens.prepare()
	var u := minf(_stage_size.x, _stage_size.y)
	var origin := _stage_size * 0.5
	# The bookend belongs to the vehicle here (see Vehicle.owns_bookend): the whole spread
	# fades, paper and all, instead of one panel fading inside a lit page.
	modulate.a = 1.0
	var fade := clampf(_bookend, 0.0, 1.0) * _dip_alpha()
	_backdrop(fade)          # once per frame, under everything, including through a turn

	# A PAGE TURN, as a real leaf on a real spine.
	#
	# The right-hand page of the outgoing spread swings left about the spine. Its FRONT is
	# that page; past the halfway point - where the leaf is edge-on and infinitely thin, so
	# the swap cannot be seen - its BACK is the incoming spread's LEFT page, mirrored about
	# the spine so that at a half turn it lands exactly where that page is about to be drawn
	# flat. Nothing cross-dissolves and nothing jumps: the leaf simply becomes the page it
	# always was the other side of.
	#
	# Under it: the incoming RIGHT page, revealed as the leaf lifts off it, and the outgoing
	# LEFT page, covered as the leaf comes down on it. Painter's order is exactly that -
	# both flat pages first, the leaf over them.
	if _turn_t >= 0.0 and _prev != null:
		var k := smoothstep(0.0, 1.0, clampf(_turn_t / TURN_TIME, 0.0, 1.0))
		var th := k * TURN_ARC
		_draw_leaf(_spread, 1, _pool, _is_cast, _att_basis, u, origin, fade, false)
		_draw_leaf(_prev, 0, 1 - _pool, _all_drawn, _att_basis, u, origin, fade, false)
		var hinge := _att_basis * Basis(Vector3.UP, -th)
		if th < TURN_ARC * 0.5:
			_draw_leaf(_prev, 1, 1 - _pool, _all_drawn, hinge, u, origin, fade, false)
		else:
			_draw_leaf(_spread, 0, _pool, _is_cast, hinge, u, origin, fade, true)
		return
	_draw_leaf(_spread, 0, _pool, _is_cast, _att_basis, u, origin, fade, false)
	_draw_leaf(_spread, 1, _pool, _is_cast, _att_basis, u, origin, fade, false)


## Has panel [param i] of the CURRENT spread been cast yet? (See _turn_spread's queue.)
func _is_cast(i: int) -> bool:
	return i < _cast.size() and _cast[i] != null and is_instance_valid(_cast[i])


## Every panel of the spread turning away carries a picture - it was fully cast before the
## turn started, and stopping its viewports kept what they had drawn.
func _all_drawn(_i: int) -> bool:
	return true


## ONE PAGE of [param sp] - the unit a leaf turns, and therefore the unit everything draws
## in. [param side] is 0 for the left page and 1 for the right.
##
## [param m] is the full placement transform (see _world): the sheet's attitude for a page
## lying flat, or that times the hinge rotation for the leaf in the air. [param mirror]
## reflects about the spine, which is how the leaf's back face draws the page printed on it.
##
## [param drawn] says which panels carry a picture. For the spread being read that is "has
## this one been cast yet" - the queue casts one per frame, so for the first few frames a
## late panel is still blank paper. For the spread turning away it is every panel, because
## it was fully cast before the turn began.
func _draw_leaf(sp: ComicSpread, side: int, pool: int, drawn: Callable, m: Basis,
		u: float, origin: Vector2, fade: float, mirror: bool) -> void:
	if fade <= 0.003 or sp == null:
		return
	var paper := Color(_paper.r, _paper.g, _paper.b, fade)
	var ink := Color(_ink.r, _ink.g, _ink.b, fade)
	# The sheet's shadow on the surface under it, before the sheet itself.
	_shadow_quad(sp, side, m, u, origin, fade, mirror)
	# THE PAPER, subdivided so its gradient is per-vertex rather than per-quad, and so a
	# steeply raked page still shades smoothly across its length.
	_paper_quad(sp, side, m, u, origin, paper, mirror)
	# THE PANELS. Painter's order along the page normal is irrelevant - they are coplanar
	# and never overlap - so reading order is the order, which is also the order the ink
	# wants (a canted panel's border must sit over its neighbour's paper, not under it).
	var on_page := sp.page_panels(side)
	for i in on_page:
		if bool(drawn.call(i)):
			_panel_quad(sp, int(i), pool, m, u, origin, fade, mirror)
	for i in on_page:
		_panel_ink(sp, int(i), m, u, origin, paper, ink, bool(drawn.call(i)), mirror)


## The surface the sheet lies on: a vignetted wash over the whole frame, painted first.
## See the note on DESK_VIGNETTE for why this is not a quad in the world.
func _backdrop(fade: float) -> void:
	var pts := PackedVector2Array()
	var cols := PackedColorArray()
	var idx := PackedInt32Array()
	var n := 6
	for j in n + 1:
		for i in n + 1:
			var fu := float(i) / float(n)
			var fv := float(j) / float(n)
			pts.append(Vector2(fu * _stage_size.x, fv * _stage_size.y))
			var d := Vector2(fu - 0.5, fv - 0.5).length() / 0.707
			var k := lerpf(1.0, 1.0 - DESK_VIGNETTE, clampf(d * d, 0.0, 1.0))
			cols.append(Color(_desk.r * k, _desk.g * k, _desk.b * k, fade))
	for j in n:
		for i in n:
			var a := j * (n + 1) + i
			idx.append_array([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1])
	RenderingServer.canvas_item_add_triangle_array(get_canvas_item(), idx, pts, cols)


## One page's shadow on that surface - its outline, offset and pushed back. It is most of
## what makes the paper read as an object lying on something rather than as a rectangle
## floating in a void, and on the leaf it is what says the page has LIFTED.
func _shadow_quad(sp: ComicSpread, side: int, m: Basis, u: float, origin: Vector2,
		fade: float, mirror: bool) -> void:
	var poly := PackedVector2Array()
	for p in sp.page_corners(side):
		var q: Vector2 = p + Vector2(SHADOW_OFF.x, SHADOW_OFF.y * sp.aspect)
		var pr := _lens.project(_world(q, m, sp.aspect, mirror, SHADOW_DEPTH))
		if pr.z <= _lens.near:
			return
		poly.append(Vector2(pr.x, pr.y) * u + origin)
	if _poly_area(poly) < 1.0:
		return
	draw_colored_polygon(poly, Color(0.0, 0.0, 0.0, 0.42 * fade))


## The page itself: a subdivided quad with a gentle darkening toward the edges. Flat white
## paper reads as a blank canvas; a whisper of a gradient reads as a printed sheet.
func _paper_quad(sp: ComicSpread, side: int, m: Basis, u: float, origin: Vector2,
		paper: Color, mirror: bool) -> void:
	var pts := PackedVector2Array()
	var cols := PackedColorArray()
	var ok := []
	var n := PAPER_GRID
	var x0 := float(side)
	for j in n + 1:
		for i in n + 1:
			var fu := float(i) / float(n)
			var fv := float(j) / float(n)
			var pr := _lens.project(_world(Vector2(x0 + fu, fv * sp.aspect), m,
				sp.aspect, mirror))
			var good := pr.z > _lens.near
			ok.append(good)
			pts.append((Vector2(pr.x, pr.y) * u + origin) if good else Vector2.ZERO)
			# radial falloff from the sheet's centre, at a few percent
			var d := Vector2(fu - 0.5, fv - 0.5).length() / 0.707
			var kk := 1.0 - 0.10 * d * d
			cols.append(Color(paper.r * kk, paper.g * kk, paper.b * kk, paper.a))
	var idx := _cells(ok, n)
	if idx.is_empty():
		return
	RenderingServer.canvas_item_add_triangle_array(get_canvas_item(), idx, pts, cols)


## Index the cells of an (n+1)x(n+1) vertex grid, SKIPPING any cell with a vertex behind the
## eye.
##
## The alternative - what every one of these draws used to do - is to abandon the whole
## surface the moment one vertex went behind the camera. On a single page that was rare
## enough to be invisible. On a spread it is not: the sheet is twice as wide, so at a rake
## the far corners go behind the eye at framings that are otherwise perfectly good, and a
## page that vanishes entirely for a second is a far worse artefact than a page missing the
## cell that could not be projected.
## RETURNS the buffer rather than filling one handed in: a PackedInt32Array is a value
## type in GDScript, so a callee that appends to a parameter appends to its own copy and the
## caller draws nothing at all.
func _cells(ok: Array, n: int) -> PackedInt32Array:
	var idx := PackedInt32Array()
	for j in n:
		for i in n:
			var a := j * (n + 1) + i
			var b := a + 1
			var c := a + n + 1
			var d := a + n + 2
			if bool(ok[a]) and bool(ok[b]) and bool(ok[c]) and bool(ok[d]):
				idx.append_array([a, b, d, a, d, c])
	return idx


## One panel's picture: its render target mapped onto a GRID-subdivided quad whose every
## vertex is projected individually. Two triangles would be affine and would warp (28.5 px
## on a 512 px frame, measured in tests/vehicle_probe.gd); subdivision makes the mapping
## piecewise-perspective and the error vanishes.
func _panel_quad(sp: ComicSpread, i: int, pool: int, m: Basis,
		u: float, origin: Vector2, fade: float, mirror: bool) -> void:
	var vp: SubViewport = _slots[pool * POOL + i]
	var rid := vp.get_texture().get_rid()
	if not rid.is_valid():
		return
	# HALF A TEXEL IN, on every side. UVs of exactly 0 and 1 sample the outermost texel's
	# EDGE, where the bilinear filter reaches for a neighbour that is not there - and what
	# it finds depends on the repeat mode, which showed as a stray coloured fringe along
	# the bottom rule of a panel. Insetting by half a texel keeps every sample inside the
	# panel's own picture; the loss is half a pixel of a 700-pixel render.
	var tw := maxf(float(vp.size.x), 1.0)
	var th := maxf(float(vp.size.y), 1.0)
	var inset := Vector2(0.5 / tw, 0.5 / th)
	var n := _grid_for(sp, i, m, u, origin, mirror)
	var pts := PackedVector2Array()
	var uvs := PackedVector2Array()
	var cols := PackedColorArray()
	var ok := []
	var col := Color(1, 1, 1, fade)
	for jj in n + 1:
		for ii in n + 1:
			var uv := Vector2(float(ii) / float(n), float(jj) / float(n))
			var pr := _lens.project(_world(sp.panel_point(i, uv), m, sp.aspect, mirror))
			var good := pr.z > _lens.near
			ok.append(good)
			pts.append((Vector2(pr.x, pr.y) * u + origin) if good else Vector2.ZERO)
			# THE UV IS NOT MIRRORED. Reflecting the page about the spine reflects where the
			# panel IS, never what is printed in it - a page seen from behind would be a
			# page seen from behind, and the leaf's back is a page seen from the FRONT.
			uvs.append(inset + uv * (Vector2.ONE - inset * 2.0))
			cols.append(col)
	var idx := _cells(ok, n)
	if idx.is_empty():
		return
	RenderingServer.canvas_item_add_triangle_array(
		get_canvas_item(), idx, pts, cols, uvs,
		PackedInt32Array(), PackedFloat32Array(), rid)


## How finely to subdivide this panel's grid: enough cells that each is about CELL_PX across
## on screen. Measured off the panel's own projected size, so a panel filling the frame gets
## a fine grid and one at the edge of shot gets a coarse one.
func _grid_for(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		mirror: bool) -> int:
	var lo := Vector2(1e9, 1e9)
	var hi := Vector2(-1e9, -1e9)
	var wraps := false
	for uv in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		var pr := _lens.project(_world(sp.panel_point(i, uv), m, sp.aspect, mirror))
		if pr.z <= _lens.near:
			wraps = true
			continue
		var p := Vector2(pr.x, pr.y) * u + origin
		lo = lo.min(p)
		hi = hi.max(p)
	if wraps:
		# A corner behind the eye means the panel is unboundedly large in screen space, so
		# the projected span says nothing. Size it to the SCREEN instead: a cell can never
		# usefully be finer than the frame it is drawn into, and the old answer here was a
		# flat GRID_MAX - a 40x40 grid, 1681 vertices, on a panel that is mostly off shot.
		return clampi(int(ceil(maxf(_stage_size.x, _stage_size.y) / CELL_PX)),
			GRID_MIN, GRID_MAX)
	var span := maxf(hi.x - lo.x, hi.y - lo.y)
	return clampi(int(ceil(span / CELL_PX)), GRID_MIN, GRID_MAX)


## The gutter side of a panel: paper painted back over the square corners the rounded
## outline cuts off, then the ink border on top.
##
## Rounding this way rather than by clipping the texture is exact under perspective and
## costs nothing: the corner wedges are computed in SPREAD space and projected like every
## other point on the sheet, so they land on the picture's corner however the page is
## raked. Clipping would have needed a stencil or a shader, for a shape the paper can
## simply cover.
func _panel_ink(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		paper: Color, ink: Color, filled: bool, mirror: bool) -> void:
	if sp.radius > 0.0004 and filled:
		for w in _corner_wedges(sp, i, m, u, origin, mirror):
			var poly: PackedVector2Array = w
			# A wedge is a few pixels across at the wide shot, and BELOW a pixel the canvas
			# triangulator rejects it outright ("Invalid polygon data, triangulation
			# failed", once per corner per panel per frame). Same guard Plane3D carries for
			# the same reason: sub-pixel geometry contributes nothing and costs an error.
			if poly.size() >= 3 and _poly_area(poly) >= 1.0:
				draw_colored_polygon(poly, paper)
	# An UNFILLED panel is drawn as an empty ruled frame - the page is a comic being
	# drawn, and the panels ahead of the story are simply not inked in yet. Lighter, so
	# it reads as ruled rather than as a black hole.
	var line := ink if filled else Color(ink.r, ink.g, ink.b, ink.a * 0.35)
	# THE BORDER IS DRAWN IN RUNS. A rule with one point behind the eye used to abandon the
	# whole outline; on a spread that happens often enough to be seen as panels losing their
	# borders in bursts, so the visible part is drawn and only the part that cannot be
	# projected is dropped.
	for run in _rounded_runs(sp, i, m, u, origin, mirror):
		var pts: PackedVector2Array = run
		if pts.size() >= 2:
			draw_polyline(pts, line, _ink_width(), true)


## Comic rules are CHUNKY - a hairline reads as a UI border around a video, which is
## exactly what this vehicle exists not to be. Scaled off the shorter screen axis so it
## is the same weight at 720p and at 4K, and sampled per session inside a narrow band
## (some books ink heavier than others; none ink thin).
## Twice the signed area of a polygon, absolute - the cheap "is this bigger than a pixel"
## test.
func _poly_area(p: PackedVector2Array) -> float:
	var a := 0.0
	for i in p.size():
		var q: Vector2 = p[i]
		var r: Vector2 = p[(i + 1) % p.size()]
		a += q.x * r.y - r.x * q.y
	return absf(a) * 0.5


func _ink_width() -> float:
	return maxf(1.5, minf(_stage_size.x, _stage_size.y) * _ink_weight)


## The panel's outline in screen space as RUNS of consecutive projectable points, corners
## rounded in SPREAD space. A fully visible panel gives exactly one run, closed.
func _rounded_runs(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		mirror: bool) -> Array:
	var uv := _rounded_uv(sp, i)
	var scr := PackedVector2Array()
	var ok := []
	for p in uv:
		var pr := _lens.project(_world(sp.panel_point(i, p), m, sp.aspect, mirror))
		ok.append(pr.z > _lens.near)
		scr.append(Vector2(pr.x, pr.y) * u + origin)
	var all := true
	for g in ok:
		if not bool(g):
			all = false
			break
	if all:
		var ring := scr.duplicate()
		ring.append(scr[0])
		return [ring]
	var out: Array = []
	var run := PackedVector2Array()
	for j in scr.size():
		if bool(ok[j]):
			run.append(scr[j])
		elif run.size() >= 2:
			out.append(run)
			run = PackedVector2Array()
		else:
			run = PackedVector2Array()
	if run.size() >= 2:
		out.append(run)
	return out


## The outline as points in the panel's own unit square. Radius is expressed in units of
## the SHORTER side and converted per axis, so a wide panel's corners stay circular
## instead of stretching into ellipses.
func _rounded_uv(sp: ComicSpread, i: int) -> PackedVector2Array:
	var r: Rect2 = sp.panels[i]["rect"]
	var short := minf(r.size.x, r.size.y)
	var rad := sp.radius * short
	var out := PackedVector2Array()
	if rad <= 0.0004:
		return PackedVector2Array([Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)])
	var ru := rad / r.size.x
	var rv := rad / r.size.y
	# corner centres in uv, and the angle each arc sweeps from
	var arcs := [
		[Vector2(ru, rv), PI],
		[Vector2(1.0 - ru, rv), -PI * 0.5],
		[Vector2(1.0 - ru, 1.0 - rv), 0.0],
		[Vector2(ru, 1.0 - rv), PI * 0.5],
	]
	for a in arcs:
		var c: Vector2 = a[0]
		var start: float = a[1]
		for s in CORNER_SEGS + 1:
			var ang := start + (PI * 0.5) * float(s) / float(CORNER_SEGS)
			out.append(c + Vector2(cos(ang) * ru, sin(ang) * rv))
	return out


## The four corner cut-offs, in screen space - the area between the square corner and the
## arc, which is painted back in paper so the picture stops at the rounded edge.
func _corner_wedges(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		mirror: bool) -> Array:
	var r: Rect2 = sp.panels[i]["rect"]
	var short := minf(r.size.x, r.size.y)
	var rad := sp.radius * short
	var ru := rad / r.size.x
	var rv := rad / r.size.y
	var specs := [
		[Vector2(0, 0), Vector2(ru, rv), PI],
		[Vector2(1, 0), Vector2(1.0 - ru, rv), -PI * 0.5],
		[Vector2(1, 1), Vector2(1.0 - ru, 1.0 - rv), 0.0],
		[Vector2(0, 1), Vector2(ru, 1.0 - rv), PI * 0.5],
	]
	var out: Array = []
	for sp_i in specs:
		var corner: Vector2 = sp_i[0]
		var c: Vector2 = sp_i[1]
		var start: float = sp_i[2]
		var uv := PackedVector2Array([corner])
		for s in CORNER_SEGS + 1:
			var ang := start + (PI * 0.5) * float(s) / float(CORNER_SEGS)
			uv.append(c + Vector2(cos(ang) * ru, sin(ang) * rv))
		var scr := PackedVector2Array()
		var ok := true
		for p in uv:
			var pr := _lens.project(_world(sp.panel_point(i, p), m, sp.aspect, mirror))
			if pr.z <= _lens.near:
				ok = false
				break
			scr.append(Vector2(pr.x, pr.y) * u + origin)
		if ok:
			out.append(scr)
	return out
