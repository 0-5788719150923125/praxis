extends Vehicle
class_name ComicVehicle

## ComicVehicle - the show as a comic page, flown over by a real perspective camera.
##
## The same scenes, the same Director, the same cutting. What changes is where a scene
## LANDS: instead of replacing the picture, each cut fills the next PANEL of a page, and
## when the page is full the next cut turns it. So `Scene hold` and `Flourishes` keep
## their exact meanings - a burst of quick cuts becomes a run of small panels filling in
## fast, which is what a comic does with a fight.
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
## THE PAGE IS REALLY IN 3D. Not a sheared 2D plane faking depth - a quad placed in a
## world and projected through a [Lens3D], free to rotate on X, Y and Z at once. Panels
## are drawn as SUBDIVIDED grids of textured triangles whose vertices are each projected
## individually, because a two-triangle quad is affinely textured and warps: measured at
## a hard yaw, the seam of a two-tone test texture landed 28.5 px away from where
## perspective puts it, on a 512 px frame.

## How many panel slots exist. Two POOLS of [constant ComicPage.MAX_PANELS], alternating
## per page, because a page turn shows both pages at once - reusing one pool would pull
## the outgoing page's pictures out from under it mid-turn.
const POOL := ComicPage.MAX_PANELS

## Panel render-target size, as a fraction of the stage's shorter side. A panel is sampled
## at close to 1:1 in the tight reading shot (which frames roughly one panel), so this is
## about right and not a guess; it is clamped below so a small window still gets a legible
## panel and above so a 4K export does not allocate twelve enormous targets.
const PANEL_SCALE := 1.0
const PANEL_MIN := 224
const PANEL_MAX := 1400

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

## Page geometry in the world: the page is this many units wide, centred on the origin.
const PAGE_W := 2.0

## HOW MANY PANELS MAY RUN AT ONCE. See [method _update_liveness]: everything on screen
## moves, and the budget is what keeps that affordable. Three covers the panel being read
## plus the neighbours crowding the frame edges at the reading distance.
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

## PAGE ATTITUDE, in radians - the rest pose the sheet is rolled into for a page, on all
## three axes. Generous on purpose: the point of putting the paper in a real 3D world is
## that it can be TURNED, and at the reading distance a raked page is what reveals it has
## depth at all. A few degrees reads as a crooked scan rather than as a camera angle.
const PITCH := 0.55
const YAW := 0.85
const ROLL := 0.40
## ...and it never stops moving. Radians per second of slow continuous drift, so the sheet
## turns under the camera through the whole page rather than settling into a pose.
const DRIFT := 0.035

## CAMERA ELEVATION off the paper, in degrees. Never square-on (at 90 the perspective
## camera and the page agree and the panel is a flat rectangle again) and never grazing
## (the panel foreshortens into a line).
const EL_MIN_DEG := 32.0
const EL_MAX_DEG := 72.0
## Camera roll about the view axis - the Dutch angle, on top of the page's own roll.
const CAM_ROLL := 0.22

## HOW FAR ONE SHOT MAY DEPART FROM THE PAGE'S SET-UP. This is the fix for "it just bounces
## all over the place... a camera ping-ponging around at random angles".
##
## Every shot used to re-roll its azimuth over the whole circle, its elevation over the
## whole range, its roll and its focal length - independently, at every reading advance. So
## consecutive shots were unrelated positions, and the interpolation between them was a
## camera swinging across the page for no reason. Smoothing it does not help: the problem
## is not the transition, it is that there is nothing in common between where it was and
## where it is going.
##
## So a page picks ONE set-up - a side, a height, a tilt, a lens - and a shot varies inside
## it. That is coverage rather than chaos, and staying on one side of the subject is the
## oldest rule in the grammar (the 180-degree line). Changing sides is now a PAGE turn,
## which is the one moment where a new set-up reads as a new scene rather than as a mistake.
const AZ_ARC := 0.42              # +/- radians of orbit around the page's side (~24 deg)
const EL_VARY_DEG := 7.0          # +/- degrees of height off the page's own
const ROLL_VARY := 0.05           # +/- radians of Dutch on top of the page's own
## The lens, chosen once per page. Narrower than it was (it went to 64): a wide lens this
## close to a panel bends its edges, which is the "unfocused, distorted" half of the report.
const FOV := Vector2(38.0, 54.0)

## HOW TALL THE PANEL BEING READ IS, in frames. 1.0 spans the frame exactly top to bottom;
## above that it overflows and the screen crops into it. Never below 1: the shot this
## vehicle exists for is the one where the panel IS the picture, and the moment the panel
## stops spanning the frame you are looking at the page again, with the desk around it.
##
## VERTICAL ONLY - see _fit. A panel much narrower than a 16:9 frame cannot fill it and
## should not try; what fills the sides is the gutter and the panels next to it, which is
## what a comic read close actually looks like.
const FILL := Vector2(1.00, 1.45)
## The hard near limit, in frames of panel height - however much the page wants covering,
## the camera stops here. Past about this the panel stops reading as a panel.
##
## SET GENEROUSLY, and the first value was not. At 2.2 this floor came out FURTHER from the
## page than the covering distance did (measured 1.66 against 1.24) and so it, not the
## framing, decided every shot - vetoing exactly the push-in that fills the frame and
## leaving the strip of desk down the side that this whole pass exists to remove. A floor
## that binds in the ordinary case is not a floor, it is the rule.
const CROP_MAX := 4.5
## How far the aim is pulled from the panel's centre toward the middle of the sheet. A
## panel at the page's edge, framed dead on, puts the trim edge - and the void behind the
## paper - into shot; pulling the aim in keeps the sheet under the panel at every panel.
const AIM_PULL := 0.12

## Pan rates (ease per second) for moving between panels: a slow drift, an ordinary walk,
## and a whip. A comic read at one speed is a slideshow on rails.
const PAN_SLOW := 0.9
const PAN_WALK := 2.2
const PAN_WHIP := 7.5
## The page's own attitude eases at its own, much slower rate - it is scenery, not a move.
const PAGE_EASE := 0.7
## A page turn has somewhere to be.
const TURN_EASE := 3.0
## How long a page turn takes, in seconds.
const TURN_TIME := 1.15

## THE PAGE LIES ON SOMETHING. Without it the frame past the page's trim edge is pure
## black, which at a hard rake is a wedge of nothing across the shot - and black is not a
## background, it is the absence of one.
##
## PAINTED IN SCREEN SPACE, not as a world quad. A surface big enough to never run out
## under a close raked camera has corners far outside the view frustum, and the projection
## has to drop any quad with a corner behind the eye (it inverts through infinity) - so the
## world-quad version simply never drew, at any of the sizes worth having. A defocused
## surface has no texture to rake anyway; what it needs to be is present and not black.
## How dark the wash goes at the corners of the frame, as a fraction of the desk colour.
const DESK_VIGNETTE := 0.55
## The page's shadow on it: offset in page widths, and how far off the paper it is cast.
const SHADOW_OFF := Vector2(0.045, 0.055)
const SHADOW_DEPTH := 0.06

# --- state -------------------------------------------------------------------
var _slots: Array = []            # POOL * 2 SubViewports; page p uses pool (p % 2)
var _pool := 0                    # which pool the CURRENT page is drawing from
var _page: ComicPage = null
var _page_i := -1
var _cast: Array = []             # per panel of the current page: its GhostScene, or null
var _read := 0                    # panel the camera is reading, and the Director's current
var _to_cast: Array = []          # panels still waiting to be cast, one per frame
var _prev_page: ComicPage = null  # the outgoing page, alive only through a turn
var _turn_t := -1.0               # seconds into a page turn, < 0 when none

## WHICH PANEL OF THIS PAGE HOLDS FOOTAGE, or -1 for none, and the clip it holds.
##
## AT MOST ONE PER PAGE, which is the whole answer to a question that would otherwise need
## a mechanism. A clip's position is a pure function of the show clock (see
## [method Films.position_at]), so two panels sampling the same clip at the same instant
## would show the same picture twice - and the page is what you see at once, so one per
## page IS one at a time. During a turn the outgoing page's panels are already stopped,
## so even then only one is playing.
var _film_at := -1
var _film_clip: Dictionary = {}
var _film_prev := ""              # last page's clip path, so two pages running do not repeat

var _lens := Lens3D.new()
var _mod: ModBank = null
var _bookend := 1.0
var _panel_px := 512
var _stage_size := Vector2(1280, 720)
var _live: Array = []             # panel indices whose viewport is running this frame
var _frame := 0                   # for the off-focal repaint phase, see OFF_FOCAL_PERIOD

# THE PAGE'S ATTITUDE, eased toward a target that itself DRIFTS. Sampled on all three axes
# with real magnitude: a page that only pitches a few degrees reads as a slightly crooked
# scan, and the whole reason to put the sheet in a real 3D world is that it can be turned.
var _att := Vector3.ZERO
var _att_target := Vector3.ZERO
var _att_rate := Vector3.ZERO     # slow continuous drift, per second

# THE CAMERA STATE, expressed in PAGE-LOCAL SPHERICAL rather than as a world position: an
# azimuth around the aim and an elevation off the paper. That is what makes the rake real -
# an eye on the page's normal sees a flat rectangle however the page is rotated - and it is
# also what makes travelling between panels a move PARALLEL TO THE PAGE, because the
# direction is held and only the point it looks at changes.
# Keys: aim (page coords), az, el, roll, fill, fov. See _station.
## The page's SET-UP: the side the camera is on, its height, its tilt and its lens. Fixed
## for the page; see AZ_ARC.
var _az_base := 0.0
var _el_base := 1.0
var _roll_base := 0.0
var _lens_fov := 46.0

var _cam := {"aim": Vector2(0.5, 0.7), "az": 0.0, "el": 1.0, "roll": 0.0,
	"fill": 1.1, "fov": 52.0}
## The move being travelled: {kind, a, b, dur, ease}. See MOVES.
var _mv := {}
var _mv_t := 0.0
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
	# The surface the page lies on: dark enough that the paper is still the brightest thing
	# in the frame by a long way, but never black.
	_desk = Color.from_hsv(_rng.randf_range(0.02, 0.12), _rng.randf_range(0.10, 0.32),
		_rng.randf_range(0.055, 0.115))
	_ink_weight = _rng.randf_range(0.0030, 0.0052)
	_mod = ModBank.new(Director.session_seed() ^ 0x9A6E)
	_reset_book()


func release() -> void:
	# The vehicle OUTLIVES a session (main owns it; the synthesis modes re-attach per
	# take), so release resets the BOOK rather than tearing the pool down - rebuilding
	# twelve render targets on every settings change is exactly the reallocation churn
	# main's governor warns about.
	_reset_book()


func _reset_book() -> void:
	for i in _slots.size():
		_blank(_slots[i])
	_cast = []
	_to_cast = []
	_page_i = -1
	_read = 0
	_prev_page = null
	_turn_t = -1.0
	_mv = {}
	_mv_t = 0.0
	_dip_t = -1.0
	_film_at = -1
	_film_clip = {}
	_film_prev = ""
	_turn_page(0)


## A window resize changes what a panel is worth in pixels, but NOTHING is resized here.
##
## A render target reallocated while it is being sampled is the black-triangle and
## colour-noise corruption main's stage governor documents at length - and worse, a resize
## also throws away the held panels' pictures. So the new figure is remembered and applied
## at the next page, where every target is repainted from scratch anyway.
func on_stage_resized(size: Vector2) -> void:
	_size_targets(size)
	queue_redraw()


func _size_targets(size: Vector2) -> void:
	_stage_size = size
	# A panel fills the frame at the reading distance, so it is worth close to the whole
	# short axis. Clamped below so a small window still gets a legible panel, and above so
	# a 4K export does not allocate twelve enormous targets.
	_panel_px = clampi(int(minf(size.x, size.y) * PANEL_SCALE), PANEL_MIN, PANEL_MAX)


func _build_slots() -> void:
	for i in POOL * 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(_panel_px, _panel_px)
		vp.transparent_bg = false
		vp.disable_3d = true          # every scene rasterizes to the 2D canvas
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.process_mode = Node.PROCESS_MODE_DISABLED
		add_child(vp)
		_slots.append(vp)


# --- the Vehicle contract ----------------------------------------------------

## The page owns its cast. See [method Vehicle.owns_cast] - this is the whole difference
## between a page and a slideshow.
func owns_cast() -> bool:
	return true


## A Director cut MOVES THE READING to the next panel; it does not build anything. The
## panel already holds a live scene, cast when the page turned.
func take_over(_outgoing: GhostScene) -> GhostScene:
	if _page == null:
		return null
	if _read + 1 >= _page.panels.size():
		_turn_page(_page_i + 1)
	elif _page_i >= 0:
		_read += 1
		_choose_move()
	# CAST ON DEMAND. The rest of the page is cast one panel per frame to keep the turn
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
			_prev_page = null
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


func _turn_page(idx: int) -> void:
	if idx > 0 and _page != null:
		_prev_page = _page
		_turn_t = 0.0
	# Let the outgoing page's cast go, but NOT its pictures: _blank stops each viewport
	# before freeing what is inside it, so the page turning away is still a drawn page.
	for i in POOL:
		_blank(_slots[_pool * POOL + i])
	_page_i = idx
	_pool = idx % 2
	_page = ComicPage.new(hash([Director.session_seed(), "comic-page", idx]))
	_cast = []
	_cast.resize(_page.panels.size())
	_read = 0
	# The panel being read is cast NOW - the show cannot open on an empty frame - and the
	# rest are queued one per frame. Building five scenes in one frame is a visible hitch,
	# and there is nowhere to hide it; spread over the next few frames it lands under the
	# page turn, and any panel the camera reaches before its turn is cast on demand.
	_to_cast = []
	for i in range(1, _page.panels.size()):
		_to_cast.append(i)
	_choose_film()               # BEFORE the first cast: panel 0 may be the film one
	_cast_panel(0)
	_choose_page_look()
	_choose_move()


## DOES THIS PAGE GET A PIECE OF FOOTAGE, and if so which panel and which clip.
##
## Seeded off the page, like everything else here, so a session replays the same footage
## in the same panels - the show is reproducible from one seed and this is not the place
## to make it stop being. A frequency of 0, or a library with nothing in it, leaves
## `_film_at` at -1 and the comic behaves exactly as it did before films existed.
func _choose_film() -> void:
	_film_at = -1
	_film_clip = {}
	if _page == null or Films.frequency() <= 0.0:
		return
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "film-page", _page_i])
	if r.randf() > Films.frequency():
		return
	var list := Films.clips()
	if list.is_empty():
		return
	var at := r.randi() % list.size()
	# NOT THE SAME CLIP TWICE RUNNING. With one film panel per page, a small library and
	# an independent draw per page, the same clip lands on consecutive pages often enough
	# to read as "that video again" rather than as a library.
	if list.size() > 1 and String((list[at] as Dictionary).get("source", "")) == _film_prev:
		at = (at + 1) % list.size()
	_film_clip = list[at]
	_film_at = _best_panel_for(_film_clip, r)
	# IS THERE ANYTHING TO PLAY YET. A clip is prepared a window at a time, cut from the
	# original when something wants it (see Films.WINDOW), so the window covering this
	# moment may still be encoding. Asking starts that cut; not getting it back means this
	# page simply goes without footage, and the panel is an ordinary scene rather than a
	# blank rectangle waiting for a file. The whole feature self-throttles on this line -
	# film appears as often as the machine can prepare it.
	if not Films.warm(_film_clip, maxf(Spectrum.current.time, 0.0)):
		_film_clip = {}
		_film_at = -1
		return
	_film_prev = String(_film_clip.get("source", ""))


## WHICH PANEL SUITS THIS CLIP'S SHAPE. Footage covers its panel, so whichever axis is
## spare gets cropped off - a 16:9 clip in a 0.45-aspect panel loses three quarters of its
## width, and there is no framing clever enough to make that not matter. A page offers
## several shapes at once, so the answer is simply to put the film in the one closest to the
## clip's own, where the crop is a trim rather than a demolition.
##
## Compared in LOG space, because aspect is a ratio: 2.0 and 0.5 are equally wrong for a
## square clip, and subtracting them would call one of them nearly right.
##
## Falls back to a plain random panel when the clip's shape is unknown, which is the honest
## answer rather than a guess - see Films.aspect_of.
func _best_panel_for(clip: Dictionary, r: RandomNumberGenerator) -> int:
	var n := _page.panels.size()
	var want := Films.aspect_of(clip)
	if want <= 0.0 or n <= 1:
		return r.randi() % maxi(n, 1)
	var best := 0
	var best_err := INF
	for i in n:
		var err := absf(log(maxf(_page.panel_aspect(i), 0.01) / want))
		if err < best_err:
			best_err = err
			best = i
	return best


## One queued panel per frame. See _turn_page.
func _cast_one() -> void:
	if _to_cast.is_empty():
		return
	_cast_panel(int(_to_cast.pop_front()))


func _cast_panel(i: int) -> void:
	if _page == null or i < 0 or i >= _cast.size():
		return
	if _cast[i] != null and is_instance_valid(_cast[i]):
		return
	var vp := _open_slot(i)
	# FOOTAGE, where the page called for it. Cast directly rather than through the
	# Director: a film is not in the catalogue (see FilmScene), because a scene that only
	# exists when the viewer has imported something would make the running order depend on
	# the library.
	if i == _film_at and not _film_clip.is_empty():
		var film := FilmScene.new()
		film.set_clip(_film_clip, maxf(Spectrum.current.time, 0.0))
		film.init_with_seed(hash([_page_i, i, "film"]), "static")
		film.scene_name = "film"
		vp.add_child(film)
		_cast[i] = film
		return
	# The salt is the panel index, so the page's panels are separate draws off the novelty
	# scheduler rather than the same one repeated - the Director's clock has not moved
	# between them (see Director.mint_scene). Quiet for every panel but the one being read:
	# casting a page is one change of scene, not five.
	var sc := Director.mint_scene(hash([_page_i, i]) & 0xFFFF, i != _read)
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
	var a := _page.panel_aspect(i)
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
## A page whose panels arrive one at a time is a slideshow with gutters; a comic page has
## all of it drawn at once. But six live scenes is not a thing that runs - the stage
## governor spends its whole budget on one - so liveness follows the CAMERA instead of the
## clock. At the reading distance one panel fills the frame and its neighbours crowd the
## edges, so what is on screen is typically one to three panels; the rest are stopped
## render targets holding their last frame, exactly as before, and they resume when the
## camera comes back to them.
##
## LIVE_MAX is the hard bound. It is ordered by distance from the panel being read, so if a
## rake ever puts more of the page in shot than the budget allows, the ones that keep
## moving are the ones the reading is about.
func _update_liveness() -> void:
	if _page == null:
		return
	var frame := Rect2(Vector2.ZERO, _stage_size).grow(_stage_size.x * 0.06)
	var want: Array = []
	for i in _cast.size():
		if _cast[i] == null or not is_instance_valid(_cast[i]):
			continue
		if i != _read and not _panel_rect(i).intersects(frame):
			continue
		want.append({"i": i, "d": absi(i - _read)})
	want.sort_custom(func(a, b): return int(a.d) < int(b.d))
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
		var on: bool = keep.has(i) and (i == _read or i == _film_at
			or (_frame + i) % OFF_FOCAL_PERIOD == 0)
		if on:
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
		var period := 1 if i == _film_at else OFF_FOCAL_PERIOD
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
##   w     relative weight in the bag
##   dur   seconds, sampled in this range
##   ease  linear (a dolly, constant speed) / smooth / out (fast then settle) / snap (a cut)
##   hard  true = the move BEGINS somewhere else entirely, so it starts on a jump cut
##         rather than continuing from where the camera was
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
	"drift":   {"w": 4.0, "dur": [8.0, 15.0], "ease": "smooth", "hard": false},
	"push":    {"w": 3.5, "dur": [7.0, 13.0], "ease": "smooth", "hard": false},
	"track":   {"w": 2.5, "dur": [9.0, 16.0], "ease": "linear", "hard": false},
	"orbit":   {"w": 2.0, "dur": [9.0, 15.0], "ease": "smooth", "hard": false},
	"sweep_h": {"w": 1.5, "dur": [9.0, 16.0], "ease": "linear", "hard": false},
	"sweep_v": {"w": 1.0, "dur": [9.0, 16.0], "ease": "linear", "hard": false},
	"pull":    {"w": 1.5, "dur": [7.0, 13.0], "ease": "smooth", "hard": false},
	# --- and the one that just looks at a panel ----------------------------------
	"hold":    {"w": 2.0, "dur": [5.0, 10.0], "ease": "smooth", "hard": false},
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
## THE AIM NEVER LEAVES THE SHEET. Page coordinates, as a margin inside the trim.
##
## A move that aims past the page edge is aiming at the desk, and no framing distance can
## then put paper across the whole frame - the covering solve has nothing to solve. Sweeps
## and tracks deliberately run to and past the edges, so they are clamped here rather than
## being written not to.
const AIM_MARGIN := 0.06
## How deep a `push` gets, in frames of panel height.
const PUSH_FILL := Vector2(2.2, 3.8)
## The far station a `swoop` starts from: elevation in degrees, and how far off.
const SWOOP_EL_DEG := Vector2(14.0, 26.0)
const SWOOP_FILL := 0.42
## The arc an `orbit` sweeps, in radians.
const ORBIT_ARC := Vector2(1.1, 2.6)
## A `dip` goes out over the first of these and back over the second, in seconds.
const DIP_OUT := 0.16
const DIP_IN := 0.34

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


## The page's attitude for this page: a real rake on all three axes, plus a slow continuous
## drift so the sheet is never still.
func _choose_page_look() -> void:
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "comic-page-look", _page_i])
	_att_target = Vector3(
		r.randf_range(-PITCH, PITCH),
		r.randf_range(-YAW, YAW),
		r.randf_range(-ROLL, ROLL))
	_att_rate = Vector3(
		r.randf_range(-DRIFT, DRIFT),
		r.randf_range(-DRIFT, DRIFT),
		r.randf_range(-DRIFT, DRIFT) * 0.5)
	# THE SET-UP FOR THIS PAGE - decided ONCE, and then everything below only varies within
	# it. See the note above AZ_ARC: re-rolling these per shot is what made the camera
	# ping-pong, and no amount of smooth interpolation between two unrelated positions
	# rescues it, because the problem is that they are unrelated.
	_az_base = r.randf_range(0.0, TAU)
	_el_base = deg_to_rad(r.randf_range(EL_MIN_DEG, EL_MAX_DEG))
	_roll_base = r.randf_range(-CAM_ROLL, CAM_ROLL)
	_lens_fov = r.randf_range(FOV.x, FOV.y)


## A camera station: where it aims on the page, and from what angle and how close.
func _station(r: RandomNumberGenerator, panel: int) -> Dictionary:
	var i := clampi(panel, 0, maxi(0, _page.panels.size() - 1))
	return {
		"aim": _page.panel_center(i).lerp(Vector2(0.5, _page.aspect * 0.5), AIM_PULL),
		# WITHIN THE PAGE'S SET-UP, never re-rolled from scratch. See AZ_ARC.
		"az": _az_base + r.randf_range(-AZ_ARC, AZ_ARC),
		# Elevation off the PAPER, never square-on: at 90 degrees the perspective camera and
		# the page agree exactly and the panel is a flat rectangle again.
		"el": clampf(_el_base + deg_to_rad(r.randf_range(-EL_VARY_DEG, EL_VARY_DEG)),
			deg_to_rad(EL_MIN_DEG), deg_to_rad(EL_MAX_DEG)),
		"roll": _roll_base + r.randf_range(-ROLL_VARY, ROLL_VARY),
		"fill": r.randf_range(FILL.x, FILL.y),
		# ONE LENS PER PAGE. A production picks a lens and covers the scene with it; changing
		# focal length every shot is a thing that reads as a mistake even to someone who
		# could not name what changed.
		"fov": _lens_fov,
	}


## THE PANELS A SWEEP MAY TRAVEL BETWEEN: the leftmost and rightmost panel centres that
## share a row with `panel`, so a horizontal pan starts on content and ends on content.
## Falls back to the panel itself when it is alone on its row - a sweep with nowhere to go
## becomes a shot that sits, which is better than one that pans across the margin.
func _row_span(panel: int) -> Vector2:
	var c := _page.panel_center(panel)
	var lo := c.x
	var hi := c.x
	for i in _page.panels.size():
		var o := _page.panel_center(i)
		if absf(o.y - c.y) < _page.aspect * 0.12:
			lo = minf(lo, o.x)
			hi = maxf(hi, o.x)
	return Vector2(lo, hi) if hi - lo > 0.02 else Vector2(c.x, c.x)


## The same down a column.
func _col_span(panel: int) -> Vector2:
	var c := _page.panel_center(panel)
	var lo := c.y
	var hi := c.y
	for i in _page.panels.size():
		var o := _page.panel_center(i)
		if absf(o.x - c.x) < 0.12:
			lo = minf(lo, o.y)
			hi = maxf(hi, o.y)
	return Vector2(lo, hi) if hi - lo > 0.02 else Vector2(c.y, c.y)


## Pick and plan the next move. Called on every reading advance and on every page turn.
func _choose_move() -> void:
	if _page == null:
		return
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "comic-move", _page_i, _read])
	var kind := _pick_move(r)
	var spec: Dictionary = MOVES[kind]
	var b := _station(r, _read)
	# CONTINUITY BY DEFAULT: a move starts from wherever the camera actually is, so the
	# picture never jumps unless the move is one whose whole point is that it jumps.
	var a: Dictionary = b.duplicate() if bool(spec["hard"]) else _cam.duplicate()
	var row: float = _page.panel_center(_read).y
	var col: float = _page.panel_center(_read).x

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
		"sweep_h":
			# ACROSS THE ROW, PANEL TO PANEL - not across the SHEET. It used to run from
			# -0.10 to 1.10 in page coordinates, which is a pan that begins and ends on the
			# desk and spends much of its length over margin and gutter: "tracking along
			# unfocused edges". A sweep is a pan over CONTENT or it is nothing.
			var span_h := _row_span(_read)
			a["aim"] = Vector2(span_h.x, row)
			b["aim"] = Vector2(span_h.y, row)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"sweep_v":
			var span_v := _col_span(_read)
			a["aim"] = Vector2(col, span_v.x)
			b["aim"] = Vector2(col, span_v.y)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"push":
			b["aim"] = a.aim.lerp(b.aim, 0.5)
			b["fill"] = r.randf_range(PUSH_FILL.x, PUSH_FILL.y)
			for k in ["az", "el", "roll"]:
				b[k] = a[k]
		"pull":
			a["fill"] = r.randf_range(PUSH_FILL.x, PUSH_FILL.y)
			a["aim"] = b.aim
		"orbit":
			a["aim"] = b.aim
			b["az"] = a.az + r.randf_range(ORBIT_ARC.x, ORBIT_ARC.y) * (1.0 if r.randf() < 0.5 else -1.0)
			for k in ["el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"swoop":
			# In off the FAR SIDE of the page, low and wide, arriving on the subject.
			a["az"] = b.az + PI
			a["el"] = deg_to_rad(r.randf_range(SWOOP_EL_DEG.x, SWOOP_EL_DEG.y))
			a["fill"] = SWOOP_FILL
			a["aim"] = b.aim + (b.aim - Vector2(0.5, _page.aspect * 0.5)).normalized() * 0.5 \
				if b.aim.distance_to(Vector2(0.5, _page.aspect * 0.5)) > 0.02 else b.aim
		"hold":
			b["aim"] = a.aim.lerp(b.aim, 0.85)

	_mv = {
		"kind": kind,
		"a": a,
		"b": b,
		"dur": maxf(0.05, r.randf_range(float(spec["dur"][0]), float(spec["dur"][1]))),
		"ease": String(spec["ease"]),
	}
	_mv_t = 0.0
	if bool(spec["hard"]):
		_cam = a.duplicate()          # the jump itself
	_dip_t = 0.0 if kind == "dip" else -1.0
	if _page_i >= 0:
		print("ghost: comic %s -> panel %d of page %d" % [kind, _read + 1, _page_i])


func _pick_move(r: RandomNumberGenerator) -> String:
	var total := 0.0
	for k in MOVES:
		total += float(MOVES[k]["w"])
	var pick := r.randf() * total
	for k in MOVES:
		pick -= float(MOVES[k]["w"])
		if pick <= 0.0:
			return String(k)
	return "drift"


func _ease(delta: float) -> void:
	# The page turns under the camera, slowly and forever - independent of the move, because
	# the sheet is scenery and the camera is the performance.
	_att_target += _att_rate * delta
	var sway := Vector3(
		_mod.value("tilt") * 0.09,
		_mod.value("sway") * 0.12,
		_mod.value("roll") * 0.05) if _mod != null else Vector3.ZERO
	_att = _att.lerp(_att_target + sway, 1.0 - exp(-PAGE_EASE * delta))
	if _dip_t >= 0.0:
		_dip_t += delta
	if _mv.is_empty():
		return
	_mv_t += delta
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


## Put the eye on its page-local spherical offset from where the camera is aiming.
##
## The offset direction is built in PAGE space and then rotated by the page's attitude, so
## it is a fixed station relative to the paper: travelling from panel to panel slides the
## camera ACROSS the page at a constant angle, which is the move a copy-stand shot makes and
## the one the eye reads as flying over a comic. Building it in world space instead would
## swing the angle around every time the page moved.
func _place_eye() -> void:
	if _page == null:
		return
	var basis := Basis.from_euler(_att)
	var aim: Vector2 = _clamp_aim(_cam.aim)
	var c := _page_point(aim, _page, _att)
	# A FIELD scene is raked gently, a subject hard - see FIELD_FLATTEN.
	var el: float = lerpf(float(_cam.el), PI * 0.5, _flatten())
	var roll: float = float(_cam.roll) * (1.0 - _flatten() * 0.7)
	_roll = roll
	_fov = float(_cam.fov)
	var dir := basis * Vector3(cos(_cam.az) * cos(el), sin(_cam.az) * cos(el), sin(el))
	# SCALE IS SET BY WHICHEVER PANEL THE CAMERA IS OVER, not by the one being read: a
	# travelling move spends most of its time between panels, and framing those seconds
	# against a panel that is off screen puts the scale somewhere arbitrary.
	var pw := _panel_world(_nearest_panel(aim))
	_cam["aim"] = aim
	# TWO CONSTRAINTS, and the nearer wins.
	#
	# The first frames the PANEL: near enough that it spans the frame top to bottom. The
	# second frames the PAGE: near enough that the sheet covers the frame's WIDTH. A comic
	# page is portrait and the screen is landscape, so on a narrow page the first alone
	# leaves a third of the picture as the surface the page is lying on - correct framing of
	# the wrong subject. Taking the nearer pushes in until the paper runs edge to edge.
	var d := minf(_fit(pw, c, float(_cam.fill), dir),
		_cover(_page_world(_page, _att), c, dir))
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


## Keep an aim on the paper - see AIM_MARGIN.
func _clamp_aim(p: Vector2) -> Vector2:
	return Vector2(
		clampf(p.x, AIM_MARGIN, 1.0 - AIM_MARGIN),
		clampf(p.y, AIM_MARGIN, _page.aspect - AIM_MARGIN))


## How much of the frame the PAGE currently covers horizontally, 1.0 = edge to edge or
## better. Below 1 there is desk in shot. Read by tests/comic_look_probe.gd, which is the
## only way "is there dead space in the frame" gets checked by something other than an eye.
func page_coverage() -> float:
	if _page == null:
		return 1.0
	var u := minf(_stage_size.x, _stage_size.y)
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0)
	var lo := 1e9
	var hi := -1e9
	for w in _page_world(_page, _att):
		var pr := _lens.project(w)
		if pr.z <= _lens.near:
			return 1.0                 # wrapping the eye: the sheet is all around us
		lo = minf(lo, pr.x)
		hi = maxf(hi, pr.x)
	return minf(minf(-lo, hi) / maxf(hx, 1e-4), 1.0)


## Which panel the camera is over. Nearest centre in page coordinates.
func _nearest_panel(aim: Vector2) -> int:
	var best := 0
	var best_d := 1e9
	for i in _page.panels.size():
		var d := aim.distance_squared_to(_page.panel_center(i))
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
	# through 90 degrees in one frame). The page's own up is always perpendicular to the
	# paper's normal, so it is a stable reference - and taking it from the page is also what
	# makes the page's roll show as a tilted horizon rather than as nothing at all.
	var basis := Basis.from_euler(_att)
	var fwd := (_look - _eye)
	fwd = fwd.normalized() if fwd.length() > 1e-6 else Vector3.FORWARD
	var up := basis * Vector3.UP
	if absf(up.dot(fwd)) > 0.9:
		up = basis * Vector3.RIGHT          # grazing along the page's up axis
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
## rearrange. Twenty-odd halvings on four points is nothing, and it is exact at any rake.
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
	var probe := Lens3D.new()
	probe.fov = _fov
	probe.look = c
	var lo := 0.15
	var hi := 24.0
	for _iter in 22:
		var mid := (lo + hi) * 0.5
		probe.eye = c + dir * mid
		probe.up = _lens.up
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


## THE COVERING DISTANCE: the FARTHEST the eye may sit along [param dir] while [param pts]
## still span the whole frame horizontally. Beyond it you can see past them.
##
## The mirror of [method _fit], and it has to bisect the other way: a projection shrinks
## with distance, so "fits inside" is true beyond some distance and "covers" is true within
## one. Same twenty-two halvings, same exactness at any rake.
func _cover(pts: Array, c: Vector3, dir: Vector3) -> float:
	var u := minf(_stage_size.x, _stage_size.y)
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0)
	var probe := Lens3D.new()
	probe.fov = _fov
	probe.look = c
	probe.up = _lens.up
	var lo := 0.15
	var hi := 24.0
	for _iter in 22:
		var mid := (lo + hi) * 0.5
		probe.eye = c + dir * mid
		probe.prepare()
		var min_x := 1e9
		var max_x := -1e9
		var wraps := false
		for pt in pts:
			var pr := probe.project(pt)
			if pr.z <= probe.near:
				# A corner BEHIND THE EYE means the camera is over the sheet rather than
				# off it, which is the most covered case there is - the projection just
				# cannot express it. Reading this as "does not cover" (which the first cut
				# did) inverts the whole search and drives the camera INSIDE a panel.
				wraps = true
				break
			min_x = minf(min_x, pr.x)
			max_x = maxf(max_x, pr.x)
		# covers when the sheet reaches past BOTH frame edges, not merely when it is wide
		# enough - a page wide enough but pushed to one side leaves the other side bare.
		if wraps or (min_x <= -hx and max_x >= hx):
			lo = mid
		else:
			hi = mid
	return lo


## The four world corners of one panel, cant and page attitude applied.
func _panel_world(i: int) -> Array:
	var out: Array = []
	for uv in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		out.append(_page_point(_page.panel_point(i, uv), _page, _att))
	return out


## The four world corners of a whole page.
func _page_world(pg: ComicPage, att: Vector3) -> Array:
	var out: Array = []
	for p in [Vector2(0, 0), Vector2(1, 0), Vector2(1, pg.aspect), Vector2(0, pg.aspect)]:
		out.append(_page_point(p, pg, att))
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


## A point in the normalized page box (x 0..1, y 0..aspect, y down) -> world space, with
## the page's attitude applied. Everything drawn on the page goes through this one call,
## so the paper, the panels, the gutters and the ink can never disagree about where the
## page is.
func _page_point(p: Vector2, pg: ComicPage, att: Vector3, spine := 0.0, depth := 0.0) -> Vector3:
	var local := Vector3(
		(p.x - 0.5) * PAGE_W,
		-(p.y - pg.aspect * 0.5) * PAGE_W,      # page y is DOWN, world y is up
		-depth * PAGE_W)                        # behind the sheet, along its own normal
	# THE TURN HINGES ON THE SPINE, not the middle. A page rotated about its own centre
	# reads as a card spinning in place; a page rotated about its left edge reads as a
	# page being turned, which is the entire gesture. Rotating about the centre is what
	# the first cut of this did, and it looked like a glitch.
	if not is_zero_approx(spine):
		var px := Vector3(-PAGE_W * 0.5, 0.0, 0.0)
		local = px + Basis(Vector3.UP, spine) * (local - px)
	return Basis.from_euler(att) * local


# --- drawing -----------------------------------------------------------------

func _draw() -> void:
	if _page == null:
		return
	_lens.eye = _eye
	_lens.look = _look
	_lens.fov = _fov
	_lens.prepare()
	var u := minf(_stage_size.x, _stage_size.y)
	var origin := _stage_size * 0.5
	# The bookend belongs to the vehicle here (see Vehicle.owns_bookend): the whole page
	# fades, paper and all, instead of one panel fading inside a lit page.
	modulate.a = 1.0
	var fade := clampf(_bookend, 0.0, 1.0) * _dip_alpha()
	_backdrop(fade)          # once per frame, under everything, including through a turn

	# A PAGE TURN. The new page is simply THERE, flat, from the first frame - it is the
	# page under the one being turned - and the outgoing page swings over it on the spine
	# and out of the frame. Both are drawn, the new one first so the turning page passes
	# over it, and that is why the two slot pools exist: the outgoing page still needs its
	# own panel textures while the new one is filling.
	if _turn_t >= 0.0 and _prev_page != null:
		var k := clampf(_turn_t / TURN_TIME, 0.0, 1.0)
		_draw_page(_page, _is_cast, _pool, _att, u, origin, fade, 0.0)
		# It is let go of as it passes edge-on. There is no printed BACK of a page here, so
		# carrying it past ninety degrees would show the front again, mirrored - a page
		# turning inside out. Gone by 0.55 of the swing, which is just before that.
		var leaving := fade * (1.0 - smoothstep(0.30, 0.55, k))
		if leaving > 0.003:
			_draw_page(_prev_page, _all_drawn, 1 - _pool, _att, u, origin, leaving,
				-smoothstep(0.0, 1.0, k) * PI * 0.62)
		return
	_draw_page(_page, _is_cast, _pool, _att, u, origin, fade, 0.0)


## Has panel [param i] of the CURRENT page been cast yet? (See _turn_page's queue.)
func _is_cast(i: int) -> bool:
	return i < _cast.size() and _cast[i] != null and is_instance_valid(_cast[i])


## Every panel of the page turning away carries a picture - it was fully cast before the
## turn started, and stopping its viewports kept what they had drawn.
func _all_drawn(_i: int) -> bool:
	return true


## [param drawn] says which panels of [param pg] carry a picture. For the page being read
## that is "has this one been cast yet" - the queue casts one per frame, so for the first
## few frames of a page a late panel is still blank paper. For the page turning away it is
## every panel, because it was fully cast before the turn began.
func _draw_page(pg: ComicPage, drawn: Callable, pool: int, att: Vector3,
		u: float, origin: Vector2, fade: float, spine: float) -> void:
	if fade <= 0.003:
		return
	var paper := Color(_paper.r, _paper.g, _paper.b, fade)
	var ink := Color(_ink.r, _ink.g, _ink.b, fade)
	# The sheet's shadow on the surface under it, before the sheet itself.
	_shadow_quad(pg, att, u, origin, fade, spine)
	# THE PAPER, subdivided so its gradient is per-vertex rather than per-quad, and so a
	# steeply raked page still shades smoothly across its length.
	_paper_quad(pg, att, u, origin, paper, spine)
	# THE PANELS. Painter's order along the page normal is irrelevant - they are coplanar
	# and never overlap - so reading order is the order, which is also the order the ink
	# wants (a canted panel's border must sit over its neighbour's paper, not under it).
	for i in pg.panels.size():
		if bool(drawn.call(i)):
			_panel_quad(pg, i, pool, att, u, origin, fade, spine)
	for i in pg.panels.size():
		_panel_ink(pg, i, att, u, origin, paper, ink, bool(drawn.call(i)), spine)


## The surface the page lies on: a vignetted wash over the whole frame, painted first.
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


## The sheet's shadow on that surface - the page's own outline, offset and pushed back.
## One quad, and it is most of what makes the paper read as an object lying on something
## rather than as a rectangle floating in a void.
func _shadow_quad(pg: ComicPage, att: Vector3, u: float, origin: Vector2,
		fade: float, spine: float) -> void:
	var poly := PackedVector2Array()
	for p in [Vector2(0, 0), Vector2(1, 0), Vector2(1, pg.aspect), Vector2(0, pg.aspect)]:
		var q: Vector2 = p + Vector2(SHADOW_OFF.x, SHADOW_OFF.y * pg.aspect)
		var pr := _lens.project(_page_point(q, pg, att, spine, SHADOW_DEPTH))
		if pr.z <= _lens.near:
			return
		poly.append(Vector2(pr.x, pr.y) * u + origin)
	if _poly_area(poly) < 1.0:
		return
	draw_colored_polygon(poly, Color(0.0, 0.0, 0.0, 0.42 * fade))


## The page itself: a subdivided quad with a gentle darkening toward the edges. Flat white
## paper reads as a blank canvas; a whisper of a gradient reads as a printed sheet.
func _paper_quad(pg: ComicPage, att: Vector3, u: float, origin: Vector2, paper: Color,
		spine: float) -> void:
	var pts := PackedVector2Array()
	var cols := PackedColorArray()
	var idx := PackedInt32Array()
	var n := PAPER_GRID
	for j in n + 1:
		for i in n + 1:
			var fu := float(i) / float(n)
			var fv := float(j) / float(n)
			var w := _page_point(Vector2(fu, fv * pg.aspect), pg, att, spine)
			var pr := _lens.project(w)
			if pr.z <= _lens.near:
				return                        # page edge-on / behind the eye: skip the frame
			pts.append(Vector2(pr.x, pr.y) * u + origin)
			# radial falloff from the sheet's centre, at a few percent
			var d := Vector2(fu - 0.5, fv - 0.5).length() / 0.707
			var k := 1.0 - 0.10 * d * d
			cols.append(Color(paper.r * k, paper.g * k, paper.b * k, paper.a))
	for j in n:
		for i in n:
			var a := j * (n + 1) + i
			idx.append_array([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1])
	RenderingServer.canvas_item_add_triangle_array(get_canvas_item(), idx, pts, cols)


## One panel's picture: its render target mapped onto a GRID-subdivided quad whose every
## vertex is projected individually. Two triangles would be affine and would warp (28.5 px
## on a 512 px frame, measured in tests/vehicle_probe.gd); subdivision makes the mapping
## piecewise-perspective and the error vanishes.
func _panel_quad(pg: ComicPage, i: int, pool: int, att: Vector3,
		u: float, origin: Vector2, fade: float, spine: float) -> void:
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
	var n := _grid_for(pg, i, att, u, origin, spine)
	var pts := PackedVector2Array()
	var uvs := PackedVector2Array()
	var cols := PackedColorArray()
	var idx := PackedInt32Array()
	var col := Color(1, 1, 1, fade)
	for jj in n + 1:
		for ii in n + 1:
			var uv := Vector2(float(ii) / float(n), float(jj) / float(n))
			var pr := _lens.project(_page_point(pg.panel_point(i, uv), pg, att, spine))
			if pr.z <= _lens.near:
				return
			pts.append(Vector2(pr.x, pr.y) * u + origin)
			uvs.append(inset + uv * (Vector2.ONE - inset * 2.0))
			cols.append(col)
	for jj in n:
		for ii in n:
			var a := jj * (n + 1) + ii
			idx.append_array([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1])
	RenderingServer.canvas_item_add_triangle_array(
		get_canvas_item(), idx, pts, cols, uvs,
		PackedInt32Array(), PackedFloat32Array(), rid)


## How finely to subdivide this panel's grid: enough cells that each is about CELL_PX across
## on screen. Measured off the panel's own projected size, so a panel filling the frame gets
## a fine grid and one at the edge of shot gets a coarse one.
func _grid_for(pg: ComicPage, i: int, att: Vector3, u: float, origin: Vector2,
		spine: float) -> int:
	var lo := Vector2(1e9, 1e9)
	var hi := Vector2(-1e9, -1e9)
	for uv in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		var pr := _lens.project(_page_point(pg.panel_point(i, uv), pg, att, spine))
		if pr.z <= _lens.near:
			return GRID_MAX          # wrapping the eye: the most extreme case there is
		var p := Vector2(pr.x, pr.y) * u + origin
		lo = lo.min(p)
		hi = hi.max(p)
	var span := maxf(hi.x - lo.x, hi.y - lo.y)
	return clampi(int(ceil(span / CELL_PX)), GRID_MIN, GRID_MAX)


## The gutter side of a panel: paper painted back over the square corners the rounded
## outline cuts off, then the ink border on top.
##
## Rounding this way rather than by clipping the texture is exact under perspective and
## costs nothing: the corner wedges are computed in PAGE space and projected like every
## other point on the sheet, so they land on the picture's corner however the page is
## raked. Clipping would have needed a stencil or a shader, for a shape the paper can
## simply cover.
func _panel_ink(pg: ComicPage, i: int, att: Vector3, u: float, origin: Vector2,
		paper: Color, ink: Color, filled: bool, spine: float) -> void:
	var outline := _rounded(pg, i, att, u, origin, spine)
	if outline.is_empty():
		return
	if pg.radius > 0.0004 and filled:
		for w in _corner_wedges(pg, i, att, u, origin, spine):
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
	var ring := outline.duplicate()
	ring.append(outline[0])
	draw_polyline(ring, line, _ink_width(), true)


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


## The panel's outline in screen space, corners rounded in PAGE space.
func _rounded(pg: ComicPage, i: int, att: Vector3, u: float, origin: Vector2,
		spine: float) -> PackedVector2Array:
	var out := PackedVector2Array()
	for p in _rounded_uv(pg, i):
		var pr := _lens.project(_page_point(pg.panel_point(i, p), pg, att, spine))
		if pr.z <= _lens.near:
			return PackedVector2Array()
		out.append(Vector2(pr.x, pr.y) * u + origin)
	return out


## The outline as points in the panel's own unit square. Radius is expressed in units of
## the SHORTER side and converted per axis, so a wide panel's corners stay circular
## instead of stretching into ellipses.
func _rounded_uv(pg: ComicPage, i: int) -> PackedVector2Array:
	var r: Rect2 = pg.panels[i]["rect"]
	var short := minf(r.size.x, r.size.y)
	var rad := pg.radius * short
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
func _corner_wedges(pg: ComicPage, i: int, att: Vector3, u: float, origin: Vector2,
		spine: float) -> Array:
	var r: Rect2 = pg.panels[i]["rect"]
	var short := minf(r.size.x, r.size.y)
	var rad := pg.radius * short
	var ru := rad / r.size.x
	var rv := rad / r.size.y
	var specs := [
		[Vector2(0, 0), Vector2(ru, rv), PI],
		[Vector2(1, 0), Vector2(1.0 - ru, rv), -PI * 0.5],
		[Vector2(1, 1), Vector2(1.0 - ru, 1.0 - rv), 0.0],
		[Vector2(0, 1), Vector2(ru, 1.0 - rv), PI * 0.5],
	]
	var out: Array = []
	for sp in specs:
		var corner: Vector2 = sp[0]
		var c: Vector2 = sp[1]
		var start: float = sp[2]
		var uv := PackedVector2Array([corner])
		for s in CORNER_SEGS + 1:
			var ang := start + (PI * 0.5) * float(s) / float(CORNER_SEGS)
			uv.append(c + Vector2(cos(ang) * ru, sin(ang) * rv))
		var scr := PackedVector2Array()
		var ok := true
		for p in uv:
			var pr := _lens.project(_page_point(pg.panel_point(i, p), pg, att, spine))
			if pr.z <= _lens.near:
				ok = false
				break
			scr.append(Vector2(pr.x, pr.y) * u + origin)
		if ok:
			out.append(scr)
	return out
