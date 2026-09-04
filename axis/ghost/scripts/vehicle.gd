extends Node2D
class_name Vehicle

## Vehicle - what the show is carried ON. The presentation axis.
##
## ghost has always had exactly one presentation and never had to name it: the
## [Director] paints ONE scene, full-bleed, edge to edge, and cuts to the next. A
## vehicle is that choice made addressable. [FullVehicle] is that behaviour,
## unchanged and default; [ComicVehicle] renders the same scenes into the panels of
## an open comic book - two facing pages across a spine - and flies a real perspective
## camera over it.
##
## NOT A MODE, and the distinction is the whole point. A mode decides what DRIVES the
## show - a song (Auto), a storyboard (Manual), a written script (Synthesis /
## Generative). A vehicle decides what the show is PRESENTED AS. They are independent
## axes, so every mode gets every vehicle for free, exactly as every scene gets every
## behavior and every render kind. Putting the comic in as a fifth mode would have
## meant a comic that only works over a song, and a second copy of the Director's
## scheduling to drive it.
##
## THE CONTRACT is deliberately four small vetoes rather than a rendering interface,
## because the Director must not learn what a comic is:
##
##   host_for(incoming)   - the node an arriving scene is added to. `full` returns the
##                          stage; `comic` returns the next panel's SubViewport, having
##                          first FROZEN the one behind it.
##   style_for(style)     - veto on the transition style the Director chose.
##   hold_outgoing()      - true when the leaving scene must NOT fade out (its panel is
##                          already drawn and must stay drawn).
##   owns_bookend()       - true when the vehicle applies the whole-show fade itself,
##                          so the Director stops applying it to the scene.
##
## A vehicle is a Node2D mounted INSIDE the stage SubViewport, so the stage governor
## still owns it: when the governor stops the stage, it stops the vehicle and every
## panel viewport nested under it, together.

## The registry. Adding a presentation is an entry here, never new control flow in the
## Director (see the four vetoes above) - the same rule [Layer] and [Primitives] follow.
##
## Keys are what `--vehicle NAME` and `[director] vehicle` in `user://ghost.cfg` take.
const REGISTRY := {
	"full": "res://scripts/vehicles/full.gd",
	"comic": "res://scripts/vehicles/comic.gd",
}

## Display names for the registry keys, in registry order - for the settings surface.
const LABELS := {
	"full": "Full frame",
	"comic": "Comic book",
}

## One line each, for the toggle's tooltip and docs/vehicles.md. Deliberately ONE string
## literal per entry, not a `+` continuation: docs.py reads this table with a regex and a
## continuation silently truncates the blurb at the first line.
const BLURBS := {
	"full": "One scene at a time, filling the frame. The original show.",
	"comic": "The same scenes drawn into the panels of an open comic book - two facing pages, flown over by a real perspective camera. Each cut fills the next panel; a full spread turns the leaf on its spine.",
}


## Build the vehicle for [param key], falling back to `full` for an unknown one (a
## stale config or a typo must never be able to stop a session starting).
static func make(key: String) -> Vehicle:
	var path: String = REGISTRY.get(key, REGISTRY["full"])
	if not REGISTRY.has(key):
		push_warning("ghost: unknown vehicle '%s' - falling back to full frame" % key)
	var v: Vehicle = load(path).new()
	v.key = key if REGISTRY.has(key) else "full"
	return v


## The registry key this instance was built from.
var key := "full"

## The stage this vehicle is mounted in - the SubViewport [main] composites. Set by
## [method mount] before the Director attaches.
var stage: SubViewport = null


## Mount on the stage. The base parents itself; a vehicle with its own furniture
## (panel viewports, a page) builds it here.
##
## SEED-INDEPENDENT WORK ONLY. Mounting happens while the stage is being created, which
## is BEFORE the Director has resolved the session seed - so anything sampled belongs in
## [method begin_session], not here. (It is the same ordering trap as reading the song
## fingerprint before the audio is loaded, and it fails silently: every session would
## roll the identical "random" page.)
func mount(st: SubViewport) -> void:
	stage = st
	st.add_child(self)


## A session is starting and [method Director.session_seed] is now valid. Roll everything
## sampled here. Called by [method Director.attach], and again for every take in the
## synthesis modes, which re-attach per take.
func begin_session() -> void:
	pass


## The node an ARRIVING scene should be added to. Called by the [Director] at every
## point it would have written `_host.add_child(...)`, and it is the hook a vehicle
## with more than one surface hangs everything off: `comic` uses the call itself as
## the signal to freeze the panel behind and open the next one.
func host_for(_incoming: GhostScene) -> Node:
	return stage


## Veto on the transition style (a [enum Director.Style] value). Return it unchanged
## to accept the Director's choice.
func style_for(style: int) -> int:
	return style


## True when the OUTGOING scene must be held at full opacity through a transition
## instead of fading out. A comic panel that is already on the paper cannot un-draw
## itself; the full frame has no such constraint.
func hold_outgoing() -> bool:
	return false


## True when this vehicle applies the whole-show bookend fade itself. The [Director]
## then stops folding it into the scene's alpha, because on a comic page that would
## fade one panel and leave the paper lit.
func owns_bookend() -> bool:
	return false


## True when this vehicle makes and owns the scenes itself, rather than showing the one
## the [Director] makes per cut.
##
## The full frame shows ONE scene, so "the scene" and "the show" are the same object and
## the Director can own it. A comic page has several on it AT ONCE - that is what makes it
## a page rather than a slideshow - so the page owns its cast and the Director's cut means
## "move the reading to the next panel" instead of "build a new scene". A vehicle that
## says true here must answer [method take_over], and the Director then never creates,
## parents or frees a scene on its own.
func owns_cast() -> bool:
	return false


## The scene the show is now ON, for a vehicle that owns its cast. Returning non-null
## means: this scene is ALREADY BUILT AND PARENTED, adopt it as current, and leave
## [param outgoing] alone - it is still on the page and must not be freed. [param outgoing]
## is null on the first call of a session.
##
## Returning null keeps the Director's own behaviour, so a vehicle can decline a
## particular change without giving up ownership.
func take_over(_outgoing: GhostScene) -> GhostScene:
	return null


## Per-frame, after the Director has advanced the schedule. [param features] is the
## live [AudioFeatures]; [param bookend] is the whole-show fade, 1 except at the ends.
func advance(_features, _delta: float, _bookend: float) -> void:
	pass


## The stage changed size (a window resize, or the export's fixed render size being
## applied). A vehicle that sizes its own render targets off the stage re-sizes them here.
func on_stage_resized(_size: Vector2) -> void:
	pass


## Session teardown (the Director detached). Release anything held.
func release() -> void:
	pass
