extends Node2D
class_name Medium

## Medium - what the show is carried ON. The presentation axis.
##
## ghost has always had exactly one presentation and never had to name it: the
## [Director] paints ONE scene, full-bleed, edge to edge, and cuts to the next. A
## medium is that choice made addressable. [FullMedium] is that behaviour,
## unchanged and default; [ComicMedium] renders the same scenes into the panels of
## an open comic book - two facing pages across a spine - and flies a real perspective
## camera over it.
##
## NOT A MODE, and the distinction is the whole point. A mode decides what DRIVES the
## show - a song (Auto), a storyboard (Manual), a written script (Synthesis /
## Generative). A medium decides what the show is PRESENTED AS. They are independent
## axes, so every mode gets every medium for free, exactly as every scene gets every
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
##   owns_bookend()       - true when the medium applies the whole-show fade itself,
##                          so the Director stops applying it to the scene.
##
## A medium is a Node2D mounted INSIDE the stage SubViewport, so the stage governor
## still owns it: when the governor stops the stage, it stops the medium and every
## panel viewport nested under it, together.

## The registry. Adding a presentation is an entry here, never new control flow in the
## Director (see the four vetoes above) - the same rule [Layer] and [Primitives] follow.
##
## Keys are what `--medium NAME` and `[director] medium` in `user://ghost.cfg` take.
const REGISTRY := {
	"full": "res://scripts/media/full.gd",
	"comic": "res://scripts/media/comic.gd",
	"book": "res://scripts/media/book.gd",
	"notebook": "res://scripts/media/notebook.gd",
}

## Display names for the registry keys, in registry order - for the settings surface.
const LABELS := {
	"full": "Full frame",
	"comic": "Comic book",
	"book": "Novel",
	"notebook": "Notebook",
}

## WHICH OPTIONAL SETTINGS EACH MEDIUM ACTUALLY USES, as a list of tags per key.
##
## A medium decides what the show is presented AS, so some of the Director's picture
## settings mean nothing to some of them: the full-frame show has no camera to be severe with
## and no panels to cut footage into, and the two controls for those sat on the panel doing
## nothing whichever medium was chosen. Reported as "there are a number of settings
## currently being displayed that ONLY work with the comic book medium".
##
## DECLARED HERE RATHER THAN BRANCHED ON IN THE PANEL, which is the same rule the rest of this
## file follows: adding a presentation is an entry in these tables, never new control flow
## somewhere else. A medium added later says what it uses and the panel follows; a panel that
## asked `if key == "comic"` would have to be found and edited instead, and it is not the file
## anyone would think to look in.
##
## Tags are the panel's own group names - a setting that EVERY medium uses (scene hold,
## flourishes, the Look filters, the bookend holds) needs no tag and is never listed.
const USES := {
	"full": [],
	"comic": ["camera", "films"],
	"book": ["camera", "illustrations"],
	"notebook": ["camera", "illustrations", "handwriting"],
}

## Does [param key]'s medium use the [param feature] group? Unknown media and unknown
## features answer TRUE, deliberately: the failure of a wrong answer here is a control the
## author cannot find, and showing a setting that does nothing is the lesser of the two.
static func uses(key: String, feature: String) -> bool:
	if not USES.has(key):
		return true
	return feature in (USES[key] as Array)


## One line each, for the toggle's tooltip and docs/media.md. Deliberately ONE string
## literal per entry, not a `+` continuation: docs.py reads this table with a regex and a
## continuation silently truncates the blurb at the first line.
const BLURBS := {
	"full": "One scene at a time, filling the frame. The original show.",
	"book": "The chapter itself, typeset into the pages of an open novel on a desk, with each word lit as it is spoken and the leaf turning as the reading reaches the next spread. Pictures come from the chapter's image markers.",
	"notebook": "The chapter handwritten into a ruled research notebook - margin times, underlined emphasis, photos paper-clipped over the writing at an angle, and sketches (`<!-- sketch: ... -->`) drawn onto the page in the same ink. For a chapter drafted as a journal or a lab report.",
	"comic": "The same scenes drawn into the panels of an open comic book - two facing pages, flown over by a real perspective camera. Each cut fills the next panel; a full spread turns the leaf on its spine.",
}


## Build the medium for [param key], falling back to `full` for an unknown one (a
## stale config or a typo must never be able to stop a session starting).
static func make(key: String) -> Medium:
	var path: String = REGISTRY.get(key, REGISTRY["full"])
	if not REGISTRY.has(key):
		push_warning("ghost: unknown medium '%s' - falling back to full frame" % key)
	var v: Medium = load(path).new()
	v.key = key if REGISTRY.has(key) else "full"
	return v


## The registry key this instance was built from.
var key := "full"

## The stage this medium is mounted in - the SubViewport [main] composites. Set by
## [method mount] before the Director attaches.
var stage: SubViewport = null


## Mount on the stage. The base parents itself; a medium with its own furniture
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
## point it would have written `_host.add_child(...)`, and it is the hook a medium
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


## True when this medium applies the whole-show bookend fade itself. The [Director]
## then stops folding it into the scene's alpha, because on a comic page that would
## fade one panel and leave the paper lit.
func owns_bookend() -> bool:
	return false


## True when this medium makes and owns the scenes itself, rather than showing the one
## the [Director] makes per cut.
##
## The full frame shows ONE scene, so "the scene" and "the show" are the same object and
## the Director can own it. A comic page has several on it AT ONCE - that is what makes it
## a page rather than a slideshow - so the page owns its cast and the Director's cut means
## "move the reading to the next panel" instead of "build a new scene". A medium that
## says true here must answer [method take_over], and the Director then never creates,
## parents or frees a scene on its own.
func owns_cast() -> bool:
	return false


## The scene the show is now ON, for a medium that owns its cast. Returning non-null
## means: this scene is ALREADY BUILT AND PARENTED, adopt it as current, and leave
## [param outgoing] alone - it is still on the page and must not be freed. [param outgoing]
## is null on the first call of a session.
##
## Returning null keeps the Director's own behaviour, so a medium can decline a
## particular change without giving up ownership.
func take_over(_outgoing: GhostScene) -> GhostScene:
	return null


## The session's words and their clock, handed over when the karaoke overlay is made.
## Return TRUE when this medium shows the words itself, and main hides the overlay - a page
## that prints the text under a subtitle of the same text would say every sentence twice.
func bind_captions(_subs) -> bool:
	return false


## Per-frame, after the Director has advanced the schedule. [param features] is the
## live [AudioFeatures]; [param bookend] is the whole-show fade, 1 except at the ends.
func advance(_features, _delta: float, _bookend: float) -> void:
	pass


## The stage changed size (a window resize, or the export's fixed render size being
## applied). A medium that sizes its own render targets off the stage re-sizes them here.
func on_stage_resized(_size: Vector2) -> void:
	pass


## Session teardown (the Director detached). Release anything held.
func release() -> void:
	pass
