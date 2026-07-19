extends Node2D
class_name GhostScene

## GhostScene - one visualizer.
##
## The base of every scene, and the heart of the pipeline. A scene is a seeded
## *definition* plus *transforms modulated by audio* - exactly the Arena /
## business-card move, pushed at sound. Two things come for free here:
##
##   mod  : a [ModBank] of organic oscillator channels (mod.value("sway")...)
##   view : a [SceneView] - zoom / tilt / rotate / off-center, drawn through
##
## Subclasses override:
##   build_params(rng) -> Dictionary   # the seeded definition
##   update(features, delta)           # set state + view, then queue_redraw()
##   _draw()                           # begin_draw(); draw shapes around (0,0)
##
## Coordinates in _draw are centered: (0,0) is the screen middle. Call
## [method begin_draw] first and the view's zoom/tilt/pan/roll apply to
## everything you draw.

## Motion behaviors - the typed "how does this scene move" axis, separate from
## *what* it draws. The same geometry can be shown frozen (pure audio reaction,
## the original look), gently drifting as a whole, or with each element moving on
## its own. The Director pairs a scene with one of these.
##   view    - how much whole-scene camera drift to apply (0 = frozen camera)
##   element - how much independent per-element motion (0 = rigid)
##   speed   - time scale for all organic motion (0 = modulators frozen)
const BEHAVIORS := {
	"static": {"view": 0.0, "element": 0.0, "speed": 0.0},
	"drift": {"view": 1.0, "element": 0.0, "speed": 1.0},
	"fluid": {"view": 0.3, "element": 1.0, "speed": 1.0},
}

## Render kind - the typed "how is this scene drawn" axis, separate from *what* it
## draws and *how it moves*. The project carries several rendering mechanisms
## forward (the "split"); naming each one makes the divergence explicit so we can
## converge on the unified 3D path. A scene declares this in build_params. One of:
##   "canvas"    - flat 2D canvas drawing (draw_* in centered screen space). Legacy.
##   "mesh3d"    - software 3D bodies ([Mesh3D]) projected onto the 2D canvas.
##   "particles" - a [ParticleSystem] substrate, drawn on the canvas.
##   "swarm"     - a [Swarm] field of many items, drawn on the canvas.
##   "scene3d"   - the unified path: a [Camera3D] + depth-sorted 3D drawables
##                 (meshes and [Plane3D] quads) under forced perspective. The target
##                 everything else migrates toward.
var render_kind := "canvas"

## The typed definition for this instance.
var params: Dictionary = {}

## The raw storyboard entry that created this scene (empty in auto mode), set by the
## [Director] before [method init_with_seed]. A data-driven scene (see scenes/stage.gd)
## builds its entire content - cast, track, camera - from this in build_params; ordinary
## scenes ignore it.
var spec: Dictionary = {}

## Identity stamped by the [Director] at creation, for telemetry / feedback capture
## (the feedback console writes these so a critique can be tied back to its scene).
var scene_name := ""
var behavior_name := "drift"
var shot_name := ""
var seed_value := 0

## Optional per-scene exit override set by the [Director] in manual (storyboard) mode:
## `{"hold": seconds}` for fixed timing, or `{"trigger": Trigger, "min":, "max":}`
## to land the cut on a chosen musical cue. Empty = the Director's default
## lifecycle + randomly-armed trigger (auto mode).
var exit_spec: Dictionary = {}

## Narrative-tempo scalar from the storyboard's `sensitivity` (see the Director), 1.0 = nominal.
## Higher = faster: the Director shrinks this scene's hold by it, and a scene that has no fixed hold
## (auto mode) divides its nominal keyframe span by it. It ONLY compresses the keyframe/phase clock -
## the ambient life of the bodies (a prism's spin, an eye's saccades) always runs on the raw delta,
## so turning sensitivity up makes events arrive sooner WITHOUT making the animation itself look fast.
var event_scale: float = 1.0

## The window (seconds) a keyframe-driven scene should pace its phases over: its fixed hold if it has
## one (already sensitivity-scaled by the Director), else its own [param nominal] design length divided
## by [member event_scale]. Scenes express keyframes as FRACTIONS of this (e.g. `_t / phase_span(8.0)`),
## so every event still lands whatever the hold - shorter holds just compress them.
func phase_span(nominal: float) -> float:
	if exit_spec.has("hold"):
		return maxf(0.05, float(exit_spec["hold"]))
	return maxf(0.05, nominal / maxf(0.05, event_scale))

## Content-aware transition typing. A scene declares the geometry it LEAVES on screen
## (`morph_out`) and the geometry it can grow IN from (`morph_in`). When the Director
## changes from A to B and `B.morph_in == A.morph_out` (and both non-empty), it plays
## a *morph* - an instant swap where B animates itself out of A's shape (e.g. one eye
## splitting into two) - instead of a cut or dip. Mismatched or empty = a normal cut.
## This is how fancy transitions are added safely: only between compatible geometries,
## so we never try to morph two things that don't line up.
var morph_out := ""
var morph_in := ""

## Transition style the [Director] uses when leaving this scene, set from the
## storyboard ("cut" / "dip" / "fade"; "" = the Director chooses). A morph, when
## compatible, always wins over this.
var transition_style := ""
## Viewport size in pixels, kept current across resizes.
var size: Vector2 = Vector2.ZERO
## Organic modulation channels (seeded). See [ModBank].
var mod: ModBank
## The camera this scene draws through. See [SceneView].
var view: SceneView
## This instance's motion behavior (one of [constant BEHAVIORS]).
var behavior: Dictionary = BEHAVIORS["drift"]

## Lifecycle - does this scene loop until the Director cuts it (`"loop"`), or play
## one self-contained sequence and end (`"oneshot"`)? A oneshot reports when its
## sequence is done via [method finished]; the Director then exits it on the next
## musical cue. A scene sets this in [method build_params] if it wants to be a
## oneshot (or to choose between the two on its seed).
var lifecycle := "loop"

## Framing class for shot selection: `"subject"` (a discrete object - gets the
## expressive cinematic shots) or `"field"` (fills the frame - gets gentle shots
## only, so panning never exposes its edges). Field scenes set this in build_params.
var framing := "subject"

## The camera framing assigned by the [Director] (see [Shots]); applied in tick().
var shot = null

## Seconds this scene has been alive - drives slow shot moves (push-in, pan).
var _life := 0.0

## Composed visual components (see [Layer]) - the appearance sibling of the physics
## [Primitives] force list. A scene adds layers in build_params (snow, fog, fireflies,
## stars, …); update_layers advances them and draw_layers paints them, in order. This
## is how weather is *integrated*: the same component is a scene on its own and an
## overlay on a city or a hillside.
var layers: Array = []
## The most recent audio frame, kept so draw-time code (and layers) can read it.
var last_f: AudioFeatures = AudioFeatures.new()


## Called once by [Director] before the scene enters the tree. Seeds everything
## deterministically and fixes the motion behavior. Do not override - override
## [method build_params] / [method seed_view].
func init_with_seed(seed_value: int, behavior_name := "drift") -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value
	self.seed_value = seed_value
	self.behavior_name = behavior_name
	behavior = BEHAVIORS.get(behavior_name, BEHAVIORS["drift"])
	mod = ModBank.new(seed_value ^ 0x5bd1e995)
	view = SceneView.new()
	params = build_params(rng)
	seed_view(rng)


## Override: roll the typed definition. Same seed -> same scene.
func build_params(_rng: RandomNumberGenerator) -> Dictionary:
	return {}


## Override (optional): seed per-scene view ranges or initial state.
func seed_view(_rng: RandomNumberGenerator) -> void:
	pass


## Override: read [param f], update state and [member view], then queue_redraw().
func update(_f: AudioFeatures, _delta: float) -> void:
	pass


## Override (oneshot scenes only): true once the scene's sequence has finished
## and it is ready to be exited. Loop scenes leave this false forever.
func finished() -> bool:
	return false


## Override (morph sources): a small typed bag of state the outgoing scene hands to
## the incoming one during a morph, so the transition is *continuous* - e.g. the eye
## passes its colour / gaze / size so the two it splits into are the SAME eye, not
## new ones. Keys are by convention; the receiver reads what it understands.
func morph_payload() -> Dictionary:
	return {}


## Override (morph targets): called by the [Director] when this scene is morphing in
## from a compatible scene (see [member morph_in]). [param from] is the outgoing
## scene - read its [method morph_payload] to continue from its state, then play the
## morph-in animation (e.g. two_eyes starts as one eye and splits apart).
func begin_morph(_from: GhostScene) -> void:
	pass


func _ready() -> void:
	size = get_viewport_rect().size
	get_viewport().size_changed.connect(_on_resize)
	# Disable premature canvas-item culling. Scenes draw through a view transform (pan /
	# zoom), so content near the frame edge - especially big soft glows / lighting whose
	# centre drifts off-screen while their halo still bleeds in - would pop out abruptly
	# when the item's auto-computed bounds crossed the viewport. A large custom rect keeps
	# the item always considered visible, so things ease off the edge instead of clipping.
	RenderingServer.canvas_item_set_custom_rect(
		get_canvas_item(), true, Rect2(-100000, -100000, 200000, 200000))


func _on_resize() -> void:
	size = get_viewport_rect().size
	queue_redraw()


## Push the view transform. Call at the top of _draw; then draw around (0,0).
func begin_draw() -> void:
	draw_set_transform_matrix(view.matrix(size))


## The shorter screen axis - use it to size geometry independent of aspect.
func unit() -> float:
	return minf(size.x, size.y)


## Set the camera framing for this scene (called once by the Director).
func set_shot(s) -> void:
	shot = s


## Advance the organic clock once per frame, scaled by the behavior's speed
## (so a "static" scene's modulators stay frozen and it reacts to audio alone),
## and apply the assigned shot's base framing. Call this at the top of update().
func tick(f: AudioFeatures, delta: float) -> void:
	_life += delta
	# The METAMORPHOSIS: while a catch is being reeled in (synthesis mode),
	# Director.aura contorts the running scene - motion tempo stretches, the
	# frame zoom-breathes and pan-sweeps, growing with the pull. Discipline-
	# safe (zoom + pan only, never roll/shear) and base-level, so it reaches
	# every scene, static behaviors included. Zero-cost when aura is 0.
	var aura: float = Director.aura
	mod.advance(delta * float(behavior.speed) * (1.0 + aura * 0.8), f.energy)
	if shot != null:
		shot.apply(view, f, _life)
	if aura > 0.001:
		# same additive convention as drift_view's zoom/offset contributions
		view.zoom += aura * 0.35 * sin(_life * 0.6 + 1.3)
		view.offset += Vector2(
			aura * 0.16 * sin(_life * 0.37),
			aura * 0.11 * sin(_life * 0.29))


## Organic camera drift, added *on top* of the shot's base framing (set in tick),
## scaled by the behavior's view gain. A "static" scene (view = 0) adds nothing
## and rides the shot alone. Deliberately only zoom + pan: rolling or shearing flat
## 2D content reads as fake 3D / spinning, so the camera never does it here -
## rotation comes only from shots, and real depth only from Mesh3D. (The `roll` and
## `tilt` parameters are kept for call-site compatibility but intentionally unused;
## a scene that genuinely wants skew sets view.skew itself, e.g. strata.)
func drift_view(f: AudioFeatures, move := 0.05, zoom := 0.08, _roll := 0.0, _tilt := 0.0) -> void:
	var g: float = behavior.view
	if g <= 0.0:
		return
	view.zoom += g * zoom * mod.value("zoom") + 0.04 * f.energy
	view.offset += Vector2(g * move * mod.value("panx"), g * move * mod.value("pany"))


## Compose a visual component by [Layer] registry key, seeded from [param rng] (use the
## one handed to build_params so the look is deterministic per session). Returns the
## layer so a scene can keep a handle on it. Layers paint back-to-front in add order.
func add_layer(key: String, rng: RandomNumberGenerator, cfg := {}) -> Layer.Base:
	var l := Layer.make(key, rng, cfg)
	layers.append(l)
	return l


## Advance every composed layer. Call from update(). Hands each layer the visible
## half-extent (in unit fractions) so it fills the frame at any aspect / resolution.
func update_layers(f: AudioFeatures, delta: float) -> void:
	last_f = f
	if layers.is_empty():
		return
	var u := maxf(1.0, unit())
	# Overdraw margin: hand layers a half-extent a bit LARGER than the frame, so they paint past every
	# edge. The camera drift (drift_view) slides/zooms the whole canvas, and without this the drawn
	# region's edge slides into view as a hard seam (a plane's border). 1.15x covers the drift range.
	var h := Vector2(size.x, size.y) / (2.0 * u) * 1.15
	for l in layers:
		l.update(f, delta, h)


## Paint composed layers onto this scene's canvas, in add order. Call from _draw() after
## begin_draw() (so the layers share the scene's view transform). [param z_filter] picks
## a depth band ("back" / "front") so a geometry scene can draw its background layers
## (stars) before its geometry and its foreground layers (snow) after; "" draws all.
func draw_layers(z_filter := "") -> void:
	if layers.is_empty():
		return
	var u := unit()
	for l in layers:
		if z_filter == "" or l.z() == z_filter:
			l.draw(self, u)


## Per-element motion primitive: an independent organic offset in -1..1 for
## element [param i], decorrelated per index and scaled by the behavior's element
## gain (0 for rigid scenes). This is how lines/shapes move on their own instead
## of the whole scene shifting as a block - the seed of the motion-primitive
## library the visualizer will grow.
func wobble(key: String, i: int) -> float:
	var g: float = behavior.element
	if g <= 0.0:
		return 0.0
	return mod.value("%s_%d" % [key, i]) * g


## Tonal colour from the live harmonic signature - the CONTINUOUS, expressive half of harmonic
## seeding (the discrete half is the seed_bias mixed into seeds; see next/harmonic_seeding.md).
## The 12 chroma bins are placed on the colour wheel and summed: the result's angle is the
## music's tonality (its "key" as a hue) and its length is how tonal the moment is. Returns
## Vector2(hue 0..1, strength 0..1); a scene can pull its palette toward the hue by the strength.
func chroma_hue() -> Vector2:
	var sig := Spectrum.harmonic_signature()
	if sig.size() < 12:
		return Vector2.ZERO
	var acc := Vector2.ZERO
	for k in 12:
		acc += Vector2(cos(TAU * float(k) / 12.0), sin(TAU * float(k) / 12.0)) * sig[k]
	return Vector2(fposmod(acc.angle() / TAU, 1.0), clampf(acc.length() * 1.6, 0.0, 1.0))
