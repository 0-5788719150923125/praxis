extends RefCounted
class_name SceneView

## SceneView - the camera every scene draws through.
##
## A scene draws in a coordinate system whose origin (0,0) is the screen center;
## this object maps that centered space to pixels: zoom in/out, tilt (skew), roll,
## and slide off-center. Scenes/shots set the *target* fields each frame; the
## actual transform eases toward them in commit(), so every camera move is smooth
## and gentle rather than snapping - cinematic, not jittery.

## Target framing (set by the shot, the scene's drift, and transitions).
var zoom: float = 1.0
var rotation: float = 0.0
var skew: float = 0.0
var offset: Vector2 = Vector2.ZERO

## 0..1 transition presence (1 = fully on screen).
var presence: float = 1.0

## How much of a scene's BUILT GEOMETRY is on the paper, 0..1 - and a RATCHET, which is the
## entire point of it existing separately from [member presence].
##
## Several scenes hold their contents back while they fade up, so a metropolis grows out of its
## landscape and a canopy grows out of its trunks instead of the whole thing arriving at once on
## the first frame. They each did that with `smoothstep(0.8, 1.0, presence)` - which is right on
## the way in and badly wrong on the way out, because presence falls through the same window on a
## fade to black. Every building in the city vanished the instant the scene reached 0.8 alpha,
## and the viewer then watched an empty landscape fade out for the rest of the transition:
## "the buildings disappear, and THEN the scene fades out."
##
## So it only ever rises. The Director arms it at zero when it puts a scene on screen behind a
## fade ([method Director._begin_transition]) and it climbs as that scene arrives; nothing lowers
## it again, so a scene fades out with everything it had still drawn on it, which is what a fade
## is supposed to mean. A scene that arrives on a CUT never had a fade to grow through and starts
## at 1, fully built, which is also what a cut is supposed to mean.
var reveal: float = 1.0

## Where the ratchet reaches full: geometry finishes arriving over the last fifth of the fade-up,
## after the scene's ground and atmosphere are already established.
const REVEAL_AT := 0.8

## Composition bias, applied ON TOP of the scene's own framing (the Director sets it during a
## LAYER transition to push two overlapping scenes to opposite regions so they don't collide at
## the focal point). The scene never touches these, so its own drift rides on top.
var bias_offset: Vector2 = Vector2.ZERO
var bias_zoom: float = 1.0

## Punch channel: a momentary zoom / roll / skew applied INSTANTLY (not eased - the Director
## drives the envelope) on top of everything else. This is how a rapid-fire modulation "stinger"
## contorts and zooms the current scene on the beat without easing-lag. Neutral = identity.
var pulse_zoom: float = 1.0
var pulse_rot: float = 0.0
var pulse_skew: float = 0.0

## How fast the actual transform chases the target (per second). Lower = gentler.
var smoothing: float = 5.0

# Smoothed actuals (what matrix() draws).
var _zoom := 1.0
var _rot := 0.0
var _skew := 0.0
var _off := Vector2.ZERO
var _bias_off := Vector2.ZERO
var _bias_zoom := 1.0


## Ease the actual transform toward the target. Called once per frame (and during
## pre-warm) by the Director, after the scene and any transition have written.
func commit(dt: float) -> void:
	# The geometry ratchet (see [member reveal]) - it tracks presence UP and never back down.
	reveal = maxf(reveal, smoothstep(REVEAL_AT, 1.0, presence))
	var a := 1.0 - exp(-smoothing * dt)
	_zoom = lerpf(_zoom, zoom, a)
	_rot = lerpf(_rot, rotation, a)
	_skew = lerpf(_skew, skew, a)
	_off = _off.lerp(offset, a)
	_bias_off = _bias_off.lerp(bias_offset, a)
	_bias_zoom = lerpf(_bias_zoom, bias_zoom, a)


## Snap the actual transform straight onto the target - no easing. Called at the END of a scene's
## pre-warm so the FIRST shown frame is already settled: otherwise the ease only reaches ~95% in the
## pre-warm's frames and the remaining slide plays out live, reading as the whole scene "shifting" into
## place during the fade-in.
func snap() -> void:
	_zoom = zoom
	_rot = rotation
	_skew = skew
	_off = offset
	_bias_off = bias_offset
	_bias_zoom = bias_zoom


## The zoom actually in force this frame - the eased value with the transition bias and
## the stinger punch folded in, i.e. exactly the scale [method matrix] will apply.
##
## A scene that must cover the WHOLE frame whatever the camera is doing (a full-bleed
## ground; see GhostScene.paint_ground) needs this: at zoom z, the visible half-extent
## in the scene's own coordinates is size * 0.5 / z, so a quad sized off the raw
## viewport falls short the moment the camera pulls back.
func zoom_actual() -> float:
	return _zoom * _bias_zoom * pulse_zoom


## The transform a scene pushes at the top of _draw, for the given viewport size.
func matrix(size: Vector2) -> Transform2D:
	var origin := size * 0.5 + (_off + _bias_off) * size
	var z := _zoom * _bias_zoom * pulse_zoom
	return Transform2D(_rot + pulse_rot, Vector2(z, z), _skew + pulse_skew, origin)
