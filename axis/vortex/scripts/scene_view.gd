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

## How fast the actual transform chases the target (per second). Lower = gentler.
var smoothing: float = 5.0

# Smoothed actuals (what matrix() draws).
var _zoom := 1.0
var _rot := 0.0
var _skew := 0.0
var _off := Vector2.ZERO


## Ease the actual transform toward the target. Called once per frame (and during
## pre-warm) by the Director, after the scene and any transition have written.
func commit(dt: float) -> void:
	var a := 1.0 - exp(-smoothing * dt)
	_zoom = lerpf(_zoom, zoom, a)
	_rot = lerpf(_rot, rotation, a)
	_skew = lerpf(_skew, skew, a)
	_off = _off.lerp(offset, a)


## The transform a scene pushes at the top of _draw, for the given viewport size.
func matrix(size: Vector2) -> Transform2D:
	var origin := size * 0.5 + _off * size
	return Transform2D(_rot, Vector2(_zoom, _zoom), _skew, origin)
