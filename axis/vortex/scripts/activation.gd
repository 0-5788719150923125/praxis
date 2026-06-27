extends RefCounted
class_name Activation

## Activation - per-element response gating with EMA decay (who moves, and how much).
##
## When every element reacts to the audio at once, a calm track reads as constant
## twitching. Activation gives each element a seeded threshold and gain, passes the
## audio drive through a soft nonlinearity (so weak signal doesn't register), and
## smooths the result with a fast-attack / slow-decay EMA. With `sparsity > 0` some
## elements sit rooted (high threshold, effectively the static floor) while others
## swell and decay; `sparsity = 0` reverts to "everything moves". This is the knob
## that makes motion feel selective and forgiving instead of uniform.

var n: int
var _thresh := PackedFloat32Array()
var _gain := PackedFloat32Array()
var _level := PackedFloat32Array()
var _attack: float
var _decay: float
const WINDOW := 0.30   # soft-knee width above threshold


func _init(count: int, rng: RandomNumberGenerator, sparsity := 0.5, attack := 6.0, decay := 1.2) -> void:
	n = count
	_attack = attack
	_decay = decay
	_thresh.resize(n)
	_gain.resize(n)
	_level.resize(n)
	for i in n:
		_thresh[i] = rng.randf() * sparsity * 1.2   # sparsity 0 -> all thresholds 0
		_gain[i] = rng.randf_range(0.75, 1.35)
		_level[i] = 0.0


## Step every element's activation toward its target for this frame's drive.
func update(drive: float, dt: float) -> void:
	for i in n:
		var target := smoothstep(_thresh[i], _thresh[i] + WINDOW, drive * _gain[i])
		var rate := _attack if target > _level[i] else _decay   # quick to rise, slow to fade
		_level[i] = lerpf(_level[i], target, 1.0 - exp(-rate * dt))


## Smoothed activation in 0..1 for element [param i].
func level(i: int) -> float:
	return _level[i % n] if n > 0 else 1.0
