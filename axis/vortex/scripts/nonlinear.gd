extends RefCounted
class_name Nonlinear

## Nonlinear - the shared library of activation / response curves.
##
## Nonlinearity is what makes a visualization feel *alive*. A linear `value * gain`
## reads as uniform and mechanical; a response passed through a curve has threshold,
## onset, and saturation - it ignores weak signal, leans into the middle, and tops
## out gracefully, the way a living thing reacts. (The same sigmoid sits at the
## heart of fvn's swarm, which is exactly why it felt alive.)
##
## This is a tiny stateless library so every scene shapes its drive the *same* way -
## one coherent feel across the whole system instead of each scene rolling its own
## ad-hoc easing. Pass a curve name and an input; `k` sets the steepness.

## The available curve names (for seeded random pick / docs).
const CURVES := ["linear", "sigmoid", "tanh", "softplus", "gauss",
	"smoothstep", "spike", "swish", "relu", "sine"]


## A nonlinear response curve. `x` is the input (curves assume roughly -1..1 or
## 0..1 depending on use), `k` the steepness / sharpness. Unknown name = identity.
static func apply(kind: String, x: float, k := 1.0) -> float:
	match kind:
		"sigmoid":
			# Centred at 0, 0..1 - smooth threshold. Shift x to move the knee.
			return 1.0 / (1.0 + exp(-k * x))
		"tanh":
			# Centred at 0, -1..1 - signed saturation.
			return tanh(k * x)
		"softplus":
			# Soft, one-sided ramp (a smooth ReLU) - growth that eases off the floor.
			return log(1.0 + exp(clampf(k * x, -30.0, 30.0))) / k
		"gauss":
			# A bump at 0 - peaks in the middle, falls off both sides (hotspots).
			return exp(-k * x * x)
		"smoothstep":
			# Classic ease over 0..1.
			return smoothstep(0.0, 1.0, x)
		"spike":
			# Sharp onset then quick saturation in 0..1 - emphatic beat response.
			var c := clampf(x, 0.0, 1.0)
			return 1.0 - pow(1.0 - c, maxf(1.0, k))
		"swish":
			# x * sigmoid(x) - self-gated, dips slightly negative then rises (lively).
			return x / (1.0 + exp(-k * x))
		"relu":
			return maxf(0.0, x)
		"sine":
			# 0..1 oscillation - for cyclic / breathing modulation.
			return 0.5 + 0.5 * sin(k * x)
		_:
			return x


## Asymmetric envelope follower: ease `level` toward `target` fast when rising and
## slow when falling. The temporal nonlinearity behind anything that "flares and
## fades" - beat glow, growth surges, activation. Generalises the fast-attack /
## slow-decay EMA that was hand-rolled in Lighting, Activation, and the rock scenes.
static func flare(level: float, target: float, dt: float, attack := 8.0, release := 1.5) -> float:
	var rate := attack if target > level else release
	return lerpf(level, target, 1.0 - exp(-rate * dt))
