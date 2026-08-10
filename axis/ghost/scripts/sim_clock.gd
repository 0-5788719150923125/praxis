extends RefCounted
class_name SimClock

## SimClock - a fixed-rate tick accumulator for scenes that run a SIMULATION.
##
## Why this exists, and why it is not optional. A scene that draws a picture can
## do its work per [code]update()[/code] call and be right. A scene that advances a
## STATE - sand settling, cloth relaxing, a flock steering, a tree growing - cannot,
## because [code]update()[/code] is called at three very different rates and one of
## them is not a rate at all:
##
##   PER FRAME, VARIABLY. [Director] sub-steps the sim in fixed chunks and may run
##   up to [code]MAX_SIM_STEPS[/code] = 15 of them in a single drawn frame while
##   catching up (see director.gd's SIM_STEP / MAX_SIM_STEPS). A heavy frame would
##   therefore advance a naive simulation fifteen generations at once.
##
##   TWELVE TIMES BEFORE THE FIRST FRAME. Every new scene is PRE-WARMED so its
##   opening frame is already settled rather than visibly assembling itself. A naive
##   sand chamber is a third full before anyone has seen an empty one, and a naive
##   cloth has already torn.
##
##   UP TO 240 TIMES AT ONCE. An [Echo] re-localization can fast-forward several
##   seconds of [code]update()[/code] calls in one frame to bring the show back in
##   line with the content. That is a catastrophic amount of simulation to run
##   between two drawn frames.
##
## So the rule is: a simulation advances on WALL-CLOCK TIME, at its own fixed rate,
## with a hard ceiling on how much of it any single call may run. Surplus time is
## DROPPED rather than banked - a sim that has fallen a second behind should carry
## on from where it is, not sprint to catch up, because the sprint is exactly the
## artifact this class exists to prevent.
##
## Usage - the whole contract:
## [codeblock]
## var _sim := SimClock.new(60.0)          # 60 sim ticks per second
##
## func update(f: AudioFeatures, delta: float) -> void:
##     for _i in _sim.ticks(delta):
##         _step(_sim.dt)                  # one generation, always the same dt
## [/codeblock]
##
## [member dt] is constant, which is the other half of the point: a solver tuned at
## one timestep is not tuned at another, and a variable dt makes a Verlet cloth or a
## granular pile behave differently on a fast machine than a slow one. Everything
## downstream of this class integrates at exactly one rate.

## Ticks per second. 60 suits contact/relaxation solvers; 20-30 is plenty for a
## cellular automaton and costs proportionally less.
var rate := 60.0

## The fixed timestep every tick represents. Read this inside the step, never `delta`.
var dt := 1.0 / 60.0

## Hard ceiling on ticks emitted from ONE [method ticks] call. This is the guard
## against the pre-warm and the Echo fast-forward: whatever the caller claims has
## elapsed, a simulation may not run away inside a single call.
var max_per_call := 3

## Total ticks this clock has ever emitted. Useful for a scene that wants a stable
## integer clock (an automaton's generation count) rather than a float time.
var generation := 0

var _acc := 0.0


func _init(rate_ := 60.0, max_per_call_ := 3) -> void:
	rate = maxf(1.0, rate_)
	dt = 1.0 / rate
	max_per_call = maxi(1, max_per_call_)


## How many fixed steps to run for this much elapsed time. Call once per update and
## loop over the result; the surplus past [member max_per_call] is discarded, not
## carried, so a stalled frame cannot produce a burst on the next one.
func ticks(delta: float) -> int:
	# A negative or absurd delta is not a real elapsed time - a paused tab, a
	# debugger, a clock correction. Treat it as no time at all rather than letting
	# it fill the accumulator.
	if delta <= 0.0 or delta > 1.0:
		delta = clampf(delta, 0.0, dt)
	_acc += delta
	var n := int(_acc / dt)
	if n <= 0:
		return 0
	if n > max_per_call:
		n = max_per_call
		_acc = 0.0                      # drop the surplus; do NOT bank it
	else:
		_acc -= float(n) * dt
	generation += n
	return n


## Discard any accumulated time without emitting ticks. For a scene that has just
## been handed a discontinuity it should not simulate through (a morph hand-off, a
## storyboard seek) - the state is already where it should be.
func skip() -> void:
	_acc = 0.0


## Seconds of simulated time this clock has produced. Independent of frame rate and
## identical between a live session and an export, which is what makes a sim scene
## reproducible.
func elapsed() -> float:
	return float(generation) * dt
