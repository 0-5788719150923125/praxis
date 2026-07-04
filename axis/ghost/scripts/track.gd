extends RefCounted
class_name Track

## Track - the timeline runner for data-driven scenes (see scenes/stage.gd).
##
## A track is a list of SPANS, each applying one [Actions] verb to named actors over
## a time window. Times are authored in seconds of the track's NOMINAL design length;
## at runtime they scale by `phase_span(nominal) / nominal`, so a storyboard's `hold`
## and `sensitivity` compress the whole timeline proportionally and every event still
## lands - the FL_* fraction idiom the bespoke scenes used, made data. (Only the
## keyframe clock scales; the actors' ambient life runs on the raw delta.)
##
## Span fields (storyboard data):
##   at            - a point event, or `from` / `to` for a window (nominal seconds)
##   action        - an [Actions] registry key
##   target        - actor id, list of ids, or "all" (the default)
##   args          - the verb's config (ranges welcome)
##   ease          - linear / smooth / in / out / spike (window fraction shaping)
##   on            - optional musical cue gate: beat / movement / lull. The span ARMS
##                   at its start time and fires on the cue's next rising edge, so an
##                   event lands ON the music (the drop's burst on the actual beat)
##   by            - cue backstop, seconds after arming (default 1.5): if the cue
##                   never comes, fire anyway - a tightly-timed piece can't stall
##   sustain       - true = after the window closes, keep applying at k = 1 for the
##                   rest of the scene instead of finishing. For motion that must
##                   never park (the finale keeps FLYING while the end card holds)
##
## All range sampling draws from per-span rngs seeded off the scene rng in
## declaration order: same seed, same show, cue jitter aside.

const MOVEMENT_THRESHOLD := 0.6      # the Director's defaults for the same cues
const LULL_THRESHOLD := 0.12
const CUE_BACKSTOP := 1.5

var nominal := 8.0
var spans: Array = []
var _beat_prev := 0.0


func _init(data: Dictionary, rng: RandomNumberGenerator) -> void:
	nominal = maxf(0.05, float(data.get("nominal", 8.0)))
	for sd in data.get("spans", []):
		if typeof(sd) != TYPE_DICTIONARY:
			continue
		var span := Span.new()
		span.t0 = float(sd.get("at", sd.get("from", 0.0)))
		span.t1 = maxf(span.t0, float(sd.get("to", span.t0)))
		span.cue = String(sd.get("on", ""))
		span.by = float(sd.get("by", CUE_BACKSTOP))
		span.ease = String(sd.get("ease", "smooth"))
		span.sustain = bool(sd.get("sustain", false))
		var child := RandomNumberGenerator.new()
		child.seed = rng.randi()
		span.action = Actions.make(String(sd.get("action", "")), sd.get("args", {}), child)
		var tgt: Variant = sd.get("target", "all")
		if typeof(tgt) == TYPE_ARRAY:
			for t in tgt:
				span.targets.append(String(t))
		elif String(tgt) != "all":
			span.targets.append(String(tgt))
		if span.action != null:
			spans.append(span)


## Advance the timeline to scene-time [param t] (seconds since the scene began),
## with [param span_h] = the scene's phase_span(nominal) - the real-seconds window
## the nominal timeline is scaled into.
func advance(t: float, span_h: float, stage, f: AudioFeatures, dt: float) -> void:
	var sc := span_h / nominal
	for span in spans:
		var sp: Span = span
		if sp.done:
			continue
		var start := sp.t0 * sc
		var dur := (sp.t1 - sp.t0) * sc
		if sp.cue != "":
			if sp.fire_t < 0.0:
				if t < start:
					continue
				if _cue_fires(sp.cue, f) or t >= start + sp.by * sc:
					sp.fire_t = t
				else:
					continue
			start = sp.fire_t
		if t < start:
			continue
		var actors := _resolve(sp, stage)
		if not sp.entered:
			sp.entered = true
			sp.action.begin(stage, actors)
		var k := 1.0 if dur <= 0.0001 else clampf((t - start) / dur, 0.0, 1.0)
		sp.action.apply(stage, actors, _eased(k, sp.ease), f, dt)
		if k >= 1.0 and not sp.sustain:
			sp.action.finish(stage, actors)
			sp.done = true
	_beat_prev = f.beat


## True once every span has run to completion (a oneshot stage can exit on it).
func finished() -> bool:
	for span in spans:
		if not (span as Span).done:
			return false
	return true


func _cue_fires(cue: String, f: AudioFeatures) -> bool:
	match cue:
		"movement":
			return f.movement >= MOVEMENT_THRESHOLD
		"lull":
			return f.energy <= LULL_THRESHOLD
		_:
			return f.beat > 0.5 and _beat_prev <= 0.5    # beat: rising edge
	return false


func _resolve(sp: Span, stage) -> Array:
	if sp.targets.is_empty():
		return stage.actors()
	var out := []
	for id in sp.targets:
		var a = stage.actor(id)
		if a != null:
			out.append(a)
	return out


func _eased(k: float, ease: String) -> float:
	match ease:
		"linear":
			return k
		"in":
			return k * k
		"out":
			return 1.0 - (1.0 - k) * (1.0 - k)
		"spike":
			return sin(PI * k)                            # up and back down
		_:
			return smoothstep(0.0, 1.0, k)


class Span extends RefCounted:
	var t0 := 0.0                # nominal-seconds window
	var t1 := 0.0
	var cue := ""                # "" | beat | movement | lull
	var by := 1.5                # cue backstop (nominal seconds after arming)
	var ease := "smooth"
	var sustain := false         # keep applying at k = 1 forever instead of finishing
	var action: Actions.Action
	var targets: Array = []      # ids; empty = all
	var entered := false
	var done := false
	var fire_t := -1.0           # real scene-time the cue landed (cue spans only)
