extends RefCounted
class_name MaskSession

## MaskSession - the data model for one mask-mode editing session.
##
## A session pairs one imported clip (video + its extracted audio, see masks/README)
## with a timeline of MARKERS: distinct points where the split/effect changes. A
## marker is not a free-form dictionary - it is a fixed-schema scalar VECTOR (see
## VECTOR_FIELDS). That is deliberate: a session's marker list is then literally a
## small matrix (one row per marker), the same shape as the harmonic-signature /
## seed-bias vectors elsewhere in this project, so it can later be inspected,
## compared, or correlated the same way instead of living as opaque nested JSON.
##
## Every marker is one of two KINDS (see MARKER_KINDS) - there is no plain/neutral
## marker. A change from one state to another is always either a RAMP (this
## marker's own values are blended UP TO, arriving exactly at its own time - the
## span is the `duration` seconds BEFORE it) or a DECAY (this marker's values hold
## at full strength exactly at its own time, then blend AWAY toward whatever comes
## next - the span is the `duration` seconds AFTER it). The kind is the whole
## direction: no separate left/right field, because which side of the marker the
## blend occupies IS what "ramp" vs "decay" means. Outside its own span, a marker
## just holds its value - so a ramp marker with duration 0 is an instant cut
## approaching from a held predecessor, and a decay with duration 0 is an instant
## cut into whatever follows; the natural degenerate case of a jump-cut is just a
## zero-length one of either kind, not a third case to special-case.
##
## Discrete fields (effect ids, swap, view_mode) can't have a "half-way" value, so
## they always snap exactly at a marker's own time regardless of its kind -
## continuous fields (hue, threshold, intensity, ...) are what ramp/decay actually
## shapes. See at_time for the full blend.

## The vector schema. Order is the contract - to_vector()/from_vector() and any
## future analysis code index into this list, so append, never reorder or remove.
const VECTOR_FIELDS := [
	"time",             # seconds into the clip
	"kind",             # 0=ramp (blends up to this marker) / 1=decay (blends away from it) - see MARKER_KINDS
	"hue_a",            # side A reference hue, 0..1
	"hue_b",            # side B reference hue, 0..1
	"threshold",        # key distance threshold, 0..1
	"feather",          # edge softness, 0..1
	"sat_floor",        # minimum saturation to key at all, 0..1
	"swap",             # 0/1 - which physical side wears which key color (discrete)
	"effect_a",         # side A effect id, see MASK_EFFECTS (discrete)
	"effect_b",         # side B effect id (discrete)
	"intensity_a",      # side A effect strength, 0..1 (continuous - this is what a ramp/decay shapes)
	"intensity_b",      # side B effect strength, 0..1 (continuous)
	"duration",         # seconds the ramp (before) or decay (after) span takes
	"view_mode",        # which rendering is shown/exported at this point (discrete) - see VIEW_MODES
]

## Fields that snap exactly at a marker's own time - no "half-way" value makes sense.
const DISCRETE_FIELDS := ["swap", "effect_a", "effect_b", "view_mode"]
## Fields a ramp/decay span actually eases - everything else in VECTOR_FIELDS minus
## "time", "kind", "duration" (structural, not blendable) and DISCRETE_FIELDS.
const CONTINUOUS_FIELDS := ["hue_a", "hue_b", "threshold", "feather", "sat_floor", "intensity_a", "intensity_b"]

const MARKER_KINDS := ["ramp", "decay"]

## The per-channel effect registry (see next/ for the rest of the spectrum this is
## designed to grow into - fire and freeze are the first two, not the only two).
const MASK_EFFECTS := ["erase", "fire", "freeze"]

## The view-mode registry (see mask_editor.gd's _apply_view_mode_id). Really a 2-axis
## matrix flattened to one discrete field - main screen (raw / fx) x inset (hidden /
## raw / fx) - kept flat because it's a per-marker EXPORTABLE choice, and one scalar
## in the vector beats two half-meaningful ones. Order is append-only (indices are
## stored in saved sessions), which is why the "evolution" display order lives in
## mask_editor.gd's VIEW_CYCLE, not here:
##   pip        - main raw, inset fx (the classic compare view)
##   masked     - main fx, no inset
##   raw        - main raw, no inset (the default; no shader pass at all)
##   pip_raw    - main raw, inset raw (the frame appears, nothing effected yet)
##   masked_pip - main fx AND inset fx (the full-evolution end state)
const VIEW_MODES := ["pip", "masked", "raw", "pip_raw", "masked_pip"]

## Untouched (no markers yet) preview state. view_mode defaults to "raw" (index 2)
## - just the source video, no shader pass at all - so nothing is masked/effected
## until you explicitly place a marker or toggle the view yourself. Channel 1
## targets red at full strength (inert until the view actually shows an fx layer);
## channel 2 is off (intensity 0) - one channel is the default working set, the
## second is opt-in (see mask_editor.gd's second-channel section).
const DEFAULTS := {
	"kind": 0.0, "hue_a": 0.02, "hue_b": 0.58, "threshold": 0.24, "feather": 0.12,
	"sat_floor": 0.18, "swap": 0.0, "effect_a": 0, "effect_b": 0,
	"intensity_a": 1.0, "intensity_b": 0.0, "duration": 1.0, "view_mode": 2.0,
}

var video_path := ""
var audio_path := ""
var source_path := ""     # the original file this session was prepared from
var duration := 0.0
var markers: Array = []   # Array[Dictionary], sorted by "time"; each has all VECTOR_FIELDS


## A new marker at `t`, seeded from the values active at that instant (so inserting
## one mid-timeline starts as a no-op blend point, not a jump back to defaults).
## `kind_id` indexes MARKER_KINDS.
func add_marker(t: float, kind_id: int = 0) -> Dictionary:
	var m := at_time(t)
	m.time = t
	m.kind = float(kind_id)
	markers.append(m)
	markers.sort_custom(func(a, b): return a.time < b.time)
	return m


func remove_marker(m: Dictionary) -> void:
	var i := markers.find(m)
	if i >= 0:
		markers.remove_at(i)


## The blended parameter dictionary at time `t`.
##
## Discrete fields snap to whichever marker currently governs (the one at/before
## `t`, or the first marker if `t` precedes everything - same fallback DEFAULTS
## already uses elsewhere).
##
## Continuous fields default to holding at that same governing marker's value, then
## two independent checks can override that hold:
##   - the NEXT marker, if it's a ramp, blends UP TO ITSELF over its own `duration`
##     seconds BEFORE its time - so if `t` falls in that window, we're approaching it.
##   - the CURRENT marker, if it's a decay, blends AWAY FROM ITSELF over its own
##     `duration` seconds AFTER its time, toward the next marker (or DEFAULTS, if
##     it's the last one) - so if `t` falls in that window, we're departing it.
## (Checked in that order; on the rare pathological overlap - both windows covering
## the same instant - decay wins, simply because it's evaluated second.)
func at_time(t: float) -> Dictionary:
	if markers.is_empty():
		return DEFAULTS.duplicate()
	var cur = null
	var nxt = null
	for m in markers:
		if m.time <= t:
			cur = m
		elif nxt == null:
			nxt = m

	# Bug fixed here: this used to fall back to markers[0] (the first marker's OWN
	# discrete values) before that marker's time - so view_mode would leak
	# BACKWARD across the whole prefix of the timeline before the first marker ever
	# arrived, instead of only taking effect exactly at its boundary. DEFAULTS is
	# the correct "nothing has happened yet" state, matching what CONTINUOUS_FIELDS
	# already does below - a discrete change (including which view mode is shown)
	# resets to default and then snaps at each marker's own time, not before it.
	var discrete_src: Dictionary = cur if cur != null else DEFAULTS
	var out := {}
	for key in DISCRETE_FIELDS:
		out[key] = discrete_src.get(key, DEFAULTS.get(key, 0.0))

	var base: Dictionary = cur if cur != null else DEFAULTS
	for key in CONTINUOUS_FIELDS:
		out[key] = float(base.get(key, DEFAULTS.get(key, 0.0)))

	# Ramp: is the marker we're approaching (nxt, or the very first marker if we're
	# before everything) pulling us toward it right now? The source value is
	# whatever was ACTUALLY active right as this ramp's window opened - recursing
	# to at_time() just before span_start rather than just reading cur's raw stored
	# value, because cur might itself be mid-decay at that instant (a decay ending
	# exactly where the next marker's ramp begins is a real case, not just
	# hypothetical - reading cur's raw value there produced a visible glitch:
	# smoothly decay to 0, then jump back up before ramping down again). The query
	# is nudged strictly BEFORE span_start (not queried at span_start itself) -
	# querying exactly at the boundary would resolve the identical window again and
	# recurse forever (hit this the hard way: real stack overflow, not theoretical).
	var approaching = nxt if nxt != null else (markers[0] if cur == null else null)
	if approaching != null and int(approaching.get("kind", 0.0)) == 0:
		var d: float = maxf(0.001, float(approaching.get("duration", 1.0)))
		var span_start: float = float(approaching.time) - d
		if t >= span_start and t <= float(approaching.time):
			var src: Dictionary = at_time(span_start - 0.001)
			var f := (t - span_start) / d
			for key in CONTINUOUS_FIELDS:
				out[key] = lerpf(float(src.get(key, DEFAULTS.get(key, 0.0))),
					float(approaching.get(key, DEFAULTS.get(key, 0.0))), f)

	# Decay: is the marker we just passed (cur) still fading away from us right now?
	if cur != null and int(cur.get("kind", 0.0)) == 1:
		var d: float = maxf(0.001, float(cur.get("duration", 1.0)))
		var span_end: float = float(cur.time) + d
		if t >= float(cur.time) and t <= span_end:
			var dst: Dictionary = nxt if nxt != null else DEFAULTS
			var f := (t - float(cur.time)) / d
			for key in CONTINUOUS_FIELDS:
				out[key] = lerpf(float(cur.get(key, DEFAULTS.get(key, 0.0))),
					float(dst.get(key, DEFAULTS.get(key, 0.0))), f)

	return out


## This marker's own active span - [time-duration, time] for a ramp (it's pulling
## the timeline toward itself from the past), [time, time+duration] for a decay
## (it's pushing away from itself into the future). Exposed so the timeline can draw
## the span it's shading without re-deriving the ramp/decay math itself.
func marker_span(m: Dictionary) -> Vector2:
	var d: float = maxf(0.001, float(m.get("duration", 1.0)))
	var anchor: float = float(m.time)
	var is_decay: bool = int(m.get("kind", 0.0)) == 1
	return Vector2(anchor, anchor + d) if is_decay else Vector2(anchor - d, anchor)


## Flatten a marker to its scalar vector, in VECTOR_FIELDS order.
static func to_vector(m: Dictionary) -> PackedFloat32Array:
	var v := PackedFloat32Array()
	for key in VECTOR_FIELDS:
		v.append(float(m.get(key, DEFAULTS.get(key, 0.0))))
	return v


## Rebuild a marker dictionary from a vector produced by to_vector().
static func from_vector(v: PackedFloat32Array) -> Dictionary:
	var m := {}
	for i in VECTOR_FIELDS.size():
		m[VECTOR_FIELDS[i]] = v[i] if i < v.size() else DEFAULTS.get(VECTOR_FIELDS[i], 0.0)
	return m


## The whole timeline as a matrix (one row per marker, columns = VECTOR_FIELDS) -
## the shape future bias-correlation work would want to read.
func to_matrix() -> Array:
	var rows := []
	for m in markers:
		rows.append(to_vector(m))
	return rows


func to_dict() -> Dictionary:
	return {
		"source_path": source_path, "video_path": video_path, "audio_path": audio_path,
		"duration": duration, "vector_fields": VECTOR_FIELDS, "markers": markers,
	}


func save(path: String) -> bool:
	var fa := FileAccess.open(path, FileAccess.WRITE)
	if fa == null:
		return false
	fa.store_string(JSON.stringify(to_dict(), "\t"))
	return true


static func load(path: String) -> MaskSession:
	if not FileAccess.file_exists(path):
		return null
	var fa := FileAccess.open(path, FileAccess.READ)
	var parsed = JSON.parse_string(fa.get_as_text())
	if typeof(parsed) != TYPE_DICTIONARY:
		return null
	var s := MaskSession.new()
	s.source_path = String(parsed.get("source_path", ""))
	s.video_path = String(parsed.get("video_path", ""))
	s.audio_path = String(parsed.get("audio_path", ""))
	s.duration = float(parsed.get("duration", 0.0))
	for m in parsed.get("markers", []):
		s.markers.append(m)
	s.markers.sort_custom(func(a, b): return a.time < b.time)
	return s
