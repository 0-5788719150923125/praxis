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
## marker. BOTH kinds are a transition TO the marker's own values; the kind is
## which side of the anchor the transition occupies:
##   RAMP  - eases in over the `duration` seconds BEFORE the anchor, arriving
##           complete exactly at the marker's time (anticipation - the change
##           builds toward a known landing point).
##   DECAY - begins AT the anchor and accumulates over the `duration` seconds
##           AFTER it - the prior state decays INTO this marker's values (the
##           underlying footage is progressively consumed by the effect, like an
##           audio decay envelope). Nothing happens before the anchor.
## Once its transition completes, a marker's values simply hold until the next
## marker's transition takes over. A zero-length marker of either kind is an
## instant cut - the natural degenerate case, not a third kind to special-case.
## (To fade an effect OUT, transition to a marker whose intensity is 0 - fading
## out is just a transition whose destination happens to be "nothing".)
##
## Discrete fields (effect ids, swap, view_mode) can't have a "half-way" value:
## for a decay they snap at the anchor (where its transition begins); for a ramp
## they snap at the START of its window (the arriving marker's stage has to be up
## while its intensities ease in - see at_time). Continuous fields (hue,
## threshold, intensity, ...) are what the transition actually shapes.

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


## The DERIVED per-layer presence amounts (0..1) a view mode implies. view_mode is
## stored discrete (a marker names a destination look, whole), but what actually
## transitions on screen is each LAYER's presence - and those blend continuously
## through ramp/decay windows like any other continuous quantity, so a move from
## "fx inset" to "both" fades the main overlay in over the span instead of popping
## it (the pop was exactly what made mode transitions read as instant regardless
## of the window). Keys are AMOUNT_FIELDS; at_time() carries them in its result.
const AMOUNT_FIELDS := ["main_fx", "inset_show", "inset_fx"]

static func mode_amounts(view_mode_val) -> Dictionary:
	var id := int(view_mode_val)
	return {
		"main_fx": 1.0 if (id == 1 or id == 4) else 0.0,             # masked, masked_pip
		"inset_show": 1.0 if (id == 0 or id == 3 or id == 4) else 0.0,  # pip, pip_raw, masked_pip
		"inset_fx": 1.0 if (id == 0 or id == 4) else 0.0,            # pip, masked_pip
	}


# The amounts of a resolved at_time() state: already carried in the dict if it came
# through at_time (mid-blend values), else derived fresh from its view_mode.
static func _amounts_of(state: Dictionary) -> Dictionary:
	if state.has("main_fx"):
		return state
	return mode_amounts(state.get("view_mode", 2.0))


## The blended parameter dictionary at time `t`. Carries CONTINUOUS_FIELDS,
## DISCRETE_FIELDS, and the derived AMOUNT_FIELDS (per-layer presences).
##
## Discrete fields snap to whichever marker currently governs (the one at/before
## `t`) - EXCEPT inside an approaching ramp's window, where the ramp marker's
## discrete fields take over at the window's START (the arriving marker's stage is
## up while it eases in). The per-layer AMOUNTS, though, always blend continuously
## across a window - intensity is "how strong the channel is", amounts are "how
## present each layer is", and a mode change is an amounts transition.
##
## Continuous fields (and amounts) default to holding at the governing state, then:
##   - the NEXT marker, if it's a ramp, blends UP TO ITSELF over its own `duration`
##     seconds BEFORE its time, from the state resolved at the window's start.
##   - the CURRENT marker, if it's a decay, ACCUMULATES from the prior resolved
##     state TOWARD ITS OWN values over the `duration` seconds AFTER its anchor -
##     the prior footage/state decays INTO this marker. Once accumulated, it holds
##     (`f` clamps at 1).
## (Checked in that order; on the rare pathological overlap - both windows covering
## the same instant - decay wins, simply because it's evaluated second.)
func at_time(t: float) -> Dictionary:
	if markers.is_empty():
		var d0 := DEFAULTS.duplicate()
		d0.merge(mode_amounts(d0.get("view_mode", 2.0)))
		return d0
	var cur = null
	var nxt = null
	for m in markers:
		if m.time <= t:
			cur = m
		elif nxt == null:
			nxt = m

	# DEFAULTS (not markers[0]) before the first marker: a marker's discrete values
	# must not leak backward across the timeline prefix before it ever arrives.
	var discrete_src: Dictionary = cur if cur != null else DEFAULTS
	var out := {}
	for key in DISCRETE_FIELDS:
		out[key] = discrete_src.get(key, DEFAULTS.get(key, 0.0))
	out.merge(mode_amounts(out.get("view_mode", 2.0)), true)

	var base: Dictionary = cur if cur != null else DEFAULTS
	for key in CONTINUOUS_FIELDS:
		out[key] = float(base.get(key, DEFAULTS.get(key, 0.0)))

	# Ramp: is the marker we're approaching (nxt, or the very first marker if we're
	# before everything) pulling us toward it right now? The source value is
	# whatever was ACTUALLY active right as this ramp's window opened - recursing
	# to at_time() just before span_start rather than just reading cur's raw stored
	# value, because cur might itself be mid-decay at that instant. The query is
	# nudged strictly BEFORE span_start - querying exactly at the boundary would
	# resolve the identical window again and recurse forever (hit this the hard
	# way: real stack overflow, not theoretical).
	var approaching = nxt if nxt != null else (markers[0] if cur == null else null)
	if approaching != null and int(approaching.get("kind", 0.0)) == 0:
		var d: float = maxf(0.001, float(approaching.get("duration", 1.0)))
		var span_start: float = float(approaching.time) - d
		if t >= span_start and t <= float(approaching.time):
			var src := at_time(span_start - 0.001)
			var f := (t - span_start) / d
			for key in CONTINUOUS_FIELDS:
				out[key] = lerpf(float(src.get(key, DEFAULTS.get(key, 0.0))),
					float(approaching.get(key, DEFAULTS.get(key, 0.0))), f)
			# The arriving marker's stage is already up while it eases in...
			for key in DISCRETE_FIELDS:
				out[key] = approaching.get(key, DEFAULTS.get(key, 0.0))
			# ...but each LAYER's presence fades across the window, so a mode
			# change is a gradual arrival, not a pop at the window's edge.
			var src_amt := _amounts_of(src)
			var dst_amt := mode_amounts(approaching.get("view_mode", 2.0))
			for key in AMOUNT_FIELDS:
				out[key] = lerpf(float(src_amt.get(key, 0.0)), float(dst_amt.get(key, 0.0)), f)

	# Decay: is the marker we're past (cur) still accumulating? The prior state
	# (whatever was actually on screen just before the anchor - resolved
	# recursively, nudged strictly earlier for the same no-infinite-recursion
	# reason as the ramp above) decays INTO cur's own values over the window.
	# No upper bound - f clamps at 1, so once fully accumulated it HOLDS cur's
	# values until the next marker takes over.
	if cur != null and int(cur.get("kind", 0.0)) == 1 and t >= float(cur.time):
		var d: float = maxf(0.001, float(cur.get("duration", 1.0)))
		var src := at_time(float(cur.time) - 0.001)
		var f := clampf((t - float(cur.time)) / d, 0.0, 1.0)
		for key in CONTINUOUS_FIELDS:
			out[key] = lerpf(float(src.get(key, DEFAULTS.get(key, 0.0))),
				float(cur.get(key, DEFAULTS.get(key, 0.0))), f)
		var src_amt := _amounts_of(src)
		var dst_amt := mode_amounts(cur.get("view_mode", 2.0))
		for key in AMOUNT_FIELDS:
			out[key] = lerpf(float(src_amt.get(key, 0.0)), float(dst_amt.get(key, 0.0)), f)

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
