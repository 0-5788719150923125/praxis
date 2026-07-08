extends RefCounted
class_name MaskSession

## MaskSession - the data model for one mask-mode editing session.
##
## A session pairs one imported clip (video + its extracted audio, see masks/README)
## with a timeline of MARKERS: distinct points where the mask changes. A marker is
## not a free-form dictionary - it is a fixed-schema scalar VECTOR (see
## VECTOR_FIELDS). That is deliberate: a session's marker list is then literally a
## small matrix (one row per marker), the same shape as the harmonic-signature /
## seed-bias vectors elsewhere in this project, so it can later be inspected,
## compared, or correlated the same way instead of living as opaque nested JSON.
##
## EVERY MARKER IS ONE LAYER. A marker carries a single channel - one target color,
## one effect, one strength, one pattern placement - plus its own transition
## envelope. Layers STACK chronologically: place a marker that devours the blues,
## and a later marker adding fire to the reds layers on top WITHOUT disturbing the
## blue layer (the old two-channels-per-marker model restated every channel at
## every marker, so any later change silently rewrote what earlier markers were
## doing - colors got "restored" that nobody asked to restore).
##
## The envelope's KIND (see MARKER_KINDS) is which side of the anchor its
## transition occupies:
##   RAMP  - the layer eases in over the `duration` seconds BEFORE the anchor,
##           arriving complete exactly at the marker's time (anticipation).
##   DECAY - the layer begins AT the anchor and accumulates over the `duration`
##           seconds AFTER it - the footage is progressively consumed (an audio
##           decay envelope). Nothing happens before the anchor.
## Once in, a layer HOLDS forever. Layering is a continuous, ADDITIVE process:
## a later marker keying a second color - however near or far from the first in
## hue - stacks WITH the earlier work, never over it. (An earlier version
## silently superseded prior layers whose hue was "close enough", which meant
## keying two nearby tones made the first quietly restore itself - implicit
## magic, wrong.) The SUBTRACTIVE half is explicit: the "restore" effect (see
## MASK_EFFECTS). A restore marker draws nothing of its own - it targets a color
## exactly like a keying marker does (its own picker, its own threshold for how
## wide around that color it reaches, its own ramp/decay envelope, its intensity
## = how completely it restores) and fades out every EARLIER layer on that
## color over its window. Mask a color out at minute one, restore it at minute
## five, mask it differently at minute six - a chain of explicit operations.
## A zero-length marker of either kind is an instant cut - the degenerate case,
## not a third kind.
##
## THE TRANSITION CONTRACT. Every visual quantity that leaves at_time() must be
## transition-safe in exactly one of three standard shapes - any new modulation
## added later MUST pick one; nothing may drive a visual straight off a discrete
## snap (every "it pops instead of fading" bug so far has been a violation of
## this rule, discovered one field at a time):
##   1. CONTINUOUS  - global keying scalars (threshold, feather, sat_floor):
##      lerped across transition windows. List: GLOBAL_CONTINUOUS.
##   2. PRESENCE    - which screen layers are up (view_mode): the discrete value
##      is kept for storage/labeling, but the visuals consume the derived
##      AMOUNT_FIELDS, which lerp. A mode change is a presence fade.
##   3. LAYER       - everything a marker's own channel carries (color, effect,
##      strength, placement, coverage, contrast, resonance): baked into that
##      marker's layer, whose ENVELOPE does all the transitioning. An identity
##      change (new effect, new placement, restored color) is two layers with
##      complementary envelopes - a dissolve, never a swap, never a glide.

## The vector schema. Order is the contract - to_vector()/from_vector() and any
## future analysis code index into this list, so append, never reorder or remove.
## (The *_b fields are legacy from the two-channel era: still stored, no longer
## consumed - a marker's layer reads only the *_a channel.)
const VECTOR_FIELDS := [
	"time",             # seconds into the clip
	"kind",             # 0=ramp (eases in before the anchor) / 1=decay (accumulates after)
	"hue_a",            # THE layer's target hue, 0..1
	"hue_b",            # legacy, unused
	"threshold",        # key distance threshold, 0..1 (global)
	"feather",          # edge softness, 0..1 (global)
	"sat_floor",        # minimum saturation to key at all, 0..1 (global)
	"swap",             # legacy, unused
	"effect_a",         # THE layer's effect id, see MASK_EFFECTS
	"effect_b",         # legacy, unused
	"intensity_a",      # THE layer's strength, 0..1
	"intensity_b",      # legacy, unused
	"duration",         # seconds the ramp (before) or decay (after) envelope takes
	"view_mode",        # which rendering is shown/exported at this point - see VIEW_MODES
	"fx_x",             # THE layer's pattern pan X (unit UV)
	"fx_y",             # pattern pan Y
	"fx_scale",         # pattern zoom (1 = nominal)
	"fx_density",       # pattern coverage 0..1 (how much of the region the wisps consume)
	"resonance",        # 0..1 - how strongly the pattern breathes with the audio envelope
	"fx_contrast",      # 0..1 - wisp edge hardness; exponential response, 0.5 = neutral
]

## Global keying environment - lerped across transition windows (contract shape 1).
const GLOBAL_CONTINUOUS := ["threshold", "feather", "sat_floor"]
## What each marker's LAYER carries (contract shape 3) - baked at the marker's own
## values; only the layer's envelope varies over time.
const LAYER_FIELDS := ["hue_a", "effect_a", "intensity_a", "fx_x", "fx_y",
	"fx_scale", "fx_density", "resonance", "fx_contrast"]

const MARKER_KINDS := ["ramp", "decay"]

## The most simultaneous layers the shader renders (its uniform arrays are sized
## to this). When more are active, the OLDEST are dropped.
const MAX_LAYERS := 6

## The per-layer effect registry. "erase" hides the keyed region outright; fire /
## freeze / smoke are volumetric CONSUMING fields (see shaders/mask_split.gdshader):
## the wisps themselves are the substance - where the drifting noise field forms a
## lick, the keyed footage is eaten to void, rimmed with a glow in the layer's own
## hue; where the field is absent, the footage stays intact. fire = rising
## domain-warped licks, freeze = near-static crystalline veins, smoke = soft
## billowing gauze. Placement/coverage ride the fx_* fields; coverage 0 =
## untouched, 1 = fully devoured.
##
## "restore" is the SUBTRACTIVE operation (see the class doc): it draws nothing -
## it fades out every earlier layer whose target hue lies within ITS OWN
## `threshold` of its picked color, over its own envelope, scaled by its
## intensity (0.5 = restore halfway). The one effect the shader never sees
## (layers_at resolves it into the other layers' envelopes).
const MASK_EFFECTS := ["erase", "fire", "freeze", "smoke", "restore"]
const EFFECT_RESTORE := 4

## The view-mode registry (see mask_editor.gd). Really a 2-axis matrix flattened to
## one discrete field - main screen (raw / fx) x inset (hidden / raw / fx) - kept
## flat because it's a per-marker EXPORTABLE choice, and one scalar in the vector
## beats two half-meaningful ones. Order is append-only (indices are stored in
## saved sessions), which is why the "evolution" display order lives in
## mask_editor.gd's VIEW_CYCLE, not here:
##   pip        - main raw, inset fx (the classic compare view)
##   masked     - main fx, no inset
##   raw        - main raw, no inset (the default; no shader pass at all)
##   pip_raw    - main raw, inset raw (the frame appears, nothing effected yet)
##   masked_pip - main fx AND inset fx (the full-evolution end state)
const VIEW_MODES := ["pip", "masked", "raw", "pip_raw", "masked_pip"]

## Untouched (no markers yet) preview state. view_mode defaults to "raw" (index 2)
## - just the source video, no shader pass at all - so nothing is masked/effected
## until you explicitly place a marker or toggle the view yourself.
const DEFAULTS := {
	"kind": 0.0, "hue_a": 0.02, "hue_b": 0.58, "threshold": 0.24, "feather": 0.12,
	"sat_floor": 0.18, "swap": 0.0, "effect_a": 0, "effect_b": 0,
	"intensity_a": 1.0, "intensity_b": 0.0, "duration": 1.0, "view_mode": 2.0,
	"fx_x": 0.0, "fx_y": 0.0, "fx_scale": 1.0, "fx_density": 0.45, "resonance": 0.0,
	"fx_contrast": 0.5,
}

var video_path := ""
var audio_path := ""
var source_path := ""     # the original file this session was prepared from
var duration := 0.0
var markers: Array = []   # Array[Dictionary], sorted by "time"; each has all VECTOR_FIELDS


## A new marker at `t`, seeded from the previous marker's stored values (or
## DEFAULTS before the first) - so a fresh marker CONTINUES the same layer (same
## color: it supersedes its predecessor with identical params, visually seamless)
## until you edit it. `kind_id` indexes MARKER_KINDS.
func add_marker(t: float, kind_id: int = 0) -> Dictionary:
	var prev = null
	for mm in markers:
		if mm.time <= t:
			prev = mm
	var src: Dictionary = prev if prev != null else DEFAULTS
	var m := {}
	for key in VECTOR_FIELDS:
		m[key] = src.get(key, DEFAULTS.get(key, 0.0))
	m.time = t
	m.kind = float(kind_id)
	markers.append(m)
	markers.sort_custom(func(a, b): return a.time < b.time)
	return m


func remove_marker(m: Dictionary) -> void:
	var i := markers.find(m)
	if i >= 0:
		markers.remove_at(i)


## Shortest distance between two hues on the wrapped 0..1 circle.
static func hue_dist(a: float, b: float) -> float:
	var d := absf(a - b)
	return minf(d, 1.0 - d)


## The DERIVED per-screen-layer presence amounts (0..1) a view mode implies.
## view_mode is stored discrete (a marker names a destination look, whole), but
## what actually transitions on screen is each surface's presence - and those
## blend continuously through transition windows (contract shape 2).
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


## This marker's own IN-envelope at `t`: 0 before its window, easing to 1 across
## it, holding 1 forever after. Closed-form - no recursion, no window chaining.
##   ramp : rises over [time - duration, time]
##   decay: rises over [time, time + duration]
static func _envelope(m: Dictionary, t: float) -> float:
	var d: float = maxf(0.001, float(m.get("duration", 1.0)))
	var anchor: float = float(m.time)
	if int(m.get("kind", 0.0)) == 0:
		return clampf((t - (anchor - d)) / d, 0.0, 1.0)
	return clampf((t - anchor) / d, 0.0, 1.0)


## The layer stack at time `t`, chronological (oldest first - the shader applies
## them in order, so later layers act on top). Each entry bakes its marker's
## LAYER_FIELDS plus "env", the layer's current envelope.
##
## Layering is ADDITIVE: an applying marker never touches earlier layers,
## whatever its color. Only an explicit RESTORE marker (effect == EFFECT_RESTORE)
## closes earlier layers - every one whose target hue lies within the restore's
## own `threshold` of its picked color fades by (1 - restore_env * restore
## intensity) over the restore's window. Restores compose multiplicatively (two
## half-restores ≈ three-quarters restored) and never appear as drawn layers
## themselves. Fully-out layers are dropped; if more than MAX_LAYERS remain, the
## oldest go (the shader can't carry them, and the newest edits are the ones
## being worked on).
func layers_at(t: float) -> Array:
	var out := []
	for i in markers.size():
		var m: Dictionary = markers[i]
		if int(m.get("effect_a", 0)) == EFFECT_RESTORE:
			continue   # restores act on other layers; they draw nothing
		var env := _envelope(m, t)
		if env <= 0.0005:
			continue
		for j in range(i + 1, markers.size()):
			var mj: Dictionary = markers[j]
			if int(mj.get("effect_a", 0)) != EFFECT_RESTORE:
				continue
			var reach: float = maxf(0.02, float(mj.get("threshold", 0.24)))
			if hue_dist(float(m.get("hue_a", 0.0)), float(mj.get("hue_a", 0.0))) <= reach:
				env *= 1.0 - _envelope(mj, t) * clampf(float(mj.get("intensity_a", 1.0)), 0.0, 1.0)
		if env <= 0.0005:
			continue
		var layer := {"env": env}
		for key in LAYER_FIELDS:
			layer[key] = m.get(key, DEFAULTS.get(key, 0.0))
		out.append(layer)
	while out.size() > MAX_LAYERS:
		out.pop_front()
	return out


## The resolved state at time `t`: the GLOBAL keying scalars (lerped through
## transition windows), the discrete view_mode + its derived presence AMOUNTS
## (lerped), and "layers" - the layer stack from layers_at(). Everything a
## consumer needs to draw the frame; see the transition contract above.
func at_time(t: float) -> Dictionary:
	if markers.is_empty():
		var d0 := DEFAULTS.duplicate()
		d0.merge(mode_amounts(d0.get("view_mode", 2.0)))
		d0["layers"] = []
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
	var governing: Dictionary = cur if cur != null else DEFAULTS
	var out := {}
	out["view_mode"] = governing.get("view_mode", DEFAULTS.get("view_mode", 2.0))
	out.merge(mode_amounts(out["view_mode"]), true)
	for key in GLOBAL_CONTINUOUS:
		out[key] = float(governing.get(key, DEFAULTS.get(key, 0.0)))

	# Ramp window: the approaching marker's stage (view_mode) is up from the
	# window's start while its presence amounts and the global scalars ease in.
	# The source is whatever was ACTUALLY active as the window opened - recursed,
	# nudged strictly earlier (querying the exact boundary would resolve this same
	# window again, forever - a real stack overflow taught that lesson).
	var approaching = nxt if nxt != null else (markers[0] if cur == null else null)
	if approaching != null and int(approaching.get("kind", 0.0)) == 0:
		var d: float = maxf(0.001, float(approaching.get("duration", 1.0)))
		var span_start: float = float(approaching.time) - d
		if t >= span_start and t <= float(approaching.time):
			var src := at_time(span_start - 0.001)
			var f := (t - span_start) / d
			for key in GLOBAL_CONTINUOUS:
				out[key] = lerpf(float(src.get(key, DEFAULTS.get(key, 0.0))),
					float(approaching.get(key, DEFAULTS.get(key, 0.0))), f)
			out["view_mode"] = approaching.get("view_mode", DEFAULTS.get("view_mode", 2.0))
			var src_amt := _amounts_of(src)
			var dst_amt := mode_amounts(out["view_mode"])
			for key in AMOUNT_FIELDS:
				out[key] = lerpf(float(src_amt.get(key, 0.0)), float(dst_amt.get(key, 0.0)), f)

	# Decay window: the just-passed marker's globals/presence accumulate from the
	# prior state toward its own. f clamps at 1, so it holds once accumulated.
	if cur != null and int(cur.get("kind", 0.0)) == 1 and t >= float(cur.time):
		var d: float = maxf(0.001, float(cur.get("duration", 1.0)))
		var src := at_time(float(cur.time) - 0.001)
		var f := clampf((t - float(cur.time)) / d, 0.0, 1.0)
		for key in GLOBAL_CONTINUOUS:
			out[key] = lerpf(float(src.get(key, DEFAULTS.get(key, 0.0))),
				float(cur.get(key, DEFAULTS.get(key, 0.0))), f)
		var src_amt := _amounts_of(src)
		var dst_amt := mode_amounts(cur.get("view_mode", 2.0))
		for key in AMOUNT_FIELDS:
			out[key] = lerpf(float(src_amt.get(key, 0.0)), float(dst_amt.get(key, 0.0)), f)

	out["layers"] = layers_at(t)
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
