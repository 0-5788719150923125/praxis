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
##   DAMP  - the layer begins AT the anchor and accumulates over the `duration`
##           seconds AFTER it - the footage is progressively consumed (an audio
##           damping envelope). Nothing happens before the anchor.
## Once in, a layer HOLDS - until one of exactly three things ends it:
##   1. THE RAW CHECKPOINT (the rebase). A full-raw marker (view_mode 2) is a
##      canvas reset: once its transition to raw has COMPLETED (at its anchor for
##      a ramp; anchor+duration for a damp - never mid-fade, or the cut would
##      pop), no earlier marker's layer exists beyond it. Raw is the one place a
##      reset is pop-free BY CONSTRUCTION - nothing is on screen while it
##      happens - so composition history never crosses a raw boundary: a
##      sliding-window model whose window edges are explicit, user-authored
##      timeline events instead of a silent time constant (which would kill
##      layers the author expects to hold - the fire-vanished bug class).
##   2. "restore" - fades out earlier layers matching ITS color (see below).
##   3. "clear" - restore for ALL colors at once: fades out every earlier layer
##      over its envelope, scaled by its intensity.
## Layering is otherwise a continuous, ADDITIVE process:
## a later marker keying a second color - however near or far from the first in
## hue - stacks WITH the earlier work, never over it. (An earlier version
## silently superseded prior layers whose hue was "close enough", which meant
## keying two nearby tones made the first quietly restore itself - implicit
## magic, wrong.) The SUBTRACTIVE half is explicit: the "restore" effect (see
## MASK_EFFECTS). A restore marker draws nothing of its own - it targets a color
## exactly like a keying marker does (its own picker, its own threshold for how
## wide around that color it reaches, its own ramp/damp envelope, its intensity
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
	"kind",             # 0=ramp (eases in before the anchor) / 1=damp (accumulates after)
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
	"duration",         # seconds the ramp (before) or damp (after) envelope takes
	"view_mode",        # which rendering is shown/exported at this point - see VIEW_MODES
	"pip_track",        # which video fills the PiP inset: 0 = the main clip (default),
	                    #   k = imported track (k-1). The V button cycles through these.
	"fx_x",             # THE layer's pattern pan X (unit UV)
	"fx_y",             # pattern pan Y
	"fx_scale",         # pattern zoom (1 = nominal)
	"fx_density",       # pattern coverage 0..1 (how much of the region the wisps consume)
	"resonance",        # 0..1 - how strongly the pattern breathes with the audio envelope
	"fx_contrast",      # 0..1 - wisp edge hardness; exponential response, 0.5 = neutral
	"fx_speed",         # drift velocity multiplier for the volumetric fields (1 = nominal)
	"fx_lag",           # echo: how far back the lagged frame reaches, seconds
	"fx_smooth",        # echo: 0 = discrete stutter, 1 = wide temporal blend (EMA-like)
	"fx_stick",         # fur: 0 = free coat everywhere (default look), 1 = strands cling
	                    #   to natural anchors (key-colour concentration, the tracked
	                    #   landmark/motion centroid, luminance) - see the shader's fur branch
]

## Global keying environment - lerped across transition windows (contract shape 1).
const GLOBAL_CONTINUOUS := ["threshold", "feather", "sat_floor"]
## What each marker's LAYER carries (contract shape 3) - baked at the marker's own
## values; only the layer's envelope varies over time.
const LAYER_FIELDS := ["hue_a", "effect_a", "intensity_a", "fx_x", "fx_y",
	"fx_scale", "fx_density", "resonance", "fx_contrast", "fx_speed", "fx_lag",
	"fx_smooth", "fx_stick"]

const MARKER_KINDS := ["ramp", "damp"]

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
## whisp / crystal / echo (appended - ids are the contract, EFFECT_RESTORE stays 4):
##   whisp   - content-aware volumetric: its field is advected along the underlying
##             picture's luminance edges (tendrils curl around features) and its
##             placement auto-locks to the target color's mass centroid, EMA-tracked
##             over short windows by the editor (see MaskEditor echo/anchor capture).
##   crystal - fractal faceted glass rendered IN PLACE of the target color: voronoi
##             facets refracting the footage, cold edge light. Projection-based like
##             erase (no gates - no rings possible).
##   echo    - temporal lag: the target color's region shows a muted, delayed echo
##             of the footage (a ring of past frames), optionally cloned with
##             accumulating X/Y offsets into staircase "time-shapes".
## "clear" is restore generalized to EVERY color: it draws nothing and fades out
## ALL earlier layers over its own envelope, scaled by its intensity - the
## explicit "delete the old effects" primitive (restoring region-by-region was
## the only way to unwind a stack, and it was clunky). Like restore, the shader
## never sees it.
## "snow" has NO key color - hue_a is unused. Foreground vs. background is
## decided per pixel, automatically, from motion (the echo ring's two newest
## captures) and color intensity: a lit, moving subject scores high on both, a
## static, desaturated background scores low - so the flakes fall over the
## background and thin out over the subject with no color pick required. The
## Contrast slider is relabeled "Sensitivity" for it, and Pan X/Y become Wind
## X/Y - a fall DIRECTION (default straight down), decoupled from Velocity's
## speed. Gust (fx_smooth, its own slider - see MaskEditor's "snow" group) adds
## irregular swings to that direction and speed together; echo's Smoothing
## slider uses the same stored field for a different purpose, but the two
## groups never show at once so there's no conflict.
## "fur" - hair ANCHORED on the keyed surface itself, crystal-style. Strand
## roots are the key color's own pixels, read live from the frame by the
## same aligned-fraction projection crystal replaces with (fur_root_mass()
## in mask_split.gdshader) - so the coat tracks the moving face with
## crystal's per-pixel accuracy, no tap-ring estimate, no CPU centroid. A
## pixel carries hair only if marching upstream against the local current
## lands on real root mass within a strand's reach: structurally nothing
## can draw where no march reaches the face (the old screen-space stripe
## field read as straight streaks across the whole frame). The current is
## one shared, slowly-breathing vector field (wind bias + smooth swirl)
## every strand rides together - the swarm dynamic: curved strands that
## sway coherently, mermaid hair underwater in slow motion. Pan steers the
## wind, Scale sets strand length + fineness, Velocity the tempo. Unlike
## every other volumetric here, its emissive body is tinted BY the key hue
## itself (fur colored like the thing it grew from) rather than a fixed
## palette - the one deliberate exception to the "never the key hue" rule,
## because fur being roughly the color it's keyed on is the whole point,
## not a halo. Its "fur" control group adds two tendril-dynamics knobs,
## fur-only views onto stored fields other effects repurpose the same way
## (the groups never show together): Undulation (fx_smooth) sends traveling
## waves down each strand, and Coil (fx_lag, pushed raw as u_l_lagf) spins
## the local flow frame into eddies and spirals.
## "oracle" - echo INVERTED: predictive, dominant. Echo lags the keyed
## region behind a live world; oracle keeps the keyed region LIVE and lags
## the WORLD around it through the same temporal kernel - relative to
## everything on screen the region runs ahead by the lag, leading the
## motion instead of trailing it (delaying everything else is the only
## causal way to put one region "in the future" in real time). Coverage
## adds PREMONITIONS - muted copies of the live region stamped ahead along
## the pan direction; Contrast mutes the delayed world; Lag is how far
## ahead the region leads.
## "serpent" - thousands of wriggling snakes as a sparse volumetric mask
## (a scalar field through the shared consume/rim path, meant to eat organic
## clusters out of mostly-flat footage). KEYLESS, like snow: it has no key
## color at all - a mostly-white plane has nothing to key on (the first cut
## gated on key membership and drew nothing there). Its contrast is
## LUMINANCE-INVERSE instead: the consume bites hardest where the footage is
## bright (dark snakes carved out of white), fading toward dark ground where
## the viridian emissive carries the bodies instead. Three rotated,
## domain-warped octaves of torus lanes; each lane hosts at most one snake
## (most lanes empty), every snake sampling its own length (squared - mostly
## small, a few long), speed, heading, girth, wriggle, banding, and glow
## from per-lane hashes. Centerlines are nested-sine serpentines whose wave
## travels backward along the body relative to motion (real snake
## locomotion); bodies wrap a torus lap longer than themselves, so each
## snake laps around with a gap. A slow fbm cluster field decides WHERE the
## swarm lives - the sparse, clustered distribution is the point - and the
## footage itself stirs it: frame-to-frame motion (the echo ring's captures)
## is treated as energy in the water, its wide slow crests displacing the
## snake domain and gathering the swarm along the moving model's wake.
## Emissive is a fixed viridian-to-jade palette (never the key hue).
## "chimera" - the imported track's video grafted INTO the main frame: two
## heads merged, not cut. Purely heuristic: a soft window rides u_anchor
## (keyed by this marker's color when the footage carries it; when the
## lighting is flat and nothing keys, the tracker falls back to the MOTION
## centroid - the moving head anchors itself). Per pixel, whichever side
## carries more feature energy (edges/eyes/mouth beat flat skin) owns the
## pixel, so the two heads interleave structurally instead of
## double-exposing; the graft's exposure is pulled toward the local footage
## first, and wherever the key color IS present the graft claims it
## outright. Scale = window/zoom, Pan = graft offset, Coverage = how much of
## the chimera is the other head, Contrast = interleave sharpness. Needs an
## imported track (T); draws nothing without one.
## "arealight" - cinematic area lighting: a small rig of light sources spread
## across the FULL SPAN of azimuth angles behind the camera (not one direction
## - a spread, so the subject catches something from wherever it turns), each
## waxing and waning on its own slow clock so the rig reads as alive rather
## than a static three-point setup. KEYLESS, like snow/serpent (there is no
## subject color to gate on - it lights the whole frame) and ADDITIVE-ONLY:
## nothing is consumed, only lit, unlike every volumetric effect above. Its
## local "lighting normal" is the same mid/wide gradient probe whisp/freeze/
## crystal already read as a feature-conformance cue, repurposed here as the
## only depth proxy available without real 3D geometry - edges catch the rim,
## flat regions don't. Deliberately ONE exposed dial (Contrast, relabeled
## Envelope in the editor) rather than a knob per light: it doesn't aim
## anything, it moves WHERE ALONG A LIGHTING MOOD the whole rig sits - and it
## GROWS the rig, not just recolors it. At 0 only the first source (a wide,
## soft, warm practical) is lit; the rest switch on staggered as the dial
## rises, each in its own place in a curated cinematic gel palette, the
## falloff sharpening from a wide wash to a tight hard rim as it goes. Past
## that, the same dial also unlocks discrete point flares - sparse
## gaussian-falloff light-source geometries scattered by the same per-cell
## hash+threshold masking snow/serpent use to scatter their own elements -
## none at 0, a scattered handful by 1. The angles and the waxing/waning are
## automatic; the dial changes what the light IS, how MANY sources are live,
## and whether standalone flares exist at all.
const MASK_EFFECTS := ["erase", "fire", "freeze", "smoke", "restore", "whisp", "crystal", "echo", "clear", "snow", "fur", "oracle", "serpent", "chimera", "arealight"]
const EFFECT_RESTORE := 4
const EFFECT_CRYSTAL := 6
const EFFECT_CLEAR := 8
const EFFECT_SNOW := 9
const EFFECT_FUR := 10
const EFFECT_ORACLE := 11
const EFFECT_SERPENT := 12
const EFFECT_CHIMERA := 13
const EFFECT_AREALIGHT := 14

## THE CONTROL HIERARCHY: which panel option groups each effect actually consumes
## (the editor shows/hides accordingly - a slider that does nothing for the
## selected effect is not shown for it). Color + intensity are universal. Groups:
##   "keying"  - threshold/feather/colorfulness steer the volumetric mask
##   "reach"   - threshold reinterpreted: how wide around the color a restore acts
##   "pattern" - field placement/coverage/contrast/resonance
## Erase consumes NO groups: it is projection-subtraction, gate-free by design -
## threshold, feather, colorfulness and the pattern have no effect on it at all.
const EFFECT_CONTROLS := {
	0: [],                        # erase
	1: ["keying", "pattern"],     # fire
	2: ["keying", "pattern"],     # freeze
	3: ["keying", "pattern"],     # smoke
	4: ["reach"],                 # restore
	5: ["keying", "pattern"],     # whisp
	6: ["pattern"],               # crystal (projection-based: no keying gates)
	7: ["pattern", "echo"],       # echo (pan=clone step, scale=step size, coverage=clones, contrast=mute)
	9: ["pattern", "snow"],       # snow (no keying group: it has no key color at all)
	8: [],                        # clear (intensity = how completely; color is meaningless)
	10: ["keying", "pattern", "fur"],  # fur (a keyed volumetric + its own tendril-dynamics knobs)
	11: ["pattern", "echo"],      # oracle (echo inverted: pan=premonition step, coverage=copies, contrast=mute, lag=lead)
	12: ["pattern"],              # serpent (keyless, like snow: contrast-driven, no color to gate on)
	13: ["pattern"],              # chimera (color steers the anchor/claim; scale=window, pan=graft offset, coverage=dominance, contrast=interleave sharpness)
	14: ["pattern"],              # arealight (keyless; pattern group exists only to expose the single Envelope/contrast dial)
}

## Second level of the same rule, INSIDE the "pattern" group: which individual
## knobs an effect actually reads. "Only show properties that can be used" -
## a knob the shader never touches for this effect is hidden, not just disabled.
## Keys: "scale", "pan", "coverage", "contrast", "velocity", "resonance". An
## effect NOT listed here shows every pattern knob (the historical default);
## list an effect only to PRUNE. Audit against shaders/mask_split.gdshader when
## an effect changes what it consumes.
const PATTERN_KNOBS_ALL := ["scale", "pan", "coverage", "contrast", "velocity", "resonance"]
const PATTERN_KNOBS := {
	# chimera reads scale (graft zoom), pan (graft nudge), coverage (dominance)
	# and contrast (interleave sharpness). It never reads speed, and its only
	# tie to resonance is a marginal audio-breathe on dominance - both hidden.
	13: ["scale", "pan", "coverage", "contrast"],
	# arealight reads ONLY contrast (relabeled Envelope in the editor) - the
	# whole point is one dial, not a knob per light.
	14: ["contrast"],
}

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
const VIEW_MODES := ["pip", "masked", "raw", "pip_raw", "masked_pip", "masked_pip_raw"]

## Untouched (no markers yet) preview state. view_mode defaults to "raw" (index 2)
## - just the source video, no shader pass at all - so nothing is masked/effected
## until you explicitly place a marker or toggle the view yourself.
const DEFAULTS := {
	"kind": 0.0, "hue_a": 0.02, "hue_b": 0.58, "threshold": 0.24, "feather": 0.12,
	"sat_floor": 0.18, "swap": 0.0, "effect_a": 0, "effect_b": 0,
	"intensity_a": 1.0, "intensity_b": 0.0, "duration": 1.0, "view_mode": 2.0,
	"pip_track": 0.0,
	"fx_x": 0.0, "fx_y": 0.0, "fx_scale": 1.0, "fx_density": 0.45, "resonance": 0.0,
	"fx_contrast": 0.5, "fx_speed": 1.0, "fx_lag": 0.35, "fx_smooth": 0.0,
	"fx_stick": 0.0,
}

var video_path := ""
var audio_path := ""
var source_path := ""     # the original file this session was prepared from
var duration := 0.0
var markers: Array = []   # Array[Dictionary], sorted by "time"; each has all VECTOR_FIELDS

## Primary-clip trim (in/out points, seconds into video_path/audio_path). clip_out
## of -1 means "uncut" (full duration) - the common case, so old sessions load with
## no trim applied. These are DELIBERATELY never read by the timeline's own pixel
## mapping (see MaskTimeline._visible_span/timeline_extent below) - dragging a trim
## handle must never rescale the ruler it's being dragged against; that's the
## "extreme right edge" bug every timeline editor seems to have at some point.
var clip_in := 0.0
var clip_out := -1.0

## The MAIN clip's audio level, 0 or 1 (the audio-mix toggle on its lane). Imported
## tracks carry their own per-track "volume" in the tracks array. All audio plays
## together, each gated by its own 0/1 - see mask_editor.gd's _sync_tracks / _play.
var main_volume := 1.0

## The MAIN clip's fade envelope (seconds): audio AND video ramp in over the first
## `main_fade_in` seconds and out over the last `main_fade_out`. Tracks carry their
## own "fade_in"/"fade_out" per entry. Both edges of a lane are two handles - the
## trim point and the fade boundary - see TrackLane / _apply_clip_fade.
var main_fade_in := 0.0
var main_fade_out := 0.0

## Where the playhead was, in seconds - persisted so reopening a session (or an
## auto-restart after the assistant edits code) lands you back exactly where you were.
var playhead := 0.0

## Timeline zoom/pan (see TimelineView), persisted alongside playhead so a reopen
## or auto-restart also lands back on the same zoomed-in window, not just the
## same playhead.
var timeline_zoom := 1.0
var timeline_view_start := 0.0

## Secondary tracks (picture-in-picture overlays) - see mask_editor.gd's "Import
## track" flow. Each: {video_path, duration, clip_in, clip_out, offset (seconds on
## the MASTER timeline where this track's clip_in lands), x, y, w, h (normalized
## 0..1 screen position/size), muted}. The primary clip (video_path/audio_path/
## duration/clip_in/clip_out above) is NOT stored in this array - it is "track 0"
## implicitly, the one thing every session has always had.
var tracks: Array = []


## The clip end for actual use (playback clamping, export stop condition): clip_out
## if trimmed, else the full duration. Never negative, never past duration.
func effective_clip_out() -> float:
	return clampf(clip_out if clip_out > 0.0 else duration, 0.0, duration)


## Where playback (live preview and export) actually stops: effective_clip_out(),
## extended to cover any track scheduled to run past it - e.g. _split_main's tail,
## which trims clip_out and appends the remainder as a track at that same offset so
## it keeps playing there instead of cutting the show short. Clamped to `duration`:
## the master clock is the main clip's OWN decode position (see mask_editor.gd's
## _sync_tracks), so it can never advance past the main clip's full length no
## matter how far out a track's offset reaches.
func content_end() -> float:
	var extent := effective_clip_out()
	for t in tracks:
		var span: float = maxf(0.0, float(t.get("clip_out", 0.0)) - float(t.get("clip_in", 0.0)))
		extent = maxf(extent, float(t.get("offset", 0.0)) + span)
	return minf(extent, duration)


## Index of the track that is a continuation of the main clip (same source
## video_path - produced by splitting the main clip, see mask_editor.gd's
## _split_main/_split_track) whose own [offset, offset+span) window covers
## master time `t`, or -1 if none. A track pointing at a DIFFERENT file is
## always just a PiP inset (see _sync_tracks) and is never a continuation.
##
## Each continuation track has its OWN independent player (mask_editor.gd's
## _track_runtime) that mask_editor.gd seeks/renders full-screen while it
## owns time `t` - unlike an earlier version of this check, there is no
## `offset == clip_in` invariant to protect here, because nothing is being
## borrowed from the main clip's own decode anymore. That invariant used to
## gate main_visible_at() so the main clip's raw decode (which never re-seeks
## for a continuation track, it just keeps decoding forward) wouldn't be
## mistaken for a track's actual content once an ordinary edit - retrimming
## the track's own in-point, or just shifting its body on the timeline (which
## only moves `offset`, see track_lane.gd's body-drag) - pulled offset and
## clip_in apart. That made picture visibility depend on a floating-point
## equality an ordinary drag could silently break, with audio (which never
## had that restriction) still playing right through the resulting black frame
## (feedback/0009, /0012, /0013, /0014 all trace back to this one invariant).
## The picture now always comes from this track's own player, seeked to its
## own `t - offset + clip_in` like every other track, so there's nothing left
## to protect - the invariant is simply gone.
func continuation_track_at(t: float) -> int:
	for i in tracks.size():
		var tr: Dictionary = tracks[i]
		if tr.get("video_path", "") != video_path:
			continue
		var offset := float(tr.get("offset", 0.0))
		var span: float = maxf(0.0, float(tr.get("clip_out", 0.0)) - float(tr.get("clip_in", 0.0)))
		if span > 0.0 and t >= offset and t < offset + span:
			return i
	return -1


## Whether SOME source (the main clip's own kept range, or a continuation
## track picking up right after it) has valid picture at time `t` at all.
func main_visible_at(t: float) -> bool:
	return t < effective_clip_out() or continuation_track_at(t) != -1


## Whether a continuation track (same video_path as the main clip) already
## claims audio ownership at time `t`, so mask_editor.gd's _apply_main_fade
## knows to mute the main clip's own audio in favor of that track's own
## independent AudioStreamPlayer instead of doubling it (feedback/0013).
##
## Only true when that track is still a GENUINE continuation - `offset` equal
## to its own `clip_in`, meaning its source position at every `t` (t - offset
## + clip_in) works out to `t` itself, exactly what the main clip's own decode
## would be showing/playing there too, hence actually a duplicate. Once a
## track's body has been dragged independently of its trim (offset and
## clip_in pulled apart - same drift continuation_track_at's doc describes for
## picture), it's playing a genuinely different moment of the source, not a
## copy of main's audio - main should keep playing so the two mix, same as
## any other pair of overlapping tracks (feedback/0021: a dragged-apart
## continuation track overlapping the still-active main clip went completely
## silent because this used to mute main for ANY covering continuation track,
## in-sync or not).
func track_owns_audio_at(t: float) -> bool:
	var i := continuation_track_at(t)
	if i == -1:
		return false
	var tr: Dictionary = tracks[i]
	return absf(float(tr.get("offset", 0.0)) - float(tr.get("clip_in", 0.0))) < 0.05


## The ruler's total extent in seconds: always at least `duration` (the primary
## clip's own full, UNTRIMMED length - trimming never shrinks this), extended to
## cover any track that runs past it, plus a fixed overflow margin so the visible
## timeline always shows some empty space past the last real content - "the max
## timeline overflows" rather than fitting content edge-to-edge, so a trim/shift
## handle dragged near the current boundary never approaches a hard edge that
## would force a rescale mid-drag.
func timeline_extent() -> float:
	var extent := duration
	for t in tracks:
		var span: float = maxf(0.0, float(t.get("clip_out", 0.0)) - float(t.get("clip_in", 0.0)))
		extent = maxf(extent, float(t.get("offset", 0.0)) + span)
	return extent + maxf(5.0, extent * 0.12)


## A new marker at `t`, seeded from the previous marker's stored values (or
## DEFAULTS before the first) - so a fresh marker CONTINUES the same layer: the
## continuation rule in layers_at dissolves the predecessor through this
## marker's envelope (identical params = visually seamless; edited params = the
## transition animation). `kind_id` indexes MARKER_KINDS.
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
		"main_fx": 1.0 if (id == 1 or id == 4 or id == 5) else 0.0,          # masked, masked_pip, masked_pip_raw
		"inset_show": 1.0 if (id == 0 or id == 3 or id == 4 or id == 5) else 0.0,  # pip, pip_raw, masked_pip, masked_pip_raw
		"inset_fx": 1.0 if (id == 0 or id == 4) else 0.0,                   # pip, masked_pip (5 = fx main + RAW pip)
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
##   damp: rises over [time, time + duration]
## Index of the latest full-raw marker whose transition to raw has COMPLETED by
## `t` (-1 if none): at its anchor for a ramp (the ramp eases in BEFORE, arriving
## complete exactly there), anchor+duration for a damp (still fading until
## then). Cutting only after completion keeps the reset pop-free: presence is
## already zero when the history disappears.
##
## But that guarantee only holds if the screen actually GOT to raw - and a
## later marker becomes at_time()'s governing `cur` from its own arrival
## onward, regardless of whether an earlier raw marker's damp window has
## finished counting down. If something later takes the screen back to
## masked/pip before the raw damp completes, main_fx never reaches 0 (it just
## holds wherever the later marker left it - never faded at all, in the
## fast-forward-to-the-next-marker case), so a rebase firing purely off the
## raw marker's own arithmetic would erase a stack that's still fully visible:
## a hard, lead-in-free pop (the "abrupt deletion" complaint). A raw marker is
## only ever superseded - never doubled back to - so checking for one later
## marker starting before `done` is sufficient.
func _barrier_index(t: float) -> int:
	var idx := -1
	for i in markers.size():
		var m: Dictionary = markers[i]
		if int(m.get("view_mode", 2.0)) != 2:
			continue
		var done: float = float(m.time)
		if int(m.get("kind", 0.0)) == 1:
			done += maxf(0.001, float(m.get("duration", 1.0)))
		if t < done:
			continue
		var superseded := false
		for j in range(i + 1, markers.size()):
			if float(markers[j].time) < done:
				superseded = true
				break
		if not superseded:
			idx = i
	return idx


static func _envelope(m: Dictionary, t: float) -> float:
	var d: float = maxf(0.001, float(m.get("duration", 1.0)))
	var anchor: float = float(m.time)
	if int(m.get("kind", 0.0)) == 0:
		return clampf((t - (anchor - d)) / d, 0.0, 1.0)
	return clampf((t - anchor) / d, 0.0, 1.0)


## Whether `b` is close enough to `a` in SCALE and PLACEMENT to plausibly be
## the same physical instance keyframed again, rather than an unrelated new
## burst that happens to share an effect and hue. Deliberately loose (50%
## scale tolerance, a real pan radius) - it only needs to reject WHOLESALE
## reconfigurations, not fine-tune what counts as "a tweak".
static func _same_instance(a: Dictionary, b: Dictionary) -> bool:
	var scale_a := maxf(0.05, float(a.get("fx_scale", 1.0)))
	var scale_b := maxf(0.05, float(b.get("fx_scale", 1.0)))
	var scale_ratio: float = maxf(scale_a, scale_b) / minf(scale_a, scale_b)
	var pos_a := Vector2(float(a.get("fx_x", 0.0)), float(a.get("fx_y", 0.0)))
	var pos_b := Vector2(float(b.get("fx_x", 0.0)), float(b.get("fx_y", 0.0)))
	return scale_ratio <= 1.5 and pos_a.distance_to(pos_b) <= 0.15


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
	# The raw checkpoint: layers only exist back to the most recent COMPLETED
	# transition into full raw (see the class doc - the rebase).
	var start := _barrier_index(t) + 1
	# Continuation partners are MERGED into whichever chain already claimed them,
	# so a later pass of the outer loop must not also emit them as a fresh,
	# independent layer (see the continuation branch below).
	var consumed := {}
	for i in range(start, markers.size()):
		if consumed.has(i):
			continue
		var m: Dictionary = markers[i]
		var e := int(m.get("effect_a", 0))
		if e == EFFECT_RESTORE or e == EFFECT_CLEAR:
			continue   # they act on other layers; they draw nothing
		var env := _envelope(m, t)
		if env <= 0.0005:
			continue
		# `cur` is the field values actually drawn - m's own, until a
		# continuation partner takes over (see below).
		var cur: Dictionary = m
		for j in range(i + 1, markers.size()):
			var mj: Dictionary = markers[j]
			var ej := int(mj.get("effect_a", 0))
			if ej == EFFECT_CLEAR:
				env *= 1.0 - _envelope(mj, t) * clampf(float(mj.get("intensity_a", 1.0)), 0.0, 1.0)
				continue
			# CONTINUATION: a later marker with the SAME effect on the SAME
			# color is the same channel keyframed again - it takes over its
			# predecessor through its own envelope. This used to run as TWO
			# separate layers with complementary envelopes (A fades out, B
			# fades in, weights summing to one) - correct for the simple
			# replacement effects (crystal/echo/erase mix toward a target
			# once), but fire/freeze/smoke/whisp add their emissive body on
			# top of an ALREADY-suppressed color, and running that pass twice
			# at partial weight doesn't equal running it once at full weight:
			# suppression compounds sub-additively (two 44%/56% consumes eat
			# LESS than one 100% consume) while the emissive term still sums
			# to the full 100% - the footage under the glow stays too bright,
			# reading as the transition flaring UP mid-crossfade instead of
			# holding steady (the "much brighter" complaint). So this is one
			# LAYER with interpolated field values, drawn once at its natural
			# env, never two partial draws of the same field.
			# Hue alone is a weak signal: fire's natural color IS ~0.02, so any
			# two independently-authored fire bursts left at the default hue
			# match this test trivially, whether or not they're remotely the
			# same instance - and unlike the whisp case, nothing else about
			# them needs to agree (a scale of 0.34 and one of 2.1 are two
			# different-sized bursts, not one burst keyframed to a new size).
			# So also require the pattern's PLACEMENT to agree closely - scale
			# within 50% and pan within a small radius - genuine continuations
			# (seeded by add_marker's copy-the-predecessor convention, then
			# tweaked a LITTLE) satisfy this easily; two coincidentally
			# same-hue bursts placed and sized independently almost never do.
			if ej == e and hue_dist(float(cur.get("hue_a", 0.0)), float(mj.get("hue_a", 0.0))) <= 0.02 \
					and _same_instance(cur, mj):
				var f := _envelope(mj, t)
				if f <= 0.0005:
					continue   # mj hasn't started rising yet - cur stands as-is
				consumed[j] = true
				if f >= 0.9995:
					cur = mj   # fully handed off - mj IS the layer now
				else:
					var blended := {}
					for key in LAYER_FIELDS:
						blended[key] = lerpf(float(cur.get(key, DEFAULTS.get(key, 0.0))),
							float(mj.get(key, DEFAULTS.get(key, 0.0))), f)
					cur = blended
				continue
			if ej != EFFECT_RESTORE:
				continue
			var reach: float = maxf(0.02, float(mj.get("threshold", 0.24)))
			if hue_dist(float(cur.get("hue_a", 0.0)), float(mj.get("hue_a", 0.0))) <= reach:
				env *= 1.0 - _envelope(mj, t) * clampf(float(mj.get("intensity_a", 1.0)), 0.0, 1.0)
		if env <= 0.0005:
			continue
		var layer := {"env": env}
		for key in LAYER_FIELDS:
			layer[key] = cur.get(key, DEFAULTS.get(key, 0.0))
		out.append(layer)
	while out.size() > MAX_LAYERS:
		out.pop_front()
	return out


## The resolved state at time `t`: the GLOBAL keying scalars (lerped through
## transition windows), the discrete view_mode + its derived presence AMOUNTS
## (lerped), and "layers" - the layer stack from layers_at(). Everything a
## consumer needs to draw the frame; see the transition contract above.
## want_layers=false skips the O(markers^2) layers_at() at the end - the recursive
## calls below (resolving the source state a transition window blends FROM) read
## only the global scalars, amounts, and pip_track off the result, never "layers",
## so computing and discarding the layer stack at every recursion level was pure
## waste that grew with both marker count and window overlap depth. External
## callers keep the default and always get layers.
func at_time(t: float, want_layers := true) -> Dictionary:
	if markers.is_empty():
		var d0 := DEFAULTS.duplicate()
		d0.merge(mode_amounts(d0.get("view_mode", 2.0)))
		d0["layers"] = []
		return d0
	var cur = null
	var cur_i := -1
	var nxt = null
	for i in markers.size():
		var m: Dictionary = markers[i]
		if m.time <= t:
			cur = m
			cur_i = i
		elif nxt == null:
			nxt = m

	# DEFAULTS (not markers[0]) before the first marker: a marker's discrete values
	# must not leak backward across the timeline prefix before it ever arrives.
	var governing: Dictionary = cur if cur != null else DEFAULTS
	var out := {}
	out["view_mode"] = governing.get("view_mode", DEFAULTS.get("view_mode", 2.0))
	# pip_track is a discrete SOURCE selector - it doesn't blend (the inset's
	# visibility does, via the amounts below). It switches with view_mode.
	out["pip_track"] = float(governing.get("pip_track", 0.0))
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
			var src := at_time(span_start - 0.001, false)
			var f := (t - span_start) / d
			for key in GLOBAL_CONTINUOUS:
				out[key] = lerpf(float(src.get(key, DEFAULTS.get(key, 0.0))),
					float(approaching.get(key, DEFAULTS.get(key, 0.0))), f)
			out["view_mode"] = approaching.get("view_mode", DEFAULTS.get("view_mode", 2.0))
			var src_amt := _amounts_of(src)
			var dst_amt := mode_amounts(out["view_mode"])
			# pip_track picks WHICH video fills the inset - a hard content swap, not a
			# blend. Popping it to the approaching marker's value the instant the window
			# opens shows the wrong source at whatever opacity the OLD side still holds
			# (fading OUT of an inset: the new, usually-irrelevant source flashes in at
			# near-full presence, then fades to nothing - the "wrong girl" bug). Whichever
			# side currently has the greater inset presence owns the content; the swap
			# only lands once the other side has faded past it, so it's pop-free.
			out["pip_track"] = float(approaching.get("pip_track", 0.0)) \
				if float(dst_amt.get("inset_show", 0.0)) >= float(src_amt.get("inset_show", 0.0)) \
				else float(src.get("pip_track", 0.0))
			for key in AMOUNT_FIELDS:
				out[key] = lerpf(float(src_amt.get(key, 0.0)), float(dst_amt.get(key, 0.0)), f)

	# Damp window: the just-passed marker's globals/presence accumulate from the
	# prior state toward its own. f clamps at 1, so it holds once accumulated.
	if cur != null and int(cur.get("kind", 0.0)) == 1 and t >= float(cur.time):
		var d: float = maxf(0.001, float(cur.get("duration", 1.0)))
		# If cur interrupted an EARLIER damp-to-raw marker's own trail (its window
		# still open when cur arrived), that suppression keeps counting down on its
		# own clock - cur doesn't start accumulating back up until it's actually
		# done. Without this, a marker placed to fade to raw over a long trail reads
		# as fully faded right up until the next marker's anchor, then instantly
		# reverses and rebuilds - the "surging back" bug (feedback/0010): the
		# overlapping trail should let the fade finish on its own promise before
		# anything new is allowed to amplify on top of it.
		var start_t := float(cur.time)
		var prev_raw_done = null   # prev's own resolved state at start_t, if gated - see below
		if cur_i > 0:
			var prev: Dictionary = markers[cur_i - 1]
			if int(prev.get("kind", 0.0)) == 1 and int(prev.get("view_mode", 2.0)) == 2:
				var prev_done: float = float(prev.time) + maxf(0.001, float(prev.get("duration", 1.0)))
				if prev_done > start_t:
					start_t = prev_done
					prev_raw_done = prev
		# While gated (t hasn't reached start_t yet), prev's OWN damp-to-raw is still
		# mid-flight - ride that curve rather than snapping straight to its endpoint.
		# cur is by construction prev's immediate successor arriving before prev's own
		# trail finishes, so prev never becomes the real barrier (see _barrier_index's
		# supersession check) - treating it as "already reached raw" the instant cur
		# arrives popped presence straight to zero (a hard cut to raw) instead of
		# continuing prev's fade to its natural endpoint, THEN handing off to cur - the
		# "particular effect reverts the whole scene to raw" bug (feedback/0029).
		if prev_raw_done != null and t < start_t:
			var pd: float = maxf(0.001, float(prev_raw_done.get("duration", 1.0)))
			var pf := clampf((t - float(prev_raw_done.time)) / pd, 0.0, 1.0)
			var psrc := at_time(float(prev_raw_done.time) - 0.001, false)
			for key in GLOBAL_CONTINUOUS:
				out[key] = lerpf(float(psrc.get(key, DEFAULTS.get(key, 0.0))),
					float(prev_raw_done.get(key, DEFAULTS.get(key, 0.0))), pf)
			var psrc_amt := _amounts_of(psrc)
			var pdst_amt := mode_amounts(prev_raw_done.get("view_mode", 2.0))
			out["pip_track"] = float(prev_raw_done.get("pip_track", 0.0)) \
				if float(pdst_amt.get("inset_show", 0.0)) >= float(psrc_amt.get("inset_show", 0.0)) \
				else float(psrc.get("pip_track", 0.0))
			for key in AMOUNT_FIELDS:
				out[key] = lerpf(float(psrc_amt.get(key, 0.0)), float(pdst_amt.get(key, 0.0)), pf)
			if want_layers:
				out["layers"] = layers_at(t)
			return out
		# `src` is the state the moment cur is actually allowed to start accumulating.
		# When gated, that's exactly prev's own fully-arrived values (querying via
		# at_time(start_t - 0.001) would re-enter THIS SAME window, since start_t is
		# still >= cur.time - infinite recursion) - prev has already reached them by
		# construction (the branch above handles every t before that), so no further
		# transitioning is needed here.
		var src: Dictionary
		if prev_raw_done != null:
			src = prev_raw_done.duplicate()
			src.merge(mode_amounts(prev_raw_done.get("view_mode", 2.0)), true)
		else:
			src = at_time(start_t - 0.001, false)
		var f := clampf((t - start_t) / d, 0.0, 1.0)
		for key in GLOBAL_CONTINUOUS:
			out[key] = lerpf(float(src.get(key, DEFAULTS.get(key, 0.0))),
				float(cur.get(key, DEFAULTS.get(key, 0.0))), f)
		var src_amt := _amounts_of(src)
		var dst_amt := mode_amounts(cur.get("view_mode", 2.0))
		# Same pip_track pop-guard as the ramp window above: the initial pass at the
		# top of this function already set out["pip_track"] to cur's (the just-arrived
		# damp marker's) own value, which is wrong while its inset presence hasn't
		# accumulated past whatever was showing before it.
		out["pip_track"] = float(cur.get("pip_track", 0.0)) \
			if float(dst_amt.get("inset_show", 0.0)) >= float(src_amt.get("inset_show", 0.0)) \
			else float(src.get("pip_track", 0.0))
		for key in AMOUNT_FIELDS:
			out[key] = lerpf(float(src_amt.get(key, 0.0)), float(dst_amt.get(key, 0.0)), f)

	if want_layers:
		out["layers"] = layers_at(t)
	return out


## This marker's own active span - [time-duration, time] for a ramp (it's pulling
## the timeline toward itself from the past), [time, time+duration] for a damp
## (it's pushing away from itself into the future). Exposed so the timeline can draw
## the span it's shading without re-deriving the ramp/damp math itself.
func marker_span(m: Dictionary) -> Vector2:
	var d: float = maxf(0.001, float(m.get("duration", 1.0)))
	var anchor: float = float(m.time)
	var is_damp: bool = int(m.get("kind", 0.0)) == 1
	return Vector2(anchor, anchor + d) if is_damp else Vector2(anchor - d, anchor)


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
		"duration": duration, "clip_in": clip_in, "clip_out": clip_out, "tracks": tracks,
		"main_volume": main_volume, "main_fade_in": main_fade_in, "main_fade_out": main_fade_out,
		"playhead": playhead, "timeline_zoom": timeline_zoom, "timeline_view_start": timeline_view_start,
		"vector_fields": VECTOR_FIELDS, "markers": markers,
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
	s.clip_in = float(parsed.get("clip_in", 0.0))
	s.clip_out = float(parsed.get("clip_out", -1.0))
	s.main_volume = float(parsed.get("main_volume", 1.0))
	s.main_fade_in = float(parsed.get("main_fade_in", 0.0))
	s.main_fade_out = float(parsed.get("main_fade_out", 0.0))
	s.playhead = float(parsed.get("playhead", 0.0))
	s.timeline_zoom = float(parsed.get("timeline_zoom", 1.0))
	s.timeline_view_start = float(parsed.get("timeline_view_start", 0.0))
	for t in parsed.get("tracks", []):
		s.tracks.append(t)
	for m in parsed.get("markers", []):
		s.markers.append(m)
	s.markers.sort_custom(func(a, b): return a.time < b.time)
	return s
