extends Node

signal scene_cut                      # a new scene just took the stage (see main's governor)

## Director - the scene registry, scheduler, and transition engine (autoload).
##
## Holds every visualizer paired with a motion behavior, and changes scenes *with
## the music*. Two typed axes govern a change:
##
##   lifecycle - a scene either loops until it is cut, or is a oneshot that plays
##               one sequence and reports finished() (e.g. glass shatters, settles,
##               ends). Loops become eligible to exit after a minimum hold.
##   trigger   - once a scene is eligible, the actual exit waits for a spectral
##               cue, chosen weighted per scene: a beat (the default - exits land
##               on the music), a movement (section change), or a lull (a drop into
##               quiet). A maximum hold is the backstop if the cue never comes.
##
## Most changes are clean jump cuts; occasionally it blends. Everything is seeded
## from the song hash, so a given track always yields the same scenes, behaviors,
## triggers, and cut/blend choices.

# Each entry pairs a scene script with a motion behavior (see GhostScene). The
# same scene appears more than once with different behaviors - that is how the
# original, un-modulated "static" looks are kept as first-class options.
const SCENES := [
	{"script": preload("res://scripts/scenes/spectrum_ring.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/spectrum_ring.gd"), "behavior": "fluid"},
	{"script": preload("res://scripts/scenes/harmonic_lattice.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/harmonic_lattice.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/rooted_growth.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/filaments.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/filaments.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/furry.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/furry.gd"), "behavior": "fluid"},
	{"script": preload("res://scripts/scenes/fog_lights.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/strata.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/bloom.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/bloom.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/wire_solid.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/planes.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/planes.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/voxel_blocks.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/cityscape.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/shatter_glass.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/gaussian_landscape.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/rocks.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/embers.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/metropolis.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/clockwork.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/clockwork.gd"), "behavior": "drift"},
	# Weather & atmosphere - composed from the shared Layer registry (see scripts/layer.gd).
	{"script": preload("res://scripts/scenes/snowfall.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/snowflakes.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/rainfall.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/clouds.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/clouds.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/fog_volume.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/fire.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/underwater.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/fireflies.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/starfield.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/starfield.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/aurora.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/fog_bank.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/petals.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/bubbles.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/motes.gd"), "behavior": "drift"},
	# Vapors - the FIELD half of the atmosphere: a fragment program, not puffs, which is
	# what lets it have a hard front and fibre (see scenes/vapors.gd). Registered on both
	# behaviors: the drifting camera suits the nebula characters and a held one suits the
	# denser ink, where the twisting is already all the movement the frame needs.
	{"script": preload("res://scripts/scenes/vapors.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/vapors.gd"), "behavior": "static"},
	# Worlds & projections - real 3D terrain, cities on it, latent geometry.
	{"script": preload("res://scripts/scenes/projection.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/projection.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/terrain.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/terrain.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/terrain_city.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/spires.gd"), "behavior": "drift"},
	# Simulation, structure and graphic work - the batch that widened the catalogue past
	# "particles over a colour bed" and "a body in a void", which between them were most
	# of what came before. Each of these is a language nothing else here speaks: a running
	# physical state, a real symmetry group, a written script, a routed graph, a plant in
	# depth, standing waves.
	{"script": preload("res://scripts/scenes/chladni.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/chladni.gd"), "behavior": "drift"},
	# wallpaper is DEREGISTERED, not deleted. Judged in the field: "everything about it
	# looked bad - the colours, the way the patterns were exactly the same everywhere, and
	# no movement at all besides a camera that shifted every once in a while."
	#
	# That is not a bug report, it is the design being wrong, and the design is mine. The
	# scene's stated bet was that a perfectly synchronized RE-INK of a static tiling would
	# be dramatic where motion would be cheap. It is not: an unchanging motif repeated to
	# the frame edge reads as a background texture, and a lattice glide on the beat reads
	# as the camera slipping rather than as the pattern stepping. The palette compounded it
	# - a black keyline against saturated inks on paper has to be composed, and these are
	# sampled.
	#
	# The file and [WallpaperGroup] stay: the seventeen plane groups are correct, tested
	# and reusable, and a motif worth repeating would still want them. What is missing is a
	# reason for the eye to stay, and that is a redesign rather than a tuning pass - one
	# that should be done where it can actually be looked at, which is what --scene is for.
	{"script": preload("res://scripts/scenes/glyphs.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/falling_sand.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/falling_sand.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/neural_field.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/canopy.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/cloth.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/murmuration.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/tidepool.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/tidepool.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/contour_map.gd"), "behavior": "drift"},
	# Two things the catalogue had no version of at all. `fractal_zoom` is the only scene whose
	# subject has detail at every scale, so it is the only one that can move FORWARD forever
	# without repeating - and the only one where the camera's motion is the whole content.
	# `tunnel_run` is the only RIDE: everything else here is looked at, and a first-person track
	# puts the viewer inside the frame instead. Both are registered on `drift` (their own camera
	# does the work; the 2D view adds almost nothing) and the fractal also on `static`, where a
	# completely fixed frame suits a fall that is already all the motion there is.
	{"script": preload("res://scripts/scenes/fractal_zoom.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/fractal_zoom.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/tunnel_run.gd"), "behavior": "drift"},
	# "the-point" scenes (camera holds, per the brief).
	{"script": preload("res://scripts/scenes/eye.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/two_eyes.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/prism.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/prism_split.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/eye_prism.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/two_prisms.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/prism_swarm.gd"), "behavior": "drift"},
]

# ----------------------------------------------------------------------------------
# The launch-lottery constant. The auto show is spectrally deterministic - a given song
# always plays the same sequence (see _resolve_seed). This salt is mixed into that seed,
# so changing it RE-ROLLS the entire show for a fixed song without touching the audio.
# Tune it: edit the value, relaunch on the launch track, watch the auto show; repeat
# until it looks best, then ship that value. Starts at the digits of Pi.
const SEED_SALT := 3141592653589793
# ----------------------------------------------------------------------------------

# The LAYER transition phases two scenes ASYNCHRONOUSLY: the incoming fades IN over the
# outgoing (both visible, layered), then the outgoing fades OUT, leaving the incoming behind -
# a slow dissolve-through-an-overlap rather than a dip to black. To keep the overlap tasteful it
# is only used when the INCOMING scene is one of these ATMOSPHERIC field scenes (they read well
# washed over another look). During the overlap the two are pushed to opposite regions (one left
# /smaller, one right/larger) so they compose instead of colliding at the focal point.
const ATMOSPHERIC := [
	"res://scripts/scenes/starfield.gd", "res://scripts/scenes/aurora.gd",
	"res://scripts/scenes/fog_bank.gd", "res://scripts/scenes/fog_lights.gd",
	"res://scripts/scenes/fireflies.gd", "res://scripts/scenes/snowfall.gd",
	"res://scripts/scenes/rainfall.gd", "res://scripts/scenes/motes.gd",
	"res://scripts/scenes/bubbles.gd", "res://scripts/scenes/petals.gd",
	"res://scripts/scenes/embers.gd", "res://scripts/scenes/clouds.gd",
	"res://scripts/scenes/underwater.gd", "res://scripts/scenes/fire.gd",
	"res://scripts/scenes/vapors.gd",
	"res://scripts/scenes/clouds.gd", "res://scripts/scenes/fog_volume.gd",
	# The soft-edged half of the new bright scenes. A LAYER dissolve suits them for the
	# same reason it suits the weather - they are full-frame fields with no silhouette to
	# collide with, and arriving through a wash rather than out of a beat of black is
	# kinder to a high-key image.
	#
	# The HARD-edged new ones are deliberately NOT here. A printed survey sheet or a flat
	# wallpaper pattern half-dissolved over another scene reads as a rendering fault
	# rather than as a transition; those want the clean dip, and a bright picture fading
	# up from black is a perfectly good entrance - it is a cut onto white that would jar.
	"res://scripts/scenes/tidepool.gd", "res://scripts/scenes/murmuration.gd",
]

enum Style { CUT, DIP, FADE, LAYER }
enum Trigger { BEAT, MOVEMENT, LULL }

# How a change is performed, weighted. Mostly a DIP to black - the old scene fades
# out, a beat of true darkness, then the new one fades up - so the eye gets a clean
# gap between scenes and two scenes never overlap (the old crossfades read as
# clipping). The occasional hard CUT keeps things punchy; a plain crossfade is rare.
const STYLE_BAG := [Style.DIP, Style.DIP, Style.DIP, Style.DIP, Style.CUT, Style.FADE, Style.LAYER]
# Exit cues, weighted: usually land the cut on a beat; sometimes on a section
# change (movement) or a drop into quiet (lull).
const TRIGGER_BAG := [Trigger.BEAT, Trigger.BEAT, Trigger.BEAT, Trigger.MOVEMENT, Trigger.MOVEMENT, Trigger.LULL]

## Don't exit a looping scene before this (seconds) - keeps cuts from thrashing. Both this and
## max_hold are DRIVE-SCALED at use (see _pacing_scale): a loud/fast/energetic passage shortens the
## holds so scenes cut faster, a calm one lengthens them. These are the calm-reference values.
@export var min_hold: float = 7.0
## Exit at least this often even if the chosen cue never arrives (seconds). Drive-scaled (see min_hold).
@export var max_hold: float = 28.0

## Scene-pacing knobs (see _pacing_scale). The hold multiplier runs from pace_drive_scale (music at
## full drive - loud/fast - so scenes cut fast) up to pace_calm_scale (quiet - scenes linger).
## pace_energy_gain sets how strongly loudness alone shortens holds. Turn pace_drive_scale DOWN (or
## pace_energy_gain UP) for punchier, faster-cutting shows.
@export var pace_drive_scale: float = 0.35
@export var pace_calm_scale: float = 1.2
@export var pace_energy_gain: float = 3.4
## Hold the final scene through the song's closing stretch: once the playback is within
## this many seconds of the end (the audio fading out), stop changing scenes and let the
## current one ride to the finish - a late cut into a near-empty tail reads as a glitch.
@export var end_hold: float = 10.0
## Movement score (0..1) that satisfies a MOVEMENT trigger.
@export var movement_threshold: float = 0.6
## Energy (0..1) at or below which a LULL trigger fires.
@export var lull_threshold: float = 0.12
## Below this smoothed audio level the track is treated as SILENT and scenes never change -
## forcing a cut with nothing playing (e.g. a song's silent tail) reads as broken. Kept well
## under lull_threshold so a musical lull still cuts but true silence holds the scene.
@export var silence_floor: float = 0.03
## Narrative TEMPO. Higher = faster: scene holds (and the auto-mode pacing bounds) shrink by this,
## so more scenes play in the same time and each scene marches through its keyframe phases sooner.
## It compresses ONLY the narrative/keyframe clock - the ambient animation of the bodies is untouched
## (see GhostScene.event_scale), so a busy session never makes the individual motions look sped-up.
## A storyboard overrides this with a top-level `sensitivity` (and per-entry `sensitivity`).
@export var sensitivity: float = 1.0
## Scene-transition PACING: a plain MULTIPLIER on how long a scene holds before it cuts. 1.0 is the
## tuned baseline, >1 lets scenes linger (fewer cuts pulling attention off the listening), <1 cuts
## faster. It multiplies the auto-mode hold bounds - and a burst's quick-cut bounds - so the
## music-driven variance is kept EXACTLY: both the minimum and the maximum move together, their
## ratio never changes, and a driving passage still cuts faster than a calm one. Deliberately a
## multiplier and never a floor/offset/clamp, which would flatten the calm end of the distribution
## onto one length and kill that variance.
## Distinct from `sensitivity`, which is narrative TEMPO and also compresses each scene's internal
## keyframe clock: this knob only changes how long a scene stays on screen, never how it animates,
## and never how long a dissolve takes (transition_time / layer_time). Authored storyboard timings
## (`hold:` / `min_hold:` / `max_hold:`) stay literal - an author who wrote `hold: 16` means 16.
## Set it through [method set_pacing]; persisted by [Settings] as `[director] pacing`.
var pacing: float = 1.0
## Bounds applied on every write to `pacing` (a quarter-speed to a quadruple-length show).
## How readily the show throws a FLOURISH - a burst of quick cuts, or a run of beat-synced
## punches. 1.0 is the tuned default, 0.0 turns them off entirely, and the top of the range makes
## them a regular feature. Separate from `pacing` on purpose: pacing is how long an ordinary scene
## holds, this is how often the show breaks that rhythm, and the reported problem was that raising
## one while the other ran hot cancelled it out.
var flourish: float = 1.0
## HOW SEVERE THE CAMERA IS, for a vehicle that flies one (see [ComicVehicle]). 0 is a slow,
## gentle drift that barely turns; 1 is the tuned default; the top of the range is fast,
## restless and cinematic, with real jump cuts in it.
##
## ONE KNOB OVER A WHOLE BEHAVIOUR, not a new setting per symptom. It scales the angular
## excursion a shot may take, how quickly that excursion is spent, how long a move lasts, how
## much of the bag is discontinuous, how deep a push goes and how fast the sheet drifts - so
## it moves the whole camera along one axis from "slow" to "chaotic" instead of asking anyone
## to balance six numbers. The full-frame vehicle has no camera and ignores it.
##
## Set it through [method set_camera]; persisted by [Settings] as `[director] camera`.
var camera: float = 1.0
## THE VEHICLE - what the show is carried on, a key from [constant Vehicle.REGISTRY]
## (`full` = the original full-frame show, `comic` = a comic page). See [Vehicle].
##
## Held HERE rather than in main, for the same reason `pacing` is: it is a property of
## the show, it must persist between sessions, and the export render is a separate
## process that reads `user://ghost.cfg` on boot - so a setting that lives in this file
## is inherited by a render with no flag to pass and nothing to keep in sync.
## `--vehicle NAME` overrides it for one run (tests, and a render of a session that was
## deliberately not the remembered setting).
var vehicle := "full"
const FLOURISH_MIN := 0.0
const FLOURISH_MAX := 4.0
## Bounds on the camera severity. 0 really is "as gentle as it goes" - a camera that drifts
## and never cuts - and 2 is the far end rather than a doubling of anything in particular.
const CAMERA_MIN := 0.0
const CAMERA_MAX := 2.0
const PACING_MIN := 0.25
const PACING_MAX := 4.0
## Seconds a dip/blend takes end to end (cuts are instant). A DIP spends the middle
## of this in darkness, so a little long reads as a deliberate breath between scenes.
@export var transition_time: float = 2.0
## A LAYER transition is slower than a dip - the two scenes overlap and shift for a while before
## the first leaves - so it gets its own, longer duration.
@export var layer_time: float = 6.0

var _host: Node = null
## THE VEHICLE - what the show is carried on (see [Vehicle]). Set by [method attach];
## null is treated exactly as `full`, so nothing here needs a vehicle to exist.
##
## The Director must never learn what a comic page is. Everything a vehicle changes it
## reaches through four small vetoes - where a scene is ADDED (_scene_host), the
## transition STYLE (_choose_style), whether the outgoing scene may FADE (_tick_schedule),
## and who applies the BOOKEND - and nothing else in this file knows the difference.
var _vehicle: Vehicle = null
## Extra entropy for ONE scene pick, so a caller can mint several scenes without the
## Director's clock having advanced between them - which is what a comic page does when it
## casts all of its panels at once (see [method mint_scene]).
##
## Folded in by XOR everywhere it is used, and 0 for the Director's own path, so at 0 every
## expression is bit-identical to what it was before this existed. That is deliberate and it
## is checked: the show is deterministic per song, and a new feature must not silently
## re-roll every existing one. tests/scene_mix_check.gd replays a seed and compares.
var _pick_salt := 0
var _prev_time := -1.0       # last frame's MUSIC-CLOCK position (Spectrum.current.time); the per-frame
                             # time step is derived from this, NOT the drawn-frame delta, so scenes and
                             # the cut schedule stay locked to the song even when a heavy scene drops FPS
var _current: GhostScene = null
var _next: GhostScene = null
var _index := -1
var _elapsed := 0.0

var _transitioning := false
var _trans_t := 0.0
var _style: Style = Style.CUT
var _trigger: Trigger = Trigger.BEAT
var _beat_prev := 0.0
var _audio_ema := 0.0        # smoothed audio level (fast attack, slow release) for the silence guard
# The material's OWN normal level: a slow running mean of _audio_ema (~LEVEL_REF_TAU seconds of
# music time, about one section). Absolute loudness is a property of the CONTENT, not of the moment -
# measured narration lives around 0.09-0.14 where a mastered track sits near 0.40 - so every "is the
# music leaning in?" test measures the fast level against THIS, never against a constant (see _lean).
# BIAS-CORRECTED (the Adam trick: carry the EMA of a constant 1 and divide by it), so at ten seconds
# in it is the honest mean of ten seconds rather than a plain EMA still climbing out of its zero
# start. Without the correction the whole opening reads as one enormous surge - the reference is far
# below the level purely because it began at zero - and a session reliably flurried in its first
# minute, which is the one place a show should be settling in.
var _audio_ref := 0.0
var _ref_acc := 0.0
var _ref_w := 0.0
const LEVEL_REF_TAU := 30.0
var _bookend_time := 6.0     # seconds of the start fade-up-from-black and end fade-down-to-black
# Rapid-fire BURST: a sparse, harmonic-gated flurry of quick jump cuts (a cinematic "3 quick
# scenes" effect) breaking up the slow holds. While a burst is live the holds shrink to a few
# seconds and every exit is a hard CUT landing on the beat.
var _burst_left := 0         # quick scenes remaining in the burst (0 = normal pacing)
var _burst_min := 1.5        # this burst's minimum hold (s)
var _burst_max := 4.0        # this burst's maximum hold (s)
# Scenes until the next flurry of each kind may start - what keeps them rare. ONE counter used to
# serve both, and that quietly starved the burst: the stinger rolls on every BEAT (a few times a
# second) while the burst rolls once per CUT (a few times a minute), so whenever the shared counter
# reached zero the stinger had two orders of magnitude more chances to spend it first, and the burst
# essentially never got the budget. Two counters, plus the mutual exclusion below (neither kind may
# begin while the other is live), keep the "never two flurries at once" property that sharing was
# really there for, without one kind eating the other's turn.
var _burst_cd := 0
var _flurry_cd := 0
var _cue_prev := 0.0    # _elapsed at the last cue offer (see _cue_taken)
# Chance of a burst at an eligible cut: BURST_BASE on an ordinary moment, BURST_BASE + BURST_GAIN
# when the material is leaning in hard, the whole thing scaled by `flourish` (the user's slider).
# A burst gets ONE roll per CUT - a slow clock, a handful of rolls a minute - and then owes a long
# cooldown, so these are per-scene odds, not the per-second kind.
#
# These are the ORIGINAL 1.2%/5%, restored. They were briefly raised to 5%/15% on the argument that
# the old numbers had never been exercised (they sat behind a level gate no spoken session could
# open - see _lean). That reasoning was sound and the conclusion was wrong: once _lean made the
# gate reachable, the SAME odds that had been inert became live, and quadrupling them on top put a
# flurry almost every subtitle - measured 29 cuts in 184 s against a 25-cut baseline. Fixing the
# gate was the whole fix; the odds needed nothing.
const BURST_BASE := 0.012
const BURST_GAIN := 0.05
const FLURRY_CD_MIN := 6     # never fewer scenes between flurries, whatever the pacing (see _flurry_spacing)
# Rapid-fire STINGER: instead of cutting through different scenes (jarring at speed), a run of
# beat-synced PUNCHES that contort / recolour / zoom the CURRENT scene - BANG, BANG, BANG - then
# settle. A universal modulation: it rides the SceneView pulse + node tint, so it works on any scene.
#
# Zoom kick ranges, sampled per punch (never baked to one value). Asymmetric on
# purpose - see the sampler in _drive_stinger and the gate in
# tests/sting_shape_check.gd, which checks the outward one against the margin
# below rather than against a remembered number.
# ...AND THE SIZE OF ONE IS ANCHORED TO THE CAMERA'S ORDINARY TRAVEL, below, rather than
# picked. The first version of this gesture was reported as "the entire scene explodes
# quickly, then recovers immediately" and the fix was its ENVELOPE - it had been a one-frame
# step, and tests/sting_shape_check.gd has gated the attack ever since. The complaint came
# back anyway, in the same words, about a punch that is now perfectly smooth: "they were
# exploding/contracting quickly, in response to harmonics... nobody expects a scene to
# explode and quickly contract." Smooth was necessary and was never sufficient. A 20% zoom
# is a fifth of the frame, arriving three times in a row, and no envelope makes that calm.
#
# DRIFT_PULL is what the camera moves by on its own, all the time, and a punch is meant to
# be an ACCENT over that rather than a different order of thing - so the inward kick tops
# out at one and a half times it. That ratio is the number worth arguing about; the rest
# follow from it.
const STING_PUSH := Vector2(0.03, 0.12)   # inward: crops, can expose nothing
const STING_PULL := Vector2(0.015, 0.04)  # outward: walks the painted edge into shot
## What a punch is worth when the session is a READING rather than a song, applied to the
## camera half of it and to the length of a run.
##
## A stinger is a music-video gesture - BANG, BANG, BANG - and ghost is also the thing a
## chapter is read over, where the scene is a background and the words are the content. The
## trigger deliberately fires in spoken sessions (it used to be gated on an absolute music
## level and never fired there at all, see `_lean`), and that is right: a reading may still
## be punctuated. It may not be interrupted. At this fraction the accent is a lean of a few
## percent - present, and nothing anyone would look up at.
const STING_READING := 0.45
## The overdraw margin layer-composed scenes are painted to - the literal 1.15 in
## [method GhostScene.update_layers], repeated here because the punch has to stay
## inside it and nothing else connects the two numbers.
const LAYER_OVERDRAW := 1.15
## The largest zoom-out the camera drift alone can ask for ([method GhostScene.drift_view]:
## `zoom` gain 0.08 at a view gain of 1, modulator in -1..1).
const DRIFT_PULL := 0.08
var _sting_left := 0         # beat-synced punches remaining in the run
var _sting_t := -1.0         # seconds into the live punch, < 0 when none is live
var _sting_span := 0.28      # that punch's whole envelope, sampled from the pulse period
var _sting_zoom := 0.0       # this punch's sampled kicks
var _sting_rot := 0.0
var _sting_skew := 0.0
var _sting_flash := 0.0
var _swaps := 0
var _rng := RandomNumberGenerator.new()
var _locked := -1            # >=0 pins one scene (authoring), set via --scene N
var _held := false           # the feedback console freezes cuts while open
var _kind_last := {}         # scene script path -> _swaps value when last shown
var _session_seed := 0       # base seed for this session (random per play; --seed pins)

# Manual mode: a storyboard is an ordered, user-authored sequence of scenes (see
# storyboards/README.md). When _storyboard_seq is non-empty the Director walks it in
# order instead of the novelty scheduler, and each entry can dictate its own exit.
var _storyboard_seq: Array = []
var _storyboard_tail: Array = []     # entries cycled after a non-looping sequence ends (never freeze)
var _storyboard_name := ""           # DISPLAY name (the JSON "name" field) - for UI only
var _storyboard_source := ""         # the loadable name/path passed to load_storyboard - for re-loading (export)
var _storyboard_loop := true
var _storyboard_transition := ""    # default transition style for a storyboard ("" = cut in manual mode)
var _storyboard_sensitivity := -1.0 # storyboard-wide tempo override (<0 = fall back to the export)
var _cur_sens := 1.0                 # the ACTIVE scene's resolved sensitivity (used by the pacing bounds)
var _step := 0

# Content re-localization (manual mode). [Echo] maps what the song SOUNDS like at each
# schedule position and matches the live harmonics against that map, so the cursor
# answers to the music itself, never the playhead: when the audio sustains a match
# somewhere the cursor is not (the song looped, a doubled track re-entered its opening,
# a finished board sat frozen), the cursor corrects there and the show re-converges.
var _echo: Echo = null
var _sched_starts: Array = []        # schedule start time (s) of each sequence entry
var _sched_end := 0.0                # schedule time where the sequence ends and the tail begins
var _manual_i := -1                  # index of the on-screen SEQUENCE entry (-1 = tail / auto)
var _heard_t := 0.0                  # monotonic listening clock (never rewinds; lags vote against it)
var _cursor_t := 0.0                 # the cursor's continuous schedule-time claim - snapped to an
                                     # entry's start when one begins, free-running through the tail,
                                     # so the echo map covers the WHOLE first hearing (outro included)

# Live performance controls (see [Dial]): created per session, seeded from the session
# seed so a dial's transformation vocabulary belongs to the song. Scenes read them
# through dial_value(); deposits persist for the whole session (across song loops).
var dials: Array = []
var _dial_demo := false              # --dial-demo: scripted turning, for headless renders/demos

# --- picture settings ---------------------------------------------------------
# [Settings] owns the file, the debounce and the flushing (see settings.gd). These setters
# write straight through it; there is deliberately no dirty flag or save timer here any
# more, because a second one is how the picture knobs and the panel they are drawn in ended
# up saving by two different mechanisms.

## THE BOOKEND, in seconds - held silence before the first sound and after the last.
## Applied by [method main._begin_session] to [member Spectrum.lead_in] / [member
## Spectrum.tail], and read back by [method _bookend_fade] so the picture arrives exactly
## as the silence ends rather than over the opening words.
##
## The defaults are not round numbers by accident. The ambience pad swells over
## [constant VoiceFX.PAD_ATTACK] = 3.5 s and releases over [constant
## VoiceFX.PAD_RELEASE] = 7 s, and its first tone is scheduled half a second in - so an
## intro much under 5 s hands the voice a bed that is still climbing, and an outro much
## under 6 s cuts the bed off mid-decay. These are the shortest values that let the
## ambience actually complete its own gesture at each end.
const INTRO_MIN := 0.0
const INTRO_MAX := 15.0
const OUTRO_MIN := 0.0
const OUTRO_MAX := 20.0
var intro_hold := 5.0
var outro_hold := 6.0


# The Director is an autoload, so this runs long before attach() builds the first scene - which is
# exactly the point: the very first hold of a session must already use the remembered pacing.
func _ready() -> void:
	_load_pacing()


## Set the scene-transition pacing multiplier (see [member pacing]) and remember it for next time.
## Safe to call on every frame of a slider drag: an unchanged value is a no-op and the disk write is
## debounced (see the top of _process).
##
## The change takes effect on the scene ALREADY on screen, not just the next one. The hold bounds are
## re-derived from `pacing` every frame in _should_change / _ready_to_exit, so nothing caches them -
## and that is the behaviour we want: someone reaching for this slider is reacting to the cutting
## they are watching right now, so dragging it up should stretch THIS scene rather than make them sit
## through one more cut to hear the difference. The one exception is a live BURST, whose bounds were
## sampled when the burst started (see _maybe_start_burst) and stay fixed for its 2-3 cuts, because a
## flurry re-scaled halfway through would come out visibly uneven. Dragging DOWNWARD can push the new
## maximum below the elapsed hold and cut immediately; that is the faster cutting being asked for,
## and it can only happen once per scene.
func set_pacing(v: float) -> void:
	var p := clampf(v, PACING_MIN, PACING_MAX)
	if is_equal_approx(p, pacing):
		return
	pacing = p
	_save_pacing()



## The intro / outro holds, in seconds.
##
## These take effect on the NEXT session rather than the running one, and that is not a
## limitation being worked around: the lead-in is a decision made when playback starts,
## and there is no coherent meaning to growing it once the audio is already speaking.
func set_intro_hold(v: float) -> void:
	var s := clampf(v, INTRO_MIN, INTRO_MAX)
	if is_equal_approx(s, intro_hold):
		return
	intro_hold = s
	_save_pacing()


func set_outro_hold(v: float) -> void:
	var s := clampf(v, OUTRO_MIN, OUTRO_MAX)
	if is_equal_approx(s, outro_hold):
		return
	outro_hold = s
	_save_pacing()


## The flourish knob.
func set_flourish(v: float) -> void:
	var f := clampf(v, FLOURISH_MIN, FLOURISH_MAX)
	if is_equal_approx(f, flourish):
		return
	flourish = f
	_save_pacing()


## The camera severity knob. Takes effect on the NEXT move the vehicle plans, which is at
## most one shot away - the same "reach for the slider and hear the difference" the pacing
## slider gives, without re-planning a move that is already travelling.
func set_camera(v: float) -> void:
	var c := clampf(v, CAMERA_MIN, CAMERA_MAX)
	if is_equal_approx(c, camera):
		return
	camera = c
	_save_pacing()


## Choose the vehicle (see [member vehicle]). Persisted like every other picture setting.
##
## Takes effect on the NEXT session, not the running one - and unlike the intro hold,
## that IS a limitation rather than a definition. Swapping presentation mid-show means
## re-hosting the live scene into a different surface while it is drawing; the honest
## version of that is a restart, so the surface says so rather than half-doing it.
func set_vehicle(key: String) -> void:
	if not Vehicle.REGISTRY.has(key) or key == vehicle:
		return
	vehicle = key
	_save_pacing()


## The vehicle this run actually uses: `--vehicle NAME` if given and known, else the
## remembered setting. Read by [main] when it builds the stage.
func resolved_vehicle() -> String:
	var args := OS.get_cmdline_user_args()
	var i := args.find("--vehicle")
	if i >= 0 and i + 1 < args.size():
		var k := String(args[i + 1])
		if Vehicle.REGISTRY.has(k):
			return k
		push_warning("ghost: --vehicle %s is not a known vehicle - using %s" % [k, vehicle])
	return vehicle


func _load_pacing() -> void:
	pacing = clampf(float(Settings.read("director", "pacing", 1.0)), PACING_MIN, PACING_MAX)
	flourish = clampf(float(Settings.read("director", "flourish", 1.0)), FLOURISH_MIN, FLOURISH_MAX)
	camera = clampf(float(Settings.read("director", "camera", 1.0)), CAMERA_MIN, CAMERA_MAX)
	intro_hold = clampf(float(Settings.read("director", "intro", intro_hold)), INTRO_MIN, INTRO_MAX)
	outro_hold = clampf(float(Settings.read("director", "outro", outro_hold)), OUTRO_MIN, OUTRO_MAX)
	var v := String(Settings.read("director", "vehicle", "full"))
	vehicle = v if Vehicle.REGISTRY.has(v) else "full"


## Hand the current values to [Settings], which owns the file and the flushing. There is no
## debounce here any more and no dirty flag: writing an unchanged value is already a no-op,
## and every remaining question about WHEN the disk is touched belongs to one place.
func _save_pacing() -> void:
	Settings.write("director", "pacing", pacing)
	Settings.write("director", "flourish", flourish)
	Settings.write("director", "camera", camera)
	Settings.write("director", "intro", intro_hold)
	Settings.write("director", "outro", outro_hold)
	Settings.write("director", "vehicle", vehicle)


func attach(host: Node, vehicle: Vehicle = null) -> void:
	_host = host
	_vehicle = vehicle
	# A fresh session hears fresh material: forget the last song's level, its reference, and any
	# flurry state. attach() runs again on every new take in synthesis mode, so without this a quiet
	# piece following a loud one would spend its opening judged against the loud one's level - and a
	# session interrupted mid-flurry would resume owing the old one's quick cuts or its cooldown.
	_audio_ema = 0.0
	_ref_acc = 0.0
	_ref_w = 0.0
	_audio_ref = 0.0
	_burst_left = 0
	_burst_cd = 0
	_sting_left = 0
	_flurry_cd = 0
	if not is_equal_approx(pacing, 1.0):
		# Worth a line: "the scenes change too fast/slow" is a common feedback report, and this says
		# straight away whether the session was running with a non-default hold multiplier.
		print("ghost: scene pacing x%.2f (holds stretched from the remembered slider)" % pacing)
	_session_seed = _resolve_seed()
	print("ghost: session seed %d (%s)" % [_session_seed, _seed_source()])
	_rng.seed = _session_seed ^ 0x1234567
	# A long-ish start/end fade, sampled per song into [3, 10] s. Derived from a HASH of the seed
	# (not a draw off _rng) so it doesn't perturb the deterministic scene sequence.
	_bookend_time = 3.0 + 7.0 * (float(hash([_session_seed, "bookend"]) & 0xFFFF) / 65535.0)
	_locked = _locked_scene_arg()
	_load_storyboard_arg()
	# A storyboard's opening is AUTHORED (the-point's eye flies in over the first
	# second); the long sampled fade-from-black would swallow it, so manual sessions
	# get a short fixed bookend and the board owns its own entrance.
	if is_manual():
		_bookend_time = minf(_bookend_time, 2.0)
	dials = [Dial.new(_session_seed ^ 0x0D1A15EE)]
	_dial_demo = OS.get_cmdline_user_args().has("--dial-demo")
	_echo = Echo.new()
	_heard_t = 0.0
	# AFTER _session_seed is resolved, BEFORE the first scene is made: the vehicle samples
	# its own look from the session seed, and the very next line asks it for a host.
	if _vehicle != null and is_instance_valid(_vehicle):
		_vehicle.begin_session()
	# A vehicle that owns its cast has just built the whole page; the show opens on
	# whichever panel it says the reading starts at, already alive in its own viewport.
	var opening := _handover(null)
	if opening != null:
		_current = opening
	else:
		_current = _make_scene()
		_scene_host(_current).add_child(_current)
	_arm()


## Ask a cast-owning vehicle for the scene to move to, and adopt whatever it hands back.
##
## Null means "not that kind of vehicle, or it declined this change" and the caller falls
## through to building a scene itself. Non-null means the scene is ALREADY built and
## parented and the outgoing one is still on the page - so there is nothing to add, nothing
## to free, and no fade to play: on a comic the change of scene IS the camera moving, and a
## crossfade between two panels that are both permanently on the paper would be nonsense.
func _handover(outgoing: GhostScene) -> GhostScene:
	if _vehicle == null or not is_instance_valid(_vehicle) or not _vehicle.owns_cast():
		return null
	var sc := _vehicle.take_over(outgoing)
	if sc == null or not is_instance_valid(sc):
		return null
	# The scene was minted with its own sensitivity already stamped on it; the hold bounds
	# read _cur_sens, so re-derive it from the scene actually taking the stage rather than
	# leaving it on whatever the last mint happened to set.
	_cur_sens = clampf(sc.event_scale, 0.05, 20.0)
	return sc


## Where an ARRIVING scene is added. Without a vehicle this is the stage, which is what
## every `_host.add_child` in this file used to say literally; with one it is whatever
## surface that vehicle opens for the incoming scene (a comic page's next panel).
##
## Called at the MOMENT OF ARRIVAL and nowhere else, which is deliberate: a vehicle that
## has to do something when the show advances (freeze the panel behind, open the next)
## hangs it off this one call rather than needing a second signal that could get out of
## step with it.
func _scene_host(incoming: GhostScene) -> Node:
	if _vehicle != null and is_instance_valid(_vehicle):
		var h := _vehicle.host_for(incoming)
		if h != null and is_instance_valid(h):
			return h
	return _host


## The seed every scene choice / shot / param roll derives from this session. Random
## per play (so each play is fresh and you see how scenes combine), unless pinned
## with `--seed N` - which the exporter passes so a render reproduces what you saw.
func session_seed() -> int:
	return _session_seed


# The session seed, by priority:
#   1. an explicit --seed N (the exporter passes it so a render reproduces a session, and
#      it is the way to roll a *different* show for the same song on purpose) - taken
#      verbatim, so the export reproduces session_seed() exactly (the salt is already in it);
#   2. otherwise the audio's own fingerprint (Spectrum.song_hash) mixed with the tunable
#      SEED_SALT - SPECTRAL DETERMINISM: the same song + same salt always yields the same
#      show; changing the salt re-rolls it (the launch-lottery knob);
#   3. random, only when no audio is loaded (idle preview).
func _resolve_seed() -> int:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if args[i] == "--seed" and i + 1 < args.size() and args[i + 1].is_valid_int():
			return int(args[i + 1])
	if Spectrum.song_hash != 0:
		return _salt_seed(Spectrum.song_hash)
	var r := RandomNumberGenerator.new()
	r.randomize()
	return r.randi()


# Mix the audio fingerprint with the launch-lottery SEED_SALT into the session seed. A
# string hash so any change in either input disperses across the whole seed (a different
# salt is a different show, not a one-bit nudge).
func _salt_seed(fingerprint: int) -> int:
	return hash("%d:%d" % [fingerprint, SEED_SALT])


# Where the session seed came from, for the log line (so determinism is observable and
# the active lottery salt is visible while tuning).
func _seed_source() -> String:
	for a in OS.get_cmdline_user_args():
		if a == "--seed":
			return "--seed override"
	if Spectrum.song_hash != 0:
		return "audio fingerprint + salt %d" % SEED_SALT
	return "random (no audio)"


## Tear down the current session: free the live scene(s) and reset all state, so a
## later attach() starts cleanly. Called when a song ends and we return to the
## splash. Does not clear a storyboard loaded for the *next* session (load it after).
func detach() -> void:
	# A cast-owning vehicle's scenes belong to its page, not to this session's current/next
	# pair - they are still parented in their own panels and its release() lets them go.
	# Freeing them here would leave the vehicle holding freed nodes.
	var borrowed := _vehicle != null and is_instance_valid(_vehicle) and _vehicle.owns_cast()
	if _transitioning and is_instance_valid(_next) and not borrowed:
		_next.queue_free()
	if is_instance_valid(_current) and not borrowed:
		_current.queue_free()
	_current = null
	_next = null
	_host = null
	# The vehicle is OWNED BY MAIN (it is mounted on main's stage and outlives a
	# take's session churn in synthesis modes); detach only lets go of the reference
	# and tells it to release whatever it was holding for THIS session.
	if _vehicle != null and is_instance_valid(_vehicle):
		_vehicle.release()
	_vehicle = null
	_transitioning = false
	_trans_t = 0.0
	_index = -1
	_swaps = 0
	_step = 0
	_elapsed = 0.0
	_cue_prev = 0.0
	_held = false
	_kind_last = {}
	_storyboard_seq = []
	_storyboard_tail = []
	_storyboard_name = ""
	_storyboard_source = ""
	dials = []
	_echo = null
	_sched_starts = []
	_sched_end = 0.0
	_manual_i = -1
	_heard_t = 0.0
	_cursor_t = 0.0


## The live performance dials' summed modulation on [param slot], in [-1, 1].
## [param i] gives element-level phase diversity (a cast modulates as a group, not in
## lockstep). Zero whenever no dial has been touched - scenes can sample it blindly.
func dial_value(slot: String, i := 0) -> float:
	if dials.is_empty():
		return 0.0
	var v := 0.0
	for d in dials:
		v += (d as Dial).value(slot, i)
	return clampf(v, -1.0, 1.0)


## The primary dial (the workspace widget drives it), or null outside a session.
func dial(index := 0) -> Dial:
	return dials[index] if index < dials.size() else null


## True when the Director is walking a user-authored storyboard (manual mode).
func is_manual() -> bool:
	return not _storyboard_seq.is_empty()


## Name of the active storyboard, or "" in auto mode.
func storyboard_name() -> String:
	return _storyboard_name


## The loadable name/path the active storyboard was loaded FROM (may differ from its display name).
## The exporter passes THIS to the render process so it re-loads the same storyboard, not the display name.
func storyboard_source() -> String:
	return _storyboard_source


## Load a storyboard by name (res://storyboards/<name>.yaml or .json) or by a full/absolute
## path, switching the Director into manual mode. Returns true on success. Safe to call
## before attach(); the splash uses this to start a manually-orchestrated session.
## Parsing / defs expansion / validation live in [Storyboard] - the Director only keeps
## the walk state.
func load_storyboard(name_or_path: String) -> bool:
	var sb := Storyboard.load_file(name_or_path)
	if not sb.ok:
		push_warning("ghost: %s" % sb.error)
		return false
	_storyboard_seq = sb.sequence
	_storyboard_tail = sb.tail
	_storyboard_loop = sb.loop
	_storyboard_name = sb.name
	_storyboard_source = name_or_path           # remember HOW it was loaded, so the export can re-load it
	_storyboard_transition = sb.transition      # e.g. "cut" forces jump cuts
	_storyboard_sensitivity = sb.sensitivity    # narrative tempo (<0 = use the export)
	_step = 0
	_sched_starts = _schedule_starts()
	print("ghost: storyboard '%s' loaded (%d scenes, loop=%s)" % [
		_storyboard_name, _storyboard_seq.size(), _storyboard_loop])
	return true


# Cumulative schedule start time (s) of each sequence entry, mirroring _make_scene's
# hold scaling (hold / sensitivity) - the coordinate system of the [Echo] map and of a
# re-localization. Cue-exit entries have no deterministic length, so they contribute a
# min-hold estimate; the map only needs entry-level granularity there. Also stamps
# `_sched_end`, where the sequence hands over to the tail.
func _schedule_starts() -> Array:
	var out: Array = []
	var acc := 0.0
	var sbs: float = _storyboard_sensitivity if _storyboard_sensitivity > 0.0 else sensitivity
	for item in _storyboard_seq:
		out.append(acc)
		var sens := clampf(float(item.get("sensitivity", sbs)), 0.05, 20.0)
		# Authored timings are literal here too. Only the UNAUTHORED fallback - an entry that gave no
		# timing at all, and so exits through _scaled_bound's auto branch at runtime - carries `pacing`,
		# so this estimate keeps mirroring what those entries actually do. Computed once at storyboard
		# load and then frozen: it is the [Echo] map's coordinate system, and re-deriving it mid-session
		# would move the schedule times the already-recorded fingerprints were filed under.
		var dur: float = 0.0
		if item.has("hold"):
			dur = float(item["hold"])
		elif item.has("min_hold"):
			dur = float(item["min_hold"])
		else:
			dur = min_hold * pacing
		acc += maxf(0.5, dur / sens)
	_sched_end = acc
	return out


# `--storyboard <name|path>` selects manual mode at launch.
func _load_storyboard_arg() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if args[i] == "--storyboard" and i + 1 < args.size():
			load_storyboard(args[i + 1])
			return


# Reset the hold clock and choose the exit cue for the scene now on screen.
# A deterministic index in [0, n) from the session _rng (itself seeded from the song fingerprint).
# This USED to XOR the live Spectrum.seed_bias() so the spectrum steered the choice, but that value
# samples the spectrum at the instant of the call and is not frame-reproducible in live playback, so
# the same song rolled a different show every run. Harmonic content already enters deterministically
# through the fingerprint-derived session seed; the pick stays song-driven AND reproducible.
func _biased(n: int) -> int:
	return absi(_rng.randi()) % maxi(1, n)


func _arm() -> void:
	_elapsed = 0.0
	_cue_prev = 0.0
	_beat_prev = Spectrum.current.beat
	if _flurry_cd > 0:
		_flurry_cd -= 1            # count down the spacing between flurries of each kind
	if _burst_cd > 0:
		_burst_cd -= 1
	if _burst_left == 0:
		_maybe_start_burst()       # rarely, kick off a rapid-fire CUT burst on this scene
	# In a burst, exits land on the beat (quick + musical); otherwise the weighted cue bag.
	_trigger = Trigger.BEAT if _burst_left > 0 else TRIGGER_BAG[_biased(TRIGGER_BAG.size())]


# How hard the music is LEANING IN right now, 0..1, measured against the material's own normal
# level rather than an absolute one - the surge, not the loudness. Returns -1 when the question is
# meaningless, which is the one place an ABSOLUTE test still belongs: this instant is silent, or the
# material's own normal level is silence. `silence_floor` is reused for both (it is already the
# "nothing is playing" line the cut logic trusts), so a dead track, a silent tail, and a session with
# no audio at all can never flurry no matter how favourable the ratio between two tiny numbers looks.
#
# This replaced a pair of hard `_audio_ema < 0.15` / `>= 0.2` gates. Those are levels a mastered
# track clears at rest (measured ~0.40) and speech mostly does not (measured ~0.09-0.14), so a spoken
# session could not fire a burst or a stinger even once - across ~50 minutes of real headless
# sessions, neither ever fired. A ratio has no genre baked into it: it fires on the passage where
# this material leans in, whatever this material happens to be.
func _lean() -> float:
	if _audio_ema < silence_floor or _audio_ref < silence_floor:
		return -1.0
	return clampf((_audio_ema - _audio_ref) / _audio_ref, 0.0, 1.0)


# The shared flurry drive: how emphatic this moment is, 0..1, on the spike curve (sharp onset, quick
# saturation). `movement` is already content-relative (short-term flux over the passage's own
# baseline); `lean` supplies the loudness half in the same relative currency.
func _flurry_drive(lean: float) -> float:
	return Nonlinear.apply("spike", clampf(Spectrum.current.movement + 0.5 * lean, 0.0, 1.0), 2.5)


# A sparse, NON-LINEAR chance to start a rapid-fire CUT burst: only in the auto show, only with
# audible material, weighted up by how much that material is moving right now (a spike curve on
# movement + lean). Rare on purpose, with a long cooldown after, so it never chains into a dozen
# cuts - and never on top of a live stinger, which would read as the show glitching rather than
# as one deliberate flourish.
func _maybe_start_burst() -> void:
	if not _storyboard_seq.is_empty() or _locked >= 0 or _burst_cd > 0 or _sting_left > 0:
		return
	var lean := _lean()
	if lean < 0.0:
		return
	if _rng.randf() < (BURST_BASE + BURST_GAIN * _flurry_drive(lean)) * flourish:
		_burst_left = _rng.randi_range(2, 3)       # short - normalizes after a couple of cuts
		# The flurry is part of the show's character, so it rides the same `pacing` multiplier as the
		# normal holds instead of staying stubbornly quick while everything around it lengthens. Scaled
		# HERE, where the pair is sampled, so one burst keeps consistent cut lengths from start to end.
		_burst_min = _rng.randf_range(1.0, 2.0) * pacing
		_burst_max = _rng.randf_range(3.0, 5.0) * pacing
		_burst_cd = _flurry_spacing()              # then a long stretch of normal pacing
		print("ghost: BURST x%d  (%.1f-%.1fs cuts)" % [_burst_left, _burst_min, _burst_max])


# Scenes to wait before the next flurry of the same kind. Counted in CUTS, not seconds, because a
# flurry is a STRUCTURAL event - "one flourish every ~N scenes" is a statement about the shape of
# the show - and a seconds-based spacing would silently make the flourish a bigger and bigger part
# of the cutting the calmer the passage.
#
# Divided by `pacing` for the one thing cuts are NOT a good clock for: the user's own tempo. At
# pacing 3.0 scenes hold three times as long, so a fixed 14-26 cuts is three times the wall-clock
# wait - the slider would quietly delete the flourish from exactly the slow, lingering show that
# most needs punctuating. Dividing keeps the spacing roughly constant in listening time while the
# count itself stays a cut count. Floored at FLURRY_CD_MIN so it can never degenerate into chaining.
func _flurry_spacing() -> int:
	return maxi(FLURRY_CD_MIN, int(round(float(_rng.randi_range(14, 26)) / pacing)))


# `--scene N` (or `--scene name`) pins a single scene for authoring - no changes.
func _locked_scene_arg() -> int:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if args[i] == "--scene" and i + 1 < args.size():
			var v := args[i + 1]
			if v.is_valid_int():
				return clampi(int(v), 0, SCENES.size() - 1)
			for s in SCENES.size():
				if String(SCENES[s].script.resource_path).contains(v):
					return s
	return -1


# The whole-show bookend fade: the picture eases UP from black over the first `_bookend_time`
# seconds of the song and back DOWN to black over the final `_bookend_time` seconds before it
# ends - reusing the very same modulate.a-to-black that the scene transitions use, so the first
# and last scenes fade like any other cut. This runs whenever a song is loaded (LIVE and export
# alike - not export-only any more), so the live animation gets the start/end fades too. Returns
# 1.0 (no fade) only when idle, where there is no defined start or end. Audio is never touched.
func _bookend_fade() -> float:
	var slen := Spectrum.song_length()
	if slen <= 0.0:
		return 1.0                                    # idle / no song: no bookend
	var t := Spectrum.current.time
	# PAST THE END IS NOT BLACK. A synthesis take reports its length once, then loops
	# inside the generator without restarting the player, so the playback position runs
	# on past it monotonically - and the old `(slen - t)/fade` then held at 0 forever,
	# leaving the stage permanently dark with no way back. An overrun means the length
	# is stale, not that the show is over.
	if t > slen + 0.5:
		return 1.0
	# The fade windows follow the BOOKEND when there is one, so the picture arrives
	# exactly as the held silence ends rather than over the opening words. Without a
	# bookend it falls back to the per-song sampled length, which is the old behaviour.
	var fade_in := maxf(0.5, Spectrum.lead_in if Spectrum.lead_in > 0.001 else _bookend_time)
	var fade_out := maxf(0.5, Spectrum.tail if Spectrum.tail > 0.001 else _bookend_time)
	var a_in := clampf(t / fade_in, 0.0, 1.0)           # 0 at the very start -> 1 after fade_in
	var a_out := clampf((slen - t) / fade_out, 0.0, 1.0)  # 1 -> 0 over the final fade_out seconds
	return minf(a_in, a_out)


# Fixed simulation timestep. The SIM is advanced in chunks of this size (decoupled from the drawn
# frame), so animations integrate stably and the cut schedule is exact regardless of render FPS.
const SIM_STEP := 1.0 / 30.0
# Cap on how many sim steps one render frame may run - a death-spiral guard. At ~0.5s of catch-up
# per frame (a renderer crawling below ~2 fps) we stop advancing and accept a small drift, rather
# than letting a heavy frame trigger an ever-growing pile of sim work.
const MAX_SIM_STEPS := 15
# Per-drawn-frame cap on how much MUSIC time the sim advances, so a single slow frame (a scene build,
# an FPS hitch) can't lurch the camera forward in one visible step. Any excess is BANKED as debt and
# paid down over the following (faster) frames, so the show still tracks the music on average - it just
# eases into place instead of snapping. STEP_CAP must exceed a normal frame's music step or a steadily
# heavy scene would stretch; DEBT bounds how far behind a sustained-slow patch may fall.
const STEP_CAP := 0.11
const DEBT_CAP := 0.4
var _time_debt := 0.0


func _process(delta: float) -> void:
	if _current == null:
		return

	# While the feedback console is open it holds the scene; freeze it entirely (no
	# update, no redraw) so a heavy scene's draw can't starve the main loop and make
	# typing in the console lag. The console dims the frozen frame anyway.
	if _held:
		return

	# Advance the performance dials (waveforms + transient decay). --dial-demo turns
	# the primary dial at a scripted, slowly-breathing rate - a hands-free tour of the
	# wedges for headless renders and demos.
	if not dials.is_empty():
		if _dial_demo:
			(dials[0] as Dial).turn(delta * (0.9 + 0.7 * sin((dials[0] as Dial).angle * 0.37)))
		for d in dials:
			(d as Dial).advance(delta)

	# The raw advance is the advance of the MUSIC CLOCK, not the drawn-frame delta. When a heavy scene
	# lags the renderer the song keeps playing, so this grows. `_prev_time < 0` marks a fresh reference
	# (very first frame, or a loop/seek).
	var raw := clampf(delta, 0.0, 0.12)
	if _prev_time >= 0.0:
		var d := Spectrum.current.time - _prev_time
		raw = d if (d >= 0.0 and d <= 2.0) else clampf(delta, 0.0, 0.1)  # d<0 loop / d>2 seek -> fallback
	_prev_time = Spectrum.current.time
	# SMOOTH the ANIMATION step: bank the raw advance as debt, spend at most STEP_CAP per drawn frame, so
	# a scene BUILD (12-step pre-warm + the first heavy draw) or an FPS hitch can't jerk the fresh camera
	# into place in one visible frame - the excess eases in over the next faster frames.
	_time_debt = minf(_time_debt + raw, DEBT_CAP)
	var anim := minf(_time_debt, STEP_CAP)
	_time_debt -= anim

	# CRITICAL split: the SCHEDULE (how long a scene holds, transition progress) runs on the RAW music
	# advance, so a scene always cuts at the right MUSIC time - a heavy scene rendering below the cap
	# does NOT overrun its hold (that made a scene last 30s+). Only the ANIMATION (camera / growth) is
	# smoothed. So duration tracks the song; the picture just eases rather than lurches under lag.
	_tick_schedule(raw)
	_tick_animation(anim)
	if _vehicle != null and is_instance_valid(_vehicle):
		# the ANIMATION step, capped - the vehicle's camera is picture, not schedule, and a
		# lag spike must ease it rather than teleport it (same reason _tick_animation exists)
		_vehicle.advance(Spectrum.current, anim, _bookend_fade())


# The SCHEDULE, advanced by the REAL music-clock step: the hold clock, transition progress + alphas,
# the smoothed audio level, the stinger, and arming the next cut. This decides WHEN things happen, so
# it must track the music exactly (never the capped animation step) or scene durations drift long.
func _tick_schedule(dt: float) -> void:
	var bookend := _bookend_fade()                  # 1, except fading from/to black at the video's ends
	# WHO APPLIES THE BOOKEND. Folding it into the scene's own alpha is right for a full
	# frame, where the scene IS the picture - and wrong for any vehicle that draws more
	# than the scene, because it would fade one comic panel and leave the paper lit. A
	# vehicle that claims it applies the same number to its own root instead; the scene
	# then just never sees it.
	var bf := 1.0 if (_vehicle != null and is_instance_valid(_vehicle) and _vehicle.owns_bookend()) else bookend
	if _transitioning:
		var dur: float = layer_time if _style == Style.LAYER else transition_time
		_trans_t += dt / maxf(0.01, dur)
		var k := clampf(_trans_t, 0.0, 1.0)
		# Alphas are sequenced so the picture is clean (a DIP never shows both scenes at once).
		var a := _transition_alphas(k)
		# HELD OUTGOING. A comic panel that is already inked on the paper cannot un-draw
		# itself, so the vehicle pins the leaving scene at full and only the ARRIVING one
		# fades - a panel developing in place. The full frame holds nothing and this is
		# the identity.
		if _vehicle != null and is_instance_valid(_vehicle) and _vehicle.hold_outgoing():
			a.x = 1.0
		_current.modulate.a = a.x * bf
		_current.view.presence = a.x
		_next.modulate.a = a.y * bf
		_next.view.presence = a.y
		if k >= 1.0:
			_finish_transition()
		return
	_elapsed += dt
	# Smoothed audio level: rises fast, falls slowly, so a momentary gap between beats doesn't
	# read as silence but a genuinely dead track (or its silent tail) does.
	var e: float = Spectrum.current.energy
	_audio_ema = lerpf(_audio_ema, e, 1.0 - exp(-(8.0 if e > _audio_ema else 0.6) * dt))
	# ... and the slow reference it is judged against (see _audio_ref). Symmetric and much slower than
	# the level itself, so it settles on what the material normally does and a surge stands out.
	var ra := 1.0 - exp(-dt / LEVEL_REF_TAU)
	_ref_acc = lerpf(_ref_acc, _audio_ema, ra)
	_ref_w = lerpf(_ref_w, 1.0, ra)
	_audio_ref = _ref_acc / maxf(_ref_w, 1e-4)
	_drive_stinger(dt, bf)                          # rapid-fire beat-synced modulation of THIS scene
	if _listen_echo(dt):
		_beat_prev = Spectrum.current.beat
		return                                      # re-localized: the cut is already underway
	_paced_t += dt
	if _should_change():
		_begin_transition()
	_beat_prev = Spectrum.current.beat


# The ANIMATION, advanced by the SMOOTHED step in fixed sub-steps (stable integration, no lurch), and
# DRAWN once (queue_redraw is idempotent per frame) - so under lag we skip intermediate renders while
# the scene(s) still tick forward.
func _tick_animation(anim: float) -> void:
	var remaining := anim
	var steps := 0
	while remaining > 1e-5 and steps < MAX_SIM_STEPS:
		var dt := minf(remaining, SIM_STEP)
		_current.update(Spectrum.current, dt)
		_current.view.commit(dt)
		if _transitioning and _next != null:
			_next.update(Spectrum.current, dt)
			_next.view.commit(dt)
		remaining -= dt
		steps += 1


# The rapid-fire stinger: on a strong beat, sparsely start a short run of beat-synced punches;
# each punch zooms / rolls / skews and brightens the CURRENT scene, decaying before the next - a
# BANG-BANG-BANG without a jarring cut. Universal (rides the view pulse + node tint).
func _drive_stinger(delta: float, bf: float) -> void:
	var f := Spectrum.current
	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	if beat_edge:
		if _sting_left > 0:                          # land the next punch on this beat
			_sting_left -= 1
			_sting_t = 0.0
			# THE PUNCH IS AS LONG AS THE MUSIC IS SLOW. A fixed 1/6 s decay is a
			# different gesture at 60 BPM than at 160: on slow material it is over
			# before the eye has followed it, which is half of why it read as a
			# glitch rather than as a hit. Bounded either side so a mis-estimated
			# period cannot leave the frame contorted or blink it.
			_sting_span = clampf(f.beat_period * 0.55, 0.28, 0.7)
			# INWARD AND OUTWARD ARE NOT SYMMETRIC. A push-in crops and can never show
			# anything that is not painted; a pull-back walks the edge of the painted
			# region into shot. Layer-composed scenes are painted to a FIXED 1.15x
			# overdraw margin (GhostScene.update_layers), sized for the camera drift's
			# +-0.08 and documented as such - so a 0.20 pull-back needed 1.25x and got
			# 1.15x, and a seam appeared and vanished with the punch. That is the other
			# half of "the whole scene explodes". Outward kicks are bounded by what the
			# margin can actually cover; inward ones keep the full range.
			_sting_zoom = _rng.randf_range(STING_PUSH.x, STING_PUSH.y)
			if _rng.randf() < 0.5:
				_sting_zoom = -_rng.randf_range(STING_PULL.x, STING_PULL.y)
			_sting_rot = _rng.randf_range(-0.055, 0.055)
			_sting_skew = _rng.randf_range(-0.04, 0.04)
			_sting_flash = _rng.randf_range(0.15, 0.35)
		elif _flurry_cd == 0 and _burst_left == 0:   # else maybe begin a run (rare, harmonic-gated)
			# Same content-relative gate as the burst (see _lean): the old absolute `_audio_ema >= 0.2`
			# is a music level, so spoken sessions never punched once either.
			var lean := _lean()
			if lean >= 0.0 and _rng.randf() < (0.02 + 0.07 * _flurry_drive(lean)) * flourish:
				# BANG, BANG (, BANG) - or half of one, over a reading.
				_sting_left = _rng.randi_range(1, 2) if _reading() else _rng.randi_range(2, 4)
				_flurry_cd = _flurry_spacing()
				print("ghost: STINGER x%d" % _sting_left)
	if _sting_t >= 0.0:
		_sting_t += delta
		if _sting_t >= _sting_span:
			_sting_t = -1.0                          # settled; back to the scene's own framing
	var p := _sting_env()
	# THE CAMERA HALF RIDES THE SCENE'S DECLARED CHARACTER, the flash does not.
	# `drift_view` already scales every per-frame camera move by `behavior.view`
	# (0 for a static scene, 0.3 for a fluid one), and the punch was the one thing
	# that moved the camera without asking - so a scene whose whole design is "hold
	# still and let the audio speak" got slammed exactly as hard as one built to
	# swoop. The floor keeps the accent present everywhere rather than deleting it
	# from half the catalogue. The brightness flash is not a camera move and stays
	# universal, which is what a static scene gets instead.
	var g: float = 0.35 + 0.65 * float(_current.behavior.get("view", 1.0))
	# ...and a reading is not a music video. See STING_READING.
	if _reading():
		g *= STING_READING
	# ...and a FIELD scene is not rolled or sheared at all. Shot selection already
	# denies fields `canted` and `pan` "so their edges never swing into view"
	# (Shots.FIELD_BAG) - and then the punch rolled them 5 degrees anyway, which is
	# the one move that rule exists to prevent. Zoom survives, because push_in and
	# pull_back ARE in that bag.
	var swing := 0.0 if _current.framing == "field" else 1.0
	_current.view.pulse_zoom = 1.0 + _sting_zoom * p * g
	_current.view.pulse_rot = _sting_rot * p * g * swing
	_current.view.pulse_skew = _sting_skew * p * g * swing
	# Brightness/tint flash via the node modulate, preserving the fade alpha. Softened over a
	# reading too, but by less: a change of light is what a static scene gets INSTEAD of a
	# camera move (see above), so taking it down as far as the camera would leave those
	# scenes with no accent at all.
	var fl := 1.0 + _sting_flash * p * (0.65 if _reading() else 1.0)
	_current.modulate = Color(fl * (1.0 + 0.06 * _sting_rot), fl, fl * (1.0 - 0.06 * _sting_rot), bf)


## Is this session a reading rather than a song?
##
## `Spectrum.is_streaming()` is true exactly when the audio is a take being synthesized into
## a stream - the Generative panel reading a chapter, or the synth editor - and false for a
## song loaded from a file. It is the only signal here that separates "the picture is the
## content" from "the picture is the background", and it is one the show should be allowed
## to know about: a flourish that suits a track is an interruption over a paragraph.
func _reading() -> bool:
	return Spectrum.is_streaming()


# The punch envelope, 0 -> 1 -> 0 across `_sting_span`. The whole point of it is
# the ATTACK.
#
# It used to be a step: `_sting = 1.0` on the beat edge, read by the same frame's
# draw, so `pulse_zoom` went 1.00 -> 1.20 between two consecutive frames and the
# roll, skew and a 1.5x brightness flash all arrived in that same frame. Nothing
# else in ghost's camera moves like that - `SceneView.commit` exists precisely so
# framing eases rather than snaps, and `snap()` is documented as a pre-warm-only
# exception - and a one-frame scale step does not read as an accent, it reads as
# the picture tearing. That is the reported "the whole scene explodes, then
# recovers immediately", and it was jarring on peaceful scenes because a
# discontinuity is jarring on anything.
#
# The envelope was always meant to BE the easing (SceneView: "not eased - the
# Director drives the envelope"); it simply never eased the onset.
#
# THE SPLIT AND THE FLOOR ARE BOTH MEASURED, by tests/sting_shape_check.gd: the
# largest change in drawn zoom between two consecutive frames, at the largest kick
# the sampler can produce (0.20). ghost's own eased camera covers 8% of its
# remaining error per frame (SceneView.smoothing = 5), so a 0.20 move starts at
# 1.6%/frame there; 5%/frame is three times that - snappy, and still a move rather
# than a jump. Hitting it needs a rise of at least ~0.11 s, which is why the span
# has a floor as well as a ceiling and why the rise takes 40% of it rather than a
# quarter. The old step measured 20%/frame - the entire kick, in one frame.
#
# The squared fall is the same soft tail the old `_sting * _sting` gave, so the
# release is the shape that always shipped; only the onset is new.
const STING_ATTACK := 0.40


func _sting_env() -> float:
	if _sting_t < 0.0:
		return 0.0
	var k := _sting_t / maxf(0.01, _sting_span)
	if k >= 1.0:
		return 0.0
	if k < STING_ATTACK:
		return smoothstep(0.0, 1.0, k / STING_ATTACK)
	var d := (k - STING_ATTACK) / (1.0 - STING_ATTACK)
	return (1.0 - d) * (1.0 - d)


# The content clock (manual mode): feed the [Echo] map at the frontier, listen for a
# re-localization, and act on one by re-seating the cursor and cutting through the
# NORMAL transition machinery (a stage->stage morph still carries live actors, so the
# show converges onto the corrected position rather than teleporting). Returns true
# when a correction fired this frame. The playhead is never consulted: a looped song,
# a doubled track, and a trimmed copy all re-converge because the AUDIO matches the
# map, not because a file position wrapped.
func _listen_echo(dt: float) -> bool:
	# Content-anchoring belongs to boards that FOLLOW the song (loop: false). A
	# `loop: true` board cycles its sequence by its own clock, deliberately unmoored
	# from audio position - re-localizing it would fight the author every wrap.
	if _echo == null or _storyboard_seq.is_empty() or _storyboard_loop \
			or Spectrum.song_length() <= 0.0:
		return false
	_heard_t += dt
	_cursor_t += dt
	# The ROLL: the map's recording cap is one full song, so a cursor that has walked
	# past it - with the sequence finished - has heard everything this content holds.
	# A looping session's next content is necessarily the top, so the arc rolls over
	# NOW, aligned by construction and with no recognition latency: the eye returns
	# the moment the song does. Echo's vote matcher below remains the backstop for
	# everything the roll can't know: trimmed, doubled, or cut-up audio, and drift.
	if _cursor_t >= Spectrum.song_length() and _step >= _storyboard_seq.size():
		print("ghost: echo - the song's map is exhausted, rolling the arc to the top")
		_step = 0
		_begin_transition()
		_cursor_t = 0.0
		return true
	if _audio_ema < silence_floor:
		return false                   # true silence: nothing to record, nothing to recognize
	# The FAST descriptor (~0.7s of context): recognition must notice the content
	# moving within a couple of seconds - the seeding descriptor's long memory would
	# spend the whole intro still tasting the outro.
	var sig := Spectrum.harmonic_signature_fast()
	# Write-once: only the frontier extends the map. The tail records too - the outro
	# must own its cells, or its mere RESEMBLANCE to earlier sections yanks the show
	# backward. The map covers exactly ONE hearing (a static content property): past
	# one song's worth of schedule the cursor's claim is stale (the audio has wrapped
	# but recognition hasn't fired yet), and recording there would poison the map.
	if _cursor_t <= Spectrum.song_length():
		_echo.record(_cursor_t, sig)
	var to := _echo.listen(_heard_t, _cursor_t, sig, dt)
	if to < 0.0:
		return false
	# A matched time past the sequence belongs to the TAIL: meaningful only if the
	# walk isn't already there (then the audio saying "outro" changes nothing).
	if to >= _sched_end - 0.5:
		if _manual_i < 0 or _storyboard_tail.is_empty():
			return false
		print("ghost: echo - audio matches schedule %.1fs, re-localizing to the tail" % to)
		_step = _storyboard_seq.size()
		_begin_transition()
		_cursor_t = to
		return true
	# HEAD rule: a match resolving into the OPENING stretch restarts the arc from the
	# very top instead of joining mid-entry. Recognition costs a couple of seconds, and
	# the opening entries are short - joining "where the audio is" would skip them
	# forever (the eye never returned; the show lived on prisms). Restarting the top a
	# little late is stable by construction: the lateness is inside Echo's self radius
	# (NEAR), so the localizer reads the offset show as "here" and leaves it alone; the
	# tail absorbs the shift before the next loop.
	if to <= Echo.NEAR:
		if _manual_i == 0:
			return false
		print("ghost: echo - audio matches schedule %.1fs, restarting the arc" % to)
		_step = 0
		_begin_transition()
		_cursor_t = 0.0
		return true
	var idx := 0
	for j in _sched_starts.size():
		if float(_sched_starts[j]) <= to + 0.01:
			idx = j
	if idx == _manual_i:
		return false                   # the match resolves to the entry already on screen
	print("ghost: echo - audio matches schedule %.1fs, re-localizing to entry %d" % [to, idx + 1])
	_step = idx
	_begin_transition()
	# Join the entry MID-FLIGHT: recognition costs a few seconds (the signature EMA
	# must shed the old content, then accumulate its votes), so by now the audio sits
	# some way INTO the entry. Enter at that offset - fast-forward the hold clock and
	# the scene's keyframe clock - so the rejoined schedule is aligned with the song
	# and STAYS aligned, instead of lagging by the recognition latency forever.
	var into := clampf(to - float(_sched_starts[idx]), 0.0, 8.0)
	var incoming: GhostScene = _next if _transitioning else _current
	if into > 0.25 and incoming != null:
		_elapsed = into
		var remaining := into
		while remaining > 1e-4:
			var dt2 := minf(remaining, SIM_STEP)
			incoming.update(Spectrum.current, dt2)
			incoming.view.commit(dt2)
			remaining -= dt2
	_cursor_t = to
	return true


func _should_change() -> bool:
	if _locked >= 0 or _held:
		return false
	if _game_paced and _paced_t < GAME_PACED_MIN * pacing:
		# `* pacing` because this floor is a hold duration like any other: left fixed it would be the
		# one gate the slider cannot move, and below pacing 1.0 a flat 12 s would swallow the whole
		# shortened range and make the control inert in synthesis mode.
		# the fishing owns the MOMENTS (catches and seed jumps cut at once);
		# past the floor, the normal harmonic exit logic below owns the tour
		return false
	# In manual mode, a non-looping storyboard with NO tail holds its final scene forever (checked
	# FIRST, before the fixed-hold check below, so the last entry's `hold` can't try to re-cut into
	# nothing). With a `tail:`, the sequence instead rolls into the tail entries, cycling on their
	# own exit rules until the song ends - so a finished arc keeps living rather than freezing.
	if not _storyboard_seq.is_empty() and not _storyboard_loop and _storyboard_tail.is_empty() \
			and _step >= _storyboard_seq.size():
		return false
	var ex: Dictionary = _current.exit_spec
	# A fixed hold is DETERMINISTIC authored timing: honor it exactly, ABOVE the auto-mode silence
	# and tail pacing gates. Those gates are heuristics for *cue-based* exits (don't cut into dead air
	# or a fading tail when we're waiting on a beat/lull); an author who wrote `hold: 16` means 16, and
	# because the value is fixed it is already identical between the live analyzer and the export bake,
	# so honoring it early keeps live/export parity while letting a tightly-timed piece (e.g. the-point,
	# whose 33s runs shorter than the 10s tail window) actually reach its finale.
	if ex.has("hold"):
		return _elapsed >= float(ex["hold"])
	# Never change scenes during silence: with no perceptible audio there is nothing to cut
	# on, and forcing a transition (a max-hold backstop, or a LULL trigger that silence
	# trivially satisfies) drops a fresh scene into dead air. Hold until the audio returns.
	if _audio_ema < silence_floor:
		return false
	# Tail gate: hold the final scene through the song's closing stretch, so a cut into a near-empty
	# FADING tail doesn't read as a glitch. This gate is DETERMINISTIC - a fixed `end_hold` window off
	# the (known) song length, NOT drive-scaled - because it decides the FINAL scene, and the live
	# real-time analyzer and the export's baked FFT read the audio slightly differently: a drive-scaled
	# gate would cross its threshold at different moments in each, so live and export could hold/land on
	# DIFFERENT last scenes. A fixed window crosses at the same song-time in both, so they always match.
	var slen := Spectrum.song_length()
	if slen > 0.0 and Spectrum.current.time >= slen - end_hold:
		return false
	# During a burst the backstop shrinks to a few seconds so quick scenes never linger.
	var hi: float = _burst_max if _burst_left > 0 else _scaled_bound(ex, "max", max_hold)
	if _elapsed >= hi:                               # backstop: the cue never came
		return true
	if not _ready_to_exit(ex):
		return false
	if not _trigger_fires(ex):
		return false
	# A CUE IS AN OFFER, NOT AN ORDER. See _cue_taken: the minimum hold used to be a hard gate,
	# and on speech the very next onset - a few hundredths of a second later - always took it, so
	# every scene exited at almost exactly the minimum. Now the offer is accepted on a probability
	# that ramps with how long the scene has already run.
	var lo_h: float = _burst_min if _burst_left > 0 else _scaled_bound(ex, "min", min_hold)
	return _cue_taken(lo_h, hi)


## HOW LONG THE CURRENT SCENE IS LIKELY TO HAVE LEFT, in seconds from now.
##
## For a vehicle that flies a camera, this is the difference between a shot and an
## interruption. [ComicVehicle] samples a move's duration from its own vocabulary - seven to
## twenty seconds - while the cut that ends it is decided here, from bounds that shrink with
## the music's drive: on a driving passage the median cut lands about five and a half seconds
## in. So the camera was routinely still converging on a panel when the next cut arrived, and
## the shot never resolved. Reported as "you can see it converging, then BOOM. The camera cuts
## to something else... we never gave the camera enough time to slow down, and settle
## somewhere, before the transition."
##
## THE MEDIAN, not the maximum, because the maximum is a backstop that usually does not
## happen. Between `lo` and `hi` the cue hazard is cubic (see [method _cue_taken]), so the
## survival curve is exp(-CUE_TOTAL * u^3) and half the scenes are gone by
## u = cbrt(ln2 / CUE_TOTAL) - about 56% of the way through the window. Sizing a move to the
## backstop would make every shot too slow; sizing it to the minimum would make every shot
## rushed. The median is the honest answer to "how long have I probably got".
##
## Returns a deliberately large number when the cutting is frozen (the feedback console, a
## probe) or when nothing is playing: a caller should then use its own natural timing rather
## than compress a move to fit a deadline that is not coming.
func hold_remaining() -> float:
	if _held or _current == null or not is_instance_valid(_current):
		return 1e9
	var ex: Dictionary = _current.exit_spec
	if ex.has("hold"):
		return maxf(0.0, float(ex["hold"]) - _elapsed)
	var hi: float = _burst_max if _burst_left > 0 else _scaled_bound(ex, "max", max_hold)
	var lo: float = _burst_min if _burst_left > 0 else _scaled_bound(ex, "min", min_hold)
	var u := pow(log(2.0) / CUE_TOTAL, 1.0 / 3.0)
	var med := lo + (hi - lo) * u
	if _elapsed < med:
		return med - _elapsed
	# PAST THE MEDIAN THE SCENE IS STILL RUNNING, so the answer is not zero.
	#
	# It used to be `maxf(0.0, med - _elapsed)`, which reports ZERO for the whole tail of any
	# hold that outlives its median - and the median is only 56% of the way through the window,
	# so that is most of the second half of every scene. A caller asking "how long have I got"
	# was told "none" over and over while the scene ran on, and [ComicVehicle] answered it the
	# way it should: by settling, again, every few seconds. Measured in an export log, 19 of 42
	# camera moves were settles, in runs of up to seven, each re-easing the shot a fifth of the
	# way toward the same panel it was already on. Reported as "the camera corrects and jumps to
	# focus on the exact same frame it's already on" and "it never holds long enough to look at
	# anything".
	#
	# Beyond the median the honest remaining time is the distance to the BACKSTOP, which is the
	# one moment the cut is certain. It shrinks to zero as that arrives, so a caller pacing
	# itself against this still tightens as the scene ends - it simply is not told the scene is
	# over half a minute before it is.
	return maxf(0.0, hi - _elapsed)


## Take this cue, or let it pass? A ramp, not a gate.
##
## THE BUG THIS FIXES. `min_hold` was a hard gate: once the scene had held that long, the next
## trigger cut. On speech the beat detector fires an onset several times a second, so "the next
## trigger" is always within a frame or two of the gate opening - measured, a 7.29 s median with an
## IQR of 0.94 s, i.e. every scene exiting at essentially the same instant. Raising the hold just
## moved the metronome: at the top of the Scene hold slider it cut every ~15 s almost on the dot.
##
## So eligibility no longer decides anything by itself. Past the minimum, an arriving cue is
## ACCEPTED with a probability that starts near zero and rises the longer the scene has run:
## possible at the minimum, likely by the backstop. `u` is the position through the window and the
## hazard goes as u², so the first seconds past the gate are genuinely unlikely to cut while a
## scene that has already run a long time is under real pressure to.
##
## IT IS A RATE PER SECOND, not a chance per cue, and that distinction is the whole point: speech
## offers cues constantly and music offers them sparsely, and integrating the hazard over the time
## since the last offer makes both content types produce the same distribution of scene lengths.
## A per-cue probability would cut speech ten times faster than music for no musical reason.
##
## CUE_TOTAL is the integrated hazard across the whole window - the expected number of "cut now"
## events if the scene somehow survived to the backstop. Around 4 puts the median a little past
## halfway and leaves only a couple of percent to reach the backstop. Because both bounds scale
## together with `pacing`, the SHAPE of the distribution is identical at every setting: the slider
## stretches the spread, it does not flatten it.
const CUE_TOTAL := 4.0


func _cue_taken(lo: float, hi: float) -> bool:
	var span := maxf(0.001, hi - lo)
	var u := clampf((_elapsed - lo) / span, 0.0, 1.0)
	var hazard := 3.0 * CUE_TOTAL / span * u * u     # per second; integrates to CUE_TOTAL over span
	# Time since the LAST offer, floored at the moment this scene became eligible so the first cue
	# past the gate cannot claim credit for the whole hold before it.
	var dt := maxf(0.0, _elapsed - maxf(_cue_prev, lo))
	_cue_prev = _elapsed
	return _rng.randf() < 1.0 - exp(-hazard * dt)


# Scene holds scale with the music's DRIVE: how hard it is pushing RIGHT NOW. A loud, fast, active
# passage shrinks both the minimum and maximum hold so scenes cut faster; a calm one lets them
# linger. Energy is the reliable backbone (a beat-period estimate alone barely moves for fast music
# and stalls when onsets are missed); a quick pulse and busy spectral flux push it further. Returns
# a hold multiplier from `pace_drive_scale` (full drive -> fast cuts) up to `pace_calm_scale`.
func _pacing_scale() -> float:
	var f := Spectrum.current
	var fast := clampf((0.58 - f.beat_period) / 0.24, 0.0, 1.0)      # 1 when the pulse is quick
	var drive := clampf(pace_energy_gain * _audio_ema + 0.45 * fast + 3.0 * f.flux, 0.0, 1.0)
	return lerpf(pace_calm_scale, pace_drive_scale, drive)


# A hold bound: a storyboard-explicit value is taken literally (already sensitivity-scaled in
# _make_scene); the auto-mode default is pace-scaled AND divided by the active sensitivity, so a
# higher tempo also makes auto cuts come faster (more of the catalogue in the same time), then
# multiplied by the user's `pacing`. Multiplying LAST and multiplying BOTH bounds by the same factor
# is what preserves the show's variance: _pacing_scale still swings the hold by ~3.4x across the
# music's drive, and max/min keeps its exact ratio - the whole distribution just slides longer.
func _scaled_bound(ex: Dictionary, key: String, base: float) -> float:
	return float(ex[key]) if ex.has(key) else base * _pacing_scale() * pacing / maxf(0.05, _cur_sens)


# Eligibility: a oneshot when its sequence ends, a loop after the minimum hold (a short one in a
# burst, so a quick scene becomes eligible to cut almost immediately - on the next beat).
func _ready_to_exit(ex: Dictionary) -> bool:
	if _current.lifecycle == "oneshot":
		return _current.finished()
	var lo: float = _burst_min if _burst_left > 0 else _scaled_bound(ex, "min", min_hold)
	return _elapsed >= lo


# Has the exit cue arrived this frame? Uses the storyboard-specified trigger if the
# scene carries one, otherwise the randomly-armed trigger (auto mode).
func _trigger_fires(ex: Dictionary) -> bool:
	var trig: int = int(ex.get("trigger", _trigger))
	var f := Spectrum.current
	match trig:
		Trigger.BEAT:
			return f.beat > 0.5 and _beat_prev <= 0.5    # rising edge of a beat
		Trigger.MOVEMENT:
			return f.movement >= movement_threshold
		Trigger.LULL:
			return f.energy <= lull_threshold
	return false


## Force the next change now (bound to Space in main).
func next() -> void:
	if _host == null:           # session not started yet (splash still up)
		return
	if not _transitioning:
		_begin_transition()


## Freeze/unfreeze scene cuts (the feedback console holds the current scene on
## screen while you type, so it doesn't change out from under your critique).
## Scenes keep animating; only the Director's exit logic is paused.
func hold(on: bool) -> void:
	_held = on


## GAME PACING (synthesis mode): the fishing game owns the cuts. Autonomous
## music-driven scene changes stop entirely - a scene changes when a catch
## jumps it (or Space skips manually) - so each new scene reads as a REWARD,
## not weather. Distinct from hold(): the feedback console toggles that and
## must not accidentally release the game's grip.
var _game_paced := false
var _paced_t := 0.0          # seconds the current scene has held in game-paced mode
const GAME_PACED_MIN := 12.0    # floor before the MUSICAL pacing may cut a
                                # game-paced scene: a just-jumped seed scene gets
                                # read as the reward it is, then the normal
                                # harmonic exits (beat / movement / lull) own the
                                # tour again. (A fixed 70 s hold was tried and
                                # read as frozen - the music makes better cuts.)

func set_game_paced(on: bool) -> void:
	_game_paced = on


## The METAMORPHOSIS bus: while a catch is being reeled in, its influence
## contorts the current scene - every GhostScene reads this in tick() and
## stretches its motion tempo, zoom-breathing, and pan-sweeps by it. 0 = no
## contortion; big catches push past 1.
var aura := 0.0

func set_aura(v: float) -> void:
	aura = clampf(v, 0.0, 1.5)


## Jump the show to the scene a seed OWNS: the index derives from the seed
## value, so each seed's scene is part of its identity - catching it, or
## restoring it from the belt, brings back its place in the catalogue.
var _jump_next := -1

func jump(seed_val: int) -> void:
	# the pending jump is stored EVEN WITHOUT a session: a seed clicked before
	# the first throw must own the session's INITIAL scene too - _pick_index
	# consumes _jump_next when attach() builds it. (The old early return made
	# pre-session jumps silent no-ops, so the first scene of a run fell back
	# to the session fingerprint - which the throw jitter re-rolls every cast:
	# "the initial scene is ALWAYS different across runs.")
	_jump_next = absi(seed_val) % maxi(SCENES.size(), 1)
	if _host == null:
		return                       # no session yet: the jump waits for attach()
	next()


## The scene name a seed value maps to (for tooltips and ledgers).
func scene_title(seed_val: int) -> String:
	var i := absi(seed_val) % maxi(SCENES.size(), 1)
	return String(SCENES[i].script.resource_path).get_file().get_basename()


## A typed snapshot of the scene currently on screen, for the feedback console.
## Everything the on-disk record needs to tie a critique back to a reproducible
## scene: identity (name/behavior/shot/seed/song), its typed definition (params),
## and the audio frame it was reacting to. Values may contain Godot types
## (Vector2 / Color); FeedbackConsole.to_jsonable flattens them before writing.
func current_descriptor() -> Dictionary:
	if _current == null:
		return {}
	var f := Spectrum.current
	var d := {
		"scene": _current.scene_name,
		"render_kind": _current.render_kind,
		"behavior": _current.behavior_name,
		"shot": _current.shot_name,
		"framing": _current.framing,
		"lifecycle": _current.lifecycle,
		"seed": _current.seed_value,
		"session_seed": _session_seed,
		"song_hash": Spectrum.song_hash,
		"params": _current.params,
		"audio": {
			"time": f.time,
			"energy": f.energy,
			"beat": f.beat,
			"bass": f.bass,
			"low_mid": f.low_mid,
			"mid": f.mid,
			"high": f.high,
			"treble": f.treble,
			"flux": f.flux,
			"movement": f.movement,
		},
	}
	# If a blend is mid-flight, record which style and how far - so feedback taken
	# during a rough transition says exactly what was happening (the "incoming"
	# scene named here is the one being revealed; the outgoing one is leaving).
	if _transitioning:
		d["transition"] = {
			"active": true,
			"style": _style_name(_style),
			"progress": clampf(_trans_t, 0.0, 1.0),
			"incoming": _next.scene_name if _next != null else "",
			"transition_time": transition_time,
		}
	return d


func _style_name(s: int) -> String:
	match s:
		Style.CUT: return "cut"
		Style.DIP: return "dip"
		Style.FADE: return "fade"
		Style.LAYER: return "layer"
	return "?"


# Outgoing (x) and incoming (y) alpha for transition progress k, by style. DIP
# sequences them so they never overlap: the old scene fades to black by ~0.38, a
# beat of darkness holds, then the new scene fades up after ~0.55 - the gap the eye
# wants. FADE is a plain crossfade; anything else is a linear dissolve.
func _transition_alphas(k: float) -> Vector2:
	match _style:
		Style.DIP:
			var out_a := smoothstep(0.0, 1.0, clampf(1.0 - k / 0.38, 0.0, 1.0))
			var in_a := smoothstep(0.0, 1.0, clampf((k - 0.55) / 0.45, 0.0, 1.0))
			return Vector2(out_a, in_a)
		Style.FADE:
			return Vector2(smoothstep(0.0, 1.0, 1.0 - k), smoothstep(0.0, 1.0, k))
		Style.LAYER:
			# Async overlap. First the incoming fades IN to a TRANSLUCENT level over the still-full
			# outgoing (so the outgoing shows through it - they layer); then the outgoing fades OUT
			# while the incoming solidifies to full, surviving. The two are offset apart (bias) so
			# they compose rather than collide.
			if k < 0.45:
				return Vector2(1.0, smoothstep(0.0, 1.0, k / 0.45) * 0.65)
			var kk := (k - 0.45) / 0.55
			return Vector2(1.0 - smoothstep(0.0, 1.0, kk), lerpf(0.65, 1.0, smoothstep(0.0, 1.0, kk)))
		_:
			return Vector2(1.0 - k, k)


func _begin_transition() -> void:
	_paced_t = 0.0
	if SCENES.size() < 2:
		_elapsed = 0.0
		_cue_prev = 0.0
		return
	var burst_cut := _burst_left > 0      # leaving a burst scene -> a hard jump cut, no morph/blend
	if _burst_left > 0:
		_burst_left -= 1                  # consume this quick scene
	# CAST-OWNING VEHICLE: the change is a handover, not a construction. Taken before the
	# stinger reset below so the punch clears off the scene being left exactly as it would
	# on any other change.
	var handed := _handover(_current)
	if handed != null:
		if _current != null and is_instance_valid(_current):
			_current.view.pulse_zoom = 1.0
			_current.view.pulse_rot = 0.0
			_current.view.pulse_skew = 0.0
			_current.modulate = Color(1.0, 1.0, 1.0, _current.modulate.a)
		_sting_left = 0
		_sting_t = -1.0
		_current = handed
		_swaps += 1
		_arm()
		return
	# Clear any rapid-fire modulation so the leaving scene doesn't freeze mid-contortion or tint.
	_sting_left = 0
	_sting_t = -1.0
	if _current != null:
		_current.view.pulse_zoom = 1.0
		_current.view.pulse_rot = 0.0
		_current.view.pulse_skew = 0.0
		_current.modulate = Color(1.0, 1.0, 1.0, _current.modulate.a)
	var nxt := _make_scene()

	# Content-aware morph: if the incoming can grow out of the outgoing's geometry,
	# swap instantly and let it animate the morph (e.g. one eye splitting into two).
	# Only ever between compatible, non-empty types - so we never morph a mismatch. (Not during a
	# burst - a flurry wants clean jump cuts, not a slow morph.)
	if not burst_cut and _current != null and not nxt.morph_in.is_empty() and nxt.morph_in == _current.morph_out:
		print("ghost: morph %s -> %s (%s)" % [_current.scene_name, nxt.scene_name, nxt.morph_in])
		var from := _current
		_scene_host(nxt).add_child(nxt)
		_current = nxt
		_swaps += 1
		nxt.begin_morph(from)         # hand over state BEFORE the source is freed
		from.queue_free()
		_arm()
		return

	_style = _choose_style()
	if burst_cut:
		_style = Style.CUT            # a burst is a run of hard jump cuts
	# A LAYER overlap only reads well when the incoming is an atmospheric wash; otherwise two
	# busy looks fight, so fall back to a clean dip.
	if _style == Style.LAYER and not ATMOSPHERIC.has(nxt.get_script().resource_path):
		_style = Style.DIP
	if _style == Style.CUT:
		_scene_host(nxt).add_child(nxt)   # instant swap, no blend
		_current.queue_free()
		_current = nxt
		_swaps += 1
		_arm()
		return

	# Layer overlap: pan / zoom only the INCOMING scene into an off-centre region so it composes
	# beside the outgoing one without sitting on the same focal point. The OUTGOING scene is left
	# exactly where it is (it just fades) - it must never re-shift to "make room", which read as a
	# jarring lurch. The incoming eases from neutral into this bias (a pan-in) and then holds it.
	if _style == Style.LAYER:
		var ang := _rng.randf_range(0.0, TAU)
		nxt.view.bias_offset = Vector2(cos(ang), sin(ang)) * _rng.randf_range(0.16, 0.30)
		nxt.view.bias_zoom = _rng.randf_range(0.80, 1.05)
		print("ghost: layer %s under %s" % [nxt.scene_name, _current.scene_name])

	# Start the incoming scene fully transparent BEFORE it is ever drawn - otherwise
	# it flashes at full alpha for the one frame between being added and the first
	# alpha update (the "appeared, disappeared, reappeared" bug).
	nxt.modulate.a = 0.0
	nxt.view.presence = 0.0
	nxt.view.reveal = 0.0             # arm the geometry ratchet: it grows in with the fade-up
	_next = nxt
	_scene_host(_next).add_child(_next)   # added last -> drawn over _current
	_transitioning = true
	_trans_t = 0.0


# The transition style for leaving the current scene: its storyboard-set style
# (cut/dip/fade), or the auto-mode weighted bag (mostly dip) when unspecified.
func _choose_style() -> int:
	var s: int = STYLE_BAG[_biased(STYLE_BAG.size())]
	match (_current.transition_style if _current != null else ""):
		"cut": s = Style.CUT
		"dip": s = Style.DIP
		"fade": s = Style.FADE
	# The vehicle has the last word, and only ever to REJECT: a comic page cannot play a
	# LAYER (two scenes composited into one panel is mud). The draw off STYLE_BAG happens
	# either way, above, so the seeded stream advances identically whatever the vehicle
	# does with the answer - switching presentation must not re-roll the whole show.
	if _vehicle != null and is_instance_valid(_vehicle):
		s = _vehicle.style_for(s)
	return s


func _finish_transition() -> void:
	_current.queue_free()
	_current = _next
	_next = null
	_transitioning = false
	_swaps += 1
	_current.modulate.a = 1.0
	_current.view.presence = 1.0
	_current.view.reveal = 1.0
	# The survivor of a LAYER HOLDS the position it took during the overlap - it does not shift
	# back to the centred focal point (that snap-back read as wrong). Its bias stays as set.
	_arm()


# Novelty-weighted scene choice. The catalogue lists several entries per scene
# *kind* (the same script with different behaviors), and a uniform random pick
# clusters - the same kind recurs while others go unseen. Instead, weight each
# candidate by how long its kind has gone unshown, so long-unseen scenes are drawn
# far more often than recent duplicates: a soft priority queue, not a hard rotation.
# Still driven by the seeded _rng, so a given song yields the same sequence.
func _pick_index() -> int:
	if _jump_next >= 0:
		var j := _jump_next
		_jump_next = -1
		return j
	if _locked >= 0:
		return _locked
	if SCENES.size() <= 1:
		return 0
	# Identity-keyed weighted selection (Efraimidis-Spirakis): each candidate gets a STABLE
	# per-cut uniform from a hash of (session, cut#, this scene's identity), and we keep the one
	# with the largest key = u^(1/weight). Because each scene's key depends only on ITS OWN
	# identity - never on the catalogue's size or order - adding a new animation can only change
	# the cuts it actually wins; it does not reshuffle the rest of the show. The hash is fully
	# DETERMINISTIC per song (the session seed is the song fingerprint): the same audio picks the
	# same scenes in the same order. It used to also fold in the live Spectrum.seed_bias(), but
	# that samples the spectrum at the cut instant and is not frame-reproducible, so it re-rolled
	# the running order every playback.
	var best := -1
	var best_key := -1.0
	for i in SCENES.size():
		var w := _novelty_weight(i)
		if w <= 0.0:
			continue
		var h := hash([_session_seed ^ (_pick_salt * 0x27D4EB2F), _swaps, _scene_key(i)])
		var u := clampf(float(h & 0xFFFFFFFF) / 4294967296.0, 1e-9, 1.0)
		var key := pow(u, 1.0 / w)
		if key > best_key:
			best_key = key
			best = i
	return best if best >= 0 else _rng.randi() % SCENES.size()   # all suppressed: fall back


# A STABLE identity for a catalogue entry: a hash of (scene name, behavior), independent of its
# position in SCENES. Keying seeds and selection off this - not the array index - is what lets us
# add or reorder scenes without changing how the existing ones are chosen or how they look.
func _scene_key(i: int) -> int:
	var e: Dictionary = SCENES[i]
	return hash(String(e.script.resource_path).get_file().get_basename() + "|" + String(e.behavior))


## How many cuts back the novelty term can see. Past this a kind is simply DUE, and no more due
## than any other kind that is also due - which is the whole difference between sampling the
## catalogue and queueing it. Chosen by measuring the first-repeat statistic, not by taste; see
## tests/scene_mix_check.gd.
const NOVELTY_SPAN := 5.0

## How sharply novelty rises inside that span. 1.0 is linear in cuts-since-last-seen.
const NOVELTY_EXP := 1.0


# Selection weight for one catalogue entry: 0 for the entry on screen (never an
# immediate repeat), tiny for another behavior of the *same* scene (so we don't
# show two of one kind back to back), and otherwise a MILD preference for kinds that
# have not been seen lately.
#
# THE ROTATION. This used to be `pow(age, 1.6)` over an UNBOUNDED age, with a kind that had
# never been shown handed `age = _swaps + 1000` by the dictionary default - a weight of about
# 251,000 against 40 for a kind last seen ten cuts ago, or 6,300 to 1. At odds like that the
# scheduler could not repeat anything until it had shown EVERYTHING, so the running order was a
# rotation through the catalogue wearing the clothes of a weighted random draw, and the function
# above promised "a soft priority queue, not a hard rotation" while delivering the opposite.
#
# Measured before the change (tests/scene_mix_check.gd, 40 sessions x 140 cuts over 52 kinds):
# the first repeated kind arrived at cut 50.0, where a genuine random draw over 52 kinds repeats
# at about cut 9.0 and a complete sweep would be cut 53. A video therefore played very nearly
# the whole catalogue once through before it ever came back to anything, which is what "the
# variety of generated scenes is not truly random" was describing.
#
# Novelty is BOUNDED now, so the tail cannot dominate. What it still buys is the thing it was
# added for: two behaviors of one scene do not follow each other, and a kind shown a moment ago
# is unlikely to return immediately. What it no longer does is decide the whole running order.
func _novelty_weight(i: int) -> float:
	if i == _index:
		return 0.0
	var kind := String(SCENES[i].script.resource_path)
	if _index >= 0 and kind == String(SCENES[_index].script.resource_path):
		return 0.05
	# An unseen kind is MAXIMALLY due, not infinitely due. The -1000 sentinel is unchanged and
	# harmless now that the clamp is what decides how far novelty reaches.
	var age := float(_swaps - int(_kind_last.get(kind, -1000)))
	return pow(clampf(age, 1.0, NOVELTY_SPAN), NOVELTY_EXP)


# Resolve the next scene to build: from the novelty scheduler (auto mode) or the
# next storyboard entry (manual mode). Returns {script, behavior, seed, shot, exit_spec}.
func _next_entry() -> Dictionary:
	if _storyboard_seq.is_empty():
		_index = _pick_index()
		_kind_last[String(SCENES[_index].script.resource_path)] = _swaps
		var e: Dictionary = SCENES[_index]
		# session identity (from the song fingerprint) ^ scene IDENTITY (not its array slot) ^ history
		# (swaps). Fully DETERMINISTIC: the same song yields the same seed here every run, so a scene's
		# structure - clockwork's gears, a terrain's shape - reproduces exactly. The live
		# Spectrum.seed_bias() used to be XOR'd in for extra harmonic steering, but it samples the
		# spectrum at the cut instant and is not frame-reproducible, which re-rolled the look each play.
		var seed := _session_seed ^ _scene_key(_index) ^ (_swaps * 0x85EBCA77) \
			^ (_pick_salt * 0x165667B1)
		return {"script": e.script, "behavior": e.behavior, "seed": seed,
			"shot": "", "exit_spec": {}, "transition": "", "sensitivity": sensitivity}   # "" -> auto STYLE_BAG
	# Manual: walk the sequence (wrap when looping; past the end of a non-looping board,
	# cycle the tail entries if there are any, else hold on the last entry).
	var n := _storyboard_seq.size()
	var i: int
	var item: Dictionary
	if _storyboard_loop:
		i = _step % n
		item = _storyboard_seq[i]
	elif _step < n or _storyboard_tail.is_empty():
		i = mini(_step, n - 1)
		item = _storyboard_seq[i]
	else:
		var t := (_step - n) % _storyboard_tail.size()
		i = n + t                                    # distinct index -> distinct derived seed
		item = _storyboard_tail[t]
	_manual_i = i if i < n else -1                   # -1 = a tail entry
	if i < n and i < _sched_starts.size():
		_cursor_t = float(_sched_starts[i])          # a sequence entry claims its scheduled start;
	_step += 1                                       # the tail free-runs from wherever it began
	var nm := String(item.get("scene", ""))
	var path := "res://scripts/scenes/%s.gd" % nm
	var script: Resource = load(path) if ResourceLoader.exists(path) else SCENES[0].script
	if not ResourceLoader.exists(path):
		push_warning("ghost: storyboard scene '%s' not found, substituting" % nm)
	# The seed is keyed to the entry's POSITION, never to how many times it has been
	# visited: the same section of the schedule must rebuild the SAME scene when the
	# audio brings the show back to it (an echo re-localization, a `loop: true` wrap).
	# (i + 1) reproduces the value the old visit-counter formula gave on a first pass,
	# so existing shows re-render unchanged.
	var seed2: int = int(item.get("seed",
		_session_seed ^ (i * 0x9E3779B1) ^ ((i + 1) * 0x85EBCA77)))
	# Sensitivity resolves per entry: the entry's own value, else the storyboard's, else the export.
	var sb_sens: float = _storyboard_sensitivity if _storyboard_sensitivity > 0.0 else sensitivity
	var sens: float = float(item.get("sensitivity", sb_sens))
	return {"script": script, "behavior": String(item.get("behavior", "drift")),
		"seed": seed2, "shot": String(item.get("shot", "")), "exit_spec": _parse_exit(item),
		"transition": String(item.get("transition", "")), "sensitivity": sens,
		"spec": item}   # the raw entry rides along: a data-driven scene (stage) reads it in build_params


# Translate a storyboard entry's timing into an exit_spec the scene carries (see
# _should_change): a fixed `hold`, a musical `exit` trigger, or just min/max bounds.
func _parse_exit(item: Dictionary) -> Dictionary:
	if item.has("hold"):
		return {"hold": float(item["hold"])}
	if item.has("exit"):
		# An authored bound is literal; an absent one falls back to the Director's OWN defaults, which
		# are not the author's timing, so they carry `pacing` like every other default bound (without
		# it, `exit: beat` alone would be the one entry shape that ignores the slider while a bare
		# entry - which exits through _scaled_bound's auto branch - obeys it). Both bounds take the
		# same factor, so the spread between them is untouched. Baked when the scene is built, so a
		# pacing change reaches these entries at the next cut rather than mid-scene.
		return {
			"trigger": _trigger_from_name(String(item["exit"])),
			"min": float(item["min_hold"]) if item.has("min_hold") else min_hold * pacing,
			"max": float(item["max_hold"]) if item.has("max_hold") else max_hold * pacing}
	var d := {}
	if item.has("min_hold"):
		d["min"] = float(item["min_hold"])
	if item.has("max_hold"):
		d["max"] = float(item["max_hold"])
	return d


func _trigger_from_name(s: String) -> int:
	match s.to_lower():
		"movement":
			return Trigger.MOVEMENT
		"lull":
			return Trigger.LULL
		_:
			return Trigger.BEAT


# Instantiate + seed the next scene with its behavior, shot, and exit rule.
## Build ONE scene, the way a cut does, WITHOUT putting it on screen - for a vehicle that
## owns its own cast ([method Vehicle.owns_cast]). [param salt] separates several mints
## made at the same instant; pass a different one per panel.
##
## It goes through the same novelty scheduler as everything else rather than picking
## scenes some other way, so a comic page is cast by the same rules that order a full-frame
## show - and because `_next_entry` records each pick in `_kind_last` as it goes, the
## panels of one page vary against each other for free.
func mint_scene(salt := 0, quiet := false) -> GhostScene:
	_pick_salt = salt
	_quiet_mint = quiet
	var sc := _make_scene()
	_pick_salt = 0
	_quiet_mint = false
	return sc


## Suppresses the per-cut [signal scene_cut] while casting a page. The signal means "a new
## scene took the stage, re-measure it" (see main's stage governor), and casting six panels
## in one frame is not six of those.
var _quiet_mint := false


func _make_scene() -> GhostScene:
	var entry := _next_entry()
	var script: Resource = entry["script"]
	var scene: GhostScene = script.new()
	var seed: int = int(entry["seed"])
	scene.spec = entry.get("spec", {})           # BEFORE init_with_seed: build_params reads it
	scene.init_with_seed(seed, String(entry["behavior"]))
	scene.scene_name = String(script.resource_path).get_file().get_basename()
	# Telemetry only: the live harmonic bucket at the cut. It is NO LONGER folded into the scene
	# seed (that broke run-to-run reproducibility); the seed is deterministic per song. Same music
	# should still print the same bucket here, a useful observability signal. (See
	# next/harmonic_seeding.md.)
	# STAMPED WITH THE SONG POSITION, always. Without it the cut list is an ordered list of names
	# and nothing else, and the only question anyone actually asks of it - "what was on screen at
	# 8:23?" - cannot be answered from a finished render at all. That is not hypothetical: a scene
	# shipped 29 seconds of pure black into an export and the log could not name it.
	var _t := maxf(Spectrum.current.time, 0.0)
	print("ghost: cut -> %s  at %d:%05.2f  harmonic bucket %d"
		% [scene.scene_name, int(_t / 60.0), fmod(_t, 60.0), Spectrum.harmonic_bucket(12)])
	if not _quiet_mint:
		scene_cut.emit()              # the stage governor re-measures per scene
	# Narrative tempo: higher sensitivity shrinks the hold (and any explicit min/max bounds), so the
	# scene is shorter; the scene paces its keyframes as fractions of that shrunken hold, so events
	# still all land. _cur_sens also feeds the auto-mode pacing bounds (see _scaled_bound).
	_cur_sens = clampf(float(entry.get("sensitivity", 1.0)), 0.05, 20.0)
	var ex: Dictionary = entry["exit_spec"]
	for k in ["hold", "min", "max"]:
		if ex.has(k):
			ex[k] = float(ex[k]) / _cur_sens
	scene.exit_spec = ex
	scene.event_scale = _cur_sens
	# Transition style, by override hierarchy (highest first): storyboard entry, then
	# the scene's own choice (set in build_params), then the storyboard's default,
	# then the mode default (manual = cut, auto = "" -> the weighted STYLE_BAG). A
	# compatible morph still wins over all of these at change time.
	var entry_tr := String(entry.get("transition", ""))
	if not entry_tr.is_empty():
		scene.transition_style = entry_tr
	elif not scene.transition_style.is_empty():
		pass                                       # keep the scene's own override
	elif not _storyboard_transition.is_empty():
		scene.transition_style = _storyboard_transition
	elif not _storyboard_seq.is_empty():
		scene.transition_style = "cut"             # manual default: jump cuts
	# Camera framing: an explicit storyboard shot if given and valid, else assigned by
	# the scene's framing class (expressive for subjects, gentle for fields, square
	# for lone planes).
	var shot_name := String(entry.get("shot", ""))
	if shot_name == "" or not Shots.REGISTRY.has(shot_name):
		var bag: Array = Shots.SUBJECT_BAG
		if scene.framing == "field":
			bag = Shots.FIELD_BAG
		elif scene.framing == "plane":
			bag = Shots.PLANE_BAG
		if _storyboard_seq.is_empty():
			shot_name = bag[_rng.randi() % bag.size()]
		else:
			# Manual mode: the pick derives from the entry's own seed, not the session
			# rng stream - a revisit (echo re-localization, loop wrap) must rebuild the
			# same framing it had the first time, whatever was drawn in between.
			shot_name = bag[absi(hash([seed, "shot"])) % bag.size()]
	scene.shot_name = shot_name
	scene.set_shot(Shots.make(shot_name, seed ^ 0x51ED2701))
	# Pre-warm stateful motion (growth envelopes, scroll phase, tumbling angles)
	# and ease the camera to its shot framing, so the first shown frame is settled.
	for w in 12:
		scene.update(Spectrum.current, 0.05)
		scene.view.commit(0.05)
	scene.view.snap()        # finish the ease EXACTLY, so the first shown frame doesn't slide into place
	# NB: this build burns real wall-time (pre-warm + the first heavy draw), which the music clock keeps
	# advancing through. That big step is now absorbed by the STEP_CAP / debt smoothing in _process (it
	# eases in over the next frames), so no clock reset is needed here.
	return scene


func _exit_tree() -> void:
	# Free live scenes so they don't report as leaked when the app quits.
	if is_instance_valid(_current):
		_current.queue_free()
	if is_instance_valid(_next):
		_next.queue_free()


