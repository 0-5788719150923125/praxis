extends Node

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
	{"script": preload("res://scripts/scenes/projection.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/projection.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/terrain.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/terrain.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/terrain_city.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/spires.gd"), "behavior": "drift"},
	# "the-point" scenes (camera holds, per the brief).
	{"script": preload("res://scripts/scenes/eye.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/two_eyes.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/prism.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/prism_split.gd"), "behavior": "static"},
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
	"res://scripts/scenes/clouds.gd", "res://scripts/scenes/fog_volume.gd",
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
## Seconds a dip/blend takes end to end (cuts are instant). A DIP spends the middle
## of this in darkness, so a little long reads as a deliberate breath between scenes.
@export var transition_time: float = 2.0
## A LAYER transition is slower than a dip - the two scenes overlap and shift for a while before
## the first leaves - so it gets its own, longer duration.
@export var layer_time: float = 6.0

var _host: Node = null
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
var _bookend_time := 6.0     # seconds of the start fade-up-from-black and end fade-down-to-black
# Rapid-fire BURST: a sparse, harmonic-gated flurry of quick jump cuts (a cinematic "3 quick
# scenes" effect) breaking up the slow holds. While a burst is live the holds shrink to a few
# seconds and every exit is a hard CUT landing on the beat.
var _burst_left := 0         # quick scenes remaining in the burst (0 = normal pacing)
var _burst_min := 1.5        # this burst's minimum hold (s)
var _burst_max := 4.0        # this burst's maximum hold (s)
var _flurry_cd := 0          # scenes until another flurry (burst or stinger) may start - keeps them rare
# Rapid-fire STINGER: instead of cutting through different scenes (jarring at speed), a run of
# beat-synced PUNCHES that contort / recolour / zoom the CURRENT scene - BANG, BANG, BANG - then
# settle. A universal modulation: it rides the SceneView pulse + node tint, so it works on any scene.
var _sting_left := 0         # beat-synced punches remaining in the run
var _sting := 0.0            # current punch envelope (1 on the beat -> 0 between)
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
var _storyboard_name := ""
var _storyboard_loop := true
var _storyboard_transition := ""    # default transition style for a storyboard ("" = cut in manual mode)
var _step := 0


func attach(host: Node) -> void:
	_host = host
	_session_seed = _resolve_seed()
	print("ghost: session seed %d (%s)" % [_session_seed, _seed_source()])
	_rng.seed = _session_seed ^ 0x1234567
	# A long-ish start/end fade, sampled per song into [3, 10] s. Derived from a HASH of the seed
	# (not a draw off _rng) so it doesn't perturb the deterministic scene sequence.
	_bookend_time = 3.0 + 7.0 * (float(hash([_session_seed, "bookend"]) & 0xFFFF) / 65535.0)
	_locked = _locked_scene_arg()
	_load_storyboard_arg()
	_current = _make_scene()
	_host.add_child(_current)
	_arm()


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
	if _transitioning and is_instance_valid(_next):
		_next.queue_free()
	if is_instance_valid(_current):
		_current.queue_free()
	_current = null
	_next = null
	_host = null
	_transitioning = false
	_trans_t = 0.0
	_index = -1
	_swaps = 0
	_step = 0
	_elapsed = 0.0
	_held = false
	_kind_last = {}
	_storyboard_seq = []
	_storyboard_name = ""


## True when the Director is walking a user-authored storyboard (manual mode).
func is_manual() -> bool:
	return not _storyboard_seq.is_empty()


## Name of the active storyboard, or "" in auto mode.
func storyboard_name() -> String:
	return _storyboard_name


## Load a storyboard by name (res://storyboards/<name>.json) or by a full/absolute path,
## switching the Director into manual mode. Returns true on success. Safe to call
## before attach(); the splash uses this to start a manually-orchestrated session.
func load_storyboard(name_or_path: String) -> bool:
	var path := name_or_path
	if not path.ends_with(".json"):
		path = "res://storyboards/%s.json" % name_or_path
	if not FileAccess.file_exists(path):
		push_warning("ghost: storyboard not found: %s" % path)
		return false
	var text := FileAccess.get_file_as_string(path)
	var data: Variant = JSON.parse_string(text)
	if typeof(data) != TYPE_DICTIONARY or not data.has("sequence"):
		push_warning("ghost: storyboard %s has no 'sequence' array" % path)
		return false
	var seq: Variant = data["sequence"]
	if typeof(seq) != TYPE_ARRAY or (seq as Array).is_empty():
		push_warning("ghost: storyboard %s sequence is empty" % path)
		return false
	_storyboard_seq = seq
	_storyboard_loop = bool(data.get("loop", true))
	_storyboard_name = String(data.get("name", name_or_path))
	_storyboard_transition = String(data.get("transition", ""))   # e.g. "cut" forces jump cuts
	_step = 0
	print("ghost: storyboard '%s' loaded (%d scenes, loop=%s)" % [
		_storyboard_name, _storyboard_seq.size(), _storyboard_loop])
	return true


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
	_beat_prev = Spectrum.current.beat
	if _flurry_cd > 0:
		_flurry_cd -= 1            # count down the spacing between flurries
	if _burst_left == 0:
		_maybe_start_burst()       # rarely, kick off a rapid-fire CUT burst on this scene
	# In a burst, exits land on the beat (quick + musical); otherwise the weighted cue bag.
	_trigger = Trigger.BEAT if _burst_left > 0 else TRIGGER_BAG[_biased(TRIGGER_BAG.size())]


# A sparse, NON-LINEAR chance to start a rapid-fire CUT burst: only in the auto show, only with
# real audio, weighted up by how much the music is moving right now (a spike curve on movement +
# energy). Rare on purpose, with a long cooldown after, so it does not chain into a dozen cuts.
func _maybe_start_burst() -> void:
	if not _storyboard_seq.is_empty() or _locked >= 0 or _audio_ema < 0.15 or _flurry_cd > 0:
		return
	var f := Spectrum.current
	var drive: float = Nonlinear.apply("spike", clampf(f.movement + 0.5 * f.energy, 0.0, 1.0), 2.5)
	if _rng.randf() < 0.012 + 0.05 * drive:        # ~1.2% baseline, up to ~6% on a strong moment
		_burst_left = _rng.randi_range(2, 3)       # short - normalizes after a couple of cuts
		_burst_min = _rng.randf_range(1.0, 2.0)
		_burst_max = _rng.randf_range(3.0, 5.0)
		_flurry_cd = _rng.randi_range(14, 26)      # then a long stretch of normal pacing
		print("ghost: BURST x%d  (%.1f-%.1fs cuts)" % [_burst_left, _burst_min, _burst_max])


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
	var fade := maxf(0.5, _bookend_time)
	var t := Spectrum.current.time
	var a_in := clampf(t / fade, 0.0, 1.0)            # 0 at the very start -> 1 after `fade`
	var a_out := clampf((slen - t) / fade, 0.0, 1.0)  # 1 -> 0 over the final `fade` seconds
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


# The SCHEDULE, advanced by the REAL music-clock step: the hold clock, transition progress + alphas,
# the smoothed audio level, the stinger, and arming the next cut. This decides WHEN things happen, so
# it must track the music exactly (never the capped animation step) or scene durations drift long.
func _tick_schedule(dt: float) -> void:
	var bf := _bookend_fade()                       # 1, except fading from/to black at the video's ends
	if _transitioning:
		var dur: float = layer_time if _style == Style.LAYER else transition_time
		_trans_t += dt / maxf(0.01, dur)
		var k := clampf(_trans_t, 0.0, 1.0)
		# Alphas are sequenced so the picture is clean (a DIP never shows both scenes at once).
		var a := _transition_alphas(k)
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
	_drive_stinger(dt, bf)                          # rapid-fire beat-synced modulation of THIS scene
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
			_sting = 1.0
			_sting_zoom = _rng.randf_range(0.07, 0.20) * (1.0 if _rng.randf() < 0.5 else -1.0)
			_sting_rot = _rng.randf_range(-0.09, 0.09)
			_sting_skew = _rng.randf_range(-0.06, 0.06)
			_sting_flash = _rng.randf_range(0.20, 0.5)
		elif _flurry_cd == 0 and _audio_ema >= 0.2:  # else maybe begin a run (rare, harmonic-gated)
			var drive: float = Nonlinear.apply("spike", clampf(f.movement + 0.5 * f.energy, 0.0, 1.0), 2.5)
			if _rng.randf() < 0.02 + 0.07 * drive:
				_sting_left = _rng.randi_range(2, 4)  # BANG, BANG (, BANG)
				_flurry_cd = _rng.randi_range(14, 26)
				print("ghost: STINGER x%d" % _sting_left)
	_sting = maxf(0.0, _sting - delta * 6.0)         # quick decay between beats
	var p := _sting * _sting                         # eased punch (snappy attack, soft tail)
	_current.view.pulse_zoom = 1.0 + _sting_zoom * p
	_current.view.pulse_rot = _sting_rot * p
	_current.view.pulse_skew = _sting_skew * p
	# Brightness/tint flash via the node modulate, preserving the fade alpha.
	var fl := 1.0 + _sting_flash * p
	_current.modulate = Color(fl * (1.0 + 0.06 * _sting_rot), fl, fl * (1.0 - 0.06 * _sting_rot), bf)


func _should_change() -> bool:
	if _locked >= 0 or _held:
		return false
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
	# In manual mode, a non-looping storyboard holds its final scene forever.
	if not _storyboard_seq.is_empty() and not _storyboard_loop and _step >= _storyboard_seq.size():
		return false
	var ex: Dictionary = _current.exit_spec
	# A fixed hold (deterministic storyboard timing) ignores cues entirely.
	if ex.has("hold"):
		return _elapsed >= float(ex["hold"])
	# During a burst the backstop shrinks to a few seconds so quick scenes never linger.
	var hi: float = _burst_max if _burst_left > 0 else _scaled_bound(ex, "max", max_hold)
	if _elapsed >= hi:                               # backstop: the cue never came
		return true
	if not _ready_to_exit(ex):
		return false
	return _trigger_fires(ex)


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


# A hold bound: a storyboard-explicit value is taken literally; the auto-mode default is pace-scaled.
func _scaled_bound(ex: Dictionary, key: String, base: float) -> float:
	return float(ex[key]) if ex.has(key) else base * _pacing_scale()


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
	if SCENES.size() < 2:
		_elapsed = 0.0
		return
	var burst_cut := _burst_left > 0      # leaving a burst scene -> a hard jump cut, no morph/blend
	if _burst_left > 0:
		_burst_left -= 1                  # consume this quick scene
	# Clear any rapid-fire modulation so the leaving scene doesn't freeze mid-contortion or tint.
	_sting_left = 0
	_sting = 0.0
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
		_host.add_child(nxt)
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
		_host.add_child(nxt)          # instant swap, no blend
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
	_next = nxt
	_host.add_child(_next)            # added last -> drawn over _current
	_transitioning = true
	_trans_t = 0.0


# The transition style for leaving the current scene: its storyboard-set style
# (cut/dip/fade), or the auto-mode weighted bag (mostly dip) when unspecified.
func _choose_style() -> int:
	match (_current.transition_style if _current != null else ""):
		"cut": return Style.CUT
		"dip": return Style.DIP
		"fade": return Style.FADE
	return STYLE_BAG[_biased(STYLE_BAG.size())]


func _finish_transition() -> void:
	_current.queue_free()
	_current = _next
	_next = null
	_transitioning = false
	_swaps += 1
	_current.modulate.a = 1.0
	_current.view.presence = 1.0
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
		var h := hash([_session_seed, _swaps, _scene_key(i)])
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


# Selection weight for one catalogue entry: 0 for the entry on screen (never an
# immediate repeat), tiny for another behavior of the *same* scene (so we don't
# show two of one kind back to back), and otherwise rising with the number of
# swaps since that kind was last seen - long-unseen kinds dominate the draw.
func _novelty_weight(i: int) -> float:
	if i == _index:
		return 0.0
	var kind := String(SCENES[i].script.resource_path)
	if _index >= 0 and kind == String(SCENES[_index].script.resource_path):
		return 0.05
	var age := float(_swaps - int(_kind_last.get(kind, -1000)))
	return pow(maxf(1.0, age), 1.6)


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
		var seed := _session_seed ^ _scene_key(_index) ^ (_swaps * 0x85EBCA77)
		return {"script": e.script, "behavior": e.behavior, "seed": seed,
			"shot": "", "exit_spec": {}, "transition": ""}   # "" -> auto STYLE_BAG
	# Manual: walk the sequence (wrap when looping, else hold on the last entry).
	var n := _storyboard_seq.size()
	var i: int = _step % n if _storyboard_loop else mini(_step, n - 1)
	_step += 1
	var item: Dictionary = _storyboard_seq[i]
	var nm := String(item.get("scene", ""))
	var path := "res://scripts/scenes/%s.gd" % nm
	var script: Resource = load(path) if ResourceLoader.exists(path) else SCENES[0].script
	if not ResourceLoader.exists(path):
		push_warning("ghost: storyboard scene '%s' not found, substituting" % nm)
	var seed2: int = int(item.get("seed",
		_session_seed ^ (i * 0x9E3779B1) ^ (_step * 0x85EBCA77)))
	return {"script": script, "behavior": String(item.get("behavior", "drift")),
		"seed": seed2, "shot": String(item.get("shot", "")), "exit_spec": _parse_exit(item),
		"transition": String(item.get("transition", ""))}   # entry-level only; resolved in _make_scene


# Translate a storyboard entry's timing into an exit_spec the scene carries (see
# _should_change): a fixed `hold`, a musical `exit` trigger, or just min/max bounds.
func _parse_exit(item: Dictionary) -> Dictionary:
	if item.has("hold"):
		return {"hold": float(item["hold"])}
	if item.has("exit"):
		return {
			"trigger": _trigger_from_name(String(item["exit"])),
			"min": float(item.get("min_hold", min_hold)),
			"max": float(item.get("max_hold", max_hold))}
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
func _make_scene() -> GhostScene:
	var entry := _next_entry()
	var script: Resource = entry["script"]
	var scene: GhostScene = script.new()
	var seed: int = int(entry["seed"])
	scene.init_with_seed(seed, String(entry["behavior"]))
	scene.scene_name = String(script.resource_path).get_file().get_basename()
	# Telemetry only: the live harmonic bucket at the cut. It is NO LONGER folded into the scene
	# seed (that broke run-to-run reproducibility); the seed is deterministic per song. Same music
	# should still print the same bucket here, a useful observability signal. (See
	# next/harmonic_seeding.md.)
	print("ghost: cut -> %s  harmonic bucket %d" % [scene.scene_name, Spectrum.harmonic_bucket(12)])
	scene.exit_spec = entry["exit_spec"]
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
		shot_name = bag[_rng.randi() % bag.size()]
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


