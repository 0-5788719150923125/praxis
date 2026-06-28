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

# Each entry pairs a scene script with a motion behavior (see VortexScene). The
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
	# "the-point" scenes (camera holds, per the brief).
	{"script": preload("res://scripts/scenes/eye.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/two_eyes.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/prism.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/prism_split.gd"), "behavior": "static"},
]

enum Style { CUT, DIP, FADE }
enum Trigger { BEAT, MOVEMENT, LULL }

# How a change is performed, weighted. Mostly a DIP to black - the old scene fades
# out, a beat of true darkness, then the new one fades up - so the eye gets a clean
# gap between scenes and two scenes never overlap (the old crossfades read as
# clipping). The occasional hard CUT keeps things punchy; a plain crossfade is rare.
const STYLE_BAG := [Style.DIP, Style.DIP, Style.DIP, Style.DIP, Style.DIP, Style.CUT, Style.FADE]
# Exit cues, weighted: usually land the cut on a beat; sometimes on a section
# change (movement) or a drop into quiet (lull).
const TRIGGER_BAG := [Trigger.BEAT, Trigger.BEAT, Trigger.BEAT, Trigger.MOVEMENT, Trigger.MOVEMENT, Trigger.LULL]

## Don't exit a looping scene before this (seconds) - keeps cuts from thrashing.
@export var min_hold: float = 7.0
## Exit at least this often even if the chosen cue never arrives (seconds).
@export var max_hold: float = 28.0
## Movement score (0..1) that satisfies a MOVEMENT trigger.
@export var movement_threshold: float = 0.6
## Energy (0..1) at or below which a LULL trigger fires.
@export var lull_threshold: float = 0.12
## Seconds a dip/blend takes end to end (cuts are instant). A DIP spends the middle
## of this in darkness, so a little long reads as a deliberate breath between scenes.
@export var transition_time: float = 2.0

var _host: Node = null
var _current: VortexScene = null
var _next: VortexScene = null
var _index := -1
var _elapsed := 0.0

var _transitioning := false
var _trans_t := 0.0
var _style: Style = Style.CUT
var _trigger: Trigger = Trigger.BEAT
var _beat_prev := 0.0
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
	_rng.seed = _session_seed ^ 0x1234567
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


func _resolve_seed() -> int:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if args[i] == "--seed" and i + 1 < args.size() and args[i + 1].is_valid_int():
			return int(args[i + 1])
	var r := RandomNumberGenerator.new()
	r.randomize()
	return r.randi()


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
		push_warning("vortex: storyboard not found: %s" % path)
		return false
	var text := FileAccess.get_file_as_string(path)
	var data: Variant = JSON.parse_string(text)
	if typeof(data) != TYPE_DICTIONARY or not data.has("sequence"):
		push_warning("vortex: storyboard %s has no 'sequence' array" % path)
		return false
	var seq: Variant = data["sequence"]
	if typeof(seq) != TYPE_ARRAY or (seq as Array).is_empty():
		push_warning("vortex: storyboard %s sequence is empty" % path)
		return false
	_storyboard_seq = seq
	_storyboard_loop = bool(data.get("loop", true))
	_storyboard_name = String(data.get("name", name_or_path))
	_storyboard_transition = String(data.get("transition", ""))   # e.g. "cut" forces jump cuts
	_step = 0
	print("vortex: storyboard '%s' loaded (%d scenes, loop=%s)" % [
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
func _arm() -> void:
	_elapsed = 0.0
	_beat_prev = Spectrum.current.beat
	_trigger = TRIGGER_BAG[_rng.randi() % TRIGGER_BAG.size()]


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


func _process(delta: float) -> void:
	if _current == null:
		return

	if _transitioning:
		_trans_t += delta / maxf(0.01, transition_time)
		var k := clampf(_trans_t, 0.0, 1.0)
		_current.update(Spectrum.current, delta)
		_next.update(Spectrum.current, delta)
		# Both scenes keep animating; their alphas are sequenced so the picture is
		# clean (a DIP never shows both at once).
		var a := _transition_alphas(k)
		_current.modulate.a = a.x
		_current.view.presence = a.x
		_next.modulate.a = a.y
		_next.view.presence = a.y
		_current.view.commit(delta)
		_next.view.commit(delta)
		if k >= 1.0:
			_finish_transition()
		return

	_current.update(Spectrum.current, delta)
	_current.view.commit(delta)
	_elapsed += delta
	if _should_change():
		_begin_transition()
	_beat_prev = Spectrum.current.beat


func _should_change() -> bool:
	if _locked >= 0 or _held:
		return false
	# In manual mode, a non-looping storyboard holds its final scene forever.
	if not _storyboard_seq.is_empty() and not _storyboard_loop and _step >= _storyboard_seq.size():
		return false
	var ex: Dictionary = _current.exit_spec
	# A fixed hold (deterministic storyboard timing) ignores cues entirely.
	if ex.has("hold"):
		return _elapsed >= float(ex["hold"])
	if _elapsed >= float(ex.get("max", max_hold)):   # backstop: the cue never came
		return true
	if not _ready_to_exit(ex):
		return false
	return _trigger_fires(ex)


# Eligibility: a oneshot when its sequence ends, a loop after the minimum hold.
func _ready_to_exit(ex: Dictionary) -> bool:
	if _current.lifecycle == "oneshot":
		return _current.finished()
	return _elapsed >= float(ex.get("min", min_hold))


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
		_:
			return Vector2(1.0 - k, k)


func _begin_transition() -> void:
	if SCENES.size() < 2:
		_elapsed = 0.0
		return
	var nxt := _make_scene()

	# Content-aware morph: if the incoming can grow out of the outgoing's geometry,
	# swap instantly and let it animate the morph (e.g. one eye splitting into two).
	# Only ever between compatible, non-empty types - so we never morph a mismatch.
	if _current != null and not nxt.morph_in.is_empty() and nxt.morph_in == _current.morph_out:
		print("vortex: morph %s -> %s (%s)" % [_current.scene_name, nxt.scene_name, nxt.morph_in])
		var from := _current
		_host.add_child(nxt)
		_current = nxt
		_swaps += 1
		nxt.begin_morph(from)         # hand over state BEFORE the source is freed
		from.queue_free()
		_arm()
		return

	_style = _choose_style()
	if _style == Style.CUT:
		_host.add_child(nxt)          # instant swap, no blend
		_current.queue_free()
		_current = nxt
		_swaps += 1
		_arm()
		return

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
	return STYLE_BAG[_rng.randi() % STYLE_BAG.size()]


func _finish_transition() -> void:
	_current.queue_free()
	_current = _next
	_next = null
	_transitioning = false
	_swaps += 1
	_current.modulate.a = 1.0
	_current.view.presence = 1.0
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
	var weights := []
	var total := 0.0
	for i in SCENES.size():
		var w := _novelty_weight(i)
		weights.append(w)
		total += w
	if total <= 0.0:                      # degenerate (everything suppressed): uniform
		return _rng.randi() % SCENES.size()
	var r := _rng.randf() * total
	var acc := 0.0
	for i in SCENES.size():
		acc += weights[i]
		if r <= acc:
			return i
	return SCENES.size() - 1


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
		var seed := _session_seed ^ (_index * 0x9E3779B1) ^ (_swaps * 0x85EBCA77)
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
		push_warning("vortex: storyboard scene '%s' not found, substituting" % nm)
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
func _make_scene() -> VortexScene:
	var entry := _next_entry()
	var script: Resource = entry["script"]
	var scene: VortexScene = script.new()
	var seed: int = int(entry["seed"])
	scene.init_with_seed(seed, String(entry["behavior"]))
	scene.scene_name = String(script.resource_path).get_file().get_basename()
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
	return scene


func _exit_tree() -> void:
	# Free live scenes so they don't report as leaked when the app quits.
	if is_instance_valid(_current):
		_current.queue_free()
	if is_instance_valid(_next):
		_next.queue_free()


