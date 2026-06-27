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
	{"script": preload("res://scripts/scenes/fog_lights.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/strata.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/koch_snowflake.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/wire_solid.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/voxel_blocks.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/cityscape.gd"), "behavior": "static"},
	{"script": preload("res://scripts/scenes/shatter_glass.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/gaussian_landscape.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/rocks.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/embers.gd"), "behavior": "drift"},
	{"script": preload("res://scripts/scenes/metropolis.gd"), "behavior": "drift"},
]

enum Style { CUT, FADE, ZOOM, BLEED }
enum Trigger { BEAT, MOVEMENT, LULL }

# Mostly cuts, sometimes a blend - weighted by how often each appears here.
const STYLE_BAG := [Style.CUT, Style.CUT, Style.CUT, Style.CUT, Style.FADE, Style.ZOOM, Style.BLEED]
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
## Seconds a blend takes (cuts are instant).
@export var transition_time: float = 1.5

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


func attach(host: Node) -> void:
	_host = host
	_rng.seed = Spectrum.song_hash ^ 0x1234567
	_locked = _locked_scene_arg()
	_current = _make_scene()
	_host.add_child(_current)
	_arm()


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
		_apply_transition(_current, 1.0 - k, false)
		_apply_transition(_next, k, true)
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
	if _locked >= 0:
		return false
	if _elapsed >= max_hold:        # backstop: the cue never came
		return true
	if not _ready_to_exit():
		return false
	return _trigger_fires()


# Eligibility: a oneshot when its sequence ends, a loop after the minimum hold.
func _ready_to_exit() -> bool:
	if _current.lifecycle == "oneshot":
		return _current.finished()
	return _elapsed >= min_hold


# Has the armed spectral cue arrived this frame?
func _trigger_fires() -> bool:
	var f := Spectrum.current
	match _trigger:
		Trigger.BEAT:
			return f.beat > 0.5 and _beat_prev <= 0.5    # rising edge of a beat
		Trigger.MOVEMENT:
			return f.movement >= movement_threshold
		Trigger.LULL:
			return f.energy <= lull_threshold
	return false


## Force the next change now (bound to Space in main).
func next() -> void:
	if not _transitioning:
		_begin_transition()


func _begin_transition() -> void:
	if SCENES.size() < 2:
		_elapsed = 0.0
		return
	_next = _make_scene()
	_host.add_child(_next)            # added last -> drawn over _current
	_style = STYLE_BAG[_rng.randi() % STYLE_BAG.size()]

	if _style == Style.CUT:
		_current.queue_free()         # instant swap, no blend
		_current = _next
		_next = null
		_swaps += 1
		_arm()
		return

	if _style == Style.BLEED:
		_set_additive(_current, true)
		_set_additive(_next, true)
	_transitioning = true
	_trans_t = 0.0


func _finish_transition() -> void:
	_current.queue_free()
	_current = _next
	_next = null
	_transitioning = false
	_swaps += 1
	_set_additive(_current, false)
	_current.modulate.a = 1.0
	_current.view.presence = 1.0
	_arm()


# Instantiate + seed the next scene with its paired behavior.
func _make_scene() -> VortexScene:
	if _locked >= 0:
		_index = _locked
	elif SCENES.size() > 1:
		var pick := _index
		while pick == _index:
			pick = _rng.randi() % SCENES.size()
		_index = pick
	else:
		_index = 0
	var entry: Dictionary = SCENES[_index]
	var scene: VortexScene = entry.script.new()
	var seed := Spectrum.song_hash ^ (_index * 0x9E3779B1) ^ (_swaps * 0x85EBCA77)
	scene.init_with_seed(seed, entry.behavior)
	# Assign a camera framing by the scene's framing class: expressive for discrete
	# subjects, gentle for field-fillers, square-on for single flat planes.
	var bag: Array = Shots.SUBJECT_BAG
	if scene.framing == "field":
		bag = Shots.FIELD_BAG
	elif scene.framing == "plane":
		bag = Shots.PLANE_BAG
	scene.set_shot(Shots.make(bag[_rng.randi() % bag.size()], seed ^ 0x51ED2701))
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


# Layer blend alpha + (for blends) a little transition motion onto a scene.
func _apply_transition(scene: VortexScene, present: float, incoming: bool) -> void:
	var p := clampf(present, 0.0, 1.0)
	var a := smoothstep(0.0, 1.0, p)
	scene.view.presence = p
	match _style:
		Style.ZOOM:
			scene.view.zoom *= lerpf(0.6, 1.0, p)
		Style.BLEED:
			a = p
		_:
			pass
	scene.modulate.a = a


func _set_additive(scene: VortexScene, on: bool) -> void:
	if on:
		var m := CanvasItemMaterial.new()
		m.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
		scene.material = m
	else:
		scene.material = null
