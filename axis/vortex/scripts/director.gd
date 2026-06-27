extends Node

## Director - the scene registry and scheduler (autoload).
##
## Holds every visualizer, picks one, seeds it from the song hash plus an index
## (so a given song always yields the same sequence of scenes), and swaps to the
## next on a timer - looping forever, the way a music video should. Determinism
## is the whole point: same song in, same video out.

## The scene registry. Add a new visualizer by dropping its script in
## scripts/scenes/ and preloading it here.
const SCENES := [
	preload("res://scripts/scenes/spectrum_ring.gd"),
	preload("res://scripts/scenes/harmonic_lattice.gd"),
]

enum Order { SEQUENTIAL, RANDOM }

## How the next scene is chosen.
@export var order: Order = Order.SEQUENTIAL
## Seconds each scene holds before advancing.
@export var scene_duration: float = 18.0

var _host: Node = null            # node scenes are parented under (main)
var _active: VortexScene = null
var _index := -1
var _elapsed := 0.0


## main calls this once to hand Director the node to attach scenes to.
func attach(host: Node) -> void:
	_host = host
	_advance()


func _process(delta: float) -> void:
	if _active == null:
		return
	_active.update(Spectrum.current, delta)
	_elapsed += delta
	if _elapsed >= scene_duration:
		_advance()


## Jump to the next scene now (also bound to Space in main).
func next() -> void:
	_advance()


func _advance() -> void:
	if _host == null or SCENES.is_empty():
		return
	_elapsed = 0.0

	if order == Order.RANDOM and SCENES.size() > 1:
		var pick := _index
		while pick == _index:
			pick = randi() % SCENES.size()
		_index = pick
	else:
		_index = (_index + 1) % SCENES.size()

	if _active != null:
		_active.queue_free()

	var scene: VortexScene = SCENES[_index].new()
	# Seed from the song so the same track always renders identically; the index
	# keeps repeated visits to a scene distinct within one run.
	scene.init_with_seed(Spectrum.song_hash ^ (_index * 0x9E3779B1))
	_host.add_child(scene)
	_active = scene
