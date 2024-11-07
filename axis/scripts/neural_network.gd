extends Node3D

var atom_scene = preload("res://scenes/atom.tscn")
@onready var atoms_container = $Atoms

func _ready() -> void:
	print("Neural Network initializing...")  # Debug print
	
	# Create initial central atom
	var central_atom = atom_scene.instantiate()
	atoms_container.add_child(central_atom)
	
	# Set its position explicitly
	central_atom.global_position = Vector3.ZERO
	
	# Make it slightly larger and highlighted
	central_atom.set_radius(0.75)
	central_atom.set_highlight(true)
	
	print("Central atom created at: ", central_atom.global_position)  # Debug print

func _process(_delta: float) -> void:
	pass

func add_atom(position: Vector3) -> void:
	var new_atom = atom_scene.instantiate()
	atoms_container.add_child(new_atom)
	new_atom.global_position = position
