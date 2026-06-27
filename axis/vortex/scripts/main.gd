extends Node2D

## vortex entry point.
##
## Almost nothing lives here on purpose: the audio is owned by [Spectrum] and the
## visuals by [Director]. main just hands Director a node to draw into and maps
## the keys. Scenes are added as children of this node.

func _ready() -> void:
	Director.attach(self)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_F11:
				_toggle_fullscreen()
			KEY_SPACE:
				Director.next()
			KEY_ESCAPE:
				get_tree().quit()


func _toggle_fullscreen() -> void:
	var mode := DisplayServer.window_get_mode()
	if mode == DisplayServer.WINDOW_MODE_FULLSCREEN:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
	else:
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
