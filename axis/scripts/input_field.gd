# input_field.gd
extends TextEdit

func _input(event: InputEvent):
	if has_focus():  # Only check if we actually have focus
		if event is InputEventMouseButton and event.is_pressed() and event.button_index == MOUSE_BUTTON_LEFT:
			var evLocal = make_input_local(event)
			if !Rect2(Vector2(), size).has_point(evLocal.position):
				release_focus()
				get_viewport().set_input_as_handled()
		elif event is InputEventScreenTouch and event.is_pressed():
			# Also handle touch events for mobile
			var evLocal = make_input_local(event)
			if !Rect2(Vector2(), size).has_point(evLocal.position):
				release_focus()
				get_viewport().set_input_as_handled()
