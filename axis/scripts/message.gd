extends PanelContainer

@onready var label = $MarginContainer/Label
@onready var margin_container = $MarginContainer

func set_message(text: String, is_user: bool):
	label.text = text
	
	# Enable text wrapping
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	
	# Adjust base sizing for mobile
	var is_mobile = OS.has_feature("mobile")
	var base_margin = 20 if is_mobile else 10
	var max_width = get_viewport().size.x * (0.9 if is_mobile else 0.8)
	max_width = min(max_width, 800 if is_mobile else 600)  # Larger max width on mobile
	
	if is_user:
		self_modulate = Color("e3f2fd")
		size_flags_horizontal = Control.SIZE_SHRINK_END
		custom_minimum_size.x = max_width
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		
		# Adjusted margins for mobile
		margin_container.add_theme_constant_override("margin_left", base_margin * 2)
		margin_container.add_theme_constant_override("margin_right", base_margin)
	else:
		self_modulate = Color("f5f5f5")
		size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
		custom_minimum_size.x = max_width
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		
		# Adjusted margins for mobile
		margin_container.add_theme_constant_override("margin_left", base_margin)
		margin_container.add_theme_constant_override("margin_right", base_margin * 2)
	
	# Set minimum width to prevent overly narrow messages
	label.custom_minimum_size.x = 150 if is_mobile else 100
