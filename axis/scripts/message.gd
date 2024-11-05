# message.gd
extends PanelContainer

@onready var label = $MarginContainer/Label
@onready var margin_container = $MarginContainer

func set_message(text: String, is_user: bool):
	label.text = text
	
	# Enable text wrapping
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.custom_minimum_size.x = 100  # Minimum width to prevent too narrow wrapping
	
	if is_user:
		self_modulate = Color("e3f2fd")  # Light blue for user
		# Align to right
		size_flags_horizontal = Control.SIZE_SHRINK_END  # Makes container only as wide as needed
		custom_minimum_size.x = min(get_viewport().size.x * 0.8, 600)  # Max width of 80% of viewport or 600px
		
		# Right-align the text itself
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		
		# Optionally, adjust margins for better visual alignment
		margin_container.add_theme_constant_override("margin_left", 20)
		margin_container.add_theme_constant_override("margin_right", 10)
	else:
		self_modulate = Color("f5f5f5")  # Light grey for assistant
		# Align to left
		size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
		custom_minimum_size.x = min(get_viewport().size.x * 0.8, 600)  # Max width of 80% of viewport or 600px
		
		# Left-align the text
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		
		# Reset margins to default
		margin_container.add_theme_constant_override("margin_left", 10)
		margin_container.add_theme_constant_override("margin_right", 20)
