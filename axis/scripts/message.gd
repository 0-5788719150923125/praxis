extends PanelContainer

var label: Control # Can be either Label or GlitchedLabel
@onready var margin_container = $MarginContainer

func set_message(text: String, is_user: bool):
	# Create appropriate label type
	if is_user:
		label = Label.new()
	else:
		label = GlitchedLabel.new()
		label.glitch_completed.connect(_on_glitch_completed)
	
	margin_container.add_child(label)
	
	# Inherit theme from parent
	label.theme = get_parent().theme
	
	# Apply font size explicitly
	var is_mobile = OS.has_feature("mobile")
	var font_size = 32 if is_mobile else 16  # Match the MOBILE_FONT_SIZE and DESKTOP_FONT_SIZE constants
	label.add_theme_font_size_override("font_size", font_size)
	
	# Set initial properties
	if is_user:
		label.text = text
	else:
		label.text = ""
		label.write(text)
	
	# Enable text wrapping
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	
	# Adjust base sizing for mobile
	var base_margin = 20 if is_mobile else 10
	var max_width = get_viewport().size.x * (0.9 if is_mobile else 0.8)
	max_width = min(max_width, 800 if is_mobile else 600)
	
	if is_user:
		self_modulate = Color("e3f2fd")
		size_flags_horizontal = Control.SIZE_SHRINK_END
		custom_minimum_size.x = max_width
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		margin_container.add_theme_constant_override("margin_left", base_margin * 2)
		margin_container.add_theme_constant_override("margin_right", base_margin)
	else:
		self_modulate = Color("f5f5f5")
		size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
		custom_minimum_size.x = max_width
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		margin_container.add_theme_constant_override("margin_left", base_margin)
		margin_container.add_theme_constant_override("margin_right", base_margin * 2)
	
	# Set minimum width to prevent overly narrow messages
	label.custom_minimum_size.x = 150 if is_mobile else 100

func _on_glitch_completed():
	# Handle any post-glitch animations or effects here
	pass
