extends PanelContainer

var label: Control
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
	var font_size = 32 if is_mobile else 16
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
	
	# Set up size flags for proper horizontal expansion
	size_flags_horizontal = Control.SIZE_SHRINK_BEGIN if not is_user else Control.SIZE_SHRINK_END
	
	# Important: Set the label to expand horizontally
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	
	if is_user:
		# User message styling
		self_modulate = Color("424242")  # Dark grey background
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		margin_container.add_theme_constant_override("margin_left", base_margin * 2)
		margin_container.add_theme_constant_override("margin_right", base_margin)
		add_theme_stylebox_override("panel", create_styled_panel(Color(0.825, 0.188, 0.22), 4))
	else:
		# Assistant message styling
		self_modulate = Color("424242")  # Dark grey background
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		margin_container.add_theme_constant_override("margin_left", base_margin)
		margin_container.add_theme_constant_override("margin_right", base_margin * 2)
		add_theme_stylebox_override("panel", create_styled_panel(Color(0.187, 0.416, 1), 4))
	
	# Set the label's text color to white for better contrast
	label.add_theme_color_override("font_color", Color.WHITE)
	
	# Calculate initial width based on text content
	await get_tree().process_frame
	var text_size = label.get_theme_font("font").get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size)
	
	# Set the width to accommodate the text, but not exceed max_width
	var needed_width = min(text_size.x + base_margin * 4, max_width)
	custom_minimum_size.x = needed_width
	
	# Ensure the label has enough width to display text properly
	label.custom_minimum_size.x = needed_width - (base_margin * 4)
	
	# Set mouse_filter to MOUSE_FILTER_IGNORE
	self.mouse_filter = Control.MOUSE_FILTER_IGNORE
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	margin_container.mouse_filter = Control.MOUSE_FILTER_IGNORE

func create_styled_panel(border_color: Color, border_width: int) -> StyleBoxFlat:
	var style = StyleBoxFlat.new()
	style.bg_color = self_modulate  # Dark grey background
	style.border_color = border_color
	style.border_width_left = border_width
	style.border_width_right = border_width
	style.border_width_top = border_width
	style.border_width_bottom = border_width
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	# Add some padding to the stylebox
	style.content_margin_left = 12
	style.content_margin_right = 12
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	return style

func _on_glitch_completed():
	# Handle any post-glitch animations or effects here
	pass
