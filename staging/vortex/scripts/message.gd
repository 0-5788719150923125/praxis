extends PanelContainer

var label: Control
var final_text: String = ""
var is_user_message: bool = false
@onready var margin_container = $MarginContainer

func set_message(text: String, is_user: bool):
	final_text = text
	is_user_message = is_user
	
	# Create appropriate label type
	if is_user:
		label = Label.new()
	else:
		label = GlitchedLabel.new()
		label.glitch_completed.connect(_on_glitch_completed)
		# Connect to the text changed signal
		label.text_changed.connect(_on_text_changed)
	
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
		_update_container_size(text)  # Set final size immediately for user messages
	else:
		label.text = ""
		label.write(text)
		_update_container_size("")  # Start with minimum size for AI messages
	
	# Enable text wrapping
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	
	# Set up size flags for proper horizontal expansion
	size_flags_horizontal = Control.SIZE_SHRINK_BEGIN if not is_user else Control.SIZE_SHRINK_END
	
	# Important: Set the label to expand horizontally
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	
	var base_margin = 20 if is_mobile else 10
	
	if is_user:
		# User message styling
		self_modulate = Color("424242")
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		margin_container.add_theme_constant_override("margin_left", base_margin * 2)
		margin_container.add_theme_constant_override("margin_right", base_margin)
		add_theme_stylebox_override("panel", create_styled_panel(Color(0.825, 0.188, 0.22), 4))
	else:
		# Assistant message styling
		self_modulate = Color("424242")
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		margin_container.add_theme_constant_override("margin_left", base_margin)
		margin_container.add_theme_constant_override("margin_right", base_margin * 2)
		add_theme_stylebox_override("panel", create_styled_panel(Color(0.187, 0.416, 1), 4))
	
	# Set the label's text color to white for better contrast
	label.add_theme_color_override("font_color", Color.WHITE)
	
	# Set mouse_filter to MOUSE_FILTER_IGNORE
	self.mouse_filter = Control.MOUSE_FILTER_IGNORE
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	margin_container.mouse_filter = Control.MOUSE_FILTER_IGNORE

func _update_container_size(current_text: String):
	var is_mobile = OS.has_feature("mobile")
	var base_margin = 20 if is_mobile else 10
	var max_width = get_viewport().size.x * (0.9 if is_mobile else 0.8)
	max_width = min(max_width, 800 if is_mobile else 600)
	var font_size = 32 if is_mobile else 16
	
	# Calculate size based on current visible text
	var text_size = label.get_theme_font("font").get_string_size(
		current_text, 
		HORIZONTAL_ALIGNMENT_LEFT, 
		-1, 
		font_size
	)
	
	# Add padding for margins and borders
	var needed_width = min(text_size.x + base_margin * 4, max_width)
	
	# Set minimum width to accommodate current text
	custom_minimum_size.x = needed_width
	
	# Update label width
	label.custom_minimum_size.x = needed_width - (base_margin * 4)

func _on_text_changed():
	if not is_user_message:
		_update_container_size(label.text)

func create_styled_panel(border_color: Color, border_width: int) -> StyleBoxFlat:
	var style = StyleBoxFlat.new()
	style.bg_color = self_modulate
	style.border_color = border_color
	style.border_width_left = border_width
	style.border_width_right = border_width
	style.border_width_top = border_width
	style.border_width_bottom = border_width
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	style.corner_radius_bottom_left = 8
	style.corner_radius_bottom_right = 8
	style.content_margin_left = 12
	style.content_margin_right = 12
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	return style

func _on_glitch_completed():
	# Ensure final size is set correctly
	_update_container_size(final_text)
