extends CanvasLayer

var message_scene = preload("res://scenes/message.tscn")
var prompt_manager = PromptManager.new()
var initial_viewport_height: float = 0.0

# Variables to store layout values
var base_button_height: int
var base_input_margin: int

# UI scaling constants
const MIN_BUTTON_HEIGHT = 70
const MIN_INPUT_HEIGHT = 70
const MOBILE_FONT_SIZE = 32
const DESKTOP_FONT_SIZE = 16

@onready var ui_root = $UIRoot
@onready var scroll_container = $UIRoot/ScrollContainer
@onready var message_container = $UIRoot/ScrollContainer/MessageContainer
@onready var input_field = $UIRoot/InputContainer/TextEdit
@onready var clear_button = $UIRoot/InputContainer/ClearButton
@onready var http_request = $UIRoot/HTTPRequest
@onready var background_touch = $BackgroundTouch
@onready var input_container = $UIRoot/InputContainer
@onready var toggle_button = $ToggleButton

const KEYBOARD_OFFSET = 1000
const INPUT_MARGIN = 10

var server_url = "http://192.168.5.94:2100"

func _ready():
	var config = ConfigFile.new()
	if config.load("user://settings.cfg") == OK:
		server_url = config.get_value("Server", "url", "http://192.168.5.94:2100")
		print("Server URL: ", server_url)
	_initialize_components()
	_create_base_theme()
	_setup_signals()
	_setup_layout()
	_apply_platform_scaling()
	hide_chat_interface()

var last_keyboard_height = 0
func _process(_delta: float) -> void:
	if OS.has_feature("mobile") and ui_root.visible:
		var keyboard_height = DisplayServer.virtual_keyboard_get_height()
		if keyboard_height != last_keyboard_height:
			_update_input_position(keyboard_height)
		last_keyboard_height = keyboard_height

func _update_input_position(keyboard_height: float):
	var offset_amount = keyboard_height + 50
	_update_layout_positions(base_button_height, base_input_margin, offset_amount)

func _initialize_components():
	initial_viewport_height = get_viewport().get_visible_rect().size.y
	input_field.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	input_field.scroll_fit_content_height = true
	background_touch.mouse_filter = Control.MOUSE_FILTER_IGNORE

func _setup_signals():
	http_request.request_completed.connect(_on_request_completed)
	get_tree().root.size_changed.connect(_on_window_resize)
	input_field.gui_input.connect(_on_input_gui_input)
	clear_button.pressed.connect(_on_clear_button_pressed)
	toggle_button.pressed.connect(_on_toggle_button_pressed)
	scroll_container.gui_input.connect(_on_scroll_container_gui_input)

func _setup_layout():
	var is_mobile = OS.has_feature("mobile")
	base_button_height = MIN_BUTTON_HEIGHT if is_mobile else 40
	base_input_margin = 20 if is_mobile else 10
	_update_layout_positions(base_button_height, base_input_margin)

func _create_base_theme():
	# Create a base theme that will be used throughout the interface
	var base_theme = Theme.new()
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	# Set default font size
	base_theme.set_default_font_size(font_size)
	
	# Store the theme for reuse
	ui_root.theme = base_theme
	
	# Ensure the theme propagates to all children
	for child in ui_root.get_children():
		if child is Control:
			child.theme = base_theme

func _apply_platform_scaling():
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	# Apply scaling to main UI elements
	if is_mobile:
		# Set minimum sizes for buttons and input
		toggle_button.custom_minimum_size.y = MIN_BUTTON_HEIGHT
		clear_button.custom_minimum_size.y = MIN_BUTTON_HEIGHT
		input_field.custom_minimum_size.y = MIN_INPUT_HEIGHT
		
		# Make buttons wider
		toggle_button.custom_minimum_size.x = 180
		clear_button.custom_minimum_size.x = 120
		
		# Increase spacing
		input_container.add_theme_constant_override("separation", 20)
	
	# Apply font sizes explicitly to ensure they take effect
	_apply_font_size_to_control(toggle_button, font_size)
	_apply_font_size_to_control(clear_button, font_size)
	_apply_font_size_to_control(input_field, font_size)
	
	# Scale existing messages
	_update_message_sizes()

func _apply_font_size_to_control(control: Control, size: int):
	# Create font settings
	control.add_theme_font_size_override("font_size", size)
	
	# Special handling for TextEdit
	if control is TextEdit:
		# Override additional font size properties specific to TextEdit
		control.add_theme_font_size_override("normal_font_size", size)
		control.add_theme_font_size_override("bold_font_size", size)
		control.add_theme_font_size_override("italic_font_size", size)
		control.add_theme_font_size_override("bold_italic_font_size", size)

func _update_message_sizes():
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	# Update all existing messages
	for message in message_container.get_children():
		message.theme = ui_root.theme
		var label = message.get_node_or_null("MarginContainer/Label")
		if label:
			_apply_font_size_to_control(label, font_size)

func _on_window_resize():
	_create_base_theme()
	_setup_layout()
	_apply_platform_scaling()

func _update_layout_positions(button_height: int, input_margin: int, keyboard_offset: int = 0):
	var is_mobile = OS.has_feature("mobile")
	
	# Adjust toggle button positioning
	toggle_button.offset_left = 20
	toggle_button.offset_top = -button_height - input_margin - keyboard_offset
	toggle_button.offset_right = 20 + (180 if is_mobile else 120)
	toggle_button.offset_bottom = -input_margin - keyboard_offset
	
	# Adjust input container positioning
	input_container.offset_left = toggle_button.offset_right + (20 if is_mobile else 10)
	input_container.offset_top = -button_height - input_margin - keyboard_offset
	input_container.offset_right = -input_margin
	input_container.offset_bottom = -input_margin - keyboard_offset
	
	# Adjust scroll container
	scroll_container.offset_bottom = -(button_height + input_margin * 2) - keyboard_offset
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	if is_mobile:
		# Set minimum sizes for buttons
		toggle_button.custom_minimum_size.y = MIN_BUTTON_HEIGHT
		clear_button.custom_minimum_size.y = MIN_BUTTON_HEIGHT
		input_field.custom_minimum_size.y = MIN_INPUT_HEIGHT
		
		# Make buttons wider
		toggle_button.custom_minimum_size.x = 180
		clear_button.custom_minimum_size.x = 120
		
		# Increase spacing
		input_container.add_theme_constant_override("separation", 20)
	
	# Create a theme for text scaling
	var theme = Theme.new()
	theme.set_default_font_size(font_size)
	
	# Apply theme to input field
	input_field.theme = theme
	
	# Update font sizes for all UI elements
	_update_font_size(toggle_button, font_size)
	_update_font_size(clear_button, font_size)
	_update_font_size(input_field, font_size)
	
	# Scale any additional text elements
	_scale_all_text_elements(ui_root, font_size)

func _scale_all_text_elements(node: Node, font_size: int):
	for child in node.get_children():
		if child is Control:
			_update_font_size(child, font_size)
		
		if child.get_child_count() > 0:
			_scale_all_text_elements(child, font_size)

func _update_font_size(control: Control, size: int):
	# Create a theme for consistent text scaling
	var theme = Theme.new()
	theme.set_default_font_size(size)
	
	if control is Button:
		control.add_theme_font_size_override("font_size", size)
		control.theme = theme
	elif control is TextEdit:
		control.add_theme_font_size_override("font_size", size)
		control.theme = theme
	elif control is Label:
		control.add_theme_font_size_override("font_size", size)
		control.theme = theme

func _on_scroll_container_gui_input(event):
	if event is InputEventMouseButton and event.is_pressed():
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			scroll_container.scroll_vertical -= 20
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			scroll_container.scroll_vertical += 20

func _on_toggle_button_pressed() -> void:
	if ui_root.visible:
		hide_chat_interface()
	else:
		show_chat_interface()

func show_chat_interface() -> void:
	ui_root.show()
	background_touch.show()
	toggle_button.text = "CLOSE"
	input_field.grab_focus()
	if OS.has_feature("mobile"):
		DisplayServer.virtual_keyboard_show("default")

func hide_chat_interface() -> void:
	ui_root.hide()
	background_touch.hide()
	toggle_button.text = "OPEN"
	if input_field.has_focus():
		input_field.release_focus()
	if OS.has_feature("mobile"):
		DisplayServer.virtual_keyboard_hide()

func clear_chat_history() -> void:
	# Clear all messages from the UI
	for child in message_container.get_children():
		child.queue_free()
	# Clear the prompt manager history
	prompt_manager.clear_history()
	# Clear the input field if it has any text
	input_field.text = ""

func _on_clear_button_pressed() -> void:
	clear_chat_history()
	input_field.grab_focus()

func _on_input_gui_input(event: InputEvent):
	if event is InputEventKey and event.pressed and not event.shift_pressed:
		if event.keycode in [KEY_ENTER, KEY_KP_ENTER]:
			_send_message()
			get_viewport().set_input_as_handled()

func add_message(text: String, is_user: bool):
	var message = message_scene.instantiate()
	message_container.add_child(message)
	
	# Apply theme and font scaling before setting the message
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	# Apply theme from parent
	message.theme = ui_root.theme
	
	# Ensure the label inside the message gets proper font size
	var label = message.get_node_or_null("MarginContainer/Label")
	if label:
		_apply_font_size_to_control(label, font_size)
	
	message.set_message(text, is_user)
	await get_tree().process_frame
	scroll_container.scroll_vertical = scroll_container.get_v_scroll_bar().max_value

func _send_message():
	var text = input_field.text.strip_edges()
	if text.length() > 0:
		add_message(text, true)
		prompt_manager.add_message("user", text)
		send_to_api(prompt_manager.get_messages())
		input_field.text = ""
		# Retain focus to keep the keyboard open
		input_field.grab_focus()

func send_to_api(messages: Array):
	var body = JSON.stringify({
		"messages": messages,
		"do_sample": true,
		"temperature": 0.45,
		"max_new_tokens": 128,
		#"eta_cutoff": 0.002,
		#"penalty_alpha": 0.4,
		#"top_k": 4,
		"repetition_penalty": 1.1
	})

	var error = http_request.request(
		server_url + "/input/",
		["Content-Type: application/json"],
		HTTPClient.METHOD_POST,
		body
	)
	
	if error != OK:
		add_message("Error connecting to server", false)

func _on_request_completed(result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray):
	if result != HTTPRequest.RESULT_SUCCESS:
		add_message("Failed to get response from server (Result Error)", false)
		return
	
	if response_code != 200:
		add_message("Server returned error code: " + str(response_code), false)
		return
	
	var json = JSON.new()
	var response_text = body.get_string_from_utf8()

	if json.parse(response_text) != OK or not json.get_data().has("response"):
		add_message("Failed to parse response: " + json.get_error_message(), false)
		return
	
	var response = json.get_data().get("response")
	prompt_manager.add_message("assistant", response)
	add_message(response, false)
