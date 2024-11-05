# chat_interface.gd
extends CanvasLayer

var message_scene = preload("res://scenes/message.tscn")
var prompt_manager = PromptManager.new()

@onready var ui_root = $UIRoot
@onready var scroll_container = $UIRoot/ScrollContainer
@onready var message_container = $UIRoot/ScrollContainer/MessageContainer
@onready var input_field = $UIRoot/InputContainer/TextEdit
@onready var http_request = $UIRoot/HTTPRequest

func _ready():
	if not _are_nodes_ready():
		push_error("Not all nodes are ready!")
		return
		
	# Connect HTTP request signal
	http_request.request_completed.connect(_on_request_completed)
	
	# Set up the UI layout
	_setup_layout()
	
	# Connect to window resize events
	get_tree().root.size_changed.connect(_on_window_resize)
	
	# Configure TextEdit for Enter key handling
	input_field.gui_input.connect(_on_input_gui_input)
	
	# Make sure TextEdit doesn't create new lines with Enter
	input_field.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	input_field.scroll_fit_content_height = true

func _setup_layout():
	# UIRoot should fill the viewport
	ui_root.anchor_right = 1.0
	ui_root.anchor_bottom = 1.0
	ui_root.offset_right = 0
	ui_root.offset_bottom = 0
	
	# ScrollContainer should fill most of the space
	scroll_container.anchor_right = 1.0
	scroll_container.anchor_bottom = 1.0
	scroll_container.offset_bottom = -60  # Leave space for input
	scroll_container.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_AUTO
	scroll_container.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	
	# Message container should expand to fill scroll container
	message_container.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	message_container.size_flags_vertical = Control.SIZE_EXPAND_FILL
	message_container.add_theme_constant_override("separation", 10)  # Space between messages
	
	# Input container at bottom
	var input_container = $UIRoot/InputContainer
	input_container.anchor_top = 1.0
	input_container.anchor_right = 1.0
	input_container.anchor_bottom = 1.0
	input_container.offset_top = -50
	input_container.offset_left = 10
	input_container.offset_right = -10
	input_container.offset_bottom = -10
	
	# TextEdit should expand horizontally
	input_field.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	input_field.custom_minimum_size.y = 40

func _on_window_resize():
	_setup_layout()
	
	# Update all existing messages
	for message in message_container.get_children():
		if message.has_method("set_message"):
			# Reapply the message settings to update the width
			message.custom_minimum_size.x = min(get_viewport().size.x * 0.8, 600)

func _are_nodes_ready() -> bool:
	return ui_root != null and \
		   scroll_container != null and \
		   message_container != null and \
		   input_field != null and \
		   http_request != null

func _on_input_gui_input(event: InputEvent):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_ENTER or event.keycode == KEY_KP_ENTER:
			if !event.shift_pressed:  # Shift+Enter can be used for newline if we want to support that
				event.pressed = false  # Consume the event
				_send_message()
				get_viewport().set_input_as_handled()

func _send_message():
	var text = input_field.text.strip_edges()
	if text.length() > 0:
		add_message(text, true)  # true for user message
		prompt_manager.add_message("INK", text)
		send_to_api(prompt_manager.build_prompt())
		input_field.text = ""

func add_message(text: String, is_user: bool):
	var message = message_scene.instantiate()
	message_container.add_child(message)
	message.set_message(text, is_user)
	# Wait for next frame to ensure message is properly sized
	await get_tree().process_frame
	scroll_container.scroll_vertical = scroll_container.get_v_scroll_bar().max_value

func send_to_api(text: String):
	var headers = ["Content-Type: application/json"]
	print("Sending prompt to API: ", text)  # Debug print
	
	var body = JSON.stringify({
			"prompt": text, 
			"do_sample": true, 
			"temperature": 0.45, 
			"max_new_tokens": 128, 
			"eta_cutoff": 0.002, 
			"penalty_alpha": 0.4, 
			"top_k": 4, 
			"repetition_penalty": 1.35,
			"stop_strings": ["\nINK:", "\nPEN:"],
			#"stop_strings": ["INK", "PEN"],
			"skip_special_tokens": false
		})
	
	var error = http_request.request("http://192.168.5.94:2100/input/", headers, HTTPClient.METHOD_POST, body)
	if error != OK:
		add_message("Error connecting to server", false)

const SPECIAL_TOKENS = [
	"[EOS]",
	"[TAC]",
	"[CAT]",
	"[CTX]",
	"[XTC]",
	"[BOS]",
	"[PAD]"
]

func _strip_special_tokens(text: String) -> String:
	var cleaned_text = text
	for token in SPECIAL_TOKENS:
		# Remove the token and any whitespace that might be around it
		cleaned_text = cleaned_text.replace(token, "")
	return cleaned_text.strip_edges()

func _extract_last_response(full_response: String) -> String:
	print("Full response received: ", full_response)  # Debug print
	
	# If response is empty or just whitespace
	if full_response.strip_edges() == "":
		return "Error: Empty response from server"
	
	# Find the last occurrence of "PEN:"
	var parts = full_response.split("PEN:", false)
	if parts.size() <= 0:
		return "Error: Malformed response"
	
	# Get the last part (most recent response)
	var response = parts[-1].strip_edges()
	
	# Stop at INK: if present
	var ink_index = response.find("INK:")
	if ink_index != -1:
		response = response.substr(0, ink_index).strip_edges()
	
	# Strip special tokens
	response = _strip_special_tokens(response)
	
	# If we ended up with empty string after cleaning
	if response.strip_edges() == "":
		return "Error: No valid response content"
		
	return response

func _on_request_completed(result, response_code, headers, body: PackedByteArray):
	if result != HTTPRequest.RESULT_SUCCESS:
		add_message("Failed to get response from server (Result Error)", false)
		return
	
	# Check for specific HTTP status codes
	if response_code != 200:
		add_message("Server returned error code: " + str(response_code), false)
		print("Error response: ", body.get_string_from_utf8())
		return
	
	# Decode and parse response
	var response_text = body.get_string_from_utf8()
	print("Raw server response: ", response_text)  # Debug print
	
	# Try to parse as JSON
	var json = JSON.new()
	var parse_result = json.parse(response_text)
	
	if parse_result != OK:
		var error_msg = "Failed to parse response: " + json.get_error_message()
		print("Parse error: ", error_msg)
		print("Response was: ", response_text)
		add_message(error_msg, false)
		return
	
	var response_data = json.get_data()
	if response_data == null:
		add_message("No valid data in response", false)
		return
	
	if not response_data.has("response"):
		add_message("Unexpected response format", false)
		print("Unexpected format: ", response_data)
		return
	
	# Extract just the new response from the full text
	var full_response = response_data.get("response")
	var clean_response = _extract_last_response(full_response)
	
	# Don't add error messages to prompt manager
	if not clean_response.begins_with("Error:"):
		prompt_manager.add_message("PEN", clean_response)
		
	add_message(clean_response, false)
	
func clear_chat_history():
	# Clear visual messages
	for child in message_container.get_children():
		child.queue_free()
	# Clear prompt history
	prompt_manager.clear_history()
