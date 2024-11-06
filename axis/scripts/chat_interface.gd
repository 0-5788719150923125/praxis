extends CanvasLayer

var message_scene = preload("res://scenes/message.tscn")
var prompt_manager = PromptManager.new()
var is_keyboard_visible: bool = false
var initial_viewport_height: float = 0.0

@onready var ui_root = $UIRoot
@onready var scroll_container = $UIRoot/ScrollContainer
@onready var message_container = $UIRoot/ScrollContainer/MessageContainer
@onready var input_field = $UIRoot/InputContainer/TextEdit
@onready var clear_button = $UIRoot/InputContainer/ClearButton
@onready var http_request = $UIRoot/HTTPRequest
@onready var background_touch = $BackgroundTouch
@onready var input_container = $UIRoot/InputContainer

const KEYBOARD_OFFSET = 1000
const INPUT_MARGIN = 10

func _ready():
	_initialize_components()
	_setup_signals()
	_setup_layout()

func _initialize_components():
	initial_viewport_height = get_viewport().get_visible_rect().size.y
	input_field.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	input_field.scroll_fit_content_height = true
	
	# Make sure background touch can receive input
	background_touch.mouse_filter = Control.MOUSE_FILTER_STOP

func _setup_signals():
	http_request.request_completed.connect(_on_request_completed)
	get_tree().root.size_changed.connect(_on_window_resize)
	input_field.gui_input.connect(_on_input_gui_input)
	input_field.focus_entered.connect(_on_focus_entered)
	input_field.focus_exited.connect(_on_focus_exited)
	background_touch.gui_input.connect(_on_background_touch)
	clear_button.pressed.connect(_on_clear_button_pressed)

func _on_clear_button_pressed() -> void:
	# Clear the conversation history
	clear_chat_history()

func clear_chat_history() -> void:
	# Clear all messages from the UI
	for child in message_container.get_children():
		child.queue_free()
	# Clear the prompt manager history
	prompt_manager.clear_history()
	# Clear the input field if it has any text
	input_field.text = ""

func _on_window_resize():
	var current_height = get_viewport().get_visible_rect().size.y
	if current_height >= initial_viewport_height and is_keyboard_visible:
		_on_focus_exited()
	
	_setup_layout()
	_update_message_sizes()

func _setup_layout():
	_update_input_position(is_keyboard_visible)

func _update_input_position(keyboard_visible: bool):
	var offset = KEYBOARD_OFFSET if keyboard_visible and OS.has_feature("mobile") else 0
	input_container.offset_top = -50 - offset
	input_container.offset_bottom = -INPUT_MARGIN - offset
	scroll_container.offset_bottom = -(50 + INPUT_MARGIN) - offset

func _on_focus_entered() -> void:
	is_keyboard_visible = true
	if OS.has_feature("mobile"):
		await get_tree().create_timer(0.05).timeout
		_update_input_position(true)

func _on_focus_exited() -> void:
	is_keyboard_visible = false
	_update_input_position(false)

func _on_background_touch(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		input_field.release_focus()
		get_viewport().set_input_as_handled()
	elif event is InputEventScreenTouch and event.pressed:
		input_field.release_focus()
		get_viewport().set_input_as_handled()

func _update_message_sizes():
	var max_width = min(get_viewport().size.x - 40, 600)  # 20px padding on each side
	for message in message_container.get_children():
		if message.has_method("set_message"):
			message.custom_minimum_size.x = max_width
			message.size.x = max_width

func _notification(what: int) -> void:
	match what:
		NOTIFICATION_WM_GO_BACK_REQUEST:
			if is_keyboard_visible:
				input_field.release_focus()
				get_viewport().set_input_as_handled()
		NOTIFICATION_APPLICATION_FOCUS_OUT:
			if is_keyboard_visible:
				_on_focus_exited()

func _on_input_gui_input(event: InputEvent):
	if event is InputEventKey and event.pressed and not event.shift_pressed:
		if event.keycode in [KEY_ENTER, KEY_KP_ENTER]:
			_send_message()
			get_viewport().set_input_as_handled()

func _send_message():
	var text = input_field.text.strip_edges()
	if text.length() > 0:
		add_message(text, true)
		prompt_manager.add_message("INK", text)
		send_to_api(prompt_manager.get_messages())
		input_field.text = ""
		if is_keyboard_visible:
			input_field.release_focus()

func add_message(text: String, is_user: bool):
	var message = message_scene.instantiate()
	message_container.add_child(message)
	message.set_message(text, is_user)
	await get_tree().process_frame
	scroll_container.scroll_vertical = scroll_container.get_v_scroll_bar().max_value

func send_to_api(messages: Array):
	var body = JSON.stringify({
		"messages": messages,
		"do_sample": true,
		"temperature": 0.45,
		"max_new_tokens": 128,
		"eta_cutoff": 0.002,
		"penalty_alpha": 0.4,
		"top_k": 4,
		"repetition_penalty": 1.35
	})

	var error = http_request.request(
		"http://192.168.5.94:2100/input/",
		["Content-Type: application/json"],
		HTTPClient.METHOD_POST,
		body
	)
	
	if error != OK:
		add_message("Error connecting to server", false)

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
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
	prompt_manager.add_message("PEN", response)
	add_message(response, false)
