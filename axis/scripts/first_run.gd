extends Control

const CONFIG_PATH = "user://settings.cfg"
const URL_PATTERN = "^(http|https)://[\\w\\.-]+(:\\d+)?$"

# UI scaling constants
const MOBILE_SCALE_FACTOR = 2.5
const MIN_BUTTON_HEIGHT = 70
const MOBILE_FONT_SIZE = 24
const DESKTOP_FONT_SIZE = 16

var config = ConfigFile.new()
var pending_url: String = ""
var http_request: HTTPRequest

func _ready():
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_ping_completed)
	
	if load_settings():
		_start_main_scene()
	_apply_platform_scaling()

func _apply_platform_scaling():
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	for button in [$VBoxContainer/DemoButton, $VBoxContainer/AddButton]:
		if is_mobile:
			button.custom_minimum_size.y = MIN_BUTTON_HEIGHT
			button.custom_minimum_size.x = 200
		else:
			button.custom_minimum_size = Vector2(150, 40)
		button.add_theme_font_size_override("font_size", font_size)

func _scale_dialog(dialog: Window, font_size: int):
	# Scale the dialog and its children
	if dialog.has_method("add_theme_font_size_override"):
		dialog.add_theme_font_size_override("font_size", font_size)
	
	# Scale all Label nodes within the dialog
	for child in dialog.get_children():
		if child is Label:
			child.add_theme_font_size_override("font_size", font_size)
		elif child is LineEdit:
			child.add_theme_font_size_override("font_size", font_size)
		elif child is Button:
			child.add_theme_font_size_override("font_size", font_size)
		elif child is Container:
			for subchild in child.get_children():
				if subchild is Label:
					subchild.add_theme_font_size_override("font_size", font_size)
				elif subchild is LineEdit:
					subchild.add_theme_font_size_override("font_size", font_size)
				elif subchild is Button:
					subchild.add_theme_font_size_override("font_size", font_size)

func process_url(url: String) -> String:
	var regex = RegEx.new()
	regex.compile("^(http|https)://([\\w\\.-]+)(:\\d+)?$")
	
	var result = regex.search(url)
	if result:
		var protocol = result.get_string(1)
		var hostname = result.get_string(2)
		var port = result.get_string(3)
		
		# If no port specified, use default based on protocol
		if port.is_empty():
			port = ":443" if protocol == "https" else ":80"
		
		return protocol + "://" + hostname + port
	
	return ""

func _on_demo_button_pressed():
	_start_main_scene()

func _on_add_button_pressed():
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	var popup = AcceptDialog.new()
	popup.title = "Enter Server URL"
	
	var vbox = VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 20)
	
	var label = Label.new()
	label.text = "Please enter the server URL:\nFormat: protocol://hostname[:port]\nPort defaults: HTTP=80, HTTPS=443"
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(label)
	
	var dialog = LineEdit.new()
	dialog.placeholder_text = "https://hostname[:port]"
	dialog.custom_minimum_size.x = 300 if is_mobile else 200
	if is_mobile:
		dialog.custom_minimum_size.y = MIN_BUTTON_HEIGHT
	vbox.add_child(dialog)
	
	popup.add_child(vbox)
	popup.min_size = Vector2(400 if is_mobile else 300, 0)
	
	popup.add_button("Cancel", false, "cancel")
	popup.confirmed.connect(_on_url_submitted.bind(dialog))
	
	add_child(popup)
	_scale_dialog(popup, font_size)
	popup.popup_centered()
	dialog.grab_focus()

func _on_url_submitted(dialog: LineEdit):
	var url = dialog.text.strip_edges()
	var processed_url = process_url(url)
	
	if processed_url.is_empty():
		_show_error("Invalid URL format.\nPlease use: protocol://hostname[:port]")
		return
	
	pending_url = processed_url
	_show_loading("Checking server connection...")
	
	var error = http_request.request(
		pending_url,
		["Content-Type: application/json"],
		HTTPClient.METHOD_GET
	)
	
	if error != OK:
		_show_error("Failed to connect to server")

func _show_loading(message: String):
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	var popup = AcceptDialog.new()
	popup.title = "Checking Connection"
	popup.dialog_text = message
	popup.min_size = Vector2(400 if is_mobile else 300, 0)
	popup.get_ok_button().hide()
	
	add_child(popup)
	_scale_dialog(popup, font_size)
	popup.popup_centered()

func _show_success(message: String):
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	var popup = AcceptDialog.new()
	popup.title = "Success"
	popup.dialog_text = message
	popup.min_size = Vector2(400 if is_mobile else 300, 0)
	
	add_child(popup)
	_scale_dialog(popup, font_size)
	popup.popup_centered()
	popup.confirmed.connect(popup.queue_free)

func _show_error(message: String):
	var is_mobile = OS.has_feature("mobile")
	var font_size = MOBILE_FONT_SIZE if is_mobile else DESKTOP_FONT_SIZE
	
	var popup = AcceptDialog.new()
	popup.title = "Error"
	popup.dialog_text = message
	popup.min_size = Vector2(400 if is_mobile else 300, 0)
	
	add_child(popup)
	_scale_dialog(popup, font_size)
	popup.popup_centered()
	popup.confirmed.connect(popup.queue_free)

func _on_ping_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	for child in get_children():
		if child is AcceptDialog and child.title == "Checking Connection":
			child.queue_free()
	
	if result != HTTPRequest.RESULT_SUCCESS:
		_show_error("Failed to connect to server.\nPlease check the URL and try again.")
		return
	
	if response_code != 200:
		_show_error("Server returned error code: " + str(response_code) + "\nPlease verify the server is running.")
		return
	
	_show_success("Server connection successful!")
	
	config.set_value("Server", "url", pending_url)
	config.save(CONFIG_PATH)
	
	await get_tree().create_timer(1.0).timeout
	_start_main_scene()

func load_settings() -> bool:
	if config.load(CONFIG_PATH) == OK:
		var url = config.get_value("Server", "url", "")
		return !url.is_empty()
	return false

func _start_main_scene():
	get_tree().change_scene_to_file("res://scenes/main.tscn")
