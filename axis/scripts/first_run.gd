extends Control

const CONFIG_PATH = "user://settings.cfg"
const URL_PATTERN = "^(http|https)://[\\w\\.-]+(:\\d+)?$"

var config = ConfigFile.new()

func _ready():
	# Check if we have existing settings
	if load_settings():
		_start_main_scene()

func _on_demo_button_pressed():
	_start_main_scene()

func _on_add_button_pressed():
	var dialog = LineEdit.new()
	dialog.placeholder_text = "https://hostname:port"
	dialog.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	
	var popup = AcceptDialog.new()
	popup.title = "Enter Server URL"
	popup.add_child(dialog)
	popup.add_button("Cancel", false, "cancel")
	popup.confirmed.connect(_on_url_submitted.bind(dialog))
	
	add_child(popup)
	popup.popup_centered()
	dialog.grab_focus()

func _on_url_submitted(dialog: LineEdit):
	var url = dialog.text.strip_edges()
	
	# Validate URL format
	var regex = RegEx.new()
	regex.compile(URL_PATTERN)
	
	if !regex.search(url):
		_show_error("Invalid URL format. Please use: protocol://hostname:port")
		return
	
	# Save the URL
	config.set_value("Server", "url", url)
	config.save(CONFIG_PATH)
	
	# Start the main scene
	_start_main_scene()

func load_settings() -> bool:
	if config.load(CONFIG_PATH) == OK:
		var url = config.get_value("Server", "url", "")
		return !url.is_empty()
	return false

func _show_error(message: String):
	var popup = AcceptDialog.new()
	popup.title = "Error"
	popup.dialog_text = message
	add_child(popup)
	popup.popup_centered()
	popup.confirmed.connect(popup.queue_free)

func _start_main_scene():
	get_tree().change_scene_to_file("res://scenes/main.tscn")
