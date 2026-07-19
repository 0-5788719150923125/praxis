extends CanvasLayer
class_name SynthEditor

## SynthEditor - the synthesis surface: text in, narrated show out, live.
##
## A left panel holds the **lyrics box** (a plain paragraph, `[K AE T]`
## phonetic escapes allowed) and the **speaker as a trait vector**: one slider
## per [Voice] trait axis (pitch, lilt, tract, pace, breath, grit, drawl). The
## zero vector is the hand-curated default speaker; **Roll** initializes the
## sliders from a seed, **Default** returns to the curated centre - but the
## sliders themselves are the voice's identity, so any speaker is replicated by
## its vector (autosaved), never by the gestures that found it.
##
## **Speak is real-time**: it hands a [VoiceStream] to main and audio starts
## the same frame - synthesis runs ~30x real time and stays a sliding window
## ahead of the playhead, so there is no pre-render wait at all. The show
## reacts to the voice as it is spoken (the stream feeds [Spectrum]'s analyzed
## bus), karaoke subtitles grow live from the stream's timing map, and the WAV
## + sidecar land in the background once the take finishes (export-ready). The
## draft autosaves (text, seed, traits) debounced after every edit and on exit.

const CFG := "user://ghost.cfg"
const AUTOSAVE_DELAY_MS := 800

var begin_stream: Callable          # set by main: (stream: VoiceStream) -> void

var _panel: PanelContainer
var _text: TextEdit
var _seed_edit: LineEdit
var _status: Label
var _sliders := {}                  # trait key -> HSlider
var _dirty := false
var _last_edit_ms := 0


func _ready() -> void:
	layer = 10
	_build_panel()
	_load_persisted()
	var args := OS.get_cmdline_user_args()
	var i := args.find("--synth")
	if i >= 0 and i + 1 < args.size() and FileAccess.file_exists(args[i + 1]):
		_text.text = FileAccess.get_file_as_string(args[i + 1])
	# connect AFTER initial load so restoring the draft doesn't mark it dirty
	_text.text_changed.connect(_mark_dirty)
	_seed_edit.text_changed.connect(func(_t): _mark_dirty())
	# --say: speak the loaded text immediately on boot (automation, demos, and
	# the headless streaming check). Deferred so main has wired begin_stream.
	if args.has("--say"):
		_on_speak.call_deferred()


func _build_panel() -> void:
	_panel = PanelContainer.new()
	_panel.position = Vector2(16, 16)
	_panel.custom_minimum_size = Vector2(380, 0)
	add_child(_panel)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	_panel.add_child(box)

	var title := Label.new()
	title.text = "Synthesis"
	title.add_theme_font_size_override("font_size", 20)
	box.add_child(title)

	var hint := Label.new()
	hint.text = "Write or paste the script. [K AE T] spells a word phonetically."
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.add_theme_font_size_override("font_size", 12)
	hint.modulate = Color(1, 1, 1, 0.6)
	box.add_child(hint)

	_text = TextEdit.new()
	_text.custom_minimum_size = Vector2(360, 220)
	_text.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_text.placeholder_text = "Once upon a time..."
	box.add_child(_text)

	# --- The speaker: one slider per trait axis; the vector IS the voice ---
	var voice_label := Label.new()
	voice_label.text = "Speaker"
	voice_label.add_theme_font_size_override("font_size", 14)
	box.add_child(voice_label)
	for key in Voice.TRAIT_KEYS:
		box.add_child(_trait_row(key))

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)
	_seed_edit = LineEdit.new()
	_seed_edit.text = "1"
	_seed_edit.tooltip_text = "Seed for Roll - initializes the sliders; the sliders are the voice"
	_seed_edit.custom_minimum_size = Vector2(80, 0)
	row.add_child(_seed_edit)
	var roll := Button.new()
	roll.text = "Roll"
	roll.tooltip_text = "Sample a speaker from the seed (sets the sliders)"
	roll.pressed.connect(_roll_traits)
	row.add_child(roll)
	var reroll := Button.new()
	reroll.text = "Reroll"
	reroll.tooltip_text = "New random seed, then Roll"
	reroll.pressed.connect(func():
		_seed_edit.text = str(randi() % 1000000)
		_roll_traits())
	row.add_child(reroll)
	var defaults := Button.new()
	defaults.text = "Default"
	defaults.tooltip_text = "The hand-curated default speaker (all traits zero)"
	defaults.pressed.connect(func():
		for key in _sliders:
			(_sliders[key] as HSlider).value = 0.0
		_mark_dirty())
	row.add_child(defaults)

	var speak := Button.new()
	speak.text = "Speak"
	speak.custom_minimum_size = Vector2(0, 36)
	speak.pressed.connect(_on_speak)
	box.add_child(speak)

	_status = Label.new()
	_status.text = "ready"
	_status.modulate = Color(1, 1, 1, 0.7)
	box.add_child(_status)

	var hide := Button.new()
	hide.text = "Hide panel (F2)"
	hide.pressed.connect(func(): _panel.visible = false)
	box.add_child(hide)


func _trait_row(key: String) -> Control:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	var label := Label.new()
	label.text = key.capitalize()
	label.custom_minimum_size = Vector2(70, 0)
	label.add_theme_font_size_override("font_size", 12)
	row.add_child(label)
	var slider := HSlider.new()
	slider.min_value = -1.0
	slider.max_value = 1.0
	slider.step = 0.01
	slider.value = 0.0
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.value_changed.connect(func(_v): _mark_dirty())
	row.add_child(slider)
	_sliders[key] = slider
	return row


func _trait_values() -> Dictionary:
	var t := {}
	for key in _sliders:
		t[key] = (_sliders[key] as HSlider).value
	return t


func _roll_traits() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = int(_seed_edit.text) if _seed_edit.text.is_valid_int() else hash(_seed_edit.text)
	var spec := Voice.Spec.sample(rng)
	for key in _sliders:
		(_sliders[key] as HSlider).set_value_no_signal(float(spec.traits.get(key, 0.0)))
	_mark_dirty()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_F2:
		_panel.visible = not _panel.visible


# ---- autosave --------------------------------------------------------------


func _mark_dirty() -> void:
	_dirty = true
	_last_edit_ms = Time.get_ticks_msec()


func _process(_delta: float) -> void:
	if _dirty and Time.get_ticks_msec() - _last_edit_ms >= AUTOSAVE_DELAY_MS:
		_persist()


func _exit_tree() -> void:
	if _dirty:
		_persist()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST and _dirty:
		_persist()


func _persist() -> void:
	_dirty = false
	var cfg := ConfigFile.new()
	cfg.load(CFG)
	cfg.set_value("synth", "text", _text.text)
	cfg.set_value("synth", "seed", _seed_edit.text)
	cfg.set_value("synth", "traits", _trait_values())
	cfg.save(CFG)


func _load_persisted() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(CFG) == OK:
		_text.text = cfg.get_value("synth", "text", "")
		_seed_edit.text = str(cfg.get_value("synth", "seed", "1"))
		var t: Dictionary = cfg.get_value("synth", "traits", {})
		for key in _sliders:
			(_sliders[key] as HSlider).set_value_no_signal(float(t.get(key, 0.0)))


# ---- speak -----------------------------------------------------------------


func _on_speak() -> void:
	var text := _text.text.strip_edges()
	if text.is_empty():
		_status.text = "nothing to say"
		return
	_persist()
	var spec := Voice.Spec.from_traits(_trait_values(),
		int(_seed_edit.text) if _seed_edit.text.is_valid_int() else hash(_seed_edit.text))
	var stream: VoiceStream = preload("res://scripts/voice_stream.gd").new()
	stream.setup(text, spec, "user://synth/take_%s" % _seed_edit.text.validate_filename())
	stream.completed.connect(func(dur: float, _wav: String):
		_status.text = "take complete (%.1fs) - looping; export-ready" % dur)
	_status.text = "speaking"
	if begin_stream.is_valid():
		begin_stream.call(stream)
