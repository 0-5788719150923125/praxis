extends CanvasLayer
class_name SynthEditor

## SynthEditor - the synthesis surface: everything is live, nothing is a render.
##
## There is no Speak button. **Speaking is implicit**: the persisted draft
## starts speaking shortly after the surface opens, and from then on every
## adjustment applies itself -
## - **timbre traits** (pitch, tract, breath, grit) retune the running stream
##   IMMEDIATELY ([method VoiceStream.retune]): drag a slider and the voice
##   bends while it speaks, landing ~[constant VoiceStream.TARGET_LEAD]s later.
## - **structural changes** (the text, the seed, and the plan-baked traits:
##   lilt, pace, drawl - they shape durations and pauses, not just the sound)
##   restart the stream in place after a short idle debounce, without touching
##   the scene session.
## The lyrics box takes a plain paragraph (`[K AE T]` phonetic escapes); the
## speaker is a trait vector whose zero point is the hand-curated default
## ([method Voice.Spec.from_traits] holds the centres); Roll/Reroll initialize
## the sliders from a seed, but the vector itself is the replicable identity.
## Draft, seed, and traits autosave debounced and on exit. `--say` applies the
## loaded text immediately on boot (skipping the debounce) for automation.

const CFG := "user://ghost.cfg"
const AUTOSAVE_DELAY_MS := 800

## Traits the running stream can absorb without a re-plan; the rest are baked
## into segment durations/pauses by Voice.plan and need a restart.
const TIMBRE := ["pitch", "tract", "breath", "grit"]

var begin_stream: Callable          # set by main: (stream: VoiceStream) -> void

var _panel: PanelContainer
var _text: TextEdit
var _seed_edit: LineEdit
var _status: Label
var _sliders := {}                  # trait key -> HSlider
var _stream: VoiceStream = null
var _dirty := false                 # autosave pending
var _restart_pending := false       # structural change awaiting the debounce
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
	_text.text_changed.connect(_mark_structural)
	_seed_edit.text_changed.connect(func(_t): _mark_structural())
	# implicit speaking: the loaded draft speaks on its own - immediately with
	# --say, after the normal debounce otherwise
	if not _text.text.strip_edges().is_empty():
		if args.has("--say"):
			_apply.call_deferred()
		else:
			_mark_structural()


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
	hint.text = "Write and it speaks. [K AE T] spells a word phonetically. Timbre sliders bend the voice live."
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
			(_sliders[key] as HSlider).set_value_no_signal(0.0)
		_mark_structural())
	row.add_child(defaults)

	_status = Label.new()
	_status.text = "ready - write and it speaks"
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
	slider.value_changed.connect(func(_v): _on_trait_changed(key))
	row.add_child(slider)
	_sliders[key] = slider
	return row


## A moved slider: timbre bends the live stream this instant; plan-baked traits
## queue a debounced restart. Either way the vector autosaves.
func _on_trait_changed(key: String) -> void:
	_dirty = true
	_last_edit_ms = Time.get_ticks_msec()
	if TIMBRE.has(key) and _stream != null and is_instance_valid(_stream) and not _restart_pending:
		_stream.retune(_current_spec())
	elif not TIMBRE.has(key):
		_restart_pending = true


func _trait_values() -> Dictionary:
	var t := {}
	for key in _sliders:
		t[key] = (_sliders[key] as HSlider).value
	return t


func _current_spec() -> Voice.Spec:
	return Voice.Spec.from_traits(_trait_values(), _seed_int())


func _seed_int() -> int:
	return int(_seed_edit.text) if _seed_edit.text.is_valid_int() else hash(_seed_edit.text)


func _roll_traits() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = _seed_int()
	var spec := Voice.Spec.sample(rng)
	for key in _sliders:
		(_sliders[key] as HSlider).set_value_no_signal(float(spec.traits.get(key, 0.0)))
	_mark_structural()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_F2:
		_panel.visible = not _panel.visible


# ---- the implicit loop: debounce -> persist + apply ------------------------


func _mark_structural() -> void:
	_dirty = true
	_restart_pending = true
	_last_edit_ms = Time.get_ticks_msec()


func _process(_delta: float) -> void:
	if _dirty and Time.get_ticks_msec() - _last_edit_ms >= AUTOSAVE_DELAY_MS:
		_persist()
		if _restart_pending:
			_apply()


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


## Start the stream, or restart the running one in place with the current text
## and voice. The scene session persists across restarts.
func _apply() -> void:
	_restart_pending = false
	var text := _text.text.strip_edges()
	if text.is_empty():
		_status.text = "write something and it will speak"
		return
	var spec := _current_spec()
	if _stream != null and is_instance_valid(_stream):
		_stream.restart(text, spec)
		_status.text = "speaking"
		return
	var stream: VoiceStream = preload("res://scripts/voice_stream.gd").new()
	stream.setup(text, spec, "user://synth/take_%s" % _seed_edit.text.validate_filename())
	stream.completed.connect(func(dur: float, _wav: String):
		_status.text = "take complete (%.1fs) - looping, export-ready" % dur)
	stream.restarted.connect(func(_base: float):
		_status.text = "speaking")
	_stream = stream
	_status.text = "speaking"
	if begin_stream.is_valid():
		begin_stream.call(stream)
