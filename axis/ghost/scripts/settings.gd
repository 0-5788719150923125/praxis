extends Node

## Settings - the one owner of `user://ghost.cfg` (autoload).
##
## THE PROBLEM THIS EXISTS FOR. Every remembered value in ghost used to be saved by
## whichever script happened to own the control: the Director's own debounce for the picture
## knobs, the Generative panel's for the voice, the splash's for the last song, the deps
## panel's for its collapse. Five writers, five debounces, and every one of them doing
## `ConfigFile.load()` -> set -> `save()` on the SAME file. That shape has three failure
## modes, and they were all live:
##
##   FORGOTTEN. A new control is persistent only if someone remembers to write the save
##   code. The Vehicle picker went in beside four sliders that the DIRECTOR saves, in a panel
##   that saves its own settings a different way, and which of those two mechanisms it fell
##   under was not visible from the code that added it.
##
##   CLOBBERED. Read-modify-write from separate owners is safe only while nothing else holds
##   a copy. Two ghost processes - and an export renders in a SECOND ONE - both read at boot
##   and both write at exit, so the last to quit silently discards the other's session.
##
##   LOST. Every one of those debounces only ever flushed when it went quiet or at exit, so
##   a kill (or a crash, or an X on the window that the OS does not turn into a close
##   request) threw away everything since the last pause in typing.
##
## So: ONE in-memory copy, ONE writer, and persistence that a control gets by construction
## rather than by remembering. [method bind] is the point of the whole file - hand it a
## control and it loads the stored value in, connects the change signal, and writes on every
## change, so a setting cannot be added without being saved.
##
## READ-ONLY IN A RENDER. An export boots a second ghost against the same file; it must read
## the session's settings and must never write them back. See [member _read_only].

const PATH := "user://ghost.cfg"

## Quiet period after the last change before the disk is touched, in ms. A slider DRAG must
## not write once per frame.
const DEBOUNCE_MS := 400
## ...and the longest a change may go unwritten however busy the control is, in ms.
##
## THE DEBOUNCE ALONE IS NOT ENOUGH, and this is the difference between "saves when you stop
## fiddling" and "saves". A continuous drag keeps pushing the quiet period out ahead of
## itself, so a debounce can be starved indefinitely; and a process that is killed rather
## than quit never reaches its exit flush at all. This bounds what a hard stop can cost to a
## couple of seconds of fiddling, which is the most that should ever be at risk.
const MAX_DIRTY_MS := 2500

var _cfg := ConfigFile.new()
var _dirty := false
var _edit_ms := 0
var _dirty_since := 0
## True in a process that must never write: an export render and the offline analyzer both
## boot the whole app against this same file (see exporter.gd), and a subprocess writing the
## user's settings back is exactly the clobbering described at the top.
var _read_only := false


func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	# The two subprocesses an export spawns: the Movie Maker render (--export, which also
	# carries --bake-file) and the offline analyzer (--bake-song / --bake-out, see
	# bake_runner.gd). Both boot the whole app against the user's own settings file.
	var why := ""
	if args.has("--export") or args.has("--bake-file") or args.has("--bake-song"):
		why = "a render"
	# A TEST PROBE MUST NOT EDIT THE USER'S SETTINGS. run_boot_probe.sh boots the real app
	# at tests/boot_probe.tscn, so a gate that pokes a setter - or merely reads a default
	# and hands it back - persists that, and the next real launch has quietly lost whatever
	# was there. It has happened: with the autoload order wrong for one run, the Director
	# read defaults and a probe wrote them back over a tuned pacing and flourish.
	elif "res://tests/boot_probe.tscn" in OS.get_cmdline_args():
		why = "a test probe"
	_read_only = not why.is_empty()
	_cfg.load(PATH)              # missing file is fine - everything falls back to its default
	if _read_only:
		print("ghost: settings are READ-ONLY in this process (%s)" % why)


## Let a gate that is specifically testing PERSISTENCE write after all. Nothing else may
## call this: the read-only rule above is what keeps every other gate from editing the
## config of whoever is running it.
func allow_writes_for_test() -> void:
	_read_only = false


## The stored value for [param section]/[param key], or [param dflt] when it has never been
## set. Reads come from the in-memory copy, so they are free.
func read(section: String, key: String, dflt: Variant) -> Variant:
	return _cfg.get_value(section, key, dflt)


## Remember [param value]. Writing an unchanged value is a no-op, so this is safe to call
## from a signal that fires every frame of a drag.
func write(section: String, key: String, value: Variant) -> void:
	if _cfg.has_section_key(section, key) and _cfg.get_value(section, key) == value:
		return
	_cfg.set_value(section, key, value)
	_edit_ms = Time.get_ticks_msec()
	if not _dirty:
		_dirty_since = _edit_ms
	_dirty = true


## Write now, whatever the debounce thinks. Called at every exit path; safe to call when
## nothing is pending.
func flush() -> void:
	if not _dirty or _read_only:
		_dirty = false
		return
	_dirty = false
	var err := _cfg.save(PATH)
	if err != OK:
		push_warning("ghost: could not save settings to %s (error %d)" % [PATH, err])


func _process(_delta: float) -> void:
	if not _dirty:
		return
	var now := Time.get_ticks_msec()
	if now - _edit_ms >= DEBOUNCE_MS or now - _dirty_since >= MAX_DIRTY_MS:
		flush()


func _exit_tree() -> void:
	flush()


func _notification(what: int) -> void:
	# Belt and braces around the exit flush: the window's close button, the app being
	# backgrounded on a platform that can kill it, and a plain quit all land here.
	if what == NOTIFICATION_WM_CLOSE_REQUEST \
			or what == NOTIFICATION_APPLICATION_FOCUS_OUT \
			or what == NOTIFICATION_PREDELETE:
		flush()


## BIND A CONTROL TO A SETTING, which is what makes persistence the default rather than
## something to remember. Loads the stored value into [param control], connects its change
## signal, and writes on every change.
##
## [param keys] is for an [OptionButton] whose choice must survive the list being reordered
## or added to: give it the stable keys behind the items (a registry's keys, say) and the
## setting stores the KEY rather than the index. Without it the index is stored, which is
## right only for a list that can never change.
##
## Returns the control, so a builder can bind inline where it would otherwise return.
func bind(control: Control, section: String, key: String, dflt: Variant,
		keys: Array = []) -> Control:
	var stored: Variant = read(section, key, dflt)
	if control is Range:                       # HSlider, VSlider, SpinBox, ProgressBar
		var r := control as Range
		r.set_value_no_signal(clampf(float(stored), r.min_value, r.max_value))
		r.value_changed.connect(func(v: float) -> void: write(section, key, v))
	elif control is OptionButton:
		var o := control as OptionButton
		if keys.is_empty():
			o.select(clampi(int(stored), 0, maxi(0, o.item_count - 1)))
			o.item_selected.connect(func(i: int) -> void: write(section, key, i))
		else:
			o.select(maxi(0, keys.find(stored)))
			o.item_selected.connect(func(i: int) -> void:
				write(section, key, keys[i] if i < keys.size() else dflt))
	elif control is BaseButton:                 # CheckBox, CheckButton, a toggle Button
		var b := control as BaseButton
		b.set_pressed_no_signal(bool(stored))
		b.toggled.connect(func(on: bool) -> void: write(section, key, on))
	elif control is LineEdit:
		var le := control as LineEdit
		le.text = String(stored)
		le.text_changed.connect(func(t: String) -> void: write(section, key, t))
	elif control is TextEdit:
		var te := control as TextEdit
		te.text = String(stored)
		te.text_changed.connect(func() -> void: write(section, key, te.text))
	else:
		push_warning("ghost: Settings.bind cannot persist a %s" % control.get_class())
	return control
