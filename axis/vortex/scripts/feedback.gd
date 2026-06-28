extends CanvasLayer
class_name FeedbackConsole

## FeedbackConsole - the authoring feedback channel.
##
## The visualizer is built by eye, but "this shape feels wrong" is hard to act on
## from a written note alone. This console closes that loop: press the toggle key,
## type what feels off about the scene on screen, hit Enter, and it writes a
## self-contained record to `res://feedback/`:
##
##   feedback/NNNN.json  - the typed scene descriptor (name / behavior / shot / seed
##                         / params / the audio frame) plus your query.
##   feedback/NNNN.png   - a screenshot of exactly that frame (the console hides
##                         itself first), so the critique carries the picture too.
##
## That pair is everything needed to understand and reproduce a complaint - the
## seed makes the scene deterministic, and the image shows what "wrong" looked like.
## The directory is git-ignored; it is a working comms channel, not an artifact.

const DIR := "res://feedback"
const TOGGLE_KEY := KEY_QUOTELEFT   # the backtick / tilde key

var _ui: Control
var _info: Label
var _edit: LineEdit
var _open := false
var _shot_img: Image = null      # the frame snapshotted the instant the console opened
var _shot_desc: Dictionary = {}  # the scene descriptor at that same instant


func _ready() -> void:
	layer = 128                      # draw above every scene
	_ensure_dir()
	_build_ui()


# --- UI construction (in code; no .tscn to hand-edit) -----------------------

func _build_ui() -> void:
	_ui = Control.new()
	_ui.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_ui.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_ui.visible = false
	add_child(_ui)

	var dim := ColorRect.new()
	dim.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	dim.color = Color(0, 0, 0, 0.55)
	dim.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_ui.add_child(dim)

	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	margin.add_theme_constant_override("margin_left", 120)
	margin.add_theme_constant_override("margin_right", 120)
	margin.add_theme_constant_override("margin_bottom", 90)
	_ui.add_child(margin)

	var col := VBoxContainer.new()
	col.alignment = BoxContainer.ALIGNMENT_END
	col.add_theme_constant_override("separation", 8)
	margin.add_child(col)

	var title := Label.new()
	title.text = "send feedback  —  what feels wrong with this scene?"
	title.add_theme_color_override("font_color", Color(1, 1, 1, 0.9))
	col.add_child(title)

	_info = Label.new()
	_info.add_theme_color_override("font_color", Color(0.7, 0.85, 1.0, 0.8))
	col.add_child(_info)

	_edit = LineEdit.new()
	_edit.placeholder_text = "describe it, then Enter to save  (Esc or ` to cancel)"
	_edit.custom_minimum_size = Vector2(640, 0)
	_edit.text_submitted.connect(_on_submit)
	col.add_child(_edit)


# --- Input ------------------------------------------------------------------

func _input(event: InputEvent) -> void:
	if not (event is InputEventKey and event.pressed and not event.echo):
		return
	var key: int = event.keycode
	if key == TOGGLE_KEY:
		_toggle()
		get_viewport().set_input_as_handled()
	elif _open and key == KEY_ESCAPE:
		_close()
		get_viewport().set_input_as_handled()
	# All other keys fall through to the LineEdit (typing) and the rest of the app.


func _toggle() -> void:
	if _open:
		_close()
	else:
		_open_console()


func _open_console() -> void:
	# Snapshot the exact frame and scene the user is reacting to, RIGHT NOW - before
	# the overlay is shown and before anything can cut. The feedback must record what
	# was on screen at the keypress, not whatever is current after typing and closing
	# (un-holding the Director on close can immediately cut a new scene in, which is
	# the scene that wrongly ended up in early screenshots).
	_shot_img = get_viewport().get_texture().get_image()
	_shot_desc = Director.current_descriptor()
	_open = true
	_ui.visible = true
	_edit.text = ""
	_edit.grab_focus()
	_info.text = _summary()
	Director.hold(true)            # also keep the scene from cutting away while typing


func _close() -> void:
	_open = false
	_ui.visible = false
	_edit.release_focus()
	Director.hold(false)


func _summary() -> String:
	var d := _shot_desc
	if d.is_empty():
		return ""
	return "%s · %s · %s · seed %d" % [
		d.get("scene", "?"), d.get("render_kind", "?"),
		d.get("behavior", "?"), int(d.get("seed", 0))]


# --- Capture + write --------------------------------------------------------

func _on_submit(text: String) -> void:
	var query := text.strip_edges()
	var img := _shot_img
	var desc := _shot_desc.duplicate(true)
	_close()
	if query.is_empty() or img == null:
		return
	_write(query, img, desc)


# Write the snapshot taken at open time (no live capture - that races the Director).
func _write(query: String, img: Image, desc: Dictionary) -> void:
	var n := _next_index()
	var stem := "%s/%04d" % [DIR, n]
	var png_ok := img.save_png(stem + ".png")
	desc["query"] = query
	desc["index"] = n
	if png_ok == OK:
		desc["screenshot"] = "%04d.png" % n
	var json := JSON.stringify(to_jsonable(desc), "\t")
	var fa := FileAccess.open(stem + ".json", FileAccess.WRITE)
	if fa != null:
		fa.store_string(json)
		fa.close()
	print("vortex: feedback saved -> ", ProjectSettings.globalize_path(stem + ".json"))


func _ensure_dir() -> void:
	if not DirAccess.dir_exists_absolute(ProjectSettings.globalize_path(DIR)):
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(DIR))


# Highest existing NNNN.json index in the feedback dir (0 if none), so records
# accumulate across runs instead of clobbering each other.
func _next_index() -> int:
	var dir := DirAccess.open(DIR)
	if dir == null:
		return 1
	var best := 0
	for fn in dir.get_files():
		if fn.ends_with(".json"):
			var stem := fn.get_basename()
			if stem.is_valid_int():
				best = maxi(best, int(stem))
	return best + 1


## Flatten a descriptor into JSON-native types - the scene params carry Godot
## values (Vector2 / Vector3 / Color / packed arrays) that JSON.stringify can't
## encode directly. Recurses through dictionaries and arrays.
static func to_jsonable(v: Variant) -> Variant:
	match typeof(v):
		TYPE_DICTIONARY:
			var o := {}
			for k in v:
				o[str(k)] = to_jsonable(v[k])
			return o
		TYPE_ARRAY:
			var a := []
			for e in v:
				a.append(to_jsonable(e))
			return a
		TYPE_VECTOR2, TYPE_VECTOR2I:
			return [v.x, v.y]
		TYPE_VECTOR3, TYPE_VECTOR3I:
			return [v.x, v.y, v.z]
		TYPE_COLOR:
			return {"h": v.h, "s": v.s, "v": v.v, "hex": v.to_html()}
		TYPE_PACKED_FLOAT32_ARRAY, TYPE_PACKED_FLOAT64_ARRAY, \
		TYPE_PACKED_INT32_ARRAY, TYPE_PACKED_INT64_ARRAY, TYPE_PACKED_VECTOR2_ARRAY:
			var p := []
			for e in v:
				p.append(to_jsonable(e))
			return p
		_:
			return v
