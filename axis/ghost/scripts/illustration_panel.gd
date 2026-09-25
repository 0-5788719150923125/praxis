extends VBoxContainer
class_name IllustrationPanel

## IllustrationPanel - the Generative panel's controls for a book's pictures ([Illustrations]).
##
## Four things, top to bottom: which painter ([ImageGen]); the STYLE every picture shares; the
## REFERENCE images that show that style rather than describe it; and the chapter's own
## pictures, one row each, with what exists, what is being painted and a button to make or
## remake it.
##
## THE LIST FOLLOWS THE TEXT, not the library. [method set_script_text] is called with the
## chapter whenever it is read, so the rows are exactly the `<!-- image: -->` markers in it,
## in reading order - a picture whose description was edited is a new row, and the old one
## simply stops being listed (its files stay in the cache in case the edit is undone).
##
## Nothing here generates on its own. See [Illustrations] for why.

## Big enough to judge a picture from without opening it - the whole point of the list is
## deciding which ones to repaint. Click one for the full-size preview.
const THUMB := 112
const REF_THUMB := 44

var _images: Array = []            # [Manuscript.images] rows for the current text
var _list: VBoxContainer
var _refs: HFlowContainer
var _style: TextEdit
var _style_kind := "image"        # which of Illustrations.STYLE_KINDS the box is editing
var _kind_pick: OptionButton
var _self_ref: CheckBox
var _pick: OptionButton
var _missing_btn: Button
var _all_btn: Button
var _confirm: ConfirmationDialog = null
var _preview: Window = null
var _status: Label
var _dialog: FileDialog = null
var _thumbs := {}                  # path -> ImageTexture, so a rebuild does not re-decode
var _seen := ""                    # what the list last drew, so it rebuilds only on change


func _ready() -> void:
	add_theme_constant_override("separation", 6)
	var head := Label.new()
	head.text = "Illustrations"
	head.add_theme_font_size_override("font_size", 12)
	head.tooltip_text = ("Pictures for the book, painted from the chapter's own "
		+ "<!-- image: ... --> markers. Each is made once and cached - relaunching never "
		+ "repaints anything. Nothing is generated until you press a button, because every "
		+ "picture spends your image quota.")
	add_child(head)

	var brow := HBoxContainer.new()
	brow.add_theme_constant_override("separation", 8)
	add_child(brow)
	var bl := Label.new()
	bl.text = "Painter"
	bl.custom_minimum_size = Vector2(72, 0)
	bl.add_theme_font_size_override("font_size", 12)
	brow.add_child(bl)
	var pick := OptionButton.new()
	_pick = pick
	pick.focus_mode = Control.FOCUS_NONE
	pick.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var keys: Array = ImageGen.REGISTRY.keys()
	var tip := "Which image model paints the pictures.\n"
	for k in keys:
		pick.add_item(String(ImageGen.LABELS.get(k, k)))
		tip += "\n%s - %s" % [ImageGen.LABELS.get(k, k), ImageGen.BLURBS.get(k, "")]
	pick.tooltip_text = tip
	pick.select(maxi(0, keys.find(Illustrations.backend())))
	pick.item_selected.connect(func(i: int) -> void: Illustrations.set_backend(String(keys[i])))
	brow.add_child(pick)

	# ONE BOX, A PICKER FOR WHICH STYLE IT EDITS: pictures and sketches are briefed separately,
	# and two boxes would double a panel that already scrolls.
	var srow := HBoxContainer.new()
	srow.add_theme_constant_override("separation", 8)
	add_child(srow)
	var sl := Label.new()
	sl.text = "Style"
	sl.add_theme_font_size_override("font_size", 12)
	sl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	srow.add_child(sl)
	_kind_pick = OptionButton.new()
	_kind_pick.focus_mode = Control.FOCUS_NONE
	for k in Illustrations.STYLE_KINDS:
		_kind_pick.add_item(String(Illustrations.STYLE_LABELS.get(k, k)))
	_kind_pick.tooltip_text = ("Which look the style box and the references below edit. "
		+ "Pictures (`<!-- image: -->`) and sketches (`<!-- sketch: -->`) each have their own.")
	_kind_pick.item_selected.connect(func(i: int) -> void:
		_style_kind = String(Illustrations.STYLE_KINDS[i])
		_style.text = Illustrations.style(_style_kind)
		_self_ref.set_pressed_no_signal(Illustrations.self_reference(_style_kind))
		_refresh_refs())
	srow.add_child(_kind_pick)
	_self_ref = CheckBox.new()
	_self_ref.text = "Match earlier"
	_self_ref.focus_mode = Control.FOCUS_NONE
	_self_ref.tooltip_text = ("Send each picture of this kind the ones already made before it in "
		+ "the chapter (the first and the most recent few), so they read as one hand. Works "
		+ "alongside the references below and never adds to them. Saved in the document.")
	_self_ref.button_pressed = Illustrations.self_reference(_style_kind)
	_self_ref.toggled.connect(func(on: bool) -> void:
		Illustrations.set_self_reference(on, _style_kind)
		_refresh_refs())
	srow.add_child(_self_ref)
	_style = TextEdit.new()
	_style.custom_minimum_size = Vector2(0, 72)
	_style.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_style.placeholder_text = "e.g. muted watercolour and ink, like a mid-century storybook plate"
	_style.tooltip_text = ("How every picture of the chosen kind should look - medium, palette, "
		+ "linework, mood. Added to each one's own description. Changing it does not "
		+ "throw away pictures already made: they are marked stale and kept until you regenerate them.")
	_style.text = Illustrations.style(_style_kind)
	_style.text_changed.connect(func() -> void:
		Illustrations.set_style(_style.text, _style_kind)
		_seen = "")
	add_child(_style)

	var rrow := HBoxContainer.new()
	rrow.add_theme_constant_override("separation", 8)
	add_child(rrow)
	var rl := Label.new()
	rl.text = "References"
	rl.add_theme_font_size_override("font_size", 12)
	rl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rrow.add_child(rl)
	var imp := Button.new()
	imp.text = "Import…"
	imp.focus_mode = Control.FOCUS_NONE
	imp.tooltip_text = ("Pick images whose STYLE this kind (the dropdown above) should share. "
		+ "They are attached to every request of that kind as style references only - their "
		+ "subjects are never copied. With none, each picture is matched to the ones already "
		+ "made before it in the chapter instead. Copied into ghost's own folder, so moving the "
		+ "originals changes nothing.")
	imp.pressed.connect(_open_dialog)
	rrow.add_child(imp)
	_refs = HFlowContainer.new()
	_refs.add_theme_constant_override("h_separation", 4)
	_refs.add_theme_constant_override("v_separation", 4)
	add_child(_refs)

	var grow := HBoxContainer.new()
	grow.add_theme_constant_override("separation", 8)
	add_child(grow)
	_missing_btn = Button.new()
	_missing_btn.focus_mode = Control.FOCUS_NONE
	_missing_btn.tooltip_text = "Paint every picture in this chapter that does not have one yet."
	_missing_btn.pressed.connect(_generate_missing)
	grow.add_child(_missing_btn)
	# REGENERATE ALL IS CONFIRMED, and says what it costs: it is the one button here that can
	# spend a chapter's worth of quota in a click.
	_all_btn = Button.new()
	_all_btn.focus_mode = Control.FOCUS_NONE
	_all_btn.tooltip_text = ("Paint every picture in this chapter again, including ones "
		+ "already made (those are kept as earlier versions). Asks first.")
	_all_btn.pressed.connect(_confirm_all)
	grow.add_child(_all_btn)
	_status = Label.new()
	_status.add_theme_font_size_override("font_size", 11)
	_status.add_theme_color_override("font_color", Color(0.55, 0.95, 0.75, 0.85))
	_status.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	# ITS OWN ROW. Beside the two buttons it was left a sliver of width, and a wrapping label
	# given a sliver wraps every character: the progress read vertically, one letter a line.
	add_child(_status)

	_list = VBoxContainer.new()
	_list.add_theme_constant_override("separation", 4)
	add_child(_list)
	_refresh_refs()
	_refresh()


## The library changed underneath the controls - a document brought its own look. Show it.
func sync_from_library() -> void:
	if _pick != null:
		_pick.select(maxi(0, ImageGen.REGISTRY.keys().find(Illustrations.backend())))
	if _style != null and _style.text != Illustrations.style(_style_kind):
		_style.text = Illustrations.style(_style_kind)
	if _self_ref != null:
		_self_ref.set_pressed_no_signal(Illustrations.self_reference(_style_kind))
	if _refs != null:
		_refresh_refs()
	_seen = ""


## The chapter's text, whenever it is read. Cheap enough to call on every edit.
func set_script_text(body: String) -> void:
	_images = Manuscript.images(body)
	Illustrations.set_chapter(_images)
	_seen = ""
	if is_inside_tree():
		_refresh()


func _process(_delta: float) -> void:
	Illustrations.pump()
	_refresh()


## Rebuild the rows only when something a row shows has changed - a status, a version, the
## look. A list rebuilt every frame would eat clicks on its own buttons.
func _refresh() -> void:
	if _list == null:
		return
	var sig := "%d|%s|%s|%d" % [Illustrations.revision, Illustrations.current_signature("image"),
		Illustrations.current_signature("sketch"), _images.size()]
	for im in _images:
		sig += "|" + Illustrations.status(String(im["key"]))
	if sig == _seen:
		return
	_seen = sig
	for c in _list.get_children():
		c.queue_free()
	var missing := 0
	for im in _images:
		if Illustrations.status(String(im["key"])) in ["missing", "error"]:
			missing += 1
		_list.add_child(_row(im))
	if _images.is_empty():
		var none := Label.new()
		none.text = "  (no <!-- image: ... --> markers in this text)"
		none.add_theme_font_size_override("font_size", 11)
		none.modulate = Color(1, 1, 1, 0.6)
		_list.add_child(none)
	_missing_btn.text = "Generate missing (%d)" % missing
	_missing_btn.disabled = missing == 0 or Illustrations.read_only()
	_all_btn.text = "Regenerate all (%d)" % _images.size()
	_all_btn.disabled = _images.is_empty() or Illustrations.read_only()
	var n := Illustrations.busy()
	_status.text = ("⏳  Painting %d…" % n) if n > 0 else ""


func _row(im: Dictionary) -> Control:
	var key := String(im["key"])
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	# The thumbnail IS the preview button: a flat button showing the picture, so the thing
	# to click is the thing you want a closer look at.
	var thumb := Button.new()
	thumb.flat = true
	thumb.focus_mode = Control.FOCUS_NONE
	thumb.custom_minimum_size = Vector2(THUMB, THUMB)
	thumb.expand_icon = true
	thumb.icon_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var path := Illustrations.path_for(key)
	if not path.is_empty():
		thumb.icon = _thumb(path, THUMB * 2)
		thumb.tooltip_text = "Click to preview full size."
		thumb.pressed.connect(func() -> void: _open_preview(im))
	else:
		thumb.text = "none"
		thumb.disabled = true
	row.add_child(thumb)

	var col := VBoxContainer.new()
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	col.add_theme_constant_override("separation", 0)
	row.add_child(col)
	var desc := Label.new()
	desc.text = String(im["prompt"])
	desc.tooltip_text = String(im["prompt"])
	desc.add_theme_font_size_override("font_size", 11)
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	desc.max_lines_visible = 2
	desc.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	col.add_child(desc)
	var st := Illustrations.status(key)
	var meta := Label.new()
	var parts := PackedStringArray([_placement_label(im), String(im.get("side", "")), st])
	var vs := Illustrations.versions(key)
	if vs.size() > 1:
		parts.append("v%d/%d" % [Illustrations.current_index(key) + 1, vs.size()])
	if Illustrations.is_stale(key):
		parts.append("stale")
	meta.text = "  ·  ".join(parts)
	meta.add_theme_font_size_override("font_size", 10)
	meta.modulate = Color(1, 1, 1, 0.6)
	if st == "error":
		meta.text += "  -  " + Illustrations.error_of(key)
		meta.tooltip_text = Illustrations.error_of(key)
		meta.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		meta.add_theme_color_override("font_color", Color(1.0, 0.6, 0.55))
	elif Illustrations.is_stale(key):
		meta.tooltip_text = "Made under a different style or reference set. Still used; regenerate to repaint it."
	col.add_child(meta)

	if vs.size() > 1:
		var at := Illustrations.current_index(key)
		for step in [-1, 1]:
			var b := Button.new()
			b.text = "‹" if step < 0 else "›"
			b.focus_mode = Control.FOCUS_NONE
			b.tooltip_text = "Show the %s version." % ("previous" if step < 0 else "next")
			b.disabled = at + step < 0 or at + step >= vs.size()
			b.pressed.connect(func() -> void: Illustrations.select_version(key, at + step))
			row.add_child(b)
	if not path.is_empty():
		var del := _delete_button("✕", key, Illustrations.current_index(key))
		row.add_child(del)
	var go := Button.new()
	go.focus_mode = Control.FOCUS_NONE
	go.text = "Regenerate" if not path.is_empty() else "Generate"
	go.tooltip_text = ("Paint it again. The current picture is kept as an earlier version."
		if not path.is_empty() else "Paint this picture.")
	go.disabled = st in ["queued", "running"] or Illustrations.read_only()
	go.pressed.connect(func() -> void: _ask(im))
	row.add_child(go)
	return row


func _ask(im: Dictionary) -> void:
	var err := Illustrations.generate(im)
	if not err.is_empty():
		_status.text = "⚠  " + err
	_seen = ""


func _confirm_all() -> void:
	if _confirm == null:
		_confirm = ConfirmationDialog.new()
		_confirm.title = "Regenerate every picture?"
		_confirm.ok_button_text = "Regenerate all"
		_confirm.confirmed.connect(_generate_all)
		add_child(_confirm)
	_confirm.dialog_text = ("This paints all %d pictures in the chapter again, one request "
		% _images.size() + "each against your image quota. Pictures already made are kept "
		+ "as earlier versions.")
	_confirm.popup_centered()


func _generate_all() -> void:
	for im in _images:
		if Illustrations.status(String(im["key"])) in ["queued", "running"]:
			continue
		var err := Illustrations.generate(im)
		if not err.is_empty():
			_status.text = "⚠  " + err
			return
	_seen = ""


## THE PREVIEW: the picture at full size in a window of its own, with its description and
## its versions, so a repaint can be judged against the one before without leaving it.
func _open_preview(im: Dictionary) -> void:
	var key := String(im["key"])
	if _preview == null or not is_instance_valid(_preview):
		_preview = Window.new()
		_preview.close_requested.connect(func() -> void: _preview.hide())
		_preview.window_input.connect(func(e: InputEvent) -> void:
			if e is InputEventKey and e.pressed and (e as InputEventKey).keycode == KEY_ESCAPE:
				_preview.hide())
		add_child(_preview)
	for c in _preview.get_children():
		c.queue_free()
	var vp := get_viewport().get_visible_rect().size
	_preview.size = Vector2i(int(vp.x * 0.8), int(vp.y * 0.85))
	_preview.title = "Illustration  ·  %s" % (_placement_label(im))
	var bg := PanelContainer.new()
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	_preview.add_child(bg)
	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 8)
	bg.add_child(col)
	var pic := TextureRect.new()
	pic.size_flags_vertical = Control.SIZE_EXPAND_FILL
	pic.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	pic.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	var img := Image.new()
	var path := Illustrations.path_for(key)
	if img.load(path) == OK:
		pic.texture = ImageTexture.create_from_image(img)
	col.add_child(pic)
	var desc := Label.new()
	desc.text = String(im["prompt"])
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	desc.add_theme_font_size_override("font_size", 12)
	col.add_child(desc)
	var bar := HBoxContainer.new()
	bar.alignment = BoxContainer.ALIGNMENT_CENTER
	bar.add_theme_constant_override("separation", 8)
	col.add_child(bar)
	var vs := Illustrations.versions(key)
	var at := Illustrations.current_index(key)
	for step in [-1, 1]:
		var b := Button.new()
		b.text = "‹ previous" if step < 0 else "next ›"
		b.disabled = at + step < 0 or at + step >= vs.size()
		b.pressed.connect(func() -> void:
			Illustrations.select_version(key, at + step)
			_open_preview(im))
		bar.add_child(b)
		if step < 0:
			var v := Label.new()
			v.text = "version %d of %d%s" % [at + 1, vs.size(),
				"  (stale)" if Illustrations.is_stale(key) else ""]
			bar.add_child(v)
	if at >= 0:
		var del := _delete_button("Delete this version", key, at)
		del.pressed.connect(func() -> void:
			if bool(del.get_meta("gone", false)):
				if Illustrations.versions(key).is_empty():
					_preview.hide()
				else:
					_open_preview(im))
		bar.add_child(del)
	var again := Button.new()
	again.text = "Regenerate"
	again.disabled = Illustrations.status(key) in ["queued", "running"] or Illustrations.read_only()
	again.pressed.connect(func() -> void:
		_ask(im)
		_preview.hide())
	bar.add_child(again)
	_preview.popup_centered()


## DELETE TAKES TWO CLICKS, because it cannot be undone: the first arms the button for a few
## seconds, the second deletes. A dialog for every bad picture would make clearing a run of
## them a chore, which is exactly when this is used.
func _delete_button(label: String, key: String, version: int) -> Button:
	var b := Button.new()
	b.text = label
	b.focus_mode = Control.FOCUS_NONE
	b.tooltip_text = ("Delete this version of the picture - the file too. The one before it is "
		+ "shown instead, or none. Later pictures stop being matched to it.")
	b.disabled = Illustrations.read_only()
	# The disarm timer is the BUTTON'S CHILD, not a SceneTree timer: the delete rebuilds the list
	# and frees this button, and a tree timer still holding it fired into a freed capture.
	var disarm := Timer.new()
	disarm.one_shot = true
	disarm.wait_time = 3.0
	disarm.timeout.connect(func() -> void:
		b.set_meta("armed", false)
		b.text = label
		b.remove_theme_color_override("font_color"))
	b.add_child(disarm)
	b.pressed.connect(func() -> void:
		if not bool(b.get_meta("armed", false)):
			b.set_meta("armed", true)
			b.text = "Delete?"
			b.add_theme_color_override("font_color", Color(1.0, 0.55, 0.5))
			disarm.start()
			return
		disarm.stop()
		b.set_meta("gone", Illustrations.delete_version(key, version))
		_thumbs.clear()
		_seen = "")
	return b


func _generate_missing() -> void:
	for im in _images:
		if Illustrations.status(String(im["key"])) in ["missing", "error"]:
			var err := Illustrations.generate(im)
			if not err.is_empty():
				_status.text = "⚠  " + err
				return
	_seen = ""


func _thumb(path: String, px: int) -> Texture2D:
	if _thumbs.has(path):
		return _thumbs[path]
	var img := Image.new()
	if img.load(path) != OK:
		return null
	var s := float(px) / float(maxi(img.get_width(), img.get_height()))
	img.resize(maxi(1, int(img.get_width() * s)), maxi(1, int(img.get_height() * s)),
		Image.INTERPOLATE_BILINEAR)
	var tex := ImageTexture.create_from_image(img)
	_thumbs[path] = tex
	return tex


# --- references --------------------------------------------------------------

func _refresh_refs() -> void:
	for c in _refs.get_children():
		c.queue_free()
	var list := Illustrations.references(_style_kind)
	if list.is_empty():
		var none := Label.new()
		none.text = ("  (none - each is matched to the ones made before it in the chapter)"
			if Illustrations.self_reference(_style_kind) else "  (none - the style text alone decides the look)")
		none.add_theme_font_size_override("font_size", 11)
		none.modulate = Color(1, 1, 1, 0.6)
		_refs.add_child(none)
		return
	for i in list.size():
		var p := ProjectSettings.globalize_path(String(list[i]))
		var box := VBoxContainer.new()
		box.add_theme_constant_override("separation", 0)
		var t := TextureRect.new()
		t.custom_minimum_size = Vector2(REF_THUMB, REF_THUMB)
		t.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		t.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_COVERED
		t.texture = _thumb(p, REF_THUMB * 2)
		t.tooltip_text = p
		box.add_child(t)
		var x := Button.new()
		x.text = "×"
		x.flat = true
		x.focus_mode = Control.FOCUS_NONE
		x.tooltip_text = "Stop using this reference. Pictures already made are kept, marked stale."
		var at := i
		x.pressed.connect(func() -> void:
			Illustrations.remove_reference(at, _style_kind)
			_refresh_refs()
			_seen = "")
		box.add_child(x)
		_refs.add_child(box)


func _open_dialog() -> void:
	if _dialog != null and is_instance_valid(_dialog):
		return
	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILES
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	# In-window, never native: the portal dialog shows nothing without xdg-desktop-portal.
	_dialog.use_native_dialog = false
	_dialog.title = "Import style references"
	_dialog.filters = PackedStringArray(["*.png, *.jpg, *.jpeg, *.webp ; Images"])
	var pics := OS.get_system_dir(OS.SYSTEM_DIR_PICTURES)
	if not pics.is_empty():
		_dialog.current_dir = pics
	_dialog.size = Vector2i(820, 560)
	_dialog.files_selected.connect(func(paths: PackedStringArray) -> void:
		var errs := Illustrations.add_references(Array(paths), _style_kind)
		_status.text = "" if errs.is_empty() else "⚠  " + "; ".join(errs)
		_refresh_refs()
		_seen = ""
		_close_dialog())
	_dialog.canceled.connect(_close_dialog)
	add_child(_dialog)
	_dialog.popup_centered()


func _close_dialog() -> void:
	if _dialog != null and is_instance_valid(_dialog):
		_dialog.queue_free()
	_dialog = null


## How a picture's placement reads in the list: "full page", "inline" or "sketch".
static func _placement_label(im: Dictionary) -> String:
	match String(im.get("placement", "")):
		"full":
			return "full page"
		"sketch":
			return "sketch"
	return "inline"
