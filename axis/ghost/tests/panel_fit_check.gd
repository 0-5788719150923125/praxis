extends Node

## panel_fit_check - that a mode's control panel cannot put a control where nobody can reach it.
##
##   tests/run_boot_probe.sh tests/panel_fit_check.gd 120
##
## THE REPORT: "the panel on the left of the generative mode has weird scaling. It is pushing
## elements off the bottom, and you can't scroll down to those elements, nor are they rescaled
## to fit the viewport correctly."
##
## THE FAILURE IS GROWTH, which is why nothing caught it and why a gate is worth having. Both
## voice panels were a bare [PanelContainer] at a fixed corner, and a [Control] outside a
## container is never asked to fit anything - so the panel was exactly as tall as its contents
## and simply ran off the bottom of the window. Every row added is harmless until one is not,
## and the row that breaks it is not the row at fault: six Look filters went onto a panel
## already at the edge, and what went out of reach was the intro/outro holds that had been
## there for months. Nothing errors, nothing is clipped in a way that looks wrong, and the
## controls down there still draw, save and reload. They just cannot be reached.
##
## SO THE CLAIM IS REACHABILITY, not tidiness, and it is asserted in two halves because
## either one alone passes on a broken panel:
##
##   THE PANEL IS INSIDE THE WINDOW. A panel that fits trivially satisfies this; so does one
##   that has silently dropped its contents.
##   ...AND EVERY ROW CAN BE BROUGHT INTO IT. Scrolled to the bottom, the LAST row of the
##   panel must be inside the panel's own rectangle. That is what fails on the build this
##   replaces - there was no scroll at all - and it is what a height cap alone would not give.
##
## Plus the property that makes it safe to apply everywhere: A PANEL THAT FITS IS UNCHANGED.
## It must not stretch to fill the window, or every mode grows a full-height sidebar.
##
## THE CONTROL is the retired arrangement, built here from the same content: a bare
## PanelContainer holding the same VBox. It must overflow at a height where the SidePanel does
## not, or this gate is measuring a window that was always big enough.

const SidePanel_ := preload("res://scripts/side_panel.gd")

## Window heights to sweep. The viewport is 1.5x these (the project stretches content), which
## is why the assertions are all made against the viewport's own rect rather than these.
const HEIGHTS := [1920, 1080, 720]

var _fails: Array = []
var _ed: GenerativeEditor
var _synth: SynthEditor


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	# Built by hand and reparented: _ready would start the voice host, and the panel has to be
	# IN the tree or it has no viewport to fit itself to.
	_ed = GenerativeEditor.new()
	_ed._build_panel()
	_ed.remove_child(_ed._panel)
	add_child(_ed._panel)
	_ed.remove_child(_ed._repace_timer)
	add_child(_ed._repace_timer)

	_synth = SynthEditor.new()
	_synth._build_panel()
	_synth.remove_child(_synth._panel)
	add_child(_synth._panel)

	await _check_fits("Generative", _ed._panel)
	await _check_fits("Synthesis", _synth._panel)
	await _check_short_panel_is_not_stretched()
	await _check_the_old_arrangement_overflows()

	_synth._panel.queue_free()
	_ed._panel.queue_free()
	_synth.free()
	_ed.free()
	if _fails.is_empty():
		print("panel_fit_check: ALL OK")
		get_tree().quit()
		return
	for f in _fails:
		print("panel_fit_check: FAIL - ", f)
	print("panel_fit_check: %d FAILED" % _fails.size())
	get_tree().quit(1)


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fails.append(what)


func _settle() -> void:
	for i in 6:
		await get_tree().process_frame


## THE TWO HALVES, at every height: the panel is inside the window, and every row can be
## scrolled into it.
func _check_fits(name: String, panel: SidePanel_) -> void:
	var overflowed := false
	for h in HEIGHTS:
		get_tree().root.size = Vector2i(1280, int(h))
		await _settle()
		var vp := get_viewport().get_visible_rect().size
		var bottom: float = panel.position.y + panel.size.y
		_ok(bottom <= vp.y + 1.0,
			"%s at window %d: the panel ends at %.0f in a %.0f-tall viewport - %.0f px of it "
			% [name, h, bottom, vp.y, bottom - vp.y] + "is off the bottom")

		var bar: ScrollBar = panel._scroll.get_v_scroll_bar()
		var hidden: float = maxf(0.0, bar.max_value - bar.page)
		var wants: float = panel.body.get_combined_minimum_size().y
		if wants <= vp.y - panel.position.y - SidePanel_.MARGIN:
			continue        # it fits outright at this height; nothing to reach for
		overflowed = true
		_ok(hidden > 0.0,
			"%s at window %d: the content wants %.0f px in %.0f of room and the panel "
			% [name, h, wants, vp.y] + "cannot scroll at all - those rows are unreachable")
		# ...AND SCROLLING REALLY BRINGS THE LAST ROW IN. A scrollbar with travel on it is not
		# the same claim: the row has to end up inside the panel's own rectangle.
		panel._scroll.scroll_vertical = int(bar.max_value)
		await _settle()
		var last: Control = panel.body.get_child(panel.body.get_child_count() - 1)
		var row: Rect2 = last.get_global_rect()
		var view: Rect2 = panel._scroll.get_global_rect()
		_ok(row.position.y >= view.position.y - 1.0 and row.end.y <= view.end.y + 1.0,
			"%s at window %d: scrolled to the bottom, the last row sits at %.0f..%.0f "
			% [name, h, row.position.y, row.end.y]
			+ "outside the %.0f..%.0f the panel shows" % [view.position.y, view.end.y])
		panel._scroll.scroll_vertical = 0
		await _settle()
	if not overflowed:
		print("panel_fit_check: note - %s fitted at every height swept; only the containment "
			% name + "half was exercised")


## A PANEL THAT FITS IS UNCHANGED. Without this the fix is "every panel is now a full-height
## sidebar", which is a different bug wearing the same patch.
func _check_short_panel_is_not_stretched() -> void:
	get_tree().root.size = Vector2i(1280, 1920)
	await _settle()
	var panel: SidePanel_ = SidePanel_.new(380.0)
	add_child(panel)
	for i in 3:
		var l := Label.new()
		l.text = "row %d" % i
		l.custom_minimum_size = Vector2(0, 24)
		panel.body.add_child(l)
	await _settle()
	var vp := get_viewport().get_visible_rect().size
	_ok(panel.size.y < vp.y * 0.5,
		"a three-row panel was stretched to %.0f px in a %.0f-tall viewport - the fit is "
		% [panel.size.y, vp.y] + "filling the window instead of bounding it")
	_ok(panel.size.y >= SidePanel_.MIN_HEIGHT - 1.0,
		"a three-row panel collapsed to %.0f px, below the %.0f floor"
		% [panel.size.y, SidePanel_.MIN_HEIGHT])
	panel.queue_free()


## THE CONTROL: the arrangement this replaced, given the same content, at the same height.
## It must overflow - otherwise the sweep above is being run in a window that was always big
## enough and proves nothing.
func _check_the_old_arrangement_overflows() -> void:
	get_tree().root.size = Vector2i(1280, 720)
	await _settle()
	var old := PanelContainer.new()
	old.position = Vector2(16, 16)
	old.custom_minimum_size = Vector2(380, 0)
	add_child(old)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	old.add_child(box)
	# The same number of rows the Generative panel carries, at a representative height.
	for i in _ed._panel.body.get_child_count():
		var l := Label.new()
		l.text = "row %d" % i
		l.custom_minimum_size = Vector2(360, 30)
		box.add_child(l)
	old.size = old.get_combined_minimum_size()
	await _settle()
	var vp := get_viewport().get_visible_rect().size
	var bottom := old.position.y + old.size.y
	_ok(bottom > vp.y,
		"the control is wrong - a bare PanelContainer with %d rows ended at %.0f inside a "
		% [box.get_child_count(), bottom] + "%.0f-tall viewport, so it never overflowed and "
		% vp.y + "the sweep above proves nothing")
	print("panel_fit_check: control - the retired arrangement ends at %.0f in a %.0f viewport"
		% [bottom, vp.y])
	old.queue_free()
