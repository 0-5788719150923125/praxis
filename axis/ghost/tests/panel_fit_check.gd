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
## AND IT KEEPS ITS WIDTH. The panel is as wide as its widest row, so one control asking for
## more pushes the whole panel across the stage: opening a tarot show made it 1080 px of a
## 1920 window, because its episode list sized itself to the longest episode title. Checked on
## every panel, the tarot one holding real long titles; its control is that list sized to its
## items again, which must blow the width out or the titles were never in it.
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
var _tarot: TarotEditor


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

	_tarot = TarotEditor.new()
	_tarot._build_panel()
	_tarot.remove_child(_tarot._panel)
	add_child(_tarot._panel)
	_tarot.remove_child(_tarot._repace_timer)
	add_child(_tarot._repace_timer)
	_tarot_episodes()
	# a voice list as the host fills it - its longest name is what a picker sized to its items
	# would be as wide as
	for e in [_ed, _tarot]:
		(e as GenerativeEditor)._voices.add_item("libritts high  (downloads)")
		(e as GenerativeEditor)._voices.add_item("en_GB southern english female medium  (downloads)")

	await _check_fits("Generative", _ed._panel)
	await _check_fits("Synthesis", _synth._panel)
	await _check_fits("Tarot", _tarot._panel)
	await _check_width("Generative", _ed._panel)
	await _check_width("Synthesis", _synth._panel)
	await _check_width("Tarot", _tarot._panel)
	await _check_long_items_would_widen()
	await _check_episode_entry_follows()
	await _check_short_panel_is_not_stretched()
	await _check_the_old_arrangement_overflows()
	await _check_wheel_skips_sliders(_ed._panel)

	_synth._panel.queue_free()
	_ed._panel.queue_free()
	_tarot._panel.queue_free()
	_synth.free()
	_ed.free()
	_tarot.free()
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


## THE WHEEL IS THE PANEL'S: "scrolling through the UI is constantly, accidentally scrolling
## values for options". Every slider on a real panel, and one added after the panel is built
## (as a voice tab adds them), is drag-only. Asserted on the flag the engine reads: a wheel
## event pushed through the viewport in a headless probe never reached a slider, so a check of
## the value passed with the fix removed.
func _check_wheel_skips_sliders(panel: SidePanel_) -> void:
	var late := HSlider.new()
	late.max_value = 100.0
	late.value = 50.0
	panel.body.add_child(late)
	await _settle()
	var sliders: Array = panel.find_children("*", "Slider", true, false)
	_ok(sliders.size() > 5, "found only %d sliders on the panel" % sliders.size())
	var loose: Array = []
	for sl in sliders:
		if (sl as Slider).scrollable:
			loose.append((sl as Slider).name)
	_ok(loose.is_empty(), "%d slider(s) still take the wheel: %s" % [loose.size(), loose.slice(0, 5)])
	late.queue_free()


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
		# THE BAR HAS CLEAR SPACE BESIDE IT, and CLEARANCE is the measurement rather than
		# non-overlap. Measured: a ScrollContainer does reserve the bar's width, so the content
		# already stopped exactly AT the bar's left edge - touching it, with nothing between
		# them, which is what the right-aligned value readouts were running into. An assertion
		# that only forbade overlap was therefore true before the gutter existed and passed on
		# a build with no gutter at all; it proved nothing. This one fails on that build.
		var bar_rect: Rect2 = bar.get_global_rect()
		var content: Rect2 = panel.body.get_global_rect()
		_ok(bar.visible, "the control is wrong - the bar is not showing, so there is nothing "
			+ "for the content to be crowded by")
		var gap: float = bar_rect.position.x - content.end.x
		_ok(gap >= SidePanel_.GUTTER - 1.0,
			"%s at window %d: only %.0f px between the last column of text and the scrollbar "
			% [name, h, gap] + "(want at least %.0f)" % SidePanel_.GUTTER)
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


## Two episodes on disk, under a root of the gate's own, with titles as long as a writer makes
## them, and the tarot panel pointed at them - its episode list and its rows then hold them.
func _tarot_episodes() -> void:
	TarotEpisode.root = "user://panel_fit_check_tarot"
	for seed in [11, 12]:
		var ep := TarotEpisode.open("fit-check", seed)
		ep.write_json("plan", {"episode_title": "Why You Keep Waking Up At 4AM (And What The Universe Is "
			+ "Trying To Tell You - It Is Usually Your Bladder) #%d" % seed,
			"spread": {"positions": [{"name": "Past"}, {"name": "Present"}, {"name": "Future"}]}})
	_tarot._knobs["show"] = "fit-check"
	_tarot._knobs["seed"] = 11
	_tarot._open_episode()


## AT ITS DECLARED WIDTH, whatever its rows hold - and whatever its buttons say: Play is
## "Resume ●" while a stale reading is paused, the widest it gets. A failure names what is too wide.
func _check_width(name: String, panel: SidePanel_) -> void:
	get_tree().root.size = Vector2i(1280, 1080)
	var go: Button = null
	for b in panel.find_children("*", "Button", true, false):
		if (b as Button).text == "Play":
			go = b
	if go != null:
		go.text = "Resume ●"
	await _settle()
	var w := panel.get_combined_minimum_size().x
	var names := PackedStringArray()
	for c in SidePanel_.overwide(panel):
		names.append("%s (%d px)" % [SidePanel_.describe(c as Control), int((c as Control).get_combined_minimum_size().x)])
	_ok(w <= panel.custom_minimum_size.x + 0.5, "%s: the panel is %.0f px wide, not %.0f - %s"
		% [name, w, panel.custom_minimum_size.x, ", ".join(names)])
	if go != null:
		go.text = "Play"


## THE EPISODE'S ENTRY FOLLOWS ITS PLAN: an episode made while it was open (2026-10-05: "I just
## clicked on Generate and all components were successful, but that picker entry still says (Not
## made yet)... I have to pick a DIFFERENT entry, then return to that one"). The control is the
## entry before the panel looks again - the stale one the report was about.
func _check_episode_entry_follows() -> void:
	var pick: OptionButton = _tarot._episode_pick
	var fresh := TarotEpisode.open("fit-check", 13)
	DirAccess.remove_absolute(fresh.file_of("plan"))       # a run before this one made it
	_tarot._knobs["seed"] = 13
	_tarot._open_episode()
	var at := _tarot._episode_seeds.find(13)
	_ok(at >= 0 and pick.get_item_text(at).ends_with("(not made yet)"),
		"a fresh episode is listed as not made yet (%s)" % (pick.get_item_text(at) if at >= 0 else "missing"))
	TarotEpisode.open("fit-check", 13).write_json("plan", {"episode_title": "Made While You Watched",
		"spread": {"positions": [{"name": "Past"}]}})
	_ok(pick.get_item_text(at).ends_with("(not made yet)"), "control: the entry is stale until the panel looks again")
	_tarot._refresh_rows()
	_ok(pick.get_item_text(at).ends_with("Made While You Watched") and pick.selected == at
		and pick.text.ends_with("Made While You Watched"),
		"its entry takes the title once the plan lands, shown on the button (%s)" % pick.text)
	DirAccess.remove_absolute(fresh.file_of("plan"))
	_tarot._knobs["seed"] = 11
	_tarot._open_episode()


## THE CONTROL for the tarot panel: its episode list sized to its items again must push the
## panel out, or the long titles never reached it and the width check above proved nothing.
func _check_long_items_would_widen() -> void:
	var pick: OptionButton = _tarot._episode_pick
	_ok(pick.item_count >= 2 and pick.get_item_text(0).length() > 80,
		"the control is wrong - the tarot episode list holds %d item(s), not the long titles" % pick.item_count)
	pick.fit_to_longest_item = true
	await _settle()
	var w := _tarot._panel.get_combined_minimum_size().x
	_ok(w > _tarot._panel.custom_minimum_size.x + 50.0,
		"the control is wrong - an episode list sized to its longest title left the panel at %.0f px" % w)
	print("panel_fit_check: control - an episode list sized to its longest title makes the tarot panel %.0f px" % w)
	pick.fit_to_longest_item = false
	await _settle()
