extends Node

## splash_agents_check - that a mode which only works with AI agents cannot be entered without
## them, and says which agents it wants without naming any in code.
##
## Run: tests/run_boot_probe.sh tests/splash_agents_check.gd 90
##
## THE ASK (2026-10-05): "gate the buttons on the home page, such that those modes could not even
## be accessed without agents... greying-out the buttons, perhaps with a mouseover tooltip that
## names the missing dependencies... I wouldn't specifically name 'Claude' or 'Codex' - I would
## list the supported writers, and suggest installing one or several of them."
##
## It drives the REAL main scene, and an agent is made missing where a launch looks for it - the
## resolution cache in [Deps] - so what is tested is the path a job would take, not a seam. Four
## states, each checked against an expectation worked out here from the registries rather than
## from the splash: this machine as it is, no agent at all, a writer but no painter, and back
## again through the Environment panel's real rescan. In the gated states it also hovers the
## grayed button (the tooltip must show on a DISABLED button) and clicks it (the mode must not
## open).
##
## THE ASSISTANT DROPDOWN, same states, same rule ("If no AI is installed, we cannot support that
## feature and should definitely gate on it"): a CLI that is not installed is listed but not
## choosable, with none installed the dropdown itself is grayed out, and whatever is stored
## [method Splash.assistant_backend] reads as Off for a CLI that is not there - the stored choice
## itself untouched. The choice is set in memory only: a probe's Settings never reach the disk.

const ROLES := {"writer": preload("res://scripts/text_gen.gd"), "painter": preload("res://scripts/image_gen.gd")}
## Every program a writer or painter backend resolves. If a backend is added on another CLI, the
## "no agent at all" state fails loudly naming it, rather than testing a machine that still has one.
const AGENT_PROGRAMS := ["claude", "codex"]
const ASSIST := preload("res://scripts/assistant_backends.gd")

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var main: Node = preload("res://scenes/main.tscn").instantiate()
	add_child(main)
	for _i in 10:
		await get_tree().process_frame
	var splash: Node = _find(main, "Splash")
	if splash == null:
		_fail("the splash never came up")
		return _report()
	var env: DepsPanel = splash._env
	# The panel's first probe resolves every program on a thread; poisoning the cache while it
	# runs would race it, so start from its answer.
	if env._rows.is_empty():
		await _probe_landed(env)

	var gates: Array = splash._gates
	var tarot: Button = null
	for g in gates:
		if String((g["button"] as Button).text).begins_with("Tarot"):
			tarot = g["button"]
			_check(Array(g["needs"]) == ["writer", "painter"], "the Tarot row needs a writer and a painter (%s)" % [g["needs"]])
	if tarot == null:
		_fail("no gated Tarot row on the home screen")
		return _report()
	var ungated: Array = []
	for b in _buttons(splash):
		if not gates.any(func(g: Dictionary) -> bool: return g["button"] == b):
			ungated.append(b)
	_check(ungated.size() >= 5, "the other modes are on the screen (%d)" % ungated.size())

	# --- 1. this machine as it is
	print("splash_agents: as installed - writers %s, painters %s" % [_have("writer"), _have("painter")])
	_state(splash, tarot, ungated, "as installed")

	# --- 2. no agent at all
	for prog in AGENT_PROGRAMS:
		Deps._resolved[prog] = ""
	for role in ROLES:
		var reg: GDScript = ROLES[role]
		for k in reg.REGISTRY:
			_check(not reg.make(k).available(), "with %s gone, the %s %s is gone too" % [AGENT_PROGRAMS, role, k])
	env.probed.emit()
	_state(splash, tarot, ungated, "no agents")
	var tip := tarot.tooltip_text
	_check(tip.contains("No AI writer or painter is installed."), "the tooltip says what is missing")
	for role in ROLES:
		for label in (ROLES[role] as GDScript).LABELS.values():
			_check(tip.contains(_whole(label)), "the tooltip lists the supported %s %s" % [role, label])
	var both := ""
	for k in TextGen.REGISTRY:
		if ImageGen.REGISTRY.has(k):
			both = String(TextGen.LABELS[k])
	if not both.is_empty():
		_check(tip.contains("%s can do both." % _whole(both)), "the tooltip names the agent that does both")
	_check(tip.contains("Install one or several of them"), "the tooltip suggests installing one or several")
	print("splash_agents: tooltip with no agents:\n%s" % tip)
	if not await _hover_and_click(splash, tarot):
		return _report()            # the mode opened: the rest has no home screen to ask
	var opt: OptionButton = splash._asst_option
	var atip := opt.tooltip_text
	_check(atip.contains("No AI assistant is installed"), "the Assistant tooltip says no AI is installed")
	for k in ASSIST.REGISTRY:
		_check(atip.contains(_whole(ASSIST.label(k))), "the Assistant tooltip lists the supported %s" % ASSIST.label(k))
	_check(atip.contains("Install one or several of them"), "the Assistant tooltip suggests installing one or several")
	_check(await _hover(opt) == opt.tooltip_text, "the grayed Assistant dropdown shows its tooltip on hover")
	_press(opt)
	await get_tree().process_frame
	_check(not opt.get_popup().visible, "a click on the grayed Assistant dropdown opens nothing")
	await _hover(null)

	# --- 3. a writer, but nothing that paints
	Deps.forget_all()
	Deps._resolved["codex"] = ""
	if _have("writer").is_empty():
		print("splash_agents: no writer here but Codex - skipping the writer-without-painter state")
	else:
		env.probed.emit()
		_state(splash, tarot, ungated, "writer, no painter")
		tip = tarot.tooltip_text
		_check(tip.contains("No AI painter is installed."), "the tooltip names the painter as missing")
		_check(not tip.contains("Supported writers"), "a filled role is not listed as missing")
		_check(not tip.contains("can do both"), "one role missing is not 'both'")
		print("splash_agents: tooltip with a writer only:\n%s" % tip)

	# --- 4. back, through the panel's own rescan
	var landed := [false]
	env.probed.connect(func() -> void: landed[0] = true, CONNECT_ONE_SHOT)
	env._rescan()
	var t0 := Time.get_ticks_msec()
	while not landed[0] and Time.get_ticks_msec() - t0 < 30000:
		await get_tree().process_frame
	_check(landed[0], "the rescan's probe landed")
	_state(splash, tarot, ungated, "after rescan")
	_report()


## The gate's whole contract in one state: lit exactly when every role has an installed agent,
## worked out here from the registries; the caption naming no agent; the other modes untouched.
func _state(splash: Node, tarot: Button, ungated: Array, label: String) -> void:
	var lit := not _have("writer").is_empty() and not _have("painter").is_empty()
	var caption := ""
	for g in splash._gates:
		if g["button"] == tarot:
			caption = (g["uses"] as Label).text
	print("splash_agents: %-18s Tarot %s - \"%s\"" % [label, "lit" if not tarot.disabled else "grayed", caption])
	_check(tarot.disabled == not lit, "%s: Tarot is %s" % [label, "lit" if lit else "grayed out"])
	_check(caption.begins_with("uses" if lit else "needs"), "%s: the caption says it %s" % [label, "uses" if lit else "needs"])
	for role in ROLES:
		for lbl in (ROLES[role] as GDScript).LABELS.values():
			_check(not caption.contains(String(lbl)), "%s: the caption names no agent (%s)" % [label, lbl])
	_check(not tarot.tooltip_text.is_empty(), "%s: Tarot has a tooltip" % label)
	if lit:
		for role in ROLES:
			for lbl in _labels(role, _have(role)):
				_check(tarot.tooltip_text.contains(_whole(lbl)), "%s: the tooltip names the installed %s %s" % [label, role, lbl])
	# Boot re-flows every tooltip on hover, folding single newlines into the line: what the splash
	# writes has to be paragraphs, or a list arrives on screen run together.
	var paras := tarot.tooltip_text.split("\n\n")
	_check(not paras.is_empty() and Array(paras).all(func(p: String) -> bool: return not p.contains("\n")),
		"%s: the tooltip is written as paragraphs" % label)
	_check(Boot.wrap_tip(tarot.tooltip_text).split("\n\n").size() == paras.size(),
		"%s: the tooltip keeps its paragraphs through Boot's re-flow" % label)
	for b in ungated:
		_check(not (b as Button).disabled, "%s: %s is not gated" % [label, (b as Button).text.strip_edges()])
	_assistant_state(splash, label)


## The Assistant dropdown against the CLIs worked out here: each listed, choosable only when
## installed, the whole dropdown grayed with none; and for every possible choice, the backend that
## would actually be used.
func _assistant_state(splash: Node, label: String) -> void:
	var opt: OptionButton = splash._asst_option
	var have := _assistants()
	var keys: Array = [""] + ASSIST.REGISTRY.keys()
	_check(opt.item_count == keys.size(), "%s: the dropdown lists Off and every assistant" % label)
	_check(not opt.is_item_disabled(0), "%s: Off can always be chosen" % label)
	for i in range(1, mini(keys.size(), opt.item_count)):
		var k := String(keys[i])
		var here := have.has(k)
		_check(opt.is_item_disabled(i) == not here, "%s: %s is %s" % [label, k, "choosable" if here else "not choosable"])
		_check(opt.get_item_text(i).contains("(not installed)") == not here, "%s: %s says whether it is installed" % [label, k])
	_check(opt.disabled == have.is_empty(), "%s: the dropdown is %s" % [label, "grayed out" if have.is_empty() else "live"])
	var paras := opt.tooltip_text.split("\n\n")
	_check(Boot.wrap_tip(opt.tooltip_text).split("\n\n").size() == paras.size(),
		"%s: the Assistant tooltip keeps its paragraphs through Boot's re-flow" % label)
	var was: Variant = Settings.read("assistant", "backend", "")
	for k in keys:
		Settings.write("assistant", "backend", k)
		var usable := have.has(k)
		_check(Splash.assistant_choice() == k, "%s: the choice '%s' is kept as chosen" % [label, k])
		_check(Splash.assistant_backend() == (k if usable else ""),
			"%s: chosen '%s', the backend is '%s'" % [label, k, Splash.assistant_backend()])
		_check(Splash.assistant_gap().is_empty() == usable, "%s: chosen '%s', the gap is '%s'" % [label, k, Splash.assistant_gap()])
	Settings.write("assistant", "backend", was)


## Point at [param c] (null: move off everything) and return the tooltip that comes up, "" if none.
func _hover(c: Control) -> String:
	var at := c.get_global_rect().get_center() if c != null else Vector2(2, 2)
	for i in 4:
		var mm := InputEventMouseMotion.new()
		mm.position = at + Vector2(i, 0)
		mm.global_position = mm.position
		get_viewport().push_input(mm, true)
		await get_tree().process_frame
	if c == null:
		return ""
	_check(get_viewport().gui_get_hovered_control() == c, "the pointer is over %s" % c.get_class())
	await get_tree().create_timer(float(ProjectSettings.get_setting("gui/timers/tooltip_delay_sec", 0.5)) + 0.7).timeout
	return _tooltip_text(get_tree().root)


## A left click on [param c], as the pointer makes one.
func _press(c: Control) -> void:
	var at := c.get_global_rect().get_center()
	for pressed in [true, false]:
		var mb := InputEventMouseButton.new()
		mb.button_index = MOUSE_BUTTON_LEFT
		mb.pressed = pressed
		mb.position = at
		mb.global_position = at
		get_viewport().push_input(mb, true)


## A disabled button must still SHOW its tooltip - the whole explanation lives there - and a click
## on it must not open the mode. False when it did.
func _hover_and_click(splash: Node, tarot: Button) -> bool:
	var at := tarot.get_global_rect().get_center()
	for i in 4:
		var mm := InputEventMouseMotion.new()
		mm.position = at + Vector2(i, 0)
		mm.global_position = mm.position
		get_viewport().push_input(mm, true)
		await get_tree().process_frame
	_check(get_viewport().gui_get_hovered_control() == tarot, "the pointer is over the Tarot button")
	await get_tree().create_timer(float(ProjectSettings.get_setting("gui/timers/tooltip_delay_sec", 0.5)) + 0.7).timeout
	var shown := _tooltip_text(get_tree().root)
	_check(shown == tarot.tooltip_text, "the grayed button shows its tooltip on hover (%s)" % ("shown" if not shown.is_empty() else "nothing shown"))
	_check(shown.split("\n\n").size() >= 4, "the tooltip on screen keeps its paragraphs (%d)" % shown.split("\n\n").size())
	print("splash_agents: on screen:\n%s" % shown)
	for pressed in [true, false]:
		var mb := InputEventMouseButton.new()
		mb.button_index = MOUSE_BUTTON_LEFT
		mb.pressed = pressed
		mb.position = at
		mb.global_position = at
		get_viewport().push_input(mb, true)
		await get_tree().process_frame
	for _i in 4:
		await get_tree().process_frame
	var stayed := is_instance_valid(splash) and splash.is_inside_tree() and not splash.is_queued_for_deletion()
	_check(stayed, "a click on the grayed button does not leave the home screen")
	var away := InputEventMouseMotion.new()
	away.position = Vector2(2, 2)
	away.global_position = away.position
	get_viewport().push_input(away, true)
	await get_tree().process_frame
	return stayed


## The text of a tooltip on screen, "" when none is.
func _tooltip_text(n: Node) -> String:
	for c in n.get_children(true):
		if c is PopupPanel and (c as PopupPanel).visible:
			for l in c.get_children(true):
				if l is Label:
					return (l as Label).text
		var t := _tooltip_text(c)
		if not t.is_empty():
			return t
	return ""


## [param label] as the tooltip sets it: one unbreakable run, so the re-flow cannot split it.
func _whole(label: String) -> String:
	return label.replace(" ", "\u00a0")


## The installed assistant CLIs, asked of the resolver directly.
func _assistants() -> Array:
	return ASSIST.REGISTRY.keys().filter(func(k: String) -> bool: return Deps.has(ASSIST.dep(k)))


## The installed agents of [param role], asked of each backend directly.
func _have(role: String) -> Array:
	var reg: GDScript = ROLES[role]
	return (reg.REGISTRY as Dictionary).keys().filter(func(k: String) -> bool: return reg.make(k).available())


func _labels(role: String, keys: Array) -> Array:
	var labels: Dictionary = (ROLES[role] as GDScript).LABELS
	return keys.map(func(k: String) -> String: return String(labels.get(k, k)))


func _probe_landed(env: DepsPanel) -> void:
	var t0 := Time.get_ticks_msec()
	while env._rows.is_empty() and Time.get_ticks_msec() - t0 < 30000:
		await get_tree().process_frame


## Every mode button: a Button whose text ends in the play glyph.
func _buttons(n: Node) -> Array:
	var out: Array = []
	for c in n.get_children():
		if c is Button and String((c as Button).text).ends_with("▶"):
			out.append(c)
		out.append_array(_buttons(c))
	return out


func _check(ok: bool, what: String) -> void:
	if not ok:
		_fail(what)


func _report() -> void:
	if _fails.is_empty():
		print("splash_agents: ALL OK")
	else:
		for f in _fails:
			print("splash_agents: FAILED - %s" % f)
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(_fails.size())


func _fail(msg: String) -> void:
	_fails.append(msg)


func _find(n: Node, cls: String) -> Node:
	if n.get_script() != null and String(n.get_script().resource_path).get_file() == cls.to_snake_case() + ".gd":
		return n
	for c in n.get_children():
		var r := _find(c, cls)
		if r != null:
			return r
	return null
