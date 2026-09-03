extends Node

## splash_furniture_check - that the shared bottom-right furniture does not sit ON the
## home screen's Environment panel, and that the ⤓ export button is not there at all.
##
## Run: tests/run_boot_probe.sh tests/splash_furniture_check.gd 90
##
## THE COMPLAINT: "The console and download buttons are currently rendered overtop of the
## environment box, on the home screen. That should not be happening, and the download
## button has no business even being on the home screen at all."
##
## Both halves are geometry, so both are measured rather than eyeballed: the toggles'
## screen rects must not intersect the panel's, and the export button must be invisible.
## The overlap half matters more than it looks - the toggles live on a HIGHER CanvasLayer
## than the splash, so a toggle drawn over an Environment row also swallows its clicks.
##
## It drives the REAL main scene rather than assembling a Chrome and a splash by hand.
## Hand-assembly would test a construction the app never performs, and the bug being
## gated is precisely one of composition.

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var main: Node = preload("res://scenes/main.tscn").instantiate()
	add_child(main)
	# Several frames: Control layout resolves over one, and the deps panel's probe
	# thread rewrites its rows (and therefore its height) shortly after.
	for _i in 20:
		await get_tree().process_frame

	var chrome: Node = get_tree().get_first_node_in_group("ghost_chrome")
	if chrome == null:
		_fail("no Chrome in the tree at all")
		return _report()
	var splash := _find(main, "Splash")
	if splash == null:
		_fail("the splash never came up")
		return _report()
	var env: Control = splash._env
	if env == null or not is_instance_valid(env):
		_fail("the splash has no Environment panel")
		return _report()

	var box := env.get_global_rect()
	print("splash_furniture: Environment panel %s, chrome claimed %.0f px" % [box, chrome.bottom_inset])
	if chrome.bottom_inset <= 0.0:
		_fail("the splash claimed no room, so nothing stepped over the panel")

	# --- the export button must not be on the home screen at all
	var btn: Control = chrome.exporter._btn
	if btn != null and btn.visible:
		_fail("the ⤓ export button is visible on the home screen")
	print("splash_furniture: export button visible=%s, suppressed=%s" % [
		btn != null and btn.visible, chrome.exporter.suppressed])

	# --- and nothing that IS on screen may sit on the panel
	for pair in [["console >_", chrome.console._toggle],
			["assistant 💬", chrome.assistant._toggle_btn],
			["export ⤓", btn]]:
		var name: String = pair[0]
		var c: Control = pair[1]
		if c == null or not is_instance_valid(c) or not c.visible:
			print("splash_furniture: %-14s not on screen" % name)
			continue
		var r := c.get_global_rect()
		var hit := r.intersects(box)
		print("splash_furniture: %-14s %s%s" % [name, r, "  <-- ON THE PANEL" if hit else "  clear"])
		if hit:
			_fail("the %s toggle overlaps the Environment panel" % name)

	# --- the claim must TRACK the panel, not be a number written down once
	var before: float = chrome.bottom_inset
	env._toggle_collapsed()
	for _i in 6:
		await get_tree().process_frame
	var after: float = chrome.bottom_inset
	print("splash_furniture: collapse toggled the claim %.0f -> %.0f px" % [before, after])
	if is_equal_approx(before, after):
		_fail("collapsing the panel did not move the claim - it is not tracking the size")
	env._toggle_collapsed()          # leave the user's remembered state alone
	for _i in 6:
		await get_tree().process_frame

	# --- THE HANDOVER, which is why the claims are keyed at all.
	#
	# A mode button calls into main FIRST and queue_frees the splash SECOND, so the
	# outgoing splash's _exit_tree runs after the incoming mode's _ready. Stand a second
	# claimant up the way a mode does, then dismiss the splash, and the mode's claims must
	# still be standing. Before the keys, the splash's release wrote 0 / false outright and
	# would have handed the export button straight back on top of Masking's own one.
	chrome.claim_bottom(&"mode", 240.0)
	chrome.suppress_export(&"mode")
	splash.queue_free()
	for _i in 6:
		await get_tree().process_frame
	print("splash_furniture: after handover - claimed %.0f px, export suppressed=%s" % [
		chrome.bottom_inset, chrome.exporter.suppressed])
	if not is_equal_approx(chrome.bottom_inset, 240.0):
		_fail("the departing splash clobbered the incoming mode's bottom claim (%.0f px)"
			% chrome.bottom_inset)
	if not chrome.exporter.suppressed:
		_fail("the departing splash handed the export button back over the incoming mode")
	# and once the mode goes too, the furniture comes back
	chrome.release_bottom(&"mode")
	chrome.release_export(&"mode")
	for _i in 3:
		await get_tree().process_frame
	if not is_equal_approx(chrome.bottom_inset, 0.0) or chrome.exporter.suppressed:
		_fail("releasing the last claim did not restore the furniture (%.0f px, suppressed=%s)"
			% [chrome.bottom_inset, chrome.exporter.suppressed])
	_report()


func _report() -> void:
	if _fails.is_empty():
		print("splash_furniture: ALL OK")
	else:
		for f in _fails:
			print("splash_furniture: FAILED - %s" % f)
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
