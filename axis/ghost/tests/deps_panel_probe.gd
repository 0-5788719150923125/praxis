extends Node

## LOOK AT THE PANEL, in both the state you have and the state you don't.
##
## A probe, not a gate: it writes PNGs and asserts nothing. It exists because the
## home screen's Environment panel is only interesting on a machine that is MISSING
## something, and the author's machine has everything - so the second shot fakes the
## failure (rows knocked to not-found, a detail pane opened) and renders it. Every
## layout bug this has caught so far was in that state and invisible in the other:
## a verdict long enough to draw over the buttons beside it, a right-hand column too
## narrow for its own text.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/deps_panel_probe.gd 60 found.png missing.png
##
## A BOOT probe: the panel reads its collapsed state through the [Settings] autoload, and a bare
## `--script` run has no autoloads - it failed to compile and then sat idle instead of quitting.
##
## With GHOST_PROBE_GPU=1, because it renders: `--headless` is the dummy driver and a
## viewport readback there returns nothing at all.

const FAKE_MISSING := ["ffmpeg", "ffprobe", "jsruntime"]


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 2:
		print("usage: ... deps_panel_probe.gd <found.png> <missing.png>")
		get_tree().quit(2)
		return
	var splash := preload("res://scripts/splash.gd").new()
	get_tree().root.add_child(splash)
	var panel: DepsPanel = null
	# The probe runs on a thread; wait for it rather than guessing a frame count.
	for i in 600:
		await get_tree().process_frame
		if panel == null:
			for c in splash.get_children():
				if c is DepsPanel:
					panel = c
		if panel != null and not panel._rows.is_empty():
			break
	if panel == null:
		print("deps_panel_probe: no panel on the splash")
		get_tree().quit(1)
		return
	get_tree().root.get_texture().get_image().save_png(args[0])

	var rows := panel._rows.duplicate(true)
	for r in rows:
		if FAKE_MISSING.has(String(r.get("key", ""))):
			r["found"] = false
			r["version"] = ""
			r["note"] = "not found"
	panel._apply(rows)
	panel._toggle_detail(FAKE_MISSING[0])
	for i in 30:
		await get_tree().process_frame
	get_tree().root.get_texture().get_image().save_png(args[1])
	print("deps_panel_probe: wrote %s and %s" % [args[0], args[1]])
	get_tree().quit(0)
