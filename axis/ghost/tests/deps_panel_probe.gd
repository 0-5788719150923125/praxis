extends Node

## LOOK AT THE PANEL, in the state you have and in the states you don't.
##
## A probe, not a gate: it writes PNGs and asserts nothing. The home screen's Environment panel is
## most interesting on a machine where ghost is still installing things, or where an install
## failed - and the author's machine has everything. So after the first shot it fakes them: FFmpeg
## mid-download, the voice environment mid-install with no fraction to show, the download
## environment failed with its detail pane open, and the badge every mode shows at the top of the
## frame while anything installs. Layout bugs live in those states (a long phase line drawn over
## its neighbor, a column too narrow for a percentage) and are invisible in the settled one.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/deps_panel_probe.gd 60 found.png busy.png
##
## A BOOT probe: the panel reads its collapsed state through the [Settings] autoload and its live
## state from the Provisioner, and a bare `--script` run has neither. The probe is read-only, so
## the Provisioner never starts a real job here; the jobs drawn are written into it by hand.
##
## With GHOST_PROBE_GPU=1, because it renders: `--headless` is the dummy driver and a viewport
## readback there returns nothing at all.


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 2:
		print("usage: ... deps_panel_probe.gd <found.png> <busy.png>")
		get_tree().quit(2)
		return
	var splash := preload("res://scripts/splash.gd").new()
	get_tree().root.add_child(splash)
	var badge := preload("res://scripts/provision_badge.gd").new()
	get_tree().root.add_child(badge)
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
	for i in 10:   # the rows landed this frame; let them draw before the shutter
		await get_tree().process_frame
	get_tree().root.get_texture().get_image().save_png(args[0])

	var agent: Node = Provision.agent()
	var now := Time.get_ticks_msec()
	agent._jobs["ffmpeg"] = {"action": "install", "phase": "downloading 9.0.2 (1 of 2)",
		"fraction": 0.42, "done": 14 * 1048576, "total": 33 * 1048576, "started": now - 9000,
		"beat": now}
	agent._jobs["voice_venv"] = {"action": "install", "phase": "downloading 19 packages (numpy done)",
		"fraction": -1.0, "done": 0, "total": 0, "started": now - 41000, "beat": now}
	agent._fail["ytdlp_venv"] = {"message": "pypi.org could not be reached (no connection)", "at": now}
	var rows := panel._rows.duplicate(true)
	for r in rows:
		if String(r.get("key", "")) in ["ffmpeg", "voice_venv", "ytdlp_venv"]:
			r["found"] = false
			r["version"] = ""
	panel._apply(rows)
	panel._toggle_detail("ytdlp_venv")
	agent.emit_signal("changed", "ffmpeg")
	for i in 30:
		await get_tree().process_frame
	get_tree().root.get_texture().get_image().save_png(args[1])
	agent._jobs.clear()
	agent._fail.clear()
	print("deps_panel_probe: wrote %s and %s" % [args[0], args[1]])
	get_tree().quit(0)
