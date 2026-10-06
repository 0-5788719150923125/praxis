extends SceneTree

## export_status_check - the export's status line can always be read: it WRAPS within the screen and
## stays ABOVE the bottom-right button row, whatever its length and wherever the row sits.
##
##   godot --headless --path . --script res://tests/export_status_check.gd
##
## Reported 2026-10-06: "the 'Rendering <title> ...' text at the bottom-right is both floating UNDER
## the export button, as well as overflowing the right side of the screen. So I cannot see any
## progress indicator." The line shared the button row unless a mode had claimed the bottom of the
## frame, and an unwrapped Label grows to its right - under the buttons and past the edge. The
## control is the old arrangement (the row's band, no wrapping), which fails both halves.

const ROW_TOP := 68.0           # the button row's top, above the bottom edge (Chrome.ROW_TOP)
const RIGHT := 28.0             # the row's right margin

var _fails := 0


func _init() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _run() -> void:
	var settings := root.get_node_or_null("Settings")
	if settings != null:
		settings.set("_read_only", true)
	# built by hand, its status line moved into the tree on a layer of its own as the exporter's is:
	# the exporter's _ready would clear a live render's override.cfg
	var ex = load("res://scripts/exporter.gd").new()
	ex._build_ui()
	var layer := CanvasLayer.new()
	root.add_child(layer)
	var status: Label = ex._status
	ex.remove_child(status)
	layer.add_child(status)
	root.size = Vector2i(1280, 720)
	var long := ("⏺  Rendering Why You Keep Waking Up At 4AM (And What The Universe Is Trying To Tell You - "
		+ "It Is Usually Your Bladder).mp4 …  12.3%   ⚠ leave the render window visible   ⇪ YouTube: signed in")
	for inset in [0.0, 120.0]:
		ex.set_bottom_inset(inset)
		ex._set_status(long, Color.WHITE)
		await process_frame
		await process_frame
		var vp := root.get_visible_rect().size
		var r := status.get_global_rect()
		_ok(r.end.x <= vp.x - RIGHT + 0.5 and r.position.x >= 0.0,
			"inset %d: a long line stays within the screen (%.0f..%.0f of %.0f)" % [inset, r.position.x, r.end.x, vp.x])
		_ok(r.end.y <= vp.y - ROW_TOP - inset + 0.5,
			"inset %d: it ends above the button row (bottom %.0f, row top %.0f)" % [inset, r.end.y, vp.y - ROW_TOP - inset])
		_ok(r.size.y > status.get_line_height() * 1.5, "inset %d: it wraps onto more lines rather than running off (%.0f px tall)" % [inset, r.size.y])
	ex.set_bottom_inset(0.0)
	ex._set_status("✓  Saved  /a/short.mp4", Color.WHITE)
	await process_frame
	await process_frame
	var one := status.get_global_rect()
	_ok(one.size.y <= 36.5 and absf(one.end.y - (root.get_visible_rect().size.y - 74.0)) < 1.0,
		"after a long line, a short one shrinks back to its own row just above the buttons (%.0f px tall, bottom %.0f)"
		% [one.size.y, one.end.y])

	# THE CONTROL: the old arrangement - the row's own band, unwrapped - fails both halves
	status.autowrap_mode = TextServer.AUTOWRAP_OFF
	status.grow_horizontal = Control.GROW_DIRECTION_END
	status.grow_vertical = Control.GROW_DIRECTION_END
	status.offset_left = -616
	status.offset_top = -64
	status.offset_right = -160
	status.offset_bottom = -28
	status.text = long
	await process_frame
	await process_frame
	var old := status.get_global_rect()
	var vp2 := root.get_visible_rect().size
	_ok(old.end.x > vp2.x and old.end.y > vp2.y - ROW_TOP,
		"control: the old line ran past the edge (%.0f > %.0f) and into the row" % [old.end.x, vp2.x])
	layer.queue_free()
	ex.free()
	print("export_status_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	quit(0 if _fails == 0 else 1)
