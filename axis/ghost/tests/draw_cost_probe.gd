extends Node

## draw_cost_probe - THE TWO NUMBERS THE AUDIT RESTS ON, measured rather than reasoned about.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/draw_cost_probe.gd 300
##
## (1) WHAT ONE CANVAS DRAW CALL COSTS FROM GDSCRIPT. The catalogue sweep says a scene's frame
## tracks its DRAW-CALL COUNT far better than its geometry (starfield/static: 3029 calls,
## 194k prims, 56 ms; rocks: 3 calls, 19k prims, 24 ms). That is a claim about the per-call
## GDScript -> RenderingServer round trip, and it is testable directly: draw the SAME triangles
## as N separate calls and as one batched triangle array, on the same canvas, in the same frame.
##
## (2) WHETHER A NESTED SubViewport STILL RENDERS WHEN ITS PARENT IS STOPPED. main.gd's stage
## governor throttles by disabling the STAGE (process off, render target UPDATE_DISABLED). The
## comic vehicle puts its live panels in SubViewports NESTED INSIDE that stage, each set to
## UPDATE_ALWAYS. If a nested target ignores its parent's state, the governor cannot throttle
## the comic at all - it is a real question about engine semantics and it decides whether the
## governor is a fix or a placebo here.
##
## Both are measured with the engine's own per-viewport render timers and the frame counters, on
## the real GPU. Nothing here touches ghost's scenes; it is a control for them.

const REPS := 6
const SHAPES := [200, 600, 1800]
const W := 1280
const H := 720


class Painter:
	extends Node2D
	var mode := "circle"
	var n := 200
	var last_us := 0

	func _draw() -> void:
		var t0 := Time.get_ticks_usec()
		match mode:
			"circle":
				# what Layer.Stars / Fireflies / Snow do, one shape at a time
				for i in n:
					var c := Vector2(float(i % 40) * 30.0 + 20.0, float(i / 40) * 30.0 + 20.0)
					draw_circle(c, 9.0, Color(0.6, 0.7, 0.9, 0.8))
			"poly":
				# what a scene drawing quads one at a time does
				for i in n:
					var c := Vector2(float(i % 40) * 30.0 + 20.0, float(i / 40) * 30.0 + 20.0)
					draw_colored_polygon(PackedVector2Array([
						c, c + Vector2(18, 0), c + Vector2(18, 18), c + Vector2(0, 18)]),
						Color(0.6, 0.7, 0.9, 0.8))
			"poly_aa":
				# GhostScene.fill_aa: the same fill PLUS an antialiased outline stroke
				for i in n:
					var c := Vector2(float(i % 40) * 30.0 + 20.0, float(i / 40) * 30.0 + 20.0)
					var p := PackedVector2Array([
						c, c + Vector2(18, 0), c + Vector2(18, 18), c + Vector2(0, 18)])
					draw_colored_polygon(p, Color(0.6, 0.7, 0.9, 0.8))
					var ring := p.duplicate()
					ring.append(p[0])
					draw_polyline(ring, Color(0.6, 0.7, 0.9, 0.8), 1.0, true)
			"batch":
				# the SAME quads as one TriBatch submit - the geometry is identical, only the
				# number of GDScript -> server round trips differs
				var tb := TriBatch.new()
				for i in n:
					var c := Vector2(float(i % 40) * 30.0 + 20.0, float(i / 40) * 30.0 + 20.0)
					tb.quad(c, c + Vector2(18, 0), c + Vector2(18, 18), c + Vector2(0, 18),
						Color(0.6, 0.7, 0.9, 0.8))
				tb.flush(self)
			"batch_raw":
				# the floor: the arrays built with no helper, one server call
				var pts := PackedVector2Array()
				var cols := PackedColorArray()
				var idx := PackedInt32Array()
				for i in n:
					var c := Vector2(float(i % 40) * 30.0 + 20.0, float(i / 40) * 30.0 + 20.0)
					var b := pts.size()
					pts.append(c)
					pts.append(c + Vector2(18, 0))
					pts.append(c + Vector2(18, 18))
					pts.append(c + Vector2(0, 18))
					for _k in 4:
						cols.append(Color(0.6, 0.7, 0.9, 0.8))
					idx.append_array([b, b + 1, b + 2, b, b + 2, b + 3])
				RenderingServer.canvas_item_add_triangle_array(
					get_canvas_item(), idx, pts, cols)
		last_us = Time.get_ticks_usec() - t0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	print("draw_cost_probe: renderer %s, display %s" % [
		RenderingServer.get_current_rendering_method(), DisplayServer.get_name()])
	await _call_cost()
	await _nested_viewport()
	for _i in 4:
		await get_tree().process_frame
	get_tree().quit(0)


# --- (1) the per-call cost ---------------------------------------------------------------

func _call_cost() -> void:
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var p := Painter.new()
	vp.add_child(p)
	print("=== (1) GDScript draw cost for IDENTICAL geometry, %dx%d, mean of %d frames ===" % [W, H, REPS])
	print("    mode        shapes   _draw ms   us/shape   draw calls   prims")
	for n in SHAPES:
		for mode in ["circle", "poly", "poly_aa", "batch", "batch_raw"]:
			p.mode = mode
			p.n = n
			var us := 0.0
			var calls := 0.0
			var prims := 0.0
			for _r in REPS:
				p.queue_redraw()
				await get_tree().process_frame
				us += float(p.last_us)
				calls += Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)
				prims += Performance.get_monitor(Performance.RENDER_TOTAL_PRIMITIVES_IN_FRAME)
			var ms := us / float(REPS) / 1000.0
			print("    %-11s %6d   %8.2f   %8.2f   %10.0f   %7.0f" % [
				mode, n, ms, us / float(REPS) / float(n), calls / REPS, prims / REPS])
	vp.queue_free()
	await get_tree().process_frame


# --- (2) does a nested SubViewport obey its parent? ---------------------------------------

func _nested_viewport() -> void:
	print("=== (2) a SubViewport NESTED in a stopped parent viewport (the stage governor's lever) ===")
	var stage := SubViewport.new()
	stage.size = Vector2i(W, H)
	stage.transparent_bg = false
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)
	# a panel viewport inside it, exactly as ComicVehicle._build_slots makes one
	var panel := SubViewport.new()
	panel.size = Vector2i(512, 512)
	panel.transparent_bg = false
	panel.disable_3d = true
	panel.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	stage.add_child(panel)
	var p := Painter.new()
	p.mode = "circle"
	p.n = 600
	panel.add_child(p)
	# something drawing in the stage itself too, as the vehicle does
	var q := Painter.new()
	q.mode = "circle"
	q.n = 600
	stage.add_child(q)

	var cases := [
		{"name": "stage ALWAYS, panel ALWAYS (governor level 0)",
			"stage": SubViewport.UPDATE_ALWAYS, "proc": Node.PROCESS_MODE_INHERIT},
		{"name": "stage DISABLED + process off, panel ALWAYS (a skipped frame)",
			"stage": SubViewport.UPDATE_DISABLED, "proc": Node.PROCESS_MODE_DISABLED},
	]
	for c in cases:
		stage.render_target_update_mode = c["stage"]
		stage.process_mode = c["proc"]
		RenderingServer.viewport_set_measure_render_time(stage.get_viewport_rid(), true)
		RenderingServer.viewport_set_measure_render_time(panel.get_viewport_rid(), true)
		var drew_panel := 0
		var drew_stage := 0
		var calls := 0.0
		var cpu := 0.0
		for _r in REPS:
			p.last_us = 0
			q.last_us = 0
			p.queue_redraw()
			q.queue_redraw()
			await get_tree().process_frame
			if p.last_us > 0:
				drew_panel += 1
			if q.last_us > 0:
				drew_stage += 1
			calls += Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME)
			cpu += RenderingServer.viewport_get_measured_render_time_cpu(stage.get_viewport_rid())
			cpu += RenderingServer.viewport_get_measured_render_time_cpu(panel.get_viewport_rid())
		print("    %-58s panel _draw ran %d/%d frames, stage _draw ran %d/%d, draw calls %5.0f, render cpu %.2f ms" % [
			c["name"], drew_panel, REPS, drew_stage, REPS, calls / REPS, cpu / REPS])
	stage.queue_free()
	await get_tree().process_frame
