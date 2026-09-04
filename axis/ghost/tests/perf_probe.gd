extends Node

## perf_probe - WHERE A LIVE FRAME'S TIME GOES, measured, per scene and per vehicle.
##
## NOT a gate. It drives a real session the way the Director does (one update + one
## view.commit per frame on the focal scene, vehicle.advance after it) and splits each
## frame's wall time into the part the probe can name - the SIM (update/commit/advance,
## all GDScript on the main thread) - and the REST (every _draw callback, the
## RenderingServer submit, the present, and every other node's _process). Alongside it
## reads the engine's own counters: draw calls, primitives and objects in the frame, and
## the object count, so a scene that allocates per frame shows up as a rising line.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/perf_probe.gd 600 \
##       -- --vehicle comic --cuts 8 --frames 90 --seed 404
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/perf_probe.gd 900 \
##       -- --catalogue --frames 60
##
## GHOST_PROBE_GPU=1 matters: under --headless the dummy renderer records the canvas
## commands but rasterizes nothing, so the REST column loses the GPU submit and the
## draw-call counters read zero. The probe still runs there, for the SIM column alone.
##
## The stage is 1920x1080 because that is what the live window is on this machine; a
## panel's render target is sized off the stage (ComicVehicle._size_targets), so a
## smaller stage would under-report the comic.

const DT := 1.0 / 60.0
const WARM := 12                  # the Director's own pre-warm count

var _w := 1920
var _h := 1080
var _vehicle_key := "full"
var _cuts := 8
var _frames := 90
var _catalogue := false
var _only := ""                   # --scene NAME: catalogue only this script
## --governed: measure with the stage in the state main.gd's governor puts it in on a
## SKIPPED frame (process_mode DISABLED, render target UPDATE_DISABLED). The Director is an
## autoload and keeps calling update()/queue_redraw() on the scene regardless, so this
## answers whether a skipped frame is actually cheaper on the CPU or only on the GPU.
var _governed := false
var _vehicle: Vehicle = null
var _live: GhostScene = null
var _rows: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_parse_args()
	# VSYNC OFF for the measurement: a frame that waits for a refresh reports the refresh,
	# not the work. The live app keeps the project default; this is the probe's choice only.
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	print("perf_probe: renderer %s, display %s, vsync %d, stage %dx%d, %d frames per sample, dt %.4f" % [
		RenderingServer.get_current_rendering_method(), DisplayServer.get_name(),
		DisplayServer.window_get_vsync_mode(), _w, _h, _frames, DT])
	print("perf_probe: cpu %s x%d" % [OS.get_processor_name(), OS.get_processor_count()])
	if _catalogue:
		await _catalogue_run()
	else:
		await _session_run()
	_summary()
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(0)


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		match args[i]:
			"--catalogue": _catalogue = true
			"--governed": _governed = true
		if i + 1 >= args.size():
			break
		match args[i]:
			"--vehicle": _vehicle_key = args[i + 1]
			"--cuts": _cuts = int(args[i + 1])
			"--frames": _frames = int(args[i + 1])
			"--w": _w = int(args[i + 1])
			"--h": _h = int(args[i + 1])
			"--scene": _only = args[i + 1]


func _stage() -> SubViewport:
	var stage := SubViewport.new()
	stage.size = Vector2i(_w, _h)
	stage.transparent_bg = false
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)
	return stage


# --- a real session: the Director attached, cuts driven as the Director drives them ----

func _session_run() -> void:
	var stage := _stage()
	Director.detach()
	_vehicle = Vehicle.make(_vehicle_key)
	_vehicle.mount(stage)
	Director.attach(stage, _vehicle)
	Director.hold(true)
	_live = Director._current
	if _governed:
		# exactly what main._process does to the stage on a frame the governor skips
		stage.process_mode = Node.PROCESS_MODE_DISABLED
		stage.render_target_update_mode = SubViewport.UPDATE_DISABLED
	print("--- %s seed %d%s ---" % [_vehicle_key, Director.session_seed(),
		" GOVERNED (stage process off, render target disabled)" if _governed else ""])
	for cut in _cuts:
		_cut()
		var name := _live.scene_name if _live != null else "?"
		var live_n := 1
		if _vehicle is ComicVehicle:
			live_n = (_vehicle as ComicVehicle)._live.size()
		var row := await _measure(name, live_n)
		row["cut"] = cut
		_rows.append(row)
		_print_row(row)
	_live = null
	Director.hold(false)
	Director.detach()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame


## One cut, exactly as comic_look_probe performs one (which is exactly as the Director does).
func _cut() -> void:
	if _vehicle.owns_cast():
		var handed := _vehicle.take_over(_live)
		if handed != null:
			_live = handed
		return
	var entry: Dictionary = Director.SCENES[randi() % Director.SCENES.size()]
	var sc: GhostScene = (entry["script"] as Resource).new()
	sc.init_with_seed(randi(), String(entry["behavior"]))
	sc.scene_name = String((entry["script"] as Resource).resource_path).get_file().get_basename()
	var prev: GhostScene = _live
	_vehicle.host_for(sc).add_child(sc)
	_live = sc
	if prev != null and is_instance_valid(prev):
		prev.queue_free()


# --- the catalogue: every registered {scene, behavior}, alone on a full frame -------------

func _catalogue_run() -> void:
	var stage := _stage()
	_vehicle = Vehicle.make("full")
	_vehicle.mount(stage)
	for i in Director.SCENES.size():
		var entry: Dictionary = Director.SCENES[i]
		var script: Resource = entry["script"]
		var name := String(script.resource_path).get_file().get_basename()
		if not _only.is_empty() and not name.contains(_only):
			continue
		var sc: GhostScene = script.new()
		sc.init_with_seed(1000 + i, String(entry["behavior"]))
		sc.scene_name = name
		_vehicle.host_for(sc).add_child(sc)
		_live = sc
		# the Director's pre-warm, so the first measured frame is a settled one
		for _wi in WARM:
			sc.update(Spectrum.current, 0.05)
			sc.view.commit(0.05)
		sc.view.snap()
		await get_tree().process_frame
		var row := await _measure("%s/%s" % [name, String(entry["behavior"])], 1)
		row["kind"] = sc.render_kind
		row["cut"] = i
		_rows.append(row)
		_print_row(row)
		sc.queue_free()
		_live = null
		for _i in 2:
			await get_tree().process_frame
	stage.queue_free()


# --- the measurement ---------------------------------------------------------------------

## THE DRAW BRACKET. CanvasItem.queue_redraw() pushes the item's redraw callback onto the
## engine's MessageQueue, and call_deferred pushes onto the SAME FIFO queue - so a marker
## deferred just BEFORE the sim and another just AFTER it are flushed either side of every
## _draw the sim queued (the focal scene's, the vehicle's, and the off-focal panels ticked
## inside vehicle.advance). The time between the two markers is the GDScript _draw cost plus
## the canvas command recording, and nothing else.
var _draw_t0 := 0
var _draw_ms := 0.0

func _mark_draw_start() -> void:
	_draw_t0 = Time.get_ticks_usec()

func _mark_draw_end() -> void:
	_draw_ms = float(Time.get_ticks_usec() - _draw_t0) / 1000.0


## Every viewport the engine will render this frame: the stage, the root window, and (comic)
## the panel slots. The engine measures its own CPU and GPU render time per viewport once
## asked to; summed, that is the RenderingServer's share of the frame.
func _viewports() -> Array:
	var out: Array = [get_viewport().get_viewport_rid()]
	for vp in _all_subviewports(self):
		out.append(vp.get_viewport_rid())
	return out


func _all_subviewports(n: Node) -> Array:
	var out: Array = []
	for c in n.get_children():
		if c is SubViewport:
			out.append(c)
		out.append_array(_all_subviewports(c))
	return out


func _measure(name: String, live_n: int) -> Dictionary:
	var sim: Array = []
	var frame: Array = []
	var eng: Array = []
	var draw: Array = []
	var rcpu: Array = []
	var rgpu: Array = []
	var calls: Array = []
	var prims: Array = []
	var objs: Array = []
	var vps := _viewports()
	for rid in vps:
		RenderingServer.viewport_set_measure_render_time(rid, true)
	var obj0 := Performance.get_monitor(Performance.OBJECT_COUNT)
	var mem0 := Performance.get_monitor(Performance.MEMORY_STATIC)
	var t_prev := Time.get_ticks_usec()
	for _i in _frames:
		var a := Time.get_ticks_usec()
		_mark_draw_start.call_deferred()
		if _live != null and is_instance_valid(_live):
			_live.update(Spectrum.current, DT)
			_live.view.commit(DT)
		if _vehicle != null and is_instance_valid(_vehicle):
			_vehicle.advance(Spectrum.current, DT, 1.0)
		_mark_draw_end.call_deferred()
		var b := Time.get_ticks_usec()
		await get_tree().process_frame
		var c := Time.get_ticks_usec()
		sim.append(float(b - a) / 1000.0)
		frame.append(float(c - t_prev) / 1000.0)
		draw.append(_draw_ms)
		t_prev = c
		eng.append(Performance.get_monitor(Performance.TIME_PROCESS) * 1000.0)
		var cpu := 0.0
		var gpu := 0.0
		for rid in vps:
			cpu += RenderingServer.viewport_get_measured_render_time_cpu(rid)
			gpu += RenderingServer.viewport_get_measured_render_time_gpu(rid)
		rcpu.append(cpu)
		rgpu.append(gpu)
		calls.append(Performance.get_monitor(Performance.RENDER_TOTAL_DRAW_CALLS_IN_FRAME))
		prims.append(Performance.get_monitor(Performance.RENDER_TOTAL_PRIMITIVES_IN_FRAME))
		objs.append(Performance.get_monitor(Performance.RENDER_TOTAL_OBJECTS_IN_FRAME))
	var kind := "?"
	if _live != null and is_instance_valid(_live):
		kind = _live.render_kind
	return {
		"name": name, "kind": kind, "live": live_n,
		"sim_mean": _mean(sim), "sim_max": _max(sim),
		"draw_mean": _mean(draw), "draw_max": _max(draw),
		"rcpu_mean": _mean(rcpu), "rgpu_mean": _mean(rgpu),
		"frame_mean": _mean(frame), "frame_max": _max(frame),
		"rest_mean": _mean(frame) - _mean(sim) - _mean(draw) - _mean(rcpu),
		"eng_mean": _mean(eng),
		"calls": _mean(calls), "prims": _mean(prims), "objs": _mean(objs),
		"obj_delta": Performance.get_monitor(Performance.OBJECT_COUNT) - obj0,
		"mem_delta_kb": (Performance.get_monitor(Performance.MEMORY_STATIC) - mem0) / 1024.0,
	}


func _print_row(r: Dictionary) -> void:
	print("  %-26s %-9s live %d | frame %6.1f ms (max %6.1f) = sim %5.1f + draw %5.1f (max %5.1f) + render cpu %5.1f + other %5.1f | gpu %5.1f | calls %5.0f prims %7.0f objs %4.0f | +%d obj %+.0f KB" % [
		r.name, r.kind, r.live, r.frame_mean, r.frame_max, r.sim_mean, r.draw_mean, r.draw_max,
		r.rcpu_mean, r.rest_mean, r.rgpu_mean, r.calls, r.prims, r.objs, int(r.obj_delta), r.mem_delta_kb])


func _summary() -> void:
	if _rows.is_empty():
		return
	var by_frame := _rows.duplicate()
	by_frame.sort_custom(func(a, b): return float(a.frame_mean) > float(b.frame_mean))
	print("=== heaviest by mean frame ms (sim = update/commit/advance; draw = _draw callbacks; render cpu = RenderingServer per viewport; other = present + everything else) ===")
	for r in by_frame.slice(0, mini(30, by_frame.size())):
		print("  %-26s %-9s frame %6.1f = sim %5.1f + draw %5.1f + render %5.1f + other %5.1f | gpu %5.1f | calls %5.0f prims %7.0f" % [
			r.name, r.kind, r.frame_mean, r.sim_mean, r.draw_mean, r.rcpu_mean, r.rest_mean,
			r.rgpu_mean, r.calls, r.prims])
	var tot := {"sim": 0.0, "draw": 0.0, "rcpu": 0.0, "rest": 0.0, "frame": 0.0, "gpu": 0.0}
	for r in _rows:
		tot.sim += float(r.sim_mean)
		tot.draw += float(r.draw_mean)
		tot.rcpu += float(r.rcpu_mean)
		tot.rest += float(r.rest_mean)
		tot.frame += float(r.frame_mean)
		tot.gpu += float(r.rgpu_mean)
	var n := float(_rows.size())
	print("=== overall: mean frame %.1f ms (%.1f fps) = sim %.1f (%.0f%%) + draw %.1f (%.0f%%) + render cpu %.1f (%.0f%%) + other %.1f (%.0f%%); gpu %.1f ms; %d samples ===" % [
		tot.frame / n, 1000.0 / maxf(0.001, tot.frame / n),
		tot.sim / n, 100.0 * tot.sim / maxf(0.001, tot.frame),
		tot.draw / n, 100.0 * tot.draw / maxf(0.001, tot.frame),
		tot.rcpu / n, 100.0 * tot.rcpu / maxf(0.001, tot.frame),
		tot.rest / n, 100.0 * tot.rest / maxf(0.001, tot.frame),
		tot.gpu / n, _rows.size()])


func _mean(a: Array) -> float:
	if a.is_empty():
		return 0.0
	var s := 0.0
	for v in a:
		s += float(v)
	return s / float(a.size())


func _max(a: Array) -> float:
	var m := 0.0
	for v in a:
		m = maxf(m, float(v))
	return m
