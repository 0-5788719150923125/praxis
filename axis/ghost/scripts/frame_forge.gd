extends RefCounted
class_name FrameForge

## FrameForge - scene geometry built OFF the main thread: the foundational
## fix for UI lag, not another mitigation.
##
## Why this exists: the engine runs ONE main loop - input, then every node's
## _process/_draw, then present. The UI composites in that same loop, so a
## scene that spends 300 ms of GDScript in a frame blocks clicks for 300 ms;
## nothing can preempt a running script. Real engines never have such frames:
## scene work lives on the GPU and worker threads, and the main thread only
## SUBMITS. This is that architecture for ghost's scenes.
##
## The contract a scene opts into:
##   - a STATIC, PURE builder function: takes one snapshot Dictionary of
##     plain data (packed arrays, scalars, a fresh-per-frame AudioFeatures),
##     touches NO nodes and NO drawing API, and returns packet chunks:
##     [{pts, cols, idx[, uvs, tex]}] - typically built with a TriBatch.
##   - update() calls kick(builder, snapshot, self) - if the worker is free,
##     the snapshot ships to a WorkerThreadPool thread; when the packet
##     lands, the scene is queue_redraw()n.
##   - _draw() calls submit(self): a few RenderingServer calls, microseconds.
##
## Properties that follow from the contract:
##   - A scene frame can NEVER block the UI again, however heavy its math
##     gets - if the worker falls behind, the backdrop just updates less
##     often; input never waits. This generalizes to ANY scene that adopts
##     the contract, which is the point.
##   - Snapshots must be safe: duplicate anything main mutates per frame
##     (swarm fields, particle arrays); immutable-after-build arrays may
##     pass by reference; AudioFeatures is fresh per frame upstream.
##   - The builder being static + snapshot-only means a scene freed mid-job
##     (a Director cut) is harmless: the weakref completion just drops.
##   - EXPORTS STAY DETERMINISTIC: in --export mode kick() runs the builder
##     SYNCHRONOUSLY, so a movie renders the same frames every run. Only
##     live sessions pay the (harmless) async frame of latency.

var _packet: Array = []
var _busy := false
var _pending_builder := Callable()
var _pending_snapshot := {}
var _has_pending := false            # explicit: a JOB-pattern snapshot is {}
                                     # (state lives on the job object), so
                                     # emptiness must never mean "no work"
var _pending_retain: RefCounted = null   # a Callable holds only an object ID -
                                         # it does NOT keep a RefCounted job
                                         # alive (measured: null::run) - so the
                                         # forge retains the job explicitly
                                         # until its build has run

# resolved lazily on first kick (a static initializer proved unreliable for
# this): exports MUST build synchronously or the movie records stale packets
static var _sync_known := false
static var _sync := false


## Register the newest frame inputs. CHEAP AND IDEMPOTENT by design: the
## Director legitimately calls a scene's update() several times per frame
## (music-clock debt substeps, the 12-step cut pre-warm), and the old
## contract absorbed that because the expensive work lived in the ONCE-
## deduped _draw. kick() therefore only stores the snapshot (newest wins);
## the build itself runs at most once per drawn frame - in submit() when
## synchronous, on the worker (pipelined, always the newest snapshot) live.
func kick(builder: Callable, snapshot: Dictionary, scene: CanvasItem,
		retain: RefCounted = null) -> void:
	if not _sync_known:
		_sync_known = true
		_sync = OS.get_cmdline_user_args().has("--export")
	_pending_builder = builder
	_pending_snapshot = snapshot
	_pending_retain = retain
	_has_pending = true
	if _sync:
		scene.queue_redraw()
		return
	if not _busy:
		_launch(scene)


func _launch(scene: CanvasItem) -> void:
	_busy = true
	var builder := _pending_builder
	var snapshot := _pending_snapshot
	var retain := _pending_retain    # closure capture = a REAL reference: the
	_pending_snapshot = {}           # job lives exactly as long as its build
	_pending_retain = null
	_has_pending = false
	var wr: WeakRef = weakref(scene)
	WorkerThreadPool.add_task(func() -> void:
		var out: Array = builder.call(snapshot)
		if retain != null:
			pass                     # (referenced so the capture is retained)
		var fin := func() -> void:
			_busy = false
			_packet = out
			var sc: Object = wr.get_ref()
			if sc != null:
				(sc as CanvasItem).queue_redraw()
				if _has_pending:
					_launch(sc as CanvasItem)   # pipeline: newest inputs next
		fin.call_deferred())


## Emit the latest packet onto `ci`'s canvas. Call from _draw(). In sync
## (export) mode this is where the once-per-frame build happens; live, a
## scene with no packet yet builds its FIRST one inline so a fresh cut
## never shows an empty backdrop.
func submit(ci: CanvasItem) -> void:
	if _has_pending and (_sync or _packet.is_empty()):
		var t0 := Time.get_ticks_usec()
		_packet = _pending_builder.call(_pending_snapshot)
		_pending_snapshot = {}
		_pending_retain = null
		_has_pending = false
		if not OS.get_environment("GHOST_PROFILE").is_empty():
			print("forge sync build: %d us" % (Time.get_ticks_usec() - t0))
	var item := ci.get_canvas_item()
	for chunk in _packet:
		var c: Dictionary = chunk
		if (c.idx as PackedInt32Array).is_empty():
			continue
		if c.has("tex"):
			RenderingServer.canvas_item_add_triangle_array(
				item, c.idx, c.pts, c.cols, c.uvs,
				PackedInt32Array(), PackedFloat32Array(), c.tex)
		else:
			RenderingServer.canvas_item_add_triangle_array(
				item, c.idx, c.pts, c.cols)
