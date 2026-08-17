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
## The job that built the packet now in `_packet`, swapped WITH it in the same
## assignment - see [method packet_source].
var _packet_src: RefCounted = null
var _fails := 0                          # consecutive builds that returned no packet

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
		# UNTYPED, then checked. A builder that aborts - any runtime error anywhere inside it -
		# returns null, and `var out: Array = <null>` raises a SECOND error that kills this lambda
		# before it can schedule `fin`. `_busy` would then stay true forever, kick() would never
		# relaunch, and the scene would show its last packet for the rest of its life - or, for a
		# scene whose _draw is only `begin_draw(); _forge.submit(self)` with no ground or layers of
		# its own (metropolis, spires, terrain_city), NOTHING AT ALL. Silently, and for the whole
		# hold. A latched failure has to degrade to a stale frame and a warning, never to a
		# permanently black stage.
		var built: Variant = builder.call(snapshot)
		if retain != null:
			pass                     # (referenced so the capture is retained)
		var ok: bool = built is Array
		var out: Array = built if ok else []
		var fin := func() -> void:
			_busy = false
			if ok:
				_packet = out
				_packet_src = retain     # in the SAME assignment as the packet, on
				                         # the MAIN thread - see packet_source()
				_fails = 0
			else:
				_note_failure()      # keep the last good packet rather than blanking the stage
			var sc: Object = wr.get_ref()
			if sc != null:
				(sc as CanvasItem).queue_redraw()
				if _has_pending:
					_launch(sc as CanvasItem)   # pipeline: newest inputs next
		fin.call_deferred())


## A build produced no packet - the builder aborted on a runtime error. SAY SO. The failure mode
## this replaces was invisible: a scene that draws only what the forge gives it went black for its
## whole hold with nothing in the log, which is precisely how a 29-second black stretch shipped in
## a finished render and was found by measuring the video afterwards rather than by any gate. Loud
## on the first failure and on every hundredth after, so a persistently broken builder is a line in
## the log and not a mystery.
func _note_failure() -> void:
	_fails += 1
	if _fails == 1 or _fails % 100 == 0:
		push_warning("ghost: FrameForge build returned no packet (%d in a row) - " % _fails
			+ "the scene is showing its last good frame; look above for the builder's own error")


## The job object whose build produced the packet that is on screen RIGHT NOW.
##
## This exists because a scene that draws anything ALONGSIDE the packet has to place it
## against the inputs the packet was built from, not against the inputs it has now - and it
## cannot ask the job it last kicked, because most kicked jobs never run (kick() keeps only
## the newest snapshot) and a build that did run lands a frame or more later.
##
## The failure it ends is specific and was visible: contour_map used to have each job write
## its warp offset into a shared slot at the end of run(), which happens ON THE WORKER, while
## the packet itself is swapped in a deferred call on the main thread. Any frame drawn in
## between compensated the OLD sheet by the NEW offset - a whole extraction cadence of warp,
## about 1.5% of the sheet's width - and then corrected itself on the next frame. One frame
## out of place reads as a hard jump, and it happened on most re-prints. Here the source and
## the packet are swapped in one assignment on one thread, so they cannot disagree.
##
## Null until the first packet lands. Cast it to your own job class.
func packet_source() -> RefCounted:
	return _packet_src


## Finish any build that has to happen on the main thread, so the packet AND
## [method packet_source] are final for this frame. [method submit] calls this itself; call
## it directly only when you must READ the source before drawing (contour_map sets its
## transform from it, and in export mode the build happens right here).
func flush() -> void:
	# `not _busy` is load-bearing and was missing. The inline first build exists so a
	# fresh cut never shows an empty backdrop, but without that guard it fires WHILE a
	# worker build of the same job is already in flight - so the builder runs twice at
	# once, on two threads.
	#
	# For a pure builder over a snapshot that is merely wasted work, which is why it went
	# unnoticed. For a JOB OBJECT that carries state it is a data race: murmuration's job
	# advances a shared Boids flock, whose packed arrays are copy-on-write, and two
	# concurrent `step()` calls tore them mid-reallocation - the flock reported more birds
	# than its position arrays held and indexing went out of bounds. It surfaced as three
	# stray "Out of bounds get index" errors across two hundred scene builds, which is
	# exactly the shape of a race and nothing like the shape of an indexing mistake.
	#
	# Skipping the inline build here costs at most one frame of empty backdrop, and only
	# on the frame a worker is already busy producing the real thing.
	if _has_pending and (_sync or (_packet.is_empty() and not _busy)):
		var t0 := Time.get_ticks_usec()
		var built: Variant = _pending_builder.call(_pending_snapshot)
		var src := _pending_retain
		# Cleared BEFORE the check, so a builder that fails every frame does not also re-run every
		# frame off a pending flag that never clears. In sync (export) mode this is the whole
		# render path, and the old `_packet = <null>` assignment aborted submit() itself - past
		# the point of no return, so not one canvas_item call below ever ran.
		_pending_snapshot = {}
		_pending_retain = null
		_has_pending = false
		if built is Array:
			_packet = built
			_packet_src = src
			_fails = 0
		else:
			_note_failure()
		if not OS.get_environment("GHOST_PROFILE").is_empty():
			print("forge sync build: %d us" % (Time.get_ticks_usec() - t0))


## Emit the latest packet onto `ci`'s canvas. Call from _draw(). In sync
## (export) mode this is where the once-per-frame build happens; live, a
## scene with no packet yet builds its FIRST one inline so a fresh cut
## never shows an empty backdrop.
func submit(ci: CanvasItem) -> void:
	flush()
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
