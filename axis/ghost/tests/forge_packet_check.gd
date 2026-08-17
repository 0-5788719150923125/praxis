extends SceneTree

## forge_packet_check - that [method FrameForge.packet_source] names the job whose geometry is
## ACTUALLY on screen, rather than the newest build to have finished.
##   godot --headless --path . --script res://tests/forge_packet_check.gd
##   godot --headless --path . --script res://tests/forge_packet_check.gd -- --race
##
## WHY THIS IS A GATE. A scene that draws anything alongside its packet has to place it
## against the inputs the packet was built from. contour_map does: the land is a sampling
## window, so a sheet built at one warp offset can be shown at another by translating it, and
## that translation is what turns a re-extraction from a jump into a refresh. It is exact -
## but only if the offset it subtracts is the offset the packet on screen really carries.
##
## THE BUG IT ENDS. That offset used to be published by the job itself, at the end of run() -
## which happens ON THE WORKER THREAD, while the packet it belongs to only reaches the main
## thread later, in a deferred call. Every frame drawn in that window compensated the OLD
## sheet by the NEW build's offset. The error is a whole extraction cadence of warp drift,
## about 1.5% of the sheet's width, and it lasted exactly one frame before correcting itself:
## reported as "the whole scene jumps, then shifts back to where it should be, like a
## correction", on most re-prints. One frame out of place is not a subtle artifact.
##
## WHAT IS ASSERTED HERE is the SYNCHRONOUS path - the export's - because it is deterministic
## and because it carries the same defect in a form that never needs luck: the build happens
## inside the draw, so a scene that reads the source before flushing gets the PREVIOUS sheet's
## offset against this sheet's lines, every single time.
##
## `-- --race` runs the threaded path instead, kicking real worker builds and comparing what
## the packet says against both schemes each frame. It is a probe rather than an assertion:
## the disagreement is a race, so a given run may not catch it. Measured here, three runs:
##
##   packet_source() disagreed on 0 frames of ~237, every run
##   the old shared-slot mimic on 19-23 - about one frame per build
##
## That is the two-sided evidence, and it is why this mode is worth keeping even though it
## cannot be an assertion. Note it exits dirty: any [WorkerThreadPool] task in a `--script`
## process faults during Godot's own teardown (reproduced with a five-line probe that does
## nothing but kick one build), so the verdict is printed BEFORE quitting and the exit code
## from that mode means nothing.

## Frames to run in --race mode, and how often a fresh job is kicked. The cadence is several
## frames because that is the regime the scenes run in - most builds finish between kicks.
const FRAMES := 240
const KICK_EVERY := 6
## Roughly how long a build should take. Long enough to straddle a frame: a build that returns
## instantly never opens the window this is about.
const WORK := 60000

var _node: Node2D
var _forge: FrameForge
var _slot := OldSlot.new()
var _serial := 0
var _frame := 0
var _checked := 0
var _src_bad := 0
var _slot_bad := 0
var _race := false
var _fails: Array = []


func _initialize() -> void:
	_node = Node2D.new()
	root.add_child(_node)
	_race = OS.get_cmdline_user_args().has("--race")
	if _race:
		_forge = FrameForge.new()
		return
	_check_sync()
	_finish()


func _process(_dt: float) -> bool:
	if not _race:
		return false
	_frame += 1
	# Stop kicking before the end so the last build lands and is observed.
	if _frame % KICK_EVERY == 1 and _frame < FRAMES - KICK_EVERY * 3:
		_serial += 1
		var job := StampJob.new()
		job.serial = _serial
		job.slot = _slot
		_forge.kick(job.run, {}, _node, job)
	_observe()
	if _frame < FRAMES:
		return false
	_report_race()
	_finish()
	return false


## The export path, and the deterministic half of the defect. The build happens INSIDE the
## draw, so the source a scene reads before drawing is the previous packet's unless flush()
## has run first - which is exactly the ordering contour_map now depends on.
func _check_sync() -> void:
	FrameForge._sync_known = true
	FrameForge._sync = true
	var forge := FrameForge.new()
	for k in 3:
		var job := StampJob.new()
		job.serial = 100 + k
		forge.kick(job.run, {}, _node, job)
		forge.flush()
		var shown := _serial_of(forge)
		var src := forge.packet_source() as StampJob
		var got := -1 if src == null else src.serial
		print("forge_packet_check: sync build %d -> packet says %d, source says %d"
			% [job.serial, shown, got])
		if shown != job.serial:
			_fails.append("flush() left the packet at %d for a build of %d"
				% [shown, job.serial])
		if got != job.serial:
			_fails.append("flush() left the source at %d while the packet on screen is %d - "
				% [got, shown] + "a scene reading it before drawing places itself against the "
				+ "sheet BEFORE the one it is about to draw")
	FrameForge._sync = false


## Compare what the packet SAYS against what each bookkeeping scheme CLAIMS.
func _observe() -> void:
	var shown := _serial_of(_forge)
	if shown <= 0:
		return
	_checked += 1
	var src := _forge.packet_source() as StampJob
	if src == null or src.serial != shown:
		_src_bad += 1
	if _slot.serial != shown:
		_slot_bad += 1


## The serial the geometry carries. The job encodes it in its VERTEX COUNT, so this reads the
## packet itself rather than anything that travelled beside it.
func _serial_of(forge: FrameForge) -> int:
	var pk: Array = forge._packet
	if pk.is_empty():
		return 0
	var pts: PackedVector2Array = (pk[0] as Dictionary)["pts"]
	return pts.size() / 3 - 1


func _report_race() -> void:
	print("forge_packet_check: %d frames, %d kicks, %d frames with a packet"
		% [FRAMES, _serial, _checked])
	print("forge_packet_check: packet_source disagreed on %d, the old shared-slot mimic on %d"
		% [_src_bad, _slot_bad])
	if _checked < 20:
		_fails.append("only %d frames ever carried a packet - the probe measured nothing"
			% _checked)
	if _src_bad > 0:
		_fails.append("packet_source() named a job other than the one whose geometry is in "
			+ "the packet, on %d of %d frames" % [_src_bad, _checked])
	if _slot_bad == 0:
		print("forge_packet_check: (the old pattern did not lose the race this run - it is a "
			+ "race, not a certainty; the point is that the new one cannot lose it)")


func _finish() -> void:
	_node.free()
	if _fails.is_empty():
		print("forge_packet_check: ALL OK")
		quit()
		return
	print("forge_packet_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	quit(1)


## The old shape, kept alive purely as the thing being measured against.
class OldSlot:
	extends RefCounted
	var serial := 0


## A build that says who it is. The serial goes into the VERTEX COUNT, so it travels inside
## the packet and cannot be read from anywhere else by accident.
class StampJob:
	extends RefCounted

	var serial := 0
	var slot: OldSlot

	func run(_s: Dictionary) -> Array:
		# Real work, so the build straddles a frame the way a scene's does.
		var acc := 0.0
		for i in WORK:
			acc += sqrt(float(i))
		var pts := PackedVector2Array()
		var cols := PackedColorArray()
		var idx := PackedInt32Array()
		for k in serial + 1:
			var b := k * 3
			pts.append(Vector2(acc * 0.0, 0.0))
			pts.append(Vector2(1.0, 0.0))
			pts.append(Vector2(0.0, 1.0))
			cols.append(Color.WHITE)
			cols.append(Color.WHITE)
			cols.append(Color.WHITE)
			idx.append(b)
			idx.append(b + 1)
			idx.append(b + 2)
		# ...and the old pattern, from the worker, one instruction before the packet starts its
		# journey to the main thread. This is the line the jump came out of.
		if slot != null:
			slot.serial = serial
		return [{"pts": pts, "cols": cols, "idx": idx}]
