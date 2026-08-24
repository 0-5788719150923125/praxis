extends Node

## Gate for the black wedges in tunnel_run's walls.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/tunnel_face_check.gd 300
##
## GHOST_PROBE_GPU is not optional: the fault is a shape drawn in the wrong place, and the
## only instrument that can see that is the picture.
##
## WHAT WENT WRONG. The wall was emitted one QUAD per (station, side) and back-face culled on
## ONE normal - the first of its two triangles. A tube quad is planar only while the
## cross-section is smooth, and three of the six profiles (ribbed, fluted, star) have a groove
## in them, so the quad spanning that groove is a saddle whose halves face opposite ways. The
## cull applied the first half's verdict to both, so the back-facing half was drawn anyway:
## the unlit outside of the far wall, laid over the near wall. Half a quad is a triangle,
## which is how it was reported - "black, triangular artifacts that often appear along faces
## of the walls". Measured, 473 to 532 quads of the ~1080 in a frame had halves that
## disagreed, and the largest stray wedge covered 33,500 px2.
##
## The fix does not cull the halves apart - that pops, because "hidden" under a painter's sort
## is draw order and not a depth buffer, and a dropped half opens a hole that reappears frame
## to frame (measured: 25 spikes in 89 frames against none for the quad rule). It SORTS them
## apart: both halves are still drawn, each at its own depth, which is what puts the dark far
## half behind the lit near one.
##
## THE CONTROL IS ON THE INSTRUMENT, NOT ON THE RULE, and that is worth being plain about. The
## honest control would be the old rule rendered alongside, but the old rule is sixty lines of
## geometry and a second copy of it in here would rot. What CAN be checked live is that the
## detector still sees a wedge when there is one, so a planted one is put in front of it every
## run. The rule's own before-and-after is recorded rather than re-measured: on this detector
## the three grooved profiles scored 2.599%, 2.194% and 2.190% before the fix and nothing above
## 0.375% after, with the next-worst seed at 0.775% before. To reproduce the failure, put the
## cull back on one normal in `tunnel_run.gd` and watch this go red.

const W := 640
const H := 360
const DT := 1.0 / 30.0
## Fraction of sampled pixels allowed to be a hole - far darker than the wall around them.
## Between the 0.375% the fixed renderer scores and the 2.19% the broken one did.
const MAX_HOLES := 0.010
## Seeds chosen because they roll the grooved profiles, which is where the fault lives. Named
## rather than swept: the point is to keep watching the cases that broke.
const SEEDS := [15, 22, 23, 13]

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


## Fraction of sampled pixels far darker than the wall a short way off in every direction.
## A hole in a lit surface, rather than a shaded part of one - shading has a gradient, and
## the four probes a few pixels out would come back dark with it.
func _holes(img: Image) -> float:
	var r := 7
	var bad := 0
	var tot := 0
	for y in range(r, H - r, 3):
		for x in range(r, W - r, 3):
			var c := img.get_pixel(x, y)
			var v := c.r + c.g + c.b
			var around := 0.0
			for d in [Vector2i(r, 0), Vector2i(-r, 0), Vector2i(0, r), Vector2i(0, -r)]:
				var n := img.get_pixel(x + d.x, y + d.y)
				around += n.r + n.g + n.b
			around *= 0.25
			tot += 1
			if around > 0.25 and v < around * 0.45:
				bad += 1
	return float(bad) / float(maxi(1, tot))


## THE INSTRUMENT'S OWN CONTROL: a lit field with one dark wedge painted into it. The detector
## has to find that and has to leave the clean field alone, or a green run below means only
## that the detector stopped working.
func _check_detector() -> void:
	var clean := Image.create_empty(W, H, false, Image.FORMAT_RGBAF)
	clean.fill(Color(0.55, 0.28, 0.62))
	var planted := clean.duplicate() as Image
	# THREE wedges, sized and scattered like the fault they stand in for - the broken renderer
	# put several in a frame, not one. A single one scores 1.03% on this detector, which is
	# barely over the threshold it is meant to prove the detector can clear: only the RIM of a
	# wedge is flagged, because the middle of a large one has dark neighbours too.
	for tri in [
			[Vector2(120.0, 60.0), Vector2(215.0, 78.0), Vector2(145.0, 165.0)],
			[Vector2(380.0, 40.0), Vector2(470.0, 96.0), Vector2(392.0, 132.0)],
			[Vector2(255.0, 210.0), Vector2(340.0, 245.0), Vector2(268.0, 300.0)]]:
		var a: Vector2 = tri[0]
		var b: Vector2 = tri[1]
		var c: Vector2 = tri[2]
		var lo := Vector2i(int(minf(a.x, minf(b.x, c.x))), int(minf(a.y, minf(b.y, c.y))))
		var hi := Vector2i(int(maxf(a.x, maxf(b.x, c.x))), int(maxf(a.y, maxf(b.y, c.y))))
		for y in range(lo.y, hi.y + 1):
			for x in range(lo.x, hi.x + 1):
				var p := Vector2(x, y)
				var d1 := (b - a).cross(p - a)
				var d2 := (c - b).cross(p - b)
				var d3 := (a - c).cross(p - c)
				if (d1 >= 0.0 and d2 >= 0.0 and d3 >= 0.0) \
						or (d1 <= 0.0 and d2 <= 0.0 and d3 <= 0.0):
					planted.set_pixel(x, y, Color(0.02, 0.01, 0.02))
	var on_clean := _holes(clean)
	var on_planted := _holes(planted)
	print("tunnel_face_check: detector - clean field %.3f%%, three planted wedges %.3f%%"
		% [on_clean * 100.0, on_planted * 100.0])
	# The bar is the gate's own threshold: shown the fault, the instrument has to report it as
	# a FAILURE, not merely notice it.
	_ok(on_planted > MAX_HOLES * 1.5,
		"the detector scored only %.3f%% on three planted wedges (needs over %.3f%%) - it cannot see the fault it is for"
		% [on_planted * 100.0, MAX_HOLES * 150.0])
	_ok(on_clean < MAX_HOLES * 0.1,
		"the detector scored %.3f%% on a flat lit field - it is reporting shading as holes"
		% (on_clean * 100.0))


func _run() -> void:
	_check_detector()
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var f := AudioFeatures.new()
	f.energy = 0.7
	f.beat_period = 0.5
	f.flux = 0.03
	f.bands = PackedFloat32Array()
	for _i in 64:
		f.bands.append(0.5)
	var grooved := 0
	for sv in SEEDS:
		var sc = load("res://scripts/scenes/tunnel_run.gd").new()
		vp.add_child(sc)
		sc.init_with_seed(int(sv), "drift")
		var prof := String(sc.params.get("profile", "?"))
		if prof in ["ribbed", "fluted", "star"]:
			grooved += 1
		var worst := 0.0
		var wf := -1
		for i in 70:
			sc.update(f, DT)
			await get_tree().process_frame
			if i < 12:
				continue
			await RenderingServer.frame_post_draw
			var h := _holes(vp.get_texture().get_image())
			if h > worst:
				worst = h
				wf = i
		print("tunnel_face_check: seed %2d %-7s / %-9s worst frame %2d at %.3f%% holes"
			% [sv, prof, String(sc.params.get("track", "?")), wf, worst * 100.0])
		_ok(worst <= MAX_HOLES,
			"seed %d (%s): %.3f%% of the frame is a hole in a lit wall (want under %.3f%%) - a face is being drawn where it should not be"
			% [sv, prof, worst * 100.0, MAX_HOLES * 100.0])
		vp.remove_child(sc)
		sc.free()
	# ...and the seeds have to still be rolling the profiles the fault lives on.
	_ok(grooved >= 3,
		"only %d of the %d seeds rolled a grooved profile - the roll has moved and these seeds no longer cover the fault"
		% [grooved, SEEDS.size()])
	vp.queue_free()
	await get_tree().process_frame
	print("")
	if _fails.is_empty():
		print("tunnel_face_check: ALL OK - no wedges in the walls.")
		get_tree().quit()
		return
	print("tunnel_face_check: %d FAILURE(S)" % _fails.size())
	for x in _fails:
		print("   ", x)
	get_tree().quit(1)
