extends Node

## Phase 0 for the comic vehicle (next/vehicles.md): the two engine assumptions the
## whole design rests on, measured rather than assumed.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/vehicle_probe.gd 90
##
## GHOST_PROBE_GPU is not optional - this reads pixels back, and --headless is the
## dummy driver whose readback returns nothing (see run_quiet.sh's header).
##
## ASSUMPTION 1 - A FROZEN PANEL KEEPS ITS PICTURE.
## The comic holds every filled panel by stopping its SubViewport and freeing the
## scene inside it. That is only free if the render target survives BOTH: the update
## mode going to DISABLED, and its last child being freed. If it does not, the whole
## design falls back to a synchronous get_image() per cut - the exact readback stall
## that cost Masking its frame rate - and would have to be rethought.
##
## ASSUMPTION 2 - A VIEWPORT TEXTURE DRAWS THROUGH A PROJECTED GRID.
## A panel is a subdivided grid of textured triangles submitted with the panel
## viewport's texture RID, its corners projected through a Lens3D. Two things are
## checked: that the texture arrives at all (a wrong RID draws flat white, which is
## easy to mistake for "working" at a glance), and that subdivision actually buys
## perspective correctness - the 1x1 case is affine and WRONG on a tilted quad, and
## the measurement here is how much the seam between its two triangles moves.

const PANEL := Vector2i(256, 256)
const OUT := Vector2i(512, 512)


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var fails := 0
	fails += 0 if await _check_freeze() else 1
	fails += 0 if await _check_textured_grid() else 1
	if fails == 0:
		print("vehicle_probe: both assumptions hold.")
	else:
		print("vehicle_probe: %d assumption(s) FAILED - see above." % fails)
	for _i in 4:
		await get_tree().process_frame
	get_tree().quit(fails)


# --- assumption 1 ------------------------------------------------------------

## Paint a solid colour into a SubViewport, freeze it, free the painter, and read the
## target back several frames later. The colour must still be there.
func _check_freeze() -> bool:
	var vp := SubViewport.new()
	vp.size = PANEL
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var painter := _Painter.new()
	painter.col = Color(0.15, 0.72, 0.35)
	vp.add_child(painter)
	# two frames: one to build the canvas item, one to be certain it rendered
	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw
	var before := _sample(vp)
	var want: Color = painter.col     # read BEFORE the free; the node is gone below

	# THE FREEZE, in the order the vehicle will do it: stop the viewport FIRST, then
	# let the scene go. Reversed, the target repaints itself empty on the frame
	# between the free and the stop.
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	vp.process_mode = Node.PROCESS_MODE_DISABLED
	painter.queue_free()
	for _i in 8:
		await RenderingServer.frame_post_draw
	var after := _sample(vp)

	var drift := _dist(before, after)
	var ok := drift < 0.02 and _dist(before, want) < 0.02
	print("vehicle_probe: freeze   painted=%s held=%s drift=%.4f -> %s" % [
		_fmt(before), _fmt(after), drift, "OK" if ok else "FAILED"])
	if not ok:
		print("  a held panel does NOT keep its picture; the comic cannot freeze for free.")
	vp.queue_free()
	return ok


# --- assumption 2 ------------------------------------------------------------

## Render a two-tone source into a panel viewport, then draw that viewport onto a
## quad tilted in 3D through a Lens3D, at 1x1 and at 8x8 subdivision. Measures
## (a) that the texture arrives, and (b) how far the affine seam is from the
## perspective-correct answer.
func _check_textured_grid() -> bool:
	var src := SubViewport.new()
	src.size = PANEL
	src.transparent_bg = false
	src.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(src)
	# left half red, right half blue: a texture that arrived is obvious, and the
	# BOUNDARY between them is where perspective correctness is measurable.
	var painter := _Painter.new()
	painter.split = true
	src.add_child(painter)
	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw

	var lens := Lens3D.new()
	lens.fov = 60.0
	# hard yaw: an affine map is only wrong when the quad is foreshortened, so a
	# gentle tilt would let the broken case pass.
	lens.eye = Vector3(2.6, 0.55, 2.6)
	lens.look = Vector3.ZERO
	lens.prepare()

	var results := {}
	for div in [1, 8]:
		var vp := SubViewport.new()
		vp.size = OUT
		vp.transparent_bg = false
		vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		add_child(vp)
		var q := _Quad.new()
		q.lens = lens
		q.tex_rid = src.get_texture().get_rid()
		q.div = div
		q.out = OUT
		vp.add_child(q)
		await RenderingServer.frame_post_draw
		await RenderingServer.frame_post_draw
		results[div] = _seam_x(vp)
		vp.queue_free()

	var affine: float = results[1]
	var correct: float = results[8]
	# the texture arrived at all: a missing/invalid RID draws the polygon in its
	# modulate colour, so NEITHER half is found and _seam_x returns -1.
	var arrived := affine >= 0.0 and correct >= 0.0
	# and subdivision moved the seam: if it did not, the projection is not actually
	# foreshortening and this probe is measuring nothing.
	var shift := absf(affine - correct) if arrived else 0.0
	var ok := arrived and shift > 2.0
	print("vehicle_probe: texture  seam_1x1=%.1f px  seam_8x8=%.1f px  shift=%.1f px -> %s" % [
		affine, correct, shift, "OK" if ok else "FAILED"])
	if not arrived:
		print("  the viewport texture never reached the triangle array (no red/blue found).")
	elif not ok:
		print("  subdivision changed nothing - the quad is not foreshortened, so this")
		print("  probe proved nothing about perspective correctness. Tilt it harder.")
	src.queue_free()
	return ok


# --- helpers -----------------------------------------------------------------

func _sample(vp: SubViewport) -> Color:
	var img := vp.get_texture().get_image()
	if img == null:
		return Color(-1, -1, -1)
	return img.get_pixel(int(vp.size.x * 0.5), int(vp.size.y * 0.5))


## The x of the red/blue boundary along the quad's centre scanline, in pixels, or -1
## if neither colour is on that line at all.
func _seam_x(vp: SubViewport) -> float:
	var img := vp.get_texture().get_image()
	if img == null:
		return -1.0
	var y := int(vp.size.y * 0.5)
	var last_red := -1
	var first_blue := -1
	for x in vp.size.x:
		var c := img.get_pixel(x, y)
		if c.r > 0.45 and c.b < 0.25:
			last_red = x
		elif c.b > 0.45 and c.r < 0.25 and first_blue < 0:
			first_blue = x
	if last_red < 0 or first_blue < 0:
		return -1.0
	return float(last_red + first_blue) * 0.5


func _dist(a: Color, b: Color) -> float:
	return absf(a.r - b.r) + absf(a.g - b.g) + absf(a.b - b.b)


func _fmt(c: Color) -> String:
	return "(%.2f,%.2f,%.2f)" % [c.r, c.g, c.b]


## Paints a flat colour, or a red/left blue/right split, over its whole viewport.
class _Painter extends Node2D:
	var col := Color.WHITE
	var split := false

	func _draw() -> void:
		var s := get_viewport_rect().size
		if split:
			draw_rect(Rect2(Vector2.ZERO, Vector2(s.x * 0.5, s.y)), Color(0.9, 0.05, 0.05))
			draw_rect(Rect2(Vector2(s.x * 0.5, 0), Vector2(s.x * 0.5, s.y)), Color(0.05, 0.05, 0.9))
		else:
			draw_rect(Rect2(Vector2.ZERO, s), col)


## Draws a unit quad in the z=0 plane, projected through the lens, as a div x div
## grid of textured triangles - the exact construction a comic panel will use.
class _Quad extends Node2D:
	var lens: Lens3D
	var tex_rid := RID()
	var div := 1
	var out := Vector2i(512, 512)

	func _draw() -> void:
		var u := float(mini(out.x, out.y))
		var origin := Vector2(out) * 0.5
		var pts := PackedVector2Array()
		var uvs := PackedVector2Array()
		var cols := PackedColorArray()
		var idx := PackedInt32Array()
		for j in div + 1:
			for i in div + 1:
				var fu := float(i) / float(div)
				var fv := float(j) / float(div)
				# the quad in world space: 2 units wide, 1.5 tall, in the z=0 plane
				var p := Vector3(lerpf(-1.0, 1.0, fu), lerpf(0.75, -0.75, fv), 0.0)
				var pr := lens.project(p)
				pts.append(Vector2(pr.x, pr.y) * u + origin)
				uvs.append(Vector2(fu, fv))
				cols.append(Color.WHITE)
		for j in div:
			for i in div:
				var a := j * (div + 1) + i
				var b := a + 1
				var c := a + div + 1
				var d := c + 1
				idx.append_array([a, b, d, a, d, c])
		RenderingServer.canvas_item_add_triangle_array(
			get_canvas_item(), idx, pts, cols, uvs,
			PackedInt32Array(), PackedFloat32Array(), tex_rid)
