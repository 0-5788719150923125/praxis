extends RefCounted
class_name TriBatch

## TriBatch - one canvas draw call for thousands of shapes.
##
## The heavy scenes were issuing tens of thousands of `draw_colored_polygon` /
## `draw_line` calls per frame - each a GDScript->server round trip with its
## own array allocation - and THAT, not the geometry count, was the frame
## cost (metropolis: 48x48 tiles x ~4-15 calls each = an unfinishable
## render). The professional shape of the fix is batching: accumulate every
## triangle into one vertex/colour/index buffer and submit it with a single
## [method RenderingServer.canvas_item_add_triangle_array] - the 2D twin of
## a MultiMesh. Triangles render in append order, so the painter's
## back-to-front discipline scenes already follow is preserved exactly.
##
## Usage, inside _draw():
##   _tb.quad(a, b, c, d, col)     # convex quad, in paint order
##   _tb.tri(a, b, c, col)
##   _tb.line(a, b, col, width)    # a thin quad - batches with everything else
##   _tb.poly(points, col)         # convex fan
##   _tb.flush(self)               # ONE draw call; the batch resets
##
## Anything that must draw OVER the batch (fog, glows) is drawn after flush().

var pts := PackedVector2Array()
var cols := PackedColorArray()
var idx := PackedInt32Array()
var uvs := PackedVector2Array()      # non-empty only in a TEXTURED run
var tex_rid := RID()                 # the textured run's texture


## Depth-ordered lists that mix textured and untextured faces batch as RUNS:
## call this before each append with the face's kind - when the kind (or
## texture) changes, the current batch flushes first, so the painter's
## interleave stays exact while same-kind stretches still collapse into
## single draw calls.
func set_run(ci: CanvasItem, textured: bool, rid: RID) -> void:
	var is_textured := tex_rid.is_valid()
	if textured == is_textured and (not textured or rid == tex_rid):
		return
	flush(ci)
	tex_rid = rid if textured else RID()


## A quad with per-vertex colours AND uvs (a textured-run face).
func quad_textured(p: PackedVector2Array, c: PackedColorArray, uv: PackedVector2Array) -> void:
	if p.size() < 4 or c.size() < 4 or uv.size() < 4:
		return
	var base := pts.size()
	for k in 4:
		pts.append(p[k])
		cols.append(c[k])
		uvs.append(uv[k])
	idx.append(base)
	idx.append(base + 1)
	idx.append(base + 2)
	idx.append(base)
	idx.append(base + 2)
	idx.append(base + 3)


func tri(a: Vector2, b: Vector2, c: Vector2, col: Color) -> void:
	var base := pts.size()
	pts.append(a)
	pts.append(b)
	pts.append(c)
	for i in 3:
		cols.append(col)
	idx.append(base)
	idx.append(base + 1)
	idx.append(base + 2)


## A triangle with per-vertex colours (the Gouraud path).
func tri_colored(a: Vector2, b: Vector2, c: Vector2,
		ca: Color, cb: Color, cc: Color) -> void:
	var base := pts.size()
	pts.append(a)
	pts.append(b)
	pts.append(c)
	cols.append(ca)
	cols.append(cb)
	cols.append(cc)
	idx.append(base)
	idx.append(base + 1)
	idx.append(base + 2)


func quad(a: Vector2, b: Vector2, c: Vector2, d: Vector2, col: Color) -> void:
	var base := pts.size()
	pts.append(a)
	pts.append(b)
	pts.append(c)
	pts.append(d)
	for i in 4:
		cols.append(col)
	idx.append(base)
	idx.append(base + 1)
	idx.append(base + 2)
	idx.append(base)
	idx.append(base + 2)
	idx.append(base + 3)


## A quad with per-vertex colours (Gouraud-shaded walls, fogged corners).
func quad_colored(p: PackedVector2Array, c: PackedColorArray) -> void:
	if p.size() < 4 or c.size() < 4:
		return
	var base := pts.size()
	for k in 4:
		pts.append(p[k])
		cols.append(c[k])
	idx.append(base)
	idx.append(base + 1)
	idx.append(base + 2)
	idx.append(base)
	idx.append(base + 2)
	idx.append(base + 3)


## Width of the alpha falloff carried along each side of a line, in pixels.
##
## This is ghost's antialiasing for edges, and it has to be, because nothing else is. Every line
## here is geometry - a thin quad - not a line primitive, so `draw_line`'s own `antialiased` flag
## never applies to it; and Godot's msaa_2d does almost nothing for this drawing path (measured:
## the fraction of edge pixels with no partial coverage moves only 75.0% -> 73.8% going from MSAA
## off to 8x, and in the export's "viewport" stretch mode MSAA is inert entirely - frames come out
## byte-identical at 0, 4x and 8x). A hard-edged 1px quad of bright colour on ghost's near-black
## backgrounds is the worst case there is: it steps, and it crawls as the geometry turns.
##
## 0.8 px rather than a full pixel: wide enough to give the rasterizer a real gradient to
## interpolate, narrow enough that a 1px line still reads as 1px rather than as a soft smear.
const FEATHER := 0.8


## A line as a thin quad, so it batches (and layers!) with the fills around
## it - a separate multiline call would break the painter's order.
##
## Emitted as three quads: a solid core, and a fading skirt down each side (see [constant FEATHER]).
## That is 6 triangles where this used to cost 2, which is the price of the only antialiasing this
## renderer has. Pass `soft = false` for bulk hairline work that does not need it - the floor bands
## behind a city's windows, say - to get the old single-quad form back.
func line(a: Vector2, b: Vector2, col: Color, width := 1.0, soft := true) -> void:
	var d := b - a
	if d.length_squared() < 0.000001:
		return
	var u := Vector2(-d.y, d.x).normalized()
	if not soft:
		var n := u * (width * 0.5)
		quad(a - n, b - n, b + n, a + n, col)
		return
	# Keep a real core even for hairlines, so a thin line stays a line rather than becoming two
	# feathers meeting in the middle (which reads as a dimmer, blurrier line, not a smoother one).
	var hw := maxf(width * 0.5, 0.4)
	var i0 := a - u * hw
	var i1 := b - u * hw
	var i2 := b + u * hw
	var i3 := a + u * hw
	quad(i0, i1, i2, i3, col)
	var fade := Color(col.r, col.g, col.b, 0.0)
	var o := u * (hw + FEATHER)
	quad_colored(PackedVector2Array([a - o, b - o, i1, i0]),
		PackedColorArray([fade, fade, col, col]))
	quad_colored(PackedVector2Array([i2, i3, a + o, b + o]),
		PackedColorArray([col, col, fade, fade]))


## A CONVEX polygon as a fan (matches draw_colored_polygon on convex input).
func poly(p: PackedVector2Array, col: Color) -> void:
	if p.size() < 3:
		return
	var base := pts.size()
	for v in p:
		pts.append(v)
		cols.append(col)
	for i in range(1, p.size() - 1):
		idx.append(base)
		idx.append(base + i)
		idx.append(base + i + 1)


## Native-key painter sort: order `items` (Array of Dictionary) far-to-near
## by the float field `key`, WITHOUT sort_custom - whose per-comparison
## GDScript lambda costs n·log n interpreter calls (a 20k-face list means
## hundreds of thousands of lambda invocations per frame; measured as a
## dominant scene cost). Quantized depth packs into the high bits of an
## int64, the item index into the low 22, and the packed array sorts
## natively. Stable (index tiebreak), ~1/128-unit depth resolution - more
## than painter's needs.
static func painter_sort(items: Array, key: String = "d") -> Array:
	var n := items.size()
	if n < 2:
		return items
	var keys := PackedInt64Array()
	keys.resize(n)
	for i in n:
		var q := int(clampf(float(items[i][key]), -100000.0, 100000.0) * 16384.0)
		keys[i] = (-q << 22) | i        # far (big d) -> most negative -> first
	keys.sort()
	var out := []
	out.resize(n)
	for k in n:
		out[k] = items[keys[k] & 0x3FFFFF]
	return out


var _chunks: Array = []


## WORKER-SIDE batching (the FrameForge contract): a builder thread cannot
## touch a canvas, so instead of flushing, runs are CUT into chunk
## dictionaries and returned as the packet. mark_run() is set_run() for
## chunk mode - a kind (or texture) change cuts the current run.
func mark_run(textured: bool, rid: RID) -> void:
	var is_textured := tex_rid.is_valid()
	if textured == is_textured and (not textured or rid == tex_rid):
		return
	cut()
	tex_rid = rid if textured else RID()


## End the current run into the chunk list (no canvas involved).
func cut() -> void:
	if idx.is_empty():
		return
	var c := {"pts": pts, "cols": cols, "idx": idx}
	if tex_rid.is_valid() and uvs.size() == pts.size():
		c["uvs"] = uvs
		c["tex"] = tex_rid
	_chunks.append(c)
	pts = PackedVector2Array()
	cols = PackedColorArray()
	idx = PackedInt32Array()
	uvs = PackedVector2Array()


## The finished packet: every chunk in append order, batch reset.
func take_chunks() -> Array:
	cut()
	var out := _chunks
	_chunks = []
	return out


## Submit everything accumulated since the last flush as ONE draw call on
## `ci`'s canvas item, then reset. Call at the end of _draw() (or mid-draw
## when something un-batchable must interleave).
func flush(ci: CanvasItem) -> void:
	if idx.is_empty():
		return
	if uvs.size() == pts.size() and tex_rid.is_valid():
		RenderingServer.canvas_item_add_triangle_array(
			ci.get_canvas_item(), idx, pts, cols, uvs,
			PackedInt32Array(), PackedFloat32Array(), tex_rid)
	else:
		RenderingServer.canvas_item_add_triangle_array(
			ci.get_canvas_item(), idx, pts, cols)
	pts.clear()
	cols.clear()
	idx.clear()
	uvs.clear()
