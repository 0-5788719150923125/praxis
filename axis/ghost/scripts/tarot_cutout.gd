extends RefCounted
class_name TarotCutout

## TarotCutout - one painted object, cut out of its background so it can stand on the table.
##
## The painter is asked for a transparent background and, failing that, a flat white one; either
## way the table needs the object alone. A picture that already has transparency round its edge
## keeps it. Otherwise the BACKGROUND IS WHAT TOUCHES THE BORDER: its color is read off the
## border itself (so a painter that went for cream instead of white still works), and the cut is a
## flood fill from the border through everything near that color - so a white cup inside a white
## background survives wherever it does not touch the edge, which a plain color key would erase.
## The edge is softened by how near each edge pixel is to the background, and the background's
## color is taken back out of it (no white fringe on a dark cloth). Specks are dropped, and the
## picture is cropped to the object, so its bottom row is where it stands.

## The long edge an object is kept at: it stands a few centimeters tall in the shot.
const EDGE := 640
## How far from the background's color a pixel may be and still be background (RGB, 0-1 each).
const TOL := 0.11
## ...and how far before an edge pixel is fully the object's.
const SOFT := 0.24
## A piece of the object smaller than this share of the largest is a speck, and goes.
const SPECK := 0.02


## Cut [param src] out of its background into [param dst] (a PNG with alpha, cropped). "" when
## done, else why not.
static func cut(src: String, dst: String) -> String:
	var img := Image.load_from_file(src) if FileAccess.file_exists(src) else null
	if img == null or img.is_empty():
		return "the object's picture could not be read"
	if img.is_compressed():
		img.decompress()
	img.convert(Image.FORMAT_RGBA8)
	var long_edge := maxi(img.get_width(), img.get_height())
	if long_edge > EDGE:
		var k := float(EDGE) / float(long_edge)
		img.resize(maxi(1, roundi(img.get_width() * k)), maxi(1, roundi(img.get_height() * k)), Image.INTERPOLATE_LANCZOS)
	var out := img if _has_alpha_edge(img) else _key(img)
	var used := out.get_used_rect()
	if used.size.x < 8 or used.size.y < 8:
		return "nothing was left of the object once its background was taken out"
	var pad := 4
	var crop := Rect2i(used.position - Vector2i(pad, pad), used.size + Vector2i(pad * 2, pad * 2)).intersection(
		Rect2i(Vector2i.ZERO, out.get_size()))
	out = out.get_region(crop)
	var tmp := dst + ".part.png"
	if out.save_png(tmp) != OK or DirAccess.rename_absolute(tmp, dst) != OK:
		return "could not write " + dst
	return ""


## Whether the picture came with its own transparency: most of its border is see-through.
static func _has_alpha_edge(img: Image) -> bool:
	var w := img.get_width()
	var h := img.get_height()
	var clear := 0
	var total := 0
	for x in range(0, w, 4):
		for y in [0, h - 1]:
			total += 1
			clear += 1 if img.get_pixel(x, y).a < 0.1 else 0
	for y in range(0, h, 4):
		for x in [0, w - 1]:
			total += 1
			clear += 1 if img.get_pixel(x, y).a < 0.1 else 0
	return clear > total / 2


## THE CUT: the border's color, flood-filled from the border; soft edges; specks dropped.
static func _key(img: Image) -> Image:
	var w := img.get_width()
	var h := img.get_height()
	var data := img.get_data()
	# the background's color: the median of the border, channel by channel
	var rs := PackedFloat32Array()
	var gs := PackedFloat32Array()
	var bs := PackedFloat32Array()
	for i in w * h:
		var x := i % w
		var y := i / w
		if x != 0 and y != 0 and x != w - 1 and y != h - 1:
			continue
		rs.append(data[i * 4] / 255.0)
		gs.append(data[i * 4 + 1] / 255.0)
		bs.append(data[i * 4 + 2] / 255.0)
	rs.sort()
	gs.sort()
	bs.sort()
	var bg := Color(rs[rs.size() / 2], gs[gs.size() / 2], bs[bs.size() / 2])
	var dist := PackedFloat32Array()
	dist.resize(w * h)
	for i in w * h:
		var dr := data[i * 4] / 255.0 - bg.r
		var dg := data[i * 4 + 1] / 255.0 - bg.g
		var db := data[i * 4 + 2] / 255.0 - bg.b
		dist[i] = sqrt(dr * dr + dg * dg + db * db)
	# flood from the border through background-colored pixels
	var back := PackedByteArray()
	back.resize(w * h)
	var queue := PackedInt32Array()
	for x in w:
		for y in [0, h - 1]:
			queue.append(y * w + x)
	for y in h:
		for x in [0, w - 1]:
			queue.append(y * w + x)
	var head := 0
	while head < queue.size():
		var i := queue[head]
		head += 1
		if back[i] == 1 or dist[i] > TOL:
			continue
		back[i] = 1
		var x := i % w
		var y := i / w
		if x > 0:
			queue.append(i - 1)
		if x < w - 1:
			queue.append(i + 1)
		if y > 0:
			queue.append(i - w)
		if y < h - 1:
			queue.append(i + w)
	_drop_specks(back, w, h)
	# the alpha: background clear; an object pixel touching background softened by how near it is
	# to the background's color, and that color taken back out of it
	var bgc := PackedFloat32Array([bg.r, bg.g, bg.b])
	for i in w * h:
		if back[i] == 1:
			data[i * 4 + 3] = 0
			continue
		var x := i % w
		var y := i / w
		var edge := (x > 0 and back[i - 1] == 1) or (x < w - 1 and back[i + 1] == 1) \
			or (y > 0 and back[i - w] == 1) or (y < h - 1 and back[i + w] == 1)
		if not edge:
			continue
		var a := clampf((dist[i] - TOL) / (SOFT - TOL), 0.25, 1.0)
		for c in 3:
			var v := data[i * 4 + c] / 255.0
			data[i * 4 + c] = clampi(roundi((v - bgc[c] * (1.0 - a)) / a * 255.0), 0, 255)
		data[i * 4 + 3] = roundi(a * 255.0)
	return Image.create_from_data(w, h, false, Image.FORMAT_RGBA8, data)


## Pieces of the object much smaller than its largest become background: a speck the flood
## could not reach is noise, not part of the thing.
static func _drop_specks(back: PackedByteArray, w: int, h: int) -> void:
	var label := PackedInt32Array()
	label.resize(w * h)
	label.fill(-1)
	var sizes: Array = []
	var queue := PackedInt32Array()
	for start in w * h:
		if back[start] == 1 or label[start] >= 0:
			continue
		var id := sizes.size()
		var n := 0
		queue.clear()
		queue.append(start)
		label[start] = id
		var head := 0
		while head < queue.size():
			var i := queue[head]
			head += 1
			n += 1
			var x := i % w
			if x > 0 and back[i - 1] == 0 and label[i - 1] < 0:
				label[i - 1] = id
				queue.append(i - 1)
			if x < w - 1 and back[i + 1] == 0 and label[i + 1] < 0:
				label[i + 1] = id
				queue.append(i + 1)
			if i >= w and back[i - w] == 0 and label[i - w] < 0:
				label[i - w] = id
				queue.append(i - w)
			if i + w < w * h and back[i + w] == 0 and label[i + w] < 0:
				label[i + w] = id
				queue.append(i + w)
		sizes.append(n)
	var biggest := 0
	for s in sizes:
		biggest = maxi(biggest, int(s))
	for i in w * h:
		if label[i] >= 0 and float(sizes[label[i]]) < float(biggest) * SPECK:
			back[i] = 1
