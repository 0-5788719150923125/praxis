extends VortexScene

## Cityscape - a skyline of rectangles that grows with the music.
##
## Layered rows of buildings; each building's height tracks a slice of the
## spectrum, so the skyline rises and falls with the track. Windows light up in a
## grid, flickering with the beat. Back layers are dimmer and shorter for depth.
## All axis-aligned rectangles - drawn static and upright, no triangulation.

var _f: AudioFeatures = AudioFeatures.new()
var _layers: Array = []


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var layer_count := rng.randi_range(2, 3)
	for l in layer_count:
		var buildings: Array = []
		var count := rng.randi_range(8, 14)
		for b in count:
			buildings.append({
				"w": rng.randf_range(0.55, 1.35),
				"band": rng.randf(),
				"wcols": rng.randi_range(2, 4),
				"wrows": rng.randi_range(4, 9),
				"phase": rng.randf() * TAU,
			})
		var depth := float(l) / float(maxi(1, layer_count - 1))
		_layers.append({"buildings": buildings, "depth": depth})
	return {
		"hue": rng.randf(),
		"max_h": rng.randf_range(0.45, 0.80),
		"win_hue": rng.randf(),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	queue_redraw()


const OVER := 1.3        # draw wider than the frame so view motion never bares an edge

func _draw() -> void:
	begin_draw()
	var halfw := size.x * 0.5 * OVER
	var halfh := size.y * 0.5
	# The baseline sits BELOW the visible bottom, so the buildings run off the bottom
	# edge - no black bar under the skyline (the old 0.92 left a strip).
	var ground := halfh * 1.18
	var hue: float = params.hue
	var max_h: float = params.max_h
	var win_hue: float = params.win_hue

	for li in _layers.size():
		var layer: Dictionary = _layers[li]
		var depth: float = layer.depth         # 0 back .. 1 front
		var buildings: Array = layer.buildings
		var bn := buildings.size()
		var bw := (halfw * 2.0) / float(bn)
		for bi in bn:
			var bd: Dictionary = buildings[bi]
			var loud: float = _f.sample(float(bd.band))
			var w := bw * 0.8 * float(bd.w)
			var x := -halfw + (bi + 0.5) * bw - w * 0.5
			var h := max_h * size.y * (0.22 + 0.78 * loud) * (0.5 + 0.5 * depth)
			var top := ground - h
			var shade := 0.10 + 0.16 * depth + 0.12 * loud
			var hh := fposmod(hue + 0.05 * depth, 1.0)
			draw_rect(Rect2(x, top, w, ground - top), Color.from_hsv(hh, 0.4, shade))
			_windows(x, top, w, ground, bd, loud, depth, win_hue)


# A grid of lit windows on one building face; which are lit drifts with time and
# flares with loudness/beat.
func _windows(x: float, top: float, w: float, ground: float, bd: Dictionary, loud: float, depth: float, win_hue: float) -> void:
	var cols := int(bd.wcols)
	var rows := int(bd.wrows)
	var pad := w * 0.16
	var cell_w := (w - pad * 2.0) / float(cols)
	var cell_h := minf(cell_w, (ground - top) / float(rows + 1))
	var ww := cell_w * 0.6
	var wh := cell_h * 0.6
	var lit_level := 0.35 + 0.5 * loud + 0.3 * _f.beat
	var col := Color.from_hsv(win_hue, 0.35, 0.7 + 0.3 * depth, 0.85)
	for wy in rows:
		var wy_top := top + pad + wy * cell_h
		if wy_top + wh > ground:
			break
		for wx in cols:
			var k := float(wx * 7 + wy * 13) + float(bd.phase)
			var on := 0.5 + 0.5 * sin(k + _f.time * 1.5)
			if on < lit_level:
				var wx_left := x + pad + wx * cell_w
				draw_rect(Rect2(wx_left, wy_top, ww, wh), col)
