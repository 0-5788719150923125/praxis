extends GhostScene

## Cityscape - a skyline of rectangles that grows with the music.
##
## Layered rows of buildings; each building's height tracks a slice of the
## spectrum, so the skyline rises and falls with the track. Windows light up in a
## grid, flickering with the beat. Back layers are dimmer and shorter for depth.
## All axis-aligned rectangles - drawn static and upright, no triangulation.
##
## Which city it is varies per session: a [Scheme] mood colours the blocks and its accent
## lights the windows (so the lit grid contrasts the mass by construction, the way sodium
## windows sit against a blue-grey tower), and a sampled PROFILE decides the silhouette -
## a crowded low downtown, a handful of wide towers, or a broad flat sprawl.

## Skyline profiles. Building count, width and height are one decision, not three: many
## narrow blocks read as downtown density, few wide ones as a tower cluster, and the window
## grid has to follow the facade it sits on.
const PROFILES := {
	"dense":   {"layers": [3, 4], "count": [16, 26], "w": [0.50, 0.95], "max_h": [0.35, 0.62],
		"wcols": [2, 3], "wrows": [5, 10]},
	"towers":  {"layers": [2, 3], "count": [5, 9],   "w": [0.95, 1.70], "max_h": [0.62, 0.95],
		"wcols": [3, 5], "wrows": [8, 14]},
	"mixed":   {"layers": [2, 3], "count": [8, 14],  "w": [0.55, 1.35], "max_h": [0.45, 0.80],
		"wcols": [2, 4], "wrows": [4, 9]},
	"lowrise": {"layers": [2, 4], "count": [12, 20], "w": [0.70, 1.30], "max_h": [0.22, 0.44],
		"wcols": [2, 4], "wrows": [3, 6]},
}

var _f: AudioFeatures = AudioFeatures.new()
var _layers: Array = []
var _sch: Scheme


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	_sch = Scheme.pick(rng)
	var profile := String(PROFILES.keys()[rng.randi() % PROFILES.size()])
	var pf: Dictionary = PROFILES[profile]
	var layer_count := rng.randi_range(int(pf.layers[0]), int(pf.layers[1]))
	for l in layer_count:
		var buildings: Array = []
		var count := rng.randi_range(int(pf.count[0]), int(pf.count[1]))
		for b in count:
			buildings.append({
				"w": rng.randf_range(float(pf.w[0]), float(pf.w[1])),
				"band": rng.randf(),
				"wcols": rng.randi_range(int(pf.wcols[0]), int(pf.wcols[1])),
				"wrows": rng.randi_range(int(pf.wrows[0]), int(pf.wrows[1])),
				"phase": rng.randf() * TAU,
				# Each block sits a little off the scheme's base hue, so a row of towers is
				# a family of related greys/blues rather than one flat repeated colour.
				"hue": _sch.vary(rng, 0.6),
			})
		var depth := float(l) / float(maxi(1, layer_count - 1))
		_layers.append({"buildings": buildings, "depth": depth})
	# Weather, composed from the shared Layer registry - the same snow/rain that are
	# scenes on their own, falling over the skyline (integration, not bespoke code). It takes
	# the scheme's accent, so the sky is lit by the same city.
	var weather := rng.randf()
	if weather < 0.30:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(80, 140),
			"hue": _sch.accent, "sat": _sch.sat * 0.3})
		add_layer("snow", rng, {"count": rng.randi_range(70, 120),
			"hue": _sch.accent, "sat": _sch.sat * 0.25, "fall": 0.09})
	elif weather < 0.50:
		add_layer("rain", rng, {"count": rng.randi_range(100, 170),
			"hue": _sch.accent, "sat": _sch.sat * 0.4, "slant": 0.22})
	elif weather < 0.65:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(90, 160),
			"hue": _sch.accent, "sat": _sch.sat * 0.3})
	return {
		"mood": _sch.name,
		"profile": profile,
		"layers": layer_count,
		"max_h": rng.randf_range(float(pf.max_h[0]), float(pf.max_h[1])),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	update_layers(f, delta)
	queue_redraw()


const OVER := 1.3        # draw wider than the frame so view motion never bares an edge

func _draw() -> void:
	begin_draw()
	draw_layers("back")          # stars in the sky, behind the skyline
	var halfw := size.x * 0.5 * OVER
	var halfh := size.y * 0.5
	# The baseline sits BELOW the visible bottom, so the buildings run off the bottom
	# edge - no black bar under the skyline (the old 0.92 left a strip).
	var ground := halfh * 1.18
	var max_h: float = params.max_h

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
			var hh := fposmod(float(bd.hue) + 0.05 * depth, 1.0)
			# Saturation/value SCALE the mood, so an ash city stays grey concrete and a
			# violet one stays violet without either restating absolute numbers.
			draw_rect(Rect2(x, top, w, ground - top), _sch.color(hh, 0.6, shade * 1.15))
			_windows(x, top, w, ground, bd, loud, depth)

	draw_layers("front")         # snow / rain falling over the city


# A stable pseudo-random value in 0..1 from two ints and a salt (no per-window storage).
# Used to assign cluster harmonics and per-window light thresholds deterministically.
func _whash(a: int, b: int, salt: float) -> float:
	var v := sin(float(a) * 127.1 + float(b) * 311.7 + salt * 7.13) * 43758.5453
	return v - floor(v)


# A grid of lit windows on one building face; which are lit drifts with time and
# flares with loudness/beat.
func _windows(x: float, top: float, w: float, ground: float, bd: Dictionary, loud: float, depth: float) -> void:
	var cols := int(bd.wcols)
	var rows := int(bd.wrows)
	var pad := w * 0.16
	var cell_w := (w - pad * 2.0) / float(cols)
	var cell_h := minf(cell_w, (ground - top) / float(rows + 1))
	var ww := cell_w * 0.6
	var wh := cell_h * 0.6
	# Lights cluster and switch on/off SPARSELY, tied to harmonics - not a linear sweep. The
	# windows are grouped into a few blocks; each block listens to its own spectral band and
	# breathes on its own slow phase, and within an active block only a sparse subset (those
	# with a low intrinsic threshold) actually light - so clusters glow together and fade as
	# their harmonic comes and goes, the way real windows switch room by room.
	var cw := maxi(1, cols / 2)          # a cluster spans ~half the columns ...
	var crows := maxi(2, rows / 3)       # ... and ~a third of the rows
	var phase := float(bd.phase)
	for wy in rows:
		var wy_top := top + pad + wy * cell_h
		if wy_top + wh > ground:
			break
		for wx in cols:
			var cx := wx / cw
			var cy := wy / crows
			var cband := _whash(cx + 3, cy + 5, phase)     # this cluster's harmonic
			var cloud := _f.sample(cband)                  # how active that harmonic is now
			var breathe := 0.5 + 0.5 * sin(_f.time * (0.20 + 0.5 * cband) + _whash(cx, cy, phase) * TAU)
			# A baseline presence (so a city is never fully dark) that the harmonic boosts; the
			# slow breathing makes clusters switch on and off over time, beats add a flare.
			var base := 0.30 + 0.70 * cloud
			var activation := base * (0.35 + 0.65 * breathe) + 0.30 * _f.beat * cloud + 0.12 * loud
			var thr := _whash(wx * 2 + 1, wy * 2 + 7, phase)   # per-window sparsity
			if activation > 0.12 + 0.7 * thr:
				var bv := clampf(0.55 + 0.5 * activation, 0.0, 1.0)
				# Lit windows sit on the scheme's ACCENT - the contrast against the block
				# mass is the mood's own complement, not a second unrelated hue.
				var col := _sch.color(fposmod(_sch.accent + 0.05 * cband, 1.0), 0.5,
					bv * (0.7 + 0.3 * depth) * 1.6, 0.9)
				var wx_left := x + pad + wx * cell_w
				draw_rect(Rect2(wx_left, wy_top, ww, wh), col)
