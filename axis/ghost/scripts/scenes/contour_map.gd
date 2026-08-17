extends GhostScene

## Contour map - the terrain from directly above, as a printed survey sheet.
##
## Warm paper, almost white, with thin dark isolines threading it at a fixed interval:
## the first time in this catalogue that land is seen from straight overhead. Every fifth
## line is an index contour drawn heavier, the band above the inked index line is ruled
## with fine diagonal hatching, a graticule of ticks runs the margins with a scale bar in
## one corner, and small survey crosses sit on local summits, each with a leader line out
## to a two-glyph label in an invented script. The land warps slowly underneath, so the
## lines migrate like a time-lapse coastline, and a survey sweep crosses the sheet
## planting marks as it goes and clearing them when it comes round again.
##
## THREE ABSENCES AT ONCE, which is why it exists. Nothing else here is a PLAN VIEW -
## every other framing is eye-level horizon, three-quarter isometric or centred abstract.
## Nothing else is HIGH-KEY. And nothing else is a DIAGRAM: the nearest thing is
## `projection`, which has no axis, tick, label, legend or rule line anywhere in it. This
## is the same substance as `terrain` - a [Field] heightfield - rendered as line art on
## paper, which is a different visual language and not a recolour.
##
## THE CONTOUR INTERVAL NEVER CHANGES, and that restraint is the whole point. A map with
## a throbbing contour interval is a lie: an interval is a claim about the land, so it is
## chosen ONCE, from the land itself (the datum and the interval are solved from the
## observed elevation range at first sight, then frozen for the session), and no audio
## feature is allowed anywhere near it. Nothing on this sheet changes SIZE.
##
## WHAT THE MUSIC MOVES INSTEAD.
##   The INKED INDEX CONTOUR - the one elevation drawn heaviest, in the accent - is
##   chosen by `chroma_hue().x` mapped across the elevation range, so the harmony picks
##   which altitude is inked and a chord change re-prints the sheet at a different height.
##   `f.flux` sets the hatch spacing between index lines. Flux measures 0.01 to 0.05 in
##   practice, so it is scaled by 12 to cross its sampled band at all rather than being
##   used as if it reached 1.
##   The SWEEP advances one sheet width per `f.beat_period` x a sampled [8,32] beats, and
##   a `f.beat` rising edge stamps the next survey cross - at the summit nearest the
##   sweep, so marks appear along the line as it passes.
##   `f.movement` over a sampled threshold eases the field offset to the next entry of a
##   pre-rolled ring over 6 to 15 seconds. The offset is a position in an infinite noise
##   field, so easing to a new one is a slow flight across a continent: the coastline
##   genuinely reorganises rather than sliding, and it does so for a fraction of the
##   machinery an erosion solver would cost.
##   `f.high` raises the graticule and annotation ink alpha, so the paperwork layer fades
##   in on bright passages and the sheet is mostly bare land on quiet ones.
##   `f.energy` does nothing at all except warm the paper through the tint, because it is
##   a mean over 64 bands that rarely passes 0.5 and there is nothing here it should move.
##
## THE COST, AND THE CADENCE. Extraction is the expense: sampling the field onto the grid
## is one GDScript call per sample and the marching squares pass is another few thousand,
## which together are tens of milliseconds - a frame budget, not a frame. It does not have
## to run at 60 Hz, because the land warps over seconds: the whole sheet is re-extracted
## on a sampled [0.3, 1.0] s cadence inside a [FrameForge] job and the finished packet is
## re-submitted every frame for microseconds. That also gives the sheet its character - a
## printed map that RE-PRINTS a couple of times a second rather than animating. The
## grid is about 128 samples across, which puts a cell near a dozen pixels; two rounds of
## Chaikin corner cutting in [Contour] turn the resulting faceted polyline into a curve,
## and a simplify pass hands the points back. That pair is why a grid coarse enough to
## sample in a worker still draws like a pen. In an export the forge builds synchronously,
## so a render pays the extraction on the main thread - slower to render, identical output.
##
## THE HIGH-KEY GROUND fights three separate dark assumptions in this project (the
## project's clear colour, [Layer]'s bed being a vignette whose brightest pixel is a mid
## tone, and the veil's alpha cap), so the paper is painted by [method
## GhostScene.paint_ground] - a flat full-bleed quad through plain `draw_colored_polygon`.
## Not `fill_aa`: that strokes the outline over the fill, and on a full-frame quad the rim
## lands exactly on the frame edge as a bright hairline. The margin is then painted back
## OVER the contours in the same colour, which is what gives the sheet a real neatline and
## a clean white margin for four quads.
##
## WHAT THE SEED DECIDES: the field kind (fbm / ridged / billow / cells), its octaves,
## frequency and whether it is domain-warped; how much of the continent the sheet covers;
## the number of contour levels and which of them are index lines; the minor and index pen
## widths; whether the sheet is cream paper with dark ink or a blueprint (a deep blue
## ground with white ink, roughly one sheet in four); the ink itself from a [Scheme] at a
## quarter of its nominal saturation, because a survey is drawn in one restrained colour;
## the hatch angle and its spacing band; the sea level and whether there is water at all;
## the graticule divisions and margin; how many survey marks the sheet may carry; the
## re-extraction cadence; whether a second colour plate prints slightly out of register;
## the sweep's period in beats; the section-change threshold and how long a warp takes;
## and the whole invented hand the labels are written in.
##
## HONEST DETERMINISM CAVEAT, the same one `glyphs` carries. The land, the ladder, the
## palette, the alphabet and the ring of warp offsets are all seed-derived and reproduce
## exactly. WHICH summits get marked and when does not: the live analyzer and the offline
## bake do not produce identical beat streams. No rng is ever drawn on an audio-conditioned
## event - the warp offsets come from a pre-rolled ring taken by a counter - so nothing
## else can drift.

## How far past the frame the sampled sheet extends, for camera-drift headroom.
const SHEET := 1.22

## Overdraw for the margin cover, matching the layers' own 1.15 convention with a little
## more room because this one must never expose a corner.
const OVER := 1.22

## The hatch march, in grid cells. Half a cell is a few pixels; every run end is then
## refined by bisection inside [method Contour.hatch], so the band edges land on the
## contour rather than on the march grid.
const HATCH_MARCH := 0.5

## Simplify tolerance in pixels, after smoothing. A third of a pixel is below what the
## feather can show, so this only ever removes points the eye could not have seen.
const SIMPLIFY_PX := 0.35

## The land. Each reads as a different country: fbm is rolling hills, ridged is a
## mountain chain, billow is dunes, cells is a fractured plateau of scarps.
const KINDS := ["fbm", "fbm", "ridged", "billow", "cells"]

var _forge: FrameForge
# The warp the packet CURRENTLY ON SCREEN was built at. Everything drawn live is placed
# against this, and the packet itself is translated by the difference to the live warp -
# see _draw. Without it the marks slid smoothly while the contours stepped on the
# extraction cadence, which is what "the plane kept jumping" and "a compass detached and
# floating around" both were: annotations adrift from the map they annotate.
#
# IT MUST BE THE WARP OF THE PACKET ON SCREEN, NOT OF THE NEWEST BUILD, and the difference
# between those two is a whole re-print's worth of drift. This used to come from a slot each
# job wrote at the end of run() - on the WORKER, while the packet it belonged to was still
# waiting on a deferred call to reach the main thread. Any frame drawn in that window
# compensated the old sheet by the new offset, so the map stepped sideways by about 1.5% of
# its width and stepped back on the next frame: a hard jump on most re-prints, which is
# exactly what "the whole scene jumps, then shifts back like a correction" was. It is read
# off [method FrameForge.packet_source] now, which is swapped WITH the packet in one
# assignment on one thread and therefore cannot be ahead of it.
var _pkt_warp := Vector2.ZERO
var _pkt_valid := false
var _sim := SimClock.new(60.0)
var _tb := TriBatch.new()
var _prng := RandomNumberGenerator.new()
var _fld: Field
var _sch: Scheme
var _gs: GlyphSet
var _f: AudioFeatures = AudioFeatures.new()
var _ch := Vector2.ZERO

# The sampled definition, unpacked into fields the per-frame code reads without a
# Dictionary lookup.
var _res := 128
var _levels := 14
var _index_every := 5
var _index_phase := 0
var _w_minor := 1.1
var _w_index := 2.4
var _w_pick := 3.2
var _span := 2.0
var _blueprint := false
var _paper_h := 0.09
var _paper_s := 0.07
var _paper_v := 0.92
var _ink_h := 0.08
var _ink_s := 0.30
var _ink_v := 0.16
var _tint := 0.30
var _hatch_deg := 45.0
var _hatch_lo := 5.0
var _hatch_hi := 12.0
var _hatch_on := true
var _sea_frac := 0.18
var _water := true
var _divs := 10
var _grat_every := 5
var _marks_max := 7
var _cadence := 0.5
var _plate := false
var _plate_px := 1.2
var _sweep_beats := 16.0
var _sweep_trail := 0.10
var _mv_thr := 0.35
var _warp_secs := 9.0
var _margin := 0.05
var _grat_lo := 0.35
var _grat_hi := 0.85
var _smooth := 2
var _cross_r := 0.010
var _leader := 0.030
var _shelf := 0.028
var _scale_cells := 4
var _scale_w := 0.16

# The elevation ladder, solved once from the land and then frozen (see the class doc).
var _lo := 0.1
var _step_v := 0.06
var _sea := 0.2
var _ladder := false

# State.
var _ring := PackedVector2Array()
var _ring_i := 0
var _warp_a := Vector2.ZERO
var _warp_b := Vector2.ZERO
var _warp_t := 1.0
var _sites: Array = []              # {n: Vector2, h: float, g: PackedInt32Array}
var _site_key := Vector2.ZERO
var _marks: Array = []              # {i: int, age: float, side: float}
var _ext_t := 999.0
var _sweep_p := 0.0
var _sweep_prev := 0.0
var _prev_beat := 0.0
var _prev_mv := 0.0
var _pick := 0
var _pick_k := 0.0          # the highlight's position on the index ladder, eased
var _pick_seeded := false
var _pick_level := 0        # the committed index line, moved only past a hysteresis band

# The label pass: anchors collected while the paperwork is drawn, flushed as one textured
# triangle array afterwards (the same ramp-texture path [GlyphSet] bakes for).
var _lab: Array = []
var _lpts := PackedVector2Array()
var _lcols := PackedColorArray()
var _luvs := PackedVector2Array()
var _lidx := PackedInt32Array()


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	_forge = FrameForge.new()
	_prng.seed = rng.randi()
	_sch = Scheme.pick(rng)

	var kind := String(KINDS[rng.randi() % KINDS.size()])
	var oct := rng.randi_range(3, 6)
	var freq := rng.randf_range(1.2, 3.4)
	_fld = Field.make(kind, rng.randi(), freq, oct)
	# Domain warp on half the sheets. It is what turns clean noise into land that looks
	# eroded, and it is also the single biggest sampling cost here - a warped field is
	# three noise evaluations per sample instead of one - so it is a coin flip rather
	# than a constant, and the grid is sized for the warped case.
	var warp_amt := 0.0
	if rng.randf() < 0.5:
		warp_amt = rng.randf_range(0.08, 0.35)
		_fld = _fld.warp(Field.make("fbm", rng.randi(), freq * 0.45, 2), warp_amt)
	_span = rng.randf_range(1.1, 2.6)
	_res = rng.randi_range(112, 144)

	_levels = rng.randi_range(8, 22)
	_index_every = rng.randi_range(4, 6)
	_index_phase = rng.randi_range(0, _index_every - 1)
	_w_minor = rng.randf_range(0.8, 1.6)
	_w_index = rng.randf_range(1.8, 3.2)
	_w_pick = _w_index * rng.randf_range(1.15, 1.5)

	_blueprint = rng.randf() < 0.25
	if _blueprint:
		# The inversion: a cyanotype is light-on-dark, so the ground is deep and the ink
		# is the pale thing on it. Still a FLAT full-bleed ground, which is the property
		# that matters - a vignette would read as a spotlight on a wall.
		_paper_h = rng.randf_range(0.55, 0.60)
		_paper_s = rng.randf_range(0.50, 0.72)
		_paper_v = rng.randf_range(0.20, 0.31)
		_ink_h = fposmod(_paper_h + rng.randf_range(-0.04, 0.04), 1.0)
		_ink_s = rng.randf_range(0.04, 0.14)
		_ink_v = rng.randf_range(0.90, 1.0)
	else:
		_paper_h = rng.randf_range(0.06, 0.13)
		_paper_s = rng.randf_range(0.04, 0.12)
		_paper_v = rng.randf_range(0.86, 0.96)
		_ink_h = _sch.vary(rng, 0.6)
		_ink_s = clampf(_sch.sat * rng.randf_range(0.15, 0.50), 0.05, 0.7)
		_ink_v = rng.randf_range(0.10, 0.26)
	_tint = rng.randf_range(0.18, 0.42)

	_hatch_on = rng.randf() < 0.85
	_hatch_deg = rng.randf_range(30.0, 60.0) * (1.0 if rng.randf() < 0.5 else -1.0)
	_hatch_lo = rng.randf_range(4.0, 8.0)
	_hatch_hi = _hatch_lo + rng.randf_range(2.0, 7.0)

	_water = rng.randf() < 0.7
	_sea_frac = rng.randf_range(0.0, 0.35)
	_divs = rng.randi_range(6, 14)
	_grat_every = rng.randi_range(2, 5)
	_marks_max = rng.randi_range(3, 12)
	_cadence = rng.randf_range(0.30, 1.0)
	_plate = rng.randf() < 0.3
	_plate_px = rng.randf_range(0.6, 2.2)
	_sweep_beats = rng.randf_range(8.0, 32.0)
	_sweep_trail = rng.randf_range(0.05, 0.18)
	# Movement is a section-change score that spends most of a track low, so a threshold
	# near 0.5 rewarps only at real changes and one near 0.2 keeps the land always moving.
	_mv_thr = rng.randf_range(0.20, 0.50)
	# ... and slower, for the same reason.
	_warp_secs = rng.randf_range(14.0, 30.0)
	_margin = rng.randf_range(0.030, 0.075)
	_grat_lo = rng.randf_range(0.22, 0.45)
	_grat_hi = minf(1.0, _grat_lo + rng.randf_range(0.25, 0.50))
	_smooth = rng.randi_range(1, 3)
	_cross_r = rng.randf_range(0.008, 0.014)
	_leader = rng.randf_range(0.022, 0.045)
	_shelf = rng.randf_range(0.020, 0.040)
	_scale_cells = rng.randi_range(3, 6)
	_scale_w = rng.randf_range(0.11, 0.20)

	# A small, tight hand: map lettering is the one place an invented script has to sit
	# quietly beside the drawing rather than being the subject.
	_gs = GlyphSet.new(rng, {
		"advance": rng.randf_range(0.011, 0.019),
		"aspect": rng.randf_range(1.15, 1.65),
		"weight": rng.randf_range(1.0, 2.0),
		"count": rng.randi_range(14, 26),
	})

	# The ring of warp offsets, in sheet fractions. ABSOLUTE positions rather than
	# accumulating deltas, so the sheet wanders inside a bounded region of the continent
	# and the summit table stays valid; and pre-rolled, because the change that picks the
	# next one is audio-conditioned and an rng draw there would diverge in an export.
	var n_ring := rng.randi_range(4, 9)
	for i in n_ring:
		# MUCH SMALLER THAN IT WAS. These were +-0.55 of a sheet, so two consecutive
		# entries could be a full sheet-width apart and the map crossed itself in the ease
		# time - a lurch rather than a survey. A map should be a stable thing you can read,
		# with the camera's own gentle drift providing the motion.
		_ring.append(Vector2(rng.randf_range(-0.16, 0.16), rng.randf_range(-0.16, 0.16)))
	_warp_a = _ring[0]
	_warp_b = _ring[0]

	return {
		"land": kind,
		"octaves": oct,
		"freq": freq,
		"warp": warp_amt,
		"span": _span,
		"grid": _res,
		"levels": _levels,
		"index_every": _index_every,
		"width_minor": _w_minor,
		"width_index": _w_index,
		"sheet": "blueprint" if _blueprint else "paper",
		"mood": _sch.name,
		"paper_hue": _paper_h,
		"paper_val": _paper_v,
		"ink_hue": _ink_h,
		"ink_sat": _ink_s,
		"hatch": _hatch_on,
		"hatch_deg": _hatch_deg,
		"hatch_px": "%.1f-%.1f" % [_hatch_lo, _hatch_hi],
		"water": _water,
		"sea_frac": _sea_frac,
		"graticule": _divs,
		"margin": _margin,
		"marks": _marks_max,
		"cadence": _cadence,
		"second_plate": _plate,
		"sweep_beats": _sweep_beats,
		"move_threshold": _mv_thr,
		"warp_seconds": _warp_secs,
		"smooth": _smooth,
		"alphabet": _gs.count(),
		"warp_ring": n_ring,
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	# A printed sheet barely moves. It is being READ, not flown over, and the moment the
	# camera breathes it stops being paper.
	drift_view(f, 0.02, 0.02)
	update_layers(f, delta)
	queue_redraw()
	if size.x < 4.0:
		return
	_ensure_sites()
	_ch = chroma_hue()
	# Events, not integration: an edge stays one edge however many times the Director
	# substeps this frame.
	if f.beat > 0.55 and _prev_beat <= 0.55:
		_stamp()
	_prev_beat = f.beat
	if f.movement > _mv_thr and _prev_mv <= _mv_thr:
		_rewarp()
	_prev_mv = f.movement
	for _i in _sim.ticks(delta):
		_step(_sim.dt)


# Everything that integrates time runs here, on the fixed clock: the warp ease, the
# sweep, the marks' fade and the re-extraction cadence. The Director substeps up to
# fifteen times in a frame, pre-warms twelve deep and lets Echo fast-forward, and a sweep
# advanced once per update() call would have crossed the sheet before its first frame.
func _step(dt: float) -> void:
	# The highlight migrates on the FIXED clock like everything else that integrates: it
	# is called from here rather than from update(), which the Director substeps up to
	# fifteen times in one frame and pre-warms twelve deep.
	_pick_index(dt)
	if _warp_t < 1.0:
		_warp_t = minf(1.0, _warp_t + dt / maxf(0.5, _warp_secs))
	var period := maxf(2.0, clampf(_f.beat_period, 0.25, 1.6) * _sweep_beats)
	_sweep_prev = _sweep_p
	_sweep_p = fposmod(_sweep_p + dt / period, 1.0)
	_age(dt)
	_ext_t += dt
	if _ext_t >= _cadence:
		_ext_t = 0.0
		_kick()


# ---------------------------------------------------------------------------------
# The land: the datum, the summit table, and the warp
# ---------------------------------------------------------------------------------

# Sample a coarse grid over the whole region the sheet can ever wander across, and take
# two things from it that are then fixed for the session: the ELEVATION LADDER (the datum
# and the interval, solved from the observed range so a flat country still gets a
# populated sheet and a mountainous one is not over-drawn), and the SUMMIT TABLE.
#
# It runs here rather than in build_params because both need the frame's aspect, and size
# is zero until the scene is in the tree. Once only - the whole point of a survey is that
# it was made once.
func _ensure_sites() -> void:
	var key := Vector2(size.x, size.y)
	if not _sites.is_empty() and key.distance_to(_site_key) < 8.0:
		return
	_site_key = key
	_sites.clear()
	_marks.clear()
	var u := maxf(1.0, unit())
	var fspan := Vector2(size.x, size.y) * (SHEET * _span / u)
	# The sheet occupies -0.5..0.5; the warp ring reaches 0.55 either way, so the table
	# has to cover a good deal more than the sheet or a wandering map runs out of summits.
	var dom := 1.20
	var nx := 56
	var ny := 36
	var h := PackedFloat32Array()
	h.resize(nx * ny)
	var lo := 1.0
	var hi := 0.0
	for y in ny:
		var qy := (float(y) / float(ny - 1) - 0.5) * 2.0 * dom * fspan.y
		for x in nx:
			var qx := (float(x) / float(nx - 1) - 0.5) * 2.0 * dom * fspan.x
			var v := _fld.at(Vector2(qx, qy))
			h[y * nx + x] = v
			lo = minf(lo, v)
			hi = maxf(hi, v)
	# The ladder. Pulled a little inside the observed range so the top and bottom levels
	# are not single specks, then the interval is that span divided evenly - and neither
	# number is ever touched again.
	var pad := (hi - lo) * 0.06
	var d_lo := lo + pad
	var d_hi := maxf(d_lo + 0.02, hi - pad)
	_step_v = (d_hi - d_lo) / float(_levels + 1)
	_lo = d_lo + _step_v
	_sea = d_lo + _sea_frac * (d_hi - d_lo)
	_ladder = true

	# Summits: strict local maxima of the coarse grid, tallest first. Their positions are
	# stored in SHEET FRACTIONS of the continent, so tracking one under a warp is a
	# subtraction - the offset moves the window, the land does not move.
	var found: Array = []
	for y in range(1, ny - 1):
		for x in range(1, nx - 1):
			var v := h[y * nx + x]
			if v < d_lo + (d_hi - d_lo) * 0.45:
				continue
			var top := true
			for dy in range(-1, 2):
				for dx in range(-1, 2):
					if (dx != 0 or dy != 0) and h[(y + dy) * nx + x + dx] >= v:
						top = false
			if not top:
				continue
			found.append({
				"n": Vector2(float(x) / float(nx - 1) - 0.5, float(y) / float(ny - 1) - 0.5) * 2.0 * dom,
				"h": v,
			})
	found.sort_custom(func(a, b): return float(a["h"]) > float(b["h"]))
	var cap := mini(found.size(), 48)
	for i in cap:
		var e: Dictionary = found[i]
		var g := PackedInt32Array()
		# Every station is NAMED, once, from the build rng - so the same peak carries the
		# same two characters every time it is marked, which is what makes the sheet read
		# as a record rather than as decoration.
		g.append(_prng.randi() % maxi(1, _gs.count()))
		g.append(_prng.randi() % maxi(1, _gs.count()))
		e["g"] = g
		_sites.append(e)


## Take the warp offset of the build whose packet is currently on screen. Call it after the
## forge has flushed, or in an export - where the build happens inside the draw - it would
## read the PREVIOUS sheet's offset against this sheet's lines.
func _adopt_packet_warp() -> void:
	var job := _forge.packet_source() as SheetJob
	if job != null:
		_pkt_warp = job.warp
		_pkt_valid = true


func _warp_now() -> Vector2:
	var t := clampf(_warp_t, 0.0, 1.0)
	return _warp_a.lerp(_warp_b, t * t * (3.0 - 2.0 * t))


# A section change eases the window to the next pre-rolled position. A change arriving
# mid-flight is ignored rather than restarting the move: the land is meant to drift for
# the whole 6 to 15 seconds, and re-aiming it every few bars would read as a jitter.
func _rewarp() -> void:
	if _warp_t < 1.0 or _ring.is_empty():
		return
	_warp_a = _warp_b
	_ring_i = (_ring_i + 1) % _ring.size()
	_warp_b = _ring[_ring_i]
	_warp_t = 0.0


func _site_pos(i: int) -> Vector2:
	var d: Dictionary = _sites[i]
	var n: Vector2 = d["n"]
	var o := _warp_now()
	return Vector2((n.x - o.x) * size.x * SHEET, (n.y - o.y) * size.y * SHEET)


# ---------------------------------------------------------------------------------
# The audio: which contour is inked, and the survey marks
# ---------------------------------------------------------------------------------

# The harmony picks an ALTITUDE, and it MIGRATES there rather than arriving.
#
# chroma_hue's angle is the music's tonal centre, and it used to be mapped straight across
# the index contours: `_pick = first + int(t * n) * _index_every`. Two things made that the
# worst mark on the sheet. The inked line is the one saturated colour here and it appears
# at EVERY occurrence of its elevation, so it is not one loop but loops all over the map -
# and an instant re-pick relights all of them somewhere else at once. Worse, hue is
# CIRCULAR while an elevation ladder is not, so a small harmonic move across the wrap
# (0.99 to 0.01) threw the highlight from the top of the country to the bottom. Reported
# as pink loops appearing and vanishing every few seconds, and reading as the whole map
# jumping - which it effectively was, since the pink was most of what the eye tracked.
#
# So the pick eases through the ladder at a bounded rate instead. One index line at a time,
# to an ADJACENT elevation, which on a contour map reads as a tide line rising and falling
# over the land - continuous, legible, and exactly the kind of motion a map can carry.
# A hue wrap now costs a slow sweep across the ladder rather than a flash.
## Index lines per second the highlight may travel. Deliberately slow, and the slowness is
## doing two jobs. A sheet may only carry two or three index contours, so "adjacent" is
## still a large visual change - every loop at that elevation, everywhere on the map. And
## a slow crossing gives free hysteresis: a fleeting harmonic wobble never completes the
## trip, so the highlight settles instead of oscillating across a boundary. Four seconds
## per line is a tide, not a switch.
const PICK_RATE := 0.25

func _pick_index(dt: float) -> void:
	var first := posmod(-_index_phase, _index_every)
	var n := 0
	var k := first
	while k < _levels:
		n += 1
		k += _index_every
	if n <= 0:
		_pick = _levels / 2
		return
	var t := _ch.x if _ch.y > 0.02 else 0.5
	var target := clampf(t * float(n), 0.0, float(n - 1))
	if not _pick_seeded:
		_pick_k = target
		_pick_level = clampi(int(round(target)), 0, n - 1)
		_pick_seeded = true
	else:
		_pick_k = move_toward(_pick_k, target, PICK_RATE * dt)
	# HYSTERESIS, not rounding. A slow travel rate alone does NOT settle this - measured,
	# it changed just as often. With the target alternating between two levels the eased
	# position simply hovers near the half-way mark, and round() then flips on every
	# micro-crossing: 21 changes in twenty seconds, which is the flashing being fixed.
	# The line only moves once the position has committed three quarters of the way to a
	# neighbour, so a hovering target holds the current elevation instead of straddling it.
	if absf(_pick_k - float(_pick_level)) > 0.75:
		_pick_level = clampi(_pick_level + (1 if _pick_k > float(_pick_level) else -1), 0, n - 1)
	_pick = first + clampi(_pick_level, 0, n - 1) * _index_every


# A beat plants the next station: the summit nearest the sweep line, so marks appear
# along it as it crosses. No rng - the choice is a function of the feature stream alone.
func _stamp() -> void:
	if _sites.is_empty() or size.x < 4.0:
		return
	var nl := _neatline()
	var sx := lerpf(-nl.x, nl.x, _sweep_p)
	var best := -1
	var best_d := INF
	for i in _sites.size():
		var p := _site_pos(i)
		if absf(p.x) > nl.x * 0.92 or absf(p.y) > nl.y * 0.86:
			continue
		var taken := false
		for m in _marks:
			var md: Dictionary = m
			if int(md["i"]) == i:
				taken = true
		if taken:
			continue
		var d := absf(p.x - sx)
		if d < best_d:
			best_d = d
			best = i
	if best < 0:
		return
	# The leader leaves on the side with more room, so a label never runs off the sheet.
	var pos := _site_pos(best)
	_marks.append({"i": best, "age": 0.0, "side": -1.0 if pos.y > 0.0 else 1.0})
	while _marks.size() > _marks_max:
		_marks.remove_at(0)


# Marks ink in behind the sweep and are cleared when it comes round to them again, so the
# sheet is continuously being re-surveyed rather than filling up and staying full.
func _age(dt: float) -> void:
	if _marks.is_empty():
		return
	var nl := _neatline()
	var span := maxf(1.0, nl.x * 2.0)
	var i := _marks.size() - 1
	while i >= 0:
		var m: Dictionary = _marks[i]
		m["age"] = float(m["age"]) + dt
		var p := _site_pos(int(m["i"]))
		var mx := clampf((p.x + nl.x) / span, 0.0, 1.0)
		var crossed := false
		if _sweep_p >= _sweep_prev:
			crossed = mx > _sweep_prev and mx <= _sweep_p
		else:
			crossed = mx > _sweep_prev or mx <= _sweep_p
		# The age guard is what stops a mark planted ON the line from being erased by the
		# same pass that planted it.
		if (crossed and float(m["age"]) > 0.75) or absf(p.x) > nl.x or absf(p.y) > nl.y:
			_marks.remove_at(i)
		i -= 1


# ---------------------------------------------------------------------------------
# The sheet: one worker job per cadence tick
# ---------------------------------------------------------------------------------

func _kick() -> void:
	if not _ladder or size.x < 4.0:
		return
	var u := maxf(1.0, unit())
	var nx := _res
	var ny := _res
	# Square-ish cells: a square grid over a 16:9 frame samples twice as finely down the
	# frame as across it, and the faceting on the coarse axis is what the eye finds.
	if size.x >= size.y:
		ny = maxi(24, int(round(float(_res) * size.y / maxf(1.0, size.x))))
	else:
		nx = maxi(24, int(round(float(_res) * size.x / maxf(1.0, size.y))))

	var job := SheetJob.new()
	job.fld = _fld
	job.nx = nx
	job.ny = ny
	job.half = Vector2(size.x, size.y) * 0.5 * SHEET
	job.fspan = Vector2(size.x, size.y) * (SHEET * _span / u)
	job.off = _warp_now() * job.fspan
	job.warp = _warp_now()
	job.lo = _lo
	job.step = _step_v
	job.count = _levels
	job.index_every = _index_every
	job.index_phase = _index_phase
	job.pick = _pick
	job.smooth_iters = _smooth
	job.simp = SIMPLIFY_PX

	var ink := _ink_color(1.0)
	job.minor_col = Color(ink.r, ink.g, ink.b, 0.72)
	job.index_col = ink
	job.pick_col = _pick_color()
	job.w_minor = _w_minor
	job.w_index = _w_index
	job.w_pick = _w_pick
	job.plate = _plate
	job.plate_off = Vector2(_plate_px, -_plate_px * 0.6)
	var pl := _sch.color(_sch.opposed(_ink_h, 0.30), 0.9, 1.0)
	job.plate_col = Color(pl.r, pl.g, pl.b, 0.30)

	job.water = _water
	job.sea = _sea
	job.water_col = _water_color()

	job.hatch_on = _hatch_on
	job.hatch_a = deg_to_rad(_hatch_deg)
	# Flux is the density knob and it measures 0.01 to 0.05, so it takes a gain near 12 to
	# cross the sampled spacing band at all. Denser ruling on a busy passage.
	job.hatch_sp = lerpf(_hatch_hi, _hatch_lo, clampf(_f.flux * 12.0, 0.0, 1.0))
	job.hatch_col = Color(ink.r, ink.g, ink.b, 0.34)
	job.hatch_w = maxf(0.7, _w_minor * 0.8)
	job.band_lo = _lo + float(_pick) * _step_v
	job.band_hi = job.band_lo + float(_index_every) * _step_v
	_forge.kick(job.run, {}, self, job)


# ---------------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------------

func _paper_color() -> Color:
	# The identical shift paint_ground performs internally, so the margin cover and the
	# ground can never come out as two slightly different whites.
	var d := _ch.x - _paper_h
	var h := fposmod(_paper_h + (d - round(d)) * clampf(_tint * _ch.y, 0.0, 1.0) * 0.5, 1.0)
	return Color.from_hsv(h, _paper_s, _paper_v)


func _ink_color(a: float) -> Color:
	return Color.from_hsv(fposmod(_ink_h, 1.0), _ink_s, _ink_v, clampf(a, 0.0, 1.0))


func _pick_color() -> Color:
	# The inked index line takes the accent, pulled toward the music's tonal centre by
	# however tonal the moment is. This is the one saturated mark on the sheet.
	var h := _sch.opposed(_ink_h, 0.22)
	var d := _ch.x - h
	h = fposmod(h + (d - round(d)) * _ch.y * 0.5, 1.0)
	if _blueprint:
		return Color.from_hsv(h, clampf(_ink_s + 0.35, 0.0, 1.0), 1.0, 0.80)
	# 0.80 rather than 0.95: this line is not one mark but every loop at its elevation, so
	# it covers far more of the sheet than "the one saturated mark" suggests.
	return Color.from_hsv(h, clampf(_ink_s + 0.40, 0.0, 1.0), clampf(_ink_v + 0.28, 0.0, 0.8), 0.80)


func _water_color() -> Color:
	var h := _sch.opposed(_ink_h, 0.26)
	if _blueprint:
		return Color.from_hsv(h, clampf(_paper_s * 0.6, 0.0, 1.0), clampf(_paper_v + 0.12, 0.0, 1.0), 0.45)
	return Color.from_hsv(h, clampf(_ink_s * 0.9 + 0.10, 0.0, 1.0), 0.90, 0.36)


# The paperwork's ink alpha rides f.high: bright passages bring the annotation layer up,
# quiet ones leave the sheet as bare land. f.high is a band mean and rarely passes 0.5,
# hence the gain.
func _paperwork_alpha() -> float:
	return lerpf(_grat_lo, _grat_hi, clampf(_f.high * 2.2, 0.0, 1.0))


func _neatline() -> Vector2:
	var m := _margin * maxf(1.0, unit())
	return Vector2(maxf(20.0, size.x * 0.5 - m), maxf(20.0, size.y * 0.5 - m))


# ---------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------

func _draw() -> void:
	begin_draw()
	paint_ground(_paper_h, _paper_s, _paper_v, _tint * _ch.y, _ch.x)
	if size.x < 4.0:
		return
	draw_layers("back")
	# Flush FIRST: in an export the sheet is built right here, inside the draw, and the offset
	# has to be that build's rather than the one before it.
	_forge.flush()
	_adopt_packet_warp()
	# GLIDE THE SHEET. The warp is a pure translation of the sampling window - the offset
	# moves the window, the land does not move - so displaying a packet built at one offset
	# as though it were built at another is EXACT, not an approximation, everywhere except
	# the strip of new ground at the leading edge that the next extraction brings in.
	#
	# That is what turns a re-extraction from a jump into a refresh: the map slides
	# continuously at the warp's own speed and the lines are simply redrawn underneath it
	# every cadence. Before this the sheet sat still for up to a second and then snapped
	# several percent of its width sideways.
	var d := _warp_now() - _pkt_warp if _pkt_valid else Vector2.ZERO
	if d != Vector2.ZERO:
		draw_set_transform_matrix(view.matrix(size) * Transform2D(0.0, Vector2.ONE, 0.0,
			Vector2(-d.x * size.x * SHEET, -d.y * size.y * SHEET)))
	_forge.submit(self)          # water, hatching, the plates, every contour
	if d != Vector2.ZERO:
		draw_set_transform_matrix(view.matrix(size))
	_paperwork()
	_labels()
	draw_layers("front")


func _paperwork() -> void:
	_lab.clear()
	var nl := _neatline()
	var u := maxf(1.0, unit())
	var paper := _paper_color()
	var a := _paperwork_alpha()
	var ink := _ink_color(a)
	var hard := _ink_color(minf(1.0, a + 0.15))

	# The sweep, over the land but under the margin - so it is clipped by the neatline
	# exactly like everything else printed on the sheet.
	var sx := lerpf(-nl.x, nl.x, _sweep_p)
	var wash := _pick_color()
	var trail := maxf(4.0, nl.x * 2.0 * _sweep_trail)
	var x0 := maxf(-nl.x, sx - trail)
	if sx > -nl.x:
		var w0 := Color(wash.r, wash.g, wash.b, 0.0)
		var w1 := Color(wash.r, wash.g, wash.b, 0.10)
		_tb.quad_colored(
			PackedVector2Array([Vector2(x0, -nl.y), Vector2(sx, -nl.y),
				Vector2(sx, nl.y), Vector2(x0, nl.y)]),
			PackedColorArray([w0, w1, w1, w0]))
	_tb.line(Vector2(sx, -nl.y), Vector2(sx, nl.y),
		Color(wash.r, wash.g, wash.b, 0.55), maxf(1.0, _w_index * 0.7), true)

	# The margin, painted back over the land: this is the neatline, and it is four quads.
	var ov := Vector2(size.x, size.y) * 0.5 * OVER / maxf(0.001, view.zoom_actual())
	_tb.quad(Vector2(-ov.x, -ov.y), Vector2(ov.x, -ov.y),
		Vector2(ov.x, -nl.y), Vector2(-ov.x, -nl.y), paper)
	_tb.quad(Vector2(-ov.x, nl.y), Vector2(ov.x, nl.y),
		Vector2(ov.x, ov.y), Vector2(-ov.x, ov.y), paper)
	_tb.quad(Vector2(-ov.x, -nl.y), Vector2(-nl.x, -nl.y),
		Vector2(-nl.x, nl.y), Vector2(-ov.x, nl.y), paper)
	_tb.quad(Vector2(nl.x, -nl.y), Vector2(ov.x, -nl.y),
		Vector2(ov.x, nl.y), Vector2(nl.x, nl.y), paper)

	Contour.box(_tb, -nl, nl, hard, _w_index * 0.8)
	var minor := u * 0.008
	var major := u * 0.016
	var gw := maxf(0.8, _w_minor)
	Contour.graticule(_tb, Vector2(-nl.x, -nl.y), Vector2(nl.x, -nl.y),
		Vector2(0.0, -1.0), _divs, minor, major, _grat_every, ink, gw)
	Contour.graticule(_tb, Vector2(-nl.x, nl.y), Vector2(nl.x, nl.y),
		Vector2(0.0, 1.0), _divs, minor, major, _grat_every, ink, gw)
	var vdiv := maxi(3, int(round(float(_divs) * size.y / maxf(1.0, size.x))))
	Contour.graticule(_tb, Vector2(-nl.x, -nl.y), Vector2(-nl.x, nl.y),
		Vector2(-1.0, 0.0), vdiv, minor, major, _grat_every, ink, gw)
	Contour.graticule(_tb, Vector2(nl.x, -nl.y), Vector2(nl.x, nl.y),
		Vector2(1.0, 0.0), vdiv, minor, major, _grat_every, ink, gw)

	# The scale bar, on its own patch of paper so it stays legible over hatching.
	var bar_w := nl.x * 2.0 * _scale_w
	var bar_h := u * 0.010
	var org := Vector2(-nl.x + u * 0.030, nl.y - u * 0.040)
	var pad := u * 0.016
	_tb.quad(org + Vector2(-pad, -pad * 1.6), org + Vector2(bar_w + pad, -pad * 1.6),
		org + Vector2(bar_w + pad, bar_h + pad), org + Vector2(-pad, bar_h + pad),
		Color(paper.r, paper.g, paper.b, 0.88))
	Contour.scale_bar(_tb, org, bar_w, _scale_cells, bar_h, hard, paper, gw)
	if _gs != null and _gs.count() > 0:
		var unit_g := PackedInt32Array()
		unit_g.append(0)
		unit_g.append(mini(1, _gs.count() - 1))
		_lab.append({"p": org + Vector2(bar_w + pad * 1.6, bar_h), "g": unit_g, "col": hard})

	# The stations.
	var cr := _cross_r * u
	for m in _marks:
		var md: Dictionary = m
		var p := _site_pos(int(md["i"]))
		var fade := clampf(float(md["age"]) / 0.45, 0.0, 1.0)
		var col := _ink_color(minf(1.0, a + 0.15) * fade)
		Contour.cross(_tb, p, cr, cr * 0.42, col, maxf(1.0, _w_index * 0.7))
		var side: float = md["side"]
		var out := 1.0 if p.x < 0.0 else -1.0
		var anchor := Contour.leader(_tb, p + Vector2(out * cr, side * cr * 0.3),
			Vector2(out, side * 0.85), _leader * u, _shelf * u, col, maxf(0.8, _w_minor))
		var d2: Dictionary = _sites[int(md["i"])]
		var g: PackedInt32Array = d2["g"]
		var lp := anchor + Vector2(0.0, -u * 0.004)
		if out < 0.0:
			lp.x -= _gs.em * u * 2.0 / maxf(0.2, _gs.aspect)
		_lab.append({"p": lp, "g": g, "col": col})

	_tb.flush(self)


# Every label on the sheet as one textured triangle array: each character is one native
# transform of its baked em-space geometry, and the stroke quads carry their own
# antialiasing in [GlyphSet]'s alpha ramp rather than in a per-line feather.
func _labels() -> void:
	if _lab.is_empty() or _gs == null or _gs.count() == 0:
		return
	_lpts.clear()
	_lcols.clear()
	_luvs.clear()
	var sc := _gs.em * maxf(1.0, unit())
	var adv := sc / maxf(0.2, _gs.aspect)
	for e in _lab:
		var d: Dictionary = e
		var p: Vector2 = d["p"]
		var col: Color = d["col"]
		var gi: PackedInt32Array = d["g"]
		var x := p.x
		for k in gi.size():
			var idx := clampi(gi[k], 0, _gs.count() - 1)
			var g: GlyphSet.Glyph = _gs.glyphs[idx]
			if g.verts.size() >= 4:
				var xf := Transform2D(0.0, Vector2(sc, sc), 0.0, Vector2(x, p.y))
				_lpts.append_array(xf * g.verts)
				_luvs.append_array(g.uvs)
				var c := PackedColorArray()
				c.resize(g.verts.size())
				c.fill(col)
				_lcols.append_array(c)
			x += adv
	var quads := _lpts.size() / 4
	if quads <= 0:
		return
	_ensure_idx(quads)
	RenderingServer.canvas_item_add_triangle_array(get_canvas_item(),
		_lidx.slice(0, quads * 6), _lpts, _lcols, _luvs,
		PackedInt32Array(), PackedFloat32Array(), GlyphSet.ramp_texture().get_rid())


# Indices depend only on a quad's ordinal and the buffer is built by concatenation, so
# one template grown to the high-water mark serves every frame.
func _ensure_idx(quads: int) -> void:
	var have := _lidx.size() / 6
	if quads <= have:
		return
	_lidx.resize(quads * 6)
	for q in range(have, quads):
		var b := q * 4
		var o := q * 6
		_lidx[o] = b
		_lidx[o + 1] = b + 1
		_lidx[o + 2] = b + 2
		_lidx[o + 3] = b
		_lidx[o + 4] = b + 2
		_lidx[o + 5] = b + 3


# ---------------------------------------------------------------------------------
# The worker job: sample the land, extract every level, draw the sheet
# ---------------------------------------------------------------------------------

## One re-print of the sheet, built off the main thread ([FrameForge]'s job form: a fresh
## object per kick, so nothing the worker reads can be mutated underneath it). It samples
## the field, extracts, smooths and simplifies every contour, and returns the finished
## triangle chunks - water first, then hatching, then the out-of-register plate, then the
## minor lines, the index lines, and the inked one last.
class SheetJob:
	extends RefCounted

	var fld: Field
	var nx := 128
	var ny := 72
	var half := Vector2.ZERO         # the sheet's half extent in pixels
	var fspan := Vector2.ONE         # the same, in field units
	var off := Vector2.ZERO          # the window's position on the continent
	## The warp offset in SHEET fractions this packet was built at. Read back off the job
	## itself through [method FrameForge.packet_source] - the scene cannot ask the job it last
	## KICKED, because the forge keeps one worker and replaces the pending snapshot when a
	## newer kick arrives, so most kicked jobs are dropped without ever running.
	var warp := Vector2.ZERO
	var lo := 0.1
	var step := 0.06
	var count := 14
	var index_every := 5
	var index_phase := 0
	var pick := 5
	var smooth_iters := 2
	var simp := 0.35
	var minor_col := Color.BLACK
	var index_col := Color.BLACK
	var pick_col := Color.BLACK
	var w_minor := 1.1
	var w_index := 2.4
	var w_pick := 3.2
	var plate := false
	var plate_off := Vector2.ZERO
	var plate_col := Color.TRANSPARENT
	var water := false
	var sea := 0.2
	var water_col := Color.TRANSPARENT
	var hatch_on := true
	var hatch_a := 0.8
	var hatch_sp := 8.0              # pixels
	var hatch_col := Color.TRANSPARENT
	var hatch_w := 1.0
	var band_lo := 0.0
	var band_hi := 0.0

	func run(_s: Dictionary) -> Array:
		var tb := TriBatch.new()
		if nx < 2 or ny < 2 or count <= 0:
			return tb.take_chunks()
		var h := PackedFloat32Array()
		h.resize(nx * ny)
		var ix := 1.0 / float(nx - 1)
		var iy := 1.0 / float(ny - 1)
		for y in ny:
			var qy := (float(y) * iy - 0.5) * fspan.y + off.y
			var row := y * nx
			for x in nx:
				h[row + x] = fld.at(Vector2((float(x) * ix - 0.5) * fspan.x + off.x, qy))

		var org := -half
		var cell := Vector2(2.0 * half.x * ix, 2.0 * half.y * iy)
		var cpx := maxf(0.5, (cell.x + cell.y) * 0.5)

		if water:
			Contour.fill_below(tb, h, nx, ny, sea, water_col, org, cell)
		if hatch_on and band_hi > band_lo:
			Contour.hatch(tb, h, nx, ny, band_lo, band_hi, hatch_a,
				maxf(1.0, hatch_sp) / cpx, 0.5, hatch_col, hatch_w, org, cell)

		var ex := Contour.new()
		var levels := ex.extract(h, nx, ny, lo, step, count)
		var eps := simp / cpx
		for k in count:
			levels[k] = Contour.simplify(Contour.smooth(levels[k], smooth_iters), eps)

		# The second colour plate, printed first and a hair out of register - the drop
		# shadow a two-pass press leaves when the paper has moved between plates.
		if plate:
			for k in count:
				if ((k + index_phase) % index_every) != 0:
					continue
				Contour.stroke(tb, levels[k], plate_col, w_index,
					org + plate_off, cell, false)
		for k in count:
			if ((k + index_phase) % index_every) == 0:
				continue
			Contour.stroke(tb, levels[k], minor_col, w_minor, org, cell, true)
		for k in count:
			if ((k + index_phase) % index_every) != 0 or k == pick:
				continue
			Contour.stroke(tb, levels[k], index_col, w_index, org, cell, true)
		if pick >= 0 and pick < count:
			Contour.stroke(tb, levels[pick], pick_col, w_pick, org, cell, true)
		return tb.take_chunks()
