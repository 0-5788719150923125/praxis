extends GhostScene

## Contour map - the terrain from directly above, as a printed survey sheet.
##
## Warm paper, almost white, with thin dark isolines threading it at a fixed interval:
## the first time in this catalogue that land is seen from straight overhead. Every fifth
## line is an index contour drawn heavier, the band above the inked index line is ruled
## with fine diagonal hatching, a graticule of ticks runs the margins with a scale bar in
## one corner, and small survey crosses sit on local summits, each with a leader line out
## to a two-glyph label in an invented script. A survey sweep crosses the sheet planting marks
## as it goes and clearing them when it comes round again.
##
## THE LAND EVOLVES, EVERYWHERE, AT DIFFERENT RATES. Ground rises and subsides under the ink and
## ridges wander - contours crowding up into a new hill here while a mile away the sheet is
## almost still, and a minute later the still part is the one reorganising - so the map reads as
## a survey being re-flown rather than as a picture. That is [method _step_tectonics], and it is
## deliberately NOT the window: moving the window moves every line in the frame at once, by the
## same vector, which is the crude thing this scene has been reported for twice. It is also not a
## few bumps on an otherwise frozen sheet, which is what the first repair made of it. The whole
## subject is written down there and gated in tests/contour_flow_check.gd.
##
## THREE ABSENCES AT ONCE, which is why it exists. Nothing else here is a PLAN VIEW -
## every other framing is eye-level horizon, three-quarter isometric or centred abstract.
## Nothing else is HIGH-KEY. And nothing else is a DIAGRAM: the nearest thing is
## `projection`, which has no axis, tick, label, legend or rule line anywhere in it. This
## is the same substance as `terrain` - a [Field] heightfield - rendered as line art on
## paper, which is a different visual language and not a recolour.
##
## A RE-PRINT MUST NOT BE VISIBLE, and almost everything below is in service of that. The
## sheet is rebuilt a couple of times a second and every difference between one build and
## the next lands as a step, all over the frame at once - which is what a viewer reads as
## "the whole scene jumps". So the three things that could differ do not: the sampling
## window is SNAPPED TO ITS OWN LATTICE, so consecutive extractions read the same points of
## the land and the sheet only has to be translated (see _kick); the hatching's pitch is
## fixed by the seed and anchored to the ground rather than to the window (see _kick and
## [method Contour.hatch]); and the tonal centre every colour here is drawn from is EASED,
## because the ink of the highlighted contour is baked into the packet (see _ease_audio).
## With those three, a re-print changes nothing but the strip of new land at the edge.
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
##   `f.flux` sets the INK DENSITY of the hatching between index lines - the same ruling,
##   pressed harder on a busy passage. Flux measures 0.01 to 0.05 in practice, so it is
##   scaled by 12 to cross its range at all rather than being used as if it reached 1. It
##   used to set the hatch SPACING, and that was the worst mark on the sheet: see _kick.
##   The SWEEP advances one sheet width per `f.beat_period` x a sampled [8,32] beats, and
##   a `f.beat` rising edge stamps the next survey cross - at the summit nearest the
##   sweep, so marks appear along the line as it passes.
##   `f.movement` over a sampled threshold eases the field offset to the next entry of a
##   pre-rolled ring, over 14 to 30 seconds. The offset is a position in an infinite noise
##   field, so easing to a new one is a slow flight across a continent. It is the GLOBAL
##   motion and it is rare on purpose - and because a spoken chapter barely moves that score
##   at all, there is now a floor under it too (`_warp_gap`, 35 to 75 s), so the window can
##   never stall for a whole song the way it was reported doing. The continuous evolution is
##   not this: it is the tectonics, which no audio feature touches.
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
## sample in a worker still draws like a pen - and it is also why the window has to be
## snapped: on cells that coarse, re-sampling the same land at a different phase moves every
## line by pixels. In an export the forge builds synchronously, so a render pays the
## extraction on the main thread - slower to render, identical output.
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
## WHAT THE SEED DECIDES: the two spectral bases the land evolves through - how many waves, how
## long, how fast, and the per-re-print budget that bounds the whole thing; the field kind
## (fbm / ridged / billow / cells), its octaves,
## frequency and whether it is domain-warped; how much of the continent the sheet covers;
## the number of contour levels and which of them are index lines; the minor and index pen
## widths; whether the sheet is cream paper with dark ink or a blueprint (a deep blue
## ground with white ink, roughly one sheet in four); the ink itself from a [Scheme] at a
## quarter of its nominal saturation, because a survey is drawn in one restrained colour;
## the hatch angle, and its spacing, which no audio feature may touch; the sea level and
## whether there is water at all; the graticule divisions and margin; how many survey marks
## the sheet may carry; the re-extraction cadence; whether a second colour plate prints
## slightly out of register; the sweep's period in beats; the section-change threshold and
## how long a warp takes; and the whole invented hand the labels are written in.
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
# The tonal centre, EASED - and eased as a VECTOR, because a hue is circular and an average
# of 0.99 and 0.01 is not 0.5. Everything colour on this sheet reads this rather than
# `chroma_hue()` itself: the ink of the highlighted contour is baked into a packet that is
# only rebuilt a couple of times a second, so a jittery tonal centre arrives as a step
# rather than as a drift. Slow enough that a re-print moves it by a fraction of a percent,
# fast enough that a chord change still re-prints the sheet in a new colour.
var _ch := Vector2.ZERO
var _ch_v := Vector2.ZERO
var _ch_raw := Vector2.ZERO
var _ch_seeded := false
## Seconds for the tonal centre and the flux to close most of the way onto a new value.
## Set from the measurement in `tests/contour_flow_check.gd`: against a tonal centre
## sweeping continuously through the wrap, this is what keeps the inked contour's colour
## step across one re-print under 0.05, and 2.5 s did not (0.076 at seed 11).
const EASE_SECS := 4.0
# Flux, eased for the same reason: it is the hatching's ink density and the hatching is in
# the packet. Flux itself is a per-frame transient detector and steps hard.
var _flux_e := 0.0

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
var _hatch_sp := 8.0
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
var _warp_gap := 45.0
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
var _warp_idle := 0.0               # seconds since the window last moved (see _step)
## THE LAND'S OWN EVOLUTION - a small spectral basis of travelling waves, evaluated in FIELD
## coordinates off one clock. See [method _step_tectonics] and [SheetJob].
##
## Two fields, both `sum_i a_i sin(k_i . q + w_i t + phi_i)`. UPLIFT adds elevation, so ground
## rises and subsides and the contours reorganise - new closed rings appear, saddles open and
## shut. DRIFT displaces the point the land is READ at, so ridges meander and, where the
## displacement converges, contours crowd - which is the only thing here that looks like two
## plates pushing against each other.
##
## Flat arrays because every one of these crosses onto a worker on each re-print, and in the
## units the PICTURE is judged in: a wavelength is a fraction of the sheet's width, an uplift
## amplitude is a contour interval, a drift amplitude is a fraction of the sheet.
var _up_kx := PackedFloat32Array()   # wavevector, radians per sheet width
var _up_ky := PackedFloat32Array()
var _up_w := PackedFloat32Array()    # radians per second, either sign
var _up_ph := PackedFloat32Array()
var _up_a := PackedFloat32Array()    # weight, normalised so sum(a * |w|) == 1
var _dr_kx := PackedFloat32Array()
var _dr_ky := PackedFloat32Array()
var _dr_w := PackedFloat32Array()
var _dr_ph := PackedFloat32Array()
var _dr_a := PackedFloat32Array()
var _dr_dx := PackedFloat32Array()   # which way this wave pushes the ground
var _dr_dy := PackedFloat32Array()
## THE WHOLE BUDGET, and the only two numbers that decide whether this reads as evolution or
## as a jump: how far the land may move IN ONE RE-PRINT, worst case. See [method _step_tectonics].
var _up_print := 0.04                # contour intervals
var _dr_print := 0.0006              # sheet widths
## The tectonic clock. The fields are a pure FUNCTION of it - nothing is integrated - so two
## builds at the same instant are the same sheet however the scene got there.
var _tect_t := 0.0
## The land's mean relief across ONE GRID CELL, measured once alongside the ladder and frozen.
## The uplift is limited against it so a basin does not swim while a scarp barely moves - see
## [method SheetJob.run].
var _grad_cell := 0.01
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
	# ONE spacing, chosen here and never touched again - see the class doc. This used to be
	# `lerp(hi, lo, flux)`, re-evaluated on every re-print, and it was the single worst mark
	# on the sheet: the ruling is anchored, so changing its pitch moves every line by its own
	# multiple of the change, and the whole hatched band slid sideways a couple of times a
	# second. Density is the seed's now, and the music is in the ink instead.
	_hatch_sp = rng.randf_range(4.0, 12.0)

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
	# The floor under the section-change trigger: however quiet the track, the window travels
	# at least this often. Long, because a section change is meant to be what usually does it.
	_warp_gap = rng.randf_range(35.0, 75.0)
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

	# THE LAND'S OWN EVOLUTION, seeded - see the class doc and [method _step_tectonics]. Two
	# spectral bases: uplift, whose wavelengths run from a third of the sheet to half again its
	# width, and drift, deliberately SHORTER so it shears the sheet rather than translating it.
	#
	# The period scales WITH the wavelength, which is what makes a rate budget divide evenly:
	# every wave then contributes about the same amount of movement per second, so no single
	# component can dominate the step between two prints. It is also the way ground behaves -
	# a gully re-cuts in a season, a range takes an age.
	var n_up := rng.randi_range(5, 9)
	var n_dr := rng.randi_range(3, 5)
	_roll_waves(rng, n_up, 0.34, 1.55, rng.randf_range(55.0, 130.0),
		_up_kx, _up_ky, _up_w, _up_ph, _up_a)
	_roll_waves(rng, n_dr, 0.26, 0.80, rng.randf_range(40.0, 95.0),
		_dr_kx, _dr_ky, _dr_w, _dr_ph, _dr_a)
	for i in n_dr:
		# Which way this wave pushes. Along its own wavevector the field is compressive - it
		# crowds and spreads contours, which is a range building - and across it the field is
		# pure shear, which slides one part of the country past another without changing how
		# steep it is. A sheet wants both, so the angle between the two is sampled.
		var rel := rng.randf_range(0.0, TAU)
		var kn := Vector2(_dr_kx[i], _dr_ky[i]).normalized()
		_dr_dx.append(kn.x * cos(rel) - kn.y * sin(rel))
		_dr_dy.append(kn.x * sin(rel) + kn.y * cos(rel))
	# The budget, per re-print rather than per second, because a print is the unit the viewer
	# actually sees: whatever the cadence, this is the largest step any line can take between
	# two frames of the sheet. An interval's worth of movement is a contour arriving where its
	# neighbour used to be, so a twentieth of one is a line creeping by a pixel.
	_up_print = rng.randf_range(0.026, 0.033)
	_dr_print = rng.randf_range(0.00036, 0.00046)

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
		"hatch_px": _hatch_sp,
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
		"uplift_waves": n_up,
		"drift_waves": n_dr,
		"uplift_per_print": _up_print,
		"drift_per_print": _dr_print,
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
	# Read once a frame - the Director substeps update(), and the signature behind this is a
	# real cost - and eased on the fixed clock in _ease_audio.
	_ch_raw = chroma_hue()
	if not _ch_seeded:
		_ch_seeded = true
		_ch = _ch_raw
		_ch_v = Vector2(cos(_ch.x * TAU), sin(_ch.x * TAU)) * _ch.y
		_flux_e = f.flux
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
	_ease_audio(dt)
	# The highlight migrates on the FIXED clock like everything else that integrates: it
	# is called from here rather than from update(), which the Director substeps up to
	# fifteen times in one frame and pre-warms twelve deep.
	_pick_index(dt)
	if _warp_t < 1.0:
		_warp_t = minf(1.0, _warp_t + dt / maxf(0.5, _warp_secs))
		_warp_idle = 0.0
	else:
		# THE WINDOW MAY NOT STALL FOREVER. Its only trigger was a `f.movement` rising edge over
		# a sampled threshold, and a narration take barely moves that score at all - so on a
		# spoken chapter the window never once moved and the sheet was reported as "zoomed, but
		# no longer moves at all". A section change is still what NORMALLY moves it; this is the
		# floor under that, on the same slow ease, so the map always eventually travels.
		_warp_idle += dt
		if _warp_idle >= _warp_gap:
			_rewarp()
	_step_tectonics(dt)
	var period := maxf(2.0, clampf(_f.beat_period, 0.25, 1.6) * _sweep_beats)
	_sweep_prev = _sweep_p
	_sweep_p = fposmod(_sweep_p + dt / period, 1.0)
	_age(dt)
	_ext_t += dt
	if _ext_t >= _cadence:
		_ext_t = 0.0
		_kick()


# Close the eased audio onto its raw reading. Both quantities end up INSIDE a packet that
# is rebuilt on the extraction cadence, so anything they drive arrives as a step of
# whatever they moved in the last half second - and a step is exactly what this sheet must
# not have. Eased here, a re-print moves them by a couple of percent.
func _ease_audio(dt: float) -> void:
	var k := 1.0 - exp(-dt / maxf(0.05, EASE_SECS))
	var raw := _ch_raw
	# On the wheel, not on the number line: the hue is an angle and its strength is the
	# vector's length, so a tonal centre with no answer shrinks toward the middle instead of
	# racing round the rim.
	_ch_v = _ch_v.lerp(Vector2(cos(raw.x * TAU), sin(raw.x * TAU)) * raw.y, k)
	if _ch_v.length() > 0.0001:
		_ch = Vector2(fposmod(_ch_v.angle() / TAU, 1.0), clampf(_ch_v.length(), 0.0, 1.0))
	else:
		_ch = Vector2(_ch.x, 0.0)
	_flux_e = lerpf(_flux_e, _f.flux, k)


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
	var fspan := _fspan()
	# The sheet occupies -0.5..0.5; the warp ring reaches 0.55 either way, so the table
	# has to cover a good deal more than the sheet or a wandering map runs out of summits.
	#
	# READ WITHOUT THE TECTONICS, deliberately. This solves the elevation ladder and the summit
	# table, and both are supposed to be FIXED: an interval is a claim about the land, and a
	# survey mark that drifted because a hill nearby breathed would be a mark that means
	# nothing. So the evolution moves the contours and leaves the ladder, the datum and the
	# marks alone - which is also what makes it safe, since neither the interval nor any label
	# can be pushed around by it.
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
	_measure_relief(fspan)

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


## Measure the land's mean relief ACROSS ONE GRID CELL, once, and freeze it. This is the
## reference the uplift is limited against in [method SheetJob.run], and the reason it exists
## is that the same elevation change does not move a contour the same distance everywhere: a
## line moves by `dv / |grad v|`, so on a basin where the contours already stand a long way
## apart a change the eye can barely justify sends a line across a quarter of the sheet, while
## on a scarp the same change is invisible. Unlimited, that is the sheet's flat parts swimming
## while its mountains sit still - a different flavour of exactly the wrong picture.
##
## Measured at the SHEET'S OWN PITCH, not on the coarse survey grid above: relief is a
## fractal, so a grid three times coarser reads a gradient several times smaller and the
## reference would be meaningless. Sixteen small patches scattered over the region the window
## can wander across, so it is the country's relief rather than one hill's.
##
## Frozen, and it has to be. Anything the limit is computed from that changed with the window
## would perturb every sample on the sheet by a hair on every window move - a global step,
## which is the one thing this scene may not have.
func _measure_relief(fspan: Vector2) -> void:
	var cs := fspan.x / float(maxi(2, _res) - 1)      # one grid cell, in field units
	var pn := 10
	var h := PackedFloat32Array()
	h.resize(pn * pn)
	var acc := 0.0
	var cnt := 0
	for py in 4:
		for px in 4:
			var o := Vector2(float(px) - 1.5, float(py) - 1.5) * 0.62 * fspan
			for y in pn:
				for x in pn:
					h[y * pn + x] = _fld.at(o + Vector2(float(x) - 4.5, float(y) - 4.5) * cs)
			for y in range(1, pn - 1):
				for x in range(1, pn - 1):
					var i := y * pn + x
					var gx := (h[i + 1] - h[i - 1]) * 0.5
					var gy := (h[i + pn] - h[i - pn]) * 0.5
					acc += sqrt(gx * gx + gy * gy)
					cnt += 1
	_grad_cell = maxf(1e-6, acc / float(maxi(1, cnt)))


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
	_warp_idle = 0.0


## THE LAND'S OWN EVOLUTION - the answer to "the map should slowly evolve, in slow morphing
## over time rather than in large jumps, and not by shifting every line uniformly".
##
## THREE THINGS HAVE BEEN WRONG HERE, and the third is the one this replaces.
##
##   First the land never changed at all and the only motion was the WINDOW, gated on a
##   `f.movement` edge a spoken chapter never produces: the sheet either sat dead still or
##   lurched as a whole. A window move is global BY CONSTRUCTION - every line in the frame
##   goes at once, by the same vector - which is exactly the uniform shift being asked against.
##
##   Then that was answered with a handful of compact BUMPS on the land, and it answered the
##   wrong question. It read as reported: a couple of circles of the sheet breathing while the
##   rest was frozen, and because one bump takes minutes to grow, rest, subside and only THEN
##   move, the same two circles were the only thing that ever moved for the whole scene. The
##   request was never "restrict the evolution to a region". It was "do not move the whole
##   sheet by the same amount at the same instant".
##
##   What is here now lets the WHOLE sheet evolve and takes the uniformity out instead, which
##   is what erosion and tectonics actually look like: everywhere is always changing, at rates
##   that differ from place to place, and the places that are changing fastest keep moving.
##
## THE MECHANISM is a small spectral basis - five to nine travelling waves for uplift, three to
## five for drift - summed in FIELD coordinates:
##
##   d(q, t) = sum_i a_i sin(k_i . q + w_i t + phi_i)
##
## Nowhere is masked out and nowhere is pinned: every point of the land is inside every wave, so
## every point evolves. What differs is HOW MUCH and WHEN, because at any instant the sum has
## crests (ground rising fast), troughs (subsiding) and nodes (holding still) scattered across
## the sheet - and since the waves travel at different speeds in different directions, those
## nodes are never in the same place twice. A region that is quiet now is the region visibly
## reorganising two minutes from now. That is the property a bump field could not have.
##
## Four things make it safe:
##
##   A RATE BUDGET, NOT AN AMPLITUDE BUDGET, and this is the whole difference between evolution
##   and a jump. The sheet is re-printed on a [0.3, 1.0] s cadence, so ANY change lands as a
##   step at that rate; what the viewer reads as jumpy is not how far the land has moved but how
##   far it moved BETWEEN TWO PRINTS. So the seed sets `_up_print` - the elevation, in contour
##   intervals, the land may travel in one re-print, worst case over the whole basis - and the
##   per-second rate is derived from it by dividing by the cadence. A twentieth of an interval
##   is a line creeping by about a pixel. Left to accumulate over a scene that same rate carries
##   the land a couple of contours, which is a hill being born.
##
##   ANCHORED TO THE LAND. The waves are functions of the FIELD coordinate, not of the sheet, so
##   the window's lattice snap still holds exactly: after a one-cell window move, grid point
##   (x, y) reads the field point (x-1, y) read - same land, same wave phase, same value. The
##   sheet is still only translated between prints, which is the property tests/contour_flow_check.gd
##   exists for. A perturbation defined in sheet coordinates would slide under that snap and
##   re-wobble every line on the sheet, which is the original fault all over again.
##
##   PURE IN TIME. Nothing is integrated - both fields are a function of `_tect_t` - so a
##   pre-warm, an Echo fast-forward and an export reach the same sheet as a live session that
##   arrived the slow way, and two builds at one instant are identical.
##
##   NO AUDIO ANYWHERE NEAR IT. The land is the one thing on this sheet the music does not
##   touch (see the class doc): a map whose ground breathes on the beat is a graphic, not a
##   survey. The window warp remains the audio-conditioned motion, and it stays rare.
##
## DRIFT is the second field and it is not decoration. Uplift alone can only inflate and deflate
## the land in place; a drift field displaces the point the land is read at, so ridges wander
## and - where the displacement converges - contours crowd together, which is the picture of
## ground being pushed. It is bounded directly in pixels rather than in elevation, since a line
## displaced by d moves by exactly d whatever the local slope is.
func _step_tectonics(dt: float) -> void:
	_tect_t += dt


## Roll one spectral basis into the arrays given. Wavelengths are sheet WIDTHS, periods scale
## with wavelength (so every wave contributes about equally to the rate, and the longest waves
## are the slowest), amplitudes follow a red spectrum - longer waves are taller, the way any
## natural relief is - and the whole set is normalised so `sum(a * |w|) == 1`. That last step
## is what makes the budget in [method _step_tectonics] mean something: multiply these weights
## by a rate and the basis moves at that rate, whatever was rolled.
func _roll_waves(rng: RandomNumberGenerator, n: int, lam_lo: float, lam_hi: float,
		period: float, kx: PackedFloat32Array, ky: PackedFloat32Array,
		w: PackedFloat32Array, ph: PackedFloat32Array, a: PackedFloat32Array) -> void:
	var norm := 0.0
	for i in n:
		var lam := rng.randf_range(lam_lo, lam_hi)
		var ang := rng.randf_range(0.0, TAU)
		var k := TAU / maxf(0.05, lam)
		kx.append(cos(ang) * k)
		ky.append(sin(ang) * k)
		# The period jitter keeps the basis from beating: with periods in an exact ratio the
		# whole sum returns to its starting shape, and the sheet would visibly loop.
		var t_i := maxf(6.0, period * lam * rng.randf_range(0.72, 1.38))
		var om := TAU / t_i
		w.append(om if rng.randf() < 0.5 else -om)
		ph.append(rng.randf_range(0.0, TAU))
		a.append(lam)
		norm += lam * om
	if norm <= 1e-9:
		return
	for i in n:
		a[a.size() - n + i] = a[a.size() - n + i] / norm


## The sheet's extent in FIELD units. One definition, used by the extraction, the summit table
## and the tectonics - they have to agree about what a sheet fraction means.
func _fspan() -> Vector2:
	var u := maxf(1.0, unit())
	return Vector2(size.x, size.y) * (SHEET * _span / u)


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
	var job := _make_job()
	if job != null:
		_forge.kick(job.run, {}, self, job)


## The next re-print, as a job object, without submitting it. Split out of [method _kick]
## so a gate can build two sheets from two states and compare them - which is the only way
## to assert the property this scene lives or dies by, that consecutive re-prints are the
## same drawing.
func _make_job() -> SheetJob:
	if not _ladder or size.x < 4.0:
		return null
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
	job.fspan = _fspan()
	# SNAP THE WINDOW TO ITS OWN SAMPLING LATTICE, and this is the fix that makes a re-print
	# invisible. The land is static - only the window moves - so two extractions taken at the
	# same lattice phase read the SAME points of the field and produce, cell for cell, the
	# same contours; the sheet then only has to be translated, which _draw does exactly.
	# Sampled off the lattice they read a different set of points, and marching squares on a
	# grid whose cells are a dozen pixels across answers a half-cell shift with a wobble of
	# several pixels EVERYWHERE AT ONCE. That is what "the entire scene shifts every half
	# second" was: not a jump in the map, a re-sampling of it.
	#
	# The sub-cell remainder is not lost - it goes into `warp`, which _draw reads back off the
	# packet and turns into the translation, so the map still glides continuously at the
	# warp's own speed. Snapping costs at most half a cell of lag on the land arriving at the
	# leading edge, under a margin that covers it.
	var cellf := Vector2(job.fspan.x / float(maxi(2, nx) - 1), job.fspan.y / float(maxi(2, ny) - 1))
	var raw := _warp_now() * job.fspan
	job.off = Vector2(round(raw.x / cellf.x) * cellf.x, round(raw.y / cellf.y) * cellf.y)
	job.warp = Vector2(job.off.x / maxf(0.0001, job.fspan.x), job.off.y / maxf(0.0001, job.fspan.y))
	# THE LAND'S EVOLUTION as of this print (see _step_tectonics), resolved out of sheet units
	# into the field units the worker samples in. Two conversions and they are the whole
	# contract with the budget:
	#
	#   The wavevectors are radians per sheet WIDTH, so dividing by the sheet's width in field
	#   units anchors every wave to the ground - which is what keeps the lattice snap exact.
	#
	#   The weights are normalised so `sum(a |w|) == 1`, so multiplying by a rate makes the
	#   basis move at that rate. The rate is the per-PRINT budget divided by the cadence,
	#   because the cadence is how often that step is actually taken: a sheet re-printed three
	#   times a second may evolve three times as fast per second and still step no further.
	var fw := maxf(1e-6, job.fspan.x)
	var up_rate := _up_print * _step_v / maxf(0.05, _cadence)
	for i in _up_a.size():
		job.up_kx.append(_up_kx[i] / fw)
		job.up_ky.append(_up_ky[i] / fw)
		job.up_p.append(_up_ph[i] + _up_w[i] * _tect_t)
		job.up_a.append(_up_a[i] * up_rate)
	var dr_rate := _dr_print * fw / maxf(0.05, _cadence)
	for i in _dr_a.size():
		job.dr_kx.append(_dr_kx[i] / fw)
		job.dr_ky.append(_dr_ky[i] / fw)
		job.dr_p.append(_dr_ph[i] + _dr_w[i] * _tect_t)
		job.dr_ax.append(_dr_a[i] * dr_rate * _dr_dx[i])
		job.dr_ay.append(_dr_a[i] * dr_rate * _dr_dy[i])
	job.relief = _grad_cell
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
	job.hatch_sp = _hatch_sp
	# The music is in the INK, not in the pitch: a busy passage rules the same lines darker.
	# Flux measures 0.01 to 0.05, hence the gain, and it is the eased flux - the alpha is
	# baked into a packet that is rebuilt twice a second, and a raw transient detector in
	# there is a flicker.
	job.hatch_col = Color(ink.r, ink.g, ink.b,
		lerpf(0.22, 0.46, clampf(_flux_e * 12.0, 0.0, 1.0)))
	job.hatch_w = maxf(0.7, _w_minor * 0.8)
	job.band_lo = _lo + float(_pick) * _step_v
	job.band_hi = job.band_lo + float(_index_every) * _step_v
	return job


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
	## THE LAND'S EVOLUTION, flattened into FIELD units with the clock already folded into the
	## phase, so the worker evaluates a static field and never needs the time (see the scene's
	## [method _step_tectonics]). UPLIFT adds elevation; DRIFT displaces the point the land is
	## read at, one 2-vector amplitude per wave. Flat arrays rather than the scene's own, because
	## this crosses onto a worker thread while the sim keeps advancing its copies.
	var up_kx := PackedFloat32Array()
	var up_ky := PackedFloat32Array()
	var up_p := PackedFloat32Array()
	var up_a := PackedFloat32Array()
	var dr_kx := PackedFloat32Array()
	var dr_ky := PackedFloat32Array()
	var dr_p := PackedFloat32Array()
	var dr_ax := PackedFloat32Array()
	var dr_ay := PackedFloat32Array()
	## The land's mean relief across one grid cell (see the scene's _measure_relief) - what the
	## uplift's contour displacement is limited against.
	var relief := 0.01
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

	## How far the uplift may carry a contour from where the base field puts it, in GRID CELLS,
	## before it starts to saturate - a hard bound on the land's wander, so nothing on flat
	## ground can run away however little slope there is to divide by.
	##
	## MEASURED, IT DOES NOT BIND at the budget this scene ships with: moving it between 1.0 and
	## 1.6 changes tests/contour_flow_check.gd's displacement figures by nothing at four
	## decimals. It is kept at the tighter of the two, as the backstop it is - what actually
	## shapes the picture is the rate budget and the slope weighting below.
	const MAX_SHIFT := 1.0

	## How far below the land's mean relief the limit stops tightening. Without a floor the
	## limit divides by a gradient that goes to zero at every summit and every basin floor, so
	## precisely the closed rings whose growth is the most legible thing on the sheet would be
	## the one place the land is not allowed to move.
	const RELIEF_FLOOR := 0.35

	## The most the slope weighting may AMPLIFY the uplift where the ground is steep.
	const SLOPE_CAP := 1.8

	func run(_s: Dictionary) -> Array:
		var tb := TriBatch.new()
		if nx < 2 or ny < 2 or count <= 0:
			return tb.take_chunks()
		return _draw_sheet(sample(), tb)


	## THE LAND THIS PRINT IS OF: the field, read at the grid points, with the evolution folded
	## in. Split out of [method run] because it is the whole of what one print differs from the
	## last BY - the drawing is a function of it - so a gate can take two of them and measure the
	## difference in the elevation the reader is looking at, rather than in the vertices that
	## happened to survive the simplify pass. Those are not the same question and the second one
	## is unanswerable: a smoothed contour is re-traced along its whole length when it moves
	## anywhere, and a point dropped as collinear in one print and kept in the next lands a whole
	## grid cell from where it was, on a line that did not move at all.
	func sample() -> PackedFloat32Array:
		var h := PackedFloat32Array()
		if nx < 2 or ny < 2:
			return h
		h.resize(nx * ny)
		var ix := 1.0 / float(nx - 1)
		var iy := 1.0 / float(ny - 1)
		# The sample positions, once. Both spectral bases are SEPARABLE - sin(A + B) is
		# sinA cosB + cosA sinB - so a wave costs one row table, one column table and two
		# multiplies a sample instead of a sin() a sample. At a dozen waves over nine thousand
		# samples that is the difference between a frame budget and several.
		var qx := PackedFloat32Array()
		qx.resize(nx)
		for x in nx:
			qx[x] = (float(x) * ix - 0.5) * fspan.x + off.x
		var qy := PackedFloat32Array()
		qy.resize(ny)
		for y in ny:
			qy[y] = (float(y) * iy - 0.5) * fspan.y + off.y

		# --- DRIFT: the land is read a little away from where the grid points sit, and by a
		# different amount in different places, so ridges wander and contours crowd where the
		# displacement converges. Applied to the sample POSITION, which bounds what it does in
		# pixels directly - a point displaced by d moves its contour by exactly d, whatever the
		# local slope is - and that is why this half needs no limiter.
		var nd := dr_ax.size()
		var dsx := _table(dr_kx, qx, PackedFloat32Array(), true)
		var dcx := _table(dr_kx, qx, PackedFloat32Array(), false)
		var dsy := _table(dr_ky, qy, dr_p, true)
		var dcy := _table(dr_ky, qy, dr_p, false)

		# --- UPLIFT: ground rising and subsiding, which is the half that can change the
		# TOPOLOGY - open a saddle, close a new ring around a hill that was not there.
		var nu := up_a.size()
		var usx := _table(up_kx, qx, PackedFloat32Array(), true)
		var ucx := _table(up_kx, qx, PackedFloat32Array(), false)
		var usy := _table(up_ky, qy, up_p, true)
		var ucy := _table(up_ky, qy, up_p, false)

		for y in ny:
			var row := y * nx
			for x in nx:
				var px := qx[x]
				var py := qy[y]
				for k in nd:
					var sn := dsx[k * nx + x] * dcy[k * ny + y] + dcx[k * nx + x] * dsy[k * ny + y]
					px += dr_ax[k] * sn
					py += dr_ay[k] * sn
				h[row + x] = fld.at(Vector2(px, py))

		if nu > 0:
			# The limit, and it is the reason a basin does not swim. A contour moves by
			# `dv / |grad v|`, so the same uplift that nudges a scarp throws a line across a flat
			# basin; the displacement is measured in cells against the LOCAL gradient, floored at
			# a fraction of the land's own mean relief, and softened rather than clipped
			# (`u / sqrt(1 + (u/lim)^2)` is the identity for small u and saturates smoothly), so
			# there is no seam where the limit begins to bite.
			#
			# The gradient is read off the base grid, which is a function of the FIELD point, so
			# after a one-cell window move every interior sample limits to exactly the value it
			# limited to before. The lattice snap survives this pass.
			# Solved into its own buffer and added afterwards, because the limit reads the
			# base grid's gradient: written in place, every sample would be differencing land
			# its own neighbour had already lifted.
			var du := PackedFloat32Array()
			du.resize(nx * ny)
			var floor_g := RELIEF_FLOOR * relief
			for y in ny:
				var row := y * nx
				var yn := maxi(0, y - 1) * nx
				var yp := mini(ny - 1, y + 1) * nx
				for x in nx:
					var u := 0.0
					for k in nu:
						u += up_a[k] * (usx[k * nx + x] * ucy[k * ny + y]
							+ ucx[k * nx + x] * usy[k * ny + y])
					var gx := (h[row + mini(nx - 1, x + 1)] - h[row + maxi(0, x - 1)]) * 0.5
					var gy := (h[yp + x] - h[yn + x]) * 0.5
					var g := maxf(sqrt(gx * gx + gy * gy), floor_g)
					# TIED TO THE LOCAL SLOPE, at the square root of it. The uplift budget is an
					# elevation and what the reader sees is a DISPLACEMENT, and the two are
					# related by the slope - so an unweighted uplift walks a line across a basin
					# while barely nudging the same line on a scarp, and the sheet's flat parts
					# swim. Weighting by the slope exactly would make the displacement uniform
					# and also make the ELEVATION change vanish where the land is flat, which is
					# where a new hill is most interesting; the square root halves the spread and
					# leaves both halves something to do. Capped, because on a cliff the exact
					# weight would be several contours of uplift in a few cells of ground.
					u *= minf(sqrt(g / relief), SLOPE_CAP)
					var lim := MAX_SHIFT * g
					var r := u / lim
					du[row + x] = u / sqrt(1.0 + r * r)
			for i in nx * ny:
				h[i] += du[i]
		return h


	func _draw_sheet(h: PackedFloat32Array, tb: TriBatch) -> Array:
		var ix := 1.0 / float(nx - 1)
		var iy := 1.0 / float(ny - 1)
		var org := -half
		var cell := Vector2(2.0 * half.x * ix, 2.0 * half.y * iy)
		var cpx := maxf(0.5, (cell.x + cell.y) * 0.5)

		if water:
			Contour.fill_below(tb, h, nx, ny, sea, water_col, org, cell)
		if hatch_on and band_hi > band_lo:
			# Where this grid's origin sits in the land, in cells - the ruling is defined on
			# the ground, so a window that has moved re-rules the same lines over the same
			# country instead of re-phasing the whole band. Whole numbers, because the window
			# is snapped to the lattice.
			var land := Vector2(off.x / maxf(1e-9, fspan.x * ix),
				off.y / maxf(1e-9, fspan.y * iy))
			Contour.hatch(tb, h, nx, ny, band_lo, band_hi, hatch_a,
				maxf(1.0, hatch_sp) / cpx, HATCH_MARCH, hatch_col, hatch_w, org, cell, land)

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


	# One axis of a separable spectral basis: sin (or cos) of `k[i] * q[j] + bias[i]` for every
	# wave i and every sample j, flattened i-major. The bias carries the phase AND the clock, so
	# it belongs to exactly one of the two axes - passing an empty array is how the other one
	# says it has none.
	static func _table(k: PackedFloat32Array, q: PackedFloat32Array,
			bias: PackedFloat32Array, want_sin: bool) -> PackedFloat32Array:
		var n := k.size()
		var m := q.size()
		var out := PackedFloat32Array()
		out.resize(n * m)
		for i in n:
			var b: float = bias[i] if i < bias.size() else 0.0
			var base := i * m
			for j in m:
				var a := k[i] * q[j] + b
				out[base + j] = sin(a) if want_sin else cos(a)
		return out
