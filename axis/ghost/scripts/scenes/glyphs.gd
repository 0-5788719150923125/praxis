extends GhostScene

## Glyphs - an invented script, written live, one stroke at a time.
##
## A page of writing in a language that does not exist. The alphabet comes from
## [GlyphSet]: a stroke grammar over a small lattice, rolled per session into twenty
## to forty characters in one consistent hand - its own slant, weight, nib angle,
## terminal style and taste in curves. This scene is the other half - the PEN and the
## PAGE. Characters are not blitted: a nib crosses the paper, ink arrives behind it
## stroke by stroke, words group, lines fill, the page scrolls, and when the music
## goes quiet the pen simply stops where it is, mid-stroke, resting on the paper.
##
## It is the catalogue's whole missing axis. Not one of the other scenes draws a
## letter, numeral, rune or ideogram, and it is the first whose content has a READING
## DIRECTION rather than being a physical process - a page has a beginning, an order,
## and an accumulating history, which is a different kind of picture from fog or fire.
##
## THE MUSIC DECIDES THE TEXT. Which character comes next is the argmax of the twelve
## chroma channels of [method Spectrum.harmonic_signature], crossed with the coarse
## low/mid/high tilt that follows them, through a seeded permutation - so a chord
## recurring in the same register writes the same character every time, and the page
## becomes a transcription of the progression rather than a random spill of shapes.
## Characters are COMMITTED on the beat, subdivided by a sampled [1,4] beats each, and
## the reveal is paced across [member AudioFeatures.beat_period] so the pen's speed is
## the track's tempo and not the frame rate. Word breaks land where
## [member AudioFeatures.movement] crosses a sampled threshold, which puts them at
## section changes; a hand with punctuation marks (see [GlyphSet]'s reserve) closes the
## word with one. [member AudioFeatures.flux] presses the nib - up to a sampled 35% of
## extra weight, and no more, because ink weight is the only thing here that may track
## amplitude at all. Everything else the sound does, it does to colour: tonality
## (`chroma_hue().y`) raises the ink's saturation and warms the paper.
##
## SILENCE IS A STATE, NOT AN IDLE. Below [member Director.silence_floor] (0.03) the
## reveal simply does not advance. The pen does not lift, does not idle-doodle, does
## not fade - it stops with the nib down, which is what a hand does when the dictation
## stops, and is far more alive than any animation would be.
##
## WHAT THE SEED DECIDES beyond the hand itself: the layout (a full page, a narrow
## column, a centred banner that recentres itself as each line grows, or a spiral
## written outward and drawn slowly back toward its own centre), the margins, the line
## height, beats per character, how strongly a section change breaks a word, whether
## the paper is ruled and carries a margin line, how much of the page is already
## written when we arrive, and the ground: high-key paper with dark ink, or bright ink
## on a dark bed. Both grounds exist because a page is the one subject in this
## catalogue that genuinely reads either way.
##
## COST. The writing is ONE draw call however full the page. Every character's
## geometry is baked once in em
## space, so a frame is one native `Transform2D * PackedVector2Array` per character
## plus three `append_array`s, and the stroke quads carry their own antialiasing in a
## 1-D alpha ramp texture instead of [TriBatch]'s three-quads-per-line feather.
## [constant MAX_GLYPHS] = 300 is the page budget: at roughly fifteen quads a
## character that is about 4.5k quads (9k triangles) in one flush, and the finest hand
## on the widest frame would otherwise ask for twice that - so rather than truncating
## the page, the whole hand is scaled up until it fits.
##
## HONEST DETERMINISM CAVEAT. The alphabet, the hand, the layout and the pre-written
## opening are all seed-derived and reproduce exactly. The written TEXT is not: the
## live analyzer and the offline export bake do not produce identical feature streams,
## so the sequence of characters in a render will differ from the preview, and the
## layout also follows the frame's aspect. Acceptable for an unreadable script, but it
## is a real difference and it is stated rather than hidden. No rng is ever drawn on
## an audio-conditioned event, so nothing else can drift.

## Characters the page may hold at once - the flush budget (see COST above).
const MAX_GLYPHS := 300

## Layouts, weighted. A page is the honest default; the others are the same writing
## in a different measure, and the spiral is the one that stops reading as paper.
const LAYOUTS := ["page", "page", "page", "column", "banner", "spiral"]

var _gs: GlyphSet
var _sch: Scheme
var _sim := SimClock.new(60.0)
var _prng := RandomNumberGenerator.new()
var _f: AudioFeatures = AudioFeatures.new()
var _sig := PackedFloat32Array()
var _ch := Vector2.ZERO

# The page: finished characters, plus the one under the nib.
var _placed: Array = []
var _cur: Dictionary = {}
var _reveal := 0.0
var _hold := 0.0                  # the pen's rest before the next character (seconds)
var _written := 0                 # characters the cursor has stepped past, ever
var _words := 0
var _pending_word := false
var _cur_mark := false
var _joined_now := false
var _mv_prev := 0.0
# The music's current 36-wide harmonic key (see GlyphSet.key), and how many characters have
# been written under it. A held key walks the alphabet on rather than repeating one shape.
var _key := -1
var _run := 0

# The layout cursor.
var _cx := 0.0
var _cy := 0.0
var _line := 0
var _shifts: Dictionary = {}      # finished line -> its frozen centring shift
var _shift_cur := 0.0
var _scroll := 0.0
var _scroll_to := 0.0
# THE CAMERA. The page used to be framed whole - a sheet of small sparse characters with
# its margins showing - which reads as a screenshot of a document rather than as writing
# happening. The view now sits IN the text: glyphs are large enough to overflow the frame,
# and the camera follows the pen instead of the paper, with a slow independent downward
# creep so the frame is never quite still even while a single character is being drawn.
var _cam := Vector2.ZERO          # eased camera centre, in the same unit space as _cx/_cy
var _cam_drift := 0.0             # seconds, for the bounded breath (see _track_pen)
var _chasing := false             # the nib is out of shot: close fast, not loosely (see _track_pen)
var _zoom_in := 2.6               # how far the framing is pushed past whole-page
var _drift_rate := 0.006

# The spiral's own cursor: arc length along the coil, and how far the whole coil has
# been drawn back in toward the centre.
var _s := 0.0
var _sshift := 0.0
var _sshift_to := 0.0
var _coil := 0.02                 # r = coil * theta
var _s_min := 0.0
var _s_max := 1.0

# Page metrics, in unit fractions.
var _half := Vector2(0.888, 0.5)
var _left := 0.0
var _right := 0.0
var _top := 0.0
var _bottom := 0.0
var _adv := 0.05
var _em := 0.08
var _line_step := 0.12
var _fit := 1.0
var _open := false                # has the page been laid out and pre-written yet

# Sampled look / behaviour.
var _layout := "page"
var _centred := false
var _margin := 0.08
var _col_frac := 0.48
var _line_h := 1.6
var _bpg := 2
var _char_beats := 0.5    # fraction of a beat one NOMINAL character takes to write
var _mean_ink := 0.0      # the hand's mean glyph ink length, the divisor in _len_scale
var _rest_char := 0.12    # the pen's rest after a character / a word / a line, as a
var _rest_word := 0.55    # fraction of one nominal character's span
var _rest_line := 1.1
var _break_thr := 0.35
var _press_lvl := 0.0             # smoothed nib pressure, 0..1
var _ease := 0.6                  # how much the stroke reveal is eased
var _prefill := 0.4
var _paper := true
var _ruled := false
var _margin_rule := false
var _ink_hue := 0.0
var _ink_wet := 0.0
var _ink_sat := 0.5
var _ink_val := 0.2
var _ink_alpha := 0.95
var _dry := 2.0
var _paper_hue := 0.1
var _paper_sat := 0.06
var _paper_val := 0.92
var _rule_col := Color(0, 0, 0, 0.12)
var _nib_col := Color(0, 0, 0, 0.8)

# Per-frame geometry scratch, kept as members: packed arrays are VALUE types in
# GDScript, so a helper that appends to a parameter would append to a copy.
var _pts := PackedVector2Array()
var _cols := PackedColorArray()
var _uvs := PackedVector2Array()
var _idx := PackedInt32Array()

# _pose's outputs, likewise: a Dictionary per character per frame is 220 allocations.
var _pxf := Transform2D.IDENTITY
var _pfade := 1.0

var _tip := Vector2.ZERO
var _tip_rot := 0.0
var _tip_ok := false


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "plane"                     # a page is square-on; PLANE_BAG keeps it there
	_prng.seed = rng.randi()
	_gs = GlyphSet.new(rng)
	# The hand's mean ink length, so a character's duration can be its OWN length against it
	# (see _len_scale) without changing the page's average pace.
	_mean_ink = 0.0
	for g in _gs.glyphs:
		_mean_ink += (g as GlyphSet.Glyph).total
	_mean_ink /= maxf(1.0, float(_gs.count()))
	_sch = Scheme.pick(rng)
	_layout = String(LAYOUTS[rng.randi() % LAYOUTS.size()])
	_centred = _layout == "banner"
	_margin = rng.randf_range(0.04, 0.14)
	_col_frac = rng.randf_range(0.38, 0.60)
	_line_h = rng.randf_range(1.30, 2.10)
	_bpg = rng.randi_range(1, 4)
	# 0.25 to 0.8 of a beat per character: at a 120 bpm estimate that is 2.5 to 8 a
	# second, which spans a quick scribble and a deliberate hand.
	_char_beats = rng.randf_range(0.25, 0.80)
	# THE HAND RESTS, and not evenly. A pen lifts for a moment between characters, waits
	# longer between words, and longest at the turn of a line - and that uneven rhythm is
	# most of what separates writing from a teleprinter. Fractions of one nominal character's
	# span, so they ride the tempo like everything else here.
	_rest_char = rng.randf_range(0.05, 0.22)
	_rest_word = rng.randf_range(0.35, 0.95)
	_rest_line = rng.randf_range(0.70, 1.80)
	# How far into the text the camera sits. At 1.0 the whole page is in frame, which is
	# the old look; past 2 the writing runs off both edges and the eye has to follow it.
	#
	# A SPIRAL IS THE EXCEPTION and is framed whole. The zoom exists so the eye has lines to
	# read along; a coil has none, its nib orbits rather than travels (see _pen_at), and at a
	# page's zoom the frame lands inside the coil's own empty hub - the hole at the centre is
	# `_line_step` across, which at 2.1x to 3.4x is the size of the visible height. So the
	# composition is shown as a composition, with just enough push to crop its outermost turn.
	_zoom_in = rng.randf_range(1.02, 1.30) if _layout == "spiral" else rng.randf_range(2.1, 3.4)
	# Slow enough to be felt rather than watched - a few percent of a line height a second.
	_drift_rate = rng.randf_range(0.004, 0.011)
	# Movement is a 0..1 section-change score that spends most of a track low, so a
	# threshold up near 0.5 gives long words broken only at real changes and one down
	# at 0.2 gives short, choppy ones. Both read as a language; sample the whole range.
	_break_thr = rng.randf_range(0.18, 0.52)
	_ease = rng.randf_range(0.20, 1.0)
	_prefill = rng.randf_range(0.15, 0.70)
	_dry = rng.randf_range(1.2, 4.0)
	_paper = rng.randf() < 0.5
	_ruled = _paper and rng.randf() < 0.45
	_margin_rule = _ruled and rng.randf() < 0.5
	if _paper:
		# High-key: the ground IS the subject, so it is flat and bright (paint_ground,
		# not a bed) and the ink is the dark thing on it.
		_paper_hue = _sch.hue
		_paper_sat = rng.randf_range(0.03, 0.12)
		_paper_val = rng.randf_range(0.86, 0.96)
		_ink_hue = _sch.opposed(_sch.hue, 0.10)
		_ink_wet = fposmod(_sch.accent + rng.randf_range(-0.03, 0.03), 1.0)
		_ink_sat = clampf(_sch.sat * rng.randf_range(0.5, 0.9), 0.05, 1.0)
		_ink_val = rng.randf_range(0.10, 0.28)
		_ink_alpha = rng.randf_range(0.88, 1.0)
		_rule_col = Color.from_hsv(fposmod(_sch.accent, 1.0), 0.40, 0.55,
			rng.randf_range(0.10, 0.22))
		_nib_col = Color.from_hsv(fposmod(_ink_hue, 1.0), _ink_sat * 0.7, _ink_val * 0.8, 0.85)
	else:
		# The dark ground is composed, not painted: the shared bed BEHIND the writing and a
		# thin drift of dust IN FRONT of it, so the ink sits inside an atmosphere rather
		# than on a flat field. (`z` defaults to "front" - the bed must ask for "back".)
		add_layer("bed", rng, {"z": "back", "hue": _sch.hue, "sat": _sch.sat * 0.7,
			"val": rng.randf_range(0.10, 0.20), "pools": rng.randi_range(2, 4)})
		add_layer("dust", rng, {"z": "front", "count": rng.randi_range(30, 70),
			"hue": _sch.accent, "sat": _sch.sat * 0.4})
		_ink_hue = _sch.vary(rng, 0.5)
		_ink_wet = _sch.accent
		_ink_sat = clampf(_sch.sat * rng.randf_range(0.4, 0.8), 0.03, 1.0)
		_ink_val = rng.randf_range(0.80, 1.0)
		_ink_alpha = rng.randf_range(0.80, 0.95)
		_rule_col = Color.from_hsv(fposmod(_sch.accent, 1.0), 0.35, 0.9, 0.07)
		_nib_col = Color.from_hsv(fposmod(_ink_hue, 1.0), _ink_sat * 0.5, 1.0, 0.75)
	var d := _gs.spec()
	d["mood"] = _sch.name
	d["layout"] = _layout
	d["ground"] = "paper" if _paper else "ink_on_dark"
	d["margin"] = _margin
	d["line_height"] = _line_h
	d["beats_per_glyph"] = _bpg
	d["char_beats"] = _char_beats
	d["rest_word"] = _rest_word
	d["break_threshold"] = _break_thr
	d["reveal_ease"] = _ease
	d["prefill"] = _prefill
	d["dry_seconds"] = _dry
	d["ruled"] = _ruled
	return d


func _ready() -> void:
	super()
	# A pressed nib samples the stroke ramp slightly past its ends (see
	# GlyphSet.uv_pressure). Clamping is what makes that read as "past the end of the
	# feather" instead of wrapping round into the opposite edge of the texture.
	texture_repeat = CanvasItem.TEXTURE_REPEAT_DISABLED


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.02, 0.03)           # a page barely moves; it is being read
	update_layers(f, delta)
	queue_redraw()
	if size.x < 4.0:
		return
	_metrics()
	if not _open:
		_open = true
		_write_opening()
		_shift_cur = _shift_target()     # arrive centred; the ease is for what happens after
		# The page arrives already part written (see _write_opening), so it should arrive
		# already FRAMED. Easing in from the centre would have the opening seconds of every
		# session drift across whatever the prefill happened to leave under the middle.
		_cam = _clamp_cam(_pen_at())
	# AFTER _metrics and the opening, never before: the tracker reads `_left` / `_right` /
	# `_bottom` / `_line_step` for its bound and the cursor for its target, and on the first
	# frame none of those exist yet - it would aim at, and clamp against, the declared defaults.
	_track_pen(delta)
	_ch = chroma_hue()
	_sig = Spectrum.harmonic_signature()
	# A character is COMMITTED on the beat, every _bpg beats. Event, not integration -
	# so it belongs here and not in the sim step (the Director may substep an identical
	# frame several times, and one edge stays one edge).
	# THE HAND DOES NOT WAIT FOR THE BAR LINE. A character used to be committed only on
	# every _bpg'th beat, while the reveal finished in 0.82 of that interval - so the pen
	# wrote, then sat on a finished character doing nothing for the remainder, every single
	# time. With _bpg up to 4 that is a pause of over a second between characters, which is
	# what "long pauses between each character being drawn" was.
	#
	# The tempo still sets the SPEED (see _span), which is the part worth keeping; it no
	# longer gates the handover. A finished character commits immediately and the next one
	# starts in the same frame, so the writing is continuous - which is what writing is.
	if f.movement > _break_thr and _mv_prev <= _break_thr:
		_pending_word = true
	_mv_prev = f.movement
	for _i in _sim.ticks(delta):
		_step(_sim.dt)


# The pen. Everything here integrates time, so it runs on the fixed clock: the
# Director substeps, pre-warms twelve frames deep and lets Echo fast-forward, and a
# reveal that advanced once per update() call would write a whole line in one frame.
func _step(dt: float) -> void:
	# Nib pressure. Flux runs about 0.01 to 0.05 in practice, so it takes a gain near 20
	# to reach a full press at all, and an asymmetric follower to keep the hand from
	# strobing between hairline and heavy on consecutive frames.
	_press_lvl = Nonlinear.flare(_press_lvl, clampf(_f.flux * 20.0, 0.0, 1.0), dt, 6.0, 1.8)
	if _f.energy < Director.silence_floor:
		return                          # silence: the nib rests where it is, mid-stroke
	# Spend this substep across rests and characters in order, carrying the OVERSHOOT rather
	# than discarding it, so a fast hand loses no fraction of a character's progress at a
	# handover - which would put a small stutter back in place of the large one the beat gate
	# used to cause. A loop, because one substep can span a rest and more than one character.
	var left := dt
	var guard := 0
	while left > 0.0 and guard < 64:
		guard += 1
		if _hold > 0.0:
			var used := minf(_hold, left)
			_hold -= used
			left -= used
			continue
		var span := _span()
		var need := (1.0 - _reveal) * span
		if left < need:
			_reveal += left / span
			break
		left -= need
		_reveal = 1.0
		_commit()                       # _start resets the reveal, _advance sets the next rest
	_scroll = lerpf(_scroll, _scroll_to, 1.0 - exp(-9.0 * dt))
	_sshift = lerpf(_sshift, _sshift_to, 1.0 - exp(-6.0 * dt))
	_ease_shift(dt)


# THE CENTRED LINE'S SHIFT, and why it is integrated here rather than computed at draw time.
#
# A banner recentres its live line, so every glyph on it is drawn at `p.x + _shift_cur`. That
# shift used to be recomputed inside _ink() as `-(_cx + _adv + _left) * 0.5` - straight off the
# page CURSOR, which advances by a whole character at a time in _advance(). So the entire row
# stepped left by half an advance the instant each character was committed, and the row a reader
# is trying to read was the one thing on the page that would not sit still. Reported as "the
# entire row of glyphs kept shifting in hard, discrete jumps... you would expect the row of
# glyphs to remain where they were written".
#
# Two changes, and both are needed:
#
#   THE TARGET MOVES WITH THE REVEAL. The line's visible extent is the committed cursor plus the
#   fraction of the current character the nib has actually inked, so the target is continuous
#   ACROSS a commit: `_cx` gains that character's width at the same moment `_reveal` drops from
#   1 to 0, and the two cancel. Without this the ease would only smear the step over a fifth of a
#   second instead of removing it.
#
#   AND IT IS EASED, on this fixed clock, because the residue is not zero: a word gap, a narrow
#   punctuation mark and a carriage return all move the target by a real amount in one substep.
#   Those become a glide.
#
# _pen_at reads the SAME eased value, so the camera and the row can never disagree about where
# the line is - and _newline freezes it as the line's final shift, so a finished line stops
# exactly where it was last drawn rather than at a recomputed value half an advance away.
func _ease_shift(dt: float) -> void:
	if not _centred:
		_shift_cur = 0.0
		return
	_shift_cur = lerpf(_shift_cur, _shift_target(), 1.0 - exp(-14.0 * dt))


func _shift_target() -> float:
	if not _centred:
		return 0.0
	var w := _adv * (0.62 if _cur_mark else 1.0)
	return -(_cx + w * clampf(_reveal, 0.0, 1.0) + _left) * 0.5


# How long the character under the nib has to be written.
#
# A pen moves at a roughly constant NIB SPEED, so a character's duration is its own ink
# length, not a fixed slice of the clock. Every character used to take exactly
# `_char_beats * beat_period` however many strokes it had - and with `_char_beats` fixed per
# session and `beat_period` near-constant over speech, that is ONE cadence for the whole
# page, which is what "all of that writing was at a more or less constant speed" was. The
# alphabet runs 2 to 7 strokes a character (GlyphSet.strokes_lo/hi), so ink lengths already
# differ by more than a factor of two; dividing by the hand's own mean leaves the average
# pace exactly where it was and only redistributes it.
func _span() -> float:
	return maxf(0.05, _base_span() * _len_scale(int(_cur.get("g", -1))))


# The tempo's share of one NOMINAL character, before that character's length and the hand's
# rests are applied. Clamped so a missing or absurd tempo estimate cannot stall or sprint the
# hand. A FRACTION of a beat, not a whole number of them: `_bpg` was an integer count because
# a character was committed ON a beat, and with that gate gone the integer floor made the
# slowest hands write at roughly one character a second. A real hand runs two to six.
func _base_span() -> float:
	return _char_beats * clampf(_f.beat_period, 0.25, 1.6)


# This character's ink length against the hand's mean. Bounded, so the shortest character is
# still visibly drawn rather than flicked out, and the longest does not stall the page.
func _len_scale(gi: int) -> float:
	if gi < 0 or _mean_ink <= 0.0 or _gs == null or gi >= _gs.glyphs.size():
		return 1.0
	var g: GlyphSet.Glyph = _gs.glyphs[gi]
	return clampf(g.total / _mean_ink, 0.55, 1.75)


# ---------------------------------------------------------------------------------
# The page: metrics, the cursor, and committing a character
# ---------------------------------------------------------------------------------

func _metrics() -> void:
	var u := maxf(1.0, unit())
	_half = Vector2(size.x, size.y) / (2.0 * u)
	if not _open:
		_adv = _gs.advance
		_em = _gs.em
		_line_step = _em * _line_h
		_bounds()
		var cap := _capacity()
		if cap > MAX_GLYPHS:
			# Scale the whole hand rather than truncating the page: capacity falls with
			# the square of the scale, so this lands exactly on the budget.
			_fit = sqrt(float(cap) / float(MAX_GLYPHS))
			_adv = _gs.advance * _fit
			_em = _gs.em * _fit
			_line_step = _em * _line_h
	_bounds()


func _bounds() -> void:
	var mx := _half.x * (1.0 - 2.0 * _margin)
	var my := _half.y * (1.0 - 2.0 * _margin)
	_top = -my
	_bottom = my
	var measure := 2.0 * mx
	if _layout == "column":
		measure *= _col_frac
	elif _layout == "banner":
		measure *= 0.86
	_left = -measure * 0.5
	_right = measure * 0.5
	# The coil: one turn per line, so the spiral's rows sit exactly as far apart as a
	# page's lines would. Arc length s of an Archimedean spiral is ~coil*theta^2/2,
	# which is where both bounds come from.
	_coil = maxf(0.000001, _line_step / TAU)
	_s_min = _line_step * _line_step / (2.0 * _coil)
	var rmax := _half.length() * 1.08
	_s_max = rmax * rmax / (2.0 * _coil)


func _capacity() -> int:
	if _layout == "spiral":
		return int(maxf(1.0, (_s_max - _s_min) / maxf(0.001, _adv)))
	var per_line := int(maxf(1.0, (_right - _left) / maxf(0.001, _adv)))
	var lines := int(maxf(1.0, (_bottom - _top) / maxf(0.001, _line_step)) + 1.0)
	return per_line * lines


# The page is already part written when we arrive. A blank sheet filling at one
# character a beat would take a whole scene to become a picture, and the subject here
# is a PAGE, not a countdown. Seeded from the build rng, so it reproduces exactly.
func _write_opening() -> void:
	_cx = _left
	_cy = _top + _em
	_s = _s_min
	var n := int(_prefill * float(_capacity()))
	_start(_prng.randi() % maxi(1, _gs.count()))
	for _i in n:
		_cur["born"] = -60.0            # long dry by the time the first frame is drawn
		_placed.append(_cur)
		_cur = {}
		if _prng.randf() < 0.16:
			_pending_word = true
		# The pre-write is instantaneous, so the centring is SNAPPED along with it - the ease is
		# for a hand that is writing. It has to be kept up to date here even so, because
		# _newline() freezes the live shift as each line's final one, and a run of opening lines
		# frozen at a stale value would left-align the whole page above the pen.
		_reveal = 1.0
		_shift_cur = _shift_target()
		_advance(_prng.randi() % maxi(1, _gs.count()))
		_shift_cur = _shift_target()
	_reveal = 0.0
	_hold = 0.0


func _commit() -> void:
	if not _cur.is_empty():
		_cur["born"] = _sim.elapsed()
		_placed.append(_cur)
		_cur = {}
	_advance(-1)
	while _placed.size() > MAX_GLYPHS * 2:
		_placed.remove_at(0)


# Step the cursor past the character just finished and put the next one under the nib.
# [param gi] < 0 means "ask the music"; the opening pass hands its own index instead,
# because a signature is not available before the first audio frame.
func _advance(gi: int) -> void:
	var w := _adv * (0.62 if _cur_mark else 1.0)
	var gap := 0.0
	var was_mark := _cur_mark
	var line0 := _line
	_written += 1
	var next_mark := false
	if _pending_word and not was_mark and _gs.reserve > 0:
		next_mark = true                # the mark closes the word; the gap follows it
	elif _pending_word:
		gap = _adv * _gs.word_gap
		_words += 1
		_pending_word = false
	_joined_now = _gs.joined and gap <= 0.0 and not was_mark and not next_mark
	if _layout == "spiral":
		_s += w + gap
		if _s - _sshift_to > _s_max:
			_sshift_to = _s - _s_max
			_drop_spiral()
	else:
		_cx += w + gap
		if _cx + _adv > _right:
			_newline()
	_cur_mark = next_mark
	var idx := gi
	if next_mark:
		idx = _gs.mark(_words)
	elif idx < 0:
		var k := _gs.key(_sig)
		if k >= 0 and _ch.y > 0.0005:
			# A NEW key opens a new word at that chord's own character; a HELD key walks the
			# alphabet on instead of repeating. The signature is a 2.5 s EMA, so its argmax
			# sits still for tens of characters - see GlyphSet.pick for the measurement.
			if k != _key:
				_key = k
				_run = 0
			else:
				_run += 1
			idx = _gs.pick(_sig, _run)
		else:
			# No tonal content to read (a session booted with no audio at all): step the
			# alphabet on the counters instead. Still a function of the feature stream,
			# so it stays reproducible - an rng here would diverge in the export.
			#
			# `_written`, not the old `_beats`. That counter belonged to the beat gate and was
			# left behind when the gate was removed, never incremented again, so the expression
			# was frozen on `_words`. It is NOT what the repetition report was: this branch is
			# very nearly unreachable, since `_sig.size()` is always 16 (HarmonicSignature.DIM)
			# and reaching it needs `_ch.y <= 0.0005` while the energy is above the silence
			# floor, which is the 12 chroma channels cancelling to one part in three thousand.
			# A latent bug found while reading, fixed while here - the repetition itself came
			# from the tonal branch above.
			idx = (_written * 7 + _words * 5) % maxi(1, _gs.count())
	_start(idx)
	# The rest that FOLLOWS the character just finished, before the next is written. Set after
	# _start, which clears it along with the reveal.
	var rest := _rest_char
	if gap > 0.0 or was_mark:
		rest = _rest_word
	if _line != line0:
		rest = _rest_line
	_hold = rest * _base_span()


func _newline() -> void:
	if _centred:
		# Freeze this line's centring the moment it is finished, at the value it was last DRAWN
		# with rather than a recomputed one - a recomputed freeze differs from the live shift by
		# the last character's half advance, which is one more jump at every carriage return.
		# The live line goes on recentring itself as it grows (see _ease_shift).
		_shifts[_line] = _shift_cur
	_line += 1
	_cx = _left
	if _centred:
		# A CARRIAGE RETURN IS A DISCONTINUITY, and easing it is wrong. The line just finished
		# keeps the shift it was drawn with (frozen above), and the new line has nothing on it
		# yet - so snapping its centring moves no ink at all, while easing it drags the first
		# glyph of the new line across the page from wherever the old line ended. That is a
		# 9-advance glide, measured, and far more visible than the per-character step this ease
		# exists to remove.
		_shift_cur = _shift_target()
	_cy += _line_step
	_joined_now = false
	if _cy > _bottom:
		_scroll_to += _line_step
		_drop_lines()


func _drop_lines() -> void:
	var keep: Array = []
	var cut := _top - 1.8 * _line_step
	var lowest := _line
	for e in _placed:
		var p: Vector2 = e["p"]
		if p.y - _scroll_to > cut:
			keep.append(e)
			lowest = mini(lowest, int(e["line"]))
	_placed = keep
	# The centring table would otherwise grow one entry per line for the life of the
	# session; nothing above the page can ever be asked for again.
	for k in _shifts.keys():
		if int(k) < lowest:
			_shifts.erase(k)


func _drop_spiral() -> void:
	var keep: Array = []
	for e in _placed:
		if float(e["s"]) - _sshift_to > _s_min * 0.9:
			keep.append(e)
	_placed = keep


func _start(gi: int) -> void:
	var e := {
		"g": clampi(gi, 0, maxi(0, _gs.count() - 1)),
		"join": _joined_now,
		"line": _line,
		"born": 0.0,
	}
	if _layout == "spiral":
		e["s"] = _s
	else:
		e["p"] = Vector2(_cx, _cy)
	_cur = e
	_reveal = 0.0
	_hold = 0.0


# ---------------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------------

func _draw() -> void:
	begin_draw()
	if _paper:
		# Tonality warms the sheet instead of dimming it - a high-key ground that reacts
		# by going dark just looks broken.
		paint_ground(_paper_hue, _paper_sat, _paper_val, _ch.y * 0.6, _ch.x)
	else:
		draw_layers("back")
	if size.x < 4.0 or _gs == null or _gs.count() == 0:
		return
	var u := unit()
	_frame(u)
	_rules(u)
	_ink(u)
	# NO NIB. A wedge marking where the ink stops was drawn here, and it read as a
	# cursor sitting on the page rather than as a pen: it has no hand, no shaft and no
	# motion blur, so at any size it is a floating arrowhead. The ink arriving IS the
	# indication that something is writing, and it is a better one.
	if not _paper:
		# Back to the SCREEN's transform first. The dust is atmosphere in the room, not ink on
		# the page: left under _frame's matrix it was magnified by _zoom_in (2.1x to 3.4x) and
		# pinned to the paper, so it slid with the writing instead of hanging in front of it.
		draw_set_transform_matrix(view.matrix(size))
		draw_layers("front")


## Push the view into the text: zoom past whole-page and centre on the tracked pen.
##
## Composed onto the scene's view matrix rather than into [member view] itself, so it
## replaces its own contribution every frame instead of accumulating one.
func _frame(u: float) -> void:
	var z := _zoom_in
	draw_set_transform_matrix(view.matrix(size)
		* Transform2D(0.0, Vector2(z, z), 0.0, -_cam * u * z))


## Follow the pen, loosely.
##
## Eased hard (a ~1.4 s time constant) rather than locked to the nib: a camera that pinned
## the pen would make the PAGE move under a stationary point, which is the opposite of
## reading and is nauseating at this zoom. Loose tracking lets the character being written
## drift across the frame and the camera catch up between words, which is how an eye
## actually follows a hand.
##
## The downward creep is independent of the writing and never stops, so a held character
## or a silent passage still has motion in the frame - without it, "the pen rests where it
## is" would freeze the entire scene.
func _track_pen(delta: float) -> void:
	# The downward travel comes from the WRITING - _cy steps down a line at a time as the
	# hand fills the page, and the camera follows it. An accumulator was wrong here and
	# would have been a slow catastrophe: it grows without bound, so after a few minutes
	# the camera would be somewhere below the page with the text long out of frame.
	#
	# What the accumulator was reaching for - motion in the frame even while the pen is
	# holding a character - is a bounded breath instead.
	_cam_drift += delta
	var breath := sin(_cam_drift * _drift_rate * TAU * 12.0) * _line_step * 0.18
	# THE SACCADE. Loose tracking is the whole feel of this scene, but a line break is not a
	# drift - it is a CARRIAGE RETURN, and an eye does not glide back across a page, it jumps.
	# Measured with the loose constant alone (tests/glyph_frame_check.gd): after every newline the
	# frame spent 1.47 s sailing over blank paper while the hand wrote at the start of the new
	# line, and the nib was out of shot for 32% of the hold. So the rate is loose while the nib
	# is IN frame and fast while it is not - ONE rule, which also covers every other jump (the
	# opening frames, a page scroll landing) without having to name them.
	#
	# The two thresholds are a commit band, the same shape as contour_map's _pick_index: it
	# starts chasing at the frame edge but does not stop until the nib is comfortably back
	# inside, so a nib hovering ON the boundary cannot chatter between the two rates.
	var target := _pen_at() + Vector2(0.0, breath)
	var h := _half / maxf(0.001, _zoom_in)
	var off := target - _cam
	if absf(off.x) > h.x or absf(off.y) > h.y:
		_chasing = true
	elif absf(off.x) < h.x * 0.7 and absf(off.y) < h.y * 0.7:
		_chasing = false
	_cam = _clamp_cam(_cam.lerp(target, 1.0 - exp((-9.0 if _chasing else -0.7) * delta)))
	# NOTE the camera is NOT written into `view`. Doing that compounds: `view.zoom` and
	# `view.offset` are only re-set each frame by the assigned SHOT, and a scene running
	# without one - the catalogue smoke gate builds every scene shotless - multiplies its
	# own zoom by _zoom_in every frame. Measured before it was caught: zoom reached 3.9e13
	# after one second and inf within half a minute. The framing is applied as its own
	# transform at draw time instead, which is idempotent by construction.


## Where the nib actually IS on screen, in the unit-fraction space [member _cam] lives in.
##
## NOT `_cx` / `_cy`. Those are the page cursor, and only three of the six layouts write at it:
##
##   SPIRAL advances `_s` along its coil and never touches `_cx` / `_cy` at all, so they keep
##   whatever _write_opening left them - the top-left corner of the page - for the entire hold.
##   The camera therefore sat on that corner watching blank paper while the writing went round
##   the middle of the frame.
##
##   BANNER recentres its live line, so a glyph is drawn at `p.x + _shift_cur` (see _pose and
##   _ink). At the start of a line that shift is nearly half the measure - about 0.5 in unit
##   fractions against a visible half-width of 0.26 to 0.42 at this zoom - so aiming at the raw
##   cursor put the pen entirely outside the frame and only walked back into it as the line
##   filled.
##
## `_cy` also needs its scroll taken off: it is an absolute coordinate on an endless page that
## only ever increases, and when it passes `_bottom` the page scrolls (`_scroll_to +=
## _line_step`) instead of resetting it, so every glyph is drawn at `p.y - _scroll`.
func _pen_at() -> Vector2:
	if _layout == "spiral":
		# A COIL IS NOT TRACKED, it is framed. The nib runs along the spiral, so its angle
		# advances as `sqrt(s)` - measured on a mid-size coil, about 0.14 rad per character at
		# eight characters a second, which is 0.44 unit-fractions a second of tangential travel.
		# The tracker's own 1.4 s constant means a camera aimed at it lags by more than half a
		# unit, i.e. further than the whole visible width, so it can only ever chase and never
		# arrive: measured at 1.3% of frames with the nib in shot, which is WORSE than the old
		# rule's parked corner. The coil re-centres itself instead (`_sshift` pulls it inward
		# and _drop_spiral retires the middle), so the composition is already where it needs to
		# be and the frame simply holds still on it - which is why `_zoom_in` is sampled low for
		# this layout in build_params, so the whole coil is in shot rather than its empty hub.
		return Vector2.ZERO
	# The SAME eased shift the glyphs are drawn with (see _ease_shift). Recomputing the raw
	# target here would have the camera stepping while the row glides.
	return Vector2(_cx + _adv * 0.5 + _shift_cur, _cy - _scroll - _line_step * 0.25)


## Keep the frame ON the writing.
##
## The tracking above is eased over about 1.4 s, which is what makes it read as an eye following
## a hand rather than as a rig bolted to it - but a lag that long crosses a lot of blank paper
## whenever the pen jumps, and a mis-aimed target (the two layouts above) parks there for good.
## Bounding the camera is what turns "the writing should stay in frame" from a hope into a
## property: the visible rectangle is held inside the region the writing occupies, so the frame
## can neither sit on empty ground nor travel far enough to bring the sheet's own edge into view.
##
## The bound is exactly tight, not conservative - at the limit the pen sits ON the frame edge
## rather than inside it - because pulling it in any further would stop the camera before the
## end of a line and the hand would appear to write off the side of its own frame.
func _clamp_cam(c: Vector2) -> Vector2:
	var h := _half / maxf(0.001, _zoom_in)          # visible half-extent, in unit fractions
	if _layout == "spiral":
		# A coil is written outward from its centre, so the written DISC is the bound, and it
		# grows with `_s` - which is also why the opening frame stays centred: at `_s_min` the
		# whole coil is smaller than the view and there is nowhere to go.
		var s := maxf(_s - _sshift, _s_min)
		var r := _coil * sqrt(2.0 * s / maxf(0.000001, _coil))
		return c.limit_length(maxf(0.0, r - minf(h.x, h.y)))
	# A page narrower or shorter than the view (a column, mostly) has no pan to do on that axis,
	# so it centres instead of clamping to an inverted range.
	var x1 := _right - h.x
	var y1 := _bottom - h.y
	return Vector2(
		clampf(c.x, -x1, x1) if x1 > 0.0 else 0.0,
		clampf(c.y, -y1, y1) if y1 > 0.0 else 0.0)


# Faint ruling under the writing. Cheap (a dozen antialiased lines) and it is what
# turns a bright ground into paper rather than a wall.
func _rules(u: float) -> void:
	if not _ruled or _layout == "spiral" or _line_step < 0.005:
		return
	var base := _top + _em
	var k := int(ceil((_top + _scroll - base) / _line_step))
	for i in 40:
		var y := base + float(k + i) * _line_step - _scroll
		if y > _bottom + _line_step * 0.5:
			break
		if y < _top:
			continue
		draw_line(Vector2(_left * u, y * u), Vector2(_right * u, y * u), _rule_col, 1.0, true)
	if _margin_rule:
		var x := (_left - _adv * 0.6) * u
		draw_line(Vector2(x, _top * u), Vector2(x, _bottom * u), _rule_col, 1.2, true)


# The whole page in one triangle array. Per character: one native transform of its
# baked vertices, three append_arrays, one colour.
func _ink(u: float) -> void:
	_pts.clear()
	_cols.clear()
	_uvs.clear()
	_tip_ok = false
	var now := _sim.elapsed()
	var prev_exit := Vector2.ZERO
	var prev_ok := false
	var hair_w := _gs.weight * u / GlyphSet.REF_UNIT * 0.34
	# One squeeze of the ramp coordinate presses the whole hand (see GlyphSet.uv_pressure).
	var pressed := _gs.pressure > 0.005
	var uvt := _gs.uv_pressure(_press_lvl)
	for e in _placed:
		if not _pose(e, u):
			prev_ok = false
			continue
		var g: GlyphSet.Glyph = _gs.glyphs[int(e["g"])]
		var col := _ink_color(now - float(e["born"]), _pfade)
		if prev_ok and bool(e["join"]):
			_hairline(prev_exit, _pxf * g.entry, hair_w, col)
		var wv: PackedVector2Array = _pxf * g.verts
		var wuv: PackedVector2Array = (uvt * g.uvs) if pressed else g.uvs
		_emit(wv, wuv, col)
		prev_exit = _pxf * g.exit
		prev_ok = true
	if not _cur.is_empty() and _pose(_cur, u):
		var gc: GlyphSet.Glyph = _gs.glyphs[int(_cur["g"])]
		# The reveal is eased, so ink leaves the nib the way a hand moves - into a stroke
		# and out of it - rather than at a constant machine rate.
		var p := lerpf(_reveal, Nonlinear.apply("smoothstep", _reveal), _ease)
		var tr := _gs.trace(int(_cur["g"]), p)
		var v: PackedVector2Array = tr["verts"]
		var uvc: PackedVector2Array = tr["uvs"]
		var col_c := _ink_color(0.0, _pfade)
		if v.size() >= 4:
			if prev_ok and bool(_cur["join"]):
				_hairline(prev_exit, _pxf * gc.entry, hair_w, col_c)
			var wv2: PackedVector2Array = _pxf * v
			var wuv2: PackedVector2Array = (uvt * uvc) if pressed else uvc
			_emit(wv2, wuv2, col_c)
		var tip_local: Vector2 = tr["tip"]
		_tip = _pxf * tip_local
		_tip_rot = _pxf.get_rotation()
		_tip_ok = true
	_flush()


func _emit(verts: PackedVector2Array, uvs: PackedVector2Array, col: Color) -> void:
	if verts.is_empty():
		return
	_pts.append_array(verts)
	_uvs.append_array(uvs)
	var c := PackedColorArray()
	c.resize(verts.size())
	c.fill(col)
	_cols.append_array(c)


# The cursive connector between two characters of the same word: one quad, the same
# ramp texture, a third of the pen's weight.
func _hairline(a: Vector2, b: Vector2, w: float, col: Color) -> void:
	var d := b - a
	if d.length_squared() < 0.01:
		return
	var n := Vector2(-d.y, d.x).normalized() * (w / GlyphSet.CORE * 0.5)
	var u0 := GlyphSet.uv_lo()
	var u1 := GlyphSet.uv_hi()
	_pts.append(a - n)
	_pts.append(b - n)
	_pts.append(b + n)
	_pts.append(a + n)
	_uvs.append(Vector2(u0, 0.5))
	_uvs.append(Vector2(u0, 0.5))
	_uvs.append(Vector2(u1, 0.5))
	_uvs.append(Vector2(u1, 0.5))
	var faint := Color(col.r, col.g, col.b, col.a * 0.55)
	for _i in 4:
		_cols.append(faint)


func _flush() -> void:
	var quads := _pts.size() / 4
	if quads <= 0:
		return
	_ensure_idx(quads)
	RenderingServer.canvas_item_add_triangle_array(get_canvas_item(),
		_idx.slice(0, quads * 6), _pts, _cols, _uvs,
		PackedInt32Array(), PackedFloat32Array(), GlyphSet.ramp_texture().get_rid())


# Indices depend only on a quad's ORDINAL, and the vertex buffer is built by
# concatenation, so one template grown to the high-water mark serves every frame -
# the alternative is six array appends per quad, thousands of times a frame.
func _ensure_idx(quads: int) -> void:
	var have := _idx.size() / 6
	if quads <= have:
		return
	_idx.resize(quads * 6)
	for q in range(have, quads):
		var b := q * 4
		var o := q * 6
		_idx[o] = b
		_idx[o + 1] = b + 1
		_idx[o + 2] = b + 2
		_idx[o + 3] = b
		_idx[o + 4] = b + 2
		_idx[o + 5] = b + 3


# Where one character sits this frame, into _pxf / _pfade. Returns false for a
# character that has scrolled off the page.
func _pose(e: Dictionary, u: float) -> bool:
	# Deliberately audio-free: the sound presses the nib and moves the colour, it never
	# scales the writing. A page that breathed with the amplitude would read as a gif.
	var sc := _em * u
	if _layout == "spiral":
		var s: float = float(e["s"]) - _sshift
		if s < _s_min * 0.9:
			return false
		var theta := sqrt(2.0 * s / _coil)
		var r := _coil * theta
		_pfade = clampf((r - _line_step) / maxf(0.001, _line_step * 0.8), 0.0, 1.0)
		_pxf = Transform2D(theta + PI * 0.5, Vector2(sc, sc), 0.0,
			Vector2(cos(theta), sin(theta)) * r * u)
		return true
	var p: Vector2 = e["p"]
	var y := p.y - _scroll
	if y < _top - 2.0 * _line_step:
		return false
	# A centred layout recentres the LIVE line as it grows and freezes the shift once the
	# line is finished, so writing pushes the line left instead of everything jumping.
	var sh := _shift_cur
	if _centred:
		var ln := int(e["line"])
		if _shifts.has(ln):
			sh = float(_shifts[ln])
	_pfade = clampf((y - _top + 1.6 * _line_step) / maxf(0.001, 1.6 * _line_step), 0.0, 1.0)
	_pxf = Transform2D(0.0, Vector2(sc, sc), 0.0, Vector2(p.x + sh, y) * u)
	return true


# Ink dries. Fresh ink is wetter, darker (or brighter, on a dark ground) and pulled
# toward the scheme's accent; over `_dry` seconds it settles to the page's own colour.
# It is the scene's whole colour dynamic, and it is free - age is a subtraction.
func _ink_color(age: float, fade: float) -> Color:
	var wet := exp(-maxf(0.0, age) / _dry)
	var h := _ink_hue
	var dh := _ink_wet - h
	h = fposmod(h + (dh - round(dh)) * wet * 0.8, 1.0)
	var s := clampf(_ink_sat * (0.55 + 0.75 * _ch.y) + 0.10 * wet, 0.0, 1.0)
	var v := _ink_val * (1.0 - 0.35 * wet) if _paper else _ink_val * (1.0 + 0.30 * wet)
	return Color.from_hsv(h, s, clampf(v, 0.0, 1.0), clampf(_ink_alpha * fade, 0.0, 1.0))


# The nib itself, resting on the paper where the ink stops. A small wedge, no more -
# the hand and the pen holding it are implied by the shaft angle and the ink behind it.
func _nib(u: float) -> void:
	if not _tip_ok:
		return
	var sc := _em * u
	var shaft := Vector2(0.42, -0.91).rotated(_tip_rot)
	var n := Vector2(-shaft.y, shaft.x)
	var l := sc * 0.60
	var w := sc * 0.085
	fill_aa(PackedVector2Array([
		_tip,
		_tip + shaft * (l * 0.5) + n * w,
		_tip + shaft * l,
		_tip + shaft * (l * 0.5) - n * w]), _nib_col, 1.0)
