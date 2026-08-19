extends Node

## Gate for WHEN the karaoke line is on screen.
##
## The overlay used to pick its sentence as "the one being spoken, or else the first one
## not yet spoken" - with no bound on how far ahead that second case reached. So before a
## word had been said it showed the opening line, and with a five second intro in front
## of the reading that line sat on screen, plate and all, for the whole intro. Nothing
## brought it off at the other end either.
##
## What is asserted here is the SHAPE of the presence envelope, because that is what was
## wrong - not the text, not the colour, not the layout, all of which were fine:
##   nothing on screen during a long silence before the first word;
##   the line arriving shortly before it is spoken, not minutes early;
##   the line leaving shortly after the last word rather than hanging through the outro;
##   and - the constraint that stops the fix becoming its own bug - NO blink across an
##   ordinary gap between two sentences.
##
## THE SECOND BUG, from a ch19 render: "at exactly the same place where the colon is, the
## subtitles briefly disappeared, then reappeared... the SAME subtitles flickered". There were
## TWO rules for what is on screen - the presence ease asked about the SENTENCE's span, the draw
## asked about the nearest WORD (+/- 0.4 s) - and they disagree for any pause longer than 1.2 s
## inside a sentence, which is what a colon's clause pause becomes once pause_scale stretches it.
## So the second half of this gate asks BOTH paths the same questions through a long internal
## pause, and its control is the case that must still go dark: a real silence between sentences.
## Asking only the presence path would have passed while the bug was live.
##
## Run inside a real boot (the overlay is a CanvasLayer, and Subtitles reads the
## Spectrum autoload):
##   tests/run_boot_probe.sh tests/subtitle_presence_check.gd 90

var _fails: Array = []


func _ready() -> void:
	var subs := Subtitles.new()
	# Two sentences: one at 5.0-6.1s (after a 5s intro), the next at 6.5-9.0s, then a
	# long tail of nothing - the exact shape a bookended take has.
	subs.words = [
		{"text": "This", "t0": 5.00, "t1": 5.30, "sentence": 0},
		{"text": "one", "t0": 5.30, "t1": 5.60, "sentence": 0},
		{"text": "is", "t0": 5.60, "t1": 5.85, "sentence": 0},
		{"text": "yours.", "t0": 5.85, "t1": 6.10, "sentence": 0},
		{"text": "The", "t0": 6.50, "t1": 6.70, "sentence": 1},
		{"text": "lake", "t0": 6.70, "t1": 7.10, "sentence": 1},
		{"text": "is", "t0": 7.10, "t1": 7.40, "sentence": 1},
		{"text": "a", "t0": 7.40, "t1": 7.55, "sentence": 1},
		{"text": "church.", "t0": 7.55, "t1": 9.00, "sentence": 1},
	]

	# --- the intro: nothing at all until the line is nearly due ---
	_off(subs, 0.0, "the very start of a 5s intro")
	_off(subs, 2.0, "two seconds into the intro")
	_off(subs, 4.0, "one second before the first word - still outside LEAD (%.1fs)"
		% Subtitles.LEAD)
	_on(subs, 4.5, "half a second before the first word - inside LEAD")

	# --- speaking ---
	_on(subs, 5.0, "the first word")
	_on(subs, 6.0, "mid first sentence")

	# --- the seam between two sentences must NOT blink ---
	_on(subs, 6.2, "in the 0.4s gap between sentences - a blink here would be worse "
		+ "than the bug being fixed")
	_on(subs, 6.5, "the second sentence's first word")
	_on(subs, 8.0, "mid second sentence")

	# --- the outro: gone shortly after the last word ---
	_on(subs, 9.2, "just after the last word, still within HANG (%.1fs)" % Subtitles.HANG)
	_off(subs, 10.0, "one second after the last word - past HANG")
	_off(subs, 12.0, "well into the outro")
	_off(subs, 15.0, "the end of the outro")

	# --- and the ease actually eases, rather than snapping ---
	_check_ease(subs)

	# --- A LONG PAUSE INSIDE ONE SENTENCE MAY NOT BLINK (the colon) ---
	# One sentence, spoken either side of a 1.8s pause where the colon is. Both paths are asked,
	# every 50ms, right through it: the drawn line must stay the SAME five words the whole way.
	var colon := Subtitles.new()
	add_child(colon)
	colon.words = [
		{"text": "men", "t0": 1.00, "t1": 1.30, "sentence": 0},
		{"text": "are", "t0": 1.30, "t1": 1.55, "sentence": 0},
		{"text": "not:", "t0": 1.55, "t1": 2.00, "sentence": 0},
		{"text": "it", "t0": 3.80, "t1": 4.00, "sentence": 0},
		{"text": "felt", "t0": 4.00, "t1": 4.40, "sentence": 0},
	]
	var dark := 0
	var thin := 0
	var t := 2.0
	while t < 3.8:
		if colon._presence_target(t) < 0.5:
			thin += 1
		if colon._overlay._current_sentence(t).size() != 5:
			dark += 1
		t += 0.05
	print("subtitle_presence_check: through a 1.8s pause inside a sentence - %d instants with "
		% dark + "no line to draw, %d with the plate faded out" % thin)
	if dark > 0:
		_fails.append("colon: the drawn line vanished at %d instants inside one sentence - "
			% dark + "that is the reported flicker")
	if thin > 0:
		_fails.append("colon: presence dropped at %d instants inside one sentence" % thin)
	# THE CONTROL. The fix must not become "the line never leaves": a genuine silence between two
	# sentences, longer than HANG + LEAD, still has to clear the frame.
	colon.words = [
		{"text": "one.", "t0": 1.00, "t1": 1.50, "sentence": 0},
		{"text": "two.", "t0": 5.00, "t1": 5.50, "sentence": 1},
	]
	var cleared := false
	t = 1.5
	while t < 5.0:
		if colon._presence_target(t) < 0.5 and colon._overlay._current_sentence(t).is_empty():
			cleared = true
			break
		t += 0.05
	print("subtitle_presence_check: a 3.5s silence BETWEEN sentences clears the frame: %s"
		% ("yes at t=%.2f" % t if cleared else "NO"))
	if not cleared:
		_fails.append("control: a 3.5s silence between sentences never cleared the line - the "
			+ "rule now holds a line on screen through anything")
	colon.queue_free()

	subs.queue_free()
	if _fails.is_empty():
		print("subtitle_presence_check: ALL OK")
		get_tree().quit()
		return
	print("subtitle_presence_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)


func _on(subs: Subtitles, t: float, why: String) -> void:
	if subs._presence_target(t) < 0.5:
		_fails.append("t=%.2fs should SHOW the line: %s" % [t, why])


func _off(subs: Subtitles, t: float, why: String) -> void:
	if subs._presence_target(t) > 0.5:
		_fails.append("t=%.2fs should show NOTHING: %s" % [t, why])


## Presence is eased, not switched. Stepping the real _process arithmetic from 0 toward 1
## must take a handful of frames - if it arrived in one, the line would pop on at full
## brightness, which is the only hard cut anywhere in the frame.
func _check_ease(subs: Subtitles) -> void:
	var p := 0.0
	var dt := 1.0 / 60.0
	var frames := 0
	while p < 0.9 and frames < 600:
		p = lerpf(p, 1.0, 1.0 - exp(-dt / maxf(0.01, Subtitles.FADE)))
		frames += 1
	var secs := float(frames) * dt
	print("subtitle_presence_check: fade to 90%% takes %.2fs over %d frames" % [secs, frames])
	if frames < 5:
		_fails.append("ease: presence reached full in %d frames - that is a pop, not a fade"
			% frames)
	if secs > 1.5:
		_fails.append("ease: %.2fs to fade in is sluggish; the line would trail the voice" % secs)
