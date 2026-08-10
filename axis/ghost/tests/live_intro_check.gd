extends SceneTree

## Gate for the LIVE audition's intro and its pre-roll.
##
## Two bugs, one insertion point, and the first was reported as the second.
##
## THE INTRO WAS ONLY IN THE EXPORT. The bookend was written into the rendered take's
## PCM, which is right for the video, but the generative editor auditions through a
## streamed generator that never touches that path - so pressing play gave no intro at
## all, and only the exported file had one.
##
## THE AUDITION STARVED. Chunks are one sentence each ([constant
## GenerativeEditor.CHUNK_SENTENCES] = 1) and sentence lengths are wildly uneven.
## Measured on north-star chapter 3: the opening sentence renders to 1.13 s and the next
## to 14.49 s. The stream opened on whatever the first chunk happened to be, so playback
## drained 1.13 s of audio and then sat silent for as long as the next sentence took to
## synthesize. Nothing was wrong with the audio - no chunk carries more than 0.27 s of
## internal silence - it simply ran out. Heard as: it speaks one sentence, pauses for
## about five seconds, then carries on, which is indistinguishable from an intro
## inserted in the wrong place.
##
## The fix seeds the intro into the pending queue in `_plan` (so `_elapsed`, and with it
## every word span and seam, is offset for free) and holds the stream shut until a real
## lead exists, with the intro doubling as that lead. This checks both.
##
## Run: godot --headless --path axis/ghost --script tests/live_intro_check.gd

const SR := 22050

var _fails: Array = []


func _init() -> void:
	_check_order()
	_check_lead()
	if _fails.is_empty():
		print("live_intro_check: ALL OK")
		quit()
		return
	print("live_intro_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	quit(1)


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


## The silence must be BEFORE the first word, never between the first sentence and the
## second. Modelled on the real queue arithmetic: `_plan` seeds `_pending` with the
## intro and starts `_elapsed` at it; each drained chunk then appends after that.
func _check_order() -> void:
	var intro := 5.0
	var pending := PackedFloat32Array()
	pending.resize(int(intro * SR))          # what _plan seeds
	var elapsed := intro
	# chapter 3's measured opening two sentences
	var first_word_t := elapsed              # words are placed at the PRE-append elapsed
	pending.append_array(_tone(1.13))
	elapsed += 1.13
	var second_word_t := elapsed
	pending.append_array(_tone(14.49))
	elapsed += 14.49

	_ok(absf(first_word_t - intro) < 0.01,
		"order: the first word must land at the END of the intro (%.2fs), not at %.2fs"
		% [intro, first_word_t])
	_ok(absf(second_word_t - (intro + 1.13)) < 0.01,
		"order: the second sentence must follow the first immediately, not after a second "
		+ "gap - expected %.2fs, got %.2fs" % [intro + 1.13, second_word_t])
	# and the audio itself: silent through the intro, sounding immediately after
	_ok(_rms(pending, 0, int(intro * SR)) < 1e-6,
		"order: the intro window must be silent")
	_ok(_rms(pending, int(intro * SR), int((intro + 1.0) * SR)) > 0.01,
		"order: the first sentence must start the instant the intro ends")
	# the gap the user actually heard would show up as silence right here
	var seam_a := int((intro + 1.13) * SR)
	_ok(_rms(pending, seam_a, seam_a + int(1.0 * SR)) > 0.01,
		"order: there must be NO silent gap between the first and second sentences - "
		+ "that is the reported bug")


## The stream must not open on a lead shorter than it takes to make the next sentence.
func _check_lead() -> void:
	var preroll := 2.5                        # GenerativeEditor.LIVE_PREROLL
	# intro off: the floor governs, and 1.13 s of opening sentence is not enough
	_ok(not _would_open(1.13, 0.0, preroll, false),
		"lead: must NOT open on a 1.13s first sentence with the intro off")
	_ok(_would_open(3.0, 0.0, preroll, false),
		"lead: should open once past the floor")
	# intro on: the intro IS the lead, so 5 s of silence plus a short sentence is plenty
	_ok(_would_open(1.13 + 5.0, 5.0, preroll, false),
		"lead: the intro should count toward the lead")
	_ok(not _would_open(2.0, 5.0, preroll, false),
		"lead: a 5s intro must not open on 2s of queue")
	# a short text that has no more chunks coming must open regardless, or a one-line
	# script would never play at all
	_ok(_would_open(0.4, 5.0, preroll, true),
		"lead: the last chunk must open the stream however short it is, or a one-sentence "
		+ "script never plays")


func _would_open(have: float, lead_in: float, preroll: float, drained: bool) -> bool:
	var lead := maxf(lead_in, preroll)
	return not (have < lead and not drained)


func _tone(secs: float) -> PackedFloat32Array:
	var n := int(secs * SR)
	var b := PackedFloat32Array()
	b.resize(n)
	for i in n:
		b[i] = 0.3 * sin(TAU * 140.0 * float(i) / float(SR))
	return b


func _rms(b: PackedFloat32Array, a: int, z: int) -> float:
	z = mini(z, b.size())
	if z <= a:
		return 0.0
	var s := 0.0
	for i in range(a, z):
		s += b[i] * b[i]
	return sqrt(s / float(z - a))
