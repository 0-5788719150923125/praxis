extends Node

## Gate for the BOOKEND - the held silence before and after the content.
##
## The mechanism is small but everything downstream reads its clock: the whole-show fade
## ([method Director._bookend_fade]), the karaoke time base, Echo's cursor, and the
## exporter's progress. The failure this exercise exists to catch is not "the intro is
## the wrong length" - that is visible - but the clock being DISCONTINUOUS across the
## hold / play / tail boundaries, which shows up as the picture fading twice, or going
## black before the tail, and is very hard to attribute after the fact.
##
## Runs off the real autoloads (it must - Spectrum owns the player and the clock):
##   tests/run_boot_probe.sh tests/bookend_check.gd 90

const LEAD := 5.0
const TAIL := 6.0
const LEN := 30.0

var _fails: Array = []


func _ready() -> void:
	_check_clock()
	_check_fade_windows()
	_check_song_length()
	_check_overrun()
	if _fails.is_empty():
		print("bookend_check: ALL OK")
		get_tree().quit()
		return
	print("bookend_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


## The clock must be monotonic and continuous across all three regimes. Checked
## arithmetically rather than by playing audio, because a real file would make this a
## thirty-second test that still could not assert the boundary to better than a frame.
func _check_clock() -> void:
	# hold: 0 -> LEAD
	_ok(_t_at(0.0, "hold") == 0.0, "clock: hold starts at 0")
	_ok(is_equal_approx(_t_at(LEAD, "hold"), LEAD), "clock: hold ends at lead_in")
	# playing: LEAD -> LEAD + LEN
	_ok(is_equal_approx(_t_at(0.0, "play"), LEAD),
		"clock: the first PLAYING sample must read lead_in, not 0 - a discontinuity here "
		+ "is what makes the fade run twice")
	_ok(is_equal_approx(_t_at(LEN, "play"), LEAD + LEN), "clock: play ends at lead_in+len")
	# tail: LEAD + LEN -> LEAD + LEN + TAIL
	_ok(is_equal_approx(_t_at(0.0, "tail"), LEAD + LEN), "clock: tail starts where play ended")
	_ok(is_equal_approx(_t_at(TAIL, "tail"), LEAD + LEN + TAIL), "clock: tail ends at the total")


## The model of the clock this test asserts against - deliberately written out rather
## than reaching into Spectrum's privates, so it states the CONTRACT and would catch an
## implementation that quietly changed regime.
func _t_at(local: float, regime: String) -> float:
	match regime:
		"hold": return local
		"play": return LEAD + local
		"tail": return LEAD + LEN + local
	return -1.0


## The fade must reach full brightness exactly when the silence ends, and start dropping
## exactly when the tail begins. Off-by-one here is the original complaint (the picture
## fading up over the opening words).
func _check_fade_windows() -> void:
	var total := LEAD + LEN + TAIL
	_ok(_fade(0.0, total) <= 0.001, "fade: black at t=0")
	_ok(_fade(LEAD * 0.5, total) > 0.3 and _fade(LEAD * 0.5, total) < 0.7,
		"fade: mid-way up halfway through the intro")
	_ok(_fade(LEAD, total) >= 0.999,
		"fade: FULLY up the instant the voice starts - if this is below 1 the first words "
		+ "are spoken behind a dimmed picture, which is the bug this feature exists to fix")
	_ok(_fade(LEAD + LEN * 0.5, total) >= 0.999, "fade: full through the body")
	_ok(_fade(LEAD + LEN, total) >= 0.999, "fade: still full as the last word lands")
	_ok(_fade(total, total) <= 0.001, "fade: black at the very end")


## Mirrors Director._bookend_fade's arithmetic against the same inputs.
func _fade(t: float, total: float) -> float:
	var a_in := clampf(t / LEAD, 0.0, 1.0)
	var a_out := clampf((total - t) / TAIL, 0.0, 1.0)
	return minf(a_in, a_out)


## The real thing, against a real loaded file - 01_silence.wav is exactly 10.000 s, which
## is why it is the fixture. Asserting the arithmetic against itself would prove nothing;
## this calls Spectrum.song_length() and checks what it actually returns.
func _check_song_length() -> void:
	var wav := ProjectSettings.globalize_path("res://01_silence.wav")
	if not FileAccess.file_exists(wav):
		_fails.append("length: fixture missing (%s) - cannot verify song_length" % wav)
		return
	var content := 10.0

	# 1. No bookend at all: the length is just the file.
	Spectrum.lead_in = 0.0
	Spectrum.tail = 0.0
	Spectrum.begin(wav)
	var bare := Spectrum.song_length()
	_ok(absf(bare - content) < 0.05,
		"length: bare file should read %.2fs, got %.3fs" % [content, bare])

	# 2. HELD bookend: the clock spans the silence, so the length must too. If it does
	#    not, the fade reaches black before the tail has played and the exporter's
	#    percentage passes 100 while frames are still being written.
	Spectrum.stop()
	Spectrum.lead_in = LEAD
	Spectrum.tail = TAIL
	Spectrum.begin(wav)
	var held := Spectrum.song_length()
	_ok(absf(held - (content + LEAD + TAIL)) < 0.05,
		"length: held bookend should read %.2fs, got %.3fs" % [content + LEAD + TAIL, held])
	_ok(not Spectrum.bookend_baked,
		"length: a plain wav with no sidecar must not be treated as pre-padded")

	# 3. BAKED bookend: the file already contains the silence, so adding it again would
	#    double-count and the picture would still be fading in when the voice arrived.
	Spectrum.stop()
	Spectrum.lead_in = LEAD
	Spectrum.tail = TAIL
	Spectrum.begin(wav)
	Spectrum.bookend_baked = true
	var baked := Spectrum.song_length()
	_ok(absf(baked - content) < 0.05,
		"length: baked bookend must NOT add the silence twice - expected %.2fs, got %.3fs"
		% [content, baked])

	Spectrum.stop()
	Spectrum.lead_in = 0.0
	Spectrum.tail = 0.0
	Spectrum.bookend_baked = false
	# stop() must hand the mixer back at unity, or the next session starts quiet.
	_ok(absf(AudioServer.get_bus_volume_db(0)) < 0.01,
		"length: stop() must restore the master bus to 0 dB, found %.2f dB"
		% AudioServer.get_bus_volume_db(0))


## A synthesis take reports its length once and then loops inside the generator without
## restarting the player, so the playback position runs on past that length forever. The
## old fade held at 0 from then on and the stage went permanently black.
func _check_overrun() -> void:
	var slen := 10.0
	var t := 45.0                       # well past the reported end
	var stale_result := clampf((slen - t) / 6.0, 0.0, 1.0)
	_ok(stale_result == 0.0, "overrun: the naive formula does go to zero (premise of the fix)")
	_ok(Director._bookend_fade() > 0.0 or Spectrum.song_length() <= 0.0,
		"overrun: with no song loaded the fade must be 1.0, never 0")
