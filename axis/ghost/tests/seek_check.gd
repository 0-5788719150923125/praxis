extends Node

## Gate for SEEKING - the scrub bar's one load-bearing operation.
##
## The clock a caller hands to [method Spectrum.seek] is the SESSION clock, bookend and
## all, because that is the only clock anything outside Spectrum ever sees. Inside, that
## position has to be resolved against three different regimes - held silence, playing
## file, run-out tail - and against the two ways a bookend can exist (held here, or baked
## into the audio). Getting the offset wrong in either direction is invisible until a seek
## lands a few seconds off, which on a ten minute narration is very hard to attribute.
##
## Run: tests/run_boot_probe.sh tests/seek_check.gd 120

const LEAD := 5.0
const TAIL := 6.0
const LEN := 10.0                   # 01_silence.wav is exactly ten seconds

var _fails: Array = []


func _ready() -> void:
	var wav := ProjectSettings.globalize_path("res://01_silence.wav")
	if not FileAccess.file_exists(wav):
		print("seek_check: fixture missing")
		get_tree().quit(1)
		return
	await _held_bookend(wav)
	await _no_bookend(wav)
	_unseekable()
	Spectrum.stop()
	Spectrum.lead_in = 0.0
	Spectrum.tail = 0.0
	if _fails.is_empty():
		print("seek_check: ALL OK")
		get_tree().quit()
		return
	print("seek_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


## The clock must READ BACK what was asked for, across all three regimes.
func _held_bookend(wav: String) -> void:
	Spectrum.stop()
	Spectrum.lead_in = LEAD
	Spectrum.tail = TAIL
	Spectrum.begin(wav)
	_ok(Spectrum.seekable(), "a loaded file must be seekable")
	_ok(absf(Spectrum.song_length() - (LEAD + LEN + TAIL)) < 0.05,
		"length should span the bookend, got %.2f" % Spectrum.song_length())

	# into the held silence
	Spectrum.seek(2.0)
	await _settle()
	_ok(absf(Spectrum.current.time - 2.0) < 0.35,
		"seek into the intro should read 2.0s, got %.2f" % Spectrum.current.time)

	# into the content - the offset is where this goes wrong if it goes wrong
	Spectrum.seek(LEAD + 4.0)
	await _settle()
	_ok(absf(Spectrum.current.time - (LEAD + 4.0)) < 0.35,
		"seek into the content should read %.1fs, got %.2f"
		% [LEAD + 4.0, Spectrum.current.time])

	# clamped past the end rather than running off
	Spectrum.seek(999.0)
	await _settle()
	_ok(Spectrum.current.time <= LEAD + LEN + TAIL + 0.5,
		"seek past the end must clamp, got %.2f" % Spectrum.current.time)
	_ok(Spectrum.current.time > LEAD,
		"seek past the end must not fall back to the start, got %.2f" % Spectrum.current.time)

	# and back to zero
	Spectrum.seek(0.0)
	await _settle()
	_ok(Spectrum.current.time < 0.6,
		"seek home should read ~0, got %.2f" % Spectrum.current.time)


## Without a bookend the session clock and the player position are the same thing, so a
## seek that quietly applied the offset anyway would land LEAD seconds early.
func _no_bookend(wav: String) -> void:
	Spectrum.stop()
	Spectrum.lead_in = 0.0
	Spectrum.tail = 0.0
	Spectrum.begin(wav)
	Spectrum.seek(6.0)
	await _settle()
	_ok(absf(Spectrum.current.time - 6.0) < 0.35,
		"unbookended seek should read 6.0s, got %.2f" % Spectrum.current.time)


## A generator has nothing behind the playhead, so the control must refuse rather than
## appear to work - the scrub bar keys off exactly this.
func _unseekable() -> void:
	Spectrum.stop()
	_ok(not Spectrum.seekable(), "a stopped session must not report itself seekable")


func _settle() -> void:
	# Spectrum publishes `current` from _process, so the clock is a frame behind the seek.
	for _i in 3:
		await get_tree().process_frame
