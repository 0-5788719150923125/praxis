extends SceneTree

## Echo LATENCY PROBE against a real song (a measurement tool, not a pass/fail check):
##   godot --path axis/ghost --headless --script res://tests/echo_song_probe.gd
## Bakes the song offline (same FFT pipeline the export render uses), then simulates a
## manual session: pass 1 records the echo map while the cursor tracks; the audio then
## LOOPS while the cursor free-runs on into the tail claim. Reports, per descriptor
## time constant: when recognition confirms after the wrap, what schedule time it
## resolves to, any false fires, and refire behavior after a top rejoin.

const Bake := preload("res://scripts/bake.gd")
const SONG := "/home/crow/Music/Beats Able/Beats Enable.wav"
const BANDS := 64
const FPS := 30
const FMIN := 30.0
const FMAX := 16000.0
const DB := 60.0

var _frames: Array = []


func _named(raw: PackedFloat32Array, lo: float, hi: float) -> float:
	var ratio := FMAX / FMIN
	var b0 := clampi(int(BANDS * log(lo / FMIN) / log(ratio)), 0, BANDS - 1)
	var b1 := clampi(int(BANDS * log(hi / FMIN) / log(ratio)), 0, BANDS - 1)
	var s := 0.0
	for b in range(b0, b1 + 1):
		s += raw[b]
	return s / float(maxi(1, b1 - b0 + 1))


func _centres() -> PackedFloat32Array:
	var c := PackedFloat32Array()
	c.resize(BANDS)
	var ratio := FMAX / FMIN
	for i in BANDS:
		var lo := FMIN * pow(ratio, float(i) / float(BANDS))
		var hi := FMIN * pow(ratio, float(i + 1) / float(BANDS))
		c[i] = sqrt(lo * hi)
	return c


# Feed frame [param idx] into [param sig] (mirrors Spectrum._process's aggregation).
func _feed(sig: HarmonicSignature, idx: int, prev: PackedFloat32Array, dt: float) -> PackedFloat32Array:
	var raw: PackedFloat32Array = _frames[idx]
	var flux := 0.0
	if prev.size() == raw.size():
		for i in raw.size():
			flux += maxf(0.0, raw[i] - prev[i])
		flux /= float(raw.size())
	var low := _named(raw, 30.0, 150.0) + _named(raw, 150.0, 500.0)
	var mid := _named(raw, 500.0, 2000.0)
	var high := _named(raw, 2000.0, 6000.0) + _named(raw, 6000.0, 16000.0)
	sig.update(raw, low, mid, high, flux, dt)
	return raw


func _probe(tau: float) -> void:
	var dt := 1.0 / float(FPS)
	var n := _frames.size()
	var song_len := float(n) * dt
	var sig := HarmonicSignature.new(_centres(), tau)
	var echo := Echo.new()
	var prev := PackedFloat32Array()
	var heard := 0.0
	var false_fires := 0
	# Pass 1: cursor tracks the audio exactly; the map records the whole hearing.
	for i in n:
		prev = _feed(sig, i, prev, dt)
		echo.record(heard, sig.vector())
		if echo.listen(heard, heard, sig.vector(), dt) >= 0.0:
			false_fires += 1
		heard += dt
	# Pass 2: the audio loops; the cursor free-runs into its tail claim (past the map).
	var cursor := heard
	var confirm := -1.0
	var target := -1.0
	var i2 := 0
	while i2 < n and confirm < 0.0:
		prev = _feed(sig, i2, prev, dt)
		var to := echo.listen(heard, cursor, sig.vector(), dt)
		if to >= 0.0:
			confirm = float(i2) * dt
			target = to
		heard += dt
		cursor += dt
		i2 += 1
	# After a TOP rejoin (cursor snapped to the schedule head, late by `confirm`):
	# does the machinery leave the deliberate small lateness alone?
	var refires := 0
	var refire_at := -1.0
	if confirm >= 0.0:
		cursor = 0.0
		while i2 < n:
			prev = _feed(sig, i2, prev, dt)
			var to2 := echo.listen(heard, cursor, sig.vector(), dt)
			if to2 >= 0.0:
				refires += 1
				if refire_at < 0.0:
					refire_at = float(i2) * dt
			heard += dt
			cursor += dt
			i2 += 1
	print("tau %.2f | pass1 false fires %d | confirm %s | target %s | refires after top rejoin %d%s" % [
		tau, false_fires,
		("%.2fs" % confirm) if confirm >= 0.0 else "NEVER",
		("%.2fs" % target) if confirm >= 0.0 else "-",
		refires,
		(" (first at %.1fs)" % refire_at) if refires > 0 else ""])


func _initialize() -> void:
	print("baking %s ..." % SONG)
	_frames = Bake.bake(SONG, FPS, BANDS, FMIN, FMAX, DB)
	if _frames.is_empty():
		print("BAKE FAILED (ffmpeg on PATH?)")
		quit(1)
		return
	print("%d frames (%.2fs)" % [_frames.size(), _frames.size() / float(FPS)])
	for tau in [2.5, 1.2, 0.7, 0.4]:
		_probe(tau)
	quit(0)
