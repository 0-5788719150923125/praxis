extends SceneTree
## Scratch: render the ambience at several severities over a synthetic voice; report levels.
const SR := 22050
func _init() -> void:
	var out := OS.get_cmdline_user_args()[0] if OS.get_cmdline_user_args().size() > 0 else "/tmp"
	for sev in [0.1, 0.5, 1.0]:
		var fx := VoiceFX.new()
		fx.pad_seed = 12345
		fx.setup(SR)
		fx.pad = sev
		var secs := 90
		var buf := PackedFloat32Array()
		buf.resize(secs * SR)
		var ph := 0.0
		for i in buf.size():
			var t := float(i) / SR
			# 3 s of "speech" (a 120 Hz buzz with a syllable envelope), 1.5 s pause
			var on := fmod(t, 4.5) < 3.0
			var syl := 0.5 + 0.5 * sin(TAU * 4.0 * t)
			ph += TAU * (120.0 + 8.0 * sin(TAU * 0.3 * t)) / SR
			var v := (sin(ph) + 0.5 * sin(2 * ph) + 0.3 * sin(3 * ph)) * 0.05 * syl if on else 0.0
			buf[i] = v
		var dry := buf.duplicate()
		fx.prime_key(buf)
		buf = fx.process(buf)
		var acc := 0.0
		var peak := 0.0
		for i in buf.size():
			var d := buf[i] - dry[i]
			acc += d * d
			peak = maxf(peak, absf(d))
		var vr := 0.0
		for i in dry.size():
			vr += dry[i] * dry[i]
		print("sev %.1f: ambience RMS %.4f peak %.3f  (voice RMS %.4f)" % [sev, sqrt(acc / buf.size()), peak, sqrt(vr / dry.size())])
		var wav := AudioStreamWAV.new()
		wav.format = AudioStreamWAV.FORMAT_16_BITS
		wav.mix_rate = SR
		var bytes := PackedByteArray()
		bytes.resize(buf.size() * 2)
		for i in buf.size():
			bytes.encode_s16(i * 2, int(clampf(buf[i], -1, 1) * 32767))
		wav.data = bytes
		wav.save_to_wav(out.path_join("amb_%02d.wav" % int(sev * 10)))
	quit()
