extends Node

# Throwaway probe: what does one tidepool frame actually COST to build, and how fast do its
# waves actually move? Both numbers decide how much resolution the scene can buy.
# Run: tests/run_boot_probe.sh tests/tidepool_cost_probe.gd 240

const W := 1920.0
const H := 1080.0
const REPS := 8


func _ready() -> void:
	var script := load("res://scripts/scenes/tidepool.gd")
	for s in 6:
		var sc = script.new()
		sc.init_with_seed(3000 + s, "drift")
		sc.size = Vector2(W, H)

		var gw: int = sc._gw
		var gh: int = sc._gh
		var over: float = sc.OVER
		var cell := maxf(W * over / float(gw - 1), H * over / float(gh - 1))

		# Wave periods, straight off the dispersion relation the field was built with.
		var wf = sc._wf
		var n: int = wf.count()
		var slow := 0.0
		var fast := 1e9
		for c in n:
			var lam: float = wf.wavelength_of(c)
			var k := TAU / maxf(0.0001, lam)
			# _om is private but equals sqrt(g k) * tempo; recover the period from it directly.
			var om: float = wf._om[c]
			var per := TAU / maxf(0.0001, om)
			slow = maxf(slow, per)
			fast = minf(fast, per)

		var snap := {
			"t": 12.0,
			"amps": sc._amp.duplicate(),
			"steep": 1.0,
			"rings": [],
			"cell": cell,
			"water": Color(0.2, 0.5, 0.7),
			"sky": Color(0.6, 0.75, 0.95),
			"sun": sc._sun_col,
			"amb": sc._amb,
			"contrast": sc._contrast,
			"sharp": sc._sharp,
			"exposure": sc._exposure,
			"glare_thr": 0.6,
			"glare_gain": sc._glare_gain,
		}
		# Amplitudes at a plausible mid drive, or the sweep skips every component as silent.
		for i in sc._amp.size():
			sc._amp[i] = sc._base[i] * 0.6
		snap["amps"] = sc._amp.duplicate()

		sc._job.run(snap)                          # warm: builds the mesh once
		var t0 := Time.get_ticks_usec()
		for _r in REPS:
			sc._job.run(snap)
		var ms := float(Time.get_ticks_usec() - t0) / 1000.0 / float(REPS)

		# Per-stage, so the budget is spent on evidence rather than on a guess about which
		# term dominates. Each stage is driven exactly as run() drives it.
		var job = sc._job
		var nn := gw * gh
		var t_sweep := 0.0
		var t_cast := 0.0
		var t_blur := 0.0
		var t_shade := 0.0
		for _r in REPS:
			job.hh.fill(0.0)
			job.gxx.fill(0.0)
			job.gyy.fill(0.0)
			job.jzz.fill(1.0)
			var a0 := Time.get_ticks_usec()
			job.wf.sweep(12.0, job.org, job.step, gw, gh, job.hh, job.gxx, job.gyy, job.jzz)
			var a1 := Time.get_ticks_usec()
			job.acc.fill(0.0)
			var total: float = job._cast()
			var a2 := Time.get_ticks_usec()
			job._blur()
			var a3 := Time.get_ticks_usec()
			var cols := PackedColorArray()
			cols.resize(nn + (gw - 1) * (gh - 1))
			job._shade(snap, float(job.wet_nodes) / maxf(0.0001, total), cols)
			job._centres(cols)
			var a4 := Time.get_ticks_usec()
			t_sweep += float(a1 - a0)
			t_cast += float(a2 - a1)
			t_blur += float(a3 - a2)
			t_shade += float(a4 - a3)
		var d := 1000.0 * float(REPS)
		print("tidepool_cost: seed=%d comps=%2d grid=%dx%d (%d nodes) cell=%.1fpx  build=%.1f ms  periods %.2f-%.2f s  bed=%d"
			% [3000 + s, n, gw, gh, nn, cell, ms, fast, slow, sc._job.bw])
		print("               sweep=%.1f  cast=%.1f  blur=%.1f (x%d)  shade+centres=%.1f  ms"
			% [t_sweep / d, t_cast / d, t_blur / d, job.blur_n, t_shade / d])
		sc.free()
	get_tree().quit()
