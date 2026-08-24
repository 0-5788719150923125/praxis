extends Node

## Gate for the SPREAD of clockwork's wheels and their wear - that the scene can build more
## than one kind of gear and corrode it more than one way.
##
##   tests/run_boot_probe.sh tests/clockwork_variety_check.gd 180
##
## No GPU: every quantity here is sampled on the CPU into the gear dictionaries, so the
## question is what the RANGES are, not what the pixels look like.
##
## WHAT WENT WRONG, and it was not a narrow range in either case - it was a constant.
##
##   THE ARMS. A spoke was a `draw_line` from the hub to very nearly the tooth root at a width
##   of 0.014 of the wheel's radius. Not sampled: written. So every spoked wheel in every
##   machine wore the same long thin arms, and the only thing that varied was how many. The
##   count could not reach two either. Reported as "the gears always have a very similar kind
##   of look: long spokes, with many of them... we could also allow for gears with fewer,
##   shorter, and wider spokes - and I just think our current implementation doesn't allow for
##   that", which was exactly right.
##
##   THE WEAR. The mark colour was `Color.from_hsv(0.05 + hue * 0.1, 0.6, 0.30)`: a hue pinned
##   to the orange-brown eighth of the wheel, and a saturation and value that were literal
##   constants across every mark, every wheel and every machine. One shape too - a soft round
##   blob. Reported as "rust is always brown, and blotchy; it always looks exactly the same".
##
## THE CONTROL IS THE RETIRED ARITHMETIC, evaluated here over the same machines. Both old
## formulas are three lines, they cannot drift, and they have to FAIL the spread checks below
## or those checks are not measuring anything. A range test with no control passes on any
## implementation that happens to be noisy.

const MACHINES := 60
## A quantity is not "varied" if its extremes are this close together. Ratio rather than
## difference so it reads the same whatever the units.
const MIN_SPREAD := 2.0
## Wear hue has to cover more than one family of colour - brown AND green AND near-neutral.
## In turns of the wheel: the retired formula spanned 0.075 of it.
const MIN_HUE_SPAN := 0.25
## Arms this few, and this many, both have to be reachable.
const FEW_ARMS := 2
const MANY_ARMS := 8
## A wheel whose arms run less than this fraction of its radius is a SHORT-armed one, and at
## least one machine in the sample has to cast some.
const SHORT_ARM := 0.25

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


func _span(a: Array) -> Array:
	var lo := INF
	var hi := -INF
	for x in a:
		lo = minf(lo, float(x))
		hi = maxf(hi, float(x))
	return [lo, hi] if a.size() > 0 else [0.0, 0.0]


func _run() -> void:
	var widths: Array = []          # arm width over wheel radius
	var lengths: Array = []         # arm length over wheel radius
	var counts := {}
	var marks := {}
	var w_h: Array = []
	var w_s: Array = []
	var w_v: Array = []
	# THE CONTROL: what the retired rules would have produced on these same machines.
	var old_w: Array = []
	var old_h: Array = []
	var old_s: Array = []
	var old_v: Array = []
	var wheels := 0
	var spoked := 0

	for sv in range(1, MACHINES + 1):
		var sc = load("res://scripts/scenes/clockwork.gd").new()
		sc.init_with_seed(sv, "drift")
		for g in sc._gears:
			var gd: Dictionary = g
			wheels += 1
			if String(gd["style"]) == "wire":
				spoked += 1
				var n := int((gd["spokes"] as Array).size())
				counts[n] = int(counts.get(n, 0)) + 1
				widths.append(float(gd["spoke_w"]) * 2.0)
				lengths.append(float(gd["rim_in"]) - float(gd["hub_r"]) * 0.85)
				old_w.append(0.014)                       # the retired constant width
			if not (gd["rust"] as Array).is_empty():
				var mk := String(gd["wear_mark"])
				marks[mk] = int(marks.get(mk, 0)) + 1
			for sp in gd["rust"]:
				var spd: Dictionary = sp
				w_h.append(float(spd["h"]))
				w_s.append(float(spd["s"]))
				w_v.append(float(spd["v"]))
				# the retired colour: hue keyed to the metal, saturation and value fixed
				old_h.append(fposmod(0.05 + float(sc._hue) * 0.1, 1.0))
				old_s.append(0.6)
				old_v.append(0.30)
		sc.free()

	var ks := counts.keys()
	ks.sort()
	var cline := ""
	for k in ks:
		cline += "%d:%d  " % [k, counts[k]]
	var ws := _span(widths)
	var ls := _span(lengths)
	var hs := _span(w_h)
	var ss := _span(w_s)
	var vs := _span(w_v)
	var ow := _span(old_w)
	var oh := _span(old_h)
	print("clockwork_variety_check: %d wheels over %d machines, %d spoked" % [wheels, MACHINES, spoked])
	print("   arms per wheel   %s" % cline)
	print("   arm width  / R   %.3f .. %.3f   (retired rule: %.3f .. %.3f)" % [ws[0], ws[1], ow[0], ow[1]])
	print("   arm length / R   %.3f .. %.3f" % [ls[0], ls[1]])
	print("   wear hue         %.3f .. %.3f   (retired rule: %.3f .. %.3f)" % [hs[0], hs[1], oh[0], oh[1]])
	print("   wear sat / val   %.2f .. %.2f / %.2f .. %.2f   (retired rule: 0.60 only / 0.30 only)"
		% [ss[0], ss[1], vs[0], vs[1]])
	print("   wear shapes      %s" % marks)

	_ok(spoked > 100, "only %d spoked wheels in %d machines - too few to say anything" % [spoked, MACHINES])
	_ok(ws[1] / maxf(ws[0], 1e-6) >= MIN_SPREAD,
		"arm width spans only %.2fx (%.3f to %.3f) - the wheels are all the same weight"
		% [ws[1] / maxf(ws[0], 1e-6), ws[0], ws[1]])
	_ok(ls[0] < SHORT_ARM,
		"the shortest arm is %.3f of the radius - nothing here casts a short-armed wheel" % ls[0])
	_ok(counts.has(FEW_ARMS), "no wheel came out with %d arms" % FEW_ARMS)
	_ok(counts.has(MANY_ARMS), "no wheel came out with %d arms" % MANY_ARMS)
	_ok(hs[1] - hs[0] >= MIN_HUE_SPAN,
		"wear hue spans only %.3f of the wheel - it is all one colour of corrosion" % (hs[1] - hs[0]))
	_ok(ss[1] / maxf(ss[0], 1e-6) >= MIN_SPREAD, "wear saturation barely varies (%.2f to %.2f)" % [ss[0], ss[1]])
	_ok(vs[1] / maxf(vs[0], 1e-6) >= MIN_SPREAD, "wear value barely varies (%.2f to %.2f)" % [vs[0], vs[1]])
	_ok(marks.size() >= 3, "only %d wear shape(s) ever appeared - %s" % [marks.size(), marks])

	# --- THE CONTROL: the retired rules have to fail the very same tests ---
	var os_ := _span(old_s)
	var ov := _span(old_v)
	_ok(ow[1] / maxf(ow[0], 1e-6) < MIN_SPREAD,
		"the RETIRED arm width was not a constant after all (%.3f to %.3f) - this check is not measuring the reported fault"
		% [ow[0], ow[1]])
	_ok((oh[1] - oh[0]) < MIN_HUE_SPAN,
		"the RETIRED wear hue already spanned %.3f of the wheel, so the hue check proves nothing"
		% (oh[1] - oh[0]))
	_ok(os_[1] / maxf(os_[0], 1e-6) < MIN_SPREAD and ov[1] / maxf(ov[0], 1e-6) < MIN_SPREAD,
		"the RETIRED wear saturation/value already varied, so those checks prove nothing")

	print("")
	if _fails.is_empty():
		print("clockwork_variety_check: ALL OK - the shop builds more than one wheel.")
		get_tree().quit()
		return
	print("clockwork_variety_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)
