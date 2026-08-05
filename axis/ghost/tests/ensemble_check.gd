extends SceneTree

## Does the BELT COMPOSE, or only select?
##
## The question this answers, in the user's words: if the first seed does not
## sing, can a second one added to the belt bring that mode in, and can a third
## modulate it further? Before `_ensemble_traits`, a throw was a jittered copy of
## exactly ONE parent, so the answer was no - adding a singer only added a
## lottery ticket, and no third seed could touch a mode a second had brought.
##
## This models the same arithmetic the editor uses, so it can run headless
## without booting the game: `out = dominant + sum_i w_i * (member_i - dominant)`
## with signed, bounded weights.
##
## Run: godot --headless --path axis/ghost --script tests/ensemble_check.gd

const W := 0.45          # must match SynthEditor.ENSEMBLE_W
const MAX_EXTRA := 2     # must match SynthEditor.ENSEMBLE_MAX


func _init() -> void:
	var fails := 0
	fails += _check_inherit()
	fails += _check_third_modulates()
	fails += _check_subtracts()
	fails += _check_no_collapse()
	fails += _check_sustain_positions()
	if fails == 0:
		print("ensemble_check: ALL OK")
	else:
		print("ensemble_check: %d FAILURE(S)" % fails)
	quit(fails)


func _blend(dominant: float, members: Array, weights: Array) -> float:
	var out := dominant
	for i in members.size():
		out = clampf(out + float(weights[i]) * (float(members[i]) - out), -1.0, 1.0)
	return out


## A non-singing dominant, one singer on the belt: the child must be able to
## come out singing (song > 0) for some reachable weight.
func _check_inherit() -> int:
	var best := -1.0
	for step in 21:
		var w: float = -W + 2.0 * W * float(step) / 20.0
		best = maxf(best, _blend(-0.6, [0.9], [w]))
	var ok: bool = best > 0.0
	print("ensemble_check: non-singer (-0.60) + singer (0.90) -> best reachable song %.2f %s" % [
		best, "ok - the mode can be inherited" if ok else "FAIL - cannot cross zero"])
	return 0 if ok else 1


## A third contributor must be able to move it FURTHER than the second alone.
func _check_third_modulates() -> int:
	var two: float = _blend(-0.6, [0.9], [W])
	var three: float = _blend(-0.6, [0.9, 0.8], [W, W])
	var ok: bool = three > two + 0.05
	print("ensemble_check: second seed -> %.2f, third pushes it to %.2f %s" % [
		two, three, "ok - further modulation" if ok else "FAIL - third contributes nothing"])
	return 0 if ok else 1


## And it must be able to move it DOWN. Combination that can only ever add is
## how a collection converges on one voice.
func _check_subtracts() -> int:
	var up: float = _blend(0.5, [0.9], [W])
	var down: float = _blend(0.5, [0.9], [-W])
	var ok: bool = down < 0.5 and up > 0.5
	print("ensemble_check: singer (0.50) with a +w singer -> %.2f, with a -w singer -> %.2f %s" % [
		up, down, "ok - signed weights subtract" if ok else "FAIL - cannot reduce"])
	return 0 if ok else 1


## The guard: composing must NOT be averaging. Over many random belts the
## children's spread has to stay comparable to the parents' - if it collapses,
## every catch drifts toward the belt mean and the game stops making new voices.
func _check_no_collapse() -> int:
	var rng := RandomNumberGenerator.new()
	rng.seed = 99
	var parents: Array = []
	var kids: Array = []
	for _i in 4000:
		var dom: float = rng.randf_range(-1.0, 1.0)
		parents.append(dom)
		var members: Array = []
		var weights: Array = []
		for _k in MAX_EXTRA:
			if rng.randf() < 0.5:
				members.append(rng.randf_range(-1.0, 1.0))
				weights.append(rng.randf_range(-W, W))
		kids.append(_blend(dom, members, weights))
	var ok: bool = _sd(kids) > _sd(parents) * 0.8
	print("ensemble_check: parent spread %.3f -> child spread %.3f (need > 80%%) %s" % [
		_sd(parents), _sd(kids), "ok - composes without collapsing" if ok else "FAIL - regresses to the mean"])
	return 0 if ok else 1


## THE POINT OF THE BANK: contributors must change WHERE a voice sings, not
## only how much. Two banks with the same total weight must hold a different
## SET of syllables - if adding a seed only scaled the drive, the held
## positions would be identical and the belt would just be a volume knob.
func _check_sustain_positions() -> int:
	var solo := Voice.Spec.from_traits({"song": 0.8, "drawl": 0.2})
	var with_two := Voice.Spec.from_traits({"song": 0.8, "drawl": 0.2})
	with_two.sustain_bank = [
		[solo.sustain_period, solo.sustain_phase, 1.0],
		[7.3, 0.31, 0.5],
	]
	var with_neg := Voice.Spec.from_traits({"song": 0.8, "drawl": 0.2})
	with_neg.sustain_bank = [
		[solo.sustain_period, solo.sustain_phase, 1.0],
		[7.3, 0.31, -0.5],
	]
	var a := _peaks(solo)
	var b := _peaks(with_two)
	var c := _peaks(with_neg)
	var ab := _jaccard(a, b)
	var ac := _jaccard(a, c)
	var bc := _jaccard(b, c)
	# same MEAN drive by construction (the added cycles are zero-mean), so any
	# difference here is positional, not a level change
	var ok: bool = ab < 0.85 and ac < 0.85 and bc < 0.85
	print("ensemble_check: held-position overlap - solo vs +w %.2f, solo vs -w %.2f, +w vs -w %.2f %s" % [
		ab, ac, bc, "ok - contributors move WHERE it sings" if ok else "FAIL - same positions, only louder"])
	print("                held syllables: solo %s | +w %s | -w %s" % [
		str(a).substr(0, 40), str(b).substr(0, 40), str(c).substr(0, 40)])
	return 0 if ok else 1


## Which of the first 60 syllables clear the sustain bar.
func _peaks(spec: Voice.Spec) -> Array:
	var out: Array = []
	for i in 60:
		# same drive form as plan(): cycle plus a mid prominence
		if 0.62 * Voice.Spec.sustain_wave(spec, i) + 0.55 * 0.6 > Voice.SUSTAIN_BAR:
			out.append(i)
	return out


func _jaccard(a: Array, b: Array) -> float:
	var inter := 0
	for v in a:
		if b.has(v):
			inter += 1
	var uni: int = a.size() + b.size() - inter
	return float(inter) / maxf(float(uni), 1.0)


func _sd(a: Array) -> float:
	var m := 0.0
	for v in a:
		m += float(v)
	m /= float(a.size())
	var s := 0.0
	for v in a:
		s += (float(v) - m) * (float(v) - m)
	return sqrt(s / float(a.size()))
