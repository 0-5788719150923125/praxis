extends RefCounted
class_name Grains

## Grains - a cellular automaton of MATTER: one material per cell, moved by local rules.
##
## Every other many-item primitive here carries a CONTINUOUS scalar. [Swarm] is a field
## of floats that grows or diffuses; [Flow2D] is a velocity field; a [ParticleSystem] is
## a list of bodies with no notion of occupancy. None of them can express the one thing a
## pile of sand is about: a cell holds EXACTLY ONE thing, two grains cannot occupy it, and
## the silhouette you end up looking at was produced by nothing but a few local rules
## applied a few thousand times. Angle of repose, avalanche, buoyancy layering and drainage
## collapse are then consequences rather than code paths - there is no "avalanche" branch
## anywhere in this file.
##
## It also cannot be a Swarm rule, and not for want of trying: Swarm is DOUBLE-BUFFERED and
## updates every cell simultaneously off the previous frame, which is exactly right for
## diffusion and exactly wrong for matter - two grains both read "the cell below is empty"
## and both move into it, and one of them is quietly deleted. Matter needs an IN-PLACE,
## ORDERED sweep: bottom row first, so a grain only ever falls into space that was vacated
## earlier in the same tick, and a per-row alternating left/right bias so the ordering does
## not bake a permanent drift into one direction.
##
## THE RULES, all of them:
##   POWDER  falls straight down; if blocked, it MAY slide diagonally, with probability
##           [code]repose[/code]. That probability is the material's angle of repose - a
##           grain that always slides spreads into a flat apron, one that never slides
##           stacks into a vertical column, and everything in between is a cone with a
##           characteristic flank angle. It is the single most important number here.
##   LIQUID  falls, slides diagonally without a gate, then searches sideways up to
##           [code]spread[/code] cells for anywhere lower or emptier - which is what makes
##           it find its level rather than pile up.
##   GAS     the same rules with gravity inverted, so it rises through anything denser.
##   Density decides who displaces whom, so oil floats on water, sand sinks through both,
##           and ash of the right density rafts on the surface. Nothing anywhere says
##           "float"; the pass table (see [method finalize]) is a single byte lookup per
##           test and buoyancy falls out of it.
##   REACTIVE materials transmute: a decay probability turns a cell into something else
##           (an ember cooling into ash), and an igniter converts flammable NEIGHBOURS into
##           whatever they burn to, while a liquid neighbour that boils quenches it.
##   DRAFT   an exposed grain may hop one cell downwind, unless it is sheltered - the LEE
##           SHADOW rule: a cell immediately downwind of an obstruction neither erodes nor
##           accepts a hop. That one rule is the whole difference between a pile that looks
##           dumped and a pile that looks wind-worked: windward faces get planed off and
##           material accumulates behind whatever stands in the way.
##
## DETERMINISM. The automaton needs a random number several times per cell per tick, and
## an [RandomNumberGenerator] cannot supply it: drawing conditionally (only when a grain is
## actually blocked) makes the number of draws depend on the state, and the state depends
## on the audio - and the live analyzer and the offline export bake do NOT produce
## frame-identical features (spectrum.gd:406-411), so the exported video would desynchronise
## from the preview and never resynchronise. Instead a NOISE TABLE is filled once from the
## seeded rng and indexed by a pure hash of (cell, generation, seed). No draws are consumed
## at all, every cell's randomness is a pure function of where and when it is, and an
## audio-conditioned decision (an ignition chance riding on flux) can therefore only change
## that one decision instead of shifting an entire stream.
##
## PERFORMANCE. The sweep is O(cells) and lives on a worker thread ([FrameForge]) in every
## scene that uses it. Two things keep the constant small: a per-row activity mask, so the
## settled interior of a deep pile is skipped entirely once it stops moving (with a full
## sweep every [constant FULL_EVERY] ticks so a slow reaction buried inside it is never
## lost), and the pass table above, which turns "may this material enter that one" into one
## byte read instead of a function call - at four tests per cell and twenty thousand cells
## a tick, that call overhead was the dominant cost.

## Movement classes. AIR is the empty cell itself - deliberately not SOLID, so the checks
## that ask "is my neighbour a wall" answer no for open space.
const AIR := -1
const SOLID := 0
const POWDER := 1
const LIQUID := 2
const GAS := 3

const KINDS := {"solid": SOLID, "powder": POWDER, "liquid": LIQUID, "gas": GAS}

## Noise table size (a power of two so the index is a mask, not a modulo).
const NOISE := 4096
const NMASK := 4095

## A full sweep is forced this often, whatever the activity mask says. The mask can only
## be woken by a neighbour that moved, so a cell whose only future is a slow transmutation
## (an ember buried in a settled pile) would otherwise sleep forever. Sixteen ticks is
## about half a second - short enough that nothing visibly freezes, long enough that the
## saving on a settled chamber is still most of the sweep.
const FULL_EVERY := 16

## The matter table. Every entry is a RANGE, sampled per instance - two songs that both
## roll sand do not get the same sand, they get sand of a different repose, density and
## colour. `hue` is a RELATIVE offset in turns from whatever base hue the scene hands the
## renderer, so a chamber stays inside its [Scheme] while its materials still read apart
## from each other (water is always most of a half-turn from sand, whatever colour sand is).
##
##   dens   - who displaces whom. The only thing buoyancy is made of.
##   repose - probability a blocked grain tries a diagonal slide. Low = steep, high = flat.
##   spread - how many cells a fluid may search sideways in one tick. Higher = runnier.
##   lift   - susceptibility to the ambient draft (light dust saltates, wet sand does not).
##   fuel   - probability of catching when an igniter is adjacent; `burns` names the result.
##   decay  - per-tick probability of turning into `becomes` ("" = vanish).
##   ignite - this material IS an igniter; `douse` is what quenching it leaves behind.
##   boil   - a liquid that an igniter flashes into this instead (water into steam).
##   emis   - how much of the scene's glow lands on this material's value.
##   grain  - per-cell value jitter, which is the difference between a slab and a texture.
const MATTER := {
	"sand": {"kind": "powder", "dens": [1.55, 1.90], "repose": [0.10, 0.42], "lift": [0.04, 0.14],
		"hue": 0.00, "sat": [0.42, 0.68], "val": [0.60, 0.86], "grain": [0.05, 0.11]},
	"salt": {"kind": "powder", "dens": [1.20, 1.48], "repose": [0.05, 0.24], "lift": [0.05, 0.16],
		"hue": 0.50, "sat": [0.02, 0.11], "val": [0.84, 0.99], "grain": [0.03, 0.08]},
	"dust": {"kind": "powder", "dens": [0.70, 1.00], "repose": [0.45, 0.88], "lift": [0.35, 0.80],
		"fuel": [0.20, 0.55], "burns": "ember",
		"hue": 0.05, "sat": [0.10, 0.26], "val": [0.44, 0.66], "grain": [0.04, 0.10]},
	"ash": {"kind": "powder", "dens": [0.88, 1.15], "repose": [0.55, 0.92], "lift": [0.25, 0.62],
		"hue": 0.62, "sat": [0.01, 0.09], "val": [0.14, 0.30], "grain": [0.04, 0.11]},
	"ember": {"kind": "powder", "dens": [1.05, 1.40], "repose": [0.18, 0.50], "lift": [0.25, 0.60],
		"ignite": [0.55, 1.00], "decay": [0.010, 0.032], "becomes": "ash", "douse": "ash",
		"emis": [0.55, 0.95],
		"hue": 0.02, "sat": [0.72, 0.95], "val": [0.86, 1.00], "grain": [0.06, 0.14]},
	"water": {"kind": "liquid", "dens": [0.98, 1.02], "repose": 1.0, "spread": [3, 6], "lift": [0.0, 0.0],
		"boil": "steam",
		"hue": 0.45, "sat": [0.44, 0.72], "val": [0.52, 0.80], "grain": [0.02, 0.06]},
	"oil": {"kind": "liquid", "dens": [0.72, 0.90], "repose": 1.0, "spread": [2, 4], "lift": [0.0, 0.0],
		"fuel": [0.45, 0.90], "burns": "ember",
		"hue": 0.63, "sat": [0.35, 0.62], "val": [0.26, 0.44], "grain": [0.02, 0.07]},
	"steam": {"kind": "gas", "dens": [0.04, 0.16], "repose": 1.0, "spread": [1, 3], "lift": [0.0, 0.0],
		"decay": [0.012, 0.035], "becomes": "",
		"hue": 0.42, "sat": [0.02, 0.10], "val": [0.72, 0.94], "grain": [0.03, 0.09]},
	"stone": {"kind": "solid", "dens": [8.0, 8.0], "repose": 0.0, "lift": [0.0, 0.0],
		"hue": 0.55, "sat": [0.05, 0.18], "val": [0.20, 0.40], "grain": [0.02, 0.07]},
}

var w := 160
var h := 96

## Per-tick inputs the owner sets before [method step]. They are plain fields rather than
## a snapshot dictionary because the step reads them once and the lookup would otherwise
## sit inside the hot loop.
var tilt := 0.0          # gravity shear, -1..1: probability the "down" cell is offset sideways
var wind := 0.0          # ambient draft strength, 0..1
var wind_dir := 1        # +1 downwind to the right, -1 to the left
var ignite := 0.0        # global scale on every igniter's per-neighbour chance
var spill := false       # open-sided chamber: whatever reaches the edge columns leaves
var wall_id := 0         # the material the chamber is built from (restored when a hatch shuts)
## How many rows deep the floor is, so an open hatch cuts through ALL of it. A hatch that
## only clears the bottom row of a four-row floor opens onto more floor and drains nothing.
var floor_h := 1
## Optional per-row colours for [member wall_id] - the chamber drawn as strata rather than
## as one flat grey. Empty = the wall paints like any other material.
var wall_band := PackedColorArray()

var gen := 0

var _cells := PackedByteArray()
var _moved := PackedByteArray()
var _hot := PackedByteArray()        # rows to sweep this tick
var _hot_next := PackedByteArray()   # rows to sweep next tick
var _nz := PackedFloat32Array()
var _salt := 0

# Material properties, parallel arrays indexed by material id (0 = empty).
var _knd := PackedInt32Array()
var _dns := PackedFloat32Array()
var _rep := PackedFloat32Array()
var _spr := PackedInt32Array()
var _lft := PackedFloat32Array()
var _fue := PackedFloat32Array()
var _dec := PackedFloat32Array()
var _ign := PackedFloat32Array()
var _emi := PackedFloat32Array()
var _hue := PackedFloat32Array()
var _sat := PackedFloat32Array()
var _val := PackedFloat32Array()
var _grn := PackedFloat32Array()
var _brn := PackedInt32Array()       # fuel -> this, when ignited
var _bec := PackedInt32Array()       # decays into this
var _boi := PackedInt32Array()       # liquid flashed by an igniter into this
var _dou := PackedInt32Array()       # igniter quenched by a boiling liquid becomes this
var _link: Array = []                # name links, resolved in finalize()
var _names: Array = []
var _by_name: Dictionary = {}
var _mn := 1                         # material count, including empty
var _pdn := PackedByteArray()        # may material m enter material t moving DOWN?
var _pup := PackedByteArray()        # ... moving UP (the gas half)

var _drain_x0 := 0
var _drain_x1 := -1


func _init(w_: int, h_: int, seed_: int) -> void:
	w = maxi(8, w_)
	h = maxi(8, h_)
	# Masked to 24 bits: the hash below multiplies the cell index and the generation by
	# large constants and adds this, and a full 64-bit session seed would push the sum
	# around the wrap point rather than acting as a small decorrelating offset.
	_salt = seed_ & 0xffffff
	_cells.resize(w * h)
	_moved.resize(w * h)
	_hot.resize(h)
	_hot_next.resize(h)
	_hot.fill(1)
	_hot_next.fill(1)
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_
	_nz.resize(NOISE)
	for i in NOISE:
		_nz[i] = rng.randf()
	# id 0 is empty space: AIR, weightless, and never processed as a cell.
	_push_material(AIR, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		{"burns": "", "becomes": "", "boil": "", "douse": ""}, "empty")


## Add one material by name, SAMPLED from its row of [constant MATTER]. Returns its id
## (0 if the name is unknown, which reads as "empty" and is harmless). Adding the same
## name twice returns the first id, so a scene can declare a dependency without checking.
func add(mname: String, rng: RandomNumberGenerator) -> int:
	if _by_name.has(mname):
		return int(_by_name[mname])
	if not MATTER.has(mname):
		return 0
	var d: Dictionary = MATTER[mname]
	var id := _mn
	_push_material(
		int(KINDS.get(String(d.get("kind", "powder")), POWDER)),
		_rf(d.get("dens", 1.0), rng),
		_rf(d.get("repose", 0.5), rng),
		_ri(d.get("spread", 0), rng),
		_rf(d.get("lift", 0.0), rng),
		_rf(d.get("fuel", 0.0), rng),
		_rf(d.get("decay", 0.0), rng),
		_rf(d.get("ignite", 0.0), rng),
		_rf(d.get("emis", 0.0), rng),
		float(d.get("hue", 0.0)),
		_rf(d.get("sat", 0.5), rng),
		_rf(d.get("val", 0.7), rng),
		_rf(d.get("grain", 0.05), rng),
		{"burns": String(d.get("burns", "")), "becomes": String(d.get("becomes", "")),
			"boil": String(d.get("boil", "")), "douse": String(d.get("douse", ""))},
		mname)
	return id


func _push_material(kind: int, dens: float, repose: float, spread: int, lift: float,
		fuel: float, decay: float, igniter: float, emis: float, hue: float, sat: float,
		val: float, grain: float, link: Dictionary, mname: String) -> void:
	_knd.append(kind)
	_dns.append(dens)
	_rep.append(repose)
	_spr.append(spread)
	_lft.append(lift)
	_fue.append(fuel)
	_dec.append(decay)
	_ign.append(igniter)
	_emi.append(emis)
	_hue.append(hue)
	_sat.append(sat)
	_val.append(val)
	_grn.append(grain)
	_brn.append(0)
	_bec.append(0)
	_boi.append(0)
	_dou.append(0)
	_link.append(link)
	_by_name[mname] = _names.size()
	_names.append(mname)
	_mn = _knd.size()


func id_of(mname: String) -> int:
	return int(_by_name.get(mname, 0))


func name_of(id: int) -> String:
	return String(_names[id]) if id >= 0 and id < _names.size() else "empty"


## Resolve the name links and build the pass tables. Call once, after every material is
## added and before the first [method step].
##
## The pass table is the whole physics of displacement, precomputed: for each ordered pair
## it stores whether m may enter t moving down (t is empty, or t is a fluid strictly less
## dense) and whether it may moving up (t strictly MORE dense - the gas case). One byte
## read replaces a comparison chain in the innermost loop, and buoyancy layering, sinking
## and bubbling all become the same lookup.
func finalize() -> void:
	for id in _mn:
		var link: Dictionary = _link[id]
		_brn[id] = int(_by_name.get(String(link["burns"]), 0))
		_bec[id] = int(_by_name.get(String(link["becomes"]), 0))
		_boi[id] = int(_by_name.get(String(link["boil"]), 0))
		_dou[id] = int(_by_name.get(String(link["douse"]), 0))
		# A fuel with nothing to burn INTO cannot catch - the scene left the ember out.
		if _brn[id] == 0:
			_fue[id] = 0.0
	_pdn.resize(_mn * _mn)
	_pup.resize(_mn * _mn)
	for m in _mn:
		for t in _mn:
			var dn := 0
			var up := 0
			if _knd[m] != SOLID and _knd[m] != AIR:
				if t == 0:
					dn = 1
					up = 1
				elif _knd[t] != SOLID:
					if _dns[t] < _dns[m] - 0.01:
						dn = 1
					if _dns[t] > _dns[m] + 0.01:
						up = 1
			_pdn[m * _mn + t] = dn
			_pup[m * _mn + t] = up


func at(x: int, y: int) -> int:
	if x < 0 or y < 0 or x >= w or y >= h:
		return 0
	return int(_cells[y * w + x])


func set_cell(x: int, y: int, m: int) -> void:
	if x < 0 or y < 0 or x >= w or y >= h:
		return
	_cells[y * w + x] = m
	_touch(y)


## Fill an inclusive rectangle, clamped to the grid. The chamber is built out of these.
func fill_rect(x0: int, y0: int, x1: int, y1: int, m: int) -> void:
	var ax := maxi(0, mini(x0, x1))
	var bx := mini(w - 1, maxi(x0, x1))
	var ay := maxi(0, mini(y0, y1))
	var by := mini(h - 1, maxi(y0, y1))
	for y in range(ay, by + 1):
		var base := y * w
		for x in range(ax, bx + 1):
			_cells[base + x] = m
		_touch(y)


## Drop [param count] cells of [param m] into the span, at row [param y]. The placement is
## a pure function of (generation, salt, ordinal) - never an rng draw - so a spout whose
## RATE rides on the audio still places its grains identically in a live session and in an
## export, and only the count differs.
func emit(x0: int, x1: int, y: int, m: int, count: int, salt: int) -> void:
	if m <= 0 or count <= 0 or y < 0 or y >= h:
		return
	var lo := maxi(0, mini(x0, x1))
	var hi := mini(w - 1, maxi(x0, x1))
	if hi < lo:
		return
	var span := hi - lo + 1
	var base := y * w
	for k in count:
		var hsh := gen * 2654435761 + salt * 40503 + k * 2246822519 + _salt
		var x := lo + int(_nz[hsh & NMASK] * float(span))
		if x > hi:
			x = hi
		if int(_cells[base + x]) == 0:
			_cells[base + x] = m
			_touch(y)


## Open a floor hatch across an inclusive column span (x1 < x0 shuts it). Shutting restores
## the floor out of [member wall_id], so the chamber is whole again and the next pile builds
## on it - the hatch is a door, not a permanent hole.
func set_drain(x0: int, x1: int) -> void:
	if x0 == _drain_x0 and x1 == _drain_x1:
		return
	if _drain_x1 >= _drain_x0 and wall_id > 0:
		for y in range(maxi(0, h - maxi(1, floor_h)), h):
			var by := y * w
			for x in range(maxi(0, _drain_x0), mini(w - 1, _drain_x1) + 1):
				_cells[by + x] = wall_id
			_touch(y)
	_drain_x0 = x0
	_drain_x1 = x1


func draining() -> bool:
	return _drain_x1 >= _drain_x0


## One generation. Bottom-up, in place, one move per cell at most.
func step() -> void:
	gen += 1
	_moved.fill(0)
	var full := (gen % FULL_EVERY) == 0
	var flip := 1 if (gen & 1) == 0 else -1
	var tl := absf(tilt)
	var tdir := 1 if tilt > 0.0 else -1
	var wnd := clampf(wind, 0.0, 1.0)
	var wdir := 1 if wind_dir >= 0 else -1
	for y in range(h - 1, -1, -1):
		if not full and _hot[y] == 0:
			continue
		var rb := flip if (y & 1) == 0 else -flip
		var base := y * w
		var down := y < h - 1
		var up := y > 0
		for k in w:
			var x := k if rb > 0 else (w - 1 - k)
			var i := base + x
			var m := int(_cells[i])
			if m == 0 or _moved[i] != 0:
				continue
			var kind := int(_knd[m])
			if kind == SOLID:
				continue
			# Three decorrelated randoms for this cell this tick, from a pure hash - see the
			# determinism note at the top of the file. No rng draw is consumed anywhere here.
			var hsh := i * 1103515245 + gen * 2654435761 + _salt
			var r0 := _nz[hsh & NMASK]
			var r1 := _nz[(hsh >> 12) & NMASK]
			var r2 := _nz[(hsh >> 21) & NMASK]

			# --- transmutation ------------------------------------------------------
			if _dec[m] > 0.0 and r0 < _dec[m]:
				var to := int(_bec[m])
				_cells[i] = to
				_touch(y)
				if to == 0:
					continue
				m = to
				kind = int(_knd[m])
				if kind == SOLID:
					continue
			if _ign[m] > 0.0 and ignite > 0.0:
				var p := _ign[m] * ignite
				var quench := false
				if x > 0 and _burn(i - 1, p, r1):
					quench = true
				if x < w - 1 and _burn(i + 1, p, r1):
					quench = true
				if up and _burn(i - w, p, r2):
					quench = true
				if down and _burn(i + w, p, r2):
					quench = true
				if quench:
					_cells[i] = int(_dou[m])
					_touch(y)
					continue

			# --- movement -----------------------------------------------------------
			var rising := kind == GAS
			var vs := -w if rising else w
			var vy := (y - 1) if rising else (y + 1)
			if vy < 0 or vy >= h:
				continue
			var pt := m * _mn
			var moved_it := false
			# Straight along gravity, which the tilt may shear one cell sideways.
			var dx := 0
			if tl > 0.001 and r1 < tl:
				dx = tdir
			if x + dx >= 0 and x + dx < w:
				var vi := i + vs + dx
				var tm := int(_cells[vi])
				if (_pup[pt + tm] if rising else _pdn[pt + tm]) != 0:
					_swap(i, vi, m, tm)
					_touch(y)
					moved_it = true
			# Diagonally, gated by the material's angle of repose.
			if not moved_it and r2 < _rep[m]:
				for s in 2:
					var dd := rb if s == 0 else -rb
					var sx := x + dd
					if sx < 0 or sx >= w:
						continue
					# Never squeeze through the diagonal joint of a wall - without this,
					# powder leaks through every corner of the chamber.
					if int(_knd[int(_cells[i + dd])]) == SOLID:
						continue
					var di := i + vs + dd
					var tm2 := int(_cells[di])
					if (_pup[pt + tm2] if rising else _pdn[pt + tm2]) != 0:
						_swap(i, di, m, tm2)
						_touch(y)
						moved_it = true
						break
			# Sideways, for fluids only: the search that makes a liquid find its level.
			if not moved_it and _spr[m] > 0:
				var sp := int(_spr[m])
				for s in 2:
					var dd := rb if s == 0 else -rb
					var best := 0
					for step_k in range(1, sp + 1):
						var sx := x + dd * step_k
						if sx < 0 or sx >= w:
							break
						var ti := i + dd * step_k
						if (_pup[pt + int(_cells[ti])] if rising else _pdn[pt + int(_cells[ti])]) == 0:
							break
						best = step_k
						# A hole under (or over) the path is worth more than distance.
						var hi2 := ti + vs
						if (_pup[pt + int(_cells[hi2])] if rising else _pdn[pt + int(_cells[hi2])]) != 0:
							break
					if best != 0:
						var li := i + dd * best
						_swap(i, li, m, int(_cells[li]))
						_touch(y)
						moved_it = true
						break
			# The ambient draft, with its lee shadow. Powder only, and only where the grain
			# is actually exposed to the air - a buried grain feels no wind.
			if not moved_it and wnd > 0.001 and kind == POWDER and _lft[m] > 0.0 and up:
				if int(_cells[i - w]) == 0:
					var lx := x - wdir
					var sheltered := lx >= 0 and lx < w and int(_cells[(y - 1) * w + lx]) != 0
					if not sheltered and r0 < wnd * _lft[m]:
						var hx := x + wdir
						if hx >= 0 and hx < w:
							var hi3 := i + wdir
							var tm3 := int(_cells[hi3])
							if tm3 == 0 or _pdn[pt + tm3] != 0:
								_swap(i, hi3, m, tm3)
								_touch(y)
	# The hatch: anything that has reached the floor inside an open span leaves the world.
	# The floor material goes with it, which is what OPENS the hatch in the first place.
	if _drain_x1 >= _drain_x0:
		for y in range(maxi(0, h - maxi(1, floor_h)), h):
			var by := y * w
			for x in range(maxi(0, _drain_x0), mini(w - 1, _drain_x1) + 1):
				if int(_cells[by + x]) != 0:
					_cells[by + x] = 0
					_touch(y)
	# An open-sided chamber loses whatever walks off the edge.
	if spill:
		for y in h:
			var b := y * w
			if int(_cells[b]) != 0:
				_cells[b] = 0
				_touch(y)
			if int(_cells[b + w - 1]) != 0:
				_cells[b + w - 1] = 0
				_touch(y)
	var tmp := _hot
	_hot = _hot_next
	_hot_next = tmp
	_hot_next.fill(0)


# Move m from i into j, carrying whatever was there back. Both ends are marked moved when
# it is a genuine swap, so a pair of materials cannot thrash back and forth within one tick.
func _swap(i: int, j: int, m: int, tm: int) -> void:
	_cells[j] = m
	_cells[i] = tm
	_moved[j] = 1
	if tm != 0:
		_moved[i] = 1
	_touch(j / w)


# One neighbour of an igniter. Returns true if the igniter is quenched by it.
func _burn(n: int, p: float, r: float) -> bool:
	var nm := int(_cells[n])
	if nm == 0:
		return false
	if _fue[nm] > 0.0 and r < _fue[nm] * p:
		_cells[n] = int(_brn[nm])
		_touch(n / w)
		return false
	if _boi[nm] > 0 and r < 0.35:
		_cells[n] = int(_boi[nm])
		_touch(n / w)
		return true
	return false


# Wake a row and its two neighbours, for this tick AND the next. Writing both arrays is
# what lets emissions and hatch changes made BETWEEN steps take effect immediately.
func _touch(y: int) -> void:
	_hot[y] = 1
	_hot_next[y] = 1
	if y > 0:
		_hot[y - 1] = 1
		_hot_next[y - 1] = 1
	if y < h - 1:
		_hot[y + 1] = 1
		_hot_next[y + 1] = 1


## Emit the grid into [param tb] as RUN-LENGTH quads: one quad per horizontal stretch of
## identical material, capped at [param run_cap] cells.
##
## The cap is not a performance limit, it is the texture: an uncapped run paints a whole
## settled layer in one flat colour, and a pile of sand that is one flat colour does not
## read as sand. Capping the run gives every few cells their own value jitter while still
## collapsing a full chamber from ~25000 cells to a few thousand quads - one draw call
## either way, but the picture has grain in it.
##
## A run also breaks where the cell above changes between solid and empty, so the exposed
## surface of a pile gets its own slightly brighter run - a lit crust along the silhouette,
## which is what tells the eye where the top of the material is.
func paint(tb: TriBatch, org: Vector2, cell: float, run_cap: int, hue_rot: float,
		glow: float, depth: float) -> void:
	var cap := maxi(1, run_cap)
	var banded := wall_band.size() == h
	var over := cell + 0.5           # a half pixel of overlap, so no seams between rows
	var inv_h := 1.0 / float(h)
	for y in h:
		var base := y * w
		var py := org.y + float(y) * cell
		var shade := 1.0 - depth * float(y) * inv_h
		var x := 0
		while x < w:
			var m := int(_cells[base + x])
			if m == 0:
				x += 1
				continue
			var exposed := 1 if (y == 0 or int(_cells[base + x - w]) == 0) else 0
			var run := 1
			while run < cap and x + run < w:
				var j := base + x + run
				if int(_cells[j]) != m:
					break
				var e2 := 1 if (y == 0 or int(_cells[j - w]) == 0) else 0
				if e2 != exposed:
					break
				run += 1
			var col: Color
			if banded and m == wall_id:
				col = wall_band[y]
			else:
				var hsh := (base + x) * 1103515245 + _salt
				var jit := (_nz[hsh & NMASK] - 0.5) * _grn[m]
				var v := _val[m] * shade + jit
				var s := _sat[m]
				if _emi[m] > 0.0:
					var hot := glow * _emi[m]
					v += hot * 0.55
					s -= hot * 0.35
				if exposed == 1:
					v += 0.055
				col = Color.from_hsv(fposmod(_hue[m] + hue_rot, 1.0),
					clampf(s, 0.0, 1.0), clampf(v, 0.0, 1.0))
			var px := org.x + float(x) * cell
			var rw := float(run) * cell + 0.5
			tb.quad(Vector2(px, py), Vector2(px + rw, py),
				Vector2(px + rw, py + over), Vector2(px, py + over), col)
			x += run


## How many cells are not empty - the "how full is the chamber" number a scene reports.
func occupancy() -> float:
	var n := 0
	for i in _cells.size():
		if int(_cells[i]) != 0:
			n += 1
	return float(n) / float(maxi(1, _cells.size()))


static func _rf(v, rng: RandomNumberGenerator) -> float:
	if v is Array:
		var a: Array = v
		if a.size() >= 2:
			return rng.randf_range(float(a[0]), float(a[1]))
		return float(a[0]) if a.size() == 1 else 0.0
	return float(v)


static func _ri(v, rng: RandomNumberGenerator) -> int:
	if v is Array:
		var a: Array = v
		if a.size() >= 2:
			return rng.randi_range(int(a[0]), int(a[1]))
		return int(a[0]) if a.size() == 1 else 0
	return int(v)
