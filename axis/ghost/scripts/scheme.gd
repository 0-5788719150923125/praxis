extends RefCounted
class_name Scheme

## Scheme - a scene's colour identity, drawn from a named mood.
##
## The companion to [Palette], and the distinction matters: a Palette is a RAMP
## sampled by a scalar (elevation to colour), while a Scheme is a small set of
## RELATED HUES a scene builds its whole look from. Terrain needed the first;
## everything else needed the second and did not have it.
##
## WHY. 42 of ghost's 45 scenes hardcoded their own hue, usually as a range a
## few hundredths wide - `rng.randf_range(0.04, 0.10)` and the like - so a scene
## was locked to one colour forever however many times it was seeded. The
## clouds were always the same dawn orange, the embers always the same red, the
## metropolis always brass. Each was a reasonable choice made once and then
## frozen, and there was no shared thing to choose differently WITH.
##
## HOW. A mood carries a base hue, a related accent, how far individual elements
## may drift from the base, and the saturation/value character that makes the
## mood read as itself - "ash" is not just grey, it is desaturated AND dark AND
## narrow. Scenes ask for a mood and then ask it for colours, so adding a mood
## here varies every scene that uses one, and a new scene inherits the whole set
## for free. That is the point: the flexibility lives in the primitive.
##
## Relationships are named rather than random, because random hue pairs clash.
## `accent` is chosen per mood as a complement, a near-neighbour or a triad
## partner, so a scene using base + accent is harmonious by construction.

# hue, accent, spread, sat, val
const MOODS := {
	"dawn":      {"hue": 0.06, "accent": 0.11, "spread": 0.05, "sat": 0.72, "val": 0.92},
	"ember":     {"hue": 0.02, "accent": 0.09, "spread": 0.04, "sat": 0.85, "val": 0.85},
	"sodium":    {"hue": 0.09, "accent": 0.05, "spread": 0.03, "sat": 0.80, "val": 0.95},
	"brass":     {"hue": 0.11, "accent": 0.55, "spread": 0.05, "sat": 0.62, "val": 0.88},
	"verdant":   {"hue": 0.33, "accent": 0.16, "spread": 0.07, "sat": 0.58, "val": 0.78},
	"teal":      {"hue": 0.47, "accent": 0.08, "spread": 0.05, "sat": 0.60, "val": 0.82},
	"glacier":   {"hue": 0.55, "accent": 0.60, "spread": 0.05, "sat": 0.42, "val": 0.95},
	"abyss":     {"hue": 0.62, "accent": 0.50, "spread": 0.05, "sat": 0.70, "val": 0.55},
	"violet":    {"hue": 0.74, "accent": 0.90, "spread": 0.06, "sat": 0.58, "val": 0.80},
	"magenta":   {"hue": 0.90, "accent": 0.45, "spread": 0.05, "sat": 0.70, "val": 0.88},
	"ash":       {"hue": 0.60, "accent": 0.08, "spread": 0.03, "sat": 0.12, "val": 0.62},
	"bone":      {"hue": 0.10, "accent": 0.58, "spread": 0.03, "sat": 0.16, "val": 0.92},
	"toxic":     {"hue": 0.22, "accent": 0.78, "spread": 0.06, "sat": 0.85, "val": 0.92},
	"rose":      {"hue": 0.95, "accent": 0.12, "spread": 0.04, "sat": 0.45, "val": 0.90},
}

var name := "dawn"
var hue := 0.06        # the base
var accent := 0.11     # a harmonious second hue, per mood
var spread := 0.05     # how far individual elements may drift from the base
var sat := 0.7
var val := 0.9


## Pick a mood at random. Deterministic for a given rng, like everything else in
## a scene's build.
static func pick(rng: RandomNumberGenerator) -> Scheme:
	var keys := MOODS.keys()
	return of(String(keys[rng.randi() % keys.size()]), rng)


## Pick from a NAMED SUBSET, for scenes where only some moods make sense - fire
## should not be glacier-blue. Falls back to the full set if none match.
static func among(names: Array, rng: RandomNumberGenerator) -> Scheme:
	var valid: Array = []
	for n in names:
		if MOODS.has(String(n)):
			valid.append(String(n))
	if valid.is_empty():
		return pick(rng)
	return of(String(valid[rng.randi() % valid.size()]), rng)


static func of(mood: String, rng: RandomNumberGenerator) -> Scheme:
	var s := Scheme.new()
	var m: Dictionary = MOODS.get(mood, MOODS["dawn"])
	s.name = mood
	# a small jitter so two scenes on the same mood are not identical twins
	s.hue = fposmod(float(m["hue"]) + rng.randf_range(-0.02, 0.02), 1.0)
	s.accent = fposmod(float(m["accent"]) + rng.randf_range(-0.02, 0.02), 1.0)
	s.spread = float(m["spread"])
	s.sat = float(m["sat"])
	s.val = float(m["val"])
	return s


## The base hue, drifted by up to `spread` - for per-element variation within
## one scene, so a hundred embers are a family rather than a hundred clones.
func vary(rng: RandomNumberGenerator, amount := 1.0) -> float:
	return fposmod(hue + rng.randf_range(-spread, spread) * amount, 1.0)


## A hue by index: 0 base, 1 accent, and beyond that evenly spaced between the
## two. Lets a scene ask for "three related hues" without inventing a scheme.
func hue_at(i: int, n := 2) -> float:
	if i <= 0:
		return hue
	if n <= 2:
		return accent          # only two hues asked for: base and accent
	if i >= n - 1:
		return accent          # the last of a series IS the accent
	var d := fposmod(accent - hue + 0.5, 1.0) - 0.5      # shortest way round
	return fposmod(hue + d * float(i) / float(n - 1), 1.0)


## A colour on this scheme. `s_mul`/`v_mul` scale the mood's own character, so a
## caller asks for "a bit darker" rather than restating absolute numbers - which
## is what kept the moods from reading as themselves before.
func color(h: float, s_mul := 1.0, v_mul := 1.0, a := 1.0) -> Color:
	return Color.from_hsv(fposmod(h, 1.0), clampf(sat * s_mul, 0.0, 1.0),
		clampf(val * v_mul, 0.0, 1.0), a)


## The base colour, and the accent colour. The two calls a scene makes most.
func base(s_mul := 1.0, v_mul := 1.0, a := 1.0) -> Color:
	return color(hue, s_mul, v_mul, a)


func accent_color(s_mul := 1.0, v_mul := 1.0, a := 1.0) -> Color:
	return color(accent, s_mul, v_mul, a)


## A second hue guaranteed to READ as different from `lead`.
##
## The accent alone is not enough for a scene that puts two colours side by side.
## Several moods keep their accent within a few hundredths of the base (sodium's
## sits 0.04 away), and `lead` is usually a DRIFTED base rather than the nominal
## one, so the gap can close further. Two prisms measured 0.104 turns apart -
## near enough to read as one colour and lose the whole point of the pair.
##
## So: prefer the mood's own accent, because it was chosen to be harmonious, and
## only push out to `min_sep` when it is too close - and push it along the
## direction the mood already leans, so verdant still opens toward amber and
## glacier toward violet rather than collapsing to a generic complement.
##
## Callers that stack further per-element variation on top should ask for more
## separation than they need, since that cast eats into the gap.
func opposed(lead: float, min_sep := 0.24) -> float:
	var d := fposmod(accent - lead + 0.5, 1.0) - 0.5      # signed, shortest way round
	if absf(d) >= min_sep:
		return accent
	return fposmod(lead + (min_sep if d >= 0.0 else -min_sep), 1.0)
