extends RefCounted
class_name Filament

## Filament - an organic growing path: root, tendril, lightning, or thread.
##
## The procedural-growth primitive the visualizer was missing. One mechanism, many
## lives: a path is grown segment by segment by *following a [Flow2D] field* (so it
## meanders and curls), turning toward that flow through a nonlinearity (so the turn
## is smooth but decisive, not a uniform arc), branching stochastically, tapering as
## it goes, and carrying a per-segment *birth time* so it can be revealed along a
## growth front - it crawls into being rather than popping in. The `variant` sets
## the character: roots that spread and droop, tendrils that coil, lightning that
## kinks and forks, threads that flow. Built once from a seed; drawn growing.
##
## All coordinates are ghost's centred unit-fraction space; widths are pixels.
## Compose it: a scene grows a handful of filaments and draws them, colour and
## growth driven by audio. See `rooted_growth` and `filaments`.

## variant -> shaping config. flow_follow: how hard it turns toward the flow.
## branch: per-step fork chance. spread: fork half-angle. jitter: per-step wobble.
## kink: occasional sharp deflection (lightning). taper: width kept per segment.
## bias_ang/bias_amt: a pull toward a fixed heading (gravity / upward reach).
const VARIANTS := {
	"root":      {"flow_follow": 0.45, "branch": 0.30, "max_depth": 4, "ratio": 0.74,
		"spread": 0.62, "jitter": 0.05, "kink": 0.0, "taper": 0.90, "bias_ang": 1.5708, "bias_amt": 0.14},
	"tendril":   {"flow_follow": 0.80, "branch": 0.16, "max_depth": 3, "ratio": 0.80,
		"spread": 0.70, "jitter": 0.03, "kink": 0.0, "taper": 0.93, "bias_ang": -1.5708, "bias_amt": 0.05},
	"lightning": {"flow_follow": 0.22, "branch": 0.34, "max_depth": 4, "ratio": 0.70,
		"spread": 0.60, "jitter": 0.30, "kink": 0.55, "taper": 0.82, "bias_ang": 1.5708,
		"bias_amt": 0.10, "tendril": 0.26},
	"thread":    {"flow_follow": 0.92, "branch": 0.05, "max_depth": 2, "ratio": 0.85,
		"spread": 0.50, "jitter": 0.01, "kink": 0.0, "taper": 0.96, "bias_ang": 0.0, "bias_amt": 0.0},
	# fur: long, mostly-unbranched strands that hold a LEAN (bias_amt is meaningfully
	# strong, reasserted every step) rather than wandering freely - the flow field
	# adds a gentle organic curl on top, it doesn't dominate the direction the way
	# root/tendril growth does. bias_ang is a per-call override in practice (see
	# scenes/furry.gd) - every tuft leans its own way, not one fixed heading.
	"fur":       {"flow_follow": 0.22, "branch": 0.05, "max_depth": 1, "ratio": 0.85,
		"spread": 0.55, "jitter": 0.05, "kink": 0.0, "taper": 0.965, "bias_ang": 0.0, "bias_amt": 0.11},
}

const MAX_SEGS := 3000        # safety cap against a pathological branch explosion

## Each segment: a, b (unit-fraction endpoints), w0, w1 (px widths), born0, born1
## (0..1 birth times along the longest root-to-tip path), depth.
var segs: Array = []
var _total := 0.0             # max arclength to any tip (normaliser for born times)
var _max_depth := 0           # deepest branch level (so trunk vs tip can be told apart)


## The deepest branch level reached (0 = an unbranched trunk). Lets a scene fade or colour
## a filament by depth - tips (high depth) versus the main channel (depth 0).
func max_depth() -> int:
	return _max_depth


## Grow a filament from [param origin] heading [param heading] (radians), of about
## [param length] over [param steps] segments, starting [param width] px wide,
## following [param flow]. Seeded by [param rng]. [param cluster] (0..1) makes the growth
## *chunky*: each forked branch's reveal time is delayed past its parent's, so the shape
## accumulates region by region in chunks (granular within a branch) instead of fanning out
## all at once. 0 = the old smooth, uniform growth front. [param bias_ang_override], if not
## NAN, replaces the variant's own fixed bias_ang - a per-CALL pull direction rather than a
## per-variant constant, for callers whose bias depends on where they're growing FROM (e.g.
## fur leaning toward a scene's own bright spots - see scenes/furry.gd).
static func grow(variant: String, origin: Vector2, heading: float, length: float,
		width: float, steps: int, flow: Flow2D, rng: RandomNumberGenerator, cluster := 0.0,
		bias_ang_override := NAN) -> Filament:
	var f := Filament.new()
	var cfg: Dictionary = (VARIANTS.get(variant, VARIANTS["root"]) as Dictionary).duplicate()
	cfg["cluster"] = cluster
	if not is_nan(bias_ang_override):
		cfg["bias_ang"] = bias_ang_override
	f._build(origin, heading, length / float(maxi(1, steps)), width, 0, steps, cfg, flow, rng, 0.0)
	if f._total > 0.0:
		for s in f.segs:
			s.born0 /= f._total
			s.born1 /= f._total
	return f


func _build(p: Vector2, ang: float, seg_len: float, width: float, depth: int,
		steps: int, cfg: Dictionary, flow: Flow2D, rng: RandomNumberGenerator, born: float) -> void:
	var w := width
	for s in steps:
		if segs.size() >= MAX_SEGS:
			return
		# Turn toward the flow (nonlinear follow), then a gentle pull toward the
		# variant's bias heading, then per-step wobble and the occasional sharp kink.
		var fdir := flow.angle_at(p, ang)
		ang = lerp_angle(ang, fdir, float(cfg.flow_follow))
		if float(cfg.bias_amt) > 0.0:
			ang = lerp_angle(ang, float(cfg.bias_ang), float(cfg.bias_amt))
		ang += rng.randf_range(-1.0, 1.0) * float(cfg.jitter)
		if float(cfg.kink) > 0.0 and rng.randf() < 0.5:
			ang += rng.randf_range(-1.0, 1.0) * float(cfg.kink)

		var np := p + Vector2(cos(ang), sin(ang)) * seg_len
		var w1 := w * float(cfg.taper)
		var born1 := born + seg_len
		segs.append({"a": p, "b": np, "w0": w, "w1": w1,
			"born0": born, "born1": born1, "depth": depth})
		_total = maxf(_total, born1)
		_max_depth = maxi(_max_depth, depth)
		p = np
		w = w1
		born = born1

		# Fork: a child continues from here, shorter, thinner, one depth deeper.
		if depth < int(cfg.max_depth) and rng.randf() < float(cfg.branch):
			var side := 1.0 if rng.randf() < 0.5 else -1.0
			var child_len := seg_len * float(cfg.ratio)
			var child_steps := maxi(2, int(float(steps) * float(cfg.ratio)))
			# Clustered growth: delay the child branch's birth time so it reveals as a later
			# CHUNK, after this run, rather than fanning out in lockstep with the trunk.
			var gap := seg_len * float(cfg.get("cluster", 0.0)) * rng.randf_range(1.5, 4.5)
			_build(p, ang + side * float(cfg.spread), child_len, w * 0.7,
				depth + 1, child_steps, cfg, flow, rng, born + gap)

			# Tiny tendril: a brief hair-thin offshoot at a sharp angle that does NOT keep
			# growing - the fine fuzz that real lightning frays into along its length.
			if float(cfg.get("tendril", 0.0)) > 0.0 and rng.randf() < float(cfg.tendril):
				var tside := 1.0 if rng.randf() < 0.5 else -1.0
				_stub(p, ang + tside * float(cfg.spread) * 1.7, seg_len * 0.5, w * 0.5,
					born + gap, int(cfg.max_depth), rng)


# A short dead-end offshoot (1-3 kinky segments, no recursion) - a "tendril".
func _stub(p: Vector2, ang: float, seg_len: float, width: float, born: float,
		depth: int, rng: RandomNumberGenerator) -> void:
	var w := width
	for s in rng.randi_range(1, 3):
		if segs.size() >= MAX_SEGS:
			return
		ang += rng.randf_range(-1.0, 1.0) * 0.5
		var np := p + Vector2(cos(ang), sin(ang)) * seg_len
		var w1 := w * 0.7
		var born1 := born + seg_len
		segs.append({"a": p, "b": np, "w0": w, "w1": w1,
			"born0": born, "born1": born1, "depth": depth})
		_total = maxf(_total, born1)
		_max_depth = maxi(_max_depth, depth)
		p = np
		w = w1
		born = born1


## Draw the filament revealed up to growth front [param grown] (0..1). `u` is the
## pixel unit. `color_for` is a Callable(depth:int) -> Color so the scene owns the
## palette. A bud glows at the live tip while it grows.
##
## `jitter` (unit-fraction) + `t` (time) give the timelapse twitch the growth wants:
## the trunk (low depth) is rock-steady while the young tips (high depth) tremble,
## and segments right at the advancing front shake hardest - so it reads as living
## growth, not a static drawing, with stable trunks and unstable new shoots.
func draw_growing(ci: CanvasItem, u: float, grown: float, color_for: Callable,
		tip: Color = Color(1, 1, 1, 0.9), jitter := 0.0, t := 0.0) -> void:
	var maxd := float(maxi(1, _max_depth))
	var i := 0
	for s in segs:
		i += 1
		if s.born0 > grown:
			continue                                  # not yet reached by the front
		var frac := 1.0
		if s.born1 > grown:
			frac = clampf((grown - s.born0) / maxf(1e-6, s.born1 - s.born0), 0.0, 1.0)
		var jo := Vector2.ZERO
		if jitter > 0.0:
			# Stable trunk -> twitchy tip (depth), strongest near the live front.
			var depth_f := float(s.depth) / maxd
			var front_f := clampf(1.0 - (grown - s.born1) / 0.18, 0.0, 1.0)
			var amt := jitter * depth_f * (0.25 + 0.75 * front_f)
			var ph := float(i) * 1.7
			jo = Vector2(sin(t * 5.0 + ph), cos(t * 4.3 + ph * 1.3)) * amt
		var a: Vector2 = (s.a + jo) * u
		var b: Vector2 = ((s.a as Vector2).lerp(s.b, frac) + jo) * u
		var w: float = maxf(0.6, lerpf(s.w0, s.w1, frac))
		# Pass the segment's position along the path (born0, 0 base .. 1 tip) so the scene can
		# shade a gradient / texture along the strand, not just a flat colour per depth.
		ci.draw_line(a, b, color_for.call(int(s.depth), float(s.born0)), w, true)
		if frac > 0.0 and frac < 1.0:                 # the live, advancing tip
			ci.draw_circle(b, w * 1.3, tip)
