extends RefCounted
class_name Filters

## Filters - the LOOK the whole picture is put through, after the scenes have drawn.
##
## THE SEAM WAS ALREADY THERE. Since the stage governor landed, scenes do not draw into the
## root viewport: they draw into ONE [SubViewport] that main presents through a single
## [TextureRect] at child index 0, with every panel and the karaoke overlay stacked above it
## on their own CanvasLayers. So a material on that one rect is a post-process over the
## entire show and over nothing else - it costs no render target, no second pass and no
## change to a single scene, and the UI stays legible because it is not underneath it.
##
## THAT IS ALSO THE LIMIT, and it is a deliberate one: subtitles are NOT filtered. Grain and
## halftone dots over a karaoke line make it hard to read, and the line is not part of the
## photograph - it is a caption on top of one.
##
## FILTERS COMBINE, which is the reason this is a registry of amounts and not a picker. Being
## asked to choose between monochrome and grain is the wrong question - black and white film
## HAS grain - so every entry has its own 0..1 dial and they are all applied, in one fragment
## shader, in one pass. A filter at 0 is arithmetically a no-op, and when every filter is at 0
## the material is removed from the rect entirely, so the default show is not merely
## unchanged but is running the code it always ran.
##
## THE REGISTRY ORDER IS THE PIPELINE ORDER, top to bottom, and it is the one thing here that
## cannot be read off the shader: dots are laid first (they RE-SAMPLE the picture, so anything
## after them grades the dots rather than the photograph), then the optics that a lens would
## add, then the grade, then the emulsion, then the frame. Monochrome before Noir is what
## makes the pair read as black-and-white noir rather than as a tinted colour picture - Noir's
## split-tone is applied to whatever it is handed.
##
## ADDING A FILTER is an entry in each of the four tables below plus a `u_<key>` uniform and a
## block in [code]shaders/stage_filter.gdshader[/code]. Nothing in main, the Director or the
## panel changes - the controls are built off [constant Filters.REGISTRY]. A key with no matching
## uniform is the failure this arrangement is exposed to, because
## `set_shader_parameter` on a name the shader does not declare is a SILENT no-op; the gate
## reads the shader source and fails on it.

const SHADER := "res://shaders/stage_filter.gdshader"

## The registry: filter key -> the shader uniform its amount drives. In PIPELINE ORDER.
##
## Keys are what `[director] filters` in `user://ghost.cfg` stores and what
## `--filter KEY=AMOUNT` takes.
const REGISTRY := {
	"slip": "u_slip",
	"pointillism": "u_pointillism",
	"bloom": "u_bloom",
	"monochrome": "u_monochrome",
	"noir": "u_noir",
	"static": "u_static",
	"dust": "u_dust",
	"vignette": "u_vignette",
}

## Display names for the registry keys, in registry order - for the settings surface.
const LABELS := {
	"slip": "Gate slip",
	"pointillism": "Pointillism",
	"bloom": "Bloom",
	"monochrome": "Monochrome",
	"noir": "Film noir",
	"static": "Static",
	"dust": "Dust & scratches",
	"vignette": "Vignette",
}

## One line each, for the checkbox tooltips and docs/filters.md. Deliberately ONE string
## literal per entry, not a `+` continuation: docs.py reads this table with a regex and a
## continuation silently truncates the blurb at the first line.
const BLURBS := {
	"slip": "Tearing: now and then a band of the picture jumps sideways for one frame and snaps back, the way a frame that does not seat squarely in a projector's gate is printed offset. The dial is how often and how far.",
	"pointillism": "The picture re-laid as overlapping dots of paint on a jittered lattice, each dot taking its colour from where it sits and its size from how bright that is. The dial is the size of the dots, so a little is a canvas texture and a lot is a painting you have to stand back from.",
	"bloom": "Light bleeding out of the bright parts, the way it does through a lens. Only what is already brighter than the picture's own highlights blooms, so it lifts lamps, sparks and speculars without fogging the whole frame.",
	"monochrome": "Colour taken out, weighted the way the eye weighs it rather than by averaging the channels - so a red and a green of the same brightness do not come out as the same grey.",
	"noir": "The hard grade: contrast pushed until the blacks close up and the highlights burn, with cold shadows against warm highlights. It grades whatever it is handed, so put Monochrome above it for black-and-white and leave it off for something closer to Technicolor.",
	"static": "Fine noise over the picture, strongest in the midtones and nearly absent in the blacks and the blown highlights, where the emulsion's own noise lives, and it moves at a film's frame rate.",
	"dust": "Dirt on the print: specks of dust (mostly dark, some light), the odd curled hair, and now and then a scratch down the frame that holds for a moment and wanders. It changes every film frame, the way dirt on a moving print does.",
	"vignette": "The corners fallen off, as a fast lens does wide open. Measured on the frame's own diagonal, so it is the same shape of falloff on a phone clip as on a widescreen render.",
}

## Where a filter's dial lands the first time it is switched on. Not 1.0: the top of every
## range here is deliberately too much (a painting, a fogged lens, a blizzard of grain), so
## an unfamiliar filter that arrived at full strength would read as broken rather than as a
## look. Picking one up gives you a usable version of it and the dial does the rest.
const DEFAULTS := {
	"slip": 0.45,
	"pointillism": 0.35,
	"bloom": 0.45,
	"monochrome": 1.0,
	"noir": 0.55,
	"static": 0.40,
	"dust": 0.45,
	"vignette": 0.45,
}


## Every registry key, in pipeline order.
static func keys() -> Array:
	return REGISTRY.keys()


## A stored dictionary cleaned up: known keys only, every amount a float in 0..1, and
## anything at zero DROPPED so "is anything on" is just `is_empty`.
##
## Everything that reads a filter set goes through here, so a key removed from the registry
## in a later build degrades to "that filter is off" rather than to a null uniform write.
static func sanitize(raw: Variant) -> Dictionary:
	var out := {}
	if not (raw is Dictionary):
		return out
	# "grain" is this filter's old name (it looked like static, so it is called that now); a
	# setting or a chapter from before keeps its amount rather than silently losing it
	if (raw as Dictionary).has("grain") and not (raw as Dictionary).has("static"):
		raw = (raw as Dictionary).duplicate()
		raw["static"] = raw["grain"]
	for k in REGISTRY:
		if not (raw as Dictionary).has(k):
			continue
		var v := clampf(float((raw as Dictionary)[k]), 0.0, 1.0)
		if v > 0.0:
			out[k] = v
	return out


## PUT THE LOOK ON THE PICTURE. [param rect] is main's stage view, [param amounts] a
## sanitized set, [param size] the rect's pixel size.
##
## NOTHING ON MEANS NO MATERIAL, not a material with every dial at zero. The arithmetic
## would come to the same picture, but "the default show runs the code it has always run"
## is worth more than saving an assignment - and it is checkable, which a shader that
## multiplies everything by zero is not.
##
## The material is REUSED across calls. Rebuilding it per change re-compiles the shader,
## which on a filter dial being dragged is a compile per frame.
static func apply(rect: CanvasItem, amounts: Dictionary, size: Vector2) -> void:
	if rect == null or not is_instance_valid(rect):
		return
	var live := sanitize(amounts)
	if live.is_empty():
		rect.material = null
		return
	var mat := rect.material as ShaderMaterial
	if mat == null:
		mat = ShaderMaterial.new()
		mat.shader = load(SHADER)
		rect.material = mat
	# EVERY uniform, every time - including the ones at zero. Writing only the live ones
	# leaves a filter that was just switched off still burning at its old amount, which is
	# a checkbox that cannot be unticked.
	for k in REGISTRY:
		mat.set_shader_parameter(String(REGISTRY[k]), float(live.get(k, 0.0)))
	# The dot lattice and the grain are sized in PIXELS, so they need the frame's real size
	# rather than UV. Without it a dot is a fixed fraction of the picture and the
	# pointillism gets coarser as the window gets bigger.
	mat.set_shader_parameter("u_size", size)


## The one-run override: `--filter monochrome=1,static=0.3` (or `--filter none`).
##
## Reads like `--medium`, and exists for the same two reasons: a gate that must pin a look
## without touching the author's settings, and a render of a session that was deliberately
## not the remembered one. Returns null when the flag is absent, so a caller can tell "no
## override" from "override to nothing".
static func from_args(args: PackedStringArray) -> Variant:
	var i := args.find("--filter")
	if i < 0 or i + 1 >= args.size():
		return null
	var spec := String(args[i + 1])
	if spec == "none":
		return {}
	var out := {}
	for part in spec.split(",", false):
		var bits := String(part).split("=")
		var key := String(bits[0]).strip_edges()
		if not REGISTRY.has(key):
			push_warning("ghost: --filter %s is not a known filter - ignored" % key)
			continue
		out[key] = clampf(float(bits[1]) if bits.size() > 1 else 1.0, 0.0, 1.0)
	return sanitize(out)


## "Monochrome 100%, Grain 40%", or "none" - for a status line and a tooltip.
static func describe(amounts: Dictionary) -> String:
	var live := sanitize(amounts)
	if live.is_empty():
		return "none"
	var parts: Array = []
	for k in REGISTRY:
		if live.has(k):
			parts.append("%s %d%%" % [LABELS.get(k, k), int(round(float(live[k]) * 100.0))])
	return ", ".join(parts)
