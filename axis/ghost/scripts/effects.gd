extends RefCounted
class_name Effects

## Effects - air that moves and light that bursts: FOG that rolls through a stretch of the scene,
## MOTES that wander about it (pixies, fireflies, embers, dust) and leave and come back, and BURSTS
## of sparks, glitter, embers, flame, smoke or stars at a moment. An agent describes them as data -
## the registries' words are what it reads ([method describe]) - [method sanitize] makes whatever it
## wrote buildable, and [method build] makes the nodes. Generic: a host names its REGIONS (boxes in
## its own space) and its MOMENTS (a time, how long, and where the emitter is through it), and the
## tarot table is the first host.
##
## EVERYTHING IS A FUNCTION OF SHOW TIME ([method Air.tick]): a mote's place, a spark's flight and the
## fog's roll are computed from the time and the seed, never stepped frame by frame - so a render and
## a scrub see the same picture. The fog is Godot's own volumetric fog (a FogVolume per stretch,
## [constant FOG_SHADER]): lights in it light it, so candles glow through it, and a mote that carries a
## light makes a colored glow in the fog around it. Sparks fly on the GPU from where and when they
## were born ([constant SPRITE_SHADER]); motes are placed on the CPU, which also moves their lights.

const FOG_SHADER := preload("res://shaders/effect_fog.gdshader")
const SPRITE_SHADER := preload("res://shaders/effect_sprite.gdshader")
const SMOKE_SHADER := preload("res://shaders/effect_smoke.gdshader")

## THE KINDS of effect, and what each is for.
const KINDS := {
	"fog": "a bank or a layer of fog or smoke, rolling slowly through one stretch of the scene, lit by the lights in it",
	"motes": "a few points of light or dust that wander about one stretch of the scene, now and then leaving it and coming back",
	"burst": "a burst of particles at a moment of the reading - sparks, glitter, embers, flame, smoke or stars - thrown from where that moment happens",
}

## What a mote can be: its shape, its size (millimeters - its core; its halo is wider) and the range
## an agent may set it in (`sizes`), how brightly it glows over `glow` 0-1 (`glows`: brighter than 1
## blooms), how fast and far it wanders, whether it blinks (fireflies), flickers (embers), rises
## (embers) or falls (snow, ash), whether it carries a little light (so fog glows round it and the
## table takes its color) and its color when none is given.
const MOTES := {
	"pixie": {"about": "tiny bright points of colored light that dart and drift, alone or in twos and threes", "shape": "orb",
		"size": 3.0, "sizes": Vector2(1.5, 8.0), "glows": Vector2(0.4, 1.3), "speed": 1.0, "wander": 1.0, "blink": 0.0,
		"flicker": 0.15, "rise": 0.0, "light": true, "color": "#ffe9a8"},
	"firefly": {"about": "slow, low-flying sparks that blink on for a moment and off for longer, yellow-green", "shape": "orb",
		"size": 2.5, "sizes": Vector2(1.2, 6.0), "glows": Vector2(0.6, 1.8), "speed": 0.45, "wander": 0.7, "blink": 1.0,
		"flicker": 0.0, "rise": 0.0, "light": true, "color": "#d8ff6e"},
	"wisp": {"about": "large, faint, slow glows that hang and drift like will-o'-the-wisps", "shape": "orb",
		"size": 14.0, "sizes": Vector2(6.0, 40.0), "glows": Vector2(0.3, 1.0), "speed": 0.25, "wander": 0.8, "blink": 0.0,
		"flicker": 0.3, "rise": 0.0, "light": true, "color": "#a8e6ff"},
	"ember": {"about": "sparks rising from a fire, flickering, winking out at the top", "shape": "orb",
		"size": 1.6, "sizes": Vector2(0.8, 4.0), "glows": Vector2(0.8, 2.2), "speed": 0.6, "wander": 0.3, "blink": 0.0,
		"flicker": 0.8, "rise": 1.0, "light": false, "color": "#ff9a3c"},
	"dust": {"about": "motes of dust drifting through the light, barely there", "shape": "orb",
		"size": 1.0, "sizes": Vector2(0.5, 3.0), "glows": Vector2(0.15, 0.5), "speed": 0.15, "wander": 0.5, "blink": 0.0,
		"flicker": 0.2, "rise": 0.0, "light": false, "color": "#fff1d6"},
	"snow": {"about": "flakes falling slowly, drifting as they fall - snow, ash, pollen", "shape": "flake",
		"size": 3.0, "sizes": Vector2(1.5, 8.0), "glows": Vector2(0.3, 0.8), "speed": 0.35, "wander": 0.4, "blink": 0.0,
		"flicker": 0.0, "rise": -1.0, "light": false, "color": "#f4f6ff"},
}

## What a burst can be: its sprite's shape, how many, how fast they leave (m/s), how widely (the
## cone's half-angle, degrees), what pulls on them (m/s2: below 0 they fall, above 0 they rise), how
## soon the air slows them, how long they live (s), their size (mm) and the range an agent may set it
## in, how brightly they glow over `glow` 0-1, how they grow, waver and twinkle.
const BURSTS := {
	"sparks": {"about": "bright streaks thrown out fast, arcing down and dying - a sparkler's", "shape": "streak",
		"count": 160, "speed": 0.5, "spread": 70.0, "gravity": -1.2, "drag": 4.0, "life": 0.6, "size": 2.0,
		"sizes": Vector2(1.0, 5.0), "glows": Vector2(0.7, 2.0), "grow": 0.0, "waver": 0.0, "twinkle": 0.25, "color": "#ffb347",
		"stretch": 0.05, "burst": 3.0},
	"glitter": {"about": "tiny flakes that glint as they drift down slowly", "shape": "star",
		"count": 110, "speed": 0.22, "spread": 180.0, "gravity": -0.08, "drag": 2.8, "life": 2.4, "size": 2.6,
		"sizes": Vector2(1.0, 6.0), "glows": Vector2(0.5, 1.8), "grow": 0.0, "waver": 0.012, "twinkle": 1.0, "color": "#ffe6a0"},
	"embers": {"about": "glowing bits that float up, flicker and wink out", "shape": "orb",
		"count": 60, "speed": 0.18, "spread": 110.0, "gravity": 0.16, "drag": 1.6, "life": 2.0, "size": 2.0,
		"sizes": Vector2(1.0, 5.0), "glows": Vector2(0.7, 2.2), "grow": 0.0, "waver": 0.03, "twinkle": 0.7, "color": "#ff8a33"},
	"flames": {"about": "tongues of fire that leap up and are gone", "shape": "flame",
		"count": 42, "speed": 0.16, "spread": 35.0, "gravity": 0.45, "drag": 2.6, "life": 0.7, "size": 9.0,
		"sizes": Vector2(4.0, 25.0), "glows": Vector2(0.6, 1.8), "grow": 0.6, "waver": 0.03, "twinkle": 0.2, "color": "#ff7a2a"},
	"smoke": {"about": "a soft puff that billows out, rises and thins away", "shape": "puff",
		"count": 26, "speed": 0.1, "spread": 120.0, "gravity": 0.05, "drag": 1.6, "life": 3.2, "size": 26.0,
		"sizes": Vector2(10.0, 60.0), "glows": Vector2(0.0, 0.0), "grow": 2.5, "waver": 0.015, "twinkle": 0.0, "color": "#b9b4ad"},
	"stars": {"about": "little four-pointed stars bursting out, turning and fading", "shape": "star",
		"count": 46, "speed": 0.3, "spread": 180.0, "gravity": 0.0, "drag": 3.2, "life": 1.3, "size": 6.0,
		"sizes": Vector2(3.0, 12.0), "glows": Vector2(0.6, 1.9), "grow": 0.0, "waver": 0.0, "twinkle": 0.8, "color": "#ffe9b0"},
}

## The sprite shapes, as the sprite shader numbers them.
const SHAPES := {"orb": 0, "streak": 1, "flame": 2, "puff": 3, "star": 4, "flake": 5}

## The most of each: effects in all, fogs, motes in one population and in all, lit motes (each a
## light), and particles a burst makes over all its moments.
const MAX_EFFECTS := 8
const MAX_FOG := 3
const MAX_MOTES := 40
const MAX_MOTES_ALL := 96
const MAX_LIGHTS := 12
const MAX_PARTICLES := 2400
## The time a moment's particles are born over, when the moment is an instant (seconds).
const BIRTH := 0.12

## A MOTE'S LIGHT is for the fog: [constant MOTE_LIGHT] is what the fog round it sees at full
## brightness and middle glow - the color bleeding through - and a surface sees [constant MOTE_TINT]
## of that. A light this small a few centimeters off the cloth burned a hot spot that bloomed into a
## blot many times the mote's size.
const MOTE_LIGHT := 0.03
const MOTE_TINT := 1.0 / 15.0
## How far a mote keeps above the top of anything it flies over (the stage's `occluders`: the table,
## the things on it) - a lit one further, so its light never burns a spot (meters) - and how fast that
## floor falls away round a thing's edge (meters per meter squared: 4 cm down at 10 cm out, 16 at 20),
## so a mote coming near is lifted over it gently: never through it, never in a jump.
const CLEAR := 0.004
const LIT_CLEAR := 0.04
const FLOOR_FALL := 4.0


## THE VOCABULARY, as an agent reads it. [param regions] and [param moments] are the host's names:
## where an effect may be, and when a burst may happen - each with a few words about it.
static func describe(regions: Dictionary, moments: Dictionary) -> String:
	var lines := PackedStringArray()
	lines.append("KINDS (an effect's `kind`):")
	for k in KINDS:
		lines.append("- %s: %s" % [k, KINDS[k]])
	lines.append("")
	lines.append("WHERE an effect is (`where`, for fog and motes):")
	for r in regions:
		lines.append("- \"%s\": %s" % [r, String(regions[r])])
	lines.append("")
	lines.append("FOG: {\"name\", \"kind\": \"fog\", \"where\", \"color\", \"density\" 0-1 (a whisper of haze to a thick bank), \"rolls\" 0-1 (how much it billows and turns over), \"drift\" -1..1 (its slow wind, left to right), \"height\" (centimeters it reaches above its floor), \"grain\" (centimeters: the size of its swirls), \"glow\" 0-1 (light of its own, for fog that should show without a light in it)}.")
	lines.append("")
	lines.append("MOTES: {\"name\", \"kind\": \"motes\", \"look\", \"where\", \"count\" 1-%d, \"colors\" [\"#rrggbb\", ...] (each mote takes one), \"size\" (millimeters: the look's own unless given, kept within its range), \"glow\" 0-1 (how bright, within the look's range), \"speed\" 0-1, \"away\" 0-1 (how much of the time each is gone from the scene), \"light\" true or false (a lit mote glows in the fog round it and tints what it passes)}. Looks, each with its size and the sizes it can be:" % MAX_MOTES)
	for k in MOTES:
		var mo: Dictionary = MOTES[k]
		lines.append("- %s: %s (%s mm, %s to %s)" % [k, String(mo["about"]), _mm(float(mo["size"])),
			_mm((mo["sizes"] as Vector2).x), _mm((mo["sizes"] as Vector2).y)])
	lines.append("")
	lines.append("BURSTS: {\"name\", \"kind\": \"burst\", \"look\", \"on\" (the moment), \"colors\" [\"#rrggbb\", ...], \"count\", \"size\" (millimeters, within the look's range), \"speed\" 0-1, \"spread\" (degrees round the way they are thrown), \"life\" (seconds), \"glow\" 0-1 (within the look's range)}. Looks, each with its size and the sizes it can be:")
	for k in BURSTS:
		var bo: Dictionary = BURSTS[k]
		lines.append("- %s: %s (%s mm, %s to %s)" % [k, String(bo["about"]), _mm(float(bo["size"])),
			_mm((bo["sizes"] as Vector2).x), _mm((bo["sizes"] as Vector2).y)])
	lines.append("")
	lines.append("MOMENTS a burst can be `on`:")
	for m in moments:
		lines.append("- \"%s\": %s" % [m, String(moments[m])])
	lines.append("")
	lines.append("At most %d effects, %d of them fog; leave out any field to take the look's own." % [MAX_EFFECTS, MAX_FOG])
	return "\n".join(lines)


# --- sanitizing -----------------------------------------------------------------------------

## WHATEVER AN AGENT WROTE, as something [method build] can make: known kinds, looks, regions and
## moments only, every number in range, every color a color (the [param palette]'s when none is
## given). [param regions] and [param moments] are the host's names. Unknown entries are dropped.
static func sanitize(raw: Variant, palette: Array, regions: Array, moments: Array) -> Array:
	var out: Array = []
	var fogs := 0
	var motes := 0
	var pal: Array = palette if not palette.is_empty() else ["#ffe9a8"]
	for e in (raw if raw is Array else []):
		if out.size() >= MAX_EFFECTS:
			break
		if not (e is Dictionary):
			continue
		var d: Dictionary = e
		var kind := String(d.get("kind", "")).strip_edges().to_lower()
		var fx := {"kind": kind, "name": Props._text(d.get("name", kind), 80)}
		match kind:
			"fog":
				if fogs >= MAX_FOG:
					continue
				var where := _pick(d.get("where", ""), regions)
				if where.is_empty():
					continue
				fogs += 1
				fx["where"] = where
				fx["color"] = Props._color(d.get("color", ""), String(pal[pal.size() - 1]))
				fx["density"] = Props._num(d.get("density"), 0.35, 0.0, 1.0)
				fx["rolls"] = Props._num(d.get("rolls"), 0.5, 0.0, 1.0)
				fx["drift"] = Props._num(d.get("drift"), 0.2, -1.0, 1.0)
				fx["height"] = Props._num(d.get("height"), 25.0, 1.0, 200.0)
				fx["grain"] = Props._num(d.get("grain"), 25.0, 3.0, 200.0)
				fx["glow"] = Props._num(d.get("glow"), 0.0, 0.0, 1.0)
			"motes":
				var look := String(d.get("look", "")).strip_edges().to_lower()
				var where := _pick(d.get("where", ""), regions)
				if not MOTES.has(look) or where.is_empty():
					continue
				var base: Dictionary = MOTES[look]
				var n := int(Props._num(d.get("count"), 8.0, 1.0, float(MAX_MOTES)))
				n = mini(n, MAX_MOTES_ALL - motes)
				if n <= 0:
					continue
				motes += n
				fx["look"] = look
				fx["where"] = where
				fx["count"] = n
				fx["colors"] = _colors(d.get("colors", d.get("color")), [String(base["color"])])
				fx["size"] = Props._num(d.get("size"), float(base["size"]), (base["sizes"] as Vector2).x, (base["sizes"] as Vector2).y)
				fx["glow"] = Props._num(d.get("glow"), 0.5, 0.0, 1.0)
				fx["speed"] = Props._num(d.get("speed"), 0.5, 0.0, 1.0)
				fx["away"] = Props._num(d.get("away"), 0.3, 0.0, 0.9)
				fx["light"] = Props._flag(d.get("light"), bool(base["light"]))
			"burst":
				var look := String(d.get("look", "")).strip_edges().to_lower()
				var on := _pick(d.get("on", ""), moments)
				if not BURSTS.has(look) or on.is_empty():
					continue
				var base: Dictionary = BURSTS[look]
				fx["look"] = look
				fx["on"] = on
				fx["colors"] = _colors(d.get("colors", d.get("color")), [String(base["color"])])
				fx["count"] = int(Props._num(d.get("count"), float(base["count"]), 1.0, 400.0))
				fx["size"] = Props._num(d.get("size"), float(base["size"]), (base["sizes"] as Vector2).x, (base["sizes"] as Vector2).y)
				fx["speed"] = Props._num(d.get("speed"), 0.5, 0.0, 1.0)
				fx["spread"] = Props._num(d.get("spread"), float(base["spread"]), 0.0, 180.0)
				fx["life"] = Props._num(d.get("life"), float(base["life"]), 0.15, 6.0)
				fx["glow"] = Props._num(d.get("glow"), 0.6, 0.0, 1.0)
			_:
				continue
		out.append(fx)
	return out


## Millimeters as the vocabulary writes them: whole above ten, one decimal below.
static func _mm(v: float) -> String:
	return str(roundi(v)) if v >= 10.0 or is_equal_approx(v, roundf(v)) else str(snappedf(v, 0.1))


## [param v] as one of [param names] (case and spaces forgiven), or "".
static func _pick(v: Variant, names: Array) -> String:
	var s := String(v).strip_edges().to_lower() if v is String else ""
	for n in names:
		if String(n).to_lower() == s:
			return String(n)
	return ""


static func _col(s: String) -> Color:
	return Color.html(s.strip_edges()) if Props._is_color(s) else Color(0.8, 0.8, 0.8)


static func _colors(v: Variant, fallback: Array) -> Array:
	var out: Array = []
	for c in (v if v is Array else ([v] if v is String else [])):
		if Props._is_color(c):
			out.append(String(c).strip_edges())
	return (out if not out.is_empty() else fallback).slice(0, 8)


# --- the stage it is built on ---------------------------------------------------------------

## The environment's volumetric fog, on for a scene with fog in it ([param on]) and off otherwise:
## the volumes alone make it (no fog everywhere), over the few meters a tabletop scene spans.
static func fog_environment(env: Environment, on: bool) -> void:
	env.volumetric_fog_enabled = on
	if not on:
		return
	env.volumetric_fog_density = 0.0
	env.volumetric_fog_albedo = Color(1, 1, 1)
	env.volumetric_fog_emission = Color(0, 0, 0)
	env.volumetric_fog_anisotropy = 0.35
	env.volumetric_fog_length = 7.0
	env.volumetric_fog_detail_spread = 2.0
	env.volumetric_fog_gi_inject = 0.0
	env.volumetric_fog_ambient_inject = 0.6
	env.volumetric_fog_sky_affect = 0.0
	env.volumetric_fog_temporal_reprojection_enabled = true
	env.volumetric_fog_temporal_reprojection_amount = 0.8


## BUILD [param effects] (sanitized) for a host whose [param stage] is
## `{regions: {name: AABB}, camera: Transform3D, fov: float (vertical, degrees), aspect: float,
## occluders: [AABB]}` - the occluders being what stands in the air (a table, the things on it).
## [param seed] varies everything sampled. An [Air] to add to the scene and tick.
static func build(effects: Array, stage: Dictionary, seed: int) -> Air:
	var air := Air.new()
	air.root = Node3D.new()
	air.root.name = "Air"
	air.stage = stage
	var lights := 0
	for i in effects.size():
		var fx: Dictionary = effects[i]
		var salt := hash([seed, i, String(fx.get("name", ""))])
		match String(fx["kind"]):
			"fog":
				air.fogs.append(_fog(fx, stage, air.root))
			"motes":
				var m := _motes(fx, stage, salt, air.root, MAX_LIGHTS - lights)
				lights += (m["lights"] as Array).size()
				air.motes.append(m)
			"burst":
				air.bursts.append(_burst(fx, salt, air.root))
	return air


## The host's region [param name]: `{box: AABB, floor: the height fog lies on, sight: how much of it a
## line of sight crosses, meters}` - a bare AABB is a box with its bottom for floor.
static func _info(stage: Dictionary, name: String) -> Dictionary:
	var r: Variant = (stage.get("regions", {}) as Dictionary).get(name)
	if r is AABB:
		return {"box": r, "floor": (r as AABB).position.y, "sight": maxf((r as AABB).size.z, 0.05), "motes": r}
	if r is Dictionary and (r as Dictionary).get("box") is AABB:
		var box: AABB = (r as Dictionary)["box"]
		return {"box": box, "floor": float((r as Dictionary).get("floor", box.position.y)),
			"sight": float((r as Dictionary).get("sight", maxf(box.size.z, 0.05))),
			"motes": (r as Dictionary)["motes"] if (r as Dictionary).get("motes") is AABB else box}
	var dflt := AABB(Vector3(-0.5, 0.0, -0.5), Vector3(1.0, 0.3, 1.0))
	return {"box": dflt, "floor": 0.0, "sight": 1.0, "motes": dflt}


static func _region(stage: Dictionary, name: String) -> AABB:
	return _info(stage, name)["box"]


static func _fog(fx: Dictionary, stage: Dictionary, root: Node3D) -> Dictionary:
	var info := _info(stage, String(fx["where"]))
	var box: AABB = info["box"]
	var vol := FogVolume.new()
	vol.shape = RenderingServer.FOG_VOLUME_SHAPE_BOX
	vol.size = box.size
	vol.position = box.get_center()
	var mat := ShaderMaterial.new()
	mat.shader = FOG_SHADER
	var c := _col(String(fx["color"]))
	mat.set_shader_parameter("color", Vector3(c.r, c.g, c.b))
	# THE DENSITY AN AGENT WRITES is how much the fog hides, looked through: 0-1 over the stretch a
	# line of sight crosses there (its `sight`) - a layer two centimeters thick and a bank three
	# meters deep at one density look as thick as each other. The fog's swirls fill about half of it.
	var hides := 0.9 * float(fx["density"])
	mat.set_shader_parameter("density", -log(1.0 - hides) / maxf(float(info["sight"]), 0.02) / 0.45)
	mat.set_shader_parameter("rolls", float(fx["rolls"]))
	mat.set_shader_parameter("wind", Vector3(float(fx["drift"]) * 0.12, 0.012, -0.02 * absf(float(fx["drift"]))))
	mat.set_shader_parameter("grain", float(fx["grain"]) * 0.01)
	mat.set_shader_parameter("floor_y", float(info["floor"]))
	# a layer thinner than the fog's own grain cannot be seen: a few centimeters at least
	mat.set_shader_parameter("height", maxf(float(fx["height"]) * 0.01, 0.05))
	mat.set_shader_parameter("glow", float(fx["glow"]) * 0.6)
	mat.set_shader_parameter("edge", clampf(minf(box.size.x, box.size.z) * 0.25, 0.04, 0.4))
	vol.material = mat
	root.add_child(vol)
	return {"fx": fx, "node": vol, "mat": mat}


## A SPRITE LAYER: [param count] quads in one MultiMesh, drawn by [param shader], never culled (its
## sprites are moved by the shader, so its bounds would lie).
static func _sprites(count: int, shader: Shader, root: Node3D) -> Dictionary:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	mm.use_custom_data = true
	var q := QuadMesh.new()
	q.size = Vector2.ONE
	mm.mesh = q
	mm.instance_count = count
	var mi := MultiMeshInstance3D.new()
	mi.multimesh = mm
	mi.custom_aabb = AABB(Vector3(-20, -20, -20), Vector3(40, 40, 40))
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	var mat := ShaderMaterial.new()
	mat.shader = shader
	mi.material_override = mat
	root.add_child(mi)
	return {"mm": mm, "node": mi, "mat": mat}


# --- motes ------------------------------------------------------------------------------------

## A POPULATION OF MOTES: each its own home in view inside its region, its own wander, visits and
## blinking, sampled once from [param salt]; up to [param lights_left] of them carry a light.
static func _motes(fx: Dictionary, stage: Dictionary, salt: int, root: Node3D, lights_left: int) -> Dictionary:
	var look: Dictionary = MOTES[String(fx["look"])]
	var box: AABB = _info(stage, String(fx["where"]))["motes"]
	var n := int(fx["count"])
	var rng := RandomNumberGenerator.new()
	rng.seed = salt
	var layer := _sprites(n, SPRITE_SHADER, root)
	var mat: ShaderMaterial = layer["mat"]
	mat.set_shader_parameter("posed", true)
	mat.set_shader_parameter("shape", int(SHAPES[String(look["shape"])]))
	var glows: Vector2 = look["glows"]
	mat.set_shader_parameter("glow", lerpf(glows.x, glows.y, float(fx["glow"])))
	var speed := float(look["speed"]) * lerpf(0.35, 1.8, float(fx["speed"]))
	var wander := float(look["wander"])
	var size := float(fx["size"]) * 0.001
	var colors: Array = fx["colors"]
	var each: Array = []
	var lights: Array = []
	var cam: Transform3D = stage.get("camera", Transform3D.IDENTITY)
	var tan_half := tan(deg_to_rad(float(stage.get("fov", 42.0)) * 0.5))
	var aspect := float(stage.get("aspect", 16.0 / 9.0))
	for i in n:
		var home := _in_view(box, stage, rng)
		# A MOTE WANDERS ACROSS A SHARE OF THE PICTURE round a home in it - measured on screen at its own
		# depth, across and up as the camera sees, with a little depth - so a mote past the table stays
		# in the strip of room the camera sees there, and one near the lens does not fill it
		var depth := maxf(-(cam.affine_inverse() * home).z, 0.1)
		var frame_h := 2.0 * depth * tan_half
		# built here and handed over whole: appended through the dictionary, a packed array grows a copy
		var w := PackedFloat32Array()
		var ph := PackedFloat32Array()
		for k in 8:
			w.append(rng.randf_range(0.05, 0.3) * speed * TAU)
			ph.append(rng.randf() * TAU)
		var m := {
			"home": home,
			"ax": cam.basis.x * frame_h * aspect * 0.08 * wander,
			"ay": cam.basis.y * frame_h * 0.05 * wander,
			"az": -cam.basis.z * depth * 0.06 * wander,
			"w": w, "ph": ph,
			"period": rng.randf_range(22.0, 60.0) / maxf(speed, 0.2),
			"off": rng.randf(),
			"exit": _exit_of(home, stage, rng),
			"blink": Vector3(rng.randf_range(2.5, 6.0), rng.randf_range(0.35, 0.9), rng.randf()),
			"seed": rng.randf() * 100.0,
			"lift": rng.randf(),
			"color": _col(String(colors[i % colors.size()])),
			"dart": rng.randf_range(0.6, 1.0) if String(fx["look"]) == "pixie" else 0.0,
		}
		each.append(m)
		var mm: MultiMesh = layer["mm"]
		# the quad holds the mote's halo as well as its core: three times the core's size
		mm.set_instance_custom_data(i, Color(0, 0, 0, size * 3.0 * rng.randf_range(0.8, 1.2)))
		if bool(fx["light"]) and lights.size() < lights_left:
			var l := OmniLight3D.new()
			l.light_color = m["color"]
			# A SMALL LIGHT: a glow in the fog round the mote, a tint on what it passes - never a lamp
			l.omni_range = 0.12
			l.omni_attenuation = 2.0
			l.shadow_enabled = false
			l.light_volumetric_fog_energy = 1.0 / MOTE_TINT
			l.light_specular = 0.2
			root.add_child(l)
			lights.append({"light": l, "mote": i})
	return {"fx": fx, "look": look, "box": box, "each": each, "layer": layer, "lights": lights,
		"glow": float(fx["glow"]), "away": float(fx["away"]), "under": stage.get("occluders", []),
		"clear": LIT_CLEAR if bool(fx["light"]) else CLEAR}


## A HOME FOR A MOTE, spread over the PICTURE rather than the region's volume: a spot on screen, and a
## depth along that line of sight where it is inside [param box], nearer than anything the stage
## `occluders` put in its way (a table) and not right at the lens - so motes are where they can be
## seen, as many near as far. Tried a few times before taking the middle of the box.
static func _in_view(box: AABB, stage: Dictionary, rng: RandomNumberGenerator) -> Vector3:
	var cam: Transform3D = stage.get("camera", Transform3D.IDENTITY)
	var k := tan(deg_to_rad(float(stage.get("fov", 42.0)) * 0.5))
	var aspect := float(stage.get("aspect", 16.0 / 9.0))
	for i in 64:
		var sx := rng.randf_range(0.06, 0.94)
		var sy := rng.randf_range(0.03, 0.92)
		var dir := (cam.basis * Vector3((sx * 2.0 - 1.0) * k * aspect, (1.0 - sy * 2.0) * k, -1.0)).normalized()
		var span := _ray_box(cam.origin, dir, box)
		var near := maxf(span.x, 0.2)
		var far := span.y
		for o in stage.get("occluders", []):
			var hit := _ray_box(cam.origin, dir, o as AABB)
			if hit.x <= hit.y and hit.x > 0.0:
				far = minf(far, hit.x - 0.01)
		if far > near:
			return cam.origin + dir * rng.randf_range(near, far)
	return box.get_center()


## Where a ray from [param from] along [param dir] is inside [param box]: (entering, leaving) distances,
## the first larger than the second when it misses.
static func _ray_box(from: Vector3, dir: Vector3, box: AABB) -> Vector2:
	var t0 := -INF
	var t1 := INF
	for a in 3:
		var o := from[a]
		var d := dir[a]
		var lo := box.position[a]
		var hi := box.end[a]
		if absf(d) < 1e-9:
			if o < lo or o > hi:
				return Vector2(1.0, -1.0)
			continue
		var ta := (lo - o) / d
		var tb := (hi - o) / d
		t0 = maxf(t0, minf(ta, tb))
		t1 = minf(t1, maxf(ta, tb))
	return Vector2(t0, t1) if t1 >= maxf(t0, 0.0) else Vector2(1.0, -1.0)


## Whether something at [param p] is behind one of the stage's `occluders` from the camera.
static func _hidden(stage: Dictionary, p: Vector3) -> bool:
	var eye: Vector3 = (stage.get("camera", Transform3D.IDENTITY) as Transform3D).origin
	for o in stage.get("occluders", []):
		if (o as AABB).intersects_segment(eye, p.lerp(eye, 0.002)):
			return true
	return false


## Where a mote goes when it leaves: well out of the picture, to one side and up.
static func _exit_of(home: Vector3, stage: Dictionary, rng: RandomNumberGenerator) -> Vector3:
	var cam: Transform3D = stage.get("camera", Transform3D.IDENTITY)
	# OFF TO A SIDE AND UP: away from the camera would keep it in a picture that widens with distance
	var side := cam.basis.x * (1.0 if rng.randf() < 0.5 else -1.0)
	var dir := (side + Vector3.UP * rng.randf_range(0.2, 0.7)).normalized()
	var out := home
	for i in 24:
		out = home + dir * (0.3 + 0.25 * float(i))
		var s: Variant = _screen(stage, out)
		if s == null or (s as Vector2).x < -0.15 or (s as Vector2).x > 1.15 or (s as Vector2).y < -0.15 or (s as Vector2).y > 1.15:
			break
	return out


## Where [param p] is in the picture, 0..1 across and down; null behind the camera.
static func _screen(stage: Dictionary, p: Vector3) -> Variant:
	var cam: Transform3D = stage.get("camera", Transform3D.IDENTITY)
	var l: Vector3 = cam.affine_inverse() * p
	if l.z > -0.01:
		return null
	var k := tan(deg_to_rad(float(stage.get("fov", 42.0)) * 0.5))
	var aspect := float(stage.get("aspect", 16.0 / 9.0))
	return Vector2(0.5 + l.x / (-l.z * k * aspect) * 0.5, 0.5 - l.y / (-l.z * k) * 0.5)


## MOTE [param m] of a population ([param pop]) at show time [param t]: `{pos, bright}`. Pure.
static func mote_at(pop: Dictionary, m: Dictionary, t: float) -> Dictionary:
	var look: Dictionary = pop["look"]
	var w: PackedFloat32Array = m["w"]
	var ph: PackedFloat32Array = m["ph"]
	var ax: Vector3 = m["ax"]
	var ay: Vector3 = m["ay"]
	var az: Vector3 = m["az"]
	var box: AABB = pop["box"]
	var p: Vector3 = m["home"]
	p += ax * (0.65 * sin(w[0] * t + ph[0]) + 0.35 * sin(w[1] * t + ph[1])) \
		+ ay * (0.65 * sin(w[2] * t + ph[2]) + 0.35 * sin(w[3] * t + ph[3])) \
		+ az * (0.65 * sin(w[4] * t + ph[4]) + 0.35 * sin(w[5] * t + ph[5]))
	# A PIXIE DARTS: now and then a quick swerve, between long drifts
	var dart := float(m["dart"])
	if dart > 0.0:
		var env := pow(maxf(0.0, sin(w[6] * 0.35 * t + ph[6])), 6.0)
		p += (ax * sin(w[7] * 4.0 * t + ph[7]) + ay * 0.6 * cos(w[7] * 3.1 * t)) * 0.7 * dart * env
	# RISING or FALLING: up (or down) through the region and round again
	var rise := float(look["rise"])
	if rise != 0.0:
		var span := maxf(box.size.y, 0.05)
		var v := absf(rise) * float(look["speed"]) * 0.08 * (0.7 + 0.6 * float(m["lift"]))
		var u := fposmod(float(m["lift"]) + v * t / span, 1.0)
		p.y = box.position.y + (u if rise > 0.0 else 1.0 - u) * span
	var bright := 1.0
	if rise != 0.0:
		var u2 := fposmod(float(m["lift"]) + absf(rise) * float(look["speed"]) * 0.08 * (0.7 + 0.6 * float(m["lift"])) * t / maxf(box.size.y, 0.05), 1.0)
		bright *= smoothstep(0.0, 0.12, u2) * (1.0 - smoothstep(0.75, 1.0, u2))
	# VISITS: here for most of its round, then away out of the picture and back
	var away := float(pop["away"])
	var presence := 1.0
	if away > 0.0:
		var period := float(m["period"])
		var c := fposmod(t / period + float(m["off"]), 1.0)
		var here := 1.0 - away
		var ramp := minf(4.0 / period, here * 0.45)
		presence = smoothstep(0.0, ramp, c) * (1.0 - smoothstep(here - ramp, here, c))
		if rise != 0.0:
			# a rising or falling mote is not off anywhere: it thins out where it is
			bright *= presence
		else:
			var e := presence * presence * (3.0 - 2.0 * presence)
			p = (m["exit"] as Vector3).lerp(p, e)
			bright *= lerpf(0.35, 1.0, presence)
	# NEVER UNDER ITS FLOOR - its region's, or the top of what stands under it - a soft floor, which it
	# glides along and lifts off, never stopping or bouncing at it
	var fl := _floor_at(pop, p)
	var soft := 0.012
	var over := (p.y - fl) / soft
	p.y = fl + soft * (over if over > 20.0 else log(1.0 + exp(maxf(over, -40.0))))
	var blink := float(look["blink"])
	if blink > 0.0:
		var b: Vector3 = m["blink"]
		var c2 := fposmod(t / b.x + b.z, 1.0)
		var on := c2 / (b.y / b.x)
		var pulse := sin(PI * on) if on < 1.0 else 0.0
		bright *= lerpf(1.0 - blink, 1.0, 0.04 + 0.96 * pulse * pulse)
	var flicker := float(look["flicker"])
	if flicker > 0.0:
		var s := float(m["seed"])
		bright *= 1.0 - flicker * 0.5 * (0.5 + 0.5 * sin(t * 11.0 + s)) * (0.5 + 0.5 * sin(t * 6.7 + s * 2.3))
	return {"pos": p, "bright": bright, "presence": presence}


## THE FLOOR UNDER [param p] for population [param pop]: its region's, or over what stands there its
## top and the population's clearance ([constant CLEAR], [constant LIT_CLEAR]), falling away round it
## as a dome ([constant FLOOR_FALL]) - so the floor never steps.
static func _floor_at(pop: Dictionary, p: Vector3) -> float:
	var fl := (pop["box"] as AABB).position.y + CLEAR
	for o in pop.get("under", []):
		var b: AABB = o
		var out := maxf(maxf(maxf(b.position.x - p.x, p.x - b.end.x), maxf(b.position.z - p.z, p.z - b.end.z)), 0.0)
		fl = maxf(fl, b.end.y + float(pop["clear"]) - FLOOR_FALL * out * out)
	return fl


# --- bursts -----------------------------------------------------------------------------------

static func _burst(fx: Dictionary, salt: int, root: Node3D) -> Dictionary:
	var look: Dictionary = BURSTS[String(fx["look"])]
	var smoke := String(look["shape"]) == "puff"
	var layer := _sprites(1, SMOKE_SHADER if smoke else SPRITE_SHADER, root)
	var mat: ShaderMaterial = layer["mat"]
	mat.set_shader_parameter("posed", false)
	mat.set_shader_parameter("shape", int(SHAPES[String(look["shape"])]))
	var glows: Vector2 = look["glows"]
	mat.set_shader_parameter("glow", lerpf(glows.x, glows.y, float(fx["glow"])))
	mat.set_shader_parameter("gravity", Vector3(0.0, float(look["gravity"]), 0.0))
	mat.set_shader_parameter("drag", float(look["drag"]))
	mat.set_shader_parameter("grow", float(look["grow"]))
	mat.set_shader_parameter("waver", float(look["waver"]))
	mat.set_shader_parameter("twinkle", float(look["twinkle"]))
	mat.set_shader_parameter("stretch", float(look.get("stretch", 0.022)))
	(layer["mm"] as MultiMesh).instance_count = 0
	return {"fx": fx, "look": look, "layer": layer, "salt": salt, "planned": ""}


## BIRTHS FOR A BURST over its [param moments] (`[{t, dur, path: [[t, Transform3D], ...], from}]`):
## each particle's birth time and place (on the emitter as it was at that time - a point, or the edge
## of a card for `from` "card"), its velocity, color, life and size. Pure: the same moments give the
## same particles. `[{born, pos, vel, color, life, size, seed}]`.
static func births(fx: Dictionary, moments: Array, salt: int) -> Array:
	var look: Dictionary = BURSTS[String(fx["look"])]
	var out: Array = []
	if moments.is_empty():
		return out
	var per := mini(int(fx["count"]), maxi(1, MAX_PARTICLES / moments.size()))
	var speed := float(look["speed"]) * lerpf(0.4, 1.7, float(fx["speed"]))
	var spread := deg_to_rad(float(fx["spread"]))
	var colors: Array = fx["colors"]
	var size := float(fx["size"]) * 0.001
	var life := float(fx["life"])
	for mi in moments.size():
		var mo: Dictionary = moments[mi]
		var rng := RandomNumberGenerator.new()
		rng.seed = hash([salt, mi, snappedf(float(mo["t"]), 0.001)])
		var t0 := float(mo["t"])
		var dur := maxf(float(mo.get("dur", 0.0)), BIRTH)
		var path: Array = mo["path"]
		var card := String(mo.get("from", "point")) == "card"
		for i in per:
			# MOST AT THE START: a burst throws most of itself as its moment begins, and trails off
			var u := (float(i) + rng.randf()) / float(per)
			var born := t0 + dur * pow(u, float(look.get("burst", 2.0)))
			var xf := _along(path, born)
			var local := Vector3.ZERO
			var out_dir := Vector3.UP
			if card:
				# a point on the card's edge (70 x 120 mm, in its own x-z plane), thrown out from it
				var a := rng.randf() * TAU
				var e := Vector2(cos(a) * 0.035, sin(a) * 0.06)
				var k := minf(0.035 / maxf(absf(e.x), 1e-4), 0.06 / maxf(absf(e.y), 1e-4))
				local = Vector3(e.x * minf(k, 1.0), 0.0, e.y * minf(k, 1.0))
				out_dir = (xf.basis * Vector3(local.x, 0.0, local.z)).normalized()
				out_dir = (out_dir + (xf.basis.y.normalized() * (1.0 if rng.randf() < 0.5 else -1.0)) * 0.35).normalized()
			else:
				local = Vector3(rng.randf_range(-1, 1), rng.randf_range(0, 1), rng.randf_range(-1, 1)) * 0.008
				out_dir = Vector3.UP
			var dir := _cone(out_dir, spread, rng)
			var v := dir * speed * rng.randf_range(0.45, 1.15)
			out.append({"born": born, "pos": xf * local, "vel": v,
				"color": _col(String(colors[rng.randi_range(0, colors.size() - 1)])),
				"life": life * rng.randf_range(0.6, 1.2), "size": size * rng.randf_range(0.7, 1.3), "seed": rng.randf()})
	return out


## The emitter at [param t] along [param path] (`[[t, Transform3D], ...]`): between two samples, its
## place and turn eased between them.
static func _along(path: Array, t: float) -> Transform3D:
	if path.size() == 1 or t <= float(path[0][0]):
		return path[0][1]
	for i in range(1, path.size()):
		if t <= float(path[i][0]):
			var a: Array = path[i - 1]
			var b: Array = path[i]
			var u := (t - float(a[0])) / maxf(float(b[0]) - float(a[0]), 1e-6)
			return (a[1] as Transform3D).interpolate_with(b[1] as Transform3D, u)
	return path[path.size() - 1][1]


## A direction within [param spread] (radians) of [param axis].
static func _cone(axis: Vector3, spread: float, rng: RandomNumberGenerator) -> Vector3:
	var ax := axis.normalized()
	var other := Vector3.RIGHT if absf(ax.dot(Vector3.RIGHT)) < 0.9 else Vector3.FORWARD
	var u := ax.cross(other).normalized()
	var w := ax.cross(u)
	var cz := lerpf(1.0, cos(minf(spread, PI)), rng.randf())
	var r := sqrt(maxf(0.0, 1.0 - cz * cz))
	var phi := rng.randf() * TAU
	return (ax * cz + u * r * cos(phi) + w * r * sin(phi)).normalized()


## AIR: an effects scene as built - its fogs, mote populations and bursts - posed from show time.
class Air:
	extends RefCounted

	var root: Node3D
	var stage := {}
	var fogs: Array = []
	var motes: Array = []
	var bursts: Array = []

	func has_fog() -> bool:
		return not fogs.is_empty()

	## The moments each burst is `on` have changed (or are first known): its particles are born again
	## from [param moments] (`{name: [{t, dur, path, from}]}`) - only where its own moments changed.
	func plan(moments: Dictionary) -> void:
		for b in bursts:
			var fx: Dictionary = b["fx"]
			var mine: Array = moments.get(String(fx["on"]), [])
			var key := str(mine.map(func(m: Dictionary) -> String: return "%.3f/%.3f" % [float(m["t"]), float(m.get("dur", 0.0))]))
			if key == String(b["planned"]):
				continue
			b["planned"] = key
			b["plans"] = int(b.get("plans", 0)) + 1
			var born := Effects.births(fx, mine, int(b["salt"]))
			var mm: MultiMesh = (b["layer"] as Dictionary)["mm"]
			mm.instance_count = born.size()
			for i in born.size():
				var p: Dictionary = born[i]
				# the velocity rides in the transform's first column, which nothing else uses
				mm.set_instance_transform(i, Transform3D(Basis(p["vel"] as Vector3, Vector3.UP, Vector3.BACK), p["pos"] as Vector3))
				mm.set_instance_color(i, p["color"])
				mm.set_instance_custom_data(i, Color(float(p["born"]), float(p["life"]), float(p["seed"]), float(p["size"])))

	## Everything as it is at show time [param t].
	func tick(t: float) -> void:
		for f in fogs:
			(f["mat"] as ShaderMaterial).set_shader_parameter("show_time", t)
		for b in bursts:
			((b["layer"] as Dictionary)["mat"] as ShaderMaterial).set_shader_parameter("show_time", t)
		for pop in motes:
			var mm: MultiMesh = (pop["layer"] as Dictionary)["mm"]
			var each: Array = pop["each"]
			var lit := {}
			for l in pop["lights"]:
				lit[int((l as Dictionary)["mote"])] = (l as Dictionary)["light"]
			for i in each.size():
				var m: Dictionary = each[i]
				var at := Effects.mote_at(pop, m, t)
				var pos: Vector3 = at["pos"]
				var bright := float(at["bright"])
				mm.set_instance_transform(i, Transform3D(Basis.IDENTITY, pos))
				var c: Color = m["color"]
				mm.set_instance_color(i, Color(c.r, c.g, c.b, bright))
				if lit.has(i):
					var light: OmniLight3D = lit[i]
					light.position = pos
					light.light_energy = MOTE_LIGHT * MOTE_TINT * bright * lerpf(0.4, 1.6, float(pop["glow"]))

	func release() -> void:
		if root != null and is_instance_valid(root):
			root.queue_free()
		root = null
