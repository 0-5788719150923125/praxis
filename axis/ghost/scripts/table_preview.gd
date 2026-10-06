extends RefCounted
class_name TablePreview

## TablePreview - a table described but not yet set, SEEN: each thing alone in a studio, and the whole
## table standing on the episode's own cloth, photographed from the camera's own place. It is what
## the set dresser looks through ([SetDresserTools]): it makes a thing, looks at it, and fixes what
## it sees.
##
## THE STUDIO shows a thing as it is built - every part, at its real size - on a ground ruled in
## centimeters (a brighter line every five), lit from above and one side so its form reads, its
## flames lit. One thing from four sides (front, its right, above, and as the episode's camera looks
## down at the table), or several side by side, one tile each, each tile captioned.
##
## THE TABLE IS THE SHOW'S OWN: a [TarotMedium] built on the episode's seed and pictures - its
## camera, its lamp, where the deck sits, the spread the cards are laid in (face down: the set dresser
## knows no card) - with the description being tried standing in for the episode's table. Where a
## thing stands, what is made smaller or left off for want of room, which light leads: the answers
## are the ones the show itself will give, because it is the show's own code giving them. The medium
## is loaded by its class name when a table is stood, and only where ghost's autoloads run (it, like
## every medium, reads the Director): naming it here would stop the gates that build prompts from
## compiling, and a command-line process cannot stand a table anyway - it says so ([method can_set]).
##
## RENDERED IN THIS PROCESS, off screen: a SubViewport per stage, drawn only while a picture is being
## taken, then stopped. A process with no renderer (headless) takes no pictures ([method can_see]),
## and the tools work from their reports alone.

## The size of every picture handed back, and how wide the studio's lens is (degrees, vertical).
const SHEET := Vector2i(1280, 720)
const STUDIO_FOV := 28.0
## Frames drawn before a picture is read back: the studio's shadows, and the table's reflections,
## settle in a few.
const STUDIO_FRAMES := 3
const TABLE_FRAMES := 6
## The moment of a reading the air is shown at: some way in, its fog rolled and its motes about.
const AIR_AT := 40.0
## The episode's table medium, by class name: looked up when it is needed, never compiled in - it
## needs ghost's autoloads, and so would anything naming it (see the class note).
const MEDIUM := "TarotMedium"
## The pictures of the episode the table is built on - copied beside the description being tried,
## never its card faces: the set dresser knows no card.
const PICTURES := ["surface.png", "backdrop.png", "backdrop.json", "back.png"]
## The studio's four sides of one thing: a direction from its middle toward the camera, and a caption.
const SIDES := [["front", Vector3(0.0, 0.14, 1.0)], ["its right side", Vector3(1.0, 0.14, 0.0)],
	["from above (its front at the bottom)", Vector3(0.0, 1.0, 0.0)], ["as the camera sees it", Vector3.ZERO]]

const GROUND_SHADER := """
shader_type spatial;
render_mode diffuse_burley, specular_disabled;
uniform vec3 ground : source_color = vec3(0.13, 0.135, 0.15);
uniform vec3 line : source_color = vec3(0.3, 0.31, 0.34);
uniform vec3 line5 : source_color = vec3(0.5, 0.51, 0.55);
varying vec3 world;
void vertex() {
	world = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
float grid(vec2 p) {
	vec2 w = max(fwidth(p), vec2(1e-5));
	vec2 g = abs(fract(p - 0.5) - 0.5) / w;
	// a line finer than a pixel fades out instead of filling the ground
	return (1.0 - min(min(g.x, g.y), 1.0)) * clamp(1.5 - max(w.x, w.y) * 3.0, 0.0, 1.0);
}
void fragment() {
	vec2 cm = world.xz * 100.0;
	vec3 c = mix(ground, line, grid(cm) * 0.8);
	ALBEDO = mix(c, line5, grid(cm / 5.0));
	ROUGHNESS = 0.92;
}
"""

## A candle's flame, as the table draws one ([constant TarotMedium.FLAME_SHADER]), turned to face
## whichever side the studio shoots from.
const FLAME_SHADER := """
shader_type spatial;
render_mode unshaded, cull_disabled, blend_add, depth_draw_never;
uniform vec3 color = vec3(1.0, 0.65, 0.3);
uniform float energy = 4.0;
void vertex() {
	MODELVIEW_MATRIX = VIEW_MATRIX * mat4(INV_VIEW_MATRIX[0], INV_VIEW_MATRIX[1], INV_VIEW_MATRIX[2], MODEL_MATRIX[3]);
}
void fragment() {
	vec2 p = UV - vec2(0.5, 0.62);
	p.x *= 2.2;
	float body = 1.0 - smoothstep(0.0, 0.42, length(vec2(p.x, p.y * (p.y < 0.0 ? 0.55 : 1.0))));
	float core = 1.0 - smoothstep(0.0, 0.16, length(vec2(p.x * 1.4, p.y + 0.12)));
	ALBEDO = mix(color, vec3(1.0, 0.97, 0.88), core) * energy * body;
	ALPHA = body;
}
"""


## What the table medium reads its episode from - the captions it is bound to, in a reading.
class Doc:
	extends RefCounted
	var document := {}


var episode: TarotEpisode
var plan: Dictionary
var _dir := ""                   # the description being tried, beside copies of the episode's pictures
var _pitch := 38.0                # the episode's camera, looking down (degrees)

var _studio: SubViewport = null
var _studio_cam: Camera3D
var _studio_label: Label
var _held: Node3D
var _flame_mat: ShaderMaterial

var _table: SubViewport = null
var _medium = null               # a TarotMedium, loaded by class name
var _doc: Doc


## [param dir]: the folder the preview keeps its copies in (inside the job's own).
func _init(ep: TarotEpisode, episode_plan: Dictionary, dir: String) -> void:
	episode = ep
	plan = episode_plan
	_dir = dir
	_pitch = float(TarotTable.layout_of(ep.seed)["pitch"])


## Whether this process can take a picture at all.
static func can_see() -> bool:
	return DisplayServer.get_name() != "headless"


## Whether it can stand the episode's table: a picture, and ghost's autoloads for the medium.
static func can_set() -> bool:
	return can_see() and _tree().root.has_node("Director")


## Give back everything this preview built.
func release() -> void:
	if _studio != null and is_instance_valid(_studio):
		_studio.queue_free()
	if _table != null and is_instance_valid(_table):
		_table.queue_free()
	_studio = null
	_table = null
	_medium = null


# --- the studio -------------------------------------------------------------------------------------

## THE THINGS in [param spec] (a sanitized table: `things`, `materials`) named in [param names] - or
## all of them - side by side, a tile each, from the camera's angle, each captioned with its number
## ([param numbers]: name -> number, the caller's; else its place here), name and size. null when no
## picture can be taken.
func things(spec: Dictionary, names: Array = [], numbers: Dictionary = {}) -> Image:
	if not can_see():
		return null
	var list: Array = []
	for t in spec.get("things", []):
		if names.is_empty() or names.has(String((t as Dictionary).get("name", ""))):
			list.append(t)
	if list.is_empty():
		return null
	_ensure_studio()
	var cols := clampi(ceili(sqrt(float(list.size()) * 1.6)), 1, list.size())
	var rows := ceili(float(list.size()) / float(cols))
	var tile := Vector2i(SHEET.x / cols, SHEET.y / rows)
	var sheet := Image.create(SHEET.x, SHEET.y, false, Image.FORMAT_RGB8)
	sheet.fill(Color(0.06, 0.06, 0.07))
	for i in list.size():
		var t: Dictionary = list[i]
		var box := _hold(t, spec.get("materials", {}), i)
		var cap := "%d. %s\n%s" % [int(numbers.get(String(t.get("name", "")), i + 1)), String(t.get("name", "")), TablePreview.size_text(box)]
		var img := await _shot(box, _camera_side(), tile - Vector2i(2, 2), cap)
		if img != null:
			sheet.blit_rect(img, Rect2i(Vector2i.ZERO, img.get_size()), Vector2i((i % cols) * tile.x + 1, (i / cols) * tile.y + 1))
	_clear_held()
	return sheet


## ONE THING from four sides ([constant SIDES]), in a sheet of four. null when no picture can be taken.
func thing(spec: Dictionary, name: String) -> Image:
	if not can_see():
		return null
	var t := {}
	for x in spec.get("things", []):
		if String((x as Dictionary).get("name", "")) == name:
			t = x
			break
	if t.is_empty():
		return null
	_ensure_studio()
	var box := _hold(t, spec.get("materials", {}), 0)
	var tile := Vector2i(SHEET.x / 2, SHEET.y / 2)
	var sheet := Image.create(SHEET.x, SHEET.y, false, Image.FORMAT_RGB8)
	sheet.fill(Color(0.06, 0.06, 0.07))
	for i in SIDES.size():
		var side: Vector3 = SIDES[i][1]
		if side == Vector3.ZERO:
			side = _camera_side()
		var cap := "%s\n%s - %s" % [String(t.get("name", "")), TablePreview.size_text(box), String(SIDES[i][0])] if i == 0 else String(SIDES[i][0])
		var img := await _shot(box, side, tile - Vector2i(2, 2), cap)
		if img != null:
			sheet.blit_rect(img, Rect2i(Vector2i.ZERO, img.get_size()), Vector2i((i % 2) * tile.x + 1, (i / 2) * tile.y + 1))
	_clear_held()
	return sheet


## A thing's size as the captions and reports give it: width x height x depth, centimeters.
static func size_text(box: AABB) -> String:
	return "%s x %s x %s cm" % [_cm(box.size.x), _cm(box.size.y), _cm(box.size.z)]


static func _cm(m: float) -> String:
	var cm := snappedf(m * 100.0, 0.1) if m < 0.1 else float(roundi(m * 100.0))
	return str(roundi(cm)) if is_equal_approx(cm, roundf(cm)) else str(cm)


## From the episode's camera: looking down across the table at its pitch.
func _camera_side() -> Vector3:
	var p := deg_to_rad(_pitch)
	return Vector3(0.0, sin(p), cos(p))


func _ensure_studio() -> void:
	if _studio != null and is_instance_valid(_studio):
		return
	_studio = SubViewport.new()
	_studio.own_world_3d = true
	_studio.msaa_3d = Viewport.MSAA_4X
	_studio.size = SHEET
	_studio.render_target_update_mode = SubViewport.UPDATE_DISABLED
	_tree().root.add_child(_studio)
	var root := Node3D.new()
	_studio.add_child(root)
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.035, 0.035, 0.04)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.55, 0.55, 0.58)
	env.ambient_light_energy = 0.35
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.tonemap_exposure = 0.95
	# SOMETHING FOR METAL TO REFLECT: a dim room-toned sky, seen in reflections only - against the
	# flat background a polished thing reflected black and read as paint
	var sky := Sky.new()
	var sm := ProceduralSkyMaterial.new()
	sm.sky_top_color = Color(0.42, 0.42, 0.45)
	sm.sky_horizon_color = Color(0.62, 0.58, 0.52)
	sm.ground_bottom_color = Color(0.1, 0.1, 0.11)
	sm.ground_horizon_color = Color(0.3, 0.28, 0.26)
	sky.sky_material = sm
	env.sky = sky
	env.reflected_light_source = Environment.REFLECTION_SOURCE_SKY
	env.glow_enabled = true
	env.glow_hdr_threshold = 1.25
	_studio_cam = Camera3D.new()
	_studio_cam.fov = STUDIO_FOV
	_studio_cam.near = 0.002
	_studio_cam.far = 20.0
	_studio_cam.environment = env
	root.add_child(_studio_cam)
	var ground := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(4.0, 4.0)
	ground.mesh = pm
	var gm := ShaderMaterial.new()
	gm.shader = Shader.new()
	gm.shader.code = GROUND_SHADER
	ground.material_override = gm
	root.add_child(ground)
	# a key from above, to the left and in front, its shadows short and sharp; a cool fill opposite;
	# a rim from behind, so a dark thing keeps its outline against the dark
	for l in [[Vector3(-52.0, -35.0, 0.0), 1.5, Color(1.0, 0.95, 0.86), true],
			[Vector3(-28.0, 140.0, 0.0), 0.45, Color(0.8, 0.86, 1.0), false],
			[Vector3(-15.0, 175.0, 0.0), 0.35, Color(1.0, 1.0, 1.0), false]]:
		var d := DirectionalLight3D.new()
		d.rotation_degrees = l[0]
		d.light_energy = float(l[1])
		d.light_color = l[2]
		d.shadow_enabled = bool(l[3])
		d.directional_shadow_max_distance = 3.0
		d.shadow_bias = 0.02
		d.shadow_normal_bias = 0.6
		root.add_child(d)
	_held = Node3D.new()
	root.add_child(_held)
	_studio_label = Label.new()
	_studio_label.position = Vector2(10, 6)
	_studio_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_studio_label.add_theme_font_size_override("font_size", 18)
	_studio_label.add_theme_color_override("font_color", Color(1.0, 0.98, 0.92))
	_studio_label.add_theme_color_override("font_outline_color", Color(0.0, 0.0, 0.0, 0.9))
	_studio_label.add_theme_constant_override("outline_size", 6)
	_studio.add_child(_studio_label)
	_flame_mat = ShaderMaterial.new()
	_flame_mat.shader = Shader.new()
	_flame_mat.shader.code = FLAME_SHADER


## Thing [param t] built and standing in the studio alone, its flames lit: its bounds.
func _hold(t: Dictionary, materials: Dictionary, salt: int) -> AABB:
	_clear_held()
	var b := Props.build(t, materials, hash([episode.seed, salt, "preview"]))
	var node: Node3D = b["node"]
	_held.add_child(node)
	var flames: Array = b["wicks"]
	for g in b.get("glows", []):
		if g is ShaderMaterial:
			(g as ShaderMaterial).set_shader_parameter("flame", 1.0)
	var mid := Vector3.ZERO
	for w in flames:
		var q := MeshInstance3D.new()
		var qm := QuadMesh.new()
		qm.size = Vector2(0.012, 0.03)
		q.mesh = qm
		q.material_override = _flame_mat
		q.position = (w as Vector3) + Vector3(0.0, 0.016, 0.0)
		_held.add_child(q)
		mid += w as Vector3
	if not flames.is_empty():
		var light := OmniLight3D.new()
		light.light_color = Color(1.0, 0.7, 0.4)
		light.omni_range = 0.5
		light.light_energy = 0.25 * float(flames.size())
		light.position = mid / float(flames.size()) + Vector3(0.0, 0.02, 0.0)
		_held.add_child(light)
	var box: AABB = b["size"]
	return box if box.size != Vector3.ZERO else AABB(Vector3(-0.02, 0.0, -0.02), Vector3(0.04, 0.04, 0.04))


func _clear_held() -> void:
	if _held == null:
		return
	for c in _held.get_children():
		c.queue_free()


## One picture of what the studio holds: framed to [param box], seen from [param side] (a direction
## from its middle toward the camera), at [param size], captioned.
func _shot(box: AABB, side: Vector3, size: Vector2i, caption: String) -> Image:
	_studio.size = size
	var c := box.get_center()
	var r := maxf(box.size.length() * 0.5, 0.01)
	var half := deg_to_rad(STUDIO_FOV) * 0.5
	var across := atan(tan(half) * float(size.x) / float(maxi(size.y, 1)))
	var dist := r / sin(minf(half, across)) * 1.06
	var dir := side.normalized()
	var up := Vector3.UP if absf(dir.y) < 0.99 else Vector3(0.0, 0.0, -1.0)
	_studio_cam.transform = Transform3D(Basis.looking_at(-dir, up), c + dir * dist)
	_studio_cam.far = dist + r * 6.0 + 1.0
	_studio_label.text = caption
	# a caption wraps within its tile: a long name would hide the size after it
	_studio_label.size = Vector2(size.x - 20, 0)
	return await _render(_studio, STUDIO_FRAMES)


# --- the table --------------------------------------------------------------------------------------

## THE WHOLE TABLE: [param raw] (the description as written) standing on the episode's table, the
## cards laid face down in their spread and the deck beside it, its air (fog, motes) as it is some way
## into a reading, from the camera's place.
## `{image, stood: [{name, place, at, k, tall}], left_off: [names], key, held_down: [names], cards,
## deck}` - `at` in centimeters from the cloth's middle (x to the reader's right, z toward them), `k`
## how far it was made smaller (1 = not at all), `tall` its share of the picture's height. `error`
## when the table cannot be stood here.
func table(raw: Dictionary) -> Dictionary:
	var err := _stand(raw)
	if not err.is_empty():
		return {"error": err}
	_pose_spread()
	_medium._tick_air(AIR_AT)
	var out := _placed(TarotTable.sanitize_table(raw, TarotTable.sanitize_look(plan.get("look", {}) if plan.get("look") is Dictionary else {})))
	out["image"] = await _render(_table, TABLE_FRAMES)
	return out


## ONE EFFECT IN MOTION: the effect named [param name] in [param raw]'s `effects`, on the episode's
## table, photographed four times - a burst a moment after the thing it marks happens (a card leaping
## from the deck, one held up and twirled, one laid down...: [constant TarotTable.MOMENTS], staged
## here as the table stages them), fog or motes a few seconds apart, so their roll and their wander
## show. A sheet of four, `{image, frames: [seconds after]}`; `error` when it cannot be shown.
func watch(raw: Dictionary, name: String) -> Dictionary:
	var err := _stand(raw)
	if not err.is_empty():
		return {"error": err}
	var air = _medium._air
	var fx := {}
	for e in (TarotTable.sanitize_table(raw, TarotTable.sanitize_look(plan.get("look", {}) if plan.get("look") is Dictionary else {}))["effects"] as Array):
		if String((e as Dictionary).get("name", "")) == name:
			fx = e
	if air == null or fx.is_empty():
		return {"error": "no effect called \"%s\" can be built" % name}
	var burst := String(fx["kind"]) == "burst"
	var on := String(fx.get("on", ""))
	var steps: Array = [0.08, 0.3, 0.6, 1.0] if burst else [0.0, 3.0, 6.0, 9.0]
	var moment := _staged(on) if burst else {}
	if burst:
		air.plan({on: moment["moments"]})
	var sheet := Image.create(SHEET.x, SHEET.y, false, Image.FORMAT_RGB8)
	sheet.fill(Color(0.06, 0.06, 0.07))
	for i in steps.size():
		var t := AIR_AT + float(steps[i])
		_pose_moment(on, moment, t)
		air.tick(t)
		var img: Image = await _render(_table, TABLE_FRAMES)
		if img == null:
			return {"error": "no picture could be taken"}
		img.resize(SHEET.x / 2 - 2, SHEET.y / 2 - 2, Image.INTERPOLATE_LANCZOS)
		sheet.blit_rect(img, Rect2i(Vector2i.ZERO, img.get_size()), Vector2i((i % 2) * (SHEET.x / 2) + 1, (i / 2) * (SHEET.y / 2) + 1))
	return {"image": sheet, "frames": steps, "burst": burst}


## The episode's table built over [param raw] (the medium mounted the first time): "" when it stands.
func _stand(raw: Dictionary) -> String:
	if not can_set():
		return "this process cannot stand the table (%s)" % ("no renderer" if not can_see() else "no ghost session")
	if _table == null or not is_instance_valid(_table):
		_table = SubViewport.new()
		_table.own_world_3d = true
		_table.size = SHEET
		_table.render_target_update_mode = SubViewport.UPDATE_DISABLED
		_tree().root.add_child(_table)
		_medium = TablePreview._load(MEDIUM)
		if _medium == null:
			_table.queue_free()
			_table = null
			return "the table medium could not be loaded"
		_medium.mount(_table)
		_doc = Doc.new()
		_medium.bind_captions(_doc)
	_sync_pictures()
	var err := TextGen.put(_dir.path_join("table.json"), JSON.stringify(raw, "\t"))
	if not err.is_empty():
		return err
	var cards: Array = []
	for i in episode.card_count():
		cards.append({"key": "card %d" % (i + 1), "name": "", "numeral": "", "reversed": false, "jumper": false,
			"position": {}, "booklet": {}, "art": ""})
	var images := {}
	for k in ["back", "surface", "backdrop"]:
		if FileAccess.file_exists(_dir.path_join(k + ".png")):
			images[k] = _dir.path_join(k + ".png")
	_doc.document = {"source": "", "title": "", "tarot": {"show": episode.show, "seed": episode.seed, "dir": _dir,
		"plan": plan, "images": images, "cards": cards}}
	_medium._key = ""
	_medium._ensure_doc()
	return ""


## A MOMENT STAGED for a burst `on` [param on], at [constant AIR_AT], the way the table stages it: where
## its emitter is through it (`moments`, as [method TarotMedium._air_moments] gives them) and how the
## card in it moves (`card`: `[[t, Transform3D], ...]`, empty for none).
func _staged(on: String) -> Dictionary:
	var m = _medium
	var consts := (m.get_script() as Script).get_script_constant_map()
	var t0 := AIR_AT
	var card: Array = []
	var moments: Array = []
	match on:
		"shuffle":
			m._cur_base = m._mid
			moments.append({"t": t0, "dur": 0.5, "from": "point",
				"path": [[t0, Transform3D(Basis.IDENTITY, (m._mid as Vector3) + Vector3(0.0, 0.03, 0.0))]]})
		"jumper":
			# a card springing off the deck in the middle and flying to land, as a jumper does
			m._cur_base = m._mid
			var fly: Vector2 = consts.get("JUMP_FLY", Vector2(1.15, 2.25))
			var path: Array = []
			for i in 9:
				var u := lerpf(fly.x, fly.y, float(i) / 8.0)
				path.append([t0 + (u - fly.x), m._jump_xf(0, u, Transform3D.IDENTITY)])
			card = path
			moments.append({"t": t0, "dur": fly.y - fly.x, "from": "card", "path": path})
		"reveal":
			var xf: Transform3D = m._present_xf(0, t0)
			card = [[t0, xf]]
			moments.append({"t": t0, "dur": 0.3, "from": "card", "path": card})
		"pirouette":
			# a card held up and twirled round three times, winding down as a hand's flourish does
			var held: Transform3D = m._present_xf(0, t0)
			var up: Vector3 = (m._cam_base as Transform3D).basis.y
			var path: Array = []
			for i in 17:
				var u := float(i) / 16.0
				var turn := PI + 5.0 * PI * (1.0 - pow(1.0 - u, 2.2))
				path.append([t0 + u * 1.8, Transform3D(Basis(up, turn) * held.basis, held.origin)])
			card = path
			moments.append({"t": t0, "dur": 1.8, "from": "card", "path": path})
		"lay":
			card = [[t0, m._slot_xf(0)]]
			moments.append({"t": t0, "dur": 0.2, "from": "card", "path": card})
		"close":
			for k in (m._cards as Array).size():
				moments.append({"t": t0, "dur": 0.8, "from": "card", "path": [[t0, m._slot_xf(k)]]})
	return {"moments": moments, "card": card}


## The table as it is at [param t] in a staged moment ([method _staged]): the spread laid and the deck
## at its side - or, for a shuffle or a jumper, the deck in the middle - and the moment's card where
## the moment has it.
func _pose_moment(on: String, moment: Dictionary, t: float) -> void:
	var m = _medium
	_pose_spread()
	if on == "shuffle" or on == "jumper":
		m._cur_base = m._mid
		for i in (m._deck as Array).size():
			(m._deck[i] as MeshInstance3D).transform = m._rest_xf(i)
		for k in (m._cards as Array).size():
			(m._cards[k] as MeshInstance3D).visible = false
	var card: Array = moment.get("card", [])
	if not card.is_empty() and (m._cards as Array).size() > 0:
		var c: MeshInstance3D = m._cards[0]
		c.visible = true
		c.transform = Effects._along(card, t)


## The episode's pictures beside the description, copied again when the episode's change (the room
## may still be being painted while the table is set).
func _sync_pictures() -> void:
	DirAccess.make_dir_recursive_absolute(_dir)
	for f in PICTURES:
		var src := episode.dir.path_join(String(f))
		var dst := _dir.path_join(String(f))
		if FileAccess.file_exists(src):
			if not FileAccess.file_exists(dst) or FileAccess.get_modified_time(src) > FileAccess.get_modified_time(dst):
				DirAccess.copy_absolute(src, dst)
		elif FileAccess.file_exists(dst):
			DirAccess.remove_absolute(dst)


## The table at the end of a reading: every card down in the spread, face down; the deck squared at
## its side; the flames burning; the channel's title off.
func _pose_spread() -> void:
	var m = _medium
	var card_t := float((m.get_script() as Script).get_script_constant_map().get("CARD_T", 0.0007))
	m._cur_base = m._deck_base
	for i in (m._deck as Array).size():
		(m._deck[i] as MeshInstance3D).transform = m._rest_xf(i)
	for k in (m._cards as Array).size():
		var s: Dictionary = m._slots[k]
		var card: MeshInstance3D = m._cards[k]
		card.visible = true
		card.transform = Transform3D(Basis(Vector3.UP, float(s["yaw"])), (s["pos"] as Vector3) + Vector3(0.0, card_t * 0.5, 0.0))
	(m._page as Node3D).visible = false
	m._tick_camera(0.0)
	m._tick_props(30.0)
	(m._env as Environment).adjustment_brightness = 1.0
	(m._attrs as CameraAttributesPractical).dof_blur_far_enabled = false


## What the medium made of [param spec] (the table made safe): where each thing stood, what found no
## room, which light leads.
func _placed(spec: Dictionary) -> Dictionary:
	var m = _medium
	var consts := (m.get_script() as Script).get_script_constant_map()
	var stood: Array = []
	var used := {}
	var left_off: Array = []
	var by_light := {}
	for t in spec.get("things", []):
		var name := String((t as Dictionary).get("name", ""))
		var found := -1
		for i in (m._things as Array).size():
			if not used.has(i) and String((m._things[i] as Dictionary)["name"]) == name:
				found = i
				break
		if found < 0:
			left_off.append(name)
			continue
		used[found] = true
		var th: Dictionary = m._things[found]
		var node: Node3D = th["node"]
		var rect: Rect2 = th["rect"]
		stood.append({"name": name, "place": String(th["place"]),
			"at": Vector2(node.position.x * 100.0, (node.position.z - float((m._cloth as Node3D).position.z)) * 100.0),
			"k": node.transform.basis.get_scale().x, "tall": rect.size.y})
		for li in th.get("lights", []):
			by_light[int(li)] = name
	var key := "the lamp"
	if int(m._key_flame) >= 0:
		key = String(by_light.get(int(m._key_flame), "a candle"))
	var held: Array = []
	for i in (m._lights as Array).size():
		var l: Dictionary = m._lights[i]
		var full := float(consts.get("KEY_ENERGY" if i == int(m._key_flame) else "FILL_ENERGY", 0.35)) * float((l["flames"] as Array).size())
		if float(l["energy"]) < full * 0.9 and by_light.has(i):
			held.append(by_light[i])
	var deck: Vector3 = m._deck_base
	return {"stood": stood, "left_off": left_off, "key": key, "held_down": held,
		"cards": (m._cards as Array).size(), "deck": "front right" if deck.x > 0.0 else "front left"}


# --- the shutter ------------------------------------------------------------------------------------

## [param vp] drawn for [param frames] frames, read back, then stopped again.
func _render(vp: SubViewport, frames: int) -> Image:
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	for i in frames:
		await RenderingServer.frame_post_draw
	var img := vp.get_texture().get_image()
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	if img == null or img.is_empty():
		return null
	img.convert(Image.FORMAT_RGB8)
	return img


static func _tree() -> SceneTree:
	return Engine.get_main_loop() as SceneTree


## A new instance of the script whose class is [param class_name_], or null.
static func _load(class_name_: String) -> Object:
	for c in ProjectSettings.get_global_class_list():
		if String((c as Dictionary).get("class", "")) == class_name_:
			var script := load(String((c as Dictionary)["path"])) as Script
			return script.new() if script != null else null
	return null
