extends Medium
class_name TabletMedium

## TabletMedium - the reading as somebody browsing on a tablet lying on a desk.
##
## The chapter is a [TabletScript]: web pages written as markdown, and a few marks for what the
## hand does between them (open a url, search, a new tab, turn to landscape, skip the rest of a
## paragraph). The voice reads what is on the page; the hand acts in the rests the voice leaves
## for it (`<!-- action-hold -->`, sized by the same [method TabletScript.phases] this medium
## schedules from), so nothing has to be timed by the author.
##
## EVERYTHING ON THE SCREEN IS A FUNCTION OF SHOW TIME, like the book's leaf: actions are
## placed in the gaps between the spoken words that bracket them ([method _build_schedule]),
## and the screen is REPLAYED from them every frame ([method _state_at]), scroll included
## ([method _build_flings]) - so an export draws exactly what the live reading drew, and a
## reading that restarts lands on the right page without having to have watched the last one.
##
## THE TABLET NEVER MOVES; THE CAMERA DOES. Turning to landscape swings the camera a quarter
## turn round the slab, and the screen's content counter-rotates against the same curve, so
## it stays upright in the picture while the device turns beneath it - shrinking to fit as an
## iPad's does, and re-laid out for the new shape at the half-way point, where the old and new
## pictures are the same size.
##
## Reuses the book's desk, lamp and springs ([BookMedium]); nothing about pages or leaves.

## Physical screen, in texture pixels (portrait, 3:4) and in world units.
const SW := 1200.0
const SH := 1600.0
const SCREEN := Vector2(0.75, 1.0)
const BEZEL := 0.05
const SLAB_T := 0.028
const CORNER := 0.075
const SCREEN_CORNER := 0.03
## The browser's chrome, in logical pixels: status bar, tab strip, address bar row.
const STATUS := 40.0
const TABS := 62.0
const BAR := 86.0
const TOP := STATUS + TABS + BAR
## How a group of actions sits in the gap the voice left for it: a beat after the last word,
## and a beat before the next.
const LEAD := 0.35
const TAIL := 0.3
## When the tablet starts to wake, in show seconds: at once, under the intro.
const WAKE_AT := 0.4
## A tap: the finger coming down, then the press and its ripple; and how long a pressed link
## stays shaded.
const TOUCH_IN := 0.22
const TOUCH_OUT := 0.6
const PRESS_HOLD := 0.7
## The camera: a long lens, as in the book, and how far it stands for the whole slab and for
## the line being read, portrait -> landscape.
const VFOV := 22.0
const WIDE := Vector2(3.2, 2.4)
const NEAR := Vector2(1.95, 1.55)
const PITCH_WIDE := 66.0
const PITCH_NEAR := 74.0
## The springs: framing settles over seconds, the angles several times slower; a context switch
## quickens the framing to QUICK_TAU for about QUICK_TIME seconds.
const FRAME_TAU := 4.2
const QUICK_TAU := 1.8
const QUICK_TIME := 3.0
const ANGLE_SLOW := 3.0
const LINE_TAU := 3.0
## A page's arc: the shares of its words spent closing in and letting go, and how many words a
## page needs before it is worth settling into fully.
const ARC_IN := 0.15
const ARC_OUT := 0.15
const ARC_WORDS := 40.0
const CAM_LEAD := 4.0
## While reading, the share of the screen's height the camera's aim stays within.
const AIM_BAND := Vector2(0.4, 0.6)
## How far into its arc the camera comes for a real page, which is looked at, not read.
const REAL_LOOK := 0.5
## Each page's set-up, degrees: an offset per page plus a slow wander. Small on purpose - a
## tablet turned much off square reads as wrong.
const YAW_SPREAD := 2.2
const YAW_WANDER := 0.9
const ROLL_SPREAD := 0.8
const ROLL_WANDER := 0.4
const TILT_SPREAD := 3.0
const TILT_WANDER := 1.4

const SCREEN_SHADER := """
shader_type spatial;
render_mode cull_disabled;
uniform sampler2D screen_tex : source_color, filter_linear, repeat_disable;
uniform vec2 size = vec2(0.75, 1.0);
uniform float radius = 0.03;
uniform float glow = 1.15;
vec4 tap(vec2 uv) {
	vec2 dx = dFdx(uv) * 0.38;
	vec2 dy = dFdy(uv) * 0.38;
	vec4 c = texture(screen_tex, uv) * 2.0;
	c += texture(screen_tex, uv + dx + dy);
	c += texture(screen_tex, uv + dx - dy);
	c += texture(screen_tex, uv - dx + dy);
	c += texture(screen_tex, uv - dx - dy);
	return c / 6.0;
}
void fragment() {
	vec2 p = (UV - 0.5) * size;
	vec2 q = abs(p) - (size * 0.5 - radius);
	if (length(max(q, 0.0)) - radius > 0.0) discard;
	ALBEDO = vec3(0.012);
	EMISSION = tap(UV).rgb * glow;
	// glass, but a matte one: a sharp reflection of the lamp washed whole lines out
	ROUGHNESS = 0.45;
	SPECULAR = 0.12;
}
"""

var _subs = null
var _doc: Dictionary = {}
var _source := ""
var _title := ""
var _rev := -1
var _lays := {}                  # "page|orient" -> TabletPage
var _textures := {}

var _map: Array = []             # subtitle word -> spoken index (or -1)
var _map_j := 0
var _map_rem := ""
var _st0 := PackedFloat32Array() # spoken index -> when it began (-1 unknown)
var _st1 := PackedFloat32Array()
var _built_n := -1               # map size the schedule was built at
var _sched: Array = []           # [{a, t0, s}] by start time
var _flings := {}                # page -> [{t, from, to, dur}]

var _vp: SubViewport
var _canvas: ScreenCanvas
var _canvas_in: ScreenCanvas     # the incoming layout during a turn's dissolve
var _root3: Node3D
var _cam: Camera3D
var _env: Environment
var _glow: OmniLight3D
var _screen_mat: ShaderMaterial
var _placeholder: GhostScene
var _seed := 0
var _hue0 := 0.0                 # where the rainbow starts, per session
var _char0 := PackedInt32Array() # word -> its first letter's place in the running text
var _st: Dictionary = {}         # this frame's screen state
var _now := 0.0

# camera springs
var _c_aim := Vector3.ZERO
var _v_aim := Vector3.ZERO
var _c_dist := 3.2
var _v_dist := 0.0
var _c_pitch := PITCH_WIDE
var _v_pitch := 0.0
var _c_wan := 0.0
var _v_wan := 0.0
var _c_roll := 0.0
var _v_roll := 0.0
var _c_tilt := 0.0
var _v_tilt := 0.0
var _line_y := 0.0
var _ctx := ""
var _quick := 0.0
var _snap := true
var _page_span := {}             # page -> Vector2i(first, last) spoken index on it


# --- mount ---------------------------------------------------------------------------

func mount(st: SubViewport) -> void:
	super.mount(st)
	_placeholder = GhostScene.new()
	_placeholder.init_with_seed(1, "drift")
	_placeholder.visible = false
	add_child(_placeholder)
	_vp = SubViewport.new()
	_vp.size = Vector2i(int(SW), int(SH))
	_vp.disable_3d = true
	_vp.transparent_bg = false
	_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(_vp)
	_canvas = ScreenCanvas.new()
	_canvas.tablet = self
	_vp.add_child(_canvas)
	_canvas_in = ScreenCanvas.new()
	_canvas_in.tablet = self
	_canvas_in.layer = 1
	_vp.add_child(_canvas_in)
	_build_world()


func _build_world() -> void:
	_root3 = Node3D.new()
	add_child(_root3)
	_env = Environment.new()
	_env.background_mode = Environment.BG_COLOR
	_env.background_color = Color(0.02, 0.018, 0.016)
	_env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	_env.ambient_light_color = Color(0.55, 0.5, 0.45)
	_env.ambient_light_energy = 0.3
	_env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	_env.adjustment_enabled = true
	_cam = Camera3D.new()
	_cam.fov = VFOV
	_cam.near = 0.05
	_cam.far = 80.0
	_cam.environment = _env
	_root3.add_child(_cam)
	var lamp := SpotLight3D.new()
	lamp.light_color = Color(1.0, 0.86, 0.68)
	lamp.light_energy = 4.2
	lamp.spot_range = 12.0
	lamp.spot_angle = 40.0
	lamp.spot_angle_attenuation = 1.6
	lamp.shadow_enabled = true
	lamp.shadow_blur = 1.2
	_root3.add_child(lamp)
	lamp.position = Vector3(-1.5, 3.4, -1.0)
	lamp.look_at_from_position(lamp.position, Vector3(0.1, 0.0, 0.1), Vector3.UP)
	var fill := DirectionalLight3D.new()
	fill.light_color = Color(0.7, 0.75, 0.9)
	fill.light_energy = 0.16
	_root3.add_child(fill)
	fill.rotation_degrees = Vector3(-55.0, 30.0, 0.0)

	var desk := MeshInstance3D.new()
	var dm := PlaneMesh.new()
	dm.size = Vector2(24.0, 24.0)
	desk.mesh = dm
	var dmat := StandardMaterial3D.new()
	dmat.albedo_color = Color(0.40, 0.24, 0.15)
	dmat.albedo_texture = BookMedium._grime(0x7AB1, 0.02, 5, 0.16)
	dmat.uv1_scale = Vector3(8.0, 8.0, 1.0)
	dmat.roughness = 0.62
	dmat.roughness_texture = dmat.albedo_texture
	desk.material_override = dmat
	desk.position = Vector3(0.0, -SLAB_T * 0.5, 0.0)
	_root3.add_child(desk)

	var slab := MeshInstance3D.new()
	slab.mesh = _rounded_slab(SCREEN + Vector2(BEZEL, BEZEL) * 2.0, SLAB_T, CORNER)
	var smat := StandardMaterial3D.new()
	smat.albedo_color = Color(0.035, 0.035, 0.04)
	smat.roughness = 0.22
	smat.metallic = 0.35
	smat.cull_mode = BaseMaterial3D.CULL_DISABLED
	slab.material_override = smat
	_root3.add_child(slab)

	var screen := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = SCREEN
	screen.mesh = pm
	var sh := Shader.new()
	sh.code = SCREEN_SHADER
	_screen_mat = ShaderMaterial.new()
	_screen_mat.shader = sh
	_screen_mat.set_shader_parameter("screen_tex", _vp.get_texture())
	_screen_mat.set_shader_parameter("size", SCREEN)
	_screen_mat.set_shader_parameter("radius", SCREEN_CORNER)
	screen.material_override = _screen_mat
	screen.position = Vector3(0.0, SLAB_T * 0.5 + 0.0008, 0.0)
	_root3.add_child(screen)

	# THE SCREEN LIGHTS THE DESK a little, as a lit tablet in a dark room does
	_glow = OmniLight3D.new()
	_glow.light_color = Color(0.8, 0.88, 1.0)
	_glow.omni_range = 1.6
	_glow.light_energy = 0.0
	_glow.position = Vector3(0.0, 0.35, 0.0)
	_root3.add_child(_glow)


## A slab with rounded corners: [param size] across x and z, [param t] thick, centred on 0.
static func _rounded_slab(size: Vector2, t: float, r: float) -> ArrayMesh:
	var outline := PackedVector2Array()
	var seg := 8
	var hw := size.x * 0.5 - r
	var hd := size.y * 0.5 - r
	var centres := [Vector2(hw, hd), Vector2(-hw, hd), Vector2(-hw, -hd), Vector2(hw, -hd)]
	for c in 4:
		for i in seg + 1:
			var a := (float(c) + float(i) / float(seg)) * PI * 0.5
			outline.append(centres[c] + Vector2(cos(a), sin(a)) * r)
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var n := outline.size()
	for i in n:
		var a := outline[i]
		var b := outline[(i + 1) % n]
		for y in [t * 0.5, -t * 0.5]:
			st.set_normal(Vector3(0.0, signf(y), 0.0))
			st.add_vertex(Vector3(0.0, y, 0.0))
			st.add_vertex(Vector3(a.x, y, a.y))
			st.add_vertex(Vector3(b.x, y, b.y))
		var na := Vector3(a.x, 0.0, a.y).normalized()
		var nb := Vector3(b.x, 0.0, b.y).normalized()
		for v in [[a, t * 0.5, na], [b, t * 0.5, nb], [b, -t * 0.5, nb],
				[a, t * 0.5, na], [b, -t * 0.5, nb], [a, -t * 0.5, na]]:
			st.set_normal(v[2])
			st.add_vertex(Vector3((v[0] as Vector2).x, v[1], (v[0] as Vector2).y))
	return st.commit()


func begin_session() -> void:
	_seed = Director.session_seed()
	_hue0 = float(hash([_seed, "hue"]) & 0xFFFF) / 65535.0
	_reset_reading()
	_snap = true


func release() -> void:
	_subs = null
	_reset_reading()


func _reset_reading() -> void:
	_map = []
	_map_j = 0
	_map_rem = ""
	var n := (_doc.get("spoken", PackedInt32Array()) as PackedInt32Array).size()
	_st0 = PackedFloat32Array()
	_st0.resize(n)
	_st0.fill(-1.0)
	_st1 = _st0.duplicate()
	_built_n = -1
	_sched = []
	_flings = {}


# --- the Medium contract ---------------------------------------------------------------

func owns_cast() -> bool:
	return true


func take_over(_outgoing: GhostScene) -> GhostScene:
	return _placeholder


func owns_bookend() -> bool:
	return true


func bind_captions(subs) -> bool:
	_subs = subs
	_source = ""
	_doc = {}
	_reset_reading()
	return true


func advance(_features, delta: float, bookend: float) -> void:
	var t := maxf(Spectrum.current.time, 0.0)
	var slen := Spectrum.song_length()
	var b := bookend if slen > 0.0 and t > slen * 0.5 else clampf(t / 1.2, 0.0, 1.0)
	_env.adjustment_brightness = clampf(b, 0.0, 1.0)
	_ensure_doc()
	_now = _subs.now() if _subs != null and is_instance_valid(_subs) else t
	if not _doc.is_empty():
		_extend_map()
		if _map.size() != _built_n:
			_built_n = _map.size()
			_build_schedule()
			_build_flings()
	_st = _state_at(_now)
	_glow.light_energy = 0.22 * float(_st.get("on", 0.0))
	_tick_camera(delta)
	_canvas.queue_redraw()
	var fade: Vector3 = _st.get("fade", Vector3(0, 0, -1))
	_canvas_in.visible = fade.z >= 0.0
	_canvas_in.modulate.a = clampf(fade.z, 0.0, 1.0)
	if _canvas_in.visible:
		_canvas_in.queue_redraw()


# --- the document ----------------------------------------------------------------------

func _ensure_doc() -> void:
	var src := ""
	var title := ""
	if _subs != null and is_instance_valid(_subs):
		var d: Dictionary = _subs.document
		src = String(d.get("source", ""))
		title = String(d.get("title", ""))
		if title.is_empty():
			title = BookLayout.field_of(src, "title")
	if src == _source and title == _title and _rev == Illustrations.revision and not _doc.is_empty():
		return
	if src.is_empty() and not _doc.is_empty():
		return
	_source = src
	_title = title
	if _rev != Illustrations.revision:
		_textures = {}
	_rev = Illustrations.revision
	_doc = TabletScript.parse(src)
	_page_span = {}
	var sp: PackedInt32Array = _doc["spoken"]
	for si in sp.size():
		var pg := int((_doc["words"][sp[si]] as Dictionary)["page"])
		var cur: Vector2i = _page_span.get(pg, Vector2i(si, si))
		_page_span[pg] = Vector2i(mini(cur.x, si), maxi(cur.y, si))
	_char0 = PackedInt32Array()
	var c := 0
	for w in _doc["words"]:
		_char0.append(c)
		c += String((w as Dictionary)["text"]).length() + 1
	_lays = {}
	_reset_reading()
	print("ghost: tablet - %d pages, %d actions, %d spoken words" % [(_doc["pages"] as Array).size(),
		(_doc["actions"] as Array).size(), (_doc["spoken"] as PackedInt32Array).size()])


## The layout of page [param p] for orientation [param o] (0 portrait, 1 landscape).
func layout(p: int, o: int) -> TabletPage:
	var k := "%d|%d" % [p, o]
	if not _lays.has(k):
		var lp := TabletPage.new()
		var L := logical(o)
		lp.build(_doc, p, L.x, (L.y - TOP) * 2.4)
		_lays[k] = lp
	return _lays[k]


static func logical(o: int) -> Vector2:
	return Vector2(SW, SH) if o == 0 else Vector2(SH, SW)


func _texture_for(key: String) -> Texture2D:
	var path := Illustrations.path_for(key)
	if path.is_empty():
		return null
	if not _textures.has(path):
		var img := Image.load_from_file(path)
		_textures[path] = ImageTexture.create_from_image(img) if img != null and not img.is_empty() else null
	return _textures[path]


# --- the words -------------------------------------------------------------------------

## Map the spoken words onto the chapter's, as far as they have arrived - the book's
## sequential match with a resync window, against SPOKEN words only, so a skipped run of any
## length never pulls the pointer off.
func _extend_map() -> void:
	if _subs == null or not is_instance_valid(_subs):
		return
	var words: Array = _subs.words
	if words.size() < _map.size():
		_reset_reading()
	var spoken: PackedInt32Array = _doc["spoken"]
	var dw: Array = _doc["words"]
	for i in range(_map.size(), words.size()):
		var w: Dictionary = words[i]
		var n := TabletScript.norm(String(w.get("text", "")))
		var got := -1
		if n.is_empty():
			got = _map_j - 1
		elif not _map_rem.is_empty() and _map_rem.begins_with(n):
			got = _map_j - 1
			_map_rem = _map_rem.substr(n.length())
		else:
			for k in range(_map_j, mini(_map_j + 12, spoken.size())):
				var ln := String((dw[spoken[k]] as Dictionary)["norm"])
				if ln == n or (n.begins_with(ln) and ln.length() >= 2 and k == _map_j):
					got = k
					_map_rem = ""
					break
				if ln.begins_with(n) and n.length() >= 2:
					got = k
					_map_rem = ln.substr(n.length())
					break
			if got >= 0:
				_map_j = got + 1
		_map.append(got)
		if got >= 0:
			_st1[got] = float(w.get("t1", 0.0))
			if _st0[got] < 0.0:
				_st0[got] = float(w.get("t0", 0.0))


## The spoken index being read now and how far through it: `{si, wi, frac}`, or empty.
func _reading() -> Dictionary:
	if _subs == null or not is_instance_valid(_subs) or _map.is_empty():
		return {}
	var c: float = _subs.cursor()
	var i := clampi(int(floor(c)), 0, _map.size() - 1)
	var k := i
	while k > 0 and int(_map[k]) < 0:
		k -= 1
	var si := int(_map[k])
	if si < 0:
		return {}
	return {"si": si, "wi": int((_doc["spoken"] as PackedInt32Array)[si]),
		"frac": clampf(c - float(i), 0.0, 1.0) if k == i else 1.0}


# --- the schedule ----------------------------------------------------------------------

## Place every action group in the gap the voice left for it. A group runs at its own pace and
## FINISHES a beat before the next words, so any slack in the gap is spent before it starts -
## the hand rests, then acts, then the voice goes on. A gap shorter than the group (a dial set
## the rests short) squeezes it to fit rather than talking over it. A group whose words have
## not arrived yet is left for later.
func _build_schedule() -> void:
	_sched = []
	var actions: Array = _doc["actions"]
	var nS := _st0.size()
	var known := -1
	for i in nS:
		if _st1[i] >= 0.0:
			known = i
	var i := 0
	while i < actions.size():
		var n := int((actions[i] as Dictionary)["after"])
		var group: Array = []
		while i < actions.size() and int((actions[i] as Dictionary)["after"]) == n:
			group.append(actions[i])
			i += 1
		if n > 0 and known < 0 or n - 1 > known:
			break
		var prev := 0.0
		for k in range(mini(n - 1, nS - 1), -1, -1):
			if _st1[k] >= 0.0:
				prev = _st1[k]
				break
		var next := INF
		for k in range(n, nS):
			if _st0[k] >= 0.0:
				next = _st0[k]
				break
		var total := 0.0
		for a in group:
			total += float((a as Dictionary)["dur"])
		var start := prev + LEAD
		var s := 1.0
		if next < INF:
			var room := next - TAIL - start
			if room >= total:
				start = next - TAIL - total
			else:
				s = maxf(room, total * 0.25) / total
		var t := start
		# THE OPENING RUN WAKES AT ONCE: the intro is the tablet waking, so the wake starts with
		# the show, and the slack the intro leaves is spent on the home screen - the rest of
		# the run still ends a beat before the first word
		if n == 0 and not group.is_empty() and String((group[0] as Dictionary)["kind"]) == "wake" \
				and start > WAKE_AT:
			var wake: Dictionary = group[0]
			_sched.append({"a": wake, "t0": WAKE_AT, "s": 1.0})
			t = maxf(start + float(wake["dur"]) * s, WAKE_AT + float(wake["dur"]))
			group = group.slice(1)
		for a in group:
			_sched.append({"a": a, "t0": t, "s": s})
			t += float((a as Dictionary)["dur"]) * s


## Every quarter turn begun by [param t]: `[{from, to, u, ph}]`, `u` in the action's own
## nominal seconds.
func _rotations(t: float) -> Array:
	var out: Array = []
	for e in _sched:
		var a: Dictionary = e["a"]
		if t < float(e["t0"]):
			break
		if a["kind"] == "rotate":
			out.append({"from": 1 - int(a["to"]), "to": int(a["to"]), "ph": _ph(a),
				"u": (t - float(e["t0"])) / float(e["s"])})
	return out


## How far the CAMERA has turned toward landscape at [param t]: 0..1, eased.
func _turn_at(t: float) -> float:
	var v := 0.0
	for r in _rotations(t):
		var k := smoothstep(0.0, 1.0, float(r["u"]) / float((r["ph"] as Dictionary)["cam1"]))
		v = lerpf(float(r["from"]), float(r["to"]), k)
	return v


## Which layout the screen holds at [param t] - the new one from the middle of its dissolve.
func _orient_at(t: float) -> int:
	var o := 0
	for r in _rotations(t):
		var ph: Dictionary = r["ph"]
		o = int(r["to"]) if float(r["u"]) >= (float(ph["fade0"]) + float(ph["fade1"])) * 0.5 else int(r["from"])
	return o


## The dissolve under way at [param t]: `Vector3(from, to, k)`, or k < 0 for none.
func _fade_at(t: float) -> Vector3:
	var rs := _rotations(t)
	if rs.is_empty():
		return Vector3(0, 0, -1)
	var r: Dictionary = rs.back()
	var ph: Dictionary = r["ph"]
	var u := float(r["u"])
	if u <= float(ph["fade0"]) or u >= float(ph["fade1"]):
		return Vector3(0, 0, -1)
	return Vector3(float(r["from"]), float(r["to"]),
		smoothstep(float(ph["fade0"]), float(ph["fade1"]), u))


## When page [param p] comes on screen, or INF.
func _shown_at(p: int) -> float:
	for e in _sched:
		var a: Dictionary = e["a"]
		if int(a.get("page", -9)) == p:
			return float(e["t0"]) + float(_ph(a)["show"]) * float(e["s"])
	return INF


static func _ph(a: Dictionary) -> Dictionary:
	return TabletScript.phases(String(a["kind"]), String(a.get("text", "")), int(a.get("n", 0)),
		int(a.get("m", 0)))


static func _vh(o: int) -> float:
	return logical(o).y - TOP


## THE SCROLL, per page, as a list of moves in time order. Two kinds of move, as a hand makes
## them: a FLICK (fast at once, then gliding to rest) to reach a link about to be tapped, and a
## DRAG (eased in and out) for everything a reader does - keeping the line being read in view,
## and a skim, which drags down to each picture it passes, rests on it, and drags on to the next
## words read. A turn re-states the place as the same share of the new layout's length.
func _build_flings() -> void:
	_flings = {}
	var pages: Array = _doc["pages"]
	var spoken: PackedInt32Array = _doc["spoken"]
	var dw: Array = _doc["words"]
	for p in pages.size():
		var t_show := _shown_at(p)
		if t_show == INF:
			continue
		var ev: Array = []
		for si in spoken.size():
			if int((dw[spoken[si]] as Dictionary)["page"]) == p and _st0[si] >= 0.0:
				ev.append({"t": _st0[si] - 0.9, "kind": "read", "wi": spoken[si]})
		for e in _sched:
			var a: Dictionary = e["a"]
			var ph := _ph(a)
			if (a["kind"] == "link" or a["kind"] == "skim") and int(a["from"]) == p and a.has("word"):
				ev.append({"t": float(e["t0"]) + float(ph["scroll0"]) * float(e["s"]), "kind": a["kind"],
					"wi": int(a["word"]), "pics": a.get("pics", []), "depth": float(a.get("depth", 1.1)),
					"w": (float(ph["scroll1"]) - float(ph["scroll0"])) * float(e["s"])})
			elif a["kind"] == "rotate":
				ev.append({"t": float(e["t0"]) + (float(ph["fade0"]) + float(ph["fade1"])) * 0.5
					* float(e["s"]) + 0.001, "kind": "turn"})
		ev.sort_custom(func(x, y): return float(x["t"]) < float(y["t"]))
		var list: Array = []
		var cur := 0.0
		var o := _orient_at(t_show)
		var busy_until := -INF
		for x in ev:
			var t := maxf(float(x["t"]), t_show)
			if x["kind"] == "turn":
				var no := _orient_at(t)
				if no == o:
					continue
				var m0 := maxf(1.0, layout(p, o).height - _vh(o))
				var m1 := maxf(0.0, layout(p, no).height - _vh(no))
				cur = clampf(_eval(list, t) / m0 * m1, 0.0, m1)
				o = no
				list.append({"t": t, "from": cur, "to": cur, "dur": 0.0, "o": o})
				continue
			var lp := layout(p, o)
			var r: Rect2 = lp.word_rect.get(int(x["wi"]), Rect2())
			var vh := _vh(o)
			var mx := maxf(0.0, lp.height - vh)
			match String(x["kind"]):
				"read":
					if t < busy_until:
						continue
					if r.end.y > cur + vh * 0.8 or r.position.y < cur + vh * 0.02:
						cur = clampf(r.position.y - vh * 0.22, 0.0, mx)
						var from := _eval(list, t)
						var dur := _drag_time(absf(cur - from))
						list.append({"t": t, "from": from, "to": cur, "dur": dur, "drag": true, "o": o})
						busy_until = t + dur
				"skim":
					# stops: each picture centred, then the next words a little below the top
					var stops: Array = []
					for bi in x["pics"]:
						var pr: Rect2 = lp.block_rect.get(int(bi), Rect2())
						if pr.size.y > 0.0:
							stops.append(clampf(pr.get_center().y - vh * 0.5, 0.0, mx))
					if int(x["wi"]) >= 0:
						stops.append(clampf(r.position.y - vh * 0.22, 0.0, mx))
					else:
						# a glance before leaving: some way further down, wherever that is - a
						# screen for an unread page, a third of one on a real page, whose top is
						# the part worth looking at
						stops.append(clampf(_eval(list, t) + vh * float(x["depth"]), 0.0, mx))
					var win := float(x["w"])
					var seg := win / float(stops.size())
					for k in stops.size():
						var ts := t + seg * float(k)
						var last := k == stops.size() - 1
						var from := _eval(list, ts)
						if absf(float(stops[k]) - from) < 6.0:
							continue
						# a picture is dragged to and then LOOKED AT, longer than it took to reach; the
						# last stop takes its whole share
						var dur := seg if last else seg * 0.4
						list.append({"t": ts, "from": from, "to": stops[k], "dur": dur, "drag": true, "o": o})
					cur = float(stops.back())
					busy_until = t + win
				"link":
					var want := clampf(r.position.y - vh * 0.45, 0.0, mx)
					var from := _eval(list, t)
					if absf(want - from) < 8.0:
						continue
					var flicks := maxi(1, int(ceil(absf(want - from) / (vh * 1.1))))
					var win := float(x["w"])
					for f in flicks:
						var tf := t + win * float(f) / float(flicks)
						var to := lerpf(from, want, float(f + 1) / float(flicks))
						list.append({"t": tf, "from": _eval(list, tf), "to": to, "o": o,
							"dur": minf(_fling_time(absf(to - from) / float(flicks)), win / float(flicks) * 1.5)})
					cur = want
					busy_until = t + win
		_flings[p] = list


static func _fling_time(d: float) -> float:
	return clampf(0.45 + 0.22 * sqrt(d / 100.0), 0.55, 1.5)


## A reader's drag is unhurried: a second for a few lines, two for a screenful.
static func _drag_time(d: float) -> float:
	return clampf(0.9 + 0.3 * sqrt(d / 100.0), 1.1, 2.2)


## Position along the moves at [param t]. A flick is fast at once and glides to rest; a drag
## eases in and out.
static func _eval(list: Array, t: float) -> float:
	var pos := 0.0
	for f in list:
		var d: Dictionary = f
		if float(d["t"]) > t:
			break
		var dur := float(d["dur"])
		var u := 1.0 if dur <= 0.0 else clampf((t - float(d["t"])) / dur, 0.0, 1.0)
		var e := smoothstep(0.0, 1.0, u) if d.has("drag") else (1.0 - exp(-4.5 * u)) / (1.0 - exp(-4.5))
		pos = lerpf(float(d["from"]), float(d["to"]), e)
	return pos


## Page [param p]'s scroll at [param t], in the layout for orientation [param o] - during a
## dissolve both layouts are on screen, and the one not yet (or no longer) the page's own is
## given the same share of its length.
func scroll_of(p: int, t: float, o := -1) -> float:
	var list: Array = _flings.get(p, [])
	var pos := _eval(list, t)
	if o < 0:
		return pos
	var lo := _orient_at(_shown_at(p))
	for f in list:
		if float((f as Dictionary)["t"]) > t:
			break
		lo = int((f as Dictionary).get("o", lo))
	if lo == o:
		return pos
	var m0 := maxf(1.0, layout(p, lo).height - _vh(lo))
	var m1 := maxf(0.0, layout(p, o).height - _vh(o))
	return clampf(pos / m0 * m1, 0.0, m1)


# --- the screen, replayed ----------------------------------------------------------------

## The whole screen at [param t], from the start: which app, which tab, what the address bar
## says, the keyboard, the finger. Cheap - a chapter has tens of actions - and it means the
## screen never depends on what the last frame happened to show.
func _state_at(t: float) -> Dictionary:
	var st := {"on": 0.0, "app": "home", "open": 0.0, "tabs": [], "tab": 0, "bar": "",
		"editing": "", "typed": "", "kb": 0.0, "key": "", "key_k": 0.0, "touch": Vector2(-1, -1),
		"touch_k": 0.0, "load": -1.0, "busy": false, "hist": {}}
	st["turn"] = _turn_at(t)
	st["orient"] = _orient_at(t)
	st["fade"] = _fade_at(t)
	var o := int(st["orient"])
	for e in _sched:
		var t0 := float(e["t0"])
		if t < t0:
			break
		var a: Dictionary = e["a"]
		var kind := String(a["kind"])
		var text := String(a.get("text", ""))
		var ph := _ph(a)
		var u := (t - t0) / float(e["s"])
		if u < float(ph["end"]):
			# a skim is reading, not handling: the camera stays on the page for it - and so does
			# the thinking before a new tab, which is the page still being looked at
			if kind == "skim" or u < float(ph.get("think", -1.0)):
				st["skim"] = true
			else:
				st["busy"] = true
		match kind:
			"wake":
				st["on"] = smoothstep(float(ph["on0"]), float(ph["on1"]), u)
			"open":
				st["on"] = 1.0
				_touch(st, u, float(ph["tap"]), _icon_rect(o, 0, true).get_center())
				if u >= float(ph["open0"]):
					st["app"] = "browser"
					st["open"] = smoothstep(float(ph["open0"]), float(ph["open1"]), u)
					if (st["tabs"] as Array).is_empty():
						(st["tabs"] as Array).append(-2)
				_load(st, u, ph, int(a["page"]))
			"link":
				var from := int(a["from"])
				var lp := layout(from, o)
				var r: Rect2 = lp.word_rect.get(int(a["word"]), Rect2())
				# the link itself shades while it is pressed, as a link does under a finger
				if u >= float(ph["tap"]) - 0.05 and u < float(ph["tap"]) + PRESS_HOLD:
					st["press_word"] = int(a["word"])
					st["press_page"] = from
					st["press_k"] = 1.0 - clampf((u - float(ph["tap"])) / PRESS_HOLD, 0.0, 1.0) * 0.6
				var tap_t := t0 + float(ph["tap"]) * float(e["s"])
				_touch(st, u, float(ph["tap"]), Vector2(r.get_center().x,
					r.get_center().y - scroll_of(from, tap_t, o) + TOP))
				_load(st, u, ph, int(a["page"]))
			"type", "search":
				var was_editing := not String(st["editing"]).is_empty()
				var target := "search" if kind == "search" else "bar"
				if not was_editing:
					var at := _bar_rect(o).get_center()
					if kind == "search":
						var sp := int(a["from"])
						at = layout(sp, o).search_rect.get_center() + Vector2(0.0, TOP - scroll_of(sp, t, o))
					_touch(st, u, float(ph["tap"]), at)
				if u >= float(ph["edit"]) and u < float(ph["go"]):
					st["editing"] = target
					var ct := TabletScript.char_times(text)
					var tu := u - float(ph["chars0"])
					var n := 0
					while n < text.length() and tu >= ct[n]:
						n += 1
					st["typed"] = text.substr(0, n)
					var kb := 1.0 if was_editing else smoothstep(float(ph["edit"]), float(ph["kb1"]), u)
					st["kb"] = kb
					if n > 0 and u < float(ph["chars1"]) + 0.15:
						st["key"] = text[n - 1]
						st["key_k"] = 1.0 - clampf((tu - ct[n - 1]) / 0.18, 0.0, 1.0)
					elif u >= float(ph["go"]) - 0.25:
						st["key"] = "\n"
						st["key_k"] = 1.0 - clampf((u - float(ph["go"]) + 0.25) / 0.25, 0.0, 1.0)
				elif u >= float(ph["go"]):
					st["editing"] = ""
					st["typed"] = ""
					st["kb"] = 1.0 - smoothstep(float(ph["go"]), float(ph["kb0"]), u)
				_load(st, u, ph, int(a["page"]))
			"back":
				_touch(st, u, float(ph["tap"]), _back_at())
				if u >= float(ph["show"]):
					var tabs: Array = st["tabs"]
					var hist: Array = (st["hist"] as Dictionary).get(int(st["tab"]), [])
					if not hist.is_empty():
						hist.pop_back()
					tabs[int(st["tab"])] = int(a["page"])
			"tab":
				_touch(st, u, float(ph["tap"]), _plus_rect(o).get_center())
				if u >= float(ph["add"]):
					var tabs: Array = st["tabs"]
					tabs.append(-1)
					st["tab"] = tabs.size() - 1
					st["editing"] = "bar"
					st["typed"] = ""
					st["kb"] = smoothstep(float(ph["add"]), float(ph["add"]) + 0.5, u)
	return st


## A finger at [param at], landing at nominal second [param tap]: `touch_dt` is how far from the
## press this moment is (negative while the finger comes down), for the press and its ripple.
func _touch(st: Dictionary, u: float, tap: float, at: Vector2) -> void:
	if u >= tap - TOUCH_IN and u <= tap + TOUCH_OUT:
		st["touch"] = at
		st["touch_dt"] = u - tap
		st["touch_k"] = 1.0


## The load bar, and the page arriving at `show`.
func _load(st: Dictionary, u: float, ph: Dictionary, page: int) -> void:
	var tabs: Array = st["tabs"]
	if tabs.is_empty():
		tabs.append(-2)
	if u >= float(ph["load0"]) and u < float(ph["load1"]):
		st["load"] = inverse_lerp(float(ph["load0"]), float(ph["load1"]), u)
	if u >= float(ph["show"]):
		var was := int(tabs[int(st["tab"])])
		if was >= 0 and was != page:
			var hd: Dictionary = st["hist"]
			if not hd.has(int(st["tab"])):
				hd[int(st["tab"])] = []
			(hd[int(st["tab"])] as Array).append(was)
		tabs[int(st["tab"])] = page
	elif u >= float(ph["load0"]) and int(tabs[int(st["tab"])]) == -1:
		tabs[int(st["tab"])] = -2


# --- geometry of the screen, in logical pixels -----------------------------------------------

## The back chevron, in logical pixels.
static func _back_at() -> Vector2:
	return Vector2(44.0, STATUS + TABS + BAR * 0.5)


func _bar_rect(o: int) -> Rect2:
	var L := logical(o)
	return Rect2(150.0, STATUS + TABS + 10.0, L.x - 300.0, BAR - 20.0)


func _plus_rect(o: int) -> Rect2:
	var L := logical(o)
	return Rect2(L.x - 74.0, STATUS + 9.0, 46.0, 46.0)


## Home screen icon [param i] (in the dock when [param dock]).
func _icon_rect(o: int, i: int, dock: bool) -> Rect2:
	var L := logical(o)
	var s := 128.0
	if dock:
		var n := 4
		var gap := 52.0
		var w := n * s + (n - 1) * gap
		return Rect2((L.x - w) * 0.5 + i * (s + gap), L.y - 64.0 - s, s, s)
	var cols := 4 if o == 0 else 6
	var gx := (L.x - cols * s) / float(cols + 1)
	var row := i / cols
	var col := i % cols
	return Rect2(gx + col * (s + gx), 130.0 + row * (s + 90.0), s, s)


## Logical canvas -> screen texture for a layout in orientation [param o]: portrait as it is,
## landscape a quarter turn round, upright to a camera stood at the slab's side. ALWAYS FULL
## SCREEN: the content never turns or shrinks on the glass - a first cut counter-rotated it
## against the camera, and it read as the screen coming off the tablet and going back on.
static func content_xf(o: int) -> Transform2D:
	if o == 0:
		return Transform2D.IDENTITY
	return Transform2D(Vector2(0.0, -1.0), Vector2(1.0, 0.0), Vector2(0.0, SH))


## A logical point as a point on the desk.
func _world_of(p: Vector2) -> Vector3:
	var tp := content_xf(int(_st.get("orient", 0))) * p
	return Vector3((tp.x / SW - 0.5) * SCREEN.x, SLAB_T * 0.5, (tp.y / SH - 0.5) * SCREEN.y)


# --- the camera ----------------------------------------------------------------------------

func _sev() -> float:
	return clampf(Director.camera, Director.CAMERA_MIN, Director.CAMERA_MAX)


## THE JOURNAL'S CAMERA, NOT A SWITCH. Each page has an ARC: it opens wide on the whole slab,
## closes in on the reading as the reading gets under way, and lets go - back to wide - over
## the last of the page's words, before the hand does the next thing ([method _arc]). A page
## with little on it to read barely closes in. Every channel is a slow critically damped
## spring, so a change of target is a pressure that builds and eases, never a move; the first
## cut toggled close/wide on every tap and every pause and read as "floaty... bouncy".
##
## THE SLAB IS NEVER SQUARE TO THE FRAME. Each page sets its own small twist (yaw and roll) and
## tilt, reached through springs several times slower still, with a wander of minutes on top -
## all far smaller than the book's, because a tablet turned much reads as wrong. Tilt is paid for
## in DISTANCE: the slab foreshortens by the sine of the pitch, so the camera stands closer by
## the same factor and the slab keeps its size in the frame.
##
## FAST ONLY ON A CONTEXT SWITCH - a new page, a new tab, the app opening, a turn - where the
## springs quicken for a few seconds and relax again ([member _quick]).
##
## The quarter turn itself is the schedule's eased curve, unsprung: it is a move the hand
## makes, and has its own duration.
func _tick_camera(delta: float) -> void:
	var t := _now
	var sev := _sev()
	var turn := float(_st.get("turn", 0.0))
	var o := int(_st.get("orient", 0))
	var page := _current_page()
	var busy := bool(_st.get("busy", false))
	# a context switch quickens everything for a few seconds
	var ctx := "%s|%d|%d|%d" % [_st.get("app", ""), page, int(_st.get("tab", 0)), o]
	if ctx != _ctx:
		_ctx = ctx
		_quick = 1.0
	_quick = maxf(0.0, _quick - delta / QUICK_TIME)
	var k := 0.0 if busy or page < 0 else _arc(page)
	# a REAL page is not read, it is looked at: part way in, on its upper half, for as long as
	# the hand lingers there
	var real := page >= 0 and bool((_doc["pages"][page] as Dictionary).get("real", false))
	if real and not busy:
		k = REAL_LOOK
	# where on the screen the reading is: the line the voice will reach a little ahead,
	# low-passed so line-by-line steps become a drift
	var L := logical(o)
	var line := TOP + _vh(o) * 0.45
	if real:
		line = TOP + _vh(o) * 0.35
	elif page >= 0 and not busy and not bool(_st.get("skim", false)):
		var wi := _camera_word(page)
		var lp := layout(page, o)
		if wi >= 0 and lp.word_rect.has(wi):
			# THE EYES STAY PUT, THE PAGE MOVES. The aim is held in the middle band of the screen
			# and only leans toward the line within it: the reading drags keep the line in that
			# band, so a scroll brings the next lines to where the camera already looks. Chasing
			# the line itself sent the camera up the slab after every scroll - "floaty".
			line = clampf((lp.word_rect[wi] as Rect2).get_center().y - scroll_of(page, t, o) + TOP,
				TOP + _vh(o) * AIM_BAND.x, TOP + _vh(o) * AIM_BAND.y)
	if _snap or _quick > 0.98:
		_line_y = line
	_line_y = lerpf(_line_y, line, 1.0 - exp(-maxf(delta, 0.0) / LINE_TAU))
	var aim := Vector3(0.0, 0.0, 0.02).lerp(_world_of(Vector2(L.x * 0.5, _line_y)), k)
	var near := lerpf(wide_of(turn), lerpf(NEAR.x, NEAR.y, turn), clampf(0.55 + 0.3 * sev, 0.0, 1.0))
	var dist := lerpf(wide_of(turn), near, k) * (1.0 + 0.012 * sin(t * 0.033 + 2.1) * sev)
	var pitch := lerpf(PITCH_WIDE, PITCH_NEAR, k)
	# this page's own small set-up, and a wander of minutes on top
	var pk := page + 7
	var yaw := (_hash01(["yaw", pk]) - 0.5) * 2.0 * YAW_SPREAD + sin(t * 0.019 + float(_seed % 97)) * YAW_WANDER * sev
	var roll := (_hash01(["roll", pk]) - 0.5) * 2.0 * ROLL_SPREAD + sin(t * 0.043 + 0.7) * ROLL_WANDER * sev
	var tilt := (_hash01(["tilt", pk]) - 0.5) * 2.0 * TILT_SPREAD + sin(t * 0.031 + 2.4) * TILT_WANDER * sev
	if _snap:
		_snap = false
		_c_aim = aim
		_c_dist = dist
		_c_pitch = pitch
		_c_wan = yaw
		_c_roll = roll
		_c_tilt = tilt
	var tau := lerpf(FRAME_TAU, QUICK_TAU, _quick) / lerpf(0.75, 1.25, clampf(sev * 0.5, 0.0, 1.0))
	var steps := maxi(1, int(ceil(delta / (1.0 / 60.0))))
	var h := delta / float(steps)
	for _i in steps:
		var ra := BookMedium._spring3(_c_aim, _v_aim, aim, tau, h)
		_c_aim = ra[0]
		_v_aim = ra[1]
		var rd := _spring(_c_dist, _v_dist, dist, tau, h)
		_c_dist = rd.x
		_v_dist = rd.y
		var rp := _spring(_c_pitch, _v_pitch, pitch, tau, h)
		_c_pitch = rp.x
		_v_pitch = rp.y
		var rw := _spring(_c_wan, _v_wan, yaw, tau * ANGLE_SLOW, h)
		_c_wan = rw.x
		_v_wan = rw.y
		var rr := _spring(_c_roll, _v_roll, roll, tau * ANGLE_SLOW, h)
		_c_roll = rr.x
		_v_roll = rr.y
		var rt := _spring(_c_tilt, _v_tilt, tilt, tau * ANGLE_SLOW, h)
		_c_tilt = rt.x
		_v_tilt = rt.y
	var p := deg_to_rad(_c_pitch + _c_tilt)
	# tilt is paid for in distance: closer as the slab foreshortens, so it keeps its size
	var d := _c_dist * sin(p) / sin(deg_to_rad(_c_pitch))
	var y := deg_to_rad(turn * 90.0 + _c_wan)
	var dir := Vector3(sin(y) * cos(p), sin(p), cos(y) * cos(p))
	_cam.position = _c_aim + dir * d
	_cam.look_at(_c_aim, Vector3.UP)
	_cam.rotate_object_local(Vector3.FORWARD, deg_to_rad(_c_roll))
	if not _cam.current:
		_cam.make_current()


static func wide_of(turn: float) -> float:
	return lerpf(WIDE.x, WIDE.y, turn)


## How far into page [param page]'s arc the reading is: 0 wide, 1 close. A pure function of how
## much of the page's text has been read, like the book's spread arc, scaled down for a page
## with little to read - a results page is glanced at, not settled into.
func _arc(page: int) -> float:
	var span: Vector2i = _page_span.get(page, Vector2i(-1, -1))
	if span.x < 0:
		return 0.0
	var r := _reading()
	if r.is_empty():
		return 0.0
	var si := int(r["si"])
	var n := float(span.y - span.x + 1)
	var p := clampf((float(si - span.x) + float(r["frac"])) / n, 0.0, 1.0)
	return smoothstep(0.0, ARC_IN, p) * (1.0 - smoothstep(1.0 - ARC_OUT, 1.0, p)) \
		* clampf(n / ARC_WORDS, 0.25, 1.0)


## The word the voice will reach [constant CAM_LEAD] seconds from now, on this page - the
## springs make the camera late, and the lead pays for it, as in the book.
func _camera_word(page: int) -> int:
	var r := _reading()
	if r.is_empty():
		return -1
	var spoken: PackedInt32Array = _doc["spoken"]
	var dw: Array = _doc["words"]
	var wi := int(r["wi"])
	if int((dw[wi] as Dictionary)["page"]) != page:
		return -1
	for k in range(int(r["si"]) + 1, spoken.size()):
		if _st0[k] < 0.0 or _st0[k] > _now + CAM_LEAD:
			break
		if int((dw[spoken[k]] as Dictionary)["page"]) != page:
			break
		wi = spoken[k]
	return wi


static func _spring(x: float, v: float, target: float, tau: float, h: float) -> Vector2:
	return BookMedium._spring(x, v, target, tau, h)


func _current_page() -> int:
	if _st.get("app", "home") != "browser":
		return -3
	var tabs: Array = _st.get("tabs", [])
	if tabs.is_empty():
		return -2
	return int(tabs[clampi(int(_st["tab"]), 0, tabs.size() - 1)])


# --- drawing the screen ---------------------------------------------------------------------

## Draw the screen. [param layer] 0 is the screen as it is; 1 is the INCOMING layout of a turn,
## drawn on a second canvas faded in over the first - a dissolve, so the glass is always full.
func draw_screen(ci: CanvasItem, layer := 0) -> void:
	var fade: Vector3 = _st.get("fade", Vector3(0, 0, -1))
	if layer == 1 and fade.z < 0.0:
		return
	ci.draw_rect(Rect2(0.0, 0.0, SW, SH), Color.BLACK)
	var on := float(_st.get("on", 0.0))
	if on <= 0.001:
		return
	var o := int(_st.get("orient", 0))
	if fade.z >= 0.0:
		o = int(fade.y) if layer == 1 else int(fade.x)
	var L := logical(o)
	var xf := content_xf(o)
	ci.draw_set_transform_matrix(xf)
	_draw_home(ci, o)
	if _st["app"] == "browser":
		var k := float(_st["open"])
		if k < 1.0:
			# the app grows out of its icon
			var ir := _icon_rect(o, 0, true)
			var sc := lerpf(ir.size.x / L.x, 1.0, k)
			var at := ir.get_center().lerp(L * 0.5, k)
			ci.draw_set_transform_matrix(xf * Transform2D(0.0, Vector2(sc, sc), 0.0, at - L * 0.5 * sc))
		_draw_browser(ci, o)
		ci.draw_set_transform_matrix(xf)
	_draw_status(ci, o, _st["app"] == "browser" and float(_st["open"]) > 0.5)
	_draw_keyboard(ci, o)
	if float(_st["touch_k"]) > 0.0:
		_draw_touch(ci, _st["touch"], float(_st.get("touch_dt", 0.0)))
	ci.draw_set_transform_matrix(Transform2D.IDENTITY)
	if on < 1.0:
		ci.draw_rect(Rect2(0.0, 0.0, SW, SH), Color(0, 0, 0, 1.0 - on))


## THE TAP, made to read on a white page: a dark press that grows as the finger comes down, and
## on the press a ring of the system blue that spreads and fades - the first cut was a pale grey
## dot that vanished against every page it was used on.
func _draw_touch(ci: CanvasItem, at: Vector2, dt: float) -> void:
	var blue := Color(0.16, 0.42, 1.0)
	if dt < 0.0:
		var k := 1.0 - (-dt) / TOUCH_IN
		ci.draw_circle(at, lerpf(20.0, 38.0, k), Color(0.08, 0.1, 0.16, 0.55 * k))
		return
	var f := clampf(dt / TOUCH_OUT, 0.0, 1.0)
	ci.draw_circle(at, 38.0, Color(0.08, 0.1, 0.16, 0.55 * (1.0 - smoothstep(0.0, 0.5, f))))
	ci.draw_circle(at, lerpf(38.0, 90.0, f), Color(blue, 0.25 * (1.0 - f)))
	# sized for the picture, not the page: the slab is a third of the frame, so the ring is bold
	ci.draw_arc(at, lerpf(40.0, 150.0, sqrt(f)), 0.0, TAU, 64, Color(blue, 1.0 - f * f), lerpf(11.0, 5.0, f), true)


## Shade the link under a pressing finger: every word of it, in its own colour, softly.
func _draw_press(ci: CanvasItem, page: int, o: int) -> void:
	var wi := int(_st.get("press_word", -1))
	if wi < 0:
		return
	var dw: Array = _doc["words"]
	var link := String((dw[wi] as Dictionary)["link"])
	var lp := layout(page, o)
	var dy := TOP - scroll_of(page, _now, o)
	var k := float(_st.get("press_k", 0.0))
	var i := wi
	while i < dw.size() and String((dw[i] as Dictionary)["link"]) == link \
			and int((dw[i] as Dictionary)["page"]) == page and lp.word_rect.has(i):
		var r: Rect2 = lp.word_rect[i]
		ci.draw_style_box(_box(Color(lp.accent, 0.34 * k), 10), Rect2(r.position + Vector2(-4.0, dy), r.size + Vector2(8.0, 0.0)))
		i += 1


func _hash01(salt) -> float:
	return float(hash([_seed, salt]) & 0xFFFF) / 65535.0


func _draw_home(ci: CanvasItem, o: int) -> void:
	var L := logical(o)
	var h0 := _hash01("wall")
	var c0 := Color.from_hsv(h0, 0.55, 0.42)
	var c1 := Color.from_hsv(fposmod(h0 + 0.18, 1.0), 0.6, 0.18)
	var bands := 24
	for i in bands:
		var y0 := L.y * float(i) / bands
		ci.draw_rect(Rect2(0.0, y0, L.x, L.y / bands + 1.0), c0.lerp(c1, float(i) / (bands - 1)))
	for b in 3:
		var bc := Vector2(_hash01("bx%d" % b) * L.x, _hash01("by%d" % b) * L.y)
		var br := L.x * lerpf(0.25, 0.5, _hash01("br%d" % b))
		var col := Color.from_hsv(fposmod(h0 + 0.5 * _hash01("bh%d" % b), 1.0), 0.5, 0.8)
		for k in 8:
			ci.draw_circle(bc, br * (1.0 - float(k) / 8.0), Color(col, 0.025))
	var f := TabletPage.face(false, 0)
	for i in 16:
		_draw_icon(ci, _icon_rect(o, i, false), 7 + i, _app_name(i), f)
	var d0 := _icon_rect(o, 0, true)
	var d3 := _icon_rect(o, 3, true)
	var dock := Rect2(d0.position - Vector2(26, 26), Vector2(d3.end.x - d0.position.x + 52, d0.size.y + 52))
	ci.draw_style_box(_box(Color(1, 1, 1, 0.22), 46), dock)
	for i in 4:
		_draw_icon(ci, _icon_rect(o, i, true), i, "", f)


## A made-up app name: two or three syllables, from the seed. Not words, on purpose.
func _app_name(i: int) -> String:
	var cons := "bcdfghklmnprstvz"
	var vow := "aeiou"
	var r := RandomNumberGenerator.new()
	r.seed = hash([_seed, "app", i])
	var s := ""
	for k in r.randi_range(2, 3):
		s += cons[r.randi() % cons.length()] + vow[r.randi() % vow.length()]
	return s.capitalize()


func _draw_icon(ci: CanvasItem, r: Rect2, i: int, label: String, f: Font) -> void:
	var browser := i == 0
	var hue := 0.58 if browser else _hash01("ih%d" % i)
	var col := Color.from_hsv(hue, 0.65, 0.85)
	ci.draw_style_box(_box(col, 30), r)
	ci.draw_style_box(_box(Color(1, 1, 1, 0.14), 30), Rect2(r.position, Vector2(r.size.x, r.size.y * 0.5)))
	var c := r.get_center()
	var s := r.size.x * 0.28
	var w := Color(1, 1, 1, 0.92)
	if browser:
		# a compass
		ci.draw_arc(c, s * 1.15, 0.0, TAU, 40, w, 5.0, true)
		ci.draw_colored_polygon(PackedVector2Array([c + Vector2(s * 0.7, -s * 0.7), c + Vector2(-s * 0.18, -s * 0.18),
			c + Vector2(s * 0.18, s * 0.18)]), Color(1.0, 0.3, 0.25))
		ci.draw_colored_polygon(PackedVector2Array([c + Vector2(-s * 0.7, s * 0.7), c + Vector2(s * 0.18, s * 0.18),
			c + Vector2(-s * 0.18, -s * 0.18)]), w)
	else:
		match hash([_seed, "glyph", i]) % 6:
			0: ci.draw_circle(c, s, w)
			1: ci.draw_arc(c, s, 0.0, TAU, 32, w, 8.0, true)
			2: ci.draw_rect(Rect2(c - Vector2(s, s) * 0.8, Vector2(s, s) * 1.6), w)
			3: ci.draw_colored_polygon(PackedVector2Array([c + Vector2(0, -s), c + Vector2(s, s * 0.8), c + Vector2(-s, s * 0.8)]), w)
			4:
				for k in 3:
					ci.draw_rect(Rect2(c.x - s + k * s * 0.75, c.y - s * (0.3 + 0.35 * k), s * 0.5, s * (0.3 + 0.35 * k) * 2.0), w)
			5:
				for k in 3:
					ci.draw_circle(c + Vector2((k - 1) * s * 0.8, 0.0), s * 0.28, w)
	if not label.is_empty():
		ci.draw_string(f, Vector2(r.position.x - 20.0, r.end.y + 34.0), label, HORIZONTAL_ALIGNMENT_CENTER,
			r.size.x + 40.0, 22, Color(1, 1, 1, 0.92))


var _boxes := {}

func _box(col: Color, radius: int) -> StyleBoxFlat:
	var k := "%s|%d" % [col.to_html(), radius]
	if not _boxes.has(k):
		var b := StyleBoxFlat.new()
		b.bg_color = col
		b.set_corner_radius_all(radius)
		b.anti_aliasing = true
		_boxes[k] = b
	return _boxes[k]


func _draw_status(ci: CanvasItem, o: int, light: bool) -> void:
	var L := logical(o)
	var col := Color(0.1, 0.1, 0.12) if light else Color.WHITE
	var f := TabletPage.face(false, 2)
	var mins := 21 * 60 + int(_hash01("clock") * 300.0) + int(_now / 60.0)
	ci.draw_string(f, Vector2(34.0, 30.0), "%d:%02d" % [(mins / 60) % 24, mins % 60],
		HORIZONTAL_ALIGNMENT_LEFT, -1, 24, col)
	var pct := clampi(int(lerpf(34.0, 88.0, _hash01("batt")) - _now / 240.0), 5, 100)
	var bx := Rect2(L.x - 74.0, 12.0, 44.0, 20.0)
	ci.draw_rect(bx, Color(col, 0.9), false, 2.0)
	ci.draw_rect(Rect2(bx.end.x + 1.0, 18.0, 3.0, 8.0), col)
	ci.draw_rect(Rect2(bx.position + Vector2(3, 3), Vector2((bx.size.x - 6.0) * pct / 100.0, bx.size.y - 6.0)), col)
	ci.draw_string(f, Vector2(L.x - 210.0, 30.0), "%d%%" % pct, HORIZONTAL_ALIGNMENT_RIGHT, 120, 22, col)
	# wifi
	var wc := Vector2(L.x - 290.0, 30.0)
	for k in 3:
		ci.draw_arc(wc, 6.0 + k * 7.0, -PI * 0.75, -PI * 0.25, 10, col, 3.0, true)


func _draw_browser(ci: CanvasItem, o: int) -> void:
	var L := logical(o)
	var page := _current_page()
	var chrome := Color(0.95, 0.95, 0.96)
	var ink := Color(0.12, 0.12, 0.14)
	ci.draw_rect(Rect2(Vector2.ZERO, L), Color.WHITE)
	# the page
	if page >= 0:
		var lp := layout(page, o)
		lp.draw(ci, TOP, scroll_of(page, _now, o), TOP, L.y, _lit(page), _now, _texture_for)
		if int(_st.get("press_page", -9)) == page:
			_draw_press(ci, page, o)
	elif page == -1:
		_draw_start(ci, o)
	# the chrome over it
	ci.draw_rect(Rect2(0.0, 0.0, L.x, TOP), chrome)
	ci.draw_line(Vector2(0.0, TOP), Vector2(L.x, TOP), Color(0, 0, 0, 0.12), 2.0)
	var tabs: Array = _st["tabs"]
	var n := maxi(1, tabs.size())
	var tw := minf(330.0, (L.x - 140.0) / n)
	var f := TabletPage.face(false, 0)
	for i in tabs.size():
		var r := Rect2(24.0 + i * (tw + 8.0), STATUS + 8.0, tw, TABS - 10.0)
		var active := i == int(_st["tab"])
		ci.draw_style_box(_box(Color.WHITE if active else Color(0.88, 0.88, 0.9), 14), r)
		ci.draw_string(f, r.position + Vector2(20.0, 33.0), _fit(f, _tab_title(int(tabs[i])), 20, r.size.x - 40.0),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 20, Color(ink, 0.9 if active else 0.55))
	var pr := _plus_rect(o)
	ci.draw_line(pr.get_center() - Vector2(13, 0), pr.get_center() + Vector2(13, 0), ink, 3.0, true)
	ci.draw_line(pr.get_center() - Vector2(0, 13), pr.get_center() + Vector2(0, 13), ink, 3.0, true)
	# back / forward: back is live when there is somewhere to go back to
	var by := STATUS + TABS + BAR * 0.5
	var can_back := not ((_st["hist"] as Dictionary).get(int(_st["tab"]), []) as Array).is_empty()
	for k in 2:
		var x := 44.0 + k * 52.0
		var d := -1.0 if k == 0 else 1.0
		var live := k == 0 and can_back
		ci.draw_polyline(PackedVector2Array([Vector2(x - 8.0 * d, by - 14.0), Vector2(x + 8.0 * d, by),
			Vector2(x - 8.0 * d, by + 14.0)]), Color(ink, 0.8 if live else 0.25), 4.0, true)
	var br := _bar_rect(o)
	var editing := String(_st["editing"]) == "bar"
	ci.draw_style_box(_box(Color(0.87, 0.87, 0.89) if not editing else Color.WHITE, 18), br)
	if editing:
		ci.draw_rect(br, Color(0.25, 0.5, 1.0, 0.7), false, 3.0)
		var typed := String(_st["typed"])
		var tx := br.position + Vector2(26.0, 44.0)
		ci.draw_string(f, tx, typed, HORIZONTAL_ALIGNMENT_LEFT, -1, 28, ink)
		if fmod(_now, 1.0) < 0.6:
			var cx := tx.x + f.get_string_size(typed, HORIZONTAL_ALIGNMENT_LEFT, -1, 28).x + 3.0
			ci.draw_line(Vector2(cx, br.position.y + 16.0), Vector2(cx, br.end.y - 16.0), Color(0.25, 0.5, 1.0), 3.0)
	else:
		var url := _shown_url(page)
		if not url.is_empty():
			var w := f.get_string_size(url, HORIZONTAL_ALIGNMENT_LEFT, -1, 26).x
			var x := br.get_center().x - w * 0.5
			# a padlock
			ci.draw_rect(Rect2(x - 34.0, by - 4.0, 18.0, 15.0), Color(ink, 0.6))
			ci.draw_arc(Vector2(x - 25.0, by - 4.0), 6.0, PI, TAU, 12, Color(ink, 0.6), 2.5, true)
			ci.draw_string(f, Vector2(x, by + 10.0), url, HORIZONTAL_ALIGNMENT_LEFT, -1, 26, ink)
	var ld := float(_st["load"])
	if ld >= 0.0:
		ci.draw_rect(Rect2(0.0, TOP - 4.0, L.x * (1.0 - pow(1.0 - ld, 2.0)), 4.0), Color(0.2, 0.48, 1.0))
	# a search being typed shows in the page's box
	if String(_st["editing"]) == "search" and page >= 0:
		var lp := layout(page, o)
		var sr := lp.search_rect
		var y := sr.position.y + TOP - scroll_of(page, _now, o)
		var r := Rect2(sr.position.x, y, sr.size.x, sr.size.y)
		ci.draw_rect(r.grow(-6.0), Color.WHITE)
		ci.draw_style_box(_ring(), r)
		ci.draw_string(f, Vector2(r.position.x + 74.0, r.get_center().y + 11.0), String(_st["typed"]),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 30, ink)


var _ring_box: StyleBoxFlat

func _ring() -> StyleBoxFlat:
	if _ring_box == null:
		_ring_box = StyleBoxFlat.new()
		_ring_box.draw_center = false
		_ring_box.border_color = Color(0.25, 0.5, 1.0)
		_ring_box.set_border_width_all(3)
		_ring_box.set_corner_radius_all(38)
		_ring_box.anti_aliasing = true
	return _ring_box


static func _fit(f: Font, s: String, fs: int, w: float) -> String:
	if f.get_string_size(s, HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x <= w:
		return s
	while s.length() > 1 and f.get_string_size(s + "…", HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x > w:
		s = s.substr(0, s.length() - 1)
	return s + "…"


func _tab_title(p: int) -> String:
	if p == -1:
		return "New Tab"
	if p < 0:
		return "Loading…"
	return layout(p, 0).title


func _shown_url(p: int) -> String:
	if p < 0:
		return ""
	return TabletScript.host_of(String((_doc["pages"][p] as Dictionary)["url"]))


func _draw_start(ci: CanvasItem, o: int) -> void:
	var L := logical(o)
	ci.draw_rect(Rect2(0.0, TOP, L.x, L.y - TOP), Color(0.97, 0.97, 0.98))
	var f := TabletPage.face(false, 2)
	ci.draw_string(f, Vector2(70.0, TOP + 90.0), "Favourites", HORIZONTAL_ALIGNMENT_LEFT, -1, 34,
		Color(0.12, 0.12, 0.14))
	var cols := 6 if o == 0 else 8
	var s := 110.0
	var gap := (L.x - 140.0 - cols * s) / float(cols - 1)
	for i in cols * 2:
		var r := Rect2(70.0 + (i % cols) * (s + gap), TOP + 140.0 + (i / cols) * (s + 80.0), s, s)
		ci.draw_style_box(_box(Color.from_hsv(_hash01("fav%d" % i), 0.35, 0.92), 22), r)
		TabletPage.squiggle(ci, Vector2(r.position.x + 14.0, r.end.y + 30.0), s - 28.0, 6.0,
			Color(0.5, 0.5, 0.55), hash([_seed, "fv", i]))


## THE BOOK'S HIGHLIGHT, letter by letter: what [method TabletPage.draw] needs to colour the
## words of [param page] as the voice reaches them - the word being said and how far through
## it, and when each word on the page already said was spoken, so its colour can cool.
## Words long cooled are left out; they are ink again.
func _lit(page: int) -> Dictionary:
	var r := _reading()
	if r.is_empty():
		return {}
	var spoken: PackedInt32Array = _doc["spoken"]
	var si := int(r["si"])
	var times := {}
	for k in range(si, -1, -1):
		if _st1[k] >= 0.0 and _now - _st1[k] > BookMedium.TRAIL_TAU * 4.0:
			break
		var wi := spoken[k]
		if int((_doc["words"][wi] as Dictionary)["page"]) != page or _st0[k] < 0.0:
			continue
		times[wi] = Vector2(_st0[k], _st1[k])
	return {"cur": int(r["wi"]), "frac": float(r["frac"]), "now": _now, "times": times,
		"char0": _char0, "hue0": _hue0}


const KEYS := ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

func _key_rect(o: int, top: float, row: int, col: int, n: int) -> Rect2:
	var L := logical(o)
	var kw := L.x / 10.8
	var kh := (L.y - top - 30.0) / 4.0
	var x0 := (L.x - n * kw) * 0.5
	return Rect2(x0 + col * kw + 5.0, top + 14.0 + row * kh + 5.0, kw - 10.0, kh - 10.0)


func _draw_keyboard(ci: CanvasItem, o: int) -> void:
	var k := float(_st["kb"])
	if k <= 0.0:
		return
	var L := logical(o)
	var h := L.y * (0.34 if o == 0 else 0.44)
	var top := L.y - h * k
	ci.draw_rect(Rect2(0.0, top, L.x, h), Color(0.8, 0.81, 0.84))
	var f := TabletPage.face(false, 0)
	var press := String(_st["key"]).to_lower()
	var pk := float(_st["key_k"])
	for row in 3:
		var keys: String = KEYS[row]
		for col in keys.length():
			var r := _key_rect(o, top, row, col, keys.length())
			var down := press == keys[col] and pk > 0.0
			ci.draw_style_box(_box(Color(0.62, 0.64, 0.68) if down else Color.WHITE, 10), r)
			ci.draw_string(f, r.position + Vector2(0.0, r.size.y * 0.66), keys[col], HORIZONTAL_ALIGNMENT_CENTER,
				r.size.x, 30, Color(0.1, 0.1, 0.12))
			if down:
				# the pop-up a key shows under the finger
				var pop := Rect2(r.position - Vector2(10.0, r.size.y * 1.3), r.size + Vector2(20.0, r.size.y * 0.3))
				ci.draw_style_box(_box(Color.WHITE, 14), pop)
				ci.draw_string(f, pop.position + Vector2(0.0, pop.size.y * 0.7), keys[col],
					HORIZONTAL_ALIGNMENT_CENTER, pop.size.x, 44, Color(0.1, 0.1, 0.12))
	var space := _key_rect(o, top, 3, 2, 10)
	space.size.x = space.size.x * 5.0 + 40.0
	ci.draw_style_box(_box(Color(0.62, 0.64, 0.68) if press == " " and pk > 0.0 else Color.WHITE, 10), space)
	var go := _key_rect(o, top, 3, 8, 10)
	go.size.x = go.size.x * 2.0 + 10.0
	ci.draw_style_box(_box(Color(0.15, 0.4, 0.95) if press == "\n" and pk > 0.0 else Color(0.25, 0.5, 1.0), 10), go)
	ci.draw_string(f, go.position + Vector2(0.0, go.size.y * 0.66), "go", HORIZONTAL_ALIGNMENT_CENTER,
		go.size.x, 28, Color.WHITE)
	var dot := _key_rect(o, top, 3, 7, 10)
	ci.draw_style_box(_box(Color(0.62, 0.64, 0.68) if press == "." and pk > 0.0 else Color.WHITE, 10), dot)
	ci.draw_string(f, dot.position + Vector2(0.0, dot.size.y * 0.66), ".", HORIZONTAL_ALIGNMENT_CENTER,
		dot.size.x, 30, Color(0.1, 0.1, 0.12))
	for c in 2:
		var mr := _key_rect(o, top, 3, c, 10)
		ci.draw_style_box(_box(Color(0.68, 0.7, 0.74), 10), mr)


## Debug line for probes.
func debug_line() -> String:
	return "cam d %.3f p %.2f yaw %.2f roll %.2f tilt %.2f aim %.3f,%.3f | " % [_c_dist, _c_pitch,
		_c_wan, _c_roll, _c_tilt, _c_aim.x, _c_aim.z] + "app %s tab %s page %d orient %d turn %.2f scroll %.0f kb %.2f typed '%s' reading %s" % [
		_st.get("app", "?"), str(_st.get("tabs", [])), _current_page(), int(_st.get("orient", 0)),
		float(_st.get("turn", 0.0)), scroll_of(_current_page(), _now, int(_st.get("orient", 0))), float(_st.get("kb", 0.0)),
		String(_st.get("typed", "")), str(_reading())]


## The screen's drawing surface: holds nothing, the medium draws.
class ScreenCanvas:
	extends Node2D
	var tablet = null
	var layer := 0

	func _draw() -> void:
		if tablet != null:
			tablet.draw_screen(self, layer)
