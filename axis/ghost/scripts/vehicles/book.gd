extends Vehicle
class_name BookVehicle

## BookVehicle - the reading as a printed novel, open on a desk, with the words lit as they
## are spoken.
##
## Where the comic draws SCENES onto pages, this draws the TEXT: the chapter is typeset once
## into real pages ([BookLayout]) and the narration is followed across them by a soft swash
## under the word being said - a finger moving along a line - so the picture draws the viewer
## into the story rather than into the video. It replaces the karaoke line entirely
## ([method bind_captions] answers true, and main hides the overlay), because a subtitle
## floating over a page of the same words would be the same sentence printed twice.
##
## NO SCENES. The Director still keeps its schedule, but a cut hands over to one hidden
## placeholder this vehicle owns ([method take_over]), so nothing is built and nothing draws.
## A book that later wants scenes in its plates would cast them the way the comic does.
##
## REAL 3D. The stage SubViewport has a world of its own, so the book is geometry in it: a
## Camera3D, a lamp, a desk, a cloth cover, two page stacks and three leaves (the left page,
## the right page, and the one being turned), each leaf a subdivided plane bent by
## `LEAF_SHADER` into the gutter curve and, while turning, rolled over the spine with a curl.
## Each visible page is a SubViewport drawn by [BookVehicle.PageCanvas]; a page is redrawn
## only when what is printed on it or lit on it changes, and otherwise its render target is
## left alone. Four page targets cover the worst case, a turn, where the old left, the old
## right (the leaf's front), the new left (its back) and the new right are all in shot.
##
## A GENTLE CAMERA, deliberately unlike the comic's. It holds a wide view of the spread most
## of the time, drifts toward the page and then the lines being read, and changes framing on
## a slow schedule drawn from the session seed and the show clock, so an export frames the
## same shots as the live reading. Every channel is a critically damped spring: nothing it
## does can be a jump. `Director.camera` scales how far and how often it moves.

## World size of one page: 1 wide, 1.5 tall, spine along z at x = 0, the reader at +z.
const PAGE_H := BookLayout.PAGE.y / BookLayout.PAGE.x
## Seconds a leaf takes to turn, and how long before the new spread's first word it starts -
## the eye is on the new page as the voice reaches it, not a beat behind.
const TURN_TIME := 1.6
const TURN_LEAD := 1.1
## The camera's shot vocabulary: where it aims, how far back it stands, how steeply it
## looks down. Distances are world units, pitch is degrees above the page plane.
const SHOTS := {
	"wide": {"dist": 5.3, "pitch": 66.0, "weight": 0.55},
	"page": {"dist": 3.75, "pitch": 70.0, "weight": 0.32},
	"close": {"dist": 2.9, "pitch": 73.0, "weight": 0.13},
}
## How long one framing holds, in seconds of show time (sampled per shot in this range).
const SHOT_HOLD := Vector2(13.0, 24.0)
## The camera springs' time constant at Camera 1. Slow, on purpose.
const SPRING_TAU := 3.2
## A LONG LENS. A wide one stood close makes every downstroke on the page converge on one
## vanishing point, so a line of roman type leans like italics toward the edges of the frame -
## the whole page reads as emphasis. Standing back with a narrow field keeps print upright.
const VFOV := 20.0
const HIGHLIGHT := Color(1.0, 0.80, 0.30)

const LEAF_SHADER := """
shader_type spatial;
render_mode cull_disabled, diffuse_burley;
uniform sampler2D front : source_color, filter_linear, repeat_disable;
uniform sampler2D back : source_color, filter_linear, repeat_disable;
uniform float side = 1.0;
uniform float angle = 0.0;
uniform float curl = 0.0;
uniform float lift = 0.05;
uniform float base_y = 0.0;
uniform float page_h = 1.5;
uniform float has_back = 0.0;

// The leaf's centre line, arc length s from the spine, as (across, up): it leaves the spine
// at `angle` and bends by `curl` along its width, then the gutter swell is added along its
// own normal so a resting page still rises out of the spine.
vec2 profile(float s) {
	float th0 = angle;
	vec2 p;
	if (abs(curl) < 1e-4) {
		p = vec2(s * cos(th0), s * sin(th0));
	} else {
		float th = th0 - curl * s;
		p = vec2((sin(th0) - sin(th)) / curl, (cos(th) - cos(th0)) / curl);
	}
	float th_s = th0 - curl * s;
	vec2 n = vec2(-sin(th_s), cos(th_s));
	float h = lift * (1.0 - exp(-s * 7.0)) - lift * 0.3 * s;
	return p + n * h;
}

void vertex() {
	float s = side > 0.0 ? UV.x : 1.0 - UV.x;
	vec2 p = profile(s);
	vec2 q = profile(s + 0.01);
	vec2 t = normalize(q - p);
	float z = (UV.y - 0.5) * page_h;
	// Both leaves reach a hair past the spine, so no crack ever opens between them for the
	// cloth to show through.
	VERTEX = vec3((p.x - 0.006) * side, p.y + base_y, z);
	vec3 tan3 = vec3(t.x * side, t.y, 0.0);
	vec3 nrm = normalize(cross(vec3(0.0, 0.0, 1.0), tan3));
	if (side < 0.0) nrm = -nrm;
	NORMAL = nrm;
}

// A small box filter over the pixel's footprint. Page targets have no mipmaps, and printed
// text minified without them shimmers as the camera drifts; five taps spread over the
// footprint are what a mip level would have averaged.
vec3 tap(sampler2D tex, vec2 uv) {
	vec2 dx = dFdx(uv) * 0.38;
	vec2 dy = dFdy(uv) * 0.38;
	vec3 c = texture(tex, uv).rgb * 2.0;
	c += texture(tex, uv + dx + dy).rgb;
	c += texture(tex, uv + dx - dy).rgb;
	c += texture(tex, uv - dx + dy).rgb;
	c += texture(tex, uv - dx - dy).rgb;
	return c / 6.0;
}

void fragment() {
	bool up = FRONT_FACING;
	if (side < 0.0) up = !up;
	vec3 col;
	if (up || has_back < 0.5) {
		col = tap(front, UV);
	} else {
		col = tap(back, vec2(1.0 - UV.x, UV.y));
	}
	ALBEDO = col;
	ROUGHNESS = 0.92;
	SPECULAR = 0.12;
}
"""

var _subs = null                 # the Subtitles node: words, clock and the eased cursor
var _layout: BookLayout = null
var _source := ""                # what the layout was built from
var _rev := -1                   # Illustrations.revision it was built against
var _fallback_n := -1            # word count a caption-only layout was built from
var _textures := {}              # path -> Texture2D

var _vps: Array = []             # 4 page SubViewports
var _canvases: Array = []
var _slot_page := [-1, -1, -1, -1]
var _slot_state := ["", "", "", ""]

var _spread := 0                 # the spread lying open
var _turn_t := -1.0              # >= 0 while a leaf is turning
var _turn_to := 0

var _map: Array = []             # subtitle word index -> layout word index (or -1)
var _map_j := 0
var _map_rem := ""
var _map_src_n := 0
var _sent_lo := {}               # subtitle sentence id -> first layout word
var _sent_hi := {}

var _root3: Node3D
var _cam: Camera3D
var _env: Environment
var _lamp: SpotLight3D
var _leaf_l: MeshInstance3D
var _leaf_r: MeshInstance3D
var _leaf_t: MeshInstance3D
var _stack_l: MeshInstance3D
var _stack_r: MeshInstance3D
var _cover: MeshInstance3D
var _placeholder: GhostScene

var _paper := Color(0.95, 0.925, 0.87)
var _ink := Color(0.11, 0.095, 0.085)
var _seed := 0

# the camera's springs: value and velocity per channel
var _c_aim := Vector3.ZERO
var _v_aim := Vector3.ZERO
var _c_dist := 5.3
var _v_dist := 0.0
var _c_pitch := 60.0
var _v_pitch := 0.0
var _c_yaw := 0.0
var _v_yaw := 0.0
var _snap := true
var _reading_z := 0.0            # the line being read, as world z, heavily smoothed


# --- mount -------------------------------------------------------------------

func mount(st: SubViewport) -> void:
	super.mount(st)
	_placeholder = GhostScene.new()
	_placeholder.init_with_seed(1, "drift")
	_placeholder.visible = false
	add_child(_placeholder)
	for i in 4:
		var vp := SubViewport.new()
		vp.size = Vector2i(BookLayout.PAGE)
		vp.transparent_bg = false
		vp.disable_3d = true
		vp.render_target_update_mode = SubViewport.UPDATE_ONCE
		add_child(vp)
		var c := PageCanvas.new()
		c.book = self
		vp.add_child(c)
		_vps.append(vp)
		_canvases.append(c)
	_build_world()


func _build_world() -> void:
	_root3 = Node3D.new()
	add_child(_root3)
	_env = Environment.new()
	_env.background_mode = Environment.BG_COLOR
	_env.background_color = Color(0.02, 0.018, 0.016)
	_env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	_env.ambient_light_color = Color(0.55, 0.5, 0.45)
	_env.ambient_light_energy = 0.32
	_env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	_env.adjustment_enabled = true
	_cam = Camera3D.new()
	_cam.fov = VFOV
	_cam.near = 0.05
	_cam.far = 80.0
	_cam.environment = _env
	_root3.add_child(_cam)
	# THE LAMP. One warm pool of light on the book, falling off across the desk into dark, is
	# most of what makes this read as someone reading rather than as a document on a screen.
	_lamp = SpotLight3D.new()
	_lamp.light_color = Color(1.0, 0.86, 0.68)
	_lamp.light_energy = 5.2
	_lamp.spot_range = 12.0
	_lamp.spot_angle = 42.0
	_lamp.spot_angle_attenuation = 1.6
	_lamp.shadow_enabled = true
	_lamp.shadow_blur = 1.2
	_root3.add_child(_lamp)
	_lamp.position = Vector3(-1.4, 3.6, 1.2)
	_lamp.look_at_from_position(_lamp.position, Vector3(0.1, 0.0, 0.0), Vector3.UP)
	var fill := DirectionalLight3D.new()
	fill.light_color = Color(0.7, 0.75, 0.9)
	fill.light_energy = 0.18
	_root3.add_child(fill)
	fill.rotation_degrees = Vector3(-55.0, 30.0, 0.0)

	var desk := MeshInstance3D.new()
	var dm := PlaneMesh.new()
	dm.size = Vector2(24.0, 24.0)
	desk.mesh = dm
	var dmat := StandardMaterial3D.new()
	dmat.albedo_color = Color(0.20, 0.12, 0.075)
	dmat.roughness = 0.62
	desk.material_override = dmat
	desk.position = Vector3(0.0, -0.075, 0.0)
	_root3.add_child(desk)

	_cover = MeshInstance3D.new()
	var cm := BoxMesh.new()
	cm.size = Vector3(2.12, 0.03, PAGE_H + 0.1)
	_cover.mesh = cm
	var cmat := StandardMaterial3D.new()
	cmat.albedo_color = Color(0.32, 0.07, 0.06)
	cmat.roughness = 0.85
	_cover.material_override = cmat
	_cover.position = Vector3(0.0, -0.058, 0.0)
	_root3.add_child(_cover)

	_stack_l = _stack()
	_stack_r = _stack()
	_leaf_l = _leaf(-1.0)
	_leaf_r = _leaf(1.0)
	_leaf_t = _leaf(1.0)
	_leaf_t.visible = false


func _stack() -> MeshInstance3D:
	var m := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(0.985, 0.02, PAGE_H - 0.01)
	m.mesh = bm
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.86, 0.83, 0.76)
	mat.roughness = 0.95
	m.material_override = mat
	_root3.add_child(m)
	return m


func _leaf(side: float) -> MeshInstance3D:
	var m := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(1.0, PAGE_H)
	pm.subdivide_width = 48
	pm.subdivide_depth = 30
	m.mesh = pm
	var sh := Shader.new()
	sh.code = LEAF_SHADER
	var mat := ShaderMaterial.new()
	mat.shader = sh
	mat.set_shader_parameter("side", side)
	mat.set_shader_parameter("page_h", PAGE_H)
	m.material_override = mat
	# The shader moves every vertex; without a generous AABB the leaf is culled mid-turn.
	m.custom_aabb = AABB(Vector3(-1.2, -0.2, -PAGE_H), Vector3(2.4, 1.5, PAGE_H * 2.0))
	_root3.add_child(m)
	return m


func begin_session() -> void:
	_seed = Director.session_seed()
	var r := RandomNumberGenerator.new()
	r.seed = _seed ^ 0xB00C
	# Paper is cream, never white; ink is a warm near-black; the cloth is one of the dark
	# colours books are actually bound in.
	_paper = Color.from_hsv(r.randf_range(0.09, 0.13), r.randf_range(0.07, 0.14),
		r.randf_range(0.93, 0.97))
	_ink = Color.from_hsv(r.randf_range(0.04, 0.10), r.randf_range(0.15, 0.35),
		r.randf_range(0.08, 0.13))
	var cloth := [Color(0.34, 0.07, 0.06), Color(0.07, 0.19, 0.14), Color(0.08, 0.11, 0.24),
		Color(0.25, 0.15, 0.07), Color(0.16, 0.16, 0.17)]
	(_cover.material_override as StandardMaterial3D).albedo_color = cloth[r.randi() % cloth.size()]
	_reset_reading()
	_snap = true


func release() -> void:
	_subs = null
	_reset_reading()


func _reset_reading() -> void:
	_spread = 0
	_turn_t = -1.0
	_map = []
	_map_j = 0
	_map_rem = ""
	_sent_lo = {}
	_sent_hi = {}
	for i in 4:
		_slot_state[i] = ""


# --- the Vehicle contract ----------------------------------------------------

func owns_cast() -> bool:
	return true


## Every cut lands on the same hidden placeholder: the book has no scenes to change to.
func take_over(_outgoing: GhostScene) -> GhostScene:
	return _placeholder


func owns_bookend() -> bool:
	return true


## The page prints the words, so the karaoke overlay must not print them again.
func bind_captions(subs) -> bool:
	_subs = subs
	_source = ""
	_fallback_n = -1
	_reset_reading()
	return true


func advance(_features, delta: float, bookend: float) -> void:
	_env.adjustment_brightness = clampf(bookend, 0.0, 1.0)
	_ensure_layout()
	if _layout == null:
		return
	_extend_map()
	_tick_turn(delta)
	_tick_camera(delta)
	_place_leaves()
	_refresh_pages()


# --- the text ------------------------------------------------------------------

## (Re)typeset when the document, the pictures or (with no document) the words change.
func _ensure_layout() -> void:
	var src := ""
	var words: Array = _subs.words if _subs != null and is_instance_valid(_subs) else []
	if _subs != null and is_instance_valid(_subs):
		src = String((_subs.document as Dictionary).get("source", ""))
	if src.is_empty() and not words.is_empty():
		# NO MANUSCRIPT, ONLY WORDS: a take without a book block (an older sidecar, the
		# fishing voice). Print what is spoken, a paragraph every few sentences, and reset
		# as it grows - rarely, since a live take only ever appends.
		if words.size() == _fallback_n and _layout != null:
			src = _source
		elif _layout == null or words.size() - _fallback_n >= 40 or _layout.words.size() < 40:
			src = _words_to_source(words)
			_fallback_n = words.size()
		else:
			src = _source
	if _layout != null and src == _source and _rev == Illustrations.revision:
		return
	var reflow := _layout != null and src == _source
	if _rev != Illustrations.revision:
		_textures = {}              # a reroll is a new file; let the old pictures go
	_source = src
	_rev = Illustrations.revision
	var lay := BookLayout.new()
	lay.build(src, _image_size)
	_layout = lay
	_map = []
	_map_j = 0
	_map_rem = ""
	_sent_lo = {}
	_sent_hi = {}
	for i in 4:
		_slot_page[i] = -1
		_slot_state[i] = ""
	_spread = clampi(_spread if reflow else 0, 0, maxi(0, lay.spreads() - 1))
	print("ghost: book typeset - %d pages, %d words" % [lay.pages.size(), lay.words.size()])


static func _words_to_source(words: Array) -> String:
	var out := ""
	var last := -1
	var sentences := 0
	for w in words:
		var si := int((w as Dictionary).get("sentence", 0))
		if si != last and last >= 0:
			sentences += 1
			if sentences % 4 == 0:
				out += "\n\n"
		last = si
		out += String((w as Dictionary).get("text", "")) + " "
	return out


func _image_size(key: String) -> Vector2:
	var t := _texture_for(key)
	return Vector2(t.get_size()) if t != null else Vector2.ZERO


func _texture_for(key: String) -> Texture2D:
	if key.is_empty():
		return null
	var path := Illustrations.path_for(key)
	if path.is_empty():
		return null
	if _textures.has(path):
		return _textures[path]
	var img := Image.load_from_file(path)
	if img == null or img.is_empty():
		_textures[path] = null
		return null
	img.generate_mipmaps()
	var tex := ImageTexture.create_from_image(img)
	_textures[path] = tex
	return tex


## Map the spoken words onto the printed ones, as far as they have arrived.
##
## SEQUENTIAL WITH RESYNC. Both lists are the same text in the same order, so a match is
## normally the next printed word; the window absorbs what the voice and the page spell
## differently (a numeral read as three words shows ONE display word, a dash-joined word may
## arrive in pieces). A spoken word matching nothing nearby is left unmapped rather than
## dragging the pointer somewhere wrong.
func _extend_map() -> void:
	if _subs == null or not is_instance_valid(_subs):
		return
	var words: Array = _subs.words
	if words.size() < _map.size():
		_map = []                    # the take restarted in place
		_map_j = 0
		_map_rem = ""
		_sent_lo = {}
		_sent_hi = {}
	var lw := _layout.words
	for i in range(_map.size(), words.size()):
		var n := _layout.norm(String((words[i] as Dictionary).get("text", "")))
		var got := -1
		if n.is_empty():
			got = maxi(_map_j - 1, -1)
		elif not _map_rem.is_empty() and _map_rem.begins_with(n):
			got = _map_j - 1
			_map_rem = _map_rem.substr(n.length())
		else:
			for k in range(_map_j, mini(_map_j + 12, lw.size())):
				var ln := String((lw[k] as Dictionary)["norm"])
				if ln == n:
					got = k
					_map_rem = ""
					break
				if ln.begins_with(n) and n.length() >= 2:
					got = k
					_map_rem = ln.substr(n.length())
					break
				if n.begins_with(ln) and ln.length() >= 2 and k == _map_j:
					got = k           # the voice's word spans two printed ones
					_map_rem = ""
					break
			if got >= 0:
				_map_j = got + 1
		_map.append(got)
		if got >= 0:
			var si := int((words[i] as Dictionary).get("sentence", 0))
			_sent_lo[si] = mini(int(_sent_lo.get(si, got)), got)
			_sent_hi[si] = maxi(int(_sent_hi.get(si, got)), got)


## Where the reading is: `{sub, layout, frac, next, sentence, alpha}`, or empty before any.
func _reading() -> Dictionary:
	if _subs == null or not is_instance_valid(_subs) or _map.is_empty():
		return {}
	var c: float = _subs.cursor()
	var i := clampi(int(floor(c)), 0, _map.size() - 1)
	var li := int(_map[i])
	var k := i
	while li < 0 and k > 0:
		k -= 1
		li = int(_map[k])
	if li < 0:
		return {}
	var nxt := int(_map[i + 1]) if i + 1 < _map.size() else -1
	return {"sub": i, "layout": li, "frac": clampf(c - float(i), 0.0, 1.0), "next": nxt,
		"sentence": int((_subs.words[i] as Dictionary).get("sentence", 0)),
		"alpha": clampf(float(_subs.presence), 0.0, 1.0)}


# --- the leaf ------------------------------------------------------------------

func _spread_of_word(li: int) -> int:
	if li < 0 or li >= _layout.words.size():
		return _spread
	return int((_layout.words[li] as Dictionary)["page"]) / 2


## TURN WHEN THE VOICE IS ABOUT TO ARRIVE. The spread to show is decided from the NEXT word
## and its start time, so the leaf is already over by the time it is spoken. A jump of more
## than one spread (a restart, a reflow) is not a turn and simply lands.
func _tick_turn(delta: float) -> void:
	if _turn_t >= 0.0:
		_turn_t += delta
		if _turn_t >= TURN_TIME:
			_turn_t = -1.0
			_spread = _turn_to
		return
	var want := _spread
	var r := _reading()
	if not r.is_empty():
		want = _spread_of_word(int(r["layout"]))
		var ni := int(r["sub"]) + 1
		if ni < _map.size() and int(_map[ni]) >= 0:
			var ns := _spread_of_word(int(_map[ni]))
			var t0 := float((_subs.words[ni] as Dictionary).get("t0", 0.0))
			if ns > want and _subs.now() >= t0 - TURN_LEAD:
				want = ns
	if want == _spread:
		return
	if want == _spread + 1:
		_turn_to = want
		_turn_t = 0.0
	else:
		_spread = clampi(want, 0, maxi(0, _layout.spreads() - 1))


func _turn_k() -> float:
	if _turn_t < 0.0:
		return 0.0
	return smoothstep(0.0, 1.0, _turn_t / TURN_TIME)


func _place_leaves() -> void:
	var n := maxi(1, _layout.spreads() - 1)
	var prog := clampf(float(_spread) / float(n), 0.0, 1.0)
	# The stacks thicken on the read side as the chapter goes by.
	var tl := 0.012 + 0.035 * prog
	var tr := 0.012 + 0.035 * (1.0 - prog)
	_stack_l.scale = Vector3(1.0, tl / 0.02, 1.0)
	_stack_r.scale = Vector3(1.0, tr / 0.02, 1.0)
	var floor_y := -0.043
	_stack_l.position = Vector3(-0.5, floor_y + tl * 0.5, 0.0)
	_stack_r.position = Vector3(0.5, floor_y + tr * 0.5, 0.0)
	var yl := floor_y + tl
	var yr := floor_y + tr
	var turning := _turn_t >= 0.0
	var left_page := _spread * 2
	var right_page := (_turn_to if turning else _spread) * 2 + 1
	_bind(_leaf_l, left_page, -1, yl)
	_bind(_leaf_r, right_page, -1, yr)
	_leaf_t.visible = turning
	if turning:
		var k := _turn_k()
		var a := PI * k
		var mat := _leaf_t.material_override as ShaderMaterial
		mat.set_shader_parameter("angle", a)
		mat.set_shader_parameter("curl", 1.1 * sin(a))
		mat.set_shader_parameter("lift", 0.05 * (1.0 - sin(a)))
		_bind(_leaf_t, _spread * 2 + 1, _turn_to * 2, lerpf(yr, yl, k) + 0.003)


func _bind(leaf: MeshInstance3D, page: int, back_page: int, y: float) -> void:
	var mat := leaf.material_override as ShaderMaterial
	mat.set_shader_parameter("base_y", y)
	mat.set_shader_parameter("front", _page_tex(page))
	mat.set_shader_parameter("has_back", 1.0 if back_page >= 0 else 0.0)
	if back_page >= 0:
		mat.set_shader_parameter("back", _page_tex(back_page))


## The texture showing [param page], giving it a page target if it has none. The four slots
## are enough by construction (see the class note); a slot is reused from a page no longer
## asked for, oldest first.
func _page_tex(page: int) -> Texture2D:
	var s := _slot_page.find(page)
	if s < 0:
		var need := _needed_pages()
		for i in 4:
			if not need.has(int(_slot_page[i])):
				s = i
				break
		if s < 0:
			s = 0
		_slot_page[s] = page
		_slot_state[s] = ""
	return (_vps[s] as SubViewport).get_texture()


func _needed_pages() -> Array:
	var out := [_spread * 2, _spread * 2 + 1]
	if _turn_t >= 0.0:
		out.append_array([_turn_to * 2, _turn_to * 2 + 1])
	return out


## Redraw a page target only when what it shows has changed: its page, or the highlight
## lying on it. Every other page keeps the picture it already has.
func _refresh_pages() -> void:
	var r := _reading()
	for i in 4:
		var page := int(_slot_page[i])
		var c: PageCanvas = _canvases[i]
		var hl := _highlight_for(page, r)
		var state := "%d|%s" % [page, str(hl)]
		if state == _slot_state[i]:
			continue
		_slot_state[i] = state
		c.page = page
		c.hl = hl
		c.queue_redraw()
		(_vps[i] as SubViewport).render_target_update_mode = SubViewport.UPDATE_ONCE


func _highlight_for(page: int, r: Dictionary) -> Dictionary:
	if r.is_empty() or page < 0 or page >= _layout.pages.size():
		return {}
	var pg: Dictionary = _layout.pages[page]
	var ws: Array = pg["words"]
	if ws.is_empty():
		return {}
	var lo := int(ws[0])
	var hi := int(ws[ws.size() - 1])
	var li := int(r["layout"])
	var si := int(r["sentence"])
	var s_lo := int(_sent_lo.get(si, li))
	var s_hi := int(_sent_hi.get(si, li))
	if s_hi < lo or s_lo > hi:
		# Not being read here - but already-read pages keep their faded ink.
		return {"read": li} if li > hi else {}
	# Quantised, so a page redraws at most a few dozen times a second of sweep rather than
	# on every float wobble of the eased cursor.
	return {"word": li, "next": int(r["next"]), "frac": snappedf(float(r["frac"]), 0.02),
		"alpha": snappedf(float(r["alpha"]), 0.02), "s_lo": s_lo, "s_hi": s_hi, "read": li}


## Draw page [param page] of the layout onto [param ci].
func draw_page(ci: CanvasItem, page: int, hl: Dictionary) -> void:
	var size := BookLayout.PAGE
	ci.draw_rect(Rect2(Vector2.ZERO, size), _paper)
	if _layout == null or page < 0 or page >= _layout.pages.size():
		return
	var pg: Dictionary = _layout.pages[page]
	# A breath of shading toward the spine: the paper turns away from the light as it goes
	# into the gutter, and a flat page reads as a printout.
	var spine_x := size.x if int(pg["side"]) == 0 else 0.0
	for k in 12:
		var w := 90.0 * (1.0 - float(k) / 12.0)
		var x := spine_x - w if spine_x > 0.0 else 0.0
		ci.draw_rect(Rect2(x, 0.0, w, size.y), Color(0.25, 0.18, 0.1, 0.02))
	var alpha := float(hl.get("alpha", 0.0))
	var li := int(hl.get("word", -1))
	var read := int(hl.get("read", -1))
	# The sentence being read, faintly tinted line by line.
	if li >= 0 and alpha > 0.0:
		var runs := _line_runs(int(hl["s_lo"]), int(hl["s_hi"]), page)
		for rr in runs:
			var box: StyleBoxFlat = _box(Color(HIGHLIGHT, 0.10 * alpha), 10.0)
			ci.draw_style_box(box, (rr as Rect2).grow_individual(8.0, 0.0, 8.0, 0.0))
		# THE FINGER: a swash under the word being said, sliding toward the next word as the
		# eased cursor moves through this one, so it travels along the line rather than
		# stepping from box to box.
		var a: Dictionary = _layout.words[li]
		var rect: Rect2 = a["rect"]
		var nx := int(hl.get("next", -1))
		if nx >= 0 and nx < _layout.words.size():
			var b: Dictionary = _layout.words[nx]
			var rb: Rect2 = b["rect"]
			if int(b["page"]) == page and absf(rb.position.y - rect.position.y) < 1.0:
				var f := smoothstep(0.55, 1.0, float(hl["frac"]))
				rect = Rect2(rect.position.lerp(rb.position, f), rect.size.lerp(rb.size, f))
		if int(a["page"]) == page:
			ci.draw_style_box(_box(Color(HIGHLIGHT, 0.42 * alpha), 9.0),
				rect.grow_individual(7.0, -2.0, 7.0, -4.0))
	for wi in pg["words"]:
		var w: Dictionary = _layout.words[int(wi)]
		var ink := _ink
		if read >= 0 and int(wi) < read:
			ink = _ink.lerp(_paper, 0.12)
		ci.draw_string(_layout.face(int(w["emph"])), w["base"], String(w["text"]),
			HORIZONTAL_ALIGNMENT_LEFT, -1, int(w["fs"]), ink)
	for lb in pg["labels"]:
		var l: Dictionary = lb
		ci.draw_string(_layout.face(int(l["emph"])), l["pos"], String(l["text"]),
			HORIZONTAL_ALIGNMENT_CENTER, float(l["align_w"]), int(l["fs"]),
			Color(_ink, float(l.get("tone", 1.0))))
	for im in pg["images"]:
		_draw_image(ci, im)
	var folio := int(pg["folio"])
	if folio > 0:
		var c := _layout.column(page)
		ci.draw_string(_layout.face(0), Vector2(c.x, size.y - BookLayout.MARGIN_BOTTOM * 0.45),
			str(folio), HORIZONTAL_ALIGNMENT_CENTER, c.y - c.x, BookLayout.FOLIO_FS,
			Color(_ink, 0.7))


var _boxes := {}


func _box(col: Color, radius: float) -> StyleBoxFlat:
	var key := "%s|%s" % [col.to_html(), radius]
	if _boxes.has(key):
		return _boxes[key]
	var b := StyleBoxFlat.new()
	b.bg_color = col
	b.set_corner_radius_all(int(radius))
	b.anti_aliasing = true
	_boxes[key] = b
	return b


## One rectangle per LINE covering layout words [lo, hi] that sit on [param page].
func _line_runs(lo: int, hi: int, page: int) -> Array:
	var out: Array = []
	var cur := Rect2()
	var have := false
	for i in range(maxi(lo, 0), mini(hi, _layout.words.size() - 1) + 1):
		var w: Dictionary = _layout.words[i]
		if int(w["page"]) != page:
			continue
		var r: Rect2 = w["rect"]
		if have and absf(r.position.y - cur.position.y) < 1.0:
			cur = cur.merge(r)
		else:
			if have:
				out.append(cur)
			cur = r
			have = true
	if have:
		out.append(cur)
	return out


func _draw_image(ci: CanvasItem, im: Dictionary) -> void:
	var rect: Rect2 = im["rect"]
	var tex := _texture_for(String(im["key"]))
	if tex == null:
		# NOT YET PAINTED: a quiet frame holding the description, so the page shows where the
		# picture goes and what it will be, and nothing reads as broken.
		ci.draw_rect(rect, Color(_ink, 0.035))
		ci.draw_rect(rect.grow(-10.0), Color(_ink, 0.22), false, 1.5)
		var f := _layout.face(1)
		var pad := 34.0
		var lines := maxi(1, int((rect.size.y - pad * 2.0 - 26.0) / 28.0))
		var est := mini(lines, int(ceil(f.get_string_size(String(im["prompt"]),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 22).x / maxf(1.0, rect.size.x - pad * 2.0))) + 1)
		ci.draw_multiline_string(f, rect.position + Vector2(pad,
			maxf(pad + 26.0, rect.size.y * 0.5 - float(est) * 14.0 + 20.0)),
			String(im["prompt"]), HORIZONTAL_ALIGNMENT_CENTER, rect.size.x - pad * 2.0, 22,
			lines, Color(_ink, 0.45))
		return
	# COVER, not contain: the picture fills its plate and the spare axis is cropped evenly.
	var ts := Vector2(tex.get_size())
	var want := rect.size.x / rect.size.y
	var src := Rect2(Vector2.ZERO, ts)
	if ts.x / ts.y > want:
		var w := ts.y * want
		src = Rect2((ts.x - w) * 0.5, 0.0, w, ts.y)
	else:
		var h := ts.x / want
		src = Rect2(0.0, (ts.y - h) * 0.5, ts.x, h)
	ci.draw_texture_rect_region(tex, rect, src)
	ci.draw_rect(rect, Color(_ink, 0.25), false, 1.0)


# --- the camera ------------------------------------------------------------------

func _sev() -> float:
	return clampf(Director.camera, Director.CAMERA_MIN, Director.CAMERA_MAX)


## The framing in force at show time [param t]: a pure function of the seed and the clock,
## so a render frames exactly what the live reading framed.
func _shot_at(t: float) -> String:
	var sev := _sev()
	# Walk the schedule from zero. Holds are sampled per shot, so the index at t is found by
	# summing them; a chapter is a few hundred shots at most.
	var acc := 0.0
	var idx := 0
	while true:
		var h := _hash01(idx, 11)
		var hold := lerpf(SHOT_HOLD.x, SHOT_HOLD.y, h) / lerpf(0.75, 1.35, sev * 0.5)
		if acc + hold > t or idx > 4000:
			break
		acc += hold
		idx += 1
	if idx == 0:
		return "wide"               # every reading opens on the whole spread
	var wide_w := float(SHOTS["wide"]["weight"]) * lerpf(1.8, 0.8, sev * 0.5)
	var page_w := float(SHOTS["page"]["weight"])
	var close_w := float(SHOTS["close"]["weight"]) * sev
	var u := _hash01(idx, 29) * (wide_w + page_w + close_w)
	if u < wide_w:
		return "wide"
	if u < wide_w + page_w:
		return "page"
	return "close"


func _hash01(i: int, salt: int) -> float:
	return float(hash([_seed, i, salt]) & 0xFFFFFF) / float(0xFFFFFF)


func _tick_camera(delta: float) -> void:
	var t := maxf(Spectrum.current.time, 0.0)
	var sev := _sev()
	var shot := _shot_at(t)
	var sh: Dictionary = SHOTS[shot]
	# Where the reading is on the spread: its page and its line, as world coordinates.
	var aim := Vector3(0.0, 0.0, 0.06)
	var r := _reading()
	var page_x := 0.0
	var line_z := 0.0
	if not r.is_empty():
		var w: Dictionary = _layout.words[int(r["layout"])]
		var rect: Rect2 = w["rect"]
		var side := int(w["page"]) % 2
		page_x = -0.5 if side == 0 else 0.5
		line_z = (rect.get_center().y / BookLayout.PAGE.y - 0.5) * PAGE_H
	var zk := 1.0 - exp(-maxf(delta, 0.0) / 4.0)
	_reading_z = lerpf(_reading_z, line_z, zk)
	match shot:
		"page":
			aim = Vector3(page_x * 0.85, 0.0, _reading_z * 0.35 + 0.05)
		"close":
			aim = Vector3(page_x, 0.0, clampf(_reading_z, -0.45, 0.5) + 0.05)
	# A slow wander of the viewpoint, so a held shot is never a still.
	var yaw := sin(t * 0.041 + float(_seed % 97)) * 5.0 * sev
	var pitch := float(sh["pitch"]) + sin(t * 0.029 + 1.3) * 2.0 * sev
	var dist := float(sh["dist"]) * (1.0 + 0.03 * sin(t * 0.033 + 2.1) * sev)
	if _snap:
		_snap = false
		_c_aim = aim
		_c_dist = dist
		_c_pitch = pitch
		_c_yaw = yaw
	var tau := SPRING_TAU / lerpf(0.6, 1.4, sev * 0.5)
	var steps := maxi(1, int(ceil(delta / (1.0 / 60.0))))
	var h := delta / float(steps)
	for _i in steps:
		var res := _spring3(_c_aim, _v_aim, aim, tau, h)
		_c_aim = res[0]
		_v_aim = res[1]
		var rd := _spring(_c_dist, _v_dist, dist, tau, h)
		_c_dist = rd.x
		_v_dist = rd.y
		var rp := _spring(_c_pitch, _v_pitch, pitch, tau, h)
		_c_pitch = rp.x
		_v_pitch = rp.y
		var ry := _spring(_c_yaw, _v_yaw, yaw, tau, h)
		_c_yaw = ry.x
		_v_yaw = ry.y
	var p := deg_to_rad(_c_pitch)
	var y := deg_to_rad(_c_yaw)
	var dir := Vector3(sin(y) * cos(p), sin(p), cos(y) * cos(p))
	_cam.position = _c_aim + dir * _c_dist
	_cam.look_at(_c_aim, Vector3.UP)
	if not _cam.current:
		_cam.make_current()


## A critically damped spring, one step: no overshoot, no jump, eases in and out.
static func _spring(x: float, v: float, target: float, tau: float, h: float) -> Vector2:
	var w := 2.0 / maxf(tau, 0.05)
	var a := w * w * (target - x) - 2.0 * w * v
	v += a * h
	x += v * h
	return Vector2(x, v)


static func _spring3(x: Vector3, v: Vector3, target: Vector3, tau: float, h: float) -> Array:
	var w := 2.0 / maxf(tau, 0.05)
	var a := (target - x) * (w * w) - v * (2.0 * w)
	v += a * h
	x += v * h
	return [x, v]


## Debug line for probes.
func debug_line() -> String:
	if _layout == null:
		return "no layout"
	var r := _reading()
	return "spread %d/%d turn %.2f shot %s reading %s" % [_spread, _layout.spreads(),
		_turn_k(), _shot_at(maxf(Spectrum.current.time, 0.0)), str(r)]


## One page target's drawing surface. It holds no state of its own beyond which page and
## which highlight it shows; the vehicle does the drawing.
class PageCanvas:
	extends Node2D
	var book = null
	var page := -1
	var hl := {}

	func _draw() -> void:
		if book != null:
			book.draw_page(self, page, hl)
