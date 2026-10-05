extends Medium
class_name TarotMedium

## TarotMedium - a tarot reading at a table, seen from the reader's chair.
##
## The tarot mode's own medium (see [constant Medium.OWNED]): the episode's cloth on a table,
## its room out of focus beyond the far edge, its candles and its painted objects standing
## about, and the deck. The reader is a voice, never a pair of hands - the cards move on their own:
##
##   the intro      the deck shuffles in the middle of the table under the voice, in RUNS -
##                  riffles, overhand passes, strings of cuts, now and then a wash across the
##                  cloth - with long stretches of nothing between them ([constant RUNS])
##   the push       before the first card, the deck is squared and pushed to its side
##   a draw         the deck squares, the top card slides off, turns over, and comes up to the
##                  camera on the LEFT; its booklet page opens on the RIGHT - shown, never read
##   the reading    the card and the page float there while the voice talks about it, turning
##                  a little on their axes; now and then the card is turned to look at its back
##   the lay        the page closes and the card goes down into its place in the spread
##   a jumper       the first card flies out of the shuffle on its own, lands face up, and is
##                  picked up and shown like a drawn one
##   the close      the whole spread lies on the table
##
## EVERYTHING IS A FUNCTION OF SHOW TIME, as the tablet's screen is: the actions are the
## reading's own marks ([TarotScript]), placed in the rests the voice left for them by a
## [ReadingFollower], and every card, page and packet of the deck is posed from that schedule
## each frame - so a render draws exactly what the live reading drew, and a scrub lands the
## table where the reading is.
##
## THE LOOK IS THE EPISODE'S. The cloth, the room, the card back and every face are its
## pictures; the frame, the type, the props and the light come from its look through
## [TarotTable]'s registries, and the rest - where the deck sits, how the spread is laid, which
## shuffles, the camera's height - is sampled from the episode's seed, so no two episodes share a
## table. A picture still being painted is a placeholder until it lands, live.
##
## FOIL. The brightest, most colorful parts of each painting - found per picture from its own
## luminance, never a fixed threshold - are printed as foil: they catch the light as a card
## tilts, breathe slowly, and now and then a glint sweeps across them, and the scene's bloom
## lets them bleed. The deck's look says how much foil it was printed with.

## A card is 70 x 120 mm. World units are meters: the table is a table.
const CARD := Vector2(0.07, 0.12)
const CARD_R := 0.0045
const CARD_T := 0.0007
## The deck: this many meshes, each standing for three cards - 78 cards stand about 3 cm.
const DECK_N := 26
## How much table a laid card leaves between itself and the deck.
const DECK_CLEAR := 0.012
const DECK_T := 0.0012
## The table (width, thickness, depth), centered at TABLE_Z, its top at y = 0; the cloth on it.
const TABLE := Vector3(1.9, 0.05, 0.85)
const TABLE_Z := 0.02
const CLOTH := Vector2(1.2, 0.72)
## Where a shown card and its booklet page float, in the camera's frame (meters right, up, and
## away). The page sits a hair farther back so the card always reads as in front.
const PRESENT_DIST := 0.23
const PRESENT := Vector2(-0.087, 0.012)
const PAGE := Vector2(0.088, 0.012)
const PAGE_H := 0.126
const VFOV := 42.0

## THE PHASES of each action, in its own seconds. They sum to [TarotScript]'s rests, which is
## what the voice waits: DRAW = 3.4, LAY = 1.7, JUMP = 4.2.
const SQUARE := 0.45          # the deck squares before a card leaves it
const SLIDE_END := 1.0        # the top card slides off toward the reader
const FLIP_END := 1.9         # ...lifts and turns over
const RISE_END := 2.8         # ...and comes up to be shown
const PAGE_IN := Vector2(2.6, 3.4)
const LAY_PAGE_OUT := 0.5
const LAY_MOVE := Vector2(0.15, 1.45)
const LAY_END := 1.7
const JUMP_FLY := Vector2(0.1, 1.2)
const JUMP_REST := 2.5
const JUMP_RISE := 3.5
const JUMP_PAGE := Vector2(3.4, 4.2)
## Lead and tail around each action group, as the tablet's: a beat after the last word before
## the cards move, and they are still a beat before the next word.
## HOW A READER SHUFFLES: in RUNS - several riffles, a string of cuts, a few overhand passes,
## one at a tempo - and then nothing for a while, the deck squared under their hands while they
## talk. A run per kind: how likely it is chosen, how many moves, each move's length and the gap
## inside the run. The wash (the deck spread and swirled across the cloth) is its own event: rare,
## long, and once at most.
const RUNS := {
	"riffle": {"weight": 3.0, "n": [2, 4], "dur": [2.2, 2.7], "gap": [0.2, 0.55]},
	"overhand": {"weight": 2.0, "n": [2, 5], "dur": [2.1, 2.6], "gap": [0.05, 0.3]},
	"cut": {"weight": 2.0, "n": [2, 5], "dur": [1.15, 1.5], "gap": [0.05, 0.25]},
	"wash": {"weight": 0.9, "n": [1, 1], "dur": [9.5, 12.5], "gap": [0.0, 0.0]},
}
## The pause after a run: mostly a few seconds, now and then a long linger - median ~3 s.
const IDLE_LOG := Vector2(1.1, 0.55)
const IDLE_RANGE := Vector2(1.6, 18.0)
## The chance a run is followed by a long linger instead.
const LINGER_CHANCE := 0.15
## A wash is sampled at this rate once, when it is planned, and posed by lookup.
const WASH_HZ := 20.0
## The first card's PUSH (TarotScript.PUSH): the deck squares, then slides to its side this long.
const PUSH_SLIDE := 0.85
## A held card is turned over now and then, to look at its back: the chance a card is, how long
## a turn takes each way, and how long its back is looked at.
const TURN_CHANCE := 0.6
const TURN := 0.75
const TURN_HOLD := Vector2(1.1, 1.8)
## The render layer candle bodies are on as well, so the flames can leave them out of their shadows.
const CANDLE_LAYER := 8
const LEAD := 0.25
const TAIL := 0.2

const FOIL_SHADER := """
shader_type spatial;
render_mode blend_mix, cull_back, diffuse_burley, specular_schlick_ggx;
uniform sampler2D tex : source_color, filter_linear_mipmap, repeat_disable;
uniform vec4 window = vec4(0.0, 0.0, 1.0, 1.0);   // the picture's part of the face, in UV
uniform float lo = 0.8;                            // the picture's own foil key (luminance)
uniform float hi = 0.95;
uniform vec3 accent = vec3(0.79, 0.64, 0.15);      // the frame's foil color
uniform float foil = 0.6;                          // how much foil the deck was printed with
uniform float pulse = 0.5;                         // the slow breath, 0..1
uniform float glint = -1.0;                        // the sweep's place along the diagonal
uniform float lift = 0.14;                         // a little self-light, so the art reads
uniform float dim = 1.0;                           // the bookend
void fragment() {
	vec4 c = texture(tex, UV);
	float lum = dot(c.rgb, vec3(0.299, 0.587, 0.114));
	float mx = max(c.r, max(c.g, c.b));
	float mn = min(c.r, min(c.g, c.b));
	float sat = (mx - mn) / max(mx, 0.0001);
	float inside = step(window.x, UV.x) * step(UV.x, window.z) * step(window.y, UV.y) * step(UV.y, window.w);
	float key = inside * smoothstep(lo, hi, lum) * mix(0.45, 1.0, sat);
	// the frame's own accent prints as foil too - on the FRAME only: inside the painting the same
	// color may be the whole ground of the picture
	key = max(key, (1.0 - inside) * (1.0 - smoothstep(0.06, 0.16, distance(c.rgb, accent))));
	key *= foil;
	float band = 0.0;
	if (glint > -0.5) {
		float d = (UV.x + UV.y) * 0.5 - glint;
		band = exp(-d * d / 0.004);
	}
	ALBEDO = c.rgb * dim;
	ROUGHNESS = mix(0.66, 0.2, key);
	METALLIC = key * 0.6;
	SPECULAR = mix(0.4, 0.8, key);
	// bright enough to cross the bloom's threshold, breathing between a glow and a blaze; a
	// foil that only ever reached the threshold was measured to change the picture by a level or two
	EMISSION = c.rgb * dim * (lift + key * (1.3 + 2.2 * pulse) + key * band * 5.0);
}
"""

const FLAME_SHADER := """
shader_type spatial;
render_mode unshaded, cull_disabled, blend_add;
uniform vec3 color = vec3(1.0, 0.65, 0.3);
uniform float energy = 4.0;
void fragment() {
	vec2 p = UV - vec2(0.5, 0.62);
	p.x *= 2.2;
	float body = 1.0 - smoothstep(0.0, 0.42, length(vec2(p.x, p.y * (p.y < 0.0 ? 0.55 : 1.0))));
	float core = 1.0 - smoothstep(0.0, 0.16, length(vec2(p.x * 1.4, p.y + 0.12)));
	vec3 col = mix(color, vec3(1.0, 0.97, 0.88), core);
	ALBEDO = col * energy * body;
	ALPHA = body;
}
"""

var _subs = null
var _pay: Dictionary = {}           # the episode, as the panel handed it over (TarotEpisode.document)
var _source := ""                   # the reading's script
var _parse: Dictionary = {}
var _follow := ReadingFollower.new()
var _sched: Array = []
var _built_n := -1
var _key := ""
var _now := 0.0
var _seed := 0
var _look: Dictionary = {}

var _placeholder: GhostScene
var _root3: Node3D
var _cam: Camera3D
var _cam_base := Transform3D.IDENTITY
var _env: Environment
var _attrs: CameraAttributesPractical
var _lamp: SpotLight3D
var _fill: DirectionalLight3D
var _table: MeshInstance3D
var _cloth: MeshInstance3D
var _cloth_mat: StandardMaterial3D
var _backdrop: MeshInstance3D
var _backdrop_mat: StandardMaterial3D
var _props: Node3D
var _flames: Array = []             # [{mesh, light, seed, base_energy}]
var _objects: Array = []            # [{path, at, height, node}] - the painted objects on the table
var _pool_tex: Texture2D = null
var _deck: Array = []               # DECK_N MeshInstance3D, bottom first
var _cards: Array = []              # one MeshInstance3D per drawn card
var _faces: Array = []              # one SubViewport per drawn card's face
var _face_canvas: Array = []
var _face_mats: Array = []
var _back_vp: SubViewport
var _back_canvas: TarotCards.Face
var _back_mat: ShaderMaterial
var _page: MeshInstance3D
var _page_vp: SubViewport
var _page_canvas: TarotCards.Page
var _page_mat: StandardMaterial3D
var _page_for := -1
var _title: TitleCard
var _card_mesh: ArrayMesh
var _edge_mat: StandardMaterial3D

# the episode's table, sampled from its seed
var _deck_base := Vector3.ZERO      # where the deck is drawn from, to one side
var _mid := Vector3.ZERO            # where it is shuffled, in the middle of the table
var _cur_base := Vector3.ZERO       # where it is at the moment being posed (see _deck_at)
var _slots: Array = []              # per card: Transform3D where it lies in the spread
var _jump_land := Vector3.ZERO
var _slot_jit: Array = []           # per deck slot: Vector3(x, z, yaw)
var _moves: Array = []              # the shuffle: [{kind, t0, dur, seed}]
var _move_rng := RandomNumberGenerator.new()
var _pitch := 45.0
var _foil := 0.6
var _noise := FastNoiseLite.new()
var _mtimes := {}
var _poll_t := 0.0
var _textures := {}


# --- mount -----------------------------------------------------------------------------------

func mount(st: SubViewport) -> void:
	super.mount(st)
	_placeholder = GhostScene.new()
	_placeholder.init_with_seed(1, "drift")
	_placeholder.visible = false
	add_child(_placeholder)
	_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	_noise.frequency = 1.0
	_build_world()
	_title = TitleCard.new()
	add_child(_title)


func _build_world() -> void:
	_root3 = Node3D.new()
	add_child(_root3)
	_env = Environment.new()
	_env.background_mode = Environment.BG_COLOR
	_env.background_color = Color(0.02, 0.018, 0.02)
	_env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	_env.ambient_light_color = Color(0.5, 0.46, 0.44)
	_env.ambient_light_energy = 0.18
	_env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	_env.tonemap_exposure = 0.82
	_env.adjustment_enabled = true
	# THE BLOOM the foil and the flames bleed through: only what is brighter than white glows
	_env.glow_enabled = true
	_env.glow_normalized = false
	_env.glow_intensity = 0.75
	_env.glow_strength = 1.0
	_env.glow_bloom = 0.0
	_env.glow_hdr_threshold = 1.25
	_env.glow_hdr_scale = 2.0
	_env.glow_blend_mode = Environment.GLOW_BLEND_MODE_SCREEN
	_env.set_glow_level(0, 0.0)
	_env.set_glow_level(1, 0.6)
	_env.set_glow_level(2, 1.0)
	_env.set_glow_level(3, 0.8)
	_env.set_glow_level(4, 0.4)
	_cam = Camera3D.new()
	_cam.fov = VFOV
	_cam.near = 0.02
	_cam.far = 40.0
	_cam.environment = _env
	# THE ROOM IS OUT OF FOCUS: the table is where the eye is, and a painted room far behind it
	# reads as a place rather than as a picture of one
	_attrs = CameraAttributesPractical.new()
	_attrs.dof_blur_far_enabled = true
	_attrs.dof_blur_far_distance = 1.4
	_attrs.dof_blur_far_transition = 1.2
	_attrs.dof_blur_amount = 0.09
	_cam.attributes = _attrs
	_root3.add_child(_cam)

	_lamp = SpotLight3D.new()
	_lamp.light_energy = 1.6
	_lamp.spot_range = 4.0
	_lamp.spot_angle = 38.0
	_lamp.spot_angle_attenuation = 2.2
	_lamp.shadow_enabled = true
	_lamp.shadow_blur = 1.6
	_root3.add_child(_lamp)
	_fill = DirectionalLight3D.new()
	_fill.light_energy = 0.12
	_fill.rotation_degrees = Vector3(-60.0, 25.0, 0.0)
	_root3.add_child(_fill)

	_table = MeshInstance3D.new()
	var tm := BoxMesh.new()
	tm.size = TABLE
	_table.mesh = tm
	_table.position = Vector3(0.0, -TABLE.y * 0.5, TABLE_Z)
	var wood := StandardMaterial3D.new()
	wood.albedo_color = Color(0.36, 0.22, 0.13)
	wood.albedo_texture = BookMedium._grime(0x51A7, 0.018, 5, 0.22)
	wood.uv1_scale = Vector3(3.0, 3.0, 1.0)
	wood.roughness = 0.55
	_table.material_override = wood
	_root3.add_child(_table)

	_cloth = MeshInstance3D.new()
	var cm := PlaneMesh.new()
	cm.size = CLOTH
	_cloth.mesh = cm
	_cloth.position = Vector3(0.0, 0.0006, -0.02)
	_cloth_mat = StandardMaterial3D.new()
	_cloth_mat.roughness = 0.92
	_cloth.material_override = _cloth_mat
	_root3.add_child(_cloth)

	_backdrop = MeshInstance3D.new()
	var bq := QuadMesh.new()
	bq.size = Vector2(1.0, 1.0)
	_backdrop.mesh = bq
	_backdrop_mat = StandardMaterial3D.new()
	_backdrop_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	_backdrop.material_override = _backdrop_mat
	_root3.add_child(_backdrop)

	_props = Node3D.new()
	_root3.add_child(_props)

	_card_mesh = _make_card_mesh(CARD, CARD_T, CARD_R)
	_edge_mat = StandardMaterial3D.new()
	_edge_mat.albedo_color = Color(0.9, 0.88, 0.82)
	_edge_mat.roughness = 0.8
	var deck_mesh := _make_card_mesh(CARD, DECK_T, CARD_R)
	_back_canvas = TarotCards.Face.new()
	_back_canvas.back = true
	_back_vp = TarotCards.viewport(self, _back_canvas, TarotCards.FACE_PX)
	_back_mat = _foil_material(_back_vp.get_texture())
	_back_mat.set_shader_parameter("lift", 0.06)
	var unseen := StandardMaterial3D.new()
	unseen.albedo_color = Color(0.92, 0.9, 0.85)
	for i in DECK_N:
		var m := MeshInstance3D.new()
		m.mesh = deck_mesh
		m.set_surface_override_material(0, unseen)
		m.set_surface_override_material(1, _back_mat)
		m.set_surface_override_material(2, _edge_mat)
		_root3.add_child(m)
		_deck.append(m)

	_page = MeshInstance3D.new()
	var pq := QuadMesh.new()
	pq.size = Vector2(PAGE_H * TarotCards.PAGE_ASPECT, PAGE_H)
	_page.mesh = pq
	_page_canvas = TarotCards.Page.new()
	_page_vp = TarotCards.viewport(self, _page_canvas, TarotCards.PAGE_PX)
	_page_mat = StandardMaterial3D.new()
	_page_mat.albedo_texture = _page_vp.get_texture()
	_page_mat.emission_enabled = true
	_page_mat.emission_texture = _page_vp.get_texture()
	_page_mat.emission_energy_multiplier = 0.32
	_page_mat.roughness = 0.85
	_page_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	_page.material_override = _page_mat
	_page.visible = false
	_root3.add_child(_page)


## A card: a thin slab with rounded corners - surface 0 the face (on -Y), 1 the back (on +Y), 2
## the edge. Width along X, height along Z with the card's TOP at -Z, so a card lying face down
## with its top away from the reader turns face up the right way round when flipped about its
## long side (see [method _face_up]).
static func _make_card_mesh(size: Vector2, t: float, r: float) -> ArrayMesh:
	var outline := PackedVector2Array()
	var seg := 6
	var hw := size.x * 0.5 - r
	var hd := size.y * 0.5 - r
	var centers := [Vector2(hw, hd), Vector2(-hw, hd), Vector2(-hw, -hd), Vector2(hw, -hd)]
	for c in 4:
		for i in seg + 1:
			var a := (float(c) + float(i) / float(seg)) * PI * 0.5
			outline.append(centers[c] + Vector2(cos(a), sin(a)) * r)
	var mesh := ArrayMesh.new()
	var n := outline.size()
	for side in [-1.0, 1.0]:
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		var y: float = side * t * 0.5
		for i in n:
			var a := outline[i]
			var b := outline[(i + 1) % n]
			var tri := [Vector2.ZERO, a, b] if side > 0.0 else [Vector2.ZERO, b, a]
			for p in tri:
				var v := p as Vector2
				st.set_normal(Vector3(0.0, side, 0.0))
				# the face is seen from below until the card is turned: its U runs the other way
				var u := (0.5 + v.x / size.x) if side > 0.0 else (0.5 - v.x / size.x)
				st.set_uv(Vector2(u, 0.5 + v.y / size.y))
				st.add_vertex(Vector3(v.x, y, v.y))
		st.commit(mesh)
	var rim := SurfaceTool.new()
	rim.begin(Mesh.PRIMITIVE_TRIANGLES)
	for i in n:
		var a := outline[i]
		var b := outline[(i + 1) % n]
		var na := Vector3(a.x, 0.0, a.y).normalized()
		var nb := Vector3(b.x, 0.0, b.y).normalized()
		for v in [[a, t * 0.5, na], [b, -t * 0.5, nb], [b, t * 0.5, nb],
				[a, t * 0.5, na], [a, -t * 0.5, na], [b, -t * 0.5, nb]]:
			rim.set_normal(v[2])
			rim.add_vertex(Vector3((v[0] as Vector2).x, v[1], (v[0] as Vector2).y))
	rim.commit(mesh)
	return mesh


func _foil_material(tex: Texture2D) -> ShaderMaterial:
	var sh := Shader.new()
	sh.code = FOIL_SHADER
	var m := ShaderMaterial.new()
	m.shader = sh
	m.set_shader_parameter("tex", tex)
	return m


# --- the Medium contract -----------------------------------------------------------------------

func owns_cast() -> bool:
	return true


func take_over(_outgoing: GhostScene) -> GhostScene:
	return _placeholder


func owns_bookend() -> bool:
	return true


## The words are spoken, not printed: the karaoke line stays (return false). Its clock is the
## table's.
func bind_captions(subs) -> bool:
	_subs = subs
	_key = ""
	return false


func begin_session() -> void:
	_key = ""


func release() -> void:
	_subs = null
	_key = ""


func on_stage_resized(_size: Vector2) -> void:
	if _title != null:
		_title.queue_redraw()


func advance(_features, delta: float, bookend: float) -> void:
	var t := maxf(Spectrum.current.time, 0.0)
	var slen := Spectrum.song_length()
	var b := bookend if slen > 0.0 and t > slen * 0.5 else clampf(t / 1.2, 0.0, 1.0)
	var dim := clampf(b * Director.live_fade, 0.0, 1.0)
	_env.adjustment_brightness = dim
	_now = _subs.now() if _subs != null and is_instance_valid(_subs) else t
	_ensure_doc()
	_poll_t -= delta
	if _poll_t <= 0.0:
		_poll_t = 1.0
		_poll_pictures()
	if not _parse.is_empty() and _subs != null and is_instance_valid(_subs):
		_follow.extend(_subs.words)
		var n := _follow.map.size()
		if n != _built_n:
			_built_n = n
			_sched = _follow.place(_parse["actions"], maxf(Director.intro_hold, 0.6), LEAD, TAIL)
	_pose(_now)
	_tick_camera(_now)
	_tick_props(_now)
	_title.alpha = _title_alpha(_now)
	_title.queue_redraw()


# --- the episode ----------------------------------------------------------------------------------

func _ensure_doc() -> void:
	if _subs == null or not is_instance_valid(_subs):
		return
	var d: Dictionary = _subs.document
	var pay: Dictionary = d.get("tarot", {}) if d.get("tarot") is Dictionary else {}
	var src := String(d.get("source", ""))
	var key := "%s|%d|%d" % [String(pay.get("dir", "")), int(pay.get("seed", 0)), hash(src)]
	if key == _key:
		return
	_key = key
	_pay = pay
	_source = src
	_parse = TarotScript.parse(src) if not src.is_empty() else {}
	var plan: Dictionary = pay.get("plan", {}) if pay.get("plan") is Dictionary else {}
	_look = TarotTable.sanitize_look(plan.get("look", {}) if plan.get("look") is Dictionary else {})
	_seed = int(pay.get("seed", 0))
	_foil = float(_look.get("foil", 0.6))
	# A READING STARTED MID-WAY (a scrub) begins where its first words are: everything before them
	# happened long ago, spaced as a voice would have said it, so the cards drawn by then are
	# already down (or up) on the first frame
	var spoken: PackedStringArray = _parse.get("spoken", PackedStringArray()) if not _parse.is_empty() \
		else PackedStringArray()
	var start_si := -1
	var sw: Variant = d.get("start_words", PackedStringArray())
	if sw is PackedStringArray and not (sw as PackedStringArray).is_empty() and not spoken.is_empty():
		start_si = TabletScript.find_run(spoken, sw, int(d.get("start_index", -1)))
	_follow.reset(spoken, start_si, _estimated_times(spoken.size()))
	_built_n = -1
	_sched = []
	_build_episode()
	print("ghost: tarot table - %s #%d, %d cards, %d actions" % [String(pay.get("show", "?")), _seed,
		(pay.get("cards", []) as Array).size(), (_parse.get("actions", []) as Array).size()])


## Each spoken word's time along the reading as a voice would say it - the past a mid-way start
## is replayed in: the intro, then a word every 0.4 s, and each action's own rest.
func _estimated_times(n: int) -> PackedFloat32Array:
	var rests := {}
	for a in _parse.get("actions", []):
		var k := int((a as Dictionary)["after"])
		rests[k] = float(rests.get(k, 0.0)) + float((a as Dictionary)["dur"]) + LEAD + TAIL
	var out := PackedFloat32Array()
	out.resize(n)
	var t := maxf(Director.intro_hold, 0.0)
	for j in n:
		t += float(rests.get(j, 0.0))
		out[j] = t
		t += 0.4
	return out


## Everything sampled from the episode's seed, and the episode's own cards.
func _build_episode() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([_seed, "tarot-table"])
	_title.channel = String(_subs.document.get("title", "")) if _subs != null else ""
	_title.episode = String((_pay.get("plan", {}) as Dictionary).get("episode_title", "")) \
		if _pay.get("plan") is Dictionary else ""
	# THE CHANNEL'S NAME IS THE CHANNEL'S, set the same way every episode - a brand, not a
	# deck's lettering (an uncial deck turned "Truthful" into "Truchful")
	_title.face = TarotTable.font("roman")
	_title.italic = TarotTable.font(TarotTable.BOOK_ITALIC)
	# THE CAMERA: the reader's eye, a little different every episode
	# shallow enough that the room always shows past the table's far edge (at 48 it was a sliver)
	_pitch = rng.randf_range(34.0, 42.0)
	var dist := rng.randf_range(0.5, 0.56)
	var yaw := deg_to_rad(rng.randf_range(-3.5, 3.5))
	var target := Vector3(rng.randf_range(-0.01, 0.01), 0.0, -0.05)
	var p := deg_to_rad(_pitch)
	var eye := target + Vector3(sin(yaw) * cos(p), sin(p), cos(yaw) * cos(p)) * dist
	_cam_base = Transform3D(Basis.looking_at(target - eye, Vector3.UP), eye)
	_cam.transform = _cam_base
	_cam.fov = VFOV + rng.randf_range(-1.5, 1.5)
	_attrs.dof_blur_far_distance = dist + rng.randf_range(0.45, 0.75)
	# THE LIGHT: the look's own, from above and to one side
	var lc := TarotTable.color(String((_look.get("light", {}) as Dictionary).get("color", "#ffb36b")))
	_lamp.light_color = Color(1, 1, 1).lerp(lc, 0.55)
	var side := -1.0 if rng.randf() < 0.6 else 1.0
	# low enough, and far enough to one side, that what stands on the table throws a shadow you see
	_lamp.position = Vector3(side * rng.randf_range(0.4, 0.6), rng.randf_range(0.75, 0.95), rng.randf_range(-0.35, -0.1))
	_lamp.look_at(Vector3(0.0, 0.0, -0.1), Vector3.UP)
	_lamp.light_energy = rng.randf_range(1.3, 1.9)
	var pal: Array = _look.get("palette", TarotTable.FALLBACK_PALETTE)
	var dark := TarotTable.color(String(pal[0]))
	_env.background_color = dark.darkened(0.6)
	_env.ambient_light_color = Color(0.5, 0.5, 0.5).lerp(dark.lightened(0.4), 0.35)
	_fill.light_color = Color(0.75, 0.8, 1.0) if String((_look.get("light", {}) as Dictionary).get("warmth", "warm")) == "warm" else lc
	# THE DECK, squared where the reader keeps it
	_deck_base = Vector3(rng.randf_range(0.17, 0.23) * (1.0 if rng.randf() < 0.7 else -1.0), 0.0,
		rng.randf_range(0.04, 0.07))
	# SHUFFLED IN THE MIDDLE, in front of the reader, and pushed to its side before the first card
	_mid = Vector3(rng.randf_range(-0.015, 0.015), 0.0, rng.randf_range(-0.05, -0.02))
	_cur_base = _mid
	_slot_jit = []
	for i in DECK_N + 1:
		_slot_jit.append(Vector3(rng.randf_range(-0.0007, 0.0007), rng.randf_range(-0.0007, 0.0007),
			deg_to_rad(rng.randf_range(-1.3, 1.3))))
	_move_rng.seed = hash([_seed, "tarot-moves"])
	_moves = []
	# the chain made now, wash plans and all, rather than a frame at a time while it plays
	_move_at(120.0)
	# THE CARDS
	for c in _cards:
		(c as Node).queue_free()
	for v in _faces:
		(v as Node).queue_free()
	_cards = []
	_faces = []
	_face_canvas = []
	_face_mats = []
	# THIS EPISODE'S PICTURES ARE KEPT across its sessions (each Speak, each scrub): decoding a
	# dozen full-size paintings on the main thread is a visible hitch. Another episode's go.
	var keep_dir := String(_pay.get("dir", "")) + "/"
	for k in _textures.keys():
		if not String(k).begins_with(keep_dir):
			_textures.erase(k)
			_mtimes.erase(String(k))
	var cards: Array = _pay.get("cards", [])
	for i in cards.size():
		var canvas := TarotCards.Face.new()
		canvas.look = _look
		canvas.card = cards[i]
		canvas.seed = _seed
		var vp := TarotCards.viewport(self, canvas, TarotCards.FACE_PX)
		var mat := _foil_material(vp.get_texture())
		mat.set_shader_parameter("window", _face_window())
		var m := MeshInstance3D.new()
		m.mesh = _card_mesh
		m.set_surface_override_material(0, mat)
		m.set_surface_override_material(1, _back_mat)
		m.set_surface_override_material(2, _edge_mat)
		m.visible = false
		_root3.add_child(m)
		_cards.append(m)
		_faces.append(vp)
		_face_canvas.append(canvas)
		_face_mats.append(mat)
	_back_canvas.look = _look
	_back_canvas.seed = _seed
	_back_mat.set_shader_parameter("window", Vector4(0.0, 0.0, 1.0, 1.0))
	_edge_mat.albedo_color = TarotTable.color(String((_look.get("frame", {}) as Dictionary).get("stock", "#efe6d2"))).darkened(0.08)
	_page_canvas.look = _look
	_page_canvas.seed = _seed
	_page_for = -1
	_slots = _spread_slots(cards.size(), rng)
	# a jumper flies out of the deck in the middle and lands on the far side from where the deck
	# is about to go
	_jump_land = _mid + Vector3(-signf(_deck_base.x) * 0.16, 0.0, 0.035)
	_build_props(rng)
	_place_backdrop()
	_poll_pictures()
	for vp in _faces:
		TarotCards.redraw(vp)
	TarotCards.redraw(_back_vp)


## The part of a face that is the painting, in UV - foil is keyed inside it (the frame's foil is
## keyed by its color instead).
func _face_window() -> Vector4:
	var style := String((_look.get("frame", {}) as Dictionary).get("style", "line"))
	var sz := Vector2(TarotCards.FACE_PX)
	var m := sz.x * (0.018 if style == "bleed" else 0.06)
	var top := 0.0 if style == "bleed" else sz.y * 0.075
	var bottom := 0.0 if style == "bleed" else sz.y * 0.105
	return Vector4(m / sz.x, (m + top) / sz.y, 1.0 - m / sz.x, 1.0 - (m + bottom) / sz.y)


## Pictures that have landed since last looked - live, a reading can start while the deck is
## still being painted.
func _poll_pictures(force := false) -> void:
	var dir := String(_pay.get("dir", ""))
	if dir.is_empty():
		return
	var accent := TarotTable.color(String((_look.get("frame", {}) as Dictionary).get("accent", "#c9a227")))
	for i in _cards.size():
		var tex := _picture(dir.path_join("card_%d.png" % (i + 1)), force)
		if tex != null and (_face_canvas[i] as TarotCards.Face).art != tex:
			(_face_canvas[i] as TarotCards.Face).art = tex
			var key: Vector2 = _textures.get(dir.path_join("card_%d.png" % (i + 1)) + "|key", Vector2(0.8, 0.95))
			var mat: ShaderMaterial = _face_mats[i]
			mat.set_shader_parameter("lo", key.x)
			mat.set_shader_parameter("hi", key.y)
			TarotCards.redraw(_faces[i])
		(_face_mats[i] as ShaderMaterial).set_shader_parameter("accent", Vector3(accent.r, accent.g, accent.b))
		(_face_mats[i] as ShaderMaterial).set_shader_parameter("foil", _foil)
	var back := _picture(dir.path_join("back.png"), force)
	if back != null and _back_canvas.art != back:
		_back_canvas.art = back
		var key: Vector2 = _textures.get(dir.path_join("back.png") + "|key", Vector2(0.8, 0.95))
		_back_mat.set_shader_parameter("lo", key.x)
		_back_mat.set_shader_parameter("hi", key.y)
		TarotCards.redraw(_back_vp)
	_back_mat.set_shader_parameter("accent", Vector3(accent.r, accent.g, accent.b))
	_back_mat.set_shader_parameter("foil", _foil * 0.8)
	var cloth := _picture(dir.path_join("surface.png"), force)
	if cloth != null:
		_cloth_mat.albedo_texture = cloth
		_cloth_mat.albedo_color = Color(1, 1, 1)
	elif _cloth_mat.albedo_texture == null:
		var pal: Array = _look.get("palette", TarotTable.FALLBACK_PALETTE)
		_cloth_mat.albedo_color = TarotTable.color(String(pal[min(4, pal.size() - 1)])).darkened(0.35)
		_cloth_mat.albedo_texture = BookMedium._grime(hash([_seed, "cloth"]) & 0xFFFF, 0.05, 4, 0.12)
	var room := _picture(dir.path_join("backdrop.png"), force)
	if room != null:
		_backdrop_mat.albedo_texture = room
		_backdrop_mat.albedo_color = Color(1, 1, 1)
	elif _backdrop_mat.albedo_texture == null:
		var pal2: Array = _look.get("palette", TarotTable.FALLBACK_PALETTE)
		_backdrop_mat.albedo_color = TarotTable.color(String(pal2[0])).darkened(0.25)
	_poll_objects()


## The picture at [param path] as a texture, reloaded when the file changes; null when absent.
## Each picture's foil key is measured as it loads: the 90th and 98th percentile of its own
## luminance, so a dark painting's stars glow and a pale one's highlights do, never a fixed line.
func _picture(path: String, force := false) -> Texture2D:
	if not FileAccess.file_exists(path):
		return null
	var mt := FileAccess.get_modified_time(path)
	if not force and int(_mtimes.get(path, -1)) == mt and _textures.has(path):
		return _textures[path]
	_mtimes[path] = mt
	var img := Image.load_from_file(path)
	if img == null or img.is_empty():
		return null
	var small := img.duplicate() as Image
	small.resize(48, 72, Image.INTERPOLATE_BILINEAR)
	var lums := PackedFloat32Array()
	for y in small.get_height():
		for x in small.get_width():
			lums.append(small.get_pixel(x, y).get_luminance())
	lums.sort()
	# IN LINEAR LIGHT: the shader samples the face as `source_color`, so it compares LINEAR values -
	# thresholds measured on these sRGB pixels sat above the very regions they were taken from, and
	# the foil keyed nothing at all
	var lo := Color(lums[int(lums.size() * 0.9)], 0, 0).srgb_to_linear().r
	var hi := Color(lums[int(lums.size() * 0.985)], 0, 0).srgb_to_linear().r
	_textures[path + "|key"] = Vector2(lo, maxf(hi, lo + 0.03))
	img.generate_mipmaps()
	var tex := ImageTexture.create_from_image(img)
	_textures[path] = tex
	return tex


## THE ROOM, square to the camera far behind the table, placed so its horizon (the lower third
## of the picture, as it was asked for) sits just above the table's far edge.
func _place_backdrop() -> void:
	var dist := 3.2
	var fwd := -_cam_base.basis.z
	var center := _cam_base.origin + fwd * dist
	var h := 2.0 * dist * tan(deg_to_rad(_cam.fov * 0.5))
	var w := h * 16.0 / 9.0
	var far_edge := Vector3(0.0, 0.0, TABLE_Z - TABLE.z * 0.5)
	var up := _cam_base.basis.y
	# where the far edge's line of sight crosses the backdrop's plane, along the camera's up axis
	var to_edge := (far_edge - _cam_base.origin).normalized()
	var along := to_edge.dot(fwd)
	var hit := _cam_base.origin + to_edge * (dist / maxf(along, 0.05))
	var edge_up := (hit - center).dot(up)
	var size := Vector2(w * 1.45, w * 1.45 / 1.5)
	# the picture's horizon (a third up from its bottom) on that line
	var offset := edge_up - (-size.y * 0.5 + size.y * (1.0 / 3.0))
	_backdrop.transform = Transform3D(_cam_base.basis, center + up * offset)
	(_backdrop.mesh as QuadMesh).size = size


# --- the spread ------------------------------------------------------------------------------------

## WHERE EACH CARD GOES DOWN: a layout the readers' videos actually use - rows, never a cross
## (measured: rows of three to ten, often two) - picked per episode, laid within reach of the
## camera and clear of the deck. Each card a hair off true, as a hand lays them.
func _spread_slots(n: int, rng: RandomNumberGenerator) -> Array:
	var out: Array = []
	if n <= 0:
		return out
	var gap := CARD.x * rng.randf_range(1.12, 1.3)
	var kinds := ["row", "arc"]
	if n >= 5:
		kinds.append("rows")
	if n == 3 or n == 6:
		kinds.append("pyramid")
	var kind := String(kinds[rng.randi_range(0, kinds.size() - 1)])
	if n * gap > 0.66:
		kind = "rows"
	var pos: Array = []
	match kind:
		"rows":
			var top := int(ceil(n / 2.0))
			for i in n:
				var row := 0 if i < top else 1
				var k := i if row == 0 else i - top
				var cnt := top if row == 0 else n - top
				pos.append(Vector3((float(k) - float(cnt - 1) * 0.5) * gap, 0.0, -0.165 + float(row) * CARD.y * 1.12))
		"pyramid":
			var rows := [[0], [1, 2]] if n == 3 else [[0], [1, 2], [3, 4, 5]]
			var z0 := -0.21 if n == 6 else -0.15
			for r in rows.size():
				var row: Array = rows[r]
				for k in row.size():
					pos.append(Vector3((float(k) - float(row.size() - 1) * 0.5) * gap, 0.0,
						z0 + float(r) * CARD.y * 0.92))
		"arc":
			var bend := rng.randf_range(0.18, 0.32)
			for i in n:
				var x := (float(i) - float(n - 1) * 0.5) * gap
				pos.append(Vector3(x, 0.0, -0.08 + bend * x * x * 3.0 - 0.02))
		_:
			for i in n:
				pos.append(Vector3((float(i) - float(n - 1) * 0.5) * gap, 0.0, -0.085))
	for i in n:
		var p: Vector3 = pos[i]
		p += Vector3(rng.randf_range(-0.003, 0.003), (float(i) + 1.0) * 0.0002, rng.randf_range(-0.003, 0.003))
		var yaw := deg_to_rad(rng.randf_range(-2.5, 2.5))
		if kind == "arc":
			yaw += -p.x * 0.6
		out.append({"pos": p, "yaw": yaw})
	# CLEAR OF THE DECK: two rows of eight or a tight arc reach the deck's corner of the table, and
	# were laid through it - the whole spread steps back, or aside, by the least that clears it
	var keep := Rect2(_deck_base.x - CARD.x * 0.5 - DECK_CLEAR, _deck_base.z - CARD.y * 0.5 - DECK_CLEAR,
		CARD.x + DECK_CLEAR * 2.0, CARD.y + DECK_CLEAR * 2.0)
	var off := TarotTable.clear_of(out, CARD, keep)
	for sl in out:
		(sl as Dictionary)["pos"] = ((sl as Dictionary)["pos"] as Vector3) + Vector3(off.x, 0.0, off.y)
	return out


## Card [param i]'s pose lying in the spread, face up, the way it came out of the deck.
func _slot_xf(i: int) -> Transform3D:
	var s: Dictionary = _slots[i] if i < _slots.size() else {"pos": Vector3.ZERO, "yaw": 0.0}
	var yaw := float(s["yaw"]) + (PI if _reversed(i) else 0.0)
	return Transform3D(Basis(Vector3.UP, yaw) * _face_up(), s["pos"] as Vector3 + Vector3(0, CARD_T * 0.5, 0))


func _reversed(i: int) -> bool:
	var cards: Array = _pay.get("cards", [])
	return i < cards.size() and bool((cards[i] as Dictionary).get("reversed", false))


## A card turned face up about its long side.
static func _face_up() -> Basis:
	return Basis(Vector3(0, 0, 1), PI)


# --- the props ---------------------------------------------------------------------------------------

## THINGS ON THE TABLE: the look's candles, and its OBJECTS - each painted alone and cut out
## ([TarotCutout]), stood SQUARE TO THE CAMERA, which never moves, so a flat picture painted from
## this angle reads as the thing itself. Where each stands is found, not listed (see
## [method _find_spot]): the tallest first, toward the back.
func _build_props(rng: RandomNumberGenerator) -> void:
	for c in _props.get_children():
		c.queue_free()
	_flames = []
	_objects = []
	var taken: Array = []                 # screen rects already stood in
	var objects: Array = (_pay.get("objects", []) as Array).duplicate()
	objects.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		return float(TarotTable.OBJECT_SIZES.get(a.get("size"), 0.1)) > float(TarotTable.OBJECT_SIZES.get(b.get("size"), 0.1)))
	for o in objects:
		var path := String((o as Dictionary).get("path", ""))
		var h := float(TarotTable.OBJECT_SIZES.get((o as Dictionary).get("size"), 0.12)) * rng.randf_range(0.9, 1.1)
		# its width from its picture when it has been painted; until then, a guess
		var tex := _picture(path)
		var aspect := float(tex.get_width()) / maxf(1.0, float(tex.get_height())) if tex != null else 0.85
		var at := Vector3(INF, 0, 0)
		for shrink in 3:
			at = _find_spot(h * aspect, h, taken, rng, 1.0)
			if at.x != INF:
				break
			h *= 0.82
		if at.x == INF:
			continue
		_objects.append({"path": path, "at": at, "node": null, "height": h})
	var pal: Array = _look.get("palette", TarotTable.FALLBACK_PALETTE)
	for c in int(_look.get("candles", 1)):
		var ch := rng.randf_range(0.07, 0.13)
		var cr := rng.randf_range(0.017, 0.026)
		var at := _find_spot(cr * 2.0, ch + 0.035, taken, rng, 0.3)
		if at.x == INF:
			continue
		_candle(at, ch, cr, rng, pal)
	_poll_objects()


## WHERE A THING CAN STAND: on the table, IN THE SHOT with a margin, clear of everywhere the cards
## go (the spread, the deck, the middle where it is shuffled and washed, where a jumper lands),
## and not standing in front of what already stands. A thing [param w] wide and [param h] tall;
## [param back] is how much it prefers the far side (the tallest want to be at the back, where they
## hide nothing). Its screen rectangle is added to [param taken]. INF when nowhere is free.
func _find_spot(w: float, h: float, taken: Array, rng: RandomNumberGenerator, back: float) -> Vector3:
	var keep_out: Array = []
	for sl in _slots:
		keep_out.append(TarotTable.footprint((sl as Dictionary)["pos"], float((sl as Dictionary)["yaw"]), CARD).grow(0.03))
	keep_out.append(Rect2(_deck_base.x - 0.08, _deck_base.z - 0.1, 0.16, 0.2))
	keep_out.append(Rect2(_mid.x - 0.22, _mid.z - 0.14, 0.44, 0.28))
	keep_out.append(Rect2(_jump_land.x - 0.07, _jump_land.z - 0.09, 0.14, 0.18))
	var best := Vector3(INF, 0, 0)
	var best_score := -INF
	var best_rect := Rect2()
	var z := -0.36
	while z <= 0.14:
		var x := -0.5
		while x <= 0.5:
			var at := Vector3(x + rng.randf_range(-0.01, 0.01), 0.0, z + rng.randf_range(-0.01, 0.01))
			x += 0.025
			var foot := Rect2(at.x - w * 0.5, at.z - w * 0.4, w, w * 0.8)
			var clear := true
			for r in keep_out:
				if (r as Rect2).intersects(foot):
					clear = false
					break
			if not clear:
				continue
			var rect := _screen_rect(at, w, h)
			# wholly in the shot side to side and at its foot; a tall thing at the back may run off
			# the top, as things at the far edge of a table do, so long as most of it shows
			if rect.position.x < 0.02 or rect.end.x > 0.98 or rect.end.y > 0.97 \
					or rect.position.y < -0.45 * rect.size.y:
				continue
			for r in taken:
				if (r as Rect2).grow(0.015).intersects(rect):
					clear = false
					break
			if not clear:
				continue
			var score := -at.z * back * 3.0 + absf(at.x) * 0.8 + rng.randf() * 0.35
			if score > best_score:
				best_score = score
				best = at
				best_rect = rect
		z += 0.025
	if best.x != INF:
		taken.append(best_rect)
	return best


## The rectangle a thing [param w] wide and [param h] tall standing at [param at] covers on the
## screen, 0-1 each way, as the episode's camera sees it.
func _screen_rect(at: Vector3, w: float, h: float) -> Rect2:
	var c := _cam_base.basis
	var pts := [at - c.x * w * 0.5, at + c.x * w * 0.5, at + c.y * h - c.x * w * 0.5, at + c.y * h + c.x * w * 0.5]
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	var k := tan(deg_to_rad(_cam.fov * 0.5))
	for p in pts:
		var l: Vector3 = _cam_base.affine_inverse() * (p as Vector3)
		var d := maxf(-l.z, 0.001)
		var s := Vector2(0.5 + l.x / (d * k * (16.0 / 9.0)) * 0.5, 0.5 - l.y / (d * k) * 0.5)
		lo = lo.min(s)
		hi = hi.max(s)
	return Rect2(lo, hi - lo)


## Stand up each object whose picture has landed (live, objects are painted while a reading
## can already be playing).
func _poll_objects() -> void:
	for o in _objects:
		var d: Dictionary = o
		if d["node"] != null:
			continue
		var tex := _picture(String(d["path"]))
		if tex == null:
			continue
		d["node"] = _object(tex, d["at"] as Vector3, float(d["height"]))


## ONE OBJECT: a quad square to the camera, its bottom edge where it stands, its picture's own
## light kept (it was painted lit as the table is) - and its SHADOW laid on the cloth as decals:
## a soft dark pool under its base, and its own silhouette stretched away from the lamp.
func _object(tex: Texture2D, at: Vector3, h: float) -> Node3D:
	var c := _cam_base.basis
	var w := h * float(tex.get_width()) / maxf(1.0, float(tex.get_height()))
	var holder := Node3D.new()
	_props.add_child(holder)
	var quad := MeshInstance3D.new()
	var qm := QuadMesh.new()
	qm.size = Vector2(w, h)
	quad.mesh = qm
	var mat := StandardMaterial3D.new()
	mat.albedo_texture = tex
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_DEPTH_PRE_PASS
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = Color(1, 1, 1).lerp(_lamp.light_color, 0.3) * 0.86
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
	quad.material_override = mat
	quad.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	quad.layers = 2                  # the decals below do not land on it
	quad.transform = Transform3D(c, at + c.y * (h * 0.5) + Vector3(0, 0.001, 0))
	holder.add_child(quad)
	# the pool of shadow under its base
	var pool := Decal.new()
	pool.texture_albedo = _pool_texture()
	pool.modulate = Color(0, 0, 0, 1)
	pool.albedo_mix = 0.7
	pool.size = Vector3(w * 1.1, 0.04, w * 0.5)
	pool.position = at + Vector3(0, 0, -w * 0.12)
	pool.cull_mask = 1
	holder.add_child(pool)
	# its silhouette, cast away from the lamp along the cloth
	var away := at - _lamp.global_position
	away.y = 0.0
	away = away.normalized() if away.length() > 0.001 else Vector3(0, 0, 1)
	var cast := Decal.new()
	cast.texture_albedo = _silhouette(tex)
	cast.modulate = Color(0, 0, 0, 1)
	cast.albedo_mix = 0.45
	var length := h * 0.55
	cast.size = Vector3(w, 0.04, length)
	cast.transform = Transform3D(Basis.looking_at(-away, Vector3.UP), at + away * (length * 0.5))
	cast.cull_mask = 1
	holder.add_child(cast)
	return holder


## A soft round pool of shade, made once.
func _pool_texture() -> Texture2D:
	if _pool_tex == null:
		var g := Gradient.new()
		g.set_color(0, Color(0, 0, 0, 0.75))
		g.set_color(1, Color(0, 0, 0, 0.0))
		var gt := GradientTexture2D.new()
		gt.gradient = g
		gt.fill = GradientTexture2D.FILL_RADIAL
		gt.fill_from = Vector2(0.5, 0.5)
		gt.fill_to = Vector2(1.0, 0.5)
		gt.width = 64
		gt.height = 64
		_pool_tex = gt
	return _pool_tex


## An object's shadow shape: its own alpha, blurred, upside down (its top falls farthest).
func _silhouette(tex: Texture2D) -> Texture2D:
	var img := tex.get_image()
	if img == null:
		return _pool_texture()
	img = img.duplicate() as Image
	if img.is_compressed():
		img.decompress()
	img.convert(Image.FORMAT_RGBA8)
	var w := 48
	var h := maxi(8, roundi(48.0 * float(img.get_height()) / maxf(1.0, float(img.get_width()))))
	img.resize(w / 2, maxi(4, h / 2), Image.INTERPOLATE_BILINEAR)
	img.resize(w, h, Image.INTERPOLATE_BILINEAR)
	for y in h:
		for x in w:
			var a := img.get_pixel(x, y).a
			img.set_pixel(x, y, Color(0, 0, 0, a * 0.8))
	img.flip_y()
	return ImageTexture.create_from_image(img)


func _candle(at: Vector3, h: float, r: float, rng: RandomNumberGenerator, pal: Array) -> void:
	var body := MeshInstance3D.new()
	var cm := CylinderMesh.new()
	cm.top_radius = r
	cm.bottom_radius = r * 1.02
	cm.height = h
	cm.radial_segments = 24
	body.mesh = cm
	var wax := StandardMaterial3D.new()
	var tint := TarotTable.color(String(pal[rng.randi_range(0, pal.size() - 1)]))
	wax.albedo_color = Color(0.8, 0.76, 0.68).lerp(tint, rng.randf_range(0.1, 0.5))
	wax.roughness = 0.75
	wax.rim_enabled = true
	wax.rim = 0.3
	wax.emission_enabled = true
	wax.emission = Color(1.0, 0.6, 0.3)
	wax.emission_energy_multiplier = 0.03
	body.material_override = wax
	body.position = at + Vector3(0, h * 0.5, 0)
	# a candle throws a shadow from the lamp, never from a flame: its own flame, just above its top,
	# printed a hard dark disc round its base
	body.layers = CANDLE_LAYER
	_props.add_child(body)
	var flame := MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = Vector2(0.012, 0.03)
	flame.mesh = q
	var fm := ShaderMaterial.new()
	var sh := Shader.new()
	sh.code = FLAME_SHADER
	fm.shader = sh
	flame.material_override = fm
	flame.position = at + Vector3(0, h + 0.014, 0)
	_props.add_child(flame)
	var light := OmniLight3D.new()
	light.light_color = Color(1.0, 0.7, 0.4)
	light.omni_range = 0.45
	light.light_energy = 0.35
	# A FLAME CASTS SHADOWS: the deck and the cards, flickering with it
	light.shadow_enabled = true
	light.shadow_caster_mask = 0xFFFFFFFF & ~CANDLE_LAYER
	light.shadow_blur = 2.0
	light.omni_shadow_mode = OmniLight3D.SHADOW_CUBE
	light.position = at + Vector3(0, h + 0.03, 0)
	_props.add_child(light)
	_flames.append({"mesh": flame, "light": light, "seed": rng.randf() * 100.0, "base": flame.position})


func _tick_props(t: float) -> void:
	for f in _flames:
		var s := float((f as Dictionary)["seed"])
		var fl := 0.5 + 0.5 * _noise.get_noise_2d(t * 2.3, s)
		var flick := 0.5 + 0.5 * _noise.get_noise_2d(t * 9.0, s + 31.0)
		var mesh: MeshInstance3D = f["mesh"]
		mesh.scale = Vector3(0.92 + 0.12 * flick, 0.85 + 0.3 * fl, 1.0)
		mesh.position = (f["base"] as Vector3) + Vector3(_noise.get_noise_2d(t * 3.1, s + 7.0) * 0.0012, 0, 0)
		# a flame is a sprite facing the reader
		mesh.look_at(mesh.global_position + (_cam.global_position - mesh.global_position) * Vector3(1, 0, 1), Vector3.UP)
		mesh.rotate_object_local(Vector3.UP, PI)
		(f["light"] as OmniLight3D).light_energy = 0.22 + 0.18 * fl + 0.08 * flick


# --- the camera ----------------------------------------------------------------------------------------

## The reader's eye: still. The Camera dial lets it breathe a little (0 is a locked-off tripod).
func _tick_camera(t: float) -> void:
	var amt := clampf(Director.camera, 0.0, 2.0) * 0.5
	if amt <= 0.0:
		_cam.transform = _cam_base
		return
	var dx := _noise.get_noise_2d(t * 0.11, 13.0) * 0.004 * amt
	var dy := _noise.get_noise_2d(t * 0.09, 57.0) * 0.003 * amt
	var rot := _noise.get_noise_2d(t * 0.07, 91.0) * deg_to_rad(0.35) * amt
	var b := _cam_base.basis.rotated(_cam_base.basis.z, rot)
	_cam.transform = Transform3D(b, _cam_base.origin + _cam_base.basis.x * dx + _cam_base.basis.y * dy)


# --- the schedule ----------------------------------------------------------------------------------------

## When each card is drawn and laid, and when the shuffle starts and stops, from the placed
## schedule: `{shuffle, end, draw: [t0, s, kind], lay: [t0, s]}` - a time of INF is one the voice
## has not reached yet.
func _times() -> Dictionary:
	var n := _cards.size()
	var draw: Array = []
	var lay: Array = []
	for i in n:
		draw.append([INF, 1.0, "draw", 0.0])
		lay.append([INF, 1.0])
	var shuffle := INF
	var first := INF
	var spread := INF
	var prev := -1
	var first_act: Array = []
	for e in _sched:
		var a: Dictionary = e["a"]
		var t0 := float(e["t0"])
		var s := float(e["s"])
		var kind := String(a["kind"])
		if kind == "shuffle":
			shuffle = t0
			continue
		if prev >= 0 and prev < n:
			lay[prev] = [t0, s]
		# a draw's own phases start once the card before is down - or, for the first card, once
		# the deck has been pushed to its side (TarotScript.PUSH)
		var lay_off := TarotScript.LAY if prev >= 0 else (TarotScript.PUSH if kind == "draw" else 0.0)
		if first_act.is_empty() and (kind == "draw" or kind == "jumper"):
			first_act = [t0, s, kind]
		if kind == "draw" or kind == "jumper":
			var k := int(a["card"]) - 1
			if k >= 0 and k < n:
				draw[k] = [t0, s, kind, lay_off]
				first = minf(first, t0)
				prev = k
		elif kind == "spread":
			spread = t0
			prev = -1
	return {"shuffle": shuffle, "end": first, "draw": draw, "lay": lay, "spread": spread, "first": first_act}


## Where the deck is at [param t]: in the middle while it is shuffled, then pushed to its side
## as the first card comes - squared first (SQUARE), then slid (PUSH_SLIDE), a hair off the cloth.
## A jumper flies out of the middle and the deck goes once it has landed.
func _deck_at(t: float, tm: Dictionary) -> Vector3:
	var first: Array = tm["first"]
	if first.is_empty():
		return _mid
	var s := maxf(float(first[1]), 0.05)
	var go := SQUARE if String(first[2]) == "draw" else JUMP_FLY.y
	var u := clampf(((t - float(first[0])) / s - go) / PUSH_SLIDE, 0.0, 1.0)
	return _mid.lerp(_deck_base, _ease(u)) + Vector3(0.0, sin(PI * u) * 0.004, 0.0)


# --- posing everything --------------------------------------------------------------------------------------

func _pose(t: float) -> void:
	var tm := _times()
	# a shuffle that began in the replayed past of a mid-way start is under way NOW, not a hundred
	# thousand seconds in (the chain of moves is only made so far)
	var ts := maxf(float(tm["shuffle"]), minf(0.0, _now))
	var te := float(tm["end"])
	_cur_base = _deck_at(t, tm)
	# THE DECK: shuffling from the shuffle mark until the first card leaves it, squared after
	for i in DECK_N:
		var xf := _rest_xf(i)
		if t >= ts:
			if t < te:
				xf = _shuffle_xf(i, t - ts)
			elif t < te + SQUARE and te < INF:
				xf = _shuffle_xf(i, te - ts).interpolate_with(_rest_xf(i), _ease((t - te) / SQUARE))
		(_deck[i] as MeshInstance3D).transform = xf
	# THE CARDS
	var showing := -1
	var page_in := 0.0
	var draws: Array = tm["draw"]
	var lays: Array = tm["lay"]
	var glint_t := t
	for k in _cards.size():
		var m: MeshInstance3D = _cards[k]
		var d: Array = draws[k]
		var td := float(d[0])
		if t < td or td == INF:
			m.visible = false
			continue
		m.visible = true
		var s := maxf(float(d[1]), 0.05)
		var kind := String(d[2])
		var local := (t - td) / s - float(d[3])
		var l: Array = lays[k]
		var tl := float(l[0])
		var up_at := td + (float(d[3]) + (JUMP_RISE if kind == "jumper" else RISE_END)) * s
		var pres := _present_xf(k, t, up_at, tl)
		if local < 0.0:
			# the card before is still going down: this one is in the deck
			m.transform = _deck_top_xf(k)
			continue
		if tl < INF and t >= tl:
			var u := (t - tl) / maxf(float(l[1]), 0.05)
			var from := _present_xf(k, tl, up_at, tl)
			if u < LAY_END:
				var e := _ease(clampf((u - LAY_MOVE.x) / (LAY_MOVE.y - LAY_MOVE.x), 0.0, 1.0))
				var to := _slot_xf(k)
				var xf := from.interpolate_with(to, e)
				xf.origin.y += sin(PI * e) * 0.05
				if u > LAY_MOVE.y:
					xf.origin.y += (1.0 - clampf((u - LAY_MOVE.y) / (LAY_END - LAY_MOVE.y), 0.0, 1.0)) * 0.002
				m.transform = xf
				if u < LAY_PAGE_OUT:
					showing = k
					page_in = 1.0 - _ease(u / LAY_PAGE_OUT)
			else:
				m.transform = _slot_xf(k)
			continue
		if kind == "jumper":
			m.transform = _jump_xf(k, local, pres)
			if local >= JUMP_PAGE.x:
				showing = k
				page_in = _ease(clampf((local - JUMP_PAGE.x) / (JUMP_PAGE.y - JUMP_PAGE.x), 0.0, 1.0))
		else:
			m.transform = _draw_xf(k, local, pres)
			if local >= PAGE_IN.x:
				showing = k
				page_in = _ease(clampf((local - PAGE_IN.x) / (PAGE_IN.y - PAGE_IN.x), 0.0, 1.0))
		# THE FOIL breathes while a card is held up, and now and then a glint crosses it
		var mat: ShaderMaterial = _face_mats[k]
		mat.set_shader_parameter("pulse", 0.5 + 0.5 * sin(glint_t * TAU / 5.5 + float(k)))
		var cyc := fmod(maxf(glint_t - td, 0.0), 7.0)
		mat.set_shader_parameter("glint", -1.0 if cyc > 1.4 else lerpf(-0.25, 1.25, cyc / 1.4))
	_pose_page(showing, page_in, t)
	# the laid cards keep a slow breath of foil; the back's own glint rides the shuffle
	_back_mat.set_shader_parameter("pulse", 0.5 + 0.5 * sin(t * TAU / 7.0))


## Deck slot [param i] at rest - the squared deck, a hair off true per slot.
func _rest_xf(i: int) -> Transform3D:
	var j: Vector3 = _slot_jit[i] if i < _slot_jit.size() else Vector3.ZERO
	return Transform3D(Basis(Vector3.UP, j.z), _cur_base + Vector3(j.x, (float(i) + 0.5) * DECK_T, j.y))


## Drawn card [param k] lying on top of the deck, face down (the way it will come up).
func _deck_top_xf(k: int) -> Transform3D:
	var j: Vector3 = _slot_jit[DECK_N] if _slot_jit.size() > DECK_N else Vector3.ZERO
	return Transform3D(Basis(Vector3.UP, j.z + (PI if _reversed(k) else 0.0)),
		_cur_base + Vector3(j.x, float(DECK_N) * DECK_T + CARD_T * 0.5, j.y))


## Card [param k] held up to the camera on the left: floating, turning a little on its axes as a
## card in a hand does, and now and then turned over to look at its back ([method _turn_of]) -
## between [param up_at], when it is fully up, and [param until], when it goes down.
func _present_xf(k: int, t: float, up_at := INF, until := INF) -> Transform3D:
	var c := _cam_base.basis
	var b := Basis(-c.x, -c.z, -c.y)
	var rng_k := float(hash([_seed, k]) & 0xFF) / 255.0
	var yaw := deg_to_rad(lerpf(4.0, 10.0, rng_k)) + sin(t * 0.5 + rng_k * 6.0) * deg_to_rad(3.2) \
		+ sin(t * 0.21 + rng_k * 2.0) * deg_to_rad(1.8)
	var roll := deg_to_rad(lerpf(-2.5, 2.5, fmod(rng_k * 7.3, 1.0))) + sin(t * 0.37 + 1.0) * deg_to_rad(1.3)
	var tilt := sin(t * 0.29 + rng_k * 4.0) * deg_to_rad(2.6)
	b = Basis(c.y, yaw + _turn_of(k, t, up_at, until)) * Basis(c.x, tilt) * Basis(c.z, roll) * b
	if _reversed(k):
		b = Basis(c.z, PI) * b
	var off := Vector3(PRESENT.x + sin(t * 0.7 + rng_k) * 0.0012, PRESENT.y + sin(t * 0.9 + 1.3) * 0.0015, -PRESENT_DIST)
	return Transform3D(b, _cam_base * off)


## How far card [param k] is turned over at [param t] (0 face on, PI showing its back): a turn,
## a look at the back, a turn back, first some seconds after it is up and then every twenty or
## so - never while it is coming up or about to go down, and not for every card.
func _turn_of(k: int, t: float, up_at: float, until: float) -> float:
	if up_at == INF or t < up_at:
		return 0.0
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([_seed, k, "turn"])
	if rng.randf() > TURN_CHANCE:
		return 0.0
	var at := up_at + rng.randf_range(5.0, 9.0)
	for i in 64:
		var hold := rng.randf_range(TURN_HOLD.x, TURN_HOLD.y)
		var total := TURN * 2.0 + hold
		if t < at or at + total > until - 1.0:
			return 0.0
		var v := t - at
		if v < total:
			if v < TURN:
				return PI * _ease(v / TURN)
			if v < TURN + hold:
				return PI
			return PI * (1.0 - _ease((v - TURN - hold) / TURN))
		at += total + rng.randf_range(14.0, 24.0)
	return 0.0


## A drawn card at [param u] seconds into its draw.
func _draw_xf(k: int, u: float, pres: Transform3D) -> Transform3D:
	var top := _deck_top_xf(k)
	if u < SQUARE:
		return top
	if u < SLIDE_END:
		var e := _ease((u - SQUARE) / (SLIDE_END - SQUARE))
		var xf := top
		xf.origin += Vector3(0, 0.004 * e, 0.05 * e)
		return xf
	var slid := top
	slid.origin += Vector3(0, 0.004, 0.05)
	var lifted := slid.origin + Vector3(0, 0.085, 0.035)
	if u < FLIP_END:
		var e := _ease((u - SLIDE_END) / (FLIP_END - SLIDE_END))
		var b := slid.basis * Basis(Vector3(0, 0, 1), PI * e)
		return Transform3D(b, slid.origin.lerp(lifted, e))
	var flipped := Transform3D(slid.basis * Basis(Vector3(0, 0, 1), PI), lifted)
	if u < RISE_END:
		return flipped.interpolate_with(pres, _ease((u - FLIP_END) / (RISE_END - FLIP_END)))
	return pres


## A jumper at [param u] seconds into its leap.
func _jump_xf(k: int, u: float, pres: Transform3D) -> Transform3D:
	var top := _deck_top_xf(k)
	var land := Transform3D(Basis(Vector3.UP, 0.6 + (PI if _reversed(k) else 0.0)) * _face_up(),
		_jump_land + Vector3(0, CARD_T * 0.5, 0))
	if u < JUMP_FLY.x:
		return top
	if u < JUMP_FLY.y:
		var e := clampf((u - JUMP_FLY.x) / (JUMP_FLY.y - JUMP_FLY.x), 0.0, 1.0)
		var pos := top.origin.lerp(land.origin, e) + Vector3(0, sin(PI * e) * 0.13, 0)
		var b := Basis(Vector3.UP, 0.6 * e + 1.3 * PI * e) * top.basis * Basis(Vector3(0, 0, 1), 3.0 * PI * e)
		return Transform3D(b, pos)
	var settle := land
	settle.origin += Vector3(0.006, 0, 0.004) * clampf((u - JUMP_FLY.y) / 0.4, 0.0, 1.0)
	if u < JUMP_REST:
		return settle
	if u < JUMP_RISE:
		var e := _ease((u - JUMP_REST) / (JUMP_RISE - JUMP_REST))
		var xf := settle.interpolate_with(pres, e)
		xf.origin.y += sin(PI * e) * 0.03
		return xf
	return pres


func _pose_page(k: int, amount: float, t: float) -> void:
	_page.visible = k >= 0 and amount > 0.001
	if not _page.visible:
		return
	if k != _page_for:
		_page_for = k
		var cards: Array = _pay.get("cards", [])
		_page_canvas.card = cards[k] if k < cards.size() else {}
		TarotCards.redraw(_page_vp)
	var c := _cam_base.basis
	var yaw := -deg_to_rad(6.0) + sin(t * 0.45 + 2.0) * deg_to_rad(2.8) + sin(t * 0.19 + 0.7) * deg_to_rad(1.4)
	var tilt := sin(t * 0.33 + 1.1) * deg_to_rad(2.0)
	var b := Basis(c.y, yaw) * Basis(c.x, tilt) * Basis(c.z, sin(t * 0.27) * deg_to_rad(0.8)) * c
	var off := Vector3(PAGE.x + (1.0 - amount) * 0.22, PAGE.y + sin(t * 0.8 + 0.4) * 0.0012,
		-PRESENT_DIST - 0.004)
	_page.transform = Transform3D(b, _cam_base * off)


# --- the shuffle ------------------------------------------------------------------------------------------

## Deck slot [param i] at [param u] seconds into the shuffle, posed about where the deck is
## ([member _cur_base]). The shuffle is a chain of RUNS (see [constant RUNS], [method _move_at]).
##
## THE DECK'S MESHES ARE INTERCHANGEABLE: every move ends with slot i's mesh in some other slot
## j, and the next move starts it in slot i again. They are identical cards, and a slot's small
## offset belongs to the slot, so the hand-over is invisible.
func _shuffle_xf(i: int, u: float) -> Transform3D:
	var m := _move_at(u)
	if m.is_empty():
		return _rest_xf(i)
	var v := u - float(m["t0"])
	var dur := float(m["dur"])
	match String(m["kind"]):
		"riffle":
			return _riffle(i, v, dur, int(m["seed"]))
		"overhand":
			return _overhand(i, v, dur, int(m["seed"]))
		"cut":
			return _cut(i, v, dur, int(m["seed"]))
		"wash":
			return _wash(i, v, m)
	return _rest_xf(i)


## The move under way at [param u], or empty in a pause. RUNS, NOT A LOTTERY: a reader does not
## pick a new shuffle for every move - they riffle three times running, cut four times, and then
## leave the deck alone for a few seconds (now and then for a long while) while they talk. So a
## run is one kind at one tempo, and the pause after it is drawn from a long-tailed spread. A wash
## comes at most once, and never first. Moves are made as far as they are needed, in order, from
## their own seeded stream - so the chain is the same however a reading gets to a moment.
func _move_at(u: float) -> Dictionary:
	while _moves.is_empty() or float((_moves[-1] as Dictionary)["t0"]) + float((_moves[-1] as Dictionary)["dur"]) \
			+ float((_moves[-1] as Dictionary)["pause"]) < u:
		var t0 := 0.15
		var last := ""
		var washed := false
		for m in _moves:
			washed = washed or String((m as Dictionary)["kind"]) == "wash"
		if not _moves.is_empty():
			var p: Dictionary = _moves[-1]
			t0 = float(p["t0"]) + float(p["dur"]) + float(p["pause"])
			last = String(p["kind"])
		# the run's kind: weighted, never the same as the last run, no second wash, no wash first
		var total := 0.0
		var pool: Array = []
		for k in RUNS:
			if k == last or (k == "wash" and (washed or _moves.is_empty())):
				continue
			pool.append(k)
			total += float((RUNS[k] as Dictionary)["weight"])
		var pick := _move_rng.randf() * total
		var kind := String(pool[0])
		for k in pool:
			pick -= float((RUNS[k] as Dictionary)["weight"])
			if pick <= 0.0:
				kind = String(k)
				break
		var run: Dictionary = RUNS[kind]
		var n := _move_rng.randi_range(int(run["n"][0]), int(run["n"][1]))
		var tempo := _move_rng.randf_range(float(run["dur"][0]), float(run["dur"][1]))
		for i in n:
			var gap := _move_rng.randf_range(float(run["gap"][0]), float(run["gap"][1]))
			if i == n - 1:
				# the pause after a run: a few seconds, or now and then a long linger
				gap = clampf(exp(_move_rng.randfn(IDLE_LOG.x, IDLE_LOG.y)), IDLE_RANGE.x, IDLE_RANGE.y)
				if _move_rng.randf() < LINGER_CHANCE:
					gap = _move_rng.randf_range(9.0, IDLE_RANGE.y)
			var move := {"kind": kind, "t0": t0, "dur": tempo * _move_rng.randf_range(0.95, 1.05),
				"pause": gap, "seed": _move_rng.randi()}
			if kind == "wash":
				move["plan"] = _wash_plan(int(move["seed"]), float(move["dur"]))
			_moves.append(move)
			t0 += float(move["dur"]) + gap
		if _moves.size() > 400:
			break
	for m in _moves:
		var d: Dictionary = m
		if u >= float(d["t0"]) and u < float(d["t0"]) + float(d["dur"]):
			return d
	return {}


func _riffle(i: int, v: float, dur: float, seed: int) -> Transform3D:
	var half := int(DECK_N * 0.5)
	var left := i < half
	var k := i if left else i - half
	var side := -1.0 if left else 1.0
	var j := mini(2 * k + (0 if left else 1), DECK_N - 1)
	var split_end := 0.5
	var drop0 := 0.55
	var drop1 := dur - 0.6
	var fall := 0.11
	var held_rot := Basis(Vector3(0, 0, 1), side * deg_to_rad(9.0)) * Basis(Vector3.UP, side * deg_to_rad(6.0))
	var held := Transform3D(held_rot, _cur_base + Vector3(side * 0.052, 0.016 + (float(k) + 0.5) * DECK_T, 0.004))
	if v < split_end:
		return _rest_xf(i).interpolate_with(held, _ease(v / split_end))
	var tj := drop0 + (drop1 - drop0) * float(j) / float(DECK_N - 1)
	if v < tj:
		return held
	var messy := _messy(j, seed)
	if v < tj + fall:
		return held.interpolate_with(messy, (v - tj) / fall)
	var sq := clampf((v - drop1 - fall) / maxf(dur - drop1 - fall, 0.05), 0.0, 1.0)
	return messy.interpolate_with(_rest_xf(j), _ease(sq))


func _overhand(i: int, v: float, dur: float, seed: int) -> Transform3D:
	var k0 := int(DECK_N * 0.42)
	if i < k0:
		return _rest_xf(i)
	var held_n := DECK_N - k0
	var packets := 5
	var lift_end := 0.45
	var casc0 := 0.5
	var casc1 := dur - 0.45
	var held := Transform3D(Basis(Vector3.UP, deg_to_rad(-7.0)),
		_cur_base + Vector3(0.07, 0.032 + (float(i - k0) + 0.5) * DECK_T, 0.012))
	if v < lift_end:
		return _rest_xf(i).interpolate_with(held, _ease(v / lift_end))
	# packets peel off the TOP of the held cards and land in turn on the deck: the top packet
	# lands first, so the held cards' order is reversed packet by packet
	var from_top := DECK_N - 1 - i
	var p := mini(packets - 1, int(float(from_top) * float(packets) / float(held_n)))
	var lo := int(ceil(float(p) * float(held_n) / float(packets)))
	var hi := int(ceil(float(p + 1) * float(held_n) / float(packets)))
	var below := lo
	var in_packet := (DECK_N - 1 - lo) - i          # 0 = the packet's top card
	var size := hi - lo
	var j := k0 + below + (size - 1 - in_packet)
	var tp := casc0 + (casc1 - casc0) * float(p) / float(packets)
	var pd := (casc1 - casc0) / float(packets) * 0.85
	if v < tp:
		return held
	var messy := _messy(j, seed)
	if v < tp + pd:
		var e := _ease((v - tp) / pd)
		var xf := held.interpolate_with(messy, e)
		xf.origin.y += sin(PI * e) * 0.012
		return xf
	var sq := clampf((v - casc1) / maxf(dur - casc1, 0.05), 0.0, 1.0)
	return messy.interpolate_with(_rest_xf(j), _ease(sq))


## A CUT, quick enough to be done several times running: the top packet (from a point of the
## move's own) is lifted off and set down beside the deck, the rest is put on top of it, and the
## deck is drawn back to its place. Each cut in a run goes to its own side.
func _cut(i: int, v: float, dur: float, seed: int) -> Transform3D:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var a := rng.randi_range(int(DECK_N * 0.3), int(DECK_N * 0.7))
	var side := -1.0 if rng.randf() < 0.5 else 1.0
	var aside := Vector3(side * rng.randf_range(0.08, 0.095), 0.0, rng.randf_range(-0.012, 0.018))
	var top := i >= a
	# where each card ends: the old top packet underneath, the old bottom on it
	var j := (i - a) if top else (DECK_N - a + i)
	var out := Vector2(0.0, 0.36) * dur          # the top packet goes aside
	var over := Vector2(0.4, 0.74) * dur         # the rest goes on top of it
	var back := Vector2(0.8, 1.0) * dur          # the whole deck back to its place
	var spot := func(slot: int, off: Vector3) -> Transform3D:
		var r := _rest_xf(slot)
		r.origin += off
		return r
	if v >= back.x:
		var e := _ease((v - back.x) / (back.y - back.x))
		return (spot.call(j, aside) as Transform3D).interpolate_with(_rest_xf(j), e)
	if top:
		if v < out.x:
			return _rest_xf(i)
		var e := _ease(clampf((v - out.x) / (out.y - out.x), 0.0, 1.0))
		var xf := _rest_xf(i).interpolate_with(spot.call(j, aside), e)
		xf.origin.y += sin(PI * e) * 0.022
		return xf
	if v < over.x:
		return _rest_xf(i)
	var e2 := _ease(clampf((v - over.x) / (over.y - over.x), 0.0, 1.0))
	var xf2 := _rest_xf(i).interpolate_with(spot.call(j, aside), e2)
	xf2.origin.y += sin(PI * e2) * 0.028
	return xf2


## A WASH, posed from its plan (see [method _wash_plan]): card [param i] at [param v] seconds in.
func _wash(i: int, v: float, m: Dictionary) -> Transform3D:
	var plan: Dictionary = m["plan"]
	var track: PackedVector4Array = (plan["tracks"] as Array)[i]
	var f := clampf(v * WASH_HZ, 0.0, float(track.size() - 1))
	var a := int(floor(f))
	var b := mini(a + 1, track.size() - 1)
	var q := track[a].lerp(track[b], f - float(a))
	# x, z about the deck's place; y above the cloth; w the card's turn. A card lying alone is a
	# card's thickness, not a deck slot's: the slot mesh is thinned while it is spread
	var thin := lerpf(1.0, CARD_T / DECK_T, _wash_spread_at(plan, i, v))
	var basis := Basis(Vector3.UP, q.w).scaled(Vector3(1.0, thin, 1.0))
	return Transform3D(basis, _cur_base + Vector3(q.x, q.y, q.z))


## How spread out card [param i] is at [param v] (0 squared in the deck, 1 lying on its own).
func _wash_spread_at(plan: Dictionary, i: int, v: float) -> float:
	var t_out := float((plan["out"] as PackedFloat32Array)[i])
	var t_in := float((plan["in"] as PackedFloat32Array)[i])
	return clampf((v - t_out) / 0.5, 0.0, 1.0) * (1.0 - clampf((v - t_in) / 0.6, 0.0, 1.0))


## THE WASH, planned once and sampled at [constant WASH_HZ] - because it is a SIMULATION, not a
## pose: two flat hands circle the cloth and drag the cards near them along, and a pose that is
## a function of time alone cannot remember where a hand left a card. Planned from the move's
## seed, so it is the same every time it is posed.
##
##   out      the deck is pushed out across the cloth, top cards first (~2 s)
##   wash     two hands swirl: a card near a hand moves with it and turns as it is dragged past
##            its center; a card no hand reaches lies still
##   gather   three or four sweeps, each from its own side. A sweep takes the cards on its side,
##            outermost first, and pushes them in: most land on the pile, some only get pushed
##            near it and wait for a later sweep, and some are missed altogether
##   square   the pile is squared into the deck
##
## Each track is per card, x / z about the deck's place, y above the cloth, w the card's turn. A
## card's height is how many cards lie under it - cards stacked where they overlap, and flat on
## the cloth where they do not.
func _wash_plan(seed: int, dur: float) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "wash"])
	var n := DECK_N
	var steps := int(ceil(dur * WASH_HZ)) + 1
	var dt := 1.0 / WASH_HZ
	var out_end := 2.2
	var square := 0.8
	var swipes := rng.randi_range(3, 4)
	var gather := clampf(dur * 0.36, 3.0, 4.6)
	var wash_end := dur - square - gather
	var rx := rng.randf_range(0.14, 0.17)
	var rz := rng.randf_range(0.075, 0.095)
	var pos: Array = []
	var yaw := PackedFloat32Array()
	var start: Array = []
	var aim: Array = []
	var t_out := PackedFloat32Array()
	for i in n:
		var jit: Vector3 = _slot_jit[i] if i < _slot_jit.size() else Vector3.ZERO
		start.append(Vector2(jit.x, jit.y))
		var r := sqrt(rng.randf()) * 0.95
		var ang := rng.randf() * TAU
		aim.append(Vector2(cos(ang) * rx * r, sin(ang) * rz * r - 0.01))
		pos.append(Vector2(jit.x, jit.y))
		yaw.append(jit.z)
		t_out.append(float(n - 1 - i) / float(n) * 0.9)
	var spin0 := PackedFloat32Array()
	for i in n:
		spin0.append(rng.randf_range(-1.1, 1.1))
	# the hands: smooth loops over the spread, easing in and out
	var hp: Array = []
	for h in 2:
		hp.append([rng.randf_range(0.6, 1.1), rng.randf_range(0.9, 1.5), rng.randf() * TAU, rng.randf() * TAU,
			rng.randf_range(0.7, 1.0)])
	var hand := func(h: int, tt: float) -> Vector2:
		var q: Array = hp[h]
		var side := -0.45 if h == 0 else 0.45
		return Vector2(rx * (side + 0.5 * sin(float(q[0]) * tt + float(q[2]))) * float(q[4]),
			rz * 0.8 * sin(float(q[1]) * tt + float(q[3])) - 0.01)
	# the sweeps, one after another across the gather
	var sweep_len := gather / float(swipes)
	var caught := PackedInt32Array()       # the sweep that brings each card in for good
	var nudged := PackedInt32Array()       # a sweep that only pushes it near the pile, or -1
	caught.resize(n)
	nudged.resize(n)
	# plain Arrays while they grow: appending to `tracks[i] as PackedVector4Array` appends to a COPY
	var tracks: Array = []
	for i in n:
		tracks.append([])
	var t_in := PackedFloat32Array()       # when a card is in the pile for good
	var t_go := PackedFloat32Array()       # when the sweep that brings it in reaches it
	t_in.resize(n)
	t_go.resize(n)
	var from_p: Array = []                 # a sweep under way: where the card was, where it goes
	var to_p: Array = []
	var from_yaw := PackedFloat32Array()
	var to_yaw := PackedFloat32Array()
	from_yaw.resize(n)
	to_yaw.resize(n)
	for i in n:
		from_p.append(Vector2.ZERO)
		to_p.append(Vector2.ZERO)
	var begun := {}
	var land_order: Array = []
	var planned_sweeps := false
	var height := PackedFloat32Array()
	height.resize(n)
	for step in steps:
		var t := float(step) * dt
		if t < out_end:
			# OUT: each card slides from the deck to its place on the cloth, turning as it goes
			for i in n:
				var e := _ease(clampf((t - t_out[i]) / 1.1, 0.0, 1.0))
				pos[i] = (start[i] as Vector2).lerp(aim[i], e)
				yaw[i] = (_slot_jit[i] as Vector3).z + spin0[i] * e if i < _slot_jit.size() else spin0[i] * e
		elif t < wash_end:
			# WASH: the hands drag what they touch
			var env := clampf((t - out_end) / 0.8, 0.0, 1.0) * clampf((wash_end - t) / 0.8, 0.0, 1.0)
			for h in 2:
				var p0: Vector2 = hand.call(h, t - out_end)
				var p1: Vector2 = hand.call(h, t - out_end + dt)
				var vel := (p1 - p0) * env
				for i in n:
					var off: Vector2 = (pos[i] as Vector2) - p0
					var w := 1.0 - smoothstep(0.018, 0.075, off.length())
					if w <= 0.0:
						continue
					pos[i] = (pos[i] as Vector2) + vel * w * 0.85
					yaw[i] += (off.x * vel.y - off.y * vel.x) / maxf(off.length_squared(), 0.0004) * w * 0.35
					# kept on the cloth, softly
					var q: Vector2 = pos[i]
					var k := (q.x / rx) * (q.x / rx) + ((q.y + 0.01) / rz) * ((q.y + 0.01) / rz)
					if k > 1.2:
						pos[i] = q * lerpf(1.0, sqrt(1.2 / k), 0.3)
		else:
			if not planned_sweeps:
				planned_sweeps = true
				# who each sweep takes: the sweeps go round the pile, each taking the next SHARE of
				# the cards by direction - so every sweep has cards to bring in, wherever the hands
				# left them (fixed directions sent whole sweeps past empty cloth). A card is
				# missed now and then and a later sweep fetches it; some are only pushed near.
				var by_angle: Array = []
				for i in n:
					var q: Vector2 = pos[i]
					by_angle.append([atan2(q.y, q.x), i])
				by_angle.sort_custom(func(x: Array, y: Array) -> bool: return float(x[0]) < float(y[0]))
				var first_card := rng.randi_range(0, n - 1)
				var share := PackedInt32Array()
				share.resize(n)
				for r in n:
					share[int((by_angle[(first_card + r) % n] as Array)[1])] = mini(swipes - 1, int(float(r) * float(swipes) / float(n)))
				for i in n:
					var k2 := share[i]
					while k2 < swipes - 1 and rng.randf() < 0.25:
						k2 += 1          # missed: a later sweep fetches it
					caught[i] = k2
					nudged[i] = -1
					if k2 < swipes - 1 and rng.randf() < 0.3:
						nudged[i] = k2   # caught, but only pushed near: brought in by the next
						caught[i] = k2 + 1
				# the pile's order is the order they arrive in
				var order: Array = []
				for i in n:
					order.append([caught[i], (pos[i] as Vector2).length(), i])
				order.sort_custom(func(x: Array, y: Array) -> bool:
					if int(x[0]) != int(y[0]):
						return int(x[0]) < int(y[0])
					return float(x[1]) > float(y[1]))
				for o in order:
					land_order.append(int(o[2]))
			var g := t - wash_end
			for i in n:
				for k in [nudged[i], caught[i]]:
					if int(k) < 0:
						continue
					# the sweep reaches the outermost cards first and pushes them in ahead of it
					var s0 := float(k) * sweep_len
					var u1 := s0 + sweep_len * 0.82
					var key := i * 8 + int(k)
					if not begun.has(key):
						var r := clampf((pos[i] as Vector2).length() / maxf(rx, rz), 0.0, 1.0)
						var u0 := s0 + (1.0 - r) * 0.28
						if g < u0:
							continue
						begun[key] = u0
						from_p[i] = pos[i]
						from_yaw[i] = yaw[i]
						var slot := land_order.find(i)
						if int(k) == caught[i]:
							# into the pile, squared-ish, on top of what is there
							var jit2: Vector3 = _slot_jit[slot] if slot >= 0 and slot < _slot_jit.size() else Vector3.ZERO
							to_p[i] = Vector2(jit2.x, jit2.y) + Vector2(rng.randf_range(-0.004, 0.004), rng.randf_range(-0.004, 0.004))
							to_yaw[i] = jit2.z + rng.randf_range(-0.12, 0.12)
							t_go[i] = wash_end + u0
							t_in[i] = wash_end + u1
						else:
							# caught, but only pushed up against the pile: the next sweep brings it in
							var away: Vector2 = (pos[i] as Vector2).normalized() if (pos[i] as Vector2).length() > 0.001 else Vector2.RIGHT
							to_p[i] = away * rng.randf_range(0.035, 0.055)
							to_yaw[i] = yaw[i] + rng.randf_range(-0.25, 0.25)
					var b0 := float(begun[key])
					if g > u1 + dt:
						continue
					var e2 := _ease(clampf((g - b0) / maxf(u1 - b0, 0.05), 0.0, 1.0))
					pos[i] = (from_p[i] as Vector2).lerp(to_p[i], e2)
					yaw[i] = lerp_angle(from_yaw[i], to_yaw[i], e2)
		# heights: a card rests on the cards under it that it overlaps - flat on the cloth alone
		for i in n:
			var under := 0
			for j in range(0, i):
				if (pos[i] as Vector2).distance_to(pos[j]) < 0.085:
					under += 1
			height[i] = 0.0004 + float(under) * (CARD_T + 0.0001)
		for i in n:
			var q: Vector2 = pos[i]
			var y := float(height[i])
			# leaving the deck and coming back to it, a card is at its deck slot's height - rising
			# early on its way in, so it lands ON the pile rather than sliding through it
			var stacked := 1.0 - clampf((t - t_out[i]) / 0.5, 0.0, 1.0)
			if t_in[i] > 0.0 and t >= t_go[i]:
				var w := clampf((t - t_go[i]) / maxf(t_in[i] - t_go[i], 0.05), 0.0, 1.0)
				stacked = 1.0 - pow(1.0 - w, 3.0)
			var slot_y := (float(land_order.find(i)) if t_in[i] > 0.0 else float(i)) * DECK_T + DECK_T * 0.5
			(tracks[i] as Array).append(Vector4(q.x, lerpf(y, slot_y, stacked), q.y, yaw[i]))
	# SQUARE: from wherever the pile left each card to its slot in the deck
	var last := steps - 1
	var sq0 := int(floor((dur - square) * WASH_HZ))
	for i in n:
		var slot := land_order.find(i)
		if slot < 0:
			slot = i
		var jit3: Vector3 = _slot_jit[slot] if slot < _slot_jit.size() else Vector3.ZERO
		var tr := PackedVector4Array(tracks[i] as Array)
		var from: Vector4 = tr[mini(sq0, tr.size() - 1)]
		var to := Vector4(jit3.x, (float(slot) + 0.5) * DECK_T, jit3.y, jit3.z)
		for st in range(sq0, tr.size()):
			var e3 := _ease(float(st - sq0) / maxf(float(last - sq0), 1.0))
			tr[st] = from.lerp(to, e3)
		tracks[i] = tr
	return {"tracks": tracks, "out": t_out, "in": t_in, "spread": 1.0, "order": land_order}


## A card just landed on the deck, not yet squared.
func _messy(j: int, seed: int) -> Transform3D:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, j, "messy"])
	var xf := _rest_xf(j)
	xf.origin += Vector3(rng.randf_range(-0.003, 0.003), 0.0, rng.randf_range(-0.003, 0.003))
	xf.basis = Basis(Vector3.UP, rng.randf_range(-0.07, 0.07)) * xf.basis
	return xf


static func _ease(x: float) -> float:
	var t := clampf(x, 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)


# --- the title -------------------------------------------------------------------------------------------

## The channel's name and the episode's title over the table while the intro holds, gone as the
## shuffle starts; the channel's name again as the reading closes.
func _title_alpha(t: float) -> float:
	var tm := _times()
	var ts := float(tm["shuffle"])
	var a := clampf((t - 0.3) / 1.1, 0.0, 1.0)
	if ts < INF:
		a *= 1.0 - clampf((t - (ts - 0.6)) / 0.9, 0.0, 1.0)
	elif not _sched.is_empty():
		a = 0.0
	_title.ending = false
	var sp := float(tm["spread"])
	if sp < INF:
		var last := _follow.known_last()
		var total := (_parse.get("spoken", PackedStringArray()) as PackedStringArray).size()
		if last >= total - 1 and last >= 0:
			var end_t := _follow.st1[last]
			var e := clampf((t - (end_t + 1.2)) / 1.4, 0.0, 1.0)
			if e > 0.0:
				_title.ending = true
				return e
	return a


class TitleCard:
	extends Node2D

	var channel := ""
	var episode := ""
	var face: Font = null
	var italic: Font = null
	var alpha := 0.0
	var ending := false

	func _draw() -> void:
		if alpha <= 0.001 or face == null or channel.is_empty():
			return
		var vp := get_viewport_rect().size
		var s := vp.y / 1080.0
		var size := int(84.0 * s)
		# the end card sits high, over the far edge, so the spread it closes on stays clear
		var y := vp.y * (0.4 if not ending else 0.2)
		# A SHADE BEHIND THE TYPE: the cloth is whatever the episode painted, often pale
		var band := 260.0 * s
		for i in 12:
			var f := float(i) / 11.0
			var h := band * (1.0 - f * 0.85)
			draw_rect(Rect2(0, y - 60.0 * s - h * 0.5 + 30.0 * s, vp.x, h), Color(0, 0, 0, 0.045 * alpha))
		var shadow := Color(0, 0, 0, 0.55 * alpha)
		var ink := Color(1.0, 0.97, 0.9, alpha)
		size = TarotCards._fit(face, channel, size, vp.x * 0.8)
		for o in [Vector2(2, 3), Vector2(0, 0)]:
			draw_string(face, Vector2(0, y) + (o as Vector2) * s, channel, HORIZONTAL_ALIGNMENT_CENTER,
				vp.x, size, shadow if o != Vector2(0, 0) else ink)
		if ending or episode.is_empty() or italic == null:
			return
		var es := TarotCards._fit(italic, episode, int(40.0 * s), vp.x * 0.78)
		var ey := y + 70.0 * s
		draw_string(italic, Vector2(2, ey + 2) * Vector2(1, 1), episode, HORIZONTAL_ALIGNMENT_CENTER, vp.x, es, shadow)
		draw_string(italic, Vector2(0, ey), episode, HORIZONTAL_ALIGNMENT_CENTER, vp.x, es, ink)
