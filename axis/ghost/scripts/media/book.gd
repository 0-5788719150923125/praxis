extends Medium
class_name BookMedium

## BookMedium - the reading as a printed novel, open on a desk, with the words lit as they
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
## placeholder this medium owns ([method take_over]), so nothing is built and nothing draws.
## A book that later wants scenes in its plates would cast them the way the comic does.
##
## REAL 3D. The stage SubViewport has a world of its own, so the book is geometry in it: a
## Camera3D, a lamp, a desk, a cloth cover, two page stacks and three leaves (the left page,
## the right page, and the one being turned), each leaf a subdivided plane bent by
## `LEAF_SHADER` into the gutter curve and, while turning, rolled over the spine with a curl.
## Each visible page is a SubViewport drawn by [BookMedium.PageCanvas]; a page is redrawn
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
## THE CAMERA'S ARC OVER ONE SPREAD. Every spread opens WIDE on both pages, closes in on the
## text being read as the reading gets under way, and pulls back out to wide before the
## leaf turns - so a page is introduced, read, and let go. It replaced a shot schedule that
## cut between wide, page and close on a clock, which zoomed in and out several times on
## one unchanging spread for no reason ("that zooming is a pointless action when the page
## hasn't changed"). `ARC_IN` / `ARC_OUT` are the shares of the spread's words spent closing
## in and pulling back; the spring does the easing.
const WIDE := {"dist": 5.3, "pitch": 66.0}
const LOCAL := {"dist": 3.3, "pitch": 71.0}
const ARC_IN := 0.12
const ARC_OUT := 0.12
## The camera springs' time constant at Camera 1. Slow, on purpose.
const SPRING_TAU := 3.2
## The slow angle: each spread's own yaw (+/- degrees), a wander over minutes on top, a
## whisper of roll, and how many times slower than the framing the yaw settles.
const YAW_SPREAD := 9.0
const YAW_WANDER := 3.0
## Clockwise/counter-clockwise roll and forward/back tilt, the same way: an offset per spread
## plus a slow wander. Degrees; all through the slow spring, so none of it reads as a move.
const ROLL_SPREAD := 2.5
const ROLL_WANDER := 1.5
const TILT_SPREAD := 2.5
const TILT_WANDER := 1.8
const ANGLE_SLOW := 3.0
## A LONG LENS. A wide one stood close makes every downstroke on the page converge on one
## vanishing point, so a line of roman type leans like italics toward the edges of the frame -
## the whole page reads as emphasis. Standing back with a narrow field keeps print upright.
const VFOV := 20.0
## THE HIGHLIGHT IS A TRAIL THAT DECAYS IN TIME. Each word lights in its own rainbow hue the
## moment it is spoken and fades back to ink over [constant TRAIL_TAU] seconds, whatever the
## sentence and paragraph boundaries - so the colour follows the voice down the page and
## cools behind it. The first cut coloured the SENTENCE, with hues drifting over time: every
## sentence reset the colour, and a long one sat bright and shifting for as long as it was on
## screen ("there is no continuity"). A word's hue is FIXED by its position, so a word never
## changes colour, it only fades. Words not yet spoken are plain ink. Darkened from the
## overlay's values to read as ink on paper.
const HUE_STEP := 0.011      # hue per LETTER along the text - the subtitles' HUE_SPAN
const TRAIL_TAU := 2.6       # seconds for a spoken word's colour to fall to about a third

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
	// cloth to show through. ALONG THE LEAF'S OWN DIRECTION at the spine, not along world x:
	// a turning leaf that has swung over is flipped, and a world-x overlap put its landed
	// text 0.012 to the side of the flat page that takes over from it - the text jumped as
	// every turn finished.
	VERTEX = vec3((p.x - 0.006 * cos(angle)) * side, p.y - 0.006 * sin(angle) + base_y, z);
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
var _title := ""                 # ...and the chapter title it was built with
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
var _lay_t := {}                 # layout word -> when it finished being spoken (show time)
var _lay_t0 := {}                # ...and when it began
var _char0 := PackedInt32Array() # layout word -> its first character's index in the text
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
var _cover: MeshInstance3D       # the right half of the case, under the right-hand pages
var _cover_l: MeshInstance3D     # the left half - the FRONT cover, carried by _pivot
var _pivot: Node3D               # hinges the whole left half of the book at the spine
var _cover_label: Label3D        # the book's name, on the outside of the front cover
var _cover_author: Label3D       # ...and its author, smaller, at the foot of the cover
var _placeholder: GhostScene

var _paper := Color(0.95, 0.925, 0.87)
var _ink := Color(0.11, 0.095, 0.085)
var _seed := 0
var _seed_hue := 0.0             # where the sentence rainbow starts, per session

# the camera's springs: value and velocity per channel
var _c_aim := Vector3.ZERO
var _v_aim := Vector3.ZERO
var _c_dist := 5.3
var _v_dist := 0.0
var _c_pitch := 60.0
var _v_pitch := 0.0
var _c_yaw := 0.0
var _v_yaw := 0.0
var _c_roll := 0.0
var _v_roll := 0.0
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
	# DIRT, not wood grain: soft gaussian blotches over a finer mottle, low contrast, so the
	# desk stops reading as a flat colour without becoming a surface anyone looks at. The
	# texture sits around mid-grey, which is why the albedo is set at twice the wanted brown.
	dmat.albedo_color = Color(0.40, 0.24, 0.15)
	dmat.albedo_texture = _grime(0x5EED, 0.02, 5, 0.16)
	dmat.uv1_scale = Vector3(8.0, 8.0, 1.0)
	dmat.roughness = 0.62
	dmat.roughness_texture = dmat.albedo_texture
	desk.material_override = dmat
	desk.position = Vector3(0.0, -0.075, 0.0)
	_root3.add_child(desk)

	# THE CASE IN TWO HALVES, because the book OPENS: the left half - front cover, left-hand
	# stack and left leaf - hangs off one pivot at the spine, folded over the right half at the
	# start of a session and swung open to the first spread (see [method _open_now]).
	var cm := BoxMesh.new()
	cm.size = Vector3(1.06, 0.03, PAGE_H + 0.1)
	var cmat := StandardMaterial3D.new()
	cmat.albedo_color = Color(0.32, 0.07, 0.06)
	# BOOK CLOTH: a fine, very faint mottle - the weave and handling of a used cover, barely
	# there. Same mid-grey convention as the desk, so begin_session doubles the cloth colour.
	cmat.albedo_texture = _grime(0xC10F, 0.09, 3, 0.06)
	cmat.uv1_scale = Vector3(2.0, 2.0, 1.0)
	cmat.roughness = 0.85
	_cover = MeshInstance3D.new()
	_cover.mesh = cm
	_cover.material_override = cmat
	_cover.position = Vector3(0.53, -0.058, 0.0)
	_root3.add_child(_cover)
	_pivot = Node3D.new()
	_root3.add_child(_pivot)
	_cover_l = MeshInstance3D.new()
	_cover_l.mesh = cm
	_cover_l.material_override = cmat          # one material: begin_session colours both
	_cover_l.position = Vector3(-0.53, -0.058, 0.0)
	_pivot.add_child(_cover_l)
	# THE NAME ON THE COVER lies on the cover's UNDERSIDE, facing down, reading toward the far
	# edge - so when the pivot folds the half over (a half turn about the spine) it faces up
	# and reads the right way round. Open, it faces the desk and nothing sees it.
	_cover_label = Label3D.new()
	var cf := SystemFont.new()
	cf.font_names = PackedStringArray(BookLayout.SERIFS)
	_cover_label.font = cf
	_cover_label.font_size = 80
	_cover_label.pixel_size = 0.0016
	_cover_label.modulate = Color(0.88, 0.74, 0.44)
	_cover_label.outline_size = 0
	_cover_label.double_sided = false
	_cover_label.shaded = true
	_cover_label.transform = Transform3D(Basis(Vector3(-1, 0, 0), Vector3(0, 0, -1),
		Vector3(0, -1, 0)), Vector3(-0.53, -0.0745, -0.22))
	_pivot.add_child(_cover_label)
	# THE AUTHOR, set the way a cover sets a byline: centred near the foot, much smaller than
	# the title and a shade quieter, in the same gilt. Same orientation trick as the title.
	_cover_author = Label3D.new()
	_cover_author.font = cf
	_cover_author.font_size = 40
	_cover_author.pixel_size = 0.0016
	_cover_author.modulate = Color(0.88, 0.74, 0.44, 0.85)
	_cover_author.outline_size = 0
	_cover_author.double_sided = false
	_cover_author.shaded = true
	_cover_author.transform = Transform3D(Basis(Vector3(-1, 0, 0), Vector3(0, 0, -1),
		Vector3(0, -1, 0)), Vector3(-0.53, -0.0745, 0.56))
	_pivot.add_child(_cover_author)

	_stack_l = _stack()
	_stack_r = _stack()
	_leaf_l = _leaf(-1.0)
	for m in [_stack_l, _leaf_l]:
		_root3.remove_child(m)
		_pivot.add_child(m)
	_leaf_r = _leaf(1.0)
	_leaf_t = _leaf(1.0)
	_leaf_t.visible = false


## A seamless, low-contrast noise texture around mid-grey: [param freq] sets the blotch size
## (per pixel of a 512 texture), [param octaves] how much finer detail rides on it, and
## [param contrast] how far it strays from grey. Built SYNCHRONOUSLY - a NoiseTexture2D
## generates on a thread, and an export's first frames would show the bare colour.
static func _grime(seed: int, freq: float, octaves: int, contrast: float) -> Texture2D:
	var n := FastNoiseLite.new()
	n.seed = seed
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	n.frequency = freq
	n.fractal_type = FastNoiseLite.FRACTAL_FBM
	n.fractal_octaves = octaves
	n.fractal_gain = 0.55
	var img := n.get_seamless_image(512, 512)
	img.convert(Image.FORMAT_RGB8)
	img.adjust_bcs(1.0, contrast, 1.0)
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


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
	_seed_hue = float(hash([_seed, "hue"]) & 0xFFFF) / 65535.0
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
	var cl: Color = cloth[r.randi() % cloth.size()]
	# x2: the cloth texture sits around mid-grey (see _build_world)
	(_cover.material_override as StandardMaterial3D).albedo_color = Color(
		minf(cl.r * 2.0, 1.0), minf(cl.g * 2.0, 1.0), minf(cl.b * 2.0, 1.0))
	# the gilt title in a colour that belongs to the cloth: lighter, warmer, never white
	_cover_label.modulate = Color(0.88, 0.74, 0.44)
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
	_lay_t = {}
	_lay_t0 = {}
	for i in 4:
		_slot_state[i] = ""


# --- the Medium contract ----------------------------------------------------

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
	# THE BOOK FADES IN ON ITS OWN, quickly. The Director's fade-in lasts the whole intro hold,
	# so the cover was still in the dark while the book opened: "I only see just the briefest
	# flash of that cover". The Director's value still brings the ending down.
	var t := maxf(Spectrum.current.time, 0.0)
	var slen := Spectrum.song_length()
	var b := bookend
	if slen <= 0.0 or t < slen * 0.5:
		b = clampf(t / FADE_IN, 0.0, 1.0)
	_env.adjustment_brightness = clampf(b, 0.0, 1.0)
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
	var title := ""
	var words: Array = _subs.words if _subs != null and is_instance_valid(_subs) else []
	if _subs != null and is_instance_valid(_subs):
		var doc: Dictionary = _subs.document
		src = String(doc.get("source", ""))
		# The title and the book's name travel BESIDE the text: in sync mode the text arrives
		# without its frontmatter, so reading them off the source gave a book with no chapter
		# title in it.
		title = String(doc.get("title", ""))
		if title.is_empty():
			title = BookLayout.field_of(src, "title")
		var bk := String(doc.get("book", ""))
		var au := String(doc.get("author", ""))
		_set_cover(bk if not bk.is_empty() else BookLayout.field_of(src, "book"),
			au if not au.is_empty() else BookLayout.field_of(src, "author"))
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
	if _layout != null and src == _source and title == _title and _rev == Illustrations.revision:
		return
	var reflow := _layout != null and src == _source
	if _rev != Illustrations.revision:
		_textures = {}              # a reroll is a new file; let the old pictures go
	_source = src
	_rev = Illustrations.revision
	var lay := BookLayout.new()
	_title = title
	lay.build(src, _image_size, title)
	_layout = lay
	# Every word's first character as an index into the running text, so each LETTER has a
	# hue of its own that never depends on where a line or a sentence happens to break.
	_char0 = PackedInt32Array()
	var c := 0
	for w in lay.words:
		_char0.append(c)
		c += String((w as Dictionary)["text"]).length() + 1
	_map = []
	_map_j = 0
	_map_rem = ""
	_sent_lo = {}
	_sent_hi = {}
	_lay_t = {}
	_lay_t0 = {}
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
		_lay_t = {}
		_lay_t0 = {}
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
			_lay_t[got] = float((words[i] as Dictionary).get("t1", 0.0))
			if not _lay_t0.has(got):
				_lay_t0[got] = float((words[i] as Dictionary).get("t0", 0.0))
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


## THE BOOK STARTS CLOSED and opens to the first spread: 0 closed, 1 open. A pure function
## of show time, so a render opens exactly as the live session did. It fades in over
## [constant FADE_IN], sits closed until [constant OPEN_DELAY] so the cover is actually seen,
## then opens over [constant OPEN_TIME] - slowly, the way a book is opened, not flipped.
const FADE_IN := 1.5
const OPEN_DELAY := 3.5
const OPEN_TIME := 5.0

func _open_now() -> float:
	var t := maxf(Spectrum.current.time, 0.0)
	return smoothstep(0.0, 1.0, clampf((t - OPEN_DELAY) / OPEN_TIME, 0.0, 1.0))


func _set_cover(title: String, author: String) -> void:
	if _cover_label != null and _cover_label.text != title:
		_cover_label.text = title
	if _cover_author != null and _cover_author.text != author:
		_cover_author.text = author


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
		# The swell is along the leaf's OWN normal, which points DOWN once the leaf is past
		# vertical - so an unsigned lift sank the landing leaf 0.05 under the page it was
		# covering, and the old page (a full-page picture, most visibly) showed through for
		# the last frames of every turn. Signed by side; it is 0 at vertical, so continuous.
		mat.set_shader_parameter("lift", 0.05 * (1.0 - sin(a)) * (1.0 if a < PI * 0.5 else -1.0))
		_bind(_leaf_t, _spread * 2 + 1, _turn_to * 2, lerpf(yr, yl, k) + 0.003)
	# THE OPENING: fold the left half over the right about a hinge just above both blocks, so
	# closed it rests on top of the right-hand pages, cover up; open it lies flat on the left.
	# The hinge has to sit ABOVE both blocks while the book is closed, so the folded half rests
	# on top of the right-hand pages - but a leaf turning about a point above its own spine
	# edge swings that edge away from the spine, and the cloth showed through the gap between
	# the pages as the book opened. So the hinge slides down to the left leaf's own surface by
	# halfway: from there the page turns about its spine edge, and nothing opens up.
	var op := _open_now()
	var hinge := lerpf((yl + yr) * 0.5 + 0.06, yl, smoothstep(0.0, 0.5, op))
	var th := -PI * (1.0 - op)
	_pivot.transform = Transform3D(Basis.IDENTITY, Vector3(0.0, hinge, 0.0)) \
		* Transform3D(Basis(Vector3(0.0, 0.0, 1.0), th), Vector3.ZERO) \
		* Transform3D(Basis.IDENTITY, Vector3(0.0, -hinge, 0.0))


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
	var now: float = _subs.now() if _subs != null and is_instance_valid(_subs) else 0.0
	# Glowing here if the voice is on this page, or has just left it and the trail is still
	# cooling; otherwise the page is settled and never redraws.
	var cooling := li > hi and now - float(_lay_t.get(hi, -1e9)) < TRAIL_TAU * 4.0
	if li < lo or (li > hi and not cooling):
		return {"read": li} if li > hi else {}
	# Quantised, so a page redraws some twenty times a second while its trail cools rather
	# than on every float wobble.
	return {"word": li, "now": snappedf(now, 0.05), "read": li,
		"frac": snappedf(float(r["frac"]), 0.02), "alpha": snappedf(float(r["alpha"]), 0.02)}


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
	var now := float(hl.get("now", 0.0))
	var frac := float(hl.get("frac", 0.0))
	for wi in pg["words"]:
		var w: Dictionary = _layout.words[int(wi)]
		var ink := _ink
		var i := int(wi)
		if read >= 0 and i < read:
			ink = _ink.lerp(_paper, 0.12)         # read: the faintest lift of the ink
		var font := _layout.face(int(w["emph"]))
		var text := String(w["text"])
		var fs := int(w["fs"])
		var lit := li >= 0 and alpha > 0.0 and i <= li
		if lit and i < li and now - float(_lay_t.get(i, -1e9)) > TRAIL_TAU * 4.0:
			lit = false                            # long cooled: all ink
		# EVERY WORD IS DRAWN THE SAME WAY, lit or not: its own shaped glyphs, each at its own
		# shaped position, each given a colour. Switching from a whole-word draw to a letter-
		# by-letter one when the highlight arrived moved letters by a pixel - two drawing
		# paths snap glyphs to the pixel grid differently, and no offset arithmetic makes
		# them agree ("the k shifts weirdly when the highlights touch it"). With one path the
		# highlight can only ever change a colour.
		var t0 := float(_lay_t0.get(i, now))
		var t1 := float(_lay_t.get(i, now))
		var n := text.length()
		var base: Vector2 = w["base"]
		for gl in _glyphs(font, text, fs):
			var k := int(gl["start"])
			var col := ink
			if lit:
				var at := (float(k) + 0.5) / float(maxi(n, 1))
				var g := 0.0
				if i < li:
					g = exp(-maxf(now - lerpf(t0, t1, at), 0.0) / TRAIL_TAU)
				elif at <= frac:
					# the word being said: lit up to the eased cursor, freshest at the front
					g = exp(-maxf((frac - at) * maxf(t1 - t0, 0.05), 0.0) / TRAIL_TAU)
				if g > 0.01:
					var ci_i := _char0[i] + k if i < _char0.size() else k
					var hue := fposmod(_seed_hue + float(ci_i) * HUE_STEP, 1.0)
					# the subtitles' saturation wave: two slow incommensurate ripples along
					# the text, so the colour breathes instead of sitting at one intensity
					var sw := 0.5 + 0.35 * sin(float(ci_i) * 0.21 - now * 0.9) \
						+ 0.15 * sin(float(ci_i) * 0.36 + now * 0.45)
					var sat := lerpf(0.45, 0.9, clampf(sw, 0.0, 1.0))
					col = ink.lerp(Color.from_hsv(hue, sat, 0.55), g * alpha)
			_ts.font_draw_glyph(gl["rid"], ci.get_canvas_item(), fs, base + (gl["pos"] as Vector2),
				int(gl["index"]), col)
	for lb in pg["labels"]:
		var l: Dictionary = lb
		ci.draw_string(_layout.face(int(l["emph"])), l["pos"], String(l["text"]),
			HORIZONTAL_ALIGNMENT_CENTER, float(l["align_w"]), int(l["fs"]),
			Color(_ink, float(l.get("tone", 1.0))))
	for im in pg["images"]:
		_draw_image(ci, im)
	# NO PAGE NUMBERS: this chapter does not start on page 2 of the real book, and a folio
	# that is wrong is worse than none.


## A word's shaped glyphs: `[{index, rid, pos, start}]`, positions relative to the baseline
## origin, kerning and fallback faces included - exactly what a whole-word draw would place.
## Cached per (font, size, text); a page draws the same few thousand words over and over.
var _glyph_cache := {}
var _ts: TextServer = TextServerManager.get_primary_interface()

func _glyphs(font: Font, text: String, fs: int) -> Array:
	var key := "%d|%d|%s" % [font.get_instance_id(), fs, text]
	if _glyph_cache.has(key):
		return _glyph_cache[key]
	var line := TextLine.new()
	line.add_string(text, font, fs)
	var out: Array = []
	var x := 0.0
	for g in _ts.shaped_text_get_glyphs(line.get_rid()):
		var d: Dictionary = g
		var rep := maxi(1, int(d.get("repeat", 1)))
		for _r in rep:
			var off: Vector2 = d.get("offset", Vector2.ZERO)
			if int(d.get("index", 0)) != 0 or not d.has("font_rid"):
				out.append({"index": int(d.get("index", 0)), "rid": d.get("font_rid", RID()),
					"pos": Vector2(x, 0.0) + off, "start": int(d.get("start", 0))})
			x += float(d.get("advance", 0.0))
	_glyph_cache[key] = out
	return out


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


## How far into the arc of the open spread the reading is: 0 wide, 1 local. A pure function
## of the reading position (which is itself a function of show time), so a render frames
## exactly what the live reading framed.
func _arc() -> float:
	if _turn_t >= 0.0:
		return 0.0                  # the leaf is turning: the whole spread is the subject
	var r := _reading()
	if r.is_empty():
		return 0.0
	var lo := -1
	var hi := -1
	for pg_i in [_spread * 2, _spread * 2 + 1]:
		if pg_i < 0 or pg_i >= _layout.pages.size():
			continue
		var ws: Array = (_layout.pages[pg_i] as Dictionary)["words"]
		if ws.is_empty():
			continue
		lo = int(ws[0]) if lo < 0 else mini(lo, int(ws[0]))
		hi = maxi(hi, int(ws[ws.size() - 1]))
	if lo < 0 or hi <= lo:
		return 0.0                  # a spread of pictures has no text to close in on
	var at := float(int(r["layout"]) - lo) + float(r["frac"])
	var p := clampf(at / float(hi - lo + 1), 0.0, 1.0)
	return smoothstep(0.0, ARC_IN, p) * (1.0 - smoothstep(1.0 - ARC_OUT, 1.0, p))


func _hash01(i: int, salt: int) -> float:
	return float(hash([_seed, i, salt]) & 0xFFFFFF) / float(0xFFFFFF)


func _tick_camera(delta: float) -> void:
	var t := maxf(Spectrum.current.time, 0.0)
	var sev := _sev()
	var k := _arc()
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
	# Wide aims at the middle of the spread; local at the line being read on its own page.
	var local := Vector3(page_x * 0.9, 0.0, clampf(_reading_z, -0.45, 0.5) + 0.05)
	var opened := _open_now()
	k *= opened                      # no closing-in on text that is still under a cover
	aim = aim.lerp(local, k)
	# A closed book is half as wide and sits on the right: frame IT, then widen as it opens.
	aim.x += 0.5 * (1.0 - opened)
	# THE BOOK'S PLACE IS NEVER CONSTANT, and never noticeably moving either. Each spread
	# sits at its own angle (hashed, so a render matches), reached through a spring several
	# times slower than the framing so the change happens under the reading rather than as a
	# move; on top of that a wander with a period of minutes. Asked for as "extremely gently
	# and almost imperceptibly... mostly unnoticeable unless you skip through the video".
	var yaw := (_hash01(_spread, 41) - 0.5) * 2.0 * YAW_SPREAD \
		+ sin(t * 0.021 + float(_seed % 97)) * YAW_WANDER * sev
	# How close "local" is scales with the Camera dial; wide is always the whole spread.
	var near := lerpf(float(WIDE["dist"]), float(LOCAL["dist"]), clampf(0.55 + 0.3 * sev, 0.0, 1.0))
	var pitch := lerpf(float(WIDE["pitch"]), float(LOCAL["pitch"]), k) + sin(t * 0.029 + 1.3) * 1.5 * sev
	var dist := lerpf(float(WIDE["dist"]), near, k) * (1.0 + 0.015 * sin(t * 0.033 + 2.1) * sev)
	dist *= lerpf(1.02, 1.0, opened)
	var roll := (_hash01(_spread, 53) - 0.5) * 2.0 * ROLL_SPREAD \
		+ sin(t * 0.047 + 0.7) * ROLL_WANDER * sev
	pitch += (_hash01(_spread, 67) - 0.5) * 2.0 * TILT_SPREAD + sin(t * 0.037 + 2.4) * TILT_WANDER * sev
	if _snap:
		_snap = false
		_c_aim = aim
		_c_dist = dist
		_c_pitch = pitch
		_c_yaw = yaw
		_c_roll = roll
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
		var ry := _spring(_c_yaw, _v_yaw, yaw, tau * ANGLE_SLOW, h)
		var rr := _spring(_c_roll, _v_roll, roll, tau * ANGLE_SLOW, h)
		_c_roll = rr.x
		_v_roll = rr.y
		_c_yaw = ry.x
		_v_yaw = ry.y
	var p := deg_to_rad(_c_pitch)
	var y := deg_to_rad(_c_yaw)
	var dir := Vector3(sin(y) * cos(p), sin(p), cos(y) * cos(p))
	_cam.position = _c_aim + dir * _c_dist
	_cam.look_at(_c_aim, Vector3.UP)
	_cam.rotate_object_local(Vector3.FORWARD, deg_to_rad(_c_roll))
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
	return "spread %d/%d turn %.2f arc %.2f reading %s" % [_spread, _layout.spreads(),
		_turn_k(), _arc(), str(r)]


## One page target's drawing surface. It holds no state of its own beyond which page and
## which highlight it shows; the medium does the drawing.
class PageCanvas:
	extends Node2D
	var book = null
	var page := -1
	var hl := {}

	func _draw() -> void:
		if book != null:
			book.draw_page(self, page, hl)
