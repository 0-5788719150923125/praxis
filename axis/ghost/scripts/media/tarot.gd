extends Medium
class_name TarotMedium

## TarotMedium - a tarot reading at a table, seen from the reader's chair.
##
## The tarot mode's own medium (see [constant Medium.OWNED]): the episode's cloth on a table,
## its room out of focus beyond the far edge, its candles, and the deck. The reader is a voice, never a pair of hands - the cards move on their own:
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
## THE ROOM, out of focus ([method _place_backdrop]): the picture drawn through a lens blur, each
## point a disc this wide - its radius, as a share of the frame's width. A lens focused on the table
## half a meter off sees a room some meters away about so soft at f/4.
const ROOM_SHADER := preload("res://shaders/tarot_room.gdshader")
const ROOM_BLUR := 0.005
## Where a shown card and its booklet page float, in the camera's frame (meters right, up, and
## away). The page sits a hair farther back so the card always reads as in front.
const PRESENT_DIST := 0.23
const PRESENT := Vector2(-0.087, 0.012)
const PAGE := Vector2(0.088, 0.012)
const PAGE_H := 0.126
const VFOV := TarotTable.VFOV

## THE PHASES of each action, in its own seconds. They sum to [TarotScript]'s rests, which is
## what the voice waits: DRAW = 3.4, LAY = 1.7, JUMP = 5.0.
const SQUARE := 0.45          # the deck squares before a card leaves it
const SLIDE_END := 1.0        # the top card slides off toward the reader
const FLIP_END := 1.9         # ...lifts and turns over
const RISE_END := 2.8         # ...and comes up to be shown
const PAGE_IN := Vector2(2.6, 3.4)
const LAY_PAGE_OUT := 0.5
const LAY_MOVE := Vector2(0.15, 1.45)
const LAY_END := 1.7
## A JUMPER FLIES OUT OF A SHUFFLE (2026-10-05: "that jump should probably happen during a shuffle -
## not when the cards are just sitting there on the table, doing nothing"): its action opens with
## one more riffle, and the card springs off the top of a half while the halves are falling. The
## riffle, the flight, the card lying there, its rise, its page.
const JUMP_RIFFLE := 2.4
const JUMP_FLY := Vector2(1.15, 2.25)
const JUMP_REST := 3.3
const JUMP_RISE := 4.3
const JUMP_PAGE := Vector2(4.2, 5.0)
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
	"wash": {"weight": 0.9, "n": [1, 1], "dur": [26.0, 32.0], "gap": [0.0, 0.0]},
}
## The pause after a run: mostly a few seconds, now and then a long linger - median ~3 s.
const IDLE_LOG := Vector2(1.1, 0.55)
const IDLE_RANGE := Vector2(1.6, 18.0)
## The chance a run is followed by a long linger instead.
const LINGER_CHANCE := 0.15
## A wash is sampled at this rate once, when it is planned, and posed by lookup.
const WASH_HZ := 20.0
## The least time a wash needs to spread, mix a little and gather (seconds): with less before the
## first card, the deck is not washed at all.
const WASH_ROOM := 9.0
## A card lying on the cloth in a wash: its center this high - clear of the cloth (at 0.6 mm) by a
## hair, so the weave never shows through it.
const WASH_FLOOR := 0.0006 + 0.00035 + 0.0002
## A card lying on another in a wash: a hair over it, besides its own thickness, so the two faces
## never fight for the same depth.
const STACK_GAP := 0.00015
## THE HANDS IN A WASH (2026-10-05, the user: "cards barely move... the movements are very small,
## very localized... It would be much more common for cards to sweep back, and forth, back, and
## forth in various directions, crossing large regions of the table, creating chaos along their
## path"): each palm had worked a 6 cm circle, and the cards under it went round that circle and
## back - 4 cm the longest straight run, 11 cm the farthest a card got. How often a palm scrubs,
## swirls or fetches ([method _wash_gesture]), how long a scrub's pass is (meters), and how often a
## palm rests a moment between gestures.
const WASH_GESTURES := {"scrub": 0.65, "swirl": 0.15, "fetch": 0.2}
const WASH_SCRUB := Vector2(0.2, 0.4)
const WASH_REST := 0.12
## A palm flat on the cards: half its width and half its length (meters), its length along the
## forearm from the reader's shoulder (to the side and toward the reader, about the deck's place).
## It comes down and lifts over WASH_TOUCH seconds, keeps its middle WASH_APART from the other
## palm's when it can, and fetches a card lying out past WASH_STRAY of the spread's reach.
const PALM := Vector2(0.045, 0.08)
const WASH_SHOULDER := Vector2(0.19, 0.46)
const WASH_TOUCH := 0.12
const WASH_APART := 0.14
const WASH_STRAY := 0.85
## Where on a card a palm takes hold (meters across and along it, from its middle), and a card's
## turning inertia over its mass (meters squared: a 70 x 120 mm sheet).
const GRIP_AT := [Vector2.ZERO, Vector2(-0.022, -0.04), Vector2(0.022, -0.04), Vector2(-0.022, 0.04),
	Vector2(0.022, 0.04)]
const CARD_I := (0.07 * 0.07 + 0.12 * 0.12) / 12.0
## HOW CARDS SLIDE in a wash (centers: each wash samples its own feel round them). How hard a palm
## can pull a card it presses whole (m/s each second - far past what the cloth holds back, so the
## card goes with it) - where another card lies over it, WASH_COVERED of that; the drag between two
## cards lying one on the other (per second: loose, and added when a palm presses the top one); how
## much of the difference in their speeds a card sliding into another gives it as they meet; how fast
## a loose card slows (m/s each second, WASH_ON_CARD of that on another card) and stops turning
## (radians/s each second); and the fastest a hand drags a card (m/s) or turns one (radians/s).
const WASH_GRIP := 40.0
const WASH_COVERED := 0.25
const WASH_DRAG := Vector2(5.0, 40.0)
const WASH_KNOCK := 0.5
const WASH_SLIDE := 3.0
const WASH_ON_CARD := 0.55
const WASH_SPIN := 40.0
const HAND_SPEED := 1.2
const WASH_TURN_MAX := 7.0
## The first card's PUSH (TarotScript.PUSH): the deck squares, then slides to its side this long.
const PUSH_SLIDE := 0.85
## A held card is turned over now and then, to look at its back ([method _look_of]): the chance a
## card is at all, the chance of each look after that, the seconds between looks, how long a turn
## takes (each its own), and how long its back is looked at - drawn evenly in its logarithm, so most
## looks are short and now and then one is long (a fixed-feeling 1.1-1.8 s was reported as "the
## exact same length, always"). Most cards are never turned: 4-5 looks in one draw was too many.
const TURN_CHANCE := 0.25
const LOOK_AGAIN := 0.25
const LOOK_GAP := Vector2(25.0, 45.0)
const TURN := Vector2(0.6, 0.95)
const TURN_HOLD := Vector2(0.6, 5.0)
## THE PIROUETTE (the user's, 2026-10-05: "the kind of trick a person might do in their own hands
## to show off"): the chance a look ends in one - over to the back and held there a few seconds,
## then on round the SAME way, five half turns more, to face on again three whole turns from where
## it started - the hold, and how long the twirl takes. Rare, and once a card at most: the plain
## half turn is the common look.
const PIROUETTE_CHANCE := 0.1
const PIROUETTE_HOLD := Vector2(1.6, 4.0)
const TWIRL := Vector2(1.5, 2.1)
## How far apart things stand on the table (meters): any two, and two of one group.
const THING_GAP := 0.014
const GROUP_GAP := 0.004
## How far a card sliding across the cloth keeps from a thing's foot (meters).
const FOOT_MARGIN := 0.004
## HOW HOT A CANDLE'S POOL MAY BURN: its light times the hottest spot of cloth round it - the
## cloth's linear luminance there over the flame's falloff (its height over the distance squared).
## The key's light on a felt of luminance 0.18 is the measure - it reads as a candle. On pale pine
## (0.64) the same light flooded half the frame through the bloom, and a cream stripe just behind a
## candle on a near-black blanket blew out though the cloth round it was dark on average.
const HEAT := 2.0
## ...looked for this far round the candle, meters, on the cloth's lightness at this grid; and the
## flame height a candle's place is judged at.
const HEAT_R := 0.25
const LUM_GRID := Vector2i(30, 18)
const HEAT_H := 0.13
## The key candle's light, the least a candle must be allowed to be the key, and every other
## candle's.
const KEY_ENERGY := 1.5
const KEY_MIN := 1.0
const FILL_ENERGY := 0.35
## The render layer of the first candle's body (the next is the next bit), so each flame can leave
## its own candle out of its shadows - and only its own: a candle stands in the key's light.
const CANDLE_LAYER := 1 << 12
## The render layer of a thing with no flame: off the cloth's layer, so the shade a thing presses
## into the cloth falls on the cloth alone.
const THING_LAYER := 1 << 1
## The intro's focus pull: it starts this long before the shuffle and takes this long after it
## (seconds), from a lens this soft (CameraAttributesPractical.dof_blur_amount).
const FOCUS_PULL := Vector2(1.4, 1.6)
const FOCUS_BLUR := 0.16
## How deep a pictured surface's relief is (Image.bump_map_to_normal_map's scale).
const RELIEF := 3.5
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
var _backdrop_mat: ShaderMaterial
var _rest_plan: Dictionary = {}     # the wash plan [member _rest_steps] are of
var _rest_steps := {}               # plan step -> every card at rest there ([method _wash_rest])
var _props: Node3D
var _probe: ReflectionProbe
var _probe_nudge := 1.0
var _lights: Array = []             # a light per lit thing: {light, base, light_base, energy, flames: [{mesh, base, flicker, glow}]}
var _flame_n := 0                     # flames lit so far on this table, each with its own flicker
var _glows: Array = []              # [{light, base, energy, flicker}] - candles in the room, out of shot
var _lamp_base := 1.6                 # the lamp's light before a candle takes the key from it
var _key_flame := -1                  # the key: the flame that leads the light, or -1 for the lamp
var _lum := PackedFloat32Array()      # the cloth's linear luminance, coarse (see _cloth_lum); empty: no picture
var _heat_cells := {}                 # grid cell -> its heat at HEAT_H, for this build
var _contact_tex: Texture2D = null
var _lit_cloth: Texture2D = null      # the cloth the table was last lit for
var _shuffle_room := INF             # how long the shuffle has, from its start to the first card
var _standing: Array = []             # what stands on the table: convex feet (x by z), for a wash to go round
var _standing_c := PackedVector2Array()   # ...their middles
var _standing_r := PackedFloat32Array()   # ...and how far they reach from them
var _things: Array = []               # what stood: [{name, group, place, node, outline, bb, rect, foot, lit, flames, meshes}]
var _collide := true                  # cards go round what stands (off only for a gate's control)
var _table_mt := -2                   # the table file's time when it was last built (-1: none)
var _stage_was := {}                  # the stage's own settings, given back on leaving it
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
var _air = null                     # the set dresser's effects, built (Effects.Air), or null
var _air_key := ""                  # the schedule its bursts were last planned on


# --- mount -----------------------------------------------------------------------------------

func mount(st: SubViewport) -> void:
	super.mount(st)
	# EDGES: the deck's stacked card edges and the shadow edges stair-stepped. 4x multisampling on
	# the stage (a light scene - cheap), and a bigger atlas whose first quadrant is one whole slot,
	# for the light whose shadows cover the most; the other lights take the next quadrant's four
	# (every candle and the lamp cast: four candles at most). Given back when the table leaves.
	_stage_was = {"msaa": st.msaa_3d, "atlas": st.positional_shadow_atlas_size,
		"quad": st.positional_shadow_atlas_quad_0}
	st.msaa_3d = Viewport.MSAA_4X
	st.positional_shadow_atlas_size = 4096
	st.positional_shadow_atlas_quad_0 = Viewport.SHADOW_ATLAS_QUADRANT_SUBDIV_1
	# the intro's bokeh: round, and fine enough to be a lens rather than a smear
	RenderingServer.camera_attributes_set_dof_blur_bokeh_shape(RenderingServer.DOF_BOKEH_CIRCLE)
	RenderingServer.camera_attributes_set_dof_blur_quality(RenderingServer.DOF_BLUR_QUALITY_HIGH, true)
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
	# NOT BY DEPTH OF FIELD: a far blur behind the table drew a band along its far edge - the sharp
	# table and the blurred room bleeding into each other where their depths meet. The room's own
	# picture is drawn through a lens blur instead ([constant ROOM_SHADER]), which is also cheaper;
	# the lens's blur is kept for the intro, when the whole table is out of focus ([method _tick_focus]).
	_attrs = CameraAttributesPractical.new()
	_attrs.dof_blur_far_enabled = false
	_cam.attributes = _attrs
	_root3.add_child(_cam)

	_lamp = SpotLight3D.new()
	_lamp.light_energy = 1.6
	_lamp.spot_range = 4.0
	_lamp.spot_angle = 38.0
	_lamp.spot_angle_attenuation = 2.2
	_lamp.shadow_enabled = true
	_lamp.shadow_blur = 1.6
	# A TABLETOP'S BIASES, as the candles' are. At Godot's own (made for rooms) the depth and normal
	# biases came to millimeters at the lamp's distance - more than a bowl's floor stands off the
	# cloth - so anything low cast nothing, a bowl only its rim's ring (a crescent), and taller things
	# a shadow standing off their feet. Softened by the blur alone: a spot's light_size threw a white
	# glare off a geode's rim facing it.
	_lamp.shadow_bias = 0.005
	_lamp.shadow_normal_bias = 0.15
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
	wood.normal_enabled = true
	wood.normal_texture = _relief("table-wood", wood.albedo_texture)
	wood.normal_scale = 0.6
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
	_backdrop_mat = ShaderMaterial.new()
	_backdrop_mat.shader = ROOM_SHADER
	_backdrop.material_override = _backdrop_mat
	_root3.add_child(_backdrop)

	_props = Node3D.new()
	_root3.add_child(_props)
	# WHAT THE THINGS REFLECT: the table and the room, caught once when the table is set (see
	# _reflect). Metal with nothing to reflect read as flat paint. The things alone take it: the
	# cards and the cloth keep the light they were tuned under.
	_probe = ReflectionProbe.new()
	_probe.update_mode = ReflectionProbe.UPDATE_ONCE
	_probe.size = Vector3(1.9, 1.0, 1.4)
	_probe.position = Vector3(0.0, 0.2, -0.05)
	_probe.box_projection = true
	_probe.max_distance = 5.0
	_probe.enable_shadows = true
	_probe.reflection_mask = THING_LAYER | (CANDLE_LAYER * 0xF)
	_root3.add_child(_probe)

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


func _exit_tree() -> void:
	if stage != null and is_instance_valid(stage) and not _stage_was.is_empty():
		stage.msaa_3d = _stage_was["msaa"]
		stage.positional_shadow_atlas_size = _stage_was["atlas"]
		stage.positional_shadow_atlas_quad_0 = _stage_was["quad"]


func on_stage_resized(_size: Vector2) -> void:
	if _title != null:
		_title.queue_redraw()


func advance(_features, delta: float, _bookend: float) -> void:
	var t := maxf(Spectrum.current.time, 0.0)
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
	# up from black as the session starts, down to it after the last word - the table owns its
	# bookend, so the Director's whole-take fade is not used; an outro mark's fade still is
	_env.adjustment_brightness = clampf(clampf(t / 1.2, 0.0, 1.0) * _end_fade(_now) * Director.live_fade, 0.0, 1.0)
	_pose(_now)
	_tick_focus(_now)
	_tick_camera(_now)
	_tick_props(_now)
	_tick_air(_now)
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
	var t0 := Time.get_ticks_msec()
	_build_episode()
	print("ghost: tarot table - %s #%d, %d cards, %d actions, %d things (built in %d ms)" % [String(pay.get("show", "?")), _seed,
		(pay.get("cards", []) as Array).size(), (_parse.get("actions", []) as Array).size(), _things.size(),
		Time.get_ticks_msec() - t0])


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
	# THE CHANNEL'S NAME ONLY: the episode's title is the video's, on the platform, not on the table
	_title.episode = ""
	# THE CHANNEL'S NAME IS THE CHANNEL'S, set the same way every episode - a brand, not a
	# deck's lettering (an uncial deck turned "Truthful" into "Truchful")
	_title.face = TarotTable.font("roman")
	_title.italic = TarotTable.font(TarotTable.BOOK_ITALIC)
	var lay := TarotTable.sample_layout(rng)
	# THE CAMERA: the reader's eye, a little different every episode
	_pitch = float(lay["pitch"])
	_cam_base = lay["camera"]
	_cam.transform = _cam_base
	_cam.fov = float(lay["fov"])
	# THE LIGHT: the look's own, from above and to one side
	var lc := TarotTable.color(String((_look.get("light", {}) as Dictionary).get("color", "#ffb36b")))
	_lamp.light_color = Color(1, 1, 1).lerp(lc, 0.55)
	_lamp.position = lay["lamp"]
	_lamp.look_at(Vector3(0.0, 0.0, -0.1), Vector3.UP)
	_lamp.light_energy = float(lay["lamp_energy"])
	_lamp_base = _lamp.light_energy
	var pal: Array = _look.get("palette", TarotTable.FALLBACK_PALETTE)
	var dark := TarotTable.color(String(pal[0]))
	_env.background_color = dark.darkened(0.6)
	_env.ambient_light_color = Color(0.5, 0.5, 0.5).lerp(dark.lightened(0.4), 0.35)
	_fill.light_color = Color(0.75, 0.8, 1.0) if String((_look.get("light", {}) as Dictionary).get("warmth", "warm")) == "warm" else lc
	# THE DECK, squared where the reader keeps it
	_deck_base = lay["deck"]
	# SHUFFLED IN THE MIDDLE, in front of the reader, and pushed to its side before the first card
	_mid = lay["mid"]
	_cur_base = _mid
	_slot_jit = []
	for i in DECK_N + 1:
		_slot_jit.append(Vector3(rng.randf_range(-0.0007, 0.0007), rng.randf_range(-0.0007, 0.0007),
			deg_to_rad(rng.randf_range(-1.3, 1.3))))
	# THE CARDS
	for c in _cards:
		(c as Node).queue_free()
	for v in _faces:
		(v as Node).queue_free()
	_cards = []
	_faces = []
	_face_canvas = []
	_face_mats = []
	# THIS EPISODE'S PICTURES ARE KEPT across its sessions (each Play, each scrub): decoding a
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
	# the cloth's lightness before anything stands on it: a candle looks for dark cloth
	_lum = _cloth_lum(String(_pay.get("dir", "")).path_join("surface.png"))
	_build_table()
	_plan_moves()
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
		_cloth_mat.normal_enabled = true
		_cloth_mat.normal_texture = _relief(dir.path_join("surface.png"), cloth)
		_cloth_mat.normal_scale = 1.0
		# its pixels square on the cloth: the part of the picture that fits, never the picture stretched
		var crop := TarotTable.cloth_crop(Vector2(cloth.get_width(), cloth.get_height()), CLOTH)
		_cloth_mat.uv1_scale = Vector3(crop.size.x, crop.size.y, 1.0)
		_cloth_mat.uv1_offset = Vector3(crop.position.x, crop.position.y, 0.0)
		_lum = _cloth_lum(dir.path_join("surface.png"))
	elif _cloth_mat.albedo_texture == null:
		_cloth_mat.albedo_color = _cloth_fallback()
		_cloth_mat.albedo_texture = BookMedium._grime(hash([_seed, "cloth"]) & 0xFFFF, 0.05, 4, 0.12)
		_cloth_mat.uv1_scale = Vector3.ONE
		_cloth_mat.uv1_offset = Vector3.ZERO
		_lum = PackedFloat32Array()
	# a cloth that landed after the candles stood (live, it is painted while a reading can already
	# be playing) lights the table again
	if _cloth_mat.albedo_texture != _lit_cloth:
		_light_the_table()
		_reflect()
	# THE TABLE, set while a reading can already be playing (live, the set dresser works beside the
	# painter): built again when it lands, and the shuffle's chain with it - a wash goes round
	# whatever stands
	var tpath := dir.path_join("table.json")
	if (FileAccess.get_modified_time(tpath) if FileAccess.file_exists(tpath) else -1) != _table_mt:
		_build_table()
		_plan_moves()
	var room := _picture(dir.path_join("backdrop.png"), force)
	if room != null:
		if _backdrop_mat.get_shader_parameter("picture") != room:
			_backdrop_mat.set_shader_parameter("picture", room)
			_backdrop_mat.set_shader_parameter("has_picture", 1.0)
			# a picture landing live may be one asked for level: placed again for its view
			_place_backdrop()
			_reflect()
	elif _backdrop_mat.get_shader_parameter("picture") == null:
		var pal2: Array = _look.get("palette", TarotTable.FALLBACK_PALETTE)
		_backdrop_mat.set_shader_parameter("tint", TarotTable.color(String(pal2[0])).darkened(0.25))


## THE RELIEF OF A PICTURED SURFACE, from the picture itself: a weave or a grain photographed from
## above is dark in its hollows and light on its ridges, so its luminance stands in for its height,
## and Godot turns that into a normal map natively. Laid under the cloth, the light low across the
## table picks the texture out as it would on a real one. Made once per picture.
func _relief(path: String, tex: Texture2D) -> Texture2D:
	var key := path + "|relief"
	if _textures.has(key) and int(_mtimes.get(key, -1)) == int(_mtimes.get(path, -2)):
		return _textures[key]
	var img := tex.get_image()
	if img == null:
		return null
	img = img.duplicate() as Image
	if img.is_compressed():
		img.decompress()
	img.clear_mipmaps()
	img.convert(Image.FORMAT_L8)
	# at half size: the photograph's finest grain is noise, not relief
	img.resize(maxi(8, img.get_width() / 2), maxi(8, img.get_height() / 2), Image.INTERPOLATE_BILINEAR)
	img.convert(Image.FORMAT_RGBA8)
	img.bump_map_to_normal_map(RELIEF)
	img.generate_mipmaps()
	var nt := ImageTexture.create_from_image(img)
	_textures[key] = nt
	_mtimes[key] = _mtimes.get(path, -2)
	return nt


## THE CLOTH'S LIGHTNESS where candles stand: its picture's linear luminance, averaged down to
## [constant LUM_GRID] (a few centimeters a cell), row by row, for [method _heat]. Made once per
## picture, from the file rather than read back from the GPU (which a probe's dummy renderer cannot
## do). Empty when there is no picture.
func _cloth_lum(path: String) -> PackedFloat32Array:
	var key := path + "|lum"
	if _textures.has(key) and int(_mtimes.get(key, -1)) == int(_mtimes.get(path, -2)):
		return _textures[key]
	var img := Image.load_from_file(path) if FileAccess.file_exists(path) else null
	if img == null or img.is_empty():
		return PackedFloat32Array()
	if img.is_compressed():
		img.decompress()
	img.clear_mipmaps()
	# the part of the picture the cloth shows, as its material is laid out
	var crop := TarotTable.cloth_crop(Vector2(img.get_width(), img.get_height()), CLOTH)
	var size := Vector2(img.get_width(), img.get_height())
	img = img.get_region(Rect2i(Vector2i((crop.position * size).round()), Vector2i((crop.size * size).round())))
	img.convert(Image.FORMAT_RGB8)
	img.srgb_to_linear()
	img.resize(LUM_GRID.x, LUM_GRID.y, Image.INTERPOLATE_LANCZOS)
	var out := PackedFloat32Array()
	for y in LUM_GRID.y:
		for x in LUM_GRID.x:
			var c := img.get_pixel(x, y)
			out.append(c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722)
	_textures[key] = out
	_mtimes[key] = _mtimes.get(path, -2)
	return out


## The cloth an episode gets while its own is not painted: a dark shade of its palette.
func _cloth_fallback() -> Color:
	var pal: Array = _look.get("palette", TarotTable.FALLBACK_PALETTE)
	return TarotTable.color(String(pal[min(4, pal.size() - 1)])).darkened(0.35)


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


## THE ROOM. A picture asked for as a LEVEL photograph from a seated eye (`backdrop.json` beside it
## names its lens, [constant TarotTable.BACKDROP_LENS]) is PROJECTED FROM THE CAMERA'S EYE: an
## upright plane far behind the table, its middle at the eye's height straight ahead, as wide as its
## lens saw - so every point of the picture lies in the direction it was seen from, and the camera,
## tilted down at the table, sees the room as it would really look past the table's edge (its
## uprights running together below, the floor where a floor would be). Laid square to the tilted
## camera with its horizon on the table's edge, a room was seen from the height of the cloth: a wall
## or a window just past the table looked wrong, a far landscape got away with it (the user,
## 2026-10-05: "the backgrounds are 2D... the angles feel wrong"). A picture made before (no view
## beside it) holds nothing below its horizon for the camera to see, so it keeps that placement.
func _place_backdrop() -> void:
	var view := _backdrop_view()
	if not view.is_empty():
		var lens := clampf(float(view.get("lens_mm", TarotTable.BACKDROP_LENS)), 8.0, 200.0)
		var reach := 3.2
		var ahead := Vector3(-_cam_base.basis.z.x, 0.0, -_cam_base.basis.z.z).normalized()
		_backdrop.transform = Transform3D(Basis.looking_at(ahead, Vector3.UP), _cam_base.origin + ahead * reach)
		# a 36 x 24 frame behind a lens of this length, seen at this reach
		(_backdrop.mesh as QuadMesh).size = Vector2(36.0, 24.0) * reach / lens
	else:
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
	_backdrop_mat.set_shader_parameter("radius", room_blur(_cam_base, _cam.fov, _backdrop.transform,
		(_backdrop.mesh as QuadMesh).size))


## HOW SOFT THE ROOM IS, as the disc's radius in its picture's UV (per axis): [constant ROOM_BLUR]
## of the frame's width, measured where the camera sees the picture - just past the table's far
## edge, under the top of the frame - since the room may be placed any way and seen at a slant.
static func room_blur(cam: Transform3D, fov: float, room: Transform3D, size: Vector2) -> Vector2:
	# the line of sight just under the top of the frame, to the picture's plane
	var sight := (cam.basis * Vector3(0.0, tan(deg_to_rad(fov * 0.5)) * 0.8, -1.0)).normalized()
	var normal := room.basis.z.normalized()
	var facing := sight.dot(normal)
	if absf(facing) < 1e-4:
		return Vector2.ZERO
	var hit := cam.origin + sight * ((room.origin - cam.origin).dot(normal) / facing)
	# how much of the frame's width a meter of the picture there takes
	var across := room.basis.x.normalized() * 0.5
	var a: Variant = TarotTable.project(cam, fov, hit - across)
	var b: Variant = TarotTable.project(cam, fov, hit + across)
	if a == null or b == null:
		return Vector2.ZERO
	var meters := ROOM_BLUR / maxf(absf((b as Vector2).x - (a as Vector2).x), 1e-4)
	return Vector2(meters / maxf(size.x, 1e-4), meters / maxf(size.y, 1e-4))


## The view the room's picture was asked for (`backdrop.json` beside it): `{view, lens_mm}`, or empty
## for a picture made before there was one.
func _backdrop_view() -> Dictionary:
	var path := String(_pay.get("dir", "")).path_join("backdrop.json")
	if not FileAccess.file_exists(path):
		return {}
	var j := JSON.new()
	if j.parse(FileAccess.get_file_as_string(path)) != OK or not (j.data is Dictionary):
		return {}
	return j.data if String((j.data as Dictionary).get("view", "")) == "level" else {}


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


# --- the things on the table --------------------------------------------------------------------------

## THE TABLE'S THINGS: what the set dresser described for this episode (`table.json` in its folder,
## made safe by [method TarotTable.sanitize_table]), or the look's candles alone until it has -
## each built ([Props]), stood where [method _place_things] finds it room, its wicks lit.
func _build_table() -> void:
	for c in _props.get_children():
		c.queue_free()
	_lights = []
	_flame_n = 0
	_standing = []
	_standing_c = PackedVector2Array()
	_standing_r = PackedFloat32Array()
	_things = []
	_heat_cells = {}
	var spec := _table_spec()
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([_seed, "tarot-things"])
	var things: Array = spec["things"]
	var built: Array = []
	for i in things.size():
		built.append(Props.build(things[i], spec["materials"], hash([_seed, i, "thing"])))
	_place_things(things, built, rng)
	_build_glows()
	_light_the_table()
	_reflect()
	_build_air(spec)


## Catch the table again for the things to reflect: a probe that updates once does so again when
## it moves, so it is moved by a hair.
func _reflect() -> void:
	if _probe == null:
		return
	_probe_nudge = -_probe_nudge
	_probe.position = Vector3(0.0, 0.2 + 0.0001 * _probe_nudge, -0.05)


## The set dresser's table, made safe - or, before there is one, the look's candles. Read from the
## FILE, as the pictures are, so a render (a second process) stands what the live table stood.
func _table_spec() -> Dictionary:
	var path := String(_pay.get("dir", "")).path_join("table.json")
	_table_mt = FileAccess.get_modified_time(path) if FileAccess.file_exists(path) else -1
	if _table_mt >= 0:
		var j := JSON.new()
		if j.parse(FileAccess.get_file_as_string(path)) == OK and j.data is Dictionary:
			return TarotTable.sanitize_table(j.data as Dictionary, _look)
	return TarotTable.default_table(_look, _seed)


## WHERE EACH THING STANDS. A thing goes in the zone the set dresser named, and things that share
## a group stand together: the group's tallest first, as near its zone's middle as there is room,
## the rest round it - the shorter ones toward the reader - so a group reads as arranged, not
## lined up. The biggest groups choose first. A thing with no room is tried smaller, then left
## off the table.
func _place_things(things: Array, built: Array, rng: RandomNumberGenerator) -> void:
	var keep_out := _keep_out()
	var groups := {}
	for i in things.size():
		var g := String((things[i] as Dictionary).get("group", ""))
		var key := g if not g.is_empty() else "#%d" % i
		if not groups.has(key):
			groups[key] = []
		(groups[key] as Array).append(i)
	var area := func(members: Array) -> float:
		var s := 0.0
		for i in members:
			var box: AABB = (built[i] as Dictionary)["size"]
			s += box.size.x * box.size.z
		return s
	var order: Array = groups.keys()
	order.sort_custom(func(a: Variant, b: Variant) -> bool: return float(area.call(groups[a])) > float(area.call(groups[b])))
	for key in order:
		var members: Array = groups[key]
		members.sort_custom(func(a: int, b: int) -> bool:
			return ((built[a] as Dictionary)["size"] as AABB).size.y > ((built[b] as Dictionary)["size"] as AABB).size.y)
		var anchor := {}
		for i in members:
			var stood := _stand(things[i], built[i], String(key), anchor, keep_out, rng)
			if stood.is_empty():
				print("ghost: tarot table - no room for %s" % String((things[i] as Dictionary).get("name", "a thing")))
				((built[i] as Dictionary)["node"] as Node).free()
			elif anchor.is_empty():
				anchor = stood


## Where nothing may stand: everywhere the cards go - the spread, the deck, the middle where the
## deck is shuffled, where a jumper lands - as rectangles on the table (x by z).
func _keep_out() -> Array:
	var out: Array = []
	for sl in _slots:
		out.append(TarotTable.footprint((sl as Dictionary)["pos"], float((sl as Dictionary)["yaw"]), CARD).grow(0.03))
	out.append(Rect2(_deck_base.x - 0.08, _deck_base.z - 0.1, 0.16, 0.2))
	out.append(Rect2(_mid.x - 0.22, _mid.z - 0.14, 0.44, 0.28))
	out.append(Rect2(_jump_land.x - 0.07, _jump_land.z - 0.09, 0.14, 0.18))
	return out


## ONE THING STOOD: the best free spot for [param t] (built as [param b]) on a grid over the table,
## at full size or, failing that, a little smaller. Every spot is ON THE TABLE, off everywhere the
## cards go, clear of what already stands (by [constant THING_GAP], or [constant GROUP_GAP] within
## its own group), WHOLLY IN THE SHOT (its whole box, projected), and - between groups - not in
## front of another in the picture. A lit thing stands only behind the middle (in front of the
## cards its flame blew out the card held up to the lens) and looks for dark cloth. The spot
## chosen ({at, k, group, height}), or empty when there is none.
func _stand(t: Dictionary, b: Dictionary, group: String, anchor: Dictionary, keep_out: Array,
		rng: RandomNumberGenerator) -> Dictionary:
	var box: AABB = b["size"]
	var lit := not (b["wicks"] as Array).is_empty()
	var yaw := deg_to_rad(float(t.get("turn", 0.0)) + rng.randf_range(-8.0, 8.0))
	var zone := String(t.get("place", "back"))
	var aim := TarotTable.zone_aim(zone, _deck_base)
	var front := (_deck_base.z + 0.03) if zone in ["by the deck", "left", "right"] else _mid.z + 0.03
	# a LOW thing may lie nearer the reader, where it hides no card and nothing behind it
	if box.size.y < 0.04:
		front = 0.2
	if lit:
		front = minf(front, _mid.z)
	var outline: PackedVector2Array = b["outline"]
	for k in [1.0, 0.88, 0.76]:
		var basis := Basis(Vector3.UP, yaw).scaled(Vector3(k, k, k))
		var shape := PackedVector2Array()
		for q in outline:
			var w := basis * Vector3(q.x, 0.0, q.y)
			shape.append(Vector2(w.x, w.z))
		var lo := Vector2(INF, INF)
		var hi := Vector2(-INF, -INF)
		for q in shape:
			lo = lo.min(q)
			hi = hi.max(q)
		var corners := PackedVector3Array()
		for cx in [box.position.x, box.end.x]:
			for cy in [box.position.y, box.end.y]:
				for cz in [box.position.z, box.end.z]:
					corners.append(basis * Vector3(cx, cy, cz))
		var best := {}
		var best_score := -INF
		var cam_inv := _cam_base.affine_inverse()
		var lens := Vector2(tan(deg_to_rad(_cam.fov * 0.5)) * (16.0 / 9.0), tan(deg_to_rad(_cam.fov * 0.5)))
		var z := -0.4
		while z <= front:
			var x := -0.62
			while x <= 0.62:
				var at := Vector2(x + rng.randf_range(-0.004, 0.004), z + rng.randf_range(-0.004, 0.004))
				x += 0.02
				var bb := Rect2(lo + at, hi - lo)
				# ON THE CLOTH: a thing past its edge, on the bare wood by the table's rim, read as about
				# to fall off
				if bb.position.x < -CLOTH.x * 0.49 or bb.end.x > CLOTH.x * 0.49 or bb.position.y < -0.02 - CLOTH.y * 0.48 \
						or bb.end.y > -0.02 + CLOTH.y * 0.48:
					continue
				# the outline itself only where the boxes meet: most spots are judged by box alone
				var placed := PackedVector2Array()
				var clear := true
				for r in keep_out:
					if (r as Rect2).intersects(bb):
						if placed.is_empty():
							placed = _translated(shape, at)
						if _convex_overlap(placed, _rect_poly(r as Rect2), 0.0):
							clear = false
							break
				if not clear:
					continue
				for th in _things:
					var gap := GROUP_GAP if String((th as Dictionary)["group"]) == group else THING_GAP
					if ((th as Dictionary)["bb"] as Rect2).grow(gap).intersects(bb):
						if placed.is_empty():
							placed = _translated(shape, at)
						if _convex_overlap(placed, (th as Dictionary)["outline"], gap):
							clear = false
							break
				if not clear:
					continue
				var rect := _screen_rect_fast(corners, Vector3(at.x, 0.0, at.y), cam_inv, lens)
				# wholly in the shot: a thing cut off by the frame's edge reads as one standing in the
				# room, not on the table
				if rect.position.x < 0.02 or rect.end.x > 0.98 or rect.position.y < 0.02 or rect.end.y > 0.97:
					continue
				# not in front of another in the picture: apart between groups, and within one only a
				# little in front - a group seen one thing through another read as a stack
				var hidden := 0.0
				for th in _things:
					var other: Rect2 = (th as Dictionary)["rect"]
					if String((th as Dictionary)["group"]) != group:
						if other.grow(0.008).intersects(rect):
							clear = false
							break
					elif other.intersects(rect):
						hidden = maxf(hidden, other.intersection(rect).get_area() / maxf(minf(other.get_area(), rect.get_area()), 1e-6))
				if not clear or hidden > 0.3:
					continue
				var score := -hidden * 2.0
				if anchor.is_empty():
					score -= at.distance_to(aim) * 4.0
				else:
					var ap: Vector2 = anchor["at"]
					score -= at.distance_to(ap) * 6.0
					# a shorter thing in a group stands toward the reader, a taller one behind
					if box.size.y * k < float(anchor["height"]):
						score += clampf((at.y - ap.y) * 4.0, -0.2, 0.2)
				if lit:
					# by dark cloth, where it can burn as the key: by pale, its light is held down
					score -= 0.8 * maxf(0.0, _heat_cell(Vector3(at.x, 0.0, at.y)) * KEY_ENERGY / HEAT - 1.0)
				score += rng.randf() * 0.05
				if score > best_score:
					best_score = score
					best = {"at": at, "rect": rect, "bb": bb, "outline": placed if not placed.is_empty() else _translated(shape, at)}
			z += 0.02
		if not best.is_empty():
			_put(t, b, best, basis, group)
			return {"at": best["at"], "height": box.size.y * k, "k": k}
	return {}


## Thing [param t] stood at the chosen spot: in the scene, its wicks lit, a soft shade where it meets
## the cloth, its foot kept for the cards to go round (see [method _card_clear]).
func _put(t: Dictionary, b: Dictionary, spot: Dictionary, basis: Basis, group: String) -> void:
	var at: Vector2 = spot["at"]
	var node: Node3D = b["node"]
	node.transform = Transform3D(basis, Vector3(at.x, 0.0, at.y))
	_props.add_child(node)
	var wicks: Array = b["wicks"]
	# A CANDLE CASTS IN EVERY FLAME'S LIGHT BUT ITS OWN: its own, just above it, printed a hard dark
	# disc round its base. So it has a render layer of its own, which only its flames leave out. A
	# thing with no flame is on its own layer too, so the cloth's shades fall on the cloth alone.
	var own := (CANDLE_LAYER << _lights.size()) if not wicks.is_empty() else THING_LAYER
	for m in b["meshes"]:
		(m as MeshInstance3D).layers = own
	var glows: Array = b.get("glows", [])
	var first := _lights.size()
	if not wicks.is_empty():
		var flames: Array = []
		for i in wicks.size():
			flames.append(_flame(node.transform * (wicks[i] as Vector3), own, glows[i] if i < glows.size() else null))
		_light_flames(flames, own)
	var foot := _translated(_turned(b["foot"], basis), at)
	if foot.size() >= 3:
		_contact(foot)
		var grown := _grown(foot, FOOT_MARGIN)
		_standing.append(grown)
		var c := Vector2.ZERO
		for q in grown:
			c += q
		c /= float(grown.size())
		var r := 0.0
		for q in grown:
			r = maxf(r, q.distance_to(c))
		_standing_c.append(c)
		_standing_r.append(r)
	_things.append({"name": String(t.get("name", "")), "group": group, "place": String(t.get("place", "back")),
		"node": node, "outline": spot["outline"], "bb": spot["bb"], "rect": spot["rect"], "foot": foot,
		"box": node.transform * (b["size"] as AABB), "lit": wicks.size(), "lights": range(first, _lights.size()),
		"meshes": b["meshes"]})


## A FLAME at [param at], the top of a wick: a wick under it (on its candle's layer, [param own])
## and the flame, flickering in its own time. Its light is its thing's ([method _light_flames]); its
## own wax glows with it, through [param glow] (its material's `flame`). The flame, as
## `{mesh, base, flicker, glow}`.
func _flame(at: Vector3, own: int, glow: Variant = null) -> Dictionary:
	var wick := MeshInstance3D.new()
	var wm := CylinderMesh.new()
	wm.top_radius = 0.0006
	wm.bottom_radius = 0.0008
	wm.height = 0.009
	wm.radial_segments = 6
	wm.rings = 1
	wick.mesh = wm
	var wmat := StandardMaterial3D.new()
	wmat.albedo_color = Color(0.07, 0.05, 0.04)
	wmat.roughness = 0.9
	wick.material_override = wmat
	wick.position = at + Vector3(0.0, 0.0035, 0.0)
	wick.layers = own
	_props.add_child(wick)
	var flame := MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = Vector2(0.012, 0.03)
	flame.mesh = q
	var fm := ShaderMaterial.new()
	var sh := Shader.new()
	sh.code = FLAME_SHADER
	fm.shader = sh
	flame.material_override = fm
	flame.position = at + Vector3(0, 0.016, 0)
	_props.add_child(flame)
	_flame_n += 1
	return {"mesh": flame, "base": flame.position, "flicker": _flicker_of(_flame_n - 1), "glow": glow}


## A LIT THING'S ONE LIGHT, for every flame on it: a candelabra's tapers or a pillar's three wicks
## light the room as one light from among them, as near together as they are - so a candle costs
## one shadow however many flames it has. It lights everything but its own thing (render layer
## [param own]) and throws shadows from all of it: a flame lighting its own holder blew it out (an
## oil lamp's flame sits a few centimeters above its body). Its brightness follows its flames' -
## each still flickers in its own time.
func _light_flames(flames: Array, own: int) -> void:
	var at := Vector3.ZERO
	for f in flames:
		at += (f as Dictionary)["base"]
	at /= float(flames.size())
	var light := OmniLight3D.new()
	light.light_color = Color(1.0, 0.7, 0.4)
	light.omni_range = 1.4
	light.light_energy = 0.35
	light.shadow_caster_mask = 0xFFFFFFFF & ~own
	light.light_cull_mask = 0xFFFFFFFF & ~own
	# a flame is a couple of centimeters across: the shadows it throws are soft at their ends. The
	# biases are for a TABLETOP: at their defaults a deck's shadow began centimeters in front of it
	light.light_size = 0.015
	light.shadow_bias = 0.004
	light.shadow_normal_bias = 0.08
	light.omni_shadow_mode = OmniLight3D.SHADOW_CUBE
	light.shadow_enabled = true
	light.position = at + Vector3(0, 0.016, 0)
	_props.add_child(light)
	_lights.append({"light": light, "base": at, "light_base": light.position, "energy": 0.28, "flames": flames})


## A soft shade on the cloth under [param foot] (a thing's outline where it meets it): what
## anything standing presses into the cloth.
func _contact(foot: PackedVector2Array) -> void:
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for q in foot:
		lo = lo.min(q)
		hi = hi.max(q)
	var size := (hi - lo).max(Vector2(0.01, 0.01))
	var contact := Decal.new()
	contact.texture_albedo = _contact_texture()
	contact.modulate = Color(0, 0, 0, 1)
	contact.albedo_mix = 0.6
	contact.size = Vector3(size.x * 1.7, 0.03, size.y * 1.7)
	contact.position = Vector3((lo.x + hi.x) * 0.5, 0.0, (lo.y + hi.y) * 0.5)
	contact.cull_mask = 1
	_props.add_child(contact)


## The rectangle in the picture that [param corners] (a thing's box, turned and sized) cover
## standing at [param at], for a camera whose inverse is [param cam_inv] and whose lens spreads
## [param lens] (tangents across and up); one wider than the frame when any lies behind the lens.
static func _screen_rect_fast(corners: PackedVector3Array, at: Vector3, cam_inv: Transform3D, lens: Vector2) -> Rect2:
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for c in corners:
		var l: Vector3 = cam_inv * (c + at)
		if l.z > -0.001:
			return Rect2(-1.0, -1.0, 3.0, 3.0)
		var sp := Vector2(0.5 + l.x / (-l.z * lens.x) * 0.5, 0.5 - l.y / (-l.z * lens.y) * 0.5)
		lo = lo.min(sp)
		hi = hi.max(sp)
	return Rect2(lo, hi - lo)


static func _translated(poly: PackedVector2Array, by: Vector2) -> PackedVector2Array:
	var out := PackedVector2Array()
	for q in poly:
		out.append(q + by)
	return out


static func _turned(poly: PackedVector2Array, basis: Basis) -> PackedVector2Array:
	var out := PackedVector2Array()
	for q in poly:
		var w := basis * Vector3(q.x, 0.0, q.y)
		out.append(Vector2(w.x, w.z))
	return out


## [param poly] (convex) pushed out by [param by] all round.
static func _grown(poly: PackedVector2Array, by: float) -> PackedVector2Array:
	var c := Vector2.ZERO
	for q in poly:
		c += q
	c /= float(maxi(poly.size(), 1))
	var out := PackedVector2Array()
	for q in poly:
		out.append(q + (q - c).normalized() * by)
	return out


static func _rect_poly(r: Rect2) -> PackedVector2Array:
	return PackedVector2Array([r.position, Vector2(r.end.x, r.position.y), r.end, Vector2(r.position.x, r.end.y)])


## Whether convex [param a] and [param b] come within [param gap] of each other: no separating
## axis among their edges' normals with that much room on it.
static func _convex_overlap(a: PackedVector2Array, b: PackedVector2Array, gap: float) -> bool:
	if a.size() < 2 or b.size() < 2:
		return false
	for poly in [a, b]:
		var p: PackedVector2Array = poly
		for i in p.size():
			var e := p[(i + 1) % p.size()] - p[i]
			if e.length_squared() < 1e-12:
				continue
			var ax := Vector2(-e.y, e.x).normalized()
			var ra := _span(a, ax)
			var rb := _span(b, ax)
			if ra.x > rb.y + gap or rb.x > ra.y + gap:
				return false
	return true


## [param poly]'s extent along [param ax]: (least, most).
static func _span(poly: PackedVector2Array, ax: Vector2) -> Vector2:
	var lo := INF
	var hi := -INF
	for q in poly:
		var d := q.dot(ax)
		lo = minf(lo, d)
		hi = maxf(hi, d)
	return Vector2(lo, hi)


## LIGHT FROM OUTSIDE THE SHOT: two or three more candles in the room - behind the camera and off
## to the sides - never seen, only felt: a warm, shifting fill on the cloth, and on a card held up
## to the lens. Without shadows - faint fills, whose shadows would be fainter still, at six shadow
## passes each - each flickering in its own time.
## Their own dice, so the rest of the table lands where it did.
func _build_glows() -> void:
	_glows = []
	var r := RandomNumberGenerator.new()
	r.seed = hash([_seed, "tarot-glows"])
	var n := r.randi_range(2, 3)
	var tries := 0
	while _glows.size() < n and tries < 60:
		tries += 1
		var at := Vector3(r.randf_range(-0.7, 0.7), r.randf_range(0.12, 0.5), _cam_base.origin.z + r.randf_range(0.2, 0.7))
		if not _glows.is_empty() and r.randf() < 0.5:
			at = Vector3(r.randf_range(0.85, 1.4) * (1.0 if r.randf() < 0.5 else -1.0), r.randf_range(0.08, 0.45),
				r.randf_range(-0.45, 0.35))
		if _in_shot(at):
			continue
		var light := OmniLight3D.new()
		light.light_color = Color(1.0, r.randf_range(0.6, 0.74), r.randf_range(0.3, 0.44))
		light.omni_range = 3.0
		light.shadow_enabled = false
		light.position = at
		_props.add_child(light)
		_glows.append({"light": light, "base": at, "energy": r.randf_range(0.12, 0.3),
			"flicker": _flicker_of("glow%d" % _glows.size())})


## Whether [param at] is inside the camera's picture (with a margin) - a light there would light
## the table from a place where nothing is burning.
func _in_shot(at: Vector3) -> bool:
	var l: Vector3 = _cam_base.affine_inverse() * at
	if l.z > -0.01:
		return false
	var k := tan(deg_to_rad(_cam.fov * 0.5))
	var sp := Vector2(0.5 + l.x / (-l.z * k * (16.0 / 9.0)) * 0.5, 0.5 - l.y / (-l.z * k) * 0.5)
	return sp.x > -0.1 and sp.x < 1.1 and sp.y > -0.1 and sp.y < 1.1


## EVERY LIGHT THROWS ITS OWN SHADOW (the user, 2026-10-05: "most scenes have multiple light
## sources, and thus should probably cast multiple shadows"): every candle and the lamp, so a thing
## has a shadow for each light near it, each turned away from its own. One light had thrown them all
## (every light casting its own then gave the deck no shadow worth the name and each half of a split
## deck two - "it's all very strange"); now the KEY leads instead of casting alone: the candle
## nearest the middle of the table, bright enough to reach the cards, so the deck's long shadow
## toward the reader reads and the others are fainter. The other candles are fills, and the lamp is
## dimmed to one; with no candle at all, the lamp is the key ([member _key_flame] -1). NO CLOTH IS LIT HOTTER THAN IT CAN TAKE ([constant HEAT]): a candle
## by pale cloth is dimmer - and so are candles standing TOGETHER, whose pools add up on the cloth
## between them (two tapers side by side, each at its own limit, burned a pale cloth white through
## the bloom). A candle the cloth cannot let burn at [constant KEY_MIN] is never the key - with
## none that can, the lamp is.
func _light_the_table() -> void:
	var fields: Array = []
	var key := -1
	var best := INF
	for i in _lights.size():
		var base: Vector3 = (_lights[i] as Dictionary)["base"]
		var lb: Vector3 = (_lights[i] as Dictionary)["light_base"]
		var field := _heat_field(Vector3(base.x, 0.0, base.z), lb.y)
		fields.append(field)
		var d := (base * Vector3(1, 0, 1)).length()
		if HEAT / maxf(_field_max(field), 0.05) >= KEY_MIN and d < best:
			best = d
			key = i
	# A LIGHT IS AS BRIGHT AS ITS FLAMES: a pillar's three wicks give three flames' light, a
	# candelabra's five tapers five - capped like any other by the cloth they light
	var flames := PackedFloat32Array()
	for i in _lights.size():
		flames.append(float(((_lights[i] as Dictionary)["flames"] as Array).size()))
	var energy := PackedFloat32Array()
	for i in _lights.size():
		energy.append(minf((KEY_ENERGY if i == key else FILL_ENERGY) * flames[i], HEAT / maxf(_field_max(fields[i]), 0.05)))
	# the cloth's hottest spot under all of them at once, brought down to what it can take by
	# dimming every flame that reaches it - a few times, as the hottest spot moves
	for pass_ in 6:
		var total := {}
		for i in _lights.size():
			for c in fields[i]:
				total[c] = float(total.get(c, 0.0)) + energy[i] * float((fields[i] as Dictionary)[c])
		var hot := -1
		var most := HEAT
		for c in total:
			if float(total[c]) > most:
				most = float(total[c])
				hot = int(c)
		if hot < 0:
			break
		for i in _lights.size():
			if (fields[i] as Dictionary).has(hot):
				energy[i] *= HEAT / most * 0.999
	if key >= 0 and energy[key] < KEY_MIN:
		key = -1
	_key_flame = key
	for i in _lights.size():
		var f: Dictionary = _lights[i]
		f["energy"] = energy[i] if i == key else minf(energy[i], FILL_ENERGY * flames[i])
	# the candle has to carry much of the light at the cards, or its shadow is lost under the lamp's
	_lamp.light_energy = _lamp_base * (1.0 if key < 0 else 0.4)
	_lit_cloth = _cloth_mat.albedo_texture


## The cloth's HOTTEST SPOT under a flame [param hf] above [param at] (see [method _heat_field]).
func _heat(at: Vector3, hf: float) -> float:
	return _field_max(_heat_field(at, hf))


static func _field_max(field: Dictionary) -> float:
	var m := 0.0
	for c in field:
		m = maxf(m, float(field[c]))
	return m


## HOW HOT A FLAME [param hf] above [param at] LIGHTS THE CLOTH, cell by cell within [constant
## HEAT_R] ([constant LUM_GRID], keyed by cell): each cell's lightness over the flame's falloff -
## height over distance squared, which is the slant of the light times an omni light's 1/d at
## Godot's default attenuation. A cloth with no picture yet is its color.
func _heat_field(at: Vector3, hf: float) -> Dictionary:
	var out := {}
	var cell := Vector2(CLOTH.x / LUM_GRID.x, CLOTH.y / LUM_GRID.y)
	var g := Vector2((at.x - _cloth.position.x) / cell.x + LUM_GRID.x * 0.5, (at.z - _cloth.position.z) / cell.y + LUM_GRID.y * 0.5)
	var reach := Vector2i(ceili(HEAT_R / cell.x), ceili(HEAT_R / cell.y))
	var flat := 0.0
	if _lum.is_empty():
		var c := _cloth_fallback().srgb_to_linear()
		flat = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722
	for gy in range(maxi(0, floori(g.y) - reach.y), mini(LUM_GRID.y, floori(g.y) + reach.y + 1)):
		for gx in range(maxi(0, floori(g.x) - reach.x), mini(LUM_GRID.x, floori(g.x) + reach.x + 1)):
			var r2 := Vector2((gx + 0.5 - g.x) * cell.x, (gy + 0.5 - g.y) * cell.y).length_squared()
			if r2 <= HEAT_R * HEAT_R:
				var lum := flat if _lum.is_empty() else _lum[gy * LUM_GRID.x + gx]
				out[gy * LUM_GRID.x + gx] = lum * hf / (r2 + hf * hf)
	return out


## [method _heat] at a grid cell's middle for a flame of [constant HEAT_H], kept for the build -
## a candle's place is judged at hundreds of spots.
func _heat_cell(at: Vector3) -> float:
	var cell := Vector2(CLOTH.x / LUM_GRID.x, CLOTH.y / LUM_GRID.y)
	var gx := clampi(floori((at.x - _cloth.position.x) / cell.x + LUM_GRID.x * 0.5), 0, LUM_GRID.x - 1)
	var gy := clampi(floori((at.z - _cloth.position.z) / cell.y + LUM_GRID.y * 0.5), 0, LUM_GRID.y - 1)
	var k := gy * LUM_GRID.x + gx
	if not _heat_cells.has(k):
		_heat_cells[k] = _heat(Vector3(_cloth.position.x + (gx + 0.5 - LUM_GRID.x * 0.5) * cell.x, 0.0,
			_cloth.position.z + (gy + 0.5 - LUM_GRID.y * 0.5) * cell.y), HEAT_H)
	return float(_heat_cells[k])


## HOW A FLAME FLICKERS - its own way, so no two keep time: a tempo, how steadily it burns, and
## how often a draft finds it. They all shared one tempo and one swing, and pulsed together.
func _flicker_of(salt: Variant) -> Dictionary:
	var r := RandomNumberGenerator.new()
	r.seed = hash([_seed, "flicker", salt])
	return {"seed": r.randf() * 1000.0, "rate": exp(r.randf_range(log(0.5), log(2.0))),
		"calm": r.randf_range(0.04, 0.12), "slot": r.randf_range(4.0, 12.0), "drafts": r.randf_range(0.3, 0.65),
		"salt": hash([_seed, "draft", salt])}


## A flame at show time [param t]: (brightness about 1, height about 1, lean). Mostly a steady burn
## - a slow sway and a fine shiver at the flame's own tempo - and now and then a DRAFT: for a second
## or two it gutters, dims and leans. A draft is drawn per slot of the flame's own length from a
## hash, so this is a pure function of time and a render and a scrub see the same flame.
func _flame_at(fk: Dictionary, t: float) -> Vector3:
	var s := float(fk["seed"])
	var r := float(fk["rate"])
	var calm := float(fk["calm"])
	var slow := _noise.get_noise_2d(t * 0.8 * r, s)
	var fast := _noise.get_noise_2d(t * 7.0 * r, s + 31.0)
	var slot := float(fk["slot"])
	var k := floori(t / slot)
	var h := hash([int(fk["salt"]), k])
	var env := 0.0
	var strength := 0.0
	if float(h & 0xFFFF) / 65535.0 < float(fk["drafts"]):
		var dur := 0.7 + 1.8 * float((h >> 16) & 0xFF) / 255.0
		var start := float(k) * slot + (slot - dur) * float((h >> 24) & 0xFF) / 255.0
		var u := (t - start) / dur
		if u > 0.0 and u < 1.0:
			env = pow(sin(PI * u), 2.0)
			strength = 0.18 + 0.2 * float((h >> 8) & 0xFF) / 255.0
	var gust := _noise.get_noise_2d(t * 13.0 * r, s + 57.0)
	return Vector3(1.0 + calm * (0.65 * slow + 0.35 * fast) + env * strength * (gust - 0.35),
		1.0 + calm * 2.5 * slow + env * strength * 1.4 * gust,
		_noise.get_noise_2d(t * 1.3 * r, s + 7.0) * 0.4 + env * gust * 1.6)


## A soft round shade, dark at its middle, made once: what a candle's base presses into the cloth.
func _contact_texture() -> Texture2D:
	if _contact_tex == null:
		var g := Gradient.new()
		g.offsets = PackedFloat32Array([0.0, 0.45, 1.0])
		g.colors = PackedColorArray([Color(0, 0, 0, 0.85), Color(0, 0, 0, 0.45), Color(0, 0, 0, 0.0)])
		var gt := GradientTexture2D.new()
		gt.gradient = g
		gt.fill = GradientTexture2D.FILL_RADIAL
		gt.fill_from = Vector2(0.5, 0.5)
		gt.fill_to = Vector2(1.0, 0.5)
		gt.width = 64
		gt.height = 64
		_contact_tex = gt
	return _contact_tex


func _tick_props(t: float) -> void:
	for l in _lights:
		# EACH FLAME IN ITS OWN TIME, and their light as bright as they are together
		var bright := 0.0
		var leans := Vector3.ZERO
		for f in (l as Dictionary)["flames"]:
			var fl := _flame_at(f["flicker"], t)
			var fast := _noise.get_noise_2d(t * 9.0 * float((f["flicker"] as Dictionary)["rate"]), float((f["flicker"] as Dictionary)["seed"]) + 91.0)
			var lean := Vector3(fl.z * 0.0015, 0, 0)
			var mesh: MeshInstance3D = f["mesh"]
			mesh.scale = Vector3(1.0 + 0.06 * fast, fl.y, 1.0)
			mesh.position = (f["base"] as Vector3) + lean
			# a flame is a sprite facing the reader
			mesh.look_at(mesh.global_position + (_cam.global_position - mesh.global_position) * Vector3(1, 0, 1), Vector3.UP)
			mesh.rotate_object_local(Vector3.UP, PI)
			if f.get("glow") is ShaderMaterial:
				(f["glow"] as ShaderMaterial).set_shader_parameter("flame", fl.x)
			bright += fl.x
			leans += lean
		var n := maxf(float(((l as Dictionary)["flames"] as Array).size()), 1.0)
		var light: OmniLight3D = (l as Dictionary)["light"]
		light.light_energy = float((l as Dictionary)["energy"]) * bright / n
		# the light leans with its flames, so the shadows breathe with them
		light.position = ((l as Dictionary)["light_base"] as Vector3) + leans / n
	for g in _glows:
		(g["light"] as OmniLight3D).light_energy = float(g["energy"]) * _flame_at(g["flicker"], t).x


# --- the camera ----------------------------------------------------------------------------------------

## The reader's eye: still.
func _tick_camera(_t: float) -> void:
	# A LOCKED-OFF CAMERA: the table is filmed from a tripod. The Camera dial's breath was a
	# millimeter - nothing anyone could see - so the dial is gone from the panel
	_cam.transform = _cam_base


## THE INTRO IS OUT OF FOCUS: the whole table behind a lens's bokeh while the channel's name is
## up, then the focus PULLS - near to far, the cloth in front of the reader first - as the
## shuffle starts, and the lens is off for the reading. The end card leaves it alone.
func _tick_focus(t: float) -> void:
	var ts := float(_times()["shuffle"])
	var pull := 1.0                      # a reading with no shuffle placed: nothing to wait for
	if ts < INF:
		pull = clampf((t - (ts - FOCUS_PULL.x)) / (FOCUS_PULL.x + FOCUS_PULL.y), 0.0, 1.0)
	elif _sched.is_empty():
		pull = 0.0                       # the intro: nothing placed yet
	if pull >= 1.0:
		_attrs.dof_blur_far_enabled = false
		return
	var e := _ease(pull)
	_attrs.dof_blur_far_enabled = true
	_attrs.dof_blur_far_distance = lerpf(0.04, 1.6, e * e)
	_attrs.dof_blur_far_transition = lerpf(0.05, 0.6, e)
	_attrs.dof_blur_amount = lerpf(FOCUS_BLUR, FOCUS_BLUR * 0.4, e)


# --- the air ---------------------------------------------------------------------------------------------

## THE AIR: the set dresser's effects (its table's `effects`, [Effects]) - fog, motes and bursts -
## built for this camera, with the stage's volumetric fog on only while there is fog to draw.
func _build_air(spec: Dictionary) -> void:
	if _air != null:
		_air.release()
		_air = null
	var fx: Array = spec.get("effects", []) if spec.get("effects") is Array else []
	if not fx.is_empty():
		_air = Effects.build(fx, _air_stage(), hash([_seed, "air"]))
		_root3.add_child(_air.root)
	var foggy: bool = _air != null and _air.has_fog()
	Effects.fog_environment(_env, foggy)
	# IN FOG, A LIGHT IS SEEN: the lamp's beam and every candle's glow scatter in it, shafts and halos
	_lamp.light_volumetric_fog_energy = 1.6 if foggy else 1.0
	for l in _lights:
		((l as Dictionary)["light"] as OmniLight3D).light_volumetric_fog_energy = 2.0 if foggy else 1.0
	_air_key = ""


## Where the air may be, the camera it is seen through, and what stands in it - the table and the
## things on it, which motes are homed in front of and fly over.
func _air_stage() -> Dictionary:
	var under: Array = [AABB(Vector3(-TABLE.x * 0.5, -TABLE.y, TABLE_Z - TABLE.z * 0.5), TABLE)]
	for th in _things:
		under.append((th as Dictionary)["box"])
	return {"regions": TarotTable.AIR, "camera": _cam_base, "fov": _cam.fov, "aspect": 16.0 / 9.0, "occluders": under}


## The air at show time [param t] - its bursts planned again whenever the schedule moved.
func _tick_air(t: float) -> void:
	if _air == null:
		return
	var key := "%d|%d" % [_built_n, _sched.size()]
	if key != _air_key:
		_air_key = key
		_air.plan(_air_moments())
	_air.tick(t)


## THE MOMENTS THE AIR CAN MARK, from the schedule: `{name: [{t, dur, path, from}]}` (see
## [constant TarotTable.MOMENTS]) - each with the time it starts, how long it lasts, and where its
## emitter is through it (`path`, `[[t, Transform3D], ...]`), read off the same poses the table draws.
## A moment the voice has not reached yet is not here.
func _air_moments() -> Dictionary:
	var out := {}
	for m in TarotTable.MOMENTS:
		out[m] = []
	var tm := _times()
	var keep := _cur_base
	if float(tm["shuffle"]) < INF:
		var ts := float(tm["shuffle"])
		(out["shuffle"] as Array).append({"t": ts, "dur": 0.5, "from": "point",
			"path": [[ts, Transform3D(Basis.IDENTITY, _mid + Vector3(0.0, DECK_T * DECK_N, 0.0))]]})
	var draws: Array = tm["draw"]
	var lays: Array = tm["lay"]
	for k in _cards.size():
		var d: Array = draws[k]
		var td := float(d[0])
		if td == INF:
			continue
		var s := maxf(float(d[1]), 0.05)
		var kind := String(d[2])
		var off := float(d[3])
		var up_at := td + (off + (JUMP_RISE if kind == "jumper" else RISE_END)) * s
		var tl := float((lays[k] as Array)[0])
		if kind == "jumper":
			# ALONG ITS FLIGHT, from springing off the riffle to landing
			var t0 := td + (off + JUMP_FLY.x) * s
			var t1 := td + (off + JUMP_FLY.y) * s
			var path: Array = []
			for i in 9:
				var tt := lerpf(t0, t1, float(i) / 8.0)
				_cur_base = _deck_at(tt, tm)
				path.append([tt, _jump_xf(k, (tt - td) / s - off, Transform3D.IDENTITY)])
			(out["jumper"] as Array).append({"t": t0, "dur": t1 - t0, "from": "card", "path": path})
		(out["reveal"] as Array).append({"t": up_at, "dur": 0.3, "from": "card",
			"path": [[up_at, _present_xf(k, up_at, up_at, tl)]]})
		for l in _looks(k, up_at, tl):
			var look: Dictionary = (l as Dictionary)["look"]
			if float(look["twirl"]) <= 0.0:
				continue
			# THROUGH THE TWIRL, the card's edges spinning with it
			var a0 := float((l as Dictionary)["at"]) + float(look["turn"]) + float(look["hold"])
			var dur := float(look["twirl"])
			var path: Array = []
			for i in 17:
				var tt := a0 + dur * float(i) / 16.0
				path.append([tt, _present_xf(k, tt, up_at, tl)])
			(out["pirouette"] as Array).append({"t": a0, "dur": dur, "from": "card", "path": path})
		if tl < INF:
			var land := tl + LAY_END * maxf(float((lays[k] as Array)[1]), 0.05)
			(out["lay"] as Array).append({"t": land, "dur": 0.2, "from": "card", "path": [[land, _slot_xf(k)]]})
	_cur_base = keep
	if float(tm["spread"]) < INF:
		var tc := float(tm["spread"])
		for k in _cards.size():
			(out["close"] as Array).append({"t": tc, "dur": 0.8, "from": "card", "path": [[tc, _slot_xf(k)]]})
	return out


# --- the schedule ----------------------------------------------------------------------------------------

## THE SHUFFLE'S CHAIN, made now, wash plans and all, rather than a frame at a time while it
## plays - and after the things on the table stand, so a wash's cards go round them, never
## through them. Again whenever the table changes.
func _plan_moves() -> void:
	_move_rng.seed = hash([_seed, "tarot-moves"])
	_moves = []
	_move_at(120.0)


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
## A jumper flies out of a riffle in the middle, and the deck goes once that riffle is done.
func _deck_at(t: float, tm: Dictionary) -> Vector3:
	var first: Array = tm["first"]
	if first.is_empty():
		return _mid
	var s := maxf(float(first[1]), 0.05)
	var go := SQUARE if String(first[2]) == "draw" else JUMP_RIFFLE
	var u := clampf(((t - float(first[0])) / s - go) / PUSH_SLIDE, 0.0, 1.0)
	return _mid.lerp(_deck_base, _ease(u)) + Vector3(0.0, sin(PI * u) * 0.004, 0.0)


# --- posing everything --------------------------------------------------------------------------------------

func _pose(t: float) -> void:
	var tm := _times()
	# a shuffle that began in the replayed past of a mid-way start is under way NOW, not a hundred
	# thousand seconds in (the chain of moves is only made so far)
	var ts := maxf(float(tm["shuffle"]), minf(0.0, _now))
	var te := float(tm["end"])
	_shuffle_room = te - ts if te < INF else INF
	_cur_base = _deck_at(t, tm)
	# a jumper's own riffle, on the jumper's clock (its action's scale)
	var first: Array = tm["first"]
	var jump_s := maxf(float(first[1]), 0.05) if not first.is_empty() and String(first[2]) == "jumper" else 0.0
	# THE DECK: shuffling from the shuffle mark until the first card leaves it, squared after - or,
	# when the first card is a jumper, riffled once more first: the card flies out of that riffle
	for i in DECK_N:
		var xf := _rest_xf(i)
		if t >= ts:
			if t < te:
				xf = _shuffle_xf(i, t - ts)
			elif jump_s > 0.0 and te < INF and t < te + JUMP_RIFFLE * jump_s:
				xf = _jump_riffle(i, (t - te) / jump_s, _shuffle_xf(i, te - ts))
			elif t < te + SQUARE and te < INF and jump_s == 0.0:
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


## How far card [param k] is turned over at [param t] (0 face on, PI showing its back - a whole
## turn more is the same pose): a look at its back ([method _look_of]), first some seconds after it
## is up and now and then another ([constant LOOK_AGAIN]) - never while it is coming up or about to
## go down, and for most cards never.
func _turn_of(k: int, t: float, up_at: float, until: float) -> float:
	if up_at == INF or t < up_at:
		return 0.0
	for l in _looks(k, up_at, until):
		var at := float((l as Dictionary)["at"])
		var look: Dictionary = (l as Dictionary)["look"]
		if t < at:
			return 0.0
		if t - at < float(look["total"]):
			return _look_angle(look, t - at)
	return 0.0


## EVERY LOOK AT CARD [param k]'s BACK while it is held - up at [param up_at], going down at
## [param until] - in order: `[{at, look}]` ([method _look_of]). What [method _turn_of] poses, and
## what the air's pirouettes are timed by, from one place.
func _looks(k: int, up_at: float, until: float) -> Array:
	var out: Array = []
	if up_at == INF:
		return out
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([_seed, k, "turn"])
	if rng.randf() > TURN_CHANCE:
		return out
	var at := up_at + rng.randf_range(5.0, 9.0)
	var spun := false
	for i in 64:
		var look := _look_of(rng, not spun)
		spun = spun or float(look["twirl"]) > 0.0
		if at + float(look["total"]) > until - 1.0:
			return out
		out.append({"at": at, "look": look})
		if rng.randf() > LOOK_AGAIN:
			return out
		at += float(look["total"]) + rng.randf_range(LOOK_GAP.x, LOOK_GAP.y)
	return out


## ONE LOOK AT A HELD CARD'S BACK, drawn: which way it turns (`way`, 1 or -1 - a hand turns a card
## either way), how long the turn over takes (`turn`), how long the back is held (`hold`), then
## either how long the turn back takes (`back`) or, for a PIROUETTE, the twirl on round (`twirl`) -
## the other 0; `total` its seconds. A card that has already pirouetted does not again
## ([param may_spin] false).
static func _look_of(rng: RandomNumberGenerator, may_spin := true) -> Dictionary:
	var way := 1.0 if rng.randf() < 0.5 else -1.0
	var turn := rng.randf_range(TURN.x, TURN.y)
	var spin := rng.randf() < PIROUETTE_CHANCE and may_spin
	var hold := rng.randf_range(PIROUETTE_HOLD.x, PIROUETTE_HOLD.y) if spin \
		else exp(rng.randf_range(log(TURN_HOLD.x), log(TURN_HOLD.y)))
	var twirl := rng.randf_range(TWIRL.x, TWIRL.y) if spin else 0.0
	var back := 0.0 if spin else rng.randf_range(TURN.x, TURN.y)
	return {"way": way, "turn": turn, "hold": hold, "twirl": twirl, "back": back,
		"total": turn + hold + twirl + back}


## How far a held card is turned (radians) [param v] seconds into [param look] ([method _look_of]):
## over to its back and held there; then back the way it came - or, in a pirouette, on round the
## same way: a flick that winds down onto its face, three whole turns from where it started.
static func _look_angle(look: Dictionary, v: float) -> float:
	var way := float(look["way"])
	var turn := float(look["turn"])
	var hold := float(look["hold"])
	if v < turn:
		return way * PI * _ease(v / turn)
	if v < turn + hold:
		return way * PI
	var twirl := float(look["twirl"])
	if twirl > 0.0:
		# quick out of the hold, long to wind down (still at both ends)
		return way * (PI + 5.0 * PI * _ease(pow(clampf((v - turn - hold) / twirl, 0.0, 1.0), 0.7)))
	return way * PI * (1.0 - _ease((v - turn - hold) / float(look["back"])))


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


## Deck slot [param i] in a jumper's riffle at [param v] (its action's seconds) - eased in over the
## first moments from [param was], wherever the shuffle left that slot.
func _jump_riffle(i: int, v: float, was: Transform3D) -> Transform3D:
	var xf := _riffle(i, v, JUMP_RIFFLE, hash([_seed, "jumper-riffle"]))
	return was.interpolate_with(xf, _ease(v / 0.35)) if v < 0.35 else xf


## Where a jumper rides before it springs: on top of the riffle's right half, one card above its
## top card (so it moves with the half), from the top of the deck as the riffle begins.
func _jump_ride(k: int, v: float) -> Transform3D:
	var top := _riffle(DECK_N - 1, v, JUMP_RIFFLE, hash([_seed, "jumper-riffle"]))
	top.origin += top.basis.y.normalized() * DECK_T
	top.basis = top.basis * Basis(Vector3.UP, PI if _reversed(k) else 0.0)
	return _deck_top_xf(k).interpolate_with(top, _ease(v / 0.35)) if v < 0.35 else top


## A jumper at [param u] seconds into its action: riding a riffle's half, springing off it as the
## halves fall, flying to land face up, lying there a moment, then picked up and shown.
func _jump_xf(k: int, u: float, pres: Transform3D) -> Transform3D:
	var land := Transform3D(Basis(Vector3.UP, 0.6 + (PI if _reversed(k) else 0.0)) * _face_up(),
		_jump_land + Vector3(0, CARD_T * 0.5, 0))
	if u < JUMP_FLY.x:
		return _jump_ride(k, u)
	if u < JUMP_FLY.y:
		var top := _jump_ride(k, JUMP_FLY.x)
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


## A WASH, posed from its plan (see [method _wash_plan]): card [param i] at [param v] seconds in,
## resting on what is under it ([method _wash_rest]) as far as it is spread on the cloth.
func _wash(i: int, v: float, m: Dictionary) -> Transform3D:
	var plan := _wash_fit(m)
	if plan.is_empty():
		return _rest_xf(i)
	var track: PackedVector4Array = (plan["tracks"] as Array)[i]
	var f := clampf(v * WASH_HZ, 0.0, float(track.size() - 1))
	var a := int(floor(f))
	var b := mini(a + 1, track.size() - 1)
	var q := track[a].lerp(track[b], f - float(a))
	# x, z about the deck's place; y above the cloth; w the card's turn. A card lying alone is a
	# card's thickness, not a deck slot's: the slot mesh is thinned while it is spread
	var spread := _wash_spread_at(plan, i, v)
	var thin := lerpf(1.0, CARD_T / DECK_T, spread)
	var ra: Vector3 = (_wash_rest(plan, a) as Array)[i]
	var rb: Vector3 = (_wash_rest(plan, b) as Array)[i]
	var r := ra.lerp(rb, f - float(a))
	var y := lerpf(q.y, r.x, spread)
	var g := Vector2(r.y, r.z) * spread
	return Transform3D(_tipped(q.w, g) * Basis.from_scale(Vector3(1.0, thin, 1.0)), _cur_base + Vector3(q.x, y, q.z))


## A card turned [param yaw] about the vertical and tipped to lie on a plane rising [param slope]
## (meters per meter across x and z): its long and short sides follow the plane, its face is square to it.
static func _tipped(yaw: float, slope: Vector2) -> Basis:
	var flat := Basis(Vector3.UP, yaw)
	if slope == Vector2.ZERO:
		return flat
	var x := flat.x + Vector3.UP * slope.dot(Vector2(flat.x.x, flat.x.z))
	var z := flat.z + Vector3.UP * slope.dot(Vector2(flat.z.x, flat.z.z))
	var up := z.cross(x).normalized()
	x = x.normalized()
	return Basis(x, up, x.cross(up).normalized())


## CARDS IN A WASH REST ON ONE ANOTHER (feedback 0007: "cards with a higher z-index in the stack
## seem to sort of hover... lifted off of the table... they cannot rest upon each other with a
## gentle tilt"). The plan lays a card flat at its place in the pile - one card's thickness above
## the deepest card it lies on - so where it hung past the cards under it, it hung in the air.
## Here every card spread on the cloth, from the bottom of the pile up, is a rigid card resting on
## whatever is under it: the cloth beneath its corners and the faces of the cards it lies on,
## wherever they cross it ([method _rest_on]). So it tips: on a card at one end, on the cloth at
## the other, and on the cards it lies on as they lie. Per step of the plan, kept for the frames
## between. Each card `Vector3(height of its middle, slope across x, slope across z)`, the plan's
## own height (and no slope) for one in a hand or the deck, which nothing rests on.
func _wash_rest(plan: Dictionary, k: int) -> Array:
	if not is_same(plan, _rest_plan):
		_rest_plan = plan
		_rest_steps = {}
	if _rest_steps.has(k):
		return _rest_steps[k]
	if _rest_steps.size() > 8:
		_rest_steps = {}
	var tracks: Array = plan["tracks"]
	var n := tracks.size()
	var v := float(k) / WASH_HZ
	var q: Array = []
	var order: Array = []
	for i in n:
		var tr: PackedVector4Array = tracks[i]
		q.append(tr[clampi(k, 0, tr.size() - 1)])
		order.append(i)
	order.sort_custom(func(x: int, y: int) -> bool:
		return (q[x] as Vector4).y < (q[y] as Vector4).y or ((q[x] as Vector4).y == (q[y] as Vector4).y and x < y))
	var out: Array = []
	out.resize(n)
	var under: Array = []      # [corners, middle, rest] of every card lying on the cloth so far
	var reach := Vector2(CARD.x, CARD.y).length()
	for i in order:
		var qi: Vector4 = q[i]
		if _wash_spread_at(plan, i, v) < 0.999:
			out[i] = Vector3(qi.y, 0.0, 0.0)
			continue
		var mid := Vector2(qi.x, qi.z)
		var corners := _card_corners(mid, qi.w)
		var pts := PackedVector2Array()
		var hs := PackedFloat32Array()
		for c in corners:
			pts.append(c - mid)
			hs.append(WASH_FLOOR)
		for u in under:
			var umid: Vector2 = u[1]
			if umid.distance_squared_to(mid) > reach * reach:
				continue
			var ur: Vector3 = u[2]
			for piece in Geometry2D.intersect_polygons(corners, u[0]):
				for p in piece:
					pts.append(p - mid)
					hs.append(ur.x + Vector2(ur.y, ur.z).dot(p - umid) + CARD_T + STACK_GAP)
		var rest := Vector3(WASH_FLOOR, 0.0, 0.0) if pts.size() == 4 else _rest_on(pts, hs)
		out[i] = rest
		under.append([corners, mid, rest])
	_rest_steps[k] = out
	return out


## A card's four corners on the cloth, its middle at [param mid], turned [param yaw].
static func _card_corners(mid: Vector2, yaw: float) -> PackedVector2Array:
	var ax := Vector2(cos(yaw), -sin(yaw)) * CARD.x * 0.5
	var az := Vector2(sin(yaw), cos(yaw)) * CARD.y * 0.5
	return PackedVector2Array([mid - ax - az, mid + ax - az, mid + ax + az, mid - ax + az])


## THE LOWEST A RIGID CARD CAN LIE on supports [param hs] high at [param pts] (about its middle):
## the plane above every one of them that is lowest at the middle - which is where a card settles,
## its weight being at its middle. It is the face, above the middle, of the hull over the
## supports, walked to from the highest of them: tip from that point toward the middle until a
## second holds it, swing about those two until a third does, and over the edge the middle lies
## beyond while it lies outside the three. Every step moves the plane only as far as the first
## support it meets, so no support ever ends up above it. `Vector3(height at the middle, slope
## across x, slope across z)`.
static func _rest_on(pts: PackedVector2Array, hs: PackedFloat32Array) -> Vector3:
	var n := pts.size()
	var top := 0
	for k in n:
		if hs[k] > hs[top]:
			top = k
	var h := hs[top]
	var g := Vector2.ZERO
	if pts[top].length() < 1e-7:
		return Vector3(h, 0.0, 0.0)
	# tip about the highest support, lowering the middle, until a second meets the plane
	var e := pts[top].normalized()
	var t := INF
	var second := -1
	for k in n:
		var de := e.dot(pts[top] - pts[k])
		if de > 1e-7 and (hs[top] - hs[k]) / de < t:
			t = (hs[top] - hs[k]) / de
			second = k
	if second < 0:
		return Vector3(h, 0.0, 0.0)
	g = e * t
	h = hs[top] - g.dot(pts[top])
	var held: Array = [top, second]
	for it in 12:
		if held.size() == 2:
			var pa: Vector2 = pts[held[0]]
			var dir: Vector2 = pts[held[1]] - pa
			if dir.length() < 1e-7:
				break
			var across := Vector2(-dir.y, dir.x).normalized()
			var side := across.dot(-pa)
			if absf(side) < 1e-7:
				break
			if side < 0.0:
				across = -across
			# swing about the two, lowering the middle's side, until a third meets the plane
			var s := INF
			var third := -1
			for k in n:
				var dn := across.dot(pts[k] - pa)
				if dn > 1e-7 and k != held[0] and k != held[1]:
					var sk := maxf(h + g.dot(pts[k]) - hs[k], 0.0) / dn
					if sk < s:
						s = sk
						third = k
			if third < 0:
				break
			g -= across * s
			h += s * across.dot(pa)
			held.append(third)
		# three hold it: settled if the middle lies among them, else over the edge it lies beyond
		var p0: Vector2 = pts[held[0]]
		var p1: Vector2 = pts[held[1]]
		var p2: Vector2 = pts[held[2]]
		var area := (p1 - p0).cross(p2 - p0)
		if absf(area) < 1e-10:
			held.remove_at(2)
			break
		var l0 := p1.cross(p2) / area
		var l1 := p2.cross(p0) / area
		var l2 := p0.cross(p1) / area
		var least := minf(l0, minf(l1, l2))
		if least >= -1e-6:
			break
		held.remove_at(0 if l0 == least else (1 if l1 == least else 2))
	return Vector3(h, g.x, g.y)


## A wash's plan for the time it has: the whole of it, or - when the first card comes before it
## would finish - one that mixes for less and gathers in time. The spreading and the hands are the
## same up to there (their own dice), so it is the same wash, cut short; without it the spread
## went back into the deck in [constant SQUARE]. With too little room to spread, mix and gather at
## all, there is no wash: the deck waits, squared, for the first card (empty).
func _wash_fit(m: Dictionary) -> Dictionary:
	var room := _shuffle_room - float(m["t0"])
	if room >= float(m["dur"]) - 0.05:
		return m["plan"]
	if room < WASH_ROOM:
		return {}
	var d := snappedf(room, 0.25)
	if float(m.get("cut_dur", -1.0)) != d:
		m["cut_dur"] = d
		m["cut"] = _wash_plan(int(m["seed"]), d, m["plan"])
	return m["cut"]


## How spread out card [param i] is at [param v] (0 squared in the deck, 1 lying on its own).
func _wash_spread_at(plan: Dictionary, i: int, v: float) -> float:
	var t_out := float((plan["out"] as PackedFloat32Array)[i])
	var t_in := float((plan["in"] as PackedFloat32Array)[i])
	return clampf((v - t_out) / 0.5, 0.0, 1.0) * (1.0 - clampf((v - t_in) / 0.6, 0.0, 1.0))


## [param at] (a card's middle, about the deck's place) moved out from anything standing on the
## table, by the least that keeps a card's whole reach clear of it; unchanged when clear.
func _clear_of_standing(at: Vector2) -> Vector2:
	var reach := Vector2(CARD.x, CARD.y).length() * 0.5 + 0.01
	var off := Vector2(_mid.x, _mid.z)
	var p := at
	for i in _standing.size():
		var c: Vector2 = _standing_c[i]
		var d := (p + off).distance_to(c)
		var need := _standing_r[i] + reach
		if d < need:
			var away := ((p + off) - c).normalized() if d > 1e-6 else Vector2(0.0, 1.0)
			p += away * (need - d)
	return p


## Whether a card's reach swept from [param a] to [param b] (about the deck's place) keeps clear of
## everything standing on the table.
func _path_clear(a: Vector2, b: Vector2) -> bool:
	var reach := Vector2(CARD.x, CARD.y).length() * 0.5
	var off := Vector2(_mid.x, _mid.z)
	for i in _standing.size():
		var c: Vector2 = _standing_c[i]
		var q := Geometry2D.get_closest_point_to_segment(c, a + off, b + off)
		if q.distance_to(c) < _standing_r[i] + reach:
			return false
	return true


## NOTHING PASSES THROUGH WHAT STANDS ON THE TABLE. A card going from [param from] to
## [param want] (about the deck's place), turned [param yaw], is moved there a few millimeters at a
## time, and each time it has run into something it is pushed back out - away from that thing's
## middle, the way it came - so it slides along it, and never jumps through to its far side.
func _card_clear(from: Vector2, want: Vector2, yaw: float) -> Vector2:
	if _standing.is_empty() or not _collide:
		return want
	var off := Vector2(_mid.x, _mid.z)
	var reach := Vector2(CARD.x, CARD.y).length() * 0.5
	var travel := from.distance_to(want)
	var near := false
	for i in _standing.size():
		if (want + off).distance_to(_standing_c[i]) < _standing_r[i] + reach + travel + 0.01:
			near = true
			break
	if not near:
		return want
	var steps := maxi(1, ceili(travel / 0.006))
	var p := from
	for s in steps:
		p += (want - from) / float(steps)
		for i in _standing.size():
			var c: Vector2 = _standing_c[i]
			if (p + off).distance_to(c) > _standing_r[i] + reach:
				continue
			p += _push_out(_card_poly(p + off, yaw), _standing[i], (p + off) - c)
	return p


## A card lying at [param at] turned [param yaw]: its four corners (x by z).
static func _card_poly(at: Vector2, yaw: float) -> PackedVector2Array:
	var ax := Vector2(cos(yaw), -sin(yaw)) * CARD.x * 0.5
	var az := Vector2(sin(yaw), cos(yaw)) * CARD.y * 0.5
	return PackedVector2Array([at - ax - az, at + ax - az, at + ax + az, at - ax + az])


## How far to move convex [param a] along [param dir] so it no longer overlaps convex [param b] -
## the least such move, by separating axes; along the shortest way out when [param dir] is
## nothing. Zero when they do not overlap.
static func _push_out(a: PackedVector2Array, b: PackedVector2Array, dir: Vector2) -> Vector2:
	var d := dir.normalized() if dir.length() > 1e-6 else Vector2.ZERO
	var need := INF
	var least := INF
	var least_ax := Vector2.ZERO
	for poly in [a, b]:
		var p: PackedVector2Array = poly
		for i in p.size():
			var e := p[(i + 1) % p.size()] - p[i]
			if e.length_squared() < 1e-12:
				continue
			var ax := Vector2(-e.y, e.x).normalized()
			var ra := _span(a, ax)
			var rb := _span(b, ax)
			var o := minf(ra.y, rb.y) - maxf(ra.x, rb.x)
			if o <= 0.0:
				return Vector2.ZERO
			if o < least:
				least = o
				least_ax = ax * (1.0 if ra.x + ra.y > rb.x + rb.y else -1.0)
			var k := d.dot(ax)
			if absf(k) > 1e-4:
				# moving a along d by s shifts it by s * k on this axis: past b's far side, or its near one
				var sep := (rb.y - ra.x) / k if k > 0.0 else (rb.x - ra.y) / k
				if sep >= 0.0:
					need = minf(need, sep)
	if need < INF:
		return d * (need + 0.0004)
	return least_ax * (least + 0.0004)


## Whether two cards lying on the table overlap: their turned rectangles, by separating axes.
static func _cards_overlap(a: Vector2, ya: float, b: Vector2, yb: float) -> bool:
	var d := b - a
	var far := 2.0 * Vector2(CARD.x, CARD.y).length() * 0.5
	if d.length_squared() > far * far:
		return false
	if d.length_squared() < CARD.x * CARD.x:
		return true
	# each card's two axes in turn (written out: a wash asks this some hundred thousand times)
	var a0 := Vector2(cos(ya), -sin(ya))
	var a1 := Vector2(sin(ya), cos(ya))
	var b0 := Vector2(cos(yb), -sin(yb))
	var b1 := Vector2(sin(yb), cos(yb))
	return not (_parted(d, a0, a0, a1, b0, b1) or _parted(d, a1, a0, a1, b0, b1)
		or _parted(d, b0, a0, a1, b0, b1) or _parted(d, b1, a0, a1, b0, b1))


## Whether two cards [param d] apart, turned to axes [param a0] [param a1] and [param b0]
## [param b1], are parted along [param v]: their reaches along it fall short of the gap.
static func _parted(d: Vector2, v: Vector2, a0: Vector2, a1: Vector2, b0: Vector2, b1: Vector2) -> bool:
	var h := CARD * 0.5
	var ra := h.x * absf(a0.dot(v)) + h.y * absf(a1.dot(v))
	var rb := h.x * absf(b0.dot(v)) + h.y * absf(b1.dot(v))
	return absf(d.dot(v)) > ra + rb


## THE WASH, planned once and sampled at [constant WASH_HZ] - because it is a SIMULATION, not a
## pose: two flat palms work the cloth and drag the cards under them along, and a pose that is a
## function of time alone cannot remember where a palm left a card. Planned from the move's seed,
## so it is the same every time it is posed.
##
##   out      the deck is pushed out across the cloth, top cards first (~2 s)
##   mix      most of the wash: the palms scrub back and forth across the spread, swirl wide and
##            fetch strays back through the middle ([method _wash_gesture]); a card a palm holds
##            goes its way ([method _wash_palm]), slides on when it lets go, and drags and turns
##            every card it passes over ([method _wash_drag])
##   gather   six to eight sweeps, each taking the next share of the cards by direction: most land
##            on the pile, some are only pushed near it and wait, some are missed and fetched later
##   square   the pile is squared into the deck
##
## THE ORDER CHANGES, NEVER THROUGH A CARD: two cards that come to overlap lie the way they met -
## the one sliding in goes on top (never over a card already above it: no stacking loops) - and
## keep that order while they touch; once apart, their next meeting decides again. A card's height
## is how many cards it lies on. Each track is per card, x / z about the deck's place, y above the
## cloth, w the card's turn.
func _wash_plan(seed: int, dur: float, base: Dictionary = {}) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "wash"])
	var n := DECK_N
	var steps := int(ceil(dur * WASH_HZ)) + 1
	var dt := 1.0 / WASH_HZ
	var out_end := minf(2.4, dur * 0.2)
	var square := 0.6
	# MANY SMALL SWEEPS: the pile is a few cards at a time, gathered one after another - never the
	# whole spread arriving at once
	var swipes := rng.randi_range(6, 8)
	var gather := clampf(dur * 0.24, 4.0, 7.0)
	var mix_end := maxf(out_end, dur - square - gather)
	# WIDE: the cards go well out across the cloth, a few of them a long way
	var rx := rng.randf_range(0.24, 0.29)
	var rz := rng.randf_range(0.12, 0.15)
	var pos := PackedVector2Array()
	var yaw := PackedFloat32Array()
	var start := PackedVector2Array()
	var aim := PackedVector2Array()
	var t_out := PackedFloat32Array()
	for i in n:
		var jit: Vector3 = _slot_jit[i] if i < _slot_jit.size() else Vector3.ZERO
		start.append(Vector2(jit.x, jit.y))
		# out to somewhere a card can get to: by a way clear of everything standing on the table (a
		# few tries, then the nearest spot clear of them)
		var target := Vector2.ZERO
		for attempt in 6:
			var r := sqrt(rng.randf()) * 0.85
			if rng.randf() < 0.18:
				r = rng.randf_range(0.95, 1.3)    # flung out further than the rest
			var ang := rng.randf() * TAU
			target = Vector2(cos(ang) * rx * r, sin(ang) * rz * r - 0.01)
			if _path_clear(Vector2(jit.x, jit.y), target):
				break
		aim.append(_clear_of_standing(target))
		pos.append(Vector2(jit.x, jit.y))
		yaw.append(jit.z)
		t_out.append(float(n - 1 - i) / float(n) * 0.9)
	var spin0 := PackedFloat32Array()
	for i in n:
		spin0.append(rng.randf_range(-1.1, 1.1))
	var was := pos.duplicate()             # where each card was at the step before, and its turn
	var was_yaw := yaw.duplicate()
	# HOW EACH CARD IS MOVING while it slides (x / z a second, and its turn a second), how hard a palm
	# held it at the last step, and whether a sweep of the gather moves it instead (or it lies in the
	# pile)
	var vel := PackedVector2Array()
	var spin := PackedFloat32Array()
	var held := PackedFloat32Array()
	var kin := PackedByteArray()
	vel.resize(n)
	spin.resize(n)
	held.resize(n)
	kin.resize(n)
	var hands := _wash_hands(seed, out_end, rx, rz)
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
	var from_p := PackedVector2Array()     # a sweep under way: where the card was, where it goes
	var to_p := PackedVector2Array()
	from_p.resize(n)
	to_p.resize(n)
	var from_yaw := PackedFloat32Array()
	var to_yaw := PackedFloat32Array()
	from_yaw.resize(n)
	to_yaw.resize(n)
	var begun := {}
	var land_order: Array = []
	var planned_sweeps := false
	var height := PackedFloat32Array()
	height.resize(n)
	# who lies on whom: pair (i * 64 + j, i < j) -> +1 i on top, -1 j on top, for pairs that overlap
	var rel := {}
	var under: Array = []                  # card -> the cards it lies directly on
	for i in n:
		under.append([])
	var over: Array = []                   # card -> the cards lying on it, at the step before
	for i in n:
		over.append([])
	var layer := PackedInt32Array()
	layer.resize(n)
	var moved := PackedFloat32Array()
	moved.resize(n)
	var stirred := PackedByteArray()       # moved or turned this step: only then can it meet or part
	stirred.resize(n)
	var restacked := true                  # who lies on whom changed this step
	# THE SAME WASH, CUT SHORT ([param base], the whole of it): up to where this one gathers, the
	# spreading and the palms are the same, so its tracks are copied rather than made again, and
	# the cards - where they lay, how they were sliding, and who on whom - are taken up from there
	var s_from := 0
	if not base.is_empty() and is_equal_approx(float((base["mix"] as Vector2).x), out_end) \
			and mix_end <= float((base["mix"] as Vector2).y):
		s_from = clampi(int(ceil(mix_end * WASH_HZ)), 2, ((base["tracks"] as Array)[0] as PackedVector4Array).size() - 1)
		for i in n:
			var tr: PackedVector4Array = (base["tracks"] as Array)[i]
			var copied: Array = []
			for st in s_from:
				copied.append(tr[st])
			tracks[i] = copied
			var q: Vector4 = tr[s_from - 1]
			var q0: Vector4 = tr[s_from - 2]
			pos[i] = Vector2(q.x, q.z)
			was[i] = pos[i]
			yaw[i] = q.w
			was_yaw[i] = q.w
			vel[i] = Vector2(q.x - q0.x, q.z - q0.z) * WASH_HZ
			spin[i] = (q.w - q0.w) * WASH_HZ
		for i in n:
			for j in range(i + 1, n):
				if _cards_overlap(pos[i], yaw[i], pos[j], yaw[j]):
					var yi := ((base["tracks"] as Array)[i] as PackedVector4Array)[s_from - 1].y
					var yj := ((base["tracks"] as Array)[j] as PackedVector4Array)[s_from - 1].y
					rel[i * 64 + j] = 1 if yi > yj else -1
					(under[i if yi > yj else j] as Array).append(j if yi > yj else i)
	for step in range(s_from, steps):
		var t := float(step) * dt
		if t < out_end:
			# OUT: each card slides from the deck to its place on the cloth, turning as it goes
			kin.fill(1)
			for i in n:
				var e := _ease(clampf((t - t_out[i]) / minf(1.1, out_end * 0.45), 0.0, 1.0))
				pos[i] = start[i].lerp(aim[i], e)
				yaw[i] = (_slot_jit[i] as Vector3).z + spin0[i] * e if i < _slot_jit.size() else spin0[i] * e
		elif t < mix_end:
			# MIX: the cloth slows what slides, cards drag what they lie on, the palms bring what they
			# press to their speed (last, so what a palm holds goes its way), and every card moves
			kin.fill(0)
			_wash_rub(hands, vel, spin, under, kin, dt)
			_wash_drag(hands, pos, vel, spin, held, kin, rel, dt)
			held.fill(0.0)
			for h in 2:
				_wash_palm(hands, h, t, dt, mix_end, pos, yaw, vel, spin, over, held)
			_wash_move(hands, pos, yaw, vel, spin, kin, dt)
		else:
			if not planned_sweeps:
				planned_sweeps = true
				# who each sweep takes: the sweeps go round the pile, each taking the next SHARE of
				# the cards by direction - so every sweep has cards to bring in, wherever the hands
				# left them (fixed directions sent whole sweeps past empty cloth). A card is
				# missed now and then and a later sweep fetches it; some are only pushed near.
				var by_angle: Array = []
				for i in n:
					by_angle.append([atan2(pos[i].y, pos[i].x), i])
				by_angle.sort_custom(func(x: Array, y: Array) -> bool: return float(x[0]) < float(y[0]))
				var first_card := rng.randi_range(0, n - 1)
				var share := PackedInt32Array()
				share.resize(n)
				for r in n:
					share[int((by_angle[(first_card + r) % n] as Array)[1])] = mini(swipes - 1, int(float(r) * float(swipes) / float(n)))
				for i in n:
					var k2 := share[i]
					while k2 < swipes - 1 and rng.randf() < 0.15:
						k2 += 1          # missed: a later sweep fetches it
					caught[i] = k2
					nudged[i] = -1
					if k2 < swipes - 1 and rng.randf() < 0.2:
						nudged[i] = k2   # caught, but only pushed near: brought in by the next
						caught[i] = k2 + 1
				# the pile's order is the order they arrive in
				var order: Array = []
				for i in n:
					order.append([caught[i], pos[i].length(), i])
				order.sort_custom(func(x: Array, y: Array) -> bool:
					if int(x[0]) != int(y[0]):
						return int(x[0]) < int(y[0])
					return float(x[1]) > float(y[1]))
				for o in order:
					land_order.append(int(o[2]))
			var g := t - mix_end
			kin.fill(0)
			for i in n:
				# in the pile for good: nothing slides it again
				if t_in[i] > 0.0 and t >= t_in[i]:
					kin[i] = 1
				for k in [nudged[i], caught[i]]:
					if int(k) < 0:
						continue
					# the sweep reaches the outermost cards first and pushes them in ahead of it
					var s0 := float(k) * sweep_len
					var u1 := s0 + sweep_len * 0.82
					var key := i * 8 + int(k)
					if not begun.has(key):
						var r := clampf(pos[i].length() / maxf(rx, rz), 0.0, 1.0)
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
							# squared as it lands, near enough: the pile is built, not tidied at the end
							to_p[i] = Vector2(jit2.x, jit2.y) + Vector2(rng.randf_range(-0.0025, 0.0025), rng.randf_range(-0.0025, 0.0025))
							to_yaw[i] = jit2.z + rng.randf_range(-0.05, 0.05)
							t_go[i] = mix_end + u0
							t_in[i] = mix_end + u1
						else:
							# caught, but only pushed up against the pile: the next sweep brings it in
							var away: Vector2 = pos[i].normalized() if pos[i].length() > 0.001 else Vector2.RIGHT
							to_p[i] = away * rng.randf_range(0.035, 0.055)
							to_yaw[i] = yaw[i] + rng.randf_range(-0.25, 0.25)
					var b0 := float(begun[key])
					if g > u1 + dt:
						continue
					var e2 := _ease(clampf((g - b0) / maxf(u1 - b0, 0.05), 0.0, 1.0))
					pos[i] = from_p[i].lerp(to_p[i], e2)
					yaw[i] = lerp_angle(from_yaw[i], to_yaw[i], e2)
					kin[i] = 1
			# WHAT NO SWEEP HAS REACHED YET SLIDES ON: the palms lift, and a card they let go of
			# mid-pass runs on and stops, rather than freezing where the gather began
			for i in n:
				if kin[i] == 1:
					vel[i] = Vector2.ZERO
					spin[i] = 0.0
			held.fill(0.0)
			_wash_rub(hands, vel, spin, under, kin, dt)
			_wash_drag(hands, pos, vel, spin, held, kin, rel, dt)
			_wash_move(hands, pos, yaw, vel, spin, kin, dt)
		# NOTHING PASSES THROUGH WHAT STANDS ON THE TABLE: every card goes from where it was to where
		# this step put it, and whatever it runs into stops it - it slides along it instead, at the
		# speed it really went
		for i in n:
			# a card lying still has nowhere new to be
			if kin[i] == 0 and vel[i] == Vector2.ZERO and spin[i] == 0.0:
				moved[i] = 0.0
				stirred[i] = 0
				continue
			pos[i] = _card_clear(was[i], pos[i], yaw[i])
			moved[i] = pos[i].distance_to(was[i])
			stirred[i] = 1 if moved[i] > 0.0 or yaw[i] != was_yaw[i] else 0
			if kin[i] == 0:
				vel[i] = (pos[i] - was[i]) * WASH_HZ
			was[i] = pos[i]
			was_yaw[i] = yaw[i]
		# WHO LIES ON WHOM: pairs that came apart forget their order; a pair that has just met lies
		# the way it met, the card sliding in on top (in the deck, the higher slot on top)
		var far2 := Vector2(CARD.x, CARD.y).length_squared()
		for i in n:
			var pi_: Vector2 = pos[i]
			for j in range(i + 1, n):
				# two cards that both lay still lie as they did - once the first step has said how
				if step > s_from and stirred[i] == 0 and stirred[j] == 0:
					continue
				var key := i * 64 + j
				var d2 := pi_.distance_squared_to(pos[j])
				var apart := d2 > far2 or (d2 > CARD.x * CARD.x and not _cards_overlap(pi_, yaw[i], pos[j], yaw[j]))
				if apart:
					if rel.has(key):
						var top := i if int(rel[key]) > 0 else j
						(under[top] as Array).erase(j if top == i else i)
						rel.erase(key)
						restacked = true
					continue
				if rel.has(key):
					continue
				var i_top := moved[i] > moved[j] + 1e-6 if absf(moved[i] - moved[j]) > 1e-6 else false
				# never over a card already above it, however far down the stack: no loops
				if i_top and _lies_on(under, j, i):
					i_top = false
				elif not i_top and _lies_on(under, i, j):
					i_top = true
				rel[key] = 1 if i_top else -1
				(under[i if i_top else j] as Array).append(j if i_top else i)
				restacked = true
				# A CARD RUN INTO IS KNOCKED: the one sliding in rides up over it and shoves it on
				# its way, turning it if it was struck off its middle ([constant WASH_KNOCK])
				if kin[i] == 0 and kin[j] == 0:
					var top := i if i_top else j
					var bot := j if i_top else i
					var c := (pos[top] + pos[bot]) * 0.5
					var dv := (vel[top] - vel[bot]) * float(hands["knock"])
					_wash_push(vel, spin, bot, c - pos[bot], dv)
					_wash_push(vel, spin, top, c - pos[top], -dv)
		# HEIGHTS: a card one above the highest card it lies on; alone, flat on the cloth just above it
		if restacked:
			restacked = false
			layer.fill(-1)
			for i in n:
				_layer_of(i, under, layer)
			for i in n:
				(over[i] as Array).clear()
			for i in n:
				for j in under[i]:
					(over[int(j)] as Array).append(i)
			for i in n:
				height[i] = WASH_FLOOR + float(layer[i]) * (CARD_T + STACK_GAP)
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
	# SQUARE: from wherever the pile left each card to its slot in the deck - turned the short way
	# round, whatever turns the wash gave it
	var last := steps - 1
	var sq0 := int(floor((dur - square) * WASH_HZ))
	for i in n:
		var slot := land_order.find(i)
		if slot < 0:
			slot = i
		var jit3: Vector3 = _slot_jit[slot] if slot < _slot_jit.size() else Vector3.ZERO
		var tr := PackedVector4Array(tracks[i] as Array)
		var from: Vector4 = tr[mini(sq0, tr.size() - 1)]
		var to := Vector4(jit3.x, (float(slot) + 0.5) * DECK_T, jit3.y, from.w + wrapf(jit3.z - from.w, -PI, PI))
		for st in range(sq0, tr.size()):
			var e3 := _ease(float(st - sq0) / maxf(float(last - sq0), 1.0))
			tr[st] = from.lerp(to, e3)
		tracks[i] = tr
	return {"tracks": tracks, "out": t_out, "in": t_in, "spread": 1.0, "order": land_order,
		"mix": Vector2(out_end, mix_end), "hands": hands["log"]}


## How many cards card [param i] lies on, one on another at the deepest: its layer in the pile, kept in
## [param layer] (-1 for one not walked yet). Who lies on whom never loops, so the walk ends.
static func _layer_of(i: int, under: Array, layer: PackedInt32Array) -> int:
	if layer[i] >= 0:
		return layer[i]
	var l := 0
	for j in under[i]:
		l = maxi(l, _layer_of(int(j), under, layer) + 1)
	layer[i] = l
	return l


## Whether card [param a] lies, however far down, on card [param b] - walking down what each lies on.
static func _lies_on(under: Array, a: int, b: int) -> bool:
	var seen := {}
	var todo: Array = [a]
	while not todo.is_empty():
		var c := int(todo.pop_back())
		for d in under[c]:
			if int(d) == b:
				return true
			if not seen.has(d):
				seen[d] = true
				todo.append(d)
	return false


## THE HANDS OF A WASH: their own dice - so a wash cut short ([method _wash_fit]) has the same hands
## up to where it was cut - and their own FEEL, each wash sampled round the centers ([constant
## WASH_GRIP] and on): how firmly a palm holds, how cards drag and slide, how briskly these hands
## move. Each palm: its gesture under way, when it next comes down, its size, and how it holds
## each card this pass; and every gesture the palms made, in order ([palm, gesture]).
func _wash_hands(seed: int, t0: float, rx: float, rz: float) -> Dictionary:
	var r := RandomNumberGenerator.new()
	r.seed = hash([seed, "wash-palms"])
	var palms: Array = []
	for h in 2:
		var grip := PackedFloat32Array()
		grip.resize(DECK_N)
		palms.append({"g": {}, "next": t0 + r.randf_range(0.0, 0.4), "pass": -1, "grip": grip,
			"size": PALM * r.randf_range(0.9, 1.12)})
	var touched := PackedFloat32Array()
	touched.resize(DECK_N)
	return {"r": r, "palms": palms, "rx": rx, "rz": rz, "touched": touched, "log": [],
		"grip": WASH_GRIP * r.randf_range(0.85, 1.15), "drag": WASH_DRAG * r.randf_range(0.8, 1.25),
		"slide": WASH_SLIDE * r.randf_range(0.85, 1.2), "spin": WASH_SPIN * r.randf_range(0.8, 1.25),
		"knock": WASH_KNOCK * r.randf_range(0.75, 1.25),
		"pace": r.randf_range(0.85, 1.15)}


## PALM [param h] for the step from [param t]. Lifted between gestures, it comes down on its next
## ([method _wash_gesture]) where the cards are. At each of [constant GRIP_AT]'s points it presses, a
## card is brought to the palm's speed - as far as the palm's friction there allows ([constant
## WASH_GRIP]): a card pressed whole goes with it, one caught at an end swings round behind it, one
## only brushed slips. Where another card lies over it the palm hardly touches it ([constant
## WASH_COVERED]), so a card half under another is pulled out by its free half. Each pass the palm
## takes hold afresh - the heel of a hand bears on some cards and skims others - so a scrub carries
## a different few each way, and what it carried out it often leaves there.
func _wash_palm(hands: Dictionary, h: int, t: float, dt: float, t_end: float, pos: PackedVector2Array,
		yaw: PackedFloat32Array, vel: PackedVector2Array, spin: PackedFloat32Array, over: Array,
		held: PackedFloat32Array) -> void:
	var p: Dictionary = (hands["palms"] as Array)[h]
	var r: RandomNumberGenerator = hands["r"]
	var g: Dictionary = p["g"]
	if not g.is_empty() and t >= float(g["t1"]):
		# lifted: a moment's reach to the next place, now and then a rest
		p["next"] = float(g["t1"]) + (r.randf_range(0.5, 1.2) if r.randf() < WASH_REST else r.randf_range(0.08, 0.3))
		g = {}
		p["g"] = g
	if g.is_empty():
		if t < float(p["next"]) or t > t_end - 0.5:
			return
		g = _wash_gesture(hands, h, t, pos, t_end)
		p["g"] = g
		p["pass"] = -1
		if g.is_empty():
			p["next"] = t + 0.25
			return
		(hands["log"] as Array).append([h, g])
	var a := _gesture_at(g, t)
	var b := _gesture_at(g, t + dt)
	var press := minf(a.z, b.z)
	if press <= 0.0:
		return
	var grip: PackedFloat32Array = p["grip"]
	var k := _gesture_pass(g, t)
	if k != int(p["pass"]):
		p["pass"] = k
		for i in grip.size():
			var u := r.randf()
			# held (it goes with the palm), dragged (it slides along slower, and falls behind), or
			# brushed (it stays): how the palm's friction there compares with the cloth's
			grip[i] = 1.0 if u < 0.45 else (r.randf_range(0.1, 0.2) if u < 0.75 else r.randf_range(0.02, 0.07))
	var at := Vector2(a.x, a.y)
	var v := (Vector2(b.x, b.y) - at) / dt
	var size: Vector2 = p["size"]
	# the hand lies along the forearm, from the reader's shoulder on its side
	var along := (at - Vector2(WASH_SHOULDER.x * (-1.0 if h == 0 else 1.0), WASH_SHOULDER.y)).normalized()
	var across := Vector2(-along.y, along.x)
	var reach := size.y * 1.05 + Vector2(CARD.x, CARD.y).length() * 0.5
	var most := float(hands["grip"]) * dt / float(GRIP_AT.size())
	var touched: PackedFloat32Array = hands["touched"]
	for i in pos.size():
		if pos[i].distance_squared_to(at) > reach * reach:
			continue
		var ax := Vector2(cos(yaw[i]), -sin(yaw[i]))
		var az := Vector2(sin(yaw[i]), cos(yaw[i]))
		var hold := press * grip[i]
		for q in GRIP_AT:
			var o: Vector2 = q
			var rr := ax * o.x + az * o.y
			var d := pos[i] + rr - at
			var e := Vector2(d.dot(across) / size.x, d.dot(along) / size.y).length()
			if e >= 1.05:
				continue
			var w := hold * (1.0 - smoothstep(0.75, 1.05, e))
			# where another card lies over it, the palm presses that one instead
			for k2 in over[i]:
				if _card_has(pos[int(k2)], yaw[int(k2)], pos[i] + rr):
					w *= WASH_COVERED
					break
			held[i] = maxf(held[i], w)
			_wash_push(vel, spin, i, rr, v - vel[i] - Vector2(rr.y, -rr.x) * spin[i], most * w)
		if held[i] > 0.2:
			touched[i] = t


## Whether a card lying at [param at] turned [param yaw] covers the point [param pt] (x by z).
static func _card_has(at: Vector2, yaw: float, pt: Vector2) -> bool:
	var d := pt - at
	return absf(d.dot(Vector2(cos(yaw), -sin(yaw)))) <= CARD.x * 0.5 and absf(d.dot(Vector2(sin(yaw), cos(yaw)))) <= CARD.y * 0.5


## Card [param i]'s point [param rr] (from its middle) brought [param dv] nearer the speed it is
## pushed toward - no more than [param most] m/s of the card's own speed changed (a friction's
## limit; the card slips past it): it moves and turns as a flat card would, pushed there, so a push
## through its middle only moves it and one at an end turns it too.
static func _wash_push(vel: PackedVector2Array, spin: PackedFloat32Array, i: int, rr: Vector2, dv: Vector2,
		most: float = INF) -> void:
	var rp := Vector2(rr.y, -rr.x)
	var j := dv - rp * (rp.dot(dv) / (CARD_I + rr.length_squared()))
	if j.length() > most:
		j *= most / j.length()
	vel[i] += j
	spin[i] += rp.dot(j) / CARD_I


## THE CLOTH SLOWS EVERY LOOSE CARD by the same each second, as cloth does - a card let go mid-pass
## runs on a little and stops, sooner on cloth than on another card - and its turning stops too.
static func _wash_rub(hands: Dictionary, vel: PackedVector2Array, spin: PackedFloat32Array, under: Array,
		kin: PackedByteArray, dt: float) -> void:
	var slide := float(hands["slide"]) * dt
	var turn := float(hands["spin"]) * dt
	for i in vel.size():
		if kin[i] == 1 or (vel[i] == Vector2.ZERO and spin[i] == 0.0):
			continue
		var sp := vel[i].length()
		if sp > 0.0:
			var mu := slide * (1.0 if (under[i] as Array).is_empty() else WASH_ON_CARD)
			vel[i] *= maxf(0.0, sp - mu) / sp
		spin[i] = signf(spin[i]) * maxf(0.0, absf(spin[i]) - turn)


## CARDS DRAG CARDS: two lying one on the other pull each toward the other's speed where they touch
## - hard when a palm pressed the top one at the last step ([constant WASH_DRAG], [param held]), so a
## card scrubbed across the spread drags and turns what it passes over and leaves a wake of cards
## knocked askew. Before the palms, so what a palm holds goes its way whatever lies under it.
func _wash_drag(hands: Dictionary, pos: PackedVector2Array, vel: PackedVector2Array, spin: PackedFloat32Array,
		held: PackedFloat32Array, kin: PackedByteArray, rel: Dictionary, dt: float) -> void:
	var far := Vector2(CARD.x, CARD.y).length()
	var drag: Vector2 = hands["drag"]
	for key in rel:
		var i := int(key) >> 6
		var j := int(key) & 63
		if kin[i] == 1 or kin[j] == 1:
			continue
		if vel[i] == Vector2.ZERO and vel[j] == Vector2.ZERO and spin[i] == 0.0 and spin[j] == 0.0:
			continue
		var top := i if int(rel[key]) > 0 else j
		var bot := j if top == i else i
		var k := (drag.x + drag.y * held[top]) * (1.0 - clampf(pos[top].distance_to(pos[bot]) / far, 0.0, 1.0))
		if k <= 0.0:
			continue
		var c := (pos[top] + pos[bot]) * 0.5
		var rt := c - pos[top]
		var rb := c - pos[bot]
		var dv := ((vel[top] + Vector2(rt.y, -rt.x) * spin[top]) - (vel[bot] + Vector2(rb.y, -rb.x) * spin[bot])) \
			* (0.5 * (1.0 - exp(-k * dt)))
		_wash_push(vel, spin, bot, rb, dv)
		_wash_push(vel, spin, top, rt, -dv)


## THE CARDS MOVE, one step: none faster than a hand ([constant HAND_SPEED]), and one going out past
## the spread's edge is turned back, the harder the farther out, as a hand would.
func _wash_move(hands: Dictionary, pos: PackedVector2Array, yaw: PackedFloat32Array, vel: PackedVector2Array,
		spin: PackedFloat32Array, kin: PackedByteArray, dt: float) -> void:
	var rx := float(hands["rx"])
	var rz := float(hands["rz"])
	for i in pos.size():
		if kin[i] == 1 or (vel[i] == Vector2.ZERO and spin[i] == 0.0):
			continue
		var sp := vel[i].length()
		if sp > HAND_SPEED:
			vel[i] *= HAND_SPEED / sp
		spin[i] = clampf(spin[i], -WASH_TURN_MAX, WASH_TURN_MAX)
		var q := Vector2(pos[i].x / rx, (pos[i].y + 0.01) / rz)
		var l := q.length()
		if l > 1.0:
			var nrm := Vector2(q.x / rx, q.y / rz).normalized()
			var vn := vel[i].dot(nrm)
			if vn > 0.0:
				vel[i] -= nrm * vn * clampf((l - 1.0) / 0.15, 0.0, 1.0)
		pos[i] += vel[i] * dt
		yaw[i] += spin[i] * dt


## A PALM'S NEXT GESTURE, from [param t]. It comes down on a card - mostly one on its own side of the
## spread, and one the hands have left alone a while - and the best of a few tries keeps clear of
## the other palm ([constant WASH_APART]). How often each ([constant WASH_GESTURES]):
##
##   scrub   passes back and forth, each across a good part of the spread ([constant WASH_SCRUB]),
##           its line turning and drifting a little between them
##   swirl   wide rounds, often the other way round from the other palm's
##   fetch   out to the card lying farthest out, and back through the middle with it
##
## A scrub or a fetch is its turning points `pts` at times `ts`, pressed `press` on each pass; a
## swirl its middle `c` (drifting `v`), radii `a` and `b` turned `rot`, where round it `ph`, how
## fast `om` and how hard `press`. Empty when there is no time left for one.
func _wash_gesture(hands: Dictionary, h: int, t: float, pos: PackedVector2Array, t_end: float) -> Dictionary:
	var r: RandomNumberGenerator = hands["r"]
	var rx := float(hands["rx"])
	var rz := float(hands["rz"])
	var pace := float(hands["pace"])
	var side := -1.0 if h == 0 else 1.0
	var touched: PackedFloat32Array = hands["touched"]
	var other: Dictionary = ((hands["palms"] as Array)[1 - h] as Dictionary)["g"]
	var best := {}
	var best_apart := -1.0
	for attempt in 6:
		# where it comes down
		var ws := PackedFloat32Array()
		var total := 0.0
		for i in pos.size():
			var w := (1.0 / (1.0 + exp(-side * pos[i].x / 0.07)) + 0.08) * (0.3 + clampf((t - touched[i]) / 2.5, 0.0, 1.0))
			ws.append(w)
			total += w
		var pick := r.randf() * total
		var land := pos[pos.size() - 1]
		for i in pos.size():
			pick -= ws[i]
			if pick <= 0.0:
				land = pos[i]
				break
		# the card lying farthest out, for a fetch
		var stray := -1
		var out_most := WASH_STRAY
		for i in pos.size():
			var k := Vector2(pos[i].x / rx, (pos[i].y + 0.01) / rz).length() * (1.15 if pos[i].x * side > 0.0 else 1.0)
			if k > out_most:
				out_most = k
				stray = i
		var u := r.randf() * (float(WASH_GESTURES["scrub"]) + float(WASH_GESTURES["swirl"]) + float(WASH_GESTURES["fetch"]))
		var g := {}
		if u < float(WASH_GESTURES["swirl"]):
			var a := rx * r.randf_range(0.28, 0.5)
			var b := rz * r.randf_range(0.5, 0.85)
			var rot := r.randf_range(-0.4, 0.4)
			var om := r.randf_range(3.2, 5.5) * pace
			if String(other.get("kind", "")) == "swirl" and r.randf() < 0.65:
				om *= -signf(float(other["om"]))
			elif r.randf() < 0.5:
				om = -om
			var ph := r.randf() * TAU
			var c := land - Vector2(cos(ph) * a, sin(ph) * b).rotated(rot)
			c = Vector2(clampf(c.x, -rx * 0.92 + a, rx * 0.92 - a), clampf(c.y, -rz * 0.92 + b - 0.01, rz * 0.92 - b - 0.01))
			var t1 := minf(t + r.randf_range(0.75, 1.5) * TAU / absf(om), t_end - 0.1)
			if t1 - t < 0.5:
				continue
			g = {"kind": "swirl", "t0": t, "t1": t1, "c": c, "v": Vector2.from_angle(r.randf() * TAU) * r.randf_range(0.0, 0.02),
				"a": a, "b": b, "rot": rot, "ph": ph, "om": om, "press": r.randf_range(0.6, 1.0)}
		else:
			var pts := PackedVector2Array()
			var ts := PackedFloat32Array([t])
			var press := PackedFloat32Array()
			if u < float(WASH_GESTURES["swirl"]) + float(WASH_GESTURES["fetch"]) and stray >= 0:
				# FETCH: onto the stray's outer edge, in through the middle and on a little past it, and
				# now and then a lighter half-pass back
				var p0 := pos[stray] + (pos[stray] + Vector2(0.0, 0.01)).normalized() * 0.02
				var p1 := _wash_into(Vector2(r.randf_range(-0.35, 0.35) * rx, r.randf_range(-0.35, 0.35) * rz - 0.01), rx, rz, 0.9)
				p1 = _wash_into(p1 + (p1 - p0).normalized() * r.randf_range(0.0, 0.08), rx, rz, 0.92)
				pts.append(p0)
				pts.append(p1)
				press.append(r.randf_range(0.8, 1.0))
				if r.randf() < 0.4:
					pts.append(p1.lerp(p0, r.randf_range(0.35, 0.6)))
					press.append(r.randf_range(0.4, 0.7))
			else:
				# SCRUB: from the card, toward somewhere a good way across the spread, then back and
				# forth along that line - turning and drifting a little each pass
				var p := _wash_into(land + Vector2(r.randf_range(-0.025, 0.025), r.randf_range(-0.02, 0.02)), rx, rz, 0.92)
				var to := _wash_into(Vector2(r.randf_range(-1.0, 1.0) * rx, r.randf_range(-1.0, 1.0) * rz - 0.01), rx, rz, 0.9)
				var dir := (to - p).normalized() if p.distance_to(to) > 0.08 else Vector2.from_angle(r.randf() * TAU)
				var span := r.randf_range(WASH_SCRUB.x, WASH_SCRUB.y)
				pts.append(p)
				for pass_ in r.randi_range(2, 6):
					var d := dir.rotated(r.randf_range(-0.3, 0.3)) * (1.0 if pass_ % 2 == 0 else -1.0)
					var q := _wash_into(p + d * span * r.randf_range(0.75, 1.1) + Vector2(-d.y, d.x) * r.randf_range(-0.035, 0.035), rx, rz, 0.92)
					if p.distance_to(q) < 0.06:
						break
					pts.append(q)
					press.append(r.randf_range(0.6, 1.0))
					p = q
			# each pass in its own time, at a hand's pace for its length
			for k in range(1, pts.size()):
				var tt := ts[k - 1] + pts[k - 1].distance_to(pts[k]) / (r.randf_range(0.32, 0.58) * pace)
				if tt > t_end - 0.1:
					break
				ts.append(tt)
			if ts.size() < 2:
				continue
			pts.resize(ts.size())
			press.resize(ts.size() - 1)
			g = {"kind": "scrub", "t0": t, "t1": ts[ts.size() - 1], "pts": pts, "ts": ts, "press": press}
		var apart := _palms_apart(g, other)
		if apart > best_apart:
			best = g
			best_apart = apart
		if apart >= WASH_APART:
			break
	return best


## Gesture [param g] at [param t]: (x, z, how hard it presses - nothing outside it, coming down and
## lifting over [constant WASH_TOUCH]). A scrub's pass eases out to a stop at each end.
static func _gesture_at(g: Dictionary, t: float) -> Vector3:
	var t0 := float(g["t0"])
	var t1 := float(g["t1"])
	if t < t0 or t > t1:
		return Vector3.ZERO
	var touch := clampf((t - t0) / WASH_TOUCH, 0.0, 1.0) * clampf((t1 - t) / WASH_TOUCH, 0.0, 1.0)
	if String(g["kind"]) == "swirl":
		var u := t - t0
		var ph := float(g["ph"]) + float(g["om"]) * u
		var p := (g["c"] as Vector2) + (g["v"] as Vector2) * u + Vector2(cos(ph) * float(g["a"]), sin(ph) * float(g["b"])).rotated(float(g["rot"]))
		return Vector3(p.x, p.y, touch * float(g["press"]))
	var ts: PackedFloat32Array = g["ts"]
	var pts: PackedVector2Array = g["pts"]
	var k := _gesture_pass(g, t)
	var p2 := pts[k].lerp(pts[k + 1], _ease((t - ts[k]) / maxf(ts[k + 1] - ts[k], 0.001)))
	return Vector3(p2.x, p2.y, touch * (g["press"] as PackedFloat32Array)[k])


## Which pass of gesture [param g] is under way at [param t]: a scrub's leg, a swirl's half-round.
static func _gesture_pass(g: Dictionary, t: float) -> int:
	if String(g["kind"]) == "swirl":
		return int(absf(float(g["om"])) * (t - float(g["t0"])) / PI)
	var ts: PackedFloat32Array = g["ts"]
	var k := 0
	while k < ts.size() - 2 and t > ts[k + 1]:
		k += 1
	return k


## How near gesture [param g] comes to the other palm's [param other] while both press (INF when
## they never do at once).
static func _palms_apart(g: Dictionary, other: Dictionary) -> float:
	if g.is_empty() or other.is_empty():
		return INF
	var least := INF
	var t := maxf(float(g["t0"]), float(other["t0"]))
	var t1 := minf(float(g["t1"]), float(other["t1"]))
	while t <= t1:
		var a := _gesture_at(g, t)
		var b := _gesture_at(other, t)
		if a.z > 0.0 and b.z > 0.0:
			least = minf(least, Vector2(a.x, a.y).distance_to(Vector2(b.x, b.y)))
		t += 0.1
	return least


## [param p] brought in to the spread's edge, scaled by [param k], when it lies out past it.
static func _wash_into(p: Vector2, rx: float, rz: float, k: float) -> Vector2:
	var q := Vector2(p.x / rx, (p.y + 0.01) / rz)
	var l := q.length()
	if l <= k:
		return p
	q *= k / l
	return Vector2(q.x * rx, q.y * rz - 0.01)


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

## The channel's name over the table while the intro holds, gone as the shuffle starts - and
## only then: the reading ends on the table fading to black, not on the name again.
func _title_alpha(t: float) -> float:
	var ts := float(_times()["shuffle"])
	var a := clampf((t - 0.3) / 1.1, 0.0, 1.0)
	if ts < INF:
		a *= 1.0 - clampf((t - (ts - 0.6)) / 0.9, 0.0, 1.0)
	elif not _sched.is_empty():
		a = 0.0
	return a


## THE OUTRO: once the voice has said its last word, the table fades to black - a beat after it,
## and black exactly as the outro's silence runs out (the take's own tail in a render, the
## Director's outro live). A function of show time like everything else, so a live reading, a
## scrub and the export fade alike. 1 until then.
func _end_fade(t: float) -> float:
	var total := (_parse.get("spoken", PackedStringArray()) as PackedStringArray).size()
	var last := _follow.known_last()
	if total == 0 or last < total - 1:
		return 1.0
	var outro := maxf(Spectrum.tail if Spectrum.bookend_baked else Director.outro_hold, 0.0)
	var beat := minf(0.8, outro * 0.15)
	return 1.0 - _ease((t - (_follow.st1[last] + beat)) / maxf(outro - beat, 0.05))


class TitleCard:
	extends Node2D

	var channel := ""
	var episode := ""
	var face: Font = null
	var italic: Font = null
	var alpha := 0.0

	func _draw() -> void:
		if alpha <= 0.001 or face == null or channel.is_empty():
			return
		var vp := get_viewport_rect().size
		var s := vp.y / 1080.0
		var size := int(84.0 * s)
		var y := vp.y * 0.4
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
		if episode.is_empty() or italic == null:
			return
		var es := TarotCards._fit(italic, episode, int(40.0 * s), vp.x * 0.78)
		var ey := y + 70.0 * s
		draw_string(italic, Vector2(2, ey + 2) * Vector2(1, 1), episode, HORIZONTAL_ALIGNMENT_CENTER, vp.x, es, shadow)
		draw_string(italic, Vector2(0, ey), episode, HORIZONTAL_ALIGNMENT_CENTER, vp.x, es, ink)
