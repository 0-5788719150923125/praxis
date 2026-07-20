extends CanvasLayer
class_name SynthEditor

## SynthEditor - the synthesis surface. The loop is a fishing trip:
##
##   THROW into the unknown -> something ANCHORS and pulls (for a long while)
##   -> PULL to set the hook -> the REEL: the adrenochrome anneals toward and
##   against your party, then FREEZES -> ACCEPT or RELEASE.
##
## A throw speaks a new candidate immediately. Its planned strikes (echo,
## stretch, pitch, hesitation - the sparse activations) are the **bites**: when
## one plays, a detection window opens and decays (an EMA over the moment) -
## catch quickly after the hit and the odds are strong; wait and the trace
## fades. It is never gone from the GENOME though: land the seed at any moment
## and every effect it carries is kept forever, and will strike again.
##
## A successful catch is not yet a keep: it presents a **card** - the seed
## drawn as a constellation (lines seeded random, colour attuned to the cosine
## similarity between the candidate and the party's centre). **Accept** folds
## it into the belt - and every member's colour re-attunes to the new party,
## in ways you only learn by doing it. **Release** changes nothing. The game
## is integration: reading the colour shifts, knowing when to hold, when to
## fold. Throwing again folds a pending card.
##
## Reward profiles (Drift / Snap / Hunt) shape how throws range and how
## catches earn. Acceptance (hold time + rewards vs the belt average) weights
## parent selection; ancestors earn from their line. Everything autosaves.

const CFG := "user://ghost.cfg"
const AUTOSAVE_DELAY_MS := 800
const BELT_MAX := 7                 # a small cache; old captures fall off the end

# Easy fishing: bites are LATENT - a strike latches an anchor that pulls for a
# long while (slow decay), occasionally letting go on its own. No reflexes.
const ANCHOR_TAU := 45.0            # seconds: how slowly an anchored pull fades
const ANCHOR_VANISH := 0.03         # chance per check that the pull just leaves
const ANCHOR_CHECK := 5.0           # seconds between vanish checks
# Nibbles: each DISTINCT fresh strike that lands while the anchor still holds
# counts as one - patience (sitting through several) steadies the pull odds,
# rather than the anchor just re-latching to the same ceiling each time.
const NIBBLE_MAX := 5
const NIBBLE_BONUS := 0.05          # added to pull odds per accumulated nibble
# The reel: once the hook sets, the ADRENOCHROME accumulates - a genome
# annealing toward the party's attractors and against its repulsors, freezing
# at the end. Sit and wait; the calculation is the catch.
const HOOK_STEPS := 140             # anneal steps across the reel

# THE TOLL - the karma. This game extracts something from a creature by
# dragging it home against its will, and nobody does that for free: the cost
# comes back through the only channel the player actually inhabits, the VOICE
# THEY HAVE TO LISTEN TO. The toll grows with how foreign the catch is (it has
# nothing to do with kin, who cost nothing) and with how far it has already
# been reeled in, and it works four ways at once:
#   1. every repulsion the party exerts is amplified - the further in it comes,
#      the harder your own collection shoves it away;
#   2. the population prior loses its grip - the anchor that keeps any voice
#      ordinary stops holding a creature this foreign;
#   3. the annealing schedule inverts, COOLING becomes HEATING - it does not
#      settle as it nears, it comes apart;
#   4. the roughness axes fray directly, so the damage is audible rather than
#      merely statistical.
# Kin (difficulty ~0) anneal calmly exactly as they always did.
const TOLL_REPEL := 2.5             # extra repulsion at full toll
const TOLL_PRIOR_LOSS := 0.75       # how much of the prior's grip the toll dissolves
const TOLL_HEAT := 3.0              # noise multiplier at full toll (vs cooling at none)
const TOLL_GENOME_SPAN := 2.0       # extra genome clamp width at full toll
# Both deformation terms are quadratic in the toll: kin pay nothing at all, and
# the cost climbs steeply only once you are genuinely dragging something that
# does not belong. They are also deliberately STRONG - a linear nudge was
# measured losing to the party's own forces by ~10:1, which left a foreign
# catch arriving no uglier than kin (worse, occasionally SMOOTHER, when the
# party happened to sit on the rough side of the axis).
const TOLL_FRAY := 0.14             # saturating drive toward damage (roughness axes)
const TOLL_EXTREME := 0.01          # drive away from the ordinary (every other axis)
# The axes that make a voice sound DAMAGED rather than merely different:
# glottal roughness, breath noise, and the static band. These are also all
# live-retunable, so the fraying is heard while the reel runs, not after it.
const FRAY_KEYS := ["grit", "air", "breath"]
# A voice may become hideous, but it must remain a VOICE: unbounded pace genes
# would stretch a take to many times its length, and every future playback,
# render and export of that seed would pay it forever.
const PACE_MIN := 0.55
const PACE_MAX := 2.2
# The trust region: a fresh draw is normalized RELATIVE to the pool being
# harmonized with (the party's centre; the curated default when the belt is
# empty). A candidate landing beyond the radius is pulled back onto it -
# extremes stay reachable by DRIFTING (the cage widens the region), they just
# stop being a routine dice roll onto a broken voice (grit x air x pitch all
# at the far edge at once synthesized as clicks and static, not character).
const TEMPER_RADIUS := 0.85

# The three MODES of fishing - what the line is doing, not just how reward is
# scored (each still earns differently):
# DRIFT - the crab cage: the cast is left behind while you keep moving, so the
#   line's scale grows with time - metres become miles become lightyears. The
#   cosmos opens (more planets visible at distance), and the longer the drift,
#   the more foreign the next throw pulls in. Time sitting with it earns.
# ANCHOR - freeze in place and search THIS area: throws stay tight around the
#   current candidate, the pull rarely leaves on its own. Quick, decisive
#   catches earn.
# REEL - retrieval: pulling toward you. The retrieval speed is data- and
#   learning-dependent - a seasoned belt (accumulated hold statistics) reels
#   faster, because it can preview the catch's influence more accurately.
#   Foreignness earns.
const PROFILES := {
	"drift": {"label": "Drift", "wild": 0.15, "jitter": 0.8, "vanish": 0.03,
		"tip": "The crab cage: leave the cast behind. The line's scale grows - miles, lightyears - the cosmos opens, and time earns."},
	"anchor": {"label": "Anchor", "wild": 0.05, "jitter": 0.6, "vanish": 0.008,
		"tip": "Freeze in place and search this area. Throws stay close, the pull holds. Quick decisive catches earn."},
	"reel": {"label": "Reel", "wild": 0.45, "jitter": 1.5, "vanish": 0.03,
		"tip": "Retrieval. A seasoned belt reels faster - accumulated statistics preview the catch's influence. Foreignness earns."},
}

var begin_stream: Callable          # set by main: (stream: VoiceStream) -> void
var end_stream: Callable            # set by main: () -> void, tears the session down

var _panel: PanelContainer
var _text: TextEdit
var _status: Label
var _stream: VoiceStream = null
var _traits := {}                   # the current voice - carried by seeds, never hand-set
var _lineage: Array = [1]           # the reading's seed chain (root + refinements)
var _belt: Array = []               # kept seeds: [{lineage, traits, m}]
var _reading_label: Label
var _belt_rows: VBoxContainer
var _inv_labels: Array = []         # metrics Label per inventory row
var _inv_glyphs: Array = []         # SeedGlyph per inventory row
var _catch_btn: Button
var _card: PanelContainer           # the reel (annealing) and the pending catch
var _card_glyph: Control
var _card_label: Label
var _accept_btn: Button
var _release_btn: Button
var _pending := {}                  # {d, reward, traits, genome} frozen, undecided
var _loop_len := 0.0                # take length once complete (time wrapping)
var _anchor := 0.0                  # the latent pull: latched by strikes, slow to fade
var _vanish_t := 0.0                # timer for the pull-leaves-on-its-own check
var _nibbles := 0                   # distinct fresh strikes felt since the anchor was last empty
var _last_strike_t := -999.0        # stream-local time of the last strike counted as a nibble
var _hook := {}                     # the live reel: rng, step, t, duration, members,
                                    # traits, genome (annealing), d, reward
var _working_genome := {}           # adrenochrome carried by the WORKING candidate
var _metrics_t := 0.0
var _status_t2 := 0.0               # throttle for the calm line readout
var _profile := "anchor"
var _profile_btns := {}             # switchboard: profile key -> toggle Button
var _hud: Control                   # the LCD water (custom drawn)
var _hud_glyph: Control             # the candidate constellation living in it
var _throw_ms := 0
var _catching := false
var _dirty := false
var _restart_pending := false
var _last_edit_ms := 0
var _cast := false                  # silent until the first throw of the session
var _landed := false                # the working voice is a landed/kept catch
var _retune_t := 0.0                # reel audition retune pacing
var _line_btn: Button               # Throw when the water is empty, Release once cast
var _line_gen := 0                  # bumped on release: in-flight animations check it


func _ready() -> void:
	layer = 10
	_build_panel()
	_load_persisted()
	var args := OS.get_cmdline_user_args()
	var i := args.find("--synth")
	if i >= 0 and i + 1 < args.size() and FileAccess.file_exists(args[i + 1]):
		_text.text = FileAccess.get_file_as_string(args[i + 1])
	_text.text_changed.connect(_mark_structural)
	# the water is SILENT at launch - no auto-speak; the first throw casts the
	# voice. --say (demos, headless checks) counts as a thrown-and-landed cast.
	if args.has("--say") and not _text.text.strip_edges().is_empty():
		_cast = true
		_landed = true
		_update_reading_label()
		_apply.call_deferred()
	_sync_line_button()


func _build_panel() -> void:
	_panel = PanelContainer.new()
	_panel.position = Vector2(16, 16)
	_panel.custom_minimum_size = Vector2(380, 0)
	add_child(_panel)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	_panel.add_child(box)

	var title_row := HBoxContainer.new()
	box.add_child(title_row)
	var title := Label.new()
	title.text = "Synthesis"
	title.add_theme_font_size_override("font_size", 20)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_row.add_child(title)
	var hide := Button.new()
	hide.text = "–"
	hide.tooltip_text = "Hide panel (F2)"
	hide.custom_minimum_size = Vector2(28, 28)
	hide.pressed.connect(func(): _panel.visible = false)
	title_row.add_child(hide)

	var hint := Label.new()
	hint.text = "Write, then throw - the water is silent until you cast. Pull when it anchors; the reel is the fight; hold or fold."
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.add_theme_font_size_override("font_size", 12)
	hint.modulate = Color(1, 1, 1, 0.6)
	box.add_child(hint)

	_text = TextEdit.new()
	_text.custom_minimum_size = Vector2(360, 180)
	_text.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_text.placeholder_text = "Once upon a time..."
	box.add_child(_text)

	# --- The HUD: the water. A dedicated LCD readout where the fishing is
	# drawn - the candidate's constellation, the line that visibly pulls when
	# something anchors, the reel's progress while the adrenochrome forms -
	# with the reward switchboard on its right edge.
	var hud_panel := PanelContainer.new()
	var hud_style := StyleBoxFlat.new()
	hud_style.bg_color = Color(0.015, 0.045, 0.03, 0.96)
	hud_style.border_color = Color(0.25, 0.55, 0.4, 0.55)
	hud_style.set_border_width_all(1)
	hud_style.set_corner_radius_all(4)
	hud_style.set_content_margin_all(6)
	hud_panel.add_theme_stylebox_override("panel", hud_style)
	box.add_child(hud_panel)
	var hud_col := VBoxContainer.new()
	hud_col.add_theme_constant_override("separation", 2)
	hud_panel.add_child(hud_col)
	var hud_row := HBoxContainer.new()
	hud_row.add_theme_constant_override("separation", 8)
	hud_col.add_child(hud_row)
	_hud = Hud.new()
	_hud.editor = self
	_hud.custom_minimum_size = Vector2(0, 84)
	_hud.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hud_row.add_child(_hud)
	_hud_glyph = SeedGlyph.new()
	_hud_glyph.position = Vector2(10, 18)
	_hud_glyph.size = Vector2(48, 48)
	_hud.add_child(_hud_glyph)
	var board := VBoxContainer.new()
	board.add_theme_constant_override("separation", 4)
	hud_row.add_child(board)
	for key in PROFILES:
		var b := Button.new()
		b.text = PROFILES[key].label
		b.toggle_mode = true
		b.focus_mode = Control.FOCUS_NONE
		b.custom_minimum_size = Vector2(64, 24)
		b.add_theme_font_size_override("font_size", 11)
		b.tooltip_text = PROFILES[key].tip
		b.button_pressed = key == _profile
		var pkey: String = key
		b.pressed.connect(func():
			_profile = pkey
			_sync_switchboard()
			_persist())
		board.add_child(b)
		_profile_btns[key] = b
	# the readout line lives IN the HUD - this is a wearable, and the text is
	# part of the instrument, not chrome below it
	_status = Label.new()
	_status.text = "the water is silent - write, then throw"
	_status.add_theme_font_size_override("font_size", 11)
	_status.add_theme_color_override("font_color", Color(0.55, 0.95, 0.75, 0.85))
	hud_col.add_child(_status)

	# --- The loop: Throw / Pull ---
	var loop_row := HBoxContainer.new()
	loop_row.add_theme_constant_override("separation", 8)
	box.add_child(loop_row)
	# ONE button for the line itself: Throw while the water is empty, Release
	# once something is out there - including while a catch is hooked, so a
	# fight can always be abandoned. Releasing returns everything to baseline.
	_line_btn = Button.new()
	_line_btn.custom_minimum_size = Vector2(110, 40)
	_line_btn.pressed.connect(_on_line_button)
	loop_row.add_child(_line_btn)
	_catch_btn = Button.new()
	_catch_btn.text = "Pull"
	_catch_btn.custom_minimum_size = Vector2(90, 40)
	_catch_btn.tooltip_text = ("Set the hook. Something anchored will pull for a "
		+ "long while - pull while it does and the hook sets or is lost; a slack "
		+ "line almost never sets. Let it nibble: each fresh strike while it holds "
		+ "steadies your odds. A rougher, more jittery line means a harder catch.")
	_catch_btn.pressed.connect(_pull)
	loop_row.add_child(_catch_btn)
	_reading_label = Label.new()
	_reading_label.add_theme_font_size_override("font_size", 12)
	_reading_label.modulate = Color(1, 1, 1, 0.7)
	loop_row.add_child(_reading_label)

	# --- The card: a caught seed awaiting judgment (hold or fold) ---
	_card = PanelContainer.new()
	_card.visible = false
	box.add_child(_card)
	var card_row := HBoxContainer.new()
	card_row.add_theme_constant_override("separation", 10)
	_card.add_child(card_row)
	_card_glyph = SeedGlyph.new()
	_card_glyph.custom_minimum_size = Vector2(64, 64)
	card_row.add_child(_card_glyph)
	var card_col := VBoxContainer.new()
	card_col.alignment = BoxContainer.ALIGNMENT_CENTER
	card_col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	card_row.add_child(card_col)
	_card_label = Label.new()
	_card_label.add_theme_font_size_override("font_size", 12)
	card_col.add_child(_card_label)
	var card_btns := HBoxContainer.new()
	card_btns.add_theme_constant_override("separation", 8)
	card_col.add_child(card_btns)
	_accept_btn = Button.new()
	_accept_btn.text = "Accept"
	_accept_btn.tooltip_text = "Fold it into the party - every member's colour re-attunes"
	_accept_btn.pressed.connect(_accept_catch)
	card_btns.add_child(_accept_btn)
	_release_btn = Button.new()
	# "Fold", not "Release": the line has its own Release now, and this is the
	# other decision entirely - hold or fold the catch you are being shown
	_release_btn.text = "Fold"
	_release_btn.tooltip_text = "Let it go - nothing changes"
	_release_btn.pressed.connect(_release_catch)
	card_btns.add_child(_release_btn)

	# --- The belt/bag: ALWAYS last ---
	_belt_rows = VBoxContainer.new()
	_belt_rows.add_theme_constant_override("separation", 4)
	box.add_child(_belt_rows)


## The exporter's take provider: render the CURRENT text through the voice
## once, write the WAV + subtitle sidecar, and hand back the path. A COROUTINE:
## the snapshot (text + spec + take name) is taken on the main thread so the
## worker never touches live UI or fishing state, then the render AND the
## per-sample WAV encode run on a WorkerThreadPool task while the app keeps
## drawing. Running them synchronously froze the whole window for as long as
## the draft was long (synthesis is ~20x real time and the WAV encode is a
## GDScript loop too) - and because the voice plays from its own thread, the
## audio kept going, so the stall read as a hang, not a wait.
## Is there anything worth exporting? The export does NOT need the live player
## running - it renders from the seeds. A belt with seeds can always speak (the
## voice is the party's blend even before a cast); a voice already cast can
## speak as itself. An empty belt with nothing cast has no voice to render, and
## the button greys out.
func can_export_take() -> bool:
	return not _text.text.strip_edges().is_empty() and (_cast or not _belt.is_empty())


func export_take() -> String:
	var text := _text.text.strip_edges()
	if text.is_empty():
		return ""
	# The export is independent of the realtime game: with nothing cast, the
	# take is read by the PARTY - the acceptance-weighted blend of the belt,
	# which is the same background voice the water would speak with. The
	# lineages on the belt also join as influences, so the export reflects the
	# whole collection rather than one arbitrary working candidate.
	var spec := _current_spec()
	if not _cast and not _belt.is_empty():
		spec = Voice.Spec.from_traits(_background_traits(), int(_lineage[0]), _lineage)
		spec.adrenochrome = _working_genome.duplicate()
		var infl: Array = []
		for e in _belt:
			infl.append((e.lineage as Array).duplicate())
		spec.influences = infl
	# distinct from the live stream's take file: the stream's worker may still
	# be synthesizing the SAME lineage and would race this write on completion
	var base := "user://synth/take_%06x_export" % (hash(str(_lineage)) & 0xFFFFFF)
	DirAccess.make_dir_recursive_absolute("user://synth")
	var done := {}
	var task := WorkerThreadPool.add_task(func():
		var result := Voice.render(text, spec)
		var wav := Voice.write_wav(base + ".wav", result.pcm)
		var side := FileAccess.open(base + ".json", FileAccess.WRITE)
		side.store_string(JSON.stringify({"words": result.words}))
		side.close()
		done["wav"] = wav)
	while not WorkerThreadPool.is_task_completed(task):
		await get_tree().process_frame
	WorkerThreadPool.wait_for_task_completion(task)
	return String(done.get("wav", ""))


func _current_spec() -> Voice.Spec:
	var spec := Voice.Spec.from_traits(_traits, int(_lineage[0]), _lineage)
	spec.adrenochrome = _working_genome.duplicate()
	return spec


func _sync_switchboard() -> void:
	for key in _profile_btns:
		var b: Button = _profile_btns[key]
		if is_instance_valid(b):
			b.set_pressed_no_signal(key == _profile)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_F2:
		_panel.visible = not _panel.visible


# ---- the loop: throw, bite, catch, accept / release ------------------------


## The line button: one gesture, two meanings, decided by whether anything is
## out on the water.
func _on_line_button() -> void:
	if _cast:
		_release_line()
	else:
		_throw()


## Keep the line button honest about what it will do.
func _sync_line_button() -> void:
	if _line_btn == null or not is_instance_valid(_line_btn):
		return
	if _cast:
		_line_btn.text = "Release"
		_line_btn.tooltip_text = ("Cut the line and return to still water: the voice "
			+ "stops, the candidate is let go, and whatever was hooked is lost. "
			+ "Your belt is untouched.")
	else:
		_line_btn.text = "Throw"
		_line_btn.tooltip_text = ("Into the unknown: a new candidate, parented from the "
			+ "belt by acceptance (sometimes wild), spoken immediately.")
	# nothing on the water can be pulled - and a pull there would roll the odds
	# against a candidate that isn't swimming yet
	if _catch_btn != null and is_instance_valid(_catch_btn):
		_catch_btn.disabled = not _cast


## RELEASE - back to baseline. The water goes quiet (the session is torn down,
## not just muted), the working candidate is let go, and anything hooked or
## waiting for judgment is abandoned. The belt keeps everything it earned.
func _release_line() -> void:
	_line_gen += 1                   # in-flight orb animations must not resurrect a hook
	_pending = {}
	_card.visible = false
	_hook = {}
	_anchor = 0.0
	_nibbles = 0
	_last_strike_t = -999.0
	_catching = false
	_loop_len = 0.0
	Director.set_aura(0.0)
	if end_stream.is_valid():
		end_stream.call()
	_stream = null
	_cast = false
	_landed = false
	# no active seed: the working candidate returns to the population baseline
	# (the party's own blend, or the curated default when the belt is empty)
	_lineage = [1]
	_traits = _background_traits()
	_working_genome = {}
	_restart_pending = false
	_sync_line_button()
	_update_reading_label()
	_refresh_inventory()
	_persist()
	_status.text = "released - the water is still"


func _throw() -> void:
	if not _pending.is_empty():
		_release_catch()             # throwing again folds the pending card
	if not _hook.is_empty():
		_hook = {}                   # abandoning the reel loses the creature
		Director.set_aura(0.0)
		_card.visible = false
		_status.text = "the line snapped - thrown again"
	_anchor = 0.0
	_nibbles = 0
	_last_strike_t = -999.0
	_cast = true                     # the first throw breaks the silence
	_landed = false                  # a fresh candidate swims far away
	var prof: Dictionary = PROFILES[_profile]
	# the crab cage: a long drift pulls the next throw from further away -
	# wilder odds, wider jitter, the cage hauled in from wherever it got to
	var drift := _drift_norm()
	if _belt.is_empty() or randf() < float(prof.wild) + 0.3 * drift:
		_lineage = [randi() % 1000000]
		var rng := RandomNumberGenerator.new()
		rng.seed = _lineage[0]
		_traits = _temper_traits(Voice.Spec.sample(rng).traits, drift)
		_working_genome = {}         # wild = a fresh lineage-derived genome
		_status.text = "thrown (wild) - the water is quiet"
	else:
		var parent: Dictionary = _pick_parent()
		parent.m.evolves += 1
		_lineage = (parent.lineage as Array).duplicate()
		_lineage.append(randi() % 1000000)
		var jitter: float = 0.22 * pow(0.75, _lineage.size() - 1) * float(prof.jitter) \
			* (1.0 + 1.5 * drift)
		var t: Dictionary = parent.traits
		_traits = {}
		for key in Voice.TRAIT_KEYS:
			_traits[key] = clampf(
				float(t.get(key, 0.0)) + randfn(0.0, maxf(jitter, 0.06)), -1.0, 1.0)
		_traits = _temper_traits(_traits, drift)
		# a child of an adrenochrome seed inherits the frozen genome verbatim -
		# the reading still varies (gates, motifs, field ride the new lineage)
		_working_genome = (parent.get("genome", {}) as Dictionary).duplicate()
		_status.text = "thrown (from %s) - the water is quiet" % _seed_name(parent.lineage)
	_throw_ms = Time.get_ticks_msec()
	_sync_line_button()
	_update_reading_label()
	_persist()
	_apply()


## Normalize a candidate's trait vector against the pool it must harmonize
## with: the party's acceptance-weighted centre ({} = the curated default when
## the belt is empty) anchors a trust region of TEMPER_RADIUS (RMS over the
## trait axes); a draw beyond it is pulled back onto the boundary, direction
## intact. Drifting widens the region - foreignness is EARNED by the cage,
## not rolled.
func _temper_traits(t: Dictionary, drift: float) -> Dictionary:
	var center := _background_traits()
	var radius := TEMPER_RADIUS * (1.0 + 0.8 * drift)
	var dist := 0.0
	for key in Voice.TRAIT_KEYS:
		var dv := float(t.get(key, 0.0)) - float(center.get(key, 0.0))
		dist += dv * dv
	dist = sqrt(dist / Voice.TRAIT_KEYS.size())
	if dist <= radius:
		return t
	var scale := radius / dist
	var out := {}
	for key in Voice.TRAIT_KEYS:
		var c := float(center.get(key, 0.0))
		out[key] = clampf(c + (float(t.get(key, 0.0)) - c) * scale, -1.0, 1.0)
	return out


func _pick_parent() -> Dictionary:
	var weights: Array = []
	var total := 0.0
	for e in _belt:
		var w := 0.1 + _acceptance(e)
		weights.append(w)
		total += w
	var roll := randf() * total
	for i in _belt.size():
		roll -= weights[i]
		if roll <= 0.0:
			return _belt[i]
	return _belt[-1]


## A strike playing right now (within the last ~1.2s) - the moment a latent
## pull LATCHES onto the line. Not the catch window: the anchor persists long
## after, fading slowly. The effect is genome-carried either way. Returns
## (amplitude, event time) so callers can tell a genuinely NEW strike (a fresh
## nibble) apart from the same strike still sitting in the trailing window.
func _fresh_strike() -> Vector2:
	if _stream == null or not is_instance_valid(_stream):
		return Vector2(0.0, -1.0)
	var t: float = Spectrum.current.time - _stream.time_base
	if _loop_len > 0.0:
		t = fmod(t, _loop_len)
	var best := 0.0
	var best_t := -1.0
	for e in _stream.events:
		var dt: float = t - float(e.t)
		if dt >= 0.0 and dt < 1.2 and float(e.a) >= best:
			best = float(e.a)
			best_t = float(e.t)
	return Vector2(best, best_t)


func _candidate_difficulty() -> float:
	if _belt.is_empty():
		return 0.35
	var cg: Dictionary = _working_genome if not _working_genome.is_empty() \
		else Voice.ProsodyWalk._lineage_genome(_lineage)
	var best := 1.0
	for e in _belt:
		var dt := 0.0
		for key in Voice.TRAIT_KEYS:
			var dv: float = float(_traits.get(key, 0.0)) \
				- float((e.traits as Dictionary).get(key, 0.0))
			dt += dv * dv
		dt = sqrt(dt / Voice.TRAIT_KEYS.size()) / 0.9
		var eg: Dictionary = _member_genome(e)
		var dg := 0.0
		for key in Voice.ProsodyWalk.PRIOR:
			var prior: float = float(Voice.ProsodyWalk.PRIOR[key])
			dg += absf(float(cg.get(key, prior)) - float(eg.get(key, prior))) \
				/ maxf(absf(prior), 0.001)
		dg = (dg / Voice.ProsodyWalk.PRIOR.size()) / 0.35
		best = minf(best, 0.5 * clampf(dt, 0.0, 1.0) + 0.5 * clampf(dg, 0.0, 1.0))
	return clampf(best, 0.0, 1.0)


func _catch_reward(difficulty: float) -> float:
	var t_s: float = (Time.get_ticks_msec() - _throw_ms) / 1000.0
	match _profile:
		"drift":
			return 2.4 * clampf(t_s / 45.0, 0.0, 1.0)
		"reel":
			return 0.4 + 2.0 * difficulty
		_:
			return 2.4 * 8.0 / (8.0 + t_s)


## How far the crab cage has drifted (0..1 over ~5 minutes) - drift mode only.
func _drift_norm() -> float:
	if _profile != "drift":
		return 0.0
	return clampf((Time.get_ticks_msec() - _throw_ms) / 1000.0 / 300.0, 0.0, 1.0)


## The drifted line's length, in honest units - the wire's timescale changing.
func _drift_caption() -> String:
	var meters := pow(10.0, 1.0 + 14.5 * _drift_norm())
	if meters < 1000.0:
		return "≈ %d m" % int(meters)
	if meters < 1.0e7:
		return "≈ %.1f km" % (meters / 1000.0)
	if meters < 1.0e10:
		return "≈ %.0f mi" % (meters / 1609.34)
	if meters < 1.0e14:
		return "≈ %.2f au" % (meters / 1.496e11)
	return "≈ %.2f ly" % (meters / 9.461e15)


## Planets become visible further out as the cage drifts - the cosmos opens.
func _planet_threshold() -> float:
	return 0.72 - 0.25 * _drift_norm()


## Accumulated information is POWER on the line: a seasoned belt (hold time +
## earned reward = what the party has LEARNED) resonates with a hooked catch
## and retrieves it faster - in EVERY mode; knowledge is not a mode. This is
## the reel's side of the fight: power against the creature's runs.
func _reel_power() -> float:
	var knowledge := 0.0
	for e in _belt:
		knowledge += e.m.s + 20.0 * float(e.m.get("r", 0.0))
	return clampf(0.9 + knowledge / 600.0, 0.9, 2.2)


## PULL - the second axis. Something anchored pulls for a long while; pull
## while it does and the hook SETS or is LOST. Lost: the pull may persist for
## another try, or leave. Set: the reel begins - the long accumulation.
func _pull() -> void:
	if _catching or not _hook.is_empty() or not _pending.is_empty():
		return
	if not _cast:
		_status.text = "nothing is out there - throw first"
		return
	for e in _belt:
		if e.lineage == _lineage:
			_status.text = "already on the belt"
			return
	var d := _candidate_difficulty()
	var nibble_bonus: float = NIBBLE_BONUS * float(_nibbles)
	var p := clampf(0.1 + 0.6 * minf(_anchor, 1.0) - 0.2 * d + nibble_bonus, 0.03, 0.9)
	var success := randf() < p
	var wobbles := 3 if success else 1 + randi() % 3
	_catching = true
	var gen := _line_gen             # a release mid-animation must void this pull
	var orb := CatchOrb.new()
	orb.anchor = _hud.get_global_rect().get_center()
	orb.hue = float(hash(str(_lineage)) % 360) / 360.0
	orb.wobbles = wobbles
	orb.success = success
	add_child(orb)
	orb.finished.connect(func():
		orb.queue_free()
		if gen != _line_gen:
			return                   # the line was cut while the orb was closing
		_catching = false
		if success:
			_begin_hook(d)
		else:
			# lost - and by coin, the pull persists for another try or leaves
			if randf() < 0.6:
				_status.text = "the hook did not set - it is still pulling"
			else:
				_anchor = 0.0
				_nibbles = 0
				_last_strike_t = -999.0
				_status.text = "the hook did not set - the pull is gone")


## Hook set: the reel begins. Duration scales with integration difficulty (a
## foreign creature takes longer to integrate) and the reward profile. During
## the reel the ADRENOCHROME anneals: each step the genome moves toward the
## party's attractors (high-acceptance members) and away from its repulsors,
## under cooling noise - integrating with you, and pulling against it.
func _begin_hook(d: float) -> void:
	_anchor = 0.0
	_nibbles = 0
	_last_strike_t = -999.0
	var profile_scale := 1.0
	match _profile:
		"anchor":
			profile_scale = 0.7
		"drift":
			profile_scale = 1.4
		"reel":
			profile_scale = 1.0      # knowledge now powers the reel globally
	var members: Array = []
	var accs: Array = []
	for e in _belt:
		accs.append(_acceptance(e))
	for i in _belt.size():
		var e: Dictionary = _belt[i]
		members.append({
			"traits": (e.traits as Dictionary).duplicate(),
			"g": _member_genome(e),
			"w": 0.1 + float(accs[i]),
			"sign": 1.0 if float(accs[i]) >= 1.0 else -1.0,
		})
	# the PRIOR is a gentle attractor - the anchor that keeps a voice ordinary.
	# Flagged, because the toll dissolves ITS grip specifically (see the consts)
	members.append({"traits": {}, "g": Voice.ProsodyWalk.PRIOR.duplicate(),
		"w": 0.4, "sign": 1.0, "prior": true})
	var rng := RandomNumberGenerator.new()
	rng.seed = hash("adrenochrome") ^ hash(str(_lineage))
	# the FIGHT is seeded per lineage: this creature always fights this way
	var frng := RandomNumberGenerator.new()
	frng.seed = hash("fight") ^ hash(str(_lineage))
	_hook = {
		"rng": rng, "step": 0, "t": 0.0,
		"progress": 0.0, "run": 0.0, "run_t": frng.randf_range(2.0, 6.0),
		"frng": frng, "power": _reel_power(),
		"duration": lerpf(45.0, 240.0, d) * profile_scale,
		"members": members, "d": d, "reward": _catch_reward(d),
		"traits": _traits.duplicate(),
		"genome": Voice.ProsodyWalk._lineage_genome(_lineage) \
			if _working_genome.is_empty() else _working_genome.duplicate(),
	}
	_retune_t = 0.0
	_hud_glyph.seed_hash = hash(str(_lineage))
	_status.text = "hook set - the fight begins"


## One anneal step: the frozen genome moves through the party's force field.
## Deterministic per (lineage, party snapshot). For KIN the noise cools toward
## the freeze, as it always did; for a foreign creature THE TOLL inverts that
## (see the TOLL_ constants) - the party shoves harder, the prior lets go, the
## noise heats, and the voice frays, all in proportion to how far in it has
## been dragged. The player hears every step of it: the reel retunes the live
## stream from these traits while it runs.
func _adreno_step() -> void:
	var h := _hook
	var rng: RandomNumberGenerator = h.rng
	var prog: float = float(h.step) / float(HOOK_STEPS)
	var cool: float = 1.0 - prog
	var toll: float = float(h.d) * prog
	var agitation: float = cool + TOLL_HEAT * toll
	for key in Voice.TRAIT_KEYS:
		var x: float = float(h.traits.get(key, 0.0))
		var force := 0.0
		for m in h.members:
			var w: float = float(m.w)
			if float(m.sign) < 0.0:
				w *= 1.0 + TOLL_REPEL * toll          # repulsions amplify
			elif bool(m.get("prior", false)):
				w *= 1.0 - TOLL_PRIOR_LOSS * toll     # the ordinary loses its grip
			force += float(m.sign) * w * (float((m.traits as Dictionary).get(key, 0.0)) - x)
		x += 0.02 * force + rng.randfn(0.0, 0.035 * agitation)
		if FRAY_KEYS.has(key):
			# a creature dragged home does not polish, it FRAYS - saturating
			# toward the rail rather than slamming into the clamp every step
			x += TOLL_FRAY * toll * toll * (1.0 - x)
		else:
			# every other axis flees the ordinary in whatever direction it
			# already leans: the voice becomes more itself, past comfort
			x += TOLL_EXTREME * toll * toll * signf(x)
		h.traits[key] = clampf(x, -1.0, 1.0)
	var span_mult: float = 1.5 + TOLL_GENOME_SPAN * toll
	for key in Voice.ProsodyWalk.PRIOR:
		var prior: float = float(Voice.ProsodyWalk.PRIOR[key])
		var span: float = maxf(absf(prior), 0.05)
		var x: float = float(h.genome.get(key, prior))
		var force := 0.0
		for m in h.members:
			var w: float = float(m.w)
			if float(m.sign) < 0.0:
				w *= 1.0 + TOLL_REPEL * toll
			elif bool(m.get("prior", false)):
				w *= 1.0 - TOLL_PRIOR_LOSS * toll
			force += float(m.sign) * w * (float((m.g as Dictionary).get(key, prior)) - x)
		x += 0.02 * force + rng.randfn(0.0, 0.04 * span * agitation)
		x = clampf(x, prior - span_mult * span, prior + span_mult * span)
		if key == "pace_hot" or key == "pace_calm":
			x = clampf(x, PACE_MIN, PACE_MAX)         # hideous, but still a voice
		h.genome[key] = x
	h.step += 1


## How badly this reel is deforming its catch, 0..1 - the toll made legible.
## Drives the status line and the HUD, so the cost is something the player can
## watch arrive rather than only discover at the freeze.
func _hook_toll() -> float:
	if _hook.is_empty():
		return 0.0
	return clampf(float(_hook.d) * float(_hook.progress), 0.0, 1.0)


## The freeze: the reel completes, the adrenochrome stops moving, and what it
## became is presented for judgment.
func _finish_hook() -> void:
	var h := _hook
	_hook = {}
	# the catch: contortion releases, and the show JUMPS to the scene this
	# seed owns - the reward, and part of the seed's genome forever
	Director.set_aura(0.0)
	Director.jump(hash(str(_lineage)))
	_pending = {"d": h.d, "reward": h.reward,
		"traits": h.traits, "genome": h.genome,
		"scene": Director.scene_title(hash(str(_lineage)))}
	var fit := _relative_fit_of(h.traits, _lineage, h.genome)
	_card_glyph.seed_hash = hash(str(_lineage))
	_card_glyph.fit = fit
	_card_glyph.queue_redraw()
	_card_label.text = "it has stopped changing · %s\n%s" % [
		_seed_name(_lineage), _fit_verdict(fit)]
	_card_label.add_theme_color_override("font_color", SeedGlyph.fit_color(fit))
	_accept_btn.visible = true
	_release_btn.visible = true
	_card.visible = true
	_status.text = "caught - hold or fold?"


## How close a belt member sits to the CURRENT candidate (cosine in the shared
## trait+genome space) - the HUD draws members as tiny planets only when this
## is genuinely high; the cosmos stays sparse.
func _member_closeness(e: Dictionary) -> float:
	return _cosine(
		_seed_vector(e.traits, e.lineage, _member_genome(e)),
		_seed_vector(_traits, _lineage, _working_genome))


## A seed's genome: the frozen adrenochrome if it carries one, else derived
## from the lineage as always.
func _member_genome(e: Dictionary) -> Dictionary:
	if e.has("genome") and not (e.genome as Dictionary).is_empty():
		return e.genome
	return Voice.ProsodyWalk._lineage_genome(e.lineage)


## Any candidate's fit (given explicit traits/genome, e.g. the mid-anneal
## adrenochrome), RELATIVE to the party's own spread: raw cosines cluster too
## tightly to see, so the scale stretches over what the bag actually contains.
func _relative_fit_of(traits: Dictionary, lineage: Array, genome: Dictionary = {}) -> float:
	if _belt.is_empty():
		return 0.6
	var party := _party_vector()
	var sims: Array = [_cosine(_seed_vector(traits, lineage, genome), party)]
	for e in _belt:
		sims.append(_cosine(_seed_vector(e.traits, e.lineage, _member_genome(e)), party))
	var norm := _relative_norm(sims)
	return norm[0]


func _fit_verdict(fit: float) -> String:
	if fit < 0.25:
		return "NOT a good fit for your party"
	if fit < 0.5:
		return "an uneasy fit - it would pull the party"
	if fit < 0.75:
		return "a workable fit"
	return "kin - it belongs with this party"


## Min-max stretch a list to [0,1]; a degenerate spread parks everyone at 0.6
## (no information = no alarm).
func _relative_norm(values: Array) -> Array:
	var lo := 1e9
	var hi := -1e9
	for v in values:
		lo = minf(lo, float(v))
		hi = maxf(hi, float(v))
	var out: Array = []
	if hi - lo < 0.05:
		for _v in values:
			out.append(0.6)
		return out
	for v in values:
		out.append((float(v) - lo) / (hi - lo))
	return out


## Accept the FROZEN creature: it joins the belt with its annealed traits and
## adrenochrome genome, and - the payoff - the working candidate BECOMES it:
## the stream restarts so you hear what you caught.
func _accept_catch() -> void:
	if _pending.is_empty():
		return
	var reward: float = _pending.reward
	for e in _belt:
		if _is_prefix(e.lineage, _lineage):
			e.m.catches += 1
			e.m.r += reward * 0.5
	_landed = true                   # the caught voice speaks at full presence
	_traits = (_pending.traits as Dictionary).duplicate()
	_working_genome = (_pending.genome as Dictionary).duplicate()
	_belt.append({
		"lineage": _lineage.duplicate(), "traits": _traits.duplicate(),
		"genome": _working_genome.duplicate(),
		"scene": _pending.get("scene", ""),
		"m": {"s": 0.0, "acts": 1, "restores": 0, "evolves": 0, "catches": 0,
			"r": reward, "d": _pending.d,
			"t": int(Time.get_unix_time_from_system())},
	})
	while _belt.size() > BELT_MAX:
		_belt.pop_front()
	_pending = {}
	_card.visible = false
	_rebuild_belt()                  # every member's colour re-attunes here
	_persist()
	_status.text = "accepted +%.1f - hear what you caught (belt %d/%d)" % [
		reward, _belt.size(), BELT_MAX]
	_apply()


func _release_catch() -> void:
	_pending = {}
	_card.visible = false
	# the reel's audition bent the stream's timbre toward the adrenochrome;
	# a fold hands the body back to the working candidate
	if _stream != null and is_instance_valid(_stream):
		_stream.retune(_current_spec())
	_status.text = "released - nothing changes"


func _restore_capture(idx: int) -> void:
	if idx < 0 or idx >= _belt.size():
		return
	var entry: Dictionary = _belt[idx]
	entry.m.restores += 1
	entry.m.acts += 1
	_cast = true
	_landed = true                   # a belt seed is already yours: full voice
	_lineage = (entry.lineage as Array).duplicate()
	_traits = (entry.traits as Dictionary).duplicate()
	_working_genome = (entry.get("genome", {}) as Dictionary).duplicate()
	_throw_ms = Time.get_ticks_msec()
	_sync_line_button()
	# the seed's scene is part of its identity: restoring it returns there
	if not String(entry.get("scene", "")).is_empty():
		Director.jump(hash(str(_lineage)))
	_update_reading_label()
	_persist()
	_apply()


func _release_capture(idx: int) -> void:
	if idx < 0 or idx >= _belt.size():
		return
	_belt.remove_at(idx)
	_rebuild_belt()
	_persist()


func _is_prefix(a: Array, b: Array) -> bool:
	return a.size() < b.size() and b.slice(0, a.size()) == a


func _seed_name(lineage: Array) -> String:
	return "g%d·%06x" % [lineage.size() - 1, hash(str(lineage)) & 0xFFFFFF]


func _update_reading_label() -> void:
	# before anything is thrown there IS no candidate: the persisted (or
	# default) working lineage is only a parent-to-be, and naming it - with a
	# difficulty word scored against an empty belt - put a phantom seed on a
	# line that has never touched the water
	if not _cast:
		_reading_label.text = "nothing on the line"
		if _hud_glyph != null and is_instance_valid(_hud_glyph):
			_hud_glyph.visible = false
		return
	_reading_label.text = "%s · %s catch" % [
		_seed_name(_lineage), _difficulty_word(_candidate_difficulty())]
	if _hud_glyph != null and is_instance_valid(_hud_glyph):
		_hud_glyph.visible = true
		_hud_glyph.seed_hash = hash(str(_lineage))
		_hud_glyph.fit = _relative_fit_of(_traits, _lineage, _working_genome)
		_hud_glyph.queue_redraw()


func _difficulty_word(d: float) -> String:
	if d < 0.25:
		return "easy"
	if d < 0.5:
		return "firm"
	if d < 0.75:
		return "hard"
	return "wild"


# ---- similarity: the party, the vectors, the colours -----------------------


## A seed as a vector: its traits plus its genome (as deltas off the PRIOR),
## the space the constellation colours attune in. An explicit genome (frozen
## or mid-anneal adrenochrome) takes precedence over lineage derivation.
func _seed_vector(traits: Dictionary, lineage: Array, genome: Dictionary = {}) -> PackedFloat32Array:
	var v := PackedFloat32Array()
	for key in Voice.TRAIT_KEYS:
		v.append(float(traits.get(key, 0.0)))
	var g: Dictionary = genome if not genome.is_empty() \
		else Voice.ProsodyWalk._lineage_genome(lineage)
	for key in Voice.ProsodyWalk.PRIOR:
		var prior: float = float(Voice.ProsodyWalk.PRIOR[key])
		v.append((float(g.get(key, prior)) - prior) / maxf(absf(prior), 0.001))
	return v


## The party's centre: acceptance-weighted mean of the belt's vectors.
func _party_vector() -> PackedFloat32Array:
	var dims := Voice.TRAIT_KEYS.size() + Voice.ProsodyWalk.PRIOR.size()
	var v := PackedFloat32Array()
	v.resize(dims)
	if _belt.is_empty():
		return v
	var total := 0.0
	for e in _belt:
		var w := 0.1 + _acceptance(e)
		total += w
		var sv := _seed_vector(e.traits, e.lineage)
		for i in dims:
			v[i] += sv[i] * w
	for i in dims:
		v[i] /= maxf(total, 0.001)
	return v


func _cosine(a: PackedFloat32Array, b: PackedFloat32Array) -> float:
	var dot := 0.0
	var na := 0.0
	var nb := 0.0
	for i in mini(a.size(), b.size()):
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	if na < 0.000001 or nb < 0.000001:
		return 0.0
	return dot / (sqrt(na) * sqrt(nb))


func _seed_value(e: Dictionary) -> float:
	return 20.0 * float(e.m.get("r", 0.0)) + e.m.s / 30.0


func _acceptance(e: Dictionary) -> float:
	var own: float = _seed_value(e) / maxf(1.0, float(e.m.acts))
	var total := 0.0
	var plays := 0
	for other in _belt:
		total += _seed_value(other)
		plays += int(other.m.acts)
	var mean := total / maxf(1.0, float(plays))
	return own / maxf(mean, 0.001)


func _rebuild_belt() -> void:
	for child in _belt_rows.get_children():
		child.queue_free()
	_inv_labels = []
	_inv_glyphs = []
	for i in _belt.size():
		var entry: Dictionary = _belt[i]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 6)
		var glyph := SeedGlyph.new()
		glyph.custom_minimum_size = Vector2(24, 24)
		glyph.seed_hash = hash(str(entry.lineage))
		row.add_child(glyph)
		_inv_glyphs.append(glyph)
		var slot := Button.new()
		slot.text = _seed_name(entry.lineage)
		slot.custom_minimum_size = Vector2(96, 26)
		var idx := i
		slot.pressed.connect(func(): _restore_capture(idx))
		row.add_child(slot)
		var metrics := Label.new()
		metrics.add_theme_font_size_override("font_size", 11)
		metrics.modulate = Color(1, 1, 1, 0.65)
		metrics.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		metrics.mouse_filter = Control.MOUSE_FILTER_PASS
		row.add_child(metrics)
		_inv_labels.append(metrics)
		var release := Button.new()
		release.text = "×"
		release.tooltip_text = "Release this seed"
		release.custom_minimum_size = Vector2(26, 26)
		release.pressed.connect(func(): _release_capture(idx))
		row.add_child(release)
		_belt_rows.add_child(row)
	_refresh_inventory()


## The bag keeps reflecting fit: each member's colour is its RELATIVE standing
## in the current party - similarity to the centre blended with its own
## acceptance record (the receptance) - restretched every refresh, so an
## Accept genuinely shifts every colour.
func _refresh_inventory() -> void:
	var party := _party_vector()
	var sims: Array = []
	var accs: Array = []
	for e in _belt:
		sims.append(_cosine(_seed_vector(e.traits, e.lineage), party))
		accs.append(_acceptance(e))
	var sim_n := _relative_norm(sims)
	var acc_n := _relative_norm(accs)
	for i in mini(_belt.size(), _inv_labels.size()):
		var e: Dictionary = _belt[i]
		var label: Label = _inv_labels[i]
		if not is_instance_valid(label):
			continue
		label.text = "%.0fs·%d× %.1fx" % [e.m.s, int(e.m.acts), _acceptance(e)]
		label.tooltip_text = _seed_tooltip(e)
		if i < _inv_glyphs.size() and is_instance_valid(_inv_glyphs[i]):
			var glyph: SeedGlyph = _inv_glyphs[i]
			glyph.fit = 0.55 * float(sim_n[i]) + 0.45 * float(acc_n[i])
			glyph.queue_redraw()


func _seed_tooltip(e: Dictionary) -> String:
	var g: Dictionary = _member_genome(e)
	var tv := ""
	for key in Voice.TRAIT_KEYS:
		var v := float((e.traits as Dictionary).get(key, 0.0))
		if absf(v) >= 0.15:
			tv += "%s %+.1f  " % [key, v]
	return ("lineage %s\nreward %.1f · held %.0fs over %d plays (%.1fx belt)\n"
		+ "caught at %s difficulty · kept %d · thrown-from %d · line kept %d\n"
		+ "opens %.2f hot, settles to %.2f (half-life %.0fs)\n"
		+ "breath every ~%.0f syllables · emphasis appetite %.2f\n"
		+ "pitch gravity %.2f · resonance ring %.2f · strike bar %.1f\n"
		+ "pace %.2f hot .. %.2f calm\n%s") % [
		str(e.lineage), float(e.m.get("r", 0.0)), e.m.s, int(e.m.acts), _acceptance(e),
		_difficulty_word(float(e.m.get("d", 0.35))),
		int(e.m.restores), int(e.m.evolves), int(e.m.catches),
		g.heat, g.baseline, 0.693 / maxf(g.settle, 0.001),
		g.breath_span, g.lean, g.gravity, g.ring, g.act_thr,
		g.pace_hot, g.pace_calm,
		(("voice: " + tv) if not tv.is_empty() else "voice: near default")
			+ (("\nhaunts " + str(e.scene)) if not str(e.get("scene", "")).is_empty() else "")]


# ---- the implicit loop: debounce -> persist + apply ------------------------


func _mark_structural() -> void:
	_dirty = true
	_restart_pending = true
	_last_edit_ms = Time.get_ticks_msec()


func _process(delta: float) -> void:
	if _dirty and Time.get_ticks_msec() - _last_edit_ms >= AUTOSAVE_DELAY_MS:
		_persist()
		if _restart_pending:
			_apply()
	if _stream != null and is_instance_valid(_stream) and _stream.words.size() > 0:
		for e in _belt:
			if e.lineage == _lineage or _is_prefix(e.lineage, _lineage):
				e.m.s += delta
		# PRESENCE: how near the voice is to the caster. A fresh cast swims
		# far (darker, a few dB down - clearly AUDIBLE; distance is the
		# filter's job, silence belongs only to the un-cast water); an
		# anchored pull brings it closer; the reel drags it in - and a RUN
		# drags it back out; landed = full voice.
		var presence := 1.0
		if not _landed:
			if _hook.is_empty():
				presence = 0.55 + 0.25 * clampf(_anchor, 0.0, 1.0)
			else:
				presence = clampf(0.6 + 0.4 * float(_hook.progress)
					- 0.2 * float(_hook.run), 0.3, 1.0)
		_stream.set_presence(presence)
		_metrics_t += delta
		if _metrics_t >= 5.0:
			_metrics_t = 0.0
			_refresh_inventory()
			_persist()
		# the latent water: strikes LATCH an anchor that pulls for a long
		# while (slow decay), and sometimes just leaves. Easy fishing - no
		# reflexes; the HUD shows the line bending for as long as it lasts.
		if _hook.is_empty():
			var strike := _fresh_strike()
			var s: float = strike.x
			if s > 0.0:
				_anchor = clampf(maxf(_anchor, 0.45 + 0.45 * minf(s, 1.2)), 0.0, 1.2)
				# a genuinely NEW strike (not just the same one still trailing)
				# is a nibble - sitting through several steadies the odds
				if strike.y > _last_strike_t + 0.01:
					_last_strike_t = strike.y
					_nibbles = mini(_nibbles + 1, NIBBLE_MAX)
			_anchor *= exp(-delta / ANCHOR_TAU)
			if _anchor < 0.05:
				_anchor = 0.0
				_nibbles = 0
				_last_strike_t = -999.0
			_vanish_t += delta
			if _vanish_t >= ANCHOR_CHECK:
				_vanish_t = 0.0
				# anchored searching holds its pull; drifting lines lose theirs
				if _anchor > 0.0 and randf() < float(PROFILES[_profile].vanish):
					_anchor = 0.0
					_nibbles = 0
					_last_strike_t = -999.0
			_status_t2 += delta
			if _status_t2 >= 0.5 and _pending.is_empty() and not _catching:
				_status_t2 = 0.0
				if _anchor > 0.1:
					_status.text = "something is pulling (%d nibble%s felt) - set the hook when ready" % [
						_nibbles, "" if _nibbles == 1 else "s"]
				elif _status.text.begins_with("something is pulling"):
					_status.text = "the line is slack"
		else:
			# THE REEL is a FIGHT, not a timer: progress is the belt's power
			# against the creature's RUNS. A run pays line back out (progress
			# can regress; the voice drops away mid-pull); knowledge reels
			# harder. Everything hooked lands eventually - the wait is the
			# accumulation (the adrenochrome anneals step by step) AND the
			# audition (presence and timbre close in as it nears), so the
			# hold-or-fold decision is informed by the time it surfaces.
			var h := _hook
			h.t += delta
			h.run = maxf(float(h.run) - delta / 2.2, 0.0)
			h.run_t = float(h.run_t) - delta
			if float(h.run_t) <= 0.0:
				var frng: RandomNumberGenerator = h.frng
				h.run_t = frng.randf_range(4.0, 12.0) / (0.4 + float(h.d))
				h.run = frng.randf_range(0.5, 1.0) * (0.35 + 0.65 * float(h.d))
			var rate: float = float(h.power) / float(h.duration) * (1.0 - 1.3 * float(h.run))
			h.progress = clampf(float(h.progress) + rate * delta, 0.0, 1.0)
			while h.step < int(float(h.progress) * HOOK_STEPS):
				_adreno_step()
			# the audition: every couple of seconds the stream's TIMBRE bends
			# toward the annealing adrenochrome (atomic retune; the reading
			# stays the thrown plan until an Accept restarts it in full)
			_retune_t += delta
			if _retune_t >= 2.0:
				_retune_t = 0.0
				_stream.retune(Voice.Spec.from_traits(h.traits, int(_lineage[0]), _lineage))
			# the metamorphosis: pulling the fish closer contorts the CURRENT
			# scene, and a foreign catch contorts it BIG
			Director.set_aura(float(h.progress) * (0.4 + 1.1 * float(h.d)))
			_status_t2 += delta
			if _status_t2 >= 0.5:
				_status_t2 = 0.0
				var pc := int(float(h.progress) * 100.0)
				var toll := _hook_toll()
				if h.run > 0.35:
					_status.text = "it runs - the line pays out (%d%%)" % pc
				elif toll > 0.55:
					_status.text = "reeling - its voice is coming apart (%d%%)" % pc
				elif toll > 0.25:
					_status.text = "reeling - something in it is bending (%d%%)" % pc
				else:
					_status.text = "reeling - it changes as it comes (%d%%)" % pc
				_hud_glyph.fit = _relative_fit_of(h.traits, _lineage, h.genome)
				_hud_glyph.queue_redraw()
			if float(h.progress) >= 1.0:
				_finish_hook()
		_hud.queue_redraw()


func _exit_tree() -> void:
	if _dirty:
		_persist()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST and _dirty:
		_persist()


func _persist() -> void:
	_dirty = false
	var cfg := ConfigFile.new()
	cfg.load(CFG)
	cfg.set_value("synth", "text", _text.text)
	cfg.set_value("synth", "traits", _traits)
	cfg.set_value("synth", "lineage", _lineage)
	cfg.set_value("synth", "belt", _belt)
	cfg.set_value("synth", "profile", _profile)
	cfg.set_value("synth", "adreno", _working_genome)
	cfg.save(CFG)


func _load_persisted() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(CFG) == OK:
		_text.text = cfg.get_value("synth", "text", "")
		_traits = cfg.get_value("synth", "traits", {})
		_lineage = cfg.get_value("synth", "lineage", [1])
		_belt = cfg.get_value("synth", "belt", [])
		_profile = str(cfg.get_value("synth", "profile", "anchor"))
		# older saves used the reward-flavour names; map them onto the modes
		if _profile == "snap":
			_profile = "anchor"
		elif _profile == "hunt":
			_profile = "reel"
		if not PROFILES.has(_profile):
			_profile = "anchor"
		_working_genome = cfg.get_value("synth", "adreno", {})
		_sync_switchboard()
		for e in _belt:
			if not e.has("m"):
				e.m = {"s": 0.0, "acts": 0, "restores": 0, "evolves": 0,
					"catches": 0, "t": 0}
			if not e.m.has("r"):
				e.m.r = 0.0
				e.m.d = 0.35
			e.erase("active")
	if _traits.is_empty():
		_traits = _background_traits()
	_update_reading_label()
	_rebuild_belt()


## The population-average background: the voice before any seed exists.
func _background_traits() -> Dictionary:
	var t := {}
	if _belt.is_empty():
		return t
	var total := 0.0
	for e in _belt:
		total += 0.1 + _acceptance(e)
	for key in Voice.TRAIT_KEYS:
		var v := 0.0
		for e in _belt:
			v += float((e.traits as Dictionary).get(key, 0.0)) * (0.1 + _acceptance(e))
		t[key] = v / maxf(total, 0.001)
	return t


func _apply() -> void:
	_restart_pending = false
	if not _cast:
		# edits before the first throw only shape the draft - the water stays
		# silent until a voice is actually cast into it
		_status.text = "the water is silent - throw to cast a voice"
		return
	var text := _text.text.strip_edges()
	if text.is_empty():
		_status.text = "write something, then throw"
		return
	_loop_len = 0.0
	var spec := _current_spec()
	if _stream != null and is_instance_valid(_stream):
		_stream.restart(text, spec)
		return
	var stream: VoiceStream = preload("res://scripts/voice_stream.gd").new()
	stream.setup(text, spec, "user://synth/take_%06x" % (hash(str(_lineage)) & 0xFFFFFF))
	stream.completed.connect(func(dur: float, _wav: String):
		_loop_len = dur)
	_stream = stream
	if begin_stream.is_valid():
		begin_stream.call(stream)


## The catch animation, anchored beside the button that was pressed: an orb in
## the seed's hue closes on the voice, wobbles (each a contest), then settles
## with a ring (caught) or bursts into shards (slipped away). The roll is
## decided before the orb appears; the orb TELLS you.
class CatchOrb:
	extends Control
	signal finished
	var anchor := Vector2.ZERO
	var hue := 0.5
	var wobbles := 3
	var success := true
	var _t := 0.0

	const CLOSE := 0.35
	const WOBBLE := 0.38
	const RESULT := 0.55

	func _ready() -> void:
		set_anchors_preset(Control.PRESET_FULL_RECT)
		mouse_filter = Control.MOUSE_FILTER_IGNORE

	func _total() -> float:
		return CLOSE + wobbles * WOBBLE + RESULT

	func _process(delta: float) -> void:
		_t += delta
		queue_redraw()
		if _t >= _total():
			finished.emit()

	func _draw() -> void:
		var centre := anchor if anchor != Vector2.ZERO \
			else get_viewport_rect().size * 0.5
		var body := Color.from_hsv(hue, 0.6, 0.95)
		var rim := Color(1, 1, 1, 0.9)
		if _t < CLOSE:
			var u := _t / CLOSE
			var r := lerpf(56.0, 18.0, u * u)
			draw_arc(centre, r, 0.0, TAU, 40, Color(body, u * 0.9), 3.0)
			draw_circle(centre, 18.0 * u, Color(body, u * 0.55))
			return
		var tw := _t - CLOSE
		if tw < wobbles * WOBBLE:
			var k := int(tw / WOBBLE)
			var u := fmod(tw, WOBBLE) / WOBBLE
			var rock: float = sin(u * TAU * 1.5) * (1.0 - u) * 8.0 * (1.0 + 0.3 * k)
			var pos := centre + Vector2(rock, -absf(rock) * 0.3)
			draw_circle(pos, 18.0, Color(body, 0.85))
			draw_arc(pos, 18.0, 0.0, TAU, 32, rim, 2.0)
			draw_circle(pos + Vector2(-5, -6), 3.5, Color(1, 1, 1, 0.5))
			return
		var u := (_t - CLOSE - wobbles * WOBBLE) / RESULT
		if success:
			draw_circle(centre, 18.0, Color(body, 0.85 * (1.0 - u * 0.4)))
			draw_arc(centre, 18.0 + u * 34.0, 0.0, TAU, 40,
				Color(1, 1, 1, 0.7 * (1.0 - u)), 2.5)
		else:
			for i in 7:
				var a := TAU * float(i) / 7.0 + hue * TAU
				var p := centre + Vector2(cos(a), sin(a)) * (9.0 + u * 52.0)
				draw_circle(p, 3.5 * (1.0 - u), Color(body, 0.8 * (1.0 - u)))


## The HUD: the water, drawn like an LCD readout. The candidate's
## constellation lives at the left; the LINE runs from it across the panel -
## slack and faint when nothing pulls, bending and throbbing downward while
## something is anchored (for as long as it pulls - minutes, not moments);
## during the reel a progress arc winds around the constellation and the
## countdown ticks in the corner while the adrenochrome forms.
class Hud:
	extends Control
	var editor: SynthEditor

	func _draw() -> void:
		if editor == null:
			return
		var s := size
		var glyph_c := Vector2(34.0, s.y * 0.5)
		var t := Time.get_ticks_msec() / 1000.0
		var line_col := Color(0.4, 0.85, 0.65, 0.35)
		var y0: float = s.y * 0.5
		if not editor._hook.is_empty():
			# the reel: progress arc + the FIGHT. Percent, not a countdown -
			# the duration is not fixed anymore; a run pays line back out
			var progress: float = clampf(float(editor._hook.progress), 0.0, 1.0)
			var run: float = float(editor._hook.get("run", 0.0))
			# the toll bleeds the readout: an easy catch reels in clean amber,
			# a foreign one sickens toward violet as it is dragged in - the
			# cost is visible while it accrues, not only at the freeze
			var toll: float = editor._hook_toll()
			var arc_col := Color(1.0, 0.75, 0.3, 0.9).lerp(
				Color(0.75, 0.25, 0.95, 0.95), toll)
			draw_arc(glyph_c, 30.0, -PI / 2.0, -PI / 2.0 + TAU * progress, 48,
				arc_col, 2.5 + 1.5 * toll)
			draw_string(get_theme_default_font(), Vector2(s.x - 52.0, s.y - 10.0),
				"%d%%" % int(progress * 100.0), HORIZONTAL_ALIGNMENT_LEFT,
				-1, 12, Color(1.0, 0.75, 0.3, 0.9))
			# tension: taut thrum while reeling; a RUN whips the line hard
			# and shifts it toward red - the push and pull, visible
			var pts := PackedVector2Array()
			for i in 25:
				var u := float(i) / 24.0
				var x := lerpf(64.0, s.x - 12.0, u)
				# the toll adds a second, faster tremor riding the taut line
				pts.append(Vector2(x, y0 + sin(u * 14.0 + t * (9.0 + 14.0 * run))
					* (2.0 + 9.0 * run)
					+ sin(u * 37.0 - t * 21.0) * 3.5 * toll))
			draw_polyline(pts, arc_col.lerp(
				Color(1.0, lerpf(0.75, 0.4, run), 0.3, 0.5 + 0.3 * run), 0.5), 1.0)
			_draw_party_planets(s)   # the attractors, visible while they pull
			return
		# the line: slack when quiet, pulled into a deepening belly while
		# something is anchored - throbbing slowly, the easy-fishing signal.
		# The catch's DIFFICULTY is animated into the same line: a hard catch
		# trembles rough and shifts toward red, an easy one stays calm and
		# green - so a failed pull reads as "it was always going to be hard",
		# not an unexplained coin flip.
		var pull: float = clampf(editor._anchor, 0.0, 1.2)
		var d: float = editor._candidate_difficulty() if pull > 0.05 else 0.0
		var belly: float = pull * (s.y * 0.3) * (1.0 + 0.15 * sin(t * 2.2))
		var pts := PackedVector2Array()
		for i in 25:
			var u := float(i) / 24.0
			var x := lerpf(64.0, s.x - 12.0, u)
			var jitter: float = sin(u * 41.0 + t * 23.0) * d * 5.0 * pull
			pts.append(Vector2(x, y0 + sin(u * PI) * belly + jitter))
		var c := line_col if pull <= 0.05 \
			else Color.from_hsv(lerpf(0.42, 0.0, clampf(d, 0.0, 1.0)), 0.85, 0.9,
				0.5 + 0.4 * minf(pull, 1.0))
		draw_polyline(pts, c, 1.5)
		# the CAST POINT (the green dot) respects the mode: anchored it holds
		# mid-water; drifting it recedes with the cage as the wire's scale
		# grows (the caption says in what units); reel mode sits close in
		var cast_u := 0.5
		match editor._profile:
			"drift":
				cast_u = lerpf(0.45, 0.94, editor._drift_norm())
			"reel":
				cast_u = 0.35
		var cp := Vector2(lerpf(64.0, s.x - 12.0, cast_u),
			y0 + sin(cast_u * PI) * belly)
		var cast_r: float = lerpf(3.2, 1.6, editor._drift_norm())   # far = small
		draw_circle(cp, cast_r + (1.2 * sin(t * 2.2) if pull > 0.05 else 0.0),
			Color(0.5, 0.95, 0.8, 0.75))
		if editor._profile == "drift":
			draw_string(get_theme_default_font(), Vector2(66.0, s.y - 8.0),
				editor._drift_caption(), HORIZONTAL_ALIGNMENT_LEFT, -1, 11,
				Color(0.4, 0.85, 0.65, 0.7))
		if pull > 0.05:
			# nibbles felt so far - brightening ticks above the line; fill up
			# toward NIBBLE_MAX as patience accumulates
			for i in editor._nibbles:
				var nx := lerpf(70.0, s.x - 20.0, float(i) / float(SynthEditor.NIBBLE_MAX - 1))
				draw_circle(Vector2(nx, y0 - 14.0), 2.0, Color(1.0, 0.9, 0.5, 0.75))
		_draw_party_planets(s)

	## The party's influence in the cosmos - SPARSELY: a member appears as a
	## tiny planet only when it is genuinely close to the current candidate,
	## drifting nearer the constellation the closer it sits. Never forced:
	## a distant party leaves an empty sky.
	func _draw_party_planets(s: Vector2) -> void:
		var threshold: float = editor._planet_threshold()
		for i in editor._belt.size():
			var e: Dictionary = editor._belt[i]
			var closeness: float = editor._member_closeness(e)
			if closeness < threshold:
				continue
			var u: float = clampf((1.0 - closeness) / maxf(1.0 - threshold, 0.05), 0.06, 1.0)
			var rng := RandomNumberGenerator.new()
			rng.seed = hash(str(e.lineage)) ^ 77
			var pos := Vector2(
				lerpf(74.0, s.x * 0.8, u),
				s.y * 0.5 + rng.randf_range(-0.3, 0.3) * s.y)
			var r: float = 2.0 + 2.0 * clampf((closeness - 0.72) / 0.28, 0.0, 1.0)
			var col := Color.WHITE
			if i < editor._inv_glyphs.size() and is_instance_valid(editor._inv_glyphs[i]):
				col = SeedGlyph.fit_color(editor._inv_glyphs[i].fit)
			draw_circle(pos, r, Color(col, 0.8))
			# the ring that makes it a planet, tilted by its own seed
			var tilt := rng.randf_range(0.2, 0.9)
			draw_arc(pos, r + 1.6, tilt, tilt + PI * 1.1, 12, Color(col, 0.45), 1.0)


## A seed drawn as a constellation: the LINES are seeded random (the seed's
## own fingerprint); the COLOUR is the fit wheel - a relative scalar projected
## red (a poor fit) through amber to green (kin), loud on purpose: a colour
## ring frames the glyph, and a poor fit FRAZZLES the constellation itself
## (scattered points, jagged lines) while kin draws calm and tight. When the
## party changes, every colour shifts; how, you learn by doing it.
class SeedGlyph:
	extends Control
	var seed_hash := 0
	var fit := 0.6                   # 0 poor .. 1 kin, RELATIVE within the bag

	static func fit_color(f: float) -> Color:
		return Color.from_hsv(lerpf(0.0, 0.42, clampf(f, 0.0, 1.0)), 0.9, 1.0)

	func _draw() -> void:
		var rng := RandomNumberGenerator.new()
		rng.seed = seed_hash
		var s := size
		var col := fit_color(fit)
		var frazzle: float = (1.0 - clampf(fit, 0.0, 1.0)) * minf(s.x, s.y) * 0.14
		var n := 5 + rng.randi() % 4
		var pts: Array = []
		for _i in n:
			var p := Vector2(
				rng.randf_range(s.x * 0.18, s.x * 0.82),
				rng.randf_range(s.y * 0.18, s.y * 0.82))
			# a poor fit scatters its stars off their seats
			p += Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)) * frazzle
			pts.append(p)
		for i in range(1, n):
			var j := rng.randi_range(0, i - 1)
			draw_line(pts[i], pts[j], Color(col, 0.55), 1.0)
		for p in pts:
			draw_circle(p, 2.0, col)
		# the ring: the fit colour, unmissable even at 24 px
		draw_arc(s * 0.5, minf(s.x, s.y) * 0.5 - 1.5, 0.0, TAU, 32, Color(col, 0.9), 2.0)
