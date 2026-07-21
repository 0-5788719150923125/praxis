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
const BELT_MAX := 23                # the Collection ceiling; oldest falls off past this

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

# THE DRIFT as a JOURNEY, not a clock. The old drift was pure wall-time, so the
# cast dot crept a fixed sliver over five minutes and nothing on the wire ever
# read as MOTION. Now the line has a velocity: it accelerates from a crawl to a
# raw maximum the longer you leave the cast out, and an ODOMETER (_drift_dist,
# 0..1) accumulates that velocity - the streaks, the cast dot, and the distance
# caption all read the odometer, so drift finally looks like travel.
#
# Raw velocity alone can only ever inch you toward the deep field - the far
# reaches (au, lightyears) sit past where patience can carry you in a sitting.
# That is what the WARP is for, and it is your idea exactly: hold the raw
# maximum for WARP_CHARGE seconds and the drive charges; once charged, if you
# have banked enough ADRENOCHROME, the wire jumps FASTER than raw velocity could
# ever go, spending the reserve as it burns. Explore locally to fill the tank,
# then spend it to reach somewhere you could not otherwise get to in time.
const DRIFT_V0 := 1.0 / 600.0       # odometer/sec at the start of a drift (a crawl)
const DRIFT_VMAX := 1.0 / 180.0     # raw maximum odometer/sec - deliberately slow: the deep field is not free
const DRIFT_ACCEL := (1.0 / 180.0 - 1.0 / 600.0) / 22.0   # ramp crawl -> raw max over ~22s of unbroken drift
const WARP_CHARGE := 10.0           # seconds held at raw max before the drive charges
const WARP_VEL := 1.0 / 16.0        # warp odometer/sec - the streaks that "move faster than velocity allows"
const REEL_DECAY := 0.45            # reel retrieval = EXPONENTIAL decay of the odometer (scaled by _reel_power) - unwinds from ANY distance in log-time, since the odometer is now unbounded
# A catch hooked at the far reaches is on a longer line, so the fight is longer:
# the reel duration stretches with how far the cage had drifted at the moment the
# hook set (0 = home, 1 = the deep field). Bounded - the deep field costs more to
# haul home, but never unboundedly. The old reel ignored distance entirely, so a
# lightyears-out catch reeled in as fast as one at your feet.
const REEL_DIST_STRETCH := 2.6      # max duration multiple, at the far end of the drift
const WARP_BURN := 0.32             # adrenochrome spent per second of warp
const WARP_MIN := 0.4               # reserve needed to IGNITE a warp (a floored charge)
# Adrenochrome is the currency the warp spends. It is minted by fishing: a
# landed catch banks its reward, and patient nibbling trickles a little in. It
# persists across runs with the belt, so a session's local exploration funds the
# next session's jumps.
const RESERVE_TRICKLE := 0.03       # adrenochrome/sec while a pull is anchored (patience pays)
const RESERVE_MAX := 40.0           # a tank ceiling so it cannot grow unbounded

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
var _coll_title: Label              # "Collection  N/23" - the count lives here
var _inv_labels: Array = []         # metrics Label per inventory row
var _inv_glyphs: Array = []         # SeedGlyph per inventory row
var _inv_slots: Array = []          # the name Button per row - the SLOT you click to cast
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
var _line_gen := 0                  # bumped on release: in-flight animations check it
var _slotted_lineage: Array = []    # the seed currently cast FROM - re-clicking it releases; clicking another swaps bait
# The drift journey (see the DRIFT_ constants). All reset by _drift_reset on a
# throw / release / mode change - a new cast starts near, at a crawl.
var _drift_dist := 0.0              # the odometer: 0..1, what every drift visual reads
var _drift_vel := 0.0              # current odometer/sec, ramping V0 -> VMAX (-> WARP_VEL)
var _drift_atmax := 0.0            # seconds held at raw max, charging the warp
var _warping := false              # the drive is lit and burning reserve right now
var _reserve := 0.0               # banked adrenochrome - fishing mints it, the warp spends it
# Autopilot: the game plays itself over a fixed take, for the UI-recorded export
# (see main.gd's --synth-autopilot and exporter.gd). Random transitions on
# timers; no live audio is generated (the take is the audio), and nothing is
# persisted (a render must never mutate the player's real save).
var _autopilot := false
var _ap_t := 0.0                    # countdown to the autopilot's next decision
var _belt_shadow_t := 0.0           # throttle for re-orienting belt moons to the cast


func _ready() -> void:
	layer = 10
	_build_panel()
	_load_persisted()
	# the Collection is never empty: force a seed in if there is nothing to slot
	_ensure_collection()
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
	# UI-recorded export: the panel is visible and the game plays itself over the
	# take (see _autopilot_tick). Loaded belt/reserve give it seeds to work with.
	if args.has("--synth-autopilot"):
		_autopilot = true
		_ap_t = 1.5


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
			# NO reset: the position holds across modes - anchor stops where you
			# drifted to, reel walks it back (see _advance_drift)
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

	# --- The loop: Pull ---
	# There is no Throw button anymore: casting a line is done by SLOTTING a seed
	# from the Collection (click a row to cast from it, click again to release -
	# see _on_seed_clicked). Pull stays here to set the hook once something bites.
	var loop_row := HBoxContainer.new()
	loop_row.add_theme_constant_override("separation", 8)
	box.add_child(loop_row)
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
	_accept_btn.text = "Hold"
	_accept_btn.tooltip_text = "Hold on to it - fold it into your Collection; every seed's colour re-attunes"
	_accept_btn.pressed.connect(_accept_catch)
	card_btns.add_child(_accept_btn)
	_release_btn = Button.new()
	# "Fold", not "Release": the line has its own Release now, and this is the
	# other decision entirely - hold or fold the catch you are being shown
	_release_btn.text = "Fold"
	_release_btn.tooltip_text = "Let it go - nothing changes"
	_release_btn.pressed.connect(_release_catch)
	card_btns.add_child(_release_btn)

	# --- The Collection: your seeds, ALWAYS last. Click a seed to cast from it
	# (slot it in), click it again to release. Count shown by the title; the list
	# scrolls once it outgrows its box, so nothing is hidden. ---
	_coll_title = Label.new()
	_coll_title.text = "Collection"
	_coll_title.add_theme_font_size_override("font_size", 13)
	_coll_title.add_theme_color_override("font_color", Color(0.55, 0.95, 0.75, 0.85))
	box.add_child(_coll_title)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(0, 300)      # ~10 rows tall, then scroll
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	box.add_child(scroll)
	_belt_rows = VBoxContainer.new()
	_belt_rows.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_belt_rows.add_theme_constant_override("separation", 4)
	scroll.add_child(_belt_rows)


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
	# capture the CURRENT anchor's reception (read editor state on the main thread):
	# the export bakes the same location colour we're hearing here, so a take caught
	# in a rich region of the drift curve exports sounding like that region
	var loc_prox := _beacon_prox()
	var loc_freq := _location_freq()
	var done := {}
	var task := WorkerThreadPool.add_task(func():
		var result := Voice.render(text, spec)
		result.pcm = VoiceStream.bake_location(result.pcm, loc_prox, loc_freq)
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


## SLOTTING: click a Collection seed to cast a line from it (a fresh candidate
## parented on that seed, spoken immediately); click the seed you are already
## fishing (or any of its ancestors on the line) to RELEASE back to still water.
## One line at a time - clicking a different seed switches the slot to it.
func _on_seed_clicked(idx: int) -> void:
	if idx < 0 or idx >= _belt.size():
		return
	# release ONLY when re-clicking the exact seed that's currently cast from (the
	# bait that's out). Clicking any OTHER seed - including an ancestor on the same
	# lit lineage - just SWAPS the bait: a new cast from that seed at the SAME
	# position, no return to still water.
	if _cast and _belt[idx].lineage == _slotted_lineage:
		_release_line()
	else:
		_throw_from(idx)


## Keep the Pull button and the slot highlights honest about the current line.
## (Kept the old name so every existing call site still routes here.)
func _sync_line_button() -> void:
	if _catch_btn != null and is_instance_valid(_catch_btn):
		# nothing on the water can be pulled - a pull there would roll the odds
		# against a candidate that isn't swimming yet
		_catch_btn.disabled = not _cast
	_update_slot_highlights()


## Colour the cast line's lineage as a GRADIENT by generation: the most recent
## seed on the line burns full colour, and each older ancestor fades toward white,
## so the family chain reads as a depth ramp instead of a flat block of green.
func _update_slot_highlights() -> void:
	# first pass: the span of lineage depths that are lit right now
	var mn := 1 << 30
	var mx := 0
	for i in mini(_belt.size(), _inv_slots.size()):
		var lin: Array = _belt[i].lineage
		if _cast and (lin == _lineage or _is_prefix(lin, _lineage)):
			mn = mini(mn, lin.size())
			mx = maxi(mx, lin.size())
	for i in mini(_belt.size(), _inv_slots.size()):
		var b = _inv_slots[i]
		if not is_instance_valid(b):
			continue
		var lin: Array = _belt[i].lineage
		var active: bool = _cast and (lin == _lineage or _is_prefix(lin, _lineage))
		if active:
			# recent (deeper) = full colour; distant ancestor (shallower) = near white
			var t: float = 1.0 if mx == mn else float(lin.size() - mn) / float(mx - mn)
			var col := Color(1, 1, 1).lerp(Color(0.4, 1.0, 0.72), lerpf(0.12, 1.0, t))
			b.add_theme_color_override("font_color", col)
			b.add_theme_color_override("font_hover_color", col.lightened(0.15))
		else:
			b.remove_theme_color_override("font_color")
			b.remove_theme_color_override("font_hover_color")


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
	_slotted_lineage = []            # nothing is baited now
	_drift_reset()                   # the journey ends with the line
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
	# no drift reset: a throw casts from wherever the line already sits (you fish
	# variations at your current spot); only a release returns home
	_sync_line_button()
	_update_reading_label()
	_persist()
	_apply()


## Cast a line from a SPECIFIC Collection seed (the slotting click): a fresh
## candidate parented on that seed, jittered by the profile, spoken at once. The
## deliberate-parent sibling of _throw's weighted/wild pick - the child gets its
## own lineage, so the reel/catch/accept loop grows the Collection from here.
func _throw_from(idx: int) -> void:
	if idx < 0 or idx >= _belt.size():
		return
	if not _pending.is_empty():
		_release_catch()
	if not _hook.is_empty():
		_hook = {}
		Director.set_aura(0.0)
		_card.visible = false
	_anchor = 0.0
	_nibbles = 0
	_last_strike_t = -999.0
	_cast = true
	_landed = false
	var parent: Dictionary = _belt[idx]
	parent.m.evolves += 1
	var prof: Dictionary = PROFILES[_profile]
	var drift := _drift_norm()
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
	_working_genome = (parent.get("genome", {}) as Dictionary).duplicate()
	_slotted_lineage = (parent.lineage as Array).duplicate()   # this seed is the bait now
	_status.text = "cast from %s" % _seed_name(parent.lineage)
	_throw_ms = Time.get_ticks_msec()
	# no drift reset - fish from wherever the line currently sits
	_sync_line_button()
	_update_reading_label()
	_persist()
	_apply()


## The Collection is never empty: if there is nothing to slot, force one seed in
## (a wild, tempered voice) so the player always has a line to cast. Called at
## boot and after the last seed is deleted.
func _ensure_collection() -> void:
	if not _belt.is_empty():
		return
	var lin: Array = [randi() % 1000000]
	var rng := RandomNumberGenerator.new()
	rng.seed = int(lin[0])
	var traits: Dictionary = _temper_traits(Voice.Spec.sample(rng).traits, 0.0)
	_belt.append({
		"lineage": lin, "traits": traits, "genome": {}, "scene": "",
		"m": {"s": 0.0, "acts": 1, "restores": 0, "evolves": 0, "catches": 0,
			"r": 0.5, "d": 0.35, "t": int(Time.get_unix_time_from_system())},
	})
	_rebuild_belt()
	_persist()


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
			# earns by DISTANCE now, not raw seconds: a catch made out in the
			# deep field (reached by warping, which spent the reserve to get
			# there) is worth the most - the reserve you burn comes back through
			# the foreign catches only the deep field holds
			return 0.4 + 2.4 * _drift_norm()
		"reel":
			return 0.4 + 2.0 * difficulty
		_:
			return 2.4 * 8.0 / (8.0 + t_s)


## WHERE THE LINE IS (0..1) - a persistent LOCATION now, not a per-cast timer.
## The odometer holds across mode changes and throws; only a RELEASE (still
## water) returns it home. Drift travels out, anchor freezes here, reel retrieves
## toward home - so the distance you earned is never lost by switching modes.
## Clamped [0,1] - for the VISUAL layout (the line has finite length) and the
## bounded gameplay quantities (reward, the reception band). The DISTANCE itself
## is uncapped: read it from _drift_reach.
func _drift_norm() -> float:
	return clampf(_drift_dist, 0.0, 1.0)


## The raw, UNBOUNDED odometer - how far the warp has actually carried the line.
## Only the distance caption reads this; everything visual/gameplay uses the
## clamped _drift_norm.
func _drift_reach() -> float:
	return maxf(_drift_dist, 0.0)


## Reset the line all the way home - ONLY on release (still water). Mode changes
## and throws no longer reset: anchoring stops where you are, reeling walks back.
func _drift_reset() -> void:
	_drift_dist = 0.0
	_drift_vel = DRIFT_V0
	_drift_atmax = 0.0
	_warping = false


## One frame of travel, per mode - the position PERSISTS, only the motion differs:
##   DRIFT  - accelerate outward; hold raw max WARP_CHARGE s to charge the warp,
##            which surges the odometer while it burns the reserve.
##   ANCHOR - stop dead where you are (velocity 0, warp off); the position holds,
##            so you fish under the conditions of wherever you drifted to.
##   REEL   - retrieve: walk the line back toward home, faster with a seasoned
##            belt (_reel_power) - the mirror of drifting out.
func _advance_drift(delta: float) -> void:
	if not _cast:
		return
	match _profile:
		"drift":
			_drift_vel = minf(_drift_vel + DRIFT_ACCEL * delta, DRIFT_VMAX)
			var at_max: bool = _drift_vel >= DRIFT_VMAX - 0.000001
			if at_max:
				_drift_atmax += delta
			else:
				_drift_atmax = 0.0
				_warping = false
			var charged: bool = _drift_atmax >= WARP_CHARGE
			if _warping:
				if _reserve <= 0.0 or not charged:
					_warping = false
			elif charged and _reserve >= WARP_MIN:
				_warping = true
			var vel := _drift_vel
			if _warping:
				vel = WARP_VEL
				_reserve = maxf(_reserve - WARP_BURN * delta, 0.0)
			# NO upper cap: the odometer runs as far as it's driven. Distance is
			# EXPONENTIAL in the odometer (see _drift_caption), so a constant warp
			# reaches any distance in log-time - there is no 0.33ly ceiling, the
			# whole point of the warp.
			_drift_dist = maxf(_drift_dist + vel * delta, 0.0)
		"anchor":
			# stop dead - hold this spot, drop any charge
			_drift_vel = 0.0
			_drift_atmax = 0.0
			_warping = false
		"reel":
			_drift_vel = 0.0
			_drift_atmax = 0.0
			_warping = false
			# exponential retrieval, so even a deep-warp distance unwinds in a
			# sensible ~log-time; snap the last sliver to home
			_drift_dist *= exp(-REEL_DECAY * _reel_power() * delta)
			if _drift_dist < 0.002:
				_drift_dist = 0.0


## The line's length, in honest units - UNCAPPED: it keeps climbing past a
## lightyear (kly, Mly, Gly, then scientific) for as long as the warp drives the
## odometer. The exponent is clamped only to keep the float finite.
func _drift_caption() -> String:
	var meters := pow(10.0, minf(1.0 + 14.5 * _drift_reach(), 300.0))
	if meters < 1000.0:
		return "≈ %d m" % int(meters)
	if meters < 1.0e7:
		return "≈ %.1f km" % (meters / 1000.0)
	if meters < 1.0e10:
		return "≈ %.0f mi" % (meters / 1609.34)
	if meters < 1.0e14:
		return "≈ %.2f au" % (meters / 1.496e11)
	var ly := meters / 9.461e15
	if ly < 1.0e3:
		return "≈ %.2f ly" % ly
	if ly < 1.0e6:
		return "≈ %.1f kly" % (ly / 1.0e3)
	if ly < 1.0e9:
		return "≈ %.1f Mly" % (ly / 1.0e6)
	if ly < 1.0e12:
		return "≈ %.2f Gly" % (ly / 1.0e9)
	# GDScript's % has no scientific specifier - build "M.Me+E" by hand
	var e10 := int(floor(log(ly) / log(10.0)))
	return "≈ %.1fe%d ly" % [ly / pow(10.0, float(e10)), e10]


## How near the line sits to the BEACON (the source at the line's end), 0..1 -
## the same geometry the HUD draws (beacon at 0.8 of the travel, half-width 0.45),
## so the audio effect and the visual approach agree. Peaks as you arrive on the
## source and falls again once you overshoot past it.
func _beacon_prox() -> float:
	var pos_u := lerpf(0.14, 1.0, _drift_norm())
	return clampf(1.0 - absf(pos_u - 0.8) / 0.45, 0.0, 1.0)


## The source's frequency (Hz): a resonant band that SWEEPS UP with distance (so
## different places sound different) plus a per-cast offset from the lineage (so
## each line's source has its own colour). This is the band the reception filter
## tunes to - the diverse patterns the player fishes for and blends.
func _location_freq() -> float:
	var h := float(hash(str(_lineage)) % 1000) / 1000.0
	var base := 180.0 * pow(2.0, 3.0 * _drift_norm())
	return clampf(base * (0.7 + 0.6 * h), 90.0, 3000.0)


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
	# NO "already on the belt" refusal: a pull ALWAYS throws the pokeball. When the
	# fished voice is one we already own (a landed catch we kept fishing), the catch
	# becomes a fresh DESCENDANT at accept time (see _accept_catch) - so a rich spot
	# on the drift curve can be worked again and again for new variations.
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
		"duration": lerpf(45.0, 240.0, d) * profile_scale \
			* lerpf(1.0, REEL_DIST_STRETCH, _drift_norm()),
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
	# a catch is ALWAYS a new seed: if we were fishing a voice we already own, fork
	# a fresh descendant lineage so we keep the caught VARIATION, never a duplicate
	for e in _belt:
		if e.lineage == _lineage:
			_lineage = _lineage.duplicate()
			_lineage.append(randi() % 1000000)
			break
	for e in _belt:
		if _is_prefix(e.lineage, _lineage):
			e.m.catches += 1
			e.m.r += reward * 0.5
	_landed = true                   # the caught voice speaks at full presence
	_slotted_lineage = _lineage.duplicate()   # the caught seed is now the bait that's out
	_reserve = minf(_reserve + reward, RESERVE_MAX)   # the catch mints adrenochrome for the warp
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
	_status.text = "held +%.1f - hear what you caught (Collection %d/%d)" % [
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
	var removed: Dictionary = _belt[idx]
	_belt.remove_at(idx)
	# if the deleted seed's line (or a descendant of it) was out, cut it
	if _cast and (removed.lineage == _lineage or _is_prefix(removed.lineage, _lineage)):
		_release_line()
	_rebuild_belt()
	_persist()
	# never leave the Collection empty: force a seed back and cast it automatically
	if _belt.is_empty():
		_ensure_collection()
		_throw_from(0)


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


## The candidate's bearing FROM the party, as a 3D vector for the HUD compass -
## direction is otherwise near-impossible to reason about in an 8D trait space.
## The eight trait axes collapse onto three legible ones the ear can name:
##   X = brightness  (pitch, lilt)      - how high/lively vs low/flat
##   Y = damage      (grit, air, breath) - the FRAY axes: smooth vs abraded
##   Z = drive       (pace, drawl, tract) - clipped/urgent vs drawn-out
## Each is the candidate-minus-party offset summed over its group, softened into
## roughly [-1, 1]. With an empty belt the party is the origin, so the compass
## simply reads the voice's own place - still a direction worth seeing.
func _bearing_of(cand: PackedFloat32Array, party: PackedFloat32Array) -> Vector3:
	var off := func(i: int) -> float:
		return float(cand[i]) - float(party[i]) if i < cand.size() and i < party.size() else 0.0
	# indices in Voice.TRAIT_KEYS order: pitch0 lilt1 tract2 pace3 breath4 grit5 drawl6 air7
	var gx: float = off.call(0) + off.call(1)
	var gy: float = off.call(5) + off.call(7) + off.call(4)
	var gz: float = off.call(3) + off.call(6) + off.call(2)
	return Vector3(clampf(gx * 0.7, -1.0, 1.0),
		clampf(gy * 0.55, -1.0, 1.0), clampf(gz * 0.55, -1.0, 1.0))


func _bearing3() -> Vector3:
	return _bearing_of(_seed_vector(_traits, _lineage, _working_genome), _party_vector())


## A belt member's bearing in the SHARED party frame - the same frame for every
## member, so the planets' colours and their shadow directions are mutually
## consistent and do NOT depend on which candidate is currently being fished
## (that is why the shadows "point in directions, consistently, no matter which
## one we're anchored to"). Only an Accept, which moves the party, re-orients
## the whole sky.
func _member_bearing(e: Dictionary, party: PackedFloat32Array) -> Vector3:
	return _bearing_of(_seed_vector(e.traits, e.lineage, _member_genome(e)), party)


## WHERE THE LINE SITS: the working candidate's position, pushed outward along
## its own heading as the drift grows. This is the frame the belt's shadows are
## read against, so the pattern of moons IS the map of the region relative to
## where you are fishing - and it keeps shifting the further out you drift. With
## nothing cast, the frame falls back to the party centre (home).
func _cast_vector() -> PackedFloat32Array:
	var party := _party_vector()
	if not _cast:
		return party
	var cand := _seed_vector(_traits, _lineage, _working_genome)
	var reach := 1.0 + 2.5 * _drift_norm()
	var out := PackedFloat32Array()
	out.resize(cand.size())
	for i in cand.size():
		var pv: float = party[i] if i < party.size() else 0.0
		out[i] = pv + (float(cand[i]) - pv) * reach
	return out


## Re-orient every belt moon (and the big candidate moon) to the current cast.
## Each seed's shadow now points and wanes by where it sits RELATIVE TO THE LINE,
## so casting somewhere new - or drifting further - re-lights the whole belt: the
## regions manifest as a changing pattern of phases. Cheap (belt <= 7); throttled
## by the caller.
func _update_belt_shadows() -> void:
	var cast := _cast_vector()
	for i in mini(_belt.size(), _inv_glyphs.size()):
		var g = _inv_glyphs[i]
		if not is_instance_valid(g):
			continue
		var b := _member_bearing(_belt[i], cast)
		g.planet_dir = atan2(b.y, b.x)
		g.planet_phase = b.z
		g.queue_redraw()
	# the big candidate planet: its OWN bearing from home, waxing full near the
	# party and waning to a crescent (pointing the way it went) as the cast
	# reaches out and the drift carries it further from home
	if _hud_glyph != null and is_instance_valid(_hud_glyph):
		var cb := _bearing3()
		var depth: float = clampf(0.5 * Vector2(cb.x, cb.y).length() + _drift_norm(), 0.0, 1.0)
		_hud_glyph.tint = SeedGlyph.behavior_color(cb)
		_hud_glyph.is_planet = true
		_hud_glyph.planet_dir = atan2(cb.y, cb.x)
		_hud_glyph.planet_phase = 1.0 - 2.0 * depth
		_hud_glyph.queue_redraw()


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
	_inv_slots = []
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
		slot.tooltip_text = "Slot this seed in - cast a line from it. Click again to release."
		var idx := i
		slot.pressed.connect(func(): _on_seed_clicked(idx))
		row.add_child(slot)
		_inv_slots.append(slot)
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
	_update_slot_highlights()        # fresh buttons: re-mark whichever is slotted
	if _coll_title != null and is_instance_valid(_coll_title):
		_coll_title.text = "Collection  %d/%d" % [_belt.size(), BELT_MAX]


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
			# COLOUR is the seed's behaviour in the PARTY frame - an identity that
			# stays put as you fish, so a seed keeps its hue and can be tracked. Its
			# SHADOW is set separately, in the CAST frame, and re-orients live as
			# you cast and drift (see _update_belt_shadows). Frazzle still = fit.
			glyph.tint = SeedGlyph.behavior_color(_member_bearing(e, party))
			glyph.is_planet = true
			glyph.queue_redraw()
	_update_belt_shadows()           # orient the moons to wherever the line sits now


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
	# the drift journey integrates every frame whether or not a stream is playing
	# (it self-gates on drift mode + cast), so the odometer and warp advance the
	# same in a live session and in the autopilot export
	_advance_drift(delta)
	# re-orient the belt's moons to the cast a few times a second, so the shadows
	# track where the line sits and keep shifting as the drift reaches out
	_belt_shadow_t += delta
	if _belt_shadow_t >= 0.12:
		_belt_shadow_t = 0.0
		_update_belt_shadows()
	if _autopilot:
		_autopilot_tick(delta)
		_hud.queue_redraw()
		return
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
		# the fishing spot's acoustics: as the line nears the beacon (the source),
		# a band tuned to it swells into the voice - interesting conditions to fish
		# under, and diverse colour to catch
		_stream.set_location(_beacon_prox(), _location_freq())
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
			# patience pays: an anchored pull slowly mints adrenochrome for the
			# warp - the "explore locally to fill the tank" half of the loop
			if _anchor > 0.1:
				_reserve = minf(_reserve + RESERVE_TRICKLE * delta, RESERVE_MAX)
			_status_t2 += delta
			if _status_t2 >= 0.5 and _pending.is_empty() and not _catching:
				_status_t2 = 0.0
				if _anchor > 0.1:
					_status.text = "something is pulling (%d nibble%s felt) - set the hook when ready" % [
						_nibbles, "" if _nibbles == 1 else "s"]
				elif _status.text.begins_with("something is pulling"):
					_status.text = "the line is slack"
		else:
			_advance_reel(delta)
		_hud.queue_redraw()


## One frame of the reel fight - the fish is dragged in, the adrenochrome
## anneals, the scene contorts, and it lands when progress hits 1. Shared by the
## live session and the autopilot (the live stream's timbre audition is skipped
## when there is no stream, e.g. an export playing itself over a fixed take).
func _advance_reel(delta: float) -> void:
	if _hook.is_empty():
		return
	# THE REEL is a FIGHT, not a timer: progress is the belt's power against the
	# creature's RUNS. A run pays line back out (progress can regress; the voice
	# drops away mid-pull); knowledge reels harder. Everything hooked lands
	# eventually - the wait is the accumulation (the adrenochrome anneals step by
	# step) AND the audition (presence and timbre close in as it nears), so the
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
	# the audition: every couple of seconds the stream's TIMBRE bends toward the
	# annealing adrenochrome (atomic retune; the reading stays the thrown plan
	# until an Accept restarts it in full). No stream (export autopilot) = no
	# audition to run.
	_retune_t += delta
	if _retune_t >= 2.0:
		_retune_t = 0.0
		if _stream != null and is_instance_valid(_stream):
			_stream.retune(Voice.Spec.from_traits(h.traits, int(_lineage[0]), _lineage))
	# the metamorphosis: pulling the fish closer contorts the CURRENT scene, and
	# a foreign catch contorts it BIG
	Director.set_aura(float(h.progress) * (0.4 + 1.1 * float(h.d)))
	_status_t2 += delta
	if _status_t2 >= 0.5:
		_status_t2 = 0.0
		# the numeric percent lives ONCE, in the HUD next to the progress arc; the
		# status line just describes what the reel is doing
		var toll := _hook_toll()
		if h.run > 0.35:
			_status.text = "it runs - the line pays out"
		elif toll > 0.55:
			_status.text = "reeling - its voice is coming apart"
		elif toll > 0.25:
			_status.text = "reeling - something in it is bending"
		else:
			_status.text = "reeling - it changes as it comes"
		_hud_glyph.fit = _relative_fit_of(h.traits, _lineage, h.genome)
		_hud_glyph.queue_redraw()
	if float(h.progress) >= 1.0:
		_finish_hook()


## The autopilot: for the UI-recorded export, the fishing plays itself so the
## panel and HUD animate over a fixed take. A small state machine on random
## timers walks the loop - Throw, let a bite build, Pull, ride the reel, then
## hold or fold - driving the same verbs (and the same Director scene jumps on a
## catch) a player would, but generating no audio of its own.
func _autopilot_tick(delta: float) -> void:
	_ap_t -= delta
	if not _pending.is_empty():
		# a catch is on the card: decide after a beat (mostly keep it)
		if _ap_t <= 0.0:
			if randf() < 0.7:
				_accept_catch()
			else:
				_release_catch()
			_ap_t = randf_range(1.5, 3.0)
	elif not _hook.is_empty():
		_advance_reel(delta)          # ride the fight to the freeze
	elif _catching:
		pass                          # the orb is deciding; wait it out
	elif not _cast:
		_throw()
		_ap_t = randf_range(1.5, 3.5)
	else:
		# cast and searching: manufacture a bite, then set the hook once it holds
		_anchor = clampf(_anchor + delta * (0.35 + 0.4 * randf()), 0.0, 1.15)
		if _nibbles < NIBBLE_MAX and randf() < delta * 1.5:
			_nibbles += 1
		if _ap_t <= 0.0 and _anchor > 0.55:
			_pull()
			_ap_t = randf_range(2.5, 5.0)


func _exit_tree() -> void:
	if _dirty:
		_persist()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST and _dirty:
		_persist()


func _persist() -> void:
	# a UI-recorded export plays the game to make a video, not to progress the
	# save - it must never write over the player's real belt/reserve
	if _autopilot:
		return
	_dirty = false
	var cfg := ConfigFile.new()
	cfg.load(CFG)
	cfg.set_value("synth", "text", _text.text)
	cfg.set_value("synth", "traits", _traits)
	cfg.set_value("synth", "lineage", _lineage)
	cfg.set_value("synth", "belt", _belt)
	cfg.set_value("synth", "profile", _profile)
	cfg.set_value("synth", "adreno", _working_genome)
	cfg.set_value("synth", "reserve", _reserve)
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
		_reserve = float(cfg.get_value("synth", "reserve", 0.0))
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
	# autopilot performs over a FIXED take (the export's audio) - it must not
	# synthesize a stream of its own; the throws are visual only
	if _autopilot:
		return
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
			# WHERE we're catching it, not a percentage - a percent is meaningless
			# when the distance is a lightyear. The winding arc already carries the
			# fight's progress visually.
			draw_string(get_theme_default_font(), Vector2(66.0, s.y - 8.0),
				editor._drift_caption(), HORIZONTAL_ALIGNMENT_LEFT, -1, 11,
				Color(0.4, 0.85, 0.65, 0.75))
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
			_draw_compass(s)         # where the catch sits, all through the fight
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
		# the WIRE bows into an arc by DISTANCE (in any mode - the position holds
		# now), the path we travelled out toward the thing at its far end
		var drift_arc: float = lerpf(0.0, s.y * 0.24, editor._drift_norm())
		var swell := belly + drift_arc
		var pts := PackedVector2Array()
		for i in 25:
			var u := float(i) / 24.0
			var x := lerpf(64.0, s.x - 12.0, u)
			var jitter: float = sin(u * 41.0 + t * 23.0) * d * 5.0 * pull
			pts.append(Vector2(x, y0 + sin(u * PI) * swell + jitter))
		var c := line_col if pull <= 0.05 \
			else Color.from_hsv(lerpf(0.42, 0.0, clampf(d, 0.0, 1.0)), 0.85, 0.9,
				0.5 + 0.4 * minf(pull, 1.0))
		draw_polyline(pts, c, 1.5)
		# out at a distance (drifting there, anchored there, or reeling back) the
		# beacon + travelling position show; the field freezes when anchored and
		# reverses when reeling (both fall out of the odometer holding/shrinking).
		# Only truly at home in anchor/reel do we fall back to a still cast dot.
		var out_there: bool = editor._profile == "drift" or editor._drift_norm() > 0.003
		if out_there:
			_draw_drift_field(s, y0, swell)
			_draw_drift_travel(s, y0, swell, t)
		else:
			var cast_u: float = 0.35 if editor._profile == "reel" else 0.5
			var cp := Vector2(lerpf(64.0, s.x - 12.0, cast_u), y0 + sin(cast_u * PI) * belly)
			draw_circle(cp, 3.2 + (1.2 * sin(t * 2.2) if pull > 0.05 else 0.0),
				Color(0.5, 0.95, 0.8, 0.75))
		if out_there:
			# the honest distance you're parked at, and (drift only) the warp state
			var cap := editor._drift_caption()
			var warp_txt := ""
			var warp_col := Color(0.4, 0.85, 0.65, 0.7)
			if editor._profile == "anchor":
				warp_txt = "  ·  anchored"
			elif editor._profile == "reel":
				warp_txt = "  ·  reeling home"
			elif editor._warping:
				warp_txt = "  ·  ⚡ WARP  (reserve %.1f)" % editor._reserve
				warp_col = Color(0.75, 0.85, 1.0, 0.9)
			elif editor._drift_atmax >= SynthEditor.WARP_CHARGE:
				warp_txt = "  ·  drive charged - need %.1f reserve" % SynthEditor.WARP_MIN \
					if editor._reserve < SynthEditor.WARP_MIN else "  ·  drive charged"
				warp_col = Color(0.9, 0.8, 1.0, 0.85)
			elif editor._drift_vel >= SynthEditor.DRIFT_VMAX - 0.000001:
				warp_txt = "  ·  charging warp %d%%" % int(editor._drift_atmax / SynthEditor.WARP_CHARGE * 100.0)
			draw_string(get_theme_default_font(), Vector2(66.0, s.y - 8.0),
				cap + warp_txt, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, warp_col)
		_draw_compass(s)
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
		var party := editor._party_vector()
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
			# JUST DOTS here - too tiny to read a phase. Colour still carries the
			# seed's behaviour so they can be told apart; the MOONS (the phase, the
			# cast-relative shadow) live on the belt and the big candidate planet.
			var col := SeedGlyph.behavior_color(editor._member_bearing(e, party))
			draw_circle(pos, r, Color(col, 0.85))
			draw_arc(pos, r + 1.4, 0.0, TAU, 12, Color(col, 0.4), 1.0)

	## The DRIFT FIELD: a stream of motes flowing along the wire, scrolling by the
	## odometer so the eye finally sees the drift as MOTION - crawling at raw
	## velocity, and elongating into hyperspace streaks the instant the warp
	## lights. Motes near the caster read bigger and brighter; the deep field
	## thins to faint specks.
	func _draw_drift_field(s: Vector2, y0: float, belly: float) -> void:
		var warp: bool = editor._warping
		# the whole field slides toward the caster as the odometer climbs; the
		# multiplier just sets how many motes pass per unit of distance travelled
		var scroll: float = fposmod(editor._drift_dist * 42.0, 4096.0)   # wrap keeps fractional precision at deep-warp distances
		var n := 18
		for i in n:
			var u: float = fposmod(float(i) / float(n) - scroll, 1.0)   # 0 caster .. 1 deep field
			var x := lerpf(64.0, s.x - 12.0, u)
			var yy := y0 + sin(u * PI) * belly
			var a := lerpf(0.5, 0.07, u)                # near = bright, far = faint
			if warp:
				var u2: float = u + 0.06
				if u2 < 1.0:                            # don't draw a streak across the wrap seam
					draw_line(Vector2(x, yy),
						Vector2(lerpf(64.0, s.x - 12.0, u2), y0 + sin(u2 * PI) * belly),
						Color(0.72, 0.86, 1.0, a), lerpf(1.8, 0.5, u))
			else:
				draw_circle(Vector2(x, yy), lerpf(2.2, 0.8, u), Color(0.55, 0.9, 0.8, a))

	## The drift as a JOURNEY TOWARD a beacon, not a recession from the cast. A
	## marker (the thing at the line's end) sits ahead, and OUR dot travels the arc
	## toward it - reaching it around 0.8 of the way and slipping PAST it toward
	## the deep field as the odometer maxes or a warp surges. The beacon swells as
	## we close on it (things loom as you near them) and dims once overshot, so a
	## warp reads as arriving at (and blowing past) the target, not fleeing it.
	func _draw_drift_travel(s: Vector2, y0: float, swell: float, t: float) -> void:
		var norm: float = editor._drift_norm()
		var x0 := 64.0
		var x1 := s.x - 12.0
		var yat := func(u: float) -> float: return y0 + sin(clampf(u, 0.0, 1.0) * PI) * swell
		var xat := func(u: float) -> float: return lerpf(x0, x1, clampf(u, 0.0, 1.0))
		var u_beacon := 0.8
		var pos_u: float = lerpf(0.14, 1.0, norm)   # us: reaches the beacon ~0.8, beyond = overshoot
		var prox: float = clampf(1.0 - absf(pos_u - u_beacon) / 0.45, 0.0, 1.0)
		var passed: bool = pos_u > u_beacon + 0.01
		# the beacon: the thing at the end of the line, looming as we near it
		var bp := Vector2(xat.call(u_beacon), yat.call(u_beacon))
		var br: float = lerpf(2.5, 6.5, prox)
		var bcol := Color(1.0, 0.82, 0.4, lerpf(0.45, 1.0, prox) * (0.5 if passed else 1.0))
		draw_arc(bp, br + 2.5 + sin(t * 3.0) * 1.0, 0.0, TAU, 20, bcol, 1.5)
		draw_circle(bp, br, bcol)
		# us: travelling toward and past it; a streak trails behind while warping
		var pp := Vector2(xat.call(pos_u), yat.call(pos_u))
		if editor._warping:
			var tu: float = maxf(pos_u - 0.1, 0.0)
			draw_line(Vector2(xat.call(tu), yat.call(tu)), pp, Color(0.72, 0.86, 1.0, 0.75), 2.0)
		draw_circle(pp, 3.4, Color(0.5, 0.95, 0.8, 0.92))

	## The 3D COMPASS: a small isometric gizmo (top-right of the wire) showing
	## which way - and how far - the current candidate sits from the party in the
	## three legible trait axes (see SynthEditor._bearing3). A bearing line with a
	## dot at its tip, plus a dropped shadow that reads the Y (damage) axis as
	## altitude, so up/down is felt as well as seen.
	func _draw_compass(s: Vector2) -> void:
		var b: Vector3 = editor._bearing3()
		var o := Vector2(s.x - 40.0, 22.0)
		var r := 15.0
		# isometric basis: X to the lower-right, Y straight up, Z to the lower-left
		var ex := Vector2(0.92, 0.40) * r
		var ey := Vector2(0.0, -1.0) * r
		var ez := Vector2(-0.92, 0.40) * r
		# the axis cross, faint - each axis a different tint so X/Y/Z read apart
		draw_line(o, o + ex, Color(1.0, 0.5, 0.5, 0.35), 1.0)
		draw_line(o, o + ey, Color(0.5, 1.0, 0.6, 0.35), 1.0)
		draw_line(o, o + ez, Color(0.6, 0.7, 1.0, 0.35), 1.0)
		draw_circle(o, 1.5, Color(1, 1, 1, 0.4))
		var tip := o + ex * b.x + ey * b.y + ez * b.z
		# the ground point: the same bearing with its altitude (Y) flattened, so
		# the vertical drop between them shows how high/low the voice sits
		var ground := o + ex * b.x + ez * b.z
		draw_line(ground, tip, Color(0.9, 0.95, 1.0, 0.25), 1.0)
		draw_line(o, tip, Color(0.95, 0.98, 1.0, 0.85), 1.5)
		draw_circle(ground, 1.3, Color(0.7, 0.8, 1.0, 0.4))
		draw_circle(tip, 2.6, Color(1.0, 0.95, 0.8, 0.95))


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
	# When set (alpha > 0), the glyph's COLOUR is this behaviour tint instead of
	# the fit wheel - so a big belt is told apart by what each seed sounds like,
	# not only how well it fits. The frazzle still reads fit (scatter = poor fit),
	# so both signals survive: colour = behaviour, tightness = belonging.
	var tint := Color(1, 1, 1, 0.0)
	# When is_planet, the glyph's body is drawn as a MOON PHASE (see draw_planet)
	# instead of just a ring - a second visible axis for telling seeds apart. The
	# fingerprint dots draw on top and may fall into the cut, which is fine.
	var is_planet := false
	var planet_dir := 0.0            # the fixed direction the dark side points
	var planet_phase := 0.0          # -1 thin crescent .. +1 near-full

	static func fit_color(f: float) -> Color:
		return Color.from_hsv(lerpf(0.0, 0.42, clampf(f, 0.0, 1.0)), 0.9, 1.0)

	## A seed's behaviour as a full-spectrum colour, from its 3D bearing (see
	## SynthEditor._bearing_of): HUE is the direction in the brightness/damage
	## plane (the whole wheel, so distinct voices land on distinct hues), SAT is
	## how far it sits from the party (kin wash out, foreigners burn), and VALUE
	## rides the drive axis. Every trait axis therefore reaches the eye.
	static func behavior_color(b: Vector3) -> Color:
		var hue: float = fposmod(atan2(b.y, b.x) / TAU + 0.5, 1.0)
		var mag: float = clampf(Vector2(b.x, b.y).length(), 0.0, 1.0)
		var sat: float = clampf(0.4 + 0.6 * mag, 0.0, 1.0)
		var val: float = clampf(0.72 + 0.28 * b.z, 0.4, 1.0)
		return Color.from_hsv(hue, sat, val)

	## Draw a seed AS A MOON: the lit body is built as an actual phase polygon -
	## a bulging lit limb on the sunlit side and a terminator that curves by the
	## phase - so the SHAPE itself is the anchor (a subtractive black mask over a
	## near-black HUD showed nothing). shadow_dir is a fixed global angle for this
	## seed (the dark side points there, consistently for every seed); phase
	## (-1..1) is the SIGN FLIP - near +1 the lit face is toward us (near-full),
	## near -1 the dark face is (a thin crescent). Only the lit part is drawn; the
	## rest of the body simply is not there.
	static func draw_planet(ci: CanvasItem, center: Vector2, r: float,
			color: Color, shadow_dir: float, phase: float) -> void:
		var k: float = clampf(phase, -0.85, 0.98)   # illumination bulge: +full .. -sliver
		var lit: float = shadow_dir + PI            # the sunlit direction (opposite the shadow)
		var cs := cos(lit)
		var sn := sin(lit)
		var pts := PackedVector2Array()
		var steps := 18
		# the lit limb: the outer semicircle facing the sun
		for i in steps + 1:
			var a := lerpf(-PI * 0.5, PI * 0.5, float(i) / float(steps))
			var lx := r * cos(a)
			var ly := r * sin(a)
			pts.append(center + Vector2(lx * cs - ly * sn, lx * sn + ly * cs))
		# the terminator: an ellipse arc back across, its bulge (and side) set by k
		for i in steps + 1:
			var y := lerpf(r, -r, float(i) / float(steps))
			var lx := -k * sqrt(maxf(r * r - y * y, 0.0))
			pts.append(center + Vector2(lx * cs - y * sn, lx * sn + y * cs))
		ci.draw_colored_polygon(pts, color)
		# a brighter lit-limb arc to catch the eye
		ci.draw_arc(center, r, lit - PI * 0.5, lit + PI * 0.5, 20,
			Color(color.lightened(0.4), 0.85), 1.0)

	func _draw() -> void:
		var rng := RandomNumberGenerator.new()
		rng.seed = seed_hash
		var s := size
		var col := tint if tint.a > 0.0 else fit_color(fit)
		# the body: a moon-phase disc when this glyph carries a bearing, else the
		# plain fit ring. Drawn FIRST so the fingerprint sits on top of it.
		if is_planet:
			draw_planet(self, s * 0.5, minf(s.x, s.y) * 0.5 - 1.5, col, planet_dir, planet_phase)
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
		# over a coloured moon the fingerprint needs contrast; on its own it keeps
		# the fit colour it always had
		var star := Color(col.lightened(0.55), 0.9) if is_planet else Color(col, 1.0)
		var link := Color(col.lightened(0.4), 0.5) if is_planet else Color(col, 0.55)
		for i in range(1, n):
			var j := rng.randi_range(0, i - 1)
			draw_line(pts[i], pts[j], link, 1.0)
		for p in pts:
			draw_circle(p, 2.0, star)
		if not is_planet:
			# the ring: the fit colour, unmissable even at 24 px
			draw_arc(s * 0.5, minf(s.x, s.y) * 0.5 - 1.5, 0.0, TAU, 32, Color(col, 0.9), 2.0)
