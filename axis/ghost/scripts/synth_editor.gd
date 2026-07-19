extends CanvasLayer
class_name SynthEditor

## SynthEditor - the synthesis surface. The entire loop is one gesture:
##
##   THROW -> it grows or it doesn't -> KEEP or don't.
##
## Nothing else is exposed - no sliders, no toggles, no seed fields. EVERY
## property (the voice's trait vector and the reading's prosody genome) lives
## inside each seed, travels with every throw, and is inspectable on the belt
## (a seed's tooltip reads out its genome and ledger). The listening ear stays
## free to judge the audio, which is the whole point.
##
## A throw produces a new candidate and speaks it immediately. The belt does
## the integration implicitly: throws parent themselves from kept seeds
## weighted by **acceptance** (hold time per play - you vote by not switching
## it off), inheriting lineage + traits with generation-decaying jitter;
## sometimes a throw goes wild (fresh root, fresh voice) so the population
## never inbreeds. Ancestors accrue hold time from their descendants'
## listening, so a seed's acceptance reflects what its LINE produces. The
## **background voice** - what you hear before any seed exists - is the
## population average: the acceptance-weighted mean of the belt's traits, or
## the hand-curated default (the zero vector) on a fresh install.
##
## The lyrics box takes a plain paragraph (`[K AE T]` phonetic escapes).
## Draft, voice, lineage, and belt autosave. `--say` speaks on boot.

const CFG := "user://ghost.cfg"
const AUTOSAVE_DELAY_MS := 800
const BELT_MAX := 7                 # a small cache; old captures fall off the end

## Reward profiles: the user picks what kind of session this is, and the
## profile shapes BOTH how throws range (wild chance, jitter) and how catches
## earn. The signal insight: a QUICK catch means the ear knew immediately -
## droning on means deliberating - so decision latency, not watch length, is
## the default reward (Snap). Drift restores the sit-with-it mode; Hunt pays
## for catching seeds that are far from the bank.
const PROFILES := {
	"drift": {"label": "Drift", "wild": 0.15, "jitter": 0.8,
		"tip": "Sit with it. Long listening earns; catches never rush you."},
	"snap": {"label": "Snap", "wild": 0.2, "jitter": 1.0,
		"tip": "Trust the gut. The quicker the catch, the higher the reward."},
	"hunt": {"label": "Hunt", "wild": 0.45, "jitter": 1.5,
		"tip": "Range widely. Hard-to-integrate seeds earn the most when caught."},
}

var begin_stream: Callable          # set by main: (stream: VoiceStream) -> void

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
var _metrics_t := 0.0               # throttle for inventory refresh/persist
var _profile := "snap"              # active reward profile key
var _profile_btn: OptionButton
var _throw_ms := 0                  # when the current candidate started playing
var _catching := false              # a catch attempt (orb animation) is in flight
var _dirty := false                 # autosave pending
var _restart_pending := false       # structural change awaiting the debounce
var _last_edit_ms := 0


func _ready() -> void:
	layer = 10
	_build_panel()
	_load_persisted()
	var args := OS.get_cmdline_user_args()
	var i := args.find("--synth")
	if i >= 0 and i + 1 < args.size() and FileAccess.file_exists(args[i + 1]):
		_text.text = FileAccess.get_file_as_string(args[i + 1])
	# connect AFTER initial load so restoring the draft doesn't mark it dirty
	_text.text_changed.connect(_mark_structural)
	# implicit speaking: the loaded draft speaks on its own - immediately with
	# --say, after the normal debounce otherwise
	if not _text.text.strip_edges().is_empty():
		if args.has("--say"):
			_apply.call_deferred()
		else:
			_mark_structural()


func _build_panel() -> void:
	_panel = PanelContainer.new()
	_panel.position = Vector2(16, 16)
	_panel.custom_minimum_size = Vector2(380, 0)
	add_child(_panel)
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	_panel.add_child(box)

	var title := Label.new()
	title.text = "Synthesis"
	title.add_theme_font_size_override("font_size", 20)
	box.add_child(title)

	var hint := Label.new()
	hint.text = "Write and it speaks. Throw until something grows; Keep what does."
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.add_theme_font_size_override("font_size", 12)
	hint.modulate = Color(1, 1, 1, 0.6)
	box.add_child(hint)

	_text = TextEdit.new()
	_text.custom_minimum_size = Vector2(360, 220)
	_text.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_text.placeholder_text = "Once upon a time..."
	box.add_child(_text)

	# --- The loop: Throw / Keep ---
	var loop_row := HBoxContainer.new()
	loop_row.add_theme_constant_override("separation", 8)
	box.add_child(loop_row)
	var throw := Button.new()
	throw.text = "Throw"
	throw.custom_minimum_size = Vector2(120, 40)
	throw.tooltip_text = ("A new candidate - voice and reading in one seed - "
		+ "parented from the belt by acceptance (sometimes wild), spoken immediately")
	throw.pressed.connect(_throw)
	loop_row.add_child(throw)
	var keep := Button.new()
	keep.text = "Keep"
	keep.custom_minimum_size = Vector2(90, 40)
	keep.tooltip_text = ("A catch ATTEMPT: seeds far from your bank fight the "
		+ "ball - it may break free. Retry costs nothing but time.")
	keep.pressed.connect(_attempt_catch)
	loop_row.add_child(keep)
	_reading_label = Label.new()
	_reading_label.add_theme_font_size_override("font_size", 12)
	_reading_label.modulate = Color(1, 1, 1, 0.7)
	loop_row.add_child(_reading_label)

	# --- The reward profile: what kind of session is this? ---
	var reward_row := HBoxContainer.new()
	reward_row.add_theme_constant_override("separation", 8)
	box.add_child(reward_row)
	var reward_label := Label.new()
	reward_label.text = "Reward"
	reward_label.add_theme_font_size_override("font_size", 12)
	reward_label.modulate = Color(1, 1, 1, 0.6)
	reward_row.add_child(reward_label)
	_profile_btn = OptionButton.new()
	_profile_btn.focus_mode = Control.FOCUS_NONE
	for key in PROFILES:
		_profile_btn.add_item(PROFILES[key].label)
	var keys := PROFILES.keys()
	_profile_btn.select(maxi(0, keys.find(_profile)))
	_profile_btn.tooltip_text = PROFILES[_profile].tip
	_profile_btn.item_selected.connect(func(idx: int):
		_profile = keys[idx]
		_profile_btn.tooltip_text = PROFILES[_profile].tip
		_persist())
	reward_row.add_child(_profile_btn)

	_belt_rows = VBoxContainer.new()
	_belt_rows.add_theme_constant_override("separation", 4)
	box.add_child(_belt_rows)

	_status = Label.new()
	_status.text = "ready - write and it speaks"
	_status.modulate = Color(1, 1, 1, 0.7)
	box.add_child(_status)

	var hide := Button.new()
	hide.text = "Hide panel (F2)"
	hide.pressed.connect(func(): _panel.visible = false)
	box.add_child(hide)


func _current_spec() -> Voice.Spec:
	return Voice.Spec.from_traits(_traits, int(_lineage[0]), _lineage)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_F2:
		_panel.visible = not _panel.visible


# ---- the loop: throw, keep, release ----------------------------------------
#
# The reward signal is HOLD TIME: while a reading plays, its exact seed (if
# kept) and every belt ANCESTOR of it accrue listened seconds - a parent earns
# from what its line produces. Acceptance = hold-per-play vs the belt's
# collective average; throws pick parents by it. catch = a descendant kept,
# keep = restored, evolve = thrown from, release = deleted.


## The one gesture. A fresh candidate: usually a child of a belt seed picked by
## acceptance (inheriting lineage + traits with generation-decaying jitter),
## sometimes wild. Speaks immediately; the listening decides its fate.
func _throw() -> void:
	var prof: Dictionary = PROFILES[_profile]
	if _belt.is_empty() or randf() < float(prof.wild):
		_lineage = [randi() % 1000000]
		var rng := RandomNumberGenerator.new()
		rng.seed = _lineage[0]
		_traits = Voice.Spec.sample(rng).traits
		_status.text = "thrown (wild)"
	else:
		var parent: Dictionary = _pick_parent()
		parent.m.evolves += 1
		_lineage = (parent.lineage as Array).duplicate()
		_lineage.append(randi() % 1000000)
		var jitter: float = 0.22 * pow(0.75, _lineage.size() - 1) * float(prof.jitter)
		var t: Dictionary = parent.traits
		_traits = {}
		for key in Voice.TRAIT_KEYS:
			_traits[key] = clampf(
				float(t.get(key, 0.0)) + randfn(0.0, maxf(jitter, 0.06)), -1.0, 1.0)
		_status.text = "thrown (from %s)" % _seed_name(parent.lineage)
	_throw_ms = Time.get_ticks_msec()
	_update_reading_label()
	_persist()
	_apply()


## Acceptance-weighted parent pick: the belt's collective judgment shapes what
## gets thrown next - the integration the user never has to do by hand.
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


## The population-average background: the voice that exists before any seed
## does - the acceptance-weighted mean of the belt's traits, or the curated
## default (the zero vector) when the belt is empty.
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


## Integration difficulty: how far the candidate sits from the NEAREST seed in
## the bank, in combined trait + genome space. Harmonic with your collection =
## an easy catch; foreign = it fights the ball. Empty belt = moderate.
func _candidate_difficulty() -> float:
	if _belt.is_empty():
		return 0.35
	var cg: Dictionary = Voice.ProsodyWalk._lineage_genome(_lineage)
	var best := 1.0
	for e in _belt:
		var dt := 0.0
		for key in Voice.TRAIT_KEYS:
			var dv: float = float(_traits.get(key, 0.0)) \
				- float((e.traits as Dictionary).get(key, 0.0))
			dt += dv * dv
		dt = sqrt(dt / Voice.TRAIT_KEYS.size()) / 0.9      # ~[0,1]
		var eg: Dictionary = Voice.ProsodyWalk._lineage_genome(e.lineage)
		var dg := 0.0
		for key in cg:
			dg += absf(float(cg[key]) - float(eg[key])) / maxf(absf(float(
				Voice.ProsodyWalk.PRIOR[key])), 0.001)
		dg = (dg / cg.size()) / 0.35                        # ~[0,1]
		best = minf(best, 0.5 * clampf(dt, 0.0, 1.0) + 0.5 * clampf(dg, 0.0, 1.0))
	return clampf(best, 0.0, 1.0)


## The catch reward, by profile. Snap: decision latency - the quicker the ear
## knew, the more it earns. Drift: time spent sitting with it. Hunt: how far
## from the bank the caught seed was.
func _catch_reward(difficulty: float) -> float:
	var t_s: float = (Time.get_ticks_msec() - _throw_ms) / 1000.0
	match _profile:
		"drift":
			return 2.4 * clampf(t_s / 45.0, 0.0, 1.0)
		"hunt":
			return 0.4 + 2.0 * difficulty
		_:
			return 2.4 * 8.0 / (8.0 + t_s)


## Keep is a catch ATTEMPT: roll against integration difficulty, animate the
## orb, then commit or break free. Retrying is allowed - but in Snap the clock
## keeps running, so hesitation costs.
func _attempt_catch() -> void:
	if _catching:
		return
	for e in _belt:
		if e.lineage == _lineage:
			_status.text = "already on the belt"
			return
	var d := _candidate_difficulty()
	var success := randf() < 1.0 - 0.75 * d
	var wobbles := 3 if success else 1 + randi() % 3
	_catching = true
	var orb := CatchOrb.new()
	orb.hue = float(hash(str(_lineage)) % 360) / 360.0
	orb.wobbles = wobbles
	orb.success = success
	add_child(orb)
	orb.finished.connect(func():
		orb.queue_free()
		_catching = false
		if success:
			_commit_catch(d)
		else:
			_status.text = "broke free (%s catch) - again?" % _difficulty_word(d))


func _commit_catch(difficulty: float) -> void:
	var reward := _catch_reward(difficulty)
	for e in _belt:
		if _is_prefix(e.lineage, _lineage):
			e.m.catches += 1        # its line lives on - attachment, earned
			e.m.r += reward * 0.5   # ancestors share the catch
	_belt.append({
		"lineage": _lineage.duplicate(), "traits": _traits.duplicate(),
		"m": {"s": 0.0, "acts": 1, "restores": 0, "evolves": 0, "catches": 0,
			"r": reward, "d": difficulty,
			"t": int(Time.get_unix_time_from_system())},
	})
	while _belt.size() > BELT_MAX:
		_belt.pop_front()
	_rebuild_belt()
	_persist()
	_status.text = "caught! +%.1f (belt %d/%d)" % [reward, _belt.size(), BELT_MAX]


func _difficulty_word(d: float) -> String:
	if d < 0.25:
		return "easy"
	if d < 0.5:
		return "firm"
	if d < 0.75:
		return "hard"
	return "wild"


func _restore_capture(idx: int) -> void:
	if idx < 0 or idx >= _belt.size():
		return
	var entry: Dictionary = _belt[idx]
	entry.m.restores += 1
	entry.m.acts += 1
	_lineage = (entry.lineage as Array).duplicate()
	_traits = (entry.traits as Dictionary).duplicate()
	_throw_ms = Time.get_ticks_msec()
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
	_reading_label.text = "%s · %s catch" % [
		_seed_name(_lineage), _difficulty_word(_candidate_difficulty())]


## Acceptance: catch rewards (weighted heavily - a catch is a decision) plus
## hold time, per play, relative to the belt's collective average. 1.0x = an
## average seed; above = the belt favors it. The profile shapes how reward is
## EARNED; this reads it uniformly.
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
	for i in _belt.size():
		var entry: Dictionary = _belt[i]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 6)
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
		metrics.mouse_filter = Control.MOUSE_FILTER_PASS   # tooltips need hover
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


func _refresh_inventory() -> void:
	for i in mini(_belt.size(), _inv_labels.size()):
		var e: Dictionary = _belt[i]
		var label: Label = _inv_labels[i]
		if not is_instance_valid(label):
			continue
		label.text = "%.0fs·%d× %.1fx" % [e.m.s, int(e.m.acts), _acceptance(e)]
		label.tooltip_text = _seed_tooltip(e)


## The seed as an inspectable property: its behavioral ledger plus what its
## genome DOES to the model - temperament, breath, gravity, ring, strike bar -
## and the voice it carries.
func _seed_tooltip(e: Dictionary) -> String:
	var g: Dictionary = Voice.ProsodyWalk._lineage_genome(e.lineage)
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
		("voice: " + tv) if not tv.is_empty() else "voice: near default"]


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
	# The attention reward: while a reading plays, its exact seed and every
	# belt ANCESTOR of it accrue hold time - a parent earns from its line.
	# Moving on stops the accrual; that IS the signal.
	if _stream != null and is_instance_valid(_stream) and _stream.words.size() > 0:
		for e in _belt:
			if e.lineage == _lineage or _is_prefix(e.lineage, _lineage):
				e.m.s += delta
		_metrics_t += delta
		if _metrics_t >= 5.0:
			_metrics_t = 0.0
			_refresh_inventory()
			_persist()


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
	cfg.save(CFG)


func _load_persisted() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(CFG) == OK:
		_text.text = cfg.get_value("synth", "text", "")
		_traits = cfg.get_value("synth", "traits", {})
		_lineage = cfg.get_value("synth", "lineage", [1])
		_belt = cfg.get_value("synth", "belt", [])
		_profile = str(cfg.get_value("synth", "profile", "snap"))
		if not PROFILES.has(_profile):
			_profile = "snap"
		if _profile_btn != null:
			_profile_btn.select(maxi(0, (PROFILES.keys() as Array).find(_profile)))
			_profile_btn.tooltip_text = PROFILES[_profile].tip
		# migrate entries kept before the metrics/reward ledger existed
		for e in _belt:
			if not e.has("m"):
				e.m = {"s": 0.0, "acts": 0, "restores": 0, "evolves": 0,
					"catches": 0, "t": 0}
			if not e.m.has("r"):
				e.m.r = 0.0
				e.m.d = 0.35
			e.erase("active")        # toggles are gone; the belt integrates itself
	if _traits.is_empty():
		_traits = _background_traits()   # the population-average background
	_update_reading_label()
	_rebuild_belt()


## Start the stream, or restart the running one in place with the current text
## and voice. The scene session persists across restarts.
func _apply() -> void:
	_restart_pending = false
	var text := _text.text.strip_edges()
	if text.is_empty():
		_status.text = "write something and it will speak"
		return
	var spec := _current_spec()
	if _stream != null and is_instance_valid(_stream):
		_stream.restart(text, spec)
		return
	var stream: VoiceStream = preload("res://scripts/voice_stream.gd").new()
	stream.setup(text, spec, "user://synth/take_%06x" % (hash(str(_lineage)) & 0xFFFFFF))
	stream.completed.connect(func(dur: float, _wav: String):
		_status.text = "take complete (%.1fs) - looping, export-ready" % dur)
	_stream = stream
	if begin_stream.is_valid():
		begin_stream.call(stream)


## The catch animation, code-drawn: an orb in the seed's own hue closes on the
## voice, wobbles like the catch is being fought, then either settles with a
## soft ring (caught) or bursts into shards (broke free). Purely visual - the
## roll is decided before the orb appears; the orb TELLS you.
class CatchOrb:
	extends Control
	signal finished
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
		var centre := get_viewport_rect().size * 0.5 + Vector2(0.0, 120.0)
		var body := Color.from_hsv(hue, 0.6, 0.95)
		var rim := Color(1, 1, 1, 0.9)
		if _t < CLOSE:
			# closing in: a wide ring collapses onto the voice
			var u := _t / CLOSE
			var r := lerpf(90.0, 26.0, u * u)
			draw_arc(centre, r, 0.0, TAU, 40, Color(body, u * 0.9), 3.0)
			draw_circle(centre, 26.0 * u, Color(body, u * 0.55))
			return
		var tw := _t - CLOSE
		if tw < wobbles * WOBBLE:
			# the fight: the orb rocks - each wobble a fresh contest
			var k := int(tw / WOBBLE)
			var u := fmod(tw, WOBBLE) / WOBBLE
			var rock: float = sin(u * TAU * 1.5) * (1.0 - u) * 10.0 * (1.0 + 0.3 * k)
			var pos := centre + Vector2(rock, -absf(rock) * 0.3)
			draw_circle(pos, 26.0, Color(body, 0.85))
			draw_arc(pos, 26.0, 0.0, TAU, 32, rim, 2.0)
			draw_circle(pos + Vector2(-7, -8), 5.0, Color(1, 1, 1, 0.5))
			return
		# the verdict
		var u := (_t - CLOSE - wobbles * WOBBLE) / RESULT
		if success:
			# settle: the orb rests, a soft ring blesses it
			draw_circle(centre, 26.0, Color(body, 0.85 * (1.0 - u * 0.4)))
			draw_arc(centre, 26.0 + u * 46.0, 0.0, TAU, 40,
				Color(1, 1, 1, 0.7 * (1.0 - u)), 2.5)
		else:
			# broke free: shards fly, the orb is gone
			for i in 7:
				var a := TAU * float(i) / 7.0 + hue * TAU
				var p := centre + Vector2(cos(a), sin(a)) * (12.0 + u * 70.0)
				draw_circle(p, 4.5 * (1.0 - u), Color(body, 0.8 * (1.0 - u)))
