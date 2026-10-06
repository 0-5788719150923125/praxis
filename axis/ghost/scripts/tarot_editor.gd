extends GenerativeEditor
class_name TarotEditor

## TarotEditor - the tarot mode: a reading nobody writes by hand.
##
## A SHOW is a document: its title is the channel, its body is the BRIEF - what the show is,
## its voice, its rules - and its `ghost: tarot:` block holds the knobs and the reader's voice.
## An EPISODE is one seed of the show, produced by agents ([TarotProducer]) step by step and
## kept on disk ([TarotEpisode]): planned, shuffled, designed, painted, and read aloud one card
## at a time, in drawing order. What they write is then read in this panel's voice - the whole
## Generative pipeline, inherited rather than copied - at the tarot table ([TarotMedium]), which
## this mode pins as its medium.
##
## GENERATION IS EXPLICIT, as with every picture in ghost: Generate makes what the episode is
## missing, New episode makes a new seed, and a step's redo deletes it (and what was made from
## it) and makes it again. Nothing runs on open - it spends the author's quota.

const YouTube := preload("res://scripts/youtube.gd")
const TagField := preload("res://scripts/tag_field.gd")

## The knobs a show keeps beside its voice. Also the schema: a stored value of the wrong shape
## falls back to these.
const KNOBS := {"show": "", "seed": 1, "cards": [3, 6], "reversals": true, "jumpers": true,
	"writer": "claude", "writer_model": "", "writer_effort": "", "painter": "codex", "painter_model": "",
	"painter_effort": ""}
## The rows of the episode list: a label and the steps it covers ("K" is each card).
const ROWS := [
	["Plan", ["plan"]],
	["Shuffle", ["draw"]],
	["Table", ["image:back", "image:surface", "image:backdrop", "table"]],
	["Intro", ["say:intro"]],
	["Card", ["design:K", "image:card:K", "say:K"]],
	["Close", ["say:close", "script"]],
]
## What a row's redo menu offers: label -> the step made again. ONLY that step is made: what
## was made from it is cleared (it no longer follows), and waits for Generate. The shuffle has
## none: the draw is a function of the seed, so dealing again deals the same cards - a different
## deal is a different episode (New episode).
const REDO := {
	"Plan": [["Make a new plan (clears the cards and the reading)", "plan"]],
	"Table": [["Paint a new card back", "image:back"], ["Paint a new cloth", "image:surface"],
		["Paint a new room", "image:backdrop"], ["Set the table again", "table"]],
	"Intro": [["Rewrite the intro (clears the readings after it)", "say:intro"]],
	"Card": [["Paint this card again (clears its reading and the ones after)", "image:card:K"],
		["Rewrite this card's reading (clears the ones after it)", "say:K"],
		["Redesign the card (clears its picture and its reading on)", "design:K"]],
	"Close": [["Rewrite the close", "say:close"]],
}

var _knobs := KNOBS.duplicate(true)
var _episode: TarotEpisode = null
var _producer: TarotProducer = null
var _episode_pick: OptionButton
var _episode_seeds: Array = []
var _new_btn: Button
var _gen_btn: Button
var _halt_btn: Button
var _del_btn: Button
var _writer_pick: OptionButton
var _painter_pick: OptionButton
var _writer_model: OptionButton
var _painter_model: OptionButton
var _writer_effort: OptionButton
var _painter_effort: OptionButton
var _cards_lo: SpinBox
var _cards_hi: SpinBox
var _reversals: CheckBox
var _jumpers: CheckBox
var _rows_box: VBoxContainer
var _episode_note: Label
var _row_t := 0.0
var _brief_changed := false
var _deck_note: Label
var _deck_seen := -1
var _table_seen := {}
var _table_log := {}
## THE EPISODE AN EXPORT IS OF, held from the moment it is asked for: synthesis takes minutes,
## and an episode picked meanwhile must not lend the take its cards or its upload notes.
## {episode, body, doc, title}, or empty.
var _export_pin := {}
## The take the last export rendered and the episode it is of: the upload describes them (see
## [method upload_meta]) after the pin is let go. {take, episode}, or empty.
var _taken := {}
## THE YOUTUBE FIELDS: the picked episode's title, description and tags, edited in place in its plan.
var _yt_title: LineEdit
var _yt_desc: TextEdit
var _yt_tags: TagField
var _yt_note: Label
var _yt_dirty := 0.0       # seconds until edits are written into the plan; 0 = nothing waiting
var _yt_seen := {}         # {path, mt}: the plan the fields were last filled from
## THE SHOW'S YOUTUBE TITLE AND DESCRIPTION, kept in its document's block - global, never a record
## per episode. The title is ONE value overwritten by whichever episode is picked (as the seed is);
## the description is the show's, the same for every episode (see [method _doc_capture]).
var _cur_title := ""
var _cur_desc := ""


func _init() -> void:
	_section = "tarot"


func _ready() -> void:
	# PINNED BEFORE ANYTHING ATTACHES: the stage is built on the first reading, and it builds
	# whatever medium resolves then (see Director.medium_override).
	Director.medium_override = "tarot"
	super._ready()
	_open_episode()


func _exit_tree() -> void:
	if _producer != null:
		_producer.stop()
	if Director.medium_override == "tarot":
		Director.medium_override = ""
	super._exit_tree()


func _process(delta: float) -> void:
	super._process(delta)
	if _yt_dirty > 0.0:
		_yt_dirty -= delta
		if _yt_dirty <= 0.0:
			_flush_upload_fields(true)
	# four times a second is plenty for runs that take tens of seconds, and every tick asks the
	# disk which steps exist
	_row_t -= delta
	if _row_t <= 0.0:
		_row_t = 0.25
		if _producer != null:
			_producer.tick()
		_refresh_rows()
		_seed_globals()


# --- the seams ---------------------------------------------------------------------------

func _panel_title() -> String:
	return "Tarot"


func _panel_hint() -> String:
	return ("Open a show: its body is the brief every agent works from. Each seed is one episode - "
		+ "planned, shuffled, painted and written one card at a time by the agents, then read here "
		+ "in the voice below, at the table.")


func _pinned_medium() -> String:
	return "tarot"


## THE READING IS THE EPISODE'S, never the document's: the document is the brief. The pull still
## happens first, because it is what flushes a voice change into the show's file.
func _reading_body() -> String:
	_doc.pull()
	return _episode.script() if _episode != null else ""


func _reading_of(body: String) -> Dictionary:
	return {"body": TarotScript.speakable(body), "title": ""}


func _cast_text() -> String:
	return _episode.script() if _episode != null else ""


func _fingerprint_text() -> String:
	return _cast_text()


func _has_reading() -> bool:
	return not _cast_text().strip_edges().is_empty()


## Editing the brief does not change a reading already made: it changes the NEXT one. Say so.
func _on_source_edited() -> void:
	_brief_changed = true


func _test_passage() -> String:
	return ("Hello, my loves, and welcome back to the channel. This is a timeless reading, so "
		+ "whenever you find it is exactly when you were meant to. Let's shuffle. Oh. Oh, wow. "
		+ "Okay, I'm getting chills - the cards are really insisting on this one tonight.")


func book_document(body := "") -> Dictionary:
	var d := super.book_document(body)
	# an export's own reading is of the episode pinned when it was asked for (see export_take)
	var pinned := not _export_pin.is_empty() and body.strip_edges() == String(_export_pin["body"])
	d["title"] = String(_export_pin["title"]) if pinned else _show_title()
	d["byline"] = String(_export_pin.get("byline", "")) if pinned else _show_byline()
	d["tarot"] = _export_pin["doc"] if pinned else (_episode.document() if _episode != null else {})
	return d


## AN EXPORT IS NAMED AFTER ITS EPISODE: the title the producer gave it (made safe for a file name
## by the exporter), or the show and seed while it has none - a folder of `ghost_1080p.mp4`s
## says nothing about which reading is which.
func export_name() -> String:
	if _episode == null:
		return ""
	var plan: Variant = _episode.read_json("plan")
	var t := String((plan as Dictionary).get("episode_title", "")).strip_edges() if plan is Dictionary else ""
	return t if not t.is_empty() else "%s %d" % [_show_title(), _episode.seed]


## THE EXPORT, and its upload notes: the take is rendered exactly as the Generative panel renders
## one, then the episode's folder gets `upload.md` - the title, the description and tags the
## producer wrote, and a chapter per card timed from the take itself. One show makes many videos;
## each should arrive ready to post.
##
## THE EPISODE IS PINNED BEFORE THE FIRST AWAIT: the take is minutes of synthesis, and the panel
## stays live through it.
func export_take() -> String:
	_doc.pull()          # a change in the show's file lands now, not part way through
	_flush_upload_fields()
	var ep := _episode
	if ep == null:
		return ""
	_export_pin = {"episode": ep, "body": ep.script().strip_edges(), "doc": ep.document(),
		"title": _show_title(), "byline": _show_byline()}
	var path: String = await super.export_take()
	_export_pin = {}
	if not path.is_empty():
		_taken = {"take": path, "episode": ep}
		var err := ep.write_upload_notes(path, _show_tags(), _cur_desc,
			upload_title(_doc.field("title"), String(ep.upload_notes()["title"])))
		_set_status(("Rendered the take; upload notes are in the episode's folder (upload.md)." if err.is_empty()
			else "Rendered the take; the upload notes failed: " + err))
	return path


## WHAT AN UPLOAD SAYS (see [member Exporter.upload_provider]): the episode's title and description
## with a chapter per card timed from [param take], the SHOW'S tags (its document's `tags:` line),
## the moment of the title screen to take the thumbnail from, and the file its uploads are recorded
## in. For the take an export just rendered it describes THAT episode, whichever is picked now; with
## no take, the picked one. {} while the episode has no plan.
func upload_meta(take: String) -> Dictionary:
	var ep: TarotEpisode = _episode
	if not take.is_empty() and String(_taken.get("take", "")) == take:
		ep = _taken["episode"]
	if ep == null:
		return {}
	var n := ep.upload_notes(take)
	if String(n["title"]).is_empty():
		return {}
	var desc := _cur_desc if not _cur_desc.is_empty() else String(n["description"])
	var chapters: PackedStringArray = n["chapters"]
	if not chapters.is_empty():
		desc += "\n\n" + "\n".join(chapters)
	return {"title": upload_title(_doc.field("title"), String(n["title"])), "description": desc, "tags": Array(_show_tags()),
		"thumbnail_at": thumbnail_moment(_take_intro(take)), "record": ep.file_of("youtube")}


## THE UPLOAD'S TITLE: the show's name, then the episode's - "Truthful Tarot: My Episode Name" (the
## user, 2026-10-06) - unless the episode's already begins with the name, or the show has none.
## YouTube's 100 characters are cut from the end ([method YouTube.fit_title]), so the name stands.
static func upload_title(show: String, episode: String) -> String:
	var name := show.strip_edges()
	var t := episode.strip_edges()
	if name.is_empty() or t.to_lower().begins_with(name.to_lower()):
		return t
	return "%s: %s" % [name, t] if not t.is_empty() else name


## WHERE THE THUMBNAIL IS TAKEN, seconds into the video: the title screen with the show's name fully
## up over the out-of-focus table - after the name fades in (by 1.4 s) and before the focus pull,
## which starts 1.4 s before the shuffle at the intro's end (see the tarot medium's `_title_alpha`
## and `_tick_focus`). [param intro] is the take's intro hold.
static func thumbnail_moment(intro: float) -> float:
	return clampf(intro * 0.4, 1.5, maxf(1.5, intro - 1.6))


## The intro hold a take was rendered with (its sidecar's bookend), else the Director's.
static func _take_intro(take: String) -> float:
	var side := take.get_basename() + ".json"
	if take.is_empty() or not FileAccess.file_exists(side):
		return Director.intro_hold
	var j := JSON.new()
	if j.parse(FileAccess.get_file_as_string(side)) != OK or not (j.data is Dictionary):
		return Director.intro_hold
	var b: Variant = (j.data as Dictionary).get("bookend", {})
	return float((b as Dictionary).get("in", Director.intro_hold)) if b is Dictionary else Director.intro_hold


## THE SHOW'S TAGS: its document's own `tags:` line, as North Star's chapters keep theirs
## (`tags: a, b, c`) - every episode goes up with them.
func _show_tags() -> PackedStringArray:
	return YouTube.split_tags(_doc.field("tags")) if _doc != null else PackedStringArray()


## The show's byline: its document's `byline:` line.
func _show_byline() -> String:
	return _doc.field("byline").strip_edges() if _doc != null else ""


# --- the show's document ---------------------------------------------------------------------

func _show_title() -> String:
	var t := _doc.field("title").strip_edges() if _doc != null else ""
	return t if not t.is_empty() else "Untitled Tarot"


## What the producer works from, read fresh: the title and byline, the brief (its card LIST taken out - see
## [method TarotDeck.strip]), the deck the brief defines (or the standard 78), the show's voices
## (every one its document names - the reader's and any other, a familiar's - for the reader to
## hand lines to), and the knobs.
func _spec() -> Dictionary:
	var body := Manuscript.strip_frontmatter(_doc.pull())
	return {"title": _show_title(), "byline": _show_byline(), "brief": TarotDeck.strip(body), "deck": TarotDeck.of(body),
		"voices": _cast_dict().keys(),
		"cards": _knobs["cards"], "reversals": _knobs["reversals"], "jumpers": _knobs["jumpers"],
		"writer": _knobs["writer"], "writer_model": _knobs["writer_model"], "writer_effort": _knobs["writer_effort"],
		"painter": _knobs["painter"], "painter_model": _knobs["painter_model"], "painter_effort": _knobs["painter_effort"]}


func _doc_capture() -> Dictionary:
	var d := super._doc_capture()
	# the Generative block's picture library and the settings no table uses are not this show's
	d.erase("illustrations")
	var pic: Dictionary = d.get("picture", {})
	for k in ["scene_hold", "flourishes", "hand", "film_frequency", "camera"]:
		pic.erase(k)
	for k in KNOBS:
		d[k] = _knobs[k]
	# THE SHOW'S YOUTUBE TITLE AND DESCRIPTION: global values, the title overwritten as the episode
	# changes (as the seed is)
	if not _cur_title.is_empty():
		d["episode_title"] = _cur_title
	if not _cur_desc.is_empty():
		d["description"] = _cur_desc
	return d


## The document's block arrives at every read of it - on open, and at every Play and export
## (see [method DocSource.pull]) - so the episode is reopened only when the block names a
## DIFFERENT one. Reopening on every read stopped a generation in progress, and reopening reads the
## document again, which used to recurse.
##
## THE KNOBS ARE THE DOCUMENT'S, EXACTLY: the block is laid over the defaults, never over the
## knobs the panel held - those belong to whatever show was open before, and a show whose block
## does not name its `show` key would otherwise make its episodes in the last show's folder,
## with the last show's history handed to its producer as its own.
func _doc_apply(cfg: Dictionary) -> void:
	var was := "%s#%d" % [String(_knobs["show"]), int(_knobs["seed"])]
	_knobs = KNOBS.duplicate(true)
	_take_knobs(cfg)
	_cur_title = str(cfg.get("episode_title", ""))
	_cur_desc = str(cfg.get("description", ""))
	super._doc_apply(cfg)
	if _episode == null or "%s#%d" % [String(_knobs["show"]), int(_knobs["seed"])] != was:
		_open_episode()
	_refresh_show_fields()


## A document arrived (opened, or synced to). One that carries a `tarot:` block has had it
## applied already; one with none is a new show, and starts from the defaults rather than from
## the show before it. Nothing is written into it until something is changed.
func _on_show_opened(_path: String) -> void:
	if not _doc_has_block():
		_fresh_show()


## Detached from the document: the show it was is not this one any more.
func _on_doc_mode(sync: bool) -> void:
	if not sync:
		_fresh_show()


func _fresh_show() -> void:
	_knobs = KNOBS.duplicate(true)
	_cur_title = ""
	_cur_desc = ""
	_show_knobs()
	_open_episode()
	_refresh_show_fields()


## Whether the synced document carries this panel's block.
func _doc_has_block() -> bool:
	if _doc == null or not _doc.is_sync():
		return false
	var raw := FileAccess.get_file_as_string(_doc.doc_path())
	var res: Dictionary = FrontMatter.read_block(raw)
	return res.get("data") is Dictionary and (res["data"] as Dictionary).get(_section) is Dictionary


func _persist() -> void:
	super._persist()
	Settings.write(_section, "knobs", _knobs.duplicate(true))


## The knobs from the last session, for an unsynced brief; a synced show's own block replaces
## them as the document is read (inside the super call), and a synced show with no block starts
## from the defaults.
func _load_persisted() -> void:
	var k: Variant = Settings.read(_section, "knobs", {})
	if k is Dictionary:
		_take_knobs(k as Dictionary)
	super._load_persisted()
	if _doc.is_sync() and not _doc_has_block():
		_knobs = KNOBS.duplicate(true)
		_show_knobs()


## Knobs from a stored block or a document, each checked against [constant KNOBS].
func _take_knobs(src: Dictionary) -> void:
	for k in KNOBS:
		if not src.has(k):
			continue
		var v: Variant = src[k]
		match k:
			"seed":
				_knobs[k] = maxi(1, int(v))
			"cards":
				if v is Array and (v as Array).size() == 2:
					_knobs[k] = [clampi(int(v[0]), 1, 10), clampi(int(v[1]), 1, 10)]
			"reversals", "jumpers":
				_knobs[k] = bool(v)
			"writer":
				_knobs[k] = String(v) if TextGen.has(String(v)) else "claude"
			"painter":
				_knobs[k] = String(v) if ImageGen.REGISTRY.has(String(v)) else "codex"
			_:
				_knobs[k] = String(v)
	_show_knobs()


func _touch() -> void:
	_dirty = true
	_last_edit_ms = Time.get_ticks_msec()


## The cast rows, built as the Generative panel builds them (the voice machinery reads them), with
## Hesitate hidden: it lengthens marks the agents do not write. Turn stays - the rest where the
## reader hands a line to another of the show's voices (a familiar's quip) and takes it back.
func _build_cast(box: VBoxContainer) -> void:
	super._build_cast(box)
	var row := _hesitate.get_parent() as Control
	if row != null:
		row.visible = false


# --- the picture ---------------------------------------------------------------------------------

## THE TABLE'S PICTURE SETTINGS: the Look and the bookends. The Generative panel's medium picker,
## films, pictures, handwriting, scene hold, flourishes and camera are not built - the table is
## the only medium here, it cuts no scenes, and it is filmed from a tripod (a dial that moved it by
## a millimeter was a dial that did nothing).
func _build_picture(box: VBoxContainer) -> void:
	_build_filters(box)
	_sync_medium_rows()
	_intro = _director_slider(box, "Intro", Director.INTRO_MIN, Director.INTRO_MAX, 0.5,
		Director.intro_hold,
		"Seconds the table holds before the reader speaks, with the channel's name and the "
		+ "episode's title over it. The deck starts shuffling a beat before the first word.",
		func(v: float) -> void: Director.set_intro_hold(v))
	_outro = _director_slider(box, "Outro", Director.OUTRO_MIN, Director.OUTRO_MAX, 0.5,
		Director.outro_hold,
		"Seconds held after the last word, the spread on the table and the channel's name over "
		+ "it, fading picture and sound out together.",
		func(v: float) -> void: Director.set_outro_hold(v))


# --- the episode section -----------------------------------------------------------------------

func _build_source(box: VBoxContainer) -> void:
	super._build_source(box)
	_doc.opened.connect(_on_show_opened)
	_doc.mode_changed.connect(_on_doc_mode)
	box.add_child(HSeparator.new())
	var head := Label.new()
	head.text = "Episode"
	head.add_theme_font_size_override("font_size", 14)
	box.add_child(head)

	var erow := HBoxContainer.new()
	erow.add_theme_constant_override("separation", 6)
	box.add_child(erow)
	_episode_pick = OptionButton.new()
	_episode_pick.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	# SIZED BY THE ROW, NOT BY ITS ITEMS: an OptionButton is as wide as its longest item unless
	# told otherwise (clip_text alone does not do it), and an episode's title is a writer's - one
	# long title pushed the whole panel out to 1080 px
	_episode_pick.fit_to_longest_item = false
	_episode_pick.clip_text = true
	_episode_pick.focus_mode = Control.FOCUS_NONE
	_episode_pick.tooltip_text = ("Every episode this show has made, by seed. Picking one loads it "
		+ "exactly as it was made - same cards, same pictures, same words.")
	_episode_pick.item_selected.connect(func(i: int) -> void:
		if _syncing or i < 0 or i >= _episode_seeds.size():
			return
		_knobs["seed"] = int(_episode_seeds[i])
		_touch()
		_open_episode())
	erow.add_child(_episode_pick)
	_new_btn = Button.new()
	_new_btn.text = "New episode"
	_new_btn.tooltip_text = ("A fresh seed: a new plan, a new deck, a new reading - made when you press "
		+ "Generate, or a part at a time with ⟳.")
	_new_btn.pressed.connect(_new_episode)
	erow.add_child(_new_btn)

	var krow := HBoxContainer.new()
	krow.add_theme_constant_override("separation", 6)
	box.add_child(krow)
	var kl := Label.new()
	kl.text = "Cards"
	kl.add_theme_font_size_override("font_size", 12)
	krow.add_child(kl)
	_cards_lo = _spin(krow, "The fewest cards an episode draws.")
	var dash := Label.new()
	dash.text = "-"
	krow.add_child(dash)
	_cards_hi = _spin(krow, "The most cards an episode draws. Each seed picks its own count in this range.")
	# the two switches on a row of their own: beside the counts the row was wider than the panel
	var orow := HBoxContainer.new()
	orow.add_theme_constant_override("separation", 12)
	box.add_child(orow)
	_reversals = CheckBox.new()
	_reversals.text = "Reversals"
	_reversals.add_theme_font_size_override("font_size", 12)
	_reversals.tooltip_text = "Cards can come up upside down, and are read that way."
	_reversals.toggled.connect(func(on: bool) -> void:
		if _syncing:
			return
		_knobs["reversals"] = on
		_touch())
	orow.add_child(_reversals)
	_jumpers = CheckBox.new()
	_jumpers.text = "Jumpers"
	_jumpers.add_theme_font_size_override("font_size", 12)
	_jumpers.tooltip_text = "Now and then the first card flies out of the shuffle on its own."
	_jumpers.toggled.connect(func(on: bool) -> void:
		if _syncing:
			return
		_knobs["jumpers"] = on
		_touch())
	orow.add_child(_jumpers)

	# WHO WRITES AND WHO PAINTS: an agent CLI each, from the registries - the plan, the booklet and
	# the reading are words, the deck and the table are pictures
	var w := _agent_pick(box, "Writer", "writer", TextGen.REGISTRY, TextGen.LABELS, TextGen.BLURBS,
		func(k: String) -> bool: return TextGen.make(k).available(),
		"Who writes the plan, the booklet and the reading.",
		func(k: String) -> Array: return (TextGen.REGISTRY.get(k, TextGen.Claude) as GDScript).models(),
		"The model that writes. Default is the CLI's own choice per job (for Claude: Opus for the plan and the reading, Sonnet for the card designs); a model chosen here writes all of it.",
		func(k: String, m: String) -> Array: return (TextGen.REGISTRY.get(k, TextGen.Claude) as GDScript).efforts(m),
		("How hard the writer thinks before it writes - the CLI's own reasoning effort. Default is what "
		+ "each CLI does unasked (Claude: your Claude Code setting for the model; Codex: medium, low for "
		+ "the card designs; Amazon Nova: off - only Nova 2 Lite thinks, billed as output). A level "
		+ "chosen here is used for every job. Higher is slower and spends more quota."))
	_writer_pick = w[0]
	_writer_model = w[1]
	_writer_effort = w[2]
	var p := _agent_pick(box, "Painter", "painter", ImageGen.REGISTRY, ImageGen.LABELS, ImageGen.BLURBS,
		func(k: String) -> bool: return ImageGen.make(k).available(),
		"Who paints the card back, the cloth, the room and every card.",
		func(k: String) -> Array: return (ImageGen.REGISTRY.get(k, ImageGen.Codex) as GDScript).models(),
		"The model of the agent that asks for each picture. The picture itself is made by the CLI's own image tool whichever model asks.",
		func(k: String, m: String) -> Array: return (ImageGen.REGISTRY.get(k, ImageGen.Codex) as GDScript).efforts(m),
		("How hard the agent that asks for each picture thinks - not the picture itself. Default is "
		+ "low for Codex (its job is to call one tool); Bedrock's painter takes no setting."))
	_painter_pick = p[0]
	_painter_model = p[1]
	_painter_effort = p[2]

	var grow := HBoxContainer.new()
	grow.add_theme_constant_override("separation", 6)
	box.add_child(grow)
	_gen_btn = Button.new()
	_gen_btn.text = "Generate"
	_gen_btn.tooltip_text = ("Make whatever this episode is missing, in order: the plan, the shuffle, "
		+ "the deck, the pictures, and the reading - one card at a time, each written knowing only "
		+ "the cards already drawn. Spends the writer's and the painter's quota.")
	_gen_btn.pressed.connect(_generate)
	grow.add_child(_gen_btn)
	_halt_btn = Button.new()
	_halt_btn.text = "Stop"
	_halt_btn.tooltip_text = "Stop making this episode. Whatever is finished is kept."
	_halt_btn.pressed.connect(func() -> void:
		if _producer != null:
			_producer.stop())
	grow.add_child(_halt_btn)
	var folder := Button.new()
	folder.text = "Folder"
	folder.tooltip_text = "Open this episode's folder: every prompt exactly as sent, every reply, every picture."
	folder.pressed.connect(func() -> void:
		if _episode != null:
			DirAccess.make_dir_recursive_absolute(_episode.dir)
			OS.shell_open(_episode.dir))
	grow.add_child(folder)
	_del_btn = Button.new()
	_del_btn.text = "Delete…"
	_del_btn.tooltip_text = ("Delete the episode picked above: its plan, cards, pictures, reading and "
		+ "upload notes go to the system trash (restore it from there). Exported videos and the "
		+ "show's other episodes are not touched.")
	_del_btn.pressed.connect(_ask_delete)
	grow.add_child(_del_btn)

	_deck_note = Label.new()
	_deck_note.add_theme_font_size_override("font_size", 11)
	_deck_note.modulate = Color(1, 1, 1, 0.7)
	_deck_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_deck_note.tooltip_text = ("THE DECK IS THE SHOW'S: a `## Cards` section in the brief, one card per "
		+ "list item - `- XVI. The Tower: what it means` - with `###` headings for its suits. Without "
		+ "one, the show reads the standard 78. Each episode's cards are a shuffle of it, from a seed "
		+ "drawn from the system's own randomness; no agent ever picks a card.")
	_deck_note.mouse_filter = Control.MOUSE_FILTER_STOP
	box.add_child(_deck_note)
	_rows_box = VBoxContainer.new()
	_rows_box.add_theme_constant_override("separation", 1)
	box.add_child(_rows_box)
	_episode_note = Label.new()
	_episode_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_episode_note.add_theme_font_size_override("font_size", 11)
	_episode_note.modulate = Color(1, 1, 1, 0.7)
	box.add_child(_episode_note)
	_build_upload_fields(box)
	_show_knobs()


## A row choosing one agent from [param registry] for knob [param knob], and beside it the MODEL
## it runs (knob + "_model", from [param models_of] for the chosen agent - the list follows the
## agent). An agent whose CLI is not installed is shown as such and not choosable. Takes effect at
## the next Generate - a step already running finishes with what it started with. Returns
## [agent picker, model picker].
func _agent_pick(box: VBoxContainer, title: String, knob: String, registry: Dictionary, labels: Dictionary,
		blurbs: Dictionary, available: Callable, what: String, models_of: Callable, model_tip: String,
		efforts_of: Callable, effort_tip: String) -> Array:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	box.add_child(row)
	var l := Label.new()
	l.text = title
	l.custom_minimum_size = Vector2(56, 0)
	l.add_theme_font_size_override("font_size", 12)
	row.add_child(l)
	var pick := OptionButton.new()
	pick.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	pick.fit_to_longest_item = false
	pick.clip_text = true
	pick.focus_mode = Control.FOCUS_NONE
	var tips := PackedStringArray([what + " Takes effect at the next Generate."])
	var keys: Array = registry.keys()
	for i in keys.size():
		var k := String(keys[i])
		var here := bool(available.call(k))
		# the CLI's own name in the row (its full label is in the tooltip): the row is shared with the model
		pick.add_item(k.capitalize() + ("" if here else "  (not installed)"))
		pick.set_item_disabled(i, not here)
		tips.append("%s: %s" % [String(labels.get(k, k)), String(blurbs.get(k, ""))])
	pick.tooltip_text = "\n\n".join(tips)
	row.add_child(pick)
	var model := OptionButton.new()
	model.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	model.fit_to_longest_item = false
	model.clip_text = true
	model.focus_mode = Control.FOCUS_NONE
	model.tooltip_text = model_tip
	model.set_meta("models_of", models_of)
	model.set_meta("knob", knob + "_model")
	row.add_child(model)
	# THE REASONING EFFORT, from the chosen agent's own levels for the chosen model
	var effort := OptionButton.new()
	effort.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	effort.size_flags_stretch_ratio = 0.7
	effort.fit_to_longest_item = false
	effort.clip_text = true
	effort.focus_mode = Control.FOCUS_NONE
	effort.tooltip_text = effort_tip
	effort.set_meta("efforts_of", efforts_of)
	effort.set_meta("knob", knob + "_effort")
	effort.set_meta("agent_knob", knob)
	effort.item_selected.connect(func(i: int) -> void:
		if _syncing:
			return
		_knobs[knob + "_effort"] = String(effort.get_item_metadata(i))
		_touch())
	row.add_child(effort)
	model.item_selected.connect(func(i: int) -> void:
		if _syncing:
			return
		_knobs[knob + "_model"] = String(model.get_item_metadata(i))
		# another model, perhaps other levels: the chosen one stays, marked if it is not offered
		_fill_efforts(effort)
		_touch())
	pick.item_selected.connect(func(i: int) -> void:
		if _syncing:
			return
		_knobs[knob] = String(keys[i])
		# another agent, another list of models and levels: its own defaults until chosen
		_knobs[knob + "_model"] = ""
		_knobs[knob + "_effort"] = ""
		_fill_models(model, String(keys[i]))
		_fill_efforts(effort)
		_touch())
	return [pick, model, effort]


## [param pick]'s list of efforts for its agent's chosen model, the knob's choice selected - kept on
## the list (marked) when this agent or model does not offer it. Grayed out where the agent takes
## no effort setting.
func _fill_efforts(pick: OptionButton) -> void:
	var agent := String(_knobs.get(String(pick.get_meta("agent_knob")), ""))
	var model := String(_knobs.get(String(pick.get_meta("agent_knob")) + "_model", ""))
	var levels: Array = (pick.get_meta("efforts_of") as Callable).call(agent, model)
	var want := String(_knobs.get(String(pick.get_meta("knob")), ""))
	var was := _syncing
	_syncing = true
	pick.clear()
	var at := 0
	for e in levels:
		pick.add_item(String((e as Dictionary)["label"]))
		pick.set_item_metadata(pick.item_count - 1, String((e as Dictionary)["key"]))
		if String((e as Dictionary)["key"]) == want:
			at = pick.item_count - 1
	if not want.is_empty() and at == 0:
		pick.add_item("%s  (not offered)" % TextGen.effort_label(want))
		pick.set_item_metadata(pick.item_count - 1, want)
		at = pick.item_count - 1
	pick.select(at)
	pick.disabled = levels.size() <= 1 and want.is_empty()
	_syncing = was


## [param pick]'s list for [param agent]'s models, the knob's choice selected - kept on the list
## (marked) even when this agent no longer offers it, so a show's choice is never silently lost.
func _fill_models(pick: OptionButton, agent: String) -> void:
	var models: Array = (pick.get_meta("models_of") as Callable).call(agent)
	var want := String(_knobs.get(String(pick.get_meta("knob")), ""))
	var was := _syncing
	_syncing = true
	pick.clear()
	var at := 0
	for m in models:
		pick.add_item(String((m as Dictionary)["label"]))
		pick.set_item_metadata(pick.item_count - 1, String((m as Dictionary)["key"]))
		if String((m as Dictionary)["key"]) == want:
			at = pick.item_count - 1
	if not want.is_empty() and at == 0:
		pick.add_item("%s  (not offered)" % want)
		pick.set_item_metadata(pick.item_count - 1, want)
		at = pick.item_count - 1
	pick.select(at)
	_syncing = was


func _spin(row: HBoxContainer, tip: String) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = 1
	s.max_value = 10
	s.step = 1
	s.tooltip_text = tip
	s.custom_minimum_size = Vector2(56, 0)
	s.value_changed.connect(func(_v: float) -> void:
		if _syncing:
			return
		_knobs["cards"] = [int(_cards_lo.value), int(_cards_hi.value)]
		_touch())
	row.add_child(s)
	return s


func _show_knobs() -> void:
	if _cards_lo == null:
		return
	_syncing = true
	_cards_lo.value = int((_knobs["cards"] as Array)[0])
	_cards_hi.value = int((_knobs["cards"] as Array)[1])
	_reversals.button_pressed = bool(_knobs["reversals"])
	_jumpers.button_pressed = bool(_knobs["jumpers"])
	_writer_pick.select(maxi(0, TextGen.REGISTRY.keys().find(String(_knobs["writer"]))))
	_painter_pick.select(maxi(0, ImageGen.REGISTRY.keys().find(String(_knobs["painter"]))))
	_fill_models(_writer_model, String(_knobs["writer"]))
	_fill_models(_painter_model, String(_knobs["painter"]))
	_fill_efforts(_writer_effort)
	_fill_efforts(_painter_effort)
	_syncing = false


## The show's key: the title's, until the first episode is made - then FIXED in the document
## (see [method _generate]), because an episode's files are found by it and renaming the channel
## must not lose them. Not fixed earlier, or a new show would be keyed by the title it had at boot.
func _show_key() -> String:
	var k := String(_knobs["show"])
	return k if not k.is_empty() else TarotEpisode.slug(_show_title())


## Point the panel at the current seed's episode (made, partly made, or not at all).
func _open_episode() -> void:
	if _episode_pick == null:
		return
	if _producer != null and _producer.running:
		_producer.stop()
	_flush_upload_fields()      # an edit still waiting belongs to the episode being left
	var seed := int(_knobs["seed"])
	_episode = TarotEpisode.open(_show_key(), seed)
	# the spec is taken when Generate is pressed (see _generate) - never here, where reading the
	# document would land back in _doc_apply
	_producer = TarotProducer.new(_episode, {})
	_brief_changed = false
	_refill_episodes()
	_rebuild_rows()
	_fill_upload_fields(true)
	_mark_stale()


func _refill_episodes() -> void:
	_syncing = true
	_episode_pick.clear()
	_episode_seeds = []
	# NEWEST FIRST: seeds are random now, so their order says nothing; when an episode was made does
	var seen := {}
	var seeds: Array = []
	var cur := int(_knobs["seed"])
	for h in TarotEpisode.history(_show_key()):
		var s := int((h as Dictionary)["seed"])
		seen[s] = String(((h as Dictionary)["plan"] as Dictionary).get("episode_title", ""))
		seeds.append(s)
	if not seen.has(cur):
		seen[cur] = ""
		seeds.push_front(cur)
	for s in seeds:
		_episode_pick.add_item(_episode_label(int(s), String(seen[s])))
		_episode_seeds.append(int(s))
	_episode_pick.select(_episode_seeds.find(cur))
	_syncing = false


## An episode's entry in the picker: its seed, and its title once its plan has one.
static func _episode_label(seed: int, title: String) -> String:
	return "#%d  %s" % [seed, title if not title.is_empty() else "(not made yet)"]


## A NEW EPISODE IS A NEW SEED, drawn from the operating system's cryptographic randomness (see
## [method TarotDeck.true_seed]): the cards it deals are nobody's choice, and the seed is kept, so
## the episode can be made again exactly.
func _new_episode() -> void:
	var used := {}
	for h in TarotEpisode.history(_show_key()):
		used[int((h as Dictionary)["seed"])] = true
	var seed := TarotDeck.true_seed()
	while used.has(seed):
		seed = TarotDeck.true_seed()
	_knobs["seed"] = seed
	_touch()
	_open_episode()


func _generate() -> void:
	_run([])


## Make what is missing - all of it, or only [param steps] (see [member TarotProducer.only]).
func _run(steps: Array) -> void:
	if _episode == null:
		return
	if Settings.is_read_only():
		_episode_note.text = "This session is read-only (a render or a probe) - nothing is generated here."
		return
	if String(_knobs["show"]).is_empty():
		_knobs["show"] = _show_key()
		_touch()
		_open_episode()
	# the spec as it is NOW: the brief may have been edited since the episode was opened
	_producer.spec = _spec()
	_brief_changed = false
	_producer.start(steps)
	_refresh_rows()


## Asked first: an episode is minutes of the writer's and the painter's quota.
func _ask_delete() -> void:
	if _episode == null or not DirAccess.dir_exists_absolute(_episode.dir):
		return
	if _exporting(_episode):
		_set_status("Episode #%d is being exported - delete it once the export is done." % _episode.seed)
		return
	var plan: Variant = _episode.read_json("plan")
	var named := String((plan as Dictionary).get("episode_title", "")) if plan is Dictionary else ""
	var ask := ConfirmationDialog.new()
	ask.title = "Delete episode"
	ask.dialog_text = ("Delete episode #%d%s?\n\nIts plan, cards, pictures, reading and upload notes go to "
		+ "the system trash. Exported videos are not touched.") % [_episode.seed,
		("  \"%s\"" % named) if not named.is_empty() else ""]
	ask.dialog_autowrap = true
	ask.ok_button_text = "Delete"
	var which := _episode
	ask.confirmed.connect(func() -> void:
		ask.queue_free()
		_delete_episode(which))
	ask.canceled.connect(ask.queue_free)
	add_child(ask)
	ask.popup_centered(Vector2i(520, 0))


## DELETE [param ep]: whatever is making it stops, a reading of it stops (the table is showing its
## pictures), its folder goes to the trash, and the panel moves to the newest episode left - or to
## a fresh seed, not made yet, when none is.
func _delete_episode(ep: TarotEpisode) -> void:
	if ep == null or _exporting(ep):
		return
	var open := _episode != null and _episode.dir == ep.dir
	if open and _producer != null:
		_producer.stop()
	if _live_episode_dir() == ep.dir:
		_stop_speaking()
	var err := ep.trash()
	if not err.is_empty():
		_set_status("Could not delete episode #%d: %s" % [ep.seed, err])
		return
	print("ghost: tarot - episode %s #%d moved to the trash" % [ep.show, ep.seed])
	if open:
		var left := TarotEpisode.history(_show_key())
		_knobs["seed"] = int((left[0] as Dictionary)["seed"]) if not left.is_empty() else TarotDeck.true_seed()
		_touch()
		_open_episode()
	else:
		_refill_episodes()
	_set_status("Episode #%d is in the system trash." % ep.seed)


## Whether an export of [param ep] is under way (see [member _export_pin]).
func _exporting(ep: TarotEpisode) -> bool:
	return not _export_pin.is_empty() and (_export_pin["episode"] as TarotEpisode).dir == ep.dir


## The episode the reading on the stage is of, "" when none is playing.
func _live_episode_dir() -> String:
	if not _reading_live() or subtitles == null or not is_instance_valid(subtitles):
		return ""
	var d: Variant = subtitles.get("document")
	var t: Variant = (d as Dictionary).get("tarot", {}) if d is Dictionary else {}
	return String((t as Dictionary).get("dir", "")) if t is Dictionary else ""


func _redo(step: String) -> void:
	if _episode == null:
		return
	if _producer.running:
		_producer.stop()
	var gone := _episode.invalidate(step)
	print("ghost: tarot redo %s - removed %s" % [step, ", ".join(PackedStringArray(gone))])
	_rebuild_rows()
	_mark_stale()
	_run([step])


# --- the rows ------------------------------------------------------------------------------------

## One row per stage of the episode, each card its own; rebuilt when the card count changes.
var _row_widgets: Array = []      # [{label, steps, status: Label, text: Label}]
var _rows_for := -1

func _rebuild_rows() -> void:
	if _rows_box == null or _episode == null:
		return
	for c in _rows_box.get_children():
		c.queue_free()
	_row_widgets = []
	var n := _episode.card_count()
	_rows_for = n
	for r in ROWS:
		var label := String(r[0])
		if label == "Card":
			for k in range(1, n + 1):
				var steps: Array = []
				for s in r[1]:
					steps.append(String(s).replace("K", str(k)))
				_add_row("Card %d" % k, label, steps, k)
		else:
			_add_row(label, label, r[1], 0)
	_refresh_rows()


func _add_row(title: String, kind: String, steps: Array, k: int) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	_rows_box.add_child(row)
	var status := Label.new()
	status.custom_minimum_size = Vector2(18, 0)
	status.add_theme_font_size_override("font_size", 12)
	row.add_child(status)
	var name := Label.new()
	name.text = title
	name.custom_minimum_size = Vector2(56, 0)
	name.add_theme_font_size_override("font_size", 12)
	row.add_child(name)
	var text := Label.new()
	text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	text.clip_text = true
	text.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	text.add_theme_font_size_override("font_size", 11)
	text.modulate = Color(1, 1, 1, 0.7)
	text.mouse_filter = Control.MOUSE_FILTER_STOP
	row.add_child(text)
	var redo := MenuButton.new()
	redo.text = "⟳"
	redo.flat = true
	redo.tooltip_text = ("Make this part again - only this part. What was made from it is cleared "
		+ "and waits for Generate.")
	var menu := redo.get_popup()
	var items: Array = REDO.get(kind, [])
	for i in items.size():
		menu.add_item(String(items[i][0]), i)
	menu.id_pressed.connect(func(id: int) -> void:
		_redo(String(items[id][1]).replace("K", str(k))))
	redo.modulate.a = 1.0 if not items.is_empty() else 0.0
	redo.disabled = items.is_empty()
	row.add_child(redo)
	_row_widgets.append({"steps": steps, "status": status, "text": text, "kind": kind, "k": k})


## What the set dresser put on the picked episode's table, by name - read again only when the file
## changes (the rows refresh four times a second).
func _table_things() -> PackedStringArray:
	var path := _episode.file_of("table") if _episode != null else ""
	var mt := FileAccess.get_modified_time(path) if not path.is_empty() and FileAccess.file_exists(path) else -1
	if mt != int(_table_seen.get("mt", -2)) or path != String(_table_seen.get("path", "")):
		var names := PackedStringArray()
		var t: Variant = _episode.read_json("table") if mt >= 0 else null
		if t is Dictionary and (t as Dictionary).get("things") is Array:
			for th in (t as Dictionary)["things"]:
				if th is Dictionary:
					names.append(String((th as Dictionary).get("name", "a thing")))
		_table_seen = {"mt": mt, "path": path, "names": names}
	return _table_seen["names"]


## WHAT THE SET DRESSER IS DOING while it sets the table with its tools: its last call, from the
## job's `tools.jsonl` (read again only when that file changes) - a run of many minutes otherwise
## shows nothing but its glyph. "" before its first call, or for a set dresser with no tools.
func _table_progress() -> String:
	var path := _episode.job_dir("table").path_join("tools.jsonl") if _episode != null else ""
	var mt := FileAccess.get_modified_time(path) if not path.is_empty() and FileAccess.file_exists(path) else -1
	if mt != int(_table_log.get("mt", -2)) or path != String(_table_log.get("path", "")):
		var lines := FileAccess.get_file_as_string(path).strip_edges().split("\n", false) if mt >= 0 else PackedStringArray()
		var last: Variant = JSON.parse_string(String(lines[lines.size() - 1])) if not lines.is_empty() else null
		var said := ""
		if last is Dictionary:
			said = "%s (call %d): %s" % [String((last as Dictionary).get("tool", "")), lines.size(),
				String((last as Dictionary).get("text", "")).get_slice("\n", 0).left(90)]
		_table_log = {"mt": mt, "path": path, "said": said}
	return String(_table_log.get("said", ""))


func _refresh_rows() -> void:
	if _episode == null or _rows_box == null:
		return
	var plan: Variant = _episode.read_json("plan")
	# THE PICKER FOLLOWS THE EPISODE: its entries are written when the list is filled, so an episode
	# made while it was open went on saying "(not made yet)" until another was picked and back
	var at := _episode_seeds.find(int(_knobs["seed"]))
	if at >= 0:
		var label := _episode_label(int(_knobs["seed"]),
			String((plan as Dictionary).get("episode_title", "")) if plan is Dictionary else "")
		if _episode_pick.get_item_text(at) != label:
			_episode_pick.set_item_text(at, label)
	if _episode.card_count() != _rows_for:
		_rebuild_rows()
		return
	var doc := {}
	for w in _row_widgets:
		var states := {}
		var why := ""
		for s in (w as Dictionary)["steps"]:
			var st := _producer.state_of(String(s))
			states[st] = true
			if st == "failed" and why.is_empty():
				why = _producer.error_of(String(s))
		var glyph := "·"
		if states.has("failed"):
			glyph = "✗"
		elif states.has("running") or states.has("queued"):
			glyph = "◌"
		elif states.size() == 1 and states.has("ready"):
			glyph = "✓"
		(w["status"] as Label).text = glyph
		var text := ""
		match String(w["kind"]):
			"Plan":
				text = String((plan as Dictionary).get("episode_title", "")) if plan is Dictionary else ""
			"Shuffle":
				var n := _episode.card_count()
				text = "%d cards" % n if _episode.has("draw") else ""
			"Table":
				var have := PackedStringArray()
				for s in ["back", "surface", "backdrop"]:
					if _episode.has("image:" + s):
						have.append({"back": "back", "surface": "cloth", "backdrop": "room"}[s])
				var things := _table_things()
				if not things.is_empty():
					have.append("%d thing%s" % [things.size(), "" if things.size() == 1 else "s"])
				text = ", ".join(have)
				if _producer.state_of("table") == "running" and not _table_progress().is_empty():
					text = "setting the table - " + _table_progress()
			"Card":
				if doc.is_empty():
					doc = _episode.document()
				var cards: Array = doc.get("cards", [])
				var k := int(w["k"])
				if k - 1 < cards.size():
					var c: Dictionary = cards[k - 1]
					text = "%s%s%s" % [String(c.get("name", "")), " (reversed)" if bool(c.get("reversed", false)) else "",
						" - jumper" if bool(c.get("jumper", false)) else ""]
			"Intro", "Close":
				var step := "say:intro" if String(w["kind"]) == "Intro" else "say:close"
				var said := _episode.read_text(step) if _episode.has(step) else ""
				text = "%d words" % said.split(" ", false).size() if not said.is_empty() else ""
		if not why.is_empty():
			text = why
		(w["text"] as Label).text = text
		(w["text"] as Label).tooltip_text = text
		if String(w["kind"]) == "Table" and why.is_empty() and not _table_things().is_empty():
			(w["text"] as Label).tooltip_text = "On the table:\n" + "\n".join(_table_things())
	# the deck, recounted whenever the brief changes
	if _text != null and _text.get_version() != _deck_seen and _deck_note != null:
		_deck_seen = _text.get_version()
		var own := TarotDeck.parse(_text.text)
		_deck_note.text = ("Deck: %d cards, from the brief's Cards." % own.size()) if not own.is_empty() \
			else "Deck: the standard 78 (the brief defines no Cards)."
	var busy := _producer.running or _producer.busy()
	_gen_btn.disabled = busy or _episode.complete()
	_halt_btn.disabled = not busy
	_del_btn.disabled = not DirAccess.dir_exists_absolute(_episode.dir) or _exporting(_episode)
	var note := ""
	if _episode.complete():
		note = "Ready - press Play to watch it, or export it."
	elif busy:
		note = "Making episode #%d…" % _episode.seed
	elif _episode.has("plan"):
		note = "Partly made - Generate finishes it."
	else:
		note = "Not made yet - Generate makes it."
	if _brief_changed and _episode.has("plan"):
		note += " The brief has changed since this episode was made; New episode uses it."
	_episode_note.text = note
	_fill_upload_fields()


# --- the YouTube fields ------------------------------------------------------------------------

## WHAT THE EPISODE GOES UP TO YOUTUBE AS. The title and description are the episode's - its plan's,
## edited here and written straight back into the plan, so the export's file name, the episode picker
## and `upload.md` all say what the panel says. THE TAGS ARE THE SHOW'S: its document's own `tags:`
## line, the way North Star's chapters keep theirs, edited here as chips and going up with every
## episode. Nothing of any one episode is kept in the document but the picked one's title, overwritten
## as the seed is (the user, 2026-10-06: "I just don't want to store a bunch of episode-specific logic
## in the frontmatter of a markdown file").
func _build_upload_fields(box: VBoxContainer) -> void:
	box.add_child(HSeparator.new())
	var head := Label.new()
	head.text = "YouTube"
	head.add_theme_font_size_override("font_size", 14)
	head.tooltip_text = ("What this episode goes up as when the export's \"Upload to YouTube\" is ticked. "
		+ "The producer writes the title and description with the plan, and an edit here is saved into "
		+ "the plan; the tags are the show's own, kept in its document. Uploads are unlisted.")
	head.mouse_filter = Control.MOUSE_FILTER_STOP
	box.add_child(head)
	_yt_title = LineEdit.new()
	_yt_title.max_length = YouTube.TITLE_MAX
	_yt_title.placeholder_text = "Title"
	_yt_title.tooltip_text = "The video's title, at most 100 characters. It names the exported file too."
	_yt_title.text_changed.connect(func(_t: String) -> void: _upload_edited())
	box.add_child(_yt_title)
	_yt_desc = TextEdit.new()
	_yt_desc.placeholder_text = "Description"
	_yt_desc.tooltip_text = ("The video's description. A chapter per card, timed from the export itself, "
		+ "is added under it as the video goes up.")
	_yt_desc.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_yt_desc.custom_minimum_size = Vector2(0, 120)
	_yt_desc.text_changed.connect(_upload_edited)
	box.add_child(_yt_desc)
	_yt_tags = TagField.new()
	_yt_tags.tooltip_text = ("The show's tags - its document's `tags:` line - going up with every episode: "
		+ "× removes one; type in the box and a comma (or Enter) makes it a tag. YouTube takes 500 "
		+ "characters of tags in all; a dimmed tag is past that.")
	_yt_tags.changed.connect(_save_tags)
	box.add_child(_yt_tags)
	_yt_note = Label.new()
	_yt_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_yt_note.add_theme_font_size_override("font_size", 11)
	_yt_note.modulate = Color(1, 1, 1, 0.7)
	box.add_child(_yt_note)


func _upload_edited() -> void:
	if _syncing:
		return
	_yt_dirty = 0.8
	_refresh_upload_note()


func _upload_focused() -> bool:
	return _yt_title != null and (_yt_title.has_focus() or _yt_desc.has_focus())


static func _mtime(path: String) -> int:
	return FileAccess.get_modified_time(path) if not path.is_empty() and FileAccess.file_exists(path) else -1


## THE FIELDS SHOW THE SHOW'S GLOBAL VALUES from its document at once - on open, on New episode, on
## any pick (the user: "it should have been grabbed immediately, since that stuff is global"): the
## description and the tags as the show keeps them, and the title the picked episode overwrites
## when it has a plan (as the seed is overwritten) - a new episode shows the last until its plan
## gives its own. Refilled when the plan changes on disk unless one is being typed in, or at once
## with [param force]. A new upload refreshes the note.
func _fill_upload_fields(force := false) -> void:
	if _yt_title == null:
		return
	var path := _episode.file_of("plan") if _episode != null else ""
	var mt := _mtime(path)
	var ups := _mtime(_episode.file_of("youtube") if _episode != null else "")
	if ups != int(_yt_seen.get("ups", -2)):
		_yt_seen["ups"] = ups
		_refresh_upload_note()
	var same := mt == int(_yt_seen.get("mt", -2)) and path == String(_yt_seen.get("path", ""))
	if not force and (same or _yt_dirty > 0.0 or _upload_focused()):
		return
	_yt_seen["path"] = path
	_yt_seen["mt"] = mt
	if mt >= 0:
		var t := String(_episode.upload_notes()["title"])
		if not t.is_empty():
			_cur_title = t
	var was := _syncing
	_syncing = true
	if _yt_title.text != _cur_title:
		_yt_title.text = _cur_title
	if _yt_desc.text != _cur_desc:
		_yt_desc.text = _cur_desc
	_syncing = was
	# the title is the episode's (an edit is written into its plan), so it waits for a plan
	_yt_title.editable = mt >= 0
	_yt_desc.editable = true
	_refresh_show_fields()


## The show's tags from its document, unless they are being typed in.
func _refresh_show_fields() -> void:
	if _yt_tags != null and not _yt_tags.typing():
		_yt_tags.set_tags(_show_tags())
	_refresh_upload_note()


## A SHOW THAT HAS NO DESCRIPTION OR TAGS YET TAKES THE PICKED EPISODE'S, once, so the section is
## never empty for want of a first one: the producer writes both with every plan. After that they
## are the show's, edited here. Run from the frame loop, not on open, so the document's autosave
## sees the change (the snapshot a just-opened document is compared with is taken after it opens);
## never in a process that may not write (a probe).
func _seed_globals() -> void:
	if not Settings.is_read_only():
		_seed_from_plan()


func _seed_from_plan() -> void:
	if _episode == null or not _episode.has("plan") or _doc == null:
		return
	if _cur_desc.is_empty() and _yt_dirty <= 0.0 and not _yt_desc.has_focus():
		var d := String(_episode.upload_notes()["description"]).strip_edges()
		if not d.is_empty():
			_cur_desc = d
			_fill_upload_fields(true)
	if _show_tags().is_empty() and not _yt_tags.typing():
		var tags: PackedStringArray = _episode.upload_notes()["tags"]
		if not tags.is_empty():
			_doc.set_field("tags", ", ".join(tags))
			_refresh_show_fields()


## The edited title and description - only when something was edited (with [param force], when the
## edit's wait is over) and only what changed. The description is the show's (into its document);
## the title is the picked episode's, written into its plan too, so the export's file name and the
## episode picker follow. An emptied title is not written: a video needs one.
func _flush_upload_fields(force := false) -> void:
	if not force and _yt_dirty <= 0.0:
		return
	_yt_dirty = 0.0
	if _yt_title == null:
		return
	_cur_desc = _yt_desc.text.strip_edges()
	var title := _yt_title.text.strip_edges()
	var plan: Variant = _episode.read_json("plan") if _episode != null else null
	if title.is_empty() or not (plan is Dictionary):
		_refresh_upload_note()
		return
	_cur_title = title
	var p: Dictionary = plan
	if title != String(p.get("episode_title", "")):
		p["episode_title"] = title
		var err := _episode.write_json("plan", p)
		if not err.is_empty():
			_set_status("The title could not be saved into the episode's plan: " + err)
			return
		_yt_seen["path"] = _episode.file_of("plan")
		_yt_seen["mt"] = _mtime(_episode.file_of("plan"))
	_refresh_upload_note()


## A tag added or removed: the show's `tags:` line, written at once.
func _save_tags() -> void:
	if _syncing or _doc == null:
		return
	if not _doc.set_field("tags", ", ".join(_yt_tags.get_tags())):
		_set_status("The tags could not be saved into the show's document.")
	_refresh_upload_note()


## Under the fields: what goes up as tags against YouTube's limit, and where the episode already is
## on YouTube. A tag past the limit is dimmed in the field, saying so.
func _refresh_upload_note() -> void:
	if _yt_note == null:
		return
	var own := _yt_tags.get_tags() if _yt_tags != null else PackedStringArray()
	var fitted := YouTube.fit_tags(Array(own))
	var going := {}
	for t in fitted:
		going[t.to_lower()] = true
	var reasons := {}
	for t in own:
		var c := YouTube.clean_tag(t).to_lower()
		if c.is_empty():
			reasons[t] = "nothing is left of it once cleaned"
		elif not going.has(c):
			reasons[t] = "YouTube takes %d characters of tags, and they are used up" % YouTube.TAGS_MAX
	if _yt_tags != null:
		_yt_tags.mark(reasons)
	if _episode == null or not _episode.has("plan"):
		_yt_note.text = "Generate the plan first: the producer writes the episode's title."
		return
	var t := "Goes up as \"%s\"" % YouTube.fit_title(upload_title(_doc.field("title"), _cur_title))
	t += (", with the show's %d tag%s, %d of YouTube's %d characters." % [fitted.size(),
		"" if fitted.size() == 1 else "s", YouTube.tags_length(fitted), YouTube.TAGS_MAX]) if not fitted.is_empty() \
		else ". No tags yet: the ones added here are the show's, and go up with every episode."
	var ups := _episode.uploads()
	if not ups.is_empty():
		var last: Dictionary = ups[ups.size() - 1]
		t += " On YouTube: %s (%s)." % [String(last.get("url", "")), String(last.get("privacy", ""))]
	_yt_note.text = t
