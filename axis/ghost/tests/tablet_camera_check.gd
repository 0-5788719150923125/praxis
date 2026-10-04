extends Node

## THE TABLET'S CAMERA HOLDS WHILE THE PAGE MOVES: a drag that scrolls the reading must not send
## the camera after the text.
##
##   tests/run_boot_probe.sh tests/tablet_camera_check.gd 240
##
## THE SCROLL is held here too, since the camera's holding depends on it: a reader swipes a
## paragraph up BEFORE reading it, so the reading stays inside the band and never runs to the
## screen's foot; a page whose first paragraph is already in a good place is not swiped on
## arrival (it hid the header); a picture a skim stops on is centered where the camera looks; and
## a reading started mid-way (a scrub) opens with the page already where the reading is, nothing
## being replayed. And THE INTRO IS A WAIT: the tablet lies dark through it, the hand starts when
## it ends, and the first word comes after the whole opening run.
##
## Reported as "the page is swiped, and the zoomed-in camera has to pan much higher to compensate
## and reorient itself". The camera leaned toward the line being read, and a drag moves that line
## from the foot of the reading band to its head, so every drag was answered by the camera
## swinging up the screen - measured on ch42, 108-152 px within 12 s of every drag.
##
## Two halves. GEOMETRY: at its closest, at every Camera setting and both ways round, the camera
## sees the whole reading band ([constant TabletMedium.READ_BAND]) with room to spare - which is
## what lets the camera look at the band's middle and never follow the line. MOTION: a long page
## is read by a synthetic voice and [TabletAudit] follows every reading drag; the camera's aim
## must stay within REDIRECT_MAX of where it was, and every spoken word on the viewport and in
## frame. A page that never scrolls passes that trivially, so the run must also HAVE drags, made
## while the camera was close in.

const Audit := preload("res://tests/tablet_audit.gd")
const W := 1280
const H := 720
const DT := 1.0 / 30.0
## Real frames are awaited only every so many steps: the medium and the subtitles are stepped
## by hand, and awaiting every step makes a long page take minutes.
const FRAME_EVERY := 30
## How far, in logical pixels, the aim may move after a reading drag: the twist and the lens
## breathe on their own, a little.
const REDIRECT_MAX := 20.0
## Of the camera's closest view above and below its aim, the share the band may fill.
const VIEW_SHARE := 0.9
## How far down the viewport a word may be read: a reader swipes before the text gets near the
## foot of the screen. Absolute, not the band's own foot, so a band moved back down fails it.
const READ_FOOT_MAX := 0.75

var _fail := 0


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_check_geometry()
	_check_table_columns()
	_check_site_name_once()
	await _check_motion()
	await _check_scrub()
	await _check_intro()
	print("tablet_camera_check: %s" % ("PASS" if _fail == 0 else "FAIL (%d)" % _fail))
	get_tree().quit(1 if _fail > 0 else 0)


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fail += 1
		print("  FAIL: ", what)
	else:
		print("  ok: ", what)


## A table column is never narrower than its widest word: a short column ("Open", "Filled")
## under a long header ("Status") ran the header into the next one.
func _check_table_columns() -> void:
	var src := "<!-- url: jobs.test -->\n\n## Open positions\n\n| Position | Status | Details |\n|---|---|---|\n" \
		+ "| Architect | Filled | The successful candidate drew houses until they were no longer houses, and still does. |\n" \
		+ "| Facilitator of meaning | Open | Must be comfortable signing papers without knowing what they say. |\n"
	var doc := TabletScript.parse(src)
	for o in [0, 1]:
		var lp := TabletPage.new()
		lp.build(doc, 0, TabletMedium.logical(o).x, 1000.0)
		var t: Dictionary = {}
		for b in (doc["pages"][0] as Dictionary)["blocks"]:
			if String((b as Dictionary)["kind"]) == "table":
				t = b
		var head: Array = t.get("head_cells", [])
		var clear := head.size() == 3
		for k in range(head.size() - 1):
			var right := -INF
			for wi in (head[k] as PackedInt32Array):
				right = maxf(right, (lp.word_rect[wi] as Rect2).end.x)
			var left := INF
			for wi in (head[k + 1] as PackedInt32Array):
				left = minf(left, (lp.word_rect[wi] as Rect2).position.x)
			clear = clear and right <= left
		_ok(clear, "%s: no table header runs into the next column" % ("portrait" if o == 0 else "landscape"))


## A later page on a site that opens on the site's own name: the band already says it, so the
## title is not set again - "Omnipedia" sat twice at the top of every article after the first.
func _check_site_name_once() -> void:
	var src := "<!-- url: omni.test/a -->\n\n# Omnipedia\n\n# First article\n\nRead this. [Next](omni.test/b)\n\n" \
		+ "<!-- url: omni.test/b -->\n\n# Omnipedia\n\n# Second article\n\nRead this too.\n"
	var doc := TabletScript.parse(src)
	var lp := TabletPage.new()
	lp.build(doc, 1, 1200.0, 1000.0)
	var b0: Dictionary = (doc["pages"][1] as Dictionary)["blocks"][0]
	var set := 0
	for wi in (b0["words"] as PackedInt32Array):
		if lp.word_rect.has(wi):
			set += 1
	_ok(set == 0 and lp.title == "Second article",
		"a later page's opening site name is the band, not a second title (%d words set, tab '%s')" % [set, lp.title])


func _check_geometry() -> void:
	var band := TabletMedium.READ_BAND
	var mid := (band.x + band.y) * 0.5
	for sev in [Director.CAMERA_MIN, 1.0, Director.CAMERA_MAX]:
		for o in [0, 1]:
			var vh := TabletMedium.logical(o).y - TabletMedium.TOP
			var cv := TabletMedium.close_view(o, sev)
			var above := (mid - band.x) * vh
			var below := (band.y - mid) * vh
			_ok(above <= cv.x * VIEW_SHARE and below <= cv.y * VIEW_SHARE,
				"camera %.1f, %s: the band (%.0f px above the aim, %.0f below) is inside the closest view (%.0f, %.0f)"
				% [sev, "portrait" if o == 0 else "landscape", above, below, cv.x, cv.y])


## Read [param body] with a synthetic voice - from the start, or from [param from] as a scrub
## would - and return the audit.
func _read(body: String, from := "", intro := 1.0) -> RefCounted:
	var stage := SubViewport.new()
	stage.size = Vector2i(W, H)
	stage.own_world_3d = true
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)
	Director.detach()
	Director.camera = 1.0
	var medium: Medium = Medium.make("tablet")
	medium.mount(stage)
	Director.attach(stage, medium)
	Director.hold(true)
	Director.intro_hold = intro        # the Director reloads its own at attach
	var subs: Subtitles = preload("res://scripts/subtitles.gd").new()
	var scrub: Dictionary = Audit.scrub_to(body, from) if not from.is_empty() else {"si": 0}
	subs.words = Audit.timeline(body, 0.3, Director.intro_hold, int(scrub["si"]))
	subs.document = {"source": body, "title": "Camera"}
	if int(scrub["si"]) > 0:
		subs.document["start_words"] = scrub["words"]
		print("tablet_camera_check: a scrub to spoken word %d, '%s'" % [int(scrub["si"]), " ".join(scrub["words"])])
	add_child(subs)
	subs.process_mode = Node.PROCESS_MODE_DISABLED     # stepped below, once per step
	if medium.bind_captions(subs):
		subs.overlay_hidden = true
	var tab := medium as TabletMedium
	var audit = Audit.new(Vector2(W, H))
	var end := float((subs.words.back() as Dictionary)["t1"]) + 1.0
	var t := 0.0
	var step := 0
	Spectrum.virtual_clock = 0.0
	while t < end:
		t += DT
		step += 1
		Spectrum.virtual_clock = t
		Spectrum.current.time = t
		subs._process(DT)
		medium.advance(Spectrum.current, DT, 1.0)
		if step % FRAME_EVERY == 0:
			await get_tree().process_frame
		audit.count(audit.frame(tab), tab, DT)
	audit.finish()
	Spectrum.virtual_clock = -1.0
	Director.hold(false)
	Director.detach()
	subs.queue_free()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame
	return audit


func _check_motion() -> void:
	var audit = await _read(_long_page())
	print("tablet_camera_check: " + audit.summary())
	# CLOSE IN for the whole watch: within a sixth of the way out from the closest distance - the
	# page's arc moves the aim as it closes in and lets go, and that is not a drag being answered
	var near := TabletMedium.near_of(0.0, 1.0)
	var close_d := near + (TabletMedium.wide_of(0.0) - near) / 6.0
	var close := 0
	var worst := 0.0
	for i in audit.redirects.size():
		if audit.drag_dist[i] < close_d and audit.drag_far[i] < close_d:
			close += 1
			worst = maxf(worst, audit.redirects[i])
	_ok(close >= 2, "the page scrolled under the reading while the camera was close in (%d drags)" % close)
	_ok(worst <= REDIRECT_MAX, "the camera held through every drag (aim moved at most %.0f px)" % worst)
	_ok(audit.read_s > 60.0, "the page was read (%.0f s)" % audit.read_s)
	_ok(audit.off_viewport_s == 0.0, "no word was spoken off the viewport (%.1f s)" % audit.off_viewport_s)
	_ok(audit.out_of_frame_s == 0.0 and audit.edge_s == 0.0,
		"no word was spoken out of frame or at its edge (%.1f s, %.1f s)" % [audit.out_of_frame_s, audit.edge_s])
	_ok(audit.lowest <= READ_FOOT_MAX, "the reading stayed off the screen's foot (lowest %.2f of the viewport)" % audit.lowest)
	var first: Dictionary = audit.first_heard.get(0, {})
	_ok(not first.is_empty() and float(first["scroll"]) == 0.0,
		"the page opened where its first paragraph already was - no swipe on arrival (scroll %s)" % str(first.get("scroll", "?")))
	_ok(audit.pictures.size() == 1, "the skim stopped on the picture (%d)" % audit.pictures.size())
	for pc in audit.pictures:
		_ok(absf(float(pc["mid"]) - 0.5) < 0.08,
			"the picture was centered where the camera looks (middle %.2f of the frame)" % float(pc["mid"]))


func _check_scrub() -> void:
	var audit = await _read(_long_page(), "Sentence two five of the long")
	print("tablet_camera_check: from a scrub - " + audit.summary())
	var first: Dictionary = audit.first_heard.get(0, {})
	_ok(not first.is_empty() and not bool(first["busy"]),
		"a scrub opens with nothing being replayed (busy %s)" % str(first.get("busy", "?")))
	_ok(not first.is_empty() and float(first["vp_y"]) >= 0.0 and float(first["vp_y"]) <= TabletMedium.READ_BAND.y,
		"a scrub opens with the page already where the reading is (first word %.2f down the viewport)"
		% float(first.get("vp_y", -1.0)))
	_ok(audit.off_viewport_s == 0.0 and audit.lowest <= READ_FOOT_MAX,
		"after a scrub nothing is read off the viewport or near its foot (%.1f s, lowest %.2f)" % [
			audit.off_viewport_s, audit.lowest])
	_ok(audit.pictures.is_empty(), "the picture before the scrub point is not shown again (%d)" % audit.pictures.size())


func _check_intro() -> void:
	const INTRO := 5.0
	var body := "---\ntitle: Intro\n---\n\n<!-- url: intro.test -->\n\n# Intro\n\nOne line, read once the tablet is awake.\n"
	var audit = await _read(body, "", INTRO)
	var run := 0.0
	for a in TabletScript.parse(body)["actions"]:
		run += float(a["dur"])
	print("tablet_camera_check: intro %.1f s, opening run %.1f s - the screen lit at %.2f s, the first word at %.2f s"
		% [INTRO, run, audit.woke_at, audit.first_word_at])
	_ok(audit.woke_at >= INTRO, "the tablet lay dark through the intro (lit at %.2f s)" % audit.woke_at)
	_ok(audit.first_word_at >= INTRO + run * 0.95,
		"the first word came after the intro and the whole opening run (%.2f s)" % audit.first_word_at)


## One page, long enough to scroll twice under a close camera.
func _long_page() -> String:
	# a title and a skipped byline - the skim on arrival, which must not move a page whose first
	# paragraph is already in place - and a picture between two paragraphs, which it must center
	var out := "---\ntitle: Camera\n---\n\n<!-- url: long.test -->\n\n# The Long Read\n\n"
	out += "<!-- skip -->\nBy the camera desk, with a byline long enough to be skimmed past rather than read.\n\n"
	var n := 0
	for para in 16:
		var lines := PackedStringArray()
		for s in 4:
			n += 1
			lines.append("Sentence %s of the long read goes on for a while, so the page has to move." % _num(n))
		out += " ".join(lines) + "\n\n"
		if para == 2:
			out += "<!-- image: a test picture for the camera check -->\n\n"
	return out


func _num(n: int) -> String:
	const ONES := ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
	return " ".join(Array(str(n).split("")).map(func(c): return ONES[int(c)]))
