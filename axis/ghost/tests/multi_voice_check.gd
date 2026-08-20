extends Node

## Gate for SPEAKING IN MORE THAN ONE VOICE - the Generative panel's tabs, and the
## speaker cues in the script that hand a passage to one of them.
##
## Every claim here fails SILENTLY if it breaks, which is the only reason this file
## exists. A cue that is not recognised does not error; it reads the second half of
## the chapter in the narrator's voice, and the first person to notice is whoever
## listens to forty minutes of it. A cue recognised where none was meant does the
## same thing in reverse. And a per-tab setting that never reaches the request is a
## dial that appears to work - it moves, it saves, it reloads - and changes nothing.
##
## Needs a real boot: [Phonemes] and the panel's own Director-backed sliders are
## autoload-bound.
##   tests/run_boot_probe.sh tests/multi_voice_check.gd 120

var _fails: Array = []
var _ed: GenerativeEditor


func _ready() -> void:
	# Built by hand rather than added to the tree: _ready would start the voice
	# host, and nothing here needs a python process.
	_ed = GenerativeEditor.new()
	_ed._build_panel()
	# THE PANEL, HOWEVER, MUST BE IN THE TREE. Range only emits value_changed for
	# an owner that is inside one (Range::Shared::emit_value_changed skips the
	# rest), so a panel built outside it has every slider callback silently dead -
	# and a gate written against that would pass whatever the callbacks did.
	_ed.remove_child(_ed._panel)
	add_child(_ed._panel)
	_ed.remove_child(_ed._repace_timer)
	add_child(_ed._repace_timer)
	_check_cues()
	_check_comments_never_spoken()
	_check_frontmatter()
	_check_macros()
	_check_subtitles_show_the_source()
	_check_chunks_carry_their_voice()
	_check_settings_reach_the_request()
	_check_room_is_per_voice()
	_check_silent_tabs_stay_silent()
	_check_turn_rest()
	_check_handover_is_sample_accurate()
	_check_tabs()
	_check_saved_shape()
	_ed.free()
	if _fails.is_empty():
		print("multi_voice_check: ALL OK")
		get_tree().quit()
		return
	for f in _fails:
		print("multi_voice_check: FAIL - ", f)
	print("multi_voice_check: %d FAILED" % _fails.size())
	get_tree().quit(1)


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fails.append(what)


func _slots(n: int) -> void:
	_ed._slots = []
	for i in n:
		_ed._slots.append(GenerativeEditor.SLOT_DEFAULTS.duplicate())
	_ed._slot = 0


## THE CUES. Both halves matter and the second one more: the HOLDS are lines that
## look like a cue and are prose, and every one of them would be swallowed whole
## and read in the wrong voice from there to the end of the chapter.
func _check_cues() -> void:
	_slots(3)
	var segs: Array = _ed._split_speakers(
		"Narrator opens.\n\n<!-- speaker: 2 -->\n\nI am the spider.\n\n"
		+ "<!-- speaker: 1 -->\n\nThe pen comes back.\n\n[speaker: 3]\n\nNo Diddy.")
	_ok(segs.size() == 4, "four passages, got %d" % segs.size())
	if segs.size() == 4:
		_ok(int(segs[0]["slot"]) == 0 and int(segs[1]["slot"]) == 1
			and int(segs[2]["slot"]) == 0 and int(segs[3]["slot"]) == 2,
			"passages went to voices %s" % [[segs[0]["slot"], segs[1]["slot"],
				segs[2]["slot"], segs[3]["slot"]]])
		_ok(String(segs[1]["text"]).begins_with("I am the spider"),
			"the second passage is %s" % JSON.stringify(segs[1]["text"]))

	# no cues at all is the case every script had before today
	_slots(1)
	var plain: Array = _ed._split_speakers("Just prose.\nMore of it.")
	_ok(plain.size() == 1 and int(plain[0]["slot"]) == 0, "an uncued script is one passage")

	# HOLDS: prose that mentions a speaker, and a cue that is not alone on its line
	for hold in ["speaker: 2 is what he said.", "He said <!-- speaker: 2 --> aloud.",
			"The speaker: 2 of them, in fact."]:
		_slots(3)
		var h: Array = _ed._split_speakers("Before.\n%s\nAfter." % hold)
		_ok(h.size() == 1, "prose taken for a cue: %s" % hold)

	# an out-of-range cue clamps rather than reaching for a tab that is not there
	_slots(2)
	var over: Array = _ed._split_speakers("One.\n<!-- speaker: 7 -->\nTwo.")
	_ok(over.size() == 2 and int(over[1]["slot"]) == 1,
		"a cue past the last tab must clamp to it, got %s" % JSON.stringify(over))

	# a cue for the voice already reading is not a passage boundary
	_slots(2)
	var same: Array = _ed._split_speakers("One.\n<!-- speaker: 1 -->\nTwo.")
	_ok(same.size() == 1, "a redundant cue split the passage anyway")


## NOTHING IN <!-- --> IS SPOKEN. The cues are comments, so the format invites
## authoring notes beside them, and the failure is that the reader says them.
func _check_comments_never_spoken() -> void:
	_slots(2)
	var chunks: Array = _ed._build_chunks(
		"The pen comes back. <!-- ask about the llama -->\n\n<!-- speaker: 2 -->\n\nNo Diddy.")
	var said := ""
	for c in chunks:
		for w in (c as Dictionary)["words"]:
			said += String((w as Dictionary)["text"]) + " "
	_ok(not said.to_lower().contains("llama") and not said.to_lower().contains("ask"),
		"an authoring note reached the voice: %s" % said)
	_ok(said.to_lower().contains("diddy"), "the text after a note went missing: %s" % said)


## Every chunk knows whose it is, and the sentence numbering does NOT restart at a
## speaker change - [Subtitles] windows the overlay by that index, so two voices
## sharing sentence 0 puts both lines on screen at once.
func _check_chunks_carry_their_voice() -> void:
	_slots(2)
	var chunks: Array = _ed._build_chunks(
		"One. Two.\n\n<!-- speaker: 2 -->\n\nThree. Four.")
	_ok(chunks.size() == 4, "four sentences, got %d chunks" % chunks.size())
	if chunks.size() != 4:
		return
	var slots: Array = []
	var nums: Array = []
	for c in chunks:
		slots.append(int((c as Dictionary).get("slot", -1)))
		nums.append(int(((c as Dictionary)["words"][0] as Dictionary)["sentence"]))
	_ok(slots == [0, 0, 1, 1], "chunk voices came out %s" % [slots])
	_ok(nums == [0, 1, 2, 3], "sentence numbering restarted at the change: %s" % [nums])


## THE DIALS ACTUALLY REACH THE HOST. A tab whose settings never leave the panel
## is the failure that looks most like success.
func _check_settings_reach_the_request() -> void:
	_slots(2)
	_ed._slots[0]["pace"] = 1.0
	_ed._slots[0]["tone"] = 0                      # Neutral
	_ed._slots[0]["pause"] = 1.0
	_ed._slots[0]["speaker"] = 0
	_ed._slots[1]["pace"] = 0.5
	_ed._slots[1]["tone"] = 4                      # Spooky: slower, three semitones down
	_ed._slots[1]["pause"] = 3.0
	_ed._slots[1]["speaker"] = 7
	var ch := {"tokens": [], "plan_u": 0.0, "plan_v": 0.0}
	var a: Dictionary = _ed._request_args(_ed._cfg(0), ch)
	var b: Dictionary = _ed._request_args(_ed._cfg(1), ch)
	_ok(float(b["length_scale"]) > float(a["length_scale"]) * 1.5,
		"a slower, lower voice asked for length_scale %.3f against %.3f"
		% [b["length_scale"], a["length_scale"]])
	_ok(int(b["speaker"]) == 7 and int(a["speaker"]) == 0,
		"the reader id did not follow the tab: %d / %d" % [a["speaker"], b["speaker"]])
	_ok(float(b["pause_scale"]) == 3.0 and float(a["pause_scale"]) == 1.0,
		"the pause scale did not follow the tab")
	_ok(absf(_ed._pitch_ratio_of(_ed._cfg(1)) - 1.0) > 0.05
		and absf(_ed._pitch_ratio_of(_ed._cfg(0)) - 1.0) < 0.001,
		"the tone's pitch shift did not follow the tab")
	# ...and the seam between sentences is the incoming voice's own rest
	_ok(_ed._seam_gap_of(_ed._cfg(1)) > _ed._seam_gap_of(_ed._cfg(0)),
		"the seam did not follow the tab")


## The room is per voice too, and it is applied by DIALLING one shared chain
## rather than by building a second - so the check is that the same chain answers
## two slots differently.
func _check_room_is_per_voice() -> void:
	_slots(2)
	_ed._slots[0]["echo"] = 0.0
	_ed._slots[0]["room"] = 0.0
	_ed._slots[0]["ambience"] = 0.0
	_ed._slots[1]["echo"] = 0.8
	_ed._slots[1]["room"] = 0.7
	_ed._slots[1]["ambience"] = 0.5
	var fx := VoiceFX.new()
	_ed._apply_fx(fx, _ed._cfg(0))
	var dry_echo := fx.echo_wet
	var dry_pad := fx.pad
	_ed._apply_fx(fx, _ed._cfg(1))
	_ok(fx.echo_wet > dry_echo + 0.5 and fx.pad > dry_pad + 0.2,
		"one chain did not re-dial between voices (echo %.2f->%.2f, pad %.2f->%.2f)"
		% [dry_echo, fx.echo_wet, dry_pad, fx.pad])


## THE HANDOVER LANDS ON A FRAME. The marks are scheduled seconds before the
## audio they describe is pushed, in absolute frames, and the failure mode is a
## whole buffer of one speaker read in the other's room - audible, and almost
## impossible to attribute after the fact.
func _check_handover_is_sample_accurate() -> void:
	_slots(2)
	_ed._slots[1]["echo"] = 0.9
	_ed._fx = VoiceFX.new()
	_ed._fx_marks = [{"at": 0, "slot": 0}, {"at": 1000, "slot": 1}]
	_ed._fx_live_slot = -1
	_ed._pushed = 0
	# at the head: the first voice is dialled in, and the push stops at the change
	_ok(_ed._fx_admit(4096) == 1000, "the push must stop at the handover, got %d"
		% _ed._fx_admit(4096))
	_ok(_ed._fx_live_slot == 0, "the opening voice's room was not dialled in")
	_ok(_ed._fx.echo_wet < 0.1, "the second voice's room arrived early")
	# short of it, nothing changes and the remaining distance is what is offered
	_ed._pushed = 600
	_ok(_ed._fx_admit(4096) == 400, "frames offered up to the handover")
	_ok(_ed._fx_live_slot == 0, "the voice changed before its own first frame")
	# on it
	_ed._pushed = 1000
	_ok(_ed._fx_admit(4096) == 4096, "past the last mark the whole buffer is free")
	_ok(_ed._fx_live_slot == 1 and _ed._fx.echo_wet > 0.5,
		"the second voice's room did not arrive at its own frame")


## THE TABS THEMSELVES. Adding one must not disturb the voice you already have,
## switching away and back must return exactly what was left there, and tab 1 has
## to be un-removable - a reading with no reader is not a state to reach.
func _check_tabs() -> void:
	_ed._slots = [GenerativeEditor.SLOT_DEFAULTS.duplicate()]
	_ed._slot = 0
	_ed._rebuild_tabs()
	_ok(_ed._tabs.tab_count == 1 and _ed._tab_del.disabled,
		"a fresh panel is one tab that cannot be removed")

	_ed._arc.value = 0.20                      # something to recognise tab 1 by
	_ed._on_tab_add()
	_ok(_ed._tabs.tab_count == 2 and _ed._slot == 1 and not _ed._tab_del.disabled,
		"adding a voice selects it and lets it be removed")
	_ok(absf(_ed._arc.value - 0.20) < 0.001, "the new tab did not start as a copy")
	_ed._arc.value = 0.90                      # ...and tab 2 by
	_ed._speaker.value = 5                     # a control that regenerates on change

	_ed._on_tab_selected(0)
	_ok(absf(_ed._arc.value - 0.20) < 0.001,
		"tab 1 came back holding %.2f, which is tab 2's" % _ed._arc.value)
	_ok(int(_ed._speaker.value) == 0, "tab 1 came back holding tab 2's reader")
	_ed._on_tab_selected(1)
	_ok(absf(_ed._arc.value - 0.90) < 0.001,
		"tab 2 came back holding %.2f" % _ed._arc.value)
	_ok(int(_ed._speaker.value) == 5, "tab 2 came back holding reader %d" % _ed._speaker.value)
	_ok(_ed._tabs.get_tab_title(0) == "1" and _ed._tabs.get_tab_title(1) == "2",
		"the tabs are numbered as the cues are")

	# switching tabs must not throw the reading away - that is a repace, and it
	# would fire on every glance at another voice's settings
	_ed._chunks = [{"tokens": [], "words": []}]
	_ed._epoch = 0
	_ed._on_tab_selected(0)
	_ed._on_tab_selected(1)
	_ok(_ed._epoch == 0, "looking at another tab regenerated the reading")
	_ed._chunks = []

	_ed._on_tab_del()
	_ok(_ed._tabs.tab_count == 1 and _ed._slot == 0 and _ed._tab_del.disabled,
		"removing the last voice returns to one un-removable tab")
	_ok(absf(_ed._arc.value - 0.20) < 0.001, "the surviving tab lost its settings")
	_ed._on_tab_del()
	_ok(_ed._slots.size() == 1, "tab 1 was removable")


## The saved shape survives a ConfigFile round trip. It stores every number as a
## float, so a tone index and a reader id come back as 3.0 - and a float where the
## host wants an int is a request the backend refuses.
func _check_saved_shape() -> void:
	_slots(2)
	_ed._slots[1]["tone"] = 3
	_ed._slots[1]["speaker"] = 12
	_ed._slots[1]["voice"] = "en_US-libritts-high"
	var path := "user://_multi_voice_probe.cfg"
	var w := ConfigFile.new()
	w.set_value("generative", "slots", _ed._slots)
	w.save(path)
	var r := ConfigFile.new()
	r.load(path)
	var back: Array = r.get_value("generative", "slots", [])
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	_ok(back.size() == 2, "two voices saved, %d came back" % back.size())
	if back.size() != 2:
		return
	var m: Dictionary = _ed._merge(back[1] as Dictionary)
	_ok(typeof(m["tone"]) == TYPE_INT and int(m["tone"]) == 3, "the tone came back as %s" % [m["tone"]])
	_ok(typeof(m["speaker"]) == TYPE_INT and int(m["speaker"]) == 12,
		"the reader id came back as %s" % [m["speaker"]])
	_ok(String(m["voice"]) == "en_US-libritts-high", "the voice id came back as %s" % [m["voice"]])
	# a slot written by an older build is short a key, not broken
	var old: Dictionary = _ed._merge({"pace": 0.8})
	_ok(absf(float(old["pace"]) - 0.8) < 0.001 and absf(float(old["presence"]) - 1.0) < 0.001,
		"a slot missing keys did not fall back to the defaults")


## A DIAL MOVED ON A SILENT TAB STAYS SILENT. The panel shows one voice's
## settings while another may be on air, so the live chain follows the voice
## being HEARD - otherwise opening the reverb on a character who has not spoken
## yet puts the narrator in a cathedral mid-sentence.
func _check_silent_tabs_stay_silent() -> void:
	_slots(2)
	_ed._slots[0]["echo"] = 0.3
	_ed._slots[1]["echo"] = 0.9
	_ed._fx = VoiceFX.new()
	_ed._fx_live_slot = 0                      # voice 1 is on air
	_ed._slot = 1                              # voice 2 is on screen
	_ed._live_fx()
	_ok(_ed._fx.echo_wet < 0.05,
		"a dial on a silent tab reached the voice on air (echo %.2f)" % _ed._fx.echo_wet)
	_ed._slot = 0                              # now the tab on screen IS on air
	_ed._live_fx()
	_ok(absf(_ed._fx.echo_wet - 0.3) < 0.001,
		"the speaking tab's own dial did not apply (echo %.2f)" % _ed._fx.echo_wet)


## THE HANDOVER RESTS LONGER THAN A SENTENCE END. Run together, two readers
## sound like one person changing their mind mid-paragraph - so the seam before a
## chunk whose predecessor belonged to somebody else takes the Turn rest on top.
## Global, not per tab: it is the boundary's rest, not either voice's.
func _check_turn_rest() -> void:
	_slots(2)
	var chunks: Array = [{"slot": 0}, {"slot": 0}, {"slot": 1}, {"slot": 1}]
	var s: Dictionary = _ed._cfg(0)
	_ed._turn.value = 1.0
	var within := _ed._gap_before(chunks, 1, s)
	var across := _ed._gap_before(chunks, 2, s)
	_ok(across > within + 0.4,
		"a handover rested %.2fs against %.2fs inside one voice" % [across, within])
	_ok(_ed._gap_before(chunks, 3, s) == within, "the rest after a handover stayed long")
	_ok(_ed._gap_before(chunks, 0, s) == within, "the first chunk was given a seam")

	# 0 hands over on the same beat as any other sentence...
	_ed._turn.value = 0.0
	_ok(_ed._gap_before(chunks, 2, s) == within, "Turn 0 still lengthened the handover")
	# ...and the whole rest is capped however far it is pushed
	_ed._turn.value = _ed.MAX_TURN_SCALE
	_ok(_ed._gap_before(chunks, 2, s) <= GenerativeEditor.TURN_CEILING + 0.001,
		"the handover ran past the ceiling: %.2fs" % _ed._gap_before(chunks, 2, s))
	_ok(_ed._gap_before(chunks, 2, s) > across, "the top of the dial is no longer than the middle")
	_ed._turn.value = 1.0


## A CHAPTER FILE OPENS WITH ITS OWN METADATA, and the reader must not announce
## it. The cue syntax is a chapter file's, so chapter files are what gets pasted
## in - `title: ...` between two `---` rules, read aloud before the first word.
func _check_frontmatter() -> void:
	_slots(2)
	var chunks: Array = _ed._build_chunks(
		"---\ntitle: Charlotte's Web of Lies\n---\n\nThere is one report.\n\n"
		+ "<!-- speaker: 2 -->\n\nI am the spider.")
	var said := ""
	for c in chunks:
		for w in (c as Dictionary)["words"]:
			said += String((w as Dictionary)["text"]) + " "
	_ok(not said.to_lower().contains("title") and not said.to_lower().contains("charlotte"),
		"the frontmatter was read aloud: %s" % said)
	_ok(said.to_lower().contains("report") and said.to_lower().contains("spider"),
		"the chapter itself went missing: %s" % said)
	# a blank line before the opening rule is still frontmatter
	var lead: Array = _ed._split_speakers("\n\n---\ntitle: A Chapter\n---\n\nThe text.")
	_ok(not String(lead[0]["text"]).to_lower().contains("chapter"),
		"a blank line above the frontmatter defeated it: %s" % JSON.stringify(lead[0]["text"]))
	# ...and a rule in the middle of the text is a rule, not frontmatter
	var mid: Array = _ed._split_speakers("Before the line.\n\n---\n\nKeep this.")
	_ok(String(mid[0]["text"]).to_lower().contains("keep this"),
		"a horizontal rule mid-text swallowed what followed it")


## A TEMPLATE MACRO READS ITS DEFAULT, NEVER ITS OWN TEXT. The manuscript writes
## `${CHAPTERS_BEFORE_IN_WORDS:twenty-one}` because only its build knows the real
## figure and ghost is not the build; the rule is enforced in [TextNorm] and
## checked there, so what this asks is the editor-level half - that it survives
## the speaker split and that the panel says which macros have no default.
func _check_macros() -> void:
	_slots(2)
	var chunks: Array = _ed._build_chunks(
		"It has said so for ${CHAPTERS_BEFORE_IN_WORDS:twenty-one} chapters.\n\n"
		+ "<!-- speaker: 2 -->\n\nAnd ${WORD_COUNT_IN_WORDS} of them.")
	var said := ""
	for c in chunks:
		for w in (c as Dictionary)["words"]:
			said += String((w as Dictionary)["text"]) + " "
	_ok(said.contains("twenty-one"), "the default was not read: %s" % said)
	_ok(not said.contains("CHAPTERS") and not said.to_lower().contains("word_count")
		and not said.contains("$"),
		"the macro's own text reached the voice: %s" % said)
	_ok(_ed._plan_note.contains("WORD_COUNT_IN_WORDS"),
		"the plan did not name the macro with no default: %s" % _ed._plan_note)

	# A PARAGRAPH-LENGTH DEFAULT survives the speaker split and the paragraph
	# placement, both of which read the RAW passage text - before TextNorm has
	# expanded anything. Nothing in the book is this size today; the author has
	# said there will be, so it is held here rather than found in a render.
	_slots(1)
	var long_chunks: Array = _ed._build_chunks(
		"Before it.\n\n${BODY:First line of it.\n\nA whole second paragraph, at length.}\n\nAfter it.")
	var long_said := ""
	for c in long_chunks:
		for w in (c as Dictionary)["words"]:
			long_said += String((w as Dictionary)["text"]) + " "
	_ok(long_said.contains("First line of it") and long_said.contains("second paragraph")
		and long_said.contains("Before it") and long_said.contains("After it"),
		"a paragraph-length default did not come through whole: %s" % long_said)
	_ok(not long_said.contains("BODY") and not long_said.contains("$"),
		"the macro's own text reached the voice: %s" % long_said)
	_ok(_ed._plan_note.is_empty(), "a usable long default was reported: %s" % _ed._plan_note)
	for c in long_chunks:
		var u := float((c as Dictionary).get("plan_u", -1.0))
		_ok(u >= 0.0 and u <= 1.0, "paragraph placement came out %.2f on a long default" % u)

	# A macro inside an authoring note is not a macro missing from the reading -
	# the note was never going to be read either way. This also checks the note
	# is REBUILT rather than appended to: a warning that outlives the text it was
	# about is worse than none, because it is read as current.
	_ed._build_chunks("Plain text. <!-- todo: ${WORD_COUNT_IN_WORDS} -->")
	_ok(_ed._plan_note.is_empty(),
		"the note survived the text being fixed: %s" % _ed._plan_note)

	# ...and an over-range speaker cue reports the same way, rather than writing
	# a status line that "Planned N chunk(s)" overwrites a moment later
	_slots(1)
	_ed._build_chunks("One.\n<!-- speaker: 4 -->\nTwo.")
	_ok(_ed._plan_note.contains("speaker 4"),
		"an out-of-range cue was not reported: %s" % _ed._plan_note)


## THE PAGE SHOWS WHAT THE PAGE SAID. `2009` is spoken "two thousand nine" and
## must still be SHOWN as `2009` - reported as a subtitle reading "...who left
## the building in two thousand nine." The rule itself lives in [TextNorm] and
## [Phonemes] and is checked in norm_check; what this holds is the editor's half,
## which is that one source run becomes ONE subtitle card covering the whole run
## rather than three cards, or one card over the first syllable.
func _check_subtitles_show_the_source() -> void:
	_slots(1)
	var chunks: Array = _ed._build_chunks(
		"He left the building in 2009. She paid $5 on the 1st.")
	var shown := ""
	var spoken := ""
	var reach := {}          # card text -> how many spoken words it covers
	for c in chunks:
		for w in (c as Dictionary)["words"]:
			var d: Dictionary = w
			shown += String(d["text"]) + " "
			reach[String(d["text"])] = int(d.get("end", d["index"])) - int(d["index"]) + 1
		for t in (c as Dictionary)["tokens"]:
			spoken += String((t as Dictionary)["text"]) + " "
	_ok(shown.contains("2009.") and shown.contains("$5") and shown.contains("1st"),
		"the source spelling is not what the subtitle shows: %s" % shown)
	_ok(shown.contains("1st."), "the ordinal lost its full stop: %s" % shown)
	_ok(not shown.contains("two thousand") and not shown.contains("five dollars")
		and not shown.to_lower().contains("first"),
		"the spoken expansion reached the subtitle: %s" % shown)
	_ok(spoken.contains("two thousand nine") and spoken.contains("five dollars"),
		"the voice stopped saying the number: %s" % spoken)
	# A CARD REACHES AS FAR AS ITS RUN IS SPOKEN. `2009` is three spoken words
	# and `$5` is two, so each card must cover that many or the numeral flashes
	# for one syllable of the several it takes to say. `1st` is one word and is
	# the control: a rewrite is not automatically a span.
	_ok(int(reach.get("2009.", 0)) == 3, "the 2009 card covers %s spoken word(s), wanted 3"
		% [reach.get("2009.", 0)])
	_ok(int(reach.get("$5", 0)) == 2, "the $5 card covers %s spoken word(s), wanted 2"
		% [reach.get("$5", 0)])
	_ok(int(reach.get("1st.", 0)) == 1, "the 1st card covers %s spoken word(s), wanted 1 (cards: %s)"
		% [reach.get("1st.", 0), reach.keys()])
	_ok(not shown.contains("  "), "a blank card was emitted: %s" % JSON.stringify(shown))
	# no card may be empty, whatever the phonemizer did with the run
	for c in chunks:
		for w in (c as Dictionary)["words"]:
			_ok(not String((w as Dictionary)["text"]).strip_edges().is_empty(),
				"an empty subtitle card was emitted")
