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
	_ed.remove_child(_ed._cast_timer)
	add_child(_ed._cast_timer)
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
	_check_hesitations()
	_check_hesitation_splice()
	_check_hum_is_held()
	_check_timestamp_pauses()
	_check_ink_travels()
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
	var names: Array = [Manuscript.NARRATOR]
	for i in range(1, n):
		names.append("Voice %d" % (i + 1))
	_cast(names)


func _cast(names: Array) -> void:
	_ed._names = PackedStringArray(names)
	_ed._slots = []
	for i in names.size():
		_ed._slots.append(GenerativeEditor.SLOT_DEFAULTS.duplicate())
	_ed._stash = {}
	_ed._slot = 0


func _who(segs: Array) -> Array:
	var out: Array = []
	for s in segs:
		out.append(String((s as Dictionary)["speaker"]))
	return out


## THE CUES. Both halves matter and the second one more: the HOLDS are lines that
## look like a cue and are prose, and every one of them would be swallowed whole
## and read in the wrong voice from there to the end of the chapter.
func _check_cues() -> void:
	_slots(3)
	var segs: Array = _ed._split_speakers(
		"Narrator opens.\n\n<!-- speaker: Charlotte -->\n\nI am the spider.\n\n"
		+ "<!-- speaker: Narrator -->\n\nThe pen comes back.\n\n[speaker: Emily White]\n\nNo Diddy.")
	_ok(segs.size() == 4, "four passages, got %d" % segs.size())
	if segs.size() == 4:
		_ok(_who(segs) == ["Narrator", "Charlotte", "Narrator", "Emily White"],
			"passages went to voices %s" % [_who(segs)])
		_ok(String(segs[1]["text"]).begins_with("I am the spider"),
			"the second passage is %s" % JSON.stringify(segs[1]["text"]))

	# no cues at all is the case every script had before names
	var plain: Array = _ed._split_speakers("Just prose.\nMore of it.")
	_ok(plain.size() == 1 and String(plain[0]["speaker"]) == Manuscript.NARRATOR,
		"an uncued script is one passage, the narrator's")

	# a NUMBER is still a name - the scripts written for numbered tabs keep working
	var num: Array = _ed._split_speakers("One.\n<!-- speaker: 2 -->\nTwo.")
	_ok(_who(num) == ["Narrator", "2"], "a numbered cue is not read as a name: %s" % [_who(num)])

	# HOLDS: prose that mentions a speaker, and a cue that is not alone on its line
	for hold in ["speaker: Ryan is what he said.", "He said <!-- speaker: Ryan --> aloud.",
			"The speaker: Ryan of them, in fact."]:
		var h: Array = _ed._split_speakers("Before.\n%s\nAfter." % hold)
		_ok(h.size() == 1, "prose taken for a cue: %s" % hold)
	_ok(Manuscript.speakers("Before.\nHe said <!-- speaker: Ryan --> aloud.\n")
			== PackedStringArray(["Narrator"]),
		"a cue inside a line of prose became a tab")

	# a cue for the voice already reading is not a passage boundary
	var same: Array = _ed._split_speakers("<!-- speaker: Ryan -->\nOne.\n<!-- speaker: Ryan -->\nTwo.")
	_ok(same.size() == 1, "a redundant cue split the passage anyway")
	# ...and a chapter that OPENS on a cue has no narrator at all
	_ok(Manuscript.speakers("<!-- speaker: Ryan -->\n\nOne.\n\n<!-- speaker: Judge -->\n\nYou.")
			== PackedStringArray(["Ryan", "Judge"]),
		"a chapter opening on a cue grew an empty narrator tab")


## NOTHING IN <!-- --> IS SPOKEN. The cues are comments, so the format invites
## authoring notes beside them, and the failure is that the reader says them.
func _check_comments_never_spoken() -> void:
	_slots(2)
	var chunks: Array = _ed._build_chunks(
		"The pen comes back. <!-- ask about the llama -->\n\n<!-- speaker: Voice 2 -->\n\nNo Diddy.")
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
		"One. Two.\n\n<!-- speaker: Voice 2 -->\n\nThree. Four.")
	_ok(chunks.size() == 4, "four sentences, got %d chunks" % chunks.size())
	if chunks.size() != 4:
		return
	var slots: Array = []
	var nums: Array = []
	for c in chunks:
		slots.append(String((c as Dictionary).get("speaker", "?")))
		nums.append(int(((c as Dictionary)["words"][0] as Dictionary)["sentence"]))
	_ok(slots == ["Narrator", "Narrator", "Voice 2", "Voice 2"], "chunk voices came out %s" % [slots])
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
	_ed._fx_marks = [{"at": 0, "speaker": "Narrator"}, {"at": 1000, "speaker": "Voice 2"}]
	_ed._fx_live_name = ""
	_ed._pushed = 0
	# at the head: the first voice is dialled in, and the push stops at the change
	_ok(_ed._fx_admit(4096) == 1000, "the push must stop at the handover, got %d"
		% _ed._fx_admit(4096))
	_ok(_ed._fx_live_name == "Narrator", "the opening voice's room was not dialled in")
	_ok(_ed._fx.echo_wet < 0.1, "the second voice's room arrived early")
	# short of it, nothing changes and the remaining distance is what is offered
	_ed._pushed = 600
	_ok(_ed._fx_admit(4096) == 400, "frames offered up to the handover")
	_ok(_ed._fx_live_name == "Narrator", "the voice changed before its own first frame")
	# on it
	_ed._pushed = 1000
	_ok(_ed._fx_admit(4096) == 4096, "past the last mark the whole buffer is free")
	_ok(_ed._fx_live_name == "Voice 2" and _ed._fx.echo_wet > 0.5,
		"the second voice's room did not arrive at its own frame")


## THE TABS ARE THE SCRIPT'S SPEAKERS. They appear as names are cued, in order of first
## appearance; a new one starts as a copy of the first voice; switching away and back
## returns exactly what was left there; and a name that leaves the script keeps its
## settings for when it returns.
func _check_tabs() -> void:
	_cast([Manuscript.NARRATOR])
	_ed._rebuild_tabs()
	_ok(_ed._tabs.get_child_count() == 1, "a fresh panel is one tab")

	_ed._arc.value = 0.20                      # something to recognise the narrator by
	_ed._refresh_cast("Opening.\n\n<!-- speaker: Emily White -->\n\nHis.\n\n"
		+ "<!-- speaker: Judge -->\n\nOverruled.")
	_ok(_ed._names == PackedStringArray(["Narrator", "Emily White", "Judge"]),
		"the tabs are not the script's names in order: %s" % [_ed._names])
	_ok(_ed._tabs.get_child_count() == 3, "one button per name, got %d" % _ed._tabs.get_child_count())
	_ok(String((_ed._tabs.get_child(1) as Button).text).begins_with("Emily White"),
		"the tab is not labelled with the name")
	_ok(_ed._slot == 0 and absf(_ed._arc.value - 0.20) < 0.001,
		"the voice on screen moved when names were added")

	_ed._on_tab_selected(1)
	_ok(absf(_ed._arc.value - 0.20) < 0.001, "a new name did not start as a copy of the first voice")
	_ed._arc.value = 0.90                      # ...and Emily by
	_ed._speaker.value = 5                     # a control that regenerates on change

	_ed._on_tab_selected(0)
	_ok(absf(_ed._arc.value - 0.20) < 0.001,
		"the narrator came back holding %.2f, which is Emily's" % _ed._arc.value)
	_ok(int(_ed._speaker.value) == 0, "the narrator came back holding Emily's reader")
	_ed._on_tab_selected(1)
	_ok(absf(_ed._arc.value - 0.90) < 0.001, "Emily came back holding %.2f" % _ed._arc.value)
	_ok(int(_ed._speaker.value) == 5, "Emily came back holding reader %d" % _ed._speaker.value)

	# switching tabs must not throw the reading away - that is a repace, and it
	# would fire on every glance at another voice's settings
	_ed._chunks = [{"tokens": [], "words": []}]
	_ed._epoch = 0
	_ed._on_tab_selected(0)
	_ed._on_tab_selected(1)
	_ok(_ed._epoch == 0, "looking at another tab regenerated the reading")
	_ed._chunks = []

	# A NAME THAT LEAVES KEEPS ITS VOICE. Rewrite Emily out, then back in.
	_ed._refresh_cast("Opening.\n\n<!-- speaker: Judge -->\n\nOverruled.")
	_ok(_ed._names == PackedStringArray(["Narrator", "Judge"]),
		"a name no longer cued is still a tab: %s" % [_ed._names])
	_ok(_ed._tab_name() == "Narrator", "the tab on screen did not fall back when its name left")
	_ok(_ed._cfg_of("Emily White")["arc"] > 0.8,
		"a name that left the script lost its settings: %s" % [_ed._cfg_of("Emily White")])
	_ed._refresh_cast("Opening.\n\n<!-- speaker: Emily White -->\n\nHis.")
	var i := _ed._names.find("Emily White")
	_ok(i >= 0 and absf(float(_ed._cfg(i)["arc"]) - 0.90) < 0.001 and int(_ed._cfg(i)["speaker"]) == 5,
		"Emily came back without the voice she left with: %s" % [_ed._cfg(i) if i >= 0 else {}])
	# ...and the cast written out carries every voice, shown or not
	_ok(_ed._cast_dict().has("Judge") and _ed._cast_dict().has("Emily White"),
		"the saved cast dropped a voice the script no longer shows")


## The saved shape survives a ConfigFile round trip. It stores every number as a
## float, so a tone index and a reader id come back as 3.0 - and a float where the
## host wants an int is a request the backend refuses. And the numbered tabs this
## replaced are MIGRATED, not lost: row N was cued `N`, row 1 was also the narrator.
func _check_saved_shape() -> void:
	_cast(["Narrator", "Emily White"])
	_ed._slots[1]["tone"] = 3
	_ed._slots[1]["speaker"] = 12
	_ed._slots[1]["voice"] = "en_US-libritts-high"
	var path := "user://_multi_voice_probe.cfg"
	var w := ConfigFile.new()
	w.set_value("generative", "cast", _ed._cast_dict())
	w.save(path)
	var r := ConfigFile.new()
	r.load(path)
	var back: Dictionary = _ed._merge_cast(r.get_value("generative", "cast", {}))
	DirAccess.remove_absolute(ProjectSettings.globalize_path(path))
	_ok(back.size() == 2, "two voices saved, %d came back" % back.size())
	if not back.has("Emily White"):
		_ok(false, "the voice came back without its name: %s" % [back.keys()])
		return
	var m: Dictionary = back["Emily White"]
	_ok(typeof(m["tone"]) == TYPE_INT and int(m["tone"]) == 3, "the tone came back as %s" % [m["tone"]])
	_ok(typeof(m["speaker"]) == TYPE_INT and int(m["speaker"]) == 12,
		"the reader id came back as %s" % [m["speaker"]])
	_ok(String(m["voice"]) == "en_US-libritts-high", "the voice id came back as %s" % [m["voice"]])
	# a slot written by an older build is short a key, not broken
	var old: Dictionary = _ed._merge({"pace": 0.8})
	_ok(absf(float(old["pace"]) - 0.8) < 0.001 and absf(float(old["presence"]) - 1.0) < 0.001,
		"a slot missing keys did not fall back to the defaults")
	var mig: Dictionary = _ed._cast_from_rows([{"pace": 0.7}, {"pace": 1.3}],
		"Intro.\n<!-- speaker: 1 -->\nOne.\n<!-- speaker: 2 -->\nTwo.")
	_ok(mig.has("Narrator") and mig.has("1") and mig.has("2")
			and absf(float(mig["Narrator"]["pace"]) - 0.7) < 0.001
			and absf(float(mig["2"]["pace"]) - 1.3) < 0.001,
		"the numbered tabs did not migrate to names: %s" % [mig])
	# ...and a single-voice chapter with no cues migrates to the narrator ALONE
	var solo: Dictionary = _ed._cast_from_rows([{"pace": 0.6}], "Just prose, no cues.")
	_ok(solo.keys() == ["Narrator"], "an uncued chapter grew a phantom voice: %s" % [solo.keys()])


## A DIAL MOVED ON A SILENT TAB STAYS SILENT. The panel shows one voice's
## settings while another may be on air, so the live chain follows the voice
## being HEARD - otherwise opening the reverb on a character who has not spoken
## yet puts the narrator in a cathedral mid-sentence.
func _check_silent_tabs_stay_silent() -> void:
	_slots(2)
	_ed._slots[0]["echo"] = 0.3
	_ed._slots[1]["echo"] = 0.9
	_ed._fx = VoiceFX.new()
	_ed._fx_live_name = "Narrator"             # voice 1 is on air
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
	var chunks: Array = [{"speaker": "A"}, {"speaker": "A"}, {"speaker": "B"}, {"speaker": "B"}]
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
		+ "<!-- speaker: Voice 2 -->\n\nI am the spider.")
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
		+ "<!-- speaker: Voice 2 -->\n\nAnd ${WORD_COUNT_IN_WORDS} of them.")
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


## HESITATIONS. A `<!-- hesitation -->` is a rest AT THAT POINT and never a word: it must not
## be spoken, it must land on the right word boundary (mid-sentence included), a marker that
## names its length keeps it, and switching the feature off removes them all. Every one of
## these fails silently - a rest in the wrong place just sounds like a slow reader.
func _check_hesitations() -> void:
	_slots(2)
	_ed._hesitate_on.button_pressed = true
	_ed._hesitate.value = 1.5
	var chunks: Array = _ed._build_chunks(
		"She comes all the way around, <!-- hesitation --> and stops on me.\n\n"
		+ "<!-- speaker: Voice 2 -->\n\n<!-- hesitation: 2.5 -->\n\nHis.")
	var said := ""
	for c in chunks:
		for t in (c as Dictionary)["tokens"]:
			said += String((t as Dictionary)["text"]) + " "
	_ok(not said.to_lower().contains("hesitation") and not said.contains(TextNorm.HOLD_MARK),
		"the marker reached the voice: %s" % said)
	_ok(said.contains("around") and said.contains("stops") and said.to_lower().contains("his"),
		"words went missing around a hesitation: %s" % said)
	_ok(chunks.size() == 2, "a mid-sentence hesitation split the sentence: %d chunks" % chunks.size())
	if chunks.size() != 2:
		return
	var h0: Array = (chunks[0] as Dictionary).get("holds", [])
	_ok(h0.size() == 1, "the mid-sentence rest was not recorded: %s" % [h0])
	if h0.size() == 1:
		var tok := int(h0[0]["tok"])
		var toks: Array = (chunks[0] as Dictionary)["tokens"]
		_ok(String(toks[tok]["text"]).to_lower().begins_with("around") and not bool(h0[0]["before"]),
			"the rest sits after %s rather than after 'around'" % [toks[tok]["text"]])
		_ok(is_equal_approx(float(h0[0]["sec"]), 1.5), "a bare marker did not take the dial: %s" % [h0])
	var h1: Array = (chunks[1] as Dictionary).get("holds", [])
	_ok(h1.size() == 1 and int(h1[0]["tok"]) == 0 and bool(h1[0]["before"])
			and is_equal_approx(float(h1[0]["sec"]), 2.5),
		"a leading marker with its own length came out %s" % [h1])
	# the subtitle still shows the word without the sentinel
	for c in chunks:
		for w in (c as Dictionary)["words"]:
			_ok(not String((w as Dictionary)["text"]).contains(TextNorm.HOLD_MARK),
				"the sentinel reached a subtitle card")

	# OFF means no rests at all, and still nothing spoken
	_ed._hesitate_on.button_pressed = false
	var off: Array = _ed._build_chunks("Around, <!-- hesitation --> and stops.")
	var any := 0
	var off_said := ""
	for c in off:
		any += ((c as Dictionary).get("holds", []) as Array).size()
		for t in (c as Dictionary)["tokens"]:
			off_said += String((t as Dictionary)["text"]) + " "
	_ok(any == 0, "Hesitate off still rested")
	_ok(not off_said.to_lower().contains("hesitation"), "Hesitate off spoke the marker: %s" % off_said)
	_ed._hesitate_on.button_pressed = true

	# a hesitation between two paragraphs rests after the first one's last word
	var para: Array = _ed._build_chunks("I will show up.\n\n<!-- hesitation -->\n\nI am here.")
	var h2: Array = (para[0] as Dictionary).get("holds", []) if para.size() > 0 else []
	_ok(para.size() == 2 and h2.size() == 1 and not bool(h2[0]["before"]),
		"a hesitation between paragraphs did not follow the first one: %s" % [h2])


## THE SPLICE. Silence of exactly the asked length goes in at the gap between the two words,
## every word after it moves by that much, and nothing before it moves at all.
func _check_hesitation_splice() -> void:
	var sr := 22050
	_ed._sr = sr
	var pcm := PackedFloat32Array()
	pcm.resize(sr)                 # one second
	pcm.fill(0.5)
	# three words: 0.0-0.3, 0.4-0.6, 0.7-0.95
	var spans: Array = [{"index": 0, "t0": 0.0, "t1": 0.3}, {"index": 1, "t0": 0.4, "t1": 0.6},
		{"index": 2, "t0": 0.7, "t1": 0.95}]
	var r: Dictionary = _ed._splice_holds(pcm, [{"tok": 0, "sec": 1.0, "before": false}], spans, 1.0)
	var out: PackedFloat32Array = r["pcm"]
	_ok(absi(out.size() - 2 * sr) <= 1, "the take grew by %d samples, not one second" % (out.size() - sr))
	var cut := int(0.35 * sr)
	_ok(absf(out[cut + sr / 2]) < 1e-6, "the rest is not silence at the gap")
	_ok(absf(out[int(0.2 * sr)] - 0.5) < 1e-6, "audio before the rest changed")
	_ok(absf(out[int(1.5 * sr)] - 0.5) < 1e-6, "audio after the rest did not move by its length")
	var cuts: Array = r["cuts"]
	_ok(is_equal_approx(GenerativeEditor._shifted(0.3, cuts, 1.0, false), 0.3),
		"the word before the rest moved")
	_ok(is_equal_approx(GenerativeEditor._shifted(0.4, cuts, 1.0, true), 1.4),
		"the word after the rest did not move by it")
	# a rest before the first word opens the chunk
	var r2: Dictionary = _ed._splice_holds(pcm, [{"tok": 0, "sec": 0.5, "before": true}], spans, 1.0)
	_ok(is_equal_approx(GenerativeEditor._shifted(0.0, r2["cuts"], 1.0, true), 0.5),
		"a leading rest did not push the first word back")
	_ok(absf((r2["pcm"] as PackedFloat32Array)[int(0.25 * sr)]) < 1e-6, "a leading rest is not silence")
	# ...and one after the last word closes it, moving nothing
	var r3: Dictionary = _ed._splice_holds(pcm, [{"tok": 2, "sec": 0.5, "before": false}], spans, 1.0)
	_ok(is_equal_approx(GenerativeEditor._shifted(0.95, r3["cuts"], 1.0, false), 0.95)
			and (r3["pcm"] as PackedFloat32Array).size() == sr + sr / 2,
		"a trailing rest moved a word or was the wrong length")
	# the resample ratio divides the model's clock
	var r4: Dictionary = _ed._splice_holds(pcm, [{"tok": 0, "sec": 1.0, "before": false}], spans, 2.0)
	_ok(is_equal_approx(GenerativeEditor._shifted(0.4, r4["cuts"], 2.0, true), 1.2),
		"the splice ignored the resample ratio")


## A HUM IS HELD. "Hmm." renders at ~0.2 s and reads as a clipped grunt; it is lengthened to
## a thinking hum by cycling its own pitch periods. Held here: the hum reaches its target
## length, the word after it moves by exactly what was added, and the stretch is STEADY - no
## dips, which is what cycling the dip between two m's produced.
func _check_hum_is_held() -> void:
	var chunks: Array = _ed._build_chunks("Hmm. Who looks like they want to speak?")
	var h: Array = (chunks[0] as Dictionary).get("holds", []) if chunks.size() > 0 else []
	_ok(h.size() == 1 and is_equal_approx(float(h[0].get("hum", 0.0)), 0.75),
		"Hmm. was not marked to be held: %s" % [h])
	var sr := 22050
	_ed._sr = sr
	var pcm := PackedFloat32Array()
	pcm.resize(int(0.6 * sr))
	for i in int(0.2 * sr):
		pcm[i] = 0.5 * sin(TAU * 150.0 * float(i) / float(sr))       # the hum, 0.0-0.2 s
	for i in range(int(0.3 * sr), int(0.6 * sr)):
		pcm[i] = 0.3 * sin(TAU * 220.0 * float(i) / float(sr))       # the next word
	var spans: Array = [{"index": 0, "t0": 0.0, "t1": 0.2}, {"index": 1, "t0": 0.3, "t1": 0.6}]
	var r: Dictionary = _ed._splice_holds(pcm, [{"tok": 0, "sec": 0.0, "before": false, "hum": 0.75}], spans, 1.0)
	var o: PackedFloat32Array = r["pcm"]
	var end := GenerativeEditor._shifted(0.2, r["cuts"], 1.0, false)
	_ok(absf(end - 0.75) < 0.03, "the hum was held to %.3fs, wanted 0.75" % end)
	_ok(absf(GenerativeEditor._shifted(0.3, r["cuts"], 1.0, true) - (0.3 + end - 0.2)) < 0.002,
		"the next word did not move by what the hum gained")
	# STEADY until it starts to fade, and only then does it die away.
	var cut: Dictionary = (r["cuts"] as Array)[0]
	var c0 := float(cut["at"]) / float(sr)
	var fade_at := c0 + GenerativeEditor.HUM_FADE_FROM * (end - c0)
	var hop := int(0.01 * sr)
	var lo := 1e9
	for f in range(2, int(fade_at * 100.0)):
		var e := 0.0
		for k in hop:
			e += o[f * hop + k] * o[f * hop + k]
		lo = minf(lo, sqrt(e / float(hop)))
	_ok(lo > 0.25, "the held hum dips before its fade (quietest 10 ms at %.3f of a 0.35 tone)" % lo)
	_check_hum_falls(cut["fill"], sr)
	_check_hum_is_smooth()


## THE HUM FALLS. A thinking "hmm" settles and drops; held flat it reads as a sustained note -
## "a hmm would typically have a downward inflection; the speaker would not just hold a
## constant note straight through". Held: the end of the fill is at least two semitones under
## its start, and it has died away rather than stopping at full strength.
func _check_hum_falls(fill: PackedFloat32Array, sr: int) -> void:
	var n := int(0.02 * sr)
	var p0: Vector3 = _ed._period_of(fill, 0, n)
	var p1: Vector3 = _ed._period_of(fill, int(0.8 * float(fill.size())) - n, n)
	var semis := 12.0 * log(p1.x / maxf(p0.x, 1.0)) / log(2.0)
	_ok(p0.x > 0.0 and p1.x > 0.0 and semis > 2.0,
		"the held hum does not fall (%.1f semitones from its start to 80%% through)" % semis)
	var rms := func(i0: int) -> float:
		var e := 0.0
		for k in n:
			e += fill[i0 + k] * fill[i0 + k]
		return sqrt(e / float(n))
	var head: float = rms.call(0)
	var tail: float = rms.call(fill.size() - n)
	_ok(tail < 0.15 * head, "the held hum stops at %.2f of its level instead of dying away" % (tail / maxf(head, 1e-9)))


## A HELD HUM IS SMOOTH on a hum shaped like a rendered one, which a pure sine is not: its
## pitch glides, it has harmonics, and its loudest stretch is FRY (period-doubled, an octave
## down). Butting copied periods end to end buzzed on exactly this - every join a step, since
## no whole number of samples is the period - and anchoring on the loudest stretch held the
## fry as a creak. Held here: no step in the fill larger than the hum's own, no loudness
## flutter, and the fill at the hum's pitch rather than the fry's.
func _check_hum_is_smooth() -> void:
	var sr := 22050
	_ed._sr = sr
	var pcm := PackedFloat32Array()
	pcm.resize(int(0.45 * sr))
	var ph := 0.0
	for i in int(0.3 * sr):
		var t := float(i) / float(sr)
		# fry fades in over 0.15-0.17 and out over 0.22-0.24, as a rendered one does
		var fry := clampf(minf((t - 0.15) / 0.02, (0.24 - t) / 0.02), 0.0, 1.0)
		var f0 := lerpf(205.0, 172.0, t / 0.3)
		ph += f0 / float(sr)
		var env := minf(1.0, t / 0.03) * minf(1.0, (0.3 - t) / 0.03)
		var v := 0.0
		for k in range(1, 7):
			v += sin(TAU * float(k) * ph) / float(k)
		# every other period louder and the ones between nearly gone: a subharmonic at f0 / 2
		v *= 1.0 + fry * (0.8 + 1.2 * sin(PI * ph))
		pcm[i] = 0.2 * env * v
	var spans: Array = [{"index": 0, "t0": 0.0, "t1": 0.3}]
	var r: Dictionary = _ed._splice_holds(pcm, [{"tok": 0, "sec": 0.0, "before": false, "hum": 0.9}], spans, 1.0)
	var cuts: Array = r["cuts"]
	_ok(cuts.size() == 1 and cuts[0].has("fill"), "the gliding hum was not held")
	if cuts.size() != 1 or not cuts[0].has("fill"):
		return
	var fill: PackedFloat32Array = cuts[0]["fill"]
	var o: PackedFloat32Array = r["pcm"]
	var s0 := int(cuts[0]["at"])
	var step := func(x: PackedFloat32Array, i0: int, i1: int) -> float:
		var m := 0.0
		for i in range(maxi(1, i0), mini(i1, x.size() - 1)):
			m = maxf(m, absf(x[i + 1] - 2.0 * x[i] + x[i - 1]))
		return m
	var own: float = maxf(step.call(pcm, 0, int(0.15 * sr)), step.call(pcm, int(0.24 * sr), int(0.3 * sr)))
	_ok(s0 < int(0.15 * sr) or s0 >= int(0.24 * sr), "the hum was anchored in its fry (%.3f s)" % (float(s0) / float(sr)))
	var held: float = step.call(o, s0 - 2, s0 + fill.size() + 2)
	_ok(held <= own * 1.05, "the held hum has a step %.4f past the hum's own largest %.4f" % [held, own])
	var hop := int(0.01 * sr)
	var rms := []
	for i in range(0, int(GenerativeEditor.HUM_FADE_FROM * float(fill.size())) - hop, hop):
		var e := 0.0
		for k in hop:
			e += fill[i + k] * fill[i + k]
		rms.append(sqrt(e / float(hop)))
	var mean := 0.0
	for v in rms:
		mean += v
	mean /= float(rms.size())
	var dev := 0.0
	for v in rms:
		dev += (v - mean) * (v - mean)
	var flutter := sqrt(dev / float(rms.size())) / maxf(mean, 1e-9)
	_ok(flutter < 0.1, "the held hum flutters (10 ms loudness varies %.2f of its mean)" % flutter)
	var pc: Vector3 = _ed._period_of(fill, 0, int(0.02 * sr))
	_ok(pc.x > 0.0 and float(sr) / pc.x > 150.0,
		"the hum was held in its fry (%.0f Hz)" % (float(sr) / maxf(pc.x, 1.0)))


## A TIME THAT OPENS A PARAGRAPH IS FOLLOWED BY A REST, as if the author had marked one - read
## straight on, "21:40 The subject..." ran the time into the sentence. Held: the paragraph-
## opening times each get the bare hesitation, welded after the time; a time mid-sentence gets
## none; an author's own hesitation after a time is not doubled; and Hesitate off removes the
## automatic ones with the rest.
func _check_timestamp_pauses() -> void:
	var body := "21:40 The subject has moved.\n\nWe met at 12:30 today.\n\n09:40 <!-- hesitation: 2 --> Arrived early.\n\n7:15 pm Dinner, alone."
	var was: bool = _ed._hesitate_on.button_pressed
	_ed._hesitate_on.button_pressed = true
	var bare: float = _ed._hesitate.value
	var kept: Array = _ed._split_speakers(body)
	var holds: Array = _ed._holds
	_ok(holds.size() == 3, "%d rests for three paragraph-opening times (one authored): %s" % [holds.size(), holds])
	if holds.size() == 3:
		_ok(is_equal_approx(float(holds[0]), bare) and is_equal_approx(float(holds[1]), 2.0)
			and is_equal_approx(float(holds[2]), bare),
			"the rests are not the bare hesitation, the author's own, the bare one: %s" % [holds])
	var text := String((kept[0] as Dictionary)["text"]) if not kept.is_empty() else ""
	_ok(text.contains("21:40" + TextNorm.HOLD_MARK) and text.contains("pm" + TextNorm.HOLD_MARK),
		"the rest is not welded right after the time (and its am/pm)")
	_ok(not text.contains("12:30" + TextNorm.HOLD_MARK), "a time mid-sentence got a rest")
	_ed._hesitate_on.button_pressed = false
	_ed._split_speakers(body)
	_ok((_ed._holds as Array).is_empty(), "Hesitate off kept the automatic rests")
	_ed._hesitate_on.button_pressed = was


## A VOICE'S INK is a setting like any other on it: it survives the panel capturing the slot
## (which rebuilds the slot from the controls - a key the controls do not carry is erased on
## the next autosave), a hex colour the Ink list does not offer is kept rather than replaced,
## and it reaches the page beside the text as the document's `inks`.
func _check_ink_travels() -> void:
	_ok(String(_ed._merge({"ink": "blue"})["ink"]) == "blue", "the slot schema drops `ink`")
	_ok(String(_ed._merge({})["ink"]) == "", "a voice with no ink is not the default black")
	var i: int = _ed._slot
	var was: Dictionary = (_ed._slots[i] as Dictionary).duplicate()
	_ed._slots[i]["ink"] = "red"
	_ed._apply_slot(i)
	_ed._capture_slot()
	_ok(String(_ed._slots[i]["ink"]) == "red", "capturing the slot lost its ink")
	_ed._slots[i]["ink"] = "#335577"
	_ed._apply_slot(i)
	_ed._capture_slot()
	_ok(String(_ed._slots[i]["ink"]) == "#335577", "a hex ink was replaced on capture")
	var doc: Dictionary = _ed.book_document("text")
	_ok((doc.get("inks", {}) as Dictionary).values().has("#335577"),
		"the ink does not travel with the document: %s" % [doc.get("inks")])
	_ed._slots[i] = was
	_ed._apply_slot(i)
