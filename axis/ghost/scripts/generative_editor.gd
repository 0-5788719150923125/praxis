extends CanvasLayer
class_name GenerativeEditor

## GenerativeEditor - the neural synthesis path (VOICE_PLAN.md P4).
##
## Deliberately NOT a backend swap inside [SynthEditor]. The fishing game's
## economy is defined over the procedural engine's parameter space - difficulty
## is nearest-neighbour distance in a 25-dimensional trait-plus-genome vector,
## the toll anneals a genome toward the belt's forces - and a neural backend
## exposes a speaker id and three global scalars. One UI serving both would be
## built for the intersection of their capabilities, which is nearly empty.
##
## Everything DOWNSTREAM is shared, because a take is just a WAV: [Spectrum],
## the scenes, the exporter, the subtitles and the ` feedback console all work
## unchanged. The panel deliberately mirrors [SynthEditor]'s - same corner, same
## width, same title/hide/hint/text shape - so this reads as another ghost mode
## rather than a bolted-on window.
##
## THE SLIDING WINDOW. A chapter is far too long to synthesize up front: at
## roughly 4x real time, twenty minutes of narration is five minutes of silence
## before anything plays. So the text is cut into chunks at sentence boundaries
## and only [constant LOOKAHEAD] of them are ever in flight. The first chunk
## plays while the second is still being made, and each finished session pulls
## the next one in. First audio arrives in seconds however long the chapter is,
## and memory stays bounded.

## Set by main: open ONE generator session for the whole chapter.
var begin_stream: Callable     # begin_stream.call(fp, sample_rate, words) -> playback

const TAKE_DIR := "user://generative"
# The same file SynthEditor persists to, under its own section: one save for
# ghost, not one per mode. Debounce and exit-save mirror that editor exactly.
const CFG := "user://ghost.cfg"
const AUTOSAVE_DELAY_MS := 800
const LOOKAHEAD := 2           # chunks in flight; 1 would stall at every seam
# ...and a bound on how much finished audio may sit AHEAD of the playhead.
#
# LOOKAHEAD alone throttles concurrency, not depth: chunks were requested as
# fast as they arrived, so at ~4x real time the whole chapter ended up decoded
# and queued within a minute. Nothing was wrong with the audio, but a tone or
# pace change then had nothing left to affect - every remaining chunk was
# already made - so the switch appeared to take "ages". Keeping only a few
# seconds buffered is what makes the window actually slide.
# Must comfortably EXCEED the generator's own ring (measured 131071 frames,
# 5.94 s at 22050) - otherwise the ring drains while the next chunk is still
# being synthesized and playback stalls, which is heard as a pause every few
# chunks and seen as the visualizer freezing with it. 3.0 was below the ring and
# did exactly that. The cost is switch latency, which is why chunks are single
# sentences: small chunks keep the window shallow in TIME while still deep
# enough in SECONDS to never starve.
const LOOKAHEAD_SECONDS := 9.0
# The silence between sentences. The host inserts this BETWEEN sentences inside
# one chunk; with one sentence per chunk it has no pair to sit between, so the
# boundary is made here instead. Without it the reading runs sentences together
# - the "rushed" problem, reintroduced by the chunk size. This is the figure at
# Pause 1.0; the slider scales it, and the host is sent the same pair so the two
# sides of a seam agree (see _seam_gap).
# Ceiling for the Pause slider. 2 was the first guess and it was far too timid: measured
# end to end through the real host, scale 2 stretched a 4.3 s sentence by only 0.81 s -
# spread over three marks, which reads as no change at all. Scale 10 stretches the same
# sentence by 3.46 s and gives a colon a 1.7 s rest, which is unmistakable. The base table
# stays conservative (it is calibrated to what the model already does) and this opens the
# range instead, so the choice is the reader's rather than baked in.
const MAX_PAUSE_SCALE := 10.0
# Longest silence to leave between two sentences, however far the slider is pushed.
const SEAM_CEILING := 1.2
const SENTENCE_GAP := 0.32
# The punctuation marks the host is allowed to receive.
#
# A mark is phonemized as part of its word, so whatever is sent here has to have
# an entry in the selected voice's phoneme_id_map - piper.py raises loudly on a
# symbol that does not, and rightly so. This list is the contract: anything else
# a text might end a word with (a bare newline, an em dash, a stray glyph the
# normalizer let through) falls back to the coarse pause_after mapping instead of
# being forwarded verbatim.
const PUNCT_ALLOWED := [".", ",", "!", "?", ":", ";"]
# One sentence per chunk. Two made every tone or pace change wait for the larger
# buffer to drain; one halves that latency, and costs nothing now that the
# inter-sentence gap is inserted at chunk boundaries too (see _drain_ready).
const CHUNK_SENTENCES := 1
## Seconds of audio that must be queued before the first sample is heard, when there is
## no intro to serve as the lead. Chunks are one sentence, and a short opening sentence
## (1.13 s, measured) cannot cover the synthesis of a long second one (14.49 s), so
## playback starves without a floor here. It is a LOWER bound on latency-to-first-word,
## which is why it is not larger.
const LIVE_PREROLL := 2.5

# TONE PRESETS.
#
# VITS has no affect control - the only handles are pace, how much variation the
# model samples, and pitch. So a "tone" here is those three moved together, plus
# a nudge to the ambience bed, which carries more of the mood than people expect.
#
# The pitch shift is done by RESAMPLING, and the model compensates: to raise the
# voice by r we ask it to speak r times SLOWER, then play back r times faster.
# The two cancel in duration and leave only the pitch change, which avoids a
# phase vocoder entirely and is artifact-free at these depths. It does shift the
# formants with the pitch, so the speaker reads as a different SIZE - which is
# exactly what "spooky" (larger, lower) and "excited" (smaller, higher) want.
const TONE_PRESETS := {
	"Neutral":  {"pace": 1.00, "semis":  0.0, "noise": 0.667, "noise_w": 0.333, "amb": 0.0},
	"Warm":     {"pace": 0.96, "semis": -0.5, "noise": 0.60,  "noise_w": 0.35,  "amb": 0.1},
	"Serious":  {"pace": 0.92, "semis": -1.5, "noise": 0.50,  "noise_w": 0.25,  "amb": 0.1},
	"Excited":  {"pace": 1.15, "semis":  2.0, "noise": 0.85,  "noise_w": 0.50,  "amb": 0.0},
	"Spooky":   {"pace": 0.85, "semis": -3.0, "noise": 0.45,  "noise_w": 0.20,  "amb": 0.35},
}     # per chunk: fewer seams than 1, still quick to first audio

var _host: VoiceHost
var _panel: PanelContainer
var _text: TextEdit
var _voices: OptionButton
var _go: Button
var _status: Label
var _rate: HSlider
var _rate_row: HBoxContainer
var _pause: HSlider
var _voice_meta: Array = []
var _want_voice := ""          # remembered selection, applied once voices load

# the window
var _chunks: Array = []        # [{tokens, words}] planned up front, cheap
var _ready_takes: Array = []   # [{pcm, index}] synthesized, awaiting the push
var _next_to_request := 0
var _next_to_play := 0
var _in_flight := 0
var _req_chunk := {}           # request id -> chunk index

# the one continuous stream
var _playback: AudioStreamGeneratorPlayback
var _pending := PackedFloat32Array()   # decoded samples not yet handed to the ring
var _read := 0                 # cursor into _pending; slicing it per frame was
                               # an O(n) copy of the whole queue 60 times a
                               # second, which is its own source of hitching
var _sr := 22050
var _elapsed := 0.0            # seconds pushed so far: the offset for chunk N's timings
var _sub_words: Array = []     # shared BY REFERENCE with the Subtitles overlay
var subtitles: Node            # set by main; its clock is re-based here
var _pushed := 0               # frames handed to the ring, for the played-time clock
var _ring_capacity := 0        # measured, never computed - see _drain_ready
var _fx := VoiceFX.new()
var _fx_echo: HSlider
var _dynamics: HSlider
var _arc: HSlider
var _effort: HSlider

var _fx_res: HSlider
var _fx_room: HSlider
var _fx_presence: HSlider
var _fx_pad: HSlider
var _tone: OptionButton
var _speaker: SpinBox
var _speaker_row: HBoxContainer
var _stream_open := false      # explicit: a null playback must not retry forever
var _epoch := 0                # bumped on a pace change; stale replies are dropped
# THE PREDICTED TIMELINE. The scrub bar used to be scaled by how much audio had been
# DECODED, which grows as the reading is synthesized - so the bar's own length changed
# under the pointer and dragging to "near the end" meant near the end of the first thirty
# seconds. A timeline has to know how long the thing is before it plays it.
var _repace_timer: Timer
var _scene_hold: HSlider
var _flourish: HSlider
var _intro: HSlider
var _lead_in := 0.0        # the intro seeded into _pending by _plan, in seconds
var _outro: HSlider
var _dirty := false
var _last_edit_ms := 0


func _ready() -> void:
	layer = 10
	DirAccess.make_dir_recursive_absolute(TAKE_DIR)
	_build_panel()
	_host = VoiceHost.new()
	add_child(_host)
	_host.host_ready.connect(_on_host_ready)
	_host.failed.connect(_on_failed)
	_host.progress.connect(func(_s: String, m: String) -> void: _set_status(m))
	_host.synthesized.connect(_on_synthesized)
	_load_persisted()
	_set_status("Starting the voice host…")
	_host.start()


## Feed the ring. The generator is the only thing keeping the session alive, so
## this must never fall behind; it is a few array copies a frame.
func _process(_delta: float) -> void:
	_process_persist()
	if _playback == null:
		return

	# RE-BASE THE SUBTITLE CLOCK FIRST, and unconditionally.
	#
	# This used to sit at the BOTTOM of the push block, after an early return on
	# an empty queue - so the moment synthesis got ahead of playback and there
	# was nothing left to push, the re-base stopped running and time_base froze.
	# Spectrum's clock kept advancing against a stale base, _now() ran past the
	# last word, and the overlay drew nothing from then on. Reported as: the
	# subtitles worked briefly, then disappeared and never came back.
	#
	# It belongs here because it describes PLAYBACK, which continues whether or
	# not there is new audio to hand over.
	if subtitles != null and is_instance_valid(subtitles) and _ring_capacity > 0:
		var queued := _ring_capacity - int(_playback.get_frames_available())
		var played := float(maxi(0, _pushed - maxi(queued, 0))) / float(_sr)
		subtitles.time_base = Spectrum.current.time - played

	# The window has to be topped up as it drains. Requests used to be driven
	# only by chunk ARRIVAL, which stops the moment the buffer bound is hit.
	if not _chunks.is_empty() and _next_to_request < _chunks.size():
		_pump()

	if _pending.size() - _read <= 0:
		return
	var room := int(_playback.get_frames_available())
	if room <= 0:
		return
	var avail := _pending.size() - _read
	var n := mini(room, avail)
	if n <= 0:
		return
	# the ambience runs HERE rather than in the host: it is stateful across
	# chunk boundaries, so a seam must not reset the echo tail or the ring
	var mono: PackedFloat32Array = _fx.process(_pending.slice(_read, _read + n))
	var buf := PackedVector2Array()
	buf.resize(n)
	for i in n:
		var v := mono[i]
		buf[i] = Vector2(v, v)
	_playback.push_buffer(buf)
	_pushed += n
	_read += n
	# compact only when the consumed head is worth reclaiming, not every frame
	if _read > 0 and (_read == _pending.size() or _read > 4 * _sr):
		_pending = _pending.slice(_read)
		_read = 0


func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.keycode == KEY_F2:
		_panel.visible = not _panel.visible


# --- panel (mirrors SynthEditor._build_panel) --------------------------------


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
	title.text = "Generative"
	title.add_theme_font_size_override("font_size", 20)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_row.add_child(title)
	var hide := Button.new()
	hide.text = "–"
	hide.tooltip_text = "Hide panel (F2)"
	hide.custom_minimum_size = Vector2(28, 28)
	hide.pressed.connect(func() -> void: _panel.visible = false)
	title_row.add_child(hide)

	var hint := Label.new()
	hint.text = "Paste a chapter. It is spoken in chunks, so the show starts while the rest is still being made. Inline phonetics still work: [K AE T]."
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.add_theme_font_size_override("font_size", 12)
	hint.modulate = Color(1, 1, 1, 0.6)
	box.add_child(hint)

	_text = TextEdit.new()
	_text.custom_minimum_size = Vector2(360, 180)
	_text.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_text.placeholder_text = "Once upon a time..."
	_text.tooltip_text = "The script to read. Paste a whole chapter - it is cut into sentences and only a couple are ever synthesized ahead, so the first words play within seconds however long it is. Square brackets pin a pronunciation: [B IY1 UW0 K S]."
	_text.text_changed.connect(func() -> void:
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec()
		if not _chunks.is_empty():
			_go.text = "Speak ●"     # the reading no longer matches the box
		)
	box.add_child(_text)

	var vrow := HBoxContainer.new()
	vrow.add_theme_constant_override("separation", 8)
	box.add_child(vrow)
	_voices = OptionButton.new()
	_voices.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_voices.tooltip_text = "Which Piper model reads. Each is a different person with its own accent, recording and licence - the licence appears under the panel when you pick one. Changing this regenerates whatever has not been played yet."
	_voices.item_selected.connect(func(_i: int) -> void:
		_show_voice_license()
		# A voice change is not a re-pace. _repace keeps the stream and appends
		# at the new setting, which for a VOICE would splice a second speaker
		# mid-narration; and leaving the old requests in flight keeps both
		# models loaded and chunking, halving the throughput of each. So: kill
		# the session and read again in the new voice.
		if not _chunks.is_empty():
			_restart_speaking()
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec())
	vrow.add_child(_voices)
	_go = Button.new()
	_go.text = "Speak"
	_go.tooltip_text = "Read the script aloud and drive the visuals from it. Press again while it is running to stop. The scenes react to the narration exactly as they would to music."
	_go.disabled = true
	_go.pressed.connect(_on_speak)
	vrow.add_child(_go)

	# Multi-speaker checkpoints carry hundreds of readers under one model -
	# libritts-high has 904 - and they are the only way to change WHO is
	# reading without changing the model. Hidden entirely for single-speaker
	# voices, like every other capability-driven control here.
	_speaker_row = HBoxContainer.new()
	_speaker_row.add_theme_constant_override("separation", 8)
	_speaker_row.visible = false
	box.add_child(_speaker_row)
	var sl2 := Label.new()
	sl2.text = "Speaker"
	sl2.custom_minimum_size = Vector2(72, 0)
	sl2.add_theme_font_size_override("font_size", 12)
	_speaker_row.add_child(sl2)
	_speaker = SpinBox.new()
	_speaker.min_value = 0
	_speaker.step = 1
	_speaker.tooltip_text = "Which reader, on a model that holds more than one (libritts carries 904). Same model and same accent, a different person - so it changes WHO is reading, not how. Greyed out on single-speaker voices. Regenerates the un-played chunks."
	_speaker.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_speaker.value_changed.connect(func(_v: float) -> void:
		# same model, different reader: only the un-played chunks need redoing
		if not _chunks.is_empty():
			_repace()
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec())
	_speaker_row.add_child(_speaker)

	var trow := HBoxContainer.new()
	trow.add_theme_constant_override("separation", 8)
	box.add_child(trow)
	var tl := Label.new()
	tl.text = "Tone"
	tl.custom_minimum_size = Vector2(72, 0)
	tl.add_theme_font_size_override("font_size", 12)
	trow.add_child(tl)
	_tone = OptionButton.new()
	_tone.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_tone.tooltip_text = "The reading's overall manner: pace, pitch, and how much the model varies itself, moved together. The pitch shift moves the formants with it, so the reader reads as a different SIZE - which is what makes Spooky larger and lower, Excited smaller and higher. Regenerates the un-played chunks."
	for k in TONE_PRESETS:
		_tone.add_item(String(k))
	_tone.item_selected.connect(func(_i: int) -> void:
		# a tone changes the model's own parameters, so un-played chunks have
		# to be regenerated - the same path a pace change takes
		if not _chunks.is_empty():
			_repace()
		# ...and it nudges the ambience bed, which is a LIVE effect: without this
		# the preset's contribution only arrived at the next stream open, so a
		# tone change mid-reading moved the voice and not the room around it.
		_sync_fx(_fx)
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec())
	trow.add_child(_tone)

	# Shown only if the backend reports duration_control - the surface is built
	# from capabilities(), never from assumptions about what a model can do.
	_rate_row = HBoxContainer.new()
	_rate_row.add_theme_constant_override("separation", 8)
	_rate_row.visible = false
	box.add_child(_rate_row)
	var rl := Label.new()
	rl.text = "Pace"
	rl.add_theme_font_size_override("font_size", 12)
	_rate_row.add_child(rl)
	_rate = HSlider.new()
	# 0.7 was arbitrary. VITS length_scale is stable well past it, and an
	# audiobook read often wants slower than "slightly slow".
	_rate.min_value = 0.4
	_rate.max_value = 1.6
	_rate.step = 0.05
	_rate.value = 1.0
	_rate.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_rate.tooltip_text = "Speaking rate. Takes effect from the next chunk - already-generated audio keeps its pace."
	_rate.value_changed.connect(func(_v: float) -> void: _repace_timer.start())
	_rate_row.add_child(_rate)

	# PAUSE. How long the reading rests on punctuation, as a multiple of the
	# host's default rests (roughly: a tenth of a second on a comma, a quarter on
	# a colon, a third at a sentence end). VITS runs straight through a comma and
	# especially through a colon, so the silence is spliced in around the mark by
	# the host rather than asked of the model.
	#
	# Always visible, unlike Pace: this needs no duration_control, because the
	# host is inserting silence rather than asking the model for a different
	# length. And it is NOT a live buffer effect like Echo/Ambience - it changes
	# what gets synthesized, so it takes the debounced re-plan path below, the
	# same one a pace change takes.
	var prow := HBoxContainer.new()
	prow.add_theme_constant_override("separation", 8)
	box.add_child(prow)
	var pl := Label.new()
	pl.text = "Pause"
	pl.custom_minimum_size = Vector2(72, 0)
	pl.add_theme_font_size_override("font_size", 12)
	prow.add_child(pl)
	_pause = HSlider.new()
	_pause.min_value = 0.0
	_pause.max_value = 10.0
	_pause.step = 0.05
	_pause.value = 1.0
	_pause.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_pause.tooltip_text = "How long to rest on punctuation - commas, semicolons, colons and sentence ends. 1 is the natural rest the model already takes; 0 runs straight through. Measured on one sentence: 1 adds about a third of a second across it, 2 about four fifths, 10 about three and a half - at 10 a colon rests for well over a second. Takes effect from the next chunk; already-generated audio keeps its pauses."
	_pause.value_changed.connect(func(_v: float) -> void: _repace_timer.start())
	prow.add_child(_pause)

	# Debounced rather than immediate: a slider drag emits a value per pixel,
	# and each change throws away work in flight. One re-plan per gesture.
	_repace_timer = Timer.new()
	_repace_timer.wait_time = 0.3
	_repace_timer.one_shot = true
	_repace_timer.timeout.connect(func() -> void:
		_repace()
		_persist())
	add_child(_repace_timer)

	# Ambience: the same effects Synthesis has, over any PCM. Off by default -
	# a narration take should sound like a reading unless asked otherwise.
	# DELIVERY, not effect. These two are the discourse layer Piper structurally
	# cannot supply: it never sees the paragraph, so every sentence starts from the
	# same register at the same rate. See piper._discourse_plan for the rules.
	_dynamics = _fx_slider(box, "Dynamics", 0.5,
		"How much the delivery varies from sentence to sentence. The model reads every sentence "
		+ "the same way because it never sees the paragraph, so this supplies what it cannot: "
		+ "sentences slow as they approach the end of a paragraph, long ones run a little quicker "
		+ "and short ones dwell, and the rhythm shifts between neighbours. 0 is the flat delivery; "
		+ "0.4 is about a fifth slower by the last sentence of a paragraph, and the top of the "
		+ "dial is roughly two and a half times that - far enough to be too much for some "
		+ "material, which is the point of having the range.")
	_effort = _fx_slider(box, "Effort", 0.35,
		"Vocal effort across a paragraph - which is not the same as volume. A louder voice has a "
		+ "brighter source spectrum, not just a higher level, so this moves the two together: "
		+ "opening a paragraph with a little more push and easing off toward the end. Level "
		+ "alone reads as the reader standing further away; the brightness is what makes it read "
		+ "as them easing off instead. 0 is off, 1 is about 6dB and a firm tilt top to bottom.")
	_arc = _fx_slider(box, "Arc", 0.4,
		"Pitch shape ACROSS a paragraph, in semitones. A speaker opens a new paragraph in a higher "
		+ "register and settles as it goes, then resets on the next - it is what makes a paragraph "
		+ "land rather than just stop. 0 is off, 0.4 is three semitones top to bottom and 1 is "
		+ "seven and a half. Past about 0.5 the resampling audibly moves the formants too, so the "
		+ "reader starts changing SIZE between paragraphs - which is either the effect you want "
		+ "or the reason to back off.")
	_fx_echo = _fx_slider(box, "Echo", 0.0,
		"The room the voice is in. Low settings are a small close room - the repeat is short enough "
		+ "to fuse with the voice rather than be heard as a separate sound. It opens out into a "
		+ "distinct slapback with a long tail as you raise it, so this changes the SIZE of the "
		+ "space, not just how loud it is.")
	# THE ROOM, the same one Masking has - see [RoomFX], which owns the dial and
	# leaves each mode only its engine. It is a separate control from Echo above
	# and has to be: Echo is discrete repeats, which the ear counts, and this is
	# the diffuse tail behind them, which it cannot. A voice with only the first
	# sounds like a voice in a corridor; with only the second, like a voice in a
	# hall. Most rooms are both.
	_fx_room = _fx_slider(box, "Room", 0.0,
		"The size of the space around the reader. Where Echo is a repeat you can hear arrive, "
		+ "this is the diffuse tail behind it - no countable events, just the room answering. "
		+ "Low is a small studio whose tail is gone before the next word; the top is a hall that "
		+ "rings for seconds. Resonance sets its colour, exactly as it does in Masking: dark and "
		+ "swallowed at 0, bright and ringing at 1. It is baked into the exported take, so what "
		+ "you hear here is what renders.")
	_fx_res = _fx_slider(box, "Resonance", 0.0,
		"Sympathetic tones tuned to the reader's own pitch, ringing when the voice rings and dying "
		+ "when it stops - the room answering, rather than a chord played over the top. It follows "
		+ "the speaking register, so it moves with the voice instead of fighting it.")
	_fx_presence = _fx_slider(box, "Presence", 1.0,
		"How close the reader is. 1 is right here; lower moves them away, dulling the high end "
		+ "first the way air does and only then dropping the level. Distance is a filter before it "
		+ "is a volume, which is why this is not a master gain.")
	_fx_pad = _fx_slider(box, "Ambience", 0.0,
		"A sustained ambient bed underneath, in the reader's own key - long tones that keep "
		+ "sounding through the pauses, rather than reverb of the voice. It ducks under speech and "
		+ "swells in the gaps, and it is what plays alone through the Intro hold.")

	# --- THE PICTURE, not the voice ------------------------------------------
	# These two live here, with every other option, rather than off in the shared
	# chrome: one place to reach for a setting beats an architecturally tidier
	# second home nobody finds. They drive the [Director], so they persist and
	# apply in every mode - this panel is just where you turn them.
	#
	# NOT called "pace" or "pacing". Those words are already taken by the voice
	# slider three rows up, and the same word on two sliders that do unrelated
	# things is how someone ends up afraid to touch either.
	var sep := HSeparator.new()
	box.add_child(sep)
	_scene_hold = _director_slider(box, "Scene hold", Director.PACING_MIN, Director.PACING_MAX, 0.05,
		Director.pacing,
		"How long each visual scene stays on screen before the show cuts to the next. 1 is the "
		+ "default; 2 roughly doubles it. The music still decides where in the range each scene "
		+ "lands, so the variety is kept - the whole range just moves. Nothing to do with the "
		+ "speaking voice.",
		func(v: float) -> void: Director.set_pacing(v))
	_flourish = _director_slider(box, "Flourishes", Director.FLOURISH_MIN, Director.FLOURISH_MAX, 0.05,
		Director.flourish,
		"How often the show breaks its rhythm - a burst of two or three quick cuts, or a run of "
		+ "beat-synced punches on the current scene. 0 turns them off entirely, 1 is the default. "
		+ "Set this to 0 first if the cutting feels busy: it separates 'too often' from 'too fast'.",
		func(v: float) -> void: Director.set_flourish(v))
	_intro = _director_slider(box, "Intro", Director.INTRO_MIN, Director.INTRO_MAX, 0.5,
		Director.intro_hold,
		"Seconds of held opening before the narration starts, so the video fades up onto "
		+ "something instead of beginning mid-word. If Ambience is on, the bed plays alone "
		+ "through it. Applies to the next render, not the take already playing. Under about "
		+ "4s the bed is still swelling when the voice arrives; 0 turns the intro off.",
		func(v: float) -> void: Director.set_intro_hold(v))
	_outro = _director_slider(box, "Outro", Director.OUTRO_MIN, Director.OUTRO_MAX, 0.5,
		Director.outro_hold,
		"Seconds held after the last word, fading picture and sound out together. The "
		+ "ambience bed takes about 7s to decay, so a shorter outro will cut its tail off. "
		+ "0 ends the video on the final syllable.",
		func(v: float) -> void: Director.set_outro_hold(v))

	_status = Label.new()
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status.add_theme_font_size_override("font_size", 11)
	_status.add_theme_color_override("font_color", Color(0.55, 0.95, 0.75, 0.85))
	box.add_child(_status)


# --- persistence (mirrors SynthEditor._persist / _load_persisted) ------------


## Debounced: a keystroke marks the draft dirty, and the write happens once the
## typing stops. Saving per keystroke would hit the disk on every character.
func _process_persist() -> void:
	if _dirty and Time.get_ticks_msec() - _last_edit_ms >= AUTOSAVE_DELAY_MS:
		_persist()


func _persist() -> void:
	_dirty = false
	var cfg := ConfigFile.new()
	cfg.load(CFG)                     # read-modify-write: never clobber [synth]
	cfg.set_value("generative", "text", _text.text)
	cfg.set_value("generative", "voice", _selected_voice_id())
	cfg.set_value("generative", "pace", _rate.value)
	cfg.set_value("generative", "pause", _pause.value)
	cfg.set_value("generative", "echo", _fx_echo.value)
	cfg.set_value("generative", "dynamics", _dynamics.value)
	cfg.set_value("generative", "arc", _arc.value)
	cfg.set_value("generative", "effort", _effort.value)
	cfg.set_value("generative", "resonance", _fx_res.value)
	cfg.set_value("generative", "room", _fx_room.value)
	cfg.set_value("generative", "presence", _fx_presence.value)
	cfg.set_value("generative", "ambience", _fx_pad.value)
	cfg.set_value("generative", "tone", _tone.selected)
	cfg.set_value("generative", "speaker", int(_speaker.value))
	cfg.save(CFG)


func _load_persisted() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(CFG) != OK:
		return
	_text.text = str(cfg.get_value("generative", "text", ""))
	_rate.value = float(cfg.get_value("generative", "pace", 1.0))
	_pause.value = float(cfg.get_value("generative", "pause", 1.0))
	_want_voice = str(cfg.get_value("generative", "voice", ""))
	_fx_echo.value = float(cfg.get_value("generative", "echo", 0.0))
	_dynamics.value = float(cfg.get_value("generative", "dynamics", 0.5))
	_arc.value = float(cfg.get_value("generative", "arc", 0.4))
	_effort.value = float(cfg.get_value("generative", "effort", 0.35))
	_fx_res.value = float(cfg.get_value("generative", "resonance", 0.0))
	_fx_room.value = float(cfg.get_value("generative", "room", 0.0))
	_fx_presence.value = float(cfg.get_value("generative", "presence", 1.0))
	_fx_pad.value = float(cfg.get_value("generative", "ambience", 0.0))
	_tone.select(clampi(int(cfg.get_value("generative", "tone", 0)), 0, TONE_PRESETS.size() - 1))
	_speaker.value = float(cfg.get_value("generative", "speaker", 0))
	# Last, and once: the Tone preset nudges the ambience bed, so the chain cannot
	# be synced before the preset is chosen.
	_sync_fx(_fx)


func _selected_voice_id() -> String:
	var i := _voices.selected
	if i < 0 or i >= _voice_meta.size():
		return _want_voice
	return String(_voice_meta[i].get("id", ""))


func _exit_tree() -> void:
	if _dirty:
		_persist()


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST and _dirty:
		_persist()


## One labelled 0..1 slider, live: these are buffer effects, so a change is
## audible on the very next frame rather than needing a re-synthesis.
## One labelled slider that drives the [Director] directly. Unlike the voice controls these need
## no re-plan and no persistence here - the Director clamps, applies immediately to the scene on
## screen, and owns its own saved value.
func _director_slider(box: VBoxContainer, name: String, lo: float, hi: float, step: float,
		initial: float, tip: String, apply: Callable) -> HSlider:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)
	var l := Label.new()
	l.text = name
	l.custom_minimum_size = Vector2(72, 0)
	l.add_theme_font_size_override("font_size", 12)
	row.add_child(l)
	var sl := HSlider.new()
	sl.min_value = lo
	sl.max_value = hi
	sl.step = step
	sl.value = clampf(initial, lo, hi)
	sl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	sl.tooltip_text = tip
	sl.value_changed.connect(apply)
	row.add_child(sl)
	return sl


## A voice/delivery slider. `tip` is not optional in practice: an unlabelled dial
## is a dial nobody touches, and every other slider in this panel carries one.
## Give every chunk its place in the reading, so the delivery plan can be a plan.
##
## THE BUG THIS FIXES. `_discourse_plan` shapes a sentence by where it sits in the unit
## it was handed - and the unit it is handed is one REQUEST, which is one sentence
## (CHUNK_SENTENCES = 1). So every sentence was simultaneously the first and last of its
## own unit: position 0 of 1, every time. Final lengthening never once fired, and the
## "arc" resolved to the same constant offset on every sentence in the chapter - a
## transposition wearing an arc's name. It measured correctly when I tested it, because I
## tested it by handing the whole paragraph over in one call, which is a path the editor
## never takes. The audible difference between Arc 0 and Arc 1 was real; it just was not
## an arc.
##
## TWO NESTED LEVELS, because discourse structure is nested and so are its pitch resets -
## the size of a reset scales with the depth of the boundary it follows (Grosz & Sidner;
## Hirschberg & Nakatani). `u` is the sentence's place in its paragraph and `v` its
## paragraph's place in a run of them. That second level is what a chapter of one-sentence
## paragraphs needs: at `u` the paragraph collapses to a point and every one of them would
## otherwise be handed the identical contour, which is exactly the flatness reported.
## Sections are runs of SECTION_PARAS paragraphs, so the slow movement is phase-locked to
## the prose the same way the fast one is.
const SECTION_PARAS := 5

func _place_chunks(out: Array, body: String) -> void:
	# Paragraph boundaries come from the SOURCE text (a blank line), which is the only place
	# they exist - Phonemes.parse hands back a flat run of sentences and knows nothing about
	# them. Counting sentence-final marks per paragraph maps one onto the other without
	# parsing the body twice.
	var per_para: Array = []
	for para in body.split("\n\n", false):
		var t := String(para).strip_edges()
		if t.is_empty():
			continue
		var c := 0
		for ch in t:
			if ch == "." or ch == "!" or ch == "?":
				c += 1
		per_para.append(maxi(1, c))
	if per_para.is_empty():
		per_para = [maxi(1, out.size())]
	# sentence index -> (paragraph index, position within it)
	var pi := 0
	var within := 0
	for i in out.size():
		if within >= int(per_para[mini(pi, per_para.size() - 1)]) and pi < per_para.size() - 1:
			pi += 1
			within = 0
		var n := int(per_para[mini(pi, per_para.size() - 1)])
		# A ONE-SENTENCE PARAGRAPH SITS IN THE MIDDLE OF ITS OWN ARC, not at the start
		# of it. Placing it at 0 hands it the opening register of a paragraph it also
		# ends, so a run of them takes a constant lift - measured +2.3 semitones at Arc 1,
		# which is a transposition again, just a subtler one. The midpoint makes the
		# paragraph term neutral where the paragraph has no extent, and leaves the section
		# to supply all of the movement, which is the whole point of having one.
		out[i]["plan_u"] = 0.5 if n <= 1 else float(within) / float(n - 1)
		# ...and the paragraph's place in its section. A one-paragraph section would put
		# every paragraph at 0 again, which is the same trap one level up.
		var sp := pi % SECTION_PARAS
		out[i]["plan_v"] = float(sp) / float(maxi(1, SECTION_PARAS - 1))
		within += 1


func _fx_slider(box: VBoxContainer, name: String, initial: float, tip := "") -> HSlider:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	row.tooltip_text = tip          # the gap between the label and the slider
	box.add_child(row)
	var l := Label.new()
	l.text = name
	l.custom_minimum_size = Vector2(72, 0)
	l.add_theme_font_size_override("font_size", 12)
	# A Label ignores the mouse by default, so pointing at the NAME - which is
	# what anyone actually does - would otherwise show nothing.
	l.mouse_filter = Control.MOUSE_FILTER_STOP
	l.tooltip_text = tip
	row.add_child(l)
	var sl := HSlider.new()
	sl.tooltip_text = tip
	sl.min_value = 0.0
	sl.max_value = 1.0
	sl.step = 0.01
	sl.value = initial
	sl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	sl.value_changed.connect(func(_v: float) -> void:
		_sync_fx(_fx)
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec())
	row.add_child(sl)
	return sl


## Push the panel's settings onto a [VoiceFX] chain. ONE definition, because there
## were four lists of assignments - the slider callback, the config load, the
## stream opening and the export - and a dial had to appear in all four to work
## everywhere. Adding Room to three of them would have left one path silently
## dry, which is the same class of bug [method _pad_level] was written to end:
## live and render disagreeing about the room around the voice.
func _sync_fx(fx: VoiceFX) -> void:
	if fx == null:
		return
	fx.echo_wet = _fx_echo.value
	fx.resonance = _fx_res.value
	fx.presence = _fx_presence.value
	fx.pad = _pad_level()
	# One dial, so [RoomFX] does the collapsing: size and wet open together, and
	# Resonance colours the tail the same way it does on Masking's bus.
	fx.room.from_dial(_fx_room.value, _fx_res.value)


func _set_status(msg: String) -> void:
	if _status != null:
		_status.text = msg


# --- host --------------------------------------------------------------------


func _on_host_ready(backends: PackedStringArray) -> void:
	_set_status("Voice host up (%s). Loading voices…" % ", ".join(backends))
	_host.capabilities()
	_host.list_voices()


func _on_failed(stage: String, message: String) -> void:
	_go.disabled = false
	_in_flight = maxi(0, _in_flight - 1)
	_set_status("%s failed: %s" % [stage, message])


func _on_synthesized(id: int, result: Dictionary) -> void:
	if result.has("backends"):
		var caps: Dictionary = result["backends"]
		for name in caps:
			if bool((caps[name] as Dictionary).get("duration_control", false)):
				_rate_row.visible = true
		return
	if result.has("voices"):
		_fill_voices(result["voices"])
		return
	if not _req_chunk.has(id):
		return
	var meta: Dictionary = _req_chunk[id]
	_req_chunk.erase(id)
	# THE SLOT IS FREED WHETHER OR NOT THE RESULT IS WANTED, and this has to happen BEFORE
	# the epoch check. It did not, and that is what made seeking slower than a cold start:
	# an abandoned request still occupies the host - the protocol has no cancel - so a jump
	# that zeroed the counter let _pump send a full lookahead on top of work already
	# queued, and the chunk actually being waited for ended up third or fourth in line.
	# Counting abandoned work as in-flight means the new request enters the queue exactly
	# when a slot frees, which without a cancel is the best available.
	_in_flight = maxi(0, _in_flight - 1)
	if int(meta["epoch"]) != _epoch:
		# Superseded by a pace change or a jump. Nothing to keep, but the pump may now
		# have room it did not have a moment ago.
		_pump()
		return
	var idx: int = int(meta["idx"])

	var wav := String(result.get("wav", ""))
	if wav.is_empty():
		_set_status("The host returned no audio.")
		return
	_sr = int(result.get("sample_rate", _sr))
	_ready_takes.append({"wav": wav, "index": idx,
		"words": _words_for(idx, result.get("tokens", []))})
	_drain_ready()
	_pump()


func _fill_voices(voices: Array) -> void:
	_voice_meta = voices
	_voices.clear()
	for v in voices:
		var mark := "" if bool(v.get("installed", false)) else "  (downloads)"
		_voices.add_item("%s%s" % [String(v.get("name", v.get("id", "?"))), mark])
	if voices.is_empty():
		_set_status("No voices available - see the >_ log.")
		return
	if not _want_voice.is_empty():
		for i in voices.size():
			if String((voices[i] as Dictionary).get("id", "")) == _want_voice:
				_voices.select(i)
				break
	_go.disabled = false        # from here on Speak is always live: pressing it
	                            # mid-reading restarts with the current text
	_show_voice_license()


## Always shown: these checkpoints are licensed individually and the terms are
## inherited through fine-tuning in a way nothing machine-readable records.
func _show_voice_license() -> void:
	var i := _voices.selected
	if i < 0 or i >= _voice_meta.size():
		return
	var v: Dictionary = _voice_meta[i]
	var n := int(v.get("speakers", 1))
	_speaker_row.visible = n > 1
	if n > 1:
		_speaker.max_value = n - 1
		_speaker.value = clampf(_speaker.value, 0, n - 1)
	_set_status("%s - %s%s" % [String(v.get("id", "")), String(v.get("license", "unknown")),
		("  (%d speakers)" % n) if n > 1 else ""])


# --- the window ---------------------------------------------------------------


func _on_speak() -> void:
	if _host == null or not _host.is_up() or _voices.selected < 0:
		return
	var body := _text.text.strip_edges()
	if body.is_empty():
		_set_status("Nothing to speak yet.")
		return
	# _plan is a full teardown - stream closed, queues emptied, pending request
	# ids dropped so replies for the OLD text are discarded on arrival rather
	# than spliced into the new reading.
	_plan(body)
	if _chunks.is_empty():
		_set_status("Nothing speakable in that text.")
		return
	_go.text = "Speak"
	_set_status("Planned %d chunk(s). Synthesizing the first…" % _chunks.size())
	_pump()


## Tear the stream down and read again from the top. _plan clears the pending
## request map, so replies still in flight for the old voice are dropped on
## arrival rather than spliced in.
func _restart_speaking() -> void:
	var body := _text.text.strip_edges()
	if body.is_empty():
		return
	_plan(body)
	if _chunks.is_empty():
		return
	_set_status("Voice changed - reading again from the start…")
	_pump()


## Cut the text into chunks at sentence boundaries, recording per chunk the
## phone stream and the word spans within it. Planning is pure front end - no
## synthesis - so it stays cheap even for a whole chapter.
func _plan(body: String) -> void:
	_chunks = []
	_ready_takes = []
	_req_chunk = {}
	_next_to_request = 0
	_next_to_play = 0
	_in_flight = 0
	_stream_open = false
	_playback = null
	_pushed = 0
	_ring_capacity = 0
	_pending = PackedFloat32Array()
	_read = 0
	_elapsed = 0.0
	# No scrub hooks are registered by this editor (see the withdrawal note above
	# _repace), but clear them anyway: a file session opened earlier in the same process
	# may have left some, and they must not describe a reading that no longer exists.
	Spectrum.scrub_pos = Callable()
	Spectrum.scrub_len = Callable()
	Spectrum.scrub_seek = Callable()
	# THE INTRO, seeded HERE rather than at the moment the stream opens, and the
	# placement is the whole trick: `_elapsed` is the clock every word span and every
	# seam is measured against, so starting it at the end of the silence offsets the
	# entire reading - subtitles included - with no other code needing to know.
	#
	# Inserting it later would put it AFTER the first sentence, which is exactly the bug
	# this replaces: the audition spoke its opening line, then went quiet for five
	# seconds, then carried on.
	_lead_in = maxf(0.0, Director.intro_hold)
	if _lead_in > 0.0:
		_pending.resize(int(_lead_in * float(_sr)))
		_elapsed = _lead_in
	_sub_words.clear()          # cleared in place: Subtitles holds this by reference

	_chunks = _build_chunks(body)


## Pure: text in, chunk plan out. Shared by playback and by export, so an export
## cannot disturb the reading in progress.
func _build_chunks(body: String) -> Array:
	var out: Array = []
	var toks: Array = []
	var words: Array = []
	var sentences := 0
	# Subtitles windows the display by this index (Overlay._current_sentence),
	# so it must be a real, globally increasing sentence number. Hardcoding 0
	# put every word of the chapter in one "sentence", which the overlay then
	# tried to draw at once - it piled up and shrank toward nothing.
	var sentence_no := 0
	for sentence in Phonemes.parse(body):
		for w in sentence:
			var ph: Array = w.phones
			var st: Array = w.get("stress", [])
			var start := toks.size()
			# ghost's dictionary returns stress 0 for every phone of most
			# monosyllables, so no stress mark was ever emitted for them and the
			# model heard a flat reading. Promote the nucleus - but only for
			# CONTENT words, because eSpeak leaves function words unstressed and
			# promoting "the" to "ðˈʌ" is worse than leaving it alone. Phrasing
			# already made that distinction; w.stressed carries it.
			var lex: Array = []
			for i in ph.size():
				lex.append(int(st[i]) if i < st.size() else -1)
			if bool(w.get("stressed", false)) and not lex.has(1):
				var nucleus := Phonemes.stress_vowel(ph, st)
				if nucleus >= 0 and nucleus < lex.size():
					lex[nucleus] = 1
			# A token carries its SOURCE TEXT so the host can phonemize it with
			# eSpeak - whose transcriptions these voices were trained on - and
			# carries ghost's ARPAbet only as a fallback, or as an authored
			# [K AE T] override, which always wins.
			var arpa: Array = []
			for i in ph.size():
				# the stress digit lives in a parallel array; the ARPAbet path
				# needs it re-attached to the phone itself
				var d: int = lex[i]
				arpa.append(String(ph[i]) + (str(d) if d >= 0 else ""))
			# SEND THE REAL MARK. phonemes.gd already walks the terminal
			# punctuation off each word and records it verbatim (including the
			# case where a quote was hiding it - `early,"`); this used to
			# collapse it back to "." or "," via the coarse pause_after class,
			# which is where the reading lost its colons - heard as a comma -
			# and its question marks, heard as a full stop, so the interrogative
			# contour was never even asked for.
			#
			# pause_after stays as the fallback for a word whose mark has no
			# printable form (a line break), and the allow-list keeps anything
			# unexpected away from the voice's phoneme_id_map.
			var punct := String(w.get("punct", ""))
			if not PUNCT_ALLOWED.has(punct):
				punct = ""
				match String(w.pause_after):
					"stop": punct = "."
					"comma": punct = ","
			var tok := {"text": String(w.text), "punct": punct, "fallback": arpa}
			if bool(w.get("literal", false)):
				tok["arpa"] = arpa
			toks.append(tok)
			words.append({"text": String(w.get("display", w.text)),
				"index": start, "sentence": sentence_no})
		sentence_no += 1
		sentences += 1
		if sentences >= CHUNK_SENTENCES:
			out.append({"tokens": toks, "words": words})
			toks = []
			words = []
			sentences = 0
	if not toks.is_empty():
		out.append({"tokens": toks, "words": words})
	_place_chunks(out, body)
	return out


## The selected preset, and the resample ratio it implies.
func _tone_preset() -> Dictionary:
	var keys := TONE_PRESETS.keys()
	var i: int = clampi(_tone.selected, 0, keys.size() - 1)
	return TONE_PRESETS[keys[i]]


func _pitch_ratio() -> float:
	return pow(2.0, float(_tone_preset()["semis"]) / 12.0)


## The Pause setting, as the multiplier the host expects. Null-guarded because
## export_take and _pump are both reachable from outside this editor's own
## lifecycle, and a missing slider should read as "the default rests", never as
## zero - silently deleting every pause would be the worst possible failure.
## The delivery settings, as the host expects them. Null-guarded like _pause_scale:
## export_take and _pump are both reachable from outside this editor's lifecycle,
## and a missing slider must read as "the old constant delivery", never as a
## half-strength one nobody asked for.
func _delivery() -> Dictionary:
	return {
		"dynamics": _open_up(_dynamics.value if _dynamics != null else 0.0),
		# ARC IN SEMITONES peak-to-peak. 3 was the old ceiling and is an ordinary
		# paragraph reset; the curve below takes the top of the dial past it.
		"prosody_arc": _open_up(_arc.value if _arc != null else 0.0) * 3.0,
		"effort": _open_up(_effort.value if _effort != null else 0.0),
	}


## The top of a delivery dial, opened up - identity at the bottom, 2.5x at the top.
##
## The first ceilings were set where I thought the results stopped being good, which
## is not my call to make: at full travel Dynamics only slowed the last sentence of a
## paragraph by 18% and Arc reached 3 semitones, so the upper half of both dials was
## doing almost nothing and the reported symptom was exactly that. Rescaling linearly
## would have been wrong too, because the LOW half is where the settings that already
## sound right live - moving those to make room at the top would break what works to
## fix what doesn't. Cubic keeps the bottom of the travel where it was (0.25 moves by
## 9%) and opens the top (1.0 goes to 2.5x), so the far end is now genuinely too much
## for some material, and choosing that is the reader's business.
func _open_up(k: float) -> float:
	return k * (1.0 + 1.5 * k * k)


## The ambience bed's level: the slider PLUS the Tone preset's own nudge.
##
## ONE definition, because there were three and they disagreed. The stream-open path and
## the export both added the preset's `amb`; the config-load and slider-change paths set
## the bare slider value. So moving the Ambience slider during a reading silently dropped
## the preset's contribution and the export then put it back - live and render disagreeing
## about the bed by the whole of that nudge. This is NOT the reported symptom (it makes the
## export louder, not quieter, and the take WAV and the finished MP4 measure within a
## decibel of each other on the pad band), but the two paths have to agree about this or
## nothing measured on one says anything about the other.
func _pad_level() -> float:
	if _fx_pad == null:
		return 0.0
	return clampf(_fx_pad.value + float(_tone_preset()["amb"]), 0.0, 1.0)


func _pause_scale() -> float:
	if _pause == null:
		return 1.0
	return clampf(_pause.value, 0.0, MAX_PAUSE_SCALE)


## The silence at a chunk seam. The host inserts the same figure between
## sentences INSIDE a chunk, so a seam and an interior boundary stay the same
## length however the reading happens to have been cut up.
func _seam_gap() -> float:
	# Mid-sentence marks scale all the way - that is the complaint this slider answers.
	# The gap BETWEEN sentences is capped, because it is the one that compounds: at 10x an
	# uncapped seam is 3.2 s of dead air after every single sentence, which over a chapter is
	# not a longer rest, it is a broken reading. 1.2 s is already a long, deliberate beat.
	return minf(SENTENCE_GAP * _pause_scale(), SEAM_CEILING)


## Keep LOOKAHEAD chunks in flight and start playback as soon as the chunk we
## are waiting for exists.
## How much finished audio is waiting to be heard: decoded but not yet pushed,
## plus whatever is still sitting in the ring.
func _buffered_seconds() -> float:
	var queued := 0
	if _playback != null and _ring_capacity > 0:
		queued = maxi(0, _ring_capacity - int(_playback.get_frames_available()))
	return float(maxi(_pending.size() - _read, 0) + queued) / float(maxi(_sr, 1))


func _pump() -> void:
	while _in_flight < LOOKAHEAD and _next_to_request < _chunks.size() \
			and _buffered_seconds() < LOOKAHEAD_SECONDS:
		var idx := _next_to_request
		_next_to_request += 1
		_in_flight += 1
		var vid := String(_voice_meta[_voices.selected].get("id", ""))
		var t := _tone_preset()
		var r := _pitch_ratio()
		# length_scale = r / pace: the model speaks r times slower so that
		# playing back r times faster restores the intended pace
		var id := _host.request("", vid, TAKE_DIR + "/chunk_%d_%d.wav" % [idx, _epoch],
			{"length_scale": r / maxf(_rate.value * float(t["pace"]), 0.1),
			 "noise_scale": float(t["noise"]), "noise_w": float(t["noise_w"]),
			 "speaker": int(_speaker.value),
			 "sentence_gap": SENTENCE_GAP, "pause_scale": _pause_scale(),
			 "dynamics": _delivery()["dynamics"],
			 "prosody_arc": _delivery()["prosody_arc"],
			 "effort": _delivery()["effort"],
			 "plan_u": float(_chunks[idx].get("plan_u", 0.0)),
			 "plan_v": float(_chunks[idx].get("plan_v", 0.0)),
			 "tokens": _chunks[idx]["tokens"]}, null)
		_req_chunk[id] = {"idx": idx, "epoch": _epoch}



## Append every chunk that is ready AND next in order to the one stream. Out of
## order arrivals wait: audio has to go out in the order it was written.
func _drain_ready() -> void:
	while true:
		var found := -1
		for i in _ready_takes.size():
			if int(_ready_takes[i]["index"]) == _next_to_play:
				found = i
				break
		if found < 0:
			return
		var take: Dictionary = _ready_takes[found]
		_ready_takes.remove_at(found)
		_next_to_play += 1

		var pcm := _read_wav(String(take["wav"]))
		if pcm.is_empty():
			continue
		var ratio := _pitch_ratio()
		if absf(ratio - 1.0) > 0.001:
			pcm = _resample(pcm, ratio)
		if _next_to_play > 1:
			# a breath between sentences, at the seam the host cannot see.
			# Scaled by Pause like every other rest, or the control would do
			# nothing at all at CHUNK_SENTENCES = 1 - every sentence boundary in
			# the reading IS a seam, and they would all stay 0.32 s.
			var seam := _seam_gap()
			var gap := PackedFloat32Array()
			gap.resize(int(seam * float(_sr)))
			_pending.append_array(gap)
			_elapsed += seam
		# the model timed each chunk from zero; shift into stream time
		for w in take["words"]:
			var d: Dictionary = (w as Dictionary).duplicate()
			# the model timed this at its own (slower) rate; resampling divided
			# every duration by the ratio, so the timings must follow
			d["t0"] = float(d["t0"]) / ratio + _elapsed
			d["t1"] = float(d["t1"]) / ratio + _elapsed
			_sub_words.append(d)
		_pending.append_array(pcm)
		_elapsed += float(pcm.size()) / float(_sr)
		_set_status("Chunk %d of %d - %.0fs of audio ready."
			% [_next_to_play, _chunks.size(), _elapsed])
		# HOLD THE FIRST SAMPLE UNTIL THERE IS A LEAD. Chunks are one sentence each
		# (CHUNK_SENTENCES = 1), and sentence lengths are wildly uneven: measured on
		# chapter 3, the opening sentence renders to 1.13 s and the next to 14.49 s. The
		# stream used to open on whatever the first chunk happened to be, so playback
		# drained that 1.13 s and then starved for as long as the 14.49 s sentence took
		# to synthesize. Nothing was wrong with the audio - no chunk carries more than
		# 0.27 s of internal silence - it simply ran out.
		#
		# So wait for a real lead before starting. The intro doubles as that lead, which
		# is why this is one condition and not two, and the floor covers the case where
		# the intro is turned off entirely. Once open it stays open: the drain is
		# continuous from here and a mid-reading stall is the pump's business, not this.
		if not _stream_open:
			var have := float(_pending.size() - _read) / float(_sr)
			var lead := maxf(_lead_in, LIVE_PREROLL)
			if have < lead and _next_to_play < _chunks.size():
				continue          # keep accumulating; nothing is lost, it is all queued
			# same text, same music: the pad's note choices are seeded
			_fx.pad_seed = hash(_text.text)
			_fx.setup(_sr)
			# every dial onto the fresh chain, in one call - the preset's own nudge
			# to the bed included: a mood is carried by the room as much as by the
			# reading
			_sync_fx(_fx)
			# the session opens on the FIRST chunk and never again - that is the
			# whole point: one unbroken take, so the Director does not re-cut
			# and the harmonic seed does not re-derive every few sentences
			_stream_open = true
			if begin_stream.is_valid():
				_playback = begin_stream.call(hash(_text.text), _sr, _sub_words)
			# NO SCRUB HOOKS HERE. Seeking a live generator was implemented and is
			# WITHDRAWN - see the note above _seek_take.
			# MEASURE the ring, do not compute it. Godot sizes a generator's
			# buffer to a power of two (131071 frames measured), not to
			# STREAM_BUFFER * sample_rate (88200) - so the computed figure made
			# `queued` NEGATIVE, `played` start ~1.9 s ahead, and the subtitles
			# open in the middle of the text. An empty ring reports its whole
			# capacity as available, and it is empty exactly here, before the
			# first push.
			if _playback != null:
				_ring_capacity = int(_playback.get_frames_available())



## SEEKING A LIVE GENERATOR IS WITHDRAWN, and this note is the record of why.
##
## It was built, it worked in the sense that the playhead moved, and it was wrong in two
## ways that only showed up in use. A generator's ring cannot be cleared while playback is
## active, so every seek had to stop and restart the stream - and repeated restarts left
## the session audibly wrong, the voice doubling and then trebling as more seeks were made,
## with the Director's transitions stalling alongside it. Neither reproduced in a headless
## measurement (a freshly synthesized take autocorrelates clean, with no delayed copy), and
## guessing further at engine-side playback state without being able to run the UI is how
## the first two attempts at this shipped.
##
## The deeper problem is that it could not have served its purpose anyway. The reason to
## scrub is to check what a given moment WILL LOOK LIKE when exported - and a seek cannot
## answer that here, because the Director is a simulation rather than a function of time:
## its scene choice and hold schedule evolve from the events it has actually seen, so
## seeking forward shows the scene that happens to be up rather than the one the export
## will have. A control that answers a different question from the one being asked is worse
## than no control.
##
## What DOES answer it: render the take, then open the rendered file. A file boot is
## seekable for real (one operation, no restart - see Spectrum.seek) and replays the same
## deterministic show from the same seed, so scrubbing it shows exactly what the export
## will contain. `--scene <name>` remains the fastest way to inspect one scene.

## Re-generate everything not yet committed to the stream, at the new pace.
##
## Audio already pushed - and audio decoded and waiting in _pending - is
## finished business: its subtitles are placed and _elapsed has advanced past
## it, so rewriting it would desync the overlay. Everything from the next
## un-drained chunk onward is thrown away and re-requested, so the new pace
## lands within a chunk or two rather than at the end of the chapter.
func _repace() -> void:
	if _chunks.is_empty() or _next_to_play >= _chunks.size():
		return
	_epoch += 1                     # in-flight replies from the old pace are now stale
	_ready_takes.clear()
	# Same reasoning as _jump_to_chunk: the host keeps computing what it was given, so the
	# accounting for it has to survive or the pump will pile more on top.
	_next_to_request = _next_to_play
	_set_status("Pace %.2fx, pause %.2fx - regenerating from chunk %d…"
		% [_rate.value, _pause_scale(), _next_to_play + 1])
	_pump()


## Linear-interpolating resample. Reading at `ratio` samples per output sample
## raises the pitch by that factor and shortens the audio by it; the model was
## asked to speak proportionally slower, so the two cancel and only the pitch
## moves. Linear is enough here - the ratios are within a few semitones, so the
## interpolation error sits far below the voice.
func _resample(src: PackedFloat32Array, ratio: float) -> PackedFloat32Array:
	var n := int(float(src.size()) / ratio)
	if n <= 1:
		return src
	var out := PackedFloat32Array()
	out.resize(n)
	for i in n:
		var pos := float(i) * ratio
		var a := int(pos)
		var b := mini(a + 1, src.size() - 1)
		out[i] = lerpf(src[a], src[b], pos - float(a))
	return out


## PCM16 mono, as written by the voice host. The 44-byte canonical header is
## ours, so this does not need to be a general WAV parser.
func _read_wav(path: String) -> PackedFloat32Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return PackedFloat32Array()
	var bytes := f.get_buffer(f.get_length())
	f.close()
	if bytes.size() <= 44:
		return PackedFloat32Array()
	var n := (bytes.size() - 44) / 2
	var out := PackedFloat32Array()
	out.resize(n)
	for i in n:
		out[i] = float(bytes.decode_s16(44 + i * 2)) / 32768.0
	return out


# --- export ------------------------------------------------------------------


## Gate for the export button. The procedural path asks whether a seed has been
## caught; here the only requirements are text and a loaded voice.
func can_export_take() -> bool:
	return not _text.text.strip_edges().is_empty() \
		and not _voice_meta.is_empty() and _host != null and _host.is_up()


## Render the WHOLE text to one WAV, independent of whatever is playing.
##
## A coroutine, because the host answers asynchronously and the exporter awaits
## this. It deliberately does NOT reuse the playback queue: export must capture
## the entire reading including the part not yet spoken, and must not disturb a
## reading in progress.
func export_take() -> String:
	var body := _text.text.strip_edges()
	if body.is_empty() or _voices.selected < 0:
		return ""
	var chunks := _build_chunks(body)
	if chunks.is_empty():
		return ""
	var vid := String(_voice_meta[_voices.selected].get("id", ""))
	var t := _tone_preset()
	var ratio := _pitch_ratio()
	var stamp := Time.get_ticks_msec()
	var pcm := PackedFloat32Array()
	var words: Array = []
	var elapsed := 0.0

	for i in chunks.size():
		_set_status("Rendering for export: %d of %d…" % [i + 1, chunks.size()])
		var id := _host.request("", vid, TAKE_DIR + "/export_%d_%d.wav" % [stamp, i],
			{"length_scale": ratio / maxf(_rate.value * float(t["pace"]), 0.1),
			 "noise_scale": float(t["noise"]), "noise_w": float(t["noise_w"]),
			 "speaker": int(_speaker.value),
			 "sentence_gap": SENTENCE_GAP, "pause_scale": _pause_scale(),
			 # The SAME delivery the preview used. An export that re-derived its own
			 # would be a different performance from the one that was auditioned,
			 # which is the one thing the render must never be.
			 "dynamics": _delivery()["dynamics"],
			 "prosody_arc": _delivery()["prosody_arc"],
			 "effort": _delivery()["effort"],
			 "plan_u": float(chunks[i].get("plan_u", 0.0)),
			 "plan_v": float(chunks[i].get("plan_v", 0.0)),
			 "tokens": chunks[i]["tokens"]}, null)
		var res: Array = []
		while true:
			res = await _host.synthesized
			if int(res[0]) == id:
				break
		var out: Dictionary = res[1]
		var wav := String(out.get("wav", ""))
		if wav.is_empty():
			continue
		var part := _read_wav(wav)
		DirAccess.remove_absolute(ProjectSettings.globalize_path(wav))
		if absf(ratio - 1.0) > 0.001:
			part = _resample(part, ratio)
		if i > 0:
			var seam := _seam_gap()
			var gap := PackedFloat32Array()
			gap.resize(int(seam * float(_sr)))
			pcm.append_array(gap)
			elapsed += seam
		var by_index := {}
		for sp in out.get("tokens", []):
			by_index[int((sp as Dictionary).get("index", -1))] = sp
		for w in chunks[i]["words"]:
			var span: Variant = by_index.get(int(w["index"]))
			if span == null:
				continue
			words.append({"text": String(w["text"]), "sentence": int(w["sentence"]),
				"t0": float((span as Dictionary).get("t0", 0.0)) / ratio + elapsed,
				"t1": float((span as Dictionary).get("t1", 0.0)) / ratio + elapsed})
		pcm.append_array(part)
		elapsed += float(part.size()) / float(_sr)

	if pcm.is_empty():
		return ""
	# THE BOOKEND, written into the take itself. Held silence at the head and tail of the
	# render, so the video opens and closes on something rather than starting mid-word.
	#
	# It goes in HERE, before the effects chain, and that ordering is the entire point.
	# VoiceFX is a filter, not a source: given real samples to write into, the ambience
	# pad - which is an independent instrument on its own clock, unlike the resonance,
	# which can only ring when the voice excites it - swells through the intro and decays
	# through the outro. Pad the PCM afterwards and both ends are digital silence.
	var intro := maxf(0.0, Director.intro_hold)
	var outro := maxf(0.0, Director.outro_hold)
	if intro > 0.0 or outro > 0.0:
		var padded := PackedFloat32Array()
		padded.resize(int(intro * float(_sr)) + pcm.size() + int(outro * float(_sr)))
		var head := int(intro * float(_sr))
		for i in pcm.size():
			padded[head + i] = pcm[i]
		pcm = padded

	# the ambience the user has been listening to belongs in the render, so the
	# export matches the audition - a fresh chain, since the live one is
	# mid-reading and carries its own tails
	var fx := VoiceFX.new()
	fx.pad_seed = hash(body)
	fx.setup(_sr)
	_sync_fx(fx)
	# SEED THE KEY, or the intro is silent anyway. The pad picks its tonic from the
	# tracked pitch of the voice, and the tracker returns 0 on silence - so during a
	# leading pad of pure zeros `_tonic` never rises off 0, `_start_tone` refuses to
	# schedule anything, and the bed only begins once the narration has already started,
	# which is precisely backwards. Measuring the voice FIRST and handing the chain its
	# key up front is what lets the bed be playing before the first word.
	if intro > 0.0:
		fx.prime_key(pcm)
	pcm = fx.process(pcm)

	var path := TAKE_DIR + "/take_%d.wav" % stamp
	var abs_path := _write_wav(path, pcm)
	if not words.is_empty() or intro > 0.0 or outro > 0.0:
		var side := FileAccess.open(path.get_basename() + ".json", FileAccess.WRITE)
		if side != null:
			# Word timings shift with the audio they describe. Doing it here, once,
			# keeps every consumer honest: the karaoke overlay, the live session and
			# the export render all read this file and none of them needs to know a
			# bookend exists.
			var shifted: Array = []
			for w in words:
				var d: Dictionary = (w as Dictionary).duplicate()
				d["t0"] = float(d.get("t0", 0.0)) + intro
				d["t1"] = float(d.get("t1", 0.0)) + intro
				shifted.append(d)
			side.store_string(JSON.stringify({
				"words": shifted, "bookend": {"in": intro, "out": outro}}))
			side.close()
	_set_status("Rendered %.1fs for export (%.0fs intro, %.0fs outro)." % [elapsed, intro, outro])
	return abs_path


## PCM16 mono WAV, written atomically - the exporter's render process may open
## this file while we are still writing it otherwise, which is how a truncated
## take once made a render record silence forever.
func _write_wav(path: String, pcm: PackedFloat32Array) -> String:
	var tmp := path + ".part"
	var f := FileAccess.open(tmp, FileAccess.WRITE)
	if f == null:
		return ""
	var bytes := PackedByteArray()
	bytes.resize(pcm.size() * 2)
	for i in pcm.size():
		bytes.encode_s16(i * 2, int(clampf(pcm[i], -1.0, 1.0) * 32767.0))
	f.store_buffer("RIFF".to_ascii_buffer()); f.store_32(36 + bytes.size())
	f.store_buffer("WAVE".to_ascii_buffer()); f.store_buffer("fmt ".to_ascii_buffer())
	f.store_32(16); f.store_16(1); f.store_16(1); f.store_32(_sr)
	f.store_32(_sr * 2); f.store_16(2); f.store_16(16)
	f.store_buffer("data".to_ascii_buffer()); f.store_32(bytes.size())
	f.store_buffer(bytes)
	f.close()
	var abs_tmp := ProjectSettings.globalize_path(tmp)
	var abs_out := ProjectSettings.globalize_path(path)
	if DirAccess.rename_absolute(abs_tmp, abs_out) != OK:
		return abs_tmp
	return abs_out


# --- subtitles ----------------------------------------------------------------


## Rebuild word timings from the per-phone durations the model returned.
##
## The backend hands back one entry per phone we sent, in order, so the word
## spans recorded at plan time index straight into it. This is what restores the
## karaoke overlay: main._attach_subtitles picks up any take that has a sidecar,
## so the neural path gets subtitles with no special casing anywhere.
func _words_for(idx: int, spans: Array) -> Array:
	if idx < 0 or idx >= _chunks.size() or spans.is_empty():
		return []
	var by_index := {}
	for s in spans:
		by_index[int((s as Dictionary).get("index", -1))] = s
	var out: Array = []
	for w in _chunks[idx]["words"]:
		var span: Variant = by_index.get(int(w["index"]))
		if span == null:
			continue
		out.append({"text": String(w["text"]), "sentence": int(w["sentence"]),
			"t0": float((span as Dictionary).get("t0", 0.0)),
			"t1": float((span as Dictionary).get("t1", 0.0))})
	return out


