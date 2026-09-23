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
## The other half of begin_stream: closes the stream, detaches the Director from the stage and
## frees the subtitle overlay. Optional - an owner that does not set it simply cannot Stop.
var end_stream: Callable

const TAKE_DIR := "user://generative"
# Persisted through [Settings], which owns the one file the whole app shares - one save
# for ghost, not one per mode.
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
# sentence by 3.46 s, which is unmistakable. The base table stays conservative (it is
# calibrated to what the model already does) and this opens the range instead, so the
# choice is the reader's rather than baked in.
#
# The number is no longer a stretch factor for our own silence: the dial scales the WHOLE
# rest along a saturating curve, so 10 means "3.2 times the natural rest" rather than "ten
# times the silence we splice". See [method _rest_for].
const MAX_PAUSE_SCALE := 10.0
const SENTENCE_GAP := 0.32
# What the MODEL rests at a sentence end on its own: the trailing silence of one rendered
# sentence plus the leading silence of the next, which with one sentence per chunk is what
# sits either side of a seam. Measured, 0.138-0.145 + 0.044. Mirrors piper.DWELL - see
# [method _rest_for] for why a rest has to know this in order to scale properly.
const SENTENCE_DWELL := 0.18
# The pause curve's exponent, which IS its reach: log(5)/log(10), so the multiplier is
# exactly 1.0 at Pause 1.0 and 5.0 at the top of the slider. Mirrors piper.PAUSE_GAIN -
# see [method _pause_multiplier] for why a power law and not a saturating one.
const PAUSE_GAIN := 0.69897
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

# VOICES, plural, and NAMED. One tab per reader, and the tabs are whoever the script says
# is speaking - a line of its own hands the text after it to that name:
#
#     <!-- speaker: Emily White -->     (what a chapter file carries)
#     [speaker: Emily White]            (the same thing, typeable)
#
# THE SCRIPT IS THE CAST LIST. There is no + and no ×: the tabs are derived from the cues
# ([method _refresh_cast]) every time the text changes or a document is read, so a name
# written into the prose is a tab the moment it exists. A name was chosen over a number
# because the author reads the prose, and "speaker 2" says nothing to a reader of it. Text
# before the first cue is [constant Manuscript.NARRATOR]'s.
#
# A tab is a WHOLE settings page, not just a checkpoint id: the voice, the reader within
# it, the tone, the pace, the pauses, the delivery dials and the room are all per-tab. Two
# speakers in one reading are rarely the same person recorded twice.
#
# A NAME THAT LEAVES THE SCRIPT KEEPS ITS VOICE ([member _stash]). Deleting a scene to
# rewrite it must not cost the character their settings, so a name no longer cued is
# hidden, not forgotten, and comes back exactly as it was.
#
# The cue must own its line (see [constant Manuscript.SPEAKER]): a mis-parsed cue does not
# error, it just reads the rest of the chapter in the wrong voice.
#
# HESITATIONS. `<!-- hesitation -->` anywhere in the text - mid-sentence included - is a
# longer rest at exactly that point, for effect; `<!-- hesitation: 2.5 -->` names its own
# length in seconds. The Hesitate row sets the length of a bare one and switches them all.
# The rest is spliced into the take at the word boundary the aligner reports, so it lands
# between two words and never inside one (see [method _splice_holds]).
const HESITATE_DEFAULT := 1.5
const HESITATE_MAX := 6.0
# THE HANDOVER. A change of speaker is a bigger boundary than a sentence end and
# wants a bigger rest: one reader stops, the other starts, and run together they
# read as one person changing their mind mid-paragraph rather than as two people.
# This is the figure at Turn 1, ON TOP of the ordinary sentence seam, and the
# slider scales it.
#
# GLOBAL, not per tab, and that is the point of it being above them: the pause
# belongs to the BOUNDARY between two voices, not to either one of them, so
# asking which tab owns it has no answer.
const TURN_GAP := 0.55
const MAX_TURN_SCALE := 6.0
# ...and a ceiling on the whole rest, the seam included. Past a few seconds a
# handover stops reading as a beat and starts reading as the file having ended.
const TURN_CEILING := 4.0
## One tab's worth of settings. Also the schema: [method _cfg] merges a stored
## slot onto this, so a slot saved by an older build is missing keys rather than
## broken, and a key added later arrives with a sane value everywhere at once.
const SLOT_DEFAULTS := {
	"voice": "", "speaker": 0, "tone": 0, "pace": 1.0, "pause": 1.0,
	"dynamics": 0.5, "arc": 0.4, "effort": 0.35,
	"echo": 0.0, "room": 0.0, "resonance": 0.0, "presence": 1.0, "ambience": 0.0,
}
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
# A TONE OWNS THE VOICE AND NOTHING ELSE. It reached into two of the panel's own
# dials once - the ambience bed and Presence - and both were mistakes, in opposite
# directions.
#
# The bed was never the voice's business: it is a drone under the reading, a choice
# about the room, and it neither competes with the voice nor belongs to any manner of
# speaking. It was also pointless at the sizes used - "if we EVER set ambience to 0.1,
# then there is essentially no ambience at all."
#
# Presence WAS the voice's business, because Gruff really is muffled, but a preset has
# no business writing a control the reader has set - and as a hidden offset it was
# worse than that: Presence rests at the TOP of its travel, so subtracting from it
# could not be undone by pushing the slider up. "There is no way to correct that; I
# can ramp presence up to 1.0 and it's still too quiet." What Gruff wanted was not
# distance anyway. It wanted to sound like a voice coming through something, which is
# a property of the SOURCE, and `muffle` is now exactly that - a filter the preset
# owns, in the backend beside the whisper, leaving every dial on the panel alone.
#
# ...and `whisper` and `muffle`, which are neither. Those two are not parameters of
# the model at all but transforms applied to what it returns (piper.py `_whisper`,
# `_muffle`): the two manners a modal-speech checkpoint categorically cannot be asked
# for are the one where the vocal folds are not vibrating and the one where the voice
# is coming through something.
#
# The pitch shift is done by RESAMPLING, and the model compensates: to raise the
# voice by r we ask it to speak r times SLOWER, then play back r times faster.
# The two cancel in duration and leave only the pitch change, which avoids a
# phase vocoder entirely and is artifact-free at these depths. It does shift the
# formants with the pitch, so the speaker reads as a different SIZE - which is
# exactly what "spooky" (larger, lower) and "excited" (smaller, higher) want.
const TONE_PRESETS := {
	"Neutral":  {"pace": 1.00, "semis":  0.0, "noise": 0.667, "noise_w": 0.333, "muffle": 0.0, "whisper": 0.0},
	"Warm":     {"pace": 0.96, "semis": -0.5, "noise": 0.60,  "noise_w": 0.35, "muffle": 0.0, "whisper": 0.0},
	"Serious":  {"pace": 0.92, "semis": -1.5, "noise": 0.50,  "noise_w": 0.25, "muffle": 0.0, "whisper": 0.0},
	"Excited":  {"pace": 1.15, "semis":  2.0, "noise": 0.85,  "noise_w": 0.50, "muffle": 0.0, "whisper": 0.0},
	"Spooky":   {"pace": 0.85, "semis": -3.0, "noise": 0.45,  "noise_w": 0.20, "muffle": 0.0, "whisper": 0.0},
	# The three below fill quadrants the first five leave empty. `pace` and
	# `noise_w` are close to independent - one is how fast the reading runs, the
	# other how EVENLY it is divided - and everything above sits on the diagonal:
	# slow readings are also metronomic (Serious, Spooky), quick ones also loose
	# (Excited). The off-diagonal corners are where the manners that are not just
	# "more" or "less" of the same delivery live.
	#
	# SARCASTIC is the drawl: slow, but unevenly slow. It is not guesswork - the
	# acoustics of sarcasm have been measured (Cheang & Pell, "The sound of
	# sarcasm", Speech Communication 50, 2008), and against neutral productions of
	# the same sentences sarcasm came out lower in mean F0 (their most robust cue,
	# ~5-7% below neutral, so about a semitone), reduced in F0 standard deviation
	# (a flatter contour), reduced in HNR (a rougher voice), and slower - 9% on
	# whole sentences, 28% on short keyphrases. So: a semitone down, a sixth slower,
	# `noise` well under Neutral to flatten the melody, and `noise_w` the highest in
	# the bank, which is what stretches some syllables and clips others. The flat
	# melody over uneven timing is the whole effect; the lowered pitch alone reads
	# as Serious.
	#
	# URGENT is the corner nothing occupied: FAST AND TIGHT. Excited is fast and
	# loose - a voice that has lost its grip on the rhythm - and the opposite of
	# that is a voice keeping a grip on it deliberately. Low `noise_w` is what
	# makes it clipped rather than merely quick, and the pitch barely moves,
	# because the tension is in the timing, not the register.
	#
	# DREAMY is slow and loose, like Sarcastic, and reads nothing like it: it is
	# a semitone and a half UP (the formants go with the pitch, so the reader is
	# smaller and lighter, not larger and darker like Spooky), the model is left
	# free to wander at the top of the `noise` range, and the ambience bed comes up
	# further than any other preset - the pad is doing as much of the work as the
	# voice is.
	"Sarcastic": {"pace": 0.86, "semis": -1.0, "noise": 0.42, "noise_w": 0.60, "muffle": 0.0, "whisper": 0.0},
	"Urgent":    {"pace": 1.22, "semis":  0.5, "noise": 0.40, "noise_w": 0.16, "muffle": 0.0, "whisper": 0.0},
	"Dreamy":    {"pace": 0.88, "semis":  1.5, "noise": 0.78, "noise_w": 0.52, "muffle": 0.0, "whisper": 0.0},
	# ...and these three are the classic vocal-emotion table, as far as it
	# translates. Murray & Arnott ("Toward the simulation of emotion in synthetic
	# speech", Speech Communication 16, 1993) reviewed the human literature FOR
	# synthesis and tabulated five emotions against neutral speech, in rate, pitch
	# average, pitch range, intensity, voice quality and inflection. Three of those
	# five survive the trip into this parameter space; see below for the two that
	# do not.
	#
	# MOURNFUL is their sadness: slightly slower, slightly lower, slightly
	# narrower pitch range, downward inflections. Narrow range is the one that
	# matters here and it is why `noise` is the lowest in the bank - this is the
	# flattest, most affectless reading available, and the flatness is doing the
	# work, not the pitch. It is a semitone down, no more: Serious already owns
	# -1.5, and past that the formants have moved far enough that it reads as a
	# different, larger reader rather than the same one grieving.
	#
	# FIERCE is their anger: quicker, higher, wider, louder, with ABRUPT pitch
	# changes on stressed syllables and a rough chest tone. Only half of that is
	# available. The pitch half is not - raising F0 here resamples, which shrinks
	# the speaker, and an angry voice that has gone SMALL reads as a complaint
	# rather than a threat - so it sits a little BELOW neutral to keep the chest in
	# it, and the two halves that do translate carry the whole thing: `noise` at
	# the top of the bank for the roughness, `noise_w` near the bottom for the
	# abruptness. Fast, rough and clipped, where Excited is fast, bright and loose.
	#
	# ANXIOUS is their fear: much quicker, much higher, and IRREGULAR VOICING -
	# which is the one emotion in the table whose signature cue is jitter, so it is
	# the one this parameter space renders most directly. Highest `noise_w` in the
	# bank. It sits close to Excited by design, because they sit close in real
	# speakers too: high-arousal emotions are the ones listeners confuse with each
	# other, and the difference is that Excited is EVENLY quick and this is not.
	#
	# The two that did not translate: HAPPINESS is Excited already, and DISGUST
	# (very much slower, very much lower, grumbled) is Spooky with worse manners.
	"Mournful":  {"pace": 0.84, "semis": -1.0, "noise": 0.30, "noise_w": 0.45, "muffle": 0.0, "whisper": 0.0},
	"Fierce":    {"pace": 1.10, "semis": -0.5, "noise": 0.90, "noise_w": 0.28, "muffle": 0.0, "whisper": 0.0},
	"Anxious":   {"pace": 1.18, "semis":  1.5, "noise": 0.80, "noise_w": 0.65, "muffle": 0.0, "whisper": 0.0},
	# GRUFF is the growl behind the mask, and it is the first preset that needed
	# `pres`. Low and rough are in reach without it - the deepest `semis` in the
	# bank puts a bigger chest behind the voice, and `noise` near the top makes the
	# source gravelly rather than clean - but MUFFLED and QUIETER are not, and they
	# are half of what this voice is - and they are a property of the SOURCE, not of
	# how far away it is standing. `muffle` is the preset's own filter (piper._muffle),
	# so Gruff sounds like it is speaking through something without spending the
	# reader's Presence dial, which stays theirs for the room.
	#
	# The timing is deliberate rather than drawled - this voice is forcing the
	# words out, not savouring them - so `noise_w` sits low, near Fierce.
	"Gruff":     {"pace": 0.90, "semis": -4.0, "noise": 0.88, "noise_w": 0.30, "muffle": 0.55, "whisper": 0.0},
	# WHISPERED is the one manner in this bank that no inference parameter can
	# reach, and the only one that is not a setting at all. A VITS checkpoint
	# trained on modal speech has no whispered speech in it to sample, so there is
	# nothing to ask for: turning the model's own variation up gives a rough voice,
	# never a breathed one. What makes a whisper is the vocal folds not vibrating -
	# no fundamental, no harmonics, the words carried entirely by the resonances -
	# and that is a filter operation on the rendered audio, not a request to the
	# model. piper.py `_whisper` does it: each frame rebuilt as noise shaped by its
	# own spectral envelope, so the vocal tract survives and the voice in it does
	# not.
	#
	# HUSHED is the same transform at half strength, which is a real manner rather
	# than a fader position - a stage whisper is a voice that has not entirely
	# left, and half is where it sits. It used to stand back a little as well, on
	# the theory that someone lowering their voice does; that came out as "too
	# quiet, and there is no way to correct that", and the distance was never the
	# reason - a half blend of two UNCORRELATED signals loses 3 dB all by itself
	# (piper._whisper, which now normalises for it). The dial is back at 1.0 and
	# the reader can push it away if they want the distance.
	"Whispered": {"pace": 0.92, "semis":  0.0, "noise": 0.55, "noise_w": 0.35, "muffle": 0.0, "whisper": 1.0},
	"Hushed":    {"pace": 0.94, "semis": -0.5, "noise": 0.55, "noise_w": 0.33, "muffle": 0.0, "whisper": 0.45},
}

var _host: VoiceHost
var _panel: PanelContainer
var _doc: DocSource
var _text: TextEdit
var _voices: OptionButton
var _go: Button
var _stop: Button
var _test: Button
var _test_player: AudioStreamPlayer
var _test_req := {}            # host request id -> chunk index, for the audition
var _test_parts := {}          # chunk index -> wav path, as they arrive
var _test_chunks: Array = []
var _test_name := ""           # the voice being auditioned, fixed when it was pressed
var _test_next := 0            # the next sentence to join onto the audition's stream
var _test_hold := PackedFloat32Array()   # joined audio not yet pushed (before playback starts)
var _test_pushed := 0
var _test_read := 0            # cursor into _test_hold: what has been pushed
var _test_tasks := {}          # chunk index -> {box, task}: decoding on a worker
var _test_dry := PackedFloat32Array()   # joined audio waiting for the room (see _pump_test_fx)
var _test_fx_task := {}        # the one piece going through the room right now: {box, task}
var _test_cap := 0             # the generator's ring, measured empty
var _test_fx: VoiceFX = null
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
var _turn: HSlider
var _hesitate: HSlider
var _hesitate_on: CheckBox
var _tabs: VBoxContainer      # one row per speaker the script names
var _cast_head: Label
var _tab_group: ButtonGroup
var _names := PackedStringArray()  # the tabs, in order of first appearance in the script
var _slots: Array = []         # [SLOT_DEFAULTS-shaped Dictionary], one per tab, beside _names
var _stash := {}               # name -> settings, for voices the script no longer cues
var _slot := 0                 # which tab the controls are currently showing
var _cast_timer: Timer         # re-derives the tabs once typing pauses
# The lengths of the hesitations in the passage being cut, in text order; [method _cut]
# consumes them as the phonemizer hands back the sentinels that mark their places.
var _holds: Array = []
var _hold_i := 0
var _syncing := false          # writing controls from a slot must not re-plan
# WHEN THE ROOM CHANGES, in frames of the one continuous stream. The effects
# chain is stateful across chunk boundaries - that is the whole reason it runs
# here rather than in the host - so a second speaker cannot get their own chain
# without cutting the first one's tail off mid-decay. What they get instead is
# the same chain re-dialled at the exact frame their first sample is heard,
# which is what a live slider move already does, just scheduled.
var _fx_marks: Array = []      # [{at: int, speaker: String}], ascending, absolute
var _fx_live_name := ""        # which voice the live chain is currently dialled to
var _fx_queued_name := ""      # ...and the last one a mark was written for
# WHAT THE LAST PLAN NOTICED about the script - a speaker cue with no tab, a
# macro with no default. Not written straight to the status line, for two
# reasons that are both bugs it had: the line is overwritten by "Planned N
# chunk(s)…" a moment later, and a warning written only when there is something
# to warn about STAYS on screen after the text has been fixed. Rebuilt from
# scratch by every _build_chunks, and read by whoever reports the plan.
var _plan_note := ""
var _stream_open := false      # explicit: a null playback must not retry forever
var _epoch := 0                # bumped on a pace change; stale replies are dropped
# THE PREDICTED TIMELINE. The scrub bar used to be scaled by how much audio had been
# DECODED, which grows as the reading is synthesized - so the bar's own length changed
# under the pointer and dragging to "near the end" meant near the end of the first thirty
# seconds. A timeline has to know how long the thing is before it plays it.
var _repace_timer: Timer
var _vehicle_pick: OptionButton
## Rows that belong to a vehicle feature: tag -> the Controls to show or hide together.
## Filled by [method _director_slider] and [method _build_films]; read by [method _sync_vehicle_rows].
var _vehicle_rows := {}
var _filter_summary: Label
var _filter_rows := {}     # filter key -> {box: CheckBox, slider: HSlider}
var _film_list: VBoxContainer
var _film_freq: HSlider
var _film_status: Label
var _film_cutting := -1     # windows being cut last frame, so the status line only changes on change
var _film_dialog: FileDialog = null
var _illustrations: IllustrationPanel   # the book's pictures - see illustration_panel.gd
var _scene_hold: HSlider
var _flourish: HSlider
var _camera: HSlider
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
	_tick_test()
	# BEFORE the playback guard: a window cut is not part of a reading, and one started
	# with nothing playing would otherwise never be noticed to have finished.
	_pump_films()
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
	var avail := _fx_admit(_pending.size() - _read)
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


## THE ROOM CHANGES AT A FRAME, not at a chunk. Dial in every scheduled change the
## push has reached, and report how many frames may go out before the next one.
##
## The coordinate is ABSOLUTE FRAMES PUSHED, which is the only clock these marks
## can use. It is not what is being heard - the ring runs seconds ahead - and it
## does not need to be: the chain transforms the samples at exactly these
## positions on their way out, so aligning to the push aligns to the audio. Cut
## short at the next mark rather than crossing it, or a whole buffer of one
## speaker is read in the other's room.
##
## Split out of [method _process] because it is the arithmetic here that can be
## wrong by a buffer, and a room that arrives a moment early is audible without
## being attributable.
func _fx_admit(avail: int) -> int:
	while not _fx_marks.is_empty() and int((_fx_marks[0] as Dictionary)["at"]) <= _pushed:
		var m: Dictionary = _fx_marks.pop_front()
		_fx_live_name = String(m["speaker"])
		_apply_fx(_fx, _cfg_of(_fx_live_name))
	if _fx_marks.is_empty():
		return avail
	return mini(avail, maxi(0, int((_fx_marks[0] as Dictionary)["at"]) - _pushed))


func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.keycode == KEY_F2:
		_panel.visible = not _panel.visible


# --- panel (mirrors SynthEditor._build_panel) --------------------------------


func _build_panel() -> void:
	# A [SidePanel] rather than a bare PanelContainer: this panel has outgrown the window,
	# and a Control outside a container is never asked to fit anything - the rows past the
	# bottom edge were unreachable rather than clipped. See side_panel.gd.
	_panel = preload("res://scripts/side_panel.gd").new(380.0)
	add_child(_panel)
	var box: VBoxContainer = _panel.body
	box.add_theme_constant_override("separation", 8)

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
	hint.text = "Paste a chapter. It is spoken in chunks, so the show starts while the rest is still being made. Inline phonetics still work: [K AE T]. A line reading <!-- speaker: Emily --> hands the rest to Emily's tab; <!-- hesitation --> is a longer rest for effect."
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.add_theme_font_size_override("font_size", 12)
	hint.modulate = Color(1, 1, 1, 0.6)
	box.add_child(hint)

	# WHERE THE WORDS COME FROM, above the box because it decides what the box IS: a draft
	# to type in, or a live view of a file on disk that is re-read at every Speak.
	_doc = preload("res://scripts/doc_source.gd").new()
	_doc.setup("generative", "generative")
	_doc.capture = _doc_capture
	_doc.apply = _doc_apply
	box.add_child(_doc)

	_text = TextEdit.new()
	_text.custom_minimum_size = Vector2(360, 180)
	_text.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	_text.placeholder_text = "Once upon a time..."
	_text.tooltip_text = "The script to read. Paste a whole chapter - it is cut into sentences and only a couple are ever synthesized ahead, so the first words play within seconds however long it is. Square brackets pin a pronunciation: [B IY1 UW0 K S]. A line of its own reading <!-- speaker: Emily --> (or [speaker: Emily]) reads everything after it with Emily's tab, and the tabs follow the names the script uses. <!-- hesitation --> anywhere is a longer rest there (<!-- hesitation: 2.5 --> for exactly 2.5 seconds). Any other HTML comment is stripped rather than spoken. A template macro reads its default and never its own text: ${CHAPTERS_BEFORE_IN_WORDS:twenty-one} is read as \"twenty-one\"."
	_text.text_changed.connect(func() -> void:
		# The cast follows the text whoever changed it - a document arriving is exactly
		# when a whole new set of names appears.
		_cast_timer.start()
		if _doc.is_quiet():
			return          # the document being shown, not the author typing
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec()
		if not _chunks.is_empty():
			_go.text = "Speak ●"     # the reading no longer matches the box
		)
	box.add_child(_text)

	# A rule between the script and the cast. Everything from here to the tab bar
	# is global; everything below the tab bar belongs to the tab that is showing.
	box.add_child(HSeparator.new())

	# THE HANDOVER REST, above the tabs because it belongs to no tab: it is the
	# silence BETWEEN two of them. Unlike Pause it needs no re-synthesis - the
	# gap is spliced in as the chunks are joined, not asked of the model - so it
	# takes effect at the next handover rather than at the next chunk.
	var turow := HBoxContainer.new()
	turow.add_theme_constant_override("separation", 8)
	box.add_child(turow)
	var tul := Label.new()
	tul.text = "Turn"
	tul.custom_minimum_size = Vector2(72, 0)
	tul.add_theme_font_size_override("font_size", 12)
	turow.add_child(tul)
	_turn = HSlider.new()
	_turn.min_value = 0.0
	_turn.max_value = MAX_TURN_SCALE
	_turn.step = 0.05
	_turn.value = 1.0
	_turn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_turn.tooltip_text = ("How long the reading rests when the script CHANGES SPEAKER, on top of "
		+ "the ordinary rest between sentences. 1 is a little over half a second - enough to "
		+ "hear one reader stop and another begin; 0 hands over on the same beat as any other "
		+ "sentence, which reads as one person changing their mind rather than as two people. "
		+ "The whole rest is capped at four seconds however far this is pushed. It applies to "
		+ "every tab, because the pause belongs to the boundary and not to either voice, and it "
		+ "takes effect at the next handover - nothing has to be generated again.")
	_turn.value_changed.connect(func(_v: float) -> void:
		if _syncing:
			return
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec())
	turow.add_child(_turn)

	# HESITATIONS, global for the same reason Turn is: a `<!-- hesitation -->` belongs to a
	# moment in the text, not to whichever voice happens to be reading it. The checkbox
	# switches every marker off at once (an author auditioning the plain reading); the dial
	# is the length of a bare marker, and a marker that names its own length keeps it.
	var hrow := HBoxContainer.new()
	hrow.add_theme_constant_override("separation", 8)
	box.add_child(hrow)
	_hesitate_on = CheckBox.new()
	_hesitate_on.text = "Hesitate"
	_hesitate_on.button_pressed = true
	_hesitate_on.custom_minimum_size = Vector2(72, 0)
	_hesitate_on.add_theme_font_size_override("font_size", 12)
	_hesitate_on.tooltip_text = ("Honour the script's <!-- hesitation --> markers: a longer rest "
		+ "at exactly that point, mid-sentence or between paragraphs, for dramatic effect. Off "
		+ "reads straight through them as if they were not there.")
	hrow.add_child(_hesitate_on)
	_hesitate = HSlider.new()
	_hesitate.min_value = 0.2
	_hesitate.max_value = HESITATE_MAX
	_hesitate.step = 0.05
	_hesitate.value = HESITATE_DEFAULT
	_hesitate.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_hesitate.tooltip_text = ("How long a bare <!-- hesitation --> rests, in seconds, on top of "
		+ "whatever pause the text already has there. A marker that names its own length - "
		+ "<!-- hesitation: 2.5 --> - keeps it whatever this says. Takes effect from the next "
		+ "chunk; nothing already made is regenerated, because the rest is spliced in rather "
		+ "than asked of the voice.")
	hrow.add_child(_hesitate)
	_slider_readout(hrow, _hesitate, "s")
	var on_hesitate := func(_v: Variant) -> void:
		if _syncing:
			return
		_hesitate.editable = _hesitate_on.button_pressed
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec()
	_hesitate.value_changed.connect(on_hesitate)
	_hesitate_on.toggled.connect(on_hesitate)

	# THE TABS - one per name the script cues, derived rather than managed (see the note on
	# the speaker cues above). Everything below this row belongs to the selected tab; the
	# Speak button beside the voice picker does not - it reads the whole script, in every
	# voice it asks for. A FLOW, not a TabBar, so a chapter with eight speakers shows all
	# eight names at once instead of hiding half of them behind scroll arrows.
	# A LIST WITH A HEADING, one row per person, each saying what voice they have now - a
	# first cut drew the names as a flow of flat buttons and it read as a line of text: no
	# heading, no visible selection, nothing saying these were the speakers or what they
	# sounded like.
	_cast_head = Label.new()
	_cast_head.add_theme_font_size_override("font_size", 12)
	_cast_head.mouse_filter = Control.MOUSE_FILTER_STOP
	_cast_head.tooltip_text = ("The voices are the speakers the script names - a line of its "
		+ "own reading <!-- speaker: Emily --> hands the text after it to Emily - and the "
		+ "list follows the script as it changes. Pick a person to edit their voice below; "
		+ "Test hears it on its own. A name that drops out of the script keeps its settings "
		+ "for when it comes back. Speak still reads the whole script.")
	box.add_child(_cast_head)
	_tabs = VBoxContainer.new()
	_tabs.add_theme_constant_override("separation", 2)
	_tabs.tooltip_text = _cast_head.tooltip_text
	box.add_child(_tabs)
	_tab_group = ButtonGroup.new()
	_cast_timer = Timer.new()
	_cast_timer.one_shot = true
	_cast_timer.wait_time = 0.4
	_cast_timer.timeout.connect(func() -> void: _refresh_cast(_text.text))
	add_child(_cast_timer)

	var vrow := HBoxContainer.new()
	vrow.add_theme_constant_override("separation", 8)
	box.add_child(vrow)
	_voices = OptionButton.new()
	_voices.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_voices.tooltip_text = "Which Piper model reads. Each is a different person with its own accent, recording and licence - the licence appears under the panel when you pick one. Changing this regenerates whatever has not been played yet."
	_voices.item_selected.connect(func(_i: int) -> void:
		if _syncing:
			return
		_show_voice_license()
		_capture_slot()
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
	_go.tooltip_text = "Read the script aloud and drive the visuals from it. Pressing it again restarts the reading from the top; Stop ends it. The scenes react to the narration exactly as they would to music."
	_go.disabled = true
	_go.pressed.connect(_on_speak)
	vrow.add_child(_go)
	# STOP, beside Speak. Ending the reading needs its own control: Speak restarts from the
	# top, which is the opposite of what someone about to export wants.
	_stop = Button.new()
	_stop.text = "Stop"
	_stop.tooltip_text = "End the reading and hand the stage back. Use this before an export - a live reading is still driving the visuals and holding the audio stream, and until now the only way to end one was to restart ghost."
	_stop.disabled = true
	_stop.pressed.connect(_stop_speaking)
	vrow.add_child(_stop)
	# AUDITION ONE VOICE, without the chapter. In a script with seven speakers the last one
	# first speaks forty minutes in, so tuning that voice by reading the whole script is not
	# a loop anyone can iterate in. This reads a fixed passage (TEST_PASSAGE) in the voice on
	# the selected tab - its tone, pace, pauses, delivery and room - and plays it straight
	# out: no stage, no subtitles, no Director.
	_test = Button.new()
	_test.text = "Test"
	_test.tooltip_text = ("Hear the selected tab's voice on its own: a short fixed passage, "
		+ "read with every setting on this tab and played without any visuals. The same "
		+ "words every time, so two voices - or two settings of one - can be compared by "
		+ "ear. Press again to stop. Starting a test ends a reading in progress.")
	_test.disabled = true
	_test.pressed.connect(_on_test)
	vrow.add_child(_test)

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
		if _syncing:
			return
		_capture_slot()
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
	_tone.tooltip_text = "The reading's overall manner: pace, pitch, how much the model varies itself, how evenly it divides the words, and whether it is whispered or muffled - all of it the VOICE, and none of it the dials below, which stay yours. The pitch shift moves the formants with it, so the reader reads as a different SIZE - which is what makes Spooky larger and lower, Excited smaller and higher. Regenerates the un-played chunks."
	for k in TONE_PRESETS:
		_tone.add_item(String(k))
	_tone.item_selected.connect(func(_i: int) -> void:
		if _syncing:
			return
		_capture_slot()
		# a tone changes the model's own parameters, so un-played chunks have
		# to be regenerated - the same path a pace change takes
		if not _chunks.is_empty():
			_repace()
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
	_rate.value_changed.connect(func(_v: float) -> void:
		if _syncing:
			return
		_repace_timer.start())
	_rate_row.add_child(_rate)
	_slider_readout(_rate_row, _rate, "x")

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
	_pause.tooltip_text = "How long to rest on punctuation - commas, semicolons, colons and sentence ends. It scales the WHOLE rest, the model's own included, so every mark keeps its share of the reading at every setting rather than the long ones running away from the short ones. 1 is the natural rest; 0 runs straight through; 10 is five times natural, a two-and-a-half second full stop, and deliberately too much for most material. A quicker Tone rests proportionally less at the same setting - that is what quicker means - so a fast reading wants a higher number here than a slow one. Takes effect from the next chunk; already-generated audio keeps its pauses."
	_pause.value_changed.connect(func(_v: float) -> void:
		if _syncing:
			return
		_repace_timer.start())
	prow.add_child(_pause)
	_slider_readout(prow, _pause, "x")

	# Debounced rather than immediate: a slider drag emits a value per pixel,
	# and each change throws away work in flight. One re-plan per gesture.
	_repace_timer = Timer.new()
	_repace_timer.wait_time = 0.3
	_repace_timer.one_shot = true
	_repace_timer.timeout.connect(func() -> void:
		# CAPTURE FIRST. _repace re-requests from the slot store, not from the
		# controls, so a pace the slider knows about and the slot does not is a
		# pace the regenerated chunks are read at the OLD value.
		_capture_slot()
		_repace()
		_persist())
	add_child(_repace_timer)

	# Ambience: the same effects Synthesis has, over any PCM. Off by default -
	# a narration take should sound like a reading unless asked otherwise.
	# DELIVERY, not effect. These two are the discourse layer Piper structurally
	# cannot supply: it never sees the paragraph, so every sentence starts from the
	# same register at the same rate. See piper._discourse_plan for the rules.
	_dynamics = _fx_slider(box, "Dynamics", 0.5,
		"How much the READING's timing follows its own structure: a sentence at the end of a "
		+ "paragraph is drawn out, a short one after a long one lands harder, a question runs a "
		+ "little quicker. 0 reads every sentence at the same rate. At the top a closing sentence "
		+ "runs about a third longer than an opening one, which is deliberately more than any "
		+ "reader would do - if a chapter seems to be getting slower and slower as it goes, this "
		+ "is the dial, not Arc.")
	_effort = _fx_slider(box, "Effort", 0.35,
		"Vocal effort across a paragraph - which is not the same as volume. A louder voice has a "
		+ "brighter source spectrum, not just a higher level, so this moves the two together: "
		+ "opening a paragraph with a little more push and easing off toward the end. Level "
		+ "alone reads as the reader standing further away; the brightness is what makes it read "
		+ "as them easing off instead. 0 is off, 1 is about 6dB and a firm tilt top to bottom.")
	_arc = _fx_slider(box, "Arc", 0.4,
		"Pitch shape ACROSS a paragraph, in semitones. A speaker opens a new paragraph in a higher "
		+ "register and settles as it goes, then resets on the next - it is what makes a paragraph "
		+ "land rather than just stop. 0 is off, 0.15 is half a semitone top to bottom, 0.4 is "
		+ "one and a half, and 1 is four. The whole travel is usable now: the shift holds the "
		+ "reader's own formants, so the register moves without the speaker changing SIZE, and "
		+ "the ceiling is set by what a speaker would actually do rather than by the DSP.")
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
	_vehicle_pick = _vehicle_option(box)
	_build_films(box)
	# THE BOOK'S PICTURES, under the vehicle picker for the reason the films are: they only
	# mean something to the vehicle that prints them, and they appear the moment it is picked.
	_illustrations = preload("res://scripts/illustration_panel.gd").new()
	box.add_child(_illustrations)
	(_vehicle_rows.get_or_add("illustrations", []) as Array).append(_illustrations)
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
	_camera = _director_slider(box, "Camera", Director.CAMERA_MIN, Director.CAMERA_MAX, 0.05,
		Director.camera,
		"How severe the camera is on the Comic book vehicle - one knob over the whole "
		+ "behaviour. 0 is a slow gentle drift that barely turns and never cuts; 1 is the "
		+ "default; 2 is fast, restless and cinematic, with real jump cuts. It scales how far "
		+ "a shot may swing, how many shots that swing is spread over, how long a move lasts "
		+ "and how deep a push goes. It is shown only for the vehicles that fly a camera.",
		func(v: float) -> void: Director.set_camera(v), "camera")
	_build_filters(box)
	# Now that every tagged row exists, show the ones this vehicle can actually use.
	_sync_vehicle_rows()
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
	# The controls are the truth for the tab on screen; every other tab's truth
	# is already in _slots. Capture before writing or the tab being edited saves
	# whatever it held when it was last switched away from.
	_capture_slot()
	Settings.write("generative", "text", _doc.draft())
	Settings.write("generative", "cast", _cast_dict())
	Settings.write("generative", "turn", _turn.value)
	# A NEW KEY, not "tab": that one held an int index, and Settings' no-op guard compares
	# the stored value with the new one - an int against a String is a script error.
	Settings.write("generative", "tab_name", _tab_name())
	Settings.write("generative", "hesitate", _hesitate.value)
	Settings.write("generative", "hesitate_on", _hesitate_on.button_pressed)


func _load_persisted() -> void:
	_syncing = true
	_turn.value = clampf(float(Settings.read("generative", "turn", 1.0)), 0.0, MAX_TURN_SCALE)
	_hesitate.value = clampf(float(Settings.read("generative", "hesitate", HESITATE_DEFAULT)),
		_hesitate.min_value, HESITATE_MAX)
	_hesitate_on.button_pressed = bool(Settings.read("generative", "hesitate_on", true))
	_hesitate.editable = _hesitate_on.button_pressed
	_syncing = false
	var cast: Variant = Settings.read("generative", "cast", {})
	if cast is Dictionary and not (cast as Dictionary).is_empty():
		_stash = _merge_cast(cast as Dictionary)
	else:
		# MIGRATION, from the numbered tabs. Tab N was cued `<!-- speaker: N -->`, and tab 1
		# also read everything before the first cue - so it becomes both "1" and the narrator.
		var rows: Variant = Settings.read("generative", "slots", [])
		_stash = _cast_from_rows(rows if rows is Array else [],
			str(Settings.read("generative", "text", "")))
		if _stash.is_empty():
			# ...and from the single-voice file before that, read exactly once.
			_stash[Manuscript.NARRATOR] = _merge({
				"voice": str(Settings.read("generative", "voice", "")),
				"speaker": int(Settings.read("generative", "speaker", 0)),
				"tone": int(Settings.read("generative", "tone", 0)),
				"pace": float(Settings.read("generative", "pace", 1.0)),
				"pause": float(Settings.read("generative", "pause", 1.0)),
			})
	var tab: Variant = Settings.read("generative", "tab_name", "")
	_names = PackedStringArray()
	_slots = []
	# From the stored draft first, so the tabs exist before the box is bound below.
	_refresh_cast(str(Settings.read("generative", "text", "")), str(tab) if tab is String else "")
	# Last, and once: every dial the chain reads is now in the slot, so this is the one
	# place the loaded session's room is applied.
	_apply_fx(_fx, _cfg(_slot))
	# ...and NOW the box is filled, after the cast is loaded, because a document that is open
	# OVERRULES what was loaded above: in sync mode the file is the source of truth for both
	# the words and the voice, which is what makes it safe for the panel to write back to it
	# on its own. See DocSource.bind_text. The text arriving re-derives the tabs.
	_doc.bind_text(_text)
	_refresh_cast(_text.text)


# --- the document's own voice -------------------------------------------------


## THE WHOLE CAST, for [DocSource] to store in a document's frontmatter.
##
## Every voice, keyed by the name the script cues it with - including names the text no
## longer uses, so rewriting a scene and restoring it does not lose a character's voice
## from the document either. The controls are captured first, because they - not
## [member _slots] - are the truth for the tab on screen.
func _doc_capture() -> Dictionary:
	_capture_slot()
	return {"turn": _turn.value, "tab": _tab_name(), "voices": _cast_dict(),
		"hesitate": _hesitate.value, "hesitate_on": _hesitate_on.button_pressed,
		"illustrations": Illustrations.look()}


## ...and back the other way, when a document that carries one is opened.
##
## A BLOCK WITH NO CAST CHANGES NOTHING. `_merge` gives every row every key, so a voice
## written by an older build arrives with sane defaults rather than half-applied - but a
## block with no `voices` at all is a document that has never been given a voice, and
## silently blanking the panel's cast for it would be the opposite of the feature. A LIST
## of voices is the numbered tabs this replaced, and is read the way the settings are.
func _doc_apply(cfg: Dictionary) -> void:
	# THE LOOK first, and on its own: a document may carry pictures' settings and no cast yet.
	if cfg.get("illustrations") is Dictionary:
		var errs := Illustrations.set_look(cfg["illustrations"] as Dictionary)
		if not errs.is_empty():
			_note("reference image(s) not found: " + "; ".join(errs))
		if _illustrations != null:
			_illustrations.sync_from_library()
	var rows: Variant = cfg.get("voices", {})
	var cast: Dictionary = _merge_cast(rows as Dictionary) if rows is Dictionary \
		else _cast_from_rows(rows if rows is Array else [], _text.text)
	if cast.is_empty():
		return
	_stash = cast
	_names = PackedStringArray()
	_slots = []
	_syncing = true
	_turn.value = clampf(float(cfg.get("turn", _turn.value)), 0.0, MAX_TURN_SCALE)
	_hesitate.value = clampf(float(cfg.get("hesitate", _hesitate.value)),
		_hesitate.min_value, HESITATE_MAX)
	_hesitate_on.button_pressed = bool(cfg.get("hesitate_on", _hesitate_on.button_pressed))
	_hesitate.editable = _hesitate_on.button_pressed
	_syncing = false
	var tab: Variant = cfg.get("tab", "")
	_refresh_cast(_text.text, str(tab) if tab is String else "")
	# The room, once, exactly as [method _load_persisted] does it: the chain is stateful
	# and a tab switch alone does not re-dial it.
	_apply_fx(_fx, _cfg(_slot))
	# Whatever is on the stream was cut against the PREVIOUS cast.
	_mark_stale()
	_dirty = true
	_last_edit_ms = Time.get_ticks_msec()


## What [method export_take] hands the book vehicle, and what main gives a live one: the
## chapter as the author wrote it, so pages can be typeset from the same words being read.
##
## The TITLE is carried separately because in sync mode the body arrives without its
## frontmatter, which is where a chapter's title lives.
func book_document(body := "") -> Dictionary:
	var src := body if not body.is_empty() else _doc.pull()
	return {"source": src, "title": _doc_title(src)}


## The chapter title: the open document's frontmatter `title:`, else the pasted text's own.
## Read TEXTUALLY, one line, for the reason FrontMatter.read_block gives - a head holding a
## construct MiniYaml refuses must still yield its title.
func _doc_title(src: String) -> String:
	var raw := src
	if _doc != null and _doc.is_sync() and FileAccess.file_exists(_doc.doc_path()):
		raw = FileAccess.get_file_as_string(_doc.doc_path())
	var fm := FrontMatter.split(raw)
	if not bool(fm["has"]):
		return ""
	for line in String(fm["head"]).split("\n"):
		var t := String(line).strip_edges()
		if t.begins_with("title:"):
			return t.substr(6).strip_edges().trim_prefix("\"").trim_suffix("\"") \
				.trim_prefix("'").trim_suffix("'")
	return ""


# --- voices, plural and named --------------------------------------------------


## Every voice the panel knows, shown or not: name -> settings.
func _cast_dict() -> Dictionary:
	var out := {}
	for k in _stash:
		out[k] = _merge(_stash[k] as Dictionary)
	for i in _names.size():
		out[_names[i]] = _cfg(i)
	return out


func _merge_cast(d: Dictionary) -> Dictionary:
	var out := {}
	for k in d:
		if d[k] is Dictionary and not str(k).strip_edges().is_empty():
			out[str(k).strip_edges()] = _merge(d[k] as Dictionary)
	return out


## The numbered tabs this replaced, as names: row i was cued `<!-- speaker: i+1 -->`, and
## the first row read everything before the first cue - so it becomes the NARRATOR, and also
## "1" only when the script actually cues `1` (otherwise every migrated document would carry
## a phantom voice named "1" from then on).
func _cast_from_rows(rows: Array, body := "") -> Dictionary:
	var cued := Manuscript.speakers(body)
	var out := {}
	for i in rows.size():
		if not (rows[i] is Dictionary):
			continue
		if i == 0:
			out[Manuscript.NARRATOR] = _merge(rows[i] as Dictionary)
			if cued.has("1"):
				out["1"] = _merge(rows[i] as Dictionary)
		else:
			out[str(i + 1)] = _merge(rows[i] as Dictionary)
	return out


## THE TABS FOLLOW THE SCRIPT. Re-derive them from [param body]: every name it cues, in order
## of first appearance, each with its own settings - kept if it had a tab, restored if it was
## stashed, and otherwise a COPY of the first voice (a new character is reached by changing
## one thing about the reader you already have, not by re-dialling a room from nothing).
## Names that left the script go to [member _stash]. The tab on screen stays on the same
## NAME when it survives, since its index means nothing once names come and go.
##
## Cheap and idempotent - it runs on every pause in typing - and it touches nothing when the
## cast is unchanged, so a reading in progress is only marked stale by a real change.
func _refresh_cast(body: String, want := "") -> void:
	# The picture list follows the same text, on the same beat.
	if _illustrations != null:
		_illustrations.set_script_text(body)
	if _rate != null:
		_capture_slot()
	var names := Manuscript.speakers(body)
	if names.is_empty():
		names = PackedStringArray([Manuscript.NARRATOR])
	var showing := want if not want.is_empty() else _tab_name()
	if names == _names and not _slots.is_empty():
		if not want.is_empty() and names.has(want) and names.find(want) != _slot:
			_select_tab(names.find(want))
		return
	var old := {}
	for i in _names.size():
		old[_names[i]] = _slots[i]
	for k in old:
		_stash[k] = old[k]
	var seed_row: Dictionary = SLOT_DEFAULTS.duplicate()
	if not _slots.is_empty():
		seed_row = _cfg(0)
	elif not _stash.is_empty():
		seed_row = _merge(_stash[_stash.keys()[0]] as Dictionary)
	var slots: Array = []
	for n in names:
		if _stash.has(n):
			slots.append(_merge(_stash[n] as Dictionary))
			_stash.erase(n)
		else:
			slots.append(seed_row.duplicate())
	var changed := _names.size() > 0 and names != _names
	_names = names
	_slots = slots
	_slot = maxi(0, _names.find(showing))
	_rebuild_tabs()
	_apply_slot(_slot)
	if changed:
		_mark_stale()
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec()


func _tab_name() -> String:
	return _names[_slot] if _slot >= 0 and _slot < _names.size() else ""


## A stored slot merged onto [constant SLOT_DEFAULTS]: every key present, every
## value the right type, and an index that cannot be out of range. Everything
## downstream reads settings through here, so a tab that vanished while its passages
## are still queued degrades to the first voice rather than to a null.
func _cfg(i: int) -> Dictionary:
	if _slots.is_empty():
		return SLOT_DEFAULTS.duplicate()
	return _merge(_slots[clampi(i, 0, _slots.size() - 1)] as Dictionary)


## The settings of the voice NAMED [param who] - which is how a chunk asks, because a tab's
## index moves whenever the script gains or loses a speaker, and a reading in flight must
## not change voice because someone typed a new name further down.
func _cfg_of(who: String) -> Dictionary:
	var i := _names.find(who)
	if i >= 0:
		return _cfg(i)
	if _stash.has(who):
		return _merge(_stash[who] as Dictionary)
	return _cfg(0)


func _merge(row: Dictionary) -> Dictionary:
	var out := SLOT_DEFAULTS.duplicate()
	for k in out:
		if not row.has(k):
			continue
		# The defaults double as the schema: an int key stays an int through a
		# ConfigFile round trip, which stores every number as a float.
		match typeof(out[k]):
			TYPE_INT: out[k] = int(row[k])
			TYPE_FLOAT: out[k] = float(row[k])
			_: out[k] = str(row[k])
	return out


## The controls -> the selected slot.
func _capture_slot() -> void:
	if _slot < 0 or _slot >= _slots.size() or _rate == null:
		return
	_slots[_slot] = {
		"voice": _selected_voice_id(), "speaker": int(_speaker.value),
		"tone": _tone.selected, "pace": _rate.value, "pause": _pause.value,
		"dynamics": _dynamics.value, "arc": _arc.value, "effort": _effort.value,
		"echo": _fx_echo.value, "room": _fx_room.value, "resonance": _fx_res.value,
		"presence": _fx_presence.value, "ambience": _fx_pad.value,
	}
	_refresh_tab_labels()


## A slot -> the controls. Every write is inside `_syncing`, because each of
## these controls answers a change by throwing away work in flight: without the
## guard, showing a tab would restart the reading and re-request every chunk at
## the settings of the tab that was just left.
func _apply_slot(i: int) -> void:
	var s := _cfg(i)
	_syncing = true
	_want_voice = String(s["voice"])
	for k in _voice_meta.size():
		if String((_voice_meta[k] as Dictionary).get("id", "")) == _want_voice:
			_voices.select(k)
			break
	_tone.select(clampi(int(s["tone"]), 0, TONE_PRESETS.size() - 1))
	_rate.value = float(s["pace"])
	_pause.value = float(s["pause"])
	_dynamics.value = float(s["dynamics"])
	_arc.value = float(s["arc"])
	_effort.value = float(s["effort"])
	_fx_echo.value = float(s["echo"])
	_fx_room.value = float(s["room"])
	_fx_res.value = float(s["resonance"])
	_fx_presence.value = float(s["presence"])
	_fx_pad.value = float(s["ambience"])
	# After the voice, because it is what sets the Speaker row's range - and a
	# speaker id is only meaningful against the model that holds it.
	_show_voice_license()
	_speaker.value = clampf(float(int(s["speaker"])), _speaker.min_value, _speaker.max_value)
	_syncing = false


func _rebuild_tabs() -> void:
	if _tabs == null:
		return
	for c in _tabs.get_children():
		_tabs.remove_child(c)
		c.queue_free()
	var on := StyleBoxFlat.new()
	on.bg_color = Color(0.30, 0.55, 0.75, 0.45)
	on.border_color = Color(0.55, 0.85, 1.0, 0.9)
	on.border_width_left = 3
	on.set_content_margin_all(4)
	var off := StyleBoxFlat.new()
	off.bg_color = Color(1, 1, 1, 0.04)
	off.set_content_margin_all(4)
	off.content_margin_left = 7
	var hover := off.duplicate() as StyleBoxFlat
	hover.bg_color = Color(1, 1, 1, 0.10)
	for i in _names.size():
		var b := Button.new()
		b.toggle_mode = true
		b.button_group = _tab_group
		b.focus_mode = Control.FOCUS_NONE
		b.alignment = HORIZONTAL_ALIGNMENT_LEFT
		b.clip_text = true
		b.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
		b.add_theme_font_size_override("font_size", 12)
		b.add_theme_stylebox_override("normal", off)
		b.add_theme_stylebox_override("hover", hover)
		b.add_theme_stylebox_override("pressed", on)
		b.add_theme_stylebox_override("hover_pressed", on)
		b.set_pressed_no_signal(i == _slot)
		var at := i
		b.pressed.connect(func() -> void: _on_tab_selected(at))
		_tabs.add_child(b)
	_refresh_tab_labels()


## Each row: the name, then the voice it has NOW - model, reader, tone - so the whole cast can
## be read at a glance, and two people accidentally left on the same voice are obvious.
func _refresh_tab_labels() -> void:
	if _tabs == null:
		return
	if _cast_head != null:
		_cast_head.text = "Voices (%d)  -  from the script's speaker cues" % _names.size()
	var tones := TONE_PRESETS.keys()
	for i in mini(_tabs.get_child_count(), _names.size()):
		var s := _cfg(i)
		var model := String(s["voice"])
		var multi := false
		for v in _voice_meta:
			if String((v as Dictionary).get("id", "")) == model:
				model = String((v as Dictionary).get("name", model))
				multi = int((v as Dictionary).get("speakers", 1)) > 1
				break
		if model.is_empty():
			model = "default voice"
		var desc := model + ((" #%d" % int(s["speaker"])) if multi else "")
		desc += "  ·  " + String(tones[clampi(int(s["tone"]), 0, tones.size() - 1)])
		var b := _tabs.get_child(i) as Button
		b.text = "%s    %s" % [_names[i], desc]
		b.tooltip_text = "%s - %s. Pick to edit this voice." % [_names[i], desc]


func _select_tab(i: int) -> void:
	_on_tab_selected(i)
	for c in _tabs.get_children():
		(c as Button).set_pressed_no_signal(c.get_index() == _slot)


func _on_tab_selected(i: int) -> void:
	if _syncing or i == _slot:
		return
	_capture_slot()
	_slot = clampi(i, 0, _slots.size() - 1)
	_apply_slot(_slot)


## The reading on the stream no longer matches the panel. The same dot the text
## box raises, and for the same reason: the cast changed under what is playing.
func _mark_stale() -> void:
	if not _chunks.is_empty() and _go != null:
		_go.text = "Speak ●"


## Split a script into passages by speaker: `[{speaker, text, lead}]`. Text before the
## first cue - and a script with no cues at all - is the narrator's.
##
## Also where the HESITATIONS are resolved, because it is the last place that still sees the
## markers in their text: each becomes a [constant TextNorm.HOLD_MARK] welded to the word it
## follows (or leads, when nothing comes before it in the passage), and its length goes on
## [member _holds] in the same order, for [method _cut] to hand back out. With the Hesitate
## box off they are simply removed. Every other comment is an authoring note and goes too.
func _split_speakers(body: String) -> Array:
	var kept: Array = []
	var on := _hesitate_on == null or _hesitate_on.button_pressed
	var bare := _hesitate.value if _hesitate != null else HESITATE_DEFAULT
	_holds = []
	for p in Manuscript.passages(body):
		var t := String((p as Dictionary)["text"])
		var out := ""
		var at := 0
		var re := Manuscript._rx(Manuscript.COMMENT)
		var hes := Manuscript._rx(Manuscript.HESITATION)
		for m in re.search_all(t):
			out += t.substr(at, m.get_start() - at)
			at = m.get_end()
			if not on:
				continue
			var hm := hes.search(m.get_string())
			if hm == null:
				continue             # an authoring note
			var v := hm.get_string(1) if not hm.get_string(1).is_empty() else hm.get_string(2)
			var secs := clampf(float(v), 0.0, 30.0) if not v.is_empty() else bare
			# WELDED, never left standing: a sentinel alone between two spaces is a token of
			# its own, and a token that is not a word confuses the sentence splitter around
			# it. Trailing whitespace is stepped over so it lands directly after the last
			# character of the previous word.
			var head := out.rstrip(" \t\n")
			out = head + TextNorm.HOLD_MARK + out.substr(head.length())
			_holds.append(secs)
		out += t.substr(at)
		# Sentinels with nothing before them in the passage LEAD the next word instead:
		# close the gap after them so they weld on to it.
		var body_t := out.strip_edges()
		var n := 0
		while body_t.begins_with(TextNorm.HOLD_MARK):
			n += 1
			body_t = body_t.substr(1).strip_edges(true, false)
		out = TextNorm.HOLD_MARK.repeat(n) + body_t
		if not out.replace(TextNorm.HOLD_MARK, "").strip_edges().is_empty():
			kept.append({"speaker": String((p as Dictionary)["speaker"]), "text": out})
		else:
			# A passage of nothing but hesitations: its rests would have no word to sit on.
			for _i in out.count(TextNorm.HOLD_MARK):
				_holds.pop_back()
	return kept


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


# --- the audition -------------------------------------------------------------

## The words every voice is auditioned with. The SAME passage for every speaker, so two
## voices are compared on identical material; chosen to exercise what a chapter will ask of
## them - a long sentence with commas, a colon, a question, an exclamation, a number, a line
## of dialogue in quotes and a short sentence landing after a long one. It OPENS on a short
## sentence on purpose: the first sound waits for the first sentence to be made.
const TEST_PASSAGE := """The rain had not stopped. Since morning it had kept on, and the lamps along the river were already lit by four o'clock. I counted them from the window: twelve on this side, eleven on the other, one of them out.

"Are you coming, or not?" she asked from the doorway. I told her I would be a minute. It was closer to twenty.

Some evenings are like that. You mean to leave, and the room keeps you, and by the time you notice it is dark. Still, I went. I am glad I did!"""


## Speak the passage in the selected tab's voice, or stop the one playing.
##
## STREAMED, like a reading: every sentence is requested at once (the host answers them in
## order), each is joined onto the stream the moment it and everything before it has
## arrived, and playback starts with the first sentence. The
## first cut waited for the whole passage and was reported as "a solid 15 seconds of
## silence" - longer than a chapter takes to start.
func _on_test() -> void:
	if _test_busy():
		_stop_test()
		return
	if _host == null or not _host.is_up() or _voice_meta.is_empty():
		return
	# One sound at a time: a test over a reading is two voices at once and neither is heard.
	if not _chunks.is_empty():
		_stop_speaking()
	_capture_slot()
	_test_name = _tab_name()
	_holds = []
	_hold_i = 0
	_test_chunks = _cut(TEST_PASSAGE, 0)["chunks"]
	_place_chunks(_test_chunks, TEST_PASSAGE)
	_test_parts = {}
	_test_req = {}
	_test_tasks = {}
	_test_dry = PackedFloat32Array()
	_test_fx_task = {}
	_test_next = 0
	_test_hold = PackedFloat32Array()
	_test_read = 0
	_test_pushed = 0
	_test_fx = null
	var s := _cfg_of(_test_name)
	for i in _test_chunks.size():
		var id := _host.request("", _voice_id_of(s), TAKE_DIR + "/test_%d.wav" % i,
			_request_args(s, _test_chunks[i]), null)
		_test_req[id] = i
	_test.text = "Test ■"
	_set_status("Testing %s…" % _test_name)


func _test_busy() -> bool:
	return not _test_req.is_empty() or (_test_player != null and _test_player.playing)


func _stop_test() -> void:
	# A worker still decoding is waited for rather than abandoned - it is a fraction of a second.
	for k in _test_tasks:
		WorkerThreadPool.wait_for_task_completion(int((_test_tasks[k] as Dictionary)["task"]))
	_test_tasks = {}
	if not _test_fx_task.is_empty():
		WorkerThreadPool.wait_for_task_completion(int(_test_fx_task["task"]))
	_test_fx_task = {}
	_test_dry = PackedFloat32Array()
	_test_req = {}          # replies still in flight are dropped on arrival
	_test_parts = {}
	_test_hold = PackedFloat32Array()
	_test_read = 0
	if _test_player != null:
		_test_player.stop()
	_test.text = "Test"


## One sentence of the audition arrived. Join everything that is now contiguous from the
## front - resample, sentence seam, the voice's room, through ONE effects chain so the room
## carries across the joins - and hand it to the stream.
func _on_test_part(id: int, result: Dictionary) -> void:
	var i := int(_test_req[id])
	_test_req.erase(id)
	_test_parts[i] = String(result.get("wav", ""))
	_sr = int(result.get("sample_rate", _sr))
	var s := _cfg_of(_test_name)
	var ratio := _pitch_ratio_of(s)
	if _test_fx == null:
		_test_fx = VoiceFX.new()
		_test_fx.pad_seed = hash(TEST_PASSAGE)
		_test_fx.setup(_sr)
		_apply_fx(_test_fx, s)
	# DECODE AND RESAMPLE OFF THE MAIN THREAD. Both are per-sample GDScript loops, and a long
	# sentence is several hundred thousand samples - measured at a 0.3 s frame on arrival.
	# Pure functions of the file, so a worker can do them; the join below waits for it.
	var box := [PackedFloat32Array()]
	var path := String(_test_parts[i])
	_test_tasks[i] = {"box": box, "task": WorkerThreadPool.add_task(func() -> void:
		var pcm := GenerativeEditor._decode_wav(path)
		if absf(ratio - 1.0) > 0.001:
			pcm = GenerativeEditor._resample_static(pcm, ratio)
		box[0] = pcm)}
	_join_test_parts()


## Join every prepared sentence that is now contiguous from the front onto the audition.
## Polled from [method _tick_test] too, because a worker can finish between replies.
func _join_test_parts() -> void:
	var s := _cfg_of(_test_name)
	while _test_tasks.has(_test_next):
		var t: Dictionary = _test_tasks[_test_next]
		if not WorkerThreadPool.is_task_completed(int(t["task"])):
			return
		WorkerThreadPool.wait_for_task_completion(int(t["task"]))
		_test_tasks.erase(_test_next)
		var wav := String(_test_parts.get(_test_next, ""))
		_test_parts.erase(_test_next)
		if not wav.is_empty():
			DirAccess.remove_absolute(ProjectSettings.globalize_path(wav))
		var part: PackedFloat32Array = (t["box"] as Array)[0]
		if _test_next > 0 and not part.is_empty():
			var gap := PackedFloat32Array()
			gap.resize(int(_seam_gap_of(s) * float(_sr)))
			gap.append_array(part)
			part = gap
		_test_next += 1
		if _test_next >= _test_chunks.size():
			# a second of silence so the room rings out rather than being cut
			var tail := PackedFloat32Array()
			tail.resize(_sr)
			part.append_array(tail)
		if not part.is_empty():
			_test_dry.append_array(part)
	_pump_test_fx()


## THE ROOM RUNS ON A WORKER. VoiceFX is GDScript DSP at about 15 microseconds a sample -
## measured, a third of the main thread for every second of audio, and 0.36 s in ONE frame
## when a sentence was processed whole: the reported "the entire UI keeps freezing, then
## releasing". So the dry audio goes through the chain on a worker, half a second at a time
## and ONE piece at a time (the chain is stateful, so pieces must run in order), and the main
## thread only moves finished samples into the ring. A dial moved mid-test reaches the next
## piece - the settings are taken when a piece starts, never changed under a running one.
const TEST_FX_PIECE := 0.5

func _pump_test_fx() -> void:
	if not _test_fx_task.is_empty():
		if not WorkerThreadPool.is_task_completed(int(_test_fx_task["task"])):
			return
		WorkerThreadPool.wait_for_task_completion(int(_test_fx_task["task"]))
		_test_hold.append_array((_test_fx_task["box"] as Array)[0])
		_test_fx_task = {}
	if _test_dry.is_empty() or _test_fx == null:
		_start_test_playback()
		return
	var n := mini(_test_dry.size(), int(TEST_FX_PIECE * float(_sr)))
	var piece := _test_dry.slice(0, n)
	_test_dry = _test_dry.slice(n)
	var fx := _test_fx
	var cfg := _cfg_of(_test_name)
	var box := [PackedFloat32Array()]
	_test_fx_task = {"box": box, "task": WorkerThreadPool.add_task(func() -> void:
		_apply_fx(fx, cfg)
		box[0] = fx.process(piece))}
	_start_test_playback()


func _start_test_playback() -> void:
	var done := _test_next >= _test_chunks.size() and _test_dry.is_empty() and _test_fx_task.is_empty()
	_set_status("Testing %s… %d of %d" % [_test_name, _test_next, _test_chunks.size()])
	# Unlike a reading, the audition starts on its FIRST sentence (a short one, see
	# TEST_PASSAGE): a test is pressed to hear a voice now, and a brief gap at one seam costs
	# less here than seconds of silence up front.
	var playing := _test_player != null and _test_player.playing
	if playing or (_test_hold.size() - _test_read <= 0 and not done):
		return
	if _test_player == null:
		_test_player = AudioStreamPlayer.new()
		add_child(_test_player)
	var gen := AudioStreamGenerator.new()
	gen.mix_rate = _sr
	# A short ring: the room is already applied (on a worker), so a frame only copies samples.
	gen.buffer_length = 1.0
	_test_player.stream = gen
	_test_player.play()
	var pb := _test_player.get_stream_playback() as AudioStreamGeneratorPlayback
	# MEASURED, not computed: Godot rounds a generator's ring to a power of two, so "empty"
	# is whatever it reports before the first push (see _drain_ready).
	_test_cap = pb.get_frames_available() if pb != null else 0


## Feed the audition's ring with finished (wet) samples, a frame's worth at a time; and end it
## once everything has been pushed and has played out (a generator never finishes by itself).
func _tick_test() -> void:
	if not _test_tasks.is_empty():
		_join_test_parts()
	if not _test_fx_task.is_empty() or not _test_dry.is_empty():
		_pump_test_fx()
	if _test_player == null or not _test_player.playing:
		return
	var pb := _test_player.get_stream_playback() as AudioStreamGeneratorPlayback
	if pb == null:
		return
	var n := mini(_test_hold.size() - _test_read, pb.get_frames_available())
	if n > 0:
		var wet := _test_hold.slice(_test_read, _test_read + n)
		var buf := PackedVector2Array()
		buf.resize(n)
		for k in n:
			buf[k] = Vector2(wet[k], wet[k])
		pb.push_buffer(buf)
		_test_pushed += n
		_test_read += n
		# reclaim the consumed head only when it is worth it, not every frame
		if _test_read > 4 * _sr:
			_test_hold = _test_hold.slice(_test_read)
			_test_read = 0
		return
	if _test_req.is_empty() and _test_next >= _test_chunks.size() and _test_dry.is_empty() \
			and _test_fx_task.is_empty() \
			and pb.get_frames_available() >= _test_cap - 64:
		_test_player.stop()
		_test.text = "Test"
		_set_status("Tested %s - %.1fs." % [_test_name, float(_test_pushed) / float(_sr)])


## One labelled 0..1 slider, live: these are buffer effects, so a change is
## audible on the very next frame rather than needing a re-synthesis.
## One labelled slider that drives the [Director] directly. Unlike the voice controls these need
## no re-plan and no persistence here - the Director clamps, applies immediately to the scene on
## screen, and owns its own saved value.
## [param tag] names the vehicle feature this row belongs to (see [constant Vehicle.USES]);
## an untagged row is one every vehicle uses and is always shown.
func _director_slider(box: VBoxContainer, name: String, lo: float, hi: float, step: float,
		initial: float, tip: String, apply: Callable, tag := "") -> HSlider:
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
	_slider_readout(row, sl)
	if not tag.is_empty():
		(_vehicle_rows.get_or_add(tag, []) as Array).append(row)
	return sl


## Show only what the chosen vehicle can use.
##
## ASKED OF THE DIRECTOR, NOT OF THE OPTIONBUTTON. The first cut read `_vehicle_pick.selected`,
## on the reasoning that the rows should follow what is being CHOSEN rather than what is
## running - and that is still the intent, it is just not what that property says. Selecting
## an item and the `item_selected` signal are separate things in Godot, so a picker driven
## from code (which is how it is exercised, and how a restored setting arrives) has the signal
## without the index and the rows stayed on the previous vehicle. `Director.vehicle` is set by
## that same callback, synchronously, one line above this call. Going through
## `resolved_vehicle` also means a run launched with `--vehicle comic` shows the comic's
## controls even though the stored setting says otherwise, which the picker alone cannot know.
##
## The rows move IMMEDIATELY even though the vehicle itself takes effect at the next reading:
## a control that stayed hidden until a restart would read as the picker not having worked.
func _sync_vehicle_rows() -> void:
	var key := Director.resolved_vehicle()
	for tag in _vehicle_rows:
		var on := Vehicle.uses(key, String(tag))
		for row in _vehicle_rows[tag] as Array:
			if is_instance_valid(row):
				(row as Control).visible = on


# --- the look: a post-process over the whole picture --------------------------
#
# THIS SITS WITH THE PICTURE SETTINGS, under Camera, because that is what it is - it belongs
# beside Vehicle and Scene hold rather than beside the voice dials. Like them it is the
# DIRECTOR'S, so a look set here is the look of every session: a reading, a synthesis take,
# a song in Auto mode, and an export render, which boots a second process against the same
# settings file and inherits it with no flag to pass.


## The filter block: one row per entry in [constant Filters.REGISTRY], built off the registry
## so a filter added there appears here with no wiring.
##
## A CHECKBOX AND A DIAL, NOT A PICKER, and that is the design rather than the layout. Being
## asked to choose between monochrome and grain is the wrong question - black and white film
## HAS grain - so every filter is independently switchable and they are all applied in one
## pass, in the registry's order.
##
## The dial is greyed rather than hidden while its filter is off: a control that vanishes
## takes its value with it as far as anyone looking can tell, and the value is kept.
func _build_filters(box: VBoxContainer) -> void:
	var head := HBoxContainer.new()
	head.add_theme_constant_override("separation", 8)
	box.add_child(head)
	var title := Label.new()
	title.text = "Look"
	title.custom_minimum_size = Vector2(72, 0)
	title.add_theme_font_size_override("font_size", 12)
	head.add_child(title)
	_filter_summary = Label.new()
	_filter_summary.add_theme_font_size_override("font_size", 11)
	_filter_summary.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_filter_summary.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	_filter_summary.modulate = Color(1, 1, 1, 0.6)
	head.add_child(_filter_summary)

	for key in Filters.REGISTRY:
		var k := String(key)
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		box.add_child(row)
		var cb := CheckBox.new()
		cb.text = String(Filters.LABELS.get(k, k))
		cb.custom_minimum_size = Vector2(128, 0)
		cb.add_theme_font_size_override("font_size", 12)
		cb.tooltip_text = String(Filters.BLURBS.get(k, ""))
		row.add_child(cb)
		var sl := HSlider.new()
		sl.min_value = 0.0
		sl.max_value = 1.0
		sl.step = 0.01
		sl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		sl.tooltip_text = ("How much of it. The top of every range here is deliberately too "
			+ "much, so the interesting settings are in the middle.\n\n"
			+ String(Filters.BLURBS.get(k, "")))
		row.add_child(sl)
		var readout := _slider_readout(row, sl)
		var live := Director.filter_amount(k)
		cb.set_pressed_no_signal(live > 0.0)
		_put_slider(sl, readout, live if live > 0.0 else float(Filters.DEFAULTS.get(k, 0.5)))
		sl.editable = live > 0.0
		# TICKING A FILTER ON MUST DO SOMETHING VISIBLE. A box whose dial happens to be at 0
		# reads as a broken checkbox, so switching on with nothing dialled in lands on the
		# registry's own starting point rather than on silence.
		cb.toggled.connect(func(on: bool) -> void:
			if on and sl.value <= 0.0:
				_put_slider(sl, readout, float(Filters.DEFAULTS.get(k, 0.5)))
			sl.editable = on
			Director.set_filter(k, sl.value if on else 0.0)
			_refresh_filters())
		sl.value_changed.connect(func(v: float) -> void:
			if not cb.button_pressed:
				return          # a greyed dial keeps its value and changes nothing
			Director.set_filter(k, v)
			_refresh_filters())
		_filter_rows[k] = {"box": cb, "slider": sl}
	_refresh_filters()


## The one-line summary beside the heading - what is actually on, in pipeline order.
func _refresh_filters() -> void:
	if _filter_summary == null or not is_instance_valid(_filter_summary):
		return
	var text := Filters.describe(Director.resolved_filters())
	_filter_summary.text = text
	_filter_summary.tooltip_text = ("Applied to the WHOLE picture, in this order, after "
		+ "every scene has drawn - and to nothing above it, so the subtitles stay clean. "
		+ "It is a Director setting like Vehicle, so it is also the look of a song in Auto "
		+ "mode and of an export render.\n\nOn now: " + text)


# --- films: real footage in a comic panel -------------------------------------
#
# THIS SITS UNDER THE VEHICLE PICKER because it only means anything to the comic, and a
# setting is easiest to understand next to the thing it qualifies.
#
# IT IS ALSO HIDDEN WHEN THE VEHICLE CANNOT USE IT, which reverses an earlier decision worth
# recording rather than quietly overwriting. The argument for always showing it was that
# someone building a library before switching over should not have to discover that the
# controls exist somewhere else first. That is answered by WHERE it sits: the picker is the
# row directly above, so the controls appear the moment the comic is chosen, in the place the
# eye is already looking. The argument against it was the stronger one - "there are a number
# of settings currently being displayed that ONLY work with the comic book vehicle" - because
# a control that does nothing teaches nothing, and there were two of them.

## The film library block: the list, an import button, and the frequency dial. Built into a
## group of its own so the whole block can be shown or hidden as one (see Vehicle.USES).
func _build_films(outer: VBoxContainer) -> void:
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	outer.add_child(box)
	(_vehicle_rows.get_or_add("films", []) as Array).append(box)
	var head := Label.new()
	head.text = "Films"
	head.add_theme_font_size_override("font_size", 12)
	head.tooltip_text = ("Real footage, cut into the comic among the drawn panels. Adding one "
		+ "is instant - nothing is converted up front. A clip is prepared in short windows, "
		+ "cut from the original only where the show is about to look, so a two-hour film "
		+ "costs the same as a two-minute one and a page that arrives before its window is "
		+ "ready simply goes without footage.\n\n"
		+ "KEEP THE ORIGINAL FILE where it is: windows are cut from it as they are needed, so "
		+ "moving or deleting it drops the clip from the list.\n\n"
		+ "A clip does NOT start from the beginning each time it appears. It plays from wherever "
		+ "it would be if it had been looping since the show started, so it reads as one film "
		+ "running behind the page that the comic occasionally cuts into.\n\n"
		+ "Only one panel at a time ever holds footage - two showing the same clip would show "
		+ "the same picture twice, because the position is decided by the clock alone.")
	box.add_child(head)

	_film_list = VBoxContainer.new()
	_film_list.add_theme_constant_override("separation", 2)
	box.add_child(_film_list)

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)
	var add := Button.new()
	add.text = "Import a clip…"
	add.focus_mode = Control.FOCUS_NONE
	add.tooltip_text = ("Pick a video file. It is added immediately - there is no transcode "
		+ "to wait for. The parts the show actually reaches are converted in the background, "
		+ "about a minute of film at a time, and the original is only ever read from.")
	add.pressed.connect(_open_film_dialog)
	row.add_child(add)
	_film_status = Label.new()
	_film_status.add_theme_font_size_override("font_size", 11)
	_film_status.add_theme_color_override("font_color", Color(0.55, 0.95, 0.75, 0.85))
	_film_status.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_film_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	row.add_child(_film_status)

	_film_freq = _director_slider(box, "How often", Films.FREQ_MIN, Films.FREQ_MAX, 0.05,
		Films.frequency(),
		"How often a comic page gives one of its panels to footage. This is per PAGE, not per "
		+ "panel, because only one panel may hold footage at a time - at 1 every page has one, "
		+ "at 0 none ever do. With no clips imported it does nothing.\n\n"
		+ "It is NOT competing with the scene types: a film is not one more entry drawn against "
		+ "the seventy-odd others, it is a separate decision made when the page turns. Measured, "
		+ "a page averages 3.3 panels, so 0.5 is film on about half the pages and one panel in "
		+ "seven; 1 is every page and one panel in three, which is the ceiling one-at-a-time "
		+ "allows.",
		func(v: float) -> void: Films.set_frequency(v))
	_refresh_films()


## Rebuild the list of imported clips. Cheap and total - the library is a handful of rows,
## and a diff would be more code than the thing it saves.
func _refresh_films() -> void:
	if _film_list == null or not is_instance_valid(_film_list):
		return
	for c in _film_list.get_children():
		c.queue_free()
	var list := Films.clips()
	if list.is_empty():
		var none := Label.new()
		none.text = "  (none imported)"
		none.add_theme_font_size_override("font_size", 11)
		none.add_theme_color_override("font_color", Color(0.7, 0.7, 0.75, 0.6))
		_film_list.add_child(none)
		return
	for i in list.size():
		var c: Dictionary = list[i]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 6)
		_film_list.add_child(row)
		var l := Label.new()
		var dur := float(c.get("duration", 0.0))
		l.text = "  %s  ·  %d:%02d" % [String(c.get("name", "clip")), int(dur / 60.0),
			int(fmod(dur, 60.0))]
		l.add_theme_font_size_override("font_size", 11)
		l.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		l.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
		l.tooltip_text = String(c.get("source", ""))
		row.add_child(l)
		var x := Button.new()
		x.text = "×"
		x.focus_mode = Control.FOCUS_NONE
		x.tooltip_text = "Forget this clip and delete the windows cut from it. The original "\
			+ "file is not touched."
		var at := i
		x.pressed.connect(func() -> void:
			Films.remove(at)
			_refresh_films())
		row.add_child(x)


func _open_film_dialog() -> void:
	if _film_dialog != null and is_instance_valid(_film_dialog):
		return
	_film_dialog = FileDialog.new()
	_film_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_film_dialog.access = FileDialog.ACCESS_FILESYSTEM
	# In-window, never native: the portal dialog shows nothing at all on a Linux box
	# without xdg-desktop-portal, which is the "I pressed it and nothing happened" report
	# Masking's own importer already carries this note for.
	_film_dialog.use_native_dialog = false
	_film_dialog.title = "Import a clip for the comic"
	_film_dialog.filters = PackedStringArray(["*.mp4, *.mov, *.mkv, *.webm, *.avi, *.ogv ; Video"])
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_film_dialog.current_dir = downloads
	_film_dialog.size = Vector2i(820, 560)
	_film_dialog.file_selected.connect(_start_film_import)
	_film_dialog.file_selected.connect(func(_p): _close_film_dialog())
	_film_dialog.canceled.connect(_close_film_dialog)
	add_child(_film_dialog)
	_film_dialog.popup_centered()


func _close_film_dialog() -> void:
	if _film_dialog != null and is_instance_valid(_film_dialog):
		_film_dialog.queue_free()
	_film_dialog = null


## ADDING A CLIP IS INSTANT. There is no transcode to wait for - a clip is prepared a
## window at a time, when something wants to play it (see Films.WINDOW), so this reads a
## duration and writes a row.
func _start_film_import(source: String) -> void:
	var err := Films.add(source)
	if not err.is_empty():
		_film_status.text = "⚠  " + err
		return
	_film_status.text = "✓  Added %s" % source.get_file().get_basename()
	_refresh_films()


## Polled from _process. A window cut is a subprocess, so something with a frame has to
## notice it finished; this is that, for as long as the panel is open. [FilmScene] does
## the same while a panel is live, which between them covers every moment one is awaited.
func _pump_films() -> void:
	Films.pump()
	# The status line follows the cutting rather than a one-shot import, because "is it
	# ready" is now a question with a running answer.
	if _film_status == null or not is_instance_valid(_film_status):
		return
	var cutting := 0
	for c in Films.clips():
		if Films.busy(c):
			cutting += 1
	if cutting != _film_cutting:
		_film_cutting = cutting
		if cutting > 0:
			_film_status.text = "⏳  Preparing %d window%s…" % [cutting,
				"" if cutting == 1 else "s"]
		elif not Films.clips().is_empty():
			_film_status.text = "✓  Ready"


## THE VEHICLE PICKER - what the show is carried on (see [Vehicle]). Built off the
## registry rather than a written-out list, so a new presentation appears here by being
## registered and nothing in this file has to know about it.
##
## It sits at the TOP of this section, above Scene hold, because it is the setting the
## ones below are qualified by: how long a scene holds means something slightly different
## when a "scene" is a panel on a page.
func _vehicle_option(box: VBoxContainer) -> OptionButton:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)
	var l := Label.new()
	l.text = "Vehicle"
	l.custom_minimum_size = Vector2(72, 0)
	l.add_theme_font_size_override("font_size", 12)
	row.add_child(l)
	var opt := OptionButton.new()
	opt.focus_mode = Control.FOCUS_NONE
	opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var keys: Array = Vehicle.REGISTRY.keys()
	var tip := "What the show is drawn ON, as opposed to what drives it - every mode gets " 		+ "every vehicle. Takes effect on the next reading, not the one already playing.\n"
	for k in keys:
		opt.add_item(String(Vehicle.LABELS.get(k, k)))
		tip += "\n%s - %s" % [Vehicle.LABELS.get(k, k), Vehicle.BLURBS.get(k, "")]
	opt.tooltip_text = tip
	# The DIRECTOR is the truth for this one (it is a whole-app setting and the export
	# render reads it), so the picker drives the setter rather than being bound directly -
	# and the setter is what persists it, through Settings like every other one.
	opt.select(maxi(0, keys.find(Director.vehicle)))
	opt.item_selected.connect(func(i: int) -> void:
		Director.set_vehicle(String(keys[i]))
		# The rows follow the PICKER immediately, even though the vehicle itself lands at the
		# next reading: a control that stayed hidden until a restart would read as the picker
		# not having worked.
		_sync_vehicle_rows()
		_note("Vehicle: %s - takes effect on the next reading." % Vehicle.LABELS.get(keys[i], keys[i])))
	row.add_child(opt)
	return opt


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
## ...and the longest run of sentences that may be called one paragraph. Nothing to
## do with style: a paragraph is where the pitch arc RESETS, so an unbroken block is
## an arc with no reset in it, and the register just falls for as long as the block
## does. Eight is a long paragraph; past that the text is not telling us where its
## paragraphs are, and reading it as consecutive ones of this length is the same
## guard SECTION_PARAS is a level up.
const PARA_CEILING := 8

func _place_chunks(out: Array, body: String) -> void:
	# Paragraph boundaries come from the SOURCE text (a blank line), which is the only place
	# they exist - Phonemes.parse hands back a flat run of sentences and knows nothing about
	# them. Counting sentence-final marks per paragraph maps one onto the other without
	# parsing the body twice.
	var per_para: Array = []
	# A BLANK LINE IS THE MARKER, BUT ONLY IF THE WRITER USED ANY. A script typed
	# with one newline between paragraphs - which is most of them, in a plain text
	# box with no formatting to lose - has no blank line anywhere in it, so this
	# found exactly one paragraph and handed the entire chapter a single arc that
	# descends from the first sentence to the last and never resets. That is half
	# of the "lower and lower and lower" report ([method _arc_semis] is the other
	# half); measured on one, the register fell monotonically across all sixteen
	# sentences instead of resetting six times.
	var para_mark := "\n\n"
	if body.find("\n\n") < 0:
		# ...but a line break is only a PARAGRAPH break if the lines are paragraphs.
		# Hard-wrapped prose breaks in the middle of sentences, and reading each of
		# those lines as a paragraph would reset the arc mid-sentence. The test is
		# whether the lines END the way sentences do - within a character or two of
		# the last one, so a closing quote or bracket still counts. Text that fails
		# it keeps the blank-line marker, finds no paragraphs, and is bounded by
		# PARA_CEILING instead, which is the right answer for prose that genuinely
		# carries no structure.
		var lines := 0
		var ended := 0
		for ln in body.split("\n", false):
			var t := String(ln).strip_edges()
			if t.is_empty():
				continue
			lines += 1
			var tail := t.substr(maxi(0, t.length() - 2))
			if tail.contains(".") or tail.contains("!") or tail.contains("?"):
				ended += 1
		if lines > 0 and float(ended) / float(lines) >= 0.6:
			para_mark = "\n"
	for para in body.split(para_mark, false):
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
	# ...and prose pasted as one unbroken block has no marker of either kind, so the
	# length of a paragraph is bounded whether the text says where they end or not.
	var capped: Array = []
	for n in per_para:
		var left := int(n)
		while left > PARA_CEILING:
			capped.append(PARA_CEILING)
			left -= PARA_CEILING
		capped.append(maxi(1, left))
	per_para = capped
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


## A right-aligned number beside a slider, kept current.
##
## Every dial in this panel was a bare track with a label and no value on it, so there was no way
## to know what any of them was actually set to - reported as "none of the toggles have actual
## scale values printed on them, so I never know what the true value is". That matters most on
## Pause, whose useful range is 0 to 10 and whose effect on a comma stops being linear past about
## 3.7 (see piper._pause_for), and on Pace, where the difference between 0.95 and 1.05 is audible
## across a chapter and invisible on the track.
##
## Wired to `value_changed`, which Godot also emits for programmatic sets, so the readout follows
## a slot being loaded as well as a drag.
## SHOW a value on a slider without telling anyone it changed.
##
## `set_value_no_signal` is the right call whenever the panel is DISPLAYING a stored value
## rather than receiving a new one - writing it back through the signal would be a change
## nobody made - but the readout built by [method _slider_readout] follows `value_changed`,
## so on its own it leaves the label saying whatever it was built with. That pair is a
## reported bug and it had two instances: every filter's readout said 0.00 until its dial was
## touched, including filters that were on and working, and ticking one on landed its default
## on the slider while the label went on saying 0.00.
func _put_slider(sl: HSlider, readout: Label, v: float, suffix := "") -> void:
	sl.set_value_no_signal(v)
	readout.text = ("%.2f" % v) + suffix


func _slider_readout(row: HBoxContainer, sl: HSlider, suffix := "") -> Label:
	var v := Label.new()
	v.custom_minimum_size = Vector2(42, 0)
	v.add_theme_font_size_override("font_size", 12)
	v.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	v.mouse_filter = Control.MOUSE_FILTER_IGNORE
	v.tooltip_text = sl.tooltip_text
	v.text = ("%.2f" % sl.value) + suffix
	sl.value_changed.connect(func(nv: float) -> void:
		v.text = ("%.2f" % nv) + suffix)
	row.add_child(v)
	return v


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
		if _syncing:
			return
		_capture_slot()
		_live_fx()
		_dirty = true
		_last_edit_ms = Time.get_ticks_msec())
	row.add_child(sl)
	_slider_readout(row, sl)
	return sl


## Push ONE VOICE'S settings onto a [VoiceFX] chain. ONE definition, because there
## were four lists of assignments - the slider callback, the config load, the
## stream opening and the export - and a dial had to appear in all four to work
## everywhere. Adding Room to three of them would have left one path silently
## dry, which is the same class of bug [method _pad_level_of] was written to end:
## live and render disagreeing about the room around the voice.
##
## It takes a SLOT rather than reading the panel, because the tab on screen is
## not necessarily the voice being heard: while tab 1 narrates, tab 2's dials are
## a room nobody is in yet.
func _apply_fx(fx: VoiceFX, s: Dictionary) -> void:
	if fx == null:
		return
	fx.echo_wet = float(s["echo"])
	fx.resonance = float(s["resonance"])
	fx.presence = _presence_of(s)
	fx.pad = _pad_level_of(s)
	# One dial, so [RoomFX] does the collapsing: size and wet open together, and
	# Resonance colours the tail the same way it does on Masking's bus.
	fx.room.from_dial(float(s["room"]), float(s["resonance"]))


## A dial moved on the panel, applied to the LIVE chain - but only if the tab
## being edited is the voice currently sounding. Turning up the reverb on a
## character who has not spoken yet must not put the narrator in a cathedral;
## the setting is stored either way, and arrives at that character's first word
## through [member _fx_marks].
func _live_fx() -> void:
	if not _fx_live_name.is_empty() and _fx_live_name != _tab_name():
		return
	_fx_live_name = _tab_name()
	_apply_fx(_fx, _cfg(_slot))


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
	if _test_req.has(id):
		_on_test_part(id, result)
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
		"words": _words_for(idx, result.get("tokens", [])),
		"spans": result.get("tokens", [])})
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
	_test.disabled = false
	_refresh_tab_labels()           # model names are known now, not just ids
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
	# THE REAL-TIME READ. In sync mode this is the file as it is on disk RIGHT NOW, not as
	# it was when the document was opened - which is what lets the author keep writing in
	# their own editor with ghost open beside it.
	var body := _doc.pull().strip_edges()
	if body.is_empty():
		_set_status("Nothing to speak yet.")
		return
	# The cast NOW, not at the next pause in typing: a name written a moment ago must read in
	# its own voice from the first Speak.
	_refresh_cast(body)
	if _test_busy():
		_stop_test()
	# _plan is a full teardown - stream closed, queues emptied, pending request
	# ids dropped so replies for the OLD text are discarded on arrival rather
	# than spliced into the new reading.
	_plan(body)
	if _chunks.is_empty():
		_set_status("Nothing speakable in that text.")
		return
	_go.text = "Speak"
	_sync_speak_buttons()
	_set_status("Planned %d chunk(s). Synthesizing the first…%s" % [_chunks.size(), _plan_note])
	_pump()


## Tear the stream down and read again from the top. _plan clears the pending
## request map, so replies still in flight for the old voice are dropped on
## arrival rather than spliced in.
func _restart_speaking() -> void:
	var body := _doc.pull().strip_edges()
	if body.is_empty():
		return
	_refresh_cast(body)
	_plan(body)
	if _chunks.is_empty():
		return
	_sync_speak_buttons()
	_set_status("Voice changed - reading again from the start…%s" % _plan_note)
	_pump()


## Cut the text into chunks at sentence boundaries, recording per chunk the
## phone stream and the word spans within it. Planning is pure front end - no
## synthesis - so it stays cheap even for a whole chapter.
## EVERY PIECE OF STATE A READING OWNS, put back. Split out of [method _plan] so that
## stopping and re-planning cannot drift apart: they were one block, and a Stop button that
## reset "most of" it is the kind that leaves a stream half open.
##
## The pending request map is the important one. Replies for chunks already asked for are still
## in flight on the host, and dropping their ids here is what makes them discarded on arrival
## instead of spliced into a reading that no longer exists.
func _reset_playback() -> void:
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
	_fx_marks = []
	_fx_live_name = ""
	_fx_queued_name = ""
	# No scrub hooks are registered by this editor (see the withdrawal note above
	# _repace), but clear them anyway: a file session opened earlier in the same process
	# may have left some, and they must not describe a reading that no longer exists.
	Spectrum.scrub_pos = Callable()
	Spectrum.scrub_len = Callable()
	Spectrum.scrub_seek = Callable()
	# The intro silence belongs to a reading that is STARTING, so it is seeded in _plan and
	# only cleared here. `_elapsed` is the clock every word span and every seam is measured
	# against, so starting it at the end of that silence offsets the whole reading, subtitles
	# included, with nothing else needing to know. Inserting it later would put it AFTER the
	# first sentence - the audition spoke its opening line, went quiet for five seconds, then
	# carried on.
	_lead_in = 0.0
	_sub_words.clear()          # cleared in place: Subtitles holds this by reference


## Speak and Stop, from one place. Two buttons whose enabled state is set at each of the
## several points a reading starts or ends is two buttons that will eventually disagree.
func _sync_speak_buttons() -> void:
	var reading := not _chunks.is_empty()
	if _stop != null:
		_stop.disabled = not reading
	if _go != null and not reading:
		_go.text = "Speak"


## Plan a reading: tear the old one down, then cut the text into chunks.
func _plan(body: String) -> void:
	_reset_playback()
	# THE INTRO, seeded HERE rather than at the moment the stream opens - see the note in
	# _reset_playback's lead-in block, which this replaces for a reading that is starting.
	_lead_in = maxf(0.0, Director.intro_hold)
	if _lead_in > 0.0:
		_pending.resize(int(_lead_in * float(_sr)))
		_elapsed = _lead_in
	_chunks = _build_chunks(body)


## STOP: end the reading and hand the stage back.
##
## There was no way to do this. The Speak button's tooltip claimed "press again while it is
## running to stop" and that was simply untrue - _on_speak re-plans and reads again from the
## top, whatever is playing. Reported as: "it would be preferable to stop the real-time scene
## before I do an export, and today I've just been restarting the program."
##
## Tearing down the STREAM is the half the editor cannot do alone: the playback, the Director's
## attachment to the stage and the subtitle overlay all belong to the owner, which is why
## `end_stream` exists and is wired beside `begin_stream` (see main._end_generative_stream).
## The synth editor has had both since it was written; this path was given only the opening
## half, so nothing was ever able to close it.
func _stop_speaking() -> void:
	if _chunks.is_empty() and _playback == null:
		return
	_reset_playback()
	if end_stream.is_valid():
		end_stream.call()
	_sync_speak_buttons()
	_set_status("Stopped. Press Speak to read again from the top.")


## Pure: text in, chunk plan out. Shared by playback and by export, so an export
## cannot disturb the reading in progress.
##
## The speaker cues are resolved HERE and nowhere else: past this point a chunk
## carries a slot number and every consumer - the request, the resampler, the
## seam, the room - reads its settings from that. Nothing downstream has to know
## that a script can change voice.
func _build_chunks(body: String) -> Array:
	var out: Array = []
	var sentence_no := 0
	var bare: PackedStringArray = PackedStringArray()
	_plan_note = ""
	var segs := _split_speakers(body)
	_hold_i = 0
	for seg in segs:
		var text := String((seg as Dictionary)["text"])
		var who := String((seg as Dictionary)["speaker"])
		# Asked per PASSAGE rather than of the whole box, so a macro sitting in a
		# stripped authoring note is not reported as one that will be missing
		# from the reading - it was never going to be read either way.
		bare.append_array(TextNorm.unresolved_macros(text))
		var part := _cut(text, sentence_no)
		var chunks: Array = part["chunks"]
		sentence_no = int(part["sentence_no"])
		if chunks.is_empty():
			continue
		# PLACEMENT IS PER PASSAGE. A speaker's turn is its own piece of
		# discourse - it opens in its own register and settles across its own
		# paragraphs - so measuring it against the whole chapter would hand the
		# second voice the contour of a paragraph it is not in.
		_place_chunks(chunks, text)
		for c in chunks:
			(c as Dictionary)["speaker"] = who
		out.append_array(chunks)
	# LAST, so it wins the status line. TextNorm has already warned into the log,
	# but this is the surface someone is looking at with their hand on Speak, and
	# a macro with no default is words missing from a reading about to be made.
	if not bare.is_empty():
		_note("%d macro(s) will not read as intended: %s - write ${NAME:value}"
			% [bare.size(), ", ".join(bare)])
	return out


## One finding about the script, for the line that reports the plan.
func _note(msg: String) -> void:
	_plan_note += ("  " if _plan_note.is_empty() else "; ") + msg


## One passage, cut at sentence boundaries. `sentence_no` runs across the WHOLE
## script rather than restarting per passage: [Subtitles] windows the display by
## that index, and two speakers sharing sentence 0 would have the overlay draw
## both at once.
func _cut(body: String, sentence_no: int) -> Dictionary:
	var out: Array = []
	var toks: Array = []
	var words: Array = []
	var holds: Array = []          # [{tok, sec, before}] - see _splice_holds
	var sentences := 0
	for sentence in Phonemes.parse(body):
		for w in sentence:
			var ph: Array = w.phones
			var st: Array = w.get("stress", [])
			var start := toks.size()
			# HESITATIONS, in the order the phonemizer met their sentinels - which is the
			# order _split_speakers queued their lengths in.
			for key in ["hold_before", "hold"]:
				var n := int(w.get(key, 0))
				if n <= 0:
					continue
				var sec := 0.0
				for _k in n:
					if _hold_i < _holds.size():
						sec += float(_holds[_hold_i])
					_hold_i += 1
				if sec > 0.0:
					holds.append({"tok": start, "sec": sec, "before": key == "hold_before"})
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
			# ONE SUBTITLE ENTRY PER SOURCE RUN. `2009` is three spoken words and
			# one thing on the page, so the words after the first in a rewritten
			# run do not get their own entry - they extend this one's END, and
			# the highlight sweeps the numeral across all three rather than
			# flashing it over the first syllable. See Phonemes.parse.
			var span := int(w.get("src_span", -1))
			if span >= 0 and not words.is_empty() \
					and int((words[words.size() - 1] as Dictionary).get("span", -1)) == span:
				(words[words.size() - 1] as Dictionary)["end"] = start
			else:
				# A run whose first word the phonemizer dropped would leave a
				# continuation with nothing to draw; show what is being said
				# rather than an empty card.
				var shown := String(w.get("display", w.text))
				words.append({"text": shown if not shown.is_empty() else String(w.text),
					"index": start, "end": start, "span": span, "sentence": sentence_no,
					"emph": int(w.get("emph", 0))})
		sentence_no += 1
		sentences += 1
		if sentences >= CHUNK_SENTENCES:
			out.append({"tokens": toks, "words": words, "holds": holds})
			toks = []
			words = []
			holds = []
			sentences = 0
	if not toks.is_empty():
		out.append({"tokens": toks, "words": words, "holds": holds})
	return {"chunks": out, "sentence_no": sentence_no}


## A slot's preset, and the resample ratio it implies.
func _preset_of(s: Dictionary) -> Dictionary:
	var keys := TONE_PRESETS.keys()
	return TONE_PRESETS[keys[clampi(int(s["tone"]), 0, keys.size() - 1)]]


func _pitch_ratio_of(s: Dictionary) -> float:
	return pow(2.0, float(_preset_of(s)["semis"]) / 12.0)


## One voice's delivery settings, as the host expects them.
##
## These used to be null-guarded against a missing slider, because export_take and
## _pump are both reachable from outside this editor's own lifecycle. They read a
## SLOT now, and a slot is [constant SLOT_DEFAULTS] merged with whatever was
## stored - so the missing-value case is answered by the schema, once, for every
## setting rather than one guard per dial.
func _delivery_of(s: Dictionary) -> Dictionary:
	return {
		"dynamics": _open_up(float(s["dynamics"])),
		"prosody_arc": _arc_semis(float(s["arc"])),
		"effort": _open_up(float(s["effort"])),
	}


## THE PARAGRAPH PITCH ARC, in semitones peak-to-peak.
##
## Its own curve rather than [method _open_up] x a ceiling, because this dial had a
## hard limit the other two do not, and it was not a limit of taste. The arc's pitch
## move is bought by RESAMPLING (piper.py `_discourse_plan` -> `_resample`, the same
## trick the Tone shift uses), and a resample moves the formants with the pitch - so
## it did not change how high the reader was speaking, it changed HOW BIG THEY WERE.
##
## Reported exactly that way: "the voice becomes lower and lower and lower; it
## completely transforms the voice into another voice by the end of the arc - the
## voice doesn't actually maintain its identity", with everything past 0.2 unusable.
## Measured on a sixteen-sentence script, the dial spanned 7.5 semitones at the top
## of its travel, a 54% change in apparent vocal-tract scale between a paragraph's
## opening sentence and its last. The dial was not too strong. It was asking a
## formant-SHIFTING resampler for a thing only a formant-PRESERVING shifter can do.
##
## So the shifter was fixed instead of the dial being capped to hide it: piper.py
## `_restore_formants` puts the speaker's own resonances back after the resample,
## leaving the pitch move and nothing else. What bounds the ceiling now is the
## linguistics rather than the DSP - 't Hart, Collier & Cohen put declination at
## one to two semitones over an utterance, with the deepest boundaries resetting
## further - so 4 semitones peak to peak is a top of travel that is more than any
## speaker would do and still the same speaker doing it.
##
## The exponent is FITTED, not chosen: it holds the old curve's value at 0.15,
## because 0.10 and 0.15 were reported as the settings that already work and a fix
## for the top of a dial has no business moving the bottom of it. It tracks the old
## curve within 5% up to about 0.4 and only then bends away, which is precisely the
## half that was broken.
const ARC_CEILING_SEMIS := 4.0
const ARC_KNEE := 1.13

func _arc_semis(k: float) -> float:
	return ARC_CEILING_SEMIS * pow(clampf(k, 0.0, 1.0), ARC_KNEE)


## The top of a delivery dial, opened up - identity at the bottom, 2.5x at the top.
##
## The first ceilings were set where I thought the results stopped being good, which
## is not my call to make: at full travel Dynamics only slowed the last sentence of a
## paragraph by 18%, so the upper half of the dial was doing almost nothing and the
## reported symptom was exactly that. (Arc was opened up the same way and has since
## been given its own curve - see [method _arc_semis] - because its ceiling is set by
## the resampler rather than by taste.)
##
## AND THIS FUNCTION IS ONLY HALF OF A DIAL. It multiplies what the backend is handed;
## the backend then multiplies its own coefficients by that. Those coefficients were
## written as their own ceilings - piper's `_discourse_plan` has documented "18% at
## full depth" since it was written - so opening the dial to 2.5x quietly took the
## paragraph's final lengthening to 45%, the section's to 17%, and a sentence sitting
## at the end of both to two and a half times its own length. Reported as "the
## cadence/pace/speed of the voice becomes slower and slower and slower", attributed
## to Arc, and measured on the reporter's own settings as 15% from Dynamics and 0.0%
## from Arc. piper.DEPTH_TOP now divides it back out, so the figures documented there
## are what the top of THIS dial delivers. A ceiling opened here has to be paid for
## there; the two files are one control.
func _open_up(k: float) -> float:
	return k * (1.0 + 1.5 * k * k)


## The ambience bed's level. Just the slider now - see [constant TONE_PRESETS] for why
## the preset writes this dial instead of secretly adding to it.
##
## Kept as a named function rather than inlined because the stream open, the export, the
## slider callback and the config load all have to agree about it, and they did not once:
## two of the four added the preset's contribution and two did not, so moving the slider
## during a reading silently dropped it and the export put it back.
func _pad_level_of(s: Dictionary) -> float:
	return clampf(float(s["ambience"]), 0.0, 1.0)


## How close the reader stands. The pad's twin, and the same story.
##
## Floored well above zero: the dial may push the voice back, never mute it, and
## [member VoiceFX.presence] is a gain as well as a filter.
func _presence_of(s: Dictionary) -> float:
	return clampf(float(s["presence"]), 0.25, 1.0)


func _pause_scale_of(s: Dictionary) -> float:
	return clampf(float(s["pause"]), 0.0, MAX_PAUSE_SCALE)


## The silence at a chunk seam. The host inserts the same figure between
## sentences INSIDE a chunk, so a seam and an interior boundary stay the same
## length however the reading happens to have been cut up.
func _seam_gap_of(s: Dictionary) -> float:
	return _rest_for(SENTENCE_GAP, SENTENCE_DWELL, _pause_scale_of(s))


## The silence to place at a mark whose natural rest is `dwell` and whose top-up at
## Pause 1.0 is `top_up`. MIRRORS piper._rest_for, and has to: the host owns the marks
## inside a sentence and this owns the seam between two of them, so a reading cut one
## way has to rest exactly as long as the same reading cut the other way.
##
## WHY THE DIAL MULTIPLIES THE WHOLE REST rather than our own share of it. What a reader
## hears at a full stop is the model's own trailing silence plus the seam we add, and
## scaling only the second half stretches the DIFFERENCES between the marks instead of
## the marks. Reported as "the pause after a comma and the pause after a sentence are
## very different... at 6.0 the comma-pauses feel about right, while the period-pauses
## feel far too slow" - no one setting could suit both, because their ratio was moving
## with the dial. It no longer moves: a full stop rests twice as long as a comma at 1.0
## and at 10.0 and everywhere between. See piper.DWELL for the measurements.
func _rest_for(top_up: float, dwell: float, scale: float) -> float:
	var mult := _pause_multiplier(scale)
	# At 1.0 the answer IS the top-up, said exactly - see piper._rest_for for the float.
	if is_equal_approx(mult, 1.0):
		return maxf(0.0, top_up)
	return maxf(0.0, (dwell + top_up) * mult - dwell)


## How much longer than natural every rest in this reading is. Mirrors piper's, and the
## power law is deliberate: the saturating curve this replaces reached 3.2 at the top of
## the dial and 3.29 at a hundred, so "the pause effect barely seems to work at 10x" could
## not have been answered by allowing a bigger number - the curve had topped out, not the
## dial. This one is exactly 1.0 at Pause 1.0 by construction, within 3% of the old curve
## up to 3.0, and 5.0 at the top, where a full stop rests two and a half seconds.
func _pause_multiplier(scale: float) -> float:
	return pow(clampf(scale, 0.0, MAX_PAUSE_SCALE), PAUSE_GAIN)


## The silence BEFORE chunk `idx` - the ordinary sentence seam, plus the turn
## rest when the chunk before it was somebody else's.
##
## Both the live window and the export join chunks, so this is written once for
## the same reason [method _request_args] is: the render has to be the reading
## that was auditioned, and a second copy of a rule is where the two part.
func _gap_before(chunks: Array, idx: int, s: Dictionary) -> float:
	var g := _seam_gap_of(s)
	if idx <= 0 or idx >= chunks.size():
		return g
	if String((chunks[idx - 1] as Dictionary).get("speaker", "")) \
			== String((chunks[idx] as Dictionary).get("speaker", "")):
		return g
	return minf(g + TURN_GAP * clampf(_turn.value if _turn != null else 1.0,
		0.0, MAX_TURN_SCALE), TURN_CEILING)


## Keep LOOKAHEAD chunks in flight and start playback as soon as the chunk we
## are waiting for exists.
## How much finished audio is waiting to be heard: decoded but not yet pushed,
## plus whatever is still sitting in the ring.
func _buffered_seconds() -> float:
	var queued := 0
	if _playback != null and _ring_capacity > 0:
		queued = maxi(0, _ring_capacity - int(_playback.get_frames_available()))
	return float(maxi(_pending.size() - _read, 0) + queued) / float(maxi(_sr, 1))


## Which checkpoint a slot reads with. A tab may name a voice this install does
## not have - a config copied between machines, a model deleted - and the answer
## to that is to read in whatever IS selected, not to fall silent halfway
## through a chapter.
func _voice_id_of(s: Dictionary) -> String:
	var want := String(s["voice"])
	for v in _voice_meta:
		if String((v as Dictionary).get("id", "")) == want:
			return want
	var i := _voices.selected
	if i >= 0 and i < _voice_meta.size():
		return String((_voice_meta[i] as Dictionary).get("id", ""))
	return ""


## A slot's reader id, clamped to what its own model actually holds. Slots are
## copied when a tab is added, so a speaker id chosen on libritts (904 readers)
## can easily outlive the switch to a single-speaker voice.
func _speaker_of(s: Dictionary) -> int:
	var vid := _voice_id_of(s)
	for v in _voice_meta:
		if String((v as Dictionary).get("id", "")) == vid:
			return clampi(int(s["speaker"]), 0, maxi(0, int((v as Dictionary).get("speakers", 1)) - 1))
	return maxi(0, int(s["speaker"]))


## Everything the host needs to read ONE chunk in ONE voice. Written once and
## used by both the live window and the export, because the render must be the
## performance that was auditioned - a second copy of this list is how the two
## drift apart a parameter at a time.
func _request_args(s: Dictionary, ch: Dictionary) -> Dictionary:
	var t := _preset_of(s)
	# length_scale = r / pace: the model speaks r times slower so that
	# playing back r times faster restores the intended pace
	var r := _pitch_ratio_of(s)
	var d := _delivery_of(s)
	return {
		"length_scale": r / maxf(float(s["pace"]) * float(t["pace"]), 0.1),
		"noise_scale": float(t["noise"]), "noise_w": float(t["noise_w"]),
		"whisper": float(t["whisper"]), "muffle": float(t["muffle"]),
		"speaker": _speaker_of(s),
		"sentence_gap": SENTENCE_GAP, "pause_scale": _pause_scale_of(s),
		"dynamics": d["dynamics"],
		"prosody_arc": d["prosody_arc"],
		"effort": d["effort"],
		"plan_u": float(ch.get("plan_u", 0.0)),
		"plan_v": float(ch.get("plan_v", 0.0)),
		"tokens": ch["tokens"],
	}


func _pump() -> void:
	if _voice_meta.is_empty():
		return
	while _in_flight < LOOKAHEAD and _next_to_request < _chunks.size() \
			and _buffered_seconds() < LOOKAHEAD_SECONDS:
		var idx := _next_to_request
		_next_to_request += 1
		_in_flight += 1
		var s := _cfg_of(String((_chunks[idx] as Dictionary).get("speaker", "")))
		var id := _host.request("", _voice_id_of(s),
			TAKE_DIR + "/chunk_%d_%d.wav" % [idx, _epoch], _request_args(s, _chunks[idx]), null)
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
		var idx := int(take["index"])
		_next_to_play += 1

		var pcm := _read_wav(String(take["wav"]))
		if pcm.is_empty():
			continue
		var who := ""
		var holds: Array = []
		if idx >= 0 and idx < _chunks.size():
			who = String((_chunks[idx] as Dictionary).get("speaker", ""))
			holds = (_chunks[idx] as Dictionary).get("holds", [])
		var s := _cfg_of(who)
		var ratio := _pitch_ratio_of(s)
		if absf(ratio - 1.0) > 0.001:
			pcm = _resample(pcm, ratio)
		# THE HESITATIONS go in now, after the resample, in the chunk's own time - so the
		# word timings below can be moved by exactly what was inserted before them.
		var spliced := _splice_holds(pcm, holds, take.get("spans", []), ratio)
		pcm = spliced["pcm"]
		if _next_to_play > 1:
			# a breath between sentences, at the seam the host cannot see.
			# Scaled by Pause like every other rest, or the control would do
			# nothing at all at CHUNK_SENTENCES = 1 - every sentence boundary in
			# the reading IS a seam, and they would all stay 0.32 s. A handover
			# takes the Turn rest on top.
			var seam := _gap_before(_chunks, idx, s)
			var gap := PackedFloat32Array()
			gap.resize(int(seam * float(_sr)))
			_pending.append_array(gap)
			_elapsed += seam
		# SCHEDULE THE ROOM, do not switch it here. This runs when a chunk is
		# DECODED, which is seconds ahead of when it is heard - dialling the
		# chain now would put the next speaker's room over the end of this one's
		# last sentence. The mark carries the frame instead, and _process applies
		# it as the playhead reaches it. The first mark sits at 0 so the intro is
		# already in the opening voice's room rather than in nobody's.
		#
		# AFTER the gap, deliberately: the handover silence belongs to the voice
		# leaving it, whose reverb is still decaying through it. Dialling the new
		# room at the start of the rest would cut that tail over to a different
		# space halfway down, which is the one thing a room never does.
		if who != _fx_queued_name:
			_fx_queued_name = who
			_fx_marks.append({
				"at": 0 if _fx_marks.is_empty() else _pushed + _pending.size() - _read,
				"speaker": who})
		# the model timed each chunk from zero; shift into stream time
		for w in take["words"]:
			var d: Dictionary = (w as Dictionary).duplicate()
			# the model timed this at its own (slower) rate; resampling divided
			# every duration by the ratio, so the timings must follow
			d["t0"] = _shifted(float(d["t0"]), spliced["cuts"], ratio, true) + _elapsed
			d["t1"] = _shifted(float(d["t1"]), spliced["cuts"], ratio, false) + _elapsed
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
			# reading. The opening voice's, not the tab on screen's.
			_fx_live_name = String((_fx_marks[0] as Dictionary)["speaker"]) \
				if not _fx_marks.is_empty() else who
			_apply_fx(_fx, _cfg_of(_fx_live_name))
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
	_set_status("%s at pace %.2fx, pause %.2fx - regenerating from chunk %d…"
		% [_tab_name(), _rate.value, clampf(_pause.value, 0.0, MAX_PAUSE_SCALE), _next_to_play + 1])
	_pump()


## SPLICE THE HESITATIONS into one chunk's audio, after its resample. Returns
## `{pcm, cuts}`, where `cuts` is `[{t, add}]` in the MODEL's time (the clock the aligner's
## spans are on) - [method _shifted] moves a word timing by every cut before it.
##
## WHERE, and why there: the aligner says where each word ends and the next begins, and the
## cut goes at the MIDDLE of that gap - the one place that is silence whatever the voice did
## either side (a cut at a word's own end lands on its decay). A rest before the chunk's first
## word opens the chunk; one after its last closes it. Both still sit inside the chunk rather
## than being folded into the seam, so the live window and the export - which join chunks
## differently - cannot disagree about them.
##
## A short fade either side of every cut, because the gap between two words can be zero and
## a hard edge inside a vowel is a click.
const HOLD_FADE := 0.006

func _splice_holds(pcm: PackedFloat32Array, holds: Array, spans: Array, ratio: float) -> Dictionary:
	var cuts: Array = []
	if holds.is_empty():
		return {"pcm": pcm, "cuts": cuts}
	var by_index := {}
	for sp in spans:
		by_index[int((sp as Dictionary).get("index", -1))] = sp
	var end := INF
	for h in holds:
		var k := int((h as Dictionary)["tok"])
		var cur: Variant = by_index.get(k)
		var t := end
		if bool((h as Dictionary)["before"]):
			var prev: Variant = by_index.get(k - 1)
			if k == 0:
				t = 0.0
			elif prev != null and cur != null:
				t = (float(prev["t1"]) + float(cur["t0"])) * 0.5
			elif cur != null:
				t = float(cur["t0"])
			elif prev != null:
				t = float(prev["t1"])
		else:
			var nxt: Variant = by_index.get(k + 1)
			if nxt == null:
				t = end              # the chunk's last word: rest after all of it
			elif cur != null:
				t = (float(cur["t1"]) + float(nxt["t0"])) * 0.5
			else:
				t = float(nxt["t0"])
		cuts.append({"t": t, "add": float((h as Dictionary)["sec"])})
	cuts.sort_custom(func(a, b): return float(a["t"]) < float(b["t"]))
	var out := PackedFloat32Array()
	var from := 0
	var fade := maxi(1, int(HOLD_FADE * float(_sr)))
	for c in cuts:
		var ct := float(c["t"])
		var at := pcm.size() if is_inf(ct) else clampi(int(round(ct / ratio * float(_sr))), from, pcm.size())
		var piece := pcm.slice(from, at)
		# fade the tail of what precedes the rest, and (below) the head of what follows it
		for i in mini(fade, piece.size()):
			piece[piece.size() - 1 - i] *= float(i) / float(fade)
		if from > 0 and not out.is_empty():
			for i in mini(fade, piece.size()):
				piece[i] *= float(i) / float(fade)
		out.append_array(piece)
		var gap := PackedFloat32Array()
		gap.resize(int(float(c["add"]) * float(_sr)))
		out.append_array(gap)
		from = at
	var rest := pcm.slice(from)
	for i in mini(fade, rest.size()):
		rest[i] *= float(i) / float(fade)
	out.append_array(rest)
	return {"pcm": out, "cuts": cuts}


## A word timing from the model's clock onto the spliced take's: divided by the resample
## ratio, then moved by every rest spliced in before it. A START at exactly a cut is moved
## (the rest precedes that word); an END at exactly a cut is not (the rest follows it).
static func _shifted(t: float, cuts: Array, ratio: float, is_start: bool) -> float:
	var add := 0.0
	for c in cuts:
		var ct := float(c["t"])
		if ct < t or (is_start and is_equal_approx(ct, t)):
			add += float(c["add"])
	return t / ratio + add


## Linear-interpolating resample. Reading at `ratio` samples per output sample
## raises the pitch by that factor and shortens the audio by it; the model was
## asked to speak proportionally slower, so the two cancel and only the pitch
## moves. Linear is enough here - the ratios are within a few semitones, so the
## interpolation error sits far below the voice.
func _resample(src: PackedFloat32Array, ratio: float) -> PackedFloat32Array:
	return _resample_static(src, ratio)


## The same, static, so a worker thread can run it (the audition decodes off the main thread).
static func _resample_static(src: PackedFloat32Array, ratio: float) -> PackedFloat32Array:
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
	return _decode_wav(path)


## The same, static, for a worker thread - see [method _resample_static].
static func _decode_wav(path: String) -> PackedFloat32Array:
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
	var body := _doc.pull().strip_edges()
	if body.is_empty() or _voices.selected < 0:
		return ""
	_refresh_cast(body)
	var chunks := _build_chunks(body)
	if chunks.is_empty():
		return ""
	var stamp := Time.get_ticks_msec()
	var pcm := PackedFloat32Array()
	var words: Array = []
	var elapsed := 0.0
	# Where the reading changes voice, in frames of the finished take - the
	# export's copy of [member _fx_marks], for the same reason and applied the
	# same way at the bottom of this function.
	var marks: Array = []
	var last_who := ""

	for i in chunks.size():
		_set_status("Rendering for export: %d of %d…" % [i + 1, chunks.size()])
		# The SAME arguments the preview used, from the SAME builder. An export
		# that re-derived its own would be a different performance from the one
		# that was auditioned, which is the one thing the render must never be.
		var who := String((chunks[i] as Dictionary).get("speaker", ""))
		var s := _cfg_of(who)
		var ratio := _pitch_ratio_of(s)
		var id := _host.request("", _voice_id_of(s),
			TAKE_DIR + "/export_%d_%d.wav" % [stamp, i], _request_args(s, chunks[i]), null)
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
		# The same splice the live window makes, from the same function.
		var spliced := _splice_holds(part, (chunks[i] as Dictionary).get("holds", []),
			out.get("tokens", []), ratio)
		part = spliced["pcm"]
		if i > 0:
			var seam := _gap_before(chunks, i, s)
			var gap := PackedFloat32Array()
			gap.resize(int(seam * float(_sr)))
			pcm.append_array(gap)
			elapsed += seam
		# after the gap, for the reason spelled out in _drain_ready
		if marks.is_empty() or who != last_who:
			last_who = who
			marks.append({"at": pcm.size(), "speaker": who})
		var by_index := {}
		for sp in out.get("tokens", []):
			by_index[int((sp as Dictionary).get("index", -1))] = sp
		var rows: Array = []
		for w in chunks[i]["words"]:
			var span: Variant = by_index.get(int(w["index"]))
			var tail: Variant = by_index.get(int(w.get("end", w["index"])))
			if tail == null:
				tail = span
			rows.append({"text": String(w["text"]), "sentence": int(w["sentence"]),
				"emph": int(w.get("emph", 0)),
				"t0": 0.0 if span == null else
					_shifted(float((span as Dictionary).get("t0", 0.0)), spliced["cuts"], ratio, true)
					+ elapsed,
				"t1": 0.0 if tail == null else
					_shifted(float((tail as Dictionary).get("t1", 0.0)), spliced["cuts"], ratio, false)
					+ elapsed,
				"ok": span != null})
		words.append_array(_bridge_words(rows, "export chunk %d" % i))
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
	# SEED THE KEY, or the intro is silent anyway. The pad picks its tonic from the
	# tracked pitch of the voice, and the tracker returns 0 on silence - so during a
	# leading pad of pure zeros `_tonic` never rises off 0, `_start_tone` refuses to
	# schedule anything, and the bed only begins once the narration has already started,
	# which is precisely backwards. Measuring the voice FIRST and handing the chain its
	# key up front is what lets the bed be playing before the first word.
	if intro > 0.0:
		fx.prime_key(pcm)
	# ONE CHAIN, RE-DIALLED per passage - never one chain per voice. The effects
	# are stateful, so a second chain would start each speaker in a dead room and
	# cut the previous one's tail off at the change; re-dialling shares the decay
	# across the join, which is what a room in the world does when the person
	# talking in it changes.
	if marks.is_empty():
		marks = [{"at": 0, "speaker": ""}]
	var head := int(intro * float(_sr))
	for m in marks:
		(m as Dictionary)["at"] = int((m as Dictionary)["at"]) + head
	(marks[0] as Dictionary)["at"] = 0        # the first voice owns the intro
	var wet := PackedFloat32Array()
	for k in marks.size():
		var a := int((marks[k] as Dictionary)["at"])
		var b := pcm.size() if k == marks.size() - 1 else int((marks[k + 1] as Dictionary)["at"])
		if b <= a:
			continue
		_apply_fx(fx, _cfg_of(String((marks[k] as Dictionary)["speaker"])))
		wet.append_array(fx.process(pcm.slice(a, b)))
	pcm = wet

	var path := TAKE_DIR + "/take_%d.wav" % stamp
	var abs_path := _write_wav(path, pcm)
	# ALWAYS written now, words or not: the book vehicle reads the chapter from it.
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
		# THE BOOK rides along: a render has no editor, and a vehicle that typesets
		# pages needs the chapter those words came from (see BookVehicle).
		side.store_string(JSON.stringify({
			"words": shifted, "bookend": {"in": intro, "out": outro},
			"book": book_document(body)}))
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


## KEEP EVERY WORD, even one the aligner had no span for.
##
## Both paths below used to `continue` past a word whose token the host did not return a span
## for, which deletes it from the karaoke line without a word anywhere about it. That is how a
## chapter render came back reading "an opponent who left the building in" with the year simply
## absent (the root cause was upstream - see TextNorm._expand_numbers - but this is what made it
## SILENT, and it would hide the next one just as well).
##
## A word with no span keeps its text and takes a timing interpolated across the gap between its
## nearest aligned neighbours, so the line reads correctly and the highlight sweeps through it at
## a plausible rate. The count is reported once per take, because an aligner that misses words is
## a real fault worth seeing in the log even when the subtitle no longer loses them.
static func _bridge_words(rows: Array, label: String) -> Array:
	var missing := 0
	var n := rows.size()
	for i in n:
		if bool((rows[i] as Dictionary)["ok"]):
			continue
		missing += 1
		# The aligned neighbours either side, and how many unaligned words share the gap.
		var lo := i - 1
		while lo >= 0 and not bool((rows[lo] as Dictionary)["ok"]):
			lo -= 1
		var hi := i + 1
		while hi < n and not bool((rows[hi] as Dictionary)["ok"]):
			hi += 1
		var t0: float = float((rows[lo] as Dictionary)["t1"]) if lo >= 0 else 0.0
		var t1: float = float((rows[hi] as Dictionary)["t0"]) if hi < n else t0 + 0.25
		if t1 <= t0:
			t1 = t0 + 0.25
		var run := float(maxi(1, (hi if hi < n else n) - (lo + 1)))
		var k := float(i - (lo + 1))
		var step := (t1 - t0) / run
		var row: Dictionary = rows[i]
		row["t0"] = t0 + step * k
		row["t1"] = t0 + step * (k + 1.0)
	if missing > 0:
		push_warning("ghost/voice: %s - the aligner returned no span for %d of %d words; "
			% [label, missing, n] + "their subtitle timing is interpolated (text is intact)")
	var out: Array = []
	for r in rows:
		var d: Dictionary = r
		out.append({"text": d["text"], "sentence": d["sentence"], "t0": d["t0"], "t1": d["t1"],
			"emph": int(d.get("emph", 0))})
	return out


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
	var rows: Array = []
	for w in _chunks[idx]["words"]:
		var span: Variant = by_index.get(int(w["index"]))
		# A run's card is up from its first word to its LAST - `end` is that word
		# when the entry covers a rewritten run, and the entry's own word when it
		# does not.
		var tail: Variant = by_index.get(int(w.get("end", w["index"])))
		if tail == null:
			tail = span
		rows.append({"text": String(w["text"]), "sentence": int(w["sentence"]),
			"emph": int(w.get("emph", 0)),
			"t0": 0.0 if span == null else float((span as Dictionary).get("t0", 0.0)),
			"t1": 0.0 if tail == null else float((tail as Dictionary).get("t1", 0.0)),
			"ok": span != null})
	return _bridge_words(rows, "chunk %d" % idx)


