extends CanvasLayer
class_name Scrubber

## Scrubber - a seek bar for the session, so a ten minute take can be reviewed.
##
## The need it answers, verbatim: "it's very hard for me to scrub through a video, to
## return to a certain point... I have to watch 10 minutes of narration just to reach this
## scene again." Reviewing a long narration by waiting through it is not review, it is
## endurance, and it makes every visual note expensive to check.
##
## WHAT SEEKING DOES AND DOES NOT MOVE. Everything that READS the clock follows a seek
## exactly and immediately: the baked spectrum is a timeline lookup, the karaoke overlay
## reads `Spectrum.current.time - time_base`, the whole-show bookend fade is a function of
## position. The [Director] is the exception, and it is a real one - it is a SIMULATION
## rather than a function of t, so its scene choice, hold schedule and RNG stream evolve
## from the events it has actually seen. Seek forward and the show carries on from the
## scene that is up; it does not jump to the scene a from-the-start playthrough would have
## been showing. ([Echo] re-localizes against the harmonic signature rather than against
## elapsed time, so it does drift back toward alignment on its own - but that is a
## tendency, not a guarantee.)
##
## That is a deliberate line. Reproducing the exact visual state of an arbitrary t means
## replaying the Director from zero with drawing off, which this architecture could do and
## which a scrub bar should not: the point here is to reach a MOMENT IN THE AUDIO quickly.
## To reach a SCENE, `--scene <name>` pins one and skips the waiting entirely.
##
## IT IS ALWAYS VISIBLE in a live session. It never appears in an export, and not by a
## check - the render process returns out of main before Chrome is built, so none of this
## furniture exists there at all.
##
## A FILE ONLY, and that is the honest scope. Seeking a live synthesis generator was built
## and withdrawn (the reasoning is recorded in generative_editor.gd, above _repace): a
## generator's ring cannot be cleared while it plays, so every seek had to restart the
## stream, and repeated restarts left the audio audibly wrong. It could not have served its
## purpose in any case - see below.
##
## SO THE REVIEW WORKFLOW IS: render the take, then open the rendered file. That is a real
## seek, one operation, and a file boot replays the same deterministic show from the same
## seed - so scrubbing it shows what the export will actually contain, which scrubbing a
## live generator never could.
const BAR_H := 4.0                  # the resting rail, px
const BAR_H_ACTIVE := 8.0           # while the pointer is over it
const PAD := 26.0                   # distance from the bottom edge
const FADE := 7.0                   # per-second ease on the reveal

var _root: Control
var _shown := 0.0                   # 0..1 eased visibility
var _dragging := false
var _label: Label


func _ready() -> void:
	# Above the subtitles (9) and the workspace (100) so it is never buried, below the
	# feedback console (128) and the exporter (250) which are modal-ish and own the corner.
	layer = 120
	_root = Control.new()
	_root.set_anchors_preset(Control.PRESET_FULL_RECT)
	_root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_root)
	_label = Label.new()
	_label.add_theme_font_size_override("font_size", 12)
	_label.add_theme_color_override("font_color", Color(0.92, 0.94, 0.98, 0.9))
	_label.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.8))
	_label.add_theme_constant_override("shadow_offset_x", 1)
	_label.add_theme_constant_override("shadow_offset_y", 1)
	_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_root.add_child(_label)
	var painter := Painter.new()
	painter.owner_node = self
	painter.set_anchors_preset(Control.PRESET_FULL_RECT)
	painter.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_root.add_child(painter)
	_painter = painter


var _painter: Control


func _process(delta: float) -> void:
	# ALWAYS ON when the session can be seeked. It was revealed by pointer proximity, on
	# the reasoning that a control sitting in the frame spoils the shot - but the shot that
	# matters is the EXPORT, and an export never has this at all: the render process
	# returns out of main before Chrome is built, so none of this furniture exists there.
	# In a live session a hidden control is just a control nobody can find.
	var want := 1.0 if Spectrum.seekable() else 0.0
	_shown = lerpf(_shown, want, 1.0 - exp(-FADE * delta))
	if _shown < 0.004:
		_shown = 0.0
	_root.visible = _shown > 0.0
	if not _root.visible:
		return
	var t := Spectrum.scrub_position()
	var total := maxf(0.001, Spectrum.scrub_length())
	_label.text = "%s / %s" % [_clock(t), _clock(total)]
	_label.position = Vector2(_rail().position.x, _rail().position.y - 20.0)
	_label.modulate.a = _shown
	_painter.queue_redraw()


func _input(event: InputEvent) -> void:
	if not Spectrum.seekable():
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		var inside := _rail().grow(10.0).has_point(event.position)
		if event.pressed and inside:
			_dragging = true
			_seek_to(event.position.x)
			get_viewport().set_input_as_handled()
		elif not event.pressed and _dragging:
			_dragging = false
			get_viewport().set_input_as_handled()
	elif event is InputEventMouseMotion and _dragging:
		_seek_to(event.position.x)
		get_viewport().set_input_as_handled()
	elif event is InputEventKey and event.pressed and not event.echo:
		# Arrow keys are the control that actually gets used while watching, because they
		# need no aim: a fixed step, repeatable, without taking the eye off the frame.
		#
		# They are taken in _input, which runs BEFORE the GUI - otherwise the focused
		# control eats them first. In the Synthesis panel that is a row of sliders, so
		# pressing right walked the focus along the panel and moved a slider instead of
		# the playhead. Marking them handled is what stops that, so it is not optional.
		match event.keycode:
			KEY_LEFT:
				Spectrum.seek(Spectrum.scrub_position() - _step(event))
				get_viewport().set_input_as_handled()
			KEY_RIGHT:
				Spectrum.seek(Spectrum.scrub_position() + _step(event))
				get_viewport().set_input_as_handled()
			KEY_HOME:
				Spectrum.seek(0.0)
				get_viewport().set_input_as_handled()


## 10 s normally, 60 s with shift - the difference between "I missed a word" and "that
## scene was a couple of minutes back".
func _step(event: InputEventKey) -> float:
	return 60.0 if event.shift_pressed else 10.0


func _seek_to(x: float) -> void:
	var r := _rail()
	var f := clampf((x - r.position.x) / maxf(1.0, r.size.x), 0.0, 1.0)
	Spectrum.seek(f * Spectrum.scrub_length())


## The clickable rail, in viewport coordinates. Inset from both edges so a click meant for
## the very start or end cannot miss the control entirely.
func _rail() -> Rect2:
	var vp := _root.get_viewport_rect().size
	var margin := maxf(40.0, vp.x * 0.06)
	return Rect2(Vector2(margin, vp.y - PAD),
		Vector2(maxf(10.0, vp.x - margin * 2.0), BAR_H))


static func _clock(t: float) -> String:
	var s := int(maxf(0.0, t))
	return "%d:%02d" % [s / 60, s % 60]


class Painter:
	extends Control
	var owner_node: Scrubber

	func _draw() -> void:
		if owner_node == null:
			return
		var a: float = owner_node._shown
		if a <= 0.0:
			return
		var r: Rect2 = owner_node._rail()
		var h := lerpf(Scrubber.BAR_H, Scrubber.BAR_H_ACTIVE,
			1.0 if owner_node._dragging else 0.0)
		r.position.y -= (h - Scrubber.BAR_H) * 0.5
		r.size.y = h
		var total: float = maxf(0.001, Spectrum.scrub_length())
		var f: float = clampf(Spectrum.scrub_position() / total, 0.0, 1.0)
		# A dark rail under a light fill, both translucent: the bar has to be readable over
		# a white tidepool and a black void alike, and neither a light nor a dark bar alone
		# manages that.
		draw_rect(r.grow(1.0), Color(0.02, 0.02, 0.03, 0.55 * a), true)
		draw_rect(Rect2(r.position, Vector2(r.size.x * f, r.size.y)),
			Color(0.86, 0.90, 0.96, 0.85 * a), true)
		# The playhead, wide enough to grab.
		var px: float = r.position.x + r.size.x * f
		draw_rect(Rect2(Vector2(px - 1.5, r.position.y - 4.0), Vector2(3.0, r.size.y + 8.0)),
			Color(1, 1, 1, 0.95 * a), true)
