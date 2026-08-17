extends Node
class_name Chrome

## Chrome - the shared session furniture every mode of ghost carries.
##
## The design lesson made explicit: modes were assembling their overlay stack
## BY HAND in per-branch code (an exporter here, an assistant there, the
## feedback console only in one path), so every new mode forgot a piece -
## synthesis shipped without ` feedback and without the export button, twice.
## Ghost's own rule - composition over hand-assembly - now applies to the app
## furniture too: main creates ONE Chrome, and any mode gets the standard set:
##
## - **exporter**  - the ⤓ render-to-video button + background pipeline
##                   (persistent: an in-flight export survives session churn).
## - **assistant** - the feedback browser / dispatch backend (persistent:
##                   queued work survives sessions; dispatch itself is gated
##                   on the splash's backend setting, see assistant.gd).
## - **feedback**  - the ` console, created on demand per session via
##                   [method attach_feedback] and wired to the assistant.
## - **console**   - the `>_` log viewer (a live tail of godot's own log
##                   file), for anyone running ghost without a terminal.
## - **scrubber**  - the seek bar, revealed by pointing near the bottom of the
##                   frame; arrow keys step it without aiming. Hidden entirely
##                   when the audio cannot be seeked (a synthesis generator).

var exporter: Node
var assistant: Node
var feedback: Node
var console: Node
var scrubber: Node

## HOW MUCH ROOM THE CURRENT MODE HAS CLAIMED at the bottom of the frame, in
## pixels. Every piece of furniture here anchors bottom-right, which is empty in
## most modes - but the Masking editor puts its marker strip and its trim/track
## lanes down there, so all three toggles (⤓ export, 💬 assistant, >_ console) sat
## ON the timeline. A mode sets this once and the whole row steps up together;
## patching one button's offsets would only have moved one of the three, and the
## next piece of furniture added here would have the same bug again.
var bottom_inset := 0.0:
	set(value):
		bottom_inset = value
		_apply_inset()


func _apply_inset() -> void:
	for n in [exporter, assistant, console]:
		if n != null and n.has_method("set_bottom_inset"):
			n.set_bottom_inset(bottom_inset)


func _ready() -> void:
	# Findable by a mode without knowing where in the tree it was parented - it is
	# main's child today, and a mode looking for it walked two levels and missed it.
	add_to_group("ghost_chrome")
	exporter = preload("res://scripts/exporter.gd").new()
	add_child(exporter)
	assistant = preload("res://scripts/assistant.gd").new()
	add_child(assistant)
	console = preload("res://scripts/console.gd").new()
	add_child(console)
	# The seek bar. Furniture rather than a mode's own control precisely because the
	# lesson at the top of this file is that anything a mode has to remember to add is a
	# thing some mode will forget - and reviewing a long take is not specific to any one
	# of them. It hides itself when the session cannot be seeked (see Spectrum.seekable).
	scrubber = preload("res://scripts/scrubber.gd").new()
	add_child(scrubber)
	# Persistent like the rest of the furniture, and deliberately NOT gated on a
	# live session: the value it edits is a saved preference the Director reads at
	# startup, so setting it on the home screen already governs the first hold of
	# the next song.


## The ` feedback console for the current session. Idempotent: returns the
## live console if one is already attached. Wired to the assistant so a
## submitted critique dispatches (when a backend is enabled).
func attach_feedback() -> Node:
	if feedback != null and is_instance_valid(feedback):
		return feedback
	feedback = preload("res://scripts/feedback.gd").new()
	add_child(feedback)
	if assistant != null and is_instance_valid(assistant):
		feedback.submitted.connect(assistant.enqueue)
	return feedback


## Tear down the per-session console (the persistent pieces stay). Callers own
## the don't-yank-it-while-open courtesy (see main._end_session).
func detach_feedback() -> void:
	if feedback != null and is_instance_valid(feedback):
		feedback.queue_free()
	feedback = null
