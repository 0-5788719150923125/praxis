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
##
## A future mode needs no wiring at all if main already made the Chrome -
## call attach_feedback() when a session starts, detach_feedback() when it
## ends. New shared furniture belongs HERE, not in a mode's branch.

var exporter: Node
var assistant: Node
var feedback: Node


func _ready() -> void:
	exporter = preload("res://scripts/exporter.gd").new()
	add_child(exporter)
	assistant = preload("res://scripts/assistant.gd").new()
	add_child(assistant)


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
