extends GhostScene
class_name FilmScene

## FilmScene - a panel of real footage, cut into the comic among the drawn ones.
##
## It is a [GhostScene] like any other, which is the whole trick: the comic vehicle casts
## it into a panel exactly as it casts a spectrum ring, and every piece of machinery
## around it - the render target, the freeze that holds a panel as a still, the camera
## that flies over the page - carries on knowing nothing about video. The only thing that
## makes it different from a drawn panel is where its pixels come from.
##
## IT PLAYS A WINDOW, NOT A FILM. The clip is never transcoded whole (see [constant
## Films.WINDOW] for why that was intolerable); what exists on disk is
## [constant Films.WINDOW] seconds of it at a time, and this plays whichever window covers
## the position the show clock asks for. That makes the panel's job slightly bigger than
## "play a file": it has to know where it is in the FILM while playing a file that starts
## at zero, and it has to cross from one window to the next without a gap.
##
## THE CROSSING IS WHY THERE ARE TWO PLAYERS. A window is loaded, played, and when the
## playhead nears its end the next one is opened alongside, primed, and swapped to. The
## windows overlap by [constant Films.LEAD] seconds precisely so that swap has slack: the
## outgoing player still has frames while the incoming one settles. One player and a
## reload would show the panel's paper for as long as the load took, every window, forever.
##
## THE SEEK, ONCE, AT THE TOP. The window is placed when it opens and then simply plays -
## it is NOT re-seeked per frame, which is Masking's scrubbing mode and costs a decoder
## flush every time. What it does instead is watch for DRIFT: a panel that was frozen
## while the camera looked elsewhere has a stopped clock while the show's kept running.
##
## NOT IN THE DIRECTOR'S CATALOGUE. This scene is never minted by [Director.mint_scene]:
## there is nothing to show unless the viewer has imported something, and a scene that is
## sometimes not there at all would make the running order depend on the library. The
## comic casts it directly. See [ComicVehicle._film_at].

## How far the clip may drift from where the show says it should be, in seconds, before it
## is pulled back. Below about this a correction is more visible than the error - a seek
## lands on a keyframe and jumps, where the drift itself is a smooth few frames of lag.
const DRIFT_TOLERANCE := 0.35

## Seconds of hold before the first drift correction. VideoStreamPlayer reports a
## stream_position of 0 for a frame or two after `play()`, and correcting against that
## reading would seek it straight back to where it was told not to start.
const SETTLE := 0.5

## How close to the end of a window the playhead gets before the next one is opened
## alongside. Comfortably inside [constant Films.LEAD], so the swap happens while the
## outgoing window still has overlap left to play.
const HANDOVER := 3.0

var clip: Dictionary = {}
## The show clock this panel opened at, and the clock it has been handed since. It is
## passed in rather than read from Spectrum so an export render - which drives its own
## deterministic clock - places the footage exactly where the live session did.
var _show_t := 0.0
var _since_open := 0.0
## The window each player is playing, and the players themselves. `_next` is null except
## across a handover.
var _win := -1
var _next_win := -1
var _player: VideoStreamPlayer = null
var _next: VideoStreamPlayer = null
var _tint := Color(1, 1, 1, 1)


## Give this scene its footage. Called before the scene enters the tree, because the
## first window is opened in _ready and there is nothing to open without a clip.
func set_clip(c: Dictionary, show_time: float) -> void:
	clip = c
	_show_t = show_time


func build_params(_rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	# The page's ink is warm and the paper is not white; footage dropped in raw reads as a
	# television someone left on in a drawing. A light warm multiply is enough to seat it.
	_tint = Color(1.0, 0.985, 0.955, 1.0)
	return {}


func _ready() -> void:
	super._ready()
	if clip.is_empty():
		return
	_open(Films.window_index(Films.position_at(clip, _show_t)))


## Where the show says this panel should be, in seconds into the FILM.
func film_position() -> float:
	return Films.position_at(clip, _show_t)


## Build a player for one window and start it at the right place inside it.
func _open(index: int) -> void:
	if not Films.window_ready(clip, index):
		return
	_win = index
	_player = _make_player(index)
	# ASK FOR THE NEXT WINDOW NOW, not when the boundary is in sight. Cutting one runs at
	# about twice realtime on hard footage, so waiting until a few seconds before the
	# handover would be asking for twenty seconds of work in three. It is not speculative
	# either: the show clock only moves forward, so the window after this one is the one
	# the NEXT film panel will want even if this panel never reaches it.
	Films.request_window(clip, Films.next_window(clip, index))
	# The seek is deferred: VideoStreamPlayer refuses a stream_position set on the same
	# frame it starts playing - the same rule mask_editor found for its PiP tracks.
	call_deferred("_seek_current")


func _make_player(index: int) -> VideoStreamPlayer:
	var p := VideoStreamPlayer.new()
	p.stream = load(ProjectSettings.globalize_path(Films.window_path(clip, index)))
	p.loop = false            # a window is a slice of a film, not the film; see _advance
	p.volume = 0.0            # the show has a soundtrack; a panel does not
	p.expand = true
	# Zero-sized and invisible: the player is a DECODER here, never a widget. What is drawn
	# is its texture, in _draw, through the scene's own view transform - handing the layout
	# to a Control child would put it outside begin_draw and outside the page.
	p.size = Vector2.ZERO
	p.visible = false
	add_child(p)
	p.play()
	return p


func _seek_current() -> void:
	if _player == null or not is_instance_valid(_player) or not _player.is_playing():
		return
	_player.stream_position = Films.window_local(film_position())


func update(f: AudioFeatures, delta: float) -> void:
	_show_t += delta
	_since_open += delta
	# Somebody with a frame has to notice a finished cut, and while a panel is live this is
	# the one that cares. See Films.pump.
	Films.pump()
	_advance()
	# The audio still reaches the picture, faintly: the footage is in the show, not beside
	# it. A little breathing on the tint rather than a shake - the panel is a photograph
	# among drawings and the joke stops working if it starts dancing.
	if f != null:
		var e := clampf(float(f.energy), 0.0, 1.0)
		_tint = Color(1.0, 0.985, 0.955).lerp(Color(1.0, 1.0, 1.0), e * 0.6)
	queue_redraw()


## Keep the right window playing at the right place. Three jobs, in the order they matter:
## open a window at all if the panel started without one, hand over at a boundary, and
## correct drift within a window.
func _advance() -> void:
	var pos := film_position()
	var want := Films.window_index(pos)
	# The clip loops, so the window after the last is the first.
	if not Films.window_exists(clip, want):
		want = 0
		pos = 0.0
	# STARTED COLD. The comic only casts a film panel when the window is ready, so this is
	# the rarer case of a panel outliving its own clip's cache; ask, and open when it lands.
	if _player == null or not is_instance_valid(_player):
		Films.request_window(clip, want)
		if Films.window_ready(clip, want):
			_since_open = 0.0
			_open(want)
		return
	# THE HANDOVER. Open the next window alongside once the playhead is inside the overlap,
	# then swap to it when the position actually belongs to it.
	if want != _win:
		if _next != null and is_instance_valid(_next) and _next_win == want:
			_swap()
		else:
			_prime(want)          # the handover was missed - open it late rather than never
			# No next window on disk yet: hold on the overlap the outgoing window carries
			# rather than showing paper. It runs out after Films.LEAD, which is the whole
			# reason the windows overlap at all.
			return
	elif Films.window_local(pos) > Films.WINDOW - HANDOVER:
		# The cut was started when this window opened; this is only the DECODER being
		# spun up, which takes a frame or two rather than twenty seconds.
		_prime(Films.next_window(clip, _win))
	if _since_open > SETTLE:
		# WHERE IT SHOULD BE against where it is. The gap opens whenever this panel was
		# frozen - the show's clock ran on, the decoder's did not - and closing it is what
		# keeps the footage reading as one film running behind the page.
		var local := Films.window_local(pos)
		if absf(local - float(_player.stream_position)) > DRIFT_TOLERANCE:
			_player.stream_position = local


## Have the next window decoding alongside, ready to be swapped to.
func _prime(index: int) -> void:
	if _next != null and is_instance_valid(_next) and _next_win == index:
		return
	if not Films.request_window(clip, index):
		return
	if _next != null and is_instance_valid(_next):
		_next.queue_free()
	_next_win = index
	_next = _make_player(index)


func _swap() -> void:
	if _player != null and is_instance_valid(_player):
		_player.queue_free()
	_player = _next
	_win = _next_win
	_next = null
	_next_win = -1
	_since_open = 0.0
	call_deferred("_seek_current")


## Never exits on its own. The Director's clock cuts away from it like any other panel.
func finished() -> bool:
	return false


## THE PANEL RECTANGLE, AND NOTHING ELSE.
##
## No [method GhostScene.begin_draw] here, and NOT [method GhostScene.view_half_px] - which
## is the bug this replaces. `view_half_px` is a deliberate OVERDRAW bound: it carries a
## 1.06 margin, it is measured about the origin so any pan inflates it, and it grows again
## under rotation or skew. That is exactly right for a generated scene, which must cover the
## frame however the camera moves and has no edge of its own to protect. Footage is the
## opposite: it HAS edges, and every pixel of overdraw is a pixel cropped off them.
##
## Measured with a fixture painted a different colour on each edge: at a 2.4-aspect panel
## ALL FOUR coloured edges were gone, and in the app the Director's shot bias slid the crop
## to one side as well - "videos that are cropped on ALL edges except at the bottom".
##
## The film IS the panel, so it is drawn in the panel's own coordinates: the viewport rect,
## top-left origin, no camera transform. The comic's real camera is already flying over the
## page; a second one inside the panel only ever takes picture away.
func _draw() -> void:
	var panel := size
	var tex: Texture2D = null
	if _player != null and is_instance_valid(_player):
		tex = _player.get_video_texture()
	if tex == null or tex.get_size().x <= 0.0:
		# A frame or two before the first decode, and forever if the file went bad. Paper,
		# not black: an empty panel that matches the page reads as a beat, and a black
		# rectangle reads as a bug.
		draw_rect(Rect2(Vector2.ZERO, panel), Color(0.88, 0.865, 0.83), true)
		return
	# COVER, on whichever axis needs the least. A panel's aspect is the page's business and
	# the clip's is the camera's; letterboxing one inside the other would put black bars
	# inside a comic frame, which is the one thing a comic frame never has. So the footage
	# fills the panel and overflows on exactly ONE axis, symmetrically, by the smallest
	# amount that covers - which is what `max` of the two ratios means when both are
	# measured against the true rectangle rather than an inflated one.
	var src := tex.get_size()
	var k := maxf(panel.x / src.x, panel.y / src.y)
	var dst := src * k
	draw_texture_rect(tex, Rect2((panel - dst) * 0.5, dst), false, _tint)
