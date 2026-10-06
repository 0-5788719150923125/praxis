extends SceneTree

## intro_blur_check - that the tarot intro's blur (`shaders/tarot_intro.gdshader`) is the blur it is
## asked for, in pixels: as wide as asked at any frame size, round, smooth and in place - and at the
## title screen's width ([constant TarotTable.TITLE_BLUR]) between the user's verdicts (2026-10-06):
## stronger than the lens's softest bokeh, which showed "too much table detail" ("an extremely strong
## blur, such that nothing is really visible in the scene except for the colors"), and softer than
## 0.0825 - itself 0.11, "just a smidge too blurry", less 25% - which was cut "another 50%".
##
##   tests/run_quiet.sh intro_blur_check
##
## NOT `--headless`: every claim is a measurement of a rendered frame.
##
## THE MEASURE IS AN EDGE. A black half beside a white half, blurred, rises as the Gaussian's own
## integral, so half the distance from its 16% to its 84% is the sigma drawn, and its 10% to 90% is
## how far an outline is spread. The shader reads the screen's mips - Godot builds each level with a
## Gaussian - and how much blur one level carries was MEASURED here (`level_sigma`): left at the
## first guess, 1.1, every blur came out 1.9 times as wide as asked.
##
## TWO-SIDED: the instrument reads the unblurred edge as sharp and the first guess's blur as too
## wide, and the title screen's band rejects both widths judged against it.

const SHADER := preload("res://shaders/tarot_intro.gdshader")
## A 10 cm thing on the table stands about this share of the frame's height (the set dresser is told
## so), and the title screen spreads an outline over between SPREAD.x and SPREAD.y times that. The
## widths judged against it: the lens's softest bokeh as a Gaussian (a sigma of 6.2 px whatever the
## frame, measured off an edge half a meter out at dof_blur_amount 0.16 - 0.0086 of a 720 frame), too
## sharp; 0.0825, too soft.
const THING := 1.0 / 6.0
const SPREAD := Vector2(0.3, 1.0)
const TOO_SHARP := 0.0086
const TOO_SOFT := 0.0825

var _fails := 0
var _checks := 0


func _init() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	_checks += 1
	if not cond:
		_fails += 1
		print("  FAIL: " + what)


func _run() -> void:
	for frame in [Vector2i(1280, 720), Vector2i(1920, 1080)]:
		var size: Vector2i = frame
		var edge := _edge(size, false)
		# THE INSTRUMENT: an unblurred edge reads as sharp
		var bare := await _render(edge, size, -1.0)
		_ok(_sigma(bare, size, false) < 1.0, "%s: the unblurred edge reads %.1f px wide - the instrument cannot see a sharp edge" % [size, _sigma(bare, size, false)])
		for w in [0.02, 0.05, TarotTable.TITLE_BLUR]:
			var want: float = w
			var img := await _render(edge, size, want)
			var got := _sigma(img, size, false) / size.y
			_ok(absf(got / want - 1.0) < 0.08, "%s: asked for %.3f of the height, drew %.4f" % [size, want, got])
			var mid := _crossing(_row(img, size.y / 2), 0.5)
			_ok(absf(mid - size.x * 0.5) < 1.5, "%s at %.3f: the edge moved, to %.1f from %d" % [size, want, mid, size.x / 2])
			var rough := _roughness(_row(img, size.y / 2))
			_ok(rough <= 3.0 / 255.0, "%s at %.3f: steps in the blur (a second difference of %.4f) - the mips' texels show" % [size, want, rough])
		# ROUND: across the frame as along it
		var across := await _render(_edge(size, true), size, 0.05)
		var along := await _render(edge, size, 0.05)
		var a := _sigma(across, size, true)
		var b := _sigma(along, size, false)
		_ok(absf(a / b - 1.0) < 0.05, "%s: %.1f px wide down the frame, %.1f across it" % [size, a, b])
		# THE CALIBRATION is what makes it right: the first guess draws far too wide
		var guess := await _render(edge, size, 0.05, 1.1)
		_ok(_sigma(guess, size, false) / size.y > 0.05 * 1.5, "%s: the uncalibrated level drew %.4f for 0.05 - the instrument cannot tell" % [size, _sigma(guess, size, false) / size.y])
	# THE TITLE SCREEN leaves only colors, and no more blur than that: an outline spread over between
	# SPREAD.x and SPREAD.y things' heights - and both widths judged against it fall outside
	var size := Vector2i(1280, 720)
	var edge := _edge(size, false)
	var things := _rise(await _render(edge, size, TarotTable.TITLE_BLUR), size) / size.y / THING
	_ok(things >= SPREAD.x and things <= SPREAD.y, "the title screen spreads an outline over %.2f things' heights, outside %.2f-%.2f" % [things, SPREAD.x, SPREAD.y])
	var sharp := _rise(await _render(edge, size, TOO_SHARP), size) / size.y / THING
	var soft := _rise(await _render(edge, size, TOO_SOFT), size) / size.y / THING
	_ok(sharp < SPREAD.x and soft > SPREAD.y, "the band does not separate the widths judged against it: %.2f at %.2f, %.2f at %.2f" % [sharp, TOO_SHARP, soft, TOO_SOFT])
	print("intro_blur_check: %s (%d checks, %d failure%s)" % ["ALL OK" if _fails == 0 else "FAILED", _checks, _fails, "" if _fails == 1 else "s"])
	quit(1 if _fails > 0 else 0)


## Black on one side, white on the other: down the frame's middle, or [param across] it.
func _edge(size: Vector2i, across: bool) -> ImageTexture:
	var img := Image.create(size.x, size.y, false, Image.FORMAT_RGBA8)
	img.fill(Color.BLACK)
	if across:
		img.fill_rect(Rect2i(0, size.y / 2, size.x, size.y - size.y / 2), Color.WHITE)
	else:
		img.fill_rect(Rect2i(size.x / 2, 0, size.x - size.x / 2, size.y), Color.WHITE)
	return ImageTexture.create_from_image(img)


## [param tex] drawn over a frame of [param size], then the blur over it at [param sigma] (a share of
## the height; below zero, no blur at all) and [param level_sigma] (the shader's own when below zero).
func _render(tex: ImageTexture, size: Vector2i, sigma: float, level_sigma := -1.0) -> Image:
	var vp := SubViewport.new()
	vp.size = size
	vp.disable_3d = true
	vp.transparent_bg = false
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	var tr := TextureRect.new()
	tr.texture = tex
	tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	tr.size = Vector2(size)
	vp.add_child(tr)
	if sigma >= 0.0:
		var blur := ColorRect.new()
		blur.size = Vector2(size)
		var mat := ShaderMaterial.new()
		mat.shader = SHADER
		mat.set_shader_parameter("sigma", sigma)
		if level_sigma >= 0.0:
			mat.set_shader_parameter("level_sigma", level_sigma)
		blur.material = mat
		vp.add_child(blur)
	root.add_child(vp)
	for i in 4:
		await process_frame
	var got := vp.get_texture().get_image()
	vp.queue_free()
	return got


## Row [param y] of [param img] - and below, a column - as values.
func _row(img: Image, y: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	for x in img.get_width():
		out.append(img.get_pixel(x, y).r)
	return out


func _column(img: Image, x: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	for y in img.get_height():
		out.append(img.get_pixel(x, y).r)
	return out


## Where [param v] first rises through [param level], to a fraction of a pixel (pixel centers).
static func _crossing(v: PackedFloat32Array, level: float) -> float:
	for i in range(1, v.size()):
		if v[i - 1] < level and v[i] >= level:
			return float(i - 1) + (level - v[i - 1]) / maxf(v[i] - v[i - 1], 1e-6) + 0.5
	return -1.0


## The sigma an edge was drawn with, in pixels: half its 16% to 84% rise.
func _sigma(img: Image, size: Vector2i, across: bool) -> float:
	var v := _column(img, size.x / 2) if across else _row(img, size.y / 2)
	return (_crossing(v, 0.8413) - _crossing(v, 0.1587)) * 0.5


## How far an edge is spread, in pixels: its 10% to 90% rise.
func _rise(img: Image, size: Vector2i) -> float:
	var v := _row(img, size.y / 2)
	return _crossing(v, 0.9) - _crossing(v, 0.1)


## The largest second difference along [param v]: a step or a kink where the blur should be smooth.
static func _roughness(v: PackedFloat32Array) -> float:
	var worst := 0.0
	for i in range(1, v.size() - 1):
		worst = maxf(worst, absf(v[i + 1] - 2.0 * v[i] + v[i - 1]))
	return worst
