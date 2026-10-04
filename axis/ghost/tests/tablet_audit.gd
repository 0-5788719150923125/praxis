extends RefCounted

## Where the word being spoken is, and where the camera looks, frame by frame - for
## tablet_look_probe's `--audit` and tablet_camera_check.
##
## A spoken word is checked against the browser's viewport (is it scrolled into view?) and the
## camera's frame (is it in the picture, clear of the edge?). A READING DRAG - the page moving
## under the reading, not a skim and not the hand's own action - is followed for REDIRECT_S, and
## the furthest the camera's aim moves in that time is kept: the camera answering a drag by
## swinging after the text is what "the zoomed-in camera has to pan much higher to compensate"
## looked like.

## How long after a reading drag the aim is watched: the camera's springs take seconds to answer.
const REDIRECT_S := 12.0

var frame_size: Vector2
var read_s := 0.0                    # seconds a word was being spoken on screen
var off_viewport_s := 0.0
var out_of_frame_s := 0.0
var edge_s := 0.0                    # in frame, but within 8% of its top or bottom
## Seconds the spoken word sat below the reading band's foot (more than 3% past it), and the
## lowest it was read, as a share of the viewport.
var low_s := 0.0
var lowest := 0.0
var lapses: Array = []
## The aim's furthest move, in logical pixels, within REDIRECT_S of each reading drag, and the
## show time of each drag.
var redirects: PackedFloat32Array = []
var drag_times: PackedFloat32Array = []
## How far the camera stood when each reading drag started, and the furthest it stood during
## the watch after it - a drag seen from wide, or one the page's arc widens out of, proves
## nothing about the close camera.
var drag_dist: PackedFloat32Array = []
var drag_far: PackedFloat32Array = []
## How far the aim moved, in logical pixels, while a page was being read.
var travel := 0.0
## Every picture a skim stops on, measured halfway through its hold: where its top, middle and
## bottom fall in the frame (0 its top edge, 1 its bottom). `[{t, page, bi, top, mid, bot}]`
var pictures: Array = []
## Each page as its first spoken word is heard: `page -> {scroll, vp_y, busy}` - whether a page
## opened where its reading is, and whether anything was still being replayed then.
var first_heard := {}
## When the screen first lit (-1 not yet), and when the first word was heard.
var woke_at := -1.0
var first_word_at := -1.0
var _pic_plan: Array = []            # the same, from the schedule, not yet reached
var _planned := -1

var _where := ""
var _last_scroll := -1.0
var _last_aim := -1.0
var _drag_t := -INF
var _drag_aim := 0.0
var _drag_dist := 0.0
var _drag_far := 0.0
var _excursion := 0.0


func _init(size: Vector2) -> void:
	frame_size = size


## Every spoken word of [param body] at a steady [param word_s] seconds, a rest at each sentence
## end, and each run of actions' rest after the word before it - the shape a real take's sidecar
## has. [param intro] is the silence before all of it - the opening run's rest comes after it,
## whole, as the panel has it. [param from_si] > 0 is a SCRUB: the voice starts again at that
## spoken word, with no intro, as `GenerativeEditor._speak_from` does.
static func timeline(body: String, word_s: float, intro: float, from_si := 0) -> Array:
	var d := TabletScript.parse(body)
	var holds := {}
	for a in d["actions"]:
		var n := int(a["after"])
		holds[n] = float(holds.get(n, 0.0)) + float(a["dur"])
	var out: Array = []
	var t := maxf(0.0, intro) if from_si <= 0 else 0.3
	var si := 0
	var spoken: PackedInt32Array = d["spoken"]
	for k in range(maxi(0, from_si), spoken.size()):
		t += float(holds.get(k, 0.0))
		var text := String((d["words"][spoken[k]] as Dictionary)["text"])
		out.append({"text": text, "t0": t, "t1": t + word_s * 0.85, "sentence": si, "emph": 0})
		t += word_s
		var tail := text.rstrip("\"')*_")
		if tail.ends_with(".") or tail.ends_with("?") or tail.ends_with("!"):
			si += 1
			t += 0.45
	return out


## Where a scrub to the first run of [param phrase] restarts the reading: its spoken index, and
## the start words `GenerativeEditor._start_words` would hand the media. `{si, words}`, si -1
## when the phrase is not read.
static func scrub_to(body: String, phrase: String) -> Dictionary:
	var d := TabletScript.parse(body)
	var norms := PackedStringArray()
	for wi in (d["spoken"] as PackedInt32Array):
		norms.append(String((d["words"][wi] as Dictionary)["norm"]))
	var want := PackedStringArray()
	for w in phrase.split(" ", false):
		want.append(TabletScript.norm(w))
	var si := TabletScript.find_run(norms, want)
	return {"si": si, "words": norms.slice(si, si + 6) if si >= 0 else PackedStringArray()}


## Measure one frame of [param tab]. The row `{wi, vp_y, fy, aim_y, scroll, busy}`, or `{}` when
## nothing on screen is being read.
func frame(tab: TabletMedium) -> Dictionary:
	_measure_pictures(tab)
	if woke_at < 0.0 and float(tab._st.get("on", 0.0)) > 0.001:
		woke_at = tab._now
	var r: Dictionary = tab._reading()
	var page := tab._current_page()
	if r.is_empty() or page < 0:
		return {}
	var wi := int(r["wi"])
	var dw: Array = tab._doc["words"]
	var o := int(tab._st.get("orient", 0))
	var turn := float(tab._st.get("turn", 0.0))
	if int((dw[wi] as Dictionary)["page"]) != page or (turn > 0.001 and turn < 0.999):
		return {}
	var lp := tab.layout(page, o)
	if not lp.word_rect.has(wi):
		return {}
	var L := TabletMedium.logical(o)
	var vh := L.y - TabletMedium.TOP
	var scroll := tab.scroll_of(page, tab._now, o)
	var wr: Rect2 = lp.word_rect[wi]
	var y0 := wr.position.y - scroll + TabletMedium.TOP
	var busy := bool(tab._st.get("busy", false))
	var cam := tab._cam
	var c := tab._world_of(Vector2(wr.get_center().x, y0 + wr.size.y * 0.5))
	var sp := cam.unproject_position(c)
	var fy := sp.y / frame_size.y
	var fx := sp.x / frame_size.x
	var aim_y := _aim_y(tab, o)
	var row := {"wi": wi, "vp_y": (y0 - TabletMedium.TOP) / vh, "fy": fy, "aim_y": aim_y,
		"scroll": scroll, "busy": busy}
	var si0 := int(r["si"])
	if not first_heard.has(page) and tab._st0[si0] != TabletMedium.NO_TIME and tab._now >= tab._st0[si0]:
		first_heard[page] = {"scroll": scroll, "vp_y": row["vp_y"], "busy": busy}
		if first_word_at < 0.0:
			first_word_at = tab._now
	var where := "%d|%d" % [page, o]
	if busy or where != _where:
		_where = where
		_last_aim = -1.0
		_last_scroll = -1.0
		return row
	var moving := _last_scroll >= 0.0 and absf(scroll - _last_scroll) > 0.5
	_last_scroll = scroll
	if moving and not bool(tab._st.get("skim", false)) and tab._now - _drag_t > 3.0:
		finish()
		_drag_t = tab._now
		_drag_aim = aim_y
		_drag_dist = tab._c_dist
		_drag_far = tab._c_dist
	if tab._now - _drag_t < REDIRECT_S and aim_y >= 0.0:
		_excursion = maxf(_excursion, absf(aim_y - _drag_aim))
		_drag_far = maxf(_drag_far, tab._c_dist)
	if aim_y >= 0.0 and _last_aim >= 0.0:
		travel += absf(aim_y - _last_aim)
	_last_aim = aim_y
	return row


## Count [param dt] seconds against the word in [param row], if it is being SPOKEN: before the
## first word, and in the rests, the reading points at a word that is not sounding.
func count(row: Dictionary, tab: TabletMedium, dt: float) -> void:
	if row.is_empty() or bool(row["busy"]):
		return
	var si := int(tab._reading().get("si", -1))
	if si < 0 or tab._st0[si] < 0.0 or tab._now < tab._st0[si] or tab._now > tab._st1[si] + 0.35:
		return
	read_s += dt
	var vp := float(row["vp_y"])
	lowest = maxf(lowest, vp)
	if vp > TabletMedium.READ_BAND.y + 0.03:
		low_s += dt
	var fy := float(row["fy"])
	var word := String((tab._doc["words"][int(row["wi"])] as Dictionary)["text"])
	var off_vp := vp < -0.004 or vp > 0.985
	var off_cam := fy < 0.0 or fy > 1.0
	if off_vp:
		off_viewport_s += dt
	if off_cam:
		out_of_frame_s += dt
	elif fy < 0.08 or fy > 0.92:
		edge_s += dt
	if (off_vp or off_cam) and (lapses.is_empty() or not String(lapses.back()).begins_with(word + "@")):
		lapses.append("%s@%.1f viewport %s frame %s (vp_y %.2f, frame_y %.2f)" % [word, tab._now,
			"OFF" if off_vp else "ok", "OUT" if off_cam else "in", vp, fy])


## The skims' picture stops, from the schedule (re-read whenever it grows), and each measured
## when the show reaches the middle of its hold.
func _measure_pictures(tab: TabletMedium) -> void:
	if tab._sched.size() != _planned:
		_planned = tab._sched.size()
		_pic_plan = []
		for e in tab._sched:
			var a: Dictionary = e["a"]
			if String(a["kind"]) != "skim" or not a.has("word"):
				continue
			var ph := TabletScript.phases("skim", "", int(a.get("n", 0)), int(a.get("m", 0)))
			var pics: Array = a.get("pics", [])
			for k in pics.size():
				var tm := float(e["t0"]) + float(ph["scroll0"]) * float(e["s"]) \
					+ TabletScript.PICTURE_DWELL * k + TabletScript.PICTURE_DRAG + TabletScript.PICTURE_HOLD * 0.5
				if tm > tab._now:
					_pic_plan.append({"t": tm, "page": int(a["from"]), "bi": int(pics[k])})
		_pic_plan.sort_custom(func(x, y): return float(x["t"]) < float(y["t"]))
	while not _pic_plan.is_empty() and tab._now >= float(_pic_plan[0]["t"]):
		var pp: Dictionary = _pic_plan.pop_front()
		var page := int(pp["page"])
		if tab._current_page() != page:
			continue
		var o := int(tab._st.get("orient", 0))
		var rect: Rect2 = tab.layout(page, o).block_rect.get(int(pp["bi"]), Rect2())
		if rect.size.y <= 0.0:
			continue
		var dy := TabletMedium.TOP - tab.scroll_of(page, tab._now, o)
		var x := rect.get_center().x
		var f := func(y: float) -> float:
			return tab._cam.unproject_position(tab._world_of(Vector2(x, y + dy))).y / frame_size.y
		pictures.append({"t": tab._now, "page": page, "bi": int(pp["bi"]), "top": f.call(rect.position.y),
			"mid": f.call(rect.get_center().y), "bot": f.call(rect.end.y)})


## Close the reading drag being followed, if any.
func finish() -> void:
	if _drag_t > -INF:
		redirects.append(_excursion)
		drag_times.append(_drag_t)
		drag_dist.append(_drag_dist)
		drag_far.append(_drag_far)
	_drag_t = -INF
	_excursion = 0.0


func worst_redirect() -> float:
	var top := 0.0
	for x in redirects:
		top = maxf(top, x)
	return top


func summary() -> String:
	var mean := 0.0
	for x in redirects:
		mean += x / float(redirects.size())
	var out := "%.0fs of reading; the spoken word below the reading band %.1fs (lowest %.2f of the viewport), off the viewport %.1fs, out of frame %.1fs, at the frame's edge %.1fs; %d reading drags, the camera's aim moving within %.0f s of each: mean %.0f px, max %.0f px; aim travel while reading %.0f px" % [
		read_s, low_s, lowest, off_viewport_s, out_of_frame_s, edge_s, redirects.size(), REDIRECT_S, mean,
		worst_redirect(), travel]
	for i in redirects.size():
		out += "\n  drag @%.1f (camera at %.2f, furthest %.2f): aim moved %.0f px" % [drag_times[i],
			drag_dist[i], drag_far[i], redirects[i]]
	for pc in pictures:
		out += "\n  picture @%.1f (page %d): top %.2f, middle %.2f, bottom %.2f of the frame" % [
			float(pc["t"]), int(pc["page"]), float(pc["top"]), float(pc["mid"]), float(pc["bot"])]
	for l in lapses.slice(0, 40):
		out += "\n  lapse " + String(l)
	return out


## Where the frame's center falls on the page, in logical y, or -1.
static func _aim_y(tab: TabletMedium, o: int) -> float:
	var cam := tab._cam
	var hit = Plane(Vector3.UP, TabletMedium.SLAB_T * 0.5).intersects_ray(cam.global_position,
		-cam.global_transform.basis.z)
	if hit == null:
		return -1.0
	var tp := Vector2(((hit as Vector3).x / TabletMedium.SCREEN.x + 0.5) * TabletMedium.SW,
		((hit as Vector3).z / TabletMedium.SCREEN.y + 0.5) * TabletMedium.SH)
	return (TabletMedium.content_xf(o).affine_inverse() * tp).y
