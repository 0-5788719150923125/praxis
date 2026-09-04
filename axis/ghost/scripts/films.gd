extends RefCounted
class_name Films

## Films - the library of imported video clips a comic panel can be filled with.
##
## A panel of the [ComicVehicle] normally holds a live [GhostScene]. Some of them can
## instead hold a piece of REAL FOOTAGE, which is the one thing a generated scene can
## never be, and a comic that cuts between drawings and a photograph is doing something
## comics have always done.
##
## This owns the LIBRARY, not the playback: the list of clips, their import, and the two
## dials over them (which clips exist, and how often a page reaches for one). The playing
## is [FilmScene]'s, and the deciding is the comic vehicle's.
##
## THE CLOCK IS VIRTUAL, and that is the one design decision here worth stating. The
## requirement was that a clip must not restart from the beginning every time it is
## sampled - it should read as one film that has been running all along, which the show
## occasionally cuts into. The obvious way to get that is to keep every clip playing in
## the background and sample whatever it happens to be showing. This does the same thing
## with no background decoding at all: [method position_at] answers "where would this
## clip be if it had been looping since the show started", and a panel simply seeks
## there when it opens. A clip that is not on screen costs nothing, and the answer is a
## pure function of the show clock - so an export renders the same frames as the live
## session, which a real background player could not promise.
##
## ONE AT A TIME. Two panels sampling one clip at the same instant would show the same
## picture twice, because the position is a function of time and nothing else. Rather
## than de-correlating them with per-panel offsets - which is a second mechanism to
## explain and to get wrong - the comic simply never runs more than one film panel on a
## page. See [member ComicVehicle._film_at].

## Where a clip's prepared WINDOWS live - one subdirectory per clip. A clip belongs to the
## library, not to any session, and survives every one of them.
const DIR := "user://films"

## IMPORT PREPARES NOTHING, AND THAT IS THE POINT.
##
## Ogg Theora is the only format [VideoStreamPlayer] decodes natively, so a clip has to be
## transcoded before it can be played - and the first version of this did what Masking's
## `_prep` does, which is transcode the WHOLE FILE at import. Reported, correctly: "for
## every clip I import, I have to sit there and wait for it to prepare the entire video for
## tens of minutes at a time." Measured on ten minutes of 1080p, that is exactly right, and
## libtheora is single-threaded so it does not get better on a bigger machine.
##
## It is also almost entirely WASTED WORK. A film panel is on screen for seconds at a time,
## and where in the clip it starts is decided by the show clock ([method position_at]) - so
## a session touches a few minutes of a two-hour film and never looks at the rest. What is
## prepared here is therefore a WINDOW: [constant WINDOW] seconds around the position
## actually wanted, extracted on demand and cached. An import is a probe and nothing else,
## so it is instant, and the first window arrives in about as long as it takes to read this
## sentence.
##
## THE COST OF THE TRADE, stated plainly: the original file is now a dependency rather than
## something consumed once. Move or delete it and the clip stops working - [method clips]
## drops entries whose source has gone, rather than keeping a row that plays nothing.
const WINDOW := 15.0
## Overlap carried past the end of each window, so a panel that reaches the boundary has
## frames to play while the next window is loading, and a seek that lands near the end has
## somewhere to land.
const LEAD := 4.0
## Longest edge of a prepared window, in pixels, and the theora quality.
##
## THIS IS SET BY THE PANEL, NOT BY THE ENCODER'S CONVENIENCE. The first version chose 640px
## because it made windows cheap, and the footage came back "very pixelated... super blurry"
## - correctly: a comic panel's render target is the stage's short side
## (`ComicVehicle._size_targets`), which is 1080 on a 1080p window, and the camera then fills
## the frame with that panel. 640 into 1080 is a 1.7x upscale of an already-compressed
## picture, and every one of those pixels is on screen at the comic's reading distance.
##
## So: DO NOT DOWNSCALE. The cap only exists to stop a 4K source being re-encoded at 4K for
## a 1080 panel. Timed on a real 1920x1080 clip: no downscale at q6 runs at 0.43x realtime,
## 1440 at 0.91x, and the sharpness difference between those two is visible at this framing.
## Slower than realtime is affordable ONLY because windows are cut in parallel - see
## PIPELINE, which is the constant that pays for this one.
const MAX_EDGE := 1920
const Q := "6"
## How many windows of one clip are kept on disk before the oldest are dropped. Worth more
## than it looks: the show clock is the AUDIO clock, so replaying the same song asks for the
## same windows, and a second viewing of a reading is cut entirely from cache. At this size a
## window is a few megabytes, so this is a few hundred for a clip watched right through.
const CACHE_WINDOWS := 48
## HOW MANY WINDOWS ARE CUT AT ONCE, and the arithmetic that makes the whole thing work.
##
## The window the show wants advances at exactly 1x realtime, because the position IS the
## clock. One cut runs at about 0.43x on a 1080p source at full resolution - slower than the
## thing it is chasing, so a single cut at a time can never keep up, whatever the window
## size. Four in flight covers 4 x 15s of film in the ~35s one of them takes, which is about
## 1.7x, and that margin is what lets the quality above be what it is. libtheora is
## single-threaded, so these are four cores rather than four threads of one - and they only
## run while film is actually being used.
const PIPELINE := 4
## `-g 25` keeps a keyframe about every second, which is what makes the seek in [FilmScene]
## land quickly.
const GOP := "25"

## The frequency dial's range. It is a PROBABILITY PER PAGE, not per panel: with only one
## film panel allowed at a time, per-page is the honest unit - at 1.0 every page has one,
## at 0 none ever do.
##
## IT DOES NOT COMPETE WITH THE SCENE CATALOGUE. A film is not a 72nd entry in
## `Director.SCENES` drawn against the other 71 - it is an independent draw made when the
## page turns, and only then does the rest of the page go to the Director. That
## distinction is the whole reason a dial can control this at all.
##
## The default is a first impression, not a taste. Measured over 400 pages (which average
## 3.3 panels): 0.3 puts film on 27% of pages and 8% of panels, which is a film panel
## roughly every dozen cuts - long enough that someone who has just imported a clip can
## watch for a minute, see nothing, and conclude it is broken. 0.5 lands on 46% of pages
## and 14% of panels, about every seventh cut, and the comic is still overwhelmingly drawn.
const FREQ_MIN := 0.0
const FREQ_MAX := 1.0
const FREQ_DEFAULT := 0.5

## Anything shorter than this is not a film, it is a glitch - and a clip shorter than a
## panel's time on screen would visibly restart while being read.
const MIN_DURATION := 1.0


## A library and a frequency supplied by a PROBE, standing in for the stored ones.
##
## It exists because the library lives in [Settings], which a test probe is forbidden to
## write - deliberately, since a gate that edits the config of whoever runs it is a gate
## that costs them their settings (see Settings.allow_writes_for_test). A look probe still
## has to be able to put footage on a page, so this is the seam, and it is the only one:
## nothing else here reads it, and an app that never calls it behaves exactly as if it did
## not exist.
## `_test_active` is a separate flag rather than "is the list non-empty", because AN EMPTY
## LIBRARY IS A STATE WORTH TESTING - it is the one every viewer starts in - and inferring
## the seam from emptiness made "no clips" silently mean "use the real ones", which is both
## the wrong answer and a gate reading the author's own library.
static var _test_active := false
static var _test_clips: Array = []
static var _test_frequency := -1.0


static func use_for_test(list: Array, freq := 1.0) -> void:
	_test_active = true
	_test_clips = list
	_test_frequency = freq


## Every imported clip: `[{source, name, duration, slug}]`, in import order.
##
## Clips whose SOURCE has gone missing are dropped on read rather than hidden: the library
## is a list of things that can actually be played, and since windows are cut from the
## original on demand, a moved or deleted file is a clip that would be picked and then draw
## nothing. That is the standing cost of not transcoding up front - see [constant WINDOW].
static func clips() -> Array:
	if _test_active:
		return _test_clips
	var out: Array = []
	for row in (Settings.read("films", "clips", []) as Array):
		if not (row is Dictionary):
			continue
		var c: Dictionary = row
		if FileAccess.file_exists(String(c.get("source", ""))):
			out.append(c)
	return out


static func _store(list: Array) -> void:
	Settings.write("films", "clips", list)


## Forget a clip AND delete the windows cut from it. They live under a directory that
## exists only for this entry, so leaving them behind would be files nothing can reach
## again. The ORIGINAL is never touched - it is the viewer's own file, and this only ever
## borrowed from it.
static func remove(i: int) -> void:
	var list := clips()
	if i < 0 or i >= list.size():
		return
	var c: Dictionary = list[i]
	_clear_windows(c)
	var dir := ProjectSettings.globalize_path(clip_dir(c))
	if DirAccess.dir_exists_absolute(dir):
		DirAccess.remove_absolute(dir)
	list.remove_at(i)
	_store(list)


## How often a page reaches for a clip, 0..1. Zero, or an empty library, means the comic
## behaves exactly as it did before any of this existed.
static func frequency() -> float:
	if _test_frequency >= 0.0:
		return _test_frequency
	return clampf(float(Settings.read("films", "frequency", FREQ_DEFAULT)), FREQ_MIN, FREQ_MAX)


static func set_frequency(v: float) -> void:
	Settings.write("films", "frequency", clampf(v, FREQ_MIN, FREQ_MAX))


## WHERE THE CLIP IS NOW, in seconds - as if it had been looping since time zero.
##
## The show clock is the argument rather than something read in here, so the caller
## decides which clock it is: the live session and an export render pass the same one
## they pass everything else, and the answer is reproducible in both.
static func position_at(clip: Dictionary, show_time: float) -> float:
	var dur := float(clip.get("duration", 0.0))
	if dur <= MIN_DURATION:
		return 0.0
	# Guard the negative: a bookend hold runs the clock below zero at the head of a
	# render, and fmod would hand back a negative seek that VideoStreamPlayer refuses.
	return fposmod(show_time, dur)


## The clip to use for a page, or `{}` when the library is empty. Seeded, so a session
## replays the same footage in the same panels - the whole show is reproducible from one
## seed and this may not be the exception.
static func pick(seed_value: int) -> Dictionary:
	var list := clips()
	if list.is_empty():
		return {}
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value
	return list[rng.randi() % list.size()]


# --- windows ------------------------------------------------------------------
#
# A window is [constant WINDOW] seconds of a clip, cut from the original on demand and
# cached. Windows tile the clip from zero, so which one covers a position is arithmetic
# and needs no bookkeeping - two panels asking for the same moment ask for the same file.

## Where this clip's windows live.
static func clip_dir(clip: Dictionary) -> String:
	return DIR.path_join(String(clip.get("slug", "clip")))


## Which window covers `pos` (seconds into the clip).
static func window_index(pos: float) -> int:
	return int(floor(maxf(pos, 0.0) / WINDOW))


## Where `pos` sits INSIDE its window. This is what [FilmScene] seeks the player to; the
## window's own timeline starts at zero however far into the film it was cut from.
static func window_local(pos: float) -> float:
	return maxf(pos, 0.0) - float(window_index(pos)) * WINDOW


## THE WINDOW LENGTH IS IN THE NAME, and that is not decoration. Windows tile the clip from
## zero, so window 3 means "45s to 64s" only for as long as WINDOW is 15 - change the
## constant and every cached file silently becomes footage of the wrong minute, played with
## complete confidence. Caught by film_clock_check when WINDOW went from 45 to 15: the panel
## drew 52s while the decoder, the clock and the arithmetic all agreed it was at 23s.
## Naming them this way turns that into a cache MISS, which costs one re-cut and nothing else.
static func window_path(clip: Dictionary, index: int) -> String:
	return clip_dir(clip).path_join("w%04d_%ds.ogv" % [maxi(index, 0), int(WINDOW)])


## Is this window cut and ready to play?
static func window_ready(clip: Dictionary, index: int) -> bool:
	return FileAccess.file_exists(ProjectSettings.globalize_path(window_path(clip, index)))


## Does this window exist at all, or is the clip simply not that long? A clip shorter than
## one window has exactly window 0, and asking for window 3 of it must not start an ffmpeg
## run that produces an empty file forever.
static func window_exists(clip: Dictionary, index: int) -> bool:
	return index >= 0 and float(index) * WINDOW < float(clip.get("duration", 0.0))


## Running window cuts, keyed by the window's path so two requests for one window are one
## ffmpeg process. Static because the library is static and a window belongs to the clip,
## not to whichever panel happened to ask first.
static var _jobs := {}


## Ask for a window. Returns true if it is ALREADY playable; false means "not yet", and a
## cut has been started if one was not already running.
##
## Cheap to call every frame - the two early exits are a file check and a dictionary
## lookup, which is what lets a caller simply ask for what it wants each frame instead of
## running a state machine.
static func request_window(clip: Dictionary, index: int) -> bool:
	if not window_exists(clip, index):
		return false
	if window_ready(clip, index):
		return true
	var dest := window_path(clip, index)
	if _jobs.has(dest):
		return false
	# A HARD CAP ON CONCURRENT CUTS, not a target. PIPELINE is how many may RUN at once, and
	# without this line it was only how many `warm` asked for - every page asks again, and
	# FilmScene asks when it opens a window, so the real number climbed past six and would
	# have kept going with more clips. libtheora pins a core each; that is a way to make the
	# machine unusable, not a way to cut faster.
	if _jobs.size() >= PIPELINE:
		return false
	if not Deps.has("ffmpeg"):
		return false
	var abs := ProjectSettings.globalize_path(dest)
	DirAccess.make_dir_recursive_absolute(abs.get_base_dir())
	# A `.part` with nothing running behind it is the wreckage of a killed run - this app's
	# or a previous launch's. Clear it rather than handing ffmpeg an output it will refuse
	# or, worse, leaving a window that can never be cut again because its debris is in the
	# way. Nothing here can be resumed: the cut is one pass, start to finish.
	if FileAccess.file_exists(abs + ".part"):
		DirAccess.remove_absolute(abs + ".part")
	var start := float(index) * WINDOW
	# `-ss` BEFORE `-i` is the fast seek - ffmpeg jumps to the keyframe rather than decoding
	# from the top of the file, which is the difference between a two-hour film taking two
	# hours to reach and taking no time at all. It has been ACCURATE as well as fast since
	# ffmpeg 2.1 (it decodes and discards from the keyframe to the exact point), so the
	# window really does begin where it says it does - which the whole clock depends on.
	var args := PackedStringArray([
		"-y", "-loglevel", "error",
		"-ss", "%.3f" % start, "-i", String(clip.get("source", "")),
		"-t", "%.3f" % (WINDOW + LEAD), "-an",
		# NEVER UPSCALES: the box is capped at the source's own size, so a small clip is
		# passed through rather than blown up to MAX_EDGE and re-encoded larger than it
		# started - which costs encode time to make the picture no better.
		"-vf", "scale=w='min(%d,iw)':h='min(%d,ih)'" % [MAX_EDGE, MAX_EDGE]
			+ ":force_original_aspect_ratio=decrease:force_divisible_by=2",
		"-c:v", "libtheora", "-q:v", Q, "-g", GOP,
		"-f", "ogg", abs + ".part"])
	var pid := Subprocess.start("ffmpeg", args, "film window")
	if pid <= 0:
		return false
	_jobs[dest] = pid
	return false


## Notice finished cuts. Call from anywhere with a frame - [FilmScene] does it while a
## panel is live, and the Generative panel does it while an import is warming, which
## between them covers every moment a window is being waited for.
##
## Nothing here BLOCKS on a cut: the subprocess runs whether or not anyone is pumping, so
## a panel that goes to sleep mid-cut simply notices later.
static func pump() -> void:
	if _jobs.is_empty():
		return
	for dest in _jobs.keys():
		var pid: int = int(_jobs[dest])
		if Subprocess.alive(pid):
			continue
		Subprocess.forget(pid)
		_jobs.erase(dest)
		var abs := ProjectSettings.globalize_path(String(dest))
		# Written as `.part` and promoted, the rule Masking paid for: a half-written media
		# file at its real name is scanned by the editor's importer, and a truncated one can
		# wedge it in a seek loop. Here it would also be a window that exists, plays a
		# fragment, and never gets cut again.
		if FileAccess.file_exists(abs + ".part"):
			if FileAccess.file_exists(abs):
				DirAccess.remove_absolute(abs)
			DirAccess.rename_absolute(abs + ".part", abs)
			_trim_cache(String(dest).get_base_dir())


## Is a cut running for this clip right now? The UI says so rather than looking idle.
static func busy(clip: Dictionary) -> bool:
	var dir := clip_dir(clip)
	for dest in _jobs.keys():
		if String(dest).begins_with(dir):
			return true
	return false


## Drop the oldest windows of one clip once there are more than [constant CACHE_WINDOWS].
## By modification time, which is last-cut rather than last-played - close enough, and it
## needs no state anywhere.
static func _trim_cache(dir: String) -> void:
	var abs := ProjectSettings.globalize_path(dir)
	var d := DirAccess.open(abs)
	if d == null:
		return
	var files: Array = []
	for f in d.get_files():
		if String(f).ends_with(".ogv"):
			files.append({"f": String(f),
				"t": FileAccess.get_modified_time(abs.path_join(String(f)))})
	if files.size() <= CACHE_WINDOWS:
		return
	files.sort_custom(func(a, b): return int(a.t) < int(b.t))
	for i in files.size() - CACHE_WINDOWS:
		DirAccess.remove_absolute(abs.path_join(String(files[i].f)))


static func _clear_windows(clip: Dictionary) -> void:
	var abs := ProjectSettings.globalize_path(clip_dir(clip))
	var d := DirAccess.open(abs)
	if d == null:
		return
	for f in d.get_files():
		DirAccess.remove_absolute(abs.path_join(String(f)))


# --- import -------------------------------------------------------------------

## Add a clip to the library. Returns "" on success, or why it could not be added.
##
## AN IMPORT IS A PROBE. It reads the duration, writes the entry, and returns - there is
## nothing to wait for, because nothing is prepared until something wants to play it. The
## first window is asked for on the way out so a clip picked up straight away has a head
## start, but that runs in the background and the library does not depend on it.
static func add(source: String) -> String:
	if not FileAccess.file_exists(source):
		return "there is no file at that path"
	if not Deps.has("ffprobe") or not Deps.has("ffmpeg"):
		return "ffmpeg is not installed - see the Dependencies panel"
	var duration := _probe_duration(source)
	if duration <= MIN_DURATION:
		return "could not read a duration from that file (is it a video?)"
	var slug := _slug(source)
	var list := clips()
	for c in list:
		if String((c as Dictionary).get("source", "")) == source:
			return "that clip is already in the library"
	var entry := {"source": source, "name": source.get_file().get_basename(),
		"duration": duration, "slug": slug, "aspect": _probe_aspect(source)}
	list.append(entry)
	_store(list)
	# A head start, not a requirement: whichever window the show is on when this clip is
	# first picked is the one that will actually be wanted, and that is not knowable here.
	request_window(entry, 0)
	return ""


## CAN THE SHOW PLAY THIS CLIP AT `show_time` RIGHT NOW - and if not, start cutting so it
## can next time.
##
## The caller is expected to ask every time it wants the clip and to have an answer ready
## for false. The comic casts an ordinary scene instead, so a window that is not cut yet is
## never a blank panel, only a page without footage on it - which also makes the whole
## thing SELF-THROTTLING: film appears as often as the machine can prepare it, and a slow
## machine gets less of it rather than a stalled one.
static func warm(clip: Dictionary, show_time: float) -> bool:
	var i := window_index(position_at(clip, show_time))
	var ready := request_window(clip, i)
	# AND THE ONES AFTER IT, WHICH IS THE HALF THAT KEEPS UP. Cutting is slower than the
	# clock it is chasing (see PIPELINE), so asking only for the window needed right now is
	# a race that is lost every time - the cut finishes, the clock has moved on, and the
	# next page asks for the next window from scratch, forever. That is not a hypothetical:
	# it is what "the video is only ever shown ONE time, on ONE page" was.
	var at := i
	for _n in PIPELINE - 1:
		at = next_window(clip, at)
		if at == i:
			break                    # a clip shorter than the pipeline: it wrapped
		request_window(clip, at)
	return ready


## The window after `index`, wrapping - a clip loops, so past the last window is the first.
static func next_window(clip: Dictionary, index: int) -> int:
	return index + 1 if window_exists(clip, index + 1) else 0


## A filename-safe, collision-resistant name for a source path. The hash is what stops a
## second `clip.mp4` from a different folder overwriting the first one's transcode.
static func _slug(source: String) -> String:
	var base := source.get_file().get_basename().to_lower()
	var out := ""
	for i in base.length():
		var c := base[i]
		out += c if (c >= "a" and c <= "z") or (c >= "0" and c <= "9") else "_"
	out = out.substr(0, 40).strip_edges()
	return "%s_%d" % [out if not out.is_empty() else "clip", hash(source) & 0xFFFFFF]


## Seconds of video in `path`, or 0. Three ways, in order of trust, because a container
## can be missing the duration in its format header, in its stream header, or in both -
## and the last resort counts the packets. Lifted from mask_editor._probe_duration, which
## grew each fallback in response to a real file that needed it.
static func _probe_duration(path: String) -> float:
	var d := _ffprobe_float(["-show_entries", "format=duration"], path)
	if d > 0.0:
		return d
	d = _ffprobe_float(["-select_streams", "v:0", "-show_entries", "stream=duration"], path)
	if d > 0.0:
		return d
	var out: Array = []
	Deps.execute("ffprobe", ["-v", "error", "-select_streams", "v:0", "-count_packets",
		"-show_entries", "stream=nb_read_packets,r_frame_rate", "-of", "default=nw=1", path], out)
	if out.size() > 0:
		var packets := 0.0
		var fps := 0.0
		for line in String(out[0]).split("\n"):
			var s := line.strip_edges()
			if s.begins_with("nb_read_packets="):
				packets = s.substr(16).to_float()
			elif s.begins_with("r_frame_rate="):
				var fr := s.substr(13).split("/")
				if fr.size() == 2 and fr[1].to_float() > 0.0:
					fps = fr[0].to_float() / fr[1].to_float()
				else:
					fps = s.substr(13).to_float()
		if packets > 0.0 and fps > 0.0:
			return packets / fps
	return 0.0


## The clip's shape, width over height. Kept on the entry because it decides WHICH PANEL
## the film goes in: cover crops whichever axis is spare, so a 16:9 clip in a tall panel
## loses most of its width, and putting it in a panel of roughly its own shape loses almost
## nothing. See ComicVehicle._choose_film.
static func _probe_aspect(path: String) -> float:
	var out: Array = []
	if Deps.execute("ffprobe", ["-v", "error", "-select_streams", "v:0",
			"-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path], out) != 0:
		return 0.0
	if out.is_empty():
		return 0.0
	var wh := String(out[0]).strip_edges().split("x")
	if wh.size() < 2 or wh[1].to_float() <= 0.0:
		return 0.0
	return wh[0].to_float() / wh[1].to_float()


## Probed shapes for clips stored before `aspect` was kept, so an existing library gets the
## better placement too. Once per clip per process: ffprobe blocks, and this is reached from
## a page turn.
static var _aspect_cache := {}


## The clip's aspect, 0 when it cannot be determined (which callers must treat as "no
## preference" rather than as a shape).
static func aspect_of(clip: Dictionary) -> float:
	var a := float(clip.get("aspect", 0.0))
	if a > 0.0:
		return a
	var src := String(clip.get("source", ""))
	if src.is_empty():
		return 0.0
	if not _aspect_cache.has(src):
		_aspect_cache[src] = _probe_aspect(src)
	return float(_aspect_cache[src])


static func _ffprobe_float(entries: Array, path: String) -> float:
	var args: Array = ["-v", "error"]
	args.append_array(entries)
	args.append_array(["-of", "default=noprint_wrappers=1:nokey=1", path])
	var out: Array = []
	if Deps.execute("ffprobe", args, out) != 0 or out.is_empty():
		return 0.0
	return String(out[0]).strip_edges().to_float()
