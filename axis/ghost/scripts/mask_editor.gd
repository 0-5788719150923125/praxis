extends CanvasLayer
class_name MaskEditor

## MaskEditor - the mask-mode authoring surface.
##
## Load a clip, key two colors apart (per-pixel hue classification, see
## shaders/mask_split.gdshader), place MARKERS where the split/effect should
## change, scrub the timeline, export. See scripts/mask_session.gd for the data
## model - a session's markers are fixed-schema scalar vectors, not free-form
## params, and every one is either a RAMP (eases in before its anchor) or a DAMP
## (accumulates after it) - there's no third, undifferentiated "marker" kind.
##
## Standalone by design: a mask session is tied to one specific external clip, not
## the audio-reactive show, so this does NOT route through Director/Spectrum. Two
## entry points (see main.gd):
##   --mask-edit [path]    interactive editor (this file's normal mode)
##   --mask-render <json>  the export relaunch's headless-ish render mode
##                         ([member render_mode] = true; no panel, autoplay, quits
##                         when the audio ends - mirrors --export/_export_mode).
##
## Widget choices mirror the rest of ghost: the timeline is a bespoke-drawn Control
## ([MaskTimeline], the [DialWidget] idiom), the export button/status/FileDialog are
## the exact pattern from exporter.gd, color entry is Godot's native
## ColorPickerButton (precision input, not an "instrument" worth hand-drawing).
##
## The preview's view modes (see MaskSession.VIEW_MODES + _build_video_composition)
## are the main screen and the inset as independent axes - raw, inset-raw, inset-fx,
## both-fx, full-fx - cycled in that "evolution" order by one button (VIEW_CYCLE).
## Only one video ever decodes ([member _player], always raw); the fx layers just
## re-draw its decoded texture through [const SHADER], each only while visible - so
## "raw" costs no shader pass at all.
##
## view_mode is a per-MARKER field (see MaskSession.VECTOR_FIELDS), not an editing-
## only preference: the toggle button edits the marker at the playhead exactly like
## every other panel control (see _edit), the live preview always renders whatever
## session.at_time() resolves to, and the export relaunch (render_mode) runs the
## identical per-frame logic - so a marker set to "raw" plays raw in the rendered
## file too, not just live. What you see while editing is what you get.
##
## THE REGION is the one control here that is spatial rather than chromatic, and
## it is UNIVERSAL - not an entry in EFFECT_CONTROLS, because "where may this
## layer act" is meaningful for every effect. Each marker carries a box in frame
## UV (reg_x0/y0/x1/y1 + reg_soft) which multiplies into its layer's on-screen
## weight, so confining a layer is the same operation as fading it, resolved per
## pixel; the default is the whole frame. It exists because a colour key cannot
## separate two things that ARE the same colour - a yellow wall and a gold coin
## in front of it - and their positions separate them trivially. The box is
## dragged on the video itself, not typed (see _build_region_overlay).
##
## ANY FRAME SHAPE. The clip's own pixel dimensions are read once ([member
## _src_size]) and are the only thing that decides the picture's geometry: the
## editor fits the whole frame inside its video area (a portrait phone clip gets
## black bars left and right, a wide one keeps them above and below), the effects'
## patterns stay isotropic through the shader's `u_aspect`, and the export records
## at the SOURCE's resolution - a 1080x1920 clip renders a 1080x1920 movie. Nothing
## in here assumes 16:9; see _src_size and _write_render_override.

const SHADER := preload("res://shaders/mask_split.gdshader")
const PAINT_SIM_SHADER := preload("res://shaders/clown_paint.gdshader")
const UMBRA_SIM_SHADER := preload("res://shaders/umbra_field.gdshader")
const MASKS_DIR := "res://masks"
## Ghost's OWN python venv - yt-dlp lives here, never the repo's venvs or the
## system site-packages. user:// keeps it out of res:// (the editor's file
## scanner must never crawl a venv) and out of git entirely.
const YT_VENV_DIR := "user://ytdlp_venv"
const YT_DL_DIR := MASKS_DIR + "/_downloads"   # inside /masks/ = already gitignored
const PANEL_W := 320
# Picked via _sort_dropdown - see _apply_sort for what each one does.
const _SORT_MODES := ["A → Z", "Z → A", "Energy"]

## Set by main.gd before open_source() for the --mask-render relaunch: skip the
## editing panel, autoplay from t=0, quit when the audio finishes.
var render_mode := false

var session: MaskSession = null
var _session_path := ""       # res://-relative or absolute; wherever it was loaded from

var _player: VideoStreamPlayer     # always the RAW decode - never carries the shader
var _audio: AudioStreamPlayer
var _audio_thread: Thread = null   # loads the (large, uncompressed) main WAV off the main thread
var _render_t := 0.0               # accumulated MOVIE time in render mode - the deterministic
                                   #   export clock (sum of the fixed-fps _dt), authoritative over
                                   #   the video/audio stream clocks, which can drift or end early
var _autostart_pending := false    # live autostart is HELD (video paused on frame 1) until the
                                   #   threaded audio attaches, so the intro never plays audio-less
                                   #   and skips - _poll_audio_thread begins playback, synced (below)
var _pending_restore := -1.0       # playhead seconds to seek to once the player is ready (see _process)
var _pending_restore_tries := 0
var _reload_check_pid := -1        # headless compile check gating a reload (see _do_restart)
var _reload_check_log := ""
## Set when a reload was requested while an export was mid-flight (_render_state !=
## "idle") - see _reload_requested. _poll_render re-fires the request once the export
## finishes, same deference the assistant already gets in reload_when_idle: a restart
## quits this process, and Godot kills the child processes it created (see assistant.gd's
## _closing doc) - including the render/transcode subprocess an export is waiting on.
var _reload_after_export := false
## Set the instant _restart_now actually commits to quitting (after its own final
## _save_session capture) - see _save_session and _exit_tree for why this exists.
var _restarting := false
var _track_audio_jobs: Array = []  # background sidecar-.ogg extractions, {pid, index, ogg}
# One material PER LAYER: the main overlay and the inset can be mid-transition at
# different presences (e.g. fx-inset -> both: the inset holds full while the main
# overlay fades in), and a layer's presence multiplies into its own intensities -
# impossible with one shared material.
var _mat_main := ShaderMaterial.new()
var _mat_inset := ShaderMaterial.new()
var _playing := false
var _audio_holding := false   # main audio paused-in-place, waiting for video to catch up (see _process)
var _cursor_idle_t := 0.0
const _CURSOR_HIDE_DELAY := 1.5   # seconds of stillness during playback before the mouse cursor hides

var _fx_overlay: TextureRect       # full-frame fx layer - shaded copy of whichever source is active
var _cont_view: TextureRect        # full-frame RAW layer for an active continuation track (see _sync_tracks) -
                                   # _player's own raw picture is only valid while session.main_visible_at's
                                   # own-clip half holds; once a continuation track owns time t, this shows
                                   # THAT track's own independently-decoded frame instead (never _player's -
                                   # see continuation_track_at's doc for why the two used to be conflated)
var _pip_view: TextureRect         # the inset's content - shaded or raw per view mode
var _mask_wrap: PanelContainer     # the inset's border/placement box (holds _pip_view)
var _view_label: Label     # passive view-mode readout (the old cycle button; V cycles now)
var _help_panel: PanelContainer
var _peek_raw := false     # DISPLAY-ONLY raw override; never touches session data (hold P)
var _last_inset_show := 0.0   # this frame's resolved inset_show - _sync_tracks reads it too,
                               # so a track's own PiP box respects the same view-mode gate as _mask_wrap
var _pip_track := 0           # this frame's resolved pip_track: 0 = main clip in the PiP, k = track (k-1)

var _timeline: MaskTimeline
var _tview: TimelineView          # shared pixel<->time mapping - see timeline_view.gd
var _video_area: AspectRatioContainer  # letterboxed video slot - see _refresh_lanes
var _lanes_col: VBoxContainer     # primary clip's trim lane + one per imported track
var _composition_parent: Control  # holds _player/_fx_overlay/_mask_wrap - track PiP views land here too

# --- THE SOURCE CLIP'S OWN FRAME SIZE -------------------------------------------
# Everything that has to know the picture's SHAPE reads this: the editor's
# letterboxed video slot (_build_editor_ui), the export's recorded resolution
# (_write_render_override), the meta mirror's hand-built pane
# (_shrink_into_video_pane) and the umbra eye fit's aspect correction. A portrait
# clip (1080x1920) is therefore not a special case anywhere - it is just this pair
# being taller than it is wide, and the editor pillarboxes it (black bars either
# side) exactly the way a short clip already letterboxes. Nothing here assumes
# 16:9; the several places that used to hardcode 1.7778 / 1920x1080 were the ONLY
# reason a vertical clip came out horizontally squashed.
#
# Seeded by ffprobe at session-ready - the editor layout and the export launch
# both need it BEFORE a frame has decoded - then confirmed from the decoded
# texture itself the first frame it exists (_sync_source_size), which is
# authoritative and also covers a machine with no ffprobe on PATH.
const _SRC_SIZE_FALLBACK := Vector2i(1920, 1080)
var _src_size := _SRC_SIZE_FALLBACK
var _src_size_confirmed := false   # the decoded texture has had its say

## Runtime state per session.tracks[i] - NOT persisted (session.tracks holds only
## the data; players/views are rebuilt from it every _ready_with_session). Each:
## {player: VideoStreamPlayer, view: TextureRect, active: bool}. `active` tracks
## whether the track is CURRENTLY the one playing (see _sync_tracks) so entering/
## leaving its window on the master timeline only seeks+starts/pauses it once,
## not every frame.
var _track_runtime: Array = []
var _import_dialog: FileDialog
var _import_pid := -1
var _import_pending := {}   # {source, video, index} - the lane is added up front, this tracks its transcode

# --- YouTube / URL import runtime (see _start_url_import) ---
var _yt_state := "idle"    # idle | venv | pip | downloading
var _yt_pid := -1
var _yt_url := ""
var _yt_log := ""          # absolute path; the current step's combined stdout/stderr
var _yt_started := 0.0     # unix time the download step began (newest-file fallback)
var _yt_step_started := 0.0   # unix time the CURRENT step began - the elapsed readout
var _yt_retried := false   # one automatic pip-upgrade retry per import (stale yt-dlp)
var _yt_echoed := {}       # WARNING/ERROR lines already print()-echoed live (a set)

# --- The clown face model (see _update_face_model): eyes / mouth / oval /
# tint, fitted per capture tick, EMA-glided exactly like the anchor. Defaults
# describe a plausible centered face so the paint shows before the first fit.
var _clown_active := false
var _face_eye_l_prev := Vector2(0.42, 0.40)
var _face_eye_l_ema := Vector2(0.42, 0.40)
var _face_eye_r_prev := Vector2(0.58, 0.40)
var _face_eye_r_ema := Vector2(0.58, 0.40)
var _face_mouth_prev := Vector2(0.5, 0.62)
var _face_mouth_ema := Vector2(0.5, 0.62)
var _face_r_prev := Vector2(0.14, 0.20)
var _face_r_ema := Vector2(0.14, 0.20)
var _face_tint_ema := Vector3(0.55, 0.1, -0.5)
var _face_lum_ema := 0.5
# Per-feature sizes, measured from each cluster's own spread, in PLAIN
# SCREEN units (uv) - deliberately NOT eye-distance units: normalizing by
# the pair re-coupled every feature's size to the pair's jitter and the
# whole mask rescaled in lockstep. Each feature breathes on its own.
var _face_eye_lr_prev := 0.035
var _face_eye_lr_ema := 0.035
var _face_eye_rr_prev := 0.035
var _face_eye_rr_ema := 0.035
var _face_mouth_r_prev := Vector2(0.05, 0.028)
var _face_mouth_r_ema := Vector2(0.05, 0.028)
# The face centroid (the coat channel's home + the cracks' local pull) and
# the nose's OWN slow anchor - derived from eyes+mouth but smoothed on its
# own clock, so it does not inherit the pair's frame jitter.
var _face_c_prev := Vector2(0.5, 0.45)
var _face_c_ema := Vector2(0.5, 0.45)
var _face_nose_prev := Vector2(0.5, 0.52)
var _face_nose_ema := Vector2(0.5, 0.52)
# The face model's OWN capture cadence - faster than the echo ring's 0.35s,
# because the mask must chase the face in near-real-time (the half-second
# drift report), while the ring's semantics (8 slots x 0.35s of history)
# must not change. A tick that coincides with an echo capture reuses its
# image instead of paying a second readback.
const _FACE_INTERVAL := 0.15
var _face_slot := -1
## The aspect every hand-tuned bound in the face model was calibrated at. The
## fitted radii are stored in raw uv (the shaders' contract), but a BOUND on a
## width has to be applied in height units - "0.145 of the frame's width" is
## 0.258 frame-heights on a 16:9 clip and 0.082 on a 9:16 phone clip, i.e. the
## same rule crushing a face to a third of its width depending only on what the
## clip's shape happens to be. Converting through this constant keeps 16:9
## footage bit-identical to what those numbers were tuned against, and gives
## every other shape the same physical bound instead of a scaled one.
# --- THE FACE TRACK: real landmarks, fitted once, read by time ------------------
# What the clown's feature positions come from now. See face_host/face_track.py for
# the pre-pass and the file format; this half bootstraps it, runs it, and reads it.
#
# Same architecture as the umbra look-ahead track (_umb_ensure_track), for the same
# three reasons: the live editor and the export relaunch are separate PROCESSES
# that must agree frame-for-frame and do because they read one cached file; no
# detection ever runs inside the render loop; and because the whole track exists
# before playback, smoothing can use frames on BOTH sides of now, which a live
# tracker cannot - it only has the past, so it must choose between lag and jitter.
#
# The venv is ghost's OWN (the yt-dlp/voice discipline): one effect's dependency
# must never become the project's, and it is bootstrapped on first use rather than
# required up front, so a session that never touches the clown never installs it.
const FACE_VENV_DIR := "user://face_venv"
const FACE_TRACK_DIR := "user://face_tracks"
## Per-user editor preferences. ONE cfg serves the whole app (splash's remembered
## song and clip, [director], [synth], [generative]), so every write here is a
## READ-MODIFY-WRITE of just the [mask] section - a fresh ConfigFile.save() would
## silently wipe all of those. Same discipline as Director._save_pacing.
const PREFS_CFG := "user://ghost.cfg"
## Google's published landmarker bundle. Fetched once into the venv dir; 3.7 MB.
const FACE_MODEL_URL := "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
const FACE_TRACK_RATE := 15.0    # samples/sec - must match what face_track.py wrote
const FACE_TRACK_POINTS := 478
const FACE_TRACK_HEADER := 20    # 4 magic + u32 version + f32 rate + u32 count + u32 points

var _ft_state := "none"   # none | venv | pip | model | tracking | ready | failed
var _ft_pid := -1
var _ft_path := ""        # the cached track for THIS clip
var _ft_log := ""
var _ft_rate := FACE_TRACK_RATE
var _ft_count := 0
var _ft_points := FACE_TRACK_POINTS
## The whole track in memory: count x points x (x, y), plus a found flag per
## sample. ~12 MB for a four-minute clip at 15 Hz, read once.
var _ft_xy := PackedFloat32Array()
var _ft_found := PackedByteArray()

const _FACE_ASP_REF := 1.7778
## What counts as the face's CORE, as a fraction of the peak weight - the mass the
## size fit is allowed to see. See _update_face_model: the weight field is broad by
## design, and letting the broad part set the face's size is what pushed the eye
## pair off the face on real footage.
const _FACE_CORE := 0.35
## KNOWN LOOSE END, measured, deliberately left alone: this fit's own bound
## (0.30 x _FACE_ASP_REF) is nearly twice as wide as the cap the paint sim draws
## the coat with (clown_paint.gdshader's FACE_CAP_XHI). Since rx also sets how far
## out the eye search ranges, on a clip where the key hue is in the background too
## the fit runs wide and the band can reach past the face - far enough to take a
## dark headscarf hanging by a shoulder as an eye. Tightening this to agree with
## the coat's cap looks obviously right and was tried: on 16:9 footage it collapsed
## the detected eye pair (separation 0.164 -> 0.072) and dropped the mouth onto the
## nose, because every other constant here is tuned against a band this wide. It
## needs the whole vertical model re-tuned together, not a one-line tightening.
var _face_prev_lum := PackedFloat32Array()   # 96x54 luminance, for the mouth's motion cue
var _face_motion_mean := 0.0                 # EMA of the face's ambient per-pixel motion
var _face_red_ema := 0.02                    # mean (r-g) over the face - the lips' baseline
var _face_bg_lum := 0.2                      # this frame's whole-frame mean luminance (the
                                             #   brightness cue's floor - see _update_face_model)
# The liquid paint simulation (see shaders/clown_paint.gdshader and
# _step_paint_sim): a ping-pong SubViewport pair holding the persistent,
# advected paint field the mask shader samples as u_clown_paint. Built
# lazily on the first clown frame; stepped on playback-time deltas.
var _paint_vps: Array = []
var _paint_rects: Array = []
var _paint_ping := 0
var _paint_reset := true
var _paint_last_pos := 0.0
var _clown_fs := 1.0   # the live clown layer's Scale knob - the sim's target sizes ride it
var _rain_squall: HSlider
var _au_echo: HSlider
var _au_time: HSlider
var _au_amb: HSlider
var _au_room: HSlider
var _au_reso: HSlider
var _au_bass: HSlider
var _clown_feather_sl: HSlider
var _clown_eye_sl: HSlider
var _clown_steady_sl: HSlider
var _clown_drip_sl: HSlider
var _clown_drip_w_sl: HSlider
var _clown_smudge_sl: HSlider
var _clown_drip_curve_sl: HSlider
var _clown_smile_sl: HSlider
var _clown_curve_sl: HSlider
var _clown_evidence_sl: HSlider
## The clown's four tuning knobs, resolved per frame from the live layer (see
## _apply_frame_state). Each is a view onto a field the keying/pattern groups own
## for other effects and clown never uses - the same idiom umbra's six follow -
## and each DEFAULT REPRODUCES the constant it replaced, so no existing session
## changes look until its author moves a slider.
var _clown_evidence := 0.0   # resonance  - drawn-shape floor vs the picture's own evidence
## THE SHAPE KNOBS, on the three fields Steady/Firm/Lead used to hold.
##
## Those three existed to fight a detector that jittered and lagged. The landmark
## track does neither - it is fitted offline over the whole clip, so there is
## nothing to smooth and nothing to predict - which left three controls on screen
## doing nothing and three stored fields free. They now carry what the effect
## actually lacked, and what a fitted ellipse could never have expressed anyway:
## the SHAPE of the drawn features. (The fallback fitter keeps the old constants
## internally, so it behaves exactly as before when there is no track.)
var _clown_eye_size := 1.0     # threshold - how far the eye patch grows past the eye
var _clown_drip := 0.0         # sat_floor - how far the black runs down the cheek
## The run's WEIGHT and its BOW. Drip says how FAR the liner runs; these two say
## what it looks like on the way, which is the half the author was left with no
## say over at all ("no options to adjust this drop shape, or its position, or
## its curve, or anything"). Width scales the source width the shader measures
## off the eye patch itself, so it stays a fraction of THIS eye whatever size
## Eye size has grown it to; curve scales how far the run bows outboard across
## the cheekbone. Both defaults are the value their field defaults to.
## How far the eye paint is rubbed past its own contour. `swap` was "legacy,
## unused" for as long as the format has existed - see MaskSession's LAYER_FIELDS.
var _clown_smudge := 0.45      # swap        - the eye edge, rubbed rather than cut
var _clown_drip_w := 0.35      # fx_y        - the run's width, x the measured lash line
var _clown_drip_curve := 0.25  # intensity_b - how far it bows out across the cheek
var _clown_smile_w := 1.0      # feather   - the smile's width, past the real mouth
var _clown_smile_curve := 0.0  # fx_speed  - its corners swept up (-) or down (+)
## How far the coat's outer edge fades out, in frame UV. On fx_x - Pan is hidden
## for the clown (it could only slide a face-tracked mask OFF the face), so the
## field is free and this is the control that most wants a home.
var _clown_feather := 0.012
## How wide the CENTRED kernel is that smooths the landmark track, in samples
## (see _ft_point). Because it is centred it costs no lag at any width - the only
## thing a wider one loses is genuinely fast motion, which it rounds off. On
## hue_b, the last field clown had free.
var _ft_sigma := 2.5
## The clown layer's region box, mirrored CPU-side so the face SEARCH honours it
## too (see _face_region_at). Whole frame until the author draws one.
var _clown_region := Vector4(0.0, 0.0, 1.0, 1.0)
## The live audio layer (see _apply_audio_fx); empty when no audio marker governs.
var _audio_layer := {}
var _clown_bleed := 0.0    # its Bleed / Settle / Hollow knobs, fed to the paint sim
var _clown_settle := 0.35
var _clown_hollow := 0.0

# --- UMBRA: the ghost in the subject's own cast shadow (see
# _update_umbra_model and shaders/umbra_field.gdshader). The detector works on
# a coarse grid - a cast shadow is a REGION, and regions survive downsampling
# in a way the clown's eye sockets never did.
const _UMB_W := 96
const _UMB_H := 54
const _UMBRA_INTERVAL := 0.15   # same cadence as the face model, same reasoning
var _umbra_active := false
var _umb_slot := -1
# The wall the shadow falls on, as a chroma DIRECTION plus its lit level. A
# near-static property of the scene, so it is EMA'd hard and the (more
# expensive) multi-hypothesis re-pick only runs occasionally - see
# _umb_repick_in.
var _umb_ref_dir := Vector3(0.0, 0.0, 0.0)
var _umb_ref_mag := 0.05
var _umb_ref_lit := 0.6
var _umb_ref_valid := false
var _umb_ref_cov := 0.0     # scene-coherence of the winning hypothesis (the confidence gate)
var _umb_repick_in := 0
# The cast direction (her centroid -> the shadow's), aspect-corrected unit.
# The light does not move, so this is smoothed almost to a constant: measured
# over 15s of the reference clip it holds to +-3.6 degrees, and letting it
# wander per-frame is exactly how the clown earned its uniform twitch.
var _umb_dir_ema := Vector2(1.0, 0.0)
var _umb_subj_c := Vector2(0.5, 0.55)
var _umb_shad_c := Vector2(0.75, 0.40)
var _umb_pivot := Vector2(0.75, 0.62)   # the silhouette's base - Scale rears UP from here
var _umb_pan := Vector2.ZERO
var _umb_reach := 0.24   # Reach  (threshold) - how close the mass comes to her
var _umb_lead := 0.19    # Lead   (feather)   - seconds the ghost moves AHEAD of her
# THE GHOST'S EYES. Her own eyes, carried across the cast offset into the
# shadow and through the silhouette transform, then LED by a velocity estimate
# so the ghost turns fractionally before she does - the puppeteering read.
var _umb_eye_l := Vector2(0.5, 0.4)
var _umb_eye_r := Vector2(0.6, 0.4)
var _umb_eye_rad := 0.05
var _umb_eyes_ok := false
# The look-ahead eye track (see _umb_ensure_track): her eyes for the WHOLE
# clip, fitted up front, so playback can read the frame she has not reached.
var _umb_track := PackedVector3Array()   # (eye_mid.x, eye_mid.y, separation); z<=0 = no face
var _umb_track_rest := Vector2(0.5, 0.35)
var _umb_track_state := "none"           # none | decoding | fitting | ready | failed
var _umb_track_raw := ""
var _umb_track_pid := -1
var _umb_track_thread: Thread = null
# The clip's aspect, snapshotted for the worker thread (_umb_fit_eyes runs off the
# main thread and must not touch live state). Every anatomical measure in the fit is
# in aspect-corrected space; leaving it at a fixed 1.7778 made the face prior and the
# eye-separation constraint wrong by the ratio between the clip's shape and 16:9 -
# on a portrait clip that is a factor of three, which is enough to reject every fit.
var _umb_fit_asp := 1.7778
var _umb_eye_amt := 0.0   # Gaze (sat_floor) - how strongly the eyes read
var _umb_have := false
var _umb_region_img: Image = null        # RGBA8 _UMB_W x _UMB_H: R=linked shadow,
var _umb_region_tex: ImageTexture = null #   G=shadowness, B=subject mask
# Scratch buffers, allocated once - this runs at ~7Hz and must not churn the heap.
var _umb_lum := PackedFloat32Array()
var _umb_cr := PackedFloat32Array()
var _umb_cg := PackedFloat32Array()
var _umb_cb := PackedFloat32Array()
var _umb_cmag := PackedFloat32Array()
var _umb_match := PackedFloat32Array()
var _umb_shadow := PackedFloat32Array()
var _umb_tmp := PackedFloat32Array()
var _umb_tmp2 := PackedFloat32Array()
var _umb_wmass := PackedFloat32Array()
var _umb_wlum := PackedFloat32Array()
var _umb_subj := PackedByteArray()
var _umb_shad := PackedByteArray()
var _umb_queue := PackedInt32Array()
var _umb_bytes := PackedByteArray()
# The umbra field simulation (ping-pong SubViewport pair, same discipline as
# the clown's paint sim: stepped on PLAYBACK deltas so pause freezes it and
# export traces the identical currents).
var _umb_vps: Array = []
var _umb_rects: Array = []
var _umb_ping := 0
var _umb_reset := true
var _umb_last_pos := 0.0
var _umb_hue := -1.0     # the live layer's key hue (biases the wall pick); <0 = none
var _umb_loom := 0.45    # Coverage, with the audio swell already folded in
var _umb_rise := 1.0     # Velocity
var _umb_roil := 0.5     # Contrast
var _umb_wisp := 0.0     # Wisp   (fx_smooth)
var _umb_cling := 0.35   # Cling  (fx_lag)
var _umb_scale := 1.0    # Scale
var _selected: Variant = null   # the marker Dictionary currently shown in the panel

var _color_a: ColorPickerButton
var _hue_a: HSlider   # the Morph slider (fx_tint - palette rotation), decoupled from _color_a
var _threshold: HSlider
var _threshold_label: Label
var _grp_color: VBoxContainer   # "Key color" swatch, pinned above the sortable options below
var _key_color_label: Label     # retitled per effect (umbra's picker names the WALL, not a key)
var _grp_options: VBoxContainer   # every effect option (label+slider pairs), reordered by _apply_sort
var _options: Array = []          # [{label: Label, control: Control}], creation order - see _register_option
var _sort_mode := 2                # index into _SORT_MODES - defaults to Energy, see _apply_sort
var _sort_dropdown: OptionButton
var _feather: HSlider
var _sat_floor: HSlider
var _fx_x: HSlider
var _fx_y: HSlider
var _fx_x_label: Label   # "Pan X/Y" relabeled "Wind X/Y" for snow - direction, not placement
var _fx_y_label: Label
var _fx_scale: HSlider
var _fx_density: HSlider
var _fx_density_label: Label   # "Coverage" relabeled "Stickiness" for crystal (feature conformance)
var _fx_contrast: HSlider
var _fx_contrast_label: Label   # "Contrast" relabeled "Sensitivity" for snow (no keying group of its own)
var _fx_speed: HSlider
var _fx_lag: HSlider
var _fx_lag_label: Label            # "Lag (s)" relabeled "Lead (s)" for oracle - same field, opposite sense
var _fx_smooth: HSlider
var _gust: HSlider   # snow's own Gust slider - a second, independent view onto fx_smooth (see _fx_smooth)
var _undul: HSlider  # fur's Undulation - fur's view onto fx_smooth (same stored-field reuse as _gust)
var _coil: HSlider   # fur's Coil - fur's view onto fx_lag (pushed raw as u_l_lagf; echo bakes its lag into u_l_ew)
var _stick: HSlider  # fur's Stickiness - its OWN field (fx_stick, u_l_stick); 0 = today's free coat
var _bleed: HSlider   # clown's Bleed   - its view onto fx_smooth (see _gust/_undul for the idiom)
var _settle: HSlider  # clown's Settle  - its view onto fx_lag
var _hollow: HSlider  # clown's Hollow  - its view onto fx_stick
var _wisp: HSlider        # umbra's Wisp  - its view onto fx_smooth
var _cling: HSlider       # umbra's Cling - its view onto fx_lag
var _umbra_depth: HSlider # umbra's Depth - its view onto fx_stick
var _umbra_reach: HSlider # umbra's Reach - its view onto threshold (umbra never keys)
var _umbra_lead: HSlider  # umbra's Lead  - its view onto feather
var _umbra_gaze: HSlider  # umbra's Gaze  - its view onto sat_floor
var _color_eye: ColorPickerButton   # umbra's eye colour - hue_b, pushed as u_l_accent
## repaint's replacement colour. Unlike every other picker here this one is a
## WHOLE colour, not a hue: it writes hue_b/fx_stick/fx_tint (h/s/v), so black
## and white are reachable and "replace the yellow wall with black" is one pick.
var _color_paint: ColorPickerButton
var _paint_reach: HSlider           # repaint's Reach - fx_contrast
var _paint_smooth: HSlider          # repaint's Smoothing - fx_smooth
## The region box: a CheckBox that turns it on for the selected marker, a slider
## for its border softness, and the draggable box itself over the video pane.
var _region_on: CheckBox
var _region_soft: HSlider
var _region_overlay: Control
var _region_drag := ""              # "" | nw/ne/sw/se | n/s/e/w | move
var _region_drag_from := Vector2.ZERO   # mouse position at press
var _region_drag_box := Rect2()         # the box at press - drags resolve from THIS
var _resonance: HSlider
var _effect_a: OptionButton
var _intensity_a: HSlider
var _intensity_label: Label   # tooltip swaps meaning for restore/clear, see _update_effect_controls
var _kind: OptionButton     # ramp / damp - see MaskSession.MARKER_KINDS
var _marker_duration: HSlider
var _marker_label: Label
var _time_label: Label
## Whether playback drags the SELECTION along with the playhead (the "Follow"
## switch beside the marker list's header). On by default; off is what lets an
## author hold one marker selected and tune it while the clip runs.
var _follow_playhead: CheckBox
var _marker_list: VBoxContainer   # sequential ramp/damp list, pinned to the panel's bottom
var _history_label: Label   # "Undo: <last action>" preview above the +Ramp/+Damp row, see _refresh_history_label

var _feedback: Node = null    # backtick console (see _build_feedback); editor mode only
var _was_playing_before_feedback := false

var _status: Label            # shared bottom-right notification - prep AND export
var _export_btn: Button
var _dialog: FileDialog
var _open_dialog: FileDialog

var _prep_video_pid := -1
var _prep_audio_pid := -1
var _prep_src_dur := 0.0         # source clip duration (seconds), for the video step's %
var _prep_state := "idle"        # idle / prepping_video / prepping_audio
var _pending := {}               # source/dir/video/audio paths mid-prep
const _PREP_PROGRESS_FILE := "user://mask_prep_progress.txt"

# Temporal capture (echo ring + whisp anchor): quarter-res snapshots of past
# frames, taken every _ECHO_INTERVAL seconds of PLAYBACK time (slot = position /
# interval, so live preview and the export relaunch capture at the same clip
# positions - the deterministic-clock discipline, one level up from u_time).
# Only runs while a whisp/echo/snow/oracle/serpent/chimera layer is actually on
# screen THIS frame (_temporal_active, set alongside _chimera_active in
# _apply_frame_state) - a session-wide "does this ever use one" check made the
# synchronous GPU readback below run for the ENTIRE session once any one of
# those markers appeared anywhere in it, stuttering unrelated stretches of a
# long timeline (feedback/0016).
var _echo_ring: Array = [null, null, null, null, null, null, null, null]   # ImageTexture, slot-indexed
var _echo_slot := -1
var _prev_pos := -1.0         # last frame's playhead position - lets capture skip an ACTIVE
                              #   scrub drag (position moving) and fire once it settles
var _chimera_active := false  # is a chimera layer on screen THIS frame - gates the track readback
                              #   so it never runs while the (distant) chimera marker isn't rendering
var _temporal_active := false # is a whisp/echo/snow/oracle/serpent/chimera layer on screen THIS
                              #   frame - gates _maybe_capture_echo's readback entirely
var _meta_amount := 0.0       # strongest meta-layer weight on screen THIS frame (0 = none);
                              #   drives the workspace capture and the render-mode chrome reveal
var _workspace_tex: ImageTexture = null  # the editor's own previous frame, captured for the META
                              #   mirror (see _capture_workspace) - null until the first capture
var _meta_chrome: Control = null  # render-mode only: the editor-chrome overlay a META section
                              #   fades in over the clean video (the recorded product demo)
var _chrome_parent: Node = null   # where _build_chrome() parents (self, or _meta_chrome in export)
# The whisp anchor is double-buffered: _anchor_ema is the EMA at the LATEST
# capture, _anchor_prev the one before. The uniform pushed per frame lerps
# between them by position-within-slot (see _push_anchor) - pushing the EMA
# directly stepped the whole pattern once per capture, a visible jump amplified
# by pattern zoom ("jittery, it resets, it jumps"). Position-keyed, so live and
# export trace the identical glide.
var _anchor_prev := Vector2(0.5, 0.5)
var _anchor_ema := Vector2(0.5, 0.5)
# Chimera's landmark FRAME (position + size), double-buffered and glided exactly
# like the centroid above. The MAIN face's modeled size (RMS radius of its
# key/motion mass) and the imported TRACK face's own centroid + size, each an
# EMA over the same thresholds. The shader maps every main pixel through the two
# frames so the graft phase-locks to the main head instead of drifting (see the
# chimera branch in mask_split.gdshader). Defaults degrade to ~the old centered
# clone when a face can't be modeled.
var _anchor_scale_prev := 0.28
var _anchor_scale_ema := 0.28
var _track_anchor_prev := Vector2(0.5, 0.5)
var _track_anchor_ema := Vector2(0.5, 0.5)
var _track_scale_prev := 0.28
var _track_scale_ema := 0.28
var _track_prev_lum := PackedFloat32Array()   # track frame's previous-luminance grid (motion fallback)
const _ECHO_INTERVAL := 0.35

# Wave impulses (whisp only): a fast head turn shows up as a big frame-to-frame
# luminance jolt in the same 48x27 grid _update_whisp_anchor already samples -
# an ONSET detector (motion vs. an adaptive EMA baseline + deviation, not a
# fixed magic threshold) fires an impulse at the anchor's current position, and
# the shader drops a decaying blob of paint there that drifts off along
# whisp's own local current, confined to the volumetric field like the rest of
# whisp (see u_wave_* / wave_wash in mask_split.gdshader) - a drop carried by
# the water, not a ring detached from it. WAVE_SLOTS must match WAVEN in the
# shader.
const _WAVE_SLOTS := 3
const _WAVE_COOLDOWN := 1.1   # seconds; keeps a shaky run from piling up waves
var _wave_prev_lum := PackedFloat32Array()   # last capture's 48x27 luminance grid
var _wave_motion_ema := 0.0
var _wave_dev_ema := 0.02     # seeded so the first few ticks aren't hyper-sensitive
var _wave_last_time := -100.0
var _wave_pos := PackedVector2Array()
var _wave_time := PackedFloat32Array()
var _wave_amp := PackedFloat32Array()
var _wave_slot := 0

var _waveform_pid := -1
var _waveform_path := ""         # set once we know it; polled in _process until it exists

# --- Hi-res waveform window (see _poll_wave_hi): the base strip is one
# 4096px showwavespic over the WHOLE clip - ~4px/s on a long video, mush when
# the timeline is zoomed well in. When the on-screen pixels-per-second
# outresolves the base, the VISIBLE window (padded) is re-rendered at the
# same recipe on an audio slice, async and debounced, and the timeline draws
# it over the base for that span. Progressive: the base shows (soft) until
# the crisp slice lands.
const _WAVEHI_PATH := "user://wave_hi.png"
var _wavehi_pid := -1
var _wavehi_span := Vector2.ZERO   # [start, end] seconds the in-flight/loaded render covers
var _wavehi_want := Vector2.ZERO   # latest desired window - the debounce reference
var _wavehi_settle := 0.0          # seconds the desired window has held still
var _audio_env := PackedFloat32Array()   # per-column amplitude from the waveform image (resonance)

var _render_state := "idle"      # idle / rendering / transcoding / done
var _render_pid := -1
var _transcode_pid := -1
var _out := ""
var _avi := ""

# Auto-save: every edit marks the session dirty; it saves shortly after the last
# change in a burst (and unconditionally on close), so work persists across
# reloads without a save button and without writing once per slider-drag pixel.
var _dirty := false
var _syncing := false   # true while _refresh_panel repaints controls - _edit must ignore echoes
var _modal_depth := 0   # open popups (colour pickers, dropdowns) - suspends the cursor auto-hide
var _autosave_cooldown := 0.0
const _AUTOSAVE_DELAY := 0.4

# Undo/redo: whole-array snapshots of session.markers (small, plain data - cheap
# enough to duplicate wholesale, no need for a diff/command log). A slider drag
# or a marker being dragged along the timeline fires the same mutation dozens of
# times a second, so a drag has to fold into ONE entry or a single Ctrl+Z would
# barely nudge the value back.
#
# THE BOUNDARY IS THE MOUSE BUTTON, not a clock - see _push_undo. A 0.9s coalescing
# window was tried and it fails in both directions at once: pause mid-drag for
# longer than the window and the drag fragments ("every single place where I stop
# will be added to the command history"), make two separate adjustments inside it
# and they merge ("Ctrl+Z often reverts several changes at once"). Both were
# reported. A press opens one entry, the release closes it, and anything that is
# not a drag is its own step.
var _undo_stack: Array = []
var _redo_stack: Array = []
# Parallel to _undo_stack/_redo_stack - a human-readable description of the
# action each entry captured, so the panel can preview what Ctrl+Z would
# revert (see _push_undo/_refresh_history_label).
var _undo_descs: Array = []
var _redo_descs: Array = []
const _UNDO_LIMIT := 200
## The undo key that the CURRENT mouse-held interaction opened, or "" when the
## button is up. See _push_undo - this is what makes a drag one history entry
## instead of one per place the pointer paused.
var _undo_press_key := ""
var _select_generation := 0   # bumped on every selection change; folded into the coalesce key


func _ready() -> void:
	layer = 100
	_mat_main.shader = SHADER
	_mat_inset.shader = SHADER
	if not render_mode:
		add_to_group("mask_editor")   # so the Assistant can trigger a reload here (see _do_restart)
		# A render that was killed mid-flight leaves its override.cfg behind, and the
		# next launch would boot into the render's resolution. Harmless to remove even
		# while a render IS running: Godot reads it once, at that process's startup.
		_clear_render_override()
		_build_status_label()   # built up front - prep needs it before a session exists
		_warn_missing_tools()


## Masking is the mode that CANNOT work without ffmpeg: every clip is transcoded
## on import, every waveform and thumbnail is an ffmpeg pass, and ffprobe answers
## the frame count. Say so at the door rather than letting the first import fail
## with an unexplained "prep did not finish". Two resolves, no version probe, so
## this costs nothing - the home screen's panel is where the full report lives, and
## this mode can be launched straight from the CLI without ever seeing it.
func _warn_missing_tools() -> void:
	var missing: PackedStringArray = []
	for prog in ["ffmpeg", "ffprobe"]:
		if not Deps.has(prog):
			missing.append(prog)
	if missing.is_empty():
		return
	_set_status("⚠  %s not found - clips cannot be imported or rendered.  %s"
		% [" and ".join(missing), Deps.hint("ffmpeg")])


## `path` is either a prepared session .json, a raw source video (transcoded once
## and cached under masks/<slug>/), or empty (prompt via a native file dialog).
func open_source(path: String) -> void:
	if path.is_empty():
		_prompt_for_source()
		return
	if _is_url(path):
		_start_url_import(path)   # downloads, then re-enters here with the local file
		return
	if path.get_extension().to_lower() == "json":
		_session_path = path
		session = MaskSession.load(path if path.begins_with("res://") else path)
		if session != null:
			_ready_with_session()
		else:
			_set_status("⚠  Could not read session: " + path)
		return
	var slug := _slugify(path)
	var dir := MASKS_DIR + "/" + slug
	var video := dir + "/video.ogv"
	var audio := dir + "/audio.wav"
	_session_path = dir + "/session.json"
	_pending = {"source": path, "dir": dir, "video": video, "audio": audio}
	# Check for a live prep BEFORE trusting anything cached on disk - a file that
	# merely EXISTS may just be mid-write (this is why the real clip's session ended
	# up with duration 0.0: it was opened while an external transcode was still
	# appending to video.ogv, and "exists" isn't "finished"). See _finish_session for
	# the matching duration>0 validation on the fast paths below.
	if _prep_looks_live(dir, video):
		_prep_state = "waiting_external"
		_set_status("⏳  Preparing clip (already in progress elsewhere)…")
		return
	var abs_session := ProjectSettings.globalize_path(_session_path)
	if FileAccess.file_exists(abs_session):
		var loaded := MaskSession.load(abs_session)
		if loaded != null and loaded.duration > 0.0:
			session = loaded
			_ready_with_session()
			return
	var abs_video := ProjectSettings.globalize_path(video)
	var abs_audio := ProjectSettings.globalize_path(audio)
	if FileAccess.file_exists(abs_video) and FileAccess.file_exists(abs_audio):
		_finish_session(path, video, audio)   # validates duration itself; re-preps if bad
		return
	_prep(path, dir, video, audio)


func _prompt_for_source() -> void:
	_open_dialog = FileDialog.new()
	_open_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_open_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_open_dialog.use_native_dialog = true
	_open_dialog.title = "Open a clip for mask mode"
	_open_dialog.filters = PackedStringArray(["*.mp4,*.mov,*.mkv,*.webm ; Video"])
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_open_dialog.current_dir = downloads
	_open_dialog.size = Vector2i(800, 560)
	_open_dialog.file_selected.connect(open_source)
	add_child(_open_dialog)
	_open_dialog.popup_centered()


# --- one-time prep: ffmpeg -> masks/<slug>/{video.ogv, audio.wav} -----------------
# Two real ffmpeg processes (not a shell chain) so each can be polled and reported
# like exporter.gd's bake/render/transcode steps: video first (the slow part - can
# be minutes on a long clip), then audio (fast, PCM decode). Progress comes from the
# same `-progress <file>` mechanism the export transcode step already uses.

func _prep(source: String, dir: String, video: String, audio: String) -> void:
	var abs_dir := ProjectSettings.globalize_path(dir)
	DirAccess.make_dir_recursive_absolute(abs_dir)
	_prep_src_dur = _probe_duration(source)
	_progress_reset(_PREP_PROGRESS_FILE)
	# Small GOP (-g 25 = ~1 keyframe/sec at typical fps) keeps scrubbing responsive on
	# long clips; theora is the only format VideoStreamPlayer decodes natively (see
	# next/mask_mode_spike notes - WebM/H.264 need a GDExtension, so every source
	# clip gets this one-time transcode regardless of its original codec).
	# Written as .part and promoted on completion (see _promote_part): a
	# half-written media file at its REAL name is scanned by the Godot editor's
	# importer, and a truncated WAV wedges it in an infinite seek loop -
	# hanging every editor boot AND every headless compile check (including
	# the assistant's reload gate) until the file is deleted by hand.
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", source, "-an",
		"-c:v", "libtheora", "-q:v", "6", "-g", "25",
		"-progress", ProjectSettings.globalize_path(_PREP_PROGRESS_FILE), "-nostats",
		"-f", "ogg", ProjectSettings.globalize_path(video) + ".part"])
	_prep_video_pid = Subprocess.start("ffmpeg", args, "clip prep (video)")
	_prep_state = "prepping_video" if _prep_video_pid > 0 else "idle"
	if _prep_video_pid <= 0:
		_set_status("⚠  Could not start ffmpeg (is it on PATH?)")
	else:
		# In the log too: for a long clip THIS is the slow stage (libtheora is
		# single-threaded), and the console should say what's grinding.
		print("ghost mask: prep started - transcoding %s (%.0fs) -> %s" % [
			source, _prep_src_dur, video])
		_touch_lock(abs_dir)
		_set_status("⏳  Preparing clip (video)…  0%")


func _start_prep_audio() -> void:
	_touch_lock(ProjectSettings.globalize_path(_pending.dir))   # bridge the gap while
	# video.ogv's mtime goes stale and audio.wav doesn't exist yet - see _prep_looks_live.
	# Two outputs, one decode: the PCM the waveform/envelope tooling needs,
	# and the compressed sidecar playback actually attaches (see
	# _ready_with_session for why the raw WAV is too slow to load live).
	var abs_a := ProjectSettings.globalize_path(String(_pending.audio))
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", String(_pending.source), "-vn",
		"-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
		"-f", "wav", abs_a + ".part",
		"-vn", "-c:a", "libvorbis", "-q:a", "5",
		"-f", "ogg", abs_a.get_basename() + ".ogg.part"])
	_prep_audio_pid = Subprocess.start("ffmpeg", args, "clip prep (audio)")
	_prep_state = "prepping_audio" if _prep_audio_pid > 0 else "idle"
	_set_status("⏳  Preparing clip (audio)…")


func _progress_reset(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string("")


# --- prep liveness: is SOMETHING actively writing this clip's output right now? ---
# Two writers on the SAME -y output path interleave and corrupt it - not
# hypothetical, this is here because it happened (twice: a second launch mid-prep
# raced the first, and separately a manual external ffmpeg run raced the app). The
# first version of this guard tracked a PID and called Subprocess.alive() on
# it - which crashes (engine-level ECHILD) the moment the PID belongs to a process
# this instance didn't spawn itself, e.g. a PREVIOUS launch's child, exactly the
# cross-instance case the guard exists for. An mtime check answers the actual
# question - "did anything touch this file recently" - without caring who the
# writer is or whether this instance could ever see its PID.
const _STALE_AFTER := 10.0   # seconds since last write before a prep counts as abandoned

func _lock_path(dir: String) -> String:
	return dir.path_join(".prep.lock")


## Move a finished prep output from its .part name into place. Outputs are
## written under a name the editor's importer ignores because a truncated
## media file at its REAL name is not just garbage, it's a trap: a killed
## prep left a half-written audio.wav whose WAV import looped forever on
## seeks past EOF (audio_stream_wav.cpp), hanging every editor scan until
## the file was hand-deleted. A missing .part (ffmpeg died at startup) is
## a no-op here - the duration validation downstream reports it.
func _promote_part(dest: String) -> void:
	var abs_dest := ProjectSettings.globalize_path(dest)
	if FileAccess.file_exists(abs_dest + ".part"):
		if FileAccess.file_exists(abs_dest):
			DirAccess.remove_absolute(abs_dest)
		DirAccess.rename_absolute(abs_dest + ".part", abs_dest)


func _touch_lock(abs_dir: String) -> void:
	var f := FileAccess.open(_lock_path(abs_dir), FileAccess.WRITE)
	if f != null:
		f.store_string(str(Time.get_unix_time_from_system()))


func _clear_lock(dir: String) -> void:
	var p := _lock_path(ProjectSettings.globalize_path(dir))
	if FileAccess.file_exists(p):
		DirAccess.remove_absolute(p)


func _fresh(path: String, now: float) -> bool:
	return FileAccess.file_exists(path) and now - FileAccess.get_modified_time(path) < _STALE_AFTER


## True if the lock file, the video output, or the audio output was modified within
## the last _STALE_AFTER seconds - covers both prep sub-steps (the lock's own touch
## bridges the gap before video.ogv exists yet; the outputs' own mtimes take over,
## and cover each other, for the rest).
func _prep_looks_live(dir: String, video: String) -> bool:
	var now := Time.get_unix_time_from_system()
	var abs_dir := ProjectSettings.globalize_path(dir)
	var abs_video := ProjectSettings.globalize_path(video)
	return _fresh(_lock_path(abs_dir), now) \
		or _fresh(abs_video, now) \
		or _fresh(abs_video + ".part", now) \
		or _fresh(abs_dir.path_join("audio.wav"), now) \
		or _fresh(abs_dir.path_join("audio.wav.part"), now)


## Same `out_time_us=` polling exporter.gd's transcode step uses, against the
## source clip's OWN duration (captured before the video step starts).
func _read_prep_pct() -> int:
	if _prep_src_dur <= 0.0 or not FileAccess.file_exists(_PREP_PROGRESS_FILE):
		return 0
	var text := FileAccess.get_file_as_string(_PREP_PROGRESS_FILE)
	var best := -1.0
	for line in text.split("\n"):
		if line.begins_with("out_time_us=") or line.begins_with("out_time_ms="):
			best = maxf(best, line.substr(12).to_float() / 1_000_000.0)
	if best < 0.0:
		return 0
	return clampi(int(round(best / _prep_src_dur * 100.0)), 0, 99)


func _finish_session(source: String, video: String, audio: String) -> void:
	var abs_video := ProjectSettings.globalize_path(video)
	var dur := _probe_duration(abs_video)
	# A theora/ogg file caught mid-write can still probe successfully - ffmpeg
	# doesn't flush on every frame, so a writer can look "stale" by mtime alone
	# during a buffering gap and still be actively growing. duration <= 0.0 catches
	# a totally unparseable file; this catches the sneakier case where it parses
	# fine but is truncated (this is exactly how a real 16:22 clip got cached at
	# 2:14 - the audio step's own duration is ground truth, extracted whole, so a
	# video duration meaningfully short of it means "not actually done yet", not
	# "shorter clip").
	var audio_dur := _probe_duration(ProjectSettings.globalize_path(audio))
	var incomplete := dur <= 0.0 or (audio_dur > 0.0 and dur < audio_dur - 2.0)
	if incomplete:
		var dir := video.get_base_dir()
		if _prep_looks_live(dir, video):
			_prep_state = "waiting_external"
			_set_status("⏳  Preparing clip (already in progress elsewhere)…")
			return
		_set_status("⚠  Cached clip looked incomplete - re-preparing…")
		_prep(source, dir, video, audio)
		return
	session = MaskSession.new()
	session.source_path = source
	session.video_path = video
	session.audio_path = audio
	session.duration = dur
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_session_path.get_base_dir()))
	session.save(ProjectSettings.globalize_path(_session_path))
	print("ghost mask: session ready - %s (%.0fs)" % [_session_path, dur])
	_ready_with_session()


## Robust video-duration probe. The fast path is the container's own duration,
## but our own libtheora transcode (see _start_track_import) routinely produces an
## .ogv whose header carries NO duration - ffprobe returns "N/A" for both
## format=duration AND stream=duration. That returned 0 here, and _finish_track_import
## reads 0 as "transcode failed" and silently adds no lane (the "Track import does
## nothing" bug). So when both are absent, fall back to counting video packets (fast,
## no decode) and dividing by the frame rate - every playable file has a definite
## packet count even when it lacks a duration field.
func _probe_duration(path: String) -> float:
	var d := _ffprobe_float(["-show_entries", "format=duration"], path)
	if d > 0.0:
		return d
	d = _ffprobe_float(["-select_streams", "v:0", "-show_entries", "stream=duration"], path)
	if d > 0.0:
		return d
	var out := []
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


## One ffprobe query for a single float value (csv, no keys), 0.0 if it comes back
## empty or "N/A". See _probe_duration for why the caller layers several of these.
func _ffprobe_float(entries: Array, path: String) -> float:
	var args := PackedStringArray(["-v", "error"])
	for e in entries:
		args.append(String(e))
	args.append("-of")
	args.append("csv=p=0")
	args.append(path)
	var out := []
	Deps.execute("ffprobe", args, out)
	if out.size() > 0:
		var s := String(out[0]).strip_edges()
		if s.is_valid_float():
			return s.to_float()
	return 0.0


## The clip's frame size in pixels, or a zero vector when ffprobe can't say.
## `csv=p=0:s=x` makes the whole answer one token - "1080x1920".
func _probe_size(path: String) -> Vector2i:
	var out := []
	Deps.execute("ffprobe", ["-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path], out)
	if out.size() > 0:
		var parts := String(out[0]).strip_edges().split("x")
		if parts.size() >= 2 and parts[0].is_valid_int() and parts[1].is_valid_int():
			var sz := Vector2i(parts[0].to_int(), parts[1].to_int())
			if sz.x > 0 and sz.y > 0:
				return sz
	return Vector2i.ZERO


## The picture's width/height - >1 landscape, <1 portrait. See _src_size.
func _source_aspect() -> float:
	return float(maxi(1, _src_size.x)) / float(maxi(1, _src_size.y))


## Seed _src_size from the prepared clip on disk, before anything has decoded.
## Silently keeps the 16:9 fallback if ffprobe is absent or unhelpful -
## _sync_source_size corrects it from the real texture a frame or two later.
func _resolve_source_size() -> void:
	if session == null or session.video_path.is_empty():
		return
	var sz := _probe_size(ProjectSettings.globalize_path(session.video_path))
	if sz != Vector2i.ZERO:
		_src_size = sz


## The decoded texture's size is the truth - adopt it the first frame it exists
## and re-fit the editor's video slot if the probe (or the fallback) disagreed.
## One-shot: after the first real frame the size never changes for a clip.
func _sync_source_size() -> void:
	if _src_size_confirmed or _player == null:
		return
	var tex := _player.get_video_texture()
	if tex == null or tex.get_width() <= 0 or tex.get_height() <= 0:
		return
	_src_size_confirmed = true
	var sz := Vector2i(tex.get_width(), tex.get_height())
	if sz == _src_size:
		return
	_src_size = sz
	if _video_area != null:
		_video_area.ratio = _source_aspect()


static func _slugify(path: String) -> String:
	var base := path.get_file().get_basename().to_lower()
	var out := ""
	for c in base:
		out += c if (c >= "a" and c <= "z") or (c >= "0" and c <= "9") else "_"
	while out.contains("__"):
		out = out.replace("__", "_")
	return out.trim_suffix("_").substr(0, 32)


# --- YouTube / URL import: paste a URL where a file path is expected --------------
# open_source() routes any http(s) source here. The pipeline is three polled
# subprocesses (the same _process pattern as prep - never blocking):
#   1. python3 -m venv user://ytdlp_venv        (once, on the first ever URL import)
#   2. <venv>/bin/pip install --upgrade yt-dlp  (once; also the retry step)
#   3. <venv>/bin/yt-dlp <url>  ->  masks/_downloads/<title>_<id>.<ext>
# then the downloaded file re-enters open_source() and the normal ffmpeg->theora
# prep takes over unchanged. The venv is DEDICATED so ghost owns its own yt-dlp
# and can upgrade it without touching anything else - which matters because
# YouTube breaking old yt-dlp versions is the norm, not the exception: a failed
# download automatically re-runs step 2 once and retries.

static func _is_url(path: String) -> bool:
	return path.begins_with("http://") or path.begins_with("https://")


## The video id, for cache hits and output matching (ported from nutube's
## YouTubeSource.id_from_url). "" for URLs that don't look like YouTube - yt-dlp
## handles plenty of other sites; those just skip the id-keyed cache.
static func _youtube_id(url: String) -> String:
	var u := url.strip_edges()
	if u.contains("watch?v="):
		u = u.get_slice("watch?v=", 1)
	elif u.contains("youtu.be/"):
		u = u.get_slice("youtu.be/", 1)
	elif u.contains("/shorts/"):
		u = u.get_slice("/shorts/", 1)
	else:
		return ""
	for sep in ["?", "&", "#", "/"]:
		u = u.get_slice(sep, 0)
	return u


func _yt_bin(tool_name: String) -> String:
	return Deps.venv_bin(YT_VENV_DIR, tool_name)


func _yt_dl_dir() -> String:
	return ProjectSettings.globalize_path(YT_DL_DIR)


## Resolve a binary. [Deps] owns this for the whole app - a GUI-launched Godot
## doesn't inherit a shell PATH, and the answer has to be the same one
## [Subprocess] and the home screen's Environment panel get.
static func _which(prog: String) -> String:
	return Deps.resolve(prog)


## The system interpreter both venv bootstraps need, named the way each platform
## names it ("python3" does not exist on Windows).
static func _python() -> String:
	return Deps.resolve_any(["python", "python3", "py"] if OS.get_name() == "Windows"
		else ["python3", "python"])


func _start_url_import(url: String) -> void:
	if _yt_state != "idle":
		_set_status("⏳  Another download is already running…")
		return
	_yt_url = url
	_yt_retried = false
	_yt_echoed = {}
	DirAccess.make_dir_recursive_absolute(_yt_dl_dir())
	print("ghost yt: import requested - ", url)
	var cached := _yt_find_download()
	if not cached.is_empty():
		# Says so in the log too - otherwise a cache hit runs NO python at all
		# and an empty console reads as "capture is broken", not "nothing ran".
		print("ghost yt: using cached download - ", cached)
		_set_status("✓  Already downloaded - opening the cached copy")
		open_source(cached)
		return
	if FileAccess.file_exists(_yt_bin("yt-dlp")):
		_yt_download()
	else:
		_yt_make_venv()


func _yt_make_venv() -> void:
	var py := _python()
	if py.is_empty():
		_set_status("⚠  Python 3 is not installed - can't bootstrap yt-dlp.  "
			+ Deps.hint("python"))
		return
	print("ghost yt: bootstrapping venv at ", ProjectSettings.globalize_path(YT_VENV_DIR),
		" with ", py)
	_yt_pid = _yt_spawn_logged(py, PackedStringArray(
		["-m", "venv", ProjectSettings.globalize_path(YT_VENV_DIR)]))
	_yt_state = "venv" if _yt_pid > 0 else "idle"
	_yt_step_started = Time.get_unix_time_from_system()
	_set_status("⏳  Setting up ghost's download venv (one-time)…" if _yt_pid > 0
		else "⚠  Could not start python3")


func _yt_pip_install() -> void:
	print("ghost yt: pip install --upgrade yt-dlp")
	_yt_pid = _yt_spawn_logged(_yt_bin("pip"), PackedStringArray(
		["install", "--upgrade", "yt-dlp"]))
	_yt_state = "pip" if _yt_pid > 0 else "idle"
	_yt_step_started = Time.get_unix_time_from_system()
	_set_status("⏳  Installing yt-dlp into the venv…" if _yt_pid > 0
		else "⚠  Could not start pip (venv incomplete?)")


func _yt_download() -> void:
	_yt_started = Time.get_unix_time_from_system()
	_yt_step_started = _yt_started
	# No codec constraints in the format pick - the prep transcodes to theora
	# regardless, so "best up to 1080p" is all that matters. --restrict-filenames
	# keeps the output shell-and-slug-safe; the title lands in the session slug.
	# Runs as `python -u -m yt_dlp`, NOT the yt-dlp entry script: with stdout
	# going to a file, python block-buffers it and the "[download] 42%" lines
	# only landed in the log kilobytes at a time - the progress readout below
	# read an empty file for minutes (the "black screen" report). -u makes the
	# log stream line-by-line, same reason exporter's ffmpeg uses -progress.
	var args := PackedStringArray([
		"-u", "-m", "yt_dlp",
		"-f", "bv*[height<=1080]+ba/b[height<=1080]",
		"--no-playlist", "--restrict-filenames", "--newline",
		# Under 100K/s yt-dlp assumes YouTube's anti-bot throttle and re-extracts
		# instead of crawling for an hour; fragmented formats also parallelize.
		"--throttled-rate", "100K", "--concurrent-fragments", "4"])
	# YouTube's nsig/PO-token challenges need a JavaScript runtime. yt-dlp's
	# default is deno, which is rarely installed - and with NO runtime the
	# download still "works", just at the punitive fallback throttle (~40KB/s:
	# the "0% to 1% took three minutes" report). Hand it whichever runtime this
	# machine actually has, by explicit path - a GUI-launched Godot's PATH is
	# not a shell's, so auto-detection can't be trusted either.
	for rt in ["deno", "node"]:
		var rt_bin := _which(rt)
		if not rt_bin.is_empty():
			args.append("--js-runtimes")
			args.append(rt + ":" + rt_bin)
	args.append_array(PackedStringArray([
		"-o", _yt_dl_dir().path_join("%(title).40s_%(id)s.%(ext)s"),
		_yt_url]))
	# The exact command, in the log: whether --js-runtimes was engaged is the
	# first thing to check when a download crawls (no runtime = YouTube's
	# ~40KB/s anti-bot throttle).
	print("ghost yt: exec ", _yt_bin("python"), " ", " ".join(args))
	_yt_pid = _yt_spawn_logged(_yt_bin("python"), args)
	_yt_state = "downloading" if _yt_pid > 0 else "idle"
	# Status lands IMMEDIATELY, success or not - the first cut only reported
	# failure here, so a clean start showed nothing until the download finished.
	_set_status("⏳  Downloading…  connecting" if _yt_pid > 0
		else "⚠  Could not start yt-dlp")


## OS.create_process can't redirect output, and both the progress readout and any
## failure diagnosis live in it - so the command runs through bash with stdout+
## stderr sent to a log file this instance then tails. The command and its
## arguments travel as REAL argv entries ("$@"), never interpolated into the
## script string (assistant.gd's prompt-as-$1 discipline) - a URL is data.
func _yt_spawn_logged(cmd: String, args: PackedStringArray) -> int:
	_yt_log = _yt_dl_dir().path_join(".yt.log")
	var script := "exec \"$@\" > \"%s\" 2>&1" % _yt_log
	var full := PackedStringArray(["-c", script, "bash", cmd])
	full.append_array(args)
	return Subprocess.start("/bin/bash", full, "clip download")


## The log's last 4KB. This is polled EVERY FRAME while a download runs (the
## same cadence _read_prep_pct polls ffmpeg's progress file), and a long
## --newline download writes tens of thousands of lines - reading the whole
## file each frame would grow linearly with the download. The freshest
## progress line always lives in the tail window.
func _yt_tail_window() -> String:
	if _yt_log.is_empty() or not FileAccess.file_exists(_yt_log):
		return ""
	var f := FileAccess.open(_yt_log, FileAccess.READ)
	if f == null:
		return ""
	var sz := f.get_length()
	if sz > 4096:
		f.seek(sz - 4096)
	return f.get_buffer(mini(int(sz), 4096)).get_string_from_utf8()


## The freshest yt-dlp "[download]  42.3% of 10.5MiB at 2.1MiB/s ETA 00:04"
## line, as display text - percent, size, speed and ETA ride along verbatim.
func _yt_pct() -> String:
	var best := ""
	for line in _yt_tail_window().split("\n"):
		if line.begins_with("[download]") and line.contains("%"):
			best = line.substr(10).strip_edges()
	return best


func _yt_log_tail() -> String:
	var lines := _yt_tail_window().strip_edges().split("\n")
	return String(lines[lines.size() - 1]).substr(0, 160) if lines.size() > 0 else ""


## The downloaded file for the current URL: id-matched when the URL is YouTube,
## else the newest completed media file since the download began. Excludes
## yt-dlp's own .part/.ytdl intermediates and the log.
func _yt_find_download() -> String:
	var dir := DirAccess.open(_yt_dl_dir())
	if dir == null:
		return ""
	var id := _youtube_id(_yt_url)
	var best := ""
	var best_mtime := 0
	for f in dir.get_files():
		if f.ends_with(".part") or f.ends_with(".ytdl") or f.begins_with("."):
			continue
		var p := _yt_dl_dir().path_join(f)
		if not id.is_empty():
			if f.contains(id):
				return p
			continue
		var mt := FileAccess.get_modified_time(p)
		if mt >= int(_yt_started) and mt > best_mtime:
			best_mtime = mt
			best = p
	return best


## Everything a finished step logged - minus the [download] progress spam -
## echoes into godot's own log via print(), which is what the in-app console
## (chrome's >_ toggle) tails. yt-dlp's diagnostics (throttling, missing JS
## runtime, geo blocks, format errors) otherwise die unseen in
## masks/_downloads/.yt.log, unreadable from a compiled app.
func _yt_echo_log(label: String) -> void:
	if _yt_log.is_empty() or not FileAccess.file_exists(_yt_log):
		return
	var kept := PackedStringArray()
	for line in FileAccess.get_file_as_string(_yt_log).split("\n"):
		var s := line.strip_edges()
		if not s.is_empty() and not s.begins_with("[download]"):
			kept.append(s)
	if kept.size() > 40:
		kept = kept.slice(kept.size() - 40)
	if kept.size() > 0:
		print("ghost yt [", label, "]:\n  ", "\n  ".join(kept))


## One venv/pip/download step just exited (see _process) - advance the machine.
## OS.create_process gives no exit code, so success is judged by what the step
## was supposed to produce (the binary, the file) - the same evidence-over-
## status-code stance as _finish_session's duration check.
func _yt_step_done() -> void:
	var state := _yt_state
	_yt_state = "idle"
	_yt_echo_log(state)
	match state:
		"venv":
			if FileAccess.file_exists(_yt_bin("pip")):
				_yt_pip_install()
			else:
				_set_status("⚠  venv bootstrap failed: " + _yt_log_tail())
		"pip":
			if FileAccess.file_exists(_yt_bin("yt-dlp")):
				_yt_download()
			else:
				_set_status("⚠  pip install yt-dlp failed: " + _yt_log_tail())
		"downloading":
			var file := _yt_find_download()
			if not file.is_empty():
				_set_status("✓  Downloaded " + file.get_file())
				open_source(file)
			elif not _yt_retried:
				# The single most common failure mode is a stale yt-dlp
				# (YouTube changed, old extractor broke) - upgrade once, retry.
				_yt_retried = true
				_set_status("⚠  Download failed - upgrading yt-dlp and retrying…")
				_yt_pip_install()
			else:
				_set_status("⚠  Download failed: " + _yt_log_tail())


# --- session ready: build preview + (unless render_mode) the editing UI -----------

func _ready_with_session() -> void:
	if _status != null:
		_status.visible = false
	_player = VideoStreamPlayer.new()
	_player.stream = load(ProjectSettings.globalize_path(session.video_path))
	_player.expand = true
	# Before any layout is built: the editor's video slot is shaped by this, and so
	# is the export's recorded resolution (see _src_size).
	_resolve_source_size()
	# No .material here, ever - the fx layers carry the shader (see
	# _build_video_composition). Which layer combination is shown is a per-marker
	# field, applied every frame in _process identically whether this is the live
	# editor or a render_mode export relaunch - export renders exactly what the
	# timeline says, not a hardcoded "always full masked" look.

	_audio = AudioStreamPlayer.new()
	# Through the effects bus from the start, not switched over later: the bus is
	# neutral until an audio marker says otherwise, and re-routing a playing stream
	# is an audible click.
	_ensure_audio_bus()
	_audio.bus = MASK_BUS
	add_child(_audio)
	# The main audio is a big uncompressed WAV (~170MB / ~3s to read). Loading it on
	# the main thread froze startup for that whole time. Live: load it on a worker and
	# attach it the moment it's ready (_process polls the thread), so the video + UI
	# come up instantly and audio joins a beat later, synced to the current position.
	# Export (render_mode) still loads synchronously - it must have audio from frame 0,
	# deterministically. AudioStreamWAV's static loader is the runtime-safe path (plain
	# load() has no loader for a raw .wav outside the import pipeline).
	var abs_audio := ProjectSettings.globalize_path(session.audio_path)
	# THE COMPRESSED SIDECAR IS THE FAST PATH. audio.wav is raw PCM - about
	# 10 MB per minute, so a 40-minute clip is ~420 MB and takes SECONDS to
	# read however it is threaded. Vorbis decodes lazily, so the same audio
	# as a ~30 MB .ogg attaches effectively instantly. Prep writes both now
	# (see _start_prep_audio); sessions prepared before that derive theirs
	# once, in the background, and are instant from the next open on. The WAV
	# stays: ffmpeg reads it for the waveform strip and the resonance
	# envelope. Same sidecar idea the imported tracks already use.
	var abs_ogg := abs_audio.get_basename() + ".ogg"
	if FileAccess.file_exists(abs_ogg):
		_audio.stream = AudioStreamOggVorbis.load_from_file(abs_ogg)
		_apply_main_volume()
		print("ghost mask: audio ready from sidecar ", abs_ogg.get_file())
	elif render_mode:
		_audio.stream = AudioStreamWAV.load_from_file(abs_audio)
		_apply_main_volume()
	else:
		_audio_thread = Thread.new()
		_audio_thread.start(_load_wav_threaded.bind(abs_audio))
		print("ghost mask: no audio sidecar yet - loading raw WAV (playback holds until it lands)")
		_ensure_audio_ogg(abs_audio, abs_ogg)   # so the NEXT open skips all this

	if render_mode:
		_build_render_view()
		# The resonance envelope must exist BEFORE the first recorded frame, or the
		# export's early frames disagree with the live preview. Nearly always cached
		# already (editing generated it); if not, block briefly on ffmpeg - an export
		# is a batch job, a one-time pause is fine where it wouldn't be live.
		_waveform_path = session.audio_path.get_base_dir().path_join("waveform_sqrt.png")
		var abs_wave := ProjectSettings.globalize_path(_waveform_path)
		if not FileAccess.file_exists(abs_wave):
			Deps.execute("ffmpeg", _waveform_args(abs_wave))
		_load_waveform(abs_wave)
	else:
		_build_editor_ui()
		_ensure_waveform()
	# Export loaded its audio synchronously (it must have sound from frame 0), so it
	# can start immediately. Live loads the WAV on a worker thread: starting now would
	# advance the video (the master clock) while the audio is still unattached, and the
	# audio would then join wherever the video already got to - the intro skip on every
	# first run. Hold the autostart instead - show frame 1 paused - and let
	# _poll_audio_thread begin playback the moment the audio is ready, synced from 0.
	if render_mode or _audio_thread == null or not _audio_thread.is_started():
		_play(true)
	else:
		_autostart_pending = true
		_play(false)   # decode + show the first frame, but hold the clock at the start
	# Land back where the playhead was last time (persisted per session) - only live;
	# an export always starts at clip_in. Deferred to the first _process tick: a
	# VideoStreamPlayer won't accept a seek the same frame it starts playing.
	if not render_mode and session.playhead > 0.05:
		_pending_restore = session.playhead
	# SELECT WHATEVER GOVERNS THE PLAYHEAD, so the panel opens showing the
	# session's real settings.
	# Nothing did this before: _selected stayed null until you clicked a marker
	# or playback happened to cross one, and _refresh_panel falls back to
	# MaskSession.DEFAULTS when nothing is selected. So every control - the wall
	# colour swatch included - came up displaying a default while the marker sat
	# there holding the values you had set. Nothing was ever lost on disk; the
	# panel simply was not reading it. Re-picking the colour from that state
	# then wrote a fresh near-identical hue, which is why the stored value
	# drifted a little on every restart instead of staying put.
	if not render_mode and not session.markers.is_empty():
		var gov: Variant = _governing_marker(
			session.playhead if session.playhead > 0.0 else 0.0)
		if gov == null:
			gov = session.markers[0]   # playhead sits before the first marker
		_select_marker(gov)


## Kick off (or discover already-cached) the timeline's waveform image - fully
## decoupled from playback readiness. Generating it (an ffmpeg pass over the whole
## audio track) can take a couple seconds on a long clip; blocking session-ready on
# that would turn "instant" cache hits back into a wait, so this fires async and the
## timeline just draws nothing until _process() notices the PNG exists.
func _ensure_waveform() -> void:
	# "waveform_sqrt": the filename carries the rendering recipe, so clips whose
	# cache predates a recipe change regenerate instead of loading the stale look.
	# sqrt scaling lifts small amplitudes into visibility - a linear plot of
	# ordinary speech/music sat near the axis and was barely visible on the strip.
	# 4096px wide so it still resolves when the timeline is zoomed well in.
	_waveform_path = session.audio_path.get_base_dir().path_join("waveform_sqrt.png")
	var abs_wave := ProjectSettings.globalize_path(_waveform_path)
	if FileAccess.file_exists(abs_wave):
		_load_waveform(abs_wave)
		return
	_waveform_pid = Subprocess.start("ffmpeg", _waveform_args(abs_wave), "waveform")


func _waveform_args(abs_out: String) -> PackedStringArray:
	return PackedStringArray([
		"-y", "-loglevel", "error", "-i", ProjectSettings.globalize_path(session.audio_path),
		"-filter_complex", "showwavespic=s=4096x160:colors=white:scale=sqrt",
		"-frames:v", "1", abs_out])


## Load the waveform image once and derive BOTH consumers from it: the timeline's
## strip texture, and the resonance envelope (per-column occupancy = amplitude).
## One file, one recipe - so the wisps breathe with exactly the wave the user sees
## on the strip, and live/export read identical values (file-deterministic; no
## real-time analyzer to drift between the two).
func _load_waveform(abs_path: String) -> void:
	var img := Image.load_from_file(abs_path)
	if img == null:
		return
	if _timeline != null:
		_timeline.waveform_texture = ImageTexture.create_from_image(img)
	var w := img.get_width()
	var h := img.get_height()
	_audio_env = PackedFloat32Array()
	_audio_env.resize(w)
	for x in w:
		var count := 0
		for y in range(0, h, 2):    # every 2nd row - envelope precision, half the cost
			if img.get_pixel(x, y).a > 0.1:
				count += 1
		# showwavespic draws a symmetric column; its filled fraction IS the (sqrt-
		# scaled) amplitude. 0.9 headroom so full-scale audio still reaches ~1.0.
		_audio_env[x] = clampf(float(count) / (float(h) * 0.5 * 0.9), 0.0, 1.0)


## Keep the timeline's waveform crisp under zoom. Every frame: harvest a
## finished slice render, then decide whether the current view needs a new
## one - only when the view outresolves the base strip, only after the view
## has held still a beat (never mid-drag), and only one ffmpeg at a time.
## Deliberately regenerates on demand instead of pre-baking LODs: the slice
## render is a sub-second ffmpeg pass over a PCM window, and most sessions
## never zoom past the base strip at all.
func _poll_wave_hi(dt: float) -> void:
	if session == null or _timeline == null or _tview == null \
			or session.duration <= 0.0 or _waveform_path.is_empty():
		return
	if _wavehi_pid > 0 and not Subprocess.alive(_wavehi_pid):
		_wavehi_pid = -1
		var img := Image.load_from_file(ProjectSettings.globalize_path(_WAVEHI_PATH))
		if img != null:
			_timeline.wavehi_texture = ImageTexture.create_from_image(img)
			_timeline.wavehi_span = _wavehi_span
			_timeline.queue_redraw()
	var span := _tview.visible_span()
	if span <= 0.0 or _timeline.size.x <= 0.0:
		return
	var base_pps := 4096.0 / session.duration
	var pps := _timeline.size.x / span
	if pps <= base_pps * 1.4:
		return   # the base strip still resolves this zoom
	var pad := span * 0.5
	var want := Vector2(
		clampf(_tview.view_start - pad, 0.0, session.duration),
		clampf(_tview.view_start + span + pad, 0.0, session.duration))
	# Already covered at sufficient resolution? (The loaded slice may be wider
	# than the view - panning inside it costs nothing.)
	var cur: Vector2 = _timeline.wavehi_span
	if cur.y > cur.x and cur.x <= _tview.view_start + 0.001 \
			and cur.y >= _tview.view_start + span - 0.001 \
			and 4096.0 / (cur.y - cur.x) >= pps * 0.9:
		_wavehi_settle = 0.0
		return
	# Debounce: the desired window must hold still before ffmpeg fires, so a
	# zoom gesture spawns one render at its end, not one per wheel tick.
	if absf(want.x - _wavehi_want.x) > span * 0.05 or absf(want.y - _wavehi_want.y) > span * 0.05:
		_wavehi_want = want
		_wavehi_settle = 0.0
		return
	_wavehi_settle += dt
	if _wavehi_settle < 0.35 or _wavehi_pid > 0:
		return
	_wavehi_settle = 0.0
	_wavehi_span = want
	# -ss/-t BEFORE -i: input seeking on PCM is sample-exact and skips the
	# decode of everything outside the window.
	_wavehi_pid = Subprocess.start("ffmpeg", PackedStringArray([
		"-y", "-loglevel", "error",
		"-ss", str(want.x), "-t", str(want.y - want.x),
		"-i", ProjectSettings.globalize_path(session.audio_path),
		"-filter_complex", "showwavespic=s=4096x160:colors=white:scale=sqrt",
		"-frames:v", "1", ProjectSettings.globalize_path(_WAVEHI_PATH)]))


# --- the liquid paint sim (clown): ping-pong SubViewports ------------------------

func _ensure_paint_sim() -> void:
	if not _paint_vps.is_empty():
		return
	for i in 2:
		var vp := SubViewport.new()
		# Finer than the echo ring: the deposits now carry per-pixel feature
		# shape, and an eye socket is only a few texels wide at 256x144.
		vp.size = Vector2i(384, 216)
		vp.disable_3d = true
		# Float buffers: the field decays multiplicatively, and 8-bit
		# quantization makes low paint values stick instead of thinning out.
		vp.use_hdr_2d = true
		# The COAT lives in the alpha channel, and a non-transparent viewport
		# FORCES alpha to 1 on its render target - the coat then read as
		# "everywhere" and the paint covered the whole frame, background
		# included. This is state, not a picture: keep every channel intact.
		vp.transparent_bg = true
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
		var rect := ColorRect.new()
		rect.size = Vector2(384, 216)
		var m := ShaderMaterial.new()
		m.shader = PAINT_SIM_SHADER
		rect.material = m
		vp.add_child(rect)
		add_child(vp)
		_paint_vps.append(vp)
		_paint_rects.append(rect)


## One simulation step per rendered frame while a clown layer is live (the
## META mirror's bounded-feedback discipline): the ping viewport re-renders
## sampling the pong's texture, then the mask materials sample the ping.
## Stepped on PLAYBACK deltas - pause freezes the liquid, and a seek (or the
## first frame) snaps the field to the targets instead of streaming paint
## across the jump.
func _step_paint_sim() -> void:
	if not _clown_active or _player == null:
		return
	_ensure_paint_sim()
	var pos := _player.stream_position
	var dtp := pos - _paint_last_pos
	_paint_last_pos = pos
	var reset := _paint_reset or dtp < -0.05 or dtp > 0.5
	dtp = clampf(dtp, 0.0, 0.1)
	if dtp <= 0.0 and not reset:
		return   # paused/held: the liquid holds; the mats keep last frame's field
	var cm := _clown_model_now()
	var mat: ShaderMaterial = _paint_rects[_paint_ping].material
	mat.set_shader_parameter("u_prev", _paint_vps[1 - _paint_ping].get_texture())
	mat.set_shader_parameter("u_dt", dtp)
	mat.set_shader_parameter("u_reset", 1 if reset else 0)
	mat.set_shader_parameter("u_time", pos)
	var vt := _player.get_video_texture()
	if vt != null and vt.get_height() > 0:
		mat.set_shader_parameter("u_aspect", float(vt.get_width()) / float(vt.get_height()))
		# The frame itself: the deposits read their evidence straight out of
		# it (dark socket, red lip line, lit nose ridge) instead of stamping
		# shapes - see clown_paint.gdshader.
		mat.set_shader_parameter("u_frame", vt)
	mat.set_shader_parameter("u_face_lum", _face_lum_ema)
	mat.set_shader_parameter("u_face_red", _face_red_ema)
	mat.set_shader_parameter("u_evidence", _clown_evidence)
	# The contour stencil, and whether there is one. Kept explicit rather than
	# inferred from the texture: a default-black sampler and a genuinely empty
	# face read identically, and one of those must fall back to the ellipses.
	if _stencil_vp != null and _ft_state == "ready":
		mat.set_shader_parameter("u_stencil", _stencil_vp.get_texture())
		mat.set_shader_parameter("u_has_stencil", 1.0)
	else:
		mat.set_shader_parameter("u_has_stencil", 0.0)
	mat.set_shader_parameter("u_drip", _clown_drip)
	mat.set_shader_parameter("u_drip_w", _clown_drip_w)
	mat.set_shader_parameter("u_eye_smudge", _clown_smudge)
	mat.set_shader_parameter("u_drip_curve", _clown_drip_curve)
	mat.set_shader_parameter("u_coat_feather", _clown_feather)
	mat.set_shader_parameter("u_eye_l", cm.eye_l)
	mat.set_shader_parameter("u_eye_r", cm.eye_r)
	mat.set_shader_parameter("u_mouth", cm.mouth)
	mat.set_shader_parameter("u_nose", cm.nose)
	mat.set_shader_parameter("u_face_c", cm.face_c)
	mat.set_shader_parameter("u_face_r", _face_r_ema)
	mat.set_shader_parameter("u_eye_lr", cm.eye_lr)
	mat.set_shader_parameter("u_eye_rr", cm.eye_rr)
	mat.set_shader_parameter("u_mouth_r", cm.mouth_r)
	mat.set_shader_parameter("u_scale", _clown_fs)
	mat.set_shader_parameter("u_bleed", _clown_bleed)
	mat.set_shader_parameter("u_settle", _clown_settle)
	mat.set_shader_parameter("u_hollow", _clown_hollow)
	_paint_vps[_paint_ping].render_target_update_mode = SubViewport.UPDATE_ONCE
	for m2 in [_mat_main, _mat_inset]:
		m2.set_shader_parameter("u_clown_paint", _paint_vps[_paint_ping].get_texture())
	_paint_ping = 1 - _paint_ping
	_paint_reset = false


## UMBRA's field simulation host - the same ping-pong pair the clown's paint
## uses, and for the same two hard-won reasons: transparent_bg because the
## GUARD lives in the alpha channel and a non-transparent viewport FORCES
## alpha to 1 (which would permit the mass everywhere, including on her), and
## CLEAR_MODE_NEVER because the field IS the state.
func _ensure_umbra_sim() -> void:
	if not _umb_vps.is_empty():
		return
	for i in 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(384, 216)
		vp.disable_3d = true
		vp.use_hdr_2d = true
		vp.transparent_bg = true
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
		var rect := ColorRect.new()
		rect.size = Vector2(384, 216)
		var m := ShaderMaterial.new()
		m.shader = UMBRA_SIM_SHADER
		rect.material = m
		vp.add_child(rect)
		add_child(vp)
		_umb_vps.append(vp)
		_umb_rects.append(rect)


## One simulation step per rendered frame while an umbra layer is live.
## Stepped on PLAYBACK deltas, so pause freezes the mass mid-curl and the
## export relaunch traces the identical currents; a seek snaps the field to its
## targets rather than streaming smoke across the jump.
func _step_umbra_sim() -> void:
	if not _umbra_active or _player == null:
		return
	_ensure_umbra_sim()
	var pos := _player.stream_position
	var dtp := pos - _umb_last_pos
	_umb_last_pos = pos
	var reset := _umb_reset or dtp < -0.05 or dtp > 0.5
	dtp = clampf(dtp, 0.0, 0.1)
	if dtp <= 0.0 and not reset:
		return
	if not _umb_have or _umb_region_tex == null:
		return   # no verdict yet - leave the field alone rather than deposit noise
	var mat: ShaderMaterial = _umb_rects[_umb_ping].material
	mat.set_shader_parameter("u_prev", _umb_vps[1 - _umb_ping].get_texture())
	mat.set_shader_parameter("u_region", _umb_region_tex)
	mat.set_shader_parameter("u_dt", dtp)
	mat.set_shader_parameter("u_reset", 1 if reset else 0)
	mat.set_shader_parameter("u_time", pos)
	var vt := _player.get_video_texture()
	if vt != null and vt.get_height() > 0:
		mat.set_shader_parameter("u_aspect", float(vt.get_width()) / float(vt.get_height()))
	mat.set_shader_parameter("u_dir", _umb_dir_ema)
	mat.set_shader_parameter("u_loom", _umb_loom)
	mat.set_shader_parameter("u_rise", _umb_rise)
	mat.set_shader_parameter("u_roil", _umb_roil)
	mat.set_shader_parameter("u_cling", _umb_cling)
	mat.set_shader_parameter("u_wisp", _umb_wisp)
	# The silhouette transform - see umbra_field.gdshader's to_region().
	mat.set_shader_parameter("u_pivot", _umb_pivot)
	mat.set_shader_parameter("u_sil_scale", _umb_scale)
	mat.set_shader_parameter("u_pan", _umb_pan)
	mat.set_shader_parameter("u_eye_l", _umb_eye_l)
	mat.set_shader_parameter("u_eye_r", _umb_eye_r)
	mat.set_shader_parameter("u_eye_rad", _umb_eye_rad)
	mat.set_shader_parameter("u_eye_amt", _umb_eye_amt if _umb_eyes_ok else 0.0)
	_umb_vps[_umb_ping].render_target_update_mode = SubViewport.UPDATE_ONCE
	for m2 in [_mat_main, _mat_inset]:
		m2.set_shader_parameter("u_umbra_field", _umb_vps[_umb_ping].get_texture())
	_umb_ping = 1 - _umb_ping
	_umb_reset = false


## The umbra model's own capture tick. Same discipline as _maybe_capture_face:
## only during playback (or once when a scrub settles), never mid-drag - the
## readback is a synchronous GPU stall.
func _maybe_capture_umbra() -> void:
	if not _umbra_active or _player == null or session == null:
		return
	var pos := _player.stream_position
	if not _playing and absf(pos - _prev_pos) >= 0.0005:
		return
	var slot := int(pos / _UMBRA_INTERVAL)
	if slot == _umb_slot:
		return
	var tex := _player.get_video_texture()
	if tex == null:
		return
	var img := tex.get_image()
	if img == null or img.is_empty():
		return
	_umb_slot = slot
	_update_umbra_model(img)


## The audio envelope at clip-time `t` (0 when unavailable), lightly smoothed so
## the wisps swell rather than flicker frame-to-frame.
func _env_at(t: float) -> float:
	if _audio_env.is_empty() or session == null or session.duration <= 0.0:
		return 0.0
	var n := _audio_env.size()
	var i := clampi(int(t / session.duration * float(n)), 1, n - 2)
	return (_audio_env[i - 1] + _audio_env[i] + _audio_env[i + 1]) / 3.0


## Stacked layers in `parent`'s full rect, shared by both the live editor and
## the render_mode export so they composite identically:
##   _player      raw video, visible underneath everything while the main clip's
##                own kept range covers the current time
##   _cont_view   raw video, same full-rect slot as _player - visible instead of it
##                while a continuation track (see MaskSession.continuation_track_at)
##                owns the current time; never both at once
##   _fx_overlay  full-frame shaded copy of whichever of the above is active (its
##                own material) - the MAIN fx layer
##   _mask_wrap   the bordered inset holding _pip_view (its own material)
## Which layers show, and how strongly, comes from the per-frame AMOUNTS
## (MaskSession.mode_amounts, blended through ramp/damp windows by at_time) -
## applied every frame in _apply_frame_state. A layer's presence multiplies into
## its own material's intensities, which is why each has its own material: the
## inset can hold full fx while the main overlay is still fading in.
func _build_video_composition(parent: Control) -> void:
	_composition_parent = parent   # secondary tracks' PiP views land here too - see _build_track_view
	parent.add_child(_player)
	_player.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

	# Sits at the same full-rect slot as _player, right underneath it in the same
	# z-position - only one of the two is ever visible at once (see _process), so
	# _fx_overlay's shaded copy always has exactly one raw picture beneath it to draw.
	_cont_view = TextureRect.new()
	_cont_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_cont_view.stretch_mode = TextureRect.STRETCH_SCALE
	_cont_view.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_cont_view.visible = false
	parent.add_child(_cont_view)

	_fx_overlay = TextureRect.new()
	_fx_overlay.material = _mat_main
	_fx_overlay.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_fx_overlay.stretch_mode = TextureRect.STRETCH_SCALE
	_fx_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	parent.add_child(_fx_overlay)

	_mask_wrap = PanelContainer.new()
	var border := StyleBoxFlat.new()
	border.bg_color = Color(0, 0, 0, 0)
	border.set_border_width_all(2)
	border.border_color = Color(1.0, 1.0, 1.0, 0.85)
	_mask_wrap.add_theme_stylebox_override("panel", border)
	# The inset's placement is fixed (bottom-right corner box); only its
	# visibility/presence animates.
	_mask_wrap.anchor_left = 0.66
	_mask_wrap.anchor_top = 0.64
	_mask_wrap.anchor_right = 0.98
	_mask_wrap.anchor_bottom = 0.96
	parent.add_child(_mask_wrap)

	_pip_view = TextureRect.new()
	_pip_view.material = _mat_inset
	_pip_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_pip_view.stretch_mode = TextureRect.STRETCH_SCALE
	_mask_wrap.add_child(_pip_view)

	if not render_mode:
		_build_region_overlay(parent)   # never in an export - it is a tool, not a look


# --- the region box: drawn on, and dragged over, the video itself ---------------
# The panel edits scalars; a rectangle over a picture is not a scalar, and typing
# four numbers to find the edge of a wall is not authoring. So the box is drawn
# where it acts, with grab handles at the corners and edges, and every drag writes
# straight through _edit into the selected marker's reg_* fields - the same path
# as every slider, so undo/coalescing/autosave all work without knowing about it.
#
# It lives INSIDE the composition parent, which is exactly the video's own rect
# (the AspectRatioContainer's letterboxed slot - see _build_editor_ui), so
# converting between a mouse position and frame UV is a plain divide and stays
# correct on any clip shape, portrait included.

const _REGION_GRAB := 14.0     # px reach of a corner/edge handle
const _REGION_MIN := 0.02      # smallest box side, in UV, so it can't be lost
const _REGION_SNAP := 0.012    # UV distance within which an edge snaps to the frame's


func _build_region_overlay(parent: Control) -> void:
	_region_overlay = Control.new()
	_region_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_region_overlay.z_index = 20           # over the fx layers AND any track PiP
	_region_overlay.visible = false        # only while a region is actually on
	_region_overlay.mouse_filter = Control.MOUSE_FILTER_STOP
	_region_overlay.draw.connect(_draw_region_overlay)
	_region_overlay.gui_input.connect(_region_overlay_input)
	parent.add_child(_region_overlay)


## The selected marker's box in this overlay's pixels, or an empty rect if there
## is nothing to draw.
func _region_rect_px() -> Rect2:
	if _selected == null or _region_overlay == null:
		return Rect2()
	var sz := _region_overlay.size
	var x0: float = minf(float(_selected.get("reg_x0", 0.0)), float(_selected.get("reg_x1", 1.0)))
	var y0: float = minf(float(_selected.get("reg_y0", 0.0)), float(_selected.get("reg_y1", 1.0)))
	var x1: float = maxf(float(_selected.get("reg_x0", 0.0)), float(_selected.get("reg_x1", 1.0)))
	var y1: float = maxf(float(_selected.get("reg_y0", 0.0)), float(_selected.get("reg_y1", 1.0)))
	return Rect2(x0 * sz.x, y0 * sz.y, (x1 - x0) * sz.x, (y1 - y0) * sz.y)


func _draw_region_overlay() -> void:
	var r := _region_rect_px()
	if r.size.x <= 0.0 or r.size.y <= 0.0:
		return
	var c := _region_overlay
	# Dim what the layer will NOT reach, so the box reads as a window rather than
	# a decoration - four bands around it rather than a rect over it, because the
	# inside must stay a true view of the effect.
	var shade := Color(0, 0, 0, 0.34)
	var sz := c.size
	c.draw_rect(Rect2(0, 0, sz.x, r.position.y), shade)
	c.draw_rect(Rect2(0, r.end.y, sz.x, sz.y - r.end.y), shade)
	c.draw_rect(Rect2(0, r.position.y, r.position.x, r.size.y), shade)
	c.draw_rect(Rect2(r.end.x, r.position.y, sz.x - r.end.x, r.size.y), shade)
	# The border, twice: a dark line under a light one, so it stays visible over
	# both a blown-out wall and a black one.
	c.draw_rect(r, Color(0, 0, 0, 0.75), false, 3.0)
	c.draw_rect(r, Color(1, 1, 1, 0.95), false, 1.0)
	# Corner handles. Drawn as filled squares - a handle you can see is a handle
	# you can hit, and these are the ones the mouse actually looks for first.
	var h := _REGION_GRAB * 0.5
	for p in [r.position, Vector2(r.end.x, r.position.y), Vector2(r.position.x, r.end.y), r.end]:
		var box := Rect2(p - Vector2(h, h), Vector2(h * 2.0, h * 2.0))
		c.draw_rect(box, Color(0, 0, 0, 0.75))
		c.draw_rect(box.grow(-1.5), Color(1, 1, 1, 0.95))


## Which part of the box the mouse is over: "nw"/"ne"/"sw"/"se" corners, "n"/"s"/
## "e"/"w" edges, "move" inside, "" outside. Corners are tested first so the
## overlapping edge bands at a corner never win.
func _region_hit(pos: Vector2) -> String:
	var r := _region_rect_px()
	if r.size.x <= 0.0:
		return ""
	var g := _REGION_GRAB
	var near_l: bool = absf(pos.x - r.position.x) <= g
	var near_r: bool = absf(pos.x - r.end.x) <= g
	var near_t: bool = absf(pos.y - r.position.y) <= g
	var near_b: bool = absf(pos.y - r.end.y) <= g
	var in_x: bool = pos.x >= r.position.x - g and pos.x <= r.end.x + g
	var in_y: bool = pos.y >= r.position.y - g and pos.y <= r.end.y + g
	if in_x and in_y:
		if near_l and near_t: return "nw"
		if near_r and near_t: return "ne"
		if near_l and near_b: return "sw"
		if near_r and near_b: return "se"
		if near_l: return "w"
		if near_r: return "e"
		if near_t: return "n"
		if near_b: return "s"
	return "move" if r.has_point(pos) else ""


const _REGION_CURSORS := {
	"nw": Control.CURSOR_FDIAGSIZE, "se": Control.CURSOR_FDIAGSIZE,
	"ne": Control.CURSOR_BDIAGSIZE, "sw": Control.CURSOR_BDIAGSIZE,
	"n": Control.CURSOR_VSIZE, "s": Control.CURSOR_VSIZE,
	"w": Control.CURSOR_HSIZE, "e": Control.CURSOR_HSIZE,
	"move": Control.CURSOR_MOVE,
}


## The PRESS and hover feedback only. Everything after the press is handled in
## _input instead - see _region_drag_motion for why a drag cannot live here.
func _region_overlay_input(event: InputEvent) -> void:
	if _selected == null:
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		_region_drag = _region_hit(event.position)
		_region_drag_from = event.position
		_region_drag_box = _region_rect_px()
		# A press OUTSIDE the box is not ours - let it fall through to whatever
		# is underneath rather than swallowing every click on the video for as
		# long as a region exists.
		if _region_drag != "":
			_region_overlay.accept_event()
		return
	# Hover feedback: the handles have to advertise themselves or the box looks
	# like a drawing rather than a tool.
	if event is InputEventMouseMotion and _region_drag == "":
		var hit := _region_hit(event.position)
		_region_overlay.mouse_default_cursor_shape = _REGION_CURSORS.get(hit, Control.CURSOR_ARROW)


## A DRAG IN PROGRESS, from the editor's own _input rather than the overlay's
## gui_input. gui_input only delivers events that land ON the control, so a drag
## that leaves the video pane silently stops updating - and the single most
## natural gesture with this tool is shoving an edge PAST the frame border to
## mean "all the way to the edge". Through gui_input that gesture stopped at
## whatever pixel the cursor last crossed on its way out, leaving the box a
## fraction of a percent short of the border: a thin unpainted band of the very
## colour being removed, hugging the top of the picture. Reading the mouse from
## _input instead lets the drag continue past the edge and clamp cleanly at it.
func _region_drag_motion(event: InputEvent) -> void:
	if _region_drag == "" or _region_overlay == null:
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT \
			and not event.pressed:
		_region_drag = ""
		return
	if event is InputEventMouseMotion:
		_region_apply_drag(_region_overlay.get_local_mouse_position())


## Resolve a drag to new box corners and write them. Everything is computed from
## the box AS IT WAS AT MOUSE-DOWN plus the total delta since - never
## incrementally from the current box, which drifts and makes a slow drag land
## somewhere different from a fast one covering the same distance.
func _region_apply_drag(pos: Vector2) -> void:
	var sz := _region_overlay.size
	if sz.x <= 0.0 or sz.y <= 0.0:
		return
	var d := pos - _region_drag_from
	var b := _region_drag_box
	var x0 := b.position.x
	var y0 := b.position.y
	var x1 := b.end.x
	var y1 := b.end.y
	if _region_drag == "move":
		# Moving keeps the SIZE and stays inside the frame, so dragging a box
		# off the edge parks it against the edge instead of shrinking it.
		var mx: float = clampf(x0 + d.x, 0.0, sz.x - b.size.x)
		var my: float = clampf(y0 + d.y, 0.0, sz.y - b.size.y)
		x0 = mx; y0 = my; x1 = mx + b.size.x; y1 = my + b.size.y
	else:
		if _region_drag.contains("w"):
			x0 = clampf(x0 + d.x, 0.0, x1 - _REGION_MIN * sz.x)
		if _region_drag.contains("e"):
			x1 = clampf(x1 + d.x, x0 + _REGION_MIN * sz.x, sz.x)
		if _region_drag.contains("n"):
			y0 = clampf(y0 + d.y, 0.0, y1 - _REGION_MIN * sz.y)
		if _region_drag.contains("s"):
			y1 = clampf(y1 + d.y, y0 + _REGION_MIN * sz.y, sz.y)
	# SNAP TO THE FRAME. An edge released within a few pixels of the picture's own
	# border means the border - nobody drags a box to 0.4% off the top on purpose,
	# and the difference is not visible in the editor but IS visible in the result
	# as a hairline of the colour being removed. Exact 0/1 also lets the shader
	# skip that side's falloff entirely (see region_mask), so a flush edge paints
	# the very first row instead of fading into it.
	_edit("reg_x0", _snap_edge(x0 / sz.x))
	_edit("reg_y0", _snap_edge(y0 / sz.y))
	_edit("reg_x1", _snap_edge(x1 / sz.x))
	_edit("reg_y1", _snap_edge(y1 / sz.y))
	_region_overlay.queue_redraw()


## Pull a box edge onto the frame's border if it is nearly there. In UV, because
## that is what gets stored; the threshold is generous enough to be reachable by
## hand and far too small to swallow a deliberate placement.
static func _snap_edge(v: float) -> float:
	if v < _REGION_SNAP:
		return 0.0
	if v > 1.0 - _REGION_SNAP:
		return 1.0
	return clampf(v, 0.0, 1.0)


## The checkbox. ON seeds a box worth dragging (the middle of the frame) rather
## than the full frame, which would have invisible handles pinned in the corners
## and look broken. OFF restores the whole frame - the layer acts everywhere
## again, which is the same state every session had before regions existed.
func _on_region_toggled(on: bool) -> void:
	if _syncing:
		return
	if on:
		_edit("reg_x0", 0.15)
		_edit("reg_y0", 0.10)
		_edit("reg_x1", 0.85)
		_edit("reg_y1", 0.55)
	else:
		_edit("reg_x0", 0.0)
		_edit("reg_y0", 0.0)
		_edit("reg_x1", 1.0)
		_edit("reg_y1", 1.0)
	_refresh_panel()   # re-reads the box, the checkbox and the edge slider's visibility


## Show the box exactly when the selected marker actually has a region. Called
## from _refresh_panel (selection/undo/reload) and after every toggle.
func _sync_region_overlay() -> void:
	if _region_overlay == null:
		return
	var on := _selected != null and MaskSession.has_region(_selected)
	_region_overlay.visible = on
	# MOUSE_FILTER_IGNORE while hidden is belt-and-braces: an invisible Control
	# does not receive input anyway, but this one sits over the whole video and a
	# future change that leaves it visible-but-empty must not eat every click.
	_region_overlay.mouse_filter = Control.MOUSE_FILTER_STOP if on else Control.MOUSE_FILTER_IGNORE
	if on:
		_region_overlay.queue_redraw()


## A secondary track's own composited view: a raw (unshaded - the masking
## effects system keys off ONE frame's colors via the primary's shader chain,
## not built for a second independent source yet) picture-in-picture box. It
## takes over the SAME inset slot _mask_wrap occupies (bottom-right) rather
## than getting its own corner - there is only ever one PiP box on screen at
## a time; see _sync_tracks, which hides _mask_wrap for as long as a track's
## own box is showing (feedback/0056: a stray, always-empty second box used
## to sit at a separate default corner on top of the real one).
func _build_track_view(i: int) -> void:
	var track: Dictionary = session.tracks[i]
	var player := VideoStreamPlayer.new()
	player.stream = load(ProjectSettings.globalize_path(String(track.video_path)))
	player.expand = true
	_composition_parent.add_child(player)

	var wrap := Control.new()
	wrap.anchor_left = _mask_wrap.anchor_left
	wrap.anchor_top = _mask_wrap.anchor_top
	wrap.anchor_right = _mask_wrap.anchor_right
	wrap.anchor_bottom = _mask_wrap.anchor_bottom
	_composition_parent.add_child(wrap)

	var view := TextureRect.new()
	view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	view.stretch_mode = TextureRect.STRETCH_SCALE
	view.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	wrap.add_child(view)

	var border := StyleBoxFlat.new()
	border.bg_color = Color(0, 0, 0, 0)
	border.set_border_width_all(2)
	border.border_color = Color(0.6, 0.9, 1.0, 0.85)
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", border)
	panel.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	wrap.add_child(panel)

	# The track's audio plays through a real AudioStreamPlayer (the SAME proven path as
	# the main clip) - VideoStreamPlayer's own embedded audio does NOT play when the
	# video is driven by manual seeking, which is why the track was silent. We demux the
	# track .ogv's Vorbis into a sidecar .ogg once (fast stream-copy, cached), and mute
	# the video player's audio so nothing double-plays.
	player.volume = 0.0

	while _track_runtime.size() <= i:
		_track_runtime.append({})
	_track_runtime[i] = {"player": player, "view": view, "wrap": wrap, "audio": null, "active": false}
	player.paused = true
	player.play()   # must be playing before .stream_position can be set (see _sync_tracks)
	_ensure_track_audio(i)   # attaches .audio now (cached sidecar) or when ffmpeg finishes


## Give track `i` its AudioStreamPlayer, from a sidecar .ogg demuxed from the track's
## .ogv. If the sidecar already exists we attach immediately; otherwise ffmpeg runs in
## the BACKGROUND (was OS.execute - a synchronous main-thread stall that could hang the
## whole editor while audio kept playing) and _poll_track_audio attaches it on finish.
## The sidecar lives beside the .ogv (same basename, .ogg). If it exists, attach it.
## Otherwise extract it - preferring the ORIGINAL SOURCE, which always has the audio:
## the old -an transcode stripped it from the .ogv, so demuxing the .ogv gave nothing.
## Re-encode to Vorbis (the source is usually aac/mp3, not copyable to ogg). Background
## + quiet; if even the source has no audio, _poll_track_audio records no_audio so we
## don't retry forever. Once a sidecar exists it's cached and this is a no-op.
func _ensure_track_audio(i: int) -> void:
	if i < 0 or i >= session.tracks.size():
		return
	var track: Dictionary = session.tracks[i]
	var video_path := String(track.get("video_path", ""))
	if video_path.is_empty():
		return
	var abs_ogg := ProjectSettings.globalize_path(video_path.get_basename() + ".ogg")
	if FileAccess.file_exists(abs_ogg):
		_attach_track_audio(i, abs_ogg)
		return
	if bool(track.get("no_audio", false)):
		return   # already learned there's no audio anywhere for this track
	# Prefer the original import (has audio); fall back to the .ogv (only new imports
	# embed audio) when the source is gone or was never recorded.
	var src := String(track.get("source_path", ""))
	var abs_src := ProjectSettings.globalize_path(src) if src.begins_with("res://") else src
	if src.is_empty() or not FileAccess.file_exists(abs_src):
		abs_src = ProjectSettings.globalize_path(video_path)
	if not FileAccess.file_exists(abs_src):
		return
	var pid := Subprocess.start("ffmpeg", ["-y", "-loglevel", "quiet", "-i", abs_src,
		"-vn", "-c:a", "libvorbis", "-q:a", "4", abs_ogg])
	if pid > 0:
		_track_audio_jobs.append({"pid": pid, "index": i, "ogg": abs_ogg})


## Load a track's sidecar .ogg into an AudioStreamPlayer and hang it on the runtime.
func _attach_track_audio(i: int, abs_ogg: String) -> void:
	if i < 0 or i >= _track_runtime.size() or not (_track_runtime[i] is Dictionary):
		return
	var stream := AudioStreamOggVorbis.load_from_file(abs_ogg)
	if stream == null:
		return
	var ap := AudioStreamPlayer.new()
	ap.stream = stream
	add_child(ap)
	_track_runtime[i]["audio"] = ap


## Poll the background sidecar extractions; attach each track's audio as ffmpeg finishes.
func _poll_track_audio() -> void:
	for j in range(_track_audio_jobs.size() - 1, -1, -1):
		var job: Dictionary = _track_audio_jobs[j]
		if not Subprocess.alive(int(job.pid)):
			var idx := int(job.index)
			if FileAccess.file_exists(String(job.ogg)):
				_attach_track_audio(idx, String(job.ogg))
			elif idx >= 0 and idx < session.tracks.size():
				# ffmpeg produced nothing - this track has no audio stream. Record it so
				# we don't re-run (and re-fail) the extraction on every future load.
				session.tracks[idx]["no_audio"] = true
				_mark_dirty()
			_track_audio_jobs.remove_at(j)


# --- multi-track: trim lanes, import, playback sync ----------------------------

# Per-lane reserved height. MUST cover a TrackLane's own minimum (26px, see
# track_lane.gd) PLUS the _lanes_col VBox's inter-lane separation (2px), or the
# lane stack overflows its allocated rect and - drawing above the marker strip it
# abuts - spills over the timeline's top edge, clipping the marker flags and the
# playhead timestamp tag (worsening with each imported track). 26 + 2 = 28.
const _LANE_H := 28.0
# The volume knob every lane gets (see _volume_knob) is anchored top-left, offset_right
# 34 - a lane's own label must start past that plus a small gap, or the two draw on top
# of each other whenever the lane's own left edge sits near local x0 (the primary lane's
# offset is always 0, so this bites it every time - see feedback/0009).
const _LANE_LABEL_LEFT := 38.0

func _track_getter(i: int, field: String) -> Callable:
	return func(): return float(session.tracks[i].get(field, 0.0))


func _track_setter(i: int, field: String) -> Callable:
	return func(v): session.tracks[i][field] = v


## Rebuild every trim/track lane from session state (the primary clip's own trim
## block, plus one per session.tracks entry) and recompute the shared
## TimelineView's cached extent. Called after any STRUCTURAL change (import,
## delete, undo/redo) - never mid-drag; see TrackLane.drag_ended and
## TimelineView.refresh's own doc for why that distinction is load-bearing here.
func _refresh_lanes() -> void:
	if _lanes_col == null:
		return
	for c in _lanes_col.get_children():
		c.queue_free()

	var primary := TrackLane.new()
	primary.tview = _tview
	primary.label = String(session.video_path).get_file()
	primary.reserved_left = _LANE_LABEL_LEFT
	primary.color = Color(0.55, 0.75, 1.0)
	primary.movable = false
	primary.full_duration = session.duration
	primary.get_in = func(): return session.clip_in
	primary.set_in = func(v): session.clip_in = v
	primary.get_out = func(): return session.effective_clip_out()
	primary.set_out = func(v): session.clip_out = v
	primary.get_fade_in = func(): return session.main_fade_in
	primary.set_fade_in = func(v): session.main_fade_in = v
	primary.get_fade_out = func(): return session.main_fade_out
	primary.set_fade_out = func(v): session.main_fade_out = v
	primary.get_snap_targets = _snap_targets_for.bind(-1)
	primary.drag_started.connect(func(): _push_undo("", "trimmed the main clip"))
	primary.drag_ended.connect(func(): _tview.refresh(session))
	primary.changed.connect(_mark_dirty)
	_lanes_col.add_child(primary)
	# The main clip's own pull-rope volume (mirrors each track's), so its level in the
	# mix is set independently.
	primary.add_child(_volume_knob(
		func(): return float(session.main_volume),
		func(v): session.main_volume = v,
		Color(0.55, 0.75, 1.0), true))
	# Split, same corner/behavior as a secondary track's - the main clip is otherwise
	# just track -1 (see _snap_targets_for), so it gets the same cut at the playhead
	# (see _split_main). No delete button here: unlike an imported track, the primary
	# clip is never optional - there's always exactly one.
	var main_split := Button.new()
	main_split.text = "✂"
	main_split.custom_minimum_size = Vector2(20, 18)
	main_split.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	main_split.offset_left = -22
	main_split.offset_top = 2
	main_split.offset_right = -2
	main_split.offset_bottom = 20
	main_split.focus_mode = Control.FOCUS_NONE
	main_split.tooltip_text = "Split the main track at the playhead"
	main_split.pressed.connect(_split_main)
	primary.add_child(main_split)

	for i in session.tracks.size():
		var lane := TrackLane.new()
		lane.tview = _tview
		lane.label = String(session.tracks[i].get("video_path", "")).get_file()
		lane.reserved_left = _LANE_LABEL_LEFT
		lane.color = Color(0.6, 0.95, 0.7)
		lane.movable = true
		lane.full_duration = float(session.tracks[i].get("duration", 0.0))
		lane.get_in = _track_getter(i, "clip_in")
		lane.set_in = _track_setter(i, "clip_in")
		lane.get_out = _track_getter(i, "clip_out")
		lane.set_out = _track_setter(i, "clip_out")
		lane.get_offset = _track_getter(i, "offset")
		lane.set_offset = _track_setter(i, "offset")
		lane.get_fade_in = _track_getter(i, "fade_in")
		lane.set_fade_in = _track_setter(i, "fade_in")
		lane.get_fade_out = _track_getter(i, "fade_out")
		lane.set_fade_out = _track_setter(i, "fade_out")
		lane.get_snap_targets = _snap_targets_for.bind(i)
		lane.get_playhead = func(): return _player.stream_position if _player != null else 0.0
		lane.drag_started.connect(func(): _push_undo("", "trimmed a track"))
		lane.drag_ended.connect(func(): _tview.refresh(session))
		lane.changed.connect(_mark_dirty)
		_lanes_col.add_child(lane)
		# The delete button OVERLAYS the lane's own top-right corner rather than
		# sitting beside it in an HBoxContainer - an inline sibling would shrink
		# the lane's own width below the primary lane's/_timeline's, and every
		# lane's x_of() must span the exact same pixel width or their blocks
		# stop lining up against a shared second.
		var del := Button.new()
		del.text = "✕"
		del.custom_minimum_size = Vector2(20, 18)
		del.set_anchors_preset(Control.PRESET_TOP_RIGHT)
		del.offset_left = -22
		del.offset_top = 2
		del.offset_right = -2
		del.offset_bottom = 20
		del.focus_mode = Control.FOCUS_NONE
		del.tooltip_text = "Remove this track"
		var idx := i
		del.pressed.connect(func(): _delete_track(idx))
		lane.add_child(del)
		# Split, just left of delete - cuts this track in two at the playhead (see
		# _split_track). Same overlay approach as delete, for the same reason: an
		# inline sibling would shrink the lane's own width out from under the
		# shared timeline's pixel mapping.
		var split := Button.new()
		split.text = "✂"
		split.custom_minimum_size = Vector2(20, 18)
		split.set_anchors_preset(Control.PRESET_TOP_RIGHT)
		split.offset_left = -46
		split.offset_top = 2
		split.offset_right = -26
		split.offset_bottom = 20
		split.focus_mode = Control.FOCUS_NONE
		split.tooltip_text = "Split this track at the playhead"
		split.pressed.connect(func(): _split_track(idx))
		lane.add_child(split)
		# Pull-rope volume knob, on the LEFT of the lane so it never sits under the floating
		# assistant chat button (bottom-right, where the old right-side controls landed).
		# The track's own AudioStreamPlayer mixes with the main clip; _sync_tracks reads
		# `volume` every frame, so pulling it changes the level live.
		lane.add_child(_volume_knob(
			func(): return float(session.tracks[idx].get("volume", 1.0)),
			func(v): session.tracks[idx]["volume"] = v,
			Color(0.6, 0.95, 0.7)))

	# The lane stack sits directly above the marker strip, and the video's own
	# letterboxed slot (see _build_editor_ui) has to shrink to match - otherwise
	# an imported track (or several) grows this stack tall enough to push its
	# lanes/delete-buttons over the bottom of the video instead of the reserved
	# strip below it.
	_apply_lane_reserved(1 + session.tracks.size())
	_tview.refresh(session)


## Collapse the lane stack (and expand the video letterbox into the space that
## frees up) to fit exactly `count` rows. Called with the full track count
## right after any structural change (above), and every frame with however
## many lanes are actually on screen right now - once TrackLane starts hiding
## itself for clips scrolled entirely out of the current zoom/pan window, the
## fixed-for-the-whole-track-count reservation left a dead black band where
## those rows used to be instead of actually collapsing - see feedback/0023.
func _apply_lane_reserved(count: int) -> void:
	var reserved := 90.0 + float(count) * _LANE_H
	_lanes_col.offset_top = -reserved
	_lanes_col.offset_bottom = -90
	if _video_area != null:
		_video_area.offset_bottom = -reserved
	# THE EXPORT BUTTON RIDES ABOVE THE BOTTOM CHROME. Its offsets are the ones
	# every mode uses (-68/-28 from the bottom edge, matching assistant.gd's
	# toggle row), and in every other mode the bottom of the frame is empty - but
	# here it is the marker strip and the lane stack, so the button sat ON the
	# timeline. `reserved` is exactly the height of that furniture, so anchoring to
	# it keeps the button clear however many tracks are open.
	# THE WHOLE BOTTOM-RIGHT ROW RIDES ABOVE THE BOTTOM CHROME - this editor's own
	# export button AND the shared furniture (⤓ / 💬 / >_), all off ONE number, and
	# all with the SAME arithmetic. They were laid out two different ways at first,
	# which is why the row came out staggered rather than in a line: the shared
	# buttons sit at -28..-68 from the bottom edge, so anything joining that row has
	# to be -28 - inset .. -68 - inset and not "inset plus a margin".
	#
	# `reserved` is exactly what the marker strip and the trim/track lanes occupy,
	# and it is what the video area is inset by, so the row stays clear however many
	# tracks are open.
	var inset := reserved - 14.0
	if _export_btn != null:
		_export_btn.offset_bottom = -28.0 - inset
		_export_btn.offset_top = -68.0 - inset
	if _status != null:
		# ITS OWN LINE, ABOVE THE BUTTON ROW - see Exporter.set_bottom_inset for
		# why. The face-track line and the export progress both live here and both
		# outgrow the space beside four buttons.
		_status.offset_bottom = -74.0 - inset
		_status.offset_top = -110.0 - inset
		_status.offset_right = -28.0
	# SET EVERY FRAME, not once at build time. The suppression used to be done in
	# _build_export_ui, which runs before the chrome has joined its group in some
	# boot orders - so it silently found nothing, the shared button stayed live on
	# its higher CanvasLayer, and it went on covering this one and eating the click.
	var ch_i := _chrome()
	if ch_i != null:
		ch_i.bottom_inset = inset
		if ch_i.exporter != null:
			ch_i.exporter.suppressed = true


func _delete_track(i: int) -> void:
	if i < 0 or i >= session.tracks.size():
		return
	_push_undo("", "deleted a track")
	if i < _track_runtime.size():
		var rt: Dictionary = _track_runtime[i]
		if rt.has("player"):
			(rt.player as Node).queue_free()
		if rt.has("wrap"):
			(rt.wrap as Node).queue_free()
		if rt.has("audio") and rt.audio != null:
			(rt.audio as Node).queue_free()
		_track_runtime.remove_at(i)
	session.tracks.remove_at(i)
	_refresh_lanes()
	_mark_dirty()


## Cut track `i` into two independent lanes at the playhead, both still pointing
## at the SAME source video/audio (video_path is shared - _ensure_track_audio's
## sidecar .ogg is keyed off that path, so the new half finds the existing
## extraction and attaches instantly, no re-demux). The left half keeps this
## track's own identity and index (just trims its clip_out); the right half is
## a full duplicate of the fields - same clip_in..clip_out span, same volume -
## then re-pointed to start exactly at the split: its clip_in advances to the
## split's LOCAL position in the source, and its offset (where that lands on
## the MASTER timeline) is set to the playhead itself, so the two blocks sit
## edge-to-edge with no gap or overlap. From there each is an ordinary lane -
## TrackLane's own drag handles already shift/resize them independently.
## Appended at the END of session.tracks (never inserted at i+1): background
## sidecar-audio jobs (_track_audio_jobs) and _track_runtime are index-keyed,
## and an insert would silently misalign any in-flight job for a LATER track.
func _split_track(i: int) -> void:
	if i < 0 or i >= session.tracks.size():
		return
	var track: Dictionary = session.tracks[i]
	var offset := float(track.get("offset", 0.0))
	var cin := float(track.get("clip_in", 0.0))
	var cout := float(track.get("clip_out", 0.0))
	var master_t: float = _player.stream_position if _player != null else 0.0
	var split_local := master_t - offset + cin
	if split_local <= cin + 0.05 or split_local >= cout - 0.05:
		_set_status("✂  Move the playhead inside this track's span to split it there")
		return
	_push_undo("", "split a track")
	var right: Dictionary = track.duplicate()
	right["clip_in"] = split_local
	right["offset"] = master_t
	right["fade_in"] = 0.0
	track["clip_out"] = split_local
	track["fade_out"] = 0.0
	session.tracks.append(right)
	_reconcile_track_runtime()
	_refresh_lanes()
	_mark_dirty()


## Cut the MAIN clip in two at the playhead, same as _split_track but for the
## primary lane: the main clip has no `offset` of its own (master time and its
## source time are the same clock - see TrackLane._bounds), so the playhead
## position IS the split point directly, no offset arithmetic needed. The left
## half stays the main clip (just trims clip_out, same as dragging its own out
## handle inward). The right half can't stay "main" - there's only ever one -
## so it's appended to session.tracks as an ordinary track pointing at the same
## video_path/source_path, picking up at the split with its own offset. From
## there it's indistinguishable from any imported track: draggable, deletable,
## splittable again - the main clip is really just track -1 (see
## _snap_targets_for), and after a split its tail end is a track in fact, not
## just in spirit.
func _split_main() -> void:
	if _player == null:
		return
	var cin := session.clip_in
	var cout := session.effective_clip_out()
	var split_t: float = _player.stream_position
	if split_t <= cin + 0.05 or split_t >= cout - 0.05:
		_set_status("✂  Move the playhead inside the main clip's span to split it there")
		return
	_push_undo("", "split the main clip")
	var right := {
		"video_path": session.video_path,
		"source_path": session.source_path,
		"duration": session.duration,
		"clip_in": split_t,
		"clip_out": cout,
		"offset": split_t,
		"fade_in": 0.0,
		"fade_out": session.main_fade_out,
		"volume": session.main_volume,
	}
	session.clip_out = split_t
	session.main_fade_out = 0.0
	session.tracks.append(right)
	_reconcile_track_runtime()
	_refresh_lanes()
	_mark_dirty()


func _prompt_import_track() -> void:
	# Already open (double-tap of T) - just bring it forward, don't stack a
	# second dialog behind the first.
	if _import_dialog != null and is_instance_valid(_import_dialog):
		_import_dialog.popup_centered()
		return
	# An import is already transcoding - don't let a second one clobber the
	# first's pending state mid-flight (they share _import_pid/_import_pending).
	if _import_pid > 0:
		_set_status("⏳  A track is still importing - one at a time")
		return
	# Immediate feedback that T registered, BEFORE the dialog - if the dialog
	# itself somehow fails to show, at least the keypress isn't silent.
	_set_status("📁  Choose a video to import as a second track…")
	_import_dialog = FileDialog.new()
	_import_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_import_dialog.access = FileDialog.ACCESS_FILESYSTEM
	# In-window dialog, NOT native. The native (portal) dialog silently shows
	# nothing on a Linux box without xdg-desktop-portal - the exact "I press T
	# and nothing happens" report. Godot's own dialog always renders in-window.
	_import_dialog.use_native_dialog = false
	_import_dialog.title = "Import a second track (picture-in-picture)"
	_import_dialog.filters = PackedStringArray(["*.mp4, *.mov, *.mkv, *.webm, *.ogv ; Video"])
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_import_dialog.current_dir = downloads
	_import_dialog.size = Vector2i(820, 560)
	_import_dialog.file_selected.connect(_start_track_import)
	# Free the dialog whichever way it closes (pick or cancel), so the next T
	# opens a fresh one and the is_instance_valid guard above reads false
	# again. _start_track_import is connected first, so it runs before the
	# close on a pick.
	_import_dialog.file_selected.connect(func(_p): _close_import_dialog())
	_import_dialog.canceled.connect(_close_import_dialog)
	add_child(_import_dialog)
	_import_dialog.popup_centered()


func _close_import_dialog() -> void:
	if _import_dialog != null and is_instance_valid(_import_dialog):
		_import_dialog.queue_free()
	_import_dialog = null


## Same one-time ffmpeg->theora transcode _prep() does for the primary clip,
## minus the audio extraction step (v1 tracks are silent - see the class doc's
## scope note) - PID-polled in _process(), never blocking, matching the
## project's one established pattern for external subprocesses.
## Add the track's LANE first, transcode second. The lane is pure metadata
## (duration/offset/trim) - it doesn't need the transcoded video at all - so it
## appears the instant you pick a file, independent of the background ffmpeg run.
## The old order (transcode -> probe -> only THEN add the lane) meant any hiccup in
## that async tail left the timeline looking like nothing happened. The PiP VIDEO
## still needs the .ogv, so _build_track_view is deferred to _finish_track_import;
## until then _sync_tracks just skips this index (no "player" in its runtime slot).
func _start_track_import(source: String) -> void:
	var slug := _slugify(source)
	var dir := _session_path.get_base_dir()
	var idx := session.tracks.size()
	var video := dir + "/track_%s_%d.ogv" % [slug, idx]
	# The SOURCE file has reliable duration metadata (unlike our libtheora output),
	# so the lane gets its real length up front; the transcoded .ogv re-confirms it
	# in _finish_track_import. Floor keeps the lane a visible width if a probe fails.
	var dur := _probe_duration(source)
	if dur <= 0.0:
		dur = maxf(session.duration, 1.0)
	_push_undo("", "imported a track")
	session.tracks.append({
		"video_path": video, "source_path": source, "duration": dur,
		"clip_in": 0.0, "clip_out": dur, "offset": 0.0,
		"x": 0.68, "y": 0.04, "w": 0.28, "h": 0.28,
		"volume": 1.0,
	})
	while _track_runtime.size() <= idx:
		_track_runtime.append({})   # empty slot: _sync_tracks skips it until the video loads
	_refresh_lanes()                # the new timeline lane is on screen NOW
	_mark_dirty()
	_import_pending = {"source": source, "video": video, "index": idx}
	_set_status("⏳  Track added - transcoding its video in the background…")
	# Keep the audio this time (was -an): the track's own VideoStreamPlayer plays the
	# embedded Vorbis, mixed alongside the main clip, with per-track mute/volume (see
	# _sync_tracks). A source with no audio just yields a silent stream - no failure.
	# .part + promote, same as the main prep - see _promote_part's doc.
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", source,
		"-c:v", "libtheora", "-q:v", "6", "-g", "25",
		"-c:a", "libvorbis", "-q:a", "4",
		"-f", "ogg", ProjectSettings.globalize_path(video) + ".part"])
	_import_pid = Subprocess.start("ffmpeg", args, "clip import")
	if _import_pid <= 0:
		_cancel_pending_track()     # couldn't even start ffmpeg - roll the lane back out
		_set_status("⚠  Could not start ffmpeg for track import")


## Roll back the placeholder lane _start_track_import added (ffmpeg failed to start,
## or the transcode produced nothing usable). The pending track is always the last
## one (imports are serialized - see _prompt_import_track's guard).
func _cancel_pending_track() -> void:
	if _import_pending.has("index"):
		var idx := int(_import_pending.index)
		if idx >= 0 and idx < session.tracks.size():
			session.tracks.remove_at(idx)
			if idx < _track_runtime.size():
				_track_runtime.remove_at(idx)
			_refresh_lanes()
			_mark_dirty()
	_import_pending = {}
	_import_pid = -1


func _finish_track_import() -> void:
	if not _import_pending.has("index"):
		_import_pending = {}
		return
	var idx := int(_import_pending.index)
	_promote_part(String(_import_pending.video))
	var abs_video := ProjectSettings.globalize_path(String(_import_pending.video))
	var dur := _probe_duration(abs_video)
	if dur <= 0.0 or idx < 0 or idx >= session.tracks.size():
		# The lane is already on screen; the transcode still failed, so take it back
		# out rather than leave a lane whose PiP video would never load.
		_cancel_pending_track()
		_set_status("⚠  Track import failed (transcode produced no usable video)")
		return
	# Correct the lane to the actual transcoded length, keeping it full if untrimmed.
	var t: Dictionary = session.tracks[idx]
	var was_full: bool = float(t.get("clip_out", 0.0)) >= float(t.get("duration", 0.0)) - 0.05
	t["duration"] = dur
	if was_full:
		t["clip_out"] = dur
	_build_track_view(idx)          # now the .ogv exists: load its PiP player
	_refresh_lanes()
	_mark_dirty()
	_import_pending = {}
	if _status != null:
		_status.visible = false


## Each imported track is driven off the PRIMARY player's own clock (never its
## own independent playback state) - the same discipline the audio sync already
## follows (see _play's class doc), so live preview and an export relaunch trace
## the identical picture. A track that isn't currently inside its
## [offset, offset+span) window on the master timeline is paused and hidden;
## entering it seeks the track's own player to the matching local position and
## lets it run from there, only re-seeking again if it drifts (video seeks are
## heavy - constant tiny re-seeks would stutter, same reasoning as the 0.15s
## audio tolerance, just a bit wider since a video seek is coarser than an
## audio one).
func _sync_tracks() -> void:
	if _player == null:
		return
	var master_t := _player.stream_position
	for i in session.tracks.size():
		if i >= _track_runtime.size():
			continue
		var rt: Dictionary = _track_runtime[i]
		if not rt.has("player"):
			continue
		var track: Dictionary = session.tracks[i]
		var offset := float(track.get("offset", 0.0))
		var cin := float(track.get("clip_in", 0.0))
		var cout := float(track.get("clip_out", 0.0))
		var local_t := master_t - offset + cin
		var inside: bool = cout > cin and local_t >= cin and local_t < cout
		var tplayer: VideoStreamPlayer = rt.player
		var wrap: Control = rt.wrap
		var taudio: AudioStreamPlayer = rt.get("audio")   # may be null (silent track)
		# This track owns the PiP only when V has selected it (pip_track == i+1). The
		# box is still gated by inset_show so track modes fade like the main PiP does.
		var selected := _pip_track == i + 1
		# The clip's fade envelope: audio AND video ramp together (coupled) over the
		# fade_in seconds after the clip's start and the fade_out seconds before its end.
		# g is 0 at a faded edge, 1 across the middle; see _clip_fade_gain / _track_level_db.
		var g := _clip_fade_gain(master_t - offset, cout - cin,
			float(track.get("fade_in", 0.0)), float(track.get("fade_out", 0.0)))
		if inside:
			wrap.visible = _last_inset_show > 0.001 and selected
			wrap.modulate.a = g   # VIDEO fade, coupled to the audio one below
			if not bool(rt.active) or absf(tplayer.stream_position - local_t) > 0.2:
				tplayer.stream_position = local_t
				rt.active = true
			tplayer.paused = not _playing
			var view: TextureRect = rt.view
			if view != null:
				view.texture = tplayer.get_video_texture()
			if taudio != null:
				# pull-rope volume × the fade envelope, as a log/dB level
				taudio.volume_db = _track_level_db(float(track.get("volume", 1.0)), g)
				if _playing:
					if not taudio.playing:
						taudio.play(local_t)
					taudio.stream_paused = false
					if absf(taudio.get_playback_position() - local_t) > 0.2:
						taudio.seek(local_t)
				else:
					taudio.stream_paused = true   # paused / scrubbing: freeze its sound
		else:
			wrap.visible = false
			tplayer.paused = true   # outside its window: no picture...
			rt.active = false
			if taudio != null and taudio.playing:
				taudio.stop()        # ...and no sound; re-entry restarts it clean
	# The main clip's PiP shows only when it is the selected source (pip_track == 0);
	# a selected track replaces it. _apply_frame_state set _mask_wrap by inset_show
	# alone, so correct it here now that we know which source V picked.
	if _pip_track != 0:
		_mask_wrap.visible = false


func _build_render_view() -> void:
	var full := Control.new()
	full.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(full)
	_build_video_composition(full)
	for i in session.tracks.size():
		_build_track_view(i)
	# Product-demo chrome: if this session uses META anywhere, build the REAL editor
	# chrome (timeline, lanes, control panel) on top of the clean composition, fully
	# transparent. _apply_meta_chrome fades it in per frame with the meta envelope, so
	# the export shows clean video normally and the working editor during a meta
	# section (the recorded demo). Purely additive - no meta markers, nothing built,
	# the clean-video export path is untouched. mouse_filter IGNORE: never interactive.
	if _session_uses_meta():
		_meta_chrome = Control.new()
		_meta_chrome.set_anchors_preset(Control.PRESET_FULL_RECT)
		_meta_chrome.mouse_filter = Control.MOUSE_FILTER_IGNORE
		_meta_chrome.modulate.a = 0.0
		add_child(_meta_chrome)
		_chrome_parent = _meta_chrome
		_build_chrome()
		_chrome_parent = null
		_refresh_panel()
	# The export starts at the TIMELINE start (0), exactly like the live editor: live
	# playback begins at 0 and main_visible_at() never gates on clip_in, so the main
	# clip is shown from source 0 and markers/scenes can (and here do) sit before
	# clip_in. Seeking the export to clip_in instead silently dropped everything before
	# it - the "main video starts ~30s too late, early scenes truncated" report, on a
	# session whose clip_in was 37s. clip_in still bounds the restore clamp and the
	# kept-range END via content_end(); it is simply not the export's start.
	_player.play()
	_player.stream_position = 0.0
	_apply_frame_state(session.at_time(0.0))
	# The export quits on the deterministic movie clock reaching content_end() (see
	# _process), NOT on the audio finishing - an audio track shorter than the session
	# would otherwise cut the movie (and its trailing raw video) short.


func _build_editor_ui() -> void:
	_video_area = AspectRatioContainer.new()
	_video_area.set_anchors_preset(Control.PRESET_FULL_RECT)
	_video_area.offset_left = PANEL_W
	# THE CLIP'S OWN SHAPE, not a 16:9 assumption: a portrait clip gets a tall
	# narrow slot with black bars left and right (pillarbox), a wide one keeps the
	# bars above and below. AspectRatioContainer centres its single child in
	# whatever is left after PANEL_W and the lane strip (see _apply_lane_reserved),
	# so the picture always fits the viewport whole and is never stretched.
	# _sync_source_size re-fits this once the decoder confirms the real size.
	_video_area.ratio = _source_aspect()
	add_child(_video_area)

	# A plain Control fills the AspectRatioContainer's one centered/letterboxed slot.
	var inner := Control.new()
	inner.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_video_area.add_child(inner)
	_build_video_composition(inner)
	for i in session.tracks.size():
		_build_track_view(i)

	_build_chrome()
	_build_export_ui()
	_build_feedback()
	_refresh_panel()
	_apply_frame_state(session.at_time(_player.stream_position))


## Where the interactive chrome (timeline, lanes, control panel) parents. Normally
## `self` (the live editor); during a render-mode export that uses META, it's the
## fading _meta_chrome overlay so a meta section can reveal the real working editor
## over the clean video (the recorded product demo).
func _chrome_host() -> Node:
	return _chrome_parent if _chrome_parent != null else self


## The editor chrome - the marker timeline, the trim/track lane stack, and the
## control panel - built into _chrome_host(). Split out of _build_editor_ui so the
## export path can build the SAME real widgets into an overlay (see _build_render_view
## / _apply_meta_chrome). Prereqs: session and _player already exist.
func _build_chrome() -> void:
	_tview = TimelineView.new()
	_tview.zoom = session.timeline_zoom          # restore the last zoom/pan (see _save_session)
	_tview.view_start = session.timeline_view_start

	_timeline = MaskTimeline.new()
	_timeline.session = session
	_timeline.player = _player
	_timeline.tview = _tview
	_timeline.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	_timeline.offset_left = PANEL_W
	_timeline.offset_bottom = -90
	_timeline.offset_top = -90
	_timeline.scrubbed.connect(_on_scrub)
	_timeline.marker_picked.connect(_select_marker)
	_timeline.marker_drag_started.connect(func(_m): _push_undo("", "moved a marker"))
	_timeline.marker_moved.connect(func(_m):
		_refresh_marker_label()
		_mark_dirty())
	_chrome_host().add_child(_timeline)

	# The trim/track lane stack - the primary clip's own trim block, plus one
	# lane per imported track - sits directly above the marker strip, sharing
	# its ruler via the same _tview (see _refresh_lanes for why its height is
	# computed and set explicitly rather than left to container auto-sizing).
	_lanes_col = VBoxContainer.new()
	_lanes_col.add_theme_constant_override("separation", 2)
	_lanes_col.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	_lanes_col.offset_left = PANEL_W
	_chrome_host().add_child(_lanes_col)
	_refresh_lanes()

	_build_panel()
	# Every popup in the editor joins the modal counter that suspends the
	# cursor auto-hide (see _guard_popups). Walked from the whole tree, not
	# just the panel, so the toolbar's dropdowns are covered too.
	_guard_popups(self)


## The META mirror source: the editor's OWN previous frame, read back from the main
## viewport into u_workspace. A full-window GPU->CPU readback (expensive), so the
## caller (_process) only runs it while a meta layer is actually on screen. Feeding
## this window back into the video surface - which lives IN this window - is the
## infinite mirror; the one-frame delay is inherent and wanted (each frame nests one
## level deeper). Downscaled before upload since the mirror draws small anyway.
func _capture_workspace() -> void:
	# Headless has no real framebuffer to read back (the dummy renderer's viewport
	# texture is null - get_image would error every frame). It also never records a
	# movie, so there is nothing to mirror; the windowed Movie Maker export IS a real
	# GPU context, so this only ever skips genuine no-op cases.
	if DisplayServer.get_name() == "headless":
		return
	var vtex := get_viewport().get_texture()
	if vtex == null:
		return
	var img := vtex.get_image()
	if img == null or img.is_empty():
		return
	if img.get_width() > 960:
		var h := int(round(960.0 * float(img.get_height()) / float(maxi(1, img.get_width()))))
		img.resize(960, maxi(1, h), Image.INTERPOLATE_BILINEAR)
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	if render_mode:
		img = _shrink_into_video_pane(img)
	var sz := Vector2i(img.get_width(), img.get_height())
	if _workspace_tex == null or _workspace_tex.get_size() != Vector2(sz):
		_workspace_tex = ImageTexture.create_from_image(img)
	else:
		_workspace_tex.update(img)
	_mat_main.set_shader_parameter("u_workspace", _workspace_tex)
	_mat_inset.set_shader_parameter("u_workspace", _workspace_tex)


## In the live editor, the mirror's nesting comes for free: the video pane the shader
## draws onto is genuinely SMALLER than the captured window (see _video_area's
## PANEL_W inset + 16:9 letterbox in _build_editor_ui), so sampling the whole capture
## back onto that smaller surface shrinks it - and since the capture already contained
## the previous shrink, each frame nests one level deeper. _build_render_view's own
## surface is edge-to-edge (the export must stay a clean, full-bleed video outside a
## meta section - see its comment), so that free shrink doesn't exist there: verified
## by a standalone Movie-Maker readback test (feedback/0001) that a same-size feedback
## quad only ghosts/blurs, never nests. This reproduces the live editor's inset by hand
## - blit a downscaled copy of the capture into the same PANEL_W/letterbox sub-rect the
## live video pane occupies, over an unshrunk copy of the capture (so the chrome
## painted at its real position, e.g. by _apply_meta_chrome, still reads at full size
## around it) - so feeding this back through the SAME edge-to-edge quad reproduces the
## same receding "hall of mirrors" the live editor gets from its smaller surface.
func _shrink_into_video_pane(img: Image) -> Image:
	var w := img.get_width()
	var h := img.get_height()
	# PANEL_W is a fraction of the LIVE EDITOR's window (the base 1920-wide canvas
	# it lays out in), applied to whatever width the capture came back at. The pane
	# inside it fits the CLIP's aspect, same as _video_area does - so a portrait
	# session mirrors a tall narrow pane, not a 16:9 one.
	var asp := _source_aspect()
	var px0 := int(round(w * float(PANEL_W) / 1920.0))
	var avail_w := maxi(1, w - px0)
	var pw := avail_w
	var ph := int(round(float(avail_w) / asp))
	if ph > h:
		ph = h
		pw = int(round(float(h) * asp))
	var py0 := int(round((h - ph) / 2.0))
	var shrunk := img.duplicate()
	shrunk.resize(maxi(1, pw), maxi(1, ph), Image.INTERPOLATE_BILINEAR)
	var canvas: Image = img.duplicate()
	canvas.blit_rect(shrunk, Rect2i(Vector2i.ZERO, Vector2i(pw, ph)), Vector2i(px0, py0))
	return canvas


## Render-mode only: fade the editor-chrome overlay in with the meta envelope, so the
## export shows clean video normally and the real working editor during a meta section.
func _apply_meta_chrome(amount: float) -> void:
	if _meta_chrome == null:
		return
	_meta_chrome.modulate.a = smoothstep(0.0, 1.0, clampf(amount, 0.0, 1.0))


func _session_uses_meta() -> bool:
	for m in session.markers:
		if int(m.get("effect_a", 0)) == MaskSession.EFFECT_META:
			return true
	return false


## The same backtick feedback console the auto/manual show has (see feedback.gd),
## with mask-specific plumbing injected: the descriptor snapshots everything needed
## to debug a masking complaint (playhead, the fully-resolved layer stack at that
## instant, every marker, keying globals, audio envelope), freeze pauses playback
## while typing (restoring the prior play state after), and advance is a no-op -
## the playhead is the user's business, not the console's.
func _build_feedback() -> void:
	_feedback = preload("res://scripts/feedback.gd").new()
	_feedback.describe = _feedback_descriptor
	_feedback.freeze = func(on: bool):
		if on:
			_was_playing_before_feedback = _playing
			_play(false)
		elif _was_playing_before_feedback:
			_play(true)
	_feedback.advance = func(): pass
	add_child(_feedback)
	# Always present, same reasoning as main.gd's own _assistant: it's also
	# the feedback browser (review/delete old submissions), which shouldn't
	# require an assistant backend selected to use. Assistant itself gates
	# actually DISPATCHING anything on the splash's persisted backend choice
	# (see splash.gd) - this just wires the console up. One editor session =
	# one Assistant instance for its whole lifetime, no re-entrancy to guard
	# against (open_source() runs once per process here).
	var assistant := preload("res://scripts/assistant.gd").new()
	add_child(assistant)
	_feedback.submitted.connect(assistant.enqueue)


## Everything I'd want to know about "this frame looks wrong": where we are, what
## the timeline resolved to (including each live layer's full parameter set and
## envelope), the raw marker list, and the session's file identity. The screenshot
## the console pairs with this carries the artifact itself.
func _feedback_descriptor() -> Dictionary:
	var t: float = _player.stream_position if _player != null else 0.0
	var p := session.at_time(t)
	return {
		"mode": "mask",
		"time": t,
		"time_str": MaskTimeline.format_time(t),
		"session_path": _session_path,
		"video_path": session.video_path,
		"source_path": session.source_path,
		"duration": session.duration,
		"resolved_state": p,             # globals + amounts + the live layer stack
		"layer_count": (p.get("layers", []) as Array).size(),
		"markers": session.markers,
		"marker_count": session.markers.size(),
		"audio_env": _env_at(t),
		"peek_raw": _peek_raw,
		"playing": _playing,
	}


func _build_panel() -> void:
	var panel := PanelContainer.new()
	panel.set_anchors_preset(Control.PRESET_LEFT_WIDE)
	panel.offset_right = PANEL_W
	panel.clip_contents = true   # belt-and-suspenders: a child's minimum size can
	# never visually push the panel past PANEL_W and over the timeline, whatever
	# happens inside (see the autowrap fix below for the actual root cause this
	# guards - a long unwrapped Label's natural width was doing exactly that).
	_chrome_host().add_child(panel)

	# Two independently-scrolling regions, stacked: the controls above (which can
	# get tall - color pickers, a dozen sliders) scroll in whatever space is left,
	# and the sequential ramp/damp list is pinned to the bottom with its own fixed-
	# height scroll, so it's always reachable without paging through everything above.
	var outer := VBoxContainer.new()
	outer.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	panel.add_child(outer)

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	outer.add_child(scroll)

	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 14)
	scroll.add_child(margin)

	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 8)
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	margin.add_child(col)

	# Two buttons up here: Help and Import track. Everything ELSE that used to
	# be a button (play, view cycle, peek, undo, redo) was pure duplication of
	# a key and moved to the keyboard + the help overlay. Import stayed a
	# button on purpose: it's the ONE action with no on-screen equivalent and
	# no way to discover (a hidden T shortcut is exactly the "I don't see a
	# clear way to do this" complaint - a keyboard-only import that also
	# depends on nothing having eaten the keystroke is not good enough for the
	# single essential action). Both a visible button AND the T key now.
	var title_row := HBoxContainer.new()
	title_row.add_theme_constant_override("separation", 6)
	col.add_child(title_row)
	var title := Label.new()
	title.text = "ghost-mask"
	title.add_theme_font_size_override("font_size", 22)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title_row.add_child(title)
	var import_btn := Button.new()
	import_btn.text = "⬆ Track"
	import_btn.tooltip_text = "Import a second video as a picture-in-picture track (T)"
	import_btn.focus_mode = Control.FOCUS_NONE
	import_btn.pressed.connect(_prompt_import_track)
	title_row.add_child(import_btn)
	var help_btn := Button.new()
	help_btn.text = "?  Help"
	help_btn.tooltip_text = "Keyboard map (F1)"
	help_btn.focus_mode = Control.FOCUS_NONE
	help_btn.pressed.connect(_toggle_help)
	title_row.add_child(help_btn)

	var time_row := HBoxContainer.new()
	time_row.add_theme_constant_override("separation", 10)
	col.add_child(time_row)
	_time_label = Label.new()
	_time_label.add_theme_font_size_override("font_size", 15)
	_time_label.add_theme_color_override("font_color", Color(0.85, 0.9, 1.0))
	time_row.add_child(_time_label)
	_view_label = Label.new()
	_view_label.add_theme_font_size_override("font_size", 13)
	_view_label.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
	time_row.add_child(_view_label)
	_build_help_overlay()

	# One channel by default (a channel = one target color + one effect + one
	# strength); the second is opt-in behind a toggle. Channels are independent
	# layers in the shader, not competing "sides" - so there's no swap control any
	# more either (swapping meant something when every pixel was forced to one side
	# or the other; independent channels just re-pick their own colors).
	_grp_color = VBoxContainer.new()
	_grp_color.add_theme_constant_override("separation", 8)
	col.add_child(_grp_color)
	_key_color_label = _label("Key color", "The color this channel targets - what it keys or paints")
	_grp_color.add_child(_key_color_label)
	_color_a = ColorPickerButton.new()
	_color_a.focus_mode = Control.FOCUS_NONE
	_color_a.custom_minimum_size = Vector2(0, 40)
	_color_a.edit_alpha = false
	_color_a.tooltip_text = "The color this channel targets - what it keys or paints. " + \
		"The HUE is what gets matched; the swatch keeps the rest of your pick so you " + \
		"can see which colour off the footage you chose"
	# Three fields, one pick - the same shape as the repaint picker below. Only the
	# HUE is keyed on; the other two exist so the swatch you get back is the swatch
	# you set (see MaskSession's key_sat/key_val).
	_color_a.color_changed.connect(func(c):
		_edit("hue_a", c.h)
		_edit("key_sat", c.s)
		_edit("key_val", c.v))
	_grp_color.add_child(_color_a)
	col.add_child(_label("Effect", "Which visual treatment this layer applies"))
	_effect_a = _effect_menu(col, func(id): _edit("effect_a", float(id)))

	# Every option below - this channel's grading through this effect's own
	# pattern/echo/weather/tendril knobs - lives in one flat, sortable list instead
	# of fixed titled groups (see feedback/0011: the group boundaries were drawn as
	# separator lines that "made no sense" and just ate space). Which rows are
	# actually visible still follows the selected effect exactly as before
	# (MaskSession.EFFECT_CONTROLS / PATTERN_KNOBS, see _update_effect_controls) -
	# only their ON-SCREEN ORDER is now driven by _sort_mode. _sort_dropdown picks
	# it directly - same shape as the Effect dropdown just above; see _apply_sort.
	var sort_tip := "How the options below are ordered - alphabetical (either " + \
		"direction), or energy: the fullest slider floats to the top, pick-type " + \
		"options with no slider sink to the bottom (A → Z among themselves)"
	col.add_child(_label("Sort", sort_tip))
	_sort_dropdown = OptionButton.new()
	_sort_dropdown.focus_mode = Control.FOCUS_NONE
	_sort_dropdown.tooltip_text = sort_tip
	for i in _SORT_MODES.size():
		_sort_dropdown.add_item(_SORT_MODES[i], i)
	_sort_dropdown.select(_sort_mode)
	_sort_dropdown.item_selected.connect(func(id):
		_sort_mode = id
		_apply_sort())
	col.add_child(_sort_dropdown)

	_grp_options = VBoxContainer.new()
	_grp_options.add_theme_constant_override("separation", 8)
	col.add_child(_grp_options)

	_intensity_a = _slider(_grp_options, "Strength", 0.0, 1.0, func(v): _edit("intensity_a", v),
		"How strongly this layer's effect applies")
	_intensity_label = _intensity_a.get_meta("field_label")
	_register_option(_intensity_a)
	# Morph: its OWN field (fx_tint), deliberately decoupled from the color
	# picker above. The old "Hue" slider was a second widget on hue_a and the
	# linkage read as a bug ("changing hue changes the picker too") - now the
	# picker says what to MASK and this says what color the drawn effect
	# becomes (a palette hue rotation; 0 = the effect's natural colors).
	_hue_a = _slider(_grp_options, "Morph", 0.0, 1.0, func(v):
		_edit("fx_tint", v),
		"Rotates this effect's own palette hue - 0 keeps its natural colors")
	_register_option(_hue_a)

	# Option rows, shown per the selected effect's needs (the control hierarchy,
	# MaskSession.EFFECT_CONTROLS): a slider that does nothing for the current
	# effect is not on screen for it. Erase shows none of these (projection is
	# gate-free); restore shows only the threshold, relabeled as its reach;
	# the volumetrics show everything. See _update_effect_controls.
	_threshold = _slider(_grp_options, "Threshold", 0.0, 1.0, func(v): _edit("threshold", v),
		"How far a pixel's hue may drift from the key color and still be masked")
	_threshold_label = _threshold.get_meta("field_label")
	_register_option(_threshold)

	_feather = _slider(_grp_options, "Feather", 0.0, 0.5, func(v): _edit("feather", v),
		"Softness of the mask's edge - 0 is a hard cutoff")
	_register_option(_feather)
	_sat_floor = _slider(_grp_options, "Min colorfulness", 0.0, 1.0, func(v): _edit("sat_floor", v),
		"Minimum saturation a pixel needs before it can be keyed at all")
	_register_option(_sat_floor)

	# The wisp field's placement - pan/zoom the pattern over the frame (keyframe a
	# tendril onto an eye), and dial its coverage from one wisp to an engulfing.
	# All continuous marker fields, so they blend through ramps/damps.
	_fx_x = _slider(_grp_options, "Pan X", -2.0, 2.0, func(v): _edit("fx_x", v),
		"Shifts the pattern horizontally over the frame")
	_fx_x.step = 0.01
	_fx_x_label = _fx_x.get_meta("field_label")
	_register_option(_fx_x)
	_fx_y = _slider(_grp_options, "Pan Y", -2.0, 2.0, func(v): _edit("fx_y", v),
		"Shifts the pattern vertically over the frame")
	_fx_y.step = 0.01
	_fx_y_label = _fx_y.get_meta("field_label")
	_register_option(_fx_y)
	_fx_scale = _slider(_grp_options, "Scale", 0.1, 8.0, func(v): _edit("fx_scale", v),
		"Zoom of the effect's pattern - 1 is nominal size")
	_fx_scale.exp_edit = true
	_register_option(_fx_scale)
	_fx_density = _slider(_grp_options, "Coverage", 0.0, 1.0, func(v): _edit("fx_density", v),
		"How much of the keyed region the pattern consumes - 0 untouched, 1 fully devoured")
	_fx_density_label = _fx_density.get_meta("field_label")
	_register_option(_fx_density)
	_fx_contrast = _slider(_grp_options, "Contrast", 0.0, 1.0, func(v): _edit("fx_contrast", v),
		"Edge hardness of the pattern - 0.5 is neutral")
	_fx_contrast_label = _fx_contrast.get_meta("field_label")
	_register_option(_fx_contrast)
	_fx_speed = _slider(_grp_options, "Velocity", 0.1, 4.0, func(v): _edit("fx_speed", v),
		"Speed multiplier for the pattern's motion")
	_fx_speed.exp_edit = true
	_register_option(_fx_speed)
	_resonance = _slider(_grp_options, "Resonance", 0.0, 1.0, func(v): _edit("resonance", v),
		"Audio drive - how strongly this layer reacts to the track's live energy")
	_register_option(_resonance)

	_fx_lag = _slider(_grp_options, "Lag (s)", 0.05, 2.4, func(v): _edit("fx_lag", v),
		"How far back the lagged frame reaches")
	_fx_lag.exp_edit = true
	_fx_lag_label = _fx_lag.get_meta("field_label")
	_register_option(_fx_lag)
	_fx_smooth = _slider(_grp_options, "Smoothing", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"Stutter → smear: 0 is a discrete stutter, 1 is a wide temporal blend")
	_register_option(_fx_smooth)

	# Snow's own view onto fx_smooth - a separate widget from Smoothing above (same
	# stored field, different meaning; the two rows never show together, see
	# _update_effect_controls, so there's no risk of them fighting over what the
	# slider looks like).
	_gust = _slider(_grp_options, "Gust", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"0 is a steady drift, 1 is chaotic gusts")
	_register_option(_gust)

	# Fur's tendril dynamics - fur-only views onto fx_smooth/fx_lag, the same
	# stored-field reuse as Gust above (the rows never show together, see
	# _update_effect_controls).
	_undul = _slider(_grp_options, "Undulation", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"Traveling waves along each strand")
	_register_option(_undul)
	_coil = _slider(_grp_options, "Coil", 0.0, 1.0, func(v): _edit("fx_lag", v),
		"Eddies and spiral curl")
	_register_option(_coil)
	# Stickiness - 0 keeps today's free coat exactly; higher values thin the strands
	# away from natural anchors so the hair clings to the keyed surface, the tracked
	# landmark/motion centroid, and brighter regions (see the shader's fur branch).
	_bleed = _slider(_grp_options, "Bleed", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"How far each feature's paint may spread from the anatomy it found, and " +
		"how much it softens as it travels")
	_register_option(_bleed)
	_settle = _slider(_grp_options, "Settle", 0.0, 1.0, func(v): _edit("fx_lag", v),
		"How sticky the paint is in time - higher holds its shape through " +
		"detection wobble (steadier, a touch slower to follow)")
	_register_option(_settle)
	# The clown's TRACKING knobs, as opposed to its look knobs above. The face
	# model is a blob fitter, not a landmark detector, and how well it holds is a
	# property of the footage as much as of the code - so the two smoothing rates,
	# the prediction, and how literally the picture is trusted are the author's to
	# set per clip. All four default to exactly what the constants they replaced
	# used to be.
	# THE CLOWN'S SHAPE KNOBS. What used to be Steady/Firm/Lead - three controls
	# for smoothing a detector that no longer needs smoothing. The landmark track
	# gives measured outlines, so what the effect was actually missing was a say
	# over how far those outlines get pushed past the face they came from.
	_clown_eye_sl = _slider(_grp_options, "Eye size", 0.0, 1.0,
		func(v): _edit("threshold", v),
		"How far the black grows past the eye it is painted on. It keeps the eye's " +
		"own shape whatever the size - it is that eye, bigger, not a bigger circle")
	_register_option(_clown_eye_sl)
	_clown_drip_sl = _slider(_grp_options, "Drip", 0.0, 1.0,
		func(v): _edit("sat_floor", v),
		"How far the eye black runs DOWN the cheek. It leaves the lash line's own " +
		"lower-outer corner - measured off the patch that is actually painted - so " +
		"it reads as liner running rather than a shape stuck underneath")
	_register_option(_clown_drip_sl)
	_clown_smudge_sl = _slider(_grp_options, "Smudge", 0.0, 1.0,
		func(v): _edit("swap", v),
		"How far the eye black is RUBBED past its own outline, and how softly it " +
		"ends - someone who has been crying and worrying at it, rather than a shape " +
		"with an edge. It thins away over a band instead of stopping, and reaches " +
		"further out in some places than others. 0 is already soft: a hard edge is " +
		"not a setting, it is the polygon showing")
	_register_option(_clown_smudge_sl)
	_clown_drip_w_sl = _slider(_grp_options, "Drip width", 0.0, 1.0,
		func(v): _edit("fx_y", v),
		"How heavy the run is where it leaves the eye. A fraction of THIS eye's " +
		"own width, so it stays in proportion however far Eye size has grown the " +
		"patch. 0 is a lean streak; wound up it is a smear")
	_register_option(_clown_drip_w_sl)
	_clown_drip_curve_sl = _slider(_grp_options, "Drip curve", 0.0, 1.0,
		func(v): _edit("intensity_b", v),
		"How far the run bows OUTWARD as it falls - down the outside of the cheek " +
		"rather than straight past the nose. The bow grows with distance, so the " +
		"run always leaves the lash line vertically whatever this is set to")
	_register_option(_clown_drip_curve_sl)
	_clown_smile_sl = _slider(_grp_options, "Smile width", 0.0, 0.5,
		func(v): _edit("feather", v),
		"How far the painted mouth runs past the real one. At the default it IS " +
		"the real mouth; wound up it goes Joker-wide, out toward the ears")
	_register_option(_clown_smile_sl)
	_clown_steady_sl = _slider(_grp_options, "Steadiness", 0.0, 1.0,
		func(v): _edit("hue_b", v),
		"How hard the landmark track is smoothed. The smoothing is CENTRED - it " +
		"weighs frames on both sides of now - so it costs no lag at any setting; " +
		"raising it only rounds off genuinely fast movement. 0 is already smoothed")
	_register_option(_clown_steady_sl)
	_clown_curve_sl = _slider(_grp_options, "Smile curve", 0.0, 2.0,
		func(v): _edit("fx_speed", v),
		"Sweeps the mouth's corners up into a grin or down into a sulk, leaving " +
		"its middle where the lips are. 1.0 is the mouth as measured")
	_register_option(_clown_curve_sl)
	_clown_feather_sl = _slider(_grp_options, "Edge feather", 0.0, 1.0,
		func(v): _edit("fx_x", v),
		"How far the coat's outer edge fades into the picture. The silhouette is " +
		"a hard-edged shape underneath; without this it reads as a sticker sitting " +
		"on the face rather than paint on it")
	_register_option(_clown_feather_sl)
	_clown_evidence_sl = _slider(_grp_options, "Evidence", 0.0, 1.0,
		func(v): _edit("resonance", v),
		"How much each feature is the PICTURE rather than the outline it is drawn " +
		"inside. At 0 a fixed share of every feature is the bare shape; raise it " +
		"and the outline only bounds where paint may go, while the frame's own " +
		"dark sockets and lip line decide what it actually lands on")
	_register_option(_clown_evidence_sl)

	_rain_squall = _slider(_grp_options, "Squall", 0.0, 1.0,
		func(v): _edit("fx_smooth", v),
		"How unsettled the weather is. Pan sets the direction the wind blows on " +
		"AVERAGE; this is how far the sheet wanders around it and how much the " +
		"drops scatter off each other. 0 is a still, straight downpour")
	_register_option(_rain_squall)

	# THE AUDIO GROUP. Six knobs, no colour, no pattern - this effect resolves to
	# bus parameters instead of pixels (see _apply_audio_fx).
	_au_echo = _slider(_grp_options, "Echo", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"How much of the sound comes back as repeats. Two taps at unrelated " +
		"spacings, so they decay into each other instead of landing as one slap")
	_register_option(_au_echo)
	_au_time = _slider(_grp_options, "Echo time", 0.0, 1.0, func(v): _edit("fx_lag", v),
		"How long between repeats - a slapback at the bottom, a room at the top")
	_register_option(_au_time)
	_au_amb = _slider(_grp_options, "Ambience", 0.0, 1.0, func(v): _edit("fx_density", v),
		"How wet the sound is - how much of what you hear arrived by reflection")
	_register_option(_au_amb)
	_au_room = _slider(_grp_options, "Room", 0.0, 2.0, func(v): _edit("fx_scale", v),
		"How BIG that space is, which is a different question from how wet it is - " +
		"a big dry room and a small wet one sound nothing alike")
	_register_option(_au_room)
	_au_reso = _slider(_grp_options, "Resonance", 0.0, 1.0, func(v): _edit("fx_contrast", v),
		"How much the repeats and the room ring rather than simply fading - it " +
		"darkens the tail so it reads as a space, not as a copy")
	_register_option(_au_reso)
	_au_bass = _slider(_grp_options, "Bass punch", 0.0, 1.0, func(v): _edit("fx_stick", v),
		"Weight AND attack together: it lifts the bottom two bands and compresses " +
		"underneath them, because punchy is not more bass - it is bass whose " +
		"transient survives. Lifting the bottom alone only makes a mix muddy")
	_register_option(_au_bass)
	_hollow = _slider(_grp_options, "Hollow", 0.0, 1.0, func(v): _edit("fx_stick", v),
		"Paint AROUND the eyes and mouth instead of over them - opens over a " +
		"visible eyeball or teeth, closes again on a blink or a shut mouth")
	_register_option(_hollow)
	_stick = _slider(_grp_options, "Stickiness", 0.0, 1.0, func(v): _edit("fx_stick", v),
		"0 is a free coat, 1 clings to the face/motion")
	_register_option(_stick)
	# Umbra's own three - the same stored fields fur/snow/clown reuse under
	# their own names (the groups never show together, so no schema growth).
	_wisp = _slider(_grp_options, "Wisp", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"How readily essence tears off the top of the mass and rises away, " +
		"like smoke leaving a fire")
	_register_option(_wisp)
	_cling = _slider(_grp_options, "Cling", 0.0, 1.0, func(v): _edit("fx_lag", v),
		"How long the shadow-mass holds its shape - higher is heavier and " +
		"slower to follow her, lower is thinner and more restless")
	_register_option(_cling)
	_umbra_depth = _slider(_grp_options, "Depth", 0.0, 1.0, func(v): _edit("fx_stick", v),
		"How much light the mass swallows - some of the wall always survives " +
		"inside it, so it deepens the shadow rather than cutting a hole")
	_register_option(_umbra_depth)
	_umbra_reach = _slider(_grp_options, "Reach", 0.0, 1.0, func(v): _edit("threshold", v),
		"How close the mass comes to the body casting it - raise it to close " +
		"the band of untouched wall between them, lower it to keep well clear")
	_register_option(_umbra_reach)
	_umbra_lead = _slider(_grp_options, "Lead", 0.0, 0.5, func(v): _edit("feather", v),
		"How far AHEAD of her the ghost moves - it turns before she does, so " +
		"it reads as the thing making her move rather than following her")
	_register_option(_umbra_lead)
	_umbra_gaze = _slider(_grp_options, "Gaze", 0.0, 1.0, func(v): _edit("sat_floor", v),
		"Hollow eyes in the mass, tracking hers across the cast offset - " +
		"0 is a faceless shadow")
	_register_option(_umbra_gaze)
	# The eye colour needed a control of its own. hue_b was wired through to the
	# shader but nothing could edit it, so the eyes were stuck on whatever the
	# session happened to have stored - 0.58, cyan, for anything saved before
	# the default changed. "Cannot change their color" was literally true.
	var eye_lbl := _label("Eye color", "The colour the ghost's eyes burn - red by default")
	_grp_options.add_child(eye_lbl)
	_color_eye = ColorPickerButton.new()
	_color_eye.focus_mode = Control.FOCUS_NONE
	_color_eye.custom_minimum_size = Vector2(0, 32)
	_color_eye.edit_alpha = false
	_color_eye.tooltip_text = eye_lbl.tooltip_text
	_color_eye.set_meta("field_label", eye_lbl)
	_color_eye.color_changed.connect(func(c): _edit("hue_b", c.h))
	_grp_options.add_child(_color_eye)
	_register_option(_color_eye)

	# Repaint's two. The paint colour is stored as three fields because a hue
	# alone cannot say "black" - and black is the whole point of the effect's
	# first use (a yellow wall painted out). Same field-reuse idiom as umbra's
	# six above: fx_stick/fx_tint mean something else under other effects, and
	# the groups never show together.
	var paint_lbl := _label("Paint color",
		"The colour the keyed colour BECOMES - a whole colour, so black and white " +
		"are both reachable. Black by default")
	_grp_options.add_child(paint_lbl)
	_color_paint = ColorPickerButton.new()
	_color_paint.focus_mode = Control.FOCUS_NONE
	_color_paint.custom_minimum_size = Vector2(0, 32)
	_color_paint.edit_alpha = false
	_color_paint.tooltip_text = paint_lbl.tooltip_text
	_color_paint.set_meta("field_label", paint_lbl)
	# Three fields, one pick. _edit creates the marker on the first call and
	# edits it on the other two, so this is one marker, not three.
	_color_paint.color_changed.connect(func(c):
		_edit("hue_b", c.h)
		_edit("fx_stick", c.s)
		_edit("fx_tint", c.v))
	_grp_options.add_child(_color_paint)
	_register_option(_color_paint)
	_paint_reach = _slider(_grp_options, "Reach", 0.0, 1.0, func(v): _edit("fx_contrast", v),
		"How far into weakly coloured pixels the paint carries - low keeps it to " +
		"the vivid core of the colour, high takes the washed-out and shadowed parts " +
		"of the same wall with it")
	_register_option(_paint_reach)
	_paint_smooth = _slider(_grp_options, "Smoothing", 0.0, 1.0, func(v): _edit("fx_smooth", v),
		"How wide the edge of the paint is averaged. Compressed video stores colour " +
		"at quarter resolution in blocks, so a boundary keyed pixel-by-pixel comes out " +
		"blocky and crawls frame to frame - raise this until it stops. Some averaging " +
		"always happens; this is how much")
	_register_option(_paint_smooth)

	# THE REGION - universal, not one effect's group: any layer can be confined
	# to a box. Two panel controls (the box itself is dragged on the video, see
	# _build_region_overlay), because a colour key cannot separate two things
	# that are the same colour and position can.
	# A label of its own even though the checkbox carries text: every row here is
	# a (label, control) pair and _apply_sort re-orders them two children at a
	# time, so a control without one desynchronises the whole list.
	var region_lbl := _label("Region",
		"Confine this layer to a box you drag on the video - for when the colour " +
		"you want gone and the colour you want kept are the SAME colour, and only " +
		"their position tells them apart. For the clown it also tells the face " +
		"detector WHERE TO LOOK, which is the fastest cure for a mask that has " +
		"latched onto a wall the colour of skin")
	_grp_options.add_child(region_lbl)
	_region_on = CheckBox.new()
	_region_on.text = "Limit to a box"
	_region_on.focus_mode = Control.FOCUS_NONE
	_region_on.tooltip_text = region_lbl.tooltip_text
	_region_on.set_meta("field_label", region_lbl)
	_region_on.toggled.connect(_on_region_toggled)
	_grp_options.add_child(_region_on)
	_register_option(_region_on)
	_region_soft = _slider(_grp_options, "Region edge", 0.0, 1.0,
		func(v): _edit("reg_soft", v),
		"How gradually the layer fades out at the region's border - 0 is a hard " +
		"rectangular cut, which reads as a rectangular cut")
	_register_option(_region_soft)

	# Every marker is a ramp or a damp - there is no plain/neutral marker (see
	# MaskSession class doc). Both transition TO this marker's values; the kind is
	# which side of the anchor the transition occupies: a ramp eases in BEFORE,
	# complete at the anchor; a damp begins AT the anchor and accumulates after.
	# Lives in _grp_options (not a fixed spot below it) so it's part of the same
	# sortable list as the effect knobs - see _apply_sort/_register_option; a
	# pick-type control with no slider, it sorts to the bottom under Energy mode.
	var _kind_label := _label("Kind",
		"Which way this marker's change runs - ramp eases in before it, damp accumulates after")
	_grp_options.add_child(_kind_label)
	_kind = OptionButton.new()
	_kind.focus_mode = Control.FOCUS_NONE
	_kind.tooltip_text = "Ramp eases in before this marker; damp accumulates after it"
	_kind.set_meta("field_label", _kind_label)
	for i in MaskSession.MARKER_KINDS.size():
		_kind.add_item(MaskSession.MARKER_KINDS[i].capitalize(), i)
	_kind.item_selected.connect(func(id): _edit("kind", float(id)))
	_grp_options.add_child(_kind)
	_register_option(_kind)
	# Exponential response: fine-grained fractions of a second on the left, whole
	# minutes on the right - one slider covers a subtle 0.2s blend and a transition
	# spanning the entire clip. (exp_edit needs a strictly positive min.)
	_marker_duration = _slider(_grp_options, "Ramp/damp span (s)", 0.05, maxf(8.0, session.duration),
		func(v): _edit("duration", v),
		"How long the ramp (before) or damp (after) transition takes, in seconds")
	_marker_duration.exp_edit = true
	_marker_duration.step = 0.01
	_register_option(_marker_duration)

	# --- create/delete + the sequential list, pinned to the panel's bottom with its
	# --- own scroll - the whole "manage markers" workflow stays visible together,
	# --- rather than the create buttons living up in the scrolling edit area where
	# --- reaching them means scrolling past everything else first.
	var list_margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		list_margin.add_theme_constant_override("margin_" + side, 10)
	outer.add_child(list_margin)

	var list_col := VBoxContainer.new()
	list_col.add_theme_constant_override("separation", 4)
	list_margin.add_child(list_col)

	_marker_label = Label.new()
	_marker_label.add_theme_font_size_override("font_size", 12)
	_marker_label.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
	_marker_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	list_col.add_child(_marker_label)

	# See feedback/0019: a preview of what Ctrl+Z would revert, so undo isn't a
	# blind guess - kept right above the buttons that create the history it
	# describes. Populated by _refresh_history_label, driven off _undo_descs.
	_history_label = Label.new()
	_history_label.add_theme_font_size_override("font_size", 12)
	_history_label.add_theme_color_override("font_color", Color(0.55, 0.6, 0.7))
	_history_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_history_label.tooltip_text = "What Ctrl+Z would revert right now"
	list_col.add_child(_history_label)
	_refresh_history_label()

	var mrow := HBoxContainer.new()
	list_col.add_child(mrow)
	var ramp_btn := Button.new()
	ramp_btn.text = "+ Ramp"
	ramp_btn.tooltip_text = "Eases IN before the playhead, arriving here complete"
	ramp_btn.focus_mode = Control.FOCUS_NONE
	ramp_btn.pressed.connect(func(): _add_marker_at_playhead(0))
	mrow.add_child(ramp_btn)
	var damp_btn := Button.new()
	damp_btn.text = "+ Damp"
	damp_btn.tooltip_text = "Begins here and accumulates over the span that follows"
	damp_btn.focus_mode = Control.FOCUS_NONE
	damp_btn.pressed.connect(func(): _add_marker_at_playhead(1))
	mrow.add_child(damp_btn)
	var del_btn := Button.new()
	del_btn.text = "Delete"
	del_btn.focus_mode = Control.FOCUS_NONE
	del_btn.pressed.connect(_delete_selected)
	mrow.add_child(del_btn)

	# The marker list's header, and beside it the switch for whether playback drags
	# the SELECTION along with it (see _process). It lives here rather than in the
	# options panel because it is about the list, not about a marker - nothing it
	# does is stored on one.
	var order_row := HBoxContainer.new()
	# THE LABEL TAKES THE SLACK, not a spacer beside it. Every label here
	# word-wraps (see _label - it is what stops a long one widening the whole
	# panel), and a wrapping label's minimum width is tiny. Put an expanding
	# spacer next to one in an HBox and the label is squeezed to that minimum,
	# which renders "In order" as a column of single letters and makes the row as
	# tall as the text is long. Letting the label expand instead leaves the
	# checkbox at its natural size on the right and the text on one line.
	var order_lbl := _label("In order")
	order_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	order_row.add_child(order_lbl)
	_follow_playhead = CheckBox.new()
	_follow_playhead.text = "Follow"
	# REMEMBERED ACROSS SESSIONS, in user://ghost.cfg rather than in the session
	# JSON. It is a preference about how the EDITOR behaves - the same reason it
	# sits by the list header and not in the options panel - so it belongs beside
	# the other per-user settings; storing it on the session would mean opening
	# somebody else's work silently switched it back.
	_follow_playhead.set_pressed_no_signal(_load_follow())
	_follow_playhead.toggled.connect(_save_follow)
	_follow_playhead.focus_mode = Control.FOCUS_NONE
	_follow_playhead.tooltip_text = "Select each marker as the playhead reaches it. " + \
		"On, scrubbing a session walks you through it without clicking every flag; " + \
		"OFF, the selection stays where you put it - which is what you want while " + \
		"tuning one marker's knobs during playback, since otherwise the next marker " + \
		"steals the panel mid-adjustment"
	order_row.add_child(_follow_playhead)
	list_col.add_child(order_row)
	var list_scroll := ScrollContainer.new()
	list_scroll.custom_minimum_size = Vector2(0, 150)
	list_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	list_col.add_child(list_scroll)

	_marker_list = VBoxContainer.new()
	_marker_list.add_theme_constant_override("separation", 2)
	_marker_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	list_scroll.add_child(_marker_list)


## `tip` is the full explanation shown on mouse-over (Godot's native tooltip) -
## labels themselves stay short so the panel reads clean; see the feedback that
## drove this (0005): inline "Label - long explanation" text was the ambiguity
## complaint, not just missing descriptions.
func _label(text: String, tip: String = "") -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
	# A long, unwrapped label's natural width becomes the whole column's minimum
	# width - that's exactly what pushed the panel wider than PANEL_W and over the
	# timeline (a real bug: "no marker selected..." is longer than most other panel
	# text, so it only showed up after deleting one). Word-wrap caps the minimum to
	# the longest WORD instead of the longest sentence.
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	l.mouse_filter = Control.MOUSE_FILTER_PASS   # let the tooltip trigger over the label itself
	l.tooltip_text = tip
	return l


func _slider(col: VBoxContainer, text: String, lo: float, hi: float, cb: Callable, tip: String = "") -> HSlider:
	var lbl := _label(text, tip)
	col.add_child(lbl)
	var s := HSlider.new()
	# Remember the label so a single knob (label + slider) can be hidden together
	# when the effect doesn't consume it - see _show_field / _update_effect_controls.
	s.set_meta("field_label", lbl)
	s.focus_mode = Control.FOCUS_NONE
	# The wheel belongs to the panel's ScrollContainer, never to a slider you
	# happen to pass over on the way down - the panel is long enough now that
	# scrolling it drags random knobs (and silently edits the marker under
	# them). Drag-only.
	s.scrollable = false
	s.min_value = lo
	s.max_value = hi
	s.step = (hi - lo) / 200.0
	s.tooltip_text = tip   # hovering the slider itself explains it too, not just the label above
	s.value_changed.connect(cb)
	col.add_child(s)
	return s


## A left-anchored pull-rope volume knob for a lane (see VolumeKnob): hold and drag
## away to set a continuous 0..1 level with an asymptotic ceiling. `getter` reads the
## stored volume, `setter` writes it - the same widget drives a track's `volume` and
## the main clip's `main_volume`. Left side deliberately: the right corner sits under
## the floating assistant chat button. `icon` shows the one-time speaker-glyph hint -
## reserved for the main clip's own knob only (see VolumeKnob.show_icon).
func _volume_knob(getter: Callable, setter: Callable, accent: Color, icon: bool = false) -> VolumeKnob:
	var k := VolumeKnob.new()
	k.accent = accent
	k.show_icon = icon
	k.set_anchors_preset(Control.PRESET_TOP_LEFT)
	# offset_left starts past TrackLane._EDGE_W (8px): the primary lane's own trim/fade
	# "in" handles sit at local x0 = 0 (its offset is always 0, and the default view
	# starts at t=0 too), so a knob planted right at the corner silently ate every click
	# meant for that edge - "drag the left edge" looked broken specifically on the main
	# track, since only ITS x0 is pinned to 0 - see feedback/0017.
	k.offset_left = 12
	k.offset_top = 2
	k.offset_right = 34
	k.offset_bottom = 20
	k.get_v = getter
	k.set_v = func(v):
		setter.call(v)
		_mark_dirty()   # cheap + debounced (autosave cooldown), fine to fire while pulling
	return k


## Show/hide one knob - the slider AND the label above it (stored on the slider
## by _slider, or passed for the hand-built pattern rows). Used to prune pattern
## knobs an effect doesn't read (MaskSession.PATTERN_KNOBS).
func _show_field(slider: Control, vis: bool, label: Control = null) -> void:
	if slider != null:
		slider.visible = vis
	var lbl: Variant = label
	if lbl == null and slider != null and slider.has_meta("field_label"):
		lbl = slider.get_meta("field_label")
	if lbl != null:
		(lbl as Control).visible = vis


## Registers a slider or pick-type control (label stashed via set_meta, either by
## _slider or by hand for non-Range controls like an OptionButton) into the flat
## sortable list - see _apply_sort.
func _register_option(ctrl: Control) -> void:
	_options.append({"label": ctrl.get_meta("field_label"), "control": ctrl})


## Re-orders _options within _grp_options per the active sort mode: alphabetical by
## each row's CURRENT label text (which itself changes per effect - e.g. Contrast ->
## Sensitivity for snow), or "energy" - how wide each slider's fill currently reads,
## fullest first (see _option_energy). Pick-type controls with no slider (e.g. Kind)
## have no fill to compare, so they sink below every slider and break ties alphabetically
## among themselves. Hidden rows (this effect doesn't use them) are left trailing after
## the visible ones - their order doesn't matter, they're off screen. Re-run whenever
## the effect/visibility changes (_update_effect_controls) or a mode is picked in
## _sort_dropdown - never on a live drag, so a slider never jumps out from under the
## mouse mid-edit.
func _apply_sort() -> void:
	if _grp_options == null:
		return
	var shown: Array = _options.filter(func(o): return o.control.visible)
	var hidden: Array = _options.filter(func(o): return not o.control.visible)
	match _sort_mode:
		1:
			shown.sort_custom(func(a, b): return a.label.text.nocasecmp_to(b.label.text) > 0)
		2:
			shown.sort_custom(func(a, b):
				var ea: float = _option_energy(a.control)
				var eb: float = _option_energy(b.control)
				if ea == eb:
					return a.label.text.nocasecmp_to(b.label.text) < 0
				return ea > eb)
		_:
			shown.sort_custom(func(a, b): return a.label.text.nocasecmp_to(b.label.text) < 0)
	var idx := 0
	for o in shown + hidden:
		_grp_options.move_child(o.label, idx)
		idx += 1
		_grp_options.move_child(o.control, idx)
		idx += 1


## How "present" a control's current value is, for the "Energy" sort mode. For a
## slider this is its actual fill fraction - the handle's position between the
## track's left and right ends, min at 0 and max at 1 - matching the bar you
## actually see, so a slider barely nudged off its floor reads as low energy even
## if that floor sits far from the range's midpoint. Range.get_as_ratio() (rather
## than a hand-rolled linear (value-min)/span) is what makes this match what's on
## screen for the exp_edit sliders (Scale/Velocity/Lag/Ramp-damp span): those are
## drawn on a logarithmic track, so a linear fraction would read a handle sitting
## visibly right-of-center as barely-there. Pick-type controls with no slider (not
## a Range, e.g. Kind's OptionButton) have nothing to compare, so they return a
## sentinel below any real slider value - see _apply_sort, where that sinks them to
## the bottom and falls back to alphabetical among themselves.
func _option_energy(ctrl: Control) -> float:
	if not (ctrl is Range):
		return -1.0
	return (ctrl as Range).get_as_ratio()


func _effect_menu(col: VBoxContainer, cb: Callable) -> OptionButton:
	var ob := OptionButton.new()
	ob.focus_mode = Control.FOCUS_NONE
	ob.tooltip_text = "Which visual treatment this layer applies"
	for i in MaskSession.MASK_EFFECTS.size():
		ob.add_item(MaskSession.MASK_EFFECTS[i], i)
	ob.item_selected.connect(cb)
	col.add_child(ob)
	return ob


# --- marker editing -----------------------------------------------------------

## Every panel edit targets the selected marker; if none is selected yet, planting
## one at the current playhead is the edit's first move (a knob you touch becomes a
## marker - no separate "create" step needed for the common case). Defaults to a
## ramp when auto-created this way; press +Damp explicitly for the other kind.
func _edit(field: String, value: float) -> void:
	if _syncing:
		return   # a programmatic panel repaint, not the user - see _refresh_panel
	var m: Variant = _selected
	var created := false
	if m == null:
		_push_undo("", "created a marker")   # about to create one - always its own boundary
		m = session.add_marker(_player.stream_position if _player != null else 0.0)
		_selected = m
		_select_generation += 1
		created = true
	else:
		_push_undo("marker:%d:%s" % [_select_generation, field], "adjusted %s" % field.capitalize())
	m[field] = value
	# Assigning a drawing effect to a marker whose OWN view shows no fx surface is
	# a silent foot-gun: the layer holds forever but this marker's view hides it -
	# "I configured fire and nothing shows" (a real session lost its fire exactly
	# this way: the view button had been cycled to Raw on the same marker). Bump
	# the view to the nearest fx-showing mode, preserving the surface choice:
	# raw -> masked, pip_raw -> pip. Never behind the user's back on an unrelated
	# knob, and never for restore/clear (they draw nothing of their own).
	#
	# THE HOLE THIS USED TO HAVE, and it swallowed a whole session: the bump fired
	# only on effect_a/intensity_a, but a marker's DEFAULT effect is already erase
	# and its default strength is already 1.0 - so anyone whose first move is the
	# colour picker (the natural first move: "erase THIS colour") never touches
	# either field, and the marker they just minted was born at view_mode raw,
	# which renders no shader pass at all. The effect was configured, saved and
	# completely invisible, and looked exactly like the effect being broken.
	# So a marker MINTED by a panel touch bumps too, whichever knob minted it -
	# it cannot have a deliberate view choice yet, it only has the inherited one.
	# Deliberately NOT widened to hue_a on an EXISTING marker: view_mode 2 on one
	# of those may be a raw checkpoint (the rebase - see MaskSession's class doc),
	# and quietly un-raw-ing a checkpoint because its colour was adjusted would
	# break the layer window it exists to close.
	if created or field == "effect_a" or field == "intensity_a":
		var vm := int(m.get("view_mode", 2.0))
		var eid := int(m.get("effect_a", 0))
		var drawing: bool = eid != MaskSession.EFFECT_RESTORE \
			and eid != MaskSession.EFFECT_CLEAR \
			and float(m.get("intensity_a", 0.0)) > 0.0
		if drawing and (vm == 2 or vm == 3):
			m["view_mode"] = 1.0 if vm == 2 else 0.0
	if field == "effect_a":
		_update_effect_controls(int(value))
	_timeline.selected = _selected
	_refresh_marker_label()
	_mark_dirty()


func _add_marker_at_playhead(kind_id: int) -> void:
	_push_undo("", "added a %s marker" % MaskSession.MARKER_KINDS[kind_id].capitalize())
	_selected = session.add_marker(_player.stream_position if _player != null else 0.0, kind_id)
	_select_generation += 1
	_timeline.selected = _selected
	_refresh_panel()
	_mark_dirty()


func _delete_selected() -> void:
	if _selected != null:
		_push_undo("", "deleted a marker")
		session.remove_marker(_selected)
		_selected = null
		_timeline.selected = null
		_refresh_panel()
		_mark_dirty()


## Temporal capture: when the playhead crosses into a new _ECHO_INTERVAL slot,
## snapshot the current frame (quarter-res) into the ring and push the ring to
## both materials in AGE ORDER (u_echo0 = newest). GPU readback at ~3Hz is cheap
## enough for a demo; frames where no whisp/echo/snow/oracle/serpent/chimera
## layer is actually on screen (_temporal_active) skip all of it.
func _maybe_capture_echo() -> void:
	if _player == null or session == null:
		return
	var pos := _player.stream_position
	# The readback below (tex.get_image()) is a synchronous GPU->CPU stall. We only
	# ever want to pay it when it MATTERS and when it WON'T fight the user:
	#   - during playback (the effects need to advance), OR
	#   - once, when a scrub has SETTLED (playhead stationary since last frame) - so
	#     dragging the playhead stays responsive and the preview refreshes on release.
	# Never per-scrub-position mid-drag, which is what made clicking the timeline lag
	# and starved the audio. render_mode auto-plays, so export still captures every
	# slot deterministically.
	if not _playing:
		var settled := absf(pos - _prev_pos) < 0.0005
		_prev_pos = pos
		if not settled:
			return
	else:
		_prev_pos = pos
	if not _temporal_active:
		return
	var slot := int(pos / _ECHO_INTERVAL)
	if slot == _echo_slot:
		return
	var tex := _player.get_video_texture()
	if tex == null:
		return
	var img := tex.get_image()
	if img == null or img.is_empty():
		return
	_echo_slot = slot
	img.resize(480, 270, Image.INTERPOLATE_BILINEAR)
	_echo_ring[slot % 8] = ImageTexture.create_from_image(img)
	for age in 8:
		var t: Variant = _echo_ring[((slot - age) % 8 + 8) % 8]
		if t == null:
			t = _echo_ring[slot % 8]
		for mat in [_mat_main, _mat_inset]:
			mat.set_shader_parameter("u_echo%d" % age, t)
	# The face model runs on its own higher-res copy BEFORE _update_whisp_anchor
	# destructively resizes img down to 48x27; claiming the face slot here means
	# _maybe_capture_face never pays a second readback on the same frame.
	if _clown_active:
		_face_slot = int(pos / _FACE_INTERVAL)
		# The blob fitter still runs for the face's COLOUR statistics - mean tint,
		# luminance, redness - which the coat needs and which landmarks cannot
		# supply (they say where the face is, not what colour it is). Its
		# GEOMETRY is then overwritten by the track, which is measured rather
		# than fitted. When there is no track the fitter's geometry stands, as
		# the fallback it now is.
		_update_face_model(img)
		if _ft_state == "ready" and _ft_has(pos):
			_ft_apply_model(pos)
	_update_whisp_anchor(img)
	# The track readback is chimera's alone - skip it entirely unless a chimera layer
	# is actually rendering this frame (it usually isn't: the marker sits at one point
	# in the timeline). This is what my earlier change was paying for everywhere.
	if _chimera_active:
		_update_track_frame()


## The face model's own capture ticks, BETWEEN echo captures (see
## _FACE_INTERVAL's doc). Same discipline as _maybe_capture_echo: only during
## playback (or once when a scrub settles), never mid-drag - the readback is a
## synchronous GPU stall and this one runs at ~7Hz while a clown layer is live.
func _maybe_capture_face() -> void:
	if not _clown_active or _player == null or session == null:
		return
	# THE TRACK WINS WHEN IT EXISTS. It is read every frame (not on the capture
	# tick) because reading it costs an array lookup, and the whole reason the
	# old path had a 6.7 Hz tick at all was that its detection was expensive.
	# Reading continuously is also what removes the last of the stepping.
	_ft_ensure()
	if _ft_state == "ready":
		if _ft_has(_player.stream_position):
			_ft_apply_model(_player.stream_position)
			_update_stencil(_player.stream_position)
		# No detection at this instant (a turn away, a hand across the face):
		# HOLD the last model rather than snapping to a guess.
		return
	var pos := _player.stream_position
	if not _playing and absf(pos - _prev_pos) >= 0.0005:
		return
	var slot := int(pos / _FACE_INTERVAL)
	if slot == _face_slot:
		return
	var tex := _player.get_video_texture()
	if tex == null:
		return
	var img := tex.get_image()
	if img == null or img.is_empty():
		return
	_face_slot = slot
	_update_face_model(img)


## The anchor uniform, glided per frame: lerp(prev EMA, latest EMA) by the
## playhead's fraction through the current capture slot. Deterministic (pure
## function of playback position + capture history) and continuous - the
## pattern drifts to each new lock instead of jumping there.
func _push_anchor() -> void:
	var f := clampf(fposmod(_player.stream_position, _ECHO_INTERVAL) / _ECHO_INTERVAL, 0.0, 1.0)
	var anchor := _anchor_prev.lerp(_anchor_ema, f)
	# Chimera's landmark frames, glided on the same fraction (position + size, both
	# faces). Pure function of playback position + capture history, so live and
	# export trace the identical lock.
	var anchor_scale := lerpf(_anchor_scale_prev, _anchor_scale_ema, f)
	var track_anchor := _track_anchor_prev.lerp(_track_anchor_ema, f)
	var track_scale := lerpf(_track_scale_prev, _track_scale_ema, f)
	for mat in [_mat_main, _mat_inset]:
		mat.set_shader_parameter("u_anchor", anchor)
		mat.set_shader_parameter("u_anchor_scale", anchor_scale)
		mat.set_shader_parameter("u_track_anchor", track_anchor)
		mat.set_shader_parameter("u_track_scale", track_scale)
		# Only push once impulses exist - a short array is silently dropped by
		# Godot (see the u_l_* comment above), so an empty/partial one is worse
		# than leaving the shader's own all-zero (inactive) uniform defaults.
		if _wave_amp.size() == _WAVE_SLOTS:
			mat.set_shader_parameter("u_wave_pos", _wave_pos)
			mat.set_shader_parameter("u_wave_time", _wave_time)
			mat.set_shader_parameter("u_wave_amp", _wave_amp)
		# The clown face model PREDICTS instead of gliding: displaying
		# lerp(prev, ema) put the mask a full tick behind the face - the
		# "always drifting just behind any movement" report. Extrapolating
		# along each feature's own velocity (ema + (ema - prev) * t) leads
		# the fit by up to one tick, which cancels the capture+EMA delay;
		# the cost is mild overshoot on direction reversals, which reads as
		# the mask swinging - far better than trailing.
		if _clown_active:
			var cm := _clown_model_now()
			mat.set_shader_parameter("u_clown_eye_l", cm.eye_l)
			mat.set_shader_parameter("u_clown_eye_r", cm.eye_r)
			mat.set_shader_parameter("u_clown_mouth", cm.mouth)
			mat.set_shader_parameter("u_clown_face_r", _face_r_ema)
			mat.set_shader_parameter("u_clown_eye_lr", cm.eye_lr)
			mat.set_shader_parameter("u_clown_eye_rr", cm.eye_rr)
			mat.set_shader_parameter("u_clown_mouth_r", cm.mouth_r)
			mat.set_shader_parameter("u_clown_face_c", cm.face_c)
			# The face's own axes, from the eye pair: direction along the eye line,
			# scaled by 1/separation. Everything drawn in this frame (the craquelure)
			# is then pinned to the skin rather than to the screen.
			var eye_v: Vector2 = (cm.eye_r - cm.eye_l) * Vector2(_source_aspect(), 1.0)
			var eye_len: float = maxf(eye_v.length(), 1e-4)
			mat.set_shader_parameter("u_clown_frame", eye_v / (eye_len * eye_len))
			mat.set_shader_parameter("u_clown_tint", _face_tint_ema)
			mat.set_shader_parameter("u_clown_lum", _face_lum_ema)


## The landmark anchor (whisp's field origin, chimera's graft window): the
## first whisp-or-chimera marker's target-color mass centroid in the captured
## frame, EMA-smoothed so the lock glides to landmarks instead of jittering
## with noise. When the key color exists NOWHERE (flat lighting), the motion
## centroid anchors instead - see the fallback below. Fur no longer reads
## this - its strands root per-pixel on the keyed surface itself (see
## fur_root_mass / the fur branch of apply_layer in mask_split.gdshader).
func _update_whisp_anchor(img: Image) -> void:
	var hue := -1.0
	for m in session.markers:
		var e := int(m.get("effect_a", 0))
		# Fur joins whisp/chimera in driving the anchor: its Stickiness cue wants the
		# key colour's centroid (with the motion-centroid fallback for flat footage) to
		# track the surface the strands root on. Fur ignores u_anchor at Stickiness 0,
		# so this has no effect on the existing look. Clown rides the same frame -
		# its whole makeup layout lives in the canonical space u_anchor defines.
		if e == 5 or e == MaskSession.EFFECT_CHIMERA or e == MaskSession.EFFECT_FUR \
				or e == MaskSession.EFFECT_CLOWN:
			hue = float(m.get("hue_a", 0.0))
			break
	if hue < 0.0:
		return
	var tc := Color.from_hsv(hue, 1.0, 1.0)
	var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
	var tdir := Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized()
	img.resize(48, 27, Image.INTERPOLATE_BILINEAR)
	# Read the frame as one flat RGBA8 buffer instead of 1296 Image.get_pixel()
	# calls. This runs every _ECHO_INTERVAL during playback (once a temporal effect
	# is live), right after the synchronous GPU readback - the per-pixel Color
	# construction get_pixel does was a measurable slice of that periodic hitch. The
	# math is unchanged; only the pixel access is. (_face_frame does the same.)
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	var acc := Vector2.ZERO
	var acc2 := 0.0          # weighted sum of |pos|^2 - second moment, for the RMS radius (size)
	var wsum := 0.0
	var macc := Vector2.ZERO
	var macc2 := 0.0
	var msum := 0.0
	var have_prev := _wave_prev_lum.size() == 48 * 27
	if not have_prev:
		_wave_prev_lum.resize(48 * 27)
	var motion := 0.0
	for y in 27:
		for x in 48:
			var idx := y * 48 + x
			var base := idx * 4
			var r := float(data[base]) / 255.0
			var g := float(data[base + 1]) / 255.0
			var b := float(data[base + 2]) / 255.0
			var l := 0.299 * r + 0.587 * g + 0.114 * b
			var pos := Vector2((float(x) + 0.5) / 48.0, (float(y) + 0.5) / 27.0)
			var pr := maxf(0.0, (r - l) * tdir.x + (g - l) * tdir.y + (b - l) * tdir.z)
			acc += pos * pr
			acc2 += pos.length_squared() * pr
			wsum += pr
			if have_prev:
				var dm := absf(l - _wave_prev_lum[idx])
				motion += dm
				macc += pos * dm
				macc2 += pos.length_squared() * dm
				msum += dm
			_wave_prev_lum[idx] = l
	if wsum > 0.01:
		_anchor_prev = _anchor_ema
		_anchor_ema = _anchor_ema.lerp(acc / wsum, 0.15)
		# Size = RMS radius of the mass about its own centroid (Var = E[|p|^2] -
		# |E[p]|^2), the second half of the main face's landmark frame.
		_anchor_scale_prev = _anchor_scale_ema
		_anchor_scale_ema = lerpf(_anchor_scale_ema,
			clampf(sqrt(maxf(1e-4, acc2 / wsum - (acc / wsum).length_squared())), 0.05, 0.9), 0.15)
	elif msum > 0.3:
		# FLAT-LIGHTING FALLBACK (chimera's first test case): standard, flat
		# footage may carry the key color NOWHERE - then the landmark is
		# wherever the pixels MOVE. The motion centroid of a talking head IS
		# the head; slower EMA than the color lock because motion is noisier
		# frame to frame.
		_anchor_prev = _anchor_ema
		_anchor_ema = _anchor_ema.lerp(macc / msum, 0.1)
		_anchor_scale_prev = _anchor_scale_ema
		_anchor_scale_ema = lerpf(_anchor_scale_ema,
			clampf(sqrt(maxf(1e-4, macc2 / msum - (macc / msum).length_squared())), 0.05, 0.9), 0.1)
	if have_prev:
		_update_wave_impulses(motion / (48.0 * 27.0))


## Model a face as a landmark FRAME - centroid + isotropic size (RMS radius) - from
## the key-colour mass in an already-48x27 frame, with the motion centroid+spread
## as the flat-lighting fallback (the same thresholds _update_whisp_anchor uses).
## This is the "simple EMA over key thresholds" model chimera phase-locks to; it
## runs on the imported TRACK frame so the graft is normalised by the OTHER head's
## own frame before being re-fitted onto the main head. Returns
## {c, s, cur_lum, ok}; cur_lum is the caller's next previous-luminance grid.
func _face_frame(img: Image, tdir: Vector3, prev_lum: PackedFloat32Array) -> Dictionary:
	var acc := Vector2.ZERO
	var acc2 := 0.0
	var wsum := 0.0
	var macc := Vector2.ZERO
	var macc2 := 0.0
	var msum := 0.0
	var cur := PackedFloat32Array()
	cur.resize(48 * 27)
	var have_prev := prev_lum.size() == 48 * 27
	# Flat RGBA8 buffer read - see _update_whisp_anchor for why (same per-tick path).
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	for y in 27:
		for x in 48:
			var idx := y * 48 + x
			var base := idx * 4
			var r := float(data[base]) / 255.0
			var g := float(data[base + 1]) / 255.0
			var b := float(data[base + 2]) / 255.0
			var l := 0.299 * r + 0.587 * g + 0.114 * b
			var pos := Vector2((float(x) + 0.5) / 48.0, (float(y) + 0.5) / 27.0)
			var pr := maxf(0.0, (r - l) * tdir.x + (g - l) * tdir.y + (b - l) * tdir.z)
			acc += pos * pr
			acc2 += pos.length_squared() * pr
			wsum += pr
			if have_prev:
				var dm := absf(l - prev_lum[idx])
				macc += pos * dm
				macc2 += pos.length_squared() * dm
				msum += dm
			cur[idx] = l
	if wsum > 0.01:
		var c0 := acc / wsum
		return {"c": c0, "s": clampf(sqrt(maxf(1e-4, acc2 / wsum - c0.length_squared())), 0.05, 0.9),
			"cur_lum": cur, "ok": true}
	elif msum > 0.3:
		var c1 := macc / msum
		return {"c": c1, "s": clampf(sqrt(maxf(1e-4, macc2 / msum - c1.length_squared())), 0.05, 0.9),
			"cur_lum": cur, "ok": true}
	return {"c": Vector2(0.5, 0.5), "s": 0.28, "cur_lum": cur, "ok": false}


## Model the imported track face (chimera's graft source) each capture tick, so its
## frame - centroid + size - can normalise the graft before it's re-fitted onto the
## main head. Keyed by the chimera marker's own colour; nothing to do without one.
func _update_track_frame() -> void:
	if _track_runtime.is_empty():
		return
	var rt: Dictionary = _track_runtime[0]
	if not rt.has("player"):
		return
	var tp: VideoStreamPlayer = rt.player
	if tp == null:
		return
	var ttex := tp.get_video_texture()
	if ttex == null:
		return
	var timg := ttex.get_image()
	if timg == null or timg.is_empty():
		return
	var hue := -1.0
	for m in session.markers:
		if int(m.get("effect_a", 0)) == MaskSession.EFFECT_CHIMERA:
			hue = float(m.get("hue_a", 0.0))
			break
	if hue < 0.0:
		return
	var tc := Color.from_hsv(hue, 1.0, 1.0)
	var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
	var tdir := Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized()
	timg.resize(48, 27, Image.INTERPOLATE_BILINEAR)
	var fr := _face_frame(timg, tdir, _track_prev_lum)
	_track_prev_lum = fr.cur_lum
	if fr.ok:
		_track_anchor_prev = _track_anchor_ema
		_track_anchor_ema = _track_anchor_ema.lerp(fr.c, 0.15)
		_track_scale_prev = _track_scale_ema
		_track_scale_ema = lerpf(_track_scale_ema, float(fr.s), 0.15)


## Onset detection for the wave impulses: motion is this capture's average
## per-pixel luminance jolt (see caller). A steady baseline (_wave_motion_ema)
## and its own deviation (_wave_dev_ema) track each clip's ambient motion level
## adaptively - talking-head footage idles near-still, a real head turn spikes
## several deviations above it - so one fixed threshold doesn't have to guess
## right for every source video. Rate-limited (see _WAVE_COOLDOWN) so a shaky
## run of frames fires one wave, not a pile of overlapping ones.
func _update_wave_impulses(motion: float) -> void:
	var onset := motion - _wave_motion_ema
	var t: float = _player.stream_position
	if onset > _wave_dev_ema * 3.5 and t - _wave_last_time >= _WAVE_COOLDOWN:
		_wave_last_time = t
		if _wave_amp.size() != _WAVE_SLOTS:
			_wave_pos.resize(_WAVE_SLOTS)
			_wave_time.resize(_WAVE_SLOTS)
			_wave_amp.resize(_WAVE_SLOTS)
		_wave_pos[_wave_slot] = _anchor_ema
		_wave_time[_wave_slot] = t
		_wave_amp[_wave_slot] = clampf(onset / maxf(_wave_dev_ema * 6.0, 0.02), 0.35, 1.0)
		_wave_slot = (_wave_slot + 1) % _WAVE_SLOTS
	_wave_motion_ema = lerp(_wave_motion_ema, motion, 0.2)
	_wave_dev_ema = lerp(_wave_dev_ema, absf(onset), 0.2)


## Fit the clown face model from one capture: face mass -> centroid, spread
## and tint; then the EYES as the two darkest clusters in the upper face band
## (split left/right) and the MOUTH as the red/dark centroid below. Pure
## heuristics, the crystal school - no landmark model, no ML - built for the
## ASMR framing this effect targets: one person, near center, facing camera,
## small head movement. Face mass prefers the marker's KEY colour (the picker
## says what the face's material is - on natural footage, pick the skin tone);
## a broad natural-skin rule backs it up, and a centered prior downweights
## background and hands at the frame's edges. Results are EMA'd with prev/ema
## pairs; _push_anchor glides the uniforms between ticks like the anchor.
## The live clown layer's region, in frame UV, as the face model sees it: 1 inside,
## 0 outside, soft at the border. Mirrors the shader's region_mask (see
## mask_split.gdshader) so the detector searches exactly the area the effect is
## allowed to paint - a box that bounded the drawing but not the SEARCH would let
## the model fit itself to something it can never paint, which is worse than no
## box at all. Default (whole frame) returns 1 everywhere.
func _face_region_at(uv: Vector2) -> float:
	var lo := Vector2(minf(_clown_region.x, _clown_region.z),
		minf(_clown_region.y, _clown_region.w))
	var hi := Vector2(maxf(_clown_region.x, _clown_region.z),
		maxf(_clown_region.y, _clown_region.w))
	var size := (hi - lo).max(Vector2(1e-4, 1e-4))
	# A softer shoulder than the shader's: the detector wants the box's EDGE to
	# fade its evidence out gradually so a face half in and half out still fits
	# rather than being sliced, where the drawn paint wants the edge the author
	# actually placed.
	var soft: float = maxf(0.06 * minf(size.x, size.y), 0.004)
	var m := 1.0
	if lo.x > 0.0005:
		m = minf(m, smoothstep(lo.x - soft, lo.x + soft, uv.x))
	if hi.x < 0.9995:
		m = minf(m, 1.0 - smoothstep(hi.x - soft, hi.x + soft, uv.x))
	if lo.y > 0.0005:
		m = minf(m, smoothstep(lo.y - soft, lo.y + soft, uv.y))
	if hi.y < 0.9995:
		m = minf(m, 1.0 - smoothstep(hi.y - soft, hi.y + soft, uv.y))
	return m


func _update_face_model(src: Image) -> void:
	var hue := -1.0
	for m in session.markers:
		if int(m.get("effect_a", 0)) == MaskSession.EFFECT_CLOWN:
			hue = float(m.get("hue_a", 0.0))
			break
	if hue < 0.0:
		return
	# prev catches up every tick, found or not - the push EXTRAPOLATES along
	# (ema - prev), so prev must always be exactly one tick behind.
	_face_eye_l_prev = _face_eye_l_ema
	_face_eye_r_prev = _face_eye_r_ema
	_face_mouth_prev = _face_mouth_ema
	_face_r_prev = _face_r_ema
	_face_eye_lr_prev = _face_eye_lr_ema
	_face_eye_rr_prev = _face_eye_rr_ema
	_face_mouth_r_prev = _face_mouth_r_ema
	_face_c_prev = _face_c_ema
	_face_nose_prev = _face_nose_ema
	var tc := Color.from_hsv(hue, 1.0, 1.0)
	var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
	var tdir := Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized()
	# EVERY anatomical ratio below is computed in ASPECT-CORRECTED space
	# (x scaled by the frame's aspect, so one unit means the same thing
	# horizontally and vertically). Mixing raw uv axes is a real bug, not a
	# rounding matter: on 16:9 it made the eye "distance" a width measure
	# while eye-to-mouth stayed a height measure, so the mouth's true offset
	# read ~1.75 instead of ~1.15 and the prior dragged the lips up under
	# the nose.
	var fasp := 1.7778
	if src.get_height() > 0:
		fasp = float(src.get_width()) / float(src.get_height())
	var img: Image = src.duplicate()
	# HALVE REPEATEDLY BEFORE THE FINAL RESIZE. Image.resize's bilinear filter
	# samples a 2x2 neighbourhood whatever the reduction factor, so taking a
	# 1080x1920 frame straight to 96x54 - a 35x reduction vertically - is very
	# nearly point sampling: which source rows happen to land under the sample
	# points changes with sub-pixel movement of the subject, so the whole weight
	# field flickers tick to tick even on a still shot, and every fit derived from
	# it inherits that flicker. shrink_x2 is a true 2x2 box average, so halving
	# down to within ~2x of the target first makes the last resize a genuine
	# average of the whole frame instead of a sparse sample of it. Costs a few
	# cheap passes at this cadence and removes a whole class of jitter.
	while img.get_width() >= 192 and img.get_height() >= 108:
		img.shrink_x2()
	img.resize(96, 54, Image.INTERPOLATE_BILINEAR)
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	# Whole-frame mean luminance FIRST - the brightness cue below is relative
	# to it (a face is bright against ITS OWN room, not against an absolute).
	var bg_acc := 0.0
	for k in 96 * 54:
		var b4 := k * 4
		bg_acc += 0.299 * float(data[b4]) + 0.587 * float(data[b4 + 1]) + 0.114 * float(data[b4 + 2])
	_face_bg_lum = bg_acc / (255.0 * 96.0 * 54.0)
	var lums := PackedFloat32Array()
	lums.resize(96 * 54)
	var wts := PackedFloat32Array()
	wts.resize(96 * 54)
	var reds := PackedFloat32Array()
	reds.resize(96 * 54)
	var acc := Vector2.ZERO
	var accxx := 0.0
	var accyy := 0.0
	var wsum := 0.0
	var lum_acc := 0.0
	var red_acc := 0.0
	var tint_acc := Vector3.ZERO
	for y in 54:
		for x in 96:
			var idx := y * 96 + x
			var base := idx * 4
			var r := float(data[base]) / 255.0
			var g := float(data[base + 1]) / 255.0
			var b := float(data[base + 2]) / 255.0
			var l := 0.299 * r + 0.587 * g + 0.114 * b
			lums[idx] = l
			reds[idx] = r - g
			var pos := Vector2((float(x) + 0.5) / 96.0, (float(y) + 0.5) / 54.0)
			var cr := r - l
			var cg := g - l
			var cb := b - l
			# KEY-COLOUR MASS, as a hue CONE rather than a raw projection - the
			# same correction repaint's selection needed, for the same reason and
			# with worse consequences here. A raw projection measures absolute
			# chroma along the key direction, so it rewards SATURATION as much as
			# similarity: measured against this effect's own default key (red), a
			# lit yellow wall projects 0.324 and the skin in front of it 0.206, so
			# the wall out-keyed the face by half again and the model fitted
			# itself to the wall - "pinned on half my face and half the wall, and
			# it never corrects itself". By the brightness-free aligned fraction
			# the same two are 45 degrees and 24 degrees off the key, and raising
			# that to a power turns a 1.6:1 loss into a 4.9:1 win for the face.
			var clen := sqrt(cr * cr + cg * cg + cb * cb)
			var align := maxf(0.0, cr * tdir.x + cg * tdir.y + cb * tdir.z) / maxf(clen, 1e-4)
			var pr := pow(align, 6.0) * smoothstep(0.02, 0.10, clen)
			# natural-skin fallback: warm chroma, red over blue, carried by
			# real light - broad on purpose, the prior does the rejecting
			var skin := 0.0
			if cr > 0.01 and l > 0.15 and cr > cb:
				skin = clampf(cr * 6.0, 0.0, 1.0) * clampf((cr - cb) * 4.0, 0.0, 1.0)
			# LUMINANCE fallback, for the many faces that carry no usable
			# chroma at all: near-monochrome grades, blue/grey night looks,
			# a lit face against a dark room. Both cues above read ~0 there
			# and the model simply stopped updating. Brightness relative to
			# THIS frame's own mean (computed on the fly below via
			# _face_bg_lum) is the one signal such footage always has.
			var bright := smoothstep(_face_bg_lum + 0.06, _face_bg_lum + 0.30, l)
			var prior := exp(-pos.distance_squared_to(Vector2(0.5, 0.45)) / 0.18)
			# THE REGION BOX IS A "LOOK FOR THE FACE HERE" HINT, not just a bound
			# on where paint may land. Colour alone cannot always separate a face
			# from its background - a warm lit wall reads as skin to every cue
			# this has (warm chroma, bright against the frame's mean, and close
			# enough in hue to key on), and on the clip this was written for it
			# genuinely does. Position separates them trivially, and the author
			# has already drawn the box that says so. Applying it HERE as well is
			# what turns it from a crop into guidance: the detector stops seeing
			# the wall at all, so it locks onto the face instead of straddling
			# both - without anyone placing a single landmark by hand.
			var wt := maxf(maxf(pr * 2.2, skin * 0.9), bright * 0.85) * prior \
				* _face_region_at(pos)
			wts[idx] = wt
			acc += pos * wt
			accxx += pos.x * pos.x * wt
			accyy += pos.y * pos.y * wt
			wsum += wt
			lum_acc += l * wt
			red_acc += (r - g) * wt
			tint_acc += Vector3(cr, cg, cb) * wt
	if wsum <= 0.5:
		return   # nothing face-like this tick - keep the previous model
	# THE FACE'S SIZE COMES FROM ITS CORE, NOT FROM EVERY PIXEL THAT LEANS WARM.
	# The weight field is deliberately generous - key-hue projection OR skin
	# chroma OR brightness-over-the-frame's-mean - so on real footage a lit wall
	# the colour of skin contributes a wide, low plateau of weight, and a plain
	# weighted variance over all of it reports the ROOM's spread as the face's.
	# Measured on a phone clip against a warm wall: rx climbed steadily to 0.42
	# where the head is really about 0.28, and because the eye separation is then
	# reined to that width (see `want` below), the pair was forced apart a little
	# further every tick until one eye sat out on the subject's headscarf. That is
	# the "only one eye is detected, the other is drawn across my forehead" report,
	# and it is a size bug, not a detection bug.
	#
	# So the spread is measured over the CONCENTRATED mass only - cells carrying a
	# real fraction of the peak weight - which is the face and not the plateau
	# around it. The centroid comes from the same cells for the same reason.
	var wpeak := 0.0
	for i in 96 * 54:
		wpeak = maxf(wpeak, wts[i])
	var core_floor := wpeak * _FACE_CORE
	var cacc := Vector2.ZERO
	var cxx := 0.0
	var cyy := 0.0
	var cw := 0.0
	for y in 54:
		for x in 96:
			var idx := y * 96 + x
			if wts[idx] < core_floor:
				continue
			var pos := Vector2((float(x) + 0.5) / 96.0, (float(y) + 0.5) / 54.0)
			var wv := wts[idx]
			cacc += pos * wv
			cxx += pos.x * pos.x * wv
			cyy += pos.y * pos.y * wv
			cw += wv
	# Fall back to the full field if the core is too thin to fit anything - a
	# nearly-flat weight field has no core, and a bad fit beats no fit here.
	if cw > wsum * 0.02:
		acc = cacc
		accxx = cxx
		accyy = cyy
		wsum = cw
	var c := acc / wsum
	# The half-width is FITTED in raw uv but BOUNDED in height units, then
	# converted back. A bound written in raw uv is a different physical size on
	# every frame shape - see _FACE_ASP_REF - and this one is what decides the
	# eye and mouth search bands below, so getting it wrong on a portrait clip
	# doesn't just mis-size the coat, it points the whole search at the wrong
	# part of the picture.
	var rx := clampf(sqrt(maxf(1e-5, accxx / wsum - c.x * c.x)) * 1.9 * fasp,
		0.06 * _FACE_ASP_REF, 0.30 * _FACE_ASP_REF) / fasp
	# The half-HEIGHT's bound scales with the frame's SHORTER side. "A face is at
	# most 0.36 of the frame's height" is a framing convention, and it is a
	# convention about the short axis: on a 16:9 clip that IS the height and this
	# is unchanged, but on a portrait clip the height covers most of a standing
	# body, so the same fraction admits an oval half again as tall as the head.
	# That matters well beyond the coat's looks - the eye and mouth search bands
	# below are struck as fractions of ry, so an ry that runs long aims them at
	# the forehead and the nose instead of the eyes and the mouth.
	var yshort: float = minf(1.0, fasp)
	var ry := clampf(sqrt(maxf(1e-5, accyy / wsum - c.y * c.y)) * 1.9,
		0.08 * yshort, 0.36 * yshort)
	var mean_lum := lum_acc / wsum
	var mean_red := red_acc / wsum
	# Blurred face mass (separable box, radius 3): an EYE is a dark spot in a
	# SKIN neighborhood. Weighting candidates by the pixel's own mass fails -
	# an eye pixel itself carries no skin chroma - and weighting by a flat
	# floor let the dark HAIR flanking the face win both clusters, which blew
	# the eye distance (and with it every feature size) up to face width.
	var wblur := PackedFloat32Array()
	wblur.resize(96 * 54)
	var wtmp := PackedFloat32Array()
	wtmp.resize(96 * 54)
	for y in 54:
		for x in 96:
			var s := 0.0
			for k in range(-3, 4):
				s += wts[y * 96 + clampi(x + k, 0, 95)]
			wtmp[y * 96 + x] = s / 7.0
	for y in 54:
		for x in 96:
			var s := 0.0
			for k in range(-3, 4):
				s += wtmp[clampi(y + k, 0, 53) * 96 + x]
			wblur[y * 96 + x] = s / 7.0
	# Second pass, inside the fitted oval only: eye clusters + mouth cluster,
	# each with its own second moment - the drawn features take their SIZES
	# from these spreads, so they breathe with the anatomy instead of holding
	# a flat mask's fixed proportions.
	var have_mo := _face_prev_lum.size() == 96 * 54
	# The eye frame from the LAST tick - stable, already smoothed, and the
	# mouth's search band rides it (see below).
	var eye_mid := (_face_eye_l_ema + _face_eye_r_ema) * 0.5
	# Eye separation in HEIGHT units (aspect-corrected) - the yardstick every
	# ratio below is expressed in.
	var eye_unit := maxf(((_face_eye_r_ema - _face_eye_l_ema)
		* Vector2(fasp, 1.0)).length(), 0.02)
	var el_acc := Vector2.ZERO
	var ela_acc := Vector2.ZERO
	var el2 := 0.0
	var el_w := 0.0
	var er_acc := Vector2.ZERO
	var era_acc := Vector2.ZERO
	var er2 := 0.0
	var er_w := 0.0
	var mo_acc := Vector2.ZERO
	var mo2x := 0.0
	var mo2y := 0.0
	var mo_w := 0.0
	var no_acc := Vector2.ZERO
	var no_w := 0.0
	var dm_sum := 0.0
	var dm_n := 0.0
	for y in 54:
		for x in 96:
			var idx := y * 96 + x
			var pos := Vector2((float(x) + 0.5) / 96.0, (float(y) + 0.5) / 54.0)
			var ex := (pos.x - c.x) / rx
			var ey := (pos.y - c.y) / ry
			if ex * ex + ey * ey > 1.3:
				continue
			var apos := Vector2(pos.x * fasp, pos.y)   # aspect-corrected twin of pos
			var dm := absf(lums[idx] - _face_prev_lum[idx]) if have_mo else 0.0
			dm_sum += dm
			dm_n += 1.0
			# EYES: darkness against the face's own mean, squared so real
			# sockets dominate faint shading, weighted by the BLURRED face
			# mass - dark-in-a-skin-neighborhood is an eye, dark-in-dark is
			# hair. Band kept off the oval's flanks for the same reason.
			if ey > -0.6 and ey < -0.05 and absf(ex) < 0.55:
				var dk := maxf(0.0, mean_lum - lums[idx])
				var we := dk * dk * wblur[idx]
				if ex < 0.0:
					el_acc += pos * we
					ela_acc += apos * we
					el2 += apos.length_squared() * we
					el_w += we
				else:
					er_acc += pos * we
					era_acc += apos * we
					er2 += apos.length_squared() * we
					er_w += we
			# This pixel in EYE-PAIR units (aspect-corrected), the frame both
			# the nose and the mouth search in.
			var mrel := ((pos - eye_mid) * Vector2(fasp, 1.0)) / eye_unit
			# NOSE: the NOSTRIL PAIR - two small dark spots between the eye
			# line and the mouth, close to the centre. It is the only
			# dependable 2D signature a nose has: the tip's highlight walks
			# around with the lighting, and deriving the nose from
			# mid-eyes-to-mouth (what this did before) inherits the error of
			# BOTH estimates, which is why the ball kept sitting off-centre
			# while the evidence-detected lips tracked fine.
			if mrel.y > 0.45 and mrel.y < 1.0 and absf(mrel.x) < 0.45:
				var nprior := exp(-pow((mrel.y - 0.85) / 0.28, 2.0))
				var wn := maxf(0.0, mean_lum - lums[idx]) * wblur[idx] * nprior
				wn *= wn   # concentrate on the actual nostril darkness (see the mouth)
				no_acc += pos * wn
				no_w += wn
			# MOUTH: MOTION first - on a talking face the mouth out-moves
			# everything else in the lower band, and unlike redness it can't
			# lock onto a warm-lit chin and sit there forever (the static
			# "swish" report). Ambient face motion (head sway, compression
			# shimmer) is subtracted; redness-over-face-mean and darkness stay
			# as tie-breakers for the quiet stretches.
			# The search band hangs off the EYE PAIR (the detection that
			# actually works), not the fitted oval: with the brightness cue
			# the oval swells over hair and shoulders, and an oval-relative
			# band put the mouth down on the chin.
			if mrel.y > 0.55 and mrel.y < 1.75 and absf(mrel.x) < 0.65:
				# Anatomical prior: the mouth sits ~1.2 eye-distances below
				# the eye line. Talking moves the whole lower face (jaw,
				# cheeks, nostrils), so raw motion alone pulled the centroid
				# up under the nose - the prior keeps it on the lips without
				# pinning it there (a wide gaussian, not a fixed offset).
				var mprior := exp(-pow((mrel.y - 1.15) / 0.38, 2.0))
				var wm := (maxf(0.0, dm - _face_motion_mean * 1.3) * 6.0
					+ maxf(0.0, reds[idx] - mean_red - 0.01) * 1.2
					+ maxf(0.0, mean_lum - lums[idx]) * 0.3) * wblur[idx] * mprior
				# SQUARED: the raw score is diffuse (compression shimmer and
				# residual head motion leak everywhere), and a diffuse weight
				# blew the cluster's variance to the clamps - the first cut
				# drew face-wide lips. Squaring concentrates centroid AND
				# spread on the actual peak: the moving mouth.
				wm *= wm
				mo_acc += pos * wm
				mo2x += pos.x * pos.x * wm
				mo2y += pos.y * pos.y * wm
				mo_w += wm
	_face_prev_lum = lums
	if dm_n > 0.0:
		_face_motion_mean = lerpf(_face_motion_mean, dm_sum / dm_n, 0.3)
	# THE TWO SMOOTHING RATES, both under the author's hand now (Steady and Firm
	# in the panel). They were constants tuned against one clip, and the right
	# value genuinely differs per clip: a locked-off shot wants heavy smoothing
	# and doesn't care about the lag, a moving one cannot afford it. Positions and
	# SIZES are separate because they fail differently - a position that lags
	# reads as the mask sliding behind the face, while a size that jitters reads
	# as the mask pulsing in and out several times a second, which is far more
	# distracting and worth a lot of lag to kill. Both mappings reproduce the old
	# constants exactly (0.35 positions, 0.30 sizes, 0.15 the coat) at the stored
	# field defaults, so nothing moves until the author moves it.
	# The fallback's own smoothing, back to the constants it was tuned with - the
	# fields that used to drive these now carry the shape knobs, and the track
	# (which needs no smoothing at all) is the path that matters.
	var a_pos := 0.35
	var a_size := 0.30
	_face_r_ema = _face_r_ema.lerp(Vector2(rx, ry), a_size * 0.5)
	_face_c_ema = _face_c_ema.lerp(c, a_pos * 0.86)
	# The EMA is a stabilizer for detection jitter, not the display's smoothing -
	# the push extrapolates (Lead), so residual EMA lag is what that covers.
	if el_w > 0.005 and er_w > 0.005:
		var el := el_acc / el_w
		var er := er_acc / er_w
		# Each eye's own spread -> its patch's own radius (eye-distance units,
		# so the shader's frame consumes it directly). Independent per side.
		var el_var := maxf(1e-6, el2 / el_w - (ela_acc / el_w).length_squared())
		var er_var := maxf(1e-6, er2 / er_w - (era_acc / er_w).length_squared())
		# Only a PLAUSIBLE pair updates the model: really apart, roughly level,
		# and the two clusters carrying COMPARABLE evidence. That last test is
		# new and it is the cheap guard against the failure the other two let
		# straight through - when one side finds a real socket and the other
		# finds a dark headscarf or a fall of hair, the two clusters' weights
		# differ by an order of magnitude even though the geometry still looks
		# fine. Holding the previous pair for a tick beats accepting half a face.
		var pair_bal: float = minf(el_w, er_w) / maxf(el_w, er_w)
		if er.x - el.x > rx * 0.3 and absf(er.y - el.y) < ry * 0.45 \
				and pair_bal > 0.12:
			# Anatomy rein: eye separation is a fraction of the face's half-width.
			# The clusters place the PAIR well but their spread leans outward
			# (sockets shade wider than pupils), so it is reined to the fitted
			# face rather than trusted raw - it's the unit every feature scales by.
			#
			# THE FLOOR IS THE DANGEROUS HALF, and it used to sit at 0.55, which
			# assumes a face looking straight down the lens. A head turned even
			# slightly has a smaller APPARENT separation - that is what turning
			# does - so the floor pushed the pair back apart to meet a frontal
			# assumption, a little further every tick as the fitted width crept up,
			# until one eye left the face altogether and landed on the forehead.
			# Dropped to a figure that still catches a COLLAPSED pair (two clusters
			# that found the same eye) without arguing with a real turned head.
			var mid := (el + er) * 0.5
			var sep := ((er - el) * Vector2(fasp, 1.0)).length()
			var facew := rx * fasp   # the face's half-width, same units as sep
			var want := clampf(sep, facew * 0.28, facew * 1.05)
			if sep > 1e-4:
				el = mid + (el - mid) * (want / sep)
				er = mid + (er - mid) * (want / sep)
			_face_eye_l_ema = _face_eye_l_ema.lerp(el, a_pos)
			_face_eye_r_ema = _face_eye_r_ema.lerp(er, a_pos)
			# Sizes in raw uv - each eye's spread stands alone, uncoupled
			# from how far apart the pair happens to read this tick.
			_face_eye_lr_ema = lerpf(_face_eye_lr_ema,
				clampf(sqrt(el_var) * 1.9, 0.015, 0.07), a_size)
			_face_eye_rr_ema = lerpf(_face_eye_rr_ema,
				clampf(sqrt(er_var) * 1.9, 0.015, 0.07), a_size)
	if mo_w > 0.0001:
		var mo := mo_acc / mo_w
		_face_mouth_ema = _face_mouth_ema.lerp(mo, a_pos)
		# The mouth's own width and height, separately - talking changes the
		# vertical spread far more than the horizontal, and the lips follow.
		var mvx := maxf(1e-6, mo2x / mo_w - mo.x * mo.x)
		var mvy := maxf(1e-6, mo2y / mo_w - mo.y * mo.y)
		# Caps ride the eye distance - a BOUND, not a normalization (the
		# stored value stays raw uv, so the lips never rescale in lockstep
		# with the eyes; this only stops a diffuse tick drawing a face-wide
		# blob).
		# Stored in raw uv (the shader's contract) but bounded in height units on
		# the x axis, same rule as rx above: the floor was a raw-uv constant, so
		# on a portrait clip it bounded a width to a third of what it means on
		# 16:9. The ceiling already rode eye_unit (height units) and only needed
		# the /fasp moved out of it.
		_face_mouth_r_ema = _face_mouth_r_ema.lerp(Vector2(
			clampf(sqrt(mvx) * 1.9 * fasp, 0.012 * _FACE_ASP_REF, eye_unit * 0.62) / fasp,
			clampf(sqrt(mvy) * 1.9, 0.008, eye_unit * 0.40)), a_size)
	if tint_acc.length() > 1e-4:
		_face_tint_ema = _face_tint_ema.lerp((tint_acc / wsum).normalized(), 0.1)
	_face_lum_ema = lerpf(_face_lum_ema, mean_lum, 0.1)
	_face_red_ema = lerpf(_face_red_ema, mean_red, 0.1)
	# The nose is DERIVED (mid-eyes toward mouth) but smoothed on its own,
	# slower clock - it must not inherit the eye pair's tick-to-tick jitter.
	# The nostril centroid is the BASE of the nose; the ball sits on the tip,
	# just above it. Falls back to the old mid-eyes-to-mouth interpolation
	# (~55% down) only when no nostril evidence turns up at all - flat
	# lighting, a raised chin, a profile.
	var nose_target := ((_face_eye_l_ema + _face_eye_r_ema) * 0.5).lerp(_face_mouth_ema, 0.55)
	var nose_alpha := a_pos * 0.43
	if no_w > 0.0001:
		nose_target = (no_acc / no_w) - Vector2(0.0, 0.14 * eye_unit)
		nose_alpha = a_pos * 0.86
	# A wide sanity box around the face's own centre line - detection may be
	# imperfect, but a nose never lands out on a cheek.
	var nose_cx := (eye_mid.x + _face_mouth_ema.x) * 0.5
	nose_target.x = clampf(nose_target.x, nose_cx - eye_unit * 0.30 / fasp,
		nose_cx + eye_unit * 0.30 / fasp)
	_face_nose_ema = _face_nose_ema.lerp(nose_target, nose_alpha)
	if OS.has_environment("GHOST_FACE_DEBUG"):
		var em := (_face_eye_l_ema + _face_eye_r_ema) * 0.5
		var eu := maxf(((_face_eye_r_ema - _face_eye_l_ema) * Vector2(fasp, 1.0)).length(), 0.02)
		print("FACEDBG t=%.2f eyeL=%.3f,%.3f eyeR=%.3f,%.3f unit=%.3f mouth=%.3f,%.3f mouthU=%.2f moW=%.4f szL=%.3f szR=%.3f mr=%.3f,%.3f nose=%.3f,%.3f noW=%.4f" % [
			_player.stream_position if _player != null else 0.0,
			_face_eye_l_ema.x, _face_eye_l_ema.y, _face_eye_r_ema.x, _face_eye_r_ema.y, eu,
			_face_mouth_ema.x, _face_mouth_ema.y, (_face_mouth_ema.y - em.y) / eu, mo_w,
			_face_eye_lr_ema, _face_eye_rr_ema, _face_mouth_r_ema.x, _face_mouth_r_ema.y,
			_face_nose_ema.x, _face_nose_ema.y, no_w])


# --- the audio effect: the timeline, applied to the sound -----------------------
# The one effect here that draws nothing. A mask session already owns its clip's
# audio - it plays it, fades it, and the export mixes it - so the timeline is
# already the right place to say "from here, wetter", and a marker's envelope is
# already the right shape for it: a reverb that arrives over a ramp is precisely
# what a ramp is for.
#
# Everything runs on ONE dedicated bus routed to Master, built once and left in
# place. Godot's own effects rather than hand-written DSP: they are sample-exact,
# they cost nothing on the main thread, and - the part that matters for this app -
# the export relaunch mixes through the same bus, so a rendered file carries the
# same sound the editor previewed. A hand-rolled process() on the video thread
# would not survive the export's fixed-fps clock at all.
#
# The chain's ORDER is the usual mixing order and is not arbitrary:
#   EQ          shape the tone first, so everything downstream reacts to the tone
#               you actually want (bass lifted here is bass the compressor hears)
#   COMPRESSOR  then even it out - this is what "punchy" is: not more bass, but
#               bass whose transients survive next to everything else
#   DELAY       echoes of the shaped, evened signal
#   REVERB      the room goes last, around all of it, as a room does
const MASK_BUS := "MaskFX"

var _bus_idx := -1
var _fx_eq: AudioEffectEQ6
var _fx_comp: AudioEffectCompressor
var _fx_delay: AudioEffectDelay
var _fx_reverb: AudioEffectReverb
## The room's SETTINGS, shared with the mode that has no bus. See [RoomFX]: the
## dial's range, its taper and what Resonance does to the tail are decided there
## and rendered here onto Godot's own reverb, so Generative's Room slider is the
## same control rather than a second one wearing the name.
var _room := RoomFX.new()


func _ensure_audio_bus() -> void:
	if _bus_idx >= 0 and AudioServer.get_bus_index(MASK_BUS) == _bus_idx:
		return
	_bus_idx = AudioServer.get_bus_index(MASK_BUS)
	if _bus_idx < 0:
		AudioServer.add_bus()
		_bus_idx = AudioServer.get_bus_count() - 1
		AudioServer.set_bus_name(_bus_idx, MASK_BUS)
		AudioServer.set_bus_send(_bus_idx, "Master")
	# Rebuild the chain from scratch: a bus that survived a reload with a partial
	# chain is worse than no bus, and these are cheap to make.
	while AudioServer.get_bus_effect_count(_bus_idx) > 0:
		AudioServer.remove_bus_effect(_bus_idx, 0)
	_fx_eq = AudioEffectEQ6.new()
	_fx_comp = AudioEffectCompressor.new()
	_fx_delay = AudioEffectDelay.new()
	_fx_reverb = AudioEffectReverb.new()
	for fx in [_fx_eq, _fx_comp, _fx_delay, _fx_reverb]:
		AudioServer.add_bus_effect(_bus_idx, fx)
	_apply_audio_fx({})   # neutral until a marker says otherwise
	if _audio != null:
		_audio.bus = MASK_BUS


## Resolve one audio layer onto the bus. `l` is the layer dict (empty = neutral).
## Every parameter is driven through the layer's own envelope in `env`, so an
## audio marker fades its sound in and out exactly like a visual one fades its
## paint - no separate concept, no separate code path.
func _apply_audio_fx(l: Dictionary) -> void:
	if _fx_eq == null:
		return
	var amt := clampf(float(l.get("env", 0.0)) * float(l.get("intensity_a", 1.0)), 0.0, 1.0)
	# BASS. Two bands lifted together rather than one, because a single band's
	# bump is audible as a resonance rather than as weight. EQ6's bands are
	# 32/100/320/1000/3200/10000 Hz; the bottom two are the ones a voice's chest
	# and a kick's body live in.
	var bass := clampf(float(l.get("fx_stick", 0.0)), 0.0, 1.0) * amt
	_fx_eq.set_band_gain_db(0, bass * 12.0)
	_fx_eq.set_band_gain_db(1, bass * 8.0)
	# PUNCH is the compressor, and it rides the same knob - because "punchy" is
	# not more bass, it is bass that keeps its transient. Raising the bass alone
	# just makes a mix muddy, which is the trap this avoids by moving both.
	_fx_comp.threshold = lerpf(0.0, -18.0, bass)
	_fx_comp.ratio = lerpf(1.0, 5.0, bass)
	_fx_comp.gain = bass * 5.0
	_fx_comp.attack_us = lerpf(20.0, 8.0, bass)     # fast enough to catch a transient
	_fx_comp.release_ms = lerpf(250.0, 120.0, bass)   # ms, unlike attack_us - the two differ
	# ECHO. Lag is the time between repeats, Bleed how much comes back.
	var echo := clampf(float(l.get("fx_smooth", 0.0)), 0.0, 1.0) * amt
	_fx_delay.dry = 1.0
	_fx_delay.tap1_active = echo > 0.001
	_fx_delay.tap1_delay_ms = lerpf(60.0, 620.0, clampf(float(l.get("fx_lag", 0.35)), 0.0, 1.0))
	_fx_delay.tap1_level_db = linear_to_db(maxf(echo * 0.7, 0.0001))
	# The second tap at a non-multiple of the first, so repeats interleave into a
	# decay instead of landing on top of each other as one loud slap.
	_fx_delay.tap2_active = echo > 0.001
	_fx_delay.tap2_delay_ms = _fx_delay.tap1_delay_ms * 1.63
	_fx_delay.tap2_level_db = linear_to_db(maxf(echo * 0.45, 0.0001))
	_fx_delay.feedback_active = echo > 0.001
	_fx_delay.feedback_level_db = linear_to_db(maxf(echo * 0.35, 0.0001))
	# RESONANCE, as the delay's own feedback colour: a tuned, ringing repeat
	# rather than a flat one. Low-passing the feedback path is what makes a tail
	# sound like a space rather than like a copy.
	var reso := clampf(float(l.get("fx_contrast", 0.5)), 0.0, 1.0)
	_fx_delay.feedback_lowpass = lerpf(16000.0, 900.0, reso * amt)
	# AMBIENCE. Room size and wet are separate on purpose - a big dry room and a
	# small wet one are different sounds, and collapsing them into one "reverb"
	# slider is what makes every preset sound the same.
	#
	# The mapping itself lives in [RoomFX] rather than here, because Generative has
	# the same Room now and cannot use this bus (it bakes its effects into a take
	# WAV; see the class docs). Only the RENDERER is mode-specific: this hands the
	# settings to Godot's reverb, that one runs the same network per sample.
	# Note the size is NOT scaled by the envelope while the wet is - a marker fades
	# a room IN, it does not grow the walls.
	_room.size = float(l.get("fx_scale", 1.0))
	_room.wet = clampf(float(l.get("fx_density", 0.45)), 0.0, 1.0) * amt
	_room.resonance = reso
	_room.to_reverb(_fx_reverb)


# --- face track: bootstrap, run, poll -------------------------------------------

## MediaPipe's canonical face-mesh indices, only the ones this effect consumes.
## Named rather than inlined because a bare 263 in the middle of a size
## calculation is unreadable and unverifiable.
## LEFT/RIGHT here are the IMAGE's left and right, not the subject's.
## CONTOUR POINT SETS, not ordered rings. Every shape below is rasterized as the
## CONVEX HULL of its set (see _ft_hull), which cannot self-intersect however the
## indices are ordered - and an ordering mistake is otherwise invisible until you
## look at a render and find a bow-tie where a nose should be, which is exactly
## what a hand-ordered nose ring produced here first time.
##
## The indices are MediaPipe's canonical face-mesh numbering. They are written out
## because mediapipe 1.0.1 is Tasks-only and no longer ships the FACEMESH_* tables
## the older `solutions` module carried; the numbering itself is fixed by the
## canonical model and has not changed since the mesh was published. Verified by
## drawing all four on real footage - the oval follows the jaw, the eyes come out
## almond rather than round, the lips sit exactly on the lips.
const FT_EYE_L := [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
const FT_EYE_R := [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
const FT_LIPS := [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
const FT_OVAL := [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
	378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
## THE BALL OF THE NOSE, not the nose. This used to open at 168 - the nasion,
## the dip BETWEEN THE EYES - and run the whole dorsum down: 168, 6, 197, 195 are
## the bridge chain, and their convex hull is a wedge covering the entire length
## of it, so the red streaked from the tip up to the brow. A clown paints the ball
## and nothing else. What is left is the supratip and tip (5, 4, 1), the columella
## and subnasale (19, 94, 2), and the alae with their creases (48, 115, 220, 45 and
## 278, 344, 440, 275) - a hull around the round part, which is the shape being
## drawn. Scale still grows it from there.
## THE BALL OF THE NOSE, not the nose. This used to open at 168 - the nasion, the
## dip BETWEEN THE EYES - and run the whole dorsum down: 168, 6, 197 and 195 are
## the bridge chain, and their convex hull is a wedge covering its entire length,
## so the red streaked from the tip up to the brow. Most clowns paint the ball and
## nothing else. What is left is the supratip and tip (5, 4, 1), the columella and
## subnasale (19, 94, 2) and the alae with their creases (48, 115, 220, 45 and 278,
## 344, 440, 275) - a hull around the round part. Scale still grows it from there.
const FT_NOSE := [5, 4, 1, 19, 94, 2, 48, 115, 220, 45, 278, 344, 440, 275]
const FT_NOSE_TIP := 1
const FT_NOSE_BRIDGE := 168
const FT_FACE_TOP := 10
const FT_FACE_CHIN := 152
const FT_FACE_LEFT := 234
const FT_FACE_RIGHT := 454


# --- THE FEATURE STENCIL: the contours, rasterized ------------------------------
# WHAT THIS REPLACES. The paint sim used to deposit through analytic ellipses -
# `win(p, centre, radius)` - which is why a nose could only ever be a circle and a
# mouth an ellipse, however well they were placed. An ellipse has no cheekbone, no
# eye corner and no cupid's bow, so at best it sat over the feature rather than on
# it. The landmarks describe the actual outlines, so the deposit now goes through
# a RASTERIZED STENCIL of those outlines instead: four polygons drawn once per
# tick into a small texture, one per channel.
#
#   R  the two eyes      G  the lips      B  the nose      A  the face oval
#
# Drawn on the GPU into a SubViewport (additively, so each polygon writes its own
# channel and nothing clobbers a neighbour) rather than scanline-filled on the CPU,
# because it is four convex polygons a frame and the GPU antialiases them for free.
# The paint sim then reads a shape instead of evaluating an ellipse, and every
# downstream behaviour it already has - advection, bleed, settle, the drip - works
# on the real silhouette without knowing anything changed.
const _STENCIL_H := 384      # tall enough for a face, small enough to be free

var _stencil_vp: SubViewport
var _stencil_draw: Node2D
var _stencil_cut: Node2D
var _stencil_cuts: Array = []
var _stencil_shapes := []    # [eyes, lips, nose, oval] as PackedVector2Array, in VIEWPORT px


func _ensure_stencil() -> void:
	if _stencil_vp != null:
		return
	var asp := _source_aspect()
	_stencil_vp = SubViewport.new()
	_stencil_vp.size = Vector2i(maxi(2, int(round(_STENCIL_H * asp))), _STENCIL_H)
	_stencil_vp.disable_3d = true
	_stencil_vp.transparent_bg = true
	# CLEARED EVERY DRAW, unlike the paint field next door: this is a statement of
	# where the features ARE this instant, not an accumulation. The persistence
	# lives in the paint sim that reads it.
	_stencil_vp.render_target_clear_mode = SubViewport.CLEAR_MODE_ALWAYS
	_stencil_vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	_stencil_draw = Node2D.new()
	var cm := CanvasItemMaterial.new()
	# PREMULTIPLIED, not ADD. Godot's additive blend multiplies the source by its
	# own alpha before adding, so a colour written to isolate one channel -
	# (1,0,0,0) for the eyes - contributes exactly nothing, and the first cut of
	# this drew only the oval (the one shape whose colour had any alpha in it).
	# In premultiplied mode the colour is taken as written: dst = src + dst*(1-a),
	# so an a=0 colour adds into RGB and leaves the alpha channel alone.
	cm.blend_mode = CanvasItemMaterial.BLEND_MODE_PREMULT_ALPHA
	_stencil_draw.material = cm
	_stencil_draw.draw.connect(_draw_stencil)
	_stencil_vp.add_child(_stencil_draw)
	# THE EYE OPENINGS ARE CUT OUT OF THE COAT, in a second pass, because
	# premultiplied blending cannot REMOVE alpha - `dst = src + dst * (1 - src.a)`
	# leaves the alpha channel alone at src.a = 0 and REPLACES it at src.a = 1, and
	# there is no colour in between that subtracts. A negative colour clamps (that
	# was tried). So the openings are drawn by a child with a SUBTRACTIVE material,
	# which children being drawn after their parent puts in exactly the right
	# order: dst.a - 1 = 0 inside each opening, and rgb untouched because the
	# colour's rgb is zero.
	#
	# What it is FOR: the eye patch is an annulus so a clown paints AROUND the eye,
	# and that worked - but the white coat underneath went on covering the opening,
	# so the hole in the black revealed white instead of an eyeball. Same fault as
	# the black over the eyeball, with the colour inverted. A hole has to be a hole
	# in the whole mask.
	_stencil_cut = Node2D.new()
	var cut_m := CanvasItemMaterial.new()
	cut_m.blend_mode = CanvasItemMaterial.BLEND_MODE_SUB
	_stencil_cut.material = cut_m
	_stencil_cut.draw.connect(_draw_stencil_cut)
	_stencil_vp.add_child(_stencil_cut)
	add_child(_stencil_vp)


## The eye openings, subtracted from the coat's silhouette. See _ensure_stencil
## for why this is a separate pass and not another entry in _stencil_shapes.
## The Follow switch, out of and into user://ghost.cfg. Defaults to ON, which is
## the behaviour every session had before the switch existed.
func _load_follow() -> bool:
	var cfg := ConfigFile.new()
	if cfg.load(PREFS_CFG) != OK:
		return true
	return bool(cfg.get_value("mask", "follow_playhead", true))


func _save_follow(on: bool) -> void:
	var cfg := ConfigFile.new()
	cfg.load(PREFS_CFG)     # read-modify-write - see PREFS_CFG
	cfg.set_value("mask", "follow_playhead", on)
	cfg.save(PREFS_CFG)


func _draw_stencil_cut() -> void:
	for pts in _stencil_cuts:
		if pts.size() >= 3:
			_stencil_cut.draw_colored_polygon(pts, Color(0, 0, 0, 1))


func _draw_stencil() -> void:
	# One channel each, and the two EYES are separate polygons sharing a channel -
	# a single hull around both is a bandit's mask across the bridge of the nose.
	# Additive blending lets the eye and nose polygons overlap the oval without
	# either erasing the other, which a normal alpha blend would.
	for entry in _stencil_shapes:
		var pts: PackedVector2Array = entry[0]
		# DEGENERATE POLYGONS ARE SKIPPED, not handed to the renderer. A ring is
		# decomposed into triangles between two hulls, and wherever those hulls
		# touch - a collapsed corner, a vertex flattened onto the bridge line beside
		# its neighbour - the triangle has no area, and Godot answers that with
		# "Invalid polygon data, triangulation failed" once per frame in the log.
		# The shape is invisible either way; the difference is a clean console.
		if pts.size() >= 3 and absf(_poly_area(pts)) > 1e-9:
			_stencil_draw.draw_colored_polygon(pts, entry[1])


## Twice the signed area - the sign is unused, only the magnitude, as a
## degeneracy test.
static func _poly_area(pts: PackedVector2Array) -> float:
	var acc := 0.0
	for i in pts.size():
		var p0 := pts[i]
		var p1 := pts[(i + 1) % pts.size()]
		acc += p0.x * p1.y - p1.x * p0.y
	return acc * 0.5


## The convex hull of a landmark set at `t`, in frame UV. Hull rather than the
## points in order: see the FT_* sets for why.
func _ft_hull(ring: Array, t: float) -> PackedVector2Array:
	var pts := PackedVector2Array()
	for i in ring:
		pts.append(_ft_point(int(i), t))
	return Geometry2D.convex_hull(pts)


## The RING between two hulls, as quads - paint AROUND a feature rather than over
## it. Both hulls come from _grow_hull of the same source, so they have matching
## point counts and correspondence, and the ring is one quad per edge.
##
## Built geometrically because it cannot be drawn. Punching a hole by drawing the
## inner shape in a negative colour reads as the obvious trick and does not work:
## on a normal target the value clamps and the hole stays solid, and on an HDR one
## it comes back at 0.92 rather than 0. Godot's polygon drawing has no hole
## support either. Quads between corresponding points have neither problem and are
## exact.
static func _ring_quads(outer: PackedVector2Array, inner: PackedVector2Array) -> Array:
	var out := []
	if outer.size() < 3 or outer.size() != inner.size():
		return out
	# TWO TRIANGLES per segment, not one quad. A quad in the order
	# (outer[i], outer[j], inner[j], inner[i]) is a valid trapezoid on paper, but
	# where the hull is nearly pointed - the corners of an eye - the four points
	# are close to collinear and the drawn polygon comes out as a BOW-TIE. Rendered
	# and looked at, the "ring" was a self-crossing outline rather than a band.
	# Triangles cannot cross themselves whatever the points do.
	for i in outer.size():
		var j := (i + 1) % outer.size()
		out.append(PackedVector2Array([outer[i], outer[j], inner[j]]))
		out.append(PackedVector2Array([outer[i], inner[j], inner[i]]))
	return out


## THE UNIT ONE STEP OF EYE SIZE IS WORTH, in aspect space. Mostly the hull's
## SHORT reach - for an eye, lid to lash - with a share of its long one.
##
## Neither alone works. Pure minor (0.0169 on the measured eye) makes Eye size
## barely move the patch past the corners, and covering the skin OUTSIDE the eye is
## what the control is for. Pure major (0.1406, eight times larger) inflates the
## short axis out of all proportion and the patch becomes a blob. The mix keeps the
## patch growing rounder rather than longer - which is the point of offsetting at
## all instead of scaling - while still reaching out across the socket.
static func _hull_step(pts: PackedVector2Array, asp: float) -> float:
	if pts.is_empty():
		return 0.0
	var a := Vector2(asp, 1.0)
	var c := Vector2.ZERO
	for p in pts:
		c += p * a
	c /= float(pts.size())
	var lo := 1e9
	var hi := 0.0
	for p in pts:
		var d := (p * a - c).length()
		lo = minf(lo, d)
		hi = maxf(hi, d)
	return lo * 0.35 + hi * 0.28


## Offset a convex hull outward by `d` along its own normals (negative = inward),
## in aspect space so the band is the same width at the top as at the side.
##
## THIS IS WHY THE PATCH GROWS ROUNDER RATHER THAN LONGER. The eye hull measures
## 0.141 along its major axis and 0.0169 across - 8.3 to 1 - so SCALING it about
## its centroid (which is what Eye size used to do) multiplies that length by the
## same factor as its height, and at the stored 2.2x the patch reaches a fifth of
## the way across the face while staying almost as thin. Its ends become long
## spikes, and the ring between two such hulls is a spike too, not a band: the
## reported "it's cutting that inner eye too short - the black ring must go all the
## way around". An offset adds the same distance in every direction, so the shape
## fills out instead of stretching, and the ring has one width the whole way round.
##
## THE UNIT IS THE MINOR RADIUS, and getting that wrong is the whole difficulty: a
## first cut offset by the hull's MEAN radius, which on an 8-to-1 lens is dominated
## by the long axis, and the patch inflated into an angular blob three times the
## size of the eye socket (rendered, looked at, reverted). The miter is clamped
## tightly for the same reason - the exact miter at a lens's corner runs away
## toward infinity, and an unclamped one is what put the spikes back.
static func _offset_hull(pts: PackedVector2Array, d: float, asp: float) -> PackedVector2Array:
	var m := pts.size()
	if m < 3 or absf(d) < 1e-6:
		return pts
	var a := Vector2(asp, 1.0)
	var area := 0.0
	for i in m:
		var q0: Vector2 = pts[i] * a
		var q1: Vector2 = pts[(i + 1) % m] * a
		area += q0.x * q1.y - q1.x * q0.y
	var w := 1.0 if area > 0.0 else -1.0
	var out := PackedVector2Array()
	for i in m:
		var p_prev: Vector2 = pts[(i - 1 + m) % m] * a
		var p: Vector2 = pts[i] * a
		var p_next: Vector2 = pts[(i + 1) % m] * a
		var e0 := (p - p_prev).normalized()
		var e1 := (p_next - p).normalized()
		if e0.length() < 0.5 or e1.length() < 0.5:
			out.append(pts[i])
			continue
		var bis := Vector2(e0.y, -e0.x) * w + Vector2(e1.y, -e1.x) * w
		if bis.length() < 1e-5:
			out.append(pts[i])
			continue
		bis = bis.normalized()
		out.append((p + bis * (d / maxf(bis.dot(Vector2(e1.y, -e1.x) * w), 0.62))) / a)
	return out


## Squeeze a hull against a line rather than cutting it on one: vertices are
## compressed smoothly as they approach `lim` and asymptote to it, so the line is
## PRESSURE against expanding further rather than a wall. Nothing ever reaches it.
##
## A hard clamp was tried first and it works geometrically - the two eye patches
## stop short of the bridge - but every vertex past the line lands exactly ON it,
## which draws a straight edge down the inner side of each socket. Reported as a
## cut, three times, in three different disguises. The soft version has no edge to
## see: the hull just gets denser as it nears the line, so the paint fills in and
## thins out instead of stopping.
##
## The response is `x / (1 + x)` - saturating, monotone, and with a slope of 1 at
## the knee, so it joins the untouched part of the hull without a crease. The
## vertex COUNT is preserved either way, which _ring_quads requires: it pairs the
## outer and inner hulls by index and draws nothing at all if their sizes differ.
static func _press_hull(pts: PackedVector2Array, origin: Vector2, n: Vector2,
		lim: float, knee: float, asp: float) -> PackedVector2Array:
	if knee <= 1e-6:
		return pts
	var a := Vector2(asp, 1.0)
	var start := lim - knee
	var out := PackedVector2Array()
	for p in pts:
		var q: Vector2 = p * a
		var d := (q - origin).dot(n)
		if d <= start:
			out.append(p)
			continue
		var x := (d - start) / knee
		out.append((q + n * (start + knee * (x / (1.0 + x)) - d)) / a)
	return out


## How far a hull reaches along `n` from `origin`.
static func _hull_extent(pts: PackedVector2Array, origin: Vector2, n: Vector2,
		asp: float) -> float:
	var a := Vector2(asp, 1.0)
	var m := -1e9
	for p in pts:
		m = maxf(m, (p * a - origin).dot(n))
	return m


## Push the inner hull away from the outer wherever the two have converged closer
## than `min_d`, so the ring between them has a real width everywhere - including
## at the corners, where two concentric SCALED copies of a pointed hull always
## meet. Vertex for vertex, so the count is preserved (_ring_quads pairs them by
## index and draws nothing at all if the sizes differ), and untouched wherever the
## gap is already wide enough - which is everywhere but the corners, so Hollow goes
## on sizing the opening exactly as before.
static func _open_band(outer: PackedVector2Array, inner: PackedVector2Array,
		min_d: float, asp: float) -> PackedVector2Array:
	if outer.size() != inner.size() or outer.is_empty() or min_d <= 0.0:
		return inner
	var a := Vector2(asp, 1.0)
	var c := Vector2.ZERO
	for p in inner:
		c += p * a
	c /= float(inner.size())
	var out := PackedVector2Array()
	for i in inner.size():
		var qi: Vector2 = inner[i] * a
		var qo: Vector2 = outer[i] * a
		var v := qi - qo
		if v.length() >= min_d:
			out.append(inner[i])
			continue
		# Toward the hull's own centre if the two points coincide, which is what
		# happens at a corner where the scaling has collapsed the gap entirely.
		var dir := v.normalized() if v.length() > 1e-6 else (c - qo).normalized()
		out.append((qo + dir * min_d) / a)
	return out


## Grow a hull, but never past a line. Same as _grow_hull except that each vertex
## is allowed only as much growth as keeps it on the near side of the half-plane
## `dot(p * asp - origin, n) <= lim` - so the hull swells freely in every direction
## except the one that is bounded, where it flattens against the line.
##
## THE TWO EYE PATCHES MUST NOT MEET ACROSS THE BRIDGE OF THE NOSE. _grow_hull
## scales about a hull's own centroid and knows nothing about the other eye, so at
## the stored Eye size (2.2x) each patch grew that far toward the nose as well and
## the pair fused into one bandit's mask - the exact fault clown_scale_check was
## written for, which bounded the ANALYTIC radii and never saw this path, because
## the stencil replaced it. Smudge then rubs the edges outward on top, so the
## clearance has to account for it too (see the caller).
##
## Capping the growth PER VERTEX rather than capping k for the whole hull is what
## keeps the patch its full size: a single k low enough to clear the bridge shrinks
## the outer corner by the same fraction, and the outer corner is the part the
## author wants big. Vertex COUNT is preserved either way, which _ring_quads
## requires - it pairs the outer and inner hulls by index and silently draws
## nothing at all if their sizes differ.
static func _grow_hull_bounded(pts: PackedVector2Array, k: float, origin: Vector2,
		n: Vector2, lim: float, asp: float) -> PackedVector2Array:
	if pts.is_empty():
		return pts
	var a := Vector2(asp, 1.0)
	var c := Vector2.ZERO
	for p in pts:
		c += p
	c /= float(pts.size())
	# How far the centroid already sits from the line. If it is past it there is
	# nothing sensible to do, so growth is simply not allowed to make it worse.
	var head := (c * a - origin).dot(n) - lim
	var out := PackedVector2Array()
	for p in pts:
		var b := ((p - c) * a).dot(n)
		var kp := k
		if b > 1e-6:
			# FLOORED AT 1.0 - THE MEASURED CONTOUR IS NEVER CLIPPED. Without this
			# floor the cap eats into the eye itself whenever the clearance is wider
			# than the space between the eye's inner corner and the midline, and it
			# is: the reported result was the inner half of the patch simply gone,
			# with the eye opening cut in two. Growth is what may be limited here,
			# never the outline the growth started from.
			kp = clampf(-head / b, 1.0, k)
		out.append(c + (p - c) * kp)
	return out


## Grow a hull about its own centroid. The features are drawn from measured
## outlines now, so "bigger eyes" has to mean a bigger version of THIS eye rather
## than a bigger circle - scaling the hull keeps the shape and only changes how
## much of the face it covers.
static func _grow_hull(pts: PackedVector2Array, k: float) -> PackedVector2Array:
	if pts.is_empty() or absf(k - 1.0) < 0.001:
		return pts
	var c := Vector2.ZERO
	for p in pts:
		c += p
	c /= float(pts.size())
	var out := PackedVector2Array()
	for p in pts:
		out.append(c + (p - c) * k)
	return out


## THE SMILE. A clown's mouth is not the wearer's mouth - the Joker's runs most of
## the way to his ears and curves up past where any face bends. So the lip hull is
## stretched horizontally about its own centre and its outer ends are swept along a
## curve, before rasterizing. Width 1 and curve 0 leave the real mouth exactly as
## measured; both push far past anatomy on purpose, because that is the look.
## The curve is applied proportionally to the SQUARE of the horizontal distance
## from centre, so the middle of the lips stays put and the corners travel - which
## is how a smile actually deforms, and what keeps it a mouth rather than a banana.
static func _smile_hull(pts: PackedVector2Array, width: float, curve: float) -> PackedVector2Array:
	if pts.is_empty():
		return pts
	var c := Vector2.ZERO
	for p in pts:
		c += p
	c /= float(pts.size())
	var half := 0.0001
	for p in pts:
		half = maxf(half, absf(p.x - c.x))
	var out := PackedVector2Array()
	for p in pts:
		var dx := (p.x - c.x) * width
		var u := clampf(dx / (half * maxf(width, 0.001)), -1.0, 1.0)
		out.append(Vector2(c.x + dx, p.y + curve * u * u))
	return out


## Rebuild the stencil for clip time `t`. Cheap enough to run every frame: four
## convex hulls of at most 36 points, then one small viewport draw.
func _update_stencil(t: float) -> void:
	if _ft_state != "ready":
		return
	_ensure_stencil()
	var sz := Vector2(_stencil_vp.size)
	var eyes := PackedVector2Array()
	# The two eyes share a channel but must NOT share a hull - one hull around both
	# is a bandit's mask across the bridge of the nose.
	# EACH EYE IS A RING, not a patch: a clown paints around the eye, not over the
	# eyeball. Hollow sizes the opening.
	#
	# BOTH RADII ARE RELATIVE TO THE MEASURED EYE OPENING, and getting that wrong
	# is why the paint still covered the eyeballs after the ring was added. The
	# outer hull was 1.0x the opening and the hole ran 0 -> 0.82x, so even at full
	# Hollow the "ring" was a thin band between 82% and 100% of the eye - entirely
	# ON the eyeball, with nothing at all around it. Backwards. The patch has to be
	# comfortably BIGGER than the eye (a clown's is), and the hole has to reach
	# slightly PAST it so the eyeball is genuinely clear rather than fringed by the
	# landmark's own error.
	var eye_l_src := _ft_hull(FT_EYE_L, t)
	var eye_r_src := _ft_hull(FT_EYE_R, t)
	# 0.9x the eye opening at the stored default, widening to 1.5x. It used to run
	# from 0, i.e. NO hole - so out of the box the patch was solid over the eyeball,
	# the control that fixes it sat at its inert end, and there was no way to tell
	# from the panel that it was the one to reach for. A clown paints around the
	# eye; that is the default now, and the slider widens the opening from there.
	var hole := 0.9 + clampf(_clown_hollow, 0.0, 1.0) * 0.6
	# THE EYE PATCH IS THE MEASURED EYE, SCALED - the original geometry, restored.
	# Three attempts to improve on it each traded one fault for another and are
	# recorded here so none of them comes back: bounding the growth against a line
	# through the bridge (to stop the two patches fusing over the nose) emptied the
	# INNER side; rebuilding both hulls as normal offsets fixed the band but made
	# the patch an angular blob, because an eye hull is 8 to 1 and an offset in any
	# averaged unit inflates the short axis out of proportion; and gating the paint
	# on the coat's alpha (to keep a wide Smudge off the eyeball) cut a straight
	# edge down each socket. Scaling wraps the eye all the way round, which is the
	# property that actually matters, and the one thing it genuinely gets wrong -
	# a ring between two SCALED copies of a pointed hull pinches to nothing at the
	# corners - is fixed by _open_band, which is a few lines and touches nothing
	# else.
	var eye_l := _grow_hull(eye_l_src, _clown_eye_size)
	var eye_r := _grow_hull(eye_r_src, _clown_eye_size)
	var eye_l_in := _grow_hull(eye_l_src, hole)
	var eye_r_in := _grow_hull(eye_r_src, hole)
	var asp_e := _source_aspect()
	var band_l := _hull_step(eye_l_src, asp_e) * 0.45
	var band_r := _hull_step(eye_r_src, asp_e) * 0.45
	# NOTHING HOLDS THE HULLS BACK FROM THE BRIDGE. Three versions tried to - a
	# per-vertex growth cap, a hard flatten against a line, then a soft asymptotic
	# squeeze - and all three drew a visible edge down the inner side of each
	# socket, because a hull here has SIXTEEN vertices and no operation on sixteen
	# points can be soft. The eye patches are simply the measured eyes, grown; the
	# bridge is kept bare per PIXEL instead, by a wide fade in the deposit (see
	# clown_paint's `bridge pressure`), which has as many samples as the frame has
	# texels and can therefore actually be gradual.
	var lips := _smile_hull(_ft_hull(FT_LIPS, t), _clown_smile_w, _clown_smile_curve)
	var nose := _ft_hull(FT_NOSE, t)
	# THE COAT'S OUTLINE IS DILATED A LITTLE past the measured jawline. Two soft
	# edges sit between the polygon and the picture - the deposit reads the
	# rasterized alpha through a smoothstep, and mask_split gates the whole layer
	# on the coat field through another - and each costs a pixel or two at the
	# boundary. Growing the hull moves BOTH of those outside the visible jaw, so
	# the jaw itself is interior and fully painted. It cannot spill onto the
	# background: the layer is separately gated on matching the face's own colour
	# (match16 in the clown branch), so paint that exists past the jaw in the
	# FIELD simply has nothing to draw on.
	var oval := _grow_hull(_ft_hull(FT_OVAL, t), 1.015)
	const EYE := Color(1, 0, 0, 0)
	# Ring quads when Hollow has opened one, the whole hull when it has not.
	var eye_shapes := []
	if hole > 0.02:
		for quad in _ring_quads(_to_px(eye_l, sz), _to_px(eye_l_in, sz)):
			eye_shapes.append([quad, EYE])
		for quad in _ring_quads(_to_px(eye_r, sz), _to_px(eye_r_in, sz)):
			eye_shapes.append([quad, EYE])
	else:
		eye_shapes.append([_to_px(eye_l, sz), EYE])
		eye_shapes.append([_to_px(eye_r, sz), EYE])
	# THE OVAL GOES FIRST. Its colour is the only one carrying alpha, and in
	# premultiplied blending an a=1 source REPLACES what is under it - drawn last
	# it would wipe the three channels drawn before it.
	_stencil_shapes = [[_to_px(oval, sz), Color(0, 0, 0, 1)]]
	_stencil_shapes.append_array(eye_shapes)
	_stencil_shapes.append([_to_px(lips, sz), Color(0, 1, 0, 0)])
	_stencil_shapes.append([_to_px(nose, sz), Color(0, 0, 1, 0)])
	# The eye openings come out of the COAT as well as out of the black - see
	# _ensure_stencil. Cut at the ring's own inner hull, so the hole in the white
	# is exactly the hole in the black and Hollow sizes both.
	_stencil_cuts = []
	if hole > 0.02:
		_stencil_cuts = [_to_px(eye_l_in, sz), _to_px(eye_r_in, sz)]
	_stencil_vp.render_target_update_mode = SubViewport.UPDATE_ONCE
	_stencil_draw.queue_redraw()
	_stencil_cut.queue_redraw()


static func _to_px(pts: PackedVector2Array, sz: Vector2) -> PackedVector2Array:
	var out := PackedVector2Array()
	for p in pts:
		out.append(Vector2(p.x * sz.x, p.y * sz.y))
	return out


## The centroid of a landmark ring at `t`.
func _ft_ring_centre(ring: Array, t: float) -> Vector2:
	var acc := Vector2.ZERO
	for i in ring:
		acc += _ft_point(int(i), t)
	return acc / float(ring.size())


## A ring's half-extent, aspect-corrected so a radius means the same thing on
## either axis. Returned in RAW UV (the shaders' contract for these fields).
func _ft_ring_radius(ring: Array, t: float, asp: float) -> float:
	var c := _ft_ring_centre(ring, t)
	var m := 0.0
	for i in ring:
		m = maxf(m, ((_ft_point(int(i), t) - c) * Vector2(asp, 1.0)).length())
	return m


## THE WHOLE FACE MODEL, from measured landmarks instead of a fitted blob. This
## is what replaces _update_face_model's ~300 lines of weighted mass, second
## moments, dark-cluster hunting and hand-tuned plausibility bounds. Every value
## below is a MEASUREMENT of a named part of the face, so there is nothing left to
## tune and nothing left to be wrong about pose: a turned head simply reports
## different landmark positions, which is the whole point.
func _ft_apply_model(t: float) -> void:
	var asp := _source_aspect()
	var eye_l := _ft_ring_centre(FT_EYE_L, t)
	var eye_r := _ft_ring_centre(FT_EYE_R, t)
	var lips := _ft_ring_centre(FT_LIPS, t)
	var top := _ft_point(FT_FACE_TOP, t)
	var chin := _ft_point(FT_FACE_CHIN, t)
	var left := _ft_point(FT_FACE_LEFT, t)
	var right := _ft_point(FT_FACE_RIGHT, t)
	# No EMA and no prediction. Both existed to paper over a detector that
	# jittered and lagged; this one is fitted offline over the whole clip, so the
	# track is already smooth and already knows the future. Writing straight
	# through is what makes the mask sit still.
	_face_eye_l_ema = eye_l
	_face_eye_r_ema = eye_r
	_face_mouth_ema = lips
	_face_nose_ema = _ft_point(FT_NOSE_TIP, t)
	_face_c_ema = (top + chin + left + right) * 0.25
	_face_r_ema = Vector2(absf(right.x - left.x) * 0.5, absf(chin.y - top.y) * 0.5)
	_face_eye_lr_ema = _ft_ring_radius(FT_EYE_L, t, asp)
	_face_eye_rr_ema = _ft_ring_radius(FT_EYE_R, t, asp)
	# The mouth's own half-width and half-height, measured corner-to-corner and
	# lip-to-lip - so it opens and closes with the real mouth instead of holding
	# a fitted ellipse's proportions.
	var mw := 0.0
	var mh := 0.0
	for i in FT_LIPS:
		var d := _ft_point(int(i), t) - lips
		mw = maxf(mw, absf(d.x))
		mh = maxf(mh, absf(d.y))
	_face_mouth_r_ema = Vector2(mw, mh)
	# The face's mean tint/luminance still come from the picture (the coat is a
	# per-pixel colour match, not a fitted oval) - see _update_face_model, which
	# still runs for those when no track is available.
	_face_prev_lum = PackedFloat32Array()


func _ft_bin(tool_name: String) -> String:
	return Deps.venv_bin(FACE_VENV_DIR, tool_name)


func _ft_model_path() -> String:
	return ProjectSettings.globalize_path(FACE_VENV_DIR).path_join("face_landmarker.task")


## Kick the pre-pass off (or discover it already cached). Called per frame while a
## clown layer is live; every branch is a cheap no-op once the track is ready.
## Deliberately lazy: a session that never uses the clown never installs anything.
func _ft_ensure() -> void:
	if _ft_state in ["ready", "failed", "venv", "pip", "model", "tracking"]:
		_ft_poll()
		return
	if session == null or session.video_path.is_empty():
		return
	var src := ProjectSettings.globalize_path(session.video_path)
	if not FileAccess.file_exists(src):
		return
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(FACE_TRACK_DIR))
	_ft_path = ProjectSettings.globalize_path(FACE_TRACK_DIR).path_join(
		str(hash(session.video_path)) + ".bin")
	_ft_log = ProjectSettings.globalize_path(FACE_TRACK_DIR).path_join("last_run.log")
	if FileAccess.file_exists(_ft_path):
		_ft_load()
		return
	if not FileAccess.file_exists(_ft_bin("python")):
		_ft_make_venv()
	elif not FileAccess.file_exists(_ft_model_path()):
		_ft_fetch_model()
	else:
		_ft_start_track()


func _ft_make_venv() -> void:
	var py := _python()
	if py.is_empty():
		_ft_fail("Python 3 is not installed - the clown can't build its face tracker.  "
			+ Deps.hint("python"))
		return
	print("ghost face: bootstrapping venv at ", ProjectSettings.globalize_path(FACE_VENV_DIR))
	_ft_pid = Subprocess.start(py, PackedStringArray(
		["-m", "venv", ProjectSettings.globalize_path(FACE_VENV_DIR)]))
	_ft_state = "venv" if _ft_pid > 0 else "failed"
	_set_status("⏳  Setting up the face tracker (one-time)…" if _ft_pid > 0
		else "⚠  Could not start python3")


func _ft_pip_install() -> void:
	# The requirements file ships beside the script it serves, so the versions
	# that were actually verified travel with it (face_host/requirements.txt).
	var req := ProjectSettings.globalize_path("res://face_host/requirements.txt")
	var args := PackedStringArray(["install", "-q"])
	if FileAccess.file_exists(req):
		args.append_array(PackedStringArray(["-r", req]))
	else:
		args.append("mediapipe")
	print("ghost face: pip ", " ".join(args))
	_ft_pid = _ft_spawn_logged(_ft_bin("pip"), args)
	_ft_state = "pip" if _ft_pid > 0 else "failed"
	_set_status("⏳  Installing the face tracker (one-time, ~100 MB)…" if _ft_pid > 0
		else "⚠  Could not start pip")


## Fetch the landmarker bundle with python itself rather than shelling to curl -
## the venv is already guaranteed at this point and curl is not.
func _ft_fetch_model() -> void:
	var code := "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])"
	_ft_pid = _ft_spawn_logged(_ft_bin("python"),
		PackedStringArray(["-c", code, FACE_MODEL_URL, _ft_model_path()]))
	_ft_state = "model" if _ft_pid > 0 else "failed"
	_set_status("⏳  Fetching the face model (one-time, 4 MB)…" if _ft_pid > 0
		else "⚠  Could not fetch the face model")


func _ft_start_track() -> void:
	var script := ProjectSettings.globalize_path("res://face_host/face_track.py")
	if not FileAccess.file_exists(script):
		_ft_fail("face_host/face_track.py is missing")
		return
	# The ORIGINAL source where we still have it, else the prepared .ogv. The
	# source is better evidence: the transcode is theora q6 and its chroma blocks
	# are exactly the artifact a landmarker has to see past.
	var src := session.source_path
	if src.is_empty() or not FileAccess.file_exists(src):
		src = ProjectSettings.globalize_path(session.video_path)
	print("ghost face: tracking ", src)
	_ft_pid = _ft_spawn_logged(_ft_bin("python"), PackedStringArray([
		script, "--video", src, "--out", _ft_path, "--model", _ft_model_path(),
		"--rate", str(FACE_TRACK_RATE),
		"--progress", ProjectSettings.globalize_path(FACE_TRACK_DIR).path_join("progress")]))
	_ft_state = "tracking" if _ft_pid > 0 else "failed"
	_set_status("⏳  Finding the face through the clip…" if _ft_pid > 0
		else "⚠  Could not start the face tracker")


func _ft_poll() -> void:
	if _ft_pid <= 0 or Subprocess.alive(_ft_pid):
		if _ft_state == "tracking":
			var pf := ProjectSettings.globalize_path(FACE_TRACK_DIR).path_join("progress")
			if FileAccess.file_exists(pf):
				var f := FileAccess.open(pf, FileAccess.READ)
				if f != null:
					_set_status("⏳  Finding the face through the clip…  %d%%"
						% int(f.get_as_text().strip_edges().to_float() * 100.0))
		return
	_ft_pid = -1
	match _ft_state:
		"venv":
			_ft_pip_install()
		"pip":
			if FileAccess.file_exists(_ft_bin("python")):
				_ft_fetch_model()
			else:
				_ft_fail("the face tracker's venv did not install (see the log)")
		"model":
			if FileAccess.file_exists(_ft_model_path()):
				_ft_start_track()
			else:
				_ft_fail("the face model did not download (see the log)")
		"tracking":
			if FileAccess.file_exists(_ft_path):
				_ft_load()
			else:
				_ft_fail("the face pre-pass produced no track (see the log)")


func _ft_fail(why: String) -> void:
	_ft_state = "failed"
	push_warning("ghost face: " + why)
	# SAY SO rather than quietly drawing a bad mask. The fallback fitter still
	# runs, but the author needs to know they are looking at the fallback - the
	# whole reason this exists is that the fallback is not good enough.
	_set_status("⚠  Face tracking unavailable - " + why)


## stdout+stderr to one file, so a failed pip or a mediapipe import error is
## readable afterwards instead of vanishing into a GUI launch's missing console.
func _ft_spawn_logged(exe: String, args: PackedStringArray) -> int:
	var quoted := PackedStringArray()
	for a in [exe] + Array(args):
		quoted.append('"' + String(a).replace('"', '\\"') + '"')
	return Subprocess.start("/bin/bash", PackedStringArray(
		["-c", " ".join(quoted) + ' >"' + _ft_log + '" 2>&1']))


func _ft_load() -> void:
	var f := FileAccess.open(_ft_path, FileAccess.READ)
	if f == null:
		_ft_fail("the cached track could not be opened")
		return
	if f.get_buffer(4).get_string_from_ascii() != "GFT1":
		f.close()
		DirAccess.remove_absolute(_ft_path)   # a stale/foreign file - rebuild next open
		_ft_fail("the cached track is not a face track")
		return
	var _version := f.get_32()
	_ft_rate = f.get_float()
	_ft_count = f.get_32()
	_ft_points = f.get_32()
	var stride := 1 + _ft_points * 8
	if _ft_count <= 0 or _ft_points <= 0 \
			or f.get_length() != FACE_TRACK_HEADER + _ft_count * stride:
		f.close()
		DirAccess.remove_absolute(_ft_path)
		_ft_fail("the cached track is truncated")
		return
	_ft_found.resize(_ft_count)
	_ft_xy.resize(_ft_count * _ft_points * 2)
	var found_n := 0
	for i in _ft_count:
		_ft_found[i] = f.get_8()
		found_n += _ft_found[i]
		var row := f.get_buffer(_ft_points * 8).to_float32_array()
		for k in _ft_points * 2:
			_ft_xy[i * _ft_points * 2 + k] = row[k]
	f.close()
	_ft_state = "ready"
	print("ghost face: track ready - %d samples at %.1f Hz, face in %d (%.0f%%)"
		% [_ft_count, _ft_rate, found_n, 100.0 * float(found_n) / float(maxi(_ft_count, 1))])
	_set_status("✓  Face tracked (%d%% of the clip)"
		% int(100.0 * float(found_n) / float(maxi(_ft_count, 1))))


## Landmark `idx` at clip time `t`, in frame UV. Linearly interpolated between the
## two nearest samples - at 15 Hz a talking head moves little enough between them
## that this is a short interpolation rather than a guess, and it means the drawn
## mask moves continuously instead of stepping at the sample rate.
func _ft_point(idx: int, t: float) -> Vector2:
	if _ft_state != "ready" or _ft_count <= 0:
		return Vector2.ZERO
	var s := clampf(t * _ft_rate, 0.0, float(_ft_count - 1))
	# A CENTRED GAUSSIAN OVER THE SAMPLES, which smooths and interpolates in one
	# pass. This replaced a plain lerp between the two nearest samples, and the
	# reasoning behind that lerp was wrong in a way worth writing down: the track
	# is fitted offline, so it needs no LAG COMPENSATION - but that is not the same
	# as needing no smoothing. The detector still wobbles a pixel or two per
	# sample, and a lerp between noisy samples is a piecewise-linear path with a
	# CORNER at every one of them, so the whole mask twitched at the sample rate.
	#
	# Weighting neighbours on BOTH sides is what the offline track exists for. A
	# live tracker only has the past, so its only smoothing is a lag; a centred
	# kernel has ZERO phase - it removes the jitter and moves the mask not one
	# frame later than the face. And because the weight is a smooth function of
	# the continuous position, the result is smooth in time by construction:
	# no corners at sample boundaries, nothing to step.
	var sigma := _ft_sigma
	var half := maxi(1, int(ceil(sigma * 2.5)))
	var lo := maxi(0, int(s) - half)
	var hi := mini(_ft_count - 1, int(s) + half + 1)
	var acc := Vector2.ZERO
	var wsum := 0.0
	var base := idx * 2
	for i in range(lo, hi + 1):
		# LOST SAMPLES ARE SKIPPED, not averaged in - their coordinates are a held
		# copy of a neighbour, so including them would drag the feature toward
		# wherever the hold happened to be.
		if _ft_found[i] == 0:
			continue
		var d := (float(i) - s) / sigma
		var w := exp(-0.5 * d * d)
		var b := i * _ft_points * 2 + base
		acc += Vector2(_ft_xy[b], _ft_xy[b + 1]) * w
		wsum += w
	if wsum <= 1e-6:
		# Every sample in the window was lost - fall back to the nearest one that
		# is not, rather than returning the origin and snapping the mask to a corner.
		var n := clampi(int(round(s)), 0, _ft_count - 1)
		var bn := n * _ft_points * 2 + base
		return Vector2(_ft_xy[bn], _ft_xy[bn + 1])
	return acc / wsum


## Is there a real detection at `t`? The mean of a few landmarks is meaningless
## when the answer is "no face here", and the caller has to be able to hold.
func _ft_has(t: float) -> bool:
	if _ft_state != "ready" or _ft_count <= 0:
		return false
	return _ft_found[clampi(int(t * _ft_rate), 0, _ft_count - 1)] != 0


## The clown model's current display state - positions velocity-extrapolated
## by the fraction through the capture tick (prediction cancels the capture +
## EMA lag), sizes taken straight from their EMAs (predicting a size just
## amplifies shape twitch). One source of truth for _push_anchor AND the
## paint sim's deposit targets.
func _clown_model_now() -> Dictionary:
	# LEAD scales the prediction. It is a jitter AMPLIFIER as much as a lag
	# canceller - it continues whatever the last tick did, so when the detection
	# is clean it cancels the capture+EMA delay, and when the detection is noisy
	# it doubles the noise and the mask visibly shakes at the capture cadence.
	# Which of those dominates is a property of the footage, so it is the
	# author's call: 0 turns the prediction off entirely (calmest, laggiest), 1
	# is the behaviour this always had, 2 over-predicts for fast motion.
	var ff := clampf(fposmod(_player.stream_position, _FACE_INTERVAL) / _FACE_INTERVAL, 0.0, 1.0)
	# WITH A TRACK, DO NOT PREDICT. The prediction exists to cancel a live
	# detector's delay, and it does that by ramping a fraction from 0 to 1 across
	# each capture tick and then snapping back to 0 - a sawtooth. On a laggy
	# detector that sawtooth is smaller than the lag it cancels and worth paying.
	# On the offline track there is NO lag to cancel (the smoothing is centred),
	# so all that is left is the sawtooth itself: the whole mask lurching forward
	# and resetting several times a second, which is exactly the twitch reported.
	if _ft_state == "ready":
		ff = 0.0
	return {
		"eye_l": _face_eye_l_ema + (_face_eye_l_ema - _face_eye_l_prev) * ff,
		"eye_r": _face_eye_r_ema + (_face_eye_r_ema - _face_eye_r_prev) * ff,
		"mouth": _face_mouth_ema + (_face_mouth_ema - _face_mouth_prev) * ff,
		"nose": _face_nose_ema + (_face_nose_ema - _face_nose_prev) * ff,
		"face_c": _face_c_ema + (_face_c_ema - _face_c_prev) * ff,
		"eye_lr": _face_eye_lr_ema, "eye_rr": _face_eye_rr_ema,
		"mouth_r": _face_mouth_r_ema,
	}


# --- umbra: the cast-shadow detector -----------------------------------------

## Separable box blur over the grid, via a RUNNING SUM - O(cells), not
## O(cells x radius). The radii this detector wants are wide (a shadow is
## big), and the naive form is what would make a ~7Hz GDScript pass expensive.
## Edges CLAMP rather than wrap: wrapping folds the door on the left into the
## wall on the right, which is precisely the contamination the whole detector
## is built to avoid.
func _umb_blur(src: PackedFloat32Array, dst: PackedFloat32Array, rx: int, ry: int) -> void:
	var tmp := _umb_tmp2
	if tmp.size() != src.size():
		tmp.resize(src.size())
		_umb_tmp2 = tmp
	var wx := float(rx * 2 + 1)
	for y in _UMB_H:
		var row := y * _UMB_W
		var acc := 0.0
		for k in range(-rx, rx + 1):
			acc += src[row + clampi(k, 0, _UMB_W - 1)]
		for x in _UMB_W:
			tmp[row + x] = acc / wx
			acc -= src[row + clampi(x - rx, 0, _UMB_W - 1)]
			acc += src[row + clampi(x + rx + 1, 0, _UMB_W - 1)]
	var wy := float(ry * 2 + 1)
	for x in _UMB_W:
		var acc2 := 0.0
		for k in range(-ry, ry + 1):
			acc2 += tmp[clampi(k, 0, _UMB_H - 1) * _UMB_W + x]
		for y in _UMB_H:
			dst[y * _UMB_W + x] = acc2 / wy
			acc2 -= tmp[clampi(y - ry, 0, _UMB_H - 1) * _UMB_W + x]
			acc2 += tmp[clampi(y + ry + 1, 0, _UMB_H - 1) * _UMB_W + x]


## Cell membership for ONE surface hypothesis: how much each cell looks like
## `dir`-coloured material, and how much of that material is in shadow.
##
## MATCH THE SURFACE FIRST. A cast shadow is the same wall under less light,
## so it keeps the wall's chroma DIRECTION and loses luminance. On the
## reference clip the shadow aligns with the lit wall at dot=+0.99 while skin
## (-0.96), hair (-0.96), a black shirt (-0.90), the mic (-0.80) and a cream
## door (-0.75) all sit far away - and her hair is at the SAME luminance as
## the shadow, so nothing luminance-based could have separated them. Leading
## with darkness (and treating chroma as a bonus) was the first design, and it
## simultaneously kept her whole face and threw away the shadow's own core.
func _umb_analyse(dir: Vector3, mag: float, lit: float) -> void:
	var n := _UMB_W * _UMB_H
	# Is this surface coloured enough for a direction test to mean anything?
	var chromatic := smoothstep(0.015, 0.040, mag)
	for i in n:
		var cmag := _umb_cmag[i]
		var align := 0.0
		if cmag > 1e-5:
			align = (_umb_cr[i] * dir.x + _umb_cg[i] * dir.y + _umb_cb[i] * dir.z) / cmag
		# Same material under less light also means proportionally LESS chroma,
		# so a cell far off the expected magnitude is some other material that
		# merely happens to point the same way.
		var expect := mag * clampf(_umb_lum[i] / maxf(lit, 1e-3), 0.05, 1.5)
		var rel := cmag / (expect + 1e-6)
		var magfit := smoothstep(0.20, 0.70, rel) * (1.0 - smoothstep(1.9, 3.8, rel))
		var chroma_match := smoothstep(0.35, 0.85, align) * magfit
		# A COLOURLESS wall cannot be matched by direction at all (grey rooms,
		# monochrome grades - the footage class that broke the clown's first
		# cut). There the only honest statement is "a shadow on grey is also
		# grey", so fall back to matching colourlessness itself.
		var neutral := 1.0 - smoothstep(0.020, 0.055, cmag)
		_umb_match[i] = neutral + chromatic * (chroma_match - neutral)
	# The local lit level, estimated ONLY from cells that matched this surface.
	# Estimating it from the neighbourhood at large is contaminated by whatever
	# object is sitting there (her own bright face raising the bar right where
	# the shadow is), which is what made the first cut classify her cheek as
	# shadowed wall.
	if _umb_wmass.size() != n:
		_umb_wmass.resize(n)
		_umb_wlum.resize(n)
	for i in n:
		_umb_tmp[i] = 1.0 if _umb_match[i] > 0.5 else 0.0
	_umb_blur(_umb_tmp, _umb_wmass, 14, 9)
	for i in n:
		_umb_tmp[i] = _umb_lum[i] * (1.0 if _umb_match[i] > 0.5 else 0.0)
	_umb_blur(_umb_tmp, _umb_wlum, 14, 9)
	var floor_lit := lit * 0.55
	for i in n:
		var llit := lit
		if _umb_wmass[i] > 0.02:
			llit = _umb_wlum[i] / _umb_wmass[i]
		# A neighbourhood that is MOSTLY shadow would drag its own reference
		# down and declare itself lit; the floor is what stops a large shadow
		# from erasing its own middle.
		llit = maxf(llit, floor_lit)
		var ratio := _umb_lum[i] / maxf(llit, 1e-3)
		# GENEROUS on purpose. A real cast shadow is mostly PENUMBRA, and a
		# tight window (0.60..0.88) kept only the dark core - about 7% of the
		# frame where the visible shadow covers nearer 15%, so the effect had
		# nothing like the whole shadow to animate. The chroma match is what
		# keeps this honest; darkness only has to say "dimmer than this wall
		# is elsewhere", not "very dark".
		var dark := 1.0 - smoothstep(0.85, 1.02, ratio)
		_umb_shadow[i] = _umb_match[i] * dark


## Flood fill over the grid from `seeds`, through cells where `pass_fn` holds.
## Returns how many cells were claimed. Iterative with a preallocated queue -
## recursion depth on a 96x54 grid is not something to hand to GDScript.
func _umb_flood(out: PackedByteArray, seeds: PackedInt32Array, field: PackedFloat32Array,
		thr: float, blocked: PackedByteArray, limit: int) -> int:
	var n := _UMB_W * _UMB_H
	for i in n:
		out[i] = 0
	if _umb_queue.size() < n:
		_umb_queue.resize(n)
	var head := 0
	var tail := 0
	for s in seeds:
		if out[s] == 0 and field[s] > thr and (blocked.is_empty() or blocked[s] == 0):
			out[s] = 1
			_umb_queue[tail] = s
			tail += 1
	var claimed := 0
	while head < tail and claimed < limit:
		var idx := _umb_queue[head]
		head += 1
		claimed += 1
		var x := idx % _UMB_W
		var y := idx / _UMB_W
		for d in 4:
			var nx := x + (1 if d == 0 else (-1 if d == 1 else 0))
			var ny := y + (1 if d == 2 else (-1 if d == 3 else 0))
			if nx < 0 or nx >= _UMB_W or ny < 0 or ny >= _UMB_H:
				continue
			var ni := ny * _UMB_W + nx
			if out[ni] != 0 or field[ni] <= thr:
				continue
			if not blocked.is_empty() and blocked[ni] != 0:
				continue
			out[ni] = 1
			if tail < n:
				_umb_queue[tail] = ni
				tail += 1
	return claimed


## Run both floods for the CURRENT contents of _umb_match/_umb_shadow, and
## score how coherent the resulting scene is. Returns {score, subj_n, shad_n}.
##
## LINKAGE IS STRUCTURAL. The subject flood grows from the most not-this-wall
## cell near frame centre; the shadow flood may only start from cells TOUCHING
## the subject and may never enter it. So every cell the shadow flood reaches
## is contiguous with her - "the shadow is linked to the human" is not a
## similarity score here, it is the shape of the search.
func _umb_solve(aspect: float) -> Dictionary:
	var n := _UMB_W * _UMB_H
	# foreign = not this surface. Reuse _umb_tmp as the flood's field.
	var best := -1.0
	var seed := 0
	for y in _UMB_H:
		for x in _UMB_W:
			var i := y * _UMB_W + x
			var foreign := 1.0 - _umb_match[i]
			_umb_tmp[i] = foreign
			var px := (float(x) + 0.5) / float(_UMB_W)
			var py := (float(y) + 0.5) / float(_UMB_H)
			# the ASMR framing prior the face model leans on too
			var dx := (px - 0.5) * aspect
			var dy := py - 0.55
			var prior: float = exp(-(dx * dx / 0.30 + dy * dy / 0.45))
			var sc := foreign * prior
			if sc > best:
				best = sc
				seed = i
	var seeds := PackedInt32Array([seed])
	var subj_n := _umb_flood(_umb_subj, seeds, _umb_tmp, 0.5, PackedByteArray(), n)
	# Seed the shadow from the subject's own boundary.
	var edge := PackedInt32Array()
	for y in _UMB_H:
		for x in _UMB_W:
			var i := y * _UMB_W + x
			if _umb_subj[i] != 0:
				continue
			var touch := false
			for d in 4:
				var nx := x + (1 if d == 0 else (-1 if d == 1 else 0))
				var ny := y + (1 if d == 2 else (-1 if d == 3 else 0))
				if nx >= 0 and nx < _UMB_W and ny >= 0 and ny < _UMB_H \
						and _umb_subj[ny * _UMB_W + nx] != 0:
					touch = true
					break
			if touch:
				edge.append(i)
	var shad_n := _umb_flood(_umb_shad, edge, _umb_shadow, 0.30, _umb_subj, int(n * 0.55))
	# Scene coherence. Under the RIGHT surface the subject flood covers the
	# middle of the frame; under the wrong one (her warm skin voting the cream
	# door in as "the wall") the "subject" comes out as the far wall instead,
	# which covers almost none of the prior. That mismatch IS the test.
	# NOT penalised for claiming a large area: under the right hypothesis the
	# subject legitimately absorbs every other non-wall surface (the door, the
	# mic), which is harmless - it only ever means "no ghost there". Penalising
	# it was what handed three of eight test frames to the door.
	var cov := 0.0
	var pw := 0.0
	for y in _UMB_H:
		for x in _UMB_W:
			var i := y * _UMB_W + x
			var px2 := (float(x) + 0.5) / float(_UMB_W)
			var py2 := (float(y) + 0.5) / float(_UMB_H)
			var dx2 := (px2 - 0.5) * aspect
			var dy2 := py2 - 0.55
			var pr: float = exp(-(dx2 * dx2 / 0.30 + dy2 * dy2 / 0.45))
			pw += pr
			if _umb_subj[i] != 0:
				cov += pr
	cov = cov / maxf(pw, 1e-5)
	var frac := float(subj_n) / float(n)
	var sane := cov * (1.0 - smoothstep(0.93, 0.99, frac))
	var score := sane * (0.35 + 0.65 * smoothstep(10.0, 220.0, float(shad_n)))
	return {"score": score, "subj_n": subj_n, "shad_n": shad_n, "cov": cov}


## THE UMBRA MODEL, fitted per capture tick. Finds the surface the subject's
## shadow falls on, the shadow region linked to her, and the direction the
## light throws it - then packs the verdict into a small texture the field
## simulation deposits into. See MaskSession's "umbra" doc for the whole idea.
func _update_umbra_model(src: Image) -> void:
	var n := _UMB_W * _UMB_H
	if _umb_lum.size() != n:
		_umb_lum.resize(n); _umb_cr.resize(n); _umb_cg.resize(n); _umb_cb.resize(n)
		_umb_cmag.resize(n); _umb_match.resize(n); _umb_shadow.resize(n)
		_umb_tmp.resize(n); _umb_subj.resize(n); _umb_shad.resize(n)
	var aspect := 1.7778
	if src.get_height() > 0:
		aspect = float(src.get_width()) / float(src.get_height())
	var img: Image = src.duplicate()
	img.resize(_UMB_W, _UMB_H, Image.INTERPOLATE_BILINEAR)
	if img.get_format() != Image.FORMAT_RGBA8:
		img.convert(Image.FORMAT_RGBA8)
	var data := img.get_data()
	for i in n:
		var b := i * 4
		var r := float(data[b]) / 255.0
		var g := float(data[b + 1]) / 255.0
		var bl := float(data[b + 2]) / 255.0
		var l := 0.299 * r + 0.587 * g + 0.114 * bl
		_umb_lum[i] = l
		var cr := r - l
		var cg := g - l
		var cb := bl - l
		_umb_cr[i] = cr; _umb_cg[i] = cg; _umb_cb[i] = cb
		_umb_cmag[i] = sqrt(cr * cr + cg * cg + cb * cb)
	# --- which surface is the shadow on?
	# The re-pick is the only expensive part (it runs the whole solve once per
	# candidate), and the answer is a property of the ROOM, not of the frame -
	# so it runs on the first tick and occasionally after, and the rest of the
	# time the stored reference is simply reused.
	_umb_repick_in -= 1
	if not _umb_ref_valid or _umb_repick_in <= 0:
		_umb_repick_in = 40   # ~6s at the 0.15s capture cadence
		_umb_pick_reference(aspect)
	if not _umb_ref_valid:
		_umb_have = false
		return
	_umb_analyse(_umb_ref_dir, _umb_ref_mag, _umb_ref_lit)
	var res := _umb_solve(aspect)
	if int(res.shad_n) < 8 or int(res.subj_n) < 8:
		_umb_have = false
		return
	# HARD CONFIDENCE GATE - draw NOTHING rather than something wrong.
	# `cov` is how much of the centred prior the subject flood claims. When the
	# scene is read correctly she is in the middle of frame and this sits near
	# 0.8; when it is read inside out (the wall taken for the subject) it falls
	# to ~0.5. Below the floor the model is not describing this scene at all,
	# and the guard built from it would be protecting the wrong thing - which
	# is exactly how the effect ended up painted across her face.
	if float(res.cov) < 0.60:
		_umb_have = false
		if OS.has_environment("GHOST_UMBRA_DEBUG"):
			print("UMBDBG t=%.2f REJECTED cov=%.2f (scene read incoherent - drawing nothing)"
				% [_player.stream_position, float(res.cov)])
		return
	# --- centroids and the cast direction
	var sacc := Vector2.ZERO
	var hacc := Vector2.ZERO
	var sw := 0.0
	var hw := 0.0
	for y in _UMB_H:
		for x in _UMB_W:
			var i := y * _UMB_W + x
			var p := Vector2((float(x) + 0.5) / float(_UMB_W), (float(y) + 0.5) / float(_UMB_H))
			if _umb_subj[i] != 0:
				sacc += p; sw += 1.0
			if _umb_shad[i] != 0:
				hacc += p; hw += 1.0
	var subj_c := sacc / maxf(sw, 1.0)
	var shad_c := hacc / maxf(hw, 1.0)
	_umb_subj_c = subj_c
	_umb_shad_c = shad_c
	# THE PIVOT the silhouette magnifies about. It must sit where the shadow is
	# SOLID, because magnification is also a WINDOW: at scale S the screen can
	# only show a 1/S-sized neighbourhood of the pivot. Anchoring at the
	# silhouette's base (the obvious choice for "rears upward") put that window
	# on the strip where the shadow borders her and dissolves into void, so
	# scaling UP made the mass shrink - measured, coverage fell 19% -> 4% going
	# from scale 1.0 to 3.5.
	# The centroid keeps the window on the body; biasing it a little toward the
	# base still throws the head off the top first, which is the look wanted.
	var ylow := shad_c.y
	for y2 in _UMB_H:
		for x2 in _UMB_W:
			if _umb_shad[y2 * _UMB_W + x2] != 0:
				ylow = maxf(ylow, (float(y2) + 0.5) / float(_UMB_H))
	var pivot := Vector2(shad_c.x, lerpf(shad_c.y, ylow, 0.35))
	_umb_pivot = _umb_pivot.lerp(pivot, 0.12) if _umb_have else pivot
	var d := (shad_c - subj_c) * Vector2(aspect, 1.0)
	if d.length() > 1e-4:
		# Deliberately glacial. The light is furniture: it does not move, and a
		# per-frame direction is a per-frame twitch in everything downstream.
		_umb_dir_ema = (_umb_dir_ema.lerp(d.normalized(), 0.06)).normalized()
	# --- pack the verdict for the field sim
	# SOFTEN BOTH MASKS BEFORE UPLOAD. The floods are binary and the grid is
	# coarse, so handing them over as-is drew the detector's own cell staircase
	# straight into the picture (plainly visible in the first render). Blurred,
	# they become ramps the field can feather across.
	# Note the asymmetry: blurring the SUBJECT mask makes its exclusion start
	# EARLIER (the guard ramps up before the hard edge), which is the safe
	# direction - the one property this effect may not trade away is staying
	# off her.
	for i in n:
		_umb_tmp[i] = 1.0 if _umb_shad[i] != 0 else 0.0
	_umb_blur(_umb_tmp, _umb_wlum, 2, 2)
	for i in n:
		_umb_tmp[i] = 1.0 if _umb_subj[i] != 0 else 0.0
	_umb_blur(_umb_tmp, _umb_wmass, 2, 2)
	# REACH closes the gap between the mass and the woman casting it. Right at
	# her outline the pixels are a BLEND of her and the wall, so they read as
	# neither cleanly - they fall to the subject flood and the shadow stops
	# short, leaving the visible band of ungraded wall between the two.
	# Two moves, one knob: shrink the subject mask's safety dilation, and grow
	# the shadow mask. At 0 this is exactly the old conservative behaviour.
	var reach := clampf(_umb_reach, 0.0, 1.0)
	var sub_gain := lerpf(1.35, 0.55, reach)
	var shad_gain := lerpf(1.0, 1.9, reach)
	# Straight into a byte buffer: 5184 set_pixel() calls at ~7Hz is real cost
	# for no reason when create_from_data takes the whole thing at once.
	if _umb_bytes.size() != n * 4:
		_umb_bytes.resize(n * 4)
	for i in n:
		var b4 := i * 4
		_umb_bytes[b4] = int(clampf(_umb_wlum[i] * shad_gain, 0.0, 1.0) * 255.0)
		_umb_bytes[b4 + 1] = int(clampf(_umb_shadow[i], 0.0, 1.0) * 255.0)
		_umb_bytes[b4 + 2] = int(clampf(_umb_wmass[i] * sub_gain, 0.0, 1.0) * 255.0)
		_umb_bytes[b4 + 3] = 255
	_umb_region_img = Image.create_from_data(_UMB_W, _UMB_H, false, Image.FORMAT_RGBA8, _umb_bytes)
	if _umb_region_tex == null:
		_umb_region_tex = ImageTexture.create_from_image(_umb_region_img)
	else:
		_umb_region_tex.update(_umb_region_img)
	_umb_solve_eyes(subj_c, shad_c)
	_umb_have = true
	if OS.has_environment("GHOST_UMBRA_DEBUG"):
		print("UMBDBG t=%.2f ref=(%.2f,%.2f,%.2f) mag=%.3f lit=%.2f subj=%d shad=%d " % [
			_player.stream_position, _umb_ref_dir.x, _umb_ref_dir.y, _umb_ref_dir.z,
			_umb_ref_mag, _umb_ref_lit, int(res.subj_n), int(res.shad_n)]
			+ "dir=(%.2f,%.2f) subjc=(%.2f,%.2f) shadc=(%.2f,%.2f) score=%.2f" % [
			_umb_dir_ema.x, _umb_dir_ema.y, subj_c.x, subj_c.y, shad_c.x, shad_c.y,
			float(res.score)]
			+ " eyeL=(%.2f,%.2f) eyeR=(%.2f,%.2f) rad=%.3f ok=%s herEye=(%.2f,%.2f)" % [
			_umb_eye_l.x, _umb_eye_l.y, _umb_eye_r.x, _umb_eye_r.y, _umb_eye_rad,
			str(_umb_eyes_ok), _face_eye_l_ema.x, _face_eye_l_ema.y])


## --- THE EYE TRACK: a real look-ahead, not a prediction ---------------------
##
## Velocity extrapolation cannot anticipate a word or a flick of the head; it
## can only continue whatever just happened, which on real footage is mostly
## detection jitter. To move BEFORE she does, the future has to be known, so
## the whole clip is analysed up front: ffmpeg decodes it once at grid
## resolution and one sample per _UMBRA_INTERVAL, a worker thread fits her eyes
## in every frame, and playback then simply reads the track at `t + Lead`.
##
## Deterministic by construction (a pure function of the clip), so the live
## preview and the export relaunch lead by exactly the same amount, and the
## per-frame cost at playback is one array lookup.
func _umb_ensure_track() -> void:
	if _umb_track_state != "none" or session == null or session.video_path.is_empty():
		return
	var src := ProjectSettings.globalize_path(session.video_path)
	if not FileAccess.file_exists(src):
		return
	# user:// deliberately: a raw dump inside res://masks would be scanned by
	# the editor's importer, which is the same class of trouble the truncated
	# audio.wav caused (see _promote_part).
	var dir := OS.get_user_data_dir() + "/umbra_tracks"
	DirAccess.make_dir_recursive_absolute(dir)
	_umb_track_raw = dir + "/" + str(hash(session.video_path)) + ".raw"
	var args := PackedStringArray([
		"-y", "-loglevel", "error", "-i", src,
		"-vf", "scale=%d:%d,fps=%f" % [_UMB_W, _UMB_H, 1.0 / _UMBRA_INTERVAL],
		"-f", "rawvideo", "-pix_fmt", "rgb24", _umb_track_raw + ".part"])
	_umb_track_pid = Subprocess.start("ffmpeg", args, "umbra track")
	if _umb_track_pid <= 0:
		_umb_track_state = "failed"
		return
	_umb_track_state = "decoding"
	_set_status("⏳  Reading ahead for the umbra…")


## Poll the decode, then hand the raw dump to a worker thread. Called per frame
## while an umbra layer is live; both halves are cheap no-ops once done.
func _umb_poll_track() -> void:
	if _umb_track_state == "decoding":
		if Subprocess.alive(_umb_track_pid):
			return
		if not FileAccess.file_exists(_umb_track_raw + ".part"):
			_umb_track_state = "failed"
			return
		DirAccess.rename_absolute(_umb_track_raw + ".part", _umb_track_raw)
		_umb_fit_asp = _source_aspect()   # snapshot before the worker reads it
		_umb_track_thread = Thread.new()
		_umb_track_thread.start(_umb_fit_track_threaded.bind(_umb_track_raw))
		_umb_track_state = "fitting"
		_set_status("⏳  Reading ahead for the umbra…")
		return
	if _umb_track_state == "fitting":
		if _umb_track_thread == null or _umb_track_thread.is_alive():
			return
		var got: Variant = _umb_track_thread.wait_to_finish()
		_umb_track_thread = null
		_umb_track = got if got is PackedVector3Array else PackedVector3Array()
		# HER RESTING PLACE, from the whole clip at once rather than an EMA
		# chasing it. Deviations are measured from this, so the ghost's gaze is
		# steady when she is and swings only when she actually moves.
		var acc := Vector2.ZERO
		var n := 0
		for s in _umb_track:
			if s.z > 0.0:
				acc += Vector2(s.x, s.y)
				n += 1
		_umb_track_rest = acc / maxf(float(n), 1.0) if n > 0 else Vector2(0.5, 0.35)
		_umb_track_state = "ready" if n > 4 else "failed"
		if OS.has_environment("GHOST_UMBRA_DEBUG"):
			print("UMBDBG track %s: %d samples, %d with a face, rest=(%.3f,%.3f)"
				% [_umb_track_state, _umb_track.size(), n, _umb_track_rest.x, _umb_track_rest.y])


## Worker: fit her eyes in every sampled frame of the raw dump.
## Returns PackedVector3Array of (eye_mid.x, eye_mid.y, eye_separation); z <= 0
## marks a frame where no face was found, so lookups can skip it.
func _umb_fit_track_threaded(path: String) -> PackedVector3Array:
	var out := PackedVector3Array()
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return out
	var stride := _UMB_W * _UMB_H * 3
	var total := int(f.get_length() / stride)
	for k in total:
		out.append(_umb_fit_eyes(f.get_buffer(stride)))
	f.close()
	return out


## One frame's eye fit, on a raw rgb24 grid buffer. Deliberately the same
## school as the clown's face model: an EYE is a DARK spot in a SKIN
## neighbourhood, so candidates are weighted by the BLURRED face mass - dark
## weighted by its own mass finds nothing (an eye carries no skin colour), and
## dark weighted by a flat floor lets the hair either side of the face win,
## which blows the pair apart.
func _umb_fit_eyes(buf: PackedByteArray) -> Vector3:
	var n := _UMB_W * _UMB_H
	if buf.size() < n * 3:
		return Vector3(0.5, 0.35, -1.0)
	var lum := PackedFloat32Array(); lum.resize(n)
	var mass := PackedFloat32Array(); mass.resize(n)
	var mean_l := 0.0
	for i in n:
		var b := i * 3
		var r := float(buf[b]) / 255.0
		var g := float(buf[b + 1]) / 255.0
		var bl := float(buf[b + 2]) / 255.0
		lum[i] = 0.299 * r + 0.587 * g + 0.114 * bl
		mean_l += lum[i]
	mean_l /= float(n)
	var acc := Vector2.ZERO
	var wsum := 0.0
	for y in _UMB_H:
		for x in _UMB_W:
			var i := y * _UMB_W + x
			var b := i * 3
			var r := float(buf[b]) / 255.0
			var g := float(buf[b + 1]) / 255.0
			var bl := float(buf[b + 2]) / 255.0
			var l := lum[i]
			var cr := r - l
			var cb := bl - l
			var skin := 0.0
			if cr > 0.01 and l > 0.15 and cr > cb:
				skin = clampf(cr * 6.0, 0.0, 1.0) * clampf((cr - cb) * 4.0, 0.0, 1.0)
			# the brightness cue that carries near-monochrome grades
			var bright: float = smoothstep(mean_l + 0.06, mean_l + 0.30, l)
			var px := (float(x) + 0.5) / float(_UMB_W)
			var py := (float(y) + 0.5) / float(_UMB_H)
			var prior: float = exp(-(pow((px - 0.5) * _umb_fit_asp, 2.0) + pow(py - 0.45, 2.0)) / 0.18)
			var wt := maxf(skin * 0.9, bright * 0.85) * prior
			mass[i] = wt
			acc += Vector2(px, py) * wt
			wsum += wt
	if wsum <= 0.5:
		return Vector3(0.5, 0.35, -1.0)
	var c := acc / wsum
	# The face's own half-width, aspect-corrected - the yardstick every
	# constraint below is expressed in. Without it the eye search has no sense
	# of scale and happily returns the hair on either side of the head as a
	# "pair", which is what made the sockets enormous and far too far apart.
	# Measured over the EYE BAND only. Taken across the whole mass it is her
	# shoulders and arms as much as her head, which inflated it to the point
	# that the separation constraint below permitted almost anything.
	var vxx := 0.0
	var vw := 0.0
	for y in _UMB_H:
		var pyb := (float(y) + 0.5) / float(_UMB_H)
		if pyb > c.y + 0.02 or pyb < c.y - 0.22:
			continue
		for x in _UMB_W:
			var i := y * _UMB_W + x
			var dxp := ((float(x) + 0.5) / float(_UMB_W) - c.x) * _umb_fit_asp
			vxx += dxp * dxp * mass[i]
			vw += mass[i]
	var half_w: float = clampf(sqrt(maxf(1e-5, vxx / maxf(vw, 1e-5))) * 1.35, 0.04, 0.26)
	# separable box blur of the mass, radius 3 - the "skin neighbourhood"
	var mb := PackedFloat32Array(); mb.resize(n)
	var tmp := PackedFloat32Array(); tmp.resize(n)
	for y in _UMB_H:
		for x in _UMB_W:
			var s := 0.0
			for k in range(-3, 4):
				s += mass[y * _UMB_W + clampi(x + k, 0, _UMB_W - 1)]
			tmp[y * _UMB_W + x] = s / 7.0
	for y in _UMB_H:
		for x in _UMB_W:
			var s2 := 0.0
			for k in range(-3, 4):
				s2 += tmp[clampi(y + k, 0, _UMB_H - 1) * _UMB_W + x]
			mb[y * _UMB_W + x] = s2 / 7.0
	# two darkest clusters in the upper face band, split left/right of centre
	var la := Vector2.ZERO
	var lw := 0.0
	var ra := Vector2.ZERO
	var rw := 0.0
	for y in _UMB_H:
		for x in _UMB_W:
			var i := y * _UMB_W + x
			var px2 := (float(x) + 0.5) / float(_UMB_W)
			var py2 := (float(y) + 0.5) / float(_UMB_H)
			if py2 > c.y + 0.02 or py2 < c.y - 0.22:
				continue
			# Inside the FACE, not merely above its centroid: an eye sits well
			# within the head's half-width, and letting candidates range to the
			# frame edge is how hair wins both clusters.
			if absf((px2 - c.x) * _umb_fit_asp) > half_w * 0.85:
				continue
			var dark := maxf(0.0, mean_l - lum[i])
			var wv := dark * mb[i]
			wv *= wv
			if wv <= 0.0:
				continue
			if px2 < c.x:
				la += Vector2(px2, py2) * wv; lw += wv
			else:
				ra += Vector2(px2, py2) * wv; rw += wv
	if lw <= 1e-6 or rw <= 1e-6:
		return Vector3(c.x, c.y - 0.08, -1.0)
	var el := la / lw
	var er := ra / rw
	# SEPARATION COMES FROM THE HEAD'S WIDTH, not from the fitted pair.
	# The darkest-cluster fit locates the MIDPOINT reliably - two dark regions
	# either side of the face centre average to about the right place even when
	# one of them is really an eyebrow or a strand of hair - but their spacing
	# is exactly the quantity that degrades, and it was coming back at 0.37
	# where a face that size supports about 0.12. Human eye separation is
	# close to half the head's width, which is a far steadier thing to measure.
	var sep_fit := absf(er.x - el.x) * _umb_fit_asp
	if sep_fit < half_w * 0.30 or sep_fit > half_w * 2.2:
		return Vector3(c.x, c.y - 0.08, -1.0)   # the fit disagrees with the anatomy
	# 0.45, calibrated against her actual face on the reference clip rather than
	# from the textbook "eyes are half a head apart": the mass-derived half_w
	# still runs wide because the eye band catches hair and neck, and at 0.90
	# the ghost's eyes came out twice her spacing.
	return Vector3((el.x + er.x) * 0.5, (el.y + er.y) * 0.5, half_w * 0.45)


## The track sampled at clip time `t`, already carrying the Lead. Returns
## z <= 0 when no usable sample exists there.
func _umb_track_at(t: float) -> Vector3:
	if _umb_track.is_empty():
		return Vector3(0.5, 0.35, -1.0)
	var fpos := t / _UMBRA_INTERVAL
	var i := clampi(int(floor(fpos)), 0, _umb_track.size() - 1)
	var j := clampi(i + 1, 0, _umb_track.size() - 1)
	var a := _umb_track[i]
	var b := _umb_track[j]
	if a.z <= 0.0:
		return b
	if b.z <= 0.0:
		return a
	return a.lerp(b, clampf(fpos - float(i), 0.0, 1.0))


## Where the GHOST's eyes sit, in screen space.
##
## Her eyes are known in screen space (the face model). The shadow is her,
## displaced - so carrying her eyes across the CAST OFFSET (shadow centroid
## minus subject centroid) lands them at the corresponding point of the
## silhouette. That result then rides the same magnify-about-pivot-and-pan
## transform the rest of the body does, so the eyes stay put in the ghost no
## matter how it is scaled or moved.
##
## THE LEAD is what sells the puppeteering: the ghost has to arrive before she
## does. This extrapolates along a smoothed velocity rather than the raw
## frame-to-frame delta, because the raw delta is mostly detection jitter and
## multiplying jitter by a lead time is just amplified twitch.
func _umb_solve_eyes(subj_c: Vector2, shad_c: Vector2) -> void:
	if _umb_track_state != "ready" or _player == null:
		_umb_eyes_ok = false
		return
	# THE LEAD IS A LOOK-AHEAD. The track holds her eyes for the WHOLE clip, so
	# this reads the frame she has not reached yet and the ghost genuinely
	# moves first - it anticipates a word or a turn of the head, which no
	# amount of extrapolating the last two samples ever could.
	var t_ahead := _player.stream_position + clampf(_umb_lead, 0.0, 2.0)
	var s := _umb_track_at(t_ahead)
	if s.z <= 0.0:
		_umb_eyes_ok = false
		return
	var her := Vector2(s.x, s.y)
	var unit := maxf(0.02, s.z)
	var dev := her - _umb_track_rest

	# THE EYES ARE ANCHORED TO THE VISIBLE MASS, NOT TO ANATOMY.
	# Carrying her eyes across the cast offset and then through the silhouette
	# magnification is geometrically correct and useless: at Scale 2.2 it put
	# them at y = -0.45, off the top of the frame, because a ghost twice her
	# size genuinely has its head above the picture. Eyes and looming would be
	# mutually exclusive. So the sockets sit in the upper body of the mass
	# WHEREVER that lands on screen, and her movement drives their DEVIATION
	# from that rest position - amplified, so a small turn of her head throws
	# the ghost's gaze further than her own.
	var head := (shad_c - _umb_pivot) * _umb_scale + _umb_pivot + _umb_pan
	# Modest, and NOT multiplied by the scale: the transform has already moved
	# this point: adding a scale-multiplied lift on top drove the sockets into
	# the top edge clamp at y=0.05 and pinned them there.
	# Barely any lift at all. The transformed centroid IS the middle of the
	# visible mass - which is where there is something to cut sockets OUT of.
	# Lifting off it put them in the thin upper fringe, where hollowing 92% of
	# very little left the sockets no brighter than the body around them.
	head.y -= 0.02
	head += dev * 1.9
	# A small nudge outward, clear of her outline - at the raw shadow centroid
	# the inner socket lands on the guarded band and has no mass to be cut from.
	# Deliberately fixed rather than proportional to the fitted separation:
	# scaling it by `unit` compounded a bad fit into a mass-wide displacement,
	# which is how the pair ended up straddling the mass edge with one socket
	# out on bare wall.
	head += _umb_dir_ema * 0.045
	head.x = clampf(head.x, 0.06, 0.94)
	# Floor well clear of the frame edge: jammed at 0.08 the sockets sat in the
	# top strip where the wall itself is darkest, so hollowing them revealed
	# almost nothing to see.
	head.y = clampf(head.y, 0.10, 0.80)
	var sc := clampf(_umb_scale, 0.6, 2.4)
	# Bounded, and horizontal: the fitted separation is itself a clamped
	# estimate, and multiplying its high end by the silhouette scale put the
	# two sockets a sixth of the frame apart, reading as unrelated holes.
	var asp := 1.7778
	var vt2 := _player.get_video_texture()
	if vt2 != null and vt2.get_height() > 0:
		asp = float(vt2.get_width()) / float(vt2.get_height())
	# HER separation, carried into the ghost and grown WITH the umbra. The
	# ceiling is high enough that Scale genuinely moves it instead of pinning
	# it at a clamp (both the spacing and the radius used to sit on their
	# limits, which is why the eyes never changed size with the mass).
	# DAMPED growth rather than a hard ceiling. Scaling the spacing linearly and
	# then clamping it meant the clamp bound from about Scale 1.5 upward, so
	# the eyes stopped responding to Scale entirely - the "they do not adjust
	# with scale" report. A sub-linear exponent lets them keep growing with the
	# umbra all the way up without ever reaching the width that read as two
	# unrelated holes. At Scale 1 they match HER spacing exactly.
	var esc: float = pow(sc, 0.6)
	var half := Vector2(clampf(unit / asp * 0.5 * esc, 0.010, 0.14), 0.0)
	_umb_eye_l = head - half
	_umb_eye_r = head + half
	# Sized FROM the spacing, so a socket is always a plausible fraction of the
	# gap between the pair rather than an independent number that can swell
	# until the two overlap.
	_umb_eye_rad = clampf(half.x * 0.62, 0.008, 0.075)
	_umb_eyes_ok = true


## Choose the wall. Buckets cells by their chroma vector's own angle (a surface
## is a material is one direction), takes the largest few by area, and runs the
## whole solve for each - the winner is the hypothesis that produces a coherent
## scene, not the one that looks best locally. Every LOCAL statistic is
## contaminated by the fact that her skin, hair and shirt are warm exactly like
## the cream door, which is what defeated area-based and luminance-spread-based
## picks on the reference clip.
##
## The key colour biases this hard: point the picker at the wall and that
## surface wins outright. Left alone, the automatic choice stands.
func _umb_pick_reference(aspect: float) -> void:
	var n := _UMB_W * _UMB_H
	const NB := 12
	var cnt := PackedInt32Array(); cnt.resize(NB)
	var sx := PackedFloat32Array(); sx.resize(NB)
	var sy := PackedFloat32Array(); sy.resize(NB)
	var sz := PackedFloat32Array(); sz.resize(NB)
	var lums: Array = []
	for b in NB:
		cnt[b] = 0; sx[b] = 0.0; sy[b] = 0.0; sz[b] = 0.0
		lums.append(PackedFloat32Array())
	for i in n:
		var cmag := _umb_cmag[i]
		if cmag <= 0.015:
			continue
		var dx := _umb_cr[i] / cmag
		var dy := _umb_cg[i] / cmag
		var dz := _umb_cb[i] / cmag
		var ang: float = atan2(dz - dy, dx - dy)
		var b := int((ang + PI) / TAU * float(NB)) % NB
		if b < 0:
			b += NB
		cnt[b] += 1
		sx[b] += _umb_cr[i]; sy[b] += _umb_cg[i]; sz[b] += _umb_cb[i]
		lums[b].append(_umb_lum[i])
	# rank buckets by area, keep the top few as hypotheses
	var order: Array = []
	for b in NB:
		if cnt[b] >= 40:
			order.append(b)
	order.sort_custom(func(a, c): return cnt[a] > cnt[c])
	if order.is_empty():
		_umb_ref_valid = false
		return
	# The picked key colour's own chroma direction - what the user means by
	# "this is the wall".
	var want := Vector3.ZERO
	if _umb_hue >= 0.0:
		var kc := Color.from_hsv(_umb_hue, 1.0, 1.0)
		var kl := 0.299 * kc.r + 0.587 * kc.g + 0.114 * kc.b
		want = Vector3(kc.r - kl, kc.g - kl, kc.b - kl).normalized()
	# PASS 1 - score every candidate on scene coherence ALONE, with no
	# reference to the picked colour whatsoever.
	var cands: Array = []
	var top_base := 0.0
	for k in mini(3, order.size()):
		var b: int = order[k]
		var mean := Vector3(sx[b], sy[b], sz[b]) / float(cnt[b])
		var mag := mean.length()
		if mag <= 1e-5:
			continue
		var dir := mean / mag
		var ls: PackedFloat32Array = lums[b]
		var sorted := Array(ls)
		sorted.sort()
		var lit: float = sorted[clampi(int(float(sorted.size()) * 0.88), 0, sorted.size() - 1)]
		_umb_analyse(dir, mag, lit)
		var r := _umb_solve(aspect)
		var base := float(r.score)
		if int(r.shad_n) < 8:
			base *= 0.2
		cands.append({"dir": dir, "mag": mag, "lit": lit, "base": base, "cov": float(r.cov)})
		top_base = maxf(top_base, base)
	if cands.is_empty():
		_umb_ref_valid = false
		return
	# PASS 2 - the picked colour BREAKS TIES between plausible surfaces; it may
	# never force an implausible one.
	#
	# This used to be an unconditional x(1 + 1.6 * affinity), up to x2.4, while
	# coherence only separates the right answer from the wrong one by about
	# x1.66. The stored default hue is 0.02 - reddish - which aligns with her
	# skin and the cream door, so a freshly placed marker actively drove the
	# detector to call HER the wall. It then read the scene inside out: the
	# teal wall became the "subject", she became the "shadow", and because the
	# guard is built from the subject mask the effect drew straight over her.
	# Every earlier test hardcoded a teal pick and so never saw it.
	var plausible := top_base * 0.75
	var best: Dictionary = cands[0]
	var best_score := -1.0
	for c in cands:
		var score: float = c.base
		if _umb_hue >= 0.0 and score >= plausible:
			score *= 1.0 + 0.45 * maxf(0.0, (c.dir as Vector3).dot(want))
		if score > best_score:
			best_score = score
			best = c
	var best_dir: Vector3 = best.dir
	var best_mag: float = best.mag
	var best_lit: float = best.lit
	_umb_ref_cov = float(best.cov)
	if _umb_ref_valid and best_dir.dot(_umb_ref_dir) > 0.5:
		# same surface as before - glide, so bucket quantization can't make the
		# reference (and with it every threshold) jitter between ticks
		_umb_ref_dir = _umb_ref_dir.lerp(best_dir, 0.25).normalized()
		_umb_ref_mag = lerpf(_umb_ref_mag, best_mag, 0.25)
		_umb_ref_lit = lerpf(_umb_ref_lit, best_lit, 0.25)
	else:
		_umb_ref_dir = best_dir
		_umb_ref_mag = best_mag
		_umb_ref_lit = best_lit
	_umb_ref_valid = true


# --- undo/redo ---------------------------------------------------------------

## Snapshot markers + the primary clip's trim + every track onto the undo
## stack, called BEFORE a mutation - so the stack always holds "what it looked
## like before this happened". Pass a `key` for edits that repeat rapidly
## during one gesture (a slider drag, a marker or trim/track handle dragged
## along the timeline): a call whose key matches the in-flight gesture just
## extends the coalescing window instead of pushing again, so the whole
## gesture undoes in one Ctrl+Z. Leave `key` empty for one-shot actions
## (add/delete a marker, import/delete a track) - those always open a fresh
## boundary. Trim/track edits are exactly as accident-prone as marker edits -
## see the whole reason this project asked for undo in the first place - so
## they ride the same stack, not a separate one.
func _snapshot() -> Dictionary:
	return {
		"markers": session.markers.duplicate(true),
		"clip_in": session.clip_in, "clip_out": session.clip_out,
		"tracks": session.tracks.duplicate(true),
	}


func _restore_snapshot(snap: Dictionary) -> void:
	session.markers = snap.markers
	session.clip_in = snap.clip_in
	session.clip_out = snap.clip_out
	session.tracks = snap.tracks


## `desc` is what a Ctrl+Z right after this action would revert - shown live
## above the Ramp/Damp buttons (see _refresh_history_label) so undo is never a
## blind guess. Coalesced calls (matching `key`) keep the first desc: the whole
## gesture is one undo step, so it should read as the one action it is (e.g.
## "adjusted Contrast", not the field's very last no-op tick).
func _push_undo(key: String = "", desc: String = "") -> void:
	# A DRAG IS ONE ACTION IN THE HISTORY. Dragging a control back and forth to see
	# how it looks emits a change per step, and the coalescing below is on a 0.9s
	# TIMER - so every place the author paused mid-drag opened a fresh undo entry,
	# and one adjustment left a dozen of them behind. Reported exactly that way:
	# "every single place where I stop will be added to the command history".
	#
	# So while the mouse button is DOWN, the first change of a given key opens the
	# boundary and every later one folds into it, however long the pauses; the
	# boundary closes when the button comes up (cleared in _process). The marker
	# itself still updates on every step - the live preview is the whole point of
	# dragging - it is only the HISTORY that waits for the release, so one undo
	# returns to where things stood before the drag began.
	#
	# Keyed on the BUTTON rather than on Slider.drag_started/drag_ended so one rule
	# covers the colour wheel and the region box too; neither has those signals and
	# both had the same fault.
	if Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
		if key != "" and key == _undo_press_key:
			return
		_undo_press_key = key
	else:
		_undo_press_key = ""
	# NO TIME-BASED COALESCING. There was a 0.9s window that merged consecutive
	# edits of the same field, and once the mouse button became the boundary above
	# it had nothing left to do except cause the OTHER half of the report: two
	# separate adjustments made within 0.9s of each other folded into one history
	# entry, so Ctrl+Z "often reverts several changes at once". A press is an
	# action, a release ends it, and anything else - a keyboard nudge, a menu pick -
	# is its own step. Nothing merges on a clock.
	_undo_stack.append(_snapshot())
	_undo_descs.append(desc)
	if _undo_stack.size() > _UNDO_LIMIT:
		_undo_stack.pop_front()
		_undo_descs.pop_front()
	_redo_stack.clear()
	_redo_descs.clear()
	_refresh_history_label()


func _undo() -> void:
	if _undo_stack.is_empty():
		return
	_redo_stack.append(_snapshot())
	_restore_snapshot(_undo_stack.pop_back())
	_redo_descs.append(_undo_descs.pop_back())
	_undo_press_key = ""   # the next edit must open its own fresh boundary
	_after_history_restore()


func _redo() -> void:
	if _redo_stack.is_empty():
		return
	_undo_stack.append(_snapshot())
	_restore_snapshot(_redo_stack.pop_back())
	_undo_descs.append(_redo_descs.pop_back())
	_undo_press_key = ""
	_after_history_restore()


## The live preview above the Ramp/Damp buttons - see feedback/0019: "if a user
## uses undo, they have a little preview of what they would be reverting."
## _undo_descs.back() is the description passed to the _push_undo() call that
## opened the CURRENT undo step, i.e. exactly the action Ctrl+Z would revert.
func _refresh_history_label() -> void:
	if _history_label == null:
		return
	if _undo_descs.is_empty():
		_history_label.text = "Undo: nothing yet"
		_history_label.tooltip_text = ""
	else:
		var desc: String = _undo_descs.back()
		_history_label.text = "Undo: " + desc
		_history_label.tooltip_text = "Ctrl+Z would revert: " + desc


## _selected points INTO the array a restore just replaced wholesale, so it's
## dangling - re-resolve it by time in the restored array (or drop the
## selection if that marker no longer exists there) before refreshing anything
## that reads it. Tracks went through the same wholesale replacement - the
## runtime players (see _track_runtime) need reconciling against whatever
## session.tracks now holds, same as _delete_track/_finish_track_import do.
func _after_history_restore() -> void:
	if _selected != null:
		var t: float = float(_selected.get("time", -1.0))
		_selected = null
		for m in session.markers:
			if absf(float(m.time) - t) < 0.0005:
				_selected = m
				break
	_select_generation += 1
	_timeline.selected = _selected
	_reconcile_track_runtime()
	_refresh_lanes()
	_refresh_marker_list()
	_refresh_panel()
	_refresh_history_label()
	_mark_dirty()


## After undo/redo swaps session.tracks wholesale, the live VideoStreamPlayers
## in _track_runtime (built incrementally by _build_track_view) no longer
## necessarily match it 1:1 - a track undo/redo added or removed can leave too
## many or too few. Rebuild runtime state to match: free anything past the
## restored count, (re)build anything missing. Existing entries are trusted
## as-is even if that track's trim/offset changed - _sync_tracks reseeks on
## the next frame regardless, same as any ordinary drag.
func _reconcile_track_runtime() -> void:
	if _composition_parent == null:
		return   # render_mode / no editor UI - tracks aren't interactive there anyway
	while _track_runtime.size() > session.tracks.size():
		var rt: Dictionary = _track_runtime.pop_back()
		if rt.has("player"):
			(rt.player as Node).queue_free()
		if rt.has("wrap"):
			(rt.wrap as Node).queue_free()
		if rt.has("audio") and rt.audio != null:
			(rt.audio as Node).queue_free()
	for i in session.tracks.size():
		if i >= _track_runtime.size():
			_build_track_view(i)


# --- auto-save --------------------------------------------------------------

func _mark_dirty() -> void:
	_dirty = true
	_autosave_cooldown = _AUTOSAVE_DELAY


func _save_session() -> void:
	if session == null or _session_path.is_empty():
		return
	# Once a restart has committed to quitting, _player's playback is no longer a
	# trustworthy live read - the engine's own shutdown sequence stops it before
	# _exit_tree runs, which reports stream_position back as 0. _restart_now already
	# took the definitive capture right before quitting; exit's own catch-all save
	# (below) must not re-derive playhead from a player that may already be torn
	# down, or it silently overwrites the correct saved value with 0 (this was why
	# a restart always landed back at the start of the timeline).
	if _player != null and not _restarting:
		session.playhead = _player.stream_position   # persist where the playhead is
	if _tview != null:
		session.timeline_zoom = _tview.zoom
		session.timeline_view_start = _tview.view_start
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_session_path.get_base_dir()))
	session.save(ProjectSettings.globalize_path(_session_path))
	_dirty = false


## Reload the app to pick up code the assistant just edited, landing back on THIS
## session at the current playhead (both persisted, see _save_session). Standalone
## Godot can't hot-reload GDScript, so this is a clean restart-and-restore.
##
## But a restart mid-flight would corrupt any OTHER assistant run still writing files,
## so it never fires while runs are in progress: it hands the restart to the Assistant,
## which holds it until every agent has returned, THEN restarts (see Assistant.reload_when_idle).
##
## Same deference applies to an export in progress (_render_state != "idle", see
## _poll_render): quitting to restart kills the render/transcode subprocess it's waiting
## on, losing the render (feedback/0027). Checked first, and re-checked by _poll_render
## once the export finishes, since one can start after this was first requested.
func _reload_requested() -> void:
	if _render_state != "idle":
		_reload_after_export = true
		_set_status("⟳  Reload queued - restarting once the export in progress finishes")
		return
	var a := get_tree().get_first_node_in_group("assistant")
	if a != null and a.has_method("reload_when_idle") and bool(a.call("is_busy")):
		a.call("reload_when_idle", Callable(self, "_do_restart"))
		_set_status("⟳  Reload queued - restarting once assistant runs finish")
	else:
		_do_restart()


## The actual restart: relaunch this same executable straight back into --mask-edit on
## the current session (so it reopens here, at the saved playhead), preserving whatever
## engine args (--path etc.) it was launched with.
func _do_restart() -> void:
	_save_session()   # captures the playhead
	if _reload_check_pid > 0:
		return   # a check is already running - don't stack another
	# NEVER restart into code that doesn't compile: the assistant's edit might have a
	# syntax error, and relaunching into it would leave the app unable to open. Validate
	# headless first (same check as scripts/scratchpad.py compile); _process reads the
	# result and only then restarts, or reports the errors and stays put.
	var exe := OS.get_executable_path()
	var proj := ProjectSettings.globalize_path("res://")
	_reload_check_log = ProjectSettings.globalize_path("user://reload_compile_check.log")
	var script := "\"%s\" --headless --path \"%s\" --editor --quit > \"%s\" 2>&1" % [exe, proj, _reload_check_log]
	_reload_check_pid = Subprocess.start("/bin/bash", ["-c", script], "reload check")
	if _reload_check_pid <= 0:
		_set_status("⚠  Couldn't run the pre-reload compile check - NOT reloading (edits left as-is)")
		_reload_check_pid = -1
		return
	_set_status("⟳  Checking the edits compile before reloading…")


## The actual restart, run only once _do_restart's compile check comes back clean:
## relaunch this same executable straight back into --mask-edit on the current session
## (so it reopens here, at the saved playhead), preserving the engine args it had.
func _restart_now() -> void:
	# One last accurate capture right before quitting - the compile check the app just
	# ran can take a while, and the app stays fully interactive while it does, so the
	# playhead _do_restart captured when the check STARTED may already be stale. This
	# is also the last point _player is guaranteed to still report a live position -
	# see _save_session/_exit_tree for why nothing may re-derive it after this.
	_save_session()
	_restarting = true
	var engine_args := PackedStringArray()
	for a in OS.get_cmdline_args():
		if a == "--":
			break                     # everything before the user-args separator
		engine_args.append(a)
	engine_args.append("--")
	engine_args.append("--mask-edit")
	engine_args.append(_session_path)
	OS.set_restart_on_exit(true, engine_args)
	get_tree().quit()


## Poll the pre-reload compile check (see _do_restart). Clean -> restart; errors ->
## block the reload and surface them, so a broken assistant edit never bricks the app.
func _poll_reload_check() -> void:
	if _reload_check_pid <= 0 or Subprocess.alive(_reload_check_pid):
		return
	_reload_check_pid = -1
	var log := FileAccess.get_file_as_string(_reload_check_log) if FileAccess.file_exists(_reload_check_log) else ""
	if not _reload_check_log.is_empty():
		DirAccess.remove_absolute(_reload_check_log)
	var errs := []
	for line in log.split("\n"):
		for marker in ["SCRIPT ERROR", "Parse Error", "Compile Error", "Identifier not found", "Failed to load"]:
			if line.contains(marker):
				errs.append(line.strip_edges())
				break
	if errs.is_empty():
		_restart_now()
	else:
		_set_status("⚠  Reload blocked - the edits don't compile (%d error%s); app left running" % [
			errs.size(), "" if errs.size() == 1 else "s"])
		push_warning("ghost: reload blocked, compile errors:\n" + "\n".join(errs))


## Whatever the debounce hasn't flushed yet lands on disk when the editor goes away -
## closing the window mid-burst never loses the last edit. Always save (not only when
## dirty) so the current playhead is captured even after a pure play/scrub with no edit.
func _exit_tree() -> void:
	# Hand the shared export button back on the way out, or leaving Masking would
	# leave every other mode without one - see _build_export_ui.
	var ch_out := _chrome()
	if ch_out != null:
		if ch_out.exporter != null:
			ch_out.exporter.suppressed = false
		ch_out.bottom_inset = 0.0
	_save_session()
	# Reap every subprocess THIS instance spawned. A detached child keeps running -
	# close the app mid-prep and the ffmpeg transcode carries on invisibly (no "godot"
	# in ps), still touching video.ogv, and the NEXT launch reads the fresh mtime as
	# "already in progress elsewhere" and waits on a writer the user can't see. Killing
	# our own pids is safe - the ECHILD hazard is polling pids we did NOT spawn (see
	# _prep_looks_live's doc). The half-written outputs are already handled: prep re-runs
	# when the lock goes stale and _finish_session rejects truncated video, yt-dlp resumes
	# its own .part.
	#
	# This list is now a BELT, not the braces: every spawn above goes through
	# [Subprocess], so `Boot` reaps anything missing from it when the app closes and the
	# kernel kills the lot if ghost is killed outright. Leaving the explicit list keeps
	# the timing right - these die when MASKING closes, not when the app does, which is
	# what frees video.ogv for the next session.
	for pid in [_prep_video_pid, _prep_audio_pid, _import_pid, _waveform_pid,
			_wavehi_pid, _yt_pid, _reload_check_pid, _render_pid, _transcode_pid,
			_umb_track_pid]:
		Subprocess.stop(int(pid))
	for job in _track_audio_jobs:
		Subprocess.stop(int(job.get("pid", -1)))
	# Join the audio loader if it's still running, or Godot warns about an orphaned
	# thread on close.
	if _audio_thread != null and _audio_thread.is_started():
		_audio_thread.wait_to_finish()
		_audio_thread = null
	if _umb_track_thread != null and _umb_track_thread.is_started():
		_umb_track_thread.wait_to_finish()
		_umb_track_thread = null
	Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _select_marker(m: Dictionary) -> void:
	_selected = m
	_select_generation += 1   # a new marker's edits must never coalesce with the last one's
	_timeline.selected = m
	_refresh_panel()


## Wire every popup under `n` into the modal counter, once, at build time.
## Walks rather than naming them so a control added later is covered for free -
## the cursor bug this exists for was reported on the colour picker, but every
## dropdown in the panel had exactly the same hole.
func _guard_popups(n: Node) -> void:
	var p: Popup = null
	if n is ColorPickerButton:
		p = (n as ColorPickerButton).get_popup()
	elif n is OptionButton:
		p = (n as OptionButton).get_popup()
	elif n is MenuButton:
		p = (n as MenuButton).get_popup()
	if p != null and not p.about_to_popup.is_connected(_on_modal_open):
		p.about_to_popup.connect(_on_modal_open)
		p.popup_hide.connect(_on_modal_close)
	for c in n.get_children():
		_guard_popups(c)


func _on_modal_open() -> void:
	_modal_depth += 1
	_cursor_idle_t = 0.0
	# Show it immediately rather than waiting for the next frame's elif: the
	# click that opens a picker is the same gesture that needs to keep aiming.
	if Input.mouse_mode == Input.MOUSE_MODE_HIDDEN:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _on_modal_close() -> void:
	_modal_depth = maxi(0, _modal_depth - 1)
	_cursor_idle_t = 0.0


func _refresh_panel() -> void:
	# Every slider below is written with set_value_no_signal, but a
	# ColorPickerButton has no such door - assigning `color` can emit
	# color_changed, which lands in _edit. With nothing selected _edit CREATES a
	# marker, so merely repainting the panel could mint one stamped with the
	# DEFAULT hue. Guard the whole sync rather than the two pickers, so any
	# control added here later is safe by default.
	# The flag is only ever meant to be true INSIDE this call. If
	# _refresh_panel_inner ever aborts partway - a null control, a bad index -
	# GDScript unwinds the function and the reset below never runs, and from
	# that moment every _edit() in the session returns early and silently
	# discards the user's work. That failure is indistinguishable from "nothing
	# saves". _process clears it defensively every frame for exactly that
	# reason; this reset is the normal path, that one is the seatbelt.
	_syncing = true
	_refresh_panel_inner()
	_syncing = false


func _refresh_panel_inner() -> void:
	var m: Dictionary = _selected if _selected != null else MaskSession.DEFAULTS
	if OS.has_environment("GHOST_PANEL_DEBUG"):
		print("PANELDBG selected=%s hue_a=%.4f hue_b=%.4f thr=%.3f feather=%.3f sat=%.3f scale=%.3f"
			% [str(_selected != null), float(m.get("hue_a", -1)), float(m.get("hue_b", -1)),
			float(m.get("threshold", -1)), float(m.get("feather", -1)),
			float(m.get("sat_floor", -1)), float(m.get("fx_scale", -1))])
	_kind.select(int(m.get("kind", 0.0)))
	# The stored pick, all three components. It used to rebuild the swatch as
	# from_hsv(hue_a, 0.85, 0.9), which round-trips a colour into a different one.
	_color_a.color = Color.from_hsv(float(m.get("hue_a", 0.02)),
		float(m.get("key_sat", 0.85)), float(m.get("key_val", 0.9)))
	if _color_eye != null:
		_color_eye.color = Color.from_hsv(float(m.get("hue_b", 0.0)), 0.88, 1.0)
	if _color_paint != null:
		# The stored colour itself, all three components - this picker IS the
		# value, not a hue preview of it.
		_color_paint.color = Color.from_hsv(float(m.get("hue_b", 0.0)),
			float(m.get("fx_stick", 0.0)), float(m.get("fx_tint", 0.0)))
	_hue_a.set_value_no_signal(float(m.get("fx_tint", 0.0)))
	_threshold.set_value_no_signal(float(m.get("threshold", 0.24)))
	_feather.set_value_no_signal(float(m.get("feather", 0.12)))
	_sat_floor.set_value_no_signal(float(m.get("sat_floor", 0.18)))
	_effect_a.select(int(m.get("effect_a", 0)))
	_intensity_a.set_value_no_signal(float(m.get("intensity_a", 1.0)))
	_marker_duration.set_value_no_signal(float(m.get("duration", 1.0)))
	_fx_x.set_value_no_signal(float(m.get("fx_x", 0.0)))
	_fx_y.set_value_no_signal(float(m.get("fx_y", 0.0)))
	_fx_scale.set_value_no_signal(float(m.get("fx_scale", 1.0)))
	_fx_density.set_value_no_signal(float(m.get("fx_density", 0.45)))
	_fx_contrast.set_value_no_signal(float(m.get("fx_contrast", 0.5)))
	_fx_speed.set_value_no_signal(float(m.get("fx_speed", 1.0)))
	_fx_lag.set_value_no_signal(float(m.get("fx_lag", 0.35)))
	_fx_smooth.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_gust.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_undul.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_coil.set_value_no_signal(float(m.get("fx_lag", 0.35)))
	_stick.set_value_no_signal(float(m.get("fx_stick", 0.0)))
	_bleed.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_settle.set_value_no_signal(float(m.get("fx_lag", 0.35)))
	_hollow.set_value_no_signal(float(m.get("fx_stick", 0.0)))
	_clown_eye_sl.set_value_no_signal(float(m.get("threshold", 0.24)))
	_clown_drip_sl.set_value_no_signal(float(m.get("sat_floor", 0.18)))
	_clown_smudge_sl.set_value_no_signal(float(m.get("swap", 0.0)))
	_clown_drip_w_sl.set_value_no_signal(float(m.get("fx_y", 0.0)))
	_clown_drip_curve_sl.set_value_no_signal(float(m.get("intensity_b", 0.0)))
	_clown_smile_sl.set_value_no_signal(float(m.get("feather", 0.12)))
	_clown_curve_sl.set_value_no_signal(float(m.get("fx_speed", 1.0)))
	_clown_steady_sl.set_value_no_signal(float(m.get("hue_b", 0.0)))
	_rain_squall.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_au_echo.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_au_time.set_value_no_signal(float(m.get("fx_lag", 0.35)))
	_au_amb.set_value_no_signal(float(m.get("fx_density", 0.45)))
	_au_room.set_value_no_signal(float(m.get("fx_scale", 1.0)))
	_au_reso.set_value_no_signal(float(m.get("fx_contrast", 0.5)))
	_au_bass.set_value_no_signal(float(m.get("fx_stick", 0.0)))
	_clown_feather_sl.set_value_no_signal(float(m.get("fx_x", 0.0)))
	_clown_evidence_sl.set_value_no_signal(float(m.get("resonance", 0.0)))
	# Umbra's six. Every control in this panel is a VIEW onto a stored field and
	# has to be re-read here, or it shows its construction value (0) while the
	# marker holds something else - and the next drag then writes from that
	# wrong starting point. These were added without their sync lines, which is
	# why umbra's settings appeared to reset on every restart.
	_wisp.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_cling.set_value_no_signal(float(m.get("fx_lag", 0.35)))
	_umbra_depth.set_value_no_signal(float(m.get("fx_stick", 0.0)))
	_umbra_reach.set_value_no_signal(float(m.get("threshold", 0.24)))
	_umbra_lead.set_value_no_signal(float(m.get("feather", 0.12)))
	_umbra_gaze.set_value_no_signal(float(m.get("sat_floor", 0.18)))
	_paint_reach.set_value_no_signal(float(m.get("fx_contrast", 0.5)))
	_paint_smooth.set_value_no_signal(float(m.get("fx_smooth", 0.0)))
	_region_on.set_pressed_no_signal(MaskSession.has_region(m))
	_region_soft.set_value_no_signal(float(m.get("reg_soft", 0.0)))
	_resonance.set_value_no_signal(float(m.get("resonance", 0.0)))
	_update_effect_controls(int(m.get("effect_a", 0)))
	_sync_region_overlay()   # selection changed - the box follows the marker
	_refresh_marker_label()


## The control hierarchy in action (MaskSession.EFFECT_CONTROLS): show only the
## option groups the selected effect consumes. Threshold doubles as restore's
## reach - same stored field, relabeled so it says what it does here.
func _update_effect_controls(effect_id: int) -> void:
	var groups: Array = MaskSession.EFFECT_CONTROLS.get(effect_id, [])
	# clear fades out EVERYTHING earlier - it has no target color at all. snow
	# picks its foreground/background split automatically - it has no target
	# color either. arealight lights the whole frame, not a keyed color.
	# meta mirrors the whole workspace - it keys on nothing, so no colour picker.
	var has_color := effect_id != MaskSession.EFFECT_CLEAR and effect_id != MaskSession.EFFECT_SNOW \
		and effect_id != MaskSession.EFFECT_SERPENT and effect_id != MaskSession.EFFECT_AREALIGHT \
		and effect_id != MaskSession.EFFECT_META and effect_id != MaskSession.EFFECT_RAIN \
		and effect_id != MaskSession.EFFECT_AUDIO
	_grp_color.visible = has_color
	# Morph shows only where there's a palette to rotate (the fixed-palette
	# emissives + crystal's glass + fur's key-tinted coat + clown's paint).
	var has_morph: bool = effect_id in [1, 2, 5, MaskSession.EFFECT_RAIN, MaskSession.EFFECT_CRYSTAL,
		MaskSession.EFFECT_SNOW, MaskSession.EFFECT_FUR, MaskSession.EFFECT_SERPENT,
		MaskSession.EFFECT_CLOWN, MaskSession.EFFECT_UMBRA]
	# Umbra's picker is not a key at all - it names the SURFACE the shadow
	# falls on, which is the one piece of scene knowledge the detector cannot
	# always infer alone. Saying so on the label is the difference between a
	# working effect and a confusing one.
	if _key_color_label != null:
		if effect_id == MaskSession.EFFECT_UMBRA:
			_key_color_label.text = "Wall color"
			_key_color_label.tooltip_text = "The surface the shadow falls on - " + \
				"pick it off the wall behind her. Leave it and the detector chooses on its own"
		elif effect_id == MaskSession.EFFECT_REPAINT:
			# Repaint has two colour pickers and they are easy to confuse - name
			# both ends of the swap rather than leaving one called "Key color".
			_key_color_label.text = "Color to replace"
			_key_color_label.tooltip_text = "The colour being painted over - " + \
				"pick it off the thing you want gone (the wall, a shirt)"
		else:
			_key_color_label.text = "Key color"
			_key_color_label.tooltip_text = "The color this channel targets - what it keys or paints"
	_show_field(_hue_a, has_morph)
	_show_field(_threshold, groups.has("keying") or groups.has("reach"), _threshold_label)
	if groups.has("reach"):
		_threshold_label.text = "Reach"
		_threshold_label.tooltip_text = "How wide around the picked color this restore reaches"
	else:
		_threshold_label.text = "Threshold"
		_threshold_label.tooltip_text = "How far a pixel's hue may drift from the key color and still be masked"
	_threshold.tooltip_text = _threshold_label.tooltip_text
	_show_field(_feather, groups.has("keying"))
	_show_field(_sat_floor, groups.has("keying"))
	# Prune the individual pattern knobs this effect never reads ("only show
	# properties that can be used"). Effects absent from PATTERN_KNOBS show them
	# all (the default); listed ones show only their subset.
	var has_pattern := groups.has("pattern")
	var knobs: Array = MaskSession.PATTERN_KNOBS.get(effect_id, MaskSession.PATTERN_KNOBS_ALL) \
		if has_pattern else []
	_show_field(_fx_scale, has_pattern and knobs.has("scale"))
	_show_field(_fx_x, has_pattern and knobs.has("pan"), _fx_x_label)
	_show_field(_fx_y, has_pattern and knobs.has("pan"), _fx_y_label)
	_show_field(_fx_density, has_pattern and knobs.has("coverage"), _fx_density_label)
	_show_field(_fx_contrast, has_pattern and knobs.has("contrast"), _fx_contrast_label)
	_show_field(_fx_speed, has_pattern and knobs.has("velocity"))
	_show_field(_resonance, has_pattern and knobs.has("resonance"))
	_show_field(_fx_lag, groups.has("echo"), _fx_lag_label)
	_show_field(_fx_smooth, groups.has("echo"))
	_show_field(_gust, groups.has("snow"))
	_show_field(_undul, groups.has("fur"))
	_show_field(_coil, groups.has("fur"))
	_show_field(_stick, groups.has("fur"))
	_show_field(_bleed, groups.has("clown"))
	_show_field(_settle, groups.has("clown"))
	_show_field(_hollow, groups.has("clown"))
	_show_field(_clown_eye_sl, groups.has("clown"))
	_show_field(_clown_drip_sl, groups.has("clown"))
	_show_field(_clown_smudge_sl, groups.has("clown"))
	_show_field(_clown_drip_w_sl, groups.has("clown"))
	_show_field(_clown_drip_curve_sl, groups.has("clown"))
	_show_field(_clown_smile_sl, groups.has("clown"))
	_show_field(_clown_curve_sl, groups.has("clown"))
	_show_field(_clown_steady_sl, groups.has("clown"))
	_show_field(_rain_squall, groups.has("rain"))
	for au in [_au_echo, _au_time, _au_amb, _au_room, _au_reso, _au_bass]:
		_show_field(au, groups.has("audio"))
	_show_field(_clown_feather_sl, groups.has("clown"))
	_show_field(_clown_evidence_sl, groups.has("clown"))
	_show_field(_wisp, groups.has("umbra"))
	_show_field(_cling, groups.has("umbra"))
	_show_field(_umbra_depth, groups.has("umbra"))
	_show_field(_umbra_reach, groups.has("umbra"))
	_show_field(_umbra_lead, groups.has("umbra"))
	_show_field(_umbra_gaze, groups.has("umbra"))
	_show_field(_color_eye, groups.has("umbra"))
	_show_field(_color_paint, groups.has("repaint"))
	_show_field(_paint_reach, groups.has("repaint"))
	_show_field(_paint_smooth, groups.has("repaint"))
	# The region is UNIVERSAL - not in EFFECT_CONTROLS, because it restricts WHERE
	# a layer acts rather than what it draws, and that question is meaningful for
	# every effect. Hidden only where the effect draws nothing of its own to
	# confine (restore and clear act on other layers, not on pixels).
	var can_region: bool = effect_id != MaskSession.EFFECT_RESTORE \
		and effect_id != MaskSession.EFFECT_CLEAR
	_show_field(_region_on, can_region)
	_show_field(_region_soft, can_region and _region_on.button_pressed)
	var is_oracle := effect_id == MaskSession.EFFECT_ORACLE
	_fx_lag_label.text = "Lead (s)" if is_oracle else "Lag (s)"
	_fx_lag_label.tooltip_text = "How far ahead it leads" if is_oracle else "How the past is worn"
	_fx_lag.tooltip_text = _fx_lag_label.tooltip_text
	var is_snow := effect_id == MaskSession.EFFECT_SNOW
	var is_arealight := effect_id == MaskSession.EFFECT_AREALIGHT
	var is_clown := effect_id == MaskSession.EFFECT_CLOWN
	var is_umbra := effect_id == MaskSession.EFFECT_UMBRA
	if is_snow:
		_fx_contrast_label.text = "Sensitivity"
		_fx_contrast_label.tooltip_text = "How far snow's fall reaches toward the subject"
	elif is_umbra:
		_fx_contrast_label.text = "Roil"
		_fx_contrast_label.tooltip_text = "Turbulence in the mass - how hard the currents " + \
			"churn inside it and how much its silhouette fluctuates"
	elif is_clown:
		_fx_contrast_label.text = "Smear"
		_fx_contrast_label.tooltip_text = "How ragged and smeared the paint is - drooping eye " + \
			"patches, chewed edges, the mouth dragged into a grin"
	elif effect_id == MaskSession.EFFECT_RAIN:
		_fx_contrast_label.text = "Depth"
		_fx_contrast_label.tooltip_text = "Where the near sheet gives way to the far one - " + \
			"0 puts all the weather BEHIND the subject (confined to the dark background), " + \
			"1 brings it all in front, across the lens"
	elif is_arealight:
		_fx_contrast_label.text = "Envelope"
		_fx_contrast_label.tooltip_text = "Where along the rig's mood this sits - warm, soft, " + \
			"single-source practical at 0, toward cold, hard, full-spectrum multi-source at 1"
	else:
		_fx_contrast_label.text = "Contrast"
		_fx_contrast_label.tooltip_text = "Edge hardness of the pattern - 0.5 is neutral"
	_fx_contrast.tooltip_text = _fx_contrast_label.tooltip_text
	_fx_x_label.text = "Wind X" if (is_snow or effect_id == MaskSession.EFFECT_RAIN) else "Pan X"
	_fx_x_label.tooltip_text = "Fall direction - horizontal component" \
		if is_snow else "Shifts the pattern horizontally over the frame"
	_fx_x.tooltip_text = _fx_x_label.tooltip_text
	_fx_y_label.text = "Wind Y" if (is_snow or effect_id == MaskSession.EFFECT_RAIN) else "Pan Y"
	_fx_y_label.tooltip_text = "Fall direction - vertical component" \
		if is_snow else "Shifts the pattern vertically over the frame"
	_fx_y.tooltip_text = _fx_y_label.tooltip_text
	var is_crystal := effect_id == MaskSession.EFFECT_CRYSTAL
	var is_rain := effect_id == MaskSession.EFFECT_RAIN
	if is_crystal:
		_fx_density_label.text = "Stickiness"
	elif is_clown:
		_fx_density_label.text = "Wear"
	elif is_umbra:
		_fx_density_label.text = "Loom"
	elif is_rain:
		_fx_density_label.text = "Amount"
	else:
		_fx_density_label.text = "Coverage"
	if is_crystal:
		_fx_density_label.tooltip_text = "Pull toward the tracked face's edges"
	elif is_rain:
		_fx_density_label.tooltip_text = "How much rain falls at all. Most columns are " + \
			"EMPTY at the bottom of the slider - that is what makes a drizzle a drizzle " + \
			"rather than a finer downpour"
	elif is_clown:
		_fx_density_label.tooltip_text = "Cracks and chips in the white paint - 0 fresh " + \
			"coat, 1 ruined. Inside the face it MARKS the paint rather than opening it: " + \
			"the coat covers the outline it was given, and wear that perforates a mask " + \
			"just reads as the mask being broken"
	elif is_umbra:
		_fx_density_label.tooltip_text = "How far the mass grows outward along the cast " + \
			"direction, away from her - the looming"
	else:
		_fx_density_label.tooltip_text = "How much of the keyed region the pattern consumes - 0 untouched, 1 fully devoured"
	_fx_density.tooltip_text = _fx_density_label.tooltip_text
	# "Strength" means something different for the two subtractive effects - the
	# ambiguity the feedback flagged (an unlabeled, unexplained "Intensity").
	if effect_id == MaskSession.EFFECT_RESTORE:
		_intensity_label.tooltip_text = "How completely this restore fades out earlier layers on this color"
	elif effect_id == MaskSession.EFFECT_CLEAR:
		_intensity_label.tooltip_text = "How completely this clears every earlier layer"
	else:
		_intensity_label.tooltip_text = "How strongly this layer's effect applies"
	_intensity_a.tooltip_text = _intensity_label.tooltip_text
	_apply_sort()   # visibility/labels just changed - re-rank the now-current set (see _apply_sort)


func _refresh_marker_label() -> void:
	if _selected == null:
		_marker_label.text = "nothing selected - editing plants a ramp here"
	else:
		var t := float(_selected.time)
		var kind_name: String = MaskSession.MARKER_KINDS[int(_selected.get("kind", 0.0))]
		_marker_label.text = "%s @ %s  (%d total)" % \
			[kind_name.capitalize(), MaskTimeline.format_time(t), session.markers.size()]
	_refresh_marker_list()


## The sequential ramp/damp list, pinned to the panel's bottom. Rebuilt wholesale -
## cheap at the marker counts a session actually has, and simpler than diffing.
## Piggybacks on _refresh_marker_label's call sites (add/delete/select/drag all
## already call it) rather than needing its own scattered call sites.
func _refresh_marker_list() -> void:
	if _marker_list == null:
		return
	for c in _marker_list.get_children():
		c.queue_free()
	for m in session.markers:
		var kind_name: String = MaskSession.MARKER_KINDS[int(m.get("kind", 0.0))]
		var b := Button.new()
		b.focus_mode = Control.FOCUS_NONE
		b.alignment = HORIZONTAL_ALIGNMENT_LEFT
		var eff_name: String = MaskSession.MASK_EFFECTS[int(m.get("effect_a", 0))]
		var view_tag := "  · raw ⟲" if int(m.get("view_mode", 2.0)) == 2 else ""
		b.text = "%s   %s · %s%s" % [MaskTimeline.format_time(float(m.time)),
			kind_name.capitalize(), eff_name, view_tag]
		if _selected != null and _selected == m:
			b.add_theme_color_override("font_color", Color(1.0, 0.85, 0.5))
		b.pressed.connect(func(): _select_marker(m))
		_marker_list.add_child(b)


func _on_scrub(t: float) -> void:
	_player.stream_position = t
	_audio.seek(t)


func _play(on: bool) -> void:
	# HOLD A START THE AUDIO ISN'T READY FOR. _ready_with_session already
	# holds the AUTOstart until the threaded load attaches, but pressing play
	# yourself walked straight past that guard: the video (the master clock)
	# ran on while the stream was still loading, so the opening seconds
	# played silent and the audio joined wherever the video had got to. Same
	# hold, same recovery path - _poll_audio_thread starts playback the frame
	# the stream lands, synced.
	if on and _audio_thread != null and _audio_thread.is_started():
		_autostart_pending = true
		_playing = false
		if _player != null:
			_player.paused = true
		_set_status("⏳  Loading audio…")
		return
	if not on:
		_autostart_pending = false   # an explicit pause cancels a pending hold
	_playing = on
	if not _player.is_playing():
		_player.play()
	if not _audio.playing:
		_audio.play(_player.stream_position)
	_player.paused = not on
	_audio.stream_paused = not on
	_audio_holding = false   # any pending catch-up hold is moot once playback state changes
	# THE VIDEO IS THE MASTER CLOCK. The two players don't pause at the same
	# instant (video stops on a decoded-frame boundary, audio on a mix chunk),
	# so every pause/resume cycle - spacebar, the feedback console's freeze -
	# banked a little offset, and nothing ever corrected it. Snap audio to the
	# video on every resume; _process keeps them corrected from there.
	if on:
		_audio.seek(_player.stream_position)
	# Track audios are driven entirely by _sync_tracks (windowed play/seek/pause), so
	# nothing to do for them here.


## Derive the compressed playback sidecar for a session whose prep predates it
## (see _ready_with_session). Fire-and-forget: this run still pays the slow WAV
## load it already started, and every later open of this clip is instant. No
## .part dance - the name is only ever read when it exists AND a session is
## being opened, and a half-written .ogg simply fails to load and falls back.
func _ensure_audio_ogg(abs_wav: String, abs_ogg: String) -> void:
	if FileAccess.file_exists(abs_ogg) or not FileAccess.file_exists(abs_wav):
		return
	Subprocess.start("ffmpeg", PackedStringArray([
		"-y", "-loglevel", "error", "-i", abs_wav,
		"-c:a", "libvorbis", "-q:a", "5", abs_ogg]))


## Worker-thread body: the blocking WAV read (see _ready_with_session). Returns the
## stream; the main thread attaches it in _process the frame the thread finishes.
func _load_wav_threaded(path: String) -> AudioStreamWAV:
	return AudioStreamWAV.load_from_file(path)


## Attach the threaded main-audio load the frame it's ready, started at the live
## position so it's in sync from its first sample. No-op until the thread finishes.
func _poll_audio_thread() -> void:
	if _audio_thread == null or _audio_thread.is_alive():
		return
	var stream = _audio_thread.wait_to_finish()
	_audio_thread = null
	if stream != null and _audio != null:
		_audio.stream = stream
		_apply_main_volume()
	# The autostart was waiting for exactly this: begin playback now, synced at the
	# current (start) position, so the intro plays with its audio from the first
	# sample instead of skipping. _play(true) seeks the audio to the video's position,
	# so it stays in sync. If the load FAILED (stream null) start anyway - a silent
	# editor beats one frozen forever waiting on audio that will never arrive.
	if _autostart_pending:
		_autostart_pending = false
		if _status != null and not render_mode:
			_status.visible = false
		_play(true)
		return
	if stream != null and _audio != null and _playing:
		_audio.play(_player.stream_position if _player != null else 0.0)
		_audio.stream_paused = false


## The main clip's 0/1 audio toggle -> the AudioStreamPlayer's level. -80 dB reads as
## silence; 0 dB is unity. Per-frame the fade overrides this (see _apply_main_fade);
## this covers the moment the toggle is flipped and the initial attach.
func _apply_main_volume() -> void:
	if _audio != null:
		_audio.volume_db = _track_level_db(session.main_volume, 1.0)


## The clip-fade fraction (0 at a faded edge, 1 across the flat middle) for a position
## `local` seconds into a clip of length `span`, given fade_in / fade_out durations.
func _clip_fade_gain(local: float, span: float, fi: float, fo: float) -> float:
	var g := 1.0
	if fi > 0.001 and local < fi:
		g = clampf(local / fi, 0.0, 1.0)
	if fo > 0.001 and local > span - fo:
		g = minf(g, clampf((span - local) / fo, 0.0, 1.0))
	return clampf(g, 0.0, 1.0)


## Exponential audio taper for the pull-rope knob's raw 0..1 reading: pushes the quiet
## end down hard (a real dead zone near the anchor, not just a shallower version of
## "audible") while leaving the top of the pull comparatively spacious, so fine-tuning
## a loud level doesn't blow past it - feedback/0025: "1% volume should be nearly
## inaudible, but isn't"; "the middle volume growth seems very fast". _TAPER_K controls
## how hard the low end is suppressed; the curve is 0 at v=0 and 1 at v=1 regardless.
const _TAPER_K := 5.0
func _volume_taper(v: float) -> float:
	return (exp(_TAPER_K * v) - 1.0) / (exp(_TAPER_K) - 1.0)


## Audio gain in dB for a clip: the pull-rope volume `v` (0..1), exponentially tapered
## (see _volume_taper), times the fade envelope `g` (0..1) gives the linear gain
## fraction. This used to run through lerpf(-40, 0, l) - a shallow floor that left a
## fade idling around a still-audible -40dB for nearly the whole marker-to-marker span,
## then jumped a discontinuous 40dB to silence in the last fraction of a percent
## (feedback/0024: "barely reduces until the very tail end"). An equal-power curve (the
## standard cinematic/DAW fade shape) into a real dB conversion spreads the perceived
## loudness change smoothly across the whole span instead, with no cliff at the end -
## and its own natural flattening near l=1 gives the fine control near the top of the
## pull that feedback/0025 asked for, on top of the taper above.
func _track_level_db(v: float, g: float) -> float:
	var l := _volume_taper(clampf(v, 0.0, 1.0)) * clampf(g, 0.0, 1.0)
	if l < 0.0005:
		return -80.0
	return clampf(linear_to_db(sin(l * PI * 0.5)), -80.0, 0.0)


## The main clip's own fade, applied every frame: the whole composite (video) dims via
## _composition_parent.modulate and the main audio ramps in dB - the same envelope,
## coupled. _composition_parent.modulate alone only reaches the RAW (unshaded) view
## though - mask_split.gdshader's fragment() samples TEXTURE directly rather than
## starting from the built-in COLOR, so the CanvasItem's modulate never reached the
## shaded fx overlay, and with any fx layer active the picture stayed full-opacity no
## matter what the envelope said (feedback/0022). u_fade is the shader's own copy of
## the same `g`, so the fade holds whether or not fx is on screen. Deterministic off
## the playhead, so live and export match.
func _apply_main_fade() -> void:
	if _player == null:
		return
	var cin := session.clip_in
	var cout := session.effective_clip_out()
	var t := _player.stream_position
	# Past the main clip's own kept range, the composite's alpha follows whichever
	# continuation track (see MaskSession.continuation_track_at) actually owns `t` now -
	# its OWN fade_in/fade_out, not main's - since the picture showing there is that
	# track's own independent frame (see _apply_frame_state/_cont_view). No owning
	# track (a gap, or past all content) holds at full opacity, same as before
	# continuation tracks respected their own fade at all.
	var g := 1.0
	if t < cout:
		g = _clip_fade_gain(t - cin, cout - cin, session.main_fade_in, session.main_fade_out)
	else:
		var cont_idx := session.continuation_track_at(t)
		if cont_idx != -1:
			var tr: Dictionary = session.tracks[cont_idx]
			var offset := float(tr.get("offset", 0.0))
			var span := float(tr.get("clip_out", 0.0)) - float(tr.get("clip_in", 0.0))
			g = _clip_fade_gain(t - offset, span, float(tr.get("fade_in", 0.0)), float(tr.get("fade_out", 0.0)))
	if _composition_parent != null:
		_composition_parent.modulate.a = g
	_mat_main.set_shader_parameter("u_fade", g)
	if _audio != null:
		# Audio cuts off exactly at cout, full stop - unlike the picture (see
		# _apply_frame_state), this is NOT extended by a continuation track's window.
		# A continuation track (_split_main's tail, same video_path) already plays its
		# OWN independent audio via _sync_tracks' taudio the moment it's active - gating
		# this on main_visible_at too meant the main clip's audio kept playing right
		# alongside it, doubled with the track's own copy of the same source audio
		# (feedback/0013). Audio ownership passes to the track the moment one exists
		# there - which, after the track's own in-point is re-trimmed independently of
		# cout, can be BEFORE cout (session.track_owns_audio_at) rather than exactly at
		# it. Without this check that brief overlap played main's audio and the track's
		# own copy of the same source at once, right at the handoff (feedback/0014).
		var audible := 0.0 if (t >= cout or session.track_owns_audio_at(t)) else g
		_audio.volume_db = _track_level_db(session.main_volume, audible)


## Master-timeline seconds a dragged clip should snap its start/end to: 0, the playhead,
## the primary clip's end, and every OTHER clip's start and end. exclude_i is the lane
## doing the dragging (-1 = the primary), so a clip never snaps to itself.
func _snap_targets_for(exclude_i: int) -> Array:
	var targets := [0.0, session.effective_clip_out()]
	if _player != null:
		targets.append(_player.stream_position)
	for j in session.tracks.size():
		if j == exclude_i:
			continue
		var t: Dictionary = session.tracks[j]
		var o := float(t.get("offset", 0.0))
		var span := float(t.get("clip_out", 0.0)) - float(t.get("clip_in", 0.0))
		targets.append(o)
		targets.append(o + span)
	return targets


## THE keyboard map (mirrored by the help overlay - keep the two in sync):
## Space play/pause · V cycle view · P hold-to-peek · T import track ·
## Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y undo/redo · F1 help · Esc close help.
## These aren't shortcuts FOR buttons any more - they're the only way; the
## toolbar collapsed to the single Help button. Only live once a clip is
## actually loaded (_player exists); main.gd defers Space entirely while this
## editor is open (see main.gd's KEY_SPACE handling), so this doesn't fight
## Director.next() for the key. echo excluded on undo/redo so a held key
## doesn't spam repeats; a real accident deserves a deliberate press each
## time it's undone.
## Plain _input (not _unhandled_input) so cursor motion resets the idle timer
## even while it's over a panel/button - GUI controls can eat mouse motion
## before it would ever reach _unhandled_input, and hovering the toolbar
## while playing should un-hide the cursor same as moving it over the video.
func _input(event: InputEvent) -> void:
	# THE DRAG BOUNDARY CLOSES HERE, on the release itself - see _push_undo. The
	# first cut closed it in _process, which is wrong for a reason worth writing
	# down: _process returns early in half a dozen states (clip still preparing,
	# audio still transcoding, render mode), so the boundary could stay open across
	# a release and swallow the NEXT adjustment into the previous undo step. _input
	# runs whatever _process is doing, and before any control consumes the event.
	if event is InputEventMouseButton and not event.pressed \
			and event.button_index == MOUSE_BUTTON_LEFT:
		_undo_press_key = ""
	if render_mode:
		return
	# A region drag in flight is tracked HERE, not in the overlay's gui_input, so
	# it survives the cursor leaving the video pane - see _region_drag_motion.
	if _region_drag != "":
		_region_drag_motion(event)
	if not event is InputEventMouseMotion:
		return
	_cursor_idle_t = 0.0
	if Input.mouse_mode == Input.MOUSE_MODE_HIDDEN:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE


func _unhandled_input(event: InputEvent) -> void:
	if render_mode or _player == null:
		return
	if not event is InputEventKey or event.echo:
		return
	# Hold-to-peek wants the RELEASE too - everything below this wants presses
	# only. Hold beats the old toggle button for its actual job ("let me just
	# look at raw for a second"): letting go always puts the effects back.
	if event.keycode == KEY_P:
		_peek_raw = event.pressed
		get_viewport().set_input_as_handled()
		return
	if not event.pressed:
		return
	if event.keycode == KEY_SPACE:
		_play(not _playing)
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_V:
		_cycle_view_mode()
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_T:
		_prompt_import_track()
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_F1:
		_toggle_help()
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_F5:
		_reload_requested()   # restart-and-restore, deferred until assistant runs finish
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_ESCAPE and _help_panel != null and _help_panel.visible:
		# Only claim Escape while help is open - otherwise it stays main.gd's
		# quit key. Consuming it here is what keeps closing the overlay from
		# ALSO quitting ghost.
		_help_panel.visible = false
		get_viewport().set_input_as_handled()
	elif event.ctrl_pressed and event.keycode == KEY_Z:
		if event.shift_pressed:
			_redo()
		else:
			_undo()
		get_viewport().set_input_as_handled()
	elif event.ctrl_pressed and event.keycode == KEY_Y:
		_redo()
		get_viewport().set_input_as_handled()


func _toggle_help() -> void:
	if _help_panel != null:
		_help_panel.visible = not _help_panel.visible


## The keyboard map, centered over the video (F1 / the panel's one Help
## button). This is where the buttons went: everything the old toolbar did is
## a key now, and this overlay is how anyone who never learned the shortcuts
## finds them. Keep in sync with _unhandled_input.
func _build_help_overlay() -> void:
	_help_panel = PanelContainer.new()
	_help_panel.visible = false
	_help_panel.z_index = 40
	var hm := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		hm.add_theme_constant_override("margin_" + side, 20)
	_help_panel.add_child(hm)
	var hv := VBoxContainer.new()
	hv.add_theme_constant_override("separation", 7)
	hm.add_child(hv)
	var ht := Label.new()
	ht.text = "Keyboard"
	ht.add_theme_font_size_override("font_size", 18)
	hv.add_child(ht)
	for pair in [
		["Space", "play / pause"],
		["V", "cycle view - raw → PiP (raw) → PiP (fx) → both → full fx"],
		["P (hold)", "peek raw footage - display only, edits nothing"],
		["T", "import a second video track (picture-in-picture)"],
		["Ctrl+Z", "undo"],
		["Ctrl+Shift+Z", "redo (also Ctrl+Y)"],
		["`", "feedback console"],
		["F11", "fullscreen"],
		["F1", "this help (Esc closes it)"],
	]:
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 14)
		var kl := Label.new()
		kl.text = pair[0]
		kl.custom_minimum_size = Vector2(130, 0)
		kl.add_theme_color_override("font_color", Color(1.0, 0.85, 0.5))
		row.add_child(kl)
		var dl := Label.new()
		dl.text = pair[1]
		dl.add_theme_color_override("font_color", Color(0.78, 0.84, 0.94))
		row.add_child(dl)
		hv.add_child(row)
	hv.add_child(HSeparator.new())
	for note in [
		"Timeline - click/drag scrubs · drag markers to move them · Ctrl+scroll zooms at the cursor, plain scroll pans when zoomed · drag clip/track edges to trim",
		"Editing - touching any knob plants a ramp at the playhead if nothing is selected · sliders are drag-only, the wheel always scrolls the panel",
	]:
		var nl := Label.new()
		nl.text = note
		nl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		nl.custom_minimum_size = Vector2(560, 0)
		nl.add_theme_font_size_override("font_size", 12)
		nl.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
		hv.add_child(nl)
	# Centered over the VIDEO side, not the whole window (the wrapper starts
	# at PANEL_W so the box doesn't sit half-under the left panel). The
	# wrapper ignores the mouse; the box itself still catches its own clicks.
	var wrap := CenterContainer.new()
	wrap.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	wrap.offset_left = PANEL_W
	wrap.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(wrap)
	wrap.add_child(_help_panel)


# --- view mode: main (raw/fx) x inset (hidden/raw/fx) --------------------------
# A per-marker field (MaskSession.VIEW_MODES), not just an editing preference - see
# class doc. There is deliberately no standalone "current mode" variable: the single
# source of truth is always session.at_time(playhead), read fresh every frame in
# _process (live AND render_mode alike) and applied via _apply_frame_state - which
# consumes the CONTINUOUS per-layer amounts, so mode changes fade across their
# marker's window instead of popping.

## The DISPLAY/cycle order - the "evolution": raw, then the inset appears (still
## raw), then the inset gets the effect, then the main screen joins it, then
## full-frame fx alone. VIEW_MODES' own order is append-only storage layout (see
## MaskSession), so the narrative order lives here.
const VIEW_CYCLE := [2, 3, 0, 5, 4, 1]   # raw -> pip_raw -> pip -> fx+raw-pip -> masked_pip -> masked

## The marker GOVERNING time `t` - the same last-wins search at_time() uses
## internally to resolve its `cur`, exposed here so callers that need the
## actual marker OBJECT (not at_time()'s resolved copy) can find it without
## re-deriving the search. Null before the first marker.
func _governing_marker(t: float) -> Variant:
	if session == null:
		return null
	var cur = null
	for m in session.markers:
		if m.time <= t:
			cur = m
	return cur


## The toggle button edits the marker at the playhead exactly like every other panel
## control (see _edit) - cycling is relative to whatever's ACTIVE right now, not to
## some separately-tracked button state, so it can never drift from the timeline.
## That means retargeting _selected to the playhead's own governing marker FIRST:
## _selected may still be pointing at whatever the marker list was last clicked on,
## which never moves the playhead - editing through the stale selection silently
## changes an unrelated marker elsewhere on the timeline and nothing visibly happens.
func _cycle_view_mode() -> void:
	# The cycle is the base looks, then per imported track a pair of stops showing
	# THAT track raw in the PiP: once with a RAW main (pip_raw, 3), once with an FX
	# main (masked_pip_raw, 5) - so you can see the fx composite beside the raw source.
	# Each stop is [view_mode, pip_track]; the list grows/shrinks with the track count.
	var stops := []
	for vm in VIEW_CYCLE:
		stops.append([vm, 0])
	for k in range(1, session.tracks.size() + 1):
		stops.append([3, k])
		stops.append([5, k])
	var cur_vm := 2
	var cur_pt := 0
	if session != null and _player != null:
		var resolved: Dictionary = session.at_time(_player.stream_position)
		cur_vm = int(resolved.get("view_mode", 2.0))
		cur_pt = int(resolved.get("pip_track", 0.0))
	var idx := -1
	for j in stops.size():
		if int(stops[j][0]) == cur_vm and int(stops[j][1]) == cur_pt:
			idx = j
			break
	var nxt: Array = stops[(idx + 1) % stops.size()] if idx >= 0 else stops[0]
	if session != null and _player != null:
		var t: float = _player.stream_position
		_selected = _governing_marker(t)
	_edit("view_mode", float(int(nxt[0])))   # plants/selects a marker if needed; applied next frame
	if _selected != null:
		_selected["pip_track"] = float(int(nxt[1]))
		_mark_dirty()


## Pushes one resolved timeline state (see MaskSession.at_time) into the layers:
## channel params to both materials, each layer's intensities scaled by its own
## PRESENCE amount - so "how present is this layer" is a continuous, blendable
## quantity and a mode transition is a fade, not a toggle. The inset's border
## fades with it (modulate), and fully-absent layers are hidden entirely so they
## cost nothing.
func _apply_frame_state(p: Dictionary) -> void:
	var main_amt := clampf(float(p.get("main_fx", 0.0)), 0.0, 1.0)
	var inset_show := clampf(float(p.get("inset_show", 0.0)), 0.0, 1.0)
	var inset_fx := clampf(float(p.get("inset_fx", 0.0)), 0.0, 1.0)
	if _peek_raw:
		main_amt = 0.0
		inset_show = 0.0
		inset_fx = 0.0
	_last_inset_show = inset_show
	# Which source fills the PiP this frame. Fall back to the main clip if the stored
	# index points past the current track count (e.g. a track was deleted).
	var pt := int(p.get("pip_track", 0.0))
	_pip_track = 0 if (_peek_raw or pt < 0 or pt > session.tracks.size()) else pt
	var t: float = _player.stream_position if _player != null else 0.0
	var env := _env_at(t)
	var layers: Array = p.get("layers", [])
	# Does a chimera layer actually render this frame? _maybe_capture_echo reads this
	# to decide whether the (expensive) track readback is worth doing right now.
	# _temporal_active is the same idea one level up: gates the echo/whisp capture
	# itself on whatever's actually on screen, not on the session's marker list.
	_chimera_active = false
	_audio_layer = {}
	_temporal_active = false
	_clown_active = false
	_umbra_active = false
	_meta_amount = 0.0
	for l in layers:
		var le := int(l.get("effect_a", 0))
		if le == MaskSession.EFFECT_CHIMERA:
			_chimera_active = true
		if le == MaskSession.EFFECT_UMBRA:
			_umbra_active = true
			_umb_hue = float(l.get("hue_a", 0.0))
			# Scale is a real geometric scale of the silhouette now, so it gets
			# room to actually loom - at 1.0 the ghost is exactly her own
			# shadow, and past ~1.5 its head leaves the top of the frame.
			_umb_scale = clampf(float(l.get("fx_scale", 1.0)), 0.4, 5.0)
			_umb_pan = Vector2(float(l.get("fx_x", 0.0)), float(l.get("fx_y", 0.0))) * 0.25
			# Resonance folds in here exactly as it does for the shader's
			# density array below - the loom breathes with the audio, so on a
			# talking clip the ghost surges when the subject speaks.
			var ures := float(l.get("resonance", 0.0))
			_umb_loom = clampf(float(l.get("fx_density", 0.45)) + 0.5 * ures * (env - 0.35), 0.0, 1.0)
			_umb_roil = clampf(float(l.get("fx_contrast", 0.5)), 0.0, 1.0)
			_umb_rise = maxf(0.05, float(l.get("fx_speed", 1.0)))
			_umb_wisp = clampf(float(l.get("fx_smooth", 0.0)), 0.0, 1.0)
			_umb_cling = clampf(float(l.get("fx_lag", 0.35)), 0.0, 1.0)
			# Three umbra-only views onto fields the keying group owns for
			# other effects (umbra never keys, so they are free here).
			_umb_reach = clampf(float(l.get("threshold", 0.24)), 0.0, 1.0)
			_umb_lead = clampf(float(l.get("feather", 0.12)), 0.0, 0.5) * 1.6
			_umb_eye_amt = clampf(float(l.get("sat_floor", 0.18)), 0.0, 1.0)
		if le == MaskSession.EFFECT_CLOWN:
			_clown_active = true
			_clown_fs = clampf(float(l.get("fx_scale", 1.0)), 0.3, 2.5)
			_clown_bleed = clampf(float(l.get("fx_smooth", 0.0)), 0.0, 1.0)
			_clown_settle = clampf(float(l.get("fx_lag", 0.35)), 0.0, 1.0)
			_clown_hollow = clampf(float(l.get("fx_stick", 0.0)), 0.0, 1.0)
			# Four keying/pattern fields clown never keys or patterns with, read
			# here under their clown names. See the vars' own doc for the mappings.
			_clown_evidence = clampf(float(l.get("resonance", 0.0)), 0.0, 1.0)
			# Ranges chosen so the STORED DEFAULTS land on "exactly the measured
			# face": eye 1.0x, no drip, smile at the real mouth's width, no sweep.
			# 1.2x the eye at the bottom of the slider, 2.2x at the stored default,
			# 5.2x wound up. It used to be 1.0 + t*6 - 1.44, which is NEGATIVE below
			# threshold 0.24 - a negative scale turns the hull inside out - and only
			# 1.0x at the default, i.e. a patch exactly the size of the eye opening.
			_clown_eye_size = 1.2 + clampf(float(l.get("threshold", 0.24)), 0.0, 1.0) * 4.0
			_clown_drip = clampf(float(l.get("sat_floor", 0.18)), 0.0, 1.0) - 0.18
			# Both of these sit on fields that default to 0, so 0 has to be a GOOD
			# value rather than "off" - a lean streak with a gentle bow is what a
			# tear of liner looks like, and the sliders take it heavier and wider
			# from there.
			# Its floor is a SOFT edge, not the polygon: a hard-edged oval is the
			# fault this control exists for, so 0 must not be it. The floor is also
			# what rounds the outline off - the patch is a 16-vertex hull pushed
			# outward along its own normals, and at any real Eye size its facets are
			# plainly visible until something averages over them.
			_clown_smudge = 0.45 + clampf(float(l.get("swap", 0.0)), 0.0, 1.0) * 0.55
			_clown_drip_w = 0.35 + clampf(float(l.get("fx_y", 0.0)), 0.0, 1.0) * 0.80
			_clown_drip_curve = 0.45 + clampf(float(l.get("intensity_b", 0.0)),
				0.0, 1.0) * 1.55
			_clown_smile_w = 1.0 + (clampf(float(l.get("feather", 0.12)), 0.0, 0.5) - 0.12) * 8.0
			_clown_smile_curve = (clampf(float(l.get("fx_speed", 1.0)), 0.0, 2.0) - 1.0) * 0.12
			# fx_x defaults to 0, and 0 must be a GOOD value: some feather is
			# always wanted, so the range runs from soft to softer.
			_clown_feather = 0.012 + clampf(float(l.get("fx_x", 0.0)), 0.0, 1.0) * 0.055
			# hue_b defaults to 0, and 0 has to be a GOOD value rather than "off" -
			# some smoothing is always wanted, so the range runs from calm to calmer.
			_ft_sigma = 2.5 + clampf(float(l.get("hue_b", 0.0)), 0.0, 1.0) * 7.5
			_clown_region = Vector4(
				float(l.get("reg_x0", 0.0)), float(l.get("reg_y0", 0.0)),
				float(l.get("reg_x1", 1.0)), float(l.get("reg_y1", 1.0)))
		if le == 5 or le == 7 or le == MaskSession.EFFECT_SNOW or le == MaskSession.EFFECT_ORACLE \
				or le == MaskSession.EFFECT_SERPENT or le == MaskSession.EFFECT_CHIMERA \
				or le == MaskSession.EFFECT_CLOWN:
			_temporal_active = true
		# The META mirror's strength - the same env x intensity the shader gets as
		# this layer's weight. Drives whether the (expensive) workspace readback runs
		# at all this frame, and how far the render-mode editor chrome has revealed.
		if le == MaskSession.EFFECT_AUDIO:
			_audio_layer = l
		if le == MaskSession.EFFECT_META:
			_meta_amount = maxf(_meta_amount,
				clampf(float(l.get("env", 0.0)) * float(l.get("intensity_a", 0.0)), 0.0, 1.0))
	# The audio layer resolves to bus parameters rather than pixels. Applied every
	# frame like everything else, so its envelope ramps the sound in exactly as a
	# visual layer's ramps its paint - and so an export, which walks the same
	# per-frame path, mixes the same sound the editor previewed.
	_ensure_audio_bus()
	_apply_audio_fx(_audio_layer)
	# Build the layer arrays ONCE; only the weights differ per surface (each
	# material multiplies its own presence in). Arrays are pushed at FULL declared
	# length - a short uniform array is silently dropped (flame.gdshader lesson).
	var n: int = mini(layers.size(), MaskSession.MAX_LAYERS)
	var hues := PackedFloat32Array()
	var effects := PackedInt32Array()
	var base_w := PackedFloat32Array()
	var offs := PackedVector2Array()
	var scales := PackedFloat32Array()
	var densities := PackedFloat32Array()
	var contrasts := PackedFloat32Array()
	var glows := PackedFloat32Array()
	var speeds := PackedFloat32Array()
	var smooths := PackedFloat32Array()   # raw fx_smooth - snow's Gust; echo bakes its own use into echo_w below
	var tdirs := PackedVector3Array()
	var echo_w := PackedFloat32Array()
	var echo_lag := PackedInt32Array()
	var lagf := PackedFloat32Array()   # raw fx_lag - fur's Coil knob (echo's use is baked into echo_w/echo_lag)
	var sticks := PackedFloat32Array()   # raw fx_stick - fur's Stickiness (0 = today's free coat)
	var tints := PackedFloat32Array()    # fx_tint - Morph, palette hue rotation (0 = natural)
	var accents := PackedFloat32Array()  # hue_b - umbra's ghost-eye colour (0 = red)
	var regions := PackedVector4Array()  # the layer's box in frame UV (x0,y0,x1,y1)
	var regsofts := PackedFloat32Array() # how gradually it fades out at that border
	var slot_frac := fposmod((_player.stream_position if _player != null else 0.0) / _ECHO_INTERVAL, 1.0)
	for i in MaskSession.MAX_LAYERS:
		if i < n:
			var l: Dictionary = layers[i]
			var res := float(l.get("resonance", 0.0))
			hues.append(float(l.get("hue_a", 0.0)))
			effects.append(int(l.get("effect_a", 0)))
			base_w.append(float(l.get("env", 0.0)) * float(l.get("intensity_a", 0.0)))
			offs.append(Vector2(float(l.get("fx_x", 0.0)), float(l.get("fx_y", 0.0))))
			scales.append(float(l.get("fx_scale", 1.0)))
			# Resonance folds in CPU-side: the audio envelope swings coverage around
			# its nominal (loud opens the field, quiet closes it) and pulses the rim.
			densities.append(clampf(float(l.get("fx_density", 0.45)) + 0.5 * res * (env - 0.35), 0.0, 1.0))
			contrasts.append(float(l.get("fx_contrast", 0.5)))
			glows.append(1.0 + res * env * 1.3)
			speeds.append(maxf(0.05, float(l.get("fx_speed", 1.0))))
			smooths.append(clampf(float(l.get("fx_smooth", 0.0)), 0.0, 1.0))
			# The echo's temporal kernel: weights over the 8 ring ages, centered
			# on the layer's lag. Age of ring index k is (k + slot_frac) slots -
			# continuous in playback time, so a spread kernel (Smoothing > 0)
			# glides through the ring with no steps; Smoothing ~ 0 collapses to
			# the nearest single frame - the held-frame stutter, now at an
			# adjustable distance. Pure function of position + fields: live and
			# export blend identically.
			var lag_slots := clampf(float(l.get("fx_lag", 0.35)) / _ECHO_INTERVAL, 0.0, 7.0)
			var smooth_amt := clampf(float(l.get("fx_smooth", 0.0)), 0.0, 1.0)
			echo_lag.append(clampi(int(round(lag_slots)), 0, 7))
			lagf.append(float(l.get("fx_lag", 0.35)))
			sticks.append(clampf(float(l.get("fx_stick", 0.0)), 0.0, 1.0))
			tints.append(clampf(float(l.get("fx_tint", 0.0)), 0.0, 1.0))
			accents.append(clampf(float(l.get("hue_b", 0.0)), 0.0, 1.0))
			var w := PackedFloat32Array()
			var wsum := 0.0
			for k in 8:
				var wv: float
				if smooth_amt < 0.02:
					wv = 1.0 if k == clampi(int(round(lag_slots - slot_frac)), 0, 7) else 0.0
				else:
					wv = exp(-absf(float(k) + slot_frac - lag_slots) / (smooth_amt * 2.5))
				w.append(wv)
				wsum += wv
			for k in 8:
				echo_w.append(w[k] / maxf(wsum, 0.0001))
			# The target hue's normalized chroma direction, for erase's
			# projection-subtraction (see the shader: erase is subtraction,
			# not classification - no gates, no boundary rings).
			var tc := Color.from_hsv(float(l.get("hue_a", 0.0)), 1.0, 1.0)
			var tl := 0.299 * tc.r + 0.587 * tc.g + 0.114 * tc.b
			tdirs.append(Vector3(tc.r - tl, tc.g - tl, tc.b - tl).normalized())
			# The layer's region box, normalized here so the shader never has to
			# wonder which corner is which (a box dragged past itself is still a
			# box). Sessions saved before regions existed have none of these
			# fields; the DEFAULTS fall back to the whole frame, which is exactly
			# what they rendered before.
			# SNAPPED HERE, not only when dragged. A drag snaps an edge that lands
			# near the frame's border onto it (see _snap_edge), but a value stored
			# before that existed - or set any other way - keeps whatever it has,
			# and an edge sitting 0.002 off the top is a four-pixel band of exactly
			# the colour the layer was added to remove, hugging the top of the
			# picture. Snapping at the point of USE means every session gets the
			# same treatment, and the shader can then skip that side's falloff
			# entirely because the edge is genuinely flush.
			regions.append(Vector4(
				_snap_edge(minf(float(l.get("reg_x0", 0.0)), float(l.get("reg_x1", 1.0)))),
				_snap_edge(minf(float(l.get("reg_y0", 0.0)), float(l.get("reg_y1", 1.0)))),
				_snap_edge(maxf(float(l.get("reg_x0", 0.0)), float(l.get("reg_x1", 1.0)))),
				_snap_edge(maxf(float(l.get("reg_y0", 0.0)), float(l.get("reg_y1", 1.0))))))
			regsofts.append(clampf(float(l.get("reg_soft", 0.0)), 0.0, 1.0))
		else:
			hues.append(0.0)
			effects.append(0)
			base_w.append(0.0)
			offs.append(Vector2.ZERO)
			scales.append(1.0)
			densities.append(0.0)
			contrasts.append(0.5)
			speeds.append(1.0)
			smooths.append(0.0)
			echo_lag.append(0)
			lagf.append(0.0)
			sticks.append(0.0)
			tints.append(0.0)
			accents.append(0.0)
			for k in 8:
				echo_w.append(1.0 if k == 0 else 0.0)
			glows.append(1.0)
			tdirs.append(Vector3(1, 0, 0))
			regions.append(Vector4(0.0, 0.0, 1.0, 1.0))   # unused slot: whole frame
			regsofts.append(0.0)
	# Which source actually has valid picture at `t`: the main clip's own kept range,
	# or - once that's ended - whichever continuation track (see
	# MaskSession.continuation_track_at) has picked it up. Each continuation track
	# renders through its OWN player (see _sync_tracks), never by borrowing _player's -
	# see continuation_track_at's doc for why that used to be a fragile invariant.
	var main_active := t < session.effective_clip_out()
	var cont_idx := -1 if main_active else session.continuation_track_at(t)
	var cont_tex: Texture2D = null
	if cont_idx != -1 and cont_idx < _track_runtime.size() and _track_runtime[cont_idx].has("player"):
		var cp: VideoStreamPlayer = _track_runtime[cont_idx].player
		if cp != null and cp.get_video_texture() != null and cp.get_video_texture().get_height() > 0:
			cont_tex = cp.get_video_texture()
	_cont_view.visible = cont_tex != null
	if cont_tex != null:
		_cont_view.texture = cont_tex
	for pair in [[_mat_main, main_amt], [_mat_inset, inset_fx]]:
		var mat: ShaderMaterial = pair[0]
		var amt: float = pair[1]
		mat.set_shader_parameter("u_threshold", p.threshold)
		mat.set_shader_parameter("u_feather", p.feather)
		mat.set_shader_parameter("u_sat_floor", p.sat_floor)
		# The wisp field's clock is the CLIP's own playback position, never
		# wall-time - live and export step the same clock, so a session
		# reproduces its exact wisps frame-for-frame (flame.gdshader discipline).
		mat.set_shader_parameter("u_time", t)
		var tex := (_player.get_video_texture() if _player != null else null) if main_active else cont_tex
		if tex != null and tex.get_height() > 0:
			mat.set_shader_parameter("u_aspect", float(tex.get_width()) / float(tex.get_height()))
			# The ACTIVE source's own pixel grid - a continuation track may be a
			# different size from the main clip, so this tracks `tex`, not _src_size.
			mat.set_shader_parameter("u_texel",
				Vector2(1.0 / float(tex.get_width()), 1.0 / float(tex.get_height())))
		var ws := PackedFloat32Array()
		for i in MaskSession.MAX_LAYERS:
			ws.append(base_w[i] * amt)
		mat.set_shader_parameter("u_l_count", n)
		mat.set_shader_parameter("u_l_hue", hues)
		mat.set_shader_parameter("u_l_effect", effects)
		mat.set_shader_parameter("u_l_w", ws)
		mat.set_shader_parameter("u_l_off", offs)
		mat.set_shader_parameter("u_l_scale", scales)
		mat.set_shader_parameter("u_l_dens", densities)
		mat.set_shader_parameter("u_l_con", contrasts)
		mat.set_shader_parameter("u_l_glow", glows)
		mat.set_shader_parameter("u_l_speed", speeds)
		mat.set_shader_parameter("u_l_smooth", smooths)
		mat.set_shader_parameter("u_l_ew", echo_w)
		mat.set_shader_parameter("u_l_elag", echo_lag)
		mat.set_shader_parameter("u_l_lagf", lagf)
		mat.set_shader_parameter("u_l_stick", sticks)
		mat.set_shader_parameter("u_l_tint", tints)
		mat.set_shader_parameter("u_l_accent", accents)
		mat.set_shader_parameter("u_l_tdir", tdirs)
		mat.set_shader_parameter("u_l_region", regions)
		mat.set_shader_parameter("u_l_regsoft", regsofts)
		# Chimera's graft source: the first track's live frame. The explicit
		# flag matters - the sampler's default-black fallback must never read
		# as footage.
		var track_tex: Texture2D = null
		if _track_runtime.size() > 0 and _track_runtime[0].has("player"):
			var tp: VideoStreamPlayer = _track_runtime[0].player
			if tp != null and tp.get_video_texture() != null \
					and tp.get_video_texture().get_height() > 0:
				track_tex = tp.get_video_texture()
		mat.set_shader_parameter("u_track_on", 1 if track_tex != null else 0)
		if track_tex != null:
			mat.set_shader_parameter("u_track", track_tex)
	# The active source's own raw frame - and everything sourced from it (the fx
	# overlay, its own PiP inset) - only while the timeline actually claims this
	# instant (feedback/0009: past the main track's own trim, with no track
	# continuing it here, video.ogv kept rendering anyway).
	_player.visible = main_active
	_fx_overlay.visible = main_amt > 0.001 and (main_active or cont_tex != null)
	# Main clip's PiP only when it's the selected source; _sync_tracks re-confirms.
	_mask_wrap.visible = inset_show > 0.001 and _pip_track == 0 and main_active
	_mask_wrap.modulate.a = inset_show
	if _view_label != null:
		if _pip_track > 0:
			var main_lbl := "fx" if main_amt > 0.5 else "raw"
			_view_label.text = "🎞  main %s · Track %d (raw)" % [main_lbl, _pip_track]
		else:
			match MaskSession.VIEW_MODES[clampi(int(p.get("view_mode", 2.0)), 0, MaskSession.VIEW_MODES.size() - 1)]:
				"raw":            _view_label.text = "🎬  Raw"
				"pip_raw":        _view_label.text = "🖼  PiP (raw)"
				"pip":            _view_label.text = "🖼  PiP (fx)"
				"masked_pip_raw": _view_label.text = "🎭  Full (fx) · PiP (raw)"
				"masked_pip":     _view_label.text = "🎭  Both (fx)"
				"masked":         _view_label.text = "🎭  Full (fx)"


# --- per-frame: push the timeline's blended params into the shader ---------------

func _process(_dt: float) -> void:
	# SEATBELT. _syncing is only valid inside a single synchronous
	# _refresh_panel() call, so finding it still set at the top of a frame means
	# that call aborted midway. Left latched it silently swallows every
	# subsequent edit - the user's settings appear to save and are simply
	# dropped. Clearing it here bounds the damage to one repaint instead of the
	# rest of the session.
	_syncing = false
	match _prep_state:
		"prepping_video":
			if Subprocess.alive(_prep_video_pid):
				_set_status("⏳  Preparing clip (video)…  %d%%" % _read_prep_pct())
				return
			_promote_part(String(_pending.video))
			_start_prep_audio()
			return
		"prepping_audio":
			if Subprocess.alive(_prep_audio_pid):
				return
			_promote_part(String(_pending.audio))
			_promote_part(String(_pending.audio).get_basename() + ".ogg")
			_clear_lock(_pending.dir)
			_prep_state = "idle"
			_finish_session(_pending.source, _pending.video, _pending.audio)
			return
		"waiting_external":
			var abs_video := ProjectSettings.globalize_path(_pending.video)
			var abs_audio := ProjectSettings.globalize_path(_pending.audio)
			if FileAccess.file_exists(abs_video) and FileAccess.file_exists(abs_audio):
				_prep_state = "idle"
				_finish_session(_pending.source, _pending.video, _pending.audio)
				return
			if _prep_looks_live(_pending.dir, _pending.video):
				return   # still going elsewhere - keep waiting
			# Nothing's touched it in a while and it never produced both files - the
			# other writer died mid-prep. Run it ourselves rather than waiting forever.
			_prep_state = "idle"
			_prep(_pending.source, _pending.dir, _pending.video, _pending.audio)
			return
		_:
			pass
	if _yt_state != "idle":
		if _yt_pid > 0 and not Subprocess.alive(_yt_pid):
			_yt_pid = -1
			_yt_step_done()
		else:
			# Live readout every frame, exporter-style: a step with no percent
			# of its own at least ticks its elapsed seconds, so a slow venv
			# bootstrap or pip install never reads as a hang.
			var el := int(Time.get_unix_time_from_system() - _yt_step_started)
			match _yt_state:
				"venv":
					_set_status("⏳  Setting up ghost's download venv (one-time)…  %ds" % el)
				"pip":
					_set_status("⏳  Installing yt-dlp into the venv…  %ds" % el)
				"downloading":
					var pct := _yt_pct()
					_set_status("⏳  Downloading…  " + (pct if not pct.is_empty()
						else "connecting  %ds" % el))
					# yt-dlp's WARNING/ERROR lines echo into godot's log the
					# moment they appear (deduped) - a throttle warning is only
					# useful WHILE the download crawls, not after it finishes.
					for line in _yt_tail_window().split("\n"):
						var s := line.strip_edges()
						if (s.begins_with("WARNING") or s.begins_with("ERROR")) \
								and not _yt_echoed.has(s):
							_yt_echoed[s] = true
							print("ghost yt: ", s)
	if _waveform_pid > 0 and not Subprocess.alive(_waveform_pid):
		_waveform_pid = -1
		var abs_wave := ProjectSettings.globalize_path(_waveform_path)
		if FileAccess.file_exists(abs_wave):
			_load_waveform(abs_wave)
	_poll_wave_hi(_dt)
	if _import_pid > 0 and not Subprocess.alive(_import_pid):
		_import_pid = -1
		_finish_track_import()
	_poll_audio_thread()   # cheap bool check; attaches the main audio once its load finishes
	if not _track_audio_jobs.is_empty():
		_poll_track_audio()
	if _lanes_col != null:
		var visible_lanes := 0
		for c in _lanes_col.get_children():
			if c is Control and (c as Control).visible:
				visible_lanes += 1
		_apply_lane_reserved(visible_lanes)
	if _reload_check_pid > 0:
		_poll_reload_check()   # gates the reload on a clean headless compile
	if session == null or _player == null:
		return
	_sync_source_size()   # one-shot; adopts the decoder's own frame size
	# THE EXPORT CLOCK. In render mode the movie is driven by accumulated fixed-fps time
	# (_render_t), NOT the video/audio stream positions - those can drift a little, stall,
	# or (when a source is shorter than the session) end early, and binding the export to
	# them truncated the movie to the shorter stream and let effects/video slide out of
	# alignment. render_pos is where the timeline is at this recorded frame; the decoded
	# video normally free-runs within ~2 frames of it (measured), so it's only re-synced
	# on real divergence, never seeked every frame (which would thrash the OGV decoder).
	var render_pos := _player.stream_position
	if render_mode:
		_render_t += _dt
		render_pos = _render_t   # export runs the whole timeline from 0 (see _build_render_view)
		if _player.is_playing() and absf(_player.stream_position - render_pos) > 0.2:
			_player.stream_position = render_pos
	# Restore the persisted playhead once, now that the player has had a frame to start
	# (a same-frame seek in _ready_with_session is ignored). Retried until it takes or a
	# short budget lapses, then the seek + audio sync catch up from there.
	if _pending_restore >= 0.0:
		var target := clampf(_pending_restore, session.clip_in, session.effective_clip_out())
		_player.stream_position = target
		if _audio != null:
			_audio.seek(target)
		if absf(_player.stream_position - target) < 1.0 or _pending_restore_tries > 20:
			_pending_restore = -1.0
		_pending_restore_tries += 1
	# Same resolve, live or exported: whatever the timeline says at this instant is
	# what's shown - render_mode doesn't special-case a fixed "always masked" look. Live
	# reads the (just-restored) video position; export reads the deterministic movie clock.
	_apply_frame_state(session.at_time(render_pos if render_mode else _player.stream_position))
	_apply_main_fade()   # dim the whole composite + main audio at the clip's fade edges
	# The fx overlay re-draws whichever raw source is actually active this frame -
	# _player while the main clip's own kept range covers it, else _cont_view's
	# texture (a continuation track's own independent decode, see _apply_frame_state) -
	# each only while actually on screen; "raw" mode skips both (and the shader
	# passes they'd otherwise cost) entirely.
	if _fx_overlay != null and _fx_overlay.visible:
		_fx_overlay.texture = _player.get_video_texture() if _player.visible else _cont_view.texture
	if _pip_view != null and _mask_wrap.visible:
		_pip_view.texture = _player.get_video_texture()
	_maybe_capture_echo()
	_maybe_capture_face()
	_maybe_capture_umbra()
	# META: while a meta layer is live, capture the editor's own frame for the mirror
	# and (in export) lerp the editor chrome into view. Both are gated on _meta_amount
	# so the expensive readback only ever runs during an actual meta section.
	if _meta_amount > 0.001:
		_capture_workspace()
	if render_mode:
		_apply_meta_chrome(_meta_amount)
	_push_anchor()
	_step_paint_sim()
	if _umbra_active:
		_umb_ensure_track()
		_umb_poll_track()
	_step_umbra_sim()
	# Standing A/V drift correction (see _play: video is the master clock).
	# 0.15s tolerance sits above audio mix-chunk granularity so this never
	# chatters. Video ahead of audio: seek audio forward (a silent skip -
	# no artifact). Audio ahead of video: HOLD audio in place (pause, no
	# seek) until video's decode catches back up, instead of seeking it
	# backward - a backward seek replays audio just heard, audible as an
	# echo/glitch (feedback/0012, which is why the backward correction was
	# dropped entirely). But dropping it left the OTHER direction fully
	# uncorrected: _maybe_capture_echo's synchronous GPU readback (whisp/
	# echo/chimera/snow/oracle/serpent) stalls this thread every
	# _ECHO_INTERVAL and freezes _player.stream_position for the stall's
	# duration while _audio keeps flowing on its own thread, so audio comes
	# out ahead after every single capture - uncorrected, that drift only
	# ever grows over a session (feedback/0025). Holding (not seeking) closes
	# that gap without ever replaying already-heard audio.
	# The hold's own exit check has to run OUTSIDE the "_audio.playing" gate:
	# setting stream_paused = true immediately flips .playing to false (that's
	# just what a paused AudioStreamPlayer reports), so gating the exit check
	# on .playing meant a hold could engage but never release - audio stayed
	# silent, permanently, until a manual pause/play cycle called _play()'s own
	# "if not _audio.playing: _audio.play(...)" restart (feedback/0027).
	if _playing and _audio_holding:
		var hold_drift := _player.stream_position - _audio.get_playback_position()
		if hold_drift >= -0.02:
			_audio.stream_paused = false
			_audio_holding = false
	elif _playing and _audio.playing and not _audio.stream_paused:
		var av_drift := _player.stream_position - _audio.get_playback_position()
		if av_drift > 0.15:
			_audio.seek(_player.stream_position)
		elif av_drift < -0.15:
			_audio.stream_paused = true
			_audio_holding = true
	_sync_tracks()
	# A trimmed clip's OUT point is a hard wall for playback (both live preview
	# and the export relaunch - export additionally needs a QUIT, not just a
	# pause, since Movie Maker keeps recording for as long as the process runs).
	# clip_in is not enforced here on purpose: scrubbing earlier to look at
	# trimmed-away footage while editing is fine, only PLAYBACK (and export) are
	# bounded to the kept range. Bounded by content_end(), not clip_out directly -
	# _split_main trims clip_out but appends the trimmed tail as a track at that
	# same offset, so the show must keep running (the master clock IS the main
	# clip's own decode position) until that track's own span is done too.
	var content_stop := session.content_end()
	if render_mode:
		# The export ends on the MOVIE clock reaching the session's own content length -
		# never on a source stream ending. Binding it to the streams truncated the movie
		# to whichever of the audio/video was shorter than the session (the 13:37 -> 13:00
		# report) and, since a stalled/ended source froze the position the effects keyed
		# off, slid the whole show out of alignment. Past a source's own end the timeline
		# simply shows raw for that region, but the movie still runs its full length with
		# the full audio. content_end() already folds in clip_out + any continuation track;
		# the export runs [0, content_end()] - the whole timeline, matching the editor.
		if _render_t >= content_stop - 0.001:
			get_tree().quit()
			return
	elif content_stop < session.duration and _player.stream_position >= content_stop and _playing:
		_play(false)
	if render_mode:
		return
	# Cross a marker during playback and it selects itself - the panel and timeline
	# follow the playhead, so scrubbing through a whole session no longer means
	# clicking every tiny flag by hand. Only while actually playing; a paused scrub
	# or a deliberate click in the marker list is left alone.
	# ...unless the author has switched Follow off, which is what makes tuning a
	# marker's knobs DURING playback possible at all: with it on, crossing the next
	# marker takes the panel away mid-adjustment.
	if _playing and _follow_playhead != null and _follow_playhead.button_pressed:
		var governing: Variant = _governing_marker(_player.stream_position)
		if governing != null and governing != _selected:
			_select_marker(governing)
	# Auto-hide the cursor once it's been still for a beat during playback -
	# _input() above resets the timer and un-hides on any motion, and pausing
	# (or a mouse click, which also fires motion-adjacent hover) restores it
	# immediately rather than leaving an editor with a phantom-hidden pointer.
	# ... and NEVER while a popup is up. A colour picker or a dropdown is an
	# embedded SUBWINDOW: mouse motion inside it never reaches this node's
	# _input(), so the idle timer keeps running, the auto-hide fires, and the
	# pointer vanishes inside the one place it most needs to be visible - with
	# no way to un-hide it, because moving the mouse is exactly what stopped
	# being heard. Suspending the timer here also un-hides on the way in, via
	# the elif.
	if _playing and _modal_depth <= 0:
		_cursor_idle_t += _dt
		if _cursor_idle_t >= _CURSOR_HIDE_DELAY and Input.mouse_mode == Input.MOUSE_MODE_VISIBLE:
			Input.mouse_mode = Input.MOUSE_MODE_HIDDEN
	elif Input.mouse_mode == Input.MOUSE_MODE_HIDDEN:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	if _time_label != null:
		_time_label.text = "%s / %s" % [
			MaskTimeline.format_time(_player.stream_position), MaskTimeline.format_time(session.duration)]
	# Auto-save: any edit marks the session dirty (see _mark_dirty); it lands on
	# disk shortly after the LAST change in a burst - a slider drag saves once,
	# not once per pixel of mouse travel.
	if _dirty:
		_autosave_cooldown -= _dt
		if _autosave_cooldown <= 0.0:
			_save_session()
	_poll_render()


# --- shared bottom-right notification: prep progress AND export progress use the
# --- same label, same position - one status line, whatever phase produced it -----

func _build_status_label() -> void:
	_status = Label.new()
	_status.name = "MaskStatus"
	_status.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	# Right edge clears the console's >_ slot (-156..-116, see console.gd).
	_status.offset_left = -592
	_status.offset_top = -64
	_status.offset_right = -160
	_status.offset_bottom = -28
	_status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_status.visible = false
	# Built before the chrome/timeline exist (see the doc comment on the _status
	# var) so prep messages show pre-session; once the chrome IS built, MaskTimeline's
	# near-opaque background (mask_timeline.gd's _draw) is added afterward and, being
	# a later sibling, painted over this label - silently swallowing every status this
	# label ever shows for the rest of the session (export progress included; feedback
	# 0026). z_index keeps it on top regardless of what gets added later.
	_status.z_index = 5
	add_child(_status)


# --- export: relaunch in Movie Maker mode (--mask-render), then ffmpeg mux ------

## The shared session furniture, if this session has any (the render relaunch and
## the gates run without it). BY GROUP, not by walking the tree: the first cut
## walked two levels down from the root and the exporter is three - root, main,
## chrome, exporter - so it silently found nothing and the suppression below never
## happened. The button stayed on top, greyed, swallowing every click.
func _chrome() -> Node:
	if get_tree() == null:
		return null
	return get_tree().get_first_node_in_group("ghost_chrome")


func _build_export_ui() -> void:
	_export_btn = Button.new()
	_export_btn.focus_mode = Control.FOCUS_NONE
	_export_btn.text = "⤓"                    # icon-only - matches assistant.gd's chat-bubble toggle
	_export_btn.tooltip_text = "Render this mask session to a video file (in the background)"
	_export_btn.custom_minimum_size = Vector2(40, 40)
	_export_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	# Same 40x40 box, same row (-28/-68), as assistant.gd's toggle, right of this one -
	# see that file's _TOGGLE_SIZE/_TOGGLE_ROW_BOTTOM doc for why the numbers match.
	_export_btn.offset_left = -112
	_export_btn.offset_top = -68
	_export_btn.offset_right = -72
	_export_btn.offset_bottom = -28
	_export_btn.pressed.connect(_on_export_pressed)
	add_child(_export_btn)
	# The SHARED export button steps aside - see Exporter.suppressed. Masking has
	# its own export (a headless relaunch against the session json rather than the
	# bake pipeline), and the shared one is on a higher CanvasLayer, so left alone
	# it covers this one and swallows the click while greyed out.
	var ch := _chrome()
	if ch != null and ch.exporter != null:
		ch.exporter.suppressed = true

	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dialog.use_native_dialog = true
	_dialog.title = "Export mask video"
	_dialog.filters = PackedStringArray(["*.mp4 ; Video (MP4, H.264 + AAC)"])
	_dialog.current_file = "ghost_mask.mp4"
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_dialog.current_dir = downloads
	_dialog.size = Vector2i(800, 560)
	_dialog.file_selected.connect(_on_export_path)
	add_child(_dialog)


func _on_export_pressed() -> void:
	_dialog.popup_centered()


func _on_export_path(out_path: String) -> void:
	_out = out_path if out_path.get_extension().to_lower() == "mp4" else out_path + ".mp4"
	_avi = _out.get_basename() + ".render.avi"
	if FileAccess.file_exists(_avi):
		DirAccess.remove_absolute(_avi)   # a stale scratch AVI from an interrupted prior export
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_session_path.get_base_dir()))
	session.save(ProjectSettings.globalize_path(_session_path))   # the relaunch reads THIS file
	# The relaunch must be TOLD what shape to record in, and it has to be told
	# before it boots - see _write_render_override.
	var rsz := _render_size()
	_write_render_override(rsz)
	var exe := OS.get_executable_path()
	var project := ProjectSettings.globalize_path("res://")
	var args := PackedStringArray([
		"--path", project, "--write-movie", _avi, "--fixed-fps", "25",
		"--", "--mask-render", _session_path])
	_render_pid = Subprocess.start(exe, args, "mask render")
	if _render_pid > 0:
		_render_state = "rendering"
		_set_status("⏺  Rendering %d×%d…" % [rsz.x, rsz.y])
	else:
		_clear_render_override()   # nothing booted; don't leave it lying for the next launch
		_set_status("⚠  Could not start the render process")


func _poll_render() -> void:
	match _render_state:
		"rendering":
			if Subprocess.alive(_render_pid):
				return
			# The render process has read its override at boot and is gone - take it
			# back out of the project root now, so a later launch (an F5 reload
			# included, see _reload_requested) comes up in the normal live window
			# mode and never inherits the render's offscreen resolution.
			_clear_render_override()
			if _file_size(_avi) > 65536:
				_repair_avi_sizes(_avi)   # Godot's 32-bit AVI sizes wrap past 4 GiB; fix before transcode
				_start_transcode()
			else:
				_set_status("⚠  Render produced no file (see console)")
				_render_state = "idle"
		"transcoding":
			if Subprocess.alive(_transcode_pid):
				return
			# Always clear the scratch AVI (the transcode's own `&& rm` usually already
			# did, but not if it failed or was interrupted) - never leave an intermediate
			# behind. remove_absolute is a harmless no-op when it's already gone.
			if _file_size(_out) > 4096:
				DirAccess.remove_absolute(_avi)
				_set_status("✓  Saved  " + _out)
			else:
				DirAccess.remove_absolute(_avi)
				_set_status("⚠  Transcode failed (see console)")
			_render_state = "idle"
	# A reload asked for while this export was in flight (see _reload_requested) is
	# held here until the export just went idle, then re-asked - re-checking the
	# assistant's own busy state fresh rather than assuming it's still the same.
	if _render_state == "idle" and _reload_after_export:
		_reload_after_export = false
		_reload_requested()


# --- the export's recorded resolution ------------------------------------------
# Movie Maker locks its output size to the project's viewport size at ENGINE
# STARTUP, before a single script in the relaunched process runs - so the only way
# to ask for anything other than project.godot's 1920x1080 is override.cfg, which
# Godot reads from the project root at boot. It is written here (in the editor, by
# the process that knows the clip) immediately before the render process is
# created, and removed the moment that process exits. Same mechanism, same
# reasoning, as exporter.gd's quality presets - see its _write_override.
#
# THE SIZE ASKED FOR IS THE SOURCE CLIP'S OWN, which is what makes a vertical clip
# export vertically: a 1080x1920 source records a 1080x1920 movie instead of being
# squeezed into a 16:9 frame. Before this, every mask export was 1920x1080 whatever
# went in - correct by coincidence for a 1080p landscape clip, an upscale for
# anything smaller, and a distortion for anything not 16:9.
#
# Capped on the long side because the fixed 1920x1080 window used to cap it there
# anyway: a 4K source should not silently turn into a 4K render (minutes per frame,
# tens of GB of scratch AVI) just because this now honours the source.
const _RENDER_MAX_SIDE := 1920


func _render_size() -> Vector2i:
	var w := float(maxi(2, _src_size.x))
	var h := float(maxi(2, _src_size.y))
	var s: float = minf(1.0, float(_RENDER_MAX_SIDE) / maxf(w, h))
	return Vector2i(_even(w * s), _even(h * s))


## Even, because yuv420p (the transcode's pixel format) cannot encode an odd
## dimension - a half-pixel would be rejected outright by x264.
static func _even(v: float) -> int:
	return maxi(2, int(round(v * 0.5)) * 2)


func _override_path() -> String:
	return ProjectSettings.globalize_path("res://override.cfg")


func _write_render_override(sz: Vector2i) -> void:
	var f := FileAccess.open(_override_path(), FileAccess.WRITE)
	if f == null:
		push_warning("ghost mask export: could not write override.cfg (render falls back to 1920x1080)")
		return
	# viewport_* is the RECORDED resolution; window_*_override shrinks the OS window
	# itself to an unobtrusive floater. In "viewport" stretch mode the two are
	# independent, so the movie is an offscreen buffer of exactly the clip's size
	# regardless of the monitor. The window keeps the clip's SHAPE (not a fixed
	# 480x270) purely so the little preview looks like what is being recorded. It
	# must stay a normal, drawable window - minimizing it makes Godot skip rendering
	# and the movie records frozen frames (see boot.gd).
	var s: float = 480.0 / float(maxi(2, maxi(sz.x, sz.y)))
	var win := Vector2i(_even(float(sz.x) * s), _even(float(sz.y) * s))
	f.store_string("[display]\n\nwindow/size/viewport_width=%d\nwindow/size/viewport_height=%d\nwindow/size/window_width_override=%d\nwindow/size/window_height_override=%d\nwindow/stretch/mode=\"viewport\"\n"
		% [sz.x, sz.y, win.x, win.y])
	f.close()


## Remove it. Called when our render exits, and once at editor startup to clear a
## stale copy left behind by a render that was killed or crashed - left in place,
## it would boot the LIVE editor into the render's offscreen resolution.
func _clear_render_override() -> void:
	var path := _override_path()
	if FileAccess.file_exists(path):
		DirAccess.remove_absolute(path)


func _start_transcode() -> void:
	_set_status("⏳  Finalizing…")
	# `-fflags +genpts` re-derives timestamps so a damaged/wrapped AVI index (see
	# _repair_avi_sizes) is bypassed instead of trusted - without it a >4 GiB render
	# transcodes to a broken file or fails outright, leaving the raw .render.avi behind.
	# `-pix_fmt yuv420p` keeps the MP4 playable everywhere (VLC/QuickTime/browsers).
	# Run through bash so the scratch AVI is deleted BY THE TRANSCODE ITSELF the moment
	# it succeeds (`&& rm`), not by a _poll_render tick that never comes if the editor is
	# closed while ffmpeg (a child that outlives it) is still finalizing - which is how
	# the orphaned .render.avi got left "alongside the final version". Paths are passed as
	# $1/$2, never interpolated, so spaces/quotes in the export path are safe.
	var script := "ffmpeg -y -loglevel error -fflags +genpts -i \"$1\" " \
		+ "-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k \"$2\" " \
		+ "&& rm -f \"$1\""
	_transcode_pid = Subprocess.start("/bin/bash", PackedStringArray(["-c", script, "bash", _avi, _out]), "mask transcode")
	_render_state = "transcoding"


# Godot's AVI writer keeps 32-bit RIFF/LIST size fields, and a full-resolution mask
# render crosses 4 GiB in a few minutes - past that the written sizes WRAP (mod 2^32)
# and the container lies about where the frame data ends, even though every 00db/01wb
# chunk after it is written correctly to EOF. Demuxers that trust those fields (and
# players, whose seeks hit the equally-wrapped idx1 offsets) stall, repeat frames, or
# break time indexing (this is why VLC's scrub bar goes wrong on the raw AVI). The
# repair is two words: RIFF size and the movi LIST size become 0 - "size unknown, read
# to end of file" - turning any demux into a clean sequential walk of the intact chunks.
# No-op for files under 4 GiB (their sizes are already correct). Mirrors exporter.gd.
func _repair_avi_sizes(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.READ_WRITE)
	if f == null:
		return
	if f.get_length() < 4294967296:
		f.close()
		return
	f.seek(4)
	f.store_32(0)                       # RIFF size -> unknown
	var pos := 12
	for i in 64:                        # walk top-level chunks to the movi LIST
		f.seek(pos)
		var tag := f.get_buffer(4).get_string_from_ascii()
		var csize := f.get_32()
		if tag == "LIST" and f.get_buffer(4).get_string_from_ascii() == "movi":
			f.seek(pos + 4)
			f.store_32(0)               # movi size -> unknown
			print("ghost mask export: repaired wrapped >4GiB AVI sizes in ", path.get_file())
			break
		if csize <= 0 or tag.is_empty():
			break
		pos += 8 + csize + (csize & 1)
	f.close()


func _file_size(path: String) -> int:
	if not FileAccess.file_exists(path):
		return 0
	var fa := FileAccess.open(path, FileAccess.READ)
	return fa.get_length() if fa != null else 0


func _set_status(text: String) -> void:
	if _status == null:     # render_mode never builds it - status there is stdout only
		print("ghost mask: ", text)
		return
	_status.text = text
	_status.visible = true
