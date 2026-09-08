extends Vehicle
class_name ComicVehicle

## ComicVehicle - the show as an open comic book, flown over by a real perspective camera.
##
## The same scenes, the same Director, the same cutting. What changes is where a scene
## LANDS: instead of replacing the picture, each cut fills the next PANEL of the spread, and
## when the spread is full the next cut turns the leaf. So `Scene hold` and `Flourishes`
## keep their exact meanings - a burst of quick cuts becomes a run of small panels filling in
## fast, which is what a comic does with a fight.
##
## TWO PAGES AT ONCE, because that is what a comic is. The first cut of this drew ONE
## portrait page and it cost three things at the same time. A portrait sheet inside a
## landscape frame cannot cover it, so there was always a wedge of desk in shot unless the
## camera pushed in until the page stopped reading as a page. A pan had one page's width of
## content to travel before it ran onto the trim edge. And a page turn had no hinge that
## meant anything, because a leaf hinges on a SPINE and one page does not have one. See
## [ComicSpread]: the spread is 2 pages wide by ~1.5 tall, which is 1.33 against the frame's
## 1.78, so the sheet can cover the whole frame at a shot that still shows several panels.
##
## ONE LIVE PANEL, THE REST HELD. The Director runs exactly one scene (two across a
## transition) and this keeps that: the newest panel is live in its own SubViewport, and
## every panel behind it is a STOPPED render target - update mode DISABLED, its scene
## freed, its last frame still sitting in VRAM. Measured in tests/vehicle_probe.gd: the
## texture survives both the stop and the free, bit-exact, so a held panel costs nothing
## at all. That is the only reason this can ship. Six live scenes is not a thing that
## runs - the stage governor already spends its budget on one - and the alternative,
## capturing each panel with get_image(), is the synchronous readback stall that cost
## Masking its frame rate.
##
## It is also simply correct. A comic panel IS a held moment; the page is a sequence of
## them. The frame is never static anyway, because the camera and the page are always
## moving.
##
## THE SPREAD IS REALLY IN 3D. Not a sheared 2D plane faking depth - two quads placed in a
## world and projected through a [Lens3D], free to rotate on X, Y and Z at once. Panels
## are drawn as SUBDIVIDED grids of textured triangles whose vertices are each projected
## individually, because a two-triangle quad is affinely textured and warps: measured at
## a hard yaw, the seam of a two-tone test texture landed 28.5 px away from where
## perspective puts it, on a 512 px frame.

## How many panel slots exist. Two POOLS of [constant ComicSpread.MAX_PANELS], alternating
## per spread, because a page turn shows both spreads at once - reusing one pool would pull
## the outgoing spread's pictures out from under it mid-turn. All FOUR half-pages carry
## textures during a turn (the old left page under the leaf, the old right page on the
## leaf's front, the new left page on its back, the new right page revealed beneath), which
## is why a pool is a whole spread and not a page.
const POOL := ComicSpread.MAX_PANELS

## Panel render-target size, as a fraction of the stage's shorter side. A panel is sampled
## at close to 1:1 in the tight reading shot (which frames roughly one panel), so this is
## about right and not a guess; it is clamped below so a small window still gets a legible
## panel and above so a 4K export does not allocate enormous targets.
const PANEL_SCALE := 1.0
const PANEL_MIN := 224
const PANEL_MAX := 1400
## What an unopened slot is sized to. There are 24 of them now and a spread opens at most
## half; allocating every one at PANEL_SCALE up front - which is what the first cut did -
## reserved a full-size render target for a panel no page had. Sixty-four squared is
## 16 KiB, and _open_slot grows the slot to its real size before anything draws it.
const SLOT_IDLE := 64

## Subdivision of a panel's textured grid, chosen so each cell is about this many pixels on
## screen, between these bounds.
##
## ADAPTIVE, because a fixed number cannot be right. Inside one cell the texture mapping is
## affine while the true mapping is projective, so the error scales with how big the cell is
## ON SCREEN and with how hard the panel is raked. A fixed 8 was measured sufficient at one
## yaw on a 512 px frame, and this vehicle now puts a single panel across a whole 1280 px
## frame at a much harder rake. Sizing the CELL rather than the count bounds the error
## directly, at any framing.
##
## NOT the fix for the banding reported on fractal_zoom - that was checked, and going from 8
## to an adaptive 6-40 changed those frames not at all. Those bands are the scene's own
## float32 precision blocks, which the close framing here simply magnifies. This is a
## correctness change made because the reasoning behind the old constant no longer held, not
## a repair of anything observed.
const CELL_PX := 56.0
const GRID_MIN := 6
const GRID_MAX := 40
## Subdivision of the paper quad. It carries only a slow gradient, so it needs far less.
const PAPER_GRID := 4
## Segments per rounded corner.
const CORNER_SEGS := 5

## ONE PAGE is this many units wide in the world, so the spread is twice it and is centred
## on the origin with the spine on the world Y axis. Deliberately unchanged from the
## single-page vehicle: a panel therefore has exactly the world size it always had, every
## framing constant below keeps the meaning it was tuned with, and the only thing that grew
## is the sheet.
const PAGE_W := 2.0

## HOW MANY PANELS MAY RUN AT ONCE. See [method _update_liveness]: everything on screen
## moves, and the budget is what keeps that affordable. Three covers the panel being read
## plus the neighbours crowding the frame edges at the reading distance. NOT raised for the
## spread even though a wider shot puts more panels in frame - the shot is wider, so each
## panel is smaller, and the budget is about scene cost rather than about screen area.
const LIVE_MAX := POOL

## ...AND THE BUDGET DOES NOT APPLY TO AN EXPORT. Every panel runs.
##
## LIVE_MAX is a REAL-TIME concession: the show has sixteen milliseconds to draw a frame, and a
## dozen simultaneous scenes do not fit in them. An export has no such deadline - it renders
## offline, one frame at a time, and a frame that takes a second costs a second of somebody's
## afternoon rather than a stutter. Spending the budget there buys nothing and costs the thing
## the vehicle is for: "you're optimizing for real-time, but we are cutting videos. So we can
## just play every scene, and we should. Freezing them looks terrible."
##
## Quite right, and it is also why the frozen panels read so much worse than they used to. On a
## single page of three to six the held ones were mostly off shot; a spread of nine to twelve
## seen wide puts most of the page in frame at once, so the budget's leftovers are no longer at
## the edges - they are the picture.
const LIVE_ALL_ON_EXPORT := true

## How often a live panel that is NOT the one being read repaints, in frames.
##
## Everything visible moves - but the panel being READ is the one anyone is looking at, and
## a background moment does not need the same temporal resolution as the foreground. Every
## other frame is invisible on a panel at the edge of shot, because those panels are ticked
## with a matching multiple of the step, so they run at the RIGHT SPEED with half the
## samples rather than in slow motion.
##
## Measured on one seed against the same scenes: it and the height-based target sizing took
## the comic's active frame from 84 ms to 73 against the full frame's 41. So three live
## panels cost about 1.8x one, not 3x - the rest of the gap is the stage governor's to
## absorb, which is what it is for.
const OFF_FOCAL_PERIOD := 2
## How many render-target pixels a panel gets per pixel it covers on screen. Above 1 so a panel
## the camera is drifting toward is already sharp before it is the subject, and so the paper's
## own perspective foreshortening does not read as a soft panel.
const PANEL_OVERSAMPLE := 1.35
## How much bigger a panel has to want to be before its target is re-opened, as a ratio. A
## target that chases the shot pixel for pixel reallocates every frame; this makes it a handful
## of steps across a whole push-in.
const REOPEN_STEP := 1.5

## HOW MANY FRAMES A PANEL MUST ACTUALLY RENDER before it is allowed to be frozen.
##
## THIS IS WHAT "4 OF THE 6 FRAMES ARE PURE BLACK, AND EMPTY" WAS, and the freeze mechanism
## was working perfectly the whole time. A held panel keeps the last picture it DREW - that is
## the measured invariant the whole vehicle rests on (tests/vehicle_probe.gd) - and a panel
## that was cast, given two frames, and frozen has a last picture that is two frames of an
## empty scene. Which is black. Nothing culled it; it was preserved exactly as found.
##
## It only became visible with the spread. A page of three to six panels, read close, kept
## nearly all of them inside the liveness budget long enough to fill in. A spread of nine to
## twelve with the same budget of three, and a reading plan that never visits most of them,
## means a panel can be cast, glimpsed and frozen having drawn nothing at all - including the
## film panel, whose player is still opening its window in those first frames. Reported as
## "the video with the girl WAS playing, but now it's black. It's like the frame got culled."
##
## So liveness is now ordered by NEED first and distance second: a panel below its warm-up
## takes priority over one that has already drawn its moment, and only once every panel on the
## sheet has a picture does the budget go back to following the camera. Twenty-four frames is
## about four tenths of a second - enough for a scene to lay down its first pass and for a
## video window to hand over its first decoded frame.
const WARM_FRAMES := 24

## HOW MANY FRAMES A PANEL MUST HAVE RENDERED BEFORE ITS TEXTURE MAY BE DRAWN.
##
## A SubViewport's render target is allocated by _open_slot and is not cleared - it holds
## whatever was in that piece of VRAM. Sampling it before the viewport has produced a frame
## therefore draws garbage, and the garbage on a freshly booted machine is the last thing the
## compositor had there: the application's own interface. Reported as "at first load, every
## single frame starts by showing the UI itself: all of the toggles, all of the options",
## replaced a moment later by the real scene - which is exactly what "one frame early" looks
## like.
##
## TWO, not one. _update_liveness raises the warm count in the same frame it resumes the
## viewport, and the viewport renders at the END of that frame, so a count of one means "asked
## to draw", not "has drawn". A panel below this is drawn as an empty ruled frame - which is
## already what the page does for a panel the story has not reached, so it costs nothing and
## looks like nothing.
const FIRST_DRAW := 2

## HOW MANY PANELS OF A SPREAD THE CAMERA ACTUALLY STOPS ON, before the leaf turns.
##
## NOT ALL OF THEM, and that is the correction. The reading used to walk every panel in order
## and turn only when it ran out - so a nine-panel spread was nine cuts, two and a half
## minutes of one sheet, and by the end the camera was hunting for somewhere it had not
## already been. Reported as "it feels like the camera tries to focus on EVERY frame before it
## transitions to the next spread. You wouldn't really expect that, since a lot of those
## frames are already shown during transitions, or during zoom-out."
##
## That is exactly right, and it is how a page is read: you take in the whole spread at a
## glance, rest on a few moments, and turn. The panels that are not rested on are not missing -
## they are drawn the entire time, and every pan across the sheet passes over them. So the
## spread now samples a PLAN of this many panels to settle on, in reading order, and turns
## when the plan is done. A denser spread gets a slightly longer plan, but never one per panel.
const FOCUS := Vector2i(3, 5)

## HOW A SPREAD GETS READ. Weighted, and the weights are the whole statement: mostly the way a
## page is read, sometimes not.
##
##   read  top-left, across, down and back, across, then the other page. The default, and it
##         should be, because it is what the eye does with a page it is actually reading.
##   grab  something catches the eye FIRST - the film panel if there is one, else the biggest
##         panel on the sheet - and the reading resumes in order afterwards. "There are times
##         when you would turn a page, and something on the RIGHT would catch your attention
##         first."
##   skim  left, right, left - a glance across the spread rather than a read of it. "There are
##         times when you may not read the pages at all."
##
## A HARD RULE WOULD BE WRONG, which is why this is a bag and not an if. Strict reading order
## every time is a machine indexing a page; the deviations are what make it read as someone
## looking at a comic. But they are deviations - six to one against - because the report was
## that the ping-ponging is the exception that had become the rule.
const PLAN_STYLES := {"read": 6.0, "grab": 2.0, "skim": 1.0}

## HOW STRONGLY A PIECE OF FOOTAGE PULLS THE EYE FIRST, as a probability that it opens the
## spread whatever style was rolled.
##
## "For most humans, an 'actual' human in a video is the first thing they want to look at,
## every time." That is right, and it is the one place this vehicle should not be even-handed:
## every other panel is an abstract field, and a face is not. Not 1.0 only because a spread
## that ALWAYS opens on the film is a rule the eye learns in two pages.
const FILM_FIRST := 0.72
## HOW MUCH SOONER A SHOT ON THE FILM PANEL ARRIVES, as a fraction of the deadline the rest of
## the vocabulary gets (see ARRIVE).
##
## "The camera never holds long enough to look at anything. I feel like we would want a bias to
## at least hold the camera to look at her for a few seconds." The settle mechanism is what
## does the holding, and the deadline is what decides when it starts - so the way to hold
## longer on a face is not to slow the move down, it is to ARRIVE EARLIER and spend the
## remainder settled. At 0.35 of the expected hold the travel is over in about a third of the
## scene and the other two thirds are a shot of her, still.
const FILM_ARRIVE := 0.35

## HOW MUCH OF THE FRAME A CONTAINED PANEL MAY FILL, on its LONGER axis.
##
## Every other panel is framed on HEIGHT alone and allowed to overflow the sides - that is the
## whole difference between a comic read close and a slideshow of pages, and it is right for an
## abstract field, where any part of it is as good as any other part.
##
## A PIECE OF FOOTAGE IS NOT AN ABSTRACT FIELD. "It can look very strange to zoom in to a
## specific corner of the embedded video, where nothing is happening in it... a video with a
## girl speaking in it, you really don't want to focus the camera on her door to the side."
## Exactly so: a 1.9-aspect film panel framed to span the frame's height is nearly twice as
## wide as the screen, so what is on show is a third of the picture and the subject is as
## likely to be off it as on it. A contained panel is fitted on BOTH axes, so the whole video
## is in shot, whole, every time it is the one being read.
##
## JUST UNDER 1, not comfortably under it. Every notch of margin here is bought by pulling the
## camera further off the sheet, and past the covering distance that is paid for in desk: at
## 0.88 the video was whole with a pleasant border and the page covered only 0.63 of the frame,
## which is a wedge of nothing across a third of the picture. At 0.98 the video still arrives
## whole - it simply reaches the edges of the shot instead of floating inside them.
const CONTAIN_FILL := 0.98
## ...and however much the video wants revealing, the camera stops this far past the distance
## at which the sheet still covers the frame. A face cropped out of shot is worse than a sliver
## of desk; a face floating in a sea of desk is worse than both, and without a bound that is
## where an extreme panel aspect would put it.
##
## A SLIVER, and the first value was not one. At 1.22 the camera could sit 22% past the
## covering distance, and measured on a real session that produced page coverage of 0.67 -
## a third of the picture was the surface the book is lying on, which is the defect the whole
## spread exists to remove, arriving through the one door left open for it. Six percent is
## enough to rescue the ordinary conflict and not enough to be seen; past that the panel is
## simply framed as tightly as covering allows and takes the crop.
## ONE, and the argument for anything above it was wrong. Going past the covering distance is
## the ONLY way desk enters a shot, and measured on a real spread six percent past cost ten
## percent of the coverage - the relationship is far steeper than it looks. Worse, it did not
## even buy the thing it was spent on: containing that panel wanted 7.76 world units against a
## cover of 4.62, so a cap of 4.89 left the video cropped AND put desk in the corner, which is
## both prices for neither benefit. The camera now stops where covering stops, and the video is
## contained as far as that allows - see FILM_FLATTEN, which is what makes that far enough.
const CONTAIN_DESK := 1.0
## HOW FAR TOWARD SQUARE-ON A CONTAINED PANEL PULLS THE SHEET, 0..1. Harder than
## FIELD_FLATTEN, and for a stronger version of the same reason: a rake foreshortens the sheet,
## so a raked page needs the camera NEARER to keep paper across the frame - which is exactly
## the constraint that fights containing a wide panel. Measured on a rolled spread at a
## moderate rake: containing the widest panel wanted 3.03 world units while the sheet stopped
## covering the frame past 2.13. Flattening buys most of that back, and it is also just the
## right shot: a face is the most recognisable thing this vehicle ever puts on the page, and a
## hard angle across a face reads as damage rather than as depth.
## LOWERED FROM 0.78, and the first value had the sign of its own effect backwards. Flattening
## turns the sheet square-on, which presents the panel at its FULL area - so the harder it
## flattens, the further the camera must retreat to fit the whole panel in, and containment
## becomes unaffordable inside the covering budget (measured: a fit of 7.76 against a cover of
## 4.62 at 83 degrees). A gentler flatten leaves the panel foreshortened, which is smaller,
## which fits nearer. The face is still turned toward the camera - that was the point - it is
## simply not pressed flat against the glass.
const FILM_FLATTEN := 0.40

## SPREAD ATTITUDE, in radians - the rest pose the book is rolled into, on all three axes.
## Generous on purpose: the point of putting the paper in a real 3D world is that it can be
## TURNED, and at the reading distance a raked page is what reveals it has depth at all. A
## few degrees reads as a crooked scan rather than as a camera angle.
##
## YAW IS TIGHTER THAN IT WAS (it went to 0.85). A spread is twice as wide as the page these
## were tuned on, so the same yaw that raked one page pleasantly foreshortens the far page of
## a spread into a sliver and throws its far corners behind the eye.
##
## THE SHEET'S ATTITUDE DOES NOT COST COVERAGE, which is worth writing down because it looks
## as though it should. Swept one axis at a time against page_coverage() on a real spread:
## pitch, yaw and roll each held 1.00 across their WHOLE range, while camera elevation ran
## 0.34 / 0.84 / 1.00 / 1.00 / 1.00 / 1.00 from 30 to 80 degrees. Rake the paper as hard as you
## like; it is how low the CAMERA sits that decides whether the picture is all paper. See
## EL_MIN_DEG.
const PITCH := 0.55
const YAW := 0.44
const ROLL := 0.40
## ...and it never stops moving. Radians per second of slow continuous drift, so the sheet
## turns under the camera through the whole spread rather than settling into a pose.
const DRIFT := 0.035

## CAMERA ELEVATION off the paper, in degrees. Never square-on (at 90 the perspective
## camera and the page agree and the panel is a flat rectangle again) and never grazing
## (the panel foreshortens into a line).
##
## THE FLOOR CAME UP, from 32, and this is the one number that decides whether there is desk
## in the picture. Measured by sweeping each axis alone against page_coverage() on a real
## spread: the sheet's own pitch, yaw and roll each held 1.00 across their entire range, and
## elevation ran 0.34 at 30 degrees, 0.84 at 40, and 1.00 from 50 up. The reason is simply
## foreshortening - a low camera sees the sheet edge-on and it shrinks along one axis faster
## than any distance can compensate for, so no framing solve can put paper across the frame.
##
## Everything else was tried first and did nothing, because everything else was the wrong
## axis. A spread at 0.90 coverage for every shot it had was reported as a wedge of desk, and
## the page's roll was reduced on the assumption that a rotated rectangle was cutting the
## corners. It changed the number not at all.
const EL_MIN_DEG := 46.0
const EL_MAX_DEG := 72.0
## Camera roll about the view axis - the Dutch angle, on top of the page's own roll.
const CAM_ROLL := 0.22

## HOW FAR ONE SHOT MAY DEPART FROM THE SPREAD'S SET-UP. This is the fix for "it just bounces
## all over the place... a camera ping-ponging around at random angles".
##
## Every shot used to re-roll its azimuth over the whole circle, its elevation over the
## whole range, its roll and its focal length - independently, at every reading advance. So
## consecutive shots were unrelated positions, and the interpolation between them was a
## camera swinging across the page for no reason. Smoothing it does not help: the problem
## is not the transition, it is that there is nothing in common between where it was and
## where it is going.
##
## So a spread picks ONE set-up - a side, a height, a tilt, a lens - and a shot varies inside
## it. That is coverage rather than chaos, and staying on one side of the subject is the
## oldest rule in the grammar (the 180-degree line). Changing sides is now a PAGE turn,
## which is the one moment where a new set-up reads as a new scene rather than as a mistake.
const AZ_ARC := 0.42              # +/- radians of orbit around the spread's side (~24 deg)
const EL_VARY_DEG := 7.0          # +/- degrees of height off the spread's own
const ROLL_VARY := 0.05           # +/- radians of Dutch on top of the spread's own

## HOW MUCH OF ITS ARC ONE SHOT MAY SPEND, as a fraction. This is the fix for "the camera can
## rotate/turn A LOT every time it shifts from one frame to another".
##
## The angles used to be an INDEPENDENT DRAW per cut: every reading advance re-rolled the
## azimuth anywhere in +/-AZ_ARC, so two consecutive shots could differ by the whole 48-degree
## span, and at a cut every few seconds that is a camera swinging back and forth about a mean.
## Staying inside the arc was never the problem - the problem was arriving anywhere in it in
## one step.
##
## So the angles are a bounded RANDOM WALK instead. A shot moves them by at most this fraction
## of the arc and the walk is clamped to it, so the same total excursion is still reachable -
## it is just spread over several shots, which is what "rotates more gently, over multiple
## per-frame pans" asks for. At a third of the arc it takes three or four cuts to cross, and
## the walk keeps its direction long enough to read as one continuous move rather than as
## jitter.
const ANG_STEP := 0.34
## The lens, chosen once per spread. Narrower than it was (it went to 64): a wide lens this
## close to a panel bends its edges, which is the "unfocused, distorted" half of the report.
const FOV := Vector2(38.0, 54.0)

## HOW TALL THE PANEL BEING READ IS, in frames. 1.0 spans the frame exactly top to bottom;
## above that it overflows and the screen crops into it.
##
## THE FLOOR CAME DOWN, from 1.00 to 0.78, and that is the spread paying for itself. The old
## floor existed because a shot looser than "the panel IS the picture" put a portrait page
## inside a landscape frame and filled the rest with desk - so a wide shot was not a shot,
## it was a framing failure. A spread covers the frame at a much looser shot, so a panel at
## 0.78 frames tall now sits inside a picture that is entirely paper and neighbouring
## panels. That is a comic being READ rather than a video with a border, and it is the shot
## the vocabulary was missing.
const FILL := Vector2(0.78, 1.30)
## The hard near limit, in frames of panel height - however much the spread wants covering,
## the camera stops here. Past about this the panel stops reading as a panel.
##
## SET GENEROUSLY, and the first value was not. At 2.2 this floor came out FURTHER from the
## page than the covering distance did (measured 1.66 against 1.24) and so it, not the
## framing, decided every shot - vetoing exactly the push-in that fills the frame and
## leaving the strip of desk down the side that this whole pass exists to remove. A floor
## that binds in the ordinary case is not a floor, it is the rule.
const CROP_MAX := 4.5
## How far the aim is pulled from the panel's centre toward the middle of ITS OWN PAGE. A
## panel at the outer trim, framed dead on, puts the trim edge - and the void behind the
## paper - into shot; pulling the aim in keeps the sheet under the panel at every panel.
##
## TOWARD THE PAGE, NOT THE SPREAD, and that distinction only exists because of the spread.
## The middle of a spread IS THE SPINE - the widest gutter on the sheet - so pulling every
## aim toward it biases every shot toward the one place with no picture in it, which is the
## defect this pass exists to remove arriving by the other door. Pulling toward the page's
## own centre keeps the original purpose exactly: the risk was always the OUTER trim, and
## the inner edge has a whole second page beyond it.
const AIM_PULL := 0.12

## Pan rates (ease per second) for moving between panels: a slow drift, an ordinary walk,
## and a whip. A comic read at one speed is a slideshow on rails.
const PAN_SLOW := 0.9
const PAN_WALK := 2.2
const PAN_WHIP := 7.5
## The spread's own attitude eases at its own, much slower rate - it is scenery, not a move.
const PAGE_EASE := 0.7
## A page turn has somewhere to be.
const TURN_EASE := 3.0
## How long a page turn takes, in seconds.
const TURN_TIME := 1.15
## How far the leaf swings. A HALF TURN exactly, because at pi the leaf lies flat on the
## left page and its back face - the incoming left page, see [method ComicSpread.mirror] -
## lands precisely where that page is about to be drawn flat. Anything short of pi would
## have to cross-dissolve the difference.
const TURN_ARC := PI

## THE SPREAD LIES ON SOMETHING. Without it the frame past the trim edge is pure black,
## which at a hard rake is a wedge of nothing across the shot - and black is not a
## background, it is the absence of one.
##
## PAINTED IN SCREEN SPACE, not as a world quad. A surface big enough to never run out
## under a close raked camera has corners far outside the view frustum, and the projection
## has to drop any quad with a corner behind the eye (it inverts through infinity) - so the
## world-quad version simply never drew, at any of the sizes worth having. A defocused
## surface has no texture to rake anyway; what it needs to be is present and not black.
## How dark the wash goes at the corners of the frame, as a fraction of the desk colour.
const DESK_VIGNETTE := 0.55
## The sheet's shadow on it: offset in page widths, and how far off the paper it is cast.
const SHADOW_OFF := Vector2(0.045, 0.055)
const SHADOW_DEPTH := 0.06

# --- state -------------------------------------------------------------------
var _slots: Array = []            # POOL * 2 SubViewports; spread s uses pool (s % 2)
var _pool := 0                    # which pool the CURRENT spread is drawing from
var _spread: ComicSpread = null
var _spread_i := -1
var _cast: Array = []             # per panel of the current spread: its GhostScene, or null
var _read := 0                    # panel the camera is reading, and the Director's current
## The panels this spread will settle on, in reading order - see FOCUS. `_read` is one of
## these, and `_step` is how far through the plan the reading has got.
var _plan: Array = []
var _step := 0
var _to_cast: Array = []          # panels still waiting to be cast, one per frame
var _prev: ComicSpread = null     # the outgoing spread, alive only through a turn
var _turn_t := -1.0               # seconds into a page turn, < 0 when none

## WHICH PANEL OF THIS SPREAD HOLDS FOOTAGE, or -1 for none, and the clip it holds.
##
## AT MOST ONE PER SPREAD, which is the whole answer to a question that would otherwise need
## a mechanism. A clip's position is a pure function of the show clock (see
## [method Films.position_at]), so two panels sampling the same clip at the same instant
## would show the same picture twice - and the spread is what you see at once, so one per
## spread IS one at a time. During a turn the outgoing spread's panels are already stopped,
## so even then only one is playing.
var _film_at := -1
var _film_clip: Dictionary = {}
var _film_prev := ""              # last spread's clip path, so two running do not repeat

var _lens := Lens3D.new()
## The two solver lenses, allocated ONCE. They used to be built inside _fit and _cover,
## which is three RefCounted allocations per frame inside a 22-iteration bisection - see
## AUDIT.md. Neither survives the call, so neither needed to be new.
var _fit_lens := Lens3D.new()
var _cover_lens := Lens3D.new()
var _mod: ModBank = null
var _bookend := 1.0
var _panel_px := 512
var _stage_size := Vector2(1280, 720)
var _live: Array = []             # panel indices whose viewport is running this frame
## Frames each panel of the current spread has actually rendered. See WARM_FRAMES.
var _warm: Array = []
var _frame := 0                   # for the off-focal repaint phase, see OFF_FOCAL_PERIOD

# THE SPREAD'S ATTITUDE, eased toward a target that itself DRIFTS. Sampled on all three axes
# with real magnitude: a book that only pitches a few degrees reads as a slightly crooked
# scan, and the whole reason to put the sheet in a real 3D world is that it can be turned.
var _att := Vector3.ZERO
var _att_target := Vector3.ZERO
var _att_rate := Vector3.ZERO     # slow continuous drift, per second
## The attitude as a BASIS, rebuilt once per advance() and read by everything that places a
## point on the paper. It used to be built inside _world (then _page_point) - six
## trigonometric evaluations and a 3x3 construction, per vertex, roughly 2500 times a frame
## with the identical argument, and AUDIT.md attributes most of the page's ~18 ms of _draw
## to it. Hoisting it is exact: the value was always the same.
var _att_basis := Basis.IDENTITY

# THE CAMERA STATE, expressed in SPREAD-LOCAL SPHERICAL rather than as a world position: an
# azimuth around the aim and an elevation off the paper. That is what makes the rake real -
# an eye on the sheet's normal sees a flat rectangle however the sheet is rotated - and it is
# also what makes travelling between panels a move PARALLEL TO THE PAPER, because the
# direction is held and only the point it looks at changes.
# Keys: aim (spread coords), az, el, roll, fill, fov. See _station.
## The spread's SET-UP: the side the camera is on, its height, its tilt and its lens. Fixed
## for the spread; see AZ_ARC.
var _az_base := 0.0
var _el_base := 1.0
var _roll_base := 0.0
var _lens_fov := 46.0
## The bounded walk each angle is currently at, as an offset from its base. See ANG_STEP.
var _az_walk := 0.0
var _el_walk := 0.0
var _roll_walk := 0.0

var _cam := {"aim": Vector2(1.0, 0.7), "az": 0.0, "el": 1.0, "roll": 0.0,
	"fill": 1.1, "fov": 52.0}
<<<<<<< Updated upstream
## WHERE THE CAMERA IS GOING. `_cam` follows this, every frame, and nothing else moves the
## camera - see the note at the head of the camera section.
var _tgt := {}
## The shot in progress: {panel, kind, rate, arrived, heading}. A shot is a SUBJECT and a way
## of looking at it, not a path; where the camera is on the way there is the follower's
## business.
var _shot := {}
var _shot_t := 0.0
## Seconds since the shot arrived, and the aim it arrived at - the origin the held shot creeps
## from. Measuring the creep from the ARRIVAL rather than from the last frame is what bounds
## it however long the hold runs.
var _hold_t := 0.0
var _hold_from := Vector2.ZERO
## Shots begun this session, part of the seed so each is reproducible without repeating.
var _shot_n := 0
## Which way the framing is currently going: +1 tightening, -1 opening, 0 neither. Cleared
## when a shot ARRIVES, which is what makes "no reversal without a hold between" expressible.
var _zoom_dir := 0
## Set by a declared discontinuity, and the only thing that lets the placed rake and distance
## skip their followers. See _cut_to.
var _snap := true
## The followed halves of _place_eye: the rake pulled toward square-on, and the eye distance.
## Both are derived from quantities that STEP - see RATE_FLAT.
var _flat_s := 0.0
var _dist_s := -1.0
=======
## The move being travelled: {kind, a, b, dur, ease}. See MOVES.
var _mv := {}
var _mv_t := 0.0
## How many moves have been CHAINED since the last cut - see [method _chain_move]. It is
## part of the seed, so a chain is reproducible without being a repeat of the move it
## followed.
var _chain := 0
## The panel the CHAIN is heading for, or -1 when it has not chosen yet. Decided ONCE per cut
## and then held - see _chain_move.
var _chain_to := -1
## How many CHAINED JOURNEYS this cut has spent, and how many it rolled. See CHAIN_TRAVEL.
var _travel := 0
var _travel_max := 1
## Which way the framing is currently going: +1 pushing in, -1 pulling out, 0 neither. A chain
## may continue a zoom but never reverse one - see _plan_move.
var _zoom_dir := 0
## THE SMOOTHED HALF OF THE RIG. `_place_eye` computes a TARGET rake and a TARGET distance
## every frame from quantities that step: which panel the aim is inside, whether that panel
## holds the footage, and what the panel being READ is framed as. These follow toward it.
## Negative distance means "nothing yet" - the first placement snaps. See POSE_EASE.
var _flat_s := 0.0
var _dist_s := -1.0
## Set by a DECLARED discontinuity, and the only thing that lets the pose skip the easing.
var _snap := true
## WHERE THE HELD SHOT ARRIVED and which way it creeps from there - both decided once, when
## the settle is planned, and then spent over however many breathes the hold has room for.
## See BREATHE_AIM for what a per-breathe reroll of this looked like.
var _hold_aim := Vector2.ZERO
var _hold_dir := Vector2.RIGHT
## The chain count the hold began at, so the creep is measured from the ARRIVAL rather than
## from however many chains the shot happened to spend getting here.
var _hold_chain := 0
>>>>>>> Stashed changes
var _dip_t := -1.0                # seconds into a dip to black, < 0 when none
var _roll := 0.0                  # the roll actually in force (field-flattened)
var _eye := Vector3(0, 0, 3.0)
var _look := Vector3.ZERO
var _fov := 52.0

# paper, sampled per session
var _ink_weight := 0.0038
var _desk := Color(0.06, 0.055, 0.07)
var _paper := Color(0.92, 0.90, 0.855)
var _ink := Color(0.07, 0.065, 0.075)
var _rng := RandomNumberGenerator.new()


func mount(st: SubViewport) -> void:
	super.mount(st)
	# ...and belt to the half-texel inset's braces: never let a UV wrap even if one
	# escapes the inset. The default is already this; saying so is cheap and the failure
	# it prevents (the far edge of a panel bleeding in along the near one) is not obvious
	# to diagnose.
	texture_repeat = CanvasItem.TEXTURE_REPEAT_DISABLED
	_size_targets(Vector2(st.size))
	_build_slots()


## Everything sampled, rolled here rather than in mount() because the session seed is not
## resolved until the Director attaches (see Vehicle.begin_session).
func begin_session() -> void:
	_rng.seed = Director.session_seed() ^ 0x0C031C
	# Paper is never pure white - a white page reads as a blank canvas rather than as
	# printed stock - and it is sampled, like everything else here.
	_paper = Color.from_hsv(_rng.randf_range(0.055, 0.11),
		_rng.randf_range(0.05, 0.13), _rng.randf_range(0.88, 0.95))
	# Ink is not black either. A cool near-black prints like ink; pure #000 prints like a
	# UI border.
	_ink = Color.from_hsv(_rng.randf_range(0.58, 0.72), _rng.randf_range(0.05, 0.20),
		_rng.randf_range(0.05, 0.13))
	# The surface the sheet lies on: dark enough that the paper is still the brightest thing
	# in the frame by a long way, but never black.
	_desk = Color.from_hsv(_rng.randf_range(0.02, 0.12), _rng.randf_range(0.10, 0.32),
		_rng.randf_range(0.055, 0.115))
	_ink_weight = _rng.randf_range(0.0030, 0.0052)
	_mod = ModBank.new(Director.session_seed() ^ 0x9A6E)
	_reset_book()


func release() -> void:
	# The vehicle OUTLIVES a session (main owns it; the synthesis modes re-attach per
	# take), so release resets the BOOK rather than tearing the pool down - rebuilding
	# twenty-four render targets on every settings change is exactly the reallocation churn
	# main's governor warns about.
	_reset_book()


func _reset_book() -> void:
	for i in _slots.size():
		_blank(_slots[i])
	_cast = []
	_warm = []
	_to_cast = []
	_plan = []
	_step = 0
	_spread_i = -1
	_read = 0
	_prev = null
	_turn_t = -1.0
	_tgt = {}
	_shot = {}
	_shot_t = 0.0
	_hold_t = 0.0
	_zoom_dir = 0
	_snap = true
	_dist_s = -1.0
	_dip_t = -1.0
	_film_at = -1
	_film_clip = {}
	_film_prev = ""
	_turn_spread(0)


## A window resize changes what a panel is worth in pixels, but NOTHING is resized here.
##
## A render target reallocated while it is being sampled is the black-triangle and
## colour-noise corruption main's stage governor documents at length - and worse, a resize
## also throws away the held panels' pictures. So the new figure is remembered and applied
## at the next spread, where every target is repainted from scratch anyway.
func on_stage_resized(size: Vector2) -> void:
	_size_targets(size)
	queue_redraw()


func _size_targets(size: Vector2) -> void:
	_stage_size = size
	# A panel fills the frame at the reading distance, so it is worth close to the whole
	# short axis. Clamped below so a small window still gets a legible panel, and above so
	# a 4K export does not allocate twenty-four enormous targets.
	_panel_px = clampi(int(minf(size.x, size.y) * PANEL_SCALE), PANEL_MIN, PANEL_MAX)


func _build_slots() -> void:
	for i in POOL * 2:
		var vp := SubViewport.new()
		vp.size = Vector2i(SLOT_IDLE, SLOT_IDLE)
		vp.transparent_bg = false
		vp.disable_3d = true          # every scene rasterizes to the 2D canvas
		vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
		vp.process_mode = Node.PROCESS_MODE_DISABLED
		add_child(vp)
		_slots.append(vp)


# --- the Vehicle contract ----------------------------------------------------

## The spread owns its cast. See [method Vehicle.owns_cast] - this is the whole difference
## between a page and a slideshow.
func owns_cast() -> bool:
	return true


## A Director cut MOVES THE READING to the next panel; it does not build anything. The
## panel already holds a live scene, cast when the spread turned.
func take_over(_outgoing: GhostScene) -> GhostScene:
	if _spread == null:
		return null
	if _step + 1 >= _plan.size():
		_turn_spread(_spread_i + 1)
	elif _spread_i >= 0:
		_step += 1
		_read = int(_plan[_step])
		_begin_shot(_read)
	# CAST ON DEMAND. The rest of the spread is cast one panel per frame to keep the turn
	# from hitching, and a fast cut can reach a panel before its turn in that queue comes
	# up - which handed the Director a null and left the reading stuck on the panel behind.
	# _cast_panel is idempotent, so this only ever builds the one that was still owed.
	_cast_panel(_read)
	return _focal()


func owns_bookend() -> bool:
	return true


func advance(features, delta: float, bookend: float) -> void:
	_bookend = bookend
	if _mod != null:
		_mod.advance(delta, _energy_of(features))
	_cast_one()
	if _turn_t >= 0.0:
		_turn_t += delta
		if _turn_t >= TURN_TIME:
			_turn_t = -1.0
			_prev = null
			# THE OUTGOING SPREAD'S TARGETS ARE NOW DEAD. Nothing samples that pool again
			# until the next turn opens it, and _open_slot sizes a slot before anything
			# draws it - so this is the one safe moment to give the memory back. Skipping
			# it kept a full spread of full-size render targets alive for the whole of the
			# next spread, which is half the vehicle's VRAM doing nothing.
			for i in POOL:
				var vp: SubViewport = _slots[(1 - _pool) * POOL + i]
				if vp.size.x > SLOT_IDLE or vp.size.y > SLOT_IDLE:
					vp.size = Vector2i(SLOT_IDLE, SLOT_IDLE)
	_ease(delta)
	_prepare_lens()
	_update_liveness()
	_tick_cast(features, delta)
	queue_redraw()


## AudioFeatures is not a type this file should have to import to read one number off, and
## a vehicle must keep working when there is no audio at all (the splash, a songless boot),
## so this asks softly rather than declaring a dependency.
func _energy_of(features) -> float:
	if features == null:
		return 0.0
	return clampf(float(features.energy), 0.0, 1.0)


# --- the book ----------------------------------------------------------------

func _focal() -> GhostScene:
	if _read < 0 or _read >= _cast.size():
		return null
	var sc = _cast[_read]
	return sc if sc != null and is_instance_valid(sc) else null


func _turn_spread(idx: int) -> void:
	if idx > 0 and _spread != null:
		_prev = _spread
		_turn_t = 0.0
	# Let the outgoing spread's cast go, but NOT its pictures: _blank stops each viewport
	# before freeing what is inside it, so the spread turning away is still a drawn spread.
	for i in POOL:
		_blank(_slots[_pool * POOL + i])
	_spread_i = idx
	_pool = idx % 2
	_spread = ComicSpread.new(hash([Director.session_seed(), "comic-spread", idx]))
	_cast = []
	_cast.resize(_spread.panels.size())
	_warm = []
	_warm.resize(_spread.panels.size())
	_warm.fill(0)
	_read = 0
	# The panel being read is cast NOW - the show cannot open on an empty frame - and the
	# rest are queued one per frame. Building a dozen scenes in one frame is a visible
	# hitch, and there is nowhere to hide it; spread over the next few frames it lands under
	# the page turn, and any panel the camera reaches before its turn is cast on demand.

	_choose_film()               # BEFORE the plan: _roll_plan forces the film panel into it
	_roll_plan()
	_read = int(_plan[0]) if not _plan.is_empty() else 0
	_step = 0
	# EVERY PANEL EXCEPT THE ONE CAST BELOW. It used to be `range(1, n)`, which was correct
	# only while the reading always opened on panel 0. It does not any more - a spread picks a
	# PLAN and opens on `_plan[0]` (see _roll_plan) - so whenever that is not panel 0, panel 0
	# was queued by nobody and cast by nobody, and stayed an empty ruled frame for the whole
	# spread. Caught by the probe's per-panel line as `1[- - 0]` on a spread whose plan began
	# at panel 2, which is exactly the "empty frame" this pass exists to remove.
	_to_cast = []
	for i in _spread.panels.size():
		if i != _read:
			_to_cast.append(i)
	_cast_panel(_read)
	_choose_spread_look()
<<<<<<< Updated upstream
	_snap = true                  # a new sheet is a new shot, not a place to ease toward
	_begin_shot(_read, true)
=======
	_snap = true                      # a new sheet is a new shot, not a place to ease toward
	_choose_move(true)
>>>>>>> Stashed changes


## WHICH PANELS THIS SPREAD WILL SETTLE ON, in reading order. See FOCUS.
##
## Sampled, spread out, and it always includes the FILM PANEL when there is one - a piece of
## footage on the sheet is the one moment that is certainly worth stopping on, and leaving it
## to chance means importing a video and then watching the camera ignore it.
##
## Spaced by picking one panel from each of `want` equal SLICES of the reading order rather
## than by drawing indices at random. A random draw of four from nine clusters: it will
## happily pick panels 5, 6, 7 and 8 and leave the whole left page unvisited, which reads as
## the camera getting stuck in a corner of the sheet. One per slice guarantees the plan walks
## the spread, while the choice inside each slice keeps it from being the same four panels
## every time.
func _roll_plan() -> void:
	_plan = []
	if _spread == null:
		return
	var n := _spread.panels.size()
	if n <= 0:
		return
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "comic-plan", _spread_i])
	var want := clampi(r.randi_range(FOCUS.x, FOCUS.y), 1, n)
	var style := _pick_style(r)
	var picked: Array = []
	if style == "skim":
		# LEFT, RIGHT, LEFT - alternating pages, and the order is NOT sorted afterwards
		# because the alternation IS the gesture. Sorting it would turn a glance back into a
		# read, which is the one thing this style exists not to be.
		var side := 0 if r.randf() < 0.5 else 1
		for _k in want:
			var best := -1
			for i in n:
				if picked.has(i) or _spread.side_of(i) != side:
					continue
				if best < 0 or r.randf() < 0.4:
					best = i
			if best >= 0:
				picked.append(best)
			side = 1 - side
	else:
		# ONE PER SLICE of the reading order, so the plan walks the spread instead of
		# clustering - a free draw of four from nine will happily take 5, 6, 7 and 8 and leave
		# the whole left page unvisited, which reads as the camera stuck in a corner.
		var seen := {}
		for k in want:
			var lo := int(floor(float(k) * float(n) / float(want)))
			var hi := int(floor(float(k + 1) * float(n) / float(want))) - 1
			var at := clampi(r.randi_range(lo, maxi(lo, hi)), 0, n - 1)
			seen[at] = true
		if _film_at >= 0:
			seen[_film_at] = true
		for i in n:                       # READING ORDER, and each panel at most once
			if seen.has(i):
				picked.append(i)
	if picked.is_empty():
		picked.append(0)
	# WHAT CATCHES THE EYE FIRST. The film panel by preference (see FILM_FIRST), otherwise the
	# biggest panel on the sheet, which is the one a page designer sized to be looked at.
	var grab := -1
	if _film_at >= 0 and r.randf() < FILM_FIRST:
		grab = _film_at
	elif style == "grab":
		grab = _largest_panel()
	if grab >= 0:
		# ERASE THEN PUSH, and erase FIRST so a grabber that was already in the plan moves to
		# the front instead of appearing twice. When it was not in the plan this adds one, so
		# the result is trimmed back to the sampled size below - a spread that rolled "three
		# panels" and then got a fourth for free is not the plan it rolled.
		picked.erase(grab)
		picked.push_front(grab)
	while picked.size() > want:
		picked.remove_at(picked.size() - 1)
	_plan = picked


func _pick_style(r: RandomNumberGenerator) -> String:
	var total := 0.0
	for k in PLAN_STYLES:
		total += float(PLAN_STYLES[k])
	var pick := r.randf() * total
	for k in PLAN_STYLES:
		pick -= float(PLAN_STYLES[k])
		if pick <= 0.0:
			return String(k)
	return "read"


## The panel with the most area on the sheet - what a page designer made big because it is the
## one to look at, and therefore the fallback grabber when there is no footage.
func _largest_panel() -> int:
	var best := 0
	var best_a := -1.0
	for i in _spread.panels.size():
		var rr: Rect2 = _spread.panels[i]["rect"]
		var a := rr.size.x * rr.size.y
		if a > best_a:
			best_a = a
			best = i
	return best


## DOES THIS SPREAD GET A PIECE OF FOOTAGE, and if so which panel and which clip.
##
## Seeded off the spread, like everything else here, so a session replays the same footage
## in the same panels - the show is reproducible from one seed and this is not the place
## to make it stop being. A frequency of 0, or a library with nothing in it, leaves
## `_film_at` at -1 and the comic behaves exactly as it did before films existed.
func _choose_film() -> void:
	_film_at = -1
	_film_clip = {}
	if _spread == null or Films.frequency() <= 0.0:
		return
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "film-spread", _spread_i])
	if r.randf() > Films.frequency():
		return
	var list := Films.clips()
	if list.is_empty():
		return
	var at := r.randi() % list.size()
	# NOT THE SAME CLIP TWICE RUNNING. With one film panel per spread, a small library and
	# an independent draw per spread, the same clip lands on consecutive spreads often
	# enough to read as "that video again" rather than as a library.
	if list.size() > 1 and String((list[at] as Dictionary).get("source", "")) == _film_prev:
		at = (at + 1) % list.size()
	_film_clip = list[at]
	_film_at = _best_panel_for(_film_clip, r)
	# IS THERE ANYTHING TO PLAY YET. A clip is prepared a window at a time, cut from the
	# original when something wants it (see Films.WINDOW), so the window covering this
	# moment may still be encoding. Asking starts that cut; not getting it back means this
	# spread simply goes without footage, and the panel is an ordinary scene rather than a
	# blank rectangle waiting for a file. The whole feature self-throttles on this line -
	# film appears as often as the machine can prepare it.
	if not Films.warm(_film_clip, maxf(Spectrum.current.time, 0.0)):
		_film_clip = {}
		_film_at = -1
		return
	_film_prev = String(_film_clip.get("source", ""))


## WHICH PANEL SUITS THIS CLIP'S SHAPE. Footage covers its panel, so whichever axis is
## spare gets cropped off - a 16:9 clip in a 0.45-aspect panel loses three quarters of its
## width, and there is no framing clever enough to make that not matter. A spread offers
## twice as many shapes as a page did, so the answer is simply to put the film in the one
## closest to the clip's own, where the crop is a trim rather than a demolition.
##
## Compared in LOG space, because aspect is a ratio: 2.0 and 0.5 are equally wrong for a
## square clip, and subtracting them would call one of them nearly right.
##
## Falls back to a plain random panel when the clip's shape is unknown, which is the honest
## answer rather than a guess - see Films.aspect_of.
func _best_panel_for(clip: Dictionary, r: RandomNumberGenerator) -> int:
	var n := _spread.panels.size()
	var want := Films.aspect_of(clip)
	if want <= 0.0 or n <= 1:
		return r.randi() % maxi(n, 1)
	var best := 0
	var best_err := INF
	for i in n:
		var err := absf(log(maxf(_spread.panel_aspect(i), 0.01) / want))
		if err < best_err:
			best_err = err
			best = i
	return best


## One queued panel per frame. See _turn_spread.
func _cast_one() -> void:
	if _to_cast.is_empty():
		return
	_cast_panel(int(_to_cast.pop_front()))


func _cast_panel(i: int) -> void:
	if _spread == null or i < 0 or i >= _cast.size():
		return
	if _cast[i] != null and is_instance_valid(_cast[i]):
		return
	var vp := _open_slot(i, _panel_px_for(i))
	# FOOTAGE, where the spread called for it. Cast directly rather than through the
	# Director: a film is not in the catalogue (see FilmScene), because a scene that only
	# exists when the viewer has imported something would make the running order depend on
	# the library.
	if i == _film_at and not _film_clip.is_empty():
		var film := FilmScene.new()
		film.set_clip(_film_clip, maxf(Spectrum.current.time, 0.0))
		film.init_with_seed(hash([_spread_i, i, "film"]), "static")
		film.scene_name = "film"
		vp.add_child(film)
		_cast[i] = film
		return
	# The salt is the panel index, so the spread's panels are separate draws off the novelty
	# scheduler rather than the same one repeated - the Director's clock has not moved
	# between them (see Director.mint_scene). Quiet for every panel but the one being read:
	# casting a spread is one change of scene, not a dozen.
	var sc := Director.mint_scene(hash([_spread_i, i]) & 0xFFFF, i != _read)
	vp.add_child(sc)
	_cast[i] = sc


## Size a panel's render target to ITS aspect and start it running.
##
## SIZED BY HEIGHT, because the camera is fitted by height (see _fit): a panel spans the
## frame vertically, so its target wants about the stage's height and its width follows
## from its aspect. The obvious alternative - constant AREA - overshoots badly at the tall
## end, where it makes a 0.44-aspect panel half again taller than the screen it will be
## drawn on, for no visible gain and 2.3x the pixels.
## EVERY PANEL IS SIZED BY HOW BIG IT IS ON SCREEN, not by how big a panel can ever be.
##
## This is what pays for every scene playing at once. The budget used to be three live panels
## out of nine or twelve, and the rest were frozen render targets holding their last drawn
## frame - which is a page of stills with three moving windows in it, and a panel frozen before
## its scene had drawn anything is simply black. "You're optimizing for real-time, but we are
## cutting videos. So we can just play every scene, and we should. Freezing them looks
## terrible."
##
## Sizing every panel at `_panel_px` is what made that unaffordable: twelve 512-pixel targets
## redrawn every frame, when eleven of them occupy a few hundred pixels of the picture. The
## cost that matters is TOTAL PIXELS, not panel count, so a panel gets the resolution its place
## in the shot actually asks for and is re-opened larger when the camera comes to it.
func _open_slot(i: int, px := -1) -> SubViewport:
	var vp: SubViewport = _slots[_pool * POOL + i]
	var a := _spread.panel_aspect(i)
	var h := px if px > 0 else _panel_px
	var w := int(round(float(h) * a))
	var want := Vector2i(maxi(SLOT_IDLE, w), maxi(SLOT_IDLE, h))
	# ONLY EVER GROW WITHIN A SPREAD. Shrinking a live target throws away the picture in it and
	# the panel flashes; the pool is reset to SLOT_IDLE when the spread turns (see advance),
	# which is the one moment nothing is reading from it.
	if want.x > vp.size.x or want.y > vp.size.y:
		vp.size = Vector2i(maxi(want.x, vp.size.x), maxi(want.y, vp.size.y))
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	vp.process_mode = Node.PROCESS_MODE_INHERIT
	return vp


## The resolution panel [param i] earns in the shot on screen now: its drawn height in pixels,
## with a floor so a panel in the far distance is still a picture and a ceiling at the full
## target so the one being read is as sharp as the vehicle can make it.
##
## THIS IS ALSO THE PIXELATION ANSWER. A panel rasterises into its target and is then textured
## onto the paper, so a shot tighter than the target is magnifying a bitmap - "some of the
## in-frame scenes looked a bit pixelated". Re-opening the focal panel at the resolution the
## camera is actually asking for is what removes that, and it costs nothing on the panels that
## are not being looked at.
func _panel_px_for(i: int) -> int:
	var r := _panel_rect(i)
	var on_screen := maxf(r.size.y, r.size.x / maxf(_spread.panel_aspect(i), 0.01))
	# THE PANEL BEING READ MAY EXCEED `_panel_px`, up to the hard ceiling. That constant is the
	# budget for a whole spread of targets, and applying it to the ONE panel filling the frame
	# is what left a tight shot magnifying its own bitmap - measured at 1.21x on a push-in, and
	# reported as "some of the in-frame scenes looked a bit pixelated... we should be able to
	# just scale and skew the live scene, and compute pixels precisely". A live scene can
	# indeed be drawn at any resolution; it just has to be ASKED for one.
	var ceiling := PANEL_MAX if (i == _read or i == _film_at) else _panel_px
	return clampi(int(round(on_screen * PANEL_OVERSAMPLE)), SLOT_IDLE, ceiling)


## Stop a viewport WITHOUT clearing it: the render target keeps its last frame. The ORDER
## matters - stop first, free second - or the target repaints itself empty on the frame
## between the two.
func _freeze(vp: SubViewport) -> void:
	vp.render_target_update_mode = SubViewport.UPDATE_DISABLED
	vp.process_mode = Node.PROCESS_MODE_DISABLED


func _resume(vp: SubViewport) -> void:
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	vp.process_mode = Node.PROCESS_MODE_INHERIT


## Stop a viewport and release the scene inside it, keeping the picture it last drew.
func _blank(vp: SubViewport) -> void:
	_freeze(vp)
	for c in vp.get_children():
		c.queue_free()


# --- liveness ----------------------------------------------------------------

## EVERY PANEL YOU CAN SEE IS MOVING, and no panel you cannot see costs anything.
##
## A spread whose panels arrive one at a time is a slideshow with gutters; a comic page has
## all of it drawn at once. But a dozen live scenes is not a thing that runs - the stage
## governor spends its whole budget on one - so liveness follows the CAMERA instead of the
## clock. At the reading distance one panel fills the frame and its neighbours crowd the
## edges, so what is on screen is typically one to three panels; the rest are stopped
## render targets holding their last frame, exactly as before, and they resume when the
## camera comes back to them.
##
## LIVE_MAX is the hard bound. It is ordered by distance from the panel being read, so if a
## rake ever puts more of the spread in shot than the budget allows, the ones that keep
## moving are the ones the reading is about.
func _update_liveness() -> void:
	if _spread == null:
		return
	var frame := Rect2(Vector2.ZERO, _stage_size).grow(_stage_size.x * 0.06)
	var mid := _stage_size * 0.5
	var want: Array = []
	for i in _cast.size():
		if _cast[i] == null or not is_instance_valid(_cast[i]):
			continue
		# A panel still below its warm-up is wanted WHEREVER IT IS, on screen or not. It has
		# no picture yet, so leaving it out is what makes it black - and the camera reaches
		# most of the sheet eventually, so "off screen right now" is not "never seen".
		var cold: bool = int(_warm[i]) < WARM_FRAMES
		var rect := _panel_rect(i)
		if i != _read and not cold and not rect.intersects(frame):
			continue
		# DISTANCE ON THE SCREEN, not distance in the panel array.
		#
		# It used to be `absi(i - _read)`, and a spread's panel array is two CONTIGUOUS BLOCKS
		# - the left page's panels and then the right's. So ranking by index distance ranks by
		# PAGE: read a panel on the left and the three live slots all go to the left page, and
		# the entire right page is a set of frozen stills however much of it is in shot.
		# Reported as "I keep seeing instances where an entire page is disabled, while we are
		# viewing the other side of that spread." Screen distance has no idea which page a
		# panel is on; it keeps alive whatever is nearest the middle of the picture, which is
		# the only thing the eye is actually asking about.
		var d := rect.get_center().distance_to(mid) if i != _read else -1.0
		want.append({"i": i, "d": d, "cold": 0 if cold else 1})
	# NEED FIRST, DISTANCE SECOND. Until every panel has drawn its moment the budget goes to
	# the ones that have not; after that it follows what is in the middle of the shot.
	want.sort_custom(func(a, b):
		return int(a.cold) < int(b.cold) if int(a.cold) != int(b.cold) else float(a.d) < float(b.d))
	var budget := _live_budget()
	# THE PANEL BEING READ IS ALWAYS LIVE, AND SO IS THE FILM. Reserved BEFORE the budget is
	# spent, not merely sorted to the front of it.
	#
	# This is the frozen picture. The sort puts every panel still below its warm-up ahead of
	# every warm one - which is right, a panel with no picture needs frames more than a panel
	# that has one - but LIVE_MAX is three, and a page turn casts a whole spread at once. Three
	# cold panels therefore filled the entire budget and froze the panel the camera was looking
	# at: the Director's own current scene, stopped, for as long as the newcomers took to warm.
	# Reported as "the scene in that comic book frame becomes stuck, frozen, and stops moving at
	# all" and "takes a long time to start again" - and confirmed off the export, where two
	# frames a full second apart have the scene's eyes in identical positions.
	var keep: Array = []
	if _read >= 0 and _read < _cast.size() and _cast[_read] != null \
			and is_instance_valid(_cast[_read]):
		keep.append(_read)
	if _film_at >= 0 and _film_at != _read and _film_at < _cast.size() \
			and _cast[_film_at] != null and is_instance_valid(_cast[_film_at]):
		keep.append(_film_at)
	for e in want:
		if keep.size() >= budget:
			break
		if not keep.has(int(e.i)):
			keep.append(int(e.i))
	_frame += 1
	for i in _cast.size():
		if _cast[i] == null or not is_instance_valid(_cast[i]):
			continue
		var vp: SubViewport = _slots[_pool * POOL + i]
		# On this frame at all, and if so, is it this one's turn? The read panel's turn is
		# every frame; the others are phased against each other by index so two of them
		# never land on the same frame.
		# The film panel is exempt from the off-focal phasing: half the samples looks
		# like a scene running at the right speed with a coarser clock, and looks like
		# BROKEN PLAYBACK on footage, which the eye reads at a much finer grain.
		# A COLD PANEL RUNS EVERY FRAME IT IS KEPT, phasing included: half the samples is
		# fine for a moment that is already on the paper and wrong for one still drawing it.
		# ...and the off-focal PHASING is the same concession at a finer grain, so it lifts on
		# an export too: half the samples is a scene running on a coarser clock, which is
		# cheaper and very slightly worse, and there is nothing to buy with it here.
		var every := budget >= POOL
		var on: bool = keep.has(i) and (every or i == _read or i == _film_at
			or int(_warm[i]) < WARM_FRAMES or (_frame + i) % OFF_FOCAL_PERIOD == 0)
		if on:
			_warm[i] = int(_warm[i]) + 1
			# GROW INTO THE SHOT. A panel cast while it was small on screen keeps that target
			# until the camera comes to it; re-opening in REOPEN_STEP jumps rather than every
			# frame keeps a push-in to a handful of reallocations instead of hundreds.
			var px := _panel_px_for(i)
			if px > vp.size.y and (i == _read or i == _film_at
					or float(px) > float(vp.size.y) * REOPEN_STEP):
				_open_slot(i, px)
			_resume(vp)
		else:
			_freeze(vp)
	_live = keep


## HOW MANY PANELS MAY RUN THIS FRAME. See LIVE_ALL_ON_EXPORT: the real-time budget, or no
## budget at all when there is no frame deadline to spend it against.
##
## Asked of the command line rather than of main, because a vehicle is built before main has
## finished deciding anything and must not reach up into it - the same reason the Director is
## asked for the session seed instead of being handed one.
func _live_budget() -> int:
	if LIVE_ALL_ON_EXPORT and OS.get_cmdline_user_args().has("--export"):
		return POOL
	return LIVE_MAX


## Drive every live panel EXCEPT the one being read - the Director drives that one, as its
## current scene, and ticking it twice would run it at double speed. This is the same pair
## of calls Director._tick_animation makes, because it is the same job.
func _tick_cast(features, delta: float) -> void:
	var focal := _focal()
	var dt := minf(delta, 1.0 / 30.0)
	for i in _live:
		var sc = _cast[i]
		if sc == null or not is_instance_valid(sc) or sc == focal:
			continue
		# The film panel runs every frame (period 1) - see _update_liveness.
		var period := 1 if (_live_budget() >= POOL or i == _film_at
			or int(_warm[i]) < WARM_FRAMES) else OFF_FOCAL_PERIOD
		if (_frame + i) % period != 0:
			continue                      # not this panel's frame - see OFF_FOCAL_PERIOD
		# ...and when it IS its frame, it is handed the whole period's worth of time, so it
		# runs at the right speed rather than at 1/period of it.
		sc.update(features, dt * period)
		sc.view.commit(dt * period)


# --- camera ------------------------------------------------------------------

## THE MOVE VOCABULARY. A cut picks one of these, and the camera then TRAVELS IT.
##
## The first cut of this had no vocabulary: every cut set a new target and the camera eased
## toward it, which is a spring settling, not a camera. It gave one behaviour - bounce to
## the next panel, sit on it, bounce again - and settling on a single panel is a shot that
## should be an EVENT, not the rule. So a move is now a PATH: a start state, an end state, a
## duration and a curve, travelled at its own speed. Most of them never settle at all.
##
##   w      relative weight in the bag
##   dur    seconds, sampled in this range
##   ease   linear (a dolly, constant speed) / smooth / out (fast then settle) / snap (a cut)
##   hard   true = the move BEGINS somewhere else entirely, so it starts on a jump cut
##          rather than continuing from where the camera was
##   open   true = the move deliberately ENDS OFF A PANEL (it runs past its subject, or
##          across a row). Such a move is exempt from the settle guard in _choose_move,
##          because pulling its end back onto a panel would delete the gesture - and it is
##          safe only because a finished move now chains (see _chain_move) instead of
##          stopping wherever it ran out.
##   chain  true = eligible as a CHAINED successor, i.e. gentle and continuous. A jump is a
##          thing a CUT does; a camera that jumps on its own, with no cut behind it, reads
##          as a glitch.
##
## THE BAG IS MOSTLY GENTLE, and that is a correction. It was first weighted the other way,
## in answer to "a constant shot of a single frame should be an event, not the rule" - and
## it overshot: 41% of the weight began somewhere else entirely, so two shots in five opened
## on a jump. Read back as "it just bounces all over the place... not cinematic at all".
##
## Punctuation is now about one shot in six. The rule the two reports agree on, once both
## are taken seriously, is that a camera should be MOVING most of the time and JUMPING
## rarely - which is neither "sit on each panel" nor "cut constantly".
<<<<<<< Updated upstream
const SHOTS := {
	# WHAT EACH ENTRY MEANS. `fill` is a band MULTIPLYING the panel's natural framing, so a
	# shot is expressed as "how much tighter or wider than this panel reads at" rather than as
	# an absolute the panel's size can invalidate. `aim` is how far the station sits off the
	# panel's centre, in page widths. `az` is how far round it swings, in radians. `rate` is
	# how fast the follower closes on it, multiplying the severity rate. `hard` means the shot
	# begins with a cut - see _cut_to, the only place in this file that assigns the camera.
	#
	# EVERY ONE OF THESE IS A DISTANCE OR A SPEED. That is the whole change: the old table gave
	# each move a DURATION and an EASE CURVE and let thirteen branches hand-roll their own
	# interpolation between two poses, which is why continuity was a convention rather than a
	# property, and why four rounds of guards never converged.
	"settle":  {"w": 4.0, "fill": [0.92, 1.12], "aim": 0.04, "az": 0.00, "rate": 1.00},
	"push":    {"w": 2.5, "fill": [1.35, 2.30], "aim": 0.03, "az": 0.00, "rate": 0.75},
	"pull":    {"w": 1.5, "fill": [0.55, 0.80], "aim": 0.03, "az": 0.00, "rate": 0.75},
	"pan":     {"w": 3.0, "fill": [0.90, 1.20], "aim": 0.30, "az": 0.10, "rate": 0.70},
	"orbit":   {"w": 1.5, "fill": [0.90, 1.20], "aim": 0.05, "az": 0.55, "rate": 0.55},
	# THE ONE THAT SHOWS THE PAGE. Nothing else in this table frames more than its own panel -
	# the widest, `pull`, still leaves the subject filling most of the picture - so the camera
	# had no way to say "here is the sheet you are reading": "it never pulls-back, to reveal the
	# greater page and multiple frames; it's just a constant bouncing around between individual
	# frames, close-up". `page` aims at the middle of the panel's own page and frames wide
	# enough that its neighbours and the gutters between them are the composition.
	"page":    {"w": 2.0, "fill": [0.26, 0.46], "aim": 0.10, "az": 0.15, "rate": 0.55,
		"wide": true},
	# The quick one. Same grammar, four times the rate - a whip is a fast move, not a jump,
	# and expressing it as a rate is what keeps it continuous however short it is.
	"whip":    {"w": 0.5, "fill": [0.85, 1.30], "aim": 0.20, "az": 0.25, "rate": 3.20},
	# ...and the one discontinuity. It must change the SUBJECT to be allowed at all, which is
	# checked in _worth_cutting rather than hoped for here.
	"cut":     {"w": 0.9, "fill": [0.80, 1.60], "aim": 0.08, "az": 0.80, "rate": 0.90,
		"hard": true},
=======
const MOVES := {
	# --- travelling: the camera is moving for the whole shot ---------------------
	"drift":   {"w": 4.0, "dur": [8.0, 15.0], "ease": "smooth", "hard": false, "chain": true},
	"push":    {"w": 3.0, "dur": [7.0, 13.0], "ease": "smooth", "hard": false, "chain": true},
	"track":   {"w": 2.0, "dur": [9.0, 16.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"orbit":   {"w": 2.0, "dur": [9.0, 15.0], "ease": "smooth", "hard": false, "chain": true},
	# THE SPREAD'S OWN MOVE, and the reason the spread earns its complexity: a lateral pan
	# from a panel on one page, across the spine, onto a panel on the other. It is the
	# longest continuous run of content the vehicle has ever had - a single page had no
	# path longer than one page's width, and half of that was margin.
	"spine":   {"w": 2.0, "dur": [12.0, 20.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"sweep_h": {"w": 1.5, "dur": [9.0, 16.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"sweep_v": {"w": 1.0, "dur": [9.0, 16.0], "ease": "linear", "hard": false, "chain": true, "open": true},
	"pull":    {"w": 1.5, "dur": [7.0, 13.0], "ease": "smooth", "hard": false, "chain": true},
	# --- and the one that just looks at a panel ----------------------------------
	"hold":    {"w": 1.5, "dur": [5.0, 10.0], "ease": "smooth", "hard": false, "chain": true},
	# --- arrivals: a gesture that lands, then holds ------------------------------
	"swoop":   {"w": 0.6, "dur": [2.6, 5.0],  "ease": "out",    "hard": true},
	"whip":    {"w": 0.4, "dur": [0.45, 0.9], "ease": "out",    "hard": false},
	# --- and the one that ARRIVES AND STAYS ARRIVED ------------------------------
	# Weight 0: never drawn from the bag, only named by _chain_move when the cut is close.
	# See ARRIVE.
	# --- and the one that SHOWS THE BOOK BEFORE IT SHOWS A PANEL ------------------
	# Weight 0: never drawn from the bag, only named by a page turn. See ESTABLISH.
	"establish": {"w": 0.0, "dur": [6.0, 11.0], "ease": "smooth", "hard": false},
	"settle":  {"w": 0.0, "dur": [3.0, 6.0],  "ease": "smooth", "hard": false},
	# --- and the one that NEVER STOPS AND NEVER GOES ANYWHERE ---------------------
	# Weight 0: never drawn from the bag, only named when a settle or another breathe runs
	# out. See BREATHE_AIM.
	"breathe": {"w": 0.0, "dur": [9.0, 16.0], "ease": "smooth", "hard": false},
	# --- discontinuities: punctuation, not grammar -------------------------------
	# THE JUMP IS THE EDIT; WHAT FOLLOWS IT IS STILL A SHOT. Both of these used the "snap"
	# curve, which returns 1 at every k - so the camera teleported onto its station and then
	# sat on exactly that pose for four to nine seconds. A cut is punctuation, not a freeze
	# frame, and a run of them is "the camera jumps four times and drifts to an edge". They
	# ease now: the discontinuity is the teleport onto `a` (see the `hard` branch in
	# _plan_move), and `b` drifts gently off it.
	"cut":     {"w": 0.8, "dur": [4.0, 9.0],  "ease": "smooth", "hard": true},
	"dip":     {"w": 0.4, "dur": [4.0, 9.0],  "ease": "smooth", "hard": true},
>>>>>>> Stashed changes
}

## How far past the subject a `track` keeps going, in page widths. The camera does not stop
## when it arrives - it passes through, which is what a tracking shot is.
const TRACK_OVERRUN := 0.45
## THE AIM NEVER LEAVES THE SHEET. Spread coordinates, as a margin inside the trim.
##
## A move that aims past the outer edge is aiming at the desk, and no framing distance can
## then put paper across the whole frame - the covering solve has nothing to solve. Sweeps
## and tracks deliberately run to and past the edges, so they are clamped here rather than
## being written not to.
const AIM_MARGIN := 0.06
## How far a `pull` opens out, as a fraction of the framing it started from. A pull is a
## widening, so it is expressed relative to where the camera IS rather than as an absolute -
## that is what keeps it from being a jump.
const PULL_OUT := 0.55
## How deep a `push` gets, in frames of panel height.
const PUSH_FILL := Vector2(2.2, 3.8)
## The far station a `swoop` starts from: elevation in degrees, and how far off.
## RAISED FROM 14-26. This file's own sweep of page_coverage() against elevation is in the note
## above EL_MIN_DEG: 0.34 of the frame is paper at 30 degrees, 0.84 at 40, 1.00 from 50. A
## swoop is a HARD move, so the camera teleports onto this station - and at 14 degrees it
## teleported onto a shot that was two thirds desk, which is "drifts to an edge where there is
## nothing to see" in its most literal form. It is still the lowest rake in the vehicle.
const SWOOP_EL_DEG := Vector2(40.0, 50.0)
const SWOOP_FILL := 0.42
## The arc an `orbit` sweeps, in radians, at severity 1.
##
## PULLED IN FROM 1.1-2.6 (63 to 149 degrees), which was most of a lap of the sheet in one
## shot. It survived the first pass because the move is only about one shot in ten, so it
## never dominated a still frame - but it dominated the MEAN: with the per-shot walk down to
## about eight degrees, the measured shot-to-shot swing at the default was still 24.6, and two
## thirds of that was orbit and swoop. A rare enormous move is exactly what "the camera can
## rotate/turn A LOT every time it shifts" describes, because rare at one shot in ten is every
## half minute. At 26 to 66 degrees it is still plainly an orbit and no longer a lap.
const ORBIT_ARC := Vector2(0.45, 1.15)
## A `dip` goes out over the first of these and back over the second, in seconds.
const DIP_OUT := 0.16
const DIP_IN := 0.34

## HOW MUCH OF THE TIME BEFORE THE NEXT CUT A MOVE MAY SPEND TRAVELLING.
##
## A move used to sample its duration from the vocabulary and nothing else, while the cut that
## ends it comes from the Director's hold - which shrinks with the music's drive. Seven to
## twenty seconds of move against a median cut at five and a half on a driving passage means
## the camera is still converging when the shot ends, every time. Reported as "you can see it
## converging, then BOOM."
##
## So a move now ARRIVES ON A DEADLINE: it is given at most this fraction of the time the
## Director expects to have left (see [method Director.hold_remaining]), and the rest of the
## hold is the camera being somewhere rather than going somewhere. The character of a move
## survives - a whip is still quick, a drift still slow - because this is a CEILING on the
## sampled duration, not a replacement for it.
const ARRIVE := 0.62
## ...and how much of it a CALM camera may spend. The share is what "severity" means at this
## end of the range: a gentle camera is not a slow camera making the same journeys, it is a
## camera that travels less and looks longer.
const ARRIVE_CALM := 0.26
## ...but never compressed below this, in seconds. A move squeezed into a fraction of a second
## is not an arrival, it is the jump cut this exists to prevent - so when the cut is already
## imminent the move simply runs long and the chain picks it up.
const ARRIVE_MIN := 1.6
## With less than this long to go, a finished move does not go anywhere new - it SETTLES.
## Chaining a fresh traverse with three seconds left is how the camera ends up permanently in
## transit, which is the same defect from the other end.
const SETTLE_ROOM := 5.0
## How far a settle is allowed to drift, as a fraction of the way to its panel's centre. Not
## zero: a dead-still camera on a page that is itself drifting reads as the page sliding out
## from under a locked-off shot. This is a breath, not a move.
const SETTLE_DRIFT := 0.22
## HOW MANY TIMES ONE CUT MAY SET OFF AGAIN before it stops travelling and holds.
##
## The chain was written to answer "the camera freezes", and it does - but it answered it
## without a budget, so a shot that ran out of move planned another one, and another, for the
## whole of a hold that can be half a minute. Every one of those re-samples a station: a fresh
## aim on the panel centre, a fresh step of the angle walk. On the panel being read that is a
## camera visibly re-correcting onto a frame it is already looking at, once every few seconds -
## "the camera keeps shifting/resetting to the fractal frame. It happens a half dozen times
## over just a few seconds, and it looks awful."
##
## So a cut may travel this many times after its own move and no more; past that the chain
## SETTLES, and a settle extends itself rather than re-planning (see _ease). That is what makes
## the hold the user asked for structural rather than lucky: "in most cases, it should be
## impossible to not hold for at least a few seconds."
const CHAIN_TRAVEL := Vector2i(1, 2)
## Chance a cut takes the top of CHAIN_TRAVEL instead of the bottom, at severity 1. Scaled by
## severity, because "how restless is this camera" is exactly what that slider means.
const CHAIN_RESTLESS := 0.35
## How far off the panels the aim has to be before a chain treats itself as LOST and recovers.
## An open move is meant to overrun, so this is well below the content core.
const CHAIN_LOST := 0.35
## THE TRAVEL BUDGET IS SECONDS, NOT MOVES, and counting it in moves is why "the camera NEVER
## holds a position: the second it lands somewhere, it's moving somewhere else."
##
## `_budget` caps a move at `room * ARRIVE`, but it CAPS - it returns `min(want, cap)`, and on
## a long hold `want` is the binding one. At camera 0.05 the duration multiplier is 0.55^-0.95
## = 1.77, so a drift sampling 8-15 s becomes 14-27 s; at scene hold 4.0 the room is enormous,
## so the move takes all 27 s and the chain then starts ANOTHER full-length move. Two journeys
## is a per-cut budget that reads as generous until you notice it can mean fifty-five seconds
## of unbroken travel. Turning the hold slider UP made the MOVES longer, not the HOLDS longer,
## which is the opposite of what that slider is for.
##
## So a cut now gets a wall-clock allowance - its share of the hold - and every travelling move
## it plans spends from it. When it is gone the chain settles, whatever the move count says.
## CHAIN_TRAVEL stays as a second ceiling for short holds, where seconds are cheap and the
## count is the thing that binds.
var _travel_left := 0.0
## HOW MUCH OF ITS FRAMING A CUT INHERITS from the shot before it, sampled per cut.
##
## A station samples `fill` freely, so a cut could follow a deep push (2.2 to 3.8 frames of
## panel) with a fresh draw from FILL (0.78 to 1.30) - the camera flies backwards, lands on the
## new panel, and the next move pushes in again: "the camera quickly jerks backward, then zooms
## back inward, while moving to a new frame... why would we not just slowly transition from one
## frame to the next, while maintaining the zoom level?" Quite. The new shot now lerps its
## sampled framing toward the framing on screen, so a cut is a change of SUBJECT before it is a
## change of distance. Sampled rather than fixed, and short of 1, so the framing still relaxes
## back toward the vocabulary over a few shots instead of pinning wherever a push left it.
const FILL_CARRY := Vector2(0.45, 0.85)
## Chance a page turn OPENS ON THE WHOLE SPREAD and travels in, rather than starting on its
## first panel. See the `establish` branch in _plan_move.
const ESTABLISH := 0.7
## How wide an establishing shot starts, in frames of panel height - wide enough that the sheet
## is the subject and the panels are its contents.
const ESTABLISH_FILL := 0.34
## A `breathe`: how far it slides the aim in page widths, how much it opens or closes the
## framing as a fraction of it, and how far it swings the azimuth in radians.
##
## THE SETTLE WAS A FREEZE AND THIS IS THE FIX. A settle copies the whole camera and eases the
## aim SETTLE_DRIFT of the way to the panel centre, so once that ease has run - three to six
## seconds - the picture is motionless, and _ease then extended its duration rather than
## replanning it. On a long hold that is exactly what was reported: "the camera drifts to a new
## frame, then holds there doing nothing for a solid 40 seconds." A settle is still the ARRIVAL
## (it resolves a shot that was mid-convergence); what follows it is a breathe, and a breathe
## follows a breathe, so the camera slows to a crawl and never actually stops. Small enough
## that it reads as a held shot rather than as a move, big enough to see over ten seconds.
## A `breathe`: how far it may creep from where the shot arrived, in page widths, and how far
## it leans the azimuth over that whole creep, in radians.
##
## THREE THINGS THE FIRST VERSION GOT WRONG, all of them visible at camera 0 and all of them
## the same mistake - a held shot was given a random walk instead of a direction.
##   1. IT HAD NO SEVERITY TERM. At 0 every other source of motion multiplies to zero - the
##      angle walks, the page drift, the roll, the hard moves, the fill range - so the breathe
##      was the only thing moving and therefore the only thing to look at.
##   2. IT COMPOUNDED THE FRAMING. Each breathe multiplied the previous one's fill and handed
##      its direction forward, so the distance ratcheted one way for a whole scene until it
##      hit the near floor or the covering ceiling: "zooming in/out at random, with hard peaks
##      at each end". A held shot does not change distance at all now.
##   3. IT RE-ROLLED ITS DIRECTION. A fresh random heading every 10 s, with the only restoring
##      force a HARD BARRIER at the panel edge, is a random walk against a wall - it parks at
##      the wall and jitters, and at the wall `_content_at` is inside its smoothstep band so
##      aim jitter becomes distance swing. "Constant vibrating... pivoting all over the page."
## One heading, chosen when the shot arrives, crept along for the rest of the hold. That is a
## held shot with a slow drift on it, which is what a held shot is.
const BREATHE_AIM := 0.11
const BREATHE_AZ := 0.05
## ...and how much of that survives at camera 0. Not zero: "never stopping at all" is the
## original ask, and a locked-off camera on a page that is itself locked off is a photograph.
const BREATHE_CALM := 0.3
## WHAT MAKES A JUMP CUT WORTH MAKING: it must land on a DIFFERENT PANEL, and it must move
## the aim at least this far in page widths.
##
## The first draft of this rule let the jump qualify on size instead - a big enough change of
## framing or angle counted. Measured against the export, that rule would have permitted seven
## of the eleven discontinuities in it, including a 1.42x tighten at 0:22 and a ~3x widen at
## 2:01, both of them on the panel the camera was already holding. A cut is a change of
## SUBJECT; scale is how it is photographed, not what it is of. "The camera 'jumps' to correct
## its position... landing on the exact same frame. This is the kind of jarring behavior we
## MUST find a way to eliminate." So: different panel, or it is not a cut - and a cut within
## one panel becomes a drift, which gets there continuously.
##
## The aim distance is a second bar rather than an alternative, because two panels can share an
## edge and a "cut" across a gutter is the same non-event by another name.
const CUT_MIN_AIM := 0.28
## HOW FAST THE PLACED POSE FOLLOWS ITS TARGET, as an exponential rate: the rake, then the
## distance. Larger is tighter to the target and closer to the old instantaneous behaviour.
##
## THE MOVE VOCABULARY IS NOT THE ONLY THING THAT MOVES THE CAMERA, and this is the half that
## nothing above could see. _place_eye derives the rake from `_flatten()`, which reads the
## panel being READ - so at a cut between a `field` scene and a `subject` scene the flatten
## steps 0 <-> FIELD_FLATTEN in ONE FRAME and takes the elevation up to 21.7 degrees with it.
## The same frame can also change which panel the aim is inside (the scale panel), and whether
## that panel is the film (which switches the distance solve to its contained branch, measured
## at 3.03 units against 2.13 for the same shot). None of those are moves; all of them are
## teleports, and they fire at EVERY cut and every gutter crossing. That is why "the camera
## jumps... landing on the exact same frame" survived every fix made in the move layer: the
## jump was never in the move layer.
##
## Easing the two SCALARS rather than the eye position keeps the rig page-local: `c` and `dir`
## stay exact, so the page's own attitude drift still rotates the camera and the sheet by the
## same matrix and the projection stays invariant to it. Easing a world-space eye would break
## that and the page would visibly slide.
const FLAT_EASE := 1.3
const POSE_EASE := 2.4


## How far a page turn may move the camera's side of the sheet, in radians at camera 1. A page
## turn is the single largest change of angle the grammar allows, and re-rolling over the whole
## circle - which is what this replaced - is a walk round the table rather than a lean.
const TURN_AZ := 1.2


## THE CAMERA KNOB, applied. See [member Director.camera] for what it is - the slider is
## labelled "Camera"; `severity` is only this file's word for it.
##
## Every constant that describes how MUCH the camera does is read through _sev(), so the slider
## moves the whole behaviour along one axis. What it scales, and why each belongs to it:
##   how far a shot travels and how wide its framing band is - see REACH_CALM
##   how fast the follower closes                            - see RATE_CALM
##   the angular arc a shot may reach, and the walk step      - how far it can swing at all
##   how far a page turn moves the set-up                     - the one big change in grammar
##   the weight of the one discontinuous shot                 - at 0 it never jump-cuts
##   the sheet's own drift rate                               - scenery reads as camera motion
##
## REACH IS THE ONE THAT MATTERS FOR HOLDING, and the previous design had only speed. Scaling
## speed alone means a gentle camera makes the same journeys more slowly, so it spends MORE of
## the hold travelling: measured at 71% held at camera 0 against 86% at camera 1, an inversion
## that is exactly why turning the slider down never helped.

## HOW MUCH GENTLER A FIELD SCENE IS RAKED, 0..1 toward square-on.
##
## ghost already types every scene as `subject` (a discrete object) or `field` (fills the
## frame), and [Shots] already gives field scenes the gentle moves only. The same rule
## belongs here for the same reason, and the reason is visible: a hard rake reads as
## FORESHORTENING when there is a recognisable object in the panel to be foreshortened, and
## as WARPING when there is not. A fractal field at 30 degrees off the paper does not look
## like a picture seen at an angle, it looks like a broken picture - which is exactly how it
## was reported.
const FIELD_FLATTEN := 0.62

## HOW MUCH PANEL IS UNDER THE AIM: the shape of the falloff, in units of the panel's own
## half-size, measured Chebyshev (so it is the panel's rectangle, not a disc inside it).
##
## THIS IS THE FIX FOR THE WHOLE REPORT, and it is one number turned into a constraint:
## "focusing a zoom-in camera at the corner of a frame is terrible". Everything before this
## tried to make the camera AIM well; nothing stopped it from being TIGHT while aiming
## badly, and tight-while-badly-aimed is the only combination that actually looks broken.
## Wide over a gutter shows four panels and reads as a page. Tight over a gutter shows
## paper.
##
## So content is scored, and the framing is a function of it (see _place_eye). Inside CORE -
## the middle 55% of a panel - the score is 1 and the shot is exactly as tight as the move
## asked for. By EDGE, just inside the panel's border, it is 0 and the shot has opened to
## OFF_FILL. Between them it is smooth, so a pan across a gutter WIDENS as it crosses and
## closes again on the far panel, which is what a camera operator does without being told.
const CONTENT_CORE := 0.55
const CONTENT_EDGE := 0.98
## The framing a shot falls back to when there is no panel under the aim, in frames of panel
## height. Well under 1: at 0.62 the panel the camera is nearest is a bit over half the
## frame, so the gutter it is actually looking at is surrounded by the four panels that
## gutter separates. That is a picture of a comic page. Note the covering solve can still
## refuse to go this wide - see _place_eye - and on a spread it usually does not have to.
const OFF_FILL := 0.62
## How much content a move's RESTING aim must have. Not 1.0: snapping every terminal aim to
## a panel's dead centre would make every settled shot the same shot, and the whole point of
## sampling a station is that it is not. This is "far enough inside the panel that the shot
## is of the panel", and _settle_aim walks the aim in until it clears the bar.
const SETTLE_CONTENT := 0.72


## The spread's attitude for this spread: a real rake on all three axes, plus a slow
## continuous drift so the sheet is never still.
func _choose_spread_look() -> void:
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "comic-spread-look", _spread_i])
	_att_target = Vector3(
		r.randf_range(-PITCH, PITCH),
		r.randf_range(-YAW, YAW),
		r.randf_range(-ROLL, ROLL))
	_att_rate = Vector3(
		r.randf_range(-DRIFT, DRIFT),
		r.randf_range(-DRIFT, DRIFT),
		r.randf_range(-DRIFT, DRIFT) * 0.5)
	# THE SET-UP FOR THIS SPREAD - decided ONCE, and then everything below only varies within
	# it. See the note above AZ_ARC: re-rolling these per shot is what made the camera
	# ping-pong, and no amount of smooth interpolation between two unrelated positions
	# rescues it, because the problem is that they are unrelated.
	var sev := _sev()
	if _spread_i <= 0:
		_az_base = r.randf_range(0.0, TAU)      # the first spread has nowhere to come from
	else:
		# A BOUNDED CHANGE OF SIDE, not a fresh draw. Re-rolling over the whole circle is the
		# largest single move the camera makes; on a spread it happens only every six to
		# twelve cuts, so it is worth having - but at a gentle setting it should be a lean
		# rather than a walk round the table.
		_az_base += r.randf_range(-TURN_AZ, TURN_AZ) * sev
	_el_base = deg_to_rad(r.randf_range(EL_MIN_DEG, EL_MAX_DEG))
	_roll_base = r.randf_range(-CAM_ROLL, CAM_ROLL) * sev
	_lens_fov = r.randf_range(FOV.x, FOV.y)
	# The walk starts each spread where the last one left it, so a page turn is the only
	# discontinuity in the angles and everything between turns is continuous.
	# The sheet's own drift is part of how busy the picture is, so it scales too.
	_att_rate *= sev


## WHERE THE READING IS GOING, in spread coordinates: from the panel being read to the next
## one in the plan. Zero when the plan is finished and the leaf is about to turn.
##
## THIS IS WHAT STOPS THE PING-PONG. The plan has always been in reading order - top-left,
## across, down and back, then the other page - but the MOVES were not. A sweep took whichever
## end of the row was FARTHER from the camera, to maximise travel; a spine traverse crossed to
## the other page whenever it was drawn. So the reading advanced rightward while the shot that
## carried it ran left, and the next cut pulled it back. Reported as "the camera can bounce up
## and down and up and down, or left to right and left to right... like ping-pong all over the
## book". Orienting the open moves along this vector costs them nothing - they still run past
## their subject, they just run past it in the direction the eye is already going.
func _reading_dir() -> Vector2:
	if _spread == null or _step + 1 >= _plan.size() or _read >= _spread.panels.size():
		return Vector2.ZERO
	# THE PLAN ENTRY NEEDS THE SAME GUARD `_read` HAS. A plan is rolled for one spread and the
	# spread can be replaced under it (a gate does exactly that, and so does a turn racing a
	# cut), which leaves an index that was valid for a nine-panel sheet pointing off the end of
	# a seven-panel one.
	var nxt := int(_plan[_step + 1])
	if nxt < 0 or nxt >= _spread.panels.size():
		return Vector2.ZERO
	return _spread.panel_center(nxt) - _spread.panel_center(_read)


## The middle of the page panel [param i] is printed on - what an aim is pulled toward, and
## what a swoop is pushed away from. See AIM_PULL for why this is the page and not the sheet.
func _page_center(i: int) -> Vector2:
	return Vector2(float(_spread.side_of(i)) + 0.5, _spread.aspect * 0.5)


## One step of a bounded random walk: move [param at] by at most ANG_STEP of [param arc] and
## keep it inside +/-arc. A zero arc pins it at zero, which is what severity 0 asks for.
##
## Clamped rather than reflected, deliberately. A walk that bounces off its bound reverses
## direction on contact, which reads as the camera being repelled by an invisible wall; one
## that saturates simply dwells at the edge of the arc until the next step happens to come
## back, which reads as a camera that has settled on one side.
func _walk(at: float, arc: float, r: RandomNumberGenerator) -> float:
	if arc <= 0.0:
		return 0.0
	return clampf(at + r.randf_range(-arc, arc) * ANG_STEP, -arc, arc)


## HOW MUCH PANEL IS UNDER [param aim], 0 to 1. See CONTENT_CORE.
##
## The MAXIMUM over panels rather than a sum: two panels never overlap, so at most one can
## contain the aim, and taking the max means a point just outside a big panel scores by its
## distance from THAT panel rather than being diluted by the eleven it is nowhere near.
func _content_at(aim: Vector2) -> float:
	if _spread == null:
		return 1.0
	var best := 0.0
	for i in _spread.panels.size():
		var q := _spread.panel_uv(i, aim)
		var d := maxf(absf(q.x - 0.5), absf(q.y - 0.5)) * 2.0
		var s := 1.0 - smoothstep(CONTENT_CORE, CONTENT_EDGE, d)
		if s > best:
			best = s
			if best > 0.999:
				break
	return best


## THE SHARE OF A HOLD THIS CAMERA MAY SPEND TRAVELLING. See ARRIVE and ARRIVE_CALM.
func _arrive_frac() -> float:
	return lerpf(ARRIVE_CALM, ARRIVE, clampf(_sev(), 0.0, 1.0))


## HOW WIDE THE SHOT OPENS WHERE THERE IS NO PANEL UNDER THE AIM. See OFF_FILL, and the note
## at the call site in _place_eye for why this is a severity question.
func _open_to() -> float:
	return lerpf(float(_cam.fill), OFF_FILL, lerpf(0.45, 1.0, clampf(_sev(), 0.0, 1.0)))


## WHICH PANEL SETS THE FRAMING SCALE: the one the aim is most inside, and only failing that
## the one whose centre is nearest.
##
## _nearest_panel alone was wrong and measurably so. It compares CENTRES, so an aim sitting
## squarely inside a wide panel can be nearer to the centre of the small panel beside it -
## and then _fit sizes the shot to the small panel, which is a framing computed for
## something that is not in the middle of the picture.
func _framing_panel(aim: Vector2) -> int:
	var best := -1
	var best_s := 0.0
	for i in _spread.panels.size():
		var q := _spread.panel_uv(i, aim)
		var d := maxf(absf(q.x - 0.5), absf(q.y - 0.5)) * 2.0
		var s := 1.0 - smoothstep(CONTENT_CORE, CONTENT_EDGE, d)
		if s > best_s:
			best_s = s
			best = i
	return best if best >= 0 else _nearest_panel(aim)


## Walk an aim in toward its panel until it is far enough inside to be worth stopping on.
##
## PULLED, NOT SNAPPED, and only as far as it has to go - see SETTLE_CONTENT. A move that
## already ends well inside a panel is returned untouched, so this costs the vocabulary
## nothing; a move that ends on a border is walked in until the shot would be of the panel
## rather than of the rule around it.
func _settle_aim(aim: Vector2) -> Vector2:
	if _spread == null:
		return aim
	var i := _framing_panel(aim)
	if i < 0:
		return aim
	var c: Vector2 = _spread.panel_center(i)
	for _step in 8:
		if _content_at(aim) >= SETTLE_CONTENT:
			break
		aim = aim.lerp(c, 0.35)
	return aim


## THE PANELS A SWEEP MAY TRAVEL BETWEEN: the leftmost and rightmost panel centres that
## share a row with `panel`, so a horizontal pan starts on content and ends on content.
## Falls back to the panel itself when it is alone on its row - a sweep with nowhere to go
## becomes a shot that sits, which is better than one that pans across the margin.
##
## ON A SPREAD THIS CROSSES THE SPINE WHENEVER THE ROWS LINE UP, for free: spread
## coordinates run 0..2 and this scans every panel of both pages, so a row band that catches
## panels on each side returns a span that is most of the sheet wide. That is the whole
## "greater spans of content to pan the camera across".
func _row_span(panel: int) -> Vector2:
	var c := _spread.panel_center(panel)
	var lo := c.x
	var hi := c.x
	for i in _spread.panels.size():
		var o := _spread.panel_center(i)
		if absf(o.y - c.y) < _spread.aspect * 0.12:
			lo = minf(lo, o.x)
			hi = maxf(hi, o.x)
	return Vector2(lo, hi) if hi - lo > 0.02 else Vector2(c.x, c.x)


## The same down a column. The tolerance is in one PAGE's width (the spread is two), so a
## column is a column of one page and not a diagonal across the spine.
func _col_span(panel: int) -> Vector2:
	var c := _spread.panel_center(panel)
	var lo := c.y
	var hi := c.y
	for i in _spread.panels.size():
		var o := _spread.panel_center(i)
		if absf(o.x - c.x) < 0.12:
			lo = minf(lo, o.y)
			hi = maxf(hi, o.y)
	return Vector2(lo, hi) if hi - lo > 0.02 else Vector2(c.y, c.y)


## The panel of page [param side] whose centre is nearest the row [param y] - the two ends
## of a `spine` traverse. Returns -1 when that page has nothing (which cannot happen, but a
## caller that indexes -1 is a crash and a caller that checks is not).
func _panel_on_side(side: int, y: float) -> int:
	var best := -1
	var best_d := INF
	for i in _spread.panels.size():
		if _spread.side_of(i) != side:
			continue
		var d := absf(_spread.panel_center(i).y - y)
		if d < best_d:
			best_d = d
			best = i
	return best


<<<<<<< Updated upstream
# --- the camera: one target, one follower ------------------------------------
#
# THE CAMERA STATE IS NEVER ASSIGNED, ONLY INTEGRATED. That sentence is the whole design, and
# everything below exists to keep it true. `_cam` follows `_tgt` exponentially, every frame,
# forever; the single exception is _cut_to, which is the only function in this file that
# writes `_cam` outright and is therefore the only place a discontinuity can come from.
#
# WHAT THIS REPLACES AND WHY. A move used to be a pair of poses, a duration and an ease curve,
# with `_ease` interpolating between them. That made continuity a CONVENTION that thirteen
# branches were each expected to honour - and at least two of them did not, while the comment
# above them asserted that none did. It also meant "is the camera holding" was a question about
# which of thirteen names was active, so every fix had to be restated for each. Four rounds of
# guards were spent on defects that were all one defect: nothing owned how much the picture may
# change in a frame.
#
# THE OTHER HALF OF THE SAME BUG WAS BELOW THE MOVE LAYER. _place_eye derives the rake from the
# panel being READ, the scale from whichever panel the aim is inside, and the distance from a
# solve that changes branch at the film panel's edge. Those are step functions no move-layer
# guard can reach - measured at up to 21.7 degrees of elevation in a single frame - so `flat`
# and the eye distance are follower channels here too.
#
# AND SPEED IS A RATE, NOT A DURATION. The old `_budget` sampled a duration and then capped it
# at a share of the Director's hold, so on a long hold the SAMPLE was the binding constraint
# and the slider stopped mattering: at scene hold 4.0 a "gentle" drift ran 27 s and chained
# another. There is no duration here. A shot has a target and a closing rate, it arrives when
# it arrives, and a cut that interrupts it is graceful because the follower simply retargets.
=======
## Pick and plan the next move. Called on every reading advance and on every page turn.
func _choose_move(turned := false) -> void:
	# HAS THE CHAIN ALREADY BROUGHT US HERE? Between cuts the chain travels ahead to the panel
	# the next cut will land on (see _chain_move), so by the time that cut arrives the camera is
	# often already looking at it. Planning a fresh shot then can draw a HARD move out of the
	# bag and jump-cut from the panel to the same panel, a few degrees round - "the camera
	# jump-cuts unexpectedly... to the exact same frame of the exact same page, from a slightly
	# different angle. That should never happen." Quite so: there is nothing to cut TO. Treating
	# it as a chain keeps the move gentle and keeps the framing it already had.
	var arrived := _chain_to == _read
	_chain = 0
	_chain_to = -1
	# NOTE _zoom_dir IS NOT CLEARED HERE. A cut is not permission to reverse the framing - see
	# the reversal guard in _plan_move. Only a settle or a breathe clears it.
	# HOW RESTLESS THIS PARTICULAR SHOT IS, rolled once and then spent - see CHAIN_TRAVEL.
	var r := RandomNumberGenerator.new()
	r.seed = hash([Director.session_seed(), "comic-travel", _spread_i, _read])
	_travel = 0
	# THE CUT'S WALL-CLOCK ALLOWANCE, opened here and spent by every travelling move it plans.
	# See _travel_left.
	_travel_left = maxf(ARRIVE_MIN, Director.hold_remaining() * _arrive_frac())
	_travel_max = CHAIN_TRAVEL.x
	if r.randf() < CHAIN_RESTLESS * _sev():
		_travel_max = CHAIN_TRAVEL.y
	var force := "establish" if (turned and r.randf() < ESTABLISH) else ""
	_plan_move(hash([Director.session_seed(), "comic-move", _spread_i, _read]), _read,
		"chain" if arrived else "", force)
>>>>>>> Stashed changes


## WHAT SHARE OF THE HOLD A SHOT SPENDS ARRIVING, at camera 0 and at camera 2.
##
## THE RATE IS DERIVED FROM THIS, NOT SET DIRECTLY, and that is the second time this file has
## had to learn the same lesson. A fixed rate makes a calm camera CLOSE SLOWLY - but the
## distance it has to close is the gap between two panels, which the reading plan sets and the
## slider does not touch. So a slow rate means a long journey between panels and a short hold
## at the end of it: measured with cuts running, 53% held at camera 0 against 82% at camera 2,
## the calm end holding LEAST. That is the same inversion as the old `_budget`, arriving by a
## different road, and the cure is the same: express the intent (how much of the scene is
## spent looking rather than travelling) and let the rate fall out of it.
##
## THE RATE ACCOUNTS FOR HOW FAR THE SHOT HAS TO GO. An exponential follower closes a gap of G
## down to the arrival threshold in ln(G)/rate seconds, so the rate that arrives in a chosen
## time is ln(G) / t - NOT the 3/t that "three time constants" suggests, which silently assumes
## every shot starts the same distance out. A shot crossing the spread starts an order of
## magnitude further out than one nudging within a panel, and with a fixed 3/t the far ones
## simply took longer: measured 12.2 s mean arrival against a 9 s ceiling.
const ARRIVE_SHARE_CALM := 0.22
const ARRIVE_SHARE_BUSY := 0.55
## ...BOUNDED AT BOTH ENDS, in seconds of approach. Both bounds are load-bearing and the
## missing one was a real defect: with the hold slider at 4.0 the room in front of a shot is
## enormous, so `share` of it is enormous, so the derived rate went to nearly zero and the
## follower crawled - measured, NO SHOT EVER ARRIVED in a 60 s run, and the hold phase
## therefore never began at all. That is the old `_budget` inversion for the third time, hiding
## inside a rate instead of a duration: any quantity derived from the Director's hold has to be
## clamped, because the hold is a user setting with no upper bound.
##
## The floor keeps an arrival from being a snap on a very short hold; the shot simply runs past
## the cut, which costs nothing because a cut is a new target rather than an interruption.
## Over what fraction of the journey the approach eases out. Above it the camera travels at a
## constant rate; below it, it lands. Small, because the taper is a landing, not the shot.
const ARRIVE_TAPER := 0.28
const ARRIVE_FLOOR := 1.8
## 10, not 17. A pan may be unhurried and still has to LAND: at 17 a shot spent fourteen
## seconds of a twenty-second scene travelling and held for 3% of it. The variety comes from
## PAN_SPEED sampling inside this bound, not from the bound itself being generous.
const ARRIVE_CEIL := 10.0
## HOW FAST THE CAMERA CROSSES THE PAPER, in page widths per second, sampled per shot.
##
## A PAN HAS A SPEED, NOT A DURATION, and that distinction is the whole of "the transitions all
## seem rather fast... I would expect more gentle, drifting camera pans". Sizing every arrival
## as a share of the hold gives a long journey and a short nudge the SAME eight seconds - so
## the long ones, which are the ones that should read as unhurried drifts across the spread,
## are the fastest thing on screen. Dividing the distance by a speed inverts that correctly:
## crossing the spine takes longer than easing across a gutter, because it is further.
##
## The range is wide on purpose. Some transitions should be quick; the complaint was that they
## ALL were, and a single tuned number cannot express "mostly gentle, occasionally not".
const PAN_SPEED := Vector2(0.028, 0.115)
## The angles follow slower than the aim. A pan that changes the rake at the same speed as the
## framing reads as the whole page swinging; letting the rake lag makes it read as a camera
## craning while it travels, which is the same motion done by something with mass.
const RATE_ANGLE := 0.62
## HOW BIG A SHOT'S JOURNEY IS, at camera 0 and camera 2, multiplying the `aim` and `az`
## distances in SHOTS and the width of its `fill` band.
##
## THIS IS WHAT MAKES THE SLIDER MONOTONE IN HOLDING, and the old design had it backwards.
## Scaling only the SPEED means a gentle camera makes the same journeys more slowly, so it
## spends MORE of the hold travelling and less of it arrived - measured on the previous build
## at 71% held at camera 0 against 86% at camera 1, which is why turning the slider down never
## helped. A gentle camera makes SMALL journeys: it arrives early and looks for a long time.
const REACH_CALM := 0.35
const REACH_BUSY := 1.35
## THE SHOT HAS ARRIVED when its remaining gap is under this, measured in the mixed units of
## _pose_gap: page widths of aim, log-ratio of framing, radians of swing. One is the threshold
## because _pose_gap divides each channel by its own tolerance first.
const ARRIVED := 1.0
## The tolerances _pose_gap normalises by: aim in page widths, framing as a log ratio (0.06 is
## about 6%), swing in radians.
const GAP_AIM := 0.035
const GAP_FILL := 0.06
const GAP_ANG := 0.035
## A shot that cannot converge - because the Director is holding it against a target it will
## not reach, or because a constraint is fighting it - is treated as arrived after this long,
## so the hold can begin. Without it a pathological shot never holds and never says why.
## The safety net for a shot that cannot converge - a target it will not reach, or a constraint
## fighting it. It must sit ABOVE any legitimate arrival or it stops being a net and starts
## being the schedule; see the note on the taper floor in _follow.
const ARRIVE_CAP := 22.0
## HOW LONG THE CAMERA MUST HAVE HELD before it may reverse the framing.
##
## Clearing the direction the instant a shot ARRIVES is not enough: a shot can arrive and be
## cut a moment later, and the next shot is then free to reverse with nothing in between -
## measured at 3 to 6 reversals per run with no hold separating them. The report asked for a
## hold, not for a touch: "no reversing direction allowed, without first HOLDING at that
## position first."
const ZOOM_FREE := 1.6
## THE HELD SHOT STILL BREATHES. How far the target creeps during a hold, in page widths, and
## how long it takes to creep the whole way. One heading, chosen when the shot arrives.
##
## A held shot that is perfectly static reads as a photograph of a page rather than a camera
## looking at one, and the original report on this vehicle was a camera that froze. A heading
## rather than a walk: an unbiased random walk against the panel edge parks AT the edge and
## jitters there, which is a defect this file has shipped once already.
const HOLD_CREEP := 0.05
const HOLD_CREEP_TIME := 22.0
## WHAT MAKES A JUMP CUT WORTH MAKING: it must land on a DIFFERENT panel, and move the aim at
## least this far. A cut is a change of SUBJECT; scale is how it is photographed, not what it
## is of. Measured on the build this replaced, four of every five discontinuities landed on the
## panel the camera was already framing - "the camera jumps to correct its position... landing
## on the exact same frame."
const CUT_MIN_AIM := 0.28
## How fast `flat` and the eye distance follow their targets, per second. These are the two
## quantities in _place_eye that STEP - see the note at the top of this section.
const RATE_FLAT := 1.3
const RATE_DIST := 2.4


## The severity in force, 0..2. Asked of the Director rather than held here so the slider takes
## effect on the next frame without anything having to be pushed at the vehicle.
func _sev() -> float:
	return clampf(Director.camera, Director.CAMERA_MIN, Director.CAMERA_MAX)


## How far a shot travels, and how fast, at the current setting. See REACH_CALM and RATE_CALM.
func _reach() -> float:
	return lerpf(REACH_CALM, REACH_BUSY, clampf(_sev(), 0.0, 2.0) * 0.5)


## The rate that arrives inside this camera's share of the hold in front of it. See
## ARRIVE_SHARE_CALM.
## How long this shot should take to arrive: the distance it has to cross, at a speed sampled
## for this shot. See PAN_SPEED. The Director's hold is a CEILING on that, not the source of
## it - a shot may not spend more than its share of the scene travelling, but within that it
## goes at its own pace.
func _arrive_time(r: RandomNumberGenerator, travel: float, kind: String) -> float:
	# BIASED TOWARD THE SLOW END, and the bias has to be applied to the SPEED to come out right
	# in the duration. Time is distance over speed, so sampling speed uniformly produces mostly
	# SHORT transitions - the fast half of the speed range covers a narrow band of durations
	# while the slow half spreads over a wide one. Squaring the draw pushes the sample toward
	# the gentle end, which is the shape asked for: "it's okay if some of them are fast... but I
	# wouldn't expect them all to be that way."
	var u := r.randf()
	var speed := lerpf(PAN_SPEED.x, PAN_SPEED.y, u * u)
	# THE TABLE'S `rate` MULTIPLIES THE SPEED. It used to DIVIDE the follower rate, which made
	# every entry mean its opposite: `whip` at 3.2 was the slowest thing in the bag and `orbit`
	# at 0.55 the quickest.
	speed *= float(SHOTS[kind].get("rate", 1.0)) * lerpf(0.7, 1.7, clampf(_sev(), 0.0, 2.0) * 0.5)
	var want := travel / maxf(speed, 0.001)
	var share := lerpf(ARRIVE_SHARE_CALM, ARRIVE_SHARE_BUSY, clampf(_sev(), 0.0, 2.0) * 0.5)
	var room := Director.hold_remaining() * share
	return clampf(minf(want, maxf(room, ARRIVE_FLOOR)), ARRIVE_FLOOR, ARRIVE_CEIL)


## BEGIN A SHOT ON [param panel]. This is the only entry point for "look at something else":
## the Director's cut calls it, and so does a page turn.
func _begin_shot(panel: int, turned := false) -> void:
	if _spread == null or _spread.panels.is_empty():
		return
	panel = clampi(panel, 0, _spread.panels.size() - 1)
	var r := RandomNumberGenerator.new()
<<<<<<< Updated upstream
	r.seed = hash([Director.session_seed(), "comic-shot", _spread_i, panel, _shot_n])
	_shot_n += 1
	var kind := _pick_shot(r)
	# A ZOOM MAY BE CONTINUED BUT NEVER REVERSED WITHOUT A HOLD BETWEEN. `_zoom_dir` is cleared
	# when a shot arrives, so this asks exactly the question the report asked: has the camera
	# held since it last changed the framing? "No reversing direction allowed, without first
	# holding at that position first."
	if (kind == "pull" and _zoom_dir > 0) or (kind == "push" and _zoom_dir < 0):
		kind = "pan"
	var tgt := _target_for(r, panel, kind)
	var hard := bool(SHOTS[kind].get("hard", false)) and _worth_cutting(tgt)
	# THE HEADING THE HOLD WILL CREEP ALONG, drawn once here rather than re-drawn per breath.
	var hd := Vector2(r.randf_range(-1.0, 1.0), r.randf_range(-1.0, 1.0))
	_shot = {
		"panel": panel,
=======
	r.seed = key
	# THE TARGET IS CHOSEN ONCE PER CUT AND THEN HELD.
	#
	# It used to be re-decided on every chain, from where the camera HAPPENED TO BE: on content
	# it looked ahead to the next panel, off content it "recovered" to the one being read. Those
	# two branches fight. The camera sets off for the panel ahead, crosses a gutter on the way,
	# the content score dips under the threshold, and the next chain reads that as being lost
	# and sends it back - whereupon it is on content again and the chain sends it forward.
	# Reported as the camera "bouncing between two diagonal corners... clearly because of some
	# kind of repulsion mechanism", and the diagnosis that came with it is the right one: "the
	# camera should have never selected the top-right frame again, in the first place." There is
	# no repulsion here and never was; there was a destination being recomputed from a position
	# that the journey itself was changing.
	var ahead := int(_plan[_step + 1]) if _step + 1 < _plan.size() else -1
	if _chain_to < 0:
		_chain_to = _read
	# HAS THIS CUT DONE ITS TRAVELLING? See CHAIN_TRAVEL. Everything below is about WHERE to
	# go next, and none of it applies once the answer is "nowhere" - a shot that has spent its
	# budget, or one whose target is the panel it is already framing and has no panel ahead,
	# has nothing left to travel to, and planning a move anyway is the re-correction the user
	# has now reported three times. It holds instead, and the hold extends itself.
	var lost := _content_at(_clamp_aim(_cam.aim)) < CHAIN_LOST
	var going := (_chain_to != _read) or lost or (ahead >= 0)
	if not going or _travel >= _travel_max or _travel_left <= 0.0 \
			or Director.hold_remaining() < SETTLE_ROOM:
		_plan_move(key, _chain_to, "chain", "settle")
		return
	_travel += 1
	if _chain_to == _read and ahead >= 0 and not lost:
		# FORWARD ONLY, and only ever this one step: from the panel being read to the panel the
		# next cut will land on. Monotone, so an oscillation is not merely unlikely, it cannot
		# be expressed - which is the property that matters, because the last version was also
		# not MEANT to oscillate.
		#
		# The alternative - freezing the target outright at the first chain - does stop the
		# bouncing, and it also stops the camera: committing to the panel it is already on
		# leaves nothing to travel to, and the aim measured 0.0096 page widths a second, which
		# is a picture that has stopped. A comic is read once through, so forward is the only
		# direction there is; the fix is to take it, not to stand still.
		_chain_to = ahead
	_plan_move(key, _chain_to, "chain")


## Plan a move to [param panel], seeded from [param key]. [param tag] is empty for a cut's
## move and "chain" for a continuation, which is both the log line and the filter on the
## bag - see the `chain` flag in MOVES.
## The held shot, continued. Aimed at whatever panel is actually in frame rather than at the
## one being read, because a settle can land the camera on a neighbour and a breathe should
## carry on looking at what is on screen.
func _breathe() -> void:
	if _spread == null:
		return
	_chain += 1
	_plan_move(hash([Director.session_seed(), "comic-breathe", _spread_i, _read, _chain]),
		_framing_panel(_clamp_aim(_cam.aim)), "chain", "breathe")


func _plan_move(key: int, panel: int, tag: String, force := "") -> void:
	if _spread == null or _spread.panels.is_empty():
		return
	# CLAMPED ONCE, HERE. _station already did this for itself, which hid the fact that the
	# match branches below index the spread with the RAW value - `side_of`, `panel_center`,
	# `_panel_world`. Every caller derives the panel from a plan or a chain target, and both of
	# those can outlive the spread they were rolled against.
	panel = clampi(panel, 0, _spread.panels.size() - 1)
	var r := RandomNumberGenerator.new()
	r.seed = key
	var kind := force if MOVES.has(force) else _pick_move(r, tag == "chain")
	# A ZOOM MAY BE CONTINUED BUT NEVER REVERSED, and the guard is NOT limited to chains.
	# It was, and that was the hole: a chain pulled out, the Director cut, and the cut - a
	# fresh shot, so the guard did not apply - drew a push. "The camera is pulling gently
	# outward, only to IMMEDIATELY start pushing in again at its apex... no reversing
	# direction allowed, without first HOLDING at that position first." So the direction now
	# outlives the cut and is cleared by exactly the thing that was asked for: a settle or a
	# breathe. Until the camera has held, it may not change its mind.
	if (kind == "pull" and _zoom_dir > 0) or (kind == "push" and _zoom_dir < 0):
		kind = "drift"
	var b := _station(r, panel)
	# A JUMP THAT CHANGES NOTHING IS NOT AN EDIT - see CUT_MIN_AIM. Checked HERE, against the
	# station the jump would land on, because these two are the only moves that teleport onto
	# their own destination (`a` is `b`); a swoop teleports somewhere else entirely and is
	# always a visible change.
	if (kind == "cut" or kind == "dip") and not _worth_cutting(b):
		kind = "drift"
	var spec: Dictionary = MOVES[kind]
	# FRAMING CARRIES ACROSS A CUT - see FILL_CARRY. Not across a chain, which already inherits
	# it outright below; not into the two moves whose whole gesture is the zoom; and not into an
	# establish or a breathe, both of which set their own distance.
	if tag != "chain" and kind != "push" and kind != "pull" and kind != "establish":
		b["fill"] = lerpf(float(b.fill), float(_cam.fill),
			r.randf_range(FILL_CARRY.x, FILL_CARRY.y))
	# CONTINUITY BY DEFAULT: a move starts from wherever the camera actually is, so the
	# picture never jumps unless the move is one whose whole point is that it jumps.
	#
	# AND NOTHING BELOW MAY OVERWRITE a["aim"] EXCEPT A DECLARED-HARD MOVE. Four of them did:
	# sweep_h and sweep_v snapped the aim to the end of a row or column, and orbit and pull
	# snapped it onto the target panel, all while declaring hard: false. Since _mv_t restarts
	# at zero, the first frame of such a move teleports the picture - measured at a mean of
	# 0.48 to 0.62 page widths, on 100% of orbits and pulls, taking the real jump-cut rate to
	# 38.7% of shots against the "about one shot in six" this table claims. They now aim
	# their DESTINATION instead, which keeps every gesture and starts all of them from the
	# picture that is already on screen.
	var a: Dictionary = b.duplicate() if bool(spec.get("hard", false)) else _cam.duplicate()
	var row: float = _spread.panel_center(panel).y
	var col: float = _spread.panel_center(panel).x

	match kind:
		"track":
			# A constant station passing THROUGH the subject and out the other side - the
			# shot that follows a car down a road. The direction is where the camera has
			# been travelling from, so the reading keeps its momentum.
			# ALONG THE READING where there is one, so the overrun carries on toward the next
			# panel instead of past the subject in some unrelated direction.
			var away: Vector2 = _reading_dir()
			if away.length() < 0.05:
				away = (b.aim - a.aim)
			if away.length() < 0.05:
				away = Vector2(1.0, 0.0).rotated(r.randf_range(-0.5, 0.5))
			b["aim"] = b.aim + away.normalized() * TRACK_OVERRUN
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"spine":
			# ACROSS THE SPINE. A panel on one page to a panel on the other at the same
			# height, station held, so it is a pure lateral pan over the longest run of
			# uninterrupted content the sheet has. Which way it goes follows which page the
			# aim is on, so the pan moves AWAY from where the camera already is.
			# ONLY WHEN THE READING IS ACTUALLY CROSSING. A traverse over the spine is the
			# spread's best move and its worst one: taken when the next panel is on the far
			# page it IS the turn of the eye from one page to the other, and taken at any
			# other moment it is the camera leaving the story behind and coming back.
			var far_side := 1 if float(a.aim.x) < ComicSpread.SPINE else 0
			var nxt := int(_plan[_step + 1]) if _step + 1 < _plan.size() else -1
			var to_i := nxt if nxt >= 0 and _spread.side_of(nxt) == far_side \
				else _panel_on_side(far_side, row)
			if to_i >= 0 and (nxt < 0 or _spread.side_of(nxt) == far_side):
				b["aim"] = _spread.panel_center(to_i)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"sweep_h":
			# ACROSS THE ROW, PANEL TO PANEL - not across the SHEET. It used to run from
			# -0.10 to 1.10 in page coordinates, which is a pan that begins and ends on the
			# desk and spends much of its length over margin and gutter: "tracking along
			# unfocused edges". A sweep is a pan over CONTENT or it is nothing.
			# THE END THE READING IS HEADED FOR - see _reading_dir. Only when the reading has
			# nowhere left to go does it fall back to the farther end.
			var span_h := _row_span(panel)
			var go_x := _reading_dir().x
			var end_x: float = span_h.y if go_x > 0.0 else span_h.x
			if is_zero_approx(go_x):
				end_x = span_h.y if absf(span_h.y - float(a.aim.x)) \
					>= absf(span_h.x - float(a.aim.x)) else span_h.x
			b["aim"] = Vector2(end_x, row)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"sweep_v":
			var span_v := _col_span(panel)
			var go_y := _reading_dir().y
			var end_y: float = span_v.y if go_y > 0.0 else span_v.x
			if is_zero_approx(go_y):
				end_y = span_v.y if absf(span_v.y - float(a.aim.y)) \
					>= absf(span_v.x - float(a.aim.y)) else span_v.x
			b["aim"] = Vector2(col, end_y)
			for k in ["az", "el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"push":
			# IT ENDS ON THE PANEL. It used to end at a.aim.lerp(b.aim, 0.5) - the MIDPOINT
			# between wherever the camera happened to be and the panel it was going to -
			# while pushing in to between 2.2 and 3.8 frames of panel height. A midpoint
			# between two panels is a gutter, and a gutter at 3.8 fill is the reported
			# defect exactly: "in the corner of a frame and fully zoomed-in... shows almost
			# nothing". A push-in ends ON its subject. That is what a push-in is.
			# ...and the other one. A push to 3.8 frames of panel height is the tightest the
			# camera ever gets; at a gentle setting it barely pushes past the panel at all.
			b["fill"] = lerpf(1.0, r.randf_range(PUSH_FILL.x, PUSH_FILL.y), _sev() * 0.5)
			for k in ["az", "el", "roll"]:
				b[k] = a[k]
		"pull":
			# OUT FROM WHERE THE CAMERA ALREADY IS. It used to set a["fill"] to a deep push
			# value - and `a` is the move's FIRST frame, so on a move declared hard:false that
			# teleported the zoom in and then eased back out. "The camera quickly zooms and
			# resets, which is a problem I thought we had fixed": the chained-move fix stopped
			# the zoom being RE-ROLLED, and this is the other way it moved without being asked.
			# ...AND IT SCALES WITH SEVERITY, WHICH ITS OPPOSITE ALREADY DID. `push` reads the
			# slider (it lerps toward the deep framing by _sev() * 0.5) and this did not, so at
			# camera 0 - where _station pins the fill at the middle of FILL, 1.04 - a pull
			# still took it to 0.57 and the framing swung 1.8x between two fixed values with
			# nothing in between: "the camera is still zooming in/out at random, with hard
			# peaks at each end", at the setting that is supposed to be the calm one.
			var deep := maxf(FILL.x * 0.6, float(a["fill"]) * PULL_OUT)
			b["fill"] = lerpf(float(a["fill"]), deep,
				lerpf(0.35, 1.0, clampf(_sev(), 0.0, 1.0)))
			# NEVER OUT PAST OFF_FILL. Below it the content coupling INVERTS - `eff` lerps FROM
			# the open framing TOWARD the requested one, so a request under it means less
			# content renders as a TIGHTER shot, and a planned pull comes out as a push
			# wherever the aim is off a panel.
			b["fill"] = maxf(float(b.fill), OFF_FILL)
			for k in ["az", "el", "roll"]:
				b[k] = a[k]
		"orbit":
			# The widest gesture in the bag - up to 149 degrees at the default - so it is one
			# of the two the severity knob pulls in hardest.
			b["az"] = a.az + r.randf_range(ORBIT_ARC.x, ORBIT_ARC.y) * _sev() \
				* (1.0 if r.randf() < 0.5 else -1.0)
			for k in ["el", "roll", "fill", "fov"]:
				b[k] = a[k]
		"breathe":
			# A HELD SHOT THAT IS STILL A SHOT: the same camera, crept a little further along
			# the heading this hold was given. See BREATHE_AIM.
			#
			# MEASURED FROM WHERE THE SHOT ARRIVED, not from where the last breathe left off,
			# so the excursion is BOUNDED however many breathes a long hold fits. A chain of
			# relative nudges is a random walk, and this is the same defect as the compounding
			# framing by another route.
			var amp := BREATHE_AIM * lerpf(BREATHE_CALM, 1.0, clampf(_sev(), 0.0, 1.0))
			var reach := clampf(float(_chain - _hold_chain) * 0.35, 0.0, 1.0)
			# A breathe cannot be the first move of a hold - _ease only reaches it from a
			# settle or another breathe, and a settle always sets this - but a zero here would
			# fling the aim at the corner of the sheet, which is not a failure worth risking.
			var base: Vector2 = _hold_aim if _hold_aim != Vector2.ZERO else a.aim
			b["aim"] = base + _hold_dir * amp * reach
			# THE DISTANCE DOES NOT MOVE AT ALL. A held shot that changes its framing is not
			# held, and one that changes it by a fixed FRACTION every time is a ratchet.
			for k in ["el", "roll", "fov", "fill"]:
				b[k] = a[k]
			b["az"] = float(a.az) + BREATHE_AZ * amp / maxf(BREATHE_AIM, 0.001) \
				* (1.0 if _hold_dir.x >= 0.0 else -1.0) * reach
		"cut", "dip":
			# LAND, THEN BREATHE. `a` is this station (the hard branch below teleports the
			# camera onto it) and `b` is a little off it, so the shot after the edit is a shot
			# and not a still. See the note in MOVES.
			var off := Vector2(r.randf_range(-1.0, 1.0), r.randf_range(-1.0, 1.0))
			b["aim"] = b.aim + (off.normalized() if off.length() > 0.001 else Vector2.RIGHT) \
				* BREATHE_AIM * lerpf(BREATHE_CALM, 1.0, clampf(_sev(), 0.0, 1.0))
		"establish":
			# THE OPEN BOOK FIRST, THEN THE PANEL. A turn planned an ordinary shot on its
			# first planned panel, and that panel is the film panel seven times in ten (see
			# FILM_FIRST) - so the page arrived with the camera already on her face: "the page
			# turns and the camera immediately zooms to the video with the girl. You would not
			# expect that; instead, you would expect to reveal the scene first, the open comic
			# book, then to move to the girl after. Like a short delay, or a decision to focus
			# there."
			#
			# Starting wide on the whole sheet costs no continuity, because the leaf flip is
			# already the one discontinuity the grammar allows (see _choose_spread_look), and
			# it buys the decision: the shot opens on the spread and then goes to her.
			a["aim"] = _spread.center()
			a["fill"] = ESTABLISH_FILL
		"swoop":
			# In off the FAR SIDE of the spread, low and wide, arriving on the subject.
			a["az"] = b.az + PI
			a["el"] = deg_to_rad(r.randf_range(SWOOP_EL_DEG.x, SWOOP_EL_DEG.y))
			a["fill"] = SWOOP_FILL
			var mid := _page_center(panel)
			a["aim"] = b.aim + (b.aim - mid).normalized() * 0.5 \
				if b.aim.distance_to(mid) > 0.02 else b.aim

	if kind == "settle":
		# THE HOLD'S HEADING, decided here and spent by the breathes that follow. One draw per
		# arrival - see BREATHE_AIM for what happened when every breathe drew its own.
		var hd := Vector2(r.randf_range(-1.0, 1.0), r.randf_range(-1.0, 1.0))
		_hold_dir = hd.normalized() if hd.length() > 0.001 else Vector2.RIGHT
		_hold_chain = _chain
		# STAY WHERE THE LAST MOVE LANDED. Only the framing eases the last of the way in, so
		# the shot resolves instead of being abandoned mid-convergence.
		b["aim"] = a.aim.lerp(b.aim, SETTLE_DRIFT)
		# FILL INCLUDED, and leaving it out was a bug you could see. A settle exists to hold
		# the shot still while the cut arrives; copying every part of the camera EXCEPT the
		# zoom meant it held the angle and then eased to a freshly sampled distance, so the
		# picture crept in or out by a few percent for no reason at all.
		for k in ["az", "el", "roll", "fov", "fill"]:
			b[k] = a[k]
		_hold_aim = b.aim

	# A CHAIN DOES NOT RE-ROLL THE ZOOM. Every move takes its target framing from a fresh
	# _station, which samples `fill` - fine at a CUT, where a new shot is the point, and wrong
	# between cuts, where the chain is meant to be the same shot continuing. Two chains in a
	# row on one panel therefore nudged the distance one way and then the other: "it zoomed-in
	# then immediately zoomed-out again, but barely... it would have been better to just remain
	# stable, or to zoom and keep zooming". The two moves whose whole gesture IS the zoom keep
	# theirs; everything else inherits what the camera already had.
	if tag == "chain" and kind != "push" and kind != "pull":
		b["fill"] = a["fill"]

	# THE SETTLE GUARD, and it is a guard rather than a rule so that it survives whatever
	# gets added to the bag next. Any move that is allowed to STOP must stop somewhere worth
	# looking at; the ones marked `open` are exempt because running past the subject is
	# their whole gesture, and _chain_move is what makes that safe.
	if not bool(spec.get("open", false)):
		b["aim"] = _settle_aim(_clamp_aim(b.aim))
	else:
		# An open move keeps its overrun, but the overrun lands inside the printed area rather
		# than out on the trim - see _printed_bounds. Clamping it HERE as well as at placement
		# matters: the move interpolates toward `b`, so an unclamped `b` drags the whole
		# journey toward the margin even though the placement clamp hides the last of it.
		b["aim"] = _clamp_aim(b.aim)
		# ...AND AN OVERRUN THAT LANDS ON NOTHING IS NOT AN OVERRUN, it is the camera adrift:
		# "the camera jumps four times and drifts to an edge where there is nothing to see."
		# The printed-area clamp keeps the aim off the desk but says nothing about gutters, and
		# a track that runs 0.45 page widths past its subject can come to rest in one. The
		# gesture survives - the move still runs past what it was looking at - but where it
		# STOPS has a picture in it.
		if _content_at(b.aim) < CHAIN_LOST:
			b["aim"] = _settle_aim(b.aim)

	_mv = {
>>>>>>> Stashed changes
		"kind": kind,
		# `secs` and `gap0` are filled in below, once the target exists to measure against.
		"arrived": false,
		"heading": hd.normalized() if hd.length() > 0.001 else Vector2.RIGHT,
	}
<<<<<<< Updated upstream
	_shot_t = 0.0
	_hold_t = 0.0
	_tgt = tgt
	# THE RATE IS SIZED TO THE JOURNEY, and can only be measured once the target is set: how far
	# this shot has to go, at the speed it drew, converted to the follower rate that gets there.
	var travel: float = (_cam.aim as Vector2).distance_to(tgt.aim) \
		+ absf(log(maxf(float(tgt.fill), 0.01) / maxf(float(_cam.fill), 0.01))) * 0.25
	_shot["secs"] = _arrive_time(r, travel, kind)
	_shot["gap0"] = maxf(_pose_gap(), 0.001)
	if hard:
		_cut_to(tgt)
=======
	# SPENT BY TRAVEL ONLY. A settle and a breathe are the shot being HELD, not the shot going
	# anywhere, so they cost nothing and cannot exhaust the allowance they are the reward for.
	if kind != "settle" and kind != "breathe":
		_travel_left -= float(_mv.dur)
	_mv_t = 0.0
	# WHICH WAY THE FRAMING IS GOING, for the reversal guard above. Held across the chains of
	# one cut and cleared by the next - a new shot is allowed to change its mind, a
	# continuation of the same shot is not.
>>>>>>> Stashed changes
	if kind == "push":
		_zoom_dir = 1
	elif kind == "pull":
		_zoom_dir = -1
<<<<<<< Updated upstream
=======
	elif kind == "settle":
		# THE HOLD THE USER ASKED FOR, spent. Having stopped somewhere, the camera is free to
		# change its mind about the distance again.
		_zoom_dir = 0
	if bool(spec.get("hard", false)):
		_cam = a.duplicate()          # the jump itself
		_snap = true                  # ...and the placed pose is allowed to jump with it
	elif kind == "establish":
		# The one non-hard move that starts somewhere else on purpose. It overwrites `a`, so
		# its first frame is a teleport whatever the table says; letting the placed pose ease
		# into that would show the camera sliding to the wide station instead of opening on it.
		_snap = true
	_dip_t = 0.0 if kind == "dip" else -1.0
>>>>>>> Stashed changes
	if _spread_i >= 0:
		print("ghost: comic %s%s -> panel %d of spread %d" % [
			kind, (" (cut)" if hard else ""), panel + 1, _spread_i])


<<<<<<< Updated upstream
## THE ONLY PLACE THIS FILE ASSIGNS THE CAMERA. Everything else integrates toward a target, so
## the number of discontinuities in a session is exactly the number of calls to this - which is
## a property, not a hope, and is what tests/comic_motion_check.gd counts.
func _cut_to(pose: Dictionary) -> void:
	_cam = pose.duplicate()
	_snap = true                  # the placed rake and distance jump with it, or it tears
	_dip_t = 0.0 if _shot.get("kind", "") == "dip" else -1.0
=======
## Fit [param want] seconds of move into the time the Director expects to have left. See
## ARRIVE. A `settle` is the exception: it is not travelling anywhere, so it is sized to FILL
## the remaining hold rather than to finish inside a fraction of it.
func _budget(want: float, kind: String, panel := -1) -> float:
	var room := Director.hold_remaining()
	if kind == "settle":
		# A SETTLE FILLS THE HOLD rather than finishing inside a fraction of it, and it is
		# EXTENDED rather than re-planned when it runs out (see _ease), so this is a starting
		# length and not a promise.
		return clampf(room, float(MOVES["settle"]["dur"][0]), float(MOVES["settle"]["dur"][1]))
	# An establish is the reveal, not the arrival on a face, so it keeps the ordinary deadline
	# even when the panel it ends on holds the footage - FILM_ARRIVE would cut the reveal to a
	# third of the hold, which is the opposite of what it is for.
	var frac := FILM_ARRIVE if (panel >= 0 and panel == _film_at and kind != "establish") \
		else _arrive_frac()
	# AND NEVER LONGER THAN THE ALLOWANCE HAS LEFT. Without this the cut's own move can sample
	# a duration longer than its whole share and blow it in one go, which is the same defect
	# the allowance exists to fix, arriving one move earlier.
	var cap := maxf(ARRIVE_MIN, minf(room * frac, _travel_left))
	return maxf(0.05, minf(want, cap))


## IS THIS JUMP A DIFFERENT SHOT? See CUT_MIN_AIM. BOTH bars have to clear: a different panel
## AND a real distance across the sheet.
func _worth_cutting(b: Dictionary) -> bool:
	if (b.aim as Vector2).distance_to(_cam.aim) < CUT_MIN_AIM:
		return false
	return _framing_panel(_clamp_aim(_cam.aim)) != _framing_panel(_clamp_aim(b.aim))
>>>>>>> Stashed changes


## Is this jump a different SHOT? See CUT_MIN_AIM.
func _worth_cutting(tgt: Dictionary) -> bool:
	if (tgt.aim as Vector2).distance_to(_cam.aim) < CUT_MIN_AIM:
		return false
	return _framing_panel(_clamp_aim(_cam.aim)) != _framing_panel(_clamp_aim(tgt.aim))


## Draw a shot kind, weighted, with the discontinuity scaled by severity squared so that at
## camera 0 the camera never jump-cuts at all.
func _pick_shot(r: RandomNumberGenerator) -> String:
	var total := 0.0
	for k in SHOTS:
		total += _shot_weight(String(k))
	if total <= 0.0:
		return "settle"
	var pick := r.randf() * total
	for k in SHOTS:
		pick -= _shot_weight(String(k))
		if pick <= 0.0:
			return String(k)
	return "settle"


func _shot_weight(kind: String) -> float:
	var w := float(SHOTS[kind]["w"])
	if bool(SHOTS[kind].get("hard", false)):
		return w * pow(clampf(_sev(), 0.0, 2.0), 2.0)
	return w


## WHERE A SHOT OF [param kind] ON [param panel] LOOKS FROM.
##
## Built as an OFFSET from the panel's own natural framing, so every entry in SHOTS is a
## distance rather than an absolute, and severity scales the distances (see REACH_CALM).
func _target_for(r: RandomNumberGenerator, panel: int, kind: String) -> Dictionary:
	var spec: Dictionary = SHOTS[kind]
	var reach := _reach()
	var base := _spread.panel_center(panel)
	# THE FILM PANEL IS FRAMED DEAD CENTRE AND WHOLE - see CONTAIN_FILL. Everywhere else the
	# aim leans toward the middle of its own page, which keeps the outer trim out of shot...
	if panel != _film_at:
		base = base.lerp(_page_center(panel), AIM_PULL)
	# ...and a WIDE shot leans most of the way there, because its subject is the page rather
	# than the panel. Not all the way: the panel the reading is on should still be the part of
	# the sheet the composition is built around.
	if bool(spec.get("wide", false)):
		base = _spread.panel_center(panel).lerp(_page_center(panel), 0.75)
	var span: Array = spec["fill"]
	# THE BAND BARELY NARROWS AS THE CAMERA CALMS, and it used to collapse. This was
	# `lerp(1.0, sampled, reach)` with reach floored at 0.35, which dragged EVERY framing 65% of
	# the way back to 1.04 at camera 0 - one close-up value for the whole vocabulary, which is
	# the whole of "it never pulls back... it's just a constant bouncing around between
	# individual frames, close-up". What a gentle camera should do less of is TRAVELLING; how
	# varied its framing is belongs to composition, not to restlessness.
	var fill := lerpf(1.0, r.randf_range(float(span[0]), float(span[1])),
		clampf(0.6 + reach * 0.4, 0.0, 1.0))
	var off := Vector2(r.randf_range(-1.0, 1.0), r.randf_range(-1.0, 1.0))
	off = off.normalized() if off.length() > 0.001 else Vector2.RIGHT
	var aim := base + off * float(spec["aim"]) * reach
	# THE ANGLES WALK, they are not re-drawn. A shot may spend a fraction of the spread's arc,
	# so a big change of angle is spread over several shots rather than happening at one.
	_az_walk = _walk(_az_walk, AZ_ARC * _sev(), r)
	_el_walk = _walk(_el_walk, deg_to_rad(EL_VARY_DEG) * _sev(), r)
	_roll_walk = _walk(_roll_walk, ROLL_VARY * _sev(), r)
	# THE SWING SCALES BY THE SLIDER, NOT BY REACH. `reach` floors at REACH_CALM so that a calm
	# camera still travels between panels - a camera that cannot move is not the gentle end of
	# the range, it is a photograph - but the contract at camera 0 is that the camera never
	# TURNS, exactly, and a floor of 0.35 on the angular term left 3.3 degrees a shot. Distance
	# and angle are different promises and they scale by different things.
	var az := _az_base + _az_walk \
		+ float(spec["az"]) * _sev() * (1.0 if r.randf() < 0.5 else -1.0)
	# A WIDE SHOT IS EXEMPT FROM _settle_aim. That guard walks an aim toward its panel until
	# enough panel is under it, which is right for a shot OF a panel and destroys a shot of the
	# page - the composition it is reaching for has gutters and neighbours in it by design.
	return {
		"aim": _clamp_aim(aim) if bool(spec.get("wide", false)) \
			else _settle_aim(_clamp_aim(aim)),
		"az": az,
		"el": clampf(_el_base + _el_walk, deg_to_rad(EL_MIN_DEG), deg_to_rad(EL_MAX_DEG)),
		"roll": _roll_base + _roll_walk,
		"fill": clampf(_panel_fill(panel) * fill, FILL.x * 0.6, PUSH_FILL.y),
		"fov": _lens_fov,
	}


## The framing a panel READS at - the middle of FILL, so `fill` in SHOTS is a multiplier of
## "this panel, filling the frame the way a panel should" rather than of an arbitrary number.
func _panel_fill(_panel: int) -> float:
	return (FILL.x + FILL.y) * 0.5


## HOW FAR THIS SHOT STILL HAS TO GO, normalised so that 1.0 is "arrived" on every channel.
func _pose_gap() -> float:
	var g: float = (_cam.aim as Vector2).distance_to(_tgt.aim) / GAP_AIM
	var fa := maxf(float(_cam.fill), 0.01)
	var fb := maxf(float(_tgt.fill), 0.01)
	g = maxf(g, absf(log(fb / fa)) / GAP_FILL)
	g = maxf(g, absf(angle_difference(float(_cam.az), float(_tgt.az))) / GAP_ANG)
	return g


## ONE FRAME OF CAMERA. Advance the shot, then follow.
func _ease(delta: float) -> void:
	# The spread turns under the camera, slowly and forever - independent of the shot, because
	# the sheet is scenery and the camera is the performance. The rig is page-local, so this
	# rotates the camera and the sheet by the same matrix and the projection is invariant to
	# it: it shows as the page living, not as the shot moving.
	_att_target += _att_rate * delta
	var sway := Vector3(
		_mod.value("tilt") * 0.09,
		_mod.value("sway") * 0.12,
		_mod.value("roll") * 0.05) if _mod != null else Vector3.ZERO
	_att = _att.lerp(_att_target + sway, 1.0 - exp(-PAGE_EASE * delta))
	_att_basis = Basis.from_euler(_att)
	if _dip_t >= 0.0:
		_dip_t += delta
	if _spread == null or _tgt.is_empty():
		return
<<<<<<< Updated upstream
	_shot_t += delta
	# ARRIVAL IS A MEASUREMENT, NOT A CLOCK. The shot is over when the picture has got there.
	if not bool(_shot.get("arrived", false)) \
			and (_pose_gap() < ARRIVED or _shot_t > ARRIVE_CAP):
		_shot["arrived"] = true
		_hold_t = 0.0
		_hold_from = _tgt.aim
	if bool(_shot.get("arrived", false)):
		_hold_t += delta
		# THE HOLD IS WHAT EARNS A REVERSAL - and it has to be a hold. See ZOOM_FREE.
		if _hold_t >= ZOOM_FREE:
			_zoom_dir = 0
		# ...AND A HELD SHOT STILL BREATHES, along the heading this shot was given. Measured
		# from where it ARRIVED, so the excursion is bounded however long the hold runs.
		# IT DOES NOT SATURATE. Clamping the creep at 1 made the held shot stop dead after
		# HOLD_CREEP_TIME, which is a freeze on any hold longer than that - the same defect an
		# earlier version shipped and a frame-by-frame reading of the export measured at 33.8
		# seconds of literally identical picture. It keeps creeping; what bounds it is
		# _settle_aim, which walks the aim back when it nears the panel's edge, so the drift is
		# limited by the geometry rather than by a clock.
		var creep := _hold_t / HOLD_CREEP_TIME
		# NOT SCALED BY REACH. A held shot is held at every setting - what the slider changes is
		# how far the camera TRAVELS between subjects, not how still it can be once it has
		# arrived. Scaling the creep by reach made a busy camera drift through its own holds
		# fast enough that they stopped reading as holds at all (12% of frames still, against
		# 44% at the calm end).
		_tgt["aim"] = _settle_aim(_clamp_aim(_hold_from
			+ (_shot["heading"] as Vector2) * HOLD_CREEP * creep))
	_follow(delta)
=======
	_mv_t += delta
	# THE MOVE IS OVER AND NOTHING HAS CUT. Chain, do not stop - see _chain_move. The new
	# move starts at _mv_t 0, so this is a continuation and not a reset of the picture.
	#
	# EXCEPT A SETTLE, WHICH JUST KEEPS SETTLING. A settle exists to hold the shot still until
	# the cut arrives; planning a fresh one when it expires re-eases the aim toward the panel
	# centre all over again, and a run of them is a camera visibly re-correcting onto a frame it
	# is already on. Extending it is what "held" actually means.
	if _mv_t >= float(_mv.dur):
		var done := String(_mv.get("kind", ""))
		if done == "settle" or done == "breathe":
			# EXTENDING THE CLOCK IS NOT HOLDING, IT IS FREEZING. A settle's ease has run by
			# then, so adding seconds to its duration adds seconds of a motionless picture -
			# forty of them on a long hold. It hands over to a breathe instead, and a breathe
			# hands over to another, so a held shot keeps creeping until the cut arrives.
			_breathe()
		else:
			_chain_move()
	var k := clampf(_mv_t / float(_mv.dur), 0.0, 1.0)
	_cam = _lerp_state(_mv.a, _mv.b, _curve(String(_mv.ease), k))
>>>>>>> Stashed changes
	_place_eye(delta)


## THE FOLLOWER, AND IT IS SPEED-LIMITED RATHER THAN EXPONENTIAL.
##
## A pure exponential covers 63% of its distance in ONE time constant and then crawls through
## the remainder, so a shot given ten seconds does nearly all of its visible movement in the
## first second or two and spends the rest converging invisibly. That is a fast transition
## followed by nothing, however long the arithmetic says the shot lasts: "when the camera
## shifts its focus from one frame to another, it does it in about 1 second. It's fast, every
## single time." The measurement agreed with the arithmetic and not with the picture, because
## it timestamped the end of the CRAWL rather than the end of the MOVEMENT.
##
## So the journey advances at a constant fraction per second - the camera covers the same
## ground in the ninth second as in the first - and eases only over the last of it, which is
## what makes a pan read as a pan rather than as a snap with a tail. It remains an integration:
## `_cam` is never assigned, a cut mid-journey is simply a new target, and nothing here can
## teleport the picture.
func _follow(delta: float) -> void:
	var secs := maxf(float(_shot.get("secs", 6.0)), 0.1)
	# How much of the ORIGINAL journey to cover this frame...
	var step := delta / secs
	# ...eased out over the last of it, so the shot lands rather than stopping dead. Above the
	# taper the speed is flat, which is the constant-rate pan.
	var left := _pose_gap() / maxf(float(_shot.get("gap0", 1.0)), 0.001)
	if left < ARRIVE_TAPER:
		# FLOORED WELL ABOVE ZERO. A taper that trails off to nothing means the shot never
		# actually converges: the last of the journey stretches past the whole budget and
		# ARRIVE_CAP - the safety net - becomes the thing that ends every shot, at exactly its
		# value. Measured, every arrival reported 14.0s, which is that constant and not a
		# camera. This lands the shot; it does not dissolve it.
		step *= maxf(smoothstep(0.0, ARRIVE_TAPER, left), 0.45)
	var kp := clampf(step / maxf(left, 0.001), 0.0, 1.0)
	var ka := clampf(kp * RATE_ANGLE, 0.0, 1.0)
	_cam["aim"] = (_cam.aim as Vector2).lerp(_tgt.aim, kp)
	_cam["fill"] = lerpf(float(_cam.fill), float(_tgt.fill), kp)
	_cam["az"] = float(_cam.az) \
		+ angle_difference(float(_cam.az), float(_tgt.az)) * ka
	_cam["el"] = lerpf(float(_cam.el), float(_tgt.el), ka)
	_cam["roll"] = lerpf(float(_cam.roll), float(_tgt.roll), ka)
	_cam["fov"] = lerpf(float(_cam.fov), float(_tgt.fov), ka)


## The whole-frame alpha a `dip` is currently at: out through black, then back.
func _dip_alpha() -> float:
	if _dip_t < 0.0:
		return 1.0
	if _dip_t < DIP_OUT:
		return 1.0 - _dip_t / DIP_OUT
	if _dip_t < DIP_OUT + DIP_IN:
		return (_dip_t - DIP_OUT) / DIP_IN
	return 1.0


## Put the eye on its spread-local spherical offset from where the camera is aiming.
##
## The offset direction is built in SPREAD space and then rotated by the sheet's attitude,
## so it is a fixed station relative to the paper: travelling from panel to panel slides the
## camera ACROSS the spread at a constant angle, which is the move a copy-stand shot makes
## and the one the eye reads as flying over a comic. Building it in world space instead
## would swing the angle around every time the sheet moved.
func _place_eye(delta := 0.0) -> void:
	if _spread == null:
		return
	var aim: Vector2 = _clamp_aim(_cam.aim)
	var c := _world(aim, _att_basis)
	# A FIELD scene is raked gently, a subject hard - see FIELD_FLATTEN.
	var fp := _framing_panel(aim)
	# THE FILM PANEL IS SHOWN WHOLE - see CONTAIN_FILL. Only while it is the panel the shot is
	# actually built around; a film at the edge of a wide shot is just another picture on the
	# page and does not get to decide the framing.
	var contain: bool = fp >= 0 and fp == _film_at
	var flat := maxf(_flatten(), FILM_FLATTEN if contain else 0.0)
<<<<<<< Updated upstream
	# FOLLOWED, NOT ADOPTED. `_flatten()` reads the panel being READ, so it steps the instant
	# the Director cuts between a field scene and a subject one - up to 21.7 degrees of
	# elevation in a single frame - and `contain` steps as the aim crosses the film panel's
	# edge. Neither is a shot, and nothing in the shot vocabulary can see either.
	if _snap or _dist_s < 0.0 or delta <= 0.0:
		_flat_s = flat
	else:
		_flat_s += (flat - _flat_s) * (1.0 - exp(-RATE_FLAT * delta))
=======
	# FOLLOWED, NOT ADOPTED - see POSE_EASE. This is a step function of which panel is being
	# read and which panel the aim is inside, and both of those change in a single frame.
	if _snap or _dist_s < 0.0 or delta <= 0.0:
		_flat_s = flat
	else:
		_flat_s += (flat - _flat_s) * (1.0 - exp(-FLAT_EASE * delta))
>>>>>>> Stashed changes
	flat = _flat_s
	var el: float = lerpf(float(_cam.el), PI * 0.5, flat)
	var roll: float = float(_cam.roll) * (1.0 - flat * 0.7)
	_roll = roll
	_fov = float(_cam.fov)
	var dir := _att_basis * Vector3(cos(_cam.az) * cos(el), sin(_cam.az) * cos(el), sin(el))
	# SCALE IS SET BY WHICHEVER PANEL THE AIM IS INSIDE, not by the nearest centre and not by
	# the one being read: a travelling move spends most of its time between panels, and
	# framing those seconds against a panel that is off screen puts the scale somewhere
	# arbitrary. See _framing_panel.
	var pw := _panel_world(fp)
	_cam["aim"] = aim
	# ZOOM IS COUPLED TO CONTENT - see CONTENT_CORE. Off a panel the requested fill is
	# ignored and the shot opens to OFF_FILL, so the camera physically cannot be tight on a
	# gutter or a corner however a move was planned or wherever a chain left it.
	# ...AND HOW FAR IT OPENS IS A SEVERITY QUESTION TOO. This is a 1.7x swing in the rendered
	# framing on EVERY lateral move - out at the gutter, back in on the far panel - and it is
	# invisible to the move vocabulary, so nothing above could smooth or forbid it. It buys
	# almost nothing at a gentle setting, where the camera is at 1.04 frames of panel and in no
	# danger of being tight on anything; it earns its keep at a severe one, where a push is at
	# 3.8 and a gutter at that framing is the original complaint.
	var eff := lerpf(_open_to(), float(_cam.fill), _content_at(aim))
	if contain:
		eff = minf(eff, CONTAIN_FILL)
	# TWO CONSTRAINTS, and the nearer wins.
	#
	# The first frames the PANEL: near enough that it spans `eff` frames top to bottom. The
	# second frames the SPREAD: near enough that the sheet covers the frame on BOTH axES. A
	# single portrait page could never satisfy the second without pushing in until the page
	# stopped reading as a page, which is why the first cut of this only ever checked the
	# width; a spread is 1.33 wide against a 1.78 frame, so covering the width covers the
	# height with room to spare and the check can finally be honest.
	# THE WHOLE VIDEO WINS OVER THE COVERING SOLVE, and that is a deliberate ordering rather
	# than an oversight. The two constraints genuinely conflict on a wide panel at a rake -
	# measured, containing it wanted 3.03 units and covering allowed only 2.13 - so one of them
	# has to give. Letting the cover win crops a speaking face out of frame to keep a strip of
	# desk off it, and between those two the answer is not close: "you really don't want to
	# focus the camera on her door to the side." FILM_FLATTEN is what keeps the price small.
	var cov := _cover(_spread_world(), c, dir)
	if contain:
		cov *= CONTAIN_DESK
	var d := minf(_fit(pw, c, eff, dir, contain), cov)
	# ...and never nearer than this, whatever the two constraints ask for. Pushing in until
	# the paper covers the frame is right; pushing in until the camera is INSIDE one panel's
	# artwork is a shot of a texture, with no page, no gutter and no comic in it.
	# The near floor never applies to a contained panel: its whole point is that the camera
	# stops where the picture fits, and a floor that pushes past that would crop it again.
	var near := 0.0
	if not contain:
		near = _fit(pw, c, CROP_MAX, dir)
		d = maxf(d, near)
<<<<<<< Updated upstream
	# ...AND SO IS THE DISTANCE. `pw` changes panel as the aim crosses a gutter and the solve
	# changes branch at the film panel's edge; this file's own measurements record 3.03 against
	# 2.13 and 7.76 against 4.62 world units for the same shot either side of that branch.
=======
	# ...AND THE DISTANCE IS FOLLOWED TOO. `pw` changes panel as the aim crosses a gutter and
	# the solve changes branch at the film panel's edge; both are one-frame steps in `d` of
	# 40-70% by this file's own measurements. A declared cut still snaps - that is what _snap
	# is for - so the vocabulary keeps its discontinuities and nothing else gets one.
	# NO TIME PASSED MEANS NO EASING IS POSSIBLE, which is also how the gates place a pose:
	# they call this directly to read a framing out of it, and a follower with no delta would
	# hand them the previous frame's answer.
>>>>>>> Stashed changes
	if _snap or _dist_s < 0.0 or delta <= 0.0:
		_dist_s = d
		_snap = false
	else:
<<<<<<< Updated upstream
		_dist_s += (d - _dist_s) * (1.0 - exp(-RATE_DIST * delta))
	# THE FOLLOWER MAY LAG, BUT IT MAY NOT BREAK THE TWO CONSTRAINTS. `cov` is the farthest the
	# eye may sit and still have paper across the whole frame, `near` the closest it may sit
	# without being inside one panel's artwork. Easing is for the INTERIOR of the feasible
	# range; its edges are not negotiable, and a follower left to drift outside them shows desk.
=======
		_dist_s += (d - _dist_s) * (1.0 - exp(-POSE_EASE * delta))
	# THE FOLLOWER MAY LAG, BUT IT MAY NOT BREAK THE TWO CONSTRAINTS. `cov` is the FARTHEST the
	# eye may sit and still have paper across the whole frame, and `near` the closest it may sit
	# without being inside one panel's artwork - so a follower drifting outside them shows desk,
	# which is exactly what it did: four of nine probe frames fell to 0.94-0.98 coverage the
	# first time this was eased. Easing is for the INTERIOR of the feasible range; the edges of
	# it are not negotiable and are enforced after the ease, not before.
>>>>>>> Stashed changes
	_dist_s = minf(_dist_s, cov)
	if near > 0.0:
		_dist_s = maxf(_dist_s, near)
	_eye = c + dir * _dist_s
	_look = c


## How far toward square-on the rake is pulled for the panel currently being read. Zero for
## a subject; FIELD_FLATTEN for a field. See that constant.
func _flatten() -> float:
	var sc := _focal()
	return FIELD_FLATTEN if (sc != null and sc.framing == "field") else 0.0


## Keep an aim on the paper - see AIM_MARGIN. Spread coordinates, so x runs to 2.
func _clamp_aim(p: Vector2) -> Vector2:
	var b := _printed_bounds()
	return Vector2(clampf(p.x, b.position.x, b.end.x), clampf(p.y, b.position.y, b.end.y))


## THE PRINTED AREA: the box that contains every panel on the spread, grown by AIM_MARGIN.
##
## The aim used to be clamped to the SHEET, so a move that ran past its subject - a track
## overruns by nearly half a page width by design - came to rest out on the trim margin, in a
## corner of the paper with no picture anywhere near it. The chain then took the camera back
## toward the reading, which is the "drifts into a corner then bounces back from it again"
## exactly: out to the corner because the move said so, back again because the reading did.
##
## Clamping to the PRINTED area instead means an overrun still leaves its panel - that is the
## gesture, and it still passes over the gutter and out to the edge of the outermost panel -
## but it cannot reach the blank margin, so there is no corner to be retrieved from. The
## outset by AIM_MARGIN is what keeps "past the last panel" from meaning "exactly on its edge".
func _printed_bounds() -> Rect2:
	if _spread == null or _spread.panels.is_empty():
		return Rect2(AIM_MARGIN, AIM_MARGIN, 2.0 - AIM_MARGIN * 2.0, 1.0)
	var r: Rect2 = _spread.panels[0]["rect"]
	for i in _spread.panels.size():
		r = r.merge(_spread.panels[i]["rect"])
	return r.grow(AIM_MARGIN)


## HOW MUCH OF THE FRAME THE SHEET COVERS. 1.0 when there is paper under all four corners
## of the picture; below that, the number is how far the WORST corner falls outside the
## sheet as a fraction of half the frame height - so 0.92 is a sliver of desk in one corner
## and 0.5 is a wedge across the shot. Read by tests/comic_look_probe.gd, which is the only
## way "is there dead space in the frame" gets checked by something other than an eye.
##
## FOUR CORNERS, not two edges of a bounding box. The old measure took the projected sheet's
## bounding box and asked whether it reached past the left and right frame edges. That is
## wrong twice over. It never looked UP or DOWN, so it reported a clean 1.00 on a frame with
## a band of black desk across the top - the exact frame the report was about. And a
## bounding box is not the sheet: a raked, rolled quad has a box that reaches every edge
## while the quad itself cuts a corner off. A number that cannot see the defect it exists to
## catch is worse than no number.
func page_coverage() -> float:
	if _spread == null:
		return 1.0
	var u := minf(_stage_size.x, _stage_size.y)
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0)
	var hy := (_stage_size.y * 0.5) / maxf(u, 1.0)
	var q := _sheet_poly(_spread_world(), _lens)
	if q.size() < 3:
		return 1.0                     # every corner behind the eye: the sheet is all around
	return clampf(1.0 + _quad_clearance(q, hx, hy) / maxf(hy, 1e-4), 0.0, 1.0)


## The sheet's outline in screen space, CLIPPED TO THE NEAR PLANE.
##
## A perspective projection cannot express a point behind the eye - it inverts through
## infinity - so every draw here bails out when one appears, and the two coverage tests used
## to bail out by declaring themselves covered. On a single page that was nearly honest: a
## page is 2 world units across, and a camera near enough to put a corner behind itself
## really was over the paper. A spread is 4 across. At the reading distance its far corners
## go behind the eye while there is still a wedge of desk in the opposite corner of the
## frame, and the escape hatch then reports a clean 1.00 on exactly the picture the coverage
## number exists to catch.
##
## So the outline is clipped instead of abandoned - Sutherland-Hodgman against the one
## plane, which turns the quad into at most a pentagon and costs one lerp per crossing. The
## result is the sheet's true silhouette, and the corner test below is then exact at any
## framing.
func _sheet_poly(pts: Array, lens: Lens3D) -> PackedVector2Array:
	var out := PackedVector2Array()
	var n := pts.size()
	for i in n:
		var a: Vector3 = pts[i]
		var b: Vector3 = pts[(i + 1) % n]
		var da := lens.depth(a) - lens.near
		var db := lens.depth(b) - lens.near
		if da >= 0.0:
			var pa := lens.project(a)
			out.append(Vector2(pa.x, pa.y))
		if (da >= 0.0) != (db >= 0.0):
			# where the edge crosses the plane, in world space, so the projection of the
			# crossing point is the projection of a point that is genuinely in front
			var t := da / (da - db)
			var pc := lens.project(a.lerp(b, t))
			out.append(Vector2(pc.x, pc.y))
	return out


## The signed distance from the WORST frame corner to the edges of the projected quad
## [param q], in unit-fraction units. Positive means every corner of the frame is inside the
## quad, i.e. the sheet covers the picture; negative is how far the worst one sticks out.
##
## The quad is convex (it is a planar rectangle under a perspective projection), so a point
## is inside exactly when it is on the same side of all four edges - and the least of those
## four signed distances is how much room there is to spare. Taking the least again over the
## four frame corners gives one number for "is there desk in shot, and how much".
##
## Wound from the quad's own signed area rather than assumed, because the page's roll and
## the camera's can put the sheet either way round on screen.
func _quad_clearance(q: PackedVector2Array, hx: float, hy: float) -> float:
	var n := q.size()
	if n < 3:
		return 0.0
	var area := 0.0
	for e in n:
		var a: Vector2 = q[e]
		var b: Vector2 = q[(e + 1) % n]
		area += a.x * b.y - b.x * a.y
	var wind := 1.0 if area >= 0.0 else -1.0
	var worst := INF
	for cx: float in [-hx, hx]:
		for cy: float in [-hy, hy]:
			for e in n:
				var a: Vector2 = q[e]
				var b: Vector2 = q[(e + 1) % n]
				var d := b - a
				var edge := d.length()
				if edge < 1e-9:
					continue
				# signed distance to the edge line, positive on the quad's inside
				var side := wind * (d.x * (cy - a.y) - d.y * (cx - a.x)) / edge
				worst = minf(worst, side)
	return 0.0 if worst == INF else worst


## HOW MUCH OF THE FRAME THE PANEL BEING READ TAKES UP, on its worst axis. 1.0 means it
## exactly spans the picture; above 1 the screen is cropping into it, which is the intended
## shot for an abstract field and the wrong one for a piece of footage - see CONTAIN_FILL.
## Read by tests/comic_look_probe.gd, so "is the whole video on screen" is a number.
func read_panel_fit() -> float:
	if _spread == null or _read < 0 or _read >= _spread.panels.size():
		return 0.0
	var u := minf(_stage_size.x, _stage_size.y)
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0)
	var hy := (_stage_size.y * 0.5) / maxf(u, 1.0)
	var lo := Vector2(1e9, 1e9)
	var hi := Vector2(-1e9, -1e9)
	for w in _panel_world(_read):
		var pr := _lens.project(w)
		if pr.z <= _lens.near:
			return 99.0                # wrapping the eye: as cropped as it gets
		lo = lo.min(Vector2(pr.x, pr.y))
		hi = hi.max(Vector2(pr.x, pr.y))
	return maxf((hi.x - lo.x) / (2.0 * maxf(hx, 1e-4)),
		(hi.y - lo.y) / (2.0 * maxf(hy, 1e-4)))


## HOW MANY SCREEN PIXELS THE READ PANEL COVERS PER TEXEL OF ITS RENDER TARGET. 1 is 1:1;
## above 1 the panel's bitmap is being magnified and will look soft or blocky.
##
## A panel is not drawn as live geometry - it is rasterized into its own SubViewport at a fixed
## size and then mapped onto a quad as a TEXTURE. That is what makes a held panel free (the
## frozen target keeps its picture for nothing) and it is also why a tight shot can pixelate:
## the shot is resampling a bitmap, exactly as it would a video.
func read_panel_magnification() -> float:
	if _spread == null or _read < 0 or _read >= _spread.panels.size():
		return 0.0
	var vp: SubViewport = _slots[_pool * POOL + _read]
	var th := maxf(float(vp.size.y), 1.0)
	return read_panel_fit() * _stage_size.y / th


## WHY THE CURRENT SHOT IS FRAMED WHERE IT IS - the numbers that decide it, for the probe.
## Not a pretty string for its own sake: "there is desk in this frame" has now had two fixes
## aimed at it that measured as no change at all, because both were aimed at a cause guessed
## from the outside instead of read off the shot that was failing.
func shot_debug() -> String:
	if _spread == null:
		return ""
	var aim: Vector2 = _clamp_aim(_cam.aim)
	var c := _world(aim, _att_basis)
	var fp := _framing_panel(aim)
	var contain: bool = fp >= 0 and fp == _film_at
	var flat := maxf(_flatten(), FILM_FLATTEN if contain else 0.0)
	var el: float = lerpf(float(_cam.el), PI * 0.5, flat)
	var dir := _att_basis * Vector3(cos(_cam.az) * cos(el), sin(_cam.az) * cos(el), sin(el))
	var eff := lerpf(_open_to(), float(_cam.fill), _content_at(aim))
	if contain:
		eff = minf(eff, CONTAIN_FILL)
	var cov_d := _cover(_spread_world(), c, dir)
	return "el %.0fdeg flat %.2f contain %s fit %.2f cover %.2f eye %.2f att(%.2f %.2f %.2f)" % [
		rad_to_deg(el), flat, "Y" if contain else "n",
		_fit(_panel_world(fp), c, eff, dir, contain), cov_d,
		(_eye - c).length(), _att.x, _att.y, _att.z]


## Which panel the camera is over. Nearest centre in spread coordinates - the fallback for
## [method _framing_panel] when the aim is inside nothing at all.
func _nearest_panel(aim: Vector2) -> int:
	var best := 0
	var best_d := 1e9
	for i in _spread.panels.size():
		var d := aim.distance_squared_to(_spread.panel_center(i))
		if d < best_d:
			best_d = d
			best = i
	return best


func _prepare_lens() -> void:
	_lens.eye = _eye
	_lens.look = _look
	_lens.fov = _fov
	# CAMERA UP, and it must not be world up: at a hard rake the view axis can come within
	# a few degrees of it and the basis collapses (Lens3D falls back and the picture rolls
	# through 90 degrees in one frame). The sheet's own up is always perpendicular to the
	# paper's normal, so it is a stable reference - and taking it from the sheet is also what
	# makes its roll show as a tilted horizon rather than as nothing at all.
	var fwd := (_look - _eye)
	fwd = fwd.normalized() if fwd.length() > 1e-6 else Vector3.FORWARD
	var up := _att_basis * Vector3.UP
	if absf(up.dot(fwd)) > 0.9:
		up = _att_basis * Vector3.RIGHT     # grazing along the sheet's up axis
	_lens.up = Basis(fwd, _roll) * up
	_lens.prepare()


## THE FRAMING DISTANCE: how far along [param dir] the eye must sit for [param pts] to span
## [param fill] frames VERTICALLY.
##
## VERTICAL ONLY, and that is the whole difference between a comic and a slideshow of
## pages. Constraining both axes means fitting the panel ENTIRELY inside the frame, and a
## panel narrower than 16:9 - which most comic panels are - then leaves the rest of the
## picture to the page, the trim edge, and the desk beyond it. Constrained on height, the
## panel always spans the frame, and what fills the sides is the gutter and the panels next
## to it. That is what reading a comic close up actually looks like.
##
## Solved by bisection rather than in closed form, because unlike a straight-down-z camera
## the view basis turns as the distance changes, so there is no single inequality to
## rearrange. A handful of halvings on four points is nothing, and it is exact at any rake.
##
## THE HALF-EXTENT IS 0.5, NOT 1. Lens3D projects into ghost's unit-fraction space, which
## the caller multiplies by unit() = the SHORTER SCREEN AXIS - so the visible vertical range
## is +/- (H/2)/min(W,H) = +/- 0.5. Reading it as 1 puts the camera at half the distance it
## needs, which is what the first cut of this did.
## [param contain] fits the points on BOTH axes instead of on height alone, so nothing
## overflows the frame. See CONTAIN_FILL - it is what puts a whole video on screen.
func _fit(pts: Array, c: Vector3, fill: float, dir: Vector3, contain := false) -> float:
	var u := minf(_stage_size.x, _stage_size.y)
	# fill is how many FRAMES tall the panel should be, so a bigger fill is a nearer
	# camera: the allowed half-extent grows and the bisection settles closer in.
	var hy := (_stage_size.y * 0.5) / maxf(u, 1.0) * fill
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0) * fill
	var probe := _fit_lens
	probe.fov = _fov
	probe.look = c
	probe.up = _lens.up
	var lo := 0.15
	var hi := 24.0
	for _iter in SOLVE_STEPS:
		var mid := (lo + hi) * 0.5
		probe.eye = c + dir * mid
		probe.prepare()
		var fits := true
		for pt in pts:
			var pr := probe.project(pt)
			if pr.z <= probe.near or absf(pr.y) > hy \
					or (contain and absf(pr.x) > hx):
				fits = false
				break
		if fits:
			hi = mid
		else:
			lo = mid
	return hi


## How many halvings the two solvers take. Sixteen resolves the 0.15-24 bracket to about
## 0.0004 world units, which is a third of a thousandth of a page width - far past anything
## a camera distance can show, and six fewer projections per point than the 22 it was.
const SOLVE_STEPS := 16


## THE COVERING DISTANCE: the FARTHEST the eye may sit along [param dir] while [param pts]
## still span the whole frame, on BOTH axes. Beyond it you can see past them.
##
## The mirror of [method _fit], and it has to bisect the other way: a projection shrinks
## with distance, so "fits inside" is true beyond some distance and "covers" is true within
## one. Same halvings, same exactness at any rake.
func _cover(pts: Array, c: Vector3, dir: Vector3) -> float:
	var u := minf(_stage_size.x, _stage_size.y)
	var hx := (_stage_size.x * 0.5) / maxf(u, 1.0)
	var hy := (_stage_size.y * 0.5) / maxf(u, 1.0)
	var probe := _cover_lens
	probe.fov = _fov
	probe.look = c
	probe.up = _lens.up
	var lo := 0.15
	var hi := 24.0
	var best_d := 0.15
	var best_clear := -1e9
	for _iter in SOLVE_STEPS:
		var mid := (lo + hi) * 0.5
		probe.eye = c + dir * mid
		probe.prepare()
		# CLIPPED, not abandoned, when a corner goes behind the eye - see _sheet_poly. An
		# empty polygon means every corner is behind, which is the most covered case there
		# is; reading THAT as "does not cover" inverts the whole search and drives the
		# camera inside a panel.
		var q := _sheet_poly(pts, probe)
		# Covers when every CORNER OF THE FRAME has paper under it - the same test
		# page_coverage() reports, so the solve and the measurement can never disagree about
		# what "covered" means. See _quad_clearance for why a bounding box is not enough.
		var clear := 1e9 if q.size() < 3 else _quad_clearance(q, hx, hy)
		# Remember the BEST distance seen, for the case where none of them cover - see below.
		if clear > best_clear:
			best_clear = clear
			best_d = mid
		if clear >= 0.0:
			lo = mid
		else:
			hi = mid
	# DID ANYTHING COVER AT ALL? The bisection assumes covering is true near and false far and
	# returns the boundary. When the sheet cannot cover the frame at ANY distance - a hard rake
	# and a rolled page can leave a corner of the picture off the paper however close the camera
	# gets - `lo` never moved, and returning it hands the caller 0.15 world units as a NEAR
	# bound, pinning the eye inside a panel's artwork.
	#
	# RETURNING A HUGE NUMBER INSTEAD WAS WORSE, and it is what "the camera's anchor jerks and
	# then flies to some other position... as if the camera were pulling on some rubber band,
	# which broke" actually was. This value is a CONSTRAINT the caller takes the minimum with,
	# so switching it between about two world units and 1e9 in one frame - which is what
	# happens the instant a drifting page crosses from just-coverable to not - releases that
	# constraint completely, and the camera leaves at whatever speed the framing solve alone
	# asks for. A binding constraint that vanishes is a snapped rubber band; that is not a
	# metaphor, it is the mechanism.
	#
	# So when nothing covers, return the distance that came CLOSEST to covering. Approaching the
	# boundary from the uncoverable side, that distance approaches the boundary itself, so the
	# value is continuous across the transition and there is nothing to snap.
	if best_clear < 0.0:
		return best_d
	return lo


## The four world corners of one panel, cant and spread attitude applied.
func _panel_world(i: int) -> Array:
	var out: Array = []
	for uv in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		out.append(_world(_spread.panel_point(i, uv), _att_basis))
	return out


## The four world corners of the whole spread.
func _spread_world() -> Array:
	var out: Array = []
	for p in _spread.spread_corners():
		out.append(_world(p, _att_basis))
	return out


## Panel [param i]'s bounding box on screen, for the visibility test.
func _panel_rect(i: int) -> Rect2:
	var u := minf(_stage_size.x, _stage_size.y)
	var origin := _stage_size * 0.5
	var r := Rect2()
	var first := true
	for w in _panel_world(i):
		var pr := _lens.project(w)
		if pr.z <= _lens.near:
			return Rect2(Vector2.ZERO, _stage_size)    # partly behind the eye: assume on screen
		var p := Vector2(pr.x, pr.y) * u + origin
		if first:
			r = Rect2(p, Vector2.ZERO)
			first = false
		else:
			r = r.expand(p)
	return r


## A point in SPREAD space (x 0..2 with the spine at 1, y 0..aspect and DOWN) -> world
## space. Everything drawn on the sheet goes through this one call, so the paper, the
## panels, the gutters and the ink can never disagree about where the spread is.
##
## [param m] IS THE WHOLE TRANSFORM, handed in already built: the spread's attitude for a
## flat page, or the attitude times the leaf's hinge rotation for a turning one. It used to
## build Basis.from_euler(att) here, per vertex - six trig calls and a matrix, roughly 2500
## times a frame with an argument that had not changed - and AUDIT.md attributes most of the
## page's cost to exactly that. Hoisting it is not an approximation; the value was always
## identical.
##
## [param mirror] reflects the point about the spine before placing it, which is what draws
## the BACK of a turning leaf. See [method ComicSpread.mirror].
func _world(p: Vector2, m: Basis, aspect := -1.0, mirror := false, depth := 0.0) -> Vector3:
	var a := aspect if aspect > 0.0 else _spread.aspect
	var x := (2.0 * ComicSpread.SPINE - p.x) if mirror else p.x
	# THE TURN HINGES ON THE SPINE, and on a spread the spine is the origin - so unlike the
	# single page, which had to rotate about an off-centre pivot to avoid reading as a card
	# spinning in place, there is no pivot term here at all. The leaf's rotation is simply
	# part of `m`.
	return m * Vector3(
		(x - ComicSpread.SPINE) * PAGE_W,
		-(p.y - a * 0.5) * PAGE_W,        # spread y is DOWN, world y is up
		-depth * PAGE_W)                  # behind the sheet, along its own normal


# --- drawing -----------------------------------------------------------------

func _draw() -> void:
	if _spread == null:
		return
	_lens.eye = _eye
	_lens.look = _look
	_lens.fov = _fov
	_lens.prepare()
	var u := minf(_stage_size.x, _stage_size.y)
	var origin := _stage_size * 0.5
	# The bookend belongs to the vehicle here (see Vehicle.owns_bookend): the whole spread
	# fades, paper and all, instead of one panel fading inside a lit page.
	modulate.a = 1.0
	var fade := clampf(_bookend, 0.0, 1.0) * _dip_alpha()
	_backdrop(fade)          # once per frame, under everything, including through a turn

	# A PAGE TURN, as a real leaf on a real spine.
	#
	# The right-hand page of the outgoing spread swings left about the spine. Its FRONT is
	# that page; past the halfway point - where the leaf is edge-on and infinitely thin, so
	# the swap cannot be seen - its BACK is the incoming spread's LEFT page, mirrored about
	# the spine so that at a half turn it lands exactly where that page is about to be drawn
	# flat. Nothing cross-dissolves and nothing jumps: the leaf simply becomes the page it
	# always was the other side of.
	#
	# Under it: the incoming RIGHT page, revealed as the leaf lifts off it, and the outgoing
	# LEFT page, covered as the leaf comes down on it. Painter's order is exactly that -
	# both flat pages first, the leaf over them.
	if _turn_t >= 0.0 and _prev != null:
		var k := smoothstep(0.0, 1.0, clampf(_turn_t / TURN_TIME, 0.0, 1.0))
		var th := k * TURN_ARC
		_draw_leaf(_spread, 1, _pool, _is_cast, _att_basis, u, origin, fade, false)
		_draw_leaf(_prev, 0, 1 - _pool, _all_drawn, _att_basis, u, origin, fade, false)
		var hinge := _att_basis * Basis(Vector3.UP, -th)
		if th < TURN_ARC * 0.5:
			_draw_leaf(_prev, 1, 1 - _pool, _all_drawn, hinge, u, origin, fade, false)
		else:
			_draw_leaf(_spread, 0, _pool, _is_cast, hinge, u, origin, fade, true)
		return
	_draw_leaf(_spread, 0, _pool, _is_cast, _att_basis, u, origin, fade, false)
	_draw_leaf(_spread, 1, _pool, _is_cast, _att_basis, u, origin, fade, false)


## Has panel [param i] of the CURRENT spread been cast yet? (See _turn_spread's queue.)
func _is_cast(i: int) -> bool:
	return i < _cast.size() and _cast[i] != null and is_instance_valid(_cast[i]) \
		and i < _warm.size() and int(_warm[i]) >= FIRST_DRAW


## Every panel of the spread turning away carries a picture - it was fully cast before the
## turn started, and stopping its viewports kept what they had drawn.
func _all_drawn(_i: int) -> bool:
	return true


## ONE PAGE of [param sp] - the unit a leaf turns, and therefore the unit everything draws
## in. [param side] is 0 for the left page and 1 for the right.
##
## [param m] is the full placement transform (see _world): the sheet's attitude for a page
## lying flat, or that times the hinge rotation for the leaf in the air. [param mirror]
## reflects about the spine, which is how the leaf's back face draws the page printed on it.
##
## [param drawn] says which panels carry a picture. For the spread being read that is "has
## this one been cast yet" - the queue casts one per frame, so for the first few frames a
## late panel is still blank paper. For the spread turning away it is every panel, because
## it was fully cast before the turn began.
func _draw_leaf(sp: ComicSpread, side: int, pool: int, drawn: Callable, m: Basis,
		u: float, origin: Vector2, fade: float, mirror: bool) -> void:
	if fade <= 0.003 or sp == null:
		return
	var paper := Color(_paper.r, _paper.g, _paper.b, fade)
	var ink := Color(_ink.r, _ink.g, _ink.b, fade)
	# The sheet's shadow on the surface under it, before the sheet itself.
	_shadow_quad(sp, side, m, u, origin, fade, mirror)
	# THE PAPER, subdivided so its gradient is per-vertex rather than per-quad, and so a
	# steeply raked page still shades smoothly across its length.
	_paper_quad(sp, side, m, u, origin, paper, mirror)
	# THE PANELS. Painter's order along the page normal is irrelevant - they are coplanar
	# and never overlap - so reading order is the order, which is also the order the ink
	# wants (a canted panel's border must sit over its neighbour's paper, not under it).
	var on_page := sp.page_panels(side)
	for i in on_page:
		if bool(drawn.call(i)):
			_panel_quad(sp, int(i), pool, m, u, origin, fade, mirror)
	for i in on_page:
		_panel_ink(sp, int(i), m, u, origin, paper, ink, bool(drawn.call(i)), mirror)


## The surface the sheet lies on: a vignetted wash over the whole frame, painted first.
## See the note on DESK_VIGNETTE for why this is not a quad in the world.
func _backdrop(fade: float) -> void:
	var pts := PackedVector2Array()
	var cols := PackedColorArray()
	var idx := PackedInt32Array()
	var n := 6
	for j in n + 1:
		for i in n + 1:
			var fu := float(i) / float(n)
			var fv := float(j) / float(n)
			pts.append(Vector2(fu * _stage_size.x, fv * _stage_size.y))
			var d := Vector2(fu - 0.5, fv - 0.5).length() / 0.707
			var k := lerpf(1.0, 1.0 - DESK_VIGNETTE, clampf(d * d, 0.0, 1.0))
			cols.append(Color(_desk.r * k, _desk.g * k, _desk.b * k, fade))
	for j in n:
		for i in n:
			var a := j * (n + 1) + i
			idx.append_array([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1])
	RenderingServer.canvas_item_add_triangle_array(get_canvas_item(), idx, pts, cols)


## One page's shadow on that surface - its outline, offset and pushed back. It is most of
## what makes the paper read as an object lying on something rather than as a rectangle
## floating in a void, and on the leaf it is what says the page has LIFTED.
func _shadow_quad(sp: ComicSpread, side: int, m: Basis, u: float, origin: Vector2,
		fade: float, mirror: bool) -> void:
	var poly := PackedVector2Array()
	for p in sp.page_corners(side):
		var q: Vector2 = p + Vector2(SHADOW_OFF.x, SHADOW_OFF.y * sp.aspect)
		var pr := _lens.project(_world(q, m, sp.aspect, mirror, SHADOW_DEPTH))
		if pr.z <= _lens.near:
			return
		poly.append(Vector2(pr.x, pr.y) * u + origin)
	if _poly_area(poly) < 1.0:
		return
	draw_colored_polygon(poly, Color(0.0, 0.0, 0.0, 0.42 * fade))


## The page itself: a subdivided quad with a gentle darkening toward the edges. Flat white
## paper reads as a blank canvas; a whisper of a gradient reads as a printed sheet.
func _paper_quad(sp: ComicSpread, side: int, m: Basis, u: float, origin: Vector2,
		paper: Color, mirror: bool) -> void:
	var pts := PackedVector2Array()
	var cols := PackedColorArray()
	var ok := []
	var n := PAPER_GRID
	var x0 := float(side)
	for j in n + 1:
		for i in n + 1:
			var fu := float(i) / float(n)
			var fv := float(j) / float(n)
			var pr := _lens.project(_world(Vector2(x0 + fu, fv * sp.aspect), m,
				sp.aspect, mirror))
			var good := pr.z > _lens.near
			ok.append(good)
			pts.append((Vector2(pr.x, pr.y) * u + origin) if good else Vector2.ZERO)
			# radial falloff from the sheet's centre, at a few percent
			var d := Vector2(fu - 0.5, fv - 0.5).length() / 0.707
			var kk := 1.0 - 0.10 * d * d
			cols.append(Color(paper.r * kk, paper.g * kk, paper.b * kk, paper.a))
	var idx := _cells(ok, n)
	if idx.is_empty():
		return
	RenderingServer.canvas_item_add_triangle_array(get_canvas_item(), idx, pts, cols)


## Index the cells of an (n+1)x(n+1) vertex grid, SKIPPING any cell with a vertex behind the
## eye.
##
## The alternative - what every one of these draws used to do - is to abandon the whole
## surface the moment one vertex went behind the camera. On a single page that was rare
## enough to be invisible. On a spread it is not: the sheet is twice as wide, so at a rake
## the far corners go behind the eye at framings that are otherwise perfectly good, and a
## page that vanishes entirely for a second is a far worse artefact than a page missing the
## cell that could not be projected.
## RETURNS the buffer rather than filling one handed in: a PackedInt32Array is a value
## type in GDScript, so a callee that appends to a parameter appends to its own copy and the
## caller draws nothing at all.
func _cells(ok: Array, n: int) -> PackedInt32Array:
	var idx := PackedInt32Array()
	for j in n:
		for i in n:
			var a := j * (n + 1) + i
			var b := a + 1
			var c := a + n + 1
			var d := a + n + 2
			if bool(ok[a]) and bool(ok[b]) and bool(ok[c]) and bool(ok[d]):
				idx.append_array([a, b, d, a, d, c])
	return idx


## One panel's picture: its render target mapped onto a GRID-subdivided quad whose every
## vertex is projected individually. Two triangles would be affine and would warp (28.5 px
## on a 512 px frame, measured in tests/vehicle_probe.gd); subdivision makes the mapping
## piecewise-perspective and the error vanishes.
func _panel_quad(sp: ComicSpread, i: int, pool: int, m: Basis,
		u: float, origin: Vector2, fade: float, mirror: bool) -> void:
	var vp: SubViewport = _slots[pool * POOL + i]
	var rid := vp.get_texture().get_rid()
	if not rid.is_valid():
		return
	# HALF A TEXEL IN, on every side. UVs of exactly 0 and 1 sample the outermost texel's
	# EDGE, where the bilinear filter reaches for a neighbour that is not there - and what
	# it finds depends on the repeat mode, which showed as a stray coloured fringe along
	# the bottom rule of a panel. Insetting by half a texel keeps every sample inside the
	# panel's own picture; the loss is half a pixel of a 700-pixel render.
	var tw := maxf(float(vp.size.x), 1.0)
	var th := maxf(float(vp.size.y), 1.0)
	var inset := Vector2(0.5 / tw, 0.5 / th)
	var n := _grid_for(sp, i, m, u, origin, mirror)
	var pts := PackedVector2Array()
	var uvs := PackedVector2Array()
	var cols := PackedColorArray()
	var ok := []
	var col := Color(1, 1, 1, fade)
	for jj in n + 1:
		for ii in n + 1:
			var uv := Vector2(float(ii) / float(n), float(jj) / float(n))
			var pr := _lens.project(_world(sp.panel_point(i, uv), m, sp.aspect, mirror))
			var good := pr.z > _lens.near
			ok.append(good)
			pts.append((Vector2(pr.x, pr.y) * u + origin) if good else Vector2.ZERO)
			# THE UV IS NOT MIRRORED. Reflecting the page about the spine reflects where the
			# panel IS, never what is printed in it - a page seen from behind would be a
			# page seen from behind, and the leaf's back is a page seen from the FRONT.
			uvs.append(inset + uv * (Vector2.ONE - inset * 2.0))
			cols.append(col)
	var idx := _cells(ok, n)
	if idx.is_empty():
		return
	RenderingServer.canvas_item_add_triangle_array(
		get_canvas_item(), idx, pts, cols, uvs,
		PackedInt32Array(), PackedFloat32Array(), rid)


## How finely to subdivide this panel's grid: enough cells that each is about CELL_PX across
## on screen. Measured off the panel's own projected size, so a panel filling the frame gets
## a fine grid and one at the edge of shot gets a coarse one.
func _grid_for(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		mirror: bool) -> int:
	var lo := Vector2(1e9, 1e9)
	var hi := Vector2(-1e9, -1e9)
	var wraps := false
	for uv in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		var pr := _lens.project(_world(sp.panel_point(i, uv), m, sp.aspect, mirror))
		if pr.z <= _lens.near:
			wraps = true
			continue
		var p := Vector2(pr.x, pr.y) * u + origin
		lo = lo.min(p)
		hi = hi.max(p)
	if wraps:
		# A corner behind the eye means the panel is unboundedly large in screen space, so
		# the projected span says nothing. Size it to the SCREEN instead: a cell can never
		# usefully be finer than the frame it is drawn into, and the old answer here was a
		# flat GRID_MAX - a 40x40 grid, 1681 vertices, on a panel that is mostly off shot.
		return clampi(int(ceil(maxf(_stage_size.x, _stage_size.y) / CELL_PX)),
			GRID_MIN, GRID_MAX)
	var span := maxf(hi.x - lo.x, hi.y - lo.y)
	return clampi(int(ceil(span / CELL_PX)), GRID_MIN, GRID_MAX)


## The gutter side of a panel: paper painted back over the square corners the rounded
## outline cuts off, then the ink border on top.
##
## Rounding this way rather than by clipping the texture is exact under perspective and
## costs nothing: the corner wedges are computed in SPREAD space and projected like every
## other point on the sheet, so they land on the picture's corner however the page is
## raked. Clipping would have needed a stencil or a shader, for a shape the paper can
## simply cover.
func _panel_ink(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		paper: Color, ink: Color, filled: bool, mirror: bool) -> void:
	if sp.radius > 0.0004 and filled:
		for w in _corner_wedges(sp, i, m, u, origin, mirror):
			var poly: PackedVector2Array = w
			# A wedge is a few pixels across at the wide shot, and BELOW a pixel the canvas
			# triangulator rejects it outright ("Invalid polygon data, triangulation
			# failed", once per corner per panel per frame). Same guard Plane3D carries for
			# the same reason: sub-pixel geometry contributes nothing and costs an error.
			if poly.size() >= 3 and _poly_area(poly) >= 1.0:
				draw_colored_polygon(poly, paper)
	# An UNFILLED panel is drawn as an empty ruled frame - the page is a comic being
	# drawn, and the panels ahead of the story are simply not inked in yet. Lighter, so
	# it reads as ruled rather than as a black hole.
	var line := ink if filled else Color(ink.r, ink.g, ink.b, ink.a * 0.35)
	# THE BORDER IS DRAWN IN RUNS. A rule with one point behind the eye used to abandon the
	# whole outline; on a spread that happens often enough to be seen as panels losing their
	# borders in bursts, so the visible part is drawn and only the part that cannot be
	# projected is dropped.
	for run in _rounded_runs(sp, i, m, u, origin, mirror):
		var pts: PackedVector2Array = run
		if pts.size() >= 2:
			draw_polyline(pts, line, _ink_width(), true)


## Comic rules are CHUNKY - a hairline reads as a UI border around a video, which is
## exactly what this vehicle exists not to be. Scaled off the shorter screen axis so it
## is the same weight at 720p and at 4K, and sampled per session inside a narrow band
## (some books ink heavier than others; none ink thin).
## Twice the signed area of a polygon, absolute - the cheap "is this bigger than a pixel"
## test.
func _poly_area(p: PackedVector2Array) -> float:
	var a := 0.0
	for i in p.size():
		var q: Vector2 = p[i]
		var r: Vector2 = p[(i + 1) % p.size()]
		a += q.x * r.y - r.x * q.y
	return absf(a) * 0.5


func _ink_width() -> float:
	return maxf(1.5, minf(_stage_size.x, _stage_size.y) * _ink_weight)


## The panel's outline in screen space as RUNS of consecutive projectable points, corners
## rounded in SPREAD space. A fully visible panel gives exactly one run, closed.
func _rounded_runs(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		mirror: bool) -> Array:
	var uv := _rounded_uv(sp, i)
	var scr := PackedVector2Array()
	var ok := []
	for p in uv:
		var pr := _lens.project(_world(sp.panel_point(i, p), m, sp.aspect, mirror))
		ok.append(pr.z > _lens.near)
		scr.append(Vector2(pr.x, pr.y) * u + origin)
	var all := true
	for g in ok:
		if not bool(g):
			all = false
			break
	if all:
		var ring := scr.duplicate()
		ring.append(scr[0])
		return [ring]
	var out: Array = []
	var run := PackedVector2Array()
	for j in scr.size():
		if bool(ok[j]):
			run.append(scr[j])
		elif run.size() >= 2:
			out.append(run)
			run = PackedVector2Array()
		else:
			run = PackedVector2Array()
	if run.size() >= 2:
		out.append(run)
	return out


## The outline as points in the panel's own unit square. Radius is expressed in units of
## the SHORTER side and converted per axis, so a wide panel's corners stay circular
## instead of stretching into ellipses.
func _rounded_uv(sp: ComicSpread, i: int) -> PackedVector2Array:
	var r: Rect2 = sp.panels[i]["rect"]
	var short := minf(r.size.x, r.size.y)
	var rad := sp.radius * short
	var out := PackedVector2Array()
	if rad <= 0.0004:
		return PackedVector2Array([Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)])
	var ru := rad / r.size.x
	var rv := rad / r.size.y
	# corner centres in uv, and the angle each arc sweeps from
	var arcs := [
		[Vector2(ru, rv), PI],
		[Vector2(1.0 - ru, rv), -PI * 0.5],
		[Vector2(1.0 - ru, 1.0 - rv), 0.0],
		[Vector2(ru, 1.0 - rv), PI * 0.5],
	]
	for a in arcs:
		var c: Vector2 = a[0]
		var start: float = a[1]
		for s in CORNER_SEGS + 1:
			var ang := start + (PI * 0.5) * float(s) / float(CORNER_SEGS)
			out.append(c + Vector2(cos(ang) * ru, sin(ang) * rv))
	return out


## The four corner cut-offs, in screen space - the area between the square corner and the
## arc, which is painted back in paper so the picture stops at the rounded edge.
func _corner_wedges(sp: ComicSpread, i: int, m: Basis, u: float, origin: Vector2,
		mirror: bool) -> Array:
	var r: Rect2 = sp.panels[i]["rect"]
	var short := minf(r.size.x, r.size.y)
	var rad := sp.radius * short
	var ru := rad / r.size.x
	var rv := rad / r.size.y
	var specs := [
		[Vector2(0, 0), Vector2(ru, rv), PI],
		[Vector2(1, 0), Vector2(1.0 - ru, rv), -PI * 0.5],
		[Vector2(1, 1), Vector2(1.0 - ru, 1.0 - rv), 0.0],
		[Vector2(0, 1), Vector2(ru, 1.0 - rv), PI * 0.5],
	]
	var out: Array = []
	for sp_i in specs:
		var corner: Vector2 = sp_i[0]
		var c: Vector2 = sp_i[1]
		var start: float = sp_i[2]
		var uv := PackedVector2Array([corner])
		for s in CORNER_SEGS + 1:
			var ang := start + (PI * 0.5) * float(s) / float(CORNER_SEGS)
			uv.append(c + Vector2(cos(ang) * ru, sin(ang) * rv))
		var scr := PackedVector2Array()
		var ok := true
		for p in uv:
			var pr := _lens.project(_world(sp.panel_point(i, p), m, sp.aspect, mirror))
			if pr.z <= _lens.near:
				ok = false
				break
			scr.append(Vector2(pr.x, pr.y) * u + origin)
		if ok:
			out.append(scr)
	return out
