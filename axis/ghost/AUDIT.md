# ghost - live rendering performance audit

Dated 2026-09-03. Run on the author's machine: Intel i7-4790K (4 cores / 8 threads, 2014),
RTX 5060 Ti 16 GB (driver 610.57), 31 GiB RAM, Arch Linux, Godot 4.7.2, launched the way the
author launches it (`godot --path axis/ghost/`).

Everything below is measured. Two new probes were written for it and they are the instrument,
not the argument - re-run them and the numbers come back:

- `tests/perf_probe.gd` - splits a live frame into the parts that can be named, per scene and
  per vehicle. See the file header for what each column means.
- `tests/draw_cost_probe.gd` - two controls the whole audit rests on: what one canvas draw call
  costs from GDScript, and whether a nested SubViewport obeys its parent.

---

## The verdict, first

**ghost is not slow because of the GPU, the renderer, the resolution, or the comic page. It is
slow because a ghost frame is tens of thousands of GDScript calls on one thread of a 2014 CPU,
and the largest single term is the cost of ISSUING a canvas draw call rather than of drawing
anything.** The card is idle at 0.2 to 0.9 ms per frame while the frame takes 30 to 170.

That is why the comparison with modern games does not hold and should stop being troubling. A
modern game is a C++ engine that spends its frame on the GPU and its CPU time across worker
threads; its 2014 CPU is doing very little. ghost's frame is an interpreted language walking
per-shape loops on the main thread, and a 4790K core is roughly where it was in 2014. The card
being new buys nothing, because nothing is asked of it.

The comic did not introduce the problem. It multiplied an existing one - three live scenes plus
a page whose every vertex is projected in GDScript - and pushed a frame that was already over
budget far enough that it became unusable.

**The three fixes with real measured headroom, in order:**

| | Fix | Measured basis | Expected recovery |
|---|---|---|---|
| 1 | Route per-shape drawing through `TriBatch` instead of per-shape `draw_*` calls | 7-26 us/shape becomes 1.2-1.7 us/shape for identical geometry | 4x to 20x on the draw term, which is 33-96% of the frame |
| 2 | Hoist the page basis out of `ComicVehicle._page_point` | ~2500 `Basis.from_euler` constructions per frame, all with the same argument | most of the comic's ~18 ms page cost |
| 3 | Make the stage governor throttle the thing that costs, or stop paying for it | a skipped frame currently saves ~0.5 ms of ~80 | the throttle becomes real rather than cosmetic |

---

## How this was measured, and what is not ghost's cost

`perf_probe` drives a real session exactly as the Director drives one - `update()`, then
`view.commit()`, then `vehicle.advance()` - and splits the frame four ways:

- **sim** - the `update`/`commit`/`advance` GDScript, timed directly.
- **draw** - every `_draw` callback the frame queued. Timed by bracketing the engine's
  MessageQueue with two `call_deferred` markers, one before the sim and one after, so the
  measurement covers the focal scene's `_draw`, the vehicle's, and every live panel's, and
  nothing else.
- **render cpu / gpu** - `RenderingServer.viewport_get_measured_render_time_*`, summed over
  every viewport in the tree (root, stage, and the comic's twelve panel slots).
- **other** - the remainder.

**The `other` column is the harness and must not be read as ghost's cost.** Under `xvfb` it sits
at a near constant 33 to 43 ms; the identical run under `--headless` puts it at 0.4 ms. That is
the virtual display's present path, and it is why every absolute frame time below is inflated by
roughly 35 ms. The comparison that matters is `sim + draw`, which is ghost's own CPU, and it is
identical in both.

The machine was also under real load throughout: a PyTorch training run pinned at 100% of one
core with 8 GB resident, the box 27 of 31 GiB used and **25 GiB into swap**, plus Firefox and
Electron, load average 3.9. That inflates everything further and is discussed at the end.

---

## What a frame is actually made of

### The whole catalogue, one scene at a time, full frame, 1920x1080

71 registered `{scene, behavior}` pairs, 40 frames each. Means:

| | sim | draw | render cpu | gpu |
|---|---|---|---|---|
| mean over 71 pairs | 2.1 ms | **11.4 ms** | 0.47 ms | 0.34 ms |

The GPU is doing nothing. The RenderingServer's own CPU work is 0.5 ms. The frame is `sim + draw`,
and `draw` dominates it by five to one.

The worst offenders, and note that they fail in two completely different ways:

| scene | sim | draw | draw calls | primitives |
|---|---|---|---|---|
| terrain/drift | 19.7 | **152.8** | 2 | 25k |
| terrain/static | 19.9 | **142.6** | 2 | 25k |
| starfield/static | 0.1 | **56.3** | 3029 | 194k |
| cloth/drift | 5.8 | 34.5 | 89 | 2.7k |
| eye_prism/static | 0.1 | 31.2 | 672 | 11k |
| two_eyes/static | 0.1 | 28.7 | 547 | 8.7k |
| starfield/drift | 0.1 | 25.8 | 916 | 59k |
| rocks/drift | 0.2 | 23.8 | **3** | 19k |
| canopy/drift | **20.9** | 1.9 | 31 | 26k |
| terrain_city/drift | **20.6** | 0.0 | 45 | 25k |
| spires/drift | **20.3** | 0.1 | 277 | 25k |
| voxel_blocks/static | 0.5 | 20.1 | 1200 | 2.4k |

`starfield/static` and `voxel_blocks` are **call-bound**: thousands of draw calls, and the time
tracks the call count, not the geometry (voxel_blocks issues 1200 calls for 2400 primitives and
pays 20 ms for it). `terrain` and `rocks` are **allocation-bound**: two or three draw calls and
still 24 to 152 ms. `canopy`, `spires` and `terrain_city` are **sim-bound** despite having
adopted FrameForge. Three separate diseases, and each needs its own treatment.

### The comic vehicle against the full frame, same scene, same seed

The cleanest isolation available. `wire_solid` costs 0.1 ms of sim and 0.7 ms of draw when it is
alone on a full frame:

| run | sim | draw | frame |
|---|---|---|---|
| `wire_solid` alone, full frame | 0.1 | 0.7 | - |
| `wire_solid` in the comic, 8 cuts, 1-3 live panels | 0.5 | **19.1** | 55.4 |

**The comic page adds roughly 18 ms of `_draw` per frame to a scene that costs 0.7 ms.** That
is the vehicle's own drawing, not the panels' contents.

### The comic with real scenes

Ten cuts, mixed catalogue, 1-3 live panels:

| scene | live panels | sim | draw | gpu |
|---|---|---|---|---|
| gaussian_landscape | 1 | 11.9 | 39.9 | 0.4 |
| terrain_city | 2 | 26.2 | 19.3 | 0.6 |
| planes | 2 | 8.2 | 39.7 | 0.9 |
| aurora | 3 | 3.8 | 45.4 | 0.7 |
| starfield | 0 | 0.5 | 35.7 | 0.2 |
| metropolis | 2 | 4.4 | 16.3 | 0.5 |
| **mean of 10** | | **9.6** | **30.1** | 0.7 |

And the same comic session run headless, where the harness contributes nothing:

> mean frame **29.4 ms** = sim 0.8 + draw 28.3 + render cpu 0.0 + other 0.4. **96% of the frame
> is GDScript `_draw`.**

So even with the display path removed entirely, and on the cheapest scene in the set, the comic
is at 34 fps of pure CPU before the real display, the UI, the audio analyzer or a heavy scene are
added. Put `terrain` or `aurora` in a panel and it is single digits.

---

## Finding 1 - the per-call cost IS the frame (broad, and the biggest single win)

`tests/draw_cost_probe.gd` draws **identical geometry** five ways on the same canvas in the same
frame. Only the number of GDScript-to-server round trips differs.

| mode | 1800 shapes, `_draw` ms | us/shape | draw calls |
|---|---|---|---|
| `draw_circle` per shape | 21.89 | 12.16 | 1800 |
| `draw_colored_polygon` per shape | 17.23 | 9.57 | 1800 |
| `fill_aa` (polygon + antialiased outline) | **47.02** | **26.12** | 7200 |
| `TriBatch` accumulate then one submit | **2.14** | **1.19** | 1 |
| raw arrays, one `canvas_item_add_triangle_array` | 1.54 | 0.85 | 1 |

Read that again: the same 3600 triangles cost 17.2 ms as 1800 calls and 2.1 ms as one. **The
geometry is free. The call is the cost.** Roughly 8 us of pure overhead per `draw_*` from
GDScript, and `draw_circle` adds tessellation on top.

`GhostScene.fill_aa` is the most expensive shape primitive in the project at 26 us - it issues
two calls per shape, and the second is an antialiased polyline, which the engine tessellates into
a triangle strip. Its own doc comment says it costs "one extra draw call per shape"; the measured
figure is that it costs **3.5x** a plain polygon and **22x** a batched one.

The codebase currently has 71 `draw_circle` call sites, 38 `draw_line`, 34 `draw_colored_polygon`,
30 `draw_polyline`, 30 `draw_rect`, 15 `draw_arc` and 11 `draw_polygon` across the scenes and the
shared drawers, against **5** files that use `canvas_item_add_triangle_array`. `TriBatch` exists,
it works, and it is barely used.

The worst concentration is `scripts/layer.gd`, which every scene composes for weather and
atmosphere. `Layer.glow` (`layer.gd:78`) is six `draw_circle` calls per glow; `Layer.soft_blob`
(`layer.gd:89`) is eight. `Stars.draw` (`layer.gd:683`) calls `Layer.glow` for every star bright
enough plus a `draw_circle` each - at the default 150 stars and a bright sky that is up to 900
calls, or **11 ms**, for the backdrop alone. That is one shared library, drawn by dozens of scenes,
and it explains `starfield/static` at 3029 calls exactly.

**Fix.** Give `Layer.Base` a `TriBatch` and convert the per-particle painters (`Stars`, `Snow`,
`Fireflies`, `Embers`, `Dust`, `Motes`, `Bubbles`, `Petals`) to accumulate into it, flushing once
per layer. A soft glow becomes a small textured quad or a few Gouraud triangles rather than six
tessellated circles. This is one file, it is the project's own already-proven mechanism, and it
reaches every scene that composes a layer. Then work outward to the scenes with the highest call
counts: `starfield`, `voxel_blocks`, `aurora`, `underwater`, `bubbles`, `harmonic_lattice`,
`eye_prism`, `two_eyes`, `cityscape`.

Expected recovery, from the table above: the draw term falls by roughly 5x to 8x wherever it is
call-bound. On the catalogue mean that is 11.4 ms down to roughly 2 to 3 ms.

---

## Finding 2 - the comic page rebuilds a rotation matrix per vertex (comic-specific, large)

`ComicVehicle._page_point` (`scripts/vehicles/comic.gd:979`, returning at line 991) ends with:

```gdscript
return Basis.from_euler(att) * local
```

`att` is the page's attitude. **It is constant for the entire frame.** Every point drawn on the
page goes through this call - by design, and the design is right, but the basis is rebuilt every
time. `Basis.from_euler` is six trigonometric evaluations and a 3x3 construction.

Count the callers per frame for a six-panel page at 1080p:

- `_panel_quad` subdivides each panel to `ceil(span / CELL_PX)` cells, clamped 6 to 40. A panel
  spanning a 1080-pixel frame gives n = 20, so **441 vertices**; several panels are in shot at once.
- `_grid_for` projects four more corners per panel, before the grid is even built.
- `_panel_ink` walks `_rounded` (24 points per panel) and `_corner_wedges` (4 wedges of 7 points).
- `_paper_quad` adds 25, `_shadow_quad` 4.

That lands at roughly **2000 to 3000 `Basis.from_euler` calls per frame**, all with the same
argument, plus a `Lens3D.project` each. It is consistent with the measured ~18 ms the page adds.

**Fix.** Compute the basis once per `_draw` (and once per `_ease`) and pass it down, or cache it
on the vehicle keyed by `att`. The spine basis in the same function has the same problem during a
page turn (`Basis(Vector3.UP, spine)`, rebuilt per vertex). This is a small, local, low-risk
change with no visual consequence whatsoever - the value is identical, it is simply computed
once instead of two thousand times.

**Second, smaller comic cost:** `_place_eye` calls `_fit` twice and `_cover` once every frame
(`comic.gd:754`; `_fit` at 870, `_cover` at 904). Each is a 22-iteration bisection that allocates a fresh `Lens3D` and calls
`prepare()` inside the loop - 66 iterations, 264 projections and 3 allocations per frame, for a
camera distance that changes smoothly. Hoist the `Lens3D` out of the loop (it is re-prepared each
iteration anyway) and consider solving at a lower iteration count; 22 halvings resolves to one
part in four million, which is far past what a camera distance needs.

---

## Finding 3 - the stage governor cannot throttle what is expensive

This one is measured directly and it changes how the governor should be understood.

`main.gd`'s governor throttles a heavy scene by rendering the stage every 2nd, 3rd or 6th frame:
on a skipped frame it sets the stage's `process_mode` to `DISABLED` and leaves its render target
un-updated. The claim in the source is that "the skipped frames are cheap".

`tests/draw_cost_probe.gd` puts that to the engine:

| stage state | panel `_draw` ran | stage `_draw` ran | draw calls | render cpu |
|---|---|---|---|---|
| stage ALWAYS, panel ALWAYS (level 0) | 6/6 frames | 6/6 frames | 1200 | 0.89 ms |
| stage DISABLED + process off, panel ALWAYS (a skipped frame) | **6/6 frames** | **6/6 frames** | 600 | 1.01 ms |

Two things follow, and both are load-bearing.

**(a) A skipped frame still runs every `_draw`.** `queue_redraw()` pushes the redraw callback onto
the engine's MessageQueue, which is flushed every idle frame regardless of whether the viewport
renders. The `Director` is an **autoload** - it is not in the stage's subtree, so
`process_mode = DISABLED` on the stage does not stop it. It keeps calling `_current.update()`,
which calls `queue_redraw()`, and the callback runs in full. What the skipped frame actually saves
is the rasterization: about **0.5 to 1 ms of a 30 to 90 ms frame**. The GDScript, which is 96% of
the cost, is paid every single frame at every governor level.

This is visible in the real app. A 60-second `--vehicle comic` session logs:

```
ghost: stage governor -> level 1 (active frame 81 ms)
ghost: stage governor -> level 2 (active frame 68 ms)
ghost: stage governor -> level 3 (active frame 62 ms)
```

It escalates to the deepest level and the active frame stays at 62 ms. It never recovers, because
throttling the render target does not remove the work. Running `perf_probe` with the stage forced
into the governed state confirms it: 79.0 ms against 75.4 ms ungoverned, i.e. no improvement at all.

**(b) A nested SubViewport ignores its parent's state.** The panel viewport rendered on all six
frames while its parent stage was disabled. The comic's live panels are `UPDATE_ALWAYS`
SubViewports nested inside the stage, and the governor never touches them - so at governor level 3,
with the stage frozen, up to three panels are still rendering their scenes at full resolution every
frame. The governor is structurally incapable of throttling the comic.

**Fix.** The governor must throttle at the source, not at the render target. On a skipped frame,
`Director` should skip `_tick_animation` and the vehicle's `advance` entirely (the schedule must
still run on the real music clock - that split already exists and is correct). Additionally,
`ComicVehicle` should expose a freeze so the governor can stop the panel viewports with the stage.
Until then the governor's cost should be understood as approximately zero benefit, and its
escalation messages in the log should not be read as the throttle working.

---

## Finding 4 - the allocation-bound scenes (`terrain` at 152 ms is the worst frame in the project)

`terrain` issues **2 draw calls** and spends **152 ms** in `_draw`. Nothing about that is a
draw-call problem. `Terrain.collect_surface` (`scripts/terrain.gd:538`) ends with:

```gdscript
quads.append({"d": ..., "poly": PackedVector2Array([...]),
    "cols": PackedColorArray([...]),
    "uvs": PackedVector2Array([...])})
```

per quad. At `RES` 112 that is about 12,300 quads, so roughly **50,000 heap allocations per frame**
(one Dictionary and three Packed arrays each), which are then sorted as an `Array` of Variants and
walked again to fill a `TriBatch`. `rocks` (3 calls, 24 ms) and `cloth` (89 calls, 35 ms) fail the
same way.

The file already knows this shape of problem - its own comment records a helper that "measured
22.3 ms/frame" and was inlined for exactly this reason. The remaining cost is the container, not
the shading.

**Fix.** Build directly into packed parallel arrays (`PackedVector2Array` of positions,
`PackedColorArray`, `PackedFloat32Array` of depths) and sort an index array by depth, rather than
materializing a Dictionary per quad. `TriBatch.painter_sort` already takes a native key; give it
indices instead of Dictionaries. Also: `terrain` is one of the 47 scenes that has **not** adopted
`FrameForge`, while the three scenes built on the same `Terrain` foundation have. It is the single
best candidate for adoption in the catalogue.

---

## Finding 5 - "off-thread" scenes still do 20 ms of main-thread work

`canopy`, `spires` and `terrain_city` all use `FrameForge`, and their `draw` is correspondingly
near zero (0.0 to 1.9 ms). Their `sim` is 20 ms. The forge moved the geometry build off the main
thread and left the rest behind:

- `Terrain.step_light` (`scripts/terrain.gd:421`) runs on the main thread every frame. It recasts
  `res/16` rows of shadow with an 18-step ray march per cell, then two separable blur passes over
  a wider band. At `RES` 112 that is on the order of 20,000 GDScript iterations per frame.
- `canopy.update` (`scripts/scenes/canopy.gd:824`) additionally marches 8 terrain height samples
  for line-of-sight, plus more for the camera floor, every frame.
- The snapshot itself duplicates per-element Dictionaries into the job.

**Fix.** `step_light` is a pure function of the light direction and the heightfield and is already
incremental - it belongs inside the forge job, not beside it. The camera's line-of-sight march can
run on a cadence (it eases over seconds; it does not need 60 answers a second).

---

## Finding 6 - engine and project configuration

Verified against `project.godot`, which is short. Ordered by measured relevance.

**The binary is a debug build, and this is the largest untested lever.** `godot --path
axis/ghost/` runs the project inside the Arch *editor* binary, which is compiled with
`DEBUG_ENABLED`. Every GDScript instruction carries range checks, type checks and profiler hooks
that an exported release template does not. For a workload that is ~96% GDScript this is not a
rounding error; published figures for release-vs-debug GDScript span roughly 1.5x to 3x.
**I did not measure this, because no Linux export template is installed** - only the Android ones
are present under `~/.local/share/godot/export_templates/4.7.2.stable/`. This is the cheapest
experiment left and it should be run first: install the 4.7.2 templates, export a Linux release
build, and run the same session. If it lands anywhere near 2x it changes the shape of everything
else in this document.

**Vsync is on** (never set, so `VSYNC_ENABLED` applies). At 30 to 90 ms frames this does not cause
the slowness, but it quantizes the result to multiples of 16.7 ms and it makes casual frame-rate
observation misleading. `perf_probe` disables it for the measurement; the app does not.

**`msaa_2d=2` (2x) is set project-wide.** It applies to the root viewport. The stage and the
comic's panel SubViewports have their own `msaa_2d`, which defaults to off, so the scenes are
almost certainly not being antialiased by this setting while the root - which draws one
`TextureRect` - is. `tri_batch.gd:125` already records that "Godot's msaa_2d does almost nothing
for this drawing path (measured)". Worth confirming and then either removing or moving to where it
would actually do something. Low impact either way: the GPU has the headroom.

**Physics runs at 60 Hz for nothing.** There is not a single `_physics_process` in the project. The
2D and 3D physics servers still step every frame. Setting
`physics/common/physics_ticks_per_second` low is free and harmless. Small.

**Forward+ for a 2D-only app.** Every viewport carries the clustered-forward setup. Given the
measured GPU time of 0.2 to 0.9 ms and RenderingServer CPU of 0.5 to 3 ms, **this is not worth
changing as a performance fix** and switching renderers risks the 3D-ish scenes. Recorded here so
it is not re-proposed: the renderer is not the problem.

**`rendering/driver/threads/thread_model` is unset** (single-threaded). Moving the RenderingServer
to its own thread would relieve at most the 0.5-3 ms render-cpu column. Not worth the stability
risk at these ratios.

---

## Finding 7 - the machine, honestly

Read at the time of the audit:

```
Mem:  31Gi total, 27Gi used, 802Mi free      Swap: 35Gi total, 25Gi USED, 10Gi free
load average: 3.86 3.59 3.20
PID 1168500  101% CPU  8.1 GB RSS  python /workspace/praxis/main.py --abstractinator-k
```

**25 GiB of swap in use on a 31 GiB machine, with 802 MiB free, is the more urgent problem in this
list.** A Godot process that touches a 1 GB working set under that pressure takes major page faults
in its frame loop, and those are milliseconds each. One of four cores is fully occupied by
training; Firefox and Electron take a further half core between them. Every number in this document
is inflated by that, probably substantially.

That said, it does not explain the shape of the result. The split - GPU idle, RenderingServer idle,
GDScript saturated - is a property of the code, not the load, and the headless control (which
removes the display path entirely) still shows 96% of the frame in `_draw`.

**The protocol for a clean re-measurement**, which should be run before acting on any absolute
number here:

```bash
# pause the training run
kill -STOP 1168500
cd /home/crow/repos/praxis/axis/ghost
GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/perf_probe.gd 400 -- \
    --vehicle comic --cuts 10 --frames 45 --seed 404
kill -CONT 1168500
```

And on the real display, where the `other` column becomes meaningful rather than an artifact of
the virtual one, the same probe can be run without `GHOST_PROBE_GPU` if a window is acceptable.

**On "my computer plays modern games fine".** It does, and that is consistent. A game asks the GPU
for nearly everything and spreads its CPU work over eight threads; the 5060 Ti answers and the
4790K coasts. ghost asks the GPU for 0.5 ms of work and asks one 2014 core to interpret tens of
thousands of GDScript operations. The hardware is not underperforming - it is barely being used.
The fix is not a faster machine, it is fewer calls.

---

## What to do, in order

1. **Install the Godot 4.7.2 Linux export templates and measure a release build.** Cheapest
   possible experiment, potentially the largest single multiplier, currently unknown.
2. **Batch `Layer`'s per-particle painters through `TriBatch`.** One file, proven mechanism,
   reaches every scene. Measured 5-8x on the draw term.
3. **Hoist the basis out of `ComicVehicle._page_point`.** Small, local, no visual change, removes
   ~2500 matrix constructions per frame.
4. **Fix the stage governor to skip the sim rather than the render target**, and let it freeze the
   comic's panel viewports. Currently it saves ~0.5 ms of ~80 and its log messages are misleading.
5. **Convert `Terrain.collect_surface` off per-quad Dictionaries**, then adopt `FrameForge` in
   `terrain`. Removes the worst frame in the project.
6. **Move `Terrain.step_light` into the forge job**, and put the camera line-of-sight march on a
   cadence. Removes 20 ms from three scenes.
7. Batch the remaining call-bound scenes: `starfield`, `voxel_blocks`, `aurora`, `underwater`,
   `bubbles`, `harmonic_lattice`, `eye_prism`, `two_eyes`, `cityscape`.
8. Housekeeping with small but free wins: drop the physics tick rate, resolve the `msaa_2d`
   question, hoist the `Lens3D` allocations out of `_fit` / `_cover`.

A note on ambition: items 2, 5 and 7 are the same fix applied in three places, and together they
address between one third and 96% of the frame depending on the scene. The project already built
the right tool for it in `TriBatch` and then used it in five files. Finishing that job is the
single highest-value piece of work available here.

---

## What I did not verify

- **The release-build multiplier.** No Linux template installed; stated as a hypothesis with a test,
  not as a finding.
- **Whether SubViewports inherit the project's `msaa_2d`.** Reasoned from the property existing
  per-viewport with an off default; not measured.
- **Real-display present cost.** Every run here went through `xvfb`, deliberately, because this
  project's convention is that no window may appear. The `other` column is therefore the virtual
  display's and tells you nothing about Wayland. The headless control bounds ghost's own CPU cost
  regardless.
- **Absolute frame times under an idle machine.** Everything was measured with the box in swap and
  a core pinned. Ratios and splits are robust to that; absolute milliseconds are not.
- The comic's page cost was isolated by differencing (`wire_solid` alone versus in a page) rather
  than by instrumenting `_draw_page` directly. The ~18 ms figure is a difference of means, not a
  direct timing.
