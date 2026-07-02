# the-point: full scene build

Brief: `../tunes/compositions/the-point/` (script / structure / style / video.txt).
33s percussion-driven iPod-style ad. Pure-black void, static symmetric shot, two
slots L/R. A human eye becoming its digital self (a Prism) then coming alive, then
a swarm. 15 numbered beats in `video.txt`.

## Decomposition decision

The 15 beats are NOT 15 scene files. Many consecutive beats share the SAME objects
on screen (an eye + a prism through beats 3-5; two prisms through 6-11). Fifteen
one-second scenes would each need to hand off the full pose of every object - heavy,
fragile. Instead I slice by **object-composition**: one scene per distinct set of
things on screen, with internal phase timelines for the sub-beats, and a **morph
handoff** at each object change (the framework's content-aware transition:
`morph_out`/`morph_in` + `morph_payload`/`begin_morph`, an instant swap where the
incoming scene continues the outgoing one's live bodies).

| scene         | beats | on screen            | morph_in    | morph_out   |
|---------------|-------|----------------------|-------------|-------------|
| `eye`         | 1     | 1 eye                | -           | `eye`       |
| `two_eyes`    | 2     | 2 eyes               | `eye`       | `eyes`      |
| `eye_prism`   | 3-5   | 1 eye + 1 blue prism | `eyes`      | `eye2prism` |
| `two_prisms`  | 6-11  | blue + red prism     | `eye2prism` | `prisms`    |
| `prism_swarm` | 12-15 | swarm -> helix       | `prisms`    | -           |

Handoff carries **live RefCounted bodies** (the same `EyeBody`/`PrismBody`
instances) in the payload dict, so an object literally continues - no re-instantiation,
no pose snapping. This is the key framework-native trick.

## Coordinate continuity

- `eye`/`two_eyes`/`eye_prism` are `Scene3D`: lens.eye=(0,0,4), fov 48. Eyes at world
  (±off, 0, 0). `lens.project(p)` -> unit-fraction offset from center; `* unit()` = px.
- `eye_prism` keeps the SAME lens/off so the surviving LEFT eye never moves at the
  morph; the blue prism is drawn (2D `PrismBody`) at `lens.project(+off,0,0)` - exactly
  where the right eye was.
- `two_prisms`/`prism_swarm` position prisms in unit-fraction offsets from center so
  they line up with the projected slots. Payloads pass slot fractions, not pixels.

## Internal timelines (driven by a scene-local `_t`, reset in build_params AND begin_morph)

- **eye_prism** (~8s): 0-2 crystallize right eye -> blue prism (dissolve+form flash);
  2-5 prism looks around, eye watches; 5-8 the eye trembles/vibrates faster, light
  builds (the riser). Exits on the DROP -> morph to two_prisms.
- **two_prisms** (~16s): burst red prism from the eye's slot on entry (the drop);
  phase-lock (both rot together, `lock_pose_to`); scan unison; blue desyncs & looks
  around while red holds; both unlock from anchors and sway (rubber-band `_disp`);
  specialize - blue swells + slow pulse, red shrinks + quickens. Exit -> swarm.
- **prism_swarm** (~5s, loop=false so it holds the card frame): gather 6-7 blue in
  formation; fly forward along a track (depth stream, receding); track blossoms into a
  double helix (blue strand + red strand, opposing twist); swarm commits to ONE side
  and jumps lanes; hold with room for a logo/date card.

Camera stays static until prism_swarm (brief guardrail: travel only from the swarm on).

## Storyboard

`storyboards/default.json` ("the-point") rewritten to the 5-scene chain with timecoded
`hold`s matching the beats, `loop:false` (hold the final card). Morphs fire
automatically because each entry's `morph_in` matches the previous `morph_out`.
</content>
