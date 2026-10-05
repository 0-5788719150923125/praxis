# ghost: the tarot mode

An automatic tarot reading. A SHOW is a document; each seed of it is an EPISODE that agents plan,
paint and write one card at a time, and ghost reads it aloud in the Generative voice at a table.
Built 2026-10-04. First show: `rift/tarot/truthful-tarot.md` ("Truthful Tarot" - the genre's
format and voice, told true).

## Using it

1. Splash -> **Tarot** (or `godot --path axis/ghost -- --tarot`). **Open…** the show
   (`rift/tarot/truthful-tarot.md`); **Edit brief…** edits its body - the rules, and the `## Cards`.
2. **New episode** draws a fresh seed from the OS's randomness - and makes nothing. **Generate**
   makes the episode in order: ~1 minute per picture (back, cloth, room, then each card), the
   reading written card by card as the paintings land. **⟳** on a row makes ONLY that part again;
   what was made from it is cleared and waits for Generate (the shuffle and the script, which cost
   nothing, follow on their own). **Writer** and **Painter** pick the agents (Claude or Codex for
   words, Codex for pictures) and, beside each, the MODEL: Claude's aliases (Fable, Opus, Sonnet -
   each the newest of its family; Default is Opus for the plan and reading, Sonnet for the card
   designs), Codex's from the catalog the CLI keeps (`~/.codex/models_cache.json`, listed entries
   only; Default is the author's config). Saved as `writer_model` / `painter_model`. The painter's
   model is the agent that asks for a picture; the picture is Codex's image tool's either way. **Folder** opens the episode: every prompt beside its reply.
3. **Speak** reads it at the table (the karaoke line tracks the voice). The voice is the panel's
   own and is saved in the show's frontmatter; **Test** auditions it.
4. **Export** renders the video, named after the episode's title; the episode's folder then holds
   `upload.md` - the title, a description, chapters timed per card from the take, and tags.
5. Any earlier episode is one pick away in the episode list, exactly as it was made.
6. **Delete…** removes the picked episode, after asking: its whole folder goes to the system
   trash (restorable from there); exported videos and the other episodes are untouched. The
   panel moves to the newest episode left.

## The pieces

| file | what it is |
|---|---|
| `scripts/text_gen.gd` | `TextGen`: who writes words - a registry beside `ImageGen` (`claude`, `codex`). Stateless writers; pictures go with the words. Claude runs with no tools; Codex is told to work from the message alone. |
| `scripts/agent_jobs.gd` | `AgentJobs`: one queue for every AI run (text and image) - limits per kind, LANES (one at a time, in order), timeouts, read-only refusal, and a rerun's folder cleared of the last run's output. Polled from `main._process`. |
| `scripts/reading_follower.gd` | `ReadingFollower`: where the voice is in a document - the tablet's sequential word match, and the schedule that puts actions in the voice's rests. |
| `scripts/tarot_episode.gd` | `TarotEpisode`: an episode on disk, one file per step, under `user://tarot/<show>/<seed>/`. A redo is a delete. |
| `scripts/tarot_producer.gd` | `TarotProducer`: makes whatever is missing, in drawing order. |
| `scripts/tarot_prompts.gd` | `TarotPrompts`: what each agent is told. Pure. |
| `scripts/tarot_script.gd` | `TarotScript`: the reading's marks; one walk gives the voice its text and the table its actions. |
| `scripts/tarot_deck.gd` | `TarotDeck`: the show's deck, parsed from its brief's `## Cards` section (or the standard 78, generated, meanings from `data/tarot/meanings.json`, CC0); the seeded shuffle; true-random seeds. |
| `scripts/tarot_table.gd` | `TarotTable`: what a look may name - title faces (`fonts/tarot/`, OFL), frames, the zones things stand in - `sanitize_look`, `sanitize_table`, and the layout a prompt can know ahead (`layout_of`, `headroom`). |
| `scripts/props.gd` | `Props`: things built from a description - shapes, materials, ornaments (registries an agent reads), `sanitize`, `build`. Generic; the tarot table is its first user. Shaders `prop.gdshader`, `prop_glass.gdshader`, `prop_lens.gdshader`, `prop_common.gdshaderinc`. |
| `scripts/tarot_cards.gd` | `TarotCards`: faces, backs and booklet pages, composed in 2D into stopped SubViewports. |
| `scripts/media/tarot.gd` | `TarotMedium`: the table. Pinned by the mode (`Medium.OWNED`, `Director.medium_override`). |
| `scripts/tarot_editor.gd` | `TarotEditor extends GenerativeEditor`: the panel. |

## The show document

Frontmatter: `title` is the channel. `ghost: tarot:` holds the knobs - `show` (the cache key,
fixed at the first episode), `seed`, `cards: [lo, hi]`, `reversals`, `jumpers`, `writer`,
`painter` - and the reader's voice (`voices`, `tab`, `picture`), exactly as a Generative chapter
carries its cast. The knobs are the document's exactly: a show opened with no block starts from
the defaults, never from the show open before it (whose key would put its episodes in that
show's folder). An export is of the episode open when it was asked for, whatever is picked
during its minutes of synthesis. The BODY is the BRIEF: what the show is, its reader, its rules. Every agent gets
it verbatim. The framework (the prompts in code) knows what a tarot video is; the brief knows
what THIS show is. One show document makes many episodes.

**The deck is the show's**, and it is data (the user, 2026-10-04: "the 78 cards are constant... we
should define the deck itself, then use true RNG"; and alternative decks with their own cards must
work too). A `## Cards` section of the brief lists it, one card per list item - an optional
numeral, the name, a colon, what it means (`- XVI. The Tower: ...`) - with `###` headings for
suits or groups; an indented line or nested bullet says more about the card above it. The heading
is `Cards` (or `The Cards`), never `Deck`: a section describing the deck's look in bullets would
become a deck of its bullets. Without one, the show reads the standard 78; `TarotDeck.standard_section()` writes
those out as a template (Truthful Tarot carries them, editable). Agents are handed the brief with
the card LIST taken out (`TarotDeck.strip`); a card's meaning reaches a writer only when it is
drawn. **No agent ever picks a card**: an episode's cards are a shuffle of the deck from its
seed, and a new episode's seed comes from the OS's cryptographic randomness
(`TarotDeck.true_seed`) and is kept, so the episode can be made again exactly. The draw writes the
cards themselves into `draw.json`, so an episode keeps what it drew whatever the deck becomes.

## The episode

| step | who | from |
|---|---|---|
| `plan` | producer (best tier) | brief, title, the seed's dice, earlier episodes |
| `draw` | ghost | the show's deck and the seed (shuffle, spread size, jumper) |
| `design:K` | deck designer (fast tier), one card per run | the look, card K, its traditional meaning |
| `image:back/surface/backdrop` | painter | the look |
| `table` | set dresser (best tier) | the plan, the cloth's painting, earlier episodes' tables - no card |
| `image:card:K` | painter | design K; the back + first + previous card as references |
| `say:intro` | reader (best tier) | the plan - no card |
| `say:K` | reader | the plan, every passage before, cards 1..K, and card K's PAINTING |
| `say:close` | reader | everything, and every card's painting |
| `script` | ghost | the passages and the marks between them |

**The reader looks at the cards** (the user, 2026-10-04: "otherwise, the text and the images are
sort of just disconnected"). A card's passage waits for that card's picture and is sent it -
upside down when the card came up reversed, as the viewer sees it - and is told to talk about
what is actually painted; the designer's plan for the painting is left out, since the painter
may have gone its own way. The close is sent the whole spread. Painting a card again therefore
rewrites its passage and every one after it. The writer takes pictures as stream-json content
blocks (`TextGen.Claude.compose`, at most 768 px on the long edge), and `prompt.txt` lists which
pictures went with the words.

**The cards move between passages, never inside one** (the user, 2026-10-05: "she's signaling the
draw before it even happened"). A passage is spoken whole and the table acts in the silence after
it, so the intro had been told to "end on the moment you stop shuffling to pull the first card" and
wrote "And there. That's the first one." - with a hesitation, trying to time the draw itself - a
beat before the card came out; a last card's "There. Beside the other two." came before it went
down. Every reader prompt now says when the cards move (`TarotPrompts.MOVES`): lead into the next
move, never report it - the acknowledgement belongs to the passage after it. Scripts written before
this keep their words until those passages are rewritten. Gate: tarot_check `_moves_after_words`.

The order is not written anywhere: a step starts when its inputs exist (`TarotEpisode.needs`).
A card's picture waits for the back and the card before it, but is not MADE from them: redoing
one picture leaves the rest of the deck alone (the Illustrations rule). There is no "shuffle
again": the draw is a function of the seed, so a different deal is a different episode.
A reply of the wrong shape (a position as a bare name, keywords as one string) is reshaped as it
lands; a step that fails says why on its row.

**No cheating.** The reader's passages are made in drawing order, each from the passages before
it and `TarotProducer._drawn(K)` - cards 1..K, nothing else. Every prompt is kept beside its
reply in `jobs/`. `tests/tarot_check.gd` asserts no later card's name, booklet or picture
reaches a reader prompt - the builder's and the producer's own (`say_prompt`) - two-sided.
A writer with tools could still read `draw.json` off the disk: Claude runs with none, and Codex
(whose `exec` keeps a read-only shell) is told plainly to work from the message alone and open no
file (`TextGen.Codex.ONLY_THIS`). The user's call, 2026-10-04: asking is enough - agents follow
it - and choosing the agent matters more than sealing it.

**Variety.** Each seed draws numeric dice (a place on Earth, a year, a hue, an hour, a
brainstorm pick) - no word lists - and the producer is shown earlier episodes' titles, topics,
decks and settings and told not to repeat them. The table samples its own layout from the seed
(camera, deck position, spread layout, shuffle moves, props, light).

**The card stock** (2026-10-05, the user: "most if not all generated tarot cards have a light tan
color"): asked only for "card stock", the producer printed all four decks on cream. The stock's
LIGHTNESS is now a die (`TarotPrompts.dice`, 5-95, its own rng so the other dice keep their
values) that the producer is held to - the one literal die; it picks the hue, ink and accent.
`TarotTable.sanitize_look` keeps the ink readable on whatever stock lands (`legible_ink`, WCAG
contrast >= `INK_CONTRAST` 3). Episodes planned before keep their cream until the plan is redone.
Gate: tarot_check `_card_stock`.

## The table

Static camera from the reader's chair (locked; the Tarot panel has no Camera dial), cloth from
the episode's `surface`, its `backdrop` out of focus beyond the far edge. The lamp is low and to
one side, so things throw shadows you can see.

THE SET TABLE (2026-10-05; the user: "the lack of objects upon the table makes the whole scene
rather boring... most tarot readers are very intentional about how they setup their
workspaces"). A step of its own, `table` (`table.json`): the SET DRESSER (`TarotPrompts.set_dresser`,
best tier) sets the reader's table from the plan, LOOKING AT the cloth (it waits for it; a new
cloth keeps the table), knowing no card. It invents the things of the episode's world - a
field station's hurricane candle, rain gauge and calabash of rainwater; a harbor's votive stone
anchor, hematite weights and murex shell - each with its reason (the four elements the suits
stand for, the reading's subject, light, devotion) and DESCRIBES EACH TO BE BUILT: parts of
shapes (`Props.SHAPES`: lathe profiles, rounded boxes, balls, crystal points and clusters,
rings, tubes along paths, sheets, blooms), in real centimeters, of named materials
(`Props.MATERIALS`, procedural surfaces in `shaders/prop.gdshader`: metal with tarnish, wax,
crystal, ceramic glaze, stone veins, wood grain...) with ornament worked in (`Props.ORNAMENTS`:
bands, flutes, stars, moons, zigzag... as relief, paint or inlay). It names each thing's zone
(`TarotTable.ZONES`), its group (things that stand together) and its turn, and is told each
zone's HEADROOM (`TarotTable.headroom`: the frame's top passes low over the far cloth, so the back
holds only short things). The reply is kept as written and made safe at every build
(`TarotTable.sanitize_table` over `Props.sanitize`: unknown shapes dropped, numbers clamped,
flames capped at `MAX_CANDLES`). Exactly the look's candle count is asked for; before a table is
set (or for an older episode) the look's candles stand alone (`TarotTable.default_table`). The
reader is told only that the reader's own things stand there - no passage depends on them, so
setting the table again keeps the reading. NO IMAGE-TO-3D: the user ruled it out on 2026-10-05
("way too slow, and my GPU is in constant use... I'm not going to expect users to load giant
models") - no Piper-sized model exists (TripoSR and SF3D are 1.6+ GB and want a GPU). Claude
writing the geometry was the user's idea.

BUILT IN THE ENGINE (`scripts/props.gd`, `Props.build`): meshes in meters, base on y = 0, each
part's surface laid out for its own girth and height so a motif keeps its shape. A candle's wax
part carries a `wick`: the flame is lit at its top, melted into a pool, and `drips` run down it.
SEEN-THROUGH material is glass for a vessel (`prop_glass`, a flame inside it shows) and a LENS
for a solid thing (`prop_lens`, screen refraction - a crystal ball turns the cloth over); a
clear crystal (clarity >= `Props.CLEAR`) is drawn as glass - opaque, a crystal ball was a pearl.
A flame lights everything but its own thing (an oil lamp's flame blew its own body white), whose
wax glows with the flicker instead (`flame` uniform). A reflection probe catches the table once
it is set, and only the things reflect it (`reflection_mask`): metal with nothing to reflect
read as paint. Look at a description alone with `tests/props_look_probe.gd`.

STONES (2026-10-05; the user: "a lot of tarot readers often have stones of all sizes: sometimes
large ones, though often a handful of small ones of various colors and properties", and "stones do
not always need to be in a dish. A lot of people would just arrange them on a table"). A `geode`
shape: a rough half rock, its hollow lined with points growing in (its rock plain stone unless
`base`). STREWN COPIES: `scatter`ed ones never touch (a crowded handful spreads wider rather than
pass through itself - three beach stones in a shell had crossed 94% of the time) and a `heap`
settles each copy into the lowest of a few spots, on the floor or resting on the copies under it,
so the floor fills before the heap rises; a `ring` can be an `arc`. A part's `material` can be a
LIST its copies take in turn (a mesh per material). THE PLAY OF LIGHT (`Props.PLAYS`, a
material's `play`), all of it lit by the engine, never emitted: silk (tiger's eye - Godot's own
ANISOTROPY, its frame turned across the fibers and rough enough to be broad), flash (labradorite -
streaky patches tipped as they lie, colored like metal), fire (opal), glitter (goldstone - tiny
tipped flakes that only the flames light), rainbow (aura quartz, paua, bismuth - a thin film in
what it reflects, its hues near the body's own: the whole rainbow read as tie-dye; a pale body
keeps its own light, or a cream shell read nearly black), glow
(moonstone); and `rings` bend a crystal's banding or a stone's veins round the middle of each copy
(agate, malachite) - every vertex carries its copy's middle and number (`CUSTOM0`, `Tris.placed`).
Lumpy balls have broad lumps (a fine second octave alone crimped a small stone like a dumpling).
FOUND ON THE WAY: `bump()` normalized a zero vector where a pore was seen edge-on, and the NaN
reached the reflection probe and blacked out everything that reflects it (a paua shell in bone);
it is guarded now and called on every pixel - a derivative in a branch some pixels skip is
undefined. A flag written "yes" crashed the sanitizer (`Props._flag`). The set dresser is told
that many readers keep stones, set on the cloth or heaped in a dish. Gate: tarot_check `_stones`.

WHERE A THING STANDS IS FOUND (`_place_things`, `_stand`): groups biggest first, a group's tallest
nearest its zone's middle, the rest round it - the shorter toward the reader; on the CLOTH (on the
wood by the rim it read as falling off), clear of everywhere the cards go (`_keep_out`), clear of
what stands (`THING_GAP`, `GROUP_GAP` within a group), wholly in the shot, and no group in front
of another in the picture - within a group a little in front at most (one seen through another
read as a stack). A lit thing stands only behind the middle (a candle in front of the cards
became the key and blew out the held card) and by dark cloth. A low thing may lie nearer the
reader. A thing with no room is tried smaller, then left off (logged).

NOTHING PASSES THROUGH WHAT STANDS (the user: "objects should probably have collision, and so
should the cards"): each thing's FOOT - its outline up to `Props.FOOT_H` - is an obstacle in the
planned wash. Every card moves from where it was to where the step puts it a few millimeters at a
time and is pushed back out of any foot it runs into, away from its middle (`_card_clear`,
`_push_out` by separating axes along that direction), so it slides along it and never jumps
through; a card is flung out only by a way clear of everything (`_path_clear`). The wash is still
planned once, a pure function of the seed - collision in the engine's physics would not be. Gate:
`tarot_place_check` (no card crosses a foot over 40 seeds; the same washes let through must).

CANDLES TOGETHER LIGHT THE CLOTH TOGETHER: the HEAT cap is on the cloth's hottest spot under ALL
the flames at once (`_heat_field` per flame, summed; every flame reaching the hottest cell dimmed).
Two tapers side by side, each at its own limit, burned a pale linen white through the bloom.

PAINTED OBJECTS WERE TRIED AND TAKEN OFF (2026-10-04): a photograph of one thing cut out of its
background brings its own camera, light and grade, and a flat card can neither take the
candlelight nor cast a shadow ("these 2D objects look awful").

FLAMES FLICKER APART (the user: "the candles flicker at exactly the same rate, which is wrong").
Every flame had one tempo and one swing, so they pulsed together. Each now has its own tempo,
steadiness and drafts: mostly a steady burn, and now and then a draft gutters it for a second or
two - a pure function of show time, so a render and a scrub see the same flame. The light leans
with its flame, so the shadows breathe. ROOM CANDLES OUT OF SHOT (the user's idea, "backlighting
behind the camera, which also flickers"): two or three shadowless, flickering lights behind the
camera and off to the sides - a warm fill on the cloth and on the card held up to the lens.
Gate: `tests/tarot_place_check.gd`.

EVERY LIGHT THROWS ITS OWN SHADOW (2026-10-05; the user, on a frame where "the rock casts a
shadow that is crescent-moon shaped": "most scenes have multiple light sources, and thus should
probably cast multiple shadows"). Every candle and the lamp cast, so a thing has a shadow for each
light near it, each turned from its own; where two fall together the shadow is darker. The KEY -
the candle nearest the middle that its cloth can take, else the lamp - is still the brightest, so
the deck's long shadow toward the reader leads. (Before, one key light cast alone - the user had
found every light casting strange: "the stack of cards casts no shadow... two independent shadows
for each side".) THE CRESCENTS WERE THE LAMP'S BIASES: Godot's defaults are made for rooms, and at
the lamp's distance they came to millimeters - more than a bowl's floor stands off the cloth - so
a bowl cast only its rim's ring, stones nothing, a geode a shadow standing off its foot; now
`shadow_bias` 0.005 / `shadow_normal_bias` 0.15 (the candles' were already 0.02 / 0.4). A
`light_size` on the lamp threw a white glare off a geode's rim, so its shadows soften by blur
alone. The room's out-of-shot candles stay shadowless (faint fills, six shadow passes each). Try a
lamp setting in seconds with `props_look_probe --lamp 1 --key 0 --lamp-shadow B,N,S`. NO CANDLE IS BRIGHTER
THAN ITS CLOTH ALLOWS: a key candle on the pale half of a "pine boards half covered by a felt
runner" surface flooded half the frame through the bloom, where the same light on the felt reads
as a candle (linear luminance 0.64 against 0.18); and on a near-black blanket a cream stripe just
behind the candles blew out, though the cloth round them was dark ON AVERAGE. So a candle's light
times the cloth's HOTTEST spot near it - lightness over the flame's falloff, height over distance
squared - is capped (`HEAT`), and only a candle whose cloth can take `KEY_MIN` may be the key -
on an all-pale cloth the lamp is. THE CLOTH HAS RELIEF:
its picture's luminance is its height, turned into a normal map natively
(`Image.bump_map_to_normal_map`, `RELIEF`), so the low light rakes the weave; the table wood too.

THE READER MARKS ITS DELIVERY (the user: "a real tarot reader will often speed up, or slow
down... get excited, or become serious"): `<!-- delivery: quicker, brighter -->` on its own line,
for the rest of that paragraph, and `<!-- hesitation -->` for a real stop. NOT THE TONE PRESETS -
the first cut leaned toward the panel's Tone presets and was rejected before it was heard: "they
all sound wildly different... like a totally different speaker... we need subtle inflection
shifts, pace shifts". So: three axes - pace (4% a step), brightness (a third of a semitone on the
host's formant-locked arc path, a little effort and melody) and pauses (a fifth) - each word a
step (`DELIVERY_WORDS`, intents as small combinations; a fourth axis, VOLUME - louder / softer,
effort alone - and a wider pause step, x1.3), two steps at most, and EASED: each
sentence goes half way from the last one's delivery (`_ease_leans`), in and back out. The host
takes `lean_semis` / `lean_effort` in `_discourse_plan` (`voice_host/test_lean.py`). A script mark
like any other (`ScriptMarks` "delivery"), so a Generative chapter can carry it too.

THE INTRO IS OUT OF FOCUS: the channel's name alone (no episode title - that is the video's, on
the platform) over the table behind a lens's bokeh, and the focus PULLS near to far as the shuffle
starts (`_tick_focus`). Otherwise there is NO depth of field: a far blur behind the table drew a
band along its far edge where the sharp table and the blurred room met, so the room's picture is
softened itself (`_soft`). The stage runs 4x multisampling and a 4096 shadow atlas with the key
light's quadrant whole, given back when the table leaves it. The camera is locked off; its dial is
gone from the panel.

THE SHUFFLE IS RUNS WITH SPARSE ACTIVITY (the user: "nonlinear behaviors with sparse
activations"): several riffles, a string of cuts, a few overhand passes - one kind at one tempo -
then nothing for a while (median ~3 s, now and then a long linger), in the MIDDLE of the table;
before the first card the deck is squared and pushed to its side (`TarotScript.PUSH`, which the
first draw's rest includes). A wash is rare, once at most, and a planned simulation: the deck
spread WIDE (some cards flung well out), two flat palms working it for most of half a minute, then
six to eight sweeps round the pile over ~7 s, each taking a few cards - some only pushed near, a few
missed and fetched by a later sweep - landing nearly squared, so the end is a light tidy, never
the whole spread arriving at once. THE MIXING IS MOST OF IT (the user, 2026-10-05: "BARELY
shuffled at all. 3 or 4 cards might shift slightly... no changing of z-order... maybe 5 or 10
seconds long"): it had been 5 s of two hands drifting through less than one slow loop. Each palm
now works a patch in a loop or two while traveling, lifts and comes down on the next, mostly on its
own side; a card under another drags less than the one on top, so cards slide over and under. The
ORDER is decided when two cards meet - the one sliding in on top - and held while they touch, so
it changes all through the wash and never through a card. A wash the first card would cut short is
replanned to fit, the same wash up to its own gather. Gate: `tests/tarot_wash_check.gd`. A draw: square, slide, flip, up to the camera on the LEFT beside the booklet page on the
RIGHT (shown, never read); both turn a little on their axes, and the card is now and then turned
to look at its back. A lay: page out, card down into the spread. A JUMPER FLIES OUT OF A SHUFFLE
(the user, 2026-10-05: "that jump should probably happen during a shuffle - not when the cards are
just sitting there on the table, doing nothing"): it had left the deck the moment the shuffle
stopped, often after seconds of the deck lying still. Its action now opens with one more riffle,
and the card rides the top of a half and springs off it while the halves fall (`JUMP_RIFFLE`,
`_jump_ride`); the deck goes aside once that riffle is done. The voice's rest for it grew to 5.0 s.
The channel's name opens it, and nothing closes it but the light: THE OUTRO (the user,
2026-10-05: "a simple fade to black, after the voice is done speaking... not display the title
again at the end") - a beat after the last word the table fades to black, reaching it exactly as
the outro's silence runs out (`_end_fade`: the take's own tail in a render, the Director's outro
live), a function of show time like the rest.

**Foil** (the user's idea, 2026-10-04): each painting's brightest, most colorful pixels - keyed
per picture from its own luminance percentiles, plus anything near the frame's accent color -
print as foil: metallic, a slow breath, a glint sweeping across now and then; the scene's bloom
lets it bleed. The look's `foil` (0-1) says how much.

Everything is a function of show time (`ReadingFollower` + the schedule), so live and export
draw the same frames.

## Probes and gates

- `godot --headless --path . --script res://tests/tarot_check.gd` - the gate.
- `tests/tarot_episode_probe.gd` - make an episode headlessly (real quota); `--only table` sets
  just the table.
- `tests/run_quiet.sh -- res://tests/props_look_probe.gd --spec <table.json> --out x.png` - a
  table description's things side by side under candlelight, no tarot table around them.
- `GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/tarot_look_probe.gd 400 --show S --seed N --marks 1`
  - the table over an episode with a synthetic voice.
- `GHOST_PROBE_GPU=1 GHOST_PROBE_MUTE=1 tests/run_boot_probe.sh tests/tarot_voice_probe.gd 900 --spec <md> --seed N --export 1`
  - the panel, the real voice and the table, end to end, and the export take.

## Not built yet

- A thumbnail.
- Pick-a-pile episodes (three piles, "all four piles say the same thing").
- Moving `Illustrations`' own job pump onto `AgentJobs`.
- Porting the tablet's follower onto `ReadingFollower`.
