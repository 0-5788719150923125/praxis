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
| `scripts/tarot_table.gd` | `TarotTable`: what a look may name - title faces (`fonts/tarot/`, OFL), frames, props - and `sanitize_look`. |
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

## The table

Static camera from the reader's chair (locked; the Tarot panel has no Camera dial), cloth from
the episode's `surface`, its `backdrop` out of focus beyond the far edge. ON THE TABLE (the user, 2026-10-04:
the 3D props were "half-baked... the novelty would wear off"): the look's candles, and nothing
else. Their flames light the cards and throw flickering shadows; each candle casts in the other
flames' light but is left out of its own (which printed a hard disc round its base), and a soft
contact shade sits where it meets the cloth. Where each stands is FOUND (`_find_spot`): wholly in
the shot (its whole projected box), clear of everywhere the cards go, not in front of another,
only behind the middle of the table (one that ran out of room stood in FRONT of the cards, became
the key light and blew out the card held up to the lens), toward an aim flanking the cloth
rather than its rim, and by dark cloth rather than pale. The lamp is low and to one side, so
things throw shadows you can see.

PAINTED OBJECTS WERE TRIED AND TAKEN OFF (2026-10-04). Each was a photograph of one thing,
painted alone and cut out of its background, stood square to the camera with decal shadows. After
fixing what could be fixed (a tint that made them see-through, the painting angle, sizes, the rim),
the user: "these 2D objects look awful. Their orientation is off, they have no lighting, their
style doesn't fit the scene, the relative scaling between them is way off, they cast no
shadows". A painting brings its own camera, light and grade, and a flat card can neither take the
candlelight nor cast a shadow - so the fault is the approach, not its tuning. The one route that
fixes it structurally is to turn each painting into a mesh with a local image-to-3D model
(TRELLIS / Hunyuan3D class: real light, real shadows, any object the planner invents) at the cost
of a heavy optional install, an NVIDIA GPU and blobby thin parts; offered, and declined for now.

FLAMES FLICKER APART (the user: "the candles flicker at exactly the same rate, which is wrong").
Every flame had one tempo and one swing, so they pulsed together. Each now has its own tempo,
steadiness and drafts: mostly a steady burn, and now and then a draft gutters it for a second or
two - a pure function of show time, so a render and a scrub see the same flame. The light leans
with its flame, so the shadows breathe. ROOM CANDLES OUT OF SHOT (the user's idea, "backlighting
behind the camera, which also flickers"): two or three shadowless, flickering lights behind the
camera and off to the sides - a warm fill on the cloth and on the card held up to the lens.
Gate: `tests/tarot_place_check.gd`.

ONE LIGHT THROWS THE SHADOWS (the user: "the stack of cards casts no shadow... two independent
shadows for each side"): the candle nearest the middle is the KEY - bright, reaching the cards,
the only light with shadows, soft at their ends (`light_size`), biased for a tabletop (at the
defaults a deck's shadow began centimeters in front of it). The other candles light without
shadows; the lamp is a shadowless fill (the key when there is no candle). NO CANDLE IS BRIGHTER
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
spread WIDE (some cards flung well out), two flat hands dragging what they touch, then six to
eight sweeps round the pile over ~6 s, each taking a few cards - some only pushed near, a few
missed and fetched by a later sweep - landing nearly squared, so the end is a light tidy, never
the whole spread arriving at once. A card lies on the highest card it truly OVERLAPS (their turned
rectangles, `_cards_overlap`), just above the cloth - a distance test let overlapping cards share
a height and cut through each other. A draw: square, slide, flip, up to the camera on the LEFT beside the booklet page on the
RIGHT (shown, never read); both turn a little on their axes, and the card is now and then turned
to look at its back. A lay: page out, card down into the spread. Jumpers fly out of the shuffle.
The channel's name and the episode's title open it; the channel's name closes it.

**Foil** (the user's idea, 2026-10-04): each painting's brightest, most colorful pixels - keyed
per picture from its own luminance percentiles, plus anything near the frame's accent color -
print as foil: metallic, a slow breath, a glint sweeping across now and then; the scene's bloom
lets it bleed. The look's `foil` (0-1) says how much.

Everything is a function of show time (`ReadingFollower` + the schedule), so live and export
draw the same frames.

## Probes and gates

- `godot --headless --path . --script res://tests/tarot_check.gd` - the gate.
- `tests/tarot_episode_probe.gd` - make an episode headlessly (real quota).
- `GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/tarot_look_probe.gd 400 --show S --seed N --marks 1`
  - the table over an episode with a synthetic voice.
- `GHOST_PROBE_GPU=1 GHOST_PROBE_MUTE=1 tests/run_boot_probe.sh tests/tarot_voice_probe.gd 900 --spec <md> --seed N --export 1`
  - the panel, the real voice and the table, end to end, and the export take.

## Not built yet

- A thumbnail.
- Pick-a-pile episodes (three piles, "all four piles say the same thing").
- Moving `Illustrations`' own job pump onto `AgentJobs`.
- Porting the tablet's follower onto `ReadingFollower`.
