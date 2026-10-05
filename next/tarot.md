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
   words, Codex for pictures). **Folder** opens the episode: every prompt beside its reply.
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
| `image:object:K` | painter | the look's object K - a photograph of it alone, cut out of its background (`TarotCutout`) |
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

Static camera from the reader's chair (the Camera dial adds a breath), cloth from the episode's
`surface`, its `backdrop` out of focus beyond the far edge. ON THE TABLE (the user, 2026-10-04:
the 3D props were "half-baked... the novelty would wear off"): the look's candles, still 3D -
their flames light the cards and throw flickering shadows (a candle's own body is left out of its
flame's shadow, which printed a hard disc round its base) - and its OBJECTS, painted: each a
photograph of one thing from the reader's angle, cut out of its background (`TarotCutout`: the
border's color flood-filled away, so a white thing's white inside survives), stood square to the
camera, which never moves, with a shadow laid on the cloth as decals. Where each stands is FOUND
(`_find_spot`): in the shot, clear of everywhere the cards go, not in front of another, the
tallest toward the back. The lamp is low and to one side, so things throw shadows you can see.

THE SHUFFLE IS RUNS WITH SPARSE ACTIVITY (the user: "nonlinear behaviors with sparse
activations"): several riffles, a string of cuts, a few overhand passes - one kind at one tempo -
then nothing for a while (median ~3 s, now and then a long linger), in the MIDDLE of the table;
before the first card the deck is squared and pushed to its side (`TarotScript.PUSH`, which the
first draw's rest includes). A wash is rare, once at most, and a planned simulation: the deck
spread, two flat hands dragging what they touch, then three or four sweeps round the pile, each
taking its share - some cards only pushed near, some missed and fetched by a later sweep - then
squared. A draw: square, slide, flip, up to the camera on the LEFT beside the booklet page on the
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
