# Ghost Notes: every project is a note

A proposal, opened 2026-10-06. Nothing here is built. It starts a conversation the user began with
two worries that turn out to have one answer: the six modes duplicate each other, and the product is
called Ghost Notes but takes no notes.

## The idea

One thing on the home screen instead of six: a **note**. A new note is a plain text editor and
nothing else. Everything ghost does today becomes a **component** attached to a note. Write a
script and attach a voice, and the note speaks. Attach the tarot table, and the voice reads at a
table. Attach scenes, and the show answers the voice. In the user's words: every project within this
app is, at its core, a note.

The left panel stops being a bespoke layout per mode - hard to read, because it shows everything a
mode can do at once - and becomes a stack of **cards**, one per attached component, each bordered in
its component's color (the generative voice in turquoise, say), so it is always plain which settings
belong to what. Cards collapse, and a row of buttons on the note, one per component, opens and
closes them.

And nothing is wired. Attaching a component is declaring it, which is how the rest of ghost already
works: a scene declares what it can morph from, a medium declares the settings it uses, and the show
is a function of what was declared. That is a selling point, and
[Why nothing is wired](#why-nothing-is-wired) is why it holds.

How components blend is the open question. What follows: what the code already has, where the
duplication actually is, a model, a proposal for the blending, prior art, risks, and a path that
ships in pieces.

## The note is already half there

**Documents already carry their components, keyed by the wrong thing.** Generative, Synthesis and
Tarot sync to a plain markdown file the user owns and keep their settings under its `ghost:`
frontmatter key (`FrontMatter`), one block per PANEL. So the same components are written in two
shapes:

|             | a Generative chapter                                                  | a Tarot show                                             |
| ----------- | --------------------------------------------------------------------- | -------------------------------------------------------- |
| the block   | `ghost: generative:`                                                  | `ghost: tarot:`                                          |
| the voices  | `turn, tab, voices{...}, hesitate`                                    | the same                                                 |
| the picture | `medium, filters, scene_hold, flourishes, camera, hand, intro, outro` | `filters, intro, outro`                                  |
| its own     | `illustrations`                                                       | `show, seed, cards, reversals, jumpers, writer, painter` |

The voices and the picture are the same components in both, and `TarotEditor._doc_capture` takes the
Generative block and erases six keys from it. Open a chapter in Tarot and its voices are not there.
In a note, `voice:`, `picture:` and `tarot:` are siblings, and a block being present is what
"attached" means.

**Ghost already keeps notes - one per mode.** An unsynced draft is saved in ghost.cfg as
`[generative] text`, `[tarot] text` and `[synth] text`: one unsaved note per mode, and nowhere to
put a second.

**Settings with no home end up in another mode's panel.** Auto has no panel. Its picture settings
(Look, medium, scene hold, flourishes, bookends) drive the Director in every mode, and the only place
to change them is the Generative panel - on purpose: "one place to reach for a setting beats an
architecturally tidier second home nobody finds" (`GenerativeEditor._build_picture`). A card is that
second home, and it is found because it appears wherever a picture is made. The same settings are
also stored twice, globally in `[director]` and per chapter in `picture:`.

**Presentation was pulled out of the modes once, and it worked.** media.md: "Not a mode. Modes decide
what _drives_ the show (a song, a storyboard, a voice). A medium decides what the show is _presented
as_, and every mode gets both." That was the first mode taken apart into pieces.

**The panel already follows declarations rather than branches.** `Medium.USES` lists the setting
groups each medium shows, written after "a number of settings currently being displayed that ONLY
work with the comic book medium", and its comment states the rule this note generalizes: "adding a
presentation is an entry in these tables, never new control flow somewhere else." `Filters.REGISTRY`
builds its own rows, and every one of the 29 entries in `ScriptMarks.REGISTRY` carries a `modes`
filter, which becomes a component filter.

**Moments exist twice already.** `TarotTable.MOMENTS` (shuffle, jumper, reveal, pirouette, lay,
close) are what the air effects fire on. In Synthesis "the fishing owns the MOMENTS"
(`Director._should_change`): a catch or a seed jump cuts the scene. Both are one part telling another
that something happened - the second through a flag the Director checks for one mode.

**Sound already meets in one place.** `Spectrum` is "the one place that knows audio exists": it
analyzes the Master bus, so a song, a voice take and a clip's soundtrack reach the scenes the same
way, and it owns the show clock (live, baked, or counted from frames in a render). The Generative
editor says it outright: "Everything DOWNSTREAM is shared, because a take is just a WAV." Pictures
meet in one place too: the Director owns the picture state and the medium.

**A blackboard already runs.** `TarotEpisode` is one: each step posts a file, a step runs when what it
needs exists, and a redo invalidates what was made from it. That is the shape agents should keep
when they work for any component.

**The card is an engine node.** Godot 4.7 ships `FoldableContainer` - a titled, collapsible container;
`add_title_bar_control()` puts buttons in its title bar, and its `panel` / `title_panel` styleboxes
carry the border - and `FoldableGroup`, which keeps one open at a time (`allow_folding_all` lets all
close). Checked against 4.7.2's ClassDB. Ghost uses neither yet.

## Where the duplication is

**The duplication is in the shells, not the engine.** Director, Spectrum, the scenes, the Exporter,
Chrome and the registries are shared. What is copied is everything around them: the panels, the mode
wiring in main.gd, and persistence.

- **Panels are built four ways.** SidePanel (Synthesis, Generative, and Tarot by inheritance),
  Manual's `Workspace` (its own PanelContainer), Masking's own full-height panel, and Auto with none.
- **Synthesis and Generative mirror each other by copy.** The header, the ScriptWriter wiring, F2 to
  hide, the status line, and the autosave with its persist-on-exit are copy-paste. Generative's
  `_write_wav` repeats `Voice.write_wav`. Document capture/apply and the subtitle sidecar were written
  in parallel and have drifted.
- **Tarot inherits Generative and spends its effort removing things.** About 25 overrides.
  `_doc_capture` erases six keys the base wrote, `_build_cast` hides the Hesitate row after the base
  builds it, Intro and Outro are declared a second time, and the base null-checks rows Tarot never
  builds. One row nobody removed: Ink, the pen a voice writes in on the Notebook, shows on the tarot
  table, and Truthful Tarot's Familiar carries `ink: red`, which nothing there reads.
- **main.gd starts a session three ways.** Three start/stop pairs (each one detaches, calls
  `Spectrum.begin*` and `Director.attach`, and sets up subtitles), three subtitle-attach paths, and
  the Exporter wired by copy with different values. Space toggles play in three places.
- **Masking predates Chrome and rebuilt it.** It has its own feedback console and its own Assistant:
  launched from the home screen, that is a second Assistant beside Chrome's, and Assistant has no
  single-instance guard. It has its own export, which never received the Exporter's virtual-display
  render or its streaming transcode. Its track import dialog was copied into Generative for films.
- **Every control is listed three or four times within its own mode.** Generative's voice:
  `SLOT_DEFAULTS`, `_capture_slot`, `_apply_slot`, the builder and `_apply_fx`. Masking's 54 options are
  built, re-synced and shown or hidden in three separate places (a comment there records the bug that
  caused). `Settings.bind` exists to remove the persistence half of this, and no mode panel uses it.
- **Three agent pickers** (Tarot, IllustrationPanel, the home screen): two check availability, one
  lists models, one binds its setting.

A component is the unit all of these collapse into: one card builder, one capture/apply, one
frontmatter block, one place it is torn down.

## The model

### The note

A markdown file, as now: frontmatter, then a body. Notes live where the user already keeps them -
rift's chapters and shows are notes today - and Ghost Notes remembers the recent ones. With nothing
attached, it is a text editor. The file stays plain markdown that any editor can open: marks are HTML
comments and settings are frontmatter, and nothing a component does may change that.

Every note has one card that cannot be removed: its own. Title, author and book (ScriptWriter's
fields today), the seed, and the bookends once anything plays. Unity's Transform is the precedent,
the one component every object has. It takes no component color; it is the paper.

### A component

An entry in one registry shaped like `Medium` (key, label, blurb, `make()`, a base class with small
hooks), which is the closest thing ghost already has. An entry declares:

| field       | Voice, for example                                                                        |
| ----------- | ----------------------------------------------------------------------------------------- |
| key         | `voice` - its block under `ghost:`                                                        |
| label, icon | Voice, a waveform                                                                         |
| family      | `voice` (below); the family gives the color                                               |
| needs       | the text                                                                                  |
| provides    | audio, a reading position, moments (sentence, speaker change)                             |
| requires    | components it attaches with it (Book brings Illustrations), as Unity's `RequireComponent` |
| marks       | its groups of the script palette (`voices`, `pronunciation`)                              |
| card        | its panel builder                                                                         |
| tools       | MCP tools an agent may call while it works (optional)                                     |

Needs and provides are typed names that ghost matches, the way a scene's `morph_out` and `morph_in`
already are.

**A family says how many.** Some components are alternatives: one presentation per note (Comic,
Book, Notebook, Tablet, the tarot table; full frame when none is attached). Some stack, like Look
filters. Some are one card with plurality inside: one Voice, with a tab per speaker. The family
declares its count, and nothing checks pairs. `Medium.OWNED` becomes a component that brings its own
presentation: attaching Tarot attaches the table.

**Speakers belong to the note, not to the voice.** The cast is read from the text's speaker cues,
and other components want per-speaker settings. The Notebook's ink is one; it lives in the voice's
slot today only because that is where the speakers were. Make the speakers the note's own and let
each component add its fields to them: Voice adds the voice, Notebook adds the ink. Tana and Logseq
solved the same collision the same way - a field has one identity, whichever tag uses it.

**In Godot terms.** A component is a node under the note's node, and Godot's own advice fits:
"design scenes to have no dependencies". The meeting points below are services the note hands its
children - Spectrum and the Director already are autoloads - and groups answer "which attached
components provide a reading" without an entity-component framework.

### A card

One `FoldableContainer` per attached component. The border is its family's color; the title bar holds
its icon, name, the toggles, and a small menu: defaults, "Same as..." (take this card's settings from
another note - the Narrator from Truthful Tarot, say; Xerox Star's "Same" command), detach, help. A
few essential controls show, and the rest sit behind "More" - two levels, never three. A value that
differs from its default is marked and has a reset, as in Unreal's Details panel, because "what did
I change" is half of why the panels are hard to read now. A need nobody provides is a sentence on the
card ("the Notebook follows a reading - attach a voice"), never a control that silently does nothing.

**The color follows the component everywhere:** the card's border, its chip on the note, its marks
highlighted in the text (speaker cues in turquoise), its moments on a timeline. The note and the
panel then explain each other.

**Color is per family, not per component.** That is how every system that uses color this way does
it: Scratch and Blockly categories, Blender's socket colors, TouchDesigner's operator families. Hues
run out near eight, and since most families allow one member per note, within a note it still reads
as one color per component. The two voice engines share turquoise; the presentations share another.
One saturation and lightness for every hue, as Blockly fixes them, checked in both themes, and never
color alone: always the icon and the name.

**Order is derived, never dragged.** Text, then producers (Tarot), voices, sound, pictures, the
presentation, and the Look last. Read top to bottom, the panel says what happens to the note. Blender
lets the user reorder modifiers because there order changes the result; Figma fixes its effect order
by kind instead, and Observable runs cells by what each needs. Here order is a fact of the families,
so the panel derives it.

### The component row

On the note, under its title: one chip per attached component in its color, then "+". A click opens
or closes that card; Ctrl-click opens it alone and folds the rest, as on Blender's panel headers.
"+" lists what can be attached, grayed with a reason when something cannot be (no agent installed -
the home screen's gating, moved here).

**"+" suggests what the text already asks for.** Speaker cues suggest a Voice, `<!-- url: -->` marks
the Tablet, a `## Cards` section Tarot, image marks Illustrations. Ghost infers this already
(`TabletScript.is_tablet`), and tablet.md's rule is "what is inferred, never written". Suggested,
not attached: attaching can start a model download or an agent.

**Three booleans, kept apart.** Blender saves all three per modifier.

- **Open** - a view state. Folding never changes the output. The viewer's, kept in ghost.cfg, so the
  user's files do not churn with it.
- **On** - evaluated or not. Off keeps every setting, so switching back restores it (Notion and
  Anytype keep a converted block's fields dormant the same way). The note's, in its frontmatter.
- **Ghosted** - evaluated but not heard or seen: Blender's Outliner splits "hide" (still evaluated)
  from "disable" (not evaluated). This one is exactly a **ghost note**, a real musical term: a note
  with rhythm and no pitch. A ghosted voice still sets the show's time and its moments and is not
  heard; a ghosted song still drives the scenes, for a picture-only export cut to its timing.
  Notation writes a ghost note in parentheses, and the chip can too. Unity's rule decides where it
  appears: a toggle shows only where it changes something, so the Look has no ghost state.

```
 panel                                note
┌─────────────────────────────────┐   ┌─────────────────────────────────────────┐
│ What is the 7th Realm?          │   │ What is the 7th Realm?                  │
│ Pen · North Star                │   │ [● Voice] [● Tablet] [○ Look] [+]       │
│                                 │   │                                         │
│ ┏ Voice ━━━━━━━━━━━━━━━━━ ● ▾ ┓ │   │ <!-- url: www.duckduckduck.mom -->      │
│ ┃ Narrator │ Familiar │ +     ┃ │   │ ...                                     │
│ ┃ Model  libritts-high   ▶   ┃ │   │                                         │
│ ┃ Pace   ─────●─────          ┃ │   │                                         │
│ ┃ More ▸                      ┃ │   │                                         │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ │   │                                         │
│ ┏ Tablet ━━━━━━━━━━━━━━━━ ● ▸ ┓ │   │                                         │
│ ┏ Look ━━━━━━━━━━━━━━━━━━ ○ ▸ ┓ │   │                                         │
└─────────────────────────────────┘   └─────────────────────────────────────────┘
```

## How components meet

The idea leaves this open, and this is my answer: **components never refer to each other.** If Tarot
has to know that scenes exist, every new component pays for every old one - n squared pairs - and
that is the trap the modes are already in (the Director's `_game_paced` is the fishing game reaching
into the scene cutter). Components meet only through things the note owns. There are five.

1. **One clock.** Every output is a pure function of show time, which films, the tablet and the
   export already obey. One component conducts - its length is the show's - and it is chosen by a
   fixed order, never asked: a voice, else a clip, else a song. That is always audio, which is also
   what ffplay and Godot's own sync guide make the master clock. With none of them, the note is not
   timed and shows no transport.
2. **Signals.** Continuous and typed: the audio (one mix; a song ducks under a voice), the spectrum
   and harmonic signature (`Spectrum`, unchanged), the reading position (`ReadingFollower`). A
   component reads what it needs and never asks who made it. Values are clamped and NaN-guarded once,
   at the source, as VCV Rack does for its one signal (a NaN has spread through ghost before).
3. **Moments.** Discrete events on one schedule: a sentence, a speaker change, a mark reached, a card
   laid, a catch, an onset, a cut. Computed ahead, the way the tablet already computes its actions
   into the voice's rests, so a moment falls on the same frame live and in an export, and a scrub
   lands in the middle of a schedule. "The scene cuts when a card is laid" is then a Scenes setting
   that names a moment - not code in Tarot, and not a flag in the Director.
4. **Places.** A presentation offers places a picture can go: comic panels, the tablet's screen, the
   notebook's clipped photos, the book's illustrations - and, why not, the faces of the tarot cards
   or a window behind the reader. Picture components fill places. The comic already does exactly
   this with scenes in panels, and it is the most direct route to "scenes + tarot" meaning
   something.
5. **Seeds.** One seed per note. Each component draws from it XORed with its own key, the fold
   `Director._pick_salt` already uses. Attaching or switching off one component then never changes
   another's draws, which is what makes switching one off a fair A/B.

Signals are the continuous half and moments the discrete half - the same split harmonic_seeding.md
makes for a song's signature, one level up. TouchDesigner and Max keep streams and events apart for
the same reason.

**Modulation is the creative half.** The five points are plumbing: they make a combination work.
What makes it interesting is modulation as Bitwig does it - any numeric control on any card can be
offset by a signal or a moment, within a range drawn on the control itself. The Look's static rising
with the voice's loudness, the scene hold shortening while the Familiar speaks, the camera easing in
on `reveal`. Ableton's macros are the other half: a few dials on the note's own card, each fanning
out to many controls with ranges. Both are data on the card that consumes them, so neither breaks the
rule above. Few dials, small steps, eased per sentence - never a switch.

**One place where order matters: text walks.** Components that own marks rewrite what the voice
reads. The tablet strips its marks and leaves rests (`TabletScript.speakable`), and Tarot swaps the
body (a brief, never read aloud) for the episode's script. Today that is an `if` in
`GenerativeEditor._reading_of`. In the model it is a chain of walks ordered by family, and it is the
only chain.

**The text as a signal, optionally.** With no voice, the words could still steer the look. SimHash,
the hash harmonic seeding runs on chroma, is best known on text - Google's near-duplicate detection
(Manku et al., 2007). The same note would give the same show, and a lightly edited note a lightly
different one. As a `seed_bias`, the way harmonic seeding chose, not as a replacement.

### Three combinations, walked through

- **Voice + Tarot + Scenes.** Tarot needs the text (as its brief) and agents, and provides an
  episode script (a text walk), the table (a presentation) and moments. The voice reads the script.
  Scenes need a place; the table offers the card faces, and Scenes cuts on `lay`. Nothing in Tarot
  names Scenes.
- **Clip + Voice**, a narrated video. The clip provides a picture and audio, and it conducts, so a
  reading longer than the clip is a gap the card names ("the reading runs 40 s past the clip").
  Masking's effects ride on the clip, or open its workspace.
- **Song + Notebook.** The notebook follows a reading, and a song provides none. The card says so and
  offers two fixes: attach a voice, or a lyrics-alignment component that would provide a reading
  from a song. The types are what find the missing piece.

## Why nothing is wired

Scratch and Blockly have the user snap blocks together; Max, VCV Rack and TouchDesigner have wires.
In those tools the parts are running processes that must be told where to send their output, so the
wire is the program. Ghost needs no wire for the same reason the rest of ghost needs none: **ghost
is declarative.** A part says what it is and what it reads, and the engine derives the rest.

- A scene declares its render kind, and the geometry it leaves and can grow from (`morph_out`,
  `morph_in`). When the types match the Director morphs, handing over a typed payload; when they do
  not it cuts, "so a bespoke transition can never break". Typed ports, matched by the engine, have
  shipped since June.
- A medium declares the settings it uses: "WHAT A MEDIUM USES IS DECLARED, NOT BRANCHED ON".
- A storyboard is data, every number a sampleable range, and the scene-spec north star is "a
  declarative spec that _samples a configuration_ ... rather than from hand-written code".
- The tarot table is "planned, not engine physics: the table is a pure function of show time", and
  so are the tablet's screen, a film's frame (`Films.position_at`) and the book's opening.
- Agents write data, not code: the set dresser describes things in the vocabularies `Props` and
  `Effects` publish, and ghost sanitizes and builds them.
- The show as a whole "is a pure function: the seed is `hash(fingerprint(audio):SEED_SALT)`", which
  is why a six-hour render's schedule replays in about a minute.

In a declarative system the declarations are the program, and the graph is implicit in what each
part reads. A spreadsheet works this way - it recalculates in the right order, and nobody draws
wires between cells - and so does Observable. Attaching a component is one more declaration, and
there is nothing left to connect. Where a real choice remains (which place a picture fills, which
moment a cut follows), it is a picker on the card of the component that consumes it: still a
declaration, never a cable.

**It explains the duplication, too.** The duplication sits exactly where ghost is NOT declarative.
The engine is registries and functions of show time, and it is shared. The shells are hand-built
panels and imperative session wiring in main.gd, and they are copied. The note model is the
declarative style finishing the job: the component registry does for modes what `Medium.USES` did
for the panel.

**Honestly graded.** Not every part is a pure function. Cloth, boids and murmurations integrate
state frame by frame, and the harmonic signature is an EMA: deterministic from the seed under the
render clock, close but not bit-exact live (harmonic_seeding.md's caveat). Agents are not functions
at all - the same prompt gives a different reply - and ghost already handles that the right way: an
agent's output is written to a file when it lands, and from then on the show is a function of that
file ("a redo is a delete"). The claim that holds everywhere is the one that matters: **the show is
determined by what is declared.**

**That is a selling point**, and it reads as one:

- **Nothing to wire.** Write, attach, play. No patching, no node graph.
- **The same note makes the same video.** Seeds and agents' outputs are kept, so an episode plays
  back as it was made, and an export can be replayed.
- **The note is the project.** One plain markdown file to read, diff, version and share (the
  generated pictures aside - see the open questions).
- **Agents write data, not code.** An agent describes and ghost builds: a bad answer is sanitized,
  never executed, and what an agent made is kept.

## Today's modes, decomposed

| mode       | as a note                                                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Auto       | Song + Scenes (seeded) + Look                                                                                                 |
| Manual     | Song + Scenes (storyboards, dials) + Look                                                                                     |
| Synthesis  | Voice (synthesis engine) + Voice lab (genome, belt, fishing) + Scenes + Look                                                  |
| Generative | Voice (neural engine) + one presentation, or Scenes full frame + Illustrations or Films, as the presentation uses them + Look |
| Tarot      | Tarot (brings the table) + Voice + Look                                                                                       |
| Masking    | Clip + Masks (markers, tracks, effects, its timeline)                                                                         |

Auto and Manual differ by one setting of one component; Synthesis and Generative by the voice engine
and the lab. Six modes are about a dozen components, and most modes share most of theirs.

**The modes survive as templates.** "Tarot ▶" on the home screen makes a new note with Tarot and a
Voice attached. The on-ramp stays; the six code paths behind it go.

## Prior art

Researched 2026-10-06. Grouped by the decision each one informs.

**The three the user named.**

- **Scratch** is the closest match to the panel. Its block palette is one long column of colored
  categories, with a column of colored circles beside it that jumps to each, and **Add Extension**
  (Text to Speech, Music, Video Sensing, Pen, Translate) adds a new category at the bottom: attach a
  component, get a section. Two details worth copying. Extensions do NOT get their own colors - all
  share one green (#0FBD8C) and are told apart by an icon on the category and on every block. And an
  extension whose blocks are unused is dropped when the project reloads. Sprites couple only through
  named broadcasts that fire "when I receive" hats - moments, in Scratch's words.
  [colors](https://github.com/scratchfoundation/scratch-gui/blob/develop/src/lib/themes/default/index.js),
  [extensions](https://github.com/scratchfoundation/scratch-vm/blob/develop/docs/extensions.md),
  [broadcasts](https://en.scratch-wiki.info/wiki/Broadcast)
- **Blockly**, the engine under Scratch's blocks, answers how to type the connections: each
  connection carries a `check` list of type names, two connect only when their lists share one, and
  an empty list accepts anything. That is needs and provides. Its colors are a hue at one fixed
  saturation and value (0.45, 0.65), so any hue fits the set, and blocks are declared in JSON - a
  registry. [checks](https://docs.blockly.com/guides/create-custom-blocks/inputs/connection-checks/),
  [colors](https://docs.blockly.com/guides/configure/toolboxes/appearance/)
- **Logseq** is the closest match to the text. A plugin's slash command writes
  `{{renderer :name, args}}` into a block and the plugin draws its UI there, and the official
  pomodoro sample writes its state back into the macro's arguments: invocation and state live in the
  text, as ghost's marks do. Its newer database version lets a block carry several tags, each with its
  own properties, with one name per property across the whole graph - Tana's supertags, in substance.
  The price was the plain markdown files (an export drops block properties). Keep the files.
  [macros](https://github.com/logseq/docs/blob/master/pages/Macros.md),
  [plugin sample](https://github.com/logseq/logseq-plugin-samples/blob/master/logseq-pomodoro-timer/index.ts),
  [database version](https://github.com/logseq/docs/blob/master/db-version.md)

**Compound documents, and why OpenDoc died.** The container idea is not what failed: few parts ever
shipped, nobody owned the experience across parts, and OLE already held the market. OLE's own flaw is
worth avoiding too: only one embedded part is active at a time, so its controls appear only when you
click into it - a panel of cards shows every attached component at once instead. Ink & Switch's
Patchwork essay (2025) restates the problem OpenDoc never solved: does the person you share a document
with have every editor it needs? [OpenDoc](https://en.wikipedia.org/wiki/OpenDoc),
[OLE](https://learn.microsoft.com/en-us/cpp/mfc/active-document-containment),
[Malleable software](https://www.inkandswitch.com/essay/malleable-software/)

**Text as the source of truth.** Potluck keeps all state in the text ("no hidden metadata") and found
its limit there: text "rules out applications with non-textual data or rich visualizations", which is
why ghost needs components at all. Embark makes views render any subtree of one outline - views over
the note, not owners of separate data. Org-mode's settings cascade (system, language, file, heading,
block, call; the most local wins) is the shape for component default, note, and a mark in a passage.
[Potluck](https://www.inkandswitch.com/potluck/), [Embark](https://www.inkandswitch.com/embark/),
[Org](https://orgmode.org/manual/Using-Header-Arguments.html)

**Inspectors.** Unity's component header is fold arrow, icon, enable checkbox, name and menu; the
checkbox appears only on components where disabling would do something; `RequireComponent` adds
dependencies; it has no built-in header colors (those come from custom editors). Blender evaluates
modifiers top to bottom, saves expanded, viewport and render as properties of each, and Ctrl-click on
a collapsed panel header opens it and closes the rest. Unreal's Details panel marks every value that
differs from its default, with a reset. Xerox Star gave every object the same property sheet and the
same few commands, "Same" among them. [Unity](https://docs.unity3d.com/Manual/UsingComponents.html),
[Blender](https://docs.blender.org/manual/en/latest/modeling/modifiers/introduction.html),
[Outliner](https://docs.blender.org/manual/en/latest/editors/outliner/interface.html),
[Unreal](https://dev.epicgames.com/documentation/en-us/unreal-engine/level-editor-details-panel-in-unreal-engine),
[Star](https://digibarn.com/friends/curbow/star/retrospect/)

**Composition.**

- Bitwig's modulators: drag from a modulator onto any parameter, the range relative and drawn on the
  control. Ableton's racks: up to 16 macros, each mapped to any number of parameters with ranges.
  [Bitwig](https://www.bitwig.com/userguide/latest/the_unified_modulation_system/),
  [Ableton](https://www.ableton.com/en/live-manual/12/instrument-drum-and-effect-racks/)
- TouchDesigner's typed families ("only operators of the same family (color) can be wired
  together", with explicit exports between them) and Max's hot/cold inlets keep streams and events
  apart. [TouchDesigner](https://docs.derivative.ca/Operator_Family),
  [Max](https://docs.cycling74.com/legacy/max8/tutorials/basicchapter06)
- VCV Rack writes down the ranges of its one universal signal and turns NaN into 0.
  [VCV](https://vcvrack.com/manual/VoltageStandards)
- Observable runs cells in dependency order. Jupyter is the cost of not doing so: of 863,878 attempted
  notebook runs, 24% ran without errors and 4% reproduced their results, hidden state and
  out-of-order cells among the causes. [Pimentel et al.](https://leomurta.github.io/papers/pimentel2019a.pdf)
- Entity-component systems keep behavior in systems keyed on which components are present; Bevy's
  `require` adds dependencies. Godot composes with child nodes "at a higher level than in a traditional
  ECS", so ghost needs no framework for it. [ECS FAQ](https://github.com/SanderMertens/ecs-faq),
  [Godot](https://godotengine.org/article/why-isnt-godot-ecs-based-game-engine/)
- Blackboard systems (Hearsay-II): sources post facts, and a controller picks what runs next.
  [Nii 1986](https://ojs.aaai.org/index.php/aimagazine/article/view/537)
- Render volumes (Unity URP, Unreal) blend two writers of one field by priority and weight, with a
  per-field override - the rule if two components ever must set the same field.
  [URP](https://docs.unity3d.com/Packages/com.unity.render-pipelines.universal@7.5/manual/Volumes.html)

**Writing on it.** Kay and Goldberg named the risk in 1977: a general medium may "collapse under the
weight of trying to be too many different tools for too many people", a "feature-laden hodgepodge".
Victor's Magic Ink: software should infer the context its data is needed in - which is what "+"
suggestions are. Matuschak and Nielsen: the component system is not the insight; each component still
needs real depth. Nielsen Norman Group: designs past two levels of disclosure "typically have low
usability". [Kay and Goldberg](https://tinlizzie.org/VPRIPapers/m1977001_dynamedia.pdf),
[Magic Ink](http://worrydream.com/MagicInk/), [Tools for thought](https://numinous.productions/ttft/),
[NN/g](https://www.nngroup.com/articles/progressive-disclosure/)

## Risks

- **Leaving a note.** A note app switches documents, and ghost cannot leave most of its modes: only
  Auto returns to the home screen, Manual loops, and the voice modes and Masking have no teardown at
  all (`_synth_editor` and `_mask_editor` are never freed). Every component needs a real detach, and
  a gate that attaches and detaches each one many times and finds no leaked nodes, Chrome claims or
  bus effects, before notes can switch.
- **Pairs.** Telecom calls it the feature interaction problem: features that each work and misbehave
  together. With n components there are 2^n combinations, and most will never run. Templates are the
  tested paths; one contract check per signal (everything that provides a reading passes the same
  test) covers the rest.
- **The hodgepodge.** Kay and Goldberg's warning. A dozen deep components beat thirty shallow ones,
  and a card that needs a third level of disclosure is two cards.
- **Cost.** AUDIT.md: a frame is tens of thousands of GDScript calls on one core of a 2014 CPU. Live
  scenes on four card faces are four scenes. A component should declare a rough cost, and "+" should
  say when the total is past what runs.
- **Workspaces are not cards.** Masking (8,200 lines, its own timeline) and Manual's workspace are
  editors, not settings. Their cards hold the essentials and a button that hands them the stage, as
  ScriptWriter's pop-up editor does now. Masking moves last, not first.
- **Sharing and versions.** Today every note is the user's own, so a one-time rewrite is enough. Once
  other people have notes, each block needs a version and a migration (Ink & Switch's Cambria shows
  translating between versions always gives something up), and a note opened without a component it
  uses must keep that block verbatim, behind a gray card that names what is missing.
- **The blank page.** An empty note invites less than "Tarot ▶". Templates again.
- **The body changes its role.** With Tarot attached the body is a brief, and the voice reads
  something else. See child notes, below.

## A path that ships in pieces

Each step stands alone, and the early ones pay for themselves even if the later ones never happen.

1. **Cards.** Wrap each mode's existing sections in colored FoldableContainers. No change in
   behavior; the panels get readable now, and the colors and the row get tried in real use.
2. **Shared cards.** One card per duplicated section (script, cast, voice, picture, Look, bookends),
   used by every mode that has it. Tarot stops inheriting Generative. One frontmatter shape per
   component, with a one-time rewrite of rift's documents rather than a reader that accepts both
   shapes.
3. **Teardown.** Every mode can be left and entered again, behind the gate above.
4. **The registry.** Components declare key, family, needs and provides, marks and card. Modes become
   templates, and main.gd's three session paths become one.
5. **The note.** Open any markdown file with no mode; "+" attaches; the note owns the clock, signals,
   moments, places and seed.
6. **The home screen.** Recent notes, a new note, and the templates.

## Open questions

- **Editor or stage?** My lean: one window, the editor while you write and the stage while it plays.
  A note with nothing visual attached never shows a stage.
- **Child notes.** Should each tarot episode be a note - its script as the body, the show's
  components inherited and overridable? Then "every project is a note" holds for generated text too,
  and the voice always reads the note that is open. Webstrates goes further, embedding one document
  in another: a deck note shared by several shows.
- **Storyboards.** Manual's storyboards are data, with no prose and no editor. Do they become note
  text (a fenced block), or stay the Scenes card's own files?
- **Where generated things live.** Tarot's `show` key already works as a note id for
  `user://tarot/<show>/`; generalize it to `user://notes/<id>/<component>/`. And a reference should be
  a content-hashed name, not an absolute path: a chapter today names
  `/home/crow/.local/share/godot/app_userdata/ghost/illustrations/refs/49b1c1e7ac9a6fb9.png`, which no
  other machine has.
- **The families, and their colors.** Roughly: text and producers, voice, sound, pictures,
  presentation, Look, video. Which hue each gets, and whether Tarot (a producer that brings a
  presentation) takes the producer's color or the table's.
- **Is "ghosted" worth its toggle?** It is the one state the other note apps lack. Two real uses so
  far (a silent conductor, a picture-only export). If those turn out rare, on and off are enough.
- **Whose defaults?** Tarot decided this once: "a show opened with no block starts from the
  defaults, never from the show open before it." Keep that for every component. A note's settings
  are its own, and ghost.cfg holds only the viewer's preferences.
