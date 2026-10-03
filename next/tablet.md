# ghost: the tablet medium

A chapter read as somebody browsing on a tablet lying on a desk - a rabbit hole of pages, links
and searches that keeps going deeper. Built 2026-10-01 as a proof of concept; worked example in
`rift/books/north-star/chapters/42-what-is-the-7th-realm.md` .

## What the author writes

Ordinary markdown is what is ON the screen. A few own-line marks say what the hand does:

| mark | meaning |
|---|---|
| `<!-- url: www.duckduckduck.mom -->` | the page below lives at this address; with NOTHING written under it, it is the REAL page there, shown as a capture |
| `<!-- search: What is the 7th Realm? -->` | type this into the page's search box; the results page follows |
| `<!-- new tab -->` | open a blank tab |
| `<!-- back -->` | back to the page before, scrolled where it was left (an unread page is glanced down first); follow it with a mark, not text |
| `<!-- landscape -->` / `<!-- portrait -->` | turn the picture |
| `<!-- skip -->` | shown, not read: inline = the rest of the paragraph from the end of its sentence; on its own line = the whole block below |
| `<!-- filler: 3 -->` | three placeholder stories here |
| `[word](url)` | a link (drawn as one, tappable, its target never spoken) |
| `<!-- image: ... -->` | an inline picture, from the same Illustrations library as the book |

Nothing is read off the screen: no title, and text before the first `url` is neither shown nor read.

## What is inferred, never written

- **How a page is reached.** First url: wake, open the browser on it. A url linked from the page on
  screen: scroll to the link, tap it. Anything else (or after a new tab): type it in the address bar.
- **Skimming.** Wherever the reading passes over something it does not read (12+ skipped words, a
  picture, placeholder stories), the hand SKIMS: a still beat, a slow drag, a stop on each picture,
  then the next read line. The voice rests for it. Reading scrolls are drags, not flicks.
- **A search box** appears on any page a search is made from; a page that is mostly a search box
  is drawn as an engine's front page (centred logo); a page reached by a search is its results.
- **Placeholders.** A heading with nothing under it gets a squiggled body; every page is padded with
  squiggle stories to ~2.4 screens so there is always something to scroll past.
- **A site's look** (face, accent, masthead) is hashed off its host.
- **Pace.** The hand is lazy: about 0.21 s a letter, uneven, with a pause between words and a
  look at what was typed before going; tabs and the keyboard are unhurried too.
- **Timing.** Nothing is timed by the author. Each run of actions becomes ONE rest in the voice
  (`<!-- action-hold: S -->`, spliced like a hesitation but independent of the Hesitate box), sized by
  the same `TabletScript.phases()` the medium schedules from.

## How it works

- `scripts/tablet_script.gd` - one line walk produces both the page model and the voice's text,
  so they cannot disagree. `speakable()` returns any chapter without `url:`/`search:` untouched.
- `scripts/tablet_page.gd` - typesets a page into a column (portrait 1200 / landscape 1600 logical px).
- `scripts/media/tablet.gd` - the 3D desk + slab, one screen SubViewport, and the schedule.
- Hooked in `GenerativeEditor._split_speakers` (two small changes).
- **The screen is a pure function of show time**: action groups are placed in the gap between the
  spoken words around them (finishing a beat before the next word), and the screen is replayed
  from the schedule every frame. Scroll is a list of decelerating flicks built from the reading
  (line nears the foot -> drag it near the top), skims, link targets (flicks), and turns.
- **The tablet never moves; the camera turns** a quarter round it for landscape. The content stays
  full-screen ON the glass through the turn, then the landscape layout dissolves in over it (a
  first cut counter-rotated and shrank it - "the screen detached from the tablet").
- The highlight is the book's per-letter rainbow trail, drawn glyph by glyph.
- **The camera is the journal's**: a per-page arc (wide on arrival, close as the page is read, wide
  again over its last words), slow springs (framing ~4 s, angles ~3x slower), quick only on a
  context switch. Each page sets a tiny twist and tilt; tilt is compensated in distance.

## Gates and probes

- `godot --headless --path . --script tests/tablet_check.gd` - inference, and that the voice's words
  equal the words the screen marks spoken (two-sided).
- `GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/tablet_look_probe.gd 600 --out <dir>/t --every 3 --screen 1`

## Not built yet (candidates, in rough order)

- Switching between existing tabs; a forward button.
- A visible finger instead of a touch dot; the keyboard's number/shift layers.
- ScriptMarks palette entries for the tablet marks (needs script_marks_check proofs).
- Pages that remember their scroll when revisited.

## Pictures

Every `<!-- image: -->` can be painted (Generate) or taken from disk (Import… on its row in the
Illustrations panel, or in its preview). An import lands as a new version like a painted one, is
marked `imported`, and is never stale.

## Real pages

A `url` with nothing written under it is the real page at that address. It appears in the
Illustrations panel as a "web page" row: **Capture** takes it once with Playwright's Chromium
(`capture_host/capture.py`, in ghost's own `user://capture_venv`, pinned to Praxis's Playwright so
the cached browser is reused), and **Import…** takes a screenshot instead, for a site that blocks
headless browsers. On the tablet the capture is the page, edge to edge, and since nothing on it is
read the hand lingers on it (a long glance down it) before moving on.

A first visit's pop-ups (cookie consent, newsletter, "open in app") are answered before the
capture - the most private choice first - and anything still covering the page is cleared. A site
that defeats this can still be screenshotted by hand and imported.
