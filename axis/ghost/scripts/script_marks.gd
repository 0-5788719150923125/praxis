extends RefCounted
class_name ScriptMarks

## ScriptMarks - every authoring mark a script may carry, registered with what it does.
##
## A script is plain Markdown, and ghost reads more out of it than the words: speaker cues,
## hesitations, picture markers, inline pronunciations, template macros, emphasis. Each of
## those used to be documented only where it was implemented, so a new author had no way to
## find out what ghost understands short of reading the source. This is the one list: the
## [ScriptWriter] palette is built from it, [ScriptHighlighter] colors a script with its
## patterns, and docs.py writes docs/script.md from it.
##
## AN ENTRY DOES NOT IMPLEMENT ITS MARK. The parsers of record stay where they are
## ([Manuscript] for cues, rests and pictures; [TextNorm] and [Phonemes] for the rest), and
## `tests/script_marks_check.gd` inserts every entry's example and asks THOSE parsers what
## they make of it - so an entry that drifts from what the reader actually does fails there,
## and a new entry without a check fails too.

## Fields of an entry:
##   label    the palette button
##   group    a key of [constant GROUPS]
##   blurb    what it does, for the palette and the docs (one or two sentences)
##   before / fill / after   the text inserted. A selection replaces `fill`, and whatever
##            ends up in the fill is left selected so typing replaces it.
##   line     "own": the mark goes on a line of its own; "start": at the head of the line;
##            "" anywhere
##   modes    the panels that honor it ("generative", "synthesis")
##   pattern  how [ScriptHighlighter] finds it. Registry ORDER is highlight priority: a span
##            claimed by an earlier entry is not recolored by a later one.

const REGISTRY := {
	"speaker": {
		"label": "Speaker cue",
		"group": "voices",
		"blurb": "Hands everything after it to the named voice, until the next cue. The panel's voices list follows the names the script uses. Must be a line of its own, or it is only a note.",
		"before": "<!-- speaker: ", "fill": "Name", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": "(?m)^[ \\t]*(?:<!--\\s*speaker\\s*:\\s*.+?\\s*-->|\\[\\s*speaker\\s*:\\s*[^\\]]+?\\s*\\])[ \\t]*$",
	},
	"hesitation": {
		"label": "Hesitation",
		"group": "timing",
		"blurb": "A longer rest at exactly this point, mid-sentence included. Its length is the panel's Hesitate dial.",
		"before": "<!-- hesitation -->", "fill": "", "after": "", "line": "",
		"modes": ["generative"],
		"pattern": Manuscript.HESITATION,
	},
	"delivery": {
		"label": "Delivery",
		"group": "timing",
		"blurb": "Lean how the rest of this paragraph is said, a step at a time: quicker / slower, brighter / graver, tighter / looser (pauses), louder / softer, or excited, serious, hushed, urgent, playful, tender, dry; steady is the voice's own. Small, and eased in and out over a few sentences.",
		"before": "<!-- delivery: ", "fill": "quicker, brighter", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": Manuscript.DELIVERY,
	},
	"hesitation_timed": {
		"label": "Timed hesitation",
		"group": "timing",
		"blurb": "A rest of exactly this many seconds, whatever the Hesitate dial says.",
		"before": "<!-- hesitation: ", "fill": "2.5", "after": " -->", "line": "",
		"modes": ["generative"],
		"pattern": Manuscript.HESITATION,
	},
	"timestamp": {
		"label": "Log-entry time",
		"group": "timing",
		"blurb": "A time opening a paragraph is read as its heading, with a hesitation after it. The Notebook sets it in the margin.",
		"before": "", "fill": "21:40", "after": " ", "line": "start",
		"modes": ["generative"],
		"pattern": Manuscript.TIMESTAMP_LEAD,
	},
	"hum": {
		"label": "Thinking hum",
		"group": "timing",
		"blurb": "Hm, Hmm, Mmm as a sentence of their own are held to most of a second, settling and falling in pitch, instead of read as a clipped syllable.",
		"before": "", "fill": "Hmm.", "after": "", "line": "",
		"modes": ["generative"],
		"pattern": "(?<![\\w'])(?:[Hh]m{1,3}|[Mm]{2,3})[.!?…]",
	},
	"image": {
		"label": "Picture",
		"group": "pictures",
		"blurb": "An illustration, described, placed where it sits: a full page when it opens the chapter or a scene, beside the text otherwise. Generated on request in the Illustrations panel.",
		"before": "<!-- image: ", "fill": "what the picture shows", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": Manuscript.IMAGE,
	},
	"image_full": {
		"label": "Full-page picture",
		"group": "pictures",
		"blurb": "A picture pinned to a page of its own, wherever it sits.",
		"before": "<!-- image (full): ", "fill": "what the picture shows", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": Manuscript.IMAGE,
	},
	"image_left": {
		"label": "Picture, left edge",
		"group": "pictures",
		"blurb": "A picture beside the text, pinned to the left edge. (right) and (inline) pin it the other ways.",
		"before": "<!-- image (left): ", "fill": "what the picture shows", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": Manuscript.IMAGE,
	},
	"sketch": {
		"label": "Sketch",
		"group": "pictures",
		"blurb": "A drawing made on the page in the writer's own ink, black on white, without the book's style or references.",
		"before": "<!-- sketch: ", "fill": "a wiring diagram", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": Manuscript.IMAGE,
	},
	"phonetic": {
		"label": "Pronunciation",
		"group": "pronunciation",
		"blurb": "Spells a word in ARPAbet phonemes, read exactly as written. For a name the voice gets wrong.",
		"before": "[", "fill": "K AE T", "after": "]", "line": "",
		"modes": ["generative", "synthesis"],
		"pattern": "\\[[A-Za-z]{1,2}[0-2]?(?:\\s+[A-Za-z]{1,2}[0-2]?)*\\]",
	},
	"macro": {
		"label": "Template macro",
		"group": "text",
		"blurb": "A placeholder for a fact only the book's build knows. Ghost Notes reads the default after the colon, never the macro itself; one with no default is skipped and named in the panel.",
		"before": "${", "fill": "CHAPTERS_BEFORE:twenty-one", "after": "}", "line": "",
		"modes": ["generative", "synthesis"],
		"pattern": "\\$\\{[A-Za-z_][A-Za-z0-9_.]*(?::(?:[^{}]|\\{[^{}]*\\})*)?\\}",
	},
	"heading": {
		"label": "Heading",
		"group": "typography",
		"blurb": "Spoken like any line, set as a heading on the page. The frontmatter's title is read first on its own.",
		"before": "# ", "fill": "Heading", "after": "", "line": "start",
		"modes": ["generative", "synthesis"],
		"pattern": "(?m)^#{1,6}[ \\t].*$",
	},
	"scene_line": {
		"label": "Scene line",
		"group": "typography",
		"blurb": "A short paragraph entirely in italics announces a change of time or place; a picture just before it gets a full page.",
		"before": "*", "fill": "Nine-thirty. Orientation.", "after": "*", "line": "own",
		"modes": ["generative"],
		"pattern": "(?m)^[ \\t]*\\*[^*\\n]{1,158}\\*[ \\t]*$",
	},
	"italic": {
		"label": "Italic",
		"group": "typography",
		"blurb": "Never spoken as a mark; shown slanted in the subtitles and on the page.",
		"before": "*", "fill": "emphasized", "after": "*", "line": "",
		"modes": ["generative", "synthesis"],
		"pattern": "(?<![*\\w])\\*(?![*\\s])[^*\\n]*?[^*\\s]\\*(?![*\\w])|(?<![*\\w])\\*[^*\\s]\\*(?![*\\w])",
	},
	"bold": {
		"label": "Bold",
		"group": "typography",
		"blurb": "Never spoken as a mark; shown emboldened in the subtitles and on the page.",
		"before": "**", "fill": "strong", "after": "**", "line": "",
		"modes": ["generative", "synthesis"],
		"pattern": "\\*\\*(?![*\\s])[^*\\n]*?\\*\\*",
	},
	"rule": {
		"label": "Scene break",
		"group": "typography",
		"blurb": "A rule across the page between two scenes. Silent.",
		"before": "* * *", "fill": "", "after": "", "line": "own",
		"modes": ["generative"],
		"pattern": "(?m)^[ \\t]*(?:\\* \\* \\*|\\*\\*\\*|---)[ \\t]*$",
	},
	"outro": {
		"label": "Outro starts here",
		"group": "timing",
		"blurb": "The end begins at this point: the reading carries on but fades out - voice and picture together, over the Outro setting - and whatever text is left is cut. Own line or mid-sentence; the first one counts.",
		"before": "<!-- outro -->", "fill": "", "after": "", "line": "",
		"modes": ["generative"],
		"pattern": Manuscript.OUTRO,
	},
	"url": {
		"label": "Web page",
		"group": "tablet",
		"blurb": "Tablet: the page below lives at this address. How it is reached is worked out - a link on the page on screen is tapped, anything else is typed in. With nothing written under it, it is the REAL page there, captured from the web.",
		"before": "<!-- url: ", "fill": "www.example.com", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*url\\s*:[^>]*-->",
	},
	"search": {
		"label": "Search",
		"group": "tablet",
		"blurb": "Tablet: type this into a search box - the page's own, on a search engine, else the address bar - and the results page follows.",
		"before": "<!-- search: ", "fill": "what is the 7th realm", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*search\\s*:[^>]*-->",
	},
	"new_tab": {
		"label": "New tab",
		"group": "tablet",
		"blurb": "Tablet: a pause to think, then a blank tab; the next web page is typed into it.",
		"before": "<!-- new tab -->", "fill": "", "after": "", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*new tab\\s*-->",
	},
	"back": {
		"label": "Back",
		"group": "tablet",
		"blurb": "Tablet: back to the page before, scrolled where it was left. An unread page is glanced down first. Follow it with a mark, not text.",
		"before": "<!-- back -->", "fill": "", "after": "", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*back\\s*-->",
	},
	"redirect": {
		"label": "Redirect",
		"group": "tablet",
		"blurb": "Tablet: put it before a url - the tap (or typed address) that reaches that page is sent through a chain of redirects the reader never asked for, the address bar jumping through them on its own. Bare: six hops made up from the page's own site, a scam's funnel. Or name them, comma-separated.",
		"before": "<!-- redirect", "fill": "", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*redirect\\s*(?::[^>]*)?-->",
	},
	"notify": {
		"label": "Notification",
		"group": "tablet",
		"blurb": "Tablet: a notification banner drops from the top of the screen a moment after the reading reaches this point - mid-paragraph or on a line of its own - holds a few seconds and slides away: an email arriving. Sender | subject | preview. Not read aloud; the voice does not wait for it.",
		"before": "<!-- notify: ", "fill": "Sender | Subject | A line of the message", "after": " -->", "line": "",
		"modes": ["generative"],
		"pattern": "<!--\\s*notify\\s*:[^>]*-->",
	},
	"landscape": {
		"label": "Landscape",
		"group": "tablet",
		"blurb": "Tablet: turn to landscape. The camera turns; the tablet stays where it lies.",
		"before": "<!-- landscape -->", "fill": "", "after": "", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*landscape\\s*-->",
	},
	"portrait": {
		"label": "Portrait",
		"group": "tablet",
		"blurb": "Tablet: turn back to portrait.",
		"before": "<!-- portrait -->", "fill": "", "after": "", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*portrait\\s*-->",
	},
	"skip": {
		"label": "Skip",
		"group": "tablet",
		"blurb": "Tablet: shown, not read. Mid-paragraph, the rest of it from the end of that sentence; on its own line, the whole paragraph below.",
		"before": "<!-- skip -->", "fill": "", "after": "", "line": "",
		"modes": ["generative"],
		"pattern": "<!--\\s*skip\\s*-->",
	},
	"filler": {
		"label": "Placeholder stories",
		"group": "tablet",
		"blurb": "Tablet: this many placeholder stories - squiggles where writing would be - on the page here.",
		"before": "<!-- filler: ", "fill": "3", "after": " -->", "line": "own",
		"modes": ["generative"],
		"pattern": "<!--\\s*filler\\s*(?::[^>]*)?-->",
	},
	"note": {
		"label": "Note to self",
		"group": "notes",
		"blurb": "Any other HTML comment is an authoring note: never spoken, never printed.",
		"before": "<!-- ", "fill": "a note to yourself", "after": " -->", "line": "",
		"modes": ["generative", "synthesis"],
		"pattern": Manuscript.COMMENT,
	},
}

## Palette sections, in order, with the color [ScriptHighlighter] gives their marks.
const GROUPS := {
	"voices": {"label": "Voices", "color": Color(0.98, 0.72, 0.35)},
	"timing": {"label": "Timing", "color": Color(0.55, 0.85, 0.95)},
	"pictures": {"label": "Pictures", "color": Color(0.75, 0.62, 1.0)},
	"pronunciation": {"label": "Pronunciation", "color": Color(0.55, 0.95, 0.6)},
	"text": {"label": "Text", "color": Color(1.0, 0.6, 0.75)},
	"typography": {"label": "Typography", "color": Color(0.95, 0.9, 0.6)},
	"tablet": {"label": "Tablet", "color": Color(0.5, 0.75, 1.0)},
	"notes": {"label": "Notes", "color": Color(0.6, 0.6, 0.66)},
}


## The keys a panel honors, in registry order.
static func for_mode(mode: String) -> PackedStringArray:
	var out := PackedStringArray()
	for k in REGISTRY:
		if (REGISTRY[k]["modes"] as Array).has(mode):
			out.append(k)
	return out


## The highlight color of a mark.
static func color_of(key: String) -> Color:
	return GROUPS[REGISTRY[key]["group"]]["color"]


## The mark's text with [param fill] (the example when empty).
static func example(key: String, fill := "") -> String:
	var e: Dictionary = REGISTRY[key]
	return String(e["before"]) + (fill if not fill.is_empty() else String(e["fill"])) \
		+ String(e["after"])


## INSERT A MARK at the caret of [param te], as the palette does.
##
## A selection becomes the fill - select a name and press Speaker cue, and it is that
## speaker's cue - except for a mark with no fill, which goes after the selection rather
## than over it. Whatever ends up in the fill is left SELECTED, so the example is replaced
## by typing. One undo step. [param value] fills the mark outright (a known speaker's name),
## and is then left unselected.
static func insert(te: TextEdit, key: String, value := "") -> void:
	var e: Dictionary = REGISTRY[key]
	var before := String(e["before"])
	var after := String(e["after"])
	var fill := String(e["fill"]) if value.is_empty() else value
	var where := String(e["line"])
	te.begin_complex_operation()
	if te.has_selection():
		var sel := te.get_selected_text()
		if fill.is_empty() or not value.is_empty() or sel.contains("\n"):
			te.set_caret_line(te.get_selection_to_line())
			te.set_caret_column(te.get_selection_to_column())
			te.deselect()
		else:
			fill = sel
			te.delete_selection()
	var line := te.get_caret_line()
	var col := te.get_caret_column()
	var text := te.get_line(line)
	if where == "start":
		col = 0
	elif where == "own":
		if not text.substr(0, col).strip_edges().is_empty():
			before = "\n" + before
		if not text.substr(col).strip_edges().is_empty():
			after += "\n"
	te.set_caret_line(line)
	te.set_caret_column(col)
	te.insert_text_at_caret(before)
	var l0 := te.get_caret_line()
	var c0 := te.get_caret_column()
	te.insert_text_at_caret(fill)
	var l1 := te.get_caret_line()
	var c1 := te.get_caret_column()
	te.insert_text_at_caret(after)
	if not fill.is_empty() and value.is_empty():
		te.select(l0, c0, l1, c1)
	te.end_complex_operation()
