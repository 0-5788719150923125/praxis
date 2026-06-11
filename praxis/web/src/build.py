"""Build system for the Praxis web frontend.

The browser loads ES modules natively (``index.html`` pulls in ``js/main.js``
with ``type="module"``), so the "build" just copies every ``src/js/*.js`` to
``static/js/`` and concatenates the CSS. There is no bundler and no separate
production path - the app only ever runs this dev build (``services.py`` calls
``build_dev`` on startup; the file watcher reruns this script on change). Pure
Python, no Node.js required.
"""

import shutil
import time
from pathlib import Path

API_DIR = Path(__file__).parent.parent
SRC_DIR = API_DIR / "src"
STATIC_DIR = API_DIR / "static"


def build_js():
    """Copy every ES module to static/js/ for native browser import.

    Globs the whole directory, so adding a new module needs no edit here.
    """
    js_static_dir = STATIC_DIR / "js"
    js_static_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for js_file in (SRC_DIR / "js").glob("*.js"):
        shutil.copy2(js_file, js_static_dir / js_file.name)
        count += 1
    print(f"  ✓ Copied {count} JS module(s)")


def guard_hover_rules(css: str) -> str:
    """Wrap every ``:hover`` rule in ``@media (hover: hover)``.

    Touch devices apply :hover on tap and KEEP it applied after the finger
    lifts - buttons look stuck in their pressed state until the next tap
    lands elsewhere. Guarding at build time fixes every current and future
    hover rule at once; rules already inside an @media get a combined
    query. Mixed selector groups split: non-hover selectors keep their
    original (unguarded) rule.
    """
    out = []
    i, n = 0, len(css)
    media_stack = []

    def emit_rule(selector: str, body: str):
        sel_list = [s.strip() for s in selector.split(",")]
        hover = [s for s in sel_list if ":hover" in s]
        plain = [s for s in sel_list if ":hover" not in s]
        if plain:
            out.append(f"{', '.join(plain)} {{{body}}}\n")
        if hover:
            cond = "(hover: hover)"
            out.append(f"@media {cond} {{ {', '.join(hover)} {{{body}}} }}\n")

    while i < n:
        # Pass comments through untouched.
        if css.startswith("/*", i):
            j = css.find("*/", i)
            j = n if j == -1 else j + 2
            out.append(css[i:j])
            i = j
            continue
        brace = css.find("{", i)
        close = css.find("}", i)
        if brace == -1 or (close != -1 and close < brace):
            # End of an @media block (or trailing text).
            if close != -1 and close < (brace if brace != -1 else n):
                out.append(css[i:close + 1])
                if media_stack:
                    media_stack.pop()
                i = close + 1
                continue
            out.append(css[i:])
            break
        header = css[i:brace]
        stripped = header.strip()
        if stripped.startswith("@") and not stripped.startswith(("@media", "@supports")):
            # @keyframes / @font-face etc: copy the whole block verbatim.
            depth, j = 1, brace + 1
            while j < n and depth:
                if css[j] == "{":
                    depth += 1
                elif css[j] == "}":
                    depth -= 1
                j += 1
            out.append(css[i:j])
            i = j
            continue
        if stripped.startswith(("@media", "@supports")):
            out.append(css[i:brace + 1])
            media_stack.append(stripped)
            i = brace + 1
            continue
        # An ordinary rule: find its closing brace (rule bodies don't nest).
        body_end = css.find("}", brace)
        body_end = n if body_end == -1 else body_end
        body = css[brace + 1:body_end]
        if ":hover" in header:
            emit_rule(header, body)
        else:
            out.append(css[i:body_end + 1] + "\n")
        i = body_end + 1
    return "".join(out)


def build_css():
    """Concatenate the modular CSS into static/styles.css (order matters)."""
    css_dir = SRC_DIR / "css"

    # Explicit order: variables/base/layout first, then components/themes/etc.
    files = [
        css_dir / "variables.css",
        css_dir / "base.css",
        css_dir / "layout.css",
        css_dir / "components.css",
        css_dir / "themes.css",
        css_dir / "animations.css",
        css_dir / "responsive.css",
    ]

    output = STATIC_DIR / "styles.css"
    STATIC_DIR.mkdir(exist_ok=True)

    with output.open("w", encoding="utf-8") as out:
        out.write("/* Praxis Web - Compiled Styles */\n")
        out.write("/* Auto-generated - edit praxis/web/src/css/ instead */\n\n")
        for file in files:
            if not file.exists():
                continue
            out.write(f"\n/* ========== {file.name} ========== */\n")
            out.write(guard_hover_rules(file.read_text(encoding="utf-8")))
            out.write("\n")

    size_kb = output.stat().st_size / 1024
    print(f"  ✓ Built styles.css ({size_kb:.1f}KB)")


def build_colormaps():
    """Generate static/js/colormaps.js from the shared colormaps.json.

    The JSON is the single source of truth (the research-paper figures read it
    too), so the dashboard and the printed PDF share one ramp. Emitted as an ES
    module: the stops plus a linear ``sampleColormap(name, t) -> [r,g,b]``.
    """
    import json

    spec_path = SRC_DIR / "colormaps.json"
    spec = {
        k: v
        for k, v in json.loads(spec_path.read_text()).items()
        if not k.startswith("_")
    }
    stops = {name: cm["stops"] for name, cm in spec.items()}

    out = STATIC_DIR / "js" / "colormaps.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "// Auto-generated from praxis/web/src/colormaps.json - do not edit.\n"
        f"export const COLORMAP_STOPS = {json.dumps(stops)};\n\n"
        "export function sampleColormap(name, t) {\n"
        "    const stops = COLORMAP_STOPS[name];\n"
        "    t = Math.max(0, Math.min(1, t));\n"
        "    for (let i = 1; i < stops.length; i++) {\n"
        "        const [p1, c1] = stops[i], [p0, c0] = stops[i - 1];\n"
        "        if (t <= p1) {\n"
        "            const f = p1 === p0 ? 0 : (t - p0) / (p1 - p0);\n"
        "            return c0.map((c, k) => Math.round(c + (c1[k] - c) * f));\n"
        "        }\n"
        "    }\n"
        "    return stops[stops.length - 1][1].slice();\n"
        "}\n",
        encoding="utf-8",
    )
    print(f"  ✓ Generated colormaps.js ({', '.join(stops)})")


def build_dev():
    """Build the frontend: copy JS modules + concatenate CSS."""
    print("🔨 Building frontend (ES6 modules)...")
    build_js()
    build_colormaps()
    build_css()
    print("✨ Build complete.\n")


def watch_and_build():
    """Watch source files and rebuild on changes."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("⚠️  watchdog not installed, watch mode unavailable")
        print("   Install with: pip install watchdog")
        return

    class BuildHandler(FileSystemEventHandler):
        def __init__(self):
            self.last_build = 0.0
            self.debounce_seconds = 0.5

        def on_modified(self, event):
            if event.is_directory or not event.src_path.endswith((".js", ".css")):
                return
            now = time.time()
            if now - self.last_build < self.debounce_seconds:
                return
            self.last_build = now
            print(f"\n📝 Changed: {Path(event.src_path).relative_to(SRC_DIR)}")
            build_dev()

    observer = Observer()
    observer.schedule(BuildHandler(), str(SRC_DIR), recursive=True)
    observer.start()
    print("\n👀 Watching praxis/web/src/ for changes...")
    print("   Press Ctrl+C to stop\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping file watcher...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    import sys

    build_dev()
    if "--watch" in sys.argv[1:]:
        watch_and_build()
