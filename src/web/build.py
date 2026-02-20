"""Simple Python build system for Praxis web frontend.

Two modes:
1. Development: Copy ES6 modules as-is to static/js/ (browser loads them natively)
2. Production: Concatenate all modules into single static/app.js file

No Node.js required - pure Python solution.
"""

import os
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "web"
STATIC_DIR = PROJECT_ROOT / "static"


def build_dev():
    """Development build - copy ES6 modules as-is for native browser import."""
    print("🔨 Building for development (ES6 modules)...")

    # Ensure static/js/ directory exists
    js_static_dir = STATIC_DIR / "js"
    js_static_dir.mkdir(parents=True, exist_ok=True)

    # Copy all JS files preserving structure
    js_src_dir = SRC_DIR / "js"
    for js_file in js_src_dir.glob("*.js"):
        shutil.copy2(js_file, js_static_dir / js_file.name)
        print(f"  ✓ Copied {js_file.name}")

    build_css()
    print("✨ Development build complete! Browser will load ES6 modules natively.\n")


def build_prod():
    """Production build - concatenate all modules into single file."""
    print("🔨 Building for production (concatenated)...")

    # Build single app.js file
    js_dir = SRC_DIR / "js"

    # Define load order (dependencies first)
    files = [
        js_dir / "state.js",
        js_dir / "components.js",
        js_dir / "api.js",
        js_dir / "dashboard.js",
        js_dir / "websocket.js",
        js_dir / "charts.js",
        js_dir / "tabs.js",
        js_dir / "mobile.js",
        js_dir / "render.js",
        js_dir / "main.js",
    ]

    output = STATIC_DIR / "app.js"
    with output.open("w", encoding="utf-8") as out:
        out.write("// Praxis Web App - Production Build\n")
        out.write("// Auto-generated - do not edit directly\n\n")

        # We need to strip import/export statements and wrap in IIFE
        out.write("(function() {\n")
        out.write("'use strict';\n\n")

        # Storage for exports from each module
        out.write("const modules = {};\n\n")

        for file in files:
            if not file.exists():
                print(f"⚠️  Warning: {file} not found, skipping...")
                continue

            out.write(f"\n// ========== {file.name} ==========\n")

            # Read and process file
            content = file.read_text(encoding="utf-8")

            # Simple transform: remove import/export for concatenated version
            # This is a naive approach - for production you'd want a real bundler
            # But it works for our simple case
            lines = content.split("\n")
            processed_lines = []

            for line in lines:
                # Skip import statements
                if line.strip().startswith("import "):
                    continue
                # Convert exports to assignments
                if line.strip().startswith("export "):
                    line = line.replace("export ", "")
                processed_lines.append(line)

            out.write("\n".join(processed_lines))
            out.write("\n")

        out.write("\n})();\n")

    size_kb = output.stat().st_size / 1024
    print(f"  ✓ Built app.js ({len(files)} modules, {size_kb:.1f}KB)")

    build_css()
    print("✨ Production build complete!\n")


def build_css():
    """Build CSS by concatenating modular files."""
    css_dir = SRC_DIR / "css"

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
        out.write("/* Auto-generated - edit src/web/css/ instead */\n\n")

        for file in files:
            if not file.exists():
                continue

            out.write(f"\n/* ========== {file.name} ========== */\n")
            out.write(file.read_text(encoding="utf-8"))
            out.write("\n")

    size_kb = output.stat().st_size / 1024
    print(
        f"  ✓ Built styles.css ({len([f for f in files if f.exists()])} files, {size_kb:.1f}KB)"
    )


def watch_and_build(mode="dev"):
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
            self.last_build = 0
            self.debounce_seconds = 0.5

        def on_modified(self, event):
            if event.is_directory:
                return

            if not event.src_path.endswith((".js", ".css")):
                return

            # Debounce
            now = time.time()
            if now - self.last_build < self.debounce_seconds:
                return

            self.last_build = now
            print(f"\n📝 Changed: {Path(event.src_path).relative_to(SRC_DIR)}")

            if mode == "dev":
                build_dev()
            else:
                build_prod()

    observer = Observer()
    handler = BuildHandler()
    observer.schedule(handler, str(SRC_DIR), recursive=True)
    observer.start()

    print(f"\n👀 Watching src/web/ for changes ({mode} mode)...")
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

    mode = "dev"  # Default to development
    watch = False

    # Parse args
    for arg in sys.argv[1:]:
        if arg == "--prod":
            mode = "prod"
        elif arg == "--watch":
            watch = True

    # Build
    if mode == "dev":
        build_dev()
    else:
        build_prod()

    # Watch if requested
    if watch:
        watch_and_build(mode)
