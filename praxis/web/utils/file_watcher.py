"""File watching utilities for live reload."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Resolve paths relative to the api package, not CWD
_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TEMPLATES_DIR = os.path.join(_API_DIR, "templates")
_STATIC_DIR = os.path.join(_API_DIR, "static")
_SRC_DIR = os.path.join(_API_DIR, "src")
_BUILD_SCRIPT = os.path.join(_SRC_DIR, "build.py")


class TemplateChangeHandler(FileSystemEventHandler):
    """Watch for changes in template and static files and emit live-reload events."""

    def __init__(self, socketio, namespace: str = "/live-reload"):
        """Initialize the handler.

        Args:
            socketio: SocketIO instance for emitting events
            namespace: WebSocket namespace for live reload
        """
        self.socketio = socketio
        self.namespace = namespace

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            # Check if the file is in templates or static directory
            if event.src_path.startswith(_TEMPLATES_DIR) or event.src_path.startswith(
                _STATIC_DIR
            ):
                # Emit reload event to all connected clients
                try:
                    self.socketio.emit("reload", namespace=self.namespace)
                except Exception as e:
                    print(f"Error sending reload signal: {str(e)}")

            # Check if the file is in src directory
            elif event.src_path.startswith(_SRC_DIR):
                # Rebuild web frontend when source files change
                try:
                    print(f"[WEB] Detected change in {event.src_path}, rebuilding...")
                    if os.path.exists(_BUILD_SCRIPT):
                        result = subprocess.run(
                            ["python", _BUILD_SCRIPT],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if result.returncode == 0:
                            print("[WEB] ✓ Build complete, reloading browser...")
                            # Emit reload after successful build
                            self.socketio.emit("reload", namespace=self.namespace)
                        else:
                            print(f"[WEB] ✗ Build failed: {result.stderr}")
                except Exception as e:
                    print(f"[WEB] Error rebuilding: {str(e)}")


class TemplateWatcher:
    """Simple watcher to monitor template, static, and web source file changes."""

    def __init__(self, socketio, namespace: str = "/live-reload"):
        """Initialize the watcher.

        Args:
            socketio: SocketIO instance for emitting events
            namespace: WebSocket namespace for live reload
        """
        self.observer: Optional[Observer] = None
        self.template_dir = _TEMPLATES_DIR
        self.static_dir = _STATIC_DIR
        self.src_web_dir = _SRC_DIR
        self.socketio = socketio
        self.namespace = namespace

    def start(self) -> None:
        """Start watching for file changes."""
        try:
            self.observer = Observer()
            event_handler = TemplateChangeHandler(self.socketio, self.namespace)
            # Watch templates, static, and src/web directories
            self.observer.schedule(event_handler, self.template_dir, recursive=True)
            self.observer.schedule(event_handler, self.static_dir, recursive=True)

            # Only watch src/web if it exists
            if os.path.exists(self.src_web_dir):
                self.observer.schedule(event_handler, self.src_web_dir, recursive=True)
                print(
                    "[WEB] Auto-rebuild enabled: watching praxis/web/src/ for changes"
                )

            self.observer.start()
        except Exception as e:
            print(f"Error starting file watcher: {str(e)}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join()
            except Exception as e:
                print(f"Error stopping template watcher: {str(e)}")
