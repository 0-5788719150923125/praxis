"""File watching utilities for live reload."""

import os
from typing import Optional
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


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
            templates_dir = os.path.abspath("templates")
            static_dir = os.path.abspath("static")

            # Check if the file is in templates or static directory
            if event.src_path.startswith(templates_dir) or event.src_path.startswith(
                static_dir
            ):
                # Emit reload event to all connected clients
                try:
                    self.socketio.emit("reload", namespace=self.namespace)
                except Exception as e:
                    print(f"Error sending reload signal: {str(e)}")


class TemplateWatcher:
    """Simple watcher to monitor template and static file changes."""

    def __init__(self, socketio, namespace: str = "/live-reload"):
        """Initialize the watcher.

        Args:
            socketio: SocketIO instance for emitting events
            namespace: WebSocket namespace for live reload
        """
        self.observer: Optional[Observer] = None
        self.template_dir = os.path.abspath("templates")
        self.static_dir = os.path.abspath("static")
        self.socketio = socketio
        self.namespace = namespace

    def start(self) -> None:
        """Start watching for file changes."""
        try:
            self.observer = Observer()
            event_handler = TemplateChangeHandler(self.socketio, self.namespace)
            # Watch both templates and static directories
            self.observer.schedule(event_handler, self.template_dir, recursive=True)
            self.observer.schedule(event_handler, self.static_dir, recursive=True)
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
