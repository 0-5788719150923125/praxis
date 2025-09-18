#!/usr/bin/env python3
"""
Simple development server with live-reload for tesla-ball-logo.html
"""

import http.server
import os
import socket
import socketserver
import threading
import time
import webbrowser
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Configuration
PORT = 8087  # Default port, will auto-find if in use
HOST = "localhost"
WATCH_FILE = "tesla-ball-logo.html"


def find_free_port(start_port=8080, max_attempts=20):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


class LiveReloadHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with live-reload injection"""

    def do_GET(self):
        if self.path == "/" or self.path == "/tesla-ball-logo.html":
            # Serve the HTML with live-reload script injected
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            with open(WATCH_FILE, "r") as f:
                content = f.read()

            # Inject live-reload script before closing body tag
            live_reload_script = """
<script>
(function() {
    let lastModified = null;

    async function checkForChanges() {
        try {
            const response = await fetch('/check-reload');
            const data = await response.json();

            if (lastModified === null) {
                lastModified = data.modified;
            } else if (lastModified !== data.modified) {
                console.log('File changed, reloading...');
                location.reload();
            }
        } catch (e) {
            console.error('Live-reload check failed:', e);
        }
    }

    // Check every 500ms
    setInterval(checkForChanges, 500);
    console.log('Live-reload enabled');
})();
</script>
"""
            content = content.replace("</body>", live_reload_script + "</body>")
            self.wfile.write(content.encode())

        elif self.path == "/check-reload":
            # Return file modification time
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            modified_time = os.path.getmtime(WATCH_FILE)
            response = f'{{"modified": {modified_time}}}'
            self.wfile.write(response.encode())

        else:
            super().do_GET()

    def log_message(self, format, *args):
        # Suppress normal request logging
        if args and isinstance(args[0], str) and "/check-reload" not in args[0]:
            print(f"{self.address_string()} - {args[0]}")


class FileChangeHandler(FileSystemEventHandler):
    """Handle file changes for notifications"""

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(WATCH_FILE):
            print(f"✨ {WATCH_FILE} modified - browser will reload automatically")


def run_server():
    """Run the development server"""
    print(f"🚀 Starting development server at http://{HOST}:{PORT}")
    print(f"📁 Serving: {os.path.abspath(WATCH_FILE)}")
    print(f"👁️  Watching for changes...")
    print(f"Press Ctrl+C to stop\n")

    # Set up file watcher
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

    # Create server with SO_REUSEADDR to avoid "Address already in use" errors
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

        def server_bind(self):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            super().server_bind()

    # Start server
    httpd = None
    try:
        httpd = ReusableTCPServer(("", PORT), LiveReloadHandler)
        # Open browser after a short delay
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")
        ).start()
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Error: Port {PORT} is already in use")
            print(f"Try one of these solutions:")
            print(f"  1. Wait a few seconds and try again")
            print(f"  2. Use a different port by editing PORT in serve.py")
            print(f"  3. Find and kill the process: lsof -i :{PORT}")
        else:
            raise
    finally:
        if httpd:
            httpd.shutdown()
            httpd.server_close()
        observer.stop()
        observer.join()


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import watchdog
    except ImportError:
        print("Installing watchdog for file watching...")
        import subprocess

        subprocess.check_call(["pip", "install", "watchdog"])
        print("Watchdog installed successfully!\n")
        import watchdog

    # Change to the staging directory
    os.chdir(Path(__file__).parent)

    # Check if HTML file exists
    if not os.path.exists(WATCH_FILE):
        print(f"❌ Error: {WATCH_FILE} not found in current directory")
        exit(1)

    # Try to find a free port if the default is taken
    available_port = find_free_port(PORT)
    if available_port and available_port != PORT:
        print(f"⚠️  Port {PORT} is in use, using port {available_port} instead")
        PORT = available_port
    elif not available_port:
        print(f"❌ Error: Could not find any free port from {PORT} to {PORT + 20}")
        exit(1)
    else:
        # Port is available as-is
        pass

    run_server()
