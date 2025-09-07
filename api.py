import glob
import inspect
import logging
import os
import socket
import sys
import time
from threading import Event, Thread

from flask import (
    Flask,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
)
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from werkzeug.serving import make_server

# Import terminal streaming components
try:
    from flask_socketio import Namespace, emit

    import interface  # Import interface for dashboard streaming functions

    terminal_available = True
except ImportError as e:
    terminal_available = False

logger = logging.getLogger("werkzeug")
logger.setLevel(logging.ERROR)

app = Flask(__name__)
app.static_folder = "static"
app.debug = True  # Enable debug mode for development

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# WSGI middleware hooks for modules
_wsgi_middleware = []


def register_wsgi_middleware(middleware_func):
    """Register a WSGI middleware function from modules"""
    _wsgi_middleware.append(middleware_func)


def apply_wsgi_middleware():
    """Apply all registered WSGI middleware to the app"""
    if _wsgi_middleware:
        for middleware in _wsgi_middleware:
            app.wsgi_app = middleware(app.wsgi_app)


# Set up SocketIO for live reload
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Integration middleware support
_request_middleware = []
_response_middleware = []
_response_headers = []  # List of (header_name, header_value) tuples to add to all responses


def register_request_middleware(func):
    """Register a request middleware function from modules"""
    _request_middleware.append(func)


def register_response_middleware(func):
    """Register a response middleware function from modules"""
    _response_middleware.append(func)


def register_response_header(header_name, header_value):
    """Register a header to be added to all responses"""
    _response_headers.append((header_name, header_value))


@app.before_request
def process_request_middleware():
    """Process all registered request middleware"""
    for middleware in _request_middleware:
        result = middleware(request, None)
        if result is not None:
            return result
    return None


@app.after_request
def process_response_middleware(response):
    """Process all registered response middleware and headers"""
    # Add any registered headers
    for header_name, header_value in _response_headers:
        response.headers[header_name] = header_value
    
    # Process middleware functions
    for middleware in _response_middleware:
        middleware(request, response)
    return response


# Explicit static file route with proper headers
@app.route("/static/<path:filename>")
def serve_static_files(filename):
    """Serve static files with proper headers for CORS"""
    response = send_from_directory(app.static_folder, filename)
    response.headers["Access-Control-Allow-Origin"] = "*"
    # Add cache control for better performance
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


# Get all template files for monitoring
templates_to_watch = glob.glob("templates/*.*")
static_to_watch = glob.glob("static/*.*")


# Live-reload socket connection endpoint
@socketio.on("connect", namespace="/live-reload")
def handle_connect():
    # print("Live-reload client connected")
    pass


@socketio.on("disconnect", namespace="/live-reload")
def handle_disconnect():
    # print("Live-reload client disconnected")
    pass


# Test endpoint to verify the server is accessible
@app.route("/api/ping", methods=["GET", "POST", "OPTIONS"])
def ping():
    """Simple endpoint to test if API is accessible"""
    response = jsonify({"status": "ok", "message": "Praxis API server is running"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response


@app.route("/api/spec", methods=["GET", "OPTIONS"])
def get_spec():
    """Get model specification including hashes and CLI arguments"""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
        return response

    try:
        # Import CLI utilities to get configuration
        import hashlib
        import json

        from cli import get_cli_args

        # Get CLI args
        args = get_cli_args()

        # Convert args to dict, filtering out non-serializable items
        args_dict = {}
        for key, value in vars(args).items():
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                args_dict[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                args_dict[key] = str(value)

        # Use the hashes from app config instead of reading from disk
        truncated_hash = app.config.get("truncated_hash")
        full_hash = app.config.get("full_hash")

        # Fallback for backward compatibility if hashes weren't provided
        if truncated_hash and not full_hash:
            import hashlib
            full_hash = hashlib.sha256(truncated_hash.encode()).hexdigest()

        # If neither hash exists, mark as unknown
        if not truncated_hash:
            truncated_hash = "unknown"
            full_hash = "unknown"

        # Get the model architecture string
        model_arch = None
        try:
            generator = app.config.get("generator")
            if generator and hasattr(generator, "model"):
                model = generator.model
                # Get the string representation of the model
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    print(model)
                model_arch = f.getvalue()
        except Exception as e:
            model_arch = f"Error getting model architecture: {str(e)}"

        # Use the simplified param_stats from the app config if available
        param_stats = app.config.get("param_stats", {})

        # If not available in config, calculate it
        if not param_stats:
            try:
                from praxis.optimizers import get_parameter_stats

                generator = app.config.get("generator")
                if generator and hasattr(generator, "model"):
                    model = generator.model
                    # Get simplified stats (just model and optimizer counts)
                    param_stats = get_parameter_stats(model)
            except:
                param_stats = {}

        # Get metadata from history.log
        timestamp = None
        command = None
        if os.path.exists("history.log"):
            with open("history.log", "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split(" | ")
                    if len(parts) >= 3:
                        timestamp = parts[0]
                        command = parts[2].strip('"')

        # Get the appropriate git URL - prioritize ngrok if active
        git_url = None

        # First check if ngrok is active (regardless of how request came in)
        ngrok_url = app.config.get("ngrok_url")
        ngrok_secret = app.config.get("ngrok_secret")
        
        if ngrok_url and ngrok_secret:
            # Ngrok is active - always use the protected URL with /praxis
            git_url = f"{ngrok_url}/{ngrok_secret}/praxis"
        else:
            # No ngrok - use direct URL based on request
            host = request.host.split(":")[0] if ":" in request.host else request.host
            
            if host.endswith(".ngrok-free.app") or host.endswith(".ngrok.io") or host.endswith(".src.eco"):
                # Accessed through ngrok but no secret configured (shouldn't happen)
                git_url = f"https://{host}/praxis"
            else:
                # Local URL with port
                port = request.host.split(":")[1] if ":" in request.host else "80"
                # Use /praxis path (works with or without .git)
                git_url = f"http://{host}:{port}/praxis"

        # Import mask_git_url function
        from praxis.utils import mask_git_url

        # Return clean data structure
        spec = {
            "truncated_hash": truncated_hash,
            "full_hash": full_hash,
            "args": args_dict,
            "model_architecture": model_arch,
            "param_stats": param_stats,
            "timestamp": timestamp,
            "command": command,
            "git_url": git_url,
            "masked_git_url": mask_git_url(git_url) if git_url else None,
            "seed": app.config.get("seed"),
        }

        response = jsonify(spec)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500


# Git HTTP backend routes - support multiple URL patterns
@app.route("/praxis.git/<path:git_path>", methods=["GET", "POST"])
@app.route("/praxis.git", methods=["GET", "POST"], defaults={"git_path": ""})
@app.route("/praxis/<path:git_path>", methods=["GET", "POST"])  # Without .git suffix
@app.route("/praxis", methods=["GET", "POST"], defaults={"git_path": ""})  # Without .git suffix
@app.route("/info/refs", methods=["GET"])  # Git discovery at root
@app.route("/git-upload-pack", methods=["POST"])  # Git fetch at root
@app.route(
    "/src/<path:git_path>", methods=["GET", "POST"]
)  # Keep for backward compatibility
def git_http_backend(git_path=None):
    """
    Simple Git HTTP backend for read-only access to the repository.
    Supports git clone and fetch operations.
    Accessible via:
    - With .git suffix: git clone https://domain.com/praxis.git
    - Without suffix: git clone https://domain.com/praxis
    - Legacy /src path: git clone https://domain.com/src
    """
    import subprocess

    from flask import Response, stream_with_context

    # Security: Only allow specific git commands for read-only access
    allowed_services = ["git-upload-pack", "git-receive-pack"]

    # Parse the service from the path
    service = request.args.get("service")

    # Handle root-level git operations (when git_path is None)
    if git_path is None:
        if request.path == "/info/refs":
            git_path = "info/refs"
        elif request.path == "/git-upload-pack":
            git_path = "git-upload-pack"

    # Handle info/refs request (git discovery)
    if (git_path == "info/refs" or git_path == "") and service:
        if not service.startswith("git-"):
            return "Invalid service", 400

        service_name = service.replace("git-", "")
        if service not in allowed_services:
            return "Service not allowed", 403

        # Only allow upload-pack for read-only access
        if service != "git-upload-pack":
            return "Only read access is allowed", 403

        try:
            # Run git command to get refs
            cmd = ["git", "upload-pack", "--stateless-rpc", "--advertise-refs", "."]
            result = subprocess.run(cmd, capture_output=True, cwd=os.getcwd())

            # Format response for git HTTP protocol
            response_data = f"001e# service={service}\n0000" + result.stdout.decode(
                "latin-1"
            )

            return Response(
                response_data,
                content_type=f"application/x-{service}-advertisement",
                headers={
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                },
            )
        except Exception as e:
            return f"Git error: {str(e)}", 500

    # Handle git-upload-pack request (actual clone/fetch)
    elif git_path == "git-upload-pack" and request.method == "POST":
        try:
            # Run git upload-pack with the request data
            cmd = ["git", "upload-pack", "--stateless-rpc", "."]
            result = subprocess.run(
                cmd, input=request.data, capture_output=True, cwd=os.getcwd()
            )

            return Response(
                result.stdout,
                content_type="application/x-git-upload-pack-result",
                headers={
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                },
            )
        except Exception as e:
            return f"Git error: {str(e)}", 500

    # Return 404 for other paths
    return "Not found", 404


@app.route("/api/agents", methods=["GET", "OPTIONS"])
def get_agents():
    """Get git remotes as peer agents with their online/offline status"""
    if request.method == "OPTIONS":
        return handle_cors_preflight()

    import concurrent.futures
    import subprocess
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from praxis.utils import mask_git_url

    agents = []

    # Always add the current instance as "self"
    try:
        # Get current port
        current_port = int(request.environ.get("SERVER_PORT", 2100))

        # Always add the current instance first by getting the data directly
        try:
            # Get our git URL - prioritize ngrok if active
            ngrok_url = app.config.get("ngrok_url")
            ngrok_secret = app.config.get("ngrok_secret")
            
            if ngrok_url and ngrok_secret:
                # Ngrok is active - use the protected URL
                git_url = f"{ngrok_url}/{ngrok_secret}/praxis"
            else:
                # No ngrok - use local URL
                git_url = f"http://localhost:{current_port}/praxis"

            # Get git hash
            import subprocess

            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                )
                full_hash = (
                    result.stdout.strip() if result.returncode == 0 else "unknown"
                )
                short_hash = full_hash[:9] if len(full_hash) >= 9 else full_hash
            except:
                full_hash = "unknown"
                short_hash = "unknown"

            agents.append(
                {
                    "name": "self",
                    "url": git_url,
                    "masked_url": mask_git_url(git_url),
                    "status": "online",
                    "commit_hash": full_hash,
                    "short_hash": short_hash,
                }
            )
        except Exception as e:
            print(f"[WARNING] Failed to add self agent: {e}")
            import traceback

            traceback.print_exc()

        # Check if this is in the standard port range (2100-2119)
        is_standard_port = 2100 <= current_port < 2120

        if is_standard_port:
            # Scan ports 2100-2119 for local instances
            local_instances = []

            def check_local_port(port):
                try:
                    import json

                    spec_url = f"http://localhost:{port}/api/spec"
                    req = urllib.request.Request(spec_url)
                    with urllib.request.urlopen(req, timeout=0.5) as response:
                        if response.status == 200:
                            spec_data = json.loads(response.read())
                            if spec_data.get("git_url"):
                                return {
                                    "port": port,
                                    "git_url": spec_data["git_url"],
                                    "masked_url": spec_data.get(
                                        "masked_git_url",
                                        mask_git_url(spec_data["git_url"]),
                                    ),
                                    "full_hash": spec_data.get("full_hash"),
                                    "truncated_hash": spec_data.get("truncated_hash"),
                                }
                except:
                    pass
                return None

            # Check ports concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for port in range(2100, 2120):
                    future = executor.submit(check_local_port, port)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=1)
                        if result:
                            local_instances.append(result)
                    except:
                        pass

            # Sort by port and create additional self agents (self-1, self-2, etc)
            # Skip the current port since we already added it
            local_instances = [
                i for i in local_instances if i and i["port"] != current_port
            ]
            local_instances.sort(key=lambda x: x["port"])

            for idx, instance in enumerate(local_instances):
                name = f"self-{idx + 1}"  # Start numbering from 1
                agents.append(
                    {
                        "name": name,
                        "url": instance["git_url"],
                        "masked_url": instance["masked_url"],
                        "status": "online",
                        "commit_hash": instance["full_hash"],
                        "short_hash": instance["truncated_hash"],
                    }
                )
    except Exception as e:
        # If local detection fails, continue with remote agents
        print(f"[DEBUG] Error in self agent detection: {e}")
        import traceback

        traceback.print_exc()

    try:
        # Get git remotes
        result = subprocess.run(
            ["git", "remote", "-v"], capture_output=True, text=True, cwd=os.getcwd()
        )

        if result.returncode != 0:
            return jsonify({"agents": [], "error": "Failed to get git remotes"}), 200

        # Parse remotes (each remote appears twice - for fetch and push)
        remotes = {}
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    url = parts[1]
                    # Only store each remote once
                    if name not in remotes:
                        remotes[name] = url

        def check_remote_status(name, url):
            """Check if a remote is accessible (online/offline) and get its latest commit"""
            agent = {
                "name": name,
                "url": url,
                "masked_url": mask_git_url(url),  # Add masked URL
                "status": "offline",
                "commit_hash": None,
                "short_hash": None,
            }

            # Try to check if the remote is accessible using git ls-remote
            # This works for all git URLs (http, https, ssh, git, local paths)
            try:
                check_result = subprocess.run(
                    ["git", "ls-remote", url, "HEAD"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    cwd=os.getcwd(),
                    timeout=3,
                    text=True,
                )
                if check_result.returncode == 0:
                    # Parse the commit hash from ls-remote output
                    # Format is: <hash>\tHEAD
                    output = check_result.stdout.strip()
                    if output:
                        commit_hash = output.split("\t")[0]
                        agent["commit_hash"] = commit_hash
                        agent["short_hash"] = commit_hash[:7] if commit_hash else None

                    # Check if this is a Praxis instance by looking for /api/agents endpoint
                    # Extract base URL if it's an HTTP(S) git URL
                    is_praxis = False
                    if url.startswith(("http://", "https://")):
                        # Remove .git suffix and path components to get base URL
                        base_url = url.replace(".git", "").rstrip("/")
                        # If it looks like a GitHub/GitLab URL, it's not a Praxis instance
                        if (
                            "github.com" in base_url
                            or "gitlab.com" in base_url
                            or "bitbucket.org" in base_url
                        ):
                            agent["status"] = "archived"
                        else:
                            # Try to check for Praxis API endpoint
                            try:
                                import urllib.request

                                api_url = f"{base_url}/api/agents"
                                req = urllib.request.Request(
                                    api_url,
                                    headers={"User-Agent": "Praxis-Agent-Check"},
                                )
                                with urllib.request.urlopen(req, timeout=2) as response:
                                    if response.status == 200:
                                        is_praxis = True
                                        agent["status"] = "online"
                                    else:
                                        agent["status"] = "archived"
                            except:
                                # If API check fails, assume it's just a regular git repo
                                agent["status"] = "archived"
                    else:
                        # For SSH/git protocol URLs, we can't easily check for Praxis
                        # Mark as archived (regular git repo)
                        agent["status"] = "archived"

            except subprocess.TimeoutExpired:
                # Timeout means offline
                pass
            except Exception:
                # If any check fails, keep status as offline
                pass

            return agent

        # Check status of each remote in parallel with timeout
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for name, url in remotes.items():
                future = executor.submit(check_remote_status, name, url)
                futures.append(future)

            # Collect results with timeout
            for future in futures:
                try:
                    agent = future.result(timeout=3)
                    agents.append(agent)
                except concurrent.futures.TimeoutError:
                    # If timeout, add as offline
                    agents.append(
                        {
                            "name": "unknown",
                            "url": "unknown",
                            "status": "offline",
                            "type": "unknown",
                        }
                    )

        # Sort agents by name
        agents.sort(key=lambda x: x["name"])

    except Exception as e:
        return jsonify({"agents": [], "error": str(e)}), 200

    response = jsonify({"agents": agents})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response


class TemplateChangeHandler(FileSystemEventHandler):
    """Watch for changes in template and static files and emit live-reload events"""

    def on_modified(self, event):
        if not event.is_directory:
            templates_dir = os.path.abspath("templates")
            static_dir = os.path.abspath("static")

            # Check if the file is in templates or static directory
            if event.src_path.startswith(templates_dir) or event.src_path.startswith(
                static_dir
            ):
                # print(f"File change detected: {event.src_path}")
                # Emit reload event to all connected clients
                try:
                    socketio.emit("reload", namespace="/live-reload")
                    # print("Sent reload signal to clients")
                except Exception as e:
                    print(f"Error sending reload signal: {str(e)}")


class TemplateWatcher:
    """Simple watcher to log template and static file changes for debugging."""

    def __init__(self):
        self.observer = None
        self.template_dir = os.path.abspath("templates")
        self.static_dir = os.path.abspath("static")

    def start(self):
        try:
            self.observer = Observer()
            event_handler = TemplateChangeHandler()
            # Watch both templates and static directories
            self.observer.schedule(event_handler, self.template_dir, recursive=True)
            self.observer.schedule(event_handler, self.static_dir, recursive=True)
            self.observer.start()
            # Template watcher started silently
        except Exception as e:
            print(f"Error starting file watcher: {str(e)}")

    def stop(self):
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join()
            except Exception as e:
                print(f"Error stopping template watcher: {str(e)}")


class APIServer:
    def __init__(
        self,
        generator,
        host="localhost",
        port=2100,
        tokenizer=None,
        integration_loader=None,
        param_stats=None,
        seed=None,
        truncated_hash=None,
        full_hash=None,
    ):
        # Initialize APIServer with parameter statistics
        self.generator = generator
        self.server_thread = None
        self.server = None
        self.started = Event()
        self.shutdown_event = Event()
        self.host = host
        while self._is_port_in_use(port):
            port += 1
        self.port = port
        self.parent_pid = os.getppid()
        self.seed = seed
        self.tokenizer = tokenizer
        self.integration_loader = integration_loader
        self.param_stats = param_stats if param_stats else {}
        self.truncated_hash = truncated_hash
        self.full_hash = full_hash
        self.template_watcher = TemplateWatcher()

        # Initialize terminal WebSocket namespace if available
        if terminal_available:
            # Register socketio for dashboard streaming
            interface.register_socketio(socketio)

            # Create a simplified terminal namespace for dashboard streaming
            class TerminalNamespace(Namespace):
                """WebSocket namespace for terminal/dashboard interaction."""

                def on_connect(self):
                    """Send current dashboard state on connect."""
                    dashboard = interface.get_active_dashboard("main")
                    if dashboard and hasattr(dashboard, "_streamer"):
                        # Get the latest rendered frame
                        frame = dashboard._streamer.get_current_frame()
                        if frame:
                            rendered = (
                                dashboard._streamer.renderer.render_frame_for_web(frame)
                            )
                            emit(
                                "dashboard_frame",
                                {
                                    "frame": rendered["text"],
                                    "metadata": {
                                        "width": rendered["width"],
                                        "height": rendered["height"],
                                        "scale_factor": rendered["scale_factor"],
                                    },
                                },
                            )

                def on_disconnect(self):
                    """Handle client disconnection."""
                    pass

                def on_start_capture(self, data):
                    """Connect to existing dashboard."""
                    dashboard = interface.get_active_dashboard("main")
                    if dashboard and hasattr(dashboard, "_streamer"):
                        dashboard._streamer.start()
                        emit("capture_started", {"status": "connected_to_existing"})

                        # Send current frame if available
                        frame = dashboard._streamer.get_current_frame()
                        if frame:
                            rendered = (
                                dashboard._streamer.renderer.render_frame_for_web(frame)
                            )
                            emit(
                                "dashboard_frame",
                                {
                                    "frame": rendered["text"],
                                    "metadata": {
                                        "width": rendered["width"],
                                        "height": rendered["height"],
                                        "scale_factor": rendered["scale_factor"],
                                    },
                                },
                            )
                    else:
                        emit("capture_started", {"status": "no_dashboard_found"})

                def on_stop_capture(self):
                    """Stop dashboard streaming."""
                    dashboard = interface.get_active_dashboard("main")
                    if dashboard and hasattr(dashboard, "_streamer"):
                        dashboard._streamer.stop()
                    emit("capture_stopped", {"status": "ok"})

            # Register the terminal namespace
            socketio.on_namespace(TerminalNamespace("/terminal"))

    def update_param_stats(self, param_stats):
        """Update the parameter statistics after optimizer creation"""
        self.param_stats = param_stats
        # Update parameter statistics silently

    def _is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _monitor_parent(self):
        """Monitor thread that checks if parent process is alive"""
        while not self.shutdown_event.is_set():
            try:
                # Check if parent process still exists
                os.kill(self.parent_pid, 0)
                time.sleep(1)  # Check every second
            except OSError:  # Parent process no longer exists
                # Parent process died, shutting down API server
                pass
                self.stop()
                break

    def start(self):
        if self.server_thread is not None:
            return  # Already started

        # Start the template watcher for logging
        self.template_watcher.start()

        # Start the server thread
        self.server_thread = Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Start the monitor thread
        self.monitor_thread = Thread(target=self._monitor_parent)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.started.wait(timeout=5)
        if not self.started.is_set():
            raise RuntimeError("Server failed to start within the timeout period")
        print(f"[API] Server started at http://{self.get_api_addr()}/")

    def stop(self):
        self.shutdown_event.set()  # Signal monitor thread to stop

        # Stop the template watcher
        if hasattr(self, "template_watcher"):
            self.template_watcher.stop()

        # Shut down SocketIO
        try:
            # Shutting down API server
            socketio.stop()
        except Exception as e:
            print(f"Error stopping server: {str(e)}")

        if self.server_thread:
            self.server_thread.join(timeout=5)
            self.server_thread = None

    def _run_server(self):
        # Start API server with parameter statistics

        # Apply any WSGI middleware registered by integrations (must be before starting server)
        apply_wsgi_middleware()

        with app.app_context():
            app.config["generator"] = self.generator
            app.config["tokenizer"] = self.tokenizer
            app.config["integration_loader"] = self.integration_loader
            app.config["seed"] = self.seed
            app.config["truncated_hash"] = self.truncated_hash
            app.config["full_hash"] = self.full_hash
            # Store param_stats if available
            if hasattr(self, "param_stats") and self.param_stats:
                app.config["param_stats"] = self.param_stats
                # Stored parameter statistics in app config
            else:
                app.config["param_stats"] = {}
                # No parameter statistics available

            # Register integration middleware FIRST
            if self.integration_loader:
                for middleware_func in self.integration_loader.get_request_middleware():
                    # Wrap the middleware to handle both request and response phases
                    def create_wrapper(func):
                        def request_wrapper(req, resp):
                            return func(req, resp)
                        return request_wrapper
                    
                    wrapper = create_wrapper(middleware_func)
                    register_request_middleware(wrapper)
                    register_response_middleware(wrapper)


            # Signal that the server will start (AFTER hooks are registered)
            self.started.set()

            # Use SocketIO's built-in server instead of werkzeug's
            socketio.run(
                app,
                host="0.0.0.0",  # Bind to all interfaces
                port=self.port,
                debug=True,
                use_reloader=False,  # We handle reloading ourselves
                allow_unsafe_werkzeug=True,
            )

    def get_api_addr(self):
        return f"{self.host}:{self.port}"


@app.route("/", methods=["GET"])
def home():
    response = make_response(render_template("index.html"))
    # Add CSP header that allows scripts from same origin and CDNs
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https:; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com https://*.ngrok-free.app https://*.ngrok.io; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' wss: ws: https: http:;"
    )
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


@app.route("/favicon.ico")
def favicon():
    """Serve favicon.ico from static folder if it exists"""
    import os

    if os.path.exists(os.path.join(app.static_folder, "favicon.ico")):
        return send_from_directory(app.static_folder, "favicon.ico")
    else:
        # Return empty 204 No Content if favicon doesn't exist
        return "", 204


@app.route("/input/", methods=["GET", "POST", "OPTIONS"])
@app.route("/input", methods=["GET", "POST", "OPTIONS"])
def generate():
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    try:
        # # Log detailed request information
        # print(f"Request received:")
        # print(f"- From: {request.remote_addr}")
        # print(f"- Method: {request.method}")
        # print(f"- Headers: {dict(request.headers)}")
        # print(f"- Origin: {request.headers.get('Origin', 'None')}")

        # Increase server timeout (30 minutes)
        request.environ.get("werkzeug.server.shutdown_timeout", 30 * 60)

        kwargs = request.get_json()
        logging.info("Received a valid request via REST.")

        prompt = kwargs.get("prompt")
        messages = kwargs.get("messages")

        if (prompt is not None) and (messages is not None):
            response = jsonify(
                {"error": "Please provide either 'prompt' or 'messages', not both."}
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400

        if (prompt is None) and (messages is None):
            response = jsonify({"error": "Please provide 'prompt' or 'messages'."})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400

        is_chatml = False
        if messages is not None:
            # Format messages using the tokenizer's chat template
            try:
                tokenizer = app.config["tokenizer"]
                prompt = format_messages_to_chatml(messages, tokenizer)
                is_chatml = True
                kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    tokenizer.sep_token_id,
                ]
                kwargs["skip_special_tokens"] = False
                del kwargs["messages"]
            except ValueError as ve:
                logging.error(ve)
                response = jsonify({"error": str(ve)})
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response, 400

        generator = app.config["generator"]
        request_id = generator.request_generation(prompt, kwargs)
        while True:
            result = generator.get_result(request_id)
            if result is not None:
                output = result
                break
            time.sleep(0.1)

        if not output:
            raise Exception("Failed to generate an output from this API.")

        if is_chatml:
            # Extract only the assistant's reply using the tokenizer
            assistant_reply = extract_assistant_reply(output, app.config["tokenizer"])
            response = {"response": assistant_reply}
        else:
            # Return the full output as before
            response = {"response": output}

    except Exception as e:
        logging.error(e)
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 400

    logging.info("Successfully responded to REST request.")
    final_response = jsonify(response)
    final_response.headers.add("Access-Control-Allow-Origin", "*")
    return final_response, 200


@app.before_request
def apply_request_middleware():
    """Apply request middleware from loaded modules."""
    integration_loader = app.config.get("integration_loader")
    if integration_loader:
        # Get all middleware functions
        middleware_funcs = integration_loader.get_request_middleware()
        for middleware_func in middleware_funcs:
            try:
                # Call middleware with request object
                # Middleware can modify request headers or add response headers
                middleware_func(request)
            except Exception as e:
                print(f"Error in request middleware: {e}")


@app.after_request
def apply_response_middleware(response):
    """Apply response middleware from loaded modules."""
    integration_loader = app.config.get("integration_loader")
    if integration_loader:
        # Get all middleware functions
        middleware_funcs = integration_loader.get_request_middleware()
        for middleware_func in middleware_funcs:
            try:
                # Call middleware with request and response objects
                # Middleware can modify response headers
                middleware_func(request, response)
            except Exception as e:
                # If middleware only takes request, that's fine
                pass
    return response


@app.route("/<path:filename>", methods=["GET", "POST", "OPTIONS", "HEAD"])
def serve_static(filename):

    # If this is a POST to input, redirect to the actual input handler
    if filename in ["input", "input/"] and request.method in ["POST", "OPTIONS"]:
        print(f"[DEBUG] Redirecting {request.method} request to /input/ handler")
        return generate()

    # Otherwise, serve static files only for GET/HEAD
    if request.method not in ["GET", "HEAD"]:
        from flask import abort

        abort(405)

    return send_from_directory(app.static_folder, filename)


def format_messages_to_chatml(messages, tokenizer):
    """Format a list of message objects using the tokenizer's chat template."""
    # Validate message roles
    for message in messages:
        role = message.get("role", "").strip()
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Invalid role: {role}")

    # Apply the chat template and add assistant generation prompt
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_assistant_reply(generated_text, tokenizer):
    """Extract the assistant's reply from the generated text."""
    # Find the pattern that marks the start of the assistant's response
    assistant_start = f"{tokenizer.bos_token}assistant"

    # Find the last occurrence of the assistant's start token
    start_index = generated_text.rfind(assistant_start)
    if start_index == -1:
        # If the start token is not found, return the whole text
        return generated_text.strip()

    # Skip past the start token AND the "assistant" role identifier
    start_index += len(assistant_start)

    # Find the end token after the start_index - check for both EOS and SEP tokens
    eos_index = generated_text.find(tokenizer.eos_token, start_index)
    sep_index = generated_text.find(tokenizer.sep_token, start_index)

    # Use whichever comes first (and exists)
    end_index = -1
    if eos_index != -1 and sep_index != -1:
        end_index = min(eos_index, sep_index)
    elif eos_index != -1:
        end_index = eos_index
    elif sep_index != -1:
        end_index = sep_index

    if end_index == -1:
        # If no end token is found, return everything after the start token
        assistant_reply = generated_text[start_index:].strip()
    else:
        assistant_reply = generated_text[start_index:end_index].strip()

    # Remove any remaining BOS token that might appear at the beginning of the response
    if assistant_reply.startswith(tokenizer.bos_token):
        assistant_reply = assistant_reply[len(tokenizer.bos_token) :].strip()

    return assistant_reply
