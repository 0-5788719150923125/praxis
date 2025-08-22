import glob
import inspect
import logging
import os
import socket
import sys
import time
from threading import Event, Thread

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from werkzeug.serving import make_server

# Import terminal streaming components
try:
    import interface  # Import interface for dashboard streaming functions
    from flask_socketio import Namespace, emit
    terminal_available = True
except ImportError as e:
    terminal_available = False
    print(f"Terminal streaming not available: {e}")

logger = logging.getLogger("werkzeug")
logger.setLevel(logging.ERROR)

app = Flask(__name__)
app.static_folder = "templates"
app.debug = True  # Enable debug mode for development

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up SocketIO for live reload
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Add ngrok bypass header to all responses
@app.after_request
def add_ngrok_header(response):
    """Add header to bypass ngrok browser warning"""
    response.headers['ngrok-skip-browser-warning'] = 'true'
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
        from cli import get_cli_args, _compute_args_hash
        import json
        import hashlib
        
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
        
        # Compute the full hash from current args
        # This gives us the actual full hash
        import sys
        full_hash = _compute_args_hash(sys.argv[1:])
        truncated_hash = full_hash[:9] if full_hash else None
        
        # Get the model architecture string
        model_arch = None
        try:
            generator = app.config.get("generator")
            if generator and hasattr(generator, 'model'):
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
        
        # Calculate parameter stats directly here
        param_stats = {}
        try:
            generator = app.config.get("generator")
            if generator and hasattr(generator, 'model'):
                model = generator.model
                # Count the parameters
                total_params = sum(p.numel() for p in model.parameters())
                
                # Get actual config values from the model
                config = model.config if hasattr(model, 'config') else None
                if config:
                    # Get actual values from config
                    batch_size = args_dict.get('batch_size', 1)
                    block_size = getattr(config, 'max_position_embeddings', getattr(config, 'block_size', 512))
                    hidden_size = getattr(config, 'hidden_size', 768)
                    
                    # In Praxis, depth is the number of forward passes through experts
                    # num_experts is the pool of available experts to choose from
                    # The actual number of layers processed is just depth
                    depth = getattr(config, 'depth', 3)
                    num_experts = getattr(config, 'num_experts', 3)
                    
                    # Simple activation estimate: batch_size * seq_len * hidden_size * depth
                    activation_params = batch_size * block_size * hidden_size * depth
                    
                    param_stats = {
                        "total_params": total_params,
                        "activation_params": activation_params,
                        "config": {
                            "batch_size": batch_size,
                            "block_size": block_size,
                            "hidden_size": hidden_size,
                            "depth": depth,
                            "num_experts": num_experts
                        }
                    }
                else:
                    param_stats = {"total_params": total_params}
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
        
        # Return clean data structure
        spec = {
            "truncated_hash": truncated_hash,
            "full_hash": full_hash,
            "args": args_dict,
            "model_architecture": model_arch,
            "param_stats": param_stats,
            "timestamp": timestamp,
            "command": command
        }
        
        response = jsonify(spec)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
        
    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500


class TemplateChangeHandler(FileSystemEventHandler):
    """Watch for changes in template files and emit live-reload events"""

    def on_modified(self, event):
        if not event.is_directory and event.src_path.startswith(
            os.path.abspath("templates")
        ):
            # print(f"Template change detected: {event.src_path}")
            # Emit reload event to all connected clients
            try:
                socketio.emit("reload", namespace="/live-reload")
                # print("Sent reload signal to clients")
            except Exception as e:
                print(f"Error sending reload signal: {str(e)}")


class TemplateWatcher:
    """Simple watcher to log template changes for debugging."""

    def __init__(self):
        self.observer = None
        self.template_dir = os.path.abspath("templates")

    def start(self):
        try:
            self.observer = Observer()
            event_handler = TemplateChangeHandler()
            self.observer.schedule(event_handler, self.template_dir, recursive=True)
            self.observer.start()
            print(f"Watching template directory: {self.template_dir}")
        except Exception as e:
            print(f"Error starting template watcher: {str(e)}")

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
        module_loader=None,
        param_stats=None,
    ):
        print(f"[DEBUG] APIServer.__init__ called with param_stats: {param_stats is not None}")
        if param_stats:
            print(f"[DEBUG] APIServer.__init__ param_stats keys: {list(param_stats.keys())}")
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
        self.tokenizer = tokenizer
        self.module_loader = module_loader
        self.param_stats = param_stats if param_stats else {}
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
                    if dashboard and hasattr(dashboard, '_streamer'):
                        # Get the latest rendered frame
                        frame = dashboard._streamer.get_current_frame()
                        if frame:
                            rendered = dashboard._streamer.renderer.render_frame_for_web(frame)
                            emit('dashboard_frame', {
                                'frame': rendered['text'],
                                'metadata': {
                                    'width': rendered['width'],
                                    'height': rendered['height'],
                                    'scale_factor': rendered['scale_factor']
                                }
                            })
                
                def on_disconnect(self):
                    """Handle client disconnection."""
                    pass
                
                def on_start_capture(self, data):
                    """Connect to existing dashboard."""
                    dashboard = interface.get_active_dashboard("main")
                    if dashboard and hasattr(dashboard, '_streamer'):
                        dashboard._streamer.start()
                        emit('capture_started', {'status': 'connected_to_existing'})
                        
                        # Send current frame if available
                        frame = dashboard._streamer.get_current_frame()
                        if frame:
                            rendered = dashboard._streamer.renderer.render_frame_for_web(frame)
                            emit('dashboard_frame', {
                                'frame': rendered['text'],
                                'metadata': {
                                    'width': rendered['width'],
                                    'height': rendered['height'],
                                    'scale_factor': rendered['scale_factor']
                                }
                            })
                    else:
                        emit('capture_started', {'status': 'no_dashboard_found'})
                
                def on_stop_capture(self):
                    """Stop dashboard streaming."""
                    dashboard = interface.get_active_dashboard("main")
                    if dashboard and hasattr(dashboard, '_streamer'):
                        dashboard._streamer.stop()
                    emit('capture_stopped', {'status': 'ok'})
            
            # Register the terminal namespace
            socketio.on_namespace(TerminalNamespace('/terminal'))
    
    def update_param_stats(self, param_stats):
        """Update the parameter statistics after optimizer creation"""
        self.param_stats = param_stats
        print(f"[DEBUG] APIServer.update_param_stats called with stats: {bool(param_stats)}")

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
                print("Parent process died, shutting down API server...")
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
        print(f"API server started at: {self.get_api_addr()}")
        print(f"Flask debug mode: {app.debug} (auto-reload enabled)")
        print(f"Visit http://{self.get_api_addr()}/ in your browser")

    def stop(self):
        self.shutdown_event.set()  # Signal monitor thread to stop

        # Stop the template watcher
        if hasattr(self, "template_watcher"):
            self.template_watcher.stop()

        # Shut down SocketIO
        try:
            print("Shutting down API server...")
            socketio.stop()
        except Exception as e:
            print(f"Error stopping server: {str(e)}")

        if self.server_thread:
            self.server_thread.join(timeout=5)
            self.server_thread = None

    def _run_server(self):
        print(f"[DEBUG] _run_server starting, self.param_stats exists: {bool(self.param_stats)}")
        if self.param_stats:
            print(f"[DEBUG] _run_server self.param_stats keys: {list(self.param_stats.keys())}")
        
        with app.app_context():
            app.config["generator"] = self.generator
            app.config["tokenizer"] = self.tokenizer
            app.config["module_loader"] = self.module_loader
            # Store param_stats if available
            if hasattr(self, 'param_stats') and self.param_stats:
                app.config["param_stats"] = self.param_stats
                print(f"[DEBUG] API Server: Stored param_stats in app.config with {len(self.param_stats)} keys")
                print(f"[DEBUG] API Server: param_stats keys: {list(self.param_stats.keys())}")
            else:
                app.config["param_stats"] = {}
                print(f"[DEBUG] API Server: No param_stats found, storing empty dict")

            # Signal that the server will start
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
    return render_template("index.html")


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
    module_loader = app.config.get("module_loader")
    if module_loader:
        # Get all middleware functions
        middleware_funcs = module_loader.get_request_middleware()
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
    module_loader = app.config.get("module_loader")
    if module_loader:
        # Get all middleware functions
        middleware_funcs = module_loader.get_request_middleware()
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
        if role not in {"system", "user", "assistant"}:
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
