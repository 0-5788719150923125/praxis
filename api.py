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
from flask_socketio import SocketIO
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from werkzeug.serving import make_server

logger = logging.getLogger("werkzeug")
logger.setLevel(logging.ERROR)

app = Flask(__name__)
app.static_folder = "templates"
app.debug = True  # Enable debug mode for development

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up SocketIO for live reload
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

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
    ):
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
        self.template_watcher = TemplateWatcher()

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
        with app.app_context():
            app.config["generator"] = self.generator
            app.config["tokenizer"] = self.tokenizer
            app.config["module_loader"] = self.module_loader

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
    # Debug log
    print(f"[DEBUG] serve_static called with: {filename}, method: {request.method}")

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
