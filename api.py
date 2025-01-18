import asyncio
import inspect
import logging
import os
import socket
import sys
import time
from threading import Event, Thread

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.serving import make_server

logger = logging.getLogger("werkzeug")
logger.setLevel(logging.ERROR)

app = Flask(__name__)
app.static_folder = "templates"


class APIServer:
    def __init__(
        self,
        generator,
        host="localhost",
        port=2100,
        bos_token="<bos>",
        eos_token="<eos>",
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
        self.bos_token = bos_token
        self.eos_token = eos_token

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

    def stop(self):
        self.shutdown_event.set()  # Signal monitor thread to stop
        if self.server:
            print("Shutting down API server...")
            self.server.shutdown()
            self.server = None
        if self.server_thread:
            self.server_thread.join(timeout=5)
            self.server_thread = None

    def _run_server(self):
        with app.app_context():
            app.config["generator"] = self.generator
            app.config["bos_token"] = self.bos_token
            app.config["eos_token"] = self.eos_token
            self.server = make_server("0.0.0.0", self.port, app)
            self.started.set()  # Signal that the server has started
            self.server.serve_forever()

    def get_api_addr(self):
        return f"{self.host}:{self.port}"


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    if filename != "input/":  # Exclude the API route
        return send_from_directory(app.static_folder, filename)


def format_messages_to_chatml(messages, bos_token, eos_token):
    """Format a list of message objects into a ChatML-formatted string."""
    formatted = ""
    for message in messages:
        role = message.get("role", "").strip()
        content = message.get("content", "").strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Invalid role: {role}")
        formatted += f"{bos_token}{role}\n{content}\n{eos_token}\n"
    # Ensure the prompt ends with the assistant's role
    if not formatted.strip().endswith("<|im_start|> assistant"):
        formatted += f"{bos_token}assistant\n"
    return formatted


def extract_assistant_reply(generated_text, bos_token, eos_token):
    """Extract the assistant's reply from the generated text."""
    # Tokens used in ChatML
    begin_token = f"{bos_token}assistant\n"
    # Find the last occurrence of the assistant's start token
    start_index = generated_text.rfind(begin_token)
    if start_index == -1:
        # If the start token is not found, return the whole text
        return generated_text.strip()
    else:
        start_index += len(begin_token)
    # Find the end token after the start_index
    end_index = generated_text.find(eos_token, start_index)
    if end_index == -1:
        # If the end token is not found, return everything after the start token
        assistant_reply = generated_text[start_index:].strip()
    else:
        assistant_reply = generated_text[start_index:end_index].strip()
    return assistant_reply


@app.route("/input/", methods=["GET", "POST"])
def generate():
    try:
        kwargs = request.get_json()
        logging.info("Received a valid request via REST.")

        prompt = kwargs.get("prompt")
        messages = kwargs.get("messages")

        if (prompt is not None) and (messages is not None):
            return (
                jsonify(
                    {"error": "Please provide either 'prompt' or 'messages', not both."}
                ),
                400,
            )
        if (prompt is None) and (messages is None):
            return jsonify({"error": "Please provide 'prompt' or 'messages'."}), 400

        is_chatml = False
        if messages is not None:
            # Format messages into ChatML format
            try:
                prompt = format_messages_to_chatml(
                    messages, app.config["bos_token"], app.config["eos_token"]
                )
                is_chatml = True
                kwargs["stop_strings"] = [app.config["eos_token"]]
                kwargs["skip_special_tokens"] = False
                del kwargs["messages"]
            except ValueError as ve:
                logging.error(ve)
                return jsonify({"error": str(ve)}), 400

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
            # Extract only the assistant's reply
            assistant_reply = extract_assistant_reply(
                output, app.config["bos_token"], app.config["eos_token"]
            )
            response = {"response": assistant_reply}
        else:
            # Return the full output as before
            response = {"response": output}

    except Exception as e:
        logging.error(e)
        return jsonify({"error": str(e)}), 400
    logging.info("Successfully responded to REST request.")
    return jsonify(response), 200
