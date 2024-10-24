import asyncio
import inspect
import logging
import time
from threading import Event, Thread

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.serving import make_server

app = Flask(__name__)
app.static_folder = "templates"


class APIServer:
    def __init__(self, generator, host="localhost", port=5000):
        self.generator = generator
        self.server_thread = None
        self.server = None
        self.started = Event()
        self.host = host
        self.port = port

    def start(self):
        self.server_thread = Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.started.wait(timeout=5)  # Wait up to 5 seconds for the server to start
        if not self.started.is_set():
            raise RuntimeError("Server failed to start within the timeout period")
        print(f"API server started at: {self.get_api_addr()}")

    def _run_server(self):
        with app.app_context():
            app.config["generator"] = self.generator
            self.server = make_server("0.0.0.0", self.port, app)
            self.started.set()  # Signal that the server has started
            self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join()

    def get_api_addr(self):
        return f"{self.host}:{self.port}"


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    if filename != "input/":  # Exclude the API route
        return send_from_directory(app.static_folder, filename)


@app.route("/input/", methods=["GET", "POST"])
def generate():
    try:
        kwargs = request.get_json()
        logging.info("Received a valid request via REST.")

        prompt = kwargs.get("prompt", "")
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
    except Exception as e:
        logging.error(e)
        return jsonify(str(e)), 400
    logging.info("Successfully responded to REST request.")
    return jsonify({"response": output}), 200
