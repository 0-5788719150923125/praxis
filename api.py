import asyncio
import logging
from threading import Thread, Event
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

app = Flask(__name__)


class APIServer:
    def __init__(self, generator):
        self.generator = generator
        self.server_thread = None
        self.server = None
        self.started = Event()
        self.port = 5000  # Default Flask port

    def start(self):
        self.server_thread = Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.started.wait(timeout=5)  # Wait up to 5 seconds for the server to start
        if not self.started.is_set():
            raise RuntimeError("Server failed to start within the timeout period")
        print(f"API Server started in the background. URL: {self.get_url()}")

    def _run_server(self):
        with app.app_context():
            self.server = make_server("0.0.0.0", self.port, app)
            self.started.set()  # Signal that the server has started
            self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()
        if self.server_thread:
            self.server_thread.join()

    def get_url(self):
        return f"http://localhost:{self.port}"

    @app.route("/generate/", methods=["GET", "POST"])
    def generate():
        try:
            kwargs = request.get_json()
            logging.error("Received a valid request via REST.")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            prompt = kwargs.get("prompt", "")
            output = loop.run_until_complete(
                app.config["generator"].generate(prompt, kwargs)
            )

            if not output:
                raise Exception("Failed to generate an output from this API.")
        except Exception as e:
            logging.error(e)
            return jsonify(str(e)), 400
        print("Successfully responded to REST request.")
        return jsonify({"response": output}), 200
