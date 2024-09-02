import asyncio
import logging

from flask import Flask, jsonify, request

app = Flask(__name__)


class APIServer:
    def __init__(self, model):
        self.model = model
        app.run(debug=False, host="0.0.0.0")

    @app.route("/generate/", methods=["GET", "POST"])
    def generate(self):
        try:
            kwargs = request.get_json()
            logging.error("Received a valid request via REST.")
            output = asyncio.run(self.model.generate(**kwargs))
            if not output:
                raise Exception("Failed to generate an output from this API.")
        except Exception as e:
            logging.error(e)
            return jsonify(e), 400
        print("Successfully responded to REST request.")
        return jsonify({"response": output}), 200
