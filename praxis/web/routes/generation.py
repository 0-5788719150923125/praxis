"""Generation endpoint routes."""

import logging
import time

from flask import Blueprint, current_app, jsonify, request

from ..utils import generate_from_messages

generation_bp = Blueprint("generation", __name__)
api_logger = logging.getLogger("praxis.web")


@generation_bp.route("/messages/", methods=["POST"])
@generation_bp.route("/messages", methods=["POST"])
def generate_messages():
    """Handle message-based generation."""
    try:
        data = request.get_json()
        messages = data.get("messages", [])

        if not messages:
            response = jsonify({"error": "Please provide 'messages' for generation."})
            return response, 400

        # Get generator and tokenizer
        if "api_server" in current_app.config:
            generator = current_app.config["api_server"].generator
        elif "generator" in current_app.config:
            generator = current_app.config["generator"]
        else:
            error_response = jsonify(
                {
                    "error": "Generator not initialized yet. Please wait for training to start."
                }
            )
            return error_response, 503

        tokenizer = current_app.config.get("tokenizer")
        if not tokenizer:
            error_response = jsonify({"error": "Tokenizer not available"})
            return error_response, 503

        # Use unified generation function
        assistant_reply = generate_from_messages(
            messages=messages,
            generator=generator,
            tokenizer=tokenizer,
            max_new_tokens=data.get("max_new_tokens", 256),
            temperature=data.get("temperature", 0.4),
            repetition_penalty=data.get("repetition_penalty", 1.15),
            do_sample=data.get("do_sample", True),
            timeout=float(data.get("timeout", 60.0)),
        )

        # A baby/untrained model may produce nothing or gibberish - never 500 over
        # it. Return whatever we got (possibly empty); the UI handles it. No
        # sanitization, just whatever the model said.
        return jsonify({"response": assistant_reply or ""}), 200

    except Exception as e:
        api_logger.error(f"Error in /messages endpoint: {e}")
        # Resilience for baby models: surface a failure as an empty turn rather
        # than a 500 that breaks the chat/loop UI.
        return jsonify({"response": "", "error": str(e)}), 200


@generation_bp.route("/input/", methods=["GET", "POST"])
@generation_bp.route("/input", methods=["GET", "POST"])
def generate():
    """Handle string-based prompt generation."""
    try:
        kwargs = request.get_json()
        prompt = kwargs.get("prompt")

        if prompt is None:
            response = jsonify(
                {"error": "Please provide 'prompt' for string-based generation."}
            )
            return response, 400

        if "messages" in kwargs:
            response = jsonify(
                {"error": "Use /messages endpoint for message-based generation."}
            )
            return response, 400

        # Get generator
        if "api_server" in current_app.config:
            generator = current_app.config["api_server"].generator
        elif "generator" in current_app.config:
            generator = current_app.config["generator"]
        else:
            error_response = jsonify(
                {
                    "error": "Generator not initialized yet. Please wait for training to start."
                }
            )
            return error_response, 503

        request_id = generator.request_generation(prompt, kwargs)
        while True:
            result = generator.get_result(request_id)
            if result is not None:
                output = result
                break
            time.sleep(0.1)

        if not output:
            raise Exception("Failed to generate an output from this API.")

        response = {"response": output}

    except Exception as e:
        api_logger.error(e)
        error_response = jsonify({"error": str(e)})
        return error_response, 400

    final_response = jsonify(response)
    return final_response, 200
