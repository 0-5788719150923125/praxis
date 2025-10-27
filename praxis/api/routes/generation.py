"""Generation endpoint routes."""

import logging
import time

from flask import Blueprint, current_app, jsonify, request

from ..utils import extract_assistant_reply

generation_bp = Blueprint("generation", __name__)
api_logger = logging.getLogger("praxis.api")


@generation_bp.route("/messages/", methods=["POST", "OPTIONS"])
@generation_bp.route("/messages", methods=["POST", "OPTIONS"])
def generate_messages():
    """Handle message-based generation."""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    try:
        data = request.get_json()
        messages = data.get("messages", [])

        if not messages:
            response = jsonify({"error": "Please provide 'messages' for generation."})
            response.headers.add("Access-Control-Allow-Origin", "*")
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
            error_response.headers.add("Access-Control-Allow-Origin", "*")
            return error_response, 503

        tokenizer = current_app.config.get("tokenizer")
        if not tokenizer:
            error_response = jsonify({"error": "Tokenizer not available"})
            error_response.headers.add("Access-Control-Allow-Origin", "*")
            return error_response, 503

        # Format messages using chat template
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            api_logger.error(f"Error formatting messages: {e}")
            formatted_prompt = "\n".join(
                [
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in messages
                ]
            )

        # Extract generation parameters
        kwargs = {
            "max_new_tokens": data.get("max_new_tokens", 256),
            "temperature": data.get("temperature", 0.4),
            "repetition_penalty": data.get("repetition_penalty", 1.15),
            "do_sample": data.get("do_sample", True),
            "use_cache": data.get("use_cache", False),
            "skip_special_tokens": data.get("skip_special_tokens", False),
        }

        # Request generation
        request_id = generator.request_generation(formatted_prompt, kwargs)

        # Wait for result
        while True:
            result = generator.get_result(request_id)
            if result is not None:
                output = result
                break
            time.sleep(0.1)

        if not output:
            raise Exception("Failed to generate an output from this API.")

        # Extract assistant's reply
        assistant_reply = extract_assistant_reply(output, tokenizer)

        response = jsonify({"response": assistant_reply})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 200

    except Exception as e:
        api_logger.error(f"Error in /messages endpoint: {e}")
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 500


@generation_bp.route("/input/", methods=["GET", "POST", "OPTIONS"])
@generation_bp.route("/input", methods=["GET", "POST", "OPTIONS"])
def generate():
    """Handle string-based prompt generation."""
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    try:
        kwargs = request.get_json()
        prompt = kwargs.get("prompt")

        if prompt is None:
            response = jsonify(
                {"error": "Please provide 'prompt' for string-based generation."}
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400

        if "messages" in kwargs:
            response = jsonify(
                {"error": "Use /messages endpoint for message-based generation."}
            )
            response.headers.add("Access-Control-Allow-Origin", "*")
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
            error_response.headers.add("Access-Control-Allow-Origin", "*")
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
        error_response.headers.add("Access-Control-Allow-Origin", "*")
        return error_response, 400

    final_response = jsonify(response)
    final_response.headers.add("Access-Control-Allow-Origin", "*")
    return final_response, 200
