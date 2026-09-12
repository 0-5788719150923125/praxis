"""Generation endpoint routes."""

import logging
import time

from flask import Blueprint, current_app, jsonify, request

from praxis.inference.prompts import parse_generation_kwargs

from ..utils import generate_from_messages, serving_defaults
from ..websocket import stream_callbacks

generation_bp = Blueprint("generation", __name__)
api_logger = logging.getLogger("praxis.web")


def _tool_counts(names):
    """``[{"name", "count"}, ...]`` in first-use order.

    Order matters more than it looks: the chips render in it, so a turn that
    searched and then read a file reads left to right the way it happened.
    """
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "count": count} for name, count in counts.items()]


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

        # Incremental publication, when the client asked for it. `stream_id` is
        # minted by the client and echoed on every frame, so the deltas travel
        # on the socket it already has open while THIS response stays exactly
        # what it was. A client that sends no id gets no streamer built at all.
        on_text, on_reset, emit_tool = stream_callbacks(data.get("stream_id"))

        # Tool use is tallied HERE rather than left to the socket, and the
        # tally is installed whether or not the client is streaming. It is not
        # a preview that the final answer supersedes: the reply extractor
        # strips the call/result exchange, so this response is otherwise the
        # one place a tool run leaves no trace. A client whose socket was down
        # still gets the counts; a client watching the socket gets them live
        # AND gets this to settle on.
        tools_used = []

        def on_tool(name):
            tools_used.append(name)
            if emit_tool is not None:
                emit_tool(name)

        # Decode knobs, in precedence order: the route's historical defaults,
        # then the run's --generation-kwargs, then whatever this request sent.
        # The client's own Settings form is an OVERRIDE of the run's values, and
        # it expresses that by sending them here - so the rule is just "last
        # writer wins" and there is nothing to reconcile server-side.
        defaults = serving_defaults()
        run_kwargs = dict(defaults.get("generation_kwargs") or {})
        try:
            request_kwargs = parse_generation_kwargs(data.get("generation_kwargs"))
        except ValueError as e:
            return jsonify({"response": "", "error": str(e)}), 400
        run_kwargs.update(request_kwargs)

        # Legacy top-level fields from older clients, applied at the same
        # precedence as an explicit override.
        for key, payload_key in (
            ("max_new_tokens", "max_new_tokens"),
            ("temperature", "temperature"),
            ("repetition_penalty", "repetition_penalty"),
            ("do_sample", "do_sample"),
            ("use_cache", "use_cache"),
            ("timeout", "timeout"),
        ):
            if payload_key in data:
                run_kwargs[key] = data[payload_key]

        # Use unified generation function
        assistant_reply = generate_from_messages(
            messages=messages,
            generator=generator,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.4,
            repetition_penalty=1.15,
            do_sample=True,
            timeout=60.0,
            on_text=on_text,
            on_reset=on_reset,
            on_tool=on_tool,
            system_prompt=defaults.get("system_prompt"),
            developer_prompt=defaults.get("developer_prompt"),
            generation_kwargs=run_kwargs,
        )

        # A baby/untrained model may produce nothing or gibberish - never 500 over
        # it. Return whatever we got (possibly empty); the UI handles it. No
        # sanitization, just whatever the model said.
        return (
            jsonify(
                {"response": assistant_reply or "", "tools": _tool_counts(tools_used)}
            ),
            200,
        )

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
