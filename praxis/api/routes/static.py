"""Static file serving routes."""

import os
from flask import Blueprint, send_from_directory, current_app, request, abort

static_bp = Blueprint("static_files", __name__)


@static_bp.route("/favicon.ico")
def favicon():
    """Serve favicon.ico from static folder if it exists."""
    if os.path.exists(os.path.join(current_app.static_folder, "favicon.ico")):
        return send_from_directory(current_app.static_folder, "favicon.ico")
    else:
        return "", 204


@static_bp.route("/static/<path:filename>")
def serve_static_files(filename):
    """Serve static files with proper headers for CORS."""
    response = send_from_directory(current_app.static_folder, filename)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@static_bp.route("/<path:filename>", methods=["GET", "POST", "OPTIONS", "HEAD"])
def serve_static(filename):
    """Catch-all route for serving static files and handling special cases."""
    # Import generation routes to handle special cases
    from .generation import generate, generate_messages

    # If this is a POST to input, redirect to the actual input handler
    if filename in ["input", "input/"] and request.method in ["POST", "OPTIONS"]:
        return generate()

    # If this is a POST to messages, redirect to the actual messages handler
    if filename in ["messages", "messages/"] and request.method in ["POST", "OPTIONS"]:
        return generate_messages()

    # Otherwise, serve static files only for GET/HEAD
    if request.method not in ["GET", "HEAD"]:
        abort(405)

    return send_from_directory(current_app.static_folder, filename)
