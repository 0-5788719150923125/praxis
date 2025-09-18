"""Git HTTP backend routes."""

import os
import subprocess
from flask import Blueprint, request, Response, stream_with_context

from ..config import GIT_ALLOWED_SERVICES, GIT_READ_ONLY_SERVICE

git_bp = Blueprint("git", __name__)


# Git HTTP backend routes - support multiple URL patterns
@git_bp.route("/praxis.git/<path:git_path>", methods=["GET", "POST"])
@git_bp.route("/praxis.git", methods=["GET", "POST"], defaults={"git_path": ""})
@git_bp.route("/praxis/<path:git_path>", methods=["GET", "POST"])
@git_bp.route("/praxis", methods=["GET", "POST"], defaults={"git_path": ""})
@git_bp.route("/info/refs", methods=["GET"])
@git_bp.route("/git-upload-pack", methods=["POST"])
@git_bp.route("/src/<path:git_path>", methods=["GET", "POST"])  # Backward compatibility
def git_http_backend(git_path=None):
    """
    Simple Git HTTP backend for read-only access to the repository.
    Supports git clone and fetch operations.

    Accessible via:
    - With .git suffix: git clone https://domain.com/praxis.git
    - Without suffix: git clone https://domain.com/praxis
    - Legacy /src path: git clone https://domain.com/src
    """
    # Parse the service from the path
    service = request.args.get("service")

    # Handle root-level git operations (when git_path is None)
    if git_path is None:
        if request.path == "/info/refs":
            git_path = "info/refs"
        elif request.path == "/git-upload-pack":
            git_path = "git-upload-pack"

    # Handle info/refs request (git discovery)
    if (git_path == "info/refs" or git_path == "") and service:
        if not service.startswith("git-"):
            return "Invalid service", 400

        service_name = service.replace("git-", "")
        if service not in GIT_ALLOWED_SERVICES:
            return "Service not allowed", 403

        # Only allow upload-pack for read-only access
        if service != GIT_READ_ONLY_SERVICE:
            return "Only read access is allowed", 403

        try:
            # Run git command to get refs
            cmd = ["git", "upload-pack", "--stateless-rpc", "--advertise-refs", "."]
            result = subprocess.run(cmd, capture_output=True, cwd=os.getcwd())

            # Format response for git HTTP protocol
            response_data = f"001e# service={service}\n0000" + result.stdout.decode(
                "latin-1"
            )

            return Response(
                response_data,
                content_type=f"application/x-{service}-advertisement",
                headers={
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                },
            )
        except Exception as e:
            return f"Git error: {str(e)}", 500

    # Handle git-upload-pack request (actual clone/fetch)
    elif git_path == "git-upload-pack" and request.method == "POST":
        try:
            # Run git upload-pack with the request data
            cmd = ["git", "upload-pack", "--stateless-rpc", "."]
            result = subprocess.run(
                cmd, input=request.data, capture_output=True, cwd=os.getcwd()
            )

            return Response(
                result.stdout,
                content_type="application/x-git-upload-pack-result",
                headers={
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                },
            )
        except Exception as e:
            return f"Git error: {str(e)}", 500

    # Return 404 for other paths
    return "Not found", 404