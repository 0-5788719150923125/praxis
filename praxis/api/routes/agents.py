"""Agent discovery routes."""

import concurrent.futures
import os
import subprocess
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, current_app, jsonify, request

from praxis.utils import mask_git_url

from ..config import (
    GIT_COMMAND_TIMEOUT,
    PORT_CHECK_TIMEOUT,
    PORT_RANGE_END,
    PORT_RANGE_START,
)

agents_bp = Blueprint("agents", __name__)


def handle_cors_preflight():
    """Handle CORS preflight requests."""
    response = jsonify({"status": "ok"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response


@agents_bp.route("/api/agents", methods=["GET", "OPTIONS"])
def get_agents():
    """Get git remotes as peer agents with their online/offline status."""
    if request.method == "OPTIONS":
        return handle_cors_preflight()

    agents = []
    self_instances = []  # Collect all self instances first

    # Collect the current instance information
    try:
        # Get current port
        current_port = int(request.environ.get("SERVER_PORT", 2100))

        # Get current instance details
        try:
            # Get our git URL - prioritize ngrok if active
            ngrok_url = current_app.config.get("ngrok_url")
            ngrok_secret = current_app.config.get("ngrok_secret")

            if ngrok_url and ngrok_secret:
                # Ngrok is active - use the protected URL
                git_url = f"{ngrok_url}/{ngrok_secret}/praxis"
            else:
                # No ngrok - use local URL
                git_url = f"http://localhost:{current_port}/praxis"

            # Get git hash and timestamp
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                )
                full_hash = (
                    result.stdout.strip() if result.returncode == 0 else "unknown"
                )
                short_hash = full_hash[:7] if len(full_hash) >= 7 else full_hash

                # Get commit timestamp (Unix epoch)
                timestamp_result = subprocess.run(
                    ["git", "show", "-s", "--format=%ct", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                )
                commit_timestamp = int(timestamp_result.stdout.strip()) if timestamp_result.returncode == 0 else None
            except:
                full_hash = "unknown"
                short_hash = "unknown"
                commit_timestamp = None

            # Add current instance to the list of self instances
            self_instances.append(
                {
                    "port": current_port,
                    "url": git_url,
                    "masked_url": mask_git_url(git_url),
                    "status": "online",
                    "commit_hash": full_hash,
                    "short_hash": short_hash,
                    "commit_timestamp": commit_timestamp,
                }
            )
        except Exception as e:
            print(f"[WARNING] Failed to add current instance: {e}")

        # Check if this is in the standard port range
        is_standard_port = PORT_RANGE_START <= current_port < PORT_RANGE_END

        if is_standard_port:
            # Scan ports for local instances
            local_instances = []

            def check_local_port(port):
                try:
                    import json

                    spec_url = f"http://localhost:{port}/api/spec"
                    req = urllib.request.Request(spec_url)
                    with urllib.request.urlopen(
                        req, timeout=PORT_CHECK_TIMEOUT
                    ) as response:
                        if response.status == 200:
                            spec_data = json.loads(response.read())
                            if spec_data.get("git_url"):
                                return {
                                    "port": port,
                                    "git_url": spec_data["git_url"],
                                    "masked_url": spec_data.get(
                                        "masked_git_url",
                                        mask_git_url(spec_data["git_url"]),
                                    ),
                                    "full_hash": spec_data.get("full_hash"),
                                    "truncated_hash": spec_data.get("truncated_hash"),
                                    "commit_timestamp": spec_data.get("commit_timestamp"),
                                }
                except:
                    pass
                return None

            # Check ports concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for port in range(PORT_RANGE_START, PORT_RANGE_END):
                    future = executor.submit(check_local_port, port)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=1)
                        if result:
                            local_instances.append(result)
                    except:
                        pass

            # Add other local instances to self_instances list
            for instance in local_instances:
                if instance and instance["port"] != current_port:
                    self_instances.append(
                        {
                            "port": instance["port"],
                            "url": instance["git_url"],
                            "masked_url": instance["masked_url"],
                            "status": "online",
                            "commit_hash": instance["full_hash"],
                            "short_hash": instance["truncated_hash"],
                            "commit_timestamp": instance.get("commit_timestamp"),
                        }
                    )
    except Exception as e:
        print(f"[DEBUG] Error in self agent detection: {e}")

    # Sort all self instances by port and assign consistent names
    self_instances.sort(key=lambda x: x["port"])
    for idx, instance in enumerate(self_instances):
        name = f"self-{idx + 1}"
        agents.append(
            {
                "name": name,
                "url": instance["url"],
                "masked_url": instance["masked_url"],
                "status": instance["status"],
                "commit_hash": instance["commit_hash"],
                "short_hash": instance["short_hash"],
                "commit_timestamp": instance.get("commit_timestamp"),
            }
        )

    try:
        # Get git remotes
        result = subprocess.run(
            ["git", "remote", "-v"], capture_output=True, text=True, cwd=os.getcwd()
        )

        if result.returncode != 0:
            return jsonify({"agents": [], "error": "Failed to get git remotes"}), 200

        # Parse remotes
        remotes = {}
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    url = parts[1]
                    if name not in remotes:
                        remotes[name] = url

        def check_remote_status(name, url):
            """Check if a remote is accessible and get its latest commit."""
            agent = {
                "name": name,
                "url": url,
                "masked_url": mask_git_url(url),
                "status": "offline",
                "commit_hash": None,
                "short_hash": None,
                "commit_timestamp": None,
            }

            # Try to check if the remote is accessible using git ls-remote
            try:
                check_result = subprocess.run(
                    ["git", "ls-remote", url, "HEAD"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    cwd=os.getcwd(),
                    timeout=GIT_COMMAND_TIMEOUT,
                    text=True,
                )
                if check_result.returncode == 0:
                    # Parse the commit hash from ls-remote output
                    output = check_result.stdout.strip()
                    if output:
                        commit_hash = output.split("\t")[0]
                        agent["commit_hash"] = commit_hash
                        agent["short_hash"] = commit_hash[:7] if commit_hash else None

                        # Get commit timestamp for this hash
                        try:
                            timestamp_result = subprocess.run(
                                ["git", "show", "-s", "--format=%ct", commit_hash],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                cwd=os.getcwd(),
                                timeout=2,
                                text=True,
                            )
                            if timestamp_result.returncode == 0:
                                agent["commit_timestamp"] = int(timestamp_result.stdout.strip())
                        except:
                            pass

                    # Check if this is a Praxis instance
                    is_praxis = False
                    if url.startswith(("http://", "https://")):
                        base_url = url.replace(".git", "").rstrip("/")
                        # If it's a known git hosting service, mark as archived
                        if (
                            "github.com" in base_url
                            or "gitlab.com" in base_url
                            or "bitbucket.org" in base_url
                        ):
                            agent["status"] = "archived"
                        else:
                            # Try to check for Praxis API endpoint
                            try:
                                api_url = f"{base_url}/api/agents"
                                req = urllib.request.Request(
                                    api_url,
                                    headers={"User-Agent": "Praxis-Agent-Check"},
                                )
                                with urllib.request.urlopen(req, timeout=2) as response:
                                    if response.status == 200:
                                        is_praxis = True
                                        agent["status"] = "online"
                                    else:
                                        agent["status"] = "archived"
                            except:
                                agent["status"] = "archived"
                    else:
                        # For SSH/git protocol URLs, mark as archived
                        agent["status"] = "archived"

            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

            return agent

        # Check status of each remote in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for name, url in remotes.items():
                future = executor.submit(check_remote_status, name, url)
                futures.append(future)

            # Collect results with timeout
            for future in futures:
                try:
                    agent = future.result(timeout=3)
                    agents.append(agent)
                except concurrent.futures.TimeoutError:
                    agents.append(
                        {
                            "name": "unknown",
                            "url": "unknown",
                            "status": "offline",
                            "type": "unknown",
                        }
                    )

        # Sort agents by name
        agents.sort(key=lambda x: x["name"])

    except Exception as e:
        return jsonify({"agents": [], "error": str(e)}), 200

    response = jsonify({"agents": agents})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response
