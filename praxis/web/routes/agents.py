"""Agent discovery routes."""

import concurrent.futures
import json
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


@agents_bp.route("/api/agents", methods=["GET"])
def get_agents():
    """Get git remotes as peer agents with their online/offline status."""
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

            configured_host = current_app.config.get("configured_host")
            configured_port = current_app.config.get("configured_port")

            if ngrok_url and ngrok_secret:
                # Ngrok is active - use the protected URL
                git_url = f"{ngrok_url}/{ngrok_secret}/praxis"
            elif configured_host and configured_host != "localhost":
                # If host includes a scheme (e.g., https://host.example.com), use it directly
                if configured_host.startswith(("https://", "http://")):
                    git_url = f"{configured_host}/praxis"
                else:
                    git_url = f"http://{configured_host}:{configured_port}/praxis"
            else:
                # No ngrok or explicit host - use local URL
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
                commit_timestamp = (
                    int(timestamp_result.stdout.strip())
                    if timestamp_result.returncode == 0
                    else None
                )
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
                                    "commit_timestamp": spec_data.get(
                                        "commit_timestamp"
                                    ),
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
                                agent["commit_timestamp"] = int(
                                    timestamp_result.stdout.strip()
                                )
                        except:
                            pass

                    if url.startswith(("http://", "https://")):
                        parsed = urllib.parse.urlparse(url)
                        # Known git hosting services are always archived
                        if (
                            "github.com" in parsed.netloc
                            or "gitlab.com" in parsed.netloc
                            or "bitbucket.org" in parsed.netloc
                        ):
                            agent["status"] = "archived"
                        else:
                            # Ping the remote to check if it's a live Praxis instance
                            try:
                                ping_url = f"{parsed.scheme}://{parsed.netloc}/api/ping"
                                req = urllib.request.Request(
                                    ping_url,
                                    headers={"User-Agent": "Praxis-Agent-Check"},
                                )
                                with urllib.request.urlopen(req, timeout=5) as response:
                                    if response.status == 200:
                                        agent["status"] = "online"
                                    else:
                                        agent["status"] = "offline"
                            except Exception:
                                agent["status"] = "offline"
                    else:
                        # SSH/git protocol URLs can't be pinged
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

    # Backend-hosted swarm experts (orchestration sidecar). These join the same
    # Hangar/Wire list as the git-remote peers and the browser ships. The
    # frontend applies the type's naming convention (arc-N) - names label the
    # agent *type* (the unified tiny transformer), not unique identities, so they
    # may repeat across backend/browser. An expert that has taken local steps on
    # real batches is a passive OBSERVE-r (blue); one that hasn't is IDLE. (They
    # never contribute back to the model yet - see the RemoteLayer stub.)
    try:
        from praxis.orchestration import status as pool_status

        for exp in pool_status.experts():
            steps = int(exp.get("steps", 0) or 0)
            agents.append(
                {
                    "uid": exp.get("uid"),
                    "url": "sidecar://localhost",
                    "status": "observe" if steps > 0 else "idle",
                    "type": "expert",
                    "kind": "backend",
                    "rank": exp.get("rank"),
                    "passes": exp.get("passes"),
                    "steps": steps,
                }
            )
    except Exception:
        pass

    response = jsonify({"agents": agents})
    return response


# Joined-expert lifecycle: browser AGREEs add experts tied to a browser session
# id and stamped with a last-seen time. They are NOT permanent - a sweep prunes
# any whose session hasn't pinged within this TTL, so closing/refreshing the tab
# eventually reclaims them instead of leaking experts into the pool forever.
_JOIN_TTL_SECONDS = 30.0


def _prune_joined(pool) -> int:
    """Drop joined experts whose owning session has gone silent past the TTL."""
    import time

    now = time.monotonic()
    removed = 0
    for e in list(pool.experts):
        seen = getattr(e, "_join_last_seen", None)
        if seen is not None and (now - seen) > _JOIN_TTL_SECONDS:
            pool.remove(e.uid)
            removed += 1
    return removed


@agents_bp.route("/api/swarm/join", methods=["POST"])
def swarm_join():
    """Add experts to the live pool for a browser session (an AGREE joins here).

    Idempotent per session: re-AGREEing from the same tab refreshes the session's
    heartbeat and tops its experts up to ``count`` rather than stacking new ones,
    so refreshes don't inflate the pool. Experts are TTL-pruned (see
    ``_prune_joined``). No-op (404) when no pool is active.
    """
    import time

    from torch import nn

    from praxis.orchestration import LocalExpert
    from praxis.orchestration import status as pool_status

    pool = pool_status.get_pool()
    if pool is None:
        return jsonify({"joined": 0, "error": "no active pool"}), 404

    body = request.get_json(silent=True) or {}
    try:
        count = max(1, min(64, int(body.get("count", 1))))
    except Exception:
        count = 1
    session = str(body.get("session", "") or "anon")

    # Reclaim stale experts first so a long-running pool doesn't drift upward.
    _prune_joined(pool)

    now = time.monotonic()
    dim = int(getattr(pool, "_join_dim", 14))
    vocab = int(getattr(pool, "_join_vocab", 16))
    # Experts already owned by this session - refresh their heartbeat.
    mine = [e for e in pool.experts if getattr(e, "_join_session", None) == session]
    for e in mine:
        e._join_last_seen = now
    # Top up to the requested count for this session (don't stack on refresh).
    for i in range(max(0, count - len(mine))):
        block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        exp = LocalExpert(
            f"joined-{session[:6]}-{len(mine) + i}",
            block,
            hidden_size=dim,
            vocab_size=vocab,
        )
        exp._join_session = session
        exp._join_last_seen = now
        pool.add(exp)
    cap = pool.capacity()  # republish so dashboards update immediately
    return jsonify({"session": session, "experts_total": cap["experts_total"]})


@agents_bp.route("/api/swarm/heartbeat", methods=["POST"])
def swarm_heartbeat():
    """Keep a browser session's joined experts alive (called periodically by the
    tab). Also runs the prune sweep so silent sessions are reclaimed."""
    import time

    from praxis.orchestration import status as pool_status

    pool = pool_status.get_pool()
    if pool is None:
        return jsonify({"ok": False}), 404
    body = request.get_json(silent=True) or {}
    session = str(body.get("session", "") or "anon")
    now = time.monotonic()
    for e in pool.experts:
        if getattr(e, "_join_session", None) == session:
            e._join_last_seen = now
    _prune_joined(pool)
    pool.capacity()
    return jsonify({"ok": True})


@agents_bp.route("/api/swarm/batch", methods=["GET"])
def swarm_batch():
    """The latest real training batch (token-id rows over the swarm's tiny vocab)
    for a browser agent to train on. Bounded depth 1: always the freshest batch,
    so a slow browser never falls behind a growing queue - it just skips ahead.
    Clients pass ``?since=<seq>`` to avoid re-training the same batch; returns
    ``{batch: null}`` when nothing newer is available."""
    from praxis.orchestration import status as pool_status

    batch = pool_status.latest_batch()
    if batch is None:
        return jsonify({"batch": None})
    try:
        since = int(request.args.get("since", -1))
    except Exception:
        since = -1
    if batch["seq"] <= since:
        return jsonify({"batch": None})  # nothing newer; client stays put
    return jsonify({"batch": batch})
