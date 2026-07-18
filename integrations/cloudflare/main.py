"""Cloudflare Pages integration: publish a static dashboard snapshot.

Running the dashboard as an always-on public server is expensive. This
integration freezes the dashboard into a directory of static files and pushes
it to Cloudflare Pages (free tier), so the showcase stays online after the
training server shuts down. We keep the charts, dynamics, architecture,
evolution, and a browsable KB; we lose the live/interactive surfaces (chat,
swarm, generation, the metrics websocket), which are gated behind an offline
banner and greyed-out controls.

The export runs *in-process* against the live Flask app's test client, so the
model-dependent endpoints (activation curves, head snapshots, dynamics) dump
with real data straight from the already-warm snapshot store - no model
plumbing, no re-implemented route logic. Every read-only route is reused as-is;
only a small client-side shim (static_mode.js) is injected into the exported
copy to redirect ``/api/*`` fetches onto the dumped files and to stub the
websocket.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from praxis.integrations.base import BaseIntegration, IntegrationSpec

# --- Export configuration ---------------------------------------------------

# Where the static site is assembled before upload. Under build/ (gitignored).
DEFAULT_OUT_DIR = "build/snapshot"

# Frozen data lives here, mirrored by static_mode.js. Kept distinct from
# ``/api`` (which does not physically exist on Pages) and ``/static``.
DATA_DIR = "data"

# KB types to include: hand-authored docs + source code only. Crawled pages and
# external links are deliberately excluded to keep the upload small.
KB_TYPES = "doc,note,code"

# Business cards: a fixed seed (client-side randomness would need a live route),
# pre-rendered across the side/theme combinations the frontend requests.
CARD_SEED = 42
CARD_SIDES = ("front", "back")
CARD_THEMES = ("light", "dark")
CARD_HUE = "161"  # the frontend's default --accent-hue

OFFLINE_MESSAGE = (
    "You are viewing an offline snapshot of the Praxis dashboard. "
    "Live features (chat, generation, the swarm, and the metrics stream) are "
    "disabled - this page is a periodic capture, not a running server."
)

# Read-only endpoints to dump verbatim: (request path, output file, kind).
# kind is "binary" for files served as-is; everything else is written bytewise
# too, the label is just documentation of the content type.
DUMP_ENDPOINTS: List[Tuple[str, str, str]] = [
    ("/api/ping", "ping.json", "json"),
    ("/api/runs", "runs.json", "json"),
    ("/api/config", "config.yaml", "text"),
    ("/api/metrics?since=0&limit=1000&downsample=lttb", "metrics.json", "json"),
    ("/api/data-metrics?since=0&limit=1000&downsample=lttb", "data-metrics.json", "json"),
    ("/api/dynamics?since=0&limit=1000", "dynamics.json", "json"),
    ("/api/head_snapshots", "head_snapshots.json", "json"),
    ("/api/activation_curves", "activation_curves.json", "json"),
    ("/api/evolution", "evolution.json", "json"),
    ("/api/spider", "spider.json", "json"),
    ("/api/agents", "agents.json", "json"),
    ("/api/print/energy", "print_energy.json", "json"),
    ("/api/print/pending", "print_pending.json", "json"),
    ("/api/loop/energy", "loop_energy.json", "json"),
]

_HERE = Path(__file__).resolve().parent
_SHIM_SRC = _HERE / "static_mode.js"

# The socket.io CDN tag in templates/index.html; our shim is injected right
# after it so window.io is overridden before main.js (a deferred module) runs.
_SOCKET_TAG = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/'
    '4.8.1/socket.io.js"></script>'
)
_SHIM_TAG = '<script src="/static/js/static_mode.js"></script>'


def _log(msg: str) -> None:
    print(f"[cloudflare] {msg}")


def _load_env() -> None:
    """Load KEY=VALUE lines from a repo-root .env into the environment, so the
    CLOUDFLARE_* creds don't have to be exported. Stdlib-only (no python-dotenv
    dependency); already-set environment variables win over .env."""
    path = Path(".env")
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:
        _log(f"could not read .env: {exc}")


# --- The exporter -----------------------------------------------------------


def export_snapshot(
    app: Any,
    out_dir: str = DEFAULT_OUT_DIR,
    live_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Assemble the static site under ``out_dir`` and return its path.

    ``app`` is the live Flask app; we drive its test client so every dumped
    payload is exactly what a real client would receive. ``live_metrics`` is an
    optional ``LiveMetrics.snapshot()`` dict, frozen in for the Terminal tab and
    the notification bell.
    """
    out = Path(out_dir)
    data = out / DATA_DIR
    # Fresh each time: a stale file from a previous run must never linger.
    if out.exists():
        shutil.rmtree(out)
    (data / "kb_item").mkdir(parents=True, exist_ok=True)

    client = app.test_client()

    _export_static_assets(out)
    _export_index(client, out)
    _export_endpoints(client, data)
    _export_spec(client, data, app)
    _export_cards(client, data, out)
    _export_kb(client, data)
    _export_paper(client, out)
    _export_metrics_live(data, live_metrics)

    _log(f"exported snapshot to {out}")
    return out


def _export_static_assets(out: Path) -> None:
    """Copy the built frontend (JS + styles.css) and drop in the shim."""
    from praxis import web

    static_src = Path(web.__file__).resolve().parent / "static"
    shutil.copytree(static_src, out / "static", dirs_exist_ok=True)
    shutil.copy2(_SHIM_SRC, out / "static" / "js" / "static_mode.js")


def _export_index(client: Any, out: Path) -> None:
    """Render ``/`` and inject the static-mode shim after the socket.io tag."""
    resp = client.get("/")
    html = resp.get_data(as_text=True)
    if _SHIM_TAG not in html:
        if _SOCKET_TAG in html:
            html = html.replace(_SOCKET_TAG, _SOCKET_TAG + "\n    " + _SHIM_TAG)
        else:
            # Fallback: inject before the closing body so it still loads.
            html = html.replace("</body>", f"    {_SHIM_TAG}\n</body>")
    (out / "index.html").write_text(html, encoding="utf-8")


def _export_endpoints(client: Any, data: Path) -> None:
    for path, out_name, _kind in DUMP_ENDPOINTS:
        try:
            resp = client.get(path)
            (data / out_name).write_bytes(resp.get_data())
        except Exception as exc:  # one bad endpoint never kills the export
            _log(f"warning: {path} dump failed: {exc}")


def _export_spec(client: Any, data: Path, app: Any) -> None:
    """Dump ``/api/spec`` for the current run. The live route builds the payload
    from the running model; if that path errors (e.g. no live generator) we fall
    back to the run's on-disk ``spec.json``, which is itself a valid response."""
    payload = None
    try:
        resp = client.get("/api/spec")
        parsed = resp.get_json(silent=True)
        if isinstance(parsed, dict) and "error" not in parsed and "args" in parsed:
            payload = resp.get_data()
    except Exception as exc:
        _log(f"warning: live /api/spec failed: {exc}")

    if payload is None:
        run_hash = app.config.get("truncated_hash")
        disk = Path("build/runs") / str(run_hash) / "spec.json"
        if disk.is_file():
            payload = disk.read_bytes()
            _log("spec: used on-disk spec.json")
        else:
            payload = json.dumps({"status": "no_data"}).encode("utf-8")
            _log("spec: no live payload and no on-disk spec.json")

    (data / "spec.json").write_bytes(payload)


def _export_cards(client: Any, data: Path, out: Path) -> None:
    default_svg = None
    for side in CARD_SIDES:
        for theme in CARD_THEMES:
            path = (
                f"/api/card/preview.svg?seed={CARD_SEED}"
                f"&side={side}&theme={theme}&hue={CARD_HUE}"
            )
            try:
                resp = client.get(path)
                if resp.status_code == 200:
                    body = resp.get_data()
                    (data / f"card_{side}_{theme}.svg").write_bytes(body)
                    if side == "front" and theme == "light":
                        default_svg = body
            except Exception as exc:
                _log(f"warning: card {side}/{theme} dump failed: {exc}")

    # Physical fallback at the real request path: an <img> begins loading its
    # original src during HTML parse, a beat before the shim's observer can
    # rewrite it. Static hosts ignore the query string, so this one file answers
    # that initial load (with the default card); the observer then corrects the
    # side/theme. Without it, every card view logs a stray 404.
    if default_svg is not None:
        card_path = out / "api" / "card"
        card_path.mkdir(parents=True, exist_ok=True)
        (card_path / "preview.svg").write_bytes(default_svg)


def _export_kb(client: Any, data: Path) -> None:
    """Dump the doc+code feed and each item's body; write an id->file index."""
    try:
        resp = client.get(f"/api/kb/search?types={KB_TYPES}")
        feed = resp.get_json() or {"status": "ok", "hits": []}
    except Exception as exc:
        _log(f"warning: KB feed dump failed: {exc}")
        feed = {"status": "ok", "hits": []}

    (data / "kb_feed.json").write_bytes(json.dumps(feed).encode("utf-8"))

    index: Dict[str, str] = {}
    for i, hit in enumerate(feed.get("hits", [])):
        item_id = hit.get("id")
        if not item_id:
            continue
        rel = f"kb_item/{i}.json"
        try:
            resp = client.get("/api/kb/item?id=" + urllib.parse.quote(item_id, safe=""))
            (data / rel).write_bytes(resp.get_data())
            index[item_id] = rel
        except Exception as exc:
            _log(f"warning: KB item {item_id} dump failed: {exc}")
    (data / "kb_item_index.json").write_bytes(json.dumps(index).encode("utf-8"))
    _log(f"KB: {len(index)} items ({KB_TYPES})")


def _export_paper(client: Any, out: Path) -> None:
    try:
        resp = client.get("/api/paper.pdf")
        if resp.status_code == 200:
            (out / "paper.pdf").write_bytes(resp.get_data())
    except Exception as exc:
        _log(f"warning: paper.pdf dump failed: {exc}")


def _export_metrics_live(data: Path, live_metrics: Optional[Dict[str, Any]]) -> None:
    """Freeze the last live-metrics snapshot with the offline banner appended as
    the newest notification event, so the bell surfaces the warning."""
    snap = dict(live_metrics or {})
    events = list(snap.get("events") or [])
    next_id = max((e.get("id", 0) for e in events), default=0) + 1
    events.append(
        {
            "id": next_id,
            "message": OFFLINE_MESSAGE,
            "level": "warning",
            "stage": snap.get("stage"),
            "hours_elapsed": snap.get("hours_elapsed", 0),
        }
    )
    snap["events"] = events
    (data / "metrics_live.json").write_bytes(json.dumps(snap).encode("utf-8"))


# --- Deploy -----------------------------------------------------------------


# Cached wrangler command prefix, resolved (and installed) at most once.
_wrangler_cmd: Optional[List[str]] = None


def _apt_install_node() -> bool:
    """Install Node.js + npm via the system package manager. The training image
    is Ubuntu (apt), where the ``nodejs`` package is new enough for wrangler.
    Returns True if npm is available afterward."""
    if not shutil.which("apt-get"):
        _log(
            "no npm and no apt-get available - install Node.js >=18 to enable "
            "deploys (the static site is still exported for manual upload)."
        )
        return False
    _log("Node.js not found; installing via apt (nodejs npm)...")
    try:
        subprocess.run(
            ["apt-get", "update", "-qq"],
            check=True, capture_output=True, text=True, timeout=300,
        )
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "nodejs", "npm"],
            check=True, capture_output=True, text=True, timeout=900,
        )
    except Exception as exc:
        _log(
            f"apt install of nodejs failed ({exc}). In the training container this "
            "usually means it's running non-root - Node.js + wrangler are baked into "
            "the Docker image instead; rebuild the image (re-run ./launch) to pick them up."
        )
        return False
    return shutil.which("npm") is not None


def _ensure_wrangler() -> Optional[List[str]]:
    """Return a command prefix that runs wrangler, installing it (and Node.js if
    needed) on first use - the way other integrations auto-install their deps.
    Gated behind ``--publish-snapshot``, so lean runs never pull node. Returns
    None if wrangler can't be made available."""
    global _wrangler_cmd
    if _wrangler_cmd is not None:
        return _wrangler_cmd
    if shutil.which("wrangler"):
        _wrangler_cmd = ["wrangler"]
        return _wrangler_cmd

    if not shutil.which("npm") and not shutil.which("npx"):
        if not _apt_install_node():
            return None

    if shutil.which("npm") and not shutil.which("wrangler"):
        _log("installing wrangler (npm install -g wrangler)...")
        try:
            subprocess.run(
                ["npm", "install", "-g", "wrangler"],
                check=True, capture_output=True, text=True, timeout=900,
            )
        except Exception as exc:
            _log(f"npm install -g wrangler failed: {exc}")

    if shutil.which("wrangler"):
        _wrangler_cmd = ["wrangler"]
    elif shutil.which("npx"):
        # npx fetches wrangler on demand - slower per call, but works.
        _wrangler_cmd = ["npx", "--yes", "wrangler"]
    else:
        _log("could not make wrangler available after install attempts")
        return None
    return _wrangler_cmd


def _branch_alias(run_hash: Optional[str]) -> str:
    """Sanitize a run hash into a Cloudflare Pages branch alias. Same hash always
    maps to the same ``<alias>.<project>.pages.dev`` URL, so re-publishing a run
    overwrites its own site instead of spawning a new one. Cloudflare lowercases
    the alias, maps non-alphanumerics to dashes, and truncates to 28 chars."""
    raw = (run_hash or "snapshot").lower()
    alias = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")[:28].strip("-")
    return alias or "snapshot"


_CF_API = "https://api.cloudflare.com/client/v4"


def _cf(method: str, path: str, token: str, body: Optional[dict] = None) -> dict:
    """Call the Cloudflare REST API (stdlib urllib, no dependency). Returns the
    parsed JSON. Raises RuntimeError carrying the API's own error text so the
    bootstrap can fail loudly and specifically."""
    import urllib.error
    import urllib.request

    if body is not None:
        data = json.dumps(body).encode("utf-8")
    else:
        data = None if method == "GET" else b""  # PATCH/POST tolerate empty body
    req = urllib.request.Request(
        _CF_API + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            errs = "; ".join(e.get("message", "") for e in payload.get("errors") or [])
        except Exception:
            errs = exc.reason
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {errs or '?'}") from None


def _project_subdomain(account: str, project: str, token: str) -> Optional[str]:
    """The project's canonical ``<name>-xxxx.pages.dev`` (NOT necessarily
    ``<name>.pages.dev`` - pages.dev names are globally unique, so a taken name
    gets a random suffix). None if the project doesn't exist yet."""
    try:
        body = _cf("GET", f"/accounts/{account}/pages/projects/{project}", token)
    except RuntimeError:
        return None
    return (body.get("result") or {}).get("subdomain")


def _ensure_project_api(account: str, project: str, prod_branch: str, token: str) -> Optional[str]:
    """Create the Pages project if missing (REST) and return its canonical
    subdomain. Idempotent."""
    sub = _project_subdomain(account, project, token)
    if sub:
        return sub
    _cf(
        "POST",
        f"/accounts/{account}/pages/projects",
        token,
        {"name": project, "production_branch": prod_branch},
    )
    _log(f"created Pages project '{project}'")
    return _project_subdomain(account, project, token)


def _find_zone(account: str, domain: str, token: str) -> Optional[tuple]:
    """(zone_id, zone_name) of the zone whose name is the longest suffix of
    ``domain`` (arc.src.eco -> the src.eco zone). None if none is visible."""
    body = _cf("GET", "/zones?per_page=50", token)
    best = None
    for z in body.get("result") or []:
        name = z.get("name", "")
        if name and (domain == name or domain.endswith("." + name)):
            if best is None or len(name) > len(best[1]):
                best = (z["id"], name)
    return best


def _ensure_dns_cname(account: str, domain: str, subdomain: str, token: str) -> bool:
    """Upsert a proxied ``domain -> subdomain`` CNAME in the owning zone.
    Cloudflare does NOT reliably auto-create this for Pages custom domains, so we
    do it ourselves, pointing at the project's *canonical* subdomain."""
    zone = _find_zone(account, domain, token)
    if zone is None:
        _log(f"no Cloudflare zone owns {domain}; add a proxied CNAME -> {subdomain} manually")
        return False
    zid, _zname = zone
    recs = _cf("GET", f"/zones/{zid}/dns_records?name={domain}", token).get("result") or []
    for r in recs:
        if r.get("type") == "CNAME":
            if r.get("content") == subdomain and r.get("proxied"):
                return True
            _cf(
                "PATCH",
                f"/zones/{zid}/dns_records/{r['id']}",
                token,
                {"type": "CNAME", "name": domain, "content": subdomain, "proxied": True},
            )
            _log(f"updated CNAME {domain} -> {subdomain}")
            return True
    _cf(
        "POST",
        f"/zones/{zid}/dns_records",
        token,
        {
            "type": "CNAME",
            "name": domain,
            "content": subdomain,
            "proxied": True,
            "comment": "praxis dashboard snapshot",
        },
    )
    _log(f"created CNAME {domain} -> {subdomain} (proxied)")
    return True


def _ensure_domain_attached(account: str, project: str, domain: str, token: str) -> None:
    api = f"/accounts/{account}/pages/projects/{project}/domains"
    names = [d.get("name") for d in (_cf("GET", api, token).get("result") or [])]
    if domain not in names:
        _cf("POST", api, token, {"name": domain})
        _log(f"attached custom domain {domain} to '{project}'")


def _revalidate_domain(account: str, project: str, domain: str, token: str) -> None:
    """Nudge Cloudflare to re-check the domain now that the CNAME exists (the
    first validation fails if it ran before DNS). Harmless once active."""
    try:
        _cf("PATCH", f"/accounts/{account}/pages/projects/{project}/domains/{domain}", token)
    except RuntimeError:
        pass


def ensure_infrastructure(project: str) -> Optional[str]:
    """Bootstrap-time, REST-only provisioning: the Pages project and (if a
    showcase domain is configured) its custom-domain attachment + DNS CNAME +
    re-validation. Fast and fail-loud, so a bad token/config/zone surfaces in
    seconds at startup instead of after torch.compile. Returns the project's
    canonical subdomain, or None on failure (publishing then no-ops)."""
    token = os.getenv("CLOUDFLARE_API_TOKEN")
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account:
        _log("CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID not set; publishing disabled")
        return None

    prod_branch = (os.getenv("CLOUDFLARE_PAGES_PRODUCTION_BRANCH") or "main").strip()
    domain = (os.getenv("CLOUDFLARE_PAGES_DOMAIN") or "").strip()
    try:
        subdomain = _ensure_project_api(account, project, prod_branch, token)
        if domain and subdomain:
            _ensure_domain_attached(account, project, domain, token)
            _ensure_dns_cname(account, domain, subdomain, token)
            _revalidate_domain(account, project, domain, token)
        where = f"https://{domain}" if domain else f"https://{subdomain}"
        _log(f"infrastructure ready: {where} (project '{project}' -> {subdomain})")
        return subdomain
    except Exception as exc:
        _log(f"ERROR: Cloudflare bootstrap failed - {exc}")
        return None


def deploy_content(
    out_dir: str, project: str, run_hash: Optional[str], subdomain: Optional[str]
) -> bool:
    """Upload the static site to an already-provisioned project via wrangler.
    Project/domain/DNS are handled at bootstrap by ``ensure_infrastructure``;
    this step only pushes content. With a domain configured it deploys to the
    production branch (so the custom domain serves it); otherwise to a per-run
    branch alias (``<hash>.<subdomain>``)."""
    token = os.getenv("CLOUDFLARE_API_TOKEN")
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account:
        _log(f"no Cloudflare creds; site exported to {out_dir} for manual upload")
        return False

    base = _ensure_wrangler()
    if base is None:
        _log(f"wrangler unavailable; site exported to {out_dir} for manual upload")
        return False

    domain = (os.getenv("CLOUDFLARE_PAGES_DOMAIN") or "").strip()
    prod_branch = (os.getenv("CLOUDFLARE_PAGES_PRODUCTION_BRANCH") or "main").strip()
    if domain:
        branch = prod_branch
        target = f"https://{domain}"
    else:
        branch = _branch_alias(run_hash)
        target = f"https://{branch}.{subdomain}" if subdomain else f"branch '{branch}'"

    env = {**os.environ, "CLOUDFLARE_API_TOKEN": token, "CLOUDFLARE_ACCOUNT_ID": account}
    cmd = base + [
        "pages",
        "deploy",
        str(out_dir),
        "--project-name",
        project,
        "--branch",
        branch,
        "--commit-dirty=true",
    ]
    _log(f"uploading snapshot to '{project}' (branch '{branch}' -> {target})...")
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except Exception as exc:
        _log(f"upload failed to launch: {exc}")
        return False

    if result.returncode != 0:
        _log(f"upload failed (exit {result.returncode}): {result.stderr.strip()[:500]}")
        return False

    tail = [ln for ln in result.stdout.splitlines() if ln.strip()]
    _log("upload succeeded. " + (tail[-1].strip() if tail else ""))
    return True


def publish(app: Any, out_dir: str, project: str, subdomain: Optional[str]) -> bool:
    """Export the dashboard then upload it. The publisher thread's per-cycle call
    (infrastructure is already provisioned at bootstrap)."""
    live = None
    try:
        from praxis.interface.state.live_metrics import LiveMetrics

        live = LiveMetrics().snapshot()
    except Exception:
        pass  # no live metrics (e.g. standalone export); banner still shows
    out = export_snapshot(app, out_dir, live_metrics=live)
    run_hash = app.config.get("truncated_hash")
    return deploy_content(str(out), project, run_hash, subdomain)


def _training_started() -> bool:
    """True once training has produced at least one step. Used to hold the first
    publish until after torch.compile / warmup, so the debut snapshot carries
    real curves instead of an empty pre-compile dashboard."""
    try:
        from praxis.interface.state.live_metrics import LiveMetrics

        snap = LiveMetrics().snapshot()
        return (snap.get("step") or 0) > 0
    except Exception:
        return False


# --- Integration ------------------------------------------------------------


class Integration(BaseIntegration):
    """Publishes static dashboard snapshots to Cloudflare Pages on a cadence."""

    def __init__(self, spec: IntegrationSpec):
        super().__init__(spec)
        self._thread = None
        self._stop = threading.Event()

    # Publishing is a runtime/infra concern - these flags must never change a
    # run's identity, so they're excluded from the args hash.
    _PUBLISH_FLAGS = ["--publish-snapshot", "--publish-project", "--publish-interval"]

    def hash_exclusions(self) -> List[str]:
        return list(self._PUBLISH_FLAGS)

    def add_cli_args(self, parser) -> None:
        group = None
        for g in parser._action_groups:
            if g.title == "networking":
                group = g
                break
        if group is None:
            group = parser.add_argument_group("networking")
        group.add_argument(
            "--publish-snapshot",
            action="store_true",
            default=False,
            help="Periodically publish a static dashboard snapshot to Cloudflare Pages",
        )
        group.add_argument(
            "--publish-project",
            type=str,
            default="praxis",
            help="Cloudflare Pages project name to deploy to (default: praxis)",
        )
        group.add_argument(
            "--publish-interval",
            type=int,
            default=30,
            help="Minutes between snapshot publishes (default: 30)",
        )

    def on_api_server_start(self, app: Any, args: Any) -> None:
        # The loader invokes this hook as ``hook(host, port)`` - the positional
        # params are NOT the Flask app / parsed args (a legacy signature). Pull
        # the real ones from the framework, the way the ngrok integration does.
        try:
            from praxis.cli import get_cli_args

            cli_args = get_cli_args()
        except Exception:
            return

        # Re-check the flag: CLI args are registered for every integration, but
        # the hook must no-op unless the user actually asked to publish.
        if not getattr(cli_args, "publish_snapshot", False):
            return
        if self._thread is not None:
            return

        _load_env()
        project = getattr(cli_args, "publish_project", "praxis")
        interval = max(1, int(getattr(cli_args, "publish_interval", 30))) * 60

        from praxis import web

        flask_app = web.app

        # Two phases, so failures surface fast and content waits for real data:
        #   1. Provision the project + domain + DNS immediately (REST, seconds) -
        #      a bad token/zone errors at bootstrap, not after a 10-min compile.
        #   2. Hold the content upload until training has actually stepped (past
        #      torch.compile), so the debut snapshot carries real curves, not an
        #      empty pre-training dashboard. Cap the wait so a stalled run still
        #      puts the architecture/KB online.
        first_publish_cap = 20 * 60

        def loop():
            subdomain = ensure_infrastructure(project)  # phase 1: fast, fail-loud

            waited, started = 0, False
            while not self._stop.is_set() and waited < first_publish_cap:
                if _training_started():
                    started = True
                    break
                if self._stop.wait(10):
                    return
                waited += 10
            _log(
                "training underway; uploading first snapshot"
                if started
                else f"first-upload wait capped ({first_publish_cap // 60} min); "
                "uploading current state"
            )
            while not self._stop.is_set():  # phase 2: content, on the interval
                try:
                    publish(flask_app, DEFAULT_OUT_DIR, project, subdomain)
                except Exception as exc:
                    _log(f"publish cycle failed: {exc}")
                if self._stop.wait(interval):
                    break

        self._thread = threading.Thread(target=loop, name="cf-publisher", daemon=True)
        self._thread.start()
        _log(
            f"publisher started (project='{project}', every {interval // 60} min); "
            "provisioning now, first upload once training steps"
        )

    def cleanup(self) -> None:
        self._stop.set()
        self._thread = None
