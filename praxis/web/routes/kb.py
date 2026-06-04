"""Knowledge-base search API.

Serves ranked, search-as-you-type results over the FTS5 index built by
``tools/index_kb.py``. Opens the index read-only by path (same pattern as the
metrics/dynamics routes) and lazily builds it on first request if missing.
"""

from pathlib import Path

from flask import Blueprint, jsonify, request

from praxis.kb import DEFAULT_DB_PATH, KBIndex
from praxis.kb.sources import REPO_ROOT
from praxis.web.app import api_logger

kb_bp = Blueprint("kb", __name__)

MAX_LIMIT = 50


@kb_bp.route("/api/kb/search", methods=["GET"])
def kb_search():
    """Ranked KB search. Params: q, types (comma-separated), limit."""
    query = request.args.get("q", "").strip()
    types_param = request.args.get("types", "").strip()
    types = [t for t in types_param.split(",") if t] or None
    try:
        limit = min(int(request.args.get("limit", 20)), MAX_LIMIT)
    except ValueError:
        limit = 20

    try:
        index = _get_index()
        # Empty query -> recent feed (newest first), so clicking into an empty
        # box surfaces the latest content instead of nothing.
        hits = (
            index.search(query, types=types, limit=limit)
            if query
            else index.recent(limit=limit, types=types)
        )
        index.close()
    except Exception as exc:  # missing index, locked db, etc.
        api_logger.warning(f"KB search failed: {exc}")
        return jsonify({"status": "error", "message": str(exc), "hits": []}), 200

    return jsonify(
        {
            "status": "ok",
            "query": query,
            "hits": [
                {
                    "id": h.item.id,
                    "type": h.item.type,
                    "label": h.item.label,
                    "title": h.item.title,
                    "uri": h.item.uri,
                    "snippet": h.snippet,
                    "score": h.score,
                    "meta": h.item.meta,
                }
                for h in hits
            ],
        }
    )


@kb_bp.route("/api/kb/item", methods=["GET"])
def kb_item():
    """Fetch one KB item's full body for inline rendering. Param: id."""
    item_id = request.args.get("id", "").strip()
    if not item_id:
        return jsonify({"status": "error", "message": "missing id"}), 400

    try:
        index = _get_index()
        item = index.get(item_id)
        index.close()
    except Exception as exc:
        api_logger.warning(f"KB item fetch failed: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 200

    if item is None:
        return jsonify({"status": "error", "message": "not found"}), 404

    # Notes are indexed per-section so search lands on a heading, but the reader
    # should see the WHOLE document with that heading as the landing point - so
    # you can scroll above/below the section you matched. Swap the stored section
    # body for the full file and hand the frontend an anchor to scroll to.
    body, anchor = item.body, None
    if item.type == "note":
        full = _full_doc_body(item.uri)
        if full:
            body = full
        # id is "note:<stem>#<i>"; section 0 is the doc top (no heading to seek).
        if not item.id.endswith("#0"):
            anchor = item.title

    return jsonify(
        {
            "status": "ok",
            "item": {
                "id": item.id,
                "type": item.type,
                "label": item.label,
                "title": item.title,
                "uri": item.uri,
                "body": body,
                "anchor": anchor,
                "meta": item.meta,
            },
        }
    )


def _full_doc_body(uri: str) -> str:
    """Full text of a note's source file, resolved under the repo root (the uri
    is repo-relative, e.g. ``next/forced_computation.md``). Returns "" if the
    path escapes the repo or can't be read."""
    try:
        path = (REPO_ROOT / uri).resolve()
        if path.is_file() and REPO_ROOT in path.parents:
            return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        pass
    return ""


def _corpus_mtime() -> float:
    """Newest mtime across the hand-authored markdown corpus (docs/ + next/).
    Runs are excluded: their metrics.db is rewritten every training step and
    would otherwise trigger a reindex on nearly every keystroke."""
    latest = 0.0
    for sub in ("docs", "next"):
        for path in (REPO_ROOT / sub).glob("*.md"):
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                pass
    return latest


def _get_index() -> KBIndex:
    """Open the index read-only, (re)building it when missing or stale - so a
    newly added or edited doc under docs/ or next/ is picked up automatically,
    no manual reindex needed."""
    db = Path(DEFAULT_DB_PATH)
    try:
        stale = not db.exists() or db.stat().st_mtime < _corpus_mtime()
    except OSError:
        stale = True
    if stale:
        writer = KBIndex()
        writer.rebuild()
        writer.close()
    return KBIndex(read_only=True)
