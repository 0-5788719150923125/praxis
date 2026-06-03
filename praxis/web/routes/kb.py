"""Knowledge-base search API.

Serves ranked, search-as-you-type results over the FTS5 index built by
``tools/index_kb.py``. Opens the index read-only by path (same pattern as the
metrics/dynamics routes) and lazily builds it on first request if missing.
"""

from pathlib import Path

from flask import Blueprint, jsonify, request

from praxis.kb import DEFAULT_DB_PATH, KBIndex
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
        hits = index.search(query, types=types, limit=limit) if query else []
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

    return jsonify(
        {
            "status": "ok",
            "item": {
                "id": item.id,
                "type": item.type,
                "label": item.label,
                "title": item.title,
                "uri": item.uri,
                "body": item.body,
                "meta": item.meta,
            },
        }
    )


def _get_index() -> KBIndex:
    """Open the index read-only, building it once if it doesn't exist yet."""
    if not Path(DEFAULT_DB_PATH).exists():
        KBIndex().rebuild()
    return KBIndex(read_only=True)
