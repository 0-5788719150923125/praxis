"""Spider link-graph API.

Serves the citation rankings behind the Research tab's "Spider Citations"
card, read straight from spider.db (the worker stays the sole writer).
"""

import sqlite3

from flask import Blueprint, jsonify

from praxis.spider.store import DEFAULT_SPIDER_DB

spider_bp = Blueprint("spider", __name__)

TOP_N = 10


@spider_bp.route("/api/spider", methods=["GET"])
def spider_citations():
    """Top cited URLs and top referrer pages from the spider's link graph."""
    if not DEFAULT_SPIDER_DB.exists():
        return jsonify({"status": "no_data", "data": None})
    try:
        conn = sqlite3.connect(f"file:{DEFAULT_SPIDER_DB}?mode=ro", uri=True)
        cited = conn.execute(
            "SELECT dst, COUNT(*) AS n FROM refs GROUP BY dst "
            "ORDER BY n DESC LIMIT ?",
            (TOP_N,),
        ).fetchall()
        referrers = conn.execute(
            "SELECT src, COUNT(*) AS n FROM refs GROUP BY src "
            "ORDER BY n DESC LIMIT ?",
            (TOP_N,),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return jsonify({"status": "no_data", "data": None})
    if not cited:
        return jsonify({"status": "no_data", "data": None})
    resp = jsonify(
        {
            "status": "ok",
            "data": {
                "cited": [{"url": u, "count": n} for u, n in cited],
                "referrers": [{"url": u, "count": n} for u, n in referrers],
            },
        }
    )
    resp.headers.add("Cache-Control", "max-age=30")
    return resp
