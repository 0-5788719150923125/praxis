"""Business card rendering endpoints."""

import io
import secrets
import zipfile

from flask import Blueprint, current_app, request, send_file

cards_bp = Blueprint("cards", __name__)


def _params():
    seed = request.args.get("seed", type=int)
    if seed is None:
        seed = secrets.randbelow(2**31)
    side = request.args.get("side", "front")
    if side not in ("front", "back"):
        side = "front"
    theme = request.args.get("theme", "light")
    hue = request.args.get("hue", 161, type=float)
    from praxis.pillars.projections import MOD_AXES

    mods = {k: request.args.get(k, type=float) for k in MOD_AXES}
    return seed, side, theme, hue, mods


def _identity():
    cfg = current_app.config
    authors = cfg.get("author") or ["Ryan J. Brooks"]
    if isinstance(authors, str):
        authors = [authors]
    return authors, cfg.get("donations", ""), cfg.get("truncated_hash", "")


@cards_bp.route("/api/card/preview.svg", methods=["GET"])
def card_preview():
    from praxis.pillars.projections import render_card

    seed, side, theme, hue, mods = _params()
    authors, donations, run_hash = _identity()
    svg = render_card(side, seed, theme, hue, authors, donations, run_hash, mods=mods)
    resp = send_file(io.BytesIO(svg), mimetype="image/svg+xml")
    resp.headers["X-Card-Seed"] = str(seed)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _offset():
    """Print-calibration nudge in mm, clamped; defaults live in projections."""
    import math

    from praxis.pillars.projections import FEED_DX, FEED_DY

    def axis(name, default):
        v = request.args.get(name, default, type=float)
        if not math.isfinite(v):
            v = default
        return max(-4.0, min(4.0, v))

    return (axis("dx", FEED_DX), axis("dy", FEED_DY))


def _zip_both(render, name_fmt, zip_name):
    seed, _, theme, hue, mods = _params()
    authors, donations, run_hash = _identity()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for side in ("front", "back"):
            pdf = render(
                side,
                seed,
                theme,
                hue,
                authors,
                donations,
                run_hash,
                mods=mods,
                offset=_offset(),
            )
            zf.writestr(name_fmt.format(side=side), pdf)
    buf.seek(0)
    return send_file(
        buf, mimetype="application/zip", as_attachment=True, download_name=zip_name
    )


@cards_bp.route("/api/card/cards.zip", methods=["GET"])
def cards_zip():
    from praxis.pillars.projections import render_single_pdf

    return _zip_both(render_single_pdf, "praxis-card-{side}.pdf", "praxis-card.zip")


@cards_bp.route("/api/card/sheets.zip", methods=["GET"])
def sheets_zip():
    from praxis.pillars.projections import render_sheet_pdf

    return _zip_both(
        render_sheet_pdf, "praxis-cards-10up-{side}.pdf", "praxis-cards-10up.zip"
    )


@cards_bp.route("/api/card/card.pdf", methods=["GET"])
def card_pdf():
    from praxis.pillars.projections import render_single_pdf

    seed, side, theme, hue, mods = _params()
    authors, donations, run_hash = _identity()
    pdf = render_single_pdf(
        side,
        seed,
        theme,
        hue,
        authors,
        donations,
        run_hash,
        mods=mods,
        offset=_offset(),
    )
    return send_file(
        io.BytesIO(pdf),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"praxis-card-{side}.pdf",
    )


@cards_bp.route("/api/card/sheet.pdf", methods=["GET"])
def sheet_pdf():
    from praxis.pillars.projections import render_sheet_pdf

    seed, side, theme, hue, mods = _params()
    authors, donations, run_hash = _identity()
    pdf = render_sheet_pdf(
        side,
        seed,
        theme,
        hue,
        authors,
        donations,
        run_hash,
        mods=mods,
        offset=_offset(),
    )
    return send_file(
        io.BytesIO(pdf),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"praxis-cards-10up-{side}.pdf",
    )
