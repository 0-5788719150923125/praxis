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


def _zip_both(render, name_fmt, zip_name):
    seed, _, theme, hue, mods = _params()
    authors, donations, run_hash = _identity()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for side in ("front", "back"):
            pdf = render(side, seed, theme, hue, authors, donations,
                         run_hash, mods=mods)
            zf.writestr(name_fmt.format(side=side), pdf)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=zip_name)


@cards_bp.route("/api/card/cards.zip", methods=["GET"])
def cards_zip():
    from praxis.pillars.projections import render_single_pdf

    return _zip_both(render_single_pdf, "praxis-card-{side}.pdf",
                     "praxis-card.zip")


@cards_bp.route("/api/card/sheets.zip", methods=["GET"])
def sheets_zip():
    from praxis.pillars.projections import render_sheet_pdf

    return _zip_both(render_sheet_pdf, "praxis-cards-8up-{side}.pdf",
                     "praxis-cards-8up.zip")


@cards_bp.route("/api/card/card.pdf", methods=["GET"])
def card_pdf():
    from praxis.pillars.projections import render_single_pdf

    seed, side, theme, hue, mods = _params()
    authors, donations, run_hash = _identity()
    pdf = render_single_pdf(side, seed, theme, hue, authors, donations, run_hash, mods=mods)
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
    pdf = render_sheet_pdf(side, seed, theme, hue, authors, donations, run_hash, mods=mods)
    return send_file(
        io.BytesIO(pdf),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"praxis-cards-8up-{side}.pdf",
    )
