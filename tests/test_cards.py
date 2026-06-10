import pytest
from flask import Flask

from praxis.pillars.projections import render_card, render_sheet_pdf
from praxis.web.routes.cards import cards_bp

AUTHORS = ["Ryan J. Brooks"]
DONATE = "https://example.com/donate"


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(cards_bp)
    app.config["author"] = AUTHORS
    app.config["donations"] = DONATE
    app.config["truncated_hash"] = "abc123"
    return app.test_client()


def test_render_card_deterministic():
    kwargs = dict(authors=AUTHORS, donations=DONATE, run_hash="abc123")
    a = render_card("front", 42, "light", 161, **kwargs)
    b = render_card("front", 42, "light", 161, **kwargs)
    assert a == b
    assert a != render_card("back", 42, "light", 161, **kwargs)
    assert a != render_card("front", 43, "light", 161, **kwargs)


def test_chaos_changes_field():
    kwargs = dict(authors=AUTHORS, donations=DONATE, run_hash="abc123")
    lo = render_card("front", 42, "light", 161, chaos=0.0, **kwargs)
    hi = render_card("front", 42, "light", 161, chaos=1.0, **kwargs)
    assert lo != hi
    assert lo == render_card("front", 42, "light", 161, chaos=0.0, **kwargs)


def test_every_field_renders():
    import praxis.pillars.projections as P

    full = dict(P.PROJECTION_REGISTRY)
    try:
        for name, fn in full.items():
            P.PROJECTION_REGISTRY.clear()
            P.PROJECTION_REGISTRY[name] = fn
            for side in ("front", "back"):
                out = render_card(side, 7, "dark", 200, AUTHORS, DONATE, "abc")
                assert out.startswith(b"<?xml"), name
    finally:
        P.PROJECTION_REGISTRY.clear()
        P.PROJECTION_REGISTRY.update(full)


def test_sheet_is_pdf():
    out = render_sheet_pdf("back", 7, "dark", 200, AUTHORS, DONATE, "abc123")
    assert out[:4] == b"%PDF"


def test_preview_route(client):
    resp = client.get("/api/card/preview.svg?seed=5&side=front&theme=dark&hue=161")
    assert resp.status_code == 200
    assert resp.mimetype == "image/svg+xml"
    assert resp.headers["X-Card-Seed"] == "5"


def test_preview_route_random_seed(client):
    resp = client.get("/api/card/preview.svg")
    assert resp.status_code == 200
    assert int(resp.headers["X-Card-Seed"]) >= 0


def test_zip_routes(client):
    import io
    import zipfile

    for path, names in [
        ("/api/card/cards.zip", {"praxis-card-front.pdf", "praxis-card-back.pdf"}),
        (
            "/api/card/sheets.zip",
            {"praxis-cards-10up-front.pdf", "praxis-cards-10up-back.pdf"},
        ),
    ]:
        resp = client.get(f"{path}?seed=5")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.data))
        assert set(zf.namelist()) == names
        for n in names:
            assert zf.read(n)[:4] == b"%PDF"


def test_pdf_routes(client):
    for path, name in [
        ("/api/card/card.pdf", "praxis-card-back.pdf"),
        ("/api/card/sheet.pdf", "praxis-cards-10up-back.pdf"),
    ]:
        resp = client.get(f"{path}?seed=5&side=back")
        assert resp.status_code == 200
        assert resp.data[:4] == b"%PDF"
        assert name in resp.headers["Content-Disposition"]
