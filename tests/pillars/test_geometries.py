"""Center-PCA figure layout: the panel grid must fit inside the text block.

Regression guard for a failure that only appears at 5+ panels, which is rare
enough to ship unnoticed: a fixed 0.46\\linewidth panel width put three rows plus
a seven-line caption past \\textheight, and the caption printed over the page
number. The width now scales with the row count.
"""

from praxis.pillars.geometries import (
    _CAPTION_ALLOWANCE,
    _MAX_PANEL_WIDTH,
    _PANEL_ASPECT,
    _TEXT_HEIGHT,
    _panel_width,
    figure_tex,
)


def test_only_overflow_narrows_the_panels():
    """The common case (up to two rows) keeps the two-up width, and no row
    count ever widens past it."""
    assert _panel_width(1) == _panel_width(2) == _MAX_PANEL_WIDTH
    assert all(_panel_width(r) <= _MAX_PANEL_WIDTH for r in range(0, 9))


def test_grid_plus_caption_fits_the_text_block():
    # Past two rows the width is derived to consume the budget exactly, so the
    # comparison lands on the boundary and needs a float tolerance rather than
    # a strict <=. The margin that matters is already inside _TEXT_HEIGHT.
    for rows in range(1, 9):
        height = rows * _panel_width(rows) * _PANEL_ASPECT + _CAPTION_ALLOWANCE
        assert height <= _TEXT_HEIGHT + 1e-9, f"{rows} rows overflow: {height:.3f}"


def test_five_panels_emit_a_narrowed_three_row_grid():
    tex = figure_tex([f"figures/geometry_{i}.png" for i in range(1, 6)], [])
    assert tex.count("includegraphics") == 5
    assert tex.count(r"\\[6pt]") == 2  # three rows -> two row breaks
    assert "0.460\\linewidth" not in tex


def test_halo_prototypes_are_not_mistaken_for_crystal_geometry():
    """``HaloGeometry`` also names its parameter ``centers`` - exclude it.

    prismatic5/6 carry a HaloClassifier arm whose ``scorer.centers`` is [vocab, dim]
    exactly like a crystal's, so the bare ``.centers`` suffix swept it into the
    figure as a fifth crystal geometry. It is not one: HALO's prototypes live on a
    hypersphere, and its gate share is near zero, so the panel was a raw ``randn``
    init - a featureless blob printed under a caption claiming settled geometry.

    The discriminator is structural: HaloGeometry owns a learnable ``gamma``
    beside its centers, CrystalGeometry owns nothing beside them.
    """
    from praxis.pillars.geometries import is_crystal_centers

    crystal = "model.classifier.branches.0.bank.experts.2.centers"
    halo = "model.classifier.branches.2.scorer.centers"
    sd = {
        crystal: object(),
        halo: object(),
        "model.classifier.branches.2.scorer.gamma": object(),
    }
    assert is_crystal_centers(sd, crystal)
    assert not is_crystal_centers(sd, halo)


def test_caption_names_the_classifier_the_run_actually_used():
    """The bank clause hardcoded ``prismatic4`` and kept asserting it.

    abstractinator-n ran ``classifier_type: prismatic6``, and the figure still told the
    reader it was looking at prismatic4's bank. The classifier name now comes from
    the run's own spec, and falls back to naming none at all rather than guessing.
    """

    def geo(classifier_type):
        return {
            "name": "abstractinator-n",
            "label": "branch 0 - expert 0",
            "intra_run": True,
            "classifier_type": classifier_type,
        }

    tex = figure_tex(["figures/geometry_1.png"], [geo("prismatic6")])
    assert "prismatic6's VEAR crystal bank" in tex
    assert "prismatic4" not in tex

    # No spec on disk: say nothing rather than assert a classifier name.
    tex = figure_tex(["figures/geometry_1.png"], [geo(None)])
    assert "a VEAR crystal bank" in tex
    assert "prismatic" not in tex


def test_intra_run_panels_respect_the_limit(tmp_path, monkeypatch):
    """A bank wider than the budget used to return every expert regardless.

    The cross-run path honoured ``limit``; the multi-geometry early return did not,
    so the figure could overflow its float page (and did, at five panels).
    """
    import torch

    import praxis.pillars.geometries as g

    ckpt = tmp_path / "last.ckpt"
    bank = {
        f"model.classifier.bank.experts.{i}.centers": torch.randn(8, 4)
        for i in range(5)
    }
    torch.save({"state_dict": bank}, ckpt)
    monkeypatch.setattr(
        g, "runs_newest_first", lambda: [(0.0, "abc123", "wide-bank", str(tmp_path))]
    )
    monkeypatch.setattr(g, "latest_checkpoint", lambda run_dir: str(ckpt))
    monkeypatch.setattr(g, "classifier_type_of", lambda run_dir: "prismatic4")

    geos = g.collect_geometries(limit=2, scan=1)
    assert len(geos) == 2
    assert all(geo["intra_run"] for geo in geos)


def test_legacy_checkpoint_keys_still_yield_geometries(tmp_path, monkeypatch):
    """Checkpoints written before the heads -> classifiers rename say
    ``model.head...lm_head.centers`` and ``heads.N``; they must still render,
    labelled by the same branch as a new checkpoint."""
    import torch

    import praxis.pillars.geometries as g

    ckpt = tmp_path / "last.ckpt"
    legacy = {
        "model.head.branches.1.heads.1.lm_head.centers": torch.randn(8, 4),
        "model.head.branches.2.lm_head.centers": torch.randn(8, 4),
        "model.head.branches.2.lm_head.gamma": torch.ones(1),
    }
    torch.save({"state_dict": legacy}, ckpt)
    monkeypatch.setattr(
        g, "runs_newest_first", lambda: [(0.0, "abc123", "old-run", str(tmp_path))]
    )
    monkeypatch.setattr(g, "latest_checkpoint", lambda run_dir: str(ckpt))
    monkeypatch.setattr(g, "classifier_type_of", lambda run_dir: "prismatic5")

    geos = g.collect_geometries(limit=4, scan=1)
    assert len(geos) == 1  # the HALO prototypes stay excluded
    assert g.branch_label("model.classifier.branches.1.stages.1.scorer.centers") == (
        "branch 1"
    )


def test_classifier_type_falls_back_to_a_legacy_spec(tmp_path):
    """Run specs written before the rename store ``args.head_type``."""
    import json

    from praxis.pillars.geometries import classifier_type_of

    (tmp_path / "spec.json").write_text(json.dumps({"args": {"head_type": "crystal"}}))
    assert classifier_type_of(str(tmp_path)) == "crystal"
    (tmp_path / "spec.json").write_text(
        json.dumps({"args": {"classifier_type": "prismatic8"}})
    )
    assert classifier_type_of(str(tmp_path)) == "prismatic8"
    assert classifier_type_of(str(tmp_path / "missing")) is None
