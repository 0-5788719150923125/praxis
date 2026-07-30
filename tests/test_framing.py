"""Framing fragments: the conditionally-rendered prose behind \\paperFraming*.

The invariant these guard is the one the pillars system exists to enforce - the
paper never claims machinery the run does not have. Four fragment families are
mutually exclusive across ``codec_mode`` (a run has exactly one input
representation), and a run that matched two of them, or none, would either
contradict itself or drop a paragraph the surrounding prose depends on.

This class of bug is why the tests exist: the codec paragraphs used to live
inline in body.tex and named CALM outright, so a byte-latent run was told about
a reconstructing codec it does not carry.
"""

import pytest

from praxis.pillars.framing import (
    FRAMING,
    active_fragments,
    resolve_config,
)

# One fragment from each family must fire, and only one. Keyed by the section
# anchor the family renders into.
EXCLUSIVE_SECTIONS = ["abstract", "intro", "outer"]

# Representative experiments covering all three codec modes.
CODEC_EXPERIMENTS = {
    "calm-d": "calm",
    "calm-e": "byte_latent",
    "abstractinator-f": "byte_latent",
    "gpt2-1": "standard",
}


@pytest.mark.parametrize("experiment,expected_mode", CODEC_EXPERIMENTS.items())
def test_codec_mode_resolves_as_expected(experiment, expected_mode):
    assert resolve_config(experiment)["codec_mode"] == expected_mode


@pytest.mark.parametrize("experiment", CODEC_EXPERIMENTS)
@pytest.mark.parametrize("section", EXCLUSIVE_SECTIONS)
def test_exactly_one_variant_per_exclusive_section(experiment, section):
    active = [f.id for f in active_fragments(resolve_config(experiment)) if f.section == section]
    assert len(active) == 1, f"{experiment}/{section}: expected 1 fragment, got {active}"


@pytest.mark.parametrize("experiment", CODEC_EXPERIMENTS)
def test_temperature_mechanism_never_doubles_up(experiment):
    """The manifold fragments each close on the same summarizing sentence, so at
    most one may fire. Zero is fine - the body's general claim stands alone."""
    active = [f.id for f in active_fragments(resolve_config(experiment)) if f.section == "manifold"]
    assert len(active) <= 1, f"{experiment}: manifold fragments collide: {active}"


def test_every_exclusive_family_covers_all_codec_modes():
    """A codec_mode with no variant would silently drop the paragraph, which is
    how a gap gets shipped: the build succeeds and the section just vanishes."""
    for section in EXCLUSIVE_SECTIONS:
        covered = set()
        for frag in FRAMING.values():
            if frag.section == section:
                covered.update(frag.requires.get("codec_mode", []))
        missing = {"calm", "byte_latent", "standard"} - covered
        assert not missing, f"section '{section}' has no fragment for {missing}"


def test_no_experiment_names_hardcoded_in_fragment_prose():
    """Fragment bodies must not name specific runs - the paper is rebuilt per
    run, so 'in calm-d, two physical layers' is wrong on every other run. Counts
    that vary belong in inlines (paperPhysicalLayers, paperRecurrentSteps)."""
    import re

    # Experiment stems like calm-d, abstractinator-f, small-b, gpt2-1.
    pattern = re.compile(r"\b(calm|abstractinator|small|gpt2)-[a-z0-9]\b", re.I)
    offenders = {
        fid: pattern.findall(frag.body)
        for fid, frag in FRAMING.items()
        if pattern.search(frag.body)
    }
    assert not offenders, f"fragment prose names specific runs: {offenders}"
