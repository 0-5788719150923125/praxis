"""Framing fragments: the conditionally-rendered prose behind \\paperFraming*.

The invariant these guard is the one the pillars system exists to enforce - the
paper never claims machinery the run does not have. Four fragment families are
mutually exclusive across ``codec_mode`` (a run has exactly one input
representation), and a run that matched two of them, or none, would either
contradict itself or drop a paragraph the surrounding prose depends on.

The codec paragraphs used to live inline in body.tex and named CALM outright,
so a byte-latent run was told about a reconstructing codec it does not carry.

Experiment-driven cases read experiments/*.yml, which is gitignored apart from
a few committed references; a case whose experiment is missing skips.
"""

import re

import pytest

from praxis import registry
from praxis.pillars.framing import FRAMING, REPO_ROOT, active_fragments, resolve_config


def _config(experiment):
    try:
        return resolve_config(experiment)
    except FileNotFoundError as exc:
        pytest.skip(f"{exc} (experiments/*.yml is machine-local)")


# One fragment from each family must fire, and only one. Keyed by the section
# anchor the family renders into.
EXCLUSIVE_SECTIONS = ["abstract", "intro", "outer", "flip"]

# Representative experiments covering all three codec modes.
CODEC_EXPERIMENTS = {
    "calm-d": "calm",
    "calm-e": "byte_latent",
    "abstractinator-f": "byte_latent",
    "gpt2-1": "standard",
}


@pytest.mark.parametrize("experiment,expected_mode", CODEC_EXPERIMENTS.items())
def test_codec_mode_resolves_as_expected(experiment, expected_mode):
    assert _config(experiment)["codec_mode"] == expected_mode


@pytest.mark.parametrize("experiment", CODEC_EXPERIMENTS)
@pytest.mark.parametrize("section", EXCLUSIVE_SECTIONS)
def test_exactly_one_variant_per_exclusive_section(experiment, section):
    active = [
        f.id
        for f in active_fragments(_config(experiment))
        if f.section == section
    ]
    assert (
        len(active) == 1
    ), f"{experiment}/{section}: expected 1 fragment, got {active}"


@pytest.mark.parametrize("experiment", CODEC_EXPERIMENTS)
def test_temperature_mechanism_never_doubles_up(experiment):
    """The temperature fragments each close on the same summarizing sentence, so
    at most one may fire. Zero is fine - the body's general claim stands alone.
    Other manifold subsections (e.g. manifold-which-variance) are independent."""
    active = [
        f.id
        for f in active_fragments(_config(experiment))
        if f.id.startswith("manifold-temperature-")
    ]
    assert len(active) <= 1, f"{experiment}: temperature fragments collide: {active}"


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


def test_every_mtp_type_has_a_mechanism_paragraph():
    """The multi-token family opens with a generic intro (order 13) and follows
    it with one mechanism paragraph per mtp_type (order 14). A mode with no
    paragraph inherits only the intro, which cannot describe every bank at once -
    that is how the transformer and conv banks came to be described by prose
    written for a shared-parameter pool."""

    modes = set(registry.namespace("mtp")) | {"vear", "serpent_rnn"}
    covered = {
        m
        for frag in FRAMING.values()
        for m in frag.requires.get("mtp_type", [])
        if m != "*"
    }
    assert (
        not modes - covered
    ), f"mtp_type with no mechanism paragraph: {modes - covered}"


# ─── Prose lint: stock phrases a reader notices ──────────────────────────────
#
# Counted per RENDERED paper (body + the fragments this run activates + the
# thread's prose), never globally: the codec_mode variants of one family are
# alternates that deliberately share parallel phrasing, and only one of them ever
# reaches a reader. Three separate "is not a metaphor" assertions shipped in the
# first nine pages before anyone counted them, which is what this guards.

STOCK_PHRASES = {
    # Asserting that something is not figurative. Once is emphasis; twice is a
    # tic, and the reader starts hearing the crutch instead of the claim.
    r"not (a |in )metaphor": 1,
    # Same reflex, milder: "X makes this literal". The variants of it live in
    # fragments that rarely co-fire, so the cap is where a regression would land.
    r"makes (this|the [a-z-]+ [a-z]+) literal|implements this literally": 2,
}


def _rendered_prose(experiment: str) -> str:
    """Everything a reader of this run's PDF actually sees, minus TikZ."""
    parts = [(REPO_ROOT / "research" / "body.tex").read_text()]
    parts += [f.body for f in active_fragments(_config(experiment))]
    parts += [
        prose for _, prose in registry.lookup("threads", "blind_watchmaking").components
    ]
    blob = "\n".join(parts)
    blob = re.sub(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", blob, flags=re.S
    )
    # Drop LaTeX comments - build.py leaves explanatory ones in body.tex, and a
    # comment naming a phrase is not the paper using it.
    return re.sub(r"(?m)^\s*%.*$", " ", blob)


@pytest.mark.parametrize("experiment", CODEC_EXPERIMENTS)
@pytest.mark.parametrize("pattern,limit", STOCK_PHRASES.items())
def test_stock_phrases_stay_under_limit(experiment, pattern, limit):
    hits = re.findall(pattern, _rendered_prose(experiment), re.I)
    assert len(hits) <= limit, (
        f"{experiment}: /{pattern}/ appears {len(hits)}x (limit {limit}). "
        "Say the specific thing instead of reaching for the stock assertion."
    )


@pytest.mark.parametrize(
    "experiment,paired",
    [("abstractinator-p", True), ("abstractinator-o", False), ("calm-d", False)],
)
def test_the_calm_arm_addendum_gates_on_the_paired_codec(experiment, paired):
    """The order-13 addendum describes TWO codecs over one patch. It must not
    fire for an Abstractinator carrying only the quantizer, or for a CALM run
    carrying only the autoencoder - either would claim a pairing the run does
    not have."""
    cfg = _config(experiment)
    ids = {f.id for f in active_fragments(cfg)}
    assert cfg["uses_calm_arm"] is paired
    assert ("harmonic-calm-arm-abstractinator" in ids) is paired


def test_no_experiment_names_hardcoded_in_fragment_prose():
    """Fragment bodies must not name specific runs - the paper is rebuilt per
    run, so 'in calm-d, two physical layers' is wrong on every other run. Counts
    that vary belong in inlines (paperPhysicalLayers, paperRecurrentSteps)."""
    # Experiment stems like calm-d, abstractinator-f, small-b, gpt2-1.
    pattern = re.compile(r"\b(calm|abstractinator|small|gpt2)-[a-z0-9]\b", re.I)
    offenders = {
        fid: pattern.findall(frag.body)
        for fid, frag in FRAMING.items()
        if pattern.search(frag.body)
    }
    assert not offenders, f"fragment prose names specific runs: {offenders}"
