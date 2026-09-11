"""Coverage guards for the auto-docs generator.

The failure these catch is silent: a registry namespace a reader can select, or
a new package, that nothing in ``docs/`` ever mentions.
"""

from pathlib import Path

import praxis
from praxis.docs import (
    INFRASTRUCTURE_PACKAGES,
    page_slug,
    registry_pages,
    undocumented_registries,
)

REPO_ROOT = Path(praxis.__file__).resolve().parent.parent


def test_every_selectable_namespace_has_a_page():
    missing = undocumented_registries()
    assert not missing, (
        "registry namespaces bound to a CLI flag with no docs page: "
        f"{missing}. Give each one a title (and doc) where it is declared."
    )


def test_every_package_is_a_registry_page_or_infrastructure():
    slugs = {page_slug(ns) for ns in registry_pages()}
    infra = {slug for slug, _ in INFRASTRUCTURE_PACKAGES}
    packages = {
        p.name
        for p in (REPO_ROOT / "praxis").iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    }
    missing = sorted(packages - slugs - infra)
    assert not missing, (
        f"praxis subpackages absent from the docs index: {missing}. Give each "
        "one a registry page or an INFRASTRUCTURE_PACKAGES one-liner."
    )
