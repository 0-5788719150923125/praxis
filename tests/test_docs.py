"""Coverage guards for the auto-docs generator.

The failure these catch is silent: a new namespace lands in
``praxis/__init__.py`` and nothing in ``docs/`` ever mentions it.
"""

from pathlib import Path

import praxis
from praxis.docs import (
    INFRASTRUCTURE_PACKAGES,
    _registries,
    _registry_attr,
    undocumented_registries,
)

REPO_ROOT = Path(praxis.__file__).resolve().parent.parent


def test_every_exported_registry_is_documented():
    missing = undocumented_registries()
    assert not missing, (
        "registries exported from praxis/__init__.py with no docs page: "
        f"{missing}. Add them to praxis.docs._registries(), or waive them in "
        "_REGISTRY_WAIVERS."
    )


def test_every_package_is_a_registry_page_or_infrastructure():
    slugs = {slug for slug, *_ in _registries()}
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


def test_registry_attr_resolves_for_every_page():
    for slug, _title, registry, *_ in _registries():
        assert _registry_attr(slug, registry) != "(unnamed)", slug
