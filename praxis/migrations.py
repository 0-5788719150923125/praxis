"""Migrations for data written by an older version of the code.

Checkpoints, run specs, and experiment configs outlive the names they were
written under. Everything that reads one of those translates it here, so a
rename never silently falls back to a default (an unknown config key is
ignored, an unknown state-dict key fails to load).
"""

from typing import Dict, MutableMapping

# Config keys: old -> new.
LEGACY_CONFIG_KEYS: Dict[str, str] = {
    "head_type": "classifier_type",
}

# State-dict path segments renamed wherever they appear: old -> new.
_LEGACY_SEGMENTS: Dict[str, str] = {
    "head": "classifier",
    "backward_head": "backward_classifier",
    "lm_head": "scorer",
    "energy_head": "generator",
    "code_heads": "code_classifiers",
    "next_code_heads": "next_code_classifiers",
    "patch_head": "patch_projection",
}

# Segments renamed only inside a classifier's subtree (SequentialClassifier's
# ModuleList was ``heads``).
_LEGACY_CLASSIFIER_SEGMENTS: Dict[str, str] = {
    "heads": "stages",
}


def pop_legacy_config_keys(config: MutableMapping) -> Dict[str, object]:
    """Remove every legacy key from ``config`` and return ``{new: value}``.

    A key present under both names keeps the new name's value."""
    found = {}
    for old, new in LEGACY_CONFIG_KEYS.items():
        if old in config:
            value = config.pop(old)
            if new not in config:
                found[new] = value
    return found


def rename_legacy_config(config: MutableMapping) -> MutableMapping:
    """Translate legacy keys in ``config`` in place and return it."""
    config.update(pop_legacy_config_keys(config))
    return config


def rename_legacy_key(key: str) -> str:
    """A state-dict key with every legacy segment renamed."""
    out = []
    inside = False
    for segment in key.split("."):
        if inside and segment in _LEGACY_CLASSIFIER_SEGMENTS:
            segment = _LEGACY_CLASSIFIER_SEGMENTS[segment]
        else:
            segment = _LEGACY_SEGMENTS.get(segment, segment)
        inside = inside or segment in ("classifier", "backward_classifier")
        out.append(segment)
    return ".".join(out)


def rename_legacy_state_dict(state_dict: MutableMapping, prefix: str = "") -> None:
    """Rename legacy keys under ``prefix`` in place. Keys already carrying the
    new names are untouched, so this is safe on any checkpoint."""
    for key in [k for k in state_dict if k.startswith(prefix)]:
        new = prefix + rename_legacy_key(key[len(prefix) :])
        if new != key:
            state_dict[new] = state_dict.pop(key)
