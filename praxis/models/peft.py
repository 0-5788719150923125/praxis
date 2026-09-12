"""Parameter-efficient finetuning profiles for a foreign model.

A published checkpoint is usually finetuned, not trained from scratch, and
full-rank finetuning of somebody else's weights is both expensive and the
easiest way to destroy them. ``--peft-type`` selects a profile here: the base
weights freeze and a small adapter trains alongside them, so the run's gradient
budget goes to the new behavior instead of to relearning the old one.

Profiles rather than a pile of ``--lora-*`` flags, for the usual reason - an arm
of an experiment should be ONE key, so two runs differ by a name and not by four
numbers somebody has to remember to keep in sync.

``peft`` is an optional dependency, installed at runtime only when a profile is
selected (the same contract as Ray in :mod:`praxis.trainers.ray_support`), so
nothing here imports it at module scope.
"""

import importlib.util
import subprocess
import sys
from typing import Any, Dict, Optional

from praxis import registry
from praxis.registry import Entry

# Matches the '[peft]' extra in pyproject.toml.
PEFT_REQUIREMENT = "peft>=0.14"

# Shape shared by every LoRA-family profile below. Only what a profile
# genuinely varies is repeated in its entry.
_LORA_BASE: Dict[str, Any] = dict(
    peft_type="LORA",
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # Every linear layer outside the embedding and the output projection. PEFT
    # resolves this per architecture, which is what keeps the profile
    # model-agnostic: naming q_proj/v_proj here would silently match nothing on
    # a model that calls them something else.
    target_modules="all-linear",
)


registry.declare(
    "peft_profiles",
    title="Parameter-efficient finetuning",
    doc=(
        "Adapter profiles for finetuning a published checkpoint (``--peft-type``). "
        "Each entry freezes the loaded weights and trains a small adapter in their "
        "place, and bundles everything that defines the variant - the method, the "
        "rank, the scaling rule and which modules are adapted - so an experiment "
        "arm is one key. Requires the optional ``peft`` package, installed on "
        "demand when a profile is selected."
    ),
    entries={
        "lora": Entry(
            dict(_LORA_BASE),
            (
                "Low-Rank Adaptation (Hu et al. 2021) at rank 16 over every "
                "linear layer outside the embeddings: the default, and the one "
                "to reach for unless an ablation says otherwise."
            ),
        ),
        "lora_wide": Entry(
            dict(_LORA_BASE, r=64, lora_alpha=128),
            (
                "LoRA with a rank-64 adapter (alpha held at the conventional "
                "2r), four times the default's capacity over the same modules. "
                "For a finetune that has to move the model further than a style "
                "shift - more data, or data further from what it was pretrained "
                "on - at four times the adapter parameters and optimizer state."
            ),
        ),
        "lora_attention": Entry(
            dict(_LORA_BASE, target_modules=["q_proj", "v_proj"]),
            (
                "LoRA on the query and value projections only - the original "
                "paper's setting, and the cheapest arm. Names the modules "
                "explicitly, so it only applies to architectures that use the "
                "conventional projection names."
            ),
        ),
        "rslora": Entry(
            dict(_LORA_BASE, use_rslora=True),
            (
                "LoRA with rank-stabilized scaling (Kalajdzievski 2023): the "
                "adapter is scaled by alpha/sqrt(r) rather than alpha/r, which "
                "stops the effective learning rate from collapsing as rank grows."
            ),
        ),
        "dora": Entry(
            dict(_LORA_BASE, use_dora=True),
            (
                "Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024): the "
                "update is split into a magnitude and a direction and only the "
                "direction is low-rank, which tracks full finetuning more "
                "closely at low rank for more compute per step."
            ),
        ),
    },
)


def ensure_peft(peft_type: Optional[str]) -> None:
    """Install ``peft`` at runtime if a profile was selected.

    No-op without one, or when it is already importable. Same contract as
    :func:`praxis.trainers.ray_support.ensure_ray`: an optional dependency is
    fetched when the parsed args actually ask for it, never at import.
    """
    if not peft_type:
        return
    if importlib.util.find_spec("peft") is not None:
        return
    print(f"[ENV] Installing optional PEFT dependency ({PEFT_REQUIREMENT})...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", PEFT_REQUIREMENT]
        )
    except subprocess.CalledProcessError:
        sys.exit(
            "[ERROR] PEFT install failed. Install manually with\n"
            f"        pip install '{PEFT_REQUIREMENT}'"
        )
    if importlib.util.find_spec("peft") is None:
        sys.exit("[ERROR] PEFT installed but is not importable in this environment.")
    print("[ENV] PEFT installation complete.")


def apply_peft(model, peft_type: str, adapter_overrides: Optional[Dict] = None):
    """Wrap ``model``'s hosted weights in the named adapter, in place.

    The adapter goes on the HOSTED model, not on the Praxis wrapper: PEFT
    rewrites the modules it targets and forwards attribute access to what it
    wraps, so the wrapper keeps reaching the output projection and the
    objectives keep seeing an ordinary model. Wrapping the outside instead would
    put PEFT's forward between the trunk and the criterion.
    """
    from peft import LoraConfig, get_peft_model

    profile = dict(registry.lookup("peft_profiles", peft_type))
    profile.update(adapter_overrides or {})
    profile.pop("peft_type", None)

    # The adapter's target names come from the loaded architecture, so a
    # family that names its projections unconventionally corrects them here
    # rather than forcing a second profile.
    from praxis.models import get_adapter

    targets = get_adapter(getattr(model.config, "model_type", None)).lora_targets
    if targets is not None:
        profile["target_modules"] = targets

    model.model = get_peft_model(model.model, LoraConfig(**profile))
    model.peft_type = peft_type
    return model


def peft_summary(model) -> str:
    """One line naming the adapter and what fraction of the model it trains."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    share = 100.0 * trainable / total if total else 0.0
    return (
        f"[PEFT] {getattr(model, 'peft_type', 'none')}: "
        f"{trainable / 1e6:.2f}M of {total / 1e6:.2f}M parameters trainable "
        f"({share:.2f}%)."
    )
