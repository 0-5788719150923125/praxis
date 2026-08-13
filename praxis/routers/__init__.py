from functools import partial
from typing import Optional, TypeVar

from praxis.routers.arc import ArcMixture
from praxis.routers.arc_smear import ArcSMEAR, ArcVEAR
from praxis.routers.distance import Distance
from praxis.routers.mixture_of_depths import MixtureOfDepths
from praxis.routers.modular import (
    ArcModularSMEAR,
    ArcModularVEAR,
    ModularSMEAR,
    ModularVEAR,
)
from praxis.routers.prismatic import Prismatic
from praxis.routers.smear import SMEAR
from praxis.routers.taxus import Taxus
from praxis.routers.vear import VEAR

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


def calculate_computational_budget(
    config: ConfigType,
    target_ratio: float = 0.4,
    min_layers: int = 2,
    budget_type: str = "linear",
) -> float:
    """
    Calculate computational budget for routers based on model configuration.

    Args:
        config: Model configuration with depth information
        target_ratio: Target ratio of layers to use (0.0-1.0)
        min_layers: Minimum number of layers to always execute
        budget_type: Type of budget calculation:
            - "linear": Simple ratio of total depth
            - "frontloaded": More budget for early layers
            - "adaptive": Based on model size and task

    Returns:
        Computational budget value
    """
    depth = getattr(config, "depth", 8)

    if budget_type == "linear":
        # Simple linear budget based on target ratio
        budget = max(min_layers, target_ratio * depth)

    elif budget_type == "frontloaded":
        # Exponentially decaying budget - encourages early processing
        # Budget represents expected number of layers before exit
        budget = min_layers + (depth - min_layers) * (1 - target_ratio)

    elif budget_type == "adaptive":
        # Adaptive budget based on model size
        # Larger models get tighter budgets to save more compute
        hidden_size = getattr(config, "hidden_size", 768)
        size_factor = min(1.0, 768 / hidden_size)  # Smaller models get more budget
        budget = max(min_layers, target_ratio * depth * size_factor)

    else:
        raise ValueError(f"Unknown budget_type: {budget_type}")

    return float(budget)


def create_taxus_with_dynamic_budget(
    config: ConfigType,
    target_depth_ratio: float = 0.3,
    budget_ratio: float = 0.4,
    **kwargs,
) -> Taxus:
    """
    Create a Taxus router with dynamically calculated computational budget.

    Args:
        config: Model configuration
        target_depth_ratio: Target ratio for average exit depth
        budget_ratio: Ratio for computational budget calculation
        **kwargs: Additional arguments passed to Taxus

    Returns:
        Configured Taxus router instance
    """
    # Calculate budget based on actual model depth
    computational_budget = calculate_computational_budget(
        config, target_ratio=budget_ratio, budget_type="linear"
    )

    # Set defaults that encourage early exits
    defaults = {
        "target_depth_ratio": target_depth_ratio,
        "temperature": 0.3,
        "entropy_weight": 0.1,  # Increased for more decisive exits
        "usage_weight": 1.0,  # Strong pressure to match target depth
        "budget_weight": 1.0,  # Strong budget enforcement
        "computational_budget": computational_budget,
    }

    # Override with any provided kwargs
    defaults.update(kwargs)

    return Taxus(config, **defaults)


ROUTER_REGISTRY = dict(
    mixture_of_depths=MixtureOfDepths,
    mixture_of_depths_u=partial(MixtureOfDepths, layout="u"),
    mixture_of_depths_decayed=partial(MixtureOfDepths, layout="decayed"),
    mixture_of_depths_ramped=partial(MixtureOfDepths, layout="ramped"),
    mixture_of_depths_skip_2=partial(MixtureOfDepths, layout="skip_2"),
    arc_mixture=ArcMixture,
    smear=SMEAR,
    # VEAR: variance-driven SMEAR - sharpened routing + inter-expert repulsion for
    # discrete, unique geometries (praxis/routers/vear.py).
    vear=VEAR,
    # Depth-aware variants: a zero-init per-recurrent-pass bias on the router
    # logits, so each pass merges its own expert mixture instead of every depth
    # sharing one routing (praxis/routers/arc_smear.py). Identity at init.
    arc_smear=ArcSMEAR,
    arc_vear=ArcVEAR,
    # Modular SMEAR (praxis/routers/modular.py): the SAME method as `smear`
    # above, applied at the granularity the paper uses. ONE shared block
    # plus N zero-init deviations per DISCOVERED TARGET, with a coefficient row
    # each - instead of N copies of the whole block under a single scalar. The
    # expert count lives in the registry key rather than in `num_experts`,
    # because the config field means "how many blocks does the decoder build"
    # and these build one; setting `num_experts: 1` alongside them is what
    # switches the legacy bank off.
    smear_modular_4=partial(ModularSMEAR, num_experts=4),
    smear_modular_2=partial(ModularSMEAR, num_experts=2),
    # Depth-aware: a zero-init per-recurrent-pass bias on the per-target logits,
    # so each pass can move each module independently. Identity at init.
    arc_smear_modular_4=partial(ArcModularSMEAR, num_experts=4),
    arc_smear_modular_2=partial(ArcModularSMEAR, num_experts=2),
    arc_smear_modular_8=partial(ArcModularSMEAR, num_experts=8),
    # Narrower target profiles, for isolating where any gain comes from.
    arc_smear_modular_4_attn=partial(ArcModularSMEAR, num_experts=4, target_profile="attn"),
    arc_smear_modular_4_gates=partial(ArcModularSMEAR, num_experts=4, target_profile="gates"),
    # VEAR's p**4 sharpening on top, for the controlled comparison. Off in the
    # plain entries: sharpening a batch-averaged coefficient cannot add
    # per-input discreteness, it only starves the losing deviations.
    arc_vear_modular_4=partial(ArcModularVEAR, num_experts=4),
    vear_modular_4=partial(ModularVEAR, num_experts=4),
    distance=Distance,
    prismatic=Prismatic,
    taxus=create_taxus_with_dynamic_budget,
    taxus_aggressive=partial(
        create_taxus_with_dynamic_budget,
        target_depth_ratio=0.25,  # Target 25% depth
        budget_ratio=0.3,  # 30% computational budget
        temperature=0.2,  # Lower temp for more decisive exits
    ),
    taxus_balanced=partial(
        create_taxus_with_dynamic_budget,
        target_depth_ratio=0.5,  # Target 50% depth
        budget_ratio=0.6,  # 60% computational budget
        temperature=0.5,  # Moderate temperature
    ),
)
