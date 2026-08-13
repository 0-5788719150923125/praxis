from functools import partial
from typing import Optional, TypeVar

from praxis.routers.arc import ArcMixture
from praxis.routers.distance import Distance
from praxis.routers.mixture_of_depths import MixtureOfDepths
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
    # SMEAR (praxis/routers/smear.py): soft-merging of experts at the paper's
    # granularity - targets discovered per module, Linear targets routed per
    # example, expert dropout, one shared block plus `num_experts` low-rank
    # deviations. ONE entry, not a family: the expert count comes from
    # `num_experts`, the per-recurrent-pass bias is folded in (zero-init, so it
    # is absent until it learns otherwise), and the reduction is an argument.
    smear=SMEAR,
    # VEAR (praxis/routers/vear.py): the repo's variant - sharpened routing plus
    # inter-expert repulsion. A departure from the paper, unlike `smear`.
    vear=VEAR,
    # Batch-reduced SMEAR: one merged geometry for the entire batch, which is
    # what this repo's routers did before 2026-08-13. Present ONLY as the
    # control for the per-example claim; see REDUCTIONS in smear.py for why it
    # is very likely never the right choice.
    smear_batch=partial(SMEAR, reduction="batch"),
    # Per-token routing. Beyond the paper, affordable only on the Linear
    # targets, and the obvious next experiment if per-example pays off.
    smear_token=partial(SMEAR, reduction="token"),
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
