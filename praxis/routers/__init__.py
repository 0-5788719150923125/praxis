from functools import partial
from typing import Optional, TypeVar

from praxis import registry
from praxis.registry import Entry
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


registry.declare(
    "routers",
    title="Token routers",
    doc=(
        (
            "Token-routing mechanisms, including the Mixture-of-Depths family that skips a "
            "fraction of tokens per layer and the SMEAR family that soft-merges expert "
            "parameters. Unset runs no router."
        )
    ),
    entries={
        "mixture_of_depths": Entry(
            MixtureOfDepths,
            (
                "Mixture-of-Depths with layers alternating between full capacity and "
                "12.5% capacity."
            ),
        ),
        "mixture_of_depths_u": Entry(
            partial(MixtureOfDepths, layout="u"),
            (
                "Mixture-of-Depths with a U-shaped layout: full capacity at the first "
                "and last layers, 12.5% between."
            ),
        ),
        "mixture_of_depths_decayed": Entry(
            partial(MixtureOfDepths, layout="decayed"),
            (
                "Mixture-of-Depths whose capacity falls smoothly with depth, from full "
                "at the first layer toward 12.5% at the last."
            ),
        ),
        "mixture_of_depths_ramped": Entry(
            partial(MixtureOfDepths, layout="ramped"),
            (
                "The reverse of ``mixture_of_depths_decayed``: capacity rises smoothly "
                "from 12.5% at the first layer toward full at the last."
            ),
        ),
        "mixture_of_depths_skip_2": Entry(
            partial(MixtureOfDepths, layout="skip_2"),
            (
                "Mixture-of-Depths where one layer in three runs at full capacity and "
                "the other two at 12.5%."
            ),
        ),
        "arc_mixture": ArcMixture,
        "smear": Entry(
            SMEAR,
            (
                "Soft-merging of experts at the paper's granularity: targets "
                "discovered per module, Linear targets routed per example, expert "
                "dropout, one shared block plus ``num_experts`` low-rank deviations. "
                "One entry, not a family: the expert count comes from ``num_experts``, "
                "the per-recurrent-pass bias is folded in (zero-init, so it is absent "
                "until it learns otherwise), and the reduction is an argument."
            ),
        ),
        "vear": Entry(
            VEAR,
            (
                "Praxis's own SMEAR variant: sharpened routing plus inter-expert "
                "repulsion. A departure from the paper, unlike ``smear``."
            ),
        ),
        "smear_batch": Entry(
            partial(SMEAR, reduction="batch"),
            (
                "SMEAR with one input-free geometry (the depth prior) for the whole "
                "batch: the control arm for the per-example routing claim. See "
                "``REDUCTIONS`` in praxis/routers/smear.py for why it is very likely "
                "never the right choice."
            ),
        ),
        "smear_token": Entry(
            partial(SMEAR, reduction="token"),
            (
                "SMEAR where every position routes on its own state. Beyond the paper, "
                "and possible only on the Linear targets, where associativity means "
                "the merged weight is never materialized."
            ),
        ),
        "distance": Distance,
        "prismatic": Prismatic,
        "taxus": Entry(
            create_taxus_with_dynamic_budget,
            (
                "Taxus, the depth-buying early-exit router, with its computational "
                "budget derived from the model depth (40% of layers) and a target exit "
                "at 30% of depth."
            ),
        ),
        "taxus_aggressive": Entry(
            partial(
                create_taxus_with_dynamic_budget,
                target_depth_ratio=0.25,  # Target 25% depth
                budget_ratio=0.3,  # 30% computational budget
                temperature=0.2,  # Lower temp for more decisive exits
            ),
            (
                "Taxus targeting an exit at 25% of depth under a 30% budget, with a "
                "lower temperature for more decisive exits."
            ),
        ),
        "taxus_balanced": Entry(
            partial(
                create_taxus_with_dynamic_budget,
                target_depth_ratio=0.5,  # Target 50% depth
                budget_ratio=0.6,  # 60% computational budget
                temperature=0.5,  # Moderate temperature
            ),
            (
                "Taxus targeting an exit at 50% of depth under a 60% budget, at a "
                "moderate temperature."
            ),
        ),
    },
)
