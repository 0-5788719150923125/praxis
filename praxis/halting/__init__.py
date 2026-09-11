from functools import partial

from praxis import registry
from praxis.halting.base import BaseHalting
from praxis.halting.kl import KLDivergenceHalting
from praxis.halting.reinject import ReinjectedKLHalting
from praxis.registry import Entry

registry.declare(
    "halting",
    title="Halting / early exit",
    doc=(
        (
            "Per-token mechanisms for early exit from recurrent depth loops. A loop is one "
            "trip through all ``--num-layers`` blocks and a run has ``depth // "
            "num_layers`` of them, so halting only has something to cut when depth is at "
            "least twice num_layers. Unset always runs full depth. Only the sequential "
            "decoder consults it."
        )
    ),
    entries={
        "none": BaseHalting,
        "kl": Entry(
            KLDivergenceHalting,
            (
                "Randomized training depth centred at half the budget - the "
                "recurrent-depth paper's own setting, and the right shape while the "
                "budget is small. KL-based halting at inference."
            ),
        ),
        "kl_log": Entry(
            partial(KLDivergenceHalting, prior="log"),
            (
                "The same halting rule as ``kl`` under a training prior whose centre "
                "grows with the log of the budget rather than half of it. Meant for "
                "deep loop counts, where the linear rule flattens into something close "
                "to uniform over the range and spends most forwards on inputs that "
                "converged after three loops. Keeps the ramp (P(r=1) < P(r=2)) and "
                "makes full depth rare rather than routine; see ``LOOP_PRIORS`` in "
                "praxis/halting/kl.py for the measured curves."
            ),
        ),
        "kl_log_reinject": Entry(
            partial(ReinjectedKLHalting, prior="log"),
            (
                "Geiping et al.'s recurrence under the log prior: the state starts "
                "from noise, every step re-reads the decoder input through an adapter, "
                "and at inference each position exits on its own against a scale set "
                "by the first refinement rather than the step out of noise."
            ),
        ),
    },
)
