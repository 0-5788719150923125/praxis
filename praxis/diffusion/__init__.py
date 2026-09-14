"""Diffusion objectives: the sequence is corrupted and reconstructed in place,
rather than extended one position at a time.

Selected with ``--diffusion-type``. When one is active the model is NOT causal
(``config.causal`` stays False), labels are UNSHIFTED, and generation runs an
iterative refinement loop instead of next-token sampling. Nothing here is
compatible with MTP, speculative decoding or a KV cache, all of which assume a
left-to-right factorisation.
"""

from functools import partial

from praxis import registry
from praxis.diffusion.masked import MaskedDiffusion
from praxis.registry import Entry

registry.declare(
    "diffusion",
    title="Diffusion objectives",
    doc=(
        "Non-autoregressive objectives. The model sees the whole sequence, "
        "corrupted, and reconstructs it; generation is iterative refinement. "
        "Unset means ordinary next-token training."
    ),
    entries={
        "masked": Entry(
            MaskedDiffusion,
            (
                "Absorbing-state masked diffusion (MDLM/LLaDA). A per-row "
                "corruption level ``t ~ U(eps, 1)`` replaces each position with "
                "the mask id independently, and the loss is the masked "
                "cross-entropy weighted by ``1/t`` - the unbiased ELBO "
                "estimator. No timestep conditioning: the mask fraction is "
                "already visible in the input."
            ),
        ),
        "masked_coarse": Entry(
            partial(MaskedDiffusion, eps=0.1),
            (
                "``masked`` with the corruption level floored at 0.1 instead of "
                "1e-3. Trades the near-clean end of the schedule, where 1/t is "
                "largest and the gradient noisiest, for a tighter weight range."
            ),
        ),
    },
)

__all__ = ["MaskedDiffusion"]
