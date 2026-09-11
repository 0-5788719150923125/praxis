"""Distance router: SMEAR with a parameter-distance loss between experts.

A diversity term pushes every expert's parameters away from expert 0's, so the
merged experts cannot collapse toward one geometry:

    L_div = -(1 / (N - 1)) * sum_{i >= 1} sum_params ||theta_i - theta_0||_2

Built on the modular router (praxis/routers/smear.py) and its causal routing. In
that router's base-plus-deviation basis expert ``e`` is ``base + delta_e``, so
``theta_i - theta_0 == delta_i - delta_0`` and the term reads the deviations
directly. It is parameter-only, so it rides ``router_aux_loss`` - collected once
per step, outside the recurrent forward - as VEAR's repulsion does.
"""

from typing import Any, Dict, List

import torch
from torch import Tensor

from praxis.routers.smear import SMEAR


class Distance(SMEAR):
    """SMEAR whose experts are pushed apart in parameter space."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.diversity_loss_coef = getattr(config, "diversity_loss_coef", 0.01)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(targets={len(self.targets)}, "
            f"num_experts={self.num_experts}, diversity_coef={self.diversity_loss_coef})"
        )

    def _expert_deviations(self) -> List[Tensor]:
        """Every target's deviations, ``[N, *param_shape]`` each."""
        out: List[Tensor] = []
        for wrapper in self.wrappers.values():
            out.append(torch.einsum("eor,eri->eoi", wrapper.lora_b, wrapper.lora_a))
        for pname in self._param_row:
            key = self._key(pname)
            if self._factored[pname]:
                b, a = self.deltas[key + "__b"], self.deltas[key + "__a"]
                out.append(torch.einsum("eor,eri->eoi", b, a))
            else:
                out.append(self.deltas[key])
        return out

    def diversity_loss(self) -> Tensor:
        """``-(1 / (N - 1)) * sum ||delta_i - delta_0||`` over experts and targets."""
        deviations = self._expert_deviations()
        loss = deviations[0].new_zeros(())
        for delta in deviations:
            flat = delta.reshape(delta.shape[0], -1)
            loss = loss - (flat[1:] - flat[:1]).norm(dim=-1).sum()
        return loss / (self.num_experts - 1)

    def router_aux_loss(self) -> Dict[str, Tensor]:
        if not self.training or self.num_experts < 2:
            return {}
        loss = self.diversity_loss()
        self._metrics["routing/diversity_loss"] = float(loss.detach())
        return {"distance_diversity": self.diversity_loss_coef * loss}
