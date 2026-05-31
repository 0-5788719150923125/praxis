"""Gated Schedule-Free: per-coordinate bias-variance control over averaging.

Subclasses ``pytorch_optimizer``'s :class:`ScheduleFreeWrapper`, overriding only
``step``. Schedule-Free averages the base iterate ``z`` (responsive = variance)
into the deployed average ``x`` (smooth = bias) with a scalar weight per step.
Here that weight becomes a per-coordinate **gate**: the bias-corrected gradient
SNR ``|EMA(grad)| / sqrt(EMA(grad^2))``, which is bounded in ``[0, 1]`` for free
(``E[g]^2 <= E[g^2]``). Coordinates with a consistent gradient (high SNR) admit
``z`` quickly (trust it); noisy coordinates hold ``x`` (denoise). Each parameter
thus picks its own point on the bias-variance axis from its own gradient
statistics, with no hyperparameter. Gate pinned to 1 is exactly the parent, so
this is a strict generalization (not a provably schedule-free method).

The extra per-parameter state (``m1``, ``m2``, ``t``) lives inside
``self.state[p]``, so the inherited ``state_dict`` (which dumps
``schedulefree_state`` wholesale) serializes it with no new keys; ``step``
lazy-inits it, so loading a plain schedule-free or pre-gate checkpoint is safe.
"""

import torch
from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.optimizer import ScheduleFreeWrapper

EPS = 1e-8


class GatedScheduleFree(ScheduleFreeWrapper):
    """Schedule-free averaging gated per-coordinate by gradient SNR."""

    def __init__(
        self,
        optimizer,
        momentum: float = 0.98,
        snr_decay: float = 0.99,
        gate_floor: float = 0.0,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        **kwargs,
    ):
        super().__init__(
            optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            r=r,
            weight_lr_power=weight_lr_power,
            **kwargs,
        )
        self.validate_range(snr_decay, "snr_decay", 0.0, 1.0, "[)")
        self.snr_decay = float(snr_decay)
        self.gate_floor = float(gate_floor)
        self._last_gate_mean: float = 1.0  # diagnostic

    def __str__(self) -> str:
        return "GatedScheduleFree"

    @property
    def gate_mean(self) -> float:
        return self._last_gate_mean

    def _gate(self, state) -> torch.Tensor:
        """Per-coordinate averaging gate in [0, 1]: bias-corrected grad SNR."""
        bc = 1.0 - self.snr_decay ** state["t"]
        m1 = state["m1"] / bc
        m2 = state["m2"] / bc
        g = m1.abs_().div_(m2.sqrt_().add_(EPS))
        return g.clamp_(min=self.gate_floor, max=1.0)

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise ValueError(
                "optimizer was not in train mode when step is called. "
                "call .train() before training"
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        d = self.snr_decay
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if "z" not in state:
                    state["z"] = p.clone()
                if "m1" not in state:  # lazy-init; safe for loaded checkpoints
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["t"] = 0
                # SNR moments of the gradient (evaluated at y).
                state["t"] += 1
                state["m1"].mul_(d).add_(grad, alpha=1.0 - d)
                state["m2"].mul_(d).addcmul_(grad, grad, value=1.0 - d)

                z = state["z"]
                self.apply_weight_decay(
                    z, grad, lr=group["lr"], weight_decay=self.weight_decay,
                    weight_decouple=True, fixed_decay=False,
                )
                self.apply_weight_decay(
                    p, grad, lr=group["lr"], weight_decay=self.weight_decay,
                    weight_decouple=True, fixed_decay=False,
                    ratio=1.0 - self.momentum,
                )
                p.lerp_(end=z, weight=1.0 - 1.0 / self.momentum)
                self.swap(z, p)

        self.optimizer.step()

        gates = []
        for group in self.param_groups:
            group["step"] = group["step"] + 1 if "step" in group else 1
            lr = group["lr"] * group.get("d", 1.0)
            lr_max = group["lr_max"] = max(lr, group.get("lr_max", 0))
            weight = (group["step"] ** group["lr"]) * (lr_max ** self.weight_lr_power)
            weight_sum = group["weight_sum"] = group.get("weight_sum", 0.0) + weight
            checkpoint = weight / weight_sum if weight_sum != 0.0 else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                gate = self._gate(state)
                gates.append(float(gate.mean()))
                z = state["z"]
                self.swap(z, p)
                # The only change from the parent: a per-coordinate gate on the
                # averaging weight (scalar checkpoint -> checkpoint * gate).
                p.lerp_(end=z, weight=checkpoint * gate)
                p.lerp_(end=state["z"], weight=1.0 - self.momentum)

        if gates:
            self._last_gate_mean = sum(gates) / len(gates)
        return loss
