"""ExpertPoolLayer: the decoder-resident handle for the remote-expert pool.

A registered ``nn.Module`` so the swarm has a visible home in the model blueprint
(``print(model)`` / the Identity tab) - a place to "put" the pool in the stack,
sitting alongside the decoder's ``locals``.

PASSIVE OBSERVER (current branch). This layer is the identity in the forward pass
and is deliberately kept OUT of the decoder's routing experts (``locals`` /
``remotes``): the live pool (driven by ExpertPoolCallback) trains its experts on
real batches but does not contribute back to the model's activations yet.

STUB - contribute-back seam: when the integration lands, this layer's forward
will call ``ExpertPool.infer(hidden_states)`` and mix the swarm vote into the
residual stream (the RemoteLayer seam), at which point it joins the routing path
and contributing agents flip from OBSERVE to TRAINING. Until then it carries no
torch parameters - the experts live in the Node sidecar / browser peers.
"""

from __future__ import annotations

from torch import Tensor, nn


class ExpertPoolLayer(nn.Module):
    def __init__(self, profile_name: str, mixing: str, sidecar: bool, init_experts: int) -> None:
        super().__init__()
        self.profile = profile_name
        self.mixing = mixing
        self.sidecar = sidecar
        self.init_experts = init_experts

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        # Passive observer: identity. The pool runs out-of-band (callback); the
        # contribute-back path is the stub documented above.
        return hidden_states

    def extra_repr(self) -> str:
        # Reflect live capacity when a pool is active (the callback publishes to
        # the shared status); otherwise show the configured profile.
        from praxis.orchestration import status

        snap = status.snapshot()
        parts = [
            "observer",
            f"profile={self.profile}",
            f"mixing={self.mixing}",
            f"sidecar={self.sidecar}",
        ]
        alive = snap.get("experts_alive")
        if alive is not None:
            parts.append(f"experts={alive}")
        else:
            parts.append(f"init_experts={self.init_experts}")
        parts.append("contribute=STUB")
        return ", ".join(parts)
