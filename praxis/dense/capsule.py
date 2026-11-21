import math
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn

from praxis.dense.base import BaseDense

_GROUP = {"group": "capsules", "group_order": 41}
_EPS = 1e-8


def squash(s: Tensor, dims: Tuple[int, ...]) -> Tensor:
    """Sabour et al.'s norm nonlinearity: length ``|s|^2 / (1 + |s|^2)``,
    direction kept, norm taken jointly over ``dims``."""
    n2 = s.pow(2).sum(dim=dims, keepdim=True)
    return s * (n2 / (1 + n2)) * torch.rsqrt(n2 + _EPS)


class HarmonicCapsules(BaseDense):
    """A capsule layer with dynamic routing (Sabour et al. 2017,
    arXiv:1710.09829) whose poses are phase harmonics.

    Each input coordinate is read as a phase ``theta = alpha * x`` (learned
    frequency starting at 1, as in Kuramoto Attention and Serpent). The phases
    split into ``hidden_size // atoms`` primary capsules; primary ``i`` has the
    complex ``[harmonics, atoms]`` pose ``M_i[n, c] = exp(i n theta_ic)``. Each
    pair ``(i, j)`` has a complex ``[atoms, atoms]`` transform applied on the
    right, vote ``u_ij = M_i W_ij``. Routing follows the reference code: logits
    start at 0, ``c_i = softmax_j``, ``s_j = sum_i c_ij u_ij``, ``v_j =
    squash(s_j)``, logits gain ``Re <u_ij, v_j>``, for ``iterations`` rounds.
    A linear map reads every output pose back to the hidden size; the squash is
    the nonlinearity, so ``--activation-type`` is unused.

    Shifting every phase by ``phi`` multiplies pose row ``n`` by
    ``exp(i n phi)``, a left multiplication that commutes with the transforms
    and leaves squash and agreement unchanged. Routing therefore sees only the
    inputs' relative phases, and the output poses carry the common phase. The
    reference adds a bias to ``s_j``; it is left out here because a fixed bias
    does not rotate and would break that symmetry.
    """

    metric_descriptions = {
        "capsule_routing_entropy": {
            "description": (
                "Mean entropy of each part's coupling over the wholes, over "
                "log(outputs). 1 = uniform: routing by agreement does nothing. "
                "Lower = parts commit to wholes."
            ),
            "chart": {
                "title": "Routing Entropy",
                "y_label": "fraction of uniform",
                "order": 10,
                **_GROUP,
            },
        },
        "capsule_length": {
            "description": (
                "Mean output capsule length after the squash, in (0, 1): the "
                "capsule's existence. Parts whose votes agree add coherently "
                "and give long capsules."
            ),
            "chart": {
                "title": "Capsule Length",
                "y_label": "length",
                "order": 20,
                **_GROUP,
            },
        },
        "capsule_readout_harmonic": {
            "description": (
                "Energy-weighted mean harmonic index of the readout weights "
                "(1 = reads only the fundamental, harmonics = only the highest). "
                "Starts near the middle."
            ),
            "chart": {
                "title": "Readout Harmonic",
                "y_label": "harmonic index",
                "order": 30,
                **_GROUP,
            },
        },
        "capsule_read_frequency": {
            "description": (
                "Mean |alpha|: radians of phase per unit of normalized input. "
                "Higher = the input is read as faster-turning phases."
            ),
            "chart": {
                "title": "Capsule Read Frequency",
                "y_label": "radians per unit",
                "order": 40,
                **_GROUP,
            },
        },
    }
    per_layer_metrics = frozenset(metric_descriptions)

    def __init__(
        self,
        config: Any,
        atoms: int = 8,
        harmonics: int = 3,
        outputs: Optional[int] = None,
        iterations: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        if hidden % atoms:
            raise ValueError(
                f"capsules take {atoms} phases each, so hidden_size must be a "
                f"multiple of {atoms} (got {hidden})"
            )
        self.atoms = atoms
        self.harmonics = harmonics
        self.inputs = hidden // atoms
        self.outputs = outputs or self.inputs
        self.iterations = iterations

        self.frequency = nn.Parameter(torch.ones(hidden))
        # Complex transforms as (real, imag); E|W|^2 = 1 / atoms puts each vote
        # entry at unit scale, since the poses are unit phasors.
        std = 1.0 / math.sqrt(2 * atoms)
        self.transform = nn.Parameter(
            torch.randn(2, self.inputs, self.outputs, atoms, atoms) * std
        )
        self.readout = nn.Linear(self.outputs * harmonics * 2 * atoms, hidden)
        self.register_buffer(
            "_stats", torch.full((2,), float("nan")), persistent=False
        )

    def route(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Output poses ``[..., outputs, harmonics, 2 * atoms]`` (real halves,
        then imaginary) and the final coupling ``[..., inputs, outputs]``."""
        lead = inputs.shape[:-1]
        # At least float32: phases and routing logits lose too much in half.
        dtype = torch.promote_types(inputs.dtype, torch.float32)
        theta = (inputs.to(dtype) * self.frequency.to(dtype)).reshape(
            -1, self.inputs, 1, self.atoms
        )
        n = torch.arange(1, self.harmonics + 1, device=theta.device)
        angle = theta * n.view(1, 1, -1, 1)
        # [N, inputs, harmonics, 2 * atoms]
        pose = torch.cat([angle.cos(), angle.sin()], dim=-1)

        # (a + ib)(c + id) as a real matmul on stacked halves.
        wr, wi = self.transform.to(dtype)
        w = torch.cat(
            [torch.cat([wr, wi], dim=-1), torch.cat([-wi, wr], dim=-1)], dim=-2
        )
        # [N, outputs, inputs, harmonics * 2 * atoms]: output-major, so both
        # routing contractions are batched matmuls with no temporaries.
        N = pose.shape[0]
        votes = torch.einsum("nikm,ijmd->njikd", pose, w).reshape(
            N, self.outputs, self.inputs, -1
        )

        logits = votes.new_zeros(N, self.outputs, 1, self.inputs)
        for step in range(self.iterations):
            coupling = logits.softmax(dim=1)
            v = squash(coupling @ votes, dims=(-1,))  # [N, outputs, 1, f]
            if step + 1 < self.iterations:
                logits = logits + (votes @ v.transpose(-1, -2)).transpose(-1, -2)

        coupling = coupling.reshape(N, self.outputs, self.inputs).transpose(1, 2)
        v = v.reshape(N, self.outputs, self.harmonics, -1)
        if self.training:
            self._record(coupling, v)
        return v.reshape(*lead, *v.shape[1:]), coupling.reshape(
            *lead, *coupling.shape[1:]
        )

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        poses, _ = self.route(inputs)
        poses = poses.flatten(-3).to(self.readout.weight.dtype)
        return self.readout(poses).to(inputs.dtype)

    def _record(self, coupling: Tensor, poses: Tensor) -> None:
        """Routing entropy and length into a buffer, in-graph: no host sync."""
        with torch.no_grad():
            entropy = -(coupling * coupling.clamp_min(1e-12).log()).sum(-1)
            entropy = entropy.mean() / math.log(max(self.outputs, 2))
            length = poses.pow(2).sum(dim=(-2, -1)).sqrt().mean()
            self._stats.copy_(torch.stack([entropy, length]))

    def training_metrics(self) -> dict:
        with torch.no_grad():
            entropy, length = self._stats.tolist()
            energy = (
                self.readout.weight.float()
                .pow(2)
                .view(-1, self.outputs, self.harmonics, 2 * self.atoms)
                .sum(dim=(0, 1, 3))
            )
            index = torch.arange(1, self.harmonics + 1, device=energy.device)
            return {
                "capsule_routing_entropy": None if math.isnan(entropy) else entropy,
                "capsule_length": None if math.isnan(length) else length,
                "capsule_readout_harmonic": float(
                    (energy * index).sum() / energy.sum().clamp_min(_EPS)
                ),
                "capsule_read_frequency": self.frequency.abs().mean().item(),
            }
