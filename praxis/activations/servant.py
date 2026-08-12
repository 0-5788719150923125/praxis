import torch
from torch import Tensor

from praxis.activations.serpent import Serpent

# Maximum fractional swing of the frequency under test-time modulation. A fixed,
# model-agnostic constant (not a per-run knob): the live signal can bend each
# feature's frequency within a*(1 +/- MOD_MAX) and no further. Bounding the
# chirp keeps the measure -> frequency loop stable.
MOD_MAX: float = 0.5

# Numerical floor inside the log-energy.
ENERGY_EPS: float = 1e-6


class Servant(Serpent):
    """Serpent with a test-time-modulated frequency: a learnable chirp.

        s     = rms(x, over features)               # live per-token energy
        m     = tanh(log(s) - log_s_ref)            # centered test-time signal in (-1, 1)
        a_eff = a * (1 + MOD_MAX * tanh(v) * m)     # frequency breathes with energy
        y     = x + sin^2(a_eff*x) * a_eff/(a_eff^2 + eps^2) + g*sin(b*x)

    Serpent learns a *static* per-feature frequency ``a``. Servant lets that
    frequency move at inference with the energy of whatever token is passing
    through it. A velocity that varies across the signal is a frequency that
    changes - i.e. a chirp (see the paper's Definitions). By Parseval the
    per-token RMS is the token's total spectral power, so the harmonic term is
    driven by a genuinely spectral quantity, not an arbitrary statistic.

    The per-feature coupling ``v`` (the watchable "velocity") is zero-initialized,
    so at init ``a_eff == a`` and Servant is *exactly* Serpent; it anneals into
    test-time dependence as ``v`` leaves zero. ``log_s_ref`` centers the live
    energy signal and is materialized from the first batch's mean log-energy.

    The modulation reads only ``x`` (reduced over the feature axis, per token),
    so it is causal, instance-local, and needs no plumbing. The energy ``s`` is
    detached: a measurement that steers the frequency, not a path the input is
    trained through. ``tanh(v)`` bounds the per-feature coupling and ``MOD_MAX``
    bounds the swing; the ``1/a_eff`` floor reuses Serpent's smooth-rectified
    rectifier, so a near-zero modulated frequency cannot explode.
    """

    def _declare_extra_parameters(self) -> None:
        self._declare_parameter("v")
        self._declare_parameter("log_s_ref")

    def _initialize_extra_parameters(self, x: Tensor) -> None:
        feature_shape = x.shape[-1:]
        device, dtype = x.device, x.dtype
        # Velocity coupling starts at zero: Servant == Serpent at init.
        initial_v = torch.zeros(feature_shape, dtype=dtype, device=device)
        # Center the live energy signal on the first batch's mean log-energy.
        # Shape [1], never 0-dim: the schedule_free optimizer wrapper swaps
        # parameters via `x.view(torch.uint8).bitwise_xor_(...)`, which raises on
        # a 0-dim tensor. Broadcasting against [..., 1] is unaffected.
        s = x.detach().square().mean(dim=-1).clamp_min(ENERGY_EPS).sqrt()
        initial_ref = s.log().mean().to(dtype=dtype).reshape(1)
        self._materialize(("v", initial_v), ("log_s_ref", initial_ref))

    def _effective_frequency(self, a: Tensor, x: Tensor) -> Tensor:
        v = self._broadcast(self.v, x)
        # Live, per-token energy reduced over the feature axis (causal,
        # instance-local). Detached: a measurement, not a trained path.
        s = x.detach().square().mean(dim=-1, keepdim=True).clamp_min(ENERGY_EPS).sqrt()
        m = torch.tanh(s.log() - self.log_s_ref)
        # Frequency breathes with the signal: a learnable chirp. v=0 -> a_eff == a.
        swing = MOD_MAX * torch.tanh(v) * m
        if self.training:
            # Realized fractional frequency swing, stashed ON-DEVICE (no host
            # sync in the hot path) and read on the logging cadence. A plain
            # detached tensor, so it holds no graph and cannot go stale across
            # iterations the way an accumulated one would.
            self._swing = swing.detach().abs().mean()
        return a * (1.0 + swing)

    # -- diagnostics -------------------------------------------------------

    metric_descriptions = {
        "servant_coupling": {
            "description": (
                "Mean |tanh(v)| across features - how strongly the frequency is "
                "allowed to follow the signal. Zero-init by construction, so "
                "this IS the gate on the whole mechanism: flat at 0 means the "
                "model declined the chirp and Servant is still exactly Serpent."
            ),
            "chart": {
                "title": "Servant Chirp Coupling",
                "y_label": "|tanh(v)|",
                "y_scale": "linear",
                "group": "servant",
                "group_order": 93,
                "order": 10,
            },
        },
        "servant_coupling_std": {
            "description": (
                "Spread of the coupling ACROSS features. Distinguishes 'every "
                "feature chirps a little' from 'some features chirp and others "
                "stay static' - the latter is per-feature specialization, the "
                "former is a uniform change of activation shape."
            ),
            "chart": {
                "title": "Servant Coupling Spread",
                "y_label": "Std |tanh(v)|",
                "y_scale": "linear",
                "group": "servant",
                "order": 20,
            },
        },
        "servant_chirp": {
            "description": (
                "Mean REALIZED fractional frequency swing on live data, "
                "|a_eff/a - 1|, bounded by MOD_MAX. Differs from the coupling "
                "when the energy signal itself is flat: a feature can be fully "
                "coupled and still barely chirp if every token carries the same "
                "energy, which would say the measurement, not the mechanism, is "
                "the thing that failed."
            ),
            "chart": {
                "title": "Servant Realized Chirp",
                "y_label": "|a_eff/a - 1|",
                "y_scale": "linear",
                "group": "servant",
                "order": 30,
            },
        },
    }

    _swing = None

    def training_metrics(self) -> dict:
        if self.has_uninitialized_params():
            return {}
        with torch.no_grad():
            coupling = torch.tanh(self.v.detach())
            out = {
                "servant_coupling": float(coupling.abs().mean()),
                "servant_coupling_std": (
                    float(coupling.std()) if coupling.numel() > 1 else 0.0
                ),
            }
            if self._swing is not None:
                out["servant_chirp"] = float(self._swing)
        return out

    def extra_repr(self) -> str:
        return (
            "a_eff = a*(1 + %s*tanh(v)*tanh(log rms(x) - log_s_ref)), "
            "v zero-init (== Serpent at init)" % MOD_MAX
        )
