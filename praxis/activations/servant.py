import torch
from torch import Tensor

from praxis.activations.serpent import ENERGY_EPS, SIGNAL_SIGMAS, Serpent

# Maximum fractional swing of the frequency under test-time modulation. A fixed,
# model-agnostic constant (not a per-run knob): the live signal can bend each
# feature's frequency within a*(1 +/- MOD_MAX) and no further. Bounding the
# chirp keeps the measure -> frequency loop stable.
MOD_MAX: float = 0.5


class Servant(Serpent):
    """Serpent with a test-time-modulated frequency: a learnable chirp.

        s     = rms(x, over features)               # live per-token energy
        z     = (log s - mean_t) / std_t            # standardized on RUNNING stats
        m     = tanh(z / SIGNAL_SIGMAS)             # test-time signal in (-1, 1)
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
    test-time dependence as ``v`` leaves zero.

    The energy signal is standardized against RUNNING mean and variance of the
    live log-energy (``Serpent._energy_signal``), not centred on a scalar frozen
    at init. Read that method before touching this one: the frozen-reference
    version saturated its tanh within 4k steps, which turned the whole mechanism
    into a static rescaling of ``a``, and there was no gradient path back out.

    The modulation reads only ``x`` (reduced over the feature axis, per token),
    so it is causal, instance-local, and needs no plumbing. The energy ``s`` is
    detached: a measurement that steers the frequency, not a path the input is
    trained through. ``tanh(v)`` bounds the per-feature coupling and ``MOD_MAX``
    bounds the swing; the ``1/a_eff`` floor reuses Serpent's smooth-rectified
    rectifier, so a near-zero modulated frequency cannot explode.
    """

    def _declare_extra_parameters(self) -> None:
        self._declare_parameter("v")
        self._declare_energy_stats()

    def _initialize_extra_parameters(self, x: Tensor) -> None:
        feature_shape = x.shape[-1:]
        device, dtype = x.device, x.dtype
        # Velocity coupling starts at zero: Servant == Serpent at init.
        initial_v = torch.zeros(feature_shape, dtype=dtype, device=device)
        self._materialize(("v", initial_v))
        self._initialize_energy_stats(x)

    def _effective_frequency(self, a: Tensor, x: Tensor) -> Tensor:
        v = self._broadcast(self.v, x)
        # A REAL training forward, as opposed to the dashboard's activation-curve
        # probe: that walks every activation module under `torch.no_grad()` with
        # a `linspace(-6, 6)` tiled across features, and deliberately does NOT
        # change train/eval mode because it races the trainer. While
        # `self.training` was the only guard, the probe both advanced the running
        # statistics and overwrote the metric stash with the numbers from a
        # synthetic ramp - visible in the logs as a bimodal `servant_chirp`, ~6%
        # of points carrying the probe's value.
        live = self.training and torch.is_grad_enabled()
        m = self._energy_signal(x, live)
        # Frequency breathes with the signal: a learnable chirp. v=0 -> a_eff == a.
        swing = MOD_MAX * torch.tanh(v) * m
        if live:
            self._stash(swing, m)
        return a * (1.0 + swing)

    def _stash(self, swing: Tensor, m: Tensor) -> None:
        """Realized chirp statistics, ON-DEVICE (no host sync in the hot path).

        Plain detached tensors, so they hold no graph and cannot go stale across
        iterations the way accumulated ones would.

        ``_swing`` is the DISPERSION of the swing across tokens, not its
        magnitude, and that distinction is the whole diagnostic. A chirp is a
        frequency that MOVES; a magnitude reads maximal precisely when the swing
        is a large constant, which is the failure this activation actually hit.
        """
        token_dims = tuple(range(swing.dim() - 1))
        tokens = 1
        for d in token_dims:
            tokens *= swing.shape[d]
        detached = swing.detach()
        self._swing = (
            detached.std(dim=token_dims).mean()
            if tokens > 1
            else torch.zeros_like(detached.mean())
        )
        self._signal = m.detach().abs().mean()

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
                "REALIZED chirp: the spread of a_eff/a ACROSS TOKENS, per "
                "feature, averaged over features. A chirp is a frequency that "
                "MOVES, so dispersion is the measure and magnitude is not - "
                "mean |a_eff/a - 1| reads maximal exactly when the swing is a "
                "large CONSTANT, which is a static rescaling of `a` wearing a "
                "chirp's name. This card read the magnitude until 2026-08-22 "
                "and reported a healthy rising chirp through 20k steps of a run "
                "that was not chirping at all; earlier values do not compare. "
                "Read it against the signal card below: near-zero chirp with a "
                "saturated signal is the mechanism failing, near-zero chirp with "
                "a graded signal is the model declining it."
            ),
            "chart": {
                "title": "Servant Realized Chirp",
                "y_label": "Std_tokens(a_eff/a)",
                "y_scale": "linear",
                "group": "servant",
                "order": 30,
            },
        },
        "servant_signal": {
            "description": (
                "Mean |m|, how hard the standardized live-energy signal is "
                "pushed into its tanh. This is the HEALTH of the measurement "
                "itself and it has one reading that means death: 1.0. A "
                "saturated m is a constant, the frequency stops moving, and "
                "there is no gradient path back - so the useful range is the "
                "graded middle, roughly 0.3 to 0.7. The predecessor centred the "
                "signal on a scalar frozen at init instead of standardizing it, "
                "and sat at 0.999 from step 4000 onward with nothing on any "
                "card able to say so."
            ),
            "chart": {
                "title": "Servant Signal Saturation",
                "y_label": "mean |m|",
                "y_scale": "linear",
                "group": "servant",
                "order": 40,
            },
        },
    }

    _swing = None
    _signal = None

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
            if self._signal is not None:
                out["servant_signal"] = float(self._signal)
        return out

    def extra_repr(self) -> str:
        return (
            "a_eff = a*(1 + %s*tanh(v)*tanh(z/%s)), z = (log rms(x) - mean)/std "
            "on running stats, v zero-init (== Serpent at init)"
            % (MOD_MAX, SIGNAL_SIGMAS)
        )
