import math
from typing import List, Tuple

import torch
from torch import Tensor

from praxis.activations.serpent import INV_FLOOR_EPS, Serpent

# Ceiling on the recurrence. Like `depth`, this is an upper bound on available
# compute, not the amount that gets used - the gates decide that per feature,
# per token. Fixed and model-agnostic.
#
# 8, and the ceiling is memory. The trajectory is solved without autograd (see
# `forward`), which removed the nonlinearity's graph: measured ~16.5 -> ~6.1 MB
# per step per call at [32, 64, 111]. What remains is the gate graph, which is
# still O(MAX_STEPS) because the halting parameters learn from the task. At the
# real batch of 64 and ~10 activation calls per forward, that is ~120 MB per
# unit of MAX_STEPS: ~1.3 GB here, and 16 would now fit if the gates ever
# saturate the ceiling.
#
# Do NOT read this as "so we could iterate to convergence like a DEQ". Serpent's
# fixed points are the lattice sin(a*x) = 0, i.e. x = k*pi/a. Forced to
# converge, it collapses 4096 distinct inputs onto 8 lattice sites - a
# quantizer that destroys the feature. Ouroboros must stay TRUNCATED iteration
# (ACT/PonderNet), never converged iteration. See next/ouroboros.md.
MAX_STEPS: int = 8

# Maximum fractional swing of the frequency under the carried state (Servant's
# constant, reused for the same reason: bounding the chirp keeps the loop stable).
MOD_MAX: float = 0.5

# Numerical floor inside the log-energy and the convergence ratio.
ENERGY_EPS: float = 1e-6

# Hard-concrete gate constants (Louizos et al., "Learning Sparse Neural Networks
# through L0 Regularization", arXiv 1712.01312). The stretched binary concrete
# puts real mass on exactly-0 and exactly-1 while staying differentiable, and it
# admits a closed form for P(z > 0) - which is what makes the step budget an
# analytic quantity rather than a sampled estimate. Their published values.
HC_BETA: float = 2.0 / 3.0
HC_GAMMA: float = -0.1
HC_ZETA: float = 1.1

# P(z > 0) = sigmoid(logit + HC_OFFSET). Derived from the constants above, not
# a knob.
HC_OFFSET: float = -HC_BETA * math.log(-HC_GAMMA / HC_ZETA)

# Gate-logit init. Step 0 saturates open, every later step saturates closed, so
# at eval the model *is* Serpent at init and the expected step count starts at
# ~1.0 - the budget the baseline already spends.
OPEN_INIT: float = 6.0
CLOSED_INIT: float = -5.0

# Live accounting. Every Ouroboros instance pushes its survival curve here
# during training; OuroborosBudget drains the stack once per forward and turns
# it into the Lagrangian and the exit-depth distribution. A module-level stack
# (rather than a reference the regularizer holds) counts exactly the calls that
# *executed* - depth-conditional paths that did not run this pass contribute
# nothing - and keeps the regularizer from registering the activations'
# parameters twice.
#
# Each entry is ``(survival, spread)``:
#   survival [MAX_STEPS]  mean P(still open at step k), carries the gradient
#   spread   [3]          detached (sum, sum-of-squares, count) of per-feature
#                         step totals - pooling SUMS rather than means is what
#                         lets call sites of different width combine.
_STEP_COUNTS: List[Tuple[Tensor, Tensor]] = []

# Accounting stays off unless a budget regularizer turns it on, so a config that
# selects `ouroboros` without the budget cannot silently accumulate live graphs.
_ACCOUNTING: bool = False

# Defensive cap: if something drains late, drop the oldest rather than hold an
# unbounded number of graphs.
_STACK_CAP: int = 4096


def enable_accounting() -> None:
    """Begin recording per-instance step counts (called by OuroborosBudget)."""
    global _ACCOUNTING
    _ACCOUNTING = True


def drain_step_counts() -> List[Tuple[Tensor, Tensor]]:
    """Take and clear everything recorded since the last drain."""
    out = list(_STEP_COUNTS)
    _STEP_COUNTS.clear()
    return out


def _feature_mean(t: Tensor) -> Tensor:
    """Average over every axis but the last, keeping the feature axis."""
    if t.dim() == 1:
        return t
    return t.mean(dim=tuple(range(t.dim() - 1)))


def _feature_rms(t: Tensor) -> Tensor:
    """Per-feature RMS, reduced over every other axis."""
    return _feature_mean(t.detach().square()).clamp_min(ENERGY_EPS).sqrt()


class Ouroboros(Serpent):
    """Serpent applied recurrently, with a per-feature, per-token gate that
    lets a feature stop iterating once it has converged.

        h_0    = 0,  open_0 = 1
        a_eff  = a * (1 + MOD_MAX * tanh(w) * h_k)
        y      = serpent(x, a_eff, b, g)
        logit  = u_k + p * m(x) + q * conv(x, y)
        z_k    = hard_concrete(logit)
        open_k = open_{k-1} * z_k          # closed stays closed
        x      = x + open_k * (y - x)
        h_{k+1}= tanh(h_k + (y - x))

    WHY THE GATE IS THE POINT. A pointwise map applied N times with nothing
    entering the loop is still a pointwise map: whatever f_N o ... o f_1 you
    learn is one fixed 1-D function per feature, reproducible by a single
    spline at 1/N the cost. Carrying ``h`` does not fix this, because ``h`` is
    itself a function of the same scalar. Two things here escape that:

    ``m`` is Servant's per-token RMS energy, reduced over the *feature* axis -
    so the trajectory depends on the whole token vector, not on x_f alone. And
    the hard-concrete gate is stochastic during training, so the map is a
    distribution rather than a function. Without them this class would be an
    expensive way to write Serpent.

    ``conv = tanh(|y - x| / (|x| + eps))`` is the per-feature convergence
    measure: how much this step still moved the feature, relative to its own
    scale. It is the signal a feature uses to notice it has arrived. Both ``m``
    and ``conv`` are detached - measurements that steer the gate, not paths the
    input is trained through.

    IDENTITY AT INIT. ``p``, ``q`` and ``w`` are zero-init and ``u_0`` / ``u_k``
    saturate the gate open / closed, so at eval Ouroboros reproduces Serpent to
    float rounding (~3e-8; the gated update writes ``x + 1*(y - x)`` rather than
    ``y``) until those couplings leave zero. In training the step-0 gate is
    drawn, so it lands on exactly 1.0 for ~98% of draws and slightly below for
    the rest - the one real departure from Servant's identity-at-init
    discipline, and the price of having a stochastic gate at all.

    THE BUDGET. sum_k P(open through step k) is the expected number of
    activation-steps this feature spends, in closed form. OuroborosBudget holds
    the mean of that at 1.0 - exactly what one Serpent costs - so the loop can
    only pay for a deep feature by making some other feature shallow. Any win
    is reallocation, not extra compute.

    See next/ouroboros.md.
    """

    # -- parameters --------------------------------------------------------

    def _declare_extra_parameters(self) -> None:
        self._declare_parameter("u")  # [MAX_STEPS, D] per-step gate bias
        self._declare_parameter("p")  # [D] coupling to token energy
        self._declare_parameter("q")  # [D] coupling to convergence
        self._declare_parameter("w")  # [D] coupling of carried state to frequency
        self._declare_parameter("log_s_ref")  # scalar energy reference

    def _initialize_extra_parameters(self, x: Tensor) -> None:
        feature_shape = x.shape[-1:]
        device, dtype = x.device, x.dtype

        # Step 0 open, every later step closed: one activation at init.
        initial_u = torch.full(
            (MAX_STEPS,) + tuple(feature_shape),
            CLOSED_INIT,
            dtype=dtype,
            device=device,
        )
        initial_u[0].fill_(OPEN_INIT)

        zeros = torch.zeros(feature_shape, dtype=dtype, device=device)
        s = x.detach().square().mean(dim=-1).clamp_min(ENERGY_EPS).sqrt()
        # Shape [1], never 0-dim: the schedule_free wrapper swaps parameters via
        # `x.view(torch.uint8).bitwise_xor_(...)`, and a 0-dim tensor cannot be
        # viewed as a different element size. Broadcasting is unaffected.
        initial_ref = s.log().mean().to(dtype=dtype).reshape(1)

        self._materialize(
            ("u", initial_u),
            ("p", zeros.clone()),
            ("q", zeros.clone()),
            ("w", zeros.clone()),
            ("log_s_ref", initial_ref),
        )

    # -- pieces ------------------------------------------------------------

    def _energy_signal(self, x: Tensor) -> Tensor:
        """Servant's centered per-token log-energy, in (-1, 1). Detached."""
        s = x.detach().square().mean(dim=-1, keepdim=True).clamp_min(ENERGY_EPS).sqrt()
        return torch.tanh(s.log() - self.log_s_ref)

    def _serpent_step(self, x: Tensor, a_eff: Tensor, b: Tensor, g: Tensor) -> Tensor:
        inv_a = a_eff / (a_eff * a_eff + INV_FLOOR_EPS * INV_FLOOR_EPS)
        return x + torch.sin(a_eff * x).square() * inv_a + g * torch.sin(b * x)

    def _hard_concrete(self, logit: Tensor) -> Tensor:
        """Stretched binary concrete, clamped to [0, 1]. Exact 0 and exact 1 are
        both attainable, so a closed gate really does stop the feature."""
        if self.training:
            u = torch.rand_like(logit).clamp_(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((u.log() - (-u).log1p() + logit) / HC_BETA)
        else:
            s = torch.sigmoid(logit)
        return (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0.0, 1.0)

    # -- forward -----------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Solve the recurrence without autograd, then reattach a gradient.

        The trajectory runs under ``no_grad``, so no step stores intermediates
        for backward and memory stops scaling with MAX_STEPS. The gradient is
        rebuilt from a SINGLE differentiable step evaluated at the live input -
        the Jacobian-free / one-step approximation implicit models are trained
        with (Fung et al., "JFB", 2202.08587; Geng et al.'s phantom gradient,
        2111.05177), which drops the ``(I - J)^-1`` factor of the exact implicit
        gradient and is still a descent direction.

        The surrogate is written so the FORWARD VALUE IS EXACT - it contributes
        ``surrogate - surrogate.detach()``, which is identically zero - while the
        backward comes entirely from it. Crucially the surrogate is a function of
        the live ``x``, not of the detached trajectory: an activation whose
        output did not depend on its input would pass no gradient upstream and
        every layer before it would go dark.

        The gate is the *realized step count* ``sum_k open_k``, so the surrogate
        reads "taking n steps moves this feature n times as far as one step
        does". At init exactly one step is open, so the surrogate is exactly one
        Serpent evaluation and BOTH the value and the gradient are exact - the
        approximation only switches on as the loop is actually recruited.
        """
        a = self._broadcast(self.a, x)
        b = self._broadcast(self.b, x)
        g = self._broadcast(self.g, x)
        p = self._broadcast(self.p, x)
        q = self._broadcast(self.q, x)
        w = self._broadcast(self.w, x)
        tanh_w = torch.tanh(w)

        m = self._energy_signal(x)

        state = x.detach()
        h = torch.zeros_like(state)
        gate_open = torch.ones_like(state)
        steps_open = torch.zeros_like(state)  # realized step count, carries grad
        reach = torch.ones_like(state)  # P(still open through step k)
        per_step: List[Tensor] = []
        per_token: List[Tensor] = []

        for step in range(MAX_STEPS):
            # The expensive part - the nonlinearity and its trajectory - is
            # graph-free. `conv` and `h` were already detached measurements, so
            # nothing is lost by solving here.
            with torch.no_grad():
                a_eff = a * (1.0 + MOD_MAX * tanh_w * h)
                y = self._serpent_step(state, a_eff, b, g)
                delta = y - state
                conv = torch.tanh(delta.abs() / (state.abs() + ENERGY_EPS))

            # The gate DOES carry a graph, but only over logits computed from
            # detached inputs - a few small tensors per step, not the sin/square
            # chain. This is what lets the halting parameters learn from the
            # task rather than from the budget term alone.
            logit = self._broadcast(self.u[step], state) + p * m + q * conv
            gate_open = gate_open * self._hard_concrete(logit)
            steps_open = steps_open + gate_open

            with torch.no_grad():
                state = state + gate_open * delta
                h = torch.tanh(h + delta)

            reach = reach * torch.sigmoid(logit + HC_OFFSET)
            per_step.append(_feature_mean(reach))
            # Same survival mass reduced on the OTHER axis: over features,
            # keeping tokens. Depth can vary between features (specialization,
            # the original hypothesis) or between tokens (adaptive compute per
            # position), and averaging over tokens before recording made the
            # second invisible - most of the observed depth variance lives
            # there, so it needs its own reduction rather than an inference.
            per_token.append(reach.mean(dim=-1))

        if _ACCOUNTING and self.training:
            self._record(torch.stack(per_step), torch.stack(per_token), x, state)

        # One differentiable step, at the live input and at the frequency the
        # trajectory settled into. `h` is detached (it always was - it is built
        # from detached deltas), so this costs nothing, but it keeps `w` in the
        # differentiable path: evaluating the surrogate at h=0 instead would
        # leave `w` with no gradient at all, silently frozen at its zero init,
        # which would delete the loop's state-dependence entirely. Evaluating
        # the one grad-carrying step at the solved state is also what JFB does.
        a_eff = a * (1.0 + MOD_MAX * tanh_w * h)
        surrogate = x + steps_open * (self._serpent_step(x, a_eff, b, g) - x)
        return state + (surrogate - surrogate.detach())

    def _record(
        self, curve: Tensor, token_curve: Tensor, x: Tensor, state: Tensor
    ) -> None:
        """Push this call's survival curve. ``curve`` is [MAX_STEPS, D]: row k
        is each feature's probability of still being open at step k, averaged
        over tokens. Keeping the feature axis this long is what makes the exit
        distribution and its spread readable - collapsing to a scalar here
        would leave only the mean, which cannot tell 'every feature half-
        commits' from 'features have split into deep and shallow groups'.

        Also records per-feature GAIN (output RMS / input RMS). Because the map
        drifts rather than converging, a feature that runs more steps exits
        louder, so depth and gain are entangled by construction. Pooling the
        cross-moment here lets the regularizer report their correlation - which
        is what separates "deep features compute more" from "deep features are
        just louder", the failure mode that would otherwise fake a positive
        result on ``ouroboros_steps_std``."""
        if len(_STEP_COUNTS) >= _STACK_CAP:
            _STEP_COUNTS.pop(0)
        steps = curve.sum(dim=0).detach()  # [D] expected steps per feature
        gain = _feature_rms(state) / _feature_rms(x)  # [D]
        count = torch.full_like(steps.sum(), steps.numel())
        # Same quantity along the token axis: expected depth per token, feature-
        # averaged. Its spread answers a different question from `steps`' spread.
        token_steps = token_curve.sum(dim=0).detach()  # [B, T]
        token_count = torch.full_like(steps.sum(), token_steps.numel())
        # Summed moments (not means) so call sites of different width pool by
        # count: [sum s, sum s^2, sum g, sum g^2, sum s*g, n_features,
        #         sum t, sum t^2, n_tokens].
        spread = torch.stack(
            [
                steps.sum(),
                steps.square().sum(),
                gain.sum(),
                gain.square().sum(),
                (steps * gain).sum(),
                count,
                token_steps.sum(),
                token_steps.square().sum(),
                token_count,
            ]
        )
        _STEP_COUNTS.append((curve.mean(dim=1), spread))

    def extra_repr(self) -> str:
        return (
            "max_steps=%d, per-feature hard-concrete halting, "
            "p/q/w zero-init (== Serpent at init)" % MAX_STEPS
        )
