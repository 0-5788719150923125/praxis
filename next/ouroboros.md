# Ouroboros: a recurrent activation with per-feature halting

Serpent applied N times, where each feature decides per token how many of those
times it actually takes. Snake -> Serpent -> Servant -> Ouroboros: the snake
eating its tail.

Files: `praxis/activations/ouroboros.py`, `praxis/losses/ouroboros_budget.py`,
`experiments/abstractinator-l.yml`.

## The collapse argument, which is the whole design constraint

The obvious version of this idea does not work, and the reason is worth keeping
because it kills most variants on sight.

A pointwise map applied N times, with nothing entering the loop from outside,
is still a pointwise map. Whatever `f_N o ... o f_1` you learn is one fixed 1-D
function per feature, and a single spline (KAN, Padé/rational unit) represents
it in one step at 1/N the compute. Learnable per-step gains do not help: they
are parameters, identical for every token, so the composition is still a fixed
scalar function. Carrying a hidden state does not help either, because the
state is itself derived from the same scalar - the trajectory is determined by
`x_f`, so the endpoint is a function of `x_f`. Even *halting* does not help on
its own: if the halt decision is a deterministic function of the trajectory,
the result is a piecewise function of `x_f`, which is still a function of
`x_f`.

So the loop has content only if something enters it that is not the scalar.
Two things do here:

1. **Cross-feature signal.** The gate logit reads Servant's per-token RMS
   energy, reduced over the feature axis. The number of steps a feature takes
   then depends on the whole token vector. Causal, instance-local, detached, no
   plumbing - Servant already established that this is available for free.
2. **Stochasticity.** The hard-concrete gate is sampled during training, so the
   activation is a distribution rather than a function. This is also what makes
   the L0 term meaningful rather than decorative.

Remove either and Ouroboros degenerates to an expensive Serpent.

## Mechanism

    h_0    = 0,  open_0 = 1
    a_eff  = a * (1 + MOD_MAX * tanh(w) * h_k)
    y      = serpent(x, a_eff, b, g)
    conv   = tanh(|y - x| / (|x| + eps))        # detached, per feature
    m      = tanh(log rms_D(x) - log_s_ref)     # detached, per token
    logit  = u_k + p * m + q * conv
    z_k    = hard_concrete(logit)
    open_k = open_{k-1} * z_k                   # closed stays closed
    x      = x + open_k * (y - x)
    h_{k+1}= tanh(h_k + (y - x))

`conv` is the convergence measure - how much this step still moved the feature
relative to its own scale. It is the signal a feature uses to notice it has
arrived, and it is the per-feature analogue of what KL halting measures at the
level of the whole latent (`praxis/halting/kl.py`).

The cumulative product is what makes halting *monotone*: a feature that closes
cannot reopen at a later step, so "stop" means stop rather than "skip this one".
Hard concrete (Louizos et al. 1712.01312) is the right gate because it puts real
mass on exactly 0 and exactly 1 - so a closed gate genuinely contributes
nothing - while staying differentiable, and because `P(z > 0)` has a closed
form, which is what turns the step count into an analytic quantity.

## Result: attempt 1 (target 1.0) - uniform collapse

Run 865ed0387, read at step 4305 over 431 samples:

    ouroboros_extra_frac   0.0322 -> 0.0027      (12x down, monotone)
    ouroboros_exit_1       0.9673 -> 0.996
    ouroboros_steps        1.033  -> 1.001
    ouroboros_steps_std    ~3e-4 throughout
    ouroboros_lambda       0.693  -> 0.881 (rising)

The gates closed. Not "never recruited" - actively shut, from the init value
down, with `steps_std` pinned at zero the whole way. A UNIFORM collapse back to
Serpent, not a redistribution. BPB tracked -k, which is the consequence rather
than a coincidence.

The mechanism: the budget's gradient is exact and hits every gate at once,
while the task's "this feature earns another step" signal is diffuse and routed
through a first-order surrogate. Exact beats approximate. At target 1.0 the only
way to win was reallocation, and reallocation never got traction.

A second reading matters here. The model already runs adaptive depth at the
BLOCK level with full sequence context (`halting_type: kl`), and in the same
fetch that halting was live and differentiated - exits spread across r_3 (300),
r_4 (837), r_6 (174). So the model does use adaptive compute; it just declines
to buy it at feature granularity, where strictly less information is available.
That is the sharpest evidence yet for the ceiling argument below.

## The per-token ceiling

Ouroboros is pointwise per (token, feature). Its *input* is not context-free -
it sits after attention, so `x_t` already depends on the whole prefix, and the
halting decision is conditioned on a contextualized vector. But it cannot
CREATE cross-position dependencies: everything it can condition on, attention
already computed.

So Ouroboros can only ever be a compute-ALLOCATION mechanism over an existing
representation, never new modeling capacity. If allocation is not the binding
constraint at this scale - and attempt 1 says it is not - there is nothing for
it to win. Any future version that wants more has to give the gate something
attention has not already provided, e.g. a causal running statistic over
positions rather than a per-token one. That is a different mechanism, not a
tuning change.

## The budget, and why it is not a knob

`sum_k P(open through step k)` is the expected number of activation-steps a
feature spends. `OuroborosBudget` holds its mean at **1.0**.

That number is not tuned. It is exactly what one Serpent costs, so Ouroboros
gets `MAX_STEPS` available and pays for anything past the first. It can only
afford a deep feature by leaving another one shallow. Two consequences:

- The comparison against -k is matched on activation compute.
- Any improvement must come from *reallocation across features*, which is the
  specialization hypothesis stated in a form that can fail.

An unbudgeted version could win by simply computing more, and would not
distinguish "per-feature specialization helps" from "more nonlinearity helps".

Enforcement is a Lagrange multiplier, not a coefficient: `log_lambda` receives a
reversed gradient, so the model's own optimizer runs the dual ascent - same
learning rate, same schedule, no separate step size, no penalty weight. It
climbs while the gates overspend and decays once they are back under budget.
`LAMBDA_MAX` clamps it; saturation is the signal that the constraint has stopped
being informative and should be retargeted rather than trusted.

## How many steps, and why this is not a DEQ

`MAX_STEPS = 8`. The ceiling is memory, and the reasoning is worth keeping
because the intuition "activations are cheap in a tiny model" is true about
FLOPs and parameters and false about the thing that actually binds.

Measured, one activation call at `[32, 64, 111]` forward+backward:

    steps    peak MB   MB/step     ms
        1       21.7      21.7    ...
        4       71.1      17.8   10.94
        8      137.0      17.1   21.06
       16      268.9      16.8   42.36
       32      532.5      16.6   78.25

That was the ORIGINAL unrolled loop: ~16.5 MB per step per call, because every
step stored its intermediates for backward. After the no-autograd rewrite below:

    steps    peak MB   MB/step
        1       22.6      22.6
        8       65.1       8.1
       32      210.8       6.6

Marginal cost fell from ~16.5 to ~6.1 MB/step. The -g lineage trunk makes ~10
activation calls per forward at depth 6 (4 instances, measured; the
abstractinator encoder adds more), so at the real batch of 64 that is ~120 MB
per unit of `MAX_STEPS`: ~1.3 GB at 8, and 16 is now affordable if the gates
ever saturate the ceiling.

## Not backpropagating through the recurrence

The trajectory runs under `no_grad`, so no step stores intermediates and the
nonlinearity's graph disappears entirely. The gradient is rebuilt from a single
differentiable step - the Jacobian-free / one-step approximation implicit models
are trained with (Fung et al., "JFB", 2202.08587; Geng et al.'s phantom
gradient, 2111.05177), which drops the `(I - J)^-1` factor of the exact implicit
gradient and remains a descent direction.

Three things had to hold, and two of them are easy to get wrong:

1. **The forward value stays exact.** The surrogate enters as
   `surrogate - surrogate.detach()`, identically zero in value. The output is
   the solved trajectory; only the backward comes from the surrogate.
2. **The surrogate is a function of the LIVE input.** An activation whose output
   did not depend on its input would pass no gradient upstream and every layer
   before it would go dark. Verified: input grad norm 16.4, not zero.
3. **Every parameter stays in the differentiable path.** The first version
   evaluated the surrogate at `h = 0`, which left `w` - the carried state's
   coupling to the frequency - with no gradient at all. Zero-init, so it would
   have sat at exactly zero forever, silently deleting the loop's
   state-dependence. Caught by `test_budget_starts_at_one_step_and_flows_gradients`.
   The surrogate now evaluates at the frequency the trajectory settled into,
   which is also what JFB does (differentiate at the solved state).

The gate multiplier is the realized step count `sum_k open_k`, so the surrogate
reads "taking n steps moves this feature n times as far as one step does". At
init exactly one step is open, so the surrogate is exactly one Serpent
evaluation and **both value and gradient are exact** (measured: gradient
difference from plain Serpent is bitwise 0). The approximation only switches on
as the loop is actually recruited.

What is *not* O(1): the gate graph, ~6 MB/step, because the halting parameters
learn from the task loss. Removing that too would mean training the gates from a
local objective instead - see the goodness-score note below.

## Why this is not a DEQ, and must not become one

Iterating Serpent with all gates open does not converge:

    default init (sampled a,b,g)   RMS  n=0:1.00  n=4:2.06  n=8:3.18  n=32:5.87
    a=b=1, g=0.1                   RMS  n=0:1.00  n=4:1.85  n=8:2.07  n=32:2.22

The `sin^2` term is non-negative, so it always adds; the default init drifts
upward with no sign of settling. The obvious repair is to damp the step into a
contraction so a fixed point exists. **Do not.** Serpent's fixed points are
where `sin(a*x) = 0`, i.e. the lattice `x = k*pi/a`. Measured, running a damped
`x + 0.2*(f(x) - x)` to convergence on 4096 distinct inputs:

    mean distance to nearest lattice site:  0.004
    distinct sites reached:                 8   (from 4096 inputs)

Convergence turns the activation into a quantizer that destroys the feature -
exactly the "contract to a fixed point and collapse" failure this design has to
avoid. So the DEQ inheritance is one-sided on purpose: take the memory
treatment (do not backprop through the solve), reject the objective (do not
iterate to convergence). Ouroboros is truncated iteration - ACT / PonderNet at
feature granularity - and the halting is what keeps it from collapsing.

## The Mono-Forward goodness score, and why it is not used here

`praxis/decoders/mono.py` cuts the graph and pays for it with a *local* loss at
each cut: a per-cut projection `M_i`, scored either against labels
(`CE(a_i @ M_i^T, labels)`) or against the decoder's own input stream
(`smooth_l1(pred, target)`). The graph-cut half of that idea is exactly what is
adopted above. The goodness half is not, for three reasons:

1. **Disproportionate.** A projection is `vocab_size x D` = ~114k parameters per
   cut, to train an activation whose entire parameter count is a few thousand
   per-feature scalars. The scaffolding would dwarf the thing it trains, at
   every activation site, at every step.
2. **No labels.** An activation is called as a bare `act(x)`; it has no access
   to labels or to the input stream a decoder cut can reach.
3. **The label-free local target is the fatal one.** The natural
   self-supervised goodness for a recurrent step is residual descent - "did this
   step reduce `|f(x) - x|`". That objective *is* the contraction, and the
   contraction is the lattice collapse measured above. A local goodness here
   would actively train the activation to destroy its own features.

So the gates keep learning from the task gradient (cheaply, through the
surrogate) plus the budget Lagrangian, and no goodness head is added.

**Depth/gain confound.** Because the map drifts, a feature that runs many steps
exits with systematically larger magnitude than one that runs few, so step count
doubles as a volume knob. ~2x spread at 8 steps, ~6x at 32 - another reason not
to raise the ceiling. If `ouroboros_steps_std` moves, check it is not simply
tracking output magnitude.

**The budget caps the payoff anyway.** Mean steps is pinned at 1.0, so extra
range is not extra computation: a feature at 8 steps has to be paid for by seven
at zero. And the gates are dense - a closed lane still computes and still
stores - so halting saves no memory. Every unit of `MAX_STEPS` is paid in full
for range the budget mostly forbids using.

## Init

`p`, `q`, `w` are zero. `u_0 = +6` saturates the step-0 gate open; `u_{k>=1} =
-5` saturates the rest closed. Therefore:

- **At eval, Ouroboros reproduces Serpent at init to float rounding** (measured
  max abs diff 3e-8). Both gates clamp to hard 0 and 1; the residue is that the
  gated update writes `x + 1*(y - x)` rather than `y`.
- In training the step-0 gate is *drawn*, so it lands on exactly 1.0 for ~98% of
  draws and slightly under for the rest. This is the one real departure from
  Servant's identity-at-init discipline, and it is the unavoidable price of a
  stochastic gate.
- Expected steps start at ~1.03 (measured), so the budget term contributes ~0 at
  step 0. No shock, no warmup schedule.

## Accounting path

Activations are called as bare `act(x)` and cannot return an auxiliary loss.
Instances push their expected per-step open mass onto a module-level stack in
`ouroboros.py`; `OuroborosBudget` drains it once per forward.

A module-level stack rather than a reference the regularizer holds, for three
reasons: a regularizer holding activation modules would double-register their
parameters in `state_dict` and the optimizer (the same hazard `modeling.py`
already documents for `classifier`); draining counts exactly the calls that
*executed*, so depth-conditional paths that did not run contribute nothing; and
multiple forwards per step accumulate correctly.

Accounting is off until a budget regularizer calls `enable_accounting()`, so
selecting `activation: ouroboros` without the regularizer cannot silently
retain live graphs. Pushes are gated on `self.training`, so eval never fills the
stack.

Under `torch.compile` the stack mutation is a graph break. The -g lineage sets
`no_compile: true`, so this costs nothing today; re-enabling compile means
revisiting it.

## Reading the run

`ouroboros_extra_frac` is the gate on the entire thread. It is the share of the
step budget spent past the first step. Flat at ~0 means the loop is never
recruited and the idea is answered for one config's cost. `ouroboros_steps`
should sit near 1.0; drift above it means the dual is losing and any loss win
against -k is no longer compute-matched.

`ouroboros_exit_{0..MAX_STEPS}` is the exit-depth distribution - the fraction of
features that stop after exactly k steps, rendered as one chart via
`series_group`. It comes free from the survival curve the budget already
computes: survival is monotone by construction, so consecutive drops are exactly
the exit bins and they sum to 1. Bin 0 is features gated off entirely; bin
MAX_STEPS is features that never converged.

`ouroboros_steps_std` is the spread of expected depth **across features**, and
it is the measurement the specialization claim actually rests on. The mean
cannot separate "every feature half-commits" from "features have split into deep
and shallow groups" - those give identical means and completely different
stories. It is **exactly 0 at init** by construction (uniform gate logits, zero
couplings), so like `router_depth_specialization` any nonzero value is learned
differentiation rather than noise. Verified: forcing half the features to run
three steps and half to run one gives mean 2.03, std 1.00, and bimodal exit
bins at 1 and 3.

The honest failure mode to watch for: `extra_frac` climbing while `steps_std`
stays near zero. That is the loop being recruited *uniformly* - every feature
taking the same extra steps - which is "more activation", not specialization,
and it would not survive the spline control arm.

## What the Activation Derivative chart is showing

`/api/activation_curves` probes each activation with
`autograd.grad(module(x).sum(), x)`. For Ouroboros the forward curve is the
true solved trajectory, but the derivative curve is the **surrogate's**
gradient, not `d(output)/d(input)` of the trajectory - the true one no longer
exists in the graph, which is the entire point of the no-autograd rewrite.

At init the two coincide exactly (measured: bitwise identical to Serpent). They
separate as the loop is recruited. This is the honest quantity to plot, since
the surrogate gradient is what actually trains the model, but it is not the
analytic derivative of the plotted forward curve and the two should not be
expected to agree once `ouroboros_extra_frac` moves.

Note also that the sampler sizes its probe from the activation's parameter
shapes. It used to use `numel`, which asked for `MAX_STEPS * D` features
because of the `[MAX_STEPS, D]` gate bias, failed to broadcast, and dropped
every Ouroboros from the chart silently - leaving only the memory Serpents
visible. Fixed to use the last axis, with a warn-once on any sampling failure
and a sweep test (`test_is_plottable_on_the_dashboard`) over the whole registry.

## Deliberately not in -l

- **Cutting `depth`.** The tempting move is to reduce recurrent depth and let
  the activation absorb the work. That is a second swap and makes a regression
  unattributable. Only worth a run if `extra_frac` moves first.
- **A retargeted budget.** 1.0 is the principled starting value because it is
  the baseline's cost. If `lambda` pins at its ceiling, the model is telling us
  the budget is the binding constraint, and *that* is when to move the target -
  as its own run, with the previous one as the control.
- **A 1-step learnable-spline control arm.** If -l wins, the honest control is
  a per-feature spline (`praxis/dense/spline.py` already has KAN machinery) at
  matched parameters, to separate "recurrence with halting" from "any learnable
  activation". Queue it as -m if and only if -l shows a win.

## References

- Louizos, Welling, Kingma, "Learning Sparse Neural Networks through L0
  Regularization" (1712.01312) - the hard-concrete gate and its closed-form L0.
- Banino et al., "PonderNet" (2107.05407) / Graves, "ACT" (1603.08983) - halting
  as learned compute allocation; this is those at feature granularity.
- Geiping et al. (2502.05171) - the recurrent-depth halting the model already
  runs at token granularity, cited in `praxis/halting/kl.py`.
- Bai, Kolter, Koltun, "Deep Equilibrium Models" (1909.01377) - what an
  iterated map converges to, and why the stable band is thin.
