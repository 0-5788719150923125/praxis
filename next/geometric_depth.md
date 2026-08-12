# Geometric supervision at every depth step

> Status: **parked on arrival** (2026-08-11). Ryan's own read on proposing it
> was "it kind of seems like the wrong approach," and I think that read is
> right, for reasons recorded below. Written anyway because three things came
> out of thinking it through that outlive the idea: a live measurement of the
> depth trajectory, a standing epistemic test for the paper, and a design
> distinction (progress vs position) worth reusing. Sibling to
> [recurrent_depth_concentration.md](recurrent_depth_concentration.md), which
> asks a neighbouring question about the same depth loop.

## The idea

If the paper's claim is that harmonic models learn to predict geometry over
time, then HALO's geometric objective could stop being a head-only term and
become an **auxiliary loss computed after every layer of the decoder** -
forcing the model to produce geometry at every stage of computation rather than
only at the classification head.

In this codebase that means something more specific than it sounds. With
`num_layers: 1` and `depth: 6`, "every layer" is **one recurrent map supervised
at six points along its own trajectory**, not six distinct layers. That is a
better-posed object than deep supervision of a stack, and it is the Koopman
shape: an observable (the radial coordinate) evolving under a nonlinear map.
The idea is not silly. It is well-aimed at the right claim.

## Why it is parked

### 1. The trajectory it would constrain is already collapsed

`depth/step_dN` is the relative step size per depth transition,
`||print[i+1] - print[i]|| / ||print[i]||` (`praxis/decoders/sequential.py:211`).
Read live off `abstractinator-j` at step 22,011 (run bc5a06698):

```
depth/step_d0             7.858        one enormous hop
depth/step_d1             0.596
depth/step_d2             0.435
depth/step_d3             0.407
depth/step_d4             0.385
depth/step_d5             0.394
depth/convergence_ratio   0.0518       last step is 5% of the first
depth/jump_concentration  3.381        slope +0.116/1k, RISING
halting/mean_loops        3.065        of a max of 6
```

Step 0 moves the fingerprint about twenty times as far as any later step, and
the concentration is *increasing* over training.

### 2. The naive form's optimum IS that collapse

HALO's regularizer is not a generic "be geometric" term. `losses/halo.py:315`:

```python
diff_true = pos - cen[target]
r_sq_true = diff_true.pow(2).mean(dim=-1)
```

It is a radial NLL around **the correct token's centroid**. Applying it at
depth 1 asks the model to already be at the answer after one step, and the
cheapest way to satisfy it at all six depths is to jump to the answer
immediately and then hold still. That is the trajectory the model already has.
The loss would not correct the pathology; its optimum *is* the pathology.

Recording the factual correction that fell out of this, since it is easy to
assume otherwise: **the HALO regularizer is not class-agnostic.** You cannot
get a "stay on the shell without knowing the answer" term by taking a subset of
it. It would have to be recomputed against the *nearest* centroid instead of
the target's, which is a modification, not a subset.

### 3. There is a feedback loop with halting

`halting_type: kl` fires on convergence between consecutive steps. Force the
geometry to be correct at every step, the inter-step KL shrinks, halting fires
earlier, and depth collapses further. This is predicted rather than speculative
and would show up in `halting/mean_loops` and `depth/convergence_ratio`.

### 4. It adds a constraint where the problem is capacity

Four objectives already pull on the same trunk features: the blended CE, HALO's
geometric term, the encoder's VQ/reconstruction path, and (since `-k`) CE
reaching the HALO arm directly. A fifth per-depth constraint on those same
features is more likely to produce a compromise than a geometry. If depth is
underused, the likelier causes are the halting criterion, the residual, or one
recurrent block at `hidden_size: 111` not having room to do six distinct
things. **An auxiliary loss does not add capacity.**

### 5. The general form of the objection

If the geometry does not emerge, that is *information about the architecture*.
Penalizing the model until it appears suppresses the information. This is the
same failure `regime_gated_priors.md` §1 warns about from the other direction -
a gate learning to evade its own prior yields zero effective enforcement while
reporting full compliance. Here it is the mirror: full compliance, purchased,
reporting nothing.

## What is worth keeping regardless

### The depth measurement, and the fact that it reads two ways

`jump_concentration = 3.381 and rising` is either:

- **a training pathology** - depth is decorative, step 0 does the work, five
  steps of settling are wasted compute; or
- **partial confirmation of the paper's own prediction** - `sec:watchmaker`
  reads recurrent depth as a wind-up that releases back to the stable basis,
  "punctuated by sparse, discrete jumps where a geometry is briefly expressed."
  One big hop then settle is exactly that shape.

These cannot both be acted on. Stated precisely so it does not get overclaimed:
`jump_concentration` is `max(s)/mean(s)` over depth *step sizes*, while the
paper's prediction is about the *deviation from the standing shape*
concentrating across positions and features. Adjacent evidence, not the test.
[recurrent_depth_concentration.md](recurrent_depth_concentration.md) owns the
actual test and still reports "not yet run" - but it can now be run against a
live trace rather than from scratch, and the step-size profile above is the
context it should be read in.

**Deciding which reading is right is worth more than this note's idea.** If it
is the wind-up reading, the settling steps are the mechanism and forcing
geometry into them would break it. If it is the pathology reading, the fix is
in halting or capacity, not in another loss. Either way the aux loss is wrong.

### The design distinction: progress vs position

The salvageable form of the idea, if it is ever built. Do not supervise where
the geometry *is* at each depth; supervise that it **gets closer**:

```
L_geo = sum_d  relu( r_sq(d+1) - r_sq(d) + margin )
```

- Cannot be satisfied by arriving at step 0. If you jump to the answer there is
  nothing left to improve, so the term stays unsatisfied for the remaining
  steps. It penalizes exactly the measured behaviour.
- It is literally the temporal claim. Position-supervision says *be here*;
  progress-supervision says *get closer*, which is what "predicts geometry over
  time" actually asserts.
- Cheap: one scalar per token per depth, a hinge, no new centroids, no new
  calibration. `sequential.py` already collects `depth_prints` for the step
  metrics, so most of the plumbing exists.

Two constraints it would inherit:

- **Weighting must be endogenous** (the no-tuning rule). `halting/train_r_1..r_6`
  already gives the live-token count per depth (12580, 19530, 18800, 14120,
  9481, 5615 on `-j`); weight the aux by the live fraction. Free, no schedule.
- **Gamma probably has to be per-depth.** The calibration holds because centers
  are `randn` and inputs are RMS-normalized, so `r_sq_init = 2.0` is true by
  construction. At intermediate depths the activation statistics differ, and one
  global gamma asserts the geometry should be equally tight at every step -
  the opposite of what a progress constraint wants. Same shape as the per-depth
  RoPE theta work.

### The epistemic test, which generalizes past this idea

**Supervising a property converts it from evidence into an assumption.**

If the paper claims the architecture *produces* geometric structure, that claim
cannot survive training the structure in at every layer. A reader who notices
discounts the section, and they are right to. The claim silently becomes "we
penalized the model until it produced geometry," which is a different and much
weaker statement.

This is recoverable only deliberately:

- Run it as an **ablation** and make the claim comparative - with per-depth
  geometry vs without, the delta is the result. "Imposing geometry per-step
  improves X" is narrower and defensible, and interesting either way.
- Or keep the aux loss **out** of the configurations the emergence claim rests
  on, and present it separately as an engineering result.

What cannot happen is adding it to the main line while keeping the emergence
language. This is the same move `lottery_engineering.md` already made once:
"Praxis proves LTH" did not survive, "Praxis engineers the lottery" did, and
the narrower claim was stronger. Worth keeping as a standing test for anything
the paper cites as emergent: **would this still be evidence if we had trained
for it?**

## What would unpark this

- `recurrent_depth_concentration.md` runs and reports the **pathology** reading
  rather than the wind-up reading, AND the causes in §4 above (halting,
  residual, capacity) have been ruled out. Then a trajectory constraint is
  attacking a real, isolated problem.
- Or the progress form gets built somewhere cheap first (a single-block probe,
  not the main line) and the hinge measurably decompresses `step_d0` without
  costing BPB. That is a contained afternoon and it is the only version I would
  volunteer for.

Neither is urgent. Nothing downstream is waiting on this.

Related: [recurrent_depth_concentration.md](recurrent_depth_concentration.md)
(the test that should run first), [harmonic_koopman.md](harmonic_koopman.md)
(the observable-over-time framing), [regime_gated_priors.md](regime_gated_priors.md)
(the evasion/compliance discipline this inverts),
[lottery_engineering.md](lottery_engineering.md) (the narrowing move the
epistemic test copies), [oscillatory_axes.md](oscillatory_axes.md) (the other
parked depth thread).
