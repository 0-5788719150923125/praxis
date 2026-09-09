# Unearned capacity: what training gives away for free, in time and in width

> Status: **raw capture + prior art** (2026-09-09). Two conversation threads,
> merged because they turned out to be the same question on two axes: one asked
> whether we can jump forward in *training time*, the other whether we can
> reclaim and re-issue *parameters*.
> Companions: [[project_harmonic_latent_koopman]],
> [harmonic_koopman.md](harmonic_koopman.md), [rlct_landscape.md](rlct_landscape.md),
> [exact_solve.md](exact_solve.md), [[project_ghost_features]],
> [[feedback_no_hyperparameter_tuning]].

## The one line

**Early training earns very little.** It accrues almost no content and needs
almost none of its width, so the early trajectory is the part that can be
skipped and the early parameters are the part that can be withheld - and the
same instrument measures both, because both are the gap between a model's
nominal size and its effective one.

Standard practice runs full width from step 0, paying for capacity nothing has
asked for yet, and pays serially for a trajectory segment that is nearly the
same across runs.

---

# Axis 1: jumping ahead in training time

Spoken as "a vortex that takes us right where we need to be" - launch from
random init, have a module solve for where the model will be at step Y, and
resume from there. (`vortex` is already the Godot visualizer thread, so: *jump*.)

## Predictable statistics, unpredictable coordinates

The early phase is highly **stereotyped** across runs: norm growth, attention
entropy falling off uniform, embedding anisotropy, induction-head emergence all
look nearly identical run to run. It is also where **basin selection** happens,
so the pointwise weight trajectory is chaotic.

Both are true, and the resolution is the useful sentence: **the statistics of
training are predictable, the coordinates are not.** A jump needs coordinates.

This predicts the measured behavior of every existing weight-predictor. Graph
HyperNetworks (Knyazev et al., NeurIPS 2021 / ICML 2023) predict every parameter
of an unseen network in one forward pass; the predicted weights land far above
random init and far below trained, and their real value is as an *initialization*
that then fine-tunes faster. Order hundreds to low thousands of steps of jump.
Take that as the calibration number for any scheme here.

It also inverts the instinct: **the jump is more available late than early**,
because late training is largely diffusion inside a basin. The urge is to skip
the boring beginning; the dynamics say the end is the skippable part.

## The cheap version is already running here

Weight-space extrapolation needs no learning and is the baseline any predictor
must beat. LookAhead (Zhang et al. 2019) takes k fast steps then jumps the slow
weights toward them. Latest Weight Averaging (Kaddour 2022) reaches a target loss
in fewer steps by averaging recent checkpoints. And **Schedule-Free**, which this
repo already runs, has an averaged iterate that *is* an extrapolation, evaluated
at a point the raw iterate has not reached.

So the open question is narrow and cheap: **does a learned extrapolator beat the
closed-form one, and by how many steps?** Posed that way the predictor's output
is a single coefficient conditioned on trajectory features, not a map into
`R^|W|`, which is what sinks naive weight generation (a hypernetwork emitting N
parameters is larger than its target unless heavily factored, and the
factorization confines it to a manifold that likely excludes the solution).
It is also learned and endogenous, so it satisfies the standing no-tuning rule.

## Koopman: the same idea, in the language this repo already speaks

Training is a nonlinear dynamical system on weights. A Koopman operator
linearizes such a system in a lifted observable space, and there **advancing n
steps is one operator raised to the n-th power** - O(1) in the number of steps
skipped. That is the vortex, written in the mathematics of the harmonic thread.

Not an analogy: Dogra & Redman (NeurIPS 2020, *Optimizing Neural Networks via
Koopman Operator Theory*) did exactly this to accelerate training, demonstrated
on shallow networks. Scaling is unestablished.

The framing also predicts its own limit, which is why it is worth adopting:
Koopman linearizes exactly only in an infinite-dimensional observable space, so
finite truncation degrades with horizon and jump length is bounded by truncation
quality. Same bounded jump the GHN line measured, reached from theory instead of
experiment.

## Parareal: where the error goes

The sketch expects the jump to be imperfect and has nowhere to put the error.
**Parareal** (Lions, Maday & Turinici 2001) is the missing skeleton: a coarse
propagator guesses future states, a fine propagator runs real steps inside each
window in parallel, and a correction iteration reconciles them, converging to the
exact serial answer.

Mapped here, the vortex is the coarse propagator and SGD is the fine one, so the
jump never has to be *right* - only close enough that correction converges in
fewer total fine steps than the serial run. Imperfection stops being fatal and
becomes a cost term. The catch to state up front: speedup is bounded by (windows
/ correction iterations), so a poor coarse propagator buys nothing.

## First experiment

Uses checkpoints this repo already writes, and produces one number.

1. Save dense checkpoints on a short run.
2. From step `t`, extrapolate with the simplest closed-form rules available:
   linear along the recent trajectory, EMA plus momentum, the Schedule-Free
   averaged iterate.
3. Measure **`K`**: how many steps ahead the extrapolated point stays at or below
   the true loss curve.

`K` is the program compressed to a scalar. Hundreds means a learned coefficient
has room and Parareal has a viable coarse propagator. Five means the ceiling is
low and this axis closes.

Second measurement if `K` is encouraging: does `K` grow with `t`? The
statistics/coordinates argument predicts it **grows**. If it shrinks, that
argument is wrong and this section needs rewriting before anything is built.

---

# Axis 2: reclaiming and re-issuing width

## Two ghosts, and they connect

`praxis/transforms/ghost.py` is Vieira Neto & Valle's structured weight tying
(arXiv:2608.07735): `d` weight blocks expanded from one stored tensor by signed
permutations. A different paper, **Sonoda, Ishikawa & Ikeda, *Ghosts in Neural
Networks: Existence, Structure and Role of Infinite-Dimensional Null Space***
(arXiv:2106.04770), establishes the property the idea below needs.

## Any blend of ghosts is a ghost

The ridgelet / integral representation is **linear in the measure**:
`f(x) = ∫ γ(a,b) σ(a·x − b) dμ`. So if `μ` and `μ'` both represent `f`, every
affine combination of them does too, and their difference is a null element - a
ghost. The null space is a linear subspace, therefore **closed under blending**.

That is exactly the property the SMEAR-of-priors idea requires ("any blend of
those discovered functions produces consistent results"), and it is *not*
generic. In ordinary finite-width coordinates the set of parameters giving one
function is non-convex - two networks equal up to a permutation average to
something worse than either, which is the whole reason Git Re-Basin has to solve
a matching problem first. The ridgelet representation is the setting where
blending is safe, and 2106.04770 is the paper that says so.

## The constraint that shapes the design: exact null is exact flat

If `θ + tν` gives an identical function for all `t`, the loss is constant along
`ν`, so the directional derivative is zero and gradient descent will never move
along it. An exact ghost is not reserve capacity waiting to be recruited; it is a
direction the optimizer cannot see. Same fact the LLC reports when it counts
degenerate directions as not-parameters.

## Which is why the paper's rate is the useful part

Finite-measure null elements admit width-`N` discretizations with **`O(N^-1/2)`
output error**. At finite width, ghosts **leak**. Shallow rather than flat:
nonzero contribution, therefore nonzero gradient, therefore recruitable. The
paper's own remark that parameter perturbations expose information hidden in
these null spaces is the same statement.

So the target is not the null space, it is the *discretization error* of the null
space, and **width is the dial on how visible the reserve is**.

## `right_inverse` already computes the ghost, and discards it

`ghost.py`'s `right_inverse` returns `mean_k P_k(W_k)`, the least-squares real
tensor for an incoming `W`. So `expand(right_inverse(W))` is the projection of
`W` onto what the tying can represent, and

    W − expand(right_inverse(W))

is the residual it cannot: **the null space of the tying map**, the finite-width
instance of the paper's object, thrown away at registration today. Re-admitting
it as a SMEAR-blended term over several sampled ghosts is the buildable form of
the idea, inside a module that already composes with SMEAR.

## Capacity is not earned early

The obvious objection to a prune-and-rebirth cycle is that the budget reclaimed
is by definition the budget that was doing nothing, so fresh weights should just
re-degenerate: a treadmill.

**The answer is that the cycle is not stationary.** Capacity is not earned in
early training. So prune hard early, when nothing has been earned and the width
is idle, and re-issue those parameters later, as the model converges and actually
needs somewhere to put what it has learned. Weights re-introduced at a later
stage of convergence do not land in the same situation the pruned ones did.

This is the sharp form of the whole note, and it makes the schedule the object of
study rather than the mechanism.

## What is already shipped, and what is not

Function-preserving change of dimensionality is solved: **Net2Net** (Chen,
Goodfellow & Shlens 2015) widens by copying units and splitting outgoing weights
with the function unchanged, and at scale that line is bert2BERT / LiGO /
"Stacking Your Transformers," with real double-digit pretraining savings. The
prune-and-regrow cycle also exists: **RigL** (Evci et al. 2020) drops small
weights and regrows by gradient magnitude on a schedule.

Two things are not standard, and they are the contribution:

1. **Collapse by degeneracy rather than by magnitude.** RigL's criterion is a
   proxy; the functionally-null directions are the principled target.
2. **Growth triggered by a measured signal rather than a schedule.** Every
   existing growth method grows on a fixed calendar.

## The trigger, and the instrument this repo already has

The LLC from [rlct_landscape.md](rlct_landscape.md) is an effective-parameter
count, so `LLC / nominal` is a utilization ratio and `nominal − effective` is
exactly the reclaimable budget. **The LLC says how many weights can be birthed,
and when.** Saturating toward nominal means the model has run out of effective
capacity: birth. Sitting far below means width is going unused: hold, or prune.
Endogenous, no schedule, no tuned constant.

The word "singularity" in the original framing is not only imagery. Singular
learning theory is precisely the study of the parameter-to-function map being
non-injective with fiber dimension jumping at singularities, and the RLCT is its
measure. The theory of this idea is already running in the repo.

## First experiment

Run the cycle and watch the LLC across rebirths. **Rising effective parameter
count means it works.** Returning to the same plateau means the task saturated
and the birthing is cosmetic - which is the treadmill, measured rather than
argued.

Second, and it is the schedule claim doing the work: compare *early* rebirth
against *late* rebirth at matched total parameter-steps. The claim predicts late
wins. If early and late are indistinguishable, "capacity is not earned early" is
wrong and the mechanism is just RigL with a better pruning criterion.

---

## Open questions across both axes

- Is `K` a function of the optimizer? Muon's orthogonalized updates and Lion's
  sign updates have very different trajectory geometry than Adam's, and one may
  be far more extrapolable. This repo runs all three
  ([[project_muon_composite]], [[project_lion_geo_optimizer]]) and can look.
- Does the LLC predict `K`? A more degenerate region should be more extrapolable.
  If so, one probe drives both axes: the landscape measurement becomes the jump
  scheduler *and* the rebirth trigger, which is the strongest reason to treat
  these as one thread.
- Are the optimizer state buffers more extrapolable than the weights? Momentum is
  already a smoothed velocity and may be the better prediction target, with
  weights following from it.
- Should the jump be predicted in Koopman coordinates rather than weight
  coordinates? Gauge-invariant coordinates sidestep the permutation problem
  instead of paying a matching solve for it - the same move that makes ridgelet
  blending safe on axis 2.
