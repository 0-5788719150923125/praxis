# Omission routing: the computation is what you left out

> Status: **design note** (2026-09-13) - nothing built, nothing measured.
> Comes out of the declarative-routing thread: if a model picks 1 function from
> a bank of 1000, could it instead call 999 and let the *omission* be the
> choice? Siblings: [declarative_insertion.md](declarative_insertion.md) (same
> credit-assignment problem one level up, and the same Toolformer-shaped
> repair), [mixture_of_widths.md](mixture_of_widths.md), [prismatic.md](prismatic.md).

## The combinatorial claim is false

`C(N, 1) = C(N, N-1) = N`. Selecting one and omitting one are the same space,
exactly. More generally `C(N, k) = C(N, N-k)`, so **inverting a subset-selection
problem can never shrink it**. The only way to move is to change `k`, and the
binomial peaks at `k = N/2`, which is the largest the space ever gets - so
"call most of them" is moving *toward* the maximum, not away from it.

That settles the question as asked. The rest of this note is about the thing
the question was actually pointing at, which is real and is not about
cardinality.

## What does change, by a factor of N

**Information per sample.**

Under 1-of-N selection, a training step tells you about the one expert you
routed to. The choice is discrete, so the gradient does not reach it, and the
standard repair is REINFORCE against downstream loss - a single high-variance
draw from a 1000-way categorical. The other 999 experts receive nothing. This
is the same dead-expert / cold-start problem that PEER already fights and that
[[project_vq_codebook_collapse]] is the codebook-shaped version of.

Under N-1-of-N, every `f_i(x)` has already been computed. So the full
leave-one-out profile is available for the same sample, and - this is the part
worth keeping - it does not cost N forward passes to read:

```
y_i = S - f_i(x)        where  S = sum_j f_j(x)
L(y_i) ~= L(S) - <grad_y L(S), f_i(x)>
```

One gradient at the layer output (which the backward pass computes anyway),
then N inner products of width `d`. Negligible against the matmuls that
produced the bank outputs. The first-order truncation is tightest exactly when
`f_i` is small relative to `S` - which is the omission regime - so the cheap
estimator is accurate precisely where this scheme lives.

That object is not new: it is the first-order Taylor importance criterion from
structured pruning (`|dL/da * a|`), evaluated per token instead of per
parameter over a corpus.

**Accounting, honestly.** Per FLOP this is a wash: N times the compute bought N
times the information. It wins on two other axes.

- *Per sample.* On a streamed, non-stationary corpus you cannot re-draw a
  context to try a different expert on it. Information per sample is the axis
  that binds, not information per FLOP.
- *Per unit variance.* A deterministic Taylor estimate replaces a REINFORCE
  draw. That is usually worth more than the FLOPs.

## The cost, and it is fatal as stated

**Conditional computation dies.** A bank exists to decouple parameters from
FLOPs. Calling 999 of 1000 is a dense layer wearing a router, and at that point
the bank is a wide layer with extra bookkeeping.

**Behavioral diversity collapses, and it gets worse as the bank grows.** For N
iid zero-mean terms of norm `m`, `||S|| ~ sqrt(N) m` while
`||y_i - y_j|| = ||f_j - f_i|| ~ sqrt(2) m`. Relative separation between any
two architectures is therefore `O(N^-1/2)` - about 4.5% at N=1000, and
`O(N^-1)` if the terms are correlated. Under 1-of-N selection the same
separation is `O(1)`, independent of N.

**Grow the bank and selection gets richer while omission gets flatter.** You
keep the cardinality and discard the reason to want it: 1000 nominally distinct
architectures, all within a few percent of each other and of the unconditional
mean.

## The crux, and the one escape from it

Stated plainly because it is the whole note:

> The regime where the training signal is richest (`k -> N`, dense, every
> expert evaluated) is exactly the regime where the actions are least
> distinguishable (`O(N^-1/2)` separation). Sparse selection has the opposite
> pair. The `k/N` knob trades one for the other and cannot give both.

**The combiner is the escape, and it is the only one found.** Everything above
assumes the bank is combined by a *sum*. Under a conjunctive or multiplicative
combiner the arithmetic inverts:

```
P   = prod_j g_j(x)
y_i = P / g_i(x)
```

If `g_i` is near zero, dropping it changes the output by orders of magnitude.
Separation is `O(1)` or unbounded, independent of N. Removing one clause from a
1000-clause conjunction can massively change the admissible set; removing one
term from a 1000-term sum cannot.

So the design rule, checkable before any code is written:

- **Bank of gates / predicates / constraints, combined multiplicatively**:
  omission is a high-leverage action and the dense signal is available. Both
  halves at once.
- **Bank of MLP outputs, combined additively**: omission is a perturbation.
  Degenerate, no matter how large N is.

This also says what the bank entries have to *be*. Not "1000 abstract
transformations" - 1000 **restrictions**. The model is not assembling a
computation from parts, it is sculpting one by declining prohibitions. That is
a different object from MoE and it is the only version of this idea that is not
a worse MoE.

## What this actually is, once named

The scheme that falls out is not "omit one of 1000". It is:

1. **Dense teacher.** Forward the whole bank. Read the full LOO profile from
   one gradient plus N dot products. This is a *hindsight* signal - it uses the
   label - so it cannot be the inference-time policy.
2. **Sparse student.** Train a cheap router on the hidden state to predict that
   profile. Standard distillation of hindsight into a forward policy; the same
   move Toolformer's filter makes in [declarative_insertion.md](declarative_insertion.md),
   one level down.
3. **Sparse deployment.** Ship the student. Conditional computation is
   recovered, because the dense pass was a training cost, not an inference one.

The dense-teacher half can be annealed off, which makes this a dense-to-sparse
gating schedule with an unusually cheap and unusually low-variance teacher.

## Prior art

The inversion has been done, extensively, under a name you would not search
for: **pruning**. Its empirical record is the real evidence for the instinct -
subtractive search is easier than additive search, not because the space is
smaller but because you start with gradients everywhere.

- **Lottery tickets** (Frankle & Carbin 2019, arXiv:1803.03635). The good
  subnetwork is found by removal from a dense whole.
- **Supermasks** (Zhou et al. 2019, arXiv:1905.01067) and **"What's Hidden in a
  Randomly Weighted Neural Network?"** (Ramanujan et al. 2020,
  arXiv:1911.13299). The extreme case and the strongest single data point: a
  binary mask over *untrained random weights*, with no weight training at all,
  reaches good accuracy. The computation is literally the omission.
- **Piggyback** (Mallya et al. 2018, arXiv:1801.06519). Per-task binary masks
  over one frozen backbone - many architectures from one network, by omission.
  The closest existing thing to "the mask is the architecture choice."
- **L0 regularization with hard-concrete gates** (Louizos et al. 2018,
  arXiv:1712.01312). The learned-omission machinery, already differentiable.
- **Taylor pruning** (Molchanov et al. 2017, arXiv:1611.06440). The first-order
  LOO criterion above, at parameter granularity.
- **Stochastic depth** (Huang et al. 2016, arXiv:1603.09382) and dropout.
  Computation defined by what is omitted, with a *random* policy, and it gains
  accuracy anyway - which bounds how much a learned policy has to buy to be
  worth its cost. This is the standing null arm for everything below.
- **SMEAR** (Muqeeth et al. 2023, arXiv:2306.03745), already in this repo at
  `praxis/routers/smear.py`. Merges expert *parameters* densely precisely
  because discrete routing has no gradient. Omission routing is a **constraint
  on the SMEAR simplex** (near-uniform weights with one zero), not new
  expressive power, so it has to be argued as inductive bias and never as
  capability.

**The gap.** Every mask above is per-task or per-training-run, over weights,
with an additive or pass-through combiner. Nothing found does a **per-token,
input-conditional mask over a bank of semantically distinct functions under a
conjunctive combiner**. That intersection is the contribution if there is one,
and the conjunctive requirement is what keeps it from being a restatement of
pruning.

*(arXiv IDs above are from memory and must be spot-checked before any of them
reach `research/`.)*

## Cheapest first probe

The premise is testable without a bank, a mask, or a policy, and the first two
steps are nearly free.

1. **Measure the separation directly.** On a live PEER checkpoint, compute the
   per-token LOO profile over the retrieved experts (`S`, then
   `<grad_y L, f_i>` for each). Two readings settle whether this is worth
   anything: the *spread* of the profile (if omitting any expert costs about
   the same, there is no structure to route on) and its *predictability* from
   the hidden state (regress the profile on `h`, report R^2 against a shuffled
   null). A flat or unpredictable profile kills the line for zero training runs.
2. **Check that the combiner claim is not theoretical.** Same measurement on a
   multiplicative site - a gate stack, not the FFN sum. The prediction is a
   heavy-tailed profile where sums give a flat one. If it is flat there too,
   the escape hatch does not exist and the whole note is dead.
3. Only then: a hard-concrete or `gumbel_sigmoid(hard=True)` mask
   (`praxis/functional/gumbel_sigmoid.py` already has the straight-through
   half) over a small conjunctive bank, trained by distilling the LOO profile,
   `k/N` swept rather than pinned at `(N-1)/N`.

Null arms, in the order they have to be beaten: **random mask at the same
budget** (this is dropout, and it is not a weak baseline), **no mask**, and
**learned 1-of-N selection at matched parameters**. A learned omission policy
that ties random masking is decoration.

## What it likely costs

Stated in advance so a bad result is not reinterpreted afterwards.

- **The dense teacher is a real training cost**, N function evaluations per
  step at the bank site, and it has to be annealed off or the scheme never
  reaches the sparse deployment it was justified by.
- **The hindsight-to-forward gap may be the whole story.** The LOO profile uses
  the label. If the student cannot predict it from `h` above the shuffled null
  (step 1 above), there was never a policy to learn and the dense signal was
  measuring something the model cannot know at inference.
- **`k` is one more thing that needs to not be hand-tuned per run**, which the
  standing rule forbids ([[feedback_no_hyperparameter_tuning]]). It has to come
  out of a budget or a learned L0 penalty, not a flag.
- **At this stack's width the bank entries are tiny.** PEER sizes to 1024
  experts at `hidden_size: 256`, and those are rank-1. A bank of 1000
  *restrictions* has to be cheaper than rank-1 to be interesting, which
  probably means scalar gates, which probably means the bank is a gate vector
  and the whole thing collapses back into an activation. Worth resolving on
  paper before step 3.

## Open questions

- Is there a combiner between sum and product that keeps `O(1)` separation
  without the numerical hazards of dividing by a near-zero gate? `logsumexp`
  and `min` are the obvious candidates; `min` makes omission exactly
  constraint-relaxation and is the cleanest statement of the idea.
- Does the first-order Taylor estimate stay accurate at the `k/N` values that
  actually give separation? The approximation is good near `k = N` and the
  actions are good near `k = 1`. If the two windows do not overlap, the note's
  thesis is that no single `k` works, and the answer is the dense-teacher /
  sparse-student split rather than a compromise `k`.
- Does the mask want to persist across tokens? A restriction with a *duration*
  ("no periodic branches for this span") is the one thing a per-token router
  structurally cannot express, and it is the same affordance the emitted-tag
  thread was chasing. That is the bridge back to
  [declarative_insertion.md](declarative_insertion.md).
- Does this compose with ghosting? `praxis/transforms/` already removes
  capacity by tying; omission removes it by masking. Stacking two subtractive
  mechanisms on one site probably measures nothing, and it should be checked
  rather than assumed.
