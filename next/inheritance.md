# Inheritance: checking that the proofs still describe the code, and measuring how far a property survives

> Status: **grounded proposal, one finding already** (2026-09-12). Opened from
> "what if we built a continuous solver - a mechanism that implements the
> scientific property a module is supposed to confer, and shows the downstream
> computation actually inherits it?" The framing needs one correction (below),
> the statistics need a different instrument than the one proposed, and Lean
> belongs later rather than first. What survives is worth building, can start at
> about a day of work, and its first job is to attack the paper rather than
> defend it. Sibling to [architecture_separation.md](architecture_separation.md),
> [harmonic_koopman.md](harmonic_koopman.md),
> [paper_self_criticism.md](paper_self_criticism.md), [grounding.md](grounding.md).

## 0. The question that seeded it: is a proof a solver?

No, and the distinction is real rather than a complaint about unreleased code.

A solver maps data to numbers. A proof is a statement about the whole solution
space. A proof of global regularity for Navier-Stokes would tell you no smooth
solution ever breaks down; direct numerical simulation of turbulence would cost
exactly what it cost the day before. Run the other way, a perfect solver
establishes nothing: finite resolution over finite time is not a statement
quantified over all time, and near a singularity the numerics are least
trustworthy exactly where the mathematics is most interesting.

The two meet in one place, **computer-assisted proof**: a solver produces an
approximate object, then rigorous interval arithmetic proves a true object exists
in a small ball around it. The network is a search device; the proof is done by
the interval arithmetic. Wang, Lai, Gomez-Serrano and Buckmaster used PINNs
exactly this way to find the first smooth self-similar blow-up profile for 2D
Boussinesq and 3D axisymmetric Euler with boundary
([arXiv:2201.06780](https://arxiv.org/abs/2201.06780), PRL 130 244002, 2023),
explicitly as the basis for a future computer-assisted proof.

The nuance worth carrying: OpenAI's September 2026 claim is a **blow-up** result
(finite-time singularity with bounded energy), not global regularity. Blow-up is
existential - there exists initial data that does this - and existential PDE
statements are the case where solver and proof sit closest together, because a
witness is the content of the claim. So the Reddit comment is most true for a
forall statement and least true for this one. (Buckmaster, who built the PINN
pipeline above, is also the person in the credit dispute around the announcement.
That is not a coincidence; the solver-to-proof pipeline is his research program.)

And the Curry-Howard direction, since it is the obvious objection: a constructive
proof does yield a program, but program extraction from an existence proof gives
you *a* program, not a fast one.

**Where this lands for us.** Our setup is cleaner than the PDE case, because the
network is the *object being certified*, not the search device. The thing we want
to say is a statement about a specific artifact we already have in memory. That
is a much smaller ask than a theorem about a function space.

## 1. Where the paper makes the claim

`praxis/pillars/framing/` (rendered into `research/framing.tex` as `\paperFramingScaling`):

> The forced structure is the mechanism by which that payment buys anything. A
> harmonic field, a continuous latent, an autoregressive factorization - each is
> an inductive bias that makes the model compute the way a particular mathematics
> computes, importing that domain's structure rather than treating the
> architecture as neutral ground. Some of what is imported is exact: an
> autoregressive factorization is a proper normalized likelihood by the chain
> rule, not an approximation of one. Most of it is structure, not a theorem - a
> smoothness, a basis, a fixed point to iterate toward - and we keep the
> distinction honest. **The stronger reading, that computing in a domain's
> mathematics confers that domain's guarantees, is the question the framework
> exists to test, not a result it assumes.**

The paper already wrote the promissory note, already split exact from structural,
and already said the stronger reading is untested. This note is about cashing it.
`body.tex:44` does the same for Koopman and even supplies the falsifier: "the
working claim is that a *finite* harmonic basis approximates this model's
behavior well enough to measure, and how well is an empirical question, not a
definition." That sentence *is* the graded-proof epistemology, in the paper,
today. Nothing measures it.

## 2. The correction the framing needs

The idea as stated was: implement the scientific property, then show the model
satisfies it more than half the time.

Two problems, and fixing them is most of the design.

**Tautology.** Measuring "does the softmax behave like softmax" at the module
that computes it measures that Python works. The paper's claim is about
*downstream* computation inheriting the property. So the measurement has to be
taken **at distance** - some number of compositions away from the module that
imports the property. Inheritance is survival under composition, and most
properties do not survive composition. That is what makes it a real question
instead of an assertion.

**"More than 50%" of what.** A property that holds on 51% of an unstated
distribution is not a property. The number only means something against a
**null**: the same measurement on an arm that does not implement the mechanism.
This is already the house standard - `next/`'s density probe demands R-squared
above null, and the ablation-arm discipline exists for exactly this. A threshold
picked by hand would also violate the no-per-run-tuning rule.

## 3. Three tiers, and only two of them are research

- **Tier 0, structural (forall theta, exact).** True for every parameter setting,
  by construction. Causality, mass conservation, a gate in [0,1], the ghostmax
  norm bound, the D/L - 1 gate count. Residual should be exactly zero up to float
  epsilon. This is a **test**, not an experiment, and it is where the cheap value
  is.
- **Tier 1, inherited (graded, measured).** The property is imported at module m
  and survives to an observation point n compositions downstream. This is the
  paper's claim and the actual research.
- **Tier 2, emergent (graded, measured, never implemented).** No module imports
  the property and the trained model satisfies it anyway. Strongest possible
  evidence for architecture-induces-behavior and the easiest place to fool
  yourself, so it needs the strictest nulls. Not speculative: Gruver et al.
  ([arXiv:2210.02984](https://arxiv.org/abs/2210.02984), ICLR 2023) measured
  exactly this and found trained non-equivariant models acquire approximate
  equivariance, with transformers more equivariant than CNNs after training.

## 4. The one finding this already produced

Reading `proof-interference-capacity.yml` against `praxis/classifiers/harmonic.py`
took about ten minutes and turned up a hypothesis mismatch.

The proposition requires distinct integer frequencies with `0 < f_k < n/2`
**strictly**, and its part (i) claims orthogonality with squared norm `n/2`
"whatever the phases are." The default is
`self.F_t = F_t or min(hidden_dim, max_positions // 2)`
(`praxis/classifiers/harmonic.py:480`), so whenever `hidden_dim >= T//2` the top
temporal bin sits exactly at `f = T/2`, the Nyquist bin the hypothesis excludes.

At that bin the algebra changes. For `f = n/2`, `cos(4 pi f x/n + 2 phi)` is
constant in `x`, so the self inner product is `n cos^2(phi)` rather than `n/2` -
phase-dependent, which is precisely the independence part (i) asserts. If the
Weyl-seeded phase at that bin lands near `pi/2` the mode nearly vanishes, so it
is not only the constant `sqrt(n/2)` that is wrong there, it is injectivity.

The feature axis already knows about this. `harmonic.py:493` carries the guard
`if self.D % 2 == 0 and self.F_d == self.D // 2: w[-1] = 1.0`, the Hermitian
doubling correction at Nyquist. The time axis at `:507` builds `pos_cos`/`pos_sin`
from `arange(1, F_t + 1)` over `self.T` with no analogous guard, and the default
`F_t` lands it on the boundary.

Two honest qualifications. The counting argument in part (ii) survives - the Gram
matrix stays diagonal, so the map is still injective away from the degenerate
phase, and the capacity conclusion does not depend on the exact constant. And the
spectrum-rendering path at `:1120` picks `Tp = max(n_points, 2*F_t + 1)`, which
dodges the collision by construction, so this is about the field-evaluation path
and not the plots. The operative work item is to confirm the consequence in
`_eval_field` / `_field_fast` and then either add the T-axis guard or tighten the
default to `(T - 1) // 2`.

The point is not the size of the finding. The point is that **every proof in
`praxis/pillars/proofs/` has hypotheses, the paper renders the conclusions, and
nothing checks that the running code satisfies the hypotheses.** That gap is
present today, it is cheap to close, and closing it can find errors in a
published claim. This is the strongest argument for the whole exercise and it
does not require believing anything about inheritance.

## 5. What to measure, concretely

A property is a **residual**: a differentiable, non-negative quantity that is
zero exactly when the property holds. Candidates that already have a home:

| Property | Residual | Tier | Source |
|---|---|---|---|
| Ghostmax attends to nothing | `relu(norm(o) - m * max norm(v))`, `m = S/(1+S)` | 0 | `proof-ghostmax.yml` |
| Causality | `d out_t / d in_s` for `s > t`, via autograd | 0 | `project_smear_causality_leak` |
| Gate count | instrument the gate, count boundaries, expect `D/L - 1` | 0 | `lemma-ternary-gate.yml` |
| Harmonic isometry | Gram matrix of the realized basis vs `(n/2) I` | 0 | `proof-interference-capacity.yml` |
| AR normalization | probabilities sum to 1; joint factorizes | 0 | `\paperFramingScaling` |
| **Koopman linearity** | **DMD residual on mode coefficients across depth steps** | **1** | **`body.tex:44`** |
| Bias/variance orthogonality | loss-Hessian cross-block via HVP | 1 | `bias-variance-decoupling.yml`, roadmap:109 |

The causality row deserves its own line: one parametrized test that instantiates
every entry in the `attention`, `routers` and `memory` namespaces and checks the
Jacobian mask would have caught the leak automatically, and is permanent.

## 6. The statistics, in the form the idea actually wanted

The instinct was right and the instrument was wrong. "Inherits the dynamics more
than 50% of the time" has an exact, knob-free, distribution-free form already:

> Draw one input. Measure the residual under the treated arm and under the null
> arm. Report `P(R_treated < R_null)`.

That is the common-language effect size - a one-sided Mann-Whitney statistic. It
needs **no tolerance parameter at all**, which matters because a hand-picked `tau`
per run is exactly the tuning the house rules forbid. It has a confidence
interval. And `> 0.5` is literally the sentence, made precise.

Where a tolerance is genuinely wanted (Tier 0, where the answer should be "never
violated"), the certificate is a one-sided Clopper-Pearson bound on the violation
rate from `n` i.i.d. samples, which is standard practice in statistical
verification of networks and yields:

> With 95% confidence, module `m` satisfies P to tolerance `tau` on at least
> 91.3% of the eval distribution, against a null arm reaching 34.1%.

State the caveat before a reviewer does: this certifies **the sampled
distribution only**. Nothing adversarial, nothing off-distribution. That is a
real limit and it is the first thing anyone will push on.

## 7. The figure that does not exist yet

Import property P at module m. Measure its residual at m's output, then at m+1,
m+2, ..., at the logits, and in sampled text. Plot residual against composition
distance, with the null arm's band drawn behind it.

Three outcomes, all informative:

- **Flat and low.** The property survives composition. The paper's stronger
  reading holds for this property, measured.
- **Decays into the null band within one or two compositions.** The property is
  local decoration; downstream inherits nothing. **This falsifies the stronger
  reading**, which is the outcome to plan for, because it is the likely one.
- **Non-monotone.** Interesting, needs explanation, probably the most valuable.

A property-decay half-life per mechanism is a quantity nobody has published for
language models, and it maps directly onto the depth machinery already here
(recurrent depth, per-depth activations, the halting budget).

## 8. Prior art, audited

The method is not novel. The systematic application is unoccupied.

- **Measuring a property in trained networks.** Gruver et al.,
  [arXiv:2210.02984](https://arxiv.org/abs/2210.02984). Local equivariance error
  via the Lie derivative, across hundreds of pretrained models, with per-layer
  attribution. This is the template, executed for one property class. Note their
  method has "minimal hyperparameters" as an explicit selling point - same
  instinct as the knob-free statistic above.
- **Graded hypothesis testing on circuits.** Causal scrubbing (Redwood, 2022):
  resample-ablate according to a hypothesis, report *fraction of loss recovered*,
  published results ranging 51-93%. Proof-as-a-spectrum is already standard
  practice in mechanistic interpretability; the framing is not a novelty claim.
- **Formal bounds on small transformers.** Gross et al.,
  [arXiv:2406.11779](https://arxiv.org/abs/2406.11779) (NeurIPS 2024). Accuracy
  lower bounds for a small transformer on Max-of-K, 102 proof strategies scored
  on length and tightness, across 151 seeds. Their finding - more faithful
  mechanistic understanding gives tighter bounds - is the direct analogue of what
  we would be measuring, and their named obstacle (compounding structureless
  errors) is precisely the decay curve in section 7 seen from the proof side.
- **Lean plus machine learning.** Selsam, Liang, Dill,
  [arXiv:1706.08605](https://arxiv.org/abs/1706.08605) (ICML 2017): Certigrad,
  a stochastic-computation-graph system *written in Lean* with a machine-checked
  proof that its sampled gradients are unbiased. The relevant detail is *written
  in Lean* - see section 9.
- **Statistical verification.** The PAC / Clopper-Pearson line for bounding
  violation rates of network properties from samples. Mature, unglamorous,
  exactly the right instrument.
- **Physics properties by construction.** Hamiltonian and Lagrangian networks,
  PINNs, structure-preserving integrators. The whole field is "implement the
  conservation law and check the drift," which is Tier 0 and Tier 1 for one
  domain.

What is missing everywhere: the property declared **alongside the module**, in a
registry, measured **depth-resolved** against **ablation nulls**, across a zoo of
swappable mechanisms. Praxis is well positioned for that not because the idea is
new but because the substrate is: a registry of pluggable mechanisms, an existing
ablation-arm culture, a proof registry, and a metrics-to-dashboard path that
already exists. The contribution would be the table of (mechanism, property,
half-life, null gap), not the method.

## 9. Where Lean fits, and where it does not

Lean cannot carry the graded claims, and should not be asked to. A statement
about a distribution of residuals is not a theorem; it is an estimate. The only
theorem nearby is the *soundness of the estimator*, which is a small probability
result (Clopper-Pearson coverage) that Mathlib can carry today and that would
make the framework's headline sentence rigorous without formalizing a single
weight. That is a genuinely good use of a week.

Lean can carry the Tier 0 forall-theta lemmas. `lemma-ternary-gate`,
`proof-ghostmax`, `lemma-xor-circle` and the Parseval half of
`proof-interference-capacity` are small, self-contained, and within reach.

The substrate is newer than expected: TorchLean
([arXiv:2602.22631](https://www.alphaxiv.org/abs/2602.22631), ICML 2026)
formalizes networks in Lean 4 with IEEE-754 binary32 semantics and interval bound
propagation, and LeanCert does verified interval arithmetic. Both are young, and
the reported state is that most Float theorems are still `sorry`-stubbed pending
Mathlib integration - which is exactly the layer the numeric claims would need.

**The reason not to start here** is the extraction gap. A Lean proof of
`lemma-ternary-gate` proves it about a *Lean model* of the gate, not about
`praxis/halting/kl.py`. Every verification project leaks at that seam. Certigrad
closed it by writing the system in Lean, which is not on the table. So the first
reviewer question is "does this prove anything about the code you actually ran?"
and the answer has to be prepared, not discovered.

On the clout argument, which is a fair argument: a Lean file proving a lemma
about a Python function nobody can see is not clout. A Lean file **plus** a
measured inheritance curve **plus** a claim of our own that the measurement
killed is a paper. Order matters more than presence.

## 10. The build

Small, and it reuses what exists.

1. **`check:` on the proof schema.** Add an optional field to
   `praxis/pillars/proofs/*.yml`: a dotted path to a callable returning a
   residual. Entries without one render as unchecked in the paper, which is
   honest and is its own forcing function.
2. **A `properties` namespace.** `registry.declare("properties", ...)` with
   entries carrying `residual`, `tier`, and what they apply to. Affirmative
   names, no invented acronym. Not a new registry constant - the one registry,
   one more namespace.
3. **Residuals ride the metrics path.** A module that claims a property emits its
   residual from `training_metrics()`, so the decay curve is a live chart with no
   new plumbing, since charts are registry-driven already.
4. **`tests/pillars/test_proof_hypotheses.py`.** One parametrized test per proof
   entry that has a `check:`. This is the part that pays for itself immediately.
5. **`tools/measure_inheritance.py`.** Runs the depth sweep against a null arm,
   writes the curve. Verb-noun, per convention.

## 11. First two experiments

**Koopman linearity decay.** The paper states the claim with its own falsifier at
`body.tex:44`, the residual is a DMD fit (about twenty lines), and the null is the
obvious one: run the same linear fit on the **raw hidden state** instead of the
mode coefficients. If the raw basis is just as linear across depth steps, the
harmonic basis is decoration and we should know. Interesting either way, which is
the test for whether an experiment is worth running.

**The causality Jacobian sweep.** Cheap, permanent, and it would have caught a bug
that was previously found by hand.

Neither needs the registry work first. Both can be scripts. Build the scaffolding
only after one of them returns a number.

## 12. What kills this

- **Confirmation.** Going in wanting to defend the paper will find that the paper
  is right. The framing that survives contact is: this is the instrument that can
  falsify our own central claim, and that is why it is worth building. If the
  first three properties all show inheritance, be more suspicious, not less.
- **A weak null.** The null arm is the entire experiment. Everything looks
  inherited against a bad null.
- **Expect negatives.** Most properties probably do not survive composition. That
  is a result, and only a publishable one if it was framed as a measurement from
  the start rather than as a defense that failed.
- **The 79th note problem.** This is another entry in a directory where most
  entries are unbuilt. Its one advantage is that step 4 of the build is roughly a
  day, returns a real answer, and has already returned one before being written.
- **Lean as a side quest.** Three to six months, an extraction gap at the end, and
  it is the most tempting part. Not first.

## 13. Naming

"Continuous solver" names it after something it does not do - nothing is being
solved. What it does is check that a stated property survives into downstream
computation and report how far. **Inheritance** is the paper's own word for the
claim, so the note is filed under it; the registry namespace should be
`properties`, and the residual is a residual.
