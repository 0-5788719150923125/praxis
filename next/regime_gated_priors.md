# Regime-gated priors: structure that binds where it applies

> Status: **reading capture** (2026-08-04). Salvage from an outside draft on
> shape-constrained spectral learning. The draft's own mechanism does not work -
> the failure is documented below because the failure is the useful part - but
> three ideas underneath it are real and have direct Praxis purchase. The
> underlying literature is public and citable; the draft is not, so nothing here
> is attributed to it.

The through-line: **a structural prior should bind conditionally, not
uniformly.** Praxis currently applies almost every prior globally - the
smoothness prior on the amplitude grid, the L2 on the harmonic delta, the
weight-decay exclusion list, the isotropy penalty. Each is a single scalar
pressed evenly against a corpus that mixes registers. The classical
shape-constrained literature (Robertson et al. 1988; Gupta et al. 2016; You et
al. 2017 for the partially-monotonic case) has the same defect and has started
to notice it: a qualitative law that holds in one regime and not another is
worse than no law, because enforcing it everywhere forfeits accuracy exactly
where it does not apply.

---

## 1. Gated constraints, and the way to get them wrong

The idea: attach to each constraint a learned gate `w(x) ∈ [0,1]` and enforce
the constraint in proportion to the gate. Constraints stay convex in the model
parameters for fixed gates, so you can alternate cheap projection with gradient
descent on the gate.

**The trap, stated plainly, because it is easy to walk into.** The obvious
implementation scales the rows of the constraint system:

```
diag(w) C θ  ≥  diag(w) d
```

This is a no-op. For any `w_r > 0`, divide row `r` through by `w_r` and recover
`c_rᵀθ ≥ d_r` exactly. The feasible set is *identical* for every gate vector in
`(0,1]` - it changes only when a gate is exactly zero, which a logistic gate
never is. A gate of 0.001 binds precisely as hard as a gate of 1. Any
"relaxation" story told on top of this is fiction, and every downstream
diagnostic that reads the gate is reading a parameter that has no effect on the
constrained solution.

**The forms that actually work.** Two, and they are not equivalent:

- **Slack form** - `C θ ≥ d − (1 − w) · M` for a scale `M` large enough to make
  a row vacuous at `w = 0`. The gate now buys real slack, monotonically. This
  keeps the constraint hard where the gate is on.
- **Gate-weighted penalty** - drop the hard constraint and minimize
  `Σ_r w_r · max(0, d_r − c_rᵀθ)²`. The gate is a per-row penalty weight. Softer,
  differentiable end-to-end, and the one that fits Praxis's existing machinery.

The second is what we want, and we nearly have it already: the regularizer
registry (`REGULARIZER_REGISTRY`, `model.reg`) computes scalar penalties from
model state. What is missing is the *per-something weight* - per position, per
frequency band, per depth step - conditioned on a measured signal.

**Where this lands in Praxis.** Three candidates, cheapest first:

- **Harmonic delta L2.** Today one global coefficient holds the
  input-conditional delta beneath the static baseline. But the whole argument in
  the paper's §3.2 is that a static grid must *compromise across registers* -
  dialogue, code, prose. A uniform delta penalty re-imposes exactly the
  compromise the delta exists to escape. The gated version: weight the delta
  penalty by a signal that already exists - patch-boundary entropy from the
  byte-latent encoder, or the halting KL - so the model may deviate more where
  the content is genuinely novel and is held tighter where it is not. Falsifier:
  if the learned weight ends up flat across the corpus, the gate bought nothing
  and the global coefficient was right.
- **Weight decay.** The paper already argues for eliminating decay on
  geometry-bearing matrices, and calls the uniform shrink "the one-knob
  instrument this section has been arguing against." That is a *hand-partitioned*
  gate: we decided which matrices are geometry-bearing. The gated version learns
  the partition, or at least conditions it on a measured norm-growth signal, and
  the paper's own stated falsifier (norms growing without bound once decay is
  removed) becomes the gate's input rather than a manual tripwire.
- **Smoothness prior on the amplitude grid.** Currently uniform over the
  frequency lattice. There is no reason the corpus rhythm should be equally
  smooth at every band, and the `1/f^α` envelope already concedes as much - it
  is a *fixed*, hand-chosen frequency-dependent prior. Making it learned-but-
  bounded is the same move one rung down.

**The discipline that keeps this honest.** Gating cannot be allowed to switch a
constraint off precisely where it would otherwise bind - that is a gate learning
to evade its own prior, and it yields zero effective enforcement while reporting
full compliance. The published defense is to keep the gate low-capacity (a
logistic on a handful of summary statistics, not a network) so it can express
*regimes* but not *per-example escapes*. Ours should additionally log the
correlation between gate value and constraint violation; a strongly negative
correlation is the evasion signature.

---

## 2. Gap-conditioned trust in spectral diagnostics

This is the best idea in the source material and the one place where its author
tied a gate to a real theorem rather than to a story.

**Davis-Kahan.** When two eigenvalues of a symmetric operator are close, the
corresponding eigenvectors are ill-conditioned: the subspace they span is
stable, but the individual vectors within it are free to rotate, and the
rotation angle is bounded by a quantity that *diverges as the spectral gap
closes* (Davis & Kahan 1970; the modern statistical form is Yu, Wang & Samworth
2015). So any statement about an individual mode - "mode 7 carries the corpus
rhythm", "mode 12 is where the input-conditional variance lives" - is only as
trustworthy as the gap around it.

**Why this matters here specifically.** The paper's central claim is that bias
and variance occupy *orthogonal subspaces* of the harmonic latent - the frozen
time-invariant modes versus the time-varying ones - and that their loss
cross-curvature vanishes. That claim is a statement about a *mode partition*.
Near-degenerate amplitudes make the partition ambiguous, and a partition that
can rotate is one whose orthogonality claim cannot be measured cleanly. The
conjecture as written ("falsified the moment the cross-curvature between the two
mode bands stays bounded away from zero") is measurable only where the bands are
separated. It needs a gap condition or the measurement is not well-posed.

**Concretely, what to build:**

- Report the **amplitude gap profile** alongside the existing Hoyer
  concentration. Hoyer says how concentrated the spectrum is; the gap profile
  says whether the concentrated cells are *individually identifiable* or only
  identifiable as a block.
- **Gate the per-mode attribution on the gap.** Any dashboard card that assigns
  a role to a specific frequency cell should grey out or widen its error band
  where the local gap is below a threshold. This is a one-line change to the
  card and it prevents a whole class of over-reading.
- The same reasoning applies to the **crystal centers**: two centers at nearly
  equal distance from a token are not two distinguishable symbols, and the
  symbol-occupancy readout that [information_geometry.md](information_geometry.md)
  proposes inherits exactly this instability. A silent bit flip that moves a
  state between two *near-degenerate* centers is not the geometric event that
  note is hunting; it is measurement noise. Gap-conditioning separates the two.

This is the rare case where an outside constraint improves a Praxis conjecture
by *narrowing* it. The claim gets weaker and testable instead of broad and
unfalsifiable.

---

## 3. Block-sparse optimizer updates

Long-standing intuition, and the source draft's alternating scheme is a bad
implementation of a good shape. Its block-coordinate structure - cheap
structured projection alternating with a gradient step, per-block active sets,
warm starts across outer iterations - is worth stealing even though its
convergence theorem does not hold (the gate block is non-convex, so the cited
Tseng/Beck-Tetruashvili results do not apply; the correct references for this
algorithm class are Grippo & Sciandrone 2000 for two-block exact minimization
and PALM, Bolte-Sabach-Teboulle 2014, for the prox-gradient case).

**The Praxis version.** Compute updates at block level, sparsely selecting which
regions to touch, so that old information is not destroyed by a dense update
that overwrites everything to move a few coordinates. This is the same diagnosis
[continual_learning.md](continual_learning.md) records: *dense global credit
assignment is what overwrites.* A block-sparse optimizer is a cure for that
diagnosis that does not require co-designing the architecture around a one-hop
delta rule.

Sketch, in decreasing order of confidence:

1. **Per-block active sets.** Compute the full gradient; apply it only to blocks
   whose update magnitude clears a threshold, and carry the rest in an error-
   feedback buffer (the standard trick from gradient compression, so nothing is
   silently dropped - it accumulates until it clears). Blocks are the natural
   units the model already has: per-head, per-expert, per-frequency-band, per
   recurrent-depth signature.
2. **Selection by curvature rather than magnitude.** Update the blocks where the
   step buys the most loss reduction per unit of disturbance to what is already
   stored. The optimizer metrics suite already logs most of the signal a
   selector needs: `opt_momentum_grad_cos` is available across Lion, AdamW and
   SGD-momentum, and `opt_update_rms` / `opt_update_weight_ratio` on the
   Adam family. Note the gap - the current body optimizer is the Muon composite,
   which is *not* Adam-family, so the update-magnitude series a selector would
   most want is not being logged for the parameters that matter most. That is a
   prerequisite, not a detail.
3. **Warm-started, alternating structure.** Alternate a cheap projection (RMS
   normalization, orthogonalization - both already in the Muon composite path)
   with the sparse gradient step, warm-starting each from the last. The source
   draft claims 3-10x from warm starts; that number is unsupported there, but
   the mechanism is standard and the measurement is cheap to run here.

The honest caution: this interacts badly with the existing optimizer wrapper
stack if it is bolted on top rather than composed into it. It belongs as a
`WRAPPER_REGISTRY` entry, not a new optimizer, and not a new CLI flag - the
selection threshold should be endogenous (derived from the update-RMS
distribution) rather than tuned per experiment.

---

## 4. Structural reproduction, distinct from computational reproduction

Small idea, immediately actionable, and the one piece of the source draft that
is genuinely worth citing on its own terms.

A run that reproduces its **loss curve** has been *computationally* reproduced.
A run that also reproduces its **structure** - which cells the amplitude grid
lit, where the halting mass sat, what geometry the crystal centers grew into -
has been *structurally* reproduced. These are different properties, and the
second is the one that matters for every claim this project makes, because none
of our conjectures are about the loss.

Two runs can match on bits/byte and disagree entirely on whether the field
crystallized. Right now we would not notice.

**What to build:** a per-run **structural fingerprint** - a small fixed vector
of the diagnostics the conjectures actually name (Hoyer concentration per arm,
harmonic delta norm, early-halt rate, mean boundary KL, abstain rate, mean
radius against shell radius, per-depth acceptance profile) - recorded alongside
the metrics and diffable across runs. Not a hash; hashing is useless here
because floating point will never match bit-for-bit and the interesting question
is whether the *structure* matches, not the bits.

This gives the paper something it currently lacks. Every conjecture in §8.10
names a falsifier phrased in terms of these quantities. A fingerprint that is
recorded per run and stable across seeds turns "the claim is falsified the
moment X is measured to be flat" from a promise into a standing test.

---

## What not to take

**Index priors with a story attached.** The source draft's headline mechanism
regularizes a gate's loadings toward *prime-numbered* spectral indices, with a
long justification about arithmetic structure in physical spectra. The
mathematics is a two-level weighted lasso: coordinates in a designated set get
penalty `λ₁`, coordinates outside get `λ₁ + λ₂`. Substituting any other fixed
index set of the same size changes nothing - not the convexity, not the
proximal operator, not a single downstream step. The number theory is
decorative.

**Keep this as a test we apply to ourselves.** For any prior we impose on an
indexed structure - the frequency lattice, the depth signatures, the expert
bank - ask: *does the mathematics change if the index set is permuted?* If not,
the prior is a mask with a narrative on it, and the narrative is doing no work.

Praxis passes this test in the one place it matters most, and it is worth
knowing why. The `1/f^α` envelope is not an arbitrary index mask: it is
monotone in frequency, so permuting the lattice changes the prior. Same for the
Weyl phase seeding - `2π frac(f_t π + f_d e)` depends on the actual index
values through an equidistribution argument, and shuffling the lattice destroys
the equidistribution. Both are *derived* from a property of the index, not
*assigned* to a subset of indices. That is the distinction to hold onto.

---

## Open questions

- Is there a signal already logged that would make a good gate input, or does
  gating require a new probe? (Halting KL and patch-boundary entropy are the
  two candidates; both are already computed.)
- Does the gap-conditioned reading change any conclusion we have already drawn
  from a spectrum heatmap? Worth re-reading the existing Hoyer numbers with the
  gap profile beside them before building anything.
- Block-sparse updates and MonoForward both attack dense credit assignment from
  different directions. Do they compose, or does layer-local training already
  capture most of the benefit?

Related: [continual_learning.md](continual_learning.md) (the diagnosis this
shares), [information_geometry.md](information_geometry.md) (which inherits the
degeneracy problem), [harmonic_koopman.md](harmonic_koopman.md),
[integration_backlog.md](integration_backlog.md).
