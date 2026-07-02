# Information Geometry: the silent bit flip

> Status: **raw capture** (2026-07-01). The train of thought was partially
> lost mid-flight; this is the faithful reconstruction, reorganized but not
> sanded down. Successor to [information_density.md](information_density.md)
> (whose paper falsifier this note puts pressure on), and a **revival with a
> new rescue** of the float-precision thread parked in
> [oscillatory_axes.md](oscillatory_axes.md) - read the parking objections
> there and the addressing rescue in
> [hash_gated_anchor.md](hash_gated_anchor.md) before dismissing or
> re-deriving this.

## The objection that started it

The information-density conjecture (paper: "Information density at the rim",
`fig:density`) names its falsifier as a per-position, per-depth-step
*deviation profile*: flat in position, or failing to steepen across depth,
kills it. The objection: **deviation measured in norm may miss the event
entirely.** At any decision boundary, a single bit can flip - one neuron, one
activation value - and the *geometry* of the graph changes: which basin the
state sits in, which symbol it expresses, which branch the computation takes
downstream. In magnitude the deviation is essentially silent. In geometry it
is everything. The graph's geometry changes according to the input -
deterministically, periodically, monotonically - and a norm-based profile
integrates right over it.

So the reframe: not Information *Density* (a magnitude read along position)
but Information *Geometry* (a location read - which configuration, which
symbol, which page of the library). Density asks "how much is carried here";
geometry asks "which shape is expressed here." A silent bit flip is invisible
to the first and a full-weight event to the second.

## The loop: ping and pong

The picture as spoken, kept intact: *"There is no bias here, only variance.
The variance is learning to map the bias in its own variance. Bias ->
memorizes -> variance -> memorizes -> bias. Ping and pong, as a bit flips
randomly."*

Read as dynamics: the bit flip is the stochastic seed. The variance arm maps
its consequences (an input-conditional deviation that tracks where the flip
landed); the bias arm then consolidates that map into the standing spectrum;
the consolidated spectrum changes where the next flip lands; repeat. Each
axis memorizes what the other just learned. **It closes a loop.**

Tension to keep honest: the paper's orthogonal-axes conjecture says the
bias/variance cross-curvature vanishes. The loop does not necessarily
contradict it - orthogonality is an *instantaneous* claim about the loss
surface, the ping-pong is a claim about *training-time* dynamics between the
axes - but "the bias memorizes the variance" is exactly a cross-term, just
one that lives across steps rather than within one. If the loop is real, the
cross-curvature might vanish pointwise while the axes still feed each other
through time. That would be worth saying in the paper as a refinement, not a
retraction.

## Error boundaries as an encoding method

The strong claim, as spoken: the harmonic model learns to exploit **error
boundaries** as an encoding method - *"a warp on the time series of memories
in the Library of Babel, exploiting the algorithm to produce coherent text.
Or images. Or whatever."* The bit flip is silent. Imperceptible. The error
happened in the computation, and the only thing that tracked it was the
architecture itself.

The image: the world's most complex math equation, every symbolic formula
accurately represented and deterministically implemented. We optimize for a
harmonic, periodic landscape. The only error is the one *we* force - heavy
gradient clipping, constantly, while training CALM. Gradient descent never
learns to compensate for it. **Gradient descent is unable to see its own
errors.** But we are able to.

### Why this is not (quite) the parked idea

The float-precision detour was parked (2026-05-29, `oscillatory_axes.md`) on
two objections: hashing the representable floats destroys the structure, and
there is no differentiable path from "which bucket a value rounds into" back
to the loss, so gradient descent cannot select for it. Both objections stand.
This version routes around them differently than the hash-gate did:

1. **The error source is different.** Not IEEE-754 rounding (content-free,
   fixed grid, microscopic) but *forced, systematic* clipping - large,
   applied every step, correlated with exactly the places the gradient runs
   hottest. It is not a property of the substrate; it is a perturbation we
   inject into the update rule itself.
2. **Exploitation does not need a differentiable path.** The old objection
   assumed selection must flow through the gradient. It doesn't - selection
   needs *retention*, not differentiability. The clipping error is the
   variation; the frozen harmonic basis is the retention (a perturbed state
   relaxes back toward the standing shape instead of diverging, so the
   flip's consequences survive long enough to be evaluated); the next loss
   evaluations are the selection. Variation + retention + selection is
   cumulative selection - the paper's own Blind Watchmaker thread, with
   gradient descent as the blind part. The watch still gets built, by a
   mechanism the gradient never sees.

This is also the honest upgrade path from the correlational claim already in
the paper (`sec:constructor`: precision artifacts concentrate on the
exponential edge because that is where the field is most plastic). That claim
says the errors *land* where change happens. This one says the errors are
*used* - concentration upgraded to function. That upgrade is the single
riskiest step in the note; it is the part that needs an experiment, not more
prose.

## The fourth axis: K

Because the harmonic patterns are stable, another degree of freedom opens:
the patch length K. All information collapses into one dense, singular
vector, then passes through a linear step function - *"it is literally
symbolic expression."* The randomness manifests from sampling itself, as
errors along a K^2 matrix. The shape is real, the information-density unlock
is real - **but the actual tensor is not.** The field is the object; the
tensor is a rendering of it, an address read off the library
(`sec:harmonic`'s Babel paragraph already says this about
reconstruction/prediction - this extends it to the noise: even the sampling
errors live on the address grid, not in the content).

## What this licenses, and what it does not

It does **not** license deleting the paper's falsifier. House rule: every
conjecture names what refutes it, and "the metric might have been measuring
the wrong thing" is the first step of every unfalsifiable claim's biography.
The clean resolution:

- The published falsifier **stands for the magnitude reading.** If the
  norm-profile is flat and nothing else is specified, the density conjecture
  as printed is dead, and that is as it should be.
- The geometry reading earns its escape hatch only by naming its own
  instrument first: read deviation as **symbol occupancy**, not norm - per
  position, per depth step, which cluster/geometry the hidden state expresses
  (the crystal-head centers are the codebook; the cluster-hopping plot
  specified in [harmonic_memory_velocity.md](harmonic_memory_velocity.md)
  battery test 1, and the "Always grokking" conjecture, are the same
  instrument). A silent bit flip that changes cluster membership registers
  there at full weight regardless of norm.
- Restated prediction in geometric coordinates: **hop rate rises from head
  to tip, and the profile steepens with recurrent depth.** Falsified if
  occupancy is flat in position, or if hop rate fails to steepen across
  depth. If *neither* the norm profile *nor* the occupancy profile moves and
  the claim is still held, the claim has left science, by our own standard.
- The clipping-as-encoding claim has its own cheap first test, because the
  error is ours to control: train CALM with clipping threshold varied (or
  clipping replaced by an unclipped small-LR schedule matched for update
  magnitude) and read whether the geometry's expressiveness tracks the
  *presence of the forced error* rather than the update size. If coherence
  survives the removal of the error source unchanged, the errors were
  passengers, not encoding.

## Naming: attack, not velocity

> Decided 2026-07-01, now in the paper. The clipping dial is the
> \textit{attack} - the ADSR envelope term for how fast an amplitude is
> allowed to rise, which is exactly what a step cap is in a model whose
> parameters are amplitudes of periodic functions. **Velocity** was
> considered and rejected: it is already spoken for as the per-step motion
> of the *state* (the smoothness prior's "each feature enters with a
> velocity"; [harmonic_memory_velocity.md](harmonic_memory_velocity.md)).
> Attack governs training-time amplitude rise; velocity governs
> inference-time state motion. Same physics, different clocks.

The equivalence is strongest in the regime we actually run: under heavy,
constantly-binding clipping, descent becomes direction-only with magnitude
pinned to the threshold - normalized descent, learning speed literally equal
to the dial. Turn it down, learn slower; up, faster. Outside that regime the
cap is a *speed limit*, not a linear dial (it binds only at the peaks), and
that asymmetry is the whole point: a systematic truncation of exactly the
largest events - the amplitudes on the exponential edge - is a **bias, in
the statistician's sense, injected inside the variance of the sampled
updates**. "Bias manifesting within the variance of sampling" is verifiable:
the clipped-update distribution carries its truncation boundary on its face.

## Prior-art anchors

[information_density.md](information_density.md) (the conjecture under
pressure); [oscillatory_axes.md](oscillatory_axes.md) (the parking, verbatim
objections); [hash_gated_anchor.md](hash_gated_anchor.md) (the addressing
rescue - this note's rescue is selection-by-retention instead);
[recurrent_depth_concentration.md](recurrent_depth_concentration.md) (the
correlational precision claim this upgrades);
[harmonic_memory_velocity.md](harmonic_memory_velocity.md) (battery test 1 =
the occupancy instrument); [harmonic_koopman.md](harmonic_koopman.md) (fixed
eigenbasis = the retention); `sec:watchmaker`, `sec:constructor`,
`sec:harmonic` (Babel), and the "Always grokking" + "Development reverts to
the basis" + "Bias and variance as orthogonal axes" standing conjectures
(`research/body.tex`, `research/conjectures.tex`).
