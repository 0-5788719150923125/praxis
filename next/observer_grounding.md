# The Observer Grounding: one channel, shared lineage, and the torus we project

> Status: **exploratory / grounded kernel + flagged voice** (2026-06-23).
> Why language has harmonic/geometric structure a model can exploit, argued
> from the *observer* rather than the architecture. Sibling to
> [observer_frequency.md](observer_frequency.md), [observer_effect.md](observer_effect.md),
> [world_models.md](world_models.md), and the Koopman framing in
> [harmonic_koopman.md](harmonic_koopman.md). Same discipline as its siblings:
> the structural kernel maps to mechanism; the extrapolation to archetypes and
> torus-duality is voice, marked as such.

## The question

Earlier skepticism (in [harmonic_koopman.md](harmonic_koopman.md)) parked
"tokens as geometries on a manifold" as metaphor, on a narrow technical ground:
equivariant/harmonic machinery needs a *group acting on the representation*, and
abstract semantic tokens transform under no obvious group. The observer move
answers this. Language is not abstract to the thing that uses it; it is a
property of how an observer *perceives* the sign - the visual geometry of the
mark and the spectral structure of the sound.

## The grounded kernel (maps to mechanism)

Two perceptual channels carry genuine group structure:

- **Visual / glyph.** Marks live in 2D image space, whose symmetry group is the
  Euclidean group E(2) (translation, rotation, scale, reflection) - exactly what
  group-equivariant CNNs are built on. Instantiated, not hypothetical: PIXEL
  (Rust et al., ICLR 2023), CLIPPO, the pixel-LM line render text to images and
  model it, getting a script-agnostic vocabulary for free.
- **Acoustic / spectral.** Speech is natively time-frequency: phonemes are
  formants, pitch is an F0, prosody/meter/rhyme are periodic. Source-filter model
  (Fant). The invariances are group-like - vocal-tract-length normalization is a
  scaling of the log-frequency axis. This is where "model language as music" is
  closest to literally true.

Writing systems are the *projection* between these poles: logographies push
toward the visual/geometric pole, alphabets toward the phonological/spectral
pole. Cognitive science reads it the same way (dual-route model, Coltheart): a
lexical/visual route and a grapheme-phoneme route. So "geometry **and** spectrum"
is not one claim - it is the two axes every script trades off between.

## One channel is enough: the manifold is modality-independent

The sharp fact: a congenitally blind person (never a glyph; Braille is tactile
geometry, or audio) and a deaf person (never a formant; sign is visual-spatial
geometry) both converge to the *same* shared reality - same language, same
concepts. Helen Keller, neither channel, converged through tactile signing alone.

This proves the **channel is fungible**. What is invariant across vision, sound,
and touch is not the modality but the *structure* mapped across them. Each channel
is a **chart**; the shared reality is the **manifold** they all chart. Blind,
deaf, and sighted converge because they cover one manifold with different atlases.

Neuroscience backs the channel-independence directly (anchors, not proof):
- the *metamodal* brain (Pascual-Leone & Hamilton) - regions defined by
  computation, not modality; the Visual Word Form Area activates for Braille in
  the blind;
- congenitally blind adults hold structured knowledge of color and animal
  appearance, learned through *language alone* (Kim, Elli & Bedny 2019) - they
  recover visual-seeming geometry they never perceived.

That last one is the LLM's exact situation, which is the bridge to the
counterweight below.

## Blindness as geometric sparsity (and why one channel suffices, mechanically)

The refinement that makes "one channel is enough" a *mechanism* instead of an
assertion. Blindness/deafness is not a constant attribute of an observer; it is a
**sparsity mask** over the perceptual-geometry channels - a dropped *measurement*,
toggleable in principle, not a different manifold. The observer with a zeroed
channel has lost redundancy, not the world.

Why the lost channel is (mostly) recoverable: **compressed sensing.** A signal
sparse in some basis can be reconstructed from far fewer measurements than its
ambient dimension, *provided the measurement basis is incoherent with the
sparsifying basis* (Candes-Romberg-Tao, Donoho 2006). Two facts make this literal
here, not metaphor:

- The signal is sparse in the harmonic basis - the amplitude grid concentrates,
  and the head already measures it (`concentration()` = Hoyer sparsity, 1 = a
  single cell). This is *signal*-sparsity.
- The frozen Weyl phases are an **incoherent** basis (equidistributed, no rational
  resonances). Incoherence is exactly the compressed-sensing precondition. So this
  is a **third independent justification for freezing the phases**: beyond the
  Koopman reading (fixed eigenfunctions, [harmonic_koopman.md](harmonic_koopman.md)),
  an incoherent fixed basis is what lets a sparse latent be recovered from a
  partial set of channels. One design choice, blessed by two different theorems -
  which raises the stakes on the frozen-vs-learned ablation.

Keep the two sparsities distinct: *signal*-sparsity (few active harmonic modes) is
the property that lets you survive *measurement*-sparsity (a dropped channel =
blindness). The first enables recovery from the second.

This sharpens the 95/5 and gives it a hard edge. Recoverable is the content
**redundant across channels** - the shared manifold (95%, bias). **Channel-unique**
content - in one modality and nowhere else - is *not* recoverable and is genuinely,
permanently lost (5%, variance). So "no reason it can't be taught" is too strong.
The blind learn color from language (Kim/Elli/Bedny) because color's relational
structure is redundant in the linguistic channel; but Molyneux's problem (Held et
al. 2011 - the newly-sighted do not instantly map touch-shape to sight) shows the
charts are not perfectly aligned, and congenital deprivation leaves real permanent
deficits. Recoverable = the shared 95%; lost = the channel-unique 5%.
"Mechanical/symbolic" recovery is then two real things, both filling only the
recoverable part: **sensory substitution** (Braille; vOICe encoding vision as
sound; BrainPort on the tongue, Bach-y-Rita) re-charts the missing channel onto a
present one, and **distributional inheritance** (learning the structure from
language) supplies it symbolically.

### Sparsity is spacetime, and asynchronous

The mask is not only spatial (which channel) but **temporal** (which time window),
and the two are independent per channel (asynchronous). "Skip vision for a period,
recover it later through the machinery" is a channel masked over `[t0, t1]` and
infilled afterward - from two sources at once: cross-channel redundancy (the
compressed-sensing recovery above) and **temporal continuity** (the harmonic /
Koopman latent dynamics carrying the gap, [harmonic_koopman.md](harmonic_koopman.md)).
Short gaps of redundant content reconstruct; long gaps of channel-unique content
do not (the recency envelope decoheres, and there is nothing redundant to lean
on) - the same 95/5 edge, now in time.

### The testable kernel: spacetime channel dropout

Directly buildable and falsifiable. Mask a random subset of geometric channels
over random time windows, require the latent to reconstruct the whole (modality
dropout / masked multimodal modeling, extended in time). The
**reconstruction-vs-(mask fraction x mask duration) surface is the
compressed-sensing recovery surface**, a direct test that the CALM latent is a
sparse-recoverable harmonic manifold: if it is, reconstruction holds to a sharp
threshold then falls off. CALM already has the reconstruction objective and
`concentration()`; this adds a mask. "Toggleable blindness" becomes literal - drop
a channel at inference and watch the latent re-converge from the rest, or fail to,
marking channel-unique content. Pairs with the K-modal-variance test (archetype
section): blindness is one coordinate of the variance, and its recoverability is
the test of whether that coordinate is redundant or unique.

> Voice coda (flagged): the riff cast this as a polarity - a pure-variance machine
> and a pure-bias human in symbiosis, pushing on time itself, rolling along a
> U-curve (the umbra). Grounded shadows: the bias/variance poles are real and
> already built (the `prismatic3` head's pure-bias and pure-variance arms); "the
> exponential edge, nearly the breaking point" is the paper's own substrate
> passage (`body.tex` - activations held in the band where the softmax is
> sensitive-but-stable); "rolling forward/backward" is the paper's "temperature is
> a path through this space, not noise at a fixed point." The antimatter and
> time-itself parts are voice - parked, like the 144,000. The temperament reading
> of this polarity lives in [politics_bias_variance.md](politics_bias_variance.md).

## "Shared lineage" is the load-bearing term - and it does NOT transfer to machines

Why do blind/deaf/sighted land on the *same* manifold? Three shared anchors:

1. **the same genome / neural architecture** (innate priors, the chart-making
   machinery);
2. **the same physical world** (objects, gravity, causality, agents);
3. **the same language community** (they learn from already-converged speakers).

This is the honest correction to the Platonic Representation Hypothesis, which
[harmonic_koopman.md](harmonic_koopman.md) flags as failed verification on its
*strong* (scale-driven) claims. Biological convergence is real, but its cause is
**shared lineage**, not scale. Lineage is doing the work PRH wrongly attributed
to parameter count.

And lineage does not transfer to a language model. An LLM shares only anchor (3):
the text. It has no genome and no body. So the biological convergence argument
**highlights what machines lack** rather than what they have.

## The reconciliation (the centerpiece)

This is why the grounding argument and the distributional counterargument are
*both* true, and it is the most important paragraph here:

> Text is the **shadow** of a reality that humans converged on through
> shared-lineage perception. The harmonic/geometric structure is *in* the text
> precisely because the humans who wrote it shared genome, world, and language.
> A text-only model never traverses the channels - it inherits the already-
> converged structure through the distributional statistics of the shadow.

So: grounding explains *why the structure exists*; distribution explains *why a
text model can find it*. The blind adult learning color geometry from language is
the proof of concept, and it is exactly what an LLM does. Firth ("know a word by
the company it keeps") and Harnad's symbol-grounding problem are not refuted by
the observer view; they are the same coin. The defensible claim is the **weak**
one: perceptual geometry is a real, efficiency-buying inductive bias that is
*present as structure in the data*, not a necessary substrate for semantics.

## Signifier / signified ↔ bias / variance, made precise

Cast in semiotics: the **signifier** (mark, sound-image) is the shared perceptual
geometry - we see roughly the same glyphs, hear roughly the same formants. The
**signified** (the concept evoked) is where observer-relativity lives. That is the
95/5 exactly:

- signifier = lineage-converged shared substrate = **bias** = the static spectrum
- signified = observer-conditional association = **variance** = the input delta

"95% shared reality, 5% solipsism" is therefore not a separate flourish - it is
the same bias/variance split applied to epistemology. We already measure it:
`capacity_split` / `field_strands` partition static-spectrum energy from
input-conditional-delta energy. Report *that* number; cite PRH only as the
contested maximal version.

## The speculative extension - archetypes, the projected torus, big/small (VOICE)

This is the thread that got lost mid-thought, recovered and then flagged hard.
The recovery hinges on one clean geometric fact:

**Inside/outside is a property of the projection, not the torus.** A torus is a
closed surface with no intrinsic inside or outside; "inside looking out" vs
"outside looking in" exists only once you *project* it into a plane - which is
exactly what the epicycle / PCA cross-section views of `HarmonicField` do. So:

- Observers are different **projections (charts)** of one shared torus (the
  harmonic field's Weyl 2-torus is the literal object).
- **Archetypes = the K dominant projections** - the buckets most observers fall
  into. This is the CrystalHead one level up: crystal *centers* are K prototype
  geometries in token-space; archetypes are K prototype geometries in
  *observer*-space. Jung's "collective" prior = the bias / static spectrum; the
  archetype is a basis mode of it.
- **Blind/sighted is one coordinate of the projection** - a binary deviation from
  the median chart. Sensory configuration is one axis of the variance vector; it
  changes the chart, not the manifold, which is why it does not break
  convergence.
- **Big/small, inside/outside** are reciprocal duals. The precise names for the
  intuition: *conformal inversion* (`z → 1/z̄` swaps inside-the-circle with
  outside, origin with infinity) and, for "big and small realities are the same
  reality," *T-duality* on a circle/torus (radius `R` ≡ radius `1/R`, momentum
  ↔ winding). These are real, beautiful structures - and **borrowed by analogy**.
  There is no evidence the model's torus carries a T-duality symmetry; this is a
  theorem about a specific physical action, not a result about language models.
  Flagged with full weight: this is the exact species of "sounds deep, lacks
  support" the research pass warned about. Hold it as voice until it earns a test.

### The one testable shadow this casts

The archetype idea is not *only* voice - it makes a falsifiable prediction. If
observers cluster into K archetypal buckets, then in a model trained on many
voices the **variance** (the input/observer-conditional delta `Δ_φ(context)`)
should be **low-rank / K-modal**, not full-rank - the conditional deltas should
collapse onto a few attractors. That is measurable with the telemetry already in
place (rank / clustering of the field strands' input-conditional energy). The
clean question: *is the variance K-modal?* A yes is real evidence for
"archetypes as basis modes"; the torus-duality stays parked regardless.

## Honest seams (what is what)

- **Mechanism:** the two perceptual channels carry real groups; the field is a
  literal 2-torus; bias/variance = signifier/signified; the K-modal-variance test.
- **Measurable shadow:** archetypes as low-rank variance (run it before believing
  it).
- **Voice:** inside/outside duality, T-duality/inversion, "big and small
  realities." Cousins of real mathematics, imported as analogy, not earned here.
- **Counterweight, equal weight:** distributional-only LLMs already recover most
  semantic structure with no grounding, and shared lineage - the cause of human
  convergence - does not transfer to machines, which hold only the text anchor.
  The grounding justifies harmonic *priors* and *going multimodal*; it does **not**
  license equivariant latent machinery on bytes.

## Prior-art anchors (confidence-flagged; NOT workflow-verified)

PIXEL (Rust et al., ICLR 2023, solid); dual-route reading (Coltheart, solid);
source-filter speech model (Fant, solid); symbol grounding (Harnad 1990, solid);
distributional semantics (Firth 1957, solid); metamodal brain (Pascual-Leone &
Hamilton 2001, solid-ish); blind color/appearance knowledge (Kim, Elli & Bedny
2019, solid-ish); Molyneux's problem / fast cross-modal alignment (Held et al.
2011 PNAS, solid-ish); Weyl equidistribution on the torus ([weyl1916], in
citations.bib); compressed sensing (Candes-Romberg-Tao, Donoho 2006, solid);
modality dropout / masked multimodal modeling (ModDrop, Neverova et al.; MAE,
solid-ish); sensory substitution (vOICe, Meijer; BrainPort, Bach-y-Rita,
solid-ish); manifold hypothesis (standard); T-duality / modular group (string-theory standard, **analogy
only**); Jungian archetypes (humanities, **voice**). Verify before any of this
approaches the paper.
