# Text diffusion on the abstractinator line

> Status: **STAGE 1 BUILT, 2026-09-13.** The objective, the corruption, the
> decode loop, the metrics and the tests are in the tree and
> `./launch --abstractinator-d` runs. Nothing has trained to convergence yet -
> the first real run is the point of `-d`. What exists is listed under "What was
> built" at the bottom; the rest of this note is the design it was built from,
> left as written. Written after
> the [TemporalMesh audit](temporal_mesh_audit.md), which is where the question
> came from: if causality is a consistency requirement between architecture and
> objective rather than a law, what does it look like to drop *both* halves
> honestly? Companion to [harmonic_koopman.md](harmonic_koopman.md) and
> [kaleidoscope.md](kaleidoscope.md).
>
> **The one thing to take from this note:** the reason to build this here rather
> than anywhere else is that `MultiStageResidualVQ` already has a coarse-to-fine
> ladder (`depth` stages over a `K`-entry codebook) and `HarmonicResidualVQ`
> already treats `latent_dim` as a **low-frequency spectral budget**. A diffusion
> schedule is normally a tuned hyperparameter. On this stack it can be
> *structural* - stage 0 before stage 1, low bands before high - which is the
> only version of this idea that passes our own no-hand-tuning rule.

## Is the mental model right?

Mostly, with one correction that changes the amount of work involved.

Right: every forward sees the whole sequence, there is no causal mask, and
generation is iterative refinement over the full sequence rather than an
extension of a prefix.

The correction: for text, continuous Gaussian denoising is not the formulation
that works. The line that won is **absorbing-state (masked) diffusion** - D3PM's
absorbing variant, then MDLM, then LLaDA and Dream. The corruption is not
additive noise, it is **replacement by a mask symbol**. Forward process:
progressively replace tokens with `[MASK]`. Reverse process: predict every
masked position at once, then *commit* a subset (usually the most confident) and
re-run with fewer masks.

Two consequences worth internalising before costing this out:

1. **The training objective is a reweighted masked cross-entropy.** MDLM's
   result is that the diffusion ELBO collapses to a weighted sum of masked CE
   terms over a randomly sampled mask ratio. We already own cross-entropy. The
   training-loop delta is: sample a corruption level, corrupt, drop the shift,
   weight the loss by the level, restrict it to corrupted positions. That is
   much less than a rewrite.
2. **The expensive changes are inference and evaluation**, not training. The
   decode loop is new, and the reported number is an ELBO *bound*, not a
   likelihood. See "The evaluation trap".

There is a real fork here, because the abstractinator gives us both a discrete
and a continuous target. Discrete masked diffusion over RVQ **code indices**, or
continuous flow matching over the **pre-quantisation latent**. Recommendation:
**discrete, over codes.** The loss is CE, which we own and which has no variance
problem; it sidesteps the ceiling documented in the CALM arm (bound `z_c`
structurally, not by KL); and it keeps the quantizer, whose telemetry we already
have, as the thing that defines the alphabet.

Note that `praxis/generators/flow.py` is **not** a head start on the loop. It is
a per-position generator conditioned on an AR hidden state - it maps one
conditioning vector to one latent sample. What is reusable from it is the
noise-level conditioning machinery (`TimestepEmbedder`, the `modulate()` AdaLN
pattern, the zero-init-so-identity-at-init discipline), not the structure.

## Why build it

What autoregression cannot do, structurally, no matter how good the attention is:

- **Revise.** An AR model can never fix a token it has emitted. Every error is
  permanent and conditions everything after it. A diffusion model can remask and
  redecide.
- **Use bidirectional context legitimately.** The thing TMT tried to steal. Here
  it is free, because the objective hides the token it asks for.
- **Decouple compute from length.** AR needs S sequential steps. Diffusion needs
  T refinement steps, all positions in parallel, and T can be far below S. This
  is the whole throughput argument, and it is the claim most likely to fail on
  contact - see "How we would know it is not working".
- **Infill and constrain natively.** Editing, infilling and constrained
  generation are the default behaviour, not a decoding trick.

Against that: no KV cache in the usual sense, T full forwards per sample, and
the headline number is a bound rather than a likelihood.

## Why the abstractinator line specifically

Diffusion cost is roughly `S x T x layer_cost`, so sequence length is the term
that decides whether this is affordable on one 5060 Ti. That is exactly the term
the abstractinator already attacks.

1. **S drops.** We diffuse over patches/codes, not bytes. The byte path is what
   makes token-level diffusion unaffordable here.
2. **The alphabet is a codebook, not a vocabulary.** Masked diffusion over `K`
   entries per stage is far better conditioned than over 50k tokens - the
   per-position marginal is small enough that confidence-ordered unmasking has
   something to rank.
3. **The RVQ depth ladder is already a schedule.** `MultiStageResidualVQ(D, depth,
   K)` is a coarse-to-fine residual decomposition with `K_eff = K**depth`. Stage
   0 carries the coarse content, later stages the residuals. That *is* a noise
   schedule with a structural justification, rather than a cosine curve someone
   picked. This is the single strongest reason to build it here.
4. **The decoder already exists.** Codes to bytes is solved; we are only
   replacing how the code sequence is produced.

**On the unstable versions.** v2 (GDN dropped) and v3 (v1 + next-code HALO) are
where the interesting encoder work is, and they are unstable. Do not compound two
unknowns: **build stage 0-2 below on v1**, which is stable, because the diffusion
question is orthogonal to what v2 and v3 change. If diffusion works on v1 it can
be lifted; if it fails on v3 we will not know which half failed. The codebook
collapse history matters doubly here - a collapsed codebook does not just hurt
reconstruction, it makes the diffusion alphabet degenerate, so the pseudo-count
and first-batch-seeding fixes are prerequisites, and VQ telemetry is a gate on
starting rather than a nice-to-have.

## The harmonic tie, and why it is not decoration

`HarmonicResidualVQ` quantises in a fixed harmonic frame: `z = h @ analysis`
projects onto harmonic coordinates, and its own docstring says `latent_dim < D`
makes the frame lossy as **"a low-frequency spectral budget"**. That sentence is
the whole argument.

In a harmonic basis, **the diffusion timestep is a frequency cutoff.** Corruption
is discarding high bands; denoising is restoring them coarse to fine. This is not
an analogy imported from image diffusion - it is what `latent_dim` already means
in that file. So the schedule we would otherwise tune is available as: admit
bands in frequency order, admit RVQ stages in depth order, and let the two be the
same ladder.

The Koopman half gives a **falsifiable prediction**, which is what makes this
worth doing rather than merely pleasant. Koopman says the dynamics are linear in
the lifted observable space, with the phase frozen and the amplitude learned. If
that holds for the latent, then the reverse diffusion step in the harmonic basis
should be close to *a linear operator plus a small correction* - a per-band gain,
not a full nonlinear denoiser. Concretely:

- **Prediction.** The learned denoiser's Jacobian, expressed in the harmonic
  frame, is near-diagonal, and its off-diagonal energy falls as training
  proceeds.
- **Measurement.** Off-diagonal energy fraction of the denoiser in the harmonic
  basis, per band, as a `training_metrics()` series.
- **If it holds**, the denoiser can be *structurally* constrained to a banded
  operator, which collapses most of the per-step cost and makes T cheap. That is
  the version of this that would actually be fast.
- **If it fails**, we have measured the spectral-attractor conjecture in a
  setting where it makes a sharp prediction, which is worth more than another
  qualitative run.

This is the same conjecture as the memory-velocity note, asked somewhere it can
be answered with a number.

## Why it might beat Kaleidoscope, and why that framing is wrong

It should not replace Kaleidoscope, and the note should not pretend otherwise:
**they are on different axes.** Kaleidoscope is attention structure. Diffusion is
sequence factorisation. A diffusion model still needs attention, and kaleidoscope
is a candidate for it.

The honest version of "better" is narrower and more interesting: **dropping
causality is a cleaner test bed for the kaleidoscope claim than AR is.** The
frozen `[T, T]` mirrors were never motivated by causality, and
[const [t, t]](kaleidoscope.md) says frozen-QK beats standard. A non-causal
denoiser removes the constraint that has been shaping every kaleidoscope result
so far, so if fixed structure really is the win, it should be *more* visible
here, not less. `kaleidoscope.py:455` already reads `config.causal` and branches
on it at line 778, so this arm costs almost nothing to try.

So: a diffusion arm, with kaleidoscope attention inside it, as a new arm - not a
replacement for anything. Existing arms stay.

## Where the changes go

The seams are mostly already there, which is the pleasant surprise of writing
this up.

**1. Causality is already a flag.** `praxis/configuration.py:167` sets
`self.causal = False` as the default, and `praxis/modeling.py:255`
(`PraxisForCausalLM.__init__`) is the *only* place forcing it True. The attention
stack already honours it: `causal.py:51`, `ssog.py:269`, `kaleidoscope.py:455`,
`core.py:32`, `modular.py:43`, `pk_attention.py:41`. So the entry point is a
sibling class - `PraxisForDiffusionLM` - that simply does not force the flag.

  Two known leaks to fix first: `praxis/attention/components.py:464` hardcodes
  `is_causal=True`, and `:491` infers it from shape. Both bypass `config.causal`.

  **Do not weaken the causality tests.** `tests/attention/test_registry.py::test_causal`
  and `tests/test_modeling.py::test_inference_is_causal` must be parameterised on
  the flag so the causal arm is still asserted causal - the TMT audit is the
  argument for why.

**2. Objective.** A `DiffusionObjectiveMixin` beside `CausalObjectiveMixin`
(`praxis/objectives.py:259`). The existing `_compute_loss` (line 343) shifts by
one - `logits[..., :-1, :]` against `labels` - and diffusion does not shift at
all; it scores corrupted positions in place. `outputs_are_aligned` on the encoder
is the existing concept for "no shift", so the alignment branch is already there
to extend. Note `_compute_bidirectional_loss` at line 379 is *not* this: it is
forward+backward next-token prediction, still causal on both passes.

**3. Loss.** New `praxis/losses/masked_diffusion.py`, registered in the losses
registry and reached through `model.criterion` like everything else. It is CE
restricted to corrupted positions, weighted by the corruption level.

**4. Corruption schedule as a registry namespace.** Follow
`praxis/generators/__init__.py` as the template for `registry.declare`. Entries:
`uniform_mask` (the MDLM baseline), `rvq_stage` (coarse-to-fine by quantizer
depth), `harmonic_band` (by frequency band). Behaviour lives in the profile,
sizes stay flags.

**5. Noise-level conditioning.** Lift `TimestepEmbedder` and `modulate()` out of
`praxis/generators/flow.py` into a shared module once there are two consumers,
and condition the block via AdaLN on the level. Keep the zero-init discipline so
the conditioning is identity at init.

**6. Decode loop.** `BaseEncoder.decoding_method()`
(`praxis/encoders/base.py:166`) is the declared seam, and
`modeling.py:689 _resolve_decoding_method` already routes an encoder-owned loop
ahead of `_sample`, ahead of speculative decode. CALM already uses it
(`abstractinator/calm.py:575`). The iterative unmask loop goes there and gets
HF's prompt handling, stopping criteria and streamer for free. Streaming is
genuinely different though - tokens are not finalised left to right, so the
streamer publishes revisions, not appends. Worth deciding early whether the
terminal and the web UI can show that honestly.

**7. Metrics.** `training_metrics()` on the computing module, declared in
`praxis/metrics/training_metrics.py` so the charts pick them up. The ones that
would actually tell us something: commits per refinement step, unmask-order
entropy, remask rate, per-stage codebook usage, and the harmonic off-diagonal
energy from the Koopman prediction above.

**8. Data - one repo-specific trap.** Sample the corruption level **inside the
model's forward**, not in the collator. Lightning builds batch N+1 before the
hooks for N run, so a level sampled in the collator will be logged against the
wrong step and every schedule chart will be off by one batch.

## The evaluation trap

A masked-diffusion model reports an **ELBO bound**, not a likelihood. It is not
comparable to our AR perplexity, and putting them on one chart would be the exact
error the TMT audit is about - two numbers that do not measure the same quantity,
plotted as if they did.

So: a separate metric key, so the dashboard cannot silently co-plot them. Compare
a diffusion run against itself and against a *masked* baseline, never against the
AR runs. If we want a cross-family number, it has to be generation quality under
a fixed compute budget, measured the same way for both.

## How we would know it is not working

Name the failure modes before building, so a bad run is diagnosable rather than
discouraging.

- **The marginal-copying degenerate.** The model learns to copy visible context
  and emit the unigram marginal at masked positions. Diagnostic: at high mask
  ratios, compare against an actual unigram baseline. If it does not beat it,
  nothing is being learned about structure.
- **T collapses to S.** If quality only arrives when the number of refinement
  steps approaches the sequence length, the throughput argument is gone and this
  is just a slower AR model. Measure quality against T early, on the smallest
  config, before building anything else.
- **Independent-commit damage.** Committing k positions in one step assumes they
  are conditionally independent given the context, which they are not. This is
  the known weak point of masked diffusion and the reason confidence-ordered
  unmasking exists. Diagnostic: quality against commits-per-step. If it is only
  good at one commit per step, see the previous bullet.
- **Codebook degeneracy.** Covered by existing VQ telemetry, but it is now a
  gate: a collapsed codebook makes the alphabet trivial and the loss will look
  deceptively healthy.

## Staging

Each stage answers one question and can be abandoned without wasting the next.

- **Stage 0 - does the stack run non-causally?** Take an existing small config,
  do not force `config.causal`, fix the two `components.py` leaks, and train
  plain masked CE at a *fixed* mask ratio. No diffusion, no schedule, no
  abstractinator. This is a BERT, and the only question is whether anything else
  in the stack quietly assumed causality.
- **Stage 1 - is it a diffusion model?** Random mask ratio plus the weighted
  loss, which makes it MDLM. Add the iterative unmask decode loop. Still
  token-level. Now T-vs-quality is measurable, which is the number that decides
  whether to continue.
- **Stage 2 - move to codes.** Diffuse over abstractinator v1 RVQ indices instead
  of tokens. Sequence length drops, alphabet shrinks. This is where the
  efficiency argument either appears or does not.
- **Stage 3 - make the schedule structural.** Replace the uniform schedule with
  `rvq_stage`, then `harmonic_band`. Add the Koopman off-diagonal metric. This is
  the stage that is actually ours rather than a reimplementation, and it is the
  only one worth writing up.

Stage 0 is small enough to be worth doing on its own even if the rest is never
built: "does anything in Praxis secretly depend on the causal mask" is a useful
thing to know, and right now nobody knows the answer.

---

## What was built (2026-09-13)

Stage 1 of the staging below, plus the parts of stage 0 that turned out to be
free. `./launch --abstractinator-d` is runnable.

**The objective.** `praxis/diffusion/masked.py` holds `MaskedDiffusion`:
`corrupt()` samples `t ~ U(eps, 1)` per ROW and replaces positions with the mask
id, never touching padding and never leaving a row empty (a row with nothing
masked still divides by `t`, so it is pure variance). `compute_loss()` is
`sum(CE over masked) / (t * L_valid)` averaged over rows - the unbiased ELBO
estimator, not a convenience mean. Registered as a `diffusion` registry
namespace with two profiles (`masked`, `masked_coarse`).

**No timestep conditioning**, following MDLM and LLaDA: the mask fraction is
already visible in the input, so an AdaLN time embedding would learn to
reproduce something the input states outright. That removed the whole reason to
lift `TimestepEmbedder` out of `flow.py`.

**It is the criterion.** `build_objectives` registers the diffusion module as
`main` when `diffusion_type` is set, so `loss_func` is ignored and the blueprint
shows the term that actually runs. The corruption happens inside
`PraxisForCausalLM.forward`, not in the collator, for the prefetch-lead reason.

**Causality.** `config.causal` was already the flag and six attention modules
already read it; `PraxisForCausalLM.__init__` was the only thing forcing it
True. Two hardcoded sites in `attention/components.py` (`is_causal=True`, and a
shape-inferred one on the cached path) now read it. Kaleidoscope runs
non-causally without changes.

**Alignment.** `outputs_are_aligned` is now a property on the MODEL, True under
diffusion whatever the encoder says. `backpropagation.py` and the lazy-init
dummy forward both ask the model rather than the encoder.

**Decoding.** `praxis/diffusion/decoding.py`: `refine()` is the loop, and
`unmask_decoding` is the thin transformers adapter installed ahead of the
encoder's own method and speculative decode. `MaskedDiffusion.generate`
delegates to the same `refine`, because the two loops existed separately for
about an hour and drifted, which a test caught.

**Metrics.** Nine, declared on the class so the charts pick them up, collected
by the existing `criterion.terms()` walk with no new plumbing. The one to read
first is `diffusion_unigram_gap` - the model's CE on corrupted positions minus a
running unigram marginal's. The degenerate solution (learn nothing, predict the
marginal) produces a healthy falling loss and a gap pinned at zero.

**Tests.** `tests/diffusion/`, 31 of them. Both causality directions are
asserted in the same file: that `-d` is bidirectional AND that the ordinary arms
are still strictly causal.

### Three bugs the tests and the smoke run caught

Worth recording because two of them are properties of the factorisation rather
than slips, and anything built on this will meet them again.

1. **Length-based stopping criteria end the run on pass 1.** The block is
   allocated at full length before the first forward, so transformers'
   `MaxLengthCriteria` is already satisfied. Wiring the prepared list into the
   refinement loop returned a block of first guesses after a single pass. The
   block length IS the stop condition here; a content-based stop would have to
   TRIM the result, which is a separate feature and not implemented.
2. **Logits processors must run BEFORE the mask column is trimmed.** A
   processor indexes `scores` with ids taken from `input_ids`, and the block is
   full of mask ids until the last pass - so trimming first puts every gather
   out of bounds. It surfaced as a CUDA device-side assert during teardown with
   a traceback pointing at the optimizer.
3. **Every site that builds labels has to ask about alignment.** The lazy-init
   dummy forward hardcoded `input_ids[..., 1:]`. `compute_loss` now refuses a
   shifted label tensor with a message naming the cause instead of crashing on
   a reshape.
4. **transformers decides the RETURN SHAPE, not the decoding method.**
   `decode_backend.py` always sets `return_dict_in_generate=True` and reads
   `.sequences`, so returning a bare tensor failed every served request with
   `'Tensor' object has no attribute 'sequences'`. Caught in production, not by
   the tests - they called `model.generate()` but never with that flag, which is
   the only way the serving path calls it. The test now parameterises over it.

### What is deliberately not built

- Stage 2 and 3 - diffusion over RVQ codes, and the structural schedule. They
  are the parts that would be ours; they are also the parts that need stage 1 to
  have worked first.
- Any content-based early stop, so generation is a fixed-length block.
- Compatibility with MTP (refused with an error), speculative decode or the KV
  cache. All three assume a left-to-right factorisation.
