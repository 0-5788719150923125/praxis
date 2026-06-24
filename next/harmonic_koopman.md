# Harmonic Latent Space is Koopman (and that's the citation we were missing)

> Status: **active framing** (2026-06-23). Companion to
> [oscillatory_axes.md](oscillatory_axes.md), [harmony.md](harmony.md),
> and the philosophical framing in [quantum_echoes.md](quantum_echoes.md).
> This note is the bridge from our homebrew harmonic stack to an established,
> benchmark-proven body of theory we should be citing, not reinventing.

## The thesis

"Modeling stable and changing geometries over time, in harmonic space" is,
almost word for word, **Koopman operator theory**. The Koopman operator lifts
nonlinear dynamics into a space where evolution is _linear_; that operator's
eigenvalues are temporal harmonics (oscillatory modes). Our bias/variance
decomposition is the language-modeling instance of a split that the forecasting
literature already makes explicit and validates on benchmarks.

This is not a loose analogy. The dictionary is exact.

## The dictionary

| Praxis construct                                                                  | Koopman / oscillatory-SSM analog                                     |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| frozen Weyl phases + fixed frequency lattice (`HarmonicField`)                    | fixed Koopman eigenfunctions - the linear "world", the gear teeth    |
| learnable static amplitude grid (`b[t,d]` same for every input)                   | time-invariant component (the global / shared operator) = **bias**   |
| input-conditional amplitude delta `Δ_φ(context)` + per-depth recurrent modulation | time-variant component (the local / context operator) = **variance** |
| CALM continuous latent (K tokens → one vector)                                    | the continuous trajectory Koopman/SSMs actually operate on           |
| recurrent depth loop re-forming the field in one step                             | one application of a fixed linear lift, iterated                     |

The precise statement (sharper than "the field is linear"): **the basis and
coupling are frozen and linear; the dynamics are a learned gain-schedule over
that frozen basis.** That is a _time-varying_ Koopman / switched-linear system -
slightly richer than a single LTI operator, and it says exactly which part is
fixed (phases/lattice) and which part is learned dynamics (amplitude/gain).

## Why frozen phase is constitutive, not a bug

The deep-research instinct was "everyone else _learns_ the spectrum; you freeze
it; ablate frozen-vs-learned." That framing is wrong, and your correction matters:

- In Koopman, once the observable basis is fixed, dynamics are linear and only
  the _coordinates in that basis_ evolve. Freezing the phases is therefore the
  **purest** form of the Koopman claim, not a deviation from it.
- Classic Koopman/LinOSS _learn_ the eigenfunctions only because they don't know
  the right basis a priori. Our bet: a Weyl-equidistributed irrational-phase
  Fourier basis is _universal enough_ (dense on the torus, no rational
  resonances - this is what [weyl1916] buys us) to freeze, so we learn only
  coordinates. Frozen phase **is** the "watch turns in one step" claim.

So the ablation survives but its meaning changes. Not "should we freeze?" but:
**what does the fixed universal basis cost in expressivity, and does it deliver
the bias/variance decoupling the theory promises, at acceptable loss?** That is
a cost-of-elegance measurement, and the honest risk it tests is real: a fixed
basis may be expressively insufficient for language's non-stationary,
non-periodic structure, and amplitude-only DOF may not compensate.

## Language as a time series - licensed by CALM

The discreteness objection ("language is symbols, not a sampled signal")
dissolves at the latent level: CALM's continuous latent stream _is_ a continuous
multivariate time series, the exact object Koopman/LinOSS operate on. The Markov
bridge is one line: the Koopman operator is the function-space lift of the Markov
transition (Perron-Frobenius) operator; a Markov chain's eigendecomposition is a
set of decaying/oscillating modes. Caveat to keep honest: language periodicity is
statistical and multi-scale, not a clean fundamental frequency, so "model
language as music" is an inductive-bias bet - strongest for prosody/meter/
repetition, weakest for long-range semantics. CALM is what makes it well-typed
rather than metaphor.

## Established vs open (verified)

**Established (cite these):**

- Koopman/DMD: Lusch-Kutz-Brunton, _Nat. Commun._ 2018 [lusch2018koopman] -
  autoencoder lift to linear latent; auxiliary net parametrizes a _continuous_
  eigenvalue spectrum (frequency varies with state).
- Koopa (NeurIPS 2023) [liu2023koopa] + KNF (ICLR 2023) [wang2023koopman]:
  **explicitly split time-invariant from time-variant operators via a Fourier
  filter.** This is our bias/variance split, reinvented from dynamical systems.
- LinOSS (ICLR 2025 Oral) [rusch2025linoss] + D-LinOSS [boyer2025dlinoss]:
  latent state built from forced harmonic oscillators; LinOSS beats Mamba ~2x
  at length 50k. Stability proven via the recurrent matrices' spectrum.
- S4D-Lin [gu2022s4d]: state basis `e^{tA}B` is provably damped Fourier basis.

**Open (the contribution surface):** Every strong result above is _time-series
forecasting_, not language. The verified corpus contains **no** paper encoding
structured harmonic/geometric content into a continuous _language_ latent. That
is genuinely under-explored - an open frontier, not a crowded one. The one direct
LM data point (Jacobian residual-stream eigenmodes ~98% complex-conjugate
"spirals", monotonic over depth, arXiv:2605.14258) is a single 2026 preprint,
unreplicated - suggestive, not load-bearing.

**Skeptical flags (do not lean on these):**

- Platonic Representation Hypothesis (arXiv:2405.07987) **failed verification**
  (0-3) on its strong claims. Use our _own_ measured bias/variance energy ratio
  (`capacity_split`/`field_strands`) for the "approximately-universal geometry"
  claim; cite PRH only as the contested strong version.
- "Transformers as continuous ODEs" contractivity branch (e.g. arXiv:2502.05656,
  withdrawn) is non-spectral and empirically thin. Not evidence for the harmonic
  view; the real spectral evidence is the Jacobian-eigendecomposition work.
- "Tokens as geometries on a manifold" (equivariant/spherical-harmonic machinery)
  is mature math with no known symmetry group for *abstract* tokens - but the
  observer-grounding rescues it to "grounded motivation": the perceptual channels
  (visual E(2), acoustic frequency-scaling) carry real groups, and shared lineage
  explains the convergence. See [observer_grounding.md](observer_grounding.md).
  Caveat unchanged: this licenses harmonic *priors* and *going multimodal*, NOT
  equivariant latent machinery on bytes; distributional-only LLMs already recover
  most structure without it.

## The clean test axis (passes the oscillatory_axes.md 4-criteria filter)

The directly-testable bridge from the established time-series result to our stack:
**make the CALM latent autoregression explicitly Koopman-structured** - the
next-latent operator as a frozen global linear part (stable spectrum) plus a
context-conditional low-rank delta (changing), with the spectrum read out as a
diagnostic.

- index `i` = patch / latent-autoregression step
- signal = the latent vector trajectory
- task-relevant = yes (it _is_ the generative target)
- equal-budget comparison = `HarmonicLatentHead` (flow in fixed harmonic coeff
  space `z = c @ Qᵀ`, already in `praxis/heads/harmonic_latent.py`) vs the plain
  `flow` head calm-d currently uses. Read the bias/variance energy split and
  whether the latent spectrum is interpretable.

This would be, as far as the verified literature shows, the first
continuous-latent _language_ model to impose Koopman structure on the latent.

## Prior-art anchors

Koopman/DMD (Lusch-Kutz-Brunton; Koopa; KNF); oscillatory SSMs (LinOSS,
D-LinOSS; Rusch & Rus); S4D (Gu et al.); coRNN/UnICORNN (Rusch & Mishra, see
[oscillatory_axes.md]); Weyl equidistribution [weyl1916]; Takens embedding
[takens1981] (both already in citations.bib). FAR (arXiv:2503.05305) -
frequency-axis autoregression for images, the only existing "autoregress along
the spectral axis" proof-of-concept; no language analog exists.
