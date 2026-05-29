# Oscillatory Substrate: One Trick, Many Axes

> Status: **pinned** (2026-05-29). Parked while we push on RL objectives.
> This note captures a recurring intuition so we can pick it up deliberately
> rather than re-deriving it from vibes each time. Companion to the
> philosophical framing in [quantum_echoes.md](quantum_echoes.md) and the
> head engineering log in [harmony.md](harmony.md).

## The thesis

Oscillation is a substrate for **long-range memory and stable propagation**,
and you can write information into it compressibly. This is not a metaphor:
coupled-oscillator RNNs (coRNN, UnICORNN) and oscillatory state-space models
(LinOSS) use literal `sin`/oscillator dynamics as the recurrence and beat
generic recurrences on long-range tasks; SIREN and Fourier Neural Operators
parametrize functions/kernels in a sinusoidal/spectral basis; Holographic
Reduced Representations store many items superposed in one vector and read
them back by phase. The common claim: a band-limited / oscillatory basis buys
longer effective range and fewer effective degrees of freedom.

The "universe computed by a wave function" framing (Wheeler-DeWitt, the
Everettian universal wavefunction, Wheeler/Lloyd/Wolfram digital physics) is
a resonant analogy, **not** a mechanism. It does not hand you an architecture.
The load-bearing import is narrower and survives the trip down from cosmology:
*wave dynamics are a good computational substrate for memory* - which biology
also bets on (neural oscillations, theta-gamma phase coding).

## One trick, four axes

We have already been applying the same idea on different axes without naming
it. That is the actual research program:

| Axis | Where | Status |
|------|-------|--------|
| position × feature | `HarmonicField` (head bias `h*(1+b)`) | live (`heads/harmonic.py`) |
| token mixing | WaveletLM block (lifting wavelet + Walsh-Hadamard) | PoC (`block_type: wavelet`) |
| weight row | harmonic weight-RL edits `w[i]*(1+α·sin(ω·i+φ))` | live (`policies/harmonic_weight_rl.py`) |
| depth / time | recurrent depth loop | live, **not yet oscillatory** |

The organizing question is **not** "is the universe a wave function." It is:
**on which axis does an oscillatory/spectral basis actually pay off, and for
what** - compression at equal parameter count, longer effective memory, or
more stable recurrence?

## What counts as a clean test axis

A reusable filter (apply this before building, so the idea predicts a number):

1. an ordered index `i` (position, depth, parameter index, time);
2. a signal that varies along `i`;
3. that signal carries **task-relevant** information;
4. so that oscillatory-basis vs flat-baseline, **at equal parameter budget**,
   produces a measurable difference (sample efficiency, long-range retention,
   compressibility).

## The float-precision detour (parked, do not re-walk cold)

Idea explored 2026-05-29: hash all representable values of a float, treat the
precision/rounding error as a "voting system," oscillate over it as a wave.

Real kernel: float32 precision error genuinely has structure - the ulp is
constant within a binade and doubles each power of two (absolute error is a
log-spaced staircase; relative error ~constant 2^-24), and rounding is
many-to-one so each float has a preimage of reals (the "votes").

Why it was parked, concretely:
- **A single hash scalar has no axis.** Reducing "all values" to one number
  destroys the staircase - the very structure that was interesting. A scalar
  is a point, not a wave (fails #1, #2).
- **Float precision is a property of the number format, not the model.** It is
  fixed by IEEE 754, identical across weights and runs, carries no task
  information, and the model can neither write to nor read from it (fails #3,
  #4). It is orthogonal to all content.
- Same wall as the earlier "weight-hashes over time" idea: instrumentation at
  best (you can *plot* a hash signature evolving), not a mechanism.

Salvageable cousin if the live wire is "precision error as a carried signal":
**error feedback** (quantize, measure the residual, add it back next step so
errors don't accumulate - the residual is a running memory) and **stochastic
rounding**. These are real and have results, but they are numerical-methods
techniques, not an oscillation axis.

## First test to run when we return

The recurrence axis passes all four criteria with the least new machinery,
because recurrent depth already exists.

- index `i` = recurrent-depth step
- signal = the hidden state evolving across iterations
- test = an oscillatory recurrence update grafted onto the existing loop,
  against the current update, on a long-range-retention metric.

Sketch (damped coupled oscillator, coRNN-style; `z` is a velocity state):
```
# per recurrent step i, with learnable/​fixed ω (frequency) and γ (damping):
z = z + dt * (tanh(W h + U x + b) - γ * z - (ω**2) * h)
h = h + dt * z
```
Compare against the current recurrent-depth update at equal parameters; read
a synthetic long-range retention task (e.g. copy/recall at increasing lag) and
whether deeper recurrence stays stable (this is where SandwichNorm mattered;
oscillatory updates may reduce that dependence - worth measuring). Multiple ω
= multiple memory horizons ("different scaling axis").

## Prior art anchors

coRNN / UnICORNN (Rusch & Mishra); LinOSS / oscillatory SSMs (Rusch & Rus);
SIREN (Sitzmann et al.); Fourier Neural Operator (Li et al.); Holographic
Reduced Representations (Plate); fast weights (Schmidhuber; Ba et al.);
error-feedback SGD; stochastic rounding.
