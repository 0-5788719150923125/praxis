# Heisenvectors: direction or length, never both

> Status: **framing note only** (2026-10-05). Nothing built, no run. A joke line
> from a book chapter, worked through until it either meant something or didn't.
> It does, in three ways, and only two of them deserve the name. Siblings:
> [harmonic_koopman.md](harmonic_koopman.md) (the harmonic latent, where every
> mode is already a phasor), [observer_effect.md](observer_effect.md) (the other
> place Heisenberg shows up here), [information_geometry.md](information_geometry.md)
> (clipped descent as direction-only descent) and [motor_channel.md](motor_channel.md)
> (a pointer whose angle and radius mean different things).

## Where this came from

Chapter 42 of *North Star* (`rift/books/north-star/chapters/42-what-is-the-7th-realm.md`)
has a professor who teaches Heisenvector analysis:

> the study of quantities whose direction can be known exactly, or whose length
> can, but never both at once. A heisenvector pointed straight at you has no
> length at all, which is why you can never tell how far away somebody is when
> they are looking right at you.

The question that followed:

> What would it mean if direction or a length could be known independently, but
> not at the same time? ... Would it be like a mode-switch, maybe, where you
> could switch off the influence for direction or length for each pass through
> some vector?

## The name

**Provenance.** The word comes from Emma O'Neil's joke faculty page
([web.pdx.edu/~emmao/toquos.html](https://web.pdx.edu/~emmao/toquos.html)), which
lists "MTHS1202-001 (Introduction to Heisenvector Analysis III)" among her
courses at the fictional Mandlbaur Institute of Technology, alongside Toquos
Theory and a restaurant menu. The page gives no definition. Both pages it links
to (`research.html`, `numberdevil.html`) were 404 on 2026-10-05, and the Wayback
Machine was rate-limiting, so archived copies are unchecked. A web search found
no other use of the word, technical or otherwise. The definition above is the
chapter's own riff on the course title.

**Etymology.** *Vector* is the physics-class definition: a quantity with a
magnitude and a direction. Every hidden state, gradient and latent in Praxis is
one - its norm is the length, its unit vector the direction. *Heisen* is
Heisenberg's uncertainty principle (1927): some pairs of quantities, like
position and momentum, can't both be exact. It's the same pun as *heisenbug*, a
bug that changes when you look at it. Together: a vector whose two defining
properties obey an uncertainty principle. The joke is that the two things that
make it a vector are the two it can't have at once.

**Naming rule for this thread.** The note keeps the name, because it names the
question. Anything built from it is named for what it does (`polar_coupling`,
`gain_shape`), because "Heisen" only describes the pieces that enforce a
tradeoff:

| Piece | Fits "Heisen"? | Why |
|---|---|---|
| Phasor under number-phase uncertainty | yes | it is the object the chapter describes |
| Gain-shape bit budget | yes | bits spent on length can't go to direction |
| Restricted reads (the mode switch itself) | yes | the tradeoff, imposed per pass |
| Cross-coupling (read one half, write the other) | no | works identically with zero uncertainty; that's Hamilton |

## The one line

A vector has an uncertainty principle between its length and its direction only
when the two are **conjugate** - two readings of the same wave, the way time and
frequency are. Ordinary vectors never are. A finite budget or a restricted read
can impose it. And the dynamics that make two quantities conjugate, each one
driving the other, need no uncertainty at all.

## When it's real

**Phasors (2D).** Any oscillation is a 2D vector: amplitude is its length, phase
its direction. For a quantized oscillator (a mode of light), photon number `N`
(length squared, counted in quanta) and phase `phi` trade off,
`dN * dphi >~ 1/2`, made rigorous by the Pegg-Barnett phase formalism. A number
state has an exact length and a uniformly random direction; a phase state is the
reverse. The reason is Fourier: a phase lives on a circle, and the Fourier dual
of a circle is the integers. Once length comes in whole quanta, "which way" and
"how many quanta" are the same wave read in two bases, and sharpening one smears
the other. Classically the same pair are the action-angle coordinates of a
harmonic oscillator - action `J = r^2 / 2` (in scaled coordinates) and angle
`phi` are canonically conjugate. Quantization is what turns that conjugacy into
an uncertainty.

**Angular momentum (3D).** The chapter's punchline is a theorem. The components
don't commute: `[L_x, L_y] = i hbar L_z`, and cyclically. Suppose a state had
definite `L_x = a` and `L_y = b`. Then
`i hbar L_z psi = (L_x L_y - L_y L_x) psi = (ab - ba) psi = 0`, so `L_z psi = 0`.
The other two commutators then give `i hbar a psi = [L_y, L_z] psi = 0` and
`i hbar b psi = [L_z, L_x] psi = 0`, so `a = b = 0` and `L^2 psi = 0`. The only
state whose direction is fully determined is `l = 0`: a heisenvector pointed
straight at you really has no length. For `l > 0` the largest projection is
`l hbar`, short of the length `sqrt(l (l + 1)) hbar`, so a nonzero angular
momentum never points exactly along any axis.

**Why ordinary vectors escape.** Scaling commutes with rotation, so applying or
reading one never disturbs the other. And copying is free: fan `x` out, read
`|x|` from one copy and `x / |x|` from the other. No-cloning is what stops this
in physics, and a network has nothing like it. The only classical leftover is
one-way: a short vector's direction is noisy. Observing
`x = r u(theta) + noise(sigma)` carries Fisher information `r^2 / sigma^2` about
`theta`, which vanishes with the length. That's the polar singularity at the
origin, not a tradeoff.

## Praxis already picked a side

Praxis reads direction almost everywhere and leaves length unread.

- **The residual stream.** Attention and the FFN read the stream through the pre
  norm (`praxis/blocks/transformer.py:100`), so its length never enters either
  sublayer, under every norm type. What they write depends on the norm. Under
  the sandwich family (`sandwich`, `sandwich_tied`, `hero`) the write is
  re-normalized before it lands (`:107`), so a sublayer picks only a direction.
  Under the default `rms_norm` the post call is a no-op, and the write keeps
  whatever length the sublayer gave it: length is written, never read. Either
  way the stream's own length acts only as inertia. An orthogonal write of size
  `w` turns a stream of length `r` by about `w / r` radians, so the longer the
  stream grows, the harder it is to turn. Nothing measures this today; there is
  no residual-stream norm metric.
- **The optimizers.** LionGeo's sign, spectral and Frobenius arms
  (`praxis/optimization/lion_geo.py:113`) are three definitions of the
  gradient's direction, all RMS-matched to ~1, and the schedule supplies the
  length. Muon's Newton-Schulz iteration approximates the orthogonal factor `Q`
  of the polar decomposition `G = QP`, the matrix version of `r e^{i theta}`
  ([Polar Express, arXiv:2505.16932](https://arxiv.org/abs/2505.16932)).
- **The harmonic bottleneck.** `HarmonicResidualVQ` says it in its own comment,
  "quantize direction on the sphere"
  (`praxis/encoders/quantization/harmonic_bottleneck.py:113`). The length is
  deleted. `bottleneck: harmonic_gdn` (`praxis/encoders/abstractinator/encoder.py:145`)
  starts as exactly the same deletion, and GDN can only let length back through
  by growing `beta` - at `beta = 0` it is scale-invariant whatever `gamma` is.

One measured hint that trained networks treat the two halves as rivals: DoRA's
weight-decomposition analysis ([arXiv:2402.09353](https://arxiv.org/abs/2402.09353))
found that full fine-tuning's changes in weight magnitude and weight direction
are negatively correlated (-0.62), while LoRA's are positively correlated
(+0.83). Not an uncertainty principle, but a measured tendency for unconstrained
training to move a weight's length or its direction rather than both.

## The budget version: gain-shape quantization

A classical vector can't have the tradeoff for free, but a coded one can.
Gain-shape (or shape-gain) VQ (Sabin & Gray, *Product Code Vector Quantizers for
Waveform and Voice Coding*, IEEE Trans. ASSP, 1984) codes the length and the
direction with separate codebooks. With `K_g` gain levels and `K_s` shape codes
a vector costs `log2 K_g + log2 K_s` bits, so at a fixed budget every bit spent
on length is a bit not spent on direction. Bits per vector play the role of
`hbar`. Opus's CELT layer works this way: it codes each band's energy explicitly
and the normalized band shape with pyramid VQ, a spherical quantizer.

The probe, if it's ever worth running: a `harmonic_gain_shape` bottleneck that
quantizes `log RMS` with a small scalar codebook beside the shape codes and
reattaches it after synthesis, against `harmonic` at matched total bits (the
gain code's bits come out of the shape codes). Falsifier: if the gain code's
perplexity sits near 1, or the loss doesn't move, length carries nothing at the
patch level and the deletion was right.

## The mode switch, and the fix

The switch as first imagined: each pass through a vector is influenced by only
its direction or only its length. Hard version: a pass sees one half and not the
other. Soft version: a pass sees both, with noise traded between them at a fixed
product, `sigma_dir * sigma_len = const` - what a squeeze parameter does in
optics. This restricted read is the honestly Heisen part. If built, the squeeze
would have to be learned per token, never set per run.

On its own it does nothing useful. If a direction pass reads and writes direction
and a length pass reads and writes length, the two passes commute and never
interact, and the length half never learns anything about content. The fix is
for each pass to **write the half it can't see**:

```
state:    x = r u,   r = |x|,   u = x / |x|

pass A    read direction, write length:     log r  <-  log r + F(u)
pass B    read length, write direction:     u      <-  R(log r) u
          R(s) = exp(sum_k g_k(s) A_k),  A_k skew-symmetric (so R is a rotation)
```

Pass A: what it is decides how much of it there is. Pass B: how much of it there
is decides how far it turns.

- **It's a coupling layer.** This is RevNet's additive coupling
  (`y1 = x1 + F(x2)`, `y2 = x2 + G(y1)`, with NICE before it), split as
  length | direction instead of two halves of the channels. Each pass inverts
  exactly (`log r <- log r - F(u)`, `u <- R(log r)^T u`), so a recurrent loop
  built from it can never merge two states, and it preserves volume in
  `(log r, u)` coordinates.
- **It's a leapfrog step.** For a separable Hamiltonian, each of a conjugate
  pair moves at a rate set only by the other, and a leapfrog integrator
  alternates exactly these two half-steps. This is the Hamilton part, and it's
  why the cross-coupling doesn't earn the name: it works identically with no
  uncertainty anywhere.
- **Optics has both passes as devices.** The Kerr effect is pass B (intensity
  rotates phase), and phase-sensitive amplification is pass A (phase sets gain).
  They're the two classic ways to squeeze light, i.e. to move uncertainty
  between length and direction (Kitagawa & Yamamoto, Phys. Rev. A 34, 3974,
  1986, for the Kerr case).
- **Invertible is not bounded.** Pass A is a shear, so nothing stops `log r`
  drifting if `F` keeps one sign. Invertibility rules out collapse, not drift.
  With `|F| <= c` the drift over a loop of `D` steps is at most `D * c`, so `F`
  needs a bound.

Where it would live:

- **Recurrent depth loops**, where a state passes through the same block many
  times and both stability and invertibility matter.
- **The harmonic latent** ([harmonic_koopman.md](harmonic_koopman.md)), where
  every sine/cosine pair at one frequency is already a phasor and amplitude and
  phase are the textbook action-angle pair. In the Koopman reading, frozen phase
  means every mode turns at a fixed rate whatever its amplitude: pass B with its
  length dependence switched off, which is what keeps the picture linear. Letting
  the rate depend on amplitude would be the Kerr term - a deliberate step out of
  the linear world, not a free addition.

## What already has a name

| Piece | Established name | Reference |
|---|---|---|
| Length and direction trade off for a phasor | number-phase uncertainty | Heisenberg 1927; Pegg-Barnett phase formalism |
| Fully known direction forces zero length | angular momentum commutators | any QM text |
| Bits split between length and direction | gain-shape (shape-gain) VQ | Sabin & Gray 1984; Opus CELT |
| A matrix's direction without its length | orthogonal polar factor (Muon) | [arXiv:2505.16932](https://arxiv.org/abs/2505.16932) |
| Read one half, write the other | additive coupling | NICE [arXiv:1410.8516](https://arxiv.org/abs/1410.8516); RevNet [arXiv:1707.04585](https://arxiv.org/abs/1707.04585) |
| Change length, keep direction | radial flows | Gerdes & Cheng 2026, [arXiv:2601.10774](https://arxiv.org/abs/2601.10774) |
| Turn at a rate set by length | action-angle coordinates | Action-Angle Networks, [arXiv:2211.15338](https://arxiv.org/abs/2211.15338) |
| Alternate conjugate half-steps | leapfrog / symplectic Euler | any numerical ODE text |
| Move uncertainty between length and direction | amplitude/phase squeezing | Kitagawa & Yamamoto 1986 |
| Length and direction given separate jobs | weight norm; capsules (length = existence probability); DoRA | [arXiv:1602.07868](https://arxiv.org/abs/1602.07868); [arXiv:1710.09829](https://arxiv.org/abs/1710.09829); [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) |

**What looks new** is the framing, not a mechanism: the inventory above (Praxis
reads direction almost everywhere and never reads length), and treating a
recurrent state's length and direction as a pair that drive each other inside a
language model. That rests on one light search on 2026-10-05, not a literature
review.

## Cheapest probes, in order

1. **Measure the inertia.** Per depth, log the residual stream's norm and the
   angle each sublayer turns it by. Prediction: the angle tracks
   `write size / stream length` and shrinks with depth. Diagnostic only - it
   says whether the stream's unread length is shaping the dynamics. It belongs
   on the residual connection (`praxis/residuals/`), whose `connect_depth` holds
   both the stream and the write.
2. **Gain-shape bottleneck** (above), at matched bits.
3. **Polar coupling in a recurrent loop**, only if 1 or 2 shows that length
   matters.

## Open questions

- Does a patch latent's length carry anything the model wants? The gain code's
  perplexity answers it.
- Is the residual stream's inertia a feature (late layers can't wreck the
  stream) or a cost (late layers can barely turn it)? Under `rms_norm` a
  sublayer can pay for a bigger turn with a longer write; under the sandwich
  family it can't.
- What would decide a pass's squeeze in the soft switch?
