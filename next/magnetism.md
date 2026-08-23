# Magnetism: The Right Instinct, Pointed at a Broken Instrument

> Status: **idea assessed 2026-08-22, one correctness fix SHIPPED, rest parked.**
> Companion to [structural_variance.md](structural_variance.md),
> [harmony.md](harmony.md), and [harmonic_koopman.md](harmonic_koopman.md).
> The shipped part is in `praxis/losses/contrastive_isotropy.py`.

## The one line

The magnetism analogy is correct, mechanically empty, and still worth having -
because following it honestly forces you to ask for a **second moment**, and the
moment you compute one you discover that `repr_anisotropy` has been reporting
the **opposite of the truth** on both of the cases that matter.

## The physics, corrected

Aligned domains are **anisotropy**, not isotropy. A magnet with every domain
facing one way has a preferred direction; a demagnetized paramagnet has none and
is isotropic. So `contrastive_isotropy` is literally a **demagnetizer**, and an
"anisotropy loss" would *increase* magnetization rather than damp it.

The correspondence is exact, not poetic. For L2-normalized token reps `u_i`,
with `m = mean(u_i)` the magnetization vector:

```
mean off-diagonal cosine  =  (T * ||m||^2 - 1) / (T - 1)
```

So `repr_anisotropy` **is** the squared magnetization (equivalently the squared
Kuramoto order parameter) of the token directions, up to an affine map with
positive slope. Pinned as a test: `test_repr_anisotropy_equals_squared_magnetization`.

Two consequences, and the second is the one that matters:

1. Mechanically this changes nothing. Same number, positive-slope relabel, no
   gradient moves. **Do not build on the rename.** It is a footnote.
2. The magnet picture is entirely about **domains** - locally aligned regions
   with different orientations. `||m||` is a **rank-1** statistic and is
   structurally blind to exactly that. Domains need a second moment. That demand
   is where the analogy earns its keep, and it is also where it dissolves: the
   sign-blind object that sees domains is the nematic tensor
   `Q = mean(u u^T) - I/D`, which is the centered covariance spectrum - i.e.
   IsoScore, effective rank, participation ratio. The physics points at the right
   instrument and then has nothing further to say.

## What the second moment revealed

Measured at the lineage's own dimensions (T = 512, D = 111):

| point cloud | `repr_anisotropy` | `repr_dimensions` | `repr_nematic` |
| --- | --- | --- | --- |
| isotropic | 0.000 | 1.00 | 0.00 |
| **only TRANSLATED off the origin** | **0.975** | **1.00** | 0.00 |
| **genuine rank-3 collapse of 111** | **0.000** | **0.03** | 0.57 |
| two antipodal domains | -0.002 | 0.01 | 0.99 |
| 8 tight domains | - | 0.07 | 0.38 |

The metric is not merely imprecise. It is **inverted on both canonical cases**.
It screams collapse at a cloud that is only shifted and reports a healthy
isotropic space for one genuinely crushed into three dimensions. The translated
row reads 0.975 against the live lineage's measured 0.977-0.98: *the observed
"collapse" is exactly what a pure mean offset looks like.*

And the third row is the point of the whole exercise. Two locally-aligned
regions facing opposite ways - the literal magnet picture - read `-0.002`,
because their mean directions cancel. The instrument could not see the structure
the idea is about.

### Live evidence this is not hypothetical

`repr_anisotropy` across four archived runs traces the same reproducible curve:
~0.07 at step 50, **0.98 by step 1000**, then a recovery to 0.29-0.45 by step
8-13k. In `e6941ca4d`, **88 of 120 logged steps have the hinge fully saturated**
(`repr_anisotropy - contrastive_loss == 0.5` exactly, meaning every off-diagonal
pair clears the margin). A hinge that is never a hinge is a plain linear penalty
at constant maximum gradient - the default regularizer is at **full authority**
during exactly the phase it exists to prevent, and the geometry moves anyway.

Whether that is a real pathology or a mean offset is now a question the run
itself answers, because the instrument to tell them apart is shipped.

## Shipped (correctness, `praxis/losses/contrastive_isotropy.py`)

Two fixes. Neither changes the loss's math - SimCTG's margin is defined on
uncentered reps and stays that way.

**1. Batch-size scaling bug.** `valid` was built at `[1, T, T]` while the
numerator summed over `[B, T, T]`, so `denom` counted one batch's pairs against
B batches' similarities. Both the loss and the metric came out **B times too
large** - but *only* on the path where the padding mask fails to broadcast it
(`input_ids.size(1) != T`, the encoder/patch case). So the same run's scale would
jump by a factor of B between the two paths. Latent on the abstractinator
lineage today (the saturation signature shows the mask branch firing), live for
any config where reps and ids differ in length. Fixed by expanding `valid` to the
batch; regression test pins loss and metric invariant to replicating one sequence
across B, on both paths.

**2. The metric could not see collapse.** Added three readings, all
trace-form (`O(T * D^2)`, **no eigendecomposition**, shape-stable under the
sequence curriculum, measured at **~1 ms/call on GPU** against a ~1.3 s step):

- `repr_dimensions` - participation ratio `(tr C)^2 / ||C||_F^2` of the centered
  covariance, over its isotropic null `D / (1 + D/T)`. **1.0 = spread.** The
  collapse detector the mean cosine cannot be.
- `repr_nematic` - axis alignment of centered directions,
  `sqrt((D/(D-1)) * (||M||_F^2 - 1/D) - 1/T)`. The `1/T` is **exact, not
  fitted**: under isotropy `E[||M||_F^2] = 1/T + (T-1)/(TD)`, so the whole
  expression has expectation zero at any T and D. Sign-blind, so it sees the
  antipodal-domain case.
- `contrastive_active_frac` - share of pairs above the margin. Makes the hinge
  saturation above visible instead of inferable.

Both geometric readings are normalized against their own finite-sample null, so
they read 1.0 and 0.0 for an isotropic cloud at **every** (T, D) tested - which
matters here specifically, because the sequence-length curriculum varies T and
would otherwise move the charts on its own. `repr_anisotropy` is kept as-is: it
is the quantity the loss is actually written against, so it stays as the loss's
own gauge, with a description that now says what it does and does not measure.

## What the literature already owns

Blunt, because the whole point of asking was to avoid re-deriving.

| component | status | who owns it |
| --- | --- | --- |
| signed isotropy loss, sign selects magnetize/demagnetize | **done, 2024** | **I-STAR**, Rudman & Eickhoff, ICLR 2024, `arXiv:2305.19358` |
| mean cosine is not an isotropy measure | **done, 2022** | IsoScore, Findings of ACL 2022, `arXiv:2108.07344`; rogue dimensions, Timkey & van Schijndel, EMNLP 2021, `arXiv:2109.04404` |
| locally isotropic clusters in a globally anisotropic space (= domains) | **done, 2021** | Cai et al., ICLR 2021; per-cluster nulling, Rajaee & Pilehvar, ACL 2021 |
| per-layer anisotropy profile of a decoder | **measured, 2024** | Razzhigaev et al., Findings of EACL 2024, `arXiv:2311.05928` - a **bell curve peaking mid-depth** |
| attention as a spin system with an exact Hamiltonian | **done** | Huo & Johnson `arXiv:2504.04600`; Bhattacharjee & Lee, Phys. Rev. E, extracted it from all 144 GPT-2 heads |
| magnetism as recurrent mixing (Kuramoto/XY) | **done** | AKOrN, ICLR 2025 Oral, `arXiv:2410.13821` - energy is literally the Heisenberg Hamiltonian with an external field |
| Kuramoto inside self-attention, byte-level LM | **done, 2026** | `arXiv:2606.11585`, `arXiv:2606.12059` |
| text rendered as pixels alongside its tokens | **done** | DualGPT, EMNLP 2024, `arXiv:2404.10710`; PIXEL ICLR 2023; CLIPPO CVPR 2023 |
| align + decorrelate two views | **done** | Wang & Isola ICML 2020 (alignment/uniformity), VICReg, Barlow Twins |
| magnetism *vocabulary* for embedding geometry | **not found** | nobody - and it buys nothing |

**I-STAR is the decisive one.** It is a signed isotropy regularizer on a
differentiable minibatch-stable measure, and **every optimal configuration it
reports uses the magnetizing sign** - decreasing isotropy improved downstream
performance across three LLMs and nine tasks. So "add an anisotropy loss" is a
published 2024 result, and Praxis ships the *other* sign by default. (Not
settled: Kudriashov et al. `arXiv:2501.05502` raised isotropy and also improved.
The field genuinely disagrees; do not write it up as decided.)

Two headwinds from I-STAR aimed straight at the per-depth plan: they **tried**
per-layer application, found it **less stable** than a global penalty, and named
it future work; and they report the penalty mostly reaches **early** layers while
later ones stay anisotropic regardless.

## What is actually open

Narrower than the pitch, and none of it is called magnetism.

1. **A per-depth signed isotropy term applied per recurrent PASS, with a learned
   coefficient.** I-STAR's own stated future work, unoccupied for recurrent
   depth specifically. The learned per-depth profile is the result. Satisfies
   the no-tuning rule only if the coefficient is learned (bounded log space plus
   per-depth delta, the learnable-RoPE-theta pattern) - a hand-set profile is a
   tuned schedule in a lab coat and is disqualified.
2. **Two-level control: isotropy WITHIN domains, anisotropy BETWEEN domain
   means.** Every piece is cited above; the combination is not. This is the
   version of the magnet picture that survives, because it is the version that
   needs a second moment.
3. **Phase synchronization as the CROSS-MODAL alignment operator.** Searched
   five ways, found nothing in ML. Neuroscience has cross-modal phase reset;
   ML has ordinary fusion with no phase in it. This is the one genuinely empty
   spot the original idea pointed at.
4. **Learned, input-conditional harmonic coupling.** Higher harmonics of a
   Kuramoto coupling buy memory capacity with *fixed* coefficients
   (`arXiv:2507.21984`); nobody has made them learned. Praxis's amplitude field
   is unusually close to this already.

## What does not survive, and why

- **The magnetism vocabulary.** Rank-1, domain-blind, and it duplicates two
  existing names for the same scalar. Violates the no-invented-lineage rule.
- **A fixed per-depth target profile.** The archived runs swing 0.07 -> 0.98 ->
  0.29 in **training time**; against that, per-depth structure is second-order.
  The "right" value is 0.08 at step 100 and 0.98 at step 1000. There is no
  constant to target.
- **Order-parameter halting.** `praxis/halting/kl.py` already halts on
  convergence, per-position over the full distribution, with an endogenous EMA
  anchor and a written contraction-mapping argument. `||m||` is a rank-1,
  batch-pooled, strictly weaker estimator of the same thing.
- **"Align the harmonics through recurrent computation" - as currently
  conceived, this names an operation on a variable that does not exist.**
  `spec_real`/`spec_imag` are frozen buffers and `self.amplitudes` is a **real**
  parameter, so per-cell phase can only flip by pi, discretely. There is no
  continuous phase DOF anywhere in the harmonic machinery. Separately,
  `grep current_depth praxis/heads/` returns **zero**: the field is never inside
  the loop. Both would have to be built, and the frozen basis is a claim
  `research/body.tex` currently rests on.
- **The dual text+pixel stream, on this config.** `H(pixels | text) = 0` at
  fixed font - a raster is a deterministic, lossy re-encoding, not an
  augmentation, so there is no nuisance variable to be invariant to. Multi-view
  theory buys you the label-relevant subspace *without labels*; Praxis has a
  supervised signal on every byte, so the precondition fails. And the one thing
  pixels genuinely deliver - sub-token orthographic structure with no vocabulary
  bottleneck - is **already bought** by `byte_level` + `byte_multihash`.

  The cost objection, though, is **withdrawn**: a glyph atlas is a 21 KiB frozen
  `[256, H*W]` table and the lookup is `F.embedding(input_ids, atlas)` -
  measured **225x faster** than PIL rendering, and Pillow is already resolved
  transitively via matplotlib. For a *byte-level* model, "render the text as
  pixels" is mechanically just **a frozen embedding table whose rows are glyph
  bitmaps**. That both deflates it (it is not vision) and makes it nearly free
  to test as an embedding prior. Confusability structure is real in that basis:
  `'O'`/`'0'` cosine 0.973, `'E'`/`'F'` 0.890, `'E'`/`'w'` 0.467.

## Sequencing

Rungs 1-2 are shipped or free. Nothing below rung 3 should be built before the
one above it reports.

| # | rung | cost | kills the direction if |
| --- | --- | --- | --- |
| 0 | **the instrument** (shipped) | done | - |
| 1 | watch `repr_dimensions` on the next run | free | it sits at ~1.0 while `repr_anisotropy` sweeps to 0.98: **the collapse never happened**, and both the incumbent demagnetizer and every proposed magnetizer are acting on a mean offset |
| 2 | plot archived `repr_anisotropy` vs step and tokens across -q/-r/-s/-t | free, data on disk | (it already shows the temporal swing dwarfs anything per-depth) |
| 3 | per-depth order parameter as a **measurement** | ~2 lines: `sequential.py:162` already collects `hidden_states.detach().mean(dim=(0,1))`, which is the *unnormalized* magnetization vector. Normalize before the mean and it is bounded. Add `cos(m_d, m_{d+1})` | the profile is flat across passes: nothing to shape, per-depth half dies at zero GPU cost |
| 4 | spectral isotropy term at the incumbent's weight, sign unchanged | 1 day + 1 run | neither measure moves vs a no-regularizer control: an additive term has no authority at this scale, and rungs 5-7 are dead |
| 5 | **flip the sign** (the I-STAR question) | one constant + 1 run | BPB degrades: the incumbent sign was right. **Abort criterion**: `repr_dimensions` declining monotonically is collapse, not shaping |
| 6 | per-pass application | 2-3 days + run | only if rung 3 showed a non-flat profile |
| 7 | learned per-depth coefficient (`OuroborosBudget` dual pattern, `[depth, 1]` shape - never 0-dim) | ~4 days + run | flat learned profile = the model does not want depth-varying geometry; report and close |
| 8 | glyph-atlas embedding arm (independent of 1-7) | 2-3 days + run | early-training BPB unchanged: the prior buys nothing, and the two-stream version must **not** be built |
| 9 | learnable harmonic phase delta (zero-init, identity at init) | a week | all three outcomes reportable - if it stays at zero, the frozen-phase claim is **confirmed**, which is a positive result for the paper |

Rung 3 before rung 4 is the discipline the lineage already runs on: measure the
profile, then point a loss at it. Rung 9 is a separate question wearing this
one's clothes and should be run as its own single-variable experiment, never
smuggled in under a magnetism banner.

## The honest summary

The instinct - *locally aligned, differently oriented between regions, is a
healthier target than globally uniform* - is **correct**, independently
supported, and not what the default regularizer optimizes. The vocabulary for it
is not magnetism. The mechanism for it is a second moment. And the reason to
have chased it is that it found a metric reporting the opposite of the truth on
a chart that has been up for the whole lineage.
