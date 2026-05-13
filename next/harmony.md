# Harmonic Head: Stabilization Notes

Working notes on the 2D irrational-rotation harmonic head (`praxis/heads/harmonic.py`). Each section captures a hypothesis, the change made (if any), and what to look for next. Added in chronological order; older sections stay as-is for reference.

---

## 2026-05-07 — Diagnosis after 2.6B tokens (`december.yml`)

**Symptoms observed in the live run:**

- "Harmonic Gradient Ratio" growing in polynomial time.
- "Harmonic Field Amplitudes" still trending logarithmically downward, no plateau.
- "Update-to-Weight Ratio" and "Gradient Flow" missing data points roughly half the time (likely NaN/inf filtered out at the metric extraction step).
- Inference outputs starting to degenerate; train loss still declining.
- "Harmonic Spectrum" heatmap still visually indistinguishable from random noise.

**Reading of the symptoms together:**

- Polynomial grad ratio + logarithmic amplitude norm is consistent with `||g_lm_head||` decaying as the classifier converges, while `||g_amp||` stays nonzero. The amplitudes become the cheapest available knob, so loss starts pushing through them - not because that produces structure, but because they remain plastic.
- Recurrent missing points in update/grad-flow charts indicate intermittent gradient explosions that clipping recovers from but never resolves. The forward `h * (1 + b)` is unbounded in `b`, so once amplitudes drift, individual `(t, d)` cells of `(1 + b)` can go strongly negative or large. That is a per-cell sign-flip / multiplicative spike on top of `h`, which the rest of the network has to absorb.
- 2.6B tokens is enough that any structure the field was going to crystallize would have started to appear in the spectrum by now. A still-noise heatmap means the field is being used as an arbitrary content channel, not as a rhythmic prior. The model is adapting to the *current* random pattern at training time, then breaking at inference because the pattern keeps drifting.

**Root cause (working hypothesis):**

Unbounded multiplicative coupling. `forward()` is `hidden_states * (1.0 + b)` with no clamp, and amplitudes are unconstrained real parameters (init std 1.0, no L2 regularization in the head itself). After `irfft2` with ortho norm, individual `b` cells can reach order-unity once amplitude norm grows. The model can find local minima that depend on a specific drifting pattern rather than on stable structure.

---

## Change applied: bound the field

`forward()` now does:

```python
return hidden_states * (1.0 + torch.tanh(b))
```

This keeps `(1 + tanh(b))` in `(0, 2)` strictly. No sign flips, no multiplicative blow-up, gradient through `tanh` is well-behaved near zero (unit slope) and saturates at the edges (which is desired - we want the field to fight back when amplitudes try to push past unit scale).

What to watch for in the next run:

- `harmonic_grad_ratio` should stop growing polynomially. Either it stabilizes, or it grows much more slowly.
- `update-to-weight ratio` and `gradient flow` charts should stop dropping samples. If they still drop samples, the instability source is not exclusively the field.
- The spectrum heatmap is the real test. With the chaos generator removed, if structure is going to form, this is when it would.
- Inference quality should stop degenerating with continued training. If it does still degenerate, the problem is deeper than bounding.

If bounding is sufficient, we stop here. If it is not, the items below are next, in order.

---

---

## 2026-05-08 — Bounding alone was not enough

After 10000 steps on `december-small.yml` with the bounded `tanh(b)` form, the model again degenerated. Charts looked similar to the unbounded run: gradient ratio rising, amplitudes drifting logarithmically, spectrum heatmap still noise. Bounding prevented multiplicative chaos but did not introduce any *signal* that rewarded harmonic structure.

The deeper observation: pressure for any property has to come from one of three places: architecture, loss, or data. The harmonic head has *partial* architectural pressure (frozen Weyl phases) and *no* loss-based or data-based pressure. NLL has no notion of "be sparse in frequency" or "be coherent across positions," so under NLL alone the field drifts to whatever local minimum reduces NLL the most, which empirically is "act as a content side-channel" rather than "act as a rhythmic prior."

### Promotion: spectral concentration loss is now active

Promoting deferred-item-1's spirit (soft anchor) and folding it together with deferred-item-2 (sparse cell mask) into a single Hoyer-sparsity aux loss:

- `HarmonicField.concentration()` returns `H = (sqrt(N) - ||a||_1 / ||a||_2) / (sqrt(N) - 1)` in `[0, 1]`. 1 = single cell, 0 = uniform. Scale-invariant.
- `HarmonicField.aux_loss()` returns `SPARSITY_LAMBDA * (1 - H)`, with `SPARSITY_LAMBDA = 0.01` initially.
- `PraxisForCausalLM.forward()` adds the aux loss to the loss container under key `harmonic_concentration` whenever a field is present and training. The strategy combines it alongside MTP / RL / encoder aux losses.
- The dynamics callback logs the *raw H value* (not the loss) as the `harmonic_concentration` metric. Frontend shows it on the Dynamics tab as a linear-scale chart.

Why this is the right pressure to add:

- The Weyl phases are already frozen, so "few-cell support" combined with "frozen harmonic phases" means *the corpus uses these specific harmonics*. The aux loss is a parameter-space prior on what a "useful" amplitude grid looks like, and it is exactly the shape we want the heatmap to show.
- It is scale-invariant. It does not fight whatever amplitude norm the NLL gradient prefers; it only objects to that norm being spread across all cells.
- It needs no targets, no input dependence, no auxiliary modules. One scalar loss term, one constant.

What to watch for:

1. `harmonic_concentration` metric should rise above its init value (~0.2 with N(0,1) amplitudes) if the loss is doing its job.
2. The spectrum heatmap should develop visible bright cells and dark regions instead of uniform speckle.
3. If concentration rises but inference quality still degenerates, the field is committing to *useless* cells, which would suggest data-driven coupling is needed (input-conditional amplitudes).
4. If concentration stays near init even with the loss active, lambda needs to go up. Try 0.05 or 0.1.

Tradeoffs and tuning notes:

- `SPARSITY_LAMBDA = 0.01` was chosen so the aux loss starts at ~0.008 against an NLL of several nats - small but visible. Adjust upward if concentration does not rise; downward if the field collapses too fast and NLL suffers.
- Nothing in the loss tells the field *which* cells should win. Ties are broken by NLL gradient. If the field keeps collapsing onto an obviously-wrong cell, anneal lambda from zero so the spectrum has time to develop preference before sparsity locks it in.

If concentration does rise but the spectrum still does not produce meaningful structure, the next move is data-driven: making amplitudes a function of input embeddings via a small projection. That is real architectural work, kept as fallback.

---

## 2026-05-09 — Concentration rose, model still degenerate

At step 9000 of `december-small.yml` with the spectral concentration loss active:

- The "Harmonic Spectrum" heatmap is now ~80% black with a few colored speckles. The aux loss is doing exactly what it was asked to do: collapse the amplitude grid onto few cells.
- The other charts (gradient ratio, amplitudes, etc.) still grow polynomially. "Update-to-Weight Ratio" is sparse with many missing data points - the same intermittent-NaN signature we saw before bounding.
- Inference outputs collapsed onto repeating `[TOOL_CALL]` tokens with empty bodies. Classic mode-collapse onto a low-NLL fixed point.

**Reading:** Hoyer rewards "fewer cells" but says nothing about *which* cells. NLL gradient breaks the tie, and empirically it picks cells that act as content side-channels rather than rhythmic priors. Once the field collapses onto a few high-amplitude cells, `(1 + tanh(b))` saturates - a saturating cell at `tanh = -1` *zeroes out* feature `d` at every position matching that mode. That is feature annihilation by gating, and the model has no way to recover the killed features. The downstream layers find a low-NLL fixed point under the gate pattern (here: `[TOOL_CALL]` repetition) and lock in.

This matches the failure scenario the previous section flagged: "concentration rises but inference quality still degenerates - the field is committing to *useless* cells."

### Change applied: bound the gate away from 0

Two small adjustments to test whether the architecture can survive sparse gating at all, before committing to the larger input-conditional-amplitudes change:

```python
SPARSITY_LAMBDA: float = 0.001        # was 0.01
COUPLING_DEPTH: float = 0.5           # new
# forward: h * (1 + COUPLING_DEPTH * tanh(b))
```

Effect: the per-cell field range becomes `(0.5, 1.5)` instead of `(0, 2)`. Saturating cells now scale features by 0.5x or 1.5x; they cannot zero them. The lower lambda gives the spectrum more time to develop preference before sparsity locks it in.

What this experiment tells us:

- **If the model trains stably**: sparse gating was viable, feature annihilation was the proximal failure. Concentration may rise more slowly with the smaller lambda, but the spectrum should still develop visible structure without the model collapsing.
- **If the model still degenerates**: the static-amplitude design is fundamentally insufficient, and we move to input-conditional amplitudes (the next deferred item).

This is a cheap experiment - one constant changed, one multiplier added. A few hours of training on `december-small.yml` is the right test bed.

---

## 2026-05-09 (later) — Hoyer is content-blind; replace with forward-shift CCA

The bound-away-from-zero gate did not save the run; the model degenerated again with similar chart shapes. The remaining problem is structural, not numeric: **Hoyer rewards "concentrated," not "concentrated on something predictive."** NLL gradient breaks the tie, and the cells NLL likes are content side-channels, not rhythmic priors. We need a loss that points the field at *which* cells should win.

### Cross-pollination from CCA

Reading [arxiv:2512.23146](https://arxiv.org/html/2512.23146v1) (ReSU networks). Their "spectrum" is eigendecomposition, not Fourier - mostly orthogonal to our problem. But one idea transfers cleanly:

**Truncated CCA selects the directions that maximize past-future predictability.** That is the principled version of "fewer cells": pick the cells that capture predictive structure, not just any cells. In our static-amplitude setup with frozen Weyl phases, this reduces to a closed-form prior on the amplitude grid - no extra modules, no sampling.

### Closed-form derivation (k=1 forward shift)

Our field is `b[t, d] = sum_{f_t, f_d} amp[f_t, f_d] * cos(2*pi*f_t*t/T + 2*pi*f_d*d/D + phi[f_t, f_d]) * decay[f_t, f_d]`.

For two adjacent positions, expanding `cos(x+y) - cos(x) = -2*sin(x + y/2)*sin(y/2)` and taking expectation over (t, d) - the Weyl phases supply equidistribution that kills cross-terms:

```
E[(b_{t+1} - b_t)^2] = 4 * sum_{f_t, f_d} (amp * decay)^2 * sin^2(pi * f_t / T)
```

In the `F_t << T` regime (`F_t ~ 256-512`, `T = 32768`), `sin^2(pi*f_t/T) ~ (pi*f_t/T)^2`. Dropping constants and the fixed decay (which is not a learnable parameter), and normalizing by amplitude norm so the loss is scale-invariant:

```
smoothness = sum |amp[f_t, f_d]|^2 * (f_t / F_t)^2 / sum |amp|^2     in [0, 1]
```

Low = field is smooth in t (predictable across positions); high = field is dominated by fast temporal modes that decorrelate in one step.

### Change applied

- New constant `SMOOTHNESS_LAMBDA = 0.01`. Replaces `SPARSITY_LAMBDA`.
- New `HarmonicField.smoothness()` returns the closed-form value above.
- `aux_loss()` now returns `SMOOTHNESS_LAMBDA * smoothness()`.
- `concentration()` (Hoyer) is kept as a *diagnostic only* - still useful for asking "is the field committing to specific cells," just no longer the loss target.
- Loss container key in `modeling.py` renamed `harmonic_concentration` → `harmonic_smoothness` so the labeling is honest.
- Dynamics callback now logs both `harmonic_concentration` and `harmonic_smoothness`. Frontend has a new "Forward-Shift Smoothness" chart alongside the existing concentration chart.

Sanity checks:

- All amplitude at `f_t = 1`: `S ~ 1/F_t^2 ~ 0` (trivially smooth).
- All amplitude at `f_t = F_t`: `S = 1` (maximally rough).
- Isotropic amplitudes at init: `S ~ 1/3` (matches the asymptotic mean of `(f_t/F_t)^2`).
- Initial aux loss ~ `0.01 * 0.33 = 0.003`. Comparable to where Hoyer started.

### Why this is structurally different from Hoyer

Hoyer is *anisotropic in the wrong direction*: it cares about the *shape* of the amplitude distribution but is invariant to *which* axis the energy lives on. Smoothness is anisotropic in the *right* direction: it explicitly prefers low `f_t`, which is exactly the part of the spectrum that captures slowly-varying structure (sentence rhythm, paragraph cohesion, conversational turn-taking). High `f_t` modes - the kind NLL reaches for as content side-channels - get penalized.

The trivial minimum is "all amplitude at `f_t = 1`," a single low-frequency mode. That is fine: NLL fights back when more frequency content actually reduces NLL, and lambda controls the strength of the prior. The point of the prior is to *bias* the search, not pin it to a specific outcome.

What to watch in the next run:

1. `harmonic_smoothness` should fall from ~0.33 toward something smaller. If it stays at ~0.33, the loss is not winning enough fights against NLL; raise lambda (try 0.05).
2. The "Harmonic Spectrum" heatmap should now develop visible *low-row* structure (mass concentrated in the first few rows of the `[F_t, F_d]` grid). The previous Hoyer run produced sparse mass anywhere; this one should produce mass in a *direction*.
3. `harmonic_concentration` (still logged) is now a derived quantity. It will rise too - low-`f_t` mass tends to be concentrated - but for a *reason* now, not by fiat.
4. If smoothness falls but inference still degenerates, we are out of static-amplitude options. Promote input-conditional amplitudes (deferred-item-2) without further hesitation.

---

## 2026-05-11 — Static amplitudes were the ceiling. Promoting input-conditional.

Confirmed the predicted failure. With smoothness loss active, the spectrum heatmap developed clean directional structure (distinct low-`f_t` band, dark high-`f_t` band, even the corpus-driven wavy boundary we hoped for) and the headline charts started rebounding from their monotonic growth - all signs of the field finding a useful equilibrium. But inference still degenerated. The new attractor is `U+FFFD` repetition (Unicode replacement character, the renderer's way of saying "model is emitting byte tokens that do not form valid UTF-8"). Different attractor, same underlying mode collapse.

**The structural diagnosis is now clean.** A static amplitude grid produces the *same* `b[t, d]` for every input. Learned amplitudes, frozen phases, smoothness prior - it is all still mathematically just a position-dependent bias multiplier with no input awareness. For a corpus that mixes registers (dialogue, code, prose), one static field has to compromise across all distributions, and the compromise inevitably becomes a content-blind side channel that the rest of the model has to absorb or route around. The mode collapses we have seen (`[TOOL_CALL]` repetition, now `U+FFFD` repetition) are the model's "route around" answer.

### Change applied: static baseline + input-conditional delta

Implementing deferred-item-2 with the Takens-flavored conditioning structure we agreed in the deferred queue:

```python
ctx = pool_with_delays(hidden_states)     # [B, D * 4]
delta = amplitude_mlp(ctx)                # [B, F_t, F_d], MLP output zeroed at init
amps  = self.amplitudes + delta           # baseline + per-sequence perturbation
field = irfft2(amps * complex_spec, ...)  # [B, T, D]
```

Design choices and why:

- **Static baseline + delta**, not full replacement. `self.amplitudes` is kept as a learned parameter; the MLP predicts a *delta* on top. The MLP output layer is zero-initialized so the field starts as today's trained static field and *anneals* into input dependence. Preserves the smoothness loss's gains and avoids a cold-start regression.
- **Per-sequence granularity**, not per-position. One amplitude grid per input sequence, applied uniformly to all positions in that sequence. Per-position would force an IRFFT2 at every token (much bigger architectural shift); per-sequence answers the more tractable question: "is this sequence dialogue vs. code vs. prose, and what rhythms should the field commit to for this input?"
- **Conditioning input: recency-biased multi-scale pooling.** Pool the last `T // f` tokens for `f in [1, 2, 4, 8]` (full, half, quarter, eighth) and concatenate. Four views at increasing temporal resolution. The Takens "delayed observations at multiple scales" intuition adapted to per-sequence pooling. Not literal scalar-delay embedding (the theorem does not strictly apply here) but the transferable spirit.
- **MLP shape**: `Linear(D * 4, D * MLP_WIDTH_MULT) -> GELU -> Linear(D * MLP_WIDTH_MULT, F_t * F_d)`. `MLP_WIDTH_MULT = 1` by default so hidden width scales with experiment hidden_dim. Output layer zero-init.
- **Delta L2 regularizer**: aux loss now combines smoothness (on baseline) + `DELTA_LAMBDA * mean(delta^2)` (on perturbation). `DELTA_LAMBDA = 0.001` to start. Keeps the static baseline load-bearing so the MLP cannot grow into a content side-channel.

New diagnostic: `harmonic_delta_norm = rms(delta) / rms(baseline)`. Reads how much the MLP is adapting the field per sequence. 0 = static; rising = input dependence is real.

What to watch in the next run:

1. **Inference quality recovers**. This is the gate. If the model produces coherent text where the static field collapsed, the architecture finally has a path that doesn't require static feature gating.
2. **`harmonic_delta_norm` rises off zero but stays bounded** (target: ~0.01-0.1). At zero, the MLP is still doing nothing; if it grows to ~1, the perturbation is the same size as the baseline and the delta L2 may need bumping.
3. **Spectrum heatmap stays directional** (low-`f_t` band, dark high-`f_t` band). The static baseline keeps shaping the long-run spectrum; only the per-input modulation is dynamic.
4. **Update-to-Weight Ratio finally fills in** (still a separate concern, but if it persists with input-conditional amplitudes that points firmly at the measurement-side NaN explanation rather than training instability).

If this works, the natural next move is varying the conditioning structure (deferred-item-4's learnable frequency offsets, or richer pooling like dilated 1D convs). If it does *not* work, we have run out of cheap moves and the next step is the larger architectural rethink (per-position amplitudes, replacing the multiplicative coupling, or abandoning the harmonic head structure entirely).

---

## 2026-05-12 — Mode collapse traced to Serpent, not the harmonic head

The input-conditional amplitudes trained stably, then degenerated again into garbage-character outputs (`;`␛^���...`). Headline charts looked clean except for `Update-to-Weight Ratio` which appeared sparse on the dashboard.

**The dashboard sparsity was a perception artifact, not a NaN problem.** Pulling the live dynamics SQLite (`bf4b9cc88`, 922 rows, max step 9210), every column was fully populated - zero NaN/null/inf values across all 9,210 steps. What looked like "missing points" was the chart's log-scale y-axis spanning 5+ orders of magnitude: dense baseline values around 1e-8 collapsed to a near-flat line at the bottom while occasional spikes up to 5e-4 sat alone at the top.

**The real symptom: intermittent gradient spikes, getting more frequent before the collapse.**

```
spike rate (>100x median) in layer_1_grad_norm, by 1000-step bins:
  0-999:    9%   (initialization noise)
  1k-7k:   0-3%  (clean training)
  7k-8k:   15%   (something changed)
  8k-9k:    4%
  9k-9.2k: 17%   (collapse imminent)
```

Crucially, `harmonic_grad_ratio` had **zero spikes**. The field we have been working on for two weeks is innocent.

### Smoking gun: Serpent's `1/α` term

The decoder uses Serpent activation (`x + sin²(αx)/α + γ·sin(βx)`, per-feature learnable α, β, γ). Inspecting the live checkpoint, the smallest `|α|` values across all layers and recurrent passes:

- Layer 0 act.1.a: `|α|_min = 6.5e-04`  → `1/α = 1,538`
- Layer 0 act.3.a: `|α|_min = 1.66e-03` → `1/α = 602`
- Layer 1 act.1.a: `|α|_min = 2.87e-03` → `1/α = 348`

α is initialized from `Exponential(rate=1.0)`, which has nontrivial mass near 0 (P[a < 0.01] ≈ 1%). Across 170 features × 4 passes × 2 layers = 1,360 `α` values, ~10-14 features start in the danger zone, and NLL pressure pushes a few even lower over training.

Mechanism: for *typical* `|x| ~ O(1)` and typical α, `sin²(αx)/α ≈ αx²` (small, well-behaved). But when a single outlier `x` for one of those tiny-α features hits the regime `|αx| ~ π/2`, the term emits `~1/α = 1500+`. The gradient `∂f/∂α` has a `1/α²` denominator; on a typical batch it stays bounded (numerator ~ α²x²), but on a batch containing the right outlier it produces a one-step gradient explosion. Each spike is recoverable by gradient clipping, but the cumulative damage across many spiky steps eventually destroys coherence.

### Change applied: smooth-rectified inverse

Replace `1/α` with `α / (α² + ε²)` in `praxis/activations/serpent.py`, `ε = 0.1`:

```python
INV_FLOOR_EPS = 0.1
inv_a = a / (a * a + INV_FLOOR_EPS ** 2)
snake = x + torch.sin(a * x).square() * inv_a
```

Properties:

- `|α| >> ε`: matches original `1/α` exactly (the healthy regime is unchanged).
- `|α| << ε`: bounded by `1/ε = 10`, well below the 1538 we were seeing.
- Smooth and differentiable everywhere - no branch, no clamp boundary.
- One-line change. No re-init, no checkpoint surgery. Existing α values in the healthy range are unaffected; the dangerous ones are gently neutralized.

Sanity-tested: previously dangerous `α = 6.5e-4` with `x = 2415` now outputs `~2415.07` (bounded) instead of `~3953` (blown up). Gradient `|∂f/∂α|` max with the same tiny α drops from potentially millions to ~42.

### What this means for the harmonic work

The previous "the static field is the ceiling" diagnosis was probably *partially* wrong. The static field genuinely was insufficient (the input-conditional change still stands), but the *proximate* cause of the collapses we attributed to the field may have been Serpent spikes corrupting the model independently. We will know which after re-training with both changes active.

Re-training should be done with both the input-conditional amplitudes *and* the Serpent fix. If the model trains stably to coherent inference, we cannot cleanly attribute the win to either change individually - but at this point getting a stable run matters more than ablation.

---

## Deferred: items to revisit if spectral concentration is not enough

### 1. Soft anchor on amplitude scale

Add a regularizer that pins the amplitude norm to a target rather than letting weight decay drift it logarithmically forever:

```
lambda * (amp_l2 / sqrt(F_t * F_d) - sigma_target) ** 2
```

with `sigma_target` around 0.3-0.5. This gives the field a fixed energy budget. Within that budget the field can still shape itself - but it cannot collapse or inflate.

Why this comes after bounding: anchoring scale on top of an unbounded multiplicative pathway just adds a hyperparameter; bounding first removes the failure mode and then anchoring controls drift within a stable regime.

### 2. Input-conditional amplitudes — *promoted, now active (see 2026-05-11 section)*

If concentration rises but inference quality still degrades, the field is committing to cells but the *content* of those cells is not coupled to the actual corpus. The fix is to make amplitudes depend on the input rather than be static parameters:

```
amplitudes = f_phi(pooled_embedding)  # small MLP, [F_t * F_d] output
```

This is a real architectural change, not a tweak. It gives the field a path to encode actual corpus rhythm rather than a population-average rhythm. Worth keeping as a fallback because it is more invasive than aux-loss tuning.

**Takens-flavored variant** (worth trying as the conditioning structure if we go down this path): instead of pooling the current hidden state, feed a small window of dilated delays to the MLP:

```
amplitudes = f_phi([h_t, h_{t-1}, h_{t-2}, h_{t-4}, h_{t-8}])
```

Inspired by Takens' Embedding Theorem (Takens, 1981) - the result that a scalar time series, sampled at delays, topologically reconstructs the state of a smooth dynamical system. The strict theorem does not apply here (language is discrete, stochastic, multivariate), but the transferable intuition is "multi-scale delays carry latent state information" - which fits cleanly into the conditioning input. Dilated delays (1, 2, 4, 8) over learned attention weights specifically because the inductive bias is what we want to test: does explicit multi-scale temporal context give the field a better view than self-attention already provides? Keep this as the *first* conditioning structure to try rather than a single pooled vector.

### 3. Forward-shift / phase-coherence prior — *promoted, now active (see 2026-05-09 later section)*

The forward-shift idea was promoted to the live aux loss. The closed-form derivation gives a quadratic-in-`f_t` penalty on amplitude variance, normalized by amplitude norm. Implemented as `HarmonicField.smoothness()` and `SMOOTHNESS_LAMBDA = 0.01`. If smoothness falls but the model still degenerates, the next move is deferred-item-2 (input-conditional amplitudes).

### 4. Learnable frequency offsets (Time2Vec / Fourier-feature flavored)

Inspired by Time2Vec ([arxiv:1907.05321](https://arxiv.org/abs/1907.05321)) and Fourier-feature networks (Tancik et al.). Currently `f_t` runs over integers `1..F_t`. If the corpus has rhythms at non-integer multiples of the fundamental (e.g. avg sentence length is 17.3 tokens, not 16 or 32), the integer grid misses them.

The cheap, careful version: parameterize

```
f_t_effective[i] = i * (1 + delta[i])     # delta initialized to 0, learnable, weight-decayed toward 0
```

The integer grid stays as the default; small learned `delta[i]` lets the field detune to whatever corpus rhythm actually exists. Weyl phases recompute from the effective frequencies each forward pass, so equidistribution still holds approximately as long as `|delta|` stays small.

Why *not* the full Time2Vec move (learn `omega` and `phi` freely): Time2Vec's main contribution is learning both frequencies and phases, but the smoothness aux loss derivation depends on Weyl-equidistributed frozen phases - learnable phases would break the closed form. The transferable insight is narrower than the paper's pitch.

This is a small architectural change, complementary to (not redundant with) deferred-item-2. The two could be combined: input-conditional amplitudes on a slightly-detuned frequency grid. Keep on the list as a fine-tuning move once the gross-structure questions are settled.

---

## Open diagnostic: bound on the field range

Even after applying `tanh`, it is worth instrumenting the field directly. Adding three live stats to the dynamics callback would close a blind spot:

- `b.abs().max()` per step.
- `b.std()` per step.
- `(1 + tanh(b)).min()` and `.max()` per step (these will be in (0, 2) by construction now, but the *distribution* within that range is still informative).

If any of these go flat at the boundary (`tanh(b) ~ +/-1` for most cells), the field is saturating and amplitudes are still being pushed past where they have meaningful gradient. That would be evidence to add scale anchoring (deferred item 1) sooner.
