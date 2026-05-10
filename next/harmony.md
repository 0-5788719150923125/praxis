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

## Deferred: items to revisit if spectral concentration is not enough

### 1. Soft anchor on amplitude scale

Add a regularizer that pins the amplitude norm to a target rather than letting weight decay drift it logarithmically forever:

```
lambda * (amp_l2 / sqrt(F_t * F_d) - sigma_target) ** 2
```

with `sigma_target` around 0.3-0.5. This gives the field a fixed energy budget. Within that budget the field can still shape itself - but it cannot collapse or inflate.

Why this comes after bounding: anchoring scale on top of an unbounded multiplicative pathway just adds a hyperparameter; bounding first removes the failure mode and then anchoring controls drift within a stable regime.

### 2. Input-conditional amplitudes

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
