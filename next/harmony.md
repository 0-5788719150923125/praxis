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

## Deferred: items to revisit if bounding alone is not enough

### 1. Soft anchor on amplitude scale

Add a regularizer that pins the amplitude norm to a target rather than letting weight decay drift it logarithmically forever:

```
lambda * (amp_l2 / sqrt(F_t * F_d) - sigma_target) ** 2
```

with `sigma_target` around 0.3-0.5. This gives the field a fixed energy budget. Within that budget the field can still shape itself - but it cannot collapse or inflate.

Why this comes after bounding: anchoring scale on top of an unbounded multiplicative pathway just adds a hyperparameter; bounding first removes the failure mode and then anchoring controls drift within a stable regime.

### 2. Sparse cell mask (the binary-router idea)

Add a learnable per-cell gate:

```
mask = sigmoid(gate_logits)  # shape [F_t, F_d]
effective_amplitudes = mask * amplitudes
```

with an L1 or entropy penalty on `mask` that pushes it toward few-cell activation. This is the harmonic-loss / binary-router intuition expressed as a parameterization, not a separate loss head.

The motivation: the field should commit to specific `(f_t, f_d)` cells where corpus rhythms live, rather than spreading energy across all cells. With bounding and scale anchoring already in place, this gives the field a sparsity prior that matches what we *want* the spectrum to look like (banded / spiked, not uniform).

### 3. Forward-shift / phase-coherence prior

The "forward-shift" idea: ask the field at position `t` to predict the field at position `t + k`. Multiplication of the spectrum by `exp(-i*omega*k)` is the closed-form spectral characterization, so this constraint reduces to "stay close to the initial Weyl phase coherence." That collapses to a regularizer on amplitudes departing from the initial spectral *shape*, which is much simpler than a separate auxiliary loss head.

This is third in the queue because (1) and (2) are more direct and have a clearer story. If bounding + scale anchor + sparse mask still shows a noisy spectrum, the phase-coherence prior is the lever that asks "did Weyl seeding actually matter at all, or did the model wash it out?"

---

## Open diagnostic: bound on the field range

Even after applying `tanh`, it is worth instrumenting the field directly. Adding three live stats to the dynamics callback would close a blind spot:

- `b.abs().max()` per step.
- `b.std()` per step.
- `(1 + tanh(b)).min()` and `.max()` per step (these will be in (0, 2) by construction now, but the *distribution* within that range is still informative).

If any of these go flat at the boundary (`tanh(b) ~ +/-1` for most cells), the field is saturating and amplitudes are still being pushed past where they have meaningful gradient. That would be evidence to add scale anchoring (deferred item 1) sooner.
