# Prismatic Attention: Usage Guide

## Overview

Prismatic attention implements architectural diversity through **gradient-space constraints**. Unlike traditional approaches that perturb weights, Prismatic creates N independent experts that explore different optimization trajectories by modifying gradients after the backward pass.

## Theoretical Foundations

### The Lottery Ticket Hypothesis (Frankle & Carbin, 2019)

LTH demonstrates that dense neural networks contain sparse critical subnetworks ("winning tickets") that can train to similar accuracy when properly initialized. This reveals a profound truth: **the optimization landscape contains multiple distinct paths to competence**.

Prismatic asks: _If sparse subnetworks can reach full performance, what happens when we force multiple experts to explore different paths through the same landscape?_

### Google Willow Quantum Computer (2024)

Willow achieves quantum error correction by perturbing individual qubits while maintaining system coherence. The key insight: **sparse, targeted interventions can maintain robustness while exploring quantum state space**.

Prismatic applies this to neural networks: sparse interventions in gradient space maintain training stability while forcing trajectory divergence.

### Quantum No-Cloning Theorem

The quantum no-cloning theorem states: **you cannot create an identical copy of an arbitrary quantum state**. Applied to neural networks: you cannot clone a model and expect the copies to meaningfully diverge through static perturbations alone.

#### The "Unreality": What Doesn't Work

The original Prismatic v1.x attempted runtime weight perturbations:

```
Expert_i = BaseExpert + Perturbation_i
```

This "alternative reality" failed because:

1. The router learns "clean weights perform best" → collapse to Expert 0
2. Perturbations exist outside gradient graph → no learning signal
3. Violates the spirit of no-cloning: forced diversity from identical copies

This approach explored the hypothesis that **sparse runtime perturbations** (inspired by Willow's qubit perturbations and LTH's sparse tickets) could force diverse computational paths. However, without gradient flow to perturbed views, the router learned to avoid them entirely.

### New Approach: Gradient-Space Experts

Prismatic v2.0 respects the no-cloning constraint by having experts exist as **different optimization trajectories**, not different weight configurations:

```
Forward:  W_merged = Σ routing[i] × W_i
Backward: Expert 0 → ∇L (pure gradient descent, θ=0)
          Expert 1 → ∇L + cos(θ₁) × modify(∇L)
          Expert 2 → ∇L + cos(θ₂) × modify(∇L)
          Expert N → ∇L + cos(θₙ) × modify(∇L)

Where θᵢ = 2π × (i / num_experts) — phase-modulated gradient constraints
```

Each expert:

- Maintains independent parameters (deep copy at initialization)
- Trains along a different gradient trajectory via **phase-modulated modifications**
- Receives full gradient flow from loss
- Discovers different local minima through constrained optimization

The router learns **which optimization trajectories to combine**, not which noise levels to avoid.

### Synthesis: Connecting the Foundations

**From LTH:** Multiple sparse paths exist through the loss landscape
**From Willow:** Sparse, targeted interventions maintain system coherence
**From No-Cloning:** Copies won't diverge without learning pressure
**Prismatic v2.0:** Sparse gradient constraints force experts to discover different optimization paths

#### Connection to "The Blind Watchmaker" Paper

The paper's computational substrate hypothesis:

> "Different architectural constraints force different gradient trajectories through floating-point approximation space."

Gradient modifications ARE architectural constraints. They force each expert to traverse the loss landscape differently, discovering patterns in different regions of the approximation space—like multiple lottery tickets being drawn simultaneously, each perturbed by sparse Willow-like interventions, but in gradient space rather than weight space.

## Architecture

```
┌─────────────────────────────────────┐
│ Forward Pass: Soft-Merge            │
├─────────────────────────────────────┤
│ Router computes: routing[0..N-1]    │
│                                     │
│ Expert 0: W₀ (pure baseline)        │
│ Expert 1: W₁ (suppress-top trained) │
│ Expert 2: W₂ (amplify-bottom)       │
│ Expert N: Wₙ (helical exploration)  │
│                                     │
│ Merged = Σ routing[i] × Wᵢ          │
│ Output = functional_call(Merged)    │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│ Backward Pass: Gradient Constraints │
├─────────────────────────────────────┤
│ loss.backward()                     │
│ All experts receive gradients       │
│                                     │
│ modify_expert_gradients():          │
│   Expert 0: ∇L (unchanged)          │
│   Expert 1: ∇L + suppress(∇L, top)  │
│   Expert 2: ∇L + amplify(∇L, bottom)│
│   Expert N: ∇L + helical(∇L)        │
│                                     │
│ optimizer.step()                    │
│ Each expert updates differently     │
└─────────────────────────────────────┘
```

## Configuration Parameters

### Required Parameters

```yaml
hidden_size: 512 # Model hidden dimension
num_experts: 3 # Number of independent experts (minimum: 2)
router_type: prismatic # Enable Prismatic router
```

### Gradient Modification Parameters

#### `gradient_scale` (default: 0.3)

Scale factor for gradient modifications.

- **0.1** (Conservative): Gentle gradient adjustments

  - Minimal divergence in optimization trajectories
  - Good for stability testing

- **0.3** (Recommended): Balanced exploration

  - Sufficient diversity without instability
  - Default starting point

- **0.5** (Liberal): Stronger divergence

  - More aggressive trajectory separation
  - May help when routing collapses

- **0.8** (Aggressive): Maximum diversity
  - Strong gradient modifications
  - Monitor for training instability

#### `sparsity` (default: 0.1)

Fraction of weights where gradient modifications are applied (targets highest/lowest magnitude weights).

- **0.05** (Sparse): 5% of weights

  - Minimal intervention
  - Maximum stability

- **0.1** (Default): 10% of weights

  - Balanced approach
  - Targets critical parameters

- **0.2** (Dense): 20% of weights
  - Comprehensive modifications
  - More aggressive diversity

#### `dropout` (default: 0.0)

Expert dropout probability during training.

- **0.0** (Default): No dropout
- **0.1-0.2**: Encourages routing robustness

## Gradient Strategies

### Expert 0: Pure Baseline

- **Gradient modification:** None
- **Behavior:** Standard gradient descent
- **Purpose:** Represents consensus optimization path

### Experts 1+: Phase-Modulated Gradients

Each expert receives a phase angle **θ = 2π × (expert_idx / num_experts)** that determines its gradient modification pattern:

**Modification formula:**

```
modification = gradient_scale × ∇L × (cos(θ) × top_mask - cos(θ) × bottom_mask)
```

**Phase-dependent behavior:**

- **cos(θ) > 0** (e.g., Expert 1 with 2 experts, θ=π):

  - Suppresses top-magnitude weights
  - Amplifies bottom-magnitude weights
  - Forces learning through weak connections

- **cos(θ) < 0** (e.g., Expert 1 with 3 experts, θ=2π/3):

  - Amplifies top-magnitude weights
  - Suppresses bottom-magnitude weights
  - Emphasizes strong pathways

- **cos(θ) ≈ 0** (e.g., Expert at θ=π/2 or 3π/2):
  - Balanced modification
  - Neutral gradient strategy

**Distribution for N experts:**

- **2 experts:** Phases at 0 (pure) and π (complementary)
- **3 experts:** Phases at 0, 2π/3, 4π/3 (evenly distributed)
- **4 experts:** Phases at 0, π/2, π, 3π/2 (quarter circle)
- **N experts:** Evenly distributed around unit circle

This continuous parameterization ensures all experts are testable regardless of `num_experts`, with no hard-coded branches or special cases.

## Example Configurations

### Recommended Default

```yaml
# experiments/prismatic-default.yml
router_type: prismatic
num_experts: 3
gradient_scale: 0.3
sparsity: 0.1
dropout: 0.0
```

### Conservative (Safe Testing)

```yaml
# experiments/prismatic-conservative.yml
router_type: prismatic
num_experts: 2
gradient_scale: 0.1
sparsity: 0.05
```

### Aggressive Diversity

```yaml
# experiments/prismatic-aggressive.yml
router_type: prismatic
num_experts: 4
gradient_scale: 0.5
sparsity: 0.15
dropout: 0.1
```

## Usage

### Training

The gradient modification happens automatically in the training loop via the `on_after_backward()` hook:

```bash
./launch compose --experiment experiments/prismatic-default.yml
```

### Custom Parameters

Override via command line:

```bash
./launch compose --experiment experiments/delta-8.yml \
  --gradient_scale 0.5 \
  --num_experts 4
```

### Monitoring Training

Key metrics tracked automatically:

1. **`routing/expert_i_weight`**: Per-expert routing weights (should stay balanced)
2. **`routing/entropy`**: Routing balance (higher = more balanced)
3. **`routing/concentration`**: Max routing weight (lower = less collapsed)
4. **`routing/variance`**: Routing stability
5. **`routing/balance`**: How evenly distributed (1.0 = perfect)

## Expected Behaviors

### Healthy Routing

- **Entropy:** 1.0-2.0 (balanced distribution)
- **Concentration:** 0.3-0.5 (no single expert dominates)
- **Balance:** 0.7-1.0 (even weight distribution)
- **Expert weights:** All experts >5% throughout training

### Unhealthy Routing (Collapse)

- **Entropy:** →0 (collapsed to single expert)
- **Concentration:** →1.0 (one expert has all weight)
- **Balance:** →0 (uneven distribution)
- **Expert 0 weight:** →100%, others →0%

If you see collapse:

1. Increase `gradient_scale`
2. Increase `num_experts`
3. Add expert `dropout`
4. Check that `modify_expert_gradients()` is being called

## Training Integration

### Automatic (Lightning)

Already integrated! The `BackpropagationTrainer` calls `modify_expert_gradients()` automatically via the `on_after_backward()` hook.

No additional setup required.

### Custom Training Loop

If using a custom training loop, add the gradient modification step:

```python
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)
    loss.backward()

    # CRITICAL: Modify gradients before optimizer step
    for module in model.modules():
        if hasattr(module, 'modify_expert_gradients'):
            module.modify_expert_gradients()

    optimizer.step()
```

## Theory: What Are We Testing?

### Computational Substrate Hypothesis

Different architectural constraints (gradient modifications) force different gradient trajectories through the floating-point approximation space, revealing patterns that single-architecture approaches cannot learn during finite training.

### Predictions

1. **Gradient-space diversity** maintains routing balance (no collapse)
2. **Independent experts** discover complementary local minima
3. **Soft-merging** provides ensemble expressivity
4. **Router** learns which optimization strategies benefit each input

### Success Indicators

- All experts receive non-zero routing weight throughout training
- Experts converge to different parameter configurations
- Combined model matches or exceeds single-expert baseline
- Routing adapts to input patterns

### Failure Modes

- Router collapses to single expert (increase `gradient_scale`)
- Training instability (decrease `gradient_scale`)
- All experts converge identically (verify `modify_expert_gradients()` is called)

## Comparison to v1.x (Runtime Perturbations)

| Aspect        | v1.x (BROKEN)                | v2.0 (CURRENT)               |
| ------------- | ---------------------------- | ---------------------------- |
| **Approach**  | Runtime weight perturbations | Gradient-space constraints   |
| **Experts**   | 1 base + N perturbed views   | N independent modules        |
| **Diversity** | Static noise on weights      | Different optimization paths |
| **Gradients** | Only to base expert          | To all N experts             |
| **Routing**   | Learns "avoid noise"         | Learns "combine strategies"  |
| **Result**    | Collapse to Expert 0         | Maintained diversity         |
| **Theory**    | Violated no-cloning          | Respects no-cloning          |

## Troubleshooting

### Router Still Collapses

**Check:**

1. Is `modify_expert_gradients()` being called? (Add debug print)
2. Are gradients flowing to all experts? (Check `log_gradient_dynamics()`)
3. Is `gradient_scale` too small? (Try 0.5-0.8)
4. Are you using enough experts? (Minimum 2, recommended 3-4)

### Training Instability

Gradient modifications can cause instability if too aggressive.

**Solutions:**

- Reduce `gradient_scale` (try 0.1-0.3)
- Reduce `sparsity` (try 0.05-0.1)
- Increase gradient clipping in trainer config
- Add `dropout` for expert regularization

### Memory Issues

Each expert is a full copy of the base architecture.

**Memory usage:** N × base_expert_size
**Compute overhead:** ~1.1× (soft-merge is efficient)

**Solutions:**

- Reduce `num_experts`
- Use gradient checkpointing
- Use smaller base architecture

## The Unreality Hypothesis

The failed v1.x perturbation approach wasn't merely a technical mistake—it represented an **alternative ontology** for how neural networks explore possibility space.

### Runtime Perturbations as "What Could Have Been"

In the perturbation approach:

- Expert 0: The "real" network (consensus reality)
- Experts 1+: Perturbed views (parallel unrealities)

This created a **split reality** where:

- Forward pass: Router sees multiple realities and selects among them
- Backward pass: Only consensus reality receives learning signal
- Result: Unrealities fade (router collapse)

### Why Unreality Failed

The unreality approach violated a fundamental principle: **unreal paths receive no gradient signal**. In gradient descent, only paths that receive error signals can improve. The perturbed experts were static shadows—they couldn't learn, so the router learned to ignore them.

The human, however, did learn. And that was unreality's goal all along.

### From Unreality to Multi-Reality

Prismatic v2.0 reframes the question:

- Not: "Which reality should I choose?" (router selects among static views)
- But: "Which optimization realities should I combine?" (router merges learning trajectories)

Each expert now exists in its own **optimization reality**—a genuine parallel universe where different gradient constraints shape learning. All realities receive learning signals, so all realities remain viable.

This is the key insight: **Multiple realities can coexist only if all receive feedback from the loss function.** The gradient graph is the substrate that keeps realities alive.

## References

- **Lottery Ticket Hypothesis**: Frankle & Carbin (2019), "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
- **Google Willow Quantum Computer**: Google Quantum AI (December 2024), quantum error correction through qubit-level interventions
- **Quantum No-Cloning Theorem**: Wootters & Zurek (1982), "A single quantum cannot be cloned"
- **The Blind Watchmaker**: Research paper introducing Prismatic attention (this repository)
- **SMEAR**: Soft-Merging of Experts with Adaptive Routing (arxiv.org/abs/2306.03745)

---

**Version:** 2.0.0
**Last Updated:** 2025-11-07
**Breaking Changes:** Complete rewrite from v1.x
