# Prismatic Attention: Usage Guide

## Overview

Prismatic attention implements multi-eye architectural diversity through **runtime sparse perturbations**, as described in "The Blind Watchmaker" paper. It creates diverse expert views from a single base expert by applying deterministic, sparse, magnitude-aware perturbations at every forward pass—ensuring architectural diversity persists as the base expert learns.

## Theoretical Foundations

### Lottery Ticket Hypothesis (Frankle & Carbin, 2019)

LTH shows that dense neural networks contain sparse critical subnetworks ("winning tickets") that can train to similar accuracy. **Prismatic inverts this**: instead of finding critical sparse subnetworks through pruning, we perturb sparse high-magnitude weights to force discovery of alternative computational paths.

### Willow Quantum Architecture (Google, 2024)

Google's Willow quantum computer achieves error correction by perturbing individual qubits. Similarly, Prismatic perturbs a small percentage (~10%) of critical weights to force robust, diverse computational paths without destroying functionality.

### Key Insight

"Train on consensus, you manifest the lowest common denominator." Runtime perturbations prevent convergence to consensus—as the base expert learns, perturbations are always applied relative to its current state, maintaining genuine diversity throughout training.

## Configuration Parameters

### Required Parameters

```yaml
hidden_size: 512 # Model hidden dimension
num_experts: 3 # Number of expert clones (1 clean + N-1 perturbed)
router_type: prismatic # Enable Prismatic router
```

### Perturbation Parameters

#### `perturbation_scale` (default: 0.8)

Scale factor for perturbations relative to weight magnitude.

- **0.8** (Default - Balanced): 80% suppression/amplification

  - Balanced exploration without excessive instability
  - Recommended starting point for most experiments

- **1.0** (Aggressive): 100% of weight magnitude

  - Adds noise equal to the weight's magnitude
  - Forces exploration genuinely outside the learned world model
  - Bottom weights: ±100% perturbation can flip activation thresholds
  - Top weights: ±100% perturbation cascades through softmax
  - Tests "forced exploration" hypothesis in its strongest form
  - Stability maintained by: 90% unperturbed, LayerNorm, soft-merging

- **0.1** (Moderate): 10% of weight magnitude

  - More conservative, variations within consensus
  - Good if 1.0 proves unstable
  - May not wake up dormant bottom weights

- **0.01** (Conservative): 1% of weight magnitude
  - Minimal perturbation
  - Bottom weights barely affected
  - Gentlest form of diversity

- **2.0** (Very Aggressive): 200% of weight magnitude
  - Maximum diversity pressure
  - Can flip weight signs frequently
  - For testing extreme limits of the hypothesis

#### `sparsity` (default: 0.1)

Fraction of weights to perturb (targets highest-magnitude weights by default).

- **0.01** (Ultra-sparse): 1% of weights

  - Willow-inspired: minimal perturbation
  - Maximum stability
  - Tests "minimal intervention" hypothesis

- **0.05** (Very sparse): 5% of weights

  - Sparse obstacles
  - Good balance of diversity and stability

- **0.1** (Default): 10% of weights

  - LTH sweet spot: "lottery tickets"
  - Targets critical parameters
  - Recommended starting point

- **0.2** (Moderate): 20% of weights
  - More comprehensive perturbation
  - Useful for larger models

#### `perturbation_strategy` (default: "dual_sided")

Strategy for selecting which weights to perturb. Controls which regions of the computational substrate are explored.

- **"dual_sided"** (Default - NEW): Perturb top sparsity/2 + bottom sparsity/2 by magnitude

  - **Top weights**: Expose coarse-grained float32 artifacts (large absolute rounding)
  - **Bottom weights**: Expose fine-grained float32 artifacts (large relative rounding, subnormal behavior)
  - Both extremes reveal different numerical precision regimes
  - Creates symmetry in exploration of the computational substrate
  - May activate dormant pathways suppressed during training
  - Tests if low-magnitude weights contain latent patterns
  - Default: 5% from top + 5% from bottom (10% total)

- **"top_only"**: Perturb top-k% by absolute magnitude (original approach)

  - Targets "lottery tickets" - most critical parameters
  - Maximum impact through exponential cascade amplification
  - Aligns with standard Lottery Ticket Hypothesis interpretation
  - Use for comparison with previous experiments

- **"bottom_only"**: Perturb bottom-k% by absolute magnitude

  - Targets suppressed/dormant parameters
  - Tests if gradient descent suppressed potentially useful patterns
  - Control experiment for understanding dual-sided behavior

- **"random"**: Randomly select k% of weights
  - Uniform distribution across parameter space
  - Control for ablation studies
  - Less biased by current parameter values

#### `perturb_by_magnitude` (default: true)

Whether to use magnitude-based selection or random selection.

- **true**: Use perturbation_strategy (dual_sided, top_only, or bottom_only)
- **false**: Equivalent to perturbation_strategy="random"

#### `perturbation_mode` (default: "attractive")

How perturbations are applied to selected weights. This fundamentally changes the nature of architectural diversity.

- **"attractive"** (DEFAULT - Neuronal Regeneration): Attract attention to dormant pathways

  - Top weights: `W - scale * W` (systematically **suppressed** toward zero)
    - Prunes lottery tickets → forces gradient corrections
    - Creates new sparse weights → continuous neuronal turnover
  - Bottom weights: `W + scale * W` (systematically **amplified** away from zero)
    - Wakes dormant neurons with large relative perturbations
    - Forces weak signals to matter
  - Creates **productive instability** through pruning/restoration cycles
  - **Expert 0**: Clean, unperturbed (baseline)
  - **Expert 1+**: Suppressed top + amplified bottom → explores dormant pathways
  - Tests: Does forced neuronal turnover aid learning?

  **Theoretical Justification:**
  - Inverts standard approach: prunes strong, amplifies weak
  - Small weights get huge perturbations (forced to participate)
  - Gradients correct → creates new small weights elsewhere
  - Cycle repeats → continuous regeneration
  - Combined with iterative reasoning: temporal oscillation creates dynamic cycling

- **"repulsive"** (Extreme Exploration): Repel weights to numerical limits

  - Top weights: `W + scale * W` (amplified to extremes)
  - Bottom weights: `W - scale * W` (suppressed toward zero)
  - Explores all floating-point precision regimes (overflow/underflow)
  - Deterministic - no randomness

- **"noise"** (Legacy): Bidirectional Gaussian noise

  - Top weights: `W + scale * |W| * N(0,1)` (random)
  - Bottom weights: `W + scale * |W| * N(0,1)` (random)
  - Creates architectural chaos
  - Use for ablation studies

**Which to use?**
- **Recommended**: Use "attractive" (default) for neuronal regeneration
- "attractive" tests if dormant pathway exploration aids learning
- "repulsive" for extreme precision regime exploration
- "noise" for ablation studies or legacy comparison

#### `focal_pattern` (default: "radial_helical")

Pattern for modulating perturbations. Controls how weight perturbations create transformation signatures that attention can learn to recognize.

- **"radial_helical"** (DEFAULT - Prismatic Lens): Radial focusing + helical waves

  - Each expert focuses at different radial positions in weight space
  - Helical modulation creates spiral patterns radiating from focal points
  - Uses π for both: focal_length (π×100) and wavelength (π×1000)
  - Creates transformation signatures combining hierarchical (radial) + periodic (helical) structure
  - **Recommended**: Richest structure, literal "prism" behavior
  - The lens focuses, the waves interfere, the router learns

- **"radial"**: Pure lens focusing

  - Each expert focuses at different positions in weight space
  - Gaussian decay from focal point: `exp(-distance/focal_length)`
  - Creates hierarchical center-to-edge gradients in transformations
  - No wave structure—just focal hierarchy

- **"helical"**: Pure wave modulation (original approach)

  - Spiral patterns with harmonic phase offsets
  - Perturbations modulated by: `cos(2π·position/wavelength + phase)`
  - Each expert gets different phase: `phase = expert_idx * 2π / num_experts`
  - Creates wave interference patterns when experts merge
  - No focal hierarchy—just harmonic oscillations

- **"none"**: No modulation
  - Simple deterministic perturbations
  - Clean baseline for ablation studies

#### `focal_length` (default: π × 100)

Focal length for radial lens pattern. Controls how quickly focal strength decays from the focal point.

- **π × 100** (Default): Smooth Gaussian focusing
- Larger values: Gentler focusing (wider lens aperture)
- Smaller values: Sharper focusing (tighter lens)
- Only used when `focal_pattern` is "radial" or "radial_helical"

#### `helical_wavelength` (default: π × 1000)

Wavelength for helical wave pattern. Controls oscillation frequency in weight space.

- **π × 1000** (Default): Smooth harmonic oscillations
- Smaller values: Tighter spirals, more frequent oscillations
- Larger values: Gentler spirals, slower oscillations
- Only used when `focal_pattern` is "helical" or "radial_helical"
- Uses π directly in the modulation formula (Euler's formula connection)

## Example Configurations

### Conservative (Safe Default)

```yaml
# experiments/prismatic-conservative.yml
router_type: prismatic
num_experts: 3
perturbation_scale: 0.01 # 1% directional shift
sparsity: 0.1 # 10% of weights (5% top, 5% bottom)
perturb_by_magnitude: true
perturbation_mode: directional # Default - amplify top, suppress bottom
```

### Moderate Exploration

```yaml
# experiments/prismatic-moderate.yml
router_type: prismatic
num_experts: 4
perturbation_scale: 0.1 # 10% directional shift
sparsity: 0.1 # 10% of weights
perturb_by_magnitude: true
perturbation_mode: directional # Default
```

### Aggressive Diversity (Forced Exploration)

```yaml
# experiments/prismatic-aggressive.yml
router_type: prismatic
num_experts: 4
perturbation_scale: 1.0 # 100% perturbations
sparsity: 0.05 # 5% of weights (stability)
perturb_by_magnitude: true
perturbation_mode: noise # Chaotic exploration
```

### Prismatic Lens (DEFAULT)

```yaml
# experiments/prismatic-default.yml
router_type: prismatic
num_experts: 4
perturbation_scale: 0.8 # 80% pruning/amplification
sparsity: 0.1 # 10% of weights (5% top, 5% bottom)
perturbation_strategy: dual_sided
perturbation_mode: attractive # Wake dormant, prune dominant (default)
focal_pattern: radial_helical # Lens + waves (default)
focal_length: 314.159  # π × 100
helical_wavelength: 3141.59 # π × 1000
```

### Ablation: Pure Radial Lens

```yaml
# experiments/prismatic-radial.yml
router_type: prismatic
num_experts: 3
focal_pattern: radial # Hierarchical focusing only, no waves
```

### Ablation: Pure Helical Waves

```yaml
# experiments/prismatic-helical.yml
router_type: prismatic
num_experts: 3
focal_pattern: helical # Wave interference only, no lens
```

### Ablation: No Modulation

```yaml
# experiments/prismatic-none.yml
router_type: prismatic
num_experts: 2
focal_pattern: none # Simple perturbations, clean baseline
```

### Extreme Exploration (Repulsive Mode)

```yaml
# experiments/prismatic-repulsive.yml
router_type: prismatic
num_experts: 4
perturbation_scale: 0.8
sparsity: 0.1
perturbation_strategy: dual_sided
perturbation_mode: repulsive # Push to numerical extremes
focal_pattern: radial_helical
```

### Willow-Inspired (Minimal Perturbation)

```yaml
# experiments/prismatic-willow.yml
router_type: prismatic
num_experts: 2
perturbation_scale: 0.5 # 50% perturbations
sparsity: 0.01 # 1% of weights (qubit-like)
perturb_by_magnitude: true
```

### Ablation Study (Random Perturbations)

```yaml
# experiments/prismatic-ablation.yml
router_type: prismatic
num_experts: 3
perturbation_scale: 0.01
sparsity: 0.1
perturb_by_magnitude: false # Random selection for comparison
```

## Usage

### Basic Usage

```bash
./launch compose --experiment experiments/prismatic-conservative.yml
```

### With Custom Parameters

You can override parameters via command line or environment variables:

```bash
./launch compose --experiment experiments/delta-8.yml \
  --perturbation_scale 0.1 \
  --sparsity 0.05
```

### Monitoring Training

Key metrics to track:

1. **Routing distribution**: Are all experts being used?
2. **Loss curves**: Does diversity help or hurt?
3. **Gradient norms**: Are perturbations causing instability?
4. **Expert contributions**: Which experts dominate at which training stages?

## Theory: What Are We Testing?

### Computational Substrate Hypothesis

Different architectural constraints force different gradient trajectories through the floating-point approximation space, revealing patterns that single-architecture approaches cannot learn during finite training.

### Predictions

1. **Sparse extreme perturbations** should force exploration of genuinely different loss landscape regions
2. **Soft-merging** should adaptively combine diverse experts based on input patterns
3. **Static diversity** should persist throughout training (no collapse to single expert)
4. **Perturbed experts** should discover complementary patterns, not redundant ones

### Expected Behaviors

#### Success Indicators

- All experts receive non-zero routing weight throughout training
- Perturbed experts converge to different solutions than clean expert
- Combined model outperforms single-expert baseline
- Routing adapts to input patterns (different experts for different tasks)

#### Failure Modes

- Router collapses to 100% weight on clean expert
- Perturbed experts fail to converge (too unstable)
- All experts converge to identical solutions (insufficient diversity)
- Training instability (gradient explosions, NaN losses)

## Experimental Recommendations

### Phase 1: Validation (Conservative)

```yaml
perturbation_scale: 0.01
sparsity: 0.1
num_experts: 2-3
```

Goal: Verify stability, measure routing behavior

### Phase 2: Exploration (Moderate)

```yaml
perturbation_scale: 0.1
sparsity: 0.1
num_experts: 3-4
```

Goal: Test diversity hypothesis with safe parameters

### Phase 3: Forced Exploration (Aggressive)

```yaml
perturbation_scale: 1.0
sparsity: 0.05
num_experts: 2-3
```

Goal: Test maximum diversity pressure with sparse obstacles

### Phase 4: Scaling Study

Vary `sparsity` and `perturbation_scale` systematically to find optimal (diversity × stability) tradeoff.

## References

- **Lottery Ticket Hypothesis**: Frankle & Carbin (2019), "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
- **Willow Quantum Computer**: Google Quantum AI (2024)
- **The Blind Watchmaker**: Research paper introducing Prismatic attention (this repository)
- **SMEAR**: Soft-Merging of Experts with Adaptive Routing (arxiv.org/abs/2306.03745)

## Implementation Details

### Runtime Perturbation Architecture

Unlike the previous approach (perturb at init), Prismatic now applies perturbations **at every forward pass**:

1. **Base expert** trains normally (single module, receives gradients)
2. At forward pass, create **perturbed views** of current base weights
3. Merge views based on routing probabilities
4. Apply merged weights via `torch.func.functional_call`

This ensures:
- Architectural diversity persists as base expert learns
- No convergence between "experts" (they're runtime views, not separate modules)
- Perturbations always relative to current learned state

**Expert views**:
- **View 0**: Clean, unperturbed - consensus reality (the "right eye")
- **Views 1+**: Perturbed in-flight - alternative realities (the "left eye(s)")
  - Each view has different focal point (radial_helical mode)
  - Each view has different phase offset (helical component)
  - Deterministic (reproducible across forward passes)
- **Router**: SMEAR-style soft-merging with learned routing probabilities

### Perturbation Formula (Runtime Application)

```python
# At each forward pass, for expert_idx > 0:

# 1. Select weights to perturb (dual_sided by default)
mask = top 5% + bottom 5% by magnitude

# 2. Calculate base perturbation (attractive mode default)
base_perturbation = -scale * W * mask  # Suppress top, amplify bottom

# 3. Apply focal pattern modulation (radial_helical default)
focal_point = (expert_idx / num_experts) * num_weights
distance = |positions - focal_point|

# Radial component (Gaussian lens, using π)
focal_strength = exp(-distance / (π × 100))

# Helical component (harmonic waves, using π)
phase = expert_idx * 2π / num_experts
helical = cos(2π * distance / (π × 1000) + phase)

# Combined: The Prismatic Lens
modulation = focal_strength * helical
perturbation = base_perturbation * modulation

# 4. Create perturbed view (non-destructive)
W_view = W_base + perturbation

# 5. Merge all views by routing weights
W_merged = Σ(routing_weight[i] * W_view[i])

# 6. Apply merged weights via functional_call
output = functional_call(base_expert, W_merged, inputs)
```

Perturbations are:
1. **Runtime** (applied every forward pass, not just at init)
2. **Sparse** (only 10% of weights by default)
3. **Magnitude-aware** (scaled by |W|)
4. **Deterministic** (reproducible, not trainable)
5. **Dual-sided** (targets both numerical extremes)
6. **π-modulated** (focal_length=π×100, wavelength=π×1000)
7. **View-based** (single base expert, multiple perturbed views)

### Key Design Choices

- Perturb ALL parameters including normalization layers (complete substrate diversity)
- Target both highest AND lowest magnitude weights by default (dual-sided exploration)
- Use "attractive" mode by default (suppress top, amplify bottom - neuronal regeneration)
- Use radial_helical focal pattern by default (lens focusing + wave interference)
- **Apply perturbations at runtime** (every forward pass, persistent diversity as base expert learns)
- Create transformation signatures through π-modulated weight perturbations
- Single base expert receives gradients (no convergence between separate expert modules)
- Explore both numerical precision regimes (overflow via top weights, underflow via bottom weights)

## Convergence Tracking

Expert routing weights are automatically tracked and visualized in the Research tab. Metrics include per-expert routing weights, entropy, concentration, and variance. Charts appear automatically when using SMEAR or Prismatic routers.

## Troubleshooting

### Router Collapses to Single Expert

- Increase `perturbation_scale` (more diversity pressure)
- Decrease `sparsity` (more stability for perturbed experts)
- Add routing auxiliary losses (encourage balanced expert usage)

### Training Instability

- Decrease `perturbation_scale`
- Decrease `sparsity` (fewer perturbations)
- Enable gradient clipping
- Reduce learning rate

### Views Not Diverse Enough

With runtime perturbations, views are always diverse by construction. If routing still collapses:

- Increase `perturbation_scale` (stronger perturbations)
- Try different `focal_pattern` (radial_helical creates richest structure)
- Verify perturbations are being applied (check test suite)

### No Performance Improvement

- May need more training steps (diversity needs time to emerge)
- Try different (sparsity × scale) combinations
- Ensure task benefits from architectural diversity
- Compare routing distributions: are experts specialized?

---

**Ready to test the computational substrate hypothesis? Start with `experiments/delta-8.yml` and experiment from there!**
