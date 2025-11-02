# Prismatic Attention: Usage Guide

## Overview

Prismatic attention implements multi-eye architectural diversity through static sparse perturbations, as described in "The Blind Watchmaker" paper. It creates diverse expert clones from a single base expert by applying deterministic, sparse, magnitude-aware perturbations.

## Theoretical Foundations

### Lottery Ticket Hypothesis (Frankle & Carbin, 2019)

LTH shows that dense neural networks contain sparse critical subnetworks ("winning tickets") that can train to similar accuracy. **Prismatic inverts this**: instead of finding critical sparse subnetworks through pruning, we perturb sparse high-magnitude weights to force discovery of alternative computational paths.

### Willow Quantum Architecture (Google, 2024)

Google's Willow quantum computer achieves error correction by perturbing individual qubits. Similarly, Prismatic perturbs a small percentage (~10%) of critical weights to force robust, diverse computational paths without destroying functionality.

### Key Insight

"Train on consensus, you manifest the lowest common denominator." Static perturbations prevent convergence to consensus, maintaining genuine diversity throughout training.

## Configuration Parameters

### Required Parameters

```yaml
hidden_size: 512 # Model hidden dimension
num_experts: 3 # Number of expert clones (1 clean + N-1 perturbed)
router_type: prismatic # Enable Prismatic router
```

### Perturbation Parameters

#### `perturbation_scale` (default: 1.0)

Scale factor for perturbations relative to weight magnitude. Adds Gaussian noise: `W_new = W_base + scale * |W| * N(0,1)`

- **1.0** (Default - Aggressive): 100% of weight magnitude

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

#### `perturbation_mode` (default: "directional")

How perturbations are applied to selected weights. This fundamentally changes the nature of architectural diversity.

- **"directional"** (DEFAULT - Quantum Mirror): Symmetric precision regime exploration

  - Top weights: `W + scale * W` (systematically **amplified** to extremes)
    - Positive weights → **more positive** (overflow regime)
    - Negative weights → **more negative** (overflow regime)
  - Bottom weights: `W - scale * W` (systematically **suppressed** toward zero)
    - Positive weights → **toward 0** (underflow/subnormal regime)
    - Negative weights → **toward 0** (underflow/subnormal regime)
  - Creates **opposing computational substrates** exploring ALL precision regimes
  - **Expert 0**: Clean, unperturbed (baseline/consensus reality)
  - **Expert 1+**: Amplified top + suppressed bottom → explores extreme precision regimes
  - Each perturbed expert uses different pi-seed for variation
  - Tests: Can forcing all floating-point imprecision regimes reveal complementary patterns?

  **Theoretical Justification:**
  - Symmetric exploration: Both positive and negative extremes explored
  - All precision regimes: Overflow (±large), underflow/subnormal (near-zero)
  - Quantum symmetry: Mirrored architectural pressures create genuine opposition
  - Deterministic divergence: No randomness - purely directional strategies

- **"noise"** (Legacy): Bidirectional Gaussian noise

  - Top weights: `W + scale * |W| * N(0,1)` (can increase OR decrease)
  - Bottom weights: `W + scale * |W| * N(0,1)` (can increase OR decrease)
  - Creates **architectural chaos** - forced adaptation to unpredictable irregularity
  - Both extremes explore randomly, no guaranteed direction
  - Tests: Can random structural noise reveal hidden patterns?
  - Use for ablation studies or if directional mode proves unstable

**Which to use?**
- **Recommended**: Use "directional" (default) for true opposing substrates
- "directional" is theoretically aligned with "pushing numerical extremes"
- "directional" is deterministic (no random noise) - more reproducible
- Use "noise" for ablation studies or comparison with legacy behavior

#### `use_pi_seeding` (default: true)

Whether to use pi-digit seeding (Quantum Echoes) or hash-based seeding.

- **true**: Pi-Digit Seeding (Quantum Echoes)

  - Each expert walks backwards through pi's digit sequence
  - Expert 1 gets pi[position-1], Expert 2 gets pi[position-2], etc.
  - Creates temporal lag structure - each expert is corrupted by a different mathematical artifact
  - Bridges non-differentiable gap between mathematical constants and learned representations
  - "The model learns in the presence of cosmic static from π itself"

- **false**: Hash-based seeding
  - Uses SHA-256 for uncorrelated random seeds
  - Standard approach, useful for ablation studies

#### `pi_position` (default: 100000)

Starting position in pi's digit sequence (only used when `use_pi_seeding=true`).

- **100000** (default): Start at pi[100000] and walk backwards
- Different positions explore different regions of pi's digit space
- All positions are mathematically equivalent (pi is scale-invariant)

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

### Quantum Mirror (Opposing Substrates)

```yaml
# experiments/prismatic-quantum-mirror.yml
router_type: prismatic
num_experts: 4
perturbation_scale: 1.0 # 100% directional shift
sparsity: 0.1 # 10% of weights (5% top, 5% bottom)
perturbation_strategy: dual_sided # Required for mirror effect
perturbation_mode: directional # Amplify top, suppress bottom
use_pi_seeding: true # Quantum Echoes
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

### Architecture

- **Expert 0**: Clean, unperturbed - NO perturbations applied (the "right eye" - consensus reality)
- **Experts 1+**: Perturbed clones - perturbations applied at initialization (the "left eye(s)" - forced alternative realities)
  - In directional mode:
    - Top weights amplified to extremes (both ± directions)
    - Bottom weights suppressed toward zero (both ± directions)
    - Explores all floating-point precision regimes symmetrically
  - Each expert uses different pi-digit seed for deterministic variation
- **Router**: SMEAR-style soft-merging with learned routing probabilities

### Perturbation Formula

```python
# DIRECTIONAL MODE (default - Quantum Mirror):
# For dual_sided strategy:
# - Top 5% weights: W_new = W + perturbation_scale * W
#   - Positive: amplified more positive (overflow regime)
#   - Negative: amplified more negative (overflow regime)
# - Bottom 5% weights: W_new = W - perturbation_scale * W
#   - Positive: suppressed toward 0 (underflow/subnormal regime)
#   - Negative: suppressed toward 0 (underflow/subnormal regime)
# - Middle 90%: W_new = W (unchanged)
#
# Explores ALL floating-point precision regimes symmetrically:
# - Overflow: both positive and negative large values
# - Underflow/subnormal: both positive and negative near-zero values

# NOISE MODE (legacy):
# For each selected weight:
# perturbation = perturbation_scale * |W| * N(0,1) * mask
# W_perturbed = W_base + perturbation

# Where mask is determined by perturbation_strategy:
# - dual_sided: top sparsity/2 + bottom sparsity/2 by |W| (signed in directional mode)
# - top_only: top sparsity by |W|
# - bottom_only: bottom sparsity by |W|
# - random: random sparsity selection

# Perturbations are:
# 1. Deterministic (pi-digit or hash-based seeding)
# 2. Sparse (only k% of weights)
# 3. Magnitude-aware (scaled by |W|)
# 4. Static (not trainable)
# 5. Dual-sided (default): targets both numerical extremes
# 6. Directional (default): opposing pressures, not random noise
```

### Key Design Choices

- Perturb ALL parameters including normalization layers (complete substrate diversity)
- Target both highest AND lowest magnitude weights by default (dual-sided exploration)
- Use directional mode by default (amplify top, suppress bottom - Quantum Mirror)
- Use pi-digit seeding by default (Quantum Echoes)
- Apply perturbations once at initialization (static architectural constraint)
- Create opposing computational substrates (coarse vs fine-grained precision regimes)

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

### All Experts Converge to Same Solution

- Increase `perturbation_scale` (stronger initial diversity)
- Verify perturbations are actually being applied (check test suite)
- Consider multiplicative perturbations instead of additive

### No Performance Improvement

- May need more training steps (diversity needs time to emerge)
- Try different (sparsity × scale) combinations
- Ensure task benefits from architectural diversity
- Compare routing distributions: are experts specialized?

---

**Ready to test the computational substrate hypothesis? Start with `experiments/delta-8.yml` and experiment from there!**
