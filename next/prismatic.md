# Prismatic Attention v7.0: Architectural Diversity via Sparse Routing

## Overview

Prismatic v7.0 returns to the core hypothesis of "The Blind Watchmaker": **architectural diversity reveals patterns single approaches cannot discover**.

**Core Concept:**
- **N experts** with cycling positional encodings via modulus
- **Architecture cycle**: [ALiBi, RoPE, ALiBi, RoPE, ...]
- **Sparse routing**: ONE expert per sequence (k=1)
- **Standard causal masking**: No temporal tricks
- **Same everything else**: Architecture, parameters, training objective

**Examples:**
- `num_experts=2`: ALiBi, RoPE
- `num_experts=3`: ALiBi, RoPE, ALiBi
- `num_experts=4`: ALiBi, RoPE, ALiBi, RoPE

## Philosophy: Back to First Principles

**From the research:**
> "Different architectural constraints force different gradient trajectories through floating-point approximation space."

**What we're testing:**
- ALiBi constrains attention through linear distance biases
- RoPE constrains attention through rotational position encoding
- Same input, same masking → different learned representations
- Router learns which constraint suits which pattern

**No temporal masking, no backward inference, no complexity.**

Just clean architectural diversity.

## Architecture

```
┌─────────────────────────────────────────────┐
│ Forward Pass (v7.0 Architectural Diversity) │
├─────────────────────────────────────────────┤
│ 1. Compute routing probabilities:           │
│    seq_repr = input.mean(dim=1)             │
│    logits = router(LayerNorm(seq_repr))     │
│    probs = softmax(logits)                  │
│                                             │
│ 2. Select expert per sequence (top-1):      │
│    expert_idx = argmax(probs)  # [batch]    │
│                                             │
│ 3. Execute selected expert:                 │
│    if expert_idx == 0:                      │
│        output = ALiBi_expert(input)         │
│    else:                                    │
│        output = RoPE_expert(input)          │
│                                             │
│ 4. Load balancing loss:                     │
│    Encourage 50/50 expert usage             │
└─────────────────────────────────────────────┘
```

## Positional Encoding Strategies

### ALiBi (Expert 0)
**Attention with Linear Biases:**
- Adds bias: `score + slope * (kv_idx - q_idx)`
- Linear distance penalty
- No learned parameters
- Better for long sequences
- Simpler gradient landscape

### RoPE (Expert 1)
**Rotary Position Embedding:**
- Rotates Q/K by position-dependent angles
- Encodes relative position through rotation
- More complex geometric constraints
- Different attention patterns
- Richer gradient landscape

**Both use standard causal masking. Only the positional encoding differs.**

## Configuration

```yaml
router_type: prismatic
num_experts: 2  # Or 3, 4, etc. - cycles through architectures
attention_type: hex
router_balance_loss_coef: 0.01
```

Experts are created with cycling `encoding` via modulus: `architectures[i % len(architectures)]`

## Usage

**With 2 experts:**
```python
config = PraxisConfig(
    num_experts=2,
    router_type="prismatic"
)
# Expert 0: ALiBi
# Expert 1: RoPE
```

**With 4 experts:**
```python
config = PraxisConfig(
    num_experts=4,
    router_type="prismatic"
)
# Expert 0: ALiBi
# Expert 1: RoPE
# Expert 2: ALiBi (cycle repeats)
# Expert 3: RoPE
```

**With 3 experts (uneven):**
```python
config = PraxisConfig(
    num_experts=3,
    router_type="prismatic"
)
# Expert 0: ALiBi
# Expert 1: RoPE
# Expert 2: ALiBi
```

## Metrics

### Routing Metrics (Research Tab)

**Expert Routing Weights:**
- `routing/expert_0_weight`: ALiBi routing probability
- `routing/expert_1_weight`: RoPE routing probability

**Architecture Usage:**
- `architecture/alibi_usage`: % sequences using ALiBi (0-100%)
- `architecture/rope_usage`: % sequences using RoPE (0-100%)

**Balance Metrics:**
- `routing/entropy`: Distribution balance
- `routing/concentration`: Max weight (collapse indicator)
- `routing/variance`: Routing stability
- `routing/balance`: Distance from uniform (1.0 = perfect)

### Gradient Dynamics (Dynamics Tab)

- `expert_0_grad_norm`: ALiBi gradient norm
- `expert_0_grad_var`: ALiBi gradient variance
- `expert_1_grad_norm`: RoPE gradient norm
- `expert_1_grad_var`: RoPE gradient variance

### What to Watch

**Healthy Training:**
- Both experts used (~50/50 or meaningful specialization)
- Entropy > 0.5 (not collapsed)
- Both receive gradients
- Loss converges to reasonable value (not 0!)

**Warning Signs:**
- One expert weight → 1.0 (routing collapse)
- Loss → 0.0 (something broke)
- One expert has zero gradients (not learning)

## What Changed from v6.0

**Removed (~200 lines of complexity):**
- ❌ Backward temporal masking
- ❌ Super-diagonal blocking
- ❌ Ghost position parameter
- ❌ Forward/backward mask creation
- ❌ Target leakage prevention hacks

**Added (~5 lines of simplicity):**
- ✅ Expert 0: `encoding = "alibi"`
- ✅ Expert 1: `encoding = "rope"`
- ✅ Standard causal masking everywhere

**Net result: Much simpler, philosophically aligned, actually valid.**

## Theoretical Foundation

### The Computational Substrate Hypothesis

**From "The Blind Watchmaker":**
> "Floating-point rounding errors create high-dimensional pattern spaces within numerical approximation artifacts. Different architectural constraints force different traversals through this space."

**ALiBi traversal:**
- Linear bias gradients
- Simple distance relationships
- Smooth gradient landscape

**RoPE traversal:**
- Rotational gradients
- Geometric position relationships
- More complex gradient landscape

**Different traversals → different patterns discovered → richer learned representations.**

### Why This Tests the Core Hypothesis

**v6.0 tested:** Temporal perspective (backward inference)
**Result:** Fundamentally broken (target leakage)

**v7.0 tests:** Architectural diversity (encoding)
**Expected:** Clean experiment testing actual research hypothesis

**The question:**
Does routing between ALiBi and RoPE outperform either architecture alone?

**If yes:** Architectural diversity helps (research validated)
**If no:** Single architecture sufficient (but we learned something)

## Expected Behavior

### Training Dynamics

**Early (random routing):**
- ~50/50 ALiBi vs RoPE
- Router learning pattern preferences

**Mid (specialization):**
- Router may discover one architecture is generally better
- Or may learn pattern-specific routing (some sequences prefer ALiBi, others RoPE)

**Late (convergence):**
- Stable routing pattern
- Both experts continue learning (or one dominates if genuinely better)

### Performance Comparison

**Baseline:** Single architecture (ALiBi only or RoPE only)

**Prismatic:** Sparse routing between ALiBi and RoPE

**Hypothesis:** Diversity should match or exceed single-architecture baseline.

## Advantages over v6.0

1. **Actually Valid**: No target leakage, no training collapse
2. **Much Simpler**: Standard causal masking, standard training
3. **Philosophically Pure**: Tests actual research hypothesis
4. **Interpretable**: Can analyze which sequences prefer which architecture
5. **Debuggable**: No temporal tricks to reason about

## Future Extensions

- More positional encoding types (Absolute, T5-style relative, etc.)
- More experts (3-4 different strategies)
- Token-level routing (finer granularity)
- Layer-specific architectures (different encodings per layer)

**The key: Simple, clean architectural diversity. Let gradient descent figure out what works.**

---

**Version:** 7.0.0
**Last Updated:** 2025-01-11
**Breaking Changes:**
- Removed all bidirectional temporal masking
- Removed ghost_position parameter
- Experts now differ by encoding (ALiBi vs RoPE), not masking
- Standard causal masking everywhere
- Much simpler implementation (~200 lines removed)
