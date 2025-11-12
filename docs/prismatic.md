# Prismatic Attention v6.0: Sparse Bidirectional Temporal Routing

## Overview

Prismatic v6.0 implements **sparse bidirectional temporal perspectives** through different attention masking strategies with sequence-level expert selection.

**Core Concept:**

- **Expert 0 (Forward Eye)**: Standard causal masking - sees past, infers future
- **Expert 1 (Backward Eye)**: Inverted causal masking - sees future, infers past
- Both use **identical architecture** (ALiBi positional encoding)
- Differentiation is purely through **temporal perspective**
- **Sparse routing**: ONE expert per sequence (not soft parameter merging)

## Key Changes from v5.0

**v5.0 (SMEAR-style soft parameter merging):**

- Soft-merged expert parameters
- Ran both experts with merged parameters
- Had fundamental contradiction: merged params but different masks needed
- Complex and confusing data flow

**v6.0 (Sparse MoE):**

- Sparse routing: ONE expert selected per sequence
- Each expert runs independently with its own mask
- Clean separation: router creates masks, experts process
- Standard MoE pattern (well-understood, debuggable)
- Load balancing loss encourages 50/50 usage

## Architecture

```
┌─────────────────────────────────────────────┐
│ Forward Pass (v6.0 Sparse Routing)          │
├─────────────────────────────────────────────┤
│ 1. Compute routing probabilities:           │
│    seq_repr = input.mean(dim=1)             │
│    logits = router(LayerNorm(seq_repr))     │
│    probs = softmax(logits)                  │
│                                             │
│ 2. Select expert per sequence (top-1):      │
│    expert_idx = argmax(probs)  # [batch]    │
│                                             │
│ 3. Create appropriate mask:                 │
│    if expert_idx == 0:                      │
│        mask = forward_mask (sees past)      │
│    else:                                    │
│        mask = backward_mask (sees future)   │
│                                             │
│ 4. Execute selected expert:                 │
│    output = expert[expert_idx](input, mask) │
│                                             │
│ 5. Load balancing loss:                     │
│    Encourage 50/50 expert usage             │
└─────────────────────────────────────────────┘
```

### Attention Masks

**Forward Masking (Expert 0 - "sees past"):**

```
Position can attend to:
     0  1  2  3
  0 [✓  ✗  ✗  ✗]
  1 [✓  ✓  ✗  ✗]
  2 [✓  ✓  ✓  ✗]
  3 [✓  ✓  ✓  ✓]

Standard autoregressive: predicts next token from history
```

**Backward Masking (Expert 1 - "sees future"):**

```
Position can attend to:
     0  1  2  3
  0 [✓  ✓  ✓  ✓]
  1 [✗  ✓  ✓  ✓]
  2 [✗  ✗  ✓  ✓]
  3 [✗  ✗  ✗  ✓]

Inverted causality: must infer past from future patterns
```

## Positional Encoding

Both experts use **ALiBi** (Attention with Linear Biases):

- Adds linear bias to attention scores based on distance
- Better for long sequences
- No learned parameters
- Identical across both experts

The differentiation comes purely from **attention masking**, not positional encoding.

## Configuration

```yaml
router_type: prismatic
num_experts: 2 # Must be 2 for bidirectional masking
attention_type: hex
pos_type: alibi # Both experts use ALiBi
router_balance_loss_coef: 0.01 # Load balancing coefficient
```

Experts are created identically - differentiation happens through masking in the router.

## Usage

### Basic Setup

```python
from praxis.configuration import PraxisConfig

config = PraxisConfig(
    hidden_size=512,
    num_heads=8,
    num_experts=2,
    router_type="prismatic",
    pos_type="alibi"
)
```

### Forward Pass

```python
inputs = torch.randn(batch_size, seq_len, hidden_size)
output, cache, state, aux_loss = model(inputs)

# aux_loss includes load balancing loss
total_loss = task_loss + aux_loss
```

## How It Works

1. **Routing**: Sequence-level softmax routing (mean pooling → router network)
2. **Sparse Selection**: Top-1 expert per sequence (argmax)
3. **Mask Creation**: Router creates forward or backward mask based on selection
4. **Execution**: Selected expert processes sequence with appropriate mask
5. **Load Balancing**: MSE loss from uniform distribution (encourages 50/50 usage)

## Metrics

### Routing Metrics (Research Tab)

**Expert Routing Weights:**

- `routing/expert_0_weight`: Forward expert average routing probability
- `routing/expert_1_weight`: Backward expert average routing probability
- Shows which temporal perspective is preferred on average

**Balance Metrics:**

- `routing/entropy`: Distribution balance (high = balanced, low = collapsed)
- `routing/concentration`: Max routing weight (1.0 = collapsed, 0.5 = balanced)
- `routing/variance`: Routing stability across experts
- `routing/balance`: Distance from uniform (1.0 = perfect balance, 0.0 = collapsed)

**Debug Metrics:**

- `routing/balance_loss`: MSE from uniform distribution
- `routing/avg_confidence`: Average max probability (routing confidence)

### Gradient Dynamics (Dynamics Tab)

Call `router.log_gradient_dynamics()` during training to enable:

- `expert_0_grad_norm`: L2 norm of expert 0 gradients
- `expert_0_grad_var`: Variance of expert 0 gradients
- `expert_1_grad_norm`: L2 norm of expert 1 gradients
- `expert_1_grad_var`: Variance of expert 1 gradients

### What to Watch

**Healthy Training:**

- Expert weights stay near 0.5 each (balanced usage)
- Entropy > 0.5 (not collapsed)
- Both experts receive gradients
- Routing learns meaningful patterns (not random)

**Warning Signs:**

- One expert weight approaches 1.0 (routing collapse)
- Entropy near 0 (deterministic routing to one expert)
- One expert has zero gradients (not learning)

## Theoretical Foundation

### Quantum Echoes: Two Temporal Observers

**The Core Idea:**
What if a model could "think backwards"? Not just predict the future from the past, but infer the past from the future?

**Forward Eye (Expert 0):**

- Sees: tokens 0...i
- Learns: "What comes next based on history"
- Standard autoregressive prediction

**Backward Eye (Expert 1):**

- Sees: tokens i...N
- Learns: "What must have come before to produce this future"
- Forced backwards inference (cannot directly see past)

**Sparse Selection:**
The model learns WHEN each perspective is useful:

- Some sequences benefit from forward reasoning
- Others benefit from backward reasoning
- Router learns to distinguish and select appropriately

### Forced Inference vs Direct Access

**Traditional transformers:**

- Direct access to all previous tokens
- Learn patterns through exposure
- "I see X, so Y follows"

**Backward expert:**

- No direct access to past
- Must **infer** what came before from what comes after
- "I see Y, so X must have preceded it"
- Develops backwards causal reasoning

### Why Sparse Instead of SMEAR?

**SMEAR's fundamental problem:**

- Merged parameters → ONE forward pass → can only apply ONE mask
- Bidirectional masking needs DIFFERENT masks per expert
- These are mutually exclusive

**Sparse MoE solution:**

- Separate forward passes per expert
- Each expert applies its own mask
- Outputs combined (not parameters)
- Standard, well-understood approach

## Advantages over v5.0

1. **Actually Works**: No parameter merging contradiction
2. **Simpler**: Standard sparse MoE pattern
3. **Debuggable**: Easy to trace which expert processes which sequence
4. **Interpretable**: Can analyze which sequences prefer which perspective
5. **Efficient**: Same computational cost as baseline when balanced (50/50 split)
6. **Standard**: Well-studied MoE dynamics and training patterns

## Implementation Details

### Clean Separation of Concerns

**Router (`prismatic.py`):**

- Creates forward/backward masks
- Selects expert per sequence
- Passes appropriate mask to expert via `attention_mask`
- Computes load balancing loss

**HexAttention (`hex.py`):**

- Accepts external mask via `attention_mask` parameter
- Applies mask using score_mod (combines with ALiBi)
- No knowledge of temporal direction

**Experts (TransformerBlocks):**

- Identical architecture
- Just process input with provided mask
- No special masking logic

### Load Balancing

Encourages 50/50 expert usage through MSE loss:

```python
# Average routing probability per expert
avg_probs = routing_probs.mean(dim=0)  # [0.5, 0.5] ideal

# MSE from uniform distribution
target = [0.5, 0.5]
balance_loss = MSE(avg_probs, target)
```

## Future Extensions

Potential enhancements:

- **Token-level routing**: More fine-grained than sequence-level
- **Top-k routing**: Combine multiple perspectives per sequence
- **More perspectives**: Near-past, far-past, near-future, far-future
- **Adaptive k**: Learn optimal number of experts per sequence
- **Hierarchical**: Different perspectives at different layers

The key principle: Same architecture, different **perspectives** on the sequence.

---

**Version:** 6.0.0
**Last Updated:** 2025-01-11
**Breaking Changes:**

- Removed SMEAR soft parameter merging (incompatible with bidirectional masking)
- Implemented sparse MoE with sequence-level routing
- Router now creates and passes masks to experts
- Experts are truly identical (no instance-level differentiation)
- Added load balancing loss for stable training
