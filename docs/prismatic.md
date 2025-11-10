# Prismatic Attention v3.0: Architectural Diversity

## Overview

Prismatic v3.0 implements architectural diversity through **different positional encoding strategies**. Instead of gradient perturbations, experts use genuinely different architectures (RoPE vs ALiBi) while sharing merged learnable parameters.

## Key Changes from v2.0

**v2.0 (Gradient Perturbations):**
- Modified gradients after backward pass
- Created optimization trajectory diversity
- Result: Router collapse (99.7%/0.3% split)

**v3.0 (Architectural Diversity):**
- Different positional encodings per expert
- Soft-merge learnable parameters
- Hard-gate architectural operations
- Result: Both architectures remain viable

## Architecture

```
┌─────────────────────────────────────┐
│ Forward Pass                        │
├─────────────────────────────────────┤
│ 1. Compute routing probabilities    │
│ 2. Soft-merge parameters:           │
│    W_merged = Σ routing[i] × W_i    │
│ 3. Hard-gate architecture:          │
│    if routing[rope] > 0.5:          │
│        use RoPE                     │
│    else:                            │
│        use ALiBi                    │
│ 4. Forward with merged params       │
└─────────────────────────────────────┘
```

## Positional Encodings

### ALiBi (Attention with Linear Biases)
- Adds linear bias to attention scores
- Better for very long sequences
- No learned parameters

### RoPE (Rotary Position Embedding)
- Rotates Q/K based on position
- Better for length extrapolation
- No learned parameters

Both are **operational** differences, not parameter differences, making them ideal for architectural gating.

## Configuration

```yaml
router_type: prismatic
num_experts: 2
architectures: ["alibi", "rope"]  # Optional, defaults to ["alibi", "rope"]
attention_type: hex  # Must support pos_type parameter
```

Prismatic automatically creates experts with different architectures using modulo cycling:
- Expert 0: `architectures[0 % len(architectures)]` → ALiBi
- Expert 1: `architectures[1 % len(architectures)]` → RoPE
- Expert 2: `architectures[2 % len(architectures)]` → ALiBi (cycles back)
- etc.

## Usage

### Creating Architecturally Diverse Experts

```python
from praxis.attention.hex import HexAttention
from praxis.routers.prismatic import Prismatic

# Prismatic creates experts automatically with different architectures
prismatic = Prismatic(config, expert_class=HexAttention)

# With 2 experts: ALiBi, RoPE
# With 4 experts: ALiBi, RoPE, ALiBi, RoPE (cycles)
```

### Custom Architecture Patterns

```python
# Custom cycling pattern
config.architectures = ["rope", "rope", "alibi"]  # 2 RoPE, 1 ALiBi
config.num_experts = 6

prismatic = Prismatic(config, expert_class=HexAttention)
# Expert 0: rope
# Expert 1: rope
# Expert 2: alibi
# Expert 3: rope (cycle repeats)
# Expert 4: rope
# Expert 5: alibi
```

### Forward Pass

```python
inputs = torch.randn(batch_size, seq_len, hidden_size)
output, state, loss = prismatic(inputs, None)
```

## How It Works

1. **Routing**: Computes probabilities for each architecture
2. **Parameter Merging**: Soft-merges all learnable weights (QKV, output projections)
3. **Architecture Gating**: Selects RoPE or ALiBi based on routing probabilities
4. **Execution**: Runs forward pass with merged parameters through selected architecture

## Metrics

Prismatic tracks routing metrics:
- `routing/expert_0_weight`: ALiBi routing weight
- `routing/expert_1_weight`: RoPE routing weight
- `routing/entropy`: Distribution balance
- `routing/concentration`: Max routing weight
- `routing/balance`: Distribution evenness (1.0 = perfect)

### Healthy Routing

- Both experts receive >10% routing weight
- Entropy >0.5
- Concentration <0.9
- Router learns meaningful input-dependent patterns

## Theoretical Foundation

### Computational Substrate Hypothesis

Different architectures traverse the loss landscape differently. RoPE and ALiBi impose different inductive biases:

- **RoPE**: Rotation-based, better at compositional patterns
- **ALiBi**: Bias-based, better at uniform attention

The router learns which inductive bias fits which input patterns.

### Connection to "The Blind Watchmaker"

> "Different architectural constraints force different gradient trajectories through floating-point approximation space."

By using different positional encodings, we create genuine architectural diversity without sabotaging learning. Both architectures are proven approaches—neither is "worse," they simply have different strengths.

## Advantages over v2.0

1. **No gradient sabotage**: Both experts learn normally
2. **Proven architectures**: RoPE and ALiBi are both validated approaches
3. **Meaningful diversity**: Different inductive biases, not artificial perturbations
4. **Stable routing**: Router learns which architecture for which patterns

## Implementation Notes

### HexAttention Support

HexAttention now accepts a `pos_type` parameter:

```python
attention = HexAttention(config, pos_type="rope")  # or "alibi"
```

The `pos_type` determines which positional encoding is applied during forward pass.

### Parameter Compatibility

RoPE and ALiBi differ only in **operations**, not parameters:
- Both use the same QKV projections
- Both use the same output projection
- Only the positional encoding step differs

This makes them perfect for soft-parameter merging with hard-operation gating.

## Future Extensions

Potential architectural differences to explore:
- Learned vs fixed positional encodings
- Different attention mechanisms (FlexAttention variants)
- Different activation functions
- Different normalization strategies

The key requirement: operations must differ while parameters remain compatible for merging.

---

**Version:** 3.0.0
**Last Updated:** 2025-01-09
**Breaking Changes:** Complete rewrite from v2.0 gradient perturbation approach
