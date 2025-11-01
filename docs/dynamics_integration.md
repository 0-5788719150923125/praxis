# Gradient Dynamics - Automatic Integration

## Overview

The Dynamics tab visualizes expert gradient learning patterns to validate the dual-sided perturbation hypothesis.

**Core Question**: Do aggressive perturbations (scale=1.0) on both top and bottom weights actually force different learning dynamics?

**What we're testing**:
- Do bottom weights "wake up" under ±100% perturbation?
- Do perturbed experts explore genuinely different gradient trajectories?
- Is the divergence between clean vs perturbed experts meaningful?

---

## Automatic Integration (Already Done!)

**Gradient dynamics are logged automatically when using Prismatic router.** No manual integration required.

When you train with:
```yaml
router_type: prismatic
```

The `DynamicsLoggerCallback` automatically:
1. Detects Prismatic routers in your model
2. Logs gradient dynamics every 10 steps (after backward, before optimizer step)
3. Writes to `build/runs/{hash}/dynamics.db`
4. Makes data available in the Dynamics tab

**That's it!** Just use Prismatic and the Dynamics tab will populate during training.

---

## Manual Integration (Advanced Users Only)

### Step 1: Import the Logging Function

```python
# In your training script (e.g., main.py or trainer)
# No new dependencies needed - already in Prismatic
```

### Step 2: Log Gradients During Training

Add this in your training loop **after `loss.backward()` but before `optimizer.step()`**:

```python
# Training loop
for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(batch)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # >>> ADD THIS: Log gradient dynamics <<<
    dynamics_metrics = log_gradient_dynamics(model, step)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Log dynamics to database (alongside regular metrics)
    if dynamics_metrics and step % log_freq == 0:
        logger.log_dynamics(step, dynamics_metrics)
```

### Step 3: Helper Function

Add this helper to extract dynamics from your model:

```python
def log_gradient_dynamics(model, step: int) -> Optional[Dict]:
    """
    Extract gradient dynamics from Prismatic routers in model.

    Returns aggregated dynamics across all Prismatic layers.
    """
    all_dynamics = []

    # Check if model uses Prismatic routing
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'locals'):
        for layer_idx, layer in enumerate(model.decoder.locals):
            # Check if this layer has a Prismatic router
            if hasattr(layer, 'router') and hasattr(layer.router, 'log_gradient_dynamics'):
                dynamics = layer.router.log_gradient_dynamics()
                if dynamics:
                    all_dynamics.append(dynamics)

    if not all_dynamics:
        return None

    # Aggregate across layers (simple mean for POC)
    # TODO: Could also track per-layer dynamics separately
    aggregated = _aggregate_dynamics(all_dynamics)

    return aggregated


def _aggregate_dynamics(dynamics_list: List[Dict]) -> Dict:
    """
    Aggregate gradient dynamics across multiple Prismatic layers.

    For POC: Simple averaging. Could be extended to track per-layer separately.
    """
    if len(dynamics_list) == 0:
        return {}

    if len(dynamics_list) == 1:
        return dynamics_list[0]

    # Average across layers
    expert_gradients = {}
    divergence_scores = {}

    # Collect all expert indices
    all_experts = set()
    for dyn in dynamics_list:
        all_experts.update(dyn.get('expert_gradients', {}).keys())

    # Average gradient norms for each expert
    for expert_key in all_experts:
        tier_sums = {}
        tier_counts = {}

        for dyn in dynamics_list:
            expert_data = dyn.get('expert_gradients', {}).get(expert_key, {})
            for metric_key, value in expert_data.items():
                if metric_key not in tier_sums:
                    tier_sums[metric_key] = 0
                    tier_counts[metric_key] = 0
                tier_sums[metric_key] += value
                tier_counts[metric_key] += 1

        expert_gradients[expert_key] = {
            k: tier_sums[k] / tier_counts[k]
            for k in tier_sums.keys()
        }

    # Average divergence scores
    for dyn in dynamics_list:
        for div_key, value in dyn.get('divergence_scores', {}).items():
            if div_key not in divergence_scores:
                divergence_scores[div_key] = []
            divergence_scores[div_key].append(value)

    divergence_scores = {
        k: sum(v) / len(v)
        for k, v in divergence_scores.items()
    }

    return {
        'expert_gradients': expert_gradients,
        'divergence_scores': divergence_scores
    }
```

---

## Database Schema

Create `build/runs/{hash}/dynamics.db` with this schema:

```sql
CREATE TABLE dynamics (
    step INTEGER PRIMARY KEY,
    expert_0_top_norm REAL,
    expert_0_bottom_norm REAL,
    expert_0_middle_norm REAL,
    expert_0_clean_norm REAL,
    expert_1_top_norm REAL,
    expert_1_bottom_norm REAL,
    expert_1_middle_norm REAL,
    expert_1_perturbed_norm REAL,
    expert_1_divergence REAL,
    expert_2_top_norm REAL,
    expert_2_bottom_norm REAL,
    expert_2_middle_norm REAL,
    expert_2_perturbed_norm REAL,
    expert_2_divergence REAL
    -- ... extend for more experts
);
```

### Logger Integration

If you're using the existing metrics logger, extend it:

```python
class DynamicsLogger:
    """Logs gradient dynamics to SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        """Create dynamics table if it doesn't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create table (will need to be dynamic based on num_experts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamics (
                step INTEGER PRIMARY KEY,
                expert_0_top_norm REAL,
                expert_0_bottom_norm REAL,
                expert_0_middle_norm REAL,
                expert_1_top_norm REAL,
                expert_1_bottom_norm REAL,
                expert_1_middle_norm REAL,
                expert_1_divergence REAL
            )
        """)

        conn.commit()
        conn.close()

    def log(self, step: int, dynamics: Dict):
        """Log gradient dynamics for a training step."""
        if not dynamics or 'expert_gradients' not in dynamics:
            return

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Build INSERT statement dynamically
        columns = ['step']
        values = [step]

        expert_grads = dynamics['expert_gradients']
        for expert_key in sorted(expert_grads.keys()):
            expert_data = expert_grads[expert_key]
            for metric_key, value in expert_data.items():
                col_name = f"{expert_key}_{metric_key}"
                columns.append(col_name)
                values.append(value)

        # Add divergence scores
        for div_key, value in dynamics.get('divergence_scores', {}).items():
            columns.append(div_key)
            values.append(value)

        # Insert or replace
        placeholders = ','.join(['?'] * len(columns))
        query = f"INSERT OR REPLACE INTO dynamics ({','.join(columns)}) VALUES ({placeholders})"

        cursor.execute(query, values)
        conn.commit()
        conn.close()
```

---

## What You'll See in the Dynamics Tab

### The Chart

**Y-axis** (log scale): Gradient norm magnitude
**X-axis**: Training step
**Lines**:
- **Blue shades**: Expert 0 (clean) gradients
  - Light blue: Top 5% weights
  - Medium blue: Bottom 5% weights
  - Dark blue: Middle 90% weights
- **Red/Orange shades**: Expert 1+ (perturbed) gradients
  - Light red: Top 5% weights
  - Medium red: Bottom 5% weights
  - Dark red: Middle 90% weights

### Key Questions to Answer

**1. Are bottom weights waking up?**
- Compare Expert 0 bottom (blue) vs Expert 1 bottom (red)
- If Expert 1 bottom >> Expert 0 bottom → **YES, aggressive perturbations are activating dormant pathways!**
- If they're similar → scale=1.0 might not be enough

**2. Is there genuine divergence?**
- Check the divergence score in the header
- Higher divergence = perturbed experts are learning differently
- Lower divergence = might need higher perturbation scale

**3. Do perturbations cascade through training?**
- Watch how gradient norms evolve over time
- Do perturbed experts maintain different trajectories, or converge to clean expert?
- Convergence = perturbations aren't strong enough to force genuine diversity

---

## Troubleshooting

### "No data" message in Dynamics tab

**Cause**: Gradient logging not enabled in training loop
**Fix**: Add the logging code from Step 2 above

### Database errors

**Cause**: dynamics.db doesn't exist or has wrong schema
**Fix**: Ensure `DynamicsLogger._ensure_schema()` is called before logging

### Missing experts in chart

**Cause**: Only Expert 0 and 1 logged, but model has more experts
**Fix**: Verify all experts are being iterated in `log_gradient_dynamics()`

### All gradient norms are zero

**Cause**: Logging called before `backward()` or after `optimizer.zero_grad()`
**Fix**: Ensure logging happens **after backward(), before step()**

---

## Performance Considerations

**Gradient logging overhead**: ~5-10% slowdown per logged step

**Recommendations**:
- Log every 10-100 steps (not every step)
- Only log when needed (not in production runs)
- Disable after validation (once you know perturbations work)

**Data volume**:
- ~50 bytes per step per expert
- 1000 steps × 2 experts = ~100KB (negligible)

---

## Expected Results

### If Dual-Sided Perturbation Works

You should see:
1. **Bottom gradient norms higher for perturbed experts** (awakened pathways)
2. **Divergence score > 0.05** (meaningful difference in learning)
3. **Sustained divergence over training** (doesn't collapse to consensus)

### If It Doesn't Work

You might see:
1. Bottom norms stay near zero for all experts (dormant)
2. Divergence score < 0.01 (experts converging to similar solutions)
3. Divergence decreases over time (collapsing to consensus)

**If it doesn't work**: Try increasing `perturbation_scale` to 2.0 or higher, or adjusting `sparsity`.

---

## Next Steps After POC

Once validated, could extend to:
- Per-layer gradient tracking (see which layers benefit most)
- Gradient distribution histograms (not just norms)
- Heatmap view (layer × expert × tier)
- Compare multiple agents side-by-side

But for now: **Just validate that aggressive dual-sided perturbations force genuinely different learning dynamics.** 🎯

---

## The Hypothesis

From "The Blind Watchmaker":
> "Train on consensus, you manifest the lowest common denominator."

**Static perturbations** force exploration outside consensus.
**Dual-sided targeting** exposes both numerical extremes.
**Aggressive scale (1.0)** ensures perturbations are meaningful.

If this works, you should see **measurably different gradient dynamics** between clean and perturbed experts, particularly in the bottom weights that were previously dormant.

**The Dynamics tab is your window into whether the theory holds.** 🔬
