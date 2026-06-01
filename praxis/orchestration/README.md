# praxis.orchestration

The distributed remote-expert pooling layer. A pool of many tiny experts (in
process today; Ray, Hivemind, or browser peers over GUN tomorrow) behind one
mixing layer. It is the Python side of the in-browser swarm - the JS twin is
`praxis/web/src/js/swarm.js`, and the vision is `next/world_models.md`.

We are building an in-house alternative to Hivemind; the `RemoteExpert` interface
is the single transport seam where a Hivemind/Ray/GUN backend plugs in.

## Why it works at all

Each expert is trained with a **local, layer-wise (Mono-Forward) loss**: it owns
its weights plus a projection `M_i` to the label space, and updates only itself.
No gradient ever crosses the network. That single property is what makes the pool
tractable with thousands of extremely tiny peers and real network latency:

- **Training is non-blocking.** `ExpertPool.train_step` dispatches the batch to
  every live expert and collects whatever finishes within a timeout; a slow or
  dead peer never blocks the others, because there is nothing to synchronize.
- **Inference is stochastic.** `ExpertPool.infer` samples a subset of experts,
  runs their forwards in parallel, and combines them with a mixing strategy. The
  sampling is the exploration; the mixer is how peers compose.

## Pieces

| File | What |
|------|------|
| `base.py` | `RemoteExpert` (the transport-agnostic ABC) + `LocalExpert` (in-process reference: one Mono-Forward block + projection + Adam). |
| `pool.py` | `ExpertPool`: membership, live capacity, non-blocking `train_step`, stochastic `infer`. |
| `mixing.py` | `MIXING_REGISTRY`: how outputs combine. |
| `status.py` | Process-global the pool publishes capacity to; the dashboards read it. |

## Selecting a profile (`--orchestration-type`)

One flag picks a named profile from `ORCHESTRATION_REGISTRY`; each bundles
whether to spawn the backend sidecar, the starter expert count, and the mixing
strategy - so new variants are registry entries, never new CLI knobs (mirrors
`--memory-type`). `none` (default) disables the pool.

| profile | behavior |
|---------|----------|
| `none` | Disabled. No remote-expert pool. |
| `swarm` | Backend Node sidecar of 4 tiny experts; CALM-style expert vote. |
| `swarm_mean` | Sidecar of 4 experts; plain mean of expert outputs. |
| `swarm_wave` | Sidecar of 4 experts; standing-wave mix over peers. |
| `frontend_only` | No backend process; mix only browser-joined experts (vote). |

## Mixing strategies

The combiner each profile names, from `MIXING_REGISTRY`:

| name | behavior |
|------|----------|
| `mean` | Pool average (consensus / pure bias). Robust, deterministic. |
| `vote` | Average per-expert distributions, return log-probs (CALM expert vote). |
| `sample` / `sample_quarter` | Keep a random half / quarter of experts, then average - stochastic excursions off the mean. |
| `wave` / `wave_high` | A **standing wave over the expert index**: expert `i` gets weight `1 + cos(2π·freq·i/E + φ)`, normalized. Peers compose by constructive / destructive interference across the peer axis, not a flat mean. Reduces to identity for a single expert. |

The standing-wave mixer is the harmonic idea (the project's recurring theme)
applied to the *peer* axis: with many tiny experts, the interference pattern over
which peers reinforce vs cancel carries information a plain average discards.

## Live capacity

`ExpertPool.capacity()` returns `{experts_total, experts_alive, total_rank,
passes, steps, mixing}` and publishes it to `status`. The terminal callback reads
`status.info_line()` into its `info_dict`, so the line - e.g. `8/10 experts, rank
112` - shows up automatically on **both** the CLI dashboard and the web Terminal
tab's System panel (both are driven by the same `info_dict`). When no pool is
active the line is omitted.

`total_rank` is the sum of every alive expert's representational width - the
quantity a buyer rents (the rank-priced connection in `next/world_models.md`).

## Quick use

```python
from torch import nn
from praxis.orchestration import build_pool, LocalExpert

def make(uid):
    block = nn.Sequential(nn.Linear(14, 14), nn.SiLU())
    return LocalExpert(uid, block, hidden_size=14, vocab_size=16)

pool = build_pool([make(f"e{i}") for i in range(8)], mixing="wave", sample_size=3)
pool.train_step(acts, labels)   # non-blocking local updates
out = pool.infer(acts)          # sample 3, mix by standing wave
```

## Status / next

Done: the pool, experts, mixing registry, capacity reporting wired to both
dashboards, `--orchestration-type` profile registry, Node sidecar + Python
proxy, the `ExpertPoolCallback` that drives it during training, tests
(`tests/test_orchestration.py`). Wired into `experiments/calm-c.yml`
(`orchestration_type: swarm`).

Next (transport): a `RemoteExpert` subclass that forwards `forward`/`train_step`
over a wire - first to the browser swarm (GUN/websocket to `swarm.js`), reusing
the existing Hivemind integration (`integrations/hivemind/`) as the discovery
reference. The pool math does not change; only the expert's transport does.

Next (population regime): at thousands of experts the pool becomes a *population*
- particle-swarm / ant-colony controllers, wave-phase specialization, and a
binned low-count vote approximation that folds back into the model as structured
variance. See the "Swarm dynamics" item in `next/roadmap.md`.
