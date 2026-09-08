# Where decode time actually goes on the byte-latent stack

> Status: **measured** (2026-09-08), on the live `abstractinator-r` checkpoint
> `b7e21ca5b` at batch 5439. RTX 5060 Ti, fp32, eager, batch 1, greedy,
> `mtp_type: per_depth`. Written because the obvious answer to "generation is
> slow" - wire up a KV cache - is the wrong one here, and the reason is a
> number rather than an argument.

## The one measurement that decides everything

One model forward, against byte length `T` (the trunk sees `1 + ceil(T/8)`
patches under `patching_mode: static, patch_size: 8`):

| T | patches | forward |
|---:|---:|---:|
| 8 | 2 | 32.2 ms |
| 64 | 9 | 32.9 ms |
| 256 | 33 | 37.5 ms |
| 512 | 65 | 42.0 ms |
| 1024 | 129 | 41.6 ms |
| 2048 | 257 | 49.6 ms |
| 4088 | 512 | 60.1 ms |

**~32 ms of every forward does not depend on the sequence length at all.** At
the context the chat UI actually runs (`infer_context: 1024`) the whole
length-dependent part is 10 ms of a 42 ms forward.

A KV cache removes exactly the length-dependent part. So its ceiling is
**1.3x at 1024 bytes** and 1.9x at the 4096-byte positional cap - and that is
the ceiling for a *perfect, free* cache, before the per-step bookkeeping a real
one adds. This is not a byte-latent problem or a patching problem. A 6.7M-param
model at batch 1 issues on the order of 250 module calls and well over a
thousand kernels per forward, each a few microseconds of work behind tens of
microseconds of dispatch. Shrinking the tensors does not reduce the count.

## What the forward is spending it on

Per-stage, T=1024, hook-instrumented (60 ms total under the hooks' own sync):

| stage | T=1024 | T=8 |
|---|---:|---:|
| trunk | 40.4 | 33.2 |
| - Titans memory (`mag_energy_stitch`) | 22.2 | 10.8 |
| - Kaleidoscope attention | 5.3 | 6.3 |
| - PEER ffn | 4.6 | 5.2 |
| head (prismatic9) | 5.4 | 4.4 |
| byte embeddings (multihash) | 3.7 | 3.7 |
| VQ bottleneck | 2.3 | 2.2 |
| local conv decoder | 1.1 | 1.1 |
| local conv encoder | 1.1 | 1.3 |

Only the memory module has a length-dependent cost worth the name, and even it
is half fixed. Everything else is flat: the byte-level convolutions, the
embeddings, the head, the quantizer all cost the same at 8 bytes as at 1024,
because none of them is doing enough arithmetic to be limited by arithmetic.

## Is a cache even *possible* here? Yes, and that was never the blocker

`prepare_inputs_for_generation` refuses to cache whenever `self.encoder` is set,
and the comment there is right that the reason is UNITS (`past_length()` counts
patches, `input_ids` counts bytes), not prefix instability. The stack really is
causal under append:

- static patching puts byte `t` in patch `1 + floor((t-1)/8)`, and no closed
  patch ever moves;
- the local conv encoder/decoder are dilated causal convolutions with a
  15-byte receptive field, so a window is all they need;
- `decoder_patch_ids_from_lengths` drops patch 0, so no byte reads the open
  patch's latent.

Measured directly on the trained checkpoint - run the model on a prefix, append
one byte, compare the settled positions, 56 lengths spanning several patch-count
crossings:

- trunk hidden states drift 1e-3 to 3e-2 absolute, which is 5e-6 to 1.3e-4
  RELATIVE to a hidden scale of ~233;
- byte logits on settled positions drift <= 1.3e-5, with **zero argmax flips**.

So a patch-level cache would produce the same text. It would not be bit-exact,
and it is worth knowing why: `KaleidoscopeAttention` reads its frozen mirrors in
RATIO coordinates and resamples them to the live `[T, T]` every forward, so
growing the trunk by one patch restretches the whole geometry; and the memory's
chunk grid shifts. On a matched random-init build, ablating `memory_type` drops
the drift to float noise even across patch-count changes, which points at the
memory rather than the mirrors as the larger contributor. Neither is big enough
to change a byte.

The design, if it is ever wanted: cache the closed patch latents `z[0..P)` plus
a 15-byte tail of local-encoder output; run the local encoder over `tail + new`;
downsample/project/quantize only patches that CLOSED this step; thread the
`NeuralMemState` (already a `NamedTuple` designed to cross decode steps, already
returned through `current_state`); give Kaleidoscope a V-cache and evaluate only
the new query ROWS of the mirrors (`grid_sample` on a `[1, n_new, T, 2]` grid
instead of `[T, T, 2]`); local decoder over the byte tail; head on the new
positions with `_bind_head_cache` supplying the offset. Every piece is
tractable. The reason not to build it first is the table at the top.

## What the levers actually are

All measured on a 1000-byte prompt writing 64 bytes greedily. Baseline at the
start of this pass: 3.01 s, **21.3 bytes/s**, 41 forwards, 1.56 bytes per
forward. Every number below is byte-identical output.

1. **Fewer ops per forward.** The patcher's invariant check was a Python double
   loop comparing tensor elements one at a time, which is one device-to-host
   sync per element. See below - this one is not really a decode finding.
2. **Stop paying for drafts that never land.** 25.2 bytes/s. The width policy
   added its growth margin on EVERY step, and `_accept_ema` starts at 1.0, so a
   model accepting runs of 1 drafted 2 forever. Swept at fixed widths 1/2/3/5:
   2.53/2.80/3.13/3.65 s, a flat ~0.28 s per extra draft, with bytes-per-forward
   pinned at 1.56 at every width. The margin is now a probe on a cadence, which
   is what its own comment always said it was for.
3. **Decode-length bucketing + compiled memory bodies.** 32.5 bytes/s, i.e.
   **1.52x end to end** (34.4 through the real `ModelBackend`). Bucketing is
   the enabler rather than the win - see `praxis/generation/bucketing.py`.
   Still opt-in, behind BOTH the `compile_decode_memory` environment feature
   and the trainer's `no_compile`, which the abstractinator line sets true.
   The reason it did not just get switched on: the memory-growth failure that
   disabled it was measured over a real 14h run, and a 105-step synthetic
   growing-context replay could not reproduce it (child memory flat at 686 MB
   bucketed, 691 MB unbucketed - Dynamo's recompile limit and automatic dynamic
   shapes cut in first). An experiment that cannot reproduce the problem cannot
   certify the fix.
4. **More bytes per forward.** 1.56 today. Every accepted draft byte rides a
   forward that was going to happen anyway, so this axis is worth more than
   anything on the cost axis - and it is bought by TRAINING the MTP heads, not
   by decode-side code.
5. **The cache.** 1.3x at the chat context, for a refactor touching
   Kaleidoscope, the memory, KL halting and the SMEAR routers. Worth doing when
   contexts get long or when the fixed cost has already been cut, not before.

## The patcher check, which turned out not to be a decode problem

`Patcher._check_non_zero_after_zero` walked `patch_lengths` with a nested
Python loop, and each `if val == 0` on a 0-d CUDA tensor is a device-to-host
sync. Timed on `patcher.patch` itself (RTX 5060 Ti):

| shape | loop | vectorized |
|---|---:|---:|
| B=1, T=1024 (decode) | 12.8 ms | 0.48 ms |
| B=64, T=512 | 368 ms | 0.56 ms |
| B=64, T=1024 | 734 ms | 0.67 ms |
| B=64, T=2048 | 1559 ms | 0.57 ms |

The cost is `B * patches * ~100 us`, so at decode (batch 1) it is a few
milliseconds and at a TRAINING batch it is hundreds. The vectorized version is
one exclusive prefix sum and a single `.item()` on the reduction - same
predicate, one sync.

How much of the live run's ~2 s `avg_step_time` this was is not established
here: `micro_rows` is governed by GNS and the length by the probe curriculum,
and neither is exposed as a metric, so the live shape could not be read off the
run. It is worth watching `avg_step_time` across the next restart. Confirming
it directly wants `py-spy dump` on the training process, which needs sudo.

## The thing that would actually be 5x, and why it still does not work

The fixed 32 ms is dispatch, and the only tool that removes dispatch is a
compiled graph. `decode_backend.eval_mode` records the attempt: compiling the
decoder made a 136-byte forward 21288 ms against 202 ms eager, 143 frames, no
sign of settling. That was measured on UNBUCKETED decode, where the sequence
length alone mints a fresh graph per step, so the obvious hypothesis was that
the shape axis - not `current_depth` - was what never settled.

**Re-run with bucketing on (2026-09-08): it still does not settle.** Twenty
minutes into a single 32-byte turn, Dynamo was at frame 34 and still climbing,
with `Graph break from Tensor.item()` reported throughout. So the shape axis was
not the blocker. Every graph break spawns resume frames, and resume frames
multiply against the depth loop.

The `.item()` sites that are actually live during an eval decode under
`model.decoder`, as candidates to chase in order:

- `praxis/memory/surfacings.py:344,359` - the stitch path reads `gid[-1]` and
  `pos.max()` as Python ints;
- `praxis/blocks/transformer.py:175` - `_is_zero_tensor`, a `.max().item()` on
  the router weights every block;
- `praxis/halting/kl.py:100,194`;
- `praxis/layers/local.py:156-170` - the early-exit signal.

Kaleidoscope's dozen `.item()`s are already behind
`not self.training or torch.compiler.is_compiling()`, which is the pattern the
rest should follow. `project_abstractinator_compile` lists the same class of
site as its known recompile drivers, so this is one job, not four. It is a
prerequisite for every remaining idea on this page - a compiled trunk, CUDA
graphs, and a cache whose decode step has one fixed shape.
