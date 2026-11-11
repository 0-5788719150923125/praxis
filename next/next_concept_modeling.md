# Next-Concept Modeling: coarse-to-fine generation on the byte-latent line

> Status: **second run training** (read at step 10k on 2026-11-07: best bound
> still at the level-0 cut; generation's replacement characters fixed, see
> "Second run"). The first run
> (9f8a6751d) collapsed onto the finest level and is void; the decoder is now
> fenced to its own patch (see "First run"). `byte-latent-j` = `-i` +
> `encoder_type: byte_latent_conv_coarse_to_fine`. Paper: Kumbong, Khare et al.
> 2026, "NCM: Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text
> Generation", arXiv:2610.14162 (read in full from the TeX source; no code
> released). This is the "stage 3" that [text_diffusion.md](text_diffusion.md)
> planned and never built: the residual-VQ depth ladder as the generation
> schedule.

**The one thing to take from this note:** the authors present NCM as an
alternative to diffusion, not a kind of it. There is no corruption process
to reverse. Text is described by a fixed hierarchy of discrete codes, coarse
to fine, and a transformer predicts each level of that hierarchy once, in
parallel, from the reconstruction of the levels above it. Every training row
trains every level, against a fixed target, which is the likely reason it
needs a fraction of the training a masked-diffusion model does.

## The method, as the paper has it

Two stages.

1. **A multiscale residual VQ-VAE (the tokenizer).** An encoder maps `L`
   tokens to `L` latents (the paper's encoder is just the embedding table).
   Level `k` pools the current residual to `l_k` vectors, quantizes each
   against that level's codebook, maps the codes back to length `L` and
   adds them to the running reconstruction. `l_1 < ... < l_K = L`. The decoder
   (1-4 transformer blocks) maps the reconstruction back to token
   distributions. Trained with nested dropout: half the time the ladder is cut
   at a random level and the decoder must reconstruct from what is left, so the
   coarse levels have to carry meaning. EMA codebooks, decay 0.95-0.99, 4096 or
   8192 codes per level.
2. **The generator.** A transformer whose input for level `k` is the
   reconstruction of levels `< k`, resized to `l_k` positions and projected. All
   levels are trained in one forward pass with block-diagonal attention: each
   level's block attends only to itself, because its input already summarizes
   the earlier levels. That pattern won their ablation (Gen. PPL 15.4 against
   19.6 for block-causal). Cross-entropy per code, weighted per code
   (`alpha = 0`, which is what the ELBO weights; `alpha <= 1` did not move
   quality). **Conditioning corruption**: some input codes are replaced by
   random codes from the same codebook, at a rate that falls across levels,
   averaging 0.5; the targets stay clean. It fixes the train/generate gap (at
   generation the model conditions on its own sampled codes) and helps a lot.
   Prompts are a prefix: prompt positions attend to each other, and each target
   block attends to the prompt and itself.

Generation: `K` sequential steps, one per level, each sampling every code of
the level at once. The decoder then argmaxes every token in parallel.

Their numbers: better Gen. PPL than the strongest diffusion baseline on LM1B
at matched scale, about 21x faster; competitive on OWT at about 10x;
XSum ROUGE-L matched at about 31x; better Sudoku than AR and diffusion
baselines. About 4.9x fewer training positions than the diffusion baselines on
LM1B. Their caveats: no wall-clock comparison against AR, around 100M scale
on BPE tokens, a fixed ladder for every input, occasional artifacts where
coarse levels hand over to fine ones. Their own future work asks for exactly
what we need: end-to-end training.

## The mapping onto byte-latent-i

| NCM | byte-latent-j |
|---|---|
| tokenizer encoder (embedding table) | the conv local encoder, max-pooled to space patches, RMS-normalized |
| one sequence of `L` tokens | chunks of 32 patches (~230 bytes) |
| level schedule (six ablated) | doubling: 1, 2, 4, 8, 16, 32 codes per chunk (`K = 6`) |
| `Down_k`, `R_k` (unspecified) | area pooling over equal spans; nearest spreading back |
| codebooks (4096 or 8192 per level) | 4096 at the last level, halving per coarser level (128 at level 0) |
| tokenizer decoder (parallel transformer) | the conv local decoder fenced to one patch: its reconstructed latent, its slot in the chunk, and its own bytes so far |
| generator (transformer) | the Kuramoto trunk |
| prompt prefix | every earlier chunk, as a causal context stream of clean latents |
| two stages, tokenizer frozen | one run: the trunk reads detached latents, codebooks learn by EMA |

**Why chunks of patches, not bytes.** A 2048-byte row has ~290 space patches.
A ladder over bytes would need ~11 levels and around 4k trunk positions; over
patches in chunks of 32 it is 320 context positions plus 10 x 63 code
positions (~950), and block-diagonal attention keeps most of that cheap.

**Why chunks at all.** A single ladder per row could only generate a whole row
from nothing. Chunks make it continue a prompt: the prompt's patches are the
context, each new chunk is generated in 6 trunk passes, its bytes are spelled,
and it joins the context. The KV-cache argument survives too: context
positions never attend to blocks.

**The trunk sequence** is `[context: n*32 patch latents][chunk 0: levels 0-5]
[chunk 1: levels 0-5] ...`. Context positions are causal within their
document, as in `-i`. A block reads context from earlier chunks (gated to the
document its chunk opens in) and its own positions, nothing else. Positions
are in patches: a context patch at its index, a code at the center of the span
it covers, so a finest-level code sits exactly where its patch does. This is
published to the attention as a layout (`praxis/attention/layout.py`) for the
duration of one forward and replaces its causal and document masks.

**The decoder alignment** is the one real change to the byte-latent path. In
`-i` the bytes of patch `j` read the trunk output of patch `j - 1` (the trunk
is causal; patch `j` has not been seen yet). Here they read patch `j`'s own
reconstructed latent. That is what a tokenizer decoder is, and it is legal
because the space patcher's boundaries are causal: whether byte `t + 1` opens
a patch is decided by byte `t`, so at generation time the decoder always knows
which latent to read next.

## The bound

The codes are a deterministic function of the bytes, so for any byte string

    -log p(x) <= -log p_trunk(codes(x)) - log p_decoder(x | codes(x))

The right side is the model's validation loss once nested dropout and
corruption are off. `val_byte_nll_bits` on this run is exactly that (decoder
bits plus code bits, over the same scored bytes), so it is an upper bound on
the model's true bits per byte and can be read against `-i` in one direction:
if it is lower, `-j` really is better. Higher is inconclusive, because the
codes can carry detail the decoder never needed and the bound pays for it.
`val_codec_bpb` is the decoder alone. Small overheads against `-i`: the code
for the row's first patch is paid although `-i` takes byte 0 as given, and
blocks may read context from the previous document when a chunk straddles a
seam (gated by the chunk's first patch).

## Joint training, and why it is safe here

The user's rule for this line is one training run, never a frozen stage. The
paper's tokenizer is frozen while the generator trains; the same separation
comes for free inside one run:

- The trunk's context stream is `z.detach()`, and its level blocks are built
  from ladder reconstructions computed under `no_grad`.
- Codebooks learn only by EMA (`requires_grad` off).
- So the code loss trains the trunk, the level projections and the code heads,
  nothing of the tokenizer; the reconstruction loss never reaches the trunk.
  Both directions are asserted in `tests/encoders/test_coarse_to_fine.py`.

Without the stop-gradient the trunk could push the encoder toward trivially
predictable codes, the usual prior/VQ collapse.

What joint training does cost: code identities move while the trunk learns
them. EMA moves codes slowly, and dead-code resets replace a code outright. So
codes are scored by distance to the live codebook (`code_logits`) rather than
by a per-index weight: a moved or reset code is scored by its new vector.

## Deviations from the paper, all deliberate

- Bytes and space patches, not BPE tokens.
- The decoder is the conv decoder, spelling a patch's bytes one at a time
  after each chunk's codes; the paper's decoder is parallel. It is fenced to
  the patch: every conv tap stops at the patch's first byte, the position
  whose own byte ends the previous patch reads a start marker, and it has its
  own byte table (the shared one hashes n-grams across boundaries). So, as in
  the paper, nothing about earlier text reaches it except through the codes.
  Until 2026-11-05 it read the causal byte path across patches, which
  collapsed the hierarchy (see "First run").
- The decoder adds a learned embedding of the patch's slot in its chunk to the
  latent, the paper decoder's positions: at a coarse cut many patches share
  one vector and the slot is what tells them apart.
- Codebooks halve per coarser level (4096 at the last, 128 at level 0), so
  each code at every level sees about as many vectors (level `k` sees `2**k`
  per chunk). With 4096 everywhere, level 0 saw ~40 vectors a forward and
  replaced each code about every 240 steps.
- Latents are RMS-normalized before the ladder, so EMA codebooks are not
  chasing a scale drift.
- `merge: normalized` in the profile. Under BLT's additive merge the byte path
  outgrew the latent (RMS 6.8 vs 1.0, content ratio 0.03 at 200 steps); with
  both normalized the content ratio was 0.47 and both reconstruction and code
  bits were lower. Under `add` the decoder would have learned to ignore the
  codes, the same failure this line had with the trunk.
- Corruption endpoints: linear from 0.75 at level 0 to 0.25 at the last
  conditioning level (mean 0.5). The paper gives the mean and "shifted linear
  decreasing", not the endpoints.
- Codebooks learn only on steps with gradient. `initialize_lazy_modules` runs a
  train-mode, no-grad dummy forward of all-ones bytes; seeding from it put
  every code on one vector and collapsed every level at init (found in the
  smoke run). `MultiStageResidualVQ._seed` has the same exposure.
- A prompt that ends mid-word is continued inside its open patch: the decoder
  reads that patch's bytes so far, and the trunk's context stops at the last
  complete patch. Until 2026-11-07, generation forced a fresh patch at the end
  of the prompt (see "Second run").

## What to read on the run

In order:

1. **Code Gain over Marginal, by level** (Coarse to Fine section): code loss
   below a predictor that only knows how often each code is used. Zero means
   the trunk learned the marginal and nothing else, the degenerate solution the
   diffusion run taught us to check first. Early on it reads high at levels 0-1
   because the EMA usage still remembers the seeding; trust it once
   utilization settles.
2. **Codebook Utilization / Dead Codes / Code Resets by level.** Level 0 sees
   only ~40 vectors per forward (one per chunk), so coarse levels are where
   codebooks collapse or churn. Resets start after ~1000 forwards (~125
   optimizer steps). A level stuck near zero utilization means that level
   says nothing and the levels below it do all the work.
3. **Bound by Ladder Cut / Reconstruction by Ladder Cut** (validation): the
   decoder reading levels 0..k only, and that plus those levels' code bits.
   Reconstruction should fall at every level; the deepest cut should give the
   lowest bound. A coarse cut with the best bound means the finer levels cost
   more than they carry (the first run's failure).
4. **`val_codec_bpb`**: the decoder alone from the full ladder. With the
   decoder fenced it starts high (it has to spell from codes) and is the
   tokenizer's quality.
5. **`val_byte_nll_bits`** (the bound) against `-i` at matched steps.
6. **Samples.** 6 trunk passes per chunk, then the bytes.

Off on this run, because the decoder's logits read their target's own code:
`val_copy_gain`, `val_trunk_swap`, `val_context_gain` (`emits_next_token_likelihood`
on the encoder). `val_loss` is the reconstruction plus the code nats per byte
(the bound in nats, as trained less commitment); before 2026-11-02 it was the
reconstruction alone, below any next-byte loss by construction. `merge_*`
measures the latent stream, not the trunk; `depth/tilt` reads the whole layout.

## Failure modes, named before the run

- **Coarse levels say nothing.** Chunk means of patch latents can be nearly
  constant, so level 0 collapses to one code. Seen in the first 60 smoke steps
  under the additive merge; the normalized merge kept level 0 informative.
  If it recurs over a long run: center the latents before pooling, or seed
  coarse codebooks from more than one batch.
- **The bound sits above `-i` because the codes over-describe.** No rate
  penalty is on the codes, so the finest level can encode detail the decoder
  does not need, and the bound pays for it. Watch `code_bits_per_byte` against
  `val_codec_bpb`: code bits rising while reconstruction is already near zero
  is the signature.
- **Code churn.** A steady slope in resets at a level means its codes keep
  changing meaning under the trunk.
- **Coarse-to-fine artifacts.** The paper sees them; MBR decoding or CFG are
  the inference-side fixes if samples need it.

## Cost

Trunk sequence ~3x the patch count: ~1,050 positions for a typical
2048-byte row of code (~350 patches), up to ~1,700 for symbol-dense rows
(~580 patches) and ~3,300 for the 1 x 4096 multiplier tier.

The layout path runs on SDPA's memory-efficient kernel with the layout's
boolean mask, not FlexAttention. The first real launch on eager flex hit a
CUDA OOM at step 2: eager flex materializes every score matrix (batch x 16
heads x T^2) in the backward, and at ~1,900 positions that is gigabytes per
layer. On SDPA, measured on the 5060 Ti, batch 4 x 2048, no compile: 0.19 s
per micro-batch forward+backward on typical rows (`-i`: 0.22 s on eager
flex), peak 2.7 GiB on the densest rows found, 1.6 GiB at 1 x 4096. The
ladder costs ~17 ms. Parameters: 8.09M trainable (+0.46M over `-i`: the level
projections and code heads) plus 6.3M of EMA codebooks (6 x 4096 x 256) that
take no gradient. Since the 2026-11-05 fixes: 8.17M trainable (the decoder's
byte table and chunk slots) and 2.1M of codebooks (8,064 codes); speed not
re-measured on the GPU (the fenced conv is three matmuls per layer, the same
FLOPs, and generation no longer reruns the local encoder per byte).

## What was built

- `praxis/encoders/quantization/multiscale.py`: `MultiScaleResidualVQ`, the
  doubling ladder over chunks; per-level `VectorQuantizer`s (EMA, first-batch
  seeding, dead-code resets, telemetry); `learn=False` touches no codebook state.
- `praxis/encoders/byte_latent/coarse_to_fine.py`: `CoarseToFineEncoder`, the
  ladder wiring, nested dropout, corruption, the trunk layout, the code loss,
  the decoder alignment, metrics, and the decoding method (`coarse_to_fine_decoding`).
- `praxis/attention/layout.py`: `BlockLayout`, the published mask (a dense
  boolean mask, built once per forward) and positions.
- `SelfAttention.set_layout` and `_layout_attention` (SDPA with the layout
  mask; ghostmax via the zero ghost key; Kuramoto's ghost mass from a no-grad
  log-sum-exp); explicit `positions` on RoPE and `rope_learned_rates`.
- `PraxisModel` publishes an encoder's `trunk_layout` around the trunk call
  and asks the encoder to `check_trunk`.
- The trainer adds the encoder's `code_nats()` to the validation bits
  (`val_byte_nll_bits` becomes the bound, `val_codec_bpb` the decoder alone),
  and logs every cut from `ladder_cuts()` (`val_cut_bound_l{k}`,
  `val_cut_recon_l{k}`).
- `ConvBlock` takes `segment_positions` (a fenced causal conv), used by the
  decoder's `_spell`; `MultiScaleResidualVQ.sizes` per level.
- `BaseEncoder.emits_next_token_likelihood`; the copy, trunk-swap and
  context-gain probes skip encoders where it is False.
- Tests: `tests/encoders/test_coarse_to_fine.py` (gradient routing both ways,
  no chunk reads a later one, deterministic eval, codebooks learn only with
  gradient, generation contract incl. `return_dict_in_generate`, probes off,
  refusals), ladder tests in `tests/encoders/test_quantization.py`, layout tests
  in `tests/attention/test_layout.py`; the causality sweep in
  `tests/test_modeling.py` treats this encoder like CALM (earlier = an earlier
  chunk).

## First run (9f8a6751d), read at step 22k

Wiring checked on the checkpoint, CPU: the generation path reproduces the
training forward's patch latents exactly and its code logits and next-byte
logits to 6e-5; each target byte reads its own patch's code; from the true
codes the decoder spells close to the source. Samples are word salad anyway,
because **the hierarchy collapsed onto the finest level**:

| decoder reads | bits/byte (12 held-out rows) |
|---|---|
| levels 0..0 (1 code/chunk) | 2.83 |
| levels 0..3 (15 codes) | 2.53 |
| levels 0..4 (31 codes) | 2.13 |
| levels 0..5 (63 codes) | 1.27 |
| unquantized latent | 0.80 |

Levels 1-3 buy 0.3 bits/byte; level 5 alone buys 0.85. With the true codes
for levels 0-4 and level 5 sampled, the text is unrelated to the source. So a
chunk is generated by choosing ~32 word-sized codes in one parallel step, none
seeing the others. The coarse codes are not concepts either: consecutive
chunks of one file share a level-0 code 3% of the time, the trunk predicts
levels 0-3 worse than their usage frequencies, and level 0 resets each of
its 4096 codes about every 240 steps (~40 vectors per forward). The bound is
best cut at level 0 (~2.9 bits/byte on these rows); every finer level costs
more code bits than it saves.

Why here and not in the paper: their decoder rebuilds tokens from the codes
alone, so when nested dropout cuts the ladder, the coarse codes are all it
has, and they are forced to carry meaning. Ours keeps the causal byte path
(~15 bytes of reach plus hashed n-grams), which takes over at a cut, so
nothing asks the coarse levels to carry anything. The deviation list called
the conv decoder a speed difference; it is the cause.

**Fixed 2026-11-05, same experiment name** (that run is void; its checkpoints
do not load): the decoder is fenced to its own patch, gets the patch's slot in
its chunk, and codebooks halve per coarser level (see Deviations), and every
cut's bound is logged at validation. What to expect: reconstruction from the
full ladder starts far worse than the first run's, since the decoder can no
longer read the previous ~15 bytes, and the cut charts should fan out by
level instead of the deepest cut being worst.

## Second run (same hash), read at step 5k

Neither prediction held. Reconstruction from the full ladder is 1.43
bits/byte, not far worse. The deepest cut is still the worst bound:

| cut | recon | bound (recon + code bits) |
|---|---|---|
| levels 0..0 | 2.94 | 2.97 |
| levels 0..1 | 2.90 | 3.01 |
| levels 0..2 | 2.83 | 3.13 |
| levels 0..3 | 2.68 | 3.40 |
| levels 0..4 | 2.34 | 3.96 |
| levels 0..5 | 1.43 | 4.78 |

Every bound is still falling, by about 0.1 to 0.4 bits per 1k steps. Each
level now buys reconstruction roughly in proportion to its codebook capacity.
The bound rises anyway, because
the trunk predicts codes barely better than their usage frequencies. On one
held-out row, its gain over the marginal is 0.43, 0.13, -0.08, -0.03, 0.19
and 0.92 nats per code for levels 0 to 5. Level 0 can carry at most 0.04
bits/byte, so the level-0 cut amounts to no codes at all: the best bound so
far is a within-word byte model.

**No usable sequences yet.** Sampling at temperature 1.0 gives word-shaped
fragments ("Examination", "institution") with no sense. Greedy decoding
repeats a word or falls into whitespace. Even from the true codes, greedy
spelling derails within a few words. Teacher-forced on the full ladder, the
decoder gets 66% of bytes right, and 57% of the first byte of a patch. One
wrong delimiter then shifts every later byte onto the wrong patch's latent.
A space emitted at a patch start never ends the patch either, because space
patching only cuts at the first delimiter of a run, so the decoder stays on
one latent for good.

The cut series were missing from the dashboard. Each train micro-batch
after validation at the same step replaced the row's `extra_metrics` JSON.
Fixed 2026-11-06 (`MetricsLogger.log` now merges the JSON key by key). The
fix takes effect when the run restarts; until then, Lightning's own
`logs/model/version_0/metrics.csv` holds the cut values.

**Step 10k (2026-11-07).** Cut bounds are 2.69, 2.72, 2.81, 3.04, 3.49 and
3.98; reconstruction from the full ladder is 0.99. The full-ladder bound falls
about 0.08 bits per 1k steps, and the level-0 bound about 0.03, so the gap is
closing at roughly 0.05 per 1k.

The dashboard's samples were full of replacement characters, and the cause was
generation. With no MTP head, the dashboard asks for one byte per call, and
every call opened a fresh patch at the end of the prompt. So every byte was
spelled as the first of a new patch, from the start marker and a freshly
sampled latent, blind to every byte before it. Simulated on the step-10k
checkpoint (T = 1/3, repetition penalty 1.15), byte-at-a-time output was `ooo
ooo oaoot ... h�s�twwho3aw1`, and one 240-byte call gave words. Fixed: a
prompt that ends mid-word is continued in its open patch. The same simulation
now gives words with no replacement characters (repetitive: "of the of of").

A smaller, structural source remains. Space patching treats a UTF-8 lead byte
(0xC0 and up) as a delimiter, so a character after a letter splits across
patches: the lead byte ends one, and its continuation bytes open the next,
where the fence hides the lead byte. Teacher-forced on the true codes, a
continuation byte that opens a patch gets 35% of the probability spread over
all continuation bytes (93% inside a patch). At T = 1.0 it accounts for about
0.4% of generated bytes being invalid.

## Next, if it trains

- Sample quality against the number of levels actually used (cut generation
  at level k and decode), the paper's Figure-style read.
- MBR decoding: cheap here, because a chunk costs 6 trunk passes.
- An input-dependent ladder (the paper's first future-work item) maps onto
  this repo's halting machinery.
- The harmonic tie from `text_diffusion.md`: levels in a harmonic basis, where
  "coarse" means low frequency rather than wide span.
