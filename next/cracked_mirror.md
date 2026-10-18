# The cracked mirror

Written 2026-10-15 from two ideas: a friend's "twist the kaleidoscope until it
fits rather than taking the first facet you land on", and a crack in a dorm
mirror where the reflection on one side sits about an inch lower than the rest.
Built as `coords="crack"` in `praxis/attention/kaleidoscope.py`, profile
`kaleido_12_zoom_crack`, run `experiments/byte-latent-h.yml`.

## The limitation, stated exactly

Kaleidoscope's score row for query `i` is

    S[i, :] = sum_k w_k(x_i) M_k[i, :]

Every `M_k[i, :]` is fixed by position. Whatever the router does, the row lies
in the span of N fixed vectors, and the query picks a point in that span. A
key's content never enters. A needle sits at a position chosen by content, so
finding it needs a row that can peak at any position, which an N-dimensional
span of fixed rows cannot do for N < T. Needle-in-a-haystack is not hard for
kaleidoscope - it is impossible, and no router, facet or larger N changes that.

## Idea 1: twisting the kaleidoscope

**Read literally, it cannot help.** Searching over blends until one fits the
input, by recurrence, by optimization, or by classification, still lands inside
the span. A better choice of `w` does not change which keys are reachable.

**But the other half of the idea is the mechanism that works.** "A local
recurrence filtered through binary classification" describes a scan: walk back
from the query over the keys, classify each one, and accumulate. That is
exactly how the crack computes its lag (a reverse cumulative sum of per-key
gates). And "take the first one we land on" is a real, published attention
rule: stick-breaking attention (Tan et al. 2024, arXiv:2410.17980) scans back
from the query and the first key that passes takes most of the weight. The
crack generalizes it. The mirrors choose WHICH crossing - the first, the
second, a spread - so "exploring other angles" becomes choosing a crack count.

## Idea 2: the crack

A crack is a place where content is allowed to move the glass. Each query draws
a hyperplane through a 32-wide key space (a normal and a threshold, both read
off the query), and every earlier key lands on one side or the other:

    g_ij = sigmoid(n(x_i) . c(x_j) + b(x_i))

That is the binary classifier. It is linear per query, which is the SVM picture
without the max-margin fit. `kaleido_crack_decisiveness` stands in for the
margin: 0 means every key sits on the hyperplane, 1 means each one is cleanly
on a side.

The crack does two things, one for each reading of the mirror:

1. **The offset** (one side sits an inch lower). Keys on the far side get a
   logit shift `drop(x_i) * (g_ij - 1/2)`. Centered, so it is exactly zero at
   init, while `drop` starts at 1 so the classifier learns from step 0. This
   gives content one bounded bit per (query, key) pair. Every graded
   preference still comes from the frozen geometry, so the content
   classifies and the geometry ranks. Summed before the softmax, the two
   multiply: mass lands where a key is on the right side AND in the right place.
2. **The displacement** (everything past the crack is shifted). A third of the
   dictionary reads lag counted in cracks instead of tokens:

       p_ij = sum_{j<k<i} g_ik + g_ij / 2

   The half step is what makes the count usable without a `q . k` term. With
   one hit at `m`: later keys sit at 0, the hit at 1/2, everything before it at
   1. The needle is the only key at its coordinate, and the k-th most recent
   hit sits alone at k - 1/2. The haystack collapses onto a few coordinates,
   which is also why a handful of random mirrors can resolve it: they only
   have to separate 0, 1/2 and 1, not every one of T lags. With all gates on,
   `p` is the token lag and the crack mirrors are the plain lag mirrors. The
   count reads the classifier's decisions without sending it gradient (see the
   toy evidence for why).

The other two thirds are kaleido_12_zoom's ratio mirrors and the split's
log-lag mirrors, so nothing the block could do before is lost.

## Prior work, and the empty cell

| method | content in the logit | position |
|---|---|---|
| RoPE | `q . k` | fixed rotations |
| PaTH (Yang et al. 2025, arXiv:2505.16381) | `q . k` | data-dependent Householder reflections, composed along the path from j to i |
| CoPE (Golovneva et al. 2024, arXiv:2405.18719) | `q . k` | gate counts, learned table `q . e[p]` |
| FoX (Lin et al. 2025, arXiv:2503.02130) | `q . k` | cumulative key-only forget gates, linear |
| Stick-breaking (Tan et al. 2024) | `log sigmoid(q . k)` | cumulative `log(1 - sigmoid)`, no PE |
| Mamba-2 | (state space) | content clock `Delta_t`, exponential decay |
| Synthesizer / kaleidoscope | none | fixed or frozen geometry |
| **cracked kaleidoscope** | **one bounded bit per pair** | **frozen nonlinear geometry, counted in cracks** |

PaTH is the closest relative in spirit, and worth reading: its relative position
between two tokens is a product of content-dependent REFLECTIONS accumulated
between them, which is "mirrors whose angle the content sets" with a `q . k`
on either side.

What nobody has run is the last row: content allowed in only as which side of a
hyperplane, never as a graded similarity, with all the graded structure from
frozen geometry.

Numbers worth not re-deriving:

- **CoPE's code model is our trunk's shape.** 4 layers, 8 heads, hidden 256,
  4096 context, 100B tokens: CoPE beat RoPE by 5% on perplexity, and
  RoPE + CoPE did not beat CoPE alone.
- **A fixed linear function of the count is not enough.** CoPE with ALiBi's
  linear slopes on the counted position (Wikitext-103) fell to roughly
  absolute-PE perplexity, 24.16 against 22.55. FoX and stick-breaking keep
  `q . k`, which is why linear suffices there. Without `q . k`, the nonlinear
  mirrors are what do the work.
- **A short count is enough.** CoPE capped at 16 positions over 1024 tokens
  matched a cap of 1024. The log warp on our canonical grid gives the first
  few cracks their own cells (0 / 0.5 / 1 / 1.5 land on columns 0 / 4.3 / 7.3 /
  9.6 at T=400, R=64).
- FoX reaches near-perfect needle-in-a-haystack inside its training length
  where every recurrent baseline failed, and stick-breaking beats softmax on
  RULER. Both show that "most recent match" with a content gate is a retrieval
  mechanism, not only a position one.

## Toy evidence

`tools/probe_recall.py`: associative recall on one 12-mirror kaleidoscope
layer. Every input carries its previous token's embedding, standing in for a
layer-1 previous-token head, so one layer suffices. 64 keys, 32 values, chance
3.1%, AdamW 3e-3, batch 64, 3000 steps on CPU. Steps to solve in parentheses.

| task | T | QK, no PE | no crack | count only | offset only | both, attached | both, detached |
|---|---|---|---|---|---|---|---|
| unique keys | 33 | 100% (500) | 13% | 12% | 100% (500) | 99.8% (~3000) | 99.9% (500) |
| unique keys | 97 | | | | 99.9% (~1500) | **8%** | 97.7% (~1250) |
| latest of repeats | 33 | 52% | | | 98.4% | 99.6% | 99.7% |
| second-latest | 33 | | | | **53%** | | 99-100% by 500, see below |

What it says:

- **The offset is what retrieves.** Alone, it matches `q . k` step for step.
  So the dorm-mirror reading - one side of the crack sits lower - is enough to
  give kaleidoscope needle retrieval, with content entering as one bounded bit
  per pair.
- **The count cannot learn on its own.** Count-only equals the no-crack
  control. Its only gradient into the classifier passes through the slopes of
  random mirrors, which is noise. With that path open, the full crack learns
  6x slower at T=33 and fails outright at T=97 (8%). So the module DETACHES
  the count: the offset trains the classifier, and the count reads its
  decisions.
- **The count does what only it can do.** Second-latest recall needs "the
  second match", which no positional mirror can express: offset-only sits at
  53% (it finds a match, then guesses which). The detached count reaches 99% by
  step 500 and leans on its crack mirrors (`crack_share` 0.61 against 1/3
  parity).
- **That last result is fragile at a high learning rate.** At 3e-3 it held
  ~100% through step 2000, then fell to 71.5% (seed 0) and 89% (seed 1). At
  1e-3 it held 97-99% to the end (98.8%). The likely mechanism: detached, the
  count cannot defend the cracks it needs, and on this task the offset is
  rewarded for demoting the latest match - the very crack that puts the
  second match at 3/2. A second plane (one for the offset, one for the count)
  would remove the conflict, at the cost of giving the count its own, noisy,
  gradient.
- On latest-of-repeats the offset alone already gets 98.4%: the lag mirrors
  break ties toward recency. The count adds about a point there.

QK has no position encoding here, which is why it cannot pick the latest
repeat. It is the reference for unique-key retrieval only.

## The run

`byte-latent-h`: `-g` with `attention_type: kaleido_12_zoom_crack`. Every
other kaleido profile is positional, so **any copy gain above zero is the
crack's** - a single run answers the question without a control arm.

Read in this order:

1. `kaleido_crack_interaction` - 0 means one content clock for every query
   (FoX's or Mamba's shape), a position result and not retrieval. Rising means
   each query classifies keys its own way.
2. `kaleido_crack_share` against 1/3 parity, and `kaleido_crack_drop` against
   `kaleido_turn_scale`. A drop that runs far past the turn scale means the
   block has become sigmoid-gated attention with the mirrors along for the
   ride. That is still a working model, but the geometry is no longer the
   claim.
3. `val_copy_gain`, then `val_context_gain`. The ratio mirrors can earn some
   context gain without retrieval (a prefix average helps late positions), so
   copy gain is the retrieval verdict.

Give it the same 50k steps `-f` and `-g` needed before their gains rose.

## Falsifiers

- `val_copy_gain` at 0 through 75k while `crack_interaction` rises: the
  classifier works and the geometry cannot use it.
- `crack_interaction` near 0: the crack became a content clock. Still a
  finding about position, not about retrieval.
- `crack_share` near 0: the router ignores the crack mirrors. Read `drop` to
  see whether the offset alone carried anything.

## Outcome (stopped at 14k, 2026-10-17)

`-h` ran 14.1k steps and was stopped. The `byte-latent-h` name now belongs to
the ghost-features run, and its `--reset` launch reused this run's directory.

- **Validation reached parity.** The `val_loss` gap to `-g` closed at every
  eval: +0.068 at 2k, +0.025 at 10k, +0.0035 at 13.3k. Steps were about 20%
  faster.
- **The crack turned into attention.** `crack_decisiveness` fell from 0.97 to
  0.62, faster each time, while `crack_drop` rose to 2.96 against a
  `turn_scale` of 0.79. In the sigmoid's linear range the offset is a scaled
  32-wide `q . k`, so the block was buying back graded similarity, one head's
  worth.
- **No retrieval.** `crack_interaction` settled at 0.14-0.18, mostly a
  key-type clock, and `val_copy_gain` sat near 0 (`-g` was about 0.009 at the
  same steps).
- **Memory was never the advantage.** The blend is per query, so every example
  builds its own `[T, T]` scores, and the crack adds 45% on top (saved for
  backward, B=4, T=373, one layer: 29.7 to 43.0 MiB). `self_attention` runs
  flex attention and never builds the matrix at all.

The verdict: a convoluted mechanism that converges toward the attention it
replaces, with no memory win, is not worth carrying. The code stays in the
registry as `kaleido_12_zoom_crack`.

## Known gaps

- No KV cache path, as before: decode recomputes the full `[T, T]`.
- The crack is single-plane per query, and the toy shows the cost: the offset
  can undo the cracks the count relies on. A second plane (the count from one
  classifier, the offset from another) is the natural next arm, especially if
  `crack_share` rises and then falls back while copy gain stalls.
- Before this change kaleidoscope ignored `block_ids` and read across packed
  documents. It now masks them, as `self_attention` does, so `-a`'s kaleido
  trunk is not directly comparable to `-h`'s.
