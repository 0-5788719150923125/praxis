# Exact-match reward: score samples by how much of them exists in the corpus

> Status: **parked** (2026-10-22). Probe: `tools/probe_exact_match.py`.
> The probe ran and the signal does not point anywhere at this index size:
> within a prompt the reward ranks samples the way the model's own likelihood
> already does, and lowering the temperature beats it. No RL or reward model
> gets built unless a much larger index changes that (see Results).

## The idea

Small models here mostly produce word salad: locally plausible fragments in
an order that means nothing. Reward the model for emitting sequences that
exist **verbatim** in the training data, with more reward for longer matches.

Plain cross-entropy already rewards "say exactly what's in the data", one
token at a time. What it never does is score the model on its **own**
samples. At inference the model drifts off the data within a few bytes and
was never trained to find its way back (exposure bias). The new part is the
on-policy scoring.

## The signal

`L_t` = length of the longest substring ending at byte `t` that occurs
somewhere in the corpus. It grows while the model stays on-corpus, drops to
~0 on a novel byte, and grows again if the model recovers. That makes it dense
(per byte, no sparse episode reward) and it rewards recovery directly.

A suffix array over the corpus answers it in `O(log n)` per byte.
Infini-gram does this over trillions of tokens in milliseconds.

## Precedent

- **Infini-gram** (Liu et al. 2024, arXiv:2401.17377) - the lookup itself:
  suffix arrays, longest-suffix match, unbounded n. Mixing the ∞-gram with
  a neural LM at decode time cuts perplexity a lot, with no training.
- **MIXER** (Ranzato et al. 2016), **Minimum Risk Training** (Shen et al. 2016) -
  REINFORCE / risk minimization against BLEU. BLEU scores n-gram overlap up to
  4-grams, so this reward is roughly unbounded-length BLEU against the whole
  corpus instead of one reference.
- **SequenceMatch** (Cundy & Ermon 2023), **IRL for language** (Wulfmeier et
  al. 2024) - generation as imitation learning, aimed at compounding error.
- **Copy Is All You Need** (Lan et al. 2023) - generate by pasting corpus spans.
- **Creativity Index** (Lu et al. 2024) - the same measurement used in reverse:
  how much of a model's output can be rebuilt from verbatim web text.
- **RL with simple sequence priors** (Saanum et al. 2023, arXiv:2305.17109) -
  rewarding compressible action sequences; the compression-as-reward relative.

A search found nobody using a suffix-array longest match directly as an RL
reward. The family is well trodden; that specific combination looks open.

## Where it breaks

- **Collapse onto the easiest strings to match.** The reward-maximizing
  policy ignores the prompt and recites boilerplate (license headers,
  Gutenberg preambles). `next/rl.md` §1.1 already saw a milder version: the
  joke policy plus an oversampled Jester corpus pulled the model onto
  "Sherlock Holmes" and "priest". Long matches are easiest on high-count
  strings, so this reward pulls the same way, harder. Fix: count only matches
  anchored in the prompt.
- **Repetition.** Web corpora contain long repetitive spam, so loops can score.
- **Capacity.** A small model holds very little verbatim (Allen-Zhu & Li:
  ~2 bits/parameter of knowledge). Rewarding recall spends that budget on
  rote text, against the compression goal.
- **The cause may be the data, not the objective.** TinyStories (Eldan &
  Li 2023) got coherent text out of a few million parameters by narrowing
  the data. Word salad from a tiny model on a broad mix can be capacity vs.
  breadth, which no reward fixes.
- **Common is not correct.** `L_t` measures how *typical* a string is, not
  whether it is the right continuation. Lowering the temperature already buys
  typicality for free. The reward has to beat that to be worth anything.

## The probe

`tools/probe_exact_match.py`, three stages, no training:

1. `index` - draws documents through the run's own samplers and chat template,
   weighted by the run's last logged mix (`data_metrics.db`), builds suffix
   arrays over nested prefixes (2 → 32 MB), and holds out prompts from the
   prose sources (prompts never in the index).
2. `sample` - loads the run's checkpoint in a separate process and samples
   continuations with the served decoding settings, 8 per prompt, batched.
3. `score` - computes `L_t` for every continuation against every index size.

Arms, all against the same index:

| arm | what it is |
|---|---|
| `human` | the held-out document's true continuation |
| `shuffled` | that continuation with its words shuffled (word salad) |
| `model` | the run's samples at default decoding |
| `model T=0.5` | the run's samples at lower temperature (typicality control) |

What would make it promising:

1. `human` reward clearly above `model` and `shuffled`, and the gap **widens**
   as the index grows. If the model already matches real text's statistics,
   there is nothing to climb.
2. Real spread **within** a prompt (a group-relative advantage needs variance
   between samples of one prompt).
3. Within a prompt, higher reward goes with better fluency under an
   independent LM (SmolLM2-135M), and **not** with self-repetition.
4. Best-of-8 by reward beats just lowering the temperature on that judge.

## Results

`byte-latent-h` (603d2b039) at step 376,801. 64 held-out prose prompts,
256-byte continuations, 8 samples per prompt at default decoding and 4 at
T=0.5. The index is 32 MB of the run's own mix (8,731 docs), with nested
prefixes from 2 MB. Judge: SmolLM2-135M-Instruct. Raw numbers are in
`build/exact_match/report.json`.

Reward (`mean log2(1 + L)`) by index size:

| index | human | shuffled | model | model T=0.5 | human - model |
|---|---|---|---|---|---|
| 2 MB | 3.228 | 2.817 | 2.990 | 3.608 | +0.238 |
| 4 MB | 3.333 | 2.894 | 3.067 | 3.710 | +0.266 |
| 8 MB | 3.458 | 2.965 | 3.141 | 3.802 | +0.317 |
| 16 MB | 3.565 | 3.034 | 3.204 | 3.871 | +0.360 |
| 32 MB | 3.675 | 3.101 | 3.266 | 3.942 | +0.409 |

At 32 MB:

| arm | L mean | L median | judge bits/byte | anchor bytes | copy8 |
|---|---|---|---|---|---|
| human | 20.6 | 10 | 0.96 | 17.8 | 0.144 |
| shuffled | 8.1 | 8 | 2.33 | 3.8 | 0.063 |
| model | 10.0 | 9 | 2.60 | 9.2 | 0.040 |
| model T=0.5 | 20.7 | 13 | 1.49 | 17.5 | 0.129 |

Against the four criteria:

1. **Half met.** Human beats the default model and word salad, and the gap
   widens with the index (+0.24 to +0.41). But **T=0.5 beats human at every
   index size**. The reward's peak lies past real text, toward
   low-temperature output, which is the "common is not correct" failure.
   The default samples score only 0.17 above word salad, and the judge rates
   them *less* fluent than word salad (2.60 vs 2.33 bits/byte).
2. **Met.** Spread within a prompt is 0.158 against 0.280 between prompts,
   which is enough for a group-relative advantage.
3. **Met, but it doesn't mean much.** Within a prompt, the reward tracks judge
   fluency (Spearman +0.88) and barely tracks self-copy (+0.24). It tracks
   the model's **own likelihood** just as closely (+0.89).
4. **Not met.** Best-of-8 by reward reaches 2.15 judge bits/byte, while
   plain T=0.5 sampling averages 1.49.

The control that settles it compares picking the best of n by reward with
picking it by the model's own likelihood. The table shows judge bits/byte
saved (positive means the corpus picked better):

| index | model | model T=0.5 |
|---|---|---|
| 2 MB | -0.004 ± 0.017 | -0.033 ± 0.012 |
| 8 MB | -0.000 ± 0.021 | -0.009 ± 0.006 |
| 32 MB | +0.005 ± 0.021 | -0.020 ± 0.011 |

The two pick equally well at default temperature at every index size, and
the corpus picks slightly worse at T=0.5. At 32 MB the index knows nothing
the model's own likelihood doesn't. That is not surprising, since the model
trained on far more than 32 MB. RL on this reward would mostly sharpen the
model toward its own mode, and temperature does that for free. In the
examples, the top-reward samples also lean on long runs of whitespace, the
easiest string to match, which is a small early case of the collapse risk.

**What could change it.** Human text's reward grows faster with the index
than any model arm (+0.11 per doubling, against +0.08 for T=0.5 and +0.07
for the default model). So the corpus does separate real text more as it
grows. Extrapolating the closing T=0.5-to-human gap gives a crossover somewhere
between ~4 GB (fitting the last two doublings) and ~25 GB (fitting all five),
which is Infini-gram scale. That is an extrapolation from five
points, and the selector control shows no trend beyond noise. A suffix
array here costs ~40 bytes of RAM per corpus byte to build, so a GB-scale
test needs either a disk-backed build or Infini-gram's public indexes
(which hold other corpora, not this run's mix).
