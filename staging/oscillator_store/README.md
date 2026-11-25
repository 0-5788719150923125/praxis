# oscillator_store

A character language model whose knowledge lives in a text store instead of in weights. A complex oscillator turns every moment of the training text into an address. Prediction reads what followed the nearest stored moments, along with an exact n-gram over the same text and a parametric reader, mixed per position by a small gate. Standalone: nothing here imports Praxis (it uses the Praxis venv for torch).

It is the "best approach" from `next/oscilloscope.md`, round 2. Tests 1-5 there showed that a few numbers cannot hold a language; Test 6 showed that an untrained complex oscillator is a good address into the text that holds it.

```
 text so far ──► oscillator: 64 damped rotations ──► address (128 numbers)
                                                         │
   ┌──────────────────────────┬──────────────────────────┼────────────────────────┐
   ▼                          ▼                          ▼                        ▼
 text store                 Kneser-Ney 9              tiny experts             GRU-256
 (every training position,  (exact suffixes,          (SMEAR over 1024         (everything in
  s-means index, 64          same text)                MLPs, hidden 4)          weights)
  nearest neighbors)
   └──────────────┬───────────┴──────────────────────────┴────────────────────────┘
                  ▼
   gate: per-position weights from each reader's confidence ──► p(next character)
```

## Pieces

| piece | what it does | trained numbers |
|---|---|---|
| `Oscillator` | h_t = λ·h_{t-1} + E[x_t] with λ = exp(-exp(ν) + iθ), 64 complex units, starting as Test 6's fixed bank (8 decay rates × 8 phases). The state, standardized, is the address. A chunked scan makes training cheap: the whole training store is re-keyed every step. | 4,480 |
| `TextStore` | every position of the training text (minus 32 warm-up positions per 1,000-character stream) with the character that followed it. An inverted-file index (Jégou, Douze & Schmid 2011): s-means or k-means centroids, 4√N of them rounded to a power of two, and each search visits the nearest 1/128 of the lists. Keys stay on the CPU in float16 and stream to the GPU a list at a time. | 0 |
| `KneserNey` (`ngram.py`) | interpolated Kneser-Ney, order 9, over the same text: exact suffix matches. | 0 (counts) |
| `TinyExperts` | SMEAR (Muqeeth, Liu & Raffel 2023) over 1,024 MLPs (128 → 4 → 34): a router reading the address weighs the experts, their weights are merged per position (base plus deviations, as in `praxis/routers/smear.py`), and the merged MLP runs once. | 835,246 |
| `GRU` | a 256-unit GRU trained on the same text, as both a baseline and the parametric reader of the kNN-LM pairing (Khandelwal et al. 2020). | 232,994 |
| `Gate` | 4 hidden units from each reader's confidence (max probability, entropy) and the nearest stored distance, to per-position mixing weights. Fit on held-out text with every reader frozen. | 47 to 60 |

The data is TinyStories with a 34-symbol alphabet (`data.py`). Store: the first 9.7M training characters. The oscillator trains on the next 200K and the gate on the 100K after that, none of which is in the store. Evaluation: Test 3's set (the first 200K validation characters in 1,000 streams, 167K scored predictions).

## Run

```
cd staging/oscillator_store
SMEANS_REPO=/path/to/s-means ../../.venv/bin/python run.py   # about 13 minutes on one GPU, writes results.json, run.log, checkpoint.pt
../../.venv/bin/python run.py --smoke                        # about 30 seconds, tiny sizes
../../.venv/bin/python run.py --data fineweb                 # the same on FineWeb (streams 11M characters once into data/)
```

`SMEANS_REPO` points at a clone of github.com/suayptalha/s-means (optional; without it the index uses 25 passes of k-means, which gives the same quality, more slowly). Its package `__init__` needs scikit-learn, so `run.py` loads `smeans.core` directly.

## Results (2026-11-23, `results.json`)

Bits per character on the evaluation set; the store holds 9,379,900 positions of 9.7M training characters.

| readers, mixed by the gate | trained numbers | bpc |
|---|---|---|
| Kneser-Ney 9 alone | 0 (4.1M counts) | 1.167 |
| store alone | 4,480 (the oscillator) | 1.148 |
| GRU-256 alone (20K steps) | 232,994 | 1.148 |
| tiny experts alone | 835,246 | 1.464 |
| store + n-gram | 4,480 + 34 | 1.081 |
| store + n-gram + experts | + 835K | 1.067 |
| **store + n-gram + GRU** | 4,480 + 233K + 47 | **1.031** |
| all four | + 1.07M | 1.030 |

In the all-four system, removing a reader costs: store +0.063, GRU +0.057, n-gram +0.015, experts +0.0005. Mean gate weights: store 0.48, GRU 0.33, n-gram 0.19, experts 0.01. A fixed global mixture instead of the gate: 1.053, so the gate is worth 0.023.

With a 1M-character store (Test 6's record), searched exactly:

| oscillator | store alone | store + n-gram | store + n-gram + experts |
|---|---|---|---|
| untrained (Test 6's bank) | 1.516 | 1.331 | - |
| trained on the store alone (first run, `results_solo_address.json`) | 1.410 | 1.319 | 1.262 |
| trained through the store + n-gram mixture (final) | 1.429 | 1.326 | 1.266 |

Kneser-Ney 9 alone at 1M: 1.375. Training moved the oscillator's timescales from 2-64 characters to 3-42.

The index: searching 128 of 16,384 lists (0.8% of the store) finds 99.98% of the exact 64 nearest neighbors. At 1M, s-means and 25-pass k-means give the same recall (99.47% and 99.46%, 32 of 4,096 lists) and the same bits as exact search (1.429 for all three).

Sample from all four readers (temperature 1, seed 0):

> once upon a time, there was a would help unpoi...ith her, she loved how they were busy muddy. he hose. we have to the car. lily crawled around, but he couldn't see what was inside.
> his mom came to the zoo with her mommy made a good friend."
> they go to the couch with their toys in the park. one day, they see a big tent in the park. she saw a bad dream. they just said to jumining, the bear won named lucy. lily looked

### On FineWeb (`results_fineweb.json`, `run.py --data fineweb`)

The same system on 9.7M characters of FineWeb (sample-10BT, streamed in order, same alphabet; evaluation on the documents that follow):

| readers, mixed by the gate | trained numbers | bpc |
|---|---|---|
| Kneser-Ney 9 alone | 0 (10.2M counts) | 1.958 |
| store alone | 4,480 | 2.012 |
| GRU-256 alone | 232,994 | 1.972 |
| tiny experts alone | 835,246 | 2.238 |
| store + n-gram | 4,514 | 1.903 |
| store + n-gram + experts | + 835K | 1.854 |
| **store + n-gram + GRU** | + 233K | **1.826** |
| all four | + 1.07M | 1.823 |

In the all-four system, removing a reader costs: n-gram +0.059, GRU +0.044, store +0.009, experts +0.003. On TinyStories the store was the most valuable reader (+0.063); on web text it is nearly redundant with the n-gram and the GRU. At 1M characters the trained and untrained oscillators give the same store + n-gram mixture (2.173 both).

Sample (all four readers):

> the history of the application, and thunder benefits the nile, transpt. by parking improve.ets oscilled out the hunts dilation and many of the yery strength incorporating and may increasing the table and taking me he made a great landscape iiratego by food trucks.

## Reading it

- **The best approach is the store, the n-gram and a small GRU: 1.031.** That is 0.12 below the same GRU alone and 0.05 below the store and n-gram alone. The record and the weights cover different things: the store holds continuations it has seen, the GRU generalizes to contexts it has not. This is kNN-LM's result (Khandelwal et al. 2020) at character level, with an address that needs no training and an index that makes the store cheap to search.
- **Outside the parameters, partly, and mostly on TinyStories.** With 4,514 trained numbers, the store and the n-gram (1.081) beat a GRU with 233K (1.148) trained on the same text. On FineWeb the same comparison is 1.903 against 1.972, and the store itself is worth 0.009 bits in the full system: TinyStories' repetition was most of the store's value. The n-gram (also read straight from the text) holds up better on both.
- **The tiny experts do not earn their place.** On their own they reach 1.46 with 835K trained numbers, against the GRU's 1.15 with 233K. Next to the store they add 0.014, and next to the GRU 0.0005. Their activations are small (4 per position), but merging 1,024 experts costs about 700K multiply-adds per position, more than the GRU-256's step (about 230K). They read a fixed linear memory, while the GRU learns its own.
- **Training the address barely matters once the n-gram is in the mix.** Trained on the store alone, the oscillator gets better alone (1.516 to 1.410) by imitating exact matches, and then adds little next to the n-gram. Trained through the mixture it ends in the same place. Its 0.005-0.006 gains at 1M are the size of the gate's own run-to-run variation (the untrained row moved 0.006 between runs). The untrained oscillator would do nearly as well.
- **s-means makes lookups cheap, and so does k-means.** Clustering belongs in the index, not in the summary (Test 6 showed centroid summaries lose). With 4√N lists and 1/128 of them searched, recall stays above 99.4% and the bits do not move. s-means' advantage is build time: 3 epochs instead of 25 passes.
- **Locally fluent, globally lost.** The sample reads like TinyStories a clause at a time and forgets the story between sentences. None of the readers keeps a story: the oscillator's longest timescale is 42 characters, the n-gram sees 8, and the GRU was trained on 128-character windows.

## Limits

- TinyStories is synthetic and repetitive, which flatters any store. FineWeb confirms it: the store's contribution drops from 0.063 to 0.009 bits.
- The GRU baseline got one fixed recipe (Test 3's, 20K steps, about 2 minutes of training). A tuned or larger model, or a small transformer, would be the real parametric bar.
- The store costs storage and search: 2.4 GB of float16 keys here, though they can be recomputed from the 9.7 MB of text, and 128 list visits per prediction.

## Status

On web text the system reduces to a Kneser-Ney model plus a small GRU, with the store adding 0.009 bits. As a way to keep a language model's knowledge outside its parameters, this does not hold up beyond repetitive text. What carries over: a complex (rotating) memory is a far better fixed address than a real (decaying) one, an inverted-file index makes any such store cheap to search, and a gate over readers beats a fixed mixture by about 0.02 bits. The reasoning and the follow-ups are in `next/oscilloscope.md`, rounds 3 and 4.

Earlier tests and the reasoning behind each piece: `next/oscilloscope.md`. Test code for Tests 1-6: `~/repos/oscilloscope/`.
