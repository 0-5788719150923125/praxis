# Oscilloscope: four numbers, 500 samples, and where the bits live

> Status: **tested** (2026-11-23). Standalone code in `~/repos/oscilloscope/` (no Praxis; it borrows the Praxis venv's interpreter for torch). Five tests, results below. Short version: the shapes are real, and every piece of the notion has a working counterpart in science or engineering, but none of them gets around the cost - they move it. What a language model knows has to be stored as bits somewhere. Four numbers can hold a sentence (Test 1); they cannot hold the knowledge that makes sentences likely.
>
> **Round 2** (same day, [below](#round-2-outside-the-parameters)): where information can live outside the parameters, the four follow-up ideas (ghost transforms on the samples, imaginary numbers, a synthesis function for text, s-means), and Test 6. Short version: an oscillator is a bad store and a good address. An untrained complex oscillator that only points into the training text predicts at 1.52 bpc, ahead of a 67K-weight GRU trained on the same text. Without the imaginary part it gets 1.99.
>
> **Round 3** (same day, [below](#round-3-one-system-stagingoscillator_store)): the best approach as one standalone system in `staging/oscillator_store/`. Store, n-gram and a small GRU reach 1.031 bpc against 1.148 for the GRU alone; the tiny SMEAR experts and the oscillator's training add almost nothing.
>
> **Round 4** (same day, [below](#round-4-causality-graphs-x-y-z-and-real-text)): causality, what "graphs over clusters with learned Markov routing" is, X, Y, Z with interference (Test 7), and the system on FineWeb. On web text the store is worth 0.009 bits next to the n-gram and the GRU; its edge was TinyStories' repetition. X, Y, Z with interference is a worse address than without it.

## The notion, as stated

- Three neurons X, Y, Z (an oscilloscope's beam: two deflections and an intensity), maybe as fast weights, plus one learned parameter: a stable anchor.
- CALM-like generation: a sequence is 500 non-causal samples (think 44.1 kHz audio) instead of discrete token vectors.
- The samples are projected into regions of space, and the regions become symbols. A wave function is fit to the symbols, and the fast weights are fit to the wave function.
- Sliding-window capsule networks over the samples model how they relate.
- Text is forced into Gaussian noise, and the parameters mostly give way to the space of possible shapes.

Two follow-ups from the same session:
1. **Coarse to fine without causality:** scale the samples' influence down along the 500 (linearly, say), the way regression takes big steps first and tiny refinements at the end.
2. **The wave as a route through every activation:** 500 samples times N values per sample, each neuron a single point along the wave, and the wave learning to route through them.

## Every piece already has a name

| Piece | What it is | Prior work |
|---|---|---|
| X, Y, Z with stable interactions, one anchor | a 3-variable dynamical system with one bifurcation parameter | Lorenz (anchor rho), the classic analog-circuit attractor on an oscilloscope |
| regions of space become symbols | symbolic dynamics: a partition of state space | Markov and generating partitions |
| fit the fast weights so the shape spells the text | pick the initial condition whose orbit spells it | arithmetic coding is the orbit of a piecewise-linear chaotic map (Nagaraj, Vaidya, Bhat 2009); small kicks steer a chaotic orbit to spell a message (Hayes, Grebogi, Ott 1993) |
| text hidden in Gaussian noise | a code whose output is noise-like | chaotic masking (Cuomo & Oppenheim 1993; Cuomo, Oppenheim, Strogatz 1993); superposition codes and OFDM; holographic reduced representations (Plate 1995); CALM's VAE latent (Shao et al. 2025) |
| 500 samples per symbol, each neuron a point on the wave | a time-multiplexed single-node reservoir: one nonlinear node, N virtual neurons | Appeltant et al. 2011 (one Mackey-Glass node, 400 virtual nodes, 0.2% spoken-digit error); Torrejon et al. 2017 (one spintronic oscillator, 400 virtual neurons, up to 99.6% on spoken digits) |
| influence decaying along the samples | successive refinement, positional notation, superposition coding with successive cancellation | Equitz & Cover 1991; power-domain NOMA (Islam et al. 2017); residual VQ |
| one parameter | one real number fits anything by storing the data in its digits | Piantadosi 2018: f(x) = sin²(2^(rx) arcsin √θ) stores r bits per data point in θ |
| fast weights fit per sequence | test-time fitting | fast weights (Ba et al. 2016), test-time training layers (Sun et al. 2024), Titans |
| a wave routing through every weight | training a big network inside a small subspace | intrinsic dimension (Li et al. 2018: ~750 for an MNIST net with 199K weights; Aghajanyan et al. 2021: 200 numbers reach 90% of full fine-tuning on MRPC); PRANC (Nooralinejad et al. 2023); SeedLM (Shafipour et al. 2025) |

## The wall: the bits have to live somewhere

Four facts settle most of this before any test runs.

1. **Samples add resolution, not information.** The 500 samples are a deterministic function of the 3 fast numbers and the anchor, so they cannot carry more than those numbers do (the data processing inequality). They are the screen, not the signal.
2. **A real number holds as many bits as its precision.** A float64 has a 53-bit significand. Piantadosi's one-parameter model works by writing the data into θ's digits: it fits everything and predicts nothing.
3. **Capacity that grows with length needs chaos, and chaos defeats gradient descent.** A smooth (quasi-periodic) figure driven by 3 numbers makes polynomially many distinct symbol strings, so its capacity grows like log L. A chaotic one makes exponentially many, growing like h·L, where the entropy rate h is at most the sum of the positive Lyapunov exponents (Ruelle's inequality, with equality for the physical measure of a dissipative attractor by Ledrappier & Young 1985). Lorenz at rho = 28: 0.906 nats = 1.31 bits per time unit. The same stretching multiplies gradients by about e^(λt), and Mikhaeil, Monfared, Durstewitz (2022) prove that an RNN producing chaos always has diverging loss gradients.
4. **Knowledge costs about 2 bits per parameter.** Allen-Zhu & Li (2024) measure 2 bits of knowledge per parameter (given ~1000 exposures per fact; it holds at int8 and falls to 0.7 at int4). A trained network is already text written into Gaussian-looking noise: its weights look random and hold the corpus. Time can stand in for width when the work is computation, not when it is knowledge: a 12-layer model looped twice beats a 24-layer model on math word problems (34.3 vs 29.3) and trails it on closed-book QA (9.3 vs 11.2) (Saunshi et al. 2025).

The honest unit of cost is a two-part code: bits(model) + bits(text | model). The notion moves bits from the first term into per-sequence fast weights, but the second term is -log2 p(text), and only the model can make it small.

## Test 1: a sentence is one number (`one_number.py`)

Arithmetic decoding is the orbit of a chaotic map: split [0, 1) into intervals sized by a model's next-symbol probabilities, read which interval x sits in, stretch that interval back to [0, 1), repeat. The map's Lyapunov exponent is the model's entropy rate. Encoding a text means finding the initial condition whose orbit spells it, and Φ⁻¹(x) turns that number into a Gaussian sample: text written into Gaussian noise, exactly invertible, with nothing trained beyond the model that sets the partition.

TinyStories, 34 symbols, 9.94M training characters (the cache came up short of 10M after normalization), validation bits per character on 200K characters:

| partition (= the model) | stored numbers | val bpc | fits in one float64 | fits in X, Y, Z (three float64) |
|---|---|---|---|---|
| uniform | 0 | 5.09 | 10 chars | 31: `spot. spot saw the shiny car an` |
| one anchor (Zipf exponent over frequency rank) | 1, plus the rank order (128 bits) | 4.35 | 11 | 36 |
| unigram | 34 | 4.25 | 12 | 37 |
| bigram | 848 | 3.26 | 13 | 47: `... and said, "wow, ki` |
| 5-gram, Kneser-Ney | 176K | 1.55 | 26 | 86: `... so bright and clean!" ` |
| 9-gram, Kneser-Ney | 4.1M | 1.16 | 26 | 93: `... clean!" kitty s` |

The same map, started from the same random Gaussian draw, under each partition:

- uniform: `kmm tlwgoa?k"fbiztpqioj'?nreblt!.abgqgx"uvwgxc.r houn"o'sz`
- one anchor: `o  eova fbn ax aeeebuiqanthpo x!lheg dewqekv. s ej  re oetmasii`
- bigram: `the bed b are ba d t wid s boghey. lke he s bog ad thon d the t.`
- 5-gram: `zie's beautiful plater to crystal! i will makery outside, and they made thoughouted,`
- 9-gram: `one day, mom. we were afraid, if it is not sad and scared, but she knew they worked hard and it was still in her living room.`

Reading it: the oscillator is identical in every row. Only the partition changes, and the partition is the model. The anchor buys 0.7 bits per character over uniform, the same as knowing which letters are common. Iterated in float64, the orbit spells 11 to 29 characters before rounding error, amplified by the stretching, takes over: 53 bits divided by the sentence's bits per character. That horizon is the Lyapunov time.

The coarse-to-fine follow-up is already inside this test: each character's influence on x shrinks by its own probability, so x is a positional number whose digits are characters of varying width. The last characters live in the last bits, which is where the precision runs out.

## Test 2: fitting X, Y, Z so a figure spells text (`fit_shapes.py`)

27 symbols (letters and space), 27 regions (k-means centers of each family's own trajectories), one region read per character. The fast weights are three numbers:
- **Lissajous:** X, Y, Z = sin(2π F u_i (k + ½) + phase_i) with u ∈ [0, 1]³; F = 1 (a smooth figure) or 8 cycles per character (a busy one).
- **Lorenz:** the initial condition, anchor rho = 28, one read every Δ = 1 or 4 time units (about 1.3 or 5.2 bits of fresh chaos per character).

Fitting, per real text window (32 windows per cell): gradient descent from 32 restarts on a soft region assignment with annealed temperature, with uniform, linearly decaying (the follow-up's proposal) or geometrically decaying position weights, or a curriculum that adds one character per stage with shrinking steps; and a gradient-free coarse-to-fine search that keeps the fast weights matching the first k characters and resamples around them (8,192 particles).

How many distinct strings each family can draw (log2 of distinct strings from 2^20 random fast weights, so 20 bits is the ceiling of the measurement):

| characters | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| Lissajous F = 1 | 4.3 | 8.6 | 11.9 | 14.2 | 17.1 |
| Lissajous F = 8 | 4.8 | 9.5 | 13.5 | 16.2 | 18.8 |
| Lorenz Δ = 1 | 4.8 | 9.1 | 11.6 | 13.5 | 16.4 |
| Lorenz Δ = 4 | 4.8 | 9.5 | 14.0 | 17.6 | ceiling |
| every string (27^L) | 4.8 | 9.5 | 14.3 | 19.0 | 28.5 |
| order-0 text entropy | 4.1 | 8.1 | 12.2 | 16.2 | 24.4 |

Real text windows spelled exactly (%), by length:

| family, method | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| Lissajous F = 1, gradient descent, uniform | 100 | 84 | 16 | 0 | 0 |
| ... linear decay | 100 | 75 | 38 | 0 | 0 |
| ... geometric decay | 100 | 81 | 34 | 3 | 0 |
| ... curriculum, shrinking steps | 100 | 97 | 56 | 12 | 0 |
| ... coarse-to-fine search | 84 | 75 | 22 | 3 | 0 |
| Lissajous F = 8, best gradient arm (curriculum) | 100 | 88 | 25 | 6 | 0 |
| ... coarse-to-fine search | 100 | 94 | 19 | 6 | 0 |
| Lorenz Δ = 1, best gradient arm (geometric or curriculum) | 100 | 41 | 9 | 3 | 0 |
| ... coarse-to-fine search | 100 | 78 | 6 | 6 | 0 |
| Lorenz Δ = 4, every gradient arm | 100 | 12-19 | 0 | 0 | - |
| ... coarse-to-fine search (two runs) | 100 | 88-100 | 31-41 | 9-22 | 0-3 |

Reading it:
- **Smooth figures run out of shapes.** At 4 characters, the smooth Lissajous figure and the slow Lorenz reads can draw only 2-4% of all 4-letter strings, and their capacity then grows by under 2 bits per character, half of what text needs (as theory predicts: about 1.3 bits per read for Lorenz at Δ = 1, and shrinking increments for any quasi-periodic figure). The best method spells 6-12% of real 4-letter windows. Here the wall is capacity, and the method barely matters.
- **The chaotic figure has the shapes, but gradients cannot find them.** Lorenz with Δ = 4 can draw 38% of all 4-letter strings, yet every gradient arm spells 0% of real 3-letter windows: each character stretches the landscape about 37-fold (e^(0.906 × 4)), so the loss surface is fractal by the third read. The gradient-free search gets one character further (31-41% at 3, 9-22% at 4) before its particles can no longer land in cells that shrink 37-fold per character, and some English transitions are rare or forbidden in the Lorenz symbolic dynamics.
- **Coarse to fine helps at the margin, not at the wall.** On the smooth figure, the curriculum lifts 3-character success from 16% to 56%, and linear decay to 38%. On the chaotic figure nothing gradient-based gets past 2 characters, and the coarse-to-fine search is the best method, which is what Test 1 predicts: successive refinement is how you find an initial condition in a chaotic system.
- Three float64 fast weights could in principle name 31-93 characters (Test 1). What they can actually be fit to name with a real oscillator is about 3.

## Test 3: 500 samples as 500 neurons (`delay_reservoir.py`)

The "every neuron is a point on the wave" follow-up is time-multiplexed reservoir computing. Each character is held for one period of 500 steps and injected through a random 500-sample mask: that character's signature, from one seed. One nonlinear node is sampled once per step, so each character produces 500 virtual neurons (500 × 3 for an X, Y, Z node), each one a point along the node's waveform. A softmax readout on those points predicts the next character. 1M training characters (1,000 streams), 200K validation characters, the first 32 characters of each stream not scored.

| model | memory between characters | trained numbers | val bpc |
|---|---|---|---|
| current character's mask only (control) | none | 17,034 | 3.29 |
| Kneser-Ney bigram | 1 character | 734 counts | 3.26 |
| Lorenz node, X, Y, Z sampled, rho = 10 / 20 / 24 / 28 | its 3 numbers | 51,034 | 3.13 / 3.34 / 3.51 / 3.52 |
| same, rho 20, mask decaying linearly along the 500 | its 3 numbers | 51,034 | 3.41 |
| same, rho 20, delayed feedback of its own waveform | 3 numbers + 500 delayed samples | 51,034 | 3.41 |
| same, rho 10, input gain 100 / 300 (default 30) | its 3 numbers | 51,034 | 3.15 / 3.19 |
| same, rho 10, gain 300, delayed feedback | 3 numbers + 500 delayed samples | 51,034 | 3.10 |
| same, rho 1 (barely oscillates), gain 300 | its 3 numbers | 51,034 | 3.06 |
| tanh node with a delay line, N = 50 / 125 / 500 samples | N delayed samples | 1,734 / 4,284 / 17,034 | 3.18 / 2.99 / 2.82 |
| random 500-unit echo state network | 500 units (a 500×500 matrix from the seed) | 17,034 | 2.27 |
| tanh delay line with masks and gain trained by backprop ("the wave learns to route") | 500 delayed samples | 35,036 | 2.01 |
| Kneser-Ney trigram | 2 characters | 6,111 counts | 2.45 |
| Kneser-Ney 4-gram | 3 characters | 27,819 counts | 1.88 |
| GRU-64 | 64 units | 21,410 | 1.84 |
| Kneser-Ney 5-gram | 4 characters | 91,249 counts | 1.62 |
| GRU-128 | 128 units | 67,362 | 1.57 |

Reading it:
- **Three numbers of memory are worth about nothing.** The Lorenz node lands at the bigram level or worse. Its waveform mixes the current character with the node's own state, so the readout reads even the current character worse than it reads the bare mask. More chaos is worse (rho 10 to 28: 3.13 to 3.52), and a ten times stronger drive does not rescue it: the best Lorenz setting of all is rho = 1, where the node barely oscillates (3.06). The anchor's best setting is the least oscillatory one. Decaying the mask along the 500 samples did not help either.
- **The neurons that help are stored values.** The tanh node improves with more samples only because its delay line holds N values in flight. The state is the delay line, not the node: 500 samples means 500 numbers of state.
- **Letting the wave learn to route helps a lot** (2.82 to 2.01; the learned gain went from 0.9 to 2.1, past the edge of stability), but it still trails a GRU-64 that trains fewer numbers (1.84) and a 4-gram (1.88).
- This matches the literature: Köster & Uchida (2026) found echo state networks with 15K-153K trained readout parameters behind a small transformer on Shakespeare characters.

## Test 4: text into Gaussian noise, coarse to fine (`holographic.py`)

A chunk of K characters becomes 500 real samples. Each character has a random signature and each position a random unitary key, both from one seed, and the chunk is the sum over positions of amplitude × (key circularly convolved with signature). Decoding either correlates with each key and picks the nearest signature (matched filter), or decodes the strongest position first and subtracts it before reading the next (successive cancellation: coarse to fine). Three amplitude schedules along the K positions, all at equal total power: flat, linear decay (the follow-up's proposal), and geometric decay solved so every position sees the same signal-to-interference ratio after cancellation.

The output looks Gaussian: kurtosis 2.97 to 3.03 at every K (a Gaussian has 3.0). Nothing is trained.

Longest chunk decoded at ≤ 1% symbol error (200 real chunks per cell):

| schedule, decoder | no noise | 20 dB | 10 dB |
|---|---|---|---|
| flat, matched filter | 20 chars | 20 | 20 |
| flat, cancellation | 20 | 20 | 20 |
| linear decay, matched filter | 0 | 0 | 0 |
| linear decay, cancellation | 40 | 20 | 10 |
| geometric, matched filter | 0 | 0 | 0 |
| geometric, cancellation | 320 (1,628 bits) | 60 (305 bits) | 40 (203 bits) |
| Shannon capacity of 500 samples | unbounded | 1,665 bits | 865 bits |

Reading it:
- Decaying influence only pays if decoding is coarse to fine. Read every position at once (matched filter) and the faint positions drown.
- Linear decay doubles the noiseless capacity, then loses under noise because its tail goes to zero. Geometric decay is the right shape: successive decoding needs each position to outweigh everything after it combined, by a fixed margin, and that forces a constant ratio from one position to the next. That is the definition of positional digits (each digit outweighs all later digits together), of RVQ's shrinking residuals, and of arithmetic coding's shrinking intervals. Without noise it packs 320 characters into 500 samples, limited only by float precision (the same wall as Test 1).
- Coarse to fine does not remove the chain; it reorders it. Level k is decoded after the levels before it. With geometric amplitudes along positions, scale order is position order, so this is left to right in disguise. The chain rule p(x) = Π p(x_k | x_<k) holds in any order. Diffusion and coarse-to-fine models pick a different order, not none.
- This is transmission, not prediction. The code says nothing about which text is likely. CALM needed its ~75M-parameter VAE for robustness (its latent still decodes at >99.9% under Gaussian noise of σ ≈ 0.3), and the hard part after that, the energy head that predicts the next latent, is the same with a fixed code. Under any model, the expected next code is a superposition of every continuation weighted by its probability: a literal wave function whose measurement (decoding) gives per-position marginals. That multimodality is exactly what an energy head has to resolve.

## Test 5: a wave through every weight (`wave_weights.py`)

The second follow-up, applied to weights: a 128-unit GRU (67,362 weights) whose weights are a random draw from one seed plus d waves along the flattened weight index. Each wave is a sinusoid at a random frequency, computed in one inverse FFT, and only the d wave coefficients are trained. Compared with fully trained GRUs of about the same trained count. Same data, steps (3,000 × 64 × 128 characters) and evaluation as Test 3.

| model | trained numbers | weights it runs | val bpc |
|---|---|---|---|
| GRU-128 in 1,024 waves | 1,024 | 67,362 | 2.91 |
| GRU-8 | 1,362 | 1,362 | 2.87 |
| GRU-16 | 3,074 | 3,074 | 2.51 |
| GRU-128 in 4,096 waves | 4,096 | 67,362 | 2.49 |
| GRU-32 | 7,650 | 7,650 | 2.14 |
| GRU-128 in 16,384 waves | 16,384 | 67,362 | 2.11 |
| GRU-64 | 21,410 | 21,410 | 1.84 |
| GRU-128 | 67,362 | 67,362 | 1.57 |

Reading it: a wave coefficient is worth at most one ordinary parameter, and less as d grows (16,384 waves land with a 7,650-weight GRU). The wave does not make knowledge cheaper to store, and it makes it more expensive to run, because the full 67K-weight network still computes every step. That fits the 2 bits per parameter picture: a trained number holds about the same amount whatever basis it is written in. Caveat: the wave arms used one learning rate (0.03, untuned), so the gap at large d may be partly optimization.

## Verdict

My read, for you to decide: not worth continuing as a way to make language models cheaper. Every test points the same way: what a result is worth tracks how many stored, trained numbers stand behind it, not the substrate (map, oscillator, delay line, wave).

1. One number holds a sentence, and X, Y, Z hold a paragraph, but only under a partition that stores 176K to 4.1M numbers. The anchor alone buys 0.7 bits per character (Test 1).
2. Three fast weights can be fit to spell about 3 characters (Test 2). Smooth figures run out of distinct shapes, chaotic figures have the shapes but gradients cannot find them, and a coarse-to-fine search gets one character further before the shrinking cells beat it.
3. As 500 neurons, one oscillator's 500 samples behave like this: a 3-number memory is bigram-level, a delay line turns 500 stored values into the state, and learned routing reaches 2.01 bpc where a GRU training fewer numbers reaches 1.84 (Test 3).
4. Text into Gaussian noise is free and exact, and geometric coarse-to-fine scaling packs 3× more into 500 samples at 20 dB, but it is a codec: prediction is untouched (Test 4).
5. A wave through every weight makes each coefficient worth at most one parameter (Test 5).

The best version of each piece got a fair shot: learned routing, a curriculum and a refinement search, a drive-strength check. This is a verdict on this architecture as a way to cut parameter cost. It is not a verdict on oscillation as a substrate for computation, which the learned arm of Test 3 and the reservoir literature support for small tasks.

## What is worth keeping

1. **Count bits, not parameters.** The cost question becomes bits(model) + bits(text | model). Knowledge comes to about 2 bits per trained number however it is expressed (Allen-Zhu & Li; Test 5), and quantization already spends most of the headroom (SeedLM keeps 97.9% of Llama 3 70B's zero-shot accuracy at 4 bits per weight). Cheaper models need fewer bits (better data efficiency, or retrieval that keeps the corpus outside the weights, like Infini-gram's 47% next-token accuracy with no parameters at all), or cheaper storage for the bits they need. A denser substrate is not on that list.
2. **Coarse to fine wants geometric decay and cancellation.** For Praxis's coarse-to-fine latents (Abstractinator RVQ, byte-latent-j's ladder), per-level residual magnitudes should fall by a roughly constant ratio, and decoding should subtract coarse levels before reading fine ones. A ladder whose level norms do not shrink geometrically is not separating scales. Cheap diagnostic on an existing checkpoint: residual RMS by level.
3. **Non-causal costs a chain somewhere.** byte-latent-j's "words without phrases" (32 finest codes drawn at once) is the wall Test 4 shows from the other side: drawing many symbols at once loses their joint dependency, and the 4 rounds put a chain back. Coarse to fine chooses the order; it cannot remove it.
4. **A filter for tiny dynamical systems.** Before building one, ask for its entropy rate (capacity per step) and its gradient horizon (about 1/λ). Capacity that grows with length needs positive entropy, and positive entropy is what makes gradients explode. A design that needs both needs a constructive encoder (like arithmetic coding), not gradient descent.
5. **Time buys computation, not knowledge.** The honest home of the oscilloscope intuition is recurrent depth and looping, already in Praxis: reusing weights over more steps buys reasoning-like computation (Saunshi et al. 2025), not storage.

## Not tested

- **Sliding-window capsules over the samples.** Capsule routing coefficients are computed per input, so they are fast weights by construction; the pose transforms are slow parameters. Over samples that are a deterministic function of four numbers, capsules can only redescribe what those numbers already fix, so they would act as the decoder, which is where the knowledge (and the parameters) would end up. Test 3's linear readout plays that role, and a capsule readout is a bigger one. The capsule work already in Praxis (byte-latent-k and -l) is the better place to ask capsule questions.

## Round 2: outside the parameters

The follow-up: "With basically any of these approaches, more neurons fixes the problem. That's the universal approximation theory. I am trying to find ways to encode information outside of parameters." Four candidates: the ghost construction applied to the samples (sliding window or K-selection), imaginary numbers, a new encoding of text (a synthesis function that turns text into objects), and new clustering (s-means).

### Five places a bit can come from

Every bit a model emits about the next character is copied from somewhere. Nothing in the machinery can create one.

| source | how bits get in | what it costs | examples |
|---|---|---|---|
| stored numbers | written by training | memory, and compute on every step | weights, n-gram counts, centroids |
| the record | read at inference | storage and lookup | kNN-LM, Infini-gram, RETRO; Test 6 |
| the input | read from the context | context length | attention, fast weights, SSM state, a cache |
| the architecture | written by the designer | design effort; pays only if the structure matches the data | convolution's translation symmetry, RoPE, a copy mechanism |
| computation | derived at run time | time | arithmetic, search; only what a short program can derive |

Randomness (seeds, random features, chaos) adds entropy, never information about the text. Universal approximation is a statement about the first row: enough stored numbers can hold any function. It says nothing about the other four, and Tests 1-5 all lived in the first row.

### Why more neurons keeps winning

The theory answers at both ends, and Tests 1-5 sit between them.

- **Without noise, one number is enough.** A finite recurrent net with rational weights is Turing complete (Siegelmann & Sontag 1995), and with real weights it computes beyond Turing machines (Siegelmann & Sontag 1994), because one real weight can hold an infinite table. Vardi, Yehudai & Shamir (2022) memorize N points with Õ(√N) parameters, but only with Õ(√N)-bit weights, and they prove that precision is necessary. The bits move into the precision; they do not go away.
- **With any Gaussian noise, a bounded window.** A recurrent analog net whose units carry Gaussian (or similar) noise recognizes exactly the definite languages, where membership is decided by a bounded suffix (Maass & Sontag 1999, Theorem 1.1; the class is Rabin's 1963 definite languages). That is the class an n-gram decides. Test 3's Lorenz node at the bigram level is this theorem in practice, and Test 1's 11 to 29 characters per float64 is the same wall at finite precision.
- **Measured.** Trained transformers hold about 2 bits of knowledge per parameter (Allen-Zhu & Li 2024) and about 3.6 bits of raw capacity (Morris et al. 2025).
- **What language asks for.** Mollica & Piantadosi (2019) put an adult English speaker's linguistic knowledge at about 12.5M bits (best guess): roughly 12M of lexical semantics, 400K of word forms, 80K of word frequencies, and only about 700 bits of syntax (range 134 to 1,394). The structure would fit in 22 float32s. The rest is an arbitrary table, and an arbitrary table can only be stored or looked up.

So more neurons fixes the problem because neurons are the default place to store a table. Information outside the parameters has to come from the other rows: the record, the input, the architecture.

### The four ideas on that map

1. **Ghost transforms on the samples.** Non-degenerate means invertible: the transform loses nothing, and by the data processing inequality it adds nothing either. On weights, ghosting saves stored numbers because weights are what is stored (byte-latent-h: 49.8% fewer parameters, trailing its twin by 0.045 to 0.051 bits at steps 9k to 11k). On samples nothing is stored, so there is nothing to save. The complex algebra's P_k are signed permutations, so they are orthogonal: the ghost key [z; iz] has exactly twice the squared distances of z, and therefore the same nearest neighbors; a linear readout on it spans the same maps. K-selection over fixed features moves the information into the selection, log2 C(N, K) bits: a randomly weighted network trained only by choosing which connections to keep can match a smaller trained one (Ramanujan et al. 2020), at the price of more weights. Test 6 checks the ghost key directly.
2. **Imaginary numbers.** A complex weight is two real numbers under a constraint, which is ghosting at d = 2 (`ghost_all_complex` is exactly this). A complex state is where they pay: the phase records when, the amplitude records what. A real decay can only trade memory length against order: with slow decay, "ab" and "ba" sum to nearly the same state. Ran-Milo et al. (2024) prove the gap: a real diagonal state space model needs a much higher dimension, or exponentially large parameter values, to express mappings a complex one expresses with moderate values. Mamba's authors still found real values fine for text once the recurrence is trained and input-dependent (Gu & Dao 2023). A wavefunction does not escape counting either: n qubits carry 2^n complex amplitudes, but at most n classical bits can be read back out (Holevo 1973).
3. **A new encoding of text.** Compression is the explored space, and the bound is the entropy. The best synthesis function from noise to text is a language model driving an arithmetic decoder (Test 1): exact, invertible, and exactly as good as the model's bits per character. A fixed synthesis from text to an object (a shape, a point in a space) is a parameter-free encoder, and Test 6's keys are exactly that: the complex oscillator bank is a sliding, decaying Fourier transform of the text, turning each history into a point where similar histories land close together.
4. **New clustering (s-means).** s-means (Kocabay, Yüksel & Yalçın 2026) is mini-batch k-means with D^1.5 seeding on a subsample, dead clusters revived by splitting the clusters with the most distortion, and empirical-Bayes shrinkage of the final centroids. It reports about 54% lower test MSE than e-means (Bessa et al. 2026) at a 3-epoch budget on its GPU benchmarks, 20% on 12 CPU datasets, and 2-3% on its one real GPU dataset. Centroids are stored numbers, so a better clusterer gives a better partition per stored number, not a new place to keep bits. Test 6 checks whether lower MSE turns into fewer bits per character.

### Test 6: the oscillator as a needle, the text as the record (`needle.py`)

Tests 1-5 asked a few numbers to hold the language. Test 6 asks them only to point at it. An untrained oscillator, drawn from a seed, turns every position of the training text into a key: 128 numbers summarizing the recent history. To predict, compute the key for the current position, find the 64 training positions with the nearest keys, and read which characters followed them, weighted by softmax(-d²/T). This is the sliding window plus K-selection idea aimed at the training text instead of the oscillator's own samples, and it is kNN-LM (Khandelwal et al. 2020) with an untrained key. The knowledge sits in the record, which is the training text itself; the keys can be recomputed from it.

Same data and scoring as Test 3 (record: the 1M training characters; 167K scored predictions). The fitted scalars (T, a unigram floor, a mixing weight) come from a separate validation slice.

| key (128 numbers per position) | trained numbers | as features for a trained softmax (4,386 weights) | as an address into the record | address + Kneser-Ney 6 |
|---|---|---|---|---|
| real decay, 8 rates × 16 units | 0 | 3.11 | 1.99 | 1.468 |
| real decay, 128 distinct rates | 0 | 3.15 | 1.97 | 1.464 |
| complex (decay and rotation), 8 rates × 8 phases | 0 | 3.05 | **1.52** | 1.394 |
| the same, ghost-expanded to [z; iz] | 0 | 3.06 (8,738 weights) | 1.52 (identical to 6 digits) | 1.394 |
| echo state network, 128 units | 0 | 2.71 | 1.54 | 1.408 |
| Lorenz node, 128 samples of x | 0 | 3.78 | 2.98 | 1.491 |
| GRU-64 hidden state | 21,410 | 1.84 (its own softmax) | 1.46 | **1.382** |

For scale (Test 3, same data): Kneser-Ney 6 1.489 (220K counts), GRU-128 1.567 (67K trained), GRU-64 1.841.

More record, complex key, nothing trained:

| record | 30K | 100K | 300K | 1M characters |
|---|---|---|---|---|
| address | 2.34 | 2.05 | 1.87 | 1.52 |
| address + Kneser-Ney 6 | 2.09 | 1.83 | 1.68 | 1.39 |
| Kneser-Ney 6 alone | 2.13 | 1.87 | 1.73 | 1.49 |

The record compressed into K centroids, each carrying next-character counts (complex key; stored numbers = K × (128 + 34)):

| K | stored numbers | k-means, 3 passes | k-means, 25 passes | s-means, 3 epochs |
|---|---|---|---|---|
| 256 | 41K | 2.64 | 2.54 | 2.53 |
| 1,024 | 166K | 2.19 | 2.11 | 2.10 |
| 4,096 | 664K | 1.86 | 1.79 | 1.78 |
| 16,384 | 2.65M | 1.65 | 1.60 | 1.59 |

Reading it:
- **The same numbers are a poor store and a good address.** As features for a trained readout, the complex key is worth 3.05 bpc; as an address it is worth 1.52 with nothing trained, ahead of a 67K-weight GRU-128 (1.57) on the same data. The oscillator never holds the language. It says where in the text the current moment has happened before.
- **Imaginary numbers: the phase is the address.** Complex beats real by 0.47 bpc with the same decay rates and the same 128 numbers, and 128 distinct real rates recover almost none of it (1.97). This is the largest single effect in this thread. With decay alone, a slowly decaying unit sums "ab" and "ba" to nearly the same state; a rotation stamps each character with when it arrived. The echo state network nearly matches the complex bank (1.54), and it rotates too: a dense random real matrix has complex eigenvalues spread over a disk (the circular law). Only a diagonal real recurrence has no phase.
- **Ghost on the samples changes nothing, exactly.** [z; iz] doubles every squared distance, so the neighbors and the result are identical to 6 digits; the linear readout spans the same maps with twice the weights. Ghosting saves numbers only where numbers are stored.
- **Chaos is a bad address.** The Lorenz key lands between the bigram and trigram levels and adds nothing to Kneser-Ney: histories that end the same way land far apart.
- **The soft address adds to the exact one.** Mixed with Kneser-Ney 6 (exact suffix matches from the same record), the complex address takes 1.489 to 1.394. Both readers are untrained, with 3 fitted scalars between them.
- **A trained key is only a little better, and its own softmax is much worse.** The GRU-64's states as the address give 1.46: 0.38 better than the GRU's own prediction (1.84), and only 0.06 better than the untrained complex key. The network learned which moments are alike, but its 2,210 output weights cannot hold what followed them, and the record can. This is kNN-LM's result in miniature.
- **The record keeps paying.** Each tripling of the record takes 0.2 to 0.35 bpc off, with no sign of flattening at 1M characters. More text plays the role more neurons played.
- **Clustering the record is a poor way to compress it, whatever the clusterer.** s-means does what it claims: 3 epochs match 25 passes of k-means (16 times faster at K = 16,384) and beat 3 passes by 0.05 to 0.1 bpc. Against a converged k-means it gains 0.006 to 0.015 bpc. At every size the centroids do worse per stored number than Kneser-Ney's counts or a trained GRU, and even 2.65M of them trail the raw record. Summaries lose the fine distinctions prediction needs.
- **What it costs.** The record is 967K entries, 14 times GRU-128's weights, and every prediction searches it (here, exact search over 1M keys on a GPU). Information outside the parameters is not free: it is stored as text instead of as trained weights. What that buys: no training, a store that grows by appending text, and knowledge that can be deleted or edited one story at a time. TinyStories is synthetic and repetitive, which flatters any record; on web text a record has to be far larger to compete (Infini-gram needed trillions of tokens).

### What round 2 changes

The verdict above stands for the notion as first stated: four numbers cannot hold a language. Round 2 finds the version that works. An oscillator is a good needle and a bad record. Drop the requirement that it hold anything, keep the text as the record, and an untrained complex oscillator predicts TinyStories better than any small network trained on the same text here. That is a place outside the parameters: the record, read through a rotating memory. The universal approximation instinct ("more neurons") is replaced by "more text", which costs storage and search instead of training.

If you continue, the open questions in order of cost:
1. **How few trained numbers does a good address need?** Train only the bank's decay rates and phases (16 to 128 numbers) against the kNN loss: the wave learns to route through the record. The bars are the untrained bank (1.52) and the GRU-64 key (1.46).
2. **Does it keep scaling?** A 10M-character record (Kneser-Ney 9 reaches 1.22 at that size), which needs approximate search or smaller keys.
3. **A later bridge to Praxis:** the harmonic latent's frozen phases are a fixed complex basis, the kind of address Test 6 favors.

## Round 3: one system (`staging/oscillator_store/`)

The follow-up: build the best approach as one standalone example, with the store indexed by s-means, the learnable oscillator, and SMEAR-routed tiny experts (hidden 4, 1,024 of them). Its README has the details and the reading. Headline, on Test 3's evaluation set with a 9.7M-character store:

| readers, mixed by a gate of 34 to 60 numbers | trained numbers | bpc |
|---|---|---|
| Kneser-Ney 9 alone | 0 | 1.167 |
| store alone | 4,480 | 1.148 |
| GRU-256 alone | 233K | 1.148 |
| store + n-gram | 4,514 | 1.081 |
| store + n-gram + tiny experts | + 835K | 1.067 |
| store + n-gram + GRU | + 233K | **1.031** |

- The best approach is the kNN-LM pairing: store, n-gram and a small GRU, 0.12 below the GRU alone. Without any trained network, the store and n-gram already beat it.
- The tiny experts lose to the GRU as the parametric reader: alone 1.46 with 835K trained numbers, and 0.0005 added next to the GRU. Merging 1,024 experts costs about 700K multiply-adds per position even though only 4 activations come out.
- Training the oscillator helps it alone (1.516 to 1.41-1.43 at 1M) and adds nothing measurable next to the n-gram, whether trained alone or through the mixture.
- s-means works as the index (99.98% recall while searching 0.8% of the store), and so does k-means; s-means' gain is build time.
- If this continues: the exchange rate between store and weights (GRU-16, -64 and -256, each with and without the store), text that repeats less, and longer memory (the sample is fluent a clause at a time and forgets the story).

## Round 4: causality, graphs, X, Y, Z, and real text

The follow-up: "this doesn't look super promising." The GRU brings the causal dependency back ("a transformer has the same problem; anything built in this way is going to need parameters to perform"), coarse to fine was meant to avoid it but is hard to train, and the new idea is "graphs over sets/clusters, with sort of a learned Markov routing, convolutions to model the graph", in the user's words "random terms, I don't fully understand what I'm asking for". Two more questions: was the oscillator modeled with X, Y, Z interference and harmonics, and FineWeb is available.

### Causality is bookkeeping, not cost

- Any model of p(text) can be written left to right, right to left, or coarse to fine, and every order gives the same total bits (Test 4). The bits belong to the text and to what the model knows, not to the order.
- Non-causal models pay more, not less: masked diffusion language models follow the same scaling law as autoregressive ones at about 16 times the compute (Nie et al. 2025).
- What non-causal buys is flexibility: infilling, constraints, editing. The same masked diffusion model breaks the reversal curse that much larger autoregressive models have.
- The store + n-gram system has no causal network at all (1.081 on TinyStories, 1.903 on FineWeb); the GRU was one optional reader. Every next-character predictor conditions on the past by definition, but a store can be read with right context just as easily, for infilling.

### What "graphs over clusters with learned Markov routing" is

Group the moments of the text into clusters that mean the same thing for what comes next ("just after 'once upon a'", "inside a quotation, expecting a verb"). Each cluster is a node. Draw an arrow from A to B labeled c whenever a moment in A is followed by character c and lands in B, and count how often: that is the Markov routing. To generate, stand on a node, take an arrow by its probability, emit its label, move. To predict, keep a belief over nodes and update it with each character. "Convolutions on the graph" let each node borrow statistics from its neighbors, so rare nodes still predict sensibly.

- **It has a name and a theory.** It is a hidden Markov model, or probabilistic automaton. When the clusters are exactly the classes of pasts that predict the same future, the graph is the epsilon-machine, the smallest graph that predicts optimally (Crutchfield & Young 1989; Shalizi & Crutchfield 2001). CSSR builds it by splitting clusters until their futures agree (Shalizi & Klinkner 2004). Clone-structured cognitive graphs learn such graphs by expectation-maximization, with several "clones" per observed symbol (George et al. 2021). HMMs with 2^15 states beat n-grams and approach, without reaching, neural models on Penn Treebank (Chiu & Rush 2020). The convolution is smoothing across neighboring nodes, which is what Kneser-Ney does across context lengths.
- **The store is already this graph, at full resolution.** Its nodes are the 9.4M stored moments, with two kinds of edges: "similar to" (the oscillator's nearest neighbors) and "followed by" (the text itself). The store reader hops along "similar to" and reads the "followed by" edges. Clustering the nodes with s-means gives the coarse graph, and Test 6 showed that a coarse graph without routing loses to the fine one. The n-gram is the same object again, with nodes that are exact suffixes.
- **What graphs buy is exact non-causal inference.** The forward-backward algorithm gives the probability of any character given everything before and after it, exact infilling, and sampling under constraints, none of which a transformer or GRU offers exactly. GeLaTo distills GPT-2 into an HMM for this reason, to guarantee lexical constraints (Zhang et al. 2023).
- **What they cannot escape is that the node is the only memory.** A graph with K nodes carries at most log2 K bits from the past to the future. Hilberg's conjecture, which Dębowski's estimates support, is that the information shared by adjacent blocks of text grows as a power of their length (Dębowski 2011), so the graph has to keep growing to keep improving. It is a different storage format with the same wall.

### Test 7: X, Y, Z with interference as the address (`xyz_address.py`)

Three complex oscillators kicked by each character and left to ring for one character period under three-wave coupling (dX/dt = (iω_x - γ_x)X + ikYZ and its two partners, with ω_x = ω_y + ω_z, so the coupling makes sum and difference tones). Decay timescales of 2, 8 and 32 characters; 64 samples per character. Keys are the samples of Re X, Re Y, Re Z without coupling (k = 0) and with it (k = 1), and the "screen" |X + Y + Z|², the interference pattern itself. Same readout as Test 6, on FineWeb: a 1M-character record, 64 nearest neighbors.

| key | numbers per position | as an address | address + Kneser-Ney 9 |
|---|---|---|---|
| X, Y, Z, no coupling | 192 | 3.92 | 2.221 |
| X, Y, Z with harmonics | 192 | 4.11 | 2.221 |
| interference pattern \|X + Y + Z\|² | 64 | 4.29 | 2.223 |
| Test 6's bank of 64 rotations | 128 | 2.42 | 2.179 |

For scale on the same record: Kneser-Ney 9 2.235, GRU-128 2.401, unigram 4.26.

Reading it: X, Y, Z barely beat the unigram, and interference makes it worse at each step (3.92, 4.11, 4.29). Three complex numbers are six numbers of memory, so the slow mode's accumulated kicks outweigh the current character. Harmonics are functions of those same six numbers, so they reshape the distances without adding anything to them, and the coupling's mixing scrambles which histories land close. We did model the coupled X, Y, Z version (Lorenz in Tests 2, 3 and 6, and now three-wave mixing); the bank works because it has 64 rotations, not because it lacks interference.

### The system on FineWeb

`staging/oscillator_store` on 9.7M characters of FineWeb (README there):

| readers | bpc |
|---|---|
| Kneser-Ney 9 alone | 1.958 |
| store alone | 2.012 |
| GRU-256 alone | 1.972 |
| store + n-gram | 1.903 |
| store + n-gram + GRU | 1.826 |
| all four | 1.823 |

In the full system, removing the store costs 0.009 bits on FineWeb, against 0.063 on TinyStories. The n-gram (+0.059) and the GRU (+0.044) carry the system. At this data size a 233K-weight GRU only ties a Kneser-Ney model, so even the "store + n-gram beats the GRU" result (1.903 against 1.972) is the small-data regime, where counts are hard to beat.

### Verdict after round 4

Agreed: not promising as a way to cut the parameters a language model needs. Seven tests and one full system say the same thing. What a model knows is stored somewhere, and the store looked like an exception only because TinyStories repeats itself. Additions to "What is worth keeping":

6. **The phase is the address.** A fixed memory of complex rotations keeps order apart (1.52 against 1.99 for real decay, Test 6). It is the reason Praxis's frozen-phase harmonic basis is a reasonable fixed basis, and the thing to reach for whenever a fixed key or position code is needed.
7. **Kneser-Ney is a strong zero-parameter baseline at small data.** On 10M characters of web text it ties a GRU-256. Worth running next to any small-model experiment before reading its numbers.
8. **Graphs buy exact non-causal inference, not fewer bits.** If the non-causal wish is about capability (infill, constraints, editing) rather than cost, a count-based graph is the honest place for it. The test would be an HMM over the store's s-means clusters, or a clone-structured graph, scored on exact infilling: a capability test, not a cost test.

## References

- Aghajanyan, A., Gupta, S. & Zettlemoyer, L. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. ACL 2021, 7319-7328. arXiv:2012.13255.
- Allen-Zhu, Z. & Li, Y. (2024). Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. ICLR 2025. arXiv:2404.05405.
- Appeltant, L. et al. (2011). Information processing using a single dynamical node as complex system. Nature Communications 2:468. doi:10.1038/ncomms1476.
- Ba, J., Hinton, G., Mnih, V., Leibo, J. & Ionescu, C. (2016). Using Fast Weights to Attend to the Recent Past. NeurIPS 2016.
- Balakrishnan, H. N., Kathpalia, A., Saha, S. & Nagaraj, N. (2019). ChaosNet: A chaos based artificial neural network architecture for classification. Chaos 29:113125.
- Bessa, S., Mazaré, P.-E., Szilvasy, G., Touvron, H., Picard, D., Douze, M. & Jégou, H. (2026). e-means: outperforming k-means (in one-tenth the epochs). arXiv:2611.09501.
- Borgeaud, S. et al. (2022). Improving language models by retrieving from trillions of tokens. ICML 2022. arXiv:2112.04426.
- Chiu, J. T. & Rush, A. M. (2020). Scaling Hidden Markov Language Models. EMNLP 2020, 1341-1349. arXiv:2011.04640.
- Crutchfield, J. P. & Young, K. (1989). Inferring statistical complexity. Physical Review Letters 63(2):105-108.
- Cuomo, K. M. & Oppenheim, A. V. (1993). Circuit implementation of synchronized chaos with applications to communications. PRL 71:65-68.
- Cuomo, K. M., Oppenheim, A. V. & Strogatz, S. H. (1993). Synchronization of Lorenz-based chaotic circuits with applications to communications. IEEE TCAS-II 40(10):626-633.
- Dębowski, Ł. (2011). Excess entropy in natural language: present state and perspectives. Chaos 21:037105. arXiv:1105.1306.
- Eldan, R. & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759.
- Equitz, W. & Cover, T. (1991). Successive refinement of information. IEEE Trans. Inf. Theory 37(2):269-275.
- Gauthier, D. J., Bollt, E., Griffith, A. & Barbosa, W. A. S. (2021). Next generation reservoir computing. Nature Communications 12:5564.
- George, D., Rikhye, R. V., Gothoskar, N., Guntupalli, J. S., Dedieu, A. & Lázaro-Gredilla, M. (2021). Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps. Nature Communications 12. doi:10.1038/s41467-021-22559-5.
- Girko, V. L. (1985). Circular law. Theory of Probability and Its Applications 29(4):694-706.
- Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
- Hayes, S., Grebogi, C. & Ott, E. (1993). Communicating with chaos. PRL 70:3031-3034.
- Hess, F., Monfared, Z., Brenner, M. & Durstewitz, D. (2023). Generalized Teacher Forcing for Learning Chaotic Dynamics. ICML 2023. arXiv:2306.04406.
- Holevo, A. S. (1973). Bounds for the quantity of information transmitted by a quantum communication channel. Problems of Information Transmission 9(3):177-183.
- Islam, S. M. R., Avazov, N., Dobre, O. A. & Kwak, K. S. (2017). Power-Domain Non-Orthogonal Multiple Access (NOMA) in 5G Systems. IEEE Commun. Surveys & Tutorials 19(2):721-742.
- Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L. & Lewis, M. (2020). Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020. arXiv:1911.00172.
- Kocabay, Ş. T., Yüksel, K. A. & Yalçın, K. (2026). s-means: outperforming e-means (which outperforms k-means). suayptalha.github.io/s-means-blog; code at github.com/suayptalha/s-means.
- Köster, F. & Uchida, A. (2026). Reservoir computing as a language model. Phys. Rev. Applied 26:024051. arXiv:2507.15779.
- Ledrappier, F. & Young, L.-S. (1985). The metric entropy of diffeomorphisms, Part I: Characterization of measures satisfying Pesin's entropy formula. Annals of Mathematics 122:509-539.
- Li, C., Farkhoor, H., Liu, R. & Yosinski, J. (2018). Measuring the Intrinsic Dimension of Objective Landscapes. ICLR 2018. arXiv:1804.08838.
- Liu, J., Min, S., Zettlemoyer, L., Choi, Y. & Hajishirzi, H. (2024). Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens. COLM 2024. arXiv:2401.17377.
- Maass, W. & Sontag, E. D. (1999). Analog neural nets with Gaussian or other common noise distributions cannot recognize arbitrary regular languages. Neural Computation 11(3):771-782. Conference version: A precise characterization of the class of languages recognized by neural nets under Gaussian and other common noise distributions, NIPS 1998.
- Mikhaeil, J., Monfared, Z. & Durstewitz, D. (2022). On the difficulty of learning chaotic dynamics with RNNs. NeurIPS 2022. arXiv:2110.07238.
- Mollica, F. & Piantadosi, S. T. (2019). Humans store about 1.5 megabytes of information during language acquisition. Royal Society Open Science 6:181393.
- Morris, J. X. et al. (2025). How much do language models memorize? arXiv:2505.24832. ICML 2026 as How much can language models memorize?
- Nagaraj, N., Vaidya, P. G. & Bhat, K. G. (2009). Arithmetic coding as a non-linear dynamical system. Commun. Nonlinear Sci. Numer. Simul. 14(4):1013-1020.
- Nie, S. et al. (2025). Scaling up Masked Diffusion Models on Text. ICLR 2025. arXiv:2410.18514.
- Nooralinejad, P. et al. (2023). PRANC: Pseudo RAndom Networks for Compacting deep models. ICCV 2023. arXiv:2206.08464.
- Piantadosi, S. T. (2018). One parameter is always enough. AIP Advances 8:095118.
- Plate, T. A. (1995). Holographic reduced representations. IEEE Trans. Neural Networks 6(3):623-641.
- Rabin, M. O. (1963). Probabilistic automata. Information and Control 6(3):230-245.
- Ramanujan, V., Wortsman, M., Kembhavi, A., Farhadi, A. & Rastegari, M. (2020). What's Hidden in a Randomly Weighted Neural Network? CVPR 2020. arXiv:1911.13299.
- Ran-Milo, Y., Lumbroso, E., Cohen-Karlik, E., Giryes, R., Globerson, A. & Cohen, N. (2024). Provable Benefits of Complex Parameterizations for Structured State Space Models. NeurIPS 2024. arXiv:2410.14067.
- Saunshi, N., Dikkala, N., Li, Z., Kumar, S. & Reddi, S. J. (2025). Reasoning with Latent Thoughts: On the Power of Looped Transformers. ICLR 2025. arXiv:2502.17416.
- Shafipour, R. et al. (2025). SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators. ICLR 2025. arXiv:2410.10714.
- Shalizi, C. R. & Crutchfield, J. P. (2001). Computational mechanics: pattern and prediction, structure and simplicity. Journal of Statistical Physics 104:817-879.
- Shalizi, C. R. & Klinkner, K. L. (2004). Blind construction of optimal nonlinear recursive predictors for discrete sequences. UAI 2004.
- Shao, C., Li, D., Meng, F. & Zhou, J. (2025). Continuous Autoregressive Language Models. arXiv:2510.27688.
- Siegelmann, H. T. & Sontag, E. D. (1994). Analog computation via neural networks. Theoretical Computer Science 131(2):331-360.
- Siegelmann, H. T. & Sontag, E. D. (1995). On the computational power of neural nets. Journal of Computer and System Sciences 50(1):132-150.
- Sun, Y. et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620.
- Torrejon, J. et al. (2017). Neuromorphic computing with nanoscale spintronic oscillators. Nature 547:428-431.
- Vardi, G., Yehudai, G. & Shamir, O. (2022). On the Optimal Memorization Power of ReLU Neural Networks. ICLR 2022. arXiv:2110.03187.
- Vieira Neto, G. & Valle, M. E. (2026). Ghost Features and Spooky Transfer Learning for Hypercomplex-Valued Neural Networks. arXiv:2608.07735.
- Zhang, H., Dang, M., Peng, N. & Van den Broeck, G. (2023). Tractable Control for Autoregressive Language Generation. ICML 2023. arXiv:2304.07438.
