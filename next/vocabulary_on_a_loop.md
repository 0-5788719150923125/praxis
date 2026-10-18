# A vocabulary on a loop: the string idea, measured

> Status: **idea assessed 2026-10-08 in two rounds, nothing built in Praxis.**
> Round 1 tested the loop as a drop-in classifier on a 4-token MLP language
> model. Round 2 ([below](#round-2-the-architecture-around-it)) put it in a
> small transformer at the paper's scale and tested the architectural
> suggestions: a router, inputs on the loops, the paper's structure-plus-delta
> split, and scarce data. Round 2 overrules round 1's recommendation. Every
> number comes from throwaway scripts on minipile, not from a Praxis run.
> Siblings: [harmonic_koopman.md](harmonic_koopman.md) (the other place phases
> do the work), [periodicity_generalization.md](periodicity_generalization.md),
> [ouroboros.md](ouroboros.md) (the Mono-Forward cost objection) and
> [harmony.md](harmony.md). Not the same thing as
> `praxis/classifiers/harmonic.py`: that field runs over (position, feature)
> and scales the hidden state. This one puts the *vocabulary* on the loop.

## The picture, as first stated

Take a very long rope, tie the ends into a loop, and bunch it up in a chaotic
pile. Position along it is one float: 0.0 where you start, 0.999... just before
it closes. A Fourier series can describe the whole loop. Map tokens to intervals
along it. Now measure the classifier's weights from an origin, and let the
origin travel along the loop: the weights' directions and lengths change, not
because the weights changed, but because the origin moved. Train the model to
predict the *shape* at each step instead of the label. The motivation:
Mono-Forward shows layers can learn local label predictions, so maybe the labels
can be continuous geometry instead of indices.

## The one line

**As a classifier, no.** No loop variant in either round (about 50 runs)
matched the standard classifier. In a real transformer the gap widens: 8 loops sit 0.55
nats behind, past this note's own 0.5 falsifier. Tying the input and output
embeddings already removes the parameter cost the loops were supposed to save.
**As a map of the vocabulary, yes.** A single learned loop sorts tokens by
grammatical role: determiners together, numbers together, modals together, code
keywords together, suffixes together. That's the structure it finds. Choosing
*within* a neighborhood is the one thing it can't do. One untested use follows
from the paper: the neighborhoods could steer sampling, so temperature explores
grammatical alternatives instead of noise.

Round 1's line, kept for the record and overruled by round 2: a few learned
loops beat a plain factorization at every budget (one number per token gets 70%
of the way from unigram to a standard classifier), so the use would be wherever
the `vocab x hidden` matrix is the expensive part.

## What each piece of the picture became

| In the picture | In the model | Verdict |
|---|---|---|
| A long closed rope | a closed curve `g(t)`, `t` in `[0, 1)` | fine |
| Its Fourier description | `g(t) = sum_k a_k cos(2 pi k t) + b_k sin(2 pi k t)`, K harmonics | this *is* the classifier's parameterization |
| Bunched up, knotted, in 3D | the coefficient map; knots only exist in 3D | the knot doesn't carry over; the crumpling is just the map |
| Tokens at intervals along it | one learned phase `t_v` per token | the core; the learned order is a readable map of grammatical roles |
| Weights measured from a moving origin | translation: invisible to softmax. Along the loop: each harmonic rotates by `2 pi k tau`. From inside the loop: a router's wrapped-Cauchy view | along the loop hurts (0.06 to 0.20); the router view gains nothing reliable |
| Predict the shape, not the label | predict the Fourier coefficients of the log-probability over the loop; cross-entropy unchanged | trains fine, still behind a standard classifier; an explicit location target collapses |
| One string connected to everything | one loop; many loops = a torus | one is too thin; more help, none match a standard classifier |

## The math, short

**The classifier.** Token `v` sits at phase `t_v`. Its classifier row is the
loop's point there, `w_v = C^T phi(t_v)`, with
`phi(t) = [cos 2 pi k t, sin 2 pi k t]` for `k = 1..K`. Then

```
logit_v = h . w_v = (C h) . phi(t_v) = f_h(t_v)
```

`f_h` is a periodic function of degree K whose Fourier coefficients are `C h`.
So the model emits a *shape* over the loop, and the label is just where on the
loop the next token sits. That's the precise form of "predict the shape, not the
label". It keeps multimodality, which a regression target can't.

**What a shape can select.** A nonzero degree-K trigonometric polynomial has at
most `2K` zeros per period. So the tokens scoring above any threshold form a
union of at most K arcs of the loop, and any K arcs are reachable (place the
`2K` zeros at the arc ends). Separately, every `phi(t)` has norm `sqrt(K)`.
The features lie on a sphere, so every token is a vertex of the hull and none
can be shut out of the argmax. That's the "stolen probability" failure
([Demeter et al. 2020, arXiv:2005.02433](https://arxiv.org/abs/2005.02433)),
which [Grivas et al. 2022, arXiv:2203.06462](https://arxiv.org/abs/2203.06462)
found in 13 of 150 real models and judged rarely harmful. Nice to have, not a
reason to build.

**The ceiling.** Language doesn't fit in K arcs. "The red ___" picks apples,
cars, roses and flags, a set that crosses any ordering of the vocabulary you
choose. A 0/1 matrix whose ones can be made contiguous in every row has the
*consecutive-ones property*. Context-by-plausible-token matrices don't have it.
Spectral seriation (Atkins, Boman, Hendrickson 1998) recovers the ordering when
one exists. Here none does, so a single loop always splits some plausible sets
into many pieces.

**Moving the origin, three readings.**

1. *Translate in space*, `w_v - o`. Every logit drops by the same `o . h`, and
   softmax ignores a shared constant. It's an exact no-op. For distance
   classifiers (crystal, HALO) it equals shifting the hidden state by `o`. Only
   an angle-only readout would see it, as parallax.
2. *Slide along the loop*, `t -> t - tau`. By the Fourier shift theorem,
   harmonic `k`'s coefficient pair rotates by `2 pi k tau`. That's RoPE applied
   to the classifier's spectrum. It's the faithful version of "the weights
   change because the origin moved". The coefficients are fixed and only their
   phases turn, high harmonics fastest.
3. *This is the grokking clock.* Transformers trained on modular addition put
   tokens on circles. They rotate to `a + b` and read logits as a sum of
   `cos(w (a + b - c))`
   ([Nanda et al. 2023, arXiv:2301.05217](https://arxiv.org/abs/2301.05217)).
   Pretrained LLMs do the same thing for days and months
   ([Engels et al. 2024, arXiv:2405.14860](https://arxiv.org/abs/2405.14860))
   and put numbers on a helix
   ([Kantamneni & Tegmark 2025, arXiv:2502.00873](https://arxiv.org/abs/2502.00873)).
   So the picture describes a real mechanism, but only for vocabularies that
   really are cyclic.

**Knots.** A closed curve given by a Fourier series in 3D is a *Fourier knot*,
and every knot has one ([Kauffman 1997, arXiv:q-alg/9711013](https://arxiv.org/abs/q-alg/9711013);
the precise (1, 1, n) statement is
[Lamm 2012, arXiv:1210.4543](https://arxiv.org/abs/1210.4543); Trautwein's 1995
thesis calls them harmonic knots). In 4 or more dimensions every closed curve
can be untied, so the knot type can't matter in a 64-to-512-dim model. A 3D
classifier would be rank 3. The "chaos" doesn't survive either. With a linear
readout, the logit along the loop has degree K however crumpled the curve is.

## Prior art for the classifier itself

[Fourier Head (Gillman et al., ICLR 2025, arXiv:2410.22269)](https://arxiv.org/abs/2410.22269)
is the closest. A linear layer emits Fourier coefficients of a density on
`[-1, 1]`, which is evaluated at **fixed, evenly spaced** bin centers. It's built
for tokens that already have an order: quantized actions and time-series bins.
The authors say to use it only when neighboring bins are similar. What's new
here is two things. The positions are **learned**, for a vocabulary with no
order. And there are **several loops**. The version here also models the
log-probability (any shape, then softmax) instead of the density. A search
turned up nothing that gives each token one learned phase and uses that as the
classifier, but a negative search is weak evidence. Neighbors:
[compositional embeddings via complementary partitions (Shi et al. 2020, arXiv:1909.02107)](https://arxiv.org/abs/1909.02107)
is the hashed, linear cousin of "a few loops". Continuous-output LMs that regress
to an embedding
([Kumar & Tsvetkov 2019, arXiv:1812.04616](https://arxiv.org/abs/1812.04616))
are the regression reading, and its weak point is multimodality.

## The toy run

Same trunk for every arm. The embeddings of the 4 previous tokens (64 each) go
through an MLP (256, 512, 512, d) with GELU, then LayerNorm, then the
classifier. The data is a 4096-entry byte-level BPE trained on 49M characters
of minipile: 16.8M tokens, 97/3 split. Training is AdamW at lr 3e-3 with 200
warmup steps and cosine decay to 10%, no weight decay, batch 4096, 4000 steps
(about one epoch). Every arm has a per-token bias initialized to log-unigram.
Validation cross-entropy uses 262k held-out positions. Unigram scores 7.08 nats.
Reruns at a second seed moved five arms by at most 0.02 nats, so read any gap
over ~0.05 as real.

"Loops M x harmonics K" always has `2MK = d`, the same rank as the standard
classifier. "Plain factorization r" gives each token r free numbers used
linearly (`W = A B`). It's the control for "r numbers per token".

**Width 64**

| classifier | numbers per token | val CE | top-1 |
|---|---|---|---|
| standard | 64 | **4.66** | 20.2% |
| 1 loop x 32 | 1 | 5.69 | 12.8% |
| 1 loop x 32, seriation init, coarse-to-fine | 1 | 5.40 | 11.4% |
| 2 loops x 16 | 2 | 5.36 | 16.9% |
| 4 loops x 8 | 4 | 5.10 | 18.0% |
| 8 loops x 4 | 8 | 4.95 | 18.5% |
| 32 loops x 1 | 32 | 4.80 | 19.2% |
| plain factorization 1 | 1 | 6.49 | 5.1% |
| plain factorization 2 | 2 | 6.01 | 7.3% |
| plain factorization 4 | 4 | 5.65 | 10.5% |
| plain factorization 8 | 8 | 5.27 | 14.6% |

**Width 256**

| classifier | numbers per token | val CE | top-1 |
|---|---|---|---|
| standard | 256 | **4.55** | 21.3% |
| 1 loop x 128, seriation init, coarse-to-fine | 1 | 5.34 | 13.5% |
| 4 loops x 32, seriation init, coarse-to-fine | 4 | 4.86 | 18.8% |
| 16 loops x 8 | 16 | 4.76 | 19.7% |
| 128 loops x 1 | 128 | 4.64 | 20.3% |
| plain factorization 4 | 4 | 5.64 | 10.5% |
| plain factorization 16 | 16 | 4.99 | 17.6% |

**Ablations** (width 64)

| change | val CE | delta |
|---|---|---|
| 1 loop x 32, random init, phases learned | 5.69 | reference |
| same, phases frozen at random | 6.31 | +0.62 |
| seriation init, phases learned | 5.55 | -0.15 |
| seriation init, phases frozen | 6.08 | +0.39 |
| random init + coarse-to-fine | 5.50 | -0.20 |
| seriation init + coarse-to-fine | 5.40 | -0.29 |
| ...with 128 harmonics (projected to rank 64) | 5.38 | about the same |
| ...with 512 harmonics | 5.69 | worse |
| origin at the last input token, 1 loop (vs 5.55) | 5.69 | +0.15 |
| origin at the last input token, 4 loops (vs 5.10) | 5.26 | +0.16 |
| origin at the last input token, 8 loops (vs 4.95) | 5.11 | +0.15 (+0.16 at seed 1) |
| origin chosen by the model, 1 loop (vs 5.40) | 5.60 | +0.20 |
| origin chosen by the model, 8 loops (vs 4.95) | 5.02 | +0.06 |
| 1 loop x 1 harmonic: one bump, the best case for a regression target | 6.40 (6.16 seriation) | |

"Seriation init" sets each token's phase from a circular spectral ordering
([Recanati et al. 2018, arXiv:1807.07122](https://arxiv.org/abs/1807.07122)).
Each token gets a context profile, `sqrt(P(previous token | this token))`.
Those profiles build a 15-nearest-neighbor graph, and the phase is the angle
between two normalized-Laplacian eigenvectors. "Coarse-to-fine" is BARF's
frequency window over the first half of training
([Lin et al. 2021, arXiv:2104.06405](https://arxiv.org/abs/2104.06405)).
Harmonic `k` fades in once the schedule passes `k`. BARF introduced it because
full-bandwidth Fourier features stall coordinate learning. It helped here too,
by 0.15 to 0.20 nats.

## What the run says

1. **One loop has a ceiling, and the cause is the ordering, not the bandwidth.**
   Four times the harmonics barely moved it, and sixteen times made it worse.
   In the three single-loop arms where chance was measured, 21-35% of misses
   land within one resolution cell (`1/2K`) of the target on the loop. Chance,
   with targets shuffled across examples, is 3-6%. It finds the right
   neighborhood and can't choose inside it. That's the consecutive-ones wall.
2. **A few loops fix most of it, and Fourier coding beats linear coding at
   every budget.** One loop roughly equals 4 linear numbers per token. Four
   loops beat 16 linear numbers at width 256. Each scalar unfolding into K
   harmonics is what pays.
3. **It never beats the standard classifier at the same rank.** Treat it as
   compression with a good quality-per-parameter curve, not a better
   classifier.
4. **Moving the origin hurt every time.** Anchoring it to the current token
   assumes the next token sits at a fixed offset along the loop, which text
   doesn't do. The trunk has to learn to undo a token-specific rotation.
   Letting the model choose the origin multiplies every angle error by
   `2 pi k`. Predict the shape in a fixed frame. The fixed offset only holds for
   cyclic vocabularies (days, months, digits), where the clock already wins.
5. **Learning the positions is most of the value.** Frozen random phases lose
   0.6 nats. Phases also need coarse-to-fine to learn well.
6. **A continuous target is a non-starter.** A unimodal target on the loop,
   trained with the best possible loss (cross-entropy), recovers 28-38% of what
   the standard classifier gains over unigram. A regression loss can only do
   worse. And with targets that are themselves learned, regression also invites
   collapse: put every token at the same point, and the target is trivially
   predicted. Cross-entropy is what prevents that.

## Round 2: the architecture around it

The objection to round 1 was fair. It tested the loop as a drop-in classifier on
a 4-token MLP, and the idea might need an architecture built around it. (The
inputs were real text throughout. Only the token positions started random.)
Round 2 uses a small causal transformer at the paper's scale: 4 layers, width
128, context 128, about 1.3M parameters. It trains one epoch on the same 16.8M
tokens at lr 2e-3, batch 32 x 128, with no weight decay. Multi-loop arms start
from random phases, single-loop arms from seriation, and all use
coarse-to-fine. Unigram scores 7.08. Two arms repeated at a second seed moved
0.004 and 0.05.

| classifier | classifier params | total params | val CE |
|---|---|---|---|
| standard | 528k | 1.86M | **4.21** |
| standard, tied to the input embedding | 528k (shared) | 1.34M | **4.22** |
| 1 loop x 64 | 25k | 1.36M | 5.19 (seed 1: 5.20) |
| 4 loops x 16 | 37k | 1.37M | 4.86 |
| 8 loops x 8 | 53k | 1.39M | 4.77 |
| 11 loops x 6 | 66k | 1.40M | 4.75 |
| plain factorization 8 | 38k | 1.37M | 5.29 |
| plain factorization 16 | 72k | 1.41M | 4.93 |
| 8 loops + zero-init rank-16 delta | 121k | 1.45M | 4.54 |
| 1 loop + router | 25k | 1.36M | 5.14 (seed 1: 5.19) |
| 1 loop + router + location loss | 25k | 1.36M | 6.67 (collapsed) |
| 1 loop + router + location loss, stop-gradient | 25k | 1.36M | 5.49 |
| 8 loops + router | 55k | 1.39M | 4.76 |
| 8 loops + router + location loss | 55k | 1.39M | 5.23 |
| 8 loops, inputs on the same loops, 6 layers | 70k | 1.28M | 5.12 |
| same + rank-16 delta | 137k | 1.34M | 5.09 |

Scarce data (best validation CE seen during training):

| classifier | 1M tokens, 12 epochs | 200k tokens, 40 epochs |
|---|---|---|
| standard, tied | **5.18** | **6.10** |
| standard | 5.24 | - |
| 8 loops x 8 | 5.57 | 6.60 |
| 8 loops + delta | 5.44 | 6.49 |
| 1 loop x 64 | 5.81 | - |

One reading per suggestion:

1. **A real trunk widens the gap.** One loop goes from 0.74 to 0.98 behind, and
   eight loops from 0.29 to 0.55. Richer hidden states give the standard
   classifier more to resolve, and the loops can't use it. The falsifier this
   note set (more than ~0.5 behind at `loops = hidden / 16`) fired.
2. **Inputs on the loops hurt.** Build the input embedding from the same
   phases and spend the saved parameters on two extra layers, and you get 5.12.
   That's 0.9 behind the tied standard classifier at a slightly smaller budget.
   The trunk needs to know exactly which token it read, and the loops blur
   neighbors at the input just as they do at the output.
3. **The neighborhood, read through the paper.** The paper defines bias as what's
   shared across inputs and variance as the input-conditional part. Inside a
   neighborhood the loop's shape is flat, so the choice falls to the per-token
   bias. That's unigram frequency: static, shared, *bias*. So the single loop
   doesn't leave the fine choice open to variance. It hands it to bias. The
   paper's own sentence covers this: "a delta held to a handful of coefficients
   has nothing to spend". Its prescription is directionally right: a structured
   part plus an input-conditional delta with real degrees of freedom. Loops +
   delta (4.54) beats both loops alone (4.77) and the delta alone (4.93, the
   plain-16 arm). But the delta ends up carrying about twice the loops' logit
   energy (ratio 2.2), the reverse of a dominant baseline. And the pair still
   trails the tied standard classifier by 0.32 with more parameters.
4. **The router** views the loop from a predicted point *inside* it. It scores
   each token's phase with the wrapped-Cauchy density, which is parallax made
   exact: near tokens loom large, far ones shrink, and as the point nears the
   rim the view narrows to one spot. It's zero-initialized so it anneals in,
   the way the paper's delta does. It opens (sharpness about 0.4) but gains
   nothing reliable: -0.05 at seed 0, -0.005 at seed 1, -0.01 at 8 loops.
5. **An explicit location target collapses.** Asking the router to predict the
   target's location (a continuous target) makes the token positions crowd into
   a few bins. Phase entropy falls to 0.31, against 0.85-0.97 in healthy arms.
   Location is then trivially predictable, and cross-entropy pays +1.5 nats.
   Blocking the collapse with a stop-gradient on the target positions still
   costs +0.30 against the plain loop. A one-bump target fights a many-bump
   distribution.
6. **Scarce data narrows the gap but never flips it.** At 1M tokens the loops
   overfit less (final train/val gap 1.9, against 2.2 for tied standard), and
   the gap to standard shrinks from 0.55 to 0.39. At 200k it's back to 0.50,
   and late in training the loops' validation CE runs away further (11.2
   against 8.6). Separating neighbors takes large coefficients, which makes the
   classifier overconfident.
7. **11 loops** (4.75) land on the curve between 8 loops and the standard
   classifier, so the count isn't special. The many-loop version does have a
   proper name. M loops put each token at a point on an M-dimensional *torus*,
   one angle per loop. String theory's extra dimensions are curled up the same
   way ("compactified" onto circles and tori), so the picture's instinct has a
   precise counterpart. It just isn't a better classifier.

### What the single loop learned

Here are the nearest frequent tokens (top 1500 by count) to a few anchors on
the learned loop (1 loop x 64, round 2):

| anchor | its neighbors on the loop |
|---|---|
| ` the` | its, these, their, this, our, those, both, each, least, his |
| ` 2` | 8, 6, 11, 5, 9, 10, 7, 16, 13, 3 |
| ` said` | ', would, did, had, might, will, must, should, could, 's |
| ` function` | public, const, set, file, self, def, return, data, line, int |
| ` government` | cap, court, children, law, order, pr, food, trial, people, years |
| ` John` | I, G, Cl, Ch, N, M, U, J, New, K |

One stretch of the loop, in phase order, is all word-continuation pieces:
`ide ane ats ule inal ays age ines ine ite ages ision ...`.

The loop sorted the vocabulary by grammatical role. That's the structure behind
"it finds the neighborhood". Choosing *within* those groups is the part it can't
do, and that's where the cross-entropy goes.

## Where it would pay in Praxis (revised)

Round 1 named the per-cut classifiers in `praxis/decoders/mono.py`. That was
wrong. Small models already share one `vocab x hidden` matrix between input and
output (tying). The same shared matrix plus a per-cut `hidden x hidden` adapter
costs about what a loop coefficient map costs per cut (16k at width 128), and
it's the better classifier. Loops only beat a *plain factorization*, so they'd
matter only where no full matrix can be shared. Nothing in Praxis is in that
position.

What's left:

- **A vocabulary map.** Fit a single loop as a probe on a trained model's final
  hidden states, with the trunk frozen. Sort by phase and you can read the
  model's grammatical categories off a circle. It's cheap.
- **Structured sampling** (untested). The paper reads temperature as
  exploration "within the basin the harmonic constraints define", and loop
  neighborhoods are such basins. A hot sample that stays in the argmax's
  neighborhood swaps "the" for "these" or "their", not for noise. That's a
  generation-quality question, so cross-entropy can't settle it.

## The classifier (reference code)

Kept for the probes above. Not recommended as a model's classifier. Exclude
`phases` from weight decay, since they're angles.

```python
class FourierLoopClassifier(nn.Module):
    """Each token holds one phase per loop; logits are a band-limited periodic
    function of those phases whose coefficients are read from the hidden state."""

    def __init__(self, hidden: int, vocab: int, loops: int, harmonics: int):
        super().__init__()
        self.loops, self.harmonics = loops, harmonics
        self.phases = nn.Parameter(torch.rand(loops, vocab))
        self.coef = nn.Linear(hidden, loops * harmonics * 2, bias=False)
        nn.init.normal_(self.coef.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(vocab))
        self.register_buffer("freqs", torch.arange(1, harmonics + 1.0))

    def forward(self, h: Tensor, band: Optional[Tensor] = None) -> Tensor:
        c = self.coef(h).unflatten(-1, (self.loops, self.harmonics, 2))
        theta = 2 * math.pi * self.phases[..., None] * self.freqs  # [M, V, K]
        feats = torch.stack([theta.cos(), theta.sin()], -1)
        if band is not None:  # coarse-to-fine weight per harmonic, [K]
            feats = feats * band[:, None]
        feats = feats / math.sqrt(self.loops * self.harmonics)
        return torch.einsum("...mkc,mvkc->...v", c, feats) + self.bias
```

The BARF window used in the toy, with `alpha` ramping from 0 to K over the
first half of training:
`band_k = (1 - cos(pi * clamp(alpha - k + 1, 0, 1))) / 2`.

The BARF window, with `alpha` ramping from 0 to K over the first half of
training: `band_k = (1 - cos(pi * clamp(alpha - k + 1, 0, 1))) / 2`. Round 2's
router adds `log(1 - |a|^2) - log(1 + |a|^2 - 2 Re(a e^{-i theta_v}))` per loop.
`a` is a point in the open disk, read from `h` and zero-initialized.

## Cheapest probes, in order

1. ~~Does the gap hold in a real trunk?~~ Done in round 2. It widens, and the
   falsifier fired.
2. **Structured sampling.** Take a trained standard model, fit a single-loop
   probe on its final hidden states, and at temperature T mix the model's
   distribution with the loop neighborhood around its argmax. Compare
   diversity at matched quality against plain temperature. *Falsifier:* no
   diversity gain at matched quality.
3. **The map over training.** Fit the same probe on checkpoints and watch the
   categories form. It would be a readable account of when grammar arrives.

## Open questions

- Would a loop reserved for a genuinely cyclic family (digits, weekdays,
  months) help, where the clock is real? Every loop here was corpus-wide.
- Is the map stable across seeds and runs, or does each run cut the circle
  differently?
- Loops combine by adding logits, which takes an intersection of
  neighborhoods. Mixing them as probabilities would take a union, which is
  closer to "apples or cars or roses". It's untested.
