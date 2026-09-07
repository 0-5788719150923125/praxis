# Ghost Features: extra channels from weights you already have

Status: **unvalidated, reading note + experiment design** (2026-09-07). Nothing
implemented. Opened from a question about "phantom neurons" - whether a model
can carry 12M parameters' worth of shape on 6M real trainable ones - and closed
onto one specific, cheap, testable mechanism. Sibling to
[mixture_of_widths.md](mixture_of_widths.md),
[lottery_engineering.md](lottery_engineering.md),
[information_density.md](information_density.md),
[exact_solve.md](exact_solve.md).

## The original question

Can parameters be *synthesized*? Weights that exist in the forward pass, do real
arithmetic, but are derived by a function from a smaller set of real trainable
ones and are never themselves trained. Target shape: a 12M model on 6M real
parameters, then scale the ratio if it holds.

The answer has two halves, and they point in opposite directions.

## Half one: the bound, and what it does not bound

If weights `W` are produced by any deterministic `g: R^N -> R^M` with `N < M`,
then the reachable weight configurations form a set of dimension at most `N`.
Three ways to say the same thing:

- **Counting.** `θ` in `N` floats of `b` bits is `2^(N·b)` distinct values. The
  image of `g` has at most that many elements no matter how large `M` is. You
  cannot distinguish more functions than you have codes for.
- **Data processing.** `D -> θ -> W` is a Markov chain, so
  `I(D; W) <= I(D; θ)`. Deterministic post-processing never creates information.
- **Rank.** The Fisher of the composed model is `Jᵀ F_W J` with `J = ∂g/∂θ` of
  shape `M x N`, so rank `<= N`. Everything reading the Fisher sees `N`:
  effective parameter counts, natural-gradient geometry, and the RLCT
  ([rlct_landscape.md](rlct_landscape.md)) that already ships.

**So "12M of capacity from 6M" is not available.** What the bound does *not*
cover, and this is where the whole idea lives:

1. **Prior.** A 6M convnet and a 6M MLP have identical bounds and completely
   different behavior. The bound is silent about which `N`-dimensional manifold
   you pick.
2. **Compute.** Parameters and arithmetic are separate resources and the bound
   counts one. Fixed-depth transformers sit in a bounded circuit class;
   recurrence provably escapes it. **Praxis already spends this one**:
   `abstractinator-a.yml` sets `num_layers: 1`, `depth: 6`. One layer's weights,
   six passes. That is a 6:1 phantom-neuron ratio already shipped, on the depth
   axis, harder than the 2:1 the original question asked for.
3. **Optimization.** Expressible and reachable-by-SGD-from-an-init are different
   questions. Redundant parameterization changes the implicit bias.
4. **Bits per parameter.** The bound is `N x (bits actually used per real)`.
   That 4-bit quantization costs so little says trained networks store well under
   what their floats hold. If a 6M model is under-using its budget, the ceiling
   is not yet the binding constraint and there is headroom that has nothing to do
   with ghosts.
5. **Frozen structure.** A tensor independent of the data carries zero bits and
   violates nothing, while still contributing shape and a basis.

Ghost features live in (5), with a side of (1). Recurrent depth lives in (2).
**They are orthogonal axes**, which is the reason this is worth a run at all
rather than a second helping of what `depth: 6` already buys.

## Half two: the paper

**Vieira Neto, G. & Valle, M. E. (2026). "Ghost Features and Spooky Transfer
Learning for Hypercomplex-Valued Neural Networks." arXiv:2608.07735 (cs.CV).**

### The mechanism, without the algebra

A hypercomplex algebra of dimension `d` is defined by `d` matrices
`P_0 ... P_{d-1}` giving the product of basis units. For quaternions (`d = 4`)
these are **signed permutations**:

```
Re{x·w}   = x0 w0 - x1 w1 - x2 w2 - x3 w3      (P_0: identity, signs + - - -)
Im_1{x·w} = x0 w1 + x1 w0 + x2 w3 - x3 w2      (P_1: swap 0<->1, 2<->3)
Im_2{x·w} = x0 w2 - x1 w3 + x2 w0 + x3 w1      (P_2: swap 0<->2, 1<->3)
Im_3{x·w} = x0 w3 + x1 w2 - x2 w1 + x3 w0      (P_3: swap 0<->3, 1<->2)
```

The paper's construction (§3): take a real layer with `dC` input channels and
`C'` outputs. Chop the input channel axis into groups of 4. The real part of the
equivalent quaternion layer reproduces the original output exactly; the `d-1`
imaginary parts are **`d-1` extra output blocks of `C'` channels each, computed
from the same weight numbers with the input channels shuffled and sign-flipped
within each group of 4.** Those extras are the ghost features. Then a trainable
`1x1` conv (or dense) mixes `d·C'` back down to `C'`.

**The critical detail:** the 4-way grouping is *imposed by slicing an existing
channel axis*, not discovered in the data. This kills three integration ideas
before they cost anything - no need to render text as images, no need for
4-grams, no need for four embedding vectors per token. Any tensor whose channel
count divides by 4 can be reinterpreted. `hidden_size: 272` is `4 x 68`.

**Implementation follows directly, and needs no quaternion arithmetic at all.**
Since every `P_k` is a signed permutation, the whole thing is an index-gather
plus a sign flip on the input axis in blocks of 4:

```python
# W: [in_features, out_features], in_features % d == 0
# PERM[k], SIGN[k]: length-d index / sign vectors from the multiplication table
blocks = W.view(in_features // d, d, out_features)
W_eff = torch.cat(
    [(blocks[:, PERM[k], :] * SIGN[k].view(1, d, 1)).view(in_features, out_features)
     for k in range(d)],
    dim=1,
)  # [in_features, d * out_features], d tied blocks, one real tensor
```

Compile-friendly, no new dependency ([[feedback_dependencies]]), and the control
arm below is the same code with different `PERM`/`SIGN`.

Read that way it stops being exotic: **ghost features are structured weight
tying.** A `d`-times-wider projection whose column blocks are fixed signed
permutations of each other.

### What the paper shows, and what it does not

EfficientNetV2-B0 on BloodMNIST, quaternion embedding of the *first* conv layer:

| model | trainable params | accuracy |
|---|---|---|
| baseline | 10,248 | 95.92 ± 0.07 |
| Spooky TL (R) | 14,376 | 98.21 ± 0.19 |
| Spooky TL (V) | 11,304 | 97.49 ± 0.51 |

**The comparison is not parameter-matched, and the gap is the entire method.**
The baseline is a frozen backbone with only an 8-way classifier trained. The
spooky model adds a trainable `1x1` conv at the front of that frozen backbone.
They unfroze part of the network and it improved. The missing control is
obvious: the same `1x1` conv in the same place, fed the original 32 channels.
Without it there is no way to attribute the 2.3 points to the quaternion
structure rather than to having 4,128 trainable parameters at the front instead
of zero.

Also: one dataset, one layer, 15 epochs, image classification, and the framing
is *transfer learning* - reusing frozen pretrained ImageNet weights. Praxis
trains from scratch. **Their number carries no information about our setting.**

The authors say as much (§6): the natural next step is information-theoretic
work "aimed at quantifying the contribution of ghost features, formally
pinpointing its source." They know the attribution is open.

**That is the opportunity, stated honestly.** Not "replicate a strong result" -
the result is not strong. It is that a clean, cheap mechanism has been proposed
and never had its control run, and running that control is a day of work in a
framework built for exactly this kind of one-variable arm.

## Where it would go

### Rejected: blanket application

Every linear layer ghost-expanded. The mixer is `d·C x C` per site and at
`hidden_size 272` with `d=4` that is ~296K per site in a model of a few million.
The ghost channels are free; the mixer that makes them usable is not. Blanket
application spends the entire budget on mixers.

### Best fit: the FFN up-projection, where the mixer already exists

`GatedLinearMLP` (`praxis/dense/glu.py:41-46`) is `up: Linear(272, 724)` then
`down: Linear(362, 272)`. **The down-projection is already the mixer the paper
adds.** So the ghost version replaces only the up-projection and adds no new
module:

| | up | down | total |
|---|---|---|---|
| GLU as-is | 272 x 724 = 197K | 362 x 272 = 99K | **296K** |
| ghost GLU (`d=4`) | 272 x 181 = 49K | 362 x 272 = 99K | **148K** |

Same 724 hidden channels, half the FFN. The honest matched baseline is therefore
**not** the 296K GLU - it is a standard GLU shrunk to 148K, which is `up_size`
363 instead of 724. State this up front or the arm will look better than it is
for the wrong reason. Compare parameter-matched, not shape-matched.

### Most Praxis-native: the PEER expert bank

The current line runs `ffn_type: peer_split` (`abstractinator-g.yml`), not a
plain GLU, and PEER's banks are `nn.Embedding`/`EmbeddingBag` of shape
`[num_experts * num_sets, hidden_size]` (`praxis/dense/peer.py:229-276`). Chop
`hidden_size` into groups of 4 and each bank row yields **four experts instead
of one, with no new bank parameters and no new optimizer state.**

That lands directly on the known wall: the PEER bottleneck is optimizer state on
the banks ([[project_peer_sparse_optimizer]]), and this multiplies `num_experts`
by 4 without touching it. Retrieval cost is sqrt-cheap because
`num_keys = sqrt(num_experts)`. If any version of this idea is worth building,
it is probably this one - but it is also the most entangled, so it should follow
the GLU arm rather than lead.

### Paper-faithful: one site at the front

The embedding output or first block input, one mixer, closest to what was
actually tested. Cheapest to build, least likely to matter, useful mainly as a
sanity check that the machinery is correct.

### Later: attention projections

Q/K/V. More interactions than it is worth while RoPE, kaleidoscope and ghostmax
are all live in that path.

## The three arms

One change each, in the usual style:

1. **Control.** No ghost expansion. Standard GLU shrunk to the ghost arm's
   parameter count (`up_size` 363). This is the arm the paper skipped.
2. **Quaternion ghosts.** `d=4`, `PERM`/`SIGN` from the quaternion table.
3. **Random ghosts.** Identical shape and parameter count, seeded arbitrary
   signed permutation, frozen. Not an ablation - the actual hypothesis test.

Read it as: if **2 ≈ 3**, the algebra is decoration and the mechanism is "fixed
expansion plus a learned mixer," which is cheaper and still a real finding
worth writing down. If **2 > 3**, the algebra earns its keep, and that is
genuinely new because nobody has run it. If **both ≈ 1**, it was the mixer all
along and the thread closes cleanly.

## The better algebra, probably

`d` is not restricted to 4. Any non-degenerate algebra works (the paper's
requirement is that every `P_k` be nonsingular, §2.1). **Complex numbers are the
`d=2` case**, and they are the one algebra Praxis has a *reason* to pick:

```
Re{x·w} = x0 w0 - x1 w1
Im{x·w} = x0 w1 + x1 w0
```

The harmonic work already carries genuine amplitude/phase structure
([harmony.md](harmony.md), [[project_harmonic_latent_koopman]]), so a 2-grouping
of those pairs would be *meaningful* rather than an arbitrary slice - which is
exactly what the quaternion version is missing. Half the expansion, a quarter of
the mixer cost, and the only variant where the algebra is not chosen by
coincidence. `272 = 2 x 136`.

Worth running `d=2` on harmonic-paired channels as a fourth arm, or possibly as
the *first* one.

## What to watch

- **`val_byte_nll_bits`** as arbiter, per the standing rule that it is the
  calibrated series ([surrogate_geometry.md](surrogate_geometry.md)).
- **Mixer mass per ghost slot.** Does the learned down-projection actually use
  slots 1..d-1, or does it drive them toward zero? One norm per slot, cheap, and
  close to decisive on its own. A mixer that zeroes the ghosts has answered the
  question without needing the loss curve.
- **Effective rank of the expanded hidden activations.** If 724 ghost-expanded
  channels carry the effective dimension of 181, the ghosts are redundant and
  the mixer is discarding them. The PCA / effective-dim machinery already exists
  (`p0_crystal_effective_dim`).
- **Activation memory.** `d=4` means 4x the pre-mixer activations. Real cost, and
  it lands on the same budget the batch governor arbitrates
  ([[project_batch_governor]]).

## How it fails

The likeliest failure is not that ghosts are useless but that **this is a
constraint dressed as a gift.** A 724-wide up-projection with 4-way tied columns
is strictly *less* expressive than a free `272 x 724`. Against a full-width GLU
it is a restriction; against a 148K GLU it is an expansion. Which baseline gets
picked decides whether the arm looks good, and picking the flattering one is the
easy mistake here. Hence arm 1.

Second failure: the ghosts are genuinely different linear functionals of the
input, but after a pointwise nonlinearity they may still be close enough that
the mixer recovers nothing a narrower dense layer would not have found. That is
what the effective-rank metric is for.

## One thing to keep out of the writeup

The thread that led here noticed that Praxis already uses "ghostmax" and that
"spooky" evokes action at a distance. Ghostmax (`praxis/attention/causal.py:262`)
is a phantom key prepended to the softmax denominator so attention can attend to
nothing; the paper's ghosts are quaternion multiplication slots. Nothing connects
them but the word, and "spooky" is a Halloween pun following "ghost." The case
for running this rests entirely on the missing control being cheap and the answer
being informative either way. That case is strong on its own, and naming
coincidences would only make it easier to knock over. Same discipline as
[grounding.md](grounding.md), applied to a small thing before it becomes a large
one.

## Verdict on phantom neurons

The original idea is sound and mostly already spent. Parameters buy stored
information (hard-capped), arithmetic shape (not capped), and optimization
geometry (not capped). "More capacity from fewer parameters" is unavailable;
"same information, more shape" is available and Praxis already takes it on the
depth axis at 6:1.

Ghost features are a second, orthogonal helping of the same trade, *within* a
tensor rather than *across* depth, at a real 2x on the FFN up-projection. Whether
the specific algebraic structure matters is unknown and cheap to find out, which
is the whole reason to bother.
