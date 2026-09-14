# TemporalMesh Transformer: audit, and the graph-attention thread it points at

> Status: **architecture rejected, nothing extracted, 2026-09-13.**
> Audit of [vignesh2027/TemporalMesh-Transformer](https://github.com/vignesh2027/TemporalMesh-Transformer)
> (18 commits; Zenodo preprint 10.5281/zenodo.20287197; HF model/dataset/Space).
> Revised after pushback on the first draft - see "What the first draft got
> wrong" for what I over-claimed and what survives.
>
> **The one thing to take from this note: nothing, and that is the finding.**
> The only mechanism worth extracting turned out to be one we already have and
> have already improved on (see "What to keep"). The note is kept for the audit
> method and so this repo is not evaluated twice. If you want the idea TMT was
> reaching for, skip to item 3 - causal block-sparse selection - which is real,
> shipping, and buildable here.

## What it claims

Three "innovations" in one decoder: (1) **Mesh Attention**, a kNN graph over
token positions rebuilt each layer from cosine similarity of live
representations, claimed O(S·k) instead of O(S²); (2) **Temporal Decay**, a
learned scalar multiplied into post-softmax attention weights; (3) **Adaptive
Depth Routing**, a per-token confidence gate that freezes a token and skips it
past the remaining layers. Plus a dual-stream FFN and 16 EMA memory anchors.
Headline: WikiText-2 PPL 29.4 vs vanilla 42.1 at 120M params, at 0.48x compute.

## The measurement

`MeshAttention` has no causal mask - the strings `causal`, `tril` and `triu`
appear nowhere in the repo - and `trainer.py` trains it with `x = ids[:, :-1]`,
`targets = ids[:, 1:]`. `build_mesh` takes top-k over the full S x S similarity
matrix, so a query at position i attends to its k most similar tokens anywhere
in the sequence, future included.

Three independent confirmations:

- **Perturbation.** Change only the last token of a 32-token sequence; logits at
  all 32 positions move, including position 0.
- **Gradient.** `d logits[0] / d embed[j]` is nonzero for 28 of 31 future
  positions.
- **Information-theoretic.** Trained on a first-order Markov source whose
  conditional entropy rate is known exactly - H = 3.2334 nats, PPL 25.37, a hard
  floor for any causal model - it reaches held-out CE 1.7834, **PPL 5.95**, and
  was still falling at step 1200. It finds the leak around step 800.

The channel is wide: on the trained model, ~50% of mesh edges point forward in
time at every layer, and 21-43% of tokens have their immediate successor
directly in their top-k neighbourhood. Content-selected attention over a
residual stream acquires positional locality whether you asked for it or not.

This is the part of the audit that is a measurement rather than an inference,
and it is the part that decides the question.

## Is causality actually required?

The right answer is not "yes, always". It is: **causality is a consistency
requirement between the architecture and the objective, not a property the
architecture owes the universe.**

Whole successful families are non-causal - BERT, T5 encoders, XLNet, and the
current diffusion LMs (LLaDA, Dream, Mercury) which see the entire noisy
sequence at once and denoise in parallel. Every one of them changes the
objective to match: BERT masks the token it predicts so it is genuinely hidden;
diffusion LMs corrupt and reconstruct; any-order models factorise over a
permutation. The token being predicted is never visible in the input. That is
the invariant, and the causal mask is just the cheapest way to enforce it for a
left-to-right factorisation.

TMT keeps the causal objective and drops the enforcement. That is not a new
paradigm, it is an unmatched pair, and PPL 5.95 against a floor of 25.37 is what
the mismatch looks like with a number on it. Its 29.4 and the baseline's 42.1 do
not measure the same quantity, so the comparison is void regardless of whether
the run happened.

**On the physics analogy.** Retrocausal interpretations of QM, Wheeler-Feynman
absorber theory, closed timelike curves, the delayed-choice eraser - these are
real topics, but no-signalling holds throughout: none of them lets you *retrieve*
information from the future, which is why the eraser needs the coincidence counts
to show anything. More to the point, it is the wrong analogy. Nothing here is a
claim about physical time. When you generate token n, token n+1 does not exist
yet - not because a law forbids it, but because you have not made it. The
constraint is about what is available at inference, and a model trained with
access to something it will not have at deployment has a train/test mismatch
regardless of any metaphysics.

**If we ever want to drop causality for real**, the direction is discrete
diffusion / any-order factorisation, where the objective changes with the
architecture. That has a genuine connection to the harmonic and Koopman line -
both are about treating the sequence as a field to be solved rather than a
stream to be extended. It has nothing to do with TMT.

### Could the decay mechanism substitute for a causal mask?

No, and the reason is worth writing down because it is general.

- **It is symmetric.** The paper's form is `σ(w·|t_i − t_j|)`, which has no sign
  and therefore cannot distinguish past from future. Causality is not a distance
  penalty, it is a direction.
- **It is soft, and a mask has to be hard.** Any finite weight on a future key
  passes the answer through, and CE is an extremely sharp incentive to find it -
  it found the leak here in 800 steps. A 0.9 multiplier on the successor's value
  is not 10% of a leak, it is a complete one.
- **The asymmetric unbounded version already has a name.** A pre-softmax additive
  bias that goes to −∞ for j > i *is* the causal mask; ALiBi is the finite
  asymmetric version and still needs the mask on top of it.

There is a real experiment hiding in the question, though - see "soft-to-hard
boundary" below.

### Does "graph attention" exempt it?

For graph attention as such, causality is meaningless - a molecule or a citation
graph has no time axis. But the moment the nodes *are* sequence positions and
the objective predicts position n from positions < n, the requirement does not
disappear, it changes notation: the graph must be a **DAG oriented along time**.
Same constraint, spelled as edge admissibility instead of a mask.

And that is the honest version of the whole idea: a content-selected sparse DAG
over past positions, rebuilt per layer. It is well developed - Routing
Transformer (k-means routing over keys), Reformer (LSH buckets), and most
directly DeepSeek's Native Sparse Attention and Moonshot's MoBA (both 2025),
which select blocks per query per layer, respect causality, and have kernels
that make the sparsity an actual speedup. The paper's related work names
Longformer, BigBird and Performer and then claims no prior work derives topology
from live representations, which is exactly what Routing Transformer does.

## What the demo runs

You noticed the HF Space does not produce causal text. It does not produce text
at all: `app.py` is a standalone numpy mock that never imports torch (not in its
`requirements.txt`) and never loads the repo's code. The "Live Forward Pass" tab
runs on `X = rng.randn(S, d)` - random Gaussian vectors, seed 42 - and the exit
gate is `W_gate = rng.randn(d) * 0.3`, freshly random per layer. It visualises a
kNN graph over noise.

So the simpler explanation for what you saw is not that TMT is a different kind
of model. It is presented unambiguously as a next-token predictor: the abstract
says "a novel autoregressive language model architecture", `loss.py` says
"standard next-token cross-entropy", and the HF model-index registers
`task: text-generation` with WikiText perplexities. The demo just has no model
in it.

Two things fall out of reading it. The demo implements `temporal_decay(i, j, S)`
as the paper's pairwise `sigmoid(w·|t_i − t_j|)` - which the actual library does
**not** implement - so the author's own demo disagrees with the author's own
code. And the Space README carries a third set of numbers that appear nowhere in
the paper or repo: Mamba at WikiText-2 PPL 31.8, and a LongBench column
(41.2 / 51.3 / 53.4) for models of this size trained 10k steps on WikiText-2.

## What the first draft got wrong

The first draft said the results "were never produced", resting partly on the
four experiment notebooks having zero executed outputs. **That was an
overclaim.** Unexecuted notebooks and absent checkpoints prove only that these
artifacts were not published; plenty of sound work has messy repos, and people
withhold artifacts deliberately. Ranked properly, the artifact evidence is weak
and the internal evidence is what matters.

What survives without assuming anything about what was or was not run:

- **The stated config cannot produce the stated parameter count.** `d_model=512,
  n_heads=8, n_layers=12, vocab=50258` builds a **60.5M** model. All eight rows
  of the results table say ~120M. Checkable from the paper and code alone.
- **A config in the paper cannot be instantiated.** Table 2's TMT-Medium is
  `d_model=512, n_heads=6`; 512 % 6 ≠ 0 raises AssertionError in
  `MeshAttention.__init__`.
- **The compute column cannot have been measured from this code.** Mesh-only is
  reported at 0.62x. The implementation is *slower* than dense attention (below),
  so no run of this code produces that number.
- **The paper describes mechanisms the code does not contain** (§6.1 anisotropic
  edge weights, §6.2 hierarchical mesh) and gives formulas that differ from what
  is implemented (table below).
- **The open benchmark data is the claim restated.** The HF `TMT-Benchmarks`
  `ablation_reference` split is the same 8 rows. `length_scaling` is arithmetic
  on S² vs S·k with no timing column. `complexity_test` is random word salad
  ("that ? take food city you an that convolutional") whose
  `expected_exit_layers` come from a hardcoded per-token-type complexity score -
  that table *is* the paper's Figure 4 "punctuation exits at 2.1 layers"
  finding, so the finding is a lookup table, not a measurement.
- `tests/test_benchmarks.py`, committed as "benchmark tests with real results",
  hardcodes the ablation table and asserts arithmetic about it
  (`assert min(ppls) == full_tmt_ppl`). No model is involved.
- PPL 29.4 for a ~120M model on WikiText-2 is GPT-2 117M's published zero-shot
  number, 29.41.

The defensible statement is not "they never ran it". It is **the reported
numbers cannot have come from this code**, and separately, this code cannot
produce a valid perplexity at all.

## Code vs paper

| Paper | Code |
|---|---|
| Eq 6: `δ_h(i,j) = σ(W·\|t_i − t_j\|)`, pairwise | per-query scalar broadcast over all keys; `\|t_i − t_j\|` does not exist |
| Eq 10: `L_gate = −Σ[c log c + (1−c)log(1−c)]` | `-(conf - 0.5).abs().mean()` |
| Eq 5: edge weight multiplied inside softmax | added to the logits |
| Eq 17: anchor EMA β=0.99 over "tokens that attend to them" | β=0.9 over the global batch mean, no attention weighting |
| §3.4: two GeLUs per FFN stream | one |
| §6.1 anisotropic edges, §6.2 hierarchical mesh | not implemented |

## The mechanisms on their own terms

Each fails independently of anything above.

**Mesh attention is slower than dense attention.** `build_mesh` materialises the
full S x S cosine matrix in a Python `for b in range(B)` loop, then
`MeshAttention` computes the full (B,H,S,S) score tensor and *adds* a (B,S,S)
fp32 `-inf` mask. Nothing sparse is ever exploited. Measured at d=128, 2 layers:
8.1 / 15.7 / 36.4 / 137.0 ms at S = 128 / 256 / 512 / 1024 - superlinear, and it
allocates strictly more memory than standard attention. The "128x reduction at
S=1024" is `1024/8` done by hand. Sparse topology only pays with a kernel; this
is the lesson NSA and MoBA are built around.

**Temporal decay is a constant** - see the extraction below for why it is still
the most interesting line in the repo. As written it multiplies post-softmax
weights by a per-query scalar, so attention rows sum to 0.512 instead of 1, and
the factor varies across the entire sequence only from 0.51187 to 0.51250. It
has no dependence on query-key distance at all.

**The exit gate saves no compute and has a degenerate objective.** `TMTLayer`
computes the full layer for every token, then overwrites exited ones with
`torch.where`: 1.568 ms/layer at 0% exited, 1.655 ms at 100% exited - *slower*
when everything exits. The auxiliary loss `-E|c-0.5|` contains no accuracy term
and no compute term, so "never exit" and "exit at layer 0" are both global
minima. Separately, `TMTModel._init_weights` runs after the submodules and
overwrites `ExitGate`'s deliberate `-2.0` "start pessimistic" bias with 0.0.
Ouroboros' hard-concrete halting with an explicit budget is strictly the better
formulation and we already have it.

**Memory anchors collapse to rank 1.** The EMA writes `x.mean(1).mean(0)`, a
single (D,) vector, broadcast to all M anchors, so every anchor gets an identical
update. Measured: mean pairwise cosine 0.0627 -> **1.0000**, max pairwise L2
distance 0.251 -> 0.0017, within 60 steps. Sixteen anchors become one vector, and
cross-attention over identical keys is a constant bias. It also writes `.data`
in place on a parameter the optimizer is concurrently updating.

**Dual-stream FFN** is two half-width MLPs blended by a per-dimension sigmoid,
`g*h_syn + (1-g)*h_sem` - a 2-expert soft mixture, which is SMEAR without the
name. The "same parameter budget as a standard FFN" claim is wrong twice: the
streams total 1x expansion where standard is 4x, and the `nn.Linear(d, d)`
fusion gate costs more than either stream.

## What to keep

**1. The accidental gate - already built here, and improved on.** `attn = attn *
head_decay` with `head_decay` broadcast over keys is, algebraically, scaling the
SDPA *output* by a per-head per-query sigmoid; the softmax has already
normalised, so a factor constant across keys passes straight to the output. That
is the G1 variant in [Gated Attention](https://arxiv.org/abs/2505.06708)
(NeurIPS 2025 best paper, in Qwen3-Next). **We already have it:**
`praxis/attention/arc.py:140` is `output = output * torch.sigmoid(self.gate(inputs))`,
and `single.py` / `kaleidoscope.py` deliberately use SiLU instead so the gate can
amplify and flip sign rather than only attenuate - strictly more expressive than
the paper's sigmoid - with `kaleido_gate_negative` measuring whether that extra
freedom is actually used. Nothing to take. TMT's version is inert anyway, since
its gate is driven by `decay_scalars.mean(-1)`, a near-constant function of
position, instead of by the query.

**2. Soft-to-hard causal boundary (worth a small test).** From your decay
question. A pre-softmax additive bias `b(i,j) = -softplus(θ) * relu(j - i)` with
learned sharpness, annealed toward a hard mask over training: a soft boundary
while the graph is forming, hard by the end. The failure mode is exactly the one
measured above - any finite θ leaks - so it must anneal to a hard mask and
`test_causal` must pass at eval, not just at the end of training. Worth knowing
whether the soft phase buys anything; cheap to answer.

**3. Causal block-sparse selection (the honest mesh, and the buildable one).**
The idea TMT was reaching for is real and shipping: DeepSeek's NSA and Moonshot's
MoBA select attention topology per query per layer from content, stay causal, and
are fast. TMT contributes no evidence either way, but the direction is not dead -
it was just badly executed here.

The thing that kills a naive version is the one measured above: **sparse topology
only pays with a kernel.** TMT's mesh is slower than dense attention because a
per-token gather that cannot be fused costs more than the dense matmul it skips.
The difference between NSA/MoBA and TMT is the unit of selection - select
*blocks*, not individual keys. Block-level top-k over past blocks produces a
block mask, which is what FlexAttention consumes and what `torch.compile` can
generate; per-token kNN produces a scatter nothing can fuse.

So the arm worth building is: score past blocks against the query (pooled keys),
take top-k blocks, attend densely inside them. Causal by construction, since only
past blocks are candidates - no mask bolted on afterwards. Overlaps
`sparse_query.py`, and the flex traps ssog already hit are the known cost. The
question it answers is one we can already ask: does content-chosen block topology
beat fixed structure, given that [const [t, t]](kaleidoscope.md) says frozen-QK
beats standard?

**4. Non-causal done properly, if we ever want it.** Discrete diffusion /
any-order factorisation, where the objective changes with the architecture.
Files with the harmonic and Koopman thread, not with this one.

**5. The entropy-floor test.** Train on a synthetic source with a known entropy
rate and check the model cannot beat it. Stronger than perturbation when a leak
is suspected but not proven, because it says how much the model is cheating
rather than only that it can. Our perturbation tests
(`tests/attention/test_registry.py::test_causal`,
`tests/test_modeling.py::test_inference_is_causal`) stay the first filter - they
are cheaper, they run in CI, and a clean codebase does not fool them. Run them on
any borrowed attention mechanism before reading the paper.
