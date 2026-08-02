# Continual learning, credit assignment, and whether we forget at all

Opened 2026-08-02, after reading *"The Art of Not Forgetting"* (Atmuri, Kumar,
Bhogarajula - Arkadhi Labs, [arXiv:2607.17944](https://arxiv.org/abs/2607.17944)).

The thread has two halves that must not be collapsed into each other:

1. **The diagnosis** - dense global credit assignment is what overwrites, so
   forgetting is structural to backprop rather than a training defect. This is
   the part the field seems to be converging on, and it is worth taking
   seriously.
2. **Their cure** - CMP, an architecture co-designed so a one-hop delta rule
   suffices. This part is almost certainly not for us, and their own Appendix D
   is the reason.

Agreeing with (1) does not commit us to (2).

---

## What CMP actually is

Seven pieces, all in service of making the readout linear enough that a local
rule can train it:

1. **Sparse relational binding** - two *frozen* random embedding matrices bound
   elementwise, then k-WTA sparsified:
   `z_t = normalize(k-WTA_k(E_L[x_{t-1}] ⊙ E_R[x_t]))`
2. **Two-tier competitive memory** - fast buffer + slow register, softmax
   content-addressed read `Σ_i softmax(τ ẑ·m̂_i) m_i`, writes gated by a
   calibrated match threshold `θ = 0.86 · E[max_{i≠j} v̂_i·v̂_j]` so a slot only
   fires on genuine match, not on baseline overlap.
3. **Hierarchy** - bind consecutive `z` into a coarser word-scale code `h_t`.
4. **Predictive coding** - `h_t` predicts `z_t`; the sparsified residual `e_t`
   becomes another context term.
5. **Linear readout** - seven terms *summed*, no MLP anywhere.
6. **Local delta rule** - `err = onehot(y) - softmax(logits)`,
   `ΔW = (η/N) Σ_t errᵀ φ(t)`. One hop, no chain rule.
7. **Weight-protect** - EWC without Fisher, because there is no gradient to
   square: `I ← I + E[|W^(i) - W^(i-1)|]`, then `ΔW ← ΔW / (1 + λI)`.

Hyperparameters: r=1024, batch 64, seq 256, 6000 steps/domain, η=0.12, λ=5.0.

Result: on a self-assembled 15-domain incremental text protocol, forgetting
drops 15-19x against a Transformer + online EWC (BWT +2.2457 → +0.1482), with a
domain-order control spanning +0.24 to +0.44.

## Why it is not a trainer for us

The obvious-looking wiring - a `TRAINER_REGISTRY["cmp"]` sibling to
`mono_forward` - does not hold up. Mono-Forward is a trainer type because it
still trains ordinary `nn.Module`s with autograd; it just cuts the graph
(`praxis/decoders/mono.py:163` is literally a `.detach()`), and autograd on a
disconnected graph *is* local learning. CMP has no autograd at all and no
transformer, so that registry entry would be a trainer with no model to train.
It is a fork, not an integration.

The linearity is not incidental, it is the price. The paper says so: with no
backprop there is no chain rule through binding, memory, and hierarchy, so every
readout weight must update from information available locally, at the readout,
this step. Add a nonlinear MLP and the rule stops working. Praxis is the
opposite of that model in every dimension that matters - recurrent depth,
per-depth RoPE thetas, SMEAR routing, PEER banks, harmonic fields - and all of
it exists because there *is* deep nonlinear credit assignment to exploit.

**Appendix D is the tell.** They tried exactly the hybrid this thread would
circle toward: merge CMP with a deeper stack that on its own reached 2.49-2.51
BPB. The merge got worse than either arm (3.48 vs 3.27). Their diagnosis: the
depth blocks use plain normalization, memory retrieval assumes a sparse code,
and feeding it dense vectors wrecked slot matching. Secondary factor, quoted:
*"signal dilution: seven or more readout terms competing for the same global
error signal."*

## Evidence quality, stated honestly

Things the authors disclose themselves, to their credit:

- Accuracy is substantially worse than the Transformer (3.1-3.27 BPB). Their
  words: *"It does not claim CMP is more accurate than a Transformer; it is not,
  by a substantial margin."*
- Split-MNIST is a **null result** - sparse and dense conditions statistically
  identical (BWT -0.0207 ± 0.0003 both). No evidence the mechanism generalizes
  past text.
- Appendix E: 5x the parameters (6.4M → 29.8M) moved BWT from +0.1482 to
  +0.1397. Nearly nothing.
- The 15-domain corpus is self-assembled, not a standard benchmark, and the
  baseline is online-EWC only.

So: one positive result, one modality, one custom benchmark. Enough to take the
*diagnosis* seriously. Not enough to adopt the *architecture*.

---

## Does our size save us?

The hypothesis worth testing: tiny models never had a forgetting problem,
because they lack the parameters to memorize anything and lean almost entirely
on their inductive biases. What lives in the architecture cannot be forgotten by
the weights.

There is real force to this, but it does not go through cleanly, and it is worth
writing down why before betting on it.

**Against the hypothesis.** Catastrophic forgetting was *first documented in
tiny nets* - McCloskey & Cohen 1989 was a small MLP, not a large one. The
mechanism is interference in shared parameters, and a small model has *less*
redundancy to absorb a new domain, not more. Larger models arguably forget less
per unit of new data precisely because they have spare capacity to route around.
So "too small to memorize" does not imply "too small to forget"; it may imply
the opposite. What changes with scale is the *kind* of forgetting: a tiny model
loses skills and representations rather than facts, which is harder to notice
and arguably worse.

**For the hypothesis, in a form that survives.** The stronger version is not
about capacity, it is about *where the knowledge is stored*. If a small model's
competence is mostly architectural prior plus a thin learned layer on top, then
domain shift perturbs a small, recoverable surface. That is a claim about the
bias/variance position of our models (see
[temperament_bias_variance.md](temperament_bias_variance.md)) and it is
measurable, not just arguable.

**The structural reason this is not urgent.** Praxis trains on a *shuffled
mixture*, not a domain-incremental stream. Catastrophic forgetting is a property
of the sequential protocol, not of backprop in the abstract. As long as
collections are i.i.d.-interleaved by the sampler, there is no forgetting
problem to solve by construction - CMP's entire benchmark protocol is one we do
not run.

**Where it becomes real anyway**, and why the item stays open:

- **RL policy switching** - `rl_type` force-includes its collections, and a
  policy change mid-run is a distribution change.
- **Sequence-length curriculum** - tiers shift the input distribution over time
  by design.
- **The KB spider** - `build/spider.db` grows, so new content arrives after
  earlier content has already been trained on. That is a continual stream.
- **`$LEEP` / resume-from-peer checkpoints** - inheriting a peer's weights and
  continuing on different data is textbook domain-incremental.
- **Any online / deployed learning story**, which is what the ecosystem is
  ultimately for.

So the honest position is: not a problem today because of the sampler, plausibly
a problem for everything the roadmap wants to become, and possibly *worse* at
our scale rather than better.

---

## What to actually do, cheapest first

### 0. Measure whether we forget at all (gate for everything below)

Nothing here should be built before this reports. Take an existing `small-*`
config, train sequentially over `DATASET_COLLECTIONS` members instead of the
shuffled mixture, and record per-collection validation BPB after each stage.
BWT is then just the mean drop on earlier collections. Run the shuffled mixture
as the control.

Three outcomes, three different next steps:

- **Near-zero forgetting** - the tiny-model hypothesis holds, the whole thread
  closes, and that is a genuinely interesting negative result worth writing up
  (it would also be a direct counterexample to the paper's framing at small
  scale).
- **Forgetting comparable to their Transformer baseline** - go to (1).
- **Forgetting *worse* than their baseline** - the interference argument wins
  over the capacity argument, which is the most informative outcome and the one
  that would justify real architectural work.

This costs an afternoon of GPU time and needs no new mechanisms. It is also
reusable: a forgetting metric belongs in the metrics registry regardless of what
we decide.

### 1. Weight-protect as an optimizer wrapper

The one piece of the paper that is fully separable from its architecture.
`I += |W_t - W_{t-1}|`, `ΔW /= (1 + λI)` needs no gradient, no Fisher, no
domain boundaries, and no architecture change. It drops into
`praxis/optimization/wrappers.py` as a `WRAPPER_REGISTRY` key and composes
through `SequentialWrapper` exactly as `low_rank_moment` and `half_lion`
already do. Roughly 100 lines.

Notably it is a continual-learning mechanism that works *with* backprop, which
is the configuration the authors never test because their model cannot do
backprop. Their λ=5.0 is a tuned constant and must not ship as a flag - derive
it from something endogenous (the optimizer metrics suite already computes
update RMS and update/weight ratio) per the no-hyperparameter-tuning rule.

### 2. The two-tier competitive memory as a `praxis/memory` sibling

Content-addressed slots with threshold-gated competitive writes is a Titans
cousin, and the machinery for parameters that update inside forward under
`no_grad` already exists - `praxis/memory/neural_memory.py:235` returns
`torch.no_grad()` in energy mode for precisely this. A CMP-style memory as a
`--memory-type` profile is a well-scoped experiment. The novel bit is the
calibrated write threshold (a slot fires only on genuine match, not baseline
overlap), and it is independent of the delta rule.

### 3. The delta-rule readout as a `ParallelHead` arm

If we want to test their claim rather than reason about it: a linear CMP readout
as a second branch, updated under `no_grad` by the delta rule, blended
detach-in-blend against the backprop head - the same contract as `HaloHead` in
prismatic5. The gate share is the verdict. If the delta arm carries meaningful
weight after a domain switch that is evidence; if it collapses to zero that is
also an answer. This is the only version that yields a signal without betting
the architecture on it.

**Not recommended:** replacing the trainer.

---

## The transferable idea

Stripped of the architecture: *dense global credit assignment is what
overwrites; sparsify the update and forgetting drops.* Praxis already expresses
fragments of this without having framed them that way - mono-forward's graph
cuts (`praxis/decoders/mono.py`) are local credit assignment, PEER's sparse
banks are sparse updates, SMEAR's soft routing partitions which parameters move
for which inputs. If the measurement in (0) says we forget, the first place to
look is not CMP but whether *those existing knobs* already modulate forgetting.
`--mono-type layer` versus no cuts, at fixed everything else, is a nearly free
ablation on a question the paper says should matter.

Related: [sparse_activation_spiking.md](sparse_activation_spiking.md) (sparsity
as the shared theme), [temperament_bias_variance.md](temperament_bias_variance.md)
(where inductive-bias reliance sits on the axis), and the `$LEEP` and
homeostatic-engagement roadmap items (both of which need a continual story to
work at all).
