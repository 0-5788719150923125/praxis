# Exact solve: retrieve a small model per input, write to it by solving

> Status: **grounded reading capture** (2026-08-10). Opened from the question
> "what if the optimizer solved for the batch exactly instead of drifting toward
> it?" Five literature lenses were surveyed and each adversarially audited; every
> citation below either survived that audit or is marked. The verdict is mixed in
> a useful way: the framing is wrong, most of the components are old, one
> algebraic bug would have wasted a month, and there is exactly one unoccupied
> cell that is worth building.

The idea as first stated: a transformer whose weights are a large sparse bank;
PEER-style product-key retrieval assembles a small expert set per forward pass;
those experts merge in **parameter** space (SMEAR) so each pass instantiates a
different small model; and the optimizer gets the same treatment - for each batch,
select a sparse parameter subset and **solve exactly** for the values that make
the model reproduce that batch, in one step, no drifting. Hypothesis: sparsity
prevents the collapse you would normally get from exactly fitting every batch,
because you never touch all the weights at once.

---

## 1. The category error, settled first because it decides everything

**Backpropagation is a credit-assignment mechanism.** It computes `∂L/∂θ` through
a composition. **Gradient descent is a step rule.** "Solve exactly" replaces the
step rule. It is not an alternative to backprop in any technical sense, and three
independent arguments say so:

- **Closed forms only exist where the parameters enter the loss linearly.** ROME
  gets its closed form by freezing everything and treating one MLP matrix as a
  linear associative memory. ELM, ACIL and FORCE get theirs by freezing the
  features. Galashov et al. (arXiv:2510.04606) get theirs on the last layer,
  under squared loss, with the backbone still trained by backprop. *Every closed
  form in this literature is bought by refusing to look below the current layer.*
  For any parameter behind a nonlinearity, "solve exactly" means solving a
  nonlinear system, which is Gauss-Newton, which needs the Jacobian, which comes
  from reverse-mode AD.
- **The step-rule identity is proven.** Needell, Srebro & Ward (NeurIPS 2014,
  arXiv:1310.5715) recast randomized Kaczmarz as SGD with importance sampling and
  a specific step size. (Caveat that survived audit: it converges to a *weighted*
  least-squares problem; partially-biased sampling is needed to recover the
  original solution. Do not overstate it as an exact equivalence.)
- **Every working system in the family is two-timescale with a backprop-trained
  outer loop.** DeltaNet, Gated DeltaNet, TTT, Titans, the mesa-layer, ACIL,
  closed-form last-layer. None of their authors claims otherwise. Wang, Shi & Fox
  (arXiv:2501.12352) show they are all one recipe - memorize by solving a
  test-time regression, then retrieve - differing only in function class and inner
  optimizer.

If this is ever pitched as "replaces backpropagation," readers discard the parts
that are good. It is an alternative to **SGD**, and that is a smaller but real
claim.

**The honest ceiling on the speed argument.** Abreu, Vyas, Kakade & Morwani,
*The Potential of Second-Order Optimization for LLMs: A Study with Full
Gauss-Newton* (arXiv:2510.09378), 45M/150M LLaMA on C4: **5.4x fewer training
iterations than SOAP and Muon**, per-step cost 4-5x, wall-clock 15-30 days against
2-3 days for AdamW/Muon. The authors explicitly disclaim it as a practical
optimizer. Their useful finding for us is the other one: *layerwise* Hessian
structure captures most of the available gain, which is an argument for the
block/routed framing specifically.

So: solving buys fewer steps and costs more wall clock. Whatever this thread is
for, it is not "training gets fast."

---

## 2. What is old

### The architecture half

- **Product-key memory**: Lample, Sablayrolles, Ranzato, Denoyer, Jégou, NeurIPS
  2019 (arXiv:1907.05242). The √N factorized-key retrieval is 2019, not 2024. It
  also owns the dead/collapsed-key pathology, fixed in-paper with BatchNorm on the
  query - which is why `praxis/dense/peer.py:202` has a BatchNorm we inherited
  without earning.
- **PEER**: Xu Owen He, *Mixture of A Million Experts*, arXiv:2407.04153.
- **Already far past PEER in scale**: Memory Layers at Scale (arXiv:2412.09764,
  128B memory params, 1T tokens), Monet (arXiv:2412.04139, 262144 experts/layer,
  ICLR 2025), UltraMemV2 (arXiv:2508.18756, 120B total / 2.5B activated).
- **The negative result to read first**: UltraMem (arXiv:2411.12364, ICLR 2025)
  only matched 2-expert MoE and fell short of 8-expert configs. UltraMemV2's fix
  was to *widen the values* (FFN-style value processing, adopted from PEER), and
  its headline is that **activation density matters more than total sparse
  parameter count**. That is a direct strike at "a million tiny experts."
- **Parameter-space merging**: SMEAR (arXiv:2306.03745, TMLR). Scaled to
  autoregressive LM pretraining by **Lory** (arXiv:2405.03133, COLM 2024): 32
  experts, 30B total / 1.5B active, 150B tokens, +13.9% ppl over parameter-matched
  dense. **Lory abandoned per-token merging as prohibitively expensive and fell
  back to causal segment routing.** Read that sentence twice; §6 depends on it.
- **"Therefore it's a hypernetwork"** is correct and is the trivial case. A
  router-weighted sum of bank entries is a *linear* hypernetwork whose embedding
  is the routing distribution (Ha et al. 2016; Jayakumar et al., ICLR 2020, place
  gating/attention/hypernets/dynamic-convs in one multiplicative-interaction
  family). It is the definition of that family, not a discovery.

### The optimizer half

Oldest part of the idea. In order of directness:

- **Kaczmarz (1937)** / Randomized Kaczmarz (Strohmer & Vershynin, JFAA 2009).
  Project onto the affine set where the current constraint holds exactly. Block
  Kaczmarz = do it for a minibatch. This is the proposal, for linear systems, with
  a complete theory.
- **POCS / alternating projections** (von Neumann 1933; Bauschke & Borwein, SIAM
  Review 1996). Non-intersecting sets give an **order-dependent limit cycle**, not
  convergence.
- **Sequential exact fitting has forgetting bounds**: Evron et al., COLT 2022
  (arXiv:2205.09588) and ALT 2026 (arXiv:2504.04579). Good news and bad news
  together - forgetting *is* bounded, and the good rates require **repetition and
  shuffling**, i.e. exactly the "drifting toward convergence over many steps" the
  idea wanted to eliminate.
- **Sparse exact projection already exists**: Schöpfer & Lorenz, *Linear
  convergence of the Randomized Sparse Kaczmarz method*, Math. Prog. 173:509-536,
  2019. Converges to the min-L1 solution at a proven linear rate. If we want
  sparsity to *fall out of* the solve rather than be imposed by a router, this is
  the machinery and it is a decade old.
- **Min-norm exact fit in ML**: Online Passive-Aggressive (Crammer et al., JMLR
  7:551-585, 2006): `min ‖w − w_t‖² s.t. loss_t(w) = 0`, closed form. **The same
  paper introduces PA-I and PA-II slack variants because the hard-constraint
  version is destroyed by one noisy example.** Nobody uses PA-0.
- **Sixty years of the same thing in DSP**: NLMS (Nagumo & Noda 1967) and the
  Affine Projection Algorithm (Ozeki & Umeda 1984). µ=1 is the exact projection;
  every deployed system uses µ<1 plus an ε in the denominator.
- **Regularized, with theory**: implicit SGD (Toulis & Airoldi, Ann. Statist.
  45(4), 2017), aProx (Asi & Duchi, SIOPT 29(3), 2019). The entire theoretical
  content is that the proximal term `(1/2η)‖θ − θ_t‖²` buys the stability. The
  proposal is the η→∞ limit, where the guarantees vanish.
- **Deep-net version, already implemented**: Stochastic Polyak Step (Loizou et
  al., AISTATS 2021) and ALI-G (Berrada et al., ICML 2020). Both compute the step
  that zeroes the current batch's loss, and both take a *fraction* `c` of it with a
  hard cap. ALI-G's published result is "comparable performance with SGD."
- **Second-order form**: the exact solve on a nonlinear net *is* the undamped
  Gauss-Newton step. Levenberg 1944 / Marquardt 1963; Hagan & Menhaj 1994
  (explicitly: efficient only for nets with "no more than a few hundred weights");
  Martens ICML 2010 (adaptive damping *is* the method). Restricting the solve to a
  coordinate subset is Randomized Subspace Newton (Gower et al., NeurIPS 2019).
- **In an LLM, on a sparse support, in closed form**: ROME (NeurIPS 2022) and
  MEMIT (ICLR 2023).
- **As a forward-pass rule, at production scale**: the delta rule
  `W ← W − β(Wk − v)kᵀ` with β=1 and normalized k **is** the Kaczmarz/NLMS
  projection in parameter space, applied to a per-input-instantiated small model.
  Schmidhuber 1992 → Schlag et al. ICML 2021 → DeltaNet (NeurIPS 2024, 1.3B params
  / 100B tokens) → Gated DeltaNet (ICLR 2025). **It ships. With learned β<1 and a
  decay gate.**
- **The fixed-feature limit**: ELM (2006), FORCE (2009), ACIL (NeurIPS 2022), LSSE
  (ICLR 2024). All work. All freeze the representation. This is the reference class
  the pure version falls into, and it has a 20-year record of losing to backprop
  for exactly one reason: no representation learning.

---

## 3. The bug that would have cost a month

**Parameter-merging PEER's single-neuron experts collapses the layer to one
neuron.** Three lines, no literature required, and it applies to
`praxis/dense/peer.py` as written.

Our expert is `e_i(x) = up_i · act(x · down_i)` - rank-1, `ROWS_PER_EXPERT = 2`.

- Output merge (what we do today, `EmbeddingBag` at `peer.py:353`):
  `Σᵢ gᵢ upᵢ act(downᵢᵀx)` → **k distinct nonlinear features.**
- Parameter merge (what the idea proposes):
  `(Σᵢ gᵢ upᵢ) · act((Σᵢ gᵢ downᵢ)ᵀx)` → **one nonlinear feature, for any k.**

Retrieving 512 experts instead of 8 buys a smoother interpolation of a single
neuron's direction. **PEER + SMEAR as stated is strictly worse than PEER.** Two
auditors found this independently. Note also that for *linear* experts the two
merges are identical, so the whole distinction only exists in the nonlinear case,
and in the nonlinear case it costs capacity.

The fix is experts of internal width r > 1, and it is the same fix UltraMemV2
found by measurement. But then the arithmetic has to be faced honestly:

| | features | FLOPs per token |
|---|---|---|
| output merge, k experts of rank r | k·r | k·r·d |
| parameter merge, k experts of rank r | r | k·r·d (merge) + r·d (apply) |

**Parameter merging never buys expressiveness. It buys compute amortization, and
only when the merge is amortized over more than one position.** Merge once per
segment of length L, apply per token: cost becomes `k·r·d/L + r·d`, a speedup of
roughly k when L >> k. That is the entire argument for it, and it is why Lory does
segment routing. Per-token merging over a million experts is the configuration
Lory could not afford with 32.

---

## 4. What actually survives

After five audits the novel residue is **one cell of a table**, not a paradigm:

**Router-determined sparse support for a constrained closed-form write, during
pretraining rather than editing.** Input-determined update support is published -
WISE (NeurIPS 2024, arXiv:2405.14768) trains "a router to decide which memory to go
through" with "a knowledge-sharding mechanism where different sets of edits reside
in distinct subspaces of parameters"; MoM (arXiv:2502.13685) routes tokens to
independent delta-updated memory states explicitly to minimize interference; GRACE
(NeurIPS 2023) uses content-based key retrieval for the edit support; sparse memory
finetuning does it with gradients. Nobody has crossed router-determined support
with an exact/constrained solve **as the training rule from scratch**.

Two adjacent things also unoccupied, in descending order of interest:

- **The open empirical question nobody has answered.** Does routed *sparsity*
  degrade more slowly than null-space *orthogonality* out to 10⁴-10⁶ sequential
  updates? Both degrade. Nobody has run the head-to-head. That is cheap and more
  interesting than either half of the original idea.
- **PEER-granularity retrieval + parameter-space merge of the retrieved top-k.**
  Unpublished, but see §3: as literally specified it is degenerate, and the fixed
  version is MEO-with-a-product-key-router, i.e. a granularity increment.

**And the one genuinely fresh *argument*, which is ours and is already in the
notes.** [`project_peer_sparse_optimizer`] records that the wall on PEER is
**optimizer state on the banks**. A solve-based bank update deletes `m` and `v`
entirely. Every backprop alternative that survived contact with measurement won on
**memory**, never on compute. That is the framing that would make this a paper.

The audit's correction, which must be priced in before claiming it: a *recursive*
exact solve carries a covariance, and that **is** optimizer state, O(d²) per slot
against Adam's O(2·params). Decoupled EKF (Puskorius & Feldkamp 1991) exists
precisely because the full covariance was intractable. A *stateless* one-shot solve
carries no state but also no memory across batches, which defeats the purpose.
There is a real crossover point here and it is computable on paper before anything
is built.

---

## 5. The objections, ranked

### 5.1 Language is not a function, so there is no batch to fit exactly

Fatal to the word "exactly," and specific to LM. Two blows:

**(a) Cross-entropy has no finite exact minimizer.** "Reproduce the batch exactly"
means probability 1, which means logits → ∞. The target of the solve does not
exist in parameter space. Either fit soft logit targets - in which case "exact" is
an arbitrary target with a hyperparameter smuggled into it - or diverge.

**(b) Even if it existed it is the wrong answer.** The conditional distribution
over the next token has large irreducible entropy. A training token is a *sample
from a distribution, not a value of a function*. The Bayes-optimal model has high
loss on every batch; that loss is the entropy of language and it is where the
scaling laws asymptote. Exact fitting is not an aggressive route to the right
answer, it is a precise specification of the wrong one - and its magnitude is set
by the noise rather than by a learning rate, so there is no small parameter to turn
down.

This also voids the one theory that would have licensed the design: SPS and ALI-G
require the **interpolation regime** (f* known and zero). LM pretraining over
trillions of tokens is not in it. (DecSPS, NeurIPS 2022, converges without
interpolation and without knowing f* - but it is a damped, decreasing-stepsize
method, the opposite of "one step, no drifting.")

**Where this points**: the one place in this repo where the exact solve is
well-posed is the one place we already built it. `LinearPrior`
(`praxis/heads/energy.py:48`) solves for a *latent* under squared loss. That is not
a coincidence, it is the structure of the problem choosing its own home.

### 5.2 Damping is not a detail, it is the method

The single most robust finding across all five lenses. Every survivor is damped and
the undamped limit is the setting each literature specifically walked away from:

| Method | The damping |
|---|---|
| Gauss-Newton | Levenberg-Marquardt λ / trust region |
| Hessian-free | adaptive damping |
| Passive-Aggressive | PA-I / PA-II slack |
| NLMS / APA | µ<1 and ε in the denominator |
| SPS | factor c (typically ½) plus hard cap |
| ALI-G | max-step cap |
| DeltaNet → Gated DeltaNet | learned β<1 plus decay gate |
| Titans | momentum **and** an explicit forget gate |
| MinSR | pseudo-inverse eigenvalue cutoffs |
| SPRING | project and **carry** the previous increment |
| ROME / MEMIT | preservation term from a precomputed key covariance |
| AlphaEdit | null-space projection |
| implicit SGD / aProx | the whole theory *is* the proximal term |

SPRING is the most instructive: Goldshlager, Abrahamsen & Lin (JCP 516, 2024) beat
MinSR and K-FAC on neural wavefunctions specifically by **not** re-solving from
scratch each batch, and still needed 40000 iterations for chemical accuracy on an
oxygen atom. "No drifting toward convergence" is the design choice everyone who has
shipped this has explicitly reversed.

### 5.3 Sparsity is the wrong invariant

Interference is governed by overlap in the **activation/key** subspace, not by
disjointness of the parameter set.

- Aljundi et al., *Selfless Sequential Learning* (ICLR 2019) tested **parameter**
  sparsity against **representation** sparsity for forgetting directly.
  Representation sparsity is the one that helps. A router imposes the weaker one.
- Masse et al. (PNAS 2018) gated ~80% of units off per task and **still needed
  synaptic stabilization**; gating alone was not enough for 100 tasks.
- The winning line is orthogonality: OWM (Nature MI 2019), OGD, GPM, Adam-NSCL,
  and in LLM editing **AlphaEdit** (ICLR 2025 Outstanding Paper, arXiv:2410.02355)
  which got +36.7% from one extra line projecting the update onto the null space of
  preserved keys.
- Even with literally non-overlapping experts, they write into the same residual
  stream and are read by the same downstream layers. Function-space interference
  survives weight-space disjointness.

**This is a "wrong mechanism," not a refutation.** *Continual Learning via Sparse
Memory Finetuning* (arXiv:2510.15103) reports NaturalQuestions F1 drop after
learning new facts of **89% full FT, 71% LoRA, 11% sparse memory finetuning** at
matched acquisition. Two caveats that decide whether we inherit the result: slots
are chosen by a TF-IDF **interference** criterion rather than the ordinary router,
and the updates are **gradient steps, not exact solves**. The follow-up
(arXiv:2604.05248) swaps TF-IDF for a KL selector and concludes it is "validating
the sparse update hypothesis."

Correct statement: sparsity is a crude uncontrolled proxy for orthogonality;
null-space projection is orthogonality by construction and costs one line.

### 5.4 The tractability argument is false for a language model

"The Gram is batch × batch" is the claim that makes the min-norm Gauss-Newton step
look cheap, and it does not hold here. The Gram is (n·d) × (n·d) where d is the
residual dimension **per sample**. MinSR works because a neural wavefunction emits
one scalar per sample. D-NGD works because PINN residuals are low-dimensional. A
transformer emits T×V logits per sequence and the softmax GGN couples all V. Korbit
& Zanon's *Fast Gauss-Newton for Multiclass Cross-Entropy* exists entirely because
of this, and its fix is to discard the within-competitor covariance to force one row
per example - buying tractability by throwing away curvature, exact only in the
binary case.

The escape hatch that does exist: Abreu et al. never form the Gram at all -
matrix-free JVPs plus an inner optimizer, at 4-5x per-step cost.

(One over-claim to retire while we are here: "unknowns ≪ constraints, therefore no
exact solution" does not follow as stated, because PA solves an *inequality*
constraint and those feasible sets are routinely nonempty. What does follow: what
you get is a least-squares projection, and a least-squares projection with
trust-region control **is a damped Gauss-Newton step** - the thing this was meant to
replace.)

### 5.5 Exactness may be anti-generalization

Contested and regime-dependent, worth instrumenting rather than deciding.

For: Wadia, Duckworth, Schoenholz, Dyer & Sohl-Dickstein (ICML 2021,
arXiv:2008.07545), verbatim - *"For a general class of models, namely models with a
fully connected first layer, we prove that the information contained in [the
sample-sample second moment matrix] is the only information which can be used to
generalize"* - and whitening or second-order optimization removes access to it.
That matrix is precisely the batch-by-batch Gram we would invert. Benzing (ICML
2022) found K-FAC significantly outperforms **true** second-order updates - i.e.
exactness is not a target we are approximating toward. Buffelli et al. (NeurIPS
2024) built reversible architectures so exact GN was tractable: it overfits each
minibatch, the NTK barely moves, training loss itself saturates.

Against: Abreu et al. ran *damped, approximately-solved, large-batch* GN on real
transformers with no lazy collapse. Goldwaser & Ge measured the price of full
laziness at 61.3% vs 69.4% on CIFAR-10, and found surprisingly few feature updates
recover most of standard training.

**Instrument the sample-sample second moment, not only the NTK.** Wadia's mechanism
is more general than laziness and it names the exact matrix the design inverts.

### 5.6 The router/interference tension - genuinely two-sided

The argument: a content router sends *similar* inputs to the *same* experts, and
similar inputs are exactly the ones whose exact fits conflict, so the router
optimizes for the overlap the hypothesis needs absent. Reinforced by representation
collapse in sparse MoE (Chi et al., NeurIPS 2022) and product-key dead slots.

The counter is stronger than the surveys allowed: MEMoE (arXiv:2405.19086)
engineers "knowledge anchor routing [so] that inputs requiring similar knowledge are
routed to the same expert, thereby **enhancing generalization**." MoM reports that
routing to separate memory states *reduces* interference. Collision is a
generalization mechanism as often as an interference mechanism, depending on whether
the colliding items want the same output.

Hypothesis to measure. Not a theorem in either direction.

---

## 6. What did NOT survive the audit

Three of the five lenses leaned on sequential model editing as the decisive
refutation - "this has been run and it collapses." **That argument is wrong and it
would have talked us out of the one thing worth building.**

- **r-ROME** (Gupta, Baskaran, Anumanchipalli, EMNLP 2024, arXiv:2403.07175),
  verbatim: disabling edits "are an artifact of irregularities in the
  implementation of ROME... we provide a more stable implementation ROME, which we
  call r-ROME and show that **model collapse is no longer observed when making
  large scale sequential edits**."
- **UltraEdit** (arXiv:2505.14679): training-, subject- and memory-free lifelong
  editing, **up to 2M sequential edits**.
- GRACE and Transformer-Patcher (ICLR 2023): thousands of sequential edits with
  locality preserved. WISE: routed disjoint subspaces merged without conflict.
- AlphaEdit's reproducibility study (arXiv:2606.26783) **reproduced** the original
  results; its finding is that null-space protection is bounded rather than
  unconditional.
- The verified collapse number is **~1400 sequential edits for MEMIT on GPT-J**,
  with MEMIT forgetting ~3x fewer facts than ROME (Findings of ACL 2024,
  arXiv:2401.07453). **The widely-repeated "~250 edits" figure is unsourced. Do not
  use it.**

And structurally: ROME/MEMIT/AlphaEdit edit **dense** matrices at one or a few
layers. That is *locality*, not *routed sparsity*. Different branch. Every
routed-sparse system that has been tested reports large interference reductions or
none at all.

**Honest status of the hypothesis: untested at the relevant scale, with published
fixes for every failure mode the dense branch hit.**

One caution that cuts against everyone, including the fixes: Liu et al., *Is Model
Editing Built on Sand?* (arXiv:2510.00625) find editing successes largely exploit
hidden shortcuts and that state-of-the-art methods "collapse even under the simplest
negation queries." Run that sanity check early, not late.

### Two claims to retract before writing anything up

- **"The merged layer is affine in the bank, so the solve is genuinely well-posed."**
  Merging is a *linear reparameterization* of a layer MEMIT already treats as linear
  in its parameters. Least-norm in the bank rather than in W is a **preconditioner**,
  not new well-posedness. And when the router moves, the objective is **bilinear** in
  (p, bank), not affine; composed over L layers it is degree-L polynomial.
- **"Exact solves carry no optimizer state."** True only for the stateless one-shot
  version, which has no memory across batches. See §4.

### The algebra that does survive, and is worth logging

For a fixed router distribution `p`, the minimum-norm way to push a desired
`ΔW_eff` back onto the bank is `ΔWᵢ = (pᵢ/‖p‖²)·ΔW_eff`, so the total squared
displacement across the bank is

```
Σᵢ ‖ΔWᵢ‖²  =  n_eff · ‖ΔW_eff‖²      where  n_eff = 1/‖p‖²
```

`n_eff` is the router's participation ratio. This is a preconditioner identity, not
a well-posedness result - but it says the damage per write is linear in how many
experts the router spread across, and `1/‖p‖²` is one line beside the entropy
already computed at `praxis/routers/smear.py:503`. **Log it. It is the interference
predictor and it is free.**

---

## 7. The buildable version

Keep backprop for the slow weights. Move the solve into the forward pass. Replace
"be sparse" with "be orthogonal."

1. **Rank-r experts.** `r ≈ 4-16`, forced by §3, validated at 120B by UltraMemV2.
   Budget them the way `peer.py` already budgets `glu`: pay for r out of the expert
   **count**, so `ROWS_PER_EXPERT = 2r` and the bank stays capacity-matched.
2. **Merge the retrieved top-k in parameter space, amortized over a segment.** Per
   §3 the merge only pays when applied to more than one position. Lory's lesson,
   confirmed by our own FLOP count.
3. **Delta-rule write into the retrieved bank rows**: `W ← W − β(Wk − v)kᵀ`. This
   *is* the Kaczmarz projection applied to a per-input-instantiated small model -
   literally "each forward pass instantiates a different small model whose weights
   are set by solving." We already have this shape as prismatic fast weights
   ([`project_prismatic_fast_weights`]); wire it onto the PEER bank rows instead of
   a separate matrix.
4. **Learn β from an endogenous signal, never β=1, plus a decay gate.** This is
   exactly what separates DeltaNet from Gated DeltaNet, and it satisfies the
   no-hyperparameter-tuning rule rather than violating it.
5. **Project the write onto the null space of a running second moment of
   previously-retrieved keys** (AlphaEdit/OWM). Uniquely natural here: the
   product-key router already hands us the keys, and the statistic lives in key
   space (d×d per layer), not parameter space.
6. **Select slots by an interference criterion**, not the raw router score. This is
   the difference between the published 11% and an unknown.
7. **Router, bank and β all trained by ordinary backprop.** This is the property
   that makes the whole thread cheap to be wrong about: **if the solve is useless,
   the outer optimizer drives β to zero and we have lost nothing but the FLOPs.**
8. Take Berges et al.'s engineering wholesale rather than rediscovering it:
   input-dependent silu gating, **qk-normalization** (they needed it for stability
   as memory grew), a memory pool shared across memory layers, and **at most ~3
   memory layers** - they measured degradation past that. Do not make the whole
   transformer a bank.

If a solve stays in the optimizer at all, it must be **constrained, not merely
sparse**: solve the ridge/proximal version and take a step of size β<1 toward it.
As λ→∞ that is a gradient step, so it is a strict generalization of SGD with one
knob, and that knob is a real research object rather than a tuned constant.

**Frame the result as optimizer-state / memory, not as backprop replacement.**

---

## 8. The first model, sketched

Deliberately boring everywhere except the one axis under test. Byte-level
throughout, so the tokenizer is not a variable.

### Config

`small-a` already gives us `encoder_type: byte_latent_conv` and
`tokenizer_type: byte_level`. Strip everything else that could explain a result:

```yaml
# experiments/exact-a.yml  (NOT written to experiments/ yet - the registry
# entries below do not exist, so this would error on launch)
extends: small-a
depth: 4                  # was 8 - we are measuring a layer, not a curve
num_layers: 2
attention_type: standard  # was arc; remove the memory confound entirely
block_type: transformer   # was recurrent-ish; no depth reuse
halting_type: none        # was kl; fixed compute per token
norm_type: sandwich       # keep - required with bias in recurrent depth
ffn_type: peer            # ARM A (baseline). B: peer_merge. C: peer_solve.
optimizer_wrappers: []    # no schedule_free; a plain decayed baseline
```

Three arms, one line apart, per the registry-over-flags rule:

```python
# praxis/dense/__init__.py
DENSE_REGISTRY = dict(
    ...,
    peer=ParameterEfficientExpertRetrieval,          # ARM A: output merge, rank-1
    peer_merge=partial(RetrievedFastWeight, write=False),   # ARM B: param merge, rank-r
    peer_solve=partial(RetrievedFastWeight, write=True),    # ARM C: B + delta write
)
```

### The module

`praxis/dense/solve.py`, subclassing `BaseDense`, reusing PEER's query net and
product keys **verbatim** so retrieval is not a variable either.

```python
RANK: int = 4          # expert internal width. r=1 reproduces the §3 bug on purpose.
ROWS_PER_EXPERT = 2 * RANK   # keeps the bank capacity-matched, exactly as `glu` does
SEGMENT: int = 32      # merge once per segment, apply per token (§3, Lory)

class RetrievedFastWeight(BaseDense):
    # banks, as [N, r, d] instead of PEER's [N, d]
    #   self.down : nn.Parameter [N, r, d]
    #   self.up   : nn.Parameter [N, d, r]
    #   self.beta : nn.Linear(d, 1)      -> learned write rate, sigmoid, never 1
    #   self.key_cov : buffer [d, d]     -> running second moment, for null space

    def forward(self, x):                      # x: [b, n, d]
        p, idx = self.retrieve(x)              # PEER query net + product keys, unchanged
        p = p.view(b, n // SEGMENT, SEGMENT, k).mean(2)   # segment-level routing

        D = einsum("b s k, b s k r d -> b s r d", p, self.down[idx])
        U = einsum("b s k, b s k d r -> b s d r", p, self.up[idx])

        z = act(einsum("b s r d, b s l d -> b s l r", D, x_seg))   # [b, s, L, r]
        y = einsum("b s d r, b s l r -> b s l d", U, z)

        if self.write:
            self.delta_write(idx, p, z, x_seg, y)   # under no_grad
        return y

    @torch.no_grad()
    def delta_write(self, idx, p, z, x, y):
        beta = sigmoid(self.beta(x)).mean()          # learned, endogenous, < 1
        k = self.null_project(z)                     # AlphaEdit: k <- (I - C(C+λI)^-1) k
        residual = self.target(x) - y                # v - Wk
        dU = -beta * einsum("...d, ...r -> ...dr", residual, k)
        self.up.index_add_(0, idx, dU / n_eff)       # spread by participation ratio
        self.update_key_cov(k)
```

Everything above is deliberately the *damped, gated, projected* version. The
undamped one is a `beta = 1.0` line, and it exists so we can measure the thing every
literature says will blow up rather than take their word for it.

### Metrics, co-located on the module

`training_metrics()` on `RetrievedFastWeight`, per the co-location rule:

- `solve/n_eff` - router participation ratio `1/‖p‖²`. The interference predictor.
- `solve/rank` - numerical rank of the layer output over a batch. **This makes
  Test 0 a standing metric instead of a one-off script.**
- `solve/delta_ratio` - `‖ΔW‖/‖W‖` per write. The disabling-edit tripwire.
- `solve/beta` - mean learned write rate. **This is the verdict.** If it goes to
  zero, the solve is useless and the model said so itself.
- `solve/slot_overlap` - fraction of retrieved slots shared with the previous
  batch. Measures whether the router is separating content or colliding it (§5.6).
- `solve/second_moment_trace` - Wadia et al.'s quantity (§5.5).

---

## 9. The test ladder, with kill criteria

Strict order. Stop at the first failure. Under a day, total.

**Test 0 - ten minutes. Is the architecture even well-posed?**
One PEER layer, fixed input, compute output-merge and parameter-merge results,
sweep k ∈ {1, 4, 16, 64}, measure output rank over a batch.
**Kill:** parameter-merge rank flat in k while output-merge rank grows → §3
confirmed on our own code; build the rank-r version before writing another line.

**Test 1 - one afternoon. Does "fit the batch now" have any headroom at all?**
Run **SPS_max** or **ALI-G** as a drop-in optimizer on `small-b`, against a
**fully-decayed-LR** AdamW *and* Muon baseline. SPS is the damped, capped,
published, one-line version of the proposed optimizer.
**Kill:** if the damped 1-D version does not beat a properly decayed baseline, the
undamped full-rank version will not either; it will just fail louder. The decay is
not optional - Kaddour et al., *No Train No Gain* (arXiv:2307.06440) showed an
under-decayed baseline is how optimizer papers accidentally claim 2x.

**Test 2 - half a day. Is the exact step numerically admissible on a routed
support?**
One batch, routed support S, compute the min-norm GN step that would zero the loss.
Log `‖δ‖/‖θ_S‖` and `σ_min(J_S)`. **Run it with ridge damping and the null-space
projection in the loop**, otherwise we are measuring an undamped straw man nobody
would build.
**Kill:** relative update in the tens of percent, or `σ_min` near machine precision
→ we reproduced the disabling-edit condition at step one.

**Test 3 - the actual question, once 0-2 pass.** Sequential updates out to 10⁴;
plot held-out BPB against update count and against `n_eff`. Drift in *unrelated*
text perplexity is the reliable early-warning signal. Published anchors: ~1400
(MEMIT), thousands (GRACE, Transformer-Patcher, r-ROME), 2M (UltraEdit).

**The control that cannot be skipped: random slot selection at matched sparsity.**
Hash Layers (Roller et al., NeurIPS 2021) matched or beat Switch and BASE with a
*fixed random hash* and no learned router; Su et al. (NeurIPS 2020) found shuffling
unmasked weights within a layer does not hurt. If the learned product-key router
does not beat a hash into the bank, there is no result.

---

## Open questions

- What does `calm_prior_r2` actually read? `next/linear_solve.md` already set the
  decision rule for the shipped solve-into-weights version and it has never been
  looked at. If r² is near zero the backbone does not linearize the sequence and
  this entire thread is answered upstream, for free.
- Does the `n_eff` identity in §6 hold empirically once the router co-adapts, or
  does the router learn to peak precisely to evade the write?
- Does routed sparsity or null-space orthogonality degrade more slowly out to 10⁴
  updates? This is the publishable question and neither half of the original idea
  is needed to ask it.
- Does the write compose with `--mono-type layer`? Both attack dense credit
  assignment; `next/continual_learning.md` already flags the same composition
  question for block-sparse updates.

## The framing, held to our own standard

`lottery_engineering.md` ran this exercise on itself: "Praxis proves LTH" was a
category error and did not survive; "Praxis engineers the lottery" did, and was
*stronger* for being narrower. Same move applies here. "A tiny model more powerful
than anything the world has ever seen" is unfalsifiable and is not a research
target. "Routed retrieval plus a constrained closed-form write removes optimizer
state from the banks, and interference scales as the router participation ratio" is
falsifiable, is unoccupied, and is measurable on a single GPU.

Also worth stating plainly, since strong LTH is what the idea leans on rather than
LTH: Pensia et al.'s O(log(dl)) width factor **and** 2x depth is a hard floor. The
bank must be strictly larger and deeper than the dense model it emulates, and the
*active* subnetwork is target-sized, not tiny. That is compatible with this design
(big bank, normal-sized slice) but it means **"tiny" here is tiny active FLOPs, not
a small file on disk**. And every SLTH proof assumes arbitrary per-weight masks; a
product-key top-k index over a Cartesian sub-key grid expresses a vanishing fraction
of those, and the mask is input-dependent rather than fixed. No SLTH theorem covers
either restriction. The theory does not reach the architecture.

Related: [linear_solve.md](linear_solve.md) (the shipped version and its gate),
[continual_learning.md](continual_learning.md) (the same diagnosis, different cure),
[regime_gated_priors.md](regime_gated_priors.md) §3 (block-sparse updates),
[lottery_engineering.md](lottery_engineering.md) (what LTH does and does not say
here), [peer_bridge.md](peer_bridge.md), [prismatic.md](prismatic.md).
