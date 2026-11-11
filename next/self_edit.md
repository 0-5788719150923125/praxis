# Self-edit: can a small model reshape its own weights?

> Status: **on hold, measured on one checkpoint, no learned policy built** (2026-11-04).
> Opened from the weight-space programming post and the question "could a tiny
> Praxis model edit itself, instead of being edited by something larger?" Every
> number here comes from frozen `gpt2-1` (5.3M params, run `62e91aa30`), in bits per
> token, on three corpora. Treat them as a map of where the gain is, not as a
> verdict on a trained programmer. That test is still open (Section 8, test 1).

The idea as first stated: a model emits tokens that edit its own weights, in the
style of chain of thought, and processes the input again with the changed weights.
Optionally, a GAN discriminates teacher responses from student responses, where the
teacher is self-distillation from training examples. Hypothesis: a small weight
space is easier to reason about, so a good enough RL task could teach a small model
to morph into a more useful shape per task.

---

## 1. Where this landed

| Version of the idea | Reading | Why |
|---|---|---|
| Edit yourself based on what you just read | Weak so far | Attention already gets several times more from the same tokens, and a fitted edit adds about nothing on top |
| Edit yourself after seeing the answer | Not usable | Reaches almost 0 bits, but it is memorization and the edit costs more than it saves |
| Morph into a task with a stored edit | **Real, keep this** | 3% of the params buys 0.74-1.11 bits on narrow domains, though part of that is not specialization |
| Remove an obstacle (looping) | Not worth it so far | A hotter sampler matches the edit's loop reduction at no cost, though sample quality was not scored (Section 9) |
| Chosen by RL | Unneeded so far | A choice among K stored edits is a routing decision, and routing trains by gradient |
| Text-level GAN | Drop | Worse than MLE plus temperature in the literature, and RL is already off here |
| Teacher and student | Keep the shape, not the GAN | It is an amortized prior, trained by direct regression or KL |

The surviving kernel is a **small bank of stored mode adapters plus a router**. It
is less exotic than the original idea, and it is old (SMEAR, Lory, PEER). What makes
it worth another look is the measured size of the effect, and that it has not been
run as a matched-parameter test here.

---

## 2. What the post does and does not show

Takehi, Teoh, Zhang and Khattab (MIT CSAIL, Nov 2 2026) point a frontier LLM at
smaller models and let it program their weights. Results are large: Qwen3.5-0.8B
MATH-500 goes from 7.9 to 48.1 with a 131,072-weight edit (0.016%), while token-limit
hits fall from 93 to 33.1. Three details decide how it transfers to Praxis:

- **The editor is far larger than the target in every success.** Nothing in the post
  is a model editing itself.
- **Its own negative result is the important one.** "Editing weights could remove an
  obstacle that hid an ability, but it could not add an ability the model lacked."
  Gemma 3 270M only moved from 6.8 to at most 7.7. The Qwen jump is a model that
  could already do the math and was blocked by loops. A Praxis model that cannot do
  the math has nothing to unblock.
- **The gain appears only with enough passes.** Experiment 8 scores 12.3 at 10
  forward passes, 14.1 at 25, and 48.1 at 50.

"A smaller weight space is easier to reason about" splits into two claims:

- **Evaluating an edit is cheaper.** True and useful. SEAL pays 30-45 s per edit, while
  a forward pass on a 5M model takes milliseconds. This is the actual scale advantage.
- **Reasoning about the weights is easier.** True for an outside reasoner. For the
  model itself it runs the wrong way, since a model has no spare capacity to hold a
  description of the object it is made of. This is an argument, not a measurement.

---

## 3. The bound that frames everything

A deterministic edit computed from the prefix and the weights contains no information
about the future beyond what the prefix and weights already hold. So a self-edit
**cannot add information. It can only change how much computation goes into
extracting it.**

Two consequences:

1. The correct baseline for chain-of-thought editing is the same compute without the
   edit: read the prefix twice, use recurrent depth, or use a larger model. An edit
   that loses to those is an expensive way to get extra forward passes.
2. New bits have to come from somewhere: the weights, the context, or an external
   store. A bank of stored adapters puts them in the weights at training time and
   pays for them in parameters. That is why it can work where the other forms cannot.

This says nothing against a learned programmer finding computation that SGD cannot,
which is the part still open.

---

## 4. Measurements

Setup: edits are forward hooks on the FFN linears of all 4 layers, fit by gradient
on a frozen model. Scores are bits per token on held-out text, TokenMonster tokens,
not bytes. Corpora are minipile (prose), TinyStories (story, documents concatenated)
and site-packages Python (code). Cold baselines: prose 5.725, story 5.459, code 4.354.

**Context alone, on the same suffix tokens.** This is the bar any edit from a prefix
has to clear.

| Prefix tokens | prose | story | code |
|---|---|---|---|
| 16 | -0.117 | -0.088 | -0.127 |
| 64 | -0.252 | -0.154 | -0.215 |
| 256 | -0.442 ± 0.061 | -0.155 ± 0.019 | -0.287 ± 0.030 |

**One edit per document.** Oracle is fit on the suffix itself (n=48, best anywhere).
Causal is fit on the 256-token prefix, with lr and steps chosen on half the documents
and reported on the other half (n=24). Shuffled applies another document's edit.

| Edit (params) | Domain | Oracle | Causal, no context | Shuffled | Causal, on top of context |
|---|---|---|---|---|---|
| gain (1,024) | prose | -2.406 | -0.048 ± 0.033 | +0.098 ± 0.013 | -0.019 ± 0.015 |
| | story | -2.418 | -0.037 ± 0.008 | -0.015 ± 0.004 | -0.007 ± 0.006 |
| | code | -2.453 | -0.033 ± 0.012 | +0.038 ± 0.009 | -0.014 ± 0.012 |
| rank 1 down (5,120) | prose | -4.496 | -0.011 ± 0.008 | +0.011 ± 0.004 | -0.005 ± 0.003 |
| | story | -4.343 | +0.003 ± 0.004 | +0.008 ± 0.003 | +0.010 ± 0.003 |
| | code | -3.845 | +0.004 ± 0.007 | +0.014 ± 0.007 | -0.003 ± 0.002 |
| rank 4 down (20,480) | prose | -5.708 | -0.007 ± 0.011 | +0.021 ± 0.006 | -0.005 ± 0.002 |
| | story | -5.452 | -0.005 ± 0.001 | -0.002 ± 0.001 | -0.002 ± 0.001 |
| | code | -4.343 | -0.004 ± 0.004 | +0.003 ± 0.003 | -0.003 ± 0.001 |
| rank 16 FFN (163,840) | prose | -5.724 | -0.053 ± 0.021 | +0.030 ± 0.005 | -0.018 ± 0.017 |
| | story | -5.459 | -0.027 ± 0.011 | -0.002 ± 0.005 | -0.008 ± 0.004 |
| | code | -4.352 | -0.025 ± 0.012 | +0.032 ± 0.012 | -0.004 ± 0.013 |

All 5.3M weights, per document (n=12, lr and steps selected on the suffix, so
optimistic): causal prose -0.128, story -0.006, code -0.065.

**One edit fit across a domain.** 1,200 windows of 384 tokens, 400 steps, scored on
held-out text from every domain.

| Edit | Fit on | prose | story | code |
|---|---|---|---|---|
| rank 1 down (5,120) | prose | -0.022 | +0.008 | -0.000 |
| | story | +0.295 | -0.332 | +0.397 |
| | code | +0.151 | +0.217 | -0.131 |
| | all three | +0.015 | -0.221 | -0.093 |
| rank 4 down (20,480) | prose | -0.027 | +0.037 | +0.036 |
| | story | +0.561 | -0.630 | +0.689 |
| | code | +0.323 | +0.580 | -0.246 |
| | all three | +0.033 | -0.419 | -0.196 |
| rank 16 FFN (163,840) | prose | +0.016 | +0.136 | +0.214 |
| | story | +1.410 | -1.111 | +1.851 |
| | code | +0.857 | +1.144 | -0.739 |
| | all three | +0.087 | -0.804 | -0.491 |

**Whole model fine-tuned on one domain** (Adam 1e-4, delta on that domain).

| Steps | prose | story | code |
|---|---|---|---|
| 100 | -0.064 | -0.957 | -0.536 |
| 300 | -0.075 | -1.262 | -0.808 |
| 1,000 | +0.005 | -1.519 | -1.061 |
| 3,000 | +0.767 | -1.304 | -0.667 |

It overfits past roughly step 1,000, since each domain only has 1,200 windows. Off-domain
damage is large at every step (at step 1,000, the story fit costs prose +1.88 and code +1.82).

---

## 5. Reading

**Per-document edits from a prefix are close to redundant with attention.** The best
prefix-fit edit recovers 11-24% of what the same prefix gives as context, and adds at
most 0.02 bits on top of it. That fits Transmuting Prompts, which treats a prompt as an
implicit weight update already, and SHINE, where a hypernetwork at best matches
in-context prompting and in-context remains the baseline to beat.

**The causal arm is a baseline, not a ceiling.** It is plain SGD from a model never
trained for it. A meta-trained programmer could do better. Only the oracle arm is an
upper bound, and it is not achievable.

**The oracle is memorization, and the bill proves it.** Dividing the bits saved over
255 tokens by the size of the edit gives 0.60 bits per parameter for gain, about 0.22
for rank 1, 0.07 for rank 4, and 0.009 for rank 16. To pay for itself, an edit would
have to be sent at below those rates. An edit is a message the model sends itself, and
the target-aware version never covers the postage.

**Domain morphing is real, but needs its control.** A single rank 16 edit, 3% of the
params, captures about 70% of what fine-tuning every weight gets at its best measured
step (story 1.11 of 1.52, code 0.74 of 1.06). But the same edit fit on all three
domains already gets story -0.80 and code -0.49, at a cost of only +0.09 on prose. Those
gains need no switching, so they say this checkpoint underserves those domains. The
training mix for this run was not confirmed, and 5.46 bits on TinyStories looks
under-trained. **True specialization, fit-domain minus fit-all, is about 0.31 bits on
story and 0.25 on code, and about zero on heterogeneous prose.**

**The router has to be right.** A wrong rank 16 adapter costs more than a right one
saves. Between two domains, the break-even routing accuracy is 54-63%. These three
domains are easy to tell apart. Mixed prose is not, and it is what these models mostly
see.

**The selection cost is trivial.** Selecting from K adapters costs log2(K) bits per
document, against about 283 bits saved on a 255-token story. The real price is the bank
in parameters, 3% per adapter at this size.

---

## 6. RL, chain of thought, and the GAN

**RL.** SEAL, the closest published precedent, is not what its reputation suggests. Its
self-edit is generated finetuning data rather than weights, and its training is 2 rounds
of ReST-EM, supervised on the best samples, rather than a policy gradient. RL is off in
this repo since 08-03 (`next/rl.md`). `harmonic_weight_rl` (a 3-D action trained with
REINFORCE) and the `cot` policy have never been enabled. If a programmer is worth building, it should start differentiable, as a hypernetwork,
and only move to discrete tokens if that works.

**Edit tokens.** A token carries at most log2(4104) = 12 bits. A meaningful edit from
tokens therefore needs an interpreter holding a codebook, which turns "emit tokens that
reshape the weights" into "select entries from a stored bank." The token version and
the bank version converge.

**GAN.** A discriminator on text is not worth building. Language GANs Falling Short
found them losing to MLE plus temperature, and GAD's working teacher is an external and
far larger model, with off-policy reward hacking starting after about 300 steps.

**The teacher is the useful part.** A target-aware teacher `q(z|x,y)` and a prefix-only
student `p(z|x)` make an amortized posterior and prior, trainable with direct regression
or KL. No discriminator is needed unless the teacher is implicit. The catch comes from
the table above: a per-document teacher memorizes (oracle -5.7, causal -0.05), so its
edit has almost nothing the student can predict. The teacher has to be fit across
**other documents of the same kind**, which leads back to the adapter bank. LTM saw the
same failure from the other side, with severe posterior collapse in its amortized
baseline.

---

## 7. What this repo has already seen

- **abstractinator-d (`mag_standard`).** With gradient reaching the update, the model
  drove `memory_write` to 1e-6..1e-7: "energy mode by construction, standard mode by
  choice." Only step size was learned, though. The write direction was a reconstruction
  gradient, so a programmer choosing direction is untested.
- **abstractinator-w.** The no-memory ablation beat -v by -0.137 bpb.
- **prismatic8.** Retired the crystal bank because LoRA deviations shed dimensions, which
  led to the rule that a classifier should be a fixed function. A bank in the trunk is a
  different placement, but this is the failure to look for.
- **PEER and SMEAR.** Bank and routing machinery exists. `next/exact_solve.md` already
  warns that parameter-merging rank-1 experts collapses to one nonlinear feature. That
  matters here only if adapters are ever merged rather than selected.

---

## 8. Tests, cheapest first

**1. Learned programmer on the frozen base.** This is the original idea's fair trial.
A small hypernetwork reads the prefix and outputs a gain or rank 1 edit. Train it over
many documents, then score held-out suffixes with context on. Compare with a second pass
with no edit (Section 3). **Kill it if it does not reach -0.05 bits on top of context on
held-out documents.** That is more than twice the best SGD arm and about 1% of cold bits.
If it passes, the base must be co-trained, since freezing limits it.

**2. Unlabeled bank versus one adapter of the same size.** Frozen base, K=4 rank 4
adapters on both FFN linears (163,840 total), hard-assignment EM with no domain labels,
router on the first 64 tokens, log2(K) charged, on a mixed held-out set. Compare with one
rank 16 adapter of the same total size. **Kill it if the bank does not beat the single
adapter.** Then read prose alone, since that decides whether this only works on easy,
labeled-looking domains or finds modes inside ordinary text.

**3. Matched parameters from scratch.** Only if 2 passes. Base + K adapters + router
against a plain larger model at the same total parameter count, both trained from zero.
This is the real falsifier, since 1 and 2 both flatter adapters by freezing the base and
ignore the fit-all effect. **If the plain model wins, the kernel is closed.**

**4. Claude as the external editor (separate thread).** Praxis sizes are inside the
post's target range (its Experiment 6 is 1.48M weights, Experiment 3 is 7.2M), and CLI
usage is flat-rate. The negative result would apply here too, so the likely payoff is
reading what a tiny model cannot do, as much as fixing it. Experiment 7 needed attention
patterns, not weights alone, to read a planted behavior.

---

## 9. Removable obstacles: the repetition test

The post's mechanism that most plausibly transfers is removing an obstacle rather than
adding an ability. The test replicates Experiment 8 on `gpt2-1`: find a repetition
direction by difference of means over 8-gram repeats, project it out of the FFN output
weights, and compare with the trivial baseline of a hotter sampler.

Setup: the direction is fit on 144 generated rows and scored on 144 others, mixed
across domains, one seed. A loop row means at least 75% of its last 64 tokens repeat an
earlier 8-gram. The standard error on loop rows is about 0.03.

**Baseline with no edit.** Real text has a repeat share of 0.037 (prose), 0.006 (story)
and 0.054 (code).

| Sampler temperature | 0 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 1.0 | 1.2 |
|---|---|---|---|---|---|---|---|---|
| Repeat share | 0.907 | 0.818 | 0.656 | 0.348 | 0.217 | 0.095 | 0.039 | 0.010 |
| Loop rows | 0.993 | 0.951 | 0.813 | 0.465 | 0.278 | 0.153 | 0.056 | 0.014 |

**Edits, sampling at T=0.8.** Bits are the change on real text, scored at T=1. A hotter
sampler leaves the weights alone, so its cost is 0 by construction.

| Arm | Repeat share | Loop rows | Bits: prose / story / code |
|---|---|---|---|
| No edit | 0.217 | 0.278 | 0 |
| All 4 layers, alpha 1 | 0.659 | 0.806 | +0.33 / +0.19 / +0.45 |
| Layer 0, alpha 1 | 0.260 | 0.361 | +0.02 / +0.04 / +0.01 |
| Layer 1, alpha 1 | 0.138 | 0.160 | +0.03 / +0.01 / +0.11 |
| Layer 1, alpha 2 | 0.083 | 0.118 | +0.15 / +0.03 / +0.42 |
| Layer 1, alpha 4 | 0.027 | 0.035 | +0.60 / +0.11 / +1.49 |
| Layer 2, alpha 1 | 0.614 | 0.813 | +0.25 / +0.13 / +0.21 |
| Layer 3, alpha 1 | 0.239 | 0.347 | +0.06 / +0.01 / +0.22 |
| Random direction, all layers, alpha 1 | 0.243 | 0.340 | +0.03 / +0.03 / +0.05 |
| No edit, T=0.9 | 0.095 | 0.153 | 0 |
| No edit, T=1.0 | 0.039 | 0.056 | 0 |

Reading:

- **A mean direction is not a handle.** At T=0.8, the mean difference between repeated
  and fresh tokens has a norm of 2.9-6.1 times the average per-dimension standard
  deviation at every layer. No classifier was tested, so this does not show how readable
  repetition is. Even so, removing that direction from layer 2 almost triples looping,
  and removing it from all four layers does the same.
- **Layer 1 is a real effect.** Looping falls with dose and random directions do not do
  it. It holds more weakly at T=0.5, where alpha 4 only moves loop rows from 0.813 to
  0.625. Layer 1 was picked after seeing these scores, so it needs a fresh seed before
  anyone relies on it.
- **A hotter sampler matches it for free on these metrics.** T=0.9 is about the same as
  layer 1 at alpha 2, and T between 1.0 and 1.2 about the same as alpha 4. The edit also
  damages real-text prediction.
- **At its own distribution the model is not stuck.** At T=1 its repeat share of 0.039
  falls inside the real-text range. The loops come from sharpening the distribution,
  which is different from Experiment 8's picture of an ability blocked inside the weights.

No hidden ability showed up, but this test could not have proven one. **Neither sample
quality nor any task score was measured**, and the post's gain was measured with a
benchmark. An edit could still produce better text than T=1 at the same repeat rate. Six
T=0.5 samples read as incoherent before they start looping, for example "the renewable
Financial Policy System Act (COVID) is a Budget Compensation Company (AMP)". That hints
at nothing hidden, but it is only an anecdote. Revisit this only with a task score that
this model can move.

---

## 10. Caveats on all of the above

- One checkpoint, three corpora, tokens rather than bytes. Byte-latent models might read
  differently, and `gpt2-1` may be unusually under-trained on stories.
- Low-rank edits only touch FFN linears. Only the full-weights arm reaches attention.
- Each domain fit used 1,200 windows, which limits the fine-tuning arm in particular.
- Everything was frozen except the edit. Co-training is untested.
- Probe scripts lived in a session scratchpad and are not in the tree.

---

## Sources

Read in full:

- Takehi, Teoh, Zhang, Khattab, *Weight-Space Programming*, https://rikiyat.github.io/blog/weight-space-programming/
- Zweiger et al., *Self-Adapting Language Models (SEAL)*, NeurIPS 2025, arXiv:2506.10943
- *SHINE*, arXiv:2602.06358 (two passes, Qwen3-8B, Meta LoRA rank 128, generated rank 8, declines on multi-turn)
- Mazzawi et al., *Transmuting Prompts*, arXiv:2510.08734
- Irie et al., *A Modern Self-Referential Weight Matrix*, ICML 2022, arXiv:2202.05780
- Kong et al., *Latent Thought Models*, arXiv:2502.01567. Be skeptical: the headline perplexity moved from 5.58 (v1) to 3.05 (v2) at the same size, no independent replication was found, and the KL handling is unverified.
- *GAD*, arXiv:2511.10643

Used for specific claims:

- Caccia et al., *Language GANs Falling Short*, ICLR 2020, arXiv:1811.02549
- Akyürek et al., *Test-Time Training*, ICML 2025, arXiv:2411.07279
- Lazaridis et al., *When Can One Neuron Fix Repetition Loops*, arXiv:2606.13705 (Gemma 4 loops 46/384 to 12/384 held-out, no knowledge added)
- *Repetition In Repetition Out*, arXiv:2310.10226
- Pari, *Self-edit part 1*, https://jyopari.github.io/posts/self-edit-part1 (edit tokens plus interpreter, no results)

Related, IDs confirmed, not read in full:

- CAVIA 1810.03642, LEO 1807.05960, Intrinsic Dimension 1804.08838, G.pt 2209.12892, Function Vectors 2310.15213, Memory Layers at Scale 2412.09764, Lory 2405.03133
- Text-to-LoRA 2506.06105, Zhyper 2510.19733, Code2LoRA 2606.06492, BoLT 2503.18866, Metz 2009.11243 (IDs not independently confirmed)
