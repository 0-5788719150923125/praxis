# RL: status, deficiencies, and where it goes next

Written 2026-08-03, from an audit of run `20f4ae891` (`--abstractinator-g --reset`)
at step ~10,923. That run had `rl_type: [engagement, joke, preference]`, inherited
from `experiments/abstractinator-d.yml`.

**Current state: RL is OFF.** `abstractinator-d.yml` now sets `rl_type: []`. The
policies remain in `RL_POLICIES_REGISTRY` with their mechanical bugs fixed, so
re-enabling is one line. They are off because none of them earns its keep on a
`hidden_size: 90` byte model, not because the code is gone.

---

## 1. What went wrong

The trigger was an observation in the web app's rolling contexts: the model kept
falling back on a small set of words and themes - "Dr. Watson", "Sherlock
Holmes", "priest". Six mechanisms turned out to be involved. They are ranked by
how much damage they did, not by how interesting they are.

### 1.1 `JOKE` was missing from the task-weight table (fixed)

`praxis/tasks/__init__.py` - `BIAS_PRETRAIN_TARGETS` listed six tasks. `JOKE`,
`PREF_CHOSEN` and `PREF_REJECTED` were added to the enum later and never added
here. `_targets_to_tensor` seeds from `torch.ones(len(TaskType))`
(`praxis/tasks/weighter.py:19`) and only overwrites listed keys, so all three
silently inherited **1.0** while `conversation` sat at **0.3**.

The dynamic weighter does not absorb this. `DifficultyTaskLossWeighter._effective()`
returns `targets * clamp(ratio**gamma, 0.1, 4.0)` (`weighter.py:210-218`), so the
static table is a multiplicative **base** the curriculum moves around, not a prior
it can overrule.

Compounding it: `JokePolicy.dataset_collections = ("joke",)` force-includes
`rated-jokes` at `PRINT_WEIGHT = 5.0` (`praxis/data/config.py:84,155-157`), a
weight whose own comment reads *"High early ... (deliberate overfit)"*. So naming
`joke` in `rl_type` silently bought a 5x-sampled, 3.3x-weighted joke corpus.

`rated-jokes` is the Jester corpus. Priest/rabbi/bar setups and the canonical
"Sherlock Holmes and Dr. Watson went camping" joke are its highest-frequency
material. **This is the most likely single source of the observed vocabulary.**

> Not verified: nobody opened the dataset (the Bash tool was unavailable during
> the audit). The weighting arithmetic is verified from code; the corpus-contents
> step is inference. One grep over a few thousand drawn documents settles it, and
> it is worth doing before treating this as closed.

**Fixed.** All three tasks are now explicit, with a comment stating the invariant
that every `TaskType` must appear. Turning RL off also drops the forced `joke`
and `print` collections entirely.

### 1.2 The engagement/joke REINFORCE term is not a policy gradient (bounded, not repaired)

`praxis/policies/engagement.py`. The reward is computed from
`pred_ids = argmax(logits)`, but the log-prob that gets weighted is the
**ground-truth label's**. REINFORCE requires `log pi(a)` for the action `a` that
earned the reward; these are different objects.

Substituting `logprob = -ce`, the term reduces to:

```
L = +rl_weight * advantage * CE
```

So it is a per-row reweighting of the ordinary cross-entropy by how well the row
was already predicted. Positive advantage amplifies the CE; negative advantage
**negates** it, which is gradient ascent on the correct token and is unbounded
below as `p(label) -> 0`. Live `engagement_advantage` sat at **-0.6234** with a
negative slope, i.e. ordinary conversation text was being actively unlearned.

Structurally this is a rich-get-richer ratchet: reinforce what the model already
emits, suppress what it does not. That is a mode-collapse operator.

**Bounded, not repaired.** Added `LOGPROB_FLOOR` and `ADVANTAGE_CLIP` so the term
cannot run away, and the module docstring now states the defect at the top. A
real fix requires sampling the action, which the forward-path contract cannot do.
See §4.1.

### 1.3 The recall reward is maximised by saying less (documented, not repaired)

`recall = |set(pred) & set(target)| / |set(pred)|` over **distinct ids**
(`engagement.py:_reward`). That is precision, not recall, and its optimum is to
emit **one** distinct id that appears in the target.

At byte granularity it is worse than gameable, it is uninformative: over a
~260-symbol alphabet two spans of English nearly always share most of their
distinct bytes, so the value is near-constant. The live bimodality
(`engagement_recall` min 0 / max 1) tracks how many bytes were in the masked span,
not whether the answer was right.

**Documented in the module docstring.** Not repaired - the metric would have to be
replaced, and §4.1 describes what should replace it.

### 1.4 `rl_weight: 0.1` was nominal, not effective (fixed)

The main CE reduces as `sum(w*ce)/sum(w)` over **every** supervised position
(`praxis/losses/reduction.py:53-57`). Each policy reduced over **its own** tokens
(`engagement.py`, `denom = mask.float().sum()`). Losses are then plain-summed
(`praxis/strategies/naive.py:11-12`, `NaiveSummation`). The per-token ratio was:

```
R = rl_weight * |advantage| * W / (w_task * N_task)
```

`N_task` in the denominator means **the rarer a policy's data, the harder it
pushed** - a policy tagging 2% of the batch got ~50x the leverage its weight
advertised. Backwards.

**Fixed** in both `engagement.py` (shared denominator over all supervised
positions) and `preference.py` (scaled by the preference share of the batch).

### 1.5 The reward baseline cold-started at zero on every restart (fixed)

`engagement.py` had `self.reward_baseline = 0.0` as a plain Python float, so it
was absent from `state_dict` and re-zeroed on **every process restart**, not just
`--reset`. Reward reaches ~1.9 within a few steps (recall saturates,
`HomeostaticEnergy` climbs fast) while the baseline is still ~0, so each restart
injected several hundred steps of one-sided positive advantage - an effective
learning-rate multiplier on whatever that policy tagged. `joke_advantage max
0.9429` and `engagement_advantage max 0.8395` are the tails of that decay.

This is why the damage is **historical**. Turning RL off does not undo an
over-imprinted corpus at `hidden_size: 90`, where there is no spare capacity for
the imprint to coexist with anything else. **The run needs restarting**, which is
the plan.

**Fixed:** now a persistent buffer.

### 1.6 The preference contrast is not paired (documented; needs a formatter change)

`abstractinator-d.yml` used to claim *"The dataset entry now emits BOTH sides of
each pair"*. It never did. `format_preference_pair`
(`praxis/data/formatters/conversation.py:152-155`) emits **one side per call,
picked 50/50 at random**:

```python
take_rejected = _random.random() < 0.5
side_key = rejected_key if take_rejected else chosen_key
```

A pair's two halves are therefore never co-resident by construction. The margin
contrasts one random hh-rlhf conversation's chosen text against a **different**
random conversation's rejected text, so what it mostly measures is the
difficulty, length and domain gap between two unrelated documents.

This is not the documented cause (block packing losing row alignment). Packing is
real, but it is downstream of a formatter that never emitted a pair in the first
place, so no amount of pair-id threading downstream would have helped.

The metrics agree it was not working: `preference_rejected_logp` **rose** over the
run (+0.170/1k - rejected text becoming *more* likely) while `preference_margin`
shrank (-0.223/1k).

**Not repaired** - the fix belongs in the formatter. Specified in §4.2.

### 1.7 Nothing in the RL path bounded anything (partly addressed)

No KL penalty, no reference model, no trust region, no ratio clipping, no entropy
bonus anywhere on the active path. `GRPO` has `kl_coeff` and `clip_ratio`, but the
single call site passes `ref_logits=None` (`praxis/modeling.py`, `# TODO: Add
reference model support`) and `grpo.py` gates the entire KL block on that being
non-None. Dead code.

**Addressed for the future** by `harmonic_kl` (§3), which is the trust region this
path never had.

---

## 2. Policy inventory

| name | class | status | notes |
|---|---|---|---|
| `engagement` | `EngagementPolicy` | **off**, bounded | not a policy gradient (§1.2); reward gameable (§1.3) |
| `joke` | `JokePolicy` | **off**, bounded | subclass of engagement; inherits every defect |
| `preference` | `PreferencePolicy` | **off**, guarded | bounded by `logsigmoid`, so never destructive; unpaired (§1.6) |
| `grpo` | `GRPO` | never enabled | takes `rewards` from the dataset, not from rollouts; KL is dead code |
| `reinforce` | `REINFORCE` | never enabled | needs a `rewards` tensor from an RL dataset |
| `cot` | `ChainOfThought` | never enabled | supervised weighted loss, not RL |
| `harmonic_weight*` | `HarmonicWeightPolicy` | never enabled | weight-editing controller, callback-driven; has the only entropy bonus in the tree |

Also touched by this work:

- `praxis/tasks/__init__.py` - all `TaskType`s now present in the weight table.
- `praxis/losses/regularizer_base.py` - regularizers now take `**ctx`.
- `praxis/losses/harmonic_kl.py` - new, opt-in (§3).
- `praxis/modeling.py` - passes `classifier` to regularizers as context.

**Known observability gap:** `preference_margin`, `preference_chosen_logp`,
`preference_rejected_logp`, `preference_chosen_tokens`, `preference_rejected_tokens`
and `joke_activation_rate` have no entries in
`praxis/metrics/training_metrics.py`, so they stream but never chart. That is part
of why a term drifting the wrong way went unnoticed for a whole run. Worth fixing
before preference is re-enabled.

---

## 3. Current direction: `harmonic_kl`

`praxis/losses/harmonic_kl.py`, registered as `harmonic_kl` in
`REGULARIZER_REGISTRY`. Not in `DEFAULT_REGULARIZERS`, so it is opt-in by name.
**Enabled on `abstractinator-g`**, which now carries:

```yaml
regularizers:
  - contrastive_isotropy
  - harmonic_kl
```

`contrastive_isotropy` is restated because the list overrides the default rather
than extending it (`build_regularizers`).

### The idea

The harmonic line in `research/main.tex` claims the readout basis is
*constitutive*: a fixed eigenbasis whose coefficients rotate while its form does
not. Operationally that says the map from representation to output distribution
should mostly never change - every output token should look, mostly, the same.

This writes that sentence as a loss, which makes the claim falsifiable instead of
rhetorical:

- If the basis really is constitutive, `harmonic_drift` sits near zero without
  being paid for, and switching the penalty on costs almost nothing.
- If it is expensive, the claim is doing less work than the paper says.

Either outcome is a result.

### How it works

Keep a non-trainable EMA copy of the output classifier's weight and bias. Project
the **same** live hidden states through both the live and the EMA classifier.
Penalise the divergence. Two projections, no second trunk pass, no second model,
nothing new for the optimizer to own.

Deliberate choices worth not undoing:

- **Direction is `KL(ema || live)`** - teacher-weighted, mass-covering. The live
  model is charged for mass the teacher assigned and it dropped. Mode-seeking
  `KL(live || ema)` would reward collapse, which is the failure actually being
  fought here.
- **Both sides go through the same projection**, rather than reusing the model's
  own logits. A head may apply a norm or projection before its classifier, and
  that structure would otherwise register as drift it is not responsible for.
- **Buffers are non-persistent.** On resume the teacher re-seeds from the live
  classifier, which makes the penalty exactly zero at that instant and lets it
  re-converge. Cold-starting to the identity is safe; cold-starting to an
  arbitrary constant is what broke §1.5.
- **Positions are subsampled** to `MAX_POSITIONS` so cost is bounded at any
  vocabulary size (both projections materialise `[N, V]`).
- **EMA advances after scoring**, so a step is never measured against a teacher
  that has already absorbed it.

### What it does not do, and why that matters more than expected

It bounds drift of the **readout**, not of the trunk. Two different trunks feeding
the same classifier are indistinguishable to it.

The first measurement says this limitation is the whole story, and it is worth
knowing before reading the charts. Over 40 AdamW steps at lr 1e-3 on the -g stack
shape (`head_type: prismatic5`, random byte data):

| step | main loss | `harmonic_drift` | `harmonic_kl_loss` | `harmonic_live_entropy` |
|---|---|---|---|---|
| 10 | 8.74 | 1.5e-5 | 1e-6 | 1.86 |
| 20 | 6.67 | 9e-6 | ~0 | 1.91 |
| 40 | 6.15 | 2e-6 | ~0 | 1.68 |

The output distribution collapsed hard - entropy fell from 5.55 (which is
`ln(260)`, i.e. uniform over the byte alphabet) to 1.68 - **while readout drift
stayed at 1e-5 and fell**. The readout barely moved; the trunk did all the work.

Two consequences, stated plainly:

1. **At `KL_WEIGHT = 0.05` this is a measurement, not a constraint.** The penalty
   is ~1e-6 against a main loss of ~6. That is the right setting for a first run -
   it measures natural drift without perturbing anything - but do not expect a
   behavioural change from it.
2. **The readout-only version will not catch mode collapse.** Collapse can happen
   entirely in the trunk, and in this run it did. If the goal is a trust region
   that actually opposes collapse, it has to include the trunk, which means an EMA
   of the whole model and a second forward pass at roughly 2x step cost. That is a
   real decision, not a tuning tweak, and it is not made here.

What the run still buys: a direct, falsifiable number for the constitutive-basis
claim as it applies to the readout, on real data rather than random bytes, at no
meaningful cost. Read `harmonic_drift` and `harmonic_live_entropy` **together** -
this toy run is exactly the "low drift, falling entropy" pattern the entropy card
exists to expose.

### What to watch

Three cards, grouped under `harmonic_kl` on the Dynamics tab. They are declared
as `metric_descriptions` on the regularizer class, which is how the descriptions
walker discovers them (`praxis/metrics/descriptions.py`) - the same mechanism
`contrastive_isotropy` uses. Nothing needs adding to `TRAINING_METRIC_REGISTRY`;
that registry is the Research-tab/SQLite surface, and regularizers go to the
Dynamics manifest.

- **Readout Drift (nats)** (`harmonic_drift`) - raw mean KL, log-scaled. The
  direct readout of the claim. Steps where the teacher is seeded or re-seeded
  emit no value at all rather than a misleading zero.
- **Readout Entropy** (`harmonic_live_entropy`) - read it **beside** the drift.
  Falling entropy with low drift is the readout sharpening in place, which is
  collapse wearing stability's clothes. This is the reading that would mean the
  penalty is actively harmful.
- **Harmonic Drift Penalty** (`harmonic_kl_loss`) - the weighted term actually
  entering the objective, so its size against `loss` is the honest cost.

### The bug that a read-through could not have caught

The first version duck-typed `classifier.weight` and `classifier.bias`. On this
model family that attribute does not exist: `head_type: prismatic5` resolves to a
`ParallelHead` whose classifier is a `HaloClassifier`, and `prismatic4` gives a
`CrystalClassifier`. Both are distance-based readouts over per-vocabulary
`centers` (`praxis/heads/crystal.py`), with no weight matrix at all. The guard hit
`weight is None` and returned zero on **every single step** - the regularizer was
a silent no-op on precisely the config it was written for, and the tests passed
because they used `nn.Linear`.

The fix is to EMA the classifier's `named_parameters()` generically and evaluate
the teacher through `torch.func.functional_call(..., tie_weights=False)`. Verified
across all three readouts:

| `head_type` | classifier | EMA'd parameters |
|---|---|---|
| `prismatic5` (what -g runs) | `HaloClassifier` | `centers`, `gamma` |
| `prismatic4` | `CrystalClassifier` | `centers` |
| `forward` | `Linear` | `weight` |

This is a better target than the original design, not just a repair: those
`centers` are the vocabulary prototypes the paper's crystal claim is about, so
"how far has the readout moved from its own recent past" is now measured directly
on the geometry in question.

`tests/test_harmonic_kl.py::test_works_on_a_readout_without_a_weight_attribute`
pins it with a `centers`-only stand-in.

### Status

**Verified.** 14/14 in `tests/test_harmonic_kl.py`, 43/43 across
`test_harmonic_kl.py` + `test_preference.py` + `test_engagement.py`. Confirmed
end-to-end on the byte-latent abstractinator stack: seeds on step 0, produces
finite drift and gradients on subsequent steps, and backward completes.

Remaining caveats:

- **Mono-forward no-ops.** `praxis/trainers/mono_forward/actor.py:250` calls
  regularizers as `reg(h_out, input_ids)` with no classifier, so the penalty is
  inert under `mono_type`. `-g` does not set it. Safe, but silent.
- **A readout it cannot drive disables itself loudly.** If the classifier raises
  when called functionally, the regularizer prints a one-line reason and stays off
  for the run rather than killing training. If the three cards never appear, look
  for `[harmonic_kl]` in the log - that is the failure mode, and it is no longer
  silent.
- `torch.compile` is default-on and the forward mutates module state (lazy buffer
  registration), so it is wrapped in `torch._dynamo.disable`. If that misbehaves,
  `--no-compile` isolates it.

---

## 4. Future work

### 4.1 A verifiable reward, via the MTP speculative-decode loop

The lesson from RPT (§5) that survives every caveat: **a reward whose optimum is a
fixed lexical pattern will be found and exploited; a reward whose correct answer
rotates every example cannot be.** `engagement` and `joke` are the first kind.
Ground truth from the corpus is the second kind, and it is free.

The interesting observation is that this codebase **already has a prefix-match
verifier**. Byte-level MTP speculative decoding drafts multiple bytes and accepts
the longest prefix matching the target. That is RPT's reward, minus the chain of
thought, computed on the model's own sampled continuation.

Sketch:

- Grade by **accepted prefix length**, not a binary hit. RPT's binary form does
  not port: its reward is gated on the prediction length landing on a token
  boundary, and for a byte-level model every byte offset is a boundary, so the
  gate is vacuous and the reward saturates. Length-grading restores the variance.
- Group-relative advantage over G drafts from the same prefix, which needs no
  value network and no reference model.
- Drop degenerate groups (all-accept or all-reject) rather than applying a
  near-zero-variance update - this is DAPO's dynamic sampling, and it is exactly
  what the live `joke_activation_rate = 1, advantage = 0.007` case needed.
- Fits `Generator` / `DecodeBackend`, not a new subsystem.

Precondition: the model has to be able to hold a format long enough for a
continuation to mean something. Right now it cannot (§4.4).

### 4.2 Real preference pairing (formatter change)

To make `preference` a genuine objective rather than a difficulty contrast between
unrelated documents:

1. `format_preference_pair` emits **both** sides, with a shared pair id in
   metadata.
2. Thread the pair id per-token alongside the task tags, the same way
   `PREF_CHOSEN` / `PREF_REJECTED` already travel.
3. `PreferencePolicy` contrasts **within** a pair id and averages over pairs.

Until then the existing guards (`MIN_SIDE_TOKENS`, batch-share scaling) keep it
from doing damage, but it should not be expected to teach anything.

Also worth reconsidering: hh-rlhf is post-training data. A preference margin on a
model that cannot yet form sentences is teaching a distinction it has no way to
represent. It is defensible as scaffolding for building the RL path; it is not
defensible as a source of capability at this scale.

### 4.3 Rewarding coherent conversation

There is a lot of conversational data in the mix (`persona-chat`, `wildchat`,
`smoltalk`, `natural-instructions`), and "have a coherent conversation" is closer
to what the model should actually be optimising than any of the three policies
that were running.

The trap is that "coherent" is exactly the kind of heuristic target that gets
hacked - it is the Quiet-STaR failure RPT calls out by name. Candidate framings
that stay ground-truth-anchored, roughly in order of how honest they are:

- **Turn-boundary prediction.** Reward the model for placing the *next speaker's*
  boundary where the data actually places it. Verifiable from the corpus, rotates
  every example, and it is precisely the capability currently missing (§4.4). This
  is the one to try first.
- **Held-out continuation matching.** Score a sampled reply against the real
  reply by accepted prefix length (§4.1 machinery, conversation data).
- **Self-consistency across turns.** Penalise contradicting earlier committed
  content. Needs a judge, so it is the least honest of the three and should wait.

Note the failed precedent: `EngagementPolicy` was already an attempt at "reward
the model for anticipating the answer", and it collapsed into rewarding whatever
the model already emitted. Any coherence reward has to be checkable against text
the model did not produce.

### 4.4 The chat format is the actual blocker

The model cannot currently respond to prompts, because it does not produce the
chat format. That was `abstractinator-g`'s whole thesis (`chat_format: prose`,
moving turn boundaries inside the generation block so the halt signal becomes a
trained target), and on the evidence it did not land.

This outranks every RL question. There is no useful reward signal over
conversations the model cannot structurally produce, and §4.1 and §4.3 both
depend on it. `abstractinator-g.yml`'s own falsifier applies: *"if termination
does not improve, the special tokens were never what blocked it and the
bottleneck's spectral budget is the place to look."*

Candidates to separate, on the restarted run:

- Whether boundaries are ever emitted at all (the direct readout).
- Whether `repetition_penalty` (terminal default 1.15) is suppressing them - a
  prose boundary is made of common bytes, so the penalty pushes against
  re-emitting exactly the bytes that end a turn. The config anticipated this; the
  honest fix is excluding boundary bytes from the penalty context, not lowering
  the penalty.
- Whether the patch/spectral budget simply has no room for the boundary.

### 4.5 Chain of thought

`ChainOfThought` exists (`praxis/policies/cot.py`) and has never been enabled. It
is supervised weighted loss, not RL, which makes it a reasonable on-ramp: it needs
no rollouts and no verifier. It is worth revisiting **after** §4.4, since a model
that cannot hold a turn boundary cannot hold a reasoning block either.

The RPT-shaped version (reason first, then predict, reward on correctness) is a
much later step and requires a base model that can already produce a parseable
answer - see §5.

### 4.6 Full-model EMA (the trust region this one is not)

Follows directly from the measurement in section 3: readout drift is ~1e-5 while
output entropy collapses, so bounding the readout does not bound the output
distribution. A trust region that actually opposes collapse needs the trunk in the
EMA, which means a second forward pass through an EMA copy of the whole model at
roughly 2x step cost.

That is affordable at this model size and it is the honest version of "a harmonic
model should mostly never change". It is deliberately not built here because
doubling step cost is the user's call, and because the cheap version's measurement
is worth having first: if readout drift turns out to be large on real data, the
cheap version was enough.

### 4.7 hh-rlhf's rejected half is now dead weight

With `rl_type: []` there is no `PreferencePolicy`, but `format_preference_pair`
still emits the rejected side on ~50% of hh-rlhf draws, and rejected tokens are
excluded from the main CE twice over (the hard zero in `_build_loss_weights` plus
`pref_rejected: 0.0` in the task table). So those positions train nothing.

Bounded: hh-rlhf is 1 of 12 datasets and packing means the rest of each row still
trains, so this is a few percent of positions, not of steps. Do NOT fix it by
giving `pref_rejected` a positive weight - training on rejected text is the card
violation the preference work existed to undo. The correct fix is for the
formatter to emit only the chosen side when no preference policy is active, which
needs the formatter to see that config.

### 4.8 Routing metrics were measuring VEAR's exponent (fixed 2026-08-04)

Not RL, but it came out of the same investigation and it is the reason a
"routing collapse" got diagnosed that was never happening.

`layer_N_routing_entropy` was computed inside `_merge_expert_parameters`, which
VEAR overrides to sharpen by `p**4` **before** calling super
(`praxis/routers/vear.py`). So every routing diagnostic was measured on the
sharpened probabilities, i.e. on `VEAR_SHARPEN`, not on anything the router
learned. The distribution saturated to float-exact one-hot, at which point the
metric stopped responding to the weights at all: a 4-arm ablation produced
**bit-identical** routing entropy across models whose losses differed by 5%.

The `+ 1e-10` epsilon was the tell. Adding it pushes a weight of exactly 1.0
above 1, so `log` goes positive and the reported "entropy" goes slightly
negative. `rent_min = -0.00000` was the metric announcing its own saturation.

Fixed by separating what is measured on what:

- **Router diagnostics** (`routing_entropy`, `routing_entropy_seq`,
  `routing_concentration`, `routing_variance`, `routing_peak`,
  `routing_specialization`, `routing_input_dependence`) now use the router's own
  pre-transform, pre-dropout output. `SMEAR._merge_expert_parameters` takes a new
  `router_probs` argument; VEAR forwards the unsharpened probabilities.
- **Merge diagnostics** (`expert_i_routing_weight`, `routing_merge_entropy`) keep
  using the post-transform weights, because that is what the merge applied. The
  gap between `routing_entropy` and `routing_merge_entropy` IS the sharpening.
- Entropy uses `clamp_min` rather than `+ eps`, so it can no longer go negative.

Two new metrics, both charted in `COMPOSITE_METRIC_REGISTRY`:

- `routing_entropy_seq` - mean per-sequence entropy, versus the existing
  batch-mean entropy. How undecided a typical single decision is, as opposed to
  how balanced the aggregate load is.
- `routing_input_dependence` - normalized mutual information,
  `(H(mean p) - mean H(p)) / log(N)`. **The gauge that was missing.** Zero means
  every sequence got the same routing distribution, so the router is effectively
  a constant. Neither load balance nor specialization can catch that: a router
  that sends the WHOLE batch to one expert scores maximum specialization while
  having learned nothing.

First reading on the -g architecture (`num_experts: 4`, random-byte data, 30
steps): `routing_entropy` 0.716 of a possible `ln(4) = 1.386`, so load balance is
fine and there was never a collapse. But `routing_input_dependence` is **3e-5**,
i.e. the router is very nearly input-independent. That is plausible on random
bytes - `SMEAR._router_forward` routes on `inputs.mean(dim=1)`, and mean-pooled
random sequences all look alike - so the number to watch is what it does on real
text. If it stays at ~0 there, the router is a constant and the whole
mixture is decorative.

Tests: `tests/test_routing_metrics.py` (8), including one that pins the
regression directly (VEAR's sharpening must not change any router diagnostic) and
one that separates a constant router from a discriminating one at identical
specialization.

### 4.9 Measurement debt

- **Confirm the Jester hypothesis** (§1.1) by grepping drawn documents.
- **Measure the real `f_task`** - log `mask.float().sum()` from the policies
  alongside `flat_w.sum()` from `reduction.py` for a few hundred steps. The
  amplification arithmetic in §1.4 is worked from the formula, not measured.
- **Register the preference metrics** (§2).
- Note that the rolling contexts in the web app are **self-conditioned**
  (`praxis/generation/context_blocks.py:123-124` feeds output back as prompt over
  a 512-char window, seeded from one random character, with the "Focused" block at
  temperature 1/3). Any attractor amplifies itself there. It is a weaker piece of
  evidence than it looks, and worth cross-checking against a fresh prompt at
  temperature 1.0 before diagnosing from it again.

---

## 5. RPT (arXiv:2506.08007), read carefully

Consulted because it looked like a possible direction. It is not one, but its
reward-design argument is the most useful thing in this document.

**What it actually is.** Despite the name it is **not** pre-training. It is
mid-training on `Deepseek-R1-Distill-Qwen-14B`, chosen explicitly *"due to its
basic reasoning capabilities"*, over 4,428 OmniMATH problems for 1,000 GRPO steps.
The paper lists *"RPT training from a standard base language model"* as future
work.

**Why it cannot start from scratch.** The reward is computable only if the model
can emit a parseable answer. From random init every rollout scores 0, every group
is degenerate, the group-relative advantage is identically zero, and nothing
learns. At step 0 with their own prompt template, ~91.5% of groups already had all
8 rollouts wrong (Appendix D, Table 8: 8.5% Pass@8) - they bootstrapped from the
surviving 8.5% only because a 14B reasoning model made it informative.

**Why the reward does not port verbatim.** `r = 1` iff the prediction is an exact
byte prefix **and** its length lands on a ground-truth token boundary. For a
byte-level model tokens are bytes, so the boundary set is every integer, the gate
is vacuous, and the reward saturates. Any port needs prefix-**length** grading, or
the abstractinator's space-patch boundaries standing in for the boundary set.

**Evidence quality, honestly.** The headline table compares RPT in reasoning mode
(up to 8,192 CoT tokens per prediction) against baselines in argmax mode - three
to four orders of magnitude more inference compute. The only next-token-prediction
control in the paper collapsed the base model by 40 points and is not a serious
baseline. There is no compute-matched control.

**What rescues it.** NVIDIA's RLP (arXiv:2510.01265, Table 3) ran RPT on
`qwen3-1.7b-base` - a plain base model, not a distilled reasoner - at matched data
and compute: 34.03 -> 41.69 overall average. So the mechanism is real and is not a
distillation artifact. RLP's own denser-signal method reached 43.35. No failed
reproduction or retraction exists.

**The transferable lessons**, scale-independent:

1. **Verifiable, not heuristic.** The answer key comes from the data, so the
   reward has no fixed optimum and cannot be farmed.
2. **The target rotates every example.** This is the structural reason RPT needed
   neither a KL penalty nor an entropy bonus (both coefficients were zero) and
   still did not collapse. A fixed-optimum reward run with no KL and no entropy
   term has nothing at all opposing collapse - which describes the setup that was
   running here.
3. **Group-relative advantage over the sampled action.** The reward must score
   what the model actually produced.
4. **Drop degenerate groups.** RPT enables DAPO dynamic sampling at step 500 for
   exactly this.
5. **Reward shape barely matters.** Appendix A found first-token-only and two
   dense variants all performed comparably. So do not tune reward curves; the
   question is only whether the reward is ground-truth-anchored at all.

RPT names this codebase's failure mode directly, in its Related Work: it rejects
Quiet-STaR's helpfulness reward because *"the helpfulness-based reward tends to be
hacked by repeating the target token in the generated rationale."* `engagement`
and `joke` are that class of reward. RPT would not have used them.
