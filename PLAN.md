# PLAN.md - "Print": an engagement-prediction reward for calm-d

Handoff plan for a fresh session. Conceptual background lives in
`next/homeostatic_engagement.md`; this is the execution doc. Read both.

## 0. Premise and the one rule that must not bend

The model leads a chat with a **question** (instead of waiting for a user prompt,
as `Evaluate` does). It also forms a **prediction of what the user will answer**.
The reward is **how well the user's actual response matches that prediction** -
"mentioning the predicted answer tokens AT ALL" is the reward activation. This is
a *model-of-the-user prediction / calibration* signal: the model is rewarded for
asking questions whose answers it can anticipate.

THE RULE: the reward is **prediction/correctness**, never **raw engagement/
attention**. Optimizing attention is the recommender-system path to manipulation
(see the note). If any variant starts producing attention-grabbing or
anxiety-inducing questions to farm responses, kill it. Add an informativeness/
difficulty term before that happens (so it can't farm trivially-answerable
questions either).

## 1. The loop (two reward sources, one reward function)

Single reward function, two sources of "the user's response":

- **Training (dense): a simulated user.** The synthetic dataset (section 5)
  carries the ground-truth short answer for each (prefix + question). At train
  time the "user response" IS that answer, so the reward is dense - this is how
  we get the sample density RL needs. Overfit here first.
- **Live (sparse): the real user** via the `Print` UI. Same reward function,
  applied to the real typed response. Slow, online, layered on top later.

Reward -> a homeostatic long-EMA "energy" variable -> REINFORCE on the model's
question-and-predicted-answer generation.

## 2. Components and where they live

- **Synthetic dataset**: `praxis/data/config.py` (`DATASETS`, `DATASET_COLLECTIONS`)
  - new `type: synthetic-print` alongside the existing `synthetic-tool-calling`.
- **Reward + homeostatic energy**: new `praxis/policies/engagement_reward.py`
  (pure functions + the EMA state); unit-tested in isolation.
- **RL policy**: new `RL_POLICIES_REGISTRY` entry. The engagement reward trains
  the LM's *generation policy* (REINFORCE on generated tokens), which is DIFFERENT
  from `harmonic_weight_rl` (a callback that edits weights). It plugs into the
  existing forward-pass policy hook in `praxis/modeling.py` (the `self.policy(...,
  rewards=...)` REINFORCE branch around lines 578-615, whose `rl_loss` is added to
  the loss container). Reuse the REINFORCE + reward-EMA patterns from
  `praxis/policies/harmonic_weight_rl.py`.
- **Generation (model leads)**: `praxis/generation/generator.py` (`Generator`,
  `request_generation`, `_eval_mode`). Print needs a "no user prompt -> model emits
  a question" entry; for CALM the path is `custom_generate`.
- **UI**: `praxis/web/src/js/components.js` (the `Read`/`Evaluate` `data-tool`
  toggles -> add `Print`), `actions.js` (tool dispatch), the Gymnasium tab
  (`state.js`). EDIT `praxis/web/src/`, then run `python praxis/web/src/build.py`.
  NEVER edit `praxis/web/static/` directly.
- **Web route**: `praxis/web/routes/` - a Print endpoint (serve the model's
  question + stash its predicted answer; accept the user response; return/record
  the reward).
- **Metrics**: `praxis/metrics/training_metrics.py` registries (engagement energy,
  activation rate, recall) -> Research/Dynamics dashboards (registry-driven; never
  hardcode chart configs in JS).
- **Experiment**: `experiments/calm-d.yml`.

## 3. Phased implementation (smallest-first; each phase ships and is testable)

**P0 - Synthetic `print` dataset. [DONE]** ~25 question formats, prefix-modulated
(section 5), with ground-truth short answers. Register `synthetic-print` in
`DATASETS`. Acceptance: it loads, interleaves via the sampler, and a batch shows
wildly-varied prefixes with the same question/answer core. No RL yet.

Implemented:
- `praxis/data/formatters/print.py`: 25 broad-spectrum format generators
  (`PRINT_FORMATS`: arithmetic, geometry, physics, chemistry, astronomy, biology,
  geography, language/grammar, prose/rhyme, logic, units, color, social) +
  `PREFIX_POOL` (cross-domain color) + `format_print`. The assistant turn leads
  with `question\nanswer` (trained target + the `A_hat` span for the reward);
  prefix sits in a leading untrained user-context turn; metadata carries
  `question`/`ground_truth`/`category`.
- `SyntheticPrintDataset` in `praxis/data/datasets/synthetic.py`; `type:
  synthetic-print` branch in `praxis/data/utils.py:get_dataset`.
- `DATASETS["synthetic-print"]` + `print` collection (`PRINT_WEIGHT=5.0`, high
  early for deliberate overfit) in `praxis/data/config.py`.
- Folded out the old "simple math" RL bootstrap: deleted
  `praxis/data/datasets/simple_math.py`, removed `mix_simple_math` from the
  `intellect-rl` config and the `HuggingfaceDataset.get_document` branch (kills
  the `[RL] Using simple math` / `[RL DEBUG]` terminal spam). Arithmetic now lives
  as one of the 25 print "kinds", not a separate mixed-in module.

**P1 - Supervised bootstrap. [WIRED]** Blend the print set into calm-c and let the
normal LM loss teach (prefix + question -> short answer). `experiments/calm-c.yml`
now sets `train_datasets: [focused, print]` (the `print` collection,
`PRINT_WEIGHT=5.0`). Acceptance: after overfitting the 25 formats, the model
reproduces a format's question and its expected answer on held-out prefixes. This
establishes the behavior before any reward exists. Overfit run not yet executed.

**P2 - Reward function + homeostatic energy (pure, unit-tested). [DONE]**
`praxis/policies/engagement_reward.py`: `activation`, graded `recall`, and the
`HomeostaticEnergy` EMA (section 4 math; fixed model-agnostic constants
decay=0.999 / gain=0.5 / e_max=1.0). Unit-tested in `tests/test_engagement.py`:
activation fires on overlap, energy accumulates-then-satiates, decays without
activation, stays bounded.

**P3 - RL with the simulated user (dense). [DONE]** `EngagementPolicy` in
`praxis/policies/engagement.py`, registered as `RL_POLICIES_REGISTRY["engagement"]`
(forward-path; `needs_rl_datasets=False` so it computes its own reward from labels
and pulls no RL collection). Over the assistant region it scores the model's
predicted answer tokens (A_hat = argmax) against the labels (R) by recall, updates
the homeostatic energy (the REINFORCE baseline), and reweights the LM's own answer
log-probs by `recall - energy`. Wired into `modeling.py` (the `rl_type ==
"engagement"` forward branch) and surfaced via `model.get_metrics()`. calm-c =
`rl_type: [engagement, harmonic_weight_wave]`. Default data + an offline reward
trace: `tools/generate_print_samples.py` (25 formats x N), and
`praxis/data/formatters/print.py:generate_print_samples`. NOTE: this is the
teacher-forced dense proxy (A_hat = the model's own argmax over the answer span,
not sampled generation); true generation-time calibration is P4/P5. Live-training
acceptance (recall/activation/energy rise on the overfit set; ablate weight 0 ->
flat) not yet run.

**P4 - Print UI (model-leads chat). [DONE]** The `Print` Gymnasium tool is a
*conditional* button - hidden until the model has actually been queried for and
produced a question. Backend `praxis/web/routes/print.py`: `/api/print/ask` (model
leads: system+developer, no user prompt -> emits a `question\nanswer`; stashes the
predicted answer; idempotent while pending), `/api/print/pending`,
`/api/print/respond` (scores the user's answer vs the stashed prediction, recall
fires on mention), `/api/print/energy`. Frontend: `state.print`, a 15s poll
(`setupPrintHook`, the environment-level hook that produces the question), the
conditional button (`renderPrintButton`), `PRESENT_PRINT_QUESTION` (injects the
model's question as an assistant turn - the model leads), and `handlePrintResponse`
(captures the next user message as the answer, shows the reward). Built via
`praxis/web/src/build.py`. Route + channel tested in `tests/test_print_route.py`.

**P5 - Live online reward channel (the real engineering). [DONE]**
`praxis/policies/engagement_channel.py`: `LIVE_ENGAGEMENT`, a thread-safe
process-global channel. The web `respond` route calls `.submit(predicted, response)`
-> folds into a live `HomeostaticEnergy` and buffers the reward. The trainer-side
consumer is `praxis/callbacks/lightning/engagement_live.py`
(`EngagementLiveRewardCallback`): every `period` train steps it `.drain()`s the
buffer and folds each live activation into the EngagementPolicy's energy baseline
(`EngagementPolicy.ingest_live`) - the slow integrated online return. Auto-added in
`callbacks/builder.py` whenever `engagement` is in `rl_type`, ordered before
MetricsLogger. Emits `engagement_live_reward/count/energy`. Dashboard energy badge:
the Gymnasium toolbar shows a live `⚡ <energy>` badge fed by `/api/print/energy`
(polled in `setupPrintHook`, rendered by `renderPrintButton`), visible once any
real-user reward exists. Tested in `tests/test_engagement.py::TestLiveDrainCallback`.

**P6 - Metrics + eval. [PARTIAL]** Research-tab charts registered in
`praxis/metrics/training_metrics.py`: dense `engagement_energy`,
`engagement_activation_rate`, `engagement_recall`, `engagement_advantage` (orders
300-330) plus live `engagement_live_reward`, `engagement_live_count`,
`engagement_live_energy` (orders 340-360). Dense flow `EngagementPolicy._metrics ->
model._engagement_metrics -> model.get_metrics() -> MetricsLogger -> charts`; live
flow via the drain callback's `trainer.callback_metrics`. No JS edits. STILL TODO:
a `BrierLM`-style offline eval of question-posing (the only remaining nice-to-have).

## 4. Reward math

Per interaction:
- `A_hat` = model's predicted answer tokens (its generation after the question,
  or an explicit short prediction span). `R` = user/simulated response tokens.
- **Activation (the headline signal):** `a = 1` if `|set(A_hat) & set(R)| > 0`
  ("mentioned AT ALL"), else `0`. Optionally graded recall
  `r = |set(A_hat) & set(R)| / max(1, |set(A_hat)|)` for a smoother gradient.
- **Homeostatic energy (extremely long EMA, fast accumulation, satiating):**
  `E <- decay * E + gain * a * (1 - E/E_max)`
  - `decay` very slow (the "multi-hour" feel = a large step horizon; pick a fixed
    model-agnostic constant, e.g. effective horizon ~ thousands of steps).
  - `gain` large so a single activation jumps `E` fast (fast accumulation).
  - `(1 - E/E_max)` is the satiation / diminishing-returns term (concave toward
    the setpoint `E_max`); prevents reward-spam.
- **REINFORCE:** advantage = `reward - baseline`, baseline = `E` (or a separate
  slow value EMA). Update the generation policy
  `grad = -log_prob(question, A_hat) * advantage`. All constants fixed, not
  per-experiment tuned (project rule).

## 5. Dataset spec (~25 formats, prefix-driven "color")

Each format: `(id, question_template, expected_answer_tokens, prefix_pool)`.
- **Prefix modulation is the point.** Every example = `random_prefix + question`,
  where the prefix is drawn wildly (from other datasets / a varied pool) so the
  question is asked in radically different contexts but converges on the same
  short structured answer. The "true color" comes from blending these with the
  normal calm-d data mix, not from the 25 formats alone.
- **Answers are short and structured** (a few words). Reward fires on mention, so
  exact phrasing is not required - recall, not exact match.
- Generate procedurally as `type: synthetic-print` (mirror
  `synthetic-tool-calling`). Tokenized with calm-d's `byte_level` tokenizer.
- **Early-stage = deliberate overfit:** high sampling weight on the print set so
  the 25 formats are learned fast; blend the weight down later for generalization.

## 6. calm-d integration

- Add the print set to calm-d's `train_datasets` (high weight early). calm-d uses
  `byte_level` + `calm_byte_small`; keep that.
- **Two RL signals coexist.** calm-c already runs `rl_type: harmonic_weight_wave`
  (a callback editing the optimizer wave). The engagement reward is a *separate*
  REINFORCE on the generation policy via the forward-pass hook. DECISION MADE:
  `rl_type` now takes a **list** (`praxis/policies/normalize_rl_types`); each entry
  is an independent RL task. The model builds at most one forward-path policy
  (raises if two are requested) and a weight-editing callback per profile entry;
  the data pipeline unions the RL collections any dataset-RL entry needs. calm-c
  is `rl_type: [harmonic_weight_wave]`; the engagement policy slots in as a second
  entry once it exists. (Also fixed a latent bug: profile keys like
  `harmonic_weight_wave` weren't resolved to their policy in `_rl_uses_datasets`,
  so calm-c was silently loading RL datasets - now it doesn't.) Still: run
  engagement RL alone first (cleanest signal) before trusting the combination.
- Gate everything behind a flag (e.g. `engagement_reward: true`) so calm-d without
  it is unchanged and A/B is clean.

## 7. Risks / open questions / kill criteria

- **Proxy-gaming:** model asks trivially-answerable questions to farm recall. Fix:
  add an informativeness/difficulty term to the reward (answerable AND non-trivial).
- **Sparsity:** live human reward is low-volume - that is why training rides the
  simulated user; the live channel only fine-tunes / provides a slow online signal.
- **Two-RL interaction:** engagement REINFORCE vs harmonic-weight controller may
  fight. Start isolated.
- **Does the prediction reward even move a small model?** Validate on the overfit
  set (P3) before trusting anything live.
- **HARD KILL:** if it ever optimizes attention over usefulness (manipulative,
  anxiety-inducing, or content-free attention-bait questions), stop. The whole
  design exists to avoid that.

## 8. Existing code to reuse (navigation for the fresh session)

- REINFORCE + reward-EMA + horizon credit: `praxis/policies/harmonic_weight_rl.py`,
  `praxis/callbacks/lightning/harmonic_weight_rl.py`.
- Forward-pass RL hook (rewards -> rl_loss in the loss container):
  `praxis/modeling.py` (`self.policy(..., rewards=...)`, ~lines 578-615).
- Policy registry: `RL_POLICIES_REGISTRY` (`praxis/policies/`).
- Synthetic dataset precedent: `synthetic-tool-calling` in `praxis/data/config.py`.
- Generation: `praxis/generation/generator.py`; CALM's `custom_generate` in
  `praxis/encoders/calm/encoder.py`.
- UI tool toggles: `praxis/web/src/js/components.js` + `actions.js`; build via
  `praxis/web/src/build.py` (never touch `static/`).
- Metric registries: `praxis/metrics/training_metrics.py`.
- Conceptual rationale + the engagement-vs-correctness fork:
  `next/homeostatic_engagement.md`.
