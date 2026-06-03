# Homeostatic engagement: the user's attention as a self-driven reward

A speculative reward design. The model proposes a question; a human engages; that
engagement feeds a slowly-replenishing "energy" variable the model is optimized
to keep near a setpoint. Like hunger or attention, the reward has diminishing
returns - the first engagement is worth a lot, the tenth in quick succession is
worth little, and the energy decays over hours so the model is drawn back to
seeking the next genuine interaction. The UI already feels like a gymnasium; this
is the natural next button after `Read` (search the KB) and `Evaluate` (chat): a
**`Print`** stage where the model emits a single prediction - one question - and
the user engages.

This note records the idea, what is genuinely good about it, the trap that would
sink it, the practical wall, and the realistic staging.

## The good part: a homeostatic, diminishing-returns reward

A reward that decays with consumption and replenishes slowly is *homeostatic RL*
(Keramati & Gutkin 2014): the agent optimizes toward a setpoint rather than
maximizing an unbounded quantity. The satiation term is the load-bearing piece -
it makes the reward self-limiting and kills the obvious failure mode (spamming
questions to farm reward; each one is worth less when recent ones were plentiful).
The shape is the strongest part of the idea, and it is on-theme with the
EMA-integrated-return machinery already in `praxis/policies/harmonic_weight_rl.py`
(long EMA, slow decay, reward as an integrated return over a horizon). "1
engagement = a ~6-hour boost in EMA energy, with diminishing returns" is exactly
a homeostatic energy variable with a slow decay constant and a concave gain.

The accompanying direction - a model that learns to *ask good questions* - is
active learning / artificial curiosity (Schmidhuber's artificial curiosity;
Pathak et al. 2017). Posing questions calibrated to what a user can actually
answer is a real, valuable capability (tutoring, clarification, calibration).

## The crux: engagement is the wrong reward, correctness is the right one

Two different rewards were conflated in the original framing, and the choice is
everything:

- **"maximize human attention / engagement"** is the recommender-system
  objective. Optimizing it is the documented road to clickbait, manipulation,
  and addictive dark patterns: a model rewarded for engagement learns to be
  attention-grabbing, not truthful or useful - provocative, anxiety-inducing, or
  cliffhanger questions maximize clicks. This is not hypothetical; it is what
  engagement-optimized systems converge to.
- **"ask questions the user can answer correctly"** is a calibration /
  active-learning signal. It rewards posing questions at the right difficulty and
  being *useful*. This is the healthy version.

Decision: the reward is **answer correctness / calibration**, with engagement only
ever a weak, capped proxy (and the satiation term bounding it). If raw attention
is the reward, this builds the thing everyone regrets building. Even with the
correctness signal, watch for proxy-gaming (the model steering toward trivially
answerable questions to farm "correct").

## The practical wall: human reward is low-volume

RL needs many reward samples to shape a policy; a human supplies a trickle. RLHF
sidesteps this by collecting preferences *offline* and training a *reward model*,
then doing RL against the cheap proxy - but that reintroduces gaming (the policy
learns to fool the reward model). Online, per-interaction human reward will not
train a generation policy from scratch at any reasonable pace.

What a low-volume signal *can* do: tune a **small controller's narrow behavior**
over time - the scale of `harmonic_weight_rl` (a tiny REINFORCE controller
nudging a few parameters against a slow EMA). Scope it there, not to training a
generation policy from scratch. For development, use a **simulated user** (a proxy
that "answers" based on question difficulty) to get the sample density needed to
debug the loop before a real human is in it.

## Praxis fit and staging (smallest first)

The Gymnasium button row is the natural home: `Read` -> `Evaluate` -> **`Print`**
(emit one question, receive one engagement). Build in three layers, each useful
on its own:

1. **Data.** `Print` logs `(question, engaged?, answered_correctly?)`. Immediately
   useful as a dataset regardless of whether the RL ever runs - and the honest
   first step (no reward, no policy, just collection).
2. **Reward.** A homeostatic "energy" variable (long EMA + satiation/concave gain
   + slow decay) computed from correctness, feeding a small controller. The real
   engineering is the **async reward channel**: the UI engagement event has to
   reach the controller, which trains on a different (training-loop) cadence -
   buffer engagements and apply them as a slow, sparse reward, the way the
   harmonic-weight callback already integrates a delayed EMA return.
3. **Eval.** Score the question-posing behavior the way `BrierLM` scores
   generation, so improvement is measurable without trusting the live reward.

Ties to existing pieces: the reward EMA / integrated-return design mirrors
`harmonic_weight_rl`; the `Print` interaction and KB access compose with the
unified-KB note (the model could ground its questions in the KB); the
"person is the proof" framing in `research/main.tex` is the philosophical home
(the human in the loop *is* the signal).

## Open questions / kill criteria

- Does a low-volume correctness signal actually move a small controller, or is it
  too sparse even at that scale? (Test with the simulated user first.)
- Does the policy collapse to trivially-answerable questions to farm "correct"?
  If so, the correctness reward needs an informativeness / difficulty term
  (reward questions that are answerable *and* non-trivial).
- Does satiation actually prevent spam, or does the model find a periodic
  exploit around the decay constant?
- Hard line: if any variant starts optimizing for attention over usefulness
  (manipulative or anxiety-inducing questions), kill it - that is the failure the
  whole design exists to avoid.
