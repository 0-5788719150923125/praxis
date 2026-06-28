# Predictive Neural Memory: what the test-time memory actually learns

Status: grounded result + marked interpretation (2026-06-28). The predictive
(next-latent) write target stopped the universal Memory Gain collapse - gain now
hovers ~0.4-0.5 instead of decaying toward 0, i.e. the memory is contributing real
signal to the residual stream. This note records what we think it is learning,
keeping the mechanism (kernel) separate from the reading (voice). Pairs with the
roadmap's tiered-memory item, [temperament_bias_variance.md](temperament_bias_variance.md),
[observer_frequency.md](observer_frequency.md), [the_dial.md](the_dial.md).
Citation anchor: NextLat (belief-state Thm 3.2, https://arxiv.org/abs/2511.05963).

## Two timescales (the part that is mechanism)

The memory learns on two clocks, and keeping them separate is the whole answer:

- **Slow (training, backprop).** The task gradient shapes the addressing
  (`to_queries`), the readout (`combine`), the norms, and the learnable initial
  associative weights `W0`. Training does not learn the associations themselves -
  it learns *how to address and read out* a memory, and a good starting map for
  the fast rule to depart from. This is a learned fast-weight / test-time-training
  operator: training shapes the adapter, it does not do the adapting.
- **Fast (inference, detached).** Within one forward pass the memory MLP's weights
  are updated by a few steps of detached Adam on the surprise gradient, fitting
  *this sequence's* `key_t -> stream_{t+1}` map. This is the in-context learning;
  it runs at test time and resets per sequence (no trace carries across them).

So "the memory learns X" is two statements: training learns an adaptation
operator; inference runs it on the current context.

## What it represents: a belief state, in direction

The fast rule fits the *next* latent, not the current one (NextLat: predicting the
next state forces the stored state toward a sufficient statistic for the future),
and it fits it in RMS-normalized (directional) space - it forecasts *where the
stream rotates next*, not its magnitude. So the retrieved vector is a forecast of
the stream's next direction, and what it adds to the residual is the part of that
next state not already implied by the current one: the input-conditional
correction.

## The reading (voice, marked): deviations from the loud mean

In bias/variance terms this is clean and worth saying out loud: the harmonic field
is the loud shared mean (bias); the memory's forecast is a quiet, input-conditional
correction on top (variance). The ~0.4-0.5 gain says that correction is substantial
and *retained* rather than routed around - which is exactly the failure the old
auto-associative target produced (an echo of the mean is redundant, so the model
suppressed it). Predicting the *next* state makes the contribution non-redundant by
construction. In the dial geometry: the memory forecasts the next turn of the
field, and contributes the nudge the current phase does not already determine.

## On the "gradient dynamics through training" hypothesis

The intuition - the memory emulates "accelerated turning of the gears at test
time" - has a real referent, but one piece needs relocating to stay falsifiable:

- **Right:** the fast update is genuine extra computation the frozen backbone does
  not do - a few optimization steps per sequence. That *is* turning a gear the
  rest of the model cannot, in-context.
- **Relocate:** it is not collecting *training* gradient dynamics. The surprise is
  the prediction-error gradient of the memory MLP w.r.t. its *own* weights on the
  *current* sequence, computed fresh at inference. It has no access to the
  backbone's optimizer trajectory and no memory across training steps. What
  persists through training is the learned init + addressing/readout, not a record
  of gradients. The correct version: training meta-learns a fast-adaptation
  operator whose *test-time trajectory* emulates "more optimization" on this
  context - in-context descent, not stored training dynamics.

## What would confirm the reading (falsifiable)

- The triple, not gain alone: gain stays ~0.4-0.5 AND `memory_surprise_norm` (now
  the next-latent Huber error) falls AND `memory_write` stays healthy.
- Ablate the readout: zero `combine` at eval and measure the loss / BrierLM hit.
  Load-bearing forecast -> real hit; cosmetic gain -> none.
- The belief-state claim predicts the contribution should be *larger* where the
  next token is less determined by the current one (boundaries, topic shifts).
  Segment-event positions are the natural place to look.
